from multiprocessing import Pool
from pathlib import Path

# Set the logging level to ERROR to suppress warnings
import torch

import onnx
from model.resnet18 import ResNet18
from onnx import shape_inference
from tools import apply_onnx_passes, run_stream


def evaluate(batch_size, soc_path, mapping_path, folder, optimizer=True):
    Path(folder).mkdir(parents=True, exist_ok=True)
    onnx_path = f"{folder}/test.onnx"
    infered_path = f"{folder}/inferred.onnx"
    model = ResNet18()
    torch_input = torch.randn(batch_size, 3, 32, 32)
    torch.onnx.export(model, torch_input, onnx_path, opset_version=13)
    shape_inference.infer_shapes_path(onnx_path, infered_path)

    # Generate Backward
    base_model = onnx.load(infered_path)
    inits = base_model.graph.initializer
    requires_grad = []
    for init in inits:
        requires_grad.append(init.name)
    print(requires_grad)

    inferred_train_onnx_path4, forward_path, _, _ = apply_onnx_passes(
        base_model, None, folder, requires_grad, "onnx", check=False, optimizer=optimizer
    )
    layer_stacks = None

    try:
        energy, latency, memory = run_stream(
            inferred_train_onnx_path4,
            soc_path,
            mapping_path,
            id=batch_size,
            output_path=folder,
            mode="fused",
            layer_stacks=layer_stacks,
        )
    except Exception as e:
        print(e)
        energy, latency, memory = 0, 0, 0

    print(f"{folder}, {batch_size}, Optimizer:{optimizer}, {energy}, {latency}, {memory}")
    return batch_size, energy, latency, memory


if __name__ == "__main__":
    folder = "results/resnet18_t/"
    soc_path = "stream/stream/inputs/examples/hardware/tpu_like_quad_core.yaml"
    mapping_path = "stream/stream/inputs/examples/mapping/tpu_like_quad_core_fused_ga_elementwise2.yaml"
    mapping_path_batch = "stream/stream/inputs/examples/mapping/tpu_like_quad_core_fused_ga_elementwise2_batch.yaml"

    batch_sizes = [1, 2, 4, 8, 16, 32]
    batch_sizes_split = [4, 8, 16, 32]
    args1 = [(batch_size, soc_path, mapping_path, f"{folder}/{batch_size}/") for batch_size in batch_sizes]
    args2 = [
        (batch_size, soc_path, mapping_path_batch, f"{folder}/{batch_size}_SplitBatch/")
        for batch_size in batch_sizes_split
    ]
    args3 = [(batch_size, soc_path, mapping_path, f"{folder}/{batch_size}_noOpt/", False) for batch_size in batch_sizes]

    args = args1 + args2 + args3
    print(args)
    # Use Pool to parallelize the evaluations
    with Pool(processes=len(batch_sizes)) as pool:
        r = pool.starmap(evaluate, args)
    print("result :", r)
