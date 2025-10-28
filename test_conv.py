import logging

import torch
from onnxruntime.training import artifacts

import onnx
from model.resnet18 import ResNet18
from onnx import shape_inference
from tools import apply_onnx_passes, run_stream, run_stream_co

# Set the logging level to ERROR to suppress warnings
import onnxruntime as ort
ort.set_default_logger_severity(4)
# logger = logging.getLogger(__name__)
# logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    folder = "results/resnet18_t/"
    onnx_path = f"{folder}/test.onnx"
    infered_path = f"{folder}/inferred.onnx"
    soc_path = "stream/stream/inputs/examples/hardware/tpu_like_quad_core.yaml"
    mapping_path = "stream/stream/inputs/examples/mapping/tpu_like_quad_core_fusec_ga_elementwise.yaml"

    # Generate, Export and Infer Shapes of a ResNet18 Model

    model = ResNet18()
    torch_input = torch.randn(4, 3, 32, 32)
    torch.onnx.export(model, torch_input, onnx_path, opset_version=13)
    inferred_model = shape_inference.infer_shapes_path(onnx_path, infered_path)

    # Generate Backward
    base_model = onnx.load(infered_path)
    inits = base_model.graph.initializer
    requires_grad = []
    for init in inits:
        # if len(init.dims) != 1 :
        print(init.name)
        # if "conv1" in init.name:
        #     requires_grad.append(init.name)
        requires_grad.append(init.name)
    loss = artifacts.LossType(2)
    print(requires_grad)

    # layer_stacks = [tuple(range(0, 11)), tuple(range(11, 22))] + list((i,) for i in range(22, 49))
    inferred_train_onnx_path4, forward_path, _, _ = apply_onnx_passes(
        base_model, None, folder, requires_grad, "onnx", check=False
    )
    layer_stacks = None
    run_stream(
        inferred_train_onnx_path4,
        soc_path,
        mapping_path,
        id=9,
        output_path=folder,
        mode="fused",
        layer_stacks=layer_stacks,
    )
