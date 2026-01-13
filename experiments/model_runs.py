import argparse
import logging

import onnx
import torch
from onnx import shape_inference
from onnxruntime.training import artifacts

from model.resnet18 import ResNet18
from model.resnet_lora import ResNet18_LoRa
from streamtest.onnx_processing import apply_onnx_passes
from streamtest.stream_runner import run_stream

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser(description="Model experiments")
    parser.add_argument(
        "--case",
        choices=["resnet", "lora", "mini_llm"],
        default="resnet",
        help="Which model pipeline to run.",
    )
    return parser.parse_args()


def run_resnet():
    folder = "results/resnet18_forward/"
    onnx_path = f"{folder}/test.onnx"
    infered_path = f"{folder}/inferred.onnx"
    soc_path = "stream/stream/inputs/examples/hardware/tpu_like_quad_core.yaml"
    mapping_path = "stream/stream/inputs/examples/mapping/tpu_like_quad_core_ga.yaml"

    # Generate, Export and Infer Shapes of a ResNet18 Model
    model = ResNet18()
    torch_input = torch.randn(4, 3, 32, 32)
    torch.onnx.export(model, torch_input, onnx_path, opset_version=13)
    shape_inference.infer_shapes_path(onnx_path, infered_path)

    # Generate Backward
    base_model = onnx.load(infered_path)
    inits = base_model.graph.initializer
    requires_grad = []
    for init in inits:
        # if len(init.dims) != 1 :
        requires_grad.append(init.name)
    loss = artifacts.LossType(2)

    inferred_train_onnx_path4, _, _, _ = apply_onnx_passes(base_model, None, folder, requires_grad, "onnx", check=False)
    run_stream(infered_path, soc_path, mapping_path, id=2, output_path=folder, mode="fused")


def run_lora():
    folder = "results/lora_resnet18/"
    onnx_path = f"{folder}/test.onnx"
    infered_path = f"{folder}/inferred.onnx"
    soc_path = "stream/stream/inputs/examples/hardware/tpu_like_quad_core.yaml"
    mapping_path = "stream/stream/inputs/examples/mapping/tpu_like_quad_core_ga.yaml"
    output_path = "results/not_lora_resnet18/"

    model = ResNet18_LoRa(lora=True)
    model.train()
    torch_input = torch.randn(4, 3, 32, 32)
    model(torch_input)
    torch.onnx.export(model, torch_input, onnx_path, opset_version=13)
    shape_inference.infer_shapes_path(onnx_path, infered_path)

    # Generate Backward
    base_model = onnx.load(infered_path)
    inits = base_model.graph.initializer
    requires_grad = []
    for init in inits:
        if "delta" in init.name:
            requires_grad.append(init.name)
    loss = artifacts.LossType(2)

    inferred_train_onnx_path4, _, _, _ = apply_onnx_passes(
        base_model, None, output_path, requires_grad, "onnx", check=False
    )
    run_stream(inferred_train_onnx_path4, soc_path, mapping_path, id=3, output_path=output_path, mode="fused")


def run_mini_llm():
    from model.mini_llm import MiniTransformerLM

    # Example usage of similar to LLama2
    divisor_factor = 8
    vocab_size = int(32000 / divisor_factor)
    max_seq_len = int(2048 / divisor_factor)
    d_model = int(4096 / divisor_factor)
    nhead = int(32 / divisor_factor)
    num_layers = int(32 / divisor_factor)
    dim_feedforward = int(11008 / divisor_factor)

    model = MiniTransformerLM(
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
    )

    # Dummy input (batch_size=1, seq_len=10)
    dummy_input = torch.randint(0, vocab_size, (1, max_seq_len))

    output_path = "results/minillm/"
    soc_path = "stream/stream/inputs/examples/hardware/tpu_like_quad_core.yaml"
    mapping_path = "stream/stream/inputs/examples/mapping/tpu_like_quad_core_ga.yaml"

    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        "mini_transformer_lm.onnx",
        input_names=["input"],
        output_names=["output"],
        opset_version=16,
    )

    base_model = onnx.load("mini_transformer_lm.onnx", load_external_data=False)
    print(len(base_model.graph.output))

    inits = base_model.graph.initializer
    requires_grad = []
    # skip the embedding layer
    for init in inits[1:]:
        # if len(init.dims) != 1 :
        requires_grad.append(init.name)

    inferred_train_onnx_path4, _, _, _ = apply_onnx_passes(
        base_model, None, output_path, requires_grad, "onnx", check=False
    )
    run_stream(inferred_train_onnx_path4, soc_path, mapping_path, id=3, output_path=output_path, mode="fused")


def main():
    args = parse_args()
    if args.case == "resnet":
        run_resnet()
    elif args.case == "lora":
        run_lora()
    else:
        run_mini_llm()


if __name__ == "__main__":
    main()
