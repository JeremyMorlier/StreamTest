import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from onnxruntime.training import artifacts

import onnx
from onnx import shape_inference
from tools import apply_onnx_passes, run_stream_co
from model.resnet18 import ResNet18


class MiniCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(64, 128, 3, bias=False)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.conv2(out)
        return out


class TestModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3)
        self.conv2 = nn.Conv2d(64, 64, 3)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv4 = nn.Conv2d(128, 64, 3)
        self.conv5 = nn.Conv2d(64, 128, 3)
        self.conv6 = nn.Conv2d(64, 128, 3)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out1 = F.relu(self.conv2(out))
        out2 = F.relu(self.conv3(out))
        out2 = F.relu(self.conv4(out2))
        out5 = F.relu(self.conv5(out2 + out1))
        out1 = F.relu(self.conv6(out1))
        return out1, out5


def test_forward_model():
    folder = "results/resnet18_t/"
    onnx_path = f"{folder}/test.onnx"
    infered_path = f"{folder}/inferred.onnx"
    soc_path = "stream/stream/inputs/examples/hardware/tpu_like_quad_core.yaml"
    mapping_path = "stream/stream/inputs/examples/mapping/tpu_lke_quad_core_fused_co.yaml"

    model = TestModel()
    torch_input = torch.randn(4, 3, 32, 32)
    torch.onnx.export(model, torch_input, onnx_path, opset_version=13)
    inferred_model = shape_inference.infer_shapes_path(onnx_path, infered_path)

    layer_stacks = None
    layer_stacks = [(0, 1), (4, 5), (6, 7), (2, 3), (8, 9, 10), (11, 12)]
    run_stream_co(
        infered_path,
        soc_path,
        mapping_path,
        id=41,
        output_path=folder,
        mode="fused",
        layer_stacks=layer_stacks,
    )


if __name__ == "__main__":
    folder = "results/resnet18_t/"
    onnx_path = f"{folder}/test.onnx"
    infered_path = f"{folder}/inferred.onnx"
    soc_path = "stream/stream/inputs/examples/hardware/tpu_like_quad_core.yaml"
    mapping_path = "stream/stream/inputs/examples/mapping/tpu_lke_quad_core_fused_co.yaml"

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
        if "conv1" in init.name:
            requires_grad.append(init.name)
        # requires_grad.append(init.name)
    loss = artifacts.LossType(2)
    print(requires_grad)

    inferred_train_onnx_path4, forward_path, _, _ = apply_onnx_passes(
        base_model, None, folder, requires_grad, "onnx", check=False
    )

    layer_stacks = None
    # layer_stacks = [
    #     (0, 1),
    #     (2,),
    #     (4,),
    #     (11, 12, 34),
    #     (5,),
    #     (35, 36, 37),
    #     (39, 40, 41, 42, 43),
    #     (14, 16, 17),
    #     (38, 44, 45, 46, 47),
    #     (19,),
    #     (48,),
    #     (20, 21, 22),
    #     (24, 25, 26, 27, 28),
    #     (23, 29, 30, 31, 32),
    #     (33,),
    # ]
    run_stream_co(
        inferred_train_onnx_path4,
        soc_path,
        mapping_path,
        id=43,
        output_path=folder,
        mode="fused",
        layer_stacks=layer_stacks,
    )
