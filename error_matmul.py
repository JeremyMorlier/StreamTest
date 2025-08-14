import torch
from torch import nn

from onnx import shape_inference
from stream.api import optimize_allocation_ga


class Reproducer(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(10, 20, (3, 3))
        self.linear2 = nn.Linear(20, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.permute(x, (0, 2, 3, 1))
        x = torch.reshape(x, (x.shape[0], x.shape[1] * x.shape[2], x.shape[3]))
        print(x.shape, (x.shape[0], 10, x.shape[1]))
        x = torch.matmul(torch.ones((x.shape[0], 10, x.shape[1])), x)
        return x


class ReproducerTranpose(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(10, 20, (3, 3))
        self.linear2 = nn.Linear(20, 10)

    def forward(self, x):
        x = self.conv1(x)
        # x = torch.permute(x, (0, 2, 3, 1))
        x = torch.reshape(x, (x.shape[0], x.shape[1], x.shape[2] * x.shape[3]))

        ones = torch.ones((x.shape[0], x.shape[2], 10))
        print(x.shape, ones.shape)
        x = torch.matmul(x, ones)
        x = torch.permute(x, (0, 2, 1))
        return x


if __name__ == "__main__":
    folder = "onnx/error_matmul"
    onnx_path = f"{folder}/test.onnx"
    infered_path = f"{folder}/inferred.onnx"
    train_onnx_path = f"{folder}/training_model.onnx"
    inferred_train_onnx_path = f"{folder}/infered_training_model.onnx"
    inferred_train_onnx_path2 = f"{folder}/infered_training_model2.onnx"
    soc_path = "stream/stream/inputs/examples/hardware/tpu_like_quad_core.yaml"
    mapping_path = "stream/stream/inputs/examples/mapping/tpu_like_quad_core.yaml"
    output_path = "output/result"
    mode = "fused"

    layer_stacks = [tuple(range(0, 11)), tuple(range(11, 22))] + list((i,) for i in range(22, 49))

    model = ReproducerTranpose()
    torch_input = torch.randn(4, 10, 32, 32)
    torch.onnx.export(model, torch_input, onnx_path, opset_version=13)
    inferred_model = shape_inference.infer_shapes_path(onnx_path, infered_path)

    scme = optimize_allocation_ga(
        hardware=soc_path,
        workload=infered_path,
        mapping=mapping_path,
        mode=mode,
        layer_stacks=layer_stacks,
        nb_ga_generations=4,
        nb_ga_individuals=4,
        experiment_id=id,
        output_path=output_path,
        skip_if_exists=False,
    )
