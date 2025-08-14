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
        x = torch.matmul(torch.ones(32, 4), x)
        x = torch.softmax(x, 1)
        x = torch.add(torch.ones_like(x), x)
        # x = torch.matmul(torch.ones((x.shape[0], 10, x.shape[1])), x)
        return x


if __name__ == "__main__":
    folder = "onnx/error_rtree"
    onnx_path = f"{folder}/test.onnx"
    infered_path = f"{folder}/inferred.onnx"

    soc_path = "stream/stream/inputs/examples/hardware/tpu_like_quad_core.yaml"
    mapping_path = "stream/stream/inputs/examples/mapping/tpu_like_quad_core.yaml"
    output_path = "output/result"
    mode = "fused"

    layer_stacks = [tuple(range(0, 11)), tuple(range(11, 22))] + list((i,) for i in range(22, 49))

    model = Reproducer()
    torch_input = torch.randn(4, 32)
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
