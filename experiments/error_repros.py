import argparse
from pathlib import Path

import torch
from onnx import shape_inference
from stream.api import optimize_allocation_ga


class GemmOperator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(10, 20, (3, 3))
        self.conv2 = torch.nn.Conv2d(20, 40, (3, 3))
        self.m = torch.nn.Softmax(dim=1)
        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.conv2(self.conv1(x))
        x = x + torch.ones(1, 40, 6, 6)
        x = self.avgpool(x)
        x = x.view(40)
        x = x + torch.ones(40)
        return x


class MatmulReproducer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(10, 20, (3, 3))
        self.linear2 = torch.nn.Linear(20, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.permute(x, (0, 2, 3, 1))
        x = torch.reshape(x, (x.shape[0], x.shape[1] * x.shape[2], x.shape[3]))
        print(x.shape, (x.shape[0], 10, x.shape[1]))
        x = torch.matmul(torch.ones((x.shape[0], 10, x.shape[1])), x)
        return x


class MatmulTransposeReproducer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(10, 20, (3, 3))
        self.linear2 = torch.nn.Linear(20, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.reshape(x, (x.shape[0], x.shape[1], x.shape[2] * x.shape[3]))

        ones = torch.ones((x.shape[0], x.shape[2], 10))
        print(x.shape, ones.shape)
        x = torch.matmul(x, ones)
        x = torch.permute(x, (0, 2, 1))
        return x


class RTreeReproducer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(10, 20, (3, 3))
        self.linear2 = torch.nn.Linear(20, 10)

    def forward(self, x):
        x = torch.matmul(torch.ones(32, 4), x)
        x = torch.softmax(x, 1)
        x = torch.add(torch.ones_like(x), x)
        return x


class ShapeReproducer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(512, 512, (3, 3))
        self.linear2 = torch.nn.Linear(20, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.relu(x)
        x = x.view(x.shape[0], x.shape[1], -1)
        x = torch.matmul(x, torch.ones((x.shape[0], x.shape[2], 4068)))
        return x


def parse_args():
    parser = argparse.ArgumentParser(description="Error reproduction cases")
    parser.add_argument(
        "--case",
        choices=["1d", "matmul", "matmul_transpose", "rtree", "shape"],
        default="1d",
        help="Which repro case to run.",
    )
    return parser.parse_args()


def run_case(folder, model, torch_input):
    onnx_path = f"{folder}/test.onnx"
    infered_path = f"{folder}/inferred.onnx"

    soc_path = "stream/stream/inputs/examples/hardware/tpu_like_quad_core.yaml"
    mapping_path = "stream/stream/inputs/examples/mapping/tpu_like_quad_core.yaml"
    output_path = "output/result"
    mode = "fused"

    layer_stacks = [tuple(range(0, 11)), tuple(range(11, 22))] + list((i,) for i in range(22, 49))

    Path(folder).mkdir(parents=True, exist_ok=True)
    torch.onnx.export(model, torch_input, onnx_path, opset_version=13)
    shape_inference.infer_shapes_path(onnx_path, infered_path)

    optimize_allocation_ga(
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


def main():
    args = parse_args()
    if args.case == "1d":
        run_case("onnx/error_1d", GemmOperator(), torch.randn(1, 10, 10, 10))
    elif args.case == "matmul":
        run_case("onnx/error_matmul", MatmulReproducer(), torch.randn(4, 10, 32, 32))
    elif args.case == "matmul_transpose":
        run_case("onnx/error_matmul", MatmulTransposeReproducer(), torch.randn(4, 10, 32, 32))
    elif args.case == "rtree":
        run_case("onnx/error_rtree", RTreeReproducer(), torch.randn(4, 32))
    elif args.case == "shape":
        run_case("onnx/error_shape", ShapeReproducer(), torch.randn(4, 512, 4, 4))


if __name__ == "__main__":
    main()
