import torch
import torch.nn as nn

from onnx import shape_inference

from stream.api import optimize_allocation_ga

class Gemm_Operator(nn.Module) :
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(10, 20, (3, 3))
        self.conv2 = nn.Conv2d(20, 40, (3, 3))
        self.m = nn.Softmax(dim=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    def forward(self, x) :
        x = self.conv2(self.conv1(x))
        x = x + torch.ones(1, 40, 6, 6)
        x = self.avgpool(x)
        x= x.view(40)
        x = x + torch.ones(40)
        return x
    
class Test(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp1 = nn.Linear(10, 20)
        self.mlp2 = nn.Linear(20, 10)
        
    def forward(self, x):
        x = self.mlp1(x)
        x = self.mlp2(x)
        x = x + 1
        return x
    

if __name__ == "__main__":
    
    folder = "onnx/"
    onnx_path = f"{folder}/test1Derror.onnx"
    infered_path =  f"{folder}/inferred1Derror.onnx"

    soc_path = "stream/stream/inputs/examples/hardware/tpu_like_quad_core.yaml"
    mapping_path = "stream/stream/inputs/examples/mapping/tpu_like_quad_core.yaml"
    output_path = "output/result"
    mode = "fused"

    layer_stacks = [tuple(range(0, 11)), tuple(range(11, 22))] + list((i,) for i in range(22, 49))
    # Generate, Export and Infer Shapes of a simple MLP
    model = Gemm_Operator()
    torch_input = torch.randn(1, 10, 10, 10)
    torch.onnx.export(model, torch_input, onnx_path)
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