import torch
import torch.nn as nn

import onnx
from onnx import shape_inference
from onnxruntime.training import artifacts
from onnxsim import simplify
from stream.api import optimize_allocation_ga

from process_onnx import process_convGrad, process_PoolGrad, process_1D_nodes, split_forward_backward
import logging
logging.basicConfig(level=logging.INFO)

import onnxruntime as ort

# Set the logging level to ERROR to suppress warnings
ort.set_default_logger_severity(3)

from resnet18 import ResNet18
class Gemm_Operator(nn.Module) :
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 20, (3, 3))
        self.conv2 = nn.Conv2d(20, 40, (3, 3))
        self.m = nn.Softmax(dim=1)

    def forward(self, x) :
        return self.conv2(self.conv1(x))
    
if __name__ == "__main__" :
    folder = "onnx/"
    onnx_path = f"{folder}/test.onnx"
    infered_path =  f"{folder}/inferred.onnx"
    train_onnx_path = f"{folder}/training_model.onnx"
    inferred_train_onnx_path = f"{folder}/infered_training_model.onnx"
    inferred_train_onnx_path2 = f"{folder}/infered_training_model2.onnx"
    inferred_train_onnx_path3 = f"{folder}/infered_training_model3.onnx"
    soc_path = "stream/stream/inputs/examples/hardware/tpu_like_quad_core.yaml"
    mapping_path = "stream/stream/inputs/examples/mapping/tpu_like_quad_core.yaml"
    output_path = "output/result"
    mode = "fused"

    layer_stacks = [tuple(range(0, 11)), tuple(range(11, 22))] + list((i,) for i in range(22, 49))
    # Generate, Export and Infer Shapes of a simple MLP
    # model = Gemm_Operator()
    # model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
    model = ResNet18()
    torch_input = torch.randn(4, 3, 32, 32)
    torch.onnx.export(model, torch_input, onnx_path, opset_version=13)
    inferred_model = shape_inference.infer_shapes_path(onnx_path, infered_path)

    # Generate Backward
    base_model = onnx.load(infered_path)
    inits = base_model.graph.initializer
    requires_grad = []
    for init in inits :
        # if len(init.dims) != 1 :
        requires_grad.append(init.name)
    loss = artifacts.LossType(2)
    # Now, we can invoke generate_artifacts with this custom loss function
    artifacts.generate_artifacts(base_model, requires_grad=requires_grad,
                                loss = loss, optimizer = artifacts.OptimType.AdamW, prefix=folder)

    # Infer training graph
    inferred_model = shape_inference.infer_shapes_path(train_onnx_path, inferred_train_onnx_path)
    inferred_model = shape_inference.infer_shapes_path(train_onnx_path, inferred_train_onnx_path)
    inferred_model = shape_inference.infer_shapes_path(train_onnx_path, inferred_train_onnx_path)

    onnx.save(process_convGrad(process_PoolGrad(onnx.load(inferred_train_onnx_path))), inferred_train_onnx_path2)
    inferred_model = shape_inference.infer_shapes_path(inferred_train_onnx_path2, inferred_train_onnx_path2)
    inferred_model = shape_inference.infer_shapes_path(inferred_train_onnx_path2, inferred_train_onnx_path2)
    model_simplified, check = simplify(onnx.load(inferred_train_onnx_path2), skipped_optimizers=["extract_constant_to_initializer"])
    onnx.save(process_1D_nodes(model_simplified), inferred_train_onnx_path3)
    inferred_model = shape_inference.infer_shapes_path(inferred_train_onnx_path3, inferred_train_onnx_path3)
    print(onnx.checker.check_model(inferred_train_onnx_path3))


    # Evaluate Using Stream
    # scme = optimize_allocation_ga(
    #     hardware=soc_path,
    #     workload=inferred_train_onnx_path3,
    #     mapping=mapping_path,
    #     mode=mode,
    #     layer_stacks=layer_stacks,
    #     nb_ga_generations=4,
    #     nb_ga_individuals=4,
    #     experiment_id=id,
    #     output_path=output_path,
    #     skip_if_exists=False,
    # )
    # Split Forward and Backward
    onnx_model = onnx.load(inferred_train_onnx_path3)
    forward_inputs, backward_inputs, forward_outputs, backward_outputs = split_forward_backward(onnx_model)
    onnx.utils.extract_model(inferred_train_onnx_path3, f"{folder}/forward.onnx", forward_inputs, forward_outputs, True)
    onnx.utils.extract_model(inferred_train_onnx_path3, f"{folder}/backward.onnx", backward_inputs, backward_outputs, True)

    onnx.utils.extract_model(f"{folder}/backward.onnx", f"{folder}/backward2.onnx", ["/layer4/layer4.1/Add_output_0", "/layer4/layer4.1/Relu_1_output_0_grad", "transpose1_output1_/layer4/layer4.1/conv2/Conv_Grad/ConvGrad_0"], ["matmul_output_/layer4/layer4.1/conv2/Conv_Grad/ConvGrad_0"], True)
    # Evaluate Using Stream
    # scme = optimize_allocation_ga(
    #     hardware=soc_path,
    #     workload=f"{folder}/forward.onnx",
    #     mapping=mapping_path,
    #     mode=mode,
    #     layer_stacks=layer_stacks,
    #     nb_ga_generations=4,
    #     nb_ga_individuals=4,
    #     experiment_id=id,
    #     output_path=output_path,
    #     skip_if_exists=False,
    # )
    try :
        scme = optimize_allocation_ga(
            hardware=soc_path,
            workload=f"{folder}/backward2.onnx",
            mapping=mapping_path,
            mode=mode,
            layer_stacks=layer_stacks,
            nb_ga_generations=4,
            nb_ga_individuals=4,
            experiment_id=id,
            output_path=output_path,
            skip_if_exists=False,
        )
    except Exception as e:
        print(f"Error during optimization: {e}")
        raise e