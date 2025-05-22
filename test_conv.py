import torch
import torch.nn as nn

import onnx
from onnx import shape_inference
from onnxruntime.training import artifacts

from stream.api import optimize_allocation_ga

from process_onnx import process_convGrad, process_convGrad2
import logging
logging.basicConfig(level=logging.INFO)


class Gemm_Operator(nn.Module) :
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(10, 20, (3, 3))
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

    soc_path = "stream/stream/inputs/examples/hardware/tpu_like_quad_core.yaml"
    mapping_path = "stream/stream/inputs/examples/mapping/tpu_like_quad_core.yaml"
    output_path = "output/result"
    mode = "fused"

    layer_stacks = [tuple(range(0, 11)), tuple(range(11, 22))] + list((i,) for i in range(22, 49))
    # Generate, Export and Infer Shapes of a simple MLP
    model = Gemm_Operator()
    torch_input = torch.randn(4, 10, 30, 30)
    torch.onnx.export(model, torch_input, onnx_path)
    inferred_model = shape_inference.infer_shapes_path(onnx_path, infered_path)


    # Generate Backward
    base_model = onnx.load(infered_path)
    inits = base_model.graph.initializer
    requires_grad = []
    for init in inits :
        # if "bias" not in init.name :
        #     requires_grad.append(init.name)
        requires_grad.append(init.name)
    loss = artifacts.LossType(2)
    # Now, we can invoke generate_artifacts with this custom loss function
    artifacts.generate_artifacts(base_model, requires_grad=requires_grad,
                                loss = loss, optimizer = artifacts.OptimType.AdamW, prefix=folder)

    # Infer training graph
    inferred_model = shape_inference.infer_shapes_path(train_onnx_path, inferred_train_onnx_path)
    # print(onnx.load(train_onnx_path))
    # model = process_convGrad(onnx.load(train_onnx_path), "test")
    # # print(model)
    onnx.save(process_convGrad2(onnx.load(inferred_train_onnx_path)), inferred_train_onnx_path)
    inferred_model = shape_inference.infer_shapes_path(inferred_train_onnx_path, inferred_train_onnx_path)
    inferred_model = shape_inference.infer_shapes_path(inferred_train_onnx_path, inferred_train_onnx_path)
    print(onnx.checker.check_model(inferred_train_onnx_path))
    # onnx.save(manual_shape_inference(onnx.load(inferred_train_onnx_path)), inferred_train_onnx_path)
    # inferred_model = shape_inference.infer_shapes_path(inferred_train_onnx_path, inferred_train_onnx_path)
    # onnx.save(manual_shape_inference_transpose(onnx.load(inferred_train_onnx_path)), inferred_train_onnx_path)
    # inferred_model = shape_inference.infer_shapes_path(inferred_train_onnx_path, inferred_train_onnx_path)
    # onnx.save(manual_shape_inference2(onnx.load(inferred_train_onnx_path)), inferred_train_onnx_path)
    # inferred_model = shape_inference.infer_shapes_path(inferred_train_onnx_path, inferred_train_onnx_path)
    # inferred_train_onnx_path = "test2.onnx"
    # Evaluate Using Stream
    scme = optimize_allocation_ga(
        hardware=soc_path,
        workload=inferred_train_onnx_path,
        mapping=mapping_path,
        mode=mode,
        layer_stacks=layer_stacks,
        nb_ga_generations=4,
        nb_ga_individuals=4,
        experiment_id=id,
        output_path=output_path,
        skip_if_exists=False,
    )