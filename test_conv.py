import logging

import torch
from onnxruntime.training import artifacts
from onnxsim import simplify

import onnx
from model.resnet18 import ResNet18
from onnx import shape_inference

# from stream.visualization.memory_usage import plot_memory_usage
# from stream.visualization.perfetto import convert_scme_to_perfetto_json
# from stream.visualization.schedule import visualize_timeline_plotly
from process_onnx import add_optimizer, process_1D_nodes, process_convGrad, process_PoolGrad, split_forward_backward
from stream.api import optimize_allocation_ga
from stream.utils import CostModelEvaluationLUT

# Set the logging level to ERROR to suppress warnings
# ort.set_default_logger_severity(4)
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def run_stream(model_path, accelerator_path, mapping_path, id, output_path):
    mode = "lbl"
    layer_stacks = [tuple(range(0, 11)), tuple(range(11, 22))] + list((i,) for i in range(22, 49))

    # Evaluate Using Stream
    # try :
    scme = optimize_allocation_ga(
        hardware=accelerator_path,
        workload=model_path,
        mapping=mapping_path,
        mode=mode,
        layer_stacks=layer_stacks,
        nb_ga_generations=4,
        nb_ga_individuals=4,
        experiment_id=id,
        output_path=output_path,
        skip_if_exists=False,
    )
    # except Exception as e:
    #     logging.error(f"Error during optimization: {e}")

    # Load in the CostModelEvaluationLUT from the run
    cost_lut_path = f"{output_path}/{id}/cost_lut.pickle"
    cost_lut = CostModelEvaluationLUT(cost_lut_path)

    with open(f"{output_path}/resultt.txt", "a") as f:
        f.write(f"{scme.energy}    {scme.latency} \n")
    # # Plotting schedule timeline of best SCME
    # visualize_timeline_plotly(
    #     scme,
    #     draw_dependencies=True,
    #     draw_communication=True,
    #     fig_path=f"{output_path}/{id}/schedule.html",
    #     cost_lut=cost_lut,
    # )
    # # Plotting memory usage of best SCME
    # plot_memory_usage(scme, (0,), (100,), fig_path=f"{output_path}/{id}/memory.png")

    # # Save json for perfetto visualization (Visualize at http://ui.perfetto.dev/)
    # convert_scme_to_perfetto_json(scme, cost_lut, json_path=f"{output_path}/{id}/scme.json")


if __name__ == "__main__":
    folder = "onnx/"
    onnx_path = f"{folder}/test.onnx"
    infered_path = f"{folder}/inferred.onnx"
    train_onnx_path = f"{folder}/training_model.onnx"
    inferred_train_onnx_path = f"{folder}/infered_training_model.onnx"
    inferred_train_onnx_path2 = f"{folder}/infered_training_model2.onnx"
    inferred_train_onnx_path3 = f"{folder}/infered_training_model3.onnx"
    inferred_train_onnx_path4 = f"{folder}/infered_training_model4.onnx"
    soc_path = "stream/stream/inputs/examples/hardware/tpu_like_quad_core.yaml"
    mapping_path = "stream/stream/inputs/examples/mapping/tpu_like_quad_core_ga.yaml"
    output_path = "output/result"

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
        requires_grad.append(init.name)
    loss = artifacts.LossType(2)
    # Now, we can invoke generate_artifacts with this custom loss function
    artifacts.generate_artifacts(
        base_model, requires_grad=requires_grad, loss=loss, optimizer=artifacts.OptimType.AdamW, prefix=folder
    )

    # Infer training graph
    inferred_model = shape_inference.infer_shapes_path(train_onnx_path, inferred_train_onnx_path)
    inferred_model = shape_inference.infer_shapes_path(train_onnx_path, inferred_train_onnx_path)
    inferred_model = shape_inference.infer_shapes_path(train_onnx_path, inferred_train_onnx_path)

    onnx.save(process_convGrad(process_PoolGrad(onnx.load(inferred_train_onnx_path))), inferred_train_onnx_path2)
    inferred_model = shape_inference.infer_shapes_path(inferred_train_onnx_path2, inferred_train_onnx_path2)
    inferred_model = shape_inference.infer_shapes_path(inferred_train_onnx_path2, inferred_train_onnx_path2)
    model_simplified, check = simplify(
        onnx.load(inferred_train_onnx_path2), skipped_optimizers=["extract_constant_to_initializer"]
    )
    onnx.save(process_1D_nodes(model_simplified), inferred_train_onnx_path3)
    inferred_model = shape_inference.infer_shapes_path(inferred_train_onnx_path3, inferred_train_onnx_path3)
    # inferred_train_onnx_path3 = inferred_train_onnx_path2
    # Add Optimizer
    optimizer_model, optimizer_inputs, optimizer_outputs = add_optimizer(onnx.load(inferred_train_onnx_path3))
    onnx.save(optimizer_model, inferred_train_onnx_path4)

    inferred_model = shape_inference.infer_shapes_path(inferred_train_onnx_path4, inferred_train_onnx_path4)
    print(onnx.checker.check_model(inferred_train_onnx_path4))
    # run_stream("stream/stream/inputs/examples/workload/resnet18.onnx", soc_path, mapping_path, "forward", output_path)
    run_stream(inferred_train_onnx_path4, soc_path, mapping_path, "forwardbackward", output_path)

    # Split Forward, Backward and Optimizer
    onnx_model = onnx.load(inferred_train_onnx_path3)
    forward_inputs, backward_inputs, forward_outputs, backward_outputs = split_forward_backward(onnx_model)
    onnx.utils.extract_model(inferred_train_onnx_path3, f"{folder}/forward.onnx", forward_inputs, forward_outputs, True)
    onnx.utils.extract_model(
        inferred_train_onnx_path3, f"{folder}/backward.onnx", backward_inputs, backward_outputs, True
    )
    onnx.utils.extract_model(
        inferred_train_onnx_path4, f"{folder}/optimizer.onnx", optimizer_inputs, optimizer_outputs, True
    )

    # onnx.utils.extract_model(f"{folder}/backward.onnx", f"{folder}/backward2.onnx", ["/layer4/layer4.1/Add_output_0", "/layer4/layer4.1/Relu_1_output_0_grad", "transpose1_output1_/layer4/layer4.1/conv2/Conv_Grad/ConvGrad_0"], ["matmul_output_/layer4/layer4.1/conv2/Conv_Grad/ConvGrad_0"], True)

    # Evaluate Using Stream

    # run_stream(f"{folder}/forward.onnx", soc_path, mapping_path, "forward", output_path)
    # run_stream(f"{folder}/backward.onnx", soc_path, mapping_path, "backward", output_path)
    run_stream(f"{folder}/optimizer.onnx", soc_path, mapping_path, "optimizer", output_path)
