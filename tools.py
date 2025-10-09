from pathlib import Path

import onnx
import torch
from onnx import shape_inference
from onnxruntime.training import artifacts
from onnxsim import simplify
from stream.api import optimize_allocation_ga
from stream.utils import CostModelEvaluationLUT
from stream.visualization.perfetto import convert_scme_to_perfetto_json

from process_onnx import (
    add_optimizer,
    expand_softmax_grad_node,
    process_1d_nodes,
    process_batch_norm,
    process_concat_nodes,
    process_convolution_grad,
    process_poolgrad,
    split_forward_backward,
)


def run_stream(model_path, accelerator_path, mapping_path, id, output_path, mode="fused"):
    Path(output_path, str(id)).mkdir(parents=True, exist_ok=True)
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
    print(scme.latency, type(scme.latency))
    with open(f"{output_path}/resultt.txt", "a") as f:
        f.write(f"{scme.energy}    {scme.latency} \n")
    # Plotting schedule timeline of best SCME
    # scme.plot_schedule(
    #     plot_full_schedule=True,
    #     draw_dependencies=True,
    #     plot_data_transfer=True,
    #     fig_path=f"{output_path}/{id}/schedule.html",
    # )

    # Plotting memory usage of best SCME
    scme.plot_memory_usage((0,), (100,), fig_path=f"{output_path}/{id}/memory.png")

    # Save json for perfetto visualization (Visualize at http://ui.perfetto.dev/)
    convert_scme_to_perfetto_json(scme, cost_lut, json_path=f"{output_path}/{id}/scme.json")
    return scme.latency, scme.energy


def apply_onnx_passes(torch_model, example_input=None, output_path="./", requires_grad=None, mode="torch", check=True):
    # Output Paths to store intermediary models
    Path(output_path).mkdir(parents=True, exist_ok=True)

    onnx_path = f"{output_path}/model.onnx"
    train_onnx_path = f"{output_path}/training_model.onnx"

    inferred_train_onnx_path1 = f"{output_path}/model1.onnx"
    inferred_train_onnx_path2 = f"{output_path}/model2.onnx"
    inferred_train_onnx_path3 = f"{output_path}/model3.onnx"
    inferred_train_onnx_path4 = f"{output_path}/model4.onnx"
    inferred_train_onnx_path5 = f"{output_path}/model5.onnx"

    # submodels paths
    forward_onnx_path = f"{output_path}/forward.onnx"
    backward_onnx_path = f"{output_path}/backward.onnx"
    optimizer_onnx_path = f"{output_path}/optimizer.onnx"

    # Export Torch Model to ONNX
    if "torch" in mode:
        onnx_model = torch.onnx.export(torch_model, example_input, onnx_path, opset_version=13, export_params=False)
    else:
        onnx_model = torch_model
    # Retrieve ONNX training graph with onnxruntime
    loss = artifacts.LossType(2)
    artifacts.generate_artifacts(
        onnx_model, requires_grad=requires_grad, loss=loss, optimizer=artifacts.OptimType.AdamW, prefix=output_path
    )

    # Multiple shapes inference pass are needed
    inferred_model = shape_inference.infer_shapes(onnx.load(train_onnx_path))
    inferred_model = shape_inference.infer_shapes(inferred_model)
    inferred_model = shape_inference.infer_shapes(inferred_model)

    processed_model1 = process_poolgrad(inferred_model)
    print(onnx.checker.check_model(processed_model1))
    processed_model1 = process_convolution_grad(processed_model1)
    print(onnx.checker.check_model(processed_model1))
    processed_model1 = expand_softmax_grad_node(processed_model1)
    print(onnx.checker.check_model(processed_model1))
    onnx.save(processed_model1, inferred_train_onnx_path1)

    inferred_model2 = shape_inference.infer_shapes(processed_model1)
    inferred_model2 = shape_inference.infer_shapes(inferred_model2)
    inferred_model2 = shape_inference.infer_shapes(inferred_model2)

    # if check:
    #     model_simplified, check = simplify(inferred_model2, skipped_optimizers=["extract_constant_to_initializer"])
    # else:
    #     model_simplified = inferred_model2

    process2 = process_batch_norm(inferred_model2)
    process2 = shape_inference.infer_shapes(process2)
    process2 = shape_inference.infer_shapes(process2)
    onnx.save(process2, inferred_train_onnx_path2)
    if check:
        print(onnx.checker.check_model(process2))

    model_simplified, check = simplify(process2, skipped_optimizers=["extract_constant_to_initializer"])
    process3 = process_1d_nodes(model_simplified)
    process3 = shape_inference.infer_shapes(process3)
    onnx.save(process3, inferred_train_onnx_path3)
    if check:
        print(onnx.checker.check_model(process3))

    # Check for ConCat nodes with more than two inputs and split them
    process3 = process_concat_nodes(process3)
    process3 = shape_inference.infer_shapes(process3)
    if check:
        print(onnx.checker.check_model(process3))
    # Add Optimizer
    optimizer_model, optimizer_inputs, optimizer_outputs = add_optimizer(process3)
    onnx.save(optimizer_model, inferred_train_onnx_path4)

    shape_inference.infer_shapes_path(inferred_train_onnx_path4, inferred_train_onnx_path4)
    if check:
        print(onnx.checker.check_model(inferred_train_onnx_path4))

    # Split Forward, Backward and Optimizer
    onnx_model = onnx.load(inferred_train_onnx_path3)
    forward_inputs, backward_inputs, forward_outputs, backward_outputs = split_forward_backward(onnx_model)

    # print(forward_inputs, backward_inputs, forward_outputs, backward_outputs)
    onnx.utils.extract_model(
        inferred_train_onnx_path3,
        forward_onnx_path,
        list(set([obj[0] for obj in forward_inputs])),
        list(set([obj[0] for obj in forward_outputs])),
        True,
    )
    if check:
        print(onnx.checker.check_model(forward_onnx_path))
    onnx.utils.extract_model(
        inferred_train_onnx_path3,
        backward_onnx_path,
        list(set([obj[0] for obj in backward_inputs])),
        list(set([obj[0] for obj in backward_outputs])),
        True,
    )
    if check:
        print(onnx.checker.check_model(backward_onnx_path))
    # onnx.utils.extract_model(
    #     inferred_train_onnx_path4, optimizer_onnx_path, list(set(optimizer_inputs)), list(set(optimizer_outputs)), True
    # )
    # if check:
    #     print(onnx.checker.check_model(optimizer_onnx_path))

    return inferred_train_onnx_path4, forward_onnx_path, backward_onnx_path, optimizer_onnx_path
