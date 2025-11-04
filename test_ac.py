import copy
import logging
from pathlib import Path

import numpy as np
import onnx.numpy_helper
import torch
from onnxruntime.training import artifacts
from onnxsim import simplify
from stream.api import optimize_allocation_ga
from stream.utils import CostModelEvaluationLUT
from zigzag.parser.onnx.utils import get_attribute_ints_with_name

import onnx
from model.resnet18 import ResNet18
from onnx import shape_inference

# from stream.visualization.memory_usage import plot_memory_usage
# from stream.visualization.perfetto import convert_scme_to_perfetto_json
# from stream.visualization.schedule import visualize_timeline_plotly
from process_onnx import (
    add_optimizer,
    expand_softmax_grad_node,
    process_1d_nodes,
    process_batch_norm,
    process_concat_nodes,
    process_convolution_grad,
    process_poolgrad,
    shape2tuple,
    split_forward_backward,
)
from math import prod

# Set the logging level to ERROR to suppress warnings
# ort.set_default_logger_severity(4)
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def get_shape(name, onnx_model):
    for info in onnx_model.graph.value_info:
        if info.name == name:
            return [dim.dim_value for dim in info.type.tensor_type.shape.dim]
    for init in onnx_model.graph.initializer:
        if init.name == name:
            # For initializers, shape is stored in the tensor itself
            return onnx.numpy_helper.to_array(init).shape
    for input in onnx_model.graph.input:
        if input.name == name:
            return shape2tuple(input)
    raise ValueError(f"Shape for {name} not found in value_info or initializers.")


def get_compute_cost(onnx_model, subgraph_node_names):
    """
    Based on a subgraph defined by their names, compute the cost of computing this subgraph in FLOPs
    """
    compute_cost = 0
    for node in onnx_model.graph.node:
        if node.name in subgraph_node_names:
            if node.op_type == "Conv":
                # Extract attributes
                attrs = node.attribute
                kernel_shape: list[int] = get_attribute_ints_with_name("kernel_shape", attrs, default=None)  # type:ignore
                _ = [attr.i for attr in node.attribute if attr.name == "group"][0]
                # For simplicity, assume group=1 and output_channels is the number of filters
                # Get input and output shapes
                input_name = node.input[0]
                output_name = node.output[0]
                input_shape = get_shape(input_name, onnx_model)
                output_shape = get_shape(output_name, onnx_model)
                # FLOPs = 2 * output_channels * input_channels*kernel_height*kernel_width*output_height*output_width
                input_channels = input_shape[1]  # Assuming NCHW format
                kh, kw = kernel_shape
                oh, ow = output_shape[2], output_shape[3]
                flops = 2 * output_shape[1] * input_channels * kh * kw * oh * ow
                compute_cost += flops

            elif node.op_type == "Relu":
                # FLOPs = number of elements in input
                input_name = node.input[0]
                input_shape = get_shape(input_name, onnx_model)
                flops = 1 * int(np.prod(input_shape))  # 1 FLOP per element
                compute_cost += flops

            elif node.op_type == "Gemm":
                # For Gemm, FLOPs = 2 * M * N * K
                # Get input shapes
                a_shape = get_shape(node.input[0], onnx_model)
                b_shape = get_shape(node.input[1], onnx_model)
                M, K = a_shape
                K, N = b_shape
                flops = 2 * M * N * K
                compute_cost += flops

            elif node.op_type == "MatMul":
                # For MatMul, FLOPs = 2 * M * N * K
                a_shape = get_shape(node.input[0], onnx_model)
                b_shape = get_shape(node.input[1], onnx_model)
                M, K = a_shape[-2:]
                K, N = b_shape[-2:]
                flops = 2 * M * N * K
                compute_cost += flops
    return compute_cost


def get_node_id(onnx_model, node_name):
    index = None
    for i, node in enumerate(onnx_model.graph.node):
        if node.name == node_name:
            index = i
            break
    if index is None:
        raise ValueError(f"Node '{node_name}' not found in the graph.")

    return index


def copy_nodes_in_onnx_model(onnx_model, subgraph_node_names, checkpoint_name, checkpoint_input_nodes_name: str):
    """
    Copies a subgraph, renames its internal edges, and inserts it just before a specific node.

    Args:
        subgraph_node_names (list): List of node names that form the subgraph to copy.
    """
    # Find the target nodes
    target_idx = None
    target_nodes_idx = []
    for node in checkpoint_input_nodes_name:
        target_nodes_idx.append([node, get_node_id(onnx_model, node)])
    target_idx = min([element[1] for element in target_nodes_idx])
    # Collect the subgraph nodes and their edges
    subgraph_nodes = []
    subgraph_inputs = set()
    subgraph_outputs = set()
    for node in onnx_model.graph.node:
        if node.name in subgraph_node_names:
            subgraph_nodes.append(node)
            for inp in node.input:
                # Check if input is produced by another node in the subgraph
                produced_in_subgraph = any(inp in other.output for other in subgraph_nodes)
                if not produced_in_subgraph:
                    subgraph_inputs.add(inp)
            for out in node.output:
                subgraph_outputs.add(out)

    # Create a mapping for renamed internal edges
    internal_edges = set()
    for node in subgraph_nodes:
        for out in node.output:
            if any(out in other.input for other in subgraph_nodes):
                internal_edges.add(out)

    # Create a mapping for renamed internal edges
    edge_mapping = {edge: f"{edge}_copy" for edge in internal_edges}
    edge_mapping[checkpoint_name] = f"{checkpoint_name}_copy"
    # Copy the subgraph nodes and rename internal edges
    copied_nodes = []
    for node in subgraph_nodes:
        new_inputs = []
        for inp in node.input:
            if inp in edge_mapping:
                new_inputs.append(edge_mapping[inp])
            else:
                new_inputs.append(inp)
        new_outputs = []
        for out in node.output:
            if out in edge_mapping:
                new_outputs.append(edge_mapping[out])
            else:
                new_outputs.append(out)

        copied_node = copy.deepcopy(node)
        copied_node.name = f"{node.name}_copy"
        for i, new_input in enumerate(new_inputs):
            copied_node.input[i] = new_input
        for i, new_output in enumerate(new_outputs):
            copied_node.output[i] = new_output
        # copied_node = make_node(
        #     node.op_type,
        #     inputs=new_inputs,
        #     outputs=new_outputs,
        #     name=f"{node.name}_copy",
        #     **node.attribute,
        # )
        # print("Copy", type(copied_node.attribute))
        copied_nodes.append(copied_node)

    # Insert the copied subgraph just before the target node
    for node in reversed(copied_nodes):
        onnx_model.graph.node.insert(target_idx, node)

    # Recreate the target nodes with updated inputs
    for _, target_node_idx in target_nodes_idx:
        target_node = onnx_model.graph.node[target_node_idx + len(copied_nodes)]
        new_target_inputs = []
        for inp in target_node.input:
            if inp in checkpoint_name:
                new_target_inputs.append(f"{checkpoint_name}_copy")
            else:
                new_target_inputs.append(inp)

        for i, new_input in enumerate(new_target_inputs):
            target_node.input[i] = new_input

    return onnx_model


def remove_checkpoint(onnx_model, checkpoint, all_checkpoints, inputs):
    """
    Remove the need to store one checkpoint and replace its need in the backward pass with a recomputation
    """
    checkpoint, nodes = checkpoint
    # all_checkpoints = [element[0] for element in all_checkpoints]
    all_checkpoints_keys = list(all_checkpoints.keys())
    # print(checkpoint, [e.name for e in all_checkpoints[checkpoint]])
    # print(all_checkpoints)
    # Store all nodes involved in the recomputation
    computation_nodes = []

    def search_output_onnx_model(tensor_name):
        for onnx_node in onnx_model.graph.node:
            if tensor_name in onnx_node.output:
                computation_nodes.append(onnx_node)
                for new_node_input in onnx_node.input:
                    recursive_find_computation_nodes(new_node_input)

    # recursively find all nodes inputs are either in the forward pass (weights, inputs) or in the checkpoints
    def recursive_find_computation_nodes(node_input):
        # print(node_input, inputs)
        if node_input in all_checkpoints_keys or node_input in [element[0] for element in inputs]:
            return
        else:
            search_output_onnx_model(node_input)

    # Find the computation nodes required to compute the checkpoint
    search_output_onnx_model(checkpoint)
    copy_nodes_in_onnx_model(
        onnx_model, [node.name for node in computation_nodes], checkpoint, [node.name for node in nodes]
    )
    # all_checkpoints
    compute_cost = get_compute_cost(onnx_model, [node.name for node in computation_nodes])
    checkpoint_shape = get_shape(checkpoint, onnx_model)
    checkpoint_mem = prod(checkpoint_shape) * 2
    return shape_inference.infer_shapes(onnx_model), compute_cost, checkpoint_mem


def apply_onnx_pass(output_path="./", check=True, model=None):
    Path(output_path).mkdir(parents=True, exist_ok=True)

    inferred_train_onnx_path1 = f"{output_path}/model1.onnx"
    inferred_train_onnx_path2 = f"{output_path}/model2.onnx"
    inferred_train_onnx_path3 = f"{output_path}/model3.onnx"
    inferred_train_onnx_path4 = f"{output_path}/model4.onnx"

    # submodels paths
    forward_onnx_path = f"{output_path}/forward.onnx"
    backward_onnx_path = f"{output_path}/backward.onnx"
    optimizer_onnx_path = f"{output_path}/optimizer.onnx"

    # Can not check before adding the domain to the PoolGrad nodes
    processed_model1 = process_poolgrad(model)
    onnx.checker.check_model(processed_model1)
    processed_model1 = process_convolution_grad(processed_model1)
    onnx.checker.check_model(processed_model1)
    processed_model1 = expand_softmax_grad_node(processed_model1)
    onnx.checker.check_model(processed_model1)
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
        onnx.checker.check_model(process2)

    # model_simplified, check = simplify(process2, skipped_optimizers=["extract_constant_to_initializer"])
    process3 = process_1d_nodes(process2)
    process3 = shape_inference.infer_shapes(process3)
    onnx.save(process3, inferred_train_onnx_path3)
    if check:
        onnx.checker.check_model(process3)

    # Check for ConCat nodes with more than two inputs and split them
    process3 = process_concat_nodes(process3)
    process3 = shape_inference.infer_shapes(process3)
    if check:
        onnx.checker.check_model(process3)
    # Add Optimizer
    optimizer_model, optimizer_inputs, optimizer_outputs = add_optimizer(process3)
    onnx.save(optimizer_model, inferred_train_onnx_path4)

    shape_inference.infer_shapes_path(inferred_train_onnx_path4, inferred_train_onnx_path4)
    if check:
        onnx.checker.check_model(inferred_train_onnx_path4)

    # Split Forward, Backward and Optimizer
    onnx_model = onnx.load(inferred_train_onnx_path3)
    forward_inputs, backward_inputs, forward_outputs, backward_outputs = split_forward_backward(onnx_model)

    # print(forward_inputs, backward_inputs, forward_outputs, backward_outputs)
    onnx.utils.extract_model(
        inferred_train_onnx_path3,
        forward_onnx_path,
        list(forward_inputs.keys()),
        list(forward_outputs.keys()),
        True,
    )
    if check:
        onnx.checker.check_model(forward_onnx_path)
    # onnx.utils.extract_model(
    #     inferred_train_onnx_path3,
    #     backward_onnx_path,
    #     list(backward_inputs.keys()),
    #     list(backward_outputs.keys()),
    #     True,
    # )
    # if check:
    #     onnx.checker.check_model(backward_onnx_path)
    # onnx.utils.extract_model(
    #     inferred_train_onnx_path4, optimizer_onnx_path, list(set(optimizer_inputs)), list(set(optimizer_outputs)), True
    # )
    # if check:
    #     print(onnx.checker.check_model(optimizer_onnx_path))

    return inferred_train_onnx_path4, forward_onnx_path, backward_onnx_path, optimizer_onnx_path


def apply_activation_checkpointing(
    torch_model,
    example_input=None,
    accelerator_path=None,
    mapping_path=None,
    output_path="./",
    requires_grad=None,
    mode="torch",
    check=True,
):
    # Output Paths to store intermediary models
    Path(output_path).mkdir(parents=True, exist_ok=True)

    onnx_path = f"{output_path}/model.onnx"
    train_onnx_path = f"{output_path}/training_model.onnx"

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
    inferred_model = shape_inference.infer_shapes(
        shape_inference.infer_shapes(shape_inference.infer_shapes(onnx.load(train_onnx_path)))
    )
    onnx.save(inferred_model, train_onnx_path)

    forward_inputs, backward_inputs, forward_outputs, backward_outputs = split_forward_backward(inferred_model)
    for i, checkpoint in enumerate(forward_outputs[1:-1]):
        inferred_model = onnx.load(train_onnx_path)
        checkpointed_model, compute_cost = remove_checkpoint(
            inferred_model, checkpoint, forward_outputs, forward_inputs
        )
        # print(compute_cost)
        onnx.save(checkpointed_model, f"{folder}ac_{i}.onnx")

        # inferred_train_onnx_path4, forward_onnx_path, backward_onnx_path, optimizer_onnx_path = apply_onnx_pass(
        #     output_path=f"{folder}ac_{i}/", model=checkpointed_model
        # )
        # run_stream(
        #     inferred_train_onnx_path4,
        #     accelerator_path=accelerator_path,
        #     mapping_path=mapping_path,
        #     id=1,
        #     output_path=f"{folder}ac_{i}/",
        # )


def run_stream(model_path, accelerator_path, mapping_path, id, output_path):
    mode = "fused"
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
    # Helper function to get shape from value_info


if __name__ == "__main__":
    folder = "results/ac_test/"
    onnx_path = f"{folder}model.onnx"
    infered_path = f"{folder}infered.onnx"
    soc_path = "stream/stream/inputs/examples/hardware/tpu_like_quad_core.yaml"
    mapping_path = "stream/stream/inputs/examples/mapping/tpu_like_quad_core_ga.yaml"
    output_path = folder

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

    apply_activation_checkpointing(base_model, None, soc_path, mapping_path, folder, requires_grad, "onnx")
    # Now, we can invoke generate_artifacts with this custom loss function
    # artifacts.generate_artifacts(
    #     base_model, requires_grad=requires_grad, loss=loss, optimizer=artifacts.OptimType.AdamW, prefix=folder
    # )

    # # Infer training graph
    # inferred_model = shape_inference.infer_shapes_path(train_onnx_path, inferred_train_onnx_path)
    # inferred_model = shape_inference.infer_shapes_path(train_onnx_path, inferred_train_onnx_path)
    # inferred_model = shape_inference.infer_shapes_path(train_onnx_path, inferred_train_onnx_path)

    # model = onnx.load(inferred_train_onnx_path)
    # forward_inputs, backward_inputs, forward_outputs, backward_outputs = split_forward_backward(model)
    # print([element[0] for element in forward_outputs], [element[0] for element in forward_inputs])

    # for i, checkpoint in enumerate(forward_outputs[2:]):
    #     onnx_model, compute_cost = remove_checkpoint(model, checkpoint, forward_inputs, forward_outputs)
    #     print(compute_cost)
    #     onnx.save(onnx_model, f"{folder}ac_{i}.onnx")
    #     # break

    # forward_inputs, backward_inputs, forward_outputs, backward_outputs = split_forward_backward(onnx_model)
    # print([element[0] for element in forward_outputs], [element[0] for element in forward_inputs])
