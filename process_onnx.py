import numpy as np
from onnx.helper import make_node, make_tensor_value_info
from zigzag.parser.onnx.utils import get_attribute_ints_with_name, get_onnx_tensor_type

import onnx
from onnx import TensorProto, helper, numpy_helper


# TODO: refactor and split the different functions for clarity and ease of maintenance
def shape2tuple(shape):
    return tuple(getattr(d, "dim_value", 0) for d in shape.type.tensor_type.shape.dim)


def get_sliding_window_shape(input_shape, kernel_shape, strides, dilations, padding):
    """
    Calculate the shape of the sliding window based on kernel shape, strides, dilations, and padding.
    """
    out_h = (input_shape[2] + 2 * padding[0] - dilations[0] * (kernel_shape[0] - 1) - 1) // strides[0] + 1
    out_w = (input_shape[3] + 2 * padding[1] - dilations[1] * (kernel_shape[1] - 1) - 1) // strides[1] + 1
    kh = kernel_shape[0]
    kw = kernel_shape[1]
    width = input_shape[3]
    # Create indices for gathering

    h_indices = []
    w_indices = []
    # Compute the indices for the height and width dimensions
    for i in range(out_h):
        temp_indices = []
        h_start = i * strides[0]
        for ki in range(kh):
            h = h_start + ki * dilations[0]
            temp_indices.append(h)
        h_indices.append(temp_indices)

    for j in range(out_w):
        temp_indices = []
        w_start = i * strides[1]
        for kj in range(kw):
            w = w_start + kj * dilations[1]
            temp_indices.append(w)
        w_indices.append(temp_indices)

    h_indices = np.transpose(np.array(h_indices))
    w_indices = np.transpose(np.array(w_indices))

    return h_indices, w_indices


def make_initializer(name, data, dtype):
    """
    Create an ONNX initializer tensor.
    """
    tensor = numpy_helper.from_array(np.array(data, dtype=dtype), name=name)
    return tensor


def split_forward_backward(onnx_model):
    # Output Format is dictlist[tensor.name, list[nodes with tensor as input]]
    forward_inputs = {}
    backward_inputs = {}
    for input_tensor in onnx_model.graph.input:
        if "grad" in input_tensor.name:
            backward_inputs[input_tensor.name] = [None]
            # backward_inputs.append([input_tensor.name, [None]])
        else:
            forward_inputs[input_tensor.name] = [None]
            # forward_inputs.append([input_tensor.name, [None]])

    forward_outputs = {}
    backward_outputs = {}
    for output_tensor in onnx_model.graph.output:
        if "grad" in output_tensor.name:
            backward_outputs[input_tensor.name] = [None]
            # backward_outputs.append([output_tensor.name, [None]])
        else:
            forward_outputs[input_tensor.name] = [None]
            # forward_outputs.append([output_tensor.name, [None]])

    # Find the index of the first LossGrad node
    for i, op_node in enumerate(onnx_model.graph.node):
        if "LossGrad" in op_node.op_type:
            sep_index = i
            break

    for i, op_node in enumerate(onnx_model.graph.node):
        if i < sep_index:
            for j, op_node2 in enumerate(onnx_model.graph.node):
                if j >= sep_index:
                    for output_name in op_node.output:
                        if output_name in op_node2.input:
                            if output_name in forward_outputs.keys():
                                forward_outputs[output_name].append(op_node2)
                            else:
                                forward_outputs[output_name] = [op_node2]

                            if output_name in backward_inputs.keys():
                                backward_inputs[output_name].append(op_node)
                            else:
                                backward_inputs[output_name] = [op_node]
                            # if output_name not in [name for name, _ in forward_outputs]:
                            #     forward_outputs.append([output_name, op_node2])
                            # if output_name not in [name for name, _ in backward_inputs]:
                            #     backward_inputs.append([output_name, op_node])

    for i, op_node in enumerate(onnx_model.graph.node):
        if i >= sep_index:
            for input_tensor in onnx_model.graph.input:
                if input_tensor.name in op_node.input:
                    if input_tensor.name in backward_inputs.keys():
                        backward_inputs[input_tensor.name].append(op_node)
                    else:
                        backward_inputs[input_tensor.name] = [op_node]

    return forward_inputs, backward_inputs, forward_outputs, backward_outputs


def process_poolgrad(onnx_model):
    """
    Process PoolGrad nodes in the ONNX model to update their domain to com.microsoft as they are not processed by default.
    """
    for i, op_node in enumerate(onnx_model.graph.node):
        if op_node.op_type in {"MaxPoolGrad", "AveragePoolGrad"}:
            op_node.domain = "com.microsoft"
    return onnx_model


def process_batch_norm(onnx_model):
    for node in onnx_model.graph.node:
        if node.op_type == "BatchNormInternal":
            shape = get_onnx_tensor_type(node.input[1], onnx_model).shape
            for output in node.output[1:]:
                onnx_model.graph.value_info.append(make_tensor_value_info(output, TensorProto.FLOAT, shape))
        if node.op_type == "BatchNormalizationGrad":
            shape = get_onnx_tensor_type(node.input[2], onnx_model).shape
            for output in node.output[1:]:
                onnx_model.graph.value_info.append(make_tensor_value_info(output, TensorProto.FLOAT, shape))

    return onnx_model


def process_1d_nodes(onnx_model):
    """
    Process Edges that contains only 1D dimension by fusing the producer/consumer nodes.
    This is a temporary solution to handle 1D nodes that are not supported by Stream.
    It will be removed once Stream supports 1D nodes.
    """
    input_nodes_1D = []
    output_nodes_1D = []

    def remove_linked_nodes(op_node, onnx_model):
        """
        Remove the all nodes that are linked to the input node.
        """

        input_names = op_node.input
        output_names = op_node.output
        # print(op_node.name, "  ", op_node.op_type, input_names, output_names)
        # If the node is a Constant node, remove it from the graph
        if op_node.op_type == "Constant":
            onnx_model.graph.node.remove(op_node)
            return

        # Remove the input and output tensors from the graph
        for input_tensor in onnx_model.graph.input:
            if input_tensor.name in input_names and input_tensor.name not in ["lazy_reset_grad", "input"]:
                onnx_model.graph.input.remove(input_tensor)
        for output_tensor in onnx_model.graph.output:
            if output_tensor.name in output_names:
                onnx_model.graph.output.remove(output_tensor)

        for i, op_node2 in enumerate(onnx_model.graph.node):
            if any(output_name in op_node2.input for output_name in output_names):
                remove_linked_nodes(op_node2, onnx_model)

        # Remove the node from the graph
        if op_node in onnx_model.graph.node:
            onnx_model.graph.node.remove(op_node)

    for i, op_node in enumerate(onnx_model.graph.node):
        # Replace SoftmaxCrossEntropyLoss By Identity
        if op_node.op_type == "SoftmaxCrossEntropyLoss":
            inputs = op_node.input
            outputs = op_node.output
            node_transpose = make_node("Identity", [inputs[0]], [outputs[1]], name=op_node.name + "_identity")

            onnx_model.graph.node.insert(i, node_transpose)
            onnx_model.graph.node.remove(op_node)
            for output_tensor in onnx_model.graph.output:
                if output_tensor.name == outputs[0]:
                    onnx_model.graph.output.remove(output_tensor)

    for i, op_node in enumerate(onnx_model.graph.node):
        if op_node.op_type not in ["Constant"]:
            inputs = op_node.input
            outputs = op_node.output

            for j, input_name in enumerate(inputs):
                try:
                    input_tensor = get_onnx_tensor_type(input_name, onnx_model)
                    if input_tensor is not None and len(input_tensor.shape) == 1:
                        input_nodes_1D.append([input_name, i, j, op_node])
                except Exception as _:
                    continue
            for j, output_name in enumerate(outputs):
                try:
                    output_tensor = get_onnx_tensor_type(output_name, onnx_model)
                    if output_tensor is not None and len(output_tensor.shape) == 1:
                        output_nodes_1D.append([output_name, i, j, op_node])
                except Exception as _:
                    continue

    for input_name, input_node_index, input_index, input_node in input_nodes_1D:
        for output_name, output_node_index, output_index, output_node in output_nodes_1D:
            if input_name == output_name:
                output_node.output[output_index] = "output_" + input_name
                remove_linked_nodes(input_node, onnx_model)

    return onnx_model


def process_convolution_grad(onnx_model):
    """
    Process ConvGrad nodes in the ONNX model to convert them into ConvTranspose nodes and reshape the inputs accordingly.
    This is necessary because ConvGrad nodes use multiple output nodes which make them not easily implementable in Stream.
    """

    updated_list = []
    for i, op_node in enumerate(onnx_model.graph.node):
        if op_node.op_type == "ConvGrad":
            attrs = op_node.attribute
            kernel_shape: list[int] = get_attribute_ints_with_name("kernel_shape", attrs, default=None)  # type:ignore
            strides: list[int] = get_attribute_ints_with_name("strides", attrs, default=[1, 1])  # type:ignore
            dilations: list[int] = get_attribute_ints_with_name("dilations", attrs, default=[1, 1])  # type:ignore
            group_size: int = get_attribute_ints_with_name("group", attrs, default=1)  # type:ignore
            padding: list[int] = get_attribute_ints_with_name("pads", attrs, default=[0, 0, 0, 0])  # type:ignore

            input_grad_shape = get_onnx_tensor_type(op_node.input[0], onnx_model).shape
            input_shape = get_onnx_tensor_type(op_node.input[1], onnx_model).shape
            weight_shape = get_onnx_tensor_type(op_node.input[2], onnx_model).shape

            transpose_attr = {"perm": [1, 0, 2, 3]}
            input_grad_attr = {
                "kernel_shape": kernel_shape,
                "strides": strides,
                "dilations": dilations,
                "group": group_size,
                "pads": padding,
            }

            # Create relevant nodes
            node_list = []
            input_list = []
            initializer_list = []

            output_name = op_node.output[0]
            if len(op_node.output[0]) == 0:
                output_name = "input_grad" + op_node.name
            node_input_grad = make_node(
                "ConvTranspose", [op_node.input[0], op_node.input[2]], [output_name], **input_grad_attr
            )
            node_list.append(node_input_grad)

            # Compute the gradient relative to the weight
            if len(op_node.output[1]) > 0:
                weight_gradient_name = "_WG_" + op_node.output[1] + "_" + op_node.name

                sliding_window_h = (
                    input_shape[2] + 2 * padding[0] - dilations[0] * (kernel_shape[0] - 1) - 1
                ) // strides[0] + 1
                sliding_window_w = (
                    input_shape[3] + 2 * padding[1] - dilations[1] * (kernel_shape[1] - 1) - 1
                ) // strides[1] + 1
                transposed_shape = [
                    input_shape[0],
                    sliding_window_h * sliding_window_w,
                    input_shape[1] * kernel_shape[0] * kernel_shape[1],
                ]

                initializer_list.append(
                    make_initializer(
                        "shape_axis" + weight_gradient_name,
                        [input_grad_shape[0], input_grad_shape[1], input_grad_shape[2] * input_grad_shape[3]],
                        np.int64,
                    )
                )

                if op_node.input[1] != onnx_model.graph.input[0].name:
                    # Add padding if necessary
                    if max(padding) > 0:
                        pads = np.array([0, 0, padding[0], padding[1], 0, 0, padding[2], padding[3]])
                        padding_value_node = helper.make_node(
                            "Constant",
                            inputs=[],
                            outputs=["WG_paddings_" + weight_gradient_name],
                            value=numpy_helper.from_array(pads),
                        )
                        pad_node = helper.make_node(
                            "Pad",
                            inputs=[op_node.input[1], "WG_paddings_" + weight_gradient_name],
                            outputs=["padded_input" + weight_gradient_name],
                            mode="constant",
                            name=weight_gradient_name + "_PaddingActivation",
                        )

                        node_list.append(padding_value_node)
                        node_list.append(pad_node)
                        input_name = "padded_input" + weight_gradient_name
                    else:
                        input_name = op_node.input[1]

                    h_indices, w_indices = get_sliding_window_shape(
                        input_shape, kernel_shape, strides, dilations, padding
                    )
                    # Create Constant Nodes for h and w indices
                    h_indices_constant_node = helper.make_node(
                        "Constant",
                        inputs=[],
                        outputs=["h_indices_" + weight_gradient_name],
                        value=numpy_helper.from_array(h_indices),
                    )
                    w_indices_constant_node = helper.make_node(
                        "Constant",
                        inputs=[],
                        outputs=["w_indices_" + weight_gradient_name],
                        value=numpy_helper.from_array(w_indices),
                    )
                    node_list.append(h_indices_constant_node)
                    node_list.append(w_indices_constant_node)

                    # Unfold Operation
                    gather_node_h = make_node(
                        "Gather",
                        [input_name, "h_indices_" + weight_gradient_name],
                        ["h_gathered" + weight_gradient_name],
                        axis=2,
                        name=weight_gradient_name + "_GatherWeight1",
                    )
                    gather_node_w = make_node(
                        "Gather",
                        ["h_gathered" + weight_gradient_name, "w_indices_" + weight_gradient_name],
                        ["w_h_gathered" + weight_gradient_name],
                        axis=4,
                        name=weight_gradient_name + "_GatherWeight2",
                    )
                    transpose_node_gather = make_node(
                        "Transpose",
                        ["w_h_gathered" + weight_gradient_name],
                        ["gathered_c" + weight_gradient_name],
                        perm=[0, 3, 5, 4, 1, 2],
                        name=weight_gradient_name + "_TransposeWeight1",
                    )

                    node_list.append(gather_node_h)
                    node_list.append(gather_node_w)
                    node_list.append(transpose_node_gather)

                    # Create reshape node
                    initializer_list.append(
                        make_initializer("shape_axis2" + weight_gradient_name, transposed_shape, np.int64)
                    )
                    reshape_node = helper.make_node(
                        "Reshape",
                        inputs=["gathered_c" + weight_gradient_name, "shape_axis2" + weight_gradient_name],
                        outputs=["transpose1_output1_" + weight_gradient_name],
                        name=weight_gradient_name + "Reshape_Axis2",
                    )
                    node_list.append(reshape_node)
                else:
                    transposed_input = make_tensor_value_info(
                        "transpose1_output1_" + weight_gradient_name, TensorProto.FLOAT, transposed_shape
                    )
                    input_list.append(transposed_input)

                node_reshape = make_node(
                    "Reshape",
                    inputs=[op_node.input[0], "shape_axis" + weight_gradient_name],
                    outputs=["shape" + weight_gradient_name],
                    name=weight_gradient_name + "Reshape_Axis",
                )
                node_list.append(node_reshape)
                node_matmul = make_node(
                    "MatMul",
                    ["shape" + weight_gradient_name, "transpose1_output1_" + weight_gradient_name],
                    ["matmul_output_" + weight_gradient_name],
                    name=weight_gradient_name + "_MatMul",
                )
                node_list.append(node_matmul)
                initializer_list.append(make_initializer("ReduceSumTensor1" + weight_gradient_name, [0], np.int64))
                node_sum = make_node(
                    "ReduceSum",
                    ["matmul_output_" + weight_gradient_name, "ReduceSumTensor1" + weight_gradient_name],
                    ["batch_sum" + weight_gradient_name],
                    name=weight_gradient_name + "_ReduceSumWeight",
                    keepdims=0,
                )
                node_list.append(node_sum)

                node_reshape_axis3 = make_node(
                    "Constant",
                    [],
                    ["shape_axis3" + weight_gradient_name],
                    value=numpy_helper.from_array(
                        np.array([weight_shape[0], weight_shape[1], weight_shape[2], weight_shape[3]], dtype=np.int64)
                    ),
                    name=weight_gradient_name + "_Constant",
                )
                node_reshape2 = make_node(
                    "Reshape",
                    ["batch_sum" + weight_gradient_name, "shape_axis3" + weight_gradient_name],
                    [op_node.output[1]],
                    name=weight_gradient_name + "_Reshape_Axis3",
                )
                node_list.append(node_reshape_axis3)
                node_list.append(node_reshape2)

            # Bias computation
            if len(op_node.output) >= 3 and len(op_node.output[2]) > 0:
                # Create Initializer for the input of the reduce sum
                node_reduce_sum_axis2 = make_node(
                    "Constant",
                    [],
                    ["ReduceSumTensor2" + op_node.name],
                    value=numpy_helper.from_array(np.array([1, 2, 3], dtype=np.int64)),
                    name=op_node.name + "_ConstantBias",
                )
                node_transpose4 = make_node(
                    "Transpose",
                    [op_node.input[0]],
                    ["transposed_bias_grad_" + op_node.name],
                    name=op_node.name + "_TransposeBias",
                    **transpose_attr,
                )
                node_bias_grad = make_node(
                    "ReduceSum",
                    ["transposed_bias_grad_" + op_node.name, "ReduceSumTensor2" + op_node.name],
                    [op_node.output[2]],
                    name=op_node.name + "_ReduceSumBias",
                    keepdims=0,
                )
                node_list.append(node_reduce_sum_axis2)
                node_list.append(node_transpose4)
                node_list.append(node_bias_grad)

            updated_list.append([op_node, i, node_list, input_list, initializer_list])

    n = 0
    for remove_node, index, append_list, new_input_list, initializer_list in updated_list:
        for new_input in new_input_list:
            onnx_model.graph.input.append(new_input)
        for new_initializer in initializer_list:
            onnx_model.graph.initializer.append(new_initializer)
        for j, op_node in enumerate(append_list):
            onnx_model.graph.node.insert(index + n, op_node)
            n += 1
        onnx_model.graph.node.remove(remove_node)
        n -= 1

    return onnx_model


def add_optimizer2(
    onnx_model, optimizer_name="Adam", learning_rate=0.001, weight_decay=0, beta1=0.9, beta2=0.999, epsilon=1e-8
):
    """
    Add the optimizer to the Forward and Backward nodes in the ONNX model.
    """
    node_list = []
    # Create an initializer for the learning rate
    onnx_model.graph.initializer.append(make_initializer("learning_rate", [learning_rate], np.float32))
    onnx_model.graph.initializer.append(make_initializer("weight_decay", [weight_decay], np.float32))
    onnx_model.graph.initializer.append(make_initializer("epsilon", [epsilon], np.float32))
    onnx_model.graph.initializer.append(make_initializer("beta1", [beta1], np.float32))
    onnx_model.graph.initializer.append(make_initializer("beta2", [beta2], np.float32))

    # Lists to store the name of all the inputs and outputs required to run only the optimizer
    optimizer_inputs = []
    optimizer_outputs = []
    nodes_to_remove = []
    outputs_to_remove = []
    inputs_to_remove = []
    # Get all gradients accumulations outputs of the model
    grad_accumulation_names = []
    for input_tensor in onnx_model.graph.input:
        if "grad.accumulation.buffer" in input_tensor.name:
            grad_accumulation_names.append(input_tensor.name)
            optimizer_inputs.append(input_tensor.name)

    # optimizer_inputs.append("lazy_reset_grad")
    inputs_to_remove.append("lazy_reset_grad")
    # Remove InPlaceAccumulatorV2 nodes from the graph

    for node in onnx_model.graph.node:
        if node.op_type == "InPlaceAccumulatorV2":
            nodes_to_remove.append(node)
            outputs_to_remove.append(node.output[0])

    for node in nodes_to_remove:
        onnx_model.graph.node.remove(node)

    for output_to_remove in outputs_to_remove:
        for output in onnx_model.graph.output:
            if output.name in outputs_to_remove:
                onnx_model.graph.output.remove(output)

    # Collect gradient weights and their corresponding accumulation buffers
    gradient_weights = []
    for input_tensor in onnx_model.graph.input:
        if any(
            input_tensor.name in grad_name and "buffer" not in input_tensor.name
            for grad_name in grad_accumulation_names
        ):
            gradient_weights.append(input_tensor.name)

    for input_name in gradient_weights:
        # Create a Sum node to add the gradient accumulation buffer and the gradient
        sum_node = make_node(
            "Sum",
            [input_name, input_name + "_grad.accumulation.buffer"],
            [input_name + "_summed_grad"],
            name=f"Sum_Grad_{input_name}",
        )
        node_list.append(sum_node)
        gradient_buffer_name = input_name + "_summed_grad"

        # Create the optimizer nodes (only SGD and Adam are supported for now)
        if weight_decay != 0:
            # Create a weight decay node
            weight_decay_node = make_node(
                "Mul",
                [input_name, "weight_decay"],
                [input_name + "_weight_decay"],
                name=f"{optimizer_name}_WeightDecay_{input_name}",
            )
            g_node = make_node(
                "Add",
                [gradient_buffer_name, input_name + "_weight_decay"],
                [gradient_buffer_name + "_optimizer_g"],
                name=f"{optimizer_name}_WeightDecay_Add_{input_name}",
            )
            node_list.append(weight_decay_node)
            node_list.append(g_node)
            gradient_buffer_name = gradient_buffer_name + "_optimizer_g"

        # For now, we do not consider maximize and amsgrad
        if optimizer_name == "Adam":
            final_gradient_name = input_name + "_optimizer"
            # Add inputs for the previous optimizers states
            weight_shape = get_onnx_tensor_type(input_name, onnx_model).shape
            onnx_model.graph.input.append(
                make_tensor_value_info(input_name + "_optimizer_first_moment", TensorProto.FLOAT, weight_shape)
            )
            onnx_model.graph.input.append(
                make_tensor_value_info(input_name + "_optimizer_second_moment", TensorProto.FLOAT, weight_shape)
            )
            optimizer_inputs.extend(
                [
                    input_name,
                    gradient_buffer_name,
                    input_name + "_optimizer_first_moment",
                    input_name + "_optimizer_second_moment",
                ]
            )
            # First Moment Computation
            mul_first_moment_node = make_node(
                "Mul",
                [input_name + "_optimizer_first_moment", "beta1"],
                [input_name + "_optimizer11"],
                name=f"{optimizer_name}_Optimizer_{input_name}_MulFirstMoment",
            )
            mul_first_moment_node2 = make_node(
                "Mul",
                [gradient_buffer_name, "beta1"],
                [input_name + "_optimizer10"],
                name=f"{optimizer_name}_Optimizer_{input_name}_MulFirstMoment2",
            )
            add_first_moment_node = make_node(
                "Add",
                [input_name + "_optimizer10", input_name + "_optimizer11"],
                [input_name + "_optimizer9"],
                name=f"{optimizer_name}_Optimizer_{input_name}_First_Moment",
            )
            mean_first_moment_node = make_node(
                "Div",
                [input_name + "_optimizer9", "beta1"],
                [input_name + "_optimizer1"],
                name=f"{optimizer_name}_Optimizer_{input_name}_MeanFirst_Moment",
            )
            # Second Moment Computation
            mul_node3 = make_node(
                "Mul",
                [gradient_buffer_name, gradient_buffer_name],
                [input_name + "_optimizer8"],
                name=f"{optimizer_name}_Optimizer_{input_name}_Mul3",
            )
            mul_node2 = make_node(
                "Mul",
                [input_name + "_optimizer8", "beta2"],
                [input_name + "_optimizer7"],
                name=f"{optimizer_name}_Optimizer_{input_name}_Mul2",
            )
            mul_node1 = make_node(
                "Mul",
                [input_name + "_optimizer_second_moment", "beta2"],
                [input_name + "_optimizer6"],
                name=f"{optimizer_name}_Optimizer_{input_name}_Mul1",
            )
            add_second_moment_node = make_node(
                "Add",
                [input_name + "_optimizer6", input_name + "_optimizer7"],
                [input_name + "_optimizer5"],
                name=f"{optimizer_name}_Optimizer_{input_name}_Second_Moment",
            )
            mean_second_moment_node = make_node(
                "Div",
                [input_name + "_optimizer5", "beta2"],
                [input_name + "_optimizer4"],
                name=f"{optimizer_name}_Optimizer_{input_name}_MeanSecond_Moment",
            )
            second_moment_node_1 = make_node(
                "Sqrt",
                [input_name + "_optimizer4"],
                [input_name + "_optimizer3"],
                name=f"{optimizer_name}_Optimizer_{input_name}_second_moment1",
            )
            second_moment_node = make_node(
                "Add",
                [input_name + "_optimizer3", "epsilon"],
                [input_name + "_optimizer2"],
                name=f"{optimizer_name}_Optimizer_{input_name}_second_moment",
            )
            final_optimizer_node = make_node(
                "Div",
                [input_name + "_optimizer1", input_name + "_optimizer2"],
                [final_gradient_name],
                name=f"{optimizer_name}_Optimizer_{input_name}",
            )
            node_list.extend(
                [
                    mul_first_moment_node,
                    mul_first_moment_node2,
                    add_first_moment_node,
                    mean_first_moment_node,
                    mul_node3,
                    mul_node2,
                    mul_node1,
                    add_second_moment_node,
                    mean_second_moment_node,
                    second_moment_node_1,
                    second_moment_node,
                    final_optimizer_node,
                ]
            )
        elif optimizer_name == "SGD":
            optimizer_inputs.append(input_name)
            optimizer_inputs.append(gradient_buffer_name)
            final_gradient_name = gradient_buffer_name

        optimizer_node_1 = make_node(
            "Mul",
            [final_gradient_name, "learning_rate"],
            [input_name + "_optimizer0"],
            name=f"{optimizer_name}_Optimizer_{input_name}",
        )
        optimizer_node_2 = make_node(
            "Sub",
            [input_name, input_name + "_optimizer0"],
            [input_name + "_optimizer_end"],
            name=f"{optimizer_name}_Update_{input_name}",
        )
        node_list.append(optimizer_node_1)
        node_list.append(optimizer_node_2)
        output_tensor = helper.make_tensor_value_info(input_name + "_optimizer_end", TensorProto.FLOAT, None)
        optimizer_outputs.append(input_name + "_optimizer_end")
        onnx_model.graph.output.append(output_tensor)

    for node in node_list:
        onnx_model.graph.node.append(node)

    return onnx_model, optimizer_inputs, optimizer_outputs


def add_optimizer(
    onnx_model, optimizer_name="Adam", learning_rate=0.001, weight_decay=0, beta1=0.9, beta2=0.999, epsilon=1e-8
):
    """
    Add the optimizer to the Forward and Backward nodes in the ONNX model.
    """
    node_list = []
    # Create an intializer for the learning rate
    onnx_model.graph.initializer.append(make_initializer("learning_rate", [learning_rate], np.float32))
    onnx_model.graph.initializer.append(make_initializer("weight_decay", [weight_decay], np.float32))
    onnx_model.graph.initializer.append(make_initializer("epsilon", [epsilon], np.float32))
    onnx_model.graph.initializer.append(make_initializer("beta1", [epsilon], np.float32))
    onnx_model.graph.initializer.append(make_initializer("beta2", [epsilon], np.float32))

    # Lists to store the name of all the inputs and outputs required to run only the optimizer
    optimizer_inputs = []
    optimizer_outputs = []
    # Get all gradients accumulations outputs of the model
    grad_accumulation_names = []
    for input_tensor in onnx_model.graph.input:
        if "grad.accumulation.buffer" in input_tensor.name:
            grad_accumulation_names.append(input_tensor.name)
            # If the input tensor is a gradient accumulation output, we add it to the input list
            optimizer_inputs.append(input_tensor.name)
    optimizer_inputs.append("lazy_reset_grad")
    # for output_tensor in onnx_model.graph.output:
    #     if "grad.accumulation" in output_tensor.name:
    #         optimizer_outputs.append(output_tensor.name)

    grad_accumulation_names2 = []
    for node in onnx_model.graph.node:
        if node.op_type == "InPlaceAccumulatorV2":
            for input in node.input:
                if "grad.accumulation.buffer" in input:
                    grad_accumulation_names2.append(node.input[1])

    # gradient_weights = []
    # for i, input_tensor in enumerate(onnx_model.graph.input):
    #     for element, element2 in zip(grad_accumulation_names, grad_accumulation_names2, strict=False):
    #         if element.startswith(input_tensor.name) and "buffer" not in input_tensor.name:
    #             # If the input tensor is a gradient accumulation output, we add it to the input list
    #             gradient_weights.append((input_tensor.name, element2))

    # Get the accumulation buffer name and the weight gradient
    accumulator_inputs = []
    outputs_to_remove = []
    nodes_to_remove = []
    for node in onnx_model.graph.node:
        if node.op_type == "InPlaceAccumulatorV2":
            if len(node.input) >= 3:
                accumulator_inputs.append([input for input in node.input] + [node.input[1][:-5]])
                nodes_to_remove.append(node)
                outputs_to_remove.append(node.output[0])

    for node in nodes_to_remove:
        onnx_model.graph.node.remove(node)

    for _ in outputs_to_remove:
        for output in onnx_model.graph.output:
            if output.name in outputs_to_remove:
                onnx_model.graph.output.remove(output)

    for gradient_buffer_name, weight_gradient, _, weight_name in accumulator_inputs:
        # Create the optimizer nodes (only SGD and Adam are supported for now)

        sum_node = make_node(
            "Sum",
            [weight_gradient, gradient_buffer_name],
            [weight_name + "_summed_grad"],
            name=f"Sum_Grad_{weight_name}",
        )
        total_gradient = weight_name + "_summed_grad"
        node_list.append(sum_node)

        if weight_decay != 0:
            # Create a weight decay node
            weight_decay_node = make_node(
                "Mul",
                [weight_name, "weight_decay"],
                [weight_name + "_weight_decay"],
                name=f"{optimizer_name}_WeightDecay_{weight_name}",
            )
            g_node = make_node(
                "Add",
                [total_gradient, weight_name + "_weight_decay"],
                [weight_name + "_optimizer_g"],
                name=f"{optimizer_name}_WeightDecay_Add_{weight_name}",
            )
            node_list.append(weight_decay_node)
            node_list.append(g_node)
            total_gradient = weight_name + "_optimizer_g"
            # gradient_buffer_name = gradient_buffer_name + "_optimizer_g"
        # For now we do not consider maximize and amsgrad (as detailed here https://docs.pytorch.org/docs/stable/generated/torch.optim.Adam.html)
        # Without loss of generality for Stream, we assume that the constant computation (1-Beta2) Beta**t is done and is Beta2 (same for Beta1)
        if optimizer_name == "Adam":
            final_gradient_name = weight_name + "_optimizer"
            # Add inputs for the previous optimizers states
            weight_shape = get_onnx_tensor_type(weight_name, onnx_model).shape
            onnx_model.graph.input.append(
                make_tensor_value_info(weight_name + "_optimizer_first_moment", TensorProto.FLOAT, weight_shape)
            )
            onnx_model.graph.input.append(
                make_tensor_value_info(weight_name + "_optimizer_second_moment", TensorProto.FLOAT, weight_shape)
            )

            optimizer_inputs.extend(
                [
                    weight_name,
                    gradient_buffer_name,
                    weight_name + "_optimizer_first_moment",
                    weight_name + "_optimizer_second_moment",
                ]
            )
            # First Moment Computation
            mul_first_moment_node = make_node(
                "Mul",
                [weight_name + "_optimizer_first_moment", "beta1"],
                [weight_name + "_optimizer11"],
                name=f"{optimizer_name}_Optimizer_{weight_name}_MulFirstMoment",
            )
            mul_first_moment_node2 = make_node(
                "Mul",
                [total_gradient, "beta1"],
                [weight_name + "_optimizer10"],
                name=f"{optimizer_name}_Optimizer_{weight_name}_MulFirstMoment2",
            )
            add_first_moment_node = make_node(
                "Add",
                [weight_name + "_optimizer10", weight_name + "_optimizer11"],
                [weight_name + "_optimizer9"],
                name=f"{optimizer_name}_Optimizer_{weight_name}_First_Moment",
            )
            mean_first_moment_node = make_node(
                "Div",
                [weight_name + "_optimizer9", "beta1"],
                [weight_name + "_optimizer1"],
                name=f"{optimizer_name}_Optimizer_{weight_name}_MeanFirst_Moment",
            )
            # Second Moment Computation
            mul_node3 = make_node(
                "Mul",
                [total_gradient, total_gradient],
                [weight_name + "_optimizer8"],
                name=f"{optimizer_name}_Optimizer_{weight_name}_Mul3",
            )
            mul_node2 = make_node(
                "Mul",
                [weight_name + "_optimizer8", "beta2"],
                [weight_name + "_optimizer7"],
                name=f"{optimizer_name}_Optimizer_{weight_name}_Mul2",
            )
            mul_node1 = make_node(
                "Mul",
                [weight_name + "_optimizer_second_moment", "beta2"],
                [weight_name + "_optimizer6"],
                name=f"{optimizer_name}_Optimizer_{weight_name}_Mul1",
            )
            add_second_moment_node = make_node(
                "Add",
                [weight_name + "_optimizer6", weight_name + "_optimizer7"],
                [weight_name + "_optimizer5"],
                name=f"{optimizer_name}_Optimizer_{weight_name}_Second_Moment",
            )
            mean_second_moment_node = make_node(
                "Div",
                [weight_name + "_optimizer5", "beta2"],
                [weight_name + "_optimizer4"],
                name=f"{optimizer_name}_Optimizer_{weight_name}_MeanSecond_Moment",
            )
            second_moment_node_1 = make_node(
                "Sqrt",
                [weight_name + "_optimizer4"],
                [weight_name + "_optimizer3"],
                name=f"{optimizer_name}_Optimizer_{weight_name}_second_moment1",
            )
            second_moment_node = make_node(
                "Add",
                [weight_name + "_optimizer3", "epsilon"],
                [weight_name + "_optimizer2"],
                name=f"{optimizer_name}_Optimizer_{weight_name}_second_moment",
            )
            final_optimizer_node = make_node(
                "Div",
                [weight_name + "_optimizer1", weight_name + "_optimizer2"],
                [final_gradient_name],
                name=f"{optimizer_name}_Optimizer_{weight_name}",
            )
            node_list.extend(
                [
                    mul_first_moment_node,
                    mul_first_moment_node2,
                    add_first_moment_node,
                    mean_first_moment_node,
                    mul_node3,
                    mul_node2,
                    mul_node1,
                    add_second_moment_node,
                    mean_second_moment_node,
                    second_moment_node_1,
                    second_moment_node,
                    final_optimizer_node,
                ]
            )
        elif optimizer_name == "SGD":
            optimizer_inputs.append(weight_name, gradient_buffer_name)
            final_gradient_name = gradient_buffer_name
        optimizer_node_1 = make_node(
            "Mul",
            [final_gradient_name, "learning_rate"],
            [weight_name + "_optimizer0"],
            name=f"{optimizer_name}_Optimizer_{weight_name}",
        )
        optimizer_node_2 = make_node(
            "Sub",
            [weight_name, weight_name + "_optimizer0"],
            [weight_name + "_optimizer_end"],
            name=f"{optimizer_name}_Update_{weight_name}",
        )
        node_list.append(optimizer_node_1)
        node_list.append(optimizer_node_2)

        output_tensor = helper.make_tensor_value_info(weight_name + "_optimizer_end", TensorProto.FLOAT, None)
        optimizer_outputs.append(weight_name + "_optimizer_end")
        onnx_model.graph.output.append(output_tensor)
    for node in node_list:
        onnx_model.graph.node.append(node)

    return onnx_model, optimizer_inputs, optimizer_outputs


def process_concat_nodes(onnx_model):
    """
    Check the ONNX Model for Concat nodes that have more than two inputs and split them as it is not supported in Stream
    """

    for i, node in enumerate(onnx_model.graph.node):
        if node.op_type in "Concat":
            n_inputs = len(node.input)
            k = 0

            attrs = node.attribute
            axis = get_attribute_ints_with_name("axis", attrs)
            # Merge the two inputs with a new concat node until only two inputs left
            while n_inputs > 2:
                concat_node = make_node(
                    "Concat",
                    [node.input[k], node.input[k + 1]],
                    [f"{node.name}_intermediary_concat_{k}"],
                    name=f"{node.name}_intermediary_concat_{k}",
                    axis=axis,
                )
                onnx_model.graph.node.insert(i, concat_node)
                node.input[0] = f"{node.name}_intermediary_concat_{k}"
                for l in range(1, len(node.input) - 1):
                    node.input[l] = node.input[l + 1]
                del node.input[-1]
                k += 1
                n_inputs = len(node.input)
    return onnx_model


def expand_softmax_grad_node(onnx_model):
    """
    Expands an ONNX SoftmaxGrad node into a sequence of ONNX operations.

    Args:
        graph: The ONNX graph containing the SoftmaxGrad node.
        node: The SoftmaxGrad node to expand.
        axis_attr: The axis attribute of the SoftmaxGrad node. Defaults to 1 if not provided.

    Returns:
        The expanded graph with the SoftmaxGrad node replaced.
    """

    for i, node in enumerate(onnx_model.graph.node):
        if any([node_type in node.op_type for node_type in ["SoftmaxGrad", "LogSoftmaxGrad"]]):
            attrs = node.attribute
            axis = get_attribute_ints_with_name("axis", attrs, default=-1)

            # Get input and output names
            Y_name = node.input[0]
            dY_name = node.input[1]
            dX_name = node.output[0]

            # Compute reduction_axes in Python
            n = len(get_onnx_tensor_type(dY_name, onnx_model).shape)
            if axis < 0:
                axis = n + axis
            reduction_axes = list(range(axis, n))
            onnx_model.graph.initializer.append(
                make_initializer("ReduceSumTensor" + node.name, reduction_axes, np.int64)
            )
            # Generate unique names for intermediate tensors
            a_name = f"{node.name}_a"
            b_name = f"{node.name}_b"
            c_name = f"{node.name}_c"
            # dy_shape = get_onnx_tensor_type(dY_name, onnx_model).shape
            # onnx_model.graph.value_info.append(make_tensor_value_info(c_name, TensorProto.FLOAT, dy_shape))
            # a = Mul(Y, dY)
            a_node = helper.make_node(
                "Mul",
                inputs=[Y_name, dY_name],
                outputs=[a_name],
            )

            # b = ReduceSum(a, reduction_axes)
            b_node = helper.make_node(
                "ReduceSum",
                inputs=[a_name, "ReduceSumTensor" + node.name],
                outputs=[b_name],
            )

            # c = Sub(dY, b)
            c_node = helper.make_node(
                "Sub",
                inputs=[b_name, dY_name],
                outputs=[c_name],
            )

            # dX = Mul(Y, c)
            dX_node = helper.make_node(
                "Mul",
                inputs=[Y_name, c_name],
                outputs=[dX_name],
            )
            node_list = [a_node, b_node, c_node, dX_node]

            # Add all nodes to the graph
            for k, new_node in enumerate(node_list):
                onnx_model.graph.node.insert(i + k, new_node)

            # Remove the original SoftmaxGrad node
            onnx_model.graph.node.remove(node)

    return onnx_model


if __name__ == "__main__":
    folder = "onnx/test"
    onnx_file = f"{folder}/simplified.onnx"
    result_file = f"{folder}/processed.onnx"

    onnx.save(process_1d_nodes(onnx.load(onnx_file)), result_file)
