import onnx
from onnx import helper
import numpy as np
import copy
from onnx.helper import (
    make_model, make_node, make_graph, set_model_props,
    make_tensor_value_info)
from onnx import numpy_helper, TensorProto

from zigzag.parser.onnx.utils import get_onnx_tensor_type, get_attribute_ints_with_name

def shape2tuple(shape):
    return tuple(getattr(d, 'dim_value', 0) for d in shape.type.tensor_type.shape.dim)

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
    for i in range(out_h) :
        temp_indices = []
        h_start = i * strides[0]
        for ki in range(kh) :
            h = h_start + ki * dilations[0]
            temp_indices.append(h)
        h_indices.append(temp_indices)

    for j in range(out_w) :
        temp_indices = []
        w_start = i * strides[1]
        for kj in range(kw) :
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
    forward_inputs = []
    backward_inputs = []
    for input_tensor in onnx_model.graph.input :
        if "grad" in input_tensor.name :
            backward_inputs.append(input_tensor.name)
        else :
            forward_inputs.append(input_tensor.name)

    forward_outputs = []
    backward_outputs = []
    for output_tensor in onnx_model.graph.output :
        if "grad" in output_tensor.name :
            backward_outputs.append(output_tensor.name)
        else :
            forward_outputs.append(output_tensor.name)

    # Find the index of the first LossGrad node
    for i, op_node in enumerate(onnx_model.graph.node):
        if "LossGrad" in op_node.op_type :
            sep_index = i
            break
    
    for i, op_node in enumerate(onnx_model.graph.node):
        if i < sep_index:
            for j, op_node2 in enumerate(onnx_model.graph.node):
                if j >= sep_index:
                    for output_name in op_node.output:
                        if output_name in op_node2.input :
                            if output_name not in forward_outputs:
                                forward_outputs.append(output_name)
                            if output_name not in backward_inputs:
                                backward_inputs.append(output_name)

    for i, op_node in enumerate(onnx_model.graph.node):
        if i >= sep_index:
            for input_tensor in onnx_model.graph.input:
                if input_tensor.name in op_node.input and input_tensor.name not in backward_inputs:
                    backward_inputs.append(input_tensor.name)

    return forward_inputs, backward_inputs, forward_outputs, backward_outputs

def process_PoolGrad(onnx_model):
    """
    Process PoolGrad nodes in the ONNX model to update their domain to com.microsoft as they are not processed by default.
    """
    for i, op_node in enumerate(onnx_model.graph.node): 
        if op_node.op_type == "MaxPoolGrad" or op_node.op_type == "AveragePoolGrad" :
            op_node.domain = "com.microsoft"
    return onnx_model

def process_1D_nodes(onnx_model):
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
        # print(op_node)
        input_names = op_node.input
        output_names = op_node.output

        # If the node is a Constant node, remove it from the graph
        if op_node.op_type == "Constant" :
            onnx_model.graph.node.remove(op_node)
            return
        
        # Remove the input and output tensors from the graph
        for input_tensor in onnx_model.graph.input :
            if input_tensor.name in input_names and input_tensor.name not in ["lazy_reset_grad"] :
                onnx_model.graph.input.remove(input_tensor)
        for output_tensor in onnx_model.graph.output :
            if output_tensor.name in output_names :
                onnx_model.graph.output.remove(output_tensor)

        for i, op_node2 in enumerate(onnx_model.graph.node) :
            if any(output_name in op_node2.input for output_name in output_names) :
                remove_linked_nodes(op_node2, onnx_model)

        # Remove the node from the graph 
        if op_node in onnx_model.graph.node :
            onnx_model.graph.node.remove(op_node)

    for i, op_node in enumerate(onnx_model.graph.node) :
        # Replace SoftmaxCrossEntropyLoss By Identity
        if op_node.op_type == "SoftmaxCrossEntropyLoss" :
            inputs = op_node.input
            outputs = op_node.output
            node_transpose = make_node("Identity", [inputs[0]], [outputs[1]], name=op_node.name + "_identity")

            onnx_model.graph.node.insert(i, node_transpose)
            onnx_model.graph.node.remove(op_node)
            for output_tensor in onnx_model.graph.output :
                if output_tensor.name == outputs[0] :
                    onnx_model.graph.output.remove(output_tensor)

    for i, op_node in enumerate(onnx_model.graph.node) :
        if op_node.op_type not in ["Constant"] :
            inputs = op_node.input
            outputs = op_node.output
            
            for j, input_name in enumerate(inputs) :
                input_tensor = get_onnx_tensor_type(input_name, onnx_model)
                if input_tensor is not None and len(input_tensor.shape) == 1 :
                    input_nodes_1D.append([input_name, i, j, op_node])
            for j, output_name in enumerate(outputs) :
                output_tensor = get_onnx_tensor_type(output_name, onnx_model)
                if output_tensor is not None and len(output_tensor.shape) == 1 :
                    output_nodes_1D.append([output_name, i, j, op_node])

    for input_name, input_node_index, input_index, input_node in input_nodes_1D :
        for output_name, output_node_index, output_index, output_node in output_nodes_1D :
            if input_name == output_name :
                output_node.output[output_index] = "output_" + input_name
                remove_linked_nodes(input_node, onnx_model)
    
    return onnx_model
def process_convGrad(onnx_model) :
    """
    Process ConvGrad nodes in the ONNX model to convert them into ConvTranspose nodes and reshape the inputs accordingly.
    This is necessary because ConvGrad nodes use multiple output nodes which make them not easily implementable in Stream.
    """

    updated_list = []
    for i, op_node in enumerate(onnx_model.graph.node) :

        if op_node.op_type == "ConvGrad" :
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
            input_grad_attr = {"kernel_shape": kernel_shape,"strides": strides, "dilations": dilations, "group": group_size, "pads": padding}
        
            # Create relevant nodes
            node_list = []
            input_list  = []
            initializer_list = []

            output_name = op_node.output[0]
            if len(op_node.output[0]) == 0 :
                output_name = "input_grad"
            node_input_grad = make_node("ConvTranspose", [op_node.input[0], op_node.input[2]], [output_name], **input_grad_attr)
            node_list.append(node_input_grad)
            

            initializer_list.append(make_initializer("shape_axis" + op_node.name, [input_grad_shape[0], input_grad_shape[1], input_grad_shape[2]*input_grad_shape[3]], np.int64))
            node_reshape = make_node("Reshape", [op_node.input[0], "shape_axis" + op_node.name], ["shape" + op_node.name])
            node_list.append(node_reshape)

            sliding_window_h = (input_shape[2] + 2 * padding[0] - dilations[0] * (kernel_shape[0] - 1) - 1) // strides[0] + 1
            sliding_window_w = (input_shape[3] + 2 * padding[1] - dilations[1] * (kernel_shape[1] - 1) - 1) // strides[1] + 1
            transposed_shape = [input_shape[0], sliding_window_h*sliding_window_w, input_shape[1]*kernel_shape[0]*kernel_shape[1]]
            if op_node.input[1] != onnx_model.graph.input[0].name :
                # Add padding if necessary
                if max(padding) > 0:
                    pads = np.array([0, 0, padding[0], padding[1], 0, 0, padding[2], padding[3]])
                    padding_value_node = helper.make_node("Constant", inputs=[], outputs=["paddings_" + op_node.name], value=numpy_helper.from_array(pads))
                    pad_node = helper.make_node('Pad', inputs=[op_node.input[1], "paddings_" + op_node.name], outputs=['padded_input' + op_node.name], mode='constant')

                    node_list.append(padding_value_node)
                    node_list.append(pad_node)
                    input_name = 'padded_input' + op_node.name
                else:
                    input_name = op_node.input[1]
                
                h_indices, w_indices = get_sliding_window_shape(input_shape, kernel_shape, strides, dilations, padding)
                # Create Constant Nodes for h and w indices
                h_indices_constant_node = helper.make_node("Constant", inputs=[], outputs=["h_indices_" + op_node.name], value=numpy_helper.from_array(h_indices))
                w_indices_constant_node = helper.make_node("Constant", inputs=[], outputs=["w_indices_" + op_node.name], value=numpy_helper.from_array(w_indices))
                node_list.append(h_indices_constant_node)
                node_list.append(w_indices_constant_node)

                # Unfold Operation
                gather_node_h = make_node("Gather", [input_name, "h_indices_" + op_node.name], ["h_gathered"+ op_node.name], axis=2)
                gather_node_w = make_node("Gather", ["h_gathered"+ op_node.name, "w_indices_" + op_node.name], ["w_h_gathered"+ op_node.name], axis=4)
                transpose_node_gather = make_node("Transpose", ["w_h_gathered"+ op_node.name], ["gathered_c"+ op_node.name], perm=[0, 3, 5, 4, 1, 2])

                node_list.append(gather_node_h)
                node_list.append(gather_node_w)
                node_list.append(transpose_node_gather)

                # Create reshape node
                initializer_list.append(make_initializer('shape_axis2' + op_node.name, transposed_shape, np.int64))
                reshape_node = helper.make_node(
                    'Reshape',
                    inputs=['gathered_c'+ op_node.name, 'shape_axis2'+ op_node.name],
                    outputs=['transpose1_output1_'+ op_node.name]
                )
                node_list.append(reshape_node)
            else :
                transposed_input = make_tensor_value_info("transpose1_output1_" + op_node.name, TensorProto.FLOAT, transposed_shape)
                input_list.append(transposed_input)
            
            node_matmul = make_node("MatMul", ["shape" + op_node.name, "transpose1_output1_" + op_node.name], ["matmul_output_" + op_node.name], name="MatMul"+op_node.name)
            node_list.append(node_matmul)
            initializer_list.append(make_initializer("ReduceSumTensor1" + op_node.name, [0], np.int64))
            # node_reduce_sum_axis1 = make_node("Constant", [], ["ReduceSumTensor1" + op_node.name], value=numpy_helper.from_array(np.array([0,], dtype=np.int64)))
            node_sum = make_node("ReduceSum", ["matmul_output_" + op_node.name, "ReduceSumTensor1" + op_node.name], ["batch_sum" + op_node.name], keepdims=0)
            # node_list.append(node_reduce_sum_axis1)
            node_list.append(node_sum)

            node_reshape_axis3 = make_node("Constant", [], ["shape_axis3" + op_node.name], value=numpy_helper.from_array(np.array([weight_shape[0], weight_shape[1], weight_shape[2], weight_shape[3]], dtype=np.int64)))
            node_reshape2 = make_node("Reshape", ["batch_sum" + op_node.name, "shape_axis3" + op_node.name], [op_node.output[1]])
            node_list.append(node_reshape_axis3)
            node_list.append(node_reshape2)

            if len(op_node.output[2]) > 0 :
                # Create Initiliazer for the input of the reduce sum
                node_reduce_sum_axis2 = make_node("Constant", [], ["ReduceSumTensor2" + op_node.name], value=numpy_helper.from_array(np.array([1, 2, 3], dtype=np.int64)))
                node_transpose4 = make_node("Transpose", [op_node.input[0]], ["transposed_bias_grad_" + op_node.name], **transpose_attr)
                node_bias_grad = make_node("ReduceSum", ["transposed_bias_grad_" + op_node.name, "ReduceSumTensor2" + op_node.name], [op_node.output[2]], keepdims=0)
                node_list.append(node_reduce_sum_axis2)
                node_list.append(node_transpose4)
                node_list.append(node_bias_grad)

            updated_list.append([op_node, i, node_list, input_list, initializer_list])

    n = 0
    for remove_node, index, append_list, new_input_list, initializer_list in updated_list :
        for new_input in new_input_list :
            onnx_model.graph.input.append(new_input)
        for new_initializer in initializer_list :
            onnx_model.graph.initializer.append(new_initializer)
        for j, op_node in enumerate(append_list) :

            onnx_model.graph.node.insert(index + n, op_node)
            n += 1
        onnx_model.graph.node.remove(remove_node)
        n -= 1

    return onnx_model

def add_optimizer(onnx_model, optimizer_name="Adam", learning_rate=0.001, weight_decay= 0, beta1=0.9, beta2=0.999, epsilon=1e-8):
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

    # Lists to store the name of all the inputs and outputs required to run only the optimizer
    optimizer_inputs = []
    optimizer_outputs = []
    # Get all gradients accumulations outputs of the model
    grad_accumulation_names = []
    for input_tensor in onnx_model.graph.input :
        if "grad.accumulation.buffer" in input_tensor.name :
            grad_accumulation_names.append(input_tensor.name)
            # If the input tensor is a gradient accumulation output, we add it to the input list
            optimizer_inputs.append(input_tensor.name)    
    optimizer_inputs.append("lazy_reset_grad")
    for output_tensor in onnx_model.graph.output :
        if "grad.accumulation" in output_tensor.name:
            optimizer_outputs.append(output_tensor.name)

    
    grad_accumulation_names2 = []
    for node in onnx_model.graph.node :
        if node.op_type == "InPlaceAccumulatorV2" :
            for input in node.input :
                if "grad.accumulation.buffer" in input :
                    grad_accumulation_names2.append(node.input[1])

    gradient_weights = []
    for i, input_tensor in enumerate(onnx_model.graph.input) :
        for element, element2 in zip(grad_accumulation_names, grad_accumulation_names2) :
            if input_tensor.name in element and "buffer" not in input_tensor.name:
                # If the input tensor is a gradient accumulation output, we add it to the input list
                gradient_weights.append((input_tensor.name, element2))

    for input_name, gradient_buffer_name in gradient_weights :
        # Create the optimizer nodes (only SGD and Adam are supported for now)

        if weight_decay != 0 :
            # Create a weight decay node
            weight_decay_node = make_node("Mul", [input_name, "weight_decay"], [input_name + "_weight_decay"], name=f"{optimizer_name}_WeightDecay_{input_name}")
            g_node = make_node("Add", [gradient_buffer_name, input_name + "_weight_decay"], [gradient_buffer_name + "_optimizer_g"], name=f"{optimizer_name}_WeightDecay_Add_{input_name}")
            node_list.append(weight_decay_node)
            node_list.append(g_node)
            gradient_buffer_name = gradient_buffer_name + "_optimizer_g"
        # For now we do not consider maximize and amsgrad (as detailed here https://docs.pytorch.org/docs/stable/generated/torch.optim.Adam.html)
        # Without loss of generality for Stream, we assume that the constant computation (1-Beta2) Beta**t is done and is Beta2 (same for Beta1)
        if optimizer_name == "Adam" :
            final_gradient_name = input_name + "_optimizer"
            # Add inputs for the previous optimizers states
            weight_shape = get_onnx_tensor_type(input_name, onnx_model).shape
            onnx_model.graph.input.append(make_tensor_value_info(input_name + "_optimizer_first_moment", TensorProto.FLOAT, weight_shape))
            onnx_model.graph.input.append(make_tensor_value_info(input_name + "_optimizer_second_moment", TensorProto.FLOAT, weight_shape))

            optimizer_inputs.extend([input_name, gradient_buffer_name, input_name + "_optimizer_first_moment", input_name + "_optimizer_second_moment"])
            # First Moment Computation
            mul_first_moment_node = make_node("Mul", [input_name + "_optimizer_first_moment", "beta1"], [input_name + "_optimizer11"], name=f"{optimizer_name}_Optimizer_{input_name}_MulFirstMoment")
            mul_first_moment_node2 = make_node("Mul", [gradient_buffer_name, "beta1"], [input_name + "_optimizer10"], name=f"{optimizer_name}_Optimizer_{input_name}_MulFirstMoment2")
            add_first_moment_node = make_node("Add", [input_name + "_optimizer10", input_name + "_optimizer11"], [input_name + "_optimizer9"], name=f"{optimizer_name}_Optimizer_{input_name}_First_Moment")
            mean_first_moment_node = make_node("Div", [input_name + "_optimizer9", "beta1"], [input_name + "_optimizer1"], name=f"{optimizer_name}_Optimizer_{input_name}_MeanFirst_Moment")
            # Second Moment Computation 
            mul_node3 = make_node("Mul", [gradient_buffer_name, gradient_buffer_name], [input_name + "_optimizer8"], name=f"{optimizer_name}_Optimizer_{input_name}_Mul3")
            mul_node2 = make_node("Mul", [input_name + "_optimizer8", "beta2"], [input_name + "_optimizer7"], name=f"{optimizer_name}_Optimizer_{input_name}_Mul2")
            mul_node1 = make_node("Mul", [input_name + "_optimizer_second_moment", "beta2"], [input_name + "_optimizer6"], name=f"{optimizer_name}_Optimizer_{input_name}_Mul1")
            add_second_moment_node = make_node("Add", [input_name + "_optimizer6", input_name + "_optimizer7"], [input_name + "_optimizer5"], name=f"{optimizer_name}_Optimizer_{input_name}_Second_Moment")
            mean_second_moment_node = make_node("Div", [input_name + "_optimizer5", "beta2"], [input_name + "_optimizer4"], name=f"{optimizer_name}_Optimizer_{input_name}_MeanSecond_Moment")
            second_moment_node_1 = make_node("Sqrt", [input_name + "_optimizer4"], [input_name + "_optimizer3"], name=f"{optimizer_name}_Optimizer_{input_name}_second_moment1")
            second_moment_node = make_node("Add", [input_name + "_optimizer3", "epsilon"], [input_name + "_optimizer2"], name=f"{optimizer_name}_Optimizer_{input_name}_second_moment")
            final_optimizer_node = make_node("Div", [input_name + "_optimizer1", input_name + "_optimizer2"], [final_gradient_name], name=f"{optimizer_name}_Optimizer_{input_name}")
            node_list.extend([mul_first_moment_node, mul_first_moment_node2, add_first_moment_node, mean_first_moment_node,
                              mul_node3, mul_node2, mul_node1, add_second_moment_node, mean_second_moment_node,
                              second_moment_node_1, second_moment_node, final_optimizer_node])
        elif optimizer_name == "SGD" :
            optimizer_inputs.append(input_name, gradient_buffer_name)
            final_gradient_name = gradient_buffer_name
        optimizer_node_1 = make_node("Mul", [final_gradient_name, "learning_rate"], [input_name + "_optimizer0"], name=f"{optimizer_name}_Optimizer_{input_name}")
        optimizer_node_2 = make_node("Sub", [input_name, input_name + "_optimizer0"], [input_name + "_optimizer_end"], name=f"{optimizer_name}_Update_{input_name}")
        node_list.append(optimizer_node_1)
        node_list.append(optimizer_node_2)

        output_tensor = helper.make_tensor_value_info(input_name + "_optimizer_end", TensorProto.FLOAT, None)
        optimizer_outputs.append(input_name + "_optimizer_end")
        onnx_model.graph.output.append(output_tensor)
    for node in node_list :
        onnx_model.graph.node.append(node)
    
    return onnx_model, optimizer_inputs, optimizer_outputs

if __name__ == "__main__":
    folder = "onnx/test"
    onnx_file = f"{folder}/simplified.onnx"
    result_file = f"{folder}/processed.onnx"

    onnx.save(process_1D_nodes(onnx.load(onnx_file)), result_file)