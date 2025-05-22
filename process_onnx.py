import onnx
from onnx import helper
import numpy as np

from onnx.helper import (
    make_model, make_node, make_graph, set_model_props,
    make_tensor_value_info)
from onnx import numpy_helper, TensorProto

from zigzag.parser.onnx.utils import get_onnx_tensor_type, get_attribute_ints_with_name

def shape2tuple(shape):
    return tuple(getattr(d, 'dim_value', 0) for d in shape.dim)

def process_convGrad(onnx_model, name) :

    node_list = []

    for node in onnx_model.graph.node :
        if node.op_type == "ConvGrad" :

            # get attributes
            attrs = node.attribute
            kernel_shape: list[int] = get_attribute_ints_with_name("kernel_shape", attrs, default=None)  # type:ignore
            strides: list[int] = get_attribute_ints_with_name("strides", attrs, default=[1, 1])  # type:ignore
            dilations: list[int] = get_attribute_ints_with_name("dilations", attrs, default=[1, 1])  # type:ignore
            group_size: int = get_attribute_ints_with_name("group", attrs, default=1)  # type:ignore
            padding: list[int] = get_attribute_ints_with_name("pads", attrs, default=[0, 0, 0, 0])  # type:ignore
            print(kernel_shape, strides, dilations, group_size, padding, node.name)
            weight_grad_shape = get_onnx_tensor_type(node.input[0], onnx_model).shape[2:]
            transpose_attr = {"perm": [1, 0, 2, 3]}
            weight_grad_attr = {"kernel_shape": weight_grad_shape, "dilations": strides, "strides": dilations, "group_size": group_size}
            input_grad_attr = {"kernel_shape": kernel_shape,"strides": strides, "dilations": dilations, "group_size": group_size}
            # Create relevant nodes
            
            node_input_grad = make_node("ConvTranspose", [node.input[0], node.input[2]], [node.output[0]], **input_grad_attr)
            node_transpose1 = make_node("Transpose", [node.input[1]], ["transpose1_output1_" + node.name], **transpose_attr) # input
            node_transpose2 = make_node("Transpose", [node.input[0]], ["transpose1_output2_" + node.name], **transpose_attr) # grad
            node_weight_grad = make_node("Conv", ["transpose1_output1_" + node.name, "transpose1_output2_" + node.name], ["grad_weight_conv_" + node.name], **weight_grad_attr)
            node_transpose3 = make_node("Transpose", ["grad_weight_conv_" + node.name], [node.output[1]], **transpose_attr)

            node_transpose4 = make_node("Transpose", [node.input[0]], ["transposed_bias_grad_" + node.name], **transpose_attr)
            node_bias_grad = make_node("ReduceSum", ["transposed_bias_grad_" + node.name], [node.output[2]], axes=[1, 2, 3])
            node_list.append(node_weight_grad)
            node_list.append(node_bias_grad)
            node_list.append(node_input_grad)
            node_list.append(node_transpose1)
            node_list.append(node_transpose2)
            node_list.append(node_transpose3)
            node_list.append(node_transpose4)
        else :
            node_list.append(node)
    
    graph = make_graph(node_list, name, onnx_model.graph.input, onnx_model.graph.output, initializer=onnx_model.graph.initializer)
    model = make_model(graph)
    model.ir_version = onnx_model.ir_version
    model.producer_name = onnx_model.producer_name
    model.producer_version = onnx_model.producer_version
    model.domain = onnx_model.domain
    model.model_version = onnx_model.model_version
    model.doc_string = onnx_model.doc_string
    if len(onnx_model.metadata_props) > 0:  # pragma: no cover
        values = {p.key: p.value for p in onnx_model.metadata_props}
        set_model_props(model, values)

    del model.opset_import[:]  # pylint: disable=E1101
    for oimp in onnx_model.opset_import:
        op_set = model.opset_import.add()  # pylint: disable=E1101
        op_set.domain = oimp.domain
        op_set.version = 22 if oimp.domain == '' else oimp.version
    return model

def process_convGrad2(onnx_model) :

    updated_list = []

    for i, op_node in enumerate(onnx_model.graph.node) :
        if op_node.op_type == "ConvGrad" :

            # get attributes
            attrs = op_node.attribute
            kernel_shape: list[int] = get_attribute_ints_with_name("kernel_shape", attrs, default=None)  # type:ignore
            strides: list[int] = get_attribute_ints_with_name("strides", attrs, default=[1, 1])  # type:ignore
            dilations: list[int] = get_attribute_ints_with_name("dilations", attrs, default=[1, 1])  # type:ignore
            group_size: int = get_attribute_ints_with_name("group", attrs, default=1)  # type:ignore
            padding: list[int] = get_attribute_ints_with_name("pads", attrs, default=[0, 0, 0, 0])  # type:ignore
            
            weight_grad_shape = get_onnx_tensor_type(op_node.input[0], onnx_model).shape[2:]
            transpose_attr = {"perm": [1, 0, 2, 3]}
            weight_grad_attr = {"kernel_shape": weight_grad_shape, "dilations": strides, "strides": dilations, "group": group_size}
            input_grad_attr = {"kernel_shape": kernel_shape,"strides": strides, "dilations": dilations, "group": group_size}
           
            # Create relevant nodes
            node_list = []
            input_list  = []
            initializer_list = []

            output_name = op_node.output[0]
            if len(op_node.output[0]) == 0 :
                output_name = "input_grad"
            node_input_grad = make_node("ConvTranspose", [op_node.input[0], op_node.input[2]], [output_name], **input_grad_attr)
            node_list.append(node_input_grad)
            
            node_transpose2 = make_node("Transpose", [op_node.input[0]], ["transpose1_output2_" + op_node.name], **transpose_attr) # grad
            node_list.append(node_transpose2)
            
            if op_node.input[1] != onnx_model.graph.input[0].name :
                node_transpose1 = make_node("Transpose", [op_node.input[1]], ["transpose1_output1_" + op_node.name], **transpose_attr) # input
                node_list.append(node_transpose1)
            else :
                input_shape = shape2tuple(onnx_model.graph.input[0].type.tensor_type.shape)
                transposed_shape = [input_shape[1], input_shape[0], input_shape[2], input_shape[3]]
                transposed_input = make_tensor_value_info("transpose1_output1_" + op_node.name, TensorProto.FLOAT, transposed_shape)
                input_list.append(transposed_input)
            node_weight_grad = make_node("Conv", ["transpose1_output1_" + op_node.name, "transpose1_output2_" + op_node.name], ["grad_weight_conv_" + op_node.name], **weight_grad_attr)
            node_list.append(node_weight_grad)
            
            
            node_transpose3 = make_node("Transpose", ["grad_weight_conv_" + op_node.name], [op_node.output[1]], **transpose_attr)
            node_list.append(node_transpose3)

            if "bias" in op_node.output[2] :
                # Create Initiliazer for the input of the reduce sum
                axis = np.array([1, 2, 3], dtype=np.int64)
                onnx_axis = numpy_helper.from_array(axis, name="ReduceSumTensor" + op_node.name)
                initializer_list.append(onnx_axis)

                node_transpose4 = make_node("Transpose", [op_node.input[0]], ["transposed_bias_grad_" + op_node.name], **transpose_attr)
                node_list.append(node_transpose4)
                node_bias_grad = make_node("ReduceSum", ["transposed_bias_grad_" + op_node.name, "ReduceSumTensor" + op_node.name], [op_node.output[2]], keepdims=0)
                node_list.append(node_bias_grad)


            # updated_list.append([op_node, i, [node_input_grad, node_transpose1, node_transpose2, node_weight_grad, node_transpose3, node_transpose4, node_bias_grad], reduce_sum_input])
            updated_list.append([op_node, i, node_list, input_list, initializer_list])

    for remove_node, index, append_list, new_input_list, initializer_list in updated_list :
        for new_input in new_input_list :
            onnx_model.graph.input.append(new_input)
        for new_initializer in initializer_list :
            onnx_model.graph.initializer.append(new_initializer)
        for j, op_node in enumerate(append_list) :
            onnx_model.graph.node.insert(index+j, op_node)
        onnx_model.graph.node.remove(remove_node)
        
    return onnx_model

    