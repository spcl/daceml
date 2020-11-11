import copy
import typing

import dace
from dace.registry import autoregister_params
from dace.sdfg.nodes import Node

import daceml.onnx as donnx
import daceml.autodiff.utils as butils
from daceml.autodiff.base_abc import BackwardImplementation, BackwardContext, BackwardResult
import daceml.util.utils as utils


@autoregister_params(node_type=donnx.ONNXSoftmax)
class ReverseSoftmax(BackwardImplementation):
    @staticmethod
    def backward(
        forward_node: Node, context: BackwardContext,
        given_gradients: typing.List[typing.Optional[str]],
        required_gradients: typing.List[typing.Optional[str]]
    ) -> typing.Tuple[typing.Union[Node, dace.SDFG], BackwardResult]:

        # elem_prod = y * dy
        # sums = elem_prod.sum(axis=dim, keepdims=True)
        # return elem_prod - y * sums

        dim = forward_node.axis

        output_shape = butils.forward_out_desc_with_name(
            forward_node, context, "output").shape
        output_dtype = butils.forward_out_desc_with_name(
            forward_node, context, "output").dtype

        sums_shape = list(copy.deepcopy(output_shape))
        sums_shape[dim] = 1

        def softmax_backward(output, output_grad, input_grad):
            prod = dace.define_local(output_shape, output_dtype)
            sums = dace.define_local(sums_shape, output_dtype)
            prod[:] = output * output_grad
            donnx.ONNXReduceSum(data=prod,
                                reduced=sums,
                                keepdims=1,
                                axes=[dim])

            input_grad[:] = output * sums
            input_grad[:] = prod - input_grad

        result = butils.backward_program_for_node(softmax_backward, context,
                                                  forward_node)

        # the backward node requires `output` as an input; connect it.
        # If more nodes need this it can be implemented more generally and added to _connect_forward_inputs

        output_edge = utils.out_edge_with_name(forward_node,
                                               context.forward_state, "output")

        # add the array so that it will be forwarded
        output_arr_name = output_edge.data.data
        assert output_arr_name not in context.backward_generator.backward_input_arrays
        data_desc = context.forward_sdfg.arrays[output_arr_name]
        context.backward_generator.backward_input_arrays[
            output_arr_name] = copy.deepcopy(data_desc)

        if context.backward_generator.separate_sdfgs:
            data_desc.transient = False
            context.backward_sdfg.add_datadesc(output_arr_name, data_desc)

        read = context.backward_state.add_read(output_arr_name)
        context.backward_state.add_edge(read, None, result[0], "output",
                                        copy.deepcopy(output_edge.data))

        return result


@autoregister_params(op="Relu", name="pure")
class PureRelu(BackwardImplementation):
    @staticmethod
    def backward(
        forward_node: Node, context: BackwardContext,
        given_gradients: typing.List[typing.Optional[str]],
        required_gradients: typing.List[typing.Optional[str]]
    ) -> typing.Tuple[Node, BackwardResult]:
        input_desc = butils.forward_in_desc_with_name(forward_node, context,
                                                      "X")

        new_sdfg = dace.SDFG("relu_backward")

        # setup arrays
        result = BackwardResult.empty()
        result.required_grad_names["X"] = butils.add_backward_desc(
            new_sdfg, context.forward_sdfg, input_desc, "X")
        result.given_grad_names["Y"] = butils.add_backward_desc(
            new_sdfg, context.forward_sdfg, input_desc, "Y")
        new_X_desc = copy.deepcopy(input_desc)
        new_X_desc.transient = False
        new_sdfg.add_datadesc("X", new_X_desc)

        # setup state
        new_state = new_sdfg.add_state()

        enum_shapes = list(enumerate(input_desc.shape))
        all_indices = ", ".join("__i{}".format(i) for i, _ in enum_shapes)

        # yapf: disable
        new_state.add_mapped_tasklet(
            "_relu_backward_",
            {
                "__i{}".format(i): "0:{}".format(s) for i, s in enum_shapes
            },
            {
                "__y_grad": dace.Memlet("Y_grad[{}]".format(all_indices)),
                "__x": dace.Memlet("X[{}]".format(all_indices))
            },
            "__x_grad = __y_grad if __x > dace.{0}(0) else dace.{0}(0)".format(
                input_desc.dtype.to_string()),
            {
                "__x_grad": dace.Memlet("X_grad[{}]".format(all_indices))
            },
            external_edges=True)
        # yapf: enable

        node = context.backward_state.add_nested_sdfg(new_sdfg, None,
                                                      {"Y_grad", "X"},
                                                      {"X_grad"})
        return node, result
