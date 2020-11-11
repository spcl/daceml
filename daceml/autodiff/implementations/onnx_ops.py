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

        output_shape = butils.forward_out_desc_with_name(forward_node, context,
                                                  "output").shape
        output_dtype = butils.forward_out_desc_with_name(forward_node, context,
                                                  "output").dtype

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

        output_edge = utils.out_edge_with_name(forward_node, context.forward_state,
                                               "output")

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
