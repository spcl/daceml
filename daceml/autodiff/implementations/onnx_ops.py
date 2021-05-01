import copy
import itertools
from typing import List, Optional, Tuple, Dict, Union

import dace
from dace.frontend.common import einsum
from dace.registry import autoregister_params
import dace.sdfg.nodes as nd

import daceml.onnx as donnx
from daceml.onnx.op_implementations import pure_implementations
import daceml.autodiff.utils as butils
from daceml.autodiff.base_abc import BackwardImplementation, BackwardContext, BackwardResult


def reverse_einsum_wrt_input(forward_node: donnx.ONNXEinsum,
                             input_name: str) -> Tuple[List[str], str]:
    """ Produce the einsum string that computes the grad of ``forward_node`` w.r.t. ``input_name``.

       :Note:
            There is an edge case we currently don't handle (can be implemented though). Something like ``'ii->i'``
            would become ``'i->ii'``. This is invalid because ``i`` is repeated in the output.

        :param forward_node: the einsum node to reverse.
        :param input_name: the connector on the forward node the produce the gradient computation for.
        :return: the list of forward node connectors required as inputs, and the einsum string. The first parameter of
                 the produced einsum string is implicitly the grad of ``Output``.
    """

    _, input_idx = donnx.parse_variadic_param(input_name)
    parser = einsum.EinsumParser(forward_node.equation)

    backward_input_expressions = [
        parser.output
    ] + parser.inputs[:input_idx] + parser.inputs[input_idx + 1:]
    backward_input_arrays = [
        f"Inputs__{i}" for i in itertools.chain(
            range(input_idx), range(input_idx + 1, len(parser.inputs)))
    ]

    einsum_str = f"{','.join(backward_input_expressions)}->{parser.inputs[input_idx]}"
    return backward_input_arrays, einsum_str


@autoregister_params(op="Einsum", name="default")
class DefaultEinsumBackward(BackwardImplementation):
    """ The symbolic autodiff can automatically derive matmuls, but the produced maps are more difficult to optimize.
    """
    @staticmethod
    def backward_can_be_applied(node: nd.Node, state: dace.SDFGState,
                                sdfg: dace.SDFG) -> bool:
        return pure_implementations.PureEinsum.forward_can_be_applied(
            node, state, sdfg)

    @staticmethod
    def backward(
        forward_node: nd.Node, context: BackwardContext,
        given_gradients: List[Optional[str]],
        required_gradients: List[Optional[str]]
    ) -> Tuple[nd.Node, BackwardResult]:

        nsdfg = dace.SDFG(forward_node.label + "_backward")
        nstate = nsdfg.add_state()

        # setup arrays
        output_desc = butils.forward_out_desc_with_name(
            forward_node, context, "Output")
        result = BackwardResult.empty()
        result.given_grad_names["Output"] = butils.add_backward_desc(
            nsdfg, context.forward_sdfg, output_desc, "Output")
        access_output_grad = nstate.add_read(result.given_grad_names["Output"])

        def create_access_node(connector: str) -> nd.AccessNode:
            nsdfg.add_datadesc(
                connector,
                butils.forward_in_desc_with_name(forward_node, context,
                                                 connector))
            return nstate.add_read(connector)

        # the forward inputs we will require
        # maps the connector name to the accessnode
        required_forward_inputs: Dict[str, nd.AccessNode] = {}

        for input_name in required_gradients:
            # we add an einsum for each required gradient
            forward_inputs, einsum_str = reverse_einsum_wrt_input(
                forward_node, input_name)

            einsum_node = donnx.ONNXEinsum(input_name + "_backward",
                                           equation=einsum_str)
            nstate.add_node(einsum_node)

            # the first input is always the output grad
            einsum_node.add_in_connector(f"Inputs__0")
            nstate.add_edge(
                access_output_grad, None, einsum_node, "Inputs__0",
                nsdfg.make_array_memlet(result.given_grad_names["Output"]))

            # add the other inputs from forward that we need
            for i, forward_input in enumerate(forward_inputs):
                connector = f"Inputs__{i + 1}"
                einsum_node.add_in_connector(connector)
                if forward_input not in required_forward_inputs:
                    required_forward_inputs[
                        forward_input] = create_access_node(forward_input)

                nstate.add_edge(required_forward_inputs[forward_input], None,
                                einsum_node, connector,
                                nsdfg.make_array_memlet(forward_input))

            # write out the gradient
            butils.forward_in_desc_with_name(forward_node, context, input_name)
            result.required_grad_names[
                input_name] = butils.add_backward_desc_for_connector(
                    nsdfg, forward_node, context, input_name, True)
            memlet = nsdfg.make_array_memlet(
                result.required_grad_names[input_name])
            nstate.add_edge(
                einsum_node, "Output",
                nstate.add_write(result.required_grad_names[input_name]), None,
                memlet)

        result_node = context.backward_state.add_nested_sdfg(
            nsdfg, None,
            set(result.given_grad_names.values()).union(
                required_forward_inputs),
            set(result.required_grad_names.values()))

        return result_node, result


@autoregister_params(op="Softmax", name="default")
class DefaultSoftmaxBackward(BackwardImplementation):
    @staticmethod
    def backward(
        forward_node: nd.Node, context: BackwardContext,
        given_gradients: List[Optional[str]],
        required_gradients: List[Optional[str]]
    ) -> Tuple[Union[nd.Node, dace.SDFG], BackwardResult]:

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
            donnx.ONNXMul(A=output, B=output_grad, C=prod)
            donnx.ONNXReduceSum(data=prod,
                                reduced=sums,
                                keepdims=1,
                                axes=[dim])

            donnx.ONNXMul(A=output, B=sums, C=input_grad)
            # let's not use ONNXSub here; not sure how this inplace op is handled by ORT...
            input_grad[:] = prod - input_grad

        result_node, result = butils.backward_program_for_node(
            softmax_backward, context, forward_node)

        butils.connect_output_from_forward(forward_node, result_node, context,
                                           "output")

        return result_node, result


@autoregister_params(op="LogSoftmax", name="default")
class DefaultLogSoftmaxBackward(BackwardImplementation):
    @staticmethod
    def backward(
        forward_node: nd.Node, context: BackwardContext,
        given_gradients: List[Optional[str]],
        required_gradients: List[Optional[str]]
    ) -> Tuple[nd.Node, BackwardResult]:

        dim = forward_node.axis
        output_shape = butils.forward_out_desc_with_name(
            forward_node, context, "output").shape
        output_dtype = butils.forward_out_desc_with_name(
            forward_node, context, "output").dtype

        sums_shape = list(copy.deepcopy(output_shape))
        sums_shape[dim] = 1

        def logsoftmax_backward(output, output_grad, input_grad):
            exp_output = dace.define_local(output_shape, output_dtype)
            donnx.ONNXExp(input=output, output=exp_output)

            grad_output_sum = dace.define_local(sums_shape, output_dtype)
            donnx.ONNXReduceSum(data=output_grad,
                                reduced=grad_output_sum,
                                keepdims=1,
                                axes=[dim])
            # let's not use ONNXMul here; not sure how this inplace op is handled by ORT...
            exp_output[:] = exp_output * grad_output_sum
            donnx.ONNXSub(A=output_grad, B=exp_output, C=input_grad)

        result_node, result = butils.backward_program_for_node(
            logsoftmax_backward, context, forward_node)

        butils.connect_output_from_forward(forward_node, result_node, context,
                                           "output")
        return result_node, result


@autoregister_params(op="Relu", name="pure")
class PureReluBackward(BackwardImplementation):
    @staticmethod
    def backward(
        forward_node: nd.Node, context: BackwardContext,
        given_gradients: List[Optional[str]],
        required_gradients: List[Optional[str]]
    ) -> Tuple[nd.Node, BackwardResult]:
        input_desc = butils.forward_in_desc_with_name(forward_node, context,
                                                      "X")

        new_sdfg = dace.SDFG(forward_node.label + "_backward")

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
