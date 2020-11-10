import typing
import inspect

import dace
import dace.sdfg.nodes as nd
import dace.data as dt
from dace.frontend.python.parser import DaceProgram

from daceml.autodiff.backward_implementation_abc import BackwardContext, BackwardResult
from daceml.util.utils import in_desc_with_name, out_desc_with_name


def forward_in_desc_with_name(forward_node: nd.Node, context: BackwardContext,
                              name) -> dt.Data:
    """ Find the descriptor of the data that connects to input connector `name`.

        :param forward_node: the node.
        :param context: the backward context.
        :param name: the input connector name.
        :return: the descriptor of the data that connects to connector `name`.
     """
    return in_desc_with_name(forward_node, context.forward_state,
                             context.forward_sdfg, name)


def forward_out_desc_with_name(forward_node: nd.Node, context: BackwardContext,
                               name) -> dt.Data:
    """ Find the descriptor of the data that connects to output connector `name`.

        :param forward_node: the node.
        :param context: the backward context.
        :param name: the output connector name.
        :return: the descriptor of the data that connects to connector `name`.
     """
    return out_desc_with_name(forward_node, context.forward_state,
                              context.forward_sdfg, name)


def backward_program_for_node(
        program, context: BackwardContext,
        forward_node: nd.Node) -> typing.Tuple[nd.Node, BackwardResult]:
    """ Expand a function to the backward function for a node.

        The dtypes for the arguments will be extracted by matching the parameter names to edges.

        Gradient parameters should be the name of the forward parameter, appended with _grad. For these arguments the
        data descriptors will match the data descriptors of the inputs/outputs they correspond to.
    """

    input_names = set(inp.name for inp in forward_node.schema.inputs)
    output_names = set(outp.name for outp in forward_node.schema.outputs)

    if input_names.intersection(output_names):
        # this is currently the case for only one onnx op
        raise ValueError(
            "program_for_node cannot be applied on nodes of this type;"
            " '{}' is both an input and an output".format(
                next(input_names.intersection(output_names))))

    def name_without_grad_in(name, collection):
        return name[-5:] == "_grad" and name[:-5] in collection

    params = inspect.signature(program).parameters

    backward_result = BackwardResult.empty()

    inputs = {}
    outputs = {}
    for name, param in params.items():
        if name in input_names:
            inputs[name] = forward_in_desc_with_name(forward_node, context,
                                                     name)

        elif name_without_grad_in(name, input_names):
            outputs[name] = forward_in_desc_with_name(forward_node, context,
                                                      name[:-5])
            backward_result.required_grad_names[name[:-5]] = name

        elif name in output_names:
            inputs[name] = forward_out_desc_with_name(forward_node, context,
                                                      name)

        elif name_without_grad_in(name, output_names):
            inputs[name] = forward_out_desc_with_name(forward_node, context,
                                                      name[:-5])
            backward_result.given_grad_names[name[:-5]] = name

        else:
            raise ValueError(
                "'{}' was not found as an input or output for {}".format(
                    name, forward_node.schema.name))

    program.__annotations__ = {**inputs, **outputs}

    sdfg = DaceProgram(program, (), {}).to_sdfg()

    result_node = context.backward_state.add_nested_sdfg(
        sdfg, None, set(inputs), set(outputs))

    return result_node, backward_result
