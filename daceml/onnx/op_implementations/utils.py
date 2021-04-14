import inspect

import dace
from dace import SDFGState, SDFG, dtypes
from dace.frontend.python.parser import DaceProgram
from dace.registry import autoregister

from daceml.onnx.nodes import onnx_op
from daceml.onnx.nodes.node_utils import parse_variadic_param
from daceml.util.utils import in_desc_with_name, out_desc_with_name


def op_implementation(op, name):
    """ A decorator that registers an op implementation.
        It should be used on classes that extend :class:`~daceml.onnx.forward_implementation_abc.ONNXForward`.

        :param op: the ONNX name of the op to register for.
        :param name: the name of the implementation.
    """
    def dec(cls):
        if cls.__doc__ is not None:
            cls.__doc__ +=\
                """
                :Implementation name: ``"{}"``
                """.format(name)
        else:
            cls.__doc__ =\
                """
                :Implementation name: ``"{}"``
                """.format(name)

        return autoregister(cls, op=op, name=name)

    return dec


def program_for_node(program, sdfg: SDFG, state: SDFGState,
                     node: onnx_op.ONNXOp) -> SDFG:
    """ Expand a function to a dace program.

        The dtypes for the arguments will be extracted by matching the parameter names to edges.
    """
    input_names = node.schema.non_variadic_inputs()
    variadic_input_names = node.schema.variadic_inputs()

    output_names = node.schema.non_variadic_outputs()
    variadic_output_names = node.schema.variadic_outputs()

    if set(input_names).intersection(output_names):
        # this is currently the case for only one onnx op
        raise ValueError(
            "program_for_node cannot be applied on nodes of this type;"
            " '{}' is both an input and an output".format(
                next(input_names.intersection(output_names))))

    params = inspect.signature(program).parameters

    annotations = {}
    for name, param in params.items():
        if name in input_names or ("__" in name
                                   and parse_variadic_param(name)[0]
                                   in variadic_input_names):
            annotations[name] = in_desc_with_name(node, state, sdfg, name)
        elif name in output_names or ("__" in name
                                      and parse_variadic_param(name)[0]
                                      in variadic_output_names):
            annotations[name] = out_desc_with_name(node, state, sdfg, name)
        else:
            raise ValueError(
                "'{}' was not found as an input or output for {}".format(
                    name, node.schema.name))

    program.__annotations__ = annotations

    result = DaceProgram(program, (), {}, False, dace.DeviceType.CPU)
    result.name = node.label + "_expansion"

    sdfg = result.to_sdfg()

    if node.schedule in [dtypes.ScheduleType.GPU_Default
                         ] + dtypes.GPU_SCHEDULES:
        sdfg.apply_gpu_transformations()

    return sdfg
