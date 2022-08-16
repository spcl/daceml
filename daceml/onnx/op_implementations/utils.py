import inspect
import copy
from typing import Dict, Tuple, Optional, Callable, Union, Any
import functools
import textwrap

import dace
from dace import SDFGState, SDFG, dtypes, nodes
from dace.frontend.python.parser import DaceProgram
from dace.registry import autoregister

from daceml.onnx.nodes import onnx_op
from daceml.onnx.forward_implementation_abc import ONNXForward
from daceml.onnx.nodes.node_utils import parse_variadic_param
from daceml.util.utils import in_desc_with_name, out_desc_with_name
from daceml.transformation import constant_folding


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


def program_for_node(program,
                     sdfg: SDFG,
                     state: SDFGState,
                     node: onnx_op.ONNXOp,
                     extra_vars: Optional[Dict[str, Any]] = None) -> SDFG:
    """ Expand a function to a dace program.

        The dtypes for the arguments will be extracted by matching the parameter names to edges.

        All inputs that are not specified as parameters will be removed using
        constant_folding.remove_node_and_computation
    """
    input_names = node.schema.non_variadic_inputs()
    variadic_input_names = node.schema.variadic_inputs()

    output_names = node.schema.non_variadic_outputs()
    variadic_output_names = node.schema.variadic_outputs()

    if set(input_names).intersection(output_names):
        # this is currently the case for only one onnx op
        raise ValueError(
            "program_for_node cannot be applied on nodes of this type;"
            " '{}' are both an input and an output".format(
                set(input_names).intersection(output_names)))

    params = inspect.signature(program).parameters
    connectors_to_remove = set(input_names).difference(params)

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

    program.__name__ = node.label + "_expansion"
    result = DaceProgram(program, (), {}, False, dace.DeviceType.CPU)
    if extra_vars is not None:
        result.global_vars.update(extra_vars)

    for conn in connectors_to_remove:
        constant_folding.remove_node_and_computation(sdfg, state, node, conn)

    sdfg = result.to_sdfg()

    if node.schedule in [dtypes.ScheduleType.GPU_Default
                         ] + dtypes.GPU_SCHEDULES:
        sdfg.apply_gpu_transformations()

    return sdfg


def empty_sdfg_for_node(
    sdfg: SDFG,
    state: SDFGState,
    node: onnx_op.ONNXOp,
    add_access_nodes=True
) -> Tuple[SDFG, SDFGState, Dict[str, nodes.AccessNode], Dict[
        str, nodes.AccessNode]]:
    """ Given a node, return an SDFG that can be used as a nested SDFG expansion for that node.

        The dtypes for the arguments will be extracted by matching the parameter names to edges.
    """
    nsdfg = SDFG(node.label + "_expansion")
    nstate = nsdfg.add_state()

    input_nodes = {}
    output_nodes = {}
    for edge, is_input in node.iter_edges(state, ignore_unknown=True):
        if is_input:
            conn_name = edge.dst_conn
            nsdfg.add_datadesc(
                conn_name,
                copy.deepcopy(in_desc_with_name(node, state, sdfg, conn_name)))
            if add_access_nodes:
                input_nodes[conn_name] = nstate.add_read(conn_name)
        else:
            conn_name = edge.src_conn
            nsdfg.add_datadesc(
                conn_name,
                copy.deepcopy(out_desc_with_name(node, state, sdfg,
                                                 conn_name)))
            if add_access_nodes:
                output_nodes[conn_name] = nstate.add_write(conn_name)
        nsdfg.arrays[conn_name].transient = False

    return nsdfg, nstate, input_nodes, output_nodes


@dace.dtypes.paramdec
def python_pure_op_implementation(func, **compute: Dict[str, Callable]):
    """
    A decorator that registers an python op implementation. The name of the
    function will be the name of the op that is being replaced.

    The compute parameter enables you to compute a variable given the node and
    its inputs/outputs. This variable will be namespaced when parsing the function.

    To use this, the argument names of the functions can be either:

    * ``node``, in which case the argument will be passed the node we are expanding,
    * or, the name of any connector of the node, in which case the argument will be
      the data descriptor for that connector

    For example, the following compute argument instantiation will make
    variables ``axis`` and ``shape`` available when the function is parsed.
    

    .. highlight:: python
    .. code-block:: python

        compute=dict(
            # grabs the axis of a node
            axis=lambda node: node.axis
            # grabs the shape of the connector with name 'data'
            shape=lambda data: data.shape
        )

    :param compute: a dictionary of functions that compute variables.
    """
    @op_implementation(op=func.__name__, name="pure")
    class PureImpl(ONNXForward):
        @staticmethod
        def forward(node: onnx_op.ONNXOp, state: SDFGState,
                    sdfg: SDFG) -> Union[nodes.Node, SDFG]:
            def compute_argument_resolver(arg: str):
                if arg == "node":
                    return node
                elif arg in node.in_connectors:
                    return in_desc_with_name(node, state, sdfg, arg)
                elif arg in node.out_connectors:
                    return out_desc_with_name(node, state, sdfg, arg)
                else:
                    raise ValueError(
                        "Got unknown compute argument {}."
                        " Arguments to compute can be either 'node',"
                        " or the name of a connector of the node".format(arg))

            extra_vars = {}
            if compute is not None:
                for var_name, function in compute.items():

                    # get the names of the lambda
                    argument_names = list(
                        inspect.signature(function).parameters)

                    args = map(compute_argument_resolver, argument_names)
                    var_value = function(*args)

                    extra_vars[var_name] = var_value

            return program_for_node(func,
                                    sdfg,
                                    state,
                                    node,
                                    extra_vars=extra_vars)

    doc = \
    """
Pure implementation parsed with
:func:`~daceml.onnx.op_implementations.utils.python_pure_op_implementation`.

.. code :: python

"""
    doc += textwrap.indent(inspect.getsource(func), prefix="    ")

    PureImpl.__module__ = func.__module__
    PureImpl.__name__ = func.__name__
    PureImpl.__qualname__ = func.__qualname__
    PureImpl.__doc__ = doc

    return PureImpl
