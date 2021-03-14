import typing
from functools import wraps

import dace
from dace import nodes as nd
from dace.sdfg.state import MultiConnectorEdge
from dace import SDFG, SDFGState
import dace.data as dt

from daceml.onnx.nodes.onnx_op import ONNXOp


def is_desc_contiguous(desc: dt.Data) -> bool:
    if type(desc) is dt.Scalar:
        return True
    elif type(desc) is dt.Array:
        contiguous_strides = [
            dt._prod(desc.shape[i + 1:]) for i in range(len(desc.shape))
        ]
        return desc.strides == contiguous_strides
    else:
        raise ValueError("Unsupported data descriptor type {}".format(
            type(desc)))


def in_desc_with_name(node: nd.Node, state: SDFGState, sdfg: SDFG,
                      name: str) -> dt.Data:
    """ Find the descriptor of the data that connects to input connector `name`.
        :param node: the node.
        :param sdfg: the sdfg.
        :param state: the state.
        :param name: the input connector name.
        :return: the descriptor of the data that connects to connector `name`.
     """
    return sdfg.arrays[in_edge_with_name(node, state, name).data.data]


def out_desc_with_name(node: nd.Node, state: SDFGState, sdfg: SDFG,
                       name: str) -> dt.Data:
    """ Find the descriptor of the data that connects to output connector `name`.
        :param node: the node.
        :param sdfg: the sdfg.
        :param state: the state.
        :param name: the output connector name.
        :return: the descriptor of the data that connects to connector `name`.
     """
    return sdfg.arrays[out_edge_with_name(node, state, name).data.data]


def in_edge_with_name(node: nd.Node, state: SDFGState,
                      name: str) -> MultiConnectorEdge:
    """ Find the edge that connects to input connector `name` on `node`.
        :param node: the node.
        :param state: the state.
        :param name: the input connector name.
        :return: the edge that connects to connector `name`.
     """

    cands = list(state.in_edges_by_connector(node, name))
    if len(cands) != 1:
        raise ValueError(
            "Expected to find exactly one edge with name '{}', found {}".
            format(name, len(cands)))
    return cands[0]


def out_edge_with_name(node: nd.Node, state: SDFGState,
                       name: str) -> MultiConnectorEdge:
    """ Find the edge that connects to output connector `name` on `node`.
        :param node: the node.
        :param state: the state.
        :param name: the output connector name.
        :return: the edge that connects to connector `name`.
     """
    cands = list(state.out_edges_by_connector(node, name))
    if len(cands) != 1:
        raise ValueError(
            "Expected to find exactly one edge with name '{}', found {}".
            format(name, len(cands)))
    return cands[0]


def find_str_not_in_set(existing: typing.Set[str],
                        target_str: typing.Optional[str]) -> str:
    """ Try to find a new str that is not in the set.

        :param existing: the existing strs.
        :param target_str: (optional) a target_str that should be used as a base for the new str.
        :return: a new str that is not in `existing`.
    """
    base_name = target_str or "temp"

    if base_name not in existing:
        return base_name

    i = 0
    while (base_name + "_" + str(i)) in existing:
        i += 1
    return base_name + "_" + str(i)


def expand_onnx_nodes(sdfg: dace.SDFG):
    """ Recursively expand all onnx library nodes in the SDFG, resulting in an SDFG that can be optimized by
        dace transformations.

        :param sdfg: the sdfg to expand nodes on.
    """
    states = list(sdfg.states())
    while len(states) > 0:
        state = states.pop()
        expanded_something = False
        for node in list(state.nodes()):  # Make sure we have a copy
            if isinstance(node, nd.NestedSDFG):
                expand_onnx_nodes(node.sdfg)
            elif isinstance(node, ONNXOp):
                impl_name = node.expand(sdfg, state)
                print(
                    "Automatically expanded library node \"{}\" with implementation \"{}\"."
                    .format(str(node), impl_name))
                # We made a copy of the original list of nodes, so we keep
                # iterating even though this list has now changed
                expanded_something = True
        if expanded_something:
            states.append(state)  # Nodes have changed. Check state again
