from functools import wraps

from dace.sdfg.nodes import Node
from dace.sdfg.state import MultiConnectorEdge
from dace import SDFG, SDFGState
import dace.data as dt


# from paramdec package
def paramdec(dec):
    """
    Create parametrized decorator.
    >>> @paramdec
    ... def dec(func, foo=42, bar=None):
    ...     def wrapper(*func_args, **func_kwargs):
    ...         # Process foo and bar
    ...         return func(*func_args, **func_kwargs)
    ...     return wrapper
    """
    @wraps(dec)
    def wrapper(func=None, **dec_kwargs):
        if callable(func) and not dec_kwargs:
            return dec(func)
        return lambda real_func: dec(real_func, **dec_kwargs)

    return wrapper


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


def in_desc_with_name(node: Node, state: SDFGState, sdfg: SDFG,
                      name: str) -> dt.Data:
    """ Find the descriptor of the data that connects to input connector `name`.
        :param node: the node.
        :param sdfg: the sdfg.
        :param state: the state.
        :param name: the input connector name.
        :return: the descriptor of the data that connects to connector `name`.
     """
    return sdfg.arrays[in_edge_with_name(node, state, name).data.data]


def out_desc_with_name(node: Node, state: SDFGState, sdfg: SDFG,
                       name: str) -> dt.Data:
    """ Find the descriptor of the data that connects to output connector `name`.
        :param node: the node.
        :param sdfg: the sdfg.
        :param state: the state.
        :param name: the output connector name.
        :return: the descriptor of the data that connects to connector `name`.
     """
    return sdfg.arrays[out_edge_with_name(node, state, name).data.data]


def in_edge_with_name(node: Node, state: SDFGState,
                      name: str) -> MultiConnectorEdge:
    """ Find the edge that connects to input connector `name` on `node`.
        :param node: the node.
        :param state: the state.
        :param name: the input connector name.
        :return: the edge that connects to connector `name`.
     """

    in_edges = state.in_edges(node)
    cands = [edge for edge in in_edges if edge.dst_conn == name]
    if len(cands) != 1:
        raise ValueError(
            "Expected to find exactly one edge with name '{}', found {}".
                format(name, len(cands)))
    return cands[0]


def out_edge_with_name(node: Node, state: SDFGState,
                       name: str) -> MultiConnectorEdge:
    """ Find the edge that connects to output connector `name` on `node`.
        :param node: the node.
        :param state: the state.
        :param name: the output connector name.
        :return: the edge that connects to connector `name`.
     """
    out_edges = state.out_edges(node)
    cands = [edge for edge in out_edges if edge.src_conn == name]
    if len(cands) != 1:
        raise ValueError(
            "Expected to find exactly one edge with name '{}', found {}".
                format(name, len(cands)))
    return cands[0]
