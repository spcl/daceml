"""
Analysis helpers for autodiff
"""
from typing import Dict, Set, Tuple, Optional
import collections

from dace import nodes, SDFG, SDFGState
from dace.sdfg import utils as sdfg_utils
from dace.transformation.passes import analysis

AccessSets = Dict[SDFGState, Tuple[Set[str], Set[str]]]


def dependency_analysis(sdfg: SDFG):
    """
    Analyze read dependencies of arrays in the SDFG.

    :param sdfg: SDFG to analyze
    :returns: A dictionary mapping array names to a list of read dependencies.
    """
    arrays = collections.defaultdict(set)
    for node, parent in sdfg.all_nodes_recursive():
        if isinstance(node, nodes.AccessNode):
            parent: SDFGState
            for edge in parent.bfs_edges(node, reverse=True):
                arrays[node.data].add(edge.data.data)
    return arrays


def inverse_reachability(sdfg: SDFG) -> Dict[SDFGState, Set[SDFGState]]:
    reachability = analysis.StateReachability().apply_pass(sdfg, {})
    inverse_reachability = collections.defaultdict(set)
    for pred, successors in reachability.items():
        for successor in successors:
            inverse_reachability[successor].add(pred)

    return inverse_reachability


def is_previously_written(sdfg: SDFG,
                          state: SDFGState,
                          node: nodes.Node,
                          array_name: str,
                          access_sets: Optional[AccessSets] = None) -> bool:
    """
    Determine whether the given array name was written before the current node.

    :param sdfg: the sdfg containing the node
    :param state: the state containing the node
    :param node: the node to check
    :param array_name: the array name to check
    :returns: True if the array was written before the node, False otherwise.
    """

    if access_sets is None:
        access_sets = analysis.AccessSets().apply_pass(sdfg, {})

    reachable = inverse_reachability(sdfg)

    # check the current state
    for subgraph in sdfg_utils.concurrent_subgraphs(state):
        if node in subgraph.nodes():
            # this is our current subgraph, check if it was written before in this subgraph
            for edge in state.bfs_edges(node, reverse=True):
                if edge.data.data == array_name:
                    return True
        else:
            # this is not our current subgraph, check the write states
            _, write_set = subgraph.read_and_write_sets()
            if array_name in write_set:
                return True

    # check other states
    for predecessor in reachable[state]:
        _, write_set = access_sets[predecessor]
        if array_name in write_set:
            return True
    return False
