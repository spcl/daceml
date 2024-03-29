"""
Analysis helpers for autodiff
"""
from typing import Dict, Set, Tuple, Optional
import collections

import networkx as nx

from dace import nodes, SDFG, SDFGState
from dace.sdfg import utils as sdfg_utils
from dace.transformation.passes import analysis

AccessSets = Dict[SDFGState, Tuple[Set[str], Set[str]]]


def dependency_analysis(sdfg: SDFG) -> Dict[str, Set[str]]:
    """
    Analyze read dependencies of arrays in the SDFG.

    :param sdfg: SDFG to analyze
    :returns: A dictionary mapping array names to a list of read dependencies.
    """

    # FIXME can be made more efficient
    dependencies = nx.DiGraph()
    for state in sdfg.nodes():
        for node in state.data_nodes():
            for edge in state.bfs_edges(node, reverse=True):
                dependencies.add_edge(node.data, edge.data.data)

    dependencies = nx.transitive_closure(dependencies)
    result = {}
    for array in dependencies:
        result[array] = {nbr for nbr in dependencies.neighbors(array)}
    return result


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
