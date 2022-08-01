"""
Analysis helpers for autodiff
"""
import collections

from dace import nodes, SDFG, SDFGState


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
