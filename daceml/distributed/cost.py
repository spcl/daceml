from typing import Iterator, Set, Tuple

import dace
from dace import dtypes, nodes, memlet
from dace.sdfg import graph
from dace.transformation.dataflow.matrix_product_transpose import SDFGState
from dace.transformation.helpers import MultiConnectorEdge

# intercont bandwidth in bytes/sec
CONNECT_BANDWIDTH = {
    (dtypes.StorageType.GPU_Global, dtypes.StorageType.GPU_Global): 1e12,
    (dtypes.StorageType.GPU_Shared, dtypes.StorageType.GPU_Shared): 1e12,
    (dtypes.StorageType.GPU_Shared, dtypes.StorageType.Register): 1e12,
    (dtypes.StorageType.Register, dtypes.StorageType.GPU_Shared): 1e12,
    (dtypes.StorageType.GPU_Global, dtypes.StorageType.Register): 1e12,
    (dtypes.StorageType.Register, dtypes.StorageType.GPU_Global): 1e12,
    # GPU to CPU
}
# FLOPS
FLOPS = 1e12


def all_memlet_trees(
        sdfg) -> Iterator[Tuple[memlet.MemletTree, dace.SDFGState]]:
    seen_edges: Set[graph.MultiConnectorEdge] = set()
    for edge, parent in sdfg.all_edges_recursive():
        if isinstance(edge, graph.MultiConnectorEdge):
            if edge in seen_edges:
                continue
            state: dace.SDFGState = parent  # type: ignore
            tree = state.memlet_tree(edge)

            # trees shouldn't intersect
            assert not seen_edges.intersection(tree)
            yield tree, state
            seen_edges.update(tree)


def tree_leaf_nodes(tree: memlet.MemletTree) -> Iterator[nodes.Node]:
    scope_nodes = (nodes.EntryNode, nodes.ExitNode)

    isleaf = lambda e: not isinstance(e.dst if tree.downwards else e.src,
                                      scope_nodes)
    for edge in tree:
        if isleaf(edge):
            yield edge.dst if tree.downwards else edge.src


def sdfg_cost(sdfg: dace.SDFG) -> float:
    breakpoint()
    cost = 0

    seen_nodes: Set[nodes.Node] = set()

    # loop over all accessnodes
    for tree, state in all_memlet_trees(sdfg):
        # get the root memlet
        root_mm: dace.Memlet = tree.root().edge.data
        if root_mm.data is None:
            # FIXME this happens on initializations
            raise ValueError("No data for memlet")
        root_desc = state.parent.arrays[root_mm.data]
        root_storage = root_desc.storage

        if root_mm.volume == 0 or root_mm.volume is None:
            # FIXME change to full array volume
            raise ValueError("data volume unbounded")

        # We assume reads are fully coalesced, and thus only take the minimum
        # bandwidth over all reads
        min_bandwidth = float("inf")
        for leaf in tree_leaf_nodes(tree):
            if isinstance(leaf, nodes.AccessNode):
                leaf_storage = state.parent.arrays[leaf.data].storage
            elif isinstance(leaf, nodes.CodeNode):
                leaf_storage = dtypes.StorageType.Register
            else:
                raise ValueError("Unexpected node type")

            # get the bandwidth
            min_bandwidth = min(min_bandwidth, CONNECT_BANDWIDTH[root_storage,
                                                                 leaf_storage])
        if min_bandwidth == float("inf"):
            raise ValueError("No leaves found")

        cost += min_bandwidth * root_mm.volume * root_desc.dtype.bytes
    return cost
