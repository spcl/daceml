import dace
from dace import nodes, dtypes


def to_mixed_precision(sdfg: dace.SDFG,
                       acctype=dace.float32,
                       dtype=dace.float16):
    """
    Convert an sdfg to mixed precision by converting the types of all
    descriptors to float16, except for accumulators.

    :param sdfg: the sdfg to convert.
    :param acctype: the dtype to use for accumulators.
    :param dtype: the dtype to use for other computation.
    :note: operates inplace.
    """

    for node, parent in sdfg.all_nodes_recursive():
        if isinstance(node, nodes.AccessNode):
            if any(e.data.wcr for e in parent.in_edges(node)):
                node.desc(parent._parent).dtype = acctype
            else:
                node.desc(parent._parent).dtype = dtype
