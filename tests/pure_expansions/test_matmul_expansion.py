import pytest
import numpy as np

import dace
import daceml.onnx as donnx


def test_matmul_expansion():
    sdfg = dace.SDFG("test_matmul_expansion")

    sdfg.add_array("X", [2, 4], dace.float32)
    sdfg.add_array("Z", [4, 3], dace.float32)
    sdfg.add_array("__return", [2, 3], dace.float32)

    state = sdfg.add_state()
    access_X = state.add_access("X")
    access_Z = state.add_access("Z")
    access_result = state.add_access("__return")

    op_node = donnx.ONNXMatMul("Matmul")

    state.add_node(op_node)
    state.add_edge(access_X, None, op_node, "A", sdfg.make_array_memlet("X"))
    state.add_edge(access_Z, None, op_node, "B", sdfg.make_array_memlet("Z"))

    state.add_edge(op_node, "Y", access_result, None,
                   sdfg.make_array_memlet("__return"))

    X = np.random.rand(2, 4).astype(np.float32)
    Z = np.random.rand(4, 3).astype(np.float32)

    sdfg.expand_library_nodes()
    # check that the expansion worked. The default ORT expansion wouldn't produce a map
    assert any(
        isinstance(n, dace.nodes.MapEntry)
        for n, _ in sdfg.all_nodes_recursive())

    result = sdfg(X=X, Z=Z)

    assert np.allclose(X @ Z, result)
