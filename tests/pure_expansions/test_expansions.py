import numpy as np
import pytest

import dace
import daceml.onnx as donnx
import daceml.onnx.converters as converters


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

def test_cast_int_to_float():
    sdfg = dace.SDFG("test_cast")

    sdfg.add_array("X", [2, 4], dace.int32)
    sdfg.add_array("__return", [2, 4], dace.float32)

    state = sdfg.add_state()
    access_X = state.add_access("X")
    access_result = state.add_access("__return")

    op_node = donnx.ONNXCast("Cast")
    op_node.to = converters.typeclass_to_onnx_tensor_type_int(dace.float32)

    state.add_node(op_node)
    state.add_edge(access_X, None, op_node, "input", sdfg.make_array_memlet("X"))

    state.add_edge(op_node, "output", access_result, None,
                   sdfg.make_array_memlet("__return"))

    X = np.random.randint(0, 10, size=(2, 4), dtype=np.int32)

    sdfg.expand_library_nodes()
    # check that the expansion worked. The default ORT expansion wouldn't produce a map
    assert any(
        isinstance(n, dace.nodes.MapEntry)
        for n, _ in sdfg.all_nodes_recursive())

    result = sdfg(X=X)

    assert np.allclose(X.astype(np.float32), result)


def test_cast_float_to_int():
    sdfg = dace.SDFG("test_cast")

    sdfg.add_array("X", [2, 4], dace.float32)
    sdfg.add_array("__return", [2, 4], dace.int32)

    state = sdfg.add_state()
    access_X = state.add_access("X")
    access_result = state.add_access("__return")

    op_node = donnx.ONNXCast("Cast")
    op_node.to = converters.typeclass_to_onnx_tensor_type_int(dace.int32)

    state.add_node(op_node)
    state.add_edge(access_X, None, op_node, "input", sdfg.make_array_memlet("X"))

    state.add_edge(op_node, "output", access_result, None,
                   sdfg.make_array_memlet("__return"))

    X = np.random.normal(scale=10, size=(2, 4)).astype(np.float32)

    sdfg.expand_library_nodes()
    # check that the expansion worked. The default ORT expansion wouldn't produce a map
    assert any(
        isinstance(n, dace.nodes.MapEntry)
        for n, _ in sdfg.all_nodes_recursive())

    result = sdfg(X=X)

    assert np.allclose(X.astype(np.int32), result)


def test_cast_float_to_long():
    sdfg = dace.SDFG("test_cast")

    sdfg.add_array("X", [2, 4], dace.float32)
    sdfg.add_array("__return", [2, 4], dace.int64)

    state = sdfg.add_state()
    access_X = state.add_access("X")
    access_result = state.add_access("__return")

    op_node = donnx.ONNXCast("Cast")
    op_node.to = converters.typeclass_to_onnx_tensor_type_int(dace.int64)

    state.add_node(op_node)
    state.add_edge(access_X, None, op_node, "input", sdfg.make_array_memlet("X"))

    state.add_edge(op_node, "output", access_result, None,
                   sdfg.make_array_memlet("__return"))

    X = np.random.normal(scale=10, size=(2, 4)).astype(np.float32)

    sdfg.expand_library_nodes()
    # check that the expansion worked. The default ORT expansion wouldn't produce a map
    assert any(
        isinstance(n, dace.nodes.MapEntry)
        for n, _ in sdfg.all_nodes_recursive())

    result = sdfg(X=X)

    assert np.allclose(X.astype(np.int64), result)


@pytest.mark.parametrize("reduce_type", ["Sum", "Max", "Mean"])
@pytest.mark.parametrize("keepdims", [True, False])
@pytest.mark.parametrize("axes", [[0], [-1], [0, -1]])
def test_reduce_nokeepdims(keepdims, reduce_type, axes):

    X = np.random.normal(scale=10, size=(2, 4, 10)).astype(np.float32)

    sdfg = dace.SDFG("test_reduce")

    sdfg.add_array("X", [2, 4, 10], dace.float32)

    numpy_func = getattr(np, reduce_type.lower())
    numpy_result = numpy_func(X.copy(), axis=tuple(axes), keepdims=keepdims)

    resulting_shape = numpy_result.shape

    sdfg.add_array("__return", resulting_shape, dace.float32)

    state = sdfg.add_state()
    access_X = state.add_access("X")
    access_result = state.add_access("__return")

    op_node = getattr(donnx, "ONNXReduce" + reduce_type)("reduce")
    op_node.axes = axes
    op_node.keepdims = 1 if keepdims else 0

    state.add_node(op_node)
    state.add_edge(access_X, None, op_node, "data", sdfg.make_array_memlet("X"))

    state.add_edge(op_node, "reduced", access_result, None,
                   sdfg.make_array_memlet("__return"))


    sdfg.expand_library_nodes()
    # check that the expansion worked. The default ORT expansion wouldn't produce a map
    assert any(
        isinstance(n, dace.nodes.MapEntry)
        for n, _ in sdfg.all_nodes_recursive())

    result = sdfg(X=X)

    assert np.allclose(numpy_result, result)

def test_reduce_scalar():
    X = np.random.normal(scale=10, size=(2, 4, 10)).astype(np.float32)

    sdfg = dace.SDFG("test_reduce")

    numpy_result = np.mean(X)

    sdfg.add_array("X", [2, 4, 10], dace.float32)
    sdfg.add_scalar("Y", dace.float32, transient=True)
    sdfg.add_array("__return", [1], dace.float32)


    state = sdfg.add_state()
    access_X = state.add_access("X")
    access_Y = state.add_access("Y")
    access_result = state.add_access("__return")

    op_node = donnx.ONNXReduceMean("mean")
    op_node.keepdims = 0

    state.add_node(op_node)
    state.add_edge(access_X, None, op_node, "data", sdfg.make_array_memlet("X"))

    state.add_edge(op_node, "reduced", access_Y, None,
                   sdfg.make_array_memlet("Y"))

    state.add_edge(access_Y, None, access_result, None,
                   sdfg.make_array_memlet("__return"))


    sdfg.expand_library_nodes()
    # check that the expansion worked. The default ORT expansion wouldn't produce a map
    assert any(
        isinstance(n, dace.nodes.MapEntry)
        for n, _ in sdfg.all_nodes_recursive())

    result = sdfg(X=X)

    assert np.allclose(numpy_result, result)

if __name__ == "__main__":
    test_reduce_nokeepdims(reduce_type="Sum", axes=[-1], keepdims=False)
