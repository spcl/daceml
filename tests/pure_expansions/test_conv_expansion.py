import pytest
import dace
from daceml.onnx import ONNXConv
import torch
import torch.nn.functional as F
import numpy as np


@pytest.mark.parametrize("num_in_channels, kernel_size, num_filters",
                         [(1, (3, 3), 8), (8, (3, 3), 3), (8, (5, 5), 3),
                          (8, (4, 4), 3)])
@pytest.mark.pure
def test_conv_simple(num_in_channels, kernel_size, num_filters):
    batch_size = 8

    X = np.random.rand(batch_size, num_in_channels, 32, 32).astype(np.float32)
    W = np.random.rand(num_filters, num_in_channels,
                       *kernel_size).astype(np.float32)

    torch_Z = F.conv2d(torch.from_numpy(X), torch.from_numpy(W)).numpy()
    dace_Z = np.zeros_like(torch_Z)

    sdfg = dace.SDFG("conv_test")
    sdfg.add_array("X_arr", X.shape, dace.float32)
    sdfg.add_array("W_arr", W.shape, dace.float32)
    sdfg.add_array("Z_arr", torch_Z.shape, dace.float32)

    state = sdfg.add_state()
    access_X = state.add_access("X_arr")
    access_W = state.add_access("W_arr")
    access_Z = state.add_access("Z_arr")

    conv = ONNXConv("MyConvNode")

    state.add_node(conv)
    state.add_edge(access_X, None, conv, "X", sdfg.make_array_memlet("X_arr"))
    state.add_edge(access_W, None, conv, "W", sdfg.make_array_memlet("W_arr"))
    state.add_edge(conv, "Y", access_Z, None, sdfg.make_array_memlet("Z_arr"))

    sdfg.expand_library_nodes()
    sdfg(X_arr=X, W_arr=W, Z_arr=dace_Z)

    print(torch_Z - dace_Z)
    assert np.allclose(torch_Z, dace_Z)
