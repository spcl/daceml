"""
Using ONNX Library Nodes
========================

This example demonstrates using ONNX library nodes.
"""

# %%
# The easiest way to use ONNX library nodes is using the dace python frontend

import dace
import daceml.onnx as donnx
import numpy as np


@dace.program
def conv_program(X_arr: dace.float32[5, 3, 10, 10], W_arr: dace.float32[16, 3,
                                                                        3, 3]):
    output = np.ndarray([5, 16, 4, 4], dtype=np.float32)
    donnx.ONNXConv(X=X_arr, W=W_arr, Y=output, strides=[2, 2])
    return output


# %%
# The resulting SDFG contains an instance of the :class:`~daceml.onnx.nodes.onnx_op.ONNXConv` library node.

conv_program.to_sdfg()

# %%
# We can now execute the program with some example inputs

X = np.random.rand(5, 3, 10, 10).astype(np.float32)
W = np.random.rand(16, 3, 3, 3).astype(np.float32)

result = conv_program(X_arr=X, W_arr=W)

# %%
# Let's check the correctness vs. PyTorch

import torch
import torch.nn.functional as F

torch_result = F.conv2d(torch.from_numpy(X), torch.from_numpy(W),
                        stride=2).numpy()

assert np.allclose(torch_result, result)

# %%
# We can also use ONNX nodes using the SDFG Python API.

from daceml.onnx import ONNXConv

sdfg = dace.SDFG("conv_example")
sdfg.add_array("X_arr", (5, 3, 10, 10), dace.float32)
sdfg.add_array("W_arr", (16, 3, 3, 3), dace.float32)
sdfg.add_array("Z_arr", (5, 16, 4, 4), dace.float32)

state = sdfg.add_state()
access_X = state.add_access("X_arr")
access_W = state.add_access("W_arr")
access_Z = state.add_access("Z_arr")

conv = ONNXConv("MyConvNode", strides=[2, 2])

state.add_node(conv)
state.add_edge(access_X, None, conv, "X", sdfg.make_array_memlet("X_arr"))
state.add_edge(access_W, None, conv, "W", sdfg.make_array_memlet("W_arr"))
state.add_edge(conv, "Y", access_Z, None, sdfg.make_array_memlet("Z_arr"))

sdfg

# %%
# The SDFG looks the same as the one above. Now let's try running it

Z = np.zeros((5, 16, 4, 4)).astype(np.float32)
sdfg(X_arr=X, W_arr=W, Z_arr=Z)
assert np.allclose(torch_result, Z)
