"""
Lenet FPGA
==========

This example demonstrates using PyTorch Models and FPGA backend to run
a Lenet inference model on FPGA.

Example adapted from https://github.com/pytorch/examples/blob/master/mnist/main.py

"""

# %%
# To run a PyTorch module through DaceML we will need to create the corresponding `DaceModule`

from daceml.torch import DaceModule
import torch
import torch.nn as nn
import torch.nn.functional as F

# %%
# We first define the PyTorch Module, that, in this case, will implement Lenet-5


class TestLeNet(nn.Module):
    def __init__(self):
        super(TestLeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(256, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, 256)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = F.softmax(x, dim=1)
        return x


# %%
# We can build the corresponding `DaceModule` by passing an instance of the PyTorch Module
# (Note: we disable auto_optimization here to allow execution on FPGA)

torch_module = TestLeNet()
daceml_module = DaceModule(torch_module, auto_optimize=False)
# %%
# We need to transform the model SDFG to run on FPGA.
# We do this by registering a few DaCe transformations as transformation hooks

from dace.transformation.interstate import FPGATransformSDFG, InlineSDFG

daceml_module.append_post_onnx_hook(
    "fpga_transform",
    lambda module: module.sdfg.apply_transformations([FPGATransformSDFG]))
daceml_module.append_post_onnx_hook(
    "expand_nodes", lambda module: module.sdfg.expand_library_nodes())
daceml_module.append_post_onnx_hook(
    "inline_nodes",
    lambda module: module.sdfg.apply_transformations_repeated([InlineSDFG]))

# %%
# We can now execute the program with some example inputs, for example a batch of
# 10, 28x28 images.
# To run the model on FPGA, we also specify that FPGA specific ONNX node implementations
# should be used.

import daceml.onnx as donnx
from dace.library import change_default

with change_default(donnx, "fpga"):
    x = torch.rand((10, 1, 28, 28))
    daceml_result = daceml_module(x)

# %%
# Let's check the correctness vs. PyTorch

torch_result = torch_module(x)
assert torch.allclose(torch_result, daceml_result)
torch.linalg.norm(torch_result - daceml_result)

# %%
# Let's take a look at the model SDFG. We can see that it has been specialized for
# execution on FPGAs.

daceml_module.sdfg
