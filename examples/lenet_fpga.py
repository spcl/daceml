"""
Lenet FPGA
========================

This example demonstrates using PyTorch Models and FPGA backend to run
a Lenet inference model on FPGA.

Example adapted from https://github.com/pytorch/examples/blob/master/mnist/main.py

"""

# %%
# To run a PyTorch module through DaceML we will need to create the corresponding `DaceModule`


from daceml.pytorch import DaceModule
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

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
# We can now execute the program with some example inputs, for example a batch of
# 10, 28x28 images

x = torch.rand((10, 1, 28, 28))
daceml_result = daceml_module(x)

# %%
# Let's check the correctness vs. PyTorch

torch_result = torch_module(x)
assert np.allclose(torch_result.detach().numpy(), daceml_result)

# %%
# At this point, we want to run the same Model on FPGA
# First, we impose to DaceML to use FPGA specific ONNX node implementations
import daceml.onnx as donnx
donnx.default_implementation = "fpga"

# %%
# Then, we need to transform the underlying SDFG representation to run on FPGA
# For doing this we resort to DaCe transformations

from dace.transformation.interstate import FPGATransformSDFG, InlineSDFG
daceml_module.sdfg.apply_transformations([FPGATransformSDFG])
daceml_module.sdfg.expand_library_nodes()
daceml_module.sdfg.apply_transformations_repeated([InlineSDFG])

# %%
# Finally, we can compute and execute the DaceML module once, again. At this point
# it will automatically run on the FPGA

daceml_module.sdfg.compile()
daceml_fpga_result = daceml_module(x)

assert np.allclose(torch_result.detach().numpy(), daceml_fpga_result)
