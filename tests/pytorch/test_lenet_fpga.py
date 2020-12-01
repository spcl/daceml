# Lenet test targeting FPGA

#TODO: conform to pytest syntax

import pytest
import numpy as np

from daceml.pytorch import DaceModule
from dace.transformation.interstate import FPGATransformSDFG

import torch
import torch.nn as nn
import torch.nn.functional as F


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.fc1 = nn.Linear(16 * 6 * 6, 120)  # 6*6 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, 576)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


import daceml.onnx as donnx
donnx.default_implementation = "pure"

input = torch.rand(8, 1, 32, 32, dtype=torch.float32)

net = LeNet()
dace_net = LeNet()
dace_net.load_state_dict(net.state_dict())
dace_net = DaceModule(dace_net)

# Check CPU Output
torch_output = net(torch.clone(input))
dace_output = dace_net(torch.clone(input))
assert np.allclose(torch_output.detach().numpy(), dace_output)

# Transform to FPGA
sdfg = dace_net.sdfg
sdfg.apply_transformations([FPGATransformSDFG])
sdfg.states()[0].location["is_FPGA_kernel"]=False
sdfg.states()[0].nodes()[0].sdfg.states()[0].location["is_FPGA_kernel"]=False
sdfg.save('/tmp/out_fpga.sdfg')

sdfg.expand_library_nodes()
sdfg.save('/tmp/out_fpga_expanded.sdfg')
dace_output_fpga = dace_net(torch.clone(input))

assert np.allclose(torch_output.detach().numpy(), dace_output_fpga)

print("Difference: ", np.linalg.norm(torch_output.detach().numpy()-dace_output_fpga)/dace_output_fpga.size)