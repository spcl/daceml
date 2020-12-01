# Simple test for evaluating 2D convolutions for FPGA

# TODO: conform to pytest syntax if needed

from dace.transformation.interstate import FPGATransformSDFG


import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

import daceml.onnx as donnx
from daceml.pytorch import DaceModule, dace_module


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(1, 4, 3, bias = False)
        # self.conv2 = nn.Conv2d(4, 4, 3)

    def forward(self, x):
        return self.conv1(x)
        # x = F.relu(self.conv1(x))
        # return F.relu(self.conv2(x))


import daceml.onnx as donnx
donnx.default_implementation = "pure"

ptmodel = Model()
x = torch.rand(1, 1, 8, 8)

dace_model = DaceModule(ptmodel)
dace_output = dace_model(x)

torch_output = ptmodel(x)
# dace_model.sdfg.expand_library_nodes()
dace_model.sdfg.save('/tmp/out.sdfg')
assert np.allclose(torch_output.detach().numpy(), dace_output, atol=1e-06)


# Transform to FPGA

donnx.ONNXConv.default_implementation = "fpga"
sdfg = dace_model.sdfg
sdfg.apply_transformations([FPGATransformSDFG])
sdfg.states()[0].location["is_FPGA_kernel"]=False
# sdfg.states()[0].nodes()[0].sdfg.states()[0].location["is_FPGA_kernel"]=False
sdfg.save('/tmp/out_fpga.sdfg')

sdfg.expand_library_nodes()
sdfg.save('/tmp/out_fpga_expanded.sdfg')
dace_output_fpga = dace_model(torch.clone(x))

print("Difference: ", np.linalg.norm(torch_output.detach().numpy()-dace_output_fpga)/dace_output_fpga.size)
assert np.allclose(torch_output.detach().numpy(), dace_output_fpga)
