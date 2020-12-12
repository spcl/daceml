# Simple test for evaluating 2D convolutions for FPGA

# TODO: conform to pytest syntax if needed
# TODO: render this a real test

from dace.transformation.interstate import FPGATransformSDFG


import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

import daceml.onnx as donnx
from daceml.pytorch import DaceModule, dace_module
import copy
import dace
from daceml.util import utils

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(1, 6, 5)

        self.conv.weight = torch.nn.Parameter(torch.ones_like(self.conv.weight))
        # self.conv = nn.Conv2d(4, 4, 3)

    def forward(self, x):
        return self.conv(x)
        # x = F.relu(self.conv1(x))
        # return F.relu(self.conv2(x))


import daceml.onnx as donnx
donnx.default_implementation = "pure"
donnx.ONNXConv.default_implementation = 'im2col'

ptmodel = Model()
data_shape = (100,1,28,28)
vec_width = 4

x = torch.rand(data_shape)

dace_model = DaceModule(ptmodel)
dace_output = dace_model(x)

torch_output = ptmodel(x)
dace_model.sdfg.save('/tmp/out.sdfg')

assert np.allclose(torch_output.detach().numpy(), dace_output, atol=1e-06)

# Save sdfg to file
sdfg = dace_model.sdfg
orig_sdfg = copy.deepcopy(sdfg)
orig_sdfg.expand_library_nodes()
orig_sdfg.save('/tmp/out_expanded.sdfg')

##################################
# Vectorize input and output container

vec_type = dace.vector(dace.float32, vec_width)
# utils.vectorize_array_and_memlet(sdfg, "ONNX_input", vec_type)
utils.vectorize_array_and_memlet(sdfg, "ONNX_3", vec_type)

##################################
# Transfor to FPGA

sdfg.apply_transformations([FPGATransformSDFG])
sdfg.states()[0].location["is_FPGA_kernel"]=False
sdfg.states()[0].nodes()[0].sdfg.states()[0].location["is_FPGA_kernel"]=False
sdfg.save('/tmp/out_fpga.sdfg')
donnx.ONNXConv.default_implementation = "fpga"

sdfg.expand_library_nodes()
sdfg.save('/tmp/out_fpga_expanded.sdfg')
dace_output_fpga = dace_model(torch.clone(x))
dace_output_fpga=dace_output_fpga.reshape(dace_output.shape)

print("Difference: ", np.linalg.norm(torch_output.detach().numpy()-dace_output_fpga)/dace_output_fpga.size)

torch_output_numpy = torch_output.detach().numpy()
diff = torch_output_numpy - dace_output_fpga

assert np.allclose(torch_output.detach().numpy(), dace_output_fpga)
