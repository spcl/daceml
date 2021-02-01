# Simple test for softmax for FPGA

# TODO: conform to pytest syntax if needed

from dace.transformation.interstate import FPGATransformSDFG, InlineSDFG

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

import daceml.onnx as donnx
from daceml.pytorch import DaceModule, dace_module
import copy


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x):
        x = F.softmax(x, dim=1)
        return x


import daceml.onnx as donnx
donnx.default_implementation = "pure"

ptmodel = Model()
x = torch.rand(1000, 10, dtype=torch.float32)

dace_model = DaceModule(ptmodel)
dace_output = dace_model(x)

torch_output = ptmodel(x)
dace_model.sdfg.save('/tmp/out.sdfg')

assert np.allclose(torch_output.detach().numpy(), dace_output, atol=1e-06)

# Transform to FPGA

sdfg = dace_model.sdfg
sdfg.save('/tmp/out.sdfg')

donnx.ONNXSoftmax.default_implementation = "fpga"
sdfg.apply_transformations([FPGATransformSDFG])
sdfg.expand_library_nodes()
sdfg.apply_transformations_repeated([InlineSDFG])

sdfg.save('/tmp/out_fpga.sdfg')


sdfg.save('/tmp/out_fpga_expanded.sdfg')
dace_output_fpga = dace_model(torch.clone(x))

print(
    "Difference: ",
    np.linalg.norm(torch_output.detach().numpy() - dace_output_fpga) /
    dace_output_fpga.size)
assert np.allclose(torch_output.detach().numpy(), dace_output_fpga)
