# Simple test for gemm for FPGA
# the GEMM ONNX operator is used when we use a fully connected layer

# TODO: conform to pytest syntax if needed

from dace.transformation.interstate import FPGATransformSDFG

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
        self.fc1 = nn.Linear(256, 120)

    def forward(self, x):
        return self.fc1(x)


import daceml.onnx as donnx
donnx.default_implementation = "pure"

ptmodel = Model()
x = torch.rand(256, 256, dtype=torch.float32)

dace_model = DaceModule(ptmodel)
dace_output = dace_model(x)

torch_output = ptmodel(x)
dace_model.sdfg.save('/tmp/out.sdfg')

assert np.allclose(torch_output.detach().numpy(), dace_output, atol=1e-06)

# Transform to FPGA

sdfg = dace_model.sdfg
orig_sdfg = copy.deepcopy(sdfg)
orig_sdfg.expand_library_nodes()
orig_sdfg.save('/tmp/out_expanded.sdfg')

donnx.ONNXGemm.default_implementation = "fpga"
sdfg.apply_transformations([FPGATransformSDFG])
sdfg.states()[0].location["is_FPGA_kernel"] = False
# one step beyond
sdfg.states()[0].nodes()[0].sdfg.states()[0].location["is_FPGA_kernel"] = False

sdfg.save('/tmp/out_fpga.sdfg')

sdfg.expand_library_nodes()
sdfg.save('/tmp/out_fpga_expanded.sdfg')
dace_output_fpga = dace_model(torch.clone(x))

diff =  np.linalg.norm(torch_output.detach().numpy() - dace_output_fpga) /dace_output_fpga.size
print("Difference: ", diff)

assert(diff < 1e-6)

# can not use np all close here
#assert np.allclose(torch_output.detach().numpy(), dace_output_fpga)
