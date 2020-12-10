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

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(6, 16, 5)

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

# numpy_array = np.arange(0, 1*2*4*4, dtype=np.float32).reshape(1,2,4,4)
# x = torch.from_numpy(numpy_array)
x = torch.rand(100, 6, 24, 24)
# x = torch.ones(1, 1, 4, 4)

dace_model = DaceModule(ptmodel)
dace_output = dace_model(x)

torch_output = ptmodel(x)
# dace_model.sdfg.expand_library_nodes()
dace_model.sdfg.save('/tmp/out.sdfg')

assert np.allclose(torch_output.detach().numpy(), dace_output, atol=1e-06)


# Transform to FPGA
#
sdfg = dace_model.sdfg
orig_sdfg = copy.deepcopy(sdfg)
orig_sdfg.expand_library_nodes()
orig_sdfg.save('/tmp/out_expanded.sdfg')
#
donnx.ONNXConv.default_implementation = "fpga"
sdfg.apply_transformations([FPGATransformSDFG])
sdfg.states()[0].location["is_FPGA_kernel"]=False
sdfg.states()[0].nodes()[0].sdfg.states()[0].location["is_FPGA_kernel"]=False
sdfg.save('/tmp/out_fpga.sdfg')

sdfg.expand_library_nodes()
sdfg.save('/tmp/out_fpga_expanded.sdfg')
dace_output_fpga = dace_model(torch.clone(x))

print("Difference: ", np.linalg.norm(torch_output.detach().numpy()-dace_output_fpga)/dace_output_fpga.size)

torch_output_numpy = torch_output.detach().numpy()
diff = torch_output_numpy - dace_output_fpga

assert np.allclose(torch_output.detach().numpy(), dace_output_fpga)
