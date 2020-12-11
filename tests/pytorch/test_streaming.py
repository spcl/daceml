# Simple test for evaluating streaming from Conv to Relu

# TODO: conform to pytest syntax if needed
# TODO: render this a real test

from dace.transformation.interstate import FPGATransformSDFG


import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

import daceml.onnx as donnx
import dace
from daceml.pytorch import DaceModule, dace_module
import copy

from daceml.util import utils
def get_library_node_by_name(sdfg, name):

    for node, _ in sdfg.all_nodes_recursive():
        if isinstance(node, dace.sdfg.nodes.LibraryNode):
            if node.name == name:
                return node

    raise Exception("LibNode {} not found".format(name))


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)

    def forward(self, x):
        x =F.relu(self.conv1(x))
        return x


import daceml.onnx as donnx
donnx.default_implementation = "pure"
donnx.ONNXConv.default_implementation = 'im2col'

ptmodel = Model()

# numpy_array = np.arange(0, 1*2*4*4, dtype=np.float32).reshape(1,2,4,4)
# x = torch.from_numpy(numpy_array)
x = torch.rand(100, 1, 28, 28)
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
##################################
# Vectorize container between the two Nodes

# find the node
vec_width = 4
relu_node = get_library_node_by_name(sdfg, "ONNX_Relu_1")
data=utils.in_desc_with_name(relu_node, sdfg.states()[0].nodes()[0].sdfg.states()[0], sdfg.states()[0].nodes()[0].sdfg, "X")
vec_type = dace.vector(dace.float32, vec_width)
data.dtype = vec_type
#adjust shape
prev_shape = data.shape
prev_shape =  prev_shape[:-1] + (prev_shape[-1]//vec_width,)
data.shape = prev_shape
import pdb
pdb.set_trace()

sdfg.expand_library_nodes()
sdfg.save('/tmp/out_fpga_expanded.sdfg')
dace_output_fpga = dace_model(torch.clone(x))

print("Difference: ", np.linalg.norm(torch_output.detach().numpy()-dace_output_fpga)/dace_output_fpga.size)

torch_output_numpy = torch_output.detach().numpy()
diff = torch_output_numpy - dace_output_fpga

assert np.allclose(torch_output.detach().numpy(), dace_output_fpga)
