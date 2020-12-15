# Simple test for evaluating streaming from Gemm to relu.
# Relu writes back plain da types


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
from dace.transformation.dataflow import streaming_memory as sm
from dace.transformation.dataflow import PruneConnectors
from dace.transformation.interstate import InlineSDFG



def get_access_node_by_name(sdfg, name):

    for node, state in sdfg.all_nodes_recursive():
        if isinstance(node, dace.sdfg.nodes.AccessNode):
            # print(node.label)
            if node.label == name:
                return node, state

    raise Exception("DataNode {} not found".format(name))

def get_library_node_by_name(sdfg, name):

    for node, _ in sdfg.all_nodes_recursive():
        if isinstance(node, dace.sdfg.nodes.LibraryNode):
            print(node.name)
            if node.name == name:
                return node

    raise Exception("LibNode {} not found".format(name))

def get_sdfg_by_name(sdfg, name):

    for node, _ in sdfg.all_nodes_recursive():
        if isinstance(node, dace.sdfg.nodes.NestedSDFG):
            print(node.label)
            if node.label == name:
                return node

    raise Exception("LibNode {} not found".format(name))


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(256, 120)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return x


import daceml.onnx as donnx
donnx.default_implementation = "pure"
donnx.ONNXConv.default_implementation = 'im2col'

ptmodel = Model()

x = torch.rand(100, 256)
# x = torch.ones(1, 1, 4, 4)

dace_model = DaceModule(ptmodel)
dace_output = dace_model(x)

torch_output = ptmodel(x)
# dace_model.sdfg.expand_library_nodes()
dace_model.sdfg.save('/tmp/out.sdfg')

assert np.allclose(torch_output.detach().numpy(), dace_output, atol=1e-06)

############################################################
# Transform to FPGA
#
sdfg = dace_model.sdfg
orig_sdfg = copy.deepcopy(sdfg)
orig_sdfg.expand_library_nodes()
orig_sdfg.save('/tmp/out_expanded.sdfg')
#
donnx.ONNXGemm.default_implementation = "fpga"
donnx.ONNXRelu.default_implementation = "fpga"
donnx.ONNXMaxPool.default_implementation = "fpga"


##################################
# Vectorize input and output container
vec_width = 2

vec_type = dace.vector(dace.float32, vec_width)
# utils.vectorize_array_and_memlet(sdfg, "ONNX_input", vec_type)

# Vectorize output B of Gemm
# This one is non vectorized: this because will be set as constant
# otherwise we will have problems
# utils.vectorize_array_and_memlet(sdfg, "ONNX_fc1DOTweight", vec_type)

#vectorize output of Gemm
utils.vectorize_array_and_memlet(sdfg, "ONNX_3", vec_type)

# But do not vectorize the ouput of Relu
#vectorize output of Relu

###################################
# Apply transformations

sdfg.apply_transformations([FPGATransformSDFG])
# sdfg.states()[0].location["is_FPGA_kernel"]=False
# sdfg.states()[0].nodes()[0].sdfg.states()[0].location["is_FPGA_kernel"]=False
sdfg.save('/tmp/out_fpga.sdfg')

sdfg.expand_library_nodes()
sdfg.save('/tmp/out_fpga_expanded_pre.sdfg')
sdfg.apply_transformations_repeated([InlineSDFG])
sdfg.save('/tmp/out_fpga_expanded_pre.sdfg')

# get the access node to transform, its predecessor and successor
data , state= get_access_node_by_name(sdfg,"fpga_ONNX_3")
node_a = state.in_edges(data)[0].src
node_b = state.out_edges(data)[0].dst

# Streaming transformation
sm.StreamingComposition.apply_to(state.parent, first=node_a, access=data, second=node_b, verify=False, options={'storage': dace.StorageType.FPGA_Local})
sdfg.apply_transformations_repeated(PruneConnectors)


sdfg.save('/tmp/out_fpga_expanded.sdfg')
dace_output_fpga = dace_model(torch.clone(x))

#reshape if vec_width is different than 1
dace_output_fpga= dace_output_fpga.reshape(dace_output.shape)

print("Difference: ", np.linalg.norm(torch_output.detach().numpy()-dace_output_fpga)/dace_output_fpga.size)

torch_output_numpy = torch_output.detach().numpy()
diff = torch_output_numpy - dace_output_fpga

assert np.allclose(torch_output.detach().numpy(), dace_output_fpga)
