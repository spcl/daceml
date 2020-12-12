# Simple test for relu for FPGA

# TODO: conform to pytest syntax if needed

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
def get_library_node_by_name(sdfg, name):

    for node, _ in sdfg.all_nodes_recursive():
        if isinstance(node, dace.sdfg.nodes.LibraryNode):
            if node.name == name:
                return node

    raise Exception("LibNode {} not found".format(name))








def get_node_predecessors(node, state):
    '''
    Returns the LibNode that are predecessors of the passed one
    :param node:
    :param graph:
    :return:
    '''
    # Check if the node has some library node as predecessor as
    predecessors = []
    for edge in state.in_edges(node):
        import pdb
        pdb.set_trace()
        # check that this edge has a predecessor
        pred = edge.src

        if isinstance(pred, dace.sdfg.nodes.AccessNode):
            predecessors.append(pred)

    return predecessors

def get_data_node_by_name(node, state, sdfg, name):
    return sdfg.arrays[utils.in_edge_with_name(node, state, name)]




class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x):
        return F.relu(x)


import daceml.onnx as donnx
donnx.default_implementation = "pure"

ptmodel = Model()

data_shape = (10,4,32,32)
# I don't get why does not takes a tuple as input
x = torch.FloatTensor(10,4,32,32).random_(-5, 5)

dace_model = DaceModule(ptmodel)
dace_output = dace_model(x)

torch_output = ptmodel(x)
dace_model.sdfg.save('/tmp/out.sdfg')

assert np.allclose(torch_output.detach().numpy(), dace_output, atol=1e-06)

# Transform to FPGA

sdfg = dace_model.sdfg
start_sdfg = copy.deepcopy(sdfg)
orig_sdfg = copy.deepcopy(sdfg)
orig_sdfg.expand_library_nodes()
orig_sdfg.save('/tmp/out_expanded.sdfg')


##################################
# Vectorize container

# find the input node
vec_width = 4
vec_type = dace.vector(dace.float32, vec_width)
utils.vectorize_array_and_memlet(sdfg, "ONNX_x", vec_type)
utils.vectorize_array_and_memlet(sdfg, "ONNX_1", vec_type)

sdfg.apply_transformations([FPGATransformSDFG])
sdfg.states()[0].location["is_FPGA_kernel"] = False
sdfg.save('/tmp/out_fpga.sdfg')

donnx.ONNXRelu.default_implementation = "fpga"



sdfg.expand_library_nodes()
sdfg.save('/tmp/out_fpga_expanded.sdfg')
dace_output_fpga = dace_model(torch.clone(x))
dace_output_fpga=dace_output_fpga.reshape(data_shape)

print(
    "Difference: ",
    np.linalg.norm(torch_output.detach().numpy() - dace_output_fpga) /
    dace_output_fpga.size)
assert np.allclose(torch_output.detach().numpy(), dace_output_fpga)
