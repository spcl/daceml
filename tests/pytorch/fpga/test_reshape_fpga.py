# Simple test for relu for FPGA

# TODO: conform to pytest syntax if needed

from dace.transformation.interstate import FPGATransformSDFG, InlineSDFG

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import onnx
import numpy as np

import daceml.onnx as donnx
from daceml.pytorch import DaceModule, dace_module
from daceml.onnx import ONNXModel
import copy
import dace
import argparse
import onnx
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
        x = x.view(-1, 256)
        return x


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("W",
                        type=int,
                        nargs="?",
                        default=1,
                        help="Vectorization width")
    parser.add_argument("--onnx_model",
                        type=str,
                        help="Load the model from the given onnx file")

    args = vars(parser.parse_args())

    vec_width = args["W"]
    onnx_file = args["onnx_model"]
    assert(vec_width == 1) #FTMB
    import daceml.onnx as donnx
    donnx.default_implementation = "pure"
    ptmodel = Model()
    data_shape = (10000, 16, 4, 4)
    x = torch.rand(data_shape)
    if onnx_file is None:
        # build the DaCe model from the pytorch model
        dace_model = DaceModule(ptmodel, dummy_inputs=x)
    else:
        # load from file
        onnx_model = onnx.load(onnx_file)
        dace_model = ONNXModel("mymodel", onnx_model)
        print("Loaded from ONNX file")



    # dace_output = dace_model(x)

    torch_output = ptmodel(x)

    # assert np.allclose(torch_output.detach().numpy(), dace_output, atol=1e-06)

    sdfg = dace_model.sdfg

    ##################################
    # Vectorize container

    # find the input node
    # vec_type = dace.vector(dace.float32, vec_width)
    # for name, desc in sdfg.arrays.items():
    #     utils.vectorize_array_and_memlet(sdfg, name, vec_type)
    #     utils.vectorize_array_and_memlet(sdfg, name, vec_type)

    ##########################################
    sdfg.save('/tmp/out.sdfg')


    sdfg.apply_transformations([FPGATransformSDFG])
    # sdfg.states()[0].location["is_FPGA_kernel"] = False

    donnx.ONNXReshape.default_implementation = 'fpga'
    sdfg.expand_library_nodes()
    sdfg.apply_transformations_repeated([InlineSDFG])
    sdfg.save('/tmp/out_fpga_expanded.sdfg')
    dace_output_fpga = dace_model(x)
    dace_output_fpga = dace_output_fpga.reshape(torch_output.detach().numpy().shape)

    print(
        "Difference: ",
        np.linalg.norm(torch_output.detach().numpy() - dace_output_fpga) /
        dace_output_fpga.size)
    assert np.allclose(torch_output.detach().numpy(), dace_output_fpga)
