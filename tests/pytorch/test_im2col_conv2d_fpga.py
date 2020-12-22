# Simple test for evaluating 2D convolutions for FPGA

# TODO: conform to pytest syntax if needed
# TODO: render this a real test

from dace.transformation.interstate import FPGATransformSDFG


import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import numpy as np

import daceml.onnx as donnx
from daceml.pytorch import DaceModule, dace_module
import copy
import dace
from daceml.util import utils
from dace.transformation.interstate import FPGATransformSDFG, InlineSDFG
from daceml.transformation import InputToConstant
from dace.transformation.dataflow import streaming_memory as sm
from dace.transformation.dataflow import PruneConnectors

import daceml.onnx as donnx
donnx.default_implementation = "pure"
donnx.ONNXConv.default_implementation = 'im2col'

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





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("N", type=int, nargs="?", default=4)
    parser.add_argument("M", type=int, nargs="?", default=4)
    parser.add_argument("-input_to_constant", action="store_true", default=False, help= "Apply InputToConstant")

    args = vars(parser.parse_args())
    input_to_constant = args["input_to_constant"]
    ptmodel = Model()
    data_shape = (1000,6,12,12)

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

    ###################################################
    # Transform for FPGA and Inline
    donnx.ONNXConv.default_implementation = "fpga"
    sdfg.apply_transformations([FPGATransformSDFG])
    sdfg.apply_transformations_repeated([InlineSDFG])

    ##################################
    # Vectorize input and output container
    vec_width = 8
    vec_type = dace.vector(dace.float32, vec_width)
    utils.vectorize_array_and_memlet(sdfg, "fpga_ONNX_3", vec_type)

    ###################################
    sdfg.save('/tmp/out_vectorized.sdfg')
    sdfg.expand_library_nodes()
    sdfg.apply_transformations_repeated([InlineSDFG])

    # ###################################################################
    # # Input to constant
    if input_to_constant:
        sdfg.apply_transformations_repeated([InputToConstant], print_report=True)

    dace_output_fpga = dace_model(torch.clone(x))
    dace_output_fpga=dace_output_fpga.reshape(dace_output.shape)

    print("Difference: ", np.linalg.norm(torch_output.detach().numpy()-dace_output_fpga)/dace_output_fpga.size)

    torch_output_numpy = torch_output.detach().numpy()
    diff = torch_output_numpy - dace_output_fpga

    assert np.allclose(torch_output.detach().numpy(), dace_output_fpga)
