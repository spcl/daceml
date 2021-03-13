# Simple test for evaluating Conv-Relu-Maxpool

from dace.transformation.interstate import FPGATransformSDFG, InlineSDFG
from daceml.transformation import InputToConstant

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
import argparse


class Model(nn.Module):
    def __init__(self, input_to_constant=False):
        super(Model, self).__init__()
        #first conv
        self.conv = nn.Conv2d(1, 6, 5)
        #second conv
        # self.conv = nn.Conv2d(6, 16, 5)
        if input_to_constant:
            #fix the weight otherwise everytime they are randomized
            self.conv.weight.data.fill_(0.1)
            self.conv.bias.data.fill_(1)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv(x)), 2)
        return x


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("W",
                        type=int,
                        nargs="?",
                        default=1,
                        help="Vectorization width")
    parser.add_argument("-input_to_constant",
                        action="store_true",
                        default=False,
                        help="Apply InputToConstant")

    args = vars(parser.parse_args())
    vec_width = args["W"]
    input_to_constant = args["input_to_constant"]

    import daceml.onnx as donnx
    donnx.default_implementation = "pure"
    donnx.ONNXConv.default_implementation = 'im2col'

    ptmodel = Model(input_to_constant)
    #first conv
    data_shape = (100, 1, 28, 28)
    #second conv
    # data_shape = (100, 6, 12, 12)
    x = torch.rand(data_shape)
    dace_model = DaceModule(ptmodel)
    dace_output = dace_model(x)

    torch_output = ptmodel(x)

    assert np.allclose(torch_output.detach().numpy(), dace_output, atol=1e-06)

    donnx.ONNXConv.default_implementation = "fpga"
    donnx.ONNXRelu.default_implementation = "fpga"
    donnx.ONNXMaxPool.default_implementation = "fpga"

    sdfg = dace_model.sdfg
    ##################################
    # Vectorize input and output container
    vec_width = vec_width
    vec_type = dace.vector(dace.float32, vec_width)

    # vectorize output of Conv
    utils.vectorize_array_and_memlet(sdfg, "ONNX_3", vec_type)
    # vectorize output of Relu
    utils.vectorize_array_and_memlet(sdfg, "ONNX_4", vec_type)

    ############################################################
    # Transform to FPGA

    donnx.ONNXConv.default_implementation = "fpga"
    donnx.ONNXRelu.default_implementation = "fpga"
    donnx.ONNXMaxPool.default_implementation = "fpga"

    # Apply transformations
    sdfg.apply_transformations([FPGATransformSDFG])
    sdfg.expand_library_nodes()
    sdfg.apply_transformations_repeated([InlineSDFG])

    if input_to_constant:
        sdfg.apply_transformations_repeated([InputToConstant],
                                            print_report=True)
    #######################################################################
    # Streaming Composition
    sdfg.apply_transformations_repeated(
        [InlineSDFG, sm.StreamingComposition],
        [{}, {
            "storage": dace.StorageType.FPGA_Local
        }])

    dace_output_fpga = dace_model(torch.clone(x))

    dace_output_fpga = dace_output_fpga.reshape(dace_output.shape)

    torch_output_numpy = torch_output.detach().numpy()
    diff = np.linalg.norm(torch_output_numpy -
                          dace_output_fpga) / dace_output_fpga.size

    print("Difference: ", diff)
    assert (diff < 1e-6)
