#!/usr/bin/env python3
# Simple test for binary operator for FPGA

from dace.transformation.interstate import FPGATransformSDFG, InlineSDFG

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

import daceml.onnx as donnx
from daceml.pytorch import DaceModule, dace_module
import copy
import dace
import argparse
from daceml.util import utils


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, y):
        return torch.add(x, y)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("W",
                        type=int,
                        nargs="?",
                        default=1,
                        help="Vectorization width")
    parser.add_argument("vendor",
                        type=str,
                        nargs="?",
                        default='xilinx',
                        choices=['xilinx', 'intel_fpga'],
                        help="FPGA Vendor")

    args = vars(parser.parse_args())

    vec_width = args["W"]
    import daceml.onnx as donnx
    donnx.default_implementation = "pure"

    ptmodel = Model()

    data_shape = (10000, 4, 32, 32)
    # x = torch.FloatTensor(1000,4,32,32).random_(-5, 5)
    x = torch.rand(data_shape) - 0.5
    y = torch.rand(data_shape) - 0.5

    # Create dace model and run in dace
    dace_model = DaceModule(ptmodel)
    dace_output = dace_model(x, y)

    # Run the computation in pytorch
    torch_output = ptmodel(x, y)

    assert np.allclose(torch_output.detach().numpy(), dace_output, atol=1e-06)

    # ################################
    # Transform to FPGA
    # ################################
    sdfg = dace_model.sdfg

    # Set vendor
    vendor = args["vendor"]
    mode = "simulation"
    if vendor == 'intel_fpga':
        mode = "emulator"
    dace.config.Config.set("compiler", "fpga_vendor", value=vendor)
    dace.config.Config.set("compiler", vendor, "mode", value=mode)

    # Vectorize container
    vec_type = dace.vector(dace.float32, vec_width)
    utils.vectorize_array_and_memlet(sdfg, "ONNX_0", vec_type)
    utils.vectorize_array_and_memlet(sdfg, "ONNX_1", vec_type)
    utils.vectorize_array_and_memlet(sdfg, "ONNX_2", vec_type)

    # Save untransformed SDFG
    sdfg.save('/tmp/dace/out.sdfg')

    # Transform for execution on FPGA
    sdfg.apply_transformations([FPGATransformSDFG])

    # Expand library nodes
    donnx.ONNXAdd.default_implementation = "fpga"
    sdfg.expand_library_nodes()
    sdfg.save('/tmp/dace/out_fpga_expanded.sdfg')

    sdfg.apply_transformations_repeated([InlineSDFG])
    dace_output_fpga = dace_model(torch.clone(x), torch.clone(y))
    dace_output_fpga = dace_output_fpga.reshape(data_shape)

    print(
        "Difference: ",
        np.linalg.norm(torch_output.detach().numpy() - dace_output_fpga) /
        dace_output_fpga.size)
    assert np.allclose(torch_output.detach().numpy(), dace_output_fpga)
