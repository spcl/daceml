# Simple test for relu for FPGA

# TODO: conform to pytest syntax if needed

import torch
import torch.nn as nn
import torch.nn.functional as F
import dace
import numpy as np
from dace.transformation.interstate import FPGATransformSDFG, InlineSDFG
from daceml.util import utils

import daceml.onnx as donnx
from daceml.pytorch import DaceModule, dace_module
import copy
import argparse


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x):
        return F.max_pool2d(x, 2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("W",
                        type=int,
                        nargs="?",
                        default=1,
                        help="Vectorization width")

    args = vars(parser.parse_args())

    vec_width = args["W"]
    import daceml.onnx as donnx
    donnx.default_implementation = "pure"

    ptmodel = Model()
    data_shape = (1000, 6, 32, 32)
    x = torch.rand(data_shape)

    dace_model = DaceModule(ptmodel)
    dace_output = dace_model(x)
    torch_output = ptmodel(x)
    assert np.allclose(torch_output.detach().numpy(),
                       dace_output.numpy(),
                       atol=1e-06)

    # Transform to FPGA
    sdfg = dace_model.sdfg

    ##################################
    # Vectorize container

    # find the input node, for the moment being maxpool writes only to non vectorized containers
    vec_type = dace.vector(dace.float32, vec_width)
    utils.vectorize_array_and_memlet(sdfg, "ONNX_0", vec_type)

    ##########################################

    donnx.ONNXMaxPool.default_implementation = "fpga"

    sdfg.apply_transformations([FPGATransformSDFG])
    sdfg.expand_library_nodes()
    sdfg.apply_transformations_repeated([InlineSDFG])
    dace_output_fpga = dace_model(torch.clone(x))

    print(
        "Difference: ",
        np.linalg.norm(torch_output.detach().numpy() -
                       dace_output_fpga.numpy()) /
        np.linalg.norm(torch_output.detach().numpy()))
    assert np.allclose(torch_output.detach().numpy(), dace_output_fpga.numpy())
