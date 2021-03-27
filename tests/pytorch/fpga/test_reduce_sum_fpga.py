# Simple test for reduce_sum for FPGA

# NOTE: for the moment being it supports only the last axis

from dace.transformation.interstate import FPGATransformSDFG, InlineSDFG

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

import daceml.onnx as donnx
from daceml.pytorch import DaceModule, dace_module
import copy
import argparse
from multiprocessing import Process, Queue


class Model(nn.Module):
    def __init__(self, axis):
        super(Model, self).__init__()
        self.axis = axis

    def forward(self, x):
        x = torch.sum(x, (self.axis), False)
        return x


def run(data_shape: tuple, axis, queue=None):

    import daceml.onnx as donnx
    donnx.default_implementation = "pure"

    ptmodel = Model(axis)
    x = torch.rand(data_shape)

    dace_model = DaceModule(ptmodel, auto_optimize=False)
    dace_output = dace_model(x)

    torch_output = ptmodel(x)
    assert np.allclose(torch_output.detach().numpy(), dace_output, atol=1e-06)

    # Transform to FPGA

    sdfg = dace_model.sdfg

    donnx.ONNXReduceSum.default_implementation = "fpga"
    sdfg.apply_transformations([FPGATransformSDFG])
    sdfg.expand_library_nodes()
    sdfg.apply_transformations_repeated([InlineSDFG])

    dace_output_fpga = dace_model(torch.clone(x))

    diff = np.linalg.norm(torch_output.detach().numpy() -
                          dace_output_fpga.numpy()) / np.linalg.norm(
                              torch_output.detach().numpy())

    print("Difference: ", diff)
    if queue is not None:
        # we are testing
        queue.put(diff)
    else:
        if diff > 1e-6:
            import pdb
            pdb.set_trace()
            assert (False)

    del dace_model, ptmodel, x


def test():
    pass  #NYI


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("W",
                        type=int,
                        nargs="?",
                        default=1,
                        help="Vectorization width")
    parser.add_argument("-test",
                        action="store_true",
                        default=False,
                        help="Perform tests (USE ONLY WITH EMULATION)")

    args = vars(parser.parse_args())

    vec_width = args["W"]
    t = args["test"]

    if t:
        test()
    else:
        data_shape = (2, 4, 16, 16)
        run(data_shape, 1)
