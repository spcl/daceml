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
from multiprocessing import Process, Queue


class Model(nn.Module):
    def __init__(self, new_shape):
        super(Model, self).__init__()
        self.new_shape = new_shape

    def forward(self, x):
        x = x.reshape(self.new_shape)
        return x


def run(data_shape: tuple, reshaped_shape: tuple, vec_width=1, queue=None):
    # dace_output = dace_model(x)

    import daceml.onnx as donnx
    donnx.default_implementation = "pure"
    ptmodel = Model(reshaped_shape)
    x = torch.rand(data_shape)

    torch_output = ptmodel(x)

    dace_model = DaceModule(ptmodel, auto_optimize=False)
    out = dace_model(x)
    sdfg = dace_model.sdfg
    sdfg.apply_transformations([FPGATransformSDFG])

    donnx.ONNXReshape.default_implementation = 'fpga'
    sdfg.expand_library_nodes()
    sdfg.apply_transformations_repeated([InlineSDFG])

    dace_output_fpga = dace_model(x)
    dace_output_fpga = dace_output_fpga.reshape(
        torch_output.detach().numpy().shape).detach().numpy()

    torch_output_numpy = torch_output.detach().numpy()
    diff = np.linalg.norm(torch_output_numpy - dace_output_fpga
                          ) / np.linalg.norm(torch_output_numpy)

    print("Difference: ", diff)
    if queue is not None:
        # we are testing
        queue.put(diff)
    else:
        if diff > 1e-9:
            import pdb
            pdb.set_trace()
            assert (False)

    del dace_model, ptmodel, x


def test():
    '''
    Evaluates multiple combination of Reshape
    :return:
    '''
    print("----------- Testing Reshape ---------------")

    # Run FPGA tests in a different process to avoid issues with Intel OpenCL tools
    # (But not in parallel)

    # each position of this lists contains a test configuration
    vec_width = [1, 1, 1, 1]
    x_shapes = [(16, 4, 4, 4), (16, 2, 32), (16, 8, 8), (8, 16, 16)]
    y_shapes = [(16, 64), (16, 8, 8), (16, 2, 32), (2, 4, 16, 16)]  # reshpaed

    for i in range(0, len(vec_width)):
        print("##########################################################")
        print(
            f"# Configuration: vw={vec_width[i]}, x_shape={x_shapes[i]}, reshaped_shape={y_shapes[i]}"
        )
        print("##########################################################")
        queue = Queue()
        p = Process(target=run,
                    args=(x_shapes[i], y_shapes[i], vec_width[i], queue))
        p.start()
        p.join()
        assert (queue.get() < 1e-9)


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
        data_shape = (16, 4, 4, 4)
        reshaped_shape = (16, 64)
        run(data_shape, reshaped_shape)
