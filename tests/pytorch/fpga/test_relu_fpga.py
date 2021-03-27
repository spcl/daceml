# Simple test for relu for FPGA

from dace.transformation.interstate import FPGATransformSDFG, InlineSDFG

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

import daceml.onnx as donnx
from daceml.pytorch import DaceModule, dace_module
import dace
import argparse
from daceml.util import utils
from multiprocessing import Process, Queue


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x):
        return F.relu(x)


def run(data_shape: tuple, vec_width=1, queue=None):
    '''
    Evaluates a specific configuration
    :param data_shape:
    :param vec_width:
    :param queue:
    :return:
    '''
    import daceml.onnx as donnx
    donnx.default_implementation = "pure"

    ptmodel = Model()
    x = torch.rand(data_shape) - 0.5
    dace_model = DaceModule(ptmodel, auto_optimize=False)
    dace_output = dace_model(x)

    torch_output = ptmodel(x)

    assert np.allclose(torch_output.detach().numpy(), dace_output, atol=1e-06)

    # Transform to FPGA

    sdfg = dace_model.sdfg

    ##################################
    # Vectorize container

    # find the input node
    vec_type = dace.vector(dace.float32, vec_width)
    utils.vectorize_array_and_memlet(sdfg, "ONNX_x", vec_type)
    utils.vectorize_array_and_memlet(sdfg, "ONNX_1", vec_type)

    ##########################################

    sdfg.apply_transformations([FPGATransformSDFG])
    donnx.ONNXRelu.default_implementation = "fpga"
    sdfg.expand_library_nodes()
    sdfg.apply_transformations_repeated([InlineSDFG])
    dace_output_fpga = dace_model(x)
    dace_output_fpga = dace_output_fpga.reshape(data_shape)
    diff = np.linalg.norm(torch_output.detach().numpy() -
                          dace_output_fpga.numpy()) / np.linalg.norm(
                              torch_output.detach().numpy())
    print("Difference: ", diff)
    if queue is not None:
        # we are testing
        queue.put(diff)
    else:
        assert diff < 1e-6
    del dace_model, ptmodel, x


def test():
    '''
    Evaluates multiple combination of input size/vecwidth
    '''
    print("----------- Testing Relu ---------------")
    vec_width = [1, 1, 2, 4]
    data_shapes = [(4, 8, 16), (100, 4, 16, 32), (8, 16, 16),
                   (1000, 4, 32, 32)]
    for i in range(0, len(vec_width)):
        print(
            "###############################################################")
        print(
            f"# Configuration: vw={vec_width[i]}, data_shape={data_shapes[i]}")
        print(
            "###############################################################")
        queue = Queue()
        p = Process(target=run, args=(data_shapes[i], vec_width[i], queue))
        p.start()
        p.join()
        assert (queue.get() < 1e-6)
    print("Success!")


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
        run((1000, 4, 32, 32), vec_width)
