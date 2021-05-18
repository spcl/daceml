# MaxPool expansion, simple testing

# TODO: add more testing

import torch
import torch.nn as nn
import torch.nn.functional as F
import dace
import pytest
import numpy as np
from dace.transformation.interstate import FPGATransformSDFG, InlineSDFG
from daceml.util import utils

import daceml.onnx as donnx
from daceml.pytorch import DaceModule, dace_module
import copy
import argparse
from multiprocessing import Process, Queue


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x):
        return F.max_pool2d(x, 2)


def run(data_shape: tuple, vec_width=1, queue=None):
    '''
    Evaluates specific configurations
    :param data_shape:
    :param vec_width:
    :param queue:
    :return:
    '''

    ptmodel = Model()
    x = torch.rand(data_shape)

    dace_model = DaceModule(ptmodel, auto_optimize=False)
    import daceml.onnx as donnx
    with dace.library.change_default(donnx.ONNXMaxPool, "pure"):
        dace_output = dace_model(x)
    torch_output = ptmodel(x)

    # Transform to FPGA
    sdfg = dace_model.sdfg

    ##################################
    # Vectorize container

    # find the input node, for the moment being maxpool writes only to non vectorized containers
    vec_type = dace.vector(dace.float32, vec_width)
    utils.vectorize_array_and_memlet(sdfg, "ONNX_0", vec_type)

    ##########################################

    with dace.library.change_default(donnx.ONNXMaxPool, "fpga"):
        sdfg.apply_transformations([FPGATransformSDFG])
        sdfg.expand_library_nodes()
        sdfg.apply_transformations_repeated([InlineSDFG])
        sdfg.compile()

    dace_output_fpga = dace_model(torch.clone(x))
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


@pytest.mark.fpga
def test():
    '''
       TODO: add more testing
    '''
    data_shape = (1000, 6, 32, 32)
    # Multiprocess is needed for testing otherwise Intel Compiler mess up with threads
    queue = Queue()
    p = Process(target=run, args=(data_shape, 1, queue))
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
        data_shape = (1000, 6, 32, 32)
        run(data_shape, vec_width)
