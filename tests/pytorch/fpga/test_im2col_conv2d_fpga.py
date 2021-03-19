# Tests for evaluating 2D convolutions for FPGA

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
from multiprocessing import Process, Queue

import daceml.onnx as donnx
donnx.default_implementation = "pure"
donnx.ONNXConv.default_implementation = 'im2col'


class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 input_to_constant):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size)
        if input_to_constant:
            #fix the weight otherwise everytime they are randomized
            self.conv.weight.data.fill_(0.1)
            self.conv.bias.data.fill_(1)

    def forward(self, x):
        return self.conv(x)


def evaluate(in_channels,
             out_channels,
             kernel_size,
             vec_width,
             data_shape: tuple,
             input_to_constant: bool,
             execute_cpu_dace: bool = False,
             queue=None):
    '''
    This function is used to evaluate a given model.
    It will build the pytorch model, transform it to a DaCe Model, apply transformation and execute on FPGA
    :return: returns if the result is correct
    '''
    # create pytorch model
    ptmodel = Model(in_channels, out_channels, kernel_size, input_to_constant)

    #create data
    x = torch.rand(data_shape)

    #evaluate pytorch model
    torch_output = ptmodel(x)

    #create dace model
    dace_model = DaceModule(ptmodel, dummy_inputs=x)

    if execute_cpu_dace:
        dace_output = dace_model(x)
        dace_model.sdfg.save('/tmp/out.sdfg')

    sdfg = dace_model.sdfg
    ##################################
    # Vectorize input and output container
    vec_type = dace.vector(dace.float32, vec_width)
    # utils.vectorize_array_and_memlet(sdfg, "fpga_ONNX_input", vec_type)
    utils.vectorize_array_and_memlet(sdfg, "ONNX_3", vec_type)
    sdfg.save("/tmp/out.sdfg")

    ###################################################
    # Transform for FPGA and Inline
    donnx.ONNXConv.default_implementation = "fpga"
    sdfg.apply_transformations([FPGATransformSDFG])

    ###################################
    sdfg.expand_library_nodes()
    sdfg.apply_transformations_repeated([InlineSDFG])

    # ###################################################################
    # # Input to constant
    if input_to_constant:
        sdfg.apply_transformations_repeated([InputToConstant],
                                            print_report=True)

    #################################
    # Execute
    sdfg.save("/tmp/out_fpga.sdfg")
    dace_output_fpga = dace_model(torch.clone(x))
    dace_output_fpga = dace_output_fpga.detach().numpy().reshape(torch_output.shape)

    diff = np.linalg.norm(torch_output.detach().numpy() -
                          dace_output_fpga) / np.linalg.norm(torch_output.detach().numpy())
    print("Difference: ", diff)
    if queue is not None:
        # we are testing
        queue.put(diff)
    else:
        assert (diff < 1e-6)

    del dace_model, ptmodel, x


def run(input_to_constant):
    '''
    Execute the program, in hardware if required, with a fixed input size
    :return:
    '''
    #evaluate(6, 16, 5, 4, (1000, 6, 12, 12), input_to_constant, False)
    #second conv
    evaluate(1, 6, 5, 1, (100, 1, 28, 28), input_to_constant, False)


def test(input_to_constant):
    '''
    Evaluates multiple combination of Convolution/input size
    :return:
    '''
    print("----------- Testing Convolution ---------------")

    # Run FPGA tests in a different process to avoid issues with Intel OpenCL tools
    # (But not in parallel)

    ####
    # No vect
    queue = Queue()
    p = Process(target=evaluate,
                args=(1, 6, 5, 1, (100, 1, 28, 28), input_to_constant, False,
                      queue))
    p.start()
    p.join()
    assert (queue.get() < 1e-6)

    p = Process(target=evaluate,
                args=(10, 1, 5, 1, (100, 10, 20, 20), input_to_constant, False,
                      queue))
    p.start()
    p.join()
    assert (queue.get() < 1e-6)

    p = Process(target=evaluate,
                args=(14, 8, 3, 1, (100, 14, 20, 20), input_to_constant, False,
                      queue))
    p.start()
    p.join()
    assert (queue.get() < 1e-6)

    # With Vectorization
    # The first two are from Lenet
    p = Process(target=evaluate,
                args=(1, 6, 5, 8, (100, 1, 28, 28), input_to_constant, False,
                      queue))
    p.start()
    p.join()
    assert (queue.get() < 1e-6)

    p = Process(target=evaluate,
                args=(6, 16, 5, 8, (100, 6, 12, 12), input_to_constant, False,
                      queue))
    p.start()
    p.join()
    assert (queue.get() < 1e-6)

    p = Process(target=evaluate,
                args=(6, 4, 5, 4, (100, 6, 12, 12), input_to_constant, False,
                      queue))
    p.start()
    p.join()
    assert (queue.get() < 1e-6)

    p = Process(target=evaluate,
                args=(3, 3, 3, 16, (100, 3, 34, 34), input_to_constant, False,
                      queue))
    p.start()
    p.join()
    assert (queue.get() < 1e-6)

    print("----------- Success! ---------------")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-input_to_constant",
                        action="store_true",
                        default=False,
                        help="Apply InputToConstant")

    parser.add_argument("-test",
                        action="store_true",
                        default=False,
                        help="Perform tests (USE ONLY WITH EMULATION)")

    args = vars(parser.parse_args())
    input_to_constant = args["input_to_constant"]
    t = args["test"]

    if t:
        test(input_to_constant)
    else:
        run(input_to_constant)
