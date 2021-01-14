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
from multiprocessing import Process, Queue

import daceml.onnx as donnx
donnx.default_implementation = "pure"
donnx.ONNXConv.default_implementation = 'im2col'


class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size)

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
    ptmodel = Model(in_channels, out_channels, kernel_size)

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

    ###################################################
    # Transform for FPGA and Inline
    donnx.ONNXConv.default_implementation = "fpga"
    sdfg.apply_transformations([FPGATransformSDFG])
    sdfg.apply_transformations_repeated([InlineSDFG])

    ##################################
    # Vectorize input and output container
    vec_type = dace.vector(dace.float32, vec_width)
    utils.vectorize_array_and_memlet(sdfg, "fpga_ONNX_3", vec_type)

    ###################################
    sdfg.expand_library_nodes()
    sdfg.apply_transformations_repeated([InlineSDFG])

    # ###################################################################
    # # Input to constant
    if input_to_constant:
        sdfg.apply_transformations_repeated([InputToConstant],
                                            print_report=True)

    sdfg.save("/tmp/out_fpga.sdfg")
    #################################
    # Execute
    dace_output_fpga = dace_model(torch.clone(x))
    dace_output_fpga = dace_output_fpga.reshape(torch_output.shape)

    diff = np.linalg.norm(torch_output.detach().numpy() -
                          dace_output_fpga) / dace_output_fpga.size
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
    # Second Conv in Lenet
    evaluate(6, 16, 5, 8, (1000, 6, 12, 12), input_to_constant, False)
    # First Conv in lenet
    # evaluate(1, 6, 5, 1, (1000, 1, 28, 28), input_to_constant, False)


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
    #
    # ptmodel = Model(6, 16, 5)
    # data_shape = (1000, 6, 12, 12)
    #
    # x = torch.rand(data_shape)
    #
    # dace_model = DaceModule(ptmodel)
    # dace_output = dace_model(x)
    #
    # torch_output = ptmodel(x)
    # dace_model.sdfg.save('/tmp/out.sdfg')
    #
    # assert np.allclose(torch_output.detach().numpy(), dace_output, atol=1e-06)
    #
    # # Save sdfg to file
    # sdfg = dace_model.sdfg
    # orig_sdfg = copy.deepcopy(sdfg)
    # orig_sdfg.expand_library_nodes()
    # orig_sdfg.save('/tmp/out_expanded.sdfg')
    #
    # ###################################################
    # # Transform for FPGA and Inline
    # donnx.ONNXConv.default_implementation = "fpga"
    # sdfg.apply_transformations([FPGATransformSDFG])
    # sdfg.apply_transformations_repeated([InlineSDFG])
    #
    # ##################################
    # # Vectorize input and output container
    # vec_width = 8
    # vec_type = dace.vector(dace.float32, vec_width)
    # utils.vectorize_array_and_memlet(sdfg, "fpga_ONNX_3", vec_type)
    #
    # ###################################
    # sdfg.save('/tmp/out_vectorized.sdfg')
    # sdfg.expand_library_nodes()
    # sdfg.apply_transformations_repeated([InlineSDFG])
    #
    # # ###################################################################
    # # # Input to constant
    # if input_to_constant:
    #     sdfg.apply_transformations_repeated([InputToConstant],
    #                                         print_report=True)
    #
    # dace_output_fpga = dace_model(torch.clone(x))
    # dace_output_fpga = dace_output_fpga.reshape(dace_output.shape)
    #
    # print(
    #     "Difference: ",
    #     np.linalg.norm(torch_output.detach().numpy() - dace_output_fpga) /
    #     dace_output_fpga.size)
    #
    # torch_output_numpy = torch_output.detach().numpy()
    # diff = torch_output_numpy - dace_output_fpga
    #
    # assert np.allclose(torch_output.detach().numpy(), dace_output_fpga)
