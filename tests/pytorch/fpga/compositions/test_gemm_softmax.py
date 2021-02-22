# Simple test for gemm->softmax for FPGA, according to the last two lenet operators
# the GEMM ONNX operator is used when we use a fully connected layer

from dace.transformation.interstate import FPGATransformSDFG, InlineSDFG

import torch
import torch.nn as nn
import torch.nn.functional as F
from dace.transformation.dataflow import streaming_memory as sm

import numpy as np

import daceml.onnx as donnx
from daceml.pytorch import DaceModule, dace_module
from daceml.util import utils
from daceml.transformation import InputToConstant

import dace
import copy
import argparse


class Model(nn.Module):
    def __init__(self, input_to_constant):
        super(Model, self).__init__()
        self.fc = nn.Linear(84, 10)
        if input_to_constant:
            #otherwise everytime they are randomized
            self.fc.weight.data.fill_(0.1)
            self.fc.bias.data.fill_(1)

    def forward(self, x):
        x = F.softmax(self.fc(x), dim=1)
        return x


def test(input_to_constant, streaming):

    import daceml.onnx as donnx
    donnx.default_implementation = "pure"

    ptmodel = Model(input_to_constant)
    x = torch.rand(10000, 84, dtype=torch.float32)

    dace_model = DaceModule(ptmodel)
    dace_output = dace_model(x)

    torch_output = ptmodel(x)

    assert np.allclose(torch_output.detach().numpy(), dace_output, atol=1e-06)

    sdfg = dace_model.sdfg

    ##################################
    # Vectorize output container (in Lenet the input is not vectorized)
    # No vectorization here
    # vec_type = dace.vector(dace.float32, vec_width)
    # utils.vectorize_array_and_memlet(sdfg, "ONNX_7", vec_type)
    sdfg.save('/tmp/out.sdfg')

    ###################################################
    # Transform for FPGA and Inline
    donnx.ONNXGemm.default_implementation = "fpga"
    donnx.ONNXSoftmax.default_implementation = "fpga"

    sdfg.apply_transformations([FPGATransformSDFG])
    sdfg.expand_library_nodes()
    sdfg.apply_transformations_repeated([InlineSDFG])

    if input_to_constant:
        sdfg.apply_transformations_repeated([InputToConstant],
                                            print_report=True)

    if streaming:
        sdfg.apply_transformations_repeated(
            [InlineSDFG, sm.StreamingComposition],
            [{}, {
                "storage": dace.StorageType.FPGA_Local
            }])

    # one step beyond
    # sdfg.states()[0].nodes()[0].sdfg.states()[0].location["is_FPGA_kernel"] = False

    sdfg.save('/tmp/out_fpga.sdfg')

    dace_output_fpga = dace_model(torch.clone(x))
    # reshape if vec_width is different than 1
    dace_output_fpga = dace_output_fpga.reshape(dace_output.shape)

    diff = np.linalg.norm(torch_output.detach().numpy() -
                          dace_output_fpga) / dace_output_fpga.size
    print("Difference: ", diff)

    assert (diff < 1e-6)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-input_to_constant",
                        action="store_true",
                        default=False,
                        help="Apply InputToConstant")

    parser.add_argument("-streaming",
                        action="store_true",
                        default=False,
                        help="Apply Streaming Composition")

    args = vars(parser.parse_args())
    input_to_constant = args["input_to_constant"]
    streaming = args["streaming"]
    test(input_to_constant, streaming)
