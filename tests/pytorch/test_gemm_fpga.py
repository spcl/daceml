# Simple test for gemm for FPGA
# the GEMM ONNX operator is used when we use a fully connected layer

# TODO: conform to pytest syntax if needed

from dace.transformation.interstate import FPGATransformSDFG, InlineSDFG

import torch
import torch.nn as nn
import torch.nn.functional as F

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
        self.fc1 = nn.Linear(256, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        if input_to_constant:
            #otherwise everytime they are randomized
            self.fc1.weight.data.fill_(0.1)
            self.fc1.bias.data.fill_(1)

    def forward(self, x):
        # x = self.fc1(x)
        # x = self.fc2(x)
        return self.fc1(x)

def test(vec_width, input_to_constant):

    import daceml.onnx as donnx
    donnx.default_implementation = "pure"

    ptmodel = Model(input_to_constant)
    x = torch.rand(1000, 256, dtype=torch.float32)

    dace_model = DaceModule(ptmodel)
    dace_output = dace_model(x)

    torch_output = ptmodel(x)

    assert np.allclose(torch_output.detach().numpy(), dace_output, atol=1e-06)


    sdfg = dace_model.sdfg

    ##################################
    # Vectorize output container (in Lenet the input is not vectorized)
    vec_type = dace.vector(dace.float32, vec_width)
    utils.vectorize_array_and_memlet(sdfg, "ONNX_7", vec_type)
    sdfg.save('/tmp/out.sdfg')

    ###################################################
    # Transform for FPGA and Inline
    donnx.ONNXGemm.default_implementation = "fpga"
    sdfg.apply_transformations([FPGATransformSDFG])
    sdfg.expand_library_nodes()
    sdfg.apply_transformations_repeated([InlineSDFG])

    if input_to_constant:
        sdfg.apply_transformations_repeated([InputToConstant],
                                            print_report=True)


    # one step beyond
    # sdfg.states()[0].nodes()[0].sdfg.states()[0].location["is_FPGA_kernel"] = False

    sdfg.save('/tmp/out_fpga.sdfg')

    dace_output_fpga = dace_model(torch.clone(x))
    # reshape if vec_width is different than 1
    dace_output_fpga = dace_output_fpga.reshape(dace_output.shape)

    diff =  np.linalg.norm(torch_output.detach().numpy() - dace_output_fpga) /dace_output_fpga.size
    print("Difference: ", diff)

    assert(diff < 1e-6)




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
    test(vec_width, input_to_constant)
