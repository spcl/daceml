# Testing the second portion of lenet: gemm->relu->Gemm->Relu->Gemm->softmax
# Relu writes back plain da types



from dace.transformation.interstate import FPGATransformSDFG


import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

import daceml.onnx as donnx
import dace
from daceml.pytorch import DaceModule, dace_module
import copy

from daceml.util import utils
from dace.transformation.dataflow import streaming_memory as sm
from dace.transformation.dataflow import PruneConnectors
from dace.transformation.interstate import InlineSDFG
from daceml.transformation import InputToConstant
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
            self.fc2.weight.data.fill_(0.1)
            self.fc2.bias.data.fill_(1)
            self.fc3.weight.data.fill_(0.1)
            self.fc3.bias.data.fill_(1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = F.softmax(x, dim=1)
        return x

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
    # vec_width = args["W"]
    input_to_constant = args["input_to_constant"]
    streaming = args["streaming"]


    import daceml.onnx as donnx
    donnx.default_implementation = "pure"
    donnx.ONNXConv.default_implementation = 'im2col'

    ptmodel = Model(input_to_constant)

    x = torch.rand(1000, 256)

    # build the DaCe model from the pytorch model
    dace_model = DaceModule(ptmodel)

    dace_output = dace_model(x)

    torch_output = ptmodel(x)
    # dace_model.sdfg.expand_library_nodes()
    dace_model.sdfg.save('/tmp/out.sdfg')
    diff = np.linalg.norm(torch_output.detach().numpy() - dace_output) / dace_output.size
    print("CPU Difference: ", diff)
    assert diff <=1e-06

    ############################################################
    # Transform to FPGA
    #
    sdfg = dace_model.sdfg

    ##################################
    # Vectorize GEMM output container
    vec_type = dace.vector(dace.float32, 8)

    # Also the first GEMM can be vect by 8
    # but the corresponding BIAS is not vectorized to not break input to consntat
    # utils.vectorize_array_and_memlet(sdfg, "ONNX_7", vec_type)

    # GEMM 10 is instead vectorized by 4
    vec_type4 = dace.vector(dace.float32, 4)
    # utils.vectorize_array_and_memlet(sdfg, "ONNX_9", vec_type4)
    # vec_type2 = dace.vector(dace.float32, 2)
    # utils.vectorize_array_and_memlet(sdfg, "ONNX_11", vec_type2)

    sdfg.save('/tmp/out.sdfg')


    ###################################
    # Apply transformations
    donnx.ONNXGemm.default_implementation = "fpga"
    donnx.ONNXRelu.default_implementation = "fpga"
    donnx.ONNXSoftmax.default_implementation = 'fpga'

    sdfg.apply_transformations([FPGATransformSDFG])
    sdfg.expand_library_nodes()
    sdfg.apply_transformations_repeated([InlineSDFG])

    if input_to_constant:
        sdfg.apply_transformations_repeated([InputToConstant],
                                            print_report=True)

    sdfg.save('/tmp/out_fpga_expanded.sdfg')

    # Streaming transformation
    if streaming:
        sdfg.apply_transformations_repeated([InlineSDFG, sm.StreamingComposition],
                                        [{}, {"storage": dace.StorageType.FPGA_Local}])

    sdfg.apply_transformations_repeated(PruneConnectors)


    sdfg.save('/tmp/out_fpga_expanded.sdfg')
    dace_output_fpga = dace_model(torch.clone(x))

    #reshape if vec_width is different than 1
    dace_output_fpga= dace_output_fpga.reshape(dace_output.shape)


    torch_output_numpy = torch_output.detach().numpy()
    diff =  np.linalg.norm(torch_output.detach().numpy()-dace_output_fpga)/dace_output_fpga.size
    print("Difference: ", diff)

    assert diff < 1e-6
