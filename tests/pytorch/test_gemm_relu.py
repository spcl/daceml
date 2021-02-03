# Simple test for evaluating a composition Gemm  -> relu.
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
import onnx
from daceml.onnx import ONNXModel




class Model(nn.Module):
    def __init__(self, input_to_constant):
        super(Model, self).__init__()
        self.fc = nn.Linear(256, 120)
        if input_to_constant:
            #otherwise everytime they are randomized
            self.fc.weight.data.fill_(0.1)
            self.fc.bias.data.fill_(1)

    def forward(self, x):
        x = F.relu(self.fc(x))
        return x

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

    parser.add_argument("-streaming",
                        action="store_true",
                        default=False,
                        help="Apply Streaming Composition")

    parser.add_argument("--save_to_onnx",
                        type=str,
                        help="Save the model to the given onnx file")

    parser.add_argument("--load_from_onnx",
                        type=str,
                        help="Load the model from the given onnx file")

    args = vars(parser.parse_args())
    vec_width = args["W"]
    input_to_constant = args["input_to_constant"]
    streaming = args["streaming"]
    onnx_output = args["save_to_onnx"]
    onnx_input = args["load_from_onnx"]

    import daceml.onnx as donnx
    donnx.default_implementation = "pure"
    donnx.ONNXConv.default_implementation = 'im2col'

    ptmodel = Model(input_to_constant)

    x = torch.rand(1000, 256)

    if onnx_input is None:
        # build the DaCe model from the pytorch model
        dace_model = DaceModule(ptmodel)
    else:
        # load from file
        onnx_model = onnx.load(onnx_input)
        dace_model = ONNXModel("mymodel", onnx_model)
        print("Loaded from ONNX file")

    if onnx_output is not None:
        print("Saving to ONNX file")
        torch.onnx.export(
            ptmodel,
            x,
            onnx_output,
            verbose=True,
            input_names=['input'],  # the model's input names
            output_names=['output'],  # the model's output names
            dynamic_axes={
                'input': {
                    0: 'batch_size',
                    # 1: "input_channels",
                    # 2: "input_height",
                    # 3: "input_width"
                },  # variable lenght axes
                'output': {
                    0: 'batch_size',
                    # 1: "output_channels",
                    # 2: "output_height",
                    # 3: "output_width"

                }
            })

    dace_output = dace_model(x)

    torch_output = ptmodel(x)
    # dace_model.sdfg.expand_library_nodes()
    dace_model.sdfg.save('/tmp/out.sdfg')
    diff = np.linalg.norm(torch_output.detach().numpy() - dace_output) / dace_output.size
    print("CPU Difference: ", diff)
    assert np.allclose(torch_output.detach().numpy(), dace_output, atol=1e-06)

    ############################################################
    # Transform to FPGA
    #
    sdfg = dace_model.sdfg

    ##################################
    # Vectorize GEMM output container
    vec_type = dace.vector(dace.float32, vec_width)
    # output_data_name = sdfg.states()[0].sink_nodes()[0].data
    utils.vectorize_array_and_memlet(sdfg, "ONNX_3", vec_type)
    # But do not vectorize the ouput of Relu
    # vectorize output of Relu
    sdfg.save('/tmp/out.sdfg')


    ###################################
    # Apply transformations
    donnx.ONNXGemm.default_implementation = "fpga"
    donnx.ONNXRelu.default_implementation = "fpga"

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
