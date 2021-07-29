#!/usr/bin/env python3

# ----------------------------------------
# VGG-16 Optimization Example
# ----------------------------------------
from dace.transformation.interstate import FPGATransformSDFG, InlineSDFG
from daceml.transformation import InputToConstant

import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import os

import numpy as np

import daceml.onnx as donnx
import dace
from daceml.pytorch import DaceModule, dace_module
import copy

from daceml.util import utils
from dace.transformation.dataflow import streaming_memory as sm
from dace.transformation.dataflow import PruneConnectors
from dace.transformation.interstate import InlineSDFG
from dace.sdfg import SDFG, SDFGState, trace_nested_access
from dace import config, data as dt, dtypes, Memlet, symbolic
from dace.sdfg import nodes as nd

from dace.transformation.optimizer import Optimizer
from dace.transformation.dataflow import streaming_memory as sm

import argparse

# Random numbers
import random

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

# ----------------------------------------
# Node Mappings
# ----------------------------------------
access_node_block_mapping = {
    0: [27, 28, 29, 30],
    1: [31, 32, 33, 34, 35],
    2: [36, 37, 38, 39, 40, 41, 42],
    3: [43, 44, 45, 46, 47, 48, 49],
    4: [50, 51, 52, 53, 54, 55, 56],
}

conv_mapping = {
    0: [0, 2],
    1: [5, 7],
    2: [10, 12, 14],
    3: [17, 19, 21],
    4: [24, 26, 28]
}

# ----------------------------------------
# Operator configurations
# ----------------------------------------
net_config = {
    "blocks": [{
        "pe": 4,
        "tile": 1120,
        "vec": 16
    }, {
        "pe": 4,
        "tile": 1120,
        "vec": 16
    }, {
        "pe": 4,
        "tile": 1120,
        "vec": 8
    }, {
        "pe": 8,
        "tile": 784,
        "vec": 4
    }, {
        "pe": 8,
        "tile": 196,
        "vec": 2
    }]
}


# ----------------------------------------
# Memory bank RR-interleaving
# ----------------------------------------
def fpga_rr_interleave_containers_to_banks(sdfg: SDFG, num_banks: int = 4):
    '''
    Allocates the (global) arrays to FPGA off-chip memory banks, interleaving them in a
    Round-Robin (RR) fashion. This applies to all the arrays in the SDFG hierarchy.
    :param sdfg: The SDFG to operate on.
    :param: num_banks: number of off-chip memory banks to consider
    :returns: a list containing  the number of (transient) arrays allocated to each bank
    :note: Operates in-place on the SDFG.
    '''

    # keep track of memory allocated to each bank
    num_allocated = [0 for i in range(num_banks)]

    i = 0
    for sd, aname, desc in sdfg.arrays_recursive():
        if not isinstance(
                desc, dt.Stream
        ) and desc.storage == dtypes.StorageType.FPGA_Global and desc.transient:
            desc.location["bank"] = i % num_banks
            num_allocated[i % num_banks] = num_allocated[i % num_banks] + 1
            i = i + 1

    return num_allocated


# def expand_library_nodes_kwargs(sdfg, recursive=True, **kwargs):
#     """
#     Recursively expand all unexpanded library nodes in the SDFG,
#     resulting in a "pure" SDFG that the code generator can handle.
#     :param recursive: If True, expands all library nodes recursively,
#                         including library nodes that expand to library nodes.
#     """

#     states = list(sdfg.states())
#     while len(states) > 0:
#         state = states.pop()
#         expanded_something = False
#         for node in list(state.nodes()):  # Make sure we have a copy
#             if isinstance(node, nd.NestedSDFG):
#                 expand_library_nodes_kwargs(node.sdfg, **kwargs)  # Call recursively
#             elif isinstance(node, nd.LibraryNode):
#                 # if str(node) == "ONNX_Conv_3":
#                 #     donnx.ONNXConv.default_implementation = "fpga"
#                 #     impl_name = node.expand(sdfg, state)
#                 if isinstance(node, donnx.ONNXConv):
#                     impl_name = node.expand(sdfg, state, **kwargs)
#                 else:
#                     impl_name = node.expand(sdfg, state)
#                 print(
#                     "Automatically expanded library node \"{}\" with implementation \"{}\"."
#                     .format(str(node), impl_name))
#                 # We made a copy of the original list of nodes, so we keep
#                 # iterating even though this list has now changed
#                 if recursive:
#                     expanded_something = True
#         if expanded_something:
#             states.append(state)  # Nodes have changed. Check state again


# ----------------------------------------
# ReLU node removal
# ----------------------------------------
def remove_relu_post_conv(sdfg, state, conv_node):

    # get nodes
    conv_result_node = state.successors(conv_node)[0]
    relu_node = state.successors(conv_result_node)[0]
    relu_result_node = state.successors(relu_node)[0]

    print(f"remove relu {relu_node}")

    # get target edge and properties
    conv_node_out = state.out_edges(conv_node)[0]
    conv_node_out_data = conv_node_out._data
    conv_node_out_src_conn = conv_node_out._src_conn

    # remove relu and respective edges
    state.remove_node(conv_result_node)
    sdfg.remove_data(str(conv_result_node))
    state.remove_node(relu_node)

    # update access node of target memlet
    conv_node_out_data._data = str(relu_result_node)

    # add new edge
    new_edge = state.add_edge(conv_node, conv_node_out_src_conn,
                              relu_result_node, None, conv_node_out_data)
    # update memlet with new edge
    conv_node_out_data._edge = new_edge


# ----------------------------------------
# Torch Model
# ----------------------------------------
class Model(nn.Module):
    def __init__(self, small=False):
        super(Model, self).__init__()

        self.small = small
        div = 64 if small else 1
        div2 = 256 if small else 1

        # block 1
        self.conv1 = nn.Conv2d(3, 64 // div, 3, padding=1)
        self.conv2 = nn.Conv2d(64 // div, 64 // div, 3, padding=1)

        # block 2
        self.conv3 = nn.Conv2d(64 // div, 128 // div, 3, padding=1)
        self.conv4 = nn.Conv2d(128 // div, 128 // div, 3, padding=1)

        # block 3
        self.conv5 = nn.Conv2d(128 // div, 256 // div2, 3, padding=1)
        self.conv6 = nn.Conv2d(256 // div2, 256 // div2, 3, padding=1)
        self.conv7 = nn.Conv2d(256 // div2, 256 // div2, 3, padding=1)

        # block 4
        self.conv8 = nn.Conv2d(256 // div2, 512 // div2, 3, padding=1)
        self.conv9 = nn.Conv2d(512 // div2, 512 // div2, 3, padding=1)
        self.conv10 = nn.Conv2d(512 // div2, 512 // div2, 3, padding=1)

        # block 5
        self.conv11 = nn.Conv2d(512 // div2, 512 // div2, 3, padding=1)
        self.conv12 = nn.Conv2d(512 // div2, 512 // div2, 3, padding=1)
        self.conv13 = nn.Conv2d(512 // div2, 512 // div2, 3, padding=1)

    def forward(self, x):

        # block 1
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)

        # block 2
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(x, 2)

        # block 3
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))
        x = F.max_pool2d(x, 2)

        # block 4
        x = F.relu(self.conv8(x))
        x = F.relu(self.conv9(x))
        x = F.relu(self.conv10(x))
        x = F.max_pool2d(x, 2)

        # block 5
        x = F.relu(self.conv11(x))
        x = F.relu(self.conv12(x))
        x = F.relu(self.conv13(x))
        x = F.max_pool2d(x, 2)

        return x


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    def aa(*args, **kwargs):
        group.add_argument(*args, **kwargs)

    group = parser.add_argument_group("Model configuration")
    aa("--small",
       dest="small",
       action="store_true",
       help="Use reduced number of filters")

    group = parser.add_argument_group("Runtime and optimization options")
    aa("-b", "--batch", dest="batch", type=int, default=1, help="Batch Size")
    aa("--relu_fuse",
       dest="relu_fuse",
       action="store_true",
       help="Apply ReLU Fusion Optimization")
    aa("--mem_banking",
       dest="mem_banking",
       action="store_true",
       help="Apply RR memory banking")
    aa("--vectorize",
       dest="vectorize",
       action="store_true",
       help="Apply vectorization according to config")

    # get command line arguments
    args = vars(parser.parse_args())

    # set default implementations for operators
    dace.library.change_default(donnx.ONNXConv, "pure")

    # instantiate Torch model
    ptmodel = Model(small=args["small"])

    # create random test data
    size = 224  # default ImageNet
    data_shape = (args["batch"], 3, size, size)
    x = torch.rand(data_shape)

    print(f"\n-------------- Run Torch Model --------------")
    torch_output = ptmodel(x)

    # Convert to SDFG
    dace_model = DaceModule(ptmodel,
                            sdfg_name="vgg16",
                            dummy_inputs=x,
                            auto_optimize=False)

    donnx.ONNXRelu.default_implementation = "fpga"
    donnx.ONNXMaxPool.default_implementation = "fpga"
    donnx.ONNXConv.default_implementation = "fpga_tiled"
    donnx.ONNXReshape.default_implementation = "fpga"

    def TransformToFPGA(dace_module):
        '''
        Transforms the given module to run on FPGA.
        This includes vectorization and library node expansions.
        :param dace_module:
        :return:
        '''

        # get SDFG
        sdfg = dace_module.sdfg
        sdfg.apply_transformations([FPGATransformSDFG, InlineSDFG])

        # Apply vectorization according to configuration
        if args["vectorize"]:
            print(f"-------------- Apply vectorization --------------")
            mapping = access_node_block_mapping
            for block in range(5):

                print(f"--> Block {block} configuration:",
                      net_config["blocks"][block])
                vec_width = net_config["blocks"][block]["vec"]
                vec_width = math.gcd(max(1, int(size / (2**block))), vec_width)

                ##################################
                # Vectorize input and output container
                vec_type = dace.vector(dace.float32, vec_width)

                # vectorize input
                if block == 0:
                    print(f"Vectorize ONNX_inputDOT1 with {vec_width}")
                    utils.vectorize_array_and_memlet(sdfg, "inputDOT1",
                                                     vec_type)

                # # vectorize outputs of layers
                for i in mapping[block]:
                    print(f"Vectorize ONNX_{i} with {vec_width}")
                    utils.vectorize_array_and_memlet(sdfg, f"ONNX_{i}",
                                                     vec_type)

        # Expand library nodes according to configuration
        print(f"-------------- Library Node Expansions --------------")

        # Fused ReLU Activation configuration
        fuse_relu = False
        if args["relu_fuse"]:
            activation = "relu"
            fuse_relu = True
        else:
            activation = None

        # Expand convolutions according to configuration
        for block in range(5):

            print(f"-------------- BLOCK {block} CONVs --------------")

            # get configuration
            pe = net_config["blocks"][block]["pe"]
            tile = net_config["blocks"][block]["tile"]

            for i in conv_mapping[block]:

                # get library node
                node, state = utils.get_library_node_by_name(sdfg, f"Conv_{i}")
                if fuse_relu:
                    remove_relu_post_conv(sdfg, state, node)

                node.expand(sdfg,
                            state,
                            pe=int(pe),
                            tiles=int(tile),
                            activation=activation)

        sdfg.apply_transformations_repeated([InlineSDFG])

        # Expand remaining Library Nodes (Pooling and Activations)
        print(f"-------------- REST (ReLu, MaxPool) --------------")
        sdfg.expand_library_nodes(sdfg)
        sdfg.apply_transformations_repeated([InlineSDFG])

        if args["mem_banking"]:
            print(f"-------------- RR memory banking --------------")
            alloc = fpga_rr_interleave_containers_to_banks(sdfg, 4)
            print("Allocations:", alloc)

        # cleanup SDFG
        sdfg.apply_transformations_repeated([InlineSDFG])
        print(f"-------------- Transformation DONE --------------\n")

    print(f"\n-------------- Run and test for FPGA --------------")
    # Reset the SDFG
    dace_model.reset_sdfg()
    # Append transformation hook
    dace_model.append_post_onnx_hook("TransformToFPGA", TransformToFPGA)

    # Execute Module with FPGA expansion
    with dace.library.change_default(donnx.ONNXConv, "fpga_tiled"):
        dace_output_fpga = dace_model(torch.clone(x))

    dace_output_fpga = dace_output_fpga.detach().numpy().reshape(
        torch_output.shape)

    torch_output_numpy = torch_output.detach().numpy()

    # perform additional activation if operator was
    # configured to perform activation before writing result
    if args["relu_fuse"]:
        torch_output_numpy = np.maximum(0, torch_output_numpy)

    diff = np.linalg.norm(torch_output_numpy - dace_output_fpga
                          ) / np.linalg.norm(torch_output_numpy)
    print("Difference: ", diff)
    assert (diff < 1e-6)
