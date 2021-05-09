import torch
from dace.transformation.dataflow import RedundantSecondArray

import daceml.onnx as donnx
from daceml.onnx.op_implementations.cudnn_implementations import CudnnConvolution
from daceml.transformation import ConstantFolding

import urllib.request
urllib.request.urlretrieve('http://spclstorage.inf.ethz.ch/~talbn/efficientnet-b0-block.onnx', 'efficientnet-b0-block.onnx')

donnx.default_implementation = "pure"
donnx.ONNXConv.default_implementation = "cuDNN"
donnx.ONNXBatchNormalization.default_implementation = "cuDNN"

inputs = torch.rand(8, 32, 224, 224).cuda()

import onnx
onnx_model = onnx.load('efficientnet-b0-block.onnx')
dace_model = donnx.ONNXModel("mbconv", onnx_model, cuda=True)

dace_model.sdfg.view()
dace_model.sdfg.apply_transformations_repeated(
        {
            ConstantFolding, RedundantSecondArray,
        },
        validate_all=True,
        strict=True)
dace_model.sdfg.view()

CudnnConvolution.default_algorithm = "gemm"

dace_model(inputs)

# def fuse_and_vec(model: DaceModule):
#    model.sdfg.apply_transformations_repeated(MapFusion, validate=True)
#    model.sdfg.apply_transformations_repeated(TaskletFusion, validate=True)
#    state = model.sdfg.node(0)
#
#    def apply_vectorization_to_map_following_access(data):
#        access = [n for n in state.nodes() if isinstance(n, nd.AccessNode) and n.data == data][0]
#        map_e = state.out_edges(access)[0].dst
#        tasklet = state.out_edges(map_e)[0].dst
#        map_x = state.exit_node(map_e)
#        Vectorization.apply_to(model.sdfg, _map_entry=map_e, _tasklet=tasklet, _map_exit=map_x)
#
#    apply_vectorization_to_map_following_access("ONNX_99")
#    apply_vectorization_to_map_following_access("ONNX_111")
#
