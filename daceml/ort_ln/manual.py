import functools
from typing import Dict

from dace import dtypes, nodes, registry
import os

from dace.codegen.codeobject import CodeObject
from dace.transformation import transformation
from dace.codegen.targets import CUDACodeGen
from dace.properties import Property, make_properties
from dace.transformation.transformation import ExpandTransformation, PatternNode
from daceml.onnx.nodes import onnx_op
from dace.sdfg import utils as sdutil

import dace.library

include_dir = os.path.join(os.path.dirname(__file__), "manual")

with open(os.path.join(include_dir, "layer_norm_impl.cu")) as f:
    layer_norm_cu_code = f.read()

with open(os.path.join(include_dir, "layer_norm_impl_bwd.cu")) as f:
    layer_norm_cu_bwd_code = f.read()

@dace.library.environment
class LayerNormEnvironment:
    cmake_minimum_version = None
    cmake_packages = []
    cmake_variables = {}
    cmake_includes = [include_dir]
    cmake_libraries = []
    cmake_compile_flags = []
    cmake_link_flags = []
    cmake_files = []
    state_fields = []
    dependencies = []
    codeobjects = [CodeObject("layer_norm_impl",
                              layer_norm_cu_code,
                              'cu',
                              CUDACodeGen,
                              "CUDA",
                              target_type=""),
                   CodeObject("layer_norm_impl_bwd",
                              layer_norm_cu_bwd_code,
                              'cu',
                              CUDACodeGen,
                              "CUDA",
                              target_type="")
                   ]
    headers = ["layer_norm_impl.h", "layer_norm_impl_bwd.h"]
    init_code = ""
    finalize_code = ""

@dace.library.expansion
class ExpandLNFwd(ExpandTransformation):

    environments = [LayerNormEnvironment]

    @staticmethod
    def expansion(node, parent_state: dace.SDFGState, parent_sdfg: dace.SDFG, **kwargs):
        shape = parent_sdfg.arrays[list(parent_state.in_edges_by_connector(node, "_X"))[0].data.data].shape
        return add_ln_tasklet(parent_state, shape, node.axis)

@dace.library.node
class LayerNorm(nodes.LibraryNode):

    # Global properties
    implementations = {
        "ORT": ExpandLNFwd,
    }
    default_implementation = "ORT"

    axis = Property(dtype=int)

    def __init__(self, name, axis, location=None):
        in_connectors = {"_X", "_scale", "_bias"}
        out_connectors = {"_Y", "_mean", "_inv_std_var"}
        super().__init__(name,
                         location=location,
                         inputs=in_connectors,
                         outputs=out_connectors)
        self.axis = axis

@registry.autoregister_params(singlestate=True)
@make_properties
class DetectLN(transformation.Transformation):

    input = PatternNode(nodes.AccessNode)
    reduce_mean = PatternNode(onnx_op.ONNXReduceMean)
    onnx_3 = PatternNode(nodes.AccessNode)
    sub = PatternNode(onnx_op.ONNXSub)
    onnx_4 = PatternNode(nodes.AccessNode)
    pow = PatternNode(onnx_op.ONNXPow)
    onnx_6 = PatternNode(nodes.AccessNode)
    reduce_mean_2 = PatternNode(onnx_op.ONNXReduceMean)
    onnx_7 = PatternNode(nodes.AccessNode)
    add = PatternNode(onnx_op.ONNXAdd)
    onnx_9 = PatternNode(nodes.AccessNode)
    sqrt = PatternNode(onnx_op.ONNXSqrt)
    onnx_10 = PatternNode(nodes.AccessNode)
    div = PatternNode(onnx_op.ONNXDiv)
    onnx_11 = PatternNode(nodes.AccessNode)
    mul = PatternNode(onnx_op.ONNXMul)
    onnx_12 = PatternNode(nodes.AccessNode)
    add_2 = PatternNode(onnx_op.ONNXAdd)
    output = PatternNode(nodes.AccessNode)
    weight = PatternNode(nodes.AccessNode)
    bias = PatternNode(nodes.AccessNode)
    pow_data = PatternNode(nodes.AccessNode)
    add_data = PatternNode(nodes.AccessNode)

    @staticmethod
    def expressions():
        graph_nodes = [DetectLN.input,
                       DetectLN.reduce_mean,
                       DetectLN.onnx_3,
                       DetectLN.sub,
                       DetectLN.onnx_4,
                       DetectLN.pow,
                       DetectLN.onnx_6,
                       DetectLN.reduce_mean_2,
                       DetectLN.onnx_7,
                       DetectLN.add,
                       DetectLN.onnx_9,
                       DetectLN.sqrt,
                       DetectLN.onnx_10,
                       DetectLN.div,
                       DetectLN.onnx_11,
                       DetectLN.mul,
                       DetectLN.onnx_12,
                       DetectLN.add_2,
                       DetectLN.output]



        graph = sdutil.node_path_graph(*graph_nodes)
        graph.add_edge(DetectLN.onnx_4, DetectLN.div, None)

        graph.add_node(DetectLN.pow_data)
        graph.add_node(DetectLN.add_data)
        graph.add_node(DetectLN.weight)
        graph.add_node(DetectLN.bias)

        graph.add_edge(DetectLN.input, DetectLN.sub, None)
        graph.add_edge(DetectLN.add_data, DetectLN.add, None)
        graph.add_edge(DetectLN.pow_data, DetectLN.pow, None)
        graph.add_edge(DetectLN.weight, DetectLN.mul, None)
        graph.add_edge(DetectLN.bias, DetectLN.add_2, None)

        return [graph]

    @staticmethod
    def can_be_applied(graph: dace.sdfg.graph.OrderedMultiDiConnectorGraph,
                       candidate: Dict[nodes.Node, int],
                       expr_index: int,
                       sdfg,
                       strict: bool = False):
        # if isinstance(DetectLN.pattern_reduce_mean, onnx_op.ONNXReduce)
        return True

    def apply(self, sdfg: dace.SDFG):
        state = sdfg.nodes()[self.state_id]

        axis = self.reduce_mean(sdfg).axes[0]
        input_shape = self.input(sdfg).desc(sdfg).shape

        input = self.input(sdfg)
        output = self.output(sdfg)
        weight = self.weight(sdfg)
        bias = self.bias(sdfg)


        nodes_to_kill = [self.reduce_mean(sdfg),
                         self.onnx_3(sdfg),
                         self.sub(sdfg),
                         self.onnx_4(sdfg),
                         self.pow(sdfg),
                         self.onnx_6(sdfg),
                         self.reduce_mean_2(sdfg),
                         self.onnx_7(sdfg),
                         self.add(sdfg),
                         self.onnx_9(sdfg),
                         self.sqrt(sdfg),
                         self.onnx_10(sdfg),
                         self.div(sdfg),
                         self.onnx_11(sdfg),
                         self.mul(sdfg),
                         self.onnx_12(sdfg),
                         self.add_2(sdfg),
                         self.pow_data(sdfg),
                         self.add_data(sdfg)]

        for n in nodes_to_kill:
            state.remove_node(n)

        ln_node = LayerNorm("detected_layernorm", axis)
        state.add_node(ln_node)

        state.add_edge(input, None, ln_node, "_X", sdfg.make_array_memlet(input.data))
        state.add_edge(weight, None, ln_node, "_scale", sdfg.make_array_memlet(weight.data))
        state.add_edge(bias, None, ln_node, "_bias", sdfg.make_array_memlet(bias.data))

        state.add_edge(ln_node, "_Y", output, None, sdfg.make_array_memlet(output.data))
        mean_name, _ = sdfg.add_array("detected_layernorm_mean", [input_shape[axis]], dace.float32, storage=dtypes.StorageType.GPU_Global, transient=True, find_new_name=True)
        std_name, _ = sdfg.add_array("detected_layernorm_std", [input_shape[axis]], dace.float32, storage=dtypes.StorageType.GPU_Global, transient=True, find_new_name=True)
        state.add_edge(ln_node, "_mean", state.add_write(mean_name), None, sdfg.make_array_memlet(mean_name))
        state.add_edge(ln_node, "_inv_std_var", state.add_write(std_name), None, sdfg.make_array_memlet(std_name))






def _prod(sequence):
    return functools.reduce(lambda a, b: a * b, sequence, 1)


def add_ln_tasklet_bwd(state: dace.SDFGState, input_shape, axis):

    n1 = _prod(input_shape[:axis])
    n2 = _prod(input_shape[axis:])
    code = f"""
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    float *scratch_1;
    float *scratch_2;
    cudaMalloc(&scratch_1, {n2} * 16 * 4);
    cudaMalloc(&scratch_2, {n2} * 16 * 4);
    onnxruntime::cuda::HostLayerNormGradient<float, float, false>(
        prop,
        __dace_current_stream,
        _dY,
        _X,
        nullptr,
        _scale,
        nullptr,
        _mean,
        _inv_std_var,
        {n1},
        {n2},
        _dX,
        _dscale,
        _dbias,
        scratch_1,
        scratch_2,
        16);
    cudaFree(scratch_1);
    cudaFree(scratch_2);
    """
    in_connectors = {"_dY", "_X", "_scale", "_bias", "_inv_std_var", "_mean"}
    out_connectors = {"_dX", "_dscale", "_dbias"}
    tasklet = state.add_tasklet("layernorm",
                                {i: dace.pointer(dace.float32)
                                 for i in in_connectors},
                                {o: dace.pointer(dace.float32)
                                 for o in out_connectors}, code, dtypes.Language.CPP)
    tasklet.environments = {"LayerNormEnvironment"}
    return tasklet


def add_ln_tasklet(state: dace.SDFGState, input_shape, axis):

    n1 = _prod(input_shape[:axis])
    n2 = _prod(input_shape[axis:])
    code = f"""
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    onnxruntime::contrib::cuda::HostApplyLayerNorm<float, float, false>(
        prop,
        __dace_current_stream,
        _Y,
        _mean,
        _inv_std_var,
        _X,
        {n1},
        {n2},
        {1e-5},
        _scale,
        _bias);
    """
    in_connectors = {"_X", "_scale", "_bias"}
    out_connectors = {"_Y", "_mean", "_inv_std_var"}
    tasklet = state.add_tasklet("layernorm",
                                 {i: dace.pointer(dace.float32)
                                  for i in in_connectors},
                                 {o: dace.pointer(dace.float32)
                                  for o in out_connectors}, code, dtypes.Language.CPP)
    tasklet.environments = {"LayerNormEnvironment"}
    return tasklet

