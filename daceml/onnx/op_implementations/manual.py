import functools

from dace import dtypes
import os

from dace.codegen.codeobject import CodeObject
from dace.codegen.targets import CUDACodeGen

from daceml.onnx.op_implementations import op_implementation

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

