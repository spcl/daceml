import functools
import os
from typing import List, Optional, Tuple

import dace.sdfg.nodes as nd
import dace.library
from dace import dtypes
from dace.codegen.codeobject import CodeObject
from dace.codegen.targets import CUDACodeGen
from dace.registry import autoregister_params

import daceml.autodiff.utils as butils
from daceml.autodiff.base_abc import BackwardImplementation, BackwardContext, BackwardResult
from daceml.onnx.op_implementations.cpp_implementations import LayerNorm, LayerNormEnvironment, add_ln_tasklet_bwd
from daceml.util import in_desc_with_name

include_dir = os.path.join(os.path.dirname(__file__), "cpp")

with open(os.path.join(include_dir, "softmax_grad_impl.cu")) as f:
    softmax_grad_impl_code = f.read()


@dace.library.environment
class ORTSoftMaxEnv:
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
    codeobjects = [
        CodeObject("softmax_grad_impl",
                   softmax_grad_impl_code,
                   'cu',
                   CUDACodeGen,
                   "CUDA",
                   target_type="")
    ]
    headers = ["softmax_grad.h"]
    init_code = ""
    finalize_code = ""


def _prod(sequence):
    return functools.reduce(lambda a, b: a * b, sequence, 1)


@autoregister_params(node_type=LayerNorm, name="onnxruntime")
class ORTLNGrad(BackwardImplementation):
    @staticmethod
    def backward_can_be_applied(node: nd.Node, state: dace.SDFGState,
                                sdfg: dace.SDFG) -> bool:
        return in_desc_with_name(node, state, sdfg, "_X").dtype is dace.float32

    @staticmethod
    def backward(
        forward_node: nd.Node, context: BackwardContext,
        given_gradients: List[Optional[str]],
        required_gradients: List[Optional[str]]
    ) -> Tuple[nd.Node, BackwardResult]:

        shape = butils.forward_in_desc_with_name(forward_node, context,
                                                 "_X").shape
        axis = forward_node.axis

        tasklet = add_ln_tasklet_bwd(context.backward_state, shape, axis)

        tasklet.environments = {LayerNormEnvironment.full_class_path()}

        result = BackwardResult.empty()
        result.given_grad_names["_Y"] = "_dY"
        result.required_grad_names["_X"] = "_dX"
        result.required_grad_names["_scale"] = "_dscale"
        result.required_grad_names["_bias"] = "_dbias"
        butils.connect_output_from_forward(forward_node, tasklet, context,
                                           "_inv_std_var")
        butils.connect_output_from_forward(forward_node, tasklet, context,
                                           "_mean")

        return tasklet, result


@autoregister_params(op="Softmax", name="onnxruntime")
class ORTSoftmaxGrad(BackwardImplementation):
    @staticmethod
    def backward_can_be_applied(node: nd.Node, state: dace.SDFGState,
                                sdfg: dace.SDFG) -> bool:
        return in_desc_with_name(node, state, sdfg,
                                 "input").dtype is dace.float32

    @staticmethod
    def backward(
        forward_node: nd.Node, context: BackwardContext,
        given_gradients: List[Optional[str]],
        required_gradients: List[Optional[str]]
    ) -> Tuple[nd.Node, BackwardResult]:

        shape = butils.forward_in_desc_with_name(forward_node, context,
                                                 "input").shape

        n = _prod(shape[:forward_node.axis])
        d = _prod(shape[forward_node.axis:])

        code = f"""
        onnxruntime::cuda::dispatch_softmax_backward<float, float, float, false>(
            __dace_current_stream,
            _dX,
            _dY,
            output,
            {d},
            {d},
            {n});
        """
        tasklet = context.backward_state.add_tasklet(
            forward_node.label + "_backward", {
                "output": dace.pointer(dace.float32),
                "_dY": dace.pointer(dace.float32)
            }, {"_dX": dace.pointer(dace.float32)},
            code,
            language=dtypes.Language.CPP)

        tasklet.environments = {ORTSoftMaxEnv.full_class_path()}

        result = BackwardResult.empty()
        result.given_grad_names["output"] = "_dY"
        result.required_grad_names["input"] = "_dX"

        butils.connect_output_from_forward(forward_node, tasklet, context,
                                           "output")
        return tasklet, result
