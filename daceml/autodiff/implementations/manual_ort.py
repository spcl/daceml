import functools
from typing import List, Optional, Tuple
import os

import dace
import dace.sdfg.nodes as nd
from dace import dtypes
import dace.library
from dace.codegen.codeobject import CodeObject
from dace.codegen.targets import CUDACodeGen
from dace.registry import autoregister_params

import daceml.autodiff.utils as butils

from daceml.autodiff.base_abc import BackwardImplementation, BackwardContext, BackwardResult



include_dir = os.path.join(os.path.dirname(__file__), "manual")

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
    codeobjects = [CodeObject("softmax_grad_impl",
                              softmax_grad_impl_code,
                              'cu',
                              CUDACodeGen,
                              "CUDA",
                              target_type="")]
    headers = ["softmax_grad.h"]
    init_code = ""
    finalize_code = ""


def _prod(sequence):
    return functools.reduce(lambda a, b: a * b, sequence, 1)

@autoregister_params(op="Softmax", name="ort")
class ORTSoftmaxGrad(BackwardImplementation):

    @staticmethod
    def backward(
            forward_node: nd.Node, context: BackwardContext,
            given_gradients: List[Optional[str]],
            required_gradients: List[Optional[str]]
    ) -> Tuple[nd.Node, BackwardResult]:

        shape = butils.forward_in_desc_with_name(forward_node, context, "input").shape


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
        tasklet = context.backward_state.add_tasklet(forward_node.label + "_backward",
                                                     {"output": dace.pointer(dace.float32),
                                                      "_dY": dace.pointer(dace.float32)},
                                                     {"_dX": dace.pointer(dace.float32)},
                                                     code,
                                                     language=dtypes.Language.CPP)

        tasklet.environments = {"ORTSoftMaxEnv"}

        result = BackwardResult.empty()
        result.given_grad_names["output"] = "_dY"
        result.required_grad_names["input"] = "_dX"

        butils.connect_output_from_forward(forward_node, tasklet, context, "output")
        return tasklet, result

