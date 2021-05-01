from typing import List, Optional, Tuple
import os

import dace
import dace.sdfg.nodes as nd
import dace.library

from daceml.autodiff.base_abc import BackwardImplementation, BackwardContext, BackwardResult



include_dir = os.path.join(os.path.dirname(__file__), "manual")

with open(os.path.join(include_dir, "layer_norm_impl.cu")) as f:
    layer_norm_cu_code = f.read()

with open(os.path.join(include_dir, "layer_norm_impl_bwd.cu")) as f:
    layer_norm_cu_bwd_code = f.read()
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


class ORTSoftmaxGrad(BackwardImplementation):

    @staticmethod
    def backward(
            forward_node: nd.Node, context: BackwardContext,
            given_gradients: List[Optional[str]],
            required_gradients: List[Optional[str]]
    ) -> Tuple[nd.Node, BackwardResult]:


        code = f"""
"""
        context.backward_state.add_tasklet(forward_node.label + "_backward",
                                           {"_Y": dace.pointer(dace.float32),
                                            "_dY": dace.pointer(dace.float32)},
                                           {"_X": dace.pointer(dace.float32)},
                                           code )

