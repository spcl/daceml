import functools
import os
from typing import List, Optional, Tuple

import dace.sdfg.nodes as nd
from dace.registry import autoregister_params

import daceml.autodiff.utils as butils
from daceml.autodiff.base_abc import BackwardImplementation, BackwardContext, BackwardResult
from daceml.onnx.op_implementations.cpp_implementations import LayerNorm, LayerNormEnvironment, add_ln_tasklet_bwd

include_dir = os.path.join(os.path.dirname(__file__), "cpp")


def _prod(sequence):
    return functools.reduce(lambda a, b: a * b, sequence, 1)


@autoregister_params(node_type=LayerNorm)
class ORTLNGrad(BackwardImplementation):
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
