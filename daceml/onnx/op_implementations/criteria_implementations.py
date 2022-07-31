"""
Criterion implementations
"""

import copy
from typing import Union

import numpy as np

import dace
from dace import SDFG, SDFGState, nodes as nd

from daceml import onnx as donnx
from daceml.onnx.op_implementations.utils import op_implementation, program_for_node
from daceml.onnx.nodes import onnx_op
from daceml.onnx.forward_implementation_abc import ONNXForward

from daceml.util import in_desc_with_name, out_desc_with_name


@op_implementation(op="SoftmaxCrossEntropyLoss", name="pure")
class PureSoftmaxCrossEntropyLoss(ONNXForward):
    @staticmethod
    def forward_can_be_applied(node: onnx_op.ONNXOp, state: SDFGState,
                               sdfg: SDFG) -> bool:
        # Softmax is weird in opset 11, so let's stick to 2D for now
        if len(in_desc_with_name(node, state, sdfg, "scores").shape) != 2:
            return False

        if node.ignore_index is not None and node.ignore_index >= 0:
            return False

        # FIXME support this
        if 'weights' in node.in_connectors:
            return False
        if 'log_prob' in node.out_connectors:
            return False

        return True

    @staticmethod
    def forward(node: onnx_op.ONNXOp, state: SDFGState,
                sdfg: SDFG) -> Union[nd.Node, SDFG]:

        if node.reduction == 'mean':
            @dace.program
            def reduction(x):
                return np.mean(x)
        elif node.reduction == 'none':
            @dace.program
            def reduction(x):
                return x
        elif node.reduction == 'sum':
            @dace.program
            def reduction(x):
                return np.sum(x)
        else:
            raise ValueError("Unsupported reduction: {}".format(
                node.reduction))

        # this implementation doesn't use donnx.LogSoftmax, and thus saves the
        # final sum reduction by just grabbing the label scores directly, and
        # skipping the computation of log softmax for all non-label scores
        def prog(scores, labels, output):
            # extract the scores for the labels

            # compute the log softmax normalization
            maximum = np.maximum.reduce(scores, axis=1, keepdims=True)
            max_sub = scores - maximum
            exponent = np.exp(max_sub)
            sum = np.add.reduce(exponent, axis=1)
            log_sum = np.log(sum)

            # compute the loss values
            label_exponents = max_sub[:, labels]
            losses = log_sum - label_exponents
            output[:] = reduction(losses)

        return program_for_node(prog, sdfg, state, node)
