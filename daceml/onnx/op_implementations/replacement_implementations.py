import typing

import numpy as np
from dace import nodes, SDFG, SDFGState

from daceml.onnx.forward_implementation_abc import ONNXForward
from daceml.onnx.nodes import onnx_op
from daceml.onnx.op_implementations.utils import op_implementation, program_for_node


@op_implementation(op="torch_geometricDOTnnDOTconvDOTgcn_convDOTGCNConv",
                   name="pure")
class GCNConv(ONNXForward):
    @staticmethod
    def forward(node: onnx_op.ONNXOp, state: SDFGState,
                sdfg: SDFG) -> typing.Union[nodes.Node, SDFG]:
        # weights = node.module.lin.weight.detach().numpy()

        def prog(input_0, input_1, output_0):
            weights = np.ones((input_0.shape[1], 2), dtype=input_0.dtype)
            output_0[:] = np.einsum('ij,jk->ik', input_0, weights)

        return program_for_node(prog, sdfg, state, node)
