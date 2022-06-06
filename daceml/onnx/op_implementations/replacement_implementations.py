import typing

import numpy as np
from dace import nodes, SDFG, SDFGState

from daceml.onnx.forward_implementation_abc import ONNXForward
from daceml.onnx.nodes import onnx_op
from daceml.onnx.op_implementations.utils import op_implementation, program_for_node


@op_implementation(op="torch_geometric.nn.conv.gcn_conv.GCNConv",
                   name="pure")
class GCNConv(ONNXForward):
    @staticmethod
    def forward(node: onnx_op.ONNXOp, state: SDFGState,
                sdfg: SDFG) -> typing.Union[nodes.Node, SDFG]:
        if 'bias' in [inp.name for inp in node.schema.inputs]:
            def prog(input_0, input_1, linDOTweight, bias, output_0):
                output_0[:] = np.einsum(
                    'ij,jk->ik', input_0, linDOTweight) + bias
        else:
            def prog(input_0, input_1, linDOTweight, output_0):
                output_0[:] = np.einsum('ij,jk->ik', input_0, linDOTweight)

        return program_for_node(prog, sdfg, state, node)
