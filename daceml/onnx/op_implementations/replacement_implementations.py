import typing

import dace
import numpy as np
from dace import nodes, SDFG, SDFGState

from daceml.onnx.forward_implementation_abc import ONNXForward
from daceml.onnx.nodes import onnx_op
from daceml.onnx.op_implementations.utils import op_implementation, program_for_node
from daceml.util.utils import in_desc_with_name


@op_implementation(op="torch_geometric.nn.conv.gcn_conv.GCNConv",
                   name="pure")
class GCNConv(ONNXForward):
    @staticmethod
    def forward(node: onnx_op.ONNXOp, state: SDFGState,
                sdfg: SDFG) -> typing.Union[nodes.Node, SDFG]:
        features_desc = in_desc_with_name(node, state, sdfg, "input_0")
        N, M = features_desc.shape
        dtype = features_desc.dtype

        def prog(input_0, input_1, linDOTweight, output_0):
            # input_0: input features, N x M
            # input_1: edge list, 2 x |E|
            # linDOTweight: F x M
            # output_0: N x F
            E = input_1.shape[1]
            A = np.zeros((N, N), dtype=dtype)

            for i in range(E):
                # Multiple edges are allowed.
                A[input_1[1, i], input_1[0, i]] += 1  # Edge connections.

            if node.module.normalize and node.module.add_self_loops:
                for i in range(N):
                    # Adding self-loops only adds the missing ones.
                    # Pytorch geometric implementation just sets all self-loops to weight 1,
                    # even if there are multiple self-loops.
                    A[i, i] = 1

            if node.module.normalize:
                degrees = np.sum(A, axis=1)
                norm = 1 / np.sqrt(degrees)
                norm[degrees == 0] = 0  # Get rid of nans.
                D = np.zeros((N, N), dtype=dace.float32)
                # D = np.diag(degrees) # Not implemented in Dace
                for i in range(N):
                    D[i, i] = norm[i]
                A = D @ A @ D

            tmp = np.einsum(
                'ij,kj->ik', input_0, linDOTweight)
            output_0[:] = np.einsum('ij,jk->ik', A, tmp)

        if 'bias' in [inp.name for inp in node.schema.inputs]:
            def bias_prog(input_0, input_1, linDOTweight, bias, output_0):
                prog(input_0, input_1, linDOTweight, output_0)
                output_0[:] = output_0 + bias

            return program_for_node(bias_prog, sdfg, state, node)
        else:
            return program_for_node(prog, sdfg, state, node)
