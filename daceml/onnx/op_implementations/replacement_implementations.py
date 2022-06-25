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

        assert not node.module.add_self_loops, "Adding self loops is not supported. Add self-loops in preprocessing."

        col_desc = in_desc_with_name(node, state, sdfg, "input_2")
        num_entries, = col_desc.shape

        # def sdmm(A_rowptr, A_col, B, result, N):
        #     K = B.shape[1]
        #     result[:] = 0
        #     for i in range(N):
        #         rstart = A_rowptr[i]
        #         rend = A_rowptr[i+1]
        #         for j in range(rstart, rend):
        #             for k in range(K):
        #                 result[A_col[j], k] += B[i, k]

        def prog(input_0, input_1, input_2, linDOTweight, output_0):
            # input_0: input features, N x M
            # input_1: rowptr, N+1
            # input_2: col, num_entries
            # linDOTweight: F x M
            # output_0: N x F

            # Two following loops call a new kernel each iteration. Use a dace.map[0:E]
            vals = np.ones((num_entries,), dtype=dtype)
            if node.module.normalize:
                degrees = np.zeros((N,))
                # The following loop is not the best.
                for entry_idx in range(num_entries):
                    degrees[input_2[entry_idx]] += 1
                norm = 1 / np.sqrt(degrees)
                norm[degrees == 0] = 0  # Get rid of nans.
                for l in range(N):
                    rstart = input_1[l]
                    rend = input_1[l+1]
                    for v in range(rstart, rend):
                        vals[v] *= norm[l] * norm[input_2[v]]

            tmp = np.einsum(
                'ij,kj->ik', input_0, linDOTweight)
            # sdmm(input_1, input_2, tmp, output_0, N)

            output_0[:] = 0
            for i in range(N):
                rstart = input_1[i]
                rend = input_1[i+1]
                for j in range(rstart, rend):
                    for k in dace.map[0:tmp.shape[1]]:
                        output_0[input_2[j], k] += tmp[i, k] * vals[j]

        if 'bias' in [inp.name for inp in node.schema.inputs]:
            def bias_prog(input_0, input_1, input_2, linDOTweight, bias, output_0):
                prog(input_0, input_1, input_2, linDOTweight, output_0)
                output_0[:] = output_0 + bias

            return program_for_node(bias_prog, sdfg, state, node)
        else:
            return program_for_node(prog, sdfg, state, node)
