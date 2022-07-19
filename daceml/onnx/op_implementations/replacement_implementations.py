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
        assert not node.module.add_self_loops, "Adding self loops is not supported. Add self-loops in preprocessing."

        features_desc = in_desc_with_name(node, state, sdfg, "input_0")
        N, M = features_desc.shape
        dtype = features_desc.dtype

        col_desc = in_desc_with_name(node, state, sdfg, "input_2")
        num_entries, = col_desc.shape
        weights_desc = in_desc_with_name(node, state, sdfg, "linDOTweight")
        num_out_features = weights_desc.shape[0]
        do_normalize = node.module.normalize

        def prog_sparse(input_0, input_1, input_2, linDOTweight, output_0):
            # input_0: input features, N x M
            # input_1: rowptr, N+1
            # input_2: col, num_entries
            # linDOTweight: F x M
            # output_0: N x F

            vals = np.ones((num_entries,), dtype=dtype)
            if do_normalize:
                degrees = np.zeros((N,), dtype=dtype)
                # The following loop is not the best.
                for entry_idx in dace.map[0:num_entries]:
                    with dace.tasklet:
                        col << input_2(1)[entry_idx]
                        in_deg << degrees(1)
                        out_deg[col] = in_deg[col] + 1
                        out_deg >> degrees(1)

                norm = 1 / np.sqrt(degrees)
                norm[degrees == 0] = 0  # Get rid of nans.
                for l in dace.map[0:N]:
                    rstart = input_1[l]
                    rend = input_1[l + 1]
                    for v in dace.map[rstart:rend]:
                        # vals[v] *= norm[l] * norm[input_2[v]]
                        with dace.tasklet:
                            colv << input_2(1)[v]
                            tmp_norm << norm(2)
                            in_val << vals(1)[v]
                            out_val = in_val * tmp_norm[l] * tmp_norm[colv]
                            out_val >> vals(1)[v]

            features = dace.define_local((N, num_out_features), dtype=dtype)
            features[:] = np.einsum(
                'ij,kj->ik', input_0, linDOTweight)

            output_0[:] = 0
            for i, k in dace.map[0:N, 0:num_out_features]:
                for j in dace.map[input_1[i]:input_1[i + 1]]:
                    inp2j = input_2[j]
                    output_0[inp2j, k] += features[i, k] * vals[j]

        if 'bias' in [inp.name for inp in node.schema.inputs]:
            def bias_prog(input_0, input_1, input_2, linDOTweight, bias, output_0):
                prog_sparse(input_0, input_1, input_2, linDOTweight, output_0)
                output_0[:] = output_0 + bias

            return program_for_node(bias_prog, sdfg, state, node)
        else:
            return program_for_node(prog_sparse, sdfg, state, node)
