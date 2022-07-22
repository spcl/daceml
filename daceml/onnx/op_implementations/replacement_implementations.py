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
        N, num_in_features = features_desc.shape
        dtype = features_desc.dtype

        col_desc = in_desc_with_name(node, state, sdfg, "input_2")
        num_entries, = col_desc.shape
        weights_desc = in_desc_with_name(node, state, sdfg, "linDOTweight")
        num_out_features = weights_desc.shape[0]
        do_normalize = node.module.normalize

        def prog_sparse(input_0, input_1, input_2, input_3, linDOTweight, output_0):
            # input_0: input features, N x M
            # input_1: rowptr, N+1
            # input_2: col, num_entries
            # input_3: values, num_entries
            # linDOTweight: F x M
            # output_0: N x F

            vals = input_3
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

            # This is ~35% slower (0.56 vs 0.41)
            # tmp = dace.define_local((N, num_in_features), dtype=dtype)
            # tmp[:] = 0
            # for i, k in dace.map[0:N, 0:num_in_features]:
            #     for j in dace.map[input_1[i]:input_1[i + 1]]:
            #         inp2j = input_2[j]
            #         tmp[inp2j, k] += input_0[i, k] * vals[j]
            #
            # output_0[:] = np.einsum('ij,kj->ik', tmp, linDOTweight)

        if 'bias' in [inp.name for inp in node.schema.inputs]:
            def bias_prog(input_0, input_1, input_2, input_3, linDOTweight, bias, output_0):
                prog_sparse(input_0, input_1, input_2,
                            input_3, linDOTweight, output_0)
                for i, j in dace.map[0:N, 0:num_out_features]:
                    output_0[i, j] += bias[j]

            return program_for_node(bias_prog, sdfg, state, node)
        else:
            return program_for_node(prog_sparse, sdfg, state, node)


@op_implementation(op="torch_geometric.nn.conv.gat_conv.GATConv",
                   name="pure")
class GATConv(ONNXForward):
    # TODO: check for bipartite, edge_dim

    @staticmethod
    def forward(node: onnx_op.ONNXOp, state: SDFGState,
                sdfg: SDFG) -> typing.Union[nodes.Node, SDFG]:
        assert not node.module.add_self_loops, "Adding self loops is not supported. Add self-loops in preprocessing."

        features_desc = in_desc_with_name(node, state, sdfg, "input_0")
        N, num_in_features = features_desc.shape
        dtype = features_desc.dtype

        col_desc = in_desc_with_name(node, state, sdfg, "input_2")
        num_entries, = col_desc.shape

        heads = node.module.heads
        num_out_features = node.module.out_channels
        negative_slope = node.module.negative_slope
        assert negative_slope < 1.0

        def prog_sparse(input_0, input_1, input_2, lin_srcDOTweight, att_src, att_dst, output_0):
            # input_0: input features, N x F
            # input_1: rowptr, N+1
            # input_2: col, num_entries
            # lin_srcDOTweight: H * F' x F
            # att_srcDOT_weight: H x F
            # output_0: N x H * F'

            # Transform input features.
            features = dace.define_local(
                (N, heads, num_out_features), dtype=dtype)
            features[:] = np.reshape(np.einsum(
                'ij,kj->ik', input_0, lin_srcDOTweight), (N, heads, num_out_features))
            # compute node attention coefficients.
            alpha_src = np.sum(features * att_src, axis=-1)  # shape: N x H
            alpha_dst = np.sum(features * att_dst, axis=-1)  # N x H

            e = np.zeros((num_entries, heads), dtype=dtype)
            softmax_sum = np.zeros((N, heads), dtype=dtype)
            for l in dace.map[0:N]:
                for v in dace.map[input_1[l]:input_1[l + 1]]:
                    # Calculating e_l->colv
                    colv = input_2[v]
                    e_tmp = alpha_src[l] + alpha_dst[colv]
                    e_tmp = np.maximum(
                        negative_slope * e_tmp, e_tmp)  # LeakyReLU
                    e_tmp = np.exp(e_tmp)
                    e[v] = e_tmp
                    softmax_sum[colv] += e_tmp

            for l in dace.map[0:N]:
                for v in dace.map[input_1[l]:input_1[l + 1]]:
                    colv = input_2[v]
                    e[v] = e[v] / softmax_sum[colv]

            output_0[:] = 0
            for l in dace.map[0:N]:
                for v in dace.map[input_1[l]:input_1[l + 1]]:
                    colv = input_2[v]
                    if heads == 1:
                        output_0[colv] += e[v] * features[l]
                    else:
                        output_0[colv] += np.reshape(np.reshape(e[v], (heads, 1)) * features[l], (heads * num_out_features,))

        if 'bias' in [inp.name for inp in node.schema.inputs]:
            def bias_prog(input_0, input_1, input_2, lin_srcDOTweight, att_src, att_dst, bias, output_0):
                prog_sparse(input_0, input_1, input_2,
                            lin_srcDOTweight, att_src, att_dst, output_0)
                for i, j in dace.map[0:N, 0:num_out_features]:
                    output_0[i, j] += bias[j]

            return program_for_node(bias_prog, sdfg, state, node)
        else:
            return program_for_node(prog_sparse, sdfg, state, node)
