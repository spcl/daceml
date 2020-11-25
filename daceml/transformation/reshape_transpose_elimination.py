import copy
from collections import deque
from typing import Dict

import numpy as np

import dace
import dace.data as dt
from dace import registry
from dace.properties import make_properties
from dace.sdfg import nodes as nd
from dace.sdfg import utils as sdutil
from dace.transformation import transformation

import daceml.onnx as donnx
from daceml.onnx.converters import clean_onnx_name
from daceml.onnx.nodes.onnx_op import ONNXOp, ONNXMatMul, ONNXReshape, ONNXTranspose
from daceml.onnx import ONNXModel
from dace.memlet import Memlet

# blocklist of nondeterministic ops
# yapf: disable
NONDETERMINISTIC_OPS = {'ONNXDropout',
                        'ONNXGradient',
                        'ONNXGraphCall',
                        'ONNXIf',
                        'ONNXLoop',
                        'ONNXMomentum',
                        'ONNXMultinomial',
                        'ONNXRandomNormal',
                        'ONNXRandomNormalLike',
                        'ONNXRandomUniform',
                        'ONNXRandomUniformLike',
                        'ONNXSVMClassifier',
                        'ONNXSVMRegressor',
                        'ONNXScan',
                        'ONNXTreeEnsembleClassifier',
                        'ONNXTreeEnsembleRegressor'}
# yapf: enable

@registry.autoregister_params(singlestate=True)
@make_properties
class ReshapeTransposeElimination(transformation.Transformation):
    # pattern matching only checks that the type of the node matches,
    _onnx_reshape_node = ONNXReshape("_")

    @staticmethod
    def expressions():
        return [
            sdutil.node_path_graph(
                ReshapeTransposeElimination._onnx_reshape_node)
        ]

    @staticmethod
    def can_be_applied(graph: dace.sdfg.graph.OrderedMultiDiConnectorGraph,
                       candidate: Dict[nd.Node, int],
                       expr_index: int,
                       sdfg,
                       strict: bool = False):

        node: ONNXReshape = graph.nodes()[candidate[
            ReshapeTransposeElimination._onnx_reshape_node]]

        # SDFG must be imported from an ONNXModel
        if not hasattr(sdfg, "_parent_onnx_model"):
            return False

        if not 'ONNX' + node.schema.name not in NONDETERMINISTIC_OPS:
            return False

        if len(graph.out_edges(node)) != 1:
            return False

        assert node.schema.name == "Reshape"

        assert isinstance(node, ONNXReshape)

        return True

    @staticmethod
    def match_to_str(graph, candidate):
        node: ONNXReshape = graph.nodes()[candidate[
            ReshapeTransposeElimination._onnx_reshape_node]]
        return "Eliminate Reshape and Transpose before MatMul"

    def apply(self, sdfg: dace.SDFG):

        parent: ONNXModel = sdfg._parent_onnx_model
        state = sdfg.nodes()[self.state_id]
        node = state.nodes()[self.subgraph[
            ReshapeTransposeElimination._onnx_reshape_node]]

        real_dst = None

        branchA_new_memlet = None
        branchA_real_src = None
        branchA_real_dst_conn = None
        branchA_transpose = None
        branchA_reshape = None
        branchA_can_be_applied = False
        branchA_reshape_before_trans = None
        branchA_reshape_info = None
        branchA_transpose_perm = None

        branchB_new_memlet = None
        branchB_real_src = None
        branchB_real_dst_conn = None
        branchB_transpose = None
        branchB_reshape = None
        branchB_can_be_applied = False
        branchB_reshape_before_trans = None
        branchB_reshape_info = None
        branchB_transpose_perm = None

        for edge in state.out_edges(node):
            dst_l1 = edge.dst
            if isinstance(dst_l1, nd.AccessNode) and len(
                    state.out_edges(dst_l1)) == 1 and isinstance(
                        state.out_edges(dst_l1)[0].dst, ONNXTranspose):
                tmp_transpose = state.out_edges(dst_l1)[0].dst
                next_onnx_node = state.out_edges(tmp_transpose)[0].dst
                if isinstance(next_onnx_node, nd.AccessNode) and len(
                        state.out_edges(next_onnx_node)) == 1 and isinstance(
                            state.out_edges(next_onnx_node)[0].dst,
                            ONNXMatMul):
                    real_dst = state.out_edges(next_onnx_node)[0].dst
                    if state.out_edges(next_onnx_node)[0].dst_conn == "A":
                        branchA_can_be_applied = True
                        branchA_real_dst_conn = state.out_edges(
                            next_onnx_node)[0].dst_conn
                        branchA_transpose = tmp_transpose
                        branchA_reshape = node
                        branchA_reshape_before_trans = True
                        for edge in state.in_edges(node):
                            if edge.dst_conn == "data":
                                branchA_real_src = edge.src
                                branchA_new_memlet = copy.deepcopy(edge.data)
                    if state.out_edges(next_onnx_node)[0].dst_conn == "B":
                        branchB_can_be_applied = True
                        branchB_real_dst_conn = state.out_edges(
                            next_onnx_node)[0].dst_conn
                        branchB_transpose = tmp_transpose
                        branchB_reshape = node
                        branchB_reshape_before_trans = True
                        for edge in state.in_edges(node):
                            if edge.dst_conn == "data":
                                branchB_real_src = edge.src
                                branchB_new_memlet = copy.deepcopy(edge.data)
            elif isinstance(dst_l1, nd.AccessNode) and len(
                    state.out_edges(dst_l1)) == 1 and isinstance(
                        state.out_edges(dst_l1)[0].dst, ONNXMatMul):
                real_dst = state.out_edges(dst_l1)[0].dst
                for edge in state.in_edges(node):
                    if edge.dst_conn == "data":
                        src_l1 = edge.src
                        if isinstance(src_l1, nd.AccessNode) and len(
                                state.in_edges(src_l1)) == 1 and isinstance(
                                    state.in_edges(src_l1)[0].src,
                                    ONNXTranspose):
                            tmp_transpose = state.in_edges(src_l1)[0].src
                            if state.out_edges(dst_l1)[0].dst_conn == "A":
                                branchA_can_be_applied = True
                                branchA_real_dst_conn = state.out_edges(
                                    dst_l1)[0].dst_conn
                                branchA_transpose = tmp_transpose
                                branchA_reshape = node
                                branchA_reshape_before_trans = False
                                for edge in state.in_edges(branchA_transpose):
                                    branchA_real_src = edge.src
                                    branchA_new_memlet = copy.deepcopy(
                                        edge.data)
                            if state.out_edges(dst_l1)[0].dst_conn == "B":
                                branchB_can_be_applied = True
                                branchB_real_dst_conn = state.out_edges(
                                    dst_l1)[0].dst_conn
                                branchB_transpose = tmp_transpose
                                branchB_reshape = node
                                branchB_reshape_before_trans = False
                                for edge in state.in_edges(branchB_transpose):
                                    branchB_real_src = edge.src
                                    branchB_new_memlet = copy.deepcopy(
                                        edge.data)

        if branchA_can_be_applied:
            for edge in state.in_edges(real_dst):
                if edge.dst_conn == "B":
                    branchB_real_dst_conn = edge.dst_conn
                    src_l1 = edge.src
                    if isinstance(src_l1, nd.AccessNode) and len(
                            state.in_edges(src_l1)) == 1 and isinstance(
                                state.in_edges(src_l1)[0].src, ONNXTranspose):
                        branchB_transpose = state.in_edges(src_l1)[0].src
                        for edge in state.in_edges(branchB_transpose):
                            if isinstance(edge.src, nd.AccessNode) and len(
                                    state.in_edges(
                                        edge.src)) == 1 and isinstance(
                                            state.in_edges(edge.src)[0].src,
                                            ONNXReshape):
                                branchB_can_be_applied = True
                                branchB_reshape = state.in_edges(
                                    edge.src)[0].src
                                branchB_reshape_before_trans = True
                                for edge in state.in_edges(branchB_reshape):
                                    if edge.dst_conn == "data":
                                        branchB_real_src = edge.src
                                        branchB_new_memlet = copy.deepcopy(
                                            edge.data)
                    elif isinstance(src_l1, nd.AccessNode) and len(
                            state.in_edges(src_l1)) == 1 and isinstance(
                                state.in_edges(src_l1)[0].src, ONNXReshape):
                        branchB_reshape = state.in_edges(src_l1)[0].src
                        for edge in state.in_edges(branchB_reshape):
                            if edge.dst_conn == "data" and isinstance(
                                    edge.src, nd.AccessNode) and len(
                                        state.in_edges(
                                            edge.src)) == 1 and isinstance(
                                                state.in_edges(
                                                    edge.src)[0].src,
                                                ONNXTranspose):
                                branchB_can_be_applied = True
                                branchB_transpose = state.in_edges(
                                    edge.src)[0].src
                                branchB_reshape_before_trans = False
                                for edge in state.in_edges(branchB_transpose):
                                    branchB_real_src = edge.src
                                    branchB_new_memlet = copy.deepcopy(
                                        edge.data)
        elif branchB_can_be_applied:
            for edge in state.in_edges(real_dst):
                if edge.dst_conn == "A":
                    branchA_real_dst_conn = edge.dst_conn
                    src_l1 = edge.src
                    if isinstance(src_l1, nd.AccessNode) and len(
                            state.in_edges(src_l1)) == 1 and isinstance(
                                state.in_edges(src_l1)[0].src, ONNXTranspose):
                        branchA_transpose = state.in_edges(src_l1)[0].src
                        for edge in state.in_edges(branchA_transpose):
                            if isinstance(edge.src, nd.AccessNode) and len(
                                    state.in_edges(
                                        edge.src)) == 1 and isinstance(
                                            state.in_edges(edge.src)[0].src,
                                            ONNXReshape):
                                branchA_can_be_applied = True
                                branchA_reshape = state.in_edges(
                                    edge.src)[0].src
                                branchA_reshape_before_trans = True
                                for edge in state.in_edges(branchA_reshape):
                                    if edge.dst_conn == "data":
                                        branchA_real_src = edge.src
                                        branchA_new_memlet = copy.deepcopy(
                                            edge.data)
                    elif isinstance(src_l1, nd.AccessNode) and len(
                            state.in_edges(src_l1)) == 1 and isinstance(
                                state.in_edges(src_l1)[0].src, ONNXReshape):
                        branchA_reshape = state.in_edges(src_l1)[0].src
                        for edge in state.in_edges(branchA_reshape):
                            if edge.dst_conn == "data" and isinstance(
                                    edge.src, nd.AccessNode) and len(
                                        state.in_edges(
                                            edge.src)) == 1 and isinstance(
                                                state.in_edges(
                                                    edge.src)[0].src,
                                                ONNXTranspose):
                                branchA_can_be_applied = True
                                branchA_transpose = state.in_edges(
                                    edge.src)[0].src
                                branchA_reshape_before_trans = False
                                for edge in state.in_edges(branchA_transpose):
                                    branchA_real_src = edge.src
                                    branchA_new_memlet = copy.deepcopy(
                                        edge.data)

        if branchA_can_be_applied and branchB_can_be_applied and branchA_reshape_before_trans and branchB_reshape_before_trans:
            state.add_edge(branchA_real_src, None, real_dst,
                           branchA_real_dst_conn, branchA_new_memlet)
            for edge in state.in_edges(branchA_reshape):
                if edge.dst_conn == "shape":
                    state.remove_node(edge.src)

            transA_in_edges = state.in_edges(branchA_transpose)
            transB_in_edges = state.in_edges(branchB_transpose)
            out_edges = state.out_edges(real_dst)
            atype = copy.deepcopy(sdfg.arrays[transA_in_edges[0].data.data])
            btype = copy.deepcopy(sdfg.arrays[transB_in_edges[0].data.data])
            ytype = copy.deepcopy(sdfg.arrays[out_edges[0].data.data])

            @dace.program
            def einsumop(A: atype, B: btype, Y: ytype):
                Y[:] = np.einsum('aibk,ajbk->abij', A, B)

            state.remove_node(state.out_edges(branchA_reshape)[0].dst)
            state.remove_node(branchA_reshape)
            state.remove_node(state.out_edges(branchA_transpose)[0].dst)
            state.remove_node(branchA_transpose)

            state.add_edge(branchB_real_src, None, real_dst,
                           branchB_real_dst_conn, branchB_new_memlet)
            for edge in state.in_edges(branchB_reshape):
                if edge.dst_conn == "shape":
                    state.remove_node(edge.src)
            state.remove_node(state.out_edges(branchB_reshape)[0].dst)
            state.remove_node(branchB_reshape)
            state.remove_node(state.out_edges(branchB_transpose)[0].dst)
            state.remove_node(branchB_transpose)

            real_output = state.out_edges(real_dst)[0].dst
            nsdfg = einsumop.to_sdfg()
            nsdfg_node = state.add_nested_sdfg(nsdfg, None, {'A', 'B'}, {'Y'})
            state.add_edge(
                branchA_real_src, None, nsdfg_node, 'A',
                Memlet.from_array(branchA_real_src.data,
                                  branchA_real_src.desc(sdfg)))
            state.add_edge(
                branchB_real_src, None, nsdfg_node, 'B',
                Memlet.from_array(branchB_real_src.data,
                                  branchB_real_src.desc(sdfg)))
            state.add_edge(
                nsdfg_node, 'Y', real_output, None,
                Memlet.from_array(real_output.data, real_output.desc(sdfg)))
            state.remove_node(real_dst)

        elif branchA_can_be_applied and not branchB_can_be_applied and not branchA_reshape_before_trans:
            state.add_edge(branchA_real_src, None, real_dst,
                           branchA_real_dst_conn, branchA_new_memlet)

            transA_in_edges = state.in_edges(branchA_transpose)
            out_edges = state.out_edges(real_dst)
            atype = copy.deepcopy(sdfg.arrays[transA_in_edges[0].data.data])
            ytype = copy.deepcopy(sdfg.arrays[out_edges[0].data.data])
            M = None
            N = None
            K = None
            for edge in state.in_edges(branchA_reshape):
                if edge.dst_conn == "shape":
                    state.remove_node(edge.src)
                if edge.dst_conn == "data":
                    M = edge.data.subset.size()[-2]
                    N = edge.data.subset.size()[-1]

            state.remove_node(state.out_edges(branchA_reshape)[0].dst)
            state.remove_node(branchA_reshape)
            state.remove_node(state.out_edges(branchA_transpose)[0].dst)
            state.remove_node(branchA_transpose)

            branchB_src = None
            real_output = state.out_edges(real_dst)[0].dst
            for edge in state.in_edges(real_dst):
                if edge.dst_conn == "B":
                    branchB_src = edge.src
                    K = edge.data.subset.size()[-1]

            @dace.program
            def einsumop(A: atype, B: dace.float32[M, N, K], Y: ytype):
                Y[:] = np.einsum('bkih,khj->bij', A, B)

            nsdfg = einsumop.to_sdfg()
            nsdfg_node = state.add_nested_sdfg(nsdfg, None, {'A', 'B'}, {'Y'})
            state.add_edge(
                branchA_real_src, None, nsdfg_node, 'A',
                Memlet.from_array(branchA_real_src.data,
                                  branchA_real_src.desc(sdfg)))
            state.add_edge(
                branchB_src, None, nsdfg_node, 'B',
                Memlet.from_array(branchB_src.data, branchB_src.desc(sdfg)))
            state.add_edge(
                nsdfg_node, 'Y', real_output, None,
                Memlet.from_array(real_output.data, real_output.desc(sdfg)))
            state.remove_node(real_dst)
        elif branchB_can_be_applied and not branchA_can_be_applied and branchB_reshape_before_trans:
            state.add_edge(branchB_real_src, None, real_dst,
                           branchB_real_dst_conn, branchB_new_memlet)

            in_edges = state.in_edges(real_dst)
            transB_in_edges = state.in_edges(branchB_transpose)
            out_edges = state.out_edges(real_dst)
            atype = copy.deepcopy(sdfg.arrays[in_edges[0].data.data])
            btype = copy.deepcopy(sdfg.arrays[transB_in_edges[0].data.data])
            ytype = copy.deepcopy(sdfg.arrays[out_edges[0].data.data])

            #for name, attr in branchB_transpose.schema.attributes.items():
            #    if hasattr(branchB_transpose, name) and str(branchB_transpose.schema.attributes[name]) == "perm":
            #        perm  = getattr(branchB_transpose, name)
            #        print("einsumop branchB_transpose perm attributes: ", perm)

            @dace.program
            def einsumop(A: atype, B: btype, Y: ytype):
                Y[:] = np.einsum('abik,akbj->abij', A, B)

            for edge in state.in_edges(branchB_reshape):
                if edge.dst_conn == "shape":
                    state.remove_node(edge.src)
            state.remove_node(state.out_edges(branchB_reshape)[0].dst)
            state.remove_node(branchB_reshape)

            state.remove_node(state.out_edges(branchB_transpose)[0].dst)
            state.remove_node(branchB_transpose)

            branchA_src = None
            real_output = state.out_edges(real_dst)[0].dst
            for edge in state.in_edges(real_dst):
                if edge.dst_conn == "A":
                    branchA_src = edge.src

            nsdfg = einsumop.to_sdfg()
            nsdfg_node = state.add_nested_sdfg(nsdfg, None, {'A', 'B'}, {'Y'})
            state.add_edge(
                branchA_src, None, nsdfg_node, 'A',
                Memlet.from_array(branchA_src.data, branchA_src.desc(sdfg)))
            state.add_edge(
                branchB_real_src, None, nsdfg_node, 'B',
                Memlet.from_array(branchB_real_src.data,
                                  branchB_real_src.desc(sdfg)))
            state.add_edge(
                nsdfg_node, 'Y', real_output, None,
                Memlet.from_array(real_output.data, real_output.desc(sdfg)))
            state.remove_node(real_dst)
