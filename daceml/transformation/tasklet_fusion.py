import ast
from typing import Dict

import astunparse
import torch
from dace import registry, nodes as nd, SDFGState, SDFG, dtypes
from dace.sdfg.utils import node_path_graph
from dace.transformation.transformation import Transformation, PatternNode

from daceml.util import find_str_not_in_set


class Renamer(ast.NodeTransformer):
    def __init__(self, repldict: Dict[str, str]):
        self.repldict = repldict

    def visit_Name(self, node):
        if node.id in self.repldict:
            node.id = self.repldict[node.id]
        return self.generic_visit(node)

class Inliner(ast.NodeTransformer):
    def __init__(self, target_id, target_ast):
        self.target_id = target_id
        self.target_ast = target_ast

    def visit_Name(self, node):
        if node.id == self.target_id:
            return ast.copy_location(self.target_ast, node)
        else:
            return self.generic_visit(node)

@registry.autoregister_params(singlestate=True)
class TaskletFusion(Transformation):
    """ Fuse a constant pad into a convolution.
    """

    tsk1 = PatternNode(nd.Tasklet)
    data = PatternNode(nd.AccessNode)
    tsk2 = PatternNode(nd.Tasklet)

    @classmethod
    def expressions(cls):
        return [node_path_graph(cls.tsk1, cls.data, cls.tsk2),
                node_path_graph(cls.tsk1, cls.tsk2)]

    def can_be_applied(self, graph: SDFGState, candidate: Dict[PatternNode,
                                                               int],
                       expr_index: int, sdfg: SDFG, strict: bool) -> bool:
        tsk1: nd.Tasklet = self.tsk1(sdfg)
        data: nd.AccessNode = self.data(sdfg) if self.expr_index == 0 else None
        tsk2: nd.Tasklet = self.tsk2(sdfg)

        if tsk1.language is not dtypes.Language.Python or tsk2.language is not dtypes.Language.Python:
            return False

        if data is not None and data.desc(sdfg).total_size != 1:
            return False


        # tsk1 is not used anywhere else
        if graph.out_degree(tsk1) != 1 or (data is not None and graph.out_degree(data) != 1):
            return False

        # tsk2 should have one out connector only
        if graph.out_degree(tsk2) != 1:
            return False


        # try to parse the tasklet
        try:
            if len(tsk1.code.code) != 1 or len(tsk2.code.code) != 1:
                return False
            if len(tsk1.code.code[0].targets) != 1:
                return False
        except:
            return False
        return True

    def apply(self, sdfg: SDFG) -> nd.Tasklet:
        state: SDFGState = sdfg.node(self.state_id)
        tsk1: nd.Tasklet = self.tsk1(sdfg)
        data: nd.AccessNode = self.data(sdfg) if self.expr_index == 0 else None
        tsk2: nd.Tasklet = self.tsk2(sdfg)


        tsk2_in_edge = state.out_edges(data if data is not None else tsk1)[0]

        # remove the connector from tsk2
        inputs = {k:v for k, v in tsk2.in_connectors.items() if k != tsk2_in_edge.dst_conn}


        # copy tsk1's in connectors
        repldict = {}
        for in_edge in state.in_edges(tsk1):
            old_value = in_edge.dst_conn
            # check if there's a conflict
            if in_edge.dst_conn in inputs:
                # conflicts are ok if the memlets are the same
                if in_edge.data != list(state.in_edges_by_connector(tsk2, in_edge.dst_conn))[0].data:
                    in_edge.dst_conn = find_str_not_in_set(set(inputs), in_edge.dst_conn)
                    repldict[old_value] = in_edge.dst_conn

            inputs[in_edge.dst_conn] = tsk1.in_connectors[old_value]

        assigned_value = tsk1.code.code[0].value
        if repldict:
            assigned_value = Renamer(repldict).visit(assigned_value)
        new_code = Inliner(tsk2_in_edge.dst_conn, assigned_value).visit(tsk2.code.code[0])
        new_code_str = astunparse.unparse(new_code)

        new_tasklet = state.add_tasklet(tsk1.label + "_fused_" + tsk2.label, inputs,
                                        tsk2.out_connectors, new_code_str)


        for in_edge in state.in_edges(tsk1):
            state.add_edge(in_edge.src, in_edge.src_conn, new_tasklet, in_edge.dst_conn, in_edge.data)

        for in_edge in state.in_edges(tsk2):
            # only connect if there is no edge connected to that connector yet
            if len(list(state.in_edges_by_connector(new_tasklet, in_edge.dst_conn))) == 0:
                state.add_edge(in_edge.src, in_edge.src_conn, new_tasklet, in_edge.dst_conn, in_edge.data)


        for out_edge in state.out_edges(tsk2):
            state.add_edge(new_tasklet, out_edge.src_conn, out_edge.dst, out_edge.dst_conn, out_edge.data)

        state.remove_node(tsk1)
        if data is not None:
            state.remove_node(data)
        state.remove_node(tsk2)



