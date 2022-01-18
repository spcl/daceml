import ast
import collections

import itertools
import copy
from typing import Dict
import astunparse

import astunparse
from dace import registry, nodes as nd, SDFGState, SDFG, dtypes
from dace.properties import CodeBlock
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


class VarsGrabber(ast.NodeTransformer):
    # TODO could be improved to only grab free vars
    def __init__(self):
        self.vars = set()

    def visit_Name(self, node):
        self.vars.add(node.id)
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
        return [
            node_path_graph(cls.tsk1, cls.data, cls.tsk2),
            node_path_graph(cls.tsk1, cls.tsk2)
        ]

    def can_be_applied(self, graph: SDFGState, candidate: Dict[PatternNode,
                                                               int],
                       expr_index: int, sdfg: SDFG, permissive: bool) -> bool:
        tsk1: nd.Tasklet = self.tsk1(sdfg)
        data: nd.AccessNode = self.data(sdfg) if self.expr_index == 0 else None
        tsk2: nd.Tasklet = self.tsk2(sdfg)

        if tsk1.language is not dtypes.Language.Python or tsk2.language is not dtypes.Language.Python:
            return False

        if data is not None and data.desc(sdfg).total_size != 1:
            return False

        # tsk1 is not used anywhere else
        if graph.out_degree(tsk1) != 1 or (data is not None and set(
                e.dst for e in graph.out_edges(data)) != {tsk2}):
            return False

        # try to parse the tasklet
        try:
            if len(tsk1.code.code) != 1:
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

        tsk2_in_edges = state.out_edges(data if data is not None else tsk1)
        conns = {e.dst_conn for e in tsk2_in_edges}

        # remove the connector from tsk2
        inputs = {
            k: v
            for k, v in tsk2.in_connectors.items() if k not in conns
        }

        # copy tsk1's in connectors
        repldict = {}
        for in_edge in state.in_edges(tsk1):
            old_value = in_edge.dst_conn
            # check if there's a conflict
            if in_edge.dst_conn in inputs:
                # conflicts are ok if the memlets are the same
                tsk2edge = list(
                    state.in_edges_by_connector(tsk2, in_edge.dst_conn))[0]
                if (in_edge.data != tsk2edge.data
                        or in_edge.data.data != tsk2edge.data.data):
                    in_edge.dst_conn = find_str_not_in_set(
                        set(inputs), in_edge.dst_conn)
                    repldict[old_value] = in_edge.dst_conn
                else:
                    # if the memlets are the same rename rename the connector
                    # on the first tasklet so that we only have one read
                    pass

            inputs[in_edge.dst_conn] = tsk1.in_connectors[old_value]

        assigned_value = tsk1.code.code[0].value
        if repldict:
            assigned_value = Renamer(repldict).visit(assigned_value)

        new_code = []
        for line in tsk2.code.code:
            for tsk2_in_edge in tsk2_in_edges:
                Inliner(tsk2_in_edge.dst_conn, assigned_value).visit(line)
            new_code.append(line)
        new_code_str = "\n".join(astunparse.unparse(line) for line in new_code)

        new_tasklet = state.add_tasklet(tsk1.label + "_fused_" + tsk2.label,
                                        inputs, tsk2.out_connectors,
                                        new_code_str)

        for in_edge in state.in_edges(tsk1):
            state.add_edge(in_edge.src, in_edge.src_conn, new_tasklet,
                           in_edge.dst_conn, in_edge.data)

        for in_edge in state.in_edges(tsk2):
            # only connect if there is no edge connected to that connector yet
            if len(
                    list(
                        state.in_edges_by_connector(new_tasklet,
                                                    in_edge.dst_conn))) == 0:
                state.add_edge(in_edge.src, in_edge.src_conn, new_tasklet,
                               in_edge.dst_conn, in_edge.data)
            else:
                state.remove_memlet_path(in_edge)

        for out_edge in state.out_edges(tsk2):
            state.add_edge(new_tasklet, out_edge.src_conn, out_edge.dst,
                           out_edge.dst_conn, out_edge.data)

        state.remove_node(tsk1)
        if data is not None:
            state.remove_node(data)
        state.remove_node(tsk2)


@registry.autoregister_params(singlestate=True)
class TaskletFission(Transformation):

    tsk = PatternNode(nd.Tasklet)

    @classmethod
    def expressions(cls):
        return [node_path_graph(cls.tsk)]

    def can_be_applied(self, graph: SDFGState, candidate: Dict[PatternNode,
                                                               int],
                       expr_index: int, sdfg: SDFG, strict: bool) -> bool:
        tsk: nd.Tasklet = self.tsk(sdfg)

        if tsk.language is not dtypes.Language.Python:
            return False

        # TODO we currently don't check that the expressions are independent!

        # try to parse the tasklet
        try:
            if len(tsk.code.code) <= 1:
                return False

            for line in tsk.code.code:
                if len(line.targets) != 1:
                    return False

                if not isinstance(line.targets[0], ast.Name):
                    return False
        except:
            return False
        return True

    def apply(self, sdfg: SDFG):
        state: SDFGState = sdfg.node(self.state_id)

        tsk: nd.Tasklet = self.tsk(sdfg)

        for i, line in enumerate(tsk.code.code):
            # todo need to check here that the output is actually an out connector
            id_to_assign = line.targets[0].id

            v = VarsGrabber()
            v.visit(line.value)
            in_connectors = {var: tsk.in_connectors[var] for var in v.vars}
            out_connectors = {id_to_assign: tsk.out_connectors[id_to_assign]}
            new_tsk = state.add_tasklet(f"{tsk.label}_{i}",
                                        in_connectors, out_connectors,
                                        astunparse.unparse(line))

            for e in state.in_edges(tsk):
                if e.dst_conn in in_connectors:
                    state.add_edge(e.src, e.src_conn, new_tsk, e.dst_conn,
                                   copy.deepcopy(e.data))

            for e in state.out_edges_by_connector(tsk, id_to_assign):
                state.add_edge(new_tsk, id_to_assign, e.dst, e.dst_conn,
                               copy.deepcopy(e.data))
        state.remove_node(tsk)


@registry.autoregister_params(singlestate=True)
class MergeTaskletReads(Transformation):
    """
    If a tasklet has two inputs that read the same thing, remove one of the
    reads
    """

    src = PatternNode(nd.Node)
    tsk = PatternNode(nd.Tasklet)

    @classmethod
    def expressions(cls):
        return [node_path_graph(cls.src, cls.tsk)]

    def can_be_applied(self, graph: SDFGState, candidate: Dict[PatternNode,
                                                               int],
                       expr_index: int, sdfg: SDFG, strict: bool) -> bool:
        src: nd.AccessNode = self.src(sdfg)
        tsk: nd.Tasklet = self.tsk(sdfg)

        if tsk.language is not dtypes.Language.Python:
            return False

        edges = graph.edges_between(src, tsk)
        for a, b in itertools.combinations(edges, 2):
            if a.data == b.data and a.data.data == b.data.data:
                return True
        return False

    def apply(self, sdfg: SDFG):

        state: SDFGState = sdfg.node(self.state_id)
        src: nd.AccessNode = self.src(sdfg)
        tsk: nd.Tasklet = self.tsk(sdfg)

        edges = state.edges_between(src, tsk)
        mergable_connectors = collections.defaultdict(set)

        for a, b in itertools.combinations(edges, 2):
            if a.data == b.data and a.data.data == b.data.data:
                mergable_connectors[a.dst_conn].add(b.dst_conn)
                mergable_connectors[b.dst_conn].add(a.dst_conn)

        to_merge = next(iter(mergable_connectors))

        new_code = [
            Renamer({v: to_merge
                     for v in mergable_connectors[to_merge]}).visit(code)
            for code in tsk.code.code
        ]
        new_code_str = astunparse.unparse(new_code)
        for v in mergable_connectors[to_merge]:
            for e in state.in_edges_by_connector(tsk, v):
                state.remove_memlet_path(e)

        tsk.code = CodeBlock(new_code_str, tsk.language)
