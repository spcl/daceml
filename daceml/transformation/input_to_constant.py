from typing import Dict

import dace
from dace import registry, dtypes, properties, memlet as mm
from dace.sdfg import nodes
from dace.sdfg import utils as sdutil
from dace.transformation import transformation as xf

from daceml.onnx import ONNXModel
from daceml.onnx.converters import clean_onnx_name

def forward_memlet_tree_with_nested(state, edge) -> mm.MemletTree:
    # Obtain the full state (to work with paths that trace beyond a scope)
    state = state._graph

    # Find tree root
    curedge = edge
    while (isinstance(curedge.src, nodes.EntryNode)
           and curedge.src_conn is not None):
        assert curedge.src_conn.startswith('OUT_')
        cname = curedge.src_conn[4:]
        curedge = next(e for e in state.in_edges(curedge.src)
                       if e.dst_conn == 'IN_%s' % cname)

    tree_root = mm.MemletTree(curedge)
    tree_root.state = state

    # Collect children (recursively)
    def add_children(treenode):
        # HACK: store the parent state as a undocumented attribute of treenode
        state = treenode.state
        is_entry_node = (isinstance(treenode.edge.dst, nodes.EntryNode)
                         and treenode.edge.dst_conn
                         and treenode.edge.dst_conn.startswith('IN_'))
        if is_entry_node:
            conn = treenode.edge.dst_conn[3:]
            treenode.children = [
                mm.MemletTree(e, parent=treenode)
                for e in state.out_edges(treenode.edge.dst)
                if e.src_conn == 'OUT_%s' % conn
            ]
            for c in treenode.children:
                c.state = state
        elif isinstance(treenode.edge.dst, nodes.NestedSDFG):
            access_nodes = ((n, parent) for n, parent in treenode.edge.dst.sdfg.all_nodes_recursive()
                            if isinstance(n, nodes.AccessNode) and n.data == treenode.edge.dst_conn)

            treenode.children = []
            for access_node, parent in access_nodes:
                def make_tree(e, parent, state):
                    tree = mm.MemletTree(e, parent=treenode)
                    tree.state = state
                    return tree

                treenode.children.extend(
                    make_tree(e, treenode, parent)
                    for e in parent.out_edges(access_node))
        else:
            return

        for child in treenode.children:
            add_children(child)

    # Start from root node (obtained from above parent traversal)
    add_children(tree_root)

    # Find edge in tree
    def traverse(node):
        if node.edge == edge:
            return node
        for child in node.children:
            res = traverse(child)
            if res is not None:
                return res
        return None

    # Return node that corresponds to current edge
    return traverse(tree_root)


@registry.autoregister_params(singlestate=True)
@properties.make_properties
class InputToConstant(xf.Transformation):
    """ Convert constant inputs to dace compile time constants.
    """

    _access_node = xf.PatternNode(nodes.AccessNode)

    @staticmethod
    def expressions():
        return [sdutil.node_path_graph(InputToConstant._access_node)]

    @staticmethod
    def can_be_applied(state: dace.SDFGState,
                       candidate: Dict[nodes.Node, int],
                       expr_index: int,
                       sdfg,
                       strict: bool = False):
        # SDFG must be imported from an ONNXModel
        if not hasattr(sdfg, "_parent_onnx_model"):
            return False

        node: nodes.AccessNode = state.nodes()[candidate[
            InputToConstant._access_node]]

        # check that the data is a onnx parameter
        if node.data not in {
                clean_onnx_name(w)
                for w in sdfg._parent_onnx_model.weights
        }:
            return False

        # check that the data is never written to
        if any(
                len(parent.in_edges(n)) > 0
                for n, parent in sdfg.all_nodes_recursive()
                if isinstance(n, nodes.AccessNode) and n.data == node.data):
            return False

        for out_edge in state.out_edges(node):
            # check that the memlet tree leaves are all tasklets
            tree = forward_memlet_tree_with_nested(state, out_edge)
            for child in tree.traverse_children(include_self=True):
                if child.children != []:
                    continue
                if not isinstance(child.edge.dst, nodes.Tasklet):
                    return False
                if child.edge.dst.language not in [dtypes.Language.Python]:
                    return False

        print(InputToConstant.match_to_str(state, candidate))
        return True

    @staticmethod
    def match_to_str(graph, candidate):
        node = graph.nodes()[candidate[InputToConstant._access_node]]
        return "Convert '{}' to a compile time constant".format(node.data)

    def apply(self, sdfg: dace.SDFG):
        parent: ONNXModel = sdfg._parent_onnx_model
        state = sdfg.nodes()[self.state_id]
        node = state.nodes()[self.subgraph[InputToConstant._access_node]]
        data_name = node.data

        # add the weight as a dace constant
        unclean_onnx_name = {clean_onnx_name(w): w
                             for w in parent.weights}[node.data]
        sdfg.add_constant(data_name, parent.weights[unclean_onnx_name],
                          sdfg.arrays[node.data])

        for out_edge in state.out_edges(node):
            tree = forward_memlet_tree_with_nested(state, out_edge)
            for child in tree.traverse_children(include_self=True):
                if child.children != []:
                    continue

                # we have reached an edge that should go into a python tasklet
                root_edge = child.edge
                tasklet = root_edge.dst
                conn_name = root_edge.dst_conn
                assert isinstance(tasklet, nodes.Tasklet)

                # remove the input from the tasklet
                tasklet.remove_in_connector(conn_name)
                root_edge.dst_conn = None

                # add the constant access to the top of the tasklet
                access_str = "{}[{}]".format(data_name,
                                             root_edge.data.subset)
                tasklet.code = properties.CodeBlock(
                    "{} = {}\n".format(conn_name, access_str) +
                    tasklet.code.as_string, tasklet.language)

            # wipe the memlets off the tree
            for edge in tree:
                if isinstance(edge.src, nodes.EntryNode):
                    edge.src.remove_out_connector(edge.src_conn)
                    edge.src_conn = None

                if isinstance(edge.dst, nodes.NestedSDFG):
                    access_nodes = [(n, parent) for n, parent in edge.dst.sdfg.all_nodes_recursive()
                                    if isinstance(n, nodes.AccessNode) and n.data == edge.dst_conn]
                    for n, parent_state in access_nodes:
                        parent_state.remove_node(n)
                    del edge.dst.sdfg.arrays[edge.dst_conn]
                    edge.dst.remove_in_connector(edge.dst_conn)

                if isinstance(edge.dst, nodes.EntryNode):
                    edge.dst.remove_in_connector(edge.dst_conn)
                    edge.dst_conn = None

                edge.data = dace.Memlet()

        state.remove_node(node)

        # if this was the last node, remove the array from the sdfg and the OnnxModel
        if not any(True for n, parent in sdfg.all_nodes_recursive()
                   if isinstance(n, nodes.AccessNode) and n.data == node.data):
            del sdfg.arrays[node.data]
            del parent.weights[unclean_onnx_name]
