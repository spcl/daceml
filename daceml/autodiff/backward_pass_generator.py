"""Automatic Differentiation of SDFGStates.
   This module exposes the add_backward_pass method that can be used to add a backward pass to an
   SDFGState.
"""
import collections
import copy
import logging
import numbers
from typing import List, Tuple, Set, Dict, Union, Deque, cast, Optional

import dace
import dace.sdfg.nodes as nd
import dace.transformation.transformation as xf
import sympy as sp
from dace import Memlet, SDFG, SDFGState
from dace import dtypes, data as dt
from dace.frontend.operations import detect_reduction_type
from dace.sdfg import graph as dgraph, state as dstate, utils as dutils

from daceml.autodiff.base_abc import (BackwardContext, BackwardResult,
                                      AutoDiffException,
                                      find_backward_implementation)
from daceml.autodiff.utils import cast_consts_to_type
from daceml.onnx.forward_implementation_abc import ONNXForward
from daceml.onnx.nodes.onnx_op import ONNXOp
from daceml.util.utils import find_str_not_in_set, in_edge_with_name

ReverseNodeReturnType = Tuple[nd.Node, BackwardResult]

log = logging.getLogger(__name__)


def _strings_to_symbols(strings: Set[str]) -> Set[sp.Symbol]:
    return {sp.symbols(string) for string in strings}


def _symbols_to_strings(symbs: Set[sp.Symbol]) -> Set[str]:
    return {str(symb) for symb in symbs}


def generate_grad_connector_names(
        existing_connectors: Set[str],
        forward_connector_names: List[str]) -> Dict[str, str]:
    """ Choose connector names for the gradients of all forward connectors.

        :param existing_connectors: existing connectors on the node.
        :param forward_connector_names: the list of connectors to generate names for.
        :returns: a mapping from entries in ``forward_connector_names`` to names for those entries.
    """

    # copy
    existing_connectors = set(existing_connectors)

    names = {}
    for n in forward_connector_names:
        result = find_str_not_in_set(existing_connectors, n + "_gradient")
        names[n] = result
        existing_connectors.add(result)

    return names


def is_initialization_state(state: SDFGState) -> bool:
    """ Check if state is an initialization state, i.e. it initializes one or more arrays with zero values
    """
    for n in state.data_nodes():
        if len(state.out_edges(n)) > 0:
            return False
    return True


def code_to_exprs(code: str, inputs: Set[str],
                  outputs: Set[str]) -> Dict[str, sp.Expr]:
    """ Convert a python string to a set of (simplified) symbolic sympy expressions. Currently, this
        supports only code consisting of assignment statements.

        :param code: the code to convert
        :param inputs: the inputs (i.e. the defined variables) for the code
        :param outputs: the outputs to generate simplified expressions for
        :return: map from outputs to symbolic expressions
    """

    inputs = list(inputs)
    outputs = list(outputs)

    code_fn = """
def symbolic_execution({}):
    # define functions from cmath.h
    from sympy import exp, log
    def log2(x):
        return log(x, 2)
    def log10(x):
        return log(x, 10)
    from sympy import sin, cos, tan, asin, acos, atan, sinh, cosh, tanh, asinh, acosh, atanh
    from sympy import sin, cos, tan, asin, acos, atan, sinh, cosh, tanh, asinh, acosh, atanh
    from sympy import Pow as pow, sqrt
    from sympy import sign, floor, ceiling as ceil, Abs as abs, Abs as fabs
    from sympy import Max as max, Min as min
    from sympy import Max as fmax, Min as fmin
{}
    return {}
    """
    code_fn = code_fn.format(
        ", ".join(inputs),
        "\n".join("    " + line.strip() for line in code.split("\n")),
        ", ".join(outputs),
    )

    try:
        # need to have dace so things like `dace.float32(1)` work
        temp_globals = {'dace': dace}
        exec(code_fn, temp_globals)

        # no idea why, but simply calling symbolic_execution doesn't work
        results = temp_globals["symbolic_execution"](
            *[sp.symbols(inp) for inp in inputs])

        if len(outputs) > 1:
            return dict(zip(outputs, results))
        else:
            return {outputs[0]: results}
    except Exception as e:
        raise AutoDiffException(
            "Exception occured while attempting to symbolically execute code:\n{}"
            .format(code)) from e


def _is_int_value(value, target_value: int) -> bool:
    if isinstance(value, numbers.Integral):
        return value == target_value

    if len(value.free_symbols) > 0 or int(value) != target_value:
        return False

    return True


def _invert_access(access: dace.AccessType) -> dace.AccessType:
    if access == dace.AccessType.ReadOnly:
        return dace.AccessType.WriteOnly
    elif access == dace.AccessType.WriteOnly:
        return dace.AccessType.ReadOnly
    return access


def _add_through_connector(node: Union[nd.MapEntry, nd.MapExit]):
    i = 1
    while ("IN_{}".format(i) in node.in_connectors
           or "OUT_{}".format(i) in node.out_connectors):
        i += 1
    assert node.add_in_connector("IN_{}".format(i))
    assert node.add_out_connector("OUT_{}".format(i))
    return "IN_{}".format(i), "OUT_{}".format(i)


def _invert_map_connector(conn):
    if conn.startswith("IN"):
        return "OUT" + conn[2:]
    elif conn.startswith("OUT"):
        return "IN" + conn[3:]
    else:
        raise AutoDiffException(
            "Could not parse map connector '{}'".format(conn))


def _has_inplace_operation(state: dace.SDFGState) -> bool:
    """Returns true if state has any inplace operations
    Note that this method is currently much stronger than required; some of the constraints can be
    loosened in the future.
    """

    sdfg = state.parent

    # check that each data descriptor has at most one access nodes
    seen_accesses: Set[str] = set()
    for node in state.nodes():
        if isinstance(node, nd.AccessNode):
            if node.data in seen_accesses:
                return True
            seen_accesses.add(node.data)

    # Edges with scalar memlets can be used to connect two code nodes together. If this feature is
    # used, it should be done using a new scalar every time.
    # When a scalar is used in a code -> code edge, it should also have an AccessNode that refers to it.
    seen_scalars = set()
    for edge in state.edges():
        memlet_data = edge.data.data
        if (isinstance(sdfg.arrays[memlet_data], dt.Scalar)
                and isinstance(edge.src, nd.CodeNode)
                and isinstance(edge.dst, nd.CodeNode)):
            if memlet_data in seen_scalars or memlet_data in seen_accesses:
                return True
            seen_scalars.add(memlet_data)
    return False


def _walk_up_memlet_tree_through_view_nodes(
    sdfg, forward_state, start_name
) -> Tuple[Union[dt.Scalar, dt.Array], str, Deque[Tuple[str, dt.Data,
                                                        Memlet]]]:
    """ Starting from the (singular) access node for ``start_name`` in ``forward_state``, walk up the
        memlet path until a non-view node is reached

        :param sdfg: the forward sdfg
        :param forward_state: the forward state
        :param start_name: the name of the array to start at
        :return: the descriptor at the root of the path, the name at the root of the path, the list of
                 array names, view data descriptor and memlets encountered along the path.
    """
    forwarded_name = start_name
    view_nodes_to_clone: Deque[Tuple[str, dt.Data,
                                     Memlet]] = collections.deque()
    if isinstance(sdfg.arrays[start_name], dt.View):
        # this is complicated slightly by views: we need to walk up the memlet path until we reach a
        # non-view access node. We then need to replicate the sequence of views in the backward SDFG.
        query = [
            n for n in forward_state.nodes()
            if isinstance(n, nd.AccessNode) and n.data == start_name
        ]
        if len(query) != 1:
            raise AutoDiffException(
                f"Could not find access node to forward with data {start_name}"
            )
        current_node = query[0]
        while isinstance(sdfg.arrays[current_node.data], dt.View):

            in_edges = forward_state.in_edges(current_node)
            if len(in_edges) != 1:
                raise AutoDiffException(
                    f"Expected view node with in degree 1, got {len(in_edges)} for view node {current_node}"
                )
            if not isinstance(in_edges[0].src, nd.AccessNode):
                raise AutoDiffException(
                    f"Expected view node {current_node} to be connected to access node, got {in_edges[0].src}"
                    f" (of type {type(in_edges[0].src)})")
            view_nodes_to_clone.append(
                (current_node.data, sdfg.arrays[current_node.data],
                 in_edges[0].data))
            current_node = in_edges[0].src
            forwarded_name = current_node.data

    return sdfg.arrays[forwarded_name], forwarded_name, view_nodes_to_clone


def _path_src_node_in_subgraph(edge: dgraph.MultiConnectorEdge,
                               subgraph: dstate.StateSubgraphView):
    path_src = subgraph.memlet_path(edge)[0].src
    return path_src in subgraph.nodes()


class BackwardPassGenerator:
    """ Class that holds the state for one backward pass creation.

        See autodiff.py, _reverse_NestedSDFG and pytorch.py for examples of usage.

        :param state: the forward pass to differentiate should be in this state
        :param given_gradients: the outputs that gradients must be provided for (i.e. access nodes will be created for
               these)
        :param required_gradients: the inputs to generate gradients for
        :param backward_sdfg: the sdfg the backward pass will be contained in. If it is the same as the forward_sdfg,
                              outputs must be a list containing a single scalar.
        :param backward_state: the state which the backward pass should be added to (must be added to `backward_sdfg`
                               before calling this method).
        :param apply_strict: whether to apply strict transformations before creating the backward pass.
    """
    def __init__(
            self,
            *,
            sdfg: SDFG,
            state: SDFGState,
            given_gradients: List[Union[nd.AccessNode, str]],
            required_gradients: List[Union[nd.AccessNode, str]],
            backward_sdfg: SDFG,  # this can be the same as SDFG
            backward_state: SDFGState,
            apply_strict=False):

        if backward_state not in backward_sdfg.nodes():
            raise AutoDiffException(
                "Expected to find backward_state in backward_sdfg")

        def str_to_access(data: str, source: str) -> nd.AccessNode:
            matches = [
                node for node in state.nodes()
                if isinstance(node, nd.AccessNode) and node.data == data
            ]
            if len(matches) != 1:
                raise AutoDiffException(
                    "Expected to find exactly one node with data"
                    " '{}' in {}, but found {}".format(data, source,
                                                       len(matches)))
            return matches[0]

        given_gradients = [
            n if isinstance(n, nd.AccessNode) else str_to_access(n, "outputs")
            for n in given_gradients
        ]
        required_gradients = [
            n if isinstance(n, nd.AccessNode) else str_to_access(n, "inputs")
            for n in required_gradients
        ]

        self.given_gradients = given_gradients
        self.required_gradients = required_gradients

        self.input_names = {n.data for n in required_gradients}
        self.output_names = {n.data for n in given_gradients}

        self.sdfg = sdfg
        self.forward_state = state
        self.backward_sdfg = backward_sdfg
        self.backward_state: SDFGState = backward_state

        #: arrays descs for the gradients
        self.backward_grad_arrays: Dict[str, dt.Array] = {}

        #: arrays descs for inputs that are required from the forward pass
        self.backward_input_arrays: Dict[str, dt.Array] = {}

        #: mapping from forward node -> backward node, and forward map -> backward map
        self.reverse_map: Dict[nd.Node, Union[nd.Node, nd.Map]] = {}

        #: mapping from forward_node -> BackwardResult for that node
        self.result_map: Dict[nd.Node, BackwardResult] = {}

        #: mapping from forward name to gradient name for arrays
        self.array_grad_map: Dict[str, str] = {}

        # checks if backward has already been applied
        self._applied = False
        self.apply_strict = apply_strict

        for outp in self.given_gradients:
            if outp not in self.forward_state:
                raise AutoDiffException(
                    "Could not find output {} in state {}".format(
                        outp, self.forward_state))

        for inp in self.required_gradients:
            if inp not in self.forward_state:
                raise AutoDiffException(
                    "Could not find input {} in state {}".format(
                        inp, self.forward_state))

        # check for inplace operations (i.e. duplicated access nodes)
        if _has_inplace_operation(self.forward_state):
            raise AutoDiffException(
                "Inplace operations are currently not supported in autodiff")

        if sdfg is backward_sdfg:
            # this only makes sense if the output is a single scalar.
            if len(given_gradients) != 1:
                raise AutoDiffException(
                    "When the forward sdfg is the same as the backward sdfg, outputs must be a"
                    "single scalar")
            if not _is_int_value(
                    sdfg.arrays[given_gradients[0].data].total_size, 1):
                raise AutoDiffException(
                    "When the forward sdfg is the same as the backward sdfg, outputs must be a"
                    "single scalar")
            self.separate_sdfgs = False
        else:
            self.separate_sdfgs = True

    def _expand_nodes(self, subgraph: dstate.StateSubgraphView) -> bool:
        """ Expand all library nodes in the graph to pure implementations. Returns whether something was expanded
        """

        expanded_something = False
        for node, state in subgraph.all_nodes_recursive():
            if isinstance(state, dstate.StateSubgraphView):
                state = state.graph

            # check if the node exists in the backward implementation repository
            if find_backward_implementation(state.parent, state,
                                            node) is not None:
                continue

            # only check others if we didn't break out of the above loop
            if isinstance(node, ONNXOp):
                for impl in ONNXForward.registered_implementations(
                        node.schema.name):
                    if impl.forward_can_be_applied(node, state, self.sdfg):
                        # try to apply the expansion
                        class Expansion(xf.ExpandTransformation):
                            environments = []
                            _expansion_result = None

                            @classmethod
                            def expansion(cls, node, state, sdfg):
                                return impl.forward(node, state, sdfg)

                            @staticmethod
                            def annotates_memlets() -> bool:
                                return True

                        Expansion._match_node = xf.PatternNode(type(node))
                        Expansion.apply_to(state.parent,
                                           verify=False,
                                           _match_node=node)
                        expanded_something = True
                        continue

            # This could later on be changed to check if the expansion is differentiable and if not, move
            # on to the next expansion. For now we will just apply the first one that matches, prioritizing ones that
            # have "pure" in the name
            if isinstance(node,
                          nd.LibraryNode) and not isinstance(node, ONNXOp):
                # try to select an expansion
                if hasattr(node, "implementations"):
                    implementations = node.implementations

                    pure_candidates = [
                        name for name, impl in implementations.items()
                        if "pure" in name
                    ]
                    if len(pure_candidates) > 0:
                        expansion = pure_candidates[0]
                    else:
                        expansion = node.implementation
                else:
                    expansion = node.implementation

                node.implementation = expansion
                node.expand(state.parent, state)
                expanded_something = True

        return expanded_something

    def _disambiguate_direction_dependent_views(self):
        """ Consider the following subgraph:
            (A) -- y --> (n) -- x --> (C)
            In dace, if B is a View node and A and C are access nodes, and y and x both have data set to A.data and
            B.data respectively, the semantics of the graph depend on the order in which it is executed, i.e. reversing
            the subgraph doesn't perform as expected anymore. To disambiguate this case, we set y.data to the View's
            data.
        """

        for n in self.forward_state.nodes():
            if isinstance(
                    n, nd.AccessNode) and type(n.desc(self.sdfg)) is dt.View:
                in_edges = self.forward_state.in_edges(n)
                out_edges = self.forward_state.out_edges(n)

                if len(in_edges) == 1 and len(out_edges) == 1:
                    A = in_edges[0].src
                    y = in_edges[0].data
                    C = out_edges[0].dst
                    x = out_edges[0].data
                    if (isinstance(A, nd.AccessNode)
                            and isinstance(C, nd.AccessNode)
                            and y.data == A.data and x.data == C.data):

                        # flip the memlet
                        y.subset, y.other_subset = y.other_subset, y.subset
                        y.data = n.data
                        y.try_initialize(self.sdfg, self.forward_state,
                                         in_edges[0])

    def backward(
        self
    ) -> Tuple[BackwardResult, Dict[str, dt.Array], Dict[str, dt.Array]]:
        """ Generate the backward pass in backward_state.

            :return: tuple of:
                     * the backward result (see :class:`~daceml.autodiff.backward_implementation.BackwardResult`)
                     * dict of data descriptors for the gradients (i.e. the outputs of the backward pass)
                     * dict of data descriptors of required outputs from the forward pass. These need to be added to the
                       parent SDFG of the backward pass.
        """

        if self._applied:
            raise AutoDiffException(
                "Backward may only be called once. Instantiate a new BackwardPassGenerator."
            )

        forward_subgraph = self._find_subgraph_to_differentiate()

        # expand until there is nothing left to expand
        while self._expand_nodes(forward_subgraph):
            # Nodes have been expanded again on the expanded graph; recalculate the forward graph
            forward_subgraph = self._find_subgraph_to_differentiate()

        if self.apply_strict:
            self.sdfg.apply_strict_transformations()
            forward_subgraph = self._find_subgraph_to_differentiate()

        # check that all edges are float
        for edge, parent_subgraph in forward_subgraph.all_edges_recursive():
            if isinstance(parent_subgraph, SDFGState):
                parent_sdfg = parent_subgraph.parent
            elif isinstance(parent_subgraph, dstate.StateSubgraphView):
                parent_sdfg = parent_subgraph.graph.parent
            elif isinstance(parent_subgraph, SDFG):
                # if there are any fancy things on the interstate edges we should probably throw an error
                continue
            else:
                raise AutoDiffException("Unexpected subgraph structure")

            if edge.data.data:
                edge_type = parent_sdfg.arrays[edge.data.data].dtype
                if edge_type not in [dace.float16, dace.float32, dace.float64]:
                    raise AutoDiffException(
                        f"Expected Subgraph to differentiate to only contain float edges, but data {edge.data}"
                        f" on edge {edge} has type {edge_type}")

        self._disambiguate_direction_dependent_views()

        # recursively reverse the subgraph
        self._reverse_subgraph(forward_subgraph)

        self._applied = True

        # in some cases (accessnode -> accessnode), the descriptors for the gradients of the function outputs are not
        # added yet. Add them now

        for given_grad in self.given_gradients:
            if self.array_grad_name(
                    given_grad.data) not in self.backward_sdfg.arrays:
                self._add_gradient_data_descriptor(given_grad.data)

        # prepare the output
        required_grad_names = {
            name.data: self.array_grad_name(name.data)
            for name in self.required_gradients
        }
        given_grad_names = {
            name.data: self.array_grad_name(name.data)
            for name in self.given_gradients
        }
        result = BackwardResult(required_grad_names=required_grad_names,
                                given_grad_names=given_grad_names)
        return result, self.backward_grad_arrays, self.backward_input_arrays

    def _find_subgraph_to_differentiate(self) -> dstate.StateSubgraphView:
        """ Determine which nodes we need to reverse; this forms the subgraph we will differentiate:
            we do a reverse BFS and a forward BFS, then take the intersection of nodes found.

            To calculate the gradients for a node x in ``required_gradients``, we need to sum up consider the gradient
            contributions from every node y where x is used as an input. We thus first do a forward BFS. Also, the
            gradient contributions of all nodes that are not connected by a path to a ``given_gradient`` node are
            implicitly zero. Thus, we take the intersection of the two BFSs.
        """
        forward_nodes = {
            n
            for e in self.forward_state.bfs_edges(self.required_gradients)
            for n in [e.src, e.dst]
        }
        backward_nodes = {
            n
            for e in self.forward_state.bfs_edges(self.given_gradients,
                                                  reverse=True)
            for n in [e.src, e.dst]
        }

        forward_subgraph = dstate.StateSubgraphView(
            self.forward_state,
            list(forward_nodes.intersection(backward_nodes)))
        return forward_subgraph

    def array_grad_name(self, forward_name: str) -> str:
        """ Return the gradient name of a name from the forward pass """
        if forward_name not in self.array_grad_map:
            self.array_grad_map[forward_name] = \
                find_str_not_in_set(set(self.backward_sdfg.arrays), forward_name + "_gradient")

        return self.array_grad_map[forward_name]

    def _init_grad(self, data: str):
        """ Add a state where `data` is initialized with zero.
            self.sdfg.arrays[data] should have type Union[dt.Array, dt.Scalar, dt.View]
        """
        state = self.backward_sdfg.add_state_before(self.backward_state,
                                                    label="init_" + data)

        arr = self.backward_sdfg.arrays[data]
        scalar = 0
        if type(arr) is dt.Array:
            state.add_mapped_tasklet(
                "_init_" + data + "_", {
                    "i{}".format(i): "0:{}".format(shape)
                    for i, shape in enumerate(arr.shape)
                }, {},
                "__out = {}".format(scalar), {
                    "__out":
                    dace.Memlet.simple(
                        data, ", ".join("i{}".format(i)
                                        for i in range(len(arr.shape))))
                },
                external_edges=True)
        elif type(arr) is dt.Scalar:
            tasklet = state.add_tasklet("_init_" + data + "_", {}, {"__out"},
                                        "__out = {}".format(scalar))
            write = state.add_write(data)
            state.add_edge(tasklet, "__out", write, None,
                           Memlet.simple(data, "0"))
        elif type(arr) is dt.View:
            # not need to initialize: the viewed array will always be visited
            # (since a view can never be a required grad), and thus the viewed array will be initialized.
            pass
        else:
            raise AutoDiffException(
                "Unsupported data descriptor {}".format(arr))

    def _reverse_subgraph(self, subgraph: dstate.StateSubgraphView):
        """ Reverse a given subgraph. All nodes in the subgraph will be reversed. """

        # a reversed topological sort is a topological sort on the reverse graph
        for node in reversed(
                list(
                    dutils.dfs_topological_sort(subgraph,
                                                subgraph.source_nodes()))):

            try:
                # output names on the forward node
                # (for which the gradient will be connected as an input on the reverse node)
                given_gradients = [
                    edge.src_conn for edge in subgraph.out_edges(node)
                    if _path_src_node_in_subgraph(edge, subgraph)
                ]

                # input names on the forward node that gradients should be generated for
                required_gradients = [
                    edge.dst_conn for edge in subgraph.in_edges(node)
                    if _path_src_node_in_subgraph(edge, subgraph)
                ]

                reversed_node, backward_result = self._get_reverse_node(
                    node, given_gradients, required_gradients)

                self.reverse_map[node] = reversed_node
                self.result_map[node] = backward_result

                # connect the required inputs of the reverse node:
                # the gradients ...
                self._connect_given_gradients(subgraph, node)
                # ... and any required input values from the forward pass
                self._connect_forward_inputs(node)

                if isinstance(node, nd.AccessNode):
                    # this means we are writing out a grad to an array.
                    # initialize the gradient if it hasn't been initialized already (this can also happen in
                    # _connect_given_gradients
                    if self.array_grad_name(
                            node.data) not in self.backward_sdfg.arrays:
                        # this grad hasn't been written before: initialize it
                        self._add_gradient_data_descriptor(node.data)

                    # we need to make all incoming gradients sum
                    for edge in self.backward_state.in_edges(reversed_node):
                        self._set_wcr_sum_if_needed(edge)

            except AutoDiffException as e:
                raise AutoDiffException(
                    "Failed at node {}".format(node)) from e

    def _set_wcr_sum_if_needed(self, edge: dgraph.MultiConnectorEdge):
        """ Set the WCR to sum for all edges along the path of edge, if needed.

            :param edge: the root edge to start from
        """

        # this method assumes that the memlet tree is iterated from the root backwards
        for path_edge in self.backward_state.memlet_tree(edge):

            src_or_dest_map = (isinstance(path_edge.src,
                                          (nd.MapExit, nd.MapEntry))
                               or isinstance(path_edge.dst,
                                             (nd.MapExit, nd.MapEntry)))
            # count the amount of in edges per connector
            connector_in_edges = collections.defaultdict(int)
            for _, _, _, dst_conn, _ in self.backward_state.in_edges(
                    path_edge.dst):
                connector_in_edges[dst_conn] += 1

            more_than_one_edge_to_connector = any(
                v > 1 for v in connector_in_edges.values())

            if more_than_one_edge_to_connector or src_or_dest_map:
                old_edges = []
                for sibling_edge in self.backward_state.in_edges(
                        path_edge.dst):
                    # If there is a nestedSDFG or LibraryNode, or CodeNode with CPP, we insert a
                    # transient node to sum the gradients.
                    old_edge, new_edge = self._insert_transient_into_edge(
                        sibling_edge, self.backward_state)

                    # then wcr sum on the whole path

                    if new_edge is not None:
                        old_edges.append(old_edge)
                    else:
                        for path_edge in self.backward_state.memlet_tree(
                                old_edge):
                            path_edge.data.wcr = "lambda x, y: x + y"

                for old_edge in old_edges:
                    # recurse on the old_edge
                    self._set_wcr_sum_if_needed(old_edge)

                # break early: we have handled all edges in the tree
                break

    def _insert_transient_into_edge(
        self, sibling_edge: dgraph.MultiConnectorEdge, state: SDFGState
    ) -> Tuple[dgraph.MultiConnectorEdge, Optional[dgraph.MultiConnectorEdge]]:
        """ Modifies a graph so that a transient is inserted into an edge.
            For example:
                (a) -e-> (b)
            becomes:
                (a) -e1-> (transient) -e2> (b)

            :param sibling_edge: the edge to split.
            :return: e1, and the edge e2 (or None if no split occurred)
        """
        if (isinstance(sibling_edge.src, (nd.LibraryNode, nd.NestedSDFG))
                or (isinstance(sibling_edge.src, nd.Tasklet)
                    and sibling_edge.src.language is dtypes.Language.CPP)):
            # write out the data to a transient, then connect the transient to the rest of
            # the path with WCR.
            desc = self.backward_sdfg.arrays[sibling_edge.data.data]
            transient_name, _ = self.backward_sdfg.add_temp_transient(
                sibling_edge.data.subset.size(),
                desc.dtype,
                storage=desc.storage,
                lifetime=desc.lifetime)
            self._init_grad(transient_name)
            access_transient = state.add_access(transient_name)
            state.remove_edge(sibling_edge)
            memlet = copy.deepcopy(sibling_edge.data)
            memlet.data = transient_name
            state.add_edge(sibling_edge.src, sibling_edge.src_conn,
                           access_transient, None, memlet)
            new_edge = state.add_edge(access_transient, None, sibling_edge.dst,
                                      sibling_edge.dst_conn, sibling_edge.data)
            return sibling_edge, new_edge
        else:
            return sibling_edge, None

    def _add_gradient_data_descriptor(self, data_name: str):
        """ Add the data descriptor for the gradient for `data_name`.
            :param data_name: the name of the forward descriptor.
        """
        grad_name = self.array_grad_name(data_name)

        if grad_name in self.backward_sdfg.arrays:
            raise AutoDiffException(
                f"descriptor for gradient of {data_name} ({grad_name}) already exists"
            )

        array = self.sdfg.arrays[data_name]

        if type(array) not in [dt.Scalar, dt.Array, dt.View]:
            raise AutoDiffException(
                "Unsupported data descriptor {}".format(array))

        cloned_datadesc = copy.deepcopy(array)

        # only the grads of the inputs and the outputs are not transient
        cloned_datadesc.transient = data_name not in self.input_names and data_name not in self.output_names

        self.backward_grad_arrays[grad_name] = cloned_datadesc
        self.backward_sdfg.arrays[grad_name] = copy.deepcopy(cloned_datadesc)

        if cloned_datadesc.transient:
            self._init_grad(grad_name)

    def _connect_given_gradients(self, subgraph: dstate.StateSubgraphView,
                                 forward_node):
        """ Connect the gradients of the outputs of forward_node as inputs to the corresponding reverse node. """

        for edge in subgraph.out_edges(forward_node):
            if not _path_src_node_in_subgraph(edge, subgraph):
                # skip connecting edges for which we don't need to generate grads.
                continue

            src_node, output_conn, dest_node, input_conn, memlet = edge
            if detect_reduction_type(memlet.wcr) not in [
                    None,
                    dtypes.ReductionType.Sum,
            ]:
                raise AutoDiffException(
                    "Unsupported reduction type {} on memlet".format(
                        detect_reduction_type(memlet.wcr)))

            memlet = copy.deepcopy(memlet)

            # remove the WCR since these are now read edges
            memlet.wcr = None

            grad_name = self.array_grad_name(memlet.data)
            if grad_name not in self.backward_sdfg.arrays:
                # this grad hasn't been written before: initialize it
                self._add_gradient_data_descriptor(memlet.data)
            memlet.data = grad_name

            self.backward_state.add_edge(
                self.reverse_map[dest_node],
                self._lookup_required_grad_name(dest_node, input_conn),
                self.reverse_map[forward_node],
                self._lookup_given_grad_name(forward_node, output_conn),
                memlet,
            )

    def _connect_forward_inputs(self, forward_node):
        """ Connect the reversed node of `forward_node` to all required non-gradient inputs.

            There are non-trivial points to handle:
            1. When we read an input from an accessnode in the forward pass, we need to route through maps in the
               backward pass.
            2. In some cases, we need to save the value of a connector to an array so that the backward pass can
               read it.
               For now, this is only supported when the node is at the "top level" of the SDFG, since it's quite
               difficult to handle otherwise (you have to decide whether to recompute or to store the value, and you
               have to store the value once for every iteration in the map)
        """

        rev = self.reverse_map[forward_node]
        ####################################
        # Determine which inputs we need to connect.
        # these are the in_connectors on the reverse node, minus the gradients.
        # (these are connected in _connect_input_gradients)
        required_inputs = set(rev.in_connectors).difference(
            self.result_map[forward_node].given_grad_names.values())

        # note we use forward state here: we might need to connect inputs that are not in the
        # forward pass
        input_edges_to_connect = (
            edge for edge in self.forward_state.in_edges(forward_node)
            if edge.dst_conn in required_inputs)

        for edge in input_edges_to_connect:
            # memlet path should be fine here because the edges connect directly to the tasklet
            path = self.forward_state.memlet_path(edge)

            ####################################
            # we can only add this edge if the first node in the path not within a map scope. Otherwise the value read
            # in the backward pass might be different to the one read in the forward pass

            if self.forward_state.scope_dict()[path[0].src] is not None:
                parent = self.forward_state.scope_dict()[path[0].src]
                raise AutoDiffException(
                    "Unexpected graph structure: unable to access value of {} in the"
                    " backward pass. This can be remedied by moving the node outside the scope it "
                    "is in (it's parent is {})".format(path[0].src, parent))

            if len(path) == 1 and isinstance(path[0].src,
                                             nd.CodeNode) and isinstance(
                                                 path[0].dst, nd.CodeNode):
                # paths of length one with scalar data are allowed; these are code -> code edges
                # however, in this case it must be a scalar edge
                if not _is_int_value(
                        self.sdfg.arrays[path[0].data.data].total_size, 1):
                    raise AutoDiffException(
                        "Unexpected graph structure: encountered code -> code edge with scalar size "
                        "!= 1 (was {})".format(
                            self.sdfg.arrays[path[0].data].total_size))

                raise NotImplementedError()
            else:
                # otherwise we expect AccessNode -> MapEntry -> ... -> MapEntry -> CodeNode
                if not (isinstance(path[0].src, nd.AccessNode)
                        and isinstance(path[-1].dst, nd.CodeNode)):
                    raise AutoDiffException(
                        "Unexpected graph structure: expected memlet path that starts with an "
                        "AccessNode and ends with CodeNode")

                conn_map = {}
                for i, path_edge in enumerate(path):

                    ####################################
                    # Get the dst node and connector

                    if i == len(path) - 1:
                        if not isinstance(path_edge.dst, nd.CodeNode):
                            raise AutoDiffException(
                                "Unexpected graph structure: expected memlet path that starts with an "
                                "AccessNode and ends with CodeNode")
                        new_edge_dst = self.reverse_map[path_edge.dst]
                        new_edge_dst_conn = edge.dst_conn
                    else:
                        # if we have more than one edge, check that all intermediate nodes are MapEntry
                        if not isinstance(path_edge.dst, nd.MapEntry):
                            raise AutoDiffException(
                                "Unexpected graph structure")

                        new_edge_dst = self._find_backward_entry_node_for_map_entry(
                            path_edge.dst)
                        new_edge_dst_conn, _src_conn = _add_through_connector(
                            new_edge_dst)
                        # save the newly added connector so that we can use for the next loop iteration
                        conn_map[new_edge_dst] = _src_conn

                    ####################################
                    # Get the src node and connector

                    if i == 0:
                        if not isinstance(path_edge.src, nd.AccessNode):
                            raise AutoDiffException(
                                "Unexpected graph structure: expected memlet path that starts with an "
                                "AccessNode and ends with CodeNode")

                        new_edge_src_conn = None
                        if path_edge.src in self.reverse_map:
                            new_edge_src = self.reverse_map[path_edge.src]
                        else:
                            # Add an AccessNode for this to the backward pass
                            data_name = path_edge.src.data
                            data_desc = copy.deepcopy(
                                self.sdfg.arrays[data_name])

                            # if the descriptor is a view, we will rebuild the sequence of views that create this view
                            # this involves walking up the path until we find a non-view access node, and then
                            # replicating that path in the backward pass
                            if type(data_desc) is dt.View:
                                data_desc, data_name, view_nodes_to_clone = _walk_up_memlet_tree_through_view_nodes(
                                    self.sdfg, self.forward_state, data_name)
                                new_edge_src = self.backward_state.add_access(
                                    data_name)

                                while len(view_nodes_to_clone) > 0:
                                    view_name, view_desc, memlet = view_nodes_to_clone.pop(
                                    )

                                    memlet = copy.deepcopy(memlet)

                                    if self.separate_sdfgs:
                                        self.backward_sdfg.add_datadesc(
                                            view_name,
                                            copy.deepcopy(view_desc))
                                    new_access = self.backward_state.add_access(
                                        view_name)
                                    self.backward_state.add_edge(
                                        new_edge_src, None, new_access, None,
                                        memlet)
                                    new_edge_src = new_access
                            else:
                                new_edge_src = self.backward_state.add_access(
                                    data_name)

                            # adding it to the backward_input_arrays will mean that any users of this SDFG
                            # will know that we require this array from the forward pass
                            assert data_name not in self.backward_input_arrays
                            self.backward_input_arrays[data_name] = data_desc

                            if self.separate_sdfgs:
                                # because we need to forward this, the descriptor is no longer transient
                                data_desc.transient = False
                                self.backward_sdfg.add_datadesc(
                                    data_name, data_desc)

                            self.reverse_map[path_edge.src] = new_edge_src

                    else:
                        # if we have more than one edge, check that all intermediate nodes are MapEntry
                        if not isinstance(path_edge.src, nd.MapEntry):
                            raise AutoDiffException(
                                "Unexpected graph structure")

                        new_edge_src = self._find_backward_entry_node_for_map_entry(
                            path_edge.src)
                        new_edge_src_conn = conn_map[new_edge_src]

                    self.backward_state.add_edge(new_edge_src,
                                                 new_edge_src_conn,
                                                 new_edge_dst,
                                                 new_edge_dst_conn,
                                                 copy.deepcopy(path_edge.data))

    def _lookup_required_grad_name(self, node: nd.Node, connector: str) -> str:
        if node not in self.result_map:
            raise AutoDiffException(
                "Attempted to access gradient of {}"
                " before the backward node was created".format(node))
        return self.result_map[node].required_grad_names[connector]

    def _lookup_given_grad_name(self, node: nd.Node, connector: str) -> str:
        if node not in self.result_map:
            raise AutoDiffException(
                "Attempted to access gradient of {}"
                " before the backward node was created".format(node))
        return self.result_map[node].given_grad_names[connector]

    def _find_backward_entry_node_for_map_entry(
            self, entry_node: nd.MapEntry) -> nd.MapExit:
        """Find the entry node in the backward pass corresponding to the exit node opened by
        `entry_node` (where `entry_node` is a node from the forward pass).
        """
        src_candidates = [
            cast(nd.MapExit, node) for node in self.backward_state.nodes()
            if isinstance(node, nd.MapEntry)
            and node.map == self.reverse_map[entry_node.map]
        ]
        if len(src_candidates) != 1:
            # this shouldn't happen; if we are within a scope, the exit nodes
            # for the scope should already exist in the backward pass
            raise AutoDiffException("Invalid graph")

        return src_candidates[0]

    def _get_reverse_node(self, node, given_gradients,
                          required_gradients) -> ReverseNodeReturnType:
        """ Add the reverse node for a node from the forward pass to the backward pass, and return it.

            Resolution order:
            1) check for methods on this class
            2) check the backward pass repository

            :param node: node on the forward pass
            :param given_gradients: output names on the forward node (for which the gradient will be connected as
                                           an input on the reverse node)
            :param required_gradients: input name on the forward node that the gradient should be generated for
            :return: the reversed node and gradient names for the connectors
        """
        log.debug("Reversing {}".format(node))

        # (1)
        if hasattr(self, "_reverse_" + type(node).__name__):
            return getattr(self, "_reverse_" + type(node).__name__)(
                node, given_gradients, required_gradients)

        # (2)
        impl = find_backward_implementation(self.sdfg,
                                            forward_state=self.forward_state,
                                            node=node)
        if impl is not None:
            return impl.backward(forward_node=node,
                                 context=BackwardContext(
                                     forward_state=self.forward_state,
                                     forward_sdfg=self.sdfg,
                                     backward_state=self.backward_state,
                                     backward_sdfg=self.backward_sdfg,
                                     backward_generator=self,
                                 ),
                                 given_gradients=given_gradients,
                                 required_gradients=required_gradients)

        raise AutoDiffException(
            "Unable to differentiate node type {}. Either add a pure forward implementation"
            "or a backward implementation to progress.".format(type(node)))

    def _reverse_NestedSDFG(
        self,
        node: nd.NestedSDFG,
        given_gradients: List[str],
        required_gradients: List[str],
    ) -> ReverseNodeReturnType:
        # check that the nested SDFG only has one state
        state_to_diff: SDFGState
        if len(node.sdfg.nodes()) != 1:
            # however we make an exception for initialization states; these are ignored
            is_init_state = [(state, is_initialization_state(state))
                             for state in node.sdfg.nodes()]
            num_non_init_states = sum(b for _, b in is_init_state)
            if num_non_init_states > 1:
                raise AutoDiffException(
                    "A nested SDFG may consist of at most one state (with the "
                    "exception of initalization states), found {} states".
                    format(num_non_init_states))
            state_to_diff = [state for state, b in is_init_state if not b][0]
        else:
            state_to_diff = node.sdfg.nodes()[0]

        reverse_sdfg = dace.SDFG(node.sdfg.name + "_backward")
        backward_state = reverse_sdfg.add_state()
        # recursive call
        gen = BackwardPassGenerator(sdfg=node.sdfg,
                                    state=state_to_diff,
                                    given_gradients=given_gradients,
                                    required_gradients=required_gradients,
                                    backward_sdfg=reverse_sdfg,
                                    backward_state=backward_state)
        backward_result, _, backward_input_arrays = gen.backward()

        # we need to defer add edges until after the arrays have been added because creation of the nested
        # sdfg fails otherwise
        deferred_edges = []

        inputs = set(backward_result.given_grad_names[name]
                     for name in given_gradients)
        # loop through the arrays that we need from the forward pass
        for name, desc in backward_input_arrays.items():
            # if the name is not already passed to the reverse SDFG node ...
            if name not in required_gradients and name not in node.in_connectors:
                # ... this array needs to be forwarded out of the forward SDFG (i.e. it is an intermediate value)
                # 1) add it to the current SDFG, and to self.backward_input_arrays
                # 2) add an out connector to the forward nested SDFG, add a write node to the current state, and an edge
                #    from the output to there
                # 3) add a read node to the backward state, and an edge into it

                desc, forwarded_name, _ = _walk_up_memlet_tree_through_view_nodes(
                    node.sdfg, state_to_diff, name)

                # (1)
                new_name = find_str_not_in_set(set(self.sdfg.arrays),
                                               forwarded_name + "_forwarded")
                if new_name in self.sdfg.arrays or new_name in self.backward_input_arrays:
                    raise AutoDiffException(
                        "Attempted to create array with name '{}', but it already existed"
                        .format(new_name))

                self.sdfg.add_datadesc(new_name, copy.deepcopy(desc))
                self.backward_input_arrays[new_name] = copy.deepcopy(desc)

                if self.separate_sdfgs:
                    to_add = copy.deepcopy(desc)
                    to_add.transient = False
                    self.backward_sdfg.add_datadesc(new_name, to_add)

                # (2)
                node.sdfg.arrays[forwarded_name].transient = False
                assert node.add_out_connector(forwarded_name)
                write = self.forward_state.add_write(new_name)
                self.forward_state.add_edge(
                    node, forwarded_name, write, None,
                    self.sdfg.make_array_memlet(new_name))

                # (3)
                read = self.backward_state.add_read(new_name)
                deferred_edges.append(
                    dict(
                        u=read,
                        u_connector=None,
                        v_connector=forwarded_name,
                        memlet=self.backward_sdfg.make_array_memlet(new_name)))
                inputs.add(forwarded_name)
            else:
                inputs.add(name)

        outputs = set(backward_result.required_grad_names[name]
                      for name in required_gradients)

        for inp in inputs:
            reverse_sdfg.arrays[inp].transient = False
        for outp in outputs:
            reverse_sdfg.arrays[outp].transient = False

        # actually create the sdfg and return it
        nsdfg = self.backward_state.add_nested_sdfg(
            reverse_sdfg,
            None,
            inputs=inputs,
            outputs=outputs,
        )

        for edge_args in deferred_edges:
            edge_args["v"] = nsdfg
            self.backward_state.add_edge(**edge_args)

        return nsdfg, BackwardResult(
            required_grad_names=backward_result.required_grad_names,
            given_grad_names=backward_result.given_grad_names)

    def _reverse_AccessNode(
        self,
        node: nd.AccessNode,
        given_gradients: List[str],
        required_gradients: List[str],
    ) -> ReverseNodeReturnType:
        rev = nd.AccessNode(self.array_grad_name(node.data),
                            access=_invert_access(node.access))
        self.backward_state.add_node(rev)
        return rev, BackwardResult(required_grad_names={None: None},
                                   given_grad_names={None: None})

    def _reverse_MapEntry(
        self,
        node: nd.MapEntry,
        given_gradients: List[str],
        required_gradients: List[str],
    ) -> ReverseNodeReturnType:

        required_grad_names = {
            n: _invert_map_connector(n)
            for n in required_gradients
        }
        given_grad_names = {
            n: _invert_map_connector(n)
            for n in given_gradients
        }
        result = BackwardResult(required_grad_names=required_grad_names,
                                given_grad_names=given_grad_names)
        rev = nd.MapExit(self.reverse_map[node.map])

        for conn in given_grad_names.values():
            assert rev.add_in_connector(conn)

        for conn in required_grad_names.values():
            assert rev.add_out_connector(conn)

        self.backward_state.add_node(rev)
        return rev, result

    def _reverse_MapExit(
        self,
        node: nd.MapExit,
        given_gradients: List[str],
        required_gradients: List[str],
    ):
        self.reverse_map[node.map] = copy.deepcopy(node.map)

        rev = nd.MapEntry(self.reverse_map[node.map])
        for conn in node.in_connectors:
            assert rev.add_in_connector(conn)

        for conn in node.out_connectors:
            assert rev.add_out_connector(conn)

        self.backward_state.add_node(rev)
        # yapf: disable
        return (
            rev,
            BackwardResult(required_grad_names={
                n: _invert_map_connector(n)
                for n in required_gradients
            },
                given_grad_names={
                    n: _invert_map_connector(n)
                    for n in given_gradients
                }),
        )
        # yapf: enable

    def _reverse_Tasklet(
        self,
        tasklet: nd.Tasklet,
        given_gradients: List[str],
        required_gradients: List[str],
    ) -> ReverseNodeReturnType:

        if tasklet.language is not dtypes.Language.Python:
            raise AutoDiffException(
                "Expected tasklet with language Python, got language {}".
                format(tasklet.language))

        # tasklets should have scalar inputs (can be relaxed)
        for _, _, _, _, memlet in self.forward_state.in_edges(tasklet):
            try:
                _is_int_value(memlet.subset.num_elements(), 1)
            except AutoDiffException as e:
                raise AutoDiffException(
                    "Autodiff only supported for tasklets with scalar inputs and outputs"
                ) from e

        for _, _, _, _, memlet in self.forward_state.out_edges(tasklet):
            try:
                _is_int_value(memlet.subset.num_elements(), 1)
            except AutoDiffException as e:
                raise AutoDiffException(
                    "Autodiff only supported for tasklets with scalar inputs and outputs"
                ) from e

        code_str = tasklet.code.as_string
        output_exprs = code_to_exprs(code_str, tasklet.in_connectors,
                                     tasklet.out_connectors)

        # for each output that an input is used in, there will be an entry for the expression of the
        # grad in this list in the final code snippet. When we generate the final code for the
        # reverse tasklet, we need to add them all up.
        rev_code = collections.defaultdict(list)

        # the outputs of the reversed nodes are the grads of inputs of the original node
        rev_outputs = set()
        rev_inputs = set()

        result = BackwardResult(required_grad_names={}, given_grad_names={})

        for output_conn in given_gradients:

            # for each output_conn...
            for inp in required_gradients:
                # ...add the code to generate {inp}_grad

                if inp not in result.required_grad_names:
                    # pick a name for the gradient
                    rev_output_grad_name = find_str_not_in_set(
                        rev_outputs, inp + "_gradient")
                    result.required_grad_names[inp] = rev_output_grad_name
                    rev_outputs.add(rev_output_grad_name)
                else:
                    rev_output_grad_name = result.required_grad_names[inp]

                output_expr = output_exprs[output_conn]

                # symbolically differentiate the output w.r.t inp
                diff_expr = output_expr.diff(sp.symbols(inp))

                if diff_expr.atoms(sp.Derivative):
                    # the final result contains a call to sp.Derivative
                    raise AutoDiffException(
                        "Unable to symbolically differentiate expression: {}".
                        format(diff_expr.expr))

                if output_conn not in result.given_grad_names:
                    # pick a name for the input gradient
                    rev_input_grad_name = find_str_not_in_set(
                        rev_inputs, output_conn + "_gradient")
                    result.given_grad_names[output_conn] = rev_input_grad_name
                else:
                    rev_input_grad_name = result.given_grad_names[output_conn]

                rev_inputs |= _symbols_to_strings(
                    diff_expr.free_symbols) | {rev_input_grad_name}

                diff_code_str = "{input} * ({diff_expr})".format(
                    input=rev_input_grad_name, diff_expr=str(diff_expr))

                # get the the final type of the gradient: this is just the type of the input connector we creating the
                # gradient for

                cands = list(
                    self.forward_state.in_edges_by_connector(tasklet, inp))
                if len(cands) != 1:
                    raise AutoDiffException(
                        f"Unexpected graph structure, could not find input edge for connector {inp}"
                        f" on tasklet {tasklet}")

                converted_code = cast_consts_to_type(
                    diff_code_str, self.sdfg.arrays[cands[0].data.data].dtype)
                converted_code = converted_code.replace("\n", " ")
                rev_code[rev_output_grad_name].append(converted_code)

        code = ""
        for output, exprs in rev_code.items():
            code += "\n" + output + " = " + " + ".join(exprs)

        rev = nd.Tasklet(
            "_" + tasklet.label + "_reverse_",
            inputs=rev_inputs,
            outputs=rev_outputs,
            code=code,
        )
        self.backward_state.add_node(rev)
        return rev, result
