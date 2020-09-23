"""Automatic Differentiation of SDFGStates.
   This module exposes the add_backward_pass method that can be used to add a backward pass to an
   SDFGState.
"""
import dace
from dace import Memlet, SDFG, SDFGState
import dace.sdfg.nodes as nd
from dace.sdfg import ScopeSubgraphView
from dace.frontend.operations import detect_reduction_type
from dace.frontend.python.replacements import _argminmax, _elementwise
from dace import dtypes, data as dt
from dace.libraries.standard import Reduce

import ast
import itertools
from collections import deque, defaultdict
from copy import deepcopy as dc
from typing import Iterator, Tuple, Deque, Dict, Set, List, Union, Optional

import aenum
import sympy as sp
from dace.sdfg.state import StateSubgraphView
from dace.sdfg.utils import dfs_topological_sort
from sympy.parsing.sympy_parser import parse_expr
from astunparse import unparse


class AutoDiffException(Exception):
    """Base class for all exceptions related to automatic differentiation"""
    pass


def _strings_to_symbols(strings: Set[str]) -> Set[sp.Symbol]:
    return {sp.symbols(string) for string in strings}


def _symbols_to_strings(symbs: Set[sp.Symbol]) -> Set[str]:
    return {str(symb) for symb in symbs}

def _add_backward_state_to_sdfg(sdfg: SDFG, forward: SDFGState, backward: SDFGState, arrs: dict):
    # add the new backward state to the SDFG
    base_label = None if forward.label is None else forward.label + "_backward"
    i = len(sdfg)
    while True:
        # Append a number. If the state already exists, increment the
        # number until it doesn't
        label = "{}_{}".format(base_label, i)
        if any([s.label == label for s in sdfg.nodes()]):
            i += 1
        else:
            break

    # add all new arrays
    for name, desc in arrs.items():
        sdfg.add_datadesc(name, desc)

    backward._parent = sdfg
    sdfg.add_node(backward, is_start_state=False)
    sdfg.add_edge(forward, backward, dace.InterstateEdge())

def is_initialization_state(state: SDFGState) -> bool:
    """Check if state is an initialization state, i.e. it initializes one or more arrays with zero values

    This is an over- and underapproximation; e.g. it doesn't check that the whole array is initalized with zero, but
    it also clearly doesn't detect all possible ways of filling an array with zero.
    """
    for sink in state.sink_nodes():
        # sink nodes should be AccessNodes
        if type(sink) is not nd.AccessNode:
            return False

        edges = list(state.bfs_edges(sink, reverse=True))

        # we expect at least one in edge
        if len(edges) == 0:
            return False

        def tasklet_writes_zero(tasklet: nd.Tasklet) -> bool:
            if type(tasklet) is not nd.Tasklet:
                return False

            # there shouldn't be any in_connectors
            if tasklet.in_connectors:
                return False

            # every output connector should be set to zero in the tasklet
            output_exprs = code_to_exprs(tasklet.code.as_string,
                                         tasklet.in_connectors,
                                         tasklet.out_connectors)
            for expr in output_exprs.values():
                if not _is_int_value(expr, 0):
                    return False
            return True

        if type(edges[0].src) is nd.MapExit:
            # we have a map exit, the next nodes should be tasklets writing 0 and finally an exit node
            for edge in edges[1:-1]:
                if not tasklet_writes_zero(edge.src):
                    return False

            if type(edges[-1].src) is not nd.MapEntry:
                return False

        elif type(edges[0].src) is nd.Tasklet:
            # there should be no other nodes in this component
            if len(edges) != 1:
                return False

            if not tasklet_writes_zero(edges[0].src):
                return False
        else:
            return False

    return True


def code_to_exprs(code: str, inputs: Set[str],
                  outputs: Set[str]) -> Dict[str, sp.Expr]:
    """Convert a python string to a set of (simplified) symbolic sympy expressions. Currently, this
    supports only code consisting of assignment statements.
    :param code: the code to convert
    :param code: the inputs (i.e. the defined variables) for the code
    :param code: the outputs to generate simplified expressions for
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
        exec(code_fn)

        # no idea why, but simply calling symbolic_execution doesn't work
        results = vars()["symbolic_execution"](
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
    if type(value) is int:
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
    node.add_in_connector("IN_{}".format(i))
    node.add_out_connector("OUT_{}".format(i))
    return "IN_{}".format(i), "OUT_{}".format(i)


def _invert_map_connector(conn):
    if conn[:2] == "IN":
        return "OUT" + conn[2:]
    elif conn[:3] == "OUT":
        return "IN" + conn[3:]
    else:
        raise AutoDiffException(
            "Could not parse map connector '{}'".format(conn))


def _has_inplace_operation(state: dace.SDFGState) -> bool:
    """Returns true if state has any inplace operations
    Note that this method is currently much stronger than required; some of the constrains can be
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

    # edges with scalar memlets can be used to connect two code nodes together. If this feature is
    # used, it should be done using a new scalar everytime.
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


def _get_matching_entry(state: SDFGState, map_exit: nd.MapExit) -> nd.MapEntry:
    """Get the matching `MapEntry` for a `MapExit`"""
    cands = [
        node for node in state.nodes()
        if isinstance(node, nd.MapEntry) and node.map is map_exit.map
    ]

    if len(cands) != 1:
        raise AutoDiffException(
            "More than one map entry found for map {}".format(map_exit.map))
    return cands[0]


class BackwardPassGenerator:
    """Class that holds the state for one backward pass creation"""
    def __init__(self, *, sdfg: SDFG, state: SDFGState,
                 outputs: List[Union[nd.AccessNode,
                                     str]], inputs: List[Union[nd.AccessNode,
                                                               str]],
                 grads: Optional[List[Union[nd.AccessNode, str]]]):
        def str_to_access(data: str, source: str) -> nd.AccessNode:
            matches = [
                node for node in state.nodes()
                if type(node) is nd.AccessNode and node.data == data
            ]
            if len(matches) != 1:
                raise AutoDiffException(
                    "Expected to find exactly one node with data"
                    " '{}' in {}, but found {}".format(data, source,
                                                       len(matches)))
            return matches[0]

        outputs = [
            n if type(n) is nd.AccessNode else str_to_access(n, "outputs")
            for n in outputs
        ]
        inputs = [
            n if type(n) is nd.AccessNode else str_to_access(n, "inputs")
            for n in inputs
        ]

        if grads is not None:
            grads = [
                n if type(n) is nd.AccessNode else str_to_access(n, "grads")
                for n in grads
            ]

        self.sdfg = sdfg
        self.forward_state = state
        # will be initialized in the backward call
        self.backward_state: SDFGState = None
        self.backward_arrays: Dict[str, dt.Array] = {}

        self.outputs = outputs
        self.inputs = inputs
        self.grads = grads

        # hooks that are executed after the backward pass is complete
        self._post_grad_hooks = []
        # this is a mapping from forward node -> backward node, and forward map -> backward map
        self.reverse_map = {}

        # ??? TODO
        self.grad_memlets: Dict[str, List[Memlet]] = defaultdict(list)

        # checks if backward has already been applied
        self._applied = False

    def backward(self) -> Tuple[SDFGState, Dict[str, dt.Array]]:
        """ Generate the backward pass, add it to the SDFG and, return it as an SDFGState """

        self.backward_state = dace.SDFGState(label=("" if self.forward_state.label is None else self.forward_state.label + "_backward"))
        if self._applied:
            raise AutoDiffException(
                "Backward may only be called once. Instantiate a new BackwardPassGenerator."
            )

        any_non_scalar = False
        for outp_idx, outp in enumerate(self.outputs):
            outp_arr = self.sdfg.arrays[outp.data]
            if not _is_int_value(outp_arr.total_size, 1):
                any_non_scalar = True

        if self.grads is None:
            # this is ok if we only have one scalar:
            if len(self.outputs) == 1 and not any_non_scalar:
                pass
            else:
                raise AutoDiffException(
                    "If function output is not one scalar, grads should be provided"
                )
        else:
            if len(self.grads) != len(self.outputs):
                raise AutoDiffException(
                    "If grads are provided, their length should match the length of outputs, but"
                    " len(grads) = {} != {} = len(outputs)".format(
                        len(self.grads), len(self.outputs)))

        for outp in self.outputs:
            if outp not in self.forward_state:
                raise AutoDiffException(
                    "Could not find output {} in state {}".format(
                        outp, self.forward_state))

        for inp in self.inputs:
            if inp not in self.forward_state:
                raise AutoDiffException(
                    "Could not find input {} in state {}".format(
                        inp, self.forward_state))

        # check for inplace operations (i.e. duplicated access nodes)
        if _has_inplace_operation(self.forward_state):
            raise AutoDiffException(
                "Inplace operations are currently not supported in autodiff")

        # determine which nodes we need to reverse; this forms the subgraph we will differentiate:
        # we do a reverse bfs and a forward bfs, then take the intersection of nodes found
        forward_nodes = {
            n
            for e in self.forward_state.bfs_edges(self.inputs) for n in [e.src, e.dst]
        }
        backward_nodes = {
            n
            for e in self.forward_state.bfs_edges(self.outputs, reverse=True)
            for n in [e.src, e.dst]
        }

        forward_subgraph = StateSubgraphView(
            self.forward_state, list(forward_nodes.intersection(backward_nodes)))
        # recursively reverse the subgraph
        self._reverse_subgraph(forward_subgraph)

        # execute any hooks that were added during the call
        for hook in self._post_grad_hooks:
            hook()
        self._applied = True
        return self.backward_state, self.backward_arrays

    def get_grad_name(self, conn: str, node: nd.Node, in_connector: bool):
        if type(node) in [nd.MapExit, nd.MapEntry]:
            return _invert_map_connector(conn)

        if type(node) is Reduce:
            # in this case the reverse node will be a NSDFG, which can't have None as connectors
            return "_none" if in_connector else "_none"

        if conn is None:
            return None
        else:
            return conn + "_grad"

    def _reverse_subgraph(self, subgraph: StateSubgraphView):
        """Reverse a given subgraph. All nodes in the subgraph will be reversed."""

        # a reversed topological sort is a topological sort on the reverse graph
        for node in reversed(
                list(dfs_topological_sort(subgraph, subgraph.source_nodes()))):

            # output name on the forward node (for which the gradient will be connected as an input on the reverse node)
            output_grad_connectors = [
                edge.src_conn for edge in subgraph.out_edges(node)
            ]

            # input name on the forward node that the gradient should be generated for
            input_grad_connectors = [
                edge.dst_conn for edge in subgraph.in_edges(node)
            ]

            rev = self._get_reverse_node(node, output_grad_connectors,
                                         input_grad_connectors)

            self.reverse_map[node] = rev

            # connect the gradients of the outputs (as inputs)
            for _, output_conn, dest_node, input_conn, memlet in subgraph.out_edges(
                    node):
                if detect_reduction_type(memlet.wcr) not in [
                        None,
                        dtypes.ReductionType.Sum,
                ]:
                    raise AutoDiffException(
                        "Unsupported reduction type {}".format(
                            detect_reduction_type(memlet.wcr)))

                memlet = dc(memlet)

                # remove the WCR since these are now read edges
                memlet.wcr = None

                if memlet.data not in self.grad_memlets:
                    # this grad hasn't been written before: initialize it
                    array = self.sdfg.arrays[memlet.data]

                    if type(array) is not dt.Scalar and type(
                            array) is not dt.Array:
                        raise AutoDiffException(
                            "Unsupported data descriptor {}".format(array))

                    # this can clearly fail if someone chooses annoying array names; let's
                    # ignore this for now
                    cloned_datadesc = dc(array)
                    assert memlet.data + "_grad" not in self.backward_arrays
                    self.backward_arrays[memlet.data + "_grad"] = cloned_datadesc

                self.grad_memlets[memlet.data].append(memlet)
                memlet.data = memlet.data + "_grad"

                self.backward_state.add_edge(
                    self.reverse_map[dest_node],
                    self.get_grad_name(input_conn,
                                       dest_node,
                                       in_connector=False),
                    rev,
                    self.get_grad_name(output_conn, node, in_connector=True),
                    memlet,
                )

            if isinstance(node, nd.AccessNode):
                # this means we are writing out a grad to an array. In this case, we need to set
                # all incoming memlets to WCR Sum
                # TODO @orausch there could/should be an intersection check here to remove this if not required...
                for edge in self.backward_state.in_edges(rev):
                    for path_edge in self.backward_state.memlet_tree(edge):
                        path_edge.data.wcr = "lambda x, y: x + y"

            # connect any required inputs from the forward pass
            required_inputs = set(rev.in_connectors).difference(
                self.get_grad_name(conn, node, True)
                for conn in output_grad_connectors)

            self._connect_inputs(subgraph, node, required_inputs)

    def _connect_inputs(self, subgraph, forward_node, required_inputs):
        """For each connector in `required_inputs`, connect the reversed node of `node` that input
        from the forward pass, routing through maps from the backward pass if required.
        """

        for edge in subgraph.graph.in_edges(forward_node):
            if edge.dst_conn in required_inputs:
                path = subgraph.graph.memlet_path(edge)
                conn_map = dict()

                for i, traversed_edge in enumerate(path):
                    throw = False
                    src = None
                    dst = None
                    src_conn = traversed_edge.src_conn
                    dst_conn = traversed_edge.dst_conn

                    if i == 0:
                        # the start of the path is in the forward pass.
                        if type(traversed_edge.src) is nd.AccessNode:
                            # we add an access node to the backward pass
                            src = self.backward_state.add_access(traversed_edge.src.data)
                        else:
                            src = traversed_edge.src

                        throw |= type(traversed_edge.dst) is not nd.MapEntry

                    if i == len(path) - 1:
                        # the end of the path should be in the backward pass
                        dst = self.reverse_map[traversed_edge.dst]
                        throw |= type(traversed_edge.src) is not nd.MapEntry

                    if i != 0 and i != len(path) - 1:
                        # leave dst and src as None; we will later replace them with the correct map nodes
                        throw |= type(traversed_edge.src) is not nd.MapEntry
                        throw |= type(traversed_edge.dst) is not nd.MapEntry

                    if len(path) == 1:
                        # if len path == 1, throw will be true because the ends are not maps
                        # however, this is fine in this case as long as we have code -> code or access -> code
                        throw = not (
                            (isinstance(traversed_edge.src, nd.CodeNode) and
                             isinstance(traversed_edge.dst, nd.CodeNode)) or
                            (isinstance(traversed_edge.src, nd.AccessNode)
                             and isinstance(traversed_edge.dst, nd.CodeNode)))

                    if throw:
                        raise AutoDiffException("Unexpected graph structure")

                    if dst is None:
                        dst = self._find_backward_entry_node_for_map_entry(
                            subgraph.graph, traversed_edge.dst)
                        dst_conn, _src_conn = _add_through_connector(dst)
                        conn_map[dst] = _src_conn

                    if src is None:
                        src = self._find_backward_entry_node_for_map_entry(
                            subgraph.graph, traversed_edge.src)
                        src_conn = conn_map[src]

                    self.backward_state.add_edge(src, src_conn, dst, dst_conn,
                                                 traversed_edge.data)

    def _find_backward_entry_node_for_map_entry(
            self, graph, entry_node: nd.MapEntry) -> nd.MapExit:
        """Find the entry node in the backward pass corresponding to the exit node opened by
        `entry_node` (where `entry_node` is a node from the forward pass).
        """
        src_candidates = [
            node for node in self.backward_state.nodes() if type(node) is nd.MapEntry
            and node.map == self.reverse_map[entry_node.map]
        ]
        if len(src_candidates) != 1:
            # this shouldn't happen; if we are within a scope, the exit nodes
            # for the scope should already exist in the backward pass
            raise AutoDiffException("Invalid graph")

        return src_candidates[0]

    def _get_reverse_node(self, node, output_grad_connectors,
                          input_grad_connectors) -> nd.Node:
        """ Add the reverse node for a node from the forward pass to the backward pass, and return it.

            :param node: node on the forward pass
            :param output_grad_connectors: output names on the forward node (for which the gradient will be connected as
                                           an input on the reverse node)
            :param input_grad_connectors: input name on the forward node that the gradient should be generated for
        """

        if isinstance(node, dace.nodes.LibraryNode):
            pass

        if not hasattr(self, "_reverse_" + type(node).__name__):
            raise AutoDiffException("Unsupported node type {}".format(
                type(node)))

        return getattr(self, "_reverse_" + type(node).__name__)(
            node, output_grad_connectors, input_grad_connectors)

    def _reverse_NestedSDFG(
        self,
        node: nd.NestedSDFG,
        output_grad_connectors: List[str],
        input_grad_connectors: List[str],
    ):
        # check that the nested SDFG only has one state
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
            state_to_diff = [state for state, b in is_init_state if b][0]
        else:
            state_to_diff = node.sdfg.nodes()[0]

        # recursive call
        gen = BackwardPassGenerator(
            sdfg=node.sdfg,
            state=state_to_diff,
            outputs=output_grad_connectors,
            inputs=input_grad_connectors,
            grads=[conn + "_grad" for conn in output_grad_connectors])
        gen.backward()
        SDFGState.add_tasklet()

    def _reverse_AccessNode(
        self,
        node: nd.AccessNode,
        output_grad_connectors: List[str],
        input_grad_connectors: List[str],
    ):
        rev = nd.AccessNode(node.data + "_grad",
                            access=_invert_access(node.access))
        self.backward_state.add_node(rev)
        return rev

    def _reverse_MapEntry(
        self,
        node: nd.MapEntry,
        output_grad_connectors: List[str],
        input_grad_connectors: List[str],
    ):
        rev = nd.MapExit(self.reverse_map[node.map])

        for conn in node.in_connectors:
            rev.add_in_connector(conn)

        for conn in node.out_connectors:
            rev.add_out_connector(conn)

        self.backward_state.add_node(rev)
        return rev

    def _reverse_MapExit(
        self,
        node: nd.MapExit,
        output_grad_connectors: List[str],
        input_grad_connectors: List[str],
    ):
        self.reverse_map[node.map] = dc(node.map)

        rev = nd.MapEntry(self.reverse_map[node.map])
        for conn in node.in_connectors:
            rev.add_in_connector(conn)

        for conn in node.out_connectors:
            rev.add_out_connector(conn)

        self.backward_state.add_node(rev)
        return rev

    def _reverse_Tasklet(
        self,
        tasklet: nd.Tasklet,
        output_grad_connectors: List[str],
        input_grad_connectors: List[str],
    ) -> nd.Tasklet:

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
        rev_code = defaultdict(list)

        # the outputs of the reversed nodes are the grads of inputs of the original node
        rev_outputs = set()
        rev_inputs = set()

        for output_conn in output_grad_connectors:

            # for each output_conn...
            for inp in input_grad_connectors:
                # ...add the code to generate {inp}_grad
                rev_output_grad_name = self.get_grad_name(inp, tasklet, False)
                rev_outputs.add(rev_output_grad_name)

                output_expr = output_exprs[output_conn]

                # symbolically differentiate the output by inp
                diff_expr = output_expr.diff(sp.symbols(inp))

                if diff_expr.atoms(sp.Derivative):
                    # the final result contains a call to sp.Derivative
                    raise AutoDiffException(
                        "Unable to symbolically differentiate expression: {}".
                        format(diff_expr.expr))

                rev_input_grad_name = self.get_grad_name(
                    output_conn, tasklet, True)
                rev_inputs |= _symbols_to_strings(
                    diff_expr.free_symbols) | {rev_input_grad_name}

                rev_code[rev_output_grad_name].append(
                    "{input} * ({diff_expr})".format(input=rev_input_grad_name,
                                                     diff_expr=str(diff_expr)))

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
        return rev
