"""Automatic Differentiation of SDFGStates.
   This module exposes the add_backward_pass method that can be used to add a backward pass to an
   SDFGState.
"""
from collections import defaultdict
from copy import deepcopy as dc
from typing import Tuple, Dict, Set, List, Union, cast

import dace
import dace.sdfg.nodes as nd
import sympy as sp
from dace import Memlet, SDFG, SDFGState
from dace import dtypes, data as dt
from dace.frontend.operations import detect_reduction_type
from dace.libraries.standard import Reduce
from dace.sdfg.graph import MultiConnectorEdge
from dace.sdfg.state import StateSubgraphView
from dace.sdfg.utils import dfs_topological_sort
from dace.transformation.transformation import ExpandTransformation

from daceml.onnx.implementation_abc import ONNXForward
from daceml.onnx.nodes.onnx_op import ONNXOp


class AutoDiffException(Exception):
    """Base class for all exceptions related to automatic differentiation"""
    pass


def _strings_to_symbols(strings: Set[str]) -> Set[sp.Symbol]:
    return {sp.symbols(string) for string in strings}


def _symbols_to_strings(symbs: Set[sp.Symbol]) -> Set[str]:
    return {str(symb) for symb in symbs}


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


class BackwardPassGenerator(object):
    """ Class that holds the state for one backward pass creation.

        See autodiff.py, _reverse_NestedSDFG and pytorch.py for examples of usage.

        :param state: the forward pass to differentiate should be in this state
        :param outputs: the outputs that gradients must be provided for (i.e. access nodes will be created for these)
        :param inputs: the inputs to generate gradients for
        :param backward_sdfg: the sdfg the backward pass will be contained in. If it is the same as the forward_sdfg,
                              outputs must be a list containing a single scalar.
        :param backward_state: the state which the backward pass should be added to (must be added to `backward_sdfg`
                               before calling this method).
    """
    def __init__(
            self,
            *,
            sdfg: SDFG,
            state: SDFGState,
            outputs: List[Union[nd.AccessNode, str]],
            inputs: List[Union[nd.AccessNode, str]],
            backward_sdfg: SDFG,  # this can be the same as SDFG
            backward_state: SDFGState):

        if backward_state not in backward_sdfg.nodes():
            raise AutoDiffException(
                "Expected to find backward_state in backward_sdfg")

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

        self.outputs = outputs
        self.inputs = inputs

        self.input_names = {n.data for n in inputs}
        self.output_names = {n.data for n in outputs}

        self.sdfg = sdfg
        self.forward_state = state
        self.backward_sdfg = backward_sdfg
        self.backward_state: SDFGState = backward_state

        # arrays descs for the gradients
        self.backward_grad_arrays: Dict[str, dt.Array] = {}
        # arrays descs for inputs that are required from the forward pass
        self.backward_input_arrays: Dict[str, dt.Array] = {}

        # if we have nested sdfgs, arrays must be added to the forward pass
        self.nested_sdfg_forwarded_arrays: Dict[str, dt.Array] = {}

        # hooks that are executed after the backward pass is complete
        self._post_grad_hooks = []
        # this is a mapping from forward node -> backward node, and forward map -> backward map
        self.reverse_map = {}

        # grad name -> all memlets that write to this grad
        self.grad_memlets: Dict[str, List[Memlet]] = defaultdict(list)

        # checks if backward has already been applied
        self._applied = False

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

        if sdfg is backward_sdfg:
            # this only makes sense if the output is a single scalar.
            if len(outputs) != 1:
                raise AutoDiffException(
                    "When the forward sdfg is the same as the backward sdfg, outputs must be a"
                    "single scalar")
            if not _is_int_value(sdfg.arrays[outputs[0].data].total_size, 1):
                raise AutoDiffException(
                    "When the forward sdfg is the same as the backward sdfg, outputs must be a"
                    "single scalar")
            self.separate_sdfgs = False
        else:
            self.separate_sdfgs = True

    def backward(self) -> Tuple[Dict[str, dt.Array], Dict[str, dt.Array]]:
        """ Generate the backward pass in backward_state.

            Returns
            1. dict of data descriptors for the gradients (i.e. the outputs of the backward pass)
            2. dict of data descriptors of required outputs from the forward pass. These need to be added to the parent
               SDFG of the backward pass.

            All returned data descriptors are not transient.
        """

        # expand ONNXOps. This should later on be changed to check if the expansion is differentiable and if not, move
        # on to the next expansion. For now we will just apply the first one that matches.

        # TODO this should cover libnodes in general, not just ONNXOps. This would mean the reduce stuff can be moved
        # out of this class
        for node in self.forward_state.nodes():
            if isinstance(node, ONNXOp):
                for impl in ONNXForward.registered_implementations(
                        node.schema.name):
                    if impl.forward_can_be_applied(node, self.forward_state,
                                                   self.sdfg):
                        # try to apply the expansion
                        class Expansion(ExpandTransformation):
                            environments = []
                            _expansion_result = None

                            @classmethod
                            def expansion(cls, node, state, sdfg):
                                return impl.forward(node, state, sdfg)

                            @classmethod
                            def postprocessing(cls, sdfg, state, expansion):
                                cls._expansion_result = expansion

                        Expansion._match_node = type(node)
                        Expansion.apply_to(self.sdfg,
                                           verify=False,
                                           match_node=node)

        if self._applied:
            raise AutoDiffException(
                "Backward may only be called once. Instantiate a new BackwardPassGenerator."
            )

        # determine which nodes we need to reverse; this forms the subgraph we will differentiate:
        # we do a reverse bfs and a forward bfs, then take the intersection of nodes found
        forward_nodes = {
            n
            for e in self.forward_state.bfs_edges(self.inputs)
            for n in [e.src, e.dst]
        }
        backward_nodes = {
            n
            for e in self.forward_state.bfs_edges(self.outputs, reverse=True)
            for n in [e.src, e.dst]
        }

        forward_subgraph = StateSubgraphView(
            self.forward_state,
            list(forward_nodes.intersection(backward_nodes)))
        # recursively reverse the subgraph
        self._reverse_subgraph(forward_subgraph)

        # execute any hooks that were added during the call
        for hook in self._post_grad_hooks:
            hook()
        self._applied = True
        return self.backward_grad_arrays, self.backward_input_arrays

    def get_grad_name(self, conn: str, node: nd.Node):
        if type(node) in [nd.MapExit, nd.MapEntry]:
            return _invert_map_connector(conn)

        if type(node) is Reduce:
            # in this case the reverse node will be a NSDFG, which can't have None as connectors
            return self.sdfg.temp_data_name()

        if conn is None:
            return None
        else:
            return conn + "_grad"

    def _init_grad(self, data: str):
        """Add a state where `data` is initialized with zero.
           self.sdfg.arrays[data] should have type Union[dt.Array, dt.Scalar]
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
        else:
            raise AutoDiffException(
                "Unsupported data descriptor {}".format(arr))

    def _reverse_subgraph(self, subgraph: StateSubgraphView):
        """ Reverse a given subgraph. All nodes in the subgraph will be reversed. """

        # a reversed topological sort is a topological sort on the reverse graph
        for node in reversed(
                list(dfs_topological_sort(subgraph, subgraph.source_nodes()))):

            try:
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

                # connect the required inputs of the reverse node: the gradients and any output values from the forward pass
                self._connect_input_gradients(subgraph, node)
                self._connect_forward_inputs(subgraph, node)

                if isinstance(node, nd.AccessNode):
                    # this means we are writing out a grad to an array. In this case, we need to set
                    # all incoming memlets to WCR Sum
                    # TODO @orausch there could/should be an intersection check here to remove this if not required...
                    for edge in self.backward_state.in_edges(rev):
                        for path_edge in self.backward_state.memlet_tree(edge):
                            path_edge.data.wcr = "lambda x, y: x + y"

            except AutoDiffException as e:
                raise AutoDiffException(
                    "Failed at node {}".format(node)) from e

    def _connect_input_gradients(self, subgraph: StateSubgraphView,
                                 forward_node):
        """ Connect the gradients of the outputs of forward_node as inputs to the corresponding reverse node. """

        for _, output_conn, dest_node, input_conn, memlet in subgraph.out_edges(
                forward_node):
            if detect_reduction_type(memlet.wcr) not in [
                    None,
                    dtypes.ReductionType.Sum,
            ]:
                raise AutoDiffException("Unsupported reduction type {}".format(
                    detect_reduction_type(memlet.wcr)))

            memlet = dc(memlet)

            # TODO what happens when multiple edges read from the same place? Should be fine because of the grad sum
            # WCR, but double check this

            # remove the WCR since these are now read edges
            memlet.wcr = None

            if memlet.data not in self.grad_memlets:
                # this grad hasn't been written before: initialize it
                array = self.sdfg.arrays[memlet.data]

                if type(array) is not dt.Scalar and type(
                        array) is not dt.Array:
                    raise AutoDiffException(
                        "Unsupported data descriptor {}".format(array))

                cloned_datadesc = dc(array)

                # only the grads of the inputs and the outputs are not transient
                cloned_datadesc.transient = memlet.data not in self.input_names and memlet.data not in self.output_names

                # TODO test with identical nodes after one another; should fail (come up with better solution)
                # this can clearly fail if someone chooses annoying array names; ignore this for now
                if memlet.data + "_grad" in self.backward_grad_arrays:
                    AutoDiffException(
                        "Unable to create array with name '{}'; it already exists"
                        .format(memlet.data + "_grad"))

                self.backward_grad_arrays[memlet.data +
                                          "_grad"] = cloned_datadesc
                self.backward_sdfg.arrays[memlet.data +
                                          "_grad"] = dc(cloned_datadesc)

                if cloned_datadesc.transient:
                    self._init_grad(memlet.data + "_grad")

            self.grad_memlets[memlet.data].append(memlet)
            memlet.data = memlet.data + "_grad"

            self.backward_state.add_edge(
                self.reverse_map[dest_node],
                self.get_grad_name(input_conn, dest_node),
                self.reverse_map[forward_node],
                self.get_grad_name(output_conn, forward_node),
                memlet,
            )

    def _connect_forward_inputs(self, subgraph: StateSubgraphView,
                                forward_node):
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
            self.get_grad_name(edge.src_conn, forward_node)
            for edge in subgraph.out_edges(forward_node))

        edges_to_connect = (edge for edge in subgraph.in_edges(forward_node)
                            if edge.dst_conn in required_inputs)

        for edge in edges_to_connect:
            path = subgraph.memlet_path(edge)

            ####################################
            # we can only add this edge if the first node in the path not within a map scope. Otherwise the value read
            # in the backward pass might be different to the one read in the forward pass

            if subgraph.scope_dict()[path[0].src] is not None:
                parent = subgraph.scope_dict()[path[0].src]
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
                if not (type(path[0].src) is nd.AccessNode
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
                        if type(path_edge.dst) is not nd.MapEntry:
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
                        if type(path_edge.src) is not nd.AccessNode:
                            raise AutoDiffException(
                                "Unexpected graph structure: expected memlet path that starts with an "
                                "AccessNode and ends with CodeNode")

                        new_edge_src_conn = None
                        if path_edge.src in self.reverse_map:
                            new_edge_src = self.reverse_map[path_edge.src]
                        else:
                            # Add an AccessNode for this to the backward pass
                            data_name = path_edge.src.data
                            data_desc = dc(self.sdfg.arrays[data_name])
                            assert data_name not in self.backward_input_arrays

                            if self.separate_sdfgs:
                                data_desc.transient = False
                                self.backward_sdfg.add_datadesc(
                                    data_name, data_desc)

                            self.backward_input_arrays[data_name] = data_desc

                            new_edge_src = self.backward_state.add_access(
                                data_name)
                            self.reverse_map[path_edge.src] = new_edge_src
                    else:
                        # if we have more than one edge, check that all intermediate nodes are MapEntry
                        if type(path_edge.src) is not nd.MapEntry:
                            raise AutoDiffException(
                                "Unexpected graph structure")

                        new_edge_src = self._find_backward_entry_node_for_map_entry(
                            path_edge.src)
                        new_edge_src_conn = conn_map[new_edge_src]

                    self.backward_state.add_edge(new_edge_src,
                                                 new_edge_src_conn,
                                                 new_edge_dst,
                                                 new_edge_dst_conn,
                                                 dc(path_edge.data))

    def _find_backward_entry_node_for_map_entry(
            self, entry_node: nd.MapEntry) -> nd.MapExit:
        """Find the entry node in the backward pass corresponding to the exit node opened by
        `entry_node` (where `entry_node` is a node from the forward pass).
        """
        src_candidates = [
            cast(nd.MapExit, node) for node in self.backward_state.nodes()
            if type(node) is nd.MapEntry
            and node.map == self.reverse_map[entry_node.map]
        ]
        if len(src_candidates) != 1:
            # this shouldn't happen; if we are within a scope, the exit nodes
            # for the scope should already exist in the backward pass
            raise AutoDiffException("Invalid graph")

        return src_candidates[0]

    def _get_reverse_node(
            self, node, output_grad_connectors,
            input_grad_connectors) -> Union[nd.Node, Tuple[nd.Node, nd.Node]]:
        """ Add the reverse node for a node from the forward pass to the backward pass, and return it.

            Resolution order:
            1) check for methods on this class
            2) check the backward pass repository
            3) check the forward pass repository, and try to differentiate the expansion (these should already be
               expanded, so no work is necessary).

            :param node: node on the forward pass
            :param output_grad_connectors: output names on the forward node (for which the gradient will be connected as
                                           an input on the reverse node)
            :param input_grad_connectors: input name on the forward node that the gradient should be generated for
        """

        if isinstance(node, dace.nodes.LibraryNode):
            pass

        # (1)
        if hasattr(self, "_reverse_" + type(node).__name__):
            return getattr(self, "_reverse_" + type(node).__name__)(
                node, output_grad_connectors, input_grad_connectors)

        # (2)
        # TODO: will be needed for a good softmax diff

        raise AutoDiffException("Unsupported node type {}".format(type(node)))

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

        reverse_sdfg = dace.SDFG(node.sdfg.name + "_backward")
        backward_state = reverse_sdfg.add_state()
        # recursive call
        gen = BackwardPassGenerator(sdfg=node.sdfg,
                                    state=state_to_diff,
                                    outputs=output_grad_connectors,
                                    inputs=input_grad_connectors,
                                    backward_sdfg=reverse_sdfg,
                                    backward_state=backward_state)
        backward_grad_arrays, backward_input_arrays = gen.backward()

        # we need to defer add edges until after the arrays have been added because creation of the nested
        # sdfg fails other wise
        edges_to_add = []

        # loop through the arrays that we need from the forward pass
        for name, desc in backward_input_arrays.items():
            # if the name is not already passed to the reverse SDFG node ...
            if name not in input_grad_connectors:
                # ... this array needs to be forwarded out of the forward SDFG (i.e. it is an intermediate value)
                # 1) add it to the current SDFG, and to self.backward_input_arrays
                # 2) add an out connector to the forward nested SDFG, add a write node to the current state, and an edge
                #    from the output to there
                # 3) add a read node to the backward state, and an edge into it

                # (1)
                new_name = name + "_forwarded"
                if name in self.sdfg.arrays:
                    raise AutoDiffException(
                        "Attempted to create array with name '{}', but it already existed"
                        .format(new_name))

                if name in self.backward_input_arrays:
                    raise AutoDiffException(
                        "Attempted to create array with name '{}', but it already existed"
                        .format(new_name))

                self.sdfg.add_datadesc(new_name, dc(desc))
                self.backward_input_arrays[new_name] = dc(desc)

                if self.separate_sdfgs:
                    to_add = dc(desc)
                    to_add.transient = False
                    self.backward_sdfg.add_datadesc(new_name, to_add)

                # (2)
                node.add_out_connector(name)
                write = self.forward_state.add_write(new_name)
                self.forward_state.add_edge(
                    node, name, write, None,
                    self.sdfg.make_array_memlet(new_name))

                # (3)
                # TODO write test that needs this, then write this

        inputs = set(
            self.get_grad_name(name, node)
            for name in output_grad_connectors).union(backward_input_arrays)
        outputs = set(
            self.get_grad_name(name, node) for name in input_grad_connectors)

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

        # add the deferred edges
        for edge in edges_to_add:
            self.backward_state.add_edge(nsdfg, *edge)
        return nsdfg

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

    def _reverse_Reduce(self, node: Reduce, output_grad_connectors: List[str],
                        input_grad_connectors: List[str]):

        reduction_type = detect_reduction_type(node.wcr)

        # NOTE: Reduce nodes should have exactly one input and one output edge
        if len(output_grad_connectors) != 1:
            raise AutoDiffException(
                "recieved invalid SDFG: reduce node {} should have exactly one output edge"
                .format(node))

        if len(input_grad_connectors) != 1:
            raise AutoDiffException(
                "recieved invalid SDFG: reduce node {} should have exactly one input edge"
                .format(node))

        cands = [
            edge for edge in self.forward_state.in_edges(node)
            if edge.dst_conn == input_grad_connectors[0]
        ]
        if len(cands) != 1:
            raise AutoDiffException("recieved invalid SDFG")
        input_array = self.sdfg.arrays[cands[0].data.data]

        cands = [
            edge for edge in self.forward_state.out_edges(node)
            if edge.src_conn == output_grad_connectors[0]
        ]
        if len(cands) != 1:
            raise AutoDiffException("recieved invalid SDFG")
        output_array = self.sdfg.arrays[cands[0].data.data]

        all_axes: List[int] = list(range(len(input_array.shape)))
        reduce_axes: List[int] = all_axes if node.axes is None else node.axes
        non_reduce_axes: List[int] = [
            i for i in all_axes if i not in reduce_axes
        ]

        if reduction_type is dtypes.ReductionType.Sum:
            # in this case, we need to simply scatter the grad across the axes that were reduced

            sdfg = SDFG("_reverse_" + str(reduction_type).replace(".", "_") +
                        "_")
            state = sdfg.add_state()

            rev_input_conn_name = self.get_grad_name(output_grad_connectors[0],
                                                     node)
            rev_output_conn_name = self.get_grad_name(input_grad_connectors[0],
                                                      node)

            _, rev_input_arr = sdfg.add_array(rev_input_conn_name,
                                              shape=output_array.shape,
                                              dtype=output_array.dtype)
            _, rev_output_arr = sdfg.add_array(rev_output_conn_name,
                                               shape=input_array.shape,
                                               dtype=input_array.dtype)

            state.add_mapped_tasklet(
                "_distribute_grad_" + str(reduction_type).replace(".", "_") +
                "_", {
                    "i" + str(i): "0:{}".format(shape)
                    for i, shape in enumerate(input_array.shape)
                }, {
                    "__in":
                    Memlet.simple(
                        rev_input_conn_name,
                        "0" if node.axes is None else ",".join(
                            "i" + str(i) for i in non_reduce_axes))
                },
                "__out = __in", {
                    "__out":
                    Memlet.simple(rev_output_conn_name, ",".join(
                        "i" + str(i) for i in all_axes))
                },
                external_edges=True)

            return self.backward_state.add_nested_sdfg(sdfg, None,
                                                       {"_reduce_in_grad"},
                                                       {"_reduce_out_grad"})
        else:
            raise AutoDiffException(
                "Unsupported reduction type '{}'".format(reduction_type))

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
                rev_output_grad_name = self.get_grad_name(inp, tasklet)
                rev_outputs.add(rev_output_grad_name)

                output_expr = output_exprs[output_conn]

                # symbolically differentiate the output by inp
                diff_expr = output_expr.diff(sp.symbols(inp))

                if diff_expr.atoms(sp.Derivative):
                    # the final result contains a call to sp.Derivative
                    raise AutoDiffException(
                        "Unable to symbolically differentiate expression: {}".
                        format(diff_expr.expr))

                rev_input_grad_name = self.get_grad_name(output_conn, tasklet)
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
