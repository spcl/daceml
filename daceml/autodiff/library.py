"""
Dace library for autodiff

Includes the BackwardPass library node, and the replacements for the python frontend
"""
from typing import Union, Sequence, Dict, Set, Optional, Tuple
import itertools
import copy
import collections

import torch
import torch.autograd

import dace
import dace.library
from dace import SDFG, SDFGState, nodes, data, properties
from dace.transformation import transformation as pm, pass_pipeline
from dace.transformation.passes import analysis, dead_dataflow_elimination
from dace.sdfg import graph

from dace.frontend.python import common
from dace.frontend.common import op_repository
from dace.frontend.python import newast

from daceml.autodiff import backward_pass_generator as engine, analysis as autodiff_analysis
from daceml.util.utils import find_str_not_in_set, in_edge_with_name, all_equal

TensorOrTensors = Union[str, Sequence[str]]


@dace.library.expansion
class ExpandBackwardPass(pm.ExpandTransformation):
    environments = []

    @staticmethod
    def expansion(node: 'BackwardPass', state: SDFGState, sdfg: SDFG):
        node.validate(sdfg, state)

        nsdfg = SDFG("backward")
        nstate = nsdfg.add_state()

        in_array_name = lambda connector_name: in_edge_with_name(
            node, state, connector_name).data.data

        array_grad_map = {}

        access_sets = analysis.AccessSets().apply_pass(sdfg, {})

        forward_state = node.determine_forward_state(sdfg,
                                                     state,
                                                     access_sets=access_sets)

        # Check for other BackwardPasses that also compute the same gradients as us
        node.propagate_conflicts(sdfg, state)

        # Remove own control dependencies
        node.clean_output_connectors(sdfg, state)

        # get the names of the output arrays in the forward pass
        given_gradients = node.outer_names_given_gradients(state)

        array_grad_map.update(node.required_gradients)
        array_grad_map.update((in_array_name(value_conn_name), grad_conn_name)
                              for grad_conn_name, value_conn_name in
                              node.given_gradients.items())

        # remove the non-grad arrays as inputs from the forward pass;
        # they were also just added for control dependencies
        for forward_non_grad_conn_name in node.given_gradients.values():
            for edge in list(
                    state.in_edges_by_connector(node,
                                                forward_non_grad_conn_name)):
                state.remove_edge(edge)
            node.remove_in_connector(forward_non_grad_conn_name)

        gen = engine.BackwardPassGenerator(
            sdfg=sdfg,
            state=forward_state,
            given_gradients=given_gradients,
            required_gradients=node.required_gradients.keys(),
            backward_sdfg=nsdfg,
            backward_state=nstate,
            zero_non_transients=False,
            array_grad_map=array_grad_map,
            conflicted_gradient_buffers=node._conflicted_gradients)

        _, _, required_forwarded_values = gen.backward()

        # Add zero initialization for all gradients which we are the first to compute
        for outer_edge in state.out_edges(node):
            gradient_we_are_writing: str = outer_edge.data.data
            is_written_with_wcr = any(
                edge.data.wcr is not None
                and edge.data.data == outer_edge.src_conn
                for edge, _ in nsdfg.all_edges_recursive()
                if isinstance(edge, graph.MultiConnectorEdge))

            anyone_written_before_us = autodiff_analysis.is_previously_written(
                sdfg,
                state,
                node,
                gradient_we_are_writing,
                access_sets=access_sets)
            if not anyone_written_before_us and is_written_with_wcr:
                engine.init_grad(gradient_we_are_writing, sdfg, state)

        for name in required_forwarded_values:
            # get the access to the forwarded_value
            # there should only be one since we don't allow inplace modification
            n = [
                n for n in state.nodes()
                if isinstance(n, nodes.AccessNode) and n.data == name
            ]
            if len(n) > 1:
                raise ValueError(
                    "Expected only one access node for forwarded value, does the graph have in-place modification?"
                )
            elif len(n) == 0:
                n = state.add_read(name)
            else:
                n = n[0]

            node.add_in_connector(name)
            state.add_edge(n, None, node, name, sdfg.make_array_memlet(name))

        nsdfg.validate()

        return nsdfg


@dace.library.node
class BackwardPass(nodes.LibraryNode):
    """
    The BackwardPass library node expands to an implementation of a BackwardPass that computes the requested gradients.

    These gradients are computed using the DaCeML autograd engine.

    The gradient will be computed for each array in the output connectors.
    For this, the names of the output connectors must match the name of the
    array for which the gradient is to be computed.
    """

    # Global properties
    implementations = {
        "differentiate": ExpandBackwardPass,
    }
    default_implementation = "differentiate"

    given_gradients = properties.DictProperty(
        key_type=str,
        value_type=str,
        desc=
        "Mapping between connector names of the given gradients and the names of the arrays they correspond to."
    )
    required_gradients = properties.DictProperty(
        key_type=str,
        value_type=str,
        desc=
        "Mapping from array name for which a gradient should be computed to the name of the connector that will receive the gradient."
    )

    _conflicted_gradients = properties.SetProperty(
        element_type=str,
        desc=
        "Keys from required_gradients for which the gradients are also computed elsewhere, and thus writes to the "
        " buffer need to be with write-conflict-resolution. Note: this field is automatically populated upon expansion."
    )

    def __init__(self, name, given_gradients: Dict[str, str], *args, **kwargs):
        super().__init__(name, *args, **kwargs)
        self.given_gradients = given_gradients
        self.required_gradients = {}

    def outer_names_given_gradients(self, state: SDFGState) -> Set[str]:
        """
        Returns the names of the arrays that are passed as given gradients.
        """
        in_array_name = lambda connector_name: in_edge_with_name(
            self, state, connector_name).data.data
        return set(map(in_array_name, self.given_gradients.values()))

    def propagate_conflicts(self, sdfg: SDFG, state: SDFGState):
        """
        Across this SDFG, check for other BackwardPasses that also compute the same gradients as us.

        If there are multiple BackwardPasses that compute the same gradients, update their list of conflicts.
        :note: this removes the control dependencies on all backward pass nodes in the SDFG by calling 
               :meth:`clean_output_connectors`.
        """

        self.clean_output_connectors(sdfg, state)
        ours = set(self.required_gradients)

        for state in sdfg.states():
            for node in state.nodes():
                if isinstance(node, BackwardPass):
                    if node is self:
                        continue
                    node.clean_output_connectors(sdfg, state)
                    conflicts = ours.intersection(node.required_gradients)
                    if conflicts:
                        self._conflicted_gradients |= conflicts
                        node._conflicted_gradients |= conflicts

    def determine_forward_state(
        self,
        sdfg: SDFG,
        state: SDFGState,
        access_sets: Optional[autodiff_analysis.AccessSets] = None
    ) -> SDFGState:
        """
        Determine what the forward pass state for this backward node is.

        This is the state containing the whole subgraph that will be differentiated.
        This can be different from the state that the node is in, for example
        when there are multiple Backward Passes parsed in the python frontend.

        :param sdfg: The SDFG containing the node.
        :param state: The state containing the node.
        :param access_sets: optionally precomupted access_sets
        :return: The state containing the forward pass that will be differentiated.
        """
        # We need to find the state that where all given_gradients are written.

        given_gradients = self.outer_names_given_gradients(state)

        if access_sets is None:
            access_sets = analysis.AccessSets().apply_pass(sdfg, {})

        candidate_states = []
        for cand in sdfg.states():
            _, write_set = access_sets[cand]
            if given_gradients.issubset(write_set):
                candidate_states.append(cand)

        if len(candidate_states) != 1:
            raise ValueError(
                "Could not find a state where all outputs are written. The "
                "DaCeML autodiff currently only supports differentiating single dataflow states."
            )
        return candidate_states[0]

    def clean_output_connectors(self, sdfg: SDFG, state: SDFGState):
        """
        When parsed in the python frontend, every .grad call is connected to
        every BackwardPass initially to introduce control dependencies.

        This method removes these control dependencies when the gradient is not actually computed by this node.
        """
        given_gradients = self.outer_names_given_gradients(state)
        # check whether the input subtree actually includes the node that we
        # should compute a gradient for.
        dependencies = autodiff_analysis.dependency_analysis(sdfg)
        for grad in list(self.required_gradients.keys()):
            if not any(grad in dependencies[g] for g in given_gradients):
                self.remove_out_connector(self.required_gradients[grad])
                for edge in state.out_edges_by_connector(
                        self, self.required_gradients[grad]):
                    state.remove_edge(edge)
                self.required_gradients.pop(grad)
        # remove isolated subgraphs that we no longer need
        pass_pipeline.Pipeline([
            dead_dataflow_elimination.DeadDataflowElimination()
        ]).apply_pass(sdfg, {})

        # remove dangling nodes, this can happen with non-transients
        for node, parent in sdfg.all_nodes_recursive():
            if (isinstance(node, nodes.AccessNode)
                    and parent.in_degree(node) + parent.out_degree(node) == 0):
                parent.remove_node(node)

    def validate(self, sdfg, state):
        # check that there is a correspondence between given gradients and inputs
        all_inputs = set(self.in_connectors)
        for given_grad, tensor_name in self.given_gradients.items():
            if given_grad not in all_inputs:
                raise ValueError(
                    "Given gradient '{}' is not an input of the node".format(
                        given_grad))

            all_inputs.remove(given_grad)
            all_inputs.remove(tensor_name)

        if all_inputs:
            raise ValueError(
                "The following in connectors were not included in given_gradients: {}"
                .format(', '.join(all_inputs)))
        # check that the forward state can be determined
        self.determine_forward_state(sdfg, state)


@op_repository.replaces('torch.autograd.backward')
def backward(pv: newast.ProgramVisitor,
             sdfg: SDFG,
             state: SDFGState,
             tensors: TensorOrTensors,
             grads: Optional[TensorOrTensors] = None):
    """
    Adds a backward pass node to the SDFG.

    This backward pass initially doesn't compute any gradients, since the outputs of the node are initially empty.

    While parsing the program, the outputs will be populated, for example when .grad of an input is accessed.
    """

    if isinstance(tensors, str):
        tensors = [tensors]

    if isinstance(grads, str):
        grads = [grads]

    if grads is None:
        grads = []
        # when the tensors are scalars, we can implicity create the grads with ones
        for tensor in tensors:
            tensor_desc = sdfg.arrays[tensor]
            if tensor_desc.total_size == 1:
                constant_name = sdfg._find_new_name("one")
                desc = data.Scalar(tensor_desc.dtype,
                                   transient=True,
                                   storage=tensor_desc.storage)
                sdfg.add_constant(constant_name, 1, dtype=desc)
                sdfg.arrays[constant_name] = desc
                grads.append(constant_name)
            else:
                raise common.DaceSyntaxError(
                    pv, None,
                    "grad can be implicitly created only for scalar outputs")

    if len(grads) != len(tensors):
        raise common.DaceSyntaxError(
            pv, None,
            "grads and tensors must correspond, but they were not the same length"
        )

    for grad, tensor in zip(grads, tensors):
        if grad not in sdfg.arrays and grad not in sdfg.constants_prop:
            raise common.DaceSyntaxError(
                pv, None, "Gradient {} is not an array".format(grad))
        if tensor not in sdfg.arrays:
            raise common.DaceSyntaxError(
                pv, None, "Tensor {} is not an array".format(tensor))

        grad_desc = sdfg.arrays[
            grad] if grad in sdfg.arrays else sdfg.constants_prop[grad][0]

        if not all_equal(grad_desc.shape, sdfg.arrays[tensor].shape):
            raise common.DaceSyntaxError(
                pv, None,
                "Gradient {} and tensor {} have different shapes".format(
                    grad, tensor))

    given_gradients = dict(zip(grads, tensors))

    bwd_node = BackwardPass('backward',
                            inputs=set(itertools.chain(tensors, grads)),
                            outputs=set(),
                            given_gradients=given_gradients)
    state.add_node(bwd_node)

    for inp in itertools.chain(tensors, grads):
        state.add_edge(state.add_read(inp), None, bwd_node, inp,
                       sdfg.make_array_memlet(inp))


@op_repository.replaces_attribute('Array', 'grad')
@op_repository.replaces_attribute('Scalar', 'grad')
def grad(pv: 'ProgramVisitor', sdfg: SDFG, state: SDFGState, arr: str) -> str:
    """
    Calling .grad on an array does two things:

    1. Allocate a gradient buffer for the array.
    2. Add dependencies to all BackwardPass nodes for the gradient buffer.

    Initially, each gradient buffer depends on all backward pass nodes, since
    we cannot determine which ones compute which gradients until after dataflow
    coarsening.
    """

    if arr not in sdfg.arrays:
        raise common.DaceSyntaxError(pv, None,
                                     "Array {} is not defined".format(arr))

    # Create a gradient buffer for the array
    grad_desc = copy.deepcopy(sdfg.arrays[arr])
    grad_desc.transient = True
    grad_name = sdfg.add_datadesc('gradient_' + arr,
                                  grad_desc,
                                  find_new_name=True)

    for node, parent in sdfg.all_nodes_recursive():
        if isinstance(node, BackwardPass):
            parent: SDFGState
            conn_name = find_str_not_in_set(node.out_connectors, grad_name)
            node.required_gradients[arr] = conn_name
            node.add_out_connector(conn_name)

            parent.add_edge(node, conn_name, parent.add_write(grad_name), None,
                            sdfg.make_array_memlet(grad_name))

    return grad_name


@op_repository.replaces_method('Array', 'backward')
@op_repository.replaces_method('Scalar', 'backward')
def backward_method(pv: newast.ProgramVisitor,
                    sdfg: SDFG,
                    state: SDFGState,
                    self: str,
                    grad: Optional[str] = None):
    backward(pv, sdfg, state, self, grad)
