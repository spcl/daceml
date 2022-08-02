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


@properties.make_properties
class Array(data.Array):
    """
    A array for which a gradient can be computed.

    This also has to have the name 'Array' because otherwise none of the python frontend replacements work.
    """
    gradient = properties.DataProperty(
        desc="The corresponding gradient buffer")

    def __init__(self, gradient, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gradient = gradient

    @staticmethod
    def make_parameter(sdfg: SDFG, name: str):
        """
        Converts an existing array into a parameter.
        This also creates a gradient buffer for the parameter.

        :param sdfg: the SDFG containing the array.
        :param name: the name of the array.
        """
        desc = sdfg.arrays[name]

        # Create a gradient buffer for the array
        grad_desc = copy.deepcopy(sdfg.arrays[name])
        grad_desc.transient = True
        grad_name = sdfg.add_datadesc('gradient_' + name,
                                      grad_desc,
                                      find_new_name=True)
        new_desc = Array(grad_name,
                         desc.dtype,
                         desc.shape,
                         storage=desc.storage,
                         location=desc.location,
                         allow_conflicts=desc.allow_conflicts,
                         transient=desc.transient,
                         strides=desc.strides,
                         offset=desc.offset,
                         lifetime=desc.lifetime,
                         alignment=desc.alignment,
                         debuginfo=desc.debuginfo,
                         total_size=desc.total_size)
        sdfg.arrays[name] = new_desc


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
                if state.in_degree(edge.src) + state.out_degree(edge.src) == 0:
                    state.remove_node(edge.src)
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
    The BackwardPass library node expands to an implementation of a
    BackwardPass that computes the requested gradients.

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
        """

        ours = set(self.required_gradients)

        for state in sdfg.states():
            for node in state.nodes():
                if isinstance(node, BackwardPass):
                    if node is self:
                        continue
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

        # check that we are computing at least one gradient
        if len(self.out_connectors) == 0:
            raise ValueError(
                "BackwardPass node '{}' does not compute any gradients".format(
                    self.name))


@op_repository.replaces('torch.autograd.backward')
def backward(pv: newast.ProgramVisitor,
             sdfg: SDFG,
             state: SDFGState,
             tensors: TensorOrTensors,
             grads: Optional[TensorOrTensors] = None):
    """
    Adds a backward pass node to the SDFG.

    This function analyses the the dependency tree of the tensors and computes
    gradients for each Parameter (i.e.
    :class:``daceml.autograd.library.Array``) that was used to compute the
    tensors.
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

    # determine what grdaients to compute
    dependencies = autodiff_analysis.dependency_analysis(sdfg)

    to_compute = {
        dependency
        for tensor in tensors for dependency in dependencies[tensor]
        if isinstance(sdfg.arrays[dependency], Array)
    }

    for param in to_compute:
        grad_name = sdfg.arrays[param].gradient

        conn_name = find_str_not_in_set(bwd_node.out_connectors, grad_name)
        bwd_node.required_gradients[param] = conn_name
        bwd_node.add_out_connector(conn_name)

        state.add_edge(bwd_node, conn_name, state.add_write(grad_name), None,
                       sdfg.make_array_memlet(grad_name))


@op_repository.replaces_attribute('Array', 'grad')
def grad(pv: 'ProgramVisitor', sdfg: SDFG, state: SDFGState, arr: str) -> str:
    """
    Returns the name of the gradient buffer of the given array.

    The Array must have been marked as requires_grad_ using
    ``arr.requires_grad_()``, otherwise there will be an error
    """

    if arr not in sdfg.arrays:
        raise common.DaceSyntaxError(pv, None,
                                     "Array {} is not defined".format(arr))
    desc = sdfg.arrays[arr]
    if not isinstance(desc, Array):
        raise common.DaceSyntaxError(
            pv, None,
            "Called .grad on an Array that was not a Parameter. Convert it to a parameter "
            " first using .requires_grad_()")

    return desc.gradient


@op_repository.replaces_method('Array', 'requires_grad_')
@op_repository.replaces_method('Scalar', 'requires_grad_')
def requires_grad_(pv: newast.ProgramVisitor, sdfg: SDFG, state: SDFGState,
                   self: str):
    """
    Converts a array to a Parameter (i.e.
    :class:``daceml.autograd.library.Array``). This creates a descriptor for
    the gradient buffer for this array.
    """

    if self not in sdfg.arrays:
        raise common.DaceSyntaxError(pv, None,
                                     "Array {} is not defined".format(self))
    Array.make_parameter(sdfg, self)


@op_repository.replaces_method('Array', 'backward')
@op_repository.replaces_method('Scalar', 'backward')
def backward_method(pv: newast.ProgramVisitor,
                    sdfg: SDFG,
                    state: SDFGState,
                    self: str,
                    grad: Optional[str] = None):
    """
    Alias for ``torch.autograd.backward(self)``
    """
    backward(pv, sdfg, state, self, grad)
