"""
Dace library for autodiff

Includes the BackwardPass library node, and the replacements for the python frontend
"""
from typing import Union, Sequence, Dict, List, Optional
import itertools
import copy

import torch
import torch.autograd

import dace
import dace.library
from dace import SDFG, SDFGState, nodes, data, properties
from dace.transformation import transformation as pm

from dace.frontend.python import common
from dace.frontend.common import op_repository
from dace.frontend.python import newast

from daceml.autodiff import backward_pass_generator as engine
from daceml.util.utils import find_str_not_in_set, in_edge_with_name, all_equal


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

        given_gradients = list(
            map(in_array_name, node.given_gradients.values()))
        required_gradients = list(node.required_gradients.keys())

        array_grad_map.update(node.required_gradients)
        array_grad_map.update((in_array_name(value_conn_name), grad_conn_name)
                              for grad_conn_name, value_conn_name in
                              node.given_gradients.items())

        # remove the non-grad arrays as inputs from the forward pass;
        # they were just added to imply data dependencies
        for forward_non_grad_conn_name in node.given_gradients.values():
            for edge in list(
                    state.in_edges_by_connector(node,
                                                forward_non_grad_conn_name)):
                state.remove_edge(edge)
            node.remove_in_connector(forward_non_grad_conn_name)

        gen = engine.BackwardPassGenerator(
            sdfg=sdfg,
            state=state,
            given_gradients=given_gradients,
            required_gradients=required_gradients,
            backward_sdfg=nsdfg,
            backward_state=nstate,
            zero_non_transients=False,
            array_grad_map=array_grad_map)

        backward_result, desc_to_grad, required_forwarded_values = gen.backward(
        )

        for name in required_forwarded_values:
            # get the access to the forwarded_value
            # there should only be one since we don't allow inplace modification
            n = [
                n for n in state.nodes()
                if isinstance(n, nodes.AccessNode) and n.data == name
            ]
            assert len(
                n
            ) == 1, "Expected only one access node for forwarded value, does the graph have in-place modification?"

            node.add_in_connector(name)
            state.add_edge(n[0], None, node, name,
                           sdfg.make_array_memlet(name))

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

    def __init__(self, name, given_gradients: Dict[str, str], *args, **kwargs):
        super().__init__(name, *args, **kwargs)
        self.given_gradients = given_gradients
        self.required_gradients = {}

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


TensorOrTensors = Union[str, Sequence[str]]


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
@op_repository.replaces_attribute('View', 'grad')
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

            parent.add_edge(node, conn_name, state.add_write(grad_name), None,
                            sdfg.make_array_memlet(grad_name))

    return grad_name
