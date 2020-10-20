import abc
from collections import namedtuple
import typing

from dace import SDFG, SDFGState

from dace.registry import make_registry
from dace.sdfg.nodes import Node

from daceml.onnx.nodes.onnx_op import ONNXOp


class BackwardContext(typing.NamedTuple):
    """ A tuple holding the graph context required to construct reverse nodes """
    forward_sdfg: SDFG  #: the forward SDFG
    forward_state: SDFGState  #: the forward SDFG state
    backward_sdfg: SDFG  #: the backward SDFG
    backward_state: SDFGState  #: the backward SDFG state


class BackwardResult(typing.NamedTuple):
    """ A tuple holding the graph context required to construct reverse nodes """
    node: Node  #: the created and added backward node
    grad_names: typing.List[typing.Optional[str]]  #: the names of gradients


@make_registry
class BackwardImplementation(abc.ABC):
    """ ABC for ONNX op forward implementations.

        This registry accepts two types of registrations.
        The register function expects an argument `node_type=TYPE` where `TYPE` is the type of node that this backward
        implementation supports.
    """
    @staticmethod
    @abc.abstractmethod
    def backward_can_be_applied(node: Node, state: SDFGState,
                                sdfg: SDFG) -> bool:
        """ Return whether this expansion can be applied.

            :param node: the candidate node.
            :param state: the candidate state.
            :param sdfg: the candidate sdfg.
        """
        ...

    @staticmethod
    @abc.abstractmethod
    def backward(
        forward_node: Node, forward_state: SDFGState, forward_sdfg: SDFG,
        backward_state: SDFGState, backward_sdfg: SDFG,
        given_gradients: typing.Dict[typing.Optional[str],
                                     typing.Optional[str]],
        required_gradients: typing.Dict[typing.Optional[str],
                                        typing.Optional[str]]
    ) -> typing.Union[Node, SDFG]:
        """ Add the reverse node for a node from the forward pass to the backward pass, and return it.

            For each input connector with name `n` of the forward in required_grads, the returned backward node must add
            an output connector with name `required_grads[n]` that will output the gradient for that input.

            If any input from the forward pass is required, simply add a connector with the same name as the connector
            on the forward node. The input will later be connected as required.


            :param forward_node: the candidate node (on the forward pass).
            :param forward_state: the candidate state (on the forward pass).
            :param forward_sdfg: the candidate forward sdfg.
            :param backward_state: the state (on the backward pass) that the backward node should be added to.
            :param backward_sdfg: the backward sdfg.
            :param given_gradients: mapping from forward node output connector name to the name of the gradient for that
                                    connector. These gradients will be connected as inputs to the returned backward
                                    node.
            :param required_gradients: The gradients that the reversed node is required to generate, in the form of a
                                       mapping from input connector name on the forward node to the gradient name for
                                       that input.
            :return: the reverse node.
        """
        ...


# register the implementations
import daceml.autodiff.implementations

# this file contains forward and backward implementations for ONNX op nodes.
import daceml.onnx.op_implementations.pure_implementations
