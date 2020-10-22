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
    """ The return type of reversing a node. It the names of the gradients the node calculates and requires. """

    #: mapping from names of output connectors to the connector name of the gradient for that connector.
    required_grad_names: typing.Dict[typing.Optional[str], typing.Optional[str]]

    #: mapping from names of input connectors to the connector name of the gradient for that connector.
    given_grad_names: typing.Dict[typing.Optional[str], typing.Optional[str]]

    @staticmethod
    def empty():
        return BackwardResult(given_grad_names={}, required_grad_names={})


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
        forward_node: Node,
        context: BackwardContext,
        given_gradients: typing.List[typing.Optional[str]],
        required_gradients: typing.List[typing.Optional[str]]
    ) -> typing.Tuple[Node, BackwardResult]:
        """ Add the reverse node for a node from the forward pass to the backward pass, and return it.

            For each input connector with name ``n`` of the forward in required_grads, the returned backward node must add
            an output connector with name ``required_grads[n]`` that will output the gradient for that input.

            If any input from the forward pass is required, simply add a connector with the same name as the connector
            on the forward node. The input will later be connected as required.

            :param forward_node: the node for which the backward pass should be generated for.
            :param context: the context for this node (see
                            :class:`~daceml.autodiff.backward_implementation.BackwardContext`).
            :param given_gradients: The names of outputs of the node that gradients will be connected for.
            :param required_gradients: The names of connectors that gradients should be generated for.
            :return: the reverse node and gradient names
                     (see :class:`~daceml.autodiff.backward_implementation.BackwardResult`).
        """
        ...


# register the implementations
import daceml.autodiff.implementations

# this file contains forward and backward implementations for ONNX op nodes.
import daceml.onnx.op_implementations.pure_implementations
