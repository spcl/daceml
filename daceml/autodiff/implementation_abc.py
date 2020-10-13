import abc
import typing

from dace.registry import make_registry
from dace.sdfg.nodes import Node

from daceml.onnx import ONNXOp, SDFGState, SDFG


@make_registry
class ONNXBackward(abc.ABC):
    """ABC for ONNX op forward implementations.

    The register function expects an argument `op` containing the ONNX op name (string).
    """
    @staticmethod
    @abc.abstractmethod
    def backward_can_be_applied(node: ONNXOp, state: SDFGState,
                                sdfg: SDFG) -> bool:
        """ Return whether this expansion can be applied.

            :param node: the candidate node.
            :param state: the candidate state.
            :param sdfg: the candidate sdfg.
        """
        ...

    @staticmethod
    @abc.abstractmethod
    def backward(node: ONNXOp, state: SDFGState,
                 sdfg: SDFG) -> typing.Union[Node, SDFG]:
        """ Expand `node` and return its expansion.

            :param node: the candidate node.
            :param state: the candidate state.
            :param sdfg: the candidate sdfg.
            :return: the expanded node.
        """
        ...


# register the implementations
import daceml.autodiff.implementations
