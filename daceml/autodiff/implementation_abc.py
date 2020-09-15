import abc

from dace.registry import make_registry
from dace.sdfg.nodes import Node
from websockets import typing

from daceml.onnx import ONNXOp, SDFGState, SDFG


@make_registry
class ONNXBackward(abc.ABC):
    """ ABC for ONNX op forward implementations.

        The register function expects an argument `op` containing the ONNX op name (string).
    """

    @staticmethod
    @abc.abstractmethod
    def backward_can_be_applied(node: ONNXOp, state: SDFGState, sdfg: SDFG) -> bool:

        # TODO docstring
        ...

    @staticmethod
    @abc.abstractmethod
    def backward(node: ONNXOp, state: SDFGState, sdfg: SDFG) -> typing.Union[Node, SDFG]:

        # TODO docstring
        ...
