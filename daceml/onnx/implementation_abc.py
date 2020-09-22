import abc
import typing

from dace import SDFG, SDFGState
from dace.sdfg.nodes import Node
from dace.registry import make_registry

from daceml.onnx.nodes.onnx_op import ONNXOp


@make_registry
class ONNXForward(abc.ABC):
    """ ABC for ONNX op forward implementations.

        The register function expects an argument `op` containing the ONNX op name (string).
    """
    @staticmethod
    @abc.abstractmethod
    def forward_can_be_applied(node: ONNXOp, state: SDFGState,
                               sdfg: SDFG) -> bool:
        # TODO docstring
        ...

    @staticmethod
    @abc.abstractmethod
    def forward(node: ONNXOp, state: SDFGState,
                sdfg: SDFG) -> typing.Union[Node, SDFG]:

        # TODO docstring
        ...


# register expansions
import daceml.onnx.op_implementations.pure_implementations
