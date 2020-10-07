import typing

import dace
from dace import SDFGState, SDFG
from dace.sdfg.nodes import Node

from daceml.autodiff.implementation_abc import ONNXBackward
from daceml.onnx import ONNXOp


class GemmBackward(ONNXBackward):
    @staticmethod
    def backward_can_be_applied(node: ONNXOp, state: SDFGState,
                                sdfg: SDFG) -> bool:
        pass

    @staticmethod
    def backward(node: ONNXOp, state: SDFGState,
                 sdfg: SDFG) -> typing.Union[Node, SDFG]:
        pass
