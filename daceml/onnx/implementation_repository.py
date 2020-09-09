import types
from collections import defaultdict
from typing import Type

import dace
from dace.transformation.pattern_matching import ExpandTransformation


class ONNXImplementations:
    _implementations = defaultdict(list)

    @staticmethod
    def get(name: str) -> list:
        """ Returns implementations of a ONNX op. """
        if name not in ONNXImplementations._implementations:
            return []
        return ONNXImplementations._implementations[name]

    @staticmethod
    def has_implementation(name: str) -> bool:
        return name in ONNXImplementations._implementations

    @staticmethod
    def register(name: str, expansion: Type[ExpandTransformation]):
        """ Register an expansion of a ONNX op. """
        ONNXImplementations._implementations[name].append(expansion)


def register_pure_expansion(op_type, can_be_applied=None):
    """
    Mark a method to return the pure expansion for an ONNX op

    :param op_type: the name of the op to expand
    :param can_be_applied: a lambda that takes a node and returns True iif the expansion can be applied
    """
    def decorator(function):
        class Expansion(ExpandTransformation):
            environments = []

        Expansion.expansion = function

        if can_be_applied is not None:
            Expansion.can_be_applied = can_be_applied

        Expansion = dace.library.expansion(Expansion)
        ONNXImplementations.register(op_type, Expansion)
        return function

    return decorator


# this will register the implementations
import daceml.onnx.op_implementations
