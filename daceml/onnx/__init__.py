from dace.library import register_library
from .environments import ONNXRuntime
from .nodes import *
from .schema import onnx_representation, ONNXAttributeType, ONNXAttribute, ONNXTypeConstraint, ONNXParameterType, ONNXSchema, ONNXParameter
from .check_impl import check_op
from .onnx_importer import ONNXModel

register_library(__name__, "onnx")
