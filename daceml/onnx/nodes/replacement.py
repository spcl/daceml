from dataclasses import dataclass
import logging
from typing import Any, Callable, Dict, List, Type

import dace
from dace import SDFG, nodes
from dace.properties import Property
from dace.transformation.transformation import ExpandTransformation
from dace.dtypes import TYPECLASS_STRINGS
from daceml.onnx.converters import clean_onnx_name
from daceml.onnx.forward_implementation_abc import ONNXForward
from daceml.onnx.nodes.node_codegen import expand_node
from daceml.onnx.nodes.onnx_op import (ONNXOp, _get_attr_docstring,
                                       _get_connector_docstring,
                                       _get_typecons_docstring)
from daceml.onnx.schema import ONNXParameterType, ONNXSchema
from daceml.onnx.shape_inference.symbolic_shape_infer import SymbolicShapeInference

log = logging.getLogger(__name__)


@dataclass
class ParamInfo:
    dtype: str
    name: str
    required: bool

    def __init__(self, dtype, name, required) -> None:
        self.dtype = dtype
        self.name = name
        self.required = required

    def to_onnx_type_info(self):
        type_name = f'{self.name}_T'
        type_info = {
            'description': '',
            'homogeneous': True,
            'name': clean_onnx_name(self.name),
            'param_type': 'Single' if self.required else 'Optional',
            'type': 'ONNXParameter',
            'type_str': type_name
        }
        type_constraint = (
            type_name, {
                'type': 'ONNXTypeConstraint',
                'type_str': type_name,
                'types': [self.dtype]
            }
        )

        return type_info, type_constraint


_modules_to_replace: Dict[str, str] = {}
_module_name_to_onnx_op: Dict[str, Type[nodes.Node]] = {}
_module_name_to_param_infos: Dict[str, List[ParamInfo]] = {}
_module_name_to_infer_shape = {}  # todo annotate type


def is_replaceable(name: str) -> bool:
    return name in _module_name_to_onnx_op


def get_replaced_onnx_op(name: str) -> nodes.Node:
    if name not in _module_name_to_onnx_op:
        raise ValueError(f'No replacement module for {name}.')
    onnx_op = _module_name_to_onnx_op[name]
    return onnx_op


def make_schema_dict(name, inputs: List[str], params: List[ParamInfo], outputs: List[str]):
    schema_dict = {
        'name': name,
        'attributes': {},
        'doc': f'Placeholder for {name}',
        'domain': '',
        'since_version': 1,
        'type': 'ONNXSchema'
    }

    def make_type_info_helper(type_list: List[str], is_input):
        data_type_list = []
        type_constraints = {}
        name = 'input' if is_input else 'output'
        for i, t in enumerate(type_list):
            assert t in TYPECLASS_STRINGS, f"{t} is not a valid ONNX type. Valid ONNX types: {TYPECLASS_STRINGS}"
            type_name = f'{name}_{i}_T'
            entry = {
                'description': '',
                'homogeneous': True,
                'name': f'{name}_{i}',
                'param_type': 'Single',
                'type': 'ONNXParameter',
                'type_str': type_name
            }
            data_type_list.append(entry)

            type_constraints[type_name] = {
                'type': 'ONNXTypeConstraint',
                'type_str': type_name,
                'types': [t]
            }
        return data_type_list, type_constraints

    inputs_info, inputs_type_constraints = make_type_info_helper(
        inputs, is_input=True)
    outputs_info, outputs_type_constraints = make_type_info_helper(
        outputs, is_input=False)

    # Sort params by name to ensure the order is the same everywhere.
    params = sorted(params, key=lambda p: (
        not p.required, clean_onnx_name(p.name)))
    params_all_info = [p.to_onnx_type_info() for p in params]
    params_info = [p for p, _ in params_all_info]

    params_type_constraints = {k: v for _, (k, v) in params_all_info}

    schema_dict.update({
        'inputs': inputs_info + params_info,
        'outputs': outputs_info,
        'type_constraints': {**inputs_type_constraints, **outputs_type_constraints, **params_type_constraints},
    })
    return schema_dict


# Generating an ONNX Library node.
def generate_onnx_op_placeholder(schema):
    attrs = {}

    def __init__(self, name, module_id, *args, location=None, **op_attributes):
        super(ONNXOp, self).__init__(
            name,
            location=location,
            # add required parameters as in/out connectors, without types for now
            inputs={
                inp.name
                for inp in self.schema.inputs
                if inp.param_type == ONNXParameterType.Single
            },
            outputs={
                out.name
                for out in self.schema.outputs
                if out.param_type == ONNXParameterType.Single
            })
        self.backward_implementation = None
        self.module_id = module_id

        if len(args) > 0:
            raise TypeError(
                "__init__() takes 2 positional arguments but {} were given".
                format(2 + len(args)))

        if len(op_attributes) > 0:
            raise TypeError(
                f"__init__() takes no keyword arguments but following were given: {op_attributes}")

    input_connector_docstrings = "\n".join(
        _get_connector_docstring(param) for param in schema.inputs)
    output_connector_docstrings = "\n".join(
        _get_connector_docstring(param) for param in schema.outputs)

    cls_name = schema.name

    # the first line of the init docstring contains the signature of the method. This will be picked up by sphinx and
    # means that the generated sphinx docs have a proper signature, and not just *args, **kwargs.
    init_docstring = "__init__(name, *, {})\n".format(
        ", ".join(attr.name if attr.required else attr.name + "=" +
                  repr(attr.default_value)
                  for _, attr in schema.attributes.items()))
    init_docstring += ":param name: the name of the node.\n" + "\n".join(
        _get_attr_docstring(attr)
        for _, attr in schema.attributes.items())

    docstring = "\n" + schema.doc
    type_docstrings = "\n".join(
        _get_typecons_docstring(cons)
        for _, cons in schema.type_constraints.items())
    docstring += "\n\n"
    docstring += ":Node Inputs:" + input_connector_docstrings
    docstring += "\n\n"
    docstring += ":Node Outputs:" + output_connector_docstrings
    docstring += "\n\n"
    docstring += ":Type Constraints:" + type_docstrings

    # TODO: Check if the documentation makes any sense. Maybe copy from GCNConv or maybe not
    attrs['__doc__'] = docstring + "\n"
    attrs['schema'] = schema
    attrs['__init__'] = __init__
    attrs['module_id'] = Property(
        dtype=int, desc='id of replaced module', allow_none=False)

    cls = type(cls_name, (ONNXOp, ), attrs)
    cls = dace.library.node(cls)
    cls.__init__.__doc__ = "\n" + init_docstring

    for impl, args in ONNXForward.extensions().items():
        if "op" in args and args["op"] == schema.name:

            class Expansion(ExpandTransformation):
                environments = []
                forward_impl: ONNXForward = impl

                @classmethod
                def expansion(cls, node, state, sdfg, **kwargs):
                    # validate
                    node.validate(sdfg, state)

                    if cls.forward_impl.forward_can_be_applied(
                            node, state, sdfg):
                        result = cls.forward_impl.forward(
                            node, state, sdfg, **kwargs)
                        if hasattr(cls.forward_impl, "environments"):
                            cls.environments.extend(
                                cls.forward_impl.environments)
                        return result
                    else:
                        log.warn(
                            'No expansion for library node "{}". '
                            'Reason: forward_can_be_applied returned False'.
                            format(node.label))
                        result = expand_node(node, state, sdfg)
                        if not isinstance(result, SDFG):
                            # when we return an SDFG the the environments will be determined recursively by codegen.
                            cls.environments = map(
                                dace.library.get_environment,
                                result.environments)
                        return result

            implementation_name = args["name"]
            cls.register_implementation(implementation_name, Expansion)

    globals()[cls_name] = cls

    return cls


# Registration of replacement.
def register_replacement(module_name: str, inputs: List[str], params: List[ParamInfo], outputs: List[str], shape_infer: Callable[[SymbolicShapeInference, Any], None]):
    _modules_to_replace[module_name] = module_name
    params = sorted(params, key=lambda p: (
        not p.required, clean_onnx_name(p.name)))
    _module_name_to_param_infos[module_name] = params
    _module_name_to_infer_shape[module_name] = shape_infer
    schema_dict = make_schema_dict(module_name, inputs, params, outputs)
    schema = ONNXSchema.from_json(schema_dict)
    _module_name_to_onnx_op[module_name] = generate_onnx_op_placeholder(
        schema)
