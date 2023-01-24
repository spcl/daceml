import logging
from copy import deepcopy
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Tuple, Type, Mapping

import dace
import torch
from dace import SDFG, nodes
from dace.properties import Property
from dace.transformation.transformation import ExpandTransformation
from onnx.onnx_pb import NodeProto

from daceml.onnx.converters import clean_onnx_name, TORCH_DTYPE_TO_TYPECLASS, typeclass_to_onnx_str, \
    TYPECLASS_TO_TORCH_DTYPE
from daceml.onnx.forward_implementation_abc import ONNXForward
from daceml.onnx.nodes.node_codegen import expand_node
from daceml.onnx.nodes.onnx_op import (ONNXOp, _get_attr_docstring,
                                       _get_connector_docstring,
                                       _get_typecons_docstring)
from daceml.onnx.schema import ONNXParameter, ONNXParameterType, ONNXSchema, ONNXTypeConstraint
from daceml.onnx.shape_inference.symbolic_shape_infer import SymbolicShapeInference

log = logging.getLogger(__name__)

ShapeFnType = Callable[..., Tuple[int, ...]]


@dataclass
class ReplacementInfo:
    module_name: str
    onnx_op: Type[nodes.Node]
    infer_shape: Callable[[Dict[str, torch.nn.Module], SymbolicShapeInference, NodeProto], None]
    shape_fn_from_module: Callable[[torch.nn.Module], ShapeFnType]
    output_dtype: torch.dtype  # todo


MODULES_TO_REPLACE: Dict[str, ReplacementInfo] = {}
_module_name_to_infer_shape = {}  # todo annotate type


def is_replaceable(name: str) -> bool:
    return name in MODULES_TO_REPLACE


def get_replaced_onnx_op(name: str) -> Type[nodes.Node]:
    if name not in MODULES_TO_REPLACE:
        raise ValueError(f'No replacement module for {name}.')
    return MODULES_TO_REPLACE[name].onnx_op


def make_schema_dict(name, inputs: Mapping[str, dace.typeclass], outputs: Mapping[str, dace.typeclass]):
    intersection = [name for name in inputs if name in outputs]
    assert len(intersection) == 0, f"Same keys for inputs and outputs not allowed: {intersection}"

    schema_dict = {
        'name': name,
        'attributes': {},
        'doc': f'Placeholder for {name}',
        'domain': '',
        'since_version': 1,
        'type': 'ONNXSchema'
    }

    def make_type_info_helper(type_mapping: Mapping[str, dace.typeclass]):
        data_type_list = []
        type_constraints = {}
        for i, (name, t) in enumerate(type_mapping.items()):
            # For some reason dace.float32 gets converted to string as 'float',
            # not 'float32' which is not understood by ONNX.
            if t is dace.float32:
                t = 'float32'
            else:
                t = typeclass_to_onnx_str(t)
            type_name = f'{name}_T'
            entry = {
                'description': '',
                'homogeneous': True,
                'name': f'{name}',
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

    inputs_info, inputs_type_constraints = make_type_info_helper(inputs)
    outputs_info, outputs_type_constraints = make_type_info_helper(outputs)

    schema_dict.update({
        'inputs': inputs_info,
        'outputs': outputs_info,
        'type_constraints': {**inputs_type_constraints, **outputs_type_constraints},
    })
    return schema_dict


def onnx_type_info_from_torch_params(params: Iterable[Tuple[str, torch.nn.Parameter]]):
    onnx_params = []
    onnx_type_constraints = {}
    for name, p in params:
        name = clean_onnx_name(name)
        type_name = name + '_T'
        onnx_params.append(ONNXParameter.from_json({
            'description': '',
            'homogeneous': True,
            'name': name,
            'param_type': 'Single',
            'type': 'ONNXParameter',
            'type_str': type_name
        }))
        onnx_type_constraints[type_name] = ONNXTypeConstraint.from_json({
            'type': 'ONNXTypeConstraint',
            'type_str': type_name,
            'types': [TORCH_DTYPE_TO_TYPECLASS[p.dtype].to_string()],
        })
    return onnx_params, onnx_type_constraints


# Generating an ONNX Library node.
def generate_onnx_op_placeholder(schema):
    attrs = {}

    def __init__(self, name, module, prefix, *args, location=None, **op_attributes):
        # Add information about module parameters to the schema.
        onnx_params, onnx_type_constraints = onnx_type_info_from_torch_params(
            module.named_parameters())
        self.schema = deepcopy(self.schema)
        self.schema.inputs += onnx_params
        self.schema.type_constraints.update(onnx_type_constraints)
        # TODO: Get input/output spec from module?

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
        self.module = module
        self.prefix = prefix

        if len(args) > 0:
            raise TypeError(
                f"__init__() takes 2 positional arguments but {2 + len(args)} were given")

        if len(op_attributes) > 0:
            raise TypeError(
                f"__init__() takes no keyword arguments but following were given: {op_attributes}")

    # TODO: the docstrings for params are missing, but are they needed?
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
    attrs['module'] = Property(
        dtype=torch.nn.Module, desc='Replaced module', allow_none=False)
    attrs['prefix'] = Property(
        dtype=str, desc='Prefix for the module.', allow_none=False)

    cls = type(cls_name, (ONNXOp,), attrs)
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
                        log.warning(
                            'No expansion for library node "{}". '
                            'Reason: forward_can_be_applied returned False'.
                                format(node.label))
                        result = expand_node(node, state, sdfg)
                        if not isinstance(result, SDFG):
                            # When we return an SDFG the environments will be determined recursively by codegen.
                            cls.environments = map(
                                dace.library.get_environment,
                                result.environments)
                        return result

            implementation_name = args["name"]
            cls.register_implementation(implementation_name, Expansion)

    globals()[cls_name] = cls

    return cls


# Registration of replacement.
def register_replacement(module_name: str,
                         inputs: Mapping[str, dace.typeclass],
                         outputs: Mapping[str, dace.typeclass],
                         shape_infer: Callable[[Dict[str, torch.nn.Module], SymbolicShapeInference, 'NodeProto'], None],
                         shape_fn_from_module: Callable[[torch.nn.Module], ShapeFnType]):
    if len(outputs) > 1:
        raise NotImplementedError("Replacing nodes with more than 1 output is not supported.")

    output_dtype = next(iter(outputs.values()))

    _module_name_to_infer_shape[module_name] = shape_infer
    schema_dict = make_schema_dict(module_name, inputs, outputs)
    schema = ONNXSchema.from_json(schema_dict)
    onnx_op = generate_onnx_op_placeholder(
        schema)
    replacement_info = ReplacementInfo(module_name=module_name,
                                       infer_shape=shape_infer,
                                       shape_fn_from_module=shape_fn_from_module,
                                       onnx_op=onnx_op,
                                       output_dtype=TYPECLASS_TO_TORCH_DTYPE[output_dtype])
    MODULES_TO_REPLACE[module_name] = replacement_info
