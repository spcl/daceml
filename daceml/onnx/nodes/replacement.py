import logging
from typing import Dict, List, Type, Optional

import dace
from dace import SDFG, nodes
from dace.transformation.transformation import ExpandTransformation
from daceml.onnx.forward_implementation_abc import ONNXForward
from daceml.onnx.nodes.node_codegen import expand_node
from daceml.onnx.nodes.onnx_op import (ONNXOp, _get_attr_docstring,
                                       _get_connector_docstring,
                                       _get_typecons_docstring)
from daceml.onnx.schema import ONNXParameterType, ONNXSchema

log = logging.getLogger(__name__)

# Generating an ONNX Library node.

_ALLOWED_TYPES = ['uint8',
                  'uint16',
                  'uint32',
                  'uint64',
                  'int8',
                  'int16',
                  'int32',
                  'int64',
                  'float16',
                  'float32',
                  'float64',
                  'bool_',
                  'complex64',
                  'complex128']

# Placeholder name is of format {qualname of replaced class}_{id of module}.
_placeholder_name_to_onnx_op: Dict[str, Type[nodes.Node]] = {}
_modules_to_replace: Dict[str, str] = {}


def module_name_to_placeholder_name(name: str, idx: Optional[int] = None):
    name = name.replace('.', 'DOT')
    if idx is not None:
        name += '_' + str(idx)
    return name


def get_replaced_placeholder_name(placeholder_name: str) -> str:
    return "_".join(placeholder_name.split('_')[:-1])


def is_replaceable(name: str) -> bool:
    return get_replaced_placeholder_name(name) in _placeholder_name_to_onnx_op


def create_replaced_onnx_op(op_name: str) -> nodes.Node:
    qualname = get_replaced_placeholder_name(op_name)
    if qualname not in _placeholder_name_to_onnx_op:
        raise ValueError(f'No replacement module for {qualname}.')
    onnx_op = _placeholder_name_to_onnx_op[qualname]
    return onnx_op(op_name)


def make_schema_dict(replaced_name, placeholder_name, inputs: List[str], outputs: List[str]):
    schema_dict = {
        'name': placeholder_name,
        'attributes': {},
        'doc': f'Placeholder for {replaced_name}',
        'domain': '',
        'since_version': 1,
        'type': 'ONNXSchema'
    }

    def make_type_info_helper(type_list: List[str], is_input):
        data_type_list = []
        type_constraints = {}
        name = 'input' if is_input else 'output'
        for i, t in enumerate(type_list):
            assert t in _ALLOWED_TYPES
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

    inputs_info, inputs_type_constrains = make_type_info_helper(
        inputs, is_input=True)
    outputs_info, outputs_type_constrains = make_type_info_helper(
        outputs, is_input=False)

    schema_dict.update({
        'inputs': inputs_info,
        'outputs': outputs_info,
        'type_constraints': {**inputs_type_constrains, **outputs_type_constrains},
    })
    return schema_dict


def generate_onnx_op_placeholder(schema):
    attrs = {}

    def __init__(self, name, *args, location=None, **op_attributes):
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

    # TODO: Check if the documentation makes any sense.
    attrs['__doc__'] = docstring + "\n"
    attrs['schema'] = schema

    attrs['__init__'] = __init__

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

def register_replacement(module_name: str, inputs: List[type] = None, outputs: List[type] = None):
    placeholder_name = module_name_to_placeholder_name(module_name)

    _modules_to_replace[module_name] = placeholder_name

    schema_dict = make_schema_dict(
        module_name, placeholder_name, inputs, outputs)
    schema = ONNXSchema.from_json(schema_dict)
    _placeholder_name_to_onnx_op[placeholder_name] = generate_onnx_op_placeholder(
        schema)


# Op implementation.
register_replacement('torch_geometric.nn.conv.gcn_conv.GCNConv', inputs=[
                     'float32', 'int64'], outputs=['float32'])
