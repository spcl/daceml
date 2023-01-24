"""Op replacement registration."""
from typing import Dict

import dace
import torch
from onnx import helper
from onnx.onnx_pb import NodeProto

from daceml.onnx.converters import convert_attribute_proto
from daceml.onnx.nodes.replacement import register_replacement
from daceml.onnx.shape_inference.symbolic_shape_infer import SymbolicShapeInference


def inferGCNConv(placeholder_id_to_module: Dict[str, torch.nn.Module], ssi: SymbolicShapeInference, node: NodeProto):
    op_attributes = {
        attribute_proto.name: convert_attribute_proto(attribute_proto)
        for attribute_proto in node.attribute
    }
    _, module = placeholder_id_to_module[op_attributes['module_id']]
    weights_shape = module.lin.weight.T.shape
    output_dtype = ssi.known_vi_[
        node.input[0]].type.tensor_type.elem_type
    ssi._compute_matmul_shape(
        node, output_dtype=output_dtype, rhs_shape=weights_shape)


def gcnconv_shape(module):
    M = module.lin.weight.shape[0]

    def shape_from_inputs(*inputs):
        x = inputs[0]
        N = x.shape[0]
        return (N, M)

    return shape_from_inputs


register_replacement('torch_geometric.nn.conv.gcn_conv.GCNConv',
                     inputs={
                         'node_features': dace.float32,
                         'rowptrs': dace.int64,
                         'columns': dace.int64,
                         'edge_vals': dace.float32,
                     },
                     outputs={'output': dace.float32},
                     shape_infer=inferGCNConv,
                     shape_fn_from_module=gcnconv_shape)


def inferGATConv(placeholder_id_to_module: Dict[str, torch.nn.Module], ssi: SymbolicShapeInference,
                 node: NodeProto) -> None:
    op_attributes = {
        attribute_proto.name: convert_attribute_proto(attribute_proto)
        for attribute_proto in node.attribute
    }
    _, module = placeholder_id_to_module[op_attributes['module_id']]
    output_dtype = ssi.known_vi_[
        node.input[0]].type.tensor_type.elem_type

    out_shape = (ssi._get_shape(node, 0)[0], module.heads * module.out_channels)
    vi = ssi.known_vi_[node.output[0]]
    vi.CopyFrom(helper.make_tensor_value_info(
        node.output[0], output_dtype, out_shape))


def gatconv_shape(module):
    heads = module.heads
    out_features = module.out_channels

    def shape_from_inputs(*inputs):
        x = inputs[0]
        N = x.shape[0]
        return (N, out_features * heads)

    return shape_from_inputs


register_replacement('torch_geometric.nn.conv.gat_conv.GATConv',
                     inputs={
                         'node_features': dace.float32,
                         'rowptrs': dace.int64,
                         'columns': dace.int64
                     },
                     outputs={'output': dace.float32},
                     shape_infer=inferGATConv,
                     shape_fn_from_module=gatconv_shape)
