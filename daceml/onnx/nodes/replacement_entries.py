"""Op replacement registration."""

import dace
from onnx import helper
from onnx.onnx_pb import NodeProto

from daceml.onnx.converters import convert_attribute_proto
from daceml.onnx.nodes.replacement import register_replacement
from daceml.onnx.shape_inference.symbolic_shape_infer import SymbolicShapeInference


def shape_infer_GCNConv(ssi: SymbolicShapeInference, node: NodeProto):
    op_attributes = {
        attribute_proto.name: convert_attribute_proto(attribute_proto)
        for attribute_proto in node.attribute
    }
    _, module = ssi.placeholder_id_to_module[op_attributes['module_id']]
    weights_shape = module.lin.weight.T.shape
    output_dtype = ssi.known_vi_[
        node.input[0]].type.tensor_type.elem_type
    ssi._compute_matmul_shape(
        node, output_dtype=output_dtype, rhs_shape=weights_shape)


def make_GCNConv_shape_fn(module):
    M = module.lin.weight.shape[0]

    def shape_from_inputs(*inputs):
        x = inputs[0]
        N = x.shape[0]
        return N, M

    return shape_from_inputs


register_replacement('torch_geometric.nn.conv.gcn_conv.GCNConv',
                     inputs={
                         'node_features': dace.float32,
                         'rowptrs': dace.int64,
                         'columns': dace.int64,
                         'edge_vals': dace.float32,
                     },
                     outputs={'output': dace.float32},
                     shape_infer=shape_infer_GCNConv,
                     shape_fn_from_module=make_GCNConv_shape_fn)


def shape_infer_GCNConv(ssi: SymbolicShapeInference, node: NodeProto) -> None:
    op_attributes = {
        attribute_proto.name: convert_attribute_proto(attribute_proto)
        for attribute_proto in node.attribute
    }
    _, module = ssi.placeholder_id_to_module[op_attributes['module_id']]
    output_dtype = ssi.known_vi_[
        node.input[0]].type.tensor_type.elem_type

    out_shape = (ssi._get_shape(node, 0)[0], module.heads * module.out_channels)
    vi = ssi.known_vi_[node.output[0]]
    vi.CopyFrom(helper.make_tensor_value_info(
        node.output[0], output_dtype, out_shape))


def make_GCNConv_shape_fn(module):
    heads = module.heads
    out_features = module.out_channels

    def shape_from_inputs(*inputs):
        x = inputs[0]
        N = x.shape[0]
        return N, out_features * heads

    return shape_from_inputs


register_replacement('torch_geometric.nn.conv.gat_conv.GATConv',
                     inputs={
                         'node_features': dace.float32,
                         'rowptrs': dace.int64,
                         'columns': dace.int64
                     },
                     outputs={'output': dace.float32},
                     shape_infer=shape_infer_GCNConv,
                     shape_fn_from_module=make_GCNConv_shape_fn)
