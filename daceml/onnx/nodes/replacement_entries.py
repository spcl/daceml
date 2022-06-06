
# Op replacement registration.
from daceml.onnx.shape_inference.symbolic_shape_infer import SymbolicShapeInference
from daceml.onnx.converters import convert_attribute_proto
from daceml.onnx.nodes.replacement import register_replacement


def inferGCNConv(ssi: SymbolicShapeInference, node):
    op_attributes = {
        attribute_proto.name: convert_attribute_proto(attribute_proto)
        for attribute_proto in node.attribute
    }
    _, module = ssi.placeholder_id_to_module_[op_attributes['module_id']]
    weights_shape = module.lin.weight.shape
    output_dtype = ssi.known_vi_[
        node.input[0]].type.tensor_type.elem_type
    ssi._compute_matmul_shape(
        node, output_dtype=output_dtype, rhs_shape=weights_shape)


register_replacement('torch_geometric.nn.conv.gcn_conv.GCNConv',
                     inputs=['float32', 'int64'],
                     outputs=['float32'],
                     shape_infer=inferGCNConv)
