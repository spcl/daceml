from daceml.onnx.shape_inference.symbolic_shape_infer import SymbolicShapeInference, get_opset


def infer_shapes(onnx_model, placeholder_id_to_module, auto_merge=False):
    result = SymbolicShapeInference.infer_shapes(onnx_model, placeholder_id_to_module,
                                                 auto_merge=auto_merge)
    if result is None:
        raise ValueError("Symbolic shape inference failed")
    return result
