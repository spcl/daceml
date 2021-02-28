from daceml.onnx.shape_inference.symbolic_shape_infer import SymbolicShapeInference


def infer_shapes(onnx_model):
    return SymbolicShapeInference.infer_shapes(onnx_model)
