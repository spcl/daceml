from daceml.onnx.shape_inference.symbolic_shape_infer import SymbolicShapeInference


class ShapeInferenceError(ValueError):
    pass


def infer_shapes(onnx_path, out_path):
    try:
        SymbolicShapeInference.infer_shapes(onnx_path, out_path)
    except SystemExit:
        raise ShapeInferenceError("Unable to infer shapes for ONNX model.")
