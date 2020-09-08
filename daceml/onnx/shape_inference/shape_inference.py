from daceml.onnx.shape_inference.symbolic_shape_infer import SymbolicShapeInference, get_opset


class ShapeInferenceError(ValueError):
    pass


def infer_shapes(onnx_model):
    int_max = 2**31 - 1
    auto_merge = False
    guess_output_rank = False
    verbose = 0

    onnx_opset = get_opset(onnx_model)
    if not onnx_opset or onnx_opset < 7:
        raise ShapeInferenceError(
            "Only support models of onnx opset 7 and above.")

    symbolic_shape_inference = SymbolicShapeInference(int_max, auto_merge,
                                                      guess_output_rank,
                                                      verbose)
    all_shapes_inferred = False
    symbolic_shape_inference._preprocess(onnx_model)
    while symbolic_shape_inference.run_:
        all_shapes_inferred = symbolic_shape_inference._infer_impl(onnx_model)
    symbolic_shape_inference._update_output_from_vi()
    if not all_shapes_inferred:
        raise ShapeInferenceError("Unable to infer shapes for ONNX model.")

    return symbolic_shape_inference.out_mp_
