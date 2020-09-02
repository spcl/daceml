import onnx

from daceml.onnx import ONNXModel


def test_reshape():
    model = onnx.load("onnx_files/reshape.onnx")
    dace_model = ONNXModel("reshape", model, cuda=True)
    dace_model()
