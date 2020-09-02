"""
Regression tests for BERT subgraphs
"""
import numpy as np

import onnx

from daceml.onnx import ONNXModel


def test_slice(gpu):
    model = onnx.load("tests/onnx_files/slice.onnx")
    dace_model = ONNXModel("slice", model, cuda=gpu)

    out = dace_model(data=np.ones((2, ), dtype=np.float32))
    assert out.shape == (1, )
    assert out[0] == 1.0


def test_reshape(gpu):
    model = onnx.load("tests/onnx_files/reshape.onnx")
    dace_model = ONNXModel("reshape", model, cuda=gpu)
    dace_model()
