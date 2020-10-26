"""
Regression tests for BERT subgraphs
"""
import os
import numpy as np

import onnx

from daceml.onnx import ONNXModel

data_directory = os.path.join(os.path.dirname(__file__), "onnx_files")


def test_slice(gpu):
    model = onnx.load(os.path.join(data_directory, "slice.onnx"))
    dace_model = ONNXModel("slice", model, cuda=gpu)

    out = dace_model(data=np.ones((2, ), dtype=np.float32))
    assert out.shape == (1, )
    assert out[0] == 1.0


def test_reshape(gpu):
    model = onnx.load(os.path.join(data_directory, "reshape.onnx"))
    dace_model = ONNXModel("reshape", model, cuda=gpu)
    dace_model()
