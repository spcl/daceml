"""
Regression tests for BERT subgraphs
"""
import os
import numpy as np

import pytest
import onnx

from daceml.onnx import ONNXModel

data_directory = os.path.join(os.path.dirname(__file__), "onnx_files")


@pytest.mark.ort
def test_slice(gpu, sdfg_name):
    model = onnx.load(os.path.join(data_directory, "slice.onnx"))
    dace_model = ONNXModel(sdfg_name, model, cuda=gpu)

    out = dace_model(data=np.ones((2, ), dtype=np.float32))
    assert out.shape == (1, )
    assert out[0] == 1.0


def test_reshape(gpu, default_implementation, sdfg_name):
    model = onnx.load(os.path.join(data_directory, "reshape.onnx"))
    dace_model = ONNXModel(sdfg_name, model, cuda=gpu)
    dace_model()
