"""
Regression tests for BERT subgraphs
"""
import os
import numpy as np

import pytest
import onnx
import torch

from daceml.onnx import ONNXModel

data_directory = os.path.join(os.path.dirname(__file__), "onnx_files")


@pytest.mark.ort
def test_slice(gpu, sdfg_name):
    model = onnx.load(os.path.join(data_directory, "slice.onnx"))
    dace_model = ONNXModel(sdfg_name, model, cuda=gpu, fold_constants=False)

    out = dace_model(data=np.ones((2, ), dtype=np.float32))
    assert out.shape == (1, )
    assert out[0] == 1.0


def test_reshape(gpu, default_implementation, sdfg_name):
    model = onnx.load(os.path.join(data_directory, "reshape.onnx"))
    dace_model = ONNXModel(sdfg_name, model, cuda=gpu, fold_constants=False)
    dace_model()


@pytest.mark.ort
def test_save_transients(gpu, sdfg_name):
    model = onnx.load(os.path.join(data_directory, "reshape.onnx"))
    transients = {}
    dace_model = ONNXModel(sdfg_name,
                           model,
                           save_transients=transients,
                           cuda=gpu)
    dace_model()
    assert torch.allclose(
        transients["ONNX_bertSLASHembeddingsSLASHReshape_4__42COLON0"].type(
            torch.int32).cpu(),
        dace_model.weights["bert/embeddings/Reshape_4/shape:0"])
