import os
import subprocess

import pytest

from daceml import onnx as donnx
import numpy as np
import onnx

from daceml.onnx import ONNXModel


def test_efficientnet(gpu, default_implementation, sdfg_name):
    if gpu:
        pytest.skip("GPU EfficientNet is currently broken due to Gemv")
    data_directory = os.path.join(os.path.dirname(__file__), "data")

    path = os.path.join(data_directory, "efficientnet.onnx")
    # Download model
    if not os.path.exists(path):
        subprocess.check_call([
            "wget",
            "http://spclstorage.inf.ethz.ch/~rauscho/efficientnet-lite4-11.onnx",
            "--output-document={}".format(path), "--no-verbose"
        ])

    model = onnx.load(path)

    dace_model = ONNXModel(sdfg_name, model, cuda=gpu)
    test_input = np.random.rand(1, 3, 224, 224).astype(np.float32)
    dace_model(test_input)
