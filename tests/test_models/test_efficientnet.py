import os
import subprocess

import pytest

from daceml import onnx as donnx
import numpy as np
import onnx

from daceml.onnx import ONNXModel


@pytest.mark.ort
def test_efficientnet(sdfg_name):
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

    dace_model = ONNXModel(sdfg_name, model)
    test_input = np.random.rand(1, 3, 224, 224).astype(np.float32)
    dace_model(test_input)
