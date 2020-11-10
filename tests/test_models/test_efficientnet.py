import os
import subprocess

import numpy as np
import onnx

from daceml.onnx import ONNXModel


def test_efficientnet(gpu, default_implementation):
    data_directory = os.path.join(os.path.dirname(__file__), "data")

    path = os.path.join(data_directory, "efficientnet.onnx")
    # Download model
    if not os.path.exists(path):
        subprocess.check_call([
            "wget",
            "https://github.com/onnx/models/raw/master/vision/classification/efficientnet-lite4/model/efficientnet-lite4-11.onnx",
            "--output-document={}".format(path)
        ])

    model = onnx.load(path)

    dace_model = ONNXModel("efficientnet", model, cuda=gpu)
    test_input = np.random.rand(1, 3, 224, 224).astype(np.float32)
    dace_model(test_input)
