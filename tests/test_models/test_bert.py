import os
import subprocess

import numpy as np
import onnx

import daceml.onnx as donnx


def test_bert_full(gpu, default_implementation, sdfg_name):
    data_directory = os.path.join(os.path.dirname(__file__), "data")

    bert_path = os.path.join(data_directory, "bert_infer.onnx")
    # Download onnx model
    if not os.path.exists(bert_path):
        subprocess.check_call([
            "wget", "http://spclstorage.inf.ethz.ch/~rauscho/bert_infer.onnx",
            "--output-document={}".format(bert_path), "--no-verbose"
        ])

    model = onnx.load(bert_path)

    dace_model = donnx.ONNXModel(sdfg_name,
                                 model,
                                 cuda=gpu,
                                 infer_shapes=False)
    feed = {
        "input_ids:0": np.load(os.path.join(data_directory, "input_ids.npy")),
        "input_mask:0": np.load(os.path.join(data_directory,
                                             "input_mask.npy")),
        "segment_ids:0":
        np.load(os.path.join(data_directory, "segment_ids.npy")),
        "ONNX_OneHot216_o0__d0": 2
    }
    # todo ONNX_OneHot can be removed once shape infer is bumped
    outputs = dace_model(**feed)
    unstack_0 = np.load(os.path.join(data_directory, "unstack_0.npy"))
    unstack_1 = np.load(os.path.join(data_directory, "unstack_1.npy"))

    assert np.all(np.abs(outputs[1] - unstack_0) < 1e-4)
    assert np.all(np.abs(outputs[0] - unstack_1) < 1e-4)
