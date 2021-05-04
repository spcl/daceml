import os
import subprocess

import numpy as np
import onnx
import torch
from dace import dtypes

import daceml.onnx as donnx
from daceml.testing import copy_to_gpu, torch_tensors_close


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

    dace_model = donnx.ONNXModel(
        sdfg_name,
        model,
        cuda=gpu,
        storage=dtypes.StorageType.Default,
        infer_shapes=False,
        # constant folding is too slow on this model
        fold_constants=False)
    feed = {
        "input_ids:0": np.load(os.path.join(data_directory, "input_ids.npy")),
        "input_mask:0": np.load(os.path.join(data_directory,
                                             "input_mask.npy")),
        "segment_ids:0":
        np.load(os.path.join(data_directory, "segment_ids.npy")),
    }
    #feed = {k: copy_to_gpu(gpu, torch.from_numpy(v)) for k, v in feed.items()}
    feed["ONNX_OneHot216_o0__d0"] = 2
    # todo ONNX_OneHot can be removed once shape infer is bumped
    outputs = dace_model(**feed)
    unstack_0 = np.load(os.path.join(data_directory, "unstack_0.npy"))
    unstack_1 = np.load(os.path.join(data_directory, "unstack_1.npy"))

    torch_tensors_close("outputs1", torch.from_numpy(unstack_0),
                        torch.from_numpy(outputs[1]))
    torch_tensors_close("outputs0", torch.from_numpy(unstack_1),
                        torch.from_numpy(outputs[0]))
