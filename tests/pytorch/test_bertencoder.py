import pytest
import numpy as np
import torch
from dace.transformation.dataflow import RedundantSecondArray
from transformers import BertConfig, BertLayer

import onnx
import onnxruntime as rt

from daceml.pytorch import DaceModule
from daceml.transformation import ConstantFolding


@pytest.mark.parametrize("apply_strict", [True, False])
def test_bert_encoder(gpu, apply_strict):
    batch_size = 8
    seq_len = 512
    hidden_size = 768

    input = torch.randn([batch_size, seq_len, hidden_size])

    ptmodel = BertLayer(BertConfig())
    pt_outputs = ptmodel(input.clone())

    dace_model = DaceModule(ptmodel, cuda=gpu)
    dace_outputs0 = dace_model(input.clone())

    onnx.save(dace_model.onnx_model, "encoder.onnx")
    sess = rt.InferenceSession("encoder.onnx")
    ort_outputs = sess.run(["133"],
                           input_feed={"input.1": input.detach().numpy()})

    dace_model.dace_model.sdfg.apply_transformations_repeated(
        [ConstantFolding, RedundantSecondArray], validate_all=True)

    dace_outputs1 = dace_model(input.clone())

    diff = np.abs(dace_outputs1 - pt_outputs[0].detach().numpy())
    print("Max diff to Pytorch:", np.max(diff))
    print("Median diff to Pytorch:", np.median(diff))

    diff = np.abs(dace_outputs1 - ort_outputs[0])
    print("Max diff to ORT:", np.max(diff))
    print("Median diff to ORT:", np.median(diff))

    assert np.allclose(dace_outputs1, ort_outputs[0], atol=1e-6)
    assert np.allclose(dace_outputs0, dace_outputs1)
