import pytest
import numpy as np
import torch
from dace.transformation.dataflow import RedundantSecondArray
from transformers import BertConfig, BertLayer

import daceml.onnx as donnx
from daceml.pytorch import DaceModule
from daceml.transformation import ConstantFolding


@pytest.mark.slow
@pytest.mark.pure
def test_bert_encoder(gpu):
    batch_size = 8
    seq_len = 512
    hidden_size = 768

    input = torch.randn([batch_size, seq_len, hidden_size])

    ptmodel = BertLayer(BertConfig()).eval()
    pt_outputs = ptmodel(input.clone())

    dace_model = DaceModule(ptmodel, cuda=gpu, train=False)
    dace_outputs0 = dace_model(input.clone())

    diff = np.abs(dace_outputs0 - pt_outputs[0].detach().numpy())

    assert np.max(diff) < 1e-5
    print("testing passed")


@pytest.mark.parametrize("apply_strict", [True, False])
@pytest.mark.ort
def test_bert_cf(apply_strict):
    batch_size = 8
    seq_len = 512
    hidden_size = 768

    input = torch.randn([batch_size, seq_len, hidden_size])

    ptmodel = BertLayer(BertConfig()).eval()
    pt_outputs = ptmodel(input.clone())

    dace_model = DaceModule(ptmodel, train=False)
    dace_outputs0 = dace_model(input.clone())

    dace_model.dace_model.sdfg.apply_transformations_repeated(
        [ConstantFolding, RedundantSecondArray], validate_all=True)

    dace_outputs1 = dace_model(input.clone())

    diff = np.abs(dace_outputs0 - pt_outputs[0].detach().numpy())

    assert np.max(diff) < 1e-5
    assert np.allclose(dace_outputs1, dace_outputs0)
    print("testing passed")


if __name__ == "__main__":
    test_bert_encoder(True)
