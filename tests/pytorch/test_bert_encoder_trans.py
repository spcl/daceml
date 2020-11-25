import pytest
import numpy as np
import torch
from dace.transformation.dataflow import RedundantSecondArray
from transformers import BertConfig, BertLayer

import daceml.onnx as donnx
from daceml.pytorch import DaceModule
from daceml.transformation import ConstantFolding, ReshapeTransposeElimination
donnx.default_implementation = "pure"


@pytest.mark.ort
def test_bert_cf(gpu):
    batch_size = 8
    seq_len = 512
    hidden_size = 768

    input = torch.randn([batch_size, seq_len, hidden_size])

    ptmodel = BertLayer(BertConfig()).eval()
    pt_outputs = ptmodel(input.clone())

    dace_model = DaceModule(ptmodel, cuda=gpu, train=False)
    dace_outputs0 = dace_model(input.clone())
    dace_model.dace_model.sdfg.save("before_trans.sdfg")

    dace_model.dace_model.sdfg.apply_transformations_repeated(
        [ConstantFolding, RedundantSecondArray], validate_all=True)
    dace_model.dace_model.sdfg.apply_transformations_repeated(
        [ReshapeTransposeElimination], validate_all=True)
    dace_model.dace_model.sdfg.save("after_trans_before_expand.sdfg")
    dace_model.dace_model.sdfg.expand_library_nodes()
    dace_model.dace_model.sdfg.apply_strict_transformations()
    dace_model.dace_model.sdfg.save("after_expand.sdfg")

    dace_outputs1 = dace_model(input.clone())

    diff = np.abs(dace_outputs0 - pt_outputs[0].detach().numpy())

    assert np.max(diff) < 1e-5
    assert np.allclose(dace_outputs1, dace_outputs0, rtol=1e-05)
    print("Transformations (including CF and RTE) testing passed")


if __name__ == "__main__":
    test_bert_cf(False)
    #test_bert_cf(True)
