import pytest
import numpy as np
import torch
from dace.libraries import blas
from dace.transformation.dataflow import RedundantSecondArray
from dace.library import change_default
from transformers import BertConfig, BertLayer

import daceml.onnx as donnx
from daceml.pytorch import DaceModule
from daceml.transformation import ConstantFolding


def test_bert_encoder(gpu, default_implementation, sdfg_name):
    if not gpu and default_implementation == 'onnxruntime':
        pytest.skip("combination is tested below")

    if gpu:
        blas_default = "cuBLAS"
    else:
        blas_default = "MKL"

    with change_default(blas, blas_default):
        batch_size = 8
        seq_len = 512
        hidden_size = 768

        input = torch.randn([batch_size, seq_len, hidden_size])

        ptmodel = BertLayer(BertConfig()).eval()
        pt_outputs = ptmodel(input.clone())

        dace_model = DaceModule(ptmodel,
                                cuda=gpu,
                                train=False,
                                sdfg_name=sdfg_name)
        dace_outputs0 = dace_model(input.clone())

        diff = np.abs(dace_outputs0 - pt_outputs[0].detach().numpy())

        assert np.max(diff) < 1e-5


@pytest.mark.ort
def test_bert_cf(sdfg_name):
    batch_size = 8
    seq_len = 512
    hidden_size = 768

    input = torch.randn([batch_size, seq_len, hidden_size])

    ptmodel = BertLayer(BertConfig()).eval()
    pt_outputs = ptmodel(input.clone())

    dace_model = DaceModule(ptmodel, train=False, sdfg_name=sdfg_name)
    dace_outputs0 = dace_model(input.clone())

    dace_model.dace_model.sdfg.apply_transformations_repeated(
        [ConstantFolding, RedundantSecondArray], validate_all=True)
    dace_model.dace_model.sdfg.expand_library_nodes()
    dace_model.dace_model.sdfg.apply_strict_transformations()

    dace_outputs1 = dace_model(input.clone())

    diff = np.abs(dace_outputs0 - pt_outputs[0].detach().numpy())

    assert np.max(diff) < 1e-5
    assert np.allclose(dace_outputs1, dace_outputs0)


if __name__ == '__main__':
    donnx.default_implementation = "pure"
    test_bert_encoder(True, "pure", "testing")
