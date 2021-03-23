import pytest
import numpy as np
import torch
from dace.transformation.dataflow import RedundantSecondArray
from transformers import BertConfig, BertLayer

import daceml.onnx as donnx
from daceml.pytorch import DaceModule
from daceml.transformation import ConstantFolding, parameter_to_transient


def test_bert_encoder(gpu, default_implementation, sdfg_name):
    if not gpu and default_implementation == 'onnxruntime':
        pytest.skip("combination is tested below")

    batch_size = 8
    seq_len = 512
    hidden_size = 768

    input = torch.randn([batch_size, seq_len, hidden_size])

    ptmodel = BertLayer(BertConfig()).eval()
    pt_outputs = ptmodel(input.clone())

    dace_model = DaceModule(ptmodel,
                            cuda=gpu,
                            train=False,
                            sdfg_name=sdfg_name,
                            apply_strict=True,
                            dummy_inputs=(input.clone(), ))

    if gpu:
        for name, _ in dace_model.model.named_parameters():
            parameter_to_transient(dace_model, name)

    dace_outputs0 = dace_model(input.clone())

    diff = np.abs(dace_outputs0.detach().numpy() -
                  pt_outputs[0].detach().numpy())

    assert np.max(diff) < 1e-5

    if default_implementation == "pure":
        ort_nodes = [
            n for n, _ in dace_model.sdfg.all_nodes_recursive()
            if hasattr(n, "environments") and any("onnx" in e.lower()
                                                  for e in n.environments)
        ]
        if len(ort_nodes) > 0:
            assert False, f"expected pure graph, found ORT nodes: {ort_nodes} "

        # check that cuBLAS is being used
        if gpu:
            assert any(
                (hasattr(n, "environments") and "cuBLAS" in n.environments or
                 hasattr(n, "implementation") and n.implementation == "cuBLAS")
                for n, _ in dace_model.sdfg.all_nodes_recursive())


@pytest.mark.ort
def test_bert_cf(sdfg_name):
    batch_size = 8
    seq_len = 512
    hidden_size = 768

    input = torch.randn([batch_size, seq_len, hidden_size])

    ptmodel = BertLayer(BertConfig()).eval()
    pt_outputs = ptmodel(input.clone())

    dace_model = DaceModule(ptmodel,
                            train=False,
                            sdfg_name=sdfg_name,
                            dummy_inputs=(input.clone(), ),
                            auto_optimize=False)

    dace_model.dace_model.sdfg.apply_transformations_repeated(
        [ConstantFolding, RedundantSecondArray],
        validate_all=True,
        strict=True)

    dace_outputs1 = dace_model(input.clone())

    diff = np.abs(dace_outputs1.detach().numpy() -
                  pt_outputs[0].detach().numpy())

    assert np.max(diff) < 1e-5
