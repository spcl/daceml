import numpy as np
import pytest
import torch
from transformers import BertConfig, BertLayer

from daceml.pytorch import DaceModule


@pytest.mark.slow
@pytest.mark.pure
def test_bert_encoder_backward(gpu, sdfg_name):
    batch_size = 2
    seq_len = 512
    hidden_size = 768

    input = torch.randn([batch_size, seq_len, hidden_size])
    ptmodel = BertLayer(BertConfig(hidden_act="relu")).eval()

    dace_model = DaceModule(ptmodel,
                            cuda=gpu,
                            train=False,
                            backward=True,
                            sdfg_name=sdfg_name,
                            apply_strict=True)

    ptinput = torch.clone(input)
    ptinput.requires_grad = True
    ptmodel(ptinput)[0].sum().backward()

    dace_input = torch.clone(input)
    dace_input.requires_grad = True
    dace_model(dace_input).sum().backward()

    diff = np.abs(dace_input.grad.detach().numpy() -
                  ptinput.grad.detach().numpy())

    assert np.max(diff) < 1e-4
