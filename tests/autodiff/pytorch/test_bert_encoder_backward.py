import pytest
import numpy as np
import torch
from dace.transformation.dataflow import RedundantSecondArray
from transformers import BertConfig, BertLayer

from daceml.pytorch import DaceModule
from daceml.transformation import ConstantFolding


def test_bert_encoder_backward():
    batch_size = 8
    seq_len = 512
    hidden_size = 768

    input = torch.randn([batch_size, seq_len, hidden_size])
    ptmodel = BertLayer(BertConfig(hidden_act="relu")).eval()

    dace_model = DaceModule(ptmodel, cuda=False, train=False, backward=True)

    ptinput = torch.clone(input)
    ptinput.requires_grad = True
    ptmodel(ptinput)[0].sum().backward()

    dace_input = torch.clone(input)
    dace_input.requires_grad = True
    dace_model(dace_input).sum().backward()

    assert np.allclose(dace_input.grad, ptinput.grad)


if __name__ == "__main__":
    test_bert_encoder_backward()
