import numpy as np
import pytest
import torch
from transformers import BertConfig, BertLayer

from daceml.torch import DaceModule
from daceml.testing import copy_to_gpu, torch_tensors_close


@pytest.mark.pure
@pytest.mark.cpublas
def test_bert_encoder_backward(gpu, sdfg_name):
    batch_size = 2
    seq_len = 512
    hidden_size = 768

    input = copy_to_gpu(gpu, torch.randn([batch_size, seq_len, hidden_size]))
    ptmodel = copy_to_gpu(gpu, BertLayer(BertConfig(hidden_act="relu")).eval())

    dace_model = DaceModule(ptmodel,
                            training=False,
                            backward=True,
                            sdfg_name=sdfg_name,
                            simplify=True)

    ptinput = torch.clone(input)
    ptinput.requires_grad = True
    ptmodel(ptinput)[0].sum().backward()

    dace_input = torch.clone(input)
    dace_input.requires_grad = True
    dace_model(dace_input).sum().backward()

    torch_tensors_close("input_grad", ptinput.grad, dace_input.grad)
