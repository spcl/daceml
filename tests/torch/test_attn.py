import torch
import numpy as np
import pytest

from daceml.torch import DaceModule

from dace.transformation.dataflow import RedundantSecondArray

from daceml.testing import copy_to_gpu, torch_tensors_close
from daceml.transformation import ConstantFolding


@pytest.mark.ort
def test_attn(gpu, sdfg_name, use_cpp_dispatcher):
    B = 2
    H = 16
    P = 64
    N = P * H
    SM, SN = 512, 512
    K, Q, V = [
        copy_to_gpu(gpu, torch.randn([SM, B, N])),
        copy_to_gpu(gpu, torch.randn([SN, B, N])),
        copy_to_gpu(gpu, torch.randn([SM, B, N]))
    ]
    ptmodel = copy_to_gpu(gpu, torch.nn.MultiheadAttention(N, H, bias=False))

    pt_outputs = ptmodel(Q, K, V)

    dace_model = DaceModule(ptmodel,
                            sdfg_name=sdfg_name,
                            compile_torch_extension=use_cpp_dispatcher)

    dace_outputs = dace_model(Q, K, V)

    torch_tensors_close("outputs_0", pt_outputs[0], dace_outputs[0])
    torch_tensors_close("outputs_1", pt_outputs[1], dace_outputs[1])
