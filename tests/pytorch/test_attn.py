import torch
import numpy as np
import pytest

from daceml.pytorch import DaceModule

from dace.transformation.dataflow import RedundantSecondArray
from daceml.transformation import ConstantFolding


@pytest.mark.ort
def test_attn():
    B = 2
    H = 16
    P = 64
    N = P * H
    SM, SN = 512, 512
    K, Q, V = [
        torch.randn([SM, B, N]),
        torch.randn([SN, B, N]),
        torch.randn([SM, B, N])
    ]
    ptmodel = torch.nn.MultiheadAttention(N, H, bias=False)

    pt_outputs = ptmodel(Q, K, V)

    dace_model = DaceModule(ptmodel)
    dace_outputs_0 = dace_model(Q, K, V)

    dace_model.dace_model.sdfg.apply_transformations_repeated(
        [ConstantFolding, RedundantSecondArray],
        validate_all=True,
        strict=True)
    dace_outputs_1 = dace_model(Q, K, V)

    assert np.allclose(pt_outputs[0].detach().numpy(),
                       dace_outputs_1[0],
                       atol=1e-06)
    assert np.allclose(pt_outputs[1].detach().numpy(),
                       dace_outputs_1[1],
                       atol=1e-06)


