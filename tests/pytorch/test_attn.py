import torch
import numpy as np

from daceml.pytorch import DACEModule


def test_attn(gpu):
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
    if gpu:
        K, Q, V = K.cuda(), Q.cuda(), V.cuda()

    ptmodel = torch.nn.MultiheadAttention(N, H, bias=False)
    pt_outputs = ptmodel(Q, K, V)

    dace_model = DACEModule(ptmodel)
    dace_outputs = dace_model(Q, K, V)

    assert np.allclose(pt_outputs[0].detach().numpy(),
                       dace_outputs[0],
                       atol=1e-07)
    assert np.allclose(pt_outputs[1].detach().numpy(),
                       dace_outputs[1],
                       atol=1e-07)
