import torch
import numpy as np

from daceml.pytorch import DaceModule


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
    ptmodel = torch.nn.MultiheadAttention(N, H, bias=False)

    pt_outputs = ptmodel(Q, K, V)

    dace_model = DaceModule(ptmodel, cuda=gpu)
    dace_outputs = dace_model(Q, K, V)

    assert np.allclose(pt_outputs[0].detach().numpy(),
                       dace_outputs[0],
                       atol=1e-06)
    assert np.allclose(pt_outputs[1].detach().numpy(),
                       dace_outputs[1],
                       atol=1e-06)
    print("testing passed")


if __name__ == "__main__":
    test_attn(False)
