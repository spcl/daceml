import numpy as np
import torch

import dace


def test_torch_scalar_argument():
    @dace.program
    def prog(a: dace.float32, B: dace.float32[2, 2]):
        return B + a

    B = np.arange(4).reshape(2, 2).astype(np.float32)
    a = torch.tensor(1.0)
    a.requires_grad = True
    result = prog(a, B)
    np.testing.assert_allclose(result, B + a.numpy())
