# TODO
# 1. test for 3 layers of nesting
# 2. test for forwarding intermediate value
import numpy as np
import torch

import dace
from dace.transformation.interstate import StateFusion
from test_single_state import SDFGBackwardRunner, run_correctness


@dace.program
def inner_sdfg(Z: dace.float32[3, 3], W: dace.float32[3, 3]):
    W[:] = dace.elementwise(lambda x: log(x), Z)


@dace.program
def middle_sqrt(Y: dace.float32[3, 3]):
    intermediate = dace.define_local([3, 3], dace.float32)
    W = dace.define_local([3, 3], dace.float32)
    intermediate[:] = dace.elementwise(lambda x: sqrt(x), Y)
    inner_sdfg(intermediate, W)
    Z = np.sum(W)
    return Z


@run_correctness
def test_nested():
    sdfg = middle_sqrt.to_sdfg(strict=False)

    sdfg.apply_transformations_repeated([StateFusion])

    def torch_func(*, Y):
        inter = torch.sqrt(Y)
        W = torch.log(inter)
        Z = torch.sum(W)
        Z.backward()
        return dict(Y_gradient=Y.grad)

    return (SDFGBackwardRunner(sdfg, "__return", strict=False), torch_func,
            dict(Y=np.random.rand(3, 3).astype(np.float32)))
