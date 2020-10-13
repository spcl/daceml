import numpy as np
import torch

import dace
from dace.transformation.interstate import StateFusion
from test_single_state import SDFGBackwardRunner, run_correctness


@dace.program
def inner_sdfg(Z: dace.float32[3, 3], W: dace.float32[3, 3]):
    W[:] = dace.elementwise(lambda x: log(x), Z)


@dace.program
def inner_sdfg_with_intermediate(Z: dace.float32[3, 3], W: dace.float32[3, 3]):
    intermediate = dace.define_local([3, 3], dace.float32)
    intermediate[:] = dace.elementwise(lambda x: sqrt(x), Z)
    W[:] = dace.elementwise(lambda x: log(x), intermediate)


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


@dace.program
def middle_sqrt_with_intermediate(Y: dace.float32[3, 3]):
    intermediate = dace.define_local([3, 3], dace.float32)
    W = dace.define_local([3, 3], dace.float32)
    intermediate[:] = dace.elementwise(lambda x: sqrt(x), Y)
    inner_sdfg_with_intermediate(intermediate, W)
    Z = np.sum(W)
    return Z


@run_correctness
def test_nested_forwarding():
    sdfg = middle_sqrt_with_intermediate.to_sdfg(strict=False)

    sdfg.apply_transformations_repeated([StateFusion])

    def torch_func(*, Y):
        inter = torch.sqrt(Y)
        inter2 = torch.sqrt(inter)
        W = torch.log(inter2)
        Z = torch.sum(W)
        Z.backward()
        return dict(Y_gradient=Y.grad)

    return (SDFGBackwardRunner(sdfg, "__return", strict=False), torch_func,
            dict(Y=np.random.rand(3, 3).astype(np.float32)))


@dace.program
def middle_sqrt_no_sum(Y: dace.float32[3, 3]):
    intermediate = dace.define_local([3, 3], dace.float32)
    W = dace.define_local([3, 3], dace.float32)
    intermediate[:] = dace.elementwise(lambda x: sqrt(x), Y)
    inner_sdfg_with_intermediate(intermediate, W)
    return W


@dace.program
def outer_sqrt_with_intermediate(Y: dace.float32[3, 3]):
    intermediate = dace.define_local([3, 3], dace.float32)
    W = dace.define_local([3, 3], dace.float32)
    intermediate[:] = dace.elementwise(lambda x: sqrt(x), Y)
    W[:] = middle_sqrt_no_sum(intermediate)
    Z = np.sum(W)
    return Z


@run_correctness
def test_triple_nested_forwarding():
    sdfg = outer_sqrt_with_intermediate.to_sdfg(strict=False)

    sdfg.apply_transformations_repeated([StateFusion])

    def torch_func(*, Y):
        inter = torch.sqrt(Y)
        inter2 = torch.sqrt(inter)
        inter3 = torch.sqrt(inter2)
        W = torch.log(inter3)
        Z = torch.sum(W)
        Z.backward()
        return dict(Y_gradient=Y.grad)

    return (SDFGBackwardRunner(sdfg, "__return", strict=False), torch_func,
            dict(Y=np.random.rand(3, 3).astype(np.float32)))
