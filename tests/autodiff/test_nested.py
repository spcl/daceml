import numpy as np
import torch

import dace
from dace import nodes as nd
from dace.transformation.interstate import StateFusion

import daceml.onnx as donnx
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


@run_correctness
def test_view_forwarding():
    # Prepare the inner sdfg
    old_default = donnx.default_implementation
    donnx.default_implementation = "pure"

    @dace.program
    def add_reshape_grad_test_nested(inp: dace.float64[9],
                                     bias: dace.float64[3],
                                     target_shape: dace.int64[2],
                                     result: dace.float64):
        reshaped = dace.define_local([3, 3], dace.float64)
        added = inp + 1
        donnx.ONNXReshape(data=added, shape=target_shape, reshaped=reshaped)
        Z = reshaped * bias
        Zl = dace.elementwise(lambda x: log(x + 1), Z)
        result[:] = np.sum(Zl)

    sdfg = add_reshape_grad_test_nested.to_sdfg(strict=False)

    sdfg.expand_library_nodes()
    sdfg.apply_strict_transformations()

    donnx.default_implementation = old_default

    # Prepare the outer SDFG

    @dace.program
    def inner_view_forwarding(inp: dace.float64[9], bias: dace.float64[3],
                              target_shape: dace.int64[2]):
        result = dace.define_local_scalar(dace.float64)
        sdfg(inp=inp, bias=bias, target_shape=target_shape, result=result)
        return result + 1

    outer_sdfg = inner_view_forwarding.to_sdfg(strict=False)
    outer_sdfg.apply_transformations_repeated([StateFusion], strict=True)

    def torch_func(*, inp, bias):
        reshaped = torch.reshape(inp + 1, [3, 3])

        Z = reshaped * bias
        Zl = torch.log(Z + 1)
        S = Zl.sum() + 1

        S.backward()
        return dict(inp_gradient=inp.grad, bias_gradient=bias.grad)

    return (SDFGBackwardRunner(outer_sdfg, "__return",
                               strict=False), torch_func,
            dict(inp=np.random.rand(9).astype(np.float64),
                 bias=np.random.rand(3).astype(np.float64)))


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
