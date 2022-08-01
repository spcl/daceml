import copy

import pytest

import numpy as np
import torch
from torch import nn, autograd

import dace

from daceml.torch import DaceModule
from daceml.testing import torch_tensors_close, copy_to_gpu, tensors_close


@pytest.mark.pure
def test_parse_backward():
    gpu = False
    module = torch.nn.Sequential(
        torch.nn.Sequential(torch.nn.Linear(12, 24), torch.nn.Linear(24, 3)),
        nn.Softmax(dim=1))
    torch_module = copy.deepcopy(module)
    dace_module = copy.deepcopy(module)

    torch_module = copy_to_gpu(gpu, torch_module)
    dace_module = copy_to_gpu(gpu, dace_module)
    dace_module = DaceModule(dace_module)

    x = copy_to_gpu(gpu, torch.randn(8, 12))
    y = copy_to_gpu(gpu, torch.empty(8, dtype=torch.long).random_(3))

    expected_output = torch_module(x)
    result = dace_module(x)
    torch_tensors_close('output', expected_output, result)

    torch_criterion = copy_to_gpu(gpu, nn.CrossEntropyLoss())
    torch_loss = torch_criterion(expected_output, y)

    dace_criterion = DaceModule(torch_criterion)
    dace_loss = dace_criterion(expected_output, y)

    torch_tensors_close('loss', torch_loss, dace_loss)

    @dace
    def train_step(x, y):
        output = dace_module(x)
        loss = dace_criterion(output, y)
        torch.autograd.backward(loss)
        return loss

    # sdfg = train_step.to_sdfg(x, y)
    # sdfg.view(8000)

    result = train_step(x, y)
    tensors_close('parsed', torch_loss, result)

    assert all(
        hasattr(p, 'grad') and p.grad is not None
        for p in dace_module.parameters())

    torch_loss.backward()
    assert all(
        hasattr(p, 'grad') and p.grad is not None
        for p in dace_module.parameters())


@pytest.mark.pure
def test_parse_backward_simple():
    x = torch.randn(10, 5)
    dy = torch.randn(10)

    @dace.program
    def train_step(x: dace.float32[10, 5], dy: dace.float32[10]):
        red = np.add.reduce(x, axis=1)
        torch.autograd.backward(red, dy)
        return x.grad

    sdfg = train_step.to_sdfg()

    result = train_step(x.clone(), dy.clone())
    tensors_close('x.grad', dy.reshape(10, 1).expand(10, 5), result)


# FIXME Cases to support
# two independent trees of .backward
# two .backward with a shared gradient buffer
