import numpy as np
import pytest

import torch
import torch.nn as nn
import torch.nn.functional as F

from daceml.pytorch import DaceModule


def run_pytorch_module(module,
                       sdfg_name,
                       shape=None,
                       use_max=False,
                       apply_strict=False):
    shape = shape or (3, 5)

    input_value = torch.rand(*shape, dtype=torch.float32)

    pytorch_input = torch.empty(*shape,
                                dtype=torch.float32,
                                requires_grad=False)
    pytorch_input.copy_(input_value)
    pytorch_input.requires_grad = True

    dace_input = torch.empty(*shape, dtype=torch.float32, requires_grad=False)
    dace_input.copy_(input_value)
    dace_input.requires_grad = True

    if use_max:
        pytorch_s = module(pytorch_input).max()
    else:
        pytorch_s = module(pytorch_input).sum()
    pytorch_s.backward()

    print("Pytorch output:")
    print(pytorch_input.grad)

    dace_module = DaceModule(module,
                             backward=True,
                             sdfg_name=sdfg_name,
                             apply_strict=apply_strict)

    if use_max:
        dace_s = dace_module(dace_input).max()
    else:
        dace_s = dace_module(dace_input).sum()
    dace_s.backward()
    print("Dace output:")
    print(dace_input.grad)
    assert torch.allclose(pytorch_input.grad,
                          dace_input.grad,
                          rtol=1e-6,
                          atol=1e-4)


def test_simple(sdfg_name):
    class Module(torch.nn.Module):
        def forward(self, x):
            x = torch.sqrt(x)
            x = torch.log(x)
            return x

    run_pytorch_module(Module(), sdfg_name)


def test_repeated(sdfg_name):
    class Module(torch.nn.Module):
        def forward(self, x):
            x = torch.sqrt(x)
            x = torch.sqrt(x)
            return x

    run_pytorch_module(Module(), sdfg_name)


def test_softmax(sdfg_name):
    class Module(torch.nn.Module):
        def forward(self, x):
            x = F.softmax(x, dim=1)
            return x

    run_pytorch_module(Module(), sdfg_name, use_max=True)


def test_reshape_on_memlet_path(sdfg_name):
    # required test: this function in a nn.Module, with apply strict so that the reshape is
    # inlined and copy is removed
    class Module(torch.nn.Module):
        def forward(self, x):
            reshaped = torch.reshape(x + 1, [3, 3])
            return torch.log(reshaped) + torch.reshape(
                torch.tensor([[3, 2, 1]]), [3])

    run_pytorch_module(Module(), sdfg_name, shape=(9, ), apply_strict=True)


def test_weights_ln(sdfg_name):
    class Module(torch.nn.Module):
        def __init__(self):
            super(Module, self).__init__()
            self.fc1 = nn.Linear(784, 120)
            self.fc2 = nn.Linear(120, 32)
            self.ln = nn.LayerNorm(32)
            self.fc3 = nn.Linear(32, 10)

        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.ln(x)
            x = self.fc3(x)
            return x

    run_pytorch_module(Module(), sdfg_name, shape=(4, 784), use_max=False)


def test_layernorm(sdfg_name):
    class Module(torch.nn.Module):
        def __init__(self):
            super(Module, self).__init__()
            self.ln = nn.LayerNorm(3)

        def forward(self, x):
            return self.ln(x)

    run_pytorch_module(Module(), sdfg_name, shape=(1, 3), use_max=True)


def test_weights(sdfg_name):
    class Module(torch.nn.Module):
        def __init__(self):
            super(Module, self).__init__()
            self.fc1 = nn.Linear(784, 120)
            self.fc2 = nn.Linear(120, 32)
            self.fc3 = nn.Linear(32, 10)

        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    run_pytorch_module(Module(), sdfg_name, shape=(4, 784), use_max=False)


def test_batched_matmul(sdfg_name):
    class Module(torch.nn.Module):
        def __init__(self):
            super(Module, self).__init__()
            self.fc1 = nn.Parameter(torch.ones([10, 5, 3]))

        def forward(self, x):
            x = self.fc1 @ x
            return x

    run_pytorch_module(Module(), sdfg_name, use_max=False)
