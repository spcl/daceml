import numpy as np
import pytest

import torch
import torch.nn as nn
import torch.nn.functional as F

from daceml.pytorch import DaceModule


def run_pytorch_module(module, shape=None, use_max=False):
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
        s = module(pytorch_input).max()
    else:
        s = module(pytorch_input).sum()
    s.backward()

    print("Pytorch output:")
    print(pytorch_input.grad)

    dace_module = DaceModule(module, backward=True)

    if use_max:
        s = dace_module(dace_input).max()
    else:
        s = dace_module(dace_input).sum()
    s.backward()
    print("Dace output:")
    print(dace_input.grad)

    assert np.allclose(pytorch_input.grad, dace_input.grad)


def test_simple():
    class Module(torch.nn.Module):
        def forward(self, x):
            x = torch.sqrt(x)
            x = torch.log(x)
            return x

    run_pytorch_module(Module())


def test_repeated():
    class Module(torch.nn.Module):
        def forward(self, x):
            x = torch.sqrt(x)
            x = torch.sqrt(x)
            return x

    run_pytorch_module(Module())


def test_softmax():
    class Module(torch.nn.Module):
        def forward(self, x):
            x = F.softmax(x, dim=1)
            return x

    run_pytorch_module(Module(), use_max=True)


def test_weights():
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

    run_pytorch_module(Module(), shape=(4, 784), use_max=False)


if __name__ == "__main__":
    test_simple()
