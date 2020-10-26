import numpy as np
import pytest

import torch
import torch.nn.functional as F

from daceml.pytorch import DaceModule


def run_pytorch_module(module, shape=None):
    shape = shape or (3, 5)

    input_values = [torch.rand(*shape, dtype=torch.float32) for _ in range(5)]
    pytorch_inputs = [
        torch.empty(*shape, dtype=torch.float32, requires_grad=False)
        for _ in range(5)
    ]
    dace_inputs = [
        torch.empty(*shape, dtype=torch.float32, requires_grad=False)
        for _ in range(5)
    ]

    pytorch_outputs = []
    for inp, inp_src in zip(pytorch_inputs, input_values):
        inp.copy_(inp_src)
        inp.requires_grad = True
        s = module(inp).sum()
        s.backward()
        pytorch_outputs.append(inp.grad)
        print(pytorch_outputs[-1])

    dace_module = DaceModule(module, backward=True)

    dace_outputs = []
    for inp, inp_src in zip(dace_inputs, input_values):
        inp.copy_(inp_src)
        inp.requires_grad = True
        s = dace_module(inp).sum()
        s.backward()
        dace_outputs.append(inp.grad.clone().detach())
        print(dace_outputs[-1])

    assert len(pytorch_outputs) == len(dace_outputs) == len(input_values)
    assert all(
        np.allclose(a, b) for a, b in zip(pytorch_outputs, dace_outputs))


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

    run_pytorch_module(Module())


@pytest.mark.skip(reason="check later")
def test_weights():
    pass


if __name__ == "__main__":
    test_simple()
