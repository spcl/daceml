import numpy as np
import torch

from daceml.pytorch import DaceModule
from daceml.util.utils import paramdec


def run_pytorch_module(module, shape=None):
    shape = shape or (3, 5)

    input_values = [torch.rand(*shape, dtype=torch.float32) for _ in range(5)]
    pytorch_inputs = [torch.empty(*shape, dtype=torch.float32, requires_grad=False) for _ in range(5)]
    dace_inputs = [torch.empty(*shape, dtype=torch.float32, requires_grad=False) for _ in range(5)]


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
    assert all(np.allclose(a, b) for a, b in zip(pytorch_outputs, dace_outputs))

def test_simple():
    class Module(torch.nn.Module):
        def forward(self, x):
            x = torch.sqrt(x)
            x = torch.log(x)
            return x
    run_pytorch_module(Module())


if __name__ == "__main__":
    test_simple()
