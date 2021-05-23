import torch
from torch import nn

from daceml.pytorch import dace_module
from daceml.testing import torch_tensors_close


@dace_module(debug_transients=True)
class Module(nn.Module):
    def forward(self, x):
        y = x + 3
        return y * 5


def test_debug_transients():

    module = Module()

    x = torch.rand(5, 5)
    output, y = module(x)

    torch_tensors_close("output", (x + 3) * 5, output)
    torch_tensors_close("y", x + 3, y)
