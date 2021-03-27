import pytest
import torch
from torch import nn

from daceml.pytorch import DaceModule
from daceml.transformation import parameter_to_transient


@pytest.mark.gpu
def test_pytorch_from_dlpack():
    class Module(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(10, 3)

        def forward(self, x):
            return self.fc1(x)

    pt_module = Module()
    dace_module = Module()
    dace_module.load_state_dict(pt_module.state_dict())

    input = torch.rand(2, 10)
    assert torch.allclose(pt_module(input), dace_module(input))

    dace_module = DaceModule(dace_module, cuda=True)

    assert torch.allclose(dace_module(input), pt_module(input))
    parameter_to_transient(dace_module, "fc1.weight")
    assert torch.allclose(dace_module(input), pt_module(input))
