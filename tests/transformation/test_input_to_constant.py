import numpy as np
import torch
import torch.nn as nn
import pytest

import dace
import daceml.onnx as donnx
from daceml.pytorch import DaceModule
from daceml.transformation import InputToConstant


class TestModule(nn.Module):
    def __init__(self):
        super(TestModule, self).__init__()
        self.fc1 = nn.Linear(5, 3)

    def forward(self, x):
        return self.fc1(x)


@pytest.mark.ort
def test_input_to_constant():
    donnx.ONNXGemm.default_implementation = "pure"

    net = TestModule()
    dace_net = DaceModule(net, dummy_inputs=(torch.rand(10, 5), ))

    inp = torch.rand((10, 5))
    #
    sdfg: dace.SDFG = dace_net.sdfg
    sdfg.expand_library_nodes()
    sdfg.apply_transformations_repeated([InputToConstant], print_report=True)

    torch_result = net(torch.clone(inp))
    dace_result = dace_net(torch.clone(inp))

    assert np.allclose(torch_result.detach().numpy(), dace_result)
