import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

import daceml.onnx as donnx
from daceml.pytorch import DaceModule, dace_module


def test_conv2d(default_implementation):
    class Model(nn.Module):
        def __init__(self):
            super(Model, self).__init__()
            self.conv1 = nn.Conv2d(1, 4, 3)
            self.conv2 = nn.Conv2d(4, 4, 3)

        def forward(self, x):
            x = F.relu(self.conv1(x))
            return F.relu(self.conv2(x))

    ptmodel = Model()
    x = torch.rand(1, 1, 8, 8)

    @dace_module
    class TestDecorator(Model):
        pass

    dace_model = DaceModule(ptmodel)
    dace_output = dace_model(x)

    dace_model_decorated = TestDecorator()
    dace_model_decorated(x)

    torch_output = ptmodel(x)

    assert np.allclose(torch_output.detach().numpy(), dace_output, atol=1e-06)
