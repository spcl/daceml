import copy

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from daceml.pytorch import DaceModule
from daceml.testing import torch_tensors_close
from daceml.transformation import PadConvFusion


class Module(nn.Module):
    def __init__(self):
        super(Module, self).__init__()
        self.conv = nn.Conv2d(3, 6, kernel_size=(3, 3))

    def forward(self, x):
        x = F.pad(x, [2, 2])
        return self.conv(x)


@pytest.mark.ort
def test_pad_conv_fusion(sdfg_name):
    torch_module = Module()
    dace_module = DaceModule(copy.deepcopy(torch_module), sdfg_name=sdfg_name)

    input = torch.rand(4, 3, 24, 24)

    def test_pcf(module: DaceModule):
        assert module.sdfg.apply_transformations(PadConvFusion) == 1

    dace_module.prepend_post_onnx_hook("pcf", test_pcf)
    torch_output = torch_module(input)
    dace_output = dace_module(input)

    torch_tensors_close("output", torch_output, dace_output)
