import numpy as np
import torch
import torch.nn as nn

import dace
import daceml.onnx as donnx
import copy
from daceml.pytorch import DaceModule
from daceml.transformation import InputToConstant
from dace.transformation.interstate import FPGATransformSDFG, InlineSDFG



class TestModule(nn.Module):
    def __init__(self):
        super(TestModule, self).__init__()
        self.fc1 = nn.Linear(5, 3)

    def forward(self, x):
        return self.fc1(x)


def test_input_to_constant():
    donnx.ONNXGemm.default_implementation = "pure"

    net = TestModule()
    dace_net = DaceModule(net, dummy_inputs=(torch.rand(10, 5), ))

    inp = torch.rand((10, 5))

    fpga_dace_net = copy.deepcopy(dace_net)
    #
    sdfg: dace.SDFG = dace_net.sdfg

    # sdfg.expand_library_nodes()
    # sdfg.apply_transformations_repeated([InputToConstant], print_report=True)

    torch_result = net(torch.clone(inp))
    # dace_result = dace_net(torch.clone(inp))
    # assert np.allclose(torch_result.detach().numpy(), dace_result)
    donnx.ONNXGemm.default_implementation = "fpga"
    sdfg.save('/tmp/out.sdfg')
    sdfg = fpga_dace_net.sdfg
    sdfg.apply_transformations([FPGATransformSDFG])

    sdfg.expand_library_nodes()
    sdfg.apply_transformations_repeated([InputToConstant], print_report=True)
    sdfg.view()
    sdfg.save('/tmp/out_fpga.sdfg')
    dace_output_fpga = fpga_dace_net(torch.clone(inp))
    assert np.allclose(torch_result.detach().numpy(), dace_output_fpga)



test_input_to_constant()
