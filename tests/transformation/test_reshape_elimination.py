from daceml.transformation import ReshapeElimination, expand_library_nodes_except_reshape
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from daceml.pytorch import DaceModule
import pytest


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(6, 16, 5)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv(x)), 2)
        x = x.view(-1, 256)
        return F.relu(x)


@pytest.mark.pure
def test_reshape_elimination(sdfg_name):

    import daceml.onnx as donnx
    donnx.default_implementation = "pure"

    ptmodel = Model()
    x = torch.rand((100, 6, 12, 12))
    dace_model = DaceModule(ptmodel, auto_optimize=False, sdfg_name=sdfg_name)

    def ApplyReshapeElimination(dace_module):
        sdfg = dace_module.sdfg
        expand_library_nodes_except_reshape(sdfg)
        applied = sdfg.apply_transformations_repeated([ReshapeElimination],
                                                      print_report=True)
        assert applied == 1

    dace_model.append_post_onnx_hook("ApplyReshapeElimination",
                                     ApplyReshapeElimination)

    dace_output = dace_model(x)
    torch_output = ptmodel(x)

    assert np.allclose(torch_output.detach().numpy(), dace_output, atol=1e-06)
