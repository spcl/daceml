import pytest
from torch.nn import functional as F

import dace
import torch

from daceml import onnx as donnx
from daceml.testing import torch_tensors_close
from daceml.transformation.init_state_fusion import InitStateFusion


@pytest.mark.pure
def test_init_state_fusion():
    @dace.program
    def test_init_state_fusion(X_arr: dace.float32[5, 3, 10, 10],
                               W_arr: dace.float32[16, 3, 3, 3],
                               Y_arr: dace.float32[5, 16, 8, 8]):
        donnx.ONNXConv(X=X_arr, W=W_arr, Y=Y_arr)

    X = torch.rand(5, 3, 10, 10)
    W = torch.rand(16, 3, 3, 3)
    Y = torch.empty(5, 16, 8, 8)

    sdfg: dace.SDFG = test_init_state_fusion.to_sdfg()
    sdfg.expand_library_nodes()
    assert sdfg.apply_transformations(InitStateFusion) == 1

    sdfg(X_arr=X, W_arr=W, Y_arr=Y)
    torch_tensors_close("output", Y, F.conv2d(X, W))
