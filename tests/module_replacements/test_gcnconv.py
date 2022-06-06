import pytest
import torch
from torch import nn
import numpy as np

from torch_geometric.nn import GCNConv

from daceml.pytorch.module import dace_module


@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.pure
def test_gcnconv(bias):
    @dace_module(sdfg_name='GCN_with_bias' if bias else 'GCN')
    class GCN(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = GCNConv(2, 3, bias=bias)
            self.conv1.lin.weight = nn.Parameter(
                torch.Tensor([[1, 0, 1], [1, 0, 0]]))
            if bias:
                self.conv1.bias = nn.Parameter(torch.Tensor([0.21, 0.37, 0]))

        def forward(self, x, edge_list):
            x = self.conv1(x, edge_list)
            return x

    model = GCN()
    x = torch.tensor([[0., 1], [1, 1], [-1, 0]])

    # Edges not considered for now.
    edges = torch.tensor([[0, 0], [0, 1], [2, 1], [2, 2]])
    pred = model(x, edges)
    expected_pred = np.array([[1.,  0.,  0.],
                              [2.,  0.,  1.],
                              [-1.,  0., -1.]]) + bias * np.array([0.21, 0.37, 0])
    print(pred, expected_pred)
    assert np.allclose(pred, expected_pred)
