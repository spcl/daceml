import pytest
import torch
from torch import nn
import numpy as np

from torch_geometric.nn import GCNConv

from daceml.pytorch.module import dace_module


@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("self_loops", [False, True])
@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.pure
def test_gcnconv(normalize, self_loops, bias):
    weights_values = torch.Tensor([[1, 1], [0, 0], [1, 0]])
    bias_values = torch.Tensor([0.21, 0.37, 0])

    @dace_module(sdfg_name=f'GCN_{self_loops}_{normalize}_{bias}')
    class GCN(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = GCNConv(
                2, 3, bias=bias, normalize=normalize, add_self_loops=self_loops)
            self.conv1.lin.weight = nn.Parameter(weights_values)
            if bias:
                self.conv1.bias = nn.Parameter(bias_values)

        def forward(self, x, edge_list):
            x = self.conv1(x, edge_list)
            return x

    model = GCN()
    x = torch.tensor([[0., 1], [1, 1], [-1, 0]])
    edges = torch.tensor([[0, 0, 0], [0, 0, 1]])
    pred = model(x, edges)

    original_gcnconv = GCNConv(
        2, 3, bias=bias, normalize=normalize, add_self_loops=self_loops)
    original_gcnconv.lin.weight = nn.Parameter(weights_values)
    if bias:
        original_gcnconv.bias = nn.Parameter(bias_values)

    expected_pred = original_gcnconv(x, edges).detach().numpy()

    print('Calculated: \n', pred)
    print('Expected: \n', expected_pred)
    assert np.allclose(pred, expected_pred)
