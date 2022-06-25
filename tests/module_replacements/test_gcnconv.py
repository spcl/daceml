import numpy as np
import pytest
import torch
from torch import nn
from torch_geometric.nn import GCNConv
from torch_sparse import SparseTensor

from daceml.pytorch.module import dace_module


@pytest.mark.parametrize("bias", [False, True], ids=['', 'bias'])
@pytest.mark.parametrize("self_loops", [False], ids=[''])
@pytest.mark.parametrize("normalize", [False, True], ids=['', 'normalize'])
@pytest.mark.pure
def test_gcnconv(normalize, self_loops, bias):
    assert not self_loops, "Adding self-loops in the module not supported."
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

        def forward(self, x, *edge_info):
            x = self.conv1(x, *edge_info)
            return x

    model = GCN()

    edges = torch.tensor([[0, 0, 0, 2, 2], [0, 1, 2, 0, 2]])
    adj_matrix = SparseTensor.from_edge_index(edges)
    rowptr, col, _ = adj_matrix.csr()
    x = torch.tensor([[0., 1], [1, 1], [-1, 0]])

    pred = model(x, rowptr, col)

    original_gcnconv = GCNConv(
        2, 3, bias=bias, normalize=normalize, add_self_loops=self_loops)
    original_gcnconv.lin.weight = nn.Parameter(weights_values)
    if bias:
        original_gcnconv.bias = nn.Parameter(bias_values)
    # PyG requires that the adj matrix is transposed when using SparseTensor.
    expected_pred = original_gcnconv(x, adj_matrix.t()).detach().numpy()

    print('\nCalculated: \n', pred)
    print('Expected: \n', expected_pred)
    assert np.allclose(pred, expected_pred)
