import numpy as np
import pytest
import torch
from torch import nn
from torch_geometric.nn import GATConv
from torch_sparse import SparseTensor

from daceml.pytorch.module import dace_module


@pytest.mark.parametrize("bias", [False, True], ids=['', 'bias'])
@pytest.mark.pure
def test_gcnconv(bias):
    weights_values = torch.Tensor([[1, 1], [0, 0], [1, 0]])
    bias_values = torch.Tensor([0.21, 0.37, 0])
    att_src_values = torch.Tensor([1, 1, 1])
    att_dst_values = torch.Tensor([1, 1, 1])

    @dace_module(sdfg_name=f'GAT_{bias}')
    class GCN(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = GATConv(
                2, 3, negative_slope=0.0, bias=bias, add_self_loops=False)
            self.conv1.lin_src.weight = nn.Parameter(weights_values)
            self.conv1.att_src = nn.Parameter(att_src_values)
            self.conv1.att_dst = nn.Parameter(att_dst_values)
            if bias:
                self.conv1.bias = nn.Parameter(bias_values)

        def forward(self, x, *edge_info):
            x = self.conv1(x, *edge_info)
            return x

    model = GCN()

    # edges = torch.tensor([[0, 0, 0, 2, 2], [0, 1, 2, 0, 2]])
    edges = torch.tensor([[0, 2], [2, 1]])
    x = torch.tensor([[0., 1], [1, 1], [-1, 0]])
    adj_matrix = SparseTensor.from_edge_index(edges, sparse_sizes=(x.shape[0], x.shape[0]))
    rowptr, col, _ = adj_matrix.csr()

    pred = model(x, rowptr, col)

    original_gcnconv = GATConv(
        2, 3, negative_slope=0.0, bias=bias, add_self_loops=False)
    original_gcnconv.lin_src.weight = nn.Parameter(weights_values)
    original_gcnconv.att_src = nn.Parameter(att_src_values)
    original_gcnconv.att_dst = nn.Parameter(att_dst_values)
    if bias:
        original_gcnconv.bias = nn.Parameter(bias_values)
    # PyG requires that the adj matrix is transposed when using SparseTensor.
    expected_pred = original_gcnconv(x, adj_matrix.t()).detach().numpy()

    print('\nCalculated: \n', pred)
    print('Expected: \n', expected_pred)
    assert np.allclose(pred, expected_pred)
