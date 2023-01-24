import numpy as np
import pytest
import torch
from torch import nn
from torch_geometric.nn import GCNConv
from torch_sparse import SparseTensor

from daceml.torch.module import dace_module, DaceModule

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

@pytest.mark.parametrize("bias", [False, True], ids=['', 'bias'])
@pytest.mark.pure
def test_gcnconv(bias):
    self_loops = False
    normalize = False

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
    edge_values = torch.tensor([1., 2., 3., 4., 5.])
    adj_matrix = SparseTensor.from_edge_index(edges, edge_attr=edge_values)
    rowptr, col, edge_vals = adj_matrix.csr()
    x = torch.tensor([[0., 1], [1, 1], [-1, 0]])

    pred = model(x, rowptr, col, edge_vals)

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


@pytest.mark.parametrize("seed", list(range(5)))
@pytest.mark.pure
def test_gcnconv_full_model(seed):
    rng = torch.Generator()
    rng.manual_seed(seed)

    size = 8
    num_nodes = size
    num_in_features = 2 * size
    num_hidden_features = 3 * size
    num_classes = 4

    weights_values_1 = torch.randn((num_hidden_features, num_in_features), generator=rng) / 10
    bias_values_1 = torch.randn((num_hidden_features,), generator=rng) / 10
    weights_values_2 = torch.randn((num_classes, num_hidden_features), generator=rng) / 10
    bias_values_2 = torch.randn((num_classes,), generator=rng) / 10
    x = torch.randn((num_nodes, num_in_features), generator=rng) / 10
    edges = torch.randint(low=0, high=2, size=(num_nodes, num_nodes), generator=rng) / 10

    class GCN(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = GCNConv(num_in_features, num_hidden_features, normalize=False, add_self_loops=False)
            self.conv2 = GCNConv(num_hidden_features, num_classes, normalize=False, add_self_loops=False)
            self.act = nn.ReLU()

            self.conv1.lin.weight = nn.Parameter(weights_values_1)
            self.conv1.bias = nn.Parameter(bias_values_1)
            self.conv2.lin.weight = nn.Parameter(weights_values_2)
            self.conv2.bias = nn.Parameter(bias_values_2)

        def forward(self, x, *edge_info):
            x = self.conv1(x, *edge_info)
            x = self.act(x)
            x = self.conv2(x, *edge_info)
            return x

    torch_model = GCN()
    dace_model = DaceModule(GCN(), sdfg_name=f'GCN_{seed}')

    adj_matrix = SparseTensor.from_dense(edges)
    rowptr, col, edges = adj_matrix.csr()

    pred = dace_model(x, rowptr, col, edges)

    # PyG requires that the adj matrix is transposed when using SparseTensor.
    expected_pred = torch_model(x, adj_matrix.t()).detach().numpy()

    print('\nCalculated: \n', pred)
    print('Expected: \n', expected_pred)
    print(f"Max abs error: {abs((pred - expected_pred)).max()}")
    assert np.allclose(pred, expected_pred)
