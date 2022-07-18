import argparse
import copy
import logging

import dace
import numpy as np
import torch
import torch.nn.functional as F
from dace.library import change_default
from dace.transformation.dataflow import TrivialMapRangeElimination, MapFusion
from torch import nn
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv
from torch_geometric.utils import add_self_loops
from torch_sparse import SparseTensor

from benchmark.util import specialize_mem_onnx, apply_dace_auto_optimize
from daceml import onnx as donnx
from daceml.pytorch.module import dace_module
from daceml.transformation import TaskletFusion
from daceml.util import utils

donnx.default_implementation = "pure"
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

torch.manual_seed(42)
np.random.seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_hidden_features = 512


class GCN(torch.nn.Module):
    def __init__(self,  normalize):
        super().__init__()
        print("normalize: ", normalize)
        self.conv1 = GCNConv(num_node_features, num_hidden_features,
                             normalize=normalize, add_self_loops=False)
        self.conv2 = GCNConv(num_hidden_features, num_hidden_features,
                             normalize=normalize, add_self_loops=False)

        self.act = nn.ReLU()
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x, *edge_info):
        x = self.conv1(x, *edge_info)
        # return x
        x = self.act(x)
        x = self.conv2(x, *edge_info)

        return self.log_softmax(x)


class LinearModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(num_node_features, 16)

    def forward(self, x):
        x = self.lin(x)
        return x


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='benchmark')
    parser.add_argument('--linear', action='store_true')
    parser.add_argument('--small', action='store_true')
    parser.add_argument('--dry', action='store_true')
    parser.add_argument('--onlydace', action='store_true')
    parser.add_argument('--no-normalize', action='store_true')
    parser.add_argument('--persistent-mem', action='store_true')
    parser.add_argument('--opt', action='store_true')
    args = parser.parse_args()
    do_gcn = not args.linear
    model_class = LinearModel if args.linear else GCN

    log = logging.getLogger(__name__)
    log.setLevel(logging.INFO)

    if not args.small:
        dataset = Planetoid(root='/tmp/Cora', name='Cora')
        data = dataset[0].to(device)
        x = data.x
        edge_index = data.edge_index
        num_node_features = dataset.num_node_features
        num_classes = dataset.num_classes
    else:
        x = torch.tensor([[0., 1], [1, 1], [-1, 0]]).to(device)
        edge_index = torch.tensor(
            [[0, 0, 0, 2, 2], [0, 1, 2, 0, 2]]).to(device)
        num_node_features = x.shape[1]
        num_classes = 2

    print("Num node features: ", num_node_features)
    print("Num classes: ", num_classes)
    print("Num hidden features: ", num_hidden_features)
    normalize = not args.no_normalize
    print("Normalize: ", normalize)

    edge_index, _ = add_self_loops(edge_index, num_nodes=x.shape[0])
    sparse_edge_index = SparseTensor.from_edge_index(edge_index)
    edge_rowptr, edge_col, _ = sparse_edge_index.csr()

    torch_model = model_class(normalize).to(device)
    dace_model = dace_module(model_class)(normalize).to(device)

    dace_model.model.conv1.lin.weight = copy.deepcopy(
        torch_model.conv1.lin.weight)
    dace_model.model.conv1.bias = copy.deepcopy(torch_model.conv1.bias)
    dace_model.model.conv2.lin.weight = copy.deepcopy(
        torch_model.conv2.lin.weight)
    dace_model.model.conv2.bias = copy.deepcopy(torch_model.conv2.bias)

    dace_model.eval()
    torch_model.eval()

    if args.opt:
        dace_model.append_post_onnx_hook("dace_auto_optimize", apply_dace_auto_optimize)

    if args.persistent_mem:
        specialize_mem_onnx(dace_model)

    dace_args = (x,) if args.linear else (
        x, edge_rowptr, edge_col)
    # pyg requires the sparse tensor input to be transposed.
    torch_sparse_args = (x,) if args.linear else (x, sparse_edge_index.t())
    torch_dense_args = (x,) if args.linear else (x, edge_index)

    if args.onlydace:
        print('Only dace model for profiling.')
        print("Dace: ", dace_model(*dace_args))
    elif args.dry:
        print("Single run of all models.")
        print("Dace: ", dace_model(*dace_args))
        print("PyG sparse: ", torch_model(*torch_sparse_args))
        print("PyG dense: ", torch_model(*torch_dense_args))
    else:
        print("Benchmarking...")
        from daceml.testing.profiling import time_funcs, print_time_statistics

        funcs = [
            lambda: dace_model(*dace_args),
            lambda: torch_model(*torch_sparse_args),
            lambda: torch_model(*torch_dense_args),
        ]

        func_names = ['dace', 'torch_sparse', 'torch_dense']
        times = time_funcs(funcs,
                           func_names=func_names,
                           warmups=10,
                           num_iters=100)
        print_time_statistics(times, func_names)

    dace_pred = dace_model(*dace_args)
    torch_pred = torch_model(*torch_dense_args)

    dace_pred_cpu = dace_pred.detach().cpu()
    torch_pred_cpu = torch_pred.detach().cpu()
    if np.allclose(dace_pred_cpu, torch_pred_cpu, atol=1.0e-4):
        print("\n==== Results correct.  ☆ ╰(o＾◡＾o)╯ ☆ ====")
    else:
        print("\n****** INCORRECT RESULTS! (ಥ﹏ಥ) ******")
        print("Max abs error: ", abs((dace_pred_cpu - torch_pred_cpu)).max())
        print(dace_pred_cpu - torch_pred_cpu)
