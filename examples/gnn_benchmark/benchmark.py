import argparse
import functools
import logging
import statistics
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import GCNNorm
from torch_sparse import SparseTensor

from daceml import onnx as donnx
from daceml.torch.module import dace_module
from examples.gnn_benchmark.models import LinearModel, GCN, GAT
from examples.gnn_benchmark.util import specialize_mem_onnx, \
    apply_dace_auto_optimize, make_maps_dynamic

donnx.default_implementation = "pure"
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

torch.manual_seed(42)
np.random.seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def stats_as_string(times, func_names, model, hidden_size):
    """ Print timing statistics.
    :param times: the result of time_funcs.
    :param func_names: a name to use for each function timed.
    """

    out = ''
    for name, func_time, in zip(func_names, times):
        row = [
            name, model, hidden_size,
            min(func_time),
            statistics.mean(func_time),
            statistics.median(func_time),
            statistics.stdev(func_time) if len(func_time) > 1 else 0.0,
            max(func_time)
        ]
        out += ','.join(map(str, row)) + '\n'
    return out


def check_correctness(dace_model, torch_model, dace_args,
                      torch_edge_list_args):
    dace_pred = dace_model(*dace_args)
    torch_pred = torch_model(*torch_edge_list_args)
    dace_pred_cpu = dace_pred.detach().cpu()
    torch_pred_cpu = torch_pred.detach().cpu()
    if np.allclose(dace_pred_cpu, torch_pred_cpu, atol=1.0e-5):
        print("\n==== Results correct.  ☆ ╰(o＾◡＾o)╯ ☆ ====")
        return True
    else:
        print("\n****** INCORRECT RESULTS! (ノಥ﹏ಥ)ノ彡┻━┻ ******")
        print("Max abs error: ", abs((dace_pred_cpu - torch_pred_cpu)).max())
        print(dace_pred_cpu - torch_pred_cpu)
        return False


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='benchmark')
    parser.add_argument('--small', action='store_true')
    parser.add_argument('--dry', action='store_true')
    parser.add_argument('--onlydace', action='store_true')
    parser.add_argument('--normalize', action='store_true')
    parser.add_argument('--persistent-mem', action='store_true')
    parser.add_argument('--opt', action='store_true')
    parser.add_argument('--threadblock-dynamic', action='store_true')
    parser.add_argument('--model', choices=['gcn', 'gat', 'linear'])
    parser.add_argument('--hidden', type=int, default=None, required=True)
    parser.add_argument('--outfile', type=str, default=None)
    parser.add_argument('--name', type=str, default='dace')
    args = parser.parse_args()
    models = {'gcn': GCN, 'linear': LinearModel, 'gat': GAT}
    model_class = models[args.model]
    num_hidden_features = args.hidden
    args.hidden = args.hidden or (8 if args.model == 'gat' else 512)
    outfile = Path(args.outfile) if args.outfile is not None else None

    log = logging.getLogger(__name__)
    log.setLevel(logging.INFO)

    if not args.small:
        dataset = Planetoid(root='/tmp/Cora', name='Cora')
        data = dataset[0].to(device)
        num_node_features = dataset.num_node_features
        num_classes = dataset.num_classes
    else:
        _x = torch.tensor([[0., 1], [1, 1], [-1, 0]]).to(device)
        _edge_index = torch.tensor([[0, 0, 0, 2, 2], [0, 1, 2, 0,
                                                      2]]).to(device)
        _edge_attr = torch.tensor([1, 1, 1, 1., 1]).to(device)
        data = Data(x=_x, edge_index=_edge_index, edge_attr=_edge_attr)
        num_node_features = _x.shape[1]
        num_classes = 2

    print("Num node features: ", num_node_features)
    print("Num classes: ", num_classes)
    print("Num hidden features: ", num_hidden_features)
    normalize = args.normalize
    print("Normalize: ", normalize)

    if args.model == 'gcn':
        gcn_norm = GCNNorm(add_self_loops=True)
        data = gcn_norm(data)
    x = data.x
    sparse_edge_index = SparseTensor.from_edge_index(
        data.edge_index, edge_attr=data.edge_weight)
    edge_rowptr, edge_col, edge_weights = sparse_edge_index.csr()

    torch_model = model_class(num_node_features, num_hidden_features,
                              num_classes, normalize).to(device)
    dace_model = dace_module(model_class)(num_node_features,
                                          num_hidden_features, num_classes,
                                          normalize).to(device)

    dace_model.model.load_state_dict(torch_model.state_dict())

    dace_model.eval()
    torch_model.eval()

    if args.opt:
        print("---> Adding auto-opt hook.")
        dace_model.append_post_onnx_hook("dace_auto_optimize",
                                         apply_dace_auto_optimize)

    if args.persistent_mem:
        print("---> Adding persistent memory hook.")
        specialize_mem_onnx(dace_model)

    if args.threadblock_dynamic:
        print("---> Adding threadblock dynamic maps hook.")
        exclude_loops = []
        if args.model == 'gcn':
            # Has to be skipped, otherwise the computation results are incorrect.
            exclude_loops = [
                'daceml_onnx_op_implementations_replacement_implementations_prog_sparse_45_4_46'
            ]
        elif args.model == 'gat':
            exclude_loops = [
                # Below two have to be excluded because only one-dimensional
                # maps are supported in DaCe for dynamic block map schedule
                # (got 2).
                '_Div__map',
                '_Mult__map',
                # Below two have to be excluded, otherwise compile errors
                # occur (the generated code is incorrect).
                'assign_137_12_map',
                'outer_fused',
            ]
        make_maps_dynamic_with_excluded_loops = functools.partial(
            make_maps_dynamic, exclude_loops=exclude_loops)
        dace_model.append_post_onnx_hook(
            "apply_threadblock_dynamic_maps",
            make_maps_dynamic_with_excluded_loops)

    dace_args = (x, ) if args.model == 'linear' else (x, edge_rowptr, edge_col)
    # pyg requires the sparse tensor input to be transposed.
    torch_csr_args = (x, ) if args.model == 'linear' else (
        x, sparse_edge_index.t())
    torch_edge_list_args = (x, ) if args.model == 'linear' else (
        x, data.edge_index)

    if edge_weights is not None and args.model == 'gcn':
        dace_args += (edge_weights, )
        torch_edge_list_args += (data.edge_weight, )

    correct = check_correctness(dace_model, torch_model, dace_args,
                                torch_edge_list_args)

    if args.onlydace:
        print('Only dace model for profiling.')
        print("Dace: ", dace_model(*dace_args))
    elif args.dry:
        print("Single run of all models.")
        print("Dace: ", dace_model(*dace_args))
        print("PyG csr: ", torch_model(*torch_csr_args))
        print("PyG edge list: ", torch_model(*torch_edge_list_args))
    else:
        print("Benchmarking...")
        from daceml.testing.profiling import time_funcs, print_time_statistics

        funcs = [
            lambda: dace_model(*dace_args),
            lambda: torch_model(*torch_csr_args),
            lambda: torch_model(*torch_edge_list_args),
        ]

        name = args.name
        if args.threadblock_dynamic:
            name += "_tb-dynamic"
        if args.opt:
            name += "_autoopt"
        if args.persistent_mem:
            name += "_persistent_mem"
        func_names = [name, 'torch_csr', 'torch_edge_list']
        times = time_funcs(funcs,
                           func_names=func_names,
                           warmups=10,
                           num_iters=100)
        print(f"\n------ {args.model.upper()} ------")
        print_time_statistics(times, func_names)
        print()

        if outfile is not None and correct:
            add_header = not outfile.exists()
            with open(outfile, 'a') as file:
                if add_header:
                    headers = [
                        'Name', 'Model', 'Size', 'Min', 'Mean', 'Median',
                        'Stdev', 'Max'
                    ]
                    file.write(','.join(headers) + '\n')
                file.write(
                    stats_as_string(times, func_names, args.model,
                                    args.hidden))
