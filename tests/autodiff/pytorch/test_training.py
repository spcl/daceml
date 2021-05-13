import os

import pytest

import numpy as np
import torch
from torchvision import datasets, transforms
from torch import nn, optim
from transformers import BertLayer, BertConfig
import copy
from daceml.pytorch import DaceModule

import daceml.onnx as donnx
from dace.sdfg import state as dace_state
from daceml.pytorch import DaceModule
from daceml.testing.utils import torch_tensors_close
from daceml.transformation import ConstantFolding
from dace import dtypes
from dace.sdfg import utils as sdutil
from dace.sdfg import nodes as sdfg_nodes
from typing import List
from dace.transformation.helpers import nest_state_subgraph

import dace
from dace.transformation.transformation import PatternNode
from dace.transformation.transformation import Transformation
from dace.transformation.pattern_matching import enumerate_matches
from dace.sdfg import utils as sdutil
from dace.transformation.dataflow.squeeze_view_remove import SqueezeViewRemove
from dace.transformation.dataflow.trivial_map_elimination import TrivialMapElimination
from dace.transformation.dataflow.trivial_map_range_elimination import TrivialMapRangeElimination
from dace.transformation.dataflow.map_expansion import MapExpansion
from dace.transformation.dataflow.map_collapse import MapCollapse
from dace.transformation.dataflow.strip_mining import StripMining
from dace.transformation.dataflow.stream_transient import AccumulateTransient
from dace.transformation.dataflow.nest_maps import NestMaps
from dace.transformation.dataflow.nest_access_nodes import NestExitAccessNode
from dace.transformation.dataflow.nest_access_nodes import NestEntryAccessNode
from dace.transformation.dataflow.nest_access_nodes import RemoveUnusedAccessNode
from dace.transformation.dataflow.nested_sdfg_fusion import NestedSDFGFusion
from dace.transformation.dataflow.clean_connectors import CleanNestedSDFGConnectors, RemoveDanglingAccessNodes, NestTransients
from dace.transformation.dataflow.nest_maps import NestMapContent
from dace.transformation.interstate.nested_map_fusion import NestedMapFusion
from dace.transformation.dataflow.clean_connectors import UnifyInOutNestedSDFGConnectors
from dace.transformation.interstate.warp_all_reduce_detection import WarpAllReduceDetectionNoTasklet
from dace.sdfg.propagation import propagate_memlets_sdfg
from dace.transformation.dataflow.add_nsdfg_connector import AddNestedSDFGInputConnector
from dace.transformation.dataflow.clean_connectors import RemoveReadSDFGConnectors
from dace.transformation.dataflow.clean_connectors import NestTransients
from dace.transformation.interstate.state_elimination import EmptyStateElimination
from dace.libraries.standard.nodes.barrier import Barrier
from dace.transformation.dataflow.clean_connectors import CleanNestedWrites
from dace.transformation.interstate.remove_unused_states import RemoveUnusedStates
from dace.transformation.dataflow import PruneConnectors
from dace.transformation.dataflow.constant_propagation import ConstantPropagation
from dace.transformation.dataflow.clean_connectors import merge_symbols
from dace.transformation.interstate.state_elimination import EmptyStateElimination
from dace.libraries.standard.nodes.barrier import Barrier
from dace.transformation.interstate.gpu_transform_sdfg import GPUTransformSDFG
from dace.transformation.dataflow.vectorize_sdfg import VectorizeSDFG


def torch_tensors_close(name, torch_v, dace_v):
    rtol = 1e-6
    atol = 1e-4
    if not torch.allclose(torch_v, dace_v, rtol=rtol, atol=atol):
        print("torch value: ", torch_v)
        print("dace value: ", dace_v)
        print("diff: ", torch.abs(dace_v - torch_v))

        failed_mask = np.abs(torch_v.numpy() - dace_v.numpy()
                             ) > atol + rtol * np.abs(dace_v.numpy())
        print(f"wrong elements torch: {torch_v[failed_mask]}")
        print(f"wrong elements dace: {dace_v[failed_mask]}")

        for x, y in zip(torch_v[failed_mask], dace_v[failed_mask]):
            print(f"lhs_failed: {abs(x - y)}")
            print(f"rhs_failed: {atol} + {rtol * abs(y)}")

        assert False, f"{name} was not close)"


def training_step(dace_model,
                  pt_model,
                  train_batch,
                  sdfg_name,
                  gpu,
                  train_criterion=None):

    x, y = train_batch
    train_criterion = train_criterion or nn.NLLLoss()

    pt_loss = train_criterion(pt_model(x), y)

    dace_output = dace_model(x)
    dace_loss = train_criterion(dace_output, y)

    diff = abs(pt_loss.item() - dace_loss.item()) / pt_loss.item()
    assert diff < 1e-5

    pt_loss.backward()
    dace_loss.backward()

    for (name, dace_param), (pt_name,
                             pt_param) in zip(pt_model.named_parameters(),
                                              dace_model.named_parameters()):
        assert 'model.' + name == pt_name
        torch_tensors_close(name, pt_param.grad, dace_param.grad)

    optimizer = optim.SGD(pt_model.parameters(), lr=0.001)
    dace_optimizer = optim.SGD(dace_model.parameters(), lr=0.001)
    optimizer.step()
    dace_optimizer.step()

    for (name, dace_param), (pt_name,
                             pt_param) in zip(pt_model.named_parameters(),
                                              dace_model.named_parameters()):
        assert 'model.' + name == pt_name
        torch_tensors_close(name, pt_param.detach(), dace_param.detach())


def test_mnist(sdfg_name, gpu):
    input_size = 784
    hidden_sizes = [128, 64]
    output_size = 10

    # initialize modules
    # yapf: disable
    model = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),
                          nn.ReLU(),
                          nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                          nn.ReLU(),
                          nn.Linear(hidden_sizes[1], output_size),
                          nn.LayerNorm(output_size),
                          nn.LogSoftmax(dim=1))

    dace_model = DaceModule(copy.deepcopy(model),
                            backward=True,
                            sdfg_name=sdfg_name,
                            cuda=gpu)
    # yapf: enable

    # check forward pass using loss
    images = torch.randn(64, 784)
    labels = torch.randint(0, 10, [64], dtype=torch.long)

    training_step(dace_model, model, (images, labels), sdfg_name, gpu)


def apply_softmax_transformations(fwd_sdfg, bwd_sdfg):
    print('encoder1.sdfg')

    fwd_sdfg.expand_library_nodes()

    print('encoder2.sdfg')

    # find softmax sdfg and state

    softmax_nodes = [
        PatternNode(dace.nodes.MapEntry),
        PatternNode(dace.nodes.Tasklet),
        PatternNode(dace.nodes.MapExit),
        PatternNode(dace.nodes.AccessNode),
        PatternNode(dace.nodes.NestedSDFG),
        PatternNode(dace.nodes.AccessNode),
        PatternNode(dace.nodes.MapEntry),
        PatternNode(dace.nodes.Tasklet),
        PatternNode(dace.nodes.MapExit),
        PatternNode(dace.nodes.AccessNode),
        PatternNode(dace.nodes.MapEntry),
        PatternNode(dace.nodes.Tasklet),
        PatternNode(dace.nodes.MapExit),
        PatternNode(dace.nodes.AccessNode),
        PatternNode(dace.nodes.NestedSDFG),
        PatternNode(dace.nodes.AccessNode),
        PatternNode(dace.nodes.MapEntry),
        PatternNode(dace.nodes.Tasklet),
        PatternNode(dace.nodes.MapExit),
    ]
    pattern_graph = sdutil.node_path_graph(*softmax_nodes)
    pattern_graph.add_edge(softmax_nodes[3], softmax_nodes[6], None)
    pattern_graph.add_edge(softmax_nodes[13], softmax_nodes[16], None)

    subgraphs = list(enumerate_matches(fwd_sdfg, pattern_graph))

    assert (len(subgraphs) == 1)
    softmax_subgraph = subgraphs[0]
    softmax_nsdfg: sdfg_nodes.NestedSDFG = nest_state_subgraph(
        softmax_subgraph.graph.parent, softmax_subgraph.graph,
        softmax_subgraph, 'softmax_nsdfg')

    softmax_sdfg: dace_sdfg.SDFG = softmax_nsdfg.sdfg
    softmax_state: dace_state.SDFGState = softmax_sdfg.nodes()[0]

    print('encoder2_0_1.sdfg')

    # enable temporary array reuse

    assert len(softmax_sdfg.nodes()) == 1

    merge_symbols(softmax_sdfg, 'output', 'exp_arr')

    print('encoder2_1.sdfg')

    # remove view nodes

    softmax_sdfg.apply_transformations_repeated([SqueezeViewRemove],
                                                validate_all=True,
                                                print_report=True)

    print('encoder2_2.sdfg')

    # eliminate trivial map dimensions

    softmax_state.parent.apply_transformations_repeated(
        [TrivialMapElimination, TrivialMapRangeElimination],
        validate_all=True,
        print_report=True)

    print('encoder3.sdfg')

    # split last dimension out of 4 dimensional maps

    pattern = sdutil.node_path_graph(
        dace.nodes.MapEntry(dace.nodes.Map('_', [], [])))
    occurences = [(subgraph.nodes()[0], subgraph.graph)
                  for subgraph in enumerate_matches(softmax_sdfg, pattern)]
    for map_entry, state in occurences:
        if map_entry.map.range.dims() == 4:
            print("Applying MapExpansion tranformation ", state.label,
                  ". Nodes:", map_entry)
            entries = MapExpansion.apply_to(sdfg=state.parent,
                                            map_entry=map_entry)
            assert len(entries) == 4
            print("Applying MapCollapse tranformation ", state.label,
                  ". Nodes:", map_entry)
            new_entry, new_exit = MapCollapse.apply_to(
                sdfg=state.parent,
                _outer_map_entry=entries[0],
                _inner_map_entry=entries[1])
            print("Applying MapCollapse tranformation again ", state.label,
                  ". Nodes:", map_entry)
            MapCollapse.apply_to(sdfg=state.parent,
                                 _outer_map_entry=new_entry,
                                 _inner_map_entry=entries[2])

    print('encoder3_1.sdfg')

    # apply strip mining for future use as warps

    pattern = sdutil.node_path_graph(
        dace.nodes.MapEntry(dace.nodes.Map('_', [], [])))

    for subgraph in enumerate_matches(softmax_sdfg, pattern):
        map_entry: sdfg_nodes.MapEntry = subgraph.nodes()[0]
        if map_entry.map.range.dims() == 1:
            print("Applying StripMining tranformation ", subgraph.graph.label,
                  ". Nodes:", subgraph.nodes())
            rb, re, rs = map_entry.map.range[0]
            seq_len = re + 1
            assert rb == 0
            assert rs == 1
            assert seq_len % 32 == 0
            StripMining.apply_to(sdfg=subgraph.graph.parent,
                                 options={
                                     'tile_size': seq_len // 32,
                                     'tiling_type': dace.TilingType.CeilRange,
                                     'divides_evenly': True
                                 },
                                 _map_entry=map_entry)

    fwd_sdfg.validate()

    print('encoder3_2.sdfg')

    # add temp transient

    pattern = sdutil.node_path_graph(
        dace.nodes.MapExit(dace.nodes.Map('_', [], [])),
        dace.nodes.MapExit(dace.nodes.Map('_', [], [])),
        dace.nodes.MapExit(dace.nodes.Map('_', [], [])))
    occurences = [(subgraph.nodes(), subgraph.graph)
                  for subgraph in enumerate_matches(softmax_sdfg, pattern)]
    for nodes, state in occurences:
        if state.edges_between(nodes[0], nodes[1])[0].data.wcr:
            print("Applying AccumulateTransient tranformation ", state.label,
                  ". Nodes:", nodes)
            AccumulateTransient.apply_to(sdfg=state.parent,
                                         map_exit=nodes[0],
                                         outer_map_exit=nodes[1])

    print('encoder3_3.sdfg')

    # nest all maps into states

    softmax_sdfg.apply_transformations_repeated([NestMaps],
                                                validate_all=True,
                                                print_report=True)

    print('encoder4.sdfg')

    # nest access nodes into maps

    softmax_sdfg.apply_transformations_repeated(
        [NestExitAccessNode, NestEntryAccessNode, RemoveUnusedAccessNode],
        validate_all=True,
        print_report=True)

    print('encoder5.sdfg')

    softmax_sdfg.apply_transformations_repeated([NestedSDFGFusion],
                                                validate_all=True,
                                                print_report=True)

    print('encoder6.sdfg')

    softmax_sdfg.apply_transformations_repeated(
        [CleanNestedSDFGConnectors, RemoveDanglingAccessNodes, NestTransients],
        validate_all=True,
        print_report=True)

    print('encoder7.sdfg')

    # Buggy behavior of TrivialMapRangeElimination that leaves empty map that can't be removed with
    # TrivialMapElimination helps here by blocking even more serious bug in ContantPropagation later
    softmax_sdfg.apply_transformations_repeated(
        [TrivialMapRangeElimination, TrivialMapElimination],
        validate_all=True,
        print_report=True)

    print('encoder7_1.sdfg')

    softmax_sdfg.apply_transformations_repeated([NestMapContent],
                                                validate_all=True,
                                                print_report=True)

    print('encoder8.sdfg')

    softmax_sdfg.apply_transformations_repeated([NestedMapFusion],
                                                validate_all=True,
                                                print_report=True)

    print('encoder9.sdfg')

    softmax_sdfg.apply_transformations_repeated(
        [CleanNestedSDFGConnectors, RemoveDanglingAccessNodes, NestTransients],
        validate_all=True,
        print_report=True)

    print('encoder10.sdfg')

    softmax_sdfg.apply_transformations_repeated(
        [UnifyInOutNestedSDFGConnectors], validate_all=True, print_report=True)

    print('encoder11.sdfg')

    softmax_sdfg.apply_transformations_repeated(
        [WarpAllReduceDetectionNoTasklet],
        validate_all=True,
        print_report=True)

    print('encoder11_1.sdfg')

    propagate_memlets_sdfg(fwd_sdfg)

    print('encoder11_2.sdfg')

    softmax_sdfg.apply_transformations_repeated([AddNestedSDFGInputConnector],
                                                validate_all=True,
                                                print_report=True)

    print('encoder11_3.sdfg')

    softmax_sdfg.apply_transformations_repeated([RemoveReadSDFGConnectors],
                                                validate_all=True,
                                                print_report=True)

    print('encoder12.sdfg')

    softmax_sdfg.apply_transformations_repeated([NestTransients],
                                                validate_all=True,
                                                print_report=True)

    print('encoder12_1.sdfg')

    # TODO: it should be done in transformation that can detect if barrier removable or not
    pattern = sdutil.node_path_graph(Barrier)

    matches = [(subgraph.graph, subgraph.nodes())
               for subgraph in enumerate_matches(softmax_sdfg, pattern)]
    for state, nodes in matches:
        print("Match found in state", state.label, ". Nodes:", nodes)

        EmptyStateElimination.apply_to(state.parent,
                                       empty_state=state,
                                       verify=False)

    print('encoder12_2.sdfg')

    softmax_sdfg.apply_transformations_repeated([CleanNestedWrites],
                                                validate_all=True,
                                                print_report=True)

    print('encoder13.sdfg')

    softmax_sdfg.apply_transformations_repeated([RemoveUnusedStates],
                                                validate_all=True,
                                                print_report=True)

    print('encoder14.sdfg')

    softmax_sdfg.apply_transformations_repeated([PruneConnectors],
                                                validate_all=True,
                                                print_report=True)

    print('encoder14_1.sdfg')

    softmax_sdfg.apply_transformations_repeated([RemoveDanglingAccessNodes],
                                                validate_all=True,
                                                print_report=True)

    print('encoder15.sdfg')

    softmax_sdfg.apply_transformations_repeated([ConstantPropagation],
                                                validate_all=True,
                                                print_report=True)

    print('encoder15_1.sdfg')

    softmax_sdfg.apply_transformations_repeated([EmptyStateElimination],
                                                validate_all=True,
                                                print_report=True)

    print('encoder15_2.sdfg')

    softmax_sdfg.apply_transformations_repeated([NestedMapFusion],
                                                validate_all=True,
                                                print_report=True)

    print('encoder15_3.sdfg')

    softmax_sdfg.apply_transformations_repeated(
        [CleanNestedSDFGConnectors, RemoveDanglingAccessNodes],
        validate_all=True,
        print_report=True)

    print('encoder15_3_1.sdfg')

    softmax_sdfg.apply_transformations_repeated(
        [UnifyInOutNestedSDFGConnectors], validate_all=True, print_report=True)

    print('encoder15_6.sdfg')

    softmax_sdfg.apply_transformations_repeated([RemoveReadSDFGConnectors],
                                                validate_all=True,
                                                print_report=True)

    print('encoder15_6_1.sdfg')

    softmax_sdfg.apply_transformations_repeated([NestTransients],
                                                validate_all=True,
                                                print_report=True)

    print('encoder15_7.sdfg')

    softmax_sdfg.apply_transformations_repeated([CleanNestedSDFGConnectors],
                                                validate_all=True,
                                                print_report=True)

    print('encoder16.sdfg')

    # remove all barriers
    # TODO: it should be done in transformation that can detect if barrier removable or not
    pattern = sdutil.node_path_graph(Barrier)

    for subgraph in enumerate_matches(softmax_sdfg, pattern):
        print("Match found in state", subgraph.graph.label, ". Nodes:",
              subgraph.nodes())

        EmptyStateElimination.apply_to(subgraph.graph.parent,
                                       empty_state=subgraph.graph,
                                       verify=False)

    print('encoder16_2.sdfg')

    softmax_sdfg.apply_transformations_repeated([EmptyStateElimination],
                                                validate_all=True,
                                                print_report=True)

    print('encoder16_3.sdfg')

    softmax_sdfg.apply_transformations_repeated([
        NestedMapFusion, CleanNestedSDFGConnectors, RemoveDanglingAccessNodes,
        NestTransients, UnifyInOutNestedSDFGConnectors,
        RemoveReadSDFGConnectors
    ],
                                                validate_all=True,
                                                print_report=True)

    print('encoder16_4.sdfg')

    # it fails with strict_transform enabled for some reason
    softmax_sdfg.apply_transformations([GPUTransformSDFG],
                                       validate_all=True,
                                       print_report=True,
                                       options={'strict_transform': False})

    print('encoder17.sdfg')

    # GPUTransformSDFG incorrectly wraps Tasklets of NestedSDFGs deep in the nesting hierarchy with empty maps
    # it is easier to fix it here by applying TrivialMapElimination
    softmax_sdfg.apply_transformations_repeated([TrivialMapElimination],
                                                validate_all=True,
                                                print_report=True)

    softmax_sdfg.save('encoder18.sdfg')
    print('encoder18.sdfg')

    pattern = sdutil.node_path_graph(dace.nodes.MapEntry(dace.nodes.Map('_', [], [])))

    vector_size = 4

    for subgraph in enumerate_matches(softmax_sdfg, pattern):
        map_entry: sdfg_nodes.MapEntry = subgraph.nodes()[0]

        if map_entry.map.range.size() != [seq_len // 32]:
            continue  # vectorization dimension should have specific length

        if map_entry.map.range.min_element() == [0]:
            continue  # min element should depend on other parametric value, it can't be zero

        print("Applying StripMining tranformation ", subgraph.graph.label, ". Nodes:", subgraph.nodes())
        StripMining.apply_to(sdfg=subgraph.graph.parent,
                             options={'tile_size': vector_size,
                                      'divides_evenly': True},
                             _map_entry=map_entry)

    softmax_sdfg.save('encoder18_1.sdfg')
    print('encoder18_1.sdfg')

    softmax_sdfg.apply_transformations_repeated([NestMapContent], validate_all=True, print_report=True)

    softmax_sdfg.save('encoder18_2.sdfg')
    print('encoder18_2.sdfg')

    softmax_sdfg.apply_transformations_repeated([VectorizeSDFG], validate_all=True, validate=True, print_report=True)

    softmax_sdfg.save('encoder18_3.sdfg')
    print('encoder18_3.sdfg')

    # TODO: it should be done in transformation that can detect if barrier removable or not
    pattern = sdutil.node_path_graph(Barrier)

    matches = [(subgraph.graph, subgraph.nodes())
               for subgraph in enumerate_matches(softmax_sdfg, pattern)]
    for state, nodes in matches:
        print("Match found in state", state.label, ". Nodes:", nodes)

        EmptyStateElimination.apply_to(state.parent,
                                       empty_state=state,
                                       verify=False)

    softmax_sdfg.expand_library_nodes()

    softmax_sdfg.save('encoder_last.sdfg')
    print('encoder_last.sdfg')


@pytest.mark.pure
def test_bert(sdfg_name, gpu):
    batch_size = 2
    seq_len = 512
    hidden_size = 768

    class BertTokenSoftmaxClf(nn.Module):
        def __init__(self):
            super(BertTokenSoftmaxClf, self).__init__()
            self.bert = BertLayer(BertConfig(hidden_act="relu")).eval()
            self.sm = nn.LogSoftmax(dim=-1)

        def forward(self, x):
            embs = self.bert(x)[0]
            return self.sm(embs.sum(dim=-1))

    model = BertTokenSoftmaxClf()
    dace_model = DaceModule(copy.deepcopy(model),
                            backward=True,
                            sdfg_name=sdfg_name,
                            cuda=gpu)

    if gpu:
        dace_model.append_post_autodiff_hook('apply_softmax_transformations',
                                             apply_softmax_transformations)

    # check forward pass using loss
    input = torch.randn([batch_size, seq_len, hidden_size])
    labels = torch.tensor([0, 123], dtype=torch.long)

    training_step(dace_model, model, (input, labels), sdfg_name, gpu)
