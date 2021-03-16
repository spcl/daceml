import pytest
import numpy as np
import torch
from dace.transformation.dataflow import RedundantSecondArray
from transformers import BertConfig, BertLayer
import dace
from dace.sdfg import sdfg as dace_sdfg

import daceml.onnx as donnx
from dace.sdfg import state as dace_state
from daceml.pytorch import DaceModule
from daceml.transformation import ConstantFolding
from dace import dtypes
from dace.sdfg import utils as sdutil
from dace.sdfg import nodes as sdfg_nodes


def test_bert_encoder(gpu, default_implementation, sdfg_name):
    if not gpu and default_implementation == 'onnxruntime':
        pytest.skip("combination is tested below")

    batch_size = 8
    seq_len = 512
    hidden_size = 768

    input = torch.randn([batch_size, seq_len, hidden_size])

    ptmodel = BertLayer(BertConfig()).eval()
    pt_outputs = ptmodel(input.clone())

    dace_model = DaceModule(ptmodel,
                            cuda=gpu,
                            train=False,
                            sdfg_name=sdfg_name)
    dace_outputs0 = dace_model(input.clone())

    diff = np.abs(dace_outputs0.detach().numpy() -
                  pt_outputs[0].detach().numpy())

    assert np.max(diff) < 1e-5


@pytest.mark.ort
def test_bert_cf(sdfg_name):
    batch_size = 8
    seq_len = 512
    hidden_size = 768

    input = torch.randn([batch_size, seq_len, hidden_size])

    ptmodel = BertLayer(BertConfig()).eval()
    pt_outputs = ptmodel(input.clone())

    dace_model = DaceModule(ptmodel, train=False, sdfg_name=sdfg_name)
    dace_outputs0 = dace_model(input.clone())

    dace_model.dace_model.sdfg.apply_transformations_repeated(
        [ConstantFolding, RedundantSecondArray],
        validate_all=True,
        strict=True)
    dace_model.dace_model.sdfg.expand_library_nodes()
    dace_model.dace_model.sdfg.apply_strict_transformations()

    dace_outputs1 = dace_model(input.clone())

    diff = np.abs(dace_outputs0.detach().numpy() -
                  pt_outputs[0].detach().numpy())

    assert np.max(diff) < 1e-5
    assert np.allclose(dace_outputs1, dace_outputs0)


def test_bert_encoder_transformations():
    default_impl = donnx.default_implementation
    donnx.default_implementation = "pure"

    batch_size = 8
    seq_len = 64
    hidden_size = 16
    num_hidden_layers = 8
    num_attention_heads = 8
    intermediate_size = 128


    input = torch.randn([batch_size, seq_len, hidden_size])

    ptmodel = BertLayer(BertConfig(hidden_size=hidden_size, num_hidden_layers=num_hidden_layers, max_position_embeddings=seq_len,
                                   num_attention_heads=num_attention_heads, intermediate_size=intermediate_size)).eval()
    pt_outputs = ptmodel(input.clone())

    dace_model = DaceModule(ptmodel, dummy_inputs=input.clone(), cuda=False, train=False)

    # Transformed version

    dace_model.sdfg.save('attn1.sdfg')
    print('attn1.sdfg')

    dace_model.sdfg.expand_library_nodes()

    dace_model.sdfg.save('attn2.sdfg')
    print('attn2.sdfg')

    # find softmax sdfg and state
    from dace.transformation.pattern_matching import enumerate_matches
    from dace.sdfg import utils as sdutil

    pattern = sdutil.node_path_graph(dace.nodes.MapExit, dace.nodes.AccessNode, dace.nodes.MapEntry)

    subgraphs = list(enumerate_matches(dace_model.sdfg, pattern))
    assert(len(subgraphs) == 2) # there should be two matches
    assert(subgraphs[0].graph == subgraphs[1].graph) # both matches should be inside the softmax
    softmax_state: dace_state.SDFGState = subgraphs[0].graph

    softmax_sdfg: dace_sdfg.SDFG = softmax_state.parent

    # eliminate trivial map dimensions
    from dace.transformation.dataflow.trivial_map_elimination import TrivialMapElimination
    from dace.transformation.dataflow.trivial_map_range_elimination import TrivialMapRangeElimination

    softmax_state.parent.apply_transformations_repeated([TrivialMapElimination, TrivialMapRangeElimination],
                                                        validate_all=True, print_report=True)

    dace_model.sdfg.save('attn3.sdfg')
    print('attn3.sdfg')

    # apply strip mining for future use as warps

    from dace.transformation.dataflow.strip_mining import StripMining

    pattern = sdutil.node_path_graph(dace.nodes.MapEntry(dace.nodes.Map('_', [], [])))

    for subgraph in enumerate_matches(softmax_sdfg, pattern):
        map_entry: sdfg_nodes.MapEntry = subgraph.nodes()[0]
        if map_entry.map.range.dims() == 1:
            print("Applying StripMining tranformation ", subgraph.graph.label, ". Nodes:", subgraph.nodes())
            StripMining.apply_to(sdfg=subgraph.graph.parent,
                                 options={'tile_size': seq_len // 32,
                                          'tiling_type': 'ceilrange',
                                          'divides_evenly': True},
                                 _map_entry=map_entry)

    dace_model.sdfg.validate()

    dace_model.sdfg.save('attn3_1.sdfg')
    print('attn3_1.sdfg')

    # add temp transient
    from dace.transformation.dataflow.stream_transient import AccumulateTransient

    softmax_sdfg.apply_transformations_repeated([AccumulateTransient], validate_all=True, print_report=True)

    dace_model.sdfg.save('attn3_2.sdfg')
    print('attn3_2.sdfg')

    # nest all maps into states

    from dace.transformation.dataflow.nest_maps import NestMaps

    softmax_sdfg.apply_transformations_repeated([NestMaps], validate_all=True, print_report=True)

    dace_model.sdfg.save('attn4.sdfg')
    print('attn4.sdfg')

    # nest access nodes into maps

    from dace.transformation.dataflow.nest_access_nodes import NestExitAccessNode
    from dace.transformation.dataflow.nest_access_nodes import NestEntryAccessNode
    from dace.transformation.dataflow.nest_access_nodes import RemoveUnusedAccessNode

    softmax_sdfg.apply_transformations_repeated([
        NestExitAccessNode, NestEntryAccessNode, RemoveUnusedAccessNode], validate_all=True, print_report=True)

    dace_model.sdfg.save('attn5.sdfg')
    print('attn5.sdfg')

    from dace.transformation.dataflow.nested_sdfg_fusion import NestedSDFGFusion

    softmax_sdfg.apply_transformations_repeated([NestedSDFGFusion], validate_all=True, print_report=True)

    dace_model.sdfg.save('attn6.sdfg')
    print('attn6.sdfg')

    from dace.transformation.dataflow.clean_connectors import CleanNestedSDFGConnectors, RemoveDanglingAccessNodes, NestTransients

    softmax_sdfg.apply_transformations_repeated(
        [CleanNestedSDFGConnectors, RemoveDanglingAccessNodes, NestTransients], validate_all=True, print_report=True)

    dace_model.sdfg.save('attn7.sdfg')
    print('attn7.sdfg')

    softmax_sdfg.apply_transformations_repeated([TrivialMapRangeElimination, TrivialMapElimination], validate_all=True, print_report=True)

    dace_model.sdfg.save('attn7_1.sdfg')
    print('attn7_1.sdfg')

    from dace.transformation.dataflow.nest_maps import NestMapContent

    softmax_sdfg.apply_transformations_repeated([NestMapContent], validate_all=True, print_report=True)

    dace_model.sdfg.save('attn8.sdfg')
    print('attn8.sdfg')

    from dace.transformation.interstate.nested_map_fusion import NestedMapFusion

    softmax_sdfg.apply_transformations_repeated([NestedMapFusion], validate_all=True, print_report=True)

    dace_model.sdfg.save('attn9.sdfg')
    print('attn9.sdfg')

    softmax_sdfg.apply_transformations_repeated(
        [CleanNestedSDFGConnectors, RemoveDanglingAccessNodes, NestTransients], validate_all=True, print_report=True)

    dace_model.sdfg.save('attn10.sdfg')
    print('attn10.sdfg')

    from dace.transformation.dataflow.clean_connectors import UnifyInOutNestedSDFGConnectors

    softmax_sdfg.apply_transformations_repeated([UnifyInOutNestedSDFGConnectors], validate_all=True, print_report=True)

    dace_model.sdfg.save('attn11.sdfg')
    print('attn11.sdfg')

    from dace.transformation.interstate.warp_all_reduce_detection import WarpAllReduceDetectionNoTasklet

    softmax_sdfg.apply_transformations_repeated([WarpAllReduceDetectionNoTasklet], validate_all=True, print_report=True)

    dace_model.sdfg.save('attn12.sdfg')
    print('attn12.sdfg')

    from dace.transformation.dataflow.clean_connectors import CleanNestedWrites

    softmax_sdfg.apply_transformations_repeated([CleanNestedWrites], validate_all=True, print_report=True)

    dace_model.sdfg.save('attn13.sdfg')
    print('attn13.sdfg')

    from dace.transformation.interstate.remove_unused_states import RemoveUnusedStates

    softmax_sdfg.apply_transformations_repeated([RemoveUnusedStates], validate_all=True, print_report=True)

    dace_model.sdfg.save('attn14.sdfg')
    print('attn14.sdfg')

    from dace.transformation.dataflow import PruneConnectors

    softmax_sdfg.apply_transformations_repeated(
        [PruneConnectors], validate_all=True, print_report=True)

    dace_model.sdfg.save('attn14_1.sdfg')
    print('attn14_1.sdfg')

    softmax_sdfg.apply_transformations_repeated(
        [RemoveDanglingAccessNodes], validate_all=True, print_report=True)

    dace_model.sdfg.save('attn15.sdfg')
    print('attn15.sdfg')

    from dace.transformation.dataflow.constant_propagation import ConstantPropagation

    softmax_sdfg.apply_transformations_repeated([ConstantPropagation], validate_all=True, print_report=True)

    dace_model.sdfg.save('attn15_1.sdfg')
    print('attn15_1.sdfg')

    softmax_sdfg.apply_transformations_repeated([CleanNestedSDFGConnectors], validate_all=True, print_report=True)

    dace_model.sdfg.save('attn16.sdfg')
    print('attn16.sdfg')

    assert len(softmax_sdfg.nodes()) == 1
    state_with_nsdfg: dace_state.SDFGState = softmax_sdfg.nodes()[0]

    from dace.transformation.dataflow.clean_connectors import merge_symbols

    for n in state_with_nsdfg.nodes():
        if isinstance(n, sdfg_nodes.NestedSDFG):
            target_sdfg = n.sdfg

            # TODO: for this we need transformation that detects opportunities for memory reuse
            # TODO: try TransientReuse
            merge_symbols(target_sdfg, 'n2_output', 'n1_tmp_out')

    dace_model.sdfg.save('attn16_1.sdfg')
    print('attn16_1.sdfg')

    from dace.transformation.interstate.state_elimination import EmptyStateElimination
    from dace.libraries.standard.nodes.barrier import Barrier

    # remove all barriers
    # TODO: it should be done in transformation that can detect if barrier removable or not
    pattern = sdutil.node_path_graph(Barrier)

    for subgraph in enumerate_matches(softmax_sdfg, pattern):
        print("Match found in state", subgraph.graph.label, ". Nodes:", subgraph.nodes())

        EmptyStateElimination.apply_to(subgraph.graph.parent, empty_state=subgraph.graph, verify=False)

    dace_model.sdfg.save('attn16_2.sdfg')
    print('attn16_2.sdfg')


    softmax_sdfg.apply_transformations_repeated([EmptyStateElimination], validate_all=True, print_report=True)

    dace_model.sdfg.save('attn16_3.sdfg')
    print('attn16_3.sdfg')

    from dace.transformation.interstate.gpu_transform_sdfg import GPUTransformSDFG

    # it fails with strict_transform enabled for some reason
    softmax_sdfg.apply_transformations([GPUTransformSDFG], validate_all=True, print_report=True, options={'strict_transform': False})

    dace_model.sdfg.save('attn17.sdfg')
    print('attn17.sdfg')

    softmax_sdfg.expand_library_nodes()

    dace_model.sdfg.save('attn_last.sdfg')
    print('attn_last.sdfg')

    dace_outputs1 = dace_model(input.clone())

    diff = np.abs(dace_outputs1 - pt_outputs[0].detach().numpy())

    assert np.max(diff) < 1e-6

    donnx.default_implementation = default_impl


if __name__ == "__main__":
    test_bert_encoder_transformations()
