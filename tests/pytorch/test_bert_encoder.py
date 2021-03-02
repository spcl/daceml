import pytest
import numpy as np
import torch
from dace.transformation.dataflow import RedundantSecondArray
from transformers import BertConfig, BertLayer
import dace
from dace.sdfg import sdfg as dace_sdfg

from dace.sdfg import state as dace_state
from daceml.pytorch import DaceModule
from daceml.transformation import ConstantFolding
from dace import dtypes
from dace.sdfg import utils as sdutil


from dace.sdfg import nodes as sdfg_nodes

@pytest.mark.parametrize("apply_strict", [True, False])
def test_bert_encoder(gpu, apply_strict):
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

    dace_model = DaceModule(ptmodel, cuda=gpu, train=False)

    #dace_outputs0 = dace_model(input.clone())

    # Transformed version

    # dace_model.dace_model.sdfg.apply_transformations_repeated(
    #     [ConstantFolding, RedundantSecondArray], validate_all=True)

    dace_model.dace_model = dace_model.initialize_sdfg(input.clone())

    dace_model.sdfg.save('attn1.sdfg')
    print('attn1.sdfg')

    dace_model.sdfg.expand_library_nodes()

    dace_model.sdfg.save('attn2.sdfg')
    print('attn2.sdfg')

    # find softmax sdfg and state
    from dace.transformation.pattern_matching import enumerate_matches
    from dace.sdfg import utils as sdutil

    pattern = sdutil.node_path_graph(dace.nodes.MapExit, dace.nodes.AccessNode, dace.nodes.MapEntry)

    for subgraph in enumerate_matches(dace_model.sdfg, pattern):
        softmax_state: dace_state.SDFGState = subgraph.graph

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

    # # fuse maps in softmax
    #
    # from dace.transformation.pattern_matching import enumerate_matches
    # from dace.sdfg import utils as sdutil
    #
    # pattern = sdutil.node_path_graph(dace.nodes.MapExit, dace.nodes.AccessNode, dace.nodes.MapEntry)
    #
    # from dace.transformation.subgraph import SubgraphFusion
    #
    # for subgraph in enumerate_matches(dace_model.sdfg, pattern):
    #     print("Match found in state", subgraph.graph.label, ". Nodes:", subgraph.nodes())
    #
    #     softmax_state: dace_state.SDFGState = subgraph.graph
    #
    #     SubgraphFusion.apply_to(softmax_state.parent, softmax_state.nodes())
    #
    # softmax_sdfg: dace_sdfg.SDFG = softmax_state.parent
    #
    # dace_model.sdfg.save('attn3.sdfg')
    # print('attn3.sdfg')
    #
    # # # move everything on GPU
    # # from dace.transformation.interstate import GPUTransformSDFG
    # #
    # # softmax_sdfg: dace_sdfg.SDFG = softmax_state.parent
    # # softmax_state.parent.apply_transformations([GPUTransformSDFG], options={'sequential_innermaps': False})
    # #
    # # softmax_state = softmax_sdfg.nodes()[0] # old state is removed
    # #
    # # dace_model.sdfg.save('attn3_1.sdfg')
    #
    # # remove trivial dimensions from maps
    #
    # from dace.transformation.dataflow.trivial_map_elimination import TrivialMapElimination
    # from dace.transformation.dataflow.trivial_map_range_elimination import TrivialMapRangeElimination
    #
    # softmax_state.parent.apply_transformations_repeated(TrivialMapElimination)
    # softmax_state.parent.apply_transformations_repeated(TrivialMapRangeElimination)
    #
    # dace_model.sdfg.save('attn4.sdfg')
    # print('attn4.sdfg')
    #
    # # nest state to enable fusion in future
    #
    # pattern = sdutil.node_path_graph(dace.nodes.MapEntry(dace.nodes.Map('_', [], [])),
    #                                  dace.nodes.MapEntry(dace.nodes.Map('_', [], [])),
    #                                  dace.nodes.Tasklet,
    #                                  dace.nodes.MapExit(dace.nodes.Map('_', [], [])),
    #                                  dace.nodes.AccessNode("_"),
    #                                  dace.nodes.NestedSDFG('_', dace.sdfg.SDFG('_'), {}, {}))
    #
    #
    # nested_sdfg1 = None
    # access_node = None
    # nested_sdfg2 = None
    #
    # for subgraph in enumerate_matches(softmax_state.parent, pattern):
    #     entry = subgraph.nodes()[1]
    #     tasklet = subgraph.nodes()[2]
    #     exit = subgraph.nodes()[3]
    #
    #     from dace.transformation.helpers import nest_state_subgraph
    #     from dace.sdfg.graph import SubgraphView
    #
    #     access_node = subgraph.nodes()[4]
    #
    #     nested_sdfg1 = nest_state_subgraph(sdfg=softmax_state.parent, state=softmax_state,
    #                         subgraph=SubgraphView(softmax_state, [entry, tasklet, exit]))
    #
    #
    #     nested_sdfg2 = subgraph.nodes()[5]
    #
    #     continue
    #
    # dace_model.sdfg.save('attn4_1.sdfg')
    # print('attn4_1.sdfg')
    #
    # from dace.transformation.dataflow.nested_sdfg_fusion import NestedSDFGFusion
    #
    # NestedSDFGFusion.apply_to(sdfg=softmax_state.parent,
    #                           _nested_sdfg1=nested_sdfg1,
    #                           _access_node=access_node,
    #                           _nested_sdfg2=nested_sdfg2)
    #
    # dace_model.sdfg.save('attn4_2.sdfg')
    # print('attn4_2.sdfg')
    #
    # from dace.transformation.interstate.state_reordering import StateReordering
    #
    # pattern = sdutil.node_path_graph(dace.sdfg.SDFGState(), dace.sdfg.SDFGState(), dace.sdfg.SDFGState())
    #
    # for subgraph in enumerate_matches(softmax_state.parent, pattern):
    #     print("Match found in state", subgraph.graph.label, ". Nodes:", subgraph.nodes())
    #
    #     StateReordering.apply_to(sdfg=subgraph.graph,
    #                              _first_state=subgraph.nodes()[0],
    #                              _second_state=subgraph.nodes()[1])
    #
    # dace_model.sdfg.save('attn4_3.sdfg')
    # print('attn4_3.sdfg')
    #
    # from dace.transformation.interstate.state_fusion import StateFusion
    #
    # pattern = sdutil.node_path_graph(dace.sdfg.SDFGState(), dace.sdfg.SDFGState(), dace.sdfg.SDFGState())
    #
    # for subgraph in enumerate_matches(softmax_state.parent, pattern):
    #     print("Match found in state", subgraph.graph.label, ". Nodes:", subgraph.nodes())
    #
    #     StateFusion.apply_to(sdfg=subgraph.graph,
    #                          first_state=subgraph.nodes()[1],
    #                          second_state=subgraph.nodes()[2])
    #
    # dace_model.sdfg.save('attn4_4.sdfg')
    # print('attn4_4.sdfg')
    #
    # pattern = sdutil.node_path_graph(dace.nodes.MapEntry(dace.nodes.Map('_', [], [])),
    #                                  dace.nodes.Tasklet('_'),
    #                                  dace.nodes.MapExit(dace.nodes.Map('_', [], [])),
    #                                  dace.nodes.AccessNode('_'),
    #                                  dace.nodes.MapEntry(dace.nodes.Map('_', [], [])),
    #                                  dace.nodes.Tasklet('_'),
    #                                  dace.nodes.MapExit(dace.nodes.Map('_', [], [])))
    #
    # for subgraph in enumerate_matches(softmax_state.parent, pattern):
    #     print("Match found in state", subgraph.graph.label, ". Nodes:", subgraph.nodes())
    #
    #     # I have to make names of both maps the same, otherwise SubgraphFusion doesn't work.
    #     subgraph.graph.replace(subgraph.nodes()[4].map.params[0], subgraph.nodes()[0].map.params[0])
    #     subgraph.nodes()[4].map.params[0] = subgraph.nodes()[0].map.params[0]
    #
    #     dace_model.sdfg.save('attn4_5.sdfg')
    #     print('attn4_5.sdfg')
    #
    #     # I have no idea why, but only with GPUTransformSDFG schedule of the first map is Sequential
    #     # but the schedule of the second map is Default.
    #     # It prevents their fusion, so I need to change schedule of the first map to Default
    #
    #     map: sdfg_nodes.Map = subgraph.nodes()[0].map
    #     map.schedule = dtypes.ScheduleType.Default
    #
    #     SubgraphFusion.apply_to(subgraph.graph.parent, subgraph)
    #
    # dace_model.sdfg.save('attn4_6.sdfg')
    # print('attn4_6.sdfg')
    #
    # # tile map inside reduction into two maps (one over warp, other is sequential)
    #
    # from dace.transformation.dataflow.tiling import MapTiling
    #
    # pattern = sdutil.node_path_graph(dace.nodes.MapExit)
    #
    # for subgraph in enumerate_matches(softmax_state.parent, pattern):
    #     map_exit: dace.nodes.MapExit = subgraph.nodes()[0]
    #     state: dace.sdfg.SDFGState = subgraph.graph
    #     sdfg: dace.sdfg.SDFG = state.parent
    #
    #     edges = state.out_edges(map_exit)
    #
    #     map_entry: dace.nodes.MapEntry = state.entry_node(map_exit)
    #
    #     if len(edges) == 1 and edges[0].data.wcr:
    #         print("Match found in state", subgraph.graph.label, ". Nodes:", subgraph.nodes())
    #
    #         MapTiling.apply_to(sdfg=state.parent,
    #                            options={'tile_sizes': (seq_len//32,)},
    #                            _map_entry=map_entry)
    #
    # dace_model.sdfg.save('attn5.sdfg')
    # print('attn5.sdfg')
    #
    # # create transients between two map exits
    # from dace.transformation.dataflow.stream_transient import AccumulateTransient
    #
    # pattern = sdutil.node_path_graph(dace.nodes.Tasklet,
    #                                  dace.nodes.MapExit(dace.nodes.Map("", [], [])),
    #                                  dace.nodes.MapExit(dace.nodes.Map("", [], [])))
    #
    # for subgraph_view in enumerate_matches(softmax_state.parent, pattern):
    #     tasklet = subgraph_view.nodes()[0]
    #     map_exit1 = subgraph_view.nodes()[1]
    #     map_exit2 = subgraph_view.nodes()[2]
    #
    #     edges = list(subgraph_view.edges_between(map_exit1, map_exit2))
    #
    #     if len(edges) != 1 or edges[0].data.wcr is None:
    #         continue
    #
    #     wcr = edges[0].data.wcr
    #
    #     from dace.frontend.operations import detect_reduction_type
    #
    #     reduction_type = detect_reduction_type(wcr)
    #
    #     if reduction_type == dace.dtypes.ReductionType.Max:
    #         identity_value = -1e9
    #     elif reduction_type == dace.dtypes.ReductionType.Sum:
    #         identity_value = 0
    #     else:
    #         raise Exception("Unknown reduction type")
    #
    #     AccumulateTransient.apply_to(subgraph_view.graph.parent, {'identity': str(identity_value)},
    #                                  _tasklet=tasklet, _map_exit=map_exit1, _outer_map_exit=map_exit2)
    #
    # dace_model.sdfg.save('attn6.sdfg')
    # print('attn6.sdfg')
    #
    # # from dace.transformation.dataflow.wcr_extraction import WCRExtraction
    # #
    # # pattern = sdutil.node_path_graph(dace.nodes.MapExit(dace.nodes.Map("", [], [])),
    # #                                  dace.nodes.AccessNode("_"))
    # #
    # # for subgraph_view in enumerate_matches(softmax_state.parent, pattern):
    # #
    # #     access_node: sdfg_nodes.AccessNode = subgraph_view.nodes()[1]
    # #     if access_node.label in {'_out', 'n2__out'}:
    # #         print("Match found in state", subgraph_view.graph.label, ". Nodes:", subgraph_view.nodes())
    # #
    # #         WCRExtraction.apply_to(sdfg=subgraph_view.graph.parent,
    # #                                _map_exit=subgraph_view.nodes()[0],
    # #                                _output_node=subgraph_view.nodes()[1])
    # #
    # # dace_model.sdfg.save('attn7.sdfg')
    #
    # # nest map at the end into nested sdfg
    #
    # pattern = sdutil.node_path_graph(dace.nodes.MapEntry(dace.nodes.Map('_', [], [])),
    #                                  dace.nodes.Tasklet,
    #                                  dace.nodes.MapExit(dace.nodes.Map('_', [], [])),
    #                                  dace.nodes.MapExit(dace.nodes.Map('_', [], [])),
    #                                  dace.nodes.AccessNode("_"))
    #
    # for subgraph in enumerate_matches(softmax_state.parent, pattern):
    #     entry = subgraph.nodes()[0]
    #     tasklet = subgraph.nodes()[1]
    #     exit = subgraph.nodes()[2]
    #
    #     from dace.transformation.helpers import nest_state_subgraph
    #     from dace.sdfg.graph import SubgraphView
    #
    #     nested_sdfg1 = nest_state_subgraph(sdfg=softmax_state.parent, state=softmax_state,
    #                                        subgraph=SubgraphView(softmax_state, [entry, tasklet, exit]))
    #
    #
    #
    # dace_model.sdfg.save('attn7_1.sdfg')
    # print('attn7_1.sdfg')
    #
    # # fuse nested sdfgs
    #
    # pattern = sdutil.node_path_graph(dace.nodes.NestedSDFG('_', dace.sdfg.SDFG('_'), {}, {}),
    #                                  dace.nodes.AccessNode("_"),
    #                                  dace.nodes.NestedSDFG('_', dace.sdfg.SDFG('_'), {}, {}))
    #
    # #softmax_sdfg.apply_transformations_repeated(NestedSDFGFusion) # doesn't work for some reason :(
    #
    # while True:
    #     all_matches = list(enumerate_matches(softmax_sdfg, pattern))
    #     if not all_matches:
    #         break
    #
    #     subgraph = all_matches[0]
    #     NestedSDFGFusion.apply_to(sdfg=subgraph.graph.parent,
    #                             _nested_sdfg1=subgraph.nodes()[0],
    #                             _access_node=subgraph.nodes()[1],
    #                             _nested_sdfg2=subgraph.nodes()[2])
    #
    # # for subgraph in enumerate_matches(softmax_sdfg, pattern):
    # #     NestedSDFGFusion.apply_to(sdfg=subgraph.graph.parent,
    # #                               nested_sdfg1=subgraph.nodes()[0],
    # #                               access_node=subgraph.nodes()[1],
    # #                               nested_sdfg2=subgraph.nodes()[2])
    #
    # dace_model.sdfg.save('attn8.sdfg')
    # print('attn8.sdfg')
    #
    # pattern = sdutil.node_path_graph(dace.sdfg.SDFGState(),
    #                                  dace.sdfg.SDFGState(),
    #                                  dace.sdfg.SDFGState(),
    #                                  dace.sdfg.SDFGState(),
    #                                  dace.sdfg.SDFGState())
    #
    # for subgraph in enumerate_matches(softmax_state.parent, pattern):
    #     print("Match found in state", subgraph.graph.label, ". Nodes:", subgraph.nodes())
    #
    #     StateReordering.apply_to(sdfg=subgraph.graph,
    #                              _first_state=subgraph.nodes()[1],
    #                              _second_state=subgraph.nodes()[2])
    #
    # dace_model.sdfg.save('attn9.sdfg')
    # print('attn9.sdfg')
    #
    # pattern = sdutil.node_path_graph(dace.sdfg.SDFGState(),
    #                                  dace.sdfg.SDFGState(),
    #                                  dace.sdfg.SDFGState(),
    #                                  dace.sdfg.SDFGState(),
    #                                  dace.sdfg.SDFGState())
    #
    # for subgraph in enumerate_matches(softmax_state.parent, pattern):
    #     print("Match found in state", subgraph.graph.label, ". Nodes:", subgraph.nodes())
    #
    #     StateFusion.apply_to(sdfg=subgraph.graph,
    #                          first_state=subgraph.nodes()[2],
    #                          second_state=subgraph.nodes()[3])
    #
    # dace_model.sdfg.save('attn10.sdfg')
    # print('attn10.sdfg')
    #
    # pattern = sdutil.node_path_graph(dace.sdfg.SDFGState(),
    #                                  dace.sdfg.SDFGState(),
    #                                  dace.sdfg.SDFGState(),
    #                                  dace.sdfg.SDFGState())
    #
    # for subgraph in enumerate_matches(softmax_state.parent, pattern):
    #     print("Match found in state", subgraph.graph.label, ". Nodes:", subgraph.nodes())
    #
    #     StateFusion.apply_to(sdfg=subgraph.graph,
    #                          first_state=subgraph.nodes()[2],
    #                          second_state=subgraph.nodes()[3])
    #
    # dace_model.sdfg.save('attn11.sdfg')
    # print('attn11.sdfg')
    #
    # from dace.transformation.dataflow.nest_access_nodes import NestExitAccessNode
    # from dace.transformation.dataflow.nest_access_nodes import NestEntryAccessNode
    # from dace.transformation.dataflow.nest_access_nodes import RemoveUnusedAccessNode
    #
    # softmax_state.parent.apply_transformations_repeated([
    #     NestExitAccessNode, NestEntryAccessNode, RemoveUnusedAccessNode], validate_all=True)
    #
    # dace_model.sdfg.save('attn12.sdfg')
    # print('attn12.sdfg')
    #
    # # from dace.transformation.dataflow.add_explicit_barrier import AddExplicitBarrier
    # #
    # # dace_model.dace_model.sdfg.apply_transformations_repeated([
    # #     AddExplicitBarrier], validate_all=True)
    #
    # # make parameter names of 3 sequential maps the same
    # pattern = sdutil.node_path_graph(dace.nodes.MapExit(dace.nodes.Map('_', [], [])),
    #                                  dace.nodes.MapEntry(dace.nodes.Map('_', [], [])))
    #
    # for subgraph in enumerate_matches(softmax_state.parent, pattern):
    #     exit = subgraph.nodes()[0]
    #     entry = subgraph.nodes()[1]
    #
    #     subgraph.graph.replace(entry.map.params[0], exit.map.params[0])
    #     entry.map.params[0] = exit.map.params[0]
    #
    # dace_model.sdfg.save('attn13.sdfg')
    # print('attn13.sdfg')
    #
    # from dace.transformation.dataflow.nested_map_fusion import NestedMapFusion
    #
    # dace_model.dace_model.sdfg.apply_transformations_repeated([NestedMapFusion], validate_all=True)
    #
    # dace_model.sdfg.save('attn14.sdfg')
    # print('attn14.sdfg')

    # from dace.transformation.dataflow.warp_all_reduce_detection import WarpAllReduceDetection
    #
    # pattern = sdutil.node_path_graph(sdfg_nodes.MapEntry(sdfg_nodes.Map("", [], [])),
    #                                  sdfg_nodes.AccessNode("_"),
    #                                  sdfg_nodes.MapExit(sdfg_nodes.Map("", [], [])),
    #                                  sdfg_nodes.AccessNode("_"))
    #
    # for subgraph_view in enumerate_matches(softmax_state.parent, pattern):
    #     print("!! Match found in state", subgraph_view.graph.label, ". Nodes:", subgraph_view.nodes())
    #
    #     WarpAllReduceDetection.apply_to(sdfg=subgraph_view.graph.parent,
    #                                     map_entry=subgraph_view.nodes()[0],
    #                                     temp_node=subgraph_view.nodes()[1],
    #                                     map_exit=subgraph_view.nodes()[2],
    #                                     output_node=subgraph_view.nodes()[3])
    #
    # dace_model.sdfg.save('attn8.sdfg')

    dace_outputs1 = dace_model(input.clone())

    #diff = np.abs(dace_outputs0 - pt_outputs[0].detach().numpy())
    diff = np.abs(dace_outputs1 - pt_outputs[0].detach().numpy())

    assert np.max(diff) < 1e-6
    #assert np.allclose(dace_outputs1, dace_outputs0,)

    print("testing passed")


if __name__ == "__main__":
    test_bert_encoder(False, False)
