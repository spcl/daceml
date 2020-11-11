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

    dace_model.sdfg.expand_library_nodes()

    dace_model.sdfg.save('attn2.sdfg')

    # fuse maps in softmax

    from dace.transformation.pattern_matching import enumerate_matches
    from dace.sdfg import utils as sdutil

    pattern = sdutil.node_path_graph(dace.nodes.MapExit, dace.nodes.AccessNode, dace.nodes.MapEntry)

    from dace.transformation.subgraph import SubgraphFusion

    for subgraph in enumerate_matches(dace_model.sdfg, pattern):
        print("Match found in state", subgraph.graph.label, ". Nodes:", subgraph.nodes())

        softmax_state: dace_state.SDFGState = subgraph.graph

        SubgraphFusion.apply_to(softmax_state.parent, softmax_state.nodes())

    dace_model.sdfg.save('attn3.sdfg')

    # # move everything on GPU
    # from dace.transformation.interstate import GPUTransformSDFG
    #
    # softmax_sdfg: dace_sdfg.SDFG = softmax_state.parent
    # softmax_state.parent.apply_transformations([GPUTransformSDFG], options={'sequential_innermaps': False})
    #
    # softmax_state = softmax_sdfg.nodes()[0] # old state is removed
    #
    # dace_model.sdfg.save('attn3_1.sdfg')

    # remove trivial dimensions from maps

    from dace.transformation.dataflow.trivial_map_elimination import TrivialMapElimination
    from dace.transformation.dataflow.trivial_map_range_elimination import TrivialMapRangeElimination

    softmax_state.parent.apply_transformations_repeated(TrivialMapElimination)
    softmax_state.parent.apply_transformations_repeated(TrivialMapRangeElimination)

    dace_model.sdfg.save('attn4.sdfg')

    # nest state to enable fusion in future

    pattern = sdutil.node_path_graph(dace.nodes.MapEntry(dace.nodes.Map('_', [], [])),
                                     dace.nodes.MapEntry(dace.nodes.Map('_', [], [])),
                                     dace.nodes.Tasklet,
                                     dace.nodes.MapExit(dace.nodes.Map('_', [], [])),
                                     dace.nodes.AccessNode("_"),
                                     dace.nodes.NestedSDFG('_', dace.sdfg.SDFG('_'), {}, {}))


    nested_sdfg1 = None
    access_node = None
    nested_sdfg2 = None

    for subgraph in enumerate_matches(softmax_state.parent, pattern):
        entry = subgraph.nodes()[1]
        tasklet = subgraph.nodes()[2]
        exit = subgraph.nodes()[3]

        from dace.transformation.helpers import nest_state_subgraph
        from dace.sdfg.graph import SubgraphView

        access_node = subgraph.nodes()[4]

        nested_sdfg1 = nest_state_subgraph(sdfg=softmax_state.parent, state=softmax_state,
                            subgraph=SubgraphView(softmax_state, [entry, tasklet, exit]))


        nested_sdfg2 = subgraph.nodes()[5]

        continue

    dace_model.sdfg.save('attn4_1.sdfg')


    from dace.transformation.dataflow.nested_sdfg_fusion import NestedSDFGFusion

    NestedSDFGFusion.apply_to(sdfg=softmax_state.parent,
                              nested_sdfg1=nested_sdfg1,
                              access_node=access_node,
                              nested_sdfg2=nested_sdfg2)

    dace_model.sdfg.save('attn4_2.sdfg')

    from dace.transformation.interstate.state_reordering import StateReordering

    pattern = sdutil.node_path_graph(dace.sdfg.SDFGState(), dace.sdfg.SDFGState(), dace.sdfg.SDFGState())

    for subgraph in enumerate_matches(softmax_state.parent, pattern):
        print("Match found in state", subgraph.graph.label, ". Nodes:", subgraph.nodes())

        StateReordering.apply_to(sdfg=subgraph.graph,
                                 first_state=subgraph.nodes()[0],
                                 second_state=subgraph.nodes()[1])

    dace_model.sdfg.save('attn4_3.sdfg')

    from dace.transformation.interstate.state_fusion import StateFusion

    pattern = sdutil.node_path_graph(dace.sdfg.SDFGState(), dace.sdfg.SDFGState(), dace.sdfg.SDFGState())

    for subgraph in enumerate_matches(softmax_state.parent, pattern):
        print("Match found in state", subgraph.graph.label, ". Nodes:", subgraph.nodes())

        StateFusion.apply_to(sdfg=subgraph.graph,
                             first_state=subgraph.nodes()[1],
                             second_state=subgraph.nodes()[2])

    dace_model.sdfg.save('attn4_4.sdfg')

    pattern = sdutil.node_path_graph(dace.nodes.MapEntry(dace.nodes.Map('_', [], [])),
                                     dace.nodes.Tasklet('_'),
                                     dace.nodes.MapExit(dace.nodes.Map('_', [], [])),
                                     dace.nodes.AccessNode('_'),
                                     dace.nodes.MapEntry(dace.nodes.Map('_', [], [])),
                                     dace.nodes.Tasklet('_'),
                                     dace.nodes.MapExit(dace.nodes.Map('_', [], [])))

    for subgraph in enumerate_matches(softmax_state.parent, pattern):
        print("Match found in state", subgraph.graph.label, ". Nodes:", subgraph.nodes())

        # I have to make names of both maps the same, otherwise SubgraphFusion doesn't work.
        subgraph.graph.replace(subgraph.nodes()[4].map.params[0], subgraph.nodes()[0].map.params[0])
        subgraph.nodes()[4].map.params[0] = subgraph.nodes()[0].map.params[0]

        dace_model.sdfg.save('attn4_5.sdfg')

        # I have no idea why, but only with GPUTransformSDFG schedule of the first map is Sequential
        # but the schedule of the second map is Default.
        # It prevents their fusion, so I need to change schedule of the first map to Default

        map: sdfg_nodes.Map = subgraph.nodes()[0].map
        map.schedule = dtypes.ScheduleType.Default

        SubgraphFusion.apply_to(subgraph.graph.parent, subgraph)

    dace_model.sdfg.save('attn4_6.sdfg')

    # tile map inside reduction into two maps (one over warp, other is sequential)

    from dace.transformation.dataflow.tiling import MapTiling

    pattern = sdutil.node_path_graph(dace.nodes.MapExit)

    for subgraph in enumerate_matches(softmax_state.parent, pattern):
        map_exit: dace.nodes.MapExit = subgraph.nodes()[0]
        state: dace.sdfg.SDFGState = subgraph.graph
        sdfg: dace.sdfg.SDFG = state.parent

        edges = state.out_edges(map_exit)

        map_entry: dace.nodes.MapEntry = state.entry_node(map_exit)

        if len(edges) == 1 and edges[0].data.wcr:
            print("Match found in state", subgraph.graph.label, ". Nodes:", subgraph.nodes())

            MapTiling.apply_to(sdfg=state.parent,
                               options={'tile_sizes': (seq_len//32,)},
                               map_entry=map_entry)

    dace_model.sdfg.save('attn5.sdfg')

    # create transients between two map exits
    from dace.transformation.dataflow.stream_transient import AccumulateTransient

    pattern = sdutil.node_path_graph(dace.nodes.Tasklet,
                                     dace.nodes.MapExit(dace.nodes.Map("", [], [])),
                                     dace.nodes.MapExit(dace.nodes.Map("", [], [])))

    for subgraph_view in enumerate_matches(softmax_state.parent, pattern):
        tasklet = subgraph_view.nodes()[0]
        map_exit1 = subgraph_view.nodes()[1]
        map_exit2 = subgraph_view.nodes()[2]

        edges = list(subgraph_view.edges_between(map_exit1, map_exit2))

        if len(edges) != 1 or edges[0].data.wcr is None:
            continue

        wcr = edges[0].data.wcr

        from dace.frontend.operations import detect_reduction_type

        reduction_type = detect_reduction_type(wcr)

        if reduction_type == dace.dtypes.ReductionType.Max:
            identity_value = -1e9
        elif reduction_type == dace.dtypes.ReductionType.Sum:
            identity_value = 0
        else:
            raise Exception("Unknown reduction type")

        AccumulateTransient.apply_to(subgraph_view.graph.parent, {'identity': identity_value},
                                     tasklet=tasklet, map_exit=map_exit1, outer_map_exit=map_exit2)

    dace_model.sdfg.save('attn6.sdfg')

    from dace.transformation.dataflow.wcr_extraction import WCRExtraction

    pattern = sdutil.node_path_graph(dace.nodes.MapExit(dace.nodes.Map("", [], [])),
                                     dace.nodes.AccessNode("_"))

    for subgraph_view in enumerate_matches(softmax_state.parent, pattern):

        access_node: sdfg_nodes.AccessNode = subgraph_view.nodes()[1]
        if access_node.label in {'_out', 'n2__out'}:
            print("Match found in state", subgraph_view.graph.label, ". Nodes:", subgraph_view.nodes())

            WCRExtraction.apply_to(sdfg=subgraph_view.graph.parent,
                                   map_exit=subgraph_view.nodes()[0],
                                   output_node=subgraph_view.nodes()[1])

    dace_model.sdfg.save('attn7.sdfg')

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
