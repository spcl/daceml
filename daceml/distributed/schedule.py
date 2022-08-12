"""
A distributed schedule for a subgraph is a mapping from each
map entry node to process grid dimensions.
"""
import copy
import collections
from numbers import Integral
from typing import List, Tuple, Dict, Iterator, Union, Sequence, Set
from dace.frontend.common.distr import ShapeType
import sympy
import functools
import itertools

import dace
from dace import nodes, SDFG, SDFGState, symbolic, subsets, memlet, data
from dace.sdfg import propagation, graph
from dace.libraries import mpi
from dace.transformation.dataflow import strip_mining
from dace.transformation import helpers as xfh
from torch import Block

from daceml.util import utils

NumBlocks = List[int]
DistributedSchedule = Dict[nodes.Map, NumBlocks]
MapNodes = Tuple[nodes.MapEntry, nodes.MapExit]

from daceml.distributed import communication_solver as comm_solver


def all_top_level_maps(sdfg: SDFG) -> Iterator[nodes.Map]:
    for state in sdfg.nodes():
        for node in state.scope_children()[None]:
            if isinstance(node, nodes.MapEntry):
                yield node.map


def find_map_nodes(sdfg: SDFG,
                   map_node: nodes.Map) -> Tuple[SDFGState, MapNodes]:
    """
    The map must be a toplevel map in ``sdfg``
    """
    found_states = []
    found = []
    for state in sdfg.nodes():
        for node in state.nodes():
            if isinstance(
                    node,
                (nodes.MapEntry, nodes.MapExit)) and node.map == map_node:
                found_states.append(state)
                found.append(node)
    assert len(
        found
    ) == 2, "Found {} map scope nodes for map {}, expected exactly 2".format(
        len(found), map_node)
    assert found_states[0] is found_states[
        1], "Found map scope nodes in different states"

    state = found_states[0]
    x, y = found
    if isinstance(x, nodes.MapEntry):
        assert isinstance(
            y, nodes.MapExit), f"Found two entry nodes for map {map_node}"
        return state, (x, y)
    else:
        assert isinstance(
            y, nodes.MapEntry), f"Found two exit nodes for map {map_node}"
        return state, (y, x)


def compute_tiled_map_range(nmap: nodes.Map,
                            num_blocks: NumBlocks) -> subsets.Range:
    """
    Compute the range of the map after rank-tiling it with num_blocks
    """
    new_ranges = []
    exact_sizes = nmap.range.size_exact()
    for (td_from, td_to, td_step), block_size, exact_size in utils.strict_zip(
            nmap.range, num_blocks, exact_sizes):
        if td_step != 1:
            raise NotImplementedError("Cannot tile map with step")

        # check that we divide evenly
        if sympy.Mod(exact_size, block_size).simplify() != 0:
            raise ValueError(f"{exact_size} is not divisible by {block_size}")
        td_to_new = exact_size // block_size - 1
        td_step_new = td_step
        new_ranges.append((0, td_to_new, td_step_new))
    return subsets.Range(new_ranges)


LocalSubsets = Dict[str, subsets.Range]


def propagate_rank_local_subsets(
    sdfg: SDFG, state: SDFGState, map_nodes: MapNodes, num_blocks: NumBlocks
) -> Tuple[Dict[str, str], Tuple[LocalSubsets, LocalSubsets]]:
    """
    Compute the subset rank local subsets we need.

    For each rank-tiled parameter, the returned dictionary contains a mapping
    from the original parameter name to the variable name of the rank index.

    The tuple contains mappings from array names to the subset expressions for the inputs and outputs
    """
    me, mx = map_nodes
    map_node = me.map

    # We need to reindex using "fake" block indices
    # Create new symbols for the block indices and block sizes
    used_vars: Set[str] = set(map_node.params)
    rank_variables_mapping = {}
    # build a new range for the fake local map
    outer_range = []

    for param, size, (start, end,
                      step), n_blocks in zip(map_node.params,
                                             map_node.range.size_exact(),
                                             map_node.range, num_blocks):
        if n_blocks != 1:
            block_variable = utils.find_str_not_in_set(used_vars,
                                                       f"__block{param}")
            used_vars.add(block_variable)
            outer_range.append((
                symbolic.pystr_to_symbolic(
                    f"{block_variable} * ({size} / {n_blocks})"),
                symbolic.pystr_to_symbolic(
                    f"({block_variable} + 1) * ({size} / {n_blocks}) - 1"),
                symbolic.pystr_to_symbolic(f"{step}"),
            ))

            rank_variables_mapping[param] = block_variable
        else:
            outer_range.append((start, end, step))

    outer_range = subsets.Range(outer_range)

    # Collect all defined variables
    scope_node_symbols = set(conn for conn in me.in_connectors
                             if not conn.startswith('IN_'))
    defined_vars = [
        symbolic.pystr_to_symbolic(s)
        for s in (state.symbols_defined_at(me).keys()
                  | sdfg.constants.keys()) if s not in scope_node_symbols
    ]
    defined_vars.extend(map(symbolic.pystr_to_symbolic, used_vars))
    defined_vars = set(defined_vars)

    results: Tuple[LocalSubsets, LocalSubsets] = ({}, {})
    for is_input, result in zip([True, False], results):

        # gather internal memlets by the out array they write to
        internal_memlets: Dict[
            str, List[memlet.Memlet]] = collections.defaultdict(list)
        edges = state.out_edges(me) if is_input else state.in_edges(mx)
        for edge in edges:
            if edge.data.is_empty():
                continue
            internal_memlets[edge.data.data].append(edge.data)

        for arr_name, memlets in internal_memlets.items():
            # compute the rank local subset using propagation through our "fake" MPI map
            rank_local = propagation.propagate_subset(
                memlets, sdfg.arrays[arr_name], map_node.params, outer_range,
                defined_vars, not is_input)
            assert isinstance(rank_local.subset, subsets.Range)
            result[arr_name] = rank_local.subset
    return rank_variables_mapping, results


def rank_tile_map(
    sdfg: SDFG, state: SDFGState, map_nodes: MapNodes, num_blocks: NumBlocks
) -> Tuple[Dict[str, str], List[Tuple[nodes.AccessNode, nodes.AccessNode,
                                      subsets.Range]]]:
    """
    Tile the map according to the given block sizes, create rank-local views 
    for the reads and writes to global arrays.

    * Detect and handle local write-conflicts
    * Handle views correctly
    * Insert rank-local arrays and reroute the map edges to go to the local views
    """
    me, mx = map_nodes

    # Compute tiled map range
    new_range = compute_tiled_map_range(me.map, num_blocks)

    rank_variables, (input_subsets,
                     output_subsets) = propagate_rank_local_subsets(
                         sdfg, state, map_nodes, num_blocks)

    # set the new range
    me.map.range = new_range

    to_iter = itertools.chain(
        zip(input_subsets.items(), itertools.repeat(True)),
        zip(output_subsets.items(), itertools.repeat(False)))

    result: List[Tuple[nodes.AccessNode, nodes.AccessNode, subsets.Range]] = []
    for (arr_name, new_subset), is_input in to_iter:

        # Determine the outer edge
        outer_edges = state.in_edges(me) if is_input else state.out_edges(mx)
        outer_edge_cands = [
            edge for edge in outer_edges if edge.data.data == arr_name
        ]
        if len(outer_edge_cands) > 1:
            # FIXME this could be supported using a preprocessing transformation
            raise NotImplementedError(
                "Multiple outer edges to one array not implemented")
        elif len(outer_edge_cands) == 0:
            raise ValueError(f"No outer edge to {arr_name}")
        outer_edge = outer_edge_cands[0]

        global_name = outer_edge.data.data
        global_node = outer_edge.src if is_input else outer_edge.dst

        if not isinstance(global_node, nodes.AccessNode):
            # FIXME tasklets should be replicated for each rank
            raise NotImplementedError("Cannot handle non-access nodes yet")
        elif isinstance(global_node.desc(sdfg), data.View):
            raise NotImplementedError("Cannot handle views yet")

        # Create the rank-local array
        local_name, _ = sdfg.add_transient(
            name="local_" + global_name,
            shape=[
                symbolic.overapproximate(r).simplify()
                for r in new_subset.size_exact()
            ],
            dtype=sdfg.arrays[global_name].dtype,
            find_new_name=True)

        local_node = state.add_access(local_name)
        result.append((global_node, local_node, new_subset))

        if is_input:
            redirect_args = dict(new_src=local_node)
        else:
            redirect_args = dict(new_dst=local_node)

        new_edge = xfh.redirect_edge(state,
                                     outer_edge,
                                     new_data=local_name,
                                     **redirect_args)
        new_edge.data.subset = new_subset

    return rank_variables, result


def lower(sdfg: SDFG, schedule: DistributedSchedule):
    """
    Lower with the given schedule
    """

    missing = set(all_top_level_maps(sdfg)).difference(schedule.keys())
    if missing:
        raise ValueError(
            f"Missing schedule for maps {', '.join(map(str, missing))}")

    for map_node, num_blocks in schedule.items():
        if len(num_blocks) != map_node.get_param_num():
            raise ValueError(
                f"Schedule for {map_node} has {len(num_blocks)} "
                f"block sizes, but {map_node.get_param_num()} are required.")

        state, map_nodes = find_map_nodes(sdfg, map_node)
        # modify the map to tile it
        rank_variables, global_to_local = rank_tile_map(
            sdfg, state, map_nodes, num_blocks)

        # insert MPI communication
        solver = comm_solver.CommunicationSolver(sdfg, state, rank_variables)

        for nglobal, nlocal, subset in global_to_local:
            if state.in_degree(nlocal) > 0:
                solver.solve_write(nlocal, nglobal, subset)
            elif state.out_degree(nlocal) > 0:
                solver.solve_read(nlocal, nglobal, subset)
            else:
                raise ValueError(f"Generated {nlocal} is isolated")
