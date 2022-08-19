"""
A distributed schedule for a subgraph is a mapping from each
map entry node to process grid dimensions.
"""
import copy
import collections
from typing import List, Tuple, Dict, Set, Optional
import itertools

import networkx as nx

import sympy

from dace import nodes, SDFG, SDFGState, symbolic, subsets, memlet, data
from dace.sdfg import propagation, utils as sdfg_utils
from dace.transformation import helpers as xfh

from daceml.autodiff import analysis
from daceml.util import utils

NumBlocks = List[int]
DistributedSchedule = Dict[nodes.Map, NumBlocks]
MapNodes = Tuple[nodes.MapEntry, nodes.MapExit]

from .communication import node
from . import utils as distr_utils


def find_map_nodes(state: SDFGState, map_node: nodes.Map) -> MapNodes:
    found = [
        node for node in state.nodes()
        if isinstance(node, (nodes.MapEntry,
                             nodes.MapExit)) and node.map == map_node
    ]

    assert len(
        found
    ) == 2, "Found {} map scope nodes for map {}, expected exactly 2".format(
        len(found), map_node)

    x, y = found
    if isinstance(x, nodes.MapEntry):
        assert isinstance(
            y, nodes.MapExit), f"Found two entry nodes for map {map_node}"
        return x, y
    else:
        assert isinstance(
            y, nodes.MapEntry), f"Found two exit nodes for map {map_node}"
        return y, x


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
RankVariables = List[Optional[symbolic.symbol]]


def propagate_rank_local_subsets(
    sdfg: SDFG, state: SDFGState, map_nodes: MapNodes, num_blocks: NumBlocks
) -> Tuple[RankVariables, Tuple[LocalSubsets, LocalSubsets]]:
    """
    Compute the subset rank local subsets we need.

    For each rank-tiled parameter, the returned dictionary contains a mapping
    from the original parameter name to the variable name of the rank index.

    The two-tuple contains mappings from array names to the subset expressions
    for the inputs and outputs.
    """
    me, mx = map_nodes
    map_node = me.map

    # We need to reindex using "fake" block indices
    # Create new symbols for the block indices and block sizes
    used_vars: Set[str] = set(map_node.params)
    rank_variables: RankVariables = []
    # build a new range for the fake local map
    outer_range = []

    for param, size, (start, end,
                      step), n_blocks in zip(map_node.params,
                                             map_node.range.size_exact(),
                                             map_node.range, num_blocks):
        if n_blocks != 1:
            rank_variable = utils.find_str_not_in_set(used_vars,
                                                      f"__block{param}")
            used_vars.add(rank_variable)
            outer_range.append((
                symbolic.pystr_to_symbolic(
                    f"{rank_variable} * ({size} / {n_blocks})"),
                symbolic.pystr_to_symbolic(
                    f"({rank_variable} + 1) * ({size} / {n_blocks}) - 1"),
                symbolic.pystr_to_symbolic(f"{step}"),
            ))

            rank_variables.append(symbolic.pystr_to_symbolic(rank_variable))
        else:
            rank_variables.append(symbolic.symbol(node.FULLY_REPLICATED_RANK))
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
            # compute the rank local subset using propagation through our
            # "fake" MPI map
            rank_local = propagation.propagate_subset(
                memlets, sdfg.arrays[arr_name], map_node.params, outer_range,
                defined_vars, not is_input)
            assert isinstance(rank_local.subset, subsets.Range)
            result[arr_name] = rank_local.subset
    return rank_variables, results


GlobalToLocal = List[Tuple[nodes.AccessNode, nodes.AccessNode, subsets.Range]]


def rank_tile_map(
    sdfg: SDFG,
    state: SDFGState,
    map_nodes: MapNodes,
    num_blocks: NumBlocks,
    derived_schedules: Dict[nodes.Map, List[nodes.Map]],
    processed_nodes=Dict[nodes.Map, Dict[str, str]]
) -> Tuple[RankVariables, GlobalToLocal, GlobalToLocal]:
    """
    Tile the map according to the given block sizes, create rank-local views
    for the reads and writes to global arrays.

    * Detect and handle local write-conflicts
    * Handle views correctly
    * Insert rank-local arrays and reroute the map edges to go to the local
      views

    :param sdfg: The SDFG to operate on.
    :param state: The state to operate on.
    :param map_nodes: The map nodes to operate on.
    :param num_blocks: The number of blocks to tile the map with in each dimension.
    :returns: created symbolic variables for each rank axis, and two lists of
              tuples associating global AccessNodes, local AccessNodes and the
              symbolic communication constraints. The first is for reads, the
              second for writes.
    """
    me, mx = map_nodes

    # Compute tiled map range
    new_range = compute_tiled_map_range(me.map, num_blocks)

    rank_variables, (input_subsets,
                     output_subsets) = propagate_rank_local_subsets(
                         sdfg, state, map_nodes, num_blocks)

    # if the map is derived from another map, we need to grab the local
    # transients from that map so that we write to the same ones
    derived_parent_array_names: Dict[str, str] = {}
    if me.map in derived_schedules:
        parents = set(derived_schedules[me.map]).intersection(processed_nodes)
        if parents:
            derived_parent = next(iter(parents))
            derived_parent_array_names.update(processed_nodes[derived_parent])

    # set the new range
    me.map.range = new_range

    to_iter = itertools.chain(
        zip(input_subsets.items(), itertools.repeat(True)),
        zip(output_subsets.items(), itertools.repeat(False)))

    result_read: GlobalToLocal = []
    result_write: GlobalToLocal = []
    for (arr_name, new_subset), is_input in to_iter:

        # Determine the outer edge
        outer_edges = state.in_edges(me) if is_input else state.out_edges(mx)
        outer_edge_cands = [
            edge for edge in outer_edges if edge.data.data == arr_name
        ]
        if len(outer_edge_cands) > 1:
            # FIXME this could be supported using a preprocessing
            # transformation
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
        if global_name in derived_parent_array_names:
            # if we are derived from another schedule, reuse that schedule's local name
            local_name = derived_parent_array_names[global_name]
        else:
            local_name, _ = sdfg.add_transient(
                name="local_" + global_name,
                shape=[
                    symbolic.overapproximate(r).simplify()
                    for r in new_subset.size_exact()
                ],
                dtype=sdfg.arrays[global_name].dtype,
                find_new_name=True)

        local_node = state.add_access(local_name)
        if is_input:
            result_read.append((global_node, local_node, new_subset))
        else:
            result_write.append((global_node, local_node, new_subset))

        if is_input:
            redirect_args = dict(new_src=local_node)
        else:
            redirect_args = dict(new_dst=local_node)

        new_edge = xfh.redirect_edge(state,
                                     outer_edge,
                                     new_data=local_name,
                                     **redirect_args)
        new_edge.data.subset = new_subset

    return rank_variables, result_read, result_write


def write_variables_per_dimension(
        data_name: str, mx: nodes.MapExit,
        state: SDFGState) -> Optional[List[symbolic.symbol]]:
    edges_for_write = (edge for edge in state.in_edges(mx)
                       if edge.data.data == data_name)

    variable_for_dim = []
    for edge in edges_for_write:
        for i, dim in enumerate(edge.data.subset.ranges):
            if len(variable_for_dim) <= i:
                variable_for_dim.append(None)
            symbols_in_dim = set()
            for d in dim:
                symbols_in_dim |= symbolic.symlist(d).keys()
            if len(symbols_in_dim) > 1:
                return None
            elif symbols_in_dim:
                variable_for_dim[i] = symbols_in_dim.pop()
            else:
                variable_for_dim[i] = None
    return variable_for_dim


def derive_inplace_schedules(
        sdfg: SDFG,
        schedule: DistributedSchedule) -> Dict[nodes.Map, nodes.Map]:
    """
    Derive schedules for inplace operations.
    updates ``schedule`` inplace, and returns a mapping from map_node to the map_node from which the schedule is derived
    """

    # maps are allowed to be missing if the operation is inplace.
    # the schedule will be derived from the other maps.
    missing = []
    for map_node, state in distr_utils.all_top_level_maps(sdfg,
                                                          yield_parent=True):
        if map_node not in schedule:
            missing.append((map_node, state))

    newly_added_schedules: DistributedSchedule = {}

    derived = nx.Graph()
    for map_node, state in missing:
        me, mx = find_map_nodes(state, map_node)
        writes = {
            e.data.data
            for e in state.out_edges(mx) if not e.data.is_empty()
        }

        # search for a map that writes to the same arrays, and has a schedule
        for other_map_node, other_state in distr_utils.all_top_level_maps(
                sdfg, yield_parent=True):
            if other_map_node not in schedule or other_map_node is map_node:
                continue
            ome, omx = find_map_nodes(other_state, other_map_node)
            other_writes = {
                e.data.data
                for e in other_state.out_edges(omx) if not e.data.is_empty()
            }
            if writes.difference(other_writes):
                # there is a write that isn't covered
                continue

            # drop dimensions which we don't need.
            # Heuristic here is to only support
            var_to_other_schedule = {
                v: n
                for v, n in zip(other_map_node.params,
                                schedule[other_map_node])
            }
            new_schedule = {}
            fail = False
            for array_name in writes:

                variables = write_variables_per_dimension(
                    array_name, mx, state)
                other_variables = write_variables_per_dimension(
                    array_name, omx, other_state)
                if variables is None or other_variables is None:
                    fail = True
                    break

                for v, ov, in zip(variables, other_variables):
                    if v not in new_schedule:
                        new_schedule[v] = var_to_other_schedule[ov]
                    elif new_schedule[v] != var_to_other_schedule[ov]:
                        # if the schedule is inconsitent, fail
                        fail = True
                        break
            if fail:
                continue

            newly_added_schedules[map_node] = [
                new_schedule[p] for p in map_node.params
            ]
            derived.add_edge(map_node, other_map_node)
            # we have sucessfully constructed a new schedule
            break
        else:
            raise ValueError(f"No schedule provided for {map_node}")

    schedule.update(newly_added_schedules)

    # now we do a transitive closure on the derived relation
    # so that we can always determine when schedules are derived from each
    # other
    derived = nx.transitive_closure(derived)
    return nx.to_dict_of_lists(derived)


def lower(sdfg: SDFG, schedule: DistributedSchedule):
    """
    Attempt to lower the SDFG to a SPMD MPI SDFG to distribute computation.

    The schedule defines the size of the process grids used to compute each of the parallel maps.

    In place operations
    -------------------
    In place operations, where an array is written twice (e.g. common with WCR
    and their initialization states) have special support. 
    To schedule a graph with in place operations, only schedule one of the maps
    where the array is written. The partitioning scheme of the map you specify
    will be applied to other in-place maps.
    

    :param sdfg: The SDFG to lower.
    :param schedule: The schedule to use.
    :note: Operates in-place.
    """
    derived_schedules = derive_inplace_schedules(sdfg, schedule)

    missing = set(distr_utils.all_top_level_maps(sdfg)).difference(
        schedule.keys())

    if missing:
        raise ValueError(
            f"Missing schedule for maps [ ', '.join(map(str, missing)) ]")

    # Order the schedule topologically for each state
    ordered_maps: Dict[SDFGState, List[nodes.Map]] = {}

    for state in sdfg.nodes():
        top_level_nodes = set(state.scope_children()[None])
        map_entries = [
            node.map for node in sdfg_utils.dfs_topological_sort(state)
            if isinstance(node, nodes.MapEntry) and node in top_level_nodes
        ]
        ordered_maps[state] = map_entries

    # each map has a main process grid
    # with the dimension given by the schedule
    maps_to_pgrids: Dict[nodes.Map, Tuple[RankVariables, str]] = {}

    processed: Dict[nodes.Map, Dict[str, str]] = collections.defaultdict(dict)
    for state, map_nodes in ordered_maps.items():
        for map_node in map_nodes:
            num_blocks = schedule[map_node]

            if len(num_blocks) != map_node.get_param_num():
                raise ValueError(
                    f"Schedule for {map_node} has {len(num_blocks)} "
                    f"block sizes, but {map_node.get_param_num()} are "
                    "required.")

            map_nodes = find_map_nodes(state, map_node)

            # Create a process grid that will be used for communication
            process_grid_name = sdfg.add_pgrid(shape=num_blocks)
            distr_utils.initialize_fields(state, [
                f'MPI_Comm {process_grid_name}_comm;',
                f'MPI_Group {process_grid_name}_group;',
                f'int {process_grid_name}_coords[{len(num_blocks)}];',
                f'int {process_grid_name}_dims[{len(num_blocks)}];',
                f'int {process_grid_name}_rank;',
                f'int {process_grid_name}_size;',
                f'bool {process_grid_name}_valid;',
            ])

            # modify the map to tile it
            rank_variables, reads, writes = rank_tile_map(
                sdfg, state, map_nodes, num_blocks, derived_schedules,
                processed)

            rank_variable_names: List[str] = list(
                map(lambda s: s.name, rank_variables))

            maps_to_pgrids[map_node] = rank_variables, process_grid_name

            to_iter = itertools.chain(zip(reads, itertools.repeat(True)),
                                      zip(writes, itertools.repeat(False)))

            for (nglobal, nlocal, subset), is_read in to_iter:
                processed[map_node][nglobal.data] = nlocal.data

                full_subset = subsets.Range.from_array(nglobal.desc(sdfg))

                global_params = dict(rank_variables=[],
                                     pgrid=None,
                                     subset=full_subset,
                                     global_array=nglobal.data)

                local_params = dict(rank_variables=rank_variable_names,
                                    pgrid=process_grid_name,
                                    subset=subset,
                                    global_array=nglobal.data)

                global_prefix = "src_" if is_read else "dst_"
                local_prefix = "dst_" if is_read else "src_"

                add_prefix = lambda d, p: {p + k: v for k, v in d.items()}

                comm = node.DistributedMemlet(
                    name="communicate_" + nglobal.data,
                    **add_prefix(global_params, global_prefix),
                    **add_prefix(local_params, local_prefix))

                state.add_node(comm)
                src = nglobal if is_read else nlocal
                dst = nlocal if is_read else nglobal
                state.add_edge(src, None, comm, None,
                               sdfg.make_array_memlet(src.data))
                state.add_edge(comm, None, dst, None,
                               sdfg.make_array_memlet(dst.data))

    utils.expand_nodes(
        sdfg, predicate=lambda n: isinstance(n, node.DistributedMemlet))

    # Now that we are done lowering, we can instatiate the process grid
    # variables with zero, since each rank only sees its section of the array
    for _, (variables, _) in maps_to_pgrids.items():
        repl_dict = {v.name: 0 for v in variables}
        sdfg.specialize(repl_dict)
