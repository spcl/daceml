"""
A distributed schedule for a subgraph is a mapping from each
map entry node to process grid dimensions.
"""
import copy
import collections
from typing import List, Tuple, Dict, Set, Optional, Union
import itertools

import networkx as nx

import sympy as sp

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
        if sp.Mod(exact_size, block_size).simplify() != 0:
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
    force_local_names: Optional[Dict[str, str]] = None,
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
    :param force_local_names: A dictionary mapping global array names to local array names.
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

        if force_local_names is not None and global_name in force_local_names:
            # reuse descriptor
            local_name = force_local_names[global_name]
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


def ordered_nodes_by_state(
        sdfg: SDFG
) -> Dict[SDFGState, List[Union[nodes.Map, nodes.NestedSDFG]]]:
    ordered_maps = {}

    for state in sdfg.nodes():
        top_level_nodes = set(state.scope_children()[None])
        result_nodes = [
            node.map if isinstance(node, nodes.MapEntry) else node
            for node in sdfg_utils.dfs_topological_sort(state)
            if isinstance(node, (nodes.MapEntry,
                                 nodes.NestedSDFG)) and node in top_level_nodes
        ]
        ordered_maps[state] = result_nodes
    return ordered_maps


def rank_tile_nested(
    sdfg: SDFG,
    state: SDFGState,
    nnode: nodes.NestedSDFG,
    schedule: DistributedSchedule,
) -> Tuple[NumBlocks, RankVariables, GlobalToLocal, GlobalToLocal]:
    """
    Rank tile a nested SDFG. To make this viable in the cases we want to support
    (typically just initialization for reductions) there are the following constraints:

    * Only a single ProcessGrid is used for the whole SDFG.
    * No communication is added within the NSDFG (this enables fusability outside of it)
    * As a result, schedules in the NSDFG must be 'consistent', in that if they
      write to the same array, the communication constraints must be the same
    """

    nsdfg = nnode.sdfg

    # rule out unsqueezes
    for edge in itertools.chain(state.in_edges(nnode), state.out_edges(nnode)):
        if not utils.all_equal(edge.data.subset.size_exact(),
                               sdfg.arrays[edge.data.data].shape):
            raise NotImplementedError(
                "Cannot handle unsqueezes on NestedSDFG connectors")

    ordered_maps = ordered_nodes_by_state(nsdfg)

    rank_variables: Dict[str, int] = {}
    num_fully_replicated: int = 0

    constraints: Dict[str, subsets.Range] = {}
    global_to_local: Dict[str, str] = {}

    # Rank tile each map, collecting the constraints by array
    for nstate, map_nodes in ordered_maps.items():
        for map_node in map_nodes:
            num_blocks = schedule[map_node]

            if len(num_blocks) != map_node.get_param_num():
                raise ValueError(
                    f"Schedule for {map_node} has {len(num_blocks)} "
                    f"block sizes, but {map_node.get_param_num()} are "
                    "required.")

            map_nodes = find_map_nodes(nstate, map_node)

            # modify the map to tile it
            map_rank_variables, reads, writes = rank_tile_map(
                nsdfg,
                nstate,
                map_nodes,
                num_blocks,
                force_local_names=global_to_local)

            # gather our current constraints, and delete the global access nodes
            current_constraints = {}
            for nglobal, nlocal, subset in reads:
                current_constraints[nglobal.data] = subset
                # update global mapping
                global_to_local[nglobal.data] = nlocal.data
                # delete global access node
                nstate.remove_node(nglobal)

            for nglobal, nlocal, subset in writes:
                # update global mapping
                global_to_local[nglobal.data] = nlocal.data
                if nglobal.data in current_constraints and current_constraints[
                        nglobal.data] != subset:
                    raise ValueError(
                        "Inconsistent constraints found within map")
                current_constraints[nglobal.data] = subset
                nstate.remove_node(nglobal)

            # the tiled map now has its own rank variables. We now need to
            # these variables consistent with the outer variables since the
            # whole nested SDFG only has one ProcessGrid
            me, mx = map_nodes

            renaming = {}

            map_block_size_per_symbol = {
                s.name: bs
                for s, bs in zip(map_rank_variables, num_blocks)
            }
            # the arrays that already have constraints from another map
            for name, subset in current_constraints.items():
                if name not in constraints:
                    constraints[name] = subset
                    continue

                previous_constraint = constraints[name]

                is_symbol = lambda x: x.is_symbol
                # try to match the constraints

                used_symbols = set(subset.free_symbols)
                used_symbols |= set(previous_constraint.free_symbols)

                def new_symbol_name(name):
                    name = utils.find_str_not_in_set(used_symbols, name)
                    used_symbols.add(name)
                    return name

                pattern = copy.deepcopy(subset)
                wilds = {
                    s: sp.Wild(new_symbol_name(f"i{i}"),
                               properties=[
                                   is_symbol, lambda x: rank_variables[x.name]
                                   == map_block_size_per_symbol[s]
                               ])
                    for i, s in enumerate(subset.free_symbols)
                }
                pattern.replace(wilds)

                def range_to_basic_expr(self):
                    exprs = []
                    for (rb, re, rs), ts in zip(self.ranges, self.tile_sizes):
                        exprs.append(rb)
                        exprs.append(re)
                        exprs.append(ts)
                    return sp.Basic(*exprs)

                match = range_to_basic_expr(previous_constraint).match(
                    range_to_basic_expr(pattern))
                if match is None:
                    raise ValueError(
                        "Inconsistent constraints found within map")

                renaming.update(match)

            map_num_replicated = 0
            # now check that the block sizes are consistent
            # The should already be consistent due to the sympy matching above,
            # but let's make sure anyway
            for v, bs in zip(map_rank_variables, num_blocks):
                if v in renaming:
                    v = renaming[v]

                if v.name == node.FULLY_REPLICATED_RANK:
                    map_num_replicated += 1
                    continue

                if v.name in rank_variables:
                    if rank_variables[v.name] != bs:
                        raise ValueError(
                            "Inconsistent block sizes found within NestedSDFG")
                else:
                    rank_variables[v.name] = bs
            num_fully_replicated = max(num_fully_replicated,
                                       map_num_replicated)

    # Concretize process grid ordering
    ordered_rank_variables = []
    block_sizes = []
    for ov, bs in rank_variables.items():
        ordered_rank_variables.append(ov)
        block_sizes.append(bs)
    ordered_rank_variables += [node.FULLY_REPLICATED_RANK
                               ] * num_fully_replicated
    block_sizes += [1] * num_fully_replicated

    # Swap out the global connectors for local view connectors on the outside of the NSDFG
    nnode.in_connectors = {
        global_to_local[k]: v
        for k, v in nnode.in_connectors.items()
    }
    nnode.out_connectors = {
        global_to_local[k]: v
        for k, v in nnode.out_connectors.items()
    }

    # create outer transients for the local views
    to_iter = itertools.chain(
        zip(state.in_edges(nnode), itertools.repeat(True)),
        zip(state.out_edges(nnode), itertools.repeat(False)))

    reads = []
    writes = []
    for edge, is_read in to_iter:
        global_name = edge.dst_conn if is_read else edge.src_conn
        local_name_in_nsdfg = global_to_local[global_name]
        inner_desc = nsdfg.arrays[local_name_in_nsdfg]
        outer_local_name = sdfg.add_datadesc(
            name=local_name_in_nsdfg,
            datadesc=copy.deepcopy(inner_desc),
            find_new_name=True)
        access = state.add_access(outer_local_name)

        if is_read:
            reads.append((edge.src, access, constraints[global_name]))
            redirect_args = dict(new_dst_conn=local_name_in_nsdfg,
                                 new_src=access)
        else:
            writes.append((edge.dst, access, constraints[global_name]))
            redirect_args = dict(new_src_conn=local_name_in_nsdfg,
                                 new_dst=access)

        new_edge = xfh.redirect_edge(state,
                                     edge,
                                     new_data=outer_local_name,
                                     **redirect_args)
        new_edge.data.subset = constraints[global_name]

    for global_name, local_name in global_to_local.items():
        if not nsdfg.arrays[global_name].transient:
            nsdfg.arrays[local_name].transient = False
        del nsdfg.arrays[global_name]

    # specialize process grid vars inside the nsdfg
    sdfg.specialize({v: 0 for v in ordered_rank_variables})
    ordered_rank_variables = list(
        map(symbolic.pystr_to_symbolic, ordered_rank_variables))
    return block_sizes, ordered_rank_variables, reads, writes


def lower(sdfg: SDFG, schedule: DistributedSchedule):
    """
    Attempt to lower the SDFG to a SPMD MPI SDFG to distribute computation.

    The schedule defines the size of the process grids used to compute each of the parallel maps.

    :param sdfg: The SDFG to lower.
    :param schedule: The schedule to use.
    :note: Operates in-place.
    """
    missing = set(distr_utils.all_top_level_maps(sdfg)).difference(
        schedule.keys())

    if missing:
        raise ValueError(
            f"Missing schedule for maps: {', '.join(map(lambda x: x.label, missing))}"
        )

    # Order the schedule topologically for each state
    ordered_nodes = ordered_nodes_by_state(sdfg)

    # each map has a main process grid
    # with the dimension given by the schedule
    process_grids: Dict[str, RankVariables] = {}

    for state, top_level_nodes in ordered_nodes.items():
        for top_level_node in top_level_nodes:
            if isinstance(top_level_node, nodes.NestedSDFG):
                num_blocks, rank_variables, reads, writes = rank_tile_nested(
                    sdfg, state, top_level_node, schedule)
            else:
                map_node: nodes.Map = top_level_node
                num_blocks = schedule[map_node]

                if len(num_blocks) != map_node.get_param_num():
                    raise ValueError(
                        f"Schedule for {map_node} has {len(num_blocks)} "
                        f"block sizes, but {map_node.get_param_num()} are "
                        "required.")

                map_nodes = find_map_nodes(state, map_node)

                # modify the map to tile it
                rank_variables, reads, writes = rank_tile_map(
                    sdfg, state, map_nodes, num_blocks)
            rank_variable_names: List[str] = list(
                map(lambda s: s.name, rank_variables))

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

            process_grids[process_grid_name] = rank_variables

            to_iter = itertools.chain(zip(reads, itertools.repeat(True)),
                                      zip(writes, itertools.repeat(False)))

            for (nglobal, nlocal, subset), is_read in to_iter:
                if not is_read:
                    # FIXME We don't need to communicate this if it is a
                    # derived schedule and one of our siblings comes after us.
                    pass

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

    sdfg.validate()
    utils.expand_nodes(
        sdfg, predicate=lambda n: isinstance(n, node.DistributedMemlet))
    sdfg.validate()

    # Now that we are done lowering, we can instatiate the process grid
    # variables with zero, since each rank only sees its section of the array
    for _, variables in process_grids.items():
        repl_dict = {v.name: 0 for v in variables}
        sdfg.specialize(repl_dict)
