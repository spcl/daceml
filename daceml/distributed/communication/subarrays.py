import typing
from typing import Optional, List, Tuple, Callable
from dace.frontend.python.memlet_parser import pystr_to_symbolic

import sympy as sp

import dace.library
from dace import SDFG, SDFGState, nodes, subsets, symbolic, data, distr_types
from dace.transformation import transformation as pm
from dace.libraries import mpi

from .. import utils as distr_utils

if typing.TYPE_CHECKING:
    from .node import DistributedMemlet, FUL


class CommunicationSolverException(Exception):
    """
    Exception raised when the communication solver fails to find a solution.
    """
    pass
MatchedDimensions = List[Tuple[Optional[symbolic.symbol], symbolic.symbol]]
def match_subset_axis_to_pgrid(subset: subsets.Range, grid_variables:
        List[symbolic.symbol]) -> MatchedDimensions:
    """
    Matches each axis of the subset to a grid variable if that axis is tiled using that grid variable.

    For example, an axis i*32:(i+1)*32 is matched to the grid variable i with block size 32, and would
    add (i, 32) to the result.

    If a dimension is not tiled by any variable, the result will contain a None for that dimension.

    :param subset: The subset to match.
    :param grid_variables: The grid variables to match against.
    :raises CommunicationSolverException: If matching fails
    :return: A list of tuples, each containing the grid variable and the matched block size.
    """

    # avoid import loop
    from .node import FULLY_REPLICATED_RANK

    blocks = []
    sizes = subset.size_exact()
    for i, (start, end, step) in enumerate(subset):
        if step != 1:
            raise CommunicationSolverException("Subset step must be 1")
        bs = sp.Wild("bs", exclude=grid_variables)

        expr = sp.Basic(*map(symbolic.pystr_to_symbolic, (start, end)))
        if not expr.free_symbols:
            # no free symbols; this is a constant range and needs to be
            # replicated to every rank
            blocks.append((None, sizes[i]))
            continue

        for p in grid_variables:
            if p.name == FULLY_REPLICATED_RANK:
                # this is not a valid variable to match
                continue

            # try to match with this block variable
            pattern = sp.Basic(p * bs, (p + 1) * bs - 1)

            matches = expr.match(pattern)
            if matches is not None:
                break
        else:
            # couldn't detect a block: exit
            raise CommunicationSolverException("Could not match subset axis {} to grid variables {}".format(subset[i], ', '.join(map(str, grid_variables))))


        blocks.append((p, matches[bs]))

    return blocks

def compute_scatter_color(parent_grid_variables: List[symbolic.symbol], parent_grid_shape: List[int], matched_dimensions: MatchedDimensions) -> List[bool]:
    # avoid import loop
    from .node import FULLY_REPLICATED_RANK

    # We need to setup a broadcast grid to make sure the ranks on the
    # remaining dimensions of the scatter grid get their values. We will
    # split the process grid into two subgrids to achieve this

    dim_to_idx = {v: i for i, v in enumerate(parent_grid_variables) if v.name != FULLY_REPLICATED_RANK}

    # these are the dimensions we need to scatter our data over. these
    # definitely need to be kept
    scattered_dims = {dim_to_idx[s] for s, _ in matched_dimensions if s is not None}

    # these are the number of unpartitioned dimensions in our subset
    # these need to be mapped to a dimension of the process grid with size one
    required_full_replication_dims = len(matched_dimensions) - len(scattered_dims)

    # The indices of dims of size 1 in the global pgrid
    empty_dims = {i for i, s in enumerate(parent_grid_variables) if s.name == FULLY_REPLICATED_RANK}
    assert all(parent_grid_shape[i] == 1 for i in empty_dims)
    # the size 1 dimensions in the scatter grid
    # the key is the index of the dimension in the scatter grid, the value
    # is the index of the dimension in the parent grid

    empty_scatter_dims = set()
    for i in range(required_full_replication_dims):
        # choose any empty rank
        parent_empty_dim = next(iter(empty_dims))
        empty_dims.remove(parent_empty_dim)
        scattered_dims.add(parent_empty_dim)
        empty_scatter_dims.add(parent_empty_dim)


    scatter_color = [i in scattered_dims for i in range(len(parent_grid_shape))]
    return scatter_color 




def try_construct_subarray(sdfg: SDFG, state: SDFGState, pgrid_name: str, global_desc: data.Data, subset: subsets.Range, grid_variables: List[symbolic.symbol], dry_run: bool) -> Optional[Tuple[str, str, Optional[str]]]:
    """
    Try to convert the given end of the distributed memlet to a subarray, creating the necessary process grids.
    Returns the name of the subarray, the name of the scatter grid and the name of the bcast/reduction grid.

    :param sdfg: The SDFG to operate on.
    :param state: The state to operate on (Dummy tasklets will be inserted here)
    :param pgrid_name: The name of the pgrid to use.
    :param global_desc: The global data descriptor of the array to communicate.
    :param subset: The end of the distributed memlet to convert.
    :param grid_variables: The process grid corresponding to the computation to which this is either an input or an output.
    :param dry_run: If True, don't actually create the grids and subarray. Instead, return None.
    :return: The name of the subarray, the name of the scatter grid and the
             name of the bcast grid (possibly ``None`` if no broadcast is necessary).
    """

    pgrid = sdfg.process_grids[pgrid_name]

    matched_dimensions = match_subset_axis_to_pgrid(subset, grid_variables)


    if len(matched_dimensions) > len(pgrid.shape):
        # Haven't thought about this yet...
        # When can this happen?
        raise NotImplementedError()
    elif len(matched_dimensions) < len(pgrid.shape):

        scatter_color = compute_scatter_color(grid_variables, pgrid.shape, matched_dimensions)
        assert sum(scatter_color) == subset.dims()
        # The broadcast grid provides the data to the ranks that are not
        # covered in the scatter grid
        bcast_color = list(map(lambda x: not x, scatter_color))


        subgrid_shape = lambda color: [s for s, c in zip(pgrid.shape, color) if c]
        scatter_shape = subgrid_shape(scatter_color)
        bcast_shape = subgrid_shape(bcast_color)

        if not dry_run:
            scatter_grid_name = sdfg.add_pgrid(shape=scatter_shape, parent_grid=pgrid_name, 
                    color=scatter_color)
            bcast_grid_name = sdfg.add_pgrid(shape=bcast_shape, parent_grid=pgrid_name,
                    color=bcast_color)

            for name, shape in ((scatter_grid_name, scatter_shape), (bcast_grid_name, bcast_shape)):
                distr_utils.initialize_fields(state, [
                    f'MPI_Comm {name}_comm;',
                    f'MPI_Group {name}_group;',
                    f'int {name}_coords[{len(shape)}];',
                    f'int {name}_dims[{len(shape)}];',
                    f'int {name}_rank;',
                    f'int {name}_size;',
                    f'bool {name}_valid;',
                    ]
                )
        current_idx = 0
        symbol_to_idx = {}
        for i, s in enumerate(grid_variables):
            if scatter_color[i]:
                symbol_to_idx[s] = current_idx
                current_idx += 1
    else:
        scatter_shape = pgrid.shape
        scatter_grid_name = pgrid_name
        bcast_grid_name = None
        symbol_to_idx = {s: i for i, s in enumerate(grid_variables)}

    # The indices of dims of size 1 in the scatter_grid
    empty_dims = {i for i, s in enumerate(scatter_shape) if s == 1}

    # Now we need to map dimensions of the subset to the dimensions in the scatter grid
    correspondence: List[int] = []

    for i, (p, bs) in enumerate(matched_dimensions):
        if p is None:
            # we can choose to map to any of the empty ranks
            chosen = next(iter(empty_dims))
            empty_dims.remove(chosen)
            correspondence.append(chosen)
        else:
            grid_index = symbol_to_idx[p]

            global_block_size = global_desc.shape[i] / pgrid.shape[grid_index]
            if global_block_size % bs != 0:
                return None

            if bs != global_block_size:
                raise CommunicationSolverException("Detected block size {} (on axis {}) does not match global block size {}".format(bs, subset[i], global_block_size))
            correspondence.append(grid_index)

    assert all(map(lambda i: 0 <= i < len(scatter_shape), correspondence))

    if not dry_run:
        # create subarray 
        subarray_name = sdfg.add_subarray(dtype=global_desc.dtype,
                    shape=global_desc.shape,
                    subshape=subset.size_exact(),
                    pgrid=scatter_grid_name,
                    correspondence=correspondence)
        distr_utils.initialize_fields(state, [f'MPI_Datatype {subarray_name};', f'int* {subarray_name}_counts;', f'int* {subarray_name}_displs;'])

        return subarray_name, scatter_grid_name, bcast_grid_name
    else:
        return None
    

class CommunicateSubArrays(pm.ExpandTransformation):

    environments = []
    
    def can_be_applied(self, state: SDFGState, *_, **__):
        node: 'DistributedMemlet' = state.node(self.subgraph[type(self)._match_node])
        sdfg = state.parent


        src_vars = list(map(symbolic.pystr_to_symbolic, node.src_rank_variables))
        dst_vars = list(map(symbolic.pystr_to_symbolic, node.dst_rank_variables))

        try:
            if node.src_pgrid is not None:
                garr = sdfg.arrays[node.src_global_array]
                try_construct_subarray(sdfg, state, node.src_pgrid, garr, node.src_subset, src_vars, dry_run=True)

            if node.dst_pgrid is not None:
                garr = sdfg.arrays[node.dst_global_array]
                try_construct_subarray(sdfg, state, node.dst_pgrid, garr, node.dst_subset, dst_vars, dry_run=True)
        except CommunicationSolverException:
            return False


        return True

    @staticmethod
    def expansion(node: 'DistributedMemlet', state: SDFGState, sdfg: SDFG):

        src_desc, dst_desc = node.validate(sdfg, state)

        src_vars = list(map(symbolic.pystr_to_symbolic, node.src_rank_variables))
        dst_vars = list(map(symbolic.pystr_to_symbolic, node.dst_rank_variables))

        # There are 3 cases:

        if node.src_pgrid is not None and node.dst_pgrid is not None:
            # 1. src and dst both have pgrids
            raise NotImplementedError()

        elif node.src_pgrid is not None or node.dst_pgrid is not None:
            # 2. only one of the two has a pgrid
            # in this case we emit a BlockScatter or BlockGather

            scatter = node.dst_pgrid is not None

            if scatter:
                pgrid_name = node.dst_pgrid
                garr = sdfg.arrays[node.dst_global_array]
                subset = node.dst_subset
                rvars = dst_vars
            else:
                pgrid_name = node.src_pgrid
                garr = sdfg.arrays[node.src_global_array]
                subset = node.src_subset
                rvars = src_vars

            subarray_name, scatter_grid, bcast_grid = try_construct_subarray(sdfg, state, pgrid_name, garr, subset, rvars, dry_run=False)

            if scatter:
                expansion = mpi.BlockScatter(node.label, subarray_type=subarray_name,
                        scatter_grid=scatter_grid,
                        bcast_grid=bcast_grid)
            else:
                expansion = mpi.BlockGather(node.label, subarray_type=subarray_name,
                        gather_grid=pgrid_name,
                        reduce_grid=bcast_grid)

            # clean up connectors to match the new node
            expansion.add_in_connector("_inp_buffer")
            expansion.add_out_connector("_out_buffer")
            state.in_edges(node)[0].dst_conn = "_inp_buffer"
            state.out_edges(node)[0].src_conn = "_out_buffer"

            return expansion
        else:
            # both have no pgrid
            # this should just be a copy (?)
            raise NotImplementedError()
