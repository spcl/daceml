import typing
from typing import Optional, List, Tuple

import sympy as sp

import dace.library
from dace import SDFG, SDFGState, subsets, symbolic, data
from dace.transformation import transformation as pm
from dace.libraries import mpi

from .. import utils as distr_utils

from . import grid_mapped_array as grid_array

if typing.TYPE_CHECKING:
    from .node import DistributedMemlet


class CommunicationSolverException(Exception):
    """
    Exception raised when the communication solver fails to find a solution.
    """
    pass


MatchedDimensions = List[Tuple[Optional[symbolic.symbol], symbolic.symbol]]


def match_subset_axis_to_pgrid(
        subset: subsets.Range,
        grid_variables: List[symbolic.symbol]) -> MatchedDimensions:
    """
    Matches each axis of the subset to a grid variable if that axis is tiled
    using that grid variable.

    For example, an axis i*32:(i+1)*32 is matched to the grid variable i with
    block size 32, and would add (i, 32) to the result.

    If a dimension is not tiled by any variable, the result will contain a None
    for that dimension.

    :param subset: The subset to match.
    :param grid_variables: The grid variables to match against.
    :raises CommunicationSolverException: If matching fails
    :return: A list of tuples, each containing the grid variable and the
             matched block size.
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
            raise CommunicationSolverException(
                "Could not match subset axis {} to grid variables {}".format(
                    subset[i], ', '.join(map(str, grid_variables))))

        blocks.append((p, matches[bs]))

    return blocks


def try_match_constraint(
        sdfg: SDFG, state: SDFGState, pgrid_name: str, global_desc: data.Data,
        subset: subsets.Range,
        grid_variables: List[symbolic.symbol]) -> List[grid_array.AxisScheme]:
    """
    Try to parse the the given communication constraint as a grid mapped array.

    :param sdfg: The SDFG to operate on.
    :param state: The state to operate on (Dummy tasklets will be inserted
                  here)
    :param pgrid_name: The name of the pgrid to use.
    :param global_desc: The global data descriptor of the array to communicate.
    :param subset: The end of the distributed memlet to convert.
    :param grid_variables: The process grid corresponding to the computation to
                           which this is either an input or an output.
    :return: the axis scheme mapping implementing the communication constraint.

    :raises: CommunicationSolverException if it cannot be solved.
    """

    global_shape = global_desc.shape
    pgrid = sdfg.process_grids[pgrid_name]

    matched_dimensions = match_subset_axis_to_pgrid(subset, grid_variables)
    # index by grid variable
    matched_dimensions = [(v, bs)
                          for i, (v, bs) in enumerate(matched_dimensions)
                          if global_desc.shape[i] > 1]

    axis_mapping: List[Optional[grid_array.AxisScheme]] = [None] * len(
        pgrid.shape)

    # Assign each partitioned axis
    for i, (matched_dim, matched_bs) in enumerate(matched_dimensions):
        if matched_dim is None:
            # this is a replicated dimension, we assign these in a second pass
            # since they have no block size constraint
            continue
        grid_index = grid_variables.index(matched_dim)
        global_block_size = global_shape[i] / pgrid.shape[grid_index]
        if matched_bs != global_block_size:
            raise CommunicationSolverException(
                f"Detected block size {matched_bs} (on axis {subset[i]}) does not"
                "match global block size {global_block_size}")

        scheme = grid_array.AxisScheme(axis=i,
                                       scheme=grid_array.AxisType.PARTITION)
        axis_mapping[grid_index] = scheme

    unassigned_dims = {i for i, a in enumerate(axis_mapping) if a is None}

    # Assign replicated axes
    for i, (matched_dim, matched_bs) in enumerate(matched_dimensions):
        if matched_dim is not None:
            continue
        if unassigned_dims:
            # assign to any unused grid dimension
            grid_index = unassigned_dims.pop()
        else:
            # if none are left, exit: unmapped array dimensions are broadcasted automatically
            break
        scheme = grid_array.AxisScheme(axis=i,
                                       scheme=grid_array.AxisType.REPLICATE)
        axis_mapping[grid_index] = scheme

    # The process grid axes are broadcast
    for i, scheme in enumerate(axis_mapping):
        if scheme is not None:
            continue
        scheme = grid_array.AxisScheme(axis=None,
                                       scheme=grid_array.AxisType.BROADCAST)
        axis_mapping[i] = scheme
    return axis_mapping


class CommunicateSubArrays(pm.ExpandTransformation):

    environments = []

    def can_be_applied(self, state: SDFGState, *_, **__):
        node: 'DistributedMemlet' = state.node(
            self.subgraph[type(self)._match_node])
        sdfg = state.parent

        src_vars = list(
            map(symbolic.pystr_to_symbolic, node.src_rank_variables))
        dst_vars = list(
            map(symbolic.pystr_to_symbolic, node.dst_rank_variables))

        try:
            if node.src_pgrid is not None:
                garr = sdfg.arrays[node.src_global_array]
                try_match_constraint(sdfg, state, node.src_pgrid, garr,
                                     node.src_subset, src_vars)

            if node.dst_pgrid is not None:
                garr = sdfg.arrays[node.dst_global_array]
                try_match_constraint(sdfg, state, node.dst_pgrid, garr,
                                     node.dst_subset, dst_vars)
        except CommunicationSolverException:
            return False

        return True

    @staticmethod
    def expansion(node: 'DistributedMemlet', state: SDFGState, sdfg: SDFG):

        src_desc, dst_desc = node.validate(sdfg, state)

        src_vars = list(
            map(symbolic.pystr_to_symbolic, node.src_rank_variables))
        dst_vars = list(
            map(symbolic.pystr_to_symbolic, node.dst_rank_variables))

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

            axis_mapping = try_match_constraint(sdfg, state, pgrid_name, garr,
                                                subset, rvars)

            cls = grid_array.ScatterOntoGrid if scatter else grid_array.GatherFromGrid

            expansion = cls(node.label,
                            grid_name=pgrid_name,
                            axis_mapping=axis_mapping)

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
