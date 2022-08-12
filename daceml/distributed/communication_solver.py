from typing import Set, Dict, List, Optional
import copy
import collections
import dataclasses

import dace
from dace import nodes, SDFG, SDFGState, memlet, symbolic, subsets, registry
from dace.sdfg import propagation
from dace.frontend.common.distr import ShapeType
from dace.libraries import mpi

from daceml.distributed.schedule import MapNodes, NumBlocks
from daceml.util import utils


@dataclasses.dataclass
class BlockScatterSolution:
    grid_variables: List[str]
    array_size: List[int]
    grid_size: List[int]
    #: the i-th axis of the array will map to correspondence[i]-th axis of the grid
    #: if correspondence[i] == None then the axis is replicated
    correspondence: List[Optional[int]]

    def compute_subset(self) -> subsets.Range:
        """
        Compute the subset of the global array that the local arrays will hold
        if this solution is used.
        The subset will include the grid variables for dimensions that are partitioned.
        """
        output_range = []
        for array_dim, grid_index in zip(self.array_size, self.correspondence):
            if grid_index is None or self.grid_size[grid_index] == 1:
                # this axis is replicated
                output_range.append((0, array_dim, 1))
                continue

            grid_dim = self.grid_size[grid_index]
            grid_var = symbolic.pystr_to_symbolic(
                self.grid_variables[grid_index])

            if array_dim % grid_dim != 0:
                raise ValueError(
                    "array_dim ({}) is not a multiple of grid_dim ({})".format(
                        array_dim, grid_dim))

            block_size = array_dim // grid_dim
            output_range.append(
                (symbolic.pystr_to_symbolic(f"{grid_var} * {block_size}"),
                 symbolic.pystr_to_symbolic(
                     f"({grid_var} + 1) * {block_size} - 1"), 1))
        return subsets.Range(output_range)

    def insert_node(self, state: SDFGState, src: nodes.AccessNode,
                    dst: nodes.AccessNode) -> mpi.BlockScatter:
        sdfg = state.parent

        global_desc = src.desc(sdfg)
        local_desc = dst.desc(sdfg)

        assert global_desc.dtype == dst.desc(sdfg).dtype
        subset = self.compute_subset()
        assert utils.all_equal(subset.size_exact(), local_desc.shape)

        # add the process grid
        grid = sdfg.add_pgrid(self.grid_size)
        if any(c is None for c in self.correspondence):
            corr: List[int] = [x for x in self.correspondence if x is not None]
            raise NotImplementedError()
            subgrid_size = [
                size for size, c in zip(self.grid_size, self.correspondence)
                if c is not None
            ]
            # sub_grid = sdfg.add_pgrid(subgrid_size)
        else:
            corr: List[int] = self.correspondence
            sub_grid = grid

        subarray = sdfg.add_subarray(dtype=global_desc.dtype,
                                     shape=global_desc.shape,
                                     subshape=local_desc.shape,
                                     pgrid=sub_grid,
                                     correspondence=corr)

        bs_node = mpi.BlockScatter(name="block_scatter_" + src.data,
                                   subarray_type=subarray,
                                   scatter_grid=sub_grid)
        init_state = mpi.Dummy(subarray, [
            f'MPI_Datatype {subarray};',
            f'int* {subarray}_counts;',
            f'int* {subarray}_displs;',
            f'MPI_Comm {grid}_comm;',
            f'MPI_Group {grid}_group;',
            f'int {grid}_coords[{len(self.grid_size)}];',
            f'int {grid}_dims[{len(self.grid_size)}];',
            f'int {grid}_rank;',
            f'int {grid}_size;',
            f'bool {grid}_valid;',
        ])
        # Pseudo-writing to a dummy variable to avoid removal of Dummy node by transformations.
        _, scal = sdfg.add_scalar(grid, dace.int32, transient=True)
        wnode = state.add_write(grid)
        state.add_node(init_state)
        state.add_edge(init_state, '__out', wnode, None,
                       dace.Memlet.from_array(grid, scal))

        state.add_node(bs_node)
        state.add_edge(src, None, bs_node, "_inp_buffer",
                       sdfg.make_array_memlet(src.data))
        state.add_edge(bs_node, "_out_buffer", dst, None,
                       sdfg.make_array_memlet(dst.data))

        return bs_node
