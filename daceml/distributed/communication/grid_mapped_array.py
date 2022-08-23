"""
A grid mapped array is an array mapped onto a process grid.
This module contains nodes to scatter and gather these arrays onto process
grids.

This is similar to using subarrays in dace, with two differences:
1. the interface is less flexible: this is not meant to abstract the MPI
   "Subarray" type like dace's Subarray.
2. It has better support for permuted layouts. The dace nodes currently fail in
   some configurations when the axes of the array and process grid are out of order.
"""
from typing import List, Optional

import aenum
import sympy as sp

import dace.library
from dace.libraries import mpi
from dace import nodes, properties, SDFG, SDFGState, transformation, dtypes, symbolic, registry

from daceml.util import utils


@registry.undefined_safe_enum
@registry.extensible_enum
class AxisType(aenum.AutoNumberEnum):
    REPLICATE = ()
    BROADCAST = ()
    PARTITION = ()


class AxisScheme:
    """
    This class is used to specifies how axes of an array map onto the process
    grid.
    Each process grid dimension has an associated AxisScheme object. For each
    dimension, there are three options:

    * :class:`AxisScheme(scheme=AxisType.PARTITION, axis=i)`: The axis ``i`` of
      the array is mapped onto this process grid axis, and partitioned. If
      ``d`` is the dimension of this process grid dimensions, then the array
      will be split into ``d`` blocks along this axis.

    * :class:`AxisScheme(scheme=AxisType.REPLICATE, axis=i)`: The axis ``i`` of
       the array is mapped onto this process grid axis, and replicated. Each
       process will recieve the whole array axis.

    * :class:`AxisScheme(scheme=AxisType.BROADCAST, axis=None)`: The array is
       broadcasted to all processes in this process grid axis. Use this when
       an array has less dimensions than the process grid.
    """
    def __init__(self, axis, scheme):
        self.scheme = scheme
        if scheme == AxisType.BROADCAST and axis is not None:
            raise ValueError('Broadcasting axis must be None')
        self.axis = axis

    def to_json(self):
        return {'axis': self.axis, 'scheme': self.scheme._name_}

    def __str__(self):
        return f'{self.scheme.name}:{self.axis}'

    def __repr__(self):
        return f'AxisScheme(scheme={self.scheme.name}, axis={self.axis})'

    @staticmethod
    def from_json(data):
        if isinstance(data['scheme'], AxisType):
            scheme = scheme
        else:
            scheme = AxisType[data['scheme']]

        if isinstance(data['axis'], int) or data['axis'] is None:
            axis = data['axis']
        else:
            axis = int(data['axis'])
        return AxisScheme(axis, scheme)


class GridMapper(nodes.LibraryNode):
    """
    Abstract class for grid mapping nodes.

    This class shouldn't be used directly (use ScatterOntoGrid or GatherFromGrid).
    """

    grid_name = properties.Property(dtype=str,
                                    desc="The grid to scatter onto.")
    axis_mapping = properties.ListProperty(element_type=AxisScheme,
                                           desc="The axis mapping.")

    def __init__(self, name, grid_name, axis_mapping):
        super().__init__(name)
        self.grid_name = grid_name
        self.axis_mapping = axis_mapping

    def validate(self, sdfg: SDFG, state: SDFGState):
        super().validate(sdfg, state)

        # get inputs and outputs
        inp_desc, out_desc = None, None
        for e in state.out_edges(self):
            if e.src_conn == "_out_buffer":
                out_desc = sdfg.arrays[e.data.data]
        for e in state.in_edges(self):
            if e.dst_conn == "_inp_buffer":
                inp_desc = sdfg.arrays[e.data.data]
        if inp_desc is None or out_desc is None:
            raise ValueError("must have input and output buffers.")

        if out_desc.dtype != inp_desc.dtype:
            raise ValueError("input and output arrays must have the same type")

        pgrid_desc = sdfg.process_grids[self.grid_name]
        if len(inp_desc.shape) != len(out_desc.shape):
            raise ValueError(
                "Input and output arrays must have the same number of dimensions."
            )

        # this node is either a scatter or a gather
        scatter = isinstance(self, ScatterOntoGrid)

        if len(self.axis_mapping) != len(pgrid_desc.shape):
            raise ValueError(
                "Axis mapping must have the same number of dimensions as the process grid."
            )

        if scatter:
            arr_desc = inp_desc
            sub_desc = out_desc
        else:
            arr_desc = out_desc
            sub_desc = inp_desc

        for grid_shape, axis_scheme in zip(pgrid_desc.shape,
                                           self.axis_mapping):
            if axis_scheme.scheme == AxisType.PARTITION:
                arr_dim = axis_scheme.axis
                if arr_dim < 0 or arr_dim >= len(inp_desc.shape):
                    raise ValueError("Axis index out of bounds")

                arr_shape = arr_desc.shape[arr_dim]
                subshape = sub_desc.shape[arr_dim]

                if sp.Mod(arr_shape, grid_shape).simplify() != 0:
                    raise ValueError(
                        "Array shape {} is not divisible by grid shape {}.".
                        format(arr_shape, grid_shape))
                if symbolic.inequal_symbols(arr_shape / grid_shape, subshape):
                    raise ValueError(
                        "Output subshape is wrong: {}/{} != {}".format(
                            arr_shape, grid_shape, subshape))
            elif axis_scheme.scheme == AxisType.REPLICATE:
                arr_dim = axis_scheme.axis
                if arr_dim < 0 or arr_dim >= len(inp_desc.shape):
                    raise ValueError("Axis index out of bounds")

                arr_shape = arr_desc.shape[arr_dim]
                subshape = sub_desc.shape[arr_dim]

                if arr_shape != subshape:
                    raise ValueError(
                        "Output subshape is wrong: {} != {}".format(
                            arr_shape, subshape))
            elif axis_scheme.scheme == AxisType.BROADCAST and axis_scheme.axis is not None:
                raise ValueError("Broadcasting axis must be None")

        return inp_desc, out_desc


class Expand(transformation.ExpandTransformation):

    environments = [mpi.environments.MPI]

    @staticmethod
    def expansion(node: GridMapper, state: SDFGState, sdfg: SDFG):

        # this node is either a scatter or a gather
        scatter = isinstance(node, ScatterOntoGrid)

        inp_desc, out_desc = node.validate(sdfg, state)
        fields_to_add = []

        pgrid_desc = sdfg.process_grids[node.grid_name]

        partition_color = [
            ax_s.scheme == AxisType.PARTITION for ax_s in node.axis_mapping
        ]

        need_replication = not all(partition_color)

        if need_replication:
            replication_color = list(map(lambda x: not x, partition_color))

            partition_grid_name = sdfg.add_pgrid(parent_grid=node.grid_name,
                                                 color=partition_color,
                                                 exact_grid=0)
            replication_grid_name = sdfg.add_pgrid(parent_grid=node.grid_name,
                                                   color=replication_color)

            for name in (partition_grid_name, replication_grid_name):
                shape = sdfg.process_grids[name].shape
                fields_to_add.extend([
                    f'MPI_Comm {name}_comm;',
                    f'MPI_Group {name}_group;',
                    f'int {name}_coords[{len(shape)}];',
                    f'int {name}_dims[{len(shape)}];',
                    f'int {name}_rank;',
                    f'int {name}_size;',
                    f'bool {name}_valid;',
                ])
        else:
            partition_grid_name = node.grid_name
            replication_grid_name = None

        pgrid_desc = sdfg.process_grids[partition_grid_name]

        in_name = state.in_edges(node)[0].data.data
        out_name = state.in_edges(node)[0].data.data
        unique_id = "{}_{}_{}_{}_{}_{}".format(
            "Scatter" if scatter else "Gather", sdfg.sdfg_id,
            sdfg.node_id(state), state.node_id(node), in_name, out_name)
        mpi_dtype = mpi.utils.MPI_DDT(out_desc.dtype.base_type)

        array_desc = inp_desc if scatter else out_desc
        subarray_desc = out_desc if scatter else inp_desc

        # we need to invert the permutation here so that
        # inverse_perm[i] is the process_grid axis of the i-th array axis
        process_grid_to_array_axis: List[Optional[int]] = [
            s.axis for s in node.axis_mapping
        ]
        inverse_perm = [
            process_grid_to_array_axis.index(l)
            for l in range(len(array_desc.shape))
        ]

        replicated_dims = {
            s.axis
            for s in node.axis_mapping if s.scheme == AxisType.REPLICATE
        }

        repl_dict = dict(array_shape=', '.join(map(str, array_desc.shape)),
                         strides=', '.join(map(str, array_desc.strides)),
                         array_dim=len(array_desc.shape),
                         subshape=', '.join(map(str, subarray_desc.shape)),
                         subshape_without_replicas=', '.join(
                             '0' if i in replicated_dims else str(s)
                             for i, s in enumerate(subarray_desc.shape)),
                         axis_mapping=', '.join(map(str, inverse_perm)),
                         parent_grid_name=node.grid_name,
                         parent_grid_dim=len(
                             sdfg.process_grids[node.grid_name].shape),
                         pgrid_name=partition_grid_name,
                         pgrid_dim=len(pgrid_desc.shape),
                         node_id=unique_id,
                         mpi_dtype=mpi_dtype,
                         ctype=out_desc.dtype.ctype)

        init_code = """
        if(__state->{pgrid_name}_valid) {{

            int sizes[{array_dim}] = {{{array_shape}}};
            int subsizes[{array_dim}] = {{{subshape}}};
            int subsizes_for_striding[{array_dim}] = {{{subshape_without_replicas}}};
            int origin[{array_dim}] = {{0}};
            int array_strides[{array_dim}] = {{{strides}}};
            int axis_mapping[{array_dim}] = {{{axis_mapping}}};

            // Make subarray type
            MPI_Datatype type;
            MPI_Type_create_subarray({array_dim}, sizes, subsizes, origin, MPI_ORDER_C, {mpi_dtype}, &type);
            MPI_Type_create_resized(type, 0, sizeof({ctype}), &__state->{node_id}_type);
            MPI_Type_commit(&__state->{node_id}_type);
            MPI_Type_free(&type);

            // Compute displacements for {node_id}
            __state->{node_id}_displacements = new int[__state->{pgrid_name}_size];
            __state->{node_id}_counts = new int[__state->{pgrid_name}_size];
            for (int i = 0; i < __state->{pgrid_name}_size; i++) {{
                int ranks1[1] = {{i}};
                int ranks2[1];
                MPI_Group_translate_ranks(__state->{pgrid_name}_group, 1, ranks1, __state->{parent_grid_name}_group, ranks2);

                int rank_coords[{parent_grid_dim}];
                MPI_Cart_coords(__state->{parent_grid_name}_comm, ranks2[0], {parent_grid_dim}, rank_coords);

                // rank coords, but indexed using the axis_mapping.
                int rank_coords_array[{array_dim}];
                for (int j = 0; j < {array_dim}; j++) {{
                    rank_coords_array[j] = rank_coords[axis_mapping[j]];
                }}

                __state->{node_id}_displacements[i] = 0;
                for (int j = 0; j < {array_dim}; j++) {{
                    __state->{node_id}_displacements[i] += rank_coords_array[j] * subsizes_for_striding[j] * array_strides[j];
                }}
                __state->{node_id}_counts[i] = 1;
            }}
        }}
        """.format(**repl_dict)

        out_buffer_size = symbolic.symstr(utils.prod(out_desc.shape))
        if scatter:
            code = f"""
                int rank;
                MPI_Comm_rank(MPI_COMM_WORLD, &rank);
                printf("rank %d scatter %d\\n", rank, __state->{partition_grid_name}_valid);
                if (__state->{partition_grid_name}_valid) {{
                    MPI_Scatterv(
                        _inp_buffer,
                        __state->{unique_id}_counts,
                        __state->{unique_id}_displacements,
                        __state->{unique_id}_type,
                        _out_buffer,
                        {out_buffer_size},
                        {mpi_dtype},
                        0,
                        __state->{partition_grid_name}_comm
                    );
                }}
                """
            if replication_grid_name is not None:
                code += f"""
                printf("rank %d bcast %d\\n", rank, __state->{replication_grid_name}_valid);
                if(__state->{replication_grid_name}_valid) {{
                   MPI_Bcast(_out_buffer, {out_buffer_size}, {mpi_dtype}, 0, __state->{replication_grid_name}_comm);
                }}
                """
        else:
            inp_buffer_size = symbolic.symstr(utils.prod(inp_desc.shape))
            if replication_grid_name is not None:
                code = f"""
                    if (__state->{partition_grid_name}_valid) {{
                        MPI_Reduce(MPI_IN_PLACE, _inp_buffer, {inp_buffer_size}, {mpi_dtype}, MPI_SUM, __state->{replication_grid_name}_rank, __state->{replication_grid_name}_comm);
                        MPI_Gatherv(
                            _inp_buffer,
                            {inp_buffer_size},
                            {mpi_dtype},
                            _out_buffer,
                            __state->{unique_id}_counts,
                            __state->{unique_id}_displacements,
                            __state->{unique_id}_type,
                            0,
                            __state->{partition_grid_name}_comm
                        );
                    }} else if (__state->{replication_grid_name}_valid) {{
                        MPI_Reduce(
                            _inp_buffer,
                            _inp_buffer,
                            {inp_buffer_size},
                            {mpi_dtype},
                            MPI_SUM,
                            0,
                            __state->{replication_grid_name}_comm
                        );
                    }}
                """
            else:
                code = f"""
                    if (__state->{partition_grid_name}_valid) {{
                        MPI_Gatherv(_inp_buffer,
                            {inp_buffer_size},
                            {mpi_dtype},
                            _out_buffer,
                            __state->{unique_id}_counts,
                            __state->{unique_id}_displacements,
                            __state->{unique_id}_type,
                            0,
                            __state->{partition_grid_name}_comm
                        );
                    }}
                """

        exit_code = f"""
             if (__state->{partition_grid_name}_valid) {{
                 delete[] __state->{unique_id}_counts;
                 delete[] __state->{unique_id}_displacements;
                 MPI_Type_free(&__state->{unique_id}_type);
             }}
         """

        fields_to_add.extend([
            f"int *{unique_id}_displacements;", f"int *{unique_id}_counts;",
            f"MPI_Datatype {unique_id}_type;"
        ])
        tasklet = nodes.Tasklet(node.name,
                                node.in_connectors,
                                node.out_connectors,
                                code,
                                language=dtypes.Language.CPP,
                                code_init=init_code,
                                code_exit=exit_code,
                                state_fields=fields_to_add)
        return tasklet


@dace.library.node
class ScatterOntoGrid(GridMapper):
    """
    Scatters the input array onto the process grid, splitting it into blocks as
    required.

    ``axis_mapping`` is a list with the same size as the process grid
    dimension, specifying how the array is mapped onto each axis of the process
    grid. See :class:`AxisScheme` for more details.
    """
    # Global properties
    implementations = {
        "mpi": Expand,
    }

    default_implementation = "mpi"
    grid_name = properties.Property(dtype=str,
                                    desc="The grid to scatter onto.")
    axis_mapping = properties.ListProperty(element_type=AxisScheme,
                                           desc="The axis mapping.")


@dace.library.node
class GatherFromGrid(GridMapper):
    """
    Gathers the input array from the process grid. Inverse of ScatterOntoGrid.

    ``axis_mapping`` is a list with the same size as the process grid
    dimension, specifying how the array is mapped onto each axis of the process
    grid. See :class:`AxisScheme` for more details.
    """
    # Global properties
    implementations = {
        "mpi": Expand,
    }

    default_implementation = "mpi"
    grid_name = properties.Property(dtype=str,
                                    desc="The grid to scatter onto.")
    axis_mapping = properties.ListProperty(element_type=AxisScheme,
                                           desc="The axis mapping.")
