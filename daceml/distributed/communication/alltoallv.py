import typing
import copy

from dace import SDFG, SDFGState, subsets, symbolic, data, nodes, dtypes
from dace.transformation import transformation as pm
from dace.libraries import mpi

from daceml.util import utils

if typing.TYPE_CHECKING:
    from .node import DistributedMemlet


class CommunicateScatterv(pm.ExpandTransformation):

    environments = []

    def can_be_applied(self, state: SDFGState, *_, **__):
        node: 'DistributedMemlet' = state.node(
            self.subgraph[type(self)._match_node])

        if len(node.dst_subset) != 1:
            # FIXME implement
            return False

        return True

    @staticmethod
    def expansion(node: 'DistributedMemlet', state: SDFGState, sdfg: SDFG):

        src_desc, dst_desc = node.validate(sdfg, state)

        in_name = state.in_edges(node)[0].data.data
        out_name = state.in_edges(node)[0].data.data
        unique_id = "Scatter_{}_{}_{}_{}_{}".format(sdfg.sdfg_id,
                                                    sdfg.node_id(state),
                                                    state.node_id(node),
                                                    in_name, out_name)

        mpi_dtype = mpi.utils.MPI_DDT(dst_desc.dtype.base_type)
        grid_name = node.dst_pgrid

        subset_start = copy.deepcopy(node.dst_subset[0][0])

        repl_dict = dict(array_shape=', '.join(map(str, src_desc.shape)),
                         strides=', '.join(map(str, src_desc.strides)),
                         array_dim=len(src_desc.shape),
                         subshape=', '.join(map(str, dst_desc.shape)),
                         grid_name=grid_name,
                         grid_dim=len(sdfg.process_grids[grid_name].shape),
                         node_id=unique_id,
                         mpi_dtype=mpi_dtype,
                         ctype=dst_desc.dtype.ctype,
                         subset_vars="\n".join(
                             f"int {v} = rank_coords[{i}];"
                             for i, v in enumerate(node.dst_rank_variables)),
                         subset_expr=symbolic.symstr(subset_start))

        init_code = """
        if(__state->{grid_name}_valid) {{
            int sizes[{array_dim}] = {{{array_shape}}};
            int subsizes[{array_dim}] = {{{subshape}}};
            int array_strides[{array_dim}] = {{{strides}}};
            int origin[{array_dim}] = {{0}};

            // Make subarray type
            MPI_Datatype type;
            MPI_Type_create_subarray({array_dim}, sizes, subsizes, origin, MPI_ORDER_C, {mpi_dtype}, &type);
            MPI_Type_create_resized(type, 0, sizeof({ctype}), &__state->{node_id}_type);
            MPI_Type_commit(&__state->{node_id}_type);
            MPI_Type_free(&type);

            // Compute displacements for {node_id}
            __state->{node_id}_displacements = new int[__state->{grid_name}_size];
            __state->{node_id}_counts = new int[__state->{grid_name}_size];
            for (int i = 0; i < __state->{grid_name}_size; i++) {{
                int rank_coords[{grid_dim}];
                MPI_Cart_coords(__state->{grid_name}_comm, i, {grid_dim}, rank_coords);
                {subset_vars}
                __state->{node_id}_displacements[i] = {subset_expr};
                __state->{node_id}_counts[i] = 1; 
            }}
        }}
        """.format(**repl_dict)

        out_buffer_size = symbolic.symstr(utils.prod(dst_desc.shape))
        code = f"""
            int rank;
            MPI_Comm_rank(MPI_COMM_WORLD, &rank);
            if (__state->{grid_name}_valid) {{
                MPI_Scatterv(
                    _inp_buffer,
                    __state->{unique_id}_counts,
                    __state->{unique_id}_displacements,
                    __state->{unique_id}_type,
                    _out_buffer,
                    {out_buffer_size},
                    {mpi_dtype},
                    0,
                    __state->{grid_name}_comm
                );
            }}
            """
        exit_code = f"""
             if (__state->{grid_name}_valid) {{
                 delete[] __state->{unique_id}_counts;
                 delete[] __state->{unique_id}_displacements;
                 MPI_Type_free(&__state->{unique_id}_type);
             }}
         """
        tasklet = nodes.Tasklet(node.name,
                                node.in_connectors,
                                node.out_connectors,
                                code,
                                language=dtypes.Language.CPP,
                                code_init=init_code,
                                code_exit=exit_code,
                                state_fields=[
                                    f"int *{unique_id}_displacements;",
                                    f"int *{unique_id}_counts;",
                                    f"MPI_Datatype {unique_id}_type;"
                                ])
        tasklet.add_in_connector("_inp_buffer")
        tasklet.add_out_connector("_out_buffer")
        state.in_edges(node)[0].dst_conn = "_inp_buffer"
        state.out_edges(node)[0].src_conn = "_out_buffer"
        return tasklet
