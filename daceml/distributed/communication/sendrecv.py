import typing
import dace.library
from dace import SDFG, SDFGState, nodes
from dace.transformation import transformation as pm
from dace.libraries import mpi


if typing.TYPE_CHECKING:
    from daceml.distributed.communication.node import SymbolicCommunication



@dace.library.expansion
class ExpandSendRecv(pm.ExpandTransformation):
    environments = [mpi.MPI]

    @staticmethod
    def expansion(node: 'SymbolicCommunication', state: SDFGState, sdfg: SDFG):
        inp_buffer, out_buffer = node.validate(sdfg, state)
        mpi_dtype_str = mpi.utils.MPI_DDT(out_buffer.dtype.base_type)

        dst_grid = sdfg.process_grids[node.dst_pgrid]

        if inp_buffer.dtype.veclen > 1:
            raise NotImplementedError

        if node.src_pgrid is None:
            loop_for_dim = lambda i: f"for (int i{i} = 0; i < {dst_grid.shape[i]} ; i{i}++)"
            n_dst_grid = len(dst_grid.shape)
            loop_stack = "\n".join(loop_for_dim(i) for i in range(n_dst_grid))
            send_code = \
            f"""
            if (rank == 0 ) {{
                {loop_stack} {{
                    // Get rank index in dst grid
                    int dst_rank = 0;
                    int coords[] = {{{','.join(f"i{i}" for i in range(n_dst_grid))}}};
                    MPI_Cart_rank(__state->{node.dst_pgrid}, coords, &dst_rank);
                    // Create subarray
                    MPI_Datatype subarray;



                    // Send data to dst rank
                    MPI_Isend({inp_buffer.name} + i{node.src_rank_variables[0]}, 1, {mpi_dtype_str}, dst_rank, 0, __state->{node.dst_pgrid});
                }}
            }}
            """
        else:
            raise NotImplementedError()


        if node.dst_pgrid is None:
            raise NotImplementedError()


        recv_code =  \
        f"""
        """


        tasklet = nodes.Tasklet(node.name, node.in_connectors,
                node.out_connectors, code, language=dtypes.Language.CPP)
        return tasklet








