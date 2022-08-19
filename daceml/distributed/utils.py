from typing import List, Iterator

import dace
from dace import SDFG, SDFGState, nodes, dtypes
from dace.libraries import mpi


def initialize_fields(state: SDFGState, fields: List[str]):
    """
    Add a dummy library node to initialize the given fields to the SDFG
    """
    sdfg = state.parent
    dummy = mpi.Dummy("initialize_fields", fields)

    state.add_node(dummy)

    # Pseudo-writing to a dummy variable to avoid removal of Dummy node by transformations.
    dummy_name, scal = sdfg.add_scalar("dummy",
                                       dace.int32,
                                       transient=True,
                                       find_new_name=True)
    wnode = state.add_write(dummy_name)
    state.add_edge(dummy, '__out', wnode, None,
                   dace.Memlet.from_array(dummy_name, scal))


def all_top_level_maps(sdfg: SDFG, yield_parent=False) -> Iterator[nodes.Map]:
    for state in sdfg.nodes():
        for node in state.scope_children()[None]:
            if isinstance(node, nodes.MapEntry):
                if yield_parent:
                    yield node.map, state
                else:
                    yield node.map


def add_debug_rank_cords_tasklet(sdfg: SDFG):
    """
    Add a tasklet to the start of the SDFG that will print out the process coordinates in each process grid.
    """
    new_state = sdfg.add_state_before(sdfg.start_state, 'debug_MPI')

    code = """{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    """

    for grid_name, desc in sdfg.process_grids.items():
        code += """
        if (__state->{grid_name}_rank == 0) {{
            printf("{grid_name} dimensions: ");
            for (int i = 0; i < {grid_dims}; i++) {{
                printf("%d ", __state->{grid_name}_dims[i]);
            }}
            printf("\\n");
        }}

        printf("Hello from global rank %d, rank %d in grid {grid_name}, with coords: ", rank, __state->{grid_name}_rank);

        for (int i = 0; i < {grid_dims}; i++) {{
            printf("%d ", __state->{grid_name}_coords[i]);
        }}
        printf("\\n");
        """.format(grid_name=grid_name, grid_dims=len(desc.shape))
    code += "}"

    # add a tasklet that writes to nothing
    tasklet = nodes.Tasklet("debug_MPI", {}, {"__out"}, code,
                            dtypes.Language.CPP)

    new_state.add_node(tasklet)

    # Pseudo-writing to a dummy variable to avoid removal of Dummy node by transformations.
    dummy_name, scal = sdfg.add_scalar("dummy",
                                       dace.int32,
                                       transient=True,
                                       find_new_name=True)
    wnode = new_state.add_write(dummy_name)
    new_state.add_edge(tasklet, '__out', wnode, None,
                       dace.Memlet.from_array(dummy_name, scal))


def add_debugprint_tasklet(sdfg: SDFG, state: SDFGState,
                           node: nodes.AccessNode):
    """
    Insert a tasklet that just prints out the given data
    """
    desc = node.desc(sdfg)
    loops = "\n".join("for (int i{v} = 0; i{v} < {s}; i{v}++)".format(v=i, s=s)
                      for i, s in enumerate(desc.shape))
    code = """
    {loops}
    {{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    printf("RANK %d: {name}[%d] = %d\\n", rank, {indices}, {name}[{indices}]);
    }}
    """.format(name=node.data,
               loops=loops,
               indices=",".join(
                   ["i{}".format(i) for i in range(len(desc.shape))]))

    # add a tasklet that writes to nothing
    tasklet = nodes.Tasklet("print_" + node.data, {"__in"}, {"__out"}, code,
                            dtypes.Language.CPP)

    state.add_node(tasklet)
    state.add_edge(node, None, tasklet, '__in',
                   sdfg.make_array_memlet(node.data))

    # Pseudo-writing to a dummy variable to avoid removal of Dummy node by transformations.
    dummy_name, scal = sdfg.add_scalar("dummy",
                                       dace.int32,
                                       transient=True,
                                       find_new_name=True)
    state.add_edge(tasklet, '__out', state.add_write(dummy_name), None,
                   dace.Memlet.from_array(dummy_name, scal))
