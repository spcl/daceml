from dace import nodes

class CommunicationSolver:

    def __init__(self, map_node: nodes.Map):
        self.map_node = map_node

        # pgrid = sdfg.add_pgrid(shape=block_sizes)

    def solve_read(self, nlocal: nodes.AccessNode, nglobal: nodes.AccessNode):
        """
        Connect the given rank-local view of the global array via MPI communication
        """

    def solve_write(self, nlocal: nodes.AccessNode, nglobal: nodes.AccessNode):
        """
        Connect the given rank-local view of the global array via MPI communication
        """

