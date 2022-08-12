from typing import Set, Dict, List, Tuple
import copy
import collections

from dace import nodes, SDFG, SDFGState, memlet, symbolic, subsets
from dace.sdfg import propagation
from dace.frontend.common.distr import ShapeType

from daceml.distributed.schedule import MapNodes, NumBlocks
from daceml.util import utils


class CommunicationSolver:
    def __init__(self, sdfg: SDFG, state: SDFGState,
                 rank_variables: Dict[str, str]):
        self.sdfg = sdfg
        self.state = state
        self.rank_variables = rank_variables

    def solve_read(self, nlocal: nodes.AccessNode, nglobal: nodes.AccessNode,
                   subset: subsets.Range):
        """
        Connect the given rank-local view of the global array via MPI communication
        """
        assert nlocal in self.state.nodes()
        assert nglobal in self.state.nodes()

    def solve_write(self, nlocal: nodes.AccessNode, nglobal: nodes.AccessNode,
                    subset: subsets.Range):
        """
        Connect the given rank-local view of the global array via MPI communication
        """
        assert nlocal in self.state.nodes()
        assert nglobal in self.state.nodes()
