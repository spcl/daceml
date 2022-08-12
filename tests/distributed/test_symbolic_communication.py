"""
Check that the symbolic communication subsets are correct.
"""
import copy

import pytest
import numpy as np

import dace
from dace import SDFG, SDFGState, symbolic, subsets
from dace.libraries import mpi
from dace.sdfg.utils import distributed_compile

from daceml.distributed.communication_solver import BlockScatterSolution
from daceml.distributed.communication.node import SymbolicCommunication
from daceml.distributed.communication.subarrays import try_match_blocks
from daceml.util import utils

def arange_with_size(size):
   return np.arange(utils.prod(size), dtype=np.int64).reshape(size).copy()


def index_using_subset(X: np.ndarray, sub: subsets.Range) -> np.ndarray:
    assert not sub.free_symbols
    return X[tuple(map(lambda x: slice(x[0], x[1] + 1, x[2]), sub.ndrange()))]


@pytest.mark.parametrize("subset, expected", [
    ("32 * i: 32 * (i + 1)", [("i", 32)]),
    ("32 * i: 32 * (i + 1), 16 * j: 16 * (j + 1)", [("i", 32), ("j", 16)]),
    ("32 * i: 32 * (i + 1), 0:64", [("i", 32), (None, 64)]),
    ("10*i:10*i + 10, 0:5*j + 5", None),
])
def test_detect_blocks(subset, expected):
    block_variables = list(map(symbolic.pystr_to_symbolic, ["i", "j", "k"]))

    subset = subsets.Range.from_string(subset)
    result = try_match_blocks(subset, block_variables)

    # map expected values to symbols
    if expected:
        expected = [(symbolic.pystr_to_symbolic(s) if s else s, bs)  for s, bs in expected]
    assert result == expected


@pytest.mark.parametrize("array_size, grid_size, correspondence", [
    ((32, 64), (2, 2), [0, 1]),
])
def test_block_scatter(grid_size, array_size, correspondence, sdfg_name):

    grid_variables = [f"i{i}" for i in range(len(grid_size))]
    solution = BlockScatterSolution(grid_variables=grid_variables,
                                    array_size=array_size,
                                    grid_size=grid_size,
                                    correspondence=correspondence)

    subset = solution.compute_subset()
    local_shape = [
        symbolic.overapproximate(r).simplify() for r in subset.size_exact()
    ]

    sdfg = SDFG(sdfg_name)
    state = sdfg.add_state()
    sdfg.add_array("X", array_size, dace.int64)
    sdfg.add_array(
        "X_local",
        shape=local_shape,
        dtype=dace.int64,
    )
    solution.insert_node(state, state.add_access("X"),
                         state.add_access("X_local"))

    from mpi4py import MPI
    commworld = MPI.COMM_WORLD
    rank = commworld.Get_rank()
    size = commworld.Get_size()
    if size < utils.prod(grid_size):
        raise ValueError("This test requires at least {} ranks".format(
            utils.prod(grid_size)))

    commcart = commworld.Create_cart(dims=grid_size)

    repl_dict = {
        v: k
        for v, k in zip(grid_variables, commcart.Get_coords(rank))
    }
    local_subset = copy.deepcopy(subset)
    local_subset.replace(repl_dict)

    reference_X = np.arange(utils.prod(array_size),
                            dtype=np.int64).reshape(array_size).copy()
    expected_local = index_using_subset(reference_X, local_subset)

    func = distributed_compile(sdfg, commworld)
    X_local = np.zeros(local_shape, dtype=np.int64)
    if rank == 0:
        X = reference_X
    else:
        X = np.zeros(array_size, dtype=np.int64)

    func(X=X, X_local=X_local)

    np.testing.assert_allclose(X_local, expected_local)


@pytest.mark.parametrize("array_size, src_grid, dst_grid, src_subset, dst_subset", 
    [
        ((32, 64), (0,), (2, 1), "0:32, 0:64", "i0*16:(i0+1)*16, 0:64"),
        #((32, 64), (2, 1), (1, 2), "i0*16: (i0 + 1) * 16,:", ":, i1*32: (i1 + 1) * 32"),
    ]
)
def test_symbolic_communication_node(sdfg_name, array_size, src_grid, dst_grid, src_subset, dst_subset):
    # Parse src and dst subsets
    src_subset = subsets.Range.from_string(src_subset)
    dst_subset = subsets.Range.from_string(dst_subset)


    # process grid shape (0,) implies no process grid
    sdfg = SDFG(sdfg_name)
    state = sdfg.add_state()
    sdfg.add_array("src", src_subset.size_exact(), dace.int64)
    sdfg.add_array("dst", dst_subset.size_exact(), dace.int64)

    # Add process grids
    if src_grid != (0,):
        src_grid = sdfg.add_pgrid(src_grid)
        src_pgrid_vars = [f"i{i}" for i in range(len(src_grid))]
    else:
        src_grid = None
        src_pgrid_vars = []
    if dst_grid != (0,):
        dst_grid = sdfg.add_pgrid(dst_grid)
        dst_pgrid_vars = [f"i{i}" for i in range(len(dst_grid))]
    else:
        dst_grid = None
        dst_pgrid_vars = []


    node = SymbolicCommunication(
        "test",
        src_rank_variables=src_pgrid_vars,
        src_pgrid=src_grid,
        src_subset=src_subset,
        dst_rank_variables=dst_pgrid_vars,
        dst_pgrid=dst_grid,
        dst_subset=dst_subset,
    )

    state.add_node(node)
    state.add_edge(state.add_access("src"), None, node, None, sdfg.make_array_memlet("src"))
    state.add_edge(node, None, state.add_access("dst"), None, sdfg.make_array_memlet("dst"))

    sdfg.validate()


    # start MPI code
    from mpi4py import MPI
    commworld = MPI.COMM_WORLD
    rank = commworld.Get_rank()
    size = commworld.Get_size()
    grid_ranks = lambda x: utils.prod(sdfg.process_grids[x].shape) if x else 1
    req_ranks = max(map(grid_ranks, (src_grid, dst_grid)))

    if size < req_ranks:
        pass
        # raise ValueError(f"This test requires at least {req_ranks} ranks"))

    func = distributed_compile(sdfg, commworld)

    # setup buffers
    if src_grid is None:
        # src array exists only on rank 0
        if rank == 0:
            src_npy = arange_with_size(src_subset.size_exact())
        else:
            src_npy = np.zeros((1,), dtype=np.int64)
    else:
        src_npy = arange_with_size(src_subset.size_exact()) + 1000000 * rank

    dst_npy = np.empty(dst_subset.size_exact(), dtype=np.int64)

    func(src=src_npy, dst=dst_npy)

