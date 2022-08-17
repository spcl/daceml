"""
These tests expect to be with 4 ranks
"""
import pytest
import numpy as np

import dace
from dace.sdfg import utils as sdfg_utils

from daceml.util import utils
from daceml.distributed import schedule, utils as distr_utils


def arange_with_size(size):
    return np.arange(utils.prod(size), dtype=np.int64).reshape(size).copy()


@pytest.mark.parametrize("sizes", [
    [2],
    [4],
])
def test_elementwise_1d(sizes):
    assert utils.prod(sizes) <= 4

    @dace
    def program(x: dace.int64[64]):
        return x + 5

    sdfg = program.to_sdfg()

    map_entry = [n for n in sdfg.node(0).scope_children().keys() if n][0]
    schedule.lower(sdfg, {map_entry.map: sizes})

    X = arange_with_size([64])
    expected = X + 5

    from mpi4py import MPI
    commworld = MPI.COMM_WORLD
    rank = commworld.Get_rank()
    size = commworld.Get_size()
    if size < utils.prod(sizes):
        raise ValueError("This test requires at least {} ranks".format(
            utils.prod(sizes)))

    func = sdfg_utils.distributed_compile(sdfg, commworld)

    if rank == 0:
        result = func(x=X.copy())
        np.testing.assert_allclose(result, expected)
    else:
        func(x=np.zeros((1, ), dtype=np.float32))


@pytest.mark.parametrize(
    "sizes",
    [
        [2, 1, 1],  # fully replicate broadcasted => use 2d broadcast grid
        [2, 2, 1],  # no broadcast grid, 1d scatter grid
        [2, 2, 2],  # no broadcast grid, 2d scatter grid
    ])
def test_bcast_simple(sizes):
    @dace
    def program(x: dace.int64[4, 8, 16], y: dace.int64[8, 16]):
        return x + y

    sdfg = program.to_sdfg()
    sdfg.expand_library_nodes()
    sdfg.simplify()

    map_entry = [n for n in sdfg.node(0).scope_children().keys() if n][0]
    schedule.lower(sdfg, {map_entry.map: sizes})

    X = arange_with_size([4, 8, 16])
    Y = arange_with_size([8, 16])
    expected = X + Y

    from mpi4py import MPI
    commworld = MPI.COMM_WORLD
    rank = commworld.Get_rank()
    size = commworld.Get_size()
    if size < utils.prod(sizes):
        raise ValueError("This test requires at least {} ranks".format(
            utils.prod(sizes)))

    func = sdfg_utils.distributed_compile(sdfg, commworld)

    if rank == 0:
        result = func(x=X.copy(), y=Y.copy())
        np.testing.assert_allclose(result, expected)
    else:
        func(x=np.zeros((1, ), dtype=np.int64),
             y=np.zeros((1, ), dtype=np.int64))
