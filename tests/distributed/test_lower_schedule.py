"""
These tests expect to be with 4 ranks
"""
import pytest
import numpy as np

import dace
from dace.transformation.dataflow import ReduceExpansion 
from dace.transformation.interstate import HoistState, InlineSDFG 
from dace.transformation.auto.auto_optimize import greedy_fuse
from dace.sdfg import utils as sdfg_utils

import daceml
from daceml.util import utils
from daceml.distributed import schedule, utils as distr_utils
from daceml import onnx as donnx

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
        func(x=np.zeros((1,), dtype=np.float32))

@pytest.mark.parametrize("sizes", [
    [2, 1, 1], # fully replicate broadcasted => use 2d broadcast grid
    [2, 2, 1], # no broadcast grid, 1d scatter grid
    [2, 2, 2], # no broadcast grid, 2d scatter grid
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
    distr_utils.add_debug_rank_cords_tasklet(sdfg)

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
        func(x=np.zeros((1,), dtype=np.int64), y=np.zeros((1,), dtype=np.int64))

@pytest.mark.parametrize("sizes", [
])
def test_reduce_simple(sizes):
    assert utils.prod(sizes) <= 4

    @dace
    def program(x: dace.int64[64, 16]):
        return np.add.reduce(x, axis=1)

    sdfg = program.to_sdfg()
    sdfg.expand_library_nodes()
    sdfg.simplify()

    plan = {m: sizes[:len(m.range)] for m in  distr_utils.all_top_level_maps(sdfg)}
    schedule.lower(sdfg, plan)
    breakpoint()

    X = arange_with_size([64, 16])
    expected = np.add.reduce(X, axis=1)

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
        func(x=np.zeros((1,), dtype=np.float32))


@pytest.mark.parametrize("sizes", [
    [2],
])
def test_elementwise_1d_two(sizes):
    assert utils.prod(sizes) <= 4

    @dace
    def program(x: dace.float32[64]):
        y = np.exp(x)
        return y + 1

    sdfg = program.to_sdfg()

    map_entries = [n for n in sdfg.node(0).scope_children().keys() if n]
    sched = {me.map: sizes for me in map_entries}
    schedule.lower(sdfg, sched)

    sdfg.validate()
    sdfg.expand_library_nodes()

@pytest.mark.parametrize("sizes", [
    [2],
])
def test_elementwise_1d_shared_read_write(sizes):
    assert utils.prod(sizes) <= 4

    @dace
    def program(x: dace.float32[64]):
        y = x + 1
        z = np.exp(x)

        return y + z

    sdfg = program.to_sdfg()

    map_entries = [n for n in sdfg.node(0).scope_children().keys() if n]
    sched = {me.map: sizes for me in map_entries}
    schedule.lower(sdfg, sched)

    sdfg.validate()




@pytest.mark.parametrize("sizes", [
    [2, 1, 1],
    [1, 1, 1],
    [2, 2, 1],
    [2, 1, 2],
    [4, 1, 1],
])
def test_elementwise(sizes):
    assert utils.prod(sizes) <= 4

    @dace
    def program(x: dace.float32[64, 64, 64]):
        return np.exp(x)

    sdfg = program.to_sdfg()

    map_entry = [n for n in sdfg.node(0).scope_children().keys() if n][0]
    schedule.lower(sdfg, {map_entry.map: sizes})


def test_rank_different_sizes():
    @dace
    def program(X: dace.float32[20, 10]):
        for i, j in dace.map[0:20, 0:10]:
            for k in dace.map[0:j + 1]:
                X[i, k] = X[i, k] + 1

    # b00 has size [0:10, 0:5]
    # b01 has size [0:10, 5:10]

    sdfg = program.to_sdfg()

    map_entry = [n for n in sdfg.node(0).scope_children().keys() if n][0]
    schedule.lower(sdfg, {map_entry.map: [2, 2]})


def test_softmax():
    @dace.program
    def program(X: dace.float32[20, 10]):
        output = np.zeros_like(X)
        donnx.ONNXSoftmax(input=X, output=output)
        return output

    sdfg = program.to_sdfg()
    sdfg.expand_library_nodes()
    sdfg.apply_transformations_repeated(HoistState)
    sdfg.apply_transformations_repeated(InlineSDFG)
    sdfg.apply_transformations_repeated(ReduceExpansion)
    greedy_fuse(sdfg, False, permutations_only=False, recursive=False)
    sdfg.view(8000)

    map_entry = [n for n in sdfg.node(0).scope_children().keys() if n][0]
    schedule.lower(sdfg, {map_entry.map: [2, 2]})



# TODO
# support views
# support broadcasting
# support redistribution (2 maps)
# support reductions
