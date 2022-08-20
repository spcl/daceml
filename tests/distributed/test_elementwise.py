"""
These tests expect to be with 4 ranks
"""
import pytest
import numpy as np

import dace

from daceml.util import utils
from daceml.distributed import schedule
from daceml.distributed.utils import find_map_containing, compile_and_call, arange_with_size


@pytest.mark.parametrize("sizes", [
    [2],
    [4],
])
def test_elementwise_1d(sizes):
    @dace
    def program(x: dace.int64[64]):
        return x + 5

    sdfg = program.to_sdfg()

    map_entry = find_map_containing(sdfg, "")
    schedule.lower(sdfg, {map_entry: sizes})

    X = arange_with_size([64])
    expected = X + 5
    compile_and_call(sdfg, {'x': X.copy()}, expected, utils.prod(sizes))


@pytest.mark.parametrize("sizes", [
    [2],
])
def test_1d_with_empty_dim(sizes):
    @dace
    def program(x: dace.int64[64, 1]):
        y = np.empty_like(x, shape=(64, ))
        for i in dace.map[0:64]:
            with dace.tasklet:
                inp << x[i, 0]
                out = inp + 5
                out >> y[i]
        return y

    sdfg = program.to_sdfg()

    map_entry = find_map_containing(sdfg, "")
    schedule.lower(sdfg, {map_entry: sizes})

    X = arange_with_size([64, 1])
    expected = (X + 5).reshape(64)
    compile_and_call(sdfg, {'x': X.copy()}, expected, utils.prod(sizes))


@pytest.mark.parametrize(
    "sizes",
    [
        [2, 1, 1],  # fully replicate broadcasted => use 2d broadcast grid
        [2, 2, 1],  # no broadcast grid, 1d scatter grid
        [2, 2, 2],  # no broadcast grid, 2d scatter grid
        [1, 2, 1],
    ])
def test_bcast_simple(sizes):
    @dace
    def program(x: dace.int64[4, 8, 16], y: dace.int64[8, 16]):
        return x + y

    sdfg = program.to_sdfg()
    sdfg.expand_library_nodes()
    sdfg.simplify()

    map_entry = find_map_containing(sdfg, "")
    schedule.lower(sdfg, {map_entry: sizes})

    X = arange_with_size([4, 8, 16])
    Y = arange_with_size([8, 16])
    expected = X + Y

    compile_and_call(sdfg, {
        'x': X.copy(),
        'y': Y.copy()
    }, expected, utils.prod(sizes))
