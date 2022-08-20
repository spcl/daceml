"""
These tests expect to be with 4 ranks
"""
import pytest
import numpy as np
from typing import Sequence, Dict

import dace
from dace import nodes
from dace.sdfg import utils as sdfg_utils

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


@pytest.mark.parametrize(
    "sizes",
    [
        [1, 1],
        [2, 1],
        [2, 2],  # parallelize along the reduction axis with MPI reduce
        [2, 4],  # parallelize along the reduction axis with MPI reduce
    ])
def test_reduce_simple(sizes):
    @dace
    def program(x: dace.int64[16, 16]):
        return np.add.reduce(x, axis=1)

    sdfg = program.to_sdfg()
    sdfg.expand_library_nodes()
    sdfg.simplify()

    reduce = find_map_containing(sdfg, "reduce_output")

    # only schedule the reduce map.
    # the init map will be derived
    schedule.lower(sdfg, {reduce: sizes})

    X = arange_with_size([16, 16])
    expected = X.copy().sum(axis=1)

    compile_and_call(sdfg, {'x': X.copy()}, expected, utils.prod(sizes))
