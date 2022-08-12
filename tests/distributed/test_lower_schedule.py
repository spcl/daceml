"""
These tests expect to be with 4 ranks
"""
import pytest
import numpy as np
import dace
import daceml

from daceml.util import utils
from daceml.distributed import schedule


@pytest.mark.parametrize("sizes", [
    [2],
])
def test_elementwise_1d(sizes):
    assert utils.prod(sizes) <= 4

    @dace
    def program(x: dace.float32[64]):
        return np.exp(x)

    sdfg = program.to_sdfg()

    map_entry = [n for n in sdfg.node(0).scope_children().keys() if n][0]
    schedule.lower(sdfg, {map_entry.map: sizes})


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


# TODO
# support views
# support broadcasting
# support redistribution (2 maps)
# support reductions
