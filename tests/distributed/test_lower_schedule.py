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


# TODO
# support views
# support broadcasting
# support redistribution (2 maps)
# support reductions
