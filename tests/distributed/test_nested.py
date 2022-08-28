import pytest
import numpy as np

import dace
from dace import nodes
from dace.transformation import interstate
from dace.sdfg import utils as sdfg_utils
import dace.library
from dace.libraries import blas

from daceml.util import utils
from daceml.distributed import schedule
from daceml.distributed.utils import find_map_containing, compile_and_call, arange_with_size


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
    # don't expand the NSDFG, this test tests that
    assert any(
        isinstance(n, nodes.NestedSDFG) for state in sdfg.nodes()
        for n in state.nodes())

    reduce = find_map_containing(sdfg, "reduce_output")
    init = find_map_containing(sdfg, "init")

    schedule.lower(sdfg, {reduce: sizes, init: sizes[:1]})

    X = arange_with_size([16, 16])
    expected = X.copy().sum(axis=1)

    compile_and_call(sdfg, {'x': X.copy()}, expected, utils.prod(sizes))


def test_nested_two_maps():
    @dace
    def nested(x):
        y = np.zeros_like(x, shape=(16, 16))
        for i, j, k in dace.map[0:16, 0:16, 0:32]:
            with dace.tasklet:
                inp << x[i, k, j]
                out = inp
                out >> y(1, lambda x, y: x + y)[i, j]
        return y

    @dace
    def program(x: dace.int64[16, 32, 16]):
        return nested(x)

    sdfg = program.to_sdfg(simplify=False)
    sdfg.apply_transformations_repeated(interstate.StateFusion)
    sdfg.sdfg_list[1].apply_transformations_repeated(interstate.InlineSDFG)
    sdfg.expand_library_nodes()
    # don't expand the NSDFG, this test tests that
    assert any(
        isinstance(n, nodes.NestedSDFG) for state in sdfg.nodes()
        for n in state.nodes())

    elementwise = find_map_containing(sdfg, "test_nested_nested")
    init = find_map_containing(sdfg, "full__map")

    schedule.lower(sdfg, {elementwise: [2, 2, 2], init: [2, 2]})
    sdfg.validate()


@pytest.mark.parametrize(
    "use_gemm",
    [pytest.param(True, id='use_gemm'),
     pytest.param(False, id='use_pure')])
@pytest.mark.parametrize(
    "sizes",
    [
        [1, 1, 1],
        [2, 1, 1],
        [1, 2, 1],
        [1, 2, 2],
        [2, 2, 1],
        # [2, 4, 1], inconsistent
    ])
def test_matmul(sizes, use_gemm):
    @dace
    def matmul(a: dace.float32[40, 40], b: dace.float32[40, 40]):
        return a @ b

    sdfg = matmul.to_sdfg()
    utils.expand_nodes(sdfg, predicate=lambda n: isinstance(n, blas.MatMul))
    if not use_gemm:
        sdfg.expand_library_nodes()
        multiply_map = find_map_containing(sdfg, "gemm_map")
        init_map = find_map_containing(sdfg, "init_map")
        scheme = {multiply_map: sizes, init_map: sizes[:-1]}
    else:
        cands = [
            n for n in sdfg.node(0).nodes()
            if isinstance(n, nodes.LibraryNode)
        ]
        assert len(cands) == 1
        lib = cands[0]
        scheme = {lib: {"gemm_map": sizes, "init_map": sizes[:-1]}}
    schedule.lower(sdfg, scheme)

    A = np.random.rand(40, 40).astype(np.float32)
    B = np.random.rand(40, 40).astype(np.float32)

    expected = A @ B
    with dace.library.change_default(blas, 'OpenBLAS'):
        compile_and_call(sdfg, {
            'a': A.copy(),
            'b': B.copy()
        }, expected, utils.prod(sizes))
