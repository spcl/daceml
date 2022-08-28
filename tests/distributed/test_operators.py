import numpy as np
import pytest

import dace
from dace.transformation.dataflow import (MapFusion, ReduceExpansion,
                                          TrivialMapElimination, Vectorization,
                                          WarpTiling)
from dace.transformation.interstate import (GPUTransformSDFG, HoistState,
                                            InlineSDFG, StateFusion)
from dace.transformation.subgraph import MultiExpansion, SubgraphFusion
from dace.sdfg import propagation
from dace.transformation.interstate import LoopToMap
from dace.transformation.dataflow import MapFusion, Vectorization, MapCollapse

from daceml.distributed.schedule import lower
from daceml.distributed.utils import find_map_containing, compile_and_call


@dace.program
def softmax_fwd(inp: dace.float32[2, 16, 128, 128]):
    rowmax = np.maximum.reduce(inp, axis=-1, keepdims=True)
    exp_arr = np.exp(inp - rowmax)
    rowsum = np.add.reduce(exp_arr, axis=-1, keepdims=True)
    return exp_arr / rowsum


# Numerically-stable version of softmax
def softmax(x):
    tmp_max = np.max(x, axis=-1, keepdims=True)
    tmp_out = np.exp(x - tmp_max)
    tmp_sum = np.sum(tmp_out, axis=-1, keepdims=True)
    return tmp_out / tmp_sum


@pytest.mark.gpu
def test_warp_softmax(vector_length=1):
    # Get SDFG
    sdfg = softmax_fwd.to_sdfg()

    # Apply transformations
    sdfg.apply_transformations_repeated(ReduceExpansion)
    sdfg.simplify()
    MultiExpansion.apply_to(sdfg, sdfg.node(0).nodes())
    SubgraphFusion.apply_to(sdfg, sdfg.node(0).nodes())
    sdfg.expand_library_nodes()
    sdfg.simplify()
    #sdfg.apply_transformations_repeated([TrivialMapElimination, MapFusion])
    #sdfg.apply_transformations(GPUTransformSDFG)
    #assert sdfg.apply_transformations(WarpTiling) == 1
    #sdfg.apply_transformations_repeated([HoistState, InlineSDFG, StateFusion])
    #sdfg.apply_transformations_repeated([TrivialMapElimination, MapFusion])
    #if vector_length != 1:
    #    sdfg.apply_transformations_repeated(
    #        Vectorization, dict(vector_len=vector_length, preamble=False, postamble=False, strided_map=False))

    to_tile = find_map_containing(sdfg, 'outer_fused')
    lower(sdfg, {to_tile: [2, 2, 1]})

    # Check correctness
    inp = np.random.rand(2, 16, 128, 128).astype(np.float32)
    reg_out = softmax(inp)

    compile_and_call(sdfg,
                     inputs={'inp': inp.copy()},
                     expected_output=reg_out,
                     num_required_ranks=4)


def initialize(M, N, nnz):
    from numpy.random import default_rng
    rng = default_rng(42)

    x = rng.random((N, ))

    from scipy.sparse import random

    matrix = random(M,
                    N,
                    density=nnz / (M * N),
                    format='csr',
                    dtype=np.float64,
                    random_state=rng)
    rows = np.uint32(matrix.indptr)
    cols = np.uint32(matrix.indices)
    vals = matrix.data

    return rows, cols, vals, x


def spmv_numpy(A_row, A_col, A_val, x):
    y = np.empty(A_row.size - 1, A_val.dtype)

    for i in range(A_row.size - 1):
        cols = A_col[A_row[i]:A_row[i + 1]]
        vals = A_val[A_row[i]:A_row[i + 1]]
        y[i] = vals @ x[cols]

    return y


@pytest.mark.skip(
    reason="miscompilation due to bug in dace, must manually fix code")
def test_spmv():
    M = 32
    A_row, A_col, A_val, x = initialize(M, 16, M * 5)
    expected = spmv_numpy(A_row, A_col, A_val, x)

    N = x.shape[0]
    nnz = A_val.shape[0]

    # Matrix-Vector Multiplication with the matrix given in Compressed Sparse Row
    # (CSR) format
    @dace.program
    def spmv(A_row: dace.uint32[M + 1], A_col: dace.uint32[nnz],
             A_val: dace.float64[nnz], x: dace.float64[N]):
        b = np.empty([M], dtype=np.float32)
        for i in dace.map[0:M]:
            # WARNING: need to manually fix a miscompilation in the generated code
            # b[i] = 0 is out of place
            b[i] = 0

            for j in dace.map[A_row[i]:A_row[i + 1]]:
                b[i] += A_val[j] * x[A_col[j]]

        return b

    sdfg = spmv.to_sdfg()
    strict_xforms = dace.transformation.simplification_transformations()

    for sd in sdfg.all_sdfgs_recursive():
        propagation.propagate_states(sd)
    sdfg.apply_transformations_repeated([LoopToMap, MapCollapse] +
                                        strict_xforms)
    schedule = {find_map_containing(sdfg, "test_operators_spmv"): [2]}
    lower(sdfg, schedule)

    compile_and_call(sdfg,
                     inputs=dict(A_row=A_row.copy(),
                                 A_col=A_col.copy(),
                                 A_val=A_val.copy(),
                                 x=x.copy()),
                     expected_output=expected,
                     num_required_ranks=2)
