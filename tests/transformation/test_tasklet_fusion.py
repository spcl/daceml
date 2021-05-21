import numpy as np
import dace
from dace.transformation.dataflow import MapFusion

from daceml.transformation import TaskletFusion


def test_basic():
    @dace.program
    def test_basic_tf(A: dace.float32[5, 5]):
        B = A + 1
        return B * 2

    sdfg: dace.SDFG = test_basic_tf.to_sdfg()

    assert sdfg.apply_transformations(MapFusion) == 1
    assert sdfg.apply_transformations(TaskletFusion) == 1

    result = np.empty((5, 5), dtype=np.float32)
    sdfg(A=np.ones_like(result), __return=result)
    assert np.allclose(result, 2 * (np.ones_like(result) + 1))


def test_same_name():
    @dace.program
    def test_same_name(A: dace.float32[5, 5]):
        B = A + 1
        C = A * 3
        return B + C

    sdfg: dace.SDFG = test_same_name.to_sdfg()

    assert sdfg.apply_transformations_repeated(MapFusion) == 2
    assert sdfg.apply_transformations_repeated(TaskletFusion) == 2

    result = np.empty((5, 5), dtype=np.float32)
    A = np.ones_like(result)
    sdfg(A=A, __return=result)
    assert np.allclose(result, A + 1 + A * 3)


def test_same_name_diff_memlet():
    @dace.program
    def test_same_name_diff_memlet(A: dace.float32[5, 5], B: dace.float32[5,
                                                                          5]):
        D = A + 1
        C = B * 3
        return D + C

    sdfg: dace.SDFG = test_same_name_diff_memlet.to_sdfg()

    assert sdfg.apply_transformations_repeated(MapFusion) == 2
    assert sdfg.apply_transformations_repeated(TaskletFusion) == 2

    result = np.empty((5, 5), dtype=np.float32)
    A = np.ones_like(result)
    B = np.ones_like(result) * 2
    sdfg(A=A, B=B, __return=result)
    assert np.allclose(result, A + 1 + B * 3)
