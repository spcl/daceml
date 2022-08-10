import dace
import numpy as np
import cost
from dace.transformation.auto import auto_optimize

def test_finite():
    @dace.program
    def add(x: dace.float32[20,20]):
        return x + 1

    sdfg = add.to_sdfg()
    sdfg_cost = cost.sdfg_cost(sdfg)
    assert sdfg_cost < float("inf")


def test_sigmoid():
    @dace.program
    def sigmoid(x: dace.float32[20, 20]):
        return 1/(1+ np.exp(-x))

    sdfg = sigmoid.to_sdfg()
    cost_before = cost.sdfg_cost(sdfg)

    auto_optimize.auto_optimize(sdfg, dace.DeviceType.CPU)
    cost_after = cost.sdfg_cost(sdfg)

    assert cost_before > cost_after



