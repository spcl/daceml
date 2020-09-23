import pytest

import dace
import dace.data as dt

from daceml.autodiff.backward_pass_generator import is_initialization_state


@pytest.mark.parametrize("shape", [[1], [2, 3], [5, 1]])
@pytest.mark.parametrize("value", [0, 1])
@pytest.mark.parametrize("scalar", [False, True])
def test_is_initialization_state_single(shape, value, scalar):
    sdfg = dace.SDFG("test_is_initialization_state")
    state = sdfg.add_state()

    if scalar:
        sdfg.add_scalar("X", dace.float32)

        tasklet = state.add_tasklet("_init_X_", {}, {"__out"},
                                    "__out = {}".format(value))
        write = state.add_write("X")
        state.add_edge(tasklet, "__out", write, None,
                       dace.Memlet.simple("X", "0"))
    else:
        sdfg.add_array("X", shape, dace.float32)
        arr = sdfg.arrays["X"]
        state.add_mapped_tasklet(
            "_init_X_",
            {
                "i{}".format(i): "0:{}".format(shape)
                for i, shape in enumerate(arr.shape)
            },
            {},
            "__out = {}".format(value),
            {
                "__out":
                dace.Memlet.simple(
                    "X", ", ".join("i{}".format(i)
                                   for i in range(len(arr.shape))))
            },
            external_edges=True,
        )
    if value == 0:
        assert is_initialization_state(state)
    else:
        assert not is_initialization_state(state)


@pytest.mark.parametrize("value", [0, 1])
def test_is_initialization_state_multiple_disconn(value):
    sdfg = dace.SDFG("test_is_initialization_state")
    state = sdfg.add_state()

    # add a scalar
    sdfg.add_scalar("Y", dace.float32)

    tasklet = state.add_tasklet("_init_Y_", {}, {"__out"},
                                "__out = {}".format(value))
    write = state.add_write("Y")
    state.add_edge(tasklet, "__out", write, None, dace.Memlet.simple("Y", "0"))

    # init multiple arrays in one tasklet
    shape = [5, 3]
    sdfg.add_array("A", shape, dace.float32)
    sdfg.add_array("B", shape, dace.float32)
    arr = sdfg.arrays["A"]
    state.add_mapped_tasklet(
        "_init_X_",
        {
            "i{}".format(i): "0:{}".format(shape)
            for i, shape in enumerate(arr.shape)
        },
        {},
        "__out0 = {}\n__out1 = {}".format(value, value),
        {
            "__out0":
            dace.Memlet.simple(
                "A", ", ".join("i{}".format(i)
                               for i in range(len(arr.shape)))),
            "__out1":
            dace.Memlet.simple(
                "B", ", ".join("i{}".format(i) for i in range(len(arr.shape))))
        },
        external_edges=True,
    )

    if value == 0:
        assert is_initialization_state(state)
    else:
        assert not is_initialization_state(state)
