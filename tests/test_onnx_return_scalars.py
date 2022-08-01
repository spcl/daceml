import pytest

import torch

from onnx import helper, checker
from onnx import TensorProto

from daceml import onnx as donnx
from daceml.testing import copy_to_gpu


@pytest.mark.pure
def test_onnx_return_scalars(gpu, sdfg_name):
    # Dace programs can't return scalars.
    # this test checks that we correctly copy out the scalars using a size [1] array

    # we will have a single operator that computes the sum of a 1D tensor

    X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [5])
    # return value is a scalar
    Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [])

    node_def = helper.make_node(
        'ReduceSum',
        ['X'],
        ['Y'],
        keepdims=0,
    )

    graph_def = helper.make_graph(
        [node_def],
        'test-scalar-return',
        [X],
        [Y],
    )

    model_def = helper.make_model(graph_def)

    checker.check_model(model_def)

    # now we can test the backend
    dace_model = donnx.ONNXModel(sdfg_name, model_def, cuda=gpu)
    inp = copy_to_gpu(gpu, torch.arange(5).type(torch.float32))
    result = dace_model(inp)
    assert result.shape == ()
    assert result[()] == 1 + 2 + 3 + 4
