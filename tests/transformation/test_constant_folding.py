import numpy as np
import onnx

import dace
from dace.transformation.dataflow import RedundantSecondArray

import daceml.onnx as donnx
from daceml.transformation import ConstantFolding


def test_bert_subgraph():

    model = onnx.load("tests/onnx_files/reshape.onnx")
    dace_model = donnx.ONNXModel("reshape", model)

    out_before = dace_model()

    dace_model.sdfg.apply_transformations_repeated(
        [ConstantFolding, RedundantSecondArray], validate_all=True)

    out_after = dace_model()

    assert np.allclose(out_before, out_after)
