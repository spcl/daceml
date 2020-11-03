import os

import numpy as np
import onnx

import dace
from dace.transformation.dataflow import RedundantSecondArray

import daceml.onnx as donnx
from daceml.transformation import ConstantFolding

data_directory = os.path.join(os.path.dirname(__file__), "..", "onnx_files")


def test_bert_subgraph():

    model = onnx.load(os.path.join(data_directory, "reshape.onnx"))
    dace_model = donnx.ONNXModel("reshape", model)

    out_before = dace_model()
    assert len(dace_model.sdfg.nodes()[0].nodes()) > 2

    dace_model.sdfg.apply_transformations_repeated(
        [ConstantFolding, RedundantSecondArray], validate_all=True)

    out_after = dace_model()

    # assert that only two nodes remain
    assert len(dace_model.sdfg.nodes()[0].nodes()) == 2
    assert np.allclose(out_before, out_after)
