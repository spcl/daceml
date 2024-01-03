import onnx
import os
import numpy as np
from daceml.onnx import ONNXModel

model = onnx.load("bert-base-uncased_1.onnx")
dace_model = ONNXModel("bert_base_uncased", model, onnx_simplify=False, auto_optimize=False, simplify=False)
dace_model.sdfg.save("bert_base_uncased.sdfg")