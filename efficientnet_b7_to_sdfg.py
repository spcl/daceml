import onnx
import os
import numpy as np
from daceml.onnx import ONNXModel

model = onnx.load("efficientnet_b0.onnx")
dace_model = ONNXModel("efficientnet_b0", model, onnx_simplify=False, auto_optimize=False, simplify=False)
dace_model.expand_onnx_nodes()
dace_model.sdfg.save("efficientnet_b0.sdfg")