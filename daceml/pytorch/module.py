import torch
import torch.nn as nn
import torch.nn.functional as F
import onnx
import numpy as np

import dace
from dace.frontend.onnx import ONNXModel
from .symbolic_shape_infer import SymbolicShapeInference

class DACEModule(nn.Module):
    def __init__(self, model, *dummy_inputs):
        super(DACEModule, self).__init__()
        
        self.model = model
        input_names = [ "actual_input_1" ]
        torch.onnx.export(self.model, dummy_inputs, "model.onnx", verbose=True, input_names=input_names, opset_version=12)
        SymbolicShapeInference.infer_shapes("model.onnx", "shape_infer.onnx")

        onnx_model = onnx.load("./shape_infer.onnx")
        self.dace_model = ONNXModel("dace_model", onnx_model)
        self.sdfg = self.dace_model.sdfg
        self.sdfg.validate()
        self.sdfg.save("./model.sdfg")

    def forward(self, *actual_inputs):
        return self.dace_model(*actual_inputs) 

