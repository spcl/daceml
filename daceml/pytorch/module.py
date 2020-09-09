import torch

import os
import tempfile
from functools import wraps

import torch.nn as nn
import onnx

from daceml.onnx import ONNXModel
from daceml.onnx.shape_inference import infer_shapes


class DaceModule(nn.Module):
    def __init__(self, model, dummy_inputs=None, cuda=False):
        super(DaceModule, self).__init__()

        self.model = model
        self.sdfg = None
        self.cuda = cuda
        if dummy_inputs is not None:
            self.dace_model = self.initialize_sdfg(dummy_inputs)

    def initialize_sdfg(self, dummy_inputs) -> ONNXModel:

        with tempfile.TemporaryDirectory() as dir_name:
            export_name = os.path.join(dir_name, "export.onnx")

            torch.onnx.export(
                self.model,
                dummy_inputs,
                export_name,
                verbose=True,  # TODO read log level
                opset_version=12)

            onnx_model = infer_shapes(onnx.load(export_name))

            dace_model = ONNXModel("dace_model", onnx_model, cuda=self.cuda)
            self.sdfg = dace_model.sdfg
            self.sdfg.validate()

            return dace_model

    def forward(self, *actual_inputs):
        if self.sdfg is None:
            self.dace_model = self.initialize_sdfg(actual_inputs)

        return self.dace_model(*actual_inputs)


def dace_module(moduleclass):
    """
    Decorator to apply on a definition of a ``torch.nn.Module`` to
    convert it to a data-centric module upon construction.
    """
    @wraps(moduleclass)
    def _create(*args, **kwargs):
        return DaceModule(moduleclass(*args, **kwargs))

    return _create
