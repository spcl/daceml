try:
    import torch
except ModuleNotFoundError:
    raise ImportError("To use the torch frontend, please install torch")

import os
import tempfile
from functools import wraps

import torch.nn as nn
import onnx

from daceml.onnx import ONNXModel
from daceml.onnx.shape_inference import infer_shapes


class DACEModule(nn.Module):
    def __init__(self, model, dummy_inputs=None):
        super(DACEModule, self).__init__()

        self.model = model
        self.sdfg = None
        if dummy_inputs is not None:
            self.dace_model = self.initialize_sdfg(dummy_inputs)

    def initialize_sdfg(self, dummy_inputs) -> ONNXModel:
        export_fd, export_name = tempfile.mkstemp(suffix=".onnx")
        infer_fd, infer_name = tempfile.mkstemp(suffix=".onnx")

        torch.onnx.export(
            self.model,
            dummy_inputs,
            export_name,
            verbose=True,  # TODO read log level
            opset_version=12)

        infer_shapes(export_name, infer_name)

        onnx_model = onnx.load(infer_name)
        dace_model = ONNXModel("dace_model", onnx_model)
        self.sdfg = dace_model.sdfg
        self.sdfg.validate()

        os.close(export_fd)
        os.close(infer_fd)
        return dace_model

    def forward(self, *actual_inputs):
        if self.sdfg is None:
            self.dace_model = self.initialize_sdfg(actual_inputs)

        return self.dace_model(*(inp.numpy() for inp in actual_inputs))


def module(moduleclass):
    """
    Decorator to apply on a definition of a ``torch.nn.Module`` to
    convert it to a data-centric module upon construction.
    """
    @wraps(moduleclass)
    def _create(*args, **kwargs):
        return DACEModule(moduleclass(*args, **kwargs))

    return _create
