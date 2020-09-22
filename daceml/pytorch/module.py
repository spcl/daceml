import logging
import os
import tempfile
from functools import wraps

import torch
import torch.nn as nn
import onnx
from torch.onnx import TrainingMode

from daceml.onnx import ONNXModel
from daceml.onnx.shape_inference import infer_shapes


class DaceModule(nn.Module):
    def __init__(self,
                 model,
                 dummy_inputs=None,
                 cuda=False,
                 train=False,
                 apply_strict=False):
        super(DaceModule, self).__init__()

        self.model = model
        self.train = train
        self.sdfg = None
        self.cuda = cuda
        self.apply_strict = apply_strict
        if dummy_inputs is not None:
            self.dace_model = self.initialize_sdfg(dummy_inputs)

    def initialize_sdfg(self, dummy_inputs) -> ONNXModel:

        # TODO change to StringIO if not too big
        with tempfile.TemporaryDirectory() as dir_name:
            export_name = os.path.join(dir_name, "export.onnx")

            torch.onnx.export(self.model,
                              dummy_inputs,
                              export_name,
                              verbose=logging.root.level <= logging.DEBUG,
                              training=(TrainingMode.TRAINING
                                        if self.train else TrainingMode.EVAL),
                              opset_version=12)

            onnx_model = infer_shapes(onnx.load(export_name))
            self.onnx_model = onnx_model

            dace_model = ONNXModel("dace_model",
                                   onnx_model,
                                   cuda=self.cuda,
                                   apply_strict=self.apply_strict)
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
