import collections
import logging
import os
import tempfile
from functools import wraps
from typing import Optional, Tuple, Callable, OrderedDict

import dace
import onnx
import torch
import torch.nn as nn
from dace.codegen import compiled_sdfg
from torch.onnx import TrainingMode

from daceml.pytorch.module_codegen import get_function_for_module
from daceml.autodiff.pytorch import make_backward_function
from daceml.onnx import ONNXModel
from daceml.onnx.shape_inference import infer_shapes
from daceml.util import utils, find_str_not_in_set

log = logging.getLogger(__name__)


class DaceModule(nn.Module):
    """ A wrapper that converts a PyTorch ``nn.Module`` to a PyTorch compatible data-centric ``nn.Module``.

        :param module: the model to wrap.
        :param dummy_inputs: a tuple of tensors to use as input when tracing ``model``.
        :param cuda: if ``True``, the module will execute using CUDA. If ``None``, it will be detected from the
                     ``module``.
        :param training: whether to use train mode when tracing ``model``.
        :param backward: whether to enable the backward pass.
        :param apply_strict: whether to apply strict transforms after conversion (this generally improves performance,
                             but can be slow).
        :param sdfg_name: the name to give to the sdfg (defaults to ``dace_model``).
        :param auto_optimize: whether to apply automatic optimizations.

        :Example:

            >>> from daceml.pytorch import DaceModule
            >>> class MyModule(nn.Module):
            ...     def forward(self, x):
            ...        x = torch.log(x)
            ...        x = torch.sqrt(x)
            ...        return x
            >>> module = MyModule()
            >>> module(torch.ones(2))
            tensor([0., 0.])
            >>> dace_module = DaceModule(module)
            >>> dace_module(torch.ones(2))
            Automatically expanded library node "ONNX_Log_0" with implementation "onnxruntime".
            Automatically expanded library node "ONNX_Sqrt_1" with implementation "onnxruntime".
            tensor([0., 0.])
    """
    def __init__(self,
                 module: nn.Module,
                 dummy_inputs: Optional[Tuple[torch.Tensor]] = None,
                 cuda: Optional[bool] = None,
                 training: bool = False,
                 backward=False,
                 apply_strict: bool = True,
                 auto_optimize: bool = True,
                 sdfg_name: Optional[str] = None):
        super(DaceModule, self).__init__()

        self.backward = backward
        self.model = module
        self.dace_model: Optional[ONNXModel] = None
        self.training = training
        self.sdfg: Optional[dace.SDFG] = None
        self.cuda = cuda
        self.sdfg_name = sdfg_name or "dace_model"
        self.auto_optimize = auto_optimize
        self.apply_strict = apply_strict

        self.function = None

        #: hooks that are executed after onnx graph is imported to an SDFG
        self.post_onnx_hooks: OrderedDict[str, Callable[
            [DaceModule], None]] = collections.OrderedDict()

        #: hooks that are executed after the backpropagation sdfg has been created
        self.post_autodiff_hooks: OrderedDict[str, Callable[
            [dace.SDFG, dace.SDFG], None]] = collections.OrderedDict()

        #: hooks that are executed after the sdfg is compiled
        self.post_compile_hooks: OrderedDict[str, Callable[
            [compiled_sdfg.CompiledSDFG], None]] = collections.OrderedDict()

        if dummy_inputs is not None:
            self.function = self._initialize_sdfg(dummy_inputs)

    def reset_sdfg(self):
        """ Clear the sdfg so that optimizations are reapplied. """
        self.function = None

    def prepend_post_onnx_hook(self, name: str, func: Callable[["DaceModule"],
                                                               None]):
        if self.function is not None:
            log.warning(
                f"Added a hook after the model was already initialized. This hook "
                f"(with name {name}) will not be executed!")
        name = find_str_not_in_set(set(self.post_onnx_hooks), name)
        self.post_onnx_hooks[name] = func
        self.post_onnx_hooks.move_to_end(name, last=False)

    def append_post_onnx_hook(self, name: str, func: Callable[["DaceModule"],
                                                              None]):
        if self.function is not None:
            log.warning(
                f"Added a hook after the model was already initialized. This hook "
                f"(with name {name}) will not be executed!")
        name = find_str_not_in_set(set(self.post_onnx_hooks), name)
        self.post_onnx_hooks[name] = func

    def prepend_post_autodiff_hook(self, name: str,
                                   func: Callable[[dace.SDFG, dace.SDFG],
                                                  None]):
        if self.function is not None:
            log.warning(
                f"Added a hook after the model was already initialized. This hook "
                f"(with name {name}) will not be executed!")
        name = find_str_not_in_set(set(self.post_autodiff_hooks), name)
        self.post_autodiff_hooks[name] = func
        self.post_autodiff_hooks.move_to_end(name, last=False)

    def append_post_autodiff_hook(self, name: str,
                                  func: Callable[[dace.SDFG, dace.SDFG],
                                                 None]):
        if self.function is not None:
            log.warning(
                f"Added a hook after the model was already initialized. This hook "
                f"(with name {name}) will not be executed!")
        name = find_str_not_in_set(set(self.post_autodiff_hooks), name)
        self.post_autodiff_hooks[name] = func

    def prepend_post_compile_hook(self, name: str,
                                  func: Callable[[compiled_sdfg.CompiledSDFG],
                                                 None]):
        if self.function is not None:
            log.warning(
                f"Added a hook after the model was already initialized. This hook "
                f"(with name {name}) will not be executed!")
        name = find_str_not_in_set(set(self.post_compile_hooks), name)
        self.post_compile_hooks[name] = func
        self.post_compile_hooks.move_to_end(name, last=False)

    def append_post_compile_hook(self, name: str,
                                 func: Callable[[compiled_sdfg.CompiledSDFG],
                                                None]):
        if self.function is not None:
            log.warning(
                f"Added a hook after the model was already initialized. This hook "
                f"(with name {name}) will not be executed!")
        name = find_str_not_in_set(set(self.post_compile_hooks), name)
        self.post_compile_hooks[name] = func

    def _initialize_sdfg(self, dummy_inputs):

        # determine whether we are using CUDA
        module_is_cuda = next(iter(dummy_inputs)).is_cuda
        if not module_is_cuda:
            # check the parameters
            try:
                module_is_cuda = next(self.model.parameters()).is_cuda
            except StopIteration:
                module_is_cuda = False

        if module_is_cuda and self.cuda is False:
            log.warning("Received a CUDA module, but cuda was set to False.")
        if self.cuda is None:
            self.cuda = module_is_cuda

        # setup optimization hooks
        if self.auto_optimize:
            if self.backward:

                def auto_optimize_backward(fwd_sdfg, bwd_sdfg):
                    utils.auto_optimize(fwd_sdfg,
                                        self.cuda,
                                        apply_strict=self.apply_strict)
                    utils.auto_optimize(bwd_sdfg,
                                        self.cuda,
                                        apply_strict=self.apply_strict)

                self.prepend_post_autodiff_hook("auto_optimize",
                                                auto_optimize_backward)
            else:
                self.prepend_post_onnx_hook(
                    "auto_optimize", lambda dace_module: utils.auto_optimize(
                        dace_module.dace_model.sdfg,
                        self.cuda,
                        apply_strict=self.apply_strict))
        elif self.apply_strict:
            if self.backward:

                def apply_strict(fwd_sdfg, bwd_sdfg):
                    fwd_sdfg.apply_strict_transformations()
                    bwd_sdfg.apply_strict_transformations()

                self.prepend_post_autodiff_hook("apply_strict", apply_strict)
            else:
                self.prepend_post_onnx_hook(
                    "apply_strict", lambda dace_module: dace_module.sdfg.
                    apply_strict_transformations())

        # TODO change to StringIO if not too big
        with tempfile.TemporaryDirectory() as dir_name:
            export_name = os.path.join(dir_name, "export.onnx")

            torch.onnx.export(
                self.model,
                dummy_inputs,
                export_name,
                verbose=logging.root.level <= logging.DEBUG,
                training=(TrainingMode.TRAINING
                          if self.training else TrainingMode.EVAL),
                opset_version=12,
                strip_doc_string=False,
                export_params=not self.backward,
                # pytorch constant folding will add new unnamed inputs to the graph and remove some of the
                # named parameters of the model: this means that we can't match with the state dict
                # anymore, so we disable this. Our CF is more flexible.
                do_constant_folding=False,
                keep_initializers_as_inputs=True)

            onnx_model = infer_shapes(onnx.load(export_name))
            self.onnx_model = onnx_model

            dace_model = ONNXModel(self.sdfg_name,
                                   onnx_model,
                                   infer_shapes=False,
                                   cuda=self.cuda,
                                   parent_pytorch_module=self.model)
            self.sdfg = dace_model.sdfg
            self.dace_model = dace_model

            self.sdfg.validate()

            for _, hook in self.post_onnx_hooks.items():
                hook(self)

            if self.backward:
                function = make_backward_function(dace_model)

                for _, hook in self.post_autodiff_hooks.items():
                    hook(function._forward_model.sdfg, function._backward_sdfg)

                function._forward_model.compile_and_init()

                def forward(*args):
                    args_and_params = list(args)
                    args_and_params.extend(self.parameters())
                    return function.apply(*args_and_params)

                return forward
            else:
                self.fwd_func = get_function_for_module(self, dummy_inputs)
                parameters_to_pass = tuple(
                    p.data for n, p in self.model.named_parameters()
                    if n in self.dace_model.inputs)

                def forward(*args):
                    return self.fwd_func.function(self.fwd_func.ptr, *args,
                                                  *parameters_to_pass)

                return forward

    def forward(self, *actual_inputs):
        """ Execute the forward pass using the traced ``module``."""
        if self.function is None:
            self.function = self._initialize_sdfg(actual_inputs)

        return self.function(*actual_inputs)


@dace.dtypes.paramdec
def dace_module(moduleclass,
                dummy_inputs: Optional[Tuple[torch.Tensor]] = None,
                cuda: Optional[bool] = None,
                training: bool = False,
                backward=False,
                apply_strict: bool = True,
                auto_optimize: bool = True,
                sdfg_name: Optional[str] = None):
    """ Decorator to apply on a definition of a ``torch.nn.Module`` to
        convert it to a data-centric module upon construction.

        :Example:

            >>> from daceml.pytorch import dace_module
            >>> @dace_module
            ... class MyModule(nn.Module):
            ...     def forward(self, x):
            ...        x = torch.log(x)
            ...        x = torch.sqrt(x)
            ...        return x
            >>> module = MyModule()
            >>> module(torch.ones(2))
            Automatically expanded library node "ONNX_Log_0" with implementation "onnxruntime".
            Automatically expanded library node "ONNX_Sqrt_1" with implementation "onnxruntime".
            tensor([0., 0.])

        :param moduleclass: the model to wrap.
        :param dummy_inputs: a tuple of tensors to use as input when tracing ``model``.
        :param cuda: if ``True``, the module will execute using CUDA. If ``None``, it will be detected from the
                     ``module``.
        :param training: whether to use train mode when tracing ``model``.
        :param backward: whether to enable the backward pass.
        :param apply_strict: whether to apply strict transforms after conversion (this generally improves performance,
                             but can be slow).
        :param auto_optimize: whether to apply automatic optimizations.
        :param sdfg_name: the name to give to the sdfg (defaults to ``dace_model``).
    """
    @wraps(moduleclass)
    def _create(*args, **kwargs):
        return DaceModule(moduleclass(*args, **kwargs),
                          dummy_inputs=dummy_inputs,
                          cuda=cuda,
                          training=training,
                          backward=backward,
                          apply_strict=apply_strict,
                          auto_optimize=auto_optimize,
                          sdfg_name=sdfg_name)

    return _create
