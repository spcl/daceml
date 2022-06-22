from typing import Dict, Tuple

import torch

from daceml.onnx.nodes.replacement import _modules_to_replace, _module_name_to_shape_from_module


def replace_modules(module: torch.nn.Module):
    replaced_idx = 0
    placeholder_id_to_module: Dict[int, Tuple[str, torch.nn.Module]] = {}

    def replace_modules_helper(module: torch.nn.Module, prefix: str):
        nonlocal replaced_idx
        nonlocal placeholder_id_to_module
        for name, submodule in module.named_children():
            cls = submodule.__class__
            cls_name = f"{cls.__module__}.{cls.__qualname__}"
            local_prefix = f'{prefix}{name}.'
            if cls_name in _modules_to_replace:
                shape_fn = _module_name_to_shape_from_module[cls_name](submodule)
                setattr(module, name, GenericPlaceholder(
                    cls_name, submodule, replaced_idx, local_prefix, torch.float32,
                    shape_fn))  # TODO type should be from the registration
                placeholder_id_to_module[replaced_idx] = (
                    local_prefix, submodule)
                replaced_idx += 1
            else:
                replace_modules_helper(submodule, local_prefix)

    replace_modules_helper(module, prefix='')
    return placeholder_id_to_module


def create_placeholder_function_class(name, module_id, dtype, shape_fn):
    @staticmethod
    def forward(ctx, *inputs):
        return torch.zeros(shape_fn(*inputs), dtype=dtype)

    # TODO: How to handle kwargs?
    @staticmethod
    def symbolic(g: torch._C.Graph, *inputs):
        return g.op(f'daceml::{name}', *inputs, module_id_i=module_id)

    attrs = {}
    attrs['symbolic'] = symbolic
    attrs['forward'] = forward
    cls = type(name, (torch.autograd.Function,), attrs)
    return cls


class GenericPlaceholder(torch.nn.Module):
    def __init__(self, placeholder_name: str, replaced_module: torch.nn.Module, module_id: int, prefix: str,
                 output_dtype, shape_fn):
        super().__init__()
        self.prefix: str = prefix
        self.placeholder_function = create_placeholder_function_class(
            placeholder_name, module_id, output_dtype, shape_fn)
        for name, p in replaced_module.named_parameters(recurse=False):
            self.register_parameter(name, p)

        for name, submodule in replaced_module.named_modules():
            if len(name) > 0:
                self.add_module(name, submodule)

    def forward(self, *inputs, **kwargs):
        if kwargs:
            raise NotImplementedError("kwargs provided but not supported.")

        return self.placeholder_function.apply(*inputs)
