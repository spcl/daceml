from typing import Dict

import torch

from daceml.onnx.nodes.replacement import _modules_to_replace


def replace_modules(module: torch.nn.Module):
    replaced_idx = 0
    placeholder_id_to_module: Dict[str, torch.nn.Module] = {}

    def replace_modules_helper(module: torch.nn.Module, prefix: str):
        nonlocal replaced_idx
        nonlocal placeholder_id_to_module
        for name, submodule in module.named_children():
            cls = submodule.__class__
            cls_name = f"{cls.__module__}.{cls.__qualname__}"
            local_prefix = f'{prefix}{name}.'
            if cls_name in _modules_to_replace:
                setattr(module, name, GenericPlaceholder(
                    cls_name, submodule, replaced_idx, local_prefix))
                placeholder_id_to_module[replaced_idx] = (
                    local_prefix, submodule)
                replaced_idx += 1
            else:
                replace_modules_helper(submodule, local_prefix)

    replace_modules_helper(module, prefix='')
    return placeholder_id_to_module


def create_placeholder_function_class(name, module_id):
    @staticmethod
    def forward(ctx, *inputs):
        return torch.zeros(())

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
    def __init__(self, placeholder_name: str, replaced_module: torch.nn.Module, module_id: int, prefix: str):
        super().__init__()
        self.prefix: str = prefix
        self.replaced_module: torch.nn.Module = replaced_module
        self.placeholder_function = create_placeholder_function_class(
            placeholder_name, module_id)

    def forward(self, *inputs, **kwargs):
        if kwargs:
            raise NotImplementedError("kwargs provided but not supported.")

        return self.placeholder_function.apply(*inputs)
