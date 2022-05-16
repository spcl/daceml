from typing import Dict

import torch
from dace import nodes as nd

from daceml.onnx.nodes.replacement import _modules_to_replace, module_name_to_placeholder_name


def replace_modules(module: torch.nn.Module):
    replaced_counter: Dict[str, int] = {}
    placeholder_name_to_module: Dict[str, torch.nn.Module] = {}

    def replace_modules_helper(module: torch.nn.Module):
        nonlocal replaced_counter
        nonlocal placeholder_name_to_module
        for name, submodule in module.named_children():
            cls = submodule.__class__
            cls_name = f"{cls.__module__}.{cls.__qualname__}"
            if cls_name in _modules_to_replace:
                idx = replaced_counter.get(cls_name, 0)
                placeholder_name = module_name_to_placeholder_name(
                    cls_name, idx)
                setattr(module, name, GenericPlaceholder(
                    placeholder_name, submodule))
                replaced_counter = idx + 1
                placeholder_name_to_module[placeholder_name] = submodule
            else:
                replace_modules_helper(submodule)

    replace_modules_helper(module)
    return placeholder_name_to_module


def restore_replaced_modules(placeholder_name_to_module, sdfg):
    def restore_helper(sdfg):
        for state in sdfg.states():
            for node in state.nodes():  # Make sure we have a copy
                if isinstance(node, nd.NestedSDFG):
                    restore_helper(node.sdfg)
                elif isinstance(node, nd.LibraryNode):
                    if node.name in placeholder_name_to_module:
                        node.module = placeholder_name_to_module[node.name]

    restore_helper(sdfg)


def create_placeholder_function_class(name):
    @staticmethod
    def forward(ctx, *inputs, **kwargs):
        return torch.zeros(())

    # TODO: How to handle kwargs?
    @staticmethod
    def symbolic(g: torch._C.Graph, *inputs):
        return g.op(f'daceml::{name}', *inputs)

    attrs = {}
    attrs['symbolic'] = symbolic
    attrs['forward'] = forward
    cls = type(name, (torch.autograd.Function,), attrs)
    return cls


class GenericPlaceholder(torch.nn.Module):
    def __init__(self, placeholder_name, replaced_module):
        super().__init__()
        self.replaced_name = placeholder_name
        self.replaced_module = replaced_module
        self.placeholder_function = create_placeholder_function_class(
            placeholder_name)

    def forward(self, *inputs, **kwargs):
        return self.placeholder_function.apply(*inputs, **kwargs)
