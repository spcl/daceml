from typing import Dict, List

import torch
from dace import nodes as nd

from daceml.onnx.nodes.replacement import ParamInfo, _modules_to_replace, _module_name_to_param_infos


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
                    cls_name, submodule, _module_name_to_param_infos[cls_name], replaced_idx, local_prefix))
                placeholder_id_to_module[replaced_idx] = (
                    local_prefix, submodule)
                replaced_idx += 1
            else:
                replace_modules_helper(submodule, local_prefix)

    replace_modules_helper(module, prefix='')
    return placeholder_id_to_module


def restore_replaced_modules(placeholder_id_to_module, sdfg):
    def restore_helper(sdfg):
        for state in sdfg.states():
            for node in state.nodes():
                if isinstance(node, nd.NestedSDFG):
                    restore_helper(node.sdfg)
                elif isinstance(node, nd.LibraryNode):
                    if hasattr(node, 'module_id') and node.module_id in placeholder_id_to_module:
                        node.module = placeholder_id_to_module[node.module_id]

    restore_helper(sdfg)


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
    def __init__(self, placeholder_name: str, replaced_module: torch.nn.Module, param_infos: List[ParamInfo], module_id: int, prefix: str):
        super().__init__()
        self.prefix: str = prefix
        self.replaced_name: str = placeholder_name
        self.replaced_module: torch.nn.Module = replaced_module
        # def copy_params_hepler(module):
        for name, p in replaced_module.named_parameters(recurse=False):
            self.register_parameter(name, p)
        for name, module in replaced_module.named_modules():
            if name:
                self.add_module(name, module)
        self.params = torch.nn.ParameterList(self.parameters())
        self.placeholder_function = create_placeholder_function_class(
            placeholder_name, module_id)

    def forward(self, *inputs, **kwargs):
        if kwargs:
            raise NotImplementedError("kwargs provided but not supported.")

        return self.placeholder_function.apply(*inputs)
