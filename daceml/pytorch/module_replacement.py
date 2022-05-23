from typing import Dict, List

import torch
from dace import nodes as nd

from daceml.onnx.nodes.replacement import ParamInfo, _modules_to_replace, _module_name_to_param_infos


def replace_modules(module: torch.nn.Module):
    replaced_idx = 0
    placeholder_id_to_module: Dict[str, torch.nn.Module] = {}

    def replace_modules_helper(module: torch.nn.Module):
        nonlocal replaced_idx
        nonlocal placeholder_id_to_module
        for name, submodule in module.named_children():
            cls = submodule.__class__
            cls_name = f"{cls.__module__}.{cls.__qualname__}"
            if cls_name in _modules_to_replace:
                setattr(module, name, GenericPlaceholder(
                    cls_name, submodule, _module_name_to_param_infos[cls_name], replaced_idx))
                placeholder_id_to_module[replaced_idx] = submodule
                replaced_idx += 1
            else:
                replace_modules_helper(submodule)

    replace_modules_helper(module)
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
    def __init__(self, placeholder_name: str, replaced_module: torch.nn.Module, param_infos: List[ParamInfo], module_id: int):
        super().__init__()
        self.replaced_name: str = placeholder_name
        self.replaced_module: torch.nn.Module = replaced_module

        # def copy_params_hepler(module):
        for name, p in replaced_module.named_parameters(recurse=False):
            self.register_parameter(name, p)
        for name, module in replaced_module.named_modules():
            if name:
                self.add_module(name, module)
        self.params = torch.nn.ParameterList(self.parameters())
        # # The parameters have to be fields of this module, otherwise TorchScript tracing fails silently.
        # self.params: torch.nn.ParameterList = torch.nn.ParameterList()
        # # self.original_to_replaced: Dict[str, str] = {}
        # # self.replaced_to_original: Dict[str, str] = {}
        # params_by_name = {name: param for name,
        #                   param in replaced_module.named_parameters()}
        # for param_info in param_infos:
        #     param = params_by_name.get(param_info.name)
        #     if param is None and param_info.required:
        #         raise ValueError(
        #             f"Missing required parameter: {param_info.name}")
        #     # TODO: Fix
        #     # if param is not None and param.dtype != param_info.dtype:
        #     #     raise ValueError(
        #     #         f"Parameter {param_info.name} in module {placeholder_name} has type {param.dtype}, expected {param_info.dtype}.")
        #     if param is not None:
        #         # replaced_param_name = f'params.{len(self.params)}'
        #         # original_param_name = f'{param_info.name}'
        #         self.params.append(param)
        #         # self.original_to_replaced[original_param_name] = replaced_param_name
        #         # self.replaced_to_original[replaced_param_name] = original_param_name

        # for p, param_info in zip(self.params, param_infos):
        #     self._parameters[param_info.name] = p
        # del self._modules['params']
        # self._parameters = self.replaced_module._parameters

        self.placeholder_function = create_placeholder_function_class(
            placeholder_name, module_id)

        # def rename_params_when_loading(module, state_dict, prefix,
        #                                local_metadata, strict, missing_keys, unexpected_keys,
        #                                error_msgs):
        #     # TODO: oredring might be incorrect
        #     new_vals = OrderedDict()
        #     for k, v in state_dict.items():
        #         local_k = k[len(prefix):]
        #         if local_k in module.original_to_replaced:
        #             new_key = prefix + module.original_to_replaced[local_k]
        #             new_vals[new_key] = (v, k)

        #     for new_key, (v, old_k) in new_vals.items():
        #         state_dict[new_key] = v
        #         del state_dict[old_k]
        #     return state_dict

        # # self._register_load_state_dict_pre_hook(
        # #     rename_params_when_loading, with_module=True)

        # def rename_params_hook(self, state_dict, prefix, local_metadata):
        #     for k, v in state_dict.items():
        #         local_k = k[len(prefix):]
        #         if local_k in self.replaced_to_original:
        #             new_key = prefix + self.replaced_to_original[local_k]
        #             state_dict[new_key] = v
        #             del state_dict[k]
        #     return state_dict

        # self._register_state_dict_hook(rename_params_hook)

    def forward(self, *inputs, **kwargs):
        if kwargs:
            raise NotImplementedError("kwargs provided but not supported.")

        return self.placeholder_function.apply(*inputs)  # *self.params

    # def named_parameters(self, prefix: str = '', recurse: bool = True):
    #     params = OrderedDict(super().named_parameters(prefix, recurse))
    #     for k, v in params.items():
    #         local_k = k[len(prefix):]
    #         if local_k in self.replaced_to_original:
    #             new_key = prefix + self.replaced_to_original[local_k]
    #             params[new_key] = v
    #             del params[k]
    #     return params
