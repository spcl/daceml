import torch
from daceml.onnx.nodes.replacement import _modules_to_replace


def replace_modules(module: torch.nn.Module):
    for name, submodule in module.named_children():
        cls = submodule.__class__
        cls_name = f"{cls.__module__}.{cls.__qualname__}"
        if cls_name in _modules_to_replace:
            setattr(module, name, GenericPlaceholder(
                _modules_to_replace[cls_name], submodule))
        else:
            replace_modules(submodule)


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
    cls = type(name, (torch.autograd.Function, ), attrs)
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
