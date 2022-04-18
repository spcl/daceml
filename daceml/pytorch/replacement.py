import torch
from torch.onnx import register_custom_op_symbolic
from torch.onnx.symbolic_helper import parse_args

# This is needed because otherwise the torch model tracing complains.
op_source = """
        #include <torch/script.h>
        torch::Tensor gcn_conv_placeholder(torch::Tensor node_features, torch::Tensor dummy_weights, torch::Tensor edge_list) {
           return torch::matmul(node_features, dummy_weights);
        }
        static auto registry =
          torch::RegisterOperators("daceml::gcn_conv_placeholder", &gcn_conv_placeholder);
        """

torch.utils.cpp_extension.load_inline(
    name="gcn_conv_placeholder",
    cpp_sources=op_source,
    is_python_module=False,
    verbose=True,
)


# Insert a dummy op.
@parse_args("v", "v", "v")
def gcn_conv_placeholder(g, node_features, dummy_weights, edge_list):
    return g.op('daceml::GcnConvPlaceholder', node_features, dummy_weights, edge_list)


# Register custom symbolic function.
register_custom_op_symbolic("daceml::gcn_conv_placeholder", gcn_conv_placeholder, 9)


class GCNConvPlaceholder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCNConvPlaceholder, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dummy_weights = torch.nn.Parameter(torch.zeros((in_channels, out_channels)))

    def forward(self, node_features, edge_list):
        return torch.ops.daceml.gcn_conv_placeholder(node_features, self.dummy_weights, edge_list)


def replace_children(module):
    for name, submodule in module.named_children():
        cls = submodule.__class__
        if f"{cls.__module__}.{cls.__qualname__}" == 'torch_geometric.nn.conv.gcn_conv.GCNConv':
            setattr(module, name, GCNConvPlaceholder(submodule.in_channels, submodule.out_channels))
        else:
            replace_children(submodule)
