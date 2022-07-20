import torch
from torch import nn
from torch_geometric.nn import GCNConv, GATConv


class GCN(torch.nn.Module):
    def __init__(self, num_node_features, num_hidden_features, num_classes, normalize):
        super().__init__()
        print("normalize: ", normalize)
        self.conv1 = GCNConv(num_node_features, num_hidden_features,
                             normalize=normalize, add_self_loops=False)
        self.conv2 = GCNConv(num_hidden_features, num_classes,
                             normalize=normalize, add_self_loops=False)

        self.act = nn.ReLU()
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x, *edge_info):
        x = self.conv1(x, *edge_info)
        x = self.act(x)
        x = self.conv2(x, *edge_info)

        return self.log_softmax(x)


class LinearModel(torch.nn.Module):
    def __init__(self, num_node_features, _unused, num_classes) -> None:
        super().__init__()
        self.lin = nn.Linear(num_node_features, num_classes)

    def forward(self, x):
        x = self.lin(x)
        return x


class GAT(torch.nn.Module):
    def __init__(self, num_node_features, num_classes, num_heads=8, features_per_head=8):
        super().__init__()
        self.conv1 = GATConv(num_node_features, features_per_head,
                             heads=num_heads, add_self_loops=False, bias=False)
        self.conv2 = GATConv(features_per_head * num_heads, num_classes,
                             heads=1, add_self_loops=False, bias=False)

        self.act = nn.ELU()
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x, *edge_info):
        x = self.conv1(x, *edge_info)
        x = self.act(x)
        x = self.conv2(x, *edge_info)

        return self.log_softmax(x)