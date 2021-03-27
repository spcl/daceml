import pytest
import numpy as np

from dace import nodes

import daceml.onnx as donnx
from daceml.pytorch import DaceModule
from daceml import transformation

import torch
import torch.nn as nn
import torch.nn.functional as F

from daceml.transformation.input_to_constant import forward_memlet_tree_with_nested_and_copies


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)

        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = F.log_softmax(x, dim=1)
        return x


@pytest.mark.parametrize("conv_impl", ["pure", "im2col"])
@pytest.mark.pure
def test_lenet(conv_impl):
    donnx.ONNXConv.default_implementation = conv_impl

    input = torch.rand(8, 1, 32, 32, dtype=torch.float32)

    net = LeNet()
    dace_net = LeNet()
    dace_net.load_state_dict(net.state_dict())
    dace_net = DaceModule(dace_net, dummy_inputs=(torch.clone(input), ))

    torch_output = net(torch.clone(input))
    dace_output = dace_net(torch.clone(input))

    transformation.expand_library_nodes_except_reshape(dace_net.sdfg)
    dace_net.sdfg.apply_transformations_repeated(
        [transformation.ReshapeElimination], print_report=True)
    dace_net.sdfg.apply_transformations_repeated(
        [transformation.InputToConstant], print_report=True)

    diff = np.linalg.norm(torch_output.detach().numpy() - dace_output)
    assert diff < 1e-5


@pytest.mark.pure
def test_lenet_input_toconstant():
    input = torch.rand(8, 1, 32, 32, dtype=torch.float32)

    net = LeNet()
    dace_net = LeNet()
    dace_net.load_state_dict(net.state_dict())
    dace_net = DaceModule(dace_net, dummy_inputs=(torch.clone(input), ))
    dace_net.sdfg.expand_library_nodes()

    torch_output = net(torch.clone(input))
    dace_output = dace_net(torch.clone(input))

    state = dace_net.sdfg.nodes()[0]

    access = [
        n for n in state.nodes()
        if isinstance(n, nodes.AccessNode) and n.data == "ONNX_inputDOT1"
    ][0]

    def print_tree(tree):
        return "{} -> {}".format(tree.edge.src, tree.edge.dst) + "".join(
            "\n |\n +- {}".format(print_tree(c)) for c in tree.children)

    print(
        print_tree(
            forward_memlet_tree_with_nested_and_copies(
                state,
                state.out_edges(access)[0])))
