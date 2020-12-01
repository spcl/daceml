import pytest
import numpy as np

from daceml.pytorch import DaceModule

import torch
import torch.nn as nn
import torch.nn.functional as F


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.fc1 = nn.Linear(16 * 6 * 6, 120)  # 6*6 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, 576)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = F.log_softmax(x, dim=1)
        return x


@pytest.mark.pure
def test_lenet():

    input = torch.rand(8, 1, 32, 32, dtype=torch.float32)

    net = LeNet()
    dace_net = LeNet()
    dace_net.load_state_dict(net.state_dict())
    dace_net = DaceModule(dace_net)

    torch_output = net(torch.clone(input))
    dace_output = dace_net(torch.clone(input))
    dace_net.sdfg.expand_library_nodes()
    assert np.allclose(torch_output.detach().numpy(), dace_output)
