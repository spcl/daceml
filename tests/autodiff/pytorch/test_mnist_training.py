import os

import pytest

import numpy as np
import torch
import torchvision
from torchvision import datasets, transforms
from torch import nn, optim

from daceml.pytorch import DaceModule


@pytest.fixture
def mnist_trainloader():
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, ), (0.5, ))])
    data_directory = os.path.join(os.path.dirname(__file__), "data")
    trainset = datasets.MNIST(data_directory,
                              download=True,
                              train=True,
                              transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset,
                                              batch_size=64,
                                              shuffle=False)
    return trainloader


def test_single_train_step(mnist_trainloader):
    input_size = 784
    hidden_sizes = [128, 64]
    output_size = 10

    # initialize modules
    model = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]), nn.ReLU(),
                          nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                          nn.ReLU(), nn.Linear(hidden_sizes[1], output_size),
                          nn.LogSoftmax(dim=1))

    dace_model = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),
                               nn.ReLU(),
                               nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                               nn.ReLU(),
                               nn.Linear(hidden_sizes[1], output_size),
                               nn.LogSoftmax(dim=1))
    # copy over the weights
    dace_model.load_state_dict(model.state_dict())
    for dace_value, value in zip(model.state_dict().values(),
                                 dace_model.state_dict().values()):
        assert np.allclose(dace_value, value)

    dace_model = DaceModule(dace_model, backward=True, apply_strict=True)

    # check forward pass using loss
    images, labels = next(iter(mnist_trainloader))
    images = images.view(images.shape[0], -1)

    criterion = nn.NLLLoss()
    loss = criterion(model(images), labels)

    dace_output = dace_model(images)
    dace_loss = criterion(dace_output, labels)

    diff = abs(loss.item() - dace_loss.item()) / loss.item()
    assert diff < 1e-5

    print(loss.requires_grad)
    print(dace_loss.requires_grad)
    loss.backward()
    dace_loss.backward()

    for (name, dace_param), (pt_name,
                             pt_param) in zip(model.named_parameters(),
                                              dace_model.named_parameters()):
        assert 'model.' + name == pt_name
        assert np.allclose(dace_param.grad.numpy(), pt_param.grad.numpy(), rtol=1e-4),\
            "grad of param {} was not close".format(name)

    optimizer = optim.SGD(model.parameters(), lr=0.001)
    dace_optimizer = optim.SGD(dace_model.parameters(), lr=0.001)
    optimizer.step()
    dace_optimizer.step()

    for (name, dace_param), (pt_name,
                             pt_param) in zip(model.named_parameters(),
                                              dace_model.named_parameters()):
        assert 'model.' + name == pt_name
        assert np.allclose(dace_param.detach().numpy(),
                           pt_param.detach().numpy(),
                           rtol=1e-4), "param {} was not close".format(name)
