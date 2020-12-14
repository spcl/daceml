""" A lenet inference script. Example adapted from https://github.com/pytorch/examples/blob/master/mnist/main.py """
import numpy as np
import argparse

from daceml.pytorch import DaceModule
import daceml.onnx as donnx
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from dace.transformation.interstate import FPGATransformSDFG, InlineSDFG
import copy
import dace
from daceml.util import utils
from daceml import transformation

def print_mnist_mean_and_std():
    train_dataset = datasets.MNIST('./data',
                                   train=True,
                                   download=True,
                                   transform=transforms.ToTensor())
    train_loader = torch.utils.data.DataLoader(train_dataset)
    all_train_images = [x for x, y in train_loader]
    stacked = torch.stack(all_train_images)
    print("Mean:", stacked.mean().item(), "std:", stacked.std().item())


def get_dataloader(train, batch_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
        # these values are chosen using print_mnist_mean_and_std
        transforms.Normalize((0.1307, ), (0.3081, ))
    ])
    dataset = datasets.MNIST('./data',
                             train=train,
                             download=True,
                             transform=transform)
    return torch.utils.data.DataLoader(dataset,
                                       batch_size=batch_size,
                                       shuffle=train)


class TrainLeNet(nn.Module):
    def __init__(self):
        super(TrainLeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(256, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, 256)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class TestLeNet(nn.Module):
    def __init__(self):
        super(TestLeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(256, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, 256)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = F.softmax(x, dim=1)
        return x


def eval_model(args, test_dataloader, model, device, single=False):
    model.eval()
    if device == 'dace':
        model.to('cpu')
        dummy_input = next(iter(test_dataloader))
        model = DaceModule(model, dummy_inputs=dummy_input[0])
        model.sdfg.save('/tmp/out.sdfg')
        transformation.expand_library_nodes_except_reshape(model.sdfg)
        model.sdfg.apply_transformations_repeated(
        [transformation.ReshapeElimination])
        model.sdfg.save('/tmp/out_expanded.sdfg')
        device = 'cpu'
    elif device == 'fpga':
        # transform to FPGA, for pytorch the device is always 'cpu'
        model.to('cpu')
        dummy_input = next(iter(test_dataloader))
        donnx.ONNXRelu.default_implementation = "fpga"
        donnx.ONNXMaxPool.default_implementation = "fpga"
        donnx.ONNXGemm.default_implementation = "fpga"
        donnx.ONNXConv.default_implementation = 'fpga'

        model = DaceModule(model, dummy_inputs=dummy_input[0])
        sdfg = model.sdfg
        sdfg.apply_transformations([FPGATransformSDFG])
        transformation.expand_library_nodes_except_reshape(sdfg)
        sdfg.states()[0].nodes()[0].sdfg.apply_transformations_repeated(
            [transformation.ReshapeElimination])
        sdfg.states()[0].location["is_FPGA_kernel"] = False
        sdfg.states()[0].nodes()[0].sdfg.states()[0].location["is_FPGA_kernel"] = False

        sdfg.save('/tmp/out_fpga.sdfg')
        device = 'cpu'
    elif device == 'pytorch':
        model.to('cpu')
        device = 'cpu'
    else:
        model.to(device)
    test_loss = 0
    correct = 0
    amount_samples = 0

    def eval_single_batch(data, target):
        data, target = data.to(device), target.to(device)
        start_time = time.time()
        output = model(data)
        elapsed_time = time.time() - start_time
        print("Inference performed in " + str(elapsed_time) + " secs.")
        pred = output.argmax(1)
        if isinstance(pred, torch.Tensor):
            pred = np.array(pred.cpu())
        target = np.array(target.cpu())
        return (pred == target).sum().item(), target.shape[0]

    with torch.no_grad():
        if single:
            data, target = next(iter(test_dataloader))
            batch_correct, batch_num_samples = eval_single_batch(data, target)
            correct += batch_correct
            amount_samples += batch_num_samples
        else:
            for batch_idx, (data, target) in enumerate(test_dataloader):
                batch_correct, batch_num_samples = eval_single_batch(data, target)
                correct += batch_correct
                amount_samples += batch_num_samples
    print("TESTING")
    print("Accuracy: {:.2f}%".format(100 * correct / amount_samples))


def train_model(args, train_dataloader, model, device):
    optimizer = torch.optim.Adadelta(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=1,
                                                gamma=args.gamma)

    criterion = nn.CrossEntropyLoss()
    model.train()
    model.to(device)
    for epoch in range(args.epochs):
        print("EPOCH", epoch)
        for batch_idx, (data, target) in enumerate(train_dataloader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            if batch_idx % args.log_interval == 0:
                print("TRAIN [{}/{}]: Loss: {:.6f}".format(
                    batch_idx, len(train_dataloader), loss.item()))
        scheduler.step()
    torch.save(model.state_dict(), "./data/weights.pt")


def run_batch_inference():
    input = torch.rand(8, 1, 28, 28, dtype=torch.float32)

    net = TestLeNet()
    dace_net = TestLeNet()
    dace_net.load_state_dict(net.state_dict())
    dace_net = DaceModule(dace_net)

    torch_output = net(torch.clone(input))
    dace_output = dace_net(torch.clone(input))
    dace_net.sdfg.expand_library_nodes()
    dace_net.sdfg.view()
    assert np.allclose(torch_output.detach().numpy(), dace_output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MNIST Example')
    parser.add_argument('--batch-size',
                        type=int,
                        default=64,
                        metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size',
                        type=int,
                        default=1000,
                        metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs',
                        type=int,
                        default=14,
                        metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument(
        '--log-interval',
        type=int,
        default=10,
        metavar='N',
        help='the interval between logging output (default: 10)')
    parser.add_argument('--gamma',
                        type=float,
                        default=0.7,
                        metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--lr',
                        type=float,
                        default=1.0,
                        metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--cuda',
                        action='store_true',
                        default=False,
                        help='enable CUDA training (using pytorch)')
    parser.add_argument(
        '--train-model',
        action='store_true',
        default=False,
        help=
        'if true, new weights will be trained and stored in the "data" directory. If false, the'
        ' script will attempt to load the weights from the directory.')
    args = parser.parse_args()

    donnx.default_implementation = 'pure'
    donnx.ONNXConv.default_implementation = 'im2col'

    train_loader = get_dataloader(False, args.batch_size)
    test_loader = get_dataloader(True, args.test_batch_size)


    if args.train_model:
        model = TrainLeNet()
        train_model(args, train_loader, model, 'cuda' if args.cuda else 'cpu')

    model = TestLeNet()
    # try to load the weights
    model.load_state_dict(torch.load("./data/weights.pt"))

    #eval_model(args, test_loader, model, 'cuda')
    eval_model(args, test_loader, model, 'cpu', single=True)
    eval_model(args, test_loader, model, 'dace', single=True)
    eval_model(args, test_loader, model, 'fpga', single=True)
