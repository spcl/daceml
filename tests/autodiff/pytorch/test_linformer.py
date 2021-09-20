import torch
from daceml.pytorch import DaceModule
from daceml.testing import torch_tensors_close
from torch import nn


def run(module, input_shape, output_shape):
    with torch.no_grad():
        dace_inputs = torch.rand(*input_shape).cuda()
        torch_inputs = torch.clone(dace_inputs)

    dace_inputs.requires_grad = True
    torch_inputs.requires_grad = True

    torch_model = module().cuda()
    dace_model = module().cuda()
    dace_model = DaceModule(dace_model, backward=True)
    dace_model.model.load_state_dict(torch_model.state_dict())

    for (dace_name,
         dace_value), (torch_name,
                       value) in zip(dace_model.model.state_dict().items(),
                                     torch_model.state_dict().items()):
        assert dace_name == torch_name
        torch_tensors_close(dace_name, value, dace_value)

    dace_output = dace_model(dace_inputs)
    torch_output = torch_model(torch_inputs)
    torch_tensors_close("output",
                        torch_output,
                        dace_output,
                        rtol=1e-4,
                        atol=1e-4)

    # check that the batch norm running means and so on are written out correctly
    for (dace_name,
         dace_value), (torch_name,
                       value) in zip(dace_model.model.state_dict().items(),
                                     torch_model.state_dict().items()):

        assert dace_name == torch_name
        if "num_batches_tracked" in dace_name:
            # we don't update this parameter
            continue
        torch_tensors_close(dace_name, value, dace_value)

    # backward pass
    dy = torch.rand(*output_shape).cuda()

    torch_output.backward(dy)
    dace_output.backward(dy)

    torch_tensors_close("input_grad", torch_inputs.grad, dace_inputs.grad)

    for (name, dace_param), (pt_name,
                             pt_param) in zip(torch_model.named_parameters(),
                                              dace_model.named_parameters()):
        assert 'model.' + name == pt_name
        torch_tensors_close(name, pt_param.grad, dace_param.grad)


class Simple(torch.nn.Module):
    def __init__(self):
        super(Module, self).__init__()
        self.linear = nn.Linear(1024, 4096)

    def forward(self, x):
        return self.linear(x)


def test_linformer_case(sdfg_name, default_implementation):
    run(Simple, input_shape=(8, 512, 1024), output_shape=(8, 512, 4096))


class PreNorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.fn = FeedForward()
        self.norm = nn.LayerNorm(1024)

    def forward(self, x):
        return self.fn(self.norm(x)) + x


def test_linformer_ff(sdfg_name, default_implementation):
    run(FeedForward, input_shape=(8, 512, 1024), output_shape=(8, 512, 1024))


def test_linformer_pn(sdfg_name, default_implementation):
    run(PreNorm, input_shape=(8, 512, 1024), output_shape=(8, 512, 1024))


class FeedForward(nn.Module):
    def __init__(self,
                 dim=1024,
                 mult=4,
                 dropout=0.,
                 activation=None,
                 glu=False):
        super().__init__()
        activation = nn.ReLU

        self.glu = glu
        self.w1 = nn.Linear(dim, dim * mult * (2 if glu else 1))
        self.act = activation()
        self.dropout = nn.Dropout(dropout)
        self.w2 = nn.Linear(dim * mult, dim)

    def forward(self, x, **kwargs):
        in_x = x
        if not self.glu:
            x = self.w1(x)
            x = self.act(x)
        else:
            x, v = self.w1(x).chunk(2, dim=-1)
            x = self.act(x) * v

        x = self.dropout(x)
        x = self.w2(x)
        return x + in_x
