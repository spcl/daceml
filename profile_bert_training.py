import torch
from transformers import BertLayer, BertConfig

from daceml.pytorch import DaceModule
from daceml.testing.utils import torch_tensors_close
import daceml.onnx as donnx

batch_size = 2
seq_len = 512
hidden_size = 768

donnx.default_implementation = "pure"

# use eval when checking correctness and compiling
pt_model = BertLayer(BertConfig(hidden_act="relu")).eval().cuda()
dace_model = BertLayer(BertConfig(hidden_act="relu")).eval().cuda()
dace_model.load_state_dict(pt_model.state_dict())
dace_model = DaceModule(dace_model, backward=True, cuda=True)

dace_model.append_post_autodiff_hook("view", lambda f, b: f.view())
dace_model.append_post_autodiff_hook("view", lambda f, b: b.view())

# check forward pass using loss
input = torch.randn([batch_size, seq_len, hidden_size]).cuda()
dy = torch.randn([batch_size, seq_len, hidden_size]).cuda()

pt_input = torch.clone(input)
pt_dy = torch.clone(dy)
dace_input = torch.clone(input)
dace_dy = torch.clone(dy)

pt_output = pt_model(pt_input)
dace_output = dace_model(dace_input)


torch_tensors_close("output", pt_output[0], dace_output)


# test bwd

pt_output[0].backward(pt_dy)
dace_output.backward(dace_dy)

for (name, dace_param), (pt_name,
                         pt_param) in zip(pt_model.named_parameters(),
                                          dace_model.named_parameters()):
    assert 'model.' + name == pt_name
    torch_tensors_close(name, pt_param.detach(), dace_param.detach())
