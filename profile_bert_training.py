import torch
import time

from dace.transformation.dataflow import RedundantSecondArray
from transformers import BertLayer, BertConfig

from daceml.ort_ln import DetectLN
from daceml.pytorch import DaceModule
from daceml.testing.utils import torch_tensors_close
import daceml.onnx as donnx
from daceml.transformation import HorizontalEinsumFusion
from daceml.util import utils

batch_size = 2
seq_len = 512
hidden_size = 768

donnx.default_implementation = "pure"
donnx.ONNXSoftmax.default_implementation = "onnxruntime"

# use eval when checking correctness and compiling
pt_model = BertLayer(BertConfig(hidden_act="relu")).eval().cuda()
dace_model = BertLayer(BertConfig(hidden_act="relu")).eval().cuda()
dace_model.load_state_dict(pt_model.state_dict())
dace_model = DaceModule(dace_model, backward=True, cuda=True)


def detect_ln(module: DaceModule):
    module.sdfg.apply_transformations_repeated(DetectLN)
dace_model.append_post_onnx_hook("detect_lns", detect_ln)

def einsum_fusion(module: DaceModule):
    donnx.ONNXTranspose.default_implementation = "einsum"
    def expand_predicate(node):

        for s in ["MatMul_61", "MatMul_54", "Transpose_37", "Transpose_52", "Transpose_53", "Transpose_62"]:
            if s in node.label:
                return False
        return isinstance(node, (donnx.ONNXMatMul, donnx.ONNXTranspose))
    utils.expand_onnx_nodes(module.sdfg, expand_predicate=expand_predicate)
    module.sdfg.apply_strict_transformations()
    module.sdfg.apply_transformations_repeated(HorizontalEinsumFusion)
    donnx.ONNXTranspose.default_implementation = "pure"

dace_model.append_post_onnx_hook("einsum_fusion", einsum_fusion)
dace_model.append_post_onnx_hook("redundant_second_array", lambda m: m.sdfg.apply_transformations_repeated(RedundantSecondArray))

# check forward pass using loss
input = torch.randn([batch_size, seq_len, hidden_size]).cuda()
dy = torch.randn([batch_size, seq_len, hidden_size]).cuda()

pt_input = torch.clone(input)
pt_dy = torch.clone(dy)
dace_input = torch.clone(input)
dace_dy = torch.clone(dy)

## CHECK CORRECTNESS + COMPILE ONCE
dace_output = dace_model(dace_input)
dace_output.backward(dace_dy)

pt_output = pt_model(pt_input)
time.sleep(0.1)
pt_output[0].backward(pt_dy)

torch_tensors_close("output", pt_output[0], dace_output)

for (name, dace_param), (pt_name,
                         pt_param) in zip(pt_model.named_parameters(),
                                          dace_model.named_parameters()):
    assert 'model.' + name == pt_name
    torch_tensors_close(name, pt_param.grad, dace_param.grad)


time.sleep(1)

# TIMING RUN
dace_output = dace_model(dace_input)
dace_output.backward(dace_dy)




