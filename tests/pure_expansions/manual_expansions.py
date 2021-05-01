import dace
from dace.dtypes import StorageType
import torch
from torch import nn
from torch.nn import functional as F

from daceml.onnx.op_implementations.cpp_implementations import add_ln_tasklet, add_ln_tasklet_bwd, DetectLN
from daceml.pytorch import DaceModule
from daceml.testing.utils import torch_tensors_close


def test_ln_detection():
    inp = torch.rand(2, 512, 768).cuda()
    dy = torch.rand(2, 512, 768).cuda()
    dace_dy = torch.clone(dy)
    dace_inp = torch.clone(inp)
    dace_inp.requires_grad = True
    inp.requires_grad = True

    module = nn.LayerNorm([768]).cuda()
    dace_module = nn.LayerNorm([768]).cuda()
    dace_module.load_state_dict(module.state_dict())

    dace_module = DaceModule(dace_module, cuda=True, backward=True)

    def detect_ln(module: DaceModule):
        module.sdfg.apply_transformations_repeated(DetectLN)

    dace_module.prepend_post_onnx_hook("detect_ln", detect_ln)
    dace_module.append_post_autodiff_hook(
        "expand", lambda f, b: f.expand_library_nodes())

    dace_outp = dace_module(dace_inp)
    pt_outp = module(inp)
    torch_tensors_close("output", pt_outp, dace_outp)

    pt_outp.backward(dy)
    dace_outp.backward(dace_dy)
    torch_tensors_close("weightgrad", module.weight.grad,
                        dace_module.model.weight.grad)
    torch_tensors_close("biasgrad", module.bias.grad,
                        dace_module.model.bias.grad)
    torch_tensors_close("grad", inp.grad, dace_inp.grad)


def test_layernorm():
    inp = torch.rand(2, 512, 768).cuda()
    dace_inp = torch.clone(inp)
    inp.requires_grad = True
    weight = torch.rand(768).cuda()
    dace_weight = torch.clone(weight)
    weight.requires_grad = True

    bias = torch.rand(768).cuda()
    dace_bias = torch.clone(bias)
    bias.requires_grad = True

    outp = F.layer_norm(inp, [768], weight=weight, bias=bias)
    doutp = torch.rand_like(outp)

    sdfg = dace.SDFG("test")
    sdfg.add_array("inp", [2, 512, 768],
                   dace.float32,
                   storage=StorageType.GPU_Global)
    sdfg.add_array("weight", [768],
                   dace.float32,
                   storage=StorageType.GPU_Global)
    sdfg.add_array("bias", [768], dace.float32, storage=StorageType.GPU_Global)
    sdfg.add_array("mean", [2, 512, 1],
                   dace.float32,
                   storage=StorageType.GPU_Global)
    sdfg.add_array("inv_std_var", [2, 512, 1],
                   dace.float32,
                   storage=StorageType.GPU_Global,
                   transient=True)
    sdfg.add_array("outp", [2, 512, 768],
                   dace.float32,
                   storage=StorageType.GPU_Global)
    sdfg.add_array("doutp", [2, 512, 768],
                   dace.float32,
                   storage=StorageType.GPU_Global)
    sdfg.add_array("dX", [2, 512, 768],
                   dace.float32,
                   storage=StorageType.GPU_Global)
    sdfg.add_array("dbias", [2, 512, 1],
                   dace.float32,
                   storage=StorageType.GPU_Global)
    sdfg.add_array("dscale", [2, 512, 1],
                   dace.float32,
                   storage=StorageType.GPU_Global)

    state = sdfg.add_state()

    tasklet = add_ln_tasklet(state, [2, 512, 768], 2)

    state.add_edge(state.add_read("inp"), None, tasklet, "_X",
                   sdfg.make_array_memlet("inp"))
    state.add_edge(state.add_read("weight"), None, tasklet, "_scale",
                   sdfg.make_array_memlet("weight"))
    state.add_edge(state.add_read("bias"), None, tasklet, "_bias",
                   sdfg.make_array_memlet("bias"))
    state.add_edge(tasklet, "_Y", state.add_write("outp"), None,
                   sdfg.make_array_memlet("outp"))

    state.add_edge(tasklet, "_mean", state.add_write("mean"), None,
                   sdfg.make_array_memlet("mean"))
    state.add_edge(tasklet, "_inv_std_var", state.add_write("inv_std_var"),
                   None, sdfg.make_array_memlet("inv_std_var"))

    bwd_state = sdfg.add_state_after(state)
    tasklet_bwd = add_ln_tasklet_bwd(bwd_state, [2, 512, 768], 2)

    bwd_state.add_edge(bwd_state.add_read("doutp"), None, tasklet_bwd, "_dY",
                       sdfg.make_array_memlet("doutp"))
    bwd_state.add_edge(bwd_state.add_read("inp"), None, tasklet_bwd, "_X",
                       sdfg.make_array_memlet("inp"))
    bwd_state.add_edge(bwd_state.add_read("weight"), None, tasklet_bwd,
                       "_scale", sdfg.make_array_memlet("weight"))
    bwd_state.add_edge(bwd_state.add_read("bias"), None, tasklet_bwd, "_bias",
                       sdfg.make_array_memlet("bias"))
    bwd_state.add_edge(bwd_state.add_read("inv_std_var"), None, tasklet_bwd,
                       "_inv_std_var", sdfg.make_array_memlet("inv_std_var"))
    bwd_state.add_edge(bwd_state.add_read("mean"), None, tasklet_bwd, "_mean",
                       sdfg.make_array_memlet("mean"))

    bwd_state.add_edge(tasklet_bwd, "_dbias", bwd_state.add_write("dbias"),
                       None, sdfg.make_array_memlet("dbias"))
    bwd_state.add_edge(tasklet_bwd, "_dscale", bwd_state.add_write("dscale"),
                       None, sdfg.make_array_memlet("dscale"))
    bwd_state.add_edge(tasklet_bwd, "_dX", bwd_state.add_write("dX"), None,
                       sdfg.make_array_memlet("dX"))

    dace_outp = torch.zeros_like(inp)
    mean = torch.zeros(2, 512, 1).cuda()

    dinp = torch.ones_like(inp)
    dbias = torch.ones_like(bias)
    dscale = torch.ones_like(weight)

    sdfg(inp=dace_inp,
         weight=dace_weight,
         bias=dace_bias,
         outp=dace_outp,
         mean=mean,
         dX=dinp,
         dbias=dbias,
         dscale=dscale,
         doutp=doutp)

    outp.backward(doutp)

    torch_tensors_close("output", outp, dace_outp)
    torch_tensors_close("mean", inp.mean(axis=-1, keepdims=True), mean)
    torch_tensors_close("ingrad", inp.grad, dinp)
    torch_tensors_close("weightgrad", weight.grad, dscale)
    torch_tensors_close("bgrad", bias.grad, dbias)
