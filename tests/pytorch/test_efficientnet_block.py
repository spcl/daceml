import pytest
import torch
from dace.library import change_default
from efficientnet_pytorch import get_model_params
from efficientnet_pytorch.model import MBConvBlock

import daceml.onnx as donnx
from daceml.onnx.op_implementations.cudnn_implementations import CudnnConvolution
from daceml.pytorch import DaceModule
from daceml.testing import torch_tensors_close


@pytest.mark.pure
@pytest.mark.gpu
@pytest.mark.parametrize("bn_impl", ["cuDNN", "pure"])
def test_mbconv(bn_impl):
    with change_default(donnx.ONNXConv, "cuDNN"),\
        change_default(donnx.ONNXBatchNormalization, bn_impl):

        with torch.no_grad():
            dace_inputs = torch.rand(8, 32, 224, 224).cuda()
            torch_inputs = torch.clone(dace_inputs)

        block_params, global_params = get_model_params("efficientnet-b0", {})

        torch_model = MBConvBlock(block_params[0], global_params).cuda()
        torch_model.set_swish(memory_efficient=False)
        dace_model = MBConvBlock(block_params[0], global_params).cuda()
        dace_model.set_swish(memory_efficient=False)
        dace_model = DaceModule(dace_model, training=True)
        dace_model.model.load_state_dict(torch_model.state_dict())

        for (dace_name,
             dace_value), (torch_name,
                           value) in zip(dace_model.model.state_dict().items(),
                                         torch_model.state_dict().items()):
            assert dace_name == torch_name
            torch_tensors_close(dace_name, value, dace_value)

        CudnnConvolution.default_algorithm = "gemm"

        dace_output = dace_model(dace_inputs)
        torch.transients = dict(zip(dace_model.dace_model.outputs,
                                    dace_output))
        torch_output = torch_model(torch_inputs)
        torch_tensors_close("output", torch_output, dace_output)

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
