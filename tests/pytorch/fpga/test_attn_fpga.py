import torch
import numpy as np
import pytest

from daceml.pytorch import DaceModule

from dace.transformation.dataflow import RedundantSecondArray
from daceml.transformation import ConstantFolding
import daceml.onnx as donnx
donnx.default_implementation = "pure"
from dace.transformation.interstate import FPGATransformSDFG, InlineSDFG
from dace.transformation.dataflow import PruneConnectors
from dace.transformation.dataflow import streaming_memory as sm
from dace import StorageType
from dace import SDFG

@pytest.mark.ort
def test_attn(execute_cpu_dace = False):
    # BERT_base: H=12, P=64 N=768, emb=4N, SM=SN=128
    # BERT_large: H=16, P=64, N=1024, emb=4N, SM=SN=512

    ##### Tiny BERT
    B = 2
    H = 4
    P = 8
    N = P * H
    SM, SN = 16, 16

    ##### SMALL BERT
    # B = 2
    # H = 12
    # P = 32
    # N = P * H
    # SM, SN = 32, 32

    ##### BASE BERT
    # B = 2
    # H = 12
    # P = 64
    # N = P * H
    # SM, SN = 128, 128

    K, Q, V = [
        torch.randn([SM, B, N]),
        torch.randn([SN, B, N]),
        torch.randn([SM, B, N])
    ]
    ptmodel = torch.nn.MultiheadAttention(N, H, bias=False)

    donnx.ONNXCast.default_implementation = "onnxruntime"


    pt_outputs = ptmodel(Q, K, V)

    if execute_cpu_dace:
        dace_model = DaceModule(ptmodel, dummy_inputs=(Q,K,V))
        # dace_outputs_0 = dace_model(Q, K, V)

    else:
        dace_model = DaceModule(ptmodel, dummy_inputs=(Q,K,V))

    dace_model.sdfg.save('/tmp/out_pre.sdfg')

    ################################################
    # Apply transformations
    dace_model.dace_model.sdfg.apply_transformations_repeated(
        [ConstantFolding, RedundantSecondArray], validate_all=True, print_report=True)
    dace_model.sdfg.save('/tmp/out.sdfg')

    if execute_cpu_dace:
        dace_outputs_1 = dace_model(Q, K, V)
        assert np.allclose(pt_outputs[0].detach().numpy(),
                           dace_outputs_1[0],
                           atol=1e-06)
        assert np.allclose(pt_outputs[1].detach().numpy(),
                           dace_outputs_1[1],
                           atol=1e-06)
    # dace_model.sdfg.from_file('/tmp/out.sdfg')
    sdfg = dace_model.sdfg
    # import pdb
    # pdb.set_trace()

    ###################################################
    # Transform to FPGA

    #TODO: why this fails if I first dont't execute it through daceml?
    donnx.ONNXMatMul.default_implementation = "fpga"
    donnx.ONNXReshape.default_implementation = "fpga"
    donnx.ONNXSoftmax.default_implementation = "fpga"
    donnx.ONNXReduceSum.default_implementation = "fpga"

    sdfg.apply_transformations([FPGATransformSDFG])
    sdfg.expand_library_nodes()
    sdfg.save('/tmp/out_fpga_pre_inlined.sdfg')

    sdfg.apply_transformations_repeated([InlineSDFG])
    sdfg.apply_transformations_repeated(PruneConnectors)
    # sdfg.states()[0].location["is_FPGA_kernel"] = False
    # sdfg.states()[0].nodes()[0].sdfg.states()[0].location["is_FPGA_kernel"] = False
    sdfg.save('/tmp/out_fpga.sdfg')

    # Streaming composition
    sdfg.apply_transformations_repeated([InlineSDFG, sm.StreamingComposition], [{}, {"storage": StorageType.FPGA_Local}], print_report=True)
    import pdb
    pdb.set_trace()
    sdfg.save('/tmp/out_fpga.sdfg')

    # Load from file
    # sdfg = SDFG.from_file('/tmp/out_fpga.sdfg')

    dace_output_fpga = dace_model(Q,K,V)

    diff0 = np.linalg.norm(pt_outputs[0].detach().numpy() - dace_output_fpga[0]) / dace_output_fpga[0].size
    diff1 = np.linalg.norm(pt_outputs[1].detach().numpy() - dace_output_fpga[1]) / dace_output_fpga[1].size


    assert np.allclose(pt_outputs[0].detach().numpy(),
                       dace_output_fpga[0],
                       atol=1e-06)
    assert np.allclose(pt_outputs[1].detach().numpy(),
                       dace_output_fpga[1],
                       atol=1e-06)



if __name__ == "__main__":
    test_attn(False)