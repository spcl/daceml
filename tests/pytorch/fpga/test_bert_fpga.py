import numpy as np
import torch
from dace.transformation.dataflow import RedundantSecondArray
from transformers import BertConfig, BertLayer

import daceml.onnx as donnx
from daceml.pytorch import DaceModule
from daceml.transformation import ConstantFolding
from dace.transformation.interstate import FPGATransformSDFG, InlineSDFG


def test_bert_cf():
    # This is needed, for the default impl
    donnx.default_implementation = "pure"

    ##### Tiny BERT
    B = 2
    H = 4
    P = 8
    N = P * H
    SM, SN = 16, 16

    batch_size = 8
    seq_len = 16
    hidden_size = N
    vocab_size = 1024

    input = torch.randn([B, seq_len, hidden_size])

    ptmodel = BertLayer(
        BertConfig(vocab_size=vocab_size,
                   hidden_size=hidden_size,
                   num_hidden_layers=H,
                   num_attention_heads=H)).eval()
    pt_outputs = ptmodel(input.clone())
    donnx.ONNXCast.default_implementation = "onnxruntime"
    dace_model = DaceModule(ptmodel, train=False)
    dace_outputs0 = dace_model(input.clone())
    dace_model.dace_model.sdfg.save("/tmp/out.sdfg")
    dace_model.dace_model.sdfg.apply_transformations_repeated(
        [ConstantFolding, RedundantSecondArray], validate_all=True)
    dace_model.dace_model.sdfg.save("/tmp/bert_enc.sdfg")
    dace_model.dace_model.sdfg.apply_strict_transformations()

    dace_outputs1 = dace_model(input.clone())

    diff = np.abs(dace_outputs0 - pt_outputs[0].detach().numpy())
    assert np.max(diff) < 1e-5
    assert np.allclose(dace_outputs1, dace_outputs0)

    #### FPGA
    sdfg = dace_model.sdfg
    ###################################################
    # Transform to FPGA
    import pdb
    pdb.set_trace()
    # TODO: why this fails if I first dont't execute it through daceml?
    donnx.ONNXMatMul.default_implementation = "fpga"
    donnx.ONNXReshape.default_implementation = "fpga"
    donnx.ONNXSoftmax.default_implementation = "fpga"
    donnx.ONNXReduceSum.default_implementation = "fpga"

    sdfg.apply_transformations([FPGATransformSDFG])
    sdfg.expand_library_nodes()
    sdfg.save('/tmp/out_fpga_pre_inlined.sdfg')

    sdfg.apply_transformations_repeated([InlineSDFG])
    # sdfg.apply_transformations_repeated(PruneConnectors)
    # sdfg.states()[0].location["is_FPGA_kernel"] = False
    # sdfg.states()[0].nodes()[0].sdfg.states()[0].location["is_FPGA_kernel"] = False
    sdfg.save('/tmp/out_fpga.sdfg')
    dace_output_fpga = dace_model(input.clone())
    diff = np.abs(dace_output_fpga - pt_outputs[0].detach().numpy())
    print("Diff: ", diff)
    assert diff < 1e-6


#test_bert_cf()
