import logging
from typing import Type, Tuple, Dict
import itertools
from collections import OrderedDict

import torch

import dace
from dace import data as dt

from daceml.autodiff.backward_pass_generator import BackwardPassGenerator
from daceml.autodiff.base_abc import AutoDiffException, BackwardResult
from daceml.onnx.converters import clean_onnx_name
from daceml.onnx.onnx_importer import create_output_array, ONNXModel
from daceml import transformation

log = logging.getLogger(__name__)


def make_backward_function(
    model: ONNXModel,
    apply_strict=False
) -> Tuple[dace.SDFG, dace.SDFG, BackwardResult, Dict[str, dt.Data]]:
    """ Convert an ONNXModel to a PyTorch differentiable function. This method should not be used on it's own.
        Instead use the ``backward=True`` parameter of :class:`daceml.pytorch.DaceModule`.

        :param model: the model to convert.
        :param apply_strict: whether to apply strict transformations before creating the backward pass.
        :return: the PyTorch compatible :class:`torch.autograd.Function`. TODO Update
    """

    if len(model.sdfg.nodes()) != 1:
        raise AutoDiffException(
            "Expected to find exactly one SDFGState, found {}".format(
                len(model.sdfg.nodes())))

    forward_sdfg = model.sdfg
    forward_state = model.sdfg.nodes()[0]

    backward_sdfg = dace.SDFG(forward_sdfg.name + "_backward")
    backward_state = backward_sdfg.add_state()

    gen = BackwardPassGenerator(
        sdfg=forward_sdfg,
        state=forward_state,
        given_gradients=[clean_onnx_name(name) for name in model.outputs],
        required_gradients=[clean_onnx_name(name) for name in model.inputs],
        backward_sdfg=backward_sdfg,
        backward_state=backward_state,
        apply_strict=apply_strict,
        zero_non_transients=False)

    backward_result, backward_grad_arrays, backward_input_arrays = gen.backward(
    )

    replaced_scalars = {}
    for name, desc in backward_input_arrays.items():
        if name not in forward_sdfg.arrays:
            raise AutoDiffException(
                "Expected to find array with name '{}' in SDFG".format(name))

        forward_desc = forward_sdfg.arrays[name]
        # we will save this output and pass it to the backward pass

        # Views should not be forwarded. Instead the backward pass generator should forward the source of the view,
        # and rebuild the sequence of required views in the backward pass.
        assert type(forward_desc) is not dt.View
        if isinstance(forward_desc, dt.Scalar):
            # we can't return scalars from SDFGs, so we add a copy to an array of size 1
            fwd_arr_name, _ = forward_sdfg.add_array(
                name + "_array", [1],
                forward_desc.dtype,
                transient=False,
                storage=forward_desc.storage,
                find_new_name=True)
            bwd_arr_name, _ = backward_sdfg.add_array(
                name + "_array", [1],
                forward_desc.dtype,
                transient=False,
                storage=forward_desc.storage,
                find_new_name=True)
            backward_sdfg.arrays[name].transient = True

            fwd_copy_state = forward_sdfg.add_state_after(forward_state,
                                                          label="copy_out_" +
                                                          fwd_arr_name)
            bwd_copy_state = backward_sdfg.add_state_before(backward_state,
                                                            label="copy_in_" +
                                                            bwd_arr_name)
            fwd_copy_state.add_edge(fwd_copy_state.add_read(name), None,
                                    fwd_copy_state.add_write(fwd_arr_name),
                                    None, dace.Memlet(name + "[0]"))

            bwd_copy_state.add_edge(bwd_copy_state.add_read(bwd_arr_name),
                                    None, bwd_copy_state.add_write(name), None,
                                    dace.Memlet(name + "[0]"))
            replaced_scalars[name] = fwd_arr_name
        else:
            forward_sdfg.arrays[name].transient = False

    backward_sdfg.validate()

    return forward_sdfg, backward_sdfg, backward_result, backward_input_arrays
