import logging
from typing import Type, Tuple, Dict, List
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
    required_grads: List[str],
) -> Tuple[dace.SDFG, dace.SDFG, BackwardResult, Dict[str, dt.Data]]:
    """ Convert an ONNXModel to a PyTorch differentiable function. This method should not be used on its own.
        Instead use the ``backward=True`` parameter of :class:`daceml.pytorch.DaceModule`.

        :param model: the model to convert.
        :param required_grads: the list of inputs names of the module that we must compute gradients for.
        :return: A 4-tuple of forward SDFG, backward SDFG, backward result, and input arrays for 
                 backward pass (as mapping of names to DaCe data descriptors).
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
        required_gradients=required_grads,
        backward_sdfg=backward_sdfg,
        backward_state=backward_state,
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
            bwd_arr_name, bwd_desc = backward_sdfg.add_array(
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
            replaced_scalars[name] = (bwd_arr_name, bwd_desc)
        else:
            forward_sdfg.arrays[name].transient = False

    for orig_name, (replaced_name, replaced_desc) in replaced_scalars.items():
        del backward_input_arrays[orig_name]
        backward_input_arrays[replaced_name] = replaced_desc

    for fwd_name, bwd_name in backward_result.required_grad_names.items():
        desc = backward_sdfg.arrays[bwd_name]
        if isinstance(desc, dt.Scalar):
            arr_name, arr_desc = backward_sdfg.add_array(bwd_name + "_array",
                                                         [1],
                                                         desc.dtype,
                                                         transient=False,
                                                         storage=desc.storage,
                                                         find_new_name=True)
            desc.transient = True
            bwd_copy_state = backward_sdfg.add_state_after(backward_state,
                                                           label="copy_out_" +
                                                           bwd_name)
            bwd_copy_state.add_edge(bwd_copy_state.add_read(bwd_name), None,
                                    bwd_copy_state.add_write(arr_name), None,
                                    dace.Memlet(bwd_name + "[0]"))
            backward_result.required_grad_names[fwd_name] = arr_name

    backward_sdfg.validate()

    return forward_sdfg, backward_sdfg, backward_result, backward_input_arrays
