import logging
from typing import Type
import itertools
from collections import OrderedDict

import torch

import dace
from dace import data as dt
from dace.transformation import interstate

from daceml.autodiff.backward_pass_generator import BackwardPassGenerator
from daceml.autodiff.base_abc import AutoDiffException
from daceml.onnx.converters import clean_onnx_name
from daceml.onnx.onnx_importer import create_output_array, ONNXModel

log = logging.getLogger(__name__)


def make_backward_function(model: ONNXModel,
                           apply_strict=False
                           ) -> Type[torch.autograd.Function]:
    """ Convert an ONNXModel to a PyTorch differentiable function. This method should not be used on it's own.
        Instead use the ``backward=True`` parameter of :class:`daceml.pytorch.DaceModule`.

        :param model: the model to convert.
        :param apply_strict: whether to apply strict transformations before creating the backward pass.
        :return: the PyTorch compatible :class:`torch.autograd.Function`.
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
        apply_strict=apply_strict)

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
            arr_name, _ = forward_sdfg.add_array(name + "_array", [1],
                                                 forward_desc.dtype,
                                                 transient=False,
                                                 find_new_name=True)
            copy_state = forward_sdfg.add_state_after(forward_state,
                                                      label="copy_out_" +
                                                      arr_name)
            copy_state.add_edge(copy_state.add_read(name), None,
                                copy_state.add_write(arr_name), None,
                                dace.Memlet(name + "[0]"))
            replaced_scalars[name] = arr_name
        else:
            forward_sdfg.arrays[name].transient = False

    backward_sdfg.validate()

    class DaceFunction(torch.autograd.Function):
        _backward_sdfg = backward_sdfg
        _forward_model = model
        _backward_result = backward_result

        @staticmethod
        def forward(ctx, *inputs):
            # setup the intermediate buffers

            if any(not inp.is_contiguous() for inp in inputs):
                log.warning("forced to copy input since it was not contiguous")

            copied_inputs = tuple(
                inp if inp.is_contiguous else inp.contiguous()
                for inp in inputs)

            # prepare the arguments
            inputs, params, symbols, outputs = model._call_args(
                args=copied_inputs, kwargs={})

            # create the empty tensors we need for the intermediate values
            for inp, val in backward_input_arrays.items():
                if isinstance(val, dt.Scalar):
                    # the value we need is actually in an array
                    inp = replaced_scalars[inp]

                if inp not in inputs and inp not in outputs and inp not in params:
                    inputs[inp] = create_output_array(symbols,
                                                      forward_sdfg.arrays[inp],
                                                      use_torch=True)

            DaceFunction._forward_model.sdfg(**inputs, **symbols, **params,
                                             **outputs)

            def _get_arr(name, desc):
                if isinstance(desc, dt.Scalar):
                    name = replaced_scalars[name]
                if name in inputs:
                    value = inputs[name]
                elif name in outputs:
                    value = outputs[name]
                elif name in params:
                    value = params[name]
                else:
                    raise AutoDiffException(
                        f"Could not get value of array {name}")

                if isinstance(desc, dt.Scalar):
                    return value.numpy()[0]
                else:
                    return value

            # save the arrays we need for the backward pass
            backward_inputs = {
                name: _get_arr(name, desc)
                for name, desc in backward_input_arrays.items()
            }
            ctx.dace_backward_inputs = backward_inputs
            ctx.dace_symbols = symbols

            if len(outputs) == 1:
                return next(iter(outputs.values()))

            return tuple(outputs.values())

        @staticmethod
        def backward(ctx, *grads):
            backward_inputs = ctx.dace_backward_inputs

            if len(grads) != len(model.outputs):
                raise ValueError("Expected to receive {} grads, got {}".format(
                    len(model.outputs), len(grads)))

            given_grads = dict(
                zip((DaceFunction._backward_result.given_grad_names[
                    clean_onnx_name(outp)] for outp in model.outputs), grads))
            for name, value in given_grads.items():
                if not isinstance(value, torch.Tensor):
                    raise ValueError(
                        "Unsupported input with type {};"
                        " currently only tensor inputs are supported".format(
                            type(value)))
                if not value.is_contiguous():
                    log.warning(
                        "forced to copy input since it was not contiguous")
                    given_grads[name] = value.contiguous()

            # these are the grads we will calculate
            input_grad_names = [
                DaceFunction._backward_result.required_grad_names[
                    clean_onnx_name(inp)]
                for inp in itertools.chain(model.inputs)
            ]

            # init the grads we will calculate with zeros
            grad_values = OrderedDict()
            for name in input_grad_names:
                grad_values[name] = create_output_array(
                    ctx.dace_symbols,
                    backward_grad_arrays[name],
                    use_torch=True,
                    zeros=True)

            DaceFunction._backward_sdfg(**grad_values, **backward_inputs,
                                        **given_grads)

            return tuple(grad_values.values())

    return DaceFunction
