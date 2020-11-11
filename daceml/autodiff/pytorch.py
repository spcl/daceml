import logging
from typing import Type
from collections import OrderedDict

import torch

import dace

from daceml.autodiff.backward_pass_generator import BackwardPassGenerator
from daceml.autodiff.base_abc import AutoDiffException
from daceml.onnx.converters import clean_onnx_name
from daceml.onnx.onnx_importer import create_output_array, ONNXModel

log = logging.getLogger(__name__)


def make_backward_function(model: ONNXModel) -> Type[torch.autograd.Function]:
    """ Convert an ONNXModel to a PyTorch differentiable function.

        :param model: the model to convert.
        :return: the PyTorch compatible :class:`torch.autograd.Function`.
    """

    if len(model.sdfg.nodes()) != 1:
        raise AutoDiffException(
            "Expected to find exactly one SDFGState, found {}".format(
                len(model.sdfg.nodes())))

    forward_sdfg = model.sdfg

    backward_sdfg = dace.SDFG(forward_sdfg.name + "_backward")
    backward_state = backward_sdfg.add_state()

    gen = BackwardPassGenerator(
        sdfg=forward_sdfg,
        state=model.sdfg.nodes()[0],
        given_gradients=[clean_onnx_name(name) for name in model.outputs],
        required_gradients=[clean_onnx_name(name) for name in model.inputs],
        backward_sdfg=backward_sdfg,
        backward_state=backward_state)

    backward_result, backward_grad_arrays, backward_input_arrays = gen.backward()

    for name, desc in backward_input_arrays.items():
        if name not in forward_sdfg.arrays:
            raise AutoDiffException(
                "Expected to find array with name '{}' in SDFG".format(name))

        # we will save this output and pass it to the backward pass
        forward_sdfg.arrays[name].transient = False

    forward_sdfg.view()
    backward_sdfg.view()

    class DaceFunction(torch.autograd.Function):
        _backward_sdfg = backward_sdfg
        _forward_model = model
        _backward_result = backward_result

        @staticmethod
        def forward(ctx, *inputs):
            # setup the intermediate buffers

            copied_inputs = []
            for inp in inputs:
                if type(inp) is not torch.Tensor:
                    raise ValueError(
                        "Currently only tensor inputs are supported")
                if not inp.is_contiguous():
                    log.warning(
                        "forced to copy input since it was not contiguous")
                    inp = inp.contiguous()
                copied_inputs.append(inp)

            # prepare the arguments
            inputs, params, symbols, outputs = model._call_args(
                args=copied_inputs, kwargs={})

            # add the intermediate values we need as empty tensors
            filtered_backward_input_arrays = {
                inp: val
                for inp, val in backward_input_arrays.items()
                if inp not in inputs and inp not in outputs
            }
            inputs.update({
                name: create_output_array(symbols, desc, use_torch=True)
                for name, desc, in filtered_backward_input_arrays.items()
            })

            DaceFunction._forward_model.sdfg(**inputs, **symbols, **params,
                                             **outputs)

            # save the arrays we need for the backward pass
            backward_inputs = {
                name: inputs[name] if name in inputs else outputs[name]
                for name in backward_input_arrays
            }
            ctx.dace_backward_inputs = backward_inputs
            ctx.dace_symbols = symbols

            if len(outputs) == 1:
                return next(iter(outputs.values()))

            return tuple(outputs.values())

        @classmethod
        def backward(cls, ctx, *grads):
            backward_inputs = ctx.dace_backward_inputs

            if len(grads) != len(model.outputs):
                raise ValueError("Expected to receive {} grads, got {}".format(
                    len(model.outputs), len(grads)))

            given_grads = dict(
                zip((DaceFunction._backward_result.given_grad_names[clean_onnx_name(outp)]
                     for outp in model.outputs), grads))
            for name, value in given_grads.items():
                if type(value) is not torch.Tensor:
                    raise ValueError(
                        "Currently only tensor inputs are supported")
                if not value.is_contiguous():
                    log.warning(
                        "forced to copy input since it was not contiguous")
                    given_grads[name] = value.contiguous()

            # these are the grads we will calculate
            input_grad_names = [
                DaceFunction._backward_result.required_grad_names[clean_onnx_name(inp)] for inp in model.inputs
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
