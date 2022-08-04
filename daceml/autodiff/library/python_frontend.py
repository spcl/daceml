"""
Integration with the dace python frontend
"""

from typing import Optional, Union, Sequence
import itertools
import warnings

import torch
import torch.autograd

from dace import SDFG, SDFGState, config, data
import dace.sdfg.sdfg
from dace.transformation import optimizer
from dace.frontend.python import common
from dace.frontend.common import op_repository
from dace.frontend.python import newast

from daceml.util.utils import all_equal, find_str_not_in_set, expand_nodes
from daceml.autodiff import analysis as autodiff_analysis

from daceml.autodiff.library.library import Array, BackwardPass

TensorOrTensors = Union[str, Sequence[str]]


@op_repository.replaces('torch.autograd.backward')
def backward(pv: newast.ProgramVisitor,
             sdfg: SDFG,
             state: SDFGState,
             tensors: TensorOrTensors,
             grads: Optional[TensorOrTensors] = None):
    """
    Adds a backward pass node to the SDFG.

    This function analyses the the dependency tree of the tensors and computes
    gradients for each Parameter (i.e.
    :class:``daceml.autograd.library.Array``) that was used to compute the
    tensors.
    """

    if isinstance(tensors, str):
        tensors = [tensors]

    if isinstance(grads, str):
        grads = [grads]

    if grads is None:
        grads = []
        # when the tensors are scalars, we can implicity create the grads with ones
        for tensor in tensors:
            tensor_desc = sdfg.arrays[tensor]
            if tensor_desc.total_size == 1:
                constant_name = sdfg._find_new_name("one")
                desc = data.Scalar(tensor_desc.dtype,
                                   transient=True,
                                   storage=tensor_desc.storage)
                sdfg.add_constant(constant_name, 1, dtype=desc)
                sdfg.arrays[constant_name] = desc
                grads.append(constant_name)
            else:
                raise common.DaceSyntaxError(
                    pv, None,
                    "grad can be implicitly created only for scalar outputs")

    if len(grads) != len(tensors):
        raise common.DaceSyntaxError(
            pv, None,
            "grads and tensors must correspond, but they were not the same length"
        )

    for grad, tensor in zip(grads, tensors):
        if grad not in sdfg.arrays and grad not in sdfg.constants_prop:
            raise common.DaceSyntaxError(
                pv, None, "Gradient {} is not an array".format(grad))
        if tensor not in sdfg.arrays:
            raise common.DaceSyntaxError(
                pv, None, "Tensor {} is not an array".format(tensor))

        grad_desc = sdfg.arrays[
            grad] if grad in sdfg.arrays else sdfg.constants_prop[grad][0]

        if not all_equal(grad_desc.shape, sdfg.arrays[tensor].shape):
            raise common.DaceSyntaxError(
                pv, None,
                "Gradient {} and tensor {} have different shapes".format(
                    grad, tensor))

    given_gradients = dict(zip(grads, tensors))

    bwd_node = BackwardPass('backward',
                            inputs=set(itertools.chain(tensors, grads)),
                            outputs=set(),
                            given_gradients=given_gradients)
    state.add_node(bwd_node)

    for inp in itertools.chain(tensors, grads):
        state.add_edge(state.add_read(inp), None, bwd_node, inp,
                       sdfg.make_array_memlet(inp))

    # determine what grdaients to compute
    dependencies = autodiff_analysis.dependency_analysis(sdfg)

    to_compute = {
        dependency
        for tensor in tensors for dependency in dependencies[tensor]
        if isinstance(sdfg.arrays[dependency], Array)
    }

    for param in to_compute:
        param_desc: Array = sdfg.arrays[param]
        grad_name = param_desc.add_gradient_buffer(sdfg, param)

        conn_name = find_str_not_in_set(bwd_node.out_connectors, grad_name)
        bwd_node.required_gradients[param] = conn_name
        bwd_node.add_out_connector(conn_name)

        state.add_edge(bwd_node, conn_name, state.add_write(grad_name), None,
                       sdfg.make_array_memlet(grad_name))


@op_repository.replaces_attribute('Array', 'grad')
def grad(pv: 'ProgramVisitor', sdfg: SDFG, state: SDFGState, arr: str) -> str:
    """
    Returns the name of the gradient buffer of the given array.

    The Array must have been marked as requires_grad_ using
    ``arr.requires_grad_()``, otherwise there will be an error
    """

    if arr not in sdfg.arrays:
        raise common.DaceSyntaxError(pv, None,
                                     "Array {} is not defined".format(arr))
    desc = sdfg.arrays[arr]
    if not isinstance(desc, Array):
        raise common.DaceSyntaxError(
            pv, None,
            "Called .grad on an Array that was not a Parameter. Convert it to a parameter "
            " first using .requires_grad_()")

    return desc.gradient


@op_repository.replaces_method('Array', 'requires_grad_')
@op_repository.replaces_method('Scalar', 'requires_grad_')
def requires_grad_(pv: newast.ProgramVisitor, sdfg: SDFG, state: SDFGState,
                   self: str):
    """
    Converts a array to a Parameter (i.e.
    :class:``daceml.autograd.library.Array``). This creates a descriptor for
    the gradient buffer for this array.
    """

    if self not in sdfg.arrays:
        raise common.DaceSyntaxError(pv, None,
                                     "Array {} is not defined".format(self))
    Array.make_parameter(sdfg, self)


@op_repository.replaces_method('Array', 'backward')
@op_repository.replaces_method('Scalar', 'backward')
def backward_method(pv: newast.ProgramVisitor,
                    sdfg: SDFG,
                    state: SDFGState,
                    self: str,
                    grad: Optional[str] = None):
    """
    Alias for ``torch.autograd.backward(self)``
    """
    backward(pv, sdfg, state, self, grad)


# FIXME hack to make sure BackwardPass nodes are expanded before other library nodes when using the python frontend
_current_opt = config.Config.get("optimizer", "interface")
_call_opt = config.Config.get("optimizer", "transform_on_call")

config.Config.set(
    "optimizer",
    "interface",
    value="daceml.autodiff.library.python_frontend.BackwardPassExpander")
config.Config.set("optimizer", "transform_on_call", value=True)


class BackwardPassExpander(optimizer.Optimizer):
    """
    An optimizer that expands the backward pass nodes of an SDFG.

    Since optimizers are called before other transformations (and compilation)
    in DaceProgram.__call_, this prevents other library nodes from being
    expanded before the backward pass nodes.

    This is clearly a hack and will be replaced with better way to achieve this.
    """
    def optimize(self):
        expand_nodes(self.sdfg, lambda n: isinstance(n, BackwardPass))

        if _call_opt:
            # try to call the original optimizer
            optclass = dace.sdfg.sdfg._get_optimizer_class(_current_opt)
            if optclass is not None:
                opt = optclass(self.sdfg)
                self.sdfg = opt.optimize()

        return self.sdfg
