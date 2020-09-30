from typing import List, Optional, Union

from dace import SDFG, SDFGState, InterstateEdge
from dace.sdfg.nodes import AccessNode

from daceml.autodiff.backward_pass_generator import (
    BackwardPassGenerator,
    AutoDiffException,
    _add_backward_state_to_sdfg
)


def add_backward_pass(
    sdfg: SDFG,
    state: SDFGState,
    outputs: List[Union[AccessNode, str]],
    inputs: List[Union[AccessNode, str]],
):
    """ Experimental: Add a backward pass to `state` using reverse-mode autodiff.

        `inputs`, `outputs` and `grads` can be provided either as `AccessNode`s, or as `str`s, in which case
        the graph will be searched for exactly one matching `AccessNode` with data matching the `str`.

        The SDFG should not contain any inplace operations. It may contain the following nodes:
        - Maps
        - AccessNodes
        - Reductions (Sum, Min, Max)
        - ONNXOps
        - NestedSDFGs containing a single SDFGState (subject to the same constraints). NestedSDFGs may contain multiple
          states as long as all other states are only used for zero initalization.

        Note that the algorithm assumes that all memlets with WCR write into zero initialized arrays!

        When differentiating `ONNXOp`s, the ONNXBackward registry will be checked for any matching backward pass
        implementations. If none are found, the ONNXForward registry will be checked for matching pure implementations.
        If one is found, symbolic differentiation of the pure implementation will be attempted. If this fails, or no
        pure forward implementation is found, the method will fail.


        :param sdfg: the parent SDFG of `state`.
        :param state: the state to add the backward pass to. This is also the state of the forward pass.
        :param outputs: the forward pass outputs of the function to differentiate.
        :param inputs: the inputs w.r.t. which the gradient will be returned.
        :param grads: A list of array names of outputs for which grads will be provided. For each name in the list,
                      the backward state will contain a non-transient array corresponding to the gradient for that array.
                      If the output is a single scalar, this parameter is not required.
    """
    sdfg.validate()

    gen = BackwardPassGenerator(sdfg=sdfg,
                                state=state,
                                outputs=outputs,
                                inputs=inputs,
                                grads=grads)

    # we ignore the input arrs because we are adding the backward state to the same sdfg;
    # the input arrays are already defined in the sdfg
    backward_state, grad_arrs, _, nested_sdfg_forwarded_arrays = gen.backward()

    _add_backward_state_to_sdfg(sdfg, state, backward_state, grad_arrs, nested_sdfg_forwarded_arrays)


