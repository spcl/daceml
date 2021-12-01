import copy
import itertools
import logging
import typing

import dace
import numpy as np
from dace import SDFGState, SDFG, nodes, subsets
from dace.frontend.common import create_einsum_sdfg
from dace.sdfg.nodes import Node

from daceml.onnx import converters
from daceml.onnx.forward_implementation_abc import ONNXForward
from daceml.onnx.nodes import onnx_op
from daceml.onnx.op_implementations.utils import op_implementation, program_for_node, empty_sdfg_for_node, \
    python_pure_op_implementation
from daceml.transformation import constant_folding
from daceml.transformation.replacement import onnx_constant_or_none
from daceml.util.utils import in_desc_with_name, out_desc_with_name, in_edge_with_name, iterables_equal

log = logging.getLogger(__name__)


@op_implementation(op="Log", name="pure")
class PureLog(ONNXForward):
    @staticmethod
    def forward_can_be_applied(node: onnx_op.ONNXOp, state: SDFGState,
                               sdfg: SDFG) -> bool:
        return in_desc_with_name(node, state, sdfg, 'input').dtype in [
            dace.float16, dace.float32, dace.float64
        ]

    @staticmethod
    def forward(node: onnx_op.ONNXOp, state: SDFGState,
                sdfg: SDFG) -> typing.Union[Node, SDFG]:
        def prog(input, output):
            output[:] = dace.elementwise(lambda x: log(x), input)

        return program_for_node(prog, sdfg, state, node)


@op_implementation(op="Sqrt", name="pure")
class PureSqrt(ONNXForward):
    @staticmethod
    def forward_can_be_applied(node: onnx_op.ONNXOp, state: SDFGState,
                               sdfg: SDFG) -> bool:
        return in_desc_with_name(node, state, sdfg, 'X').dtype in [
            dace.float16, dace.float32, dace.float64
        ]

    @staticmethod
    def forward(node: onnx_op.ONNXOp, state: SDFGState,
                sdfg: SDFG) -> typing.Union[Node, SDFG]:
        def prog(X, Y):
            Y[:] = dace.elementwise(lambda x: sqrt(x), X)

        return program_for_node(prog, sdfg, state, node)


@op_implementation(op="Pow", name="pure")
class PurePow(ONNXForward):
    @staticmethod
    def forward_can_be_applied(node: onnx_op.ONNXOp, state: SDFGState,
                               sdfg: SDFG) -> bool:
        return in_desc_with_name(node, state, sdfg, 'X').dtype in [
            dace.float16, dace.float32, dace.float64
        ]

    @staticmethod
    def forward(node: onnx_op.ONNXOp, state: SDFGState,
                sdfg: SDFG) -> typing.Union[Node, SDFG]:
        def prog(X, Y, Z):
            Z[:] = X**Y

        return program_for_node(prog, sdfg, state, node)


@op_implementation(op="Clip", name="pure")
class PureClip(ONNXForward):
    @staticmethod
    def forward_can_be_applied(node: onnx_op.ONNXOp, state: SDFGState,
                               sdfg: SDFG) -> bool:
        min_node = next(state.in_edges_by_connector(node, 'min')).src
        max_node = next(state.in_edges_by_connector(node, 'max')).src
        # TODO other cases
        return (onnx_constant_or_none(sdfg, min_node) is not None
                and onnx_constant_or_none(sdfg, max_node) is not None)

    @staticmethod
    def forward(node: onnx_op.ONNXOp, state: SDFGState,
                sdfg: SDFG) -> typing.Union[Node, SDFG]:

        min_node = next(state.in_edges_by_connector(node, 'min')).src
        max_node = next(state.in_edges_by_connector(node, 'max')).src
        minval = onnx_constant_or_none(sdfg, min_node)
        maxval = onnx_constant_or_none(sdfg, max_node)

        input_dtype = in_desc_with_name(node, state, sdfg, "input").dtype
        minstr = f"dace.{input_dtype.to_string()}({minval})"
        maxstr = f"dace.{input_dtype.to_string()}({maxval})"

        lfunc = f"lambda x: min(max(x, {minstr}), {maxstr})"

        def prog(input, output):
            output[:] = dace.elementwise(lfunc, input)

        return program_for_node(prog, sdfg, state, node)


@python_pure_op_implementation
def Add(A, B, C):
    C[:] = A + B


@python_pure_op_implementation
def Sub(A, B, C):
    C[:] = A - B


@python_pure_op_implementation
def Mul(A, B, C):
    C[:] = A * B


@python_pure_op_implementation
def Div(A, B, C):
    C[:] = A / B


@python_pure_op_implementation
def Where(condition, X, Y, output):
    output[:] = np.where(condition, X, Y)


@op_implementation(op="ReduceMean", name="pure")
class PureReduceMean(ONNXForward):
    @staticmethod
    def forward(node: onnx_op.ONNXOp, state: SDFGState,
                sdfg: SDFG) -> typing.Union[Node, SDFG]:
        axes = node.axes

        # when keepdims is true, this works but there is a useless copy. We just leave this for now; this can be fixed
        # with a reshape node when those exist.
        def prog(data, reduced):
            reduced[:] = np.mean(data, axis=axes)

        return program_for_node(prog, sdfg, state, node)


@op_implementation(op="Erf", name="pure")
class PureErf(ONNXForward):
    @staticmethod
    def forward_can_be_applied(node: onnx_op.ONNXOp, state: SDFGState,
                               sdfg: SDFG) -> bool:
        return in_desc_with_name(node, state, sdfg, 'input').dtype in [
            dace.float16, dace.float32, dace.float64
        ]

    @staticmethod
    def forward(node: onnx_op.ONNXOp, state: SDFGState,
                sdfg: SDFG) -> typing.Union[Node, SDFG]:
        def prog(input, output):
            output[:] = dace.elementwise(lambda x: erf(x), input)

        return program_for_node(prog, sdfg, state, node)


@op_implementation(op="MatMul", name="pure")
class PureMatMul(ONNXForward):
    @staticmethod
    def forward_can_be_applied(node: onnx_op.ONNXOp, state: SDFGState,
                               sdfg: SDFG) -> bool:
        input0_dim = len(in_desc_with_name(node, state, sdfg, "A").shape)
        input1_dim = len(in_desc_with_name(node, state, sdfg, "B").shape)

        if input0_dim == 1 or input1_dim == 1:
            return False
        return True

    @staticmethod
    def forward(node: onnx_op.ONNXOp, state: SDFGState,
                sdfg: SDFG) -> typing.Union[Node, SDFG]:
        A_desc = in_desc_with_name(node, state, sdfg, "A")
        B_desc = in_desc_with_name(node, state, sdfg, "B")
        Y_desc = out_desc_with_name(node, state, sdfg, "Y")
        input0_dim = A_desc.shape
        input1_dim = B_desc.shape

        # list containing letters from z-a
        letters = [chr(ord('z') - i) for i in range(26)]
        # i j k are used for the last dimensions
        letters = [l for l in letters if l not in ['i', 'j', 'k']]

        if len(input0_dim) == 1:
            if len(input1_dim) != 2:
                raise ValueError("invalid dimensions")
            arg1 = 'k'
            arg2 = 'kj'
            result = 'j'
        elif len(input1_dim) == 1:
            if len(input0_dim) != 2:
                raise ValueError("invalid dimensions")
            arg1 = 'ik'
            arg2 = 'k'
            result = 'i'
        else:
            # build the einsum. The last two dimensions are always just the matrix multiply einsum
            # dace will later specialize to a batched matmul if possible
            arg1 = 'ik'
            arg2 = 'kj'
            result = 'ij'
            if input0_dim[-2] != input0_dim[-1]:
                if dace.symbolic.issymbolic(input0_dim[-2]):
                    log.warning(
                        f"overriding symbol {input0_dim[-2]} with value {input1_dim[-1]} in descriptor of input A of node {node}"
                    )
                    new_shape = list(A_desc.shape)
                    new_shape[-1] = input1_dim[-2]
                    A_desc.shape = new_shape
                elif dace.symbolic.issymbolic(input1_dim[-1]):
                    log.warning(
                        f"overriding symbol {input0_dim[-1]} with value {input0_dim[-2]} in descriptor of input B of node {node}"
                    )
                    new_shape = list(B_desc.shape)
                    new_shape[-2] = input0_dim[-1]
                    B_desc.shape = new_shape
            input0_dim = input0_dim[:-2]
            input1_dim = input1_dim[:-2]
            for dim0, dim1 in itertools.zip_longest(reversed(input0_dim),
                                                    reversed(input1_dim)):
                if dim0 is None:
                    # only dim0 exists
                    letter = letters.pop()
                    arg2 = letter + arg2
                    result = letter + result
                elif dim1 is None:
                    # only dim1 exists
                    letter = letters.pop()
                    arg1 = letter + arg1
                    result = letter + result
                else:
                    # both exist
                    letter = letters.pop()
                    arg1 = letter + arg1
                    arg2 = letter + arg2
                    result = letter + result

        einsum_str = '{},{}->{}'.format(arg1, arg2, result)

        # we lower to an ONNXEinsum node instead straight to the dace einsum to make the autodiff simpler
        nsdfg = dace.SDFG(node.label + "_expansion")
        nstate = nsdfg.add_state()
        einsum_node: nodes.LibraryNode = onnx_op.ONNXEinsum(
            node.label + "_einsum_expansion", equation=einsum_str)

        nstate.add_node(einsum_node)
        einsum_node.add_in_connector("Inputs__0")
        einsum_node.add_in_connector("Inputs__1")
        nsdfg.add_datadesc("A", copy.deepcopy(A_desc))
        nsdfg.add_datadesc("B", copy.deepcopy(B_desc))
        nsdfg.add_datadesc("Y", copy.deepcopy(Y_desc))
        nsdfg.arrays["A"].transient = False
        nsdfg.arrays["B"].transient = False
        nsdfg.arrays["Y"].transient = False

        nstate.add_edge(nstate.add_read("A"), None, einsum_node, "Inputs__0",
                        nsdfg.make_array_memlet("A"))
        nstate.add_edge(nstate.add_read("B"), None, einsum_node, "Inputs__1",
                        nsdfg.make_array_memlet("B"))
        nstate.add_edge(einsum_node, "Output", nstate.add_write("Y"), None,
                        nsdfg.make_array_memlet("Y"))

        return nsdfg


@op_implementation(op="Einsum", name="pure")
class PureEinsum(ONNXForward):
    @staticmethod
    def forward_can_be_applied(node: onnx_op.ONNXOp, state: SDFGState,
                               sdfg: SDFG) -> bool:
        if "..." in node.equation:
            return False
        return True

    @staticmethod
    def forward(node: onnx_op.ONNXOp, state: SDFGState,
                sdfg: SDFG) -> typing.Union[Node, SDFG]:
        nsdfg = dace.SDFG(node.label + "_expansion")
        nstate = nsdfg.add_state()

        for e in node.iter_inputs_in_onnx_order(state):
            desc = copy.deepcopy(
                in_desc_with_name(node, state, sdfg, e.dst_conn))
            desc.transient = False
            nsdfg.add_datadesc(e.dst_conn, desc)
        for e in node.iter_outputs_in_onnx_order(state):
            desc = copy.deepcopy(
                out_desc_with_name(node, state, sdfg, e.src_conn))
            desc.transient = False
            nsdfg.add_datadesc(e.src_conn, desc)

        create_einsum_sdfg(None,
                           nsdfg,
                           nstate,
                           node.equation.replace(" ", ""),
                           *(e.dst_conn
                             for e in node.iter_inputs_in_onnx_order(state)),
                           output="Output")
        return nsdfg


@python_pure_op_implementation
def Identity(input, output):
    output[:] = input


@op_implementation(op="Expand", name="pure")
class PureExpand(ONNXForward):
    """ Handle no-op case for Expand """
    @staticmethod
    def forward_can_be_applied(node: onnx_op.ONNXOp, state: SDFGState,
                               sdfg: SDFG) -> bool:
        return iterables_equal(
            in_desc_with_name(node, state, sdfg, "input").shape,
            out_desc_with_name(node, state, sdfg, "output").shape)

    @staticmethod
    def forward(node: onnx_op.ONNXOp, state: SDFGState,
                sdfg: SDFG) -> typing.Union[Node, SDFG]:

        node.remove_in_connector("shape")
        shape_node = in_edge_with_name(node, state, "shape").src
        constant_folding.remove_node_and_computation(sdfg, state, shape_node)

        def prog(input, output):
            output[:] = input

        return program_for_node(prog, sdfg, state, node)


@op_implementation(op="Reciprocal", name="pure")
class PureReciprocal(ONNXForward):
    @staticmethod
    def forward_can_be_applied(node: onnx_op.ONNXOp, state: SDFGState,
                               sdfg: SDFG) -> bool:
        return in_desc_with_name(node, state, sdfg, 'X').dtype in [
            dace.float16, dace.float32, dace.float64
        ]

    @staticmethod
    def forward(node: onnx_op.ONNXOp, state: SDFGState,
                sdfg: SDFG) -> typing.Union[Node, SDFG]:
        dtype = in_desc_with_name(node, state, sdfg, 'X').dtype
        tanh_lambda = "lambda x: dace.{}(1) / x".format(dtype.to_string())

        def prog(X, Y):
            Y[:] = dace.elementwise(tanh_lambda, X)

        return program_for_node(prog, sdfg, state, node)


@python_pure_op_implementation
def Tanh(input, output):
    output[:] = dace.elementwise(lambda x: tanh(x), input)


@op_implementation(op="ReduceSum", name="pure")
class PureReduceSum(ONNXForward):
    @staticmethod
    def forward(node: onnx_op.ONNXOp, state: SDFGState,
                sdfg: SDFG) -> typing.Union[Node, SDFG]:
        axes = node.axes

        # when keepdims is true, this works but there is a useless copy. We just leave this for now; this can be fixed
        # with a reshape node when those exist.
        def prog(data, reduced):
            reduced[:] = np.sum(data, axis=axes)

        return program_for_node(prog, sdfg, state, node)


@op_implementation(op="ReduceMax", name="pure")
class PureReduceMax(ONNXForward):
    @staticmethod
    def forward(node: onnx_op.ONNXOp, state: SDFGState,
                sdfg: SDFG) -> typing.Union[Node, SDFG]:
        axes = node.axes

        # when keepdims is true, this works but there is a useless copy. We just leave this for now; this can be fixed
        # with a reshape node when those exist.
        def prog(data, reduced):
            reduced[:] = np.max(data, axis=axes)

        return program_for_node(prog, sdfg, state, node)


@op_implementation(op="ReduceMin", name="pure")
class PureReduceMin(ONNXForward):
    @staticmethod
    def forward(node: onnx_op.ONNXOp, state: SDFGState,
                sdfg: SDFG) -> typing.Union[Node, SDFG]:
        axes = node.axes

        # when keepdims is true, this works but there is a useless copy. We just leave this for now; this can be fixed
        # with a reshape node when those exist.
        def prog(data, reduced):
            reduced[:] = np.min(data, axis=axes)

        return program_for_node(prog, sdfg, state, node)


@op_implementation(op="Softmax", name="pure")
class PureSoftmax(ONNXForward):
    @staticmethod
    def forward(node: onnx_op.ONNXOp, state: SDFGState,
                sdfg: SDFG) -> typing.Union[Node, SDFG]:

        axis = node.axis

        reduced_shape = list(
            copy.deepcopy(in_desc_with_name(node, state, sdfg, "input").shape))
        reduced_shape[axis] = 1

        def prog(input, output):
            maximum = np.max(input, axis=axis)
            max_keepdims = np.reshape(maximum, reduced_shape)
            exp_arr = np.exp(input - max_keepdims)
            sum = np.sum(exp_arr, axis=axis)
            sum_keepdims = np.reshape(sum, reduced_shape)
            output[:] = exp_arr / sum_keepdims

        return program_for_node(prog, sdfg, state, node)


@op_implementation(op="Transpose", name="pure")
class PureTranspose(ONNXForward):
    @staticmethod
    def forward(node: onnx_op.ONNXOp, state: SDFGState,
                sdfg: SDFG) -> typing.Union[Node, SDFG]:
        perm = node.perm

        def prog(data, transposed):
            transposed[:] = np.transpose(data, axes=perm)

        return program_for_node(prog, sdfg, state, node)


@op_implementation(op="Cast", name="pure")
class PureCast(ONNXForward):
    @staticmethod
    def forward_can_be_applied(node: onnx_op.ONNXOp, state: SDFGState,
                               sdfg: SDFG) -> bool:

        if (in_desc_with_name(node, state, sdfg,
                              "input").dtype == out_desc_with_name(
                                  node, state, sdfg, "output").dtype):
            return True

        target_type = node.to
        try:
            converters.onnx_tensor_type_to_typeclass(target_type)
        except ValueError:
            return False

        return True

    @staticmethod
    def forward(node: onnx_op.ONNXOp, state: SDFGState,
                sdfg: SDFG) -> typing.Union[Node, SDFG]:
        input_desc = in_desc_with_name(node, state, sdfg, "input")
        output_desc = out_desc_with_name(node, state, sdfg, "output")
        if (input_desc.dtype == output_desc.dtype):

            def prog(input, output):
                output[:] = input

            return program_for_node(prog, sdfg, state, node)
        else:

            nsdfg, nstate, _, _ = empty_sdfg_for_node(sdfg,
                                                      state,
                                                      node,
                                                      add_access_nodes=False)

            shape = out_desc_with_name(node, state, sdfg, "output").shape
            map_ranges = {f"i{i}": f"0:{s}" for i, s in enumerate(shape)}
            index_str = f"{', '.join(map_ranges.keys())}"
            tasklet, _, _ = nstate.add_mapped_tasklet(
                node.label + "_tasklet",
                map_ranges=map_ranges,
                inputs={f"__input": dace.Memlet(f"input[{index_str}]")},
                code=f"__output = __input",
                outputs={"__output": dace.Memlet(f"output[{index_str}]")},
                external_edges=True)

            return nsdfg


# @op_implementation(op="Gemm", name="pure")
# class PureGemm(ONNXForward):
#     @staticmethod
#     def forward_can_be_applied(node: onnx_op.ONNXOp, state: SDFGState,
#                                sdfg: SDFG) -> bool:
#         if node.alpha == 1.0 and node.beta == 1.0 and node.transA == 0 and node.transB == 1:
#             return True
#         return False

#     @staticmethod
#     def forward(node: onnx_op.ONNXOp, state: SDFGState,
#                 sdfg: SDFG) -> typing.Union[Node, SDFG]:
#         assert node.alpha == 1.0 and node.beta == 1.0 and node.transA == 0 and node.transB == 1

#         # the gemm libnode is broken for now, so we just do it manually
#         if "C" in node.in_connectors:

#             def prog(A, B, C, Y):
#                 Y[:] = A @ np.transpose(B) + C
#         else:

#             def prog(A, B, Y):
#                 Y[:] = A @ np.transpose(B)

#         sdfg = program_for_node(prog, sdfg, state, node)
#         sdfg.apply_strict_transformations()
#         return sdfg


@op_implementation(op="Gemm", name="pure")
class PureGemm(ONNXForward):
    @staticmethod
    def forward_can_be_applied(node: onnx_op.ONNXOp, state: SDFGState,
                               sdfg: SDFG) -> bool:
        return True

    @staticmethod
    def forward(node: onnx_op.ONNXOp, state: SDFGState,
                sdfg: SDFG) -> typing.Union[Node, SDFG]:
        A_desc = in_desc_with_name(node, state, sdfg, "A")
        B_desc = in_desc_with_name(node, state, sdfg, "B")
        Y_desc = out_desc_with_name(node, state, sdfg, "Y")
        input0_dim = A_desc.shape
        input1_dim = B_desc.shape

        # list containing letters from z-a
        letters = [chr(ord('z') - i) for i in range(26)]
        # i j k are used for the last dimensions
        letters = [l for l in letters if l not in ['i', 'j', 'k']]

        if len(input0_dim) == 1:
            if len(input1_dim) != 2:
                raise ValueError("invalid dimensions")
            arg1 = 'k'
            arg2 = 'kj'
            result = 'j'
        elif len(input1_dim) == 1:
            if len(input0_dim) != 2:
                raise ValueError("invalid dimensions")
            arg1 = 'ik'
            arg2 = 'k'
            result = 'i'
        else:
            # build the einsum. The last two dimensions are always just the matrix multiply einsum
            # dace will later specialize to a batched matmul if possible
            arg1 = 'ik'
            arg2 = 'kj'
            result = 'ij'
            if input0_dim[-2] != input0_dim[-1]:
                if dace.symbolic.issymbolic(input0_dim[-2]):
                    log.warning(
                        f"overriding symbol {input0_dim[-2]} with value {input1_dim[-1]} in descriptor of input A of node {node}"
                    )
                    new_shape = list(A_desc.shape)
                    new_shape[-1] = input1_dim[-2]
                    A_desc.shape = new_shape
                elif dace.symbolic.issymbolic(input1_dim[-1]):
                    log.warning(
                        f"overriding symbol {input0_dim[-1]} with value {input0_dim[-2]} in descriptor of input B of node {node}"
                    )
                    new_shape = list(B_desc.shape)
                    new_shape[-2] = input0_dim[-1]
                    B_desc.shape = new_shape
            input0_dim = input0_dim[:-2]
            input1_dim = input1_dim[:-2]
            for dim0, dim1 in itertools.zip_longest(reversed(input0_dim),
                                                    reversed(input1_dim)):
                if dim0 is None:
                    # only dim0 exists
                    letter = letters.pop()
                    arg2 = letter + arg2
                    result = letter + result
                elif dim1 is None:
                    # only dim1 exists
                    letter = letters.pop()
                    arg1 = letter + arg1
                    result = letter + result
                else:
                    # both exist
                    letter = letters.pop()
                    arg1 = letter + arg1
                    arg2 = letter + arg2
                    result = letter + result

        if node.transA == 1:
            arg1 = ''.join(reversed(arg1))
        if node.transB == 1:
            arg2 = ''.join(reversed(arg2))

        einsum_str = '{},{}->{}'.format(arg1, arg2, result)

        # we lower to an ONNXEinsum node instead straight to the dace einsum to
        # make the autodiff simpler
        nsdfg = dace.SDFG(node.label + "_expansion")
        nstate = nsdfg.add_state()

        # Einsum: "A", "B" -> mm_result
        einsum_node: nodes.LibraryNode = onnx_op.ONNXEinsum(
            node.label + "_einsum_expansion", equation=einsum_str)

        nstate.add_node(einsum_node)
        einsum_node.add_in_connector("Inputs__0")
        einsum_node.add_in_connector("Inputs__1")
        nsdfg.add_datadesc("A", copy.deepcopy(A_desc))
        nsdfg.add_datadesc("B", copy.deepcopy(B_desc))
        nsdfg.add_datadesc("Y", copy.deepcopy(Y_desc))
        nsdfg.arrays["A"].transient = False
        nsdfg.arrays["B"].transient = False
        nsdfg.arrays["Y"].transient = False

        # Decide on array names based on alpha and beta
        uid = state.node_id(node)
        mm_result = "Y"
        if node.alpha != 1 or node.beta != 0:
            mm_result = f"Ytmp_{uid}"
        scal_result = mm_result
        if node.alpha != 1:
            scal_result = f"scaled_{uid}"

        # Create arrays according to alpha and beta
        if node.alpha != 1 or node.beta != 0:
            Ytmp_desc = out_desc_with_name(node, state, sdfg, "Y")
            nsdfg.add_datadesc(f"Ytmp_{uid}", copy.deepcopy(Ytmp_desc))
            nsdfg.arrays[f"Ytmp_{uid}"].transient = True
        if node.beta != 0:
            beta_desc = out_desc_with_name(node, state, sdfg, "Y")
            nsdfg.add_datadesc(f"scaled_{uid}", copy.deepcopy(beta_desc))
            nsdfg.arrays[f"scaled_{uid}"].transient = True

        nstate.add_edge(nstate.add_read("A"), None, einsum_node, "Inputs__0",
                        nsdfg.make_array_memlet("A"))
        nstate.add_edge(nstate.add_read("B"), None, einsum_node, "Inputs__1",
                        nsdfg.make_array_memlet("B"))
        mm_result_node = nstate.add_write(mm_result)
        nstate.add_edge(einsum_node, "Output", mm_result_node, None,
                        nsdfg.make_array_memlet(mm_result))

        # Multiply by alpha: mm_result -> scal_result
        if node.alpha != 1:
            nstate.add_mapped_tasklet(
                node.label + '_alphascale',
                {k: f'0:{Ytmp_desc.shape[i]}'
                 for i, k in enumerate(result)},
                dict(a=dace.Memlet(data=mm_result, subset=','.join(result))),
                f'o = a * dace.{Ytmp_desc.dtype}({node.alpha})',
                dict(o=dace.Memlet(data=scal_result, subset=','.join(result))),
                external_edges=True,
                input_nodes=dict(a=mm_result_node),
            )

        # Multiply by beta: scal_result, "C" -> "Y"
        if node.beta != 0:
            C_desc = in_desc_with_name(node, state, sdfg, "C")
            nsdfg.add_datadesc("C", copy.deepcopy(C_desc))
            nsdfg.arrays["C"].transient = False
            scal_result_node = next(n for n in nstate.sink_nodes()
                                    if isinstance(n, dace.nodes.AccessNode)
                                    and n.data == scal_result)
            beta_scale_code = f'o = s + c * dace.{C_desc.dtype}({node.beta})'
            if node.beta == 1:
                beta_scale_code = f'o = s + c'

            # Support broadcasting in C -> Y
            c_index = result[-len(C_desc.shape):]
            for c_shp, y_shp in zip(reversed(C_desc.shape),
                                    reversed(Y_desc.shape)):
                if c_shp != y_shp:
                    raise ValueError('Could not broadcast dimensions from C '
                                     'to Y in ONNXGemm')

            nstate.add_mapped_tasklet(
                node.label + '_betascale',
                {k: f'0:{Y_desc.shape[i]}'
                 for i, k in enumerate(result)},
                dict(s=dace.Memlet(data=scal_result, subset=','.join(result)),
                     c=dace.Memlet(data="C", subset=','.join(c_index))),
                beta_scale_code,
                dict(o=dace.Memlet(data="Y", subset=','.join(result))),
                external_edges=True,
                input_nodes={scal_result: scal_result_node},
            )

        return nsdfg


@op_implementation(op="Relu", name="pure")
class PureRelu(ONNXForward):
    @staticmethod
    def forward(node: onnx_op.ONNXOp, state: SDFGState,
                sdfg: SDFG) -> typing.Union[Node, SDFG]:
        input_dtype = in_desc_with_name(node, state, sdfg, "X").dtype
        cast_lambda = "lambda x: max(x, dace.{}(0))".format(
            input_dtype.to_string())

        def prog(X, Y):
            Y[:] = dace.elementwise(cast_lambda, X)

        return program_for_node(prog, sdfg, state, node)


@op_implementation(op="LeakyRelu", name="pure")
class PureLeakyRelu(ONNXForward):
    @staticmethod
    def forward(node: onnx_op.ONNXOp, state: SDFGState,
                sdfg: SDFG) -> typing.Union[Node, SDFG]:
        input_dtype = in_desc_with_name(node, state, sdfg, "X").dtype
        cast_lambda = "lambda x: (max(x, dace.{}(0)) + {} * min(x, dace.{}(0)))".format(
            input_dtype.to_string(), node.alpha, input_dtype.to_string())

        def prog(X, Y):
            Y[:] = dace.elementwise(cast_lambda, X)

        return program_for_node(prog, sdfg, state, node)


@op_implementation(op="Reshape", name="pure")
class PureReshape(ONNXForward):
    '''
        Reshape expansion: this relies on views
    '''
    @staticmethod
    def forward(node: onnx_op.ONNXOp, state: SDFGState,
                sdfg: SDFG) -> typing.Union[Node, SDFG]:

        input_name = "data"
        output_name = "reshaped"
        flatten = False

        # if called from Flatten
        if "input" in node._in_connectors.keys():
            input_name = "input"
            output_name = "output"
            flatten = True

        new_shape = out_desc_with_name(node, state, sdfg, output_name).shape

        if not flatten:
            node.remove_in_connector("shape")
            shape_node = in_edge_with_name(node, state, "shape").src
            constant_folding.remove_node_and_computation(
                sdfg, state, shape_node)

        if not flatten:

            def prog(data, reshaped):
                reshaped[:] = np.reshape(data, new_shape)
        else:

            def prog(input, output):
                output[:] = np.reshape(input, new_shape)

        return program_for_node(prog, sdfg, state, node)


@op_implementation(op="Flatten", name="pure")
class PureFlatten(ONNXForward):
    '''
        Flatten Expansion, reuses Reshape implementation
    '''
    @staticmethod
    def forward(node: onnx_op.ONNXOp, state: SDFGState,
                sdfg: SDFG) -> typing.Union[Node, SDFG]:

        # Reuse Reshape implementation
        return PureReshape.forward(node, state, sdfg)


@op_implementation(op="Sum", name="pure")
class PureSum(ONNXForward):
    @staticmethod
    def forward_can_be_applied(node: onnx_op.ONNXOp, state: SDFGState,
                               sdfg: SDFG) -> bool:
        # check that all shapes are arrays, and that the shapes are all equal
        shape = None
        for edge in node.iter_inputs_in_onnx_order(state):
            desc = in_desc_with_name(node, state, sdfg, edge.dst_conn)
            if shape is None:
                shape = desc.shape

            if not iterables_equal(shape, desc.shape):
                return False

        if not iterables_equal(
                shape,
                out_desc_with_name(node, state, sdfg, "sum").shape):
            return False

        return True

    @staticmethod
    def forward(node: onnx_op.ONNXOp, state: SDFGState,
                sdfg: SDFG) -> typing.Union[Node, SDFG]:

        nsdfg = dace.SDFG(node.name)
        input_names = []
        for e in node.iter_inputs_in_onnx_order(state):
            new_desc = copy.deepcopy(
                in_desc_with_name(node, state, sdfg, e.dst_conn))
            new_desc.transient = False
            nsdfg.add_datadesc(e.dst_conn, new_desc)
            input_names.append(e.dst_conn)

        new_desc = copy.deepcopy(out_desc_with_name(node, state, sdfg, "sum"))
        new_desc.transient = False
        nsdfg.add_datadesc("sum", new_desc)

        nstate = nsdfg.add_state()
        # we know all shapes are equal to the output shape
        shape = out_desc_with_name(node, state, sdfg, "sum").shape
        map_ranges = {f"i{i}": f"0:{s}" for i, s in enumerate(shape)}
        index_str = f"{', '.join(map_ranges.keys())}"
        tasklet, _, _ = nstate.add_mapped_tasklet(
            node.name + "_tasklet",
            map_ranges=map_ranges,
            inputs={
                f"__{inp}": dace.Memlet(f"{inp}[{index_str}]")
                for inp in input_names
            },
            code=f"__sum = {' + '.join(f'__{inp}' for inp in input_names)}",
            outputs={"__sum": dace.Memlet(f"sum[{index_str}]")},
            external_edges=True)

        tasklet.in_connectors = {
            f"__{inp}": in_desc_with_name(node, state, sdfg, inp).dtype
            for inp in input_names
        }
        tasklet.out_connectors = {
            "__sum": out_desc_with_name(node, state, sdfg, "sum").dtype
        }
        return nsdfg


@op_implementation(op="LogSoftmax", name="pure")
class PureLogSoftmax(ONNXForward):
    @staticmethod
    def forward(node: onnx_op.ONNXOp, state: SDFGState,
                sdfg: SDFG) -> typing.Union[nodes.Node, SDFG]:

        axis = node.axis

        reduced_shape = list(
            copy.deepcopy(in_desc_with_name(node, state, sdfg, "input").shape))
        reduced_shape[axis] = 1

        def prog(input, output):
            maximum = np.max(input, axis=axis)
            max_keepdims = np.reshape(maximum, reduced_shape)
            max_sub = input - max_keepdims
            exp_arr = np.exp(max_sub)
            sum = np.sum(exp_arr, axis=axis)
            sum_keepdims = np.reshape(sum, reduced_shape)
            log_sum = np.log(sum_keepdims)
            output[:] = max_sub - log_sum

        return program_for_node(prog, sdfg, state, node)


@op_implementation(op="Slice", name="pure")
class PureSlice(ONNXForward):
    '''
        Slice expansion
    '''
    @staticmethod
    def forward_can_be_applied(node: onnx_op.ONNXOp, state: SDFGState,
                               sdfg: SDFG) -> bool:
        # check that all the inputs (even the optional ones) are present and constant

        if not hasattr(sdfg, "_parent_onnx_model"):
            return False

        if in_edge_with_name(
                node, state, "starts"
        ).src.data not in sdfg._parent_onnx_model.clean_weights:
            return False
        if in_edge_with_name(
                node, state,
                "ends").src.data not in sdfg._parent_onnx_model.clean_weights:
            return False

        # optional inputs
        is_axes_present = True
        try:
            if in_edge_with_name(
                    node, state, "axes"
            ).src.data not in sdfg._parent_onnx_model.clean_weights:
                return False
        except ValueError:
            is_axes_present = False

        is_steps_present = True
        try:
            if in_edge_with_name(
                    node, state, "steps"
            ).src.data not in sdfg._parent_onnx_model.clean_weights:
                return False
        except ValueError:
            is_steps_present = False

        # Current constraints: axes and steps must be explict. Axes must be zero and steps must be 1
        if not is_axes_present or not is_steps_present:
            return False

        step = sdfg._parent_onnx_model.clean_weights[in_edge_with_name(
            node, state, "steps").src.data].numpy()[0]
        axis = sdfg._parent_onnx_model.clean_weights[in_edge_with_name(
            node, state, "axes").src.data].numpy()[0]

        if step != 1 or axis != 0:
            return False

        return True

    @staticmethod
    def forward(node: onnx_op.ONNXOp, state: SDFGState,
                sdfg: SDFG) -> typing.Union[Node, SDFG]:

        start = sdfg._parent_onnx_model.clean_weights[in_edge_with_name(
            node, state, "starts").src.data].numpy()[0]
        end = sdfg._parent_onnx_model.clean_weights[in_edge_with_name(
            node, state, "ends").src.data].numpy()[0]

        output_shape = out_desc_with_name(node, state, sdfg, "output").shape
        if end == np.iinfo(np.int64).max:
            # Pytorch exporter artifact
            end = start + output_shape[0]

        def prog(data, output):
            tmp = data[start:end:1, :]
            # We need reshape to avoid Invalid Edge errors
            output[:] = np.reshape(tmp, output.shape)

        return program_for_node(prog, sdfg, state, node)


@python_pure_op_implementation
def Softplus(X, Y):
    Y[:] = np.log(1 + np.exp(X))


@op_implementation(op="Sigmoid", name="pure")
class PureSigmoid(ONNXForward):
    @staticmethod
    def forward(node: onnx_op.ONNXOp, state: SDFGState,
                sdfg: SDFG) -> typing.Union[Node, SDFG]:
        dtype = in_desc_with_name(node, state, sdfg, "X").dtype

        def prog(X, Y):
            Y[:] = dace.elementwise(lambda x: dtype(1) / (dtype(1) + exp(-x)),
                                    X)

        return program_for_node(prog, sdfg, state, node)


@op_implementation(op="Transpose", name="einsum")
class EinsumTranspose(ONNXForward):
    @staticmethod
    def forward(node: onnx_op.ONNXOp, state: SDFGState,
                sdfg: SDFG) -> typing.Union[Node, SDFG]:
        perm = node.perm
        input_desc = in_desc_with_name(node, state, sdfg, "data")
        output_desc = out_desc_with_name(node, state, sdfg, "transposed")

        letters = [chr(ord('z') - i) for i in range(26)]
        input_letters = "".join(letters[i]
                                for i, _ in enumerate(input_desc.shape))
        output_letters = "".join(letters[i] for i in perm)
        equation_str = f"{input_letters}->{output_letters}"

        nsdfg = dace.SDFG(node.label + "_expansion")
        nstate = nsdfg.add_state()
        einsum_node: nodes.LibraryNode = onnx_op.ONNXEinsum(
            node.label + "_einsum_expansion", equation=equation_str)

        nstate.add_node(einsum_node)
        einsum_node.add_in_connector("Inputs__0")
        nsdfg.add_datadesc("data", copy.deepcopy(input_desc))
        nsdfg.add_datadesc("transposed", copy.deepcopy(output_desc))
        nsdfg.arrays["data"].transient = False
        nsdfg.arrays["transposed"].transient = False

        nstate.add_edge(nstate.add_read("data"), None, einsum_node,
                        "Inputs__0", nsdfg.make_array_memlet("data"))
        nstate.add_edge(einsum_node, "Output", nstate.add_write("transposed"),
                        None, nsdfg.make_array_memlet("transposed"))

        return nsdfg


@op_implementation(op="Split", name="pure")
class SplitPure(ONNXForward):
    @staticmethod
    def forward_can_be_applied(node: onnx_op.ONNXOp, state: SDFGState,
                               sdfg: SDFG) -> bool:
        return True

    @staticmethod
    def forward(node: onnx_op.ONNXOp, state: SDFGState,
                sdfg: SDFG) -> typing.Union[Node, SDFG]:

        nsdfg = dace.SDFG(node.label + "_expansion")
        nstate = nsdfg.add_state()

        split_dim = node.axis
        sizes = node.split
        idesc = in_desc_with_name(node, state, sdfg, "input")
        nsdfg.add_datadesc("input", copy.deepcopy(idesc))
        nsdfg.arrays["input"].transient = False

        rnode = nstate.add_read("input")

        offset = 0
        for i, odim in enumerate(sizes):
            # Set up new node shape and memlet
            new_shape = list(idesc.shape)
            new_shape[split_dim] = odim
            rng = subsets.Range([(0, s - 1, 1) if j != split_dim else
                                 (offset, offset + odim - 1, 1)
                                 for j, s in enumerate(new_shape)])
            offset += odim

            # Set up data descriptor
            oname = f"outputs__{i}"
            odesc = copy.deepcopy(out_desc_with_name(node, state, sdfg, oname))
            odesc.transient = False
            nsdfg.add_datadesc(oname, odesc)
            wnode = nstate.add_write(oname)

            # Perform copy (view)
            nstate.add_nedge(
                rnode, wnode,
                dace.Memlet(data="input",
                            subset=rng,
                            other_subset=subsets.Range.from_array(odesc)))

        return nsdfg


@op_implementation(op="Slice", name="pure")
class PureSliceAllConstant(ONNXForward):
    @staticmethod
    def _get_constant(conn: str, node: onnx_op.ONNXOp, state: SDFGState,
                      sdfg: SDFG):
        try:
            srcnode = next(state.in_edges_by_connector(node, conn)).src
        except StopIteration:
            return None
        # Scalar copied to GPU
        if 'gpu_' in srcnode.data:
            srcnode = state.predecessors(srcnode)[0]
        return onnx_constant_or_none(sdfg, srcnode)

    @staticmethod
    def forward_can_be_applied(node: onnx_op.ONNXOp, state: SDFGState,
                               sdfg: SDFG) -> bool:
        for inconn in ("axes", "ends", "starts", "steps"):
            if PureSliceAllConstant._get_constant(inconn, node, state,
                                                  sdfg) is None:
                return False
        return True

    @staticmethod
    def forward(node: onnx_op.ONNXOp, state: SDFGState,
                sdfg: SDFG) -> typing.Union[Node, SDFG]:
        axes = PureSliceAllConstant._get_constant('axes', node, state, sdfg)
        ends = PureSliceAllConstant._get_constant('ends', node, state, sdfg)
        starts = PureSliceAllConstant._get_constant('starts', node, state,
                                                    sdfg)
        steps = PureSliceAllConstant._get_constant('steps', node, state, sdfg)

        nsdfg = dace.SDFG(node.label + "_expansion")
        nstate = nsdfg.add_state()

        idesc = in_desc_with_name(node, state, sdfg, "data")
        odesc = out_desc_with_name(node, state, sdfg, "output")
        nsdfg.add_datadesc("data", copy.deepcopy(idesc))
        nsdfg.add_datadesc("output", copy.deepcopy(odesc))
        nsdfg.arrays["data"].transient = False
        nsdfg.arrays["output"].transient = False

        if not isinstance(axes, (tuple, list)):
            axes = [axes]
            ends = [ends]
            starts = [starts]
            steps = [steps]

        # Set up slicing memlet
        rng = [(0, s - 1, 1) for s in idesc.shape]
        for axis, start, end, step in zip(axes, starts, ends, steps):
            s = idesc.shape[axis]
            if end > s:
                end = s
            rng[axis] = (start, end - 1, step)

        sbs = subsets.Range(rng)
        osbs = subsets.Range.from_array(odesc)

        # Make copy / view
        rnode = nstate.add_read("data")
        wnode = nstate.add_write("output")

        nstate.add_nedge(
            rnode, wnode,
            dace.Memlet(data="data", subset=sbs, other_subset=osbs))

        return nsdfg
