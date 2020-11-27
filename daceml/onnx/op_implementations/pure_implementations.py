import copy
import inspect
import itertools
import logging
import typing

import dace
from dace import SDFGState, SDFG, dtypes, data as dt, nodes
from dace.frontend.common import create_einsum_sdfg
from dace.frontend.python.parser import DaceProgram
from dace.registry import autoregister_params
import dace.libraries.blas as blas
from dace.sdfg.nodes import Node
from dace.sdfg import nodes, propagation
from dace.sdfg.nodes import Node
from dace.symbolic import symstr

from daceml.transformation import constant_folding
from daceml.onnx.nodes import onnx_op
from daceml.onnx.nodes.node_utils import parse_variadic_param
from daceml.onnx import converters
from daceml.onnx.forward_implementation_abc import ONNXForward
import numpy as np

from daceml.util.utils import in_desc_with_name, out_desc_with_name, in_edge_with_name

log = logging.getLogger(__name__)


def program_for_node(program, sdfg: SDFG, state: SDFGState,
                     node: onnx_op.ONNXOp) -> SDFG:
    """ Expand a function to a dace program.

        The dtypes for the arguments will be extracted by matching the parameter names to edges.
    """
    input_names = node.schema.non_variadic_inputs()
    variadic_input_names = node.schema.variadic_inputs()

    output_names = node.schema.non_variadic_outputs()
    variadic_output_names = node.schema.variadic_outputs()

    if set(input_names).intersection(output_names):
        # this is currently the case for only one onnx op
        raise ValueError(
            "program_for_node cannot be applied on nodes of this type;"
            " '{}' is both an input and an output".format(
                next(input_names.intersection(output_names))))

    params = inspect.signature(program).parameters

    annotations = {}
    for name, param in params.items():
        if name in input_names or ("__" in name
                                   and parse_variadic_param(name)[0]
                                   in variadic_input_names):
            annotations[name] = in_desc_with_name(node, state, sdfg, name)
        elif name in output_names or ("__" in name
                                      and parse_variadic_param(name)[0]
                                      in variadic_output_names):
            annotations[name] = out_desc_with_name(node, state, sdfg, name)
        else:
            raise ValueError(
                "'{}' was not found as an input or output for {}".format(
                    name, node.schema.name))

    program.__annotations__ = annotations

    result = DaceProgram(program, (), {}, False, dace.DeviceType.CPU)
    result.name = node.label + "_expansion"

    sdfg = result.to_sdfg()

    if node.schedule in [dtypes.ScheduleType.GPU_Default
                         ] + dtypes.GPU_SCHEDULES:
        sdfg.apply_gpu_transformations()

    return sdfg


@autoregister_params(op="Log", name="pure")
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


@autoregister_params(op="Sqrt", name="pure")
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

        node.validate(sdfg, state)

        def prog(X, Y):
            Y[:] = dace.elementwise(lambda x: sqrt(x), X)

        return program_for_node(prog, sdfg, state, node)


@autoregister_params(op="Pow", name="pure")
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

        node.validate(sdfg, state)

        def prog(X, Y, Z):
            Z[:] = X**Y

        return program_for_node(prog, sdfg, state, node)


@autoregister_params(op="Add", name="pure")
class PureAdd(ONNXForward):
    @staticmethod
    def forward(node: onnx_op.ONNXOp, state: SDFGState,
                sdfg: SDFG) -> typing.Union[Node, SDFG]:

        node.validate(sdfg, state)

        def prog(A, B, C):
            C[:] = A + B

        return program_for_node(prog, sdfg, state, node)


@autoregister_params(op="Sub", name="pure")
class PureSub(ONNXForward):
    @staticmethod
    def forward(node: onnx_op.ONNXOp, state: SDFGState,
                sdfg: SDFG) -> typing.Union[Node, SDFG]:

        node.validate(sdfg, state)

        def prog(A, B, C):
            C[:] = A - B

        return program_for_node(prog, sdfg, state, node)


@autoregister_params(op="Mul", name="pure")
class PureMul(ONNXForward):
    @staticmethod
    def forward(node: onnx_op.ONNXOp, state: SDFGState,
                sdfg: SDFG) -> typing.Union[Node, SDFG]:

        node.validate(sdfg, state)

        def prog(A, B, C):
            C[:] = A * B

        return program_for_node(prog, sdfg, state, node)


@autoregister_params(op="Div", name="pure")
class PureDiv(ONNXForward):
    @staticmethod
    def forward(node: onnx_op.ONNXOp, state: SDFGState,
                sdfg: SDFG) -> typing.Union[Node, SDFG]:

        node.validate(sdfg, state)

        def prog(A, B, C):
            C[:] = A / B

        return program_for_node(prog, sdfg, state, node)


@autoregister_params(op="ReduceMean", name="pure")
class PureReduceMean(ONNXForward):
    @staticmethod
    def forward(node: onnx_op.ONNXOp, state: SDFGState,
                sdfg: SDFG) -> typing.Union[Node, SDFG]:

        node.validate(sdfg, state)

        axes = node.axes

        # when keepdims is true, this works but there is a useless copy. We just leave this for now; this can be fixed
        # with a reshape node when those exist.
        def prog(data, reduced):
            reduced[:] = np.mean(data, axis=axes)

        return program_for_node(prog, sdfg, state, node)


@autoregister_params(op="Erf", name="pure")
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

        node.validate(sdfg, state)

        def prog(input, output):
            output[:] = dace.elementwise(lambda x: erf(x), input)

        return program_for_node(prog, sdfg, state, node)


@autoregister_params(op="MatMul", name="pure")
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
        node.validate(sdfg, state)

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


@autoregister_params(op="Einsum", name="pure")
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
        node.validate(sdfg, state)

        nsdfg = dace.SDFG(node.label + "_expansion")
        nstate = nsdfg.add_state()

        for e in node.iter_inputs_in_onnx_order(state):
            nsdfg.add_datadesc(
                e.dst_conn, in_desc_with_name(node, state, sdfg, e.dst_conn))
        for e in node.iter_outputs_in_onnx_order(state):
            nsdfg.add_datadesc(
                e.src_conn, out_desc_with_name(node, state, sdfg, e.src_conn))

        create_einsum_sdfg(None,
                           nsdfg,
                           nstate,
                           node.equation.replace(" ", ""),
                           *(e.dst_conn
                             for e in node.iter_inputs_in_onnx_order(state)),
                           output="Output")
        return nsdfg


@autoregister_params(op="Identity", name="pure")
class PureIdentity(ONNXForward):
    @staticmethod
    def forward_can_be_applied(node: onnx_op.ONNXOp, state: SDFGState,
                               sdfg: SDFG) -> bool:
        return True

    @staticmethod
    def forward(node: onnx_op.ONNXOp, state: SDFGState,
                sdfg: SDFG) -> typing.Union[Node, SDFG]:

        node.validate(sdfg, state)

        def prog(input, output):
            output[:] = input

        return program_for_node(prog, sdfg, state, node)


@autoregister_params(op="Reciprocal", name="pure")
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

        node.validate(sdfg, state)

        dtype = in_desc_with_name(node, state, sdfg, 'X').dtype
        tanh_lambda = "lambda x: dace.{}(1) / x".format(dtype.to_string())

        def prog(X, Y):
            Y[:] = dace.elementwise(tanh_lambda, X)

        return program_for_node(prog, sdfg, state, node)


@autoregister_params(op="Tanh", name="pure")
class PureTanh(ONNXForward):
    @staticmethod
    def forward(node: onnx_op.ONNXOp, state: SDFGState,
                sdfg: SDFG) -> typing.Union[Node, SDFG]:

        node.validate(sdfg, state)

        def prog(input, output):
            output[:] = dace.elementwise(lambda x: tanh(x), input)

        return program_for_node(prog, sdfg, state, node)


@autoregister_params(op="ReduceSum", name="pure")
class PureReduceSum(ONNXForward):
    @staticmethod
    def forward(node: onnx_op.ONNXOp, state: SDFGState,
                sdfg: SDFG) -> typing.Union[Node, SDFG]:
        node.validate(sdfg, state)

        axes = node.axes

        # when keepdims is true, this works but there is a useless copy. We just leave this for now; this can be fixed
        # with a reshape node when those exist.
        def prog(data, reduced):
            reduced[:] = np.sum(data, axis=axes)

        return program_for_node(prog, sdfg, state, node)


@autoregister_params(op="ReduceMax", name="pure")
class PureReduceMax(ONNXForward):
    @staticmethod
    def forward(node: onnx_op.ONNXOp, state: SDFGState,
                sdfg: SDFG) -> typing.Union[Node, SDFG]:
        node.validate(sdfg, state)

        axes = node.axes

        # when keepdims is true, this works but there is a useless copy. We just leave this for now; this can be fixed
        # with a reshape node when those exist.
        def prog(data, reduced):
            reduced[:] = np.max(data, axis=axes)

        return program_for_node(prog, sdfg, state, node)


@autoregister_params(op="ReduceMin", name="pure")
class PureReduceMin(ONNXForward):
    @staticmethod
    def forward(node: onnx_op.ONNXOp, state: SDFGState,
                sdfg: SDFG) -> typing.Union[Node, SDFG]:
        node.validate(sdfg, state)

        axes = node.axes

        # when keepdims is true, this works but there is a useless copy. We just leave this for now; this can be fixed
        # with a reshape node when those exist.
        def prog(data, reduced):
            reduced[:] = np.min(data, axis=axes)

        return program_for_node(prog, sdfg, state, node)


@autoregister_params(op="Softmax", name="pure")
class PureSoftmax(ONNXForward):
    @staticmethod
    def forward(node: onnx_op.ONNXOp, state: SDFGState,
                sdfg: SDFG) -> typing.Union[Node, SDFG]:

        axis = node.axis

        reduced_shape = list(
            copy.deepcopy(in_desc_with_name(node, state, sdfg, "input").shape))
        reduced_shape[axis] = 1

        def prog(input, output):
            max = np.max(input, axis=axis)
            max_keepdims = np.reshape(max, reduced_shape)
            exp_arr = np.exp(input - max_keepdims)
            sum = np.sum(exp_arr, axis=axis)
            sum_keepdims = np.reshape(sum, reduced_shape)
            output[:] = exp_arr / sum_keepdims

        return program_for_node(prog, sdfg, state, node)


@autoregister_params(op="Transpose", name="pure")
class PureTranspose(ONNXForward):
    @staticmethod
    def forward(node: onnx_op.ONNXOp, state: SDFGState,
                sdfg: SDFG) -> typing.Union[Node, SDFG]:

        node.validate(sdfg, state)
        perm = node.perm

        def prog(data, transposed):
            transposed[:] = np.transpose(data, axes=perm)

        return program_for_node(prog, sdfg, state, node)


@autoregister_params(op="Cast", name="pure")
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
        else:

            def prog(input, output):
                output[:] = dace.elementwise(lambda x: x, input)

        return program_for_node(prog, sdfg, state, node)


@autoregister_params(op="Gemm", name="pure")
class PureGemm(ONNXForward):
    @staticmethod
    def forward_can_be_applied(node: onnx_op.ONNXOp, state: SDFGState,
                               sdfg: SDFG) -> bool:
        if node.alpha == 1.0 and node.beta == 1.0 and node.transA == 0 and node.transB == 1:
            return True
        return False

    @staticmethod
    def forward(node: onnx_op.ONNXOp, state: SDFGState,
                sdfg: SDFG) -> typing.Union[Node, SDFG]:
        node.validate(sdfg, state)

        assert node.alpha == 1.0 and node.beta == 1.0 and node.transA == 0 and node.transB == 1

        # the gemm libnode is broken for now, so we just do it manually
        if "C" in node.in_connectors:

            def prog(A, B, C, Y):
                Y[:] = A @ np.transpose(B) + C
        else:

            def prog(A, B, Y):
                Y[:] = A @ np.transpose(B)

        sdfg = program_for_node(prog, sdfg, state, node)
        sdfg.apply_strict_transformations()
        return sdfg


@autoregister_params(op="Relu", name="pure")
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


@autoregister_params(op="Reshape", name="pure")
class PureReshape(ONNXForward):
    @staticmethod
    def forward(node: onnx_op.ONNXOp, state: SDFGState,
                sdfg: SDFG) -> typing.Union[Node, SDFG]:
        new_shape = out_desc_with_name(node, state, sdfg, "reshaped").shape
        node.remove_in_connector("shape")

        shape_node = in_edge_with_name(node, state, "shape").src
        constant_folding.remove_node_and_computation(sdfg, state, shape_node)

        def prog(data, reshaped):
            reshaped[:] = np.reshape(data, new_shape)

        return program_for_node(prog, sdfg, state, node)


def _2d_sliding_window_index_expr(x_or_y, stride, kernel_size):
    index_expression = "out_{x_or_y} * {stride} + h{x_or_y}"
    return index_expression.format(x_or_y=x_or_y, stride=stride)


@autoregister_params(op="MaxPool", name="pure")
class PureMaxPool2D(ONNXForward):
    @staticmethod
    def forward_can_be_applied(node: ONNXOp, state: SDFGState,
                               sdfg: SDFG) -> bool:
        X = in_desc_with_name(node, state, sdfg, "X")

        if "Indices" in {e.src_conn for e in state.out_edges(node)}:
            return False

        image_dims = len(X.shape) - 2

        # only do 2D for now
        if image_dims != 2:
            return False

        if node.pads is not None and (not all(p == 0 for p in node.pads)
                                      or len(node.pads) != image_dims * 2):
            return False

        if node.strides is not None and len(node.strides) != image_dims:
            return False

        if node.auto_pad != 'NOTSET':
            return False

        if node.ceil_mode != 0 or node.storage_order != 0:
            return False

        if node.dilations is not None and (not all(d == 1
                                                   for d in node.dilations) or
                                           len(node.dilations) != image_dims):
            return False
        return True

    @staticmethod
    def forward(node: ONNXOp, state: SDFGState,
                sdfg: SDFG) -> typing.Union[nodes.Node, SDFG]:
        X = in_desc_with_name(node, state, sdfg, "X")
        Y = out_desc_with_name(node, state, sdfg, "Y")

        image_dims = len(X.shape) - 2
        batch_size = X.shape[0]
        num_channels = X.shape[1]
        strides = node.strides if node.strides is not None else [
            1 for _ in range(image_dims)
        ]
        stride_x, stride_y = strides
        filter_hx, filter_hy = node.kernel_shape
        output_size_y, output_size_x = Y.shape[2:]

        new_sdfg = dace.SDFG("pure_maxpool")

        init_state = new_sdfg.add_state("init")

        new_state = new_sdfg.add_state_after(init_state, "compute")
        new_sdfg.add_datadesc("X", copy.deepcopy(X))
        new_sdfg.add_datadesc("Y", copy.deepcopy(Y))

        new_sdfg.arrays["X"].transient = False
        new_sdfg.arrays["Y"].transient = False

        # add init state
        # yapf: disable
        init_state.add_mapped_tasklet("init",
                                      map_ranges={
                                          "i{}".format(i): "0:{}".format(s)
                                          for i, s in enumerate(Y.shape)
                                      },
                                      inputs={},
                                      code="y = {}".format(dtypes.min_value(Y.dtype)),
                                      outputs=dict(
                                          y=dace.Memlet("Y[{}]".format(
                                              ", ".join("i{}".format(i)
                                                        for i, _ in enumerate(Y.shape))))
                                      ),
                                      external_edges=True)
        # yapf: enable

        # the outer map loops over every entry in the output array
        outer_me, outer_mx = new_state.add_map(
            'outer_conv_map',
            dict(b="0:{}".format(batch_size),
                 c="0:{}".format(num_channels),
                 out_x="0:{}".format(output_size_x),
                 out_y="0:{}".format(output_size_y)))

        # the inner map computes the value for a single entry in the output array (i.e. Y[b, c, x, y])
        inner_me, inner_mx = new_state.add_map(
            'inner_conv_map',
            dict(hx="0:{}".format(filter_hx), hy="0:{}".format(filter_hy)))

        compute_tasklet = new_state.add_tasklet("compute_entry",
                                                inputs={"image_in"},
                                                outputs={"output"},
                                                code="output = image_in")

        x_idx = _2d_sliding_window_index_expr(x_or_y="x",
                                              stride=stride_x,
                                              kernel_size=filter_hx)
        y_idx = _2d_sliding_window_index_expr(x_or_y="y",
                                              stride=stride_y,
                                              kernel_size=filter_hy)

        image_memlet = dace.Memlet("X[b, c, {}, {}]".format(x_idx, y_idx))

        new_state.add_edge(inner_me, None, compute_tasklet, "image_in",
                           image_memlet)

        # hook up X
        read_X = new_state.add_read("X")
        inner_image_memlet = propagation.propagate_memlet(
            new_state, image_memlet, inner_me, False)
        outer_image_memlet = propagation.propagate_memlet(
            new_state, inner_image_memlet, outer_me, False)
        new_state.add_edge(outer_me, None, inner_me, None, inner_image_memlet)
        new_state.add_edge(read_X, None, outer_me, None, outer_image_memlet)

        # hook up outputs
        output_memlet = dace.Memlet("Y[b, c, out_x, out_y]",
                                    wcr="lambda x, y: max(x, y)")
        inner_output_memlet = propagation.propagate_memlet(
            new_state, output_memlet, inner_me, False)
        outer_output_memlet = propagation.propagate_memlet(
            new_state, inner_output_memlet, outer_me, False)
        new_state.add_edge(compute_tasklet, "output", inner_mx, None,
                           output_memlet)

        write_Y = new_state.add_write("Y")
        new_state.add_edge_pair(outer_mx, inner_mx, write_Y,
                                inner_output_memlet, outer_output_memlet)

        new_sdfg.fill_scope_connectors()
        return new_sdfg


@autoregister_params(op="Conv", name="pure")
class PureConv2D(ONNXForward):
    """
    The "trivial" convolution implementation, i.e. two nested maps.
    """
    @staticmethod
    def forward_can_be_applied(node: ONNXOp, state: SDFGState,
                               sdfg: SDFG) -> bool:
        X = in_desc_with_name(node, state, sdfg, "X")
        W = in_desc_with_name(node, state, sdfg, "W")
        try:
            B = in_desc_with_name(node, state, sdfg, "B")
        except Exception as e:
            B = None

        image_dims = len(X.shape) - 2
        num_filters = W.shape[0]
        num_channels = X.shape[1]

        if (X.dtype not in [dace.float16, dace.float32, dace.float64]
                or W.dtype not in [dace.float16, dace.float32, dace.float64]):
            return False

        # only do 2D for now
        if len(X.shape) != 4 or len(W.shape) != 4:
            return False

        if node.group != 1:
            return False

        if num_channels != W.shape[1]:
            return False

        if node.dilations is not None and (not all(d == 1
                                                   for d in node.dilations) or
                                           len(node.dilations) != image_dims):
            return False

        if node.pads is not None and (not all(p == 0 for p in node.pads)
                                      or len(node.pads) != image_dims * 2):
            return False

        if node.strides is not None and len(node.strides) != image_dims:
            return False

        if B is not None and B.shape[0] != num_filters:
            return False

        if node.auto_pad != 'NOTSET':
            return False

        return True

    @staticmethod
    def forward(node: ONNXOp, state: SDFGState,
                sdfg: SDFG) -> typing.Union[nodes.Node, SDFG]:
        X = in_desc_with_name(node, state, sdfg, "X")
        W = in_desc_with_name(node, state, sdfg, "W")
        Y = out_desc_with_name(node, state, sdfg, "Y")
        try:
            B = in_desc_with_name(node, state, sdfg, "B")
        except Exception as e:
            B = None

        image_dims = len(X.shape) - 2
        strides = node.strides if node.strides is not None else [
            1 for _ in range(image_dims)
        ]
        stride_x, stride_y = strides

        if node.kernel_shape is not None:
            filter_hx, filter_hy = node.kernel_shape
        else:
            filter_hx, filter_hy = W.shape[2:]

        num_filters = W.shape[0]
        num_channels = X.shape[1]
        batch_size = X.shape[0]

        output_size_y, output_size_x = Y.shape[2:]

        new_sdfg = dace.SDFG("pure_conv")

        init_state = new_sdfg.add_state("init")
        new_state = new_sdfg.add_state_after(init_state, "compute")
        new_sdfg.add_datadesc("X", copy.deepcopy(X))
        new_sdfg.add_datadesc("W", copy.deepcopy(W))
        new_sdfg.add_datadesc("Y", copy.deepcopy(Y))
        if B is not None:
            new_sdfg.add_datadesc("B", copy.deepcopy(B))
            new_sdfg.arrays["B"].transient = False

        new_sdfg.arrays["X"].transient = False
        new_sdfg.arrays["W"].transient = False
        new_sdfg.arrays["Y"].transient = False

        # add init state
        # yapf: disable
        init_state.add_mapped_tasklet("init",
                                      map_ranges={
                                          "i{}".format(i): "0:{}".format(i, s)
                                          for i, s in enumerate(Y.shape)
                                      },
                                      inputs={},
                                      code="y = 0",
                                      outputs=dict(
                                          y=dace.Memlet("Y[{}]".format(
                                              ", ".join("i{}".format(i)
                                                        for i, _ in enumerate(Y.shape))))
                                      ),
                                      external_edges=True)
        # yapf: enable

        # the outer map loops over every entry in the output array
        outer_me, outer_mx = new_state.add_map(
            'outer_conv_map',
            dict(b="0:{}".format(batch_size),
                 m="0:{}".format(num_filters),
                 out_x="0:{}".format(output_size_x),
                 out_y="0:{}".format(output_size_y)))

        # the inner map computes the value for a single entry in the output array (i.e. Y[b, m, x, y])
        inner_me, inner_mx = new_state.add_map(
            'inner_conv_map',
            dict(cin="0:{}".format(num_channels),
                 hx="0:{}".format(filter_hx),
                 hy="0:{}".format(filter_hy)))

        compute_tasklet = new_state.add_tasklet(
            "compute_entry",
            inputs={"image_in", "filter_in"},
            outputs={"output"},
            code="output = image_in * filter_in")

        filter_memlet = dace.Memlet("W[m, cin, hx, hy]")

        x_idx = _2d_sliding_window_index_expr(x_or_y="x",
                                              stride=stride_x,
                                              kernel_size=filter_hx)
        y_idx = _2d_sliding_window_index_expr(x_or_y="y",
                                              stride=stride_y,
                                              kernel_size=filter_hy)

        image_memlet = dace.Memlet("X[b, cin, {}, {}]".format(x_idx, y_idx))

        # hook up the inner map to the tasklet
        new_state.add_edge(inner_me, None, compute_tasklet, "filter_in",
                           filter_memlet)
        new_state.add_edge(inner_me, None, compute_tasklet, "image_in",
                           image_memlet)

        # hook up filter
        read_W = new_state.add_read("W")
        inner_filter_memlet = propagation.propagate_memlet(
            new_state, filter_memlet, inner_me, False)
        outer_filter_memlet = propagation.propagate_memlet(
            new_state, inner_filter_memlet, outer_me, False)
        new_state.add_edge(outer_me, None, inner_me, None, inner_filter_memlet)
        new_state.add_edge(read_W, None, outer_me, None, outer_filter_memlet)

        # hook up X
        read_X = new_state.add_read("X")
        inner_image_memlet = propagation.propagate_memlet(
            new_state, image_memlet, inner_me, False)
        outer_image_memlet = propagation.propagate_memlet(
            new_state, inner_image_memlet, outer_me, False)
        new_state.add_edge(outer_me, None, inner_me, None, inner_image_memlet)
        new_state.add_edge(read_X, None, outer_me, None, outer_image_memlet)

        # hook up outputs
        output_memlet = dace.Memlet("Y[b, m, out_x, out_y]",
                                    wcr="lambda x, y: x + y")
        inner_output_memlet = propagation.propagate_memlet(
            new_state, output_memlet, inner_me, False)
        outer_output_memlet = propagation.propagate_memlet(
            new_state, inner_output_memlet, outer_me, False)
        new_state.add_edge(compute_tasklet, "output", inner_mx, None,
                           output_memlet)

        write_Y = new_state.add_write("Y")
        new_state.add_edge_pair(outer_mx, inner_mx, write_Y,
                                inner_output_memlet, outer_output_memlet)

        # hook up B if required
        if B is not None:
            read_B = new_state.add_read("B")
            B_memlet = dace.Memlet("B[m]")
            new_state.add_edge(
                read_B, None, outer_me, None,
                propagation.propagate_memlet(new_state, B_memlet, outer_me,
                                             False))

            add_bias_tasklet = new_state.add_tasklet("add_bias", {"bias_in"},
                                                     {"output"},
                                                     "output = bias_in")
            new_state.add_edge(outer_me, None, add_bias_tasklet, "bias_in",
                               B_memlet)
            new_state.add_edge_pair(outer_mx,
                                    add_bias_tasklet,
                                    write_Y,
                                    output_memlet,
                                    outer_output_memlet,
                                    internal_connector="output")

        new_sdfg.fill_scope_connectors()

        return new_sdfg
