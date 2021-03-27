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


@autoregister_params(op="LogSoftmax", name="pure")
class PureLogSoftmax(ONNXForward):
    @staticmethod
    def forward(node: onnx_op.ONNXOp, state: SDFGState,
                sdfg: SDFG) -> typing.Union[Node, SDFG]:

        # NOTE: once there is a reshape node this whole expansion becomes much simpler:
        #
        # exp = np.exp(X - np.max(X, axis=axis, keepdims=True))
        # sum = np.sum(exp, axis=axis, keepdims=True)

        # result = exp / sum

        node.validate(sdfg, state)
        inparr = in_desc_with_name(node, state, sdfg, "input")

        axis = node.axis
        if type(axis) is not int or not (-len(inparr.shape) <= axis < len(
                inparr.shape)):
            raise ValueError("expected axis to be an integer in range"
                             " [-{}, {}), got {}".format(
                                 len(inparr.shape), len(inparr.shape), axis))

        if axis < 0:
            axis += len(inparr.shape)
        out_tmp_shape = inparr.shape
        out_tmp_dtype = inparr.dtype

        tmp_max_shape = list(copy.deepcopy(inparr.shape))
        tmp_max_shape.pop(axis)

        ##################
        # exp (X - max)
        exp_minus_max = dace.SDFG("exp_minus_max")
        exp_minus_max.add_array("exp_tmp_max", tmp_max_shape, inparr.dtype)
        exp_minus_max.add_array("exp_input", inparr.shape, inparr.dtype)
        exp_minus_max.add_array("exp_output", out_tmp_shape, out_tmp_dtype)
        exp_minus_max.add_state().add_mapped_tasklet(
            "_softmax_exp_",
            map_ranges={
                "__i" + str(i): "0:" + str(shape)
                for i, shape in enumerate(inparr.shape)
            },
            inputs={
                '__max':
                dace.Memlet.simple(
                    "exp_tmp_max", ','.join("__i" + str(i)
                                            for i in range(len(inparr.shape))
                                            if i != axis)),
                '__x':
                dace.Memlet.simple(
                    "exp_input",
                    ','.join("__i" + str(i) for i in range(len(inparr.shape))))
            },
            code='__out = exp(__x - __max)',
            outputs={
                '__out':
                dace.Memlet.simple(
                    "exp_output",
                    ','.join("__i" + str(i) for i in range(len(inparr.shape))))
            },
            external_edges=True)

        ##################
        # out_tmp / sum
        out_tmp_div_sum = dace.SDFG("out_tmp_div_sum")
        out_tmp_div_sum.add_array("div_tmp", inparr.shape, inparr.dtype)
        out_tmp_div_sum.add_array("div_sum", tmp_max_shape, inparr.dtype)
        out_tmp_div_sum.add_array("div_X", inparr.shape, inparr.dtype)
        out_tmp_div_sum.add_array("div_max", tmp_max_shape, inparr.dtype)
        out_tmp_div_sum.add_array("div_output", out_tmp_shape, out_tmp_dtype)

        out_tmp_div_sum.add_state().add_mapped_tasklet(
            "_softmax_div_",
            map_ranges={
                "__i" + str(i): "0:" + str(shape)
                for i, shape in enumerate(inparr.shape)
            },
            inputs={
                '__sum':
                dace.Memlet.simple(
                    "div_sum", ','.join("__i" + str(i)
                                        for i in range(len(inparr.shape))
                                        if i != axis)),
                '__max':
                dace.Memlet.simple(
                    "div_max", ','.join("__i" + str(i)
                                        for i in range(len(inparr.shape))
                                        if i != axis)),
                '__x':
                dace.Memlet.simple(
                    "div_X",
                    ','.join("__i" + str(i) for i in range(len(inparr.shape))))
            },
            code='__out = __x - __max - log(__sum)',
            outputs={
                '__out':
                dace.Memlet.simple(
                    "div_output",
                    ','.join("__i" + str(i) for i in range(len(inparr.shape))))
            },
            external_edges=True)

        ##################
        # put everything together as a program
        def prog(input, output):
            tmp_max = np.max(input, axis=axis)

            # this holds exp (X - max)
            out_tmp = dace.define_local(out_tmp_shape, out_tmp_dtype)
            exp_minus_max(exp_tmp_max=tmp_max,
                          exp_input=input,
                          exp_output=out_tmp)

            tmp_sum = np.sum(out_tmp, axis=axis)

            # this holds exp (X - max)
            out_tmp_div_sum(div_X=input,
                            div_max=tmp_max,
                            div_tmp=out_tmp,
                            div_sum=tmp_sum,
                            div_output=output)

        return program_for_node(prog, sdfg, state, node).to_sdfg()
