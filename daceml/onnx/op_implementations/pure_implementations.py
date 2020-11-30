import copy
import inspect
import typing

import dace
from dace import SDFGState, SDFG, dtypes
from dace.frontend.python.parser import DaceProgram
from dace.registry import autoregister_params
from dace.sdfg import nodes, propagation
from dace.sdfg.nodes import Node
from dace.symbolic import symstr

from daceml.onnx.nodes.onnx_op import ONNXOp
from daceml.onnx import converters
from daceml.onnx.implementation_abc import ONNXForward
import numpy as np

from daceml.util.utils import in_desc_with_name, out_desc_with_name


def program_for_node(program, sdfg: SDFG, state: SDFGState,
                     node: ONNXOp) -> DaceProgram:
    """ Expand a function to a dace program.

        The dtypes for the arguments will be extracted by matching the parameter names to edges.
    """
    input_names = set(inp.name for inp in node.schema.inputs)
    output_names = set(outp.name for outp in node.schema.outputs)

    if input_names.intersection(output_names):
        # this is currently the case for only one onnx op
        raise ValueError(
            "program_for_node cannot be applied on nodes of this type;"
            " '{}' is both an input and an output".format(
                next(input_names.intersection(output_names))))

    params = inspect.signature(program).parameters

    annotations = {}
    for name, param in params.items():
        if name in input_names:
            annotations[name] = in_desc_with_name(node, state, sdfg, name)
        elif name in output_names:
            annotations[name] = out_desc_with_name(node, state, sdfg, name)
        else:
            raise ValueError(
                "'{}' was not found as an input or output for {}".format(
                    name, node.schema.name))

    program.__annotations__ = annotations

    result = DaceProgram(program, (), {})

    return result


@autoregister_params(op="Sqrt", name="pure")
class PureSqrt(ONNXForward):
    @staticmethod
    def forward_can_be_applied(node: ONNXOp, state: SDFGState,
                               sdfg: SDFG) -> bool:
        return in_desc_with_name(node, state, sdfg, 'X').dtype in [
            dace.float16, dace.float32, dace.float64
        ]

    @staticmethod
    def forward(node: ONNXOp, state: SDFGState,
                sdfg: SDFG) -> typing.Union[nodes.Node, SDFG]:

        node.validate(sdfg, state)

        def prog(X, Y):
            Y[:] = dace.elementwise(lambda x: sqrt(x), X)

        return program_for_node(prog, sdfg, state, node).to_sdfg()


@autoregister_params(op="Pow", name="pure")
class PurePow(ONNXForward):
    @staticmethod
    def forward_can_be_applied(node: ONNXOp, state: SDFGState,
                               sdfg: SDFG) -> bool:
        if node.schedule is dtypes.ScheduleType.GPU_Default:
            # TODO fix this in a follow up PR (this returns NaN in the PT bert encoder test; check
            # how ORT implements Pow for cuda...) Issue #21
            return False

        return in_desc_with_name(node, state, sdfg, 'X').dtype in [
            dace.float16, dace.float32, dace.float64
        ]

    @staticmethod
    def forward(node: ONNXOp, state: SDFGState,
                sdfg: SDFG) -> typing.Union[nodes.Node, SDFG]:

        node.validate(sdfg, state)

        def prog(X, Y, Z):
            Z[:] = X**Y

        return program_for_node(prog, sdfg, state, node).to_sdfg()


@autoregister_params(op="Add", name="pure")
class PureAdd(ONNXForward):
    @staticmethod
    def forward(node: ONNXOp, state: SDFGState,
                sdfg: SDFG) -> typing.Union[nodes.Node, SDFG]:

        node.validate(sdfg, state)

        def prog(A, B, C):
            C[:] = A + B

        return program_for_node(prog, sdfg, state, node).to_sdfg()


@autoregister_params(op="Sub", name="pure")
class PureSub(ONNXForward):
    @staticmethod
    def forward(node: ONNXOp, state: SDFGState,
                sdfg: SDFG) -> typing.Union[nodes.Node, SDFG]:

        node.validate(sdfg, state)

        def prog(A, B, C):
            C[:] = A - B

        return program_for_node(prog, sdfg, state, node).to_sdfg()


@autoregister_params(op="Mul", name="pure")
class PureMul(ONNXForward):
    @staticmethod
    def forward(node: ONNXOp, state: SDFGState,
                sdfg: SDFG) -> typing.Union[nodes.Node, SDFG]:

        node.validate(sdfg, state)

        def prog(A, B, C):
            C[:] = A * B

        return program_for_node(prog, sdfg, state, node).to_sdfg()


@autoregister_params(op="Div", name="pure")
class PureDiv(ONNXForward):
    @staticmethod
    def forward(node: ONNXOp, state: SDFGState,
                sdfg: SDFG) -> typing.Union[nodes.Node, SDFG]:

        node.validate(sdfg, state)

        def prog(A, B, C):
            C[:] = A / B

        return program_for_node(prog, sdfg, state, node).to_sdfg()


@autoregister_params(op="ReduceMean", name="pure")
class PureReduceMean(ONNXForward):
    @staticmethod
    def forward(node: ONNXOp, state: SDFGState,
                sdfg: SDFG) -> typing.Union[nodes.Node, SDFG]:

        node.validate(sdfg, state)

        axes = node.axes

        # when keepdims is true, this works but there is a useless copy. We just leave this for now; this can be fixed
        # with a reshape node when those exist.
        def prog(data, reduced):
            reduced[:] = np.mean(data, axis=axes)

        return program_for_node(prog, sdfg, state, node).to_sdfg()


@autoregister_params(op="Erf", name="pure")
class PureErf(ONNXForward):
    @staticmethod
    def forward_can_be_applied(node: ONNXOp, state: SDFGState,
                               sdfg: SDFG) -> bool:
        return in_desc_with_name(node, state, sdfg, 'input').dtype in [
            dace.float16, dace.float32, dace.float64
        ]

    @staticmethod
    def forward(node: ONNXOp, state: SDFGState,
                sdfg: SDFG) -> typing.Union[nodes.Node, SDFG]:

        node.validate(sdfg, state)

        def prog(input, output):
            output[:] = dace.elementwise(lambda x: erf(x), input)

        return program_for_node(prog, sdfg, state, node).to_sdfg()


@autoregister_params(op="MatMul", name="pure")
class PureMatMul(ONNXForward):
    @staticmethod
    def forward_can_be_applied(node: ONNXOp, state: SDFGState,
                               sdfg: SDFG) -> bool:
        in_edges = state.in_edges(node)
        input0_dim = len(in_desc_with_name(node, state, sdfg, "A").shape)
        input1_dim = len(in_desc_with_name(node, state, sdfg, "B").shape)

        if input0_dim == 4 and input1_dim == 4:
            return True

        if input0_dim == 3 and input1_dim == 2:
            return True

        if input0_dim == 2 and input1_dim == 2:
            return True

        return False

    @staticmethod
    def forward(node: ONNXOp, state: SDFGState,
                sdfg: SDFG) -> typing.Union[nodes.Node, SDFG]:

        node.validate(sdfg, state)
        in_edges = state.in_edges(node)
        out_edges = state.out_edges(node)

        atype = None
        btype = None
        if in_edges[0].dst_conn == "A" and in_edges[1].dst_conn == "B":
            atype = copy.deepcopy(sdfg.arrays[in_edges[0].data.data])
            btype = copy.deepcopy(sdfg.arrays[in_edges[1].data.data])
        if in_edges[0].dst_conn == "B" and in_edges[1].dst_conn == "A":
            atype = copy.deepcopy(sdfg.arrays[in_edges[1].data.data])
            btype = copy.deepcopy(sdfg.arrays[in_edges[0].data.data])

        ctype = copy.deepcopy(sdfg.arrays[out_edges[0].data.data])

        input0_dim = len(in_desc_with_name(node, state, sdfg, "A").shape)
        input1_dim = len(in_desc_with_name(node, state, sdfg, "B").shape)

        if input0_dim == 4 and input1_dim == 4:

            @dace.program
            def einsumop(A: atype, B: btype, Y: ctype):
                Y[:] = np.einsum('abik,abkj->abij', A, B)

            return einsumop.to_sdfg()

        if input0_dim == 3 and input1_dim == 2:

            @dace.program
            def einsumop(A: atype, B: btype, Y: ctype):
                Y[:] = np.einsum('bik,kj->bij', A, B)

            return einsumop.to_sdfg()

        if input0_dim == 2 and input1_dim == 2:
            sdfg_exp = dace.SDFG('matmulExpansion')
            ii = in_edges[0].data.subset.size()[0]
            kk = in_edges[0].data.subset.size()[1]
            jj = in_edges[1].data.subset.size()[1]

            I = str(ii)
            K = str(kk)
            J = str(jj)
            sdfg_exp.add_array('A', (ii, kk),
                               sdfg.arrays[in_edges[0].data.data].dtype)
            sdfg_exp.add_array('B', (kk, jj),
                               sdfg.arrays[in_edges[1].data.data].dtype)
            sdfg_exp.add_array('Y', (ii, jj),
                               sdfg.arrays[out_edges[0].data.data].dtype)

            init_state = sdfg_exp.add_state()
            init_state.add_mapped_tasklet(
                'batched_matmul_init', {
                    '_o%d' % i: '0:%s' % symstr(d)
                    for i, d in enumerate((ii, jj))
                }, {},
                'out = 0', {
                    'out':
                    dace.Memlet.simple(
                        'Y', ','.join(
                            ['_o%d' % i for i in range(len((ii, jj)))]))
                },
                external_edges=True)

            state_exp = sdfg_exp.add_state_after(init_state)

            state_exp.add_mapped_tasklet(
                '_MatMult_',
                {'__i%d' % i: '0:%s' % s
                 for i, s in enumerate([I, J, K])}, {
                     '_a': dace.Memlet.simple("A", ('__i0, __i2')),
                     '_b': dace.Memlet.simple("B", ('__i2, __i1'))
                 },
                '_c = _a * _b', {
                    '_c':
                    dace.Memlet.simple(
                        "Y", '__i0, __i1', wcr_str='lambda x, y: x + y')
                },
                external_edges=True)
            return sdfg_exp


@autoregister_params(op="Identity", name="pure")
class PureIdentity(ONNXForward):
    @staticmethod
    def forward_can_be_applied(node: ONNXOp, state: SDFGState,
                               sdfg: SDFG) -> bool:
        return True

    @staticmethod
    def forward(node: ONNXOp, state: SDFGState,
                sdfg: SDFG) -> typing.Union[nodes.Node, SDFG]:

        node.validate(sdfg, state)

        def prog(input, output):
            output[:] = input

        return program_for_node(prog, sdfg, state, node).to_sdfg()


@autoregister_params(op="Reciprocal", name="pure")
class PureReciprocal(ONNXForward):
    @staticmethod
    def forward_can_be_applied(node: ONNXOp, state: SDFGState,
                               sdfg: SDFG) -> bool:
        return in_desc_with_name(node, state, sdfg, 'X').dtype in [
            dace.float16, dace.float32, dace.float64
        ]

    @staticmethod
    def forward(node: ONNXOp, state: SDFGState,
                sdfg: SDFG) -> typing.Union[nodes.Node, SDFG]:

        node.validate(sdfg, state)

        dtype = in_desc_with_name(node, state, sdfg, 'X').dtype
        tanh_lambda = "lambda x: dace.{}(1) / x".format(dtype.to_string())

        def prog(X, Y):
            Y[:] = dace.elementwise(tanh_lambda, X)

        return program_for_node(prog, sdfg, state, node).to_sdfg()


@autoregister_params(op="Tanh", name="pure")
class PureTanh(ONNXForward):
    @staticmethod
    def forward(node: ONNXOp, state: SDFGState,
                sdfg: SDFG) -> typing.Union[nodes.Node, SDFG]:

        node.validate(sdfg, state)

        def prog(input, output):
            output[:] = dace.elementwise(lambda x: tanh(x), input)

        return program_for_node(prog, sdfg, state, node).to_sdfg()


@autoregister_params(op="ReduceSum", name="pure")
class PureReduceSum(ONNXForward):
    @staticmethod
    def forward(node: ONNXOp, state: SDFGState,
                sdfg: SDFG) -> typing.Union[nodes.Node, SDFG]:
        node.validate(sdfg, state)

        axes = node.axes

        # when keepdims is true, this works but there is a useless copy. We just leave this for now; this can be fixed
        # with a reshape node when those exist.
        def prog(data, reduced):
            reduced[:] = np.sum(data, axis=axes)

        return program_for_node(prog, sdfg, state, node).to_sdfg()


@autoregister_params(op="ReduceMax", name="pure")
class PureReduceMax(ONNXForward):
    @staticmethod
    def forward(node: ONNXOp, state: SDFGState,
                sdfg: SDFG) -> typing.Union[nodes.Node, SDFG]:
        node.validate(sdfg, state)

        axes = node.axes

        # when keepdims is true, this works but there is a useless copy. We just leave this for now; this can be fixed
        # with a reshape node when those exist.
        def prog(data, reduced):
            reduced[:] = np.max(data, axis=axes)

        return program_for_node(prog, sdfg, state, node).to_sdfg()


@autoregister_params(op="ReduceMin", name="pure")
class PureReduceMin(ONNXForward):
    @staticmethod
    def forward(node: ONNXOp, state: SDFGState,
                sdfg: SDFG) -> typing.Union[nodes.Node, SDFG]:
        node.validate(sdfg, state)

        axes = node.axes

        # when keepdims is true, this works but there is a useless copy. We just leave this for now; this can be fixed
        # with a reshape node when those exist.
        def prog(data, reduced):
            reduced[:] = np.min(data, axis=axes)

        return program_for_node(prog, sdfg, state, node).to_sdfg()


@autoregister_params(op="Softmax", name="pure")
class PureSoftmax(ONNXForward):
    @staticmethod
    def forward(node: ONNXOp, state: SDFGState,
                sdfg: SDFG) -> typing.Union[nodes.Node, SDFG]:

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
        # exp - max
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
                '__exp':
                dace.Memlet.simple(
                    "div_tmp",
                    ','.join("__i" + str(i) for i in range(len(inparr.shape))))
            },
            code='__out = __exp / __sum',
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

            out_tmp = dace.define_local(out_tmp_shape, out_tmp_dtype)
            exp_minus_max(exp_tmp_max=tmp_max,
                          exp_input=input,
                          exp_output=out_tmp)

            tmp_sum = np.sum(out_tmp, axis=axis)

            out_tmp_div_sum(div_tmp=out_tmp,
                            div_sum=tmp_sum,
                            div_output=output)

        return program_for_node(prog, sdfg, state, node).to_sdfg()


@autoregister_params(op="Transpose", name="pure")
class PureTranspose(ONNXForward):
    @staticmethod
    def forward(node: ONNXOp, state: SDFGState,
                sdfg: SDFG) -> typing.Union[nodes.Node, SDFG]:

        node.validate(sdfg, state)
        perm = node.perm

        def prog(data, transposed):
            transposed[:] = np.transpose(data, axes=perm)

        return program_for_node(prog, sdfg, state, node).to_sdfg()


@autoregister_params(op="Cast", name="pure")
class PureCast(ONNXForward):
    @staticmethod
    def forward_can_be_applied(node: ONNXOp, state: SDFGState,
                               sdfg: SDFG) -> bool:

        if node.schedule is dtypes.ScheduleType.GPU_Default:
            # TODO fix this (this breaks bert_full) because of a GPU scalar cast. Issue #20
            return False

        target_type = node.to
        try:
            converters.onnx_tensor_type_to_typeclass(target_type)
        except ValueError as v:
            return False

        return True

    @staticmethod
    def forward(node: ONNXOp, state: SDFGState,
                sdfg: SDFG) -> typing.Union[nodes.Node, SDFG]:
        def prog(input, output):
            output[:] = dace.elementwise(lambda x: x, input)

        return program_for_node(prog, sdfg, state, node).to_sdfg()


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
                sdfg: SDFG) -> typing.Union[Node, SDFG]:
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
                                          "i{}".format(i): "0:{}".format(s)
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


@autoregister_params(op="Gemm", name="pure")
class PureGemm(ONNXForward):
    @staticmethod
    def forward_can_be_applied(node: ONNXOp, state: SDFGState,
                               sdfg: SDFG) -> bool:
        if node.alpha == 1.0 and node.beta == 1.0 and node.transA == 0 and node.transB == 1:
            return True
        return False

    @staticmethod
    def forward(node: ONNXOp, state: SDFGState,
                sdfg: SDFG) -> typing.Union[Node, SDFG]:
        node.validate(sdfg, state)

        assert node.alpha == 1.0 and node.beta == 1.0 and node.transA == 0 and node.transB == 1

        # the gemm libnode is broken for now, so we just do it manually
        atype = in_desc_with_name(node, state, sdfg, "A")
        if "C" in node.in_connectors:

            def prog(A, B, C, Y):
                Y[:] = A @ np.transpose(B) + C
        else:

            def prog(A, B, Y):
                Y[:] = A @ np.transpose(B)

        sdfg = program_for_node(prog, sdfg, state, node).to_sdfg()
        sdfg.apply_strict_transformations()
        return sdfg


@autoregister_params(op="Relu", name="pure")
class PureRelu(ONNXForward):
    @staticmethod
    def forward(node: ONNXOp, state: SDFGState,
                sdfg: SDFG) -> typing.Union[Node, SDFG]:
        input_dtype = in_desc_with_name(node, state, sdfg, "X").dtype
        cast_lambda = "lambda x: max(x, dace.{}(0))".format(
            input_dtype.to_string())

        def prog(X, Y):
            Y[:] = dace.elementwise(cast_lambda, X)

        return program_for_node(prog, sdfg, state, node).to_sdfg()


@autoregister_params(op="Reshape", name="pure")
class PureReshape(ONNXForward):
    @staticmethod
    def forward(node: ONNXOp, state: SDFGState,
                sdfg: SDFG) -> typing.Union[Node, SDFG]:
        node.validate(sdfg, state)
        if (in_desc_with_name(node, state, sdfg, "data").dtype !=
                out_desc_with_name(node, state, sdfg, "reshaped")):
            raise ValueError(
                "Expected input and output to have the same dtype.")

        expansion = dace.SDFG("_reshape_expansion_")
        expansion.add_datadesc(
            "shape",
            copy.deepcopy(in_desc_with_name(node, state, sdfg, "shape")))
        expansion.add_datadesc(
            "data",
            copy.deepcopy(in_desc_with_name(node, state, sdfg, "data")))
        expansion.add_datadesc(
            "reshaped",
            copy.deepcopy(out_desc_with_name(node, state, sdfg, "reshaped")))
        expansion.arrays["shape"].transient = False
        expansion.arrays["data"].transient = False
        expansion.arrays["reshaped"].transient = False
        state = expansion.add_state()
        data = state.add_read("data")
        reshaped = state.add_write("reshaped")
        memlet = expansion.make_array_memlet("data")
        memlet.allow_oob = True
        state.add_edge(data, None, reshaped, None, memlet)
        return expansion
