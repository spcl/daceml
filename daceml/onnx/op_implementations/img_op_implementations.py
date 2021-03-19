import copy
import typing

import dace
from dace import SDFGState, SDFG, dtypes
from dace.registry import autoregister_params
from dace.sdfg import nodes, propagation

from daceml.onnx.forward_implementation_abc import ONNXForward
from daceml.onnx.nodes.onnx_op import ONNXOp
from daceml.util.utils import in_desc_with_name, out_desc_with_name


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
    """ The "trivial" convolution implementation, i.e. two nested maps.
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


@autoregister_params(op="Conv", name="im2col")
class Im2ColConv(ONNXForward):
    """ Conv implementation based on Gemm

        Note interesting CPU optimizations for Im2Col:
        https://github.com/BVLC/caffe/pull/3536
        (might be relevant)
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

        if node.kernel_shape is not None:
            filter_hx, filter_hy = node.kernel_shape
        else:
            filter_hx, filter_hy = W.shape[2:]

        num_filters = W.shape[0]
        num_channels = X.shape[1]
        batch_size = X.shape[0]

        output_size_x, output_size_y = Y.shape[2:]

        new_sdfg = dace.SDFG("im2col_conv")

        # setup inputs and outputs
        new_state = new_sdfg.add_state()
        new_sdfg.add_datadesc("X", copy.deepcopy(X))

        new_sdfg.add_datadesc("W", copy.deepcopy(W))
        new_sdfg.add_datadesc("Y", copy.deepcopy(Y))
        if B is not None:
            new_sdfg.add_datadesc("B", copy.deepcopy(B))
            new_sdfg.arrays["B"].transient = False

        new_sdfg.arrays["X"].transient = False
        new_sdfg.arrays["W"].transient = False
        new_sdfg.arrays["Y"].transient = False

        # the batch map loops over every image in the batch
        batch_me, batch_mx = new_state.add_map(
            'batch_map',
            dict(b="0:{}".format(batch_size)),
            schedule=dtypes.ScheduleType.
            Sequential  # todo why does non-sequential fail on CPU
        )

        # for each image, we create the im2col matrix
        # im2col_map fills one entry in I per "iteration"
        ##############################################################
        new_sdfg.add_array(
            "I",
            [num_channels, filter_hx, filter_hy, output_size_x, output_size_y],
            X.dtype,
            transient=True)
        access_I = new_state.add_access("I")
        im2col_me, im2col_mx = new_state.add_map(
            'im2col_map',
            dict(cin="0:{}".format(num_channels),
                 hx="0:{}".format(filter_hx),
                 hy="0:{}".format(filter_hy),
                 x="0:{}".format(output_size_y),
                 y="0:{}".format(output_size_x)))

        # add im2col tasklet and connect it to the im2col map
        im2col_tasklet = new_state.add_tasklet("im2col_copy", {"input"},
                                               {"output"}, "output = input")

        im2col_input_memlet = dace.Memlet("X[b, cin, x + hx, y + hy]")
        im2col_output_memlet = dace.Memlet("I[cin, hx, hy, x, y]")

        new_state.add_edge(im2col_me, None, im2col_tasklet, "input",
                           im2col_input_memlet)
        new_state.add_edge(im2col_tasklet, "output", im2col_mx, None,
                           im2col_output_memlet)

        # connect the im2col_map to the im2col buffer:
        new_state.add_edge(
            im2col_mx, None, access_I, None,
            propagation.propagate_memlet(new_state, im2col_output_memlet,
                                         im2col_me, False))

        # connect the image to the im2col_map
        im2col_me_memlet = propagation.propagate_memlet(
            new_state, im2col_input_memlet, im2col_me, False)
        new_state.add_edge(batch_me, None, im2col_me, None, im2col_me_memlet)
        new_state.add_edge(
            new_state.add_read("X"), None, batch_me, None,
            propagation.propagate_memlet(new_state, im2col_me_memlet, batch_me,
                                         False))

        # add a gemm_node within a nested sdfg to multiply the weights and the im2col matrix
        # we use the nested sdfg to reshape the weights, biases and matrix

        im2col_desc = X.dtype[num_channels * filter_hx * filter_hy,
                              output_size_x * output_size_y]
        weights_desc = X.dtype[num_filters,
                               num_channels * filter_hx * filter_hy]
        result_desc = X.dtype[num_filters, output_size_x * output_size_y]

        # avoid import loop
        import daceml.onnx as donnx
        if B is not None:
            # biases must be reshaped for correct broadcasting
            biases_desc = X.dtype[num_filters, 1]

            @dace.program
            def matmul_nsdfg(weights: weights_desc, im2col: im2col_desc,
                             biases: biases_desc, result: result_desc):
                donnx.ONNXGemm(A=weights, B=im2col, C=biases, Y=result)

            gemm_sdfg = new_state.add_nested_sdfg(
                matmul_nsdfg.to_sdfg(), None, {"weights", "im2col", "biases"},
                {"result"})

            # connect biases -> matmul
            new_state.add_edge(new_state.add_read("B"), None, batch_me, None,
                               new_sdfg.make_array_memlet("B"))
            new_state.add_edge(batch_me, None, gemm_sdfg, "biases",
                               new_sdfg.make_array_memlet("B"))
        else:

            @dace.program
            def matmul_nsdfg(weights: weights_desc, im2col: im2col_desc,
                             result: result_desc):
                donnx.ONNXGemm(A=weights, B=im2col, Y=result)

            gemm_sdfg = new_state.add_nested_sdfg(matmul_nsdfg.to_sdfg(), None,
                                                  {"weights", "im2col"},
                                                  {"result"})

        # connect im2col -> matmul
        new_state.add_edge(access_I, None, gemm_sdfg, "im2col",
                           new_sdfg.make_array_memlet("I"))

        # connect weights -> matmul
        new_state.add_edge(new_state.add_read("W"), None, batch_me, None,
                           new_sdfg.make_array_memlet("W"))
        new_state.add_edge(batch_me, None, gemm_sdfg, "weights",
                           new_sdfg.make_array_memlet("W"))

        # connect matmul -> Y
        new_state.add_edge(
            gemm_sdfg, "result", batch_mx, None,
            dace.Memlet("Y[b, 0:{}, 0:{}, 0:{}]".format(
                num_filters, output_size_x, output_size_y)))
        new_state.add_edge(batch_mx, None, new_state.add_write("Y"), None,
                           new_sdfg.make_array_memlet("Y"))

        new_sdfg.fill_scope_connectors()

        return new_sdfg
