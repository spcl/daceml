import copy
import functools
import typing

import numpy as np

import dace
from dace import SDFGState, SDFG, dtypes
from dace.sdfg import nodes, propagation
from dace.transformation.dataflow import MapExpansion, MapCollapse

from daceml.onnx.forward_implementation_abc import ONNXForward
from daceml.onnx.nodes.onnx_op import ONNXOp
from daceml.onnx.op_implementations.utils import op_implementation, program_for_node
from daceml.util.utils import in_desc_with_name, out_desc_with_name, in_edge_with_name, out_edge_with_name
from daceml.onnx.op_implementations.utils import python_pure_op_implementation


def _2d_sliding_window_index_expr(x_or_y, stride, kernel_size):
    index_expression = "out_{x_or_y} * {stride} + h{x_or_y}"
    return index_expression.format(x_or_y=x_or_y, stride=stride)


def _prod(sequence):
    return functools.reduce(lambda a, b: a * b, sequence, 1)


@op_implementation(op="MaxPool", name="pure")
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
            'outer_pool_map',
            dict(b="0:{}".format(batch_size),
                 c="0:{}".format(num_channels),
                 out_x="0:{}".format(output_size_x),
                 out_y="0:{}".format(output_size_y)))

        # the inner map computes the value for a single entry in the output array (i.e. Y[b, c, x, y])
        inner_me, inner_mx = new_state.add_map(
            'inner_pool_map',
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


@op_implementation(op="Conv", name="pure")
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

        dtype = X.dtype

        @dace.program
        def broadcast(x: dtype[batch_size, num_filters, output_size_y,
                               output_size_x], y: dtype[num_filters]):

            for b, m, out_x, out_y in dace.map[0:batch_size, 0:num_filters,
                                               0:output_size_y,
                                               0:output_size_x]:
                with dace.tasklet:
                    inp << y[m]
                    outp >> x[b, m, out_x, out_y]
                    outp = inp

        @dace.program
        def zero_init(x: dtype[batch_size, num_filters, output_size_y,
                               output_size_x]):
            for b, m, out_x, out_y in dace.map[0:batch_size, 0:num_filters,
                                               0:output_size_y,
                                               0:output_size_x]:
                with dace.tasklet:
                    outp >> x[b, m, out_x, out_y]
                    outp = 0

        if B is None:

            def conv(X, Y, W):
                zero_init(Y)
                for b, m, out_x, out_y, cin, hx, hy in dace.map[
                        0:batch_size, 0:num_filters, 0:output_size_y,
                        0:output_size_x, 0:num_channels, 0:filter_hx,
                        0:filter_hy]:
                    with dace.tasklet:
                        filter << W[m, cin, hx, hy]
                        image << X[b, cin, hx + out_x, hy + out_y]
                        out >> Y(1, lambda x, y: x + y)[b, m, out_x, out_y]
                        out = filter * image
        else:

            def conv(X, Y, W, B):
                broadcast(Y, B)
                for b, m, out_x, out_y, cin, hx, hy in dace.map[
                        0:batch_size, 0:num_filters, 0:output_size_y,
                        0:output_size_x, 0:num_channels, 0:filter_hx,
                        0:filter_hy]:
                    with dace.tasklet:
                        filter << W[m, cin, hx, hy]
                        image << X[b, cin, hx + out_x, hy + out_y]
                        out >> Y(1, lambda x, y: x + y)[b, m, out_x, out_y]
                        out = filter * image

        nsdfg = program_for_node(conv, sdfg, state, node)

        compute_state = nsdfg.node(1)
        nsdfg.apply_transformations(MapExpansion, states=[compute_state])

        read_X = [
            n for n in compute_state.nodes()
            if isinstance(n, nodes.AccessNode) and n.data == "X"
        ]
        assert len(read_X) == 1
        read_X = read_X[0]
        path = compute_state.memlet_path(compute_state.out_edges(read_X)[0])

        entry_nodes = [
            e.dst for e in path if isinstance(e.dst, nodes.EntryNode)
        ]

        # merge the first 4 maps
        me = entry_nodes[0]
        for i in range(1, 4):
            me, _ = MapCollapse.apply_to(nsdfg,
                                         _outer_map_entry=me,
                                         _inner_map_entry=entry_nodes[i])

        # merge the second 3 maps
        me = entry_nodes[4]
        for i in range(5, 7):
            me, _ = MapCollapse.apply_to(nsdfg,
                                         _outer_map_entry=me,
                                         _inner_map_entry=entry_nodes[i])

        return nsdfg


@op_implementation(op="BatchNormalization", name="pure")
class PureBatchNormalization(ONNXForward):
    @staticmethod
    def forward_can_be_applied(node: ONNXOp, state: SDFGState,
                               sdfg: SDFG) -> bool:
        X = in_desc_with_name(node, state, sdfg, "X")
        if len(X.shape) != 4:
            return False

        # only for training for now
        if not {"out_mean", "out_var", "saved_mean", "saved_var"}.issubset(
                node.out_connectors):
            return False
        if not {"scale", "B"}.issubset(node.in_connectors):
            return False

        return True

    @staticmethod
    def forward(node: ONNXOp, state: SDFGState,
                sdfg: SDFG) -> typing.Union[nodes.Node, SDFG]:

        reduce_axes = list(
            copy.deepcopy(in_desc_with_name(node, state, sdfg, "X").shape))
        num_channels = reduce_axes.pop(1)

        N = _prod(reduce_axes)
        broadcast_shape = [num_channels, 1, 1]
        dtype = in_desc_with_name(node, state, sdfg, "X").dtype
        eps = node.epsilon
        momentum = node.momentum
        inv_momentum = 1 - node.momentum

        def prog(X, scale, B, in_mean, in_var, Y, out_mean, out_var,
                 saved_mean, saved_var):
            saved_mean[:] = np.add.reduce(X, axis=[0, 2, 3]) / N

            saved_mean_broadcastable = dace.define_local(
                broadcast_shape, dtype)
            # this copy will get removed after parsing -- using reshape here would be nicer
            # but it messes with statefusion
            saved_mean_broadcastable[:] = saved_mean

            X_minus_mean = (X - saved_mean_broadcastable)

            saved_var[:] = np.add.reduce(X_minus_mean * X_minus_mean,
                                         axis=(0, 2, 3)) / N
            saved_var_eps = np.reshape(saved_var + eps, broadcast_shape)

            normalized = X_minus_mean * dace.elementwise(
                lambda x: dace.float32(1.0) / sqrt(x), saved_var_eps)

            scale_reshaped = np.reshape(scale, broadcast_shape)
            bias_reshaped = np.reshape(B, broadcast_shape)
            Y[:] = normalized * scale_reshaped + bias_reshaped

            out_mean[:] = in_mean * momentum + saved_mean * inv_momentum
            out_var[:] = in_var * momentum + saved_var * inv_momentum

        new_sdfg = program_for_node(prog, sdfg, state, node)

        # write the mean and var back to the parameters so that they are updated
        # this is a bit of a hack, but the ONNX spec is currently not really working for training
        new_state = sdfg.add_state_after(sdfg.nodes()[0])
        mean_data_name = out_edge_with_name(node, state, "out_mean").data.data
        read_mean = new_state.add_read(mean_data_name)
        write_mean = new_state.add_read(
            in_edge_with_name(node, state, "in_mean").data.data)
        new_state.add_edge(read_mean, None, write_mean, None,
                           sdfg.make_array_memlet(mean_data_name))

        var_data_name = out_edge_with_name(node, state, "out_var").data.data
        read_var = new_state.add_read(var_data_name)
        write_var = new_state.add_read(
            in_edge_with_name(node, state, "in_var").data.data)
        new_state.add_edge(read_var, None, write_var, None,
                           sdfg.make_array_memlet(var_data_name))

        return new_sdfg


@python_pure_op_implementation
def GlobalAveragePool(X, Y):
    Y[:] = np.mean(X, axis=[2, 3])
