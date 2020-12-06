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
import math

from daceml.util.utils import in_desc_with_name, out_desc_with_name


def _2d_sliding_window_index_expr(x_or_y, stride, kernel_size):
    index_expression = "out_{x_or_y} * {stride} + h{x_or_y}"
    return index_expression.format(x_or_y=x_or_y, stride=stride)


@autoregister_params(op="Conv", name="fpga")
class FPGAConv2D(ONNXForward):
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

        new_sdfg = dace.SDFG("fpga_conv")

        new_state = new_sdfg.add_state("compute")
        new_sdfg.add_datadesc("X", copy.deepcopy(X))
        new_sdfg.add_datadesc("W", copy.deepcopy(W))
        new_sdfg.add_datadesc("Y", copy.deepcopy(Y))
        if B is not None:
            new_sdfg.add_datadesc("B", copy.deepcopy(B))
            new_sdfg.arrays["B"].transient = False

        #TODO: stride
        assert (stride_x == 1 and stride_y == 1)

        # add local storage for weights
        new_sdfg.add_array('local_W',
                           shape=W.shape,
                           dtype=W.dtype,
                           storage=dace.dtypes.StorageType.FPGA_Local,
                           transient=True)

        # add local storage for X and Y, to increase reuse

        # for X we will reuse the data of a given input channel to update the result for all output channels
        new_sdfg.add_array('local_X',
                           shape=[num_channels, filter_hx, filter_hy],
                           dtype=X.dtype,
                           storage=dace.dtypes.StorageType.FPGA_Local,
                           transient=True)

        # for Y we will reuse by accumulating on the same output channel
        new_sdfg.add_array('local_Y',
                           shape=[num_filters],
                           dtype=Y.dtype,
                           storage=dace.dtypes.StorageType.FPGA_Local,
                           transient=True)

        new_sdfg.arrays["X"].transient = False
        new_sdfg.arrays["W"].transient = False
        new_sdfg.arrays["Y"].transient = False

        # we don't need init state for Y. This is done on the fly in the tasklet

        # preload weights
        preload_W_map_entry, preload_W_map_exit = new_state.add_map(
            'preload_weights_map',
            dict(m='0:{}'.format(num_filters),
                 cin="0:{}".format(num_channels),
                 hx="0:{}".format(filter_hx),
                 hy="0:{}".format(filter_hy)))
        preload_W_task = new_state.add_tasklet("preload_weights_tasklet",
                                               inputs={"w_in"},
                                               outputs={"w_out"},
                                               code="w_out = w_in")
        # add edges
        preload_W_read = new_state.add_read("W")
        local_W_access = new_state.add_access("local_W")

        new_state.add_memlet_path(
            preload_W_read,
            preload_W_map_entry,
            preload_W_task,
            dst_conn='w_in',
            memlet=dace.Memlet(f"{preload_W_read.data}[m, cin, hx, hy]"))
        new_state.add_memlet_path(
            preload_W_task,
            preload_W_map_exit,
            local_W_access,
            src_conn='w_out',
            memlet=dace.Memlet(f"{local_W_access.data}[m, cin,hx,hy]"))

        # In pure implementation we have two maps:
        # - the outer map loops over every entry in the output array
        # - the inner inner map computes the value for a single entry in the output array (i.e. Y[b, m, x, y])

        # Here we want to increase reuse of the input feature, that is read the input once and oupdate all the
        # m output channels. Therefore we interchange some of maps indices.
        # - the outer map loops over every entry in the ouput array, not considering the channel (Y[b,:,x,y])
        # - a mid map over the input channels (this is splitted from the inner map just to have more control on unrolling)
        # - the inner computes the value for all the entries of a given point

        # the outer map loops over every entry in the output array
        outer_me, outer_mx = new_state.add_map(
            'outer_conv_map',
            dict(b="0:{}".format(batch_size),
                 out_x="0:{}".format(output_size_x),
                 out_y="0:{}".format(output_size_y)))

        mid_me, mid_mx = new_state.add_map(
            'mid_conv_map', dict(cin="0:{}".format(num_channels)))

        # the inner map computes the value for a single entry in the output array (i.e. Y[b, m, x, y])
        inner_me, inner_mx = new_state.add_map(
            'inner_conv_map',
            dict(m="0:{}".format(num_filters),
                 hx="0:{}".format(filter_hx),
                 hy="0:{}".format(filter_hy)),
            unroll=True)

        # we have to fill local_x properly: this should happen between the outer and the innermost map
        # The actual loading into local_X will be done in the tasklet, where we can add `if` conditions
        # Note: this is not pure SDFG API: the cleanest solution would involve creating another nested SDFG
        local_X_read = new_state.add_access("local_X")

        # empty memlet to create the storage
        new_state.add_memlet_path(outer_me, local_X_read, memlet=dace.Memlet())

        # Similarly, we will use local_Y to accumulate while computing in the innermost map
        local_Y_read = new_state.add_access("local_Y")
        local_Y_write = new_state.add_write("local_Y")
        new_state.add_memlet_path(outer_me, local_Y_read, memlet=dace.Memlet())

        inputs = {"image_in", "local_X_in", "filter_in", "local_Y_in"}
        if B is not None:
            inputs.add("B_in")

        # In the tasklet we read local_X (for every given input channel) and
        # we write the final result if we are computing over the last input channel
        compute_tasklet = new_state.add_tasklet(
            "compute_entry",
            inputs=inputs,
            outputs={"output", "local_Y_out"},
            code="if m==0: local_X_in = image_in\n"
            "local_Y_out = (0 if hx == 0 and hy==0 and cin==0 else local_Y_in)  + local_X_in * filter_in\n"
            # "local_X_out = local_X_in\n"
            "if hx == {}-1 and hy == {}-1 and cin=={}-1: output = local_Y_out {}"
            .format(filter_hx, filter_hy, num_channels,
                    "+ B_in" if B is not None else ""))

        filter_memlet = dace.Memlet("local_W[m, cin, hx, hy]")

        x_idx = _2d_sliding_window_index_expr(x_or_y="x",
                                              stride=stride_x,
                                              kernel_size=filter_hx)
        y_idx = _2d_sliding_window_index_expr(x_or_y="y",
                                              stride=stride_y,
                                              kernel_size=filter_hy)

        image_memlet = dace.Memlet("X[b, cin, {}, {}]".format(x_idx, y_idx))
        # hook up the inner map to the tasklet

        # local X goes inside the tasklet. Being a dynamic element, this will be codegenerated as a pointer
        # and therefore will also write back into the tile of X
        new_state.add_memlet_path(local_X_read,
                                  mid_me,
                                  inner_me,
                                  compute_tasklet,
                                  dst_conn='local_X_in',
                                  memlet=dace.Memlet(
                                      f"{local_X_read.data}[cin, hx, hy]",
                                      dynamic=True))

        # similarly, local Y
        new_state.add_memlet_path(
            local_Y_read,
            mid_me,
            inner_me,
            compute_tasklet,
            dst_conn='local_Y_in',
            memlet=dace.Memlet(f"{local_Y_read.data}[m]"))
        new_state.add_memlet_path(
            compute_tasklet,
            inner_mx,
            mid_mx,
            local_Y_write,
            src_conn='local_Y_out',
            memlet=dace.Memlet(f"{local_Y_write.data}[m]"))

        # hook up filter
        # new_state.add_edge(inner_me, None, compute_tasklet, "filter_in",
        #                    filter_memlet)
        # inner_filter_memlet = propagation.propagate_memlet(
        #     new_state, filter_memlet, inner_me, False)
        # outer_filter_memlet = propagation.propagate_memlet(
        #     new_state, inner_filter_memlet, outer_me, False)
        # new_state.add_edge(outer_me, None, inner_me, None, inner_filter_memlet)
        # new_state.add_edge(local_W_access, None, outer_me, None, outer_filter_memlet)
        new_state.add_memlet_path(local_W_access,
                                  outer_me,
                                  mid_me,
                                  inner_me,
                                  compute_tasklet,
                                  dst_conn='filter_in',
                                  memlet=filter_memlet)

        # hook up X: this goes directly to the tasklet
        read_X = new_state.add_read("X")
        # new_state.add_edge(inner_me, None, compute_tasklet, "image_in",
        #                    image_memlet)
        # inner_image_memlet = propagation.propagate_memlet(
        #     new_state, image_memlet, inner_me, False)
        # outer_image_memlet = propagation.propagate_memlet(
        #     new_state, inner_image_memlet, outer_me, False)
        # new_state.add_edge(outer_me, None, inner_me, None, inner_image_memlet)
        # new_state.add_edge(read_X, None, outer_me, None, outer_image_memlet)
        new_state.add_memlet_path(read_X,
                                  outer_me,
                                  mid_me,
                                  inner_me,
                                  compute_tasklet,
                                  dst_conn='image_in',
                                  memlet=image_memlet)

        # hook up outputs
        # The output memlet is set to be dynamic, so that the value is only written at the end of the computation
        output_memlet = dace.Memlet("Y[b, m, out_x, out_y]", dynamic=True)
        write_Y = new_state.add_write("Y")
        # inner_output_memlet = propagation.propagate_memlet(
        #     new_state, output_memlet, inner_me, False)
        # outer_output_memlet = propagation.propagate_memlet(
        #     new_state, inner_output_memlet, outer_me, False)
        # new_state.add_edge(compute_tasklet, "output", inner_mx, None,
        #                    output_memlet)
        #
        # new_state.add_edge_pair(outer_mx, inner_mx, write_Y,
        #                         inner_output_memlet, outer_output_memlet)

        new_state.add_memlet_path(compute_tasklet,
                                  inner_mx,
                                  mid_mx,
                                  outer_mx,
                                  write_Y,
                                  src_conn='output',
                                  memlet=output_memlet)

        # hook up B if required
        if B is not None:
            read_B = new_state.add_read("B")
            B_memlet = dace.Memlet("B[m]")
            new_state.add_memlet_path(read_B,
                                      outer_me,
                                      mid_me,
                                      inner_me,
                                      compute_tasklet,
                                      dst_conn='B_in',
                                      memlet=B_memlet)

        new_sdfg.fill_scope_connectors()
        new_sdfg.save('/tmp/conv.sdfg')
        return new_sdfg


@autoregister_params(op="Relu", name="fpga")
class FPGARelu(ONNXForward):
    @staticmethod
    def forward(node: ONNXOp, state: SDFGState,
                sdfg: SDFG) -> typing.Union[Node, SDFG]:

        X = in_desc_with_name(node, state, sdfg, "X")
        Y = out_desc_with_name(node, state, sdfg, "Y")

        # as vec width take the gcd between 32 (max vect width) and the shape of X
        vec_width = math.gcd(X.shape[-1], 32)

        # Build map ranges: one loop per dimension, with the last one being
        # strip mined to expose vectorization
        map_ranges = {
            '__i%d' % i: '0:%s' % n
            for i, n in enumerate(X.shape[:-1])
        }
        map_ranges[f'__i{len(X.shape)-1}'] = f"0:{X.shape[-1]//vec_width}"

        new_sdfg = dace.SDFG("fpga_relu")

        new_state = new_sdfg.add_state("compute")
        new_sdfg.add_datadesc("X", copy.deepcopy(X))
        new_sdfg.add_datadesc("Y", copy.deepcopy(Y))

        outer_me, outer_mx = new_state.add_map('outer_relu_map', map_ranges)

        inner_me, inner_mx = new_state.add_map(
            'inner_relu_map', dict(i="0:{}".format(vec_width)), unroll=True)

        tasklet = new_state.add_tasklet('relu_task', ['x_con'], ['y_con'],
                                        'y_con = max(0.0, x_con)')
        x_read = new_state.add_read("X")
        y_write = new_state.add_write("Y")

        new_state.add_memlet_path(
            x_read,
            outer_me,
            inner_me,
            tasklet,
            dst_conn='x_con',
            memlet=dace.Memlet("X[{}, __i{}*{}+i]".format(
                ",".join(['__i%d' % i for i in range(len(X.shape) - 1)]),
                len(X.shape) - 1, vec_width)))
        new_state.add_memlet_path(
            tasklet,
            inner_mx,
            outer_mx,
            y_write,
            src_conn='y_con',
            memlet=dace.Memlet("Y[{}, __i{}*{}+i]".format(
                ",".join(['__i%d' % i for i in range(len(X.shape) - 1)]),
                len(X.shape) - 1, vec_width)))
        new_sdfg.fill_scope_connectors()
        new_sdfg.save('/tmp/relu.sdfg')
        return new_sdfg


@autoregister_params(op="MaxPool", name="fpga")
class FPGAMaxPool2D(ONNXForward):
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

        # MAX Pool: the current implementation exploit a sliding window. Considering a single batch and a single
        # channel, we will read one input element at a time, shifting

        #TODO: this implementation depends on how data will be streamed
        # for the moment being we assume it sends one channel after the other

        # TODO: unroll reads from memory/stream
        # TODO: pay attention to do not mix height, width

        X = in_desc_with_name(node, state, sdfg, "X")
        Y = out_desc_with_name(node, state, sdfg, "Y")

        image_dims = len(X.shape) - 2
        batch_size = X.shape[0]
        num_channels = X.shape[1]
        strides = node.strides if node.strides is not None else [
            1 for _ in range(image_dims)
        ]
        stride_height, stride_width = strides
        filter_height, filter_width = node.kernel_shape
        input_size_height, input_size_width = X.shape[2:]
        output_size_y, output_size_x = Y.shape[2:]

        new_sdfg = dace.SDFG("fpga_maxpool")
        new_state = new_sdfg.add_state("compute")

        # we don't need initialization

        new_sdfg.add_datadesc("X", copy.deepcopy(X))
        new_sdfg.add_datadesc("Y", copy.deepcopy(Y))
        new_sdfg.arrays["X"].transient = False
        new_sdfg.arrays["Y"].transient = False

        #shift register
        shift_register_size = input_size_width * (filter_height - 1) + (
            filter_width - 1) + 1
        new_sdfg.add_array("shift_register", [shift_register_size],
                           X.dtype,
                           storage=dace.StorageType.FPGA_ShiftRegister,
                           transient=True)
        # variable for reduction
        new_sdfg.add_array("max_res", [1],
                           X.dtype,
                           storage=dace.StorageType.FPGA_Registers,
                           transient=True)
        # the outer map loops over every entry in the input array
        # (useful also in the case of streaming input, we can't skip data
        outer_me, outer_mx = new_state.add_map(
            'outer_pool_map',
            dict(b="0:{}".format(batch_size),
                 c="0:{}".format(num_channels),
                 in_y="0:{}".format(input_size_height),
                 in_x="0:{}".format(input_size_width)))

        # TODO: use the pipeline?
        # TODO: che draining if the input is a stream (in case add a conditional read)

        # the inner map computes the pooling
        inner_me, inner_mx = new_state.add_map(
            'inner_pool_map',
            dict(hy="0:{}".format(filter_height),
                 hx="0:{}".format(filter_width)),
            unroll=True)

        # compute the maximum: we can compute always, but we can write the result only
        # according to the slide and at the end of the filter loops
        compute_tasklet = new_state.add_tasklet(
            "compute_entry",
            inputs={"image_in", "max_in"},
            outputs={"output", "max_out"},
            #code="output = image_in"
            code="if hx == 0 and hy == 0: max_in = {}\n"  #init
            "max_out = float(max(max_in, image_in))\n"
            "if hy == {} - 1 and hx == {} -1 and  in_y % {} == {} - 1 and in_x % {} == {} -1: output = max_out"
            .format(dtypes.min_value(Y.dtype), filter_height, filter_width,
                    filter_height, filter_height, filter_height, filter_width))

        shift_register = new_state.add_access("shift_register")
        read_X = new_state.add_read("X")
        write_Y = new_state.add_write("Y")
        read_max_res = new_state.add_access("max_res")
        write_max_res = new_state.add_write("max_res")

        # memlet: from input image to shift register
        new_state.add_memlet_path(
            read_X,
            outer_me,
            shift_register,
            memlet=dace.Memlet("X[b, c, in_y, in_x]",
                               other_subset="{}".format(shift_register_size -
                                                        1)))

        # memlet from shift register to max tasklet
        new_state.add_memlet_path(
            shift_register,
            inner_me,
            compute_tasklet,
            dst_conn="image_in",
            memlet=dace.Memlet(
                "shift_register[hy*{}+hx]".format(input_size_width)))

        #memlets for max
        new_state.add_memlet_path(read_max_res,
                                  inner_me,
                                  compute_tasklet,
                                  dst_conn="max_in",
                                  memlet=dace.Memlet("max_res[0]"))
        new_state.add_memlet_path(outer_me, read_max_res, memlet=dace.Memlet())

        new_state.add_memlet_path(compute_tasklet,
                                  inner_mx,
                                  write_max_res,
                                  src_conn="max_out",
                                  memlet=dace.Memlet("max_res[0]"))

        y_memlet = dace.Memlet("Y[b,c, in_y//{}, in_x//{}]".format(
            filter_height, filter_width),
                               dynamic=True)
        #dynamic memlet (to access only when needed) from compute tasklet to out image
        # Attention: use propagate=False otherwise it does not validate
        new_state.add_memlet_path(compute_tasklet,
                                  inner_mx,
                                  outer_mx,
                                  write_Y,
                                  src_conn="output",
                                  memlet=y_memlet,
                                  propagate=False)

        new_sdfg.fill_scope_connectors()
        new_sdfg.save("/tmp/maxpool.sdfg")
        return new_sdfg
