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


@autoregister_params(op="Conv", name="naive_fpga")
class FPGAConv2D(ONNXForward):
    """
    The "trivial" convolution implementation, i.e. two nested maps.
    Does not work in hardware...needs some work on the unrolling etc. et.c
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


@autoregister_params(op="Conv", name="fpga")
class FPGAIm2ColConv(ONNXForward):
    """ Conv implementation based on Gemm

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

        new_sdfg = dace.SDFG("fpga_im2col_conv")

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

        # GEMM Parameters

        #N = num_filters
        K = num_channels * filter_hx * filter_hy
        M = output_size_y * output_size_x
        P = num_filters  # Num PEs  #TODO parametric
        #TODO: maybe this should depend also on output_size_x?
        vec_width = math.gcd(output_size_x, 16)  # TODO: parametric
        def make_read_W(state):
            # this will read the weights, organized as a matrix of size
            # num_filters x (num_channels * filter_hx * filter_hy)

            # The original weight matrix has shape [num_filters, num_channels, filter_hx, filter_hy]

            # TODO: vectorize also this, by reading more than one element at a time, to be memory friendly
            entry, exit = state.add_map(
                "read_weights",
                {
                    "b": "0:{}".format(
                        batch_size
                    ),  # the batch map loops over every image in the batch
                    "n0": "0:{}/{}".format(num_filters, P),
                    "cin": "0:{}".format(num_channels),
                    "hx": "0:{}".format(filter_hx),
                    "hy": "0:{}".format(filter_hy),
                    "n1": "0:{}".format(P)
                },
                schedule=dace.ScheduleType.FPGA_Device)

            mem = state.add_read("W")
            pipe = state.add_write("W_pipe")
            tasklet = state.add_tasklet("read_W", {"from_memory"},
                                        {"to_kernel"},
                                        "to_kernel = from_memory")

            state.add_memlet_path(
                mem,
                entry,
                tasklet,
                dst_conn="from_memory",
                memlet=dace.Memlet("W[n0 * {} + n1, cin, hx, hy]".format(P)))
            state.add_memlet_path(tasklet,
                                  exit,
                                  pipe,
                                  src_conn="to_kernel",
                                  memlet=dace.Memlet("W_pipe[0]"))

        def make_read_im2col(state, sdfg, vec_width=1):

            # Matrix B will be the im2col matrix. We will build it row-by-row
            # to facilitate streaming in the systolic GEMM, avoiding storing it back to memory
            # Note: this will require to load multiple times the input feature, yet this save I/Os
            # The im2col matrix has size (num_channels * filter_hx * filter_hy) x (output_size_y * output_size_x)

            # gear boxing: we read plain data types, we stream vector data types
            # Therefore we have two maps, the innermost is unrolled
            im2col_me, im2col_mx = state.add_map(
                "im2col_map",
                {
                    "b": "0:{}".format(batch_size),
                    "n": "0:{}/{}".format(
                        num_filters, P),  # repeat B for computing the result
                    "cin": "0:{}".format(num_channels),
                    "hx": "0:{}".format(filter_hx),
                    "hy": "0:{}".format(filter_hy),
                    "x": "0:{}".format(output_size_y),
                    "y0": "0:{}/{}".format(output_size_x,
                                           vec_width),  #TODO vectorize read
                },
                schedule=dace.ScheduleType.FPGA_Device)

            read_map_entry, read_map_exit = state.add_map(
                "unrolled_reads_X", {"y1": "0:{}".format(vec_width)},
                schedule=dace.ScheduleType.FPGA_Device,
                unroll=True)

            # local storage to accumulate data
            sdfg.add_array('vec_data_im2col',
                           shape=[vec_width],
                           dtype=dace.float32,
                           transient=True,
                           storage=dace.dtypes.StorageType.FPGA_Registers)

            X = state.add_read("X")
            pipe = state.add_write("im2col_pipe")
            vect_data = state.add_access("vec_data_im2col")
            tasklet = state.add_tasklet("read_X", {"from_memory"},
                                        {"to_kernel"},
                                        "to_kernel = from_memory")

            im2col_input_memlet = dace.Memlet(
                "X[b, cin, x + hx, y0*{}+y1 + hy]".format(vec_width))

            # TODO check that offset to X are right in the codegenerated code

            # In the innermost map we read W=vec_width data elements and we store them into `vec_data`
            state.add_memlet_path(X,
                                  im2col_me,
                                  read_map_entry,
                                  tasklet,
                                  dst_conn="from_memory",
                                  memlet=im2col_input_memlet)

            state.add_memlet_path(tasklet,
                                  read_map_exit,
                                  vect_data,
                                  src_conn="to_kernel",
                                  memlet=dace.Memlet("vec_data_im2col[y1]"))

            # then we transfer them to the output stream
            copy_out_tasklet = state.add_tasklet('pack_and_copy_to_stream_B',
                                                 {'in_con'}, {'out_con'},
                                                 'out_con = in_con')
            state.add_memlet_path(vect_data,
                                  copy_out_tasklet,
                                  dst_conn="in_con",
                                  memlet=dace.Memlet("vec_data_im2col"))

            state.add_memlet_path(copy_out_tasklet,
                                  im2col_mx,
                                  pipe,
                                  src_conn="out_con",
                                  memlet=dace.Memlet("im2col_pipe[0]"))

        def make_write_Y(state, sdfg, vec_width, add_bias=True):

            # The resulting matrix will have size num_filter x (output_size_x, output_size_y)
            # Given the current systolic implementation, we will receive it one row at a time

            # We don't need to accumulate on Y, but we need to add Biases (if present)

            # C data arrives as expressed in vect. data type. Needs to be unpacked
            # For doing so we first store it into a local buffer and then we write it in memory
            # as gear boxing works on local data only (not global memory)

            pipe = state.add_read("Y_pipe")
            mem = state.add_write("Y")
            if add_bias is True:
                B = state.add_read("B")
            entry_map, exit_map = state.add_map(
                "write_Y", {
                    "b": "0:{}".format(batch_size),
                    "n": "0:{}".format(num_filters),
                    "x": "0:{}".format(output_size_x),
                    "y0": "0:{}/{}".format(output_size_y, vec_width)
                },
                schedule=dace.ScheduleType.FPGA_Device)

            # TODO: deal with vect data type
            write_map_entry, write_map_exit = state.add_map(
                "unrolled_write_Y", {"y1": "0:{}".format(vec_width)},
                schedule=dace.ScheduleType.FPGA_Device,
                unroll=True)

            # local storage to accumulate data
            sdfg.add_array('vec_data_Y',
                           shape=[vec_width],
                           dtype=dace.float32,
                           transient=True,
                           storage=dace.dtypes.StorageType.FPGA_Registers)

            vect_data = state.add_access("vec_data_Y")

            copy_in_tasklet = state.add_tasklet('copy_from_stream_Y',
                                                {'in_con'}, {'out_con'},
                                                'out_con = in_con')

            state.add_memlet_path(pipe,
                                  entry_map,
                                  copy_in_tasklet,
                                  dst_conn="in_con",
                                  memlet=dace.Memlet("Y_pipe[{}-1]".format(P)))
            # this will trigger gear boxing
            state.add_memlet_path(copy_in_tasklet,
                                  vect_data,
                                  src_conn="out_con",
                                  memlet=dace.Memlet("vec_data_Y"))

            # then we copy that to memory, adding biases
            input_connectors = {"from_kernel"}
            if add_bias is True: input_connectors.add("bias")
            tasklet = state.add_tasklet(
                "write_Y", input_connectors, {"to_memory"},
                "to_memory = from_kernel {}".format(
                    "+ bias" if add_bias is True else ""))
            state.add_memlet_path(vect_data,
                                  write_map_entry,
                                  tasklet,
                                  dst_conn="from_kernel",
                                  memlet=dace.Memlet("vec_data_Y[y1]"))

            if add_bias is True:
                state.add_memlet_path(B,
                                      entry_map,
                                      write_map_entry,
                                      tasklet,
                                      dst_conn="bias",
                                      memlet=dace.Memlet("B[n]"))

            state.add_memlet_path(tasklet,
                                  write_map_exit,
                                  exit_map,
                                  mem,
                                  src_conn="to_memory",
                                  memlet=dace.Memlet(
                                      "Y[b, n,x, y0*{}+y1]".format(vec_width)))
            # dace.Memlet("Y[b, 0:{}, 0:{}, 0:{}]".format(

        def make_compute(sdfg, state, vec_width=1):
            vec_type = dace.vector(dace.float32, vec_width)
            W_pipe_in = state.add_read("W_pipe")
            W_pipe_out = state.add_write("W_pipe")
            im2col_pipe_in = state.add_read("im2col_pipe")
            im2col_pipe_out = state.add_write("im2col_pipe")
            Y_pipe_in = state.add_read("Y_pipe")
            Y_pipe_out = state.add_write("Y_pipe")

            # batch_entry, batch_exit = state.add_map(
            #     "batch",  {"b": "0:{}".format(batch_size)},
            #     schedule=dace.ScheduleType.FPGA_Device)

            entry_n0, exit_n0 = state.add_map(
                "batch_n0", {
                    "b": "0:{}".format(batch_size),
                    "n0": "0:{}/{}".format(num_filters, P),
                },
                schedule=dace.ScheduleType.FPGA_Device)
            entry_k, exit_k = state.add_map(
                "k", {"k": "0:{}".format(K)},
                schedule=dace.ScheduleType.FPGA_Device)
            entry_w, exit_w = state.add_map(
                "buffer_W", {"n1": "0:{}".format(P)},
                schedule=dace.ScheduleType.FPGA_Device)

            # As we are using vectorized data types for im2col, we have to consider it into these
            # two maps
            entry_m, exit_m = state.add_map(
                "m", {"m": "0:{}/{}".format(M, vec_width)},
                schedule=dace.ScheduleType.FPGA_Device)
            entry_y, exit_y = state.add_map(
                "write_Y", {
                    "n1": "0:{}".format(P),
                    "m": "0:{}/{}".format(M, vec_width)
                },
                schedule=dace.ScheduleType.FPGA_Device)

            # Instantiate buffers
            sdfg.add_scalar("W_reg",
                            dtype=dace.float32,
                            transient=True,
                            storage=dace.dtypes.StorageType.FPGA_Registers)
            W_reg = state.add_write("W_reg")

            # For C result we are going to use vectorized data type
            sdfg.add_array("Y_buffer", [M / vec_width],
                           dtype=vec_type,
                           transient=True,
                           storage=dace.dtypes.StorageType.FPGA_Local)
            Y_buffer_in = state.add_read("Y_buffer")
            Y_buffer_out = state.add_write("Y_buffer")

            # every PE: reads input data, buffer the data assigned to it, forwards the data
            buffer_w_tasklet = state.add_tasklet(
                "buffer_w", {"w_in"}, {"w_reg", "w_out"}, """\
if n1 == {P} - p - 1:
    w_reg = w_in
if p < {P} - 1:
    w_out = w_in""".format(P=P))
            state.add_memlet_path(W_pipe_in,
                                  entry_n0,
                                  entry_k,
                                  entry_w,
                                  buffer_w_tasklet,
                                  memlet=dace.Memlet("W_pipe[p]",
                                                     dynamic=False),
                                  dst_conn="w_in")
            state.add_memlet_path(buffer_w_tasklet,
                                  exit_w,
                                  W_reg,
                                  memlet=dace.Memlet("W_reg[0]", dynamic=True),
                                  src_conn="w_reg")
            state.add_memlet_path(buffer_w_tasklet,
                                  exit_w,
                                  exit_k,
                                  exit_n0,
                                  W_pipe_out,
                                  memlet=dace.Memlet("W_pipe[p + 1]",
                                                     dynamic=True),
                                  src_conn="w_out")
            # Compute and forward B
            compute_tasklet = state.add_tasklet(
                "multiply_add", {"w_in", "im2col_in", "y_in"},
                {"im2col_out", "y_out"}, """\
y_prev = 0 if k == 0 else y_in
y_out = y_prev + w_in * im2col_in
if p < {P} - 1:
    im2col_out = im2col_in""".format(P=P))

            state.add_memlet_path(W_reg,
                                  entry_m,
                                  compute_tasklet,
                                  dst_conn="w_in",
                                  memlet=dace.Memlet("W_reg[0]"))
            state.add_memlet_path(im2col_pipe_in,
                                  entry_n0,
                                  entry_k,
                                  entry_m,
                                  compute_tasklet,
                                  memlet=dace.Memlet("im2col_pipe[p]",
                                                     dynamic=False),
                                  dst_conn="im2col_in")
            state.add_memlet_path(compute_tasklet,
                                  exit_m,
                                  exit_k,
                                  exit_n0,
                                  im2col_pipe_out,
                                  memlet=dace.Memlet("im2col_pipe[p + 1]",
                                                     dynamic=True),
                                  src_conn="im2col_out")
            state.add_memlet_path(Y_buffer_in,
                                  entry_k,
                                  entry_m,
                                  compute_tasklet,
                                  dst_conn="y_in",
                                  memlet=dace.Memlet("Y_buffer[m]"))
            state.add_memlet_path(entry_n0, Y_buffer_in, memlet=dace.Memlet())
            state.add_memlet_path(compute_tasklet,
                                  exit_m,
                                  exit_k,
                                  Y_buffer_out,
                                  src_conn="y_out",
                                  memlet=dace.Memlet("Y_buffer[m]"))
            state.add_memlet_path(Y_buffer_out, exit_n0, memlet=dace.Memlet())

            write_y_tasklet = state.add_tasklet(
                "write_y", {"buffer_in", "forward_in"}, {"y_out"}, """\
if n1 <= p:
    y_out = forward_in if p > 0 and n1 > 0 else buffer_in""")
            state.add_memlet_path(Y_buffer_out,
                                  entry_y,
                                  write_y_tasklet,
                                  memlet=dace.Memlet("Y_buffer[m]",
                                                     dynamic=True),
                                  dst_conn="buffer_in")
            state.add_memlet_path(Y_pipe_in,
                                  entry_n0,
                                  entry_y,
                                  write_y_tasklet,
                                  memlet=dace.Memlet("Y_pipe[p-1]",
                                                     dynamic=True),
                                  dst_conn="forward_in")
            state.add_memlet_path(write_y_tasklet,
                                  exit_y,
                                  exit_n0,
                                  Y_pipe_out,
                                  src_conn="y_out",
                                  memlet=dace.Memlet("Y_pipe[p]",
                                                     dynamic=True))

            # Unroll processing elements
            compute_entry, compute_exit = state.add_map(
                "unroll_compute", {"p": "0:{}".format(P)},
                schedule=dace.ScheduleType.FPGA_Device,
                unroll=True)

            # Bring data nodes into scope
            state.add_memlet_path(compute_entry,
                                  W_pipe_in,
                                  memlet=dace.memlet.Memlet())
            state.add_memlet_path(compute_entry,
                                  im2col_pipe_in,
                                  memlet=dace.memlet.Memlet())
            state.add_memlet_path(compute_entry,
                                  Y_pipe_in,
                                  memlet=dace.memlet.Memlet())
            state.add_memlet_path(W_pipe_out,
                                  compute_exit,
                                  memlet=dace.memlet.Memlet())
            state.add_memlet_path(im2col_pipe_out,
                                  compute_exit,
                                  memlet=dace.memlet.Memlet())
            state.add_memlet_path(Y_pipe_out,
                                  compute_exit,
                                  memlet=dace.memlet.Memlet())


        # build the compute State
        vec_type = dace.vector(dace.float32, vec_width)

        new_sdfg.add_stream("W_pipe",
                            dace.float32,
                            transient=True,
                            shape=(P + 1, ),
                            storage=dace.dtypes.StorageType.FPGA_Local,
                            buffer_size=str(P))
        new_sdfg.add_stream("im2col_pipe",
                            vec_type,
                            transient=True,
                            shape=(P + 1, ),
                            storage=dace.dtypes.StorageType.FPGA_Local)
        new_sdfg.add_stream("Y_pipe",
                            vec_type,
                            transient=True,
                            shape=(P + 1, ),
                            storage=dace.dtypes.StorageType.FPGA_Local)

        make_read_W(new_state)
        make_read_im2col(new_state, new_sdfg, vec_width)
        make_compute(new_sdfg, new_state, vec_width)
        make_write_Y(new_state, new_sdfg, vec_width, add_bias=(B is not None))

        new_sdfg.fill_scope_connectors()
        # Specialize the new sdfg, by using the input shapes
        new_sdfg.save("/tmp/conv.sdfg")
        new_sdfg.validate()
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

        # To create the shift register outside the map, add an empty memlet path
        shift_register_write = new_state.add_write("shift_register")
        shift_register_read = new_state.add_read("shift_register")
        new_state.add_memlet_path(shift_register_read,
                                  outer_me,
                                  inner_me,
                                  inner_mx,
                                  outer_mx,
                                  shift_register_write,
                                  memlet=dace.Memlet())

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
        #empty memlet
        new_state.add_memlet_path(outer_me, read_max_res, memlet=dace.Memlet())

        new_state.add_memlet_path(compute_tasklet,
                                  inner_mx,
                                  write_max_res,
                                  src_conn="max_out",
                                  memlet=dace.Memlet("max_res[0]"))
        #empty memlet
        new_state.add_memlet_path(write_max_res,
                                  outer_mx,
                                  memlet=dace.Memlet())

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


@autoregister_params(op="Gemm", name="fpga")
class FPGAGemm(ONNXForward):
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

        A = in_desc_with_name(node, state, sdfg, "A")
        B = in_desc_with_name(node, state, sdfg, "B")
        C = in_desc_with_name(node, state, sdfg, "C")
        Y = out_desc_with_name(node, state, sdfg, "Y")

        new_sdfg = dace.SDFG("fpga_gemm")
        new_state = new_sdfg.add_state("compute")
        new_sdfg.add_datadesc("A", copy.deepcopy(A))
        new_sdfg.add_datadesc("B", copy.deepcopy(B))
        new_sdfg.add_datadesc("C", copy.deepcopy(C))
        new_sdfg.add_datadesc("Y", copy.deepcopy(Y))
        new_sdfg.arrays["A"].transient = False
        new_sdfg.arrays["B"].transient = False
        new_sdfg.arrays["C"].transient = False
        new_sdfg.arrays["Y"].transient = False

        # GEMM Parameters

        N = A.shape[0]
        K = A.shape[1]
        M = C.shape[0]
        P = math.gcd(N, 16)  # Num PEs
        vec_width = math.gcd(M, 8)

        ####################################################
        # Build the SDFG: starting point: gemm_fpga_systolic vectorized sample

        def make_read_A(state):

            # TODO: vectorize also this, by reading more than one element at a time
            entry, exit = state.add_map("read_A", {
                "n0": "0:{}/{}".format(N, P),
                "k": "0:{}".format(K),
                "n1": "0:{}".format(P)
            },
                                        schedule=dace.ScheduleType.FPGA_Device)

            mem = state.add_read("A")
            pipe = state.add_write("A_pipe")
            tasklet = state.add_tasklet("read_A", {"from_memory"},
                                        {"to_kernel"},
                                        "to_kernel = from_memory")

            state.add_memlet_path(mem,
                                  entry,
                                  tasklet,
                                  dst_conn="from_memory",
                                  memlet=dace.Memlet(
                                      "A[n0 * {} + n1, k]".format(P)))
            state.add_memlet_path(tasklet,
                                  exit,
                                  pipe,
                                  src_conn="to_kernel",
                                  memlet=dace.Memlet("A_pipe[0]"))

        def make_read_B(state, sdfg, vec_width=1):

            # NOTE: We are reading this transposed: B is originally a matrix MxK

            # B is accessed by row
            # gear boxing: we read plain data types, we stream vector data types
            # Therefore we have two maps, the innermost is unrolled
            entry, exit = state.add_map("read_B", {
                "n": "0:{}/{}".format(N, P),
                "m": "0:{}".format(K),
                "k0": "0:{}/{}".format(M, vec_width)
            },
                                        schedule=dace.ScheduleType.FPGA_Device)

            read_map_entry, read_map_exit = state.add_map(
                "unrolled_reads_B", {"k1": "0:{}".format(vec_width)},
                schedule=dace.ScheduleType.FPGA_Device,
                unroll=True)

            # local storage to accumulate data
            sdfg.add_array('vec_data_B',
                           shape=[vec_width],
                           dtype=dace.float32,
                           transient=True,
                           storage=dace.dtypes.StorageType.FPGA_Registers)
            mem = state.add_read("B")
            pipe = state.add_write("B_pipe")
            vect_data = state.add_access("vec_data_B")
            tasklet = state.add_tasklet("read_B", {"from_memory"},
                                        {"to_kernel"},
                                        "to_kernel = from_memory")

            # In the innermost map we read W=vec_width data elements and we store them into `vec_data`
            state.add_memlet_path(mem,
                                  entry,
                                  read_map_entry,
                                  tasklet,
                                  dst_conn="from_memory",
                                  memlet=dace.Memlet(
                                      "B[k0*{}+k1, m]".format(vec_width)))

            state.add_memlet_path(tasklet,
                                  read_map_exit,
                                  vect_data,
                                  src_conn="to_kernel",
                                  memlet=dace.Memlet("vec_data_B[k1]"))

            # then we transfer them to the output stream
            copy_out_tasklet = state.add_tasklet('pack_and_copy_to_stream_B',
                                                 {'in_con'}, {'out_con'},
                                                 'out_con = in_con')
            state.add_memlet_path(vect_data,
                                  copy_out_tasklet,
                                  dst_conn="in_con",
                                  memlet=dace.Memlet("vec_data_B"))

            state.add_memlet_path(copy_out_tasklet,
                                  exit,
                                  pipe,
                                  src_conn="out_con",
                                  memlet=dace.Memlet("B_pipe[0]"))

        def make_write_C(state, sdfg, vec_width):

            # C data arrives as expressed in vect. data type. Needs to be unpacked
            # For doing so we first store it into a local buffer and then we write it in memory
            # as gear boxing works on local data only (not global memory)

            pipe = state.add_read("C_pipe")
            mem_read = state.add_read("C")
            mem = state.add_write("Y")

            entry_map, exit_map = state.add_map(
                "write_C", {
                    "n": "0:{}".format(N),
                    "m0": "0:{}/{}".format(M, vec_width)
                },
                schedule=dace.ScheduleType.FPGA_Device)

            write_map_entry, write_map_exit = state.add_map(
                "unrolled_write_C", {"m1": "0:{}".format(vec_width)},
                schedule=dace.ScheduleType.FPGA_Device,
                unroll=True)

            # local storage to accumulate data
            sdfg.add_array('vec_data_C',
                           shape=[vec_width],
                           dtype=dace.float32,
                           transient=True,
                           storage=dace.dtypes.StorageType.FPGA_Registers)

            vect_data = state.add_access("vec_data_C")

            # then we transfer them to the output stream
            copy_in_tasklet = state.add_tasklet('copy_from_stream_C',
                                                {'in_con'}, {'out_con'},
                                                'out_con = in_con')

            state.add_memlet_path(pipe,
                                  entry_map,
                                  copy_in_tasklet,
                                  dst_conn="in_con",
                                  memlet=dace.Memlet("C_pipe[{}-1]".format(P)))
            # this will trigger gear boxing
            state.add_memlet_path(copy_in_tasklet,
                                  vect_data,
                                  src_conn="out_con",
                                  memlet=dace.Memlet("vec_data_C"))

            # then we copy that to memory
            tasklet = state.add_tasklet("write_C", {"from_kernel", "prev_c"},
                                        {"to_memory"},
                                        "to_memory = from_kernel + prev_c")
            state.add_memlet_path(vect_data,
                                  write_map_entry,
                                  tasklet,
                                  dst_conn="from_kernel",
                                  memlet=dace.Memlet("vec_data_C[m1]"))
            # pay attention if C has a single dimension (could be the case of batch =1)
            state.add_memlet_path(mem_read,
                                  entry_map,
                                  write_map_entry,
                                  tasklet,
                                  dst_conn="prev_c",
                                  memlet=dace.Memlet("C[{}m0*{}+m1]".format(
                                      "n, " if len(C.shape) == 2 else "",
                                      vec_width)))

            state.add_memlet_path(tasklet,
                                  write_map_exit,
                                  exit_map,
                                  mem,
                                  src_conn="to_memory",
                                  memlet=dace.Memlet(
                                      "Y[n, m0*{}+m1]".format(vec_width)))

        def make_compute(sdfg, state, vec_width=1):

            vec_type = dace.vector(dace.float32, vec_width)
            A_pipe_in = state.add_read("A_pipe")
            A_pipe_out = state.add_write("A_pipe")
            B_pipe_in = state.add_read("B_pipe")
            B_pipe_out = state.add_write("B_pipe")
            C_pipe_in = state.add_read("C_pipe")
            C_pipe_out = state.add_write("C_pipe")

            entry_n0, exit_n0 = state.add_map(
                "n0", {
                    "n0": "0:{}/{}".format(N, P),
                },
                schedule=dace.ScheduleType.FPGA_Device)
            entry_k, exit_k = state.add_map(
                "k", {"k": "0:{}".format(K)},
                schedule=dace.ScheduleType.FPGA_Device)
            entry_a, exit_a = state.add_map(
                "buffer_A", {"n1": "0:{}".format(P)},
                schedule=dace.ScheduleType.FPGA_Device)

            # As we are using vectorized data types for B, we have to consider it into these
            # two maps
            entry_m, exit_m = state.add_map(
                "m", {"m": "0:{}/{}".format(M, vec_width)},
                schedule=dace.ScheduleType.FPGA_Device)
            entry_c, exit_c = state.add_map(
                "write_C", {
                    "n1": "0:{}".format(P),
                    "m": "0:{}/{}".format(M, vec_width)
                },
                schedule=dace.ScheduleType.FPGA_Device)

            # Instantiate buffers
            sdfg.add_scalar("A_reg",
                            dtype=dace.float32,
                            transient=True,
                            storage=dace.dtypes.StorageType.FPGA_Registers)
            A_reg = state.add_write("A_reg")

            # For C result we are going to use vectorized data type
            sdfg.add_array("C_buffer", [M / vec_width],
                           dtype=vec_type,
                           transient=True,
                           storage=dace.dtypes.StorageType.FPGA_Local)
            C_buffer_in = state.add_read("C_buffer")
            C_buffer_out = state.add_write("C_buffer")

            # every PE: reads input data, buffer the data assigned to it, forwards the data
            buffer_a_tasklet = state.add_tasklet(
                "buffer_a", {"a_in"}, {"a_reg", "a_out"}, """\
if n1 == {P} - p - 1:
    a_reg = a_in
if p < {P} - 1:
    a_out = a_in""".format(P=P))
            state.add_memlet_path(A_pipe_in,
                                  entry_n0,
                                  entry_k,
                                  entry_a,
                                  buffer_a_tasklet,
                                  memlet=dace.Memlet("A_pipe[p]",
                                                     dynamic=False),
                                  dst_conn="a_in")
            state.add_memlet_path(buffer_a_tasklet,
                                  exit_a,
                                  A_reg,
                                  memlet=dace.Memlet("A_reg[0]", dynamic=True),
                                  src_conn="a_reg")
            state.add_memlet_path(buffer_a_tasklet,
                                  exit_a,
                                  exit_k,
                                  exit_n0,
                                  A_pipe_out,
                                  memlet=dace.Memlet("A_pipe[p + 1]",
                                                     dynamic=True),
                                  src_conn="a_out")
            # Compute and forward B
            compute_tasklet = state.add_tasklet(
                "multiply_add", {"a_in", "b_in", "c_in"}, {"b_out", "c_out"},
                """\
c_prev = 0 if k == 0 else c_in
c_out = c_prev + a_in * b_in
if p < {P} - 1:
    b_out = b_in""".format(P=P))

            state.add_memlet_path(A_reg,
                                  entry_m,
                                  compute_tasklet,
                                  dst_conn="a_in",
                                  memlet=dace.Memlet("A_reg[0]"))
            state.add_memlet_path(B_pipe_in,
                                  entry_n0,
                                  entry_k,
                                  entry_m,
                                  compute_tasklet,
                                  memlet=dace.Memlet("B_pipe[p]",
                                                     dynamic=False),
                                  dst_conn="b_in")
            state.add_memlet_path(compute_tasklet,
                                  exit_m,
                                  exit_k,
                                  exit_n0,
                                  B_pipe_out,
                                  memlet=dace.Memlet("B_pipe[p + 1]",
                                                     dynamic=True),
                                  src_conn="b_out")
            state.add_memlet_path(C_buffer_in,
                                  entry_k,
                                  entry_m,
                                  compute_tasklet,
                                  dst_conn="c_in",
                                  memlet=dace.Memlet("C_buffer[m]"))
            state.add_memlet_path(entry_n0, C_buffer_in, memlet=dace.Memlet())
            state.add_memlet_path(compute_tasklet,
                                  exit_m,
                                  exit_k,
                                  C_buffer_out,
                                  memlet=dace.Memlet("C_buffer[m]"),
                                  src_conn="c_out")
            state.add_memlet_path(C_buffer_out, exit_n0, memlet=dace.Memlet())

            write_c_tasklet = state.add_tasklet(
                "write_c", {"buffer_in", "forward_in"}, {"c_out"}, """\
if n1 <= p:
    c_out = forward_in if p > 0 and n1 > 0 else buffer_in""")
            state.add_memlet_path(C_buffer_out,
                                  entry_c,
                                  write_c_tasklet,
                                  memlet=dace.Memlet("C_buffer[m]",
                                                     dynamic=True),
                                  dst_conn="buffer_in")
            state.add_memlet_path(C_pipe_in,
                                  entry_n0,
                                  entry_c,
                                  write_c_tasklet,
                                  memlet=dace.Memlet("C_pipe[p-1]",
                                                     dynamic=True),
                                  dst_conn="forward_in")
            state.add_memlet_path(write_c_tasklet,
                                  exit_c,
                                  exit_n0,
                                  C_pipe_out,
                                  memlet=dace.Memlet("C_pipe[p]",
                                                     dynamic=True),
                                  src_conn="c_out")

            # Unroll processing elements
            compute_entry, compute_exit = state.add_map(
                "unroll_compute", {"p": "0:{}".format(P)},
                schedule=dace.ScheduleType.FPGA_Device,
                unroll=True)

            # Bring data nodes into scope
            state.add_memlet_path(compute_entry,
                                  A_pipe_in,
                                  memlet=dace.memlet.Memlet())
            state.add_memlet_path(compute_entry,
                                  B_pipe_in,
                                  memlet=dace.memlet.Memlet())
            state.add_memlet_path(compute_entry,
                                  C_pipe_in,
                                  memlet=dace.memlet.Memlet())
            state.add_memlet_path(A_pipe_out,
                                  compute_exit,
                                  memlet=dace.memlet.Memlet())
            state.add_memlet_path(B_pipe_out,
                                  compute_exit,
                                  memlet=dace.memlet.Memlet())
            state.add_memlet_path(C_pipe_out,
                                  compute_exit,
                                  memlet=dace.memlet.Memlet())

        # build the compute State
        vec_type = dace.vector(dace.float32, vec_width)

        new_sdfg.add_stream("A_pipe",
                            dace.float32,
                            transient=True,
                            shape=(P + 1, ),
                            storage=dace.dtypes.StorageType.FPGA_Local,
                            buffer_size=str(P))
        new_sdfg.add_stream("B_pipe",
                            vec_type,
                            transient=True,
                            shape=(P + 1, ),
                            storage=dace.dtypes.StorageType.FPGA_Local)
        new_sdfg.add_stream("C_pipe",
                            vec_type,
                            transient=True,
                            shape=(P + 1, ),
                            storage=dace.dtypes.StorageType.FPGA_Local)

        make_read_A(new_state)
        make_read_B(new_state, new_sdfg, vec_width)
        make_compute(new_sdfg, new_state, vec_width)
        make_write_C(new_state, new_sdfg, vec_width)

        new_sdfg.fill_scope_connectors()
        # Specialize the new sdfg, by using the input shapes
        new_sdfg.save("/tmp/gemm.sdfg")
        new_sdfg.validate()
        return new_sdfg
