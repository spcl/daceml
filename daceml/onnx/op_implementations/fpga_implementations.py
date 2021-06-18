import copy
import typing

import dace
from dace import SDFGState, SDFG, dtypes
from dace.sdfg import nodes
from dace.sdfg.nodes import Node
from dace.symbolic import symstr

from daceml.onnx.nodes.onnx_op import ONNXOp
from daceml.onnx.forward_implementation_abc import ONNXForward
import numpy as np
import math

from daceml.util.utils import in_desc_with_name, out_desc_with_name, in_edge_with_name
from daceml.transformation import constant_folding
from daceml.onnx.op_implementations.utils import op_implementation, program_for_node


def _2d_sliding_window_index_expr(x_or_y, stride, kernel_size):
    return f"out_{x_or_y} * {stride} + h{x_or_y}"


def search_fpga_name_in_weights(fpga_name: str, sdfg: SDFG) -> list:
    '''
    Searches among the model weights, and returns a list comprising weights W such that
    W is a substring of the given fpga_name.
    Can be used to relate containers name after FPGA Transform
    :param fpga_name:
    :param sdfg: the sdfg to search into
    :return: a list with all the occurences
    '''
    found = []
    for k in sdfg._parent_onnx_model.clean_weights:
        # After transforming for FPGA, containers have `_in`/`_out` as prefix
        if k + "_" in fpga_name:
            found.append(k)
    return found


@op_implementation(op="Conv", name="naive_fpga")
class FPGAConv2D(ONNXForward):
    """
    The "trivial" convolution implementation, i.e. two nested maps.
    It may not synthesize to hardware, due to high resource consumption
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
            dict(m=f"0:{num_filters}",
                 cin=f"0:{num_channels}",
                 hx=f"0:{filter_hx}",
                 hy=f"0:{filter_hy}"))
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

        # Here we want to increase reuse of the input feature, that is read the input once and update all the
        # m output channels. Therefore we interchange some of maps indices.
        # - the outer map loops over every entry in the output array, not considering the channel (Y[b,:,x,y])
        # - a mid map over the input channels (this is splitted from the inner map just to have more control on unrolling)
        # - the inner computes the value for all the entries of a given point

        # the outer map loops over every entry in the output array
        outer_me, outer_mx = new_state.add_map(
            'outer_conv_map',
            dict(b=f"0:{batch_size}",
                 out_x=f"0:{output_size_x}",
                 out_y=f"0:{output_size_y}"))

        mid_me, mid_mx = new_state.add_map('mid_conv_map',
                                           dict(cin=f"0:{num_channels}"))

        # the inner map computes the value for a single entry in the output array (i.e. Y[b, m, x, y])
        inner_me, inner_mx = new_state.add_map('inner_conv_map',
                                               dict(m=f"0:{num_filters}",
                                                    hx=f"0:{filter_hx}",
                                                    hy=f"0:{filter_hy}"),
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
            code=f"""\
if m==0: local_X_in = image_in
local_Y_out = (0 if hx == 0 and hy==0 and cin==0 else local_Y_in)  + local_X_in * filter_in
local_X_out = local_X_in
if hx == {filter_hx}-1 and hy == {filter_hy}-1 and cin=={num_channels}-1: 
    output = local_Y_out {'+ B_in' if B is not None else ''}""")

        filter_memlet = dace.Memlet("local_W[m, cin, hx, hy]")

        x_idx = _2d_sliding_window_index_expr(x_or_y="x",
                                              stride=stride_x,
                                              kernel_size=filter_hx)
        y_idx = _2d_sliding_window_index_expr(x_or_y="y",
                                              stride=stride_y,
                                              kernel_size=filter_hy)

        image_memlet = dace.Memlet(f"X[b, cin, {x_idx}, {y_idx}]")
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

        new_state.add_memlet_path(local_W_access,
                                  outer_me,
                                  mid_me,
                                  inner_me,
                                  compute_tasklet,
                                  dst_conn='filter_in',
                                  memlet=filter_memlet)

        # hook up X: this goes directly to the tasklet
        read_X = new_state.add_read("X")

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
        return new_sdfg


@op_implementation(op="Conv", name="fpga")
class FPGAIm2ColConv(ONNXForward):
    """
        Im2Col implementation of Convolution.
        Underneath it applies a Matrix Matrix Multiplication
    """
    @staticmethod
    def forward_can_be_applied(node: ONNXOp, state: SDFGState,
                               sdfg: SDFG) -> bool:
        X = in_desc_with_name(node, state, sdfg, "X")
        W = in_desc_with_name(node, state, sdfg, "W")
        Y = out_desc_with_name(node, state, sdfg, "Y")

        try:
            B = in_desc_with_name(node, state, sdfg, "B")
        except Exception as e:
            B = None

        image_dims = len(X.shape) - 2
        num_filters = W.shape[0]
        num_channels = X.shape[1]

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

        # Support all same padding
        if node.pads is not None and (not all(p == node.pads[0]
                                              for p in node.pads)
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

        # TODO
        #  - The current implementation support vectorization on Y only. Support vectorization also for X
        #  - for the weights, we may want vectorization as well (but this may cut out some transformation such
        #   as InputToConstant), or, in any case, we want to be more memory-friendly by reading burst of data
        #   since it is accessed as a transposed matrix

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

        # Padding
        padding = node.pads[0]  # currently only equal padding supported
        offset = 2 * (filter_hx // 2 - padding)

        # Take output size: note, tat this accounts for vectorization (if present)
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
        vec_width = Y.veclen
        x_base_type = X.dtype.base_type

        K = num_channels * filter_hx * filter_hy
        M = output_size_y * output_size_x
        P = num_filters  # Num PEs  #TODO parametric

        # safe delay: see explanation in the make_compute function
        L = max(11 - M, 0)

        # TODO: add correctness check, see MatMul expansion

        def make_read_W(state):
            # this will read the weights, organized as a matrix of size
            # num_filters x (num_channels * filter_hx * filter_hy)
            # The original weight matrix has shape [num_filters, num_channels, filter_hx, filter_hy]

            entry, exit = state.add_map(
                "read_weights",
                {
                    "b":
                    f"0:{batch_size}",  # the batch map loops over every image in the batch
                    "n0": f"0:{num_filters}/{P}",
                    "cin": f"0:{num_channels}",
                    "hx": f"0:{filter_hx}",
                    "hy": f"0:{filter_hy}"
                },
                schedule=dace.ScheduleType.FPGA_Device)

            # use a different map, and unroll it if necessary (otherwise reading weights will slow down everythin)
            unroll_inner_map = P > (M + L) and P <= 16
            send_map_entry, send_map_exit = state.add_map(
                "send_weights", {"n1": f"0:{P}"},
                schedule=dace.ScheduleType.FPGA_Device,
                unroll=unroll_inner_map)

            mem = state.add_read("W")
            pipe = state.add_write("W_pipe")
            tasklet = state.add_tasklet("read_W", {"from_memory"},
                                        {"to_kernel"},
                                        "to_kernel = from_memory")

            state.add_memlet_path(
                mem,
                entry,
                send_map_entry,
                tasklet,
                dst_conn="from_memory",
                memlet=dace.Memlet(f"W[n0 * {P} + n1, cin, hx, hy]"))
            state.add_memlet_path(tasklet,
                                  send_map_exit,
                                  exit,
                                  pipe,
                                  src_conn="to_kernel",
                                  memlet=dace.Memlet(f"W_pipe[{P} -n1 -1]"))

        def make_read_im2col(state, sdfg, vec_width=1):

            # Matrix B will be the im2col matrix. We will build it row-by-row
            # to facilitate streaming in the systolic MMM, avoiding storing it back to memory
            # Note: this will require to load multiple times the input feature, yet this save I/Os
            # The im2col matrix has size (num_channels * filter_hx * filter_hy) x (output_size_y * output_size_x)

            # gear boxing: we read plain data types, we stream vector data types
            # Therefore we have two maps, the innermost is unrolled
            im2col_me, im2col_mx = state.add_map(
                "im2col_map",
                {
                    "b": f"0:{batch_size}",
                    "n":
                    f"0:{num_filters}/{P}",  # repeat B for computing the result
                    "cin": f"0:{num_channels}",
                    "hx": f"0:{filter_hx}",
                    "hy": f"0:{filter_hy}",
                    "x": f"0:{output_size_x}",
                    "y0": f"0:{output_size_y}",
                },
                schedule=dace.ScheduleType.FPGA_Device)

            read_map_entry, read_map_exit = state.add_map(
                "unrolled_reads_X", {"y1": f"0:{vec_width}"},
                schedule=dace.ScheduleType.FPGA_Device,
                unroll=True)

            # local storage to accumulate data
            sdfg.add_array('vec_data_im2col',
                           shape=[vec_width],
                           dtype=x_base_type,
                           transient=True,
                           storage=dace.dtypes.StorageType.FPGA_Registers)

            X = state.add_read("X")
            pipe = state.add_write("im2col_pipe")
            vect_data = state.add_access("vec_data_im2col")
            tasklet = state.add_tasklet(
                "read_X", {"from_memory"}, {"to_kernel"}, f"""
if ((x + hx - {padding} < {output_size_x} + {offset}) and 
        (x + hx  - {padding} >= 0) and 
        (y0*{vec_width}+y1 + hy  - {padding} < {output_size_y} * {vec_width} + {offset})  and
        (y0*{vec_width}+y1 + hy  - {padding} >= 0)):
    to_kernel = from_memory
else:
    to_kernel = 0
""")

            im2col_input_memlet = dace.Memlet(
                f"X[b, cin, x + hx - {padding}, y0*{vec_width}+y1 + hy - {padding}]",
                allow_oob=True,
                dynamic=True)

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

            # Y data arrives as expressed in vect. data type. Needs to be unpacked
            # For doing so we first store it into a local buffer and then we write it in memory
            # as gear boxing works on local data only (not global memory)

            pipe = state.add_read("Y_pipe")
            mem = state.add_write("Y")
            if add_bias is True:
                B = state.add_read("B")
            entry_map, exit_map = state.add_map(
                "write_Y", {
                    "b": f"0:{batch_size}",
                    "n": f"0:{num_filters}",
                    "x": f"0:{output_size_x}",
                    "y": f"0:{output_size_y}"
                },
                schedule=dace.ScheduleType.FPGA_Device)

            # TODO: Xilinx: do we need to unroll bias addition?

            input_connectors = {"in_con"}
            if add_bias is True: input_connectors.add("bias")
            copy__add_bias__tasklet = state.add_tasklet(
                "copy_from_stream_Y", input_connectors, {"out_con"},
                f"out_con = in_con {'+ bias' if add_bias is True else ''}")

            state.add_memlet_path(pipe,
                                  entry_map,
                                  copy__add_bias__tasklet,
                                  dst_conn="in_con",
                                  memlet=dace.Memlet(f"Y_pipe[{P}-1]"))

            if add_bias is True:
                state.add_memlet_path(B,
                                      entry_map,
                                      copy__add_bias__tasklet,
                                      dst_conn="bias",
                                      memlet=dace.Memlet("B[n]"))

            # Memlet to memory

            state.add_memlet_path(copy__add_bias__tasklet,
                                  exit_map,
                                  mem,
                                  src_conn="out_con",
                                  memlet=dace.Memlet("Y[b, n, x, y]"))

        def make_compute(sdfg, state, vec_width=1):
            vec_type = dace.vector(x_base_type, vec_width)
            W_pipe_in = state.add_read("W_pipe")
            im2col_pipe_in = state.add_read("im2col_pipe")
            im2col_pipe_out = state.add_write("im2col_pipe")
            Y_pipe_in = state.add_read("Y_pipe")
            Y_pipe_out = state.add_write("Y_pipe")

            # Create a single pipeline with all the flattened loops

            entry_pipeline, exit_pipeline = state.add_pipeline(
                "compute_and_drain",
                {
                    "b": f"0:{batch_size}",
                    "n0": f"0:{num_filters}/{P}",
                    "k": f"0:{K}",
                    "m": f"0:{M} + {L}"
                    # The + L is a safe delay between computing and drain. It must be computed by
                    #considering the latency for updating the same result (not just the FP32 multiply add, but
                    # also for reading/writing
                },
                drain_size=P * M,
                drain_overlap=False,
                additional_iterators={
                    'm_drain': 0,
                    'k_drain': 0
                },
                schedule=dace.ScheduleType.FPGA_Device)

            # Instantiate buffers
            sdfg.add_scalar("W_reg",
                            dtype=W.dtype.base_type,
                            transient=True,
                            storage=dace.dtypes.StorageType.FPGA_Registers)
            W_reg_init = state.add_access("W_reg")
            W_reg = state.add_access("W_reg")

            # For C result we are going to use vectorized data type
            sdfg.add_array(
                "Y_buffer",
                [M],  #M already accounts for vec width
                dtype=vec_type,
                transient=True,
                storage=dace.dtypes.StorageType.FPGA_Local)
            Y_buffer_in = state.add_read("Y_buffer")
            Y_buffer_out = state.add_access("Y_buffer")

            # Buffering of im2col data (B)
            sdfg.add_array("im2col_reg",
                           shape=[1],
                           dtype=vec_type,
                           transient=True,
                           storage=dace.dtypes.StorageType.FPGA_Local)
            im2col_reg = state.add_access("im2col_reg")

            # every PE: reads input data, buffer the data assigned to it
            buffer_w_tasklet = state.add_tasklet(
                "buffer_w", {"w_in"}, {"w_reg"}, f"""\
if m == 0 and not {entry_pipeline.pipeline.drain_condition()}:
    w_reg = w_in""")
            state.add_memlet_path(W_pipe_in,
                                  entry_pipeline,
                                  buffer_w_tasklet,
                                  memlet=dace.Memlet("W_pipe[p]",
                                                     dynamic=True),
                                  dst_conn="w_in")
            state.add_memlet_path(buffer_w_tasklet,
                                  W_reg,
                                  memlet=dace.Memlet("W_reg[0]", dynamic=True),
                                  src_conn="w_reg")

            # FEED B (im2col matrix)
            # Read B: done outside of the compute tasklet to help type inference

            buffer_im2col_tasklet = state.add_tasklet(
                "buffer_im2col", {"im2col_in"}, {"im2col_reg_out"}, f"""\
if  m>={L} and not {entry_pipeline.pipeline.drain_condition()}:
    im2col_reg_out = im2col_in""")

            state.add_memlet_path(im2col_pipe_in,
                                  entry_pipeline,
                                  buffer_im2col_tasklet,
                                  memlet=dace.Memlet("im2col_pipe[p]",
                                                     dynamic=True),
                                  dst_conn="im2col_in")
            state.add_memlet_path(buffer_im2col_tasklet,
                                  im2col_reg,
                                  memlet=dace.Memlet("im2col_reg[0]",
                                                     dynamic=True),
                                  src_conn="im2col_reg_out")

            # COMPUTE AND DRAIN
            # Compute and forward B: this is done if we are not in the init phase of the pipeline
            compute_tasklet = state.add_tasklet(
                "compute_and_drain",
                {"w_in", "im2col_in", "y_in", "forward_in"},
                {"im2col_out", "y_out", "y_pipe_out"}, f"""\
if m>= {L} and not {entry_pipeline.pipeline.drain_condition()}:
    y_prev = 0 if k == 0 else y_in     
    y_out =  y_prev + w_in * im2col_in
    if p < {P} - 1:
        im2col_out = im2col_in
# Drain
# when we have to drain:
# - if k = K-1 and m>=L: drain my own result
#-  otherwise, if k_drain<p forward data coming from previous PEs (this could happens also in the drain phase)
if ((b>0  or n0 > 0)  and k_drain <p and m_drain <{M}) or  (k=={K}-1 and m>= {L}) or ({entry_pipeline.pipeline.drain_condition()} and k_drain < p):
    # if p!=0 and (k_drain != {K}-1 or {entry_pipeline.pipeline.drain_condition()}):
    #     tmp = forward_in
    # y_pipe_out = tmp
    y_pipe_out = y_out if (p==0 or (k_drain=={K}-1 and not {entry_pipeline.pipeline.drain_condition()})) else forward_in

# adjust draining iterators
if not {entry_pipeline.pipeline.drain_condition()}:
    if m_drain >= {L} +  {M} -1:
        m_drain = 0
        if k_drain >= {K} - 1:
            k_drain = 0
        else:
            k_drain = k_drain +1
    else:
        m_drain = m_drain + 1
else:
    if m_drain >=  {M} -1:
        m_drain = 0
        if k_drain >= {K} - 1:
            k_drain = 0
        else:
            k_drain = k_drain +1
    else:
        m_drain = m_drain + 1
""")

            state.add_memlet_path(W_reg,
                                  compute_tasklet,
                                  dst_conn="w_in",
                                  memlet=dace.Memlet("W_reg[0]"))
            state.add_memlet_path(im2col_reg,
                                  compute_tasklet,
                                  memlet=dace.Memlet("im2col_reg[0]",
                                                     dynamic=False),
                                  dst_conn="im2col_in")
            state.add_memlet_path(compute_tasklet,
                                  exit_pipeline,
                                  im2col_pipe_out,
                                  memlet=dace.Memlet("im2col_pipe[p + 1]",
                                                     dynamic=True),
                                  src_conn="im2col_out")
            state.add_memlet_path(Y_buffer_in,
                                  entry_pipeline,
                                  compute_tasklet,
                                  dst_conn="y_in",
                                  memlet=dace.Memlet(f"Y_buffer[m-{L}]",
                                                     allow_oob=True))
            state.add_memlet_path(compute_tasklet,
                                  exit_pipeline,
                                  Y_buffer_out,
                                  src_conn="y_out",
                                  memlet=dace.Memlet(f"Y_buffer[m-{L}]",
                                                     allow_oob=True,
                                                     dynamic=True))

            state.add_memlet_path(Y_pipe_in,
                                  entry_pipeline,
                                  compute_tasklet,
                                  memlet=dace.Memlet("Y_pipe[p-1]",
                                                     dynamic=True),
                                  dst_conn="forward_in")
            state.add_memlet_path(compute_tasklet,
                                  exit_pipeline,
                                  Y_pipe_out,
                                  memlet=dace.Memlet("Y_pipe[p]",
                                                     dynamic=True),
                                  src_conn="y_pipe_out")

            # Unroll processing elements
            compute_entry, compute_exit = state.add_map(
                "unroll_compute", {"p": f"0:{P}"},
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

            state.add_memlet_path(im2col_pipe_out,
                                  compute_exit,
                                  memlet=dace.memlet.Memlet())
            state.add_memlet_path(Y_pipe_out,
                                  compute_exit,
                                  memlet=dace.memlet.Memlet())
            state.add_memlet_path(compute_entry,
                                  W_reg_init,
                                  memlet=dace.memlet.Memlet())
            state.add_memlet_path(W_reg_init,
                                  entry_pipeline,
                                  memlet=dace.memlet.Memlet())
            im2col_init = state.add_access("im2col_reg")
            state.add_memlet_path(compute_entry,
                                  im2col_init,
                                  memlet=dace.Memlet())
            state.add_memlet_path(im2col_init,
                                  entry_pipeline,
                                  memlet=dace.Memlet())
            state.add_memlet_path(compute_entry,
                                  Y_buffer_in,
                                  memlet=dace.Memlet())
            state.add_memlet_path(Y_buffer_out,
                                  compute_exit,
                                  memlet=dace.Memlet())

        # build the compute State
        vec_type = dace.vector(x_base_type, vec_width)

        new_sdfg.add_stream("W_pipe",
                            W.dtype.base_type,
                            transient=True,
                            shape=(P, ),
                            storage=dace.dtypes.StorageType.FPGA_Local,
                            buffer_size=str(P))
        new_sdfg.add_stream("im2col_pipe",
                            vec_type,
                            transient=True,
                            shape=(P + 1, ),
                            buffer_size=2,
                            storage=dace.dtypes.StorageType.FPGA_Local)
        new_sdfg.add_stream("Y_pipe",
                            vec_type,
                            transient=True,
                            shape=(P + 1, ),
                            buffer_size=M,
                            storage=dace.dtypes.StorageType.FPGA_Local)

        make_read_W(new_state)
        make_read_im2col(new_state, new_sdfg, vec_width)
        make_compute(new_sdfg, new_state, vec_width)
        make_write_Y(new_state, new_sdfg, vec_width, add_bias=(B is not None))

        new_sdfg.fill_scope_connectors()
        return new_sdfg


@op_implementation(op="Relu", name="fpga")
class FPGARelu(ONNXForward):
    @staticmethod
    def forward(node: ONNXOp, state: SDFGState,
                sdfg: SDFG) -> typing.Union[Node, SDFG]:

        X = in_desc_with_name(node, state, sdfg, "X")
        Y = out_desc_with_name(node, state, sdfg, "Y")

        vec_width = X.veclen

        # Handle the case in which the vectorization width used for the input is different from
        # the one used for the output
        if X.veclen != Y.veclen:
            # NOTE: for the moment being, tested with Y veclen = 1
            vec_width_mismatch = True
        else:
            vec_width_mismatch = False

        # Build map ranges: one loop per dimension
        map_ranges = {'__i%d' % i: '0:%s' % n for i, n in enumerate(X.shape)}

        new_sdfg = dace.SDFG("fpga_relu")

        new_state = new_sdfg.add_state("compute")
        new_sdfg.add_datadesc("X", copy.deepcopy(X))
        new_sdfg.add_datadesc("Y", copy.deepcopy(Y))

        new_sdfg.arrays["X"].transient = False
        new_sdfg.arrays["Y"].transient = False
        outer_me, outer_mx = new_state.add_map('relu_map', map_ranges)

        new_sdfg.add_array("vec_data_in", [vec_width],
                           dtype=X.dtype.base_type,
                           transient=True,
                           storage=dace.dtypes.StorageType.FPGA_Registers)
        new_sdfg.add_array("vec_data_out", [1],
                           dtype=X.dtype.base_type,
                           transient=True,
                           storage=dace.dtypes.StorageType.FPGA_Registers)

        vec_data_in = new_state.add_access("vec_data_in")
        vec_data_out = new_state.add_access("vec_data_in")

        # Unrolled map to compute the elementwise max
        inner_me, inner_mx = new_state.add_map('inner_relu_map',
                                               dict(i=f"0:{vec_width}"),
                                               unroll=True)

        tasklet = new_state.add_tasklet('relu_task', ['x_con'], ['y_con'],
                                        'y_con = max(0.0, x_con)')
        x_read = new_state.add_read("X")
        y_write = new_state.add_write("Y")

        #unpack vector data
        #memlet from memory
        memlet_idx = f"{','.join(['__i%d' % i for i in range(len(X.shape))])}"
        new_state.add_memlet_path(x_read,
                                  outer_me,
                                  vec_data_in,
                                  memlet=dace.Memlet(f"X[{memlet_idx}]"))

        # connect to tasklet
        new_state.add_memlet_path(vec_data_in,
                                  inner_me,
                                  tasklet,
                                  dst_conn='x_con',
                                  memlet=dace.Memlet("vec_data_in[i]"))

        # pack
        new_state.add_memlet_path(tasklet,
                                  inner_mx,
                                  vec_data_out,
                                  src_conn='y_con',
                                  memlet=dace.Memlet("vec_data_in[i]"))

        # if there is a mismatch between input and output veclen (e.g. GEMM->Relu in Lenet)
        # we need an extra loop here

        if vec_width_mismatch:
            #TODO: right now this handle the case Y.veclen==1
            assert (Y.veclen == 1)
            write_out_me, write_out_mx = new_state.add_map(
                'relu_write_out_map', dict(i=f"0:{vec_width}"), unroll=True)
            tasklet = new_state.add_tasklet('read_tasklet', ['_in'], ['_out'],
                                            code="_out = _in")
            # write out
            new_state.add_memlet_path(vec_data_out,
                                      write_out_me,
                                      tasklet,
                                      dst_conn="_in",
                                      memlet=dace.Memlet("vec_data_in[i]"))
            # TODO: special case for GEMM->Relu, do the right memlet
            new_state.add_memlet_path(
                tasklet,
                write_out_mx,
                outer_mx,
                y_write,
                src_conn="_out",
                memlet=dace.Memlet(f"Y[__i0, __i1*{vec_width}+i]"))

        else:
            #write out
            memlet_idx = f"{','.join(['__i%d' % i for i in range(len(X.shape))])}"
            new_state.add_memlet_path(vec_data_out,
                                      outer_mx,
                                      y_write,
                                      memlet=dace.Memlet(f"Y[{memlet_idx}]"))
        new_sdfg.fill_scope_connectors()
        return new_sdfg


@op_implementation(op="MaxPool", name="fpga")
class FPGAMaxPool2D(ONNXForward):
    @staticmethod
    def forward_can_be_applied(node: ONNXOp, state: SDFGState,
                               sdfg: SDFG) -> bool:
        X = in_desc_with_name(node, state, sdfg, "X")
        Y = out_desc_with_name(node, state, sdfg, "Y")

        if Y.veclen != 1:  # if output vectorized must match
            _, filter_width = node.kernel_shape
            if not (X.veclen == Y.veclen or X.veclen == filter_width * Y.veclen
                    ):  # support reducing vector size proportionally to filter
                return False

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

        # Max Pool: the current implementation exploit a sliding window. Considering a single batch and a single
        # channel, we will read one input element at a time, shifting

        # TODO: this implementation depends on how data will be streamed
        #  for the moment being we assume it sends one channel after the other
        # TODO: support Xilinx

        X = in_desc_with_name(node, state, sdfg, "X")
        Y = out_desc_with_name(node, state, sdfg, "Y")
        vec_width = X.veclen
        out_vec_width = Y.veclen

        # Xilinx implementation specific
        vendor = dace.config.Config.get("compiler", "fpga_vendor")
        xilinx = True if vendor == "xilinx" else False

        # use registers for now, careful with large images
        # the buffer covers approximately an entire row of the image
        xilinx_buffer_type = dace.StorageType.FPGA_Registers

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

        #shift register. Note that this contains plain data types
        shift_register_size = input_size_width * vec_width * (
            filter_height - 1) + (filter_width - 1) + 1

        buffer_type = xilinx_buffer_type if xilinx else dace.StorageType.FPGA_ShiftRegister
        new_sdfg.add_array("shift_register", [shift_register_size],
                           X.dtype.type,
                           storage=buffer_type,
                           transient=True)
        # variable for reduction
        new_sdfg.add_array("max_res", [1],
                           X.dtype.type,
                           storage=dace.StorageType.FPGA_Registers,
                           transient=True)
        new_sdfg.add_array('vec_data',
                           shape=[
                               vec_width,
                           ],
                           dtype=X.dtype.type,
                           transient=True,
                           storage=dace.dtypes.StorageType.FPGA_Registers)
        # temporary storage for unpacked vector data type

        # Buffer for vectorized output
        if Y.veclen > 1:
            new_sdfg.add_array("vec_data_out",
                               shape=[
                                   out_vec_width,
                               ],
                               dtype=Y.dtype.type,
                               transient=True,
                               storage=dace.dtypes.StorageType.FPGA_Registers)
            # temporary storage for unpacked output vector
            vec_out = new_state.add_access("vec_data_out")

        # the outer map loops over every entry in the input array
        # (useful also in the case of streaming input, we can't skip data
        # Note that `input_size_width` accounts for vectorization
        outer_me, outer_mx = new_state.add_map(
            'outer_pool_map',
            dict(b=f"0:{batch_size}",
                 c=f"0:{num_channels}",
                 in_y=f"0:{input_size_height}",
                 in_x=f"0:{input_size_width}"))

        # if vec_width >1 this will deal with it
        vect_me, vect_mx = new_state.add_map('vect_pool_map',
                                             dict(w=f"0:{vec_width}"),
                                             unroll=True)

        # the inner map computes the pooling
        inner_me, inner_mx = new_state.add_map('inner_pool_map',
                                               dict(hy=f"0:{filter_height}",
                                                    hx=f"0:{filter_width}"),
                                               unroll=True)

        # read data into vec data
        # tasklet = new_state.add_tasklet('read_tasklet', ['_in'], ['_out'], code="_out = _in")

        # compute the maximum: we can compute always, but we can write the result only
        # according to the slide and at the end of the filter loops
        # NOTE: in_x could reflect the fact that it is vctorized
        compute_tasklet = new_state.add_tasklet("compute_entry",
                                                inputs={"image_in", "max_in"},
                                                outputs={"output", "max_out"},
                                                code=f"""\
if hx == 0 and hy == 0: max_in = {dtypes.min_value(Y.dtype.base_type)}  #init
max_out = float(max(max_in, image_in))
if hy == {filter_height} - 1 and hx == {filter_width} -1 and  in_y % {filter_height} == {filter_height} - 1 and (in_x *{vec_width}+w) % {filter_width} == {filter_width} -1: 
    output = max_out""")

        shift_register = new_state.add_access("shift_register")

        read_X = new_state.add_read("X")
        write_Y = new_state.add_write("Y")
        read_max_res = new_state.add_access("max_res")
        write_max_res = new_state.add_write("max_res")
        vec_data = new_state.add_access("vec_data")

        new_state.add_memlet_path(read_X,
                                  outer_me,
                                  vec_data,
                                  dst_conn="_in",
                                  memlet=dace.Memlet("X[b, c, in_y, in_x]"))

        # memlet: from input image to shift register
        other_subset = f"{shift_register_size - 1}"
        if xilinx:
            other_subset = f"(in_y * {(input_size_width * vec_width)} + (in_x * {vec_width}) + w) % {shift_register_size}"

        to_shift_register_memlet = dace.Memlet(
            "vec_data[{}]".format('0' if vec_width == 1 else 'w'),
            other_subset=other_subset)

        # explicitly set oob otherwise it is not taken
        to_shift_register_memlet.allow_oob = True
        new_state.add_memlet_path(vec_data,
                                  vect_me,
                                  shift_register,
                                  memlet=to_shift_register_memlet,
                                  propagate=False)

        # To create the shift register outside the map, add an empty memlet path
        # shift_register_write = new_state.add_write("shift_register")
        shift_register_read = new_state.add_read("shift_register")

        new_state.add_memlet_path(shift_register_read,
                                  outer_me,
                                  memlet=dace.Memlet())

        # create vector output buffer outside map, empty memlet path
        if Y.veclen != 1:
            vec_out_read = new_state.add_read("vec_data_out")
            new_state.add_memlet_path(vec_out_read,
                                      outer_me,
                                      memlet=dace.Memlet())

        # memlet from shift register to max tasklet
        # NOTE: vec width
        element = f"(hy*{input_size_width * vec_width}+hx)"
        access = f"shift_register[{element}]"
        if xilinx:
            access = f"shift_register[(in_y * {(input_size_width * vec_width)} + (in_x * {vec_width} + w + 1) + {element}) % {shift_register_size}]"
        new_state.add_memlet_path(shift_register,
                                  inner_me,
                                  compute_tasklet,
                                  dst_conn="image_in",
                                  memlet=dace.Memlet(access))

        #memlets for max
        new_state.add_memlet_path(read_max_res,
                                  inner_me,
                                  compute_tasklet,
                                  dst_conn="max_in",
                                  memlet=dace.Memlet("max_res[0]"))
        #empty memlet
        new_state.add_memlet_path(vect_me, read_max_res, memlet=dace.Memlet())

        new_state.add_memlet_path(compute_tasklet,
                                  inner_mx,
                                  write_max_res,
                                  src_conn="max_out",
                                  memlet=dace.Memlet("max_res[0]"))
        #empty memlet
        new_state.add_memlet_path(write_max_res, vect_mx, memlet=dace.Memlet())
        #Attention, the storing location must take into account that the input was vectorized
        if vec_width != 1 and Y.veclen == 1:
            y_memlet = dace.Memlet(
                f"Y[b,c, in_y//{filter_height}, (in_x*{vec_width}+w)//{filter_width}]"
            )
        elif vec_width == 1 and Y.veclen == 1:
            y_memlet = dace.Memlet(
                f"Y[b,c, in_y//{filter_height}, in_x//{filter_width}]")
        else:
            x_access = f"int_floor(in_x, {filter_width})"
            if X.veclen == filter_width * Y.veclen:
                # if input vector size is a multiple of the output vector size
                # we can output to each index of the input on the output
                x_access = "in_x"


        #    dynamic memlet (to access only when needed) from compute tasklet to out image
            y_memlet = dace.Memlet(
                f"Y[b,c, int_floor(in_y, {filter_height}), {x_access}]",
                allow_oob=True,
                dynamic=True)


        # Attention: use propagate=False otherwise it does not validate
        if Y.veclen == 1:

            
            # plain data type output for plain data types or unrolled writes on Intel
            new_state.add_memlet_path(
                compute_tasklet,
                inner_mx,
                vect_mx,
                outer_mx,
                write_Y,
                src_conn="output",
                memlet=y_memlet,
                propagate=True,
            )

        else:

            # vector buffer output for vectorized output buffer case
            new_state.add_memlet_path(
                compute_tasklet,
                inner_mx,
                vect_mx,
                vec_out,
                src_conn="output",
                memlet=dace.Memlet(
                    f"vec_data_out[int_floor(in_x * {vec_width} + w, {filter_width}) % {out_vec_width}]",
                    dynamic=True))

            if X.veclen == filter_width * Y.veclen:
                # if the input vector size is a kernel size multiple of the output vector size
                # we can write to the output on every iteration while reading the correct row
                # on the input
                code = f"if in_y % {filter_height} == {filter_height} - 1: to_mem = vec"
            else:
                code = f"if in_y % {filter_height} == {filter_height} - 1 and in_x % {filter_width} == {filter_width} - 1: to_mem = vec"
            to_memory_task = new_state.add_tasklet(
                "to_memory_task",
                inputs={"vec": dace.vector(X.dtype.base_type, out_vec_width)},
                outputs={"to_mem"},
                code=code,
            )

            new_state.add_memlet_path(vec_out,
                                      to_memory_task,
                                      dst_conn="vec",
                                      memlet=dace.Memlet(f"vec_data_out"))

            new_state.add_memlet_path(
                to_memory_task,
                outer_mx,
                write_Y,
                src_conn="to_mem",
                memlet=y_memlet,
            )

        new_sdfg.fill_scope_connectors()
        return new_sdfg


@op_implementation(op="Gemm", name="fpga")
class FPGAGemm(ONNXForward):
    '''
        GEMM expansion: currently it supports A non transposed and B transposed
        TODO: support more cases
    '''
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

        # TODO: generalize
        # for Lenet, the sake of optimization, the input C is non vectorized
        # while the output Y can be vectorized
        M_C = C.shape[0]
        M_Y = Y.shape[1]
        P = math.gcd(N, 16)  # Num PEs
        vec_width = Y.veclen

        # Tile size, for the moment being the same as M_Y, the output size
        T = M_Y
        # safe delay
        L = max(10 - M_Y, 0)

        ####################################################
        # Build the SDFG: starting point: gemm_fpga_systolic vectorized sample

        def make_read_A(state):
            # TODO: vectorize also this (same rationale of Conv)
            entry, exit = state.add_map(
                "read_A",
                {
                    "n0": f"0:{N}/{P}",
                    "tm":
                    f"0:{M_Y}/{T}",  # must be repeated according to the tile size
                    "k": f"0:{K}"
                },
                schedule=dace.ScheduleType.FPGA_Device)
            # use a different map, and unroll it if necessary
            unroll_inner_map = P > (M_Y + L) and P <= 16
            send_map_entry, send_map_exit = state.add_map(
                "send_A", {"n1": f"0:{P}"},
                schedule=dace.ScheduleType.FPGA_Device,
                unroll=unroll_inner_map)

            mem = state.add_read("A")
            pipe = state.add_write("A_pipe")
            tasklet = state.add_tasklet("read_A", {"from_memory"},
                                        {"to_kernel"},
                                        "to_kernel = from_memory")

            state.add_memlet_path(mem,
                                  entry,
                                  send_map_entry,
                                  tasklet,
                                  dst_conn="from_memory",
                                  memlet=dace.Memlet(f"A[n0 * {P} + n1, k]"))
            state.add_memlet_path(tasklet,
                                  send_map_exit,
                                  exit,
                                  pipe,
                                  src_conn="to_kernel",
                                  memlet=dace.Memlet(f"A_pipe[{P} - n1 - 1]"))

        def make_read_B(state, sdfg, vec_width=1):

            # NOTE: We are reading this transposed: B is originally a matrix MxK
            # B is accessed by row for the GEMM in LENET
            # gear boxing: we read plain data types, we stream vector data types
            # Therefore we have two maps, the innermost is unrolled
            entry, exit = state.add_map("read_B", {
                "n": f"0:{N}/{P}",
                "tm": f"0:{M_Y}/{T}",
                "m": f"0:{K}",
                "k0": f"0:{M_C}/{vec_width}"
            },
                                        schedule=dace.ScheduleType.FPGA_Device)

            read_map_entry, read_map_exit = state.add_map(
                "unrolled_reads_B", {"k1": f"0:{vec_width}"},
                schedule=dace.ScheduleType.FPGA_Device,
                unroll=True)

            # local storage to accumulate data
            sdfg.add_array('vec_data_B',
                           shape=[vec_width],
                           dtype=B.dtype.base_type,
                           transient=True,
                           storage=dace.dtypes.StorageType.FPGA_Registers)
            mem = state.add_read("B")
            pipe = state.add_write("B_pipe")
            vect_data = state.add_access("vec_data_B")
            tasklet = state.add_tasklet("read_B", {"from_memory"},
                                        {"to_kernel"},
                                        "to_kernel = from_memory")

            # In the innermost map we read W=vec_width data elements and we store them into `vec_data`
            state.add_memlet_path(
                mem,
                entry,
                read_map_entry,
                tasklet,
                dst_conn="from_memory",
                memlet=dace.Memlet(f"B[k0*{vec_width}+k1, tm*{T} + m]"))

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

            # Terrible hack to deal with different vec size between C and Y
            if C.veclen != Y.veclen:
                deal_with_misread = True
            else:
                deal_with_misread = False

            pipe = state.add_read("C_pipe")
            mem_read = state.add_read("C")
            mem = state.add_write("Y")

            entry_map, exit_map = state.add_map(
                "write_C",
                {
                    "n": f"0:{N}",
                    "m": f"0:{M_Y}"  #consider also vectorization
                },
                schedule=dace.ScheduleType.FPGA_Device)

            # then we copy that to memory

            if deal_with_misread:
                add_map_entry, add_map_exit = state.add_map(
                    "add_C", {"m1": f"0:{vec_width}"},
                    schedule=dace.ScheduleType.FPGA_Device,
                    unroll=True)
                # local storage to accumulate data
                sdfg.add_array('vec_data_C',
                               shape=[vec_width],
                               dtype=C.dtype.base_type,
                               transient=True,
                               storage=dace.dtypes.StorageType.FPGA_Registers)

                vect_data = state.add_access("vec_data_C")
                # local storage to accumulate data
                sdfg.add_array('vec_res',
                               shape=[vec_width],
                               dtype=C.dtype.base_type,
                               transient=True,
                               storage=dace.dtypes.StorageType.FPGA_Registers)
                vect_res = state.add_access("vec_res")

                # then we transfer them to the output stream
                copy_in_tasklet = state.add_tasklet('copy_from_stream_C',
                                                    {'in_con'}, {'out_con'},
                                                    'out_con = in_con')

                state.add_memlet_path(pipe,
                                      entry_map,
                                      copy_in_tasklet,
                                      dst_conn="in_con",
                                      memlet=dace.Memlet(f"C_pipe[{P}-1]"))
                # this will trigger gear boxing
                state.add_memlet_path(copy_in_tasklet,
                                      vect_data,
                                      src_conn="out_con",
                                      memlet=dace.Memlet("vec_data_C"))

                # add C
                add_C_tasklet = state.add_tasklet('add_C_tasklet',
                                                  {'in_con', 'prev_c'},
                                                  {'out_con'},
                                                  'out_con = in_con + prev_c')
                state.add_memlet_path(vect_data,
                                      add_map_entry,
                                      add_C_tasklet,
                                      dst_conn="in_con",
                                      memlet=dace.Memlet("vec_data_C[m1]"))
                state.add_memlet_path(
                    mem_read,
                    entry_map,
                    add_map_entry,
                    add_C_tasklet,
                    dst_conn="prev_c",
                    memlet=dace.Memlet(f"C[m*{vec_width}+m1]"))

                # write out
                state.add_memlet_path(add_C_tasklet,
                                      add_map_exit,
                                      vect_res,
                                      src_conn="out_con",
                                      memlet=dace.Memlet("vec_res[m1]"))
                state.add_memlet_path(vect_res,
                                      exit_map,
                                      mem,
                                      memlet=dace.Memlet("Y[n,m]"))

            else:
                tasklet = state.add_tasklet(
                    "write_C", {"from_kernel", "prev_c"}, {"to_memory"},
                    "to_memory = from_kernel + prev_c")
                state.add_memlet_path(pipe,
                                      entry_map,
                                      tasklet,
                                      dst_conn="from_kernel",
                                      memlet=dace.Memlet(f"C_pipe[{P}-1]"))
                state.add_memlet_path(mem_read,
                                      entry_map,
                                      tasklet,
                                      dst_conn="prev_c",
                                      memlet=dace.Memlet("C[m]"))
                state.add_memlet_path(tasklet,
                                      exit_map,
                                      mem,
                                      src_conn="to_memory",
                                      memlet=dace.Memlet("Y[n, m]"))

        def make_compute(sdfg, state, vec_width=1):

            vec_type = dace.vector(B.dtype.base_type, vec_width)
            A_pipe_in = state.add_read("A_pipe")
            B_pipe_in = state.add_read("B_pipe")
            B_pipe_out = state.add_write("B_pipe")
            C_pipe_in = state.add_read("C_pipe")
            C_pipe_out = state.add_write("C_pipe")

            entry_pipeline, exit_pipeline = state.add_pipeline(
                "compute_and_drain", {
                    "n0": f"0:{N}/{P}",
                    "tm": f"0:{M_Y}/{T}",
                    "k": f"0:{K}",
                    "m": f"0:{T} + {L}"
                },
                drain_size=P * T,
                drain_overlap=False,
                additional_iterators={
                    'm_drain': 0,
                    'k_drain': 0
                },
                schedule=dace.ScheduleType.FPGA_Device)

            # Instantiate buffers
            sdfg.add_scalar("A_reg",
                            dtype=A.dtype,
                            transient=True,
                            storage=dace.dtypes.StorageType.FPGA_Registers)
            A_reg = state.add_write("A_reg")
            A_reg_init = state.add_access("A_reg")

            # For C result we are going to use vectorized data type

            # Note: for some of the Sacred Mysteries of Intel OpenCL Compiler (TM), if this buffer is smaller
            # than 24 floats, the II of the pipeline will be 5. Therefore we check this (with 32 to be
            # more compliant with standard vector size) and in case we enlarge it

            buffer_size = max(M_Y * vec_width, 32) / vec_width
            sdfg.add_array("C_buffer", [buffer_size],
                           dtype=vec_type,
                           transient=True,
                           storage=dace.dtypes.StorageType.FPGA_Local)
            C_buffer_in = state.add_read("C_buffer")
            C_buffer_out = state.add_write("C_buffer")

            # Feed A
            # every PE: reads input data, buffer the data assigned to it
            buffer_a_tasklet = state.add_tasklet(
                "buffer_a", {"a_in"}, {
                    "a_reg",
                }, f"""\
if m == 0 and not {entry_pipeline.pipeline.drain_condition()}:
    a_reg = a_in""")
            state.add_memlet_path(A_pipe_in,
                                  entry_pipeline,
                                  buffer_a_tasklet,
                                  memlet=dace.Memlet("A_pipe[p]",
                                                     dynamic=True),
                                  dst_conn="a_in")
            state.add_memlet_path(buffer_a_tasklet,
                                  A_reg,
                                  memlet=dace.Memlet("A_reg[0]", dynamic=True),
                                  src_conn="a_reg")

            # Feed B
            # Read B: done outside of the compute tasklet to help type inference
            sdfg.add_array("B_reg",
                           shape=[1],
                           dtype=vec_type,
                           transient=True,
                           storage=dace.dtypes.StorageType.FPGA_Local)
            B_reg = state.add_access("B_reg")
            buffer_b_tasklet = state.add_tasklet(
                "buffer_b", {"b_in"}, {"b_reg_out"}, f"""\
if  m>={L} and not {entry_pipeline.pipeline.drain_condition()}:
    b_reg_out = b_in""")

            state.add_memlet_path(B_pipe_in,
                                  entry_pipeline,
                                  buffer_b_tasklet,
                                  memlet=dace.Memlet("B_pipe[p]",
                                                     dynamic=True),
                                  dst_conn="b_in")
            state.add_memlet_path(buffer_b_tasklet,
                                  B_reg,
                                  memlet=dace.Memlet("B_reg[0]", dynamic=True),
                                  src_conn="b_reg_out")
            # COMPUTE AND DRAIN
            # Compute and forward B: this is done if we are not in the init phase of the pipeline
            compute_tasklet = state.add_tasklet(
                "compute_and_drain", {"a_in", "b_in", "c_in", "forward_in"},
                {"b_out", "c_out", "c_pipe_out"}, f"""\
if m>= {L} and not {entry_pipeline.pipeline.drain_condition()}:
    c_prev = 0 if k == 0 else c_in     
    c_out =  c_prev + a_in * b_in
    if p < {P} - 1:
        b_out = b_in
# Drain
# when we have to drain:
# - if k = K-1 and m>=L: drain my own result
#-  otherwise, if k_drain<p forward data coming from previous PEs (this could happens also in the drain phase)
if((n0 > 0 or tm > 0)  and k_drain <p and m_drain <{T}) or  (k=={K}-1 and m>= {L}) or ({entry_pipeline.pipeline.drain_condition()} and k_drain < p):
   c_pipe_out = c_out if (p==0 or (k_drain=={K}-1 and not {entry_pipeline.pipeline.drain_condition()})) else forward_in

# adjust draining iterators
if not {entry_pipeline.pipeline.drain_condition()}:
    if m_drain >= {L} +  {T} -1:
        m_drain = 0
        if k_drain >= {K} - 1:
            k_drain = 0
        else:
            k_drain = k_drain +1
    else:
        m_drain = m_drain + 1
else:
    if m_drain >=  {T} -1:
        m_drain = 0
        if k_drain >= {K} - 1:
            k_drain = 0
        else:
            k_drain = k_drain +1
    else:
        m_drain = m_drain + 1
            """)

            state.add_memlet_path(A_reg,
                                  compute_tasklet,
                                  dst_conn="a_in",
                                  memlet=dace.Memlet("A_reg[0]"))
            state.add_memlet_path(B_reg,
                                  compute_tasklet,
                                  memlet=dace.Memlet("B_reg[0]",
                                                     dynamic=False),
                                  dst_conn="b_in")

            state.add_memlet_path(compute_tasklet,
                                  exit_pipeline,
                                  B_pipe_out,
                                  memlet=dace.Memlet("B_pipe[p + 1]",
                                                     dynamic=True),
                                  src_conn="b_out")
            state.add_memlet_path(C_buffer_in,
                                  entry_pipeline,
                                  compute_tasklet,
                                  dst_conn="c_in",
                                  memlet=dace.Memlet(f"C_buffer[m-{L}]",
                                                     allow_oob=True))

            state.add_memlet_path(compute_tasklet,
                                  exit_pipeline,
                                  C_buffer_out,
                                  memlet=dace.Memlet(f"C_buffer[m-{L}]",
                                                     allow_oob=True,
                                                     dynamic=True),
                                  src_conn="c_out")

            state.add_memlet_path(C_pipe_in,
                                  entry_pipeline,
                                  compute_tasklet,
                                  memlet=dace.Memlet("C_pipe[p-1]",
                                                     dynamic=True),
                                  dst_conn="forward_in")
            state.add_memlet_path(compute_tasklet,
                                  exit_pipeline,
                                  C_pipe_out,
                                  memlet=dace.Memlet("C_pipe[p]",
                                                     dynamic=True),
                                  src_conn="c_pipe_out")

            # Unroll processing elements
            compute_entry, compute_exit = state.add_map(
                "unroll_compute", {"p": f"0:{P}"},
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

            state.add_memlet_path(B_pipe_out,
                                  compute_exit,
                                  memlet=dace.memlet.Memlet())
            state.add_memlet_path(C_pipe_out,
                                  compute_exit,
                                  memlet=dace.memlet.Memlet())

            state.add_memlet_path(compute_entry,
                                  A_reg_init,
                                  memlet=dace.memlet.Memlet())
            state.add_memlet_path(A_reg_init,
                                  entry_pipeline,
                                  memlet=dace.memlet.Memlet())
            b_init = state.add_access("B_reg")
            state.add_memlet_path(compute_entry, b_init, memlet=dace.Memlet())
            state.add_memlet_path(b_init, entry_pipeline, memlet=dace.Memlet())
            state.add_memlet_path(compute_entry,
                                  C_buffer_in,
                                  memlet=dace.Memlet())

        # build the compute State
        vec_type = dace.vector(B.dtype.base_type, vec_width)

        new_sdfg.add_stream("A_pipe",
                            A.dtype.base_type,
                            transient=True,
                            shape=(P, ),
                            storage=dace.dtypes.StorageType.FPGA_Local,
                            buffer_size=str(P))
        new_sdfg.add_stream("B_pipe",
                            vec_type,
                            transient=True,
                            shape=(P + 1, ),
                            buffer_size=2,
                            storage=dace.dtypes.StorageType.FPGA_Local)
        new_sdfg.add_stream("C_pipe",
                            vec_type,
                            transient=True,
                            shape=(P + 1, ),
                            buffer_size=T,
                            storage=dace.dtypes.StorageType.FPGA_Local)

        make_read_A(new_state)
        make_read_B(new_state, new_sdfg, vec_width)
        make_compute(new_sdfg, new_state, vec_width)
        make_write_C(new_state, new_sdfg, vec_width)

        new_sdfg.fill_scope_connectors()
        new_sdfg.validate()
        return new_sdfg


@op_implementation(op="Reshape", name="fpga")
class FPGAReshape(ONNXForward):
    '''
        Reshape expansion: this relies on views
        TODO: have a transformation to get rid of reshapes. On device they should be useless.
    '''
    @staticmethod
    def forward(node: ONNXOp, state: SDFGState,
                sdfg: SDFG) -> typing.Union[Node, SDFG]:
        node.validate(sdfg, state)
        if (in_desc_with_name(node, state, sdfg, "data").dtype !=
                out_desc_with_name(node, state, sdfg, "reshaped")):
            raise ValueError(
                "Expected input and output to have the same dtype.")

        new_shape = out_desc_with_name(node, state, sdfg, "reshaped").shape
        node.remove_in_connector("shape")

        shape_node = in_edge_with_name(node, state, "shape").src
        constant_folding.remove_node_and_computation(sdfg, state, shape_node)

        def prog(data, reshaped):
            reshaped[:] = np.reshape(data, new_shape)

        return program_for_node(prog, sdfg, state, node)


@op_implementation(op="Softmax", name="fpga")
class FPGASoftmax(ONNXForward):
    @staticmethod
    def forward_can_be_applied(node: ONNXOp, state: SDFGState,
                               sdfg: SDFG) -> bool:

        inparr = in_desc_with_name(node, state, sdfg, "input")
        axis = node.axis
        # ad hoc implementation, which accepts only the last axis needs to be generalized
        return len(inparr.shape) - 1 == axis

    @staticmethod
    def forward(node: ONNXOp, state: SDFGState,
                sdfg: SDFG) -> typing.Union[nodes.Node, SDFG]:
        # TODO: check stability
        # try to avoid max computation, this could have
        # problems for numerical stability
        # https://stackoverflow.com/questions/34968722/how-to-implement-the-softmax-function-in-python
        # result = exp / sum

        node.validate(sdfg, state)
        inparr = in_desc_with_name(node, state, sdfg, "input")
        outarr = out_desc_with_name(node, state, sdfg, "output")

        axis = node.axis
        if type(axis) is not int or not (-len(inparr.shape) <= axis < len(
                inparr.shape)):
            raise ValueError(
                f"expected axis to be an integer in range [-{len(inparr.shape)}, {len(inparr.shape)}), got {axis}"
            )

        if axis < 0:
            axis += len(inparr.shape)

        new_sdfg = dace.SDFG("fpga_softmax")
        new_state = new_sdfg.add_state("compute")
        new_sdfg.add_datadesc("input", copy.deepcopy(inparr))
        new_sdfg.add_datadesc("output", copy.deepcopy(outarr))

        # Add registers to store exp results
        # TODO: ok in small models
        new_sdfg.add_array("exp_data", [inparr.shape[-1]],
                           dtype=inparr.dtype.base_type,
                           transient=True,
                           storage=dace.dtypes.StorageType.FPGA_Registers)
        new_sdfg.add_array("sum_data", [1],
                           dtype=inparr.dtype.base_type,
                           transient=True,
                           storage=dace.dtypes.StorageType.FPGA_Registers)

        ##################
        # exp of all elements, store them into registers

        # Create a two level maps: outermost is for each batch element
        # Inside we will have two maps, one after the other, that computes
        # the exp and the div

        #batch map
        map_ranges = {
            '__i%d' % i: '0:%s' % n
            for i, n in enumerate(inparr.shape[:-1])
        }

        batch_me, batch_mx = new_state.add_map("softmax_map", map_ranges)

        #exp map
        exp_me, exp_mx = new_state.add_map("softmax_exp",
                                           dict(i=f"0:{inparr.shape[-1]}"))

        #div map
        div_me, div_mx = new_state.add_map("softmax_max",
                                           dict(i=f"0:{inparr.shape[-1]}"))

        exp_tasklet = new_state.add_tasklet(
            'exp_task',
            ['_in', '_in_sum'],
            ['_out', '_out_sum'],
            '_exp = float(0)\n'  #for type inference
            '_exp = exp(_in)\n'
            'prev_sum = _in_sum if i!=0 else float(0)\n'
            '_out_sum = prev_sum + _exp\n'
            '_out = _exp')
        div_tasklet = new_state.add_tasklet('div_task', ['_in', '_sum'],
                                            ['_out'], '_out = _in/_sum')

        in_read = new_state.add_read("input")
        out_write = new_state.add_write("output")
        exp_data = new_state.add_access("exp_data")
        sum_in = new_state.add_access("sum_data")
        sum_accum = new_state.add_access("sum_data")
        init_tasklet = new_state.add_tasklet('init_task', [], ['_out'],
                                             '_out = float(0)')

        memlet_except_axis = f"{','.join(['__i%d' % i for i in range(len(inparr.shape) - 1)])}"

        new_state.add_memlet_path(
            in_read,
            batch_me,
            exp_me,
            exp_tasklet,
            dst_conn="_in",
            memlet=dace.Memlet(f"input[{memlet_except_axis},i]"))

        new_state.add_memlet_path(init_tasklet,
                                  sum_in,
                                  src_conn="_out",
                                  memlet=dace.Memlet("sum_data[0]"))

        new_state.add_memlet_path(sum_in,
                                  exp_me,
                                  exp_tasklet,
                                  dst_conn="_in_sum",
                                  memlet=dace.Memlet("sum_data[0]"))
        new_state.add_memlet_path(batch_me, init_tasklet, memlet=dace.Memlet())
        new_state.add_memlet_path(exp_tasklet,
                                  exp_mx,
                                  exp_data,
                                  src_conn="_out",
                                  memlet=dace.Memlet("exp_data[i]"))
        new_state.add_memlet_path(exp_tasklet,
                                  exp_mx,
                                  sum_accum,
                                  src_conn="_out_sum",
                                  memlet=dace.Memlet("sum_data[0]"))

        ###### DIV

        new_state.add_memlet_path(exp_data,
                                  div_me,
                                  div_tasklet,
                                  dst_conn="_in",
                                  memlet=dace.Memlet("exp_data[i]"))

        new_state.add_memlet_path(sum_accum,
                                  div_me,
                                  div_tasklet,
                                  dst_conn="_sum",
                                  memlet=dace.Memlet("sum_data[0]"))
        new_state.add_memlet_path(
            div_tasklet,
            div_mx,
            batch_mx,
            out_write,
            src_conn="_out",
            memlet=dace.Memlet(f"output[{memlet_except_axis}, i]"),
            propagate=False)

        new_sdfg.fill_scope_connectors()
        return new_sdfg


@op_implementation(op="MatMul", name="fpga")
class FPGAMatMul(ONNXForward):
    '''
        Matmul expansion. It is currently based on the same systolic architecture of Conv/GEMM
        This expansion deal with specific EINSUM configurations

        TODO: improve expansion. Right now the #PEs in certain case depends only on one axis
        '''
    @staticmethod
    def forward_can_be_applied(node: ONNXOp, state: SDFGState,
                               sdfg: SDFG) -> bool:

        input0_dim = len(in_desc_with_name(node, state, sdfg, "A").shape)
        input1_dim = len(in_desc_with_name(node, state, sdfg, "B").shape)
        if input0_dim == 4 and input1_dim == 4:
            return False  # TODO

        if input0_dim == 3 and input1_dim == 2:
            return True

        if input0_dim == 3 and input1_dim == 3:
            return True

        if input0_dim == 2 and input1_dim == 2:
            return False  # TODO

        return False

    @staticmethod
    def forward(node: ONNXOp, state: SDFGState,
                sdfg: SDFG) -> typing.Union[nodes.Node, SDFG]:

        node.validate(sdfg, state)

        A = in_desc_with_name(node, state, sdfg, "A")
        B = in_desc_with_name(node, state, sdfg, "B")
        Y = out_desc_with_name(node, state, sdfg, "Y")
        input0_dim = len(A.shape)
        input1_dim = len(B.shape)

        # TODO: factorize: currently there are three different implementations
        # also because of the systolic array architecture.
        # We can factorize more than this, for example by allowing 3D-3D and 3D-2D to
        # be the same but with a different # PE selection (+ some memlets)

        new_sdfg = dace.SDFG("fpga_matmul")
        new_state = new_sdfg.add_state("mmm_compute")

        # Input/Output shapes and strides are inferred by ONNX shape inference
        # Matrix A, has shape (BATCH, N, K)
        BATCH, N, K = A.shape
        # its strides are (sAB, sAN, sAK)

        # Matrix B has shape ([BATCH,] K, M)
        M = B.shape[-1]  # Note, this accounts for vectorization
        # its strides are (sBB, sBK, sBM)

        # Matrix Y, the result has shape (BATCH, N, M)
        # its shape is (sCB, sCN, sCM)

        ###############################
        # Add the containers to the new_sdfg
        new_sdfg.add_datadesc("A", copy.deepcopy(A))
        new_sdfg.add_datadesc("B", copy.deepcopy(B))
        new_sdfg.add_datadesc("Y", copy.deepcopy(Y))
        new_sdfg.arrays["A"].transient = False
        new_sdfg.arrays["B"].transient = False
        new_sdfg.arrays["Y"].transient = False

        # This depends on the input. We deal with disalignment in input/output vectorization widths
        vec_width = B.veclen

        # This expansions performs the following einsum:
        # - 'bik,bkj->bij' (batched matmul)
        # - 'bik,kj->bij' (B is a 2D tensor)

        # TODO: tiling
        # TODO: choose PE in a wiser way, and deal with PEs that do not divide N (or whatever dimension is meaningul)
        #   For this, check the GEMM generic implementation on the "generic" branch
        T = M  #T is expressed in vector data type (e.g. float4)

        # safe delay (see explanation later, when the pipeline scope is created)
        L = max(11 - T, 0)
        P = math.gcd(N, 16) if input1_dim != 2 else math.gcd(N * BATCH,
                                                             16)  # Num PEs
        P = math.gcd(
            K, P
        )  # (this to ensure that the cycles needed to compute on each PE > number of cycle to drain everything; see later)

        # In order to guarantee correctness an deadlock free:
        # -  we have to ensure that the number of cycles needed to drain everything must be less or equal to
        #    the number of cycles needed for a PE to compute one row of result
        # If this condition is not met, this will return a wrong result/deadlock
        # It is quite complicated to always satisfy this condition in current implementation.

        assert (K <= P * T)  # validity check.

        def make_read_A(state):

            if input1_dim != 2:
                entry, exit = state.add_map(
                    "read_A",
                    {
                        "b": f"0:{BATCH}",
                        "n0": f"0:{N}/{P}",
                        "tm":
                        f"0:{M}/{T}",  # must be repeated according to the tile size
                        "k": f"0:{K}"
                    },
                    schedule=dace.ScheduleType.FPGA_Device)
            else:
                entry, exit = state.add_map(
                    "read_A",
                    {
                        "b_n": f"0:({BATCH}*{N})/{P}",
                        "tm":
                        f"0:{M}/{T}",  # must be repeated according to the tile size
                        "k": f"0:{K}"
                    },
                    schedule=dace.ScheduleType.FPGA_Device)

            # use a different map, and unroll it if necessary
            unroll_inner_map = P > (M + L) and P <= 16
            send_map_entry, send_map_exit = state.add_map(
                "send_A", {"n1": f"0:{P}"},
                schedule=dace.ScheduleType.FPGA_Device,
                unroll=unroll_inner_map)

            mem = state.add_read("A")
            pipe = state.add_write("A_pipe")
            tasklet = state.add_tasklet("read_A", {"from_memory"},
                                        {"to_kernel"},
                                        "to_kernel = from_memory")
            if input1_dim != 2:
                memlet_A = dace.Memlet(f"A[b, n0 * {P} + n1, k]")
            else:
                memlet_A = dace.Memlet(
                    f"A[(b_n*{P}+n1)//{N}, (b_n*{P}+ n1)%{N} , k]")
            state.add_memlet_path(mem,
                                  entry,
                                  send_map_entry,
                                  tasklet,
                                  dst_conn="from_memory",
                                  memlet=memlet_A)
            state.add_memlet_path(tasklet,
                                  send_map_exit,
                                  exit,
                                  pipe,
                                  src_conn="to_kernel",
                                  memlet=dace.Memlet(f"A_pipe[{P} - n1 - 1]"))

        def make_read_B(state):

            if input1_dim != 2:
                entry, exit = state.add_map(
                    "read_B", {
                        "b": f"0:{BATCH}",
                        "n": f"0:{N}/{P}",
                        "tm": f"0:{M}/{T}",
                        "k": f"0:{K}",
                        "m": f"0:{T}"
                    },
                    schedule=dace.ScheduleType.FPGA_Device)
            else:
                entry, exit = state.add_map(
                    "read_B", {
                        "b_n": f"0:({BATCH}*{N})/{P}",
                        "tm": f"0:{M}/{T}",
                        "k": f"0:{K}",
                        "m": f"0:{T}"
                    },
                    schedule=dace.ScheduleType.FPGA_Device)

            mem = state.add_read("B")
            pipe = state.add_write("B_pipe")
            tasklet = state.add_tasklet("read_B", {"from_memory"},
                                        {"to_kernel"},
                                        "to_kernel = from_memory")
            if input1_dim != 2:
                memlet_B = dace.Memlet(f"B[b, k, tm*{M / T} + m]")
            else:
                memlet_B = dace.Memlet(f"B[k, tm*{M / T} + m]")

            state.add_memlet_path(mem,
                                  entry,
                                  tasklet,
                                  dst_conn="from_memory",
                                  memlet=memlet_B)

            state.add_memlet_path(tasklet,
                                  exit,
                                  pipe,
                                  src_conn="to_kernel",
                                  memlet=dace.Memlet("B_pipe[0]"))

        def make_write_Y(state, vec_width=1):
            # Y data arrives as expressed in vect. data type

            pipe = state.add_read("Y_pipe")
            mem = state.add_write("Y")

            # Temp: allow Y to have different vec width from B
            if Y.veclen != B.veclen:
                different_vec_width = True
            else:
                different_vec_width = False

            if input1_dim != 2:
                entry_map, exit_map = state.add_map(
                    "write_Y",
                    {
                        "b": f"0:{BATCH}",
                        "n0": f"0:{N}/{P}",
                        "tm": f"0:{M}/{T}",
                        "n1": f"0:{P}",
                        "m": f"0:{T}"  # considers also vectorization
                    },
                    schedule=dace.ScheduleType.FPGA_Device)
            else:
                entry_map, exit_map = state.add_map(
                    "write_Y",
                    {
                        "b_n": f"0:({BATCH}*{N})/{P}",
                        "tm": f"0:{M}/{T}",
                        "n1": f"0:{P}",
                        "m": f"0:{T}"  # considers also vectorization
                    },
                    schedule=dace.ScheduleType.FPGA_Device)

            tasklet = state.add_tasklet("write_Y_tasklet", {"from_kernel"},
                                        {"to_memory"},
                                        "to_memory = from_kernel")
            if not different_vec_width:
                # write directly in memory
                state.add_memlet_path(pipe,
                                      entry_map,
                                      tasklet,
                                      dst_conn="from_kernel",
                                      memlet=dace.Memlet(f"Y_pipe[{P}-1]"))

                if input1_dim != 2:
                    memlet_Y = dace.Memlet(f"Y[b, n0 * {P} + n1, tm*{T}+ m]")
                else:
                    memlet_Y = dace.Memlet(
                        f"Y[(b_n*{P}+n1)//{N}, (b_n*{P}+n1)%{N}, tm*{T}+ m]")
                state.add_memlet_path(tasklet,
                                      exit_map,
                                      mem,
                                      src_conn="to_memory",
                                      memlet=memlet_Y)
            else:
                entry_write_map, exit_write_map = state.add_map(
                    "write_Y_unrolled", {"i": f"0:{B.veclen}"}, unroll=True)
                # local storage to unpack vectorized data
                new_sdfg.add_array(
                    'vec_res',
                    shape=[B.veclen],
                    dtype=Y.dtype,
                    transient=True,
                    storage=dace.dtypes.StorageType.FPGA_Registers)
                vec_res = state.add_access("vec_res")
                state.add_memlet_path(pipe,
                                      entry_map,
                                      vec_res,
                                      memlet=dace.Memlet(f"Y_pipe[{P}-1]"))
                state.add_memlet_path(vec_res,
                                      entry_write_map,
                                      tasklet,
                                      dst_conn="from_kernel",
                                      memlet=dace.Memlet("vec_res[i]"))
                if input1_dim != 2:
                    memlet_Y = dace.Memlet(
                        f"Y[b, n0 * {P} + n1, (tm*{T}+ m)*{vec_width} + i]")
                else:
                    memlet_Y = dace.Memlet(
                        f"Y[(b_n*{P} + n1)//{N}, (b_n*{P}+ n1)%{N}, (tm*{T}+ m)*{vec_width} + i]"
                    )
                #write to memory
                state.add_memlet_path(tasklet,
                                      exit_write_map,
                                      exit_map,
                                      mem,
                                      src_conn="to_memory",
                                      memlet=memlet_Y)

        def make_compute(sdfg, state, vec_width=1):
            vec_type = dace.vector(Y.dtype.base_type, vec_width)
            A_pipe_in = state.add_read("A_pipe")
            B_pipe_in = state.add_read("B_pipe")
            B_pipe_out = state.add_write("B_pipe")
            Y_pipe_in = state.add_read("Y_pipe")
            Y_pipe_out = state.add_write("Y_pipe")

            if input1_dim != 2:
                entry_pipeline, exit_pipeline = state.add_pipeline(
                    "compute_and_drain",
                    {
                        "b": f"0:{BATCH}",
                        "n0": f"0:{N}/{P}",
                        "tm": f"0:{M}/{T}",
                        "k": f"0:{K}",
                        "m": f"0:{T} + {L}"
                    },  # The + L is a safe delay between computing and drain. It must be computed by
                    #considering the latency for updating the same result (not just the FP32 multiply add, but
                    # also for reading/writing from BRAM)
                    drain_size=P * T,
                    drain_overlap=False,
                    additional_iterators={
                        'm_drain': 0,
                        'k_drain': 0
                    },
                    schedule=dace.ScheduleType.FPGA_Device)
            else:
                entry_pipeline, exit_pipeline = state.add_pipeline(
                    "compute_and_drain",
                    {
                        "b_n": f"0:({BATCH}*{N})/{P}",
                        "tm": f"0:{M}/{T}",
                        "k": f"0:{K}",
                        "m": f"0:{T} + {L}"
                    },  # The + L is a safe delay between computing and drain. It must be computed by
                    # considering the latency for updating the same result (not just the FP32 multiply add, but
                    # also for reading/writing from BRAM)
                    drain_size=P * T,
                    drain_overlap=False,
                    additional_iterators={
                        'm_drain': 0,
                        'k_drain': 0
                    },
                    schedule=dace.ScheduleType.FPGA_Device)

            # Instantiate buffers
            sdfg.add_scalar("A_reg",
                            dtype=A.dtype.base_type,
                            transient=True,
                            storage=dace.dtypes.StorageType.FPGA_Registers)
            A_reg = state.add_write("A_reg")
            A_reg_init = state.add_access("A_reg")

            # For C result we are going to use vectorized data type

            # Note: for some of the Sacred Mysteries of Intel OpenCL Compiler (TM), if this buffer is smaller
            # than 24 floats, the II of the pipeline will be 5. Therefore we check this (with 32 to be
            # more compliant with standard vector size) and in case we enlarge it
            # TODO: not sure what happens with vec data type
            buffer_size = max(M * vec_width, 32) / vec_width
            sdfg.add_array("Y_buffer", [buffer_size],
                           dtype=vec_type,
                           transient=True,
                           storage=dace.dtypes.StorageType.FPGA_Local)
            Y_buffer_in = state.add_read("Y_buffer")
            Y_buffer_out = state.add_write("Y_buffer")

            # Feed A
            # every PE: reads input data, buffer the data assigned to it
            buffer_a_tasklet = state.add_tasklet(
                "buffer_a", {"a_in"}, {
                    "a_reg",
                }, f"""\
if m == 0 and not {entry_pipeline.pipeline.drain_condition()}:
    a_reg = a_in""")
            state.add_memlet_path(A_pipe_in,
                                  entry_pipeline,
                                  buffer_a_tasklet,
                                  memlet=dace.Memlet("A_pipe[p]",
                                                     dynamic=True),
                                  dst_conn="a_in")
            state.add_memlet_path(buffer_a_tasklet,
                                  A_reg,
                                  memlet=dace.Memlet("A_reg[0]", dynamic=True),
                                  src_conn="a_reg")

            # Feed B
            # Read B: done outside of the compute tasklet to help type inference
            sdfg.add_array("B_reg",
                           shape=[1],
                           dtype=vec_type,
                           transient=True,
                           storage=dace.dtypes.StorageType.FPGA_Local)
            B_reg = state.add_access("B_reg")
            buffer_b_tasklet = state.add_tasklet(
                "buffer_b", {"b_in"}, {"b_reg_out"}, f"""\
if  m>={L} and not {entry_pipeline.pipeline.drain_condition()}:
    b_reg_out = b_in""")

            state.add_memlet_path(B_pipe_in,
                                  entry_pipeline,
                                  buffer_b_tasklet,
                                  memlet=dace.Memlet("B_pipe[p]",
                                                     dynamic=True),
                                  dst_conn="b_in")
            state.add_memlet_path(buffer_b_tasklet,
                                  B_reg,
                                  memlet=dace.Memlet("B_reg[0]", dynamic=True),
                                  src_conn="b_reg_out")
            # COMPUTE AND DRAIN
            # Compute and forward B: this is done if we are not in the init phase of the pipeline

            if input1_dim != 2:
                compute_tasklet = state.add_tasklet(
                    "compute_and_drain",
                    {"a_in", "b_in", "y_in", "forward_in"},
                    {"b_out", "y_out", "y_pipe_out"}, f"""\
if m>= {L} and not {entry_pipeline.pipeline.drain_condition()}:
    y_prev = 0 if k == 0 else y_in     
    y_out =  y_prev + a_in * b_in
    if p < {P} - 1:
        b_out = b_in
# Drain
# when we have to drain:
# - if we are working on the second batch, or second assigned row or second tile and we have something to drain
# - if k = K-1 and m>=L: then the PE drains its own result
# - if we are in the draining phase
# How: 
# - if k = K-1 and m>=L: then the PE drains its own result
#-  otherwise, if k_drain<p forward data coming from previous PEs (this could happens also in the drain phase)
if((b>0 or n0 > 0 or tm > 0)  and k_drain <p and m_drain <{T}) or  (k=={K}-1 and m>= {L}) or ({entry_pipeline.pipeline.drain_condition()} and k_drain < p):
    y_pipe_out = y_out if (p==0 or (k_drain=={K}-1 and not {entry_pipeline.pipeline.drain_condition()})) else forward_in

# adjust draining iterators
if not {entry_pipeline.pipeline.drain_condition()}:
    if m_drain >= {L} +  {T} -1:
        m_drain = 0
        if k_drain >= {K} - 1:
            k_drain = 0
        else:
            k_drain = k_drain +1
    else:
        m_drain = m_drain + 1
else:
    if m_drain >=  {T} -1:
        m_drain = 0
        if k_drain >= {K} - 1:
            k_drain = 0
        else:
            k_drain = k_drain +1
    else:
        m_drain = m_drain + 1
        """)
            else:
                # COMPUTE AND DRAIN
                # Compute and forward B: this is done if we are not in the init phase of the pipeline
                compute_tasklet = state.add_tasklet(
                    "compute_and_drain",
                    {"a_in", "b_in", "y_in", "forward_in"},
                    {"b_out", "y_out", "y_pipe_out"}, f"""\
if m>= {L} and not {entry_pipeline.pipeline.drain_condition()}:
    y_prev = 0 if k == 0 else y_in     
    y_out =  y_prev + a_in * b_in
    if p < {P} - 1:
        b_out = b_in
# Drain
# when we have to drain:
# - if we are working on the second batch, or second assigned row or second tile and we have something to drain
# - if k = K-1 and m>=L: then the PE drains its own result
# - if we are in the draining phase
# How: 
# - if k = K-1 and m>=L: then the PE drains its own result
#-  otherwise, if k_drain<p forward data coming from previous PEs (this could happens also in the drain phase)
if((((b_n*{P})/{N})>0 or (b_n*{P})%{N} > 0 or tm > 0)  and k_drain <p and m_drain <{T}) or  (k=={K}-1 and m>= {L}) or ({entry_pipeline.pipeline.drain_condition()} and k_drain < p):
    y_pipe_out = y_out if (p==0 or (k_drain=={K}-1 and not {entry_pipeline.pipeline.drain_condition()})) else forward_in

# adjust draining iterators
if not {entry_pipeline.pipeline.drain_condition()}:
    if m_drain >= {L} +  {T} -1:
        m_drain = 0
        if k_drain >= {K} - 1:
            k_drain = 0
        else:
            k_drain = k_drain +1
    else:
        m_drain = m_drain + 1
else:
    if m_drain >=  {T} -1:
        m_drain = 0
        if k_drain >= {K} - 1:
            k_drain = 0
        else:
            k_drain = k_drain +1
    else:
        m_drain = m_drain + 1
                    """)

            state.add_memlet_path(A_reg,
                                  compute_tasklet,
                                  dst_conn="a_in",
                                  memlet=dace.Memlet("A_reg[0]"))
            state.add_memlet_path(B_reg,
                                  compute_tasklet,
                                  memlet=dace.Memlet("B_reg[0]",
                                                     dynamic=False),
                                  dst_conn="b_in")

            state.add_memlet_path(compute_tasklet,
                                  exit_pipeline,
                                  B_pipe_out,
                                  memlet=dace.Memlet("B_pipe[p + 1]",
                                                     dynamic=True),
                                  src_conn="b_out")
            state.add_memlet_path(Y_buffer_in,
                                  entry_pipeline,
                                  compute_tasklet,
                                  dst_conn="y_in",
                                  memlet=dace.Memlet(f"Y_buffer[m-{L}]",
                                                     allow_oob=True))

            state.add_memlet_path(compute_tasklet,
                                  exit_pipeline,
                                  Y_buffer_out,
                                  memlet=dace.Memlet(f"Y_buffer[m-{L}]",
                                                     allow_oob=True,
                                                     dynamic=True),
                                  src_conn="y_out")

            state.add_memlet_path(Y_pipe_in,
                                  entry_pipeline,
                                  compute_tasklet,
                                  memlet=dace.Memlet("Y_pipe[p-1]",
                                                     dynamic=True),
                                  dst_conn="forward_in")
            state.add_memlet_path(compute_tasklet,
                                  exit_pipeline,
                                  Y_pipe_out,
                                  memlet=dace.Memlet("Y_pipe[p]",
                                                     dynamic=True),
                                  src_conn="y_pipe_out")

            # Unroll processing elements
            compute_entry, compute_exit = state.add_map(
                "unroll_compute", {"p": f"0:{P}"},
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
                                  Y_pipe_in,
                                  memlet=dace.memlet.Memlet())

            state.add_memlet_path(B_pipe_out,
                                  compute_exit,
                                  memlet=dace.memlet.Memlet())

            state.add_memlet_path(Y_pipe_out,
                                  compute_exit,
                                  memlet=dace.memlet.Memlet())

            state.add_memlet_path(compute_entry,
                                  A_reg_init,
                                  memlet=dace.memlet.Memlet())
            state.add_memlet_path(A_reg_init,
                                  entry_pipeline,
                                  memlet=dace.memlet.Memlet())
            b_init = state.add_access("B_reg")
            state.add_memlet_path(compute_entry, b_init, memlet=dace.Memlet())
            state.add_memlet_path(b_init, entry_pipeline, memlet=dace.Memlet())
            state.add_memlet_path(compute_entry,
                                  Y_buffer_in,
                                  memlet=dace.Memlet())

        # build the compute State
        vec_type = dace.vector(Y.dtype.base_type, vec_width)

        new_sdfg.add_stream("A_pipe",
                            A.dtype.base_type,
                            transient=True,
                            shape=(P, ),
                            storage=dace.dtypes.StorageType.FPGA_Local,
                            buffer_size=str(P))
        new_sdfg.add_stream("B_pipe",
                            vec_type,
                            transient=True,
                            shape=(P + 1, ),
                            buffer_size=2,
                            storage=dace.dtypes.StorageType.FPGA_Local)
        new_sdfg.add_stream("Y_pipe",
                            vec_type,
                            transient=True,
                            shape=(P + 1, ),
                            buffer_size=T,
                            storage=dace.dtypes.StorageType.FPGA_Local)

        make_read_A(new_state)
        make_read_B(new_state)
        make_compute(new_sdfg, new_state, vec_width)
        make_write_Y(new_state, vec_width)

        new_sdfg.fill_scope_connectors()
        new_sdfg.validate()
        return new_sdfg


@op_implementation(op="ReduceSum", name="fpga")
class FPGAReduceSum(ONNXForward):
    @staticmethod
    def forward_can_be_applied(node: ONNXOp, state: SDFGState,
                               sdfg: SDFG) -> bool:
        axes = node.axes
        indata = in_desc_with_name(node, state, sdfg, "data")

        # TODO: improve coverage
        if axes[0] != 1:
            return False

        if len(indata.shape) != 4:
            return False

        if node.keepdims != False:
            return False

        return True

    @staticmethod
    def forward(node: ONNXOp, state: SDFGState,
                sdfg: SDFG) -> typing.Union[nodes.Node, SDFG]:
        node.validate(sdfg, state)
        axes = node.axes

        # TODO: ad hoc implementation for MHA, needs to be generalized
        # Take a look to Dace Reduce
        # It exploits single clock cycle accumulator of Intel

        indata = in_desc_with_name(node, state, sdfg, "data")
        outdata = out_desc_with_name(node, state, sdfg, "reduced")

        new_sdfg = dace.SDFG("fpga_reduce_sum_expansion")
        new_sdfg.add_datadesc("data", copy.deepcopy(indata))
        new_sdfg.add_datadesc("reduced", copy.deepcopy(outdata))
        new_sdfg.arrays["data"].transient = False
        new_sdfg.arrays["reduced"].transient = False
        new_state = new_sdfg.add_state()

        # variable for reduction
        new_sdfg.add_array("sum_res", [1],
                           indata.dtype.base_type,
                           storage=dace.StorageType.FPGA_Registers,
                           transient=True)

        # outer map along all dimension except axes
        outer_me, outer_mx = new_state.add_map(
            'outer_pool_map',
            dict(o0=f"0:{indata.shape[0]}",
                 o1=f"0:{indata.shape[2]}",
                 o2=f"0:{indata.shape[3]}"))

        # the inner map computes the pooling
        # TODO: unroll/vectorize
        inner_me, inner_mx = new_state.add_map('inner_pool_map',
                                               dict(i0=f"0:{indata.shape[1]}"))

        # accumulate sum
        compute_tasklet = new_state.add_tasklet(
            "sum",
            inputs={"accum_in", "data_in"},
            outputs={"accum_out"},
            code="accum_out = data_in + accum_in")
        sum_in = new_state.add_access("sum_res")
        sum_accum = new_state.add_access("sum_res")
        input_data = new_state.add_read("data")
        out_data = new_state.add_write("reduced")

        init_tasklet = new_state.add_tasklet('init_task', {}, {'_out'},
                                             '_out = float(0)')

        store_tasklet = new_state.add_tasklet('store_tasklet', {'in_res'},
                                              {'out_res'},
                                              code='out_res = in_res')

        # compute tasklet memlets
        # data in
        new_state.add_memlet_path(input_data,
                                  outer_me,
                                  inner_me,
                                  compute_tasklet,
                                  dst_conn="data_in",
                                  memlet=dace.Memlet("data[o0,i0,o1,o2]"))

        #accum in
        new_state.add_memlet_path(sum_in,
                                  inner_me,
                                  compute_tasklet,
                                  dst_conn="accum_in",
                                  memlet=dace.Memlet("sum_res[0]"))

        #accum out
        new_state.add_memlet_path(compute_tasklet,
                                  inner_mx,
                                  sum_accum,
                                  src_conn="accum_out",
                                  memlet=dace.Memlet("sum_res[0]"))

        #store to memory
        new_state.add_memlet_path(sum_accum,
                                  store_tasklet,
                                  dst_conn="in_res",
                                  memlet=dace.Memlet("sum_res[0]"))
        # init accumulator
        new_state.add_memlet_path(init_tasklet,
                                  sum_in,
                                  src_conn="_out",
                                  memlet=dace.Memlet("sum_res[0]"))
        new_state.add_memlet_path(outer_me, init_tasklet, memlet=dace.Memlet())

        new_state.add_memlet_path(store_tasklet,
                                  outer_mx,
                                  out_data,
                                  src_conn="out_res",
                                  memlet=dace.Memlet("reduced[o0, o1, o2]"))

        new_sdfg.fill_scope_connectors()
        new_sdfg.validate()
        return new_sdfg


@op_implementation(op="Slice", name="fpga")
class PureSlice(ONNXForward):
    '''
        Slice expansion
    '''
    @staticmethod
    def forward_can_be_applied(node: ONNXOp, state: SDFGState,
                               sdfg: SDFG) -> bool:
        # check that all the inputs (even the optional ones) are present and constant

        if not hasattr(sdfg, "_parent_onnx_model"):
            return False

        if len(
                search_fpga_name_in_weights(
                    in_edge_with_name(node, state, "starts").src.data,
                    sdfg)) != 1:
            return False

        if len(
                search_fpga_name_in_weights(
                    in_edge_with_name(node, state, "ends").src.data,
                    sdfg)) != 1:
            return False

        # optional inputs
        is_axes_present = True
        try:
            if len(
                    search_fpga_name_in_weights(
                        in_edge_with_name(node, state, "axes").src.data,
                        sdfg)) != 1:
                return False
        except ValueError:
            is_axes_present = False

        is_steps_present = True
        try:
            if len(
                    search_fpga_name_in_weights(
                        in_edge_with_name(node, state, "steps").src.data,
                        sdfg)) != 1:
                return False
        except ValueError:
            is_steps_present = False

        # Current constraints: axes and steps must be explict. Axes must be zero and steps must be 1
        if not is_axes_present or not is_steps_present:
            return False

        # Current constraints: axis must be zero and steps must be 1
        step = sdfg._parent_onnx_model.clean_weights[
            search_fpga_name_in_weights(
                in_edge_with_name(node, state, "steps").src.data,
                sdfg)[0]].numpy()[0]
        axis = sdfg._parent_onnx_model.clean_weights[
            search_fpga_name_in_weights(
                in_edge_with_name(node, state, "axes").src.data,
                sdfg)[0]].numpy()[0]
        if step != 1 or axis != 0:
            return False

        return True

    @staticmethod
    def forward(node: ONNXOp, state: SDFGState,
                sdfg: SDFG) -> typing.Union[Node, SDFG]:

        start = sdfg._parent_onnx_model.clean_weights[
            search_fpga_name_in_weights(
                in_edge_with_name(node, state, "starts").src.data,
                sdfg)[0]].numpy()[0]
        end = sdfg._parent_onnx_model.clean_weights[
            search_fpga_name_in_weights(
                in_edge_with_name(node, state, "ends").src.data,
                sdfg)[0]].numpy()[0]

        # Step is 1 and axis is 0
        output_shape = out_desc_with_name(node, state, sdfg, "output").shape
        if end == end == np.iinfo(np.int64).max:
            # Pytorch exporter artifact
            end = start + output_shape[0]

        def prog(data, output):
            tmp = data[start:end, :]
            # We need reshape to avoid Invalid Edge errors
            output[:] = np.reshape(tmp, output.shape)

        return program_for_node(prog, sdfg, state, node)
