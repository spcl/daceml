import functools
from typing import Union, Optional, Tuple

import dace
from dace import SDFGState, nodes as nd, SDFG, dtypes, data as dt

from daceml.onnx import environments
from daceml.onnx.converters import clean_onnx_name
from daceml.onnx.forward_implementation_abc import ONNXForward
from daceml.onnx.nodes import onnx_op
from daceml.onnx.op_implementations import op_implementation, empty_sdfg_for_node
from daceml.util import in_desc_with_name, out_desc_with_name


def _prod(sequence):
    return functools.reduce(lambda a, b: a * b, sequence, 1)


def _iterables_equal(a, b) -> bool:
    if len(a) != len(b):
        return False
    return all(x == y for x, y in zip(a, b))


def _get_tensor_layout(desc: dt.Array) -> Optional[str]:
    """ Detect the layout of a 4d tensor.

        :param desc: the tensor.
        :return: "NCHW", "NHWC" or None.
    """

    if len(desc.shape) != 4:
        raise ValueError("Tensor with dimension != 4 is not supported")

    # in ONNX, tensor the dimensions are ordered N C H W
    # strides that the contiguous tensor would have
    cont_strides = [_prod(desc.shape[i + 1:]) for i in range(4)]

    nhwc_shape = [desc.shape[0], desc.shape[3], desc.shape[1], desc.shape[2]]

    # strides that a nhwc tensor would have if it was contiguous
    nhwc_contiguous_strides = [_prod(nhwc_shape[i + 1:]) for i in range(4)]
    # strides that the nhwc tensor would have if viewed as a nchw tensor
    nhwc_reshaped_strides = [
        nhwc_contiguous_strides[0], nhwc_contiguous_strides[3],
        nhwc_contiguous_strides[1], nhwc_contiguous_strides[2]
    ]

    if _iterables_equal(desc.strides, cont_strides):
        return "NCHW"
    elif _iterables_equal(desc.strides, nhwc_reshaped_strides):
        return "NHWC"
    else:
        return None


def _cudnn_tensor_descriptor_code(desc: dt.Array, state_field_name: str,
                                  filter: bool) -> Tuple[str, str]:
    """ Emit the cudnn code for the tensor descriptor for a given dace descriptor.

        :param desc: the descriptor of the dace tensor.
        :param state_field_name: the name of the pointer variable where the descriptor should be stored.
        :param filter: True if the tensor is a filter.
        :return: the init and exit code
    """

    # detect layout
    layout = _get_tensor_layout(desc)
    assert layout is not None, "layout changed after can_be_applied"
    f_or_t_str = 'Filter' if filter else 'Tensor'

    layout_str = f"CUDNN_TENSOR_{layout}"
    dtype_str = _DACE_DTYPE_TO_CUDNN_DTYPE[desc.dtype]
    init_code = f"""
    __state->{state_field_name} = new cudnn{f_or_t_str}Descriptor_t;
    daceml::cudnn::CheckCudnnError(cudnnCreate{f_or_t_str}Descriptor(__state->{state_field_name}));
    daceml::cudnn::CheckCudnnError(cudnnSet{f_or_t_str}4dDescriptor(
        *__state->{state_field_name}, 
        {dtype_str if filter else layout_str},
        {layout_str if filter else dtype_str},
        {",".join(str(s) for s in desc.shape)}
    ));
    """
    exit_code = f"""\
    daceml::cudnn::CheckCudnnError(cudnnDestroy{f_or_t_str}Descriptor(*__state->{state_field_name}));
    delete __state->{state_field_name};
    """
    return init_code, exit_code


# yapf: disable
_DACE_DTYPE_TO_CUDNN_DTYPE = {
    dace.float32: "CUDNN_DATA_FLOAT",
    dace.float64: "CUDNN_DATA_DOUBLE",
    dace.uint8: "CUDNN_DATA_UINT8",
    dace.int8: "CUDNN_DATA_INT8",
    dace.int32: "CUDNN_DATA_INT32"
}
# yapf: enable


@op_implementation(op="Conv", name="cuDNN")
class CudnnConvolution(ONNXForward):
    """ Convolution implementation that uses cuDNN.

        This node will check for the existence of a _algorithm attribute on the ONNXConv node it is expanding.
        If this attribute does not exist, it will use `CudnnConvolution.default_algorithm`.

        Available algorithm types are:
        ["implicit_gemm",
         "implicit_precomp_gemm",
         "gemm",
         "direct",
         "fft",
         "fft_tiling",
         "winograd",
         "winograd_nonfused"]
    """
    default_algorithm = "gemm"
    environments = []

    @staticmethod
    def forward_can_be_applied(node: onnx_op.ONNXOp, state: SDFGState,
                               sdfg: SDFG) -> bool:

        descs = [("X", in_desc_with_name(node, state, sdfg, "X")),
                 ("W", in_desc_with_name(node, state, sdfg, "W")),
                 ("Y", out_desc_with_name(node, state, sdfg, "Y"))]

        if "B" in node.in_connectors:
            descs.append(("B", in_desc_with_name(node, state, sdfg, "B")))

        for name, desc in descs:
            # check that the dtype is supported by cudnn
            if desc.dtype not in [
                    dace.float32, dace.float64, dace.uint8, dace.int8,
                    dace.int32
            ]:
                return False
            # only 2d convs for now; ONNX supports N dimensional
            if name != "B" and len(desc.shape) != 4:
                return False

            if not isinstance(desc, dt.Array):
                return False

            # check that the layout is supported by cudnn
            if name != "B" and _get_tensor_layout(desc) is None:
                return False

        return True

    @staticmethod
    def forward(node: onnx_op.ONNXOp, state: SDFGState,
                sdfg: SDFG) -> Union[nd.Node, SDFG]:

        nsdfg, nstate, inputs, outputs = empty_sdfg_for_node(sdfg, state, node)

        if "B" in inputs:
            nstate.remove_node(inputs["B"])
            Y = out_desc_with_name(node, state, sdfg, "Y")
            # add broadcast state
            init_state = nsdfg.add_state_before(nstate, label="broadcast_bias")
            # yapf: disable
            init_state.add_mapped_tasklet("broadcast_bias",
                                          map_ranges={
                                              "i{}".format(i): "0:{}".format(s)
                                              for i, s in enumerate(Y.shape)
                                          },
                                          inputs=dict(
                                              b=dace.Memlet("B[i1]")
                                          ),
                                          code="y = b".format(),
                                          outputs=dict(
                                              y=dace.Memlet("Y[{}]".format(
                                                  ", ".join("i{}".format(i)
                                                            for i, _ in enumerate(Y.shape))))
                                          ),
                                          external_edges=True)
            # yapf: enable

        X_desc = in_desc_with_name(node, state, sdfg, "X")

        T = X_desc.dtype

        in_connectors = {"_X": dace.pointer(T), "_W": dace.pointer(T)}

        unique_id = "{}_{}_{}_{}".format(clean_onnx_name(node.name),
                                         sdfg.sdfg_id, sdfg.node_id(state),
                                         state.node_id(node))

        class Environment:
            cmake_minimum_version = None
            cmake_packages = []
            cmake_variables = {}
            cmake_includes = []
            cmake_libraries = []
            cmake_compile_flags = []
            cmake_link_flags = []
            cmake_files = []
            state_fields = [
                f"cudnnConvolutionDescriptor_t *{unique_id}_conv_desc;",
                f"cudnnTensorDescriptor_t *{unique_id}_Y_desc;",
                f"cudnnTensorDescriptor_t *{unique_id}_X_desc;",
                f"cudnnFilterDescriptor_t *{unique_id}_W_desc;",
                f"float *{unique_id}_workspace;"
                f"size_t *{unique_id}_workspace_size;"
            ]
            dependencies = [environments.cuDNN]
            headers = []
            init_code = ""
            finalize_code = ""

        Environment.__name__ = unique_id + "_environment"
        dace.library.environment(Environment)
        CudnnConvolution.environments = [Environment]

        # setup tensor descriptors
        for edge, is_input in node.iter_edges(state):
            conn = edge.dst_conn if is_input else edge.src_conn
            desc = in_desc_with_name(node, state, sdfg,
                                     conn) if is_input else out_desc_with_name(
                                         node, state, sdfg, conn)
            assert isinstance(desc, dt.Array)
            if conn == "B":
                # bias will be handled separately
                continue
            is_filter = conn == "W"
            init, exit = _cudnn_tensor_descriptor_code(
                desc, f"{unique_id}_{conn}_desc", is_filter)
            Environment.init_code += init
            Environment.finalize_code += exit

        if hasattr(node, "_algorithm"):
            algo = node._algorithm
        else:
            algo = CudnnConvolution.default_algorithm

        # setup conv descriptor
        pad_h, pad_w = 0, 0
        stride_h, stride_w = node.strides
        dilation_h, dilation_w = node.dilations
        Environment.init_code += f"""
        __state->{unique_id}_conv_desc = new cudnnConvolutionDescriptor_t; 
        daceml::cudnn::CheckCudnnError(cudnnCreateConvolutionDescriptor(__state->{unique_id}_conv_desc));
        daceml::cudnn::CheckCudnnError(cudnnSetConvolution2dDescriptor(
            *__state->{unique_id}_conv_desc,
            {pad_h},
            {pad_w},
            {stride_h},
            {stride_w},
            {dilation_h},
            {dilation_w},
            CUDNN_CROSS_CORRELATION,
            {_DACE_DTYPE_TO_CUDNN_DTYPE[T]}));
        """
        Environment.finalize_code += f"""
        daceml::cudnn::CheckCudnnError(cudnnDestroyConvolutionDescriptor(*__state->{unique_id}_conv_desc));
        delete __state->{unique_id}_conv_desc;
        """

        # setup workspace
        Environment.init_code += \
            f"""
        {environments.cuDNN.handle_setup_code(node, init_stream=False)}
        // Setup workspace for {unique_id}
        size_t ws_size;
        daceml::cudnn::CheckCudnnError(cudnnGetConvolutionForwardWorkspaceSize(
            __dace_cudnn_handle,
            *__state->{unique_id}_X_desc,
            *__state->{unique_id}_W_desc,
            *__state->{unique_id}_conv_desc,
            *__state->{unique_id}_Y_desc,
            CUDNN_CONVOLUTION_FWD_ALGO_{algo.upper()},
            &ws_size));
        __state->{unique_id}_workspace_size = new size_t;
        *__state->{unique_id}_workspace_size = ws_size;
        cudaMalloc(&__state->{unique_id}_workspace, ws_size);
        """
        Environment.finalize_code += f"""
        cudaFree(__state->{unique_id}_workspace);
        delete __state->{unique_id}_workspace_size;
        """

        tasklet_code = f"""
        {environments.cuDNN.handle_setup_code(node)}
        float alpha = 1.f;
        float beta = {"1.f" if "B" in inputs else "0.f"};
        daceml::cudnn::CheckCudnnError(cudnnConvolutionForward(
            __dace_cudnn_handle,
            &alpha,
            *__state->{unique_id}_X_desc,
            _X,
            *__state->{unique_id}_W_desc,
            _W,
            *__state->{unique_id}_conv_desc,
            CUDNN_CONVOLUTION_FWD_ALGO_{algo.upper()},
            __state->{unique_id}_workspace,
            *__state->{unique_id}_workspace_size,
            &beta,
            *__state->{unique_id}_Y_desc,
            _Y
        ));
        """

        tasklet = nstate.add_tasklet(unique_id, in_connectors,
                                     {"_Y": dace.pointer(T)}, tasklet_code,
                                     dtypes.Language.CPP)
        nstate.add_edge(inputs["X"], None, tasklet, "_X",
                        nsdfg.make_array_memlet("X"))
        nstate.add_edge(inputs["W"], None, tasklet, "_W",
                        nsdfg.make_array_memlet("W"))

        nstate.add_edge(tasklet, "_Y", outputs["Y"], None,
                        nsdfg.make_array_memlet("Y"))

        return nsdfg
