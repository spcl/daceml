import ctypes

import dace
import numpy as np

from daceml.onnx.schema import ONNXAttributeType
from daceml.onnx.converters import ONNX_DTYPES_TO_DACE_TYPE_CLASS
from daceml.ort_api.raw_api_bindings import OrtCUDAProviderOptions, ORTCAPIInterface, ORTAPIError


class Env:
    def __init__(self, api):
        self.api = api

    def __enter__(self):
        self.ptr = ctypes.c_void_p()
        self.api.CreateEnv("ORT_LOGGING_LEVEL_WARNING", b"ort_api",
                           ctypes.byref(self.ptr))
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.api.ReleaseEnv(self.ptr)


class SessionOptions:
    def __init__(self, api):
        self.api = api
        self.env = Env(api)

    def __enter__(self):
        self.env.__enter__()
        self.ptr = ctypes.c_void_p()

        self.api.CreateSessionOptions(ctypes.byref(self.ptr))

        self.api.dll.OrtSessionOptionsAppendExecutionProvider_CPU(
            self.ptr, ctypes.c_int(0))

        if hasattr(self.api.dll,
                   "OrtSessionOptionsAppendExecutionProvider_CUDA"):
            cuda_opts = OrtCUDAProviderOptions(
                device_id=0,
                cudnn_conv_algo_search=self.api.get_enum_value("DEFAULT"),
                cuda_mem_limit=np.iinfo(ctypes.c_size_t).max,
                do_copy_in_default_stream=1,
                has_user_compute_stream=0,
                user_compute_stream=0)

            self.api.SessionOptionsAppendExecutionProvider_CUDA(
                self.ptr, ctypes.byref(cuda_opts))

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.api.ReleaseSessionOptions(self.ptr)
        self.env.__exit__(exc_type, exc_val, exc_tb)


class KernelSession:
    def __init__(self, api):
        self.api = api
        self.session_options = SessionOptions(api)

    def __enter__(self):
        so_ptr = self.session_options.__enter__()
        self.ptr = ctypes.c_void_p()

        self.api.CreateKernelSession(so_ptr.ptr, ctypes.byref(self.ptr), 12)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.api.ReleaseKernelSession(self.ptr)
        self.session_options.__exit__(exc_type, exc_val, exc_tb)


class ExecutableKernelContext:
    def __init__(self, api: ORTCAPIInterface, kernel_session: KernelSession,
                 name, op_type):
        self.kernel_session = kernel_session
        self.api = api
        self.n_inputs = 0
        self.n_outputs = 0
        self.name = name
        self.op_type = op_type
        self.dt_to_onnx_string = {
            v: k.upper()
            for k, v in ONNX_DTYPES_TO_DACE_TYPE_CLASS.items()
        }

    def __enter__(self):
        self.ptr = ctypes.c_void_p()
        self.api.CreateExecutableKernelContext(self.name.encode("ascii"),
                                               self.op_type.encode("ascii"),
                                               ctypes.byref(self.ptr))
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.api.ReleaseExecutableKernelContext(self.ptr)

    def add_input(self, dtype: dace.typeclass):
        self.n_inputs += 1
        self.api.ExecutableKernelContext_AddInput(
            self.ptr,
            f"ONNX_TENSOR_ELEMENT_DATA_TYPE_{self.dt_to_onnx_string[dtype]}")

    def add_output(self, dtype: dace.typeclass):
        self.n_outputs += 1
        self.api.ExecutableKernelContext_AddOutput(
            self.ptr,
            f"ONNX_TENSOR_ELEMENT_DATA_TYPE_{self.dt_to_onnx_string[dtype]}")

    def add_attribute(self, attr_name, attr_value,
                      attr_type: ONNXAttributeType):
        if attr_value is None:
            return
        attr_name = attr_name.encode("ascii")
        add_attr_function = getattr(
            self.api, f"ExecutableKernelContext_AddAttribute{attr_type.name}")

        if attr_type == ONNXAttributeType.Int or attr_type == ONNXAttributeType.Float or attr_type == ONNXAttributeType.String:
            add_attr_function(self.ptr, attr_name, attr_value)
        elif attr_type == ONNXAttributeType.Ints or attr_type == ONNXAttributeType.Floats or attr_type == ONNXAttributeType.Strings:
            get_elem_ctype = {
                ONNXAttributeType.Ints: ctypes.c_int64,
                ONNXAttributeType.Floats: ctypes.c_float,
                ONNXAttributeType.Strings: ctypes.c_char_p
            }
            elem_ctype = get_elem_ctype[attr_type]
            array_type = elem_ctype * len(attr_value)
            data_p = array_type(*attr_value)
            add_attr_function(self.ptr, attr_name, data_p, len(attr_value))
        elif attr_type == ONNXAttributeType.Tensor:

            data = [data_val.item() for data_val in np.nditer(attr_value)]
            ctype = np.ctypeslib.as_ctypes_type(attr_value.dtype)
            type_str = self.dt_to_onnx_string[dace.DTYPE_TO_TYPECLASS[
                attr_value.dtype.type]]
            type = f"ONNX_TENSOR_ELEMENT_DATA_TYPE_{type_str}"
            p_data = (ctype * len(data))(*data)
            p_data = ctypes.cast(p_data, ctypes.c_void_p)
            shape = (ctypes.c_int64 * len(attr_value.shape))(*attr_value.shape)
            add_attr_function(self.ptr, attr_name, p_data, len(data), shape,
                              len(attr_value.shape), type)

    def try_create_kernel(self, provider_id: int) -> "ExecutableKernel":
        return ExecutableKernel(self.api, self, provider_id)


class ExecutableKernel:
    def __init__(self, api, kernel_context: ExecutableKernelContext,
                 provider_id: int):
        self.api = api
        self.provider_id = provider_id
        self.kernel_context = kernel_context

    def __enter__(self):
        self.ptr = ctypes.c_void_p()
        self.api.CreateExecutableKernel(self.kernel_context.kernel_session.ptr,
                                        self.kernel_context.ptr,
                                        self.provider_id,
                                        ctypes.byref(self.ptr))
        return self

    def check_io_locations(self):

        outputs_on_cpu = []
        inputs_on_cpu = []

        for i in range(self.kernel_context.n_outputs):
            result = ctypes.c_int(-1)
            self.api.ExecutableKernel_IsOutputOnCpu(self.ptr, i,
                                                    ctypes.byref(result))
            if result == -1:
                raise ORTAPIError("Could not determine output storage of op")
            outputs_on_cpu.append(bool(result))

        for i in range(self.kernel_context.n_inputs):
            result = ctypes.c_int(-1)
            self.api.ExecutableKernel_IsInputOnCpu(self.ptr, i,
                                                   ctypes.byref(result))
            if result == -1:
                raise ORTAPIError("Could not determine output storage of op")
            inputs_on_cpu.append(bool(result))

        return inputs_on_cpu, outputs_on_cpu

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.api.ReleaseExecutableKernel(self.ptr)
