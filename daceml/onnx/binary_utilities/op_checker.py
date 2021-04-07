import ctypes
from typing import Optional, List, Tuple

import dace
import numpy as np
from dace.dtypes import DTYPE_TO_TYPECLASS

from daceml.onnx.converters import ONNX_DTYPES_TO_DACE_TYPE_CLASS
from daceml.onnx.schema import ONNXAttributeType
from daceml.ort_api import ORTAPIError, ORTCAPIInterface, OrtCUDAProviderOptions


class ONNXOpValidationError(ORTAPIError):
    pass


class OpChecker:
    def __init__(self, op_type: str, name: str, check_io_locations=False):

        self.n_outputs = 0
        self.n_inputs = 0
        self.check_io_locations = check_io_locations
        self.name = name.encode("ascii")
        self.op_type = op_type.encode("ascii")

    def __enter__(self):
        self._api = ORTCAPIInterface().__enter__()
        self._env = ctypes.c_void_p()
        self._api.CreateEnv("ORT_LOGGING_LEVEL_WARNING", b"ort_api",
                            ctypes.byref(self._env))

        self._session_options = ctypes.c_void_p()
        self._api.CreateSessionOptions(ctypes.byref(self._session_options))

        self._api.dll.OrtSessionOptionsAppendExecutionProvider_CPU(
            self._session_options, ctypes.c_int(0))
        if hasattr(self._api.dll,
                   "OrtSessionOptionsAppendExecutionProvider_CUDA"):
            cuda_opts = OrtCUDAProviderOptions(
                device_id=0,
                cudnn_conv_algo_search=self._api.get_enum_value("DEFAULT"),
                cuda_mem_limit=np.iinfo(ctypes.c_size_t).max,
                do_copy_in_default_stream=1,
                has_user_compute_stream=0,
                user_compute_stream=0)

            self._api.SessionOptionsAppendExecutionProvider_CUDA(
                self._session_options, ctypes.byref(cuda_opts))

        self._session = ctypes.c_void_p()
        self._api.CreateKernelSession(self._session_options,
                                      ctypes.byref(self._session), 12)

        self._context = ctypes.c_void_p()
        self._api.CreateExecutableKernelContext(self.name, self.op_type,
                                                ctypes.byref(self._context))

        self.dt_to_onnx_string = {
            v: k.upper()
            for k, v in ONNX_DTYPES_TO_DACE_TYPE_CLASS.items()
        }

        return self

    def try_create(self,
                   cuda=False) -> Optional[Tuple[List[bool], List[bool]]]:
        kernel = ctypes.c_void_p()
        self._api.CreateExecutableKernel(self._session,
                                         self._context, 1 if cuda else 0,
                                         ctypes.byref(kernel))

        if self.check_io_locations:
            outputs_on_cpu = []
            inputs_on_cpu = []
            for i in range(self.n_outputs):
                result = ctypes.c_int(-1)
                self._api.ExecutableKernel_IsOutputOnCpu(
                    kernel, i, ctypes.byref(result))
                if result == -1:
                    self._api.ReleaseExecutableKernel(kernel)
                    raise ONNXOpValidationError(
                        "Could not determine output storage of op")
                outputs_on_cpu.append(bool(result))

            for i in range(self.n_inputs):
                result = ctypes.c_int(-1)
                self._api.ExecutableKernel_IsInputOnCpu(
                    kernel, i, ctypes.byref(result))
                if result == -1:
                    self._api.ReleaseExecutableKernel(kernel)
                    raise ONNXOpValidationError(
                        "Could not determine output storage of op")
                inputs_on_cpu.append(bool(result))

            self._api.ReleaseExecutableKernel(kernel)
            return inputs_on_cpu, outputs_on_cpu
        else:
            self._api.ReleaseExecutableKernel(kernel)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if hasattr(self, "_context"):
            self._api.ReleaseExecutableKernelContext(self._context)
        if hasattr(self, "_session"):
            self._api.ReleaseKernelSession(self._session)
        if hasattr(self, "_session_options"):
            self._api.ReleaseSessionOptions(self._session_options)
        if hasattr(self, "_env"):
            self._api.ReleaseEnv(self._env)
        self._api.__exit__(exc_type=exc_type, exc_val=exc_val, exc_tb=exc_tb)

    def add_input(self, dtype: dace.typeclass):
        self.n_inputs += 1
        self._api.ExecutableKernelContext_AddInput(
            self._context,
            f"ONNX_TENSOR_ELEMENT_DATA_TYPE_{self.dt_to_onnx_string[dtype]}")

    def add_output(self, dtype: dace.typeclass):
        self.n_outputs += 1
        self._api.ExecutableKernelContext_AddOutput(
            self._context,
            f"ONNX_TENSOR_ELEMENT_DATA_TYPE_{self.dt_to_onnx_string[dtype]}")

    def add_attribute(self, attr_name, attr_value,
                      attr_type: ONNXAttributeType):
        if attr_value is None:
            return
        attr_name = attr_name.encode("ascii")
        add_attr_function = getattr(
            self._api, f"ExecutableKernelContext_AddAttribute{attr_type.name}")
        if attr_type == ONNXAttributeType.Int or attr_type == ONNXAttributeType.Float or attr_type == ONNXAttributeType.String:
            add_attr_function(self._context, attr_name, attr_value)
        elif attr_type == ONNXAttributeType.Ints or attr_type == ONNXAttributeType.Floats or attr_type == ONNXAttributeType.Strings:
            get_elem_ctype = {
                ONNXAttributeType.Ints: ctypes.c_int64,
                ONNXAttributeType.Floats: ctypes.c_float,
                ONNXAttributeType.Strings: ctypes.c_char_p
            }
            elem_ctype = get_elem_ctype[attr_type]
            array_type = elem_ctype * len(attr_value)
            data_p = array_type(*attr_value)
            add_attr_function(self._context, attr_name, data_p,
                              len(attr_value))
        elif attr_type == ONNXAttributeType.Tensor:

            data = [data_val.item() for data_val in np.nditer(attr_value)]
            ctype = np.ctypeslib.as_ctypes_type(attr_value.dtype)
            type_str = self.dt_to_onnx_string[DTYPE_TO_TYPECLASS[
                attr_value.dtype.type]]
            type = f"ONNX_TENSOR_ELEMENT_DATA_TYPE_{type_str}"
            p_data = (ctype * len(data))(*data)
            p_data = ctypes.cast(p_data, ctypes.c_void_p)
            shape = (ctypes.c_int64 * len(attr_value.shape))(*attr_value.shape)
            add_attr_function(self._context, attr_name, p_data, len(data),
                              shape, len(attr_value.shape), type)


def check_op(sdfg, state, node, cuda=False) -> Tuple[List[bool], List[bool]]:
    """ Check whether a ONNXOp node has an implementation in ORT """

    with OpChecker(node.schema.name, node.name,
                   check_io_locations=True) as checker:
        for attribute, onnx_attribute in node.schema.attributes.items():
            if hasattr(node, attribute):
                checker.add_attribute(attribute, getattr(node, attribute),
                                      onnx_attribute.type)

        for edge, is_input in node.iter_edges(state):
            edge_data = edge.data.data
            edge_dtype = sdfg.arrays[edge_data].dtype
            if is_input:
                checker.add_input(edge_dtype)
            else:
                checker.add_output(edge_dtype)

        return checker.try_create(cuda=cuda)
