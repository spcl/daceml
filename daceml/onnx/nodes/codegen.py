import logging
from collections import Iterable, defaultdict
from copy import deepcopy
from functools import reduce

import numpy as np

import dace
import dace.data as dt
import dace.sdfg.nodes as nd
from dace import SDFG, ScheduleType, StorageType
from dace.dtypes import DTYPE_TO_TYPECLASS, can_access
from dace.libraries.standard.nodes.code import _get_inputs_and_outputs

from daceml.onnx.check_impl import check_op, ONNXOpValidationError
from daceml.onnx.converters import ONNX_DTYPES_TO_DACE_TYPE_CLASS, clean_onnx_name, typeclass_to_onnx_str
from daceml.onnx.nodes.node_utils import get_position
from daceml.onnx.schema import ONNXAttributeType, _ATTR_TYPE_TO_PYTHON_TYPE, ONNXAttribute

log = logging.getLogger(__name__)


def _add_ort_init_code(sdfg: SDFG):
    """ Add onnxruntime initialization code to the SDFG if required """

    if "OrtKernelSession" not in sdfg.global_code['frame'].as_string:
        sdfg.append_global_code("""
        // Start global ORT setup
        const OrtApi* __ort_api = OrtGetApiBase()->GetApi(ORT_API_VERSION);

        // helper function to check for status
        void __ort_check_status(OrtStatus* status)
        {
            if (status != NULL) {
                const char* msg = __ort_api->GetErrorMessage(status);
                fprintf(stderr, "%s\\n", msg);
                __ort_api->ReleaseStatus(status);
                exit(1);
            }
        }
        OrtEnv* __ort_env;
        OrtKernelSession* __ort_session;
        OrtSessionOptions* __ort_session_options;

        OrtMemoryInfo* __ort_cpu_mem_info;
        """)

        sdfg.append_init_code("""
        __ort_check_status(__ort_api->CreateCpuMemoryInfo(OrtDeviceAllocator, OrtMemTypeDefault, &__ort_cpu_mem_info));
        __ort_check_status(__ort_api->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "dace_graph", &__ort_env));
        __ort_check_status(__ort_api->CreateSessionOptions(&__ort_session_options));
        __ort_check_status(OrtSessionOptionsAppendExecutionProvider_CPU(__ort_session_options, /*use_arena=*/0));
        """)

        session_cleanup_code = """
        __ort_api->ReleaseMemoryInfo(__ort_cpu_mem_info);
        __ort_api->ReleaseKernelSession(__ort_session);
        __ort_api->ReleaseSessionOptions(__ort_session_options);
        __ort_api->ReleaseEnv(__ort_env);
        """

        if any(
                hasattr(node, "schedule")
                and node.schedule in [ScheduleType.GPU_Device, ScheduleType.GPU_Persistent, ScheduleType.GPU_Default]
                for state in sdfg.nodes() for node in state.nodes()):
            # if the SDFG contains a GPU node, add the CUDA provider and the memory_info
            sdfg.append_global_code("OrtMemoryInfo* __ort_cuda_mem_info;\n")
            sdfg.append_global_code(
                "OrtMemoryInfo* __ort_cuda_pinned_mem_info;\n")
            sdfg.append_init_code("""
            __ort_check_status(__ort_api->CreateMemoryInfo("Cuda", /*allocator_type=*/OrtDeviceAllocator, /*device=*/0, /*mem_type=*/OrtMemTypeDefault, &__ort_cuda_mem_info));
            __ort_check_status(__ort_api->CreateMemoryInfo("CudaPinned", /*allocator_type=*/OrtDeviceAllocator, /*device=*/0, /*mem_type=*/OrtMemTypeCPU, &__ort_cuda_pinned_mem_info));
            __ort_check_status(OrtSessionOptionsAppendExecutionProvider_CUDA(__ort_session_options, /*device=*/0));
            """)
            session_cleanup_code = ("""
            __ort_api->ReleaseMemoryInfo(__ort_cuda_mem_info);
            __ort_api->ReleaseMemoryInfo(__ort_cuda_pinned_mem_info);
            """ + session_cleanup_code)

        sdfg.append_global_code("// End global ORT setup\n")
        sdfg.prepend_exit_code(session_cleanup_code)
        sdfg.append_init_code("""
        __ort_check_status(__ort_api->CreateKernelSession(__ort_session_options, &__ort_session, 12));
        """)


def _gen_attr_init_code(kernel_context: str, attr: ONNXAttribute,
                        value) -> str:
    """ Get the code to setup an attribute on an onnx::NodeProto
        :param kernel_context: the variable name of the kernel context
        :param attr: the attribute to setup
    """
    if value is None:
        return ""

    def assert_type(val, expected_type):
        if not isinstance(val, expected_type):
            raise ValueError(
                "Expected value of attribute '{}' to have type {}, got {} (type {})"
                .format(attr.name, expected_type, val, type(val)))

    init_code = """{{
    // Setup attribute {name}
    """.format(name=attr.name)

    def value_to_str(value):
        return '"{}"'.format(
            value) if attr.type == ONNXAttributeType.String else str(value)

    if attr.type in [
            ONNXAttributeType.Int, ONNXAttributeType.Float,
            ONNXAttributeType.String
    ]:
        assert_type(value, _ATTR_TYPE_TO_PYTHON_TYPE[attr.type])

        init_code += """
        __ort_check_status(__ort_api->ExecutableKernelContext_AddAttribute{type_str}({kernel_context}, "{name}", {value}));
        """.format(type_str=attr.type.name,
                   kernel_context=kernel_context,
                   name=attr.name,
                   value=value_to_str(value))
    elif attr.type in [
            ONNXAttributeType.Ints, ONNXAttributeType.Floats,
            ONNXAttributeType.Strings
    ]:
        if not isinstance(value, Iterable):
            raise ValueError(
                "Expected iterable value for attribute '{}', got {}".format(
                    attr.name, value))

        values = list(value)
        if attr.type == ONNXAttributeType.Ints:
            c_type = "int64_t"
        elif attr.type == ONNXAttributeType.Floats:
            c_type = "float"
        elif attr.type == ONNXAttributeType.String:
            c_type = "char*"

        init_code += "{type} values[{length}];\n".format(type=c_type,
                                                         length=len(values))

        for i, values_elem in enumerate(values):
            assert_type(i, _ATTR_TYPE_TO_PYTHON_TYPE[attr.type])
            init_code += "values[{i}] = {value};\n".format(
                i=i, value=value_to_str(values_elem))

        init_code += """
        __ort_check_status(__ort_api->ExecutableKernelContext_AddAttribute{type_str}({kernel_context}, "{name}", values, {length}));
        """.format(type_str=attr.type.name,
                   kernel_context=kernel_context,
                   name=attr.name,
                   length=len(values))

    elif attr.type == ONNXAttributeType.Tensor:
        assert_type(value, _ATTR_TYPE_TO_PYTHON_TYPE[attr.type])

        dace_typeclass = DTYPE_TO_TYPECLASS[value.dtype.type]

        supported_types = {
            dace.float16: dace.float32,
            dace.float32: dace.float32,
            dace.float64: dace.float64,
            dace.int8: dace.int8,
            dace.int16: dace.int16,
            dace.int32: dace.int32,
            dace.int64: dace.int64,
            dace.uint8: dace.uint8,
            dace.uint16: dace.uint16,
            dace.uint32: dace.uint32,
            dace.uint64: dace.uint64
        }

        if dace_typeclass not in supported_types:
            raise NotImplementedError(
                "ONNX support for type {} has not been implemented for ONNX Tensor attributes (at attribute with name {})"
                .format(value.dtype.type, attr.name))

        type_to_generate = supported_types[dace_typeclass]

        init_code += """
        ONNXTensorElementDataType element_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_{};
        """.format(typeclass_to_onnx_str(type_to_generate).upper())
        init_code += "int64_t shape[{}];\n".format(len(value.shape))
        for i, dim in enumerate(value.shape):
            init_code += "shape[{}] = {};\n".format(i, dim)

        init_code += "{} p_data[{}];\n".format(type_to_generate.ctype,
                                               value.size)
        for i, data_val in enumerate(np.nditer(value)):
            data_val = data_val.item()
            init_code += "p_data[{}] = {};\n".format(i, data_val)

        init_code += """
        __ort_check_status(__ort_api->ExecutableKernelContext_AddAttributeTensor({kernel_context}, "{name}", static_cast<void*>(p_data), {data_length}, shape, {shape_length}, element_type));
        """.format(kernel_context=kernel_context,
                   name=attr.name,
                   data_length=value.size,
                   shape_length=len(value.shape))

    else:
        raise NotImplementedError(
            "Got unsupported attribute type {} for '{}'".format(
                attr.dtype, attr.name))
    init_code += "}\n"
    return init_code


def expand_node(node, state, sdfg):
    inputs, outputs = _get_inputs_and_outputs(sdfg, state, node)

    unique_id = "{}_{}_{}_{}".format(clean_onnx_name(node.name), sdfg.sdfg_id,
                                     sdfg.node_id(state), state.node_id(node))
    _add_ort_init_code(sdfg)

    sdfg.append_global_code(
        "OrtExecutableKernel *__ort_kernel_{};\n".format(unique_id))
    sdfg.append_global_code(
        "OrtExecutableKernelContext *__ort_context_{};\n".format(unique_id))

    sdfg.append_init_code("""
    {{
    // Setup for {name}
    __ort_check_status(__ort_api->CreateExecutableKernelContext("{name}", "{op_type}", &__ort_context_{name}));
    """.format(name=unique_id, op_type=node.schema.name))

    # check if ORT supports CUDA for this node
    ##########################################

    # Default: all parameters are on CPU if we execute using cpu
    outputs_on_host = [True for _ in range(len(outputs))]
    inputs_on_host = [True for _ in range(len(inputs))]

    actual_node_schedule = node.schedule
    if node.schedule == ScheduleType.CPU_Multicore or node.schedule == ScheduleType.Default:
        provider_index = 0
    elif node.schedule in [ScheduleType.GPU_Device, ScheduleType.GPU_Persistent, ScheduleType.GPU_Default]:
        provider_index = 1
        try:
            # the ith position indicates whether the ith output is in host memory
            inputs_on_host, outputs_on_host = check_op(sdfg,
                                                       state,
                                                       node,
                                                       cuda=True)

        except ONNXOpValidationError as e:
            # fallback to CPU
            print("Falling back to CPU for node {}. Reason:\n{}".format(
                node.name, str(e)))
            provider_index = 0
            actual_node_schedule = ScheduleType.CPU_Default
    else:
        raise NotImplementedError(
            "ORT expansion for schedule '{}' is not implemented".format(
                node.schedule))

    # check if we need to insert device copies
    ##########################################

    # maps the connectors for which a copy will be required to the storage type required to be connected to the tasklet
    input_copy_required = defaultdict(dict)
    output_copy_required = defaultdict(dict)

    assert len(node.iter_outputs_in_onnx_order(state)) == len(outputs_on_host)
    assert len(node.iter_inputs_in_onnx_order(state)) == len(inputs_on_host)

    # check outputs
    for edge, output_on_host in zip(node.iter_outputs_in_onnx_order(state),
                                    outputs_on_host):
        # get the memlet for this output
        array = sdfg.arrays[edge.data.data]

        if output_on_host:
            is_device_mismatch = not can_access(ScheduleType.Default,
                                                array.storage)
        else:
            is_device_mismatch = not can_access(ScheduleType.GPU_Device,
                                                array.storage)

        if isinstance(
                array,
                dt.Scalar) and actual_node_schedule in [ScheduleType.GPU_Device, ScheduleType.GPU_Persistent, ScheduleType.GPU_Default]:
            # ORT kernels expect scalars to be cudaMalloced. We will copy during expansion to enforce this
            is_device_mismatch = True
            output_copy_required[edge.src_conn]['copy_to_array'] = True

        if is_device_mismatch:
            # we need to insert a copy
            output_copy_required[edge.src_conn][
                'storage'] = StorageType.CPU_Heap if output_on_host else StorageType.GPU_Global

    # check inputs (same thing again)
    for edge, input_on_host in zip(node.iter_inputs_in_onnx_order(state),
                                   inputs_on_host):
        array = sdfg.arrays[edge.data.data]

        if input_on_host:
            is_device_mismatch = not can_access(ScheduleType.Default,
                                                array.storage)
        else:
            is_device_mismatch = not can_access(ScheduleType.GPU_Device,
                                                array.storage)

        if isinstance(
                array,
                dt.Scalar) and actual_node_schedule in [ScheduleType.GPU_Device, ScheduleType.GPU_Persistent, ScheduleType.GPU_Default]:
            # ORT kernels expect scalars to be cudaMalloced. We will copy during expansion to enforce this
            is_device_mismatch = True
            input_copy_required[edge.dst_conn]['copy_to_array'] = True

        if is_device_mismatch:
            # we need to insert a copy
            input_copy_required[edge.dst_conn][
                'storage'] = StorageType.CPU_Heap if input_on_host else StorageType.GPU_Global

    # begin codegen
    ##########################################
    tasklet_setup_code = ""
    tasklet_code = ""
    tasklet_cleanup_code = ""

    reversed_onnx_dtype_map = {
        v: k
        for k, v in ONNX_DTYPES_TO_DACE_TYPE_CLASS.items()
    }

    # emit code for inputs and outputs
    ##########################################
    in_connectors = {}
    out_connectors = {}

    for edge, is_input in node.iter_edges(state):

        parameter_name = edge.dst_conn if is_input else edge.src_conn

        if len(output_copy_required) != 0 or len(input_copy_required) != 0:
            edge_connector_name = "_conn_" + parameter_name
        else:
            edge_connector_name = parameter_name

        input_output_string = "input" if is_input else "output"
        connector_dict = in_connectors if is_input else out_connectors
        memlet = edge.data
        desc = sdfg.arrays[memlet.data]
        sdfg.append_init_code("""
        // Add parameter {parameter_name}
        __ort_check_status(__ort_api->ExecutableKernelContext_Add{input_output_string}(__ort_context_{id}, ONNX_TENSOR_ELEMENT_DATA_TYPE_{type_string}));
        """.format(id=unique_id,
                   type_string=reversed_onnx_dtype_map[desc.dtype].upper(),
                   parameter_name=parameter_name,
                   input_output_string=input_output_string.capitalize()))

        ort_value_name = "ort_value_{input_output_string}_{parameter_name}".format(
            input_output_string=input_output_string,
            parameter_name=parameter_name)

        copy_to_array = (
            (parameter_name in output_copy_required
             and 'copy_to_array' in output_copy_required[parameter_name])
            or (parameter_name in input_copy_required
                and 'copy_to_array' in input_copy_required[parameter_name]))
        if desc.storage in [StorageType.Default, StorageType.CPU_Heap]:
            mem_info = "__ort_cpu_mem_info"
        elif desc.storage == StorageType.GPU_Global:
            mem_info = "__ort_cuda_mem_info"
        elif desc.storage == StorageType.CPU_Pinned:
            mem_info = "__ort_cuda_pinned_mem_info"
        else:
            raise ValueError(
                "Unsupported storage type {} for input to ONNX node".format(
                    desc.storage))
        if (isinstance(desc, dt.Scalar) and
                # when copying to array, the ort value is not a scalar but an array
                not copy_to_array):

            tasklet_setup_code += """
            OrtValue* {ort_value_name};
            __ort_check_status(__ort_api->CreateTensorWithDataAsOrtValue(
                {mem_info},
                &{edge_connector_name},
                {data_size} * sizeof({ctype}),
                nullptr,
                0,
                ONNX_TENSOR_ELEMENT_DATA_TYPE_{type_str},
                &{ort_value_name}
            ));
            """.format(input_output_string=input_output_string,
                       mem_info=mem_info,
                       edge_connector_name=edge_connector_name,
                       data_size=reduce(lambda x, y: x * y, desc.shape),
                       ctype=desc.dtype.ctype,
                       type_str=reversed_onnx_dtype_map[desc.dtype].upper(),
                       ort_value_name=ort_value_name)
            connector_dict[parameter_name] = None

        elif isinstance(desc, dt.Array) or copy_to_array:

            # when we copy a scalar to an array, that scalar ofc has shape []
            dims = [] if copy_to_array else desc.shape

            # setup dims array
            tasklet_setup_code += """
            int64_t {input_output_string}_{parameter_name}_dims[{dims_size}] = {{{dims}}};
            """.format(input_output_string=input_output_string,
                       parameter_name=parameter_name,
                       dims_size=len(dims),
                       dims=", ".join(str(s) for s in dims))

            connector_dict[parameter_name] = dace.pointer(desc.dtype)
            data = "const_cast < void * > (reinterpret_cast < const void * > ({}))".format(
                edge_connector_name)

            tasklet_setup_code += """
            OrtValue* {ort_value_name};
            __ort_check_status(__ort_api->CreateTensorWithDataAsOrtValue(
                {mem_info},
                {data},
                {data_size} * sizeof({ctype}),
                {input_output_string}_{parameter_name}_dims,
                {dims_size},
                ONNX_TENSOR_ELEMENT_DATA_TYPE_{type_str},
                &{ort_value_name}
            ));
            """.format(input_output_string=input_output_string,
                       data=data,
                       mem_info=mem_info,
                       parameter_name=parameter_name,
                       data_size=reduce(lambda x, y: x * y, desc.shape),
                       ctype=desc.dtype.ctype,
                       dims_size=len(dims),
                       type_str=reversed_onnx_dtype_map[desc.dtype].upper(),
                       ort_value_name=ort_value_name)
        else:
            raise NotImplementedError(
                "Data-descriptor type {} not supported for ONNX nodes".format(
                    type(desc)))


        tasklet_code += "__ort_check_status(__ort_api->ExecutableKernel_Set{input_output_string_capital}(" \
                        "__ort_kernel_{unique_id}, {position}, {ort_value_name}));\n".format(
            input_output_string_capital=input_output_string.
                capitalize(),
            ort_value_name=ort_value_name,
            unique_id=unique_id,
            position=get_position(node.schema, is_input,
                                  parameter_name))

        tasklet_cleanup_code += "__ort_api->ReleaseValue(ort_value_{input_output_string}_{parameter_name});\n".format(
            input_output_string=input_output_string,
            parameter_name=parameter_name)

    sdfg.append_init_code("// Setup attributes\n")

    for name, attr in node.schema.attributes.items():
        if hasattr(node, name):
            sdfg.append_init_code(
                _gen_attr_init_code("__ort_context_{}".format(unique_id),
                                    node.schema.attributes[name],
                                    getattr(node, name)))

    sdfg.prepend_exit_code(
        "__ort_api->ReleaseExecutableKernelContext(__ort_context_{});\n".
        format(unique_id))
    sdfg.prepend_exit_code(
        "__ort_api->ReleaseExecutableKernel(__ort_kernel_{});\n".format(
            unique_id))

    if logging.root.level <= logging.DEBUG:
        tasklet_code += 'fprintf(stderr, "Launching {}\\n");\n'.format(
            unique_id)

    tasklet_code += "__ort_check_status(__ort_api->ExecutableKernel_Compute(__ort_kernel_{}));\n".format(
        unique_id)

    sdfg.append_init_code(
        "__ort_check_status(__ort_api->CreateExecutableKernel("
        "__ort_session, __ort_context_{id}, /*provider_index=*/{provider_index}, &__ort_kernel_{id}));\n"
        .format(provider_index=provider_index, id=unique_id))
    sdfg.append_init_code("}} // end setup for context_{}".format(unique_id))

    tasklet_code = tasklet_setup_code + tasklet_code + tasklet_cleanup_code
    tasklet = nd.Tasklet(unique_id + '_onnx_code',
                         in_connectors,
                         out_connectors,
                         tasklet_code,
                         language=dace.dtypes.Language.CPP)
    tasklet.environments = {"ONNXRuntime"}

    if len(output_copy_required) != 0 or len(input_copy_required) != 0:
        nsdfg = dace.SDFG("nested_{}".format(unique_id))
        nstate = nsdfg.add_state()
        ntasklet = deepcopy(tasklet)

        # add a prefix to connectors to prevent shadowing of array names
        ntasklet.in_connectors = {
            "_conn_" + k: v
            for k, v in tasklet.in_connectors.items()
        }
        ntasklet.out_connectors = {
            "_conn_" + k: v
            for k, v in tasklet.out_connectors.items()
        }

        nstate.add_node(ntasklet)

        for edge, is_input in node.iter_edges(state):
            parameter_name = edge.dst_conn if is_input else edge.src_conn

            memlet = edge.data
            desc = sdfg.arrays[memlet.data]

            # add the original array
            original_desc = deepcopy(desc)
            original_desc.transient = False
            nsdfg.add_datadesc(parameter_name, original_desc)
            if not (isinstance(desc, dt.Array) or isinstance(desc, dt.Scalar)):
                raise ValueError(
                    "Unsupported data type {} connected to an ONNX tasklet".
                    format(type(desc)))

            if parameter_name not in (input_copy_required
                                      if is_input else output_copy_required):
                if is_input:
                    access = nstate.add_read(parameter_name)
                    nstate.add_edge(access, None, ntasklet,
                                    "_conn_" + parameter_name,
                                    nsdfg.make_array_memlet(parameter_name))
                else:
                    access = nstate.add_write(parameter_name)
                    nstate.add_edge(ntasklet, "_conn_" + parameter_name,
                                    access, None,
                                    nsdfg.make_array_memlet(parameter_name))
                continue

            copy_options = input_copy_required[
                parameter_name] if is_input else output_copy_required[
                    parameter_name]

            # add the copy of the descriptor
            if 'copy_to_array' in copy_options:
                copy_desc = dt.Array(shape=[1], dtype=desc.dtype)
            else:
                copy_desc = deepcopy(desc)

            copy_desc.transient = True
            copy_desc.storage = copy_options['storage']
            nsdfg.add_datadesc("copy_" + memlet.data, copy_desc)

            nmemlet = deepcopy(memlet)
            nmemlet.data = "copy_" + nmemlet.data
            if is_input:
                access = nstate.add_read(parameter_name)
                access_copy = nstate.add_access("copy_" + memlet.data)
                nstate.add_edge(access, None, access_copy, None,
                                nsdfg.make_array_memlet("copy_" + memlet.data))
                nstate.add_edge(access_copy, None, ntasklet,
                                "_conn_" + parameter_name, nmemlet)
            else:
                access = nstate.add_write(parameter_name)
                access_copy = nstate.add_access("copy_" + memlet.data)
                nstate.add_edge(ntasklet, "_conn_" + parameter_name,
                                access_copy, None, nmemlet)
                nstate.add_edge(access_copy, None, access, None,
                                nsdfg.make_array_memlet("copy_" + memlet.data))

        return nsdfg

    else:
        return tasklet
