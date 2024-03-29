import logging
from collections.abc import Iterable
from copy import deepcopy
from functools import reduce
from typing import Dict, Tuple, List, Optional

import dace
import dace.data as dt
import dace.library
import dace.sdfg.nodes as nd
import numpy as np
from dace import dtypes, SDFGState, SDFG
from dace.codegen import cppunparse
from dace.libraries.standard.nodes.code import _get_inputs_and_outputs

from daceml.onnx.binary_utilities.op_checker import check_op
from daceml.onnx.converters import clean_onnx_name, typeclass_to_onnx_str
from daceml.onnx.environments import ONNXRuntime, ONNXRuntimeCUDA
from daceml.onnx.nodes.node_utils import get_position
from daceml.onnx.schema import ONNXAttributeType, _ATTR_TYPE_TO_PYTHON_TYPE, ONNXAttribute
from daceml.ort_api import ORTAPIError
from daceml.util import utils

log = logging.getLogger(__name__)


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

    init_code = "{\n"

    def value_to_str(value):
        return '"{}"'.format(
            value) if attr.attribute_type == ONNXAttributeType.String else str(
                value)

    if attr.attribute_type in [
            ONNXAttributeType.Int, ONNXAttributeType.Float,
            ONNXAttributeType.String
    ]:
        assert_type(value, _ATTR_TYPE_TO_PYTHON_TYPE[attr.attribute_type])

        init_code += """
        __ort_check_status(__state->ort_api, __state->ort_api->ExecutableKernelContext_AddAttribute{type_str}({kernel_context}, "{name}", {value}));
        """.format(type_str=attr.attribute_type.name,
                   kernel_context=kernel_context,
                   name=attr.name,
                   value=value_to_str(value))
    elif attr.attribute_type in [
            ONNXAttributeType.Ints, ONNXAttributeType.Floats,
            ONNXAttributeType.Strings
    ]:
        if not isinstance(value, Iterable):
            raise ValueError(
                "Expected iterable value for attribute '{}', got {}".format(
                    attr.name, value))

        values = list(value)
        if attr.attribute_type == ONNXAttributeType.Ints:
            c_type = "int64_t"
        elif attr.attribute_type == ONNXAttributeType.Floats:
            c_type = "float"
        elif attr.attribute_type == ONNXAttributeType.String:
            c_type = "char*"

        init_code += "{type} values[{length}];\n".format(type=c_type,
                                                         length=len(values))

        for i, values_elem in enumerate(values):
            assert_type(i, _ATTR_TYPE_TO_PYTHON_TYPE[attr.attribute_type])
            init_code += "values[{i}] = {value};\n".format(
                i=i, value=value_to_str(values_elem))

        init_code += """
        __ort_check_status(__state->ort_api, __state->ort_api->ExecutableKernelContext_AddAttribute{type_str}({kernel_context}, "{name}", values, {length}));
        """.format(type_str=attr.attribute_type.name,
                   kernel_context=kernel_context,
                   name=attr.name,
                   length=len(values))

    elif attr.attribute_type == ONNXAttributeType.Tensor:
        assert_type(value, _ATTR_TYPE_TO_PYTHON_TYPE[attr.attribute_type])

        dace_typeclass = dtypes.DTYPE_TO_TYPECLASS[value.dtype.type]

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
        __ort_check_status(__state->ort_api, __state->ort_api->ExecutableKernelContext_AddAttributeTensor({kernel_context}, "{name}", static_cast<void*>(p_data), {data_length}, shape, {shape_length}, element_type));
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


def check_required_copies(
    node: nd.Node, state: SDFGState, sdfg: SDFG, outputs_on_host: List[bool],
    inputs_on_host: List[bool]
) -> Tuple[Dict[str, dtypes.StorageType], Dict[str, dtypes.StorageType]]:
    """ Check whether copies are required for all parameters.
        :param node: the node.
        :param state: the state.
        :param sdfg: the sdfg.
        :param outputs_on_host: boolean list, where the ith bool indicates if the ith output should be on host.
        :param inputs_on_host: boolean list, where the ith bool indicates if the ith input should be on host.
        :return: two dicts containing storage types for each of the connectors that require copies. The first
                 dict is for the inputs, the second is for the outputs.
    """

    # maps the connectors for which a copy will be required to the storage type required to be connected to the tasklet
    input_copy_required: Dict[str, dtypes.StorageType] = {}
    output_copy_required: Dict[str, dtypes.StorageType] = {}

    assert len(node.iter_outputs_in_onnx_order(state)) == len(outputs_on_host)
    assert len(node.iter_inputs_in_onnx_order(state)) == len(inputs_on_host)

    # check outputs
    for edge, output_on_host in zip(node.iter_outputs_in_onnx_order(state),
                                    outputs_on_host):
        # get the memlet for this output
        array = sdfg.arrays[edge.data.data]

        if output_on_host:
            is_device_mismatch = not dtypes.can_access(
                dtypes.ScheduleType.Default, array.storage)
        else:
            is_device_mismatch = not dtypes.can_access(
                dtypes.ScheduleType.GPU_Device, array.storage)

        if is_device_mismatch:
            # we need to insert a copy
            storage = dtypes.StorageType.CPU_Heap if output_on_host else dtypes.StorageType.GPU_Global
            output_copy_required[edge.src_conn] = storage

    # check inputs (same thing again)
    for edge, input_on_host in zip(node.iter_inputs_in_onnx_order(state),
                                   inputs_on_host):
        array = sdfg.arrays[edge.data.data]

        if input_on_host:
            is_device_mismatch = not dtypes.can_access(
                dtypes.ScheduleType.Default, array.storage)
        else:
            is_device_mismatch = not dtypes.can_access(
                dtypes.ScheduleType.GPU_Device, array.storage)

        if is_device_mismatch:
            # we need to insert a copy
            storage = dtypes.StorageType.CPU_Heap if input_on_host else dtypes.StorageType.GPU_Global
            input_copy_required[edge.dst_conn] = storage

    return input_copy_required, output_copy_required


def emit_setup_code_for_ortvalue(node: nd.CodeNode, parameter_name: str,
                                 edge_connector_name: str, desc: dt.Data,
                                 required_copy: Optional[dtypes.StorageType],
                                 is_input: bool, ort_value_name: str,
                                 connector_dict: dict) -> str:
    """ Emit the code that creates the OrtValue for a parameter. Also set the connector types on the parent node.

        :param node: the parent node that we are expanding
        :param parameter_name: the onnx name of the parameter.
        :param edge_connector_name: the name of the edge connector to the tasklet.
        :param desc: the dace input descriptor connected to this parameter.
        :param required_copy: the ``StorageType`` to copy to for this parameter, if a copy is required.
        :param is_input: whether the parameter is an input.
        :param ort_value_name: the name for the ort_value.
        :param connector_dict: either the input connector or output connector dict for the expanded node, depending on
                               whether this is an input or an output.
        :return: the code that creates the OrtValue for ``parameter_name``.
    """

    parent_connector_dict = node.in_connectors if is_input else node.out_connectors
    input_output_string = "input" if is_input else "output"
    code = ""

    if required_copy is not None:
        storage = required_copy
    else:
        storage = desc.storage

    if storage in [dtypes.StorageType.Default, dtypes.StorageType.CPU_Heap]:
        mem_info = "__state->ort_cpu_mem_info"
    elif storage is dtypes.StorageType.GPU_Global:
        mem_info = "__state->ort_cuda_mem_info"
    elif storage is dtypes.StorageType.CPU_Pinned:
        mem_info = "__state->ort_cuda_pinned_mem_info"
    else:
        raise ValueError(
            "Unsupported storage type {} for input to ONNX node".format(
                desc.storage))

    if isinstance(desc, dt.Scalar):

        on_gpu = storage is dtypes.StorageType.GPU_Global

        code += """
        OrtValue* {ort_value_name};
        __ort_check_status(__state->ort_api, __state->ort_api->CreateTensorWithDataAsOrtValue(
            {mem_info},
            {maybe_ref}{edge_connector_name},
            {data_size} * sizeof({ctype}),
            nullptr,
            0,
            ONNX_TENSOR_ELEMENT_DATA_TYPE_{type_str},
            &{ort_value_name}
        ));
        """.format(mem_info=mem_info,
                   edge_connector_name=edge_connector_name,
                   data_size=cppunparse.pyexpr2cpp(str(utils.prod(
                       desc.shape))),
                   ctype=desc.dtype.ctype,
                   type_str=typeclass_to_onnx_str(desc.dtype).upper(),
                   ort_value_name=ort_value_name,
                   maybe_ref="" if on_gpu else "&")

        if on_gpu:
            connector_dict[edge_connector_name] = dace.pointer(desc.dtype)
            parent_connector_dict[parameter_name] = dace.pointer(desc.dtype)
        else:
            connector_dict[edge_connector_name] = desc.dtype
            parent_connector_dict[parameter_name] = desc.dtype
    elif isinstance(desc, dt.Array):

        # setup dims array
        code += """
        int64_t {input_output_string}_{parameter_name}_dims[{dims_size}] = {{{dims}}};
        """.format(input_output_string=input_output_string,
                   parameter_name=parameter_name,
                   dims_size=len(desc.shape),
                   dims=", ".join(str(s) for s in desc.shape))

        data = "const_cast < void * > (reinterpret_cast < const void * > ({}))".format(
            edge_connector_name)

        code += """
        OrtValue* {ort_value_name};
        __ort_check_status(__state->ort_api, __state->ort_api->CreateTensorWithDataAsOrtValue(
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
                   data_size=cppunparse.pyexpr2cpp(str(utils.prod(
                       desc.shape))),
                   ctype=desc.dtype.ctype,
                   dims_size=len(desc.shape),
                   type_str=typeclass_to_onnx_str(desc.dtype).upper(),
                   ort_value_name=ort_value_name)
        connector_dict[edge_connector_name] = dace.pointer(desc.dtype)
        parent_connector_dict[parameter_name] = dace.pointer(desc.dtype)
    else:
        raise NotImplementedError(
            "Data-descriptor type {} not supported for ONNX nodes".format(
                type(desc)))
    return code


def expand_node(node, state, sdfg):
    if not ONNXRuntime.is_installed():
        raise RuntimeError(
            "ONNXRuntime is not installed, cannot expand node "
            "{}. You can either install ONNX Runtime as described in the "
            "docs, or add a pure node implementation for the {} op.".format(
                node, node.schema.name))

    inputs, outputs = _get_inputs_and_outputs(sdfg, state, node)

    unique_id = "{}_{}_{}_{}".format(clean_onnx_name(node.name), sdfg.sdfg_id,
                                     sdfg.node_id(state), state.node_id(node))

    # check if ORT supports CUDA for this node using the op checker
    ###############################################################

    # Default: all parameters are on CPU if we execute using cpu
    outputs_on_host = [True for _ in range(len(outputs))]
    inputs_on_host = [True for _ in range(len(inputs))]

    actual_node_schedule = node.schedule
    if node.schedule == dtypes.ScheduleType.CPU_Multicore or node.schedule == dtypes.ScheduleType.Default:
        provider_index = 0
    elif node.schedule in dtypes.GPU_SCHEDULES + [
            dtypes.ScheduleType.GPU_Default
    ]:
        provider_index = 1
        try:
            # the ith position indicates whether the ith output is in host memory
            inputs_on_host, outputs_on_host = check_op(sdfg,
                                                       state,
                                                       node,
                                                       cuda=True)

        except ORTAPIError as e:
            # fallback to CPU
            log.warning("Falling back to CPU for node {}. Reason:\n{}".format(
                node.name, str(e)))
            provider_index = 0
    else:
        raise NotImplementedError(
            "ORT expansion for schedule '{}' is not implemented".format(
                node.schedule))

    # check if we need to insert device copies
    ##########################################

    input_copy_required, output_copy_required = check_required_copies(
        node, state, sdfg, outputs_on_host, inputs_on_host)

    # begin codegen
    ##########################################
    tasklet_setup_code = ""
    tasklet_code = ""
    tasklet_cleanup_code = ""
    env_init_code = ("""
    __ort_check_status(__state->ort_api, __state->ort_api->CreateExecutableKernelContext("{name}", "{op_type}", &__state->ort_context_{name}));
    """.format(name=unique_id, op_type=node.schema.name))

    # emit code for inputs and outputs
    ##########################################
    in_connectors = {}
    out_connectors = {}

    for edge, is_input in node.iter_edges(state):
        parameter_name = edge.dst_conn if is_input else edge.src_conn

        input_output_string = "input" if is_input else "output"
        memlet = edge.data
        desc = sdfg.arrays[memlet.data]
        env_init_code += """
        __ort_check_status(__state->ort_api, __state->ort_api->ExecutableKernelContext_Add{input_output_string}(__state->ort_context_{id}, ONNX_TENSOR_ELEMENT_DATA_TYPE_{type_string}));
        """.format(id=unique_id,
                   type_string=typeclass_to_onnx_str(desc.dtype).upper(),
                   parameter_name=parameter_name,
                   input_output_string=input_output_string.capitalize())

        ort_value_name = "ort_value_{input_output_string}_{parameter_name}".format(
            input_output_string=input_output_string,
            parameter_name=parameter_name)

        # We always emit a NestedSDFG, so the edge connector names must be prefixed (otherwise there would be a conflict
        # of names).
        edge_connector_name = "__" + parameter_name

        copy_options_dict = input_copy_required if is_input else output_copy_required
        copy_options = copy_options_dict.get(parameter_name, None)

        tasklet_setup_code += emit_setup_code_for_ortvalue(
            node=node,
            parameter_name=parameter_name,
            edge_connector_name=edge_connector_name,
            desc=desc,
            required_copy=copy_options,
            is_input=is_input,
            ort_value_name=ort_value_name,
            connector_dict=in_connectors if is_input else out_connectors)

        tasklet_code += "__ort_check_status(__state->ort_api, __state->ort_api->ExecutableKernel_Set" \
                        "{input_output_string_capital}(" \
                        "__state->ort_kernel_{unique_id}, {position}, {ort_value_name}));\n".format(
            input_output_string_capital=input_output_string.
                capitalize(),
            ort_value_name=ort_value_name,
            unique_id=unique_id,
            position=get_position(node.schema, is_input,
                                  parameter_name))

        tasklet_cleanup_code += "__state->ort_api->ReleaseValue(" \
                                "ort_value_{input_output_string}_{parameter_name});\n".format(
            input_output_string=input_output_string,
            parameter_name=parameter_name)

    env_init_code += "\n"

    for name, attr in node.schema.attributes.items():
        if hasattr(node, name):
            env_init_code += _gen_attr_init_code(
                "__state->ort_context_{}".format(unique_id),
                node.schema.attributes[name], getattr(node, name))

    env_finalize_code = """
        __state->ort_api->ReleaseExecutableKernel(__state->ort_kernel_{});\n
        __state->ort_api->ReleaseExecutableKernelContext(__state->ort_context_{});\n
    """.format(unique_id, unique_id)

    if logging.root.level <= logging.DEBUG:
        tasklet_code += 'fprintf(stderr, "Launching {}\\n");\n'.format(
            unique_id)

    tasklet_code += "__ort_check_status(__state->ort_api, __state->ort_api->ExecutableKernel_Compute(__state->ort_kernel_{}));\n".format(
        unique_id)

    tasklet_code = tasklet_setup_code + tasklet_code + tasklet_cleanup_code

    if ONNXRuntimeCUDA.use_streams:
        raise ValueError("Currently not supported anymore.")

    env_init_code += f"""
                    __ort_check_status(__state->ort_api, __state->ort_api->CreateExecutableKernel(
                    __state->ort_session, __state->ort_context_{unique_id}, /*provider_index=*/{provider_index},
                     &__state->ort_kernel_{unique_id}));
                    """

    env_init_code = "{\n" + env_init_code + "\n}"
    env_finalize_code = "{\n" + env_finalize_code + "\n}"

    tasklet = nd.Tasklet(
        unique_id + '_onnx_code',
        in_connectors,
        out_connectors,
        tasklet_code,
        state_fields=[
            "OrtExecutableKernelContext *ort_context_{};\n".format(unique_id),
            "OrtExecutableKernel *ort_kernel_{};\n".format(unique_id),
        ],
        code_init=env_init_code,
        code_exit=env_finalize_code,
        language=dace.dtypes.Language.CPP)

    env = ONNXRuntimeCUDA if node.schedule in dtypes.GPU_SCHEDULES + [
        dtypes.ScheduleType.GPU_Default
    ] else ONNXRuntime
    tasklet.environments = {env.full_class_path()}

    nsdfg = dace.SDFG("nested_{}".format(unique_id))
    nstate = nsdfg.add_state()
    ntasklet = deepcopy(tasklet)

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
                "Unsupported data type {} connected to an ONNX tasklet".format(
                    type(desc)))

        copy_options_dict = input_copy_required if is_input else output_copy_required
        # handle parameters for which no copies are required
        if parameter_name not in copy_options_dict:
            copied_memlet = deepcopy(memlet)
            copied_memlet.data = parameter_name
            if is_input:
                access = nstate.add_read(parameter_name)
                nstate.add_edge(access, None, ntasklet, "__" + parameter_name,
                                copied_memlet)
            else:
                access = nstate.add_write(parameter_name)
                nstate.add_edge(ntasklet, "__" + parameter_name, access, None,
                                copied_memlet)
            continue

        # add the copy of the descriptor
        copy_desc = deepcopy(desc)

        copy_desc.transient = True
        copy_desc.storage = copy_options_dict[parameter_name]
        # there can be name conflicts here if an input is given to multiple
        # connectors. We could technically share the copied result, but it's
        # likely not worth the effort since these are just scalars.
        copy_name = nsdfg.add_datadesc("copy_" + memlet.data,
                                       copy_desc,
                                       find_new_name=True)

        nmemlet = deepcopy(memlet)
        nmemlet_copy = deepcopy(memlet)
        nmemlet_copy.data = copy_name
        nmemlet.data = copy_name
        if is_input:
            access = nstate.add_read(parameter_name)
            access_copy = nstate.add_access(copy_name)
            nstate.add_edge(access, None, access_copy, None, nmemlet_copy)
            nstate.add_edge(access_copy, None, ntasklet, "__" + parameter_name,
                            nmemlet)
        else:
            access = nstate.add_write(parameter_name)
            access_copy = nstate.add_access(copy_name)
            nstate.add_edge(ntasklet, "__" + parameter_name, access_copy, None,
                            nmemlet)
            nstate.add_edge(access_copy, None, access, None, nmemlet_copy)

    return nsdfg
