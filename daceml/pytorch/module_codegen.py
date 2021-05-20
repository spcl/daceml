""" Code generation for PyTorch C++ dispatched operators. """
import copy
import dataclasses
import os
import operator
import itertools
from typing import List, Tuple, Callable

import numpy as np
import torch
from dace import dtypes as dt, data, ctypes
import dace.library
from dace.codegen import targets, compiler
from dace.codegen.codeobject import CodeObject
from dace.codegen.compiled_sdfg import CompiledSDFG

from daceml.onnx.converters import clean_onnx_name
from daceml.pytorch.environments import PyTorch
from daceml.util import is_cuda, platform_library_name


@dataclasses.dataclass
class CompiledTorchFunction:
    """ A tuple holding the context for an executable function """
    function: Callable  #: the torch callable function
    compiled_sdfg: CompiledSDFG  #: the compiled SDFG holding the state
    ptr: torch.Tensor  #: the state ptr argument to use when calling the function


def get_arglist(
        module: 'daceml.pytorch.DaceModule') -> Tuple[List[str], List[str]]:
    """ Get the list of forward-pass argument names for a module

        :param module: the module
        :return: the list of strings that are the argnames to the module, and the list of names of the outputs
    """
    arglist = [clean_onnx_name(i) for i in module.dace_model.inputs]

    # add any parameters that are required
    named_params = [
        clean_onnx_name(n) for n, _ in module.model.named_parameters()
    ]
    arglist.extend(n for n in named_params
                   if n not in arglist and n in module.sdfg.arrays
                   and not module.sdfg.arrays[n].transient)

    outputs = [clean_onnx_name(o) for o in module.dace_model.outputs]
    return arglist, outputs


_TYPECLASS_TO_TORCH_DTYPE_STR = {
    dt.int8: "kInt8",
    dt.uint8: "kUInt8",
    dt.int16: "kInt16",
    dt.int32: "kInt32",
    dt.int64: "kInt64",
    dt.float16: "kFloat16",
    dt.float32: "kFloat32",
    dt.float64: "kFloat64",
}


def tensor_init_for_desc(name: str, desc: data.Data) -> str:
    """ Emit the initialization code for a descriptor.
    """
    return f"""\
Tensor {name}_ = torch::empty(
    {{{', '.join(str(s) for s in desc.shape)}}},
    torch::TensorOptions()
        .dtype(torch::{_TYPECLASS_TO_TORCH_DTYPE_STR[desc.dtype]})
        .device(torch::{'kCUDA' if is_cuda(desc.storage) else 'kCPU'})
        .layout(torch::kStrided));
    """


def initialize_outputs_code(module: 'daceml.pytorch.DaceModule',
                            output_names: List[str]) -> str:
    """ Generate the code that initializes the output tensors

        :param module: the module
        :return: the code
    """
    arglist = module.sdfg.arglist()
    code = ""
    for name in output_names:
        code += tensor_init_for_desc(name, arglist[name])

    return code


def argument_codegen(module: 'daceml.pytorch.DaceModule',
                     input_names: List[str],
                     output_names: List[str]) -> Tuple[str, str, str]:
    """ Generate the code that grabs the pointers of inputs and outputs.

        :param module: the module
        :return: the code for initializing the argument, the sdfg arguments in order, and the init call arguments
    """
    arglist = module.sdfg.arglist()

    # initialize the inputs and outputs
    ptr_init_code = "\n    // setup input and output pointers\n    "
    # inputs: make these contiguous if they're not
    ptr_init_code += '\n    '.join(
        f"{arglist[name].dtype.ctype} *{name}_ptr = {name}_.contiguous().data_ptr<{arglist[name].dtype.ctype}>();"
        for name in input_names)
    # outputs
    ptr_init_code += '\n    '.join(
        f"{arglist[name].dtype.ctype} *{name}_ptr = {name}_.data_ptr<{arglist[name].dtype.ctype}>();"
        for name in output_names)
    ptr_init_code += "\n    // setup constant arguments\n"

    # initialize all remaining parameters
    remaining = set(arglist).difference(
        itertools.chain(input_names, output_names))
    for name in remaining:

        # remaining args must be constants
        if name not in module.dace_model.clean_weights:
            raise ValueError(
                f"Cannot generate PyTorch module C++ code: SDFG argument {name} is not an input or output"
                f" of the PyTorch Module, and not a constant.")
        if arglist[name].total_size > 1000:
            raise ValueError(
                f"Cannot generate PyTorch module C++ code: SDFG argument {name} is not an input or output"
                f" of the PyTorch Module, and is too large.")

        value = module.dace_model.clean_weights[name]
        ptr_init_code += f"    {constant_initializer_code(name + '_ptr', arglist[name], value)}\n"

    arguments = ", ".join(f"{n}_ptr" for n in arglist)
    init_arguments = ", ".join(f"{n}_ptr" for n, desc in arglist.items()
                               if isinstance(desc, data.Scalar))

    return ptr_init_code, arguments, init_arguments


def item_to_cpp_literal(item) -> str:
    dtype = str(item.dtype)
    if dtype == "float32":
        return f"{item}f"
    elif dtype == "int64":
        return f"{item}l"
    elif dtype in ["float64", "int32", "int16", "int8"]:
        return str(item)
    else:
        raise ValueError(f"Unsupported tensor type {item.dtype}")


def constant_initializer_code(name: str, desc: data.Data, value) -> str:
    if isinstance(desc, data.Array):
        iterator = np.nditer(value.cpu().numpy(), order="C")
        return f"{desc.dtype.ctype} {name}[{desc.total_size}] =" \
               f" {{{', '.join(item_to_cpp_literal(e) for e in iterator)}}};"
    else:
        return f"{desc.dtype.ctype} {name} = {str(value.item())};"


def code_for_backward_function(module: 'daceml.pytorch.DaceModule',
                               forward_sdfg: dace.SDFG,
                               backward_sdfg: dace.SDFG) -> str:
    pass


def code_for_module(module: 'daceml.pytorch.DaceModule') -> str:
    """ Generate the code for an operator that calls the sdfgs in the module.

        :param module: the module
    """

    inputs, outputs = get_arglist(module)
    sdfg_name = module.sdfg.name

    if module.backward:
        raise NotImplemented("todo")
    else:
        ptr_init_code, sdfg_call_arguments, init_arguments = argument_codegen(
            module, inputs, outputs)
        return f"""
#include <torch/torch.h>
#include <torch/script.h>
#include "{sdfg_name}.h"

using torch::Tensor;
using torch::DeviceType;
using torch::autograd::tensor_list;
using torch::autograd::AutogradContext;

TORCH_LIBRARY(daceml_{sdfg_name}, m) {{
    m.def("{sdfg_name}(int handle_ptr, {", ".join('Tensor ' + arg for arg in inputs)}) -> {'Tensor' if len(outputs) == 1
        else "(" + ", ".join(['Tensor'] * len(outputs)) + ")"}");
}}

// function definition
{"Tensor" if len(outputs) == 1 else f"std::tuple<{', '.join(['Tensor'] * len(outputs))}>"}
{sdfg_name}(int64_t handle_ptr, {",".join(f"const Tensor& {name}_" for name in inputs)}) {{

    // initialize outputs
    {initialize_outputs_code(module, outputs)}
    
    {ptr_init_code}

    // get SDFG state handle
    {sdfg_name}Handle_t handle = reinterpret_cast<{sdfg_name}Handle_t>(handle_ptr);

    // call SDFG
    __program_{sdfg_name}(handle, {sdfg_call_arguments});

    // return to torch
    return {f"{outputs[0]}_" if len(outputs) == 1
        else f"{{{', '.join(o + '_' for o in outputs)}}}"};
}}

TORCH_LIBRARY_IMPL(daceml_{sdfg_name}, {'CUDA' if module.use_cuda else 'CPU'}, m) {{
    m.impl("{sdfg_name}", {sdfg_name});
}}
        """


def compile_and_get_function(module: 'daceml.pytorch.DaceModule',
                             dummy_inputs) -> CompiledTorchFunction:
    """ Get a torch callable for the module. This will compile the sdfg, compile a PyTorch C++ operator, register it
        with PyTorch and return the function that calls it.

        :param module: the module.
        :param dummy_inputs: dummy inputs to initialize the model with.
        :return: the callable function for the SDFG.
    """

    # build the SDFG
    sdfg_build_path = os.path.abspath(module.sdfg.build_folder)
    # set all states to not-sync
    for state in module.sdfg.nodes():
        state.nosync = True
    compiled: CompiledSDFG = module.dace_model.compile_and_init()

    args = tuple(dummy_inputs) + tuple(
        p.data for n, p in module.model.named_parameters()
        if n in module.dace_model.inputs)
    # construct the arguments and initialize the SDFG
    inputs, symbols, outputs = module.dace_model._call_args(args=args,
                                                            kwargs={})
    _, initargtuple = compiled._construct_args({
        **inputs,
        **outputs,
        **symbols,
        **module.dace_model.initialized_parameters
    })
    compiled.initialize(*initargtuple)

    for _, hook in module.post_compile_hooks.items():
        hook(compiled)

    handle_ptr = torch.tensor([compiled._libhandle.value]).squeeze(0)

    class SDFGEnvironment:
        """ Environment for the SDFG
        """

        cmake_minimum_version = None
        cmake_packages = []
        cmake_variables = {}
        cmake_includes = [os.path.join(sdfg_build_path, "include")]
        cmake_compile_flags = []
        cmake_link_flags = []
        cmake_files = []
        cmake_libraries = [
            os.path.join(sdfg_build_path, "build",
                         platform_library_name(module.sdfg.name))
        ]
        state_fields = []
        dependencies = []
        headers = []
        init_code = ""
        finalize_code = ""

    SDFGEnvironment.__name__ = module.sdfg.name
    dace.library.environment(SDFGEnvironment)

    # build the PyTorch module
    code = code_for_module(module)
    libname = f"torch_{module.sdfg.name}"
    program = CodeObject(libname,
                         code,
                         "cpp",
                         targets.cpu.CPUCodeGen,
                         f"Torch{module.sdfg_name}",
                         environments={
                             PyTorch.full_class_path(),
                             SDFGEnvironment.full_class_path()
                         })
    torch_module_build_path = os.path.join('.dacecache',
                                           f"torch_{module.sdfg.name}")

    compiler.generate_program_folder(None, [program], torch_module_build_path)
    compiler.configure_and_compile(torch_module_build_path)

    torch.ops.load_library(
        os.path.join(torch_module_build_path, "build",
                     platform_library_name(libname)))

    result = CompiledTorchFunction(function=operator.attrgetter(
        f"daceml_{module.sdfg.name}.{module.sdfg.name}")(torch.ops),
                                   compiled_sdfg=compiled,
                                   ptr=handle_ptr)
    return result
