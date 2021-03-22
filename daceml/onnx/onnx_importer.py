import typing
from collections import OrderedDict
from copy import deepcopy
from itertools import chain, repeat

import numpy as np
import torch

import onnx
from onnx import numpy_helper

import dace
import dace.data as dt
from dace.frontend.python.parser import infer_symbols_from_shapes
from dace.sdfg import SDFG, SDFGState
from dace.dtypes import AccessType, StorageType, AllocationLifetime
import dace.sdfg.nodes as nd
from dace.symbolic import pystr_to_symbolic

from daceml.onnx.shape_inference import shape_inference
from daceml.onnx.converters import convert_attribute_proto, onnx_tensor_type_to_typeclass, clean_onnx_name
from daceml.onnx.schema import ONNXParameterType
from daceml.onnx.nodes.onnx_op import get_onnx_node, has_onnx_node

numpy_to_torch_dtype_dict = {
    np.bool: torch.bool,
    np.uint8: torch.uint8,
    np.int8: torch.int8,
    np.int16: torch.int16,
    np.int32: torch.int32,
    np.int64: torch.int64,
    np.float16: torch.float16,
    np.float32: torch.float32,
    np.float64: torch.float64,
    np.complex64: torch.complex64,
    np.complex128: torch.complex128
}

torch_to_numpy_dtype_dict = {
    v: k
    for k, v in numpy_to_torch_dtype_dict.items()
}


def _nested_HasField(obj, full_attr):
    """Performs a nested hasattr check, separating attr on dots."""
    attrs = full_attr.split(".")
    for attr in attrs:
        if obj.HasField(attr):
            obj = getattr(obj, attr)
        else:
            return False
    return True


class ONNXModel:
    """ Loads an ONNX model into an SDFG.

        :Example:
            First download an ONNX model, such as
            `efficientnet <http://spclstorage.inf.ethz.ch/~rauscho/efficientnet-lite4-11.onnx>`_.

            .. testsetup::

                import subprocess
                model_path = os.path.join("..", "tests", "onnx_files", "efficientnet.onnx")
                # Download model
                if not os.path.exists(model_path):
                    subprocess.check_call([
                        "wget",
                        "http://spclstorage.inf.ethz.ch/~rauscho/efficientnet-lite4-11.onnx",
                        "--output-document={}".format(model_path)
                    ])


            .. testcode::

                import onnx
                import os
                import numpy as np
                from daceml.onnx import ONNXModel

                model_path = os.path.join("..", "tests", "onnx_files", "efficientnet.onnx")
                model = onnx.load(model_path)
                dace_model = ONNXModel("efficientnet", model)

                test_input = np.random.rand(1, 3, 224, 224).astype(np.float32)
                dace_model(test_input)

            .. testoutput::
                :hide:
                :options: +ELLIPSIS

                ...
    """
    def __init__(self,
                 name: str,
                 model: onnx.ModelProto,
                 infer_shapes: bool = True,
                 cuda: bool = False,
                 apply_strict: bool = False):
        """
        :param name: the name for the SDFG.
        :param model: the model to import.
        :param infer_shapes: whether to infer shapes for the model. If this is ``False``, the model must have
                             value infos (with shapes) for all arrays, including intermediate values.
        :param cuda: if ``True``, the model will be executed on the GPU.
        :param apply_strict: if ``True``, apply strict transformations after all nodes have
                             been expanded calling (warning: this can be very slow!)
        """

        if infer_shapes:
            model = shape_inference.infer_shapes(model)

        graph: onnx.GraphProto = model.graph

        self.sdfg: SDFG = SDFG(name)  #: the generated SDFG.
        self.sdfg._parent_onnx_model = self
        self.cuda = cuda
        self.apply_strict = apply_strict
        self.state: SDFGState = self.sdfg.add_state(
        )  #: the state containing the model computation.

        # Add all values to the SDFG, check for unsupported ops
        ##########################################

        self.value_infos = {}

        self.inputs: typing.List[str] = []  #: the inputs to the model
        self.outputs: typing.List[str] = []  #: the outputs of the model

        for value, is_input in chain(zip(graph.input, repeat(True)),
                                     zip(graph.output, repeat(False))):
            if not value.HasField("name"):
                raise ValueError("Got input or output without name")
            if is_input:
                self.inputs.append(value.name)
            else:
                self.outputs.append(value.name)

            self.value_infos[value.name] = value
            self._add_value_info(value)

        for value in graph.value_info:
            if not value.HasField("name"):
                raise ValueError("Got input or output without name")
            if value.name not in self.value_infos:
                self.value_infos[value.name] = value

        # add weights
        self.weights: typing.Dict[str, torch.Tensor] = {
        }  #: mapping from weight name to array
        for init in graph.initializer:
            self._add_constant_tensor(init)

        access_nodes = {}
        self._idx_to_node = []
        for i, node in enumerate(graph.node):
            if not has_onnx_node(node.op_type):
                raise ValueError("Unsupported ONNX operator: '{}'".format(
                    node.op_type))

            # extract the op attributes

            op_attributes = {
                attribute_proto.name: convert_attribute_proto(attribute_proto)
                for attribute_proto in node.attribute
            }

            if node.HasField("name"):
                node_name = clean_onnx_name(node.name)
            else:
                node_name = node.op_type + "_" + str(i)

            # construct the dace node
            op_node = get_onnx_node(node.op_type)(node_name, **op_attributes)
            self.state.add_node(op_node)
            self._idx_to_node.append(op_node)

            for param_idx, (name, is_input) in chain(
                    enumerate(zip(node.input, repeat(True))),
                    enumerate(zip(node.output, repeat(False)))):
                if clean_onnx_name(name) not in self.sdfg.arrays:
                    if name not in self.value_infos:
                        raise ValueError(
                            "Could not find array with name '{}'".format(name))
                    self._add_value_info(self.value_infos[name])

                # get the access node
                if name in access_nodes:
                    access = access_nodes[name]
                    self._update_access_type(access, is_input)
                else:
                    access = nd.AccessNode(
                        clean_onnx_name(name), AccessType.ReadOnly
                        if is_input else AccessType.WriteOnly)
                    self.state.add_node(access)
                    access_nodes[name] = access

                # get the connector name
                params = op_node.schema.inputs if is_input else op_node.schema.outputs
                params_len = len(params)
                if param_idx >= params_len:
                    # this is a variadic parameter. Then the last parameter of the parameter must be variadic.
                    if params[-1].param_type != ONNXParameterType.Variadic:
                        raise ValueError(
                            "Expected the last {i_or_o} parameter to be variadic,"
                            " since the {i_or_o} with idx {param_idx} has more parameters than the schema ({params_len})"
                            .format(i_or_o="input" if is_input else "output",
                                    param_idx=param_idx,
                                    params_len=params_len))
                    conn_name = params[-1].name + "__" + str(param_idx -
                                                             params_len + 1)
                elif params[
                        param_idx].param_type == ONNXParameterType.Variadic:
                    # this is a variadic parameter, and it is within the range of params, so it must be the first
                    # instance of a variadic parameter
                    conn_name = params[param_idx].name + "__0"
                else:
                    conn_name = params[param_idx].name

                data_desc = self.sdfg.arrays[clean_onnx_name(name)]

                # add the connector if required, and add an edge
                if is_input:
                    if conn_name not in op_node.in_connectors:
                        assert op_node.add_in_connector(conn_name)
                    self.state.add_edge(
                        access, None, op_node, conn_name,
                        dace.Memlet.from_array(clean_onnx_name(name),
                                               data_desc))
                else:
                    if conn_name not in op_node.out_connectors:
                        assert op_node.add_out_connector(conn_name)

                    self.state.add_edge(
                        op_node, conn_name, access, None,
                        dace.Memlet.from_array(clean_onnx_name(name),
                                               data_desc))

        if self.cuda:
            # set all weights to be GPU_Global
            # this was messing with the ORT arena allocator, probably because PT has its own
            # for name, tensor in self.weights.items():
            #     self.weights[name] = self.weights[name].cuda()
            #     self.sdfg.arrays[clean_onnx_name(name)].storage = StorageType.GPU_Global

            self.sdfg.apply_gpu_transformations()

            # set all gpu transients to be persistent
            for _, _, arr in self.sdfg.arrays_recursive():
                if arr.transient and arr.storage == StorageType.GPU_Global:
                    arr.lifetime = AllocationLifetime.Persistent

    @staticmethod
    def _update_access_type(node: dace.nodes.AccessNode, is_input: bool):
        if node.access == AccessType.ReadOnly and not is_input:
            node.access = AccessType.ReadWrite
        elif node.access == AccessType.WriteOnly and is_input:
            node.access = AccessType.ReadWrite

    def _add_constant_tensor(self, tensor: onnx.TensorProto):
        if not tensor.HasField("name"):
            raise ValueError("Got tensor without name")

        if not tensor.HasField("data_type"):
            raise ValueError("Initializer tensor '{}' has no type".format(
                tensor.name))

        name = clean_onnx_name(tensor.name)

        dtype = onnx_tensor_type_to_typeclass(tensor.data_type)

        if len(tensor.dims) == 0:
            # this is a scalar
            self.sdfg.add_scalar(name, dtype)
        else:
            dims = [d for d in tensor.dims]
            if name not in self.sdfg.arrays:
                self.sdfg.add_array(name, dims, dtype)
            else:
                existing_arr = self.sdfg.arrays[name]
                if existing_arr.dtype != dtype:
                    raise ValueError(
                        "Invalid ONNX model; found two values with name '{}', but different dtypes ({} and {})"
                        .format(name, existing_arr.dtype, dtype))
                if tuple(existing_arr.shape) != tuple(dims):
                    raise ValueError(
                        "Invalid ONNX model; found two values with name '{}', but different dimensions ({} and {})"
                        .format(name, existing_arr.shape, dims))

        weight_arr = numpy_helper.to_array(tensor)
        # we need to copy here because the weight_arr tensor is not writable
        self.weights[tensor.name] = torch.from_numpy(weight_arr.copy())

    def _add_value_info(self, value_info: onnx.ValueInfoProto):
        if not value_info.HasField("name"):
            raise ValueError("Got value without name")

        name = value_info.name

        if not _nested_HasField(value_info, "type.tensor_type.shape"):
            raise ValueError(
                "Value '{}' does not have a shape in this graph."
                " Please run shape inference before importing.".format(name))

        tensor_type = value_info.type.tensor_type

        if not tensor_type.HasField("elem_type"):
            raise ValueError(
                "Value '{}' does not have a type in this graph."
                " Please run type inference before importing.".format(name))

        shape = []
        for d in tensor_type.shape.dim:
            if d.HasField("dim_value"):
                shape.append(d.dim_value)
            elif d.HasField("dim_param"):
                parsed = pystr_to_symbolic(d.dim_param)

                for sym in parsed.free_symbols:
                    if clean_onnx_name(str(sym)) not in self.sdfg.symbols:
                        self.sdfg.add_symbol(clean_onnx_name(str(sym)),
                                             stype=int)
                    parsed = parsed.subs(
                        sym, dace.symbol(clean_onnx_name(str(sym))))

                shape.append(parsed)
            else:
                raise ValueError(
                    "Value '{}' does not have a shape in this graph."
                    " Please run shape inference before importing.".format(
                        name))
        transient = name not in self.inputs and name not in self.outputs
        if len(shape) == 0:
            self.sdfg.add_scalar(clean_onnx_name(name),
                                 dtype=onnx_tensor_type_to_typeclass(
                                     tensor_type.elem_type),
                                 transient=transient)
        else:
            self.sdfg.add_array(clean_onnx_name(name),
                                shape=shape,
                                dtype=onnx_tensor_type_to_typeclass(
                                    tensor_type.elem_type),
                                transient=transient)

    @property
    def clean_weights(self):
        return {clean_onnx_name(k): v for k, v in self.weights.items()}

    def __call__(
            self, *args,
            **kwargs) -> typing.Union[np.ndarray, typing.Tuple[np.ndarray]]:
        """ Execute the model.

            :param args: positional arguments to the model. The i-th argument will be passed as the i-th input of the
                         model.
            :param kwargs: named arguments to the model. The passed names should match the names in the ONNX model.
            :return: the output of the model (or a tuple of outputs if there are multiple).
        """

        inputs, params, symbols, outputs = self._call_args(args=args,
                                                           kwargs=kwargs)

        sdfg = deepcopy(self.sdfg)
        sdfg.expand_library_nodes()

        if self.apply_strict:
            sdfg.apply_strict_transformations()

        sdfg(**inputs, **outputs, **params, **symbols)

        if len(outputs) == 1:
            return next(iter(outputs.values()))

        return tuple(outputs.values())

    def _call_args(
        self,
        *,
        args,
        kwargs,
        torch_outputs: bool = None
    ) -> typing.Tuple[typing.Dict[str, typing.Any], typing.Dict[
            str, typing.Any], typing.Dict[str, typing.Any], typing.OrderedDict[
                str, typing.Any]]:
        """ Prepare the arguments for a call.

            This returns 4 dicts; one for each of the following:
            1. the inputs
            2. the weights
            3. inferred values for symbols for dynamic dimensions
            4. outputs

            These arguments can be passed to `self.sdfg`.

            :param args: model positional args
            :param kwargs: model kwargs
            :param torch_outputs: if not None, the outputs will be torch tensors depending on the boolean value.
                                  Otherwise the outputs will be torch tensors only if at least one of the inputs is a
                                  torch tensor.
            :return: the tuple of dicts
        """
        inputs = kwargs

        # convert the positional args to kwargs
        if len(args) > len(self.inputs):
            raise ValueError("Expected {} arguments, got {}".format(
                len(self.inputs), len(args)))

        inputs.update(dict(zip(self.inputs, args)))

        # check that there are no missing inputs
        if len(set(self.inputs).difference(inputs)) != 0:
            raise ValueError("Missing inputs {}".format(", ".join(
                set(self.inputs).difference(inputs))))

        # check that there are no unknown inputs
        # NOTE symbols can only be passed as kwargs
        if len(
                set(inputs).difference(self.inputs).difference(
                    self.sdfg.free_symbols)) != 0:
            raise ValueError("Unknown inputs {}".format(", ".join(
                set(inputs).difference(self.inputs))))

        clean_inputs = {}
        for input, arr in inputs.items():
            if input in self.sdfg.free_symbols:
                clean_inputs[input] = arr
            else:
                clean_inputs[clean_onnx_name(input)] = arr

        # add the weights
        params = {}
        for name, arr in self.weights.items():
            if clean_onnx_name(name) in self.sdfg.arrays:
                desc = self.sdfg.arrays[clean_onnx_name(name)]
                if type(desc) is dt.Scalar:
                    params[clean_onnx_name(name)] = arr.cpu().numpy()[()]
                else:
                    params[clean_onnx_name(name)] = arr.clone()

        inferred_symbols = infer_symbols_from_shapes(self.sdfg, {
            **clean_inputs,
            **params
        })
        inferred_symbols = {k: int(v) for k, v in inferred_symbols.items()}

        if torch_outputs is None:
            torch_outputs = any(
                isinstance(inp, torch.Tensor)
                for _, inp in clean_inputs.items())

        outputs = OrderedDict()
        # create numpy arrays for the outputs
        for output in self.outputs:
            clean_name = clean_onnx_name(output)
            outputs[clean_name] = create_output_array(
                inferred_symbols,
                self.sdfg.arrays[clean_name],
                use_torch=torch_outputs)

        # check that there's no overlap
        seen = set()
        for parameters in [clean_inputs, params, outputs, inferred_symbols]:
            new_parameters = set(parameters)
            assert not seen.intersection(new_parameters)
            seen |= new_parameters

        return clean_inputs, params, inferred_symbols, outputs


def create_output_array(
        inferred_symbols: typing.Dict[str, int],
        desc: dt.Data,
        use_torch=False,
        zeros: bool = False) -> typing.Union[np.ndarray, torch.tensor]:
    """ Create the array for an output. This is either a numpy array or a torch tensor depending on `use_torch`

        When `self.force_torch_outputs` is True, the outputs will be tensors. Otherwise, the outputs will be tensors
        :param inferred_symbols: the symbols inferred from `infer_symbols_from_shapes`.
        :param desc: the data descriptor for the array
        :param use_torch: whether to return a numpy array or a torch tensor.
        :param zeros: if true init with zeros else empty.
    """
    def eval_dim(dim):
        for sym in dim.free_symbols:
            dim = dim.subs(sym, inferred_symbols[sym.name])
        return dim

    shape = [eval_dim(d) if type(d) is dace.symbol else d for d in desc.shape]
    if desc.dtype.veclen > 1:
        shape.append(desc.dtype.veclen)

    if use_torch:
        # as_numpy_dtype doesn't seem to work for indexing into the dict
        return (torch.zeros if zeros else torch.empty)(
            shape,

            dtype=numpy_to_torch_dtype_dict[getattr(np,
                                                    desc.dtype.to_string())])
    else:
        return (np.zeros if zeros else np.empty)(shape,
                                                 dtype=getattr(
                                                     np,
                                                     desc.dtype.to_string()))