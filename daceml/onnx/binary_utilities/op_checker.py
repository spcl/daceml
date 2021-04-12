import ctypes
from typing import Optional, List, Tuple

import dace
import numpy as np
from dace.dtypes import DTYPE_TO_TYPECLASS

from daceml.onnx.converters import ONNX_DTYPES_TO_DACE_TYPE_CLASS
from daceml.onnx.schema import ONNXAttributeType
from daceml.ort_api import KernelSession, ExecutableKernelContext, ExecutableKernel
from daceml.ort_api import ORTAPIError, ORTCAPIInterface


def check_op(sdfg, state, node, cuda=False) -> Tuple[List[bool], List[bool]]:
    """ Check whether a ONNXOp node has an implementation in ORT """

    with ORTCAPIInterface() as api:
        with KernelSession(api) as session:
            with ExecutableKernelContext(api, session, node.name,
                                         node.schema.name) as context:
                for attribute, onnx_attribute in node.schema.attributes.items(
                ):
                    if hasattr(node, attribute):
                        context.add_attribute(attribute,
                                              getattr(node, attribute),
                                              onnx_attribute.type)

                for edge, is_input in node.iter_edges(state):
                    edge_data = edge.data.data
                    edge_dtype = sdfg.arrays[edge_data].dtype
                    if is_input:
                        context.add_input(edge_dtype)
                    else:
                        context.add_output(edge_dtype)
                with context.try_create_kernel(1 if cuda else 0) as kernel:
                    return kernel.check_io_locations()
