import copy
import logging
from collections import deque
from typing import Dict

import numpy as np

import dace
import torch
from dace import data as dt, dtypes
from dace import registry
from dace.properties import make_properties
from dace.transformation import transformation
from dace.sdfg import nodes as nd
from dace.sdfg import utils as sdutil

import daceml.onnx as donnx
from daceml.onnx.converters import clean_onnx_name
from daceml.onnx.nodes.onnx_op import ONNXOp
from daceml.onnx import ONNXModel

log = logging.getLogger(__name__)

# blocklist of nondeterministic ops
# yapf: disable
NONDETERMINISTIC_OPS = {'ONNXDropout',
                        'ONNXGradient',
                        'ONNXGraphCall',
                        'ONNXIf',
                        'ONNXLoop',
                        'ONNXMomentum',
                        'ONNXMultinomial',
                        'ONNXRandomNormal',
                        'ONNXRandomNormalLike',
                        'ONNXRandomUniform',
                        'ONNXRandomUniformLike',
                        'ONNXSVMClassifier',
                        'ONNXSVMRegressor',
                        'ONNXScan',
                        'ONNXTreeEnsembleClassifier',
                        'ONNXTreeEnsembleRegressor'}
# yapf: enable

global UNIQUE_ID
UNIQUE_ID = 0


@registry.autoregister_params(singlestate=True)
@make_properties
class ConstantFolding(transformation.Transformation):
    """ Remove nodes where all inputs are known and replace them with constant nodes by precomputing the output.
    """
    # pattern matching only checks that the type of the node matches,
    _onnx_node = ONNXOp("_")

    @staticmethod
    def expressions():
        return [sdutil.node_path_graph(ConstantFolding._onnx_node)]

    @staticmethod
    def is_constant(sdfg: dace.SDFG, state: dace.SDFGState, node) -> bool:
        if len(state.in_edges(node)) > 0:
            return False

        # the ONNX importer adds a _parent_onnx_model attribute to the sdfg
        if isinstance(node, nd.AccessNode
                      ) and node.data in sdfg._parent_onnx_model.clean_weights:
            return True

        return False

    @staticmethod
    def can_be_applied(graph: dace.sdfg.graph.OrderedMultiDiConnectorGraph,
                       candidate: Dict[nd.Node, int],
                       expr_index: int,
                       sdfg,
                       strict: bool = False):

        node: ONNXOp = graph.nodes()[candidate[ConstantFolding._onnx_node]]

        # SDFG must be imported from an ONNXModel
        if not hasattr(sdfg, "_parent_onnx_model"):
            return False

        if not 'ONNX' + node.schema.name not in NONDETERMINISTIC_OPS:
            return False

        if isinstance(node, donnx.ONNXShape):
            return True

        # all inputs are constant
        for edge in graph.in_edges(node):
            if not ConstantFolding.is_constant(sdfg, graph, edge.src):
                return False

        return True

    @staticmethod
    def match_to_str(graph, candidate):
        node: ONNXOp = graph.nodes()[candidate[ConstantFolding._onnx_node]]
        return "Precompute outputs of {}".format(node)

    def apply(self, sdfg: dace.SDFG):
        # Extract the subgraph, execute it and insert an AccessNode to the result
        # this method of execution is slow but simple. A better option would be to call the ORT
        # C API from a python object (like the OpChecker).

        parent: ONNXModel = sdfg._parent_onnx_model
        state = sdfg.nodes()[self.state_id]
        node = state.nodes()[self.subgraph[ConstantFolding._onnx_node]]
        log.debug(f"Applying constant folding: {node} in {state}")

        if isinstance(node, donnx.ONNXShape):
            # if we have a shape node, replace it with a constant
            assert len(state.in_edges(node)) == 1
            shape_in_edge = state.in_edges(node)[0]
            assert shape_in_edge.dst_conn == "data"
            shape_desc = sdfg.arrays[shape_in_edge.src.data]

            constant_name = sdfg.temp_data_name()
            clean_constant_name = clean_onnx_name(constant_name)
            sdfg.add_array(clean_constant_name, (len(shape_desc.shape), ),
                           dace.int64)

            assert constant_name not in parent.clean_weights
            parent.weights[constant_name] = torch.from_numpy(
                np.array(shape_desc.shape, np.int64))

            assert len(state.out_edges(node)) == 1
            output_edge = state.out_edges(node)[0]
            access_shape = state.add_access(clean_constant_name)
            state.add_edge(access_shape, None, output_edge.dst,
                           output_edge.dst_conn,
                           sdfg.make_array_memlet(clean_constant_name))
        else:
            # otherwise compute the result of the op
            global UNIQUE_ID
            UNIQUE_ID += 1
            sub_sdfg = dace.SDFG("sub_sdfg_" + str(UNIQUE_ID))
            sub_state = sub_sdfg.add_state()

            node_copy = copy.deepcopy(node)
            sub_state.add_node(node_copy)

            inputs = {}
            for edge in state.in_edges(node):
                # we know from can_be_applied that all in edges are from AccessNodes
                assert (isinstance(edge.src, nd.AccessNode)
                        and hasattr(sdfg, "_parent_onnx_model") and
                        edge.src.data in sdfg._parent_onnx_model.clean_weights)

                desc = copy.deepcopy(sdfg.arrays[edge.data.data])
                desc.transient = False
                sub_sdfg.add_datadesc('array_' + edge.dst_conn, desc)

                input_value = sdfg._parent_onnx_model.clean_weights[
                    edge.src.data]

                if len(input_value.shape) == 0:
                    inputs['array_' +
                           edge.dst_conn] = input_value.cpu().numpy()[()]
                else:
                    inputs['array_' + edge.dst_conn] = input_value.clone()

                access = sub_state.add_access('array_' + edge.dst_conn)
                sub_state.add_edge(
                    access, None, node_copy, edge.dst_conn,
                    sub_sdfg.make_array_memlet('array_' + edge.dst_conn))

            outputs = {}
            for edge in state.out_edges(node):
                desc = copy.deepcopy(sdfg.arrays[edge.data.data])
                if isinstance(desc, dt.Scalar):
                    # we need to copy to an array of size [1] so that we can "return" the output from the sdfg
                    desc.transient = True
                    sub_sdfg.add_datadesc('scalar_array_' + edge.src_conn,
                                          desc)
                    sub_sdfg.add_array('array_' + edge.src_conn, [1],
                                       desc.dtype,
                                       transient=False)

                    access_scalar = sub_state.add_access('scalar_array_' +
                                                         edge.src_conn)
                    access = sub_state.add_access('array_' + edge.src_conn)
                    sub_state.add_edge(
                        node_copy, edge.src_conn, access_scalar, None,
                        sub_sdfg.make_array_memlet('scalar_array_' +
                                                   edge.src_conn))

                    sub_state.add_edge(
                        access_scalar, None, access, None,
                        sub_sdfg.make_array_memlet('array_' + edge.src_conn))
                else:
                    desc.transient = False
                    sub_sdfg.add_datadesc('array_' + edge.src_conn, desc)
                    access = sub_state.add_access('array_' + edge.src_conn)
                    sub_state.add_edge(
                        node_copy, edge.src_conn, access, None,
                        sub_sdfg.make_array_memlet('array_' + edge.src_conn))

                if len(desc.shape) == 0:
                    empty_array = np.empty((1, ), desc.dtype.as_numpy_dtype())
                else:
                    empty_array = np.empty(tuple(desc.shape),
                                           desc.dtype.as_numpy_dtype())

                empty_array = torch.from_numpy(empty_array)

                if desc.storage is dtypes.StorageType.GPU_Global:
                    empty_array = empty_array.cuda()

                outputs['array_' + edge.src_conn] = empty_array

            sub_sdfg(**outputs, **inputs)

            for edge in state.out_edges(node):
                desc = copy.deepcopy(sdfg.arrays[edge.data.data])
                desc.transient = False
                output_value = outputs['array_' + edge.src_conn]

                constant_name = sdfg.temp_data_name()
                clean_constant_name = clean_onnx_name(constant_name)
                sdfg.add_datadesc(clean_constant_name, desc)

                assert constant_name not in parent.weights
                assert type(output_value) is torch.Tensor

                if not dtypes.can_access(dtypes.ScheduleType.CPU_Multicore,
                                         desc.storage):
                    cpu_desc = copy.deepcopy(desc)
                    cpu_desc.storage = dtypes.StorageType.CPU_Heap
                    cpu_desc.transient = False
                    desc.transient = True
                    copy_in_name = sdfg.temp_data_name()
                    clean_copy_in_name = clean_onnx_name(copy_in_name)
                    sdfg.add_datadesc(clean_copy_in_name, cpu_desc)

                    access_constant = state.add_access(clean_constant_name)
                    state.add_edge(state.add_read(clean_copy_in_name), None,
                                   access_constant, None,
                                   sdfg.make_array_memlet(clean_copy_in_name))

                    name_to_add = copy_in_name
                else:
                    access_constant = state.add_read(clean_constant_name)
                    name_to_add = constant_name

                if isinstance(desc, dt.Scalar):
                    parent.weights[name_to_add] = output_value.reshape(())
                else:
                    parent.weights[name_to_add] = output_value

                state.add_edge(access_constant, None, edge.dst, edge.dst_conn,
                               sdfg.make_array_memlet(clean_constant_name))

        # remove all now useless nodes with a reverse BFS
        remove_node_and_computation(sdfg, state, node)


def remove_node_and_computation(sdfg: dace.SDFG, state: dace.SDFGState,
                                node: nd.Node):
    """ Remove a node and the parent nodes that compute this node, if the outputs are not used elsewhere.

        :param sdfg: the sdfg containing the node.
        :param state: the state containing the node.
        :param node: the node to remove
    """
    queue = deque([node])
    while len(queue) > 0:
        current_node = queue.popleft()

        edges = state.in_edges(current_node)
        state.remove_node(current_node)
        for e in edges:
            next_node = e.src
            data_used_in_other_states = isinstance(next_node, nd.AccessNode) and \
                                        any(n.data == next_node.data
                                            for s in sdfg.nodes()
                                            for n in s.nodes() if s is not state)

            if len(state.out_edges(
                    next_node)) == 0 and not data_used_in_other_states:
                queue.append(next_node)
