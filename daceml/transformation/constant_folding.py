import copy
from collections import deque
from typing import Dict

import numpy as np

import dace
import dace.data as dt
from dace import registry
from dace.properties import make_properties
from dace.transformation import transformation
from dace.sdfg import nodes as nd
from dace.sdfg import utils as sdutil

import daceml.onnx as donnx
from daceml.onnx.converters import clean_onnx_name
from daceml.onnx.nodes.onnx_op import ONNXOp
from daceml.onnx import ONNXModel

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

        parent: ONNXModel = sdfg._parent_onnx_model
        state = sdfg.nodes()[self.state_id]
        node = state.nodes()[self.subgraph[ConstantFolding._onnx_node]]

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
            parent.weights[constant_name] = np.array(shape_desc.shape,
                                                     np.int64)

            assert len(state.out_edges(node)) == 1
            output_edge = state.out_edges(node)[0]
            access_shape = state.add_access(clean_constant_name)
            state.add_edge(access_shape, None, output_edge.dst,
                           output_edge.dst_conn,
                           sdfg.make_array_memlet(clean_constant_name))
        else:
            # otherwise compute the result of the op
            sub_sdfg = dace.SDFG("sub_sdfg")
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
                    inputs['array_' + edge.dst_conn] = input_value[()]
                else:
                    inputs['array_' + edge.dst_conn] = input_value.copy()

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
                    outputs['array_' + edge.src_conn] = np.empty(
                        (1, ), desc.dtype.as_numpy_dtype())
                else:
                    outputs['array_' + edge.src_conn] = np.empty(
                        tuple(desc.shape), desc.dtype.as_numpy_dtype())

            sub_sdfg(**outputs, **inputs)

            for edge in state.out_edges(node):
                desc = copy.deepcopy(sdfg.arrays[edge.data.data])
                desc.transient = False
                output_value = outputs['array_' + edge.src_conn]

                constant_name = sdfg.temp_data_name()
                clean_constant_name = clean_onnx_name(constant_name)
                sdfg.add_datadesc(clean_constant_name, desc)

                assert constant_name not in parent.weights
                if isinstance(desc, dt.Scalar):
                    parent.weights[constant_name] = output_value.reshape(())
                else:
                    parent.weights[constant_name] = output_value

                access_constant = state.add_access(clean_constant_name)
                state.add_edge(access_constant, None, edge.dst, edge.dst_conn,
                               sdfg.make_array_memlet(clean_constant_name))

        # remove all now useless nodes with a reverse BFS
        removed_nodes = []
        queue = deque([node])
        while len(queue) > 0:
            current_node = queue.popleft()

            edges = state.in_edges(current_node)
            state.remove_node(current_node)
            removed_nodes.append(current_node)

            for e in edges:
                next_node = e.src
                if len(state.out_edges(next_node)) == 0:
                    queue.append(next_node)

        # Remove the array corresponding to removed access nodes if possible
        for rn in removed_nodes:
            if isinstance(rn, nd.AccessNode):
                for ostate in sdfg.nodes():
                    if ostate is state:
                        continue
                    if any(n.data == rn.data for n in state.data_nodes()):
                        break
                else:
                    del sdfg.arrays[rn.data]
