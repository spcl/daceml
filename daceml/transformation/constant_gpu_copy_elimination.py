from typing import Dict

from dace import registry, properties, SDFG, nodes as nd, dtypes
from dace.sdfg import graph
from dace.transformation import transformation
from dace.sdfg import utils as sdutil

from daceml.onnx import ONNXModel
from daceml.onnx.converters import clean_onnx_name


@registry.autoregister_params(singlestate=True)
@properties.make_properties
class ConstantDeviceCopyElimination(transformation.Transformation):
    """ Move Host to Device copies to SDFG initialization by adding a post_compile_hook
    """

    # pattern matching only checks that the type of the node matches,
    _host_node = transformation.PatternNode(nd.AccessNode)
    _device_node = transformation.PatternNode(nd.AccessNode)

    @staticmethod
    def expressions():
        return [
            sdutil.node_path_graph(ConstantDeviceCopyElimination._host_node,
                                   ConstantDeviceCopyElimination._device_node)
        ]

    @staticmethod
    def can_be_applied(graph: graph.OrderedMultiDiConnectorGraph,
                       candidate: Dict[nd.Node, int],
                       expr_index: int,
                       sdfg,
                       strict: bool = False):

        host_node: nd.AccessNode = graph.nodes()[candidate[
            ConstantDeviceCopyElimination._host_node]]
        device_node: nd.AccessNode = graph.nodes()[candidate[
            ConstantDeviceCopyElimination._device_node]]

        # SDFG must be imported from an ONNXModel
        if not hasattr(sdfg, "_parent_onnx_model"):
            return False

        # the only edge out of the host node must be to the device node
        if graph.out_degree(host_node) > 1:
            return False

        # the only edge into the device node must be from the host node
        if graph.in_degree(device_node) > 1:
            return False

        # host node must be non-transient, device node must be transient
        if host_node.desc(
                sdfg).transient or not device_node.desc(sdfg).transient:
            return False

        # only support GPU for now
        if device_node.desc(sdfg).storage is not dtypes.StorageType.GPU_Global:
            return False

        return host_node.data in sdfg._parent_onnx_model.clean_weights

    @staticmethod
    def match_to_str(graph, candidate):
        host_node: nd.AccessNode = graph.nodes()[candidate[
            ConstantDeviceCopyElimination._host_node]]
        return "Move host-to-device copy of {} to SDFG initialization".format(
            host_node.data)

    def apply(self, sdfg: SDFG):
        parent: ONNXModel = sdfg._parent_onnx_model
        state = sdfg.nodes()[self.state_id]
        host_node: nd.AccessNode = state.nodes()[self.subgraph[
            ConstantDeviceCopyElimination._host_node]]
        device_node: nd.AccessNode = state.nodes()[self.subgraph[
            ConstantDeviceCopyElimination._device_node]]

        onnx_host_name = find_unclean_onnx_name(parent, host_node.data)
        # onnx_device_name = utils.find_unclean_onnx_name(
        #     parent, device_node.data)
        device_node.desc(sdfg).transient = False

        state.remove_node(host_node)

        parent.weights[device_node.data] = parent.weights[onnx_host_name]


def find_unclean_onnx_name(model: ONNXModel, name: str) -> str:
    unclean_name = [n for n in model.weights if clean_onnx_name(n) == name]
    if len(unclean_name) != 1:
        raise ValueError(f"Could not find unclean name for name {name}")
    return unclean_name[0]
