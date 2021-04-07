"""
A faster way of evaluating ONNX nodes directly using the ORT C API directly from python.

This is mainly used for constant folding
"""
from typing import Dict

import torch


def build_evaluator():
    pass


def evaluate_node(sdfg, state, node) -> Dict[str, torch.Tensor]:
    """ Evaluate the given node and return the outputs produced.
        :param sdfg: the sdfg of the node.
        :param state: the state of the node.
        :param node: the node to evaluate:
        :return: a mapping from node output connector to the result tensor.
    """
