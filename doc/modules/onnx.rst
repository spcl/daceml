.. _mod_onnx:

daceml.onnx
===========

.. py:module:: daceml.onnx

.. autofunction:: has_onnx_node
.. autofunction:: get_onnx_node

.. autoclass:: daceml.onnx.ONNXModel
    :members:
    :special-members: __call__
    :no-undoc-members:

.. autoclass:: daceml.onnx.nodes.onnx_op.ONNXOp
    :members:
    :no-undoc-members:
    :show-inheritance:

Schema Representation
---------------------

.. autofunction:: daceml.onnx.onnx_representation

.. autoclass:: daceml.onnx.ONNXParameterType
    :members:
    :no-undoc-members:

.. autoclass:: daceml.onnx.ONNXAttributeType
    :members:
    :no-undoc-members:

.. autoclass:: daceml.onnx.ONNXAttribute
    :members:
    :no-undoc-members:

.. autoclass:: daceml.onnx.ONNXTypeConstraint
    :members:
    :no-undoc-members:

.. autoclass:: daceml.onnx.ONNXParameter
    :members:
    :no-undoc-members:

.. autoclass:: daceml.onnx.ONNXSchema
    :members:
    :no-undoc-members:

Op Implementation Registration
------------------------------

.. autoclass:: daceml.onnx.implementation_abc.ONNXForward
    :members:

.. _pure-ops:

Pure ONNX Implementations
-------------------------

.. automodule:: daceml.onnx.op_implementations.pure_implementations
    :members:
    :show-inheritance:
    :exclude-members: program_for_node, forward_can_be_applied, forward

Dace CMake Environments
-----------------------

.. automodule:: daceml.onnx.environments.onnxruntime
    :members:

Supported ONNX Operators
------------------------
The following documentation is mostly automatically generated from the ONNX documentation, except for the removal of unsupported attributes and nodes.

.. automodule:: daceml.onnx.nodes.onnx_op
    :members:
    :exclude-members: Expansion, has_onnx_node, get_onnx_node, ONNXOp
    :show-inheritance:
    :no-undoc-members:
