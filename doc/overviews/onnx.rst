ONNX
====

Library Nodes
-------------
This package adds `ONNX Operators <https://github.com/onnx/onnx/blob/master/docs/Operators.md>`_ as library nodes to the SDFG IR.

There exists a ONNX library node for each supported ONNX operator. For example, :class:`~daceml.onnx.nodes.onnx_op.ONNXConv` is the ONNX library node for the ``Conv`` operator.

Operator Parameters (Inputs and Outputs)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The parameters for an operator are specified by adding a connector with the name of the parameter. By default, ops already have connectors for the required parameters, but connectors for optional parameters need to be added manually.

Variadic Parameters
~~~~~~~~~~~~~~~~~~~
Variadic parameters are specified by adding ``__`` followed by the index of the variadic parameter. For example, the :class:`~daceml.onnx.nodes.onnx_op.ONNXSum` operator has a variadic input named ``data_0``. If we wanted to add 3 inputs that connect to this variadic parameter, we would add the connectors: ``data_0__0``, ``data_0__1`` and ``data_0__2``. The indices after ``__`` specify the order of the variadic parameters.

.. note::
    For the variadic parameters to parse correctly, the indices should not have leading zeros. Furthermore, the indices should be sequential without gaps. For example, adding connectors ``data_0__0``, ``data_0__2`` to the :class:`~daceml.onnx.nodes.onnx_op.ONNXSum` operator is invalid because the parameter with index 1 is missing.

Attributes
~~~~~~~~~~
Attributes are set by passing them to the constructor (as python types). For example, the following code sets the stride parameter.

.. testcode::

    from daceml.onnx import ONNXConv
    conv = ONNXConv("MyConvNode", strides=[2, 2])

The following attribute types are supported:

* ``INT`` -- passed as ``int``.
* ``INTS`` -- passed as ``List[int]``.
* ``STRING`` -- passed as ``str``.
* ``STRINGS`` -- passed as ``List[str]``.
* ``FLOAT`` -- passed as ``double``.
* ``FLOATS`` -- passed as ``List[double]``.
* ``TENSOR`` -- passed as ``numpy.ndarray``.

Example
~~~~~~~

The following example sets up and runs an SDFG containing an ONNX Conv operator using :class:`~daceml.onnx.nodes.onnx_op.ONNXConv`.

.. testcode::

    import dace
    from daceml.onnx import ONNXConv
    import numpy as np

    sdfg = dace.SDFG("conv_example")
    sdfg.add_array("X_arr", (5, 3, 10, 10), dace.float32)
    sdfg.add_array("W_arr", (16, 3, 3, 3), dace.float32)
    sdfg.add_array("Z_arr", (5, 16, 8, 8), dace.float32)

    state = sdfg.add_state()
    access_X = state.add_access("X_arr")
    access_W = state.add_access("W_arr")
    access_Z = state.add_access("Z_arr")

    conv = ONNXConv("MyConvNode")

    state.add_node(conv)
    state.add_edge(access_X, None, conv, "X", sdfg.make_array_memlet("X_arr"))
    state.add_edge(access_W, None, conv, "W", sdfg.make_array_memlet("W_arr"))
    state.add_edge(conv, "Y", access_Z, None, sdfg.make_array_memlet("Z_arr"))

    X = np.random.rand(5, 3, 10, 10).astype(np.float32)
    W = np.random.rand(16, 3, 3, 3).astype(np.float32)
    Z = np.zeros((5, 16, 8, 8)).astype(np.float32)

    sdfg(X_arr=X, W_arr=W, Z_arr=Z)

.. testoutput::

    Automatically expanded library node "MyConvNode".

Importing ONNX models
---------------------
ONNX models can be imported using the :class:`~daceml.onnx.ONNXModel` frontend.

.. testcode::

    import onnx
    import os
    import numpy as np
    from daceml.onnx import ONNXModel

    # Download an ONNX model. For example:
    # https://github.com/onnx/models/raw/master/vision/classification/efficientnet-lite4/model/efficientnet-lite4-11.onnx
    model_path = os.path.join("..", "tests", "onnx_files", "efficientnet.onnx")
    model = onnx.load(model_path)
    dace_model = ONNXModel("efficientnet", model)

    test_input = np.random.rand(1, 3, 224, 224).astype(np.float32)
    dace_model(test_input)

.. testoutput::
    :hide:
    :options: +ELLIPSIS

    ...
