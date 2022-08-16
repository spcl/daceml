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
* ``INTS`` -- passed as ``List`` [``int``].
* ``STRING`` -- passed as ``str``.
* ``STRINGS`` -- passed as ``List`` [``str``].
* ``FLOAT`` -- passed as ``float``.
* ``FLOATS`` -- passed as ``List`` [``float``].
* ``TENSOR`` -- passed as ``numpy.ndarray``.

Example
~~~~~~~

The following examples setup and run an SDFG containing an ONNX Conv operator using :class:`~daceml.onnx.nodes.onnx_op.ONNXConv`.

This can be done using the python frontend:

.. testcode::

    import dace
    import daceml.onnx as donnx
    import numpy as np

    @dace.program
    def conv_program(X_arr: dace.float32[5, 3, 10, 10],
                     W_arr: dace.float32[16, 3, 3, 3]):
        output = dace.define_local([5, 16, 4, 4], dace.float32)
        donnx.ONNXConv(X=X_arr, W=W_arr, Y=output, strides=[2, 2])
        return output

    X = np.random.rand(5, 3, 10, 10).astype(np.float32)
    W = np.random.rand(16, 3, 3, 3).astype(np.float32)

    result = conv_program(X_arr=X, W_arr=W)

or the SDFG API:

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

.. _node_implementations:

Node Implementations
--------------------
The ONNX library nodes work like library nodes in dace: they can have multiple implementations that can be selected
prior to compilation. By default, the nodes use the ``onnxruntime`` implementation which calls the kernels from
ONNXRuntime.

The implementation of a node can be chosen either by specifying the default implementation for the whole ONNX library:

.. code-block:: python

    import daceml.onnx as donnx
    donnx.default_implementation = "pure"

Or for a specific node:

.. code-block:: python

    import daceml.onnx as donnx
    donnx.ONNXMatMul.default_implementation = "pure"

Note that if an implementation doesn't exist, or cannot be applied, the node expansion will fall back to
``onnxruntime``.

Implementation Registration
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Implementations for an ONNX node can be registered by implementing the abstract
:class:`~daceml.onnx.forward_implementation_abc.ONNXForward` class. The implementation can be registered using the
:func:`~daceml.onnx.op_implementations.utils.op_implementation` decorator. For registration, the parameters ``op`` and
``name`` must be passed, where ``op`` is the name of the ONNX op (without the ``ONNX`` prefix), and ``name`` is the name
of the implementation.

For example:

.. code-block:: python

    import daceml.onnx as donnx
    from daceml.onnx.op_implementations import op_implementation
    from daceml.onnx.forward_implementation_abc import ONNXForward

    @op_implementation(op="MatMul", name="my_implementation_name")
    class MyMatMul(ONNXForward):
        ...

    # can then be used with the library nodes
    donnx.ONNXMatMul.default_implementation = "my_implementation_name"


Implementations should assume that the node has been validated before it was passed to the expansion. This means that:

* input edges are only connected to valid connectors for the ONNX op
* all ONNX required parameters have been connected
* variadic parameters are correctly passed (correctly formed, and without gaps in the indexing)
* the types of parameters solve correctly according to the type constraints given by the onnx specification
* all required attributes have been set

Implementations can choose to reject certain classes of the node the expand, by implementing the (optional)
:meth:`~daceml.onnx.forward_implementation_abc.ONNXForward.forward_can_be_applied` method.

.. _Pure Implementations:

Pure Implementations
~~~~~~~~~~~~~~~~~~~~
Several nodes have an pure SDFG implementation (i.e. not ONNXRuntime based).
The list of all implementations can be found :ref:`here <pure-ops>`. These
implementations can be analyzed by DaCeML, which allows us to apply optimizing
transformations as well as :ref:`autodiff <Autodiff>`.

The :func:`~daceml.onnx.op_implementations.utils.python_pure_op_implementation`
decorator makes implementing these operators easier, for example:

.. code-block:: python

  @python_pure_op_implementation
  def Exp(input, output):
      output[:] = np.exp(input)


Running ONNX Node Tests
~~~~~~~~~~~~~~~~~~
After implementing a pure operator, you can run the built-in ONNX node tests using:
``pytest test/pure_expansions/tests/pure_expansions/test_onnx_cases.py``
(use the ``-k <test_name>`` flag to run a specific test).

The source code for the test cases for a certain operator can be found in the
`ONNX repository <https://github.com/onnx/onnx/tree/v1.7.0/onnx/backend/test/case/node>`_.

Importing ONNX models
---------------------
ONNX models can be imported using the :class:`~daceml.onnx.ONNXModel` frontend.

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

    # Download an ONNX model. For example:
    # https://github.com/onnx/models/raw/master/vision/classification/efficientnet-lite4/model/efficientnet-lite4-11.onnx
    model_path = os.path.join("..", "tests", "onnx_files", "efficientnet.onnx")
    model = onnx.load(model_path)
    dace_model = ONNXModel("efficientnet", model)

    test_input = np.random.rand(1, 3, 224, 224).astype(np.float32)
    dace_model(test_input)


Schema Representation & Protobuf conversion
-------------------------------------------
ONNX protobufs are imported and converted to python property classes that can be serialized to and from json by
dace (for example :class:`~daceml.onnx.ONNXSchema`). ONNX protobuf instances can be converted to these classes using the
``from_onnx_proto`` class method that is present on these objects.

These objects are created using :func:`~daceml.onnx.onnx_representation`. Other ONNX protobuf types can likely be
supported in this manner as well. For examples, see the source file ``daceml/onnx/schema.py``.
