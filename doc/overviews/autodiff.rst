Automatic Differentiation
=========================

.. warning::

    The symbolic automatic differentiation feature still experimental.

DaCeML takes a different approach to automatic differentiation than most deep learning frameworks. Instead of
hand-writing backward passes for all differentiable operators, DaceML has a symbolic reverse-mode differentation engine.

Using Autodiff
--------------
There are two main ways to generate backward passes in DaCeML.

:class:`~daceml.pytorch.DaceModule`
    This class includes a ``backward`` parameter. If ``True``, the autodiff engine will be used to add a backward pass
    to the PyTorch module, and the resulting module can be seamlessly used with other PyTorch code. For example:

    .. testcode::

        import torch.nn.functional as F
        from daceml.torch import dace_module

        @dace_module(backward=True)
        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(784, 120)
                self.fc2 = nn.Linear(120, 32)
                self.fc3 = nn.Linear(32, 10)
                self.ls = nn.LogSoftmax(dim=-1)

            def forward(self, x):
                x = F.relu(self.fc1(x))
                x = F.relu(self.fc2(x))
                x = self.fc3(x)
                x = self.ls(x)
                return x


        x = torch.randn(8, 784)
        y = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7], dtype=torch.long)

        model = Net()

        criterion = nn.NLLLoss()
        prediction = model(x)
        loss = criterion(prediction, y)
        print(f"gradients before: {model.model.fc3.weight.grad}")

        # gradients can flow through model!
        loss.backward()

        print(f"gradients after: {model.model.fc3.weight.grad}")

    .. testoutput::
        :hide:
        :options: +ELLIPSIS

        gradients before: None
        gradients after: ...


:func:`~daceml.autodiff.add_backward_pass`

    The autodiff engine can also be run on plain SDFGs. Here, the output ``S`` of the dace function/sdfg
    is differentiated w.r.t to ``X`` and ``Y``.

    .. testcode::

        from daceml.autodiff import add_backward_pass

        @dace.program
        def dace_gemm(
            X: dace.float32[5, 4],
            Y: dace.float32[4, 3],
            Z: dace.float32[5, 3],
            S: dace.float32[1],
        ):

            Z[:] = X @ Y

            @dace.map(_[0:5, 0:3])
            def summap(i, j):
                s >> S(1, lambda a, b: a + b)[0]
                z << Z[i, j]
                s = z

        sdfg = dace_gemm.to_sdfg()
        add_backward_pass(sdfg=sdfg, state=sdfg.nodes()[0], inputs=["X", "Y"], outputs=["S"])


Architecture
------------
At its core, the automatic differentiation engine attempts to `lift` the SymPy scalar differentiation engine to tensor
programs. The SDFG IR is especially suitable for this for two reasons:

* In most SDFGs, computation (i.e. Tasklets) operates on scalars, which can often be differentiated symbolically by
  SymPy.
* The SDFG IR precisely specifies which Tasklets read and write to which memory locations. This information makes it
  simple to correctly sum the gradient contribution from each tasklet.

At a high level, it operates as follows:

1. Find the ``AccessNode`` for each input and output of the ``SDFGState``. Use these to determine the subgraph to
   differentiate.
2. Traverse the subgraph in reverse topological order. For each node:

    * Call a function that `reverses` the node. To reverse the node, the engine checks the
      :class:`~daceml.autodiff.BackwardImplementation` repository for a registered & applicable backward implementation
      for that node. If no such function exists and the node is a ``LibraryNode``, attempt to differentiate the `pure`
      expanded version of the node. Otherwise, call the relevant function
      on :class:`~daceml.autodiff.backward_pass_generator.BackwardGenerator`.
      Main subtleties here are clarified in :ref:`mod_extending`. Note that this includes a recursive call for
      ``NestedSDFG`` nodes (forwarding intermediate values is a source of complexity here).

    * Connect required inputs. This includes gradients of outputs of the node, as well as the values of inputs of the
      node (which potentially need to be routed through reversed maps, or through the hierarchy of ``NestedSDFG`` s).

.. _mod_extending:

Extending the Engine
--------------------

When attempting to differentiate a ``LibraryNode``, the engine will recursively expand the node until it is in a form
that the engine can differentiate. Usually, this means that the engine will expand the node down to the "pure"
implementation consisting of simple tasklets and maps.

However, it is sometimes desirable to "exit" this expansion process at a stage earlier than the lowest level.
For instance, consider differentiating the :class:`~daceml.onnx.nodes.onnx_op.ONNXMatMul` library node. Since no
backward implementation exists for this node, it will be expanded to its pure version, an
:class:`~daceml.onnx.nodes.onnx_op.ONNXEinsum`. Fully expanding this node into its pure form would result in a mapped
tasklet, which we could differentiate. However, we would like to use BLAS nodes on the forward and backward pass where
possible. To achieve this, a custom backward implementation is registered for
:class:`~daceml.onnx.nodes.onnx_op.ONNXEinsum`, which returns a ``NestedSDFG`` containing other einsums. Since we avoid
lowering to the lowest level, we are able to preserve information, and can later potentially expand both the forward and
backward pass einsums to more efficient BLAS calls.

Another example is :class:`~daceml.onnx.nodes.onnx_op.ONNXSoftmax`: a typical implementation includes a maximum
operation for numerical stablility. Differentiating this implementation results in several argmax calls, which is not
desirable.

In situations like these, it makes sense to provide a custom backward pass implementation.

These implementations are registered using :class:`~daceml.autodiff.BackwardImplementation`. This requires
implementation of :meth:`~Daceml.autodiff.BackwardImplementation.backward`. Examples of this are
:class:`daceml.autodiff.implementations.onnx_ops.DefaultEinsumBackward` and
:class:`daceml.autodiff.implementations.onnx_ops.DefaultSoftmaxBackward`.
