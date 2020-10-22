Automatic Differentiation
=========================

.. warning::

    The symbolic automatic differentiation feature still experimental.

DaceML takes a different approach to automatic differentation than most deep learning frameworks. Instead of
hand-writing backward passes for all differentiable operators, DaceML has a symbolic reverse-mode differentation engine.

Using Autodiff
--------------
There are two main ways to generate backward passes in DaceML.

:class:`~daceml.pytorch.DaceModule`
    This class includes a ``backward`` parameter. If ``True``, the autodiff engine will be used to add a backward pass
    to the PyTorch module, and the resulting module can be seamlessly used with other PyTorch code. For example:

    TODO Make this example cooler (with parameters?)

    .. testcode::
        from daceml.pytorch import DaceModule

        class Module(torch.nn.Module):
            def forward(self, x):
                x = torch.log(x)
                return x

        dace_module = DaceModule(Module(), backward=True)

        x = torch.rand(5, 5, requires_grad=True)
        y = dace_module(x).sum()

        # gradients can flow through dace_module!
        y.backward()

:func:`~daceml.autodiff.add_backward_pass`

    The autodiff engine can also be run on plain SDFGs.

    .. testcode::

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
                s >> S(1, lambda x, y: x + y)[0]
                z << Z[i, j]
                s = z

        sdfg = dace_gemm.to_sdfg()

        add_backward_pass(sdfg=sdfg, state=sdfg.nodes()[0], ["S"], ["X", "Y"])


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
      node (which potentially need to be routed through reversed maps).

.. _mod_extending:

Extending the Engine
--------------------
The automatic differentiation engine currently has several limitations that may cause it to be unable to differentiate
certain library nodes. An example is :class:`~daceml.onnx.ONNXSoftmax`; a typical implementation includes a maximum
operation for numerical stablility. Differentiating this implementation results in several argmax calls, which is not
desirable. In situations like these, it makes sense to provide a custom backward pass implementation.

These implementations are registered using :class:`~daceml.autodiff.BackwardImplementation`. This requires implementation
of :meth:`~Daceml.autodiff.BackwardImplementation.backward`. TODO Insert example here.
