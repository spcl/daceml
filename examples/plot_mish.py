"""
Optimizing the Mish Operator
============================

DaCeML allows users to optimize DNN modules at all levels of granularity, from operators to full models. In this
example, we optimize the Mish operator [1]_, a relatively novel activation function that,
among other uses, has been applied successfully in image segmentation. [2]_

Due to its novelty, it has, at the time of writing, not been implemented in PyTorch, ONNX or ONNX Runtime. We
demonstrate how DaCeML can be used to optimize this operator.

.. [1] Diganta Misra. Mish: A self regularized non-monotonic activation function. In 31st British Machine Vision
   Conference 2020, BMVC 2020, Virtual Event, UK, September 7-10, 2020. BMVA Press, 2020.
.. [2] Alexey Bochkovskiy, Chien-Yao Wang, and Hong-Yuan Mark Liao. Yolov4: Optimal speed and accuracy of object
   detection. CoRR, abs/2004.10934, 2020.
"""

# %%
# We begin with code for the PyTorch Module, and import it into DaCeML by annotating it with the ``@dace_module``
# decorator.
import torch
from torch import nn
from torch.nn import functional as F

from daceml.pytorch import dace_module


@dace_module(cuda=True, backward=True)
class DaCeMish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x * (torch.tanh(F.softplus(x)))
        return x


# %%
# The module works immediately with DaCeML for the forward pass.
#
# The first time we tested this, we found that the automatic differentiation failed due to an missing
# pure implementation for :class:`~daceml.onnx.nodes.onnx_op.ONNXSoftplus`. Fortunately, adding these implementations
# is easily done using the DaCe python frontend. The following code shows the pure implementation that was added.
#
# .. code-block:: python
#
#    @python_pure_op_implementation
#    def Softplus(X, Y):
#        Y[:] = np.log(1 + np.exp(X))

# %%
# Let's test the operator and compare with a PyTorch version


class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x * (torch.tanh(F.softplus(x)))
        return x


# create test inputs (size taken from YOLOv4)
with torch.no_grad():
    dace_input = torch.rand(8, 32, 224, 224).cuda()
    torch_input = torch.clone(dace_input)
    dace_dy = torch.rand(8, 32, 224, 224).cuda()
    torch_dy = torch.clone(dace_dy)

dace_input.requires_grad = True
torch_input.requires_grad = True

torch_mish = Mish().cuda()
dace_mish = DaCeMish()

dace_output = dace_mish(dace_input)
dace_output.backward(dace_dy)
torch_output = torch_mish(torch_input)
torch_output.backward(torch_dy)

assert torch.allclose(dace_output, torch_output)
assert torch.allclose(dace_input.grad, torch_input.grad)

# %%
# Let's profile this implementation
from daceml.testing.profiling import time_funcs, print_time_statistics


def run_dace():
    out = dace_mish(dace_input)
    out.backward(dace_dy)


def run_torch():
    out = torch_mish(torch_input)
    out.backward(torch_dy)


times = time_funcs([run_dace, run_torch],
                   func_names=["dace", "torch"],
                   warmups=5,
                   num_iters=100)
print_time_statistics(times, ["dace", "torch"])

# %%
# Inspection
# ------------

# Let's inspect the forward pass SDFG first.
dace_mish.forward_sdfg

# %%
# We can see that there is a lot of unnecessary data movement on the forward pass. Fusing the different maps would
# greatly improve runtime.
#
# Now let's look at the backward pass.

dace_mish.backward_sdfg

# %%
# We also see another opportunity for optimization: The DaCeML autodiff engine is "forwarding" intermediate values to
# perform the differentiation. This means that the intermediate values have to be written out in the forward pass, and
# and read in the backward pass.
#
# Optimization
# ------------
#
# To improve the runtime, we'll apply 3 transformations.
#
# Firstly, we'll use ``SubgraphFusion`` to fuse all the maps into a single kernel.
# To tackle the second issue, we'll use the :class:`~daceml.transformation.TaskletFusion` transformation. By fusing the
# tasklets into a single tasklet before running automatic differentiation, the engine will differentiate the whole
# expression at once, eliminating the need to access the intermediate values. This is an easy way to tune recomputation
# vs. storage in automatic differentiation.
#
# Finally, we'll apply ``Vectorization`` to make our kernels operate on more than one element at once.

from daceml.transformation import TaskletFusion
from dace.transformation.dataflow import Vectorization, TrivialMapRangeElimination
from dace.transformation.subgraph import SubgraphFusion
from daceml.util import utils
from dace.library import change_default
from daceml import onnx as donnx

# reset the compiled sdfg
dace_mish.reset_sdfg()


# expand the onnx nodes, and apply automatic transformations like inlining
def expand_and_strict_transforms(module):
    # use the pure expansions of operators
    with change_default(donnx, "pure"):
        utils.auto_optimize(module.sdfg, cuda=True, apply_strict=True)


dace_mish.append_post_onnx_hook("auto_optimize", expand_and_strict_transforms)


# apply subgraph fusion
def fuse_sg(module):
    sdfg = module.sdfg
    sdfg.apply_transformations_repeated(TrivialMapRangeElimination)
    SubgraphFusion.apply_to(sdfg, *sdfg.node(0).nodes())


dace_mish.append_post_onnx_hook("subgraph_fusion", fuse_sg)

# apply tasklet fusion
dace_mish.append_post_onnx_hook("fuse_tasklets", lambda x:\
        x.dace_model.sdfg.apply_transformations_repeated(TaskletFusion))


# apply vectorization
def vectorize(fwd, bwd):
    fwd.apply_transformations(Vectorization)
    bwd.apply_transformations(Vectorization)


dace_mish.append_post_autodiff_hook("vectorize", vectorize)

# %%
# Let's check that the new SDFG is still correct.

dace_output = dace_mish(dace_input)
dace_output.backward(dace_dy)
torch_output = torch_mish(torch_input)
torch_output.backward(torch_dy)

assert torch.allclose(dace_output, torch_output)
assert torch.allclose(dace_input.grad, torch_input.grad)

# %%
# After running the module once, we can also inspect the compiled SDFG for the forward and backward pass.

dace_mish.forward_sdfg

# %%
dace_mish.backward_sdfg

# %%
# Now we can profile the optimized module.
times = time_funcs([run_dace, run_torch],
                   func_names=["dace", "torch"],
                   warmups=5,
                   num_iters=100)
print_time_statistics(times, ["dace", "torch"])

# %%
# Let's also try PyTorch JIT compilation.
import torch.jit

torch_jit = torch.jit.trace(Mish(), torch_input)


def run_torch_jit():
    out = torch_jit(torch_input)
    out.backward(torch_dy)


times = time_funcs([run_dace, run_torch, run_torch_jit],
                   func_names=["dace", "torch", "torch_jit"],
                   warmups=5,
                   num_iters=100)
print_time_statistics(times, ["dace", "torch", "torch_jit"])
