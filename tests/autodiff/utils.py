from functools import reduce

import dace.sdfg.nodes as nd
import numpy as np
import pytest
import torch

from daceml.autodiff import add_backward_pass


@pytest.mark.skip()
def test_correctness(func):
    def test_correctness():
        runner, pytorch_func, inputs = func()
        sdfg_dict = {name: arr.copy() for name, arr in inputs.items()}
        torch_dict = {
            name: torch.tensor(arr.copy(), requires_grad=True)
            for name, arr in inputs.items()
        }

        sdfg_results = runner.run(**sdfg_dict)
        torch_results = pytorch_func(**torch_dict)

        for k, v in torch_results.items():
            v = v.detach().numpy()
            diff = np.linalg.norm(sdfg_results[k] - v) / reduce(
                lambda x, y: x * y, v.shape)

            print("-" * 10, k, "-" * 10)
            print("Difference:", diff)

            print("Torch results:", "-" * 10)
            print(v)
            print("SDFG results:", "-" * 10)
            print(sdfg_results[k])

            assert diff < 1e-5

    return test_correctness


class SDFGBackwardRunner:
    def __init__(self, sdfg, target, strict=True):
        if strict:
            sdfg.apply_strict_transformations()
        self.sdfg = sdfg
        self.target = target
        state = sdfg.nodes()[0]
        required_grads = list(node for node in state.source_nodes()
                              if isinstance(node, nd.AccessNode))

        add_backward_pass(self.sdfg, state, [self.target], required_grads)
        self.sdfg.apply_strict_transformations()
        self.debug = False

    def run(self, **inputs):

        # zero out all arrays
        intermediate_arrs = {
            name: np.zeros(arr.shape, dtype=getattr(np, arr.dtype.to_string()))
            for name, arr in self.sdfg.arrays.items()
            if name != self.target + "_grad" if not name.startswith("__")
            if name not in inputs if not arr.transient
        }
        inputs.update(intermediate_arrs)
        inputs[self.target + "_grad"] = np.ones(
            (1, ),
            dtype=getattr(np, self.sdfg.arrays[self.target].dtype.to_string()))

        print("Pre-execution arrays")
        print("-" * 10)
        for k, v in inputs.items():
            print(k, "-" * 10)
            print(v.dtype)
            print("is_contiguous:", v.flags['C_CONTIGUOUS'])
            print(v)

        self.sdfg(**inputs)

        print("Post-execution arrays")
        print("-" * 10)
        for k, v in inputs.items():
            print(k, "-" * 10)
            print(v.dtype)
            print("is_contiguous:", v.flags['C_CONTIGUOUS'])
            print(v)

        results = {
            name: arr
            for name, arr in inputs.items()
            # if name.endswith("_grad") and name != self.target + "_grad"
        }
        return results
