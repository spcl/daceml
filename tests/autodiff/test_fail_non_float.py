import pytest
import torch
import torch.nn as nn

from daceml.autodiff import AutoDiffException
from daceml.pytorch import dace_module


def test_fail_non_float():

    with pytest.raises(AutoDiffException) as info:

        @dace_module(backward=True,
                     dummy_inputs=(torch.ones(10, dtype=torch.long), ))
        class MyModule(nn.Module):
            def forward(self, x):
                return x + 1

        MyModule()

    assert "float edges" in str(info.value)
