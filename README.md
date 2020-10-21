[![Build Status](https://travis-ci.org/spcl/daceml.svg?branch=master)](https://travis-ci.org/spcl/daceml)
[![codecov](https://codecov.io/gh/spcl/daceml/branch/master/graph/badge.svg)](https://codecov.io/gh/spcl/daceml)
[![Documentation Status](https://readthedocs.org/projects/daceml/badge/?version=latest)](https://daceml.readthedocs.io/en/latest/?badge=latest)

# DaceML

*Machine learning powered by data-centric parallel programming.*

This project adds PyTorch and ONNX model loading support to [DaCe](https://github.com/spcl/dace), and adds ONNX
 operator library nodes to the SDFG IR. With access to DaCe's rich transformation library and
productive development environment, **DaceML can generate highly efficient implementations that can be executed on CPUs, GPUs
and FPGAs.**

The white box approach allows us to see computation at all levels of granularity: from coarse operators, to kernel
implementations, and even down to every scalar operation and memory access.

Converting PyTorch modules is as easy as adding a decorator...
```python
@dace_module
class Model(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 4, kernel_size)
        self.conv2 = nn.Conv2d(4, 4, kernel_size)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        return F.relu(self.conv2(x))
```
... and ONNX models can also be directly imported using the model loader:
```python
model = onnx.load(model_path)
dace_model = ONNXModel("mymodel", model)
```

*Read more @ [daceml.readthedocs.io](https://daceml.readthedocs.io)*
