[![Build Status](https://travis-ci.com/spcl/daceml.svg?branch=master)](https://travis-ci.com/spcl/daceml)
[![codecov](https://codecov.io/gh/spcl/daceml/branch/master/graph/badge.svg)](https://codecov.io/gh/spcl/daceml)
[![Documentation Status](https://readthedocs.org/projects/daceml/badge/?version=latest)](https://daceml.readthedocs.io/en/latest/?badge=latest)

# DaceML

*Machine learning powered by data-centric parallel programming.*

This project adds PyTorch and ONNX model loading support to [DaCe](https://github.com/spcl/dace), and adds ONNX
 operator library nodes to the SDFG IR. With access to DaCe's rich transformation library and
productive development environment, **DaceML can generate highly efficient implementations that can be executed on CPUs, GPUs
and FPGAs.**

The white box approach allows us to see computation at **all levels of granularity**: from coarse operators, to kernel
implementations, and even down to every scalar operation and memory access.

![IR visual example](doc/ir.png)

## Library Nodes
DaceML extends the DaCe IR with machine learning operators. The added nodes perform computation as specificed by the
ONNX specification. DaceML leverages high performance kernels from ONNXRuntime, as well as pure SDFG implementations
that are introspectable and transformable with data centric transformations.

*Read more: [Library Nodes](https://daceml.readthedocs.io/en/latest/overviews/onnx.html#library-nodes)*
## Integration
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

*Read more: [PyTorch Integration](https://daceml.readthedocs.io/en/latest/overviews/pytorch.html) and 
[Importing ONNX models](https://daceml.readthedocs.io/en/latest/overviews/onnx.html#importing-onnx-models).*


## Setup
The easiest way to get started is to run

    make install
    
This will setup DaceML in a newly created virtual environment.

*For more detailed instructions, including ONNXRuntime installation, see [Installation](https://daceml.readthedocs.io/en/latest/overviews/installation.html).*

## Development
Common development tasks are automated using the `Makefile`. 
See [Development](https://daceml.readthedocs.io/en/latest/overviews/development.html) for more information.
