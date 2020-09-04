[![Build Status](https://travis-ci.org/spcl/daceml.svg?branch=master)](https://travis-ci.org/spcl/daceml)
[![codecov](https://codecov.io/gh/spcl/daceml/branch/master/graph/badge.svg)](https://codecov.io/gh/spcl/daceml)
# daceml
Machine learning powered by DaCe. 

This project add ONNX operators as library nodes to DaCe, as well as a frontend to load ONNX models.

## Setup
ONNX ops can be run using the ONNX Runtime as a backend. This requires the `ONNXRuntime` environment to be set up. There are two options for this

### Download prebuilt release (CPU only)
Run the following commands 
	
	wget https://github.com/orausch/onnxruntime/releases/download/build1/onnxruntime_dist_cpu.tar.gz
	tar -xzf onnxruntime_dist_cpu.tar.gz

Finally, set the environment variable `ORT_RELEASE` to the location of the extracted file (for example: `export ORT_RELEASE=onnxruntime_dist_cpu`).

### Build from source
To do this, clone the [patched onnxruntime](https://github.com/orausch/onnxruntime), and run the following commands

	./build.sh --build_shared_lib --parallel --config Release

To enable CUDA, you should add the relevant arguments. For example:

	./build.sh --use_cuda --cuda_version=10.1 --cuda_home=/usr/local/cuda --cudnn_home=/usr/local/cuda --build_shared_lib --parallel --config Release

See ``onnxruntime/BUILD.md`` for more details.

Finally, set the environment variable `ORT_ROOT` to the location of the built repository.


## Importing ONNX models
ONNX models can be imported using the `ONNXModel` frontend.
```python
import onnx
from dace.frontend.onnx import ONNXModel

model = onnx.load("resnet50.onnx")
dace_model = ONNXModel("MyResnet50", model)

test_input = np.random.rand(1, 3, 224, 224).astype(np.float32)
output = dace_model(test_input)
```

## Library Nodes
The ONNX Library nodes implement [ONNX Operators](https://github.com/onnx/onnx/blob/master/docs/Operators.md).

### Example
The following code sets up and runs an SDFG containing an ONNX Convolution Operator
```python
from dace.libraries.onnx.nodes.onnx_op import ONNXConv
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
state.add_edge(access_X, None, conv, "X", sdfg.get_array_memlet("X_arr"))
state.add_edge(access_W, None, conv, "W", sdfg.get_array_memlet("W_arr"))
state.add_edge(conv, "Y", access_Z, None, sdfg.get_array_memlet("Z_arr"))

X = np.random.rand(5, 3, 10, 10).astype(np.float32)
W = np.random.rand(16, 3, 3, 3).astype(np.float32)
Z = np.zeros((5, 16, 8, 8)).astype(np.float32)

sdfg(X_arr=X, W_arr=W, Z_arr=Z)
```
### Parameters (Inputs and Outputs)
The parameters for an op are specified by adding a connector with the name of the parameter. By default, ops already have connectors for the required parameters.

### Variadic Parameters
Variadic parameters are specified by adding a `__` followed by the index of the variadic parameter. For example, the [Sum operator](https://github.com/onnx/onnx/blob/master/docs/Operators.md#sum) has a variadic input named `data_0`. If we wanted to add 3 inputs that connect to this variadic parameter, we would add the connectors: `data_0__0`, `data_0__1` and `data_0__2`. The indices after `__` specify the order of the variadic parameters.

Notes:
* the indices should not have leading zeros
* the indices should not contain any gaps. For example, adding connectors `data_0__0`, `data_0__2` to the `Sum` operator is invalid because the parameter with index 1 is missing.

### Attributes
You can set attributes by passing them to the `ONNXConv` constructor. For example
```python
conv = ONNXConv("MyConvNode", strides=[2, 2])
```

## Development
The `Makefile` contains a few commands for development tasks such as running tests, checking formatting or installing the package.

For example, the following command would install the package and run tests
	VENV_PATH='' make install test

If you would like to create a virtual environment and install to it, remove `VENV_PATH=''` from the above command
