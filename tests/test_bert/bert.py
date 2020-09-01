import dace
from dace.frontend.onnx import ONNXModel
from dace import Config
import pickle
import numpy as np
import onnx
print("sync:", Config.get("compiler", "cuda", "syncdebug"))
print("streams:", Config.get("compiler", "cuda", "max_concurrent_streams"))
print("build_type :", Config.get("compiler", "build_type"))
model = onnx.load("bert_infer.onnx")
#model = onnx.load("bert_opt.onnx")




dace_model = ONNXModel("bert", model, cuda=False)
dace_model.sdfg.states()[0].instrument = dace.InstrumentationType.Timer
prefix = "./data"
feed = {
        "input_ids:0": np.load(prefix + "/input_ids.npy"),
        "input_mask:0": np.load(prefix + "/input_mask.npy"),
        "segment_ids:0": np.load(prefix + "/segment_ids.npy"),
        "ONNX_OneHot216_o0__d0": 2
}
outputs = dace_model(**feed)
unstack_0 = np.load(prefix + "/unstack_0.npy")
unstack_1 = np.load(prefix + "/unstack_1.npy")

print(unstack_0.shape)
print(unstack_1.shape)

print("Close: {}, {}".format(np.allclose(outputs[1], unstack_0), np.allclose(outputs[0], unstack_1)))

darray = np.isclose(unstack_0, outputs[1])
error_count = 0
for i in range(darray.shape[0]):
    for j in range(darray.shape[1]):
        if darray[i][j] == False:
            error_count += 1
            print(i, j, ': onnxrt = ', unstack_0[i][j], 'dace = ', outputs[1][i][j])

print("error number 0 = ", error_count)

darray = np.isclose(unstack_1, outputs[0])
error_count = 0
for i in range(darray.shape[0]):
    for j in range(darray.shape[1]):
        if darray[i][j] == False:
            error_count += 1
            print(i, j, ': onnxrt = ', unstack_1[i][j], 'dace = ', outputs[0][i][j])
print("error number 1 = ", error_count)
