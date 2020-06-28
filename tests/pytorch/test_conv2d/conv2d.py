import torch
import torch.nn as nn
import torch.nn.functional as F
import onnx
import numpy as np
import onnxruntime as ort

import dace
from dace.frontend.onnx import ONNXModel
from dace.frontend.pytorch import DACEModule

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(1, 4, 3)
        self.conv2 = nn.Conv2d(4, 4, 3)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        return F.relu(self.conv2(x))


ptmodel = Model()
dummy_input = torch.rand(1, 1, 8, 8)
x = torch.rand(1, 1, 8, 8)
np_input = x.numpy()

############################ DaCe #################################
my_dace_model = DACEModule(ptmodel, dummy_input)
dace_output = my_dace_model(np_input)
print(dace_output)

############################# Ort #################################
ort_session = ort.InferenceSession('shape_infer.onnx')
ort_output = ort_session.run(None, {'actual_input_1': np_input})
print(ort_output)

############################# Torch ################################
torch_output = ptmodel(x)
print(torch_output.detach().numpy())

assert np.allclose(ort_output, dace_output)
assert np.allclose(torch_output.detach().numpy(), ort_output)
assert np.allclose(torch_output.detach().numpy(), dace_output)
