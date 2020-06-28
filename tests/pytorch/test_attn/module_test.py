import torch
import torch.nn as nn
import torch.nn.functional as F
import onnx
import numpy as np
import onnxruntime as ort

import dace
from dace.frontend.onnx import ONNXModel
from dace.frontend.pytorch import DACEModule




B = 2
H = 16
P = 64
N = P*H
SM, SN = 512, 512
#K, Q, V = (torch.randn([SM, B, N], requires_grad=True).cuda(),
#           torch.randn([SN, B, N], requires_grad=True).cuda(),
#           torch.randn([SM, B, N], requires_grad=True).cuda())
K, Q, V = (torch.randn([SM, B, N]),
           torch.randn([SN, B, N]),
           torch.randn([SM, B, N]))

ptmodel = torch.nn.MultiheadAttention(N, H, bias=False)


############################ DaCe #################################
my_dace_model = DACEModule(ptmodel, Q, K, V)
dace_outputs = my_dace_model(Q.numpy(), K.numpy(), V.numpy()) 
print(dace_outputs[0])

############################## Ort #################################
sess = ort.InferenceSession("shape_infer.onnx")
ort_outputs = sess.run(None, {"actual_input_1": Q.numpy(), "key": K.numpy(), "value": V.numpy()})
print(ort_outputs[0])

############################## Torch ################################
pt_outputs = ptmodel(Q, K, V)
print(pt_outputs[0])


assert np.allclose(dace_outputs[0], ort_outputs[0])
assert np.allclose(dace_outputs[1], ort_outputs[1])
#assert np.allclose(pt_outputs[0].detach().numpy(), ort_outputs[0], rtol=1e-01)
#assert np.allclose(pt_outputs[1].detach().numpy(), ort_outputs[1], rtol=1e-01)
