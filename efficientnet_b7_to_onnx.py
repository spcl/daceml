import torch
from efficientnet_pytorch import EfficientNet

model = EfficientNet.from_pretrained('efficientnet-b0')
dummy_input = torch.randn(1, 3, 240, 240)

model.set_swish(memory_efficient=False)

torch.onnx.export(
    model, 
    dummy_input,
    f="efficientnet_b0.onnx",  
    do_constant_folding=True, 
    opset_version=12, 
)
