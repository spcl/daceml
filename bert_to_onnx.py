from daceml.onnx import ONNXModel

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# load model and tokenizer
model_id = "bert-base-uncased"
model = AutoModelForSequenceClassification.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)
dummy_model_input = tokenizer("This is a sample", return_tensors="pt")

input_ids = torch.randint(low=1, high=4000, size=(1, 256), dtype=torch.int)
token_type_ids = torch.zeros((1, 256), dtype=torch.int)
attention_mask = torch.ones((1, 256), dtype=torch.int)

# export
torch.onnx.export(
    model, 
    (input_ids, token_type_ids, attention_mask),
    f="bert-base-uncased_1.onnx",  
    input_names=['input_ids', "token_type_ids", 'attention_mask'], 
    output_names=['logits'], 
    # dynamic_axes={'input_ids': {0: 'batch_size'}, 
    #               'attention_mask': {0: 'batch_size'}, 
    #               'logits': {0: 'batch_size'}}, 
    do_constant_folding=True, 
    opset_version=12, 
)
