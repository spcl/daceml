import dace
import numpy as np

def initialize():
    input_ids = np.random.randint(low=1, high=4000, size=(1, 256), dtype=np.int32)
    token_type_ids = np.zeros((1, 256), dtype=np.int32)
    attention_mask = np.ones((1, 256), dtype=np.int32)
    return input_ids, token_type_ids, attention_mask

input_ids, token_type_ids, attention_mask = initialize()

sdfg = dace.SDFG.from_file("efficientnet_b7.sdfg")
sdfg.expand_library_nodes()

output = sdfg(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
print(output)
