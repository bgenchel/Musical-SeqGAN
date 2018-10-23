import json
import os
import os.path as op
import torch

from generator import Generator

model_dir = op.join('runs', 'Oct18-18_08:49:52')
model_inputs = json.load(open(op.join(model_dir, 'model_inputs.json'), 'r'))
model_state = torch.load(op.join(model_dir, 'generator_state.pt'), map_location='cpu')
gen = Generator(**model_inputs)
gen.load_state_dict(model_state)
gen.eval()

output = gen.sample(batch_size=1, seq_len=100)
print(output.size())
print(output)
