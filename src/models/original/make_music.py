import json
import os
import os.path as op
import numpy as np
import torch

from generator import Generator
from reverse_pianoroll import piano_roll_to_pretty_midi

model_dir = op.join('runs', 'Nov01-18_01:43:28')
model_inputs = json.load(open(op.join(model_dir, 'model_inputs.json'), 'r'))
model_state = torch.load(op.join(model_dir, 'generator_state.pt'), map_location='cpu')
gen = Generator(**model_inputs)
gen.load_state_dict(model_state)
gen.eval()

output = torch.squeeze(gen.sample(batch_size=1, seq_len=100))
print(output)

pr = np.zeros([128, len(output)])
for i in range(len(output)):
    pr[output[i], i] = 1

pm = piano_roll_to_pretty_midi(pr, fs=1/0.4)
pm.write('midi_out2.mid')