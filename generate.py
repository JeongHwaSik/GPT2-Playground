import os
import time
import torch

import tiktoken
from train_gpt2 import GPT2, GPTConfig

enc = tiktoken.get_encoding("gpt2")

config = GPTConfig
config.vocab_size = 50304
model = GPT2(config)


# load weights
checkpoint_pth = os.path.join("checkpoints", "gpt2_finewebedu10B_2ep.pth")
model.load_state_dict(torch.load(checkpoint_pth, weights_only=True), strict=False)

# generate
model.eval()

gen_idx = model.generate(torch.zeros((1,1), dtype=torch.long), max_new_tokens=100)[0]

i = 0
while i < len(gen_idx)-1:
    text = enc.decode(list(gen_idx[i:i+2]))
    print(text, end='')
    i += 2
    time.sleep(0.1) # look like GPT text generation haha
print("\n")