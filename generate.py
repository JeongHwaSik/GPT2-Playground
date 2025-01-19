import os
import time
import torch

import tiktoken
from model import GPT2
from config import GPTConfig


config = GPTConfig
config.vocab_size = 50304
model = GPT2(config)


# load weights
checkpoint_pth = os.path.join("checkpoints", "gpt2_lyrics_50ep.pth")
model.load_state_dict(torch.load(checkpoint_pth))

# generate
model.eval()

# get input from the user
text = str(input("What topics do you want to generate?: "))
enc = tiktoken.get_encoding("gpt2")
tokens = enc.encode(text)

gen_idx = model.generate(torch.tensor(tokens, dtype=torch.long).unsqueeze(0), max_new_tokens=400)[0]

i = 1
while i < len(gen_idx)-1:
    text = enc.decode(list(gen_idx[i:i+1]))
    print(text, end="")
    i += 1
    time.sleep(0.2) # look like GPT text generation haha
print("\n")