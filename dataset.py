import torch
import tiktoken
import pandas as pd

class DataLoaderLite:
    """
    Download Spotify Million Song Dataset from here:
    https://www.kaggle.com/datasets/notshrirang/spotify-million-song-dataset
    """
    def __init__(self, data_dir, B, T, process_rank, num_processes, master_process):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        df = pd.read_csv(data_dir)
        text = df["text"].str.cat(sep="\n")

        enc = tiktoken.get_encoding("gpt2")
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)
        print(len(self.tokens))
        self.current_position = 0

    def next_batch(self):
        B, T = self.B, self.T

        # buffer: (B*T+1,)
        buf = self.tokens[self.current_position: self.current_position + B*T+1]
        x = buf[:-1].view(B, T) # (B*T,) -> (B, T)
        y = buf[1:].view(B, T) # (B*T,) -> (B, T)

        self.current_position += B * T * self.num_processes

        # this will discard last buffer if remained buffer size is less than B*T+1
        if self.current_position + B * T * self.num_processes + 1 > len(self.tokens):
            self.current_position = 0

        return x, y
