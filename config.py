from dataclasses import dataclass

@dataclass
class GPTConfig:
    num_layers: int = 12 # number of decoder layer
    num_heads: int = 8 # number of heads(M) in MHSA
    vocab_size: int = 50257 # number of tokens
    block_size: int = 1024 # sequence length(T) of input
    emb_dim: int = 768 # embedding dimension(C) in Attention