import tiktoken
import inspect
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


class CausalSelfAttention(nn.Module):
    """
    Causal MultiHeadSelfAttention(MHSA) layer
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_heads = config.num_heads
        self.emb_dim = config.emb_dim

        # c_attn for query, key, value
        self.c_attn = nn.Linear(config.emb_dim, 3 * config.emb_dim, bias=True)
        # projection layer at the last of attention computation
        self.c_proj = nn.Linear(config.emb_dim, config.emb_dim, bias=True)
        # 🐣 modified initialization which accounts for the accumulation on the residual path
        self.c_proj.RESIDUAL_SCALE_INIT = 1 # flag

        # mask for Masked Attention (bias: (1, 1, T, T))
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size()

        # query, key, value all with shape of (B, T, C)
        q, k, v = self.c_attn(x).split(self.emb_dim, dim=-1)
        # query, key, value: (B, T, C) -> (B, T, M, C//M) -> (B, M, T, C//M)
        q = q.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)
        k = k.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)
        v = v.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)

        # # MHSA mechanism (B, M, T, T)
        # # ❗need to divide by (M//C) for learning stability in SoftMax --> see q.var(), k.var(), attn_weight.var()❗
        # attn_weight = (q @ k.transpose(-2, -1)) * (1 / math.sqrt(q.size(-1)))
        # attn_weight = attn_weight.masked_fill(self.bias[:,:,:T,:T] == 0, float("-inf"))
        # attn_weight = F.softmax(attn_weight, dim=-1)
        # out = attn_weight @ v # (B, M, T, C//M)

        # ⚡️FlashAttention
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        out = out.transpose(1,2).contiguous().view(B, T, C) # (B, T, C)
        out = self.c_proj(out)
        return out


class MLP(nn.Module):
    """
    MultiLayer Perceptron(MLP) layer
    """
    def __init__(self, config):
        super().__init__()

        # fc layer
        self.c_fc = nn.Linear(config.emb_dim, 4 * config.emb_dim, bias=True)
        # GELU non-linearity
        self.gelu = nn.GELU(approximate='tanh')
        # projection layer
        self.c_proj = nn.Linear(4 * config.emb_dim, config.emb_dim, bias=True)
        # 🐣 modified initialization which accounts for the accumulation on the residual path
        self.c_proj.RESIDUAL_SCALE_INIT = 1  # flag

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x



class Block(nn.Module):
    """
    (LN - MHSA - LN - MLP) block
    """
    def __init__(self, config):
        super().__init__()

        self.ln_1 = nn.LayerNorm(config.emb_dim)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.emb_dim)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


@dataclass
class GPTConfig:
    num_layers: int = 12 # number of decoder layer
    num_heads: int = 8 # number of heads(M) in MHSA
    vocab_size: int = 50257 # number of tokens
    block_size: int = 1024 # sequence length(T) of input
    emb_dim: int = 768 # embedding dimension(C) in Attention


class nanoGPT2(nn.Module):
    """
    This nanoGPT2 architecture is same as huggingface GPT2LMHeadModel to get pre-trained weights
    see: https://huggingface.co/docs/transformers/v4.45.1/en/model_doc/gpt2#transformers.GPT2LMHeadModel
    """
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Transformer Decoders
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.emb_dim), # (B, T) -> (B, T, C)
            wpe = nn.Embedding(config.block_size, config.emb_dim), # (T,) -> (T, C)
            h = nn.ModuleList([Block(config) for _ in range(config.num_layers)]),
            ln_f = nn.LayerNorm(config.emb_dim),
        ))
        # Final Linear Head
        self.lm_head = nn.Linear(config.emb_dim, config.vocab_size, bias=False) # (B, T, C) -> (B, T, vocab_size)

        # 🦄weight sharing scheme (saved 30% of the total parameters)
        self.transformer.wte.weight = self.lm_head.weight

        # 🐣init parameters
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, "RESIDUAL_SCALE_INIT"):
                std += (2 * self.config.num_heads) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x, targets=None):
        B, T = x.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"

        # Positional + Token Embedding
        pos = torch.arange(0, T, dtype=torch.long, device=x.device) #❗️new tensor should be on the same device❗
        pos_emb = self.transformer.wpe(pos)
        token_emb = self.transformer.wte(x)
        x = pos_emb + token_emb

        # Transformer Decoders
        for block in self.transformer.h:
            x = block(x)

        # Linear Head
        logits = self.lm_head(x) # (B, T, vocab_size)
        loss = None
        if targets is not None:
            # logits: (B, T, vocab_size) -> (B * T, vocab_size)
            # targets: (B, T) -> (B * T,)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    def configure_optimizer(self, weight_decay, learning_rate, device):
        # all parameters that requires_grad
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}

        # any parameters that is 2D will be weight decayed, otherwise no
        # e.g. all biases and layer_norms do not decay
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")

        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters # check if possible b/c it's brand new
        use_fused = fused_available and 'cuda' in device
        print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, weight_decay=weight_decay, fused=use_fused)
        return optimizer

    @classmethod
    def from_pretrained(cls, model_type):
        """
        Load pre-trained GPT-2 Model weights from huggingface GPT2LMHeadModel
        """
        assert model_type in ['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl']
        from transformers import GPT2LMHeadModel
        print(f"loading weights from pretrained GPT: {model_type}")

        config_args = {
            'gpt2': dict(num_layers=12, num_heads=12, emb_dim=768),  # 124M params
            'gpt2-medium': dict(num_layers=24, num_heads=16, emb_dim=1024),  # 350M params
            'gpt2-large': dict(num_layers=36, num_heads=20, emb_dim=1280),  # 774M params
            'gpt2-xl': dict(num_layers=48, num_heads=25, emb_dim=1600),  # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257
        config_args['block_size'] = 1024

        # from scratch model
        config = GPTConfig(**config_args)
        model = nanoGPT2(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith(".attn.bias")]  # ignore .attn.bias buffer

        # model from huggingface
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith(".attn.bias")] # ignore .attn.bias buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith(".attn.masked_bias")]  # ignore .attn.masked_bias buffer

        #  ❗basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        #  this means that we have to transpose these weights when we import them❗
        need_transpose = ['.attn.c_attn.weight', '.attn.c_proj.weight', '.mlp.c_fc.weight', '.mlp.c_proj.weight']

        # copy sd_hf -> sd
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in need_transpose):
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

class DataLoaderLite:
    # can get shakespeare.txt by using this command:
    # wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
    def __init__(self, B, T):
        self.B = B
        self.T = T

        with open("shakespeare.txt", "r") as f:
            text = f.read()
        enc = tiktoken.get_encoding("gpt2")
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens) # (338025,)

        self.current_position = 0

    def next_batch(self):
        B, T = self.B, self.T

        # buffer: (B*T+1,)
        buf = self.tokens[self.current_position: self.current_position + B*T+1]
        x = buf[:-1].view(B, T) # (B*T,) -> (B, T)
        y = buf[1:].view(B, T) # (B*T,) -> (B, T)

        self.current_position += B * T

        # this will discard last buffer if remained buffer size is less than B*T+1
        if self.current_position + B * T + 1 > len(self.tokens):
            self.current_position = 0

        return x, y




if __name__ == '__main__':

    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(0)

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device="mps"
    print(f"using device: {device}")

    B = 8
    T = 1024

    max_lr = 6e-4
    min_lr = max_lr * 0.1
    warmup_steps = 10
    max_steps = 50

    # Data Loader
    train_loader = DataLoaderLite(B=B, T=T)

    # 👻 use TF32 for matrix multiplication and convolution operation
    torch.set_float32_matmul_precision('high')

    # Model
    model = nanoGPT2(GPTConfig(vocab_size=50304)) # ✊ more beautiful number
    model = model.to(device)

    # ☠️ model compile (think of it like gcc; torch>=2.0.0)
    model = torch.compile(model)

    print("Fused Implementation")

    # LR scheduler
    def get_lr(iter):
        """
        CosineLR scheduler with LinearWarmup from scratch
        """
        if iter < warmup_steps:
            return max_lr * (iter + 1) / warmup_steps
        if iter > max_steps:
            return min_lr
        decay_ratio = (iter - warmup_steps) / (max_steps - warmup_steps)
        assert 0<= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))

        return min_lr + coeff * (max_lr - min_lr)

    # Optimizer
    # optimizer = torch.optim.AdamW(model.parameters(), lr=6e-4, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)

    # 👬 optimizer with parameter grouping (fused implementation)
    optimizer = model.configure_optimizer(weight_decay=0.1, learning_rate=6e-4, device=device)

    # optimize!!!
    for step in range(max_steps):
        t0 = time.time() # t0 --------------------------------------------------------------------
        optimizer.zero_grad()

        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)

        # 👽 use mixed precision of FP32 and BF32 as a tensor format
        # ❗️must use scaler when using FP16 as it truncates exponent (range) part❗️
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits, loss = model.forward(x, y)

        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        loss.backward()

        # ✂️ gradient clipping after calculating all gradients
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        torch.cuda.synchronize() # wait GPU to finish all the above works scheduled before
        t1 = time.time() # t1 --------------------------------------------------------------------

        # calculate some useful metrics
        dt = (t1 - t0) * 1000
        tokens_processed = train_loader.B * train_loader.T
        tokens_per_sec = tokens_processed / dt
        print(f"step: {step} | loss: {loss.item():.6f} | norm:{norm:.4f} | lr: {lr:.4e} | dt: {dt:.2f}ms | tok/sec: {tokens_per_sec:.2f}")




