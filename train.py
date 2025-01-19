import os
import math
import time
import torch
import torch.distributed as dist
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

from model import GPT2
from config import GPTConfig
from dataset import DataLoaderLite


# setup DistributedDataParallel(DDP)
ddp = int(os.environ.get('RANK', -1)) != -1
if ddp:
    # DDP mode
    assert torch.cuda.is_available(), "need CUDA for DDP"
    init_process_group(backend='nccl')  # <-> destroy_process_group
    ddp_rank = int(os.environ['RANK'])  # all processes will have different ddp_rank
    ddp_local_rank = int(os.environ['LOCAL_RANK'])  # ❗️need ddp_local_rank for multi-node❗️
    ddp_world_size = int(os.environ['WORLD_SIZE'])  # total number of processes running
    device = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0
else:
    # non-DDP mode
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    print(f"using device: {device}")

torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed(0)

total_batch_size = 524288  # 2^19 ~0.5M, total number of tokens per batch
B = 8  # micro batch
T = 1024  # sequence length
assert total_batch_size % (
            B * T * ddp_world_size) == 0, "make sure total_batch_size is divisible by B * T * ddp_world_size"
grad_accum_steps = total_batch_size // (B * T * ddp_world_size)
if master_process:
    print(f"total desired batch size: {total_batch_size}")
    print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

epoch = 50
max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 715  # 375e6 / 524288 = 715 (according to GPT2 paper, they warm up the lr with 375e6 tokens)
max_steps = 13 * epoch  # 70426172 / 524288 = 13.xx

# Data Loader
train_loader = DataLoaderLite(data_dir="data/spotify_millsongdata.csv", B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, master_process=master_process)

# use TF32 for matrix multiplication and convolution operation
torch.set_float32_matmul_precision('high')

# Model
model = GPT2(GPTConfig(vocab_size=50304))  # ✊ more beautiful number
model = model.to(device)

# # model compile (think of it like gcc; torch>=2.0.0)
# model = torch.compile(model) # TODO: How can I save and load compiled model??

# wrap the model with DDP
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank],
                find_unused_parameters=True)  # ❗️device_ids should be a ddp_local_rank NOT ddp_rank❗
raw_model = model.module if ddp else model


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
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))

    return min_lr + coeff * (max_lr - min_lr)

# optimizer with parameter grouping (fused implementation)
optimizer = raw_model.configure_optimizer(weight_decay=0.1, learning_rate=6e-4, device=device)

# optimize!!!
for step in range(max_steps):
    t0 = time.time()  # t0 --------------------------------------------------------------------
    optimizer.zero_grad()

    loss_accum = 0.0
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)

        # 👽 use mixed precision of FP32 and BF32 as a tensor format
        # ❗️must use scaler when using FP16 as it truncates exponent (range) part❗️
        with torch.autocast(device_type='cuda' if 'cuda' in device else 'cpu', dtype=torch.bfloat16):
            logits, loss = model.forward(x, y)
        loss = loss / grad_accum_steps  # ❗️F.cross_entropy() has reduction='mean' by default❗
        loss_accum += loss.detach()  # only for logging

        # 🦈 all gradients in different GPUs will be synchronized(averaged out) only in the last micro_step
        if ddp:
            model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
        loss.backward()  # this will accumulate(+=) the gradients

    # this will synchronize(average out) all losses in different GPUs
    if ddp:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)

    # gradient clipping after calculating all gradients
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    optimizer.step()
    torch.cuda.synchronize()  # wait GPU to finish all the above works scheduled before
    t1 = time.time()  # t1 --------------------------------------------------------------------

    # calculate some useful metrics
    dt = (t1 - t0) * 1000
    tokens_processed = train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size
    tokens_per_sec = tokens_processed / dt

    if master_process:
        print(
            f"step: {step:4d} | loss: {loss_accum.item():.6f} | norm:{norm:.4f} | lr: {lr:.4e} | dt: {dt:.2f}ms | tok/sec: {tokens_per_sec:.2f}")

# ❗️after training, need to destroy all the processes❗️
if ddp:
    destroy_process_group()

torch.save(raw_model.state_dict(), f"checkpoints/gpt2_lyrics_{epoch}ep.pth")