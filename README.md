# GPT-2 Optimization Analysis with Pytorch

## 🌴 Branch Overview

## 0. Baseline Architecture: GPT-2
I implemented the GPT-2 model from scratch based on the "Attention Is All You Need" paper and the [GPT-2 paper](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf). This serves as the baseline architecture for subsequent optimization analysis. The model's architecture is identical to the GPT2LMHeadModel implemented by the [Hugging Face transformer module](https://huggingface.co/docs/transformers/v4.45.1/en/model_doc/gpt2#transformers.GPT2LMHeadModel). It is a transformer decoder-only model that learns tokens using a bigram-like approach, predicting the next token based on the current one.

## 1. CPU vs. GPU
First, I compared the performance of the CPU, GPU1 (NVIDIA GeForce RTX 3090), and GPU2 (NVIDIA RTX A5000) by training the default GPT2 architecture baseline explained in the Section 2. I used two performance analysis metrics: forward-backward-update duration time (dt) and throughput (tok/sec). The results showed that training on the GPU outperformed the CPU significantly, being x24 faster with x23 higher throughput, which is expected. When comparing the two GPUs, GPU1 slightly outperformed GPU2. This is likely due to GPU1 having nearly 2,000 more CUDA cores and around 200GB/sec higher memory bandwidth than GPU2. 

Both GPUs exhibited their poorest performance during the initial iteration compared to subsequent ones. This bottleneck stems from the hardware architecture, as data must be transferred to the GPU for computation and tensor initialization. The CPU, however, did not display this bottleneck.
(cf. Later experiments will be using NVIDIA GeForce RTX 3090)

## 2. Parameter Sharing
According to the ["Attention Is All You Need" paper](https://arxiv.org/pdf/1706.03762), the transformer decoder shares the same matrix between the word-to-token embedding and the final linear head. This approach was first introduced in this [paper](https://arxiv.org/pdf/1608.05859). The idea is to make these two matrices behave similarly. For the input embedding, if two words are semantically similar, we want their embeddings to be close to each other. Similarly, in the final linear head, if two words are semantically related, their output probabilities should be similar. By tying these two matrices together, I was able to reduce nearly 30% of the total parameters in the GPT-2 model. (The model has a total of 124 million parameters, with the embedding matrix alone accounting for 40 million of those parameters.)

## 3. Weight Initialization
To stabilize model training, the original GPT-2 model used small random weight initialization for linear and embedding layers. This initialization follows a normal distribution with a mean of 0.0 and a standard deviation of 0.02. The choice of 0.02 for the standard deviation is because it performs similarly to [Xavier initialization](https://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf). For instance, with GPT-2's embedding dimension of 768, $1/\sqrt(768) ≈ 0.03$ and $1/sqrt(3*768) ≈ 0.02$.

The [GPT-2 paper](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) goes one step further in weight initialization by addressing the residual path. Since there are multiple residual paths for training stabilization, the variance of the data accumulates. Therefore, the paper argues that it's necessary to scale the weights of residual layers at initialization by a factor of 1/√N, where N is the number of residual layers.

## 4. Tensor Float 32
By default, PyTorch calculates all tensors in FP32, a single-precision 32-bit floating-point number representation. This 32-bit format consumes a significant amount of memory. A [recent paper](https://arxiv.org/pdf/2403.02243) argues that many deep learning models can tolerate training in lower precision, such as Tensor Float 32 (TF32). TF32 is a new math mode introduced with NVIDIA's Ampere GPU architecture. According to the [NVIDIA Ampere GA102 GPU Architecture specification](https://www.nvidia.com/content/PDF/nvidia-ampere-ga-102-gpu-architecture-whitepaper-v2.1.pdf), TF32 precision can boost training throughput by up to 5 times. This improvement stems from TF32's ability to create tensors specialized for hardware architecture, accelerating convolution and matrix multiplication computations. This optimization can be achieved by adding a single line of PyTorch code.

(Note: While some older GPUs may not support TF32, the RTX 3090 is based on the NVIDIA Ampere GA102 GPU architecture, which does support TF32 precision.)

`torch.set_float32_matmul_precision('high')`

The results demonstrated that using TF32 accelerates the model's throughput by approximately 1.5 times, which is relatively modest. This acceleration likely stems from faster matrix multiplication. However, as all tensors still move in FP32 format, there remains a memory bandwidth inefficiency mentioned in this [paper](https://arxiv.org/pdf/2311.12816).


## 5. Brain Float 16
In the previous section 3-4, I mentioned that PyTorch uses FP32 as the default format for all tensors. Even when using TF32, tensors still move in memory with the FP32 format. To truly reduce the format, we need to use lower-bit formats like FP16 or BF16 (See Figure 3). FP16 has a 5-bit exponent (range) and a 10-bit mantissa (precision). Since FP16 has reduced exponent bits compared to FP32, it may not fully represent FP32's range, potentially causing scaling problems. Therefore, when casting to FP16, it's necessary to use a gradient scaler to fully emulate the FP32 training range. On the other hand, BF16 maintains an 8-bit exponent part, while significantly reducing the precision part to 7 bits. It remains nearly as representable as FP32 since the exponent part has the same number of bits. This format transformation can be achieved by adding a single line of PyTorch context manager.

(Note: `autocast` should be applied only during the forward pass, including loss computation, as the backward pass is more sensitive to precision changes)

`with torch.autocast(device_type=device, dtype=torch.bfloat16):`

The results showed that autocasting to BF16 made my model train 1.5x faster. This acceleration makes sense, as not all tensors change format—which is why PyTorch refers to this as "mixed precision." You can see [here](https://pytorch.org/docs/stable/amp.html#autocast-op-reference) which tensor operations change their format.

## 6. Model Compile
Just by adding a single line of code, I could make my model run 30% faster than before. Think of torch.compile as a GCC compiler in C language. Unlike the Python interpreter that reads and interprets code line by line—and thus doesn't know what operations are being calculated in the future—torch.compile sees the entire model code and knows which operations are coming next, allowing for process optimization. According to the [official Pytorch documentation](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html), this optimization speedup mainly comes from reducing Python overhead and minimizing GPU read/writes between GPU chips and HBM by kernel fusion.

`torch.compile(model)`

## 7. FlashAttention
Although model compilation provides significant speedup, there are still areas where PyTorch's torch.compile cannot apply kernel fusion. One such case is the attention operation, which involves multiple matrix multiplications, dropout, and softmax operations. According to the [FlashAttention paper](https://arxiv.org/pdf/2205.14135), by algorithmically reorganizing the code for this attention mechanism, we can utilize kernel fusion to achieve a substantial 7.6x speedup—even though the algorithm actually performs more FLOPs. Our implementation of this approach resulted in a 10% speed improvement.

## 8. Beautiful Numbers
In terms of CUDA and neural networks, numbers can be categorized as "ugly" or "beautiful." Ugly numbers are typically prime or odd, like 13 or 15, while beautiful numbers are even or powers of 2, such as 128 or 256. We call these numbers beautiful because CUDA operations work efficiently with powers of 2, and many kernels are optimized for such numbers.

In my model, I discovered that the vocab_size was an ugly odd number: 50257. I changed this to 50304, which is divisible by 8, 16, 32, 64, and 128. Despite increasing the total computations, this change resulted in a 6% speedup.

## 9. Gradient Clipping
According to the [GPT-3 paper](https://arxiv.org/pdf/2005.14165), they clipped the global norm of the gradient at 1.0. This means that after calculating all the gradients, we combine them into a single vector and calculate its L2 norm. If this L2 norm exceeds 1.0, all gradients are scaled down proportionally so that the final L2 norm becomes 1.0. This process helps stabilize model training by preventing excessively large gradients. Occasionally, unlucky batches can trigger a significant loss, leading to huge gradients that may destabilize the model. By using gradient clipping, we can train our model more stably. This can be implemented with a single line of code, as shown below.

`norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)`

Printing out this norm value for each iteration can be a good metric to show how stable our models are. Results showed that for the first few iterations, norms were very high as the models were randomly initialized. However, after the first few steps, they began learning correctly, causing the norm values to decrease.

## 10. Optimizer with Parameter Grouping & Fused Implementation
Weight decay forces all weights towards a uniform distribution, but it's uncommon to apply this to one-dimensional tensors like biases and layer normalization parameters. Therefore, I separated the parameters into two groups (as Pytorch allows its implementation): those with dimensions greater than two, and all others. I then configured the AdamW optimizer to apply weight decay only to the group with weight dimensions greater than two.  

In addition to parameter grouping, I set `fused=True` in the optimizer. The default `fused=False` optimizer updates all model parameters using a for-loop, launching numerous CUDA kernels. In contrast, a fused implementation combines parameters into a single group and updates them together, launching only one CUDA kernel. This approach can significantly improve speed. However, according to the PyTorch documentation, the effectiveness of fused implementation varies depending on the optimizer. In my case, it didn't work well and even slowed down the update process.

## 11. Gradient Accumulation
Up to section 3-10, we used a batch size (number of tokens) of 8192 ($B×T=8×1024=8192$) for training our language model. However, according to the GPT-3 paper(), GPT-3 Small used a much larger batch size of 0.5M. It's crucial to match certain hyper-parameters (e.g. batch size, learning rate, weight decay, etc) with the paper for faithful results, as these are mathematically correlated. Simply setting $B=500$ to approach 0.5M wouldn't work due to the memory limitations of our single RTX 3090 GPU (24GB). To bridge this gap, I implemented gradient accumulation, which allows us to effectively use a larger batch size without exceeding memory limits. This technique accumulates gradients until the total batch size nears 0.5M before updating.

For our experiments, I set a total batch size of 524,288 to approach 0.5M while maintaining a "beautiful number" ($2^{19}=524288$) as mentioned in the section 3-8. With this much larger total batch size compared to previous sections, the model learned more diverse features (indicated by smaller loss) and exhibited more stable learning (with the gradient norm decreasing to less than one). Moreover, I compared two variants with the same total batch size but different micro batch sizes. As expected, the outputs showed almost same results (see the losses for each step).

