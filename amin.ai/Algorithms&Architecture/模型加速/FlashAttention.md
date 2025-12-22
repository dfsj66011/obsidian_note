
### 核心总结

标准的自注意力机制在序列长度 $N$ 增加时表现不佳，因为它需要计算完整的 $N×N$ 注意力矩阵，并进行 $O(N^2·d)$ 量级的运算和内存存储。尤其在长上下文模型中，计算量和内存消耗会急剧膨胀。现有的近似方法（如 Linformer、Performer）往往需要牺牲准确性，或由于 GPU 效率问题而无法在实际应用中实现运行时间的改进。

> [!important]
> FlashAttention 的核心洞见在于 IO 感知：它认识到 GPU 上的主要性能瓶颈并非浮点运算（FLOPs），而是高带宽内存（HBM）与片上缓存（SRAM/寄存器）之间的数据传输。
> 
> FlashAttention 不仅优化计算过程，还重构了内存访问模式。它将注意力计算分块处理，使每个块能完全容纳在 SRAM 中，并按顺序处理这些块，从而大幅减少昂贵的片外内存流量。关键的是，该技术会重新计算某些中间值（如 softmax 归一化常数）而非存储它们，这比从 HBM 读取数据成本更低。

下图 ([source](https://huggingface.co/docs/text-generation-inference/en/conceptual/flash_attention)) 对比了标准注意力机制与 FlashAttention，展示了 FlashAttention 如何通过将操作融合为更少的内存事务来减少内存读写次数。

![|600](https://aman.ai/primers/ai/assets/flashattention/flash-attn.jpg)



**FlashAttention‑1:**

* 每个注意力头/层使用单个融合的CUDA内核，将QKT计算、掩码处理、softmax归一化、dropout随机失活及输出乘法运算整合为一体，以最大限度减少高带宽内存（HBM）与静态随机存储器（SRAM）之间的数据传输。
* 将Q、K、V分块成 SRAM 大小的块；为数值稳定性重新计算每块的 softmax 归一化。
* 并行性主要体现在批次和头数上；序列长度并发的使用有限。
* I/O 最优设计——在实际 SRAM 容量下，经证明需要 O(N⋅d) 内存流量的下限。

**FlashAttention‑2**：

* 通过在多个线程块上分割头部计算，实现了序列长度维度的并行处理。
* 通过延迟 softmax 缩放操作来减少非 GEMM 浮点运算量——从而消除跨块冗余归一化。
* 使用 CUTLASS 和 CuTe 实现，旨在提高占用率和线程块协调性。
* 增强型 warp 组分区以减少共享内存同步开销。

**FlashAttention‑3**:

* 专为 NVIDIA Hopper（H100）硬件设计，利用 warp专业化和异步调度技术：部分 warp 执行WGMMA GEMM 运算，其他 warp 执行 softmax/scaling 操作，实现计算重叠。
* 以乒乓方式在每个块内跨线程束组进行流水线 GEMM 和 softmax 操作，以最大化利用张量核心和张量内存加速器（TMA）。
* 引入分块式 FP8 量化技术，采用非连贯处理与动态异常值处理机制，以最大限度减少数值误差。
* 利用 Hopper 的 WGMMA 和 TMA 指令，在低精度和标准 FP16/BF16 模式下均能保持高吞吐量。

### 性能比较

|**Version**|**Target GPU**|**Forward Speedup**|**Peak Throughput**|**Backward Speedup**|**Numerical Accuracy (Low‑prec)**|
|---|---|---|---|---|---|
|FlashAttention‑1|Ampere / A100|~3× over PyTorch on GPT‑2 (seq=1K)|~30–50% utilization|~ similar to baseline|Full FP16/BF16 accuracy; exact attention|
|FlashAttention‑2|Ampere / A100|~2× over v1|~225TFLOPs/s (~72%)|~2–4× over naive backward|Same full precision accuracy|
|FlashAttention‑3|Hopper / H100|~1.5–2× over v2 (FP16)|~740TFLOPs/s (~75% BF16); ~1.2–1.3PFLOPs/s (FP8)|~1.5–1.75× over v2|FP8 RMSE ~2.6× lower than baseline FP8; full precision accuracy preserved.|

### 算法 & I/O 差异

所有版本均保持 I/O 最优行为：FlashAttention-1 实现了 O(N·d) 的数据移动量，并且对于典型的 SRAM 大小而言，可证明是渐进最优的。

FlashAttention-2 保持了相同的 I/O 特性，但通过减少归一化传递次数来降低额外的计算开销。

FlashAttention-3 在保持 I/O 效率的同时，引入了异步重叠和低精度格式，以减少 HBM 带宽的使用，并最大化片上计算能力。


## 准确性权衡、实际考量与集成指南

### 集成与 API 详情

FlashAttention 可通过 PyTorch C++/CUDA 扩展（flash_attn_interface）或 Triton 实现。典型用法是替换 PyTorch 或 DeepSpeed/Megatron 等框架中的标准 scaled_dot_product_attention 函数。

对于 FlashAttention-2 和 3，该库提供了优化的内核，并根据 GPU 架构和精度标志（如 FP16 与 FP8）自动调度。某些流水线框架（如 Triton 或 CUDA Graphs）可能需要手动配置以实现最佳的低延迟推理。

### 资源与内存使用

所有版本均保持线性内存使用，即 O(N⋅d)，而传统注意力机制为 O(N²)，这使得在保持全精度的情况下，上下文长度可达 64K 甚至更长。

GPU 共享内存/寄存器使用经过严格优化。在 FlashAttention-3 中，通过 Hopper 架构的 TMA 和WGMMA 技术实现了大尺寸分块和异步重叠计算，但寄存器压力可能增加，从而限制最大头维度或批处理规模。

### 何时使用哪个版本

如果你使用的是安培级 GPU，或者在进行 FP16/BF16 训练/推理，并希望获得一个稳健且经过充分测试的解决方案：FlashAttention-2 是安全且高性能的默认选择。

如果您需要与旧款 GPU 完全兼容或集成需求较为简单，FlashAttention-1 仍然能在不依赖特定硬件的情况下显著节省内存并提升速度。

如果你能使用 Hopper GPU 并追求最大吞吐量（尤其是 FP8），FlashAttention-3 是最佳选择——但需注意硬件和软件要求。量化精度表现优异，但早期版本对 FP8 的反向支持可能有限。

## 性能基准测试、代码集成示例和调优技巧

### 性能基准

#### FlashAttention‑2 (Ampere / A100)

在 NVIDIA A100 GPU上，FlashAttention-2 的前向传播吞吐量最高可达 **230TFLOPs/s**，约为理论FP16/BF16峰值性能的50-73%。后向传播性能最高可达峰值的63%，较第一代版本显著提升。针对GPT类模型的端到端训练吞吐量，每块A100 GPU可达约225TFLOPs/s，模型FLOPs利用率达到约72%。相比FlashAttention-1提速约2倍，在基准测试中比原生PyTorch注意力机制最高可提速3-9倍。

#### FlashAttention‑3 (Hopper / H100)

在 NVIDIA H100 GPU 上，FP16/BF16 模式可达到约740TFLOPs/s（约75%利用率），FP8模式接近约1.2PFLOPs/s，相比FlashAttention-2实现了1.5至2倍的加速。FP8运算的数值误差（RMSE）也比基准FP8注意力实现降低了约2.6倍。

#### 比较摘要

对于安培/A100 架构，FlashAttention-2 相比第一代版本实现了约 2 倍的性能提升。在霍珀/H100平台上，FlashAttention-3 将 FP16 计算吞吐量提高了 1.5 至 2 倍，FP8 性能达到 1.2PFLOPs/秒，同时保持高精度。各代演进中，当使用 FP8 精度时，注意力机制性能从第一代的约 50 TFLOPs/秒跃升至第三代的超过 1P FLOP/秒

### Integration & Code Examples

#### 安装 FlashAttention (v1 & V2)

```shell
pip install flash-attn
```

这提供了 FlashAttention-1 和 -2 在官方 `flash-attn` PyPI 包（v2.x系列）中的实现。

#### PyTorch 使用模式

```python
import torch 
from flash_attn.flash_attn_interface import flash_attn_forward   

# Inputs: Q， K， V as [batch， seq， heads， head_dim] FP16/BF16 
output = flash_attn_forward(Q， K， V， causal=True， dropout_p=0.0)
```

这取代了典型的 `F.scaled_dot_product_attention`，通常集成到 DeepSpeed、Megatron-LM 或自定义的 PyTorch 模块中。

#### FlashAttention‑3 / FP8 Usage

从 v3 测试版开始，FlashAttention-3 支持 Hopper GPU 上的 FP16/BF16 前向/反向传播以及 FP8 前向传播。如果在 H100 上运行，内核选择将自动调度。

使用FP8（e4m3、e5m2）时，请确保 CUDA版本≥12.3，并配备适当的硬件以获得最佳性能。

### Tuning Tips & Best Practices

根据 GPU 选择版本：

- 在安培架构（如A100等）上使用 FlashAttention-2，以在 FP16/BF16 下获得稳定的高性能。
- 在霍珀架构/H100 上使用 FlashAttention-3，以实现 FP8 支持的最大吞吐量。

序列长度与头维度调优：块大小是根据头维度和共享内存容量设计的。头维度过小或过大可能会因寄存器/共享空间限制而降低效率——尤其是在 FlashAttention-3 中。

批量大小考量：为了获得最佳的每 GPU 吞吐量，需确保每个 GPU 上有足够的 token 级并行性——例如，批处理多个长度 ≥512 的序列可保证较高的线程占用率。

因果掩码：FlashAttention-2 和 -3 均支持因果掩码。无论是否使用掩码，性能表现均保持较高水平，仅存在微小的开销差异。

混合精度策略：对于支持 FP8 的推理场景，使用 FlashAttention-3 的 FP8 模式可在保持接近 FP16 精度的同时实现最大吞吐量。若 FP8 反向传播尚未稳定，则训练时采用 BF16 格式。

库集成：FlashAttention 自动检测 GPU 架构并调度相应的内核。对于 Triton、CUDA Graphs 或 DeepSpeed 等框架，如需启用 FP8 流水线，请手动确保其开启并通过测试。

## 自定义头部尺寸与长序列优化的代码走查

### 支持更大的头部尺寸

FlashAttention-2 扩展了对更大头维度（最高可达 256）的支持，使其能够兼容诸如 GPT-J、CodeGen 和 Stable Diffusion 1.x 等模型。在实际应用中，现已支持并优化了超过 128 的头维度使用。在 PyTorch 中定义自定义 Transformer 层时：

```python
import torch 
from flash_attn.flash_attn_interface import flash_attn_forward  

head_dim = 192  # any value up to 256 

Q = torch.randn(batch_size， seq_len， num_heads， head_dim， device=device，dtype=torch.float16) 
...  
output = flash_attn_forward(Q， K， V， causal=True)
```

FlashAttention 内核会根据 `head_dim` 自动调整内部的分块策略。对于 v2 版本，更大的维度意味着更大的分块尺寸，从而更好地利用共享内存和寄存器，同时还能在 Ampere/A100 GPU 上保持较高的占用率。

### 长序列处理（例如 64K 上下文）

FlashAttention 的设计目标是使内存消耗随序列长度线性增长，即 O(N·d)，从而即使在处理 64K 个标记时也能保持高效运行。对于长序列推理或训练：

```python
seq_len = 65536 

Q = torch.randn(1， seq_len， num_heads， head_dim， device=device， dtype=torch.bfloat16)
 ...  
output = flash_attn_forward(Q， K， V， causal=True)
```

该内核流式处理适合片上 SRAM 的 Q、K 和 V 分块数据，采用最大-求和流式方法逐块重计算归一化，避免生成完整的 N×N 矩阵。该设计确保内存占用恒定，不受每令牌/负载需求影响——即使处理数万个 tokens 时依然如此。FlashAttention-3 通过异步流水线和 FP8 支持，进一步提升了 Hopper GPU 在处理长序列时的吞吐量。

### 长上下文性能优化技巧

较大的分块尺寸可以提高吞吐量，但需注意：它们会增加共享内存和寄存器的压力。对于16K至64K token的序列，默认分块尺寸经过测试，以平衡占用率和寄存器使用。用户验证日志显示，FlashAttention-3在FP16/BF16模式下，处理长序列时在H100上的实际吞吐量接近740 TFLOPs/s；FP8模式在控制RMSE的情况下（比基准FP8低约2.6倍）可实现约1.2 PFLOPs/s。在调整头维度时，请保持在支持范围内（≤256），以确保内核调度优化；对于消费级GPU，头维度为256时，对dropout和FP8的反向支持可能受限。

## 最佳实践/建议

- 在安培架构 GPU（如 A100）上使用 **FlashAttention-2**，以获得 FP16/BF16 训练或推理的稳定且经过充分测试的性能。
- 如果使用 Hopper 架构 GPU 且支持 FP8（CUDA ≥ 12.3），则选择 **FlashAttention-3**​ 以实现最大吞吐量。它在保持良好数值精度的同时，可显著提升速度（约 1.2 PFLOPs/s）。
- 若需兼容旧款 GPU 或追求简单性，请坚持使用 **FlashAttention-1**。它无需依赖先进硬件，仍能大幅提升速度并节省内存。
- 对于长上下文场景（例如处理 64K token 序列的推理），**FlashAttention**​ 系列方法可实现内存线性扩展，并支持基于分块的流式处理以保持高性能。FP8 模式（v3/Hopper 架构）可进一步降低内存带宽占用。
- **FlashAttention-2**​ 和 **FlashAttention-3**​ 支持并优化了 128 至 256 的头维度，但可能增加寄存器/共享内存压力。建议针对具体变体进行基准测试。


## 深入探讨Softmax流式处理与I/O复杂度分析

### 通过流式平铺实现的 Softmax 归一化

FlashAttention-1 采用在线 softmax 归一化技术，逐块处理查询 Qi 和键值对块 Kj、Vj。对于每个查询块，算法通过迭代计算 mi=maxjsij、ℓi=∑jexp(sij−mi) 并逐步更新输出来转换O=softmax(QK⊤d√)V：以流式处理方式整合每个分块的归一化贡献。该方案避免了存储完整的N×N对数矩阵，将内存开销降低至O(N·d)，从而支持高达64K标记的上下文长度。同时，在典型SRAM容量下，该方案达到了内存流量的理论下限。

### I/O 复杂度：最优流量缩减

标准注意力机制需要对高带宽存储器（HBM，即高延迟全局内存）进行O(N²)次读写操作，包括完整的日志记录和softmax中间结果存储。FlashAttention的在线处理方式通过将数据块流式传输至SRAM（静态随机存取存储器），并动态重新计算softmax归一化因子而非缓冲大型中间矩阵，确保总数据传输量仅为O(N·d)。FlashAttention-2和-3在保持这种I/O最优模型的同时，进一步提升了计算吞吐量，且未影响渐进内存使用量。

### FlashAttention-2：块并行硬件平铺（图1）

FlashAttention-2 采用块并行方式排列查询和键值块：每个线程块本地加载一个查询向量块，同时以块为单位流式传输键/值，以更新每个查询的 softmax 状态和输出贡献。这种并行硬件结构消除了块间的顺序依赖，并支持查询级别的并发性。

### FlashAttention-3：乒乓调度与 GEMM 和 Softmax 的重叠计算

FlashAttention-3 引入了跨两个或多个 warp 组的乒乓调度机制：当一个 warp 组执行点积运算的GEMM（使用 TensorCore）时，另一个组则利用多功能单元执行 softmax /指数运算。通过同步屏障（如 `bar.sync`）协调各迭代间的这种重叠操作，从而最大化两个计算单元的有效利用率。

此外，在单个 warpgroup 内部，warpgroup 内部流水线技术允许 softmax 的部分计算与 GEMM 并行执行，从而进一步提升吞吐量。这种两级流水线技术将 FP16 前向计算性能从约 570 TFLOPs/s 提升至约 640 TFLOPs/s（例如：序列长度=8K，头维度=128）。

### 理论与实践的影响

* 通过将算法分块与 GPU 内存架构对齐，FlashAttention（所有版本）实现了**I/O最优行为**，随着序列长度的增加，显著降低了延迟和内存带宽需求。
* FlashAttention-2的分块并行处理消除了顺序依赖，提高了在头和批次维度上的延迟和占用率。
* FlashAttention-3的warp专业化和异步重叠进一步最小化了空闲计算阶段，并将缓慢的非矩阵乘法softmax操作合并到GEMM活跃的周期中。

## 与现代框架和基准测试脚本的集成

### 框架支持与安装

`flash_attn` 库（v2及以上版本）通过 PyTorch C++/CUDA 扩展（`flash_attn_interface`）实现无缝集成，兼容 PyTorch 2.2+及安培、Ada 和 Hopper架构 GPU。FlashAttention-3 需配备H100/H800 GPU 和 CUDA≥12.3（推荐12.8）以获得完整的 FP8 支持。

### 高级集成

#### PyTorch (DeepSpeed， Megatron-LM， Hugging Face)

典型替换代码：

```python
from flash_attn.flash_attn_interface import flash_attn_forward 
# Q， K， V shaped [batch， seq_len， heads， head_dim]， dtype FP16/BF16 
O = flash_attn_forward(Q， K， V， causal=True， dropout_p=0.0)
```

根据可用 GPU 和精度自动分配合适的内核版本（v2或v3）。DeepSpeed 和 Megatron-LM 通常将 FlashAttention 作为标准缩放点积注意力的即插即用替代方案集成。

#### Triton & XFormers Backends

Trident 实现（例如在 Triton 语言中）提供了替代的融合内核，其前向传播速度可能比 `FlashAttention-2` 慢 1.3 至 1.5 倍。xFormers 和 Triton 在 API 兼容性方面存在差异；为实现最高速度，建议使用官方 FlashAttention-2 实现。

### 基准脚本示例

通常用户会使用类似以下的脚本进行基准测试：

```shell
python bench_attention.py \   
    --seq-len 4096 \   
    --batch-size 8 \   
    --num-heads 16 \   
    --head-dim 128 \   
    --use-causal \   
    --dtype fp16
```

特殊标志如 `--use-flash-attn` 或环境变量 `USE_FLASH_ATTN=1` 会激活 FlashAttention 内核，而非默认的 PyTorch 注意力机制。

* FlashAttention-2 的基准测试结果显示：​ ​
	* 在 A100 GPU 上，端到端 GPT 训练吞吐量高达 225 TFLOPs/s（约 72% 的浮点运算利用率）。​ ​
	* 前向+反向组合性能比 FlashAttention-1 快 1.7-3.0 倍，根据配置不同比 PyTorch 基线快达 9 倍，在头维度 64 和 128 上均保持一致的加速效果。​​￼
* 关于 FlashAttention-3 的基准测试：​ ​ 
	* FP16/BF16 模式：在 H100 上约 740 TFLOPs/s（约 75% 利用率）。
	* FP8模式：接近 1.2-1.3 PFLOPs/s，量化误差比基线 FP8 低约 2.6 倍。

### 自动调度与精准处理

- 在运行时，FlashAttention 会检查硬件 ID 以决定使用 v2（Ampere 架构）还是 v3（Hopper 架构）的内核。
- 如需支持 FP8（FlashAttention-3），用户可能需要启用实验性 API 标志（例如 `precision='fp8'`）或依赖夜间构建版本。
- 如果 FP8 的反向传播支持尚不完善，工作流程可以回退到使用 BF16 或 FP16 进行梯度计算。

### 部署与推理准备

FlashAttention 同样适用于 HuggingFace 等框架的推理场景——其内核调度机制能高效处理因果掩码。NVIDIA 的 FlashInfer 基于 FlashAttention-3 技术，优化了 KV 缓存感知推理，与标准后端相比，将 tokens 间延迟降低了 29% 至 69%。

## References

- [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135)
- [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691)
- [FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision](https://arxiv.org/abs/2407.08608)
- [The I/O Complexity of Attention， or How Optimal is Flash Attention?](https://arxiv.org/abs/2402.07443)
- [FlashInfer: Low-Latency Generative Inference with FlashAttention](https://arxiv.org/abs/2410.16663)
- [Online Pseudo-average Shifting Attention (PASA) for Robust Low-precision LLM Inference](https://arxiv.org/abs/2312.11918)
- [FlashAttention-3 blog post](https://tridao.me/blog/2024/flash3/)
- [FlashAttention-2 paper site](https://tridao.me/publications/flash2/flash2.pdf)
- [FlashAttention GitHub repository](https://github.com/Dao-AILab/flash-attention)
- [FlashAttention PyPI package](https://pypi.org/project/flash-attn/)
- [The Evolution of FlashAttention: Revolutionizing Transformer Efficiency (Medium article)](https://medium.com/@sailakkshmiallada/the-evolution-of-flash-attention-revolutionizing-transformer-efficiency-8a039918d507)

# Umar 讲解

我们需要为 GPU 编写内核程序，具体来说，是针对我们的需求定制一个内核，使用 Triton，它能够将 Python 代码直接转换为可在 GPU 上运行的 CUDA 内核程序，可以把 Triton 看作是一个编译器，它接收 Python 代码并将其转换为能在 GPU 上运行的程序。

本篇文章要讨论的主题包括：

* 多头注意力机制（MHA）
* safe softmax
* online softmax，
* 然后，我们将深入了解 GPU，因为我们要编写一个在 GPU 上运行的内核程序，因此，我们需要理解 CPU 和 GPU 之间的区别，什么事内核程序，以及它与 CPU 编写的普通程序有何不同，
* 我们将研究张量在内存中的布局方式，比如行优先布局、列优先布局，步幅等，
* 我们将探讨分块矩阵乘法，
* Triton 的软件流水线，以及 Triton 对我们代码所做的所有优化
* 最后，我们将能够编写 Flash Attention 的前向传播代码，当然，仅仅编写前向传播代码并不能让我们满足，
* 我们还希望编写反向传播代码，但要编写反向传播代码，我们还需要理解在自定义操作的情况下，自动微分 autograd 和梯度下降是如何工作的，
* 因此，我们需要理解什么是导数、梯度和雅可比矩阵，
* 然后计算我们在 FlashAttention 中使用的常见操作的梯度，
* 最终，我们将掌握足够的知识来编写反向传播代码，


### 1、多头注意力机制

快速回顾一下 MHA 是什么以及它是如何工作的，公式如下：$$\begin{align*}
\text{Attention}(Q， K， V) &= \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V \\
\text{MultiHead}(Q， K， V) &= \text{Concat}(\text{head}_1， \ldots， \text{head}_h)W^O \\
\text{head}_i &= \text{Attention}(QW_i^Q， KW_i^K， VW_i^V)
\end{align*}$$
实际上，Flash Attention 主要关注的是 Attention 部分的优化，$Q，K，V$ 的线性层投影以及输出层投影都是常规的矩阵相乘，这一点在 GPU 中已经高度优化。

### 2、缩放点积注意力机制的局限性

#### 2.1 问题一、GPU I/O 限制

一个非常关键的问题是：我们为何需要改进注意力机制的实现方式，如果你查阅 [Flash Attention 1 论文](https://arxiv.org/pdf/2205.14135)，会注意到章节 2.2：

![[Pasted image 20250318151046.png|650]]

GPU 主要由两种内存构成，一种是 HBM，即动态随机存取存储器（DRAM），也就是 GPU 的内存，例如 A100 的 40GB 内存，这是 GPU 中容量最大的内存；此外还存在共享内存。

GPU 面临的问题是，访问 HBM（全局内存）与访问共享内存相比，速度极其缓慢，然而，与 HBM 相比，共享内存的容量要小得多，FlashAttention 论文中指出，*注意力机制的操作是 I/O 受限的*，这意味着，如果我们频繁访问全局内存，那么计算注意力机制的整体操作速度慢，这并不是因为计算这些操作本身慢，而是因为频繁访问速度较慢的全局内存导致的，因此我们可以将这类操作称为 I/O 受限型操作。

因此，改善这一状况的唯一方法是，在 GPU 的共享内存中计算注意力机制，尽管共享内存的容量要小得多，但共享内存更靠近实际执行计算的 kernel，因此，我们需要将注意力计算拆分为更小的块，以便这些块能够放入共享内存中，然后在那里计算输出矩阵的一部分，再将这部分复制到位于 HBM 中的输出矩阵中，并针对查询、键、和值矩阵划分的所有块，重复这一过程。

在论文中，他们称之为“分块（tiling）”，这是一种在编写 GPU 内核时常用的技术，尤其是在涉及矩阵乘法的情况下。现在我们了解了 FlashAttention 试图解决的核心问题。

#### 2.2 问题二、softmax

这种分块计算的最大难题在于 softmax，因为 softmax 需要访问整个 $S$ 矩阵的所有元素才能完成计算，因为需要计算归一化因子，这个因子是对所有元素逐行计算指数后的总和。

### 3、（Safe）Softmax

#### 3.1 softmax 计算上的问题
$$
\mathbf{S} = \mathbf{QK}^\top \in \mathbb{R}^{N \times N}， \quad \mathbf{P} = \text{softmax}(\mathbf{S}) \in \mathbb{R}^{N \times N}， \quad \mathbf{O} = \mathbf{PV} \in \mathbb{R}^{N \times d}，$$
这里的 $QKV$ 都是经过相应线性层转换后的矩阵，$Q，K$ 的维度均为 $N \times d$（$N$ 是序列长度，$d$ 是每个 head 中 token 的嵌入维度，已完成多头切分），点积运算后，其输出矩阵 $S$ 的维度为 $N \times N$，softmax 操作按行处理，并不改变矩阵维度，其结果最后与 $V$ 相乘，输出维度为 $N \times d$。

softmax 操作的作用是什么呢？它会将这些点积结果进行转换，使得它们以某种方式变为一种概率分布，*按行计算*，这意味着每个数字都介于 0 到 1 之间，softmax 的定义如下：$$\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_{j=1}^{N} e^{x_j}}$$分母是向量所有维度指数的总和，这被称为归一化因子，为的是使所有这些数字介于 0 和 1 之间，使用 softmax 是因为我们希望这些数字都是正数（概率值），这是使用指数函数的原因，

但这里*存在一个问题*，问题在于，想象一下我们的输入向量由许多可能很大的数字组成，比如 100 的指数，会造成计算机结果上溢，即数值不稳定性，在计算机科学中，“数值不稳定性”意味着数字无法用我们现有的位数（通常是 32 位或 16 位）在固定表示形式中表示出来。

#### 3.2 解决方案

为了使这个 softmax 操作在数值上保持稳定，我们希望这些数字不会爆炸或变得太小以至于无法表示，我们需要找到一种解决方案，如下：$$\begin{align}\frac{e^{x_i}}{\sum_{j=1}^{N} e^{x_j}} 
&= \frac{c \cdot e^{x_i}}{c \cdot \sum_{j=1}^{N} e^{x_j}} = \frac{c e^{x_i}}{\sum_{j=1}^{N} c e^{x_j}} = \frac{e^{\log(c)} e^{x_i}}{\sum_{j=1}^{N} e^{\log(c)} e^{x_j}} \\[1.2ex]
&= \frac{e^{x_i + \log(c)}}{\sum_{j=1}^{N} e^{x_j + \log(c)}} = \frac{e^{x_i - k}}{\sum_{j=1}^{N} e^{x_j - k}} \quad \text{where } k = -\log(c)\end{align}$$
用因子 $c$ 乘以分子和分母，通过上面推导过程，我们可以看到，如果我们巧妙地选择一个值插入到这个指数函数中，就能有效减少指数部分的计算量，我们将选择这个 $k$ 值等于输入向量中需要应用 softmax 的最大元素（$k=\max_i(x_i)$），这样一来，每个指数的参数要么为 0（当 $x_i$ 等于向量中的最大值时），要么小于 0，其指数结果介于 0 到 1 之间，这样的数值用 32 位浮点数就能轻松表示。

#### 3.3 safe softmax 算法

$$\text{softmax}(x_i) = \frac{e^{x_i - x_{\text{max}}}}{\sum_{j=1}^{N} e^{x_j - x_{\text{max}}}}$$
给定一个 $N \times N$ 的矩阵，对于每一行：

1. 寻找每一行的最大值，时间复杂度：$O(n)$，内存占用：$O(n)$
2. 计算分母归一化因子，时间复杂度：$O(n)$，内存占用：$O(n)$
3. 对向量中的每一个元素应用 softmax，时间复杂度：$O(n)$，内存占用：$O(n)$

伪代码如下：
```python
m_0 = -infty
for i=1 to N
    m_i = max(m_{i-1}， x_i)

l_0 = 0
for J=1 to N
    l_J = l_{J-1} + e^{x_J - m_N}

for K=1 to N
    x_K <- e^{x_K-m_N} / l_N
```

这段伪代码描述的算法相当慢，显而易见，这里存在 3 个 for 循环。所以接下来优化的思路就是寻找一种策略，合并其中的某些操作，减少循环次数。

### 4、Online Softmax

#### 4.1 online softmax

我们尝试将前两个操作融合到一个 for 循环中，这意味着我们只需要遍历数组一次，同时计算 $m_i$，并尝试计算 $l_j$，当然，我们无法在此刻计算 $l_j$，因为无法得知全局最大值，但我们可以尝试使用当前已知的局部最大值作为估算值来进行计算，即我们尝试用 $m_i$ 替代 $m_n$。

当后续迭代过程中发现更大值时，需要对过去计算项进行修正，实际上这个校正因子非常容易计算，以 $x=[3，2，5，1]$ 为例，在前两轮中最大值为 3，第三次迭代时，最大值为 5，即在第三轮迭代中，

* 错误迭代计算：$l_3 = l_2 + e^{5-5}=e^{3-3}+e^{2-3}+e^{5-5}$
* 正确修正方法：$l_3 = l^2 \cdot \textcolor{blue}{e^{3-5}} + e^{5-5}=(e^{3-3}+e^{2-3})\textcolor{blue}{e^{3-5}}+e^{5-5}$

显然这个修正因子的计算方法为过去的最大值与当前新的最大值之间的差。

因此，softmax 新算法如下：

```python
m_0 = -infty
l_0 = 0
for i=1 to N
    m_i = max(m_{i-1}， x_i)
    l_i = l_{i-1}*e^{m_{i-1} - m_i} + e^{x_i - m_i}

for K=1 to N
    x_K <- e^{x_K-m_N} / l_N
```

#### 4.2 数学归纳法证明

1. 证明对于大小为 $N=1$ 的向量，该命题成立：$$\begin{align}
m_1 &= \max(-\infty， x_1) = x_1 = \max_i(x_i) = x_{\max} \\[1.2ex]
l_1 &= 0 \times e^{-\infty} + e^{x_1 - x_1} = \sum_{j=1}^{N} e^{x_j - x_{\max}}\end{align}$$
2. 如果假设该命题对大小为 $N$ 的向量成立，证明它对大小为 $N+1$ 的向量也成立$$\begin{align}
m_{N+1} &= \max(m_N， x_{N+1}) = \max_i(x_i) \\[1.2ex]
l_{N+1} &= l_N \cdot e^{m_N - m_{N+1}} + e^{x_{N+1} - m_{N+1}} \\
&= \left(\sum_{j=1}^{N} e^{x_j - m_N}\right)e^{m_N-m_{N+1}} + e^{x_{N+1} - m_{N+1}} \\
&= \sum_{j=1}^{N} e^{x_j - m_{N+1}} + e^{x_{N+1} - m_{N+1}} \\
&= \sum_{j=1}^{N+1} e^{x_j - m_{N+1}} \end{align}$$
---------------------------------- 视频 47:28 ----------------------------------

### 5、分块矩阵乘法

![[Pasted image 20250319151749.png|500]]

#### 5.1 忽略 softmax

目前我们先暂时忽略 softmax 的部分，即：$$\mathbf{S} = \mathbf{QK}^\top \in \mathbb{R}^{N \times N}，  \quad \mathbf{O} = \mathbf{SV} \in \mathbb{R}^{N \times d}，$$当然，这种做法是不正确的，但它简化了我们接下来要处理的内容，

![[Pasted image 20250320101609.png|500]]

现在，每个 query 是由 Q 矩阵中的两行组成的一个组，key 也做相应的分块，在此基础上做分块矩阵乘法，如下所示：

![[Pasted image 20250320101933.png|400]]

以 $S$ 中左上角第一个分块为例，$Q_1$ 的维度为 $(2,128)$，$K^T$ 的维度是 $(128, 2)$，即 $S_{11}$ 实际上是一个 $(2, 2)$ 的小矩阵。接下来将 $S$ 矩阵与 $V$ 相乘，其结果也是显而易见的：
![[Pasted image 20250320102445.png|400]]
其运算结果为：

从宏观上看，S 矩阵 $(4,4)$，V 矩阵 $(4, 1)$，所以 O 矩阵大小 $(4,1)$，以 $O_{11}$ 为例：$$O_{11}=(Q_{1}K_{1}^T)V_{1}+(Q_{1}K_{2}^T)V_{2}+(Q_{1}K_{3}^T)V_{3}+(Q_{1}K_{4}^T)V_{4}$$这里的 $(Q_{1}K_{1}^T)=(2,128)\times(128,2)=(2,2)$，$V_{1}=(2,128)$，因此整体还是 $(2,128)$

伪代码如下：

```python
FOR EACH BLOCK Q_i
    O_i = zeroes(2， 128)                // Output is initially zeroes
    FOR EACH BLOCK K_j
        O_i ← O_i + (Q_i * K_j^T) * V_j
    END FOR
END FOR
```

#### 5.2 softmax$^\star$

softmax$^\star$ 是去除了归一化的 softmax，$$\text{SOFTMAX}^*\left(S_{ij}\right) = \exp\left[S_{ij} - \text{rowmax}\left(S_{ij}\right)\right]
$$
将 softmax$^\star$ 应用于 $S$ 矩阵的每个块上，可以得到：
![[Pasted image 20250320142554.png|500]]

但是需要注意的是，理论上，应用于每个小块内部元素上，我们需要知道该行的最大值，但目前暂时无法获知，举例而言，假设 $S_{11}= [a \quad b;  c \quad d]$，假设第一行的最大值是 $a$，第二行的最大值是 $d$，则 $P_{11}=[e^{a-a}\quad e^{b-a}; e^{c-d}\quad e^{d-d}]$，接下来，将 $P$ 矩阵与 $V$ 矩阵相乘，得到 $O$ 矩阵。 

*再次强调：这里计算每个 softmax$^\star$ 的最大值，并不是 $S$ 矩阵这一行的全局最大值，而是每个块的局部最大值，这实际上是错误的；*

#### 5.3 修正后的 softmax

如何修正这个问题？仍然用前面介绍的 Online Softmax，我们的目标是设计一个算法，既能修正用于计算每个分块下的最大值，又能同时计算归一化因子，具体实现方法如下所述，

**初始化：**

1. $m_0 = \begin{bmatrix} -\infty \\ -\infty \end{bmatrix}$ （我们的分块中有两行，每行都有一个最大值）
2. $l_0 = \begin{bmatrix} 0 \\ 0 \end{bmatrix}$ （用于累积分母部分，归一化因子）
3. $O_0 = \begin{bmatrix} 0 & 0 & \cdots & 0 \\ 0 & 0 & \cdots & 0 \end{bmatrix}$ （2x128 矩阵）

**步骤 1：**

1. 计算 $S_1 = Q_1 K_1^T$，<font color="blue">可以将</font> $S_1$ <font color="blue">想象成一个</font> $(2 \times 2)$ <font color="blue">的矩阵</font> $[a，b; c， d]$
2. 计算 $m_1 = \max(\text{rowmax}(Q_1 K_1^T)， m_0)= \begin{bmatrix}a \\ d \end{bmatrix}$，<font color="blue">按行统计的最大值</font>
3. 计算 $l_1 = \text{rowsum}\left(\exp(S_1 - m_1)\right) + l_0 \cdot \exp(m_0 - m_1)$，<font color="blue">按行累积最大值修正后的归一化因子值</font>
4. 计算 $P_{1,1} = \exp(S_1 - m_1)=[e^{a-a}\quad e^{b-a}; e^{c-d}\quad e^{d-d}]$，<font color="blue">每个元素都减去对应行中的最大值</font>
5. 更新 $O_1 = \text{diag}\left(\exp(m_0 - m_1)\right) O_0 + P_{1,1} V_1$，<font color="blue">第二项就是常规</font> $PV$ <font color="blue">计算</font>，<font color="red">第一项是对历史数据做最大值更新，同时</font> $\text{diag}$ <font color="red">是对角矩阵的含义，形式化为</font> $[m，0; 0，n] * (2 \times 128)$，<font color="red">则</font> $m$ <font color="red">仅参与和</font> $O_0$ <font color="red">的第一行计算中，这样可以保证每行的最大值仅参与该行的值更新</font>，这一点在步骤 2 中更明显。

注意：在该过程中暂时没有对 “softmax” 值进行归一化，此外，应该注意到这里实际是应用了两次 online softmax，分别用于在块内寻找局部最大值，并进行迭代更新，以及在块间寻找行内全局最大值，再次基于块整体迭代更新。

**步骤 2：**

1. 计算 $S_2 = Q_1 K_2^T[x，p; q， y]$
2. 计算 $m_2 = \max(\text{rowmax}(Q_1 K_2^T)， m_1)= \begin{bmatrix}x \\ y \end{bmatrix}$
3. 计算 $l_2 = \text{rowsum}\left(\exp(S_2 - m_2)\right) + l_1 \cdot \exp(m_1 - m_2)$
4. 计算 $P_{1,2} = \exp(S_2 - m_2)=[e^{x-x}\quad e^{p-x}; e^{q-y}\quad e^{y-y}]$
5. 更新 $O_1 = \text{diag}\left(\exp(m_1 - m_2)\right) O_1 + P_{1,2} V_2$，这里的 $O_{1}$ 是在此前最大值 $a$ 和 $d$ 基础上计算的，现在需要更新到最大值 $x$ 和 $y$ 上。

继续进行该行下的步骤 3 和 步骤 4，直到最后一步，然后应用 “$l$” 归一化因子。

**步骤 5：**

1. 计算 $O_5 = \left[\text{diag}(l_4)\right]^{-1} O_4$，<font color="red">对角矩阵的逆，就是各个元素的倒数矩阵，相当于</font> $[1/m， 0; 0， 1/n]$<font color="red">，这样就实现了除以归一化因子的目的</font>

至此，对于 $Q_1$ 的注意力计算结束，可以看到，行内是逐块顺序执行的，而行间则是并行实现的，因此后续对 $Q_2$、$Q_3$ 等的计算可以和 $Q_1$ 并行实现
 
---------------------------------- 视频 01:44:06 ----------------------------------

### 6、Flash Attention 前向传播
![[Pasted image 20250321163124.png|600]]

这是 *Flash Attention 2 的前向传播过程*，关于 Flash Attention 1 和 Flash Attention 2 之间的区别，会在后面解释。

1. 对 $Q， K， V$ 进行分块，分块大小取决于参数 $B_r$，因此每个块的大小为 $B_c \times d$
2. 初始化 $O， L$，然后接下来准备计算 softmax
3. line 3: 对 $Q_i$ 有个外层循环；line 6: 对 $K_j$ 有个内循环，与前面伪代码一致
4. line 12：计算 $O_i$，这与 5.2 中的步骤 5 完全一致
5. line 13：计算 $L_i$，这实际上是归一化因子的 $\log$，$$ \log\left(\sum_{i} \exp(x_i)\right) = x_{\text{max}} + \log\left(\sum_{i} \exp(x_i - x_{\text{max}})\right) $$
6. line 17：返回注意力 $O$，以及归一化因子的 $\log$ 值 $L$，$L$ 值用于反向传播使用。交叉熵对 logits 的梯度为 $\frac{\partial L}{\partial z_i}=y_i - t_i$，而这里的 $y_{i}=\frac{e^{s_i}}{\sum_j e^{s_j}}$，而 $L = \log \sum_j e^{s_j}$，所以 $y_i = e^{s_i - L}$

### 7、GPU、CUDA 简介

GPU 是我们购买的硬件单元，而 CUDA 是由 Nvidia 开发的软件堆栈，GPU 的任务不是同时处理多种不同的事情，而是专注于一件事或少数几件事，但处理的是海量数据，因此，我们在 GPU 上执行的操作需要大量计算，正因为如此，GPU 的大部分物理面积都用于计算单元。

#### 7.1 向量加法示例

以向量加法为例：两个向量 A 和 B，各包含 8 个元素，

![[Pasted image 20250321174457.png|600]]

CUDA 的工作机制是，当我们要求它并行启动 n 个线程时，它会分配 n 个线程，并为每个线程分配一个唯一的标识符，在这个简单的示例中，我们可以这样理解，第一个线程会被分配索引 0，每个线程处理的数据项正好对应其线程索引号，

* line 14：通过代码，根据线程标识符，指定每个线程处理的数据项，
* line 15：if 语句，指定启动 8 个线程，为什么需要加 if ？在 CUDA 中，启动的线程数总是 32 的倍数，这是 CUDA 的一个基本单位（线程束，Wrap），它们共享一个控制单元，控制单元是 GPU 硬件的一部分，负责确定接下来执行哪条指令，这意味着，这组线程将始终执行相同的指令，也就是说它们会同时到达这里的 if。由于每个线程有自己的寄存器，执行时使用的数据可能各不相同，这种编程模型称为 SIMD（data），或 SIMT（thread）；因此一些通过 if 的线程执行加法，而未通过 if 的也“不得不进入”，因为它们共享同一控制单元，但需要解决这种控制流分叉问题。大致流程为：满足条件的正常执行，不满足条件的线程进入 for 循环，但啥也不干，所有的线程必须保持同步执行相同的指令，这个现象叫控制流分支分化，显然这种空闲状态会降低程序执行效率。因此应尽可能减少这种情况的发生。

> [!NOTE]
> 在 CUDA 中，控制流分支分化（branch divergence）是指同一个线程束（warp）中的不同线程执行不同控制流路径的情况。这会导致性能下降，因为 GPU 必须顺序执行每个分支路径，而不是并行执行。控制流分支分化的影响：
> 1. 线程束执行：当线程束中的线程遇到不同的条件分支（如 `if` 语句）时，GPU 会按顺序执行每个路径，直到所有线程完成。这意味着一些线程会处于空闲状态，等待其他线程完成。
> 2. 性能下降：分支分化会导致线程束内的线程不能完全并行执行，从而降低执行效率。

#### 7.2 向量分块示例

简单示例中 8 个元素，扩大到 1 M 个元素，一次分配 1 百万个线程来执行任务，CUDA 会拒绝这样的请求，因为它超出了限制，当计算核心不足时，该如何管理并行计算呢？将输入向量划分为若干元素块，例如 GPU 由 32 个计算核，我们可以将输入向量划分为大小为 32 的块；而如果数据块大小为 32，GPU 有 64 个核，则一次可处理两个数据块，因此需要为 GPU 提供一定的工作粒度，需要增大数据的粒度，以便 GPU 能自主决定同时调度处理多少个数据块，这正是 CUDA 中引入块（blocks）概念的原因。

![[Pasted image 20250326093800.png]]

- grid：定义网格的维度（即线程块的数量和布局）。
- ​block：定义线程块（Block）的维度（即每个线程块中的线程数量和布局）。
- ​参数列表：传递给内核函数的参数（例如数组指针、标量值等）。

我们希望块的数量等于 $N / \text{block\_size}$，向上取整的值，因为 N 可能不是块大小的整数倍，接下来的问题是: 我们该如何将这些任务分配给每一个线程呢? 见 7.1 章节的图：

```c
int i = blockIdx.x * blockDim.x + threadIdx.x;
```

公式是：$\text{块 ID} \times \text{块大小} + \text{线程 ID}$

如果 GPU 有足够的空闲核心， 它可以选择同时运行一个或两个块，这就是为什么我们希望按块处理工作，因为这使得 GPU 在有足够核心的情况下，能够自主决定如何并行化操作，而且我们并不需要为 n 个元素的向量准备 n 个核心，我们可以将其划分为更小的块， 让 GPU 来管理调度。

#### 7.3 矩阵加法示例

![[Pasted image 20250326111258.png]]

公式是：$\text{（行 id）} \times \text{（一共几列）} + \text{（列 id）}$

在 C 或 C++ 中分配数组时，采用的是将所有行依次排列的扁平化数组结构， 因此我们需要依据行索引和列索引来定位数组中的元素，而这就是我们用来确定元素位置的公式.

![[Pasted image 20250326132814.png]]

grid 和 block 中都是三维尺寸，这里暂时用不到 Z 维度。

### 8、张量布局（Tensor Layouts）

#### 8.1 扁平化存储

当把一个矩阵或向量传递给 CUDA 或 CUDA 内核时， CUDA 并不会像 Python 那样一次性提供整个矩阵， 让你可以通过索引访问每个元素，而是只会给你一个指针，这个指针指向的是那个特定矩阵或向量的起始元素，然后，我们需要自己计算出所有剩余元素的内存地址。

不管是在 CPU 的内存中， 还是在 GPU 中，它将按照以下方式存储，假设第一个元素的起始地址是 100，并且每个元素由一个 16 位的浮点数组成，这意味着每个元素将占用两个字节，因此，第二个元素的起始地址将是 102，第三个元素是 104 等，这正是在 C 语言中使用 malloc 分配向量或矩阵时所得到的结果，C 语言或内存分配器会分配足够的内存来存储所有元素，并会给你一个指向该内存起始地址的指针。

矩阵会扁平化存储，按行扁平化处理的称为行主序布局，列主序布局的方式我们这里不讨论。行主序布局意味着，矩阵在内存中的存储方式为，先存储第一行的元素， 紧接着是第二行的元素。

#### 8.2 步幅属性

```text
1  2  3
5  8  13           # shape: [2, 3]   stride: [3, 1]

在内存中的实际存储样子：   1   2   3   5   8   13
           address:    62  64  66  68  70  72
```

步幅属性告诉我们，在每个维度中需要跳过多少个元素才能到达该维度的下一个索引位置，以上图为例，从一行跳到相邻另一行，需要跳过 3 个元素，从一列跳相邻另一列，只需跳过 1 个元素。*那步幅为什么有用呢？*

##### 8.2.1 矩阵重塑（Reshape）

步幅之所以有用，是因为它让我们能够轻松地重塑张量，而无需进行任何计算，

```text
1   2   3                          1   2
5   8   13           --->          3   5
                                   8   13

stride: [3,1]                      stride: [2, 1]

在内存中的实际存储样子：   1   2   3   5   8   13
           address:    62  64  66  68  70  72
```

将一个 (2,3) 的矩阵，重塑为一个 (3,2) 的矩阵，可以通过改变步幅来重塑，而无需实际改变其内存布局。

##### 8.2.2 矩阵转置（Transpose）

```text
1   2   3                          1   5
5   8   13           --->          2   8
                                   3   13

stride: [3,1]                      stride: [1, 3]

在内存中的实际存储样子：   1   2   3   5   8   13
           address:    62  64  66  68  70  72
```

可以在不改变内存中任何内容的情况下，将同一矩阵既视为未转置的版本，又视为转置后的版本，只需交换这两个维度上的步幅即可。

由此可见， *步长基本上能让我们实现两件事*，一是它允许我们重塑张量，而无需在内存中重新分配其存储结构，二是能够在不重新排列内存中元素的情况下转置矩阵，这非常棒，因为移动内存数据的开销很大。在 PyTorch 中，有两种方法可以重塑张量，`reshape` 和 `view`，在通过交换两个维度的步幅来转置矩阵后，就无法再免费重塑张量了，见下面的示例。

```python
>>> import torch
>>> tensor = torch.tensor([[[1, 2, 3], [5, 8, 13], [21, 34, 55],[9, 11, 13]], [[72, 42, 2],[31, 1, 92], [7, 4, 32], [88, 3, 14]]])
>>> tensor.shape
torch.Size([2, 4, 3])
>>> tensor.stride()            # 步幅属性满足下面的公式
(12, 3, 1)
>>> tensor.is_contiguous()     # 逻辑上是连续的
True
>>> transposed = tensor.permute(0, 2, 1)   # 对最后两个维度进行转置
>>> transposed.shape
torch.Size([2, 3, 4])
>>> transposed.stride()        # 内存实际存储不变，仅交换了步幅
(12, 1, 3)
>>> transposed.is_contiguous()  # 此时逻辑上已经不再连续
False
>>> transposed.view(2, 4, 3)      # 因为无法再使用 view
Traceback (most recent call last):
  File "<stdin>"， line 1， in <module>
RuntimeError: view size is not compatible with input tensor's size and stride (at least one dimension spans across two contiguous subspaces). Use .reshape(...) instead.
>>> new_tensor = transposed.reshape(2, 4, 3)   # reshape 是重建副本
>>> new_tensor.stride()                      # 步幅满足特性要求
(12， 3， 1)
>>> new_tensor.is_contiguous()               # 逻辑上也是连续的
True
```

> [!tip]
> 在 PyTorch 中，`reshape` 和 `view` 都用于重塑张量，但它们在某些情况下的行为有所不同。
> 
> - `view` 需要原始张量是连续的（即内存中数据的存储顺序没有变化），否则可能会失败。
> - `reshape` 更灵活，可以在必要时自动创建张量的副本。
> 
> 当通过改变步幅（如转置操作）来交换两个维度时，张量在内存中的存储顺序发生了变化，数据不再是连续的。这种情况下，`view` 可能无法正常工作，因为它依赖于数据的连续性。而 `reshape` 则可以处理这种情况，因为它会在需要时创建数据的副本来满足要求。

张量的步幅本质上是什么呢? 步幅是如何计算的呢? 步幅其实就是所有后续维度形状的乘积，$$\text{stride}[i] = 
\begin{cases} 
\displaystyle\prod_{j=i+1}^{N} \text{shape}[j]， & \text{if } i < N \\[1.5ex]
1， & \text{if } i = N 
\end{cases}$$
例如，一个 3D 矩阵的形状是 `(2, 4, 3)`，则步幅属性值为 `(12, 3, 1)`，转置后，例如变为 `(12, 1, 3)`，就破坏了这种规律，失去了步幅特性。

当我们进行转置时， 这种步幅特性就会丢失，在通过交换步幅完成矩阵转置后我们无法再进行进一步的形状重塑操作，从根本上说，是因为张量在逻辑上不是连续的。基本上，在 PyTorch 中，无法在张量转置后对其进行视图操作，因为 PyTorch 在转置张量时只是交换了两个步幅，但失去了步幅特性，这本质上是步幅将不再等于后续形状的乘积，参见上面的示例。

---------------------------------- 视频 02:40:49 ----------------------------------

### 9、进入 Triton

[Triton 官方文档](https://triton-lang.org/main/index.html)，本篇教程参考的[教程](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html#sphx-glr-getting-started-tutorials-06-fused-attention-py)，但做了一些修改，并大幅简化了代码，例如，去掉了 FP8的实现部分，此外，这里的代码仅适用于反向传播过程，并且只针对因果注意力机制，而我们要实现的代码则同时适用于因果和非因果注意力机制。

第二个修改是，他们没有使用这里提到的指数工具来加速运算。因为指数工具是通过一个更快的单元实现的，我们使用的是 Flash Attention 的原始实现，使用以 $e$ 为底的指数函数等，因此，尽可能简化了代码，使其易于理解，而不是一味追求优化，所以我们的代码肯定比这里看到的融合注意力要慢一些。

[官网向量加法教程](https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html)

在 Triton 中，我们只需指定需要多少个线程块，而不强制规定要启动多少个线程，Triton 会自行决定启动多少个线程，我们只需告诉每组线程应该执行什么任务。以该例子来说，我们将元素数量 $N$ (这里是 98432)，划分为大小为 1024 的块，块的数量需要向上取整。

当启动内核时，可以在方括号中指定启动网格，然后在圆括号中指定这个内核的参数。进入内核部分， 我们发现 Triton 并未直接让我们访问张量 $x$，它提供的是指向该张量第一个元素的指针，这让我们回想起张量的布局方式，我们之所以研究张量布局、步幅等概念，是因为这段代码将在 GPU 上运行，而 GPU 并不能像 PyTorch 那样通过多维索引以及广播等高级功能直接操作张量，GPU 只会提供指向内存中张量首元素的指针。

首先， 我们需要确定当前处理的是哪个块，在 CUDA 中我们使用名为 `blockid.x` 的变量来标识块的编号，它指示我们应该处理哪一组元素，在 Triton 中， 可以通过使用 `program_id` 实现相同的功能，在 CUDA 中块 ID 可以沿着 x、y 和 z 轴分布，而在 Triton 中，这些被称为维度 0、1 和 2，这里我们处理的是单维数据，因此只需使用一个轴来指定块的索引。

超出实际需要的线程数的多余线程应保持闲置状态，不加载任何数据，也不参与任何计算求和的操作，正因如此，我们需要借助这个掩码来实现上述控制。掩码的作用是，在所有的偏移量中，指示当前线程块仅处理那些实际存在，且掩码值为真的元素。

接下来计算输出 $x+y$ 的结果，在 CUDA 中使用 `output[i] = x[i] + y[i]`，那时我们是一次处理一个元素，因为每个线程只负责一个索引，而在这里， 我们是以一组元素为单位进行处理的，这里的 $x$ 是一个元素块，最后将结果写入 output，可以看到这里的 ptr 是指向输出向量第一个元素的指针， 我们得确定这个输出向量该存放在哪里，它的尺寸和形状是怎样的呢? 我们应该把它保存在我们加载 $x$ 的相同偏移位置，同时还要使用掩码。

*在 CUDA 中编写的程序是以线程为单位的*，每个线程需要明确自己应该执行什么操作，而*在 Triton 中操作的单位是数据块*，我们是以线程块为单位进行操作的。

### 10、Flash Attention 中的并行化

![[Pasted image 20250321163124.png|600]]

这段代码是 FlashAttention 的前向传播过程，我们可以并行化输出的计算过程， 因为输出块依赖于查询块以及所有的键块，不同的输出块依赖于不同的查询块与所有键块，它们可以相互独立地工作，当然它们共享相同的键块。

#### 10.1 共享内存

在 GPU 中，我们拥有高带宽内存，类似于计算机中的 RAM，如 A100 的 40GB 高带宽内存（HBM）的容量，也就是 DRAM。

![[Pasted image 20251216182042.png|500]]

如图所示，最下面是 DRAM，它是 GPU 所具备的大容量内存，接着，每个流式多处理器 (线程块)，实际上还拥有一块共享内存，它比 DRAM 小得多，区别在于访问 DRAM 的速度非常慢，而访问共享内存的速度则极快。

CUDA 与 Triton 的一个区别在于，当你在 CUDA 中加载某些数据时，你是直接从全局内存中加载的，例如，在使用 torch 实现的原始版本注意力计算中，当我们启动一个 CUDA 内核时，我们先将张量或向量从 CPU 复制到 GPU，它们会驻留在 GPU 的全局内存中，然后我们直接从全局内存中加载这些元素。

在 Triton 中，每当加载一些数据时，实际上是将数据从全局内存复制到共享内存中，然后，所有操作都在共享内存上完成，而当存储数据时，再将数据从共享内存复制回全局内存，这一过程大大提升了速度。因此，我们始终操作的是那些已加载到共享内存中的元素，本质上是由同一线程块内的所有线程共享的。

#### 10.2 FlashAttention 算法回顾

**FlashAttention-1 vs. FalshAttention-2：**

在 FlashAttention-2 中，有一个外循环，用于遍历所有的 $Q$ 块，以及一个内循环，用于遍历所有的 $K$ 块，而在 FlashAttention-1 中，外循环处理的是 $K$ 块，而内循环处理的是 $Q$ 块，这种设计降低了算法的并行化能力，为什么？

由于注意力机制的输出可以针对每个 $Q$ 块独立计算，因此，对于 $Q$ 的外循环，实际上并不需要真正运行一个循环，而是可以启动多个内核并行处理，而内循环 $K$ 块则是我们必须遍历的部分，因此每个 Triton 内核将负责处理一个 $Q$ 块，并逐一遍历所有的 $K$ 块，在 $K$ 块内部，我们将执行之前探讨过的那些操作，在循环结束时，我们需要将输出结果存储回 HBM 内存中。

另外需要注意的是，这里的 $Q$、$K$ 和 $V$ 都是 $n \times d$ 维的矩阵，而通常处理的是一个 batch 的序列，因此，我们还可以在批次中的序列上进行并行化处理，每个批次可以独立工作。在每个序列内部，每个注意力头可以独立工作，每个小 $Q$ 块也能独立正作，这就是我们实现并行化的方式。那么我们最多能有多少个程序同时并行运行呢？它等于 `batch_size * head_num * q_block_num`

### 11、Triton 编码实现

**自定义实现与 Triton 文档版本区别：**

1. 没有使用 FP8，这对我们的解释来说并不必要，当然，使用 FP8 
2. Triton 网站上的 FlashAttention 中，反向传播仅针对因果注意力机制实现，而我们会同时支持因果和非因果注意力机制，尽管速度会慢一些
3. Triton 中显式的使用了 softmax 缩放因子，而我们实际上在需要时才应用这个缩放因子
4. Triton 在线计算 FlashAttention 时，使用的是 $2^x$ 并非 $e^x$，然后通过使用对数进行补偿

### 12、从导数到雅可比矩阵

导数：$$f'(x)=\lim_{ h \to 0 } \frac{f(x+h)-f(x)}{h}=\frac{\partial f(x)}{\partial x}=\frac{\partial y}{\partial x}$$
稍加整理：$$\begin{align*}
f(x + h) &\approx f'(x) \cdot h + f(x) \\[0.5em]
f(x + \Delta x) &\approx f'(x) \cdot \Delta x + f(x) \\[0.5em]
f(x + \Delta x) &\approx \frac{\mathrm{d}y}{\mathrm{d}x} \cdot \Delta x + f(x) \\[0.5em]
y^{\text{NEW}} &\approx \frac{\mathrm{d}y}{\mathrm{d}x} \cdot \Delta x + y^{\text{OLD}}
\end{align*}$$
所以，当 $x$ 改变了 $\Delta x$，$y$ 将近似的改变 $\frac{\mathrm{d}y}{\mathrm{d}x}\cdot \Delta x$

#### 12.1 链式法则

假设：$z=f(g(x))$

由 $x^{\text{new}}=x^{\text{old}}+\Delta x$  推导出  $y^{\text{new}} \approx \frac{\mathrm{d}y}{\mathrm{d}x} \cdot \Delta x + y^{\text{old}}$

由 $y^{\text{new}}=y^{\text{old}}+\Delta y$  推导出  $z^{\text{new}} \approx \frac{\mathrm{d}z}{\mathrm{d}y} \cdot \Delta y + z^{\text{old}}$，

将上式 $\Delta y$ 部分带入可得，$z^{\text{new}} \approx z^{\text{old}} +\frac{\mathrm{d}z}{\mathrm{d}y} \cdot \frac{\mathrm{d}y}{\mathrm{d}x} \cdot \Delta x$

于是，可得：$\frac{\partial z}{\partial x}=\frac{\partial z}{\partial y}\cdot \frac{\partial y}{\partial x}$
 
#### 12.2 梯度

梯度：函数的输入是向量，输出是标量。$f$：$R^{N}\to R$

例如：$f\!\left(\begin{bmatrix} x_1 \\ x_2 \end{bmatrix}\right) = y$，$y^{\text{new}} \approx y^{\text{old}} + \nabla f \cdot \Delta x$，这里的 $\Delta x$ 也不再是标量，是向量，将其与梯度点积

梯度定义：$\nabla f = \begin{pmatrix} \displaystyle \frac{\partial y}{\partial x_1}, & \displaystyle \frac{\partial y}{\partial x_2}, & \dots \end{pmatrix}$，所以有：$y^{\text{new}} \;\approx\; y^{\text{old}} + \frac{\partial y}{\partial x_1} \Delta x_1 + \frac{\partial y}{\partial x_2} \Delta x_2 + \dots$

梯度实际上是一个由输出相对于输入向量中，每个变量的偏导数组成的。

#### 12.3 雅可比矩阵

雅可比：函数的输入是向量，输出也是向量。$f$：$R^{N}\to R^M$

例如：$f\left(\begin{bmatrix} x_1 \\ x_2 \end{bmatrix}\right) =\begin{bmatrix} y_1 \\ y_2  \\ y_3 \end{bmatrix}$，$x^{\text{old}} \to x^{\text{old}} + \Delta x$，有 $y^{\text{new}} \;\xrightarrow{\approx}\; y^{\text{old}} + \frac{\partial y}{\partial x} \Delta x$

雅可比矩阵：$$\text{Jacobian} = 
\begin{bmatrix}
\displaystyle \frac{\partial y_1}{\partial x_1} & \displaystyle \frac{\partial y_1}{\partial x_2} & \cdots & \displaystyle \frac{\partial y_1}{\partial x_N} \\[1.5em]
\displaystyle \frac{\partial y_2}{\partial x_1} & \displaystyle \frac{\partial y_2}{\partial x_2} & \cdots & \displaystyle \frac{\partial y_2}{\partial x_N} \\[1.5em]
\vdots & \vdots & \ddots & \vdots \\[1.5em]
\displaystyle \frac{\partial y_M}{\partial x_1} & \displaystyle \frac{\partial y_M}{\partial x_2} & \cdots & \displaystyle \frac{\partial y_M}{\partial x_N}
\end{bmatrix}$$
其中 $\frac{\partial y}{\partial x} \Delta x$ 是矩阵-向量乘法，$(m,n) \times (n,1)=(m,1)$ 

#### 12.4 广义雅可比矩阵

广义雅可比矩阵：函数输入是张量，输出也是张量。$f: \mathbb{R}^{N_1 \times \cdots \times N_{D_x}} \to \mathbb{R}^{M_1 \times \cdots \times M_{D_y}}$

例如：$f\bigl( D_x\text{-dimensional tensor } \mathbf{x} \bigr) = D_y\text{-dimensional tensor } \mathbf{y}$

其中 $\frac{\partial y}{\partial x} \Delta x$ 是张量乘法，$(M_1 \times \cdots \times M_{D_y})\times(N_1 \times \cdots \times N_{D_x})$ 

#### 12.5 自动求导（Autograd）

```
(a)   -----  (x)  -----  (+)  -----  (z^2)  ------  (phi)
              |           |
              |           |
             (w_1)       (b_1)
```

表达式：$\phi=y_{3}=(y_{2})^2=(y_{1}+b_{1})^2=(aw_{1}+b_{1})^2$

对 $w_{1}$ 求偏导：$\frac{\partial\phi}{\partial w_{1}}=2(aw_{1}+b_{1})(a)=2a(aw_{1}+b_{1})$

链式求导：$\frac{\partial\phi}{\partial w_{1}}=\frac{\partial\phi}{\partial y_{3}} \cdot \frac{{\partial y_{3}}}{\partial y_{2}} \cdot \frac{{\partial y_{2}}}{\partial y_{1}} \cdot \frac{{\partial y_{1}}}{\partial w_{1}}$


#### 12.6 雅可比矩阵稀疏性

由于输入 $X$ 的维度 $(N,D)$ 以及权重 $W$ 的维度 $(D,M)$ 都非常大，会导致雅可比矩阵非常的大，然而这个雅可比矩阵及其稀疏，

```
--------------              - - - - - - - -               = = = = = = = 
--------------              |             |               = = = = = = =
--------------      x       |             |        =      = = = = = = =
--------------              |             |               = = = = = = =
--------------              |--------------               = = = = = = =
  X (N,D)                       W (D,M)                       Y (N,M)
```

以图示为例，$X$ 每一行是一个 token，维度是 $D$，输出中每一行是该 token 的某种表示，以 $Y$ 中的第一行值为例，$Y$ 中的第一行是由 $X$ 的第一行与 $W$ 的每一列乘积得到的，也就是说与 $X$ 的其他行无关，其偏导自然为 0，这就产生大量的稀疏性。不过我们一般无需对 $X$ 求导，对 $W$ 求导一样的。
 
#### 12.7 如何不实际生成雅可比矩阵进行优化

(05: 00: 00)



如何在不实际生成 Jacobian 的情况下优化这个计算，
因为这对 Flash Attention 来说是必需的.
好的， 各位.
在继续讨论反向传播之前，
我们先来看一下 Flash Attention 反向传播的公式.
在讨论 Flash Attention 之前，
我们先来看看如何计算矩阵乘法操作相对于其输入的梯度.
假设我们已经知道， Py Torch 实际上已经提供了
如何通过损失函数
相对于矩阵乘法输入的梯度
来计算矩阵乘法输入的梯度的方法.
但在 Flash Attention 中， 我们正在创建一个自定义内核，
这意味着这个自定义内核将多个操作融合为一个操作.
因此， 当 Py Torch 调用我们的操作符时， 它会询问我们的操作符
也就是我们构建的 Triton Attention 操作符
损失函数相对于q 、k和v的梯度是多少
因为这些都是我们函数的输入.
如果我们看一下目前已经构建的代码，

你会发现我们的 Triton 操作将在计算图中作为一个节点
它接收 Q、K和 V作为输入， 并生成一个输出，
然后，Py Torch 会提供损失函数相对于该输出的梯度
因此， Py Torch 会给我们一个d O (即损失函数相对于输出的梯度)
也就是损失函数相对于输出 O的导数.
接下来， 我们会要求这个类 一一 即 Triton Attention -
计算损失函数相对于 Q、 K和 V的梯度
因为我们是将多个操作融合在一起执行.
因此， 我们是在实时计算
查询(query )与键(key) 转置相乘后的接着再进行soft max 操作， 并将其与值( V )相乘， 从而得到输出结果
我们需要在内部计算这个梯度，! 以便计算输入( Q、 K、 V)白 的梯度.
因此， 由于我们正在将这些操作融合在一起执行
其中包括矩阵乘法 我们需要手动推导矩阵乘法的梯度
即损失函数相对于矩阵乘法操作
输入的梯度，
以便能够将其提供给 Py Torch.
这正是我们需要推导这个公式的原因
我会用一种非常简单的方式来推导它
然后我们也会对softmax进行同样的推导
因为这两部分是我们需要手动推导的
以便得出 Flash Attention 反向传播的公式.
那么， 我们开始吧.
假设我们在计算图中有一个节点称为矩阵乘法，
这个节点正在执行矩阵乘法操作.
也就是说， 它正在计算以下操作:y等于×乘以 W.
现在， 当 Py Torch 计算这个节点的反向传播时， 它会提供什么作为输入呢?
Py Torch 会提供损失函数的梯度，
也就是损失函数相对于这个节点输出的梯度， 即d中/dy，
并要求我们计算损失函数相对于输入×和参数w的梯度，
和 do /dw.
我将演示其中最简单的一个，
另一个则不会在视频中展示，
但我会附上 PDF 幻灯片来说明它是如何计算的，
因为它们的计算方式非常相似，
我不想因为不必要的重复而使视频变得过长.
现在， 我们来计算损失函数相对于输入
×的梯度.
那么， 如何在不显式构建雅可比矩阵的情况下手动计算呢?
因为， 正如 我们所看到的， 我们不能直接通过显式构建雅可比矩阵来使用链式法则
虽然这是最简单的方法，
但雅可比矩阵是一个非常大的矩阵， 甚至无法放入 GPU 的内存中
因此， 我们需要找到一种更巧妙的方法.
我们利用了雅可比矩阵稀疏的特性，
希望最终能得到一个不需要显式构建庞大
稀疏雅可比矩阵的公式.
让我们来看看具体怎么做.
嗯， 让我们看看， 嗯， 让我们换个角度.
在处理这类推导时， 我总是建议从一些具体的例子入手:
比如张量.
假设×是一个大小为... 的张量.
假设x是一个大小为nxd 的张量， 其中n设为 1， d设为
而 W 也是一个张量， 或者说矩阵， 形状为dxm，
其中m设为4.
因此， Y的形状将是nx m.
所以它的形状将是1×4.
Py Torch 将为我们提供以下结果.
它将生成这个结果.
即损失函数相对于该操作符输出
Y 的梯度.
因此， 它将生成一个维度为n×m的向量
或张量.
我们需要计算损失函数相对于×的梯度
这将是一个形状为n xd 的张量. 因为在处理梯度时，
它总是与输入变量的形状相同，
这是由于输出是一个标量， 相对于输入中的每个元素而言，
因此梯度的形状与分母一致.
好的.
因此， 在处理这类问题时， 我通常建议先创建示例矩阵，
观察输出结果的变化， 然后再尝试推导出梯度矩阵.
那么， 我们就开始吧.
让我们来看看输出是如何计算的.
输出将是一个1×4的矩阵.
其计算过程如下.
输入是一个1×3的矩阵.
我们将输入记为 X11、 X12、 X130
这个输入将与一个3×4维的矩阵 W相乘.
矩阵 W有3行4列.
矩阵 W的元素记为 W11、 W12、 W13、 W140
如果我们进行矩阵乘法运算， 结果将会是这样的.
运算后将生成如下矩阵， 这是正确的.
这是一个1行3列的矩阵，
这是一个3行4列的矩阵
因此， 输出的矩阵将是1行4列的.
即1行4 列的矩阵.
为了便于展示， 我将用较小的字体书写， 否则这里可能放不下.
那么， 我们就这么处理吧.
结果将是 X1乘以 W11， 加上 X12乘以 W21，
XW+ XW2 再加上 X3乘以 W310
X W + X 这将是输出矩阵的第一个元素.
输出矩阵的第二个元素将是×11乘以 W12，
加上 X12乘以 W22， 再加上 X13乘以 W320
这将是输出矩阵的第二个元素.
输出矩阵的第三个元素将是 一一让我把这些内容移到左边
否则可能放不下.
好了， 现在应该能放下了.
这个元素将是×1乘以 W13， 加上 X12乘以 W23，
再加上 X13乘以 W330
接 着， 我们将同一行与最后一列相乘， 得到 X11乘以 W14， 加上 X12乘以 W24
再加上 X13乘以 W340
这就是矩阵乘法得到的输出 Y.
这就是 Py Torch 会提供给我们的结果.
它会给出损失函数的梯度.
它会给出delta phi 相对于deltay 的梯度， 因为这就是梯度的含义.
它的形状与分母相同， 因此其形状为1×4.
我们暂且称之为未知值， 因为目前还不清楚这个数值会是多少.
这些值将由 Py Torch 提供给我们.
我们暂且给它们起个通用名称， 比如dy 11、dy 12、
dy13和dy14. 现在，
为了计算我们需要提供给 Py Torch 的下游梯度，
我们应该构建 Jacobian 矩阵， 也就是
好的， 让我们写下链式法则的公式.
因此， 我们需要提供delta phi 相对于delta x 的梯度，
这等于delta phi 相对于deltay (由 Py Torch提供)
乘以 Jacobian 矩阵， 即deltay 相对于deltax的梯度.
现在， 与其直接构建这个 Jacobian 矩阵， 我们不妨尝试另一种方法.
现在， 我们尝试将其具体化，
并对这两个量进行乘法运算， 看看是否能简化某些部分.
那么这里的部分就是dy 相对于dx 的导数，
也就是每个输出y 相对于每个输入×的导数.
我们有多少个输出呢?
我们有四个元素作为输出， 也就是这里的这些.
而输入矩阵×中有三个元素.
因此， 结果将如下所示:我
无法直接复制它， 因为我的屏幕不够大
我记得x是x1、x2和x3.
因此， dy 相对于dx 的导数将包含以下项:
y1相对于x11的导数一一可以看到，
y1中只有一个x11与w11相乘.
因此， 相对于x11的导数将是w11， 然后是y11.
这就是结果.
相对于x12， 它将是w21， 然后是×.
Y11相对于×13的导数将是 W31.
这个矩阵的第二行将是
第二个输出 
相对于所有×输入的偏导数，
也就是这里的这些项相对于每个×的偏导数， 分别是 W12、 W22
和 W32.
现在让我检查一下我做的对不对.
是的， 因为我已经做过了， 所以我可以随时复查一下.
然后我们有 W， 这里的这些项相对于所有×的偏导数
分别是 W13、 W23和 W33.
然后是最后一个输出y4 相对于所有×的偏导数，
分别是w14、w24和w34.
我们得到了如下的 Jacobian 矩阵.
但这个 Jacobian 矩阵， 如你所见， 其实就是w 的转置.
所以我们不需要具体化这个 Jacobian 矩阵.
我们只需要将 Py Torch 提供的梯度
与 W 的转置相乘， 就能得到下游的梯度.
那么让我重写一下， 这样我们就清楚自己在做什么了.
因此， dΦ/dx等于dΦ/dy乘以dy/dx.
大 但我们已经看到， dy/dx其实就是w的转置.
所以， d/dx等于dΦ/dy乘以w的转置，
这样我们就得到了下游的梯度.
因此， 为了提供 Py Torch 所需的下游梯度，
我们只需将 Py Torch 提供的梯度与w的转置相乘
大 就能得到损失函数
关于矩阵乘法输入×的梯度.
同样地，
我们也可以写出损失函数关于 W 的梯度公式
它等于x的转置乘以d中/dy关于dw的部分.
如何记住这些公式呢?
这里有一个记忆法则， 那就是
这些是唯一能让这个公式符合×的形状，
那个公式符合 W的形状的方式.
因为这里的这个部分， 会与 Y 的形状相同.
因此它的形状将是n乘以m.
而这里的这个部分， 形状将与 W 的转置相同.
W 的尺寸是d乘以m，
所以w的转置应该是m乘以d， 而矩阵乘法
或张量乘法的结果将是/n乘以d， 这与×的形状完全一致.
在这种情况下， xt 是t的转置， 尺寸为n乘以d，
因此是d乘以n再乘以d中/dy， 这是一个梯度，
所以它与分母的形状相同.
因此， 它的尺寸是n乘以m， 而输出的形状将是d乘以m， 这正好与
W 的形状一致.
所以如果你要记住这些关系， 这是唯一能让形状匹配的方式，
否则就无法成立.
因此， 这是一个用于记忆
如何根据矩阵乘法输出的损失梯度，
计算输入梯度的记忆公式.
而矩阵乘法的输入是输入矩阵
和参数矩阵 W
现在我们需要推导soft max 输出
相对于其输入的梯度，
因为这是我们在融合注意力机制中进行的另一个操作.
我们将多个操作融合在一起
包括矩阵乘法和soft max.
因此， 这是理解 Flash Attention 反向传播
所需的第二个关键要素.
那么， 让我们开始吧.
在进行这个推导时， 我将
采用与 Flash Attention 论文中相同的符号表示.
首先， 让我们为这部分内容写上标题:
通过soft max 的梯度计算.
在计算注意力机制时， 我们首先进行的操作是
计算查询向量与键向量转置的乘积.
我们以分块的方式进行计算， 即逐块处理，
但这并不影响最终结果， 因为最终效果是相同的
因此， 我们可以将 S表示为 Q与键转置的乘积.
接着， 我们对这个结果应用soft max 函数.
将这一操作的结果称为 P， 也就是 S的soft max 输出.
在应用soft max 之后， 我们取其输出
并将其与 V 相乘， 从而得到最终的结果.
因此， 输出等于 P乘以 V.
现在我们需要理解如何进行计算， 因为正如我之前提到的
Py Torch 的 Auto Grad 机制是按照以下方式工作的.
Py Torch 会将我们的注意力计算视为一个黑箱.
因此， 我们将得到如下的计算图.
我们将有一个 查询输入、一个键输入和一个值输入， 这些都是由一系列标记组成的序列
每个标记都有一定的嵌入维度.
这些输入被送入一个称为注意力的黑箱
这是我们自己实现的注意力机制，
也就是我们之前开始编写的那个函数.
这些输入将作为计算图中这个节点的输入，
而计算图将输出一个张量0.
Py Torch 会给我们什么?
Py Torch 会提供损失相对于输出的梯度.
正如你所记得的， Py Torch 会逐个访问每个运算符并询问:
如果我将损失相对于你输出的梯度提供给你，
你能返回给我损失相对于你输入的梯度吗?
这正是我们需要解决的问题.
因此， 在已知损失相对于输出的梯度的情况下，
我们需要弄清楚如何计算损失相对于 WQ 的梯度.
以及损失相对于 WK 的梯度， 和损失相对于 WV 的梯度.
然而， 由于存在两个中间操作，
Q 与 O之间或 K与 O之间并没有直接的连接.
首先是一个矩阵乘法， 接着是soft max 操作
然后再进行另一个矩阵乘法.
然而，
我们拥有工具能够帮助我们理解
梯度是如何通过这些操作传播的.
当多个操作依次应用时，
这就是所谓的链式法则.
然而，
我们已经看到， 如果以最直接的方式应用链式法则
并具体化雅可比矩阵， 实际上是不可行的.
因此， 我们需要理解如何在不具体化雅可比矩阵的情况下应用链式法则
这正是我们将要
针对注意力计算中的一个操作
-soft max 一- 去解决的问题.
这就是为什么我们要进行这次推导，
我保证这是我们将要做的最后一次推导.
然后， 我们最终将着手编写flash attention 的反向传播代码.
我们无法直接着手编写flash 的反向传播代码.
如果我们直接看attention 的计算公式，
是无法理解推导过程是如何得出的.
好的， 现在我们可以开始了.
让我把这些内容删掉吧.
删除.
为了简化理解， 假设我们现在对s 矩阵逐行应用soft max，
也就是说每一行都独立地进行soft max 计算.
那么， 让我们看看矩阵的某一行会发生什么变化， 为了简化，
我将这一行称为 S.
所以， S代表s矩阵中的某一行.
我也可以称它为s的第i行
但如果这样表示， 我们就得一直带着这个索引1了.
好吧， 伙计们， 动手做吧.
我们会带着这个索引一起推导.
好的， 我们就用 SI来表示 S矩阵中的某一行吧.
因此， SI的表达式， 用张量表示法一一也就是 Py Torch 的张量表示法会是这样的.
也就是说， 从矩阵 S或者说张量 S中， 我们取出第i行以及所有的列.
这就是 SI 的定义.
我知道这个表示法看起来不太美观， 但它有助于你们理解.
这是一个具有特定大小和维度的向量.
我们在 这个向量上应用soft max 函数， 将会得到一个输出向量， 我们称之为 Pl.
PI等于 SI的soft max.
正如我们所见， soft max 操作不会改变输入的形状，
它只是逐个元素地改变数值.
因此， 输出也将是一个大小为r的n次方的向量.
那么， 什么是soft max 呢?
s of max 的定义如下.
即 pij 的 soft max.
因此， p-i向量的第j个元素等于
s-i向量的第j个元素的指数，
除以一个归一化因子， 这个因子按如下方式计算:
这里不用j， 我们用k来表示.
不用k， 我们用 I来表示.
等于从 1到n的e的si次方的和.
好的， 首先， 你可能会疑惑:
我们在计算注意力机制的前向传播过程中使用的soft max，
并不是这个原始的soft max.
因为如果你 还记得之前我们应用的方法， 我们实际上使用的是经过调整的其中每个指数函数的参数都减去了
该向量中的最大值元素.
所以它大致是这样的.
即 Sij 减去 Simax.
也就是 Sij 向量中的最大值元素.
同时， 分母中的参数也减去了 Simax.
然而，
我们也证明了这里的操作
与不进行参数减法的标准soft max 是等价的，
因为这种参数减法只是
为了确保数值计算的安全性.
但从数学角度来看， 不进行减法的计算方式也是等价的.
在计算机上， 当然， 这样做可能会导致数值不稳定，
但从数学的角度来看， 它们是相同的.
这也意味着无论你如何计算前向传播过程， 结果都是一样的.
如果它与另一个数学定义等价，
你总是可以使用那个数学定义来计算反向传播.
最终得到的结果将是相同的.
如果你没听懂我刚才说的， 让我用一个更简单的例子来说明:
想象你有一个a.
你还记得高中时学过的那个公式吗?
就是这个:cos2x+ sin2x=1.
现在， 假设我们计算一个输出:y=cos2x.
然后我们需要计算y 对×的导数.
无论你是将cosx 对×求导，
还是将1-Sin2x对x求导
结果都会完全相同，
因为这两个定义是等价的.
正因为如此， 我们无需在指数部分额外添加这个因子，
因为从数学上讲， 这两种定义是等价的.
我们只需采用数值上更安全的那个定义， 因为在计算机上进行计算时
我们需要确保数值稳定性.
这样就不会出现溢出问题.
好的， 那么， 我们想要得到什么呢?
所以， 我们希望在已知损失函数关于soft max 输出(即 Pi向量)的梯度的情况下
所以， 我们希望在已知损失函数关于soft max 输出(即 Pi 向量)的梯度的情况下
计算出损失函数
关于soft max 输入向量(即 Si 向量)
的梯度.
通过链式法则， 我们可以得到这个结果.
将其乘以 Pi关于 Si的雅可比矩阵.
现在， 链式法则始终是有效的.
让我们来看看这个雅可比矩阵是什么样子的.
嗯， 好的， 那么这个雅可比矩阵就是 DPI 关于delta SI的导数.
嗯， 我们需要完成这个计算.
让我们来仔细看看这个雅可比矩阵中每个元素的具体形式.
那么， 第j个元素相对于第k个元素的偏导数.
我们正在计算， 或者说，
我们正在研究这个雅可比矩阵中每个元素的具体形式，
也就是雅可比矩阵到底是什么.
它指的是雅 可比矩阵中输出(分子)的每个元素相对于输入(分母)的每个元素的偏导数
也就是这个分数形式中的每个分量.
也就是说， 我们正在分析输出向量中的每一个元素
相对于输入向量中每一个元素的偏导数.
这就是我们在这里写的内容. 那么， 输出向量是如何得到的呢?
嗯， Pij， 我们知道它等于.
根据soft max 的定义， 它是通过以下方式得到的.
即e 的 Si次方除以归一化因子， 我们称之为 L，
等于 1到 n.
e的 Si L 次方， 全部对 Sik 求导.
所以， 我们正在尝试做的是， 我们知道 P 向量是
假设它是一个包含三个元素的向量， 那么这是p1，
这是p11， p12 和p13.
向量也将是一个包含三个元素的向量， 因此它将包含s11、
我们的标是计算 Jacobian 矩阵，
然后
Jacobian 矩阵的第二行将是
这个向量对每一个输入元素的导数.
接着， Jacobian 矩阵的第三行将是这里的内容
对 S向量中每一个输入元素的导数.
我们正在努力理解这一点.
这个 Jacobian矩阵中的通用元素是什么样子的.
基于输出向量的第j个元素，
因此 这个j索引指的是输出向量， 而k索引指的是输入向量中的第k个元素
好的， 当我们计算这个 Jacobian 时， 可能会出现一种情况.
这里的这个表达式是两个函数的商的导数，
我们从高中就知道， 两个函数的商的导数
如下所示.
所以， f(x)对g(x)的导数一一让我这样写
关于×的导数等于
f(x)乘以g(x)减去g'(x)乘以f(x)， 除以g(x)的平方.
f(x)除以g(x)的平方.
就像这样.
现在让我们在这里应用它.
所以这将变成， 这里我们会有两种情况.
要么我们求导的变量
即这个s_i_k， 与被求导的变量具有相同的索引.
所以我们要么是在计算 P11对 S11的导数，
要么是在计算 P11对另一个不同索引的变量的导数.
比如 P11对 S12或 S13的导数.
因此， 我们需要考虑两种情况.
假设我们在计算 P11对 S11的导数，
或者 P12对 S12. 的导数， 或者 P13对 S13的导数.
也就是说， 我们在计算输出向量中某个元素
对输入向量中具有相同索引的元素的导数.
在这种情况下， 这个导数会呈现如下形式:
它是函数f
对分母中具有相同索引的变量的导数.
也就是说， 在这种情况下， j等于k.
因此， 分子对 Sij的导数， 也就是e的 Sij次方对 Sij的导数，
结果就是e的 Sij 次方.
因为e的x1次方对x1的导数就是e的x1次方.
所以这等于， 我现在要缩小一下尺寸.
e的 Siji次方， 然后我们需要将其乘以分母
也就是这里的这个求和项.
所以对所有可能的 L求和e的 Sij 次方，
再减去分母对所求导变量的导数.
因此， 这个分母是所有输入元素的指数之和.
如果我们对某一个特定的输入元素求导，
至少会有一项包含该输入元素，
而其他所有项的结果都会是零.
因此， 唯一剩下的导数将是e的sik 次方
对sik 的导数.
所以我们写成减去e的sik 次方， 乘以分子，
也就是e的sij 次方.
所有这些除以分母的平方.
也就是这里的这个求和.
所以 I从1到n， e的sil 次方， 全部平方.
而这里的这一部分将等于， 我们可以看到， 这两个项，
这个和这个有一个共同因子， 即e 的sij 次方.
所以我们可以提取出来.
所以e的sii次方乘以求和减去e的sik次方，
所有这些除以分母
也就是这里这个东西的平方.
那么让我复制粘贴一下， 同时旋转一下
因为我不知道为什么我总是写得小小的
好了， 这里的东西等于， 嗯， 我们可以把这两项分开.
所以我们可以把这里的这一项和这一项分开.
因为分母是平方，
所以我们可以这样写:e的 Sij 次方除以分母，
即 I从1到n的e的 Sil次方的和，
乘以这里的东西.
所以这里的东西除以相同的分母，
即 I从1到n的e的 Sil次方的和，
减去e的 Sik 次方，
再除以相同的分母， Sil.
现在这个可以写成:这里的东西不过是输出元素 Pij，
因为这个只是应用于 Sij 元素的soft max， 我们都知道.
因为应用于 Sij 元素的soft max 输出称为 Pij，
因为它是我们称为 P 的输出向量的一个元素.
所以这里的东西等于 Pij，
乘以这里的东西将等于1 减去这里的东西.
这里的东西是应用于 SIK 元素的soft max 输出，
所以它将是 PIK.
所以它等于1减去 PIK.
好的， 这是在这种情况下.
我们求导的变量与分子具有相同的索引.
在这个分数中， 在这个导数中.
另一种情况是当两个变量， 即输出的索引
与输入的索引， 不相同的时候.
在这种情况下， 我们会有另一种情况.
那么我们会得到丨吗?
嗯， 让我再写一遍.
所以这里的内容， 我希望我能全部复制下来而不出错.
在另一种情况下， 即当s不等于j时，， s是什么?
S ik 是j不等于k， 所以j不等于k.
在这种情况下会发生什么?
嗯， 这将是分子的导数，
因为我们需要再次应用这里的这个公式.
所以， 分子相对于不同变量的导数
将为零，
因为这就像计算e 的×一次方
相对于×二的导数，
结果会是零.
所以它将为零.
因此， 无论g(x)是什么， 这里的整个第一项都会变成零
再减去这个分数
分母相对于变量sik 的导数
g， 关于 sik 的导数
所以这是输入中的所有变量
而我们正在对输入中的一个特定变量求导.
因此， 在求和过程中只有一项会保留下来， 那就是sik 这一项.
所以它将是e的sik 次方乘以f(x)，
也就是这个分数的分子， 即e的幂次.
哦， 我们漏掉了减去e的 sij 次方这一项.
让我看看是否遗漏了什么.
所有这些除以这里分数的分母的平方，
所以它等于从 I等于1到n， e的sil 次方的总和
再整体平方.
嗯， 我想我没有遗漏任何东西， 那么我们继续吧.
所以这里我们可以看到， 这部分是因为一一， 好吧， 我们分开来看
减去e的sik 次方， 除以从 I等于1到n， e的sil次方的总和，
再乘以e的sij 次方，
除以从 I 等于1到n， e的sil 次方的总和.
这里的这一项不过是对 SI向量的第j个元素应用soft max 的结果
所以我们知道这些是什么了.
我们知道我们称之为 p减去 pik乘以 pij.
所以最终我们有两种情况.
一种是这里这个东西的导数.
看起来如下.
雅可比矩阵中的每一项看起来如下.
当分子和分母具有相同索引时.
即j等于k.
这里的东西现在等于.
这里的符号表示有误， 我不应该用等号来书写.
不过没关系， 朋友们， 我们只是稍微讨论一下.
让我检查一下.
P是的， 一种情况是当」不等于 K时.
那么这里的内容一一让我这样写
将等于负的 PIK 乘以 PIJ 既然我们已经了解了雅可比矩阵的两种典型情况，
现在让我们来看看这个雅可比矩阵在矩阵形式中的具体表现.
所以这个雅可比矩阵将如下所示.
它将是一个大致如下的矩阵.
它将是一个nx n的矩阵， 其中n是输入向量和输出向量的大小.
这里是雅可比矩阵的第一个元素.
如你所见， 如你所记得的
按照分子布局， 雅可比矩阵的第一行
是第一个输出对所有输入的导数.
因此， 这里的第一个项将是 P11对 S11的导数.
因此， 在这种情况下， J和 K匹配， 所以我们知道它将等于 P.
P11乘以1减去 P11
这个元素右边的第二个元素.
所以元素 P12将是 P12对... 的导数.
抱歉， 是 P11对 S12的导数.
J和 K不匹配， 因此我们将处于这种情况， 所以它将是负的 P11
乘以 P12.
第三个元素.
你可以自己验证一下.
它将等于负的 P 11乘以 P13， 依此类推， 直到最后一项
是负的 P11乘以 P1n.
雅可比矩阵的第二行将如下所示:
因此， 它将是 P12对 S11的导数.
J和 K不匹配， 所以我们处于这种情况.
因此， 它将等于负的p12乘以p11.
然后是下一个元素:它将是p12对s12的导数.
因此， j和k匹配， 我们处于第一种情况.
因此， 它将等于p12乘以1减去p12.
那么这里的部分将等于.
接着， 第三个元素将是p12对p13的导数， 以此类推.
直到我们到达最后一个元素， 即负的p12乘以p1n，
而不是 p12对 p1n的导数.
所有元素都是如此， 直到最后一行.
最后一行将是.....
最后一行的第一个元素将是
最后一个输出元素对第一个输入元素的导数.
因此， 它将是p1n对s11的导数.
所以， 这两个索引并不匹配.
所以我们处于第二种情况.
所以它将是负的p1n乘以p11.
这将依次为负的p1n乘以p12， 依此类推.
既然我们在这里， 我也来算一下第三个元素吧.
所以是负的p1n乘以p13， 以此类推.
直到最后一行的最后一个元素， 我想应该是负的p1n乘以p1n.
哦不， 这样不对， 因为这两个索引是匹配的.
所以应该是 P1n乘以 1减去 P1n.
这就是雅可比矩阵的样子.
让我们看看能否通过模式识别找到更好的方法
来生成这个雅可比矩阵.
让我们换一种方式来写.
首先， 我们可以注意到这个雅可比矩阵是对称的.
因此， 你可以看到这个元素等于这个元素.
如果你展开第三行， 你会发现它等于这个元素.
右上角的这个元素等于左下角的那个元素.
所以这个矩阵是对称的.
其次， 我们可以注意到
只有对角线上的元素是不同的.
它们有一个额外的项， 因为你可以看这里的这个元素， 让我写出来
这里的这个元素也可以写成p11减去p11乘以p11.
第二行中的第二个元素，
也就是这个矩阵的第二个对角线元素， 是p12减去p12乘以p12.
所以对角线上的这个元素实际上看起来和其他元素一样.
它们只是多了一个额外的项， 第一个对角线元素中是 P11 they just 第二个对角线元素中是 P12.
因此我们也可以说这里的这个矩阵是
所有可能的 Pij与 Pik 组合的乘积，
这些组合可以通过外积获得，
甚至可以通过一列与其转置的乘积得到.
所以， 如果你取一个列向量一一例如， 假设 P 是一个列向量
然后你计算 P乘以 P的转置
你会得到这两个向量所有可能的乘积组合，
因为这将是一个.
我可以举一个简单的例子.
所以 P11， P1， 我们称它为 P2， P3.
乘以行向量p1， p2， p3.
这将生成p1与p之间所有可能的乘积组合，
因为这将是一个3乘1的矩阵.
这是一个1乘3的矩阵， 因此 会生成一个3 乘3的矩阵， 它将等于p1乘以p1， p1乘以 p2， p1乘以 p3
等等.
等等， 依此类推.
此外， 我们可以看到在矩阵的对角线上有额外的项，
第一个对角线元素中是p1，
第二个对角线元素中是p12， 第三个对角线元素中是p13.
我实际上称它为p1.
这是错误的， 因为我应该称它为pi.
这就是为什么我不想引入i 这个索引.
所以实际上不应该是p1， 而应该是pi， pi， pi，
因为我们是在为通用的第i个pi向量进行这个操作.
让我来修正一下索引.
这是一个
好的， 所以这是 PI， Pl.
好的， 我们可以得到， 嗯， 所以我们也可以将这个雅可比矩阵写成
一个对角矩阵，
其对角线上的元素都是p 的元素?
i向量减去p向量， 再乘以其自身的转置.
所以与其自身相乘， 但需要转置.
因为我们需要所有元素都是 P 的一个元素
与另一个元素的某种组合，
而在对角线上， 我们还需要一些额外的项，
即p 的元素，
所有这些p的输出元素与p的转置相乘后都会被取负.
这就是为什么我们需要这个减号.
所 以如果你看一下 Flash Attention 的论文， 他们会在这里给你这个公式
他们说如果 Y 等于×的soft max，
那么雅可比矩阵将如下所示， 会是一个对角矩阵.
的对角矩阵减去y 乘以y 的转置， 其中y 是一个列向量. 好了， 各位
我知道这已经很长了， 所以我们先暂停一下， 现在终于要开始写代码了
首先， 让我们来验证一下 Flash Attention 反向传播的数学推导.
我们会简要地看一下.
我不会再做任何推导， 但我会解释一下.
然后我们最终转向编写代码.
那么， 我们开始吧.
终于可以看看 Flash Attention. 的反向传播 好了，
你会看到 B. 2部分， 1
现在， 我不想一步步推导整个过程， 因为那会太完长
但我想提供所有必要的工具， 养 帮助你们理解它
需要复习
个矩阵的命名
正如你所知， 在前向注意力机制中， 在前向传播过程中
我们将查询(query ) 与键(key ) 的转置相乘
这个输出的结果我们称之为 S
然后我们对这个
S 矩阵应用
Soft max函数 它就变成了
P矩阵.
与√矩阵相乘 从而得到注意力的输出
让我们看看， 例如，
按照惯例
但这个特定向量的来源实际上是输出矩阵的一行.
现在hi c这样我们就能理解如何从这里到这里了.
假设它如下所示.
我们只写一行.
实际上让我再放大一下， 我想写小一点， 这样我们就有足够的空间
所以我们做一个有一行的矩阵.
让我们称它为a1， a2， a3.
然后我们进行乘法运算.
这将是一个具有多行的矩阵， 就像这个一样，
因为我们只想研究一行的效果， 并将其与另一个矩阵相乘.
让我们称这个矩阵为 A， 它有， 比方说
n行3列.
然后我们应该有另一个矩阵 B，
它有3 列和一定数量的行，
比方说4列.
因此， 我们称第一行为， 让我再放大一点， 所以它是 B11， B12， B13， B14
然后这个应该是 B21， B22， B23， B24， 这个应该是 B31， B32， B33， B34
我知道我的符号表示不够严谨.
我应该用大写字母 A
和大写字母 B来称呼所有这些元素.
这就是你在引用矩阵的单个元素时使用的符号.
但请原谅我这一点.
这个矩阵乘法的输出将是另一个n×4的矩阵，
也就是说， 输出的每一行都会有4列.
我想用另一种方式来表达输出.
因此， 我想这样写， 仅作为一个向量， 将第一行输出作为一个向量
并想了解这个向量的每个维度是什么.
因为否则我在这里没有足够的空间来书写.
那么首先， 让我们把它写下来.
让我们称之为口.
我想写下○的第一个元素， 即输出的第一行，
但以列向量的形式呈现所以o的第b个元素将在这里
我们应该用小写字母. 表示第一个元素， 它应该是一个向量，
其中第一个维度是这里这些内容的点积.
即 A矩阵的第一行与 B矩阵的第一列的点积.
那么第一个维度就是 A1与 B11的点积.
我也应该把这个称为 A11， 实际上是 A12.
我也应该把这个标为 X1hy实除 P是atedly
以及 A13， 即 A13.
由于 A 矩阵有多行， 让我使用正确的命名方式
因此， 这将是 A11与 B11的乘积， A11乘以 B11加上 A12乘以 B21
再加kr Al3r乘以5 B
这将是输出短阵第v行的第rep维度tedlysun
输出矩阵○第一行的第二个维度将是
矩阵的这一行与 B矩阵第二列的点积.
让我在这里写上 B.
因此， 它将是 A11乘以 B12加上 A12乘以 B22再加上 A13乘以 B32.
第三个维度将是 A11乘以 B13加上 A12乘以 B23再加上 A13乘以 B33
第四个维度将是 A11乘以 B14加上 A12乘以 B24
再加上 A13乘以 B34.
现在， 这是输出矩阵○的第一行输出，
称为向量01， 其中:这是该向量的第一个维度，
这是第二个， 这是第三个， 而这是第四个维度，
这里的每一项都是一个标量.
因此， 输出o1， 即输出矩阵的第一行，
也可以表示为第一个元素.
如你所见0它是多个向量的和， 其中第一个元素是 A11乘以.
让我用一个更小的， 这个， 但我想用一个更小的.
我无法改变这里的大小.
好的， 没关系.
所以如你所见， 这里 A1每次乘以一个不同的 B数.
所以这是 B11， B12， B13， B14.
这是 B矩阵的第一行.
所以它等于 B1和第一行的所有维度.
这是 B矩阵的第二行.
即b2及其所有维度.
及其所有维度.
这个式子也可以写成对所有可能的
从 1到3的求和，
其中 1到3表示 A矩阵中有多少列， 即 Aij， 嗯， A1.
让我们把这个j称为j， 抱歉.
让我们称客湘
每个都莱以br矩阵的相应行，
所以我们可以写成a ii 乘以b， 其中bj是b的行.
我们也可以这样写， 以表明这是一个向量.
这正是他们在这里所做的.

当我们进行乘法运算， 即 P 乘以 V时， 输出矩阵的第i行，
我们称之为 它是一个向量， 但在表示上是 个列向量
这只是表示方式， 各位 等于 P的第i行.
因此，
乘以. V 矩阵的所有列
的第i 行所有元素的和
即左边矩阵的第1行的所有元素
其中v Vi中的第j(个矩阵是 V矩阵的每一行.
是soft max 的输出
soft max 的输出是输入完素的指数函数.
是查询与键的转置相乘的结巢?
齿此， 它是个询与个键的点
这就是为什么崔指数菌数中会有这些内容.
所以这是理解这个推导过程的第一步.
我们还学习
在矩阵乘法中 让我们回顾一
假设给定一个矩阵乘法， 即y等于×乘以w， 我们知道，
给定损失函数相对于y(即该操作的输出)的梯度，
我们可以
推导出损失相对于
该函数的一个输入
(x或w)的梯度.
为了得到相对于×的梯度， 我们需要取上游梯度
(即相对于输出的梯度)乘以 W的转置.
为了得到相对于 W的梯度， 我们需要将输入的转置
乘以上游梯度.
这个公式我们没有推导，
但它们的推导过程完全相同.
在注意力机制中，
我们最后的乘积操作是adr等手p乘以yuted Li， d
小在反向传播过程中
损失相对于输出的梯度
来推导出损失相对于
这样它就可以在反向传播过程中
被计算图中的前序操作符使用
好的， 但为了得到相对于
key 和value 的梯度
我们需要先推导出每个中间操作的梯度
因此， 在已知损失相对于◎的梯度的情况
损失相对于√的梯度的计算方式，
与矩阵乘法中损失相对于×的梯度的计算方式
与矩阵乘法中损失相对于axr的梯度的计算方式d Li， d
完金相圆
我们知道官等于
我们知道官等于
所以， 各位e这景是类比
大家应该能理解这里的类比关系
也就是左侧矩阵的转置乘以t 游梯度
在论文中 T l他们是这样写的d Q and d K are
因此， dv等于pt乘以dogr这就是你在这里看到的公式
另一个推导是如何求出相对于dp 的梯度.
左侧矩阵的梯度.
因此， 这就像在参考公式中推导损失相对于×的梯度
一样.
它等于上游梯度乘以另一个矩阵的转置，
在论文的符号表示中， 他们将其写为dp， 等于d?
○乘以v的转置.
就是这个公式.
并将其写为pij乘以do.
如何得出这个公式呢?
好的， 我们开始推导吧.
让我写下来.
好的， 2理论上我们可以人这个推导中得知ndso:
首先我们简化一下操作:"每次着到转置符号
如果你不喜欢在矩阵乘法中处理转置，
不妨给它换个名字， 用新名字进行推导，
等公式得出后， 再把转置操作代回去.
在这个例子中， 我们计算的是:dv等于p的转置乘以d?
我们把 P 的转置称为.
我们给它起一个之前没用过的名字.
那就叫它 F 吧.
有空的时候， 我总是用 F.
于是我们称之为 DV.
等于f 和 do 的乘积.
从上面的推导可知，
即矩阵cddv T的第ljs行，
Q 即矩阵dav的第t行
等于第gr个短阵f第j 行元素的总和
让我们来看看接下来怎么做
让我们来看看接下来怎么做
那就按i 来操作吧
那就按a来操作吧 Ov T， and so:
这是对第 H介矩阵第i行中所有d元素的求和
所以是 F的第li行第s个无素
T 乘以点积d而不是点积
这是一个标量向量， 乘以一个向量的乘法， 也就是一
让我确认下公式 是另一个矩阵的第j行.
即o的第i行， 其中
这是第i行的第个元素， a这是v矩阵的第1行， 不过我们并不需要
我们知道+并不是我们它有的矩阵! 它实除是p的转置， j
这o赚着dv等手
Sin 因为在矩阵转置中y两个索引会直换.
所以这是对 P 的所有可能的1进行求和， 不是和而是1和乘以的第个元素
这应该写若边着到的公式相同ted Li， dv j
这应该写看边看到的公式相同. Li， dv ; can
这样你就可以计算v矩阵中的一行输出so:
而我们知道 直pij其实就是soft max 的输出.
soft max 的输出是
soft max 输入的指数值，
除以与该行相关的归一化因子.
因为我们正在遍历i的行， 所以
soft归x化因子将与eoic的行高度相美an of y = softmax(x)
所以我们知道p的公式等于s的soft max.
现在p的第i行将是s的第i行的soft max，
这就是这里所写的内容.
我们从推导中得知， 关于soft max 操作的雅可比矩阵
如果我们有一个输入×， 车 输出是y，
那么y关于×的雅可比矩阵等于对角矩阵y.
它是一个由向量y的元素组成的对角矩阵， 减去y乘以y的转置
而且我们之前也看到过， 这个矩阵是对称的.
然而， 你可能不理解这里的公式， 因为我们在链式法则中看到
我们总是这样写.
我们总是写下游梯度， 比如dx的dphi，
应该等于上游梯度，
即dphi关于dy乘以dy关于dx.
这只有在你把这个矩阵放在分子约定中时才能奏效.
分子约定是生成雅可比矩阵的两种约定之一.
到目前为止， 我们一直将其写为分子约定.
如果你使用分子约定， 这是一个行向量， 这也是一个行向量.
然而， 如果你想将这里的量视为列向量
那么你需要取其转置，
或者需要在分母约定中生成雅可比矩阵.
如何得到这个公式呢?
因为这个公式基本上是将雅可比矩阵
而不是梯度上游梯度乘以雅可比矩阵.
这只是因为在这里我们将其视为列向量.
当你想要将行向量转换为列向量时，
你需要对方程两边都进行转置.
让我们实际操作一下.
我们对等式两边都应用转置.
在矩阵乘法中， 如果你对 AB 进行转置，
它会变成 B转置乘以 A转置.
因此， 转置操作会独立地应用于矩阵乘法的每个输入，
但我们会反转矩阵乘法的顺序.
如果你还记得， 矩阵乘法是不可交换的.
所以我们在这里的做法是， 我们说， 好吧， 这将是dx的dphi，
这里他们称之为dsi.
因此， 它基本上就变成了 dx上的 d phi.
如果你把这个当作列向量
那么这个列向量将等于dy 在dx 上的列向量
也就是分母布局下的雅可比矩阵， 在这种情况下
乘以dv 在dy 上的列向量， 这个也是列向量
这是一个列向量， 这也是你在这里看到的
这就是为什么雅可比矩阵位于上游梯度的左侧.
需要什么?
我知道这个推导过程中有很多内容，
但我更倾向于直接看代码

那么， 我们直接来看代码吧在编写代码的过程中"我会回头参考公式
这样我们就能将实际操作与论文中的公式对应起来.
我认为这是最好的方式
好了，

这就是 Flash " Attention 1的论文
Titon网站上代码的结构来进行. 立
所以 文样拆分: 但我简化了它
因为我的版本是简化版，
并直适用于茵巢和非因果注意力机制
首先， 如巢你看这个算法，
你会发现我们有一个外部循环遍历所有的 K和 V块，
一个内部循环遍历所有的查询块.
如你所着到的， 为了计算 DQ，

因
这可能会影响效率.
如果我们不想写人数据， 京
就需要在块之间进行某种同步 此外，
这同样效率不高.
因为可以看到每个dg的计算依赖于对 K的循环，
我们将固定第k个块， 为了计算 并遍历所有的q 块.
我们将进行另一轮迭代， 不
接着， 在这轮迭代中固定q块
这京 这个思路
Titon网站上的原始实现中借鉴的
这里称为di 的信息它是两者共享的.
因此， 我们可以预先计算它， 然后将其复用于qi 向量，
以计算qi向量和dk向量
这个di是什么呢? memory.
所以我们要做的第一件事是遍历9
和d O中的所有向量， 并计算它们的点积， 以得到这个di元素.
然后我们将使用这个d 元素， 实际上， 让我想想.
是的.
接着， 我们会利用这个di元素来更新并计算， DQ 、和 DK.
此外， 我们还将进行另外两个循环， 一个循环中固定查询
并遍历所有键r(keys)5， 另一个循环中固定键
并遍历所有查询 O(d)
现在， r我们对代码的结构有了大致的了解.
因此我们首先在这里编写这个反向传播函数.
让我确认一下， 女 好的.
好的， 还记得这个保存的张量吗?
这些都是我们在前向传播过程中保存的信息
以便计算反向传播
现在， 为了优化内存利用率， 我们采用 Flash Attention算法.
在注意为机制中我们不保存查询与键矩阵转置相乘的结果
因为这会生成一个序列乘以序列的矩阵， 体积过于庞大.
我们将其保存到 HBM(高带宽内存)中， 位于
DRAM(动态随机存取存储器)内，
随后再从 HBM 取回至本地内存.
Q， K， V，
我想提醒您， 在 Triton中， 与 CUDA相比
我们采取的做法是将数据从高带宽内存加载到共享内存
也就是 SRAM(静态随机存取存储器)中
我们在共享内存中完成所有运算操作后， 再调用存储方法
将结果从共享内存保存至高带宽内存.
为了避免完整地生成这个 S矩阵
Q， K， V，
将其保存到 HBM后再重新取回一一这一过程可能非常耗时.
其次， 实际上，
这样做成本很高， 因为当前我们通常
是在成干上万的token上计算注意力.
试想一下， 保存一个5000乘以5000大小的矩阵，
对于每个批次、每个注意力头来说， 保存如此庞大的矩阵都是不小的负担
因此，:保存这样的矩阵确实会消耗过多资源
Flash Attention 的核心思路在于， 在反向传播过程中实时重新计算
那些可以即时得出的结果， 因为无论如何
如果我们选择加载这些数据， 都会受到内存 I/0的限制.
因此， 相比从内存中保存和恢复数据， 即时重新计算反而更加高效.
这便是 Flash Attention 的设计理念
好的，:我们在前向传播过程中保存了一些数据
现在在反向传播时可以访问这些数据
这些数据被存储在上下文中
就像是由 Py Torch提供的一个字典一样.
没错， 这样我们就能取回查询(query)、键(key)和 和值(value)这些数据
众所周知
在自动求导过程中
Py Torch 会直接提供损失函数相对于我们实现的注意力机制输出结果的梯度.
这就是 Triton实现的注意力机制.
接下来， 我们需要仅利用
损失函数相对于输出的梯度， 来计算查询(dq)、键(dk)和值(dv)白 的
我们还需要进行一些验证检查
那么在这里
我明白这段代码还有优化空间， 比如
以检查一下这里使用的步幅 富(stride)， 这样能让代码变得更简洁高效
实际上， 在代码内部， 我总是假设步幅是相同的
不过这一点并不影响整体功能.
我只是从 Triton 的代码中提取出来， 并尝试将其简化.
我的自标是简化代码， 而不是优化它.
好的我们创建了一些向量和张量
用来存储反向传播的结果， 也就是dgq、dk和 dv.
正如我们从梯度定义中所了解的
梯度向量的输出大小
与计算梯度的向量大小相同
因为分子总是一个标量
而我们要对输入向量中的所有元素计算梯度.
因此 梯度本身的输出是一个与计算梯度的元素大小相同的
向量
所以我们得到了一些关于批量大小的信息.
 等等， 等等， 等等.
稍后我们会了解这个warp数量和stage 数量是什么意思.
我现在先不解释这个，
Torch中的warp数量
决定了我们希望在网格中启动多少线程
数量实际上指的是软件流水线中使用的阶段数.
稍后讨论自动调优时， 我们会了解什么是软件流水线，
然后我们定义了一些块.
在原始代码中， 我想它们被称为 KV1、 KV2、(
我觉得这有点让人困惑
我称之为宏块和微块， 因为
我们将固定的部分和送代的部分分别对应查询的不同处理方式.
因此， 我们固定查询块， 遍历所有键
然后固定键和值块， 再重新遍历查询.
我们遍历的是微块， 固定的是宏块.
这是我
， 我使用的命名方式.
我们需要预先计算之前在论文中看到的 DI元素.
这就是我们要启动的第一个内核.
这个内核会有自己的启动网格
因为之后我们想要优化这个内核的调优.
稍后我们会讨论针对其自身参数的调优.
让我想想.
我们接下来要做什么呢?
我们要启动的第一个内核就是这个预处理内核.
这个预处理内核会预先计算我们需要计算的所有di 元素，
我记得 dk 和 dv.
如果考虑dq和dk， 这个di元素仅依赖于o和do
那么我们就来实现它， 并创建另一个名为backward preprocessor 的函数
预处理网格的流程是什么?
这是该函数或内核的启动网格
它将针对每个批次和每个头独立启动
它将处理向量0的块大小
向量的数量是多少?
它将是块大小宏， 因此是128个0向量
那么， 让我复制这个函数的签名.
就在这里.
那么， 我们在这里写下来吧.
我觉得这样就可以了.
这个函数接收矩阵0作为输入.
因此它是指向矩阵 O的指针.
它是指向矩阵 DO的指针，-同时也是一个指向矩阵 D的指针
我们将在这个矩阵-D中存储这些 DI元素.
而且， 我们在输出中为每个向量都准备了一个.
这就是为什么这个 D的形状是批量大小、头数、序列长度.
这意味着， 在注意力机制的输出中， 每个输出元素都对应一个这样的 这个 DI实际上位于哪里呢?
不是这个， 而是那个， 对， 就像 M 一样
因此， 它的形状与 M相同， 正如你所见， 它的大小就是这样的
所以 它的维度是批量大小、头数和序列长度.
如果你还记得， 是我们在前向传播过程中保存的矩阵
它包含子softmax的归一化因子以及最大元素
但采用的是log-sum-exp 形式.
因此， 当我们应用它时，"它会自动为每一行应用最大元素
并同时进行归一化， 这一点我之前应该已经证明过了.
那么， 让我来操作一下
于是， 我们这样写
于是， 我们提取出
这个程序的索引
因此这个程序有两个标识符， 类似于索引.
这相当于 CUDA的标识符， 并且它是沿着第零轴的
那么， 让我们看看我们在第零轴上启动了什么.
在这个启动网格的第零轴上
我们定义了该特定程序将处理的向量块，
而第二轴则决定了该程序将处理哪个批次
以及批次中的哪个注意力头.
因此这标识了 Q 的 
即该特定程序将处理 O矩阵中的哪一组向量
这里之所以称之为 Q
是因为我直接从原始代码中复制过来的， 他们称之为 Q.
不过， 我其实也可以把它称为0.
简而言之， 这意味着我们正针对这个程序进行定义.
我们需要跳过一些查询向量， 这些向量要么已经被
其他并行程序处理过， 要么即将被处理.
因此， 我们只会处理 O中具有以下索引的查询
向量块
假设查询块的大小是128， 这是我们之前的定义方式.
但为了简化理解， 假设它是4.
所以这个值将会是.
那么查询向量的数量是多少呢?
序列长度即查询向量的数量， 我们可以想象这些查询向量是.
总的来说， 真体数量我也不确定.
假设总共有64个， 其中32个由其他程序处理
那么这一特定批次的索引将会是33、34、35和36.
这表示在输出矩阵0的所有向量中
当前程序将处理哪些查询向量或具体哪些向量
接下来我们还要提取批次的索引， 这告诉我们当前程序将处理哪个批次
以及每个批次中的哪个注意力头
这正是我们启动网格的第一个维度.
接着， 我们定义维度的偏移量
因为需要加载每个向量的所有维度.
因此， 这是一个向量， 指示我们需要从每个向量加载哪些维度
而我们将加载所有这些维度，
因此， 我们不在注意力头维度上进行划分， 而是仅在序列长度维度上划分
由多个程序共同分担加载任务.
在这部分视频中你会看到， 当我们编写反向传播时
不会像前向传播那样使用
所以这里的这个函数.
我们将直接通过使用步幅来进行索引操作.
那么， 我们开始吧
现在， 我们来加载○的一个行块.
的形状与 Q 相同
HE AD_ DIM_ O，
正因如此， 我们可以称这个块的大小为 Q块大小
我们正在加载的 O块就是 O本身
这里的加载函数接收一个指向待加载内容的指针.
实际上， 它接收的不是单一指针， 而是一个指针数组
或者说是多维指针数组， 以便于加载多维数据.
实际上，
Toad 函数也支持加载二维数据
在此例中， 我们将加载二维数据， 即 O的一个行块.
这个行块是一个张量， 其形状为块大小 Q which should be a 与头维度相乘的结果.
但我们需要告诉函数在◎矩阵中具体哪个位置找到这个行块.
首先， 我们需要根据其他程序将要处理的批次和头数
跳过相应的部分，
因此依据本程序要处理的批次和头索引
我们需要略过所有其他的批次和头.
我们来写出这个张量的形状.
因此， 张量的形状为:批次大小、头数、序列长度
以及头维度.
每个块和每个头将包含序列长度乘以头维度数量的元素.
那么根据我们的索引， 需要跳过多少元素呢?
我们的索引值乘以头维度再乘以序列长度.
具体来说
批次0和头0将包含序列长度
乘以头维度数量的元素.
批次0 的头 2同样如此.
那么， 我们需要从○张量的起始位置跳过多少元素呢?
答案是序列长度乘以头维度.
这等于当前批次和头指示器的索引值
由于这个索引同时标识了批次中的头
和每个批次内部的头
因为它已经是头和批次的乘积
所以根据这个索引， 我们需要跳过多少元素呢?
当我们定位到当前批次和当前头的起始位置后
需要选择一个二维张量，
其中行的偏移量由offsq指示
这就是为什么会有这个一一手 我不确定该怎么称呼它.
这是索引l， 分号索引， 它指示了ofsg中的所有向量
并额外增加了一个列维度， 这些列将由of sd im 表示.
with an additional dimension for the columns， and these columns will be the ofs dim.
> TIMELINE OUTLINE ef forward (ctx， Q， K， V， causal， soft max _scale ):
index _batch _head * HEAD _ DIM * SEQ _ LEN 简而言之， 这将选择一个具有以下形状的张量:
so basically， this will'select a tensor of the following shape :
> TIMELINE OUTLINE def forward (ctx， Q， K， V， causal， soft max _scale ):
在这个包含批次大小和头数量的大张量内部
inside of this big tensor that includes pet size and number of heads > TIMELINE OUTLINE def forward (ctx， Q， K， V， causal， soft max _scale ):
+index _batch _head * HEAD _ DI H* SEQ_ LEN 这就是我们正在做的操作.
Triton Attention (torch. au This is what we are doing.
> TIMELINE OUTLINE def forward (ctx， Q， K， V， causal， soft max _scale ):
@static method
+index _batch _head * HE AD_ DIH * SEQ_ LEN
+offs _q[:， None]* HEAD_ DIM +offs_dim[ None，:] 也就是说
( BLOCK _ SIZE Q， HEAD_ DIH)
8
Triton Attention (torch. auto grad. Functi or So we are saying :
> TIMELINE OUTLINE 267 @static method def forward (ctx， Q， K， V， causal， soft max _scale ):
259
index_batch_head * HEAD_ DIH * SEQ _ LEN 我们要在一个由四个维度组成的张量中， 选择这样一个大小的张量
select a tensor of this size inside of one that is made up of four dimensions，
> TIMELINE OUTLINE def forward (ctx， Q， K， V， causal， soft max _scale ):
Index _batch _head * HEAD _ DI H* SEQ_ LEN 跳过其他程序将处理的所有批次和头的元素.
skipping the elements of all the batch and heads that will be processed by other programs.
> TIMELINE OUTLINE def forward (ctx， Q， K， V， causal， soft max _scale ):
59
index_batch_head * HEAD_ DIH * SEQ _ LEN 我总是用"程序"来表述， 因为在 Triton 中它们被称为程序
I always talk in terms of programs because in Triton these are called programs，
> TIMELINE OUTLINE def forward (ctx， Q， K， V， causal， soft max _scale ):
+index _batch _head * HEAD _ DIH * SEQ_ LEN 而在 CUDA中， 你会称它们为内核
in CUDAyou would refer to them as kernels.
> TIMELINE OUTLINE @static metho
def forward (ctx， Q， K， V， causal， soft max _scale ):
+1ndex_batch_head * HEAD _p IH* SEQ_ LEN 没错， 所以这一部分已经完成了，
8
Triton Attention (torch. auto gra right， so this one is done.
> TIMELINE OUTLINE def forward (ctx， Q， K， V， causal， soft max _scale ):
@static method
+off s_ql:，
+off s_di n[ N 希望我已经解释得足够清楚了.
ihope it is decently clear.
> TIMELINE OUTLINE @static method def forward (ctx， Q， K， V， causal， soft max _scale ):
6
Lndex_batch_head * HEAD _ DIH* SEQ_ LEN 嗯 好的， 那么我们也加载一个单独的d块吗?
264
265
um， all right， so then we also load a single block of d?
> TIMELINE OUTLINE 266
6
def forward(ctx， Q， K， V， causal， soft max _scale ):
+index _batch _head * HEAD _ DIH* SEQ_ LEN
+offs_q[:， None]* HEAD_ DIH
+offs_dim [ No ne，:] 哦， 同样的方式
o in the same way，
> TIMELINE OUTLINE def forward (ctx， Q， K， V， causal， soft max _scale ):
@static method
index _batch _head * HEAD _ DIM * SEQ _ LEN 因为我们要从整个序列长度中加载一组向量， 也包括d吗?
because we are going to load a group of vectors from all the sequence length， also from d?
> TIMELINE OUTLINE def forward (ctx， Q， K， V， causal， soft max _scale ):
+index _batch _head * HEAD _ DIM * SEQ _ LEN
270
). to(tl. float32] 哦， 还有d呢?
271
272
class Triton Attention (torch. auto grad.
O， and the d?
> TIMELINE OUTLINE 274
275
@static method def forward (ctx， Q， K， V， causal， soft max _scale ):
index _batch _head * HEAD _ DIM * SEQ_ LEN 的形状与. 相同， 而. 的形状又与q相同
271
272
ohas the same shape as o， which has the same shape as q，
> TIMELINE OUTLINE 273
274
75
def forward(ctx， Q， K， V， causal， soft max _scale ):
index _batch _head * HEAD _ DIM * SEQ_ LEN 这就是为什么我们可以使用threadldx.
Trito
and that's why we can use the The block index.
> TIMELINE OUTLINE def forward (ctx， Q， K， V， causal， soft max _scale ):
@static m
index _batch _head * HEAD _ DIM * SEQ_ LEN 我们称之为 Q， 因为它们是等价的， 具有相同的形状.
we call it Q， because it's equivalent， because they have the same shape.
> TIMELINE OUTLINE forward (ctx， Q， K， V， causal， soft max _scale ):
+index batch _head *
HEAD _ DIH * SEQ _ LEN 好的， 那么如何计算这个di 元素呢
Okay， and how to compute this di element.
> TIMELINE OUTLINE 275 @static method def forward (ctx， Q， K， V， causal， soft max _scale ):
+index _batch _head * HEAD _ DIH * SEQ_ LEN
+offs_q[:， None]
+offs_din [ Non
* HEAD_ DIH
278
. to(tl. float32) 嗯， 这在论文中有详细说明
OUTLINE 275
274
@static method Well， it's written in the paper.
> TIMELINE def forward (ctx， Q， K， V， causal， soft max _scale ):
所以如果我们深入探讨这个问题， 具体是什么呢?
274
273
class Triton Attention So if we go in the inthe What is it man?
> TIMELINE OUTLINE 275 @static method def forward (ctx， Q， K， V， causal， soft max _scale ):
21 :
Write d Qd Q+rd SKRB， xd to HBM.
22:
On chip. compute d K
d K+d SQ∈ RBxd
23:
end for
24:
Write d K←d K， d V
26: Return d Q. d K. d V
25:end for
if we go here.
21 :
Write d Qd Q+rd SKRB， xd to HBM.
22: 示了如何根据给定的do块和. 块来计算每个di.
it shows you'how. to. compute the di of each given a block of do and a block of o.
21 :
Write d Qd Q+d SKRB， xd to HBM.
22:
On chip. compute dkdk +rd SQe RB. xd
23:
end for
25:end for
24:
Writed K←d K， d V;d V;to HBMI.
26: Return d Q. d K. d V
21 :
Write d Qd Q+d SKRB， xd to HBM.
22: 也就是行求和， 就是 即按行进行累加
21:
Write d Qd Q+rd SKRB， xd to HBM.
22:
23:
Writedkdk付
end for 24:
25:end for
21 :
Write d Qd Q+rd SKRB， xd to HBM.
22
On chip. compute d K 对于○知 矩阵中的每一 个向量 我们都会得到一个逐元素乘积的和
21:
Write d Qd Q+rd SK RB， xd to HBM.
2:
Onchip. compute d Kd K+rd SQ∈ RBxd
21 :
Write d Qd Q+d SKRBxd to HBM.
22:
On chip， compute dkdk +rd SQe B. xd
23:
end for
25:end for
24:
Writed Kd K， d Vd V;to HBM.
26: Return d Q. d K. d V
21 :
Write d Qd Q+rd SKRB， xd to HBM.
22:
On chip. compute dk 是矩阵乘法， 而是逐元素相乘
23:
endf好
24:
21:
Write d Qd Q+rd SKRBxd to HBM.
22:
On chip. compute dkdk +rd SQe RB. xd
23:
end for
25:end for
24:
Writed Kd K， d Vd V;to HBM.
26: Return d Q. d K. d V
21 :
Write d Qd Q+rd SKRBxd to HBM.
22: 一个矩阵的每个元素
23:
endfor 24:
Writed Kd K. d V
25:end for
21 :
Write d Qd Q+rd SKRBxd to HBM.
22:
On chip. compute dk
23:
endfor 二个矩阵的对应元素相乘.
24:
Writed K
25:endf
21 :
Write d Qd Q+d SKRB， xd to HBM.
22:
On chip. compute dkdk +rd SQe B. xd
23:
end for
25:end for
24:
Writed Kd Kd V d Vto HBMI.
26: Return d Q. d K. d V
21:
Write d Qd Q+d SKRB， xd to HBM.
22:
23:
endfor
24:
Writed K
21 :
Write d Qd Q+rd SKRB， xd to HBM.
22:
On chip，. compute dkdk+rd SQB. xd
23:
end for
25:end for
24:
Writed K;d K， d V;d V;to HBMI.
26: Return d Q. d K. d V
21:
Write d Qd Q+rd SKRB， xd to HBM.
22:
23:
end for 24:
Write d K
25:end for
index _batch _head * HEAD _ DIM * SEQ _ LEN +off s_q:，
* HEAD _ DIM
). to(tl. f 好的， 接下来我们计算这个 DI块.
OUTLINE
275
274
@static method Okay， so we compute this Dl block.
> TIMELINE def forward (ctx， Q， K， V， causal， soft max _scale ):
267
+index_batch_head * HEAD_ DIM * SEQ _ LEN 它的形状将是 Q 的块大小， 因为每个向量都会对应一个求和结果.
which will have shape block size Q， because we will have one sum for each vector.
> TIMELINE OUTLINE def forward (ctx， Q， K， V， causal， soft max _scale ):
+of fs_dim[
. to(tl. float32) 接下来我们需要将结果存储在某个位置
8
class Triton Atte Then， well， we need to store it somewhere，
> TIMELINE OUTLINE def forward (ctx， Q， K， V， causal， soft max _scale ):
@static method
+of fs_dim[
. to(tl. float32)
272
273 因此需要计算它在 D矩阵中的具体存储位置.
8
so we need to calculate where to store it inside of the D matrix.
> TIMELINE OUTLINE def forward (ctx， Q， K， V， causal， soft max _scale ):
+of fs_dim[
). to(tl. float32)
_block 我记得 D矩阵的形状与 M相同
Well， t
the D matrix is， Tremember correctly， has the same shape as M > TIMELINE OUTLINE def forward (ctx， Q， K， V， causal， soft max _scale ):
DIM * SEQ _ LEN 因此它的大小应该与批次大小一致.
Dbtock=tlsudobtock*obu So it should be batch size.
> TIMELINE OUTLINE ass Triton Attention (torch. auto grad. Function ):
+of fs_q:，
DIH* SEQ_ LEN
). to(tl. float32)
+off s_din 包含头数和序列长度两个维度.
ek t. anumber of. heads and a sequence length.
> TIMELINE OUTLINE lass Triton Attention (torch. auto grad. Function ):
HEAD _ DIH* SEQ_ LEN 因此我们需要根据当前的threadldx立方体
273
272
bc=tdo so. we. need to. select. the right batch > TIMELINE OUTLINE rad. Function ):
* SEQ _ LEN +off s_ql:，
+offs_dim 在序列长度中选择正确的批次，
270
. to(tl. float32)
and the right. head and also. the right position inside of the sequence length 271 > TIMELINE OUTLINE Triton Attention (torch. auto grad. Function ):
+off s_q:， None DIM* SEQ_ LEN
). to(tl. float32)
+offs_dim[ N 正确的头以及正确的位置.
based on. the-block index cube that we have.
OUTLINE
D_block > TIMELINE lass Triton Attention (torch. auto grad. Function ):
D _block class Triton Attention (to r 好的接下来让我来索引.
@static method > TIMELINE OUTLINE HEAD _ DIM _ V= V. shape[-1]
HEAD_ DIM_ Q， HEAD_ DIM_ K
271
272
Compute the D block 好的， 没
(d0 block*0_block， axis=1) 问题， 既然我们已经完成子这一步， 就像之前一样， 我们可以直接跳过继续
okay， All right， because we already， so we skip again， just like before.
> TIMELINE OUTLINE HEAD _ DIM_ V= V. shape[-1]
Compute the D block Store the D blo 我们知道· D·的大小是这样的
D_block _ptr s= D
class Triton Attention (torch We know that the D isof this size.
> TIMELINE OUTLINE 280
def forward(ctx， Q， K， V， causal， soft max _scale ):
@static method I-11
每个批次和每个头都会有序列长度数量的元素.
Each batch and each head will have sequence length number of elements.
> TIMELINE OUTLINE def forward (ctx， Q， K， V， causal， soft max _scale ):
pute the D block 因此， 我们需要从张量的起始位置跳过的元素数量
So how many number of elements we need to skip from the starting of the tensor > TIMELINE OUTLINE def forward (ctx， Q， K， V， causal， soft max _scale ):
274 是序列长度乘以批次大小与头编号的组合索引.
issequence length multiplied by the combined index batch size head number.
> TIMELINE OUTLINE
272
271 此外;我们还需要根据threadldx队列跳过一些查询
And plus， we. need to also skip some queries based on our block index queue，
> TIMELINE OUTLINE def forward (ctx， Q， K， V， causal， soft max _scale ):
272
271 个跳过操作已经在偏移队列中完成了， 月 所以只需加上偏移队列的值即可
and this skipping is already done inside of off queue， so we add off queue.
> TIMELINE OUTLINE _scale ):
272 然后， 当我们计算出应该存储这个 DI块的索引位置时
8
And then， once we have computed the index where we should store this Dl block，
> TIMELINE OUTLINE
def fonwardctx， Q， K， V， causal， soft (max _scale ):
Store the D bloch 我为什么要称它为 D块呢?
D_block_ptrs= D+
class Triton Attention (torch. auto grad. Functi why did I even call it D block?
> TIMELINE OUTLINE 280
def forward(ctx， Q. K， V， causal， soft max scale ):
@static method
Compute the D block D_block=tl. sum(do_block *0_block， axis=1)# Sha Store the ( BLOCK _ SIZE _ Q，
D_block _ptrs 让我们把它存储起来， 月 所以让我
Triton Attention (torch. auto grad.
Let's store it， so let me..
> TIMELINE OUTLINE def forward (ctx， Q， K， V， causal， soft max _scale ):
@static method
Compute the D block
273
_block=tl. s 我并没有称它为 D块， 我想这是原代码中已有的命名， 但这里指的是 DI I didn't call it D block， I think it was already in the original code， but this is Dl.
> TIMELINE OUTLINE def forward (ctx， Q， K， V， causal， soft max _scale ):
Compute the D block Store (do_block*0_block， axis=1)# Shape :( BLOCK _ SIZE _ Q，)
_block 这个大矩阵 D*实际上包含了所有 DI，
And this big matrix Dis actually the matrix that includes all the Dl.
> TIMELINE OUTLINE ef forward (ctx， Q， K， V， causal， soft max _scale ):
D_block
block*0_block， axis=1) Shape :( BLOCK _ SIZE _ Q，) 每个 Dl 对应序列长度中的一个token.
one for each'token in the sequence length.
> TIMELINE OUTLINE @static meth def forward (ctx， Q， K， V， causal， soft max _scale ):
Compute the block D _block =tl. s
Store the bloc
m(do_block *0_block， axis=1)# Shape :( BLOCK _ SIZE _0，)
tl. store( D_bloc
_block ptrs= 好了， 预处理工作已经完成
lass Tri to All right， so the pre-processing has been done.
> TIMELINE OUTLINE @static met def forward (ctx， Q， K， V， causal， soft max _scale ):
Compute the block # Store the
D _block =tl
(do_block*0_block， axis=1)# Shape :( BLOCK _ SIZE _ Q，)
_block ptr 现在我们需要准备两个for 循环.
ass Triton A Now we need to prepare the two for loops.
> TIMELINE OUTLINE @static methoc
def forward(ctx， Q， K， V， causal， soft max _scale ):
_block 正如我之前提到的， 我们将执行两个for 循环:
As you remember I said before， we will be doing two for loops :
> TIMELINE OUTLINE forward (ctx， Q， K， V， causal， soft max _scale ):
_block lock*0_block， axis=1) Shap 个循环中， 我们固定查询(guery)并遍历所有键(key )和值(value ):
one in which we fix the query and we iterate through all the keys and values，
> TIMELINE OUTLINE def forward (ctx， Q， K， V， causal， soft max _scale ):
273
D_block =
Compute the block *0_block，
axis=1)
( BLOCK_ SIZE_0， 第二个循环中;我们固定键值块(key-value block )并遍历所有查询.
and one in which we fix the key and value block and we iterate through all the queries > TIMELINE OUTLINE def forward (ctx， Q， K， V， causal， soft max _scale ):
273
block
( BLOCK_ SIZE_0， 在编写代码的过程中， 我会一直展示论文中的公式， 所以不用担心.
And while coding it， I will always show you the formula from the paper， so don't worry.
> TIMELINE OUTLINE ef forward (ctx， Q， K， V， causal， soft max _scale ):
d ef
test_op( BATCH _ SIZE， NUM _ HEADS， SEQ_ LEN， HEAD_ DIM， causal， dtype=torch. float16):
torch. enpty(
( BATCH_ SIZE， N
normal_(mean =0. 0， 让我们开始下一次迭代吧.
. requires_grad_()
Let'sstartwiththenextiteration.
> TIMELINE
OUTLINE
torc
( BATCH SIZE. NUM HEADS. SEO LEN. HEAD DIM). dtve=dtvoe. device="cuda
npty (
test _op ( BATCH _ SIZE， NUM _ HEADS， SEQ _ LEN， HEAD _ DIM， causal， dtype=torch. float16): 首先 我们为下一次送代创建启动网格.
So first we create the launch grid for the next iteration.
> TIMELINE OUTLINE 380
BATCH SIZE. NUM HEADS. SEO LEN. HEAD DIM). dtv De=dtv De. device =cuda
d ef
test_op( BATCH _ SIZE， NUM _ HEADS， SEQ_ LEN， HEAD_ DIH， causal， dtype=torch. f Loat16):
torch. enpty(
( BATCH_ SIZE， NUM_ HE
normal_(mean =0. 0， 启动网格始终保持不变
. requires_grad_()
The launch grid is always the same，
> TIMELINE
OUTLINE
torcl
( BATCH SIZE. NUM HEADS. SEO LEN. HEAD DIM). dtv De=dtv De. device="cuda
npty (
因为我们需要固定一个块， 同时遍历所有其他块.
because we need to keep one block fixed and iterate through all the other blocks.
> TIMELINE OUTLINE
op ( BATCH _ SIZE， NUM _ HEADS， SEQ_ LEN， HEAD DIM， Cau Sal， dty P 我们固定的块将决定并行运行的程序数量
The block that we keep fixed will define how many programs we have that run in parallel > TIMELINE OUTLINE
7 而固定的块包含一个由宏定义的块大小数量的元素.
and the block that is fixed has a block size macro number of elements.
> TIMELINE OUTLINE
373
def test_op( BATCH_ SIZE， NUM_ HEADS， SEQ _ LEN， HEAD _ DIM， causal， dtype
torch. float16): 这就是为什么我们创建了序列长度除以块大小宏的块数
That's why we create a sequence length divided by block size macro number of blocks.
> TIMELINE OUTLINE
de f
test_op( BATCH _ SIZE， NUM _ HEADS， SEO _ LEN， HEAD _ DIH， causal， d type =torch. float16):
=(
torch. enpty( 在这一轴上创建线程块或程序.
BATCH_ S
> TIMELINE OUTLINE requires _grat
372
73
torch. float16): 在这个网格中， 轴2一一我也可以同样随意地使用轴1.
the axis2inthisgridis-icould have used also the axis1， indifferently > TIMELINE OUTLINE
我认
73
def test_op( BATCH_ SIZE， NUM_ HEADS， SEQ_ LEN， HEAD_ DIM，
loat16): 为在原始代码中已经完成了这一部分 它将指示我们将处理哪个批次
i think it Was already done here in the original code-it will indicate which batch 福
> TIMELINE OUTLINE
de f test_op( BATCH _ SIZE， NUM _ HEADS， SEQ _ LEN， HEAD _ DIH， causal， d type =torch. float16):
=(
torch. enpty( 以及每个批次中的哪个头.
and which head inside of each batch we are going to work with.
377
BATCH_ SIZE，
> TIMELINE OUTLINE
373
def test_op( BATCH_ SIZE， NUM_ HEADS， SEQ _ LEN， HEAD _ DIM， causal， dtype=
=torch. float16): 因此， 与正向传播类似， 我们也将使用一个名为stage的变量
so， and just like the forward pass， we will also use a variable called stage，
> TIMELINE OUTLINE
373 如果我们要计算的是因果注意力，! 则其值为三:
that if the attention that we are computing is causal， it will be equal to three，
> TIMELINE OUTLINE
如果是非因果注意力则其值为一.
77 and if we are computing a non-causal attention， then it will be equal to one.
> TIMELINE OUTLINE
在第一次迭代中我们将固定 K和 V块
( The first iteration :we'will fix K and V blocks > TIMELINE OUTLINE normal _(mean=0. 0， std=0. 5)
reauires arad(
375 并遍历所有块:这些 Q块的大小为块大小
and we witliterate through "all'the Q blocks in size of block size，
> TIMELINE OUTLINE
test _op ( BATCH _ SIZE， NUM _ HE 即查询向量的微观数量
torch. en pty (
( BATCH _ SIZE， NUM _ H micro number of query vectors.
> TIMELINE OUTLINE normal _(mean=0. 6， std=0. 5)
. reauires arad ()
HEAD _ DIH=ctx. HEAD_ DIH，
BLOCK _ KV= BLOCK_ SIZE _ MACRO STAGE =stage 那么， 让我们来看一下这个函数的签名.
So let'slook at the signature.
> TIMELINE OUTLINE def
test_op BATCH _ SIZE， NUM _ HEADS， SEQ _ LEN，
HEAD _ DIM =ctx. HEAD _ DIM，
STAGE =stage， 因此， 我们以网格启动的方式调用它
So we pass， we launch it as a launch grid because -
> TIMELINE OUTLINE
HEAD _ DIM=ctx. HEAD_ DIM，
393
STAGE =stage 因为我们已经定义了程序的数量， 所以我们也知道了会有多少个k V 块
and we have defined how many. programs we have， so we have how many kv blocks we will have OUT LIN TIMELINE
HEAD _ DIM =ctx. HEAD _ DIM，
STAGE =stage，
_stages = NUM 即序列长度除以宏块大小
it's the sequence land divided by the block size macro，
> TIMELINE OUTLINE 401
est_op( BATCH_ SIZE
HEAD _ DIM =ctx. HEAD _ DIH，
393
STAGE=stage， 因为在这个函数的for循环中， 我们将保持这些块固定不变，
399
because > TIMELINE OUTLINE 400
01
Q=(
HEAD _ DIM=ctx. HEAD_ DIH，
393
STAGE =stage， 因为在这个函数的for 循环中， 我们将保持这些块固定不变.
that's the the block that we will keep fixed in this uh for loop， in this function.
> TIMELINE OUTLINE
HEAD _ DIM=ctx. HEAD_ DIH，
BLOCK _ KV= BLOCK_ SIZE _ MACRO STAGE =stage 接着， 我们遍历所有大小为微块(我将其定义为32 )的查询块
and then we go through all the query blocks in size of block size micro，
> TIMELINE OUTLINE
NUM _ HEADS = NUM _ HEADS，
SEQ _ LEN= SEQ_ LEN，
BLOCK _ KV = BLOCK_ SI HEAD _ DIM =ctx. HEAD 稍后我们会讨论自动调优
num _stages = NUM _ STAGES，
num _warps WARPS，
which i defined itas32 > TIMELINE OUTLINE
stride _dim Q. stride(3)，
NUM_ HEADS = NUM_ HEADS，
BLOCK _ Q BLOCK _ SIZE _ MICF
SEQ_ LEN-SEO_ LEN， 以及如何调整这些值.
HEAD_ DIM=ctx. HEAD BLOCK _ KV-BLOCK _ SIZE and later we will talk about autotuning and how to tune these values.
> TIMELINE OUTLINE
num _warps = NUM _ WARPS，
num _stages = NUM _ STAGES 好的， 那么我传入查询向量、键向量和值向量.
All right， so T pass the query vector， the key vector and the V vector.
> TIMELINE OUTLINE ( BATCH _ SIZE， NUM _ HEADS， SEO _ LEN， HEAD _ IM)， dtype =dtype， device =cuda
_stages = NUM _ STAGES 抱歉， 不是向量， 而是张量.
Q =(
> TIMELINE OUTLINE torch. en pty (
( BATCH _ SIZE， NUM _ HEADS， SEO_ LEN， HEAD_ DIM)， dtypdtype， device =cuda
um _stages = NUM _ STAGES， 现在传入的是查询张量
de f
test_op( BATCH _ SIZE， NUM _ HEADS， SEQ _ LE
Q=(
Now the query tensor，
> TIMELINE OUTLINE torch. en pty (
( BATCH _ SIZE， NUM _ HEADS， SEO _ LEN， HEAD _ DI M)， dtype =dtype， device cuda
张量和 V张量， 它们指向张量的起始位置
K tensor and V tensor and they are pointing to the beginning of the tensor，
tride_batch= Q. stride(0)
> TIMELINE OUTLINE
stride_dim= Q. stride(3)，
attn bwd dk dv Igrid]
Q= Q， 这意味着它们从张量的第一个批次
which means that they are beginning to the first batch > TIMELINE OUTLINE 382
d V=d V，
attn_bwd dk_dv Igrid]
Q= Q 个头第一个标记以及第一个维度开始.
and the first head and the first token and the first dimension of the tensors.
> TIMELINE OUTLINE
NUM _ HEADS = NUM _ HEADS，
SEQ _ LEN= SEQ_ LEN， 接着我们传入soft max 的缩放因子
Then we pass the soft max scale.
> TIMELINE OUTLINE
SEQ_ LEN= SEQ_ LEN，
BLOCK 我们传入了 O、 DQ、 DK 和 DV.
_warps=
Wepassthe O， DQ， DK， and DV.
> TIMELINE OUTLINE
SEQ _ LEN= SEQ_ LEN，
BLOCK _ Q = BLOCK_ SIZ M是计算过程中所需的关键项
Mis the one that is needed to compute.
> TIMELINE OUTLINE
389
SEQ_ LEN= SEQ_ LEN， 正如之前提到的， 我们没有将 P矩阵存储在 HBM中
BLOCK _ Q = BLOCK as you remember， from what we said before， we did not save the P matrix in the HBM > TIMELINE OUTLINE
389
SEQ_ LEN= SEQ_ LEN， 因为我们的目标是在反向传播过程中动态地重新计算它.
because we want to recompute it on the fly during the backward pass.
> TIMELINE OUTLINE
389
SEQ_ LEN= SEQ_ LEN， 因此 查询与键的转置相乘会生成一个非常大的矩阵
BLOCK _ Q= E
394
Sothe-query multiply by transpose of the keys，
OUTLINE 396
395
> TIMELINE
397
NUM _ HEADS = NUM_ HEADS，
SEQ_ LEN= SEQ_ LEN， 若将其存储在 HBM中并恢复， 会占用大量资源
it's a very-big matrix to save in the HBM and restore it，
> TIMELINE OUTLINE 396
397
SEQ _ LEN= SEQ_ LEN， 为此我们选择在需要时动态计算这个矩阵
so we want to compute it on the fly.
> TIMELINE OUTLINE
SEQ _ LEN= SEQ_ LEN，
BLOCK _ KV = BLOCK BLOCK _ Q = BLOCK_ SIZ 但无需重新计算归一化因子
MICI STAGEs stage HEAD DIM =ct X But we-don-tneed to recompute the normalization factor > TIMELINE OUTLINE 396
397
SEQ _ LEN= SEQ_ LEN，
BLOCK _ Q = 和每行的最大元素来应用soft max，
and the maximum element for each row to apply the softmax
9
> TIMELINE OUTLINE
SEQ _ LEN= SEQ_ LEN， 因为这些已在正向传播过程中计算完毕
BLOCK that was-already computed during the forward pass > TIMELINE OUTLINE
SEQ _ LEN= SEQ_ LEN，
BLOCK _ Q-BLOCK _ SIZE _ MI BLOCK _ KV = BLOCK _ SIZE 并保存到了矩阵 M 中.
STAGE =stage，
HEAD DIM = Ct X. HEAD u _warps OUTLINE um _stages = NUM _ STA and saved into this matrix M，
> TIMELINE
SEQ _ LEN= SEQ_ LEN，
M 包含了每行最大值的指数对数求和
which includes the log sum exp of the maximum of each row 94
> TIMELINE
OUTLINE
97
stride _batch = Q. stride (e)
stride _head = Q. stride (1)，
stride _seq= Q. stride(2)，
stride_dim-Q. strid
SEO_ LEN-SEO_ LEN， 以及归一化因子的对数.
plus. the logarithm of the normalization factor.
> TIMELINE OUTLINE
385
stride _batch-Q. stride(0)
stride_head = Q. stride(1)， 借助指数 对数求和技巧， 我们只需直接应用它， 便能同时实现每个值的归一化处理
Withthelogsume
exp trick. we can just apply it and it will also normalize each value.
> TIMELINE OUTLINE
BLOCK _ Q = BLOCK _ SIZE _ MICRO，
SEQ _ LEN= SEQ _ LEN 接着， 我们得到了这里计算出的 D张量， 其中包含了所有 DI 值
Then we have the D tensor that we computed here with all the Dl values，
> TIMELINE OUTLINE
每个 DI值对应 O张量中的一个向量.
one for each vector in the O tensor.
> TIMELINE OUTLINE
BLOCK _ Q = BLOCK _ SIZE _ MICRO， 接下来， 我们需要传入头的数量、序列长度
Then we need to pass the number of heads， the sequence length.
> TIMELINE OUTLINE
BLDC E_ Q= BLOCK _ SIZE _ MICRO，
BLOCK _ KV = BLOCK _ SIZE _ MACRO 以及我们想用于 KV 的块大小， 即宏块大小
the block size that we want to use for the KV， which is the macro block size，
> TIMELINE OUTLINE
BLOCK _ KV = BLOCK 而微块大小则是我们始终迭代处理的那个.
and the micro block size is always the one that weiterateon.
395
> TIMELINE OUTLINE
390
391 我认为采用这种命名方式， 应该更容易理解我们在迭代哪一个
I think using this name， it should be easier to understand which one we are iterating > TIMELINE OUTLINE
BLOCK _ Q = BLOCK_ SZE _ MICRO，
SEQ _ LEN= SEQ_ LEN，
BLOCK _ KV = BLOCK 以及我们希望保持固定的是哪一个
and which we want to keep fixed.
> TIMELINE OUTLINE
BLOCK _ Q = BLOCK_ SZE _ MICRO，
BLOCK _ KV = BLOCK 因此， 固定的是宏块， 而迭代的是微块.
So the fixed one is macro and the iterating one is the micro.
> TIMELINE OUTLINE
BLOCK _ Q = BLOCK _ SIZE _ MICRO 嗯还有头维度(head dimension )
BLOCK _ KV = BLOCK _ SIZE _ M
um， head dimension， um，
> TIMELINE OUTLINE
9 稍后我们会明白为何要使用不同的块大小进行迭代
and later we will see why we would use a different block size to iterate from，
> TIMELINE OUTLINE
BLOCK _ O= B'
(parameter )ctx: A
BLOCK_ KV=
393
394 因为这涉及到 Triton 能够利用软件流水线技术
because this is related to the number of stages that triton can divide your for loop into > TIMELINE OUTLINE
BLOCK _ Q= B'
(paraneter) ctx: 将你的for循环划分为多个阶段的原因.
BLOCK_ KV=
thanks to software pipelining.
> TIMELINE OUTLINE
BLOCK _ Q= B'然后我们还有头维度
(head dimension )
then we have head dimension.
> TIMELINE OUTLINE
391
BLOCK_ KV= E 阶段
(stage)参数指示了我们在前向传播中计算注意力时
397
9
the stage indicates if the attention > TIMELINE OUTLINE 398
BLOCK _ Q BLOCK _ SIZE _ MICRO，
SEQ _ LEN= SEQ_ LEN，
BLOCK variable )stage : Literal [3， 1 是否采用了因果性(causal )机制.
that we computed in the forward pass was causal or not causal， um，
> TIMELINE OUTLINE
BLOCK _ Q = BLOCK _ SIZE _ MICRO，
391
BLOCK _ KV= BLOCK_ SIZE _ MAC 此外 还有我们设定为固定的warp 数量和流水线阶段数.
the number of warps and the number of stages which we defined as fixed.
> TIMELINE OUTLINE
stride_seq = Q. stride (2)，
stride_dim= Q. stride(3)， 不过， 稍后我们会讨论自动调优(autotuning )白 的相关内容.
but later we will talk about auto tuning.
> TIMELINE OUTLINE 395
394
+
num_stages = NUM_ STAGES
Live Shan
HEAD _ DIM =ctx. HEAD_ DIM，
STAGE=stage，
393 所以我有时候会反复重复同样的内容， 这一点我应该改进一下.
so sometimes i repeat the same stuff over and over， so i should change that.
> TIMELINE OUTLINE
393
HEAD_ DIM=ctx. HEAD_ DIM，
STAGE =stage， 嗯， 好的， 让我们先写出这个函数的签名， 然后把它放在这里
um， okay， let's write the signature of this function and let's put it here.
> TIMELINE OUTLINE
BLOCK _ Q :tl. constexpr，
BLOCK_ KV :tl. constexpr
HEAD_ DIH:t1. c
STAGE:tl. co 我们已经描述了这个函数的签名
 让我们直接进入正题吧，
首先， 我们需要理解的是， 确定如何调整
查询(query )、 键(key )和值(value )的偏移量
扁移量首先取决于
定位到正确的批次(batch)以及每个批次中正确的注意力头(head )
我们通过将程序索引(program index )
即头索引
(head index )与批次索引(batch index )的乘积一一进行划
来计算批次的索引这与前向传播中的做法一致.
我们将其除以头的数量， 以确定当前程序正在处理哪个批次.
的头， 我们只需进行取模运算， 就像在单次遍历的for循环中所做的那样
批次和头的偏移量指示了 让我确认一下它的具体用途
好的， 它帮助我们定位到正确的批次和正确的头
那么这里的步幅(stride )是什么呢?
如果你还记得的话，
步幅(stride )告诉我们在该维度上需要跳过多少个元素
才能到达同一维度中的下一个索引.
们想要跳过批次的索引数量，? 只需将其乘以批次步幅
即到达下一个批次需要跳过的元素数量.
此外， 我们还需要定位到正确的头(head)
因此， 我们将头的索引乘以头的步幅(stride of the head )
这样， 我们就能准确地定位到 QK 和 V矩阵中对应的头的位置
此外我记得这个步幅也会用于 M和 D
因为 M和 D没有头的维度它们仅与批次大小相关
头的数量， 序列长度.
因此我们只需将批次索引乘以序列长度
因为对于每个批次和每个头， 我们都会有序列长度的数据项.
你可以将其理解为从一个批次的头移动到下一个批次的头
的步幅
接下来， 我们移动指针， 操作就是这样，

d0 += offset _batch _head 我们通过偏移量 batch head 移动指针 Q、 K 和 V
因为我们希望在大张量中定位到正确的批次和正确的头，

DK 和 DV 也进行相同的操作， 因为它们的形状与 Q、 K和 V一致
而 D O 的形状也与 Q 相同
因此，
它们的形状相同， 所以我们使用相同的偏移量来移动
好的， 接着我们移动 m和d， 将它们定位到
当前批次和当前头的序列起始点.
这样， 它们就指向了专属于当前批次
和当前头的序列的第一个向量.
同样地，
q、"k、v*以及do、dq、dk 和v也遵循相同的逻辑.
好的， 接下来我们加载一些其他数据.
因为在这个迭代中， 我们固定了kv 并通过q进行循环迭代.
因此， 我们首先需要加载kv 的这个分块，
我们按照以下步骤进行操作.
如下所示.
因此我们知道需要加载一个二维张量
所以需要定义每个向量( K和v)在第二维度上的范围
而这个范围由这个因子决定
接下来， 我们需要明确当前程序将要处理的 KV块是哪一个.
因此这个特定程序会跳过一些 KV块
这些块将由可能并行运行的其他程序处理.
如何根据程序(由序列长度除以块大小宏定义) 来确定该程序应处理的内容.
如果你还记得， 块大小宏是我们设定的固定值.
因此， 程序 ID 为零会告诉我们
有多少块大小宏的 KV已经被其他程序管理
我们无需关心它们.
因此， 我们跳过这些部分，
让我们回到这里.
这是我们需要跳过的向量数量. 因此， 其他 KB 从起始 KB 开始
我们需要加载多少取决于块 KB的大小
这个块 KB等于块大小宏.
因此， 它将包含128个向量.
因此， 我们定义了二维张量， 并将其存储在 SRAM中
BLoc K_ KV 因为在 Triton 中， 每次加载数据时都是从 HBM 加载到 SRAM 的
因此， 我们定义了它们在 SRAM中的存储位置
初始值为零， 现在开始加载它们.
我们按如下方式加载它们.
我们确认在 K张量指针中
它已经指向了正确的索引、正确的批次和正确的头
因为这是我们在前面步骤中已经处理好的
我们声明需要加载正确的键序列， 它应该从键的起始位置开始
因为这里已经包含了在序列长度维度上应该跳过的数量，
对于每个向量， 我们需要加载头维度中的所有维度
因为键
(如果我想提醒您的话)白 的结构是批次、头数量、
序列长度和头维度.
现在， 通过使用这行代码， 我们跳转到了正确的批次和正确的头.
这意味着我们已经在这里进行了索引， 并在此处选择了特定的索引.
因此， 当前 K 指向一个二维张量的起始位置， 而我们明确表示:
我们不需要整个序列， 只需要序列的某一部分.
哪一部分呢?
由这个start KV 指示的那一部分.
以及我们想要序列长度中的多少部分.
嗯， 我们想要 我觉得这样写会更清晰， 我们可以这样表达
开始到 start KV 加上 block KV 结束的部分.
因此， 我们想要这个数量的张量， 正好位于这个位置. 那么， 对于头维度
我们想要选择什么呢?
我们想要选择所有的维度.
因此
我们可以这样表示:从零到head dimension 的范围， 也就是这个:offs dim
s好的， 我们对k块执行这个操作， 同时也对v块执行同样的操作.
这里我觉得我没有修改注释.
应该是blockkv， i 而这里应该是 block KB， 之前它被称为 block KB1
就像原始代码中那样
我对命名做了一些简化.
我觉得这个版本更好， 更容易理解
因为在原始代码中他们也是用了两个for循环
但在第二个循环中他们用了倒序的方式
只是为了不改变循环的结构
但我觉得我的版本虽然更详细， 但更容易理解.
而且可能效率更低， 我的版本效率确实低得多
接着我们有off sq，
因为我们需要理解每个查询块需要加载多少向量
这由. off sg 表示， 并说明它们的数量
它是一个 blockq.
这个方法中的block g 颜色表示的是块大小-micro， t 也就是32个向量
好的， 现在我们需要访问 Q向量和 O向量， 不对
是 Q向量， 但它已经转置了.
还有 O向量， 我们也需要访问它们，
因为我们要遍历查询和 O向量， 为什么呢?
因为让我们看看这里， 看看论文中的公式:为了计算
向量uts

的原因
这就是为什么我们需要， 以及为什么我们需要以转置形式访问g的原因
因为我们需要计算， 让我在这里展示一下， pij转置.
亿计算

结里
的转置

这就是为什么我们访问的是查询转置而不是查询本身
就是
而我们访问查询转置的方式就是通过调整步幅来实现的
那 么我们就这么做吧， 我也在代码中写了注释， 角 解释为什么我们可以这样做
所以这相当于访问查询
首先， 好吧， 这个操作是什么?
这里的这个操作是什么?
这段代码的意思是:定位到查询的起始指针
这个指针已经指向了正确的批次和正确的注意力头
当前程序应该处理的就是这部分数据.
然后选择一个二维向量， 在这个向量中， 沿着列方向重复查询的起点
但实际上我们应该沿着行方向重复
因为我们想要选择查询的行.
不过， 如果我们想要选择查询的转置， 只需将两个维度互换即可，
所以， 让我在不进行查询转置的情况下展示给你看，
我们可以简化成这样.
要访问查询的指针而不进行转置，
我们可以这样做:定位到查询张量， 创建一个二维张量

在行中放置你想要获取的每个查询的起始点
并将每个这样的指针在列上也复制一份.
这就是添加这个" None"维度的含义.
这相当于在 Py Torch中使用unsqueeze操作.
 就像你在调用unsqueeze函数时， 我想是用了参数1.

因此 这等同于给这个张量增加一个列维度
并将列上的所有值重复一遍，
会有多少列呢?
当我们将其与这里的这个张量相加时， 它会进行广播操作.
这是un squeeze 和广播操作的结合.
因此， 我们正在提取由offsg指定的查询向量， 然后
对于每个查询向量， 我们都在选择由dim指示的所有头维度.
如果你反转这个广播操作， 就会生成
你想要访问的查询向量的转置.
所以这里的这些操作等同于这两行代码
即访问查询向量然后进行转置这是你可以实现的操作.
我可以从指针层面来阐述这个过程
基本上你需要将off-skew 视为一个指针向量.
我们乘以了序列步长(segue ncest ride variable ) 这告诉我们从一个查询向量到下一个需要跳过多少个元素
因为每个步长队列的步长将等于这个值
在头维度为128的情况下
序列维度的步长将是这意味着从一个查询向量到下一个， 你需要
向前移动128个元素， 因为我想提醒你
在内存中张量总是以扁平化的方式存储
即每个维度都与下个维度扁平化存储在一起.
想象一下你有一个三行四列的矩阵

但首先你会存储前三行， 然后是第一行.
也就是先存储前四列， 接着是接下来的四列， 然后是再接下来的四列
逐行存储.
不写下来确实很难直观想象.
那要怎么写下来呢?
那么一开始的偏移量是多少呢?
具体是0、
以此类推
我们将每个元素与序列的步幅相乘
这样就不会跳过任何元素.

这样正好跳过， 意味着健康维度是128 this will skip 这样会跳过两倍的128个元素
348 然后我们还要给这个向量添加另一个维度， 月 所以它将成为一个多维向量
接着， 你会在头部维度列数上进行广播
并向每一个添加一个数值.
因此它会变成一个类似于fall 的向量
好的， 各位， 我还是直接演示下吧， 不然我觉得这可能会让人太困惑
啊， 明白了， 那么我们得到的向量是这样的:首先是0， 接着是128然后是两倍的128， 再是三倍的128， 以此类推
我们正在添加由stream 指示的列数所以它们各自有多少列.
因此， 它已经包含了列的数目
为了简化起见 我们假设这不是128 维的
让我们假设它是四维的
那么， 这里就是四
这里将是四的两倍， 而这里则是四的三倍
我们正在添加另一个维度， 即dim 维度
每一个都乘以dim的步幅
这个步幅将为一， 因为它是最后一个维度
步幅dim， 那么我们正在添加多少列呢?
四列
我们正在添加一、零一 三， 我猜是这样的
对吧?
同样地， 我们还要加上这个， 哦， 天哪
同样地我们还要加上这个， 零、一、二、
然后我们还要在这个上面加上零、一、二、 二
那么， 这会选择什么呢?
这将从指针 Q 的起始点开始选择
它将选择第零个元素， 接着是第一个元素
然后是第二个元素， 最后是第三个元素.
这些正是我们应当选取的第一个向量的头部维度.
接着:它会从向量的起始点选择第四个元素
让我写下这个操作的结果.
这个结果将是0、12、
然后， 它会选择第4567 个元素.
接下来， 它会选择第8个元素， 我猜还有910、11.
之后， 它会选择第1213、1415个元素
因此， 从这个队列指针指向的起始位置开始
它会选择紧邻队列之后的第一个元素、
第二个元素、第三个元素，
依此类推

你可以看到， 这将成为第一个查询向量.
这将成为第一个查询向量
这将成为第三个查询向量
这是第四个查询向量， 因为它们在内存中是连续存储的
它们被展平了
因此， 在内存中， 它们是这样存储的.
它们的存储方式如下，它们一个接一个地这样存储.因此， 它将选择所有这些同时， 它还会创建一个具有我们想要可视化形状的虚拟张量.
正如我们之前所见， 当你在内存中处理张量布局时总是可以根据所需的形状将其视为任意形态
而重塑操作总是零成本的

它并不涉及改变内存中元素的排列方式
希望现在变得更清楚了
那么现在我们可以继续深入探讨了?
天啊，"这真是相当复杂呢."
所以每当遇到难题时， 我就会动手画图， 我也建议你这么做
so whenever i get stuck， i just draw things， and i think you should do it too TIMELINE OUTLINE le)
l因为这是学习的唯一途径.
如果试图在脑海中想象所有内容， 往往很困难.
对于 O向量， 我们也采用同样的方法处理，
处理 O向量时， 我们不需要以转置形式访问它， 因为转置在此处并非必需
只有 Q向量， 我们需要以转置形式处理
好的， 它沿着查询的序列维度快速遍历.
于是， 我们从第零个查询开始.
在当前情况下.
在查询中， 我们需要遍历整个序列长度维度
因为只有在键中， 我们才能选择想要处理的正确键.
这里我想提醒大家， 我们固定了键， 并遍历所有查询
而查询需要从零开始， 直到序列长度结束.
因此 这个for 循环的步数将是序列长度除以查询块大小
如果序列中有. 1000个元素， 而查询块大小为32

那么步数就是1000除以32.
选择1000不太合适， 应该是1024， 召 否则就无法整除了
于是， 在这个for循环中， 我们遍历每个块， 并加载一个查询块
第一个块由我们的指针指示.
在迭代结束时， 我们将指针移动到下一个查询块，

好的， 我们还会加上存储在 M矩阵中的logsum exp 值
因为我们想要实时计算 PT.
PT 是查询与键相乘后经过soft max 的转置
但我们希望避免先计算查询与键的转置相乘
再进行转置操作.
由于我们已经以转置形式访问 Q
因此可以直接计算 PT而无需先计算 P再进行转置操作
因此， 我们从logsumexp矩阵(即前向传播过程中计算的 M矩阵) 中
加载所需元素的偏移量
并且我们每次访问一个查询块
即当前迭代中正在处理的那个块.
由于键已经转置， 我们直接进行相关计算
如果你想得到 PT， P应该是.

实际上这不是 P， 因为我们还没有进行softmax操作， 它实际上是 ST.
不过， 如果你想要得到 PT， 你需要对 ST进行softmax 操作
对 ST 进行soft max 操作的结果就是它
A是 S的转置.
S是查询query 广乘以它的转置
因此， 要得到 ST， 你需要对键( Key)进行转置， 而不是键乘以查询的转置
正如你所记得的， 在矩阵乘法中如果你对矩阵乘法进行转置
你也需要将矩阵乘法中的两个元素进行交换.
这就是为什么我们要进行键乘以查询转置的操作.

这样我们就能得到 S 的转置，
在应用之前我们还会用softmax的缩放因子对其进行缩放
要应用softma×我们只需对每个元素取指数， 减去其最大值
再除以归一化值.
但使用对数求和技巧时我们只需将每个元素减去m值
这个m值已经包含了归一化因子
我想我已经推导过这部分了
所以我们不需要再重复一遍.
好的， 现在我们实际上有了 pt块
所以在这个公式中， 我其实应该写成st.
好的， 那么在实现因果注意力机制时， 我们还需要屏蔽掉一些值
正如你在这里所看到的
因此， 在这种情况下， 因果掩码是在计算完so ftmax之后应用的
因为在这个步骤中
你通常是在计算softmax之前应用因果掩码的
但这实际上是在前向传播过程中进行的
373 因为你不希望归化因子受到那些本应为零的元素
的影响

但我们已经计算了归一化因子， 所以它不会再受到影响了.因此， 我们可以在应用:soft max 之后进行掩码操作因为归化因子已经基于我们之前应用的掩码计算
这就是为什么我们可以在应用softmax之后再进行掩码操作.
嗯， 所以掩码始终是相同的
因此， 如果查询的索引超过了某个值
那么在这种情况下:掩码对于所有不需要被屏蔽的值来说都是有效的
所以;所有不需要被屏蔽的值就是这里的这些
而其他所有值都将被替换为零
好的， 所以在我们已经对pt块进行掩码操作之后， 京 就可以计算dv了:
我会在论文中指出正确的公式.
所以我们加载一个d的块吗?
为什么我们不加载一个d的块呢?
让我们看看论文.
那么如何计算dv块呢?


它是一个累加和旧的dv加上 PT.所以这里 PT. 的下标表示应用了dropout 之后的 PIJ.在这个实现中， 我们并不支持dropout，而且实际上很少有模型会在注意力机制中使用dropout. 所以 PT 乘以doi.所以doi 的块和doi 是同一个块.那应该也是;嗯. HBM.Wridoic和-kid Qgi始终指向各自张量中的同一行块.这就是原因， 因为内层迭代中的i 表示g的 F个块和 O的一个块，因为 DO和 DQ的形状是相同的，因此，需要 DO而计算 PK则需要

这就是为什么我们要按照论文中的方法计算 DV，所以用 PT块乘以 DO. 这样， 我们就计算好了 DO块，接下来， 我们需要加载预先计算好的 DI元素.第一次调用函数时， 称为注意力反向预处理器，因为计算dk时需要用到它，让我们看看. 我们加载了多少个? 加载的数量与查询数量完全相同，因为我们总是加载相同数量的块大小微向量. 好的， 我会复制一些内容并逐步解释. 那么， 接下来我们需要进行的操作是计算这个dk.为了计算dk， 我们需要dst.为了计算dst， 我们需要得到 dp. 所以我们一步一步来

我们从这些公式的末尾开始， 逐步回溯到它们的起点， 这样我们就能理解
每个部分是如何被使用和创建的那么， 我们从decay 开始吧.如果你看论文，等守旧的decay 加上ds的转置乘以o的=个块yrquire嗯这就是这里所写的内容.

所以这里的加等号基本上就是旧的加上新的
这是一种增量累加的方式.
所以用这里的新内容来更新旧的k.
所以soft max 的缩放因子(这里用tau 表示)
Q 会乘以 dst块
与 Q转置的矩阵乘法结果.
你可以在这里看到这个 Q， 但我们目前还没有 Q
我们有一个( 的转置所以我们对 Q转置再取一次转置， 它就变回 Q了
395 让我们来看看这个 Dst块. 根据论文中的公式， Dst的计算方式如下，我们有 Ds

这里的 Ds等于:对，就是这里， 它等于一个块pij 逐个元素乘以dpi 减去 di.
不过我们不需要s， 我们需要的是s的转置所以计算:的转置时， 这是一个逐元素乘法而不是矩阵乘法.

这意味着当你对这个操作取转置时不需要反转任何东西 只需对两个操作数取转置即可.因此， 为了计算st我们对p取转置， 得到pt， 这我们已经有了然后对括号内的所有内容取转置，么， 这个dpt减去di 在这里我们将行与列互换， 所以这个dpt是什么呢在论文中， 我们知道 dp的公式:dp在这里， 它等于d.等一下， 这里的dp等于 Do乘以 V的转置. 下这里的dp 等于 DO乘以 V的转置. 但我们不需要 DP， 我们需要的是 DPT.在这种情况下这不是逐元素乘法， 而是矩阵乘法.因此为了得到 DPT而不是 DP 因此， 为了得到 DPT而不是 DP，我们需要对这个矩阵乘法的两个操作数取转置.在矩阵乘法中， 当你取转置时，

还需要交换两个操作数的顺序.因此，. 我们需要对 VT取转置， 它就变成了 V， 也就是 V块，因此， 我们需要对 V丁取转置， 它就变成了 V， 也就是 V块再与另一个操作数进行矩阵乘法， 即 DOI的转置，再与另一个操作数进行矩阵乘法， 民 即 DOI的转置,这就是为什么我们要对 DO 进行转置操作,现在我不打算逐一讲解所有指针,因为我已经告诉过你们如何检查指针指向的内容,以及偏移量所指的是什么.

希望现在你们对 Triton中指针的工作原理有了更深入的理解,其实它们在 CUDA 中的工作方式也是一样的,因为在(GPU 中我们只能获取到张量起始地址的指针,然后需要自己计算出所有的索引.我们已经计算完了dk 块， 现在转到下一个查询也就是下一个查询块，因为我们固定了k和v块， 并遍历所有查询

所以需要移动查询， 通过步长序列转置指针,这意味着我们如何从一个查询转到下一个查询.我们与当前块g相乘是一个向量，指示我们正在访问的g中当前元素的指针,对. 也进行同样的操作， 并使用块 g 作为元素和步长 q， 因为 d、和g都具有相同的形状，在我们运行完所有查询的for循环后， 就可以存储这个dk和dv块了

因此， 我们按以下方式将其写回 BLOCK_ Q* stride_seq 这就是我们函数的结尾了， 伙计们.因此 我们将dv 块准确地保存在当前dv已经指向的位置,我相信它已经指向了正确的批次和头,因为我们在这里以及dk的情况下都对其进行了递增,然后我们需要在序列维度中告诉它应该在何处保存这个k和v块这是由此指示的.我们像之前那样创建指针，伙计们， 别让我再重复一遍了.这真的很简单.

如果你像写这个键和值指针向量一样写下来实际上它们不是指针， 而是你需要从序列维度中提取的键和值范围

你只需添加另一个维度也就是列维度， 然后在列中重复每个值最后在这里添加头维度，总之， 在我们计算完指针后， 也就是确定dk和dv应该存储的位置后我们将它们存储起来，我们将这些指针存储在dv 中我的意思是， 我们将它们存储在 DV 张量和 DK张量中我们保存了什么?我们保存了 DV 块和 DK 块,也就是我们在编写的for 循环中逐步修改的部分.好的， 现在我们完成了这部分,接下来可以进人下一个函数， 它将处理另一个for循环.那就开始吧.

好的， 现在我们来处理迭代的第二部分， 也就是这一块内容.我先复制一下这段代码然后再详细说明它的功能,我们把它写在这里吧.好的， 我们沿用之前的启动网格配置.我们得先声明这个函数.再次强调， 由于网格是根据块大小宏定义的,而我们要保持这一部分固定不变，接着， 在四重迭代的内部， 我们以微块大小为步长进行处理.这里我们固定 Q不变而遍历 K和 V,因为需要计算 DQ.目前我们已经计算出了 DK和 DV.

好的， 我认为参数设置与之前相同,事实上这也是 Triton 官网原始实现中,作者决定使用相同for循环但不同参数的原因,我觉得那样有点让人困惑,所以我把它们分开了.我只是把代码重复写了两遍,这个视频的目标是尽可能易于理解而不是追求最高效率.我们回到这里， 让我再复制一下函数签名我来定义这个函数.同样地， 我们需要先将查询、 键和值移动到正确的指针位置 这些指针将指向当前程序正在处理的批次

BATCH_ SIZE， NUM_ HEADS， SEQ _ LEN， HEAD 和注意力头. 那么， 我们开始吧让我看看代码在哪里. 第一部分和我们之前写的其他for 循环完全一样. 

我们回到这里， 实际上我只是复制了代码， 所以它完全一样.我们检查当前的批次和注意力头索引 将查询键和值的指针移动到正确的位置.将查询q、 键(k)、值(v)白 的指针移动到正确 的位置， 同时将掩码(m)和归一化因子(d)的指针也移动到正确的位置和之前的操作完全一样.所以我觉得没有必要再重复解释了.然后我们加载-个查询(g):的块， 这个块在后续计算中会保持不变所以， 查询(q) 实际上我在这里加载了很多内容.

好的， 我们定义了在头维度上加载键( K )和值( V)块所需的偏移量， 因为我们将对键和值进行迭代.我们将以转置块的形式访问它们.因此， 我们不直接访问键( K)和值( V) 而是以转置后的形式访问它们， 民 即 KT和您知道， 这只需要通过改变步幅(strides ) 就可以实现.在这种情况下， 因为我们将它们视为二维向量来处理.当我们想访问键(k) E 时， 我们将偏移量kv视为未转置的k.您将这个偏移量k V视为一个行向量， 抱歉， 是列向量，因此， 您需要在行上重复， 每次访问所需的k偏移量

在这种情况下， 我们将其视为行向量， 因此它将在行上重复.抱歉， 它将在列维度上进行广播这样您就可以访问转置后的k以及转置后的我们还在加载g 向量， 这个向量的位置取决于偏移量.(skew) 而偏移量文基于起始队列(start queue )这进一步取决于程序开始处理的具体起点.这是因为该程序是以二维方式运行的第一个维度表示程序应处理哪个批次和哪个注意力头， 而第二个维度即程序索引1号0)则指示在所有的序列长度中该程序具体要处理哪个查询.这一点由索引块(index block )来指示.在这个情况下， 实际上应该是q.我忘记改名字了， 所以让我现在改一下所以它是索引1 Q， 因为我们跳过了部分 Q 我们跳过了多少 Q? 根据当前程序的索引，乘以前序程序已处理的块数来决定，这将告诉我们在序列长度范围内，当前程序需要选择哪些查询，因此， 我们使用起始查询加上块队列的范围来确定，

假设在所有序列长度中， 这个程序的起始查询是100，那么它将加载查询行100、101、102， 依此类推，直到100加上块队列减一的位置，这就是我们在当前程序中要加载的查询向量的范围，我们通过使用 Q加上列方向重复的偏移量来加载 Q的块，因此，我们将其视为列向量， 但在行向量上重复广播，其中每一列将是一个头维度乘以步幅.在这种情况下， 我们实际上也可以不乘以步幅，因为在维度维度中的步幅， 即批次的最后一个维度的步幅是一

因为从一个元素移动到下一个元素，你只需移动一个元素的位置，因此最后一个维度的步幅总是一，因此， 我们加载了dq， 即本次迭代中要计算的内容 然后我们需要加载do， 并且do使用与g相同的偏移量因为和dg具有相同的形状并且以相同的方式工作. 因此， 我们加载了q的一个块， 并加载了对应的 DO块(在本例中) DO的形状与 O相同， 而 O的形状又与 Q相同

此外， 我们还需要加载 M归一化因子， 这些因子位于 M矩阵中，对应于我们将在本程序中处理的，特定查询组，我们从偏移量开始.如您所见， 偏移量是从零位置开始的第一个 KV块.因此， 国 因为我们将会遍历所有的 KV， 月 所以从零位置的 KV开始，即从第零个键向量和第零个值向量开始.然后， 在每次送代中， 我们将按 KV块中的向量数量向前移动.

希望我没有讲得太快，因为这里写的大部分内容与我们在另一个for循环中已经完成的内容非常相似，所以我不想过多重复，真正重要的是我们将使用的公式，这些公式与论文中的完全一致，我们遍历这些 KV 块，我们加载第一个 K转置块和 V转置块，像往常一样这样加载，以及另一个要加载的元素指针，Triton就会将你请求的块加载到 SRAM中，因此 这些数据都存储在 SRAM 中， Q 和 DO也同样存储在 SRAM中

接着，我们计算查询与键转置的乘积，因为需要计算 P块，因此， qk块就是当前查询块中的查询与当前键块中的k转置相乘的结果但由于我们访问键时已经进行了转置， 所以不需要再次转置.

即使需要转置，也不需要任何计算，只需以不同的方式访问它即可.因为在内存布局中它总是以扁平数组的形式存储. 接着， 我们计算p 块， 它是 softmax 的输出，对于每个查询键我们减去该查询块的logsumexp值这就是为什么我们在加载m块时使用正在加载的查询的偏移量如您所知m块巴经包含了归一化因子，因为每个m实际上是每行的最大值加上归一化因子的对数，当您应用指数属性时， 它会进入分母，好的， 然后我们再次应用自回归掩码，哎呀， 我做了什么?

让我回到这里的代码.所以我们有这个阶段，因此，当我们启动反向传播时，阶段三表示在前向传播中我们计算了因果注意力，如果我们在前向传播中计算了因果注意力，那么在反向传播中也需要屏蔽这些元素 mask_block 因此，我们创建了一个掩码这个掩码仅在查询索引大于键索引的元素时为真，如果条件为真则不进行掩码处理，否则进行掩码处理. N 接下来我们计算下一个操作， 即计算dp和ds.
实际上，我们直接计算dk，: 然后像之前一样进行解释，我们从末尾开始， 逐步回溯到计算所需的部分，如果你查看公式让我确认一下这个.我认为我们不需要. 好的， 我们转到i Pad 上来继续. 好的， 我们这里要计算的是dq.

论文所示 dq 等于原有的dq 加上 tau，即 softmax 缩放因子，也就是这里的这个值)，再乘以ds，这里的ds块就是 ds 块，而 k 块则是 kt 块的转置，这里的 ds 块就是 ds 块，而 k 块则是 kt 块的转置，因为我们已经以转置块的形式访问了 k，我们也可以通过反转来直接访问未转置的 k 块，如果你不想以转置块的形式访问它，就像这样操作，像这里一样，不转置这样会将其视为行向量，并在列方向上广播，因此，这里也需要调整，因为需要将其视为列向量来处理.

但如果你想以 K 转置的形式访问，只需反转这两个操作即可，希望我没有搞错什么，我们继续往下推进吧，好的，我们知道 DQ 的计算公式与论文中完全一致，但这个 DS 块是什么呢? 让我们来看看论文，这个 DS块正是源自这里的这些内容，DS 块正是源自这里的这些内容，因此，我认为这里的 DS块， 是一个 π(pi) 元素，通过逐元素乘法与 dπ 减去 di 结合，即 dπ 减去 di.

那么这个 p 块是什么呢? p 块实际上就是 softmax 的输出， 而这个输出我们已经得到了 dp 块是什么呢? dp 块实际上就是 d O 乘以 V 的转置，其中 d O 我们已经加载好了，而 V也是以转置形式加载的，这就是我们计算 d Q 的方式

接下来，我们当然需要移动到下一个键值块(key Vs)，因此我们像之前一样递增指针，于是，我们移动到下个键值块 (keys 和 values)，同时，我们也像之前一样移除指针，接着，我们需要存储 DQ 的结果
通过像下面这样分割 for 循环，我们只需对 HBM 进行一次写入操作，如果你看原始算法，我不确定它是否真的对应他们在 CUDA 中的实现，但在原始算法的论文中，还需要访问所有队列，这种方式并不优化，因为每介键值的更新仅依赖于特定的队列块，抱歉

因为 的一个块依赖于 Ks 的所有块，因此我们进行了分割，这就是我们写的第二个循环，现在我们已经写好了实现 Flash Attention 所需的所有内容，包括前向传播和反向传播。

所以，我们应该准备好启动内核了，希望我在复制代码时没有出错，所以我不打算尝试启动它，如果有任何错误，我会直接使用我已经写好的参考代码，这些代码我之前已经用作复制，到目前为止，我的参考代码和我们刚刚写的代码之间唯一的区别是自动调优，这一点我还没有解释。

那么， 我们来谈谈自动调优吧，自动调优功能在原始论文中就已经存在，我直接保留了它，没有做任何改动，我移除了反向传播的自动调优功能，但在前向传播中，如果你仔细看，这里有一段代码，它指示了用于 Triton 的自动调优配置，因此 Triton 基本上无法预先知道最佳的块大小，也无法确定查询键和值的最佳块大小，或者我们其他维度中的最佳块大小，我们需要根据运行的硬件条件，SRAM 的可用性，以及 Triton 能够应用的线程粗化策略来进行尝试。

另外，我还没提到线程粗化这个概念，简单来说，在 CUDA 中，你可以选择每个线程执行一个原子操作，例如，在矩阵加法中，每个线程要么负责输出矩阵中一个特定元素的加法运算，要么管理多个元素的计算，这就是所谓的线程粗化，我认为，我没有查阅文档，但我相信 Triton 会根据你指定的块大小
和所需的 warp 数量自动处理线程粗化，数量指的是什么? 一组线程块? 组由 32 个线程组成的协作单元，它们同时运行相同的指令，阶段数量则更为有趣，这是 Triton 进行的一项优化，本质上，这并不是循环展开，那么，我们实际上来讨论一下，来讨论一下软件流水线，因为这是我们理解这段代码的最后一部分，也就是自动调优.

所以我认为这里最有趣的部分并不是选择 Q 和 K 的块大小，因为这其实只是基本操作，你只需要根据运行时间，尝试各种配置，找出效果最好的那个，实际上会为你运行所有这些配置，每当序列长度或头维度发生变化时，对于每一对头维度和序列长度的组合，它都会选择运行时间最短的最佳配置，这实际上能为你带来最佳的吞吐量，那么，我们来看看这个 numstages，它是什么以及它是如何工作的，那么， 我们开始吧。

好的，软件流水线化是在你有一个类似 for 循环的情况下使用的，你有一个顺序操作，在这个操作中每次送代都不依赖于前一次迭代，因此，你在一次送代中执行的操作与你在前一次选代中所做的操作是独立的，这或多或少与我们之前在for 循环中所做的工作类似，实际上，我相信在某些情况下，这一点并非必须成立。

也就是说，即使操作之间存在依赖关系，你仍然可以进行软件流水线化，比如想象你肴一个媚下的 for 循环：从1到的循环， 首先你加载一些数据，然后加载另一些数据，接着进行矩阵乘法，最后存储一些数据，这里你在读取数据，这里你也在读取数据，这里你在进行计算，而这里你在写入数据，如果我们观察每次迭代中发生的事情，会看到以下情况。这个单元负责从内存中读取数据或向内存写入数据.

从时间轴上看，在第一次送代中，首先我们会 读取一些数据，而此时计算单元处于空闲状态，因为它需要等待这些数据，接着我们会读取更多数据，计算单元依然空闲，因为它需要等待这些数据。

最后，当我们有了足够的数据时，计算单元就可以执行操作了，而读取单元此时则处于空闲状态，然后我们会将一些数据写回内存，而计算单元再淡进入空闲状态，直到它获得足够的数据才能继续执行计算。正如你所见，这种方式效率并不高， 因为在往荷时刻，因此，优化这个循环的一种方法是采用软件流水线技术。你可以通过告诉 Triton 你希望有多少个流水线阶段，让它为你的循环实现这一优化

接下来， 我们来看看它是如何工作的，因此， 对循环进行流水线化意味着首先需要将所有操作转换为异步，存在异步内存加载和异步内存写入操作，它是否已完成，并继续执行下一条指令，在这里，我会启动一个加载迭代，它会立即返回并继续执行下一条指令，然后我可以进行计算，我会先检查这两个操作是否已完成。

因此，我可以立即启动两个读取操作，然后只需检查它们是否已完成，通过软件流水线技术，我们将不同送代的接 操作整合到一个选代中，接下来， 在第二次迭代中，我们读取第二次选代的第个矩阵，我将其称为读取 A 和读取 B， A 表示读取我们需要的第一个矩阵，所有这些操作都是异步的。然后， 我在第三次送代中启动另个异步操作内容是：读取第空次迭代的第不矩阵，随后进行矩阵乘法计算，因为在第三次迭代时， 这两个操作应该已经完成了，但在计算矩阵乘法的同时， 我不会让加载单元闲置，这只有在能够启动异步操作的情况下才能实现，因此在第三次送代中，然而， 在计算矩阵乘法的时，

我已经启动了一些异步操作，因此，在第四次迭代中，我将启动第四次迭代的数据加载操作同时加载第三次迭代的数据，在编程语言或 CUDA 语言中有一些原语，可以用来检查操作是否完成，因此，实际上在进行乘法运算之前，我们会先检查异步操作是否已经完成。

从时间的角度来看这就像在 JavaScript 中一样的东西，你可以在真正需要它们之前等待 Promise 完成，在 C# 中， 我想它们被称为任务因此你可以生成任意数量的住务 然后在需要时只需等待你所需的那一个，而其他任务仍在后台算步运行，这就是软件流水线的核心理念，正如你所见，软件流水线我们可能拥有足够的数据来执行前两次送代以及第三次送代的一半数据因此我们需要增SRAM 的内存需求。

好的，Triton 会为你处这种软件流水线操作，它会将所有加载存储操作并为你完成这种流水线处理，因为我们在模型训练中它经采用了类似的做法，这种方法被称为流水线并行.

流水线并行的工作方式如下，我们有一个非常大的神经网络，无法完全放入单个 GPU 中，假设这个神经网络由三层组成，分别是第一层、第二层和第三层，但由于模型规模过大，无法完全容纳在一个 GPU 中，于是，一种解决方案是将每一层分别放入一个 GPU 中，例如，我们将第一层放入 GPU1，第二层放入 GPU2，第三层则放入 GPU3.

假设我们为这个神经网络提供了一个输入，于是，我们将其送入第一个 GPU，GPU1 会处理第一层
并生成一些输出，这些输出随后会被传输到 GPU2，GPU2将计算其自身的输出并传输到 GPU3，GPU3 再计算其自身的输出，最终我们就能得到神经网络的最终输出，问题在于，当你将 GPU1 的输出发送到 GPU2，以便 GPU2 执行其自身任务时 GPU1 此刻便空闲了，这样会造成资源浪费，我们应始终让 GPU 保持忙碌状态，因此，我们可以采取的一种方法是，不将所有数据一次性发送到 GPU，而是分批发送多个较小的数据块，它是如何工作的呢?

想象一下，我们将批次0，也就是第 0 批数据，发送到 GPU1，GPU1将计算其输出并发送到 GPU2，此时，GPU2正在处理第0批数据，此时，第0批数据已不在此处，但此时 GPU1已空闲， 因此我们发送另一个称为批次 1 的微批次数据，接着，GPU2将完成对批次 0 的处理，并将其发送至 GPU3，此时，GPU3 已接收批次 0 的数据，而 GPU2 则处于空闲状态，于是我们进行了数据传输，同时希望  GPU1 也已完成了任务，因此，我们将批次 1 的数据从 GPU1 传输至 GPU2，随后 GPU1 将恢复空闲状态，于是我们进行了数据传输，此时，GPU 1 变为空闲状态，而 GPU2 开始处理新的任务，由于  GPU1 现已空闲，我们可以开始处理另一个批次的数据，接下来是批次 2 的数据，以此类推，不断循环这一过程。

因此，每当我们将一个批次的数据从一个 GPU 转移到另一个 GPU 时，我们都会在流水线的起始端引入一个新的批次，并在每次送代中使各个批次向前移动一个位置，这种操作方式确保了 GPU 始终处于忙碌状态。

流水线并行技术存在一个问题， 即所谓的"气泡效应"，这是因为在流水线初始阶段，需要一定的时间来填满流水线，此外，流水线并行还面临反向传播步骤的挑战，在反向传播过程中，必须严格按照接收微批次的顺序逆向执行，而在 Triton 中实现软件流水线时， 多会遇到前导和尾声的问题，这是因为需要先建立流水线才能开始流水线处理并且在流水线结束时必须处理完流水线中所有的数据。

因此，只有在 for 循环的初始阶段和最后阶段，GPU 的所有计算单元可能不会同时工作，这意味着，为了有效利用流水线技术，需要确保 for 循环的迭代次数远大于将迭代过程划分的阶段数，在这个例子中，我们有四个阶段，这些阶段被称为流水线阶段，因此，我们希望迭代次数要远失手。

好了各位，我终于完成了这个视频，我相信我们可以运行 Triton 代码，那么让我们实际运行一下，我已经把代码都复制好了，我相信我们也放了代码来测试，但还没放主方法，现在可以复制过来，真的希望没有错误，真心希望如此，让我确认一下是否在正确的机器上，是的，没问题。

直接运行程序吧，祈祷吧，如果有错误，我就直接复制自己的参考实现，但我希望它能正常运行，否则可能是我漏掉了什么，我正在 H100上运行我的代码， 因为公司有 H100，如果你用的是较小的 GPU，可以尝试缩短序列长度，你可以减小批次大小.

我觉得它已经是一了，当我们调用它时，哎呀，最佳大小没了，你可以减小批次大小、头数或序列长度，你甚至可以将头维度设为 8，序列长度设为 16。

我们来看看，运行反向传播，尝试执行反向传播，返回的梯度数量不正确，预期五个，实际得到一个，我们可能漏掉了一些返回语句，我觉得是这样，没错，我在这里忘记写返回语句了，因此，在运行完最后一个循环后，我们需要返回计算得到的结果，再次祈祷好运。

好了，通过了，所以由 Torch 计算的反向传播与我们的反向补丁在 10 的负二次方的绝对误差范围内是等价的，所以，正如你所见，我们在这里运行的反向传播与之前的不同，因为当你应用 triton attention 时，会在我们张量的计算图中引入一个新的计算图，其中会包含这个 triton attention 操作符，而当 PyTorch 想要计算反向传播时，它会直接调用这个 triton attention 的反向函数来进行计算.

它会为所有作为这个 triton attention 输入的张量填充 `grad` 值，这就是 PyTorch 自动求导的工作原理， 各位，感谢大家观看我的视频。

这真是非常、非常、非常有挑战性，我花了好几个月的时间自学 Triton、CUDA、flash attention 等等，而且，我还有一份全职工作，所以制作这样的视频真的很难，我需要投入大量的时间，比如晚上、早晨和周末，我花了三天时间才录完这个视频。

因为有时候我不满意自己的解释方式， 有时候会出错或者有时候需要重录，因为我的做法不对等等，而且我相信到目前为止我所做的内容应该没有大的错误，但可以肯定的是，我的符号表示可能不太规范，因为我所有的数学知识都是自学的，我都是自己学的，所以，因为我不是在学术环境中学习的，我有些不好的习惯，正在努力改正，所以我用的符号表示可能不太标准，有时候我用大写字母表示，有时候用小写字母，有时候甚至忘了加下标等等，所以我正在努力解决这些问题，我觉得我已经解释清楚了，所以应该没问题。你应该已经掌握了推导 FlashAttention 论文中所有公式所需的知识，同时也应该对注意力计算是如何逐块进行的有了一个清晰的内部理解。

我知道我本可以花 20 个小时把事情讲得更清楚，我也有自己的生活，还有妻子要陪伴，所以我不可能做出 100 小时的视频，另外，制作这些视频时还有一些干扰，我拔了几颗智齿，恢复至少花了一周多的时间，因为实在太疼了，所以，感谢大家观看我的视频，希望这次你们也学到了很多。正如你所见，Triton 是一个新事物，相关的文档资料并不多，因此，我所说的关于 Triton 的内容可能并不完全准确，因为确实缺乏足够的文档参考，我所掌握的 Triton 知识，都是通过研究他人编写的代码并试图理解它而获得的，我想就到这里吧， 各位

