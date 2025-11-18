

[source](https://aman.ai/primers/ai/speculative-decoding/)

## 一、背景

不同于每次仅预测一个标记，这些方法尝试并行猜测多个未来标记并进行高效验证。这一通用理念可通过多种方式实现，以下列举三种最常见的方法：

1. 通过草稿模型进行推测性解码：一个更小、更快的“草稿模型”在主模型之前生成一系列候选标记。然后，完整模型并行验证这些标记，接受正确的预测，仅在检测到不匹配时回退到标准的自回归解码。这种方法在实践中可以实现 2 倍至 6 倍的加速。

2. 基于树的多头验证（Medusa）：不同于使用单独的草稿模型，主模型配备了多个并行的“验证头”，以树状结构生成和检查替代令牌路径。这使得可以同时探索多个候选延续路径，从而减少所需的全前向传播次数。

3. 多令牌预测头（自推测解码）：模型通过额外增加的输出头进行增强，这些输出头经过训练可直接从当前隐藏状态预测多个未来令牌。这些预测在内部进行验证，使模型能够绕过多个顺序步骤，而无需引入外部模型或分支搜索结构。

在这些方法中，核心原则始终如一：在每次前向传播时执行更多推测性工作，然后并行验证正确性，从而缓解逐个生成下一个标记的严格串行特性。

**示例场景：**

假设我们要生成 5 个标记。传统解码方法需要进行 5 次前向传播。而使用推测性解码时，草稿模型可能一次性生成所有 5 个标记。然后目标模型会对这批标记进行验证——根据需要接受或修正。例如，如果有 4 个标记被接受，则仅需 2 次前向传播，从而实现 2.5 倍的加速。

这一优化特别适用于：

* 实时应用（聊天机器人、代码补全）
* 边缘部署场景
* 高负载服务器环境

下图（来源）直观展示了非推测性生成（左）与推测性生成（右）的对比。

## 二、核心技术

推测解码有多种形式，但核心理念是一致的：使用轻量级方法猜测多个标记，然后通过原始（或“目标”）模型进行验证。接下来我们将探讨主要策略、实现模式及其权衡。

### 2.1 通过草稿模型进行推测解码

由Leviathan等人在2023年发表的《通过推测解码实现Transformer快速推理》中提出。

Pipeline 概述：

1. **起草**：使用一个更小（更快）的模型来生成 $γ$ 推测性标记。
2. **验证**：运行大模型对所有标记进行评分，直至 $γ$。
3. **接受度：** 接受与大模型预测结果相匹配的前缀标记。
4. **后备方案**：若令牌出现偏差，则回退至大模型采样进行修正。

论文中的下图展示了无条件语言建模案例中说明的一种技术。每条线代表算法的一次迭代。绿色标记是近似模型（此处为一个在1m1b数据集上训练、具有600万参数、处理8k标记的类GPT Transformer解码器）提出的建议，被目标模型（此处为相同设置下具有9700万参数的类GPT Transformer解码器）接受；而红色和蓝色标记分别是被拒绝的建议及其修正。例如，第一行中目标模型仅运行一次，生成了5个标记。

算法概要（根据 Leviathan 等人简化）：

```python
def speculative_decode(draft_model, target_model, prompt, gamma):
      draft_tokens = draft_model.generate(prompt, max_new_tokens=gamma)
      scores = target_model.score(prompt + draft_tokens)
        
      # accept up to the first mismatch
      n_accept = count_agreement(draft_tokens, scores)
      accepted = draft_tokens[:n_accept]
        
      # complete the next token from the target model
      next_token = target_model.sample(prompt + accepted)
      return accepted + [next_token]
```

**优点：**:

- 无需重新训练即可插入现有模型。
- 无需对大模型进行架构更改。
- 完全保留输出分布。

**挑战**:

- 维护一个独立的草稿模型会增加系统复杂性。
- 草稿模型与目标模型之间的分布不匹配可能会降低接受率。
- 如果两个模型都很大，会带来内存和计算压力。

### 2.2 基于树的多头验证（Medusa）

美杜莎（Medusa）由Cai等人（2024年）在论文《美杜莎：基于多重解码头的简易LLM推理加速框架》中提出，该框架通过采用树状注意力机制，对基础的多头推测解码技术进行了改进。

博客（来源）中的下图展示了Medusa在Vicuna-7b上的性能表现。

**核心功能：**

* 美杜莎头：每个头从最后一个隐藏状态预测未来 $k+1$ 个标记。
* 候选组合：将每个头部的 top-k 输出组合起来，形成推测树。
* 树注意力：一种自定义的注意力掩码确保标记仅关注其路径上的前驱节点。
* 接纳方案：两种选择：
	* 拒绝采样（匹配基础模型）
	* 典型验收（启发式，更快）

**好处：**

- 在质量下降最小的情况下实现更高的加速（生产环境中约 2.3-2.8 倍）。
- 无需重新训练（Medusa-1）或通过联合训练（Medusa-2）即可轻松集成到现有模型中。
- 适用于批量大小为 1，这与实际使用场景（如聊天）相符。

**实现细节：**

pkt=softmax(W2k∗(SiLU(W1k∗ht)+ht))

其中 W1k 初始化为 0，W2k 是基础模型语言模型头部的克隆副本。

### 2.3 多令牌预测头（自推测解码）

由Gloeckle等人（2024年）在《通过多令牌预测实现更好更快的语言模型》中提出。

最近的一个趋势是不使用单独的草稿模型，而是直接在主模型中构建推测能力。这就是多令牌预测头（multi-token prediction heads）的用武之地。

论文中的下图展示了多令牌预测的概览。（上）在训练过程中，模型通过共享主干和4个专用输出头，一次性预测4个未来令牌。在推理阶段，我们仅使用下一个令牌的输出头。可选地，其他三个头可用于加速推理时间。（下）多令牌预测提高了MBPP代码任务中的pass@1性能，且随着模型规模的增大效果尤为显著。误差条表示通过数据集样本的bootstrap计算得出的90%置信区间。

**架构**

- 共享的变压器主干对上下文进行编码。
- 多个解码器头（每个对应一个未来标记）进行独立预测。
- 第一个头部是标准的下一令牌预测器；其他头部预测第二、第三……第n个令牌。

每个头部都通过交叉熵损失在其各自的位置上进行训练：Ln=−ΣtΣni=1logP(xt+i|z1:t)

其中𝑧1:𝑡是共享的潜在上下文，每个𝑃(𝑥𝑡+𝑖)通过其专用头计算得出

**内存优化**

训练过程中不是为所有𝑛个头生成所有logits，而是顺序处理每个头以减少GPU内存占用：

* 计算前向和后向传播（头部1）
* 释放对数概率，移至头部2
* 在共享主干上累积梯度

这将峰值内存从𝑂(𝑛𝑉+𝑑)降低到𝑂(𝑉+𝑑)，且不会影响速度。

**优势**:

* 无需单独的草稿模型。
* 统一架构（更易于部署、量化和训练）。
* 兼容推测解码方法，如块级并行或Medusa。

**缺点**：

* 需要在预训练期间修改模型
* 收益仅在大规模模型（70亿以上）中显现
* 微调这些模型可能需要小心操作以保持对齐性。

## 三、对比分析

在本节中，我们将系统性地比较前文讨论的关键推测式解码策略——基于草稿模型的解码、多令牌预测头以及Medusa方法。我们将从性能表现、集成便捷性、训练需求以及部署复杂度等维度权衡这些策略的优劣。

|**Criteria**|**Draft Model  <br>(Leviathan et al., Nov 2022)**|**Medusa Tree‑Attention  <br>(Cai et al., Jan 2024)**|**Multi‑Token Prediction Heads  <br>(Gloeckle et al., Apr 2024)**|
|---|---|---|---|
|Model changes required|None|Optional (Medusa‑1) / joint (Medusa‑2)|Yes (requires modifying output heads during pre-training)|
|Training cost|Low (can use off-the-shelf models as draft and target models)|Moderate (fine‑tune extra heads)|High (requires pre-training)|
|Inference speedup (observed)|∼2×–3×|∼2.2×–3.6× (typically 2.3×–2.8×)|∼3× (4‑token), up to ∼6× (8‑token draft window)|
|Output quality|Identical to base model|High (rejection + typical acceptance schemes)|Matches next‑token head|
|Deployment ease|Moderate (dual‑model system)|High (single model with extra heads)|High (single model if integrated from pretraining)|
|Memory overhead (training)|High (two model states / KV‑cache)|Low (single trunk + small head layers)|Efficient (O(V+d) peak memory)|
|Batch‑size friendliness|High|Optimized for batch size = 1|High|
|Implementation maturity|Widely used since 2022 (T5, GPT)|Early adoption in LLMs like Vicuna, Zephyr|[DeepSeek V3](https://arxiv.org/html/2412.19437v1)|
### When to Use Each Technique

- **Draft Model (Leviathan-style speculative decoding)**:
    
    - Ideal when you can’t modify or retrain the base model.
    - Suitable for legacy systems or commercial APIs.
    - Offers “plug-and-play” inference acceleration with minimal integration overhead.
    - Best when a strong, compact draft model is already available.
- **Medusa (Cai et al., 2024)**:
    
    - Ideal for single-user interactive settings (e.g., chatbots).
    - Offers fine-grained control via Medusa-1 (frozen backbone) or Medusa-2 (joint fine-tuning).
    - Introduces **tree attention** to optimize speculative token verification.
    - Can outperform others when output diversity or control is key.
- **Multi-token Prediction Heads (Gloeckle et al., 2024)**:
    
    - Recommended during full model pretraining.
    - Best for institutions training models from scratch or at scale.
    - Enables **self-speculative decoding** with minimal architectural footprint.
    - Very efficient for longer inputs or batch decoding workloads.

### Implementation Details

- **Draft-based Implementation**:
    
    - Ensure the draft model is **close enough** in distribution to the main model; divergence kills speedup.
    - Batch speculative runs and base model verifications.
    - Use caching (KV cache reuse) to reduce redundant computations.
- **Multi-token Heads Implementation**:
    
    - Train with n-token loss: each head predicts future token i.
    - Use gradient checkpointing or staggered backprop to control memory.
    - At inference, use blockwise or greedy speculative decoding.
- **Medusa Implementation**:
    
    - Add feedforward speculative heads:
        
        pkt=softmax(W2k@(SiLU(W1k@ht)+ht))
        
    - For **tree attention**, modify attention masks to ensure tokens only see ancestors.
    - Use **typical acceptance** scheme to boost accepted token length without complex sampling.

### Empirical Results Snapshot

- From [Better & Faster Large Language Models via Multi-token Prediction](https://arxiv.org/abs/2404.19737) by Gloeckle et al. (2024), using 4-token prediction on a 7B model:
    
    - 3× faster inference.
    - +4% higher pass@1 on HumanEval (code generation).
    - Optimal token prediction window (n) varies: 4 is best for natural language, 8 for byte-level models.
- From [Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads](https://arxiv.org/abs/2401.10774) by Cai et al. (2024), Medusa on Vicuna-7B:
    
    - 2.3–2.8× speedup.
    - Quality preserved via training schemes (especially Medusa-2).
    - Compatible with quantized backbones (QLoRA).

### Key Takeaways

- **Speed vs Simplicity**: Draft-based methods are simpler but less efficient long-term. Integrated heads unlock better scaling.
- **Training Budget Matters**: If you’re training from scratch, invest in multi-token or Medusa heads.
- **Serving Constraints**: For distributed serving or edge deployment, Medusa-1 or next-token heads provide clean integration.

## Implementation Deep Dive: How to Build Speculative Decoders

- This section focuses on the nuts and bolts of implementing speculative decoding. We cover architecture layouts, essential training routines, memory-saving tricks, and reference code patterns for each of the three major approaches.

### Draft Model-Based Speculative Decoding

- This method involves using two models:
    
    - **Target model**: The large, accurate LLM whose output must be preserved.
    - **Draft model**: A smaller model trained to approximate the target model’s predictions.
- **Architecture Overview**:
    
    - The following figure from the paper shows the workflow of draft model-based speculative decoding: proposal, parallel verification, selective acceptance. In the case of unconditional language modeling, each line represents one iteration of the algorithm. The green tokens are the suggestions made by the approximation model (here, a GPT-like Transformer decoder with 6M parameters trained on lm1b with 8k tokens) that the target model (here, a GPT-like Transformer decoder with 97M parameters in the same setting) accepted, while the red and blue tokens are the rejected suggestions and their corrections, respectively. For example, in the first line the target model was run only once, and 5 tokens were generated.
    
    ![](https://aman.ai/images/papers/SD.jpg)
    
- Each decoding step proceeds as follows:
    
    1. Generate a speculative prefix of γ tokens using the draft model (e.g., γ = 4).
    2. Run the target model in parallel to verify each token.
    3. Accept matching tokens; reject mismatches and resume standard decoding from there.
- **Key Implementation Elements**:
    
    - **Speculative Sampling**: Uses rejection sampling to ensure distributional equivalence:
        
        ![](https://aman.ai/images/copy.png)
        
        `def accept_token(p_large, p_draft, x):     if p_draft[x] <= p_large[x]:         return True     else:         accept_prob = p_large[x] / p_draft[x]         return random.random() < accept_prob`
        
    - **Parallel Verification**: Run γ + 1 parallel forward passes of the target model:
        
        ![](https://aman.ai/images/copy.png)
        
        `with torch.no_grad():     logits = target_model(prefix + draft_tokens)     verified_probs = softmax(logits)`
        
    - **Fallback Correction**: If a token is rejected, sample again from an adjusted distribution:
        
        ![](https://aman.ai/images/copy.png)
        
        `residual = torch.clamp(p_large - p_draft, min=0) residual /= residual.sum() next_token = torch.multinomial(residual, num_samples=1)`
        
- **Optimization Tip**: Cache activations across reused prefixes to avoid redundant computation.

### Medusa: Tree Attention + Parallel Heads

- Medusa extends multi-token decoding with a novel attention mechanism that verifies multiple speculative paths simultaneously.
    
- **Architecture Overview**:
    
    - The following figure from the paper shows the proposed tree attention in Medusa: parallel candidates from multiple heads form branches that are verified simultaneously.
    
    ![](https://aman.ai/primers/ai/assets/speculative-decoding/Medusa_treeattn.jpg)
    
    - Multiple lightweight Medusa heads project from the last hidden state.
    - Each head proposes tokens at future positions (t+1, t+2, …, t+K).
    - Tree-structured attention masks control information flow to ensure correctness.
- **Medusa Head Definition**:
    
    ![](https://aman.ai/images/copy.png)
    
      `def medusa_head(h_t, W1_k, W2_k):       ff_out = F.silu(W1_k @ h_t) + h_t       return softmax(W2_k @ ff_out)`
    
- - W1k is initialized as zero, W2k cloned from LM head.
- **Tree Attention Implementation**:
    
    - Construct Cartesian product of top-k predictions from each head.
    - Use attention mask that only allows intra-branch communication.
    - Modify positional encodings for tree-based candidate verification.
- **Candidate Verification**:
    
    ![](https://aman.ai/images/copy.png)
    
      `# Assume 2 heads with top-2 and top-3 predictions   # Generate 6 branches, verify each in parallel   mask = build_tree_attention_mask(branch_structure)   attention_output = transformer_with_mask(input_ids, mask)`
    
- **Acceptance Strategy**:
    
    - **Rejection sampling** ensures fidelity.
    - **Typical acceptance** (heuristic cutoff on deviation from target) boosts speed.

### Multi-Token Prediction Heads

- This approach modifies the LLM architecture to predict n future tokens at once during training.
    
- **Architecture Overview**:
    
    - The following figure from the paper shows the implementation structure of multi-token prediction: one trunk, multiple future-predicting heads, and staged loss computation.
    
    ![](https://aman.ai/images/papers/MTP.jpg)
    
    - A shared transformer trunk generates a hidden state.
    - n lightweight output heads decode tokens t+1 to t+n.
- **Model Structure**:
    
    ![](https://aman.ai/images/copy.png)
    
      `# Trunk   z = transformer_trunk(x)    # Heads   logits = [head_i(z) for i in range(n)]   outputs = [softmax(logit) for logit in logits]`
    
    - Each head minimizes its own cross-entropy loss:
        
        ![](https://aman.ai/images/copy.png)
        
          `loss = sum([F.cross_entropy(logits[i], target[i]) for i in range(n)])`
        
- **Memory-Efficient Training**:
    
    - **Sequential gradient computation** for each head reduces memory:
        
        ![](https://aman.ai/images/copy.png)
        
        `for head in heads:     output = head(z)     loss = F.cross_entropy(output, target)     loss.backward(retain_graph=True)`
        
    
    **Inference Options**:
    
    - Use the next-token head for traditional generation.
    - Use the other heads to propose speculative sequences for greedy decoding (e.g., blockwise).

## Future Directions

- The field is still rapidly evolving. What began with speculative sampling is now branching into hybrid pipelines, adaptive acceptance, and tree-structured reasoning paths. With integration into quantized and edge-deployable models, speculative decoding is becoming not just an optimization—but a design paradigm for future LLM systems.
- The core techniques of speculative decoding have opened the door to a range of optimization opportunities for LLM inference. In this section, we explore emerging variants, hybrid models, and promising research directions that could further accelerate decoding while maintaining output fidelity.

### Hybrid Approaches: Combining Draft + Head

- Some systems now combine **draft models** with **multi-token or Medusa heads** to maximize acceptance rates and throughput.
    
- **Motivation**:
    
    - Use a draft model for a long speculative prefix.
    - Use Medusa or multi-token heads to verify batches of predictions instead of verifying token-by-token.
- **Example Pipeline**:
    
    1. Draft model proposes γ tokens.
    2. Medusa-style heads are used within the large model to validate candidate branches.
    3. Longest valid candidate is accepted.
- **Advantages**:
    
    - Combines high-quality approximation from draft with structural verification efficiency.
    - Supports deeper pipelines (e.g., hierarchical draft-check loops).
    - Naturally extensible to distributed and batched decoding.

### Integration with Quantization & Pruning

- Speculative decoding can synergize with model compression techniques:
    
    - **Quantized Models** (e.g., QLoRA, GPTQ):
        - Medusa heads can be trained/fine-tuned atop a frozen quantized model. Even the trunk used in multi-token prediction can be quantized (as in Medusa-1).
    - **Pruned Heads**:
        - Lightweight speculative heads use <0.1% of model parameters. This makes them ideal candidates for post-training head-specific pruning or low-rank approximations.
    - **Shared KV Caches**:
        - As seen in IBM’s PyTorch implementation, speculative tokens and trunk outputs can reuse the same attention cache with minimal overhead by adapting the paged attention kernel.

### Speculative Decoding for Byte-Level Model

- Recent experiments show that speculative decoding is **especially effective for byte-level tokenization** models.
    
- **Why?**
    
    - Byte-level tokenizers (e.g., Tiktoken with vocab size 256) produce longer sequences for the same semantic content.
    - This increases the number of decoding steps per input and exacerbates autoregressive latency.
- **Findings from [Better & Faster Large Language Models via Multi-token Prediction](https://arxiv.org/abs/2404.19737) by Gloeckle et al. (2024)**:
    
    - 8-byte prediction outperforms single-token next prediction by 67% on MBPP pass@1.
    - Inference speedup of 6.4×, fully amortizing byte-level overhead.

### Beyond Decoding: Speculative Sampling for Diverse Output

- While initial work focused on greedy or top-k decoding, speculative techniques are being extended to support:
    
    - **Diverse sampling** (via top-p or temperature-controlled typical decoding)
    - **Beam search variants** (speculative beam candidates + top-scoring path verification)
    - **Stochastic acceptance** (accept “close enough” tokens under Wasserstein distance or KL threshold)
- This makes speculative decoding viable for tasks requiring diversity, such as story generation, summarization, and open-ended Q\&A.

### Future Research Directions

- Several open questions and promising directions remain:
    
    - **Speculative Training**: Can models be explicitly trained to improve speculative token acceptance rates (e.g., contrastive token alignment)? This would unify training and decoding under a shared goal.
        
    - **Reinforcement-Tuned Speculators**: How can RLHF-style alignment guide draft model predictions or head outputs for better human preference alignment?
        
    - **Adaptive Drafting**: Can models dynamically adjust the speculative prefix length based on uncertainty, entropy, or input complexity?
        
    - **Token-Free Decoding**: Recent proposals like “latent decoding” (generating hidden states directly) could be paired with speculative strategies to push inference latency even lower.


## Takeaways

- Speculative decoding represents a pivotal advancement in making LLM inference faster, more efficient, and more scalable—without compromising model accuracy or requiring massive retraining. In this primer, we’ve explored the conceptual underpinnings, design patterns, and technical implementations behind speculative decoding.
- While hybrid speculative models (e.g., Medusa + draft) offer a path to greater speed and flexibility, future systems will likely feature _dynamic, train-time-aware_ speculative inference pipelines tailored to use case and device constraints.
    
- **Takeaways**:
    - **Autoregressive inference is inherently sequential**, but speculative decoding introduces parallelism by “guessing” future tokens and verifying them.
        
    - **Three main strategies dominate**:
        
        - **Draft model decoding**: Uses a separate small model for speculative suggestions.
        - **Multi-token prediction heads**: Built into the model at pretraining time, allowing for native speculative output.
        - **Medusa**: Enhances multi-head prediction with tree attention and flexible acceptance schemes.
    - **Speedups are real and measurable**:
        
        - 2–3× (draft models),
        - 3–6× (multi-token heads),
        - 2.3–2.8× (Medusa in real-world batch-1 usage).
    - **Memory-efficient implementations** are critical to unlocking the full benefits of speculative decoding, especially when dealing with large vocabularies and long sequences.
        
    - **Use-case dependent**:
        
        - Draft models excel in low-latency deployment pipelines.
        - Medusa is great for chatbots and single-user scenarios.
        - Multi-token heads are most effective when trained from scratch.


# Model Acceleration


## Training Optimizations

### Overview

- Training optimizations for large language models (LLMs) focus on reducing computational and memory overhead during the training phase while preserving model quality. As LLMs scale in size and sequence length, traditional attention mechanisms and dense architectures become bottlenecks due to their high compute and memory requirements—most notably the quadratic complexity of self-attention.
    
- This section explores innovations aimed at accelerating training through both algorithmic and systems-level enhancements. These include:
    
    - **Memory-aware attention algorithms** like FlashAttention and FlashAttention-2 that optimize data movement between GPU memory hierarchies (e.g., from HBM to SRAM), significantly reducing memory bandwidth usage and computation time. These approaches prioritize hardware efficiency through techniques such as tiling, recomputation, and parallelization of attention blocks.
        
    - **Multi-query and grouped-query attention methods**, such as those proposed in the Fast Transformer Decoding and GQA papers, which reduce redundancy in attention heads by sharing key/value projections. These techniques are especially valuable for speeding up decoding and inference but also reduce the number of parameters and computational cost during training.
        
    - **Sparse and localized attention schemes** like those introduced in Longformer, which replace global self-attention with a combination of local windowed and task-specific global attention. This approach reduces memory consumption and compute time from quadratic to linear with respect to sequence length, enabling efficient training on longer sequences.
        
- Together, these methods represent a growing body of work that rethinks the Transformer architecture and its memory-compute tradeoffs. They aim to make LLM training more scalable, efficient, and accessible—paving the way for faster iterations and the deployment of increasingly capable models on constrained hardware. Subsequent sections provide a closer look at specific techniques and their empirical results.


### [FlashAttention](https://arxiv.org/abs/2205.14135)

- Proposed in [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135) by Dao et al. from Stanford.
- Transformers are slow and memory-hungry on long sequences, since the time and memory complexity of self-attention are quadratic in sequence length. Approximate attention methods have attempted to address this problem by trading off model quality to reduce the compute complexity, but often do not achieve wall-clock speedup. They argue that a missing principle is making attention algorithms IO-aware – accounting for reads and writes between levels of GPU memory.
- This paper by Dao et al. from Stanford in 2022 proposes FlashAttention, an IO-aware exact attention algorithm that uses tiling to reduce the number of memory reads/writes between GPU high bandwidth memory (HBM) and GPU on-chip SRAM. Specifically, FlashAttention reorders the attention computation and leverages classical techniques (tiling, recomputation) to significantly speed it up and reduce memory usage from quadratic to linear in sequence length.
- They analyze the IO complexity of FlashAttention, showing that it requires fewer HBM accesses than standard attention, and is optimal for a range of SRAM sizes. They also extend FlashAttention to block-sparse attention, yielding an approximate attention algorithm that is faster than any existing approximate attention method.
- FlashAttention trains Transformers faster than existing baselines: 15% end-to-end wall-clock speedup on BERT-large (seq. length 512) compared to the MLPerf 1.1 training speed record, 3x speedup on GPT-2 (seq. length 1K), and 2.4x speedup on long-range arena (seq. length 1K-4K).
- FlashAttention and block-sparse FlashAttention enable longer context in Transformers, yielding higher quality models (0.7 better perplexity on GPT-2 and 6.4 points of lift on long-document classification) and entirely new capabilities: the first Transformers to achieve better-than-chance performance on the Path-X challenge (seq. length 16K, 61.4% accuracy) and Path-256 (seq. length 64K, 63.1% accuracy).
- The figure below from the paper shows: (Left) FlashAttention uses tiling to prevent materialization of the large N×N attention matrix (dotted box) on (relatively) slow GPU HBM. In the outer loop (red arrows), FlashAttention loops through blocks of the K and V matrices and loads them to fast on-chip SRAM. In each block, FlashAttention loops over blocks of Q matrix (blue arrows), loading them to SRAM, and writing the output of the attention computation back to HBM. Right: Speedup over the PyTorch implementation of attention on GPT-2. FlashAttention does not read and write the large N×N attention matrix to HBM, resulting in an 7.6x speedup on the attention computation.

![](https://aman.ai/images/papers/FlashAttention.jpg)

- [Code](https://github.com/Dao-AILab/flash-attention)
- A detailed discourse on this topic is available in our [FlashAttention](https://aman.ai/primers/ai/flashattention) primer.


