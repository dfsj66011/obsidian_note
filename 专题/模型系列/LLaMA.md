
- [Training Protocol](https://aman.ai/primers/ai/LLaMA/#training-protocol)
- [Results](https://aman.ai/primers/ai/LLaMA/#results)
- [Llama 3](https://aman.ai/primers/ai/LLaMA/#llama-3)


### 一、引言

- LLaMA 是 Meta AI（FAIR）发布的一系列语言模型。
- 与 ChatGPT 或 GPT-4 不同，LLaMA 是开源的。
- 需要注意的是，虽然代码公开，但模型权重仅限非商业用途，需确认使用场景后方可获取。
- 受 Chinchilla 缩放定律论文启发，LLaMA 论文提出了一组“小型”大语言模型，其参数量比 GPT-3（175B）少 10 倍以上（13B），但性能更优。更大的 65B 版本甚至超越了 PaLM-540B。简言之，该论文提出的开源小模型基于公开数据训练，性能却优于近年某些专有大模型。
- LLaMA 模型的表现优于 GPT-3，为开源社区提供了更优选择，此前的 OPT 和 BLOOM 等开源模型被认为性能不及 GPT-3。

### 二、架构设计决策

以下是 LLaMA 采用的一些架构设计决策，以提升性能并超越近期的大型语言模型；其最小规模（7B）模型在许多语言任务上与 GPT-3 表现相当。LLaMA 架构相比原始 Transformer 进行了四项改进：

1. 采用 RMSNorm 进行预归一化
2. 使用旋转位置编码（Rotary Embeddings）
3. 采用 SwiGLU 激活函数
4. 注意力机制优化

#### 2.1 预归一化

为提高训练稳定性，LLaMA 对每个 Transformer 子层的输入进行归一化，而非归一化输出。LLaMA 采用 RMSNorm 归一化函数。

* 预归一化是一种在数据输入神经网络前对其进行归一化的技术。
- 其目标是通过归一化减少输入特征的方差和相关性，从而提高训练过程的效率和稳定性。
- 预归一化可采用多种形式，但最常见的方法是减去均值并除以训练数据集中各特征的标准差。
- 这确保每个特征的均值为零、标准差为一，使神经网络更容易学习特征间的关系，而无需过度受特征尺度或量级的影响。
- 当处理尺度或量级差异极大的特征时，预归一化尤为重要，否则可能导致神经网络过度关注某些特征而忽视其他特征。
- 通过预归一化，这些差异得以平衡，神经网络能更好地学习数据中的潜在模式和关联。

#### 2.2 SwiGLU 激活函数（Swish 门控线性单元）

LLaMA采用 SwiGLU 激活函数替代 ReLU 非线性单元，该函数由 Shazeer（2020）在论文《Swish-Gated Linear Units for Neural Network Function Approximation》中提出，能提供更优性能。  

SwiGLU 基于 Swish 激活函数设计，后者是一种平滑且非单调的函数，在某些神经网络结构中表现优于 ReLU 和 sigmoid 等常用激活函数。  $$
\text{SwiGLU}(x)=x \times \text{sigmoid}(\beta x)+(1−\text{sigmoid}(\beta x)) x  $$
实验结果表明，在图像分类和语言建模任务中，SwiGLU 的性能可能超越 ReLU、Swish 和 GELU（高斯误差线性单元）等其他激活函数。  但 SwiGLU 的效果取决于具体架构和数据集，因此并非适用于所有应用场景的最佳选择。

#### 2.3 旋转位置编码  

LLaMA 没有采用原始 Transformer 中使用的绝对位置编码来嵌入信息的序列性概念，而是在网络每一层使用了苏剑林等人（2021）在《旋转位置编码》中提出的旋转位置编码（RoPE）。  

旋转编码的核心思想是为深度学习模型中的位置编码引入额外的结构。位置编码用于将序列中每个元素（如句子中的单词）的位置表示为向量，然后与相应的元素嵌入结合，形成模型的输入。在传统的位置编码中，表示不同位置的向量彼此正交。然而，这种正交性可能导致模型中的某些对称性，从而限制其表达能力。

旋转编码通过在不同维度的位置编码之间引入相位偏移来解决这一问题。这种相位偏移是通过一个基于高维空间旋转特性的特殊形式矩阵实现的。生成的编码不再正交，但保留了某些旋转对称性，从而增强了模型的表达能力。实验结果表明，旋转编码可以提升深度学习模型在机器翻译和语言建模等任务上的性能。

#### 2.4 注意力优化技术

LLaMA 同时采用了内存[高效注意力机制](https://arxiv.org/abs/2112.05682)和 FlashAttention 技术，这两种技术通过高效的因果多头注意力实现方案来降低内存消耗和运行时间。前者提出了一种极其简洁的注意力算法，该算法仅需 $O(1)$ 级内存（与序列长度无关），其自注意力扩展版本也仅需 $O(\log n)$ 级内存。

该技术通过不存储注意力权重，并跳过因语言建模任务的因果特性而被掩码处理的键/查询分数计算来实现优化。这有效提升了训练效率和模型收敛速度。这也意味着模型很可能具备大幅扩展上下文长度的潜力。

![|500](https://aman.ai/primers/ai/assets/llama/attention.png)

### 三、视觉摘要

Sebastian Raschka 提供的视觉总结详述了 LLaMA 实现这一性能所采用的方法：预归一化、SwiGLU激活函数和旋转嵌入。Sebastian Raschka 还指出，图中展示训练损失与训练 token 数量的关系时呈现陡峭的负斜率，这表明作者本应将模型训练超过 1-2 个周期。

![|650](https://aman.ai/primers/ai/assets/llama/llama.jpeg)

### 五、模型变种

LLaMA 提供多种参数规模的版本（7B, 13B, 33B, 和 65B 参数）。

### 六、训练协议

LLaMA 65B 和 LLaMA 33B 的训练使用了 1.4T token，而 LLaMA 7B 的训练使用了 1T token。 LLaMA与大多数语言模型的训练方式相同，通过输入单词序列并预测下一个单词进行训练。该模型训练涉及 20 种不同语言，重点关注拉丁字母和西里尔字母。


## Results

![](https://aman.ai/primers/ai/assets/llama/1.png)

- LLaMA-13B outperforms GPT-3 (175B) on most benchmarks.
- LLaMA-65B is competitive with the best models, Chinchilla70B and PaLM-540B.

## Llama 3

- [Llama 3](https://ai.meta.com/blog/meta-llama-3/) by Meta offers substantial enhancements and novelties in the capabilities of the model. An analysis of its development illustrates a significant advance over its predecessors in multiple aspects, reflecting a sustained effort towards refining language model technology.
- **Tokenizer Enhancements**: Llama 3 has seen a notable expansion in its tokenizer capacity, increasing from 32,000 tokens in Llama 2 to 128,000 tokens. This enlargement allows for more efficient sequence compression, with a reduction in sequence length by approximately 15%, thus potentially enhancing downstream task performance due to a denser information representation.
- **Architectural Developments**: Despite no radical changes in the overall architecture from Llama 2, all variants of Llama 3 now incorporate Grouped Query Attention (GQA), a scheme previously reserved for larger models. GQA facilitates a more compact representation of the keys/values in the Attention mechanism, significantly reducing the footprint of the Key-Value (KV) cache during inference, thus optimizing computational efficiency.
- **Sequence Length Capacity**: The context window for Llama 3 has been increased to 8,192 tokens, up from 4,096 in Llama 2 and 2,048 in Llama 1. While this expansion is modest compared to the capabilities of models like GPT-4, which supports up to 128,000 tokens, it marks a progressive improvement, with potential future enhancements in subsequent versions.
- **Training Data Scope**: The training dataset size for Llama 3 has escalated dramatically to 15 trillion tokens, a substantial increment from the 2 trillion tokens used for Llama 2. This dataset not only focuses on English but also includes a 5% representation from over 30 different languages, incorporating a richer diversity in data, albeit still predominantly English-centric.
- **Scaling Laws and Efficiency**: The utilization of a 15 trillion token dataset to train a model with 8 billion parameters represents an unconventional approach by current standards, where such large datasets are typically reserved for much larger models. Meta’s approach indicates a shift towards maximizing model capability and efficiency beyond traditional compute-to-performance ratios, as indicated by scaling laws such as those outlined in the Chinchilla study.
- **Systems and Infrastructure**: Llama 3’s training was executed on a system of 16,000 GPUs, achieving an observed throughput of 400 TFLOPS. This figure suggests approximately 40% utilization of the peak theoretical output based on NVIDIA’s stated capabilities for the H100 GPUs at fp16 precision, acknowledging the adjustments required for realistic sparsity conditions.
- **Model “Strength”**: Incorporating insights from the model card for Llama 3, the performance comparison between the 8 billion parameter version (Llama 3 8B) and the larger 70 billion parameter version (Llama 2 70B) reveals intriguing nuances. Notably, Llama 3 8B, which was trained with a staggering 15 trillion tokens, exhibits comparable performance to Llama 2 70B, which was trained with just 2 trillion tokens. This discrepancy in training data volume underscores the significant impact of extensive training on model performance.
    - **Performance Metrics Based on Computational Training**: The metrics defining the strength of Llama 3 8B highlight its computational intensity. The model accrued approximately 1.8×10241.8×1024 floating point operations (FLOPs) over 1.3 million GPU hours, assuming a throughput of 400 TFLOPS. In contrast, an alternative calculation method estimating FLOPs as 6ND6ND (where NN is the number of parameters and DD is the number of tokens) yields approximately 7.2×10237.2×1023 FLOPs, suggesting some variability in these estimates. Prioritizing the more comprehensive GPU hours calculation, Llama 3 8B’s total computational input stands around 2×10242×1024 FLOPs.
    - **Comparative Analysis with Llama 3 70B and 400B Models**: For Llama 3 70B, the computational input is substantially higher, reaching approximately 9.2×10249.2×1024 FLOPs calculated over 6.4 million GPU hours, which aligns closely with the alternative method’s estimate of 6.3×10246.3×1024 FLOPs. Should the 400 billion parameter model train on the same dataset, the expected computational investment would scale up to approximately 4×10254×1025 FLOPs. This projection places it just below the threshold outlined in regulatory frameworks such as the recent Biden Executive Order, which sets a reporting requirement at 1×10261×1026 FLOPs.
    - **The Significance of Data Quality and Comprehensive Model Evaluation**: Beyond raw computational power, the quality of training data plays a critical role in shaping a model’s effectiveness. The integration of diverse and high-quality data can significantly enhance model performance, emphasizing the importance of not reducing the model’s capability to merely its computational input. However, when simplifying the comparison across models, total FLOPs provide a useful measure, amalgamating the scale of the model and the extent of its training into a singular metric indicative of its overall ‘strength.’
- In conclusion, Llama 3’s architecture and training regimen illustrate Meta’s strategic emphasis on maximizing model efficiency and performance through both scaled parameter counts and extensive training, setting new benchmarks in the landscape of language models. This approach not only boosts performance but also extends the model’s applicability and utility across a wider range of tasks and scenarios.
- **Conclusion**: The advancements in Llama 3 underscore Meta’s commitment to pushing the boundaries of what small yet powerfully trained models can achieve. This strategy not only enhances the capabilities of such models but also broadens their applicability in real-world scenarios, paving the way for future innovations in machine learning landscapes. Moreover, the anticipation surrounding the potential release of a 400 billion parameter model highlights the community’s eagerness for more robust, accessible AI tools, reflecting a growing trend towards democratizing high-performance computational models.
- [Blog](https://ai.meta.com/blog/meta-llama-3/); [Model Demo](https://meta.ai/); [Model Card](https://github.com/meta-llama/llama3/blob/main/MODEL_CARD.md); [TorchTune](https://github.com/pytorch/torchtune)

-  [](https://github.com/amanchadha)|  [](https://citations.amanchadha.com/)|  [](https://twitter.com/i_amanchadha)|  [](mailto:hi@aman.ai)| 

[www.amanchadha.com](https://www.amanchadha.com/)