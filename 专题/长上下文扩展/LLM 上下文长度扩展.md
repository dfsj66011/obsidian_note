

- [Background: NTK, NTK-Aware, and Dynamic NTK](https://aman.ai/primers/ai/context-length-extension/#background-ntk-ntk-aware-and-dynamic-ntk)
    - [NTK (Neural Tangent Kernel)](https://aman.ai/primers/ai/context-length-extension/#ntk-neural-tangent-kernel)
    - [NTK-Aware Method](https://aman.ai/primers/ai/context-length-extension/#ntk-aware-method)
    - [Dynamic NTK Method](https://aman.ai/primers/ai/context-length-extension/#dynamic-ntk-method)
- [Related Papers](https://aman.ai/primers/ai/context-length-extension/#related-papers)
    - [Extending Context Window of Large Language Models Via Positional Interpolation](https://aman.ai/primers/ai/context-length-extension/#extending-context-window-of-large-language-models-via-positional-interpolation)
    - [YaRN: Efficient Context Window Extension of Large Language Models](https://aman.ai/primers/ai/context-length-extension/#yarn-efficient-context-window-extension-of-large-language-models)
    - [LongLoRA: Efficient Fine-tuning of Long-Context Large Language Models](https://aman.ai/primers/ai/context-length-extension/#longlora-efficient-fine-tuning-of-long-context-large-language-models)
    - [LongQLoRA: Efficient and Effective Method to Extend Context Length of Large Language Models](https://aman.ai/primers/ai/context-length-extension/#longqlora-efficient-and-effective-method-to-extend-context-length-of-large-language-models)
    - [MemGPT: Towards LLMs As Operating Systems](https://aman.ai/primers/ai/context-length-extension/#memgpt-towards-llms-as-operating-systems)
    - [LM-Infinite: Simple On-The-Fly Length Generalization for Large Language Models](https://aman.ai/primers/ai/context-length-extension/#lm-infinite-simple-on-the-fly-length-generalization-for-large-language-models)
    - [LLM Maybe LongLM: Self-Extend LLM Context Window Without Tuning](https://aman.ai/primers/ai/context-length-extension/#llm-maybe-longlm-self-extend-llm-context-window-without-tuning)
    - [In Search of Needles in a 11M Haystack: Recurrent Memory Finds What LLMs Miss](https://aman.ai/primers/ai/context-length-extension/#in-search-of-needles-in-a-11m-haystack-recurrent-memory-finds-what-llms-miss)
- [Citation](https://aman.ai/primers/ai/context-length-extension/#citation)

## Overview

LLMs 在各领域的广泛应用突显了一个重大挑战：其预定义的上下文长度限制。这一局限影响了效率，尤其是在需要处理大量文档或大规模对话的应用场景中。虽然直接训练 LLMs 以处理更长上下文是一种潜在的解决方案，但这并不总是高效，且可能耗费大量资源。

上下文长度是决定大语言模型效能的关键参数。将上下文长度提升至 10 万是一项显著成就。然而，随着时间推移和技术发展，这一成就的价值可能会被重新评估。

让我们来看看现有的解决方案如何应对上下文长度限制：

* **摘要与链式提示：​**​ 当前的方法通常涉及使用复杂的摘要技术与链式提示相结合。
* **向量数据库**：用于存储自定义文档的嵌入向量，随后可根据相似度指标进行查询。
* **使用自定义数据进行微调**：这种方法虽然有效，但由于某些商业大语言模型的限制以及开源大语言模型的复杂性，并非普遍适用。
* **定制化大语言模型**：开发更小、以数据为中心的大语言模型是另一种解决方案，尽管这本身也带来了一系列挑战。

## 扩展上下文长度的优势

- 具有更长上下文长度的 LLM 可以通过处理用户特定数据而无需重新校准模型，提供更加定制化和高效的交互。这种利用内存处理的即时学习方法，有望提高准确性、流畅性和创造性。
- 类比说明：就像计算机内存能保留软件应用的运行状态一样，更长的上下文窗口让 LLM 能持续处理更广泛的用户数据。

## 背景：插值及其如何增加上下文长度

**什么是非整数位置的插值？​**

插值是一种数学方法，用于确定两个已知值之间的未知值。在 Llama 2 模型的背景下，“在非整数位置进行插值”指的是一种技术，即使数据超出了模型的原始上下文窗口，也可以通过调整标记（数据或文本片段）的位置来适应这一窗口。这种方法不是使用整数来表示位置，而是利用介于整数之间的值（非整数）。

通过使用这种插值方法，Llama 2 能够处理远超其设计容量或原始窗口大小的文本输入。这种技术的优势在于，尽管处理的数据块更大，模型的性能却不会受到影响。它能够处理更多数据，同时仍保持高效运行。

简单来说：Meta 采用了一种巧妙的方法，让 Llama 2 在不降低速度或效率的情况下一次性处理更多数据。他们通过调整数据位置的计算方式，使用介于整数之间的值来实现这一点。

![|500](https://aman.ai/primers/ai/assets/context-length/2.jpeg)


### Paper: 通过位置插值扩展大型语言模型的上下文窗口

* https://arxiv.org/pdf/2306.15595.pdf
* meta (2023)

本文介绍了一种称为位置插值（PI）的技术，旨在扩展大型语言模型（如 Llama）的上下文长度，同时不损害其性能。LLM 使用位置编码（如 RoPE）来表示序列中标记的顺序。然而，直接在更长的上下文上对这些模型进行微调可能会效率低下且效果不佳，尤其是在将上下文长度大幅扩展（例如 8 倍）时。

PI 背后的核心洞见是，将位置编码外推到训练范围之外可能导致不稳定和超出分布范围的注意力分数。相反，PI 通过在训练过的整数步长之间进行插值，来创建平滑且稳定的位置编码。为此，PI 在计算位置编码之前会先对位置索引进行降尺度处理。例如，若原始上下文长度为 2048，PI 会将索引范围从 \[0, 4096\] 重新缩放至 \[0, 2048\]，使其与原始长度匹配。这种方法实际上是在原始整数步长之间对位置编码进行插值，从而减小最大相对距离，使注意力分数更加稳定。

在微调过程中，模型能快速适应内插位置编码，其稳定性优于外推式编码。作者证明内插注意力分数的上界远小于外推注意力分数，从而确保模型行为保持稳定且可预测。实验表明，PI 仅用 1000 次训练步数就成功将 Llama-7B 等模型扩展至可处理长达 32768 的上下文长度。在语言建模、问答和检索等多项任务上的评估证实，扩展后的模型能有效利用长上下文，且不会牺牲短上下文环境下的性能表现。

因此，位置插值提供了一种简单而有效的方法来扩展像 Llama 这样的大型语言模型的上下文长度。通过缩小位置索引并在训练过的整数步长之间进行插值，位置插值生成了平滑且稳定的位置编码，使模型能够适应更长的上下文，同时保持稳定性和性能。

该技术最初由 Reddit 用户 u/emozilla 提出，名为 [“动态缩放 RoPE 无需微调即可进一步提升长上下文 Llama 模型的性能”](https://www.reddit.com/r/LocalLlama/comments/14mrgpr/dynamically_scaled_rope_further_increases/)，通过动态插值 RoPE 来表示更长的序列，同时保持性能不变，使我们能够在无需微调的情况下扩展模型的上下文长度。虽然开箱即用效果良好，但通过额外微调还能进一步提升性能。借助 RoPE 缩放技术，企业现在可以轻松将开源大语言模型的上下文长度扩展到适合其特定用例的范围。

---

这篇帖子内容如下：

当 u/kaiokendev 首次发布关于线性插值 RoPE 以处理更长序列的帖子时，我（以及其他一些人）曾想过是否可以根据序列长度动态选择正确的缩放参数，而不是必须在最大序列长度和较短序列性能之间做出固定的权衡。我的想法是，在前 2k 上下文中使用精确的位置值（毕竟，为什么要破坏一个好东西？），然后在模型逐个生成 token 时，为每个新的序列长度重新计算位置向量。本质上，将缩放比例设置为原始模型上下文长度除以当前序列长度。这样做的效果是随着序列长度的增加而缓慢增加缩放比例。

我做了一些实验，发现这种方法性能非常强，远优于简单的线性插值。当 [u/bloc97](https://www.reddit.com/user/bloc97/) 发布他的 NTK 感知方法时，其性能表现更接近这种动态线性缩放。与动态线性缩放相比，NTK 感知方法在较短序列上的困惑度较高，但在序列长度尾部表现更好。遗憾的是，它也像常规 RoPE 和静态线性缩放一样，会出现灾难性的困惑度暴增问题。

NTK-Aware 的主要超参数是 $\alpha$。与静态线性缩放类似，它代表了短序列/长序列性能之间的权衡。于是我想，为什么不将同样的动态缩放方法应用于 NTK-Aware 呢？对于动态 NTK，$\alpha$ 的缩放比例设置为（$\alpha$ * 当前序列长度 / 原始模型上下文长度）-（$\alpha-1$ ）。其核心思想同样是随着序列长度的增加动态调整超参数。

----

Hugging Face Transformers 现已支持 RoPE 缩放（旋转位置嵌入）技术，可扩展 Llama、GPT-NeoX 或 Falcon 等大型语言模型的上下文长度。因此，从本质上讲，RoPE 缩放会根据输入长度动态调整相对位置差异，就像一根可以伸缩的绳子。


## 背景：NTK、NTK 感知和动态 NTK

让我们来探讨一下 NTK 和动态 NTK 如何与扩展大型语言模型的上下文长度相关联，使其能够处理和理解更长的文本序列。

### NTK (Neural Tangent Kernel，神经正切核)

NTK 是机器学习和神经网络中的一个基本概念，它描述了神经网络，特别是深度神经网络，在特定条件下训练过程中的演变方式。从技术上讲，NTK 是一种在神经网络无限宽度极限下出现的核函数，捕捉了它们在训练期间的行为。研究人员利用 NTK 来理解神经网络的行为、收敛速度以及架构选择的影响。


### NTK-Aware Method

- Proposed in [NTK-Aware Scaled RoPE allows LLaMA models to have extended (8k+) context size without any fine-tuning and minimal perplexity degradation.](https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/) by `bloc97`.
- NTK-Aware addresses the challenge of extending the context window by preserving the model’s sensitivity to high-frequency components in the data.
- High-frequency components are crucial in language processing for capturing fine-grained details and nuances in text.
- The term “NTK” in “NTK-Aware” relates to the Neural Tangent Kernel, a theoretical framework describing how the output of a neural network changes in response to small changes in its parameters.
- When extending the context window, NTK-Aware adjusts the model to prevent the loss of sensitivity to high-frequency components. This may involve weight or architecture modifications to compensate for potential detail loss when processing longer sequences.

### Dynamic NTK Method

- Dynamic NTK interpolation, a part of YaRN, dynamically adapts the model’s attention mechanism based on input sequence length.
- Instead of a fixed setting for extended contexts, the model adjusts its processing according to the actual sequence length.
- For shorter sequences near the training length, adjustments are minimal, but as sequence length increases, Dynamic NTK scales the adaptations.
- NTK-Aware focuses on specific adjustments to preserve high-frequency information in extended contexts, while Dynamic NTK offers flexibility and efficiency by tailoring adjustments to varying context sizes. Both methods enable language models to handle longer text sequences beyond their original training limits. Dynamic NTK mechanism thus interpolates frequencies unevenly, keeping high frequencies intact.
- Dynamic NTK interpolation is a key component of [YaRN](https://aman.ai/primers/ai/context-length-extension/#yarn-efficient-context-window-extension-of-large-language-models) that empowers language models to handle extended context windows with improved efficiency and performance.

## Related Papers



### [YaRN: Efficient Context Window Extension of Large Language Models](https://arxiv.org/abs/2309.00071)

- This paper by Peng et al. from Nous Research, EleutherAI, and the University of Geneva, proposes Yet Another RoPE extensioN method (YaRN) to efficiently extend the context window of transformer-based language models using Rotary Position Embeddings (RoPE).
- The authors address the limitation of transformer-based language models, specifically their inability to generalize beyond the sequence length they were trained on. YaRN demonstrates a compute-efficient way to extend the context window of such models, requiring significantly fewer tokens and training steps compared to previous methods.
- YaRN enables Llama models to effectively utilize and extrapolate to context lengths much longer than their original pre-training would allow. This method surpasses previous state-of-the-art approaches in context window extension.
- The paper details various technical aspects of YaRN, including its capability to extrapolate beyond the limited context of a fine-tuning dataset. The models fine-tuned using YaRN have been reproduced online, supporting context lengths up to 128k.
- YaRN introduces an innovative technique known as “Dynamic NTK” (Neural Tangents Kernel) interpolation, which modifies the attention mechanism of the model. This dynamic scaling allows the model to handle longer contexts without extensive retraining. By doing so, YaRN surpasses previous approaches in context window extension and significantly reduces the computational resources required. Put simply, Dynamic NTK is designed to address the challenge of extending the context window of transformer-based language models using Rotary Position Embeddings (RoPE). It achieves this by dynamically scaling the attention mechanism of the model, allowing it to efficiently process longer text sequences without requiring extensive retraining.
- Dynamic NTK interpolation modifies the traditional attention mechanism to adapt to extended contexts, ensuring that the model can effectively utilize and extrapolate to context lengths much longer than its original pre-training would allow. This dynamic scaling approach optimizes the use of available resources and computational power.
- Dynamic NTK interpolation is a key component of YaRN that empowers language models to handle extended context windows with improved efficiency and performance, making it a valuable advancement in the field of large language models.
- Additionally, YaRN incorporates a temperature parameter that affects the perplexity across different data samples and token positions within the extended context window. Adjusting this temperature parameter modifies the attention mechanism, enhancing the model’s ability to handle extended context lengths efficiently.
- Extensive experiments demonstrate YaRN’s efficacy. For instance, it achieves context window extension of language models with RoPE as the position embedding, using only about 0.1% of the original pre-training corpus, a significant reduction in computational resources.
- The following figure from the paper illustrates that evaluations focus on several aspects, such as perplexity scores of fine-tuned models with extended context windows, the passkey retrieval task, and performance on standardized LLM benchmarks. YaRN models show strong performance across all contexts, effectively extending the context window of Llama 2 models to 128k. The following figure from the paper illustrates the sliding window perplexity (S = 256) of ten 128k Proof-pile documents truncated to evaluation context window size.

![](https://aman.ai/images/papers/YaRN.jpg)

- The paper concludes that YaRN improves upon all existing RoPE interpolation methods and acts as a highly efficient drop-in replacement. It preserves the original abilities of fine-tuned models while attending to very large context sizes and allows for efficient extrapolation and transfer learning under compute-constrained scenarios.
- The research illustrates YaRN as a significant advancement in extending the context window of large language models, offering a more compute-efficient approach with broad implications for model training and performance.
- [Code](https://github.com/jquesnelle/yarn)

### [LongLoRA: Efficient Fine-tuning of Long-Context Large Language Models](https://arxiv.org/abs/2309.12307)

- This paper by Chen et al. from CUHK and MIT presents LongLoRA, an efficient fine-tuning approach that extends the context sizes of pre-trained large language models (LLMs) during fine-tuning, with limited computation cost.
- Typically, training LLMs with long context sizes is computationally expensive, requiring extensive training hours and GPU resources. For example, training on the context length of 8192 needs 16x computational costs in self-attention layers as that of 2048.
- LongLoRA speeds up the context extension of LLMs in two aspects. On the one hand, although dense global attention is needed during inference, fine-tuning the model can be effectively and efficiently done by sparse local attention. The proposed shift short attention ($$S^2$-attention) effectively enables context extension, leading to non-trivial computation savings with similar performance to fine-tuning with vanilla attention. Particularly, it can be implemented with only two lines of code in training, while being optional in inference.
- $$S^2$-attention splits the context into groups and only attends within each group. Tokens are shifted between groups in different heads to enable information flow. This approximates full attention but is much more efficient.
- On the other hand, they revisit the parameter-efficient fine-tuning regime for context expansion. Notably, they find that LoRA for context extension works well under the premise of trainable embedding and normalization. LongLoRA demonstrates strong empirical results on various tasks on Llama 2 models from 7B/13B to 70B.
- LongLoRA adopts Llama 2 7B from 4k context to 100k, or Llama 2 70B to 32k on a single 8x A100 machine. LongLoRA extends models’ context while retaining their original architectures, and is compatible with most existing techniques, like FlashAttention-2.
- In addition, to make LongLoRA practical, they collect a dataset, LongQA, for supervised fine-tuning. It contains more than 3k long context question-answer pairs.
- In summary, the key ideas are:
    1. **Shift Short Attention (S2-Attn):** During fine-tuning, standard full self-attention is very costly for long contexts. S2-Attn approximates the full attention using short sparse attention within groups of tokens. It splits the sequence into groups, computes attention in each group, and shifts the groups in half the heads to allow information flow. This is inspired by Swin Transformers. S2-Attn enables efficient training while allowing full attention at inference.
    2. **Improved LoRA:** Original LoRA only adapts attention weights. For long contexts, the gap to full fine-tuning is large. LongLoRA shows embedding and normalization layers are key. Though small, making them trainable bridges the gap.
    3. **Compatibility with optimizations like FlashAttention-2:** As S2-Attn resembles pre-training attention, optimizations like FlashAttention-2 still work at both train and inference. But many efficient attention mechanisms have large gaps to pre-training attention, making fine-tuning infeasible.
    4. **Evaluation:** LongLoRA extends the context of Llama 2 7B to 100k tokens, 13B to 64k tokens, and 70B to 32k tokens on one 8x A100 machine. It achieves strong perplexity compared to full fine-tuning baselines, while being much more efficient. For example, for Llama 2 7B with 32k context, LongLoRA reduces training time from 52 GPU hours to 24 hours.
- The following table from the paper illustrates an overview of LongLoRA designs. LongLoRA introduces shift short attention during finetuning. The trained model can retain its original standard self-attention during inference. In addition to plain LoRA weights, LongLoRA additionally makes embedding and normalization layers trainable, which is essential to long context learning, but takes up only a small proportion of parameters.

![](https://aman.ai/images/papers/LongLoRA.jpg)

- The following table from the paper shows a performance and efficiency comparison between full fine-tuning, plain LoRA, and our LongLoRA. They fine-tune LLaMA2 7B on various context lengths, with FlashAttention-2 and DeepSpeed stage 2. Perplexity is evaluated on the Proof-pile test set. Plain LoRA baseline spends limited GPU memory cost, but its perplexity gets worse as the context length increases. LongLoRA achieves comparable performance to full fine-tuning while the computational cost is much less.

![](https://aman.ai/images/papers/LongLoRA2.jpg)

- [Code](https://github.com/dvlab-research/LongLoRA).

### [LongQLoRA: Efficient and Effective Method to Extend Context Length of Large Language Models](https://arxiv.org/abs/2311.04879)

- This paper by Yang from SYSU China introduces LongQLoRA, a novel method to extend the context length of large language models, particularly focusing on LLaMA2 models. The key innovation of LongQLoRA is its combination of Position Interpolation, QLoRA, and Shift Short Attention from LongLoRA to increase context lengths efficiently with limited resources.
- LongQLoRA was tested on LLaMA2 7B and 13B models, successfully extending their context length from 4096 to 8192, and even up to 12k tokens, using just a single 32GB V100 GPU and a mere 1000 fine-tuning steps. This performance was compared with other methods like LongLoRA and MPT-7B-8K, showing that LongQLoRA achieves competitive, if not superior, performance.
- The paper detailed the implementation of LongQLoRA, emphasizing its efficient fine-tuning method, which leverages quantization and low-rank adapter weights. Specifically, it sets the LoRA rank to 64 and adds these weights to all layers without training word embeddings and normalization layers. This approach significantly reduces the GPU memory footprint, enabling performance enhancements on resource-limited settings.
- The figure below from the original paper shows the evaluation perplexity of 7B models on PG19 validation and Proof-pile test datasets in evaluation context length from 1024 to 8192. All models are quantized to 4-bit in inference. LongQLoRA is finetuned based on LLaMA2-7B for 1000 steps with RedPajama dataset on a single V100 GPU. ‘LongLoRA-Full’ and ‘LongLoRA-LoRA’ mean LLaMA2-7B published by LongLoRA with full finetuning and LoRA finetuning respectively. MPT-7B-8K are better than LLaMA2, LongLoRA and LongQLoRA in context length from 1024 to 4096. LLaMA2-7B has very poor performance beyond the pre-defined context length of 8192. LongQLoRA outperforms LongLoRA-LoRA on both datasets in context length from 1024 to 8192. In context length of 8192, LongQLoRA is extremely close to LongLoRA-Full on Proof-pile test dataset, even better than MPT-7B-8K on PG19 validation dataset.

![](https://aman.ai/images/papers/LongQLoRA.jpg)

- Ablation studies were conducted to analyze the effects of LoRA rank, fine-tuning steps, and attention patterns in inference. These studies demonstrated the robustness and effectiveness of LongQLoRA across various settings and tasks.
- The authors also collected and built a long instruction dataset of 39k data points, focusing on tasks like book summarization and Natural Questions, to test LongQLoRA’s performance in both long and short context generation tasks.
- The results showed that LongQLoRA outperforms LongLoRA in several key metrics and closely matches the performance of more resource-intensive models like MPT-7B-8K. The model, training data, and code are publicly available for further research, emphasizing the paper’s contribution to the field of efficient language model training and context length extension.

### [MemGPT: Towards LLMs As Operating Systems](https://arxiv.org/abs/2310.08560)

- This paper by Packer et al. from UC Berkeley introduces MemGPT, a groundbreaking system that revolutionizes the capabilities of large language models (LLMs) by enabling them to handle contexts beyond their standard limited windows. Drawing inspiration from traditional operating systems’ hierarchical memory systems, MemGPT manages different memory tiers within LLMs to provide extended context capabilities.
- The innovative MemGPT system allows LLMs to have virtually unlimited memory, mirroring the functionality of operating systems in managing memory hierarchies. This approach enables the LLMs to handle much larger contexts than their inherent limits would usually permit. The virtual context management technique, similar to virtual memory paging in operating systems, allows for effective data movement between fast and slow memory tiers, giving the appearance of a larger memory resource.
- MemGPT consists of two primary memory types: main context (analogous to OS main memory/RAM) and external context (similar to OS disk memory). The main context is the standard context window for LLMs, while external context refers to out-of-context storage. The system’s architecture enables intelligent management of data between these contexts. The LLM processor is equipped with function calls, which help manage its own memory, akin to an operating system’s management of physical and virtual memory.
- The following figure from the paper illustrates that in MemGPT (components shaded), a fixed-context LLM is augmented with a hierarchical memory system and functions that let it manage its own memory. The LLM processor takes main context (analogous to OS main memory/RAM) as input, and outputs text interpreted by a parser, resulting either in a yield or a function call. MemGPT uses functions to move data between main context and external context (analogous to OS disk memory). When the processor generates a function call, it can request control ahead of time to chain together functions. When yielding, the processor is paused until the next external event (e.g., a user message or scheduled interrupt).

![](https://aman.ai/images/papers/MemGPT.jpg)

- One of the key features of MemGPT is its capability to easily connect to external data sources. This feature enhances the system’s flexibility and utility in various applications. Furthermore, MemGPT comes with built-in support for LanceDB (YC W22), providing scalable semantic search via archival storage. This integration significantly boosts MemGPT’s ability to retrieve and process large amounts of data efficiently.
- The following figure from the paper illustrates the deep memory retrieval task. In the example shown, the user asks a question that can only be answered using information from a prior session (no longer in-context). Even though the answer is not immediately answerable using the in-context information, MemGPT can search through its recall storage containing prior conversations to retrieve the answer.

![](https://aman.ai/images/papers/MemGPT2.jpg)

- The paper evaluates MemGPT’s performance in two critical domains: document analysis and conversational agents. In document analysis, MemGPT exceeds the capabilities of traditional LLMs by analyzing large documents that surpass the underlying LLM’s context window. For conversational agents, MemGPT demonstrates its effectiveness in maintaining long-term interactions with enhanced coherence and personalization, resulting in more natural and engaging dialogues.
- MemGPT’s versatility is further highlighted by its compatibility with many LLMs out of the box and the option to plug it into a custom LLM server. This feature ensures that MemGPT can be widely adopted and integrated into various existing LLM frameworks, enhancing their capabilities.
- Despite its pioneering approach and capabilities, the paper also notes limitations, such as reliance on specialized GPT-4 models for function calling and the proprietary nature of these models.
- Overall, MemGPT’s innovative approach, inspired by operating system techniques, represents a significant advancement in LLM capabilities. It opens new avenues for future research and applications, particularly in domains requiring massive or unbounded contexts and in integrating different memory tier technologies. The system’s ability to manage its own memory and connect to external sources, along with LanceDB support, positions MemGPT as a versatile and powerful tool in the realm of artificial intelligence.
- [Code](https://memgpt.ai/); [Blog](https://blog.lancedb.com/memgpt-os-inspired-llms-that-manage-their-own-memory-793d6eed417e)

### [LM-Infinite: Simple On-The-Fly Length Generalization for Large Language Models](https://arxiv.org/abs/2308.16137)

- This paper by Han et al. from University of Illinois Urbana-Champaign and Meta, presents LM-Infinite, a method for enabling large language models (LLMs) to generate longer texts without retraining or additional parameters.
- LM-Infinite addresses the issue of LLMs’ length generalization failures, particularly on unseen sequence lengths. The authors identify three out-of-distribution (OOD) factors contributing to this problem: (1) unseen distances, (2) unseen number of tokens, and (3) implicitly encoded absolute position. The figure below from the paper shows a diagnosis of three OOD factors in LLMs.

![](https://aman.ai/images/papers/LM-Infinite1.jpg)

- The proposed solution, LM-Infinite, introduces a ΛΛ-shaped attention mask and a distance limit during attention. This method is computationally efficient, requiring only O(n)O(n) time and space, and is compatible with various LLMs using relative-position encoding methods.
- In terms of implementation, LM-Infinite uses a sliding window approach with a ΛΛ-shaped mask for attention calculation. It allows for longer sequence handling by dynamically adjusting the window size based on token distances. The model also incorporates a novel truncation scheme to handle tokens outside the attention window, which are summarized and reused to maintain context relevance without significant computational overhead. The figure below from the paper shows: (a) LM-Infinite is a plug-and-play solution for various LLMs, consisting of a ΛΛ-shaped mask and a distance constraint during attention. (b) A conceptual model for understanding how relative position encoding works.

![](https://aman.ai/images/papers/LM-Infinite2.jpg)

- An overview of LM-Infinite is illustrated in the figure above. This simple solution consists of two components: a ΛΛ–shaped attention mask and a distance limit. As visualized in the figure, the ΛΛ-shaped attention mask has two branches: a global branch on the left and a local branch on the right. The global branch allows each token to attend to the starting nglobal nglobal  tokens if they appear before the current token. The local branch allows each token to attend to preceding tokens within nlocal nlocal  distance. Any other tokens outside these two branches are ignored during attention. Here they heuristically set nlocal =Lpretrain nlocal =Lpretrain  as equal to the training length limit. This choice includes the “comfort zone” of LLMs in attention. The selection of nglobal nglobal  has less effect on model performance, and they find that the range [10,100][10,100] is generally okay. Note that nglobal =0nglobal =0 will lead to immediate quality degradation. This design is based on the OOD factors 2 and 3 above, where they aim to control the number of tokens to be attended to, while also ensuring the inclusion of starting tokens. Theoretically, LM-Infinite can access information from a context as long as nlayer Lpretrain nlayer Lpretrain , because each deeper layer allows the attention to span Lpretrain Lpretrain  farther than the layer above.
- The distance limit involves bounding the “effective distance” d$withind$withinL_{\text {pretrain }}$. This only affects tokens that are in the global branch. In specific, recall that in relative positional encoding, the attention logit is originally calculated as w(q,k,d)$,wherew(q,k,d)$,whered$ is the distance between two tokens. Now they modify it as $$w\left(\mathbf{q}, \mathbf{k}, \min \left(d, L_{\text {pretrain }}\right)\right)$. This design is motivated by the OOD factor 1 and ensures that LLMs are not exposed to distances unseen during pre-training.
- LM-Infinite demonstrates consistent text generation fluency and quality for lengths up to 128k tokens on the ArXiv and OpenWebText2 datasets, achieving 2.72x decoding speedup without parameter updates.
- Evaluation includes experiments on ArXiv and OpenWebText2 corpora with models like LLaMA, Llama-2, MPT-7B, and GPT-J. LM-Infinite shows comparable or superior performance to LLMs fine-tuned on long sequences in terms of perplexity and BLEU/ROUGE metrics.
- The paper suggests that LM-Infinite can extend task-solving ability to longer sequences than training samples and offers potential for future work in understanding information in the masked-out attention region.

### [LLM Maybe LongLM: Self-Extend LLM Context Window Without Tuning](https://arxiv.org/abs/2401.01325)

- This paper by Jin et al. addresses the limitation of Large Language Models (LLMs) in managing long context sequences, particularly due to their training on fixed-length sequences. This results in unpredictable behavior and performance degradation when LLMs encounter long input sequences during inference. The authors propose that LLMs inherently possess the ability to handle long contexts without requiring fine-tuning.
- Their core contributions are as follows:
    1. **Self-Extend Method:** The paper introduces ‘Self-Extend’, a method aimed at leveraging the intrinsic long context capabilities of LLMs. Self-Extend uses a FLOOR operation to map unseen large relative positions (encountered during inference) to known positions from the training phase. This approach is designed to address the positional out-of-distribution issue, allowing LLMs to process longer texts coherently without additional fine-tuning.
    2. **Bi-Level Attention Information:** Self-Extend constructs bi-level attention - group level and neighbor level. Group level attention is for tokens with long distances, employing the FLOOR operation on positions. Neighbor level attention targets nearby tokens without any modifications. This design aims to retain the relative ordering of information in extended texts.
- The figure below from the paper shows the attention score matrix (the matrix before Softmax operation) of the proposed Self-Extend while a sequence of length 10 is input to a LLM with pretraining context window (LL) of length 7. The number is the relative distance between the corresponding query and key tokens. Self-Extend has two kinds of attention mechanism: for neighbor tokens within the neighbor window (wnwn, in this figure, it’s 4), it adapts the normal self-attention in transformers; for tokens out of the window, it adapts the values from the grouped attention. The group size (GG) is set to 2. After the two parts merge, the same as the normal attention, the softmax operation is applied to the attention value matrix and gets the attention weight matrix.

![](https://aman.ai/images/papers/LM-Infinite2.jpg)

- The effectiveness of Self-Extend was evaluated on popular LLMs like Llama-2, Mistral, and SOLAR across a variety of tasks, including language modeling, synthetic long context tasks, and real-world long context tasks. The results showed significant improvements in long context understanding, often surpassing fine-tuning-based methods.
- Performance was analyzed along the following dimensions:
    1. **Language Modeling:** Self-Extend was evaluated for language modeling on the PG19 dataset, which comprises long books. It effectively maintained low perplexity outside the pretraining context window for models like Llama-2-chat and Mistral.
    2. **Synthetic Long Context Tasks:** The method was assessed using the passkey retrieval task, testing the LLMs’ ability to recognize information throughout the entire length of a long text sequence.
    3. **Real Long Context Tasks:** The evaluation included benchmarks such as Longbench and L-Eval. Results showed notable performance improvements across different datasets. Self-Extend was found to perform comparably or better than several fine-tuned models, with some exceptions attributed to the specific characteristics of certain datasets.
- The paper concludes that Self-Extend, which is effective during inference without needing fine-tuning, can achieve performance on par with or superior to learning-based methods. This represents a significant advancement in enhancing the natural and efficient handling of longer contexts by LLMs.

### [In Search of Needles in a 11M Haystack: Recurrent Memory Finds What LLMs Miss](https://arxiv.org/abs/2402.10790)

- This paper by Kuratov et al. from AIRI, MIPT, and London Institute for Mathematical Sciences addresses the challenge of processing long documents using generative transformer models. The authors introduce BABILong, a new benchmark designed to assess the capabilities of models in extracting and processing distributed facts within extensive texts. This benchmark aims to simulate real-world scenarios where essential information is interspersed with large amounts of irrelevant data, making it a suitable test for long-context processing.
- The authors developed BABILong by extending the bAbI tasks to include much longer contexts, using the PG19 dataset as background text. This benchmark allows the evaluation of models on document lengths up to millions of tokens, far exceeding the typical lengths used in existing benchmarks.
- The paper evaluates GPT-4 and RAG (Retrieval-Augmented Generation) on the BABILong tasks. Results show that these models can effectively handle sequences up to 104 elements but struggle with longer contexts. Specifically, GPT-4 and RAG were unable to maintain accuracy as context length increased, demonstrating a significant drop in performance for longer inputs.
- The figure below from the paper illustrates the fact that the memory augmented transformer answers questions about facts hidden in very long texts when retrieval augmented generation fails. We create a new BABILong dataset by randomly distributing simple episodic facts inside book corpus. Common RAG method fails to answer questions because order of facts matters. GPT-4 LLM effectively uses only fraction of the context and falls short for a full 128K window. Small LM (GPT-2) augmented with recurrent memory and fine-tuned for the task generalise well up to record 11M tokens. The parameter count for GPT-4 is based on public discussions.

![](https://aman.ai/images/papers/11MNIAH.jpg)

- The figure below from the paper illustrates an example generation for BABILong dataset. Statements relevant for the question from a bAbILong sample are hidden inside a larger irrelevant texts from PG19.

![](https://aman.ai/images/papers/11MNIAH_1.jpg)

- To address the limitations of existing models, the authors propose fine-tuning GPT-2 with recurrent memory augmentations. This approach allows the model to handle tasks involving up to 11 million tokens. The recurrent memory transformer (RMT) extends the context size by segmenting sequences and processing them recurrently, resulting in linear scaling with input size.
- The RMT uses memory tokens processed alongside segment tokens to retain information from previous segments. Each memory state consists of multiple memory tokens, allowing different views of past inputs. The model updates its memory state at each step, enabling it to maintain relevant information over long contexts. Additionally, the authors implemented self-retrieval for RMT, which involves cross-attention between all past states and the current memory state.
- The figure below from the paper illustrates the Recurrent Memory Transformer with self-retrieval from memory. (a) Recurrent memory transformer encodes information from the current segment to the memory vectors `[mem]`. Memory vectors from the previous segment are passed over to the next segment and updated. (b) With self-retrieval it processes each segment sequentially and collects the corresponding memory states. Here, while processing segment 4, the model can retrieve from previous states from segments 1-3. This overcomes the memory bottleneck that can occur in purely recurrent models, similar to attention in RNNs.

![](https://aman.ai/images/papers/11MNIAH_3.jpg)

- The authors used a curriculum training strategy for RMT and RMT-R (RMT with self-retrieval), starting with short sequences and gradually introducing longer ones. The models were trained on sequences up to 16k tokens and evaluated on sequences up to 10 million tokens. The results showed that both RMT and RMT-R significantly outperformed GPT-4 on long-context tasks, maintaining high performance even for extremely long sequences.
- The paper provides an analysis of memory states and attention patterns, showing that RMT effectively stores and retrieves relevant facts from memory. The visualizations demonstrate how the model updates its memory when encountering new facts and retrieves them when needed to answer questions.
- Overall, the proposed recurrent memory transformer demonstrates a substantial improvement in processing long sequences, setting a new record for input size handled by a neural network model. The approach highlights the potential of recurrent memory mechanisms to enhance the capabilities of language models in handling extensive contexts, paving the way for future research and development in this area.
- [Code](https://github.com/booydar/babilong)

## Citation

![](https://aman.ai/images/copy.png)

`@article{Chadha2020DistilledContextLengthExtension,   title   = {LLM Context Length Extension},   author  = {Chadha, Aman and Jain, Vinija},   journal = {Distilled AI},   year    = {2020},   note    = {\url{https://aman.ai}} }`

-  [](https://github.com/amanchadha)|  [](https://citations.amanchadha.com/)|  [](https://twitter.com/i_amanchadha)|  [](mailto:hi@aman.ai)| 

[www.amanchadha.com](https://www.amanchadha.com/)