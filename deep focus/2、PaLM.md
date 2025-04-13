
在本概述中，我们将探讨 Pathways 语言模型（PaLM），这是一个使用谷歌的 Pathways 框架训练的 5400 亿参数的大型语言模型。通过消除流水线并行，该架构实现了惊人的训练吞吐量，使 PaLM 能够在更大规模的数据集上进行预训练。最终模型的少样本性能达到了最先进的水平。此外，PaLM 在解决复杂推理任务方面也表现出一定能力。简而言之，PaLM 清楚地表明，关于规模，LLM 的性能尚未达到瓶颈。只要有足够高效的训练基础设施，允许在更多数据上预训练更大规模的模型，我们就能持续看到性能的提升。

### 架构修改

除了使用改进的训练框架，PaLM 在基础的仅解码器架构上也进行了相当多的修改。这些变化大多借鉴了先前的研究，揭示了最大化 LLM 训练效率和性能的最佳实践。

**SwiGLU 激活函数**   大多数 LLM 在每层中使用的前馈神经网络结构相似。即，这个网络执行两个前馈变换（不使用偏置，并单独应用于序列中的每个 token 向量），中间使用 ReLU 激活函数。然而，后续研究表明，选择其他激活函数可能会更好。

特别是，PaLM 使用 SwiGLU 激活函数，它结合了 Swish 和 GLU 激活。该激活函数由以下公式给出：$$\text{SwiGLU}(x) = \text{Swish}(xW) \cdot xV$$其中，Swish 激活函数定义为：$$\text{Swish}(x) = x \cdot \text{Sigmoid}(\beta x)$$换句话说，SwiGLU 是输入的两个线性变换的逐元素乘积，其中一个应用了 Swish 激活。虽然这个激活函数需要进行三次矩阵乘法，但最近的研究发现，在固定计算量下，它能带来性能优势。与 ReLU 等普通激活相比，SwiGLU 似乎提供了显著的性能提升。

**并行 Transformer 块**    PaLM 还使用了并行版本的 Transformer 块，而不是常规的（串行）变体。这两种形式的区别在下图中展示。

![|500](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F34e6163f-c322-4665-812b-c89e906d6d2d_1886x988.png)

Parallel vs. serialized transformer blocks.

在模型足够大的情况下，使用并行 Transformer 块可以将训练过程加速 15%。这种加速会导致较小的 LLM（例如，80 亿参数的模型）性能略有下降，但对于全尺寸的LLM，性能通常与并行块相似。

![|500](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Faa21fa14-e4d3-4cbd-a3a1-bd6874eb512f_1676x1052.png)

**旋转位置嵌入（RoPE）**    PaLM 使用旋转位置嵌入（RoPE），而不是绝对或相对位置嵌入，RoPE 嵌入通过以下方式结合了绝对和相对位置信息：

1. 使用旋转矩阵编码绝对位置。
2. 将相对位置直接融入自注意力机制。

直观上，RoPE 在绝对和相对位置嵌入之间找到了一个中间地带。图中展示了 RoPE 始终优于其他嵌入策略。

**多查询注意力**    最后，PaLM 用一种称为多查询注意力的结构替代了典型的多头自注意力机制。多查询注意力在每个注意力头之间共享键和值向量（下图中红色部分），而不是为每个头执行单独的投影。这一改变虽然没有加快训练速度，但显著提高了 LLM 的自回归解码（即用于推理或生成）的效率。

![|200](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F4134cb5d-82b4-492b-858a-9d4d2bd62267_822x1168.png)


### [PaLM: Scaling Language Modeling with Pathways](https://arxiv.org/abs/2204.02311) 

现在，我们将概述 PaLM，这是一种拥有 5400 亿参数的密集语言模型，通过 Pathways 框架进行高效训练。PaLM 是迄今为止训练的最大密集型 LLM 之一，其高效的训练策略使其能够在超过 7000 亿个标记的大型数据集上进行预训练。这种大规模语言模型与广泛预训练语料库的结合，带来了一些有趣的结果，我们将在本节中进行探讨。

#### PaLM 如何工作？

PaLM 是一种大型 LLM，通过广泛的预训练（由高效的 Pathways 架构支持）和对基础模型架构的一些修改，实现了令人印象深刻的小样本学习性能。我们现在将概述 PaLM 的架构和训练方案的细节。

**模型**    PaLM 使用一个仅包含解码器的 Transformer，具有 5400 亿参数。然而，这个模型超越了典型的仅解码器架构，进行了几项修改：

- 在 MLP 层中使用 SwiGLU 激活函数（而不是 ReLU）。
- 在注意力层中使用多查询注意力。
- 仅使用并行 Transformer 块。
- 绝对或相对位置嵌入被 RoPE 嵌入取代。

为了理解模型规模的影响，作者测试了三种不同规模的 PaLM，见下文。

![|500](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fd8e3b7c2-95cf-4396-8767-fa7fa9336e95_1656x414.png)

尽管幂律表明模型性能应在上述模型之间平稳提升，但分析发现，使用最大（5400 亿参数）模型时，我们常常看到不成比例的性能提升。结合更广泛的预训练过程，较大的 LLM 带来了出人意料的大幅收益。

> “对于某些任务，我们观察到不连续的改进，从 62B 扩展到 540B 时，准确性出现显著跃升，而从 8B 扩展到 62B 时没有。这表明当模型达到足够规模时，大型语言模型的新能力可能会出现，并且这些能力会在先前研究的规模之外继续涌现。” 

**数据集**    PaLM 的预训练语料库包含 7800 亿个标记。虽然比用于训练 Chinchilla 的数据集稍小，但仍大于大多数先前的 LLM；见下文。

![|500](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F416ae668-520c-4b4b-bc83-78b862295461_1686x662.png)

创建高性能的 LLM 不仅仅是将模型做大。最近关于 LLM 的扩展规律研究表明，性能会随着模型大小和预训练语料库大小的共同增长而提高。因此，PaLM 有机会通过使用更大的预训练语料库，显著超越像 MT-NLG 这样的模型（尽管它只稍大一些）。

PaLM 的预训练语料库来源于高质量的网页、书籍、维基百科、新闻、文章、代码和社交媒体对话。它包含 22% 的非英语数据（见下文），受训练 LaMDA 和 GLaM 的语料库启发。所有模型在这个数据集上都只训练一个 epoch。

![|500](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F903e263a-462e-4156-ba31-3071c0db025d_1658x572.png)

**使用大型词汇表**   由于预训练语料库中有相当一部分是非英语内容，作者采用了[SentencePiece 分词器](https://github.com/google/sentencepiece)，词汇表大小为 256K。分词器将原始文本输入并从中提取词元（即单词或子词）。这个分词过程基于一个底层词汇表（即已知词元集合），所有从文本中提取的词元必须是词汇表中的成员。如果某个词元不在底层词汇表中，它将被分解为更小的块（甚至可能是字符），直到分解为有效词元，或替换为通用的 “`[UNK]`” token。

使用小型词汇表意味着许多重要的词元无法被正确捕获，这会影响 LLM 的性能。对于多语言模型，我们通常会显著增加底层词汇表的大小以避免这种影响，因为来自多种语言的数据会使用更广泛的 token。PaLM 也不例外：作者采用了比通常更大的词汇表大小，以避免错误分词数据，并允许在多种语言中更有效地学习。

**训练系统**    在概述 PaLM 的训练框架之前，我们需要了解一些与分布式训练相关的概念。最重要的是，我们需要理解模型并行、数据并行和流水线并行之间的区别。
<img src="https://pbs.twimg.com/media/Fo7qa42aAAAXM30?format=jpg&name=4096x4096" width="500">
PaLM 使用分布在两个 [TPU pods](https://cloud.google.com/tpu/docs/training-on-tpu-pods) 上的 6144 个 TPU 芯片进行训练（即通过高速网络接口连接的 TPU 组）。在发表时，这是当时描述的最大配置；详情见下文。

![|600](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F031ea65b-b342-4139-a8ed-9b10fde34b68_2844x952.png)

在一个 pod 内，TPU 之间的通信速度非常快，但 pod 之间的通信速度要慢得多。通常，模型和数据并行对带宽的要求太高，难以在 TPU pods 间高效训练。大多数先前的工作通过以下方式处理这一问题：

1. 将训练限制在单个 TPU pod 内。
2. 在 pod 之间使用带宽要求较低的流水线并行。

然而，流水线并行有许多显著缺点，比如在清空或填充流水线时让加速器闲置，以及高内存需求。使用 Pathways 系统，PaLM 通过结合模型和数据并行（即不使用流水线并行）高效地在 TPU pods 间进行训练。这种新颖的训练模式显著提高了效率。

![|600](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F9d7f3d6f-ee0f-4ac5-b228-c815ab4100f7_1898x512.png)

例如，PaLM 的模型 FLOPs 利用率（即每秒处理的 tokens 数量与系统的理论最大吞吐量之比）达到了 46.2%，而之前的系统很难超过 30%；详情见上文。关于 Pathways 系统及其如何在大规模语言模型训练效率上实现如此大幅提升的信息，请查看下文的文章。

[Pathways Architecture](https://blog.google/technology/ai/introducing-pathways-next-generation-ai-architecture/)

#### PaLM 的表现如何？

PaLM 被证明能够有效处理多种语言，具有更好的推理能力，比小型模型表现更好，甚至在某些任务上超越了人类的语言理解水平。

**多语言大规模语言模型** 之前的大规模语言模型（如 GPT-3）在机器翻译方面表现出一定能力，尤其是在将其他语言翻译成英语时。在以英语为中心的数据对和设置中，我们发现 PaLM 相较于之前的大规模语言模型提升了翻译性能；详情见下文。

![|600](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F47bcaf3e-6adf-4099-a1b9-46076e2994b8_1370x1120.png)

在低资源和非英语为中心的数据上，PaLM 仍表现相对较好，但在现有的监督翻译方法面前表现不及；如上所述。然而，由于之前的研究很少考虑非英语环境，PaLM 在这种环境下的相对良好表现令人印象深刻。总体而言，这一分析表明，PaLM 的语言翻译能力有所提升，但仍不及监督技术。

除了语言翻译，我们还观察到 PaLM 在多语言生成任务中表现良好。正如预期的那样，PaLM 的语言生成能力在英语中表现最佳，但在非英语生成任务中，该模型仍优于以往的大型语言模型。总体来看，这些结果表明，通过进行小的修改（即增加非英语预训练数据和为分词器使用更大的词汇表），可以显著提高大型语言模型的多语言能力。

**超越人类表现。** [BIG-bench 数据集](https://github.com/google/BIG-bench) 包含 150 个任务，主题包括逻辑推理、翻译、问答、数学等。与之前的大型语言模型相比，我们看到 PaLM 在大多数这些任务上都取得了更好的表现；见下文。

![|500](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F1fd670a7-04b7-48de-ade8-cb7c0f204fcd_1432x1640.png)

比起仅仅超越以往的大型语言模型，PaLM 在 BIG-bench 的大多数任务上也超越了人类的平均表现；见下文。对于其中一些任务，超越人类可能仅仅表明 PaLM 能够记忆数据或在多种语言中进行推理。然而，这并不总是如此！在其他任务上（例如因果关系识别），我们看到 PaLM 似乎提升了语言理解能力。

![|600](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F7ee4b3d8-dee0-42b0-a4dc-97a6e02da6ef_1444x750.png)

**幂律法则总是适用吗？** 当我们将 PaLM 的表现分解为具体任务类别时，可以看到模型规模对某些任务特别有帮助。例如，在逻辑序列任务（即将一组词按逻辑顺序排列）中，最大的 PaLM 模型相较于较小的模型有显著的性能提升。而在其他任务（例如数学归纳法）中，模型规模的影响则很小。

![|600](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F6a25a6ec-320e-4c68-ac7b-3b73e844d08b_1554x1282.png)

总体而言，PaLM 的性能并不总是随着模型规模遵循幂律。在某些情况下，使用更大的模型会导致性能大幅提升，而在其他情况下，最大的模型仅比较小的版本略有改善；见上文。

**学习推理**   尽管语言模型在许多任务上表现良好，但在解决基本推理任务上却表现不佳。许多研究者认为这是大型语言模型“浅层”语言理解的证明。然而，最近的研究利用“思维链提示”（即在最终输出前在模型内生成多个推理“步骤”）来提升大型语言模型的推理能力；见下文。

![|600](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F76ddeb3c-90e0-44aa-8657-74c5ea6fcbea_1432x636.png)

在评估 PaLM 时，作者发现，将这种规模的模型与思维链提示结合，足以在算术和常识推理任务上达到最新的准确率。此前的方法依赖于领域特定的架构、微调，甚至是任务特定的验证模块来解决这些推理任务。相比之下，PaLM 仅通过少样本思维链提示（以及用于算术推理任务的外部计算器模块）来解决这些任务；见下文。

![|600](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F8316252d-fe5d-43f6-ae34-ec5a16a8ce93_1408x1376.png)

有趣的是，我们发现最大的 PaLM 模型在推理能力上明显优于较小的变体。鉴于之前的研究发现模型规模对推理性能的影响常常是混合的（有时是负面的），这一发现很有趣。PaLM 的结果表明，在正确的提示方法下，模型（和数据）规模似乎能提升推理性能。

![|600](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F0f388369-e932-4c10-8d04-4d2eeae0ebba_1432x536.png)

#### 关键要点

尽管最初尝试训练超过 GPT-3 规模的 LLMs 并不太成功，但在 PaLM 上我们看到，只需要一个高效的训练框架来进行更广泛的预训练。通过使用 Pathways 框架，PaLM 能够在比同规模的先前模型（如 MT-NLG）更大的数据集上进行训练。最终的 LLM 具备了令人印象深刻的多语言理解和推理能力，我们发现增加模型规模通常能带来显著的好处。以下是 PaLM 的一些重要收获。

**幂律总是成立吗？** 关于 LLMs 的众多出版物显示，LLM 性能与各种量（如非嵌入模型参数、数据集大小、训练计算量等）之间存在幂律关系。虽然这种趋势在整体性能上成立，但在分别考察每项任务的性能时，情况会更复杂。某些任务从规模中获得的好处不成比例，而其他任务则没有太大收益。因此，规模对 LLMs 通常有帮助，但结果因所解决的下游任务而异。

**我们应该避免流水线并行吗？** PaLM 的主要卖点之一是其使用的高效 Pathways 训练框架。通常，在多个 TPU pods 或计算节点上进行训练需要使用流水线并行，因为内存带宽有限。然而，通过去除流水线并行，仅使用数据和模型并行进行 TPU pods 间的训练，PaLM 实现了突破性的训练效率和吞吐量。这些训练框架的提升使 PaLM 能够在更多数据上进行训练，从而实现了模型的出色表现。

**LLM 规模与推理能力。** 之前关于 LLMs 的研究常常指出其推理能力较差。事实上，LLMs 执行推理任务的能力似乎随着规模的增加而下降。然而，在 PaLM 上我们看到情况并非总是如此。如果将更大的 LLMs 与更多的预训练数据和正确的提示方法（如链式思维提示）结合，我们会看到 LLM 推理能力的显著提升！
