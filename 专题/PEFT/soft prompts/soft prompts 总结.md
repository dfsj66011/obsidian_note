

## Soft prompts

训练大型预训练语言模型非常耗时且计算密集。随着它们的规模不断扩大，人们对更高效的训练方法（如 *prompting*）越来越感兴趣。提示学习通过在输入中加入描述任务甚至展示任务示例的文本提示，来激活冻结的预训练模型以完成特定的下游任务。借助提示学习，你可以避免为每个下游任务完全训练一个单独的模型，而是使用同一个冻结的预训练模型。这种方法简单得多，因为你可以用同一个模型完成多个不同的任务，而且训练和存储少量提示参数比训练模型的所有参数要高效得多。

提示方法分为两类：

- *硬提示* 是手动精心设计的文本提示，包含离散的输入标记；缺点是需要花费大量精力来创建一个好的提示。
- *软提示* 是与输入嵌入连接的、可学习的张量，可以针对数据集进行优化；缺点是它们不可被人读取，因为你无法将这些“虚拟标记”与真实单词的嵌入相匹配。

本概念指南简要介绍了 🤗 PEFT 中包含的软提示方法：prompt tuning, prefix tuning, P-tuning 和 multitask prompt tuning。


### 1、Prompt tuning

![|500](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/peft/prompt-tuning.png)
只需训练并存储一组显著更小的特定任务提示参数（[图像来源](https://huggingface.co/papers/2104.08691)）。

提示调优（Prompt tuning）是为 T5 模型上的文本分类任务开发的，所有下游任务都被转化为文本生成任务。例如，序列分类通常为一段文本分配一个单一的类别标签。通过将其转化为文本生成任务，构成类别标签的标记会被生成。提示以一系列 tokens 的形式添加到输入中。通常，模型参数是固定的，这意味着提示 tokens 也由模型参数固定。

提示调优背后的关键思想是，提示标记拥有独立更新的自身参数。这意味着你可以保持预训练模型的参数冻结，仅更新提示标记嵌入的梯度。其结果与训练整个模型的传统方法相当，并且提示调优的性能会随着模型规模的增大而提升。

### 2、Prefix tuning

![|500](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/peft/prefix-tuning.png)

优化每个任务的前缀参数（[图片来源](https://hf.co/papers/2101.00190)）

前缀调整（Prefix tuning）是为 GPT 模型上的自然语言生成（NLG）任务而设计的。它与提示调整（prompt tuning）非常相似；前缀调整同样是在输入前添加一系列特定于任务的向量，这些向量可以训练和更新，同时保持预训练模型其余参数冻结不变。

主要区别在于，前缀参数被插入到模型的所有层中，而提示调优仅将提示参数添加到模型输入嵌入中。前缀参数还通过单独的前馈网络（FFN）进行优化，而不是直接在软提示上进行训练，因为这会导致不稳定并影响性能。在更新软提示后，前馈网络会被丢弃。

因此，作者发现，尽管前缀调整的参数比完全微调模型少 1000 倍，但其表现与完全微调模型相当，并且在数据量较少的情况下表现更佳。

### 3、P-tuning

![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/peft/p-tuning.png)

提示词可以插入到输入序列的任意位置，并由提示编码器进行优化（[图片来源](https://hf.co/papers/2103.10385)）。

P-tuning 专为自然语言理解（NLU）任务和所有语言模型而设计。它是一种软提示方法的另一种变体；P-tuning 同样添加了一个可训练的嵌入张量，该张量可以被优化以找到更好的提示，并且它使用一个提示编码器（双向长短期记忆网络或 LSTM）来优化提示参数。不过，与 prefix tuning 不同的是：

- 提示词标记可以插入到输入序列的任何位置，并不限于仅在开头插入。
- 提示词标记仅添加到输入中，而不是添加到模型的每一层。
- 引入 *auchor* 标记可以提高性能，因为它们指示了输入序列中某个组件的特征。

研究结果表明，P-tuning 比手动设计提示语更高效，并且它使类似 GPT 的模型能够在自然语言理解（NLU）任务上与类似 BERT 的模型相竞争。


## [](https://huggingface.co/docs/peft/conceptual_guides/prompting#multitask-prompt-tuning)Multitask prompt tuning

![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/peft/mpt.png)

[Multitask prompt tuning enables parameter-efficient transfer learning](https://hf.co/papers/2303.02861).

[Multitask prompt tuning (MPT)](https://hf.co/papers/2303.02861) learns a single prompt from data for multiple task types that can be shared for different target tasks. Other existing approaches learn a separate soft prompt for each task that need to be retrieved or aggregated for adaptation to target tasks. MPT consists of two stages:

1. source training - for each task, its soft prompt is decomposed into task-specific vectors. The task-specific vectors are multiplied together to form another matrix W, and the Hadamard product is used between W and a shared prompt matrix P to generate a task-specific prompt matrix. The task-specific prompts are distilled into a single prompt matrix that is shared across all tasks. This prompt is trained with multitask training.
2. target adaptation - to adapt the single prompt for a target task, a target prompt is initialized and expressed as the Hadamard product of the shared prompt matrix and the task-specific low-rank prompt matrix.

![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/peft/mpt-decomposition.png)

[Prompt decomposition](https://hf.co/papers/2103.10385).

## [](https://huggingface.co/docs/peft/conceptual_guides/prompting#context-aware-prompt-tuning-cpt)Context-Aware Prompt Tuning (CPT)

![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/peft/cpt.png)

CPT optimizing only specific token embeddings while keeping the rest of the model frozen [(image source)](https://huggingface.co/papers/2410.17222).

[Context-Aware Prompt Tuning (CPT)](https://huggingface.co/papers/2410.17222) is designed to enhance few-shot classification by refining only context embeddings. This approach combines ideas from In-Context Learning (ICL), Prompt Tuning (PT), and adversarial optimization, focusing on making model adaptation both parameter-efficient and effective. In CPT, only specific context token embeddings are optimized, while the rest of the model remains frozen. To prevent overfitting and maintain stability, CPT uses controlled perturbations to limit the allowed changes to context embeddings within a defined range. Additionally, to address the phenomenon of recency bias—where examples near the end of the context tend to be prioritized over earlier ones—CPT applies a decay loss factor.

Take a look at [Example](https://github.com/huggingface/peft/blob/main/examples/cpt_finetuning/README.md) for a step-by-step guide on how to train a model with CPT.

[<>Update on GitHub](https://github.com/huggingface/peft/blob/main/docs/source/conceptual_guides/prompting.md)

Soft prompts

[←Adapters](https://huggingface.co/docs/peft/conceptual_guides/adapter)[IA3→](https://huggingface.co/docs/peft/conceptual_guides/ia3)

[Soft prompts](https://huggingface.co/docs/peft/conceptual_guides/prompting#soft-prompts)[Prompt tuning](https://huggingface.co/docs/peft/conceptual_guides/prompting#prompt-tuning)[Prefix tuning](https://huggingface.co/docs/peft/conceptual_guides/prompting#prefix-tuning)[P-tuning](https://huggingface.co/docs/peft/conceptual_guides/prompting#p-tuning)[Multitask prompt tuning](https://huggingface.co/docs/peft/conceptual_guides/prompting#multitask-prompt-tuning)[Context-Aware Prompt Tuning (CPT)](https://huggingface.co/docs/peft/conceptual_guides/prompting#context-aware-prompt-tuning-cpt)