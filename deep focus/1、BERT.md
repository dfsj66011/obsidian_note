
### 1、什么是 BERT？

实体识别等）。在提出时，BERT 在 11 种不同的语言理解任务中取得了新的最先进成果，迅速成名并持续至今。

BERT 的卓越效果源于：

1. 通过自监督学习在大量原始文本数据上进行预训练
2. 为序列中的每个标记构建丰富的双向特征表示

尽管先前的研究表明，语言建模任务从大规模文本语料库的预训练中受益，BERT 通过设计一套简单而有效的自监督预训练任务扩展了这一理念，使相关特征得以学习。此外，BERT 摒弃了常用的单向自注意力机制，这种机制通常用于在语言理解任务中实现语言建模风格的预训练。相反，BERT 在其每一层中利用双向自注意力，揭示了双向预训练对于实现强大的语言表示至关重要。

**BERT 非常有用。** 你可能会想：_为什么要专门为这个模型写一整篇文章？_ 简单的答案是 BERT 非常通用——这种单一的模型架构可以用于解决许多不同的任务，并达到最先进的准确性，包括 token 级别（如命名实体识别）和句子级别（如情感分类）的语言理解任务。此外，其应用已扩展到 NLP 领域之外，用于解决多模态分类、语义分割等问题。

声称某个深度学习模型是“最有用的”有点夸张（尽管这很吸引眼球！）。然而，BERT 无疑是任何深度学习从业者的重要工具之一。简单来说，这种架构只需进行最少的任务特定修改，就可以下载并以低计算成本微调，以解决 NLP 及其他领域的众多潜在问题—— _它是深度学习的 “瑞士军刀 ”_ ！

### 2、BERT 的构建模块

在概述 BERT 架构的具体细节之前，了解 BERT 所基于的核心组件和理念是很重要的。这些主要概念可以归纳为以下几点：

- （双向）自注意力机制
- Transformer 编码器
- 自监督学习

#### 2.1、自注意力

![|500](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F7f42e9e5-422b-486b-b31d-0f40323ccd74_1031x369.png)

从高层次来看，自注意力是一种非线性变换，它将 “token” 序列（即序列中的单个元素）作为输入，每个标记用向量表示。与此输入序列相关的矩阵如上所示。然后，这些标记表示被转换，返回一个新的标记表示矩阵。

**在这个转换中发生了什么？** 对于每个单独的标记向量，自注意力执行以下操作：

- 将该标记与序列中的每个其他标记进行比较
- 计算每对的注意力得分
- 根据序列中其他标记的注意力得分，调整当前标记的表示

直观地说，自注意力只是根据序列中的其他标记调整每个标记的向量表示，形成一个更具上下文意识的表示。

![|500](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2Fdb41d46e-ebea-46c8-9005-472d2983a35b_1164x848.png)

**多头注意力**    自注意力通常以多头的方式实现，即在并行应用多个自注意力模块后，将其输出连接在一起。每个注意力头内部的自注意力机制仍然相同，不过在应用自注意力之前，标记向量会被线性投影到较低维度（以避免计算成本过高）。

这种多头方法的好处在于，多头注意力层中的每个注意力头可以学习序列中不同的注意力模式。因此，模型不会因为单个自注意力层中可以“关注”的其他标记数量而受到限制。相反，不同的头可以捕捉到标记关系的不同模式。

**单向与双向**    在为序列中的每个标记创建上下文感知的表示时，有两种基本选项来定义这个上下文：

- 考虑所有标记
- 考虑当前标记左侧的所有标记

这两种选项，如下图所示，产生了自注意力的两种不同变体：双向和单向。

![|500](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2Fd8c58829-dd45-4b9e-87c3-4cb541e9d5fc_1444x604.png)

单向自注意力确保每个标记的表示仅依赖于序列中它之前的标记（即，通过在自注意力中“屏蔽”序列中后来的标记）。这种修改对于语言建模等应用是必要的，因为不应允许模型“向前看”以预测下一个词。

相反，双向自注意力基于序列中的所有其他标记来构建每个标记的表示。双向自注意力是 BERT 成功的关键，因为许多之前的建模方法：

1. 使用了单向自注意力
2. 通过连接句子在前向和后向的单向表示来构建浅层的双向特征

这些方法远不如 BERT 使用双向自注意力有效，这强调了双向特征表示在超越语言建模的任务中的优势。

#### 2.2 Transformer 编码器

Transformer 架构通常有两个组件——编码器和解码器。然而，BERT 仅使用了 Transformer 的编码器组件。

![|150](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F9ec3e8bd-88f4-43fd-a5c4-14c120dd31ee_330x384.png)

可以看到，Transformer 编码器只是几个重复的层，每层包含（双向、多头）自注意力和前馈变换，每个操作后跟随有层归一化和残差连接。非常简单！

**为什么只用编码器？** Transformer 架构的两个组件往往有不同的用途。

- 编码器： 利用双向自注意力将原始输入序列编码为一系列可区分的标记特征。
- 解码器： 接收丰富的编码表示，并将其解码为新的、期望的序列（例如，将原始序列翻译成另一种语言）。

在 BERT 的情况下，我们将在本文的其余部分看到，这种解码过程并不是必需的。BERT 的目的是简单地构建初始的编码表示，然后可以用于解决不同的下游任务。

#### 2.3 自监督学习

![|380](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2Fdfe41ce8-3962-4b32-b119-b8302101337d_878x860.png)

BERT 卓越性能的关键之一在于其能够以自监督方式进行预训练。总体而言，这种训练非常有价值，因为可以在原始、未标注的文本上进行。由于此类数据在网上广泛可得（例如，通过在线书籍库或像维基百科这样的网站），可以收集大量的文本数据集进行预训练，使 BERT 能够从比大多数监督/标注数据集大得多的多样化数据集中学习。

虽然存在许多自监督训练目标的例子，但一些例子（我们将在本文中进一步概述）包括：

- 掩码语言模型 (MLM)：在句子中掩盖/移除某些词并尝试预测它们。
- 下一句预测 (NSP)：给定一对句子，预测这些句子在文本语料库中是否相互衔接。

这些任务都不需要人工标注，而是可以用未标注的文本数据来完成。

**这是无监督学习吗？** 这里值得区分的一点是自监督学习和无监督学习的区别。无监督和自监督学习都不利用标注数据。然而，无监督学习专注于发现和利用数据本身的潜在模式，而自监督学习则在数据中找到某种已存在的监督训练信号并利用其进行训练，因此不需要人工干预。

### 3、BERT 的实际工作原理…

虽然我们已经概述了一些 BERT 背后的基本思想，但在本节中，我将更详细地描述 BERT，重点介绍其架构和训练方案。

#### 3.1 BERT 的架构

如前所述，BERT 的架构只是 Transformer 模型的编码器部分（即，仅编码器的 Transformer 架构）。在最初的发布中，提出了两种不同大小的 BERT：

- BERT Base：12 层，768 维隐藏表示，每个自注意力模块中有 12 个注意力头，共 110M 参数。
- BERT Large：24 层，1024 维隐藏表示，每个自注意力模块中有 16 个注意力头，共 340M 参数。

值得注意的是，BERT Base 的大小与 OpenAI 的 GPT 相同，这使得模型之间的比较更加公平。

**这有什么不同？** BERT 与先前提出的语言理解模型（例如 OpenAI GPT）的主要区别在于使用了双向自注意力。尽管先前的工作利用了自监督预训练，但仅使用了单向自注意力，这极大地限制了模型能够学习的标记表示的质量。

**构建输入序列**    直观地说，BERT 将一些文本序列作为输入。特别是，这个序列通常是单个句子或一对连续的句子。

![|500](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F3eaa245c-7a13-4872-90f4-6c748b682f32_1549x792.png)

这个高层次的概念很简单，但你可能会想：如何从原始文本生成 BERT 兼容的输入序列？这个过程可以分为几个步骤：

- 分词：将原始文本数据分解为表示单词或词的一部分的单个标记或元素。
- 插入“特殊”标记：BERT 的输入序列以 `[CLS]` 标记开始，以 `[SEP]` 标记结束，表示句子的开始/结束。如果使用两个连续的句子，则在它们之间放置另一个 `[SEP]` 标记。
- 嵌入：将每个标记转换为其对应的 WordPiece 嵌入向量。
- 加性嵌入：输入数据现在是一个向量序列。可学习的嵌入被添加到序列中的每个元素，表示元素在序列中的位置以及它是第一句还是第二句的一部分。因为自注意力机制无法区分元素在序列中的位置，所以需要这些信息。

通过这些步骤，原始文本数据被转换为 BERT 可以处理的向量序列。分词、插入特殊标记和嵌入过程在上图中展示，而加性嵌入过程在下图中展示。

![|500](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F30b03109-5bcd-4624-8c4e-55806dfd00d9_2138x688.png)


#### 3.2 训练 BERT

BERT 的训练过程分为两个步骤：

1. 预训练
2. 微调

这两个步骤的架构几乎相同，尽管可能会使用一些小的、特定任务的模块（例如，MLM 和 NSP 都使用一个额外的分类层）。

**预训练**    在预训练期间，BERT 模型通过两种不同的任务在未标注数据上进行训练：MLM（也称为 Cloze 任务）和 NSP。值得注意的是，BERT 不能使用以往工作中常用的语言建模目标进行训练，其中模型迭代地尝试预测序列中的下一个词。使用双向自注意力会使 BERT 通过简单地观察和复制下一个标记来作弊。

![|500](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F984d381d-383b-491b-9422-adcc8d45b7d0_2050x1016.png)

如上图所示，NSP 任务相当简单。来自预训练语料库的连续句子（即句子 A 和 B）被输入到 BERT 中，其中 50% 的情况下，第二个句子会被替换为另一个随机句子。然后，经过 BERT 处理后的 `[CLS]` 标记的最终表示被传递到一个分类模块中，该模块预测输入的句子是否实际匹配。

![|500](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2Ffe697a9b-6a11-489b-9701-15d42d3feb15_2006x1206.png)

如上所示，MLM 不是像 NSP 那样的序列级任务。它通过将输入序列中 15% 的标记随机替换为特殊的 `[MASK]` 标记来进行掩码。然后，这些 `[MASK]` 标记的最终表示被传递到一个分类层，以预测被掩盖的词。然而，作者并不是总是以这种方式掩盖标记，而是 80% 的情况下用 `[MASK]` 替换，10% 的情况下用随机标记替换，10% 的情况下保持原始标记不变。这样的修改是为了避免 `[MASK]` 标记在预训练中存在但在微调中不存在的问题。

通过这些任务，BERT 在由 BooksCorpus 和英文维基百科组成的语料库上进行预训练。有趣的是，使用文档级语料库（而不是打乱的句子语料库）对预训练质量至关重要。你的语料库需要在句子之间具有长距离依赖，以便 BERT 学习最佳特征。后来的研究也证实了这一有趣的发现。事实上，即使是基于 TF-IDF 分数对随机打乱的句子重新排序以形成合成的长期依赖，也能提高预训练质量。

![|500](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2Fa3c76554-5ade-4210-b612-98f2f5062abf_1224x690.png)

**微调**    BERT 的自注意力机制设计得非常简单，以便能够轻松地建模不同类型的下游任务。在大多数情况下，只需将任务的输入输出结构与 BERT 的输入输出结构匹配，然后对所有模型参数进行微调。以下是一些示例：

- 标记级任务：正常处理序列，然后将每个标记的输出表示通过一个单独的模块来预测给定标记。
- 句子/文档级任务：正常处理序列，然后将 `[CLS]` 标记的输出表示（输入序列的聚合嵌入）通过一个单独的模块来进行序列级预测。
- 文本对任务：在 BERT 的输入结构中将文本对的每一部分编码为“句子 A”和“句子 B”，然后将 `[CLS]` 标记的输出表示通过一个单独的模块来基于文本对进行预测。

上述一般任务结构表明 BERT 是一个多功能的模型。许多不同的任务可以通过简单地将它们映射到 BERT 的输入输出结构来解决，相对于预训练所需的架构修改最小。请参见下方 BERT 可解决的不同语言理解任务的示例。

![|400](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F2ed88333-36d9-4790-8fa4-62d1cc5d2ae8_500x533.png)

在微调过程中，所有 BERT 参数都是端到端训练的。与预训练相比，BERT 的微调过程成本较低。事实上，论文中的所有结果在单个 GPU 上复现都不到 2 小时。如果不相信，可以自己试试！

[在 GLUE 上微调 BERT](https://github.com/huggingface/transformers/tree/main/examples/pytorch/text-classification)

### 4、BERT 可能是自切片面包以来最好的东西！

在结束这个概述之前，我想概述一下 BERT 所取得的一些实证结果。虽然可以轻松阅读论文来查看结果，但我认为值得简要介绍一下，原因是——_强调 BERT 在 NLP 任务中的出色表现_。BERT 在各种不同任务上取得的结果如下所示。

![|500](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F58bcb60d-4e58-4350-a4df-62b319a230a3_1607x1088.png)

你可能会注意到 BERT 在这些实验中的表现很有趣——它从未被超越（除非是人类，但也仅在某些情况下）。在发表时，BERT 在**十一项不同的 NLP 基准测试**中设立了新的记录。此外，这些任务大多以前由特定任务的专用模型解决，而 BERT（如你在概述中所见）是一个通用的语言理解模型，可以应用于许多不同的任务。

BERT 实证评估中的其他一些有趣发现如下：

- BERT Large 和 BERT Base 在所有考虑的任务中都显著优于之前的所有方法。
- BERT Large 在所有任务中显著优于 BERT Base，尤其是在训练数据较少的任务中表现出色。
- 移除 NSP 或 MLM（即使用单向语言建模目标）会显著降低 BERT 的性能。

尽管较大的模型在小数据集上表现更好似乎违反直觉（这似乎是过拟合的配方），但 BERT 的结果表明，使用较大的模型对低资源任务（即训练数据较少的任务）是有益的，只要有足够的预训练。

### 5、要点

尽管 BERT 在当前深度学习研究的快速发展中相对较旧，但我希望这个概述能恰当地强调模型的简单性和深刻性。BERT 是一个非常强大的工具，使用简单且成本低廉。

**是什么让 BERT 如此强大？** BERT 的关键在于两个核心概念：双向自注意力和自监督学习。BERT 改进了先前的方法，部分原因是它摒弃了使用单向自注意力进行语言建模式预训练的常见方法。相反，BERT 利用双向自注意力来制定一组自监督的预训练任务，从而产生更强大的特征表示。最近，研究人员表明，这些自监督任务的制定本身（而不仅仅是用于预训练的大量数据）是 BERT 成功的关键。

**普通从业者可以使用吗？** 使用 BERT，你可以简单地：

1. [下载](https://huggingface.co/transformers/v3.3.1/pretrained_models.html)一个预训练模型
2. [微调](https://huggingface.co/docs/transformers/training)这个模型，以在大量 NLP 任务中实现最先进的性能

微调 BERT 的计算成本低，可以在相对简单的硬件配置（例如单个 GPU）上进行。因此，BERT 是任何深度学习从业者工具库中的一个非常好的工具——你会惊讶于 BERT 是你在许多不同任务中的最佳选择。

**进一步阅读**    我在这个概述中只涵盖了一篇论文，但 BERT 已被无数后续出版物扩展。以下是我最喜欢的一些：

1. ELECTRA 提出了一种新的预训练任务，使得高性能的小型 BERT 模型的训练成为可能
2. ALBERT 提出参数减少技术，使 BERT 预训练更快且内存占用更少
3. Vilbert 是 BERT 的一个推广，用于联合视觉和语言任务）

**个人笔记**    BERT 是第一个让我对深度学习产生兴趣的模型。尽管我目前的研究更专注于计算机视觉（或多模态学习，BERT 在这方面仍然表现很好！），但 BERT 的多功能性至今仍让我印象深刻。_简单而有效的想法是稀有而美丽的_。


GPT-3 等语言模型彻底改变了 NLP 的现代深度学习应用，获得了广泛的宣传和认可。然而有趣的是，GPT-3 的大部分技术创新都继承自其前身 GPT 和 GPT-2。因此，对 GPT 和 GPT-2 的实用理解有助于更好地掌握当前的 NLP 方法。

GPT 和 GPT-2 模型探索的基本方法很简单。事实上，它可以归结为几个步骤：

1. 使用大量原始文本数据预训练语言模型
2. 调整此预训练模型来解决下游任务

但是描述有点模糊。_语言模型的预训练是如何进行的？我们如何“调整”语言模型来解决不同的任务？_

在本概述中，我们将对语言建模、其在 GPT 和 GPT-2 中的使用以及它如何用于解决不仅仅是生成连贯文本的问题建立基本的了解。尽管由于最近提出了更大、更强大的模型，GPT 和 GPT-2 有些过时，但它们所基于的基本概念仍然与现代深度学习应用高度相关。


### 1、先决条件

GPT 和 GPT-2 背后的基本原理是使用通用的、预训练的语言模型以高精度解决各种语言建模任务。为了充分理解这种方法，我们必须首先介绍一些关于语言模型如何工作以及如何在 GPT 和 GPT-2 中利用它们的基本概念。

#### 1.1 LM

![|600](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F89687ad1-ab5d-4c72-840c-343d7fa26ab2_1854x1030.png)


GPT 模型是使用语言建模目标在未标记文本数据的语料库/数据集上进行预训练的。简而言之，这意味着我们通过 _(i)_ 从数据集中抽取一些文本和 _(ii)_ 训练模型来预测下一个单词来训练模型。这种预训练过程是一种自监督学习，因为只需查看数据集中的下一个单词就可以确定正确的“下一个”单词。

**数学中的语言建模**   要理解语言建模，我们只需要掌握上面概述的基本思想。然而，为了使这一点更加严格，我们可以注意到我们的语料库只是一组标记。我们可以将标记视为数据集中的单个单词，但这并不完全正确。实际上，标记可能是子词甚至字符。

我们将组成我们预训练数据集的这组标记（大小为 $N$）表示如下：$$u=\{u_1, u_2, \cdots,u_N\}$$
给定一个具有参数 $\theta$ 的深度学习模型，语言建模目标试图最大化如下所示的可能性。

![|500](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F3430b67c-2d19-4840-9207-09e68a25d03a_1318x444.png)

简而言之，这个表达式描述了模型在给定前 $k$ 个标记作为上下文的情况下预测正确的下一个标记的概率。

使用语言建模损失（它仅表示我们的模型准确预测序列中的下一个标记的能力！），我们可以按照以下步骤预训练我们的模型的参数 $θ$ 以最小化损失：

1. 来自预训练语料库的示例文本
2. 使用我们的模型预测下一个标记
3. 使用随机梯度下降 (SGD) 或任何其他优化器来增加下一个正确的标记的概率

通过多次重复这种（自我监督）训练过程，我们的模型最终将变得非常擅长语言建模（即预测序列中的下一个标记）。

**什么是语言模型？**  使用这种自监督语言建模目标进行预训练的模型通常称为语言模型 (LM)。LM 的规模越大（即层数、参数等越多），其效率就越高。因此，我们经常会看到这些模型的更大版本（例如 GPT-3），它们被称为大型语言模型 (LLM)。 

**为什么 LM 有用？** LM 可以通过迭代预测最有可能的下一个标记来生成连贯的文本，从而实现从文本自动完成到聊天机器人等一系列应用。然而，除了生成能力之外，NLP 领域的先前研究表明，LM 预训练对各种任务都非常有益；例如，预训练的词嵌入在下游任务中很有用，LM 预训练可以提高 LSTM的性能。

除了这些方法之外，GPT 模型还探索了使用 Transformer 进行语言模型预训练。与顺序模型（例如 LSTM）相比，Transformer _(i)_ 具有极强的表达能力（即高表示容量、许多参数等）；_(ii)_ 更适合现代 GPU 的并行计算能力，允许使用更大的模型和更多的数据进行 LM 预训练。这种可扩展性使探索 LLM 成为可能，而 LLM 已经彻底改变了 NLP 应用。

#### 1.2 仅解码器的 transformers

GPT 和 GPT-2 都使用仅解码器的 Transformer 架构。Transformer 架构有两个主要组件：编码器和解码器。

![|300](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F0235fd2f-26f4-47ff-b95e-eddf6a4593b0_782x1152.png)

仅解码器架构从变压器中移除了以下组件：

- 整个编码器模块
- 解码器中的所有编码器-解码器自注意模块

移除这些组件后，解码器的每一层都只包含一个掩蔽的自注意力层和一个前馈神经网络。将多个这样的层堆叠在一起，形成了一个深度的、仅用于解码器的 Transformer 架构，例如用于 GPT 或 GPT-2 的架构。

![|400](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F91a045da-57be-437d-962c-529ee5bc93fb_1234x828.png)

**为什么是解码器？** 选择使用解码器架构（而不是编码器）作为 LM 并不是任意的。解码器中的掩蔽自注意力层确保模型在制作 token 的表示时不能向前看序列。相反，双向自注意力（如编码器中使用的）允许根据序列中的所有其他 token 调整每个 token 的表示。

语言建模需要使用掩蔽自注意力，因为我们在预测下一个标记时不应该向前看句子。使用掩蔽自注意力会产生一个自回归架构（即，模型在时间的输出 $t$ 被用作时间的输入$t+1$），它可以连续预测序列中的下一个标记。

![|500](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F83ac8a81-a3b8-42e8-bc53-6ab3a505effc_1880x1010.png)

然而，对于不需要掩蔽自注意力的任务（例如句子分类、标记等），我们应该记住使用双向自注意力确实是有益的。

#### 1.3 创建基础模型

现在我们对语言建模和相关架构有了基本的了解，我们可以理解 GPT LM 背后的灵感，它始于以下观察： 

- 未标记的文本语料库非常丰富
- 标记数据稀缺 

对于大多数深度学习系统来说，需要大量标记数据才能执行判别性语言理解任务。_当前的深度学习系统是狭隘的专家_。该模型只是在大型监督数据集上进行训练，以便它学会准确地执行特定任务。

![|500](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F51e535d4-8edb-4218-9c41-3298fca62643_1624x1112.png)

尽管这种方法很常用，但它也存在一些主要限制：

1. 有些领域没有太多标记数据
2. 我们必须为每个想要解决的任务训练一个新模型（并且训练深度学习模型的成本很高！）

**基础模型**   GPT和 GPT-2 摆脱了深度学习中狭隘专家的范式。我们不必为每个应用程序训练一个新模型，而是可以预先训练一个 LM，然后以某种方式调整该模型来解决许多任务。用于解决许多任务的通用模型称为基础模型。

这种方法通过在大型、多样化的数据集上进行预训练来缓解数据稀缺问题。此外，这些模型可以重复使用或调整以解决其他任务，从而让我们避免不断训练新模型。将基础模型调整到下游任务的一种方法是在监督数据集上进行微调（即更多训练）。然而，最近，首选方法是通过零次或少量推理。

**通过提示进行零次/少次推理**。GPT 模型接收文本作为输入并产生文本作为输出。我们可以通过提供如下输入来利用这种通用的输入输出结构：

- “将这句话翻译成英语：`<sentence> =>`”
- “总结以下文件：`<document> =>`”。

这些解决任务的“提示”支持使用 LM 进行零样本（即，无需查看正确输出的示例）推理。根据这些提示，LM 的最合适输出应该可以解决该任务（例如，翻译成英文或总结文档）！要执行少样本推理，我们可以构建一个类似的提示，并在开始时提供正确输出的示例。

![|300](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F20c74320-2996-47ed-9507-08e1967a36d9_736x1262.png)

### 2、论文

现在我们将概述 GPT 和 GPT-2 的细节。这些模型由 OpenAI 的研究人员发布，率先使用通用 LM 来解决下游任务。它们为 GPT-3 等突破性进展奠定了基础。这些模型之间的主要区别仅在于底层 LM 的大小。

#### 2.1 GPT 

GPT 是一种通用语言理解模型，训练分为两个阶段：预训练和微调。

![|200](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F47d302b2-a7e6-4ee9-bf10-7c161a9e4057_342x648.png)

GPT 使用 12 层、仅解码器的 Transformer 架构，与原始 Transformer 解码器 相匹配（除了使用可学习的位置嵌入之外）。GPT 首先在 BooksCorpus 数据集上执行语言模型预训练，然后在各种判别性语言理解任务上分别进行微调（以监督方式）。

![|500](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F60d46502-4340-48d7-8db6-057993f82060_1622x816.png)

我们不会修改 GPT 的架构来解决不同的任务，而是以特定于任务的结构提供输入，然后将模型的输出传递到单独的分类层。例如，在蕴涵任务中，我们将输入的句子连接起来，用特殊的分隔符将它们分开，将此输入提供给 GPT，然后将 GPT 的输出传递到单独的分类层。gpt 的第 3.3 节进一步解释了使用不同监督任务对 GPT 进行微调，并在上文进行了说明。

![|500](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F636f5316-9d99-4b79-8c91-e4b3c76da2ef_1600x332.png)

GPT 已在上述各种任务上进行了评估。作者发现，在包含长篇连续文本（而不是单个、打乱顺序的句子）的语料库上对 GPT 进行预训练至关重要。在实验设置中，我们看到 GPT 在 12 项任务中的 9 项上实现了最佳性能，甚至始终优于模型集成。

![|600](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2Fb925c519-b72b-40a7-8d32-2f5c8b3804dc_1968x1078.png)

从这些实验中，我们了解到通用 LM 能够相对较好地理解语言概念，并且能够学习文本数据中的复杂模式（例如长期依赖关系、语言歧义等）。在不使用任何特定于任务的架构或修改的情况下，GPT 的表现远远优于许多基准，包括许多用于解决单个任务的专门解决方案。

#### 2.2 GPT-2

GPT-2 的提案遵循了与其前身类似的模式。该模型使用语言建模目标进行预训练，但不执行微调，而是选择以零样本方式解决下游任务。简而言之，GPT-2 通过以下方式执行多任务学习：

1. 对原始文本数据进行通用 LM 预训练
2. 使用文本“提示”对各种任务执行零样本推理

预训练是在自定义 WebText 数据集上进行的，该数据集是通过从 Reddit 中抓取热门链接构建的，并测试了四种不同大小的 LM。最小的模型与 GPT 的大小相匹配，最大的模型是 GPT-2。

![|300](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F80a786e0-35eb-4216-be14-7e32b88f5ff8_1186x460.png)

该模型架构与 GPT 相同，除了一些细微的差异（例如不同的权重初始化、更大的词汇量、更长的输入序列等）。尽管这些 LM 规模很大，但在预训练期间发现它们与 WebText 数据集的拟合度不足，这表明更大的 LM 表现会更好。

GPT-2 在多个任务（即语言建模、问答、翻译等）上进行了评估，取得了令人鼓舞的结果（但并不总是最先进的）。例如，在下表中，我们可以看到 GPT-2 在语言建模和阅读理解任务上表现良好，但在总结和问答方面远远落后于基准。

![|500](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F206e161b-36a7-40a1-8dd7-06d73725deb9_1982x586.png)

虽然性能并不出色，但我们需要记住，_GPT-2 无需进行微调即可解决任何这些任务_。所有这些结果都是通过零样本推理实现的，这使得 GPT 在某些任务上的竞争性能相当令人印象深刻。

有趣的是，零样本性能随着底层 LM 的大小而不断提高，这表明增加 LM 的大小/容量可以提高其在预训练期间学习相关特征的能力。

![|500](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F2a242e94-54ba-44e1-9f99-663a8330a67d_1966x800.png)

预训练和微调是一种有效的迁移学习范式，但 GPT-2 向我们展示了更简单、更通用的迁移方法。鉴于它们是在足够大的语料库上进行预训练的，LM 似乎能够学习下游任务，即使没有任何架构或参数修改。虽然 GPT-2 的表现并不令人印象深刻，但作者指出，更大的 LM 会好得多。

> “... 具有足够容量的语言模型将开始学习推断和执行自然​​语言序列中所展示的任务，以便更好地预测它们，无论它们的采购方法如何。” -摘自[2]

### 3、总结

GPT 和 GPT-2 教会了我们很多关于深度学习的知识。虽然从准确性的角度来看，它们在下游任务上的有效性并不令人印象深刻，但它们让我们看到了 LM 作为基础模型的巨大潜力，并为 GPT-3 等 LLM 的出现奠定了方法论基础。这些模型的影响是深远的，但我试图在下面总结一些从 GPT 和 GPT-2 研究中得出的最有用的结论和想法。

**语言模型预训练非常棒**  Transformers 能够高效利用计算资源，因此能够大规模执行语言模型预训练。在此预训练过程中学习到的表示法使预训练的 LM 能够很好地推广到解决其他任务。简而言之，_LM 不仅擅长语言建模_，它们还可以解决其他任务！

**尺寸很重要。** 正如我们在从 GPT 到 GPT-2 的过渡中所看到的，增加预训练 LM 的尺寸可以提高学习表征的质量；例如，GPT-2 在零样本/少样本推理方面远远优于 GPT。在（更大的）GPT-3 模型发布后，这一趋势变得更加明显。

**我们应该利用基础模型。** 大多数深度学习模型都是为完成单一、狭窄的任务而训练的。然而，在许多情况下，我们可以从以下方面受益：_(i)_ 通过对未标记数据的自监督学习对更大的模型进行预训练；_(ii)_ 调整此模型来解决许多任务。这种对大型基础模型的重新利用在计算上是高效的（即计算在许多任务之间共享），并且不特定于 LM。我们也可以为计算机视觉等领域训练基础模型！

#### 代码和资源

对于那些有兴趣尝试使用 GPT-2 的应用程序的人来说，代码是[公开的](https://github.com/openai/gpt-2)！但是，预训练这样的模型在计算上非常昂贵。更好的方法是[下载预先训练的语言模型](https://huggingface.co/models?sort=downloads)并对其[进行微调](https://huggingface.co/docs/transformers/v4.14.1/en/training)或执行零次/少次推理（例如，通过使用下面的演示）。

## 3、扩展定律和 GPT-3

![Deep (Learning) Focus](https://substackcdn.com/image/fetch/w_80,h_80,c_fill,f_auto,q_auto:good,fl_progressive:steep,g_auto/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2Fab9b43fb-52d5-40da-995d-5b7cd3f91064_896x896.png)

# [Deep (Learning) Focus](https://cameronrwolfe.substack.com/)

SubscribeSign in

# Language Model Scaling Laws and GPT-3

### Understanding why LLMs like GPT-3 work so well...

[

![](https://substackcdn.com/image/fetch/w_36,h_36,c_fill,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F69aba7df-b571-4609-aa47-fc2d031c11b8_1242x1595.jpeg)



](https://substack.com/@cwolferesearch)

[Cameron R. Wolfe, Ph.D.](https://substack.com/@cwolferesearch)

Dec 05, 2022

11

[](https://cameronrwolfe.substack.com/p/language-model-scaling-laws-and-gpt/comments)

Share

[

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F9ac27842-5db9-4cef-be53-85abbd7f37ac_1231x638.png)



](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F9ac27842-5db9-4cef-be53-85abbd7f37ac_1231x638.png)

This newsletter is supported by [Alegion](https://www.alegion.com/). At Alegion, I work on a range of problems from online learning to diffusion models. Feel free to check out our [data annotation platform](https://www.alegion.com/products) or [contact me](https://cameronrwolfe.me/) about potential collaboration/opportunities!

If you like this newsletter, please subscribe, share it, or follow me on [twitter](https://twitter.com/cwolferesearch). Thank you for your support!

Subscribe

---

[

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F2d3563e0-505c-4230-ab15-bf09355eb586_1818x1170.png)



](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F2d3563e0-505c-4230-ab15-bf09355eb586_1818x1170.png)

(from [1] and [2])

Language models (LMs) are incredibly generic–they take text as input and produce text as output. Recent research has revealed that this generic text-to-text structure can be exploited to solve a variety of tasks without task-specific adaptation (i.e., no fine-tuning or architectural modifications) by using prompting techniques to perform accurate zero and few-shot inference. Put simply, we can pre-train the LM over a large, unlabeled text corpus (using a language modeling objective), then ask the LM via textual prompts to solve a problems. In this way, the pre-trained model can easily be repurposed for solving different problems.

Although LMs hold incredible potential as task-agnostic foundation models, initial attempts at transferring pre-trained LMs to solving downstream tasks (e.g., GPT and GPT-2 [4, 5]) did not work well. Within this overview, we will learn how recent research has built upon these initial attempts and created LMs that achieve much better task-agnostic performance. The key finding within this line of work is that _LMs become much more powerful as you scale them up._

More specifically, we will learn that large LMs (LLMs) are _(i)_ more sample efficient than their smaller counterparts and _(ii)_ more capable of task-agnostic transfer to downstream tasks. Interestingly, the performance of these LLMs follows predictable trends with respect to various factors (e.g., model size and the amount of training data). The empirical observation of these trends eventually led to the creation of GPT-3, a 175 billion parameter LLM that far surpasses the task-agnostic performance of its predecessors and even outperforms state-of-the-art, supervised deep learning techniques on certain tasks.

# Background

Most prerequisite information needed to understand LMs has already been covered in one of my prior posts. These prerequisites include the language modeling objective, decoder-only transformer models, and how these ideas can be combined to generate powerful foundation models. Check out the link below to learn more.

[LM Prerequisites](https://cameronrwolfe.substack.com/i/85568430/prerequisites-for-gpt)

I will give a quick overview of these ideas here, as well as explain a few additional concepts that are useful for understanding LLMs like GPT-3.

### language modeling at a glance

Modern LMs use generic pre-training procedures to solve a wide variety of tasks without the need for downstream adaptation (i.e., no architectural modifications, fine-tuning etc.). Using a large corpus of unlabeled text, we pre-train our LM using a language modeling objective that _(i)_ samples some text from our corpus and _(ii)_ tries to predict the next word that occurs. This is a form of self-supervised learning, as we can always find the ground truth next word by simply looking at the data in our corpus; see below.

[

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F26fd2d00-2b35-460f-83c1-4699cc5f23f6_1854x1030.png)



](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F26fd2d00-2b35-460f-83c1-4699cc5f23f6_1854x1030.png)

The language model pre-training process

**architecture.** Modern LMs use decoder-only transformer architectures, which apply a sequence of layers consisting of masked self-attention and feed forward transformations to the model’s input. Masked self-attention is used instead of [bidirectional self-attention](https://cameronrwolfe.substack.com/i/76273144/self-attention), as it prevents the model from “looking forward” in a sequence to discover the next word. 

Beyond these decoder-only layers, the LM architecture contains embedding layers that store vectors corresponding to all possible tokens within a fixed-size vocabulary. Using these embedding layers, raw text can be converted into a model-ingestible input matrix as follows:

1. [Tokenize](https://towardsdatascience.com/how-to-build-a-wordpiece-tokenizer-for-bert-f505d97dddbb) raw text into individual tokens (i.e., words or sub-words)
    
2. Lookup the corresponding embedding vector for each input token
    
3. Concatenate token embeddings, forming a matrix/sequence of token vectors
    
4. Add position (and other) embeddings to each token
    

See the figure below for an illustration of this process.

[

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2Fc2eff169-d176-4ce5-9993-25df080dd039_1504x742.png)



](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2Fc2eff169-d176-4ce5-9993-25df080dd039_1504x742.png)

Converting raw text into model-compatible matrices of token embeddings

The differentiation between embedding and decoder-only layers within the LM is important to understand. For example, some later work in this overview will study the number of parameters within the underlying LM by excluding parameters in the embedding layer and only counting those contained within decoder-only layers.

**adaptation.** By pre-training LMs over a large corpus, we obtain a model that can accurately predict the next token given a sequence of tokens as context. But, _how do we use such a model to solve language understanding tasks like sentence classification and language translation?_ 

For modern LMs, the answer to this question is actually quite simple–we don’t change the LM at all. Instead, we exploit the generic nature of the model’s text-to-text input-output structure by providing textual “prompts” to the model, such as:

- “Translate this sentence to English: <sentence> =>”
    
- “Summarize the following document: <document> =>”.
    

Given these problem-solving prompts, a good LM should output a textual sequence that solves the problem for us! For problems in which we must choose from a fixed set of solutions (i.e., multiple choice or classification) instead of just generating text, we can use the LM to measure the probability of generating each potential solution and choose the most probable solution.

[Examples of Prompts](https://www.buildgpt3.com/)

**main takeaway.** The crux of modern LLMs is that we can use language model pre-training as a tool for creating generic foundation models that solve various problems without the need to adapt or fine-tune the model. Although prior LMs like GPT and GPT-2 [4, 5] perform poorly compared to fine-tuned or supervised language understanding techniques, such a learning framework is quite promising and—as we will see with GPT-3–can even perform quite well when the underlying LM becomes much larger.

### power laws

This overview will contain several references to the idea of [power laws](https://en.wikipedia.org/wiki/Power_law). For example, a paper may make a statement like the following:

> “The LM’s test loss varies as a power law with respect to the number of model parameters”. 

This sentence simply tells us that a relationship exists between two quantities–the loss and the number of model parameters–such that a change in one quantity produces a relative, [scale-invariant](http://felix.physics.sunysb.edu/~allen/540-05/scaling.html#:~:text=scale%20invariance%20of%20power%20law%20functions&text=The%20function%20y%3Dxp,by%20a%20scale%20factor%20a.) change in the other.

To make this a bit more concrete, a power law is expressed via the following equation.

[

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2Fddaa159e-1468-41a9-872a-7844a36e09ef_120x34.png)



](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2Fddaa159e-1468-41a9-872a-7844a36e09ef_120x34.png)

Here, the two quantities we study are `x` and `y`, while `a` and `p` dictate the shape/behavior of the power law between these quantities. Plotting this power law (with `a = 1`, `p = 0.5`, and `0 < x, y < 1`) yields the illustration below, where converting both axes to a log scale produces a signature linear trend that is characteristic of power laws.

[

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2Fc3b410c6-03f9-4214-8d6a-074b1cbbf6ec_800x300.png)



](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2Fc3b410c6-03f9-4214-8d6a-074b1cbbf6ec_800x300.png)

Depiction of a basic power law relationship between two quantities

Power laws simply tell us that one quantity varies as a power of another quantity. The work we will see in this overview considers an inverse version of a power law, as shown below.

[

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F56b20fb6-20b4-4fbf-8b9f-5c7d149862b3_188x89.png)



](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F56b20fb6-20b4-4fbf-8b9f-5c7d149862b3_188x89.png)

Notably, this is the same equation as before with a negative exponent for `p`. This negative exponent yields the graph shown below, where one quantity decreases as the other increases.

[

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2Faeb36a47-1a10-41c1-ad10-dd0a5a7df7d2_800x300.png)



](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2Faeb36a47-1a10-41c1-ad10-dd0a5a7df7d2_800x300.png)

Depiction of an inverse power law between two quantities

We will encounter power laws that resemble the figure above within our analysis of LMs. Namely, the LM loss tends to decrease according to a power law with respect to several different factors, such as the model or dataset size. We will expand upon this more in later sections.

### other useful details

In addition to the core ideas behind language modeling, there are a few additional concepts that might be helpful to know moving forward.

**distributed training.** The main idea of the papers within this overview is scaling up models like GPT and GPT-2 [4, 5] to make them better. As our models get bigger and bigger, however, training becomes more difficult due to an increase in computational and memory overhead. To help with this, we can leverage distributed training techniques, which use more hardware (i.e., more servers/GPUs) to make large-scale training processes more tractable and efficient.

There are a couple of different ways to distribute the training process for neural networks. One of these techniques is _data parallel training_, in which we:

1. Take a large mini-batch
    
2. Split this mini-batch into several, smaller sub-batches
    
3. Perform the computation related to each sub-batch in parallel on a different GPU
    
4. Accumulate the sub-batch results from each GPU into a centralized model update
    

Such an approach enables improved training efficiency by parallelizing model computation over a large mini-batch across several GPUs.

Somewhat differently, we can perform _model-parallel training_, which splits the model itself (i.e., instead of the mini-batch) across multiple GPUs. For example, we can send each layer of a model–or even smaller portions of each layer–to a separate GPU. Practically, this means that the forward pass is spread across several devices or GPUs that each contain a small portion of the underlying model. Such an approach enables larger models to be trained (i.e., because each GPU only stores a small portion of the model!) and can yield improvements in training efficiency via smart pipelining and parallelization of the model’s forward pass.

For the purposes of this overview, we just need to know that we can leverage distribution across many GPUs to make LLM training more tractable and efficient. Data and model parallel training are examples of popular distributed training techniques. Many considerations and alternative methodologies for distributed training exist–this is an entire field of study within deep learning that yields a lot of awesome, practical results.

To learn more, I would recommend checking out the following articles:

- Data and Model Parallel Training [[blog](https://analyticsindiamag.com/data-parallelism-vs-model-parallelism-how-do-they-differ-in-distributed-training/)]
    
- Making LM Training More Efficient [[LM blog](https://www.mosaicml.com/blog/billion-parameter-gpt-training-made-easy)] [[LLM blog](https://www.mosaicml.com/blog/gpt-3-quality-for-500k)]
    

**critical batch size.** Given that using large batches for data parallel training can benefit computational efficiency, we should just make our batches as big as possible, right? Well, this isn’t quite correct, as _(i)_ larger batches might deteriorate model performance and _(ii)_ increasing the batch size increases compute costs and requires extra hardware. Put simply, increasing the batch size too much has diminishing returns; see below.

[

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2Fd9f1d8e5-35e1-4edd-a463-ed5a9d88187f_2338x1254.png)



](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2Fd9f1d8e5-35e1-4edd-a463-ed5a9d88187f_2338x1254.png)

(from [3])

With this in mind, we might begin to wonder: _what’s the best batch size to use?_ This question was answered empirically with the proposal of the critical batch size in [3]. This work uses a metric called the _gradient noise scale_ to estimate the largest useful batch size across a variety of domains. Beyond this critical batch size, we start to see diminishing returns in terms of performance and compute efficiency. Because adopting different batch sizes can impact the efficiency and quality of training, some work–as we will see in this overview–adopts the critical batch size as a standard practice for resource efficient training.

**Beam search.** LM’s solve problems by outputting a textual sequence in response to a prompt. These sequences can be generated autoregressively by continually predicting the next word, adding this word to the input prompt, predicting another word, and so on; see the figure below.

[

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2Ff759c4d3-def7-4904-b7c7-210e33d4ae08_1880x1010.png)



](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2Ff759c4d3-def7-4904-b7c7-210e33d4ae08_1880x1010.png)

Generating autoregressive output with a language model

However, the greedy approach of continually predicting the most probable next word is not optimal! This is because the probability of a sequence of tokens (assuming each token is generated [independently](https://www.mathsisfun.com/data/probability-events-independent.html)) is the product of each word’s conditional probability given preceding tokens (i.e., due to the [chain rule](https://www.hackerearth.com/practice/machine-learning/prerequisites-of-machine-learning/bayes-rules-conditional-probability-chain-rule/tutorial/) of probability). Greedily choosing the most probable next token might not maximize this probability; e.g., initially choosing a low probability token might subsequently lead to higher probability tokens in the rest of the sequence.

Instead of testing all combinations of possible output tokens to find the best output sequence, we can find an approximate solution with _beam search_. The idea behind beam search is simple: instead of choosing the most probable next token at each step, choose the top-k most probable generations, maintain a list of possible output sequences based on these top choices, then select the most probable of these sequences at the end.

[

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F95e07e0d-4cd6-42f8-b0ed-d933fb49015e_1118x700.png)



](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F95e07e0d-4cd6-42f8-b0ed-d933fb49015e_1118x700.png)

Beam search with k=2

Thanks for reading Deep (Learning) Focus! Subscribe for free to receive new posts and support my work.

Subscribe

# Publications

We will now overview publications that predict [1] and empirically validate [2] the incredible practical utility of LLMs like GPT-3. From these publications, we will gain a better understanding of why LLMs are so powerful and see extensive analysis of their performance in practical applications.

### [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361) [1]

GPT and GPT-2 [4, 5] showed us that LMs have incredible potential as generic foundation models, but their performance when transferring to downstream tasks still leaves a lot to be desired. Thus, we might begin to ask: _how can we make these models better?_

In [1], authors study one potential direction for making LMs more powerful–scaling them up. In particular, they train a bunch of decoder-only LMs and analyze their test loss (i.e., cross-entropy language modeling loss over a hold-out test set) as a function of several factors, including:

- Model size
    
- Amount of data
    
- Amount of training compute
    
- Batch size 
    
- Architectural details (i.e., model width/depth, number of attention heads, etc.)
    
- Context length (i.e., number of tokens used to predict the next token)
    

This analysis reveals several fundamental properties of LM training behavior. For example, tweaking architectural details has minimal impact on LM performance if the total number of parameters is fixed. However, the LM’s test loss follows a power law with respect to model size, data size, and the amount of training compute across several orders of magnitude; see below.

[

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2Fda994e00-8afd-4b11-b29e-4a6d074f2a9a_2344x976.png)



](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2Fda994e00-8afd-4b11-b29e-4a6d074f2a9a_2344x976.png)

(from [1])

To make this a bit more clear, the authors in [1] consider three main factors: model size (N), data size (D), and the amount of training compute (C). To study scaling behavior with respect to any one of these factors, we _(i)_ make sure that the other two factors are sufficiently large (i.e., so they aren’t a bottleneck to performance), then _(ii)_ measure the LM’s test loss over a wide range of values for the factor we are studying. For example, to study the scaling properties of C, we make sure the model and dataset are sufficiently large, then measure LLM performance across different settings of C. We will now consider each of these factors individually.

**model size.** To study scaling properties with respect to model size, authors train different LM to convergence over the full dataset from [1]–WebText2, an extended version WebTest from GPT-2 [2] that is ~10X larger. Then, by adopting several LMs with different numbers of total parameters, we can obtain the figure shown below.

[

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F2af05f5c-d140-47bc-96d0-25469c55f856_2048x830.png)



](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F2af05f5c-d140-47bc-96d0-25469c55f856_2048x830.png)

(from [1])

By plotting the LM’s test loss as a function of the total number of parameters within the decoder-only layers (i.e., excluding all parameters in the embedding layer), we can see that LM loss follows a smooth power law with respect to N. In other words, _increasing the size of the LM yields a steady improvement in its performance_.

[

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2Ffd73449b-e9e4-4e05-b6cb-4f9a8c47527f_660x610.png)



](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2Ffd73449b-e9e4-4e05-b6cb-4f9a8c47527f_660x610.png)

(from [1])

**data and compute.** To study how LM performance scales with the amount of training data, authors of [1] adopt a sufficiently-large LM and perform separate training trails over differently-sized datasets. For each trial, the model is trained until the test loss begins to increase, an indication of overfitting. Again, this analysis shows us that test loss decreases according to a power law with respect to the size of the dataset; see above.

[

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2Fbd1b4ffe-d02a-47ed-bd1f-5c7f926065b2_714x628.png)



](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2Fbd1b4ffe-d02a-47ed-bd1f-5c7f926065b2_714x628.png)

(from [1])

We see a very similar trend when varying the amount of training compute, defined as C = 6NBS for batch size B and number of training iterations S. Given a sufficiently-large dataset and fixed batch size B, we can scan over numerous LM sizes N to obtain the result shown above. Here, we see that the optimal results for each compute budget C are achieved using different combinations of N and S, but the best LM loss decreases according to a power law with respect to the amount of training compute.

Going further, we can see from these results that LM sample efficiency (i.e., how many samples it takes for the model to perform well) improves with increasing N. To show this more clearly, authors of [1] analyze the performance of different-sized LMs with respect to the total number of samples observed during training, yielding the plot shown below. Here, we can clearly see LM performance improves more quickly as the models become larger.

[

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F72f34bc3-ce21-48e2-a4a9-5101c0b7e1f6_904x824.png)



](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F72f34bc3-ce21-48e2-a4a9-5101c0b7e1f6_904x824.png)

(from [1])

**pairwise scaling laws.** Beyond the power laws observed by analyzing N, D, and C in isolation, varying pairs of these factors simultaneously can also yield predictable behavior; e.g., by jointly varying N and D we can obtain the plot shown below. Here, we observe that _(i)_ larger models begin to overfit on smaller datasets and _(ii)_ LM loss follows a strict power law with respect to N given a sufficiently large dataset.

[

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2Fd7ca3ed5-acb1-4a5c-96d3-dade2313f900_1372x756.png)



](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2Fd7ca3ed5-acb1-4a5c-96d3-dade2313f900_1372x756.png)

(from [1])

At a high level, this tells us that we must make the dataset larger in order to avoid overfitting when we increase the size of the underlying LM. However, authors in [1] find that scaling the data size sub-linearly (i.e., proportional to `N^0.74` specifically) is sufficient to avoid overfitting.

**Takeaways.** Though we have discussed the power laws outlined in [1] at a high level, the actual publication makes these laws quite concrete and even proposes an accurate predictive framework for the test loss of any LM. For simplicity, we avoid these details here, instead focusing on the following takeaways for training LMs.

[

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F1e45145e-6913-4a83-847d-0b8aca7360f6_1436x672.png)



](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F1e45145e-6913-4a83-847d-0b8aca7360f6_1436x672.png)

How to properly invest more compute into LLM training (from [1])

If we are increasing the scale of LM training, we should:

1. Invest most of the extra compute into an increased model size (i.e., larger models are more sample efficient)
    
2. Increase the size of the dataset (but not as much as the model size) to avoid overfitting.
    
3. Slightly increase the batch size (i.e., according to the critical batch size [3]).
    
4. Stop training the model significantly short of convergence to optimize the use of training compute.
    

The power laws observed in [1] continue seemingly unimpeded for several orders of magnitude. Although this scaling will eventually reach a limit, it nonetheless shows that (properly) increasing the scale of LM training yields measurable performance benefits, hinting that exploring LLMs (like GPT-3) could prove to be incredibly beneficial.

> “Our results strongly suggest that larger models will continue to perform better, and will also be much more sample efficient than has been previously appreciated. Big models may be more important than big data.” - from [1]

Thanks for reading Deep (Learning) Focus! Subscribe for free to receive new posts and support my work.

Subscribe

### [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165) [2]

Prior work on GPT and GPT-2 [4, 5] began to reveal the utility of general purpose LMs for solving textual understanding tasks. However, these models still had limitations:

- GPT was not fully task-agnostic (i.e., required task-specific fine-tuning)
    
- GPT-2 performed far worse than supervised state-of-the-art in the zero-shot regime
    

Existing work provides a “proof of concept” that LMs could remove the need for task specification by performing zero/few-shot, task-agnostic inference. However, the poor performance of LMs relative to supervised techniques makes them less practical. Luckily, the power laws observed within [1] provide hope that larger LMs (i.e., LLMs) could narrow the gap between task-agnostic and task-specific/supervised performance.

Moving in this direction, GPT-3, which shares the same decoder-only architecture as GPT-2 (aside from the addition of some sparse attention layers [6]), builds upon the size of existing LMs by several orders of magnitude. In particular, it is an LLM with over 175 billion parameters (i.e., for reference, GPT-2 [5] contains 1.5 billion parameters); see below.

[

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F2d05fa6b-0603-48b2-ba60-9187a0bd38af_1262x406.png)



](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F2d05fa6b-0603-48b2-ba60-9187a0bd38af_1262x406.png)

(from [2])

With GPT-3, we finally begin to see promising task-agnostic performance with LLMs, as the model’s few-shot performance approaches that of supervised baselines on several tasks. Similar to GPT-2, authors pre-train the LLM using a language modeling objective, but they adopt a larger dataset based upon a filtered version of CommonCrawl and some additional, high-quality corpora. The breakdown of the full dataset used for pre-training is shown below.

[

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F1e0630ce-c5b2-4880-a8e7-6859f12a6b17_2210x700.png)



](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F1e0630ce-c5b2-4880-a8e7-6859f12a6b17_2210x700.png)

(from [2])

Pre-training with GPT-3 is conducted similarly to GPT-2, but the model is trained for much longer. To make the training process computationally feasible, the authors adopt a model parallel distributed training approach that distributes portions of each LM layer across separate GPUs. Because each GPU only stores a small portion of the full model, training can be conducted without exceeding memory constraints.

The learning process of GPT-3 has two components: un/self-supervised pre-training and in-context learning. These two components are illustrated in the figure below.

[

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F3f8961e8-7484-44e3-9a78-eb4d5365cf63_2214x1276.png)



](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F3f8961e8-7484-44e3-9a78-eb4d5365cf63_2214x1276.png)

(from [2])

Put simply, we first pre-train the general purpose LLM over a large unsupervised text corpus, then guide this model to solve downstream tasks using in-context learning. This in-context learning process can be performed via task-specific fine-tuning (as in GPT) or even using techniques like few-shot learning that require no gradient updates to the LM. The difference between fine-tuning and different variants of zero, one, and few-shot learning is depicted below.

[

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2Fcc1b83b2-6cb7-4a36-80a2-0d1ac53d013f_1580x1380.png)



](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2Fcc1b83b2-6cb7-4a36-80a2-0d1ac53d013f_1580x1380.png)

(from [2])

Unlike prior variants, GPT-3 is evaluated solely using zero and few-shot learning techniques. The authors do not adapt or fine-tune the model to any of the downstream datasets used for evaluation. Rather, they pre-train this incredibly large model over a massive text corpus and study whether in-context learning can be accurately performed using only few-shot prompting techniques that contain varying numbers of “in-context examples” as shown in the figure above.

By evaluating GPT-3 on a range of language understanding tasks, we immediately see that using a larger model significantly benefits few-shot performance. On sentence completion tasks, for example, GPT-3 improves the current state-of-the-art (i.e., including approaches that use supervised training or fine-tuning!) on several popular datasets, and providing more in-context examples seems to further improve performance; see below.

[

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F7c8b6ee1-49db-4981-90aa-17bf68141839_1392x1222.png)



](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F7c8b6ee1-49db-4981-90aa-17bf68141839_1392x1222.png)

(from [2])

On question answering tasks, we see that GPT-3 is outperformed by models like T5 [7] or RoBERTa [8]. However, these models perform extensive, supervised fine-tuning, while GPT-3 achieves comparable results via task-agnostic, few-shot inference. Put simply, GPT-3’s performance on these tasks is still impressive because it is a completely generic LLM that has not been specialized to solving these tasks in any way.

When evaluating GPT-3 on translation tasks, we observe that GPT-3 is better than state-of-the-art unsupervised neural machine translation (NMT) techniques at translating from other languages into English. Such results are surprising given that GPT-3’s pre-training set contains only 7% non-English content and no explicit mixing of or translation between languages. Interestingly, GPT-3 is much less effective at translating from English into other languages; see below.

[

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F0bc961b3-9741-41b5-9b70-22c5f81253c5_2028x774.png)



](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F0bc961b3-9741-41b5-9b70-22c5f81253c5_2028x774.png)

(from [2])

Authors also evaluate GPT-3 on the [SuperGLUE benchmark](https://super.gluebenchmark.com/), which contains a wide variety of different language understanding tasks. The results are summarized within the figure below, where we can see that _(i)_ using more in-context examples benefits GPT-3’s performance and _(ii)_ GPT-3 can even surpass the performance of popular, fine-tuned baselines like [BERT](https://cameronrwolfe.substack.com/p/language-understanding-with-bert) [9].

[

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F1e797415-7032-4ab3-b2d3-7ad2f8385a27_2154x1368.png)



](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F1e797415-7032-4ab3-b2d3-7ad2f8385a27_2154x1368.png)

(from [2])

Across all benchmarks, GPT-3 shows us that LLMs become more effective at task-agnostic, in-context learning as they grow in size. We can use in-context examples to prompt accurate responses from LLMs on a variety of tasks, making GPT-3 the first practical example of using general purpose LLMs to perform highly-accurate inference on a variety of downstream tasks without any task-specific modifications.

Despite the incredible leaps made by GPT-3 towards creating task-agnostic foundation models for language, these advancements come at a significant computational cost. GPT-3 was pre-trained on a special-purpose GPU cluster and its pre-training process required significantly more compute than any previous model that has been studied; see below.

[

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2Ffaa4655d-d69f-4115-b343-fc294984270a_1584x962.png)



](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2Ffaa4655d-d69f-4115-b343-fc294984270a_1584x962.png)

(from [2])

Although recent work has [drastically reduced the training cost of GPT-3](https://www.mosaicml.com/blog/gpt-3-quality-for-500k) (i.e., from >$10M in compute costs to <$500K), such foundation models are still not cheap to obtain. If we want to create our own foundation model like GPT-3, we better make sure it performs well. 

**open-sourcing GPT-3.** After the original proposal of GPT-3 in [2], the model was not publicly released. Rather, it was made accessible only via [paid APIs](https://openai.com/api/). Although the model’s API was heavily used, this lack of open-source access to the model itself (and its training code) hindered further analysis and experimentation.

To eliminate this issue, an open-sourced version of GPT-3, called OPT-175B, was created and analyzed in [10]. The release of OPT-175B also included a full code repository and several logbooks that provided valuable insights into the LLM training process. To learn more about OPT-175B (and see code you can use to train LLMs like GPT-3!), check out the overview below. 

[Learn about OPT-175B](https://cameronrwolfe.substack.com/p/understanding-the-open-pre-trained-transformers-opt-library-193a29c14a15)

# Takeaways

GPT models were originally proposed and explored with the goal of creating generic language models that are capable of solving a wide variety of tasks. These models operate under the assumption that if we can understand language modeling (i.e., predicting the next word within a sequence) at a very granular level, then we can generalize this understanding in a lot of useful ways without the need for task-specific fine-tuning or adaptation.

Initially, LMs like GPT and GPT-2 fell short of this goal. Their task-agnostic performance was far worse than supervised baselines. Within this overview, however, we have learned that increasing the scale of these LMs is a viable path forward in creating high-performing, task-agnostic models for language understanding. Eventually, this line of thinking led to the proposal and analysis of GPT-3, a massive LLM (i.e., ~100X bigger than GPT-2) that far surpassed the task-agnostic performance of prior LMs.

**scaling laws.** Scaling up LMs (i.e., using larger models, more data, and more compute) can drastically improve their performance. As we increase the scale of LM training, we learn from findings in [1] that we should _(i)_ significantly increase the size of the underlying model and _(ii)_ increase the amount of data used for pre-training (and the batch size) to a lesser extent. Larger language models are more sample efficient, and their performance improves as a power law with respect to model size, data size, and the amount of training compute across several orders of magnitude. In other words, _LMs get much better as we make them bigger_. 

**how much can we scale?** GPT-3 (an LLM with 175 billion parameters) empirically validates the trends outlined in [1] at an unprecedented scale. When we adopt this massive model and pre-train it over a large textual corpus, we see large improvements in task-agnostic, few-shot performance. GPT-3 is still outperformed by supervised techniques on several baselines, but findings in [2] provide clear evidence that LLMs improve in their ability to perform in-context learning as they grow in size. Though GPT-3 is technically similar to GPT-2, training a model of this scale is a feat of engineering that demonstrates the incredible potential of language foundation models.

