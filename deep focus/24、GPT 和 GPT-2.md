
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

# 25、扩展定律和 GPT-3

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

