
> https://cameronrwolfe.substack.com/p/nano-moe


鉴于许多前沿 LLMs 开始使用基于 MoE 的架构，深入理解 MoE 变得重要。在这篇文章中，我们将朝这个方向迈出一步，通过在 PyTorch 中从零开始构建（并预训练）一个中型 MoE  模型——称为nanoMoE。nanoMoE 的所有代码都在该[仓库](https://github.com/wolfecameron/nanoMoE)中可用，该仓库是 [Andrej Karpathy](https://karpathy.ai/) 的 [nanoGPT](https://github.com/karpathy/nanoGPT) 库的一个分支，已扩展以支持 MoE 预训练。

### 1、仅解码器 Transformer 的基础知识

仅解码器 Transformer 更常用于现代 LLMs，它简单地从架构中移除了编码器，仅使用解码器。实际上，这意味着仅解码器 Transformer 架构的每一层包含以下内容：

1. 一个掩码自注意力层。
2. 一个前馈层。

要形成完整的仅解码器 Transformer 架构，我们只需将这些结构相同但权重独立的层堆叠 $L$ 次即可：

![|550](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F414bf0b5-2043-4fb5-bdab-e0153f893861_1634x808.png)


#### 1.1 从文本到 Tokens

模型的输入是一个 token 向量列表。如果我们将文本作为输入传递给模型，_我们如何从文本输入生成这些向量呢？_

![|350](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F56dd3364-44d1-4587-a0b8-3909f1f02f31_1132x282.png)


**分词**    构建 LLM 输入的第一步是将原始文本输入——_一个字符序列_——分解为离散的 token。其中 Byte-Pair Encoding (BPE) 分词器是最常见的。

![[Pasted image 20250318174119.png|600]]

用于训练和与 LLM 交互的软件包（例如 HuggingFace 或 torchtune）提供了与分词器交互的接口。此外，OpenAI 发布了 tiktoken 包，用于与 GPT 分词器交互。
```
Original Text: This raw text will be tokenized 
Tokens: ['This', 'Ġraw', 'Ġtext', 'Ġwill', 'Ġbe', 'Ġtoken', 'ized']
```

这里，`Ġ` 字符表示一个词元紧跟在空格之后。这些特殊字符是依赖于分词器的。例如，许多分词器使用 `#` 字符来表示一个词的延续，这样在上述序列中，最后两个词元会是 `['token', '#ized']`。

**词汇表**：LLM 理解的词元集（即分词器生成的那些）是固定的，词汇表大小在不同模型之间有所不同，并取决于多个因素（例如，多语言模型往往有更大的词汇表），但对于最近的 LLM，词汇表大小在 64K 到 256K 个词元之间相对常见。

![|500](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fb8aadf17-3bf6-4b79-9688-b6bfbc5840b1_1830x888.png)

**词元 ID 和嵌入**：LLM 词汇表中的每个词元都与一个唯一的整数 ID 相关联。例如，之前的代码在对文本进行分词时生成了以下 ID 序列：`[1986, 7112, 1467, 686, 387, 3950, 1506] `。这些 ID 中的每一个都与一个向量相关联，称为词元嵌入，位于一个嵌入层中。嵌入层实际上是一个大型矩阵，存储了许多行的向量嵌入。要检索一个词元的嵌入，我们只需查找嵌入层中对应的行——由词元 ID 给出。

![|400](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fe2f723f2-056a-4fc0-a3f7-7aa151fe297e_1194x1026.png)

词元嵌入矩阵的大小为 $[C, d]$，其中 $C$ 是输入中的词元数量，$d$ 是 LLM 采用的词元嵌入维度。通常，我们有一个包含 $B$ 个输入序列的批次，而不是单个输入序列，形成大小为 $[B, C, d]$ 的输入矩阵。维度 $d$ 影响 Transformer 内所有层或激活的大小，因此 $d$ 是一个重要的超参数选择。在将这个矩阵作为输入传递给 Transformer 之前，我们还需要为输入中的每个词元添加一个位置嵌入，以向 Transformer 传达每个词元在其序列中的位置。

#### 1.2 (掩码和多头)自注意力
![|450](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fe97978e4-cc11-41e0-8fb4-0010039c3769_1456x818.webp)
**什么是自注意力**    简单来说，自注意力根据序列中每个标记与其他标记的关系来转换每个标记的表示。直观地说，自注意力基于序列中与该标记最相关的其他标记（包括其自身）来调整每个标记的表示。换句话说，_我们学习在理解序列中某个标记的含义时应该“关注”哪些标记_。

**缩放点积注意力**    给定大小为 $[C, d]$ 的输入标记矩阵，我们首先通过三个独立的线性投影来投影输入，形成三组独立的标记向量。这些投影称为键、查询和值投影：

这种命名方式可能看起来是随机的，但它源于信息检索的先前研究。每个投影名称的直观解释如下：

- *Query* 是用于搜索信息的。它代表我们想要为其找到序列中其他相关标记的当前标记。
- *Key* 代表序列中的其他标记，并作为索引将查询与序列中其他相关标记匹配。
- *Value* 是一旦查询匹配键后检索到的实际信息。值用于计算自注意力中每个标记的输出。

**计算注意力分数**    在投影输入后，我们为输入序列中的每对标记 $[i, j]$ 计算一个注意力分数 $a[i, j]$。直观上，这个范围在 $[0, 1]$ 之间的注意力分数捕捉了给定标记应该“关注”序列中另一标记的程度——_更高的注意力分数表示一对标记彼此非常相关。_ 

这个操作形成一个大小为 $[C, C]$ 的矩阵——称为注意力矩阵——其中包含整个序列的所有成对注意力分数。接下来，我们将注意力矩阵中的每个值除以 $\sqrt{d}$ ——这种方法被发现可以提高训练的稳定性——然后对注意力矩阵的每一行应用 softmax 操作。应用 softmax 后，注意力矩阵的每一行形成一个有效的概率分布——每一行包含正值且总和为 1。注意力矩阵的第 $i$ 行存储了第 $i$ 个标记与序列中每个其他标记之间的概率。

**计算输出**    一旦得到了注意力分数，推导自注意力的输出就很简单了。每个标记的输出是值向量的加权组合，其中权重由注意力分数给出。为了计算这个输出，我们只需将注意力矩阵乘以值矩阵即可。值得注意的是，自注意力保留了其输入的大小——对于输入中的每个标记向量，会生成一个转换后的 $d$ 维输出向量。

**掩码自注意力**    仅解码器的 Transformer 使用掩码自注意力，通过“掩盖”序列中每个标记之后的标记来修改基础注意力模式。每个标记只能考虑在其之前的标记——后续标记被掩盖。

![|450](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fa3d910cc-fd59-45dd-b2b6-9452a6f69bf0_2316x694.png)

**掩码自注意力禁止我们向序列的后方查看！** 实际上，这是通过将这些标记的所有注意力分数设置为负无穷大来实现的，在应用 softmax 后，掩码标记的成对概率为零。

**注意力头**    我们描述的注意力操作使用 softmax 来归一化整个序列的注意力分数。虽然这种方法形成了有效的概率分布，但也限制了自注意力在序列中关注多个位置的能力——*概率分布很容易被一个（或几个）词主导*。为了解决这个问题，我们通常并行计算多个“头”的注意力。

具体地说，我们将每个注意力头中向量的维度从 $d$ 改为 $d // H$，其中 $H$ 是注意力头的数量，以保持多头自注意力的计算成本（相对）固定。

![|500](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F8c1a2682-07ad-4daa-a3ae-f4d3c59d9fb0_2194x992.png)

现在，我们有多个注意力头并行计算自注意力。然而，我们仍然需要从多个注意力头中生成单一的输出表示。我们有几种选择来组合每个注意力头的输出，例如：连接、平均、投影等。然而，标准的多头自注意力实现如下：

- 连接每个头的输出。
- 线性投影连接后的输出。

由于每个注意力头输出的标记向量维度为 $d // H$，所以所有注意力头连接后的输出维度为 $d$。因此，多头自注意力操作仍然保持输入的原始大小。

![[Pasted image 20250319094811.png|600]]

**代码实现**   在这里，我们处理大小为 $[B, C, d]$ 的输入批次：

* 第 52-59 行：为每个注意力头计算 K、Q 和 V 的投影，并根据需要拆分/重塑它们。
* 第 62-65 行：计算注意力得分，对其进行掩码处理，然后对结果应用 softmax 变换。
* 第 68 行：通过将注意力矩阵与值矩阵相乘来计算输出向量。
* 第 71-72 行：将每个注意力头的输出连接起来，并应用线性投影以形成最终输出。


#### 1.3 前馈转换层

![|600](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F252f6acf-2ef1-4531-8ce4-2dce7778f1a0_1870x564.png)

除了掩码自注意力机制之外，Transformer 的每个模块还包含一个逐点前馈变换。这个变换将序列中的每个标记向量通过相同的前馈神经网络。通常，这是一个具有非线性激活（例如 ReLU、GeLU 或 SwiGLU）的两层网络。在大多数情况下，隐藏层的维度比标记嵌入的原始维度更大（例如，大 4 倍）。在 PyTorch 中实现前馈神经网络很容易，可以使用 Linear 模块。

![[Pasted image 20250319104124.png|600]]

#### 1.4 仅解码器 Transformer 块

![|300](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F1da32a13-6bcf-4b1a-a276-fad3f4315c58_906x1110.png)

为了构建一个仅解码器的 Transformer 块，我们使用了之前提到的两个组件——*掩码自注意力和前馈转换*，并在组件之间加入归一化操作和残差连接。

**残差连接** 简单地将神经网络层的输入与该层的输出相加，然后将此表示传递给下一层。残差连接在深度学习中被广泛使用，可以应用于任何类型的神经网络层。添加残差连接有助于避免**梯度消失/爆炸**问题，并通过提供“捷径”来改善训练的稳定性，使梯度在反向传播过程中能够顺畅流动。

![|400](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Ff125254a-28af-43c4-80e0-d2273b1702c9_1888x612.png)

对神经网络层的输入（或输出）进行**归一化**也有助于训练的稳定性。虽然存在多种归一化方法，但对于 Transformers/LLMs，最常用的归一化变体是 Layer 归一化，有两个组成部分：

1. 执行归一化。
2. 应用（可学习的）仿射变换。

换句话说，我们将归一化后的值乘以权重并加上偏置，而不是直接使用归一化输出。权重和偏置都是可学习的参数，可以与其他网络参数一起训练。

![[Pasted image 20250319132446.png|600]]


#### 1.5 仅解码器 Transformer jia架构

我们只需重复相同的块 $L$ 次即可！对于每个块，模型输入的大小 $[B, C, d]$ 保持不变，因此第 $L$ 个仅解码器 Transformer 块的输出也是这个大小的张量

![|500](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F98768d59-2bb6-442d-a84d-4fc9e5f1dd9f_1736x934.png)


下面提供了一个完整的仅解码器 Transformer 架构的实现。该架构包含多个组件，包括两个嵌入层（即用于 tokens 和位置的嵌入层）、所有 $L$ 个 Transformer 块，以及一个最终的分类模块——包括层归一化和线性层——用于在给定输出 token 嵌入作为输入的情况下执行下一个 token 的预测。模型通过将其输入——大小为 $[B, C]$ 的输入 token ID 集——依次传递给这些组件来生成一组输出 token ID。

![[Pasted image 20250319142644.png|600]]

**生成输出（解码）**    大型语言模型（LLMs）专门用于执行下一个 token 预测。如我们所知，模型的输出只是与每个输入 token 对应的输出 token 向量列表。因此，我们可以通过以下步骤预测任何输入 token 的下一个 token：

1. 获取特定 token 的输出嵌入。
2. 将此嵌入传递通过一个线性层，输出大小为模型词汇表的维度。
3. 对模型的输出进行 argmax 以获得最大 token ID。

要生成一段文本，我们只需继续重复此过程。我们将文本提示作为输入，传递通过仅解码器 Transformer，获取输出序列中的最后一个 token 向量，预测下一个 token，将此下一个 token 添加到输入序列中并重复。这种自回归解码过程被所有 LLMs 用于生成输出。

### 2、创建 MoE 模型

将模型架构转换为 MoE 并不困难，但有许多细节必须正确实现才能使模型表现良好。此外，正确训练这些模型需要额外的关注和理解——_MoE 模型比标准 LLM 更难训练_。

#### 2.1 专家层

与标准的仅解码器 Transformer 相比，MoE 模型的主要修改在于 Transformer 块的前馈组件。通常，这个块有一个前馈网络，以逐点方式应用于所有 token 向量。MoE 则创建了多个前馈网络，每个都有其独立的权重。

**PyTorch 实现**    在 PyTorch 中实现的主要复杂性在于我们不使用 PyTorch 中的标准 Linear 层。相反，我们将所有专家的权重包装成多个 Parameter 对象，这样我们可以使用 batch matrix multiplication 操作来批量计算所有专家的输出。此实现避免了逐个循环计算每个专家的输出，从而大大提高了效率。

![[Pasted image 20250319162614.png|600]]


**创建一个 MoE**    要创建一个基于 MoE 的仅解码器 Transformer，我们只需将 Transformer 的前馈层转换为 MoE（或专家）层。MoE 层中的每个专家的架构与该层中原始前馈网络相同。我们只是在一个专家层中拥有多个原始前馈网络的独立副本。

然而，我们不需要在 Transformer 的每个前馈层中使用专家。大多数基于 MoE 的大型语言模型使用步长 $P$，这意味着每第 $P$ 个层被转换为专家层，而其他层保持不变。

> _“ST-MoE 模型有 32 个专家，专家层的频率为 1/4（每第四个 FFN 层被替换为 MoE 层）。”_

这种想法的高级实现见下方伪代码。这些“交错” MoE 层控制了 MoE 中的专家总数，是平衡性能和效率的有用机制。

```python
transformer_blocks = []
for i in range(num_blocks):
    use_moe = (i % P) == 0

    # when use_moe = False, this is regular transformer block
    # when use_moe = True, this is an expert layer
    transformer_blocks.append(Block(use_moe=use_moe))
```

#### 2.2 Routing Tokens to Experts

MoE 架构的主要优势在于其效率，但仅使用专家并不能提高效率！实际上，为模型的每一层添加更多专家会显著增加模型的总参数数量——以及所需的计算量。为了提高效率，我们需要在每一层中稀疏地选择和使用专家的子集。通过稀疏利用专家，我们可以在不显著增加训练和推理计算成本的情况下，获得更大模型的好处。

> “使用 MoE 架构可以比密集模型通常实现的更好地平衡模型质量和推理效率。” 

**选择专家**    让我们考虑一个单一的标记——由一个 $d$ 维的标记向量表示。我们的目标是选择一个专家子集（大小为 $k$）来处理这个标记。在 MoE 文献中，我们通常说该标记将被“路由”到这些专家。我们需要一个算法来计算和优化这个路由操作。

![|350](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F1189a50c-ad49-4e09-8fca-b800532e101a_1156x856.png)

最简单的路由算法是对标记向量应用线性变换，形成一个大小为 $N$（即专家数量）的向量。然后，我们可以应用 softmax 函数，为标记在专家集合上形成概率分布。我们可以使用这个分布选择标记应路由到的专家，通过选择分布中的前 $K$ 个专家。前 $K$ 个值——“专家概率”——也很重要。

**简单路由器实现**    如上所述，这种路由机制实际上非常简单——它只是一个线性层！下面展示了这种 softmax 路由器的实现，其中路由器的输出是：

1. 输入中每个标记的前 $K$ 个专家索引。
2. 与选择的专家相关联的前 $K$ 个专家概率（即每个前 $K$ 个索引的概率值）。

尽管简单，这种路由机制有效且能很好地发挥作用。大多数现代 MoE 都采用类似的线性路由方案与 softmax。

我们可以选择在路由机制中添加噪声，这是在 [8] 中提出的方法——这是将 MoE 应用于神经网络的最早研究之一。通过在路由机制的输出中添加少量（可学习的）噪声（详情见下文），可以帮助正则化 MoE 的训练过程。

![|450](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fe6453620-af80-438f-b824-80a41a86a822_1916x1132.png)

**激活参数**    因为在 MoE 层中我们只选择一部分专家来处理每个标记，所以在 MoE 文献中有“活跃”参数的概念。简单来说，只有 MoE 模型总参数中的一小部分——由每个 MoE 层中选择的专家决定——在处理给定标记时是活跃的。MoE 的总计算量与活跃参数的数量成正比，而不是与总参数数量成正比。

#### 2.3 专家容量

> _“为了提高硬件利用率，大多数稀疏模型的实现对每个专家使用固定的批量大小。专家容量指的是可以路由到每个专家的 token 数量。如果超过这个容量，溢出的 token 会通过残差连接传递到下一层。”_ 

在专家层中执行的计算是动态的。我们根据路由器的输出选择每个专家要计算的 token，这会根据输入到 MoE 的 token 序列而变化。每个专家的输入是动态且不可预测的，这使得专家层的实现变得有些复杂：_我们如何处理每个专家的输入大小不同且不可预测的情况？_

![|400](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fde21dffa-d12f-479e-92e5-617f48c9f4d1_2368x934.png)

**专家容量**    大多数实际的 MoE 实现通过为每个专家使用固定的批量大小来避免这个问题——_这是一种提高硬件利用率的有效技巧_。每个专家使用相同的静态批量大小，被称为“专家容量”。专家容量——_在上述方程中定义_——决定了每个批次中可以发送到任一专家的最大 token 数量。

专家容量通过容量因子设置进行控制。容量因子为 1 意味着 token 被均匀路由，而将容量因子设置为大于 1 则提供额外的缓冲，以处理专家之间不平衡的 token 路由——_这会导致更高的内存使用和较低的效率_。

![|400](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F417c5fc8-2524-48e1-a9ef-460b4476d323_1784x1184.png)

如果路由到某个专家的 token 数量超过了专家容量，那么我们会“丢弃”这些额外的 token，通过不进行计算，让它们的表示通过变压器的残差连接直接流向下一层；参见上文。MoE 在相对较低的容量因子下表现良好，但我们应确保避免丢弃过多的 token。容量因子在训练和评估期间也可以不同；例如，ST-MoE 在训练和评估期间分别使用 1.25 和 2.0 的容量因子。

**PyTorch 实现。** 现在我们了解了专家容量和专家层内路由的细节，我们需要实现一个功能齐全的路由器。这个路由器将与我们之前的实现共享相同的逻辑（即线性层加 softmax），但它将通过为每个专家创建固定大小的输入张量来超越这种实现；详见下文。鉴于这是一个功能齐全的实现，下面的路由器比之前更复杂。然而，我们可以将此实现提炼为以下组件：

- _第 41-47 行_：计算（带噪声的）线性路由器的输出。
- _第 49-52 行_：计算前 `K` 个专家及其相关概率。
- _第 55-58 行_：计算专家容量。
- _第 60-88 行_：使用高级 PyTorch 索引和张量操作来构建专家输入的批次。
- _第 90-93 行_：构建最终的专家输入批次。

#### 2.4 负载均衡和辅助损失

> _“门控网络往往会收敛到一种状态，即总是为同几个专家生成较大的权重。这种不平衡是自我强化的，因为被偏爱的专家训练得更快，因此门控网络会选择它们更多。”_ 

到目前为止，我们设计的路由系统并没有明确鼓励在每一层中平衡选择专家。因此，模型会收敛到一种状态，即对每个 token 重复选择相同的几个专家，而不是充分利用所有专家。这种现象被称为“路由崩溃”。

![|500](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F644cec45-8dff-491e-9d41-e53ee4b0c7df_1574x764.png)

**负载均衡损失**    为了在训练期间鼓励平衡选择专家，我们可以在训练损失中添加一个额外的组件，奖励模型均匀利用其专家。具体来说，我们创建了一个辅助损失项，用于衡量专家的重要性（即分配给每个专家的概率）和负载均衡（即发送给每个专家的 token 数量）。这种方法在 [2] 中提出，作者创建了一个损失，考虑了两个量：

1. 分配给每个专家的路由概率的比例。
2. 派遣到每个专家的 token 的比例。

如果将这两个量存储在各自的 `N` 维向量中，可以通过取这两个向量的点积来创建一个单一的损失项。当专家接收到均匀的概率和负载时，该损失最小化。

下面是 PyTorch 中实现的负载均衡损失。此实现包含以下关键组件：

- **第 9-17 行**：定义用于计算负载均衡损失的所有常量和输入张量。
- **第 19-24 行**：计算发送给每个专家的 token 比例。
- **第 26-27 行**：计算分配给每个专家的概率比例。
- **第 29-31 行**：对每个专家的 token 比例和概率进行（缩放）点积。


**路由器 z 损失** 为了补充负载均衡损失，[3] 中的作者提出了一个额外的辅助损失项，称为路由器 z 损失。路由器 z 损失限制了由路由机制预测的 logits 的大小（注意：这是在应用 softmax 之前，不是概率）。具体公式如下。

![|500](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F1790688e-5328-45f2-98c0-717ba6041470_2090x636.png)

我们不希望这些 logits 过大，因为路由器包含一个（指数）softmax 函数。然而，这些 logits 在训练过程中可能变得非常大，导致舍入误差，从而破坏训练过程的稳定性——即使使用全精度（`float32`）。路由器 z 损失鼓励 MoE 保持这些 logits 较小，从而避免舍入误差。

> “路由器以 float32 精度计算专家的概率分布。然而，在最大规模时，我们发现这不足以提供可靠的训练。” 

下面是路由器 z 损失的实现，包含三个关键步骤：

1. **第 8-14 行**：创建计算路由器 z 损失所需的输入张量（即来自路由机制的 logits）。
2. **第 21 行**：对路由器 logits 进行平方的 logsumexp。这是应用指数、求和和对数运算的数值稳定简写。
3. **第 24 行**：对所有 token 的上述操作结果求和，并除以 token 总数（即取平均值）。

**结合辅助损失**    鉴于存在多个辅助损失，我们可能会想在实践中应该使用哪些。答案是：**全部使用**！我们可以在训练期间将这些损失加到标准语言模型损失中。每个辅助损失都会乘以一个缩放因子，然后将所有（缩放后的）损失相加。默认的缩放因子分别为负载均衡损失的 `0.001` 和路由器 z 损失的 `0.01`。

![|500](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F726a1e49-0aaa-45dd-a9d0-5386edc2ecc1_2522x288.png)

**当前研究**    如我们所见，本节中介绍的辅助损失效果很好。然而，最近的研究 [8] 表明，*根据缩放因子的设置方式*，这些辅助损失在某些情况下可能会牺牲模型性能以换取训练稳定性。因此，训练 MoE 的最佳过程和策略仍然是一个非常活跃的研究领域。

![|500](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F551dc85c-ee09-412d-b6d2-922a60c8badb_1036x310.png)

例如，最近提出的 DeepSeek-v3 模型——用于创建 DeepSeek-R1 推理模型的基础模型——采用了一种无辅助损失的负载平衡策略。在选择前 `K` 个专家时，该策略仅对路由器输出添加动态偏差。对于未被充分选择的专家增加偏差，对于被选择过多的专家减少偏差，从而**提高未充分利用的专家被选择的机会**。这种动态偏差在不牺牲模型性能的情况下改善了负载平衡。然而，研究 [8] 中仍然使用了负载平衡损失（只是缩放因子更小）。

> *“我们持续监控每个训练步骤整个批次的专家负载。在每个步骤结束时，如果对应的专家负载过重，我们将减少偏差项；如果负载不足，我们将增加偏差项，其中 𝛾 是一个称为偏差更新速度的超参数。”*

#### 2.5 仅解码器 MoE shi'x实现

![|500](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F51369997-34e1-41d9-a3de-83a3edcde279_2192x912.png)

我们现在已经了解了专家层的所有主要组件。接下来，让我们将这些概念结合起来，创建一个完整的基于 MoE 的仅解码器架构。该模型中的 MoE 块（如上所示）将包含：

- 一个常规（带掩码）的自注意力层
- 一个专家层——用在模型的每第 `P` 层，代替普通的前馈层。

这种块结构类似于标准的仅解码器 Transformer，但在模型的部分层中，我们将前馈层替换为专家层，从而形成一个 MoE 块。首先，让我们介绍一些关于如何计算专家层最终输出的剩余细节。然后，我们将展示一个基于 MoE 的仅解码器 Transformer 的完整实现。

![|500](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fe03c2af3-0014-41f8-9c2b-d79bbb75265e_1674x1188.png)

**计算专家层输出**    一旦我们使用路由机制确定了给定标记的活跃专家集，就可以按以下步骤计算该专家层的最终输出：

1. 将标记发送到其活跃专家。
2. 计算这些标记的活跃专家输出。
3. 对每个标记的专家输出进行加权平均，权重即为路由器分配给每个活跃专家的概率。

上述过程在图中为单个标记进行了展示。最近关于 MoE 的研究还引入了“共享”专家的概念，这些专家对所有标记始终处于活跃状态。共享专家略微修改了路由逻辑，但上述核心思想仍然适用；有关此主题的更多详细信息，请参见[此处](https://cameronrwolfe.substack.com/i/154340424/computing-the-output-of-an-moe-layer)。

下面提供了完整专家层的实现，我们看到这些思想在 PyTorch 中的应用。在第 49 行，我们从路由器获取每个专家的数据批次及其关联的专家概率。然后，我们将这些批次通过专家前馈网络（第 52 行）以获得每个专家的输出。最后，我们在第 54-58 行将每个专家的输出乘以关联概率，从而形成专家层的最终输出。

**在 PyTorch 中实现 MoE**    现在，我们可以修改仅解码器的 Transformer 块，选择性地用专家层替代通常的前馈层。通过下面的代码实现这一点，我们将 `MLP` 模块直接替换为新的 `MoELayer`，从而形成一个 `MoEBlock`。

从这里开始，我们的 MoE 架构的最终实现与之前的仅解码器 Transformer（`GPT`）实现完全一致。唯一的变化是我们将每第 `P` 个 `Block` 替换为 `MoEBlock`。由于代码与之前定义的 `GPT` 模型完全相同，只是插入了 MoE 块，因此我们将避免在此处详细写出实现。


### 3、从头开始预训练 nanoMoE

现在我们了解了 MoE 的工作原理，让我们使用这种架构从头开始预训练一个大型语言模型（LLM）。下面的仓库中提供了基于 MoE 的 LLM 的完整实现。这个实现——称为 nanoMoE——基于 [Andrej Karpathy](https://karpathy.ai/) 的 [nanoGPT](https://github.com/karpathy/nanoGPT) 仓库。然而，原始的 GPT 架构已被修改为使用基于 MoE 的仅解码器 Transformer 架构。[nanoMoE 仓库](https://github.com/wolfecameron/nanoMoE)

nanoMoE 仓库重用了我们在本文中看到的所有 MoE 组件的代码。这个实现的关键组件包括：

- _模型实现_：查看 `GPT` 模型定义，添加了构建 MoE 模型的能力。[链接](https://github.com/wolfecameron/nanoMoE/blob/master/model.py)
- _训练_：所有训练代码都在一个文件中，未对原始 nanoGPT 代码进行实质性修改。[链接](https://github.com/wolfecameron/nanoMoE/blob/master/train.py)    
- _数据集_：nanoMoE 在 OpenWebText 数据集的 250 亿个 token 子集上进行预训练（与 nanoGPT 相同，但 token 较少）。[链接](https://github.com/wolfecameron/nanoMoE/tree/master/data/openwebtext)
- _配置_：用于预训练 nanoMoE 的最终训练配置，我们将在下一节中解释，可以在[这里](https://github.com/wolfecameron/nanoMoE/blob/master/config/train_nano_moe.py)找到。

在本节中，我们将进一步概述成功预训练 nanoMoE 的最佳实践，回顾预训练的结果，并概述为这个中型 MoE 模型发现的最佳预训练设置。

#### 3.1 MoE 训练的最佳实践

> _“尽管 MoE 取得了一些显著成功，但由于复杂性、通信成本和训练不稳定性，其广泛采用受到阻碍。”_ 

虽然 MoE 很早就被提出，但它们在大型语言模型研究中的受欢迎程度直到最近才显著提高。多年来，MoE 难以使用是其普及的主要障碍。相较于密集模型，MoE 更复杂，且在训练过程中通常容易不稳定。

**为什么 MoE 不稳定**    正如我们所见，基于 MoE 的 LLM 仅对仅解码器的 transformer 架构进行了轻微修改。考虑到这一点，我们可能会想：_MoE 架构中究竟是什么导致了训练的困难？_ _为什么 MoE 的训练比标准 LLM 更不稳定？_

![|400](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F213eacf6-6f4c-48ac-9fec-b81a24580b4b_1370x804.png)

训练 MoE 时会出现两个主要问题：

1. *路由崩溃*：模型反复使用相同的专家。
2. *数值不稳定*：MoE 可能会遇到舍入误差，尤其是在路由器中（由于 softmax 中指数的使用）。

这些问题导致训练不稳定，意味着模型的损失可能在训练过程中发散；请参见上文关于 nanoMoE 训练的具体示例。当这种情况发生时，我们需要停止训练过程并从保存的检查点重新开始，这既耗时又低效（即，GPU 空闲时间很多）。理想情况下，我们希望有一个稳定的训练过程，以避免这些不稳定性。接下来，我们讨论改善 MoE 训练稳定性的最佳实践。

**辅助损失**    如前所述，在训练 MoE 时，我们不必在辅助损失之间做选择。相反，我们可以将多个辅助损失合并为单个损失函数。在 nanoMoE 的情况下，我们在训练中使用标准的辅助负载平衡损失和路由器 z-loss。使用正确的辅助损失可以改善训练稳定性，确保专家的均匀使用并避免训练中的路由崩溃。

**训练精度**    在训练 LLM 时，通常使用混合精度训练，这会将模型的某些组件转换为较低的 `float16` 或 `bfloat16` 精度格式，而不是完整的 `float32` 精度。PyTorch 中的自动混合精度 (AMP) 模块自动支持此功能，可以显著降低训练成本而不损害模型性能。换句话说，这是一种“免费”的预训练加速，可以通过最小的代码更改轻松启用。

> _“与 BF16 基线相比，我们的 FP8 训练模型的相对损失误差始终低于 0.25%，这一水平完全在训练随机性的可接受范围内。”_

混合精度已使用了一段时间，但研究人员最近探索了进一步降低 LLM 训练精度的方法——低于 16 位。例如，DeepSeek-v3 使用 8 位精度进行训练。然而，随着训练精度的降低，保持相同水平的模型质量变得更加困难。使用 `FP8` 精度实现大规模 LLM 训练需要新颖且复杂的量化技术。否则，以如此低的精度训练 LLM 可能会对模型性能产生负面影响。

```python
with torch.amp.autocast(device_type='cuda', enabled=False):
    # AMP 在此代码块中被禁用！
    <路由器代码在此处>
```

**这与 MoE 有何相关**    正如我们之前提到的，MoE 中的路由机制容易出现数值不稳定。在较低精度下计算路由器的输出会使这个问题更糟！在 [6] 中明确指出了这一问题，作者发现低精度训练导致路由器中出现较大的舍入误差。为了解决这个问题，我们必须在训练中即使使用 AMP 也要以完整 (`float32`) 精度运行路由器，这可以通过在 MoE 的路由机制中简单地禁用 AMP 来实现；见上文。

**权重初始化**    传统上，稳定训练大型神经网络的关键因素之一是使用正确的权重初始化策略，例如 Glorot 或 He 初始化。这些技术以及批量归一化等策略，解锁了训练极深层神经网络的能力，这在以前是相当困难的。对于 LLM，我们通常采用这些权重初始化策略。然而，[6] 中的作者建议采用专为 MoE 设计的略微修改的权重初始化方案。

```python
# 在 torch 中，线性层的维度是翻转的 ([out_dim, in_dim])
w_fan_in = module.weight.shape[-1]
w_std = (scale / w_fan_in) ** 0.5
torch.nn.init.trunc_normal_(
    module.weight,
    mean=0.0,
    std=w_std,
    a=-2*w_std,
    b=2*w_std,
)
```

这种权重初始化策略从一个均值为零 (`µ = 0`) 的截断正态分布中采样权重，其标准差为 `σ = \sqrt{s/n}`，其中 `s` 是一个缩放超参数，`n` 是被初始化层的输入大小（即，[fan-in 策略](https://stackoverflow.com/questions/42670274/how-to-calculate-fan-in-and-fan-out-in-xavier-initialization-for-neural-networks)）。[6] 中的作者还建议使用缩小的缩放超参数 `s = 0.1` 来“提高质量并减少训练不稳定的可能性”。上面提供了在 PyTorch 中实现的这种修改后的权重初始化策略。

**MoE 微调**    在此概述中，我们只关注 nanoMoE 的预训练。然而，我们也应该意识到，与标准密集模型相比，MoE 的微调可能更加困难。特别是，由于 MoE 拥有大量参数，因此容易过拟合。这些大型模型适合在大规模数据集上进行预训练，但在小量数据上进行微调时可能会过拟合。我们应该意识到这个问题，并尽力在微调 MoE 时防止过拟合（例如，通过提高 dropout 比例）。我们将 nanoMoE 的微调探索——以及防止过拟合——留作未来的工作。


#### 3.2 nanoMoE 预训练实验

现在我们了解了如何稳定地训练 MoE 的各种技巧，让我们通过从头开始预训练 nanoMoE 来实际测试它们。要亲自测试这些命令，你需要访问一个或多个 GPU。在这里的实验中，我使用了个人工作站上的两块 RTX 3090 GPU。这些是普通的 GPU——它们的内存不多（只有 24 GB）。预训练设置已相应缩小，以便于在 GPU 内存中运行，并在不到一周的时间内完成。

**一般预训练设置**    用于预训练的最终配置在[这里](https://github.com/wolfecameron/nanoMoE/blob/master/config/train_nano_moe.py)，具体设置如下：

- *模型架构*：六层（或块），每个自注意力层有六个注意力头，`d = 368`，`N = 8`（总专家数），`K = 2`（活跃专家数），`P = 2`（每隔一层使用 MoE 块）。
- *专家容量*：训练时容量因子为 1.25，评估时为 2.0。
- *辅助损失*：我们使用负载平衡辅助损失（缩放因子为 `0.01`）和路由器 z-loss（缩放因子为 `0.001`）。
- *精度*：训练使用自动混合精度（`bfloat16`），但路由器始终使用完整精度（`float32`）。
- *学习率*：采用标准的 LLM 学习率策略——训练开始时从 `6e-5` 线性升温到 `6e-4`，然后余弦衰减到 `6e-5`。
- *权重初始化*：我们使用 [6] 中提出的权重初始化方案以提高 MoE 训练的稳定性。

**预训练数据集**    类似于 nanoGPT，我们使用 [OpenWebText 数据集](https://huggingface.co/datasets/Skylion007/openwebtext) 进行 nanoMoE 的预训练。预训练过程缩减到大约 250 亿个 token——约为 nanoGPT 预训练 token 的 10%。这个较小的数据集允许在两块 3090 GPU 上大约 5 天内完成预训练。然而，我们可以通过更好的 GPU 设置（例如，8×A100 GPU）和将 `max_iters` 设置为 600,000（而不是 50,000）轻松扩展到完整的预训练运行。

**稳定性实验**    为了测试不同设置对 nanoMoE 训练稳定性的影响，我们进行了五个不同的实验。首先，我们在没有辅助损失或最佳实践的情况下预训练一个基线 nanoMoE 模型，导致负载不平衡和不稳定。然后，我们逐一启用几个改进，以观察它们对预训练稳定性的影响：

1. 辅助负载平衡损失。
2. 路由器 z-loss。
3. 路由器的全精度。
4. 改进的权重初始化方案。

这五个实验的结果显示在下图中。可以看到，每个预训练过程的改进都略微提升了训练稳定性——预训练的发散在训练过程中稍晚发生。当我们将所有改进一起启用时，模型实际上完成了整个训练过程而没有任何问题！我们可以清楚地看到，这些讨论的想法对 nanoMoE 的训练稳定性有切实的影响。

![|500](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F6a03cfe9-7023-4580-ac83-1ee1c19930f1_2012x922.png)

对于感兴趣的人，我鼓励你们自己尝试这些想法！只需调整训练配置，并使用下面的命令执行预训练过程。此命令假设你在一个节点上运行预训练，并且有一个或多个 GPU 可用。

```shell
torchrun --standalone --nproc_per_node=<GPU数量> train.py <配置文件路径；例如 config/train_nano_moe.py>
```

### 4、混合专家模型的深入学习

在这一概述中，我们深入了解了基于混合专家（MoE）的大型语言模型（LLM）是如何运作的。我们从标准的仅解码器的 Transformer 架构开始，修改为使用 MoE 架构。然后，通过在 OpenWebText 数据集上从头开始预训练一个中等规模的基于 MoE 的 LLM，称为 nanoMoE，应用了这些理念。尽管 MoE 被认为比标准 LLM 更难训练，但我们的实验表明，如何通过辅助损失、混合精度、更好的权重初始化等方法成功训练 MoE（即没有任何不稳定性）！

虽然 nanoMoE 是一个很好的学习工具，但大多数 MoE 的实际实现会比这复杂得多。要了解 MoE 在 LLM 研究中的实际应用，我们应该查看用于高效训练和推理的生产级 MoE 框架（例如，[OpenMoE](https://github.com/XueFuzhao/OpenMoE) 或 [Megablocks](https://github.com/databricks/megablocks)），以及关于 MoE 主题的最新出版物，例如 [Mixtral](https://arxiv.org/abs/2401.04088)、[DeepSeek-v3](https://arxiv.org/abs/2412.19437) 或 [DBRX](https://www.databricks.com/blog/introducing-dbrx-new-state-art-open-llm)。
