
> [Vision Large Language Models (vLLMs)](https://cameronrwolfe.substack.com/p/vision-llms)

正如我们将了解到的，尽管 vLLMs 功能强大，但它们与基于文本的 LLMs 并没有太大不同。

### 1、vLLMs 的构建模块

#### 1.1 交叉注意力（和 Transformers）

我们先来理解自注意力，然后再讨论交叉注意力。

![|500](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fc8fec4d1-3b72-4e17-8a01-2e7c4f3b7a5c_1949x930.png)

**自注意力概览**   自注意力机制的输入是一个标记向量序列。自注意力通过考虑序列中的所有其他标记，为每个标记形成一个输出表示。为此，自注意力操作对标记向量创建三个独立的线性投影——称为键、查询和值。然后，我们可以使用键和查询来计算序列中每对标记之间的注意力分数。这个注意力分数捕捉了每个标记对序列中其他标记的重要性——或者某个标记应该“关注”另一个标记的程度。我们可以将这些注意力分数乘以值来获得最终输出。下面提供了一个自注意力的基本实现。

```python
class SelfAttention(nn.Module):  
  
    def __init__(self, d: int):  
        super(SelfAttention, self).__init__()  
        self.d = d  
        self.c_attn = nn.Linear(self.d, 3*self.d, bias=False)  
  
    def forward(self, x):  
        q, k, v = self.c_attn(x).split(self.d, dim=2)  
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))  
        att = F.softmax(att, dim=-1)  
        y = att @ v  
        return y
```

**交叉注意力如何工作？**  下面提供了交叉注意力的示意图。可以看到，这个模块与自注意力并没有太大区别。关键区别在于用于计算键、查询和值矩阵的初始线性投影。与其通过线性投影单个标记向量序列来计算这三个矩阵，我们是对两个不同的向量序列进行线性投影。

![|500](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fc806734f-037f-4f6a-8863-b7383964d8ec_2098x1058.png)

查询矩阵是通过对第一个序列进行线性投影生成的，而键和值矩阵是通过对第二个序列进行线性投影生成的。因此，我们的注意力矩阵包含了第一个序列和第二个序列中所有标记之间的成对注意力得分。序列的长度不必相等，输出的长度将与第一个序列的长度相匹配。

```python
class CrossAttention(nn.Module):  
  
    def __init__(self, d: int):  
        super(CrossAttention, self).__init__()  
        self.d = d  
        self.w_q = nn.Linear(d, d, bias=False)  
        self.w_kv = nn.Linear(d, 2*d, bias=False)  
  
    def forward(self, x_1, x_2):  
        q = self.w_q(x_1)  
        k, v = self.w_kv(x_2).split(self.d, dim=2)  
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))  
        att = F.softmax(att, dim=-1)  
        y = att @ v  
        return y
```

![|500](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F528bb4b8-06a4-4e44-81a5-c49d7c285f42_2232x1316.png)

**在 vLLMs 中的应用**    我们对交叉注意力的解释在概述中可能显得有些随意。然而，正如我们将看到的，交叉注意力在多模态大型语言模型的研究中被频繁使用。我们可以利用交叉注意力，将视觉模型生成的图像表示融合到基于文本的 LLM 中。换句话说，我们可以在 LLM 生成输出时整合视觉信息，使模型能够除了文本之外，还能接收和解释图像（或其他类型的数据）作为输入。

#### 1.2 ViT

> “我们直接将标准 Transformer 应用于图像，尽可能少地进行修改。为此，我们将图像分割成若干小块，并将这些小块的线性嵌入序列作为 Transformer 的输入。” 

虽然 Transformer（及其许多变体如 BERT 和 GPT）最初是为自然语言处理应用提出的，但这种有影响力的模型架构已经扩展到计算机视觉领域的应用。视觉 Transformer（或简称 ViT）是目前最常用的架构。正如下图所示，这种架构与仅编码器（BERT 风格）的 Transformer 架构非常相似。我们只需将一系列向量作为输入，并应用包含双向自注意力和前馈变换的 Transformer 块序列。

![|500](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F36e2d7b7-26a1-4e68-939d-caaab3f133d9_1758x1200.png)

**处理输入图像**    视觉 Transformer 的输入是图像。然而，为了将图像作为输入传递给 Transformer，我们需要将图像转换为一个向量列表，这类似于文本标记向量序列。对于 ViT，我们通过将图像分割为一组小块，并将每个小块展平为一个向量来实现这一点。接下来，这些向量可能与 Transformer 期望的大小不同，因此我们需要对它们进行线性投影以得到正确的维度。

![|500](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F3b21b27b-e8a1-4542-b323-c3c17abbe379_1078x662.png)

类似于普通的 Transformer，我们为每个图像块的向量添加位置嵌入。这里，位置嵌入捕捉每个图像块在图像中的二维位置。这个 Transformer 架构的输出是每个图像块的向量序列，大小与输入相同。为了完成图像分类等任务，我们可以在模型末尾添加一个额外的分类模块（例如，一个线性层）。

**为什么使用编码器**    我们为 ViT 使用的是仅编码器的 Transformer 架构，而不是大多数 LLM 使用的仅解码器架构。原因在于 ViT 不是生成式的。对于 LLM，我们通过预测下一个标记来训练模型以生成文本序列。因此，我们需要在每个 Transformer 层中使用掩码自注意力，以防止模型在序列中向前看未来的标记，否则模型在预测下一个标记时可能会作弊！相比之下，ViT 需要查看整个图像块序列，以形成对图像的良好表示——我们不需要预测输入序列中的下一个图像块！

![|400](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F6013a7d8-f5e8-4b43-a15f-8690e0bbe93c_1516x412.png)

**训练 ViT**    原始的 ViT 模型与 BERT 架构相同。如上所示，训练了多种尺寸的 ViT，其中最大的是 ViT-H（或称 ViT-Huge）——我们将在后续概述中再次看到这个模型。所有的 ViT 模型都是通过在不同规模的数据集上进行有监督的图像分类训练的。当 ViT 在小型或中型数据集（例如 ImageNet）上训练时，其性能与同等大小的 ResNet 相当，或略逊一筹。然而，当在更大规模的数据集（例如 JFT-300M）上进行预训练并在下游任务上进行微调时，ViT 的优势开始显现。

![|500](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F2a07f5db-8f99-4901-a703-5f64d5dac7c1_2078x1142.png)


#### 1.3 CLIP

标准的 ViT 是在大型的有监督图像分类数据集上进行训练的。这些模型在使用大量人工标注的数据进行预训练时表现最佳，但获取这些数据既困难又昂贵。在 CLIP 中，作者探索了一种替代方法，使用更容易在线获取的图像-字幕对来训练一个强大的图像表示模型。这种方法被称为对比语言-图像预训练（CLIP）。

![|500](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F0e019441-7531-4d38-8a45-7e523dabebac_3284x1548.png)

**CLIP 架构**    CLIP 模型由两个独立的组件组成：一个图像编码器和一个文本编码器。给定一个图像-文本对作为输入，我们分别将这些输入传递给对应的编码器，以获得相关的向量表示。图像编码器是一个标准的 ViT 模型，而文本编码器是一个仅解码器的 Transformer（即典型的 GPT 风格的 LLM）。CLIP 的文本编码器并不用于生成文本，但作者使用仅解码器架构来简化未来将 CLIP 扩展到生成应用的过程。CLIP 架构的示意图如上所示。

> “简单的预训练任务是预测哪个字幕与哪个图像匹配，这是一种高效且可扩展的方法，可以从头开始在从互联网上收集的 4 亿（图像，文本）对的数据集上学习图像表示。” 

**对比学习**    我们可以通过多种方式来训练上述 CLIP 模型。例如，我们可以根据字幕中的词对图像进行分类，或使用架构中的 LLM 组件根据图像生成字幕。然而，先前的研究发现这些目标要么表现不佳，要么导致模型学习缓慢。CLIP 的关键贡献是提出了一种简单而高效的训练目标——基于对比学习的理念——从图像-文本对中学习。

![|500](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F9d7c264c-a19a-43f4-b867-353f68581cbe_864x430.png)
更具体地说，CLIP 通过一个简单的任务进行训练，即从一组候选字幕中为图像分类出正确的字幕（即，训练批次中的所有其他字幕）。实际上，这个目标通过以下方式实现：

1. 将一组图像和文本字幕通过各自的编码器（即，图像使用 ViT，文本使用 LLM）传递。
2. 最大化真实图像-字幕对的图像和文本嵌入之间的余弦相似度。
3. 最小化所有其他图像-字幕对之间的余弦相似度。

这个目标被称为多类 N 对（或 InfoNCE）损失，常用于对比学习和度量学习文献中。$$   \text{Loss} = -\frac{1}{N} \sum_{i=1}^{N} \log \frac{\exp(\text{sim}(\text{img}_i, \text{text}_i) / \tau)}{\sum_{j=1}^{N} \exp(\text{sim}(\text{img}_i, \text{text}_j) / \tau)}$$
![|500](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F9e4e6f45-c169-4d06-887e-9a23b5bf1a55_2380x1048.png)

**使用 CLIP**    尽管 CLIP 模型同时训练了图像和文本编码器，但在本概述中，我们主要使用 CLIP 的图像编码器。CLIP 的关键贡献不在于模型架构，而在于其训练目标。使用图像和文本编码器让我们能够利用上述的对比目标来训练图像编码器，这种方法非常高效，且不依赖大量的监督数据。CLIP 模型架构本身也很有用；例如，我们可以用它进行零样本图像分类。然而，我们也可以训练 CLIP 模型，仅仅为了获得高质量的图像编码器！

#### 1.4 从图像到视频

要用 LLM 处理图像，我们可以简单地将图像传递给图像编码器（例如 CLIP），生成一组向量——或嵌入——来表示图像。然后，LLM 可以将这些嵌入作为额外输入（稍后在概述中会详细介绍）。然而，*如果我们有视频而不是图像呢？* 有趣的是，用 LLM 处理视频输入与处理图像输入并没有太大区别——我们只需要一些策略将视频转换为一组向量，类似于处理图像！

**什么是视频**    在最简单的层面上，视频就是一系列有序的图像，通常称为“帧”。通常，图像以 RGB 格式存储。例如，下面图中的图像有三个颜色通道——红、蓝和绿——以及高度和宽度各为五。这个图像的大小是 `3（颜色通道）× 5（高度）× 5（宽度）`。我们还可以将多张图像堆叠成一个小批次，形成大小为 `batch × 3 × 5 × 5` 的张量。

![|500](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F5357ad54-32e7-44d4-9fcc-0236114a062c_1630x824.png)

视频的结构与图像没有太大区别——视频只是一个有序帧的集合。按正确的时间顺序查看这些帧时，它们揭示了场景随时间的运动，形成了视频。与图像类似，这些帧通常以 RGB 格式表示，视频中的所有帧具有相同的空间分辨率。例如，上图中的视频有三个帧，每个帧有三个颜色通道，高度和宽度均为五，形成一个大小为 `3（帧）× 3（颜色通道）× 5（高度）× 5（宽度）` 的张量。我们也可以创建视频的小批次，但必须确保每个视频具有相同数量的帧——这通常通过从视频中提取固定长度的“片段”（例如 64 帧）来实现。

![|500](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fb253f0e6-f5e7-475f-adee-b5b3dd94430a_1884x1122.png)

**帧率**    视频通常以固定的每秒帧数（FPS）录制。例如，24 FPS 是一个常见的帧率，这意味着每秒视频包含 24 帧。对于观看电影或玩电子游戏来说，帧率的精细度很重要——我们不希望视频帧之间有任何视觉上可察觉的间隙。然而，神经网络不需要以这种精细度处理视频。我们可以通过在视频中对帧进行子采样来节省计算成本；例如，采样 24 FPS 视频的每第八帧以模拟 3 FPS。

**编码视频**    一旦对视频帧进行了子采样，我们可以简单地将视频视为一组图像！通常，我们将每个视频帧独立传递通过图像编码器（如 CLIP），生成相应的向量集来表示每个视频帧。然后，LLM 可以将这些视频帧的向量作为额外输入，类似于处理图像。但这里仍然存在一个问题：*视频生成的向量数量庞大且有时不可预测，因为视频可以有任意长度*。我们需要一个额外的模块，将视频的帧表示聚合成一个固定大小的向量集！

![|600](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fc0a19ba8-3763-4bc1-baf5-026d74507645_2172x986.png)

这就是 **Perceiver** 和 **Perceiver Resampler** 派上用场的地方。Perceiver（如上图所示）是一种基于注意力的神经网络架构，可以接收高维输入——例如从视频帧生成的可变大小的大量向量——并基于这些输入输出固定大小的表示。简单来说，这意味着我们可以将所有视频向量传递给 Perceiver，它会返回一个固定大小的向量集。然后，我们可以轻松地将这个额外输入与 LLM 集成，就像处理图像一样！

![|500](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F4277d6a0-7606-4ad8-a8f2-e70308ddd1de_1792x926.png)

Perceiver 最初由 Flamingo 应用于多模态 LLMs，Flamingo 提出了 Perceiver Resampler。Flamingo 以每秒 1 帧的速度对视频进行采样（即每秒视频取一个帧）。视频的每个子采样帧独立通过图像编码器处理，生成相应的图像嵌入。然而，在将这些图像嵌入传递给基于文本的 LLM 之前，我们通过 Perceiver 架构处理它们，为视频生成固定数量（64个）的视觉标记向量。然后，*我们使用之前描述的交叉注意力将这些向量整合到 LLM 中*。

### 2、vLLM 架构和训练策略

我们现在已经了解了与 vLLMs 相关的大部分背景概念。接下来，我们将利用这些概念从头开始构建对 vLLMs 的理解。在本节中，我们将重点讨论常用于创建 vLLMs 的架构和训练策略。我们暂时保持讨论的概念性，然后在下一节中将这些想法应用于实现一个真实的 vLLM。

#### 2.1 vLLM 架构变体

vLLM 的架构始终有两个主要组件：LLM 主干和视觉编码器。LLM 主干只是一个标准的仅解码器的 Transformer，而视觉编码器通常是一个 CLIP / ViT 模型（如果我们想处理基于视频的输入，可以选择使用 Perceiver Resampler）。有两种常见的 vLLM 架构变体将这些组件融合在一起：*统一嵌入架构和跨模态注意力架构*。我们使用 [Sebastian Raschka](https://sebastianraschka.com/) 在他关于 vLLMs 的[精彩综述](https://magazine.sebastianraschka.com/p/understanding-multimodal-llms)中提出的命名方案。现在，让我们了解这些架构如何工作。

![|500](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F42d39309-0469-4908-9b37-5204415f85c1_1648x842.png)

**标记向量**    LLM 主干以原始文本为输入，但首先将文本分词成一组离散的标记，并通过从嵌入层检索每个标记的相应嵌入，将其转换为标记向量；见上文。同样，对于图像（或视频），我们通过视觉编码器处理图像或视频，生成一组标记向量，视觉编码器返回一组视觉标记向量作为输出。

![|400](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fcd022205-f5bc-4580-a4b3-3a03648d37d1_1288x1066.png)

**统一嵌入**    现在，我们有一组文本和图像（或视频）标记向量作为输入。第一种常见的 vLLM 架构简单地：

1. 将这两种模态的向量连接在一起，形成一个标记向量的单一序列。
2. 将这些连接后的向量直接作为输入传递给仅解码器的 Transformer 架构。

这种架构被称为统一嵌入架构，如下图所示。值得注意的是，视觉标记向量的大小可能与文本标记向量不匹配。因此，我们通常在连接之前，将视觉编码器的标记向量线性投影到正确的维度。

![|500](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F0f88da96-bb2a-49c7-a3db-171fa92bb2fa_1498x1158.png)

统一嵌入架构在概念上很简单，但它增加了传递给 LLM 的输入长度，这可能导致训练和推理时计算成本的大幅增加。*这些视觉标记会经过我们强大的 LLM 主干的每一层*！幸运的是，我们可以通过使用一种稍微不同的 vLLM 架构来解决这个问题。

**跨模态注意力**    我们不再将文本和视觉标记向量连接在一起，而是仅将文本标记向量作为输入传递给 LLM。为了整合视觉信息，我们可以在 LLM 的选定层中添加额外的跨注意力模块，这些模块在文本和视觉标记向量之间执行跨注意力——通常是每隔两层或四层。这种架构变体通常被称为跨模态注意力架构；见下图。值得注意的是，这种架构看起来与原始 Transformer 解码器非常相似——*我们只是用图像编码器进行跨注意力，而不是用 Transformer 编码器* ！

![|500](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fc46c381b-da70-4f32-9067-570ca1fbb56b_1660x1076.png)

这种架构的优势在于，我们不需要增加传递给大型语言模型（LLM）的输入长度，而是通过使用跨注意力机制将视觉信息合并到 LLM 中，这样计算效率更高。此外，跨模态注意力架构在模型中添加了新层以融合视觉和文本信息，而不是依赖 LLM 现有的层来完成这种融合。因此，我们可以在训练时保持 LLM 主干不变，只训练新增的层，从而确保 LLM 在仅文本任务上的性能完全不变。

#### 2.2 如何训练 vLLMs？

在这个概述中，我们只考虑能够接收视觉输入的 LLMs——*这些模型仍然只生成文本输出*。因此，我们可以使用类似于其他 LLM 的方式来训练这些模型：使用下一个词预测。即使是统一嵌入架构，我们主要通过预测文本标记来训练模型——通常不尝试预测视觉标记（即执行下一个图像预测）。

> “Gemini 模型的视觉编码受到我们在 Flamingo、CoCa 和 PaLI 上的基础性工作的启发，重要的区别在于这些模型从一开始就是多模态的，并且可以使用离散图像标记原生输出图像。” 

除了训练目标之外，还有几种策略可以用于训练 vLLM。例如，我们可以进行原生多模态训练，这意味着从头开始初始化架构的所有组件，并从一开始就使用多模态数据（即文本、图像、视频等）训练模型；例如，这种方法用于训练 Gemini。

然而，实际上，原生多模态性复杂且困难。使用这种方法可能会遇到许多问题：

- 获取大量配对的图像和文本数据很困难。
- 在预训练规模上对视觉数据进行高效的标记化很困难。
- 模态之间可能出现不平衡；例如，模型可能会学会忽略图像，因为文本通常提供足够的信息用于下一个词预测。

因此，vLLMs 更常使用 *组合方法* 进行训练。具体来说，这意味着我们首先独立预训练 LLM 主干和视觉编码器。然后，我们进入一个额外的训练阶段——称为融合阶段——将文本和视觉模型结合成一个 vLLM。这种方法有几个优点：

- 文本和图像模型的开发可以并行化。
- 可以使用现有的、非常强大和先进的文本 LLM 作为训练 vLLMs 的起点。
- 可以使用更多的数据量，因为我们可以利用仅文本、仅视觉以及配对的文本和视觉数据进行训练。

在融合阶段，我们可以选择是否训练完整的 vLLM 架构。例如，当使用跨模态注意力架构时，我们可以在融合过程中冻结 LLM 主干，只训练跨注意力和视觉编码器层。这种方法在文献中很常见，因为它允许我们从现有的基于文本的 LLM 开始创建相应的 vLLM，而无需对底层 LLM 主干进行任何修改。正如我们将看到的，这正是用于训练 LLaMA-3.2 视觉模型的方法。

### 3、LLaMA-3.2 Vision: 强大的开源 vLLMs

现在我们了解了 vLLMs 的基本概念，来看一个实际案例研究。LLaMA-3 LLMs 最初仅限于文本输入，但后来扩展到处理图像（和视频）输入。这些模型大多是开源的，因此我们可以通过研究其技术报告和代码深入理解它们。在本节中，我们将详细研究 LLaMA-3 套件的 LLMs 如何扩展以创建相应的 vLLMs。

#### 3.1 扩展 LLaMA-3 到图像和视频

LLaMA-3 是最流行和强大的开源 LLMs 套件之一。LLaMA-3 模型都是密集型的——这意味着它们不使用 MoE 架构——并且有三种不同的规模：8B、70B 和 405B。这些模型在先前的 LLaMA-2 模型基础上有数量级的改进——它们的上下文窗口大了 30 倍（128k vs. 4k），数据集大了 30 倍（15.6T tokens vs. 1.8T tokens），计算量增加了 50 倍。

> “我们发现 Llama 3 在许多任务上提供了与领先语言模型如 GPT-4 相当的质量……论文还展示了通过组合方法将图像、视频和语音功能集成到 Llama 3 中的实验结果。”

初始的 LLaMA-3 模型仅接受文本输入。然而，作者包含了结合视觉（即图像和视频）和语音特征的实验。我们将在本节中学习 LLaMA-3 如何在视觉输入上进行训练。

**组合 vLLMs**    LLaMA-3 采用组合方法创建多模态模型。我们首先独立预训练视觉编码器和仅文本的 LLM。其中，仅文本的 LLM 是基于文本的 LLaMA-3 模型，而视觉编码器是预训练的 CLIP 模型。采用跨模态注意力架构，我们在这两个模型之间插入跨注意力层，并专注于训练这些额外的层。为了方便，我们将这些跨注意力层称为“图像适配器”。通过这样做，LLM 学会在生成输出时整合额外的视觉特征。

LLaMA-3 的视觉编码器基于 ViT 架构——特别是 630M 参数的 ViT-H 模型 —— 通过对 25 亿图文对的对比目标进行预训练。换句话说，这个模型几乎与 CLIP 架构的图像编码器组件相同。我们通过将图像输入模型并提取相应的嵌入来创建视觉特征。

![|400](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F54ca3ac2-9461-4c6b-b22c-ec7535f726cc_1178x1178.png)

值得注意的是，根据先前研究，使用对比（CLIP 风格）目标训练的图像编码器能够捕捉语义信息，但无法捕捉图像的细粒度感知细节。因此，任何依赖此类视觉特征的 LLM 可能无法回答需要精确定位图像中细节的问题；参见下方 [GPT-4V](https://cdn.openai.com/papers/GPTV_System_Card.pdf) 的示例。正如上文所示，LLaMA-3 通过从视觉编码器的多个不同层提取视觉特征并将它们连接在一起来解决此问题。

![|400](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F266ca8c9-2059-43b2-aa4e-ff22f37e7608_1072x1148.png)

LLaMA-3 还在图像编码器之后、与 LLM 融合之前增加了几个自注意力层——最终的图像编码器总共有 850M 参数。这个编码器为输入图像中的每个块生成 7680 维的嵌入，每个图像共有 `16 × 16 = 256` 个块。

> “我们在图像编码器生成的视觉标记表示和语言模型生成的标记表示之间引入跨注意力层。” 

**图像适配器**    为了将图像编码器的特征整合到 LLaMA-3 中，我们使用了基于跨注意力的图像适配器。具体来说，在 LLM 的每第四个 transformer 块中添加计算 LLM 文本标记和图像编码器图像嵌入之间注意力的跨注意力层。这些跨注意力层显著增加了模型的大小；例如，带有图像适配器的 LLaMA-3-405B 模型有约 500B 参数。然而，图像适配器允许 LLM 在生成文本时将图像编码器的信息整合到其标记表示中。

**视频适配器**    除了图像，作者还扩展了 LLaMA-3 以支持视频输入。由于视频只是图像（或帧）的序列，我们不需要显著修改现有架构。模型以 64 帧为输入，每帧都通过现有的图像编码器。为了捕捉帧之间的时间关系，我们使用 Perceiver Resampler，将 32 连续帧的表示聚合为一个。最后，将额外的视频跨注意力层添加到 LLM 中。

![|500](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fe96d2d1d-97f2-4d42-a94a-08219ec4deee_1604x998.png)

多模态 LLaMA-3 的完整架构，包括视频和图像组件，如上所示。我们可以看到，图像和视频输入首先通过图像编码器处理，然后通过跨注意力层整合到 LLM 中。对于视频，我们添加了一个额外的聚合模块——Perceiver Resampler——以捕捉视频帧之间的顺序关系。

**预训练数据集**    图像编码器和跨注意力层在大型图文对数据集上进行训练。该数据集经过过滤以：_i)_ 删除非英语字幕，_ii)_ 删除重复项，_iii)_ 删除低质量数据，_iv)_ 最大化多样性（基于 n-gram TF-IDF 分数）。收集视频文本对以训练视频适配器的过程非常相似。

为了提高 LLaMA-3 的文档理解能力，作者还将 OCR 输出连接到每个文本字幕的末尾，并收集大量以图像表示的文档及其相关文本。LLaMA-3 的其他重要多模态训练数据来源包括：

- *视觉指向*：文本中的名词短语与图像中的边界框/掩码链接，这些边界框/掩码要么在图像中叠加，要么通过文本中的（归一化）坐标指定。
- *截图解析*：从 HTML 代码生成的截图，模型需要预测截图中由叠加边界框指示的元素生成的代码。
- *问答对*：来自多个来源的大量问答数据。
- *合成字幕*：由 LLaMA-3 早期版本生成的合成字幕的图像。作者观察到，合成字幕往往比原始人工撰写的字幕更全面。
- *合成结构化图像*：图表、表格、流程图、数学方程等，并附有结构化表示（例如，markdown 或 LaTeX）。

**图像适配器训练**    在训练图像适配器之前，图像编码器会在数据集中描述的图文对上进行若干个周期的预训练。在训练适配器时，图像编码器的权重会继续更新，而大型语言模型（LLM）的权重则被冻结。因此，多模态 LLaMA-3 模型的 LLM 主干与仅文本的 LLaMA-3 相同，确保在仅文本任务上的一致性。

图像适配器的训练分为两个阶段，两个阶段都使用标准的语言建模目标应用于文本标题。在第一阶段，所有图像都被调整为较低分辨率，以提高训练效率。初始训练阶段之后是一个较短的第二阶段，在这个阶段中，我们提高图像的分辨率，并使用原始数据集的一个较小（采样）版本，重点是最高质量的数据。在完成两个训练阶段后，我们开始训练视频适配器——从完全训练好的图像编码器和适配器开始——在视频文本数据集上使用类似的过程。

> _“在预训练后，我们在高度精心策划的多模态对话数据上微调模型，以实现聊天功能。我们进一步实施直接偏好优化（DPO）以提升人工评估表现，并通过拒绝采样来提高多模态推理能力。”_

**后训练**    类似于文本基础的 LLaMA-3 模型，多模态模型经过完整的后训练程序，以使模型与人类偏好对齐，教会它如何遵循指令，提高其处理对话输入的能力等。与仅文本的 LLaMA-3 模型类似，多模态模型通过监督微调（SFT）、拒绝采样（RS）和直接偏好优化（DPO）的组合进行多次顺序应用（即“轮次”）。这个过程如下图所示，LLaMA-3 的后训练完整概述可以在[这里](https://www.interconnects.ai/p/frontier-model-post-training)找到。

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fefcd982d-ef70-4c48-be86-79ae58b6496b_1276x614.png)

与训练图像编码器和适配器时不同，在后训练过程中，我们不使用基础 LLaMA-3 模型的权重。相反，我们用已经经过广泛后训练的 LLaMA-3-Instruct 模型的权重替换基础模型的权重。后训练的数据集来自多种来源：

- 通过模板或使用 LLM 重写的学术数据集，以对话格式呈现。
- 人工标注的数据集，通过以下方式收集：  
  _i)_ 提供一个种子图像或视频，让人类撰写相关对话；  
  _ii)_ 要求人类比较模型输出以形成偏好对。
- 合成数据集，通过将图像或视频的文本表示（即标题）提供给 LLM，提示模型生成相关的问答对。
- 现有模型输出被 LLM 细微（但有意义）地扰动以产生错误，从而形成偏好对。

采用了多种独特策略来优化后训练模型的性能。例如，作者在后训练的每个阶段训练多个模型（使用不同的超参数），通过取这些模型权重的平均值获得最终模型。这种模型合并方法的表现优于通过超参数网格搜索获得的最佳模型。

#### 3.2 LLaMA-3.2: 中等大小的视觉 LLMs

初步实验中提到的多模态 LLaMA-3 模型已提供，但这些模型直到 LLaMA-3.2 才正式发布。LLaMA-3.2 Vision 模型的 11B 和 90B 参数版本是首批支持图像输入的 LLaMA 模型，在图像字幕和文档理解等视觉理解任务上表现强劲。[1] 中探索的其他模态（如语音和视频）未包含在 LLaMA-3.2 中，且最大的 405B 参数 LLaMA-3.1 模型没有发布多模态版本。

**LLaMA-3.2 Vision 架构。** [2] 中描述的 LLaMA-3.2 Vision 模型架构与 [1] 中的初步模型完全匹配。这些模型由以下部分组成：

- 预训练的 LLM 主干。
- 预训练的视觉编码器。
- LLM 和视觉编码器之间的多个交叉注意力层。

LLaMA-3.2 的 LLM 主干是仅文本的 LLaMA-3.1-8B 和 LLaMA-3.1-70B 模型。视觉 LLMs 在图文对上分多个阶段进行训练，但训练过程中不更新 LLM 主干——我们只更新图像编码器和适配器层。因此，LLaMA-3.2 Vision 模型在仅文本任务上的性能相对于 LLaMA-3.1 保持不变。

**训练阶段。** 如前所述，LLaMA-3.2 Vision 模型分多个阶段训练。首先，必须独立预训练 LLM 主干和图像编码器。然后，通过在两者之间添加交叉注意力层来整合这些模型，并在大型（且噪声较多）的图文对数据集上预训练组合的视觉模型。最后，在中等规模的高质量增强数据集上进一步训练模型并进行后训练。视觉模型的后训练策略包括多轮 SFT、拒绝采样和 DPO（与 LLaMA-3.1 相同）。

![|500](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F05814b16-71ad-4958-8622-d1e662c48939_1452x1116.png)

**模型评估**   对于基于文本的任务，LLaMA-3.2 模型的性能与 LLaMA-3.1 相同——多模态预训练过程未改变 LLM 主干。然而，作者在 [2] 中对 LLaMA-3.2 Vision 模型在各种视觉理解任务上进行了评估。最显著的是，这些模型在涉及文档、图表或图解的任务上表现强劲。这种能力并不令人惊讶，因为模型在大量文档-文本对以及合成图表和表格图像上进行了训练。在其他视觉理解任务上，LLaMA-3.2 也表现良好，与一些领先的基础模型竞争力相当。

#### 3.3 LLaMA-3.2 Vision 实现

现在我们了解了 LLaMA-3.2 Vision 模型，让我们深入研究它们的实现。为此，我们将在 [torchtune](https://github.com/pytorch/torchtune) 中研究其代码。为简单起见，我们将省略实现中的一些细节，而是呈现概述关键建模组件的伪代码。不过，有兴趣的人可以随时阅读 torchtune 中的 [完整代码](https://github.com/pytorch/torchtune/tree/main/torchtune/models/llama3_2_vision)！

![|500](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F4e163a5c-aa41-48ce-a6b8-1fea597ef0a0_1476x1128.png)

**顶层结构**    如果查看用于实例化 LLaMA-3.2 Vision 架构的主要函数，会发现模型由两个主要组件组成：图像编码器和 LLM 主干（在上文中称为视觉解码器）。这两个模型在一个 `FusionModel` 中组合。如上所示，我们可以切换这个 `FusionModel` 的可训练组件，该模型处理每个组件是否可训练，并以通用方式将视觉编码器的输出传递给视觉解码器。

```python
# compute the output of the vision encoder
encoder_embed = None
if encoder_input is not None:
    encoder_embed = self.encoder(**encoder_input)

# pass the vision encoder output to the vision decoder
output = self.decoder(
   tokens=tokens,
   mask=mask,
   encoder_input=encoder_embed,
   encoder_mask=encoder_mask,
   input_pos=input_pos,
)
```

显著的是，`FusionModel` 的输入输出结构与 PyTorch 中的标准 Transformer 解码器相同——这两种类型的模型可以互换使用。如上面的代码所示，我们还可以提供一个编码器掩码，使我们能够从选定的文本标记中屏蔽任何图像标记。

> “DeepFusion 是一种融合模型架构，其中预训练的编码器与预训练的解码器（LLM）结合……该模块对编码器和解码器如何融合没有假设；它只是将编码器嵌入传递给解码器，让解码器处理任何融合。” 

LLaMA-3.2 Vision 使用的视觉编码器是标准的基于 CLIP 的视觉编码器。这个编码器通过 CLIP 处理输入图像以获取一组图像嵌入。从这里开始，我们不会直接将 CLIP 的输出传递给视觉解码器——在 CLIP 和视觉解码器之间还有一个额外的 `VisionProjectionHead` 模块。实现如下所示。

![|500](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fe9aed6e0-ffe0-490f-b768-e30454fb5ea6_1542x3028.png)

这个模块在 CLIP 嵌入被视觉解码器处理之前，通过几个额外的自注意力层。此外，投影头从 CLIP 模型的多个隐藏层提取特征——而不仅仅是使用最后一层的输出——以确保感知信息不会丢失。所有这些嵌入被连接在一起，并进行线性投影，以匹配视觉解码器使用的文本标记向量的大小。

![|500](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Ff3d9f263-68b8-4df6-9fad-113744c8755b_1664x2766.png)

LLaMA-3.2 Vision 的视觉解码器几乎与标准的基于文本的 LLM 相同。我们只是对这个架构进行修改，在解码器的某些层中添加交叉注意力层。为此，使用了一个 `FusionLayer`，它将交叉注意力层和解码块的参数分开。这样，我们可以选择是否训练这些组件。例如，LLaMA-3.2 在多模态训练过程中训练交叉注意力层，而保持 LLM 主干不变。

### 结论

从这个概述中，我们应该了解到 vLLMs 与标准的基于文本的 LLM 并没有太大区别。我们只是为这个模型添加了一个额外的图像编码器，以及一些额外的层来融合这两个模型。图像编码器和基于文本的 LLM 之间的融合可以通过统一的嵌入架构或跨模态注意力来实现。接下来，我们可以通过图像-文本对（分多个阶段）训练这个组合模型，形成一个强大的 vLLM。虽然 vLLMs 存在许多变体，但其背后的基本思想确实很简单！