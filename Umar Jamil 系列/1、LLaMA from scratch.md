
> https://www.bilibili.com/video/BV1xe4peAESX/?spm_id_from=333.1387.collection.video_card.click&vd_source=aced32e35ad9cff83fe98c60854f183c

在本篇文章中，我们将了解什么是 LLaMA，它在结构上与 Transformer 有何不同，并逐一构建其各个模块，不仅从概念上解释每个模块的功能，还将从数学角度和编程角度进行探讨，以便将理论与实践结合起来。

文章内容主要包括：

* Vanilla Transformer 和 LLaMA 模型之间的架构差异
* RMS 归一化
* 旋转位置编码
* KV 缓存
* 多查询注意力
* 分组多查询注意力
* 前馈层的 SwiGLU 激活函数。


### 0、先决条件

* Transformer 的结构、注意力机制工作原理
* Transformer 的训练与推理
* 线性代数：矩阵乘法、点乘
* 复数：欧拉公式，$e^{ix}=\cos x+ i\sin x$


### 1、Transformer vs LLaMA

* 首先，在 LLaMA 中只有解码器；
* 其次，在嵌入层之后，没有位置编码，而是 RMS 归一化，实际上，所有的归一化都移到了块之前；
* 位置编码不再是 Transformer 的位置编码，而是旋转位置编码，并且他们只应用于 $Q$ 和 $K，$不包含 $V$；
* 自注意力机制带有 $KV$ 缓存；
* 自注意机制为分组多查询注意力；
* 前馈层中激活函数由 ReLU 变为 SwiGLU 函数


我们将从底层开始构建每一个子模块，并详细展示这些块具体做什么，它们如何工作，如何相互作用，背后的数学原理是什么，以及它们试图解决的问题是什么。

### 2、LLaMA 简介

LLaMA 于 2023 年 2 月发布，他们为这个模型设定了四个维度：

![[Pasted image 20250318203939.png|550]]

在原始的 Transformer 中，$\text{dimension} = 512, n\text{ head}=8, n\text{ layers}=6$，在 LLaMA 2 中，大多数参数都翻倍了，
![[Pasted image 20250318204234.png|550]]

上下文长度（即，序列长度），是指模型能处理的最长序列，模型训练所用的 token 数量也翻倍了，从 1T 到 2T，每个模型的大小都有所增加，而参数量大致保持不变，最后两个模型使用了 GQA 技术表示的模型，稍后会介绍其工作原理。

### 3、嵌入层

对句子进行分词，将其转换为 token，分词通常是通过 BPE 分词器完成的，在 LLaMA 中，token 向量大小为 4096，这些嵌入向量是可学习的，因此它们是模型的参数，在模型训练过程中，这些嵌入向量会发生变化，以便捕捉它们所映射单词的含义。

### 4、归一化层

这是紧接着嵌入层之后的层，我们将从数学层面分析理解归一化是如何工作的，假设我们有一个线性层，信息的流动控制公式为：$$O=XW^T+b$$一个神经元对某个数据项的输出，取决于输入数据项的特征和神经元的参数，我们可以将 $X$ 视为前一层的输出，如果前一层由于梯度下降更新了其权重，导致输出 $X$ 发生巨大变化，将产生与以往大不相同的输出，下一层也将因此大幅改变其输出，在梯度下降的下一步，它将被迫大幅调整其权重，这种导致神经元内部节点分布发生变化的现象被称为*内部协变量偏移（Internal Covariate Shift）*，我们希望避免这种现象，它会使网络训练变慢，因为神经元被迫因前一层输出的剧烈变化而在一个方向或另一个方向上大幅调整其权重。

#### 4.1 Layer Norm
层正则化公式如下：$$y = \frac{x - \mathbb{E}[x]}{\sqrt{\text{Var}[x]} + \epsilon} \times \gamma + \beta $$
- 每个 item 用其标准化值更新，这将使其变为均值为 0、方差为 1 的正态分布。
- 两个参数 $\gamma$ 和 $\beta$ 是可学习参数，它们允许模型根据损失函数的需要“放大”每个特征的尺度或对特征进行平移。

#### 4.2 均方根 RMS Norm
![[Pasted image 20250320211350.png|600]]
大致翻译：LayerNorm 成功的一个著名解释是其 *重新中心化* 和 *重新缩放的不变性* 特性，什么意思呢？无论特征是什么，它们都将被重新中心化到均值为 0，并重新缩放到方差为 1，前者使模型对输入和权重的偏移噪声不敏感，而后者在输入和权重随机缩放时保持输出表示不变，*在本文中，我们假设 LayerNorm 成功的原因是重新缩放不变性，而非重新中心化不变性*

他们所做的基本上是说，能否找到另一个不依赖于均值的统计量，因为他们认为重新中心化不是必要的，所以他们使用均方根统计量，定义如下：$$
\bar{a}_i = \frac{a_i}{\text{RMS}(a)} g_i, \quad \text{where} \quad \text{RMS}(a) = \sqrt{\frac{1}{n} \sum_{i=1}^{n} a_i^2}.$$其中，$g \in \mathbb{R}^n$ 是用于重新缩放标准化后的输入总和的增益参数，初始值设为 1。可以看到，这里不再使用均值信息，而在之前计算方差的过程中需要均值信息：$$
\mu = \frac{1}{n} \sum_{i=1}^{n} a_i, \quad \sigma = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (a_i - \mu)^2}. $$
**那为什么要选择使用 RMSNorm？** 与层归一化相比，计算量更少，只需要计算一个均方根统计量，不需要计算均值和标准差，这带来了计算上的优势，而且在实践中效果很好，所以实际上，*论文作者的假设是正确的*，不需要重新中心化，至少在 LLaMA 中是这样的。

### 5、旋转位置编码

绝对位置编码和相对位置编码之间的区别：

- 绝对位置编码是固定向量，添加到标记的嵌入中以表示其在句子中的绝对位置。因此，它*一次处理一个标记*。
- 相对位置编码则一次处理两个标记，并在计算注意力时涉及其中。由于注意力机制捕捉两个词之间相关程度的“强度”，相对位置编码告诉注意力机制这两个词之间的*距离*。因此，给定*两个*标记，我们创建一个表示其距离的向量。

![[Pasted image 20250320214214.png|400]]

相对位置编码首次在这篇来自谷歌的论文中被引入，Vaswani 是 Transformer 模型的作一。

在原始的注意力机制中，计算两个 token 之间的点积，其位置信息已经包含在各自的 token 中，因此可在两个向量中直接计算：$$
e_{ij} = \frac{(x_i W^Q)(x_j W^K)^T}{\sqrt{d_z}}$$在相对位置编码中，则有三个向量，其中向量 $a_{ij}$ 表示这两个 token 之间的距离：$$ 
e_{ij} = \frac{x_i W^Q (x_j W^K + a_{ij}^K)^T}{\sqrt{d_z}}$$
![[Pasted image 20250320223001.png|500]]

旋转位置编码在这篇论文（RoFormer）中被引入，论文的作者想要做的是，能否找到一种内积，只依赖于注意力机制中使用的两个向量（Q 和 K）本身以及它们所代表的 token 的相对距离，也就是说，给定 Q 和 K，它们只包含它们所代表的词的嵌入以及它们在句子中的位置，$$q_m = f_q(x_q, m), \quad k_n = f_k(x_k, n)$$这里的 $m, n$ 都是它们在句子中的绝对位置，我们能否找到一个内积，使得：$$\mathbf{q}_m^\top \mathbf{k}_n = \langle f_q(x_m, m), f_k(x_n, n) \rangle = g(x_m, x_n, n - m),$$
是的，我们可以找到这样一个函数，这个函数定义在复数空间中，并且可以通过使用欧拉公式转换，我们可以定义一个函数 $g$，它仅依赖于两个嵌入向量 $\mathbf{q}$ 和 $\mathbf{k}$ 及其相对距离：$$\begin{align}
  &f_q(x_m, m) = (W_q x_m) e^{im\theta} \quad 
  f_k(x_n, n) = (W_k x_n) e^{in\theta} \\[1.2ex]
  &g(x_m, x_n, m-n) = \Re\left[(W_q x_m) (W_k x_n)^* e^{i(m-n)\theta}\right] \end{align}$$使用欧拉公式，我们可以将其写成矩阵形式：$$
  f_{\{q,k\}}(x_m, m) = 
  \begin{pmatrix}
  \cos m\theta & -\sin m\theta \\
  \sin m\theta & \cos m\theta
  \end{pmatrix}
  \begin{pmatrix}
  W^{(11)}_{\{q,k\}} & W^{(12)}_{\{q,k\}} \\
  W^{(21)}_{\{q,k\}} & W^{(22)}_{\{q,k\}}
  \end{pmatrix}
  \begin{pmatrix}
  x_m^{(1)} \\
  x_m^{(2)}
  \end{pmatrix}$$等号右侧第一项表示在二维空间中的旋转矩阵，因此称为旋转位置嵌入。

![[Pasted image 20250320224246.png|200]]
示例：旋转矩阵为$$
\mathbf{R}_\theta = 
\begin{bmatrix}
\cos \theta & -\sin \theta \\
\sin \theta & \cos \theta
\end{bmatrix},$$我们有：$$
\mathbf{v}' = \mathbf{R}_\theta \mathbf{v}_0.$$
考虑到嵌入维度并不是 2 维，在 LLaMA 中为 4096 维度，则采用如下稀疏矩阵 
![[Pasted image 20250320224607.png|600]]
但由于其稀疏性，这会使 GPU 计算效率很低，为此需要转换一个格式：

![[Pasted image 20250321202640.png]]
与绝对位置编码一样，只计算一次，可以对我们将要训练模型的所有句子重复使用；

另一个有趣特性是*长期衰减*，作者通过改变两个标记之间的距离计算了内积的上限，并证明了随着相对距离的增加，该内积会衰减。这意味着，使用旋转位置嵌入编码的两个标记之间的“关系强度”会随着它们之间距离的增加而数值变小，这是我们希望从旋转位置嵌入中得到的理想特性。

**旋转位置嵌入仅应用于 $Q$ 和 $K$，不包含 $V$，为什么？**  因为这仅仅用于计算注意力分数，在注意力机制中，旋转位置嵌入是在向量 $\mathbf{q}$ 和 $\mathbf{k}$ 被 $W$ 矩阵相乘之后应用的，而在普通的 Transformer 中，它们是在相乘之前应用的。

---------

现在到了有趣的部分，我们将看看 llama 中的自注意力是如何工作的，但在我们讨论 llama 中使用的自注意力之前，我们需要至少简要回顾一下原始 Transformer 中的自注意力，我们从矩阵 Q 开始，这是一个序列乘以模型的矩阵，这意味着我们在行上开始，这是一个序列乘以模型的矩阵，这意味着我们在行上有标记，在列上有嵌入向量的维度，所以我们可以这样想，

我们可以想象它有六行，每一行都是 512 维的向量，表示该标记的嵌入，现在让我删除，然后我们根据这个公式进行乘法运算，所以 Q 乘以 K 的转置，即 K 的转置除以 512 的平方根，这是嵌入向量的维度，其中 K 等于 Q、V 也等于 Q，因为这是自注意力，所以这三个矩阵实际上是相同的序列，
然后我们应用 softmax，得到这个矩阵，所以我们有一个 6 乘 512 的矩阵乘以另一个 512 乘 6 的矩阵，我们将得到一个 6 乘 6 的矩阵，其中每个元素代表第一个标记与自身的点积，然后是第一个标记与第二个标记的点积，第一个标记与第三个标记的点积，第一个标记与第四个标记的点积，等等，所以这个矩阵捕捉了两个标记之间的关系强度，这就是这个 softmax 的输出，乘以 V 矩阵以获得注意力序列，所以自注意力的输出是另一个与初始矩阵维度相同的矩阵，因此，它将生成一个序列，其中嵌入不仅捕捉每个标记的含义，不仅捕捉每个标记的位置，还捕捉该标记与所有其他标记之间的关系，如果你不理解这个概念，请去看我之前关于 Transformer 的视频，我在那里非常仔细且详细的解释了它

现在让我们来看看多头注意力，非常简单的说，多头注意力基本上意味着我们有一个输入序列，我们将其复制到 q、k 和 v 中，所以它们是相同的矩阵，我们乘以参数矩阵，然后将其分割成多个较小的矩阵，每个头一个，并计算这些头之间的注意力，所以头一，头二，头三，头四，然后我们将这些头的输出连接起来，乘以输出矩阵 $W^O$，最后得到多头注意力的输出，

让我们看看什么是第一个 KV 缓存，因此，在介绍 KV 缓存之前，我们需要了解 llama 是如何训练的，以及什么是下一个标记预测任务，因此，llama 与大多数 llms 一样，已经在下一个标记预测任务上进行了训练，这意味着，给定一个序列，它将尝试预测下一个标记，即最有可能继续提示的一个标记，因此，例如，如果我们告诉他一首诗，例如，没有最后一个词，它很可能会想出那首诗中缺失的最后一个词，在这种情况下，我将使用但丁·阿利吉耶里的一段非常著名的段落，我将不使用意大利语翻译器，而是使用这里的英语翻译器，所以我只处理第一行，你可以在这里看到：爱能迅速俘获温柔的心，所以让我们在这个句子中训练 llama，训练是如何进行的，好吧，我们将输入提供给模块输入是这样构想的，我们首先添加句子开始标记，然后目标是这样构建的，我们附加一个句子结束标记，为什么，因为模型，这个 transformer 模型，是一个序列到序列模型，它将输入序列中的每个位置映射到输出序列中的另一个位置，所以基本上，输入序列的第一个标记将映射到输出序列的第二个标记，等等，这也意味着，如果我们给模型输入 sos，它将输出第一个标记，所以是爱，然后如果我们给出前两个标记，它将输出第二个标记，所以是爱能，如果我们给出前三个标记，它将输出第三个标记，当然，模型也会为前两个标记生成输出，但让我们通过一个例子来看，所以如果你还记得我之前的视频，我在其中进行推理，当我们训练模型时，我们只做一步，所以我们给出输入，给出目标，计算损失，我们没有 for 循环来训练一个单一的句子模型，但对于推理，我们需要逐个标记的进行，所以在这种推理中，我们从时间戳 1 开始，我们只给出输入 sos，即句子开始，输出是爱，然后我们在这里取输出标记，爱，并将其附加到输入中，然后我们再次将其提供给模型，模型将生成下一个标记，爱能，然后我们取模型输出的最后一个标记，我们再次将其附加到输入中，模型将生成下一个标记，然后我们再次取下一个标记，即 can，我们将其附加到输入中，并再次将其提供给模型，模型将快速输出下一个标记，我们为所有必要的步骤进行这一过程，知道我们达到句子结束标记，那时我们就知道模型已经完成了输出，现在，这实际上并不是 llama 的训练方式，但这是一个很好的例子，向你展示下一个标记预测任务是如何工作的，

现在，这种方法存在一个问题，让我们看看为什么，在推理的每一步，我们只对模型输出的最后一个标记感兴趣，因为我们已经有了之前的标记，然而，模型需要访问所有之前的标记来决定输出哪个标记，因为它们构成了其上下文或提示，所以我的意思是，例如，要输出单词 D，模型必须看到这里所有的输入，我们不能只给出 seize，模型需要看到所有的输入才能输出这个最后的标记 D，但关键是，这是一个序列到序列的模型，所以他会生成这个序列作为输出，即使我们只关心最后一个标记，所以我们做了很多不必要的计算，重新计算了这些标记，而这些标记我们实际上在之前的步骤中已经有了，所以让我们找到一种方法，避免这种无用的计算，这就是我们使用 KV 缓存所做的事情，因此，KV 缓存是一种在推理过程中减少对已见标记计算量的方法，因此，它仅在 Transformer 模型的推理过程中应用，它不仅适用于 LLaMA 中的 Transformer，还适用于所有 Transformer 模型，因为所有 Transformer 模型都是这样工作的

这是一个描述，展示了在下一个标记预测任务中自注意力机制的工作原理，所以，正如你再我的前几张幻灯片中看到的，我们在这里有一个包含 n 个标记的查询矩阵，然后是键的转置，因此，查询可以被认为是向量行，其中第一个向量代表第一个标记，第二个向量代表第二个标记，等等，然后键的转置是相同的标记，但转置了，所以行变成了列，这产生了一个 n 乘 n 的矩阵，所以如果初始输入矩阵是 9，输出矩阵将是 9 乘 9,然后我们将其乘以 v 矩阵，这将产生注意力，然后，注意力被输入到 Transformer 的线性层，然后线性层将生成 logits，logits 被输入到 softmax 中，softmax 帮助我们决定从词汇表中选择哪个标记，再次提醒，如果你不熟悉这个过程，请观看我之前关于 Transformer 推理的视频，你会更清楚的理解，所以这是一个在自注意力机制中一般层面上发生的事情的描述，现在让我们一步步来看，

所以在推理的第一步，我们只有一个标记，如果你还记得之前，我们只使用了句子的起始标记，所以我们取句子的起始标记，将其与自身相乘，所以转置后，它将产生一个 1 乘 1 的矩阵，所以这个矩阵是 1 乘 4096，乘以另一个 4096 乘 1 的矩阵，这将产生一个 1 乘 1 的矩阵，为什么是 4096，因为 LLaMA 中的嵌入向量是 4096，然后输出，这个 1 乘 1 的矩阵乘以 V，将在这里产生输出标记，而这将成为我们输出的第一个标记，然后我们取这个输出标记，将其附加到下一步的输入中，所以现在我们有两个标记作为输入，它们与自身的转置版本相乘，将产生一个 2 乘 2 的矩阵，然后该矩阵乘以 V 矩阵，将产生两个输出标记，但我们只对模型输出的最后一个标记感兴趣，就是这个注意力，也是，然后将其附加到时间步 3 的输入矩阵中，所以在时间步 3，我们有 3 个标记，它们与自身的转置版本相乘，将产生一个 3 乘 3 的矩阵，然后该矩阵乘以 V 矩阵，我们得到这 3 个输出标记，但我们只对模型输出的最后一个标记感兴趣，所以我们再次将其作为输入附加到 Q 矩阵，现在有 4 个标记，它们与自身的转置版本相乘，将产生一个 4 乘 4 的输出矩阵，然后该矩阵乘以这里的矩阵，将产生这里的注意力矩阵，但我们只对最后一个注意力感兴趣，它将被再次添加到下一步的输入中，但我们已经注意到一些东西，

首先，我们已经在计算这个标记与这个、这个标记与这个、这个标记与这个之间的点积的矩阵中，所以这个矩阵是这两个矩阵之间所有点积的结果，我们可以看到一些东西，第一件事是，我们已经在之前的步骤中计算了这些点积，我们可以缓存它们吗？所以让我们回到之前，如你所见，这个矩阵在增长，二、三、四，看，有很多注意力，因为每次我们推理变换器时，我们都在给它，给变换器一些输入，所以它在重新计算所有这些点积，这很不方便，因为我们实际上已经在之前的步骤中计算过它们了，所以有没有办法不再计算它们，我们可以缓存他们吗，是的，我们可以，然后，由于模型是因果性的，我们不关心一个标记与其前驱的注意力，而只关心它与前一个标记的注意力，所以，如你所记得的，在自注意力中，我们应用了一个掩码，对吧，所以掩码基本上是我们不希望一个词与它后面的词进行点积，而只希望与它前面的词进行点积，所以基本上我们不希望这个矩阵主对角线上方的所有数值，这就是为什么我们在自注意力中应用了掩码，但重点是我们不需要计算所有这些点积，我们唯一感兴趣的点积是这最后一行，所以因为我们在上一步的基础上增加了标记四作为输入，所以我们只有这个新标记，标记四，我们想要知道标记四如何与所有其他标记交互，所以基本上，我们只对这里的最后一行感兴趣，而且，因为我们只关心最后一个标记的注意力，因为我们想从词汇表中选择单词，所以我们只关心最后一行，我们不关心在自注意力输出序列中生成这三个注意力分数，我们只关心最后一个，那么有没有办法去除所有这些冗余计算呢？

是的，我们可以用 KV 缓存来实现，让我们看看如何实现，所以使用 KV 缓存，我们基本上是缓存了，键和值，每次我们有一个新标记时，我们将其附加到键和值上，而查询仅是前一步的输出，所以在开始时，我们没有任何前一步的输出，所以我们只使用第一个标记，所以推理的第一步与没有缓存时是一样的，我们有标记一，它与自身生成一个 1 乘 1 的矩阵，乘以一个标记，然后产生一个注意力，然而，在第二步时，我们不将其附加到之前的查询中，我们只是用这里的新标记替换之前的标记，然而，我们保留键的缓存，所以我们保留键中的前一个标记，并将最后一个输出附加到这里的键和值中，如果你进行这个乘法，他会生成一个 1 乘 2 的矩阵，其中第一个元素是标记二与标记一的点积，以及标记二与标记二的点积，这实际上是我们想要的，如果我们在乘以 V 矩阵，它只会产生一个注意力分数，这正是我们想要的，我们再次这样做，所以我们取这个注意力，这将成为下一步推理的输入，这个标记三，我们将其附加到之前缓存的 K 矩阵和 V 矩阵中，这个乘法将产生一个输出矩阵，我们可以在这里看到，这个输出矩阵与 V 矩阵的乘法将产生一个输出标记，就是这个，我们知道使用这个来选择哪个标记，然后我们将其作为下一步推理的的输入，通过将其附加到缓存的键和缓存的 V 矩阵中，我们进行这个乘法，我们将得到这个矩阵，它是 1 乘 4 的，这是标记 4 与标记 1、标记 4 与标记 2、标记 4 与标记 3 以及标记 4 与自身的点积，我们乘以 V 矩阵，这将只产生一个注意力，这正是我们选择输出标记所需的，

这就是为什么它被称为 KV 缓存的原因，因为我们保留了键和值的缓存，正如你所见，KV 缓存让我们节省了很多计算，因为我们不再进行以前需要做的大量点积运算，这使得推理速度更快，

接下来，我们要讨论的是分组多查询注意力，但在讨论分组多查询注意力之前，我们需要先介绍它的前身，多查询注意力，让我们来看看，所以让我们从问题开始，问题是 GPU 太快了，如果你看这份数据表，这是来自英伟达 A100 GPU，我们可以看到 GPU 在计算和执行计算方面非常快，但在从其内存传输数据方面并不那么快，这意味着，例如，A100 可以使用 32 位精度每秒执行 19.5万亿次浮点运算，而它每秒只能传输 1.9 千兆字节，在数据传输方面，它的速度几乎是计算速度的 10 倍慢，这意味着有时瓶颈不在于我们执行了多少操作，而在于我们操作需要多少数据传输，而这取决于我们计算中涉及的张量的大小和数量，例如，如果我们在同一个张量上计算相同的操作 n 次，可能比在 n 个不同标记上计算相同的操作更快，即使它们的大小相同，这是因为 GPU 可能需要移动这些张量，因此，这意味着我们的目标不仅应该是优化我们算法执行的操作数量，还应该尽量减少算法执行的内存访问和内存传输，因为与计算相比，内存访问和内存传输在时间上更为昂贵，这在软件中也会发生，当我们进行 I/O 操作时，例如，如果我们复制，在 CPU 中进行一些乘法运算，或者从硬盘读取一些数据，从硬盘读取数据比在 CPU 上进行大量计算要慢的多，这是一个问题，

现在，在这篇论文中，我们介绍了多查询注意力，这篇论文来自 Noam Shazeer，他也是 attention is all you need 这篇论文的作者之一，在这篇论文中，他提出了这个问题，他说，好吧，让我们看看多头注意力，也就是批量多头注意力，这是在原论文 attention is all you need 中提出的多头注意力，让我们看看这个算法，并计算一下执行的算数操作数量以及这些操作涉及的总内存，他计算出算数操作的数量是 O(bnd^2)，其中 b 是批量大小，n 是序列长度，d 是嵌入向量的大小，而操作中涉及的总内存，包括所有参与计算的张量（包括派生的张量）等于 O(bnd+bhn^2+d^2)，其中 h 是多头注意力中的头数，现在，如果我们计算总内存与算数操作数量之间的比率，我们得到这个表达式：O(1/k+1/bn)，在这种情况下，比率远小于 1,这意味着我们执行的内存访问次数远小于算数操作的数量，因此，在这种情况下，内存访问不是瓶颈，我的意思是，这个算法的瓶颈不是内存访问，实际上是计算的数量，正如你之前看到的，当我们引入 KV 缓存时，我们试图解决的问题是计算的数量，但通过引入 KV 缓存，我们创造了一个新问题，我的意思是，不是新问题，而是我们，我们实际上有了一个新的瓶颈，而且不再是计算了，所以，这个算法是使用 KV 缓存的多头自注意力，这减少了执行的操作数量，因此，如果我们看执行的算数操作数量，他是 bnd2，操作中涉及的总内存是 bn2d+nd2，两者的比率是这个，O(n/d+1/b)，这是总内存与算术操作数量之间的比率，

这意味着，当 n 非常接近 d 时，这个比率将变为 1,  或者当 b 非常接近 1 是，或者在批量大小为 1 的极限情况下，这个比率将变为 1,这是一个问题，因为现在，当这个条件被验证时，确实如此，内存访问成为了算法的瓶颈，这也意味着，要么我们保持嵌入向量的维度远大于序列长度，但如果我们增加序列长度而不使用嵌入向量的维度大幅增加，内存访问将成为瓶颈，所以我们能做的是，我们需要找到一个更好的方法，为了解决前一个算法中内存成为瓶颈的问题，我们引入了多查询注意力，

作者的做法是移除K和V中的H维度，同时保留Q的H维度，它仍然是一个多头注意力，但仅针对 Q，这就是为什么他被称为多查询注意力，因此，我们将只为 Q 设置多个头，但 K 和 V 将由多有头共享，如果我们使用这个算法，比率变为：1/d +n/dh + 1/b，所以与之前的 n/d 相比，现在是 n/dh，因此，我们减少了 n/d 因子，将 n/d 的比率减少了 h 倍，因为我们移除了 k 和 v 的h个头，所以，收益，即性能提升，实际上是重要的，因为现在这个比率不太可能变为 1,但当然，通过从 k 和 v 中移除头，我们的模型也将拥有更少的参数，它也将拥有更少的自由度和复杂性，这可能会降低模型的质量，它确实会稍微降低模型的质量，我们将会看到，因此，如果我们比较一下，例如，在从英语到德语的翻译任务中的 BLEU 分数，我们可以看到，多头注意力即原始注意力论文中的注意力的 BLEU 分数为 26.7,而多查询注意力的 BLEU 分数为 26.5,作者还将它与多头、局部和多查询局部进行了比较，其中局部意味着他们将注意力计算限制在每个 token 的前31 个位置，我们可以在这里看到，但通过减少k和v的头数所带来的性能提升是显著的，因为你可以看到推理时间，例如，在原始多头注意力和多查询注意力中，推理时间从 1.7 微妙加上解码器的 46微妙，减少到 1.5微妙加上解码器的 3.8微妙，所以 在这里总共，或多或少，我们用了 48 微妙，而在这里，我们用了大约 6 微妙用于多查询，因此，从性能角度来看，在推理过程中这是一个巨大的好处

我们来谈谈分组多查询注意力，因为现在，我们刚刚介绍了 KV 缓存和多查询注意力，但多查询注意力的下一步是分组多查询注意力，这是在 LLaMA 中使用的方法，所以让我们来看看它，

在多查询中，我们只为查询设置了多个头，但键和值只有一个头，在分组多查询注意力中，基本上，我们将查询分成多个组，例如，这是第一组，这是第二组，第三组和第四组，而对于每个组，我们有一个不同的 K 和 V 头，这是多头和多查询之间的一个很好的折中，在多头中是一对一的对应关系，而在多查询中是 n 对 一 的对应关系，所以在这种情况下，我们仍然有多个键和值的头，但他们在数量上比查询的头要少，这是模型质量和速度之间的一个很好的折中，因为无论如何，我们在这里从减少键和值头数的计算优势中受益，但在质量方面没有牺牲太多，

现在模型的最后一部分，正如你在这里看到的，LLaMA 模型中的前馈层已经转换为，其激活函数已经改为 SwiGLU 函数，让我们看看它是如何工作的，

所以 SwiGLU 函数在这篇由 Noam Shazeer 撰写的著名论文中进行了分析，他也是我们之前看到的注意力模型和多查询注意力的作者之一，所以让我们来看看这篇论文，因此，作者通过在 Transformer 架构的前馈层中使用不同的激活函数，比较了 Transformer 模型的性能，我们感兴趣的是这里的 SwiGLU，它基本上是开关函数，其中 beta 等于 1,计算在 x 乘以 w 矩阵（这是一个参数矩阵），然后乘以 x 乘以 v（V是另一个参数矩阵）和 w2（这是另一个参数矩阵），所以将这个与原始的前馈网络进行比较，这里我们有三个参数矩阵，而在原始的前馈网络中我们只有两个，为了使比较公平，作者减少了这些矩阵的大小，使得模型的总参数量与普通 Transformer 保持一致，

在普通 Transformer 中，我们有一个前馈网络，它是 ReLU 函数，即这个最大值，零，等等，是 ReLU 函数，而我们只有两个参数矩阵，实际上，Transformer 的一些后续版本没有偏置项，所以这是从论文中提取的公式，但实际上有很多实现是没有偏置项的，而在 LLaMA 中，我们使用这种计算作为前馈网络，这是我从 LLaMA 的代码库中提取的代码，正如你所见，它正是模型所说的，它是 SwiGLU 函数，为什么是 SwiGLU 函数，因为它是一个 beta 等于 1 的开关函数，当开关函数具有这个表达式并且我们给 beta 等于 1 时，它被称为 sigmoid 线性单元，它具有这个图形，并被称为 SiLU，

所以它是 SiLU 函数，在 w1 乘以 x 后，再乘以 w3,然后我们将其应用于 w2,所以我们有三个矩阵，这三个矩阵基本上是线性层，现在它们使用这种线性层的并行版本，但它们仍然是线性层，如果我们看这个 siLU 函数的图形，我们可以看到它有点像 ReLU，但在这里，在达到零之前，我们不会立即取消激活，我们在这里保留一个小尾巴，以便即使是非常接近零的负值也不会被函数自动取消，那么让我们看看他的表现如何

这个 silu 函数实际上表现的非常好，他们评估了当我们使用这个特定函数时模型的困惑度，我们可以看到这里的困惑度是最低的，困惑度基本上意味着模型对其选择的确定性有多低，而 swiglu 函数表现良好，然后他们还在许多基准测试上进行了比较，我们看到 swiglu 函数在其中许多测试中表现相当不错，那么为什么 swiglu 激活函数表现的如此出色呢？

如果我们看这篇论文的结论，我们会发现，我们并没有解释为什么这种架构似乎有效，我们将其成功归因于，和其他一切一样，神的恩典，实际上，这有点好笑，但也有些真实，因为在大多数深度学习研究中，我们并不知道为什么事情会以它们的方式工作，因为想象一下，你有一个拥有 70B 参数的模型，你怎么能在修改了一个激活函数后，证明每个参数会发生什么变化呢？要构建一个能够解释模型为何以特定方式反应的模型并不容易，通常我们做的是，我们可以简化模型，这样我们就可以使用这个非常小的模型，然后对事情为何以他们的方式工作做出一些假设，或者我们可以在实际层面上进行，所以我们拿一个模型，稍微修改一下，做一些消融研究，然后检查哪个表现更好，这在机器学习的许多领域也会发生，例如，我们做了很多网格搜索来为模型找到合适的参数，因为我们无法事先知道哪一个会表现良好，哪一个需要增加或减少，因为他取决于很多因素，不仅取决于所使用的算法，还取决于数据，以及所使用的特定计算，还包括所使用的归一化，所以有很多因素，并没有一个公式可以解释所有事情，这就是为什么研究需要对模型的方差进行大量研究，以找到可能在某个领域有效但在其他领域效果不佳的东西，所以在这种情况下，我们使用 SwiGLU 主要是因为它在实践中与这种模型配合的很好，

感谢大家观看这个长视频，希望你们能更深入了解 LLaMA 的工作原理以及它与标准 Transformer 模型的不同之处，我知道这个视频相当长，而且我知道有些部分可能很难跟上，所以我建议你们多看几遍，特别是那些你们不太熟悉的部分，并将这个视频与我之前关于 Transformer 的视频结合起来看，我会把章节列出来，这样你们可以轻松找到想要的部分，但这是你需要做的，你需要多次观同一个概念才能真正掌握它，我希望再做一个视频，从零开始编写 LLaMA 模型，这样我们就可以将所有这些理论付诸实践，但你知道，我是在空闲实践做这个的，而我的空闲时间并不多，所以感谢大家观看我的视频，并请订阅我的视频，因为这是我继续发布关于人工智能和机器学习的精彩内容的最好动力，感谢观看，祝你们度过美好的一天，