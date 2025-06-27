
https://huggingface.co/spaces/nanotron/ultrascale-playbook

发表时间：2025.02.19
建议阅读时长：2-4 天
作者：Nouamane Tazi, Ferdinand Mom, Haojun Zhao, Phuc Nguyen, Mohamed Mekkouri, Leandro Werra, Thomas Wolf


数千块 GPU 完美协同运作，发出嗡嗡声响。这就是训练当今最强大的人工智能模型所需要的——一种计算能力的交响乐，直到最近，这还是精英研究实验室的专属领域。开源改变了这一局面，但并未完全改变。没错，你可以下载最新的 Llama 或 DeepSeek 模型。没错，你可以阅读它们的技术和实验报告。但最具挑战性的部分——训练代码，以及协调 GPU 来训练这些庞大系统所需的知识和技术，仍然隐藏在复杂性之中，并分散在一系列相互独立的论文中，而且很多时候还存在于私有代码库中 。

这本开源书籍旨在改变这种状况。从基础开始，我们将引导你了解将大型语言模型的训练从单个 GPU 扩展到数十个、数百个甚至数千个 GPU 所需的知识，并通过实际代码示例和可复现的基准测试来阐释理论。

随着用于训练这些模型的集群规模不断扩大，人们发明了各种技术，如数据并行、张量并行、流水线并行或上下文并行以及 ZeRO 或内核融合等，以确保 GPU 始终得到充分利用。这大大缩短了训练时间，并充分利用了这种昂贵的硬件。更重要的是，随着扩大人工智能训练规模的挑战超出了构建初始模型的范畴，团队发现，在专门数据上对大型模型进行微调通常能产生最佳结果，而这一般也涉及相同的分布式训练技术。在本书中，我们将循序渐进地介绍所有这些技术——从最简单到最精妙的技术——同时保持一条连贯的主线，以便理解每种方法的由来。

我们将假设你对当前大型语言模型（LLM）架构有一些基本的简单了解，并且大致熟悉深度学习模型的训练方式，但你可能总体上对分布式训练不太熟悉。如有需要，可以在 [DeepLearning.ai](https://www.deeplearning.ai/)  的优质课程或 [PyTorch 教程板块](https://pytorch.org/tutorials/beginner/basics/intro.html) 找到模型训练的基础知识。本书可看作是三部曲中的第二部，第一部是我们关于预训练数据处理的博客文章，即所谓的“[FineWeb 博客文章](https://huggingface.co/spaces/HuggingFaceFW/blogpost-fineweb-v1)”。读完这两篇博客文章后，你应该几乎掌握了充分了解如今大型语言模型构建方式所需的所有核心知识，只是缺少关于数据混合和架构选择的一些最终要点来完善整个知识体系（敬请关注第三部……）。

这本书建立在以下 *三个一般性基础* 之上：

*理论和方法简介*：在深入研究代码和实验之前，我们希望从高层次上了解每种方法的工作原理以及其优势和局限性。你将了解语言模型的哪些部分消耗了你的内存以及训练过程中何时发生这种情况。你将学习如何通过并行化模型来解决内存限制问题，并通过扩展 GPU 来提高吞吐量。因此，你将了解以下小部件如何计算 Transformer 模型的内存细分。

(请注意，在此小部件中我们仍然缺少流水线并行处理。作为留给读者的一项练习，后续会添加相关内容。)

[交互式组件]

（如果你不知道这个小部件中发生了什么，别担心。这就是我们在这里的原因。）

虽然这个部件给出了理论上的细分情况，但我们还制作了[以下工具](https://huggingface.co/spaces/nanotron/predict_memory)，可用于预测训练运行期间的内存使用情况：
![Predict Memory Tool|600](https://nanotron-ultrascale-playbook.static.hf.space/assets/images/predict_memory_tool.png)

*清晰的代码实现*：理论是一回事，但在实际实现过程中，我们会发现各种各样的边缘情况和重要细节。这就是为什么我们在可能的情况下会提供实现参考链接。根据具体情况，我们会使用两个代码参考：

* [picotron](https://github.com/huggingface/picotron)  存储库是为教育而构建的，因此它通常在单个独立的短文件中实现概念。
* 另一方面，要查看可用于生产的代码，我们将参考 [nanotron](https://github.com/huggingface/nanotron)  实现，这是在 Hugging Face 使用的一个用于生产的训练代码库。

如果你想观看关于分布式训练的视频，而不是阅读博客或 Picotron 代码，那就去看看 [Ferdinand's YouTube channel](https://www.youtube.com/watch?v=u2VSwDDpaBM&list=PL-_armZiJvAnhcRr6yTJ0__f3Oi-LLi9S)。

*真实的训练效率基准测试*：最后，如何真正扩展你的 LLM 训练取决于你的基础设施，比如芯片类型、互连等，我们无法给出一个统一的方案。不过，我们将提供一种对几种设置进行基准测试的方法，这正是我们在我们的集群上所做的！我们进行了超过 4100 次分布式实验（包括测试运行在内超过 16000 次），使用了多达 512 个 GPU 来扫描许多可能的分布式训练布局和模型大小。

[交互式图]

正如你所见，有很多内容需要涵盖。在深入分布式训练的细节之前，让我们先快速地从宏观层面看一下本书将要涉及的挑战。

## 一、High level 概述

本书中将要介绍的所有技术都针对以下三个关键挑战中的一个或几个，我们在整本书中都会不断遇到这些挑战：

1. *内存使用*：这是一个硬性限制——如果一个训练步骤无法放入内存，训练就无法进行。
2. *计算效率*：我们希望我们的硬件大部分时间都用于计算，因此需要减少在数据传输或等待其他 GPU 执行工作上花费的时间。
3. *通信开销*：我们希望尽量减少通信开销，因为它会使 GPU 处于空闲状态。为了实现这一点，我们将尽量充分利用节点内（快速）和节点间（较慢）的带宽，并尽可能使通信与计算重叠。

在许多地方，我们会看到我们可以用其中一个（计算、通信、内存）来换取另一个（例如重新计算或张量并行）。找到正确的平衡是扩展训练的关键。

由于这本书内容非常丰富，我们制作了一份[备忘单](https://nanotron-ultrascale-playbook.static.hf.space/dist/assets/images/ultra-cheatsheet.svg)来帮助你浏览本书并了解主要要点。在这艰难时刻，请将其牢记于心！

> [!tip]
> 
> #### 第一步：将模型加载到内存中
> 
> **GPU 资源充足情况：**
> 
> * *小模型 (<10B)*：使用单一并行技术，例如 TP 或 ZeRO-3/DP，并在 8 个 GPU 上进行完全重计算。
> - *大模型 (10B+)*：需要超过 8 个 GPU，你有以下选项：
> 	- 结合 TP (TP-8) 与 PP
>     - 结合 TP (TP-8) 与 DP (ZeRO-3)
>     - 仅使用 ZeRO-3（即纯数据并行）
> - *512+ GPU 规模*：纯 DP/ZeRO-3 因通信成本而变得低效，最好将其与 TP 或 PP 结合使用。
> - *1024+ GPU 规模*：推荐设置是 TP-8 与 PP（ZeRO-2 + CP）
> - 特殊情况：对于*长上下文*，考虑使用 CP；对于 *MoE 架构*，使用 EP。
> 
> **GPU 资源不足情况：**
> 
> - *减少内存使用*：使用完全激活检查点和/或梯度累积。
> 
> #### 第二步：满足目标全局批次大小
> 
> 实验告诉我们哪种批次大小适合训练（4-40M tokens）。因此，我们需要根据第一步调整批次大小。
> 
> - *增加全局批次大小*：扩大 DP 或 CP 规模，或增加梯度累积步骤。
> - *减少全局批次大小*：减少 DP 或 CP，转而使用其他并行策略。
> 
> #### 第三步：优化训练吞吐量
> 
> 目前没有最佳配置的一般方案，因此我们应进行实验：
> 
> - *扩大 TP* 到节点规模以减少其他并行策略的需求
> - 在保持目标 GBS 的同时*增加 DP* 与 ZeRO-3
> - *使用 PP* 如果通信因 DP 的瓶颈而变得不可行
> - *调整微批次大小* 以平衡最大 GBS、模型大小、计算/通信


[![Cheatsheet](https://nanotron-ultrascale-playbook.static.hf.space/dist/assets/images/ultra-cheatsheet.svg)](https://nanotron-ultrascale-playbook.static.hf.space/dist/assets/images/ultra-cheatsheet.svg)

## 二、第一步：在单个 GPU 上训练

如果你想在阅读体验中增添一些播客的感觉，那么在阅读本书开篇章节时，不妨听听 NotebookLM 主持人对这些章节的讨论。

在开始扩展到多个 GPU 之前，让我们先快速回顾一下模型训练的基础知识。当在单个 GPU 上训练模型时，训练通常包括三个步骤：

1. 前向传播，将输入通过模型以产生输出；
2. 反向传播以计算梯度；
3. 使用梯度更新参数的优化步骤。

（正如我们稍后将看到的，这些步骤可能会重复或交织在一起，但现在我们先从简单的开始。）

它通常看起来是这样的：

[交互图]

在此图中，顶部的各框可视为模型内的连续层（最后一行同理）。红色框表示在反向传播过程中针对这些层分别计算得到的相关梯度 。

*批量大小（$bs$）* 是模型训练的重要超参数之一，影响模型收敛和吞吐量。

在训练初期，较小的批量大小可能有用，能够快速在训练空间中移动，找到最佳学习点。然而，在模型训练后期，小批量大小会使梯度保持噪声，模型可能无法收敛到最优的最终性能。在另一个极端，大批量大小虽然能提供非常准确的梯度估计，但往往会较少利用每个训练样本，导致收敛速度变慢，并可能浪费计算资源。你可以在 OpenAI 关于大批量训练的论文[^1] 或 MiniMax-01 [技术报告](https://filecdn.minimax.chat/_Arxiv_MiniMax_01_Report.pdf)的第 4.2 节中找到关于这个话题的早期讨论。

例如，在 DeepSeek-V3/R1 的训练中，“在前 469B 个 tokens 的训练过程中，批量大小从 3072 个输入序列逐渐增加到 15360 个，然后在剩余的训练中保持在 15360 个输入样本”。

批量大小也会影响在给定文本数据集上进行训练所需的时间：较小的批量大小将需要在相同数量的样本上进行更多的优化器步骤训练。优化器步骤在计算时间上是昂贵的，因此与使用较大的批量大小相比，总的训练时间将会增加。话虽如此，请注意，批量大小通常可以在最佳批量大小附近进行相当大的调整，而对模型的性能没有重大影响，即最终模型性能对确切批量大小值的敏感度通常在最佳批量大小附近相当低。

在大型语言模型（LLM）预训练社区中，批量大小通常以标记（token）而非样本数量来表示（$bst$ = batch size tokens），这使得训练数据量通常与训练过程中使用的具体输入序列长度无关。

在最简单的情况下，即在单台机器上进行训练时，可以根据模型输入序列长度（seq）按以下方式计算$bs$（以样本为单位）和 $bst$：$$\text{bst} = \text{bs} \times \text{seq}$$从这里开始，我们将以样本为单位展示批量大小的公式，但你可以通过将其与序列长度相乘，始终得到以其 token 为单位的对应值。

近期大型语言模型（LLM）训练的一个理想批次规模通常在每个批次 4-60M 个 tokens 之间。多年来，批次大小以及训练语料库一直在稳步增长：Llama-1 是在 ~4M 个 tokens 的批次大小下，使用 1.4T 个 tokens 进行训练的；而 DeepSee k则是在 ~60M 个 tokens 的批次大小下，使用 14T 个 tokens 进行训练的。

*而当我们将模型的训练扩展到这些大批次大小时，我们的第一个挑战已经来临：内存不足问题。当我们的 GPU 没有足够的内存来容纳目标批次大小的一整批数据时，我们该怎么办？*

让我们首先快速了解是什么导致了我们的内存不足问题。这将帮助我们对训练模型所需的内存有一些有用的直觉。

### 2.1 Transformers 中的内存用量

在训练神经网络模型时，会在内存中存储以下几项：

- 模型权重
- 模型梯度
- 优化器状态
- 计算梯度所需的激活值

> [!NOTE]
> 你可能会认为对于一个模型，你可以精确计算其内存需求，但有一些额外的内存占用因素使得精确计算变得困难：
> 
> - CUDA 内核通常需要 1-2 GB 的 GPU 内存，你可以通过运行 `import torch; torch.ones((1, 1)).to("cuda")`，然后使用 `nvidia-smi` 检查 GPU 内存来快速验证这一点。
> - 缓冲区、中间结果占用的一些剩余内存，以及由于碎片化而无法使用的一些内存  
> 
> 我们将忽略最后这两个因素，因为它们通常较小且是固定因素。

这些项目以不同 *形状* 和 *精度* 的张量形式存储。*形状* 由批量大小、序列长度、模型隐藏维度、注意力头、词汇量大小以及我们稍后将看到的潜在模型分片等超参数决定。*精度* 指的是 FP32、BF16 或 FP8 等格式，分别需要 4、2 或 1 个字节来存储张量中的每个值。我们将在混合精度训练部分全面讨论不同的精度及其权衡，现在我们只需记住这些不同格式的内存需求是不同的，这将影响我们需要存储的项目的内存使用情况。

那么，如何能快速通过这些变量确定内存使用情况呢？一种简单的方法是通过实验来实际测量一下。

#### 2.1.1 分析内存使用情况

使用 Pytorch profiler，我们可以了解整个训练过程中内存是如何分配的。我们可以看到，内存利用率并非一个静态的事物，在训练过程以及单个训练步骤中都会有很大的变化。

[交互式图]

（查看 A1：分布式训练性能剖析，了解如何对你的模型进行性能剖析的操作指南。）

显然，第一步与后续步骤看起来非常不同，但让我们首先看一下一个步骤的总体结构：首先，在进行前向传播时，激活值会迅速增加，然后，在反向传播过程中，梯度会逐渐累积，并且随着反向传播的进行，用于计算梯度的存储激活值会逐渐被清除。最后，我们执行优化步骤，在此期间我们需要所有梯度，然后在开始下一次前向传播之前更新优化器状态。

为什么第一步看起来不同：激活值迅速增加，然后平稳一段时间。在这第一步中，torch 缓存分配器进行了大量准备工作，为后续步骤准备内存分配，以便它们之后不需要搜索空闲内存块（参见[Zach 的博客](https://zdevito.github.io/2022/08/04/cuda-caching-allocator.html)）。在第一步之后，我们还看到优化器状态出现，这通常会抵消进一步训练步骤的内存使用。

你是否注意到，有时训练在第一步成功，但在后续训练步骤中却出现内存不足（OOM）的情况？这可以通过第一步之后优化器状态的累积来解释。

现在我们已经对内存有了初步了解，让我们看看放大训练规模通常是如何在满足各种项目（激活值、参数、梯度、优化器状态）的内存需求的同时，最大化计算效率并使其符合 GPU 的内存限制。

#### 2.1.2 权重、梯度、优化器状态内存占用

让我们从列表中的前 3 项开始：模型的权重、梯度和优化器状态。实际上我们可以很容易地估算出它们所需的内存。

对于一个简单的 Transformer 语言模型，其参数数量由[以下公式](https://michaelwornow.net/2024/01/18/counting-params-in-transformer)给出：$$N = h \times v + L \times (12 \times h^2 + 13 \times h) + 2 \times h$$
> [!tip]
> 计算如下：
> * 嵌入层：$h \times v$
> * 层归一化 ln_1：参数 $\gamma$ 和 $\beta$,  $2h$
> * 注意力层：Q、K、V、O = $4h^2+4h$，带偏置项
> * 层归一化 ln_2：参数 $\gamma$ 和 $\beta$,  $2h$
> * MLP 层：$h\times 4h + 4h + 4h \times h + h$  
> * 最后还有一层 ln：$2h$
> 

（我们排除了位置嵌入计数，因为我们没有使用固定的位置嵌入。）

在该等式中，$h$ 是隐藏维度，$v$​ 是词汇量大小，$L$​ 是模型中的层数。请注意，观察该等式我们可以发现，在较大的隐藏维度下起主导作用的项将是 $h^2$​ 项，因为当我们扩展参数时，它是唯一呈二次方增长的项。

参数和梯度的内存需求简单来说就是参数数量乘以每个参数的字节数。在传统的全精度（FP32）训练中，参数和梯度都需要 4 个字节，而优化器（如果我们使用 Adam）需要存储动量和方差，这又增加了每个参数另外两个 4 字节。总结如下：$$
\begin{align*}
m_{\text{params}} &= 4 \times N \\
m_{\text{grad}} &= 4 \times N \\
m_{\text{opt}} &= (4 + 4) \times N
\end{align*}$$现在让我们看看如果我们使用较低的精度，情况会发生怎样的变化。出于稳定性原因（见下面的混合精度训练部分），我们通常不使用完全的低精度训练，而是使用高低精度混合的方法，称为“混合精度”[^2]。如今混合精度训练的默认做法通常是使用 BF16 进行大部分计算——每个参数和梯度需要 2 字节——同时还需要额外的一份模型权重和梯度副本以 FP32 格式保存，因此每个参数总共需要 12 字节。除了参数和梯度之外，我们还需要存储优化器状态：对于 Adam 优化器来说，这需要动量和方差，为了数值稳定性，这些通常以 FP32 格式存储，每个使用 4 字节。

（当我们介绍 ZeRO 方法时，请查看下面的一些详细信息。）

这里是总结：$$\begin{align*}
m_{\text{params}} &= 2 \times N \\
m_{\text{grad}} &= 2 \times N \\
m_{\text{params\_fp32}} &= 4 \times N \\
m_{\text{opt}} &= (4 + 4) \times N
\end{align*}$$

> [!NOTE]
> 有些库以 fp32 格式存储梯度，这需要额外的 $m_{\text{grad\_fp32}}$ 的内存。例如在 nanotron 中就是这样做的，因为对于较小的值，bf16 是有损的，并且我们始终优先考虑稳定性。有关更多信息，请参阅[这个DeepSpeed 问题](https://github.com/microsoft/DeepSpeed/issues/1773)。
> 
> 参数的 FP32 副本（$m_{\text{params\_fp32}}$）在文献和代码库中有时被称为“主权重”。

有趣的是，*混合精度本身并不会节省总体内存*，因为它只是将内存以不同方式分配到三个组件中。实际上，如果我们以 BP16 累积梯度，相比全精度训练还会额外增加 4 个字节。不过它仍然具有优势，因为在半精度下进行前向/反向传播计算能让我们（1）在 GPU 上使用经过优化的更低精度运算，这些运算速度更快；（2）减少前向传播期间的激活内存需求，从前面的图表中我们可以看到，这部分在前向传播中占用了大量内存 。

让我们了解一下对于一个模型（全精度和混合精度给出相同的总体值），我们需要多少通用内存。

| **模型参数** | **FP32 或 BF16 不带 FP32 梯度累加** | **BF16 带有 FP32 梯度累加** |
| -------- | ---------------------------- | --------------------- |
| 1B       | 16 GB                        | 20 GB                 |
| 7B       | 112 GB                       | 140 GB                |
| 70B      | 1120 GB                      | 1400 GB               |
| 405B     | 6480 GB                      | 8100 GB               |
（使用 FP8 训练代替 BF16 会进一步减少内存使用，但它不太稳定，是一个非常活跃的研究课题（见这条[推文](https://x.com/xariusrke/status/1826669126955278401)），我们稍后会更详细地介绍。）

我们可以看到，一旦我们达到 *7B*（!），权重和优化器的需求就开始显著增加，并超过典型 GPU 内存的大小，例如 H100 GPU 的 80GB。

但目前，让我们从仍然适合单个 GPU 的模型开始，看看我们内存预算的最后一大贡献者：激活内存。

#### 2.1.3 激活内存占用

激活内存的计算比权重、梯度和优化器状态要复杂一些，部分原因在于它依赖于模型的输入。如果你不确定为何我们甚至在反向传播中需要存储激活值，这个[参考资料](https://www.determined.ai/blog/act-mem-2)是一个很好的快速回顾。在仔细检查了反向传播是如何计算之后，我们可以估算出混合精度中激活所需的总内存，并得出以下等式：$$
m_{\text{act}} = L \cdot \text{seq} \cdot bs \cdot h \cdot \left(34 + \frac{5 \cdot n_{\text{heads}} \cdot \text{seq}}{h}\right)$$这里，$L$ 表示层数，$seq$ 表示序列长度，$bs$ 表示样本中的批量大小，$h$ 表示模型的隐藏维度，$n_\text{heads}$ 表示头的数量。

要精确推导这些数字，你可以参考这篇关于重新计算的原始 NVIDIA 论文[^3]，它本质上要求你对 Transformer 层中每个操作之间的所有中间激活的大小进行一些核算。

这里有一个有趣的观察结果是，对于给定的模型，内存使用量并非静态的；相反，它会随批量大小呈线性扩展，并随序列长度呈二次方扩展。这意味着激活内存是当我们增加批量大小或使用更长序列进行训练时将会急剧增长的部分。例如，对于 Llama 模型（bs=1），我们可以使用这个公式来查看不同序列长度下内存使用量的变化情况 。

[交互图]

这张图讲述了一个引人注目的故事：*对于短序列（或对于小批量大小类似的情况），激活几乎可以忽略不计，但从大约 2-4k 个 tokens 开始，它们占用了大量内存*，而参数、梯度和优化器状态的使用量（我们稍后将讨论）大致独立于序列长度和批量大小。

*对于大的输入 tokens（即大批量/序列），激活成为迄今为止最大的内存负担。*

有没有办法驯服这种“激活爆炸”？好问题，读者！

现在是时候解释我们的第一种技术了——称为激活重新计算（activation recomputation）——这将有助于我们控制激活内存占用。这是当今大型模型训练工具箱中的一个必备工具。

### 2.2 激活重算

激活重新计算（也称为 *梯度检查点* 或 *重新物化*）背后的总体思想是在前向传播过程中丢弃一些激活以节省内存，并在反向传播过程中额外花费一些计算资源来即时重新计算这些激活。如果不进行重新计算，我们会存储两个可学习操作（例如前馈、层归一化等）之间的每个隐藏状态，以便在反向传播过程中使用它们来计算梯度。当我们使用重新计算时，通常只会在模型架构的几个关键点存储激活，丢弃其余的激活，并在反向传播过程中从最近的保存的激活中即时重新计算它们，基本上是再次执行前向传播的一个子部分，以用计算资源换取内存。通常情况如下：

[交互图]

有几种策略可用于选择要存储的关键激活值：

* *Full*：我们在 Transformer 模型每层的转换点处对激活进行检查点处理。这通常被称为完整策略，因为它本质上需要在反向传播过程中通过每层进行一次前向传播。这种策略节省的内存最多，但在计算方面成本最高。它通常会使计算成本和时间增加 30-40%，这是非常明显的。
* *Selective*：一般来说，我们可以比完整策略做得更好。重新计算论文[^3]的作者进行了详细分析，研究哪些激活增长最大且在浮点运算次数（FLOPs）方面的重新计算成本最低。结果表明，注意力计算属于此类，因此我们通常可以舍弃它们，专注于对代价高昂的前馈计算进行检查点处理。对于GPT-3（175B）模型而言，这意味着*激活内存减少 70%，计算成本增加 2.7%*。

（在像 DeepSeek V3 这样的最新模型中，会执行选择性检查点操作，使用所谓的“多头潜在注意力”（MLA）存储更小规模的注意力激活值，以优化激活内存的使用。）

让我们看看重新计算策略在实践中能将内存占用降低多少，以及选择性重新计算如何在节省内存和重新计算成本之间达到良好的平衡。

[交互图]

这里另一个明显的趋势是，*对于较小的模型，长序列的激活作用更大，因此重新计算的效果变得更加明显。*


> [!NOTE]
> 当你衡量训练设置的效率以及它如何利用 GPU/TPU/加速器时，通常需要考虑重新计算以计算总的FLOPS（每秒浮点运算次数），并将其与 GPU/TPU/加速器 的理论最大 FLOPS 进行比较。在计算训练步骤的 FLOPS 时考虑重新计算会得到一个称为“硬件 FLOPS”的值，这是加速器上实际执行的操作数量。将此数值除以训练步骤的持续时间和最大加速器 FLOPS，就得到了*硬件 FLOPS 利用率（HFU）*。
> 
> 然而，归根结底，真正重要的是在给定数据集上训练模型所需的端到端时间。因此，在比较各种GPU/TPU/加速器 时，如果其中一个加速器（例如）提供足够的内存来跳过重新计算，从而每秒执行的操作更少（较低的 HFU），但训练速度更快，那么它应该得到奖励而不是惩罚。因此，一种替代方法是计算所谓的模型 *浮点运算利用率（MFU）*，与 HFU 不同，它只考虑模型前向传播+反向传播所需的操作，并且不包括在测量的 FLOP 中的重新计算。因此，这个值比训练实现更具体地针对模型。

如今，大多数训练框架都使用 FlashAttention（我们将在下文进一步介绍），它通过在反向传播中重新计算注意力分数和矩阵，而不是存储它们，从而在其优化策略中原生地集成激活重新计算。因此，大多数使用 Flash Attention 的人已经在利用选择性重新计算了。

*正如你现在所理解的，由于重新计算，激活重新计算会略微增加浮点运算次数（FLOPs），同时显著减少内存访问开销。*

这种权衡在具有小型高速内存的硬件（如 GPU）上特别有利，因为访问内存通常比执行计算要慢。尽管涉及额外的操作，但总体效果通常也是更快的计算速度，此外内存占用也大大降低。

现在我们已经了解了重新计算，正如我们在上面的图表中所看到的，我们可以控制激活内存的使用！

然而，激活仍然对批量大小存在线性依赖关系，而且我们上面条形图中的所有分析都是使用 bs=1 进行的，因此当我们转向更大的批量大小时，这可能再次成为一个问题。不要绝望，因为我们还有另一个工具——*梯度累积* 来解决问题！

### 2.3 梯度累积

梯度累积是一种非常直接的方法，用于避免内存爆炸，其方法是将我们的批次分割成微批次。我们将依次对每个微批次执行前向和后向传播，计算梯度，并且，正如名称所示，在执行优化器步骤之前，将所有微批次的梯度相加。实际上，优化步骤不是基于梯度的总和而是基于梯度的平均值进行的，这样结果就与梯度累积步骤的数量无关。

我们将每次前向传播的批量大小称为*微批量大小*（$mbs$）。我们将每次优化器步骤之间的整体批量大小称为*全局批量大小*（$gbs$）。如果我们对每 8 次前向/后向传播执行一次优化器步骤，则全局批量大小将是微批量大小的 8 倍。

我们现在所说的“全局批量大小”（global batch size），因此对应于到目前为止我们为简便起见仅称之为“批量大小”（batch size）的概念（我们现在使术语更加精确以避免歧义）。

通过梯度累积，全局批量大小可以按以下方式简单计算：$$\text{bs} = \text{gbs} = \text{mbs} \times \text{grad\_acc}$$梯度累积使我们能够在内存占用保持不变的情况下，有效地将批量大小增加到无限大（甚至更大！）。梯度累积也与激活重新计算兼容，以进一步减少内存。

![image.png|500](https://nanotron-ultrascale-playbook.static.hf.space/assets/images/gradaccumulation_diag.png)

(使用梯度累积意味着我们需要保留缓冲区，在整个训练步骤中持续累积梯度。而不使用梯度累积时，在反向传播过程中计算梯度的同时会释放激活内存，这意味着峰值内存更低。)

梯度累积允许我们通过仅计算部分微批次，来减少随批次大小线性增长的激活的内存占用。

*然而，一个缺点是，梯度累积需要在每个优化步骤中进行多次连续的前向/后向传播，从而增加了计算开销并减慢了训练速度。天下没有免费的午餐！*

但如果你仔细观察，可能会注意到每个微批量的前向/后向传播实际上可以并行运行。前向/后向传播彼此独立，唯一的区别是输入样本相互独立。看来是时候开始将我们的训练扩展到多个 GPU 上了！

在此之前，让我们快速了解一下如何通过分布式训练工具箱中最有用的工具之一——分析器（profiler）的简要介绍来可视化计算和通信。这个工具对于理解和验证 GPU 之间的通信和计算是如何进行的以及瓶颈在哪里非常有用。

#### 2.3.1 剖析 GPU 计算和通信

PyTorch 的 [profiler](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html) 允许我们精确追踪和可视化在训练期间 CPU 和 GPU 上正在发生的情况。它原生集成在 PyTorch 中。让我们看看如何使用它：

```python
with torch.profiler.profile( 
	activities=[ 
		torch.profiler.ProfilerActivity.CPU,
		torch.profiler.ProfilerActivity.CUDA, 
	], 
	schedule=torch.profiler.schedule(wait=1, warmup=1, active=3),
	on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/profile'),
	with_stack=True
) as prof: 
	for step in range(steps): 
		train_step() 
		prof.step()
```

这会生成一个跟踪记录，我们可以在 TensorBoard 或 Chrome 的跟踪查看器中对其进行可视化。该跟踪记录显示：

* CPU 线程异步向 GPU 启动内核  
* 多个 CUDA 流并行处理计算和通信  
* 内核执行时间和内存分配

![profile_trace_annotated.png|600](https://nanotron-ultrascale-playbook.static.hf.space/assets/images/profile_trace_annotated.png)

示例跟踪显示 CPU 线程异步向 GPU 启动内核，计算内核和通信在不同 CUDA 流中并行进行。

该跟踪有助于识别以下瓶颈：

- 可重叠的顺序计算和通信
- 等待数据传输时的 GPU 空闲时间
- CPU 与 GPU之间的内存移动
- 来自 CPU 的内核启动开销

理解这些模式对于优化分布式训练性能至关重要。例如，正如我们稍后将讨论的，跟踪将清楚地显示梯度同步是否与反向计算正确重叠。

现在让我们获取一个更大的工作站🖥️ ，配备几个 GPU，然后开始研究我们的第一种扩展技术，即*数据并行性*，正如我们将看到的，它*只是梯度累积的并行版本*。


## 三、数据并行（DP）

数据并行（DP）背后的理念是在多个 GPU 上复制模型（我们将副本称为“模型实例”），并针对每个 GPU 并行地对不同的微批次数据进行前向传播和反向传播，因此得名数据并行。你可能已经在简单的训练示例中见过数据并行，但正如你很快会看到的，在本节中我们将深入探讨这一内容，所以即使你已经了解一般方法，也请继续关注。

![image.png|500](https://nanotron-ultrascale-playbook.static.hf.space/assets/images/dp_diagram.png)

（如果你不熟悉 broadcast、gather 或 all-reduce 等分布式通信模式，我们在 A0：并行编程速成课程中准备了一个小型速成课程。）

每个 GPU 使用不同的微批次意味着每个 GPU 中会有不同的梯度，因此为了使不同 GPU 上的模型实例保持同步，将使用一种称为 “all-reduce” 的操作对来自模型实例的梯度进行平均处理，该操作在反向传播期间、优化器步骤之前进行。

这涉及我们的第一个“分布式通信”原语：***all-reduce***，它处理 GPU 实例和节点之间的同步和通信。

![image.png|600](https://nanotron-ultrascale-playbook.static.hf.space/assets/images/dp_overlap1.svg)

一个简单的分布式数据并行（DP）实现方式是等待反向传播完成，这样我们就有了所有梯度，然后触发所有分布式数据并行 ranks 之间的一次 all-reduce 操作来同步这些梯度。但这种先计算后通信的顺序步骤是***大忌***！因为我们不希望像上图那样，在进行通信时我们的 GPU 处于闲置状态。

相反，我们应该尽可能地让通信和计算重叠，使它们尽可能同时发生。

让我们来看看三种优化方法，它们能让我们比最初的简单实现做得更好！

### 3.1 三种优化方法

#### 3.1 方案一：将梯度同步与反向传播重叠

我们刚刚描述的朴素 DP 方法的主要缺点是，在反向传播（*计算*）之后，我们必须等待梯度同步（*通信*）才能更新参数。我们能否将此通信与我们的计算重叠？答案是肯定的！

如下图所示，在计算前面层的梯度之前，就可以收集并求和某一层的梯度。例如，一旦最后一层的反向传播完成，这些梯度就可以在为前面的层继续进行反向计算的同时被收集和求和。

![image.png](https://nanotron-ultrascale-playbook.static.hf.space/assets/images/dp_overlap2.svg)

这可以在 PyTorch 中通过每个参数上附加一个 *all-reduce 钩子函数* 实现 。一旦该参数的梯度准备好，就会触发 all-reduce 操作，而其他参数的梯度仍在计算中。这种方法将大部分 all-reduce 操作与梯度计算重叠，从而提高效率。以下是一个用于附加钩子的简单函数：

```python
def register_backward_hook(self, hook):
    """
    Registers a backward hook for all parameters of the model that 
    require gradients.
    """
    for p in self.module.parameters():
        if p.requires_grad is True:
            p.register_post_accumulate_grad_hook(hook)
```

计算和通信的重叠减少了等待整个模型梯度同步的时间。梯度同步可以（至少部分地）与反向传播并行进行，显著加快数据并行速度。以下是具有同步重叠的朴素数据并行（DP）的完整实现：

👉 Picotron 中存在重叠的朴素动态规划实现（点击展开）

```python
class DataParallelNaive(nn.Module):
    """
    Naive Data Parallelism. Not used in practice. But it is a good starting point to understand how data parallelism works.
    It implements a simple all-reduce operation to synchronize gradients across multiple processes.
    And `no_sync` context manager to disable gradient synchronization.
    """
    def __init__(self, module):
        """
        Initializes the DataParallel wrapper for a given module.

        Args:
            module (nn.Module): The model to be wrapped for data parallelism.
            process_group (torch.distributed.ProcessGroup): The process group used for gradient synchronization. 
                                                            It could be a data parallel or context parallel group.
        """
        super().__init__()
        self.module = module
        self.require_backward_grad_sync = True # whether to synchronize gradients during backward pass. Set to False when using gradient accumulation
        self.register_backward_hook(self._allreduce_grads)
    
    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)
    
    def register_backward_hook(self, hook):
        """
        Registers a backward hook for all parameters of the model that require gradients.    
        """
        for p in self.module.parameters():
            if p.requires_grad is True:
                p.register_hook(hook)
                
    def _allreduce_grads(self, grad):
        """
        Performs an all-reduce operation to synchronize gradients across multiple processes.    
        """
        # No synchronization needed during gradient accumulation, except at the final accumulation step.
        if self.require_backward_grad_sync:
            dist.all_reduce(grad, op=dist.ReduceOp.SUM, group=pgm.process_group_manager.cp_dp_group)
            grad /= pgm.process_group_manager.cp_dp_world_size
        return grad 
    
    @contextlib.contextmanager
    def no_sync(self):
        """
        A context manager to temporarily disable gradient synchronization. 
        This is useful for performing multiple backward passes during gradient accumulation without synchronizing 
        gradients in between.
        """
        self.require_backward_grad_sync = False
        yield
        self.require_backward_grad_sync = True
```


> [!important]
> [all-reduce 和 ring-reduce 在数据同步上的示意图](https://blog.dailydoseofds.com/p/all-reduce-and-ring-reduce-for-model)


这是我们第一个 “*计算与通信重叠*” 的例子，在本文中我们将多次讨论它，这是实现最大扩展效率的一项关键技术。但我们可以进一步提高效率！

#### 3.2 方案二：梯度分桶

GPU 操作在处理大张量时通常比在多个小张量上运行许多操作更高效。通信操作也是如此。因此，我们可以将梯度有利地分组到桶中，并对同一桶内的所有梯度启动单个 all-reduce，而不是对每个梯度执行独立的 all-reduce。通常看起来如下：

![dp_overlap3.svg|600](https://nanotron-ultrascale-playbook.static.hf.space/assets/images/dp_overlap3.svg)

这就像在装运前将物品装入箱子一样。发送几个大箱子比发送许多小箱子更高效。通过对每个桶执行单个 all-reduce 操作，我们可以显著减少通信开销并加快通信操作。

以下是采用分桶方式的代码实现：

👉 Bucket DP 在 Picotron 中的实现（点击展开）

```python
class DataParallelBucket(nn.Module):
    """
    Data Parallelism with gradient grouped into buckets to reduce the communication overhead.
    """
    def __init__(self, module, bucket_cap_mb=25, grad_type = torch.float32):
        """
        Initialize the DataParallelBucket module.
        
        Args:
            module (nn.Module): The model to be parallelized.
            process_group: The process group for gradient synchronization, which can be either 
                           a data parallel group or a context parallel group.
            bucket_cap_mb (int, optional): The maximum size of each gradient synchronization bucket in megabytes. 
                                           Defaults to 25 MB.
            grad_type (torch.dtype, optional): The data type of gradients, defaulting to float32.
        """
        super().__init__()
        self.module = module
        self.require_backward_grad_sync = True # whether to synchronize gradients during backward pass. Set to False when using gradient accumulation
        grad_size = 2 if grad_type == torch.bfloat16 else 4 # float32 gradient: 4 bytes
        bucket_size = bucket_cap_mb * 1024 * 1024 // grad_size # number of gradients in one bucket
        self.bucket_manager = BucketManager(module.parameters(), pgm.process_group_manager.cp_dp_group, bucket_size, grad_type)
        self.register_backward_hook()
        self._post_backward_callback_set = False # whether the callback for wait gradient synchronization is set
        
    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    def backward(self, input_tensor, output_tensor, output_tensor_grad):
        return self.module.backward(input_tensor, output_tensor, output_tensor_grad)
    
    def register_backward_hook(self):
        """
        Registers a backward hook to manually accumulate and synchronize gradients.
        
        This hook serves two main purposes:
        1. PyTorch does not natively support gradient accumulation with mixed precision.
        2. After gradient accumulation, it flags parameters as ready for synchronization.
        
        The gradient accumulation functions are stored to prevent them from going out of scope.
        
        References:
        - https://github.com/NVIDIA/Megatron-LM/issues/690
        - https://pytorch.org/docs/stable/generated/torch.autograd.graph.Node.register_hook.html
        - https://arxiv.org/abs/2006.15704 (page 5)
        """
        self.grad_accs = []
        for param in self.module.parameters():
            if param.requires_grad:
                # Expand so we get access to grad_fn.
                param_tmp = param.expand_as(param)
                # Get the gradient accumulator function.
                grad_acc_fn = param_tmp.grad_fn.next_functions[0][0]
                grad_acc_fn.register_hook(self._make_param_hook(param, self.bucket_manager))
                self.grad_accs.append(grad_acc_fn)
                
    def _make_param_hook(self, param: torch.nn.Parameter,bucket_manager: BucketManager):
        """
        Creates the a hook for each parameter to handle gradient accumulation and synchronization.
        """
        def param_hook(*unused):
            """
            The hook called after the gradient is ready. It performs the following:
            1. Accumulates the gradient into the main gradient.
            2. Adds a post-backward callback to wait for gradient synchronization completion.
            3. Marks the parameter as ready for synchronization.
            """
            if param.requires_grad:
                assert param.grad is not None
                param.main_grad.add_(param.grad.data) # accumulate the gradients
                param.grad = None
                
                # skip the gradient synchronization (gradient accumulation/PP micro batches)
                if self.require_backward_grad_sync:
                    # Add a callback to wait for gradient synchronization. Ensures the callback is added only once.
                    # Callback is executed after the backward pass. It should be added per backward pass.
                    if not self._post_backward_callback_set:
                        Variable._execution_engine.queue_callback(self._post_backward)
                        self._post_backward_callback_set = True
                        
                    # mark the parameter as ready for gradient synchronization. 
                    bucket_manager.mark_param_as_ready(param) 
        return param_hook
    
    @contextlib.contextmanager
    def no_sync(self):
        """A context manager to disable gradient synchronization."""
        self.require_backward_grad_sync = False
        yield
        self.require_backward_grad_sync = True
        
    def _post_backward(self):
        """
        A post-backward callback that waits for gradient synchronization to finish, then copies 
        the synchronized gradients back to the parameters' grad attribute.
        
        This method is called after the backward pass and before the optimizer step.
        """
        self.bucket_manager.wait()
        self._post_backward_callback_set = False
        # copy to params.grad so we can use the optimizer to update the parameters
        for p in self.module.parameters():
            if p.requires_grad:
                p.grad = p.main_grad.to(p.dtype) # In PyTorch, you cannot assign a gradient with one data type to a tensor of another data type.

    def reset(self):
        """
        Reset the bucket manager and zero out gradients in the model
        """
        self.bucket_manager.reset() 
```

#### 3.3 方案三：与梯度累积的相互作用

最后，正如我们之前看到的，梯度累积通过在用 `optimizer.step()` 更新参数之前执行多次前向和后向传播来工作。当将梯度累积与数据并行性结合时，我们希望在同步梯度时要小心。

在一个简单版本中，在累积过程中每次反向传播后都会自动触发 all-reduce 操作，这是次优的，因为在最后一步之后进行单次 reduce 将产生相同的效果，同时减少开销。

在 PyTorch 中，通常的解决方法是在不需要进行 reduce 的后向传播过程中添加一个 [`model.no_sync()`](https://github.com/pytorch/pytorch/blob/5ea67778619c31b13644914deef709199052ee55/torch/nn/parallel/distributed.py#L1408-L1435)装饰器，该装饰器可以禁用梯度同步。

> [!NOTE]
> 在执行通信操作时，张量在内存中必须是连续的，以避免多余的内存拷贝。为了以最优方式实现这一点，我们通常会预先分配大小与激活值或模型参数相匹配的连续缓冲区，专门用于通信。虽然这加快了通信速度，但在一定程度上也导致了训练期间的峰值内存使用量增加。

现在让我们看看这对全局批量大小意味着什么。

### 3.2 重新审视全局批量大小

我们可以使用新添加的数据并行和梯度累积参数来更新我们的批量大小公式：$$\text{bs} = \text{gbs} = \text{mbs} \times \text{grad\_acc} \times \text{dp}$$这里 $\text{grad\_acc}$ 是梯度累积步数，$\text{dp}$ 是用于数据并行的并行实例数量。

给定一个目标全局批量大小，我们因此可以通过梯度累积步骤来换取数据并行进程，从而加速训练。

在实际应用中，由于数据并行本质上是并行的，而梯度累积具有顺序性，人们倾向于尽可能多地增加数据并行节点（DP）而非采用梯度累积。当仅扩展数据并行性在 GPU 用完之前不足以达到目标全局批量大小时，就在数据并行的基础上添加梯度累积。

(关于数据并行性进一步阅读的一个好的资源是 https://siboehm.com/articles/22/data-parallel-training)

能够将训练分布到不同的样本上，为我们提供了第一个并行化的维度，因此这被称为 1D 并行（我们后续将逐步介绍另外四个维度）。

### 3.3 到目前为止我们的旅程

让我们快速总结一下如何设置我们的第一个 1D 并行训练，并为最佳数据并行设置提供一个草案配方：

1. 我们首先应通过查阅文献或开展测量模型收敛情况的实验来确定最佳的（全局）批量大小（以 tokens 为单位，`GBST`）。
2. 然后我们选择一个用于训练的序列长度，同样可以通过查阅文献或开展实验来确定。一般来说，对于我们目前的评估工作，2-8k 个 tokens 能可靠地发挥良好效果（我们在此不深入探讨训练方法，不过各团队通常会在训练结束时增加序列长度，混入一些更长上下文的数据样本，以达到如今的更长上下文尺寸）。
3. 现在我们已经知道了批量大小（`GBS`）。我们可以通过逐渐增加本地批量大小，直至耗尽内存，从而找出单个 GPU 上的最大本地批量大小（`MBS`）。
4. 最后，我们确定目标 DP 可用的 GPU 数量。GBS 与 DP 的比值能让我们得出实现所需 GBS 还需要的梯度累积步数。

(例如，DeepSeek 和 Llama 模型在主要预训练阶段是以 4k tokens 的序列长度进行训练的。)

(2-8k 在预训练中效果很好的原因是，网络上非常长的文档极为罕见。有关详细分析，请参阅 [Harm 的博客文章](https://www.harmdevries.com/post/context-length/)。)

如果梯度累积比率小于 1，也就是说我们有太多的 GPU（称为 GPU 丰富🤑），我们可以选择不使用所有的 GPU，探索更大的全局批量大小，或者测试较小的 MBS（每个 GPU 的批量大小）是否会加速训练。在后一种情况下，我们会优先考虑整体吞吐量而不是单个 GPU 的计算效率，使用比可能的更小的 MBS 来加快训练速度。

现在是时候举一个具体的例子了：假设我们想要训练一个最近提出的模型，该模型的全局批量大小（GBS）为 4M tokens，序列长度为 4k。因此，我们的批量大小将是 1024 个样本（我们选择最接近的 2 的幂次方）。假设我们观察到单个 GPU 在内存中只能容纳微批量大小 MBS=2，并且有 128 个 GPU 可用于训练。这意味着通过 4 个梯度累积步骤，我们将实现每个训练步骤 1024 个样本或 4M tokens 的目标。现在，如果我们突然有 512 个 GPU 可用呢？我们可以通过保持 MBS=2 并将梯度累积步骤设置为 1 来实现相同的 GBS，从而实现相同的训练，并获得更快的训练速度！

> [!NOTE]
> 请记住，在 512 个及以上 GPU 的规模下，根据所使用的网络，通信操作将开始受*环形延迟*（信号沿环形传输一圈所需的时间）的限制，这意味着我们无法再完全重叠数据并行（DP）通信。这将降低我们的计算效率并影响吞吐量。在这种情况下，我们应该开始探索其他并行维度。

虽然数据并行性能够很好地将  all-reduce 梯度同步与反向计算重叠以节省时间，但这种优势在大规模情况下开始崩溃。为什么呢？因为随着我们添加越来越多的 GPU（数百个或数千个），协调它们之间的开销显著增长，并且网络需求对于所获得的收益来说变得过大。结果，我们每向系统中添加一个额外的GPU，我们的设置将变得越来越低效。

让我们通过一些基准测试来看看这在实践中是如何实现的：

[交互图]

我们发现，在超过某个限制后，我们的吞吐量开始显著下降，而每个 GPU 的内存使用量保持不变，并且不会因为增加更多的 DP ranks 而受到影响。

*数据并行是我们首个（简单）的策略，用于将训练扩展到更多的 GPU 上。这种技术类似于梯度累积，但它对微批次的前向传播和反向传播进行并行处理，从而提高吞吐量！*

然而，敏锐的读者可能已经注意到，这是假设我们至少能将一个输入样本的前向传播（mbs=1）装入我们的 GPU 内存。但并非总是如此！我们可以看到，即使启用了激活重新计算，较大的模型也无法装入单个 GPU 中：

> [!tip]
> 提示：你可以通过将模型参数数量乘以 2 来快速估算模型参数所需的最小内存，例如 70B → 140GB（=133GiB）

[交互图]

我们还发现，在达到一定的扩展水平后，数据并行开始出现一些限制性的通信开销。对于这些更大的模型或大批量大小，我们还有其他选择吗？幸运的是，我们确实有一些解决方案。它们要么涉及将一些张量移动到 CPU，要么将权重/梯度/优化器状态张量拆分到 GPU 设备上！让我们开始深入了解它们。

有两种主要的拆分方法：并行性（张量并行、上下文并行或流水线并行）和共享（DeepSpeed Zero 或 PyTorch FSDP）。这两种方法在某种程度上是正交的，实际上可以结合起来！

共享范式与 DP 密切相关，因此我们将首先通过研究 ZeRO 方法来对其进行了解！

### 3.4 ZeRO (**Ze**ro **R**edundancy **O**ptimizer)

在本节中，我们将介绍 DeepSpeed ZeRO（零冗余优化器），这是一种内存优化技术，旨在减少大型语言模型训练中的内存冗余。

虽然数据并行是一种有效的扩展训练的方式，但在每个 DP rank 上简单复制优化器状态、梯度和参数会引入显著的内存冗余。ZeRO 通过将优化器状态、梯度和参数在数据并行维度上进行划分来消除内存冗余，同时仍然允许使用完整的参数集进行计算。这有时需要在 DP rank 之间进行更多的通信，这些通信是否能够完全重叠，我们接下来将会看到！

在本博客中，我们将重点关注 ZeRO-1 到 ZeRO-3，因为这应该能让我们全面了解它如何帮助减少内存占用，同时展示需要考虑的权衡。你可以在 [DeepSpeed 文档](https://www.deepspeed.ai/tutorials/zero/) 中找到更多 ZeRO 的相关内容。

这种方法分为 ZeRO 的三个可能的优化阶段：

- ZeRO-1：优化器状态分区
- ZeRO-2：优化器状态+梯度分区
- ZeRO-3（也称为 FSDP，即“完全分片数据并行”）：优化器状态+梯度+参数分区

（当我们说分区时，是指沿着 DP 轴进行分区，因为 ZeRO 是数据并行的一部分。稍后我们会看到，我们还可以沿着其他轴进行分区。）

你可能忽略了我们在可进行分片处理的事物中的激活操作。由于模型的每个 DP 副本接收不同的微批次，因此每个 DP rank 上的激活操作也各不相同，所以它们不会被复制，也就无法进行分片！

让我们更仔细地看看通过对每个 ZeRO 阶段进行分区，我们能节省多少！

#### 3.4.1 内存使用情况再探

你可能还记得我们在前面的章节中提到的标准训练期间优化器状态、梯度和参数的内存使用情况。我们把模型参数的数量记为 $Ψ$（之前用 $N$ 表示，但这里我们使用原始 ZeRO 论文的符号表示法）。在使用 Adam 优化器的混合精度训练中（更多细节见后面的章节），我们需要存储的每一项的内存使用量为：

- 模型的参数（半精度，即 bf16/fp16）：$2Ψ$
- 模型的梯度（半精度，即 bf16/fp16）：$2Ψ$
- 模型的 fp32 参数和优化器状态：$4Ψ+(4Ψ+4Ψ)$
- 模型的 fp32 梯度：$4Ψ$（可选，仅在我们要以 fp32 累积梯度时计算）

如果我们不在 fp32 中累积梯度，那么总的内存消耗为 $2Ψ+2Ψ+12Ψ$；如果我们进行累积，那么将是$2Ψ+6Ψ+12Ψ$。为简单起见，我们现在先关注不进行 fp32 梯度累积的情况，不过你可以将受 ZeRO-2 和 ZeRO-3 影响的梯度项的额外字节数加上去。

ZeRO 的理念是将这些对象分片到 DP 各个 rank 中，每个节点仅存储这些项的一个切片，当且仅当需要时才对这些项进行重构，从而将内存使用量按数据并行度 $N_d$​ 进行划分 。

![zero_memory.svg|600](https://nanotron-ultrascale-playbook.static.hf.space/assets/images/zero_memory.svg)
这里 $Ψ$ 表示参数数量，$k$ 表示优化器状态的内存乘数（如我们刚刚看到的，对于 Adam，$k=12$），$N_d$ 表示 DP 度。

让我们通过探究每个 ZeRO 阶段的工作原理来解释这张图及其数值。我们将从 ZeRO-1 开始。

#### 3.4.2 ZeRO-1: 分区优化器状态

在普通 DP 中，所有进程在后向传播后收集相同的梯度，并同时执行相同的优化器步骤。这看起来像是很多重复的工作。我们能否避免这种情况，同时减少内存使用呢？

在 ZeRO-1 中，优化器状态被划分为 $N_d$ 个相等部分，其中 $N_d$ 是数据并行（DP）度。这意味着分布在每个 DP rank 上的每个模型副本仅跟踪 $1/N_d$ 的优化器状态。在优化步骤中，只有 $1/N_d$ 的 float32 权重被更新。

然而，在前向传播过程中，每个副本都需要所有参数，因此我们需要在优化器步骤之后添加一个额外的 ***all-gather*** 操作（这是我们遇到的第二种通信原语！），以便每个模型副本都有完整的更新后的权重集。

这解释了我们在上图中看到的内存占用公式 $2Ψ+2Ψ+kΨ/N_d$，以下是单个训练步骤的操作顺序总结：

- 在每个副本上使用相同的完整 bf16 参数集进行前向传播，但不同副本处理不同的微批次。
- 在每个副本上使用相同的完整梯度集进行反向传播，但不同副本处理不同的微批次。
- 对梯度执行 reduce-scatter 操作（我们将在下图中解释 reduce-scatter 原语）。
- 每个副本在其本地优化器上执行一步优化器操作（仅有 $1/N_d$ 优化器状态），以获得更新的 $1/N_d$ fp32 参数，然后将其转换为完整 bf16 参数集的 $1/N_d$。
- 在 bf16 参数之间执行 all-gather 操作，将缺失的切片发送回每个副本。这是 ZeRO 中的新操作，在普通的数据并行（DP）中未使用。

> [!NOTE]
> 注意：reduce-scatter 比 all-reduce 快 2 倍！_耶，第三种通信原语！_
> 

你可能会想知道这个 “reduce-scatter” 操作是什么，以及这一切看起来是怎样的，所以让我们借助下面的图示让这一切更加直观。我们将详细讲解前向/反向传播周期的所有步骤：

![dp_zero1.gif|600](https://nanotron-ultrascale-playbook.static.hf.space/assets/images/dp_zero1.gif)

在实际通信方面，与普通 DP 相比，Zero-1 将我们的 “all-reduce” 梯度通信更改为 “reduce-scatte” 操作，并在优化器步骤之后添加一个针对所有参数的 “all-gather” 操作。其过程如下：

![dp_zero1_overlap.svg|600](https://nanotron-ultrascale-playbook.static.hf.space/assets/images/dp_zero1_overlap.svg)

如果你一直关注，会从普通 DP 中回想起，我们可以在反向传播计算过程中重叠进行 all-reduce 梯度通信。在 ZeRO-1 中，我们还可以研究如何高效地重叠新添加的 bf16 参数 all-gather 操作。主要有两种策略：

- 在优化器步骤期间：我们可以在优化器更新部分参数后立即启动 all-gather 操作。这使得通信有可能与其他参数的更新重叠。
- 在前向传播期间：我们可以将每层参数的 all-gather 操作与前向传播过程重叠起来。

> [!NOTE]
> 不幸的是，这些技术并不容易实现，并且需要巧妙地使用钩子/分桶。在实际应用中，我们可以直接使用 PyTorch 原生的 ZeRO-3/FSDP 实现，并将 FSDPUnit 设置为整个模型，关于这个的更多细节稍后会介绍。

在 ZeRO-1 中，优化器状态已被分区，这意味着每个副本仅更新 $1/N_d$ 的优化器状态。敏锐的读者肯定已经注意到，其实一开始并不需要所有 DP ranks 上都有所有梯度，因为优化步骤只需要其中一部分梯度。这就引出了 ZeRO-2！

#### 3.4.3 ZeRO-2: 添加梯度分割

由于我们只需要在每个副本上拥有与优化器状态分片相对应的梯度分片，因此将梯度也类似地分片是有意义的。在反向传播过程中，我们不是对梯度执行 all-reduce 操作，而是只执行 reduce-scatter 操作！我们只在内存中传播所需的 $1/N_d$ 梯度，从而比 ZeRO-1 节省更多内存。

在 FP32 梯度累积的情况下，我们只需要保留 $1/N_d$ fp32_grads，用于累积来自 reduce-scatter 的 bf16 梯度。在优化器步骤中，我们使用这 $1/N_d$ fp32_grads。

![dp_zero2.gif|600](https://nanotron-ultrascale-playbook.static.hf.space/assets/images/dp_zero2.gif)

现在很容易看出，对梯度进行分片会导致 $2Ψ+\frac{2Ψ+kΨ}{N_d}$，并且随着 $N_d$​ 的增加，与基线相比，我们可以节省多达 8 倍的内存。在通信方面，与 ZeRO-1 的过程相同，唯一的区别是我们即时进行通信并释放。总的来说，就通信而言，ZeRO-2 也因此等同于普通的 DP 训练。

在通信方面，ZeRO-2 与 ZeRO-1 相似，它们都需要对梯度进行 reduce-scatter 操作，并对所有参数进行 all-gather 操作。
![dp_zero2_overlap.svg|600](https://nanotron-ultrascale-playbook.static.hf.space/assets/images/dp_zero2_overlap.svg)

> [!tip]
> 注意：您可能会注意到，与 ZeRO-1 相比，使用 ZeRO-2 并没有真正的额外开销，实际上 ZeRO-2 通常是最佳选择。


现在我们已经对梯度进行了分片处理，那么我们是否已经完成了任务，还是可以继续这样做呢？嗯，差不多。接下来就是 ZeRO-3！

#### 3.4.4 ZeRO-3: 添加参数分区

对于第 3 阶段，我们将上述在数据并行（DP）副本上对优化器状态和梯度进行分片的方法扩展到对模型的参数进行分片。

> [!NOTE]
> 这个阶段在 PyTorch 原生实现中也被称为 FSDP（完全共享数据并行）。在本文中，我们仅使用 ZeRO-3 这个术语，但无论何时看到它，你都可以将其理解为 FSDP 。
> 

那么，如果模型的所有部分都是分布式存储的，我们在实践中如何进行前向传播或反向传播呢？很简单，我们在需要时按需收集它们。在前向传播中，过程如下：

![dp_zero3_fwd.svg|600](https://nanotron-ultrascale-playbook.static.hf.space/assets/images/dp_zero3_fwd.svg)

因此，在进行前向传播并依次通过各层时，我们会按需检索必要的参数，并在不再需要这些参数时立即将它们从内存中清除。反向传播的工作方式相同，只是流程相反，我们会生成梯度分片：

![dp_zero3_bwd.svg|600](https://nanotron-ultrascale-playbook.static.hf.space/assets/images/dp_zero3_bwd.svg)

另一个问题是，在前向传播和反向传播步骤中，我们需要持续执行这些全规约操作。与 Zero-2 相比，在一个训练步骤中，这相当于额外增加了 $2⋅\text{num\_layers}−1$ 次 all-gathers 操作，而且正如我们在下图中看到的，每次操作都会带来一定的基础延迟开销 。

![dp_zero3_overlap.svg|600](https://nanotron-ultrascale-playbook.static.hf.space/assets/images/dp_zero3_overlap.svg)

在前向传播过程中，当我们需要参数时，我们会对它们执行 all-gather 操作，因此会产生 $Ψ$ 的通信开销。由于在前向传播中一旦用到参数就会立即丢弃，所以在反向传播过程中我们还需要再进行一次 all-gather 操作，这又产生了 $Ψ$ 的通信开销。最后，和 ZeRO-2 一样，我们对梯度也需要进行相同的 ***reduce-scatter*** 操作，这在通信方面同样需要 $Ψ$ 的开销。综上，总的通信开销为 $3Ψ$，而 ZeRO-2 的通信开销为 $2Ψ$。

这听起来可能像是会有大量的通信开销，但实际上情况还挺好的，因为我们可以采用所谓的预取（prefetching）技术，将下一层参数的通信与当前层的前向传播过程重叠起来。通过预取，在进行前向传播时计算当前层（第 $n$ 层）的前向过程的同时，我们会 “all-gather” 第 $n+1$ 层的权重；同样地，在计算第 $n$ 层的反向传播过程时，我们会 “all-gather” 第 $n-1$ 层的权重。当然，只有当我们对数据并行（DP）的扩展程度不太大时，这种重叠才是有效的。（经验法则：数据并行的规模不应超过 512）

在内存方面，我们可以看到我们的方程现在达到了其最终形式 $\frac{2Ψ+2Ψ+kΨ}{N_d}$，这意味着如果我们能够增加 DP ranks，至少对于模型相关参数而言，我们可以无限降低内存使用量。注意，这对中间激活值并无帮助，对于中间激活值，正如我们在前面章节中所看到的，我们可以使用激活值检查点和梯度累积的方法。

*让我们总结一下迄今为止在分布式数据并行（DP）和 ZeRO 方面的探索历程：我们已经看到，通过简单地增加模型副本，利用分布式数据并行（DP）可以显著提高训练的吞吐量。而借助 ZeRO，我们甚至能够训练那些通常无法放入单个 GPU 的模型，方法是将参数、梯度和优化器状态在分布式数据并行（DP）中进行分片处理，不过这会带来一定的通信开销。*

如果你想了解更多关于 FSDP1、FSDP2 以及它们周围一些实现复杂性的内容，你应该花些时间仔细阅读[这篇不错的博客](https://christianjmills.com/posts/mastering-llms-course-notes/conference-talk-012/)。

然而，这里存在一个限制，即 DP 仅在模型的一个层能适配单个 GPU 时才有效，而 ZeRO 只能对参数、梯度和优化器状态进行分区，却无法对激活内存进行分区！我们从激活内存的讨论中回忆一下，这部分内存随着序列长度和批量大小而扩展。自然地，我们可以简单地限制这些因素，但在实践中，我们并不希望由于硬件的限制而只能使用短序列长度进行训练。

[交互图]

为了克服这些问题，是时候探索一种新的、正交的并行性轴——张量并行性（TP）了。与依赖大量参数通信的 ZeRO3 不同，TP 提出在设备间对参数、梯度、优化器状态以及激活进行分片，而不需要在GPU 之间进行模型参数的通信。

什么？这怎么可能？！让我们一起探索这种看似神奇的方法吧！ 🙂


## 四、张量并行（TP）

所以我们已经使用 ZeRO 对模型的参数、梯度和优化器状态进行了分片，但是一旦激活内存超过我们的内存预算，我们就遇到了一个限制。欢迎张量并行（TP），这是一种对权重、梯度、优化器状态以及激活进行分片的方法，并且不需要在计算之前将它们全部收集起来。这听起来像是一个梦想！让我们首先看看张量并行是如何通过简单的矩阵乘法来工作的。

张量并行利用了矩阵乘法 $A\times B$ 的数学特性。要理解其工作原理，让我们来看看使这种并行化成为可能的两个基本方程式：$$\begin{align*}
1. \quad & A \cdot B = A \cdot \begin{bmatrix} B_1 & B_2 & \cdots \end{bmatrix} = \begin{bmatrix} AB_1 & AB_2 & \cdots \end{bmatrix} \\[1.2ex]
2. \quad & A \cdot B = \begin{bmatrix} A_1 & A_2 & \cdots \end{bmatrix} \begin{bmatrix} B_1 \\ B_2 \\ \vdots \end{bmatrix} = \sum_{i=1}^{n} A_i B_i
\end{align*}$$
这意味着我们可以通过以下两种方式来计算矩阵乘积：1）分别乘以 $B$ 的每一列；或者2）分别乘以每一行并将结果组合起来。在神经网络中，矩阵乘法通常以以下格式表示：$X \times W$，其中：

* $X$ 表示输入或激活值  
* $W$ 表示 `nn.Linear` 的权重

在实际操作中，该操作的一个小示例是这样的：

![TP diagram|240](https://nanotron-ultrascale-playbook.static.hf.space/assets/images/tp_diagram.svg)

让我们看看如何对这个操作进行并行化处理！在张量并行中，张量将沿着特定维度被分割成 $N$ 个分片，并分布在 $N$ 个 GPU 上。矩阵可以在列部分或行部分进行分割，从而实现行并行和列并行。接下来我们会看到，选择行分片还是列分片将需要不同的通信原语。

我们的第一个选择是使用按列分片（也称为***列线性*** ）：我们将把完整的输入矩阵复制到每个工作节点，这需要一个称为***广播*** 的操作，并将权重矩阵分割成列。然后将输入与部分权重矩阵相乘，最后使用 ***all-gather*** 操作合并结果。

![image.png|500](https://nanotron-ultrascale-playbook.static.hf.space/assets/images/tp_diagram2.png)

以下是按列进行张量并行的代码实现：

👉 Picotron 中的列并行张量并行实现（点击展开）

```python
class ColumnParallelLinear(torch.nn.Module):
    """Column Parallel Linear layer
    Y = XW + b, where weight matrix W is parallelized along its second dimension. W = [W_1, ..., W_p]
    This module returns the results of Y_i = XW_i + b_i in the forward method, Y_i is parallelized in the second dimension.
    Arguments:
        in_features: first dimension of weight matrix W.
        out_features: second dimension of weight matrix W.
        bias: If true, add bias
        init_method: method to initialize weights
        gather_output: If true, gather the output from all the partitions. This is used for the last linear layer
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        gather_output: bool = False,
        async_all_reduce: bool = False,
    ) -> None:
        super(ColumnParallelLinear, self).__init__()

        self.tp_world_size = pgm.process_group_manager.tp_world_size
        self.tp_rank = pgm.process_group_manager.tp_rank 

        self.in_features = in_features
        self.out_features = out_features
        assert out_features % self.tp_world_size == 0, "Hidden dimension must be divisible by the tensor parallel world size"
        self.output_size_per_partition = out_features // self.tp_world_size
        self.gather_output = gather_output
        self.async_all_reduce = async_all_reduce
        # Allocate space for the weight and bias
        # Note: torch.nn.functional.linear performs XW^T + b so we exchange the order of dimensions
        self.weight = nn.Parameter(torch.Tensor(self.output_size_per_partition, self.in_features)) # W_i
        if bias:
            self.bias = nn.Parameter(torch.Tensor(self.output_size_per_partition))
            with torch.no_grad():
                self.bias.zero_()
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        # Initialize weight tensor with the default initialization method used for nn.Linear in PyTorch
        master_weight = torch.empty(
            self.out_features, 
            self.in_features, 
            dtype=self.weight.dtype,
            device=self.weight.device,
            requires_grad=False
        )
        
        # Calculate bound based on master weight's input dimension
        k = 1 / master_weight.size(1)
        bound = math.sqrt(k)
        torch.nn.init.uniform_(master_weight, -bound, bound)
        
        # Split the model into size of self.output_size_per_partition
        weight_list = torch.split(master_weight, self.output_size_per_partition, dim=0)
        self.weight.data = weight_list[self.tp_rank].contiguous()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:  
        if self.async_all_reduce:
            output = linear_with_async_all_reduce(x, self.weight, self.bias) 
        else:
            output = linear_with_all_reduce(x, self.weight, self.bias) 
        if self.gather_output:
            output = GatherFromModelParallelRegion.apply(output)
        return output
```

第二个选项称为按行分片（也称为***行线性***分片）：细心的读者可能会猜到，行线性分片意味着我们将权重矩阵分割成行块。然而，这也要求我们对输入进行分割，这需要一个 ***scatter*** 操作，而不是像列线性分片中使用的广播。每个工作器上的结果已经是正确的形状，但需要求和以得到最终结果，因此在这种情况下需要一个 all-reduce 操作。

我们在这里看到了我们的第四个分布式原语：**_scatter_**！

![image.png|500](https://nanotron-ultrascale-playbook.static.hf.space/assets/images/tp_diagram3.png)

以下是按行进行张量并行的实现方式：

👉 Picotron 中的行并行张量并行实现（点击展开）

```python
class RowParallelLinear(nn.Module):
    """Linear layer with row parallelism.
    Y = XW + b. W is parallelized along its first dimension and X along its second dimension as:
               -   -
              | W_1 |
              | .   |
          W = | .   |        X = [X_1, ..., X_p]
              | .   |
              | W_p |
               -   -
    We assume that X is already parallelized. This is the case after ColumnParallelLinear.
    This module returns the results of Y = sum(X_i * W_i + b_i) in the forward method.
    Arguments:
        in_features: first dimension of matrix W.
        out_features: second dimension of matrix W.
        bias: If true, add bias
        init_method: method to initialize weights.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool):
        super(RowParallelLinear, self).__init__()

        self.tp_world_size = pgm.process_group_manager.tp_world_size
        self.tp_rank = pgm.process_group_manager.tp_rank 

        self.in_features = in_features
        self.out_features = out_features
        assert in_features % self.tp_world_size == 0, "Hidden dimension must be divisible by the tensor parallel world size"
        self.input_size_per_partition = in_features // self.tp_world_size

        self.weight = nn.Parameter(torch.Tensor(self.out_features, self.input_size_per_partition))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(self.out_features))
            # Always initialize bias to zero.
            with torch.no_grad():
                self.bias.zero_()
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        # Initialize weight tensor with same dtype and device as self.weight
        master_weight = torch.empty(
            self.out_features, 
            self.in_features, 
            dtype=self.weight.dtype,
            device=self.weight.device,
            requires_grad=False
        )
        
        # Calculate bound based on master weight's input dimension
        k = 1 / master_weight.size(1)
        bound = math.sqrt(k)    
        torch.nn.init.uniform_(master_weight, -bound, bound)
        
        # Split the model into size of self.input_size_per_partition
        weight_list = torch.split(master_weight, self.input_size_per_partition, dim=1)
        self.weight.data = weight_list[self.tp_rank].contiguous()

    def forward(self, x):
        # X_i * W_i^T + b
        output_parallel = F.linear(x, self.weight)
        # All-reduce across all the partitions.
        output = ReduceFromModelParallelRegion.apply(output_parallel)
        return output if self.bias is None else output + self.bias
```

既然我们已经了解了 Transformer 的基本构建模块，现在让我们看看如何在 Transformer 层中有效地组合它们！

### 4.1 Transformer 块中的张量并行性

为了提出一个可遵循的策略，让我们从一个简单的示例过渡到一个真实的模型构建模块。Transformer模型由两个主要的构建模块组成：前馈层（MLP）和多头注意力（MHA）。我们可以对这两者都应用张量并行性。

前馈部分可以通过先进行“列线性”操作，再进行“行线性”操作来实现并行化，这相当于在前向传播中进行广播以复制输入并进行 all-reduce 操作。请注意，在实际训练中不需要广播，因为我们可以确保输入已经在 TP rank 之间同步。这种设置比先进行“行线性”操作，再进行“列线性”操作更高效，因为我们可以跳过两个拆分操作之间的中间 all-reduce 操作。

![image.png|600](https://nanotron-ultrascale-playbook.static.hf.space/assets/images/tp_diagram4.png)

既然我们已经找到了 Transformer 前馈部分的一个高效模式，那么让我们来看看多头注意力块（MHA）。

我们通常可以采用类似的方法，其中 Q、K 和 V 矩阵以列并行的方式拆分，输出投影沿行维度拆分。对于多头注意力机制，列并行方法有一个非常自然的解释：每个工作器计算单个头或一组头的注意力。这种方法同样适用于多查询（MQA）或分组查询注意力（GQA），在这些机制中，键和值在查询之间是共享的。

然而，值得注意的是，张量并行度不应超过查询/键/值（Q/K/V）头的数量，因为每个张量并行（TP）等级都需要完整的头（否则我们就无法在每个 GPU 上独立计算注意力，并且将需要额外的通信操作）。如果我们使用分组查询注意力（GQA），张量并行度实际上应小于键/值（K/V）头的数量。例如，LLaMA-3 8B 有 8 个键/值头，所以张量并行度最好不超过 8。如果我们对这个模型使用 TP=16，我们需要在每个 GPU 上复制键/值头，并确保它们保持同步。

![image.png|650](https://nanotron-ultrascale-playbook.static.hf.space/assets/images/tp_full_diagram.png)

最后请注意，张量并行性仍然不是训练的万能良方。我们在模型的计算路径中直接添加了几种分布式通信原语，因此很难像我们在 ZeRO 中所做的那样将其与计算完全隐藏/重叠，我们的最终性能将是计算和内存增益与增加的通信开销之间权衡的结果。让我们来说明这一点：

![Forward pass in Tensor Parallelism|650](https://nanotron-ultrascale-playbook.static.hf.space/assets/images/tp_overlap.svg)

（通过对分块矩阵乘法并结合异步通信/计算，可以部分隐藏这种通信。）

观察张量并行多层感知机（MLP）（注意力机制同样适用）的操作时间线，我们能更好地理解其中涉及的权衡。在每个解码器层的前向传播中，我们会遇到一个同步点，即AllReduce操作，该操作无法与计算重叠。这种 *显露出来的通信开销* 对于在最终应用层归一化（LayerNorm）之前合并张量并行 ranks 之间的部分结果是必要的。

（例如，Megatron-LM/Nanotron 实现了 all-gather 与 FC1 计算的部分重叠，其中矩阵乘法结果的一部分将在另一部分仍在计算时开始发送到其他 GPU。）

张量并行确实有助于减少矩阵乘法的激活内存，因为中间激活被分片到多个 GPU 上。然而，对于像 LayerNorm 这样的操作，我们仍然需要收集完整的激活，这意味着我们没有获得我们本可以获得的全部内存优势。此外，张量并行引入了显著的通信需求，这在很大程度上取决于网络基础设施。无法将这种特定的 AllReduce 完全隐藏在计算背后意味着它会直接增加前向传播的关键路径。

（这一研究领域仍然是一个活跃的研究领域，近期的研究工作如 Domino[4] 探索了最大化这种重叠的新技术。）

让我们更仔细地看看在扩展 TP（张量并行）度时所涉及的权衡：

[交互图]

增加训练进程数（TP）会导致每个 GPU 的吞吐量降低（左图），但它能够处理更大的批量大小（右图），这说明了分布式训练中计算效率和内存可用性之间的权衡。

实际上，正如我们在左图中看到的那样，当扩展到 8 个 GPU 以上时，张量并行的通信开销变得尤为明显。虽然在单个节点内可以利用快速的 NVLink 互连实现张量并行，但跨节点则需要较慢的网络连接。我们观察到从 TP=8 增加到 TP=16 时出现了显著的下降，并且从 TP=16 增加到 TP=32 时下降更为陡峭。在更高的并行度下，通信开销变得如此之高，以至于迅速占据了计算时间。

话虽如此，张量并行通过将模型参数、梯度、优化器状态以及（在一定程度上）激活分布到多个 GPU 上，为内存使用提供了重要优势。让我们看看这对一个 70B 参数模型产生的影响：

[交互图]

提高张量并行度可减少每个 GPU 上模型参数、梯度和优化器状态所需的内存，从而让我们能够开始在单个 8 GPU 节点上拟合大型模型。

有没有办法从这种技术中获得更多的好处呢？我们已经看到，层归一化和 dropout 仍然需要在每个 GPU 上收集全部激活值，这在一定程度上抵消了内存节省的效果。我们可以通过寻找并行化这些剩余操作的方法来做得更好。

> [!NOTE]
> 关于张量并行训练中层归一化的一个有趣说明——由于每个张量并行（TP）ranks 在 all-gather 后看到相同的激活值，因此在反向传播后，层归一化权重实际上不需要 all-reduce 来同步它们的梯度。它们自然会在各等级之间保持同步。然而，对于 dropout 操作，我们必须确保跨 TP ranks 同步随机种子，以维持确定性行为 。

接下来让我们探讨一下张量并行的一种小型且自然的扩展方式，即*序列并行*，它所做的事情正是如此。

### 4.2 序列并行

序列并行（SP）涉及对张量并行（TP）未处理的模型部分（如 Dropout 和 LayerNorm）的激活和计算进行拆分，但沿输入序列维度而非隐藏维度进行拆分。

> [!NOTE]
> “序列并行”（Sequence Parallelism）这个术语有些含义过载：本节中的序列并行与张量并行紧密相关，并且适用于dropout（随机失活）和层归一化（layer norm）操作。然而，当我们将处理更长的序列时，注意力计算将成为瓶颈，这就需要诸如环形注意力（Ring-Attention）之类的技术，这些技术有时也被称为“序列并行”，但为了区分这两种方法，我们将把它们称为“上下文并行”（Context Parallelism）。所以，每次看到“序列并行”时，要记住它是与张量并行一起使用的（与上下文并行相对，上下文并行可以独立使用）。

这是因为这些操作需要访问完整的隐藏维度才能正确计算。例如，LayerNorm 需要完整的隐藏维度来计算均值和方差：$$\text{LayerNorm}(x) = \gamma \cdot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta$$
其中，$\mu = \text{mean}(x)$ 和 $\sigma^2 = \text{var}(x)$ 是在隐藏维度 $h$ 上计算得到的。

因此，即使这些操作在计算上成本较低，但由于需要完整的隐藏维度，它们仍然需要大量的激活内存。序列并行（SP）允许我们通过沿序列维度进行拆分，在多个 GPU 之间分担这一内存负担。

在实际操作中，我们将从左图过渡到右图：
![|400](https://nanotron-ultrascale-playbook.static.hf.space/assets/images/tp_sp_diagram.png)
该图表展示了我们如何使用不同的集合操作（分别标记为 “f” 和 “g”）在张量并行区域和序列并行区域之间进行转换。关键挑战在于高效地管理这些转换，同时保持较低的内存使用量并确保正确性。

在前向传播中：

* "f" 是一个空操作（无操作），因为激活已在各 ranks 之间进行了复制  
* "f*" 是一个 all-reduce 操作，用于同步激活并确保正确性

在反向传播中：

* "f*" 是无操作（no-op），因为梯度已在各 ranks 中被复制  
* "f"是 all-reduce 操作，用于同步梯度

这些运算 “f” 和 “f*” 被称为*共轭* 对，因为它们相互补充——当一个在正向中是无操作（no-op）时，另一个在反向中就是 all-reduce，反之亦然 。

对于序列并行（SP），我们使用标记为 “g” 和 “g*” 的不同操作。具体而言，我们在 SP 区域中避免使用 all-reduce 操作，因为这需要收集全部激活值，从而增加我们的峰值内存使用量，违背了 SP 的初衷。

那么这里到底发生了什么？正如一个著名的 LLM 所说，让我们一步一步来：

* **初始层归一化（SP 区域）**:
	* 输入张量 $X_1$ 和 $X_2$ $(b,s/2,h)$ 进入层归一化，已在序列维度上拆分
	- 每个 GPU 独立地在其序列块上计算层归一化，并给出 $Y_1$ 和 $Y_2$

* **第一次转换（SP → TP）:**
	- “g” 操作（all-gather）将 $Y_1$ 和 $Y_2$ 重新组合为完整序列长度
	- 恢复 $Y$ $(b,s,h)$，因为列线性需要完整的隐藏维度 $h$

* **第一次线性（TP 区域）:**
	* $A_1$ 是列线性的，因此它沿隐藏维度分割 $Y$  
	* GeLU 在每个 GPU 上独立应用
	* $Z_1$ 是 $(b,s,h/2)$

* **第二次线性（TP 区域）：**  
	* $B_1$ 是行线性的，因此它恢复隐藏维度  
	* $W_1$ 是 $(b,s,h)$ 

* **最终转换（TP → SP）：** 
	* “g*” 操作（reduce-scatter），在沿序列维度分散的同时确保前一行线性的正确性  
	* $W_1$ 是 $(b,s/2,h)$

![image.png|240](https://nanotron-ultrascale-playbook.static.hf.space/assets/images/tp_sp_diagram_zoomed.png)

序列并行性的一个关键优势在于它减小了我们需要存储的最大激活大小。在仅使用张量并行的情况下，我们不得不在不同位置存储形状为 $(b,s,h)$ 的激活值。然而，有了序列并行性，由于我们总是在序列维度或隐藏维度上进行拆分，最大激活大小减小为 $\frac{b⋅s⋅h}{tp}$。

跟踪在 TP 和 TP/SP 中以不同方式进行分片的所有部分有点困难——相信我们，我们也觉得很难进行映射，所以我们制作了这个小表格来总结在前向传播过程中，激活（即 `hidden_states`）的形状在隐藏维度 $h$ 和序列维度 $s$ 上是如何变化的：

|Region|TP only|TP with SP|
|---|---|---|
|Enter TP (Column Linear)|h: sharded (weight_out is sharded)  <br>s: full|h: sharded (weight_out is sharded)  <br>s: **all-gather** to full|
|TP Region|h: sharded  <br>s: full|h: sharded  <br>s: full|
|Exit TP (Row Linear)|h: full (weight_out is full + **all-reduce** for correctness)  <br>s: full|h: full (weight_out is full + **reduce-scatter** for correctness)  <br>s: **reduce-scatter** to sharded|
|SP Region|h: full  <br>s: full|h: full  <br>s: sharded|

并且对于嵌入层：

|Region|Vanilla TP|TP with SP|
|---|---|---|
|Embedding Layer (Row Linear sharded on vocab)|h: full (weight_out is full + **all-reduce** for correctness)  <br>s: full|h: full (weight_out is full + **reduce-scatter** for correctness)  <br>s: **reduce-scatter** to sharded|

通过使用序列并行性，我们可以实现更大的激活内存节省，从而使我们能够将批量大小和序列长度推得比仅使用张量并行性时更远。让我们看看这对我们之前的 70B 模型示例意味着什么：

[交互图]

正如我们所见，我们再次大幅降低了每个 GPU 的最大内存使用量，使我们能够在 TP/SP=16 的情况下处理 16k tokens 的序列长度，这比普通 TP 情况有所改进！（如前一节所述，TP=16 仍然有点大，但我们将在下一节中看到如何改进这一点）。

你可能会问自己的一个问题是，使用 TP+SP 是否比普通 TP 产生更多的通信量？嗯，答案是有时是，有时不是。在普通 TP 的前向传播中，每个 Transformer 块有两个 all-reduce 操作，而在 SP 中，每个 Transformer 块有两个 all-gather 和两个 reduce-scatter 操作。所以 SP 的通信操作数量是 TP 的两倍。但由于一个 all-reduce 操作可以分解为一个 all-gather + reduce-scatter（见附录中的“快速关注 Ring AllReduce”部分），它们在通信方面实际上是等效的。后向传播的推理也相同，因为我们只是使用每个操作的共轭（无操作 ↔ allreduce 和 allgather ↔ reducescatter）。

如果你一直密切关注，就会注意到我们正在讨论每层的 4 个通信操作（注意力机制的 2 个和多层感知机的 2 个）。使用张量+序列并行时，多层感知机的分析情况如下：

![tp_sp_overlap.svg|600](https://nanotron-ultrascale-playbook.static.hf.space/assets/images/tp_sp_overlap.svg)

就像普通的张量并行（TP）一样，张量并行（TP）+流水线并行（SP）也不能轻易地与计算重叠，这使得吞吐量严重依赖于通信带宽。同样，在这里，就像普通的张量并行（TO）一样，张量并行（TP）+流水线并行（SP）通常也只在单个节点内进行（使张量并行度保持在每个节点的GPU数量之下，例如张量并行度 ≤ 8 ）。

我们可以衡量随着张量并行性扩展，这种通信开销变得日益棘手的情况。让我们在针对序列长度为 4096 的 3B 参数模型，将张量并行（TP）与流水线并行（SP）一起扩展时，测量吞吐量和内存利用率：

[交互图]

在这里，计算效率（左）和内存容量（右）之间又存在一种权衡。虽然更高的并行度通过减少激活内存能够处理显著更大的批量大小，但它们也降低了每个 GPU 的吞吐量，特别是在对应于每个节点的 GPU 数量的阈值以上。

让我们总结一下我们的观察结果：

- 对于这两种方法，我们注意到当从 TP=8 切换到 TP=16 时，性能下降最为明显，因为此时是从仅在单个节点内通信（NVLink），转变为节点间通信（EFA）。
- 在使用 TP 与 SP 时，激活内存的节省使我们能够处理比仅使用 TP 时大得多的批次。

*我们已经看到 TP 如何通过在隐藏维度上拆分注意力和前馈操作来帮助我们在多个 GPU 上分片激活，以及 SP 如何通过在序列维度上拆分来自然地补充其余操作。*

> [!NOTE]
> 由于 SP 区域中的层归一化（LayerNorms）对序列的不同部分进行操作，因此它们在张量并行（TP）ranks 上的梯度会有所不同。为了确保权重保持同步，我们需要在后向传播期间对它们的梯度进行 all-reduce 操作，类似于数据并行（DP）确保权重保持同步的方式。然而，这是一种较小的通信开销，因为层归一化的参数相对较少。

然而，TP 和 SP 有两个限制：1）如果我们扩展序列长度，激活内存仍然会在 TP 区域爆炸式增长；2）如果模型太大而无法适应 TP=8，那么由于节点间连接问题，我们将看到巨大的减速。

我们可以用上下文并行性来解决问题 1)，用流水线并行性来解决问题 2)。让我们先来看看上下文并行性！

## 五、上下文并行（CP）

通过张量并行和序列并行，我们可以显著降低每个 GPU 的内存需求，因为模型权重和激活都分布在多个 GPU 上。然而，当在越来越长的序列上训练模型时（例如，当扩展到每个序列 128k 或更多 tokens 时），由于在 TP 区域内我们仍然需要处理完整的序列长度，我们可能仍然会超出单个节点上的可用内存。

此外，即使我们使用完全重新计算激活值（这会带来约 30% 的巨大计算开销），我们仍然需要在层边界处保留一些与序列长度线性相关的激活值在内存中。让我们来看一下上下文并行性如何帮助我们：

[交互图]

上下文并行性的核心思想是将类似的思路应用于序列并行性方法（即沿序列长度进行拆分），但要应用于我们已经应用张量并行性的模块。因此，我们将沿两个维度对这些模块进行拆分，从而减少序列长度的影响。在我们已经讨论过的内容之后，你会发现这种方法相当直观……但这其中有个技巧，所以要保持清醒！

为了实现上下文并行性；就像序列并行性一样，我们将沿着序列维度拆分输入，但现在我们将这种拆分应用于整个模型，而不仅仅是我们之前在张量+序列并行性中所做的模型的序列并行区域。

拆分序列不会影响大多数模块，如 MLP 和 LayerNorm，其中每个标记都是独立处理的。它也不需要像TP那样进行昂贵的通信，因为只拆分了输入而不是权重矩阵。就像数据并行一样，在计算梯度后，会启动一个 all-reduce 操作来同步上下文并行组中的梯度。

不过有一个重要的例外情况，那就是我们需要特别关注注意力块（哈哈，双关语😀）。在注意力模块中，每个标记都需要访问所有其他序列标记的键/值对；在因果注意力的情况下，至少要对每个前面的标记予以关注。

由于上下文并行性会沿序列维度将输入拆分到多个 GPU 上，因此注意力模块将需要在 GPU 之间进行完整的通信，以交换必要的键/值数据。

“如果我们天真地去做这件事，那听起来成本非常高。有没有一种既能高效又能快速完成的方法呢！值得庆幸的是，有这样的方法：一种用于高效处理键值对通信的核心技术叫做*环形注意力（Ring Attention）*。”

> [!NOTE]
> 上下文并行性与 Flash Attention（稍后详述）在概念上有一些相似之处——这两种技术都依赖于在线 softmax 计算来减少内存使用。虽然 Flash Attention 专注于优化单个 GPU 上的注意力计算本身，但上下文并行性通过将序列分布到多个 GPU 上来实现内存减少。


### 5.1 发现环形注意力机制

在这种注意力机制的实现中，每个 GPU 首先发起一个异步通信操作，将其键/值对发送到其他 GPU。在等待其他 GPU 的数据时，它会计算其内存中已有数据部分的注意力分数。理想情况下，在本次计算完成之前，会从另一个 GPU 接收到下一个键/值对，从而使 GPU 在完成第一轮计算后能够立即开始下一轮计算。

让我们来说明一下。假设我们有 4 个 GPU 和 4 个输入标记。最初，输入序列会沿着序列维度均匀拆分，因此每个 GPU 将仅有一个标记以及其对应的 Q/K/V 值。假设 Q1、K1 和 V1 分别代表位于第 1 个 GPU 上的第一个标记的查询、键和值。注意力计算将需要 4 个时间步来完成。在每个时间步，每个 GPU 执行这三个连续的操作：

1. 以非阻塞方式将“当前的键和值”发送到下一台机器，但在最后一个时间步除外，这样我们就可以在这一步完成之前开始下一步。
2. 在本地计算其已有的“当前的键和值”上的注意力分数，这通常涉及执行 $\text{Softmax}\left(\frac{QK^T}{\sqrt{d}}\right) \times V$。
3. 等待接收来自前一个 GPU 的键和值，然后循环回到步骤 1。此时“当前的键和值”现在是刚刚从前一个 GPU 接收到的键/值。

我们执行这 3 个步骤四次以完成注意力计算。

以下动画展示了使用 4 个 GPU 的整个过程：
![ring-attention.gif|500](https://nanotron-ultrascale-playbook.static.hf.space/assets/images/ring-attention.gif)
在这个动画中，作者选择将这种方法称为环形注意力机制（Ring Attention），这一点对您来说可能显而易见。

不过有一个大问题是，环形注意力（Ring Attention）的简单实现会由于因果注意力矩阵的形状导致 GPU 之间出现严重的不平衡。让我们通过考虑带有因果注意力掩码的注意力得分矩阵来看看 SoftMax 计算。
![cp_attnmask.svg|500](https://nanotron-ultrascale-playbook.static.hf.space/assets/images/cp_attnmask.svg)

SoftMax 是按行计算的，这意味着只要 GPU 接收到一行的所有 tokens，就可以进行计算。我们看到 GPU1 可以立即计算它，因为它从 token 1 - 4开始，并且 GPU1 实际上不需要从任何其他 GPU 接收任何信息。然而，GPU2 将需要等待第二轮才能也接收到 1-4，从而拥有 tokens 1-8 的所有值。此外，GPU1 似乎比所有其他 GPU 执行的工作量要少得多。

让我们看看能否更好地平衡我们的计算。

### 5.2 锯齿（之字形）环注意力——一种平衡的计算实现

我们需要一种更好的方法来分配输入序列。这可以通过不纯粹按顺序将 tokens 分配给 GPU，而是稍微混合一下顺序来实现，这样每个 GPU 上都有早期和晚期 tokens 的良好混合。这种方法称为锯齿形（Z 字形）注意力，在这种新的排列中，注意力掩码将显示计算量的均匀分布，但如果你数一数有颜色的方块数量，就会发现计算量现在在所有 GPU 之间是平衡的。

![cp_zigzagmask.svg|500](https://nanotron-ultrascale-playbook.static.hf.space/assets/images/cp_zigzagmask.svg)

与此同时，我们还会看到，为了完成所有行，每个 GPU 都需要来自其他所有 GPU 的信息。

我们有两种通用的方法来重叠计算和通信，要么通过执行常规的 all-gather 操作，同时在每个 GPU 上重新分组所有的键值对（以 Zero-3 类型的方式），要么根据需要从每个 GPU 逐个收集到其他 GPU。

![cp_overlap_allgather.svg|500](https://nanotron-ultrascale-playbook.static.hf.space/assets/images/cp_overlap_allgather.svg)

![cp_overlap_all2all.svg|500](https://nanotron-ultrascale-playbook.static.hf.space/assets/images/cp_overlap_all2all.svg)

这两种实现方式的关键区别在于它们的通信模式和内存使用情况：

1. **AllGather实现**：
    - 所有 GPU 同时从所有其他 GPU 收集完整的键/值对
    - 由于每个 GPU 需要一次性存储完整的 KV 对，因此需要更多的临时内存
    - 通信在一步内完成，但具有更大的内存开销
2. **全互联（环形）实现**：
    - GPU 以环形模式一次交换一个数据块的 KV 对
    - 更节省内存，因为每个 GPU 只需要临时存储一个额外的数据块
    - 通信分散并与计算重叠，尽管由于多个通信步骤而有一些额外的基本延迟开销

全互联（All-to-All）方法通常以稍微复杂的通信模式为代价提供更好的内存效率，而全收集（AllGather）方法更简单，但在注意力计算期间需要更多的临时内存。

我们现在看到了如何通过在一个节点上使用张量并行（TP）来拆分模型以应对大型模型，以及如何使用序列并行（CP）来应对长序列中的激活爆炸问题。

然而，我们仍然知道TP 在跨节点扩展方面表现不佳，那么如果模型权重不容易放在 1 个节点上，我们该怎么办呢？另一种并行度，我们的第四种，*并行流水线（Pipeline Parallelism）* 来拯救了！

## 六、流水线并行（PP）

在“张量并行”部分，我们看到，尝试将张量并行扩展到单个节点的 GPU 数量（通常为 4 或 8）以上时，会遇到一种带宽较低的网络，称为“节点间连接”，这可能会严重损害我们的性能。例如，当我们在跨多个节点（每个节点有 8 个 GPU）的集群上对其进行基准测试时，我们可以清楚地看到这一点，比如 all-reduce 操作。

[交互图]

不同节点数量下节点间通信带宽的测量结果，展示了AllReduce、AllGather 和 ReduceScatter 操作的中位数（线条）以及第 5 至第 95 百分位范围（阴影区域）。

序列并行和上下文并行对于长序列可能有帮助，但如果序列长度并非我们内存问题的根本原因，而是模型本身的大小导致的，那么这两种并行方式作用就不大了。对于大型模型（70B 参数及以上），仅权重的大小就可能超过单个节点上 4 到 8 块 GPU 的极限。我们可以通过引入第四个（也是最后一个）并行维度——“流水线并行”来解决这个问题 。

流水线并行是一种简单但强大的技术 —— 我们将模型的层分布到多个 GPU 上！例如，如果我们有 8 个 GPU，我们可以将第 1-4 层放在 GPU 1 上，第 5-8 层放在 GPU 2 上，依此类推。这样，每个 GPU 只需要存储和处理模型层的一部分，显著减少了每个 GPU 的内存需求。让我们看看流水线并行在 8B 模型的内存使用上的效果：

（这种技术可能会让你想起我们在讨论 ZeRO-3 时的情况，当时我们将模型参数分割到多个 GPU 上。在后面的 “5D 并行简介” 部分，我们会详细对比这两种技术。）

[交互图]

观察上图，我们注意到一个有趣的现象：虽然模型参数在各个 GPU 上分配得很好，但每个 GPU 上的激活内存却保持不变！这是因为每个 GPU 仍然需要处理完整的数据批次，只是处理不同的层。一个 GPU 层的激活将被发送到下一个 GPU 以继续前向传播。

这引入了一种新的通信模式：我们现在不是像在数据并行中使用 ZeRO-3 那样通信参数，而是在 GPU 之间按顺序“流水线式”传递激活张量。虽然概念上很简单，但高效实现这种技术相当棘手。让我们直接深入细节！

### 6.1 在不同节点上拆分层 - 全部前向，全部后向

所以，假设我们只是将各层分布到几个设备上，例如，第一个 GPU 将处理前几层，第二个 GPU 将处理模型的第二部分，依此类推。现在，通过我们模型的前向传播简单地涉及按顺序将数据批次沿模型传递，从而连续使用每个计算设备。

我们有一个直接的首要优势：所需的互连带宽保持相当低，因为我们在模型深度的少数位置仅发送中等大小的激活。与例如张量并行中的通信相比，这可以产生巨大差异，后者在每层内会发生数次。

但也许你开始感觉到一些即将到来的麻烦： **“sequentially”** 和 **“successively”**？！？在并行计算的世界里，这听起来并不高效，特别是在我们讨论了计算和通信重叠之后。

确实，读者朋友们！流水线并行中的主要挑战在于如何有效地规避流水线并行的顺序性，使我们的 GPU 始终保持忙碌状态，避免出现一个 GPU 在计算而其他 GPU 在等待的情况。下面展示的是我们在对模型进行简单直接的向前和向后传播时 GPU 的利用率情况（此处数字表示模型的各层）：

![image.png|600](https://nanotron-ultrascale-playbook.static.hf.space/assets/images/pp_afab.svg)

*一个在 4 个 GPU 上分布有 16 层的模型的流水线并行示例。数字对应层 ID。*

剩余的空闲时间以灰色表示，通常称为“气泡”，在花费了这么多时间优化吞吐量之后，看到这个可能会让你心碎。

我们可以通过观察因“气泡”而损失的时间来量化流水线设置的效率。设 $t_f$ 和 $t_b$ 分别为前向和后向传播的时间（针对一个微批次和流水线的一个阶段测量）。一个简单的假设是 $t_b \approx 2 \times t_f$，如上图所示。如果我们能够完美并行化，理想的总时间将是 $t_{\text{id}} = t_f + t_b$。然而，由于流水线气泡，额外的时间为 $t_{\text{pb}} = (p-1) \times (t_f + t_b)$，其中 $p$ 是流水线并行度，即上图中的 GPU 数量。这表示每个 GPU 在其他 GPU 计算时的等待时间。

我们可以计算额外气泡时间相对于理想时间的比率：$$r_{\text{bubble}} = \frac{(p - 1) \times (t_f + t_b)}{t_f + t_b} = p - 1$$
随着我们增加更多阶段，气泡时间因此增加，利用率下降。正如我们所看到的，在简单实现中，气泡可能会非常大！

值得庆幸的是，人们已经设计出了各种流水线并行方案来*减小气泡规模*。

让我们从工具箱中取出第一个工具，思考将我们的批次分割成更小的、可以并行或几乎并行处理的小块（部分），就像我们之前在数据并行中所做的那样。现在，当第二个 GPU 忙于处理微批次1时，第一个 GPU 已经可以开始处理微批次 2 了。以下是使用 8 个微批次的调度安排：

![pp_afab2.svg|650](https://nanotron-ultrascale-playbook.static.hf.space/assets/images/pp_afab2.svg)

（在之前的图表中，数字表示的是层，但从现在起（包括本图）的所有流水线并行图中，数字表示的是微批次。你可以将这里的每个方格看作包含若干层，就像前一幅图中所展示的那样。）

上述调度方式被称为***全前向全后向（AFAB）*** 调度，因为我们首先进行所有前向传播，然后再只进行所有后向传播。其优势在于前向步骤和后向步骤总体上仍然是顺序执行的，因此我们保留了模型训练代码的一般组织结构。这使得这种流水线并行（PP）实现成为最容易实现的方式之一 。

你可以在 picotron 中找到 AFAB pipeline 的完整实现。

👉 AFAB PP 在 Picotron 中的 PP 实现（点击展开）

```python
def train_step_pipeline_afab(model, data_loader, tensor_shapes, device, dtype):
    logging_loss: torch.float32 = 0.0
    input_tensors, output_tensors = [], []
    requires_grad_sync = pgm.process_group_manager.cp_dp_world_size > 1

    for _ in range(data_loader.grad_acc_steps): # All forward passes
        input_tensor = pipeline_communicate(operation='recv_forward', shapes=tensor_shapes, device=device, dtype=dtype)
        batch = next(data_loader)
        batch["hidden_states"] = input_tensor.to(device) if input_tensor is not None else input_tensor
        output_tensor = model.forward(input_ids=batch["input_ids"].to(device), position_ids=batch["position_ids"].to(device), hidden_states=batch["hidden_states"])
        pipeline_communicate(operation='send_forward', tensor=output_tensor, device=device, dtype=dtype)
        
        # calculate loss on the last stage
        if pgm.process_group_manager.pp_is_last_stage:
            output_tensor = F.cross_entropy(output_tensor.transpose(1, 2), batch["target_ids"].to(device), reduction='mean')
            logging_loss += output_tensor.item() / data_loader.grad_acc_steps

        input_tensors.append(input_tensor)
        output_tensors.append(output_tensor)

    for ith_microbatch in range(data_loader.grad_acc_steps): # All backward passes
        if requires_grad_sync:
            is_last_iteration = (ith_microbatch == data_loader.grad_acc_steps - 1)
            model.require_backward_grad_sync = is_last_iteration
        output_tensor_grad = pipeline_communicate(operation='recv_backward', shapes=tensor_shapes, device=device, dtype=dtype)
        input_tensor, output_tensor = input_tensors.pop(0), output_tensors.pop(0)
        input_tensor_grad = model.backward(input_tensor, output_tensor, output_tensor_grad)
        pipeline_communicate(operation='send_backward', tensor=input_tensor_grad, device=device, dtype=dtype)

    return logging_loss
```

让我们在这个例子中估算一下气泡。与我们的第一个例子不同的是，现在处理 $m$ 个小批量的理想时间是 $t_{id}=m \times (t_f+t_b)$：$$r_{\text{bubble}} = \frac{(p - 1) \times (t_f + t_b)}{m \times (t_f + t_b)} = \frac{p - 1}{m}$$
正如我们所见，通过增加更多的微批次，我们可以将气泡的大小缩小 $m$ 倍，从而解决流水线阶段的一些低效率问题。

然而，与气泡同样烦人的是存储所有激活所需的存储空间。在达到反向阶段之前，我们需要将所有激活保存在内存中，这导致这些 PP 实现中的内存迅速爆炸。我们能做得更好，避免这种内存爆炸吗？

由于内存爆炸是由我们为反向传播存储的激活触发的，让我们尝试看看是否可以在我们仍在进行计算的前向部分时就开始执行反向传播。这将使我们能够尽快丢弃一些用于反向传播所需的激活。

### 6.2 “向前一步-向后一步”及 LLama 3.1 方案

这种调度被称为*一前一后（1F1B）*，因为中间/稳定状态涉及交替进行一次前向传播和一次反向传播。其总体思路是尽早开始进行反向传播。调度过程如下：
![image.png|650](https://nanotron-ultrascale-playbook.static.hf.space/assets/images/pp_1f1b.svg)

如果你仔细计算就会发现气泡大小仍然相同，因此我们的训练效率并没有显著提高。然而，我们只需存储 $p$ 个微批次的激活值（其中 $p$ 是流水线并行度），而不是 $m$ 个（其中 $m$ 是微批次的数量），这样可以减少 AFAB（假设为某种调度方式，具体需结合上下文确定准确含义）调度中出现的激活值内存爆炸问题。因此，我们可以增加更多微批次，而这实际上会减少气泡。

这种设置的复杂性（如上图所示）在于，前向传播和反向传播不再是清晰的顺序执行，而是在设备间并行执行并交错进行。这意味着我们将不得不在每个设备上独立地安排从前向传播到反向传播的切换，而不是像往常一样在一个简单且通用的中央训练循环中进行。

这就是实施流水线并行通常需要对训练代码以及建模代码进行相当广泛的修改的原因之一。

你也可以在 picotron 中找到 1F1B 的完整实现：

👉 1F1B PP 在 Picotron 中的实现（点击展开）

```python
def train_step_pipeline_1f1b(model, data_loader, tensor_shapes, device, dtype):    
    num_warmup_microbatches = min(pgm.process_group_manager.pp_world_size - pgm.process_group_manager.pp_rank - 1, data_loader.grad_acc_steps)
    num_microbatches_remaining = data_loader.grad_acc_steps - num_warmup_microbatches
    logging_loss, input_tensors, output_tensors  = 0.0, [], []
    requires_grad_sync = pgm.process_group_manager.cp_dp_world_size > 1
    
    def _forward_step(input_tensor):
        batch = next(data_loader)
        batch["hidden_states"] = input_tensor.to(device) if input_tensor is not None else input_tensor
        output_tensor = model.forward(input_ids=batch["input_ids"].to(device), position_ids=batch["position_ids"].to(device), hidden_states=batch["hidden_states"])
        
        # calculate loss on the last stage
        if pgm.process_group_manager.pp_is_last_stage:
            output_tensor = F.cross_entropy(output_tensor.transpose(1, 2), batch["target_ids"].to(device), reduction='mean')
            nonlocal logging_loss
            logging_loss += output_tensor.item() / data_loader.grad_acc_steps
        return output_tensor

    for _ in range(num_warmup_microbatches): # Warmup forward passes
        input_tensor = pipeline_communicate(operation='recv_forward', shapes=tensor_shapes, device=device, dtype=dtype)
        output_tensor = _forward_step(input_tensor)
        pipeline_communicate(operation='send_forward', tensor=output_tensor, device=device, dtype=dtype)
        input_tensors.append(input_tensor)
        output_tensors.append(output_tensor)

    if num_microbatches_remaining > 0:
        input_tensor = pipeline_communicate(operation='recv_forward', shapes=tensor_shapes, device=device, dtype=dtype)
    
    if requires_grad_sync:
        model.require_backward_grad_sync = False

    for ith_microbatch in range(num_microbatches_remaining):  # 1F1B steady state
        is_last_iteration = (ith_microbatch == num_microbatches_remaining - 1)
        output_tensor = _forward_step(input_tensor)
        output_tensor_grad = bidirectional_pipeline_communicate(operation='send_fwd_recv_bwd', send_tensor=output_tensor, recv_shapes=tensor_shapes, device=device, dtype=dtype)
        input_tensors.append(input_tensor)
        output_tensors.append(output_tensor)
        input_tensor, output_tensor = input_tensors.pop(0), output_tensors.pop(0)
        
        # Trigger gradient sync on the last microbatch but only when last rank (the one that has num_warmup_microbatches = 0) has finished computing its backward pass.
        if num_warmup_microbatches == 0 and is_last_iteration:
            model.require_backward_grad_sync = True

        input_tensor_grad = model.backward(input_tensor, output_tensor, output_tensor_grad)
        
        if is_last_iteration:
            input_tensor = None
            pipeline_communicate(operation='send_backward', tensor=input_tensor_grad, device=device, dtype=dtype)
        else:
            input_tensor = bidirectional_pipeline_communicate(operation='send_bwd_recv_fwd', send_tensor=input_tensor_grad, recv_shapes=tensor_shapes, device=device, dtype=dtype)

    for ith_warmup_microbatches in range(num_warmup_microbatches): # Cooldown backward passes
        if requires_grad_sync:
            is_last_iteration = (ith_warmup_microbatches == num_warmup_microbatches - 1)
            model.require_backward_grad_sync = (ith_warmup_microbatches == num_warmup_microbatches - 1)
        input_tensor, output_tensor = input_tensors.pop(0), output_tensors.pop(0)
        output_tensor_grad = pipeline_communicate(operation='recv_backward', shapes=tensor_shapes, device=device, dtype=dtype)
        input_tensor_grad = model.backward(input_tensor, output_tensor, output_tensor_grad)
        pipeline_communicate(operation='send_backward', tensor=input_tensor_grad, device=device, dtype=dtype)

    return logging_loss
```

让我们通过在我们的集群上进行的一些基准测试，来看一下 1F1B 流水线并行调度在实践中是如何扩展的：

![Throughput scaling of Pipeline Parallelism with varying microbatch sizes|600](https://nanotron-ultrascale-playbook.static.hf.space/assets/images/pp_1f1b_scaling.png)

在左侧，微批处理数量等于或小于流水线并行（PP）度减一（$m = p - 1$）时，我们可以看到流水线气泡的危害有多大——性能很低，而且随着流水线并行度的增加甚至还会下降。右侧图表显示，使用远多于流水线并行度的微批处理数量（$m = 32 \gg p - 1$）有助于改善低流水线并行度下的性能，但在非常大的流水线并行度下仍然受限。实际上，由于最终受到目标全局批量大小的限制，我们无法随意增加微批处理数量以保持 $m \gg p - 1$ 的比例。随着流水线并行度的增加，当微批处理数量达到可能的最大值时，我们最终必须根据 $r_\text{bubble}=\frac{p−1}{m}$ 来增加气泡大小 。

有趣的是，在微批次数量较少时，从单个节点（$p = 8$）扩展到两个节点（$p = 16$），性能仅下降了 14% ——这比张量并行性在类似的跨节点场景中通常出现的约 43% 的性能退化要好得多。当遇到节点间低带宽网络时，这种行为使得流水线并行性在跨多个节点的分布式训练中特别有吸引力。

虽然 1F1B 显著减少了我们的激活内存占用，但从这最后一张图中我们可以看到，流水线气泡仍然是一个主要的效率瓶颈。由于气泡大小仍与流水线阶段数成正比，我们让宝贵的 GPU 计算能力处于闲置状态。我们能否设计出一种更巧妙的调度方案来尽量减少这种浪费的计算时间呢？

### 6.3 交错阶段

1F1B 调度让我们改善了内存使用情况，但对空闲包的大小改善不大。无论如何，我们还能推进这一边界吗？

原来，如果我们愿意引入一些额外的通信操作，这是可行的。现在是时候谈谈*交错阶段*了。

到目前为止，我们一直沿着模型深度维度天真地对模型进行切片处理，例如将第 1-4 层放在第一个 GPU 上，将第 5-8 层放在第二个 GPU 上。但是，我们还可以考虑其他对层进行切片的方式，比如将奇数层（第 1、3、5、7 层）放在第一个 GPU 上，将偶数层（第 2、4、6、8 层）放在第二个 GPU 上。

这通常可以被看作是一种“环形管道”，微批次在通过模型的前向传播时会从一个 GPU 循环移动到下一个 GPU。让我们通过图形来看看这是如何工作的：

![pp_1f1b_interleaved.svg|650](https://nanotron-ultrascale-playbook.static.hf.space/assets/images/pp_1f1b_interleaved.svg)

*一个在 4 个 GPU 上分布各层的模型的交错流水线并行的示例。编号仍对应微批处理 ID，但为了清晰起见，我们对模型的第一层和最后一层进行了不同着色，以说明各层是如何分布在 GPU 上的。*

因此，我们看到由于模型针对同一计算在每个 GPU 上要经过多次（而此前只需一次），从而出现了额外的通信情况。不过，每次前向传播和反向传播都被一个因子 $v$ 所分摊，其中 $v$ 是阶段数或者每个 GPU 的模型块数，因为我们现在能够更好地交错进行前向传播和反向传播。$$\begin{align}
t_{pb} &= \frac{(p - 1) \times (t_f + t_b)}{v} \\[1.2ex]
r_{\text{bubble}} &= \frac{1}{v} \frac{(p - 1) \times (t_f + t_b)}{m \times (t_f + t_b)} = \frac{p - 1}{v \times m}\end{align}$$
因此，我们现在可以通过添加微批次和交错阶段来减小气泡，但请注意，从数量上来说，通信量也会因 $v$ 而增加，所以这是一种权衡。在下面的图中，你可以看到 $p = 8$ 的 PP 设置的几种配置，其中特殊情况 $m = 1, v = 1$ 对应于简单的流水线并行，$v = 1$ 的配置是 AFAB 或 1F1B 设置，而 $v ≠ 1$ 是交错配置。

[交互图]

调度在这里也变得更加复杂，因为我们必须在给定的 GPU 上和给定的时刻决定我们是优先处理先到达的微批次数据，让它们通过后续层（即尽快完成前向和反向传播循环，也就是所谓的“深度优先”，即优先让批次数据尽快离开模型），还是优先处理后续的微批次数据，让它们通过前面的层（即所谓的“广度优先”，即优先尽可能填满流水线）。这种选择在精彩的“广度优先流水线”论文[^6]中有详细解释。

你现在拥有了理解 Llama 3.1 中流水线并行方法的所有要素，该方法采用一次前向传播一次反向传播的设置，各阶段交错进行，并且优先级设置可以在深度优先和广度优先之间进行调整。

![pp_llama3.1_schedule.png|650](https://nanotron-ultrascale-playbook.static.hf.space/assets/images/pp_llama3.1_schedule.png)

然而，我们尚未到达可能的流水线调度方案的尽头，最近已经有一些方法被提出来将气泡减少到几乎为零！例如，这些技术已在 DeepSeek V3/R1 实现[^7]中使用。勾起你的好奇心了吗？在我们离开流水线并行世界之前，让我们最后快速看一眼这些神奇的调度方案！

### 6.4 零气泡和双管

最近提出了更为复杂的方法来减少气泡，这些方法几乎达到了“零气泡”状态。这里的秘诀是在更细粒度的层面上拆分相关操作，以最有效的方式将它们交错执行。例如，DeepSeek V3/R1 中的管道实现方法，称为 DualPipe，几乎达到了零气泡状态。

（DeepSeek V3 技术报告[^7]中的终极“灵活配置”，作者在报告中指出他们的设置“实现了近乎为零的全互联通信开销”。）

让我们简要地通过总结作为 DualPipe 前身的 ZeroBubble[^8] 的工作来看看这是如何运作的。 ZeroBubble 的基本观察结果是，矩阵乘法的反向传播实际上涉及两个分离的操作：输入（B）的反向操作和权重（W）的反向操作。

虽然输入的反向传播（即 B 的输出）对于执行较低层的反向传播是必要的，但权重的反向传播（即 W 的反向传播）对于其余的反向传播过程并非必要，并且通常只需在优化器步骤之前执行。我们可以在下图中看到这一点：

![image.png|600](https://nanotron-ultrascale-playbook.static.hf.space/assets/images/pp_zerobubble_compgraph.png)

这意味着 W 可以灵活地安排在同一阶段的相应 B 之后的任何位置。这使得可以策略性地放置 W 以填充流水线气泡。右上角的 ZB-H2 时间表是利用这种细粒度分解实现零气泡的（理论）时间表示例。

![image.png|650](https://nanotron-ultrascale-playbook.static.hf.space/assets/images/pp_zerobubble_ppschedule.png)

*在顶部（ZeroBubble 论文中的图 2）：经典的 1F1B 调度，交错进行前向和后向传播，但保持粗粒度的后向传播。在下面的两个图中（ZeroBubble 论文中的图 3），ZeroBubble 调度的两个变体，将后向操作拆分为更细粒度的 “B” 和 “W” 操作。最后一个调度，称为 “ZB-H2”，是一个（理论上）利用这种细粒度分解实现零气泡的调度示例。*

DeepSeek 的 DualPipe 在其 V3 技术报告[^7]中介绍了这种分解方法的扩展，即针对从 PP 维度的两端传播的两个流的情况，这些流交错排列，以进一步减少 GPU 中的空闲时间。该调度方案显示在下面的调度图中，比之前的方案更为复杂。

![image.png|650](https://nanotron-ultrascale-playbook.static.hf.space/assets/images/pp_zerobubble_dualpipe.png)

一般来说，要完全优化这种复杂的调度安排，需要仔细测量各种细粒度操作的持续时间，并求解一个整数线性规划（ILP）问题以使最终的气泡时间最小化。例如，可参考 ZeroBubble 论文[^8]中对执行此类调度所采用的启发式算法和算法的讨论。因此，ZeroBubble 和 DualPipe 的调度安排过于复杂，在此无法给出代码片段，但你应该开始对所涉及的概念有一个大致的了解。

这就结束了我们对管道调度和气泡世界的参观。希望您喜欢这次导览之旅！

现在是时候探讨我们将详细介绍的最后一种并行方法了，即我们可以用它来高效训练大型模型的方法：*专家并行*。

## 七、专家并行（EP）

这是我们要讨论的最后一个并行方法。在探讨它之前，如果您对专家混合模型（Mixture-of-Experts）没有任何了解，欢迎阅读我们之前发布的一篇较短的博客文章，这篇文章应该能帮助您更好地理解专家混合模型（MoE）架构。

专家混合模型近期因 GPT - 4、Mixtral[^9] ，以及更近期的 DeepSeek-V3/R1 等模型而受到关注并崭露头角。其基本思想是，我们可以在每一层设置多个并行模块，而非单个前馈模块，并将令牌通过其中一个或另一个模块进行路由，以进行不同的处理。

![ep_moe.png|600](https://nanotron-ultrascale-playbook.static.hf.space/assets/images/ep_moe.png)
*取自 Switch Transformers 论文[^10]的 MoE 层示意图*

MoE 层的设计实际上使得跨专家维度实现并行性变得非常容易，我们将其称为专家并行性（EP）。由于前馈层是完全独立的，我们可以简单地将每个专家的前馈层放在不同的工作器上。与 TP 相比，它要轻量得多，因为我们不需要拆分矩阵乘法，只需要将 tokens 的隐藏状态路由到正确的专家即可。

在实际应用中，专家并行（EP）通常会与其他并行形式（例如数据并行）结合使用。这是因为专家并行仅影响混合专家（MoE）层，并且不会对输入标记进行分片（不像上下文并行那样沿序列长度维度对标记进行分片）。这意味着，如果我们仅使用专家并行，我们的图形处理单元（GPU）将对所有非 MoE 块执行冗余计算。通过将专家并行与数据并行相结合，我们可以像下面的简化示意图中所示，有效地在 GPU 之间对专家和输入批次进行分片 。

![ep_schema.png|650](https://nanotron-ultrascale-playbook.static.hf.space/assets/images/ep_schema.png)
*来源：《专家混合模型综述》[^11]*

但先别高兴得太早——在接下来的部分中，我们将具体探讨不同并行策略之间的所有交互，所以如果你还没理解这最后一张图，也别担心。

在实际应用中，要使专家并行（EP）高效运行有一些技巧，并且这些技巧与模型设计密切相关。例如，DeepSeek-V3 在路由器中施加了一种约束，确保每个 token 最多被发送到 M 个节点（在他们的案例中，M 为 4），以将 token 保留在单个节点上并减少通信开销。虽然专家并行已经存在一段时间了[^12]，但随着混合专家（MoE）架构越来越受到关注，它现在才重新受到重视 。

我们计划在 picotron/nanotron 中尽快添加一个更完整的 EP 示例，敬请关注更多内容！

## 八、5D 并行性概述

恭喜读者，您现在已经了解了可用于扩展模型训练的所有 5 种并行策略：

1. DP ——沿批次维度
2. TP ——沿隐藏维度
3. SP/CP ——沿序列维度
4. PP ——沿模型层
5. EP ——沿模型专家

以及 3 种 ZeRO 策略，这些策略可与数据并行性相结合以减少内存占用：

1. ZeRO-1 – 在 DP 副本间对优化器状态进行分片
2. ZeRO-2 – 在 DP 副本间对优化器状态和梯度进行分片
3. ZeRO-3 – 在 DP 副本间对优化器状态、梯度和参数进行分片

在这个阶段，你可能好奇的一个方面是，所有这些并行策略和 ZeRO 策略如何相互比较和交互。换句话说，我们应该使用哪些策略并将它们有效地组合在一起，哪些策略我们应该保持分开？

让我们来看看相似之处以及相互作用。我们将首先对比流水线并行和 ZeRO-3，因为它们有一些非常相似的地方，但也有重要的区别。

*流水线并行与 ZeRO-3*——流水线并行（PP）和 ZeRO-3 都是在多个 GPU 上划分模型权重，并沿着模型深度轴进行通信/计算的方法（例如在 ZeRO-3 中，我们在计算的同时预取下一层）。这意味着在这两种情况下，每个设备上都计算完整的层操作，而不是像张量并行（TP）或专家并行（EP）那样在子层单元上执行计算。

（以下我们将“一层”简称为“一层”（一般应称为“一组层”，作为模型的基础分片单元）。）

然而，PP 方法与 ZeRO-3 方法之间存在几个主要差异：

|                           | ZeRO-3                          | 管道并行                         |
|---------------------------|---------------------------------|---------------------------------|
| 每个计算单元存储               | 仅存储一层的一部分                  | 整层                             |
| 用于传输的通信                 | 权重                             | 激活值                           |
| 编排                       | 模型无关                          | 模型无关                         |
| 实现挑战                    | 处理模型划分和通信较复杂             | 处理高效管道并行调度较复杂             |
| 扩展考虑                    | 偏好较大的 `mbs` 和 `seq_len` 隐藏通信 | 偏好较大的 `grad_acc` 隐藏气泡     |

如你所见，ZeRO-3 和 PP解决了相同的挑战，但涉及不同的方法，选择两者中的哪一个将取决于你是决定将通信重点放在权重上还是激活值上。虽然它们可以结合使用，但在实践中并不经常这样做，因为这样做需要显著增加全局批量大小以分摊通信成本，从而在全局批量大小、模型大小、网络带宽和训练效率之间进行权衡。如果你决定将它们结合起来，应配置 ZeRO-3 在一系列 PP 微批次期间将权重保留在内存中，以尽可能减少不必要的通信开销。

另一方面，专注于优化器状态和梯度的 ZeRO-1 和 ZeRO-2 可以很容易地与流水线并行相结合，并且与之互补。将它们结合起来不会引发任何特别新的挑战。例如，DeepSeek-v3 的训练就使用了流水线并行结合 ZeRO-1（原文如此）。

*张量并行*（与序列并行）本质上是互补的，它可以与流水线并行和 ZeRO-3 结合使用，因为它依赖于矩阵乘法的分配属性，这使得权重和激活可以在组合之前被分片并独立计算。

![TP & SP diagram](https://nanotron-ultrascale-playbook.static.hf.space/assets/images/5d_nutshell_tp_sp.svg)

我们不想仅出于并行性而使用张量并行（TP）的主要原因是，在实践中，正如前文所述，张量并行存在两个限制：首先，由于其通信操作是计算关键路径的一部分，因此很难在通信开销开始占据主导地位的某个点之后实现良好扩展。其次，与对模型没有要求的ZeRO和流水线并行（PP）不同，张量并行需要仔细处理激活分片——有时是在张量并行区域内沿隐藏维度进行，有时是在流水线并行区域内沿序列维度进行——这使得正确实现变得更加繁琐，并且需要特定于模型的知识来确保整个过程中分片模式正确。

因此，在结合并行策略时，通常会为高速的节点内通信保留 TP，而 ZeRO-3 或 PP 可用于跨较低速度节点间通信的并行组，因为它们的通信模式需要的带宽较少（对于 PP）或可以更容易地与计算重叠（对于 ZeRO-3）。结合这些技术时的主要考虑因素是有效地将 GPU 组织到每个并行维度的组中，以最大化吞吐量并最小化通信开销，同时要注意 TP 的扩展限制。例如，为 TP 进行通信的 GPU 组应保留在节点内。

*上下文并行* 和 *专家并行* 也有助于我们对激活值进行分片处理，并且可以看作是对张量并行的补充。前者处理长序列，而后者支持分布式专家混合训练，而且它们可以组合在一起，不会产生任何特定问题。

*上下文并行（CP）* 专门针对在非常长的序列上进行训练的挑战，通过沿序列维度在 GPU 之间对激活进行分片来处理。虽然大多数操作（如多层感知机（MLP）和层归一化（LayerNorm））可以独立处理这些分片序列，但注意力层需要通信，因为每个 token 都需要访问整个序列的键/值。正如我们在 CP 部分中看到的，通过环状注意力模式可以高效地处理这种情况，该模式使计算和通信重叠。当扩展到极端序列长度（128k+ tokens）时，CP 尤其有价值，即使使用完整的激活重新计算，在单个 GPU 上，注意力的内存需求也会高得令人望而却步。

![CP diagram](https://nanotron-ultrascale-playbook.static.hf.space/assets/images/5d_nutshell_cp.svg)

*专家并行（EP）* 专门针对通过在 GPU 之间对专门的“专家”进行分片，并在计算过程中动态地将标记路由到相关专家来训练专家混合（MoE）模型的挑战。EP中的关键通信操作是将标记路由到其分配的专家并收集结果的 `all-to-all` 操作。虽然此操作引入了一些通信开销，但它能够显著扩展模型容量，因为每个标记在推理（和训练）期间仅由总参数的一小部分处理。在分布式训练/推理方面，当模型扩展到大量专家时，跨 GPU 划分专家变得相关。

（例如，DeepSeek V3 使用了 256 个专家。）

![EP diagram](https://nanotron-ultrascale-playbook.static.hf.space/assets/images/5d_nutshell_ep.svg)

> [!NOTE]
> EP 和 DP 在输入处理方面的这种相似性是为什么一些实现将专家并行视为数据并行的一个子集的原因，关键区别在于 EP 使用专门的专家路由，而不是让所有 GPU 通过相同的模型副本处理输入。

*范围和重点*   让我们也快速总结一下模型中的一些不同并行策略影响最大的子部分：

- 张量并行（和序列并行）通过分片权重和激活来影响整个模型的计算。
- 上下文并行主要影响注意力层，因为需要跨序列通信，其他层则在分片序列上独立操作。
- 专家并行主要影响MoE层（取代标准 MLP 块），注意力和其他组件保持不变。
- 流水线并行和 ZeRO 并不特别针对任何子模块或组件，除了流水线并行中模块和层需要平衡外，由于额外的嵌入层，第一层和最后一层通常被区别对待。

| 张量 + 序列并行        | 上下文并行       | 专家并行           |
| ---------------- | ----------- | -------------- |
| 沿隐藏/序列维度分片权重和激活值 | 沿序列维度分片激活值  | 分片专用专家权重和激活值   |
| 矩阵乘法操作的通信（列/行线性） | 注意力键/值的通信   | token 路由到专家的通信 |
| 需要特定模型的实现        | 除了注意力外，模型无关 | 除了 MoE 层外，模型无关 |
| 偏好高带宽节点内通信       | 偏好较长的序列长度   | 需要 MoEs        |

*总结一下*——现在，将我们看到的所有技术整合到一个图表中，并将它们全部结合起来，会怎么样呢？没错，我们愿意接受这个挑战！

在这个总结图表中，你将看到单个 Transformer 层（以其 MoE 变体形式）的激活过程和模块的图示。我们还展示了各种并行方向以及在前面的所有章节中一直在讨论的通信操作。

![image.png](https://nanotron-ultrascale-playbook.static.hf.space/assets/images/5d_full.svg)

我们还可以并列展示每种策略的内存节省情况的*全面概览*。我们将针对不同的序列长度以及选择性（顶部）和完全（底部）重新计算来绘制它们，这样您就可以看到它们是如何与激活一起作用的：

![5Dparallelism_8Bmemoryusage.svg](https://nanotron-ultrascale-playbook.static.hf.space/assets/images/5Dparallelism_8Bmemoryusage.svg)

让我们从宏观角度审视所有这些技术、它们的主要基本思想以及主要瓶颈，以此结束本节内容：

| 方法         | 内存节省具体应用于                        | 并行/分片维度                   | 缺点                                   |
|--------------|---------------------------------|-------------------------------|---------------------------------------|
| DP           | 激活值（减少本地批大小）                 | 批次                          | 受限于最大批大小                         |
| PP           | 模型参数                            | 模型层                         | 空闲气泡和复杂的调度                       |
| TP/SP        | 模型参数和激活值                      | 隐藏维度 / 序列长度             | 需要高带宽通信                            |
| CP           | 激活值                             | 序列长度                       | 在注意力模块中增加通信开销                   |
| EP           | 专家参数                            | 专家维度                       | 需要 MoE 层，增加路由通信开销               |
| ZeRO-1       | 优化器状态                          | 在 DP 副本间分片               | 参数通信开销                              |
| ZeRO-2       | 优化器状态和梯度                      | 在 DP 副本间分片               | 参数通信开销                              |
| ZeRO-3       | 优化器状态、梯度和模型参数               | 在 DP 副本间分片               | 参数通信开销                              |

显然，这些技术中没有哪一种是实现神奇扩展的灵丹妙药，我们通常需要以某种方式将它们结合起来。我们能否制定出一些规则，帮助我们找到一个好的起点，从而在它们之间进行选择并加以结合呢？这将是下一节的主题。

## 九、寻找最佳训练配置

我们现在已经介绍了实际用于分布式训练更大模型的所有并行技术，以及它们如何且为何可以组合在一起。还有一个普遍的问题：最终我们应该选择哪些技术，以及如何确定具体的组合方式？

我们在上一节简单提及了这个内容，现在让我们详细地逐步探讨一个可能的决策过程。要记住，鉴于计算集群的各种物理属性、网络带宽、每个节点的 GPU 数量、每个 GPU 的内存量等因素，你始终需要运行一些实验，才能找到该计算集群的最终最佳配置。

### 9.1 步骤 1：将一个训练步骤适配到内存中

首先，我们需要弄清楚如何使一个完整的模型实例适配我们的 GPU。一般有两种情况。

*GPU 资源丰富的情况 🤑* —— 当你有大量可用的 GPU 时：

- 对于参数量在 10B 以下的模型，你可以使用单一的并行技术，例如在 8 个 GPU 上使用张量并行（Tensor Parallelism）或 ZeRO-3/DP（数据并行）并配合全量重新计算（Full Recompute）
- 对于参数量在 10B 到 100B 之间且需要超过 8 个 GPU 的模型，你有几种选择：
    - 将张量并行（TP = 8）与流水线并行（Pipeline Parallelism）相结合
    - 将张量并行（TP = 8）与数据并行（ZeRO - 3）相结合
    - 仅使用 ZeRO-3（即仅使用纯数据并行）
- 在 512 个及以上 GPU 规模时，由于通信成本的原因，纯数据并行/ZeRO-3将开始变得低效——此时将数据并行（DP）与张量并行或流水线并行相结合可能效果更好
- 在 1024 个及以上 GPU 规模时，一种推荐的设置是张量并行 TP = 8，配合数据并行（ZeRO-2）和流水线并行

目前我们专注于适配单个实例 —— 尽管我们可能会使用 DP（数据并行）来实现 ZeRO（零冗余优化器）以达成这一目标 —— 但在这里我们仅关注它与 ZeRO-3 结合使用时在模型参数内存节省方面所带来的效果 。

特殊考虑事项：

- 对于非常长的序列，您可能需要在节点间添加上下文并行（CP）。
- 对于专家混合架构，跨节点使用专家并行（EP）将更有优势。


*GPU 资源不足的情况😭* ——当你可能缺少 GPU 资源时：

- 你可以启用完全激活重新计算，以牺牲一些计算量来换取内存（这样训练速度会稍慢一些）。
- 你可以增加梯度累积量，以便在有限的内存下处理更大的批次。  

现在我们有了第一个模型实例正在训练，我们需要确保批量大小合适。

### 9.2 步骤 2：实现目标全局批量大小

根据第一步在微批次大小和 DP 方面的情况，我们当前的批次大小可能太小或太大。现在是时候达到我们的目标批次大小了。  

要增加我们当前的全局批次大小：

- 我们可以扩展数据并行性或梯度累积步骤
- 对于长序列，我们可以利用上下文并行性  

要减少我们当前的全局批次大小：

- 我们可以减少数据并行性以支持其他并行化策略
- 对于长序列，我们可以减少上下文并行性  

好的，现在我们已经让模型按照我们想要的模型大小和批次大小的一般配置运行，但我们是否以最快的方式对其进行训练？现在让我们开始尽可能优化吞吐量。

### 9.3 步骤 3: 优化训练吞吐量

所以我们希望确保训练尽可能快速地运行，这样我们所有宝贵的 GPU 就能始终得到充分利用。只要内存和通信不是瓶颈，我们可以尝试以下操作：

- 利用节点内高速带宽扩展张量并行度，直至接近节点规模的程度，这样我们就能减少其他并行度。
- 在保持目标批量大小的同时，使用ZeRO - 3增加数据并行度。
- 当数据并行度的通信开始成为瓶颈时，过渡到使用流水线并行度。
- 尝试逐个扩展不同的并行度。
- 尝试几种微批量大小（mbs），以实现最大GBS、模型规模、计算和通信之间的最佳平衡 。

### 9.4 对数千种配置进行基准测试

既然我们已经讲完了具体步骤，那现在就在实际中实施这一搜索过程吧。

你将在 nanotron 代码库中找到几个脚本，可使用这些脚本来运行我们上述讨论的所有实验，并能够对现实生活中的自有模型和集群进行基准测试。

实际上，我们在 *数千种分布式配置* 上对我们自己进行了基准测试，这些配置涵盖了上述所有模型规模，以及我们能尝试的大量集群配置（即 1 到 64 个节点的 8xH100），以便得出本书到目前为止所涵盖的结果 。

（我们想借此机会就阻塞了大部分科学集群向我们的同事们道歉，并进而原谅可能已经被私下低语的任何威胁。）

现在让我们退一步，收集和分析我们所有基准测试的结果，看看除了理论之外，我们是否能在真实数据上发现各种配置相互之间的表现如何。

所有以下基准测试均在序列长度为 4096 且全局批量大小为 1M tokens 的情况下进行。我们收集了每个模型和集群大小的所有最佳配置，并将它们绘制在以下热图中：

![image.png](https://nanotron-ultrascale-playbook.static.hf.space/assets/images/what_we_learnt_heatmap.svg)

*热图可视化展示了在不同模型规模和计算节点数量（每个节点有8块GPU）下的最优训练配置。对于每种组合，配置细节包括数据并行（DP）、张量并行（TP）、流水线并行（PP）、梯度累积步数（GAS）、微批量大小（MBS）以及ZeRO优化阶段。颜色深浅表示模型FLOPs利用率（MFU），颜色越亮表示效率越高。*

从这个高层次的可视化中，我们可以得出几个重要的见解：

首先，随着我们增加节点数量（提高并行性），我们观察到效率有所下降。这种效应在较小的模型中尤为明显，这些模型的计算与模型大小比率较低。虽然我们可能会通过增加批量大小来补偿小模型大小，但我们受到全局批量大小限制为 1M 的约束。

其次，更大的模型带来了不同的挑战。随着模型规模的增大，内存需求大幅增长。这在节点数量较少的情况下会产生两种情形：要么模型根本无法适配，要么勉强适配，但由于运行时接近 GPU 内存限制而导致效率低下（例如在 4 个节点上训练 80B 参数模型的情况）。

最终，我们的基准测试显示性能在很大程度上取决于实现质量。当我们最初实施这两种并行策略时，张量并行（TP）的性能优于流水线并行（PP）。在优化了我们的 PP 代码之后，它成为了更快的选择。现在我们正在改进 TP 实现中的通信重叠，我们预计它将重新获得性能优势。

### 9.5 基准测试的经验教训

本书的目标不仅是讨论理论和实现，还要提供实际的数据点。因此计划很简单：让我们针对每个模型以及一系列集群规模（即 8xH100 的 1-64 个节点）运行所有可能的分布式配置。即便排除了不可能的配置后，我们仍需进行数千次实验。

从理论上讲，这听起来很容易：我们可以在集群上轻松启动大量作业。然而，当我们启动第一批实验时，问题就出现了：

- PyTorch 进程有时无法正确清理
- Slurm 作业管理器会强制终止作业，导致节点故障
- 本应只需几分钟的简单基准测试可能会延长到数小时
- 有些作业会无限期挂起

在有限的时间内运行所有实验需要额外的工程工作，我们最终在以下事情上花费了大量时间：

- 最小化集群重启时间并优化空闲时间
- 分析详细的 NCCL 调试日志
- 了解内存使用模式和 CUDA 内存分配器行为
- 提升多节点上的流水线并行性能

这些挑战值得单独讲述，但它们让我们深刻认识到分布式训练基础设施的复杂性。理论上看起来简单的东西，在实践中往往需要对许多相互关联的部分给予细致关注 。

在实践中复现理论结果颇具挑战性，尤其是在生产训练代码获取有限的情况下。通过像 nanotron 和 picotron 这样的开源项目，我们希望能够助力分布式训练技术变得更加易于获取，并且围绕简单高效的代码库展开合作，从而帮助研究人员和从业者充分利用他们的硬件资源。

---

至此，我们深入探讨了 5D 并行性的分发方法。

退一步来看，到目前为止我们的讨论常常依赖一个关键假设——在 GPU 上能够高效地将计算和通信重叠进行，且不会对计算吞吐量产生任何影响。但实际情况更为复杂微妙。当使用像 NCCL 的 send/recv 这类常见的通信原语时，由于通信内核通常会使用与计算相同的 GPU 流式多处理器（SM），计算资源和通信资源之间就会存在隐藏的争用情况，从而导致在计算和通信重叠进行时吞吐量下降。为了真正优化我们的分布式训练，我们需要更深入地探究 GPU 架构本身。

此外，在计算和通信重叠时的同步模式可能并不总是适合我们的并行策略。例如，你可以在 Pytorch 团队的[这篇博客文章](https://discuss.pytorch.org/t/distributed-w-torchtitan-introducing-async-tensor-parallelism-in-pytorch/209487)中找到一个例子。

是时候关灯并启动 CUDA 模式了！

## 十、深入GPU——融合、线程处理、混合

到目前为止，我们的讨论主要集中在模型操作的高层组织上。我们在各种加速器上移动计算，同时考虑到一般的内存限制和计算单元的高层调度。

但这忽略了我们通过仔细了解模型操作在每个 GPU 上的调度和执行方式，在更低层次上所能进行的所有优化。

本节将更深入地探讨 GPU 架构的诸多细节，特别是 NVIDIA 的 GPU 架构，但通常来说，其总体思路可以在类似的加速器单元上复用。

在介绍 Flash-Attention 革命、如何高效地在 GPU 上调度工作负载以及最终解释如何在 GPU 上高效使用各种精度之前，我们将简要说明 GPU 的组织方式。

### 10.1 GPU 入门知识

通常，GPU 具有非常分层的组织结构。在本入门知识中，我们将把讨论保持在对于我们后续演示所需的概念层面。

在计算方面，GPU 由一组称为流式多处理器（SM）的计算单元组成。每个 SM 包含并控制一组流处理器，也称为核心。例如，Nvidia H100 GPU 具有 132 个 SM，每个 SM 有 128 个核心，共计 16,896 个核心（有关张量核心的详细信息，请参见[文档](https://resources.nvidia.com/en-us-tensor-core)），每个核心都能够同时处理多个线程。

![image.png|500](https://nanotron-ultrascale-playbook.static.hf.space/assets/images/diving_primergpu.svg)

Source: https://blog.codingconfessions.com/p/gpu-computing

内存方面也具有高度层级结构，包含多层缓存和内存：寄存器是最小的单元，在执行期间专属于各个线程；共享内存（Shared Memory）和一级缓存（L1 cache）在单个流式多处理器（SM）上运行的线程之间共享；再往上是所有流式多处理器共享的二级缓存（L2 cache）；最后是全局内存（Global Memory），它是 GPU 上最大的内存（例如 H100 宣传的 80 GB），但也是访问和查询速度最慢的。

![image.png|500](https://nanotron-ultrascale-playbook.static.hf.space/assets/images/diving_primergpu2.svg)

Source: https://www.youtube.com/watch?v=ZQKMZIP3Fzg

GPU 的目标将是通过利用这种计算/内存的分层组织，在 GPU 核心上并行运行尽可能多的工作负载。

在 GPU 内核上运行的一段代码称为内核（kernel）。例如，它可以使用 CUDA 或 Triton 等高级语言编写，然后编译为并行线程执行（PTX），即 NVIDIA GPU 所使用的低级汇编语言。

要运行内核，你还需要一个特定的代码部分，称为主机代码（host code），它在 CPU/主机上执行，负责准备数据分配以及加载数据和代码。

```python
// Host code                
void vecAdd(float* h_A, float *h_B, float *h_c, int n) {
    // Allocate vectors in device memory
    int size = n * sizeof(float);
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    // Copy vectors from host memory to device memory
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Invoke kernel
    int threadsPerBlock = 256;
    int blocksPerGrid =
            (N + threadsPerBlock - 1) / threadsPerBlock;
    VecAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    // Copy result from device memory to host memory
    // h_C contains the result in host memory
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}
```

用于添加两个向量的 CUDA 内核的主机代码。改编自 https://docs.nvidia.com/cuda/cuda-c-programming-guide/ 和 https://blog.codingconfessions.com/p/gpu-computing

```python
// Device code
__global__ void VecAdd(float* A, float* B, float* C, int N)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N)
        C[i] = A[i] + B[i];
}
```

包含从 https://docs.nvidia.com/cuda/cuda-c-programming-guide/ 和 https://blog.codingconfessions.com/p/gpu-computing 适配而来的矢量加法内核定义的设备代码

内核通常按以下方式调度：

- 线程被分组到大小为 32 的线程束（warps）中。一个线程束中的所有线程同步执行指令，但在数据的不同部分上执行。
- 线程束被分组到大小更灵活的更大块（block）中（例如大小为 256），每个块仍然分配给一个流式多处理器（SM）。一个 SM 可以并行运行多个块，然而，根据资源情况，并非所有块都会立即分配执行，有些可能会被列入等待列表等待资源。

从这些细节中要记住的主要一点是，存在各种规模和分配方面的限制（各类内存的大小、线程束中并发块和线程的数量），要最有效地使用 GPU 架构，就需要考虑这些限制。

大多数情况下，你不需要达到这种精度水平，并且幸运的是，你可以复用社区其他成员准备好的内核和代码。但无论如何，我们都想给你一个关于如何开始使用内核的入门指导！

### 10.2 如何通过内核提高性能？

如果您想添加一个缺乏优化内核的新操作，或者加速现有的 PyTorch 函数，从头编写内核似乎是最直接的途径。然而，从头创建高性能的 CUDA 内核需要丰富的经验和陡峭的学习曲线。通常更好的入门方法是利用 `torch.compile`，它通过捕获您的操作并生成更低级别、高性能的 Triton 内核来动态优化 PyTorch 代码。

假设你想为一个名为指数线性单元（Exponential Linear Unit）的激活函数编写一个内核。$$
\text{ELU}(x) = 
\begin{cases} 
e^x - 1 & \text{if } x < 0 \\
x & \text{if } x \geq 0 
\end{cases}$$
你可以从一个简单的 PyTorch 实现开始，然后直接在顶部添加 `@torch.compile` 装饰器：

```python
@torch.compile
def elu(x, alpha=1.0):
    return torch.where(x < 0, alpha * (torch.exp(x) - 1), x)
```

编译版和非编译版之间的差异非常显著，尤其是考虑到我们仅仅添加了一个装饰器。这种显著的差异在下图中得到了说明（$N$ 为列数）。

![image.png|500](https://nanotron-ultrascale-playbook.static.hf.space/assets/images/torch-compile-triton.png)

然而，如果这种性能提升不够，你可以考虑实现 Triton 内核。作为起点，你可以看看由 `@torch.compile` 生成的 triton 内核。为此，你只需将环境变量 `TORCH_LOGS` 设置为 `"output_code"`：

```bash
export TORCH_LOGS="output_code"
```

运行带有 `@torch.compile` 装饰器的 Python 脚本后，它将生成并输出相应的 Triton 内核，在本例中为：

```python
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 100000000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = 0.0
    tmp2 = tmp0 < tmp1
    tmp3 = tl_math.exp(tmp0)
    tmp4 = 1.0
    tmp5 = tmp3 - tmp4
    tmp6 = tl.where(tmp2, tmp5, tmp0)
    tl.store(out_ptr0 + (x0), tmp6, xmask)
```

为了提高可读性，我们可以修改变量名、添加注释并进行一些小的调整（或者让大型语言模型为我们做这件事），如下所示：

```python
@triton.jit
def elu_kernel(input_ptr, output_ptr, num_elements, BLOCK_SIZE: tl.constexpr):
    # Calculate the starting index for this block
    block_start = tl.program_id(0) * BLOCK_SIZE
    # Create an array of indices for this block
    block_indices = block_start + tl.arange(0, BLOCK_SIZE)[:]
    # Create a mask to ensure only valid indices are processed
    valid_mask = block_indices < num_elements
    # Load input values from the input pointer based on valid indices
    input_values = tl.load(input_ptr + block_indices, valid_mask)
    # Define the ELU parameters
    zero_value = 0.0  # Threshold for ELU activation
    negative_mask = input_values < zero_value
    exp_values = tl.math.exp(input_values)
    # Define the ELU output shift
    one_value = 1.0
    shifted_exp_values = exp_values - one_value

    output_values = tl.where(negative_mask, shifted_exp_values, input_values)

    # Store the computed output values back to the output pointer
    tl.store(output_ptr + block_indices, output_values, valid_mask)
```

在这里，`tl.program_id(0)` 提供了一个唯一的块 ID，我们用它来确定该块将处理数据的哪个部分。使用这个块 ID，`block_start` 计算每个块部分的起始索引，而 `block_indices` 指定该部分内的索引范围。一个 `valid_mask` 确保只处理 `num_elements` 内的索引，使用 `tl.load` 安全地加载数据。然后应用 ELU 函数，根据值是否为负对其进行修改，并将结果使用 `tl.store` 写回内存。

当我们使用 `triton.testing.Benchmark` 对生成的内核进行基准测试时，我们得到以下性能：

![image.png|500](https://nanotron-ultrascale-playbook.static.hf.space/assets/images/torch-compile-triton-kernel.png)

这个独立内核甚至在较小尺寸上相比 `@torch.compile` 展示了更优的性能，但这很可能只是`torch.compile` 编译时间的一个假象。无论如何，与其从头开始，不如记住你可以从这些生成的内核开始，并将注意力集中在优化其性能上，从而在过程中节省大量时间。

即使在 Triton 中，有时由于处理低级细节（如共享内存和流式多处理器（SM）内的调度）的语言限制，我们也无法完全发挥设备的峰值性能。Triton 的能力仅限于块以及跨 SM 的块调度。若要获得更深入的控制，您需要直接在 CUDA 中实现内核，在那里您将能够访问所有底层的低级细节。

深入到 CUDA 领域，可以采用多种技术来提高内核的效率。这里我们仅介绍几种：优化内存访问模式以减少延迟；使用共享内存存储频繁访问的数据；以及管理线程工作负载以尽量减少空闲时间。

在我们深入研究 CUDA 示例之前，让我们总结一下我们见过的那些可让我们编写内核代码以在 GPU 上执行指令的工具：

1. Pytorch：简单但速度慢
2. torch.compile：简单、快速，但不灵活
3. triton：更难、更快且更灵活
4. CUDA：最难、最快且最灵活（如果你能正确运用的话）

让我们来谈谈在 CUDA 中我们可以使用的最常见的技术之一：优化内存访问。GPU 中的全局内存（我们上面图表中最大的内存）与缓存相比具有较长的延迟和较低的带宽，这通常会成为大多数应用程序的主要瓶颈。高效地从全局内存中访问数据可以大幅提高性能。

#### 10.2.1 内存合并

为了有效利用全局内存的带宽，理解其架构至关重要。在 CUDA 设备中，全局内存是通过动态随机存取存储器（DRAM）实现的。

内存合并利用了 DRAM 在访问内存地址时以突发方式或连续内存地址范围的方式传输数据的特点。每次访问 DRAM 位置时，包括所请求位置在内的一系列连续位置会被 DRAM 芯片中的多个传感器并行读取。一旦读取，这些数据就可以作为突发快速传输到处理器。在 CUDA 中，合并利用这种突发行为，通过确保一个 warp 中的线程（32 个以锁步方式执行相同指令的线程（SIMD））访问连续的内存位置，来最大化内存访问效率。例如，如果线程 0 访问位置 M，线程 1 访问 M+1，线程 2 访问 M+2，依此类推，GPU 硬件会将这些请求合并或组合成一个大的、高效的 DRAM 突发访问请求，而不是单独处理每个访问。

让我们以矩阵乘法为例。一种简单直接的方法是让每个线程计算输出矩阵的一个元素，如下所示：

```clike
__global__ void matmul_naive(int M, int N, int K, const float *A, const float *B, float *C) {
    const uint x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < M && y < N) {
        float tmp = 0.0;
        for (int i = 0; i < K; ++i) {
            tmp += A[x * K + i] * B[i * N + y];
        }
        C[x * N + y] = tmp;
    }
}
```

这里有一个来自这篇精彩[博客文章](https://siboehm.com/articles/22/CUDA-MMM)的关于内核的优秀可视化示例：

![image.png|600](https://nanotron-ultrascale-playbook.static.hf.space/assets/images/memorycoalescing.png)

然而，当使用像 ncu 这样的工具对这个内核进行性能分析时，我们可以看到一些问题，包括内存吞吐量低和内存访问未合并。
![image.png](https://nanotron-ultrascale-playbook.static.hf.space/assets/images/memorycoalescing2.png) ![image.png](https://nanotron-ultrascale-playbook.static.hf.space/assets/images/memorycoalescing3.png)

原因在于，在该内核中，同一线程块中线程 ID 为 (0, 0) 和 (1, 0) 的两个线程（它们最终会被划分到同一个线程束中）都会从矩阵 B 的同一列加载数据，但从矩阵 A 的不同行加载数据。由于矩阵元素是以行优先顺序存储的（即行元素存储在连续的内存地址中，如下图所示），因此在第一次迭代 i = 0 时，线程 (0, 0) 会加载 A₀,₀ ，而线程 (1, 0) 会加载 A₁,₀ 。这些元素在内存中并非紧密相邻存储，并且在每次迭代中都会存在这种未对齐的情况，从而无法实现内存访问的合并 。

![image.png|600](https://nanotron-ultrascale-playbook.static.hf.space/assets/images/memorycoalescing4.png)

为了提高我们内核的性能，我们可以将坐标 x 和 y 的计算方式更改为以下方式：

```clike
const int x = blockIdx.x * BLOCKSIZE + (threadIdx.x / BLOCKSIZE);
const int y = blockIdx.y * BLOCKSIZE + (threadIdx.x % BLOCKSIZE);

if (x < M && y < N) {
float tmp = 0.0;
for (int i = 0; i < K; ++i) {
    tmp += A[x * K + i] * B[i * N + y];
}
C[x * N + y] = tmp;
}
```

我们改用一维块，并重新定义确定 `x` 和 `y` 值的方式。在这种新方法中，同一线程束（`threadIdx.x` 值相近）内的线程将共享相同的 `x` 值，但具有不同的 `y` 值。这意味着它们将加载矩阵 `A` 的同一行，但加载矩阵 `B` 的不同列。因此，对于行主序矩阵，内存访问可以实现合并。

当我们对新内核进行分析时，我们注意到关于非合并内存访问的警告消失了，并且 GPU 的内存吞吐量提高了大约 10 倍。

![image.png](https://nanotron-ultrascale-playbook.static.hf.space/assets/images/memorycoalescing5.png)

我们还注意到内核的执行时间减少了 10 倍！太神奇了。  

现在让我们来介绍另一种在文献中经常提到的技术：平铺（tiling）。

#### 10.2.2 平铺（tiling）

平铺是一种利用 *共享内存* 来优化内存访问模式的技术。正如我们上面提到的，共享内存是一种小型、快速的内存，可由一个块内的所有线程访问。它允许数据被多个线程重复使用，减少了从较慢的全局内存中重复加载数据的需要。

例如，在矩阵乘法中，一个块中的每个线程可能需要两个矩阵（假设为A和B）的元素。如果每个线程独立地从全局内存加载其所需的行和列，那么由于一个块中的多个线程会访问重叠的数据，最终会产生许多冗余的加载操作。相反，我们可以使用平铺（tiling）技术，将A和B的一个块（或平铺块）一次性加载到共享内存中，这样该块中的所有线程就可以重复使用相同的共享数据。

在平铺方法中，每次迭代都涉及块内的所有线程协同加载两个平铺块——一个来自矩阵 A，另一个来自矩阵 B——到共享内存中。具体来说，线程加载矩阵 A 的一个平铺块（大小为 `BLOCK_SIZE_M` 乘以`BLOCK_SIZE_K`）和矩阵 B 的一个平铺块（大小为 `BLOCK_SIZE_K` 乘以 `BLOCK_SIZE_N`）。一旦平铺块进入共享内存，线程就在这些平铺块上执行矩阵乘法，由于所有必要的数据都能快速访问，从而实现高效计算。平铺块乘法的结果存储在一个累积矩阵中，该矩阵保存中间结果。每次迭代后，当前平铺块乘法的结果都会添加到这个累积矩阵中，直到处理完两个矩阵的所有平铺块为止。

![image.png|400](https://nanotron-ultrascale-playbook.static.hf.space/assets/images/tiling.png)

From [https://cnugteren.github.io/tutorial/pages/page4.html](https://cnugteren.github.io/tutorial/pages/page4.html)

让我们来看一下在实现中你需要理解的重要部分：

```clike
// Set pointers to the starting elements
A += blockRow * TILE_SIZE * K; // Start at row = blockRow, column = 0
B += blockCol * TILE_SIZE; // Start at row = 0, column = blockCol
C += blockRow * TILE_SIZE * N + blockCol * TILE_SIZE; // Start at row = blockRow, column = blockCol
float sum = 0.0;
// The outer loop moves through tiles of A (across columns) and B (down rows)
for (int tileIdx = 0; tileIdx < K; tileIdx += TILE_SIZE) {
sharedA[localRow * TILE_SIZE + localCol] = A[localRow * K + localCol];
sharedB[localRow * TILE_SIZE + localCol] = B[localRow * N + localCol];

// Ensure all threads in the block have completed data loading
__syncthreads();

// Shift pointers to the next tile
A += TILE_SIZE;
B += TILE_SIZE * N;

// Compute the partial dot product for this tile
for (int i = 0; i < TILE_SIZE; ++i) {
    sum += sharedA[localRow * TILE_SIZE + i] * sharedB[i * TILE_SIZE + localCol];
}
// Synchronize again to prevent any thread from loading new data
// into shared memory before others have completed their calculations
__syncthreads();
}
C[localRow * N + localCol] = sum;
```

为简单起见，我们考虑采用方形平铺。

每个线程首先从矩阵 A 和矩阵 B 中各加载一个元素到共享内存中。在这种情况下，通过将 `threadIdx.x` 赋值为局部列索引（localCol），实现合并内存访问变得简单，同一 warp 中的线程将访问两个矩阵的相邻元素。在块中的每个线程完成将其元素加载到共享内存后（通过调用 `__syncthreads()` 确保），它们继续计算两个分块的点积。一旦线程遍历完所有分块——矩阵 A 水平方向和矩阵 B 垂直方向——最终的和将存储在矩阵 C 的相应位置。

当使用 ncu 对这个内核进行基准测试时，我们注意到内存吞吐量增加到了410 Gb/s，内核执行时间减少了约 43%，性能达到了约 6.6 TFLOPs。

#### 10.2.3 线程粗化

平铺技术显著提高了我们内核的性能。然而，在分析用于量化每个状态所花费周期数的线程束状态时，我们观察到以下情况：

![image.png](https://nanotron-ultrascale-playbook.static.hf.space/assets/images/threadcoarsening.png)

这些神秘的状态名称的含义可以在 [NVIDIA 的性能分析指南](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#metrics-reference)的“线程束停滞原因”部分找到。在那里我们可以读到：

*`“smsp__pcsamp_warps_issue_stalled_mio_throttle`：线程束因等待内存输入/输出（MIO）指令队列不满而停滞。在 MIO 流水线极度使用的情况下（包括特殊数学指令、动态分支以及共享内存指令），这种停滞原因会增多。当由共享内存访问引起时，尝试使用更少但更宽的加载操作可以减轻流水线压力。”*

所以看来，弯曲（warp，此处可能为特定术语，如 CUDA 编程中的线程束概念）在等待共享内存访问返回时处于停滞状态！为解决这个问题，我们可以采用一种称为“线程粗化”（Thread Coarsening）的技术，该技术涉及将多个线程合并成一个粗化后的线程。这将显著减少共享内存访问次数，因为每个粗化后的线程可以处理多个输出元素。

让我们简要探讨一下在编写或改进自定义内核时的最后一个重要考虑因素：最小化控制分歧。

#### 10.2.4 最小化控制分歧

流式多处理器（SM）旨在使用单指令多数据（SIMD）模型执行一个线程束中的所有线程。这意味着在任何给定时刻，一个指令会同时被获取并执行，以用于该线程束内的所有线程。当执行一个线程束时，其中的线程操作数据的不同部分，但遵循相同的指令，因此得名单指令多数据。SIMD的主要优势在于其效率；负责指令获取和分派的控件硬件在多个执行单元之间共享。这种设计最小化了与控制功能相关的硬件开销，使更大一部分硬件专注于提高算术吞吐量。

当同一warp中的线程采取不同的执行路径时，就会发生控制分歧。例如，如果一个条件语句（如 `if` 语句）导致一些线程执行一个代码块，而其他线程执行另一个代码块，那么该warp必须对这些执行进行串行化处理，从而导致一些线程空闲等待其他线程完成。为了尽量减少控制分歧，我们需要设计内核，以确保同一warp中的线程遵循相同的执行路径。这可以通过重构代码以减少分支、使用确保所有线程遵循相似执行路径的数据结构，或者采用诸如预测执行之类的技术来实现。

---

我们已经介绍了编写自定义内核以及提高 GPU 操作的性能和内存占用的一些主要考虑因素。但在进入实际示例之前，还有一个更重要的概念，即“内核融合”。

### 10.3 融合内核

现在我们在几个地方提到了 GPU 和 CPU 操作可以是异步的。特别是，CPU 上的主机代码可以以非阻塞的方式在 GPU 上调度工作负载。

非阻塞在重叠通信和计算方面可能很有用——正如我们在旅程中多次看到的那样——但可以将其扩展到更普遍的思路，即尽量避免在主机和 GPU 内核命令之间来回切换。

[贺拉斯·赫（Horace He）](https://horace.io/brrr_intro.html)在这些图表中非常形象地阐释了这个观点：

![image.png|300](https://nanotron-ultrascale-playbook.static.hf.space/assets/images/fused_kernels1.png)

需要在全局内存和计算单元之间反复传输的一系列内核操作

![image.png|300](https://nanotron-ultrascale-playbook.static.hf.space/assets/images/fused_kernels2.png)

我们不是将三角形发送回全局内存然后再重新读取它，而是将所有操作一次性完成。

我们怎样才能避免这种反复呢？最好的方法是让我们的 GPU 尽可能地自主运行。这可以通过在单个内核中将尽可能多的连续计算操作组合在一起来实现，GPU 将运行这个内核，称为“融合内核”。

融合内核对于在每个输入标记上独立执行的类点操作的连续操作特别高效且易于编写。在这种情况下，在将计算值移至共享内存并启动新内核之前，没有必要将计算值带回全局内存。在完成连续计算之前，将所有值保留在本地要高效得多。

在 Transformer 模型中，有很多地方可以应用这种“融合”方法：每次我们有一系列逐点操作时，例如在层归一化所涉及的计算中。

我们现在完全理解了内核工程的一个真正杰作：Flash Attention，不禁为之惊叹。

### 10.4 Flash Attention 1-3

“Flash attention” 由 [Tri Dao](https://tridao.me/) 引入，旨在通过编写自定义 CUDA 内核来优化注意力计算，使其速度更快且内存效率更高。Flash Attention 背后的理念是高效利用 GPU 的各种内存，避免过度依赖最慢的一种：GPU 的全局内存。

（请注意，GPU 的全局内存被令人困惑地称为“高带宽内存”（HBM 🫠）。）

注意力机制的一种基本实现涉及内存和工作器之间的大量数据传输。它需要在高带宽内存（HBM）中实例化 S 和 P 矩阵，这意味着需要将结果发送到 HBM，然后再发送回静态随机存取存储器（SRAM）以进行后续计算。

![image.png|400](https://nanotron-ultrascale-playbook.static.hf.space/assets/images/flashattn.png)

由于高带宽内存（HBM）中的带宽要低得多，这在注意力计算中引入了一个严重的瓶颈。我们能做得更好吗？特里·达（Tri Dao）说可以！

关键要素是将 S 矩阵以小块形式进行计算，这些小块能够适配共享内存单元（SM）中较小的共享内存。但我们可以做得更好，即完全避免将非常大的S矩阵实例化，而是仅保留计算 softmax 归一化因子所需的统计信息。这样一来，我们就可以直接在静态随机存取存储器（SRAM）中一次性计算部分OO ，而无需来回移动中间结果。在这种情况下，我们不仅没有利用共享内存，还消除了因实例化模型中（在长上下文长度情况下）最大的激活矩阵之一——注意力矩阵而导致的内存瓶颈。

![image.png|600](https://nanotron-ultrascale-playbook.static.hf.space/assets/images/flashattn2.png)

Source: FlashAttention paper[13]

“闪存注意力”的想法解决了模型训练中的诸多瓶颈，因此它迅速成为所有变压器中执行注意力的默认方式：

- 通过避免对 S 矩阵进行显式计算，我们减轻了注意力机制的内存负担
- 我们还消除了注意力机制 S² 成本的大部分直接影响

因此，所有线性注意力的变体以及近似注意力的次二次方方法（这些方法是在变压器架构发明后不久开发的）大多都被搁置一旁，转而采用这种精确且快速的闪存注意力实现和机制。

继 Flash-attention 1 之后，同一实验室发布了两个连续改进的版本：Flash-attention 2 和 3。与Flash-attention 1 相比，Flash-attention 2 和 3 的改进不太在于一般的注意力机制，而在于通过（1）尽可能减少非矩阵乘法操作的数量（2）在 wraps 和线程块之间仔细划分工作负载（对于 Flash Attention 2），以及针对最新的 Hopper（H100）架构上的 FP8 和 Tensor Core 支持仔细优化（对于Flash Attention 3），使其低级实现更具体地适配 GPU。

“Flash attention 对能够加速的注意力模式有一定限制。可以看看 [FlexAttention](https://pytorch.org/blog/flexattention/)，它是一种快速且灵活的变体。”

“闪电注意力”（Flash-Attention）是一个典范示例，它展示了当你考虑到当前 GPU 加速器的内部内存/计算设计时所能带来的突破性改进。

---

到目前为止，在本操作融合部分所描述的技术要求我们实现建模代码更改，并为某些操作编写自定义内核，以加快训练速度。

在对计算操作本身的低层次深入探讨的最后一部分中，我们将介绍一系列对建模代码不可知的方法，这些方法适用于任何模型，并且被广泛使用以至于已成为行业标准的：混合精度训练！

### 10.5 混合精度训练

在本书的各个部分，我们已经讨论了较低精度格式及其对存储激活值、参数和优化器状态所需内存的影响。现在是时候更深入地研究这些格式的细节，并更好地理解它们的权衡、优势和局限性。

混合精度训练，顾名思义，就是在训练过程中混合使用不同的精度。PyTorch 张量的默认数值精度是单精度浮点格式，也称为 FP32 或 float32，这意味着存储的每个数字占用 32 位或 4 个字节。表示一个数字的可用位被分为 3 部分：

- 符号位：第一位决定数字是正数还是负数
- 尾数：决定数字的有效数字
- 指数：控制数字的大小

![sign-mantissa-exponent.svg|500](https://nanotron-ultrascale-playbook.static.hf.space/assets/images/sign-mantissa-exponent.svg)

通过回顾数字的科学计数法（例如 −5.734×10⁷ ，其中首先有符号，然后是尾数和指数），可以很容易地说明浮点数的原理。因此，我们可以用自适应的精度表示很大范围的数值。虽然 float32 是默认的，但 PyTorch 中有一系列的浮点格式可用：

| 格式         | 总位数 | 符号位 | 指数位 | 尾数位 |
|--------------|--------|--------|--------|--------|
| float32      | 32     | 1      | 8      | 23     |
| float16      | 16     | 1      | 5      | 10     |
| bfloat16     | 16     | 1      | 8      | 7      |
| float8 (e4m3)| 8      | 1      | 4      | 3      |
| float8 (e5m2)| 8      | 1      | 5      | 2      |
（注意：您可能想知道 bfloat16 中的“b”是从哪里来的。这种格式是在谷歌大脑（Google Brain）中开发的，因此“b”代表“大脑（brain）”。）

减少总位数是需要付出代价的（这里也没有免费的午餐），但我们可以在一定程度上控制如何付出代价。要么我们可以牺牲更多的尾数位，要么牺牲更多的指数位。因此，还存在两种 float8 格式，根据指数和尾数来命名，以便灵活选择最合适的格式。我们可以看看每种格式可能的数字范围：

![image.png|500](https://nanotron-ultrascale-playbook.static.hf.space/assets/images/mixedprecision.png)

我们可以看到，float32 的数值跨度达到 80 个数量级，而 float16 牺牲了很大的范围，bfloat16 则保持了完整的范围。两种 float8 格式进一步缩小了范围，其中 e5e2 可以保持 float16 的范围，而 e4m3 的范围则更小。

为什么有些格式能够保持范围而有些则不能？让我们通过在 1 到 2 之间绘制 10,000 个点来研究分辨率：每个点将被四舍五入到每种格式中最接近的可表示数字。

![image.png|500](https://nanotron-ultrascale-playbook.static.hf.space/assets/images/mixedprecision_2.png)

我们可以在这里看到，与 float16 相比，bfloat16 保持了 float32 的范围，但这是以牺牲更多精度为代价的。在 float8 的情况下，情况更加严峻，因为 e4m3 在 1 - 2 区间只能表示 7 个数，而 e5m2 只能表示 3 个数 。

衡量格式分辨率的一个常用指标是 epsilon：即 1.00 之后第一个可表示的数字 1.00。我们可以看到，对于 float32 格式，$10^{−4}$ 是一个上限（实际为 $1.19^{−7}$）。对于 float16，该值约为 $10^{−3}$，而对于bfloat，这个值还要高出 10 倍。

混合精度训练的思想是在保持全精度训练性能的同时使用其中一些较低精度格式。

事实证明，我们不能完全放弃 float32，通常需要保留一些全精度部分。这就是为什么低精度训练通常被称为混合精度训练。

现在让我们来看看用 16 位训练模型，然后看看能否更进一步，一直降到 8 位。

#### 10.5.1 FP16 和 BF16 训练

天真地将所有张量和操作都转换为 float16 是不行的，结果通常是损失发散。然而，最初的混合精度训练论文[2]提出了三种技巧来匹配 float32 训练：

1. 权重的 FP32 副本：float16 权重可能存在两个问题。在训练过程中，一些权重可能会变得非常小并被四舍五入为 0。然而，即使权重本身不接近零，如果更新非常小，量级的差异也可能导致权重在加法过程中下溢。一旦权重变为零，由于不再有梯度信号传入，它们将在剩余的训练过程中保持为 0。
2. 损失缩放：梯度也存在类似的问题，因为梯度往往远小于 1，因此有下溢的风险。一个简单而有效的策略是在反向传播之前缩放损失，并在反向传播之后对梯度进行反缩放。这确保了在反向传播过程中不会发生下溢，并且在处理梯度（例如裁剪）和优化步骤之前我们进行反缩放，因此缩放不会影响训练。
3. 累积：最后，在执行某些 16 位精度的算术运算（如平均值或求和）时，我们也可能面临下溢或上溢的问题。一种解决方案是在操作期间将中间结果累积在 float32 中，并且仅在最后将最终结果转换回 16 位精度。

借助这些技术，我们可以在受益于更快、更低精度算术运算带来的更高吞吐量的同时，获得稳定的训练效果。自然地，作为一个充满好奇心的读者——并且到目前为止有点痴迷于实现吞吐量最大化——你可能会问这样一个问题：我们能否超越 16 位精度，实现更进一步的加速呢？

或许吧！

#### 10.5.2 FP8 预训练

即使我们将通信与计算完美地重叠，我们最终还是会遇到硬件本身低层次理论浮点运算次数（FLOPS）的限制，即我们硬件上每个单独操作的效率。这时数值精度就变得至关重要。例如，在英伟达（NVIDIA）的 H100 GPU 上，FP8 矩阵乘法（GEMM 运算）的理论浮点运算次数是 bfloat16 的两倍，这使得低精度训练成为进一步优化的有吸引力的途径。

近期研究（包括FP8-LM[14]、torchao[15]和DeepSeek-V3[7]）已经证明了FP8训练用于大规模模型的潜力。不过，FP8预训练引入了一个重大挑战：稳定性。在较低精度下，数值不稳定常常导致损失发散，使得难以达到更高精度训练的准确率。

我们知道，对于固定的模型大小，学习率上升时不稳定性会增加[16]，这使得 FP8 预训练特别棘手。

以下是 FP8 训练中典型的发散损失曲线的一个示例：

[交互图]

首次成功的大规模 FP8 混合精度训练在 DeepSeek-V3 上被公开报道。作者仔细分析了前向传播（Fprop）以及激活（Dgrad）和权重（Wgrad）反向传播的每个操作。与 BF16 混合精度训练类似，一些聚合和主权重保持较高精度，而操作本身以 FP8 执行。

![image.png|600](https://nanotron-ultrascale-playbook.static.hf.space/assets/images/fp8_diagram.png)

为了从高精度（例如 FP32 或 BF16）切换到范围更小的低精度（例如 FP16 或 FP8），我们需要对激活值的范围进行归一化处理，例如通过计算它们的绝对最大值。DeepSeek-V3 进一步引入了一种特定的量化方案，其中范围按瓦片进行归一化：输入/激活为 1x128，权重和缩放元素为 128x128。这使得归一化受激活值中异常值的影响较小。他们还提出了一些额外的技巧来进一步减少内存和通信开销，你可以在 DeepSeek-V3 技术报告[7]的第 3.3 节中了解这些技巧。

以下是一些已知的 FP8 训练方法的总结：

| GEMM 精度        | 主模型权重 | 累积梯度 | 模型权重 | 梯度  | 优化器状态    | 总内存                     |
|------------------|------------|----------|----------|-------|---------------|----------------------------|
| bfloat16 与 fp32 混合精度基线 | bf16       | fp32     | bf16     | bf16  | fp32 + fp32 | 4 + 4 + 2 + 2 + 4 + 4 = 20 字节 |
| 无 FP32 梯度累积 | bf16       | n/a      | bf16     | bf16  | fp32 + fp32 | 4 + 2 + 2 + 4 + 4 = 16 字节 |
| Transformer 引擎 | fp8        | n/a      | fp32     | fp32  | fp32 + fp32 | 4 + 4 + 4 + 4 = 16 字节 (减少 20%) |
| FP8-LM 的 O3 级别 | fp8        | fp16     | fp16     | fp8   | fp8 + fp16  | 2 + 2 + 1 + 1 + 1 + 2 = 9 字节 (减少 55%) |
| DeepSeek-V3      | fp8        | fp32     | fp32     | fp8   | bf16 + bf16 | 4 + 4 + 1 + 2 + 2 + 2 = 15 字节 (减少 25%) |
| nanotron 的 FP8  | fp8        | bf16     | fp16     | fp8   | fp8 + fp8   | 2 + 4 + 1 + 1 + 1 + 1 = 10 字节 (减少 50%) |

总体而言，在 2025 年初，FP8 仍是一种实验性技术，相关方法仍在不断发展。鉴于其明显的优势，它很可能会成为标准，并很快取代 bf16 混合精度。若要关注 FP8 训练技术的开源实现，请查看[此拉取请求](https://github.com/huggingface/nanotron/pull/70)中 nanotron 的实现。

展望未来，英伟达的下一代芯片 Blackwell [已宣布](https://www.nvidia.com/en-us/data-center/technologies/blackwell-architecture/)支持 FP4 训练，这将进一步提高训练速度，但无疑也会带来新的训练稳定性挑战。

---

这最后一部分为我们在数十乃至数千个 GPU 上训练快速大型模型的漫长旅程画上了句号。现在是时候让我们的 GPU 集群慢慢停止运行，退一步来总结一下我们在此过程中所学到的一切了。

## 十一、总结

恭喜你，亲爱的读者，你坚持到了最后！我们完成了一段相当漫长的旅程：从了解如何在单个 GPU 上训练一个简单模型开始，一直到掌握在数千个 GPU 上高效训练像 Llama-405B 和 DeepSeek-V3 这样的大型语言模型的所有复杂技术。到现在为止，你可以（相对）轻松地读懂一个图表，比如Llama-3的 4D 并行设置。

![image.png|600](https://nanotron-ultrascale-playbook.static.hf.space/assets/images/conclusion_llama3_parallelism.png)

协调大型 GPU 集群以高效训练大型语言模型（LLMs）并非易事。我们学会了如何优化 GPU 之间的计算和通信，使它们始终以最大利用率运行。这涉及为给定的模型和集群大小选择合适的并行策略，在可能的情况下重叠通信和计算，并编写自定义内核，考虑硬件布局以便在 GPU 上尽可能快地执行操作。

你可能仍然认为这些知识有点小众，只与预训练大型语言模型（LLMs）的一小部分人有关。从历史上看，这可能是真的，但随着人工智能构建者社区和模型规模都在迅速增长，使用分布式技术进行推理、微调及训练的人员群体也在呈指数级增长，这使得分布式训练设置变得越来越普遍。因此，更深入地探究分布式相关的所有内容或许非常适时。

这是一段漫长的学习之旅，但不仅仅是对你们而言！在 GPU 集群上运行数千个基准测试比我们预期的更具挑战性，我们也想分享一下我们自己学习过程中的一些亮点。

### 11.1 那么，接下来是什么？

你现在对主要的分布式训练概念有了很好的了解，但与此同时，我们只是浅尝辄止地涉及了其中一些工具和技术。深入了解一个主题有很多方法，但我们推荐以下几个步骤：

* 仔细阅读一些具有里程碑意义或非常近期的论文。你可以在参考文献中找到一份非常详尽的最具影响力的论文、博客文章和书籍的清单。
* 从零开始自己实现一个算法。通常，只有当你自己实现了一个方法时，它才会真正“豁然开朗”。
* 深入研究一个广泛使用的框架并开始贡献：修复漏洞、解答问题或实现新功能。这是进入任何机器学习领域的最佳方式！  

我们希望本书能帮助你开启分布式训练之旅，并且你能训练出下一代强大的模型，伴随着你的 GPU 集群的嗡嗡声！

---

给我们的首批读者最后说几句。我们对这篇作品非常满意，决定限量制作一些实体印刷版作为礼物送给首批读者。

如果您是前 50 个在下方填写电子邮件地址的人之一，我们将在今年晚些时候与您联系，在将其排版为印刷本后给您寄送一份实体书。

我们预计这本书大约会有 100 - 150 页，并且涵盖与博客文章相同的内容，但我们也可能会根据作为印刷品是否合理来决定对其进行缩短或扩充。

如需获取纸质版，请在以下谷歌表单中填写您的电子邮箱地址。

无论你是我们最早的一批读者，还是很久之后才看到这篇博客文章的读者，我们都很高兴看到你喜欢这次知识分享。愿开源和开放科学的精神永远与你同在。

### 11.2 致谢  

我们感谢 Elie 进行了全面的审查，并使用 NotebookLM 创建了音频组件。特别感谢 Hynek 优化了前端性能。我们还感谢 Simon 解决了 Hub 的一些问题。

### 11.3 讨论页

如果您想讨论这篇博客文章的内容、提出问题、建议修改或者只是打个招呼，请在讨论页面上开一个主题帖。

## 十二、参考文献

### 12.1 具有里程碑意义的大型语言模型扩展论文

* [**Megatron-LM**](https://arxiv.org/abs/1909.08053)：介绍用于训练大型语言模型的张量并行和高效模型并行技术。
* [**Megatron-Turing NLG 530B**](https://developer.nvidia.com/blog/using-deepspeed-and-megatron-to-train-megatron-turing-nlg-530b-the-worlds-largest-and-most-powerful-generative-language-model/)：描述了使用 DeepSpeed 和 Megatron-LM 框架组合训练一个 530B 参数模型的过程。
* [**PaLM**](https://arxiv.org/abs/2204.02311)：介绍了谷歌的 Pathways 语言模型，该模型在数百种语言任务和推理能力方面展现出强大的性能。
* [**Gemini**](https://arxiv.org/abs/2312.11805)：介绍谷歌的多模态模型架构，该架构能够处理文本、图像、音频和视频输入。
* [**Llama 3**](https://arxiv.org/abs/2407.21783)：Llama 3 模型群
* [**DeepSeek-V3**](https://arxiv.org/abs/2412.19437v1)：DeepSeek 关于 DeepSeek-V3 模型架构与训练的报告。

### 12.2 训练框架

* [**Nanotron**](https://github.com/huggingface/nanotron)：我们用于训练大型语言模型的框架，该框架采用多种并行策略
* [**Megatron-LM**](https://github.com/NVIDIA/Megatron-LM)：英伟达用于训练大型语言模型的框架，该框架采用多种并行策略。
* [**DeepSpeed**](https://www.deepspeed.ai/)：微软的深度学习优化库，具有 ZeRO 优化阶段和各种并行策略。
* [**FairScale**](https://github.com/facebookresearch/fairscale/tree/main)：用于大规模训练的 PyTorch 扩展库，提供各种并行和优化技术。
* [**ColossalAI**](https://colossalai.org/)：集成了多种优化技术的大规模模型训练系统。
* [**torchtitan**](https://github.com/pytorch/torchtitan)：一个用于大模型训练的 PyTorch 原生库。
* [**GPT-NeoX**](https://github.com/EleutherAI/gpt-neox)：EleutherAI 用于训练大型语言模型的框架，曾用于训练 GPT-NeoX-20B。
* [**LitGPT**](https://github.com/Lightning-AI/litgpt)：Lightning AI 对最先进的开源 LLMs 的实现，重点在于可复现性。
* [**DiLoco**](https://github.com/PrimeIntellect-ai/OpenDiLoCo)：使用 DiLoCo 跨计算集群训练语言模型。
* [**torchgpipe**](https://github.com/kakaobrain/torchgpipe)：PyTorch 中的 GPipe 实现。
* [**OSLO**](https://github.com/EleutherAI/oslo)：奥斯陆：大规模优化的开源软件。


### 12.3 Debugging

* [**Speed profiling**](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)：使用分析器分析模型性能和瓶颈的官方 PyTorch 教程。
* [**Memory profiling**](https://pytorch.org/blog/understanding-gpu-memory-1/)：全面了解和优化 PyTorch 中 GPU 内存使用的指南
* [**Memory profiling walkthrough on a simple example**](https://huggingface.co/blog/train_memory)：可视化和理解 PyTorch 中的 GPU 内存
* [**TensorBoard Profiler Tutorial**](https://pytorch.org/tutorials/intermediate/tensorboard_profiler_tutorial.html)：使用 TensorBoard 的 PyTorch 模型分析工具指南。

### 12.4 分布式技术

* [**数据并行**](https://siboehm.com/articles/22/data-parallel-training)：深度学习中数据并行训练的全面解释。
* [**ZeRO**](https://arxiv.org/abs/1910.02054)：引入零冗余优化器以优化内存的方式训练大型模型。
* [**FSDP**](https://arxiv.org/abs/2304.11277)：PyTorch 中完全分片数据并行训练的实现。
* [**Tensor and Sequence Parallelism + Selective Recomputation**](https://arxiv.org/abs/2205.05198)：结合不同并行策略的高效大规模模型训练的高级技术。
* [**Pipeline parallelism**](https://developer.nvidia.com/blog/scaling-language-model-training-to-a-trillion-parameters-using-megatron/#pipeline_parallelism)：英伟达关于为大模型训练实现流水线并行的指南。
* [**Breadth first Pipeline Parallelism**](https://arxiv.org/abs/2211.05953)：包括围绕 PP 进度表的广泛讨论。
* [**All-reduce**](https://andrew.gibiansky.com/blog/machine-learning/baidu-allreduce/)：分布式训练中使用的环形全规约算法的详细解释。
* [**Ring-flash-attention**](https://github.com/zhuzilin/ring-flash-attention)：结合闪存注意力机制的环形注意力机制实现高效训练。
* [**Ring attention tutorial**](https://coconut-mode.com/posts/ring-attention/)：解释环形注意力概念和实现的教程。
* [**ZeRO and 3D**](https://www.deepspeed.ai/tutorials/large-models-w-deepspeed/#understanding-performance-tradeoff-between-zero-and-3d-parallelism)：深度学习优化库 DeepSpeed 中关于理解 ZeRO 和 3D 并行策略之间权衡的指南。
* [**Mixed precision training**](https://arxiv.org/abs/1710.03740)：介绍用于深度学习模型的混合精度训练技术。
* [**Visualizing 6D Mesh Parallelism**](https://main-horse.github.io/posts/visualizing-6d/)：解释 6D 并行网格中涉及的集体通信。


### 12.5 硬件

* [**Fire-Flyer - a 10,000 PCI chips cluster**](https://www.arxiv.org/abs/2408.14158)：DeepSeek 关于设计一个拥有 1 万个 PCI GPU 的集群的报告。
* [**Meta's 24k H100 Pods**](https://engineering.fb.com/2024/03/12/data-center-engineering/building-metas-genai-infrastructure/)：Meta 使用英伟达 H100 GPU 构建的大规模人工智能基础设施的详细概述。
* [**Semianalysis - 100k H100 cluster**](https://www.semianalysis.com/p/100000-h100-clusters-power-network)：大规模 H100 GPU 集群分析及其对人工智能基础设施的影响
* [**Modal GPU Glossary**](https://modal.com/gpu-glossary/readme)：面向人类的 CUDA 文档

### 12.6 其他

* [**Stas Bekman's Handbook**](https://github.com/stas00/ml-engineering)：涵盖训练大型语言模型各个方面的综合性手册。
* [**Bloom training chronicles**](https://github.com/bigscience-workshop/bigscience/blob/master/train/tr11-176B-ml/chronicles.md)：BLOOM 模型训练过程及挑战的详细文档。
* [**OPT logbook**](https://github.com/facebookresearch/metaseq/blob/main/projects/OPT/chronicles/OPT175B_Logbook.pdf)：Meta 记录 OPT-175B 模型训练过程的详细日志。
* [**Harm's law for training smol models longer**](https://www.harmdevries.com/post/model-size-vs-compute-overhead/)：关于模型大小与训练开销之间关系的调查。
* [**Harm's blog for long context**](https://www.harmdevries.com/post/context-length/)：对长上下文训练在数据和训练成本方面的调查。
* [**GPU Mode**](https://www.youtube.com/@GPUMODE/videos)：一个 GPU 研读小组和社区。
* [**EleutherAI Youtube channel**](https://youtube.com/playlist?list=PLvtrkEledFjqOLuDB_9FWL3dgivYqc6-3&si=fKWPotx8BflLAUkf)：机器学习可扩展性与性能阅读小组
* [**Google Jax Scaling book**](https://jax-ml.github.io/scaling-book/)：如何扩展你的模型
* [**@fvsmassa & @TimDarcet FSDP**](https://github.com/facebookresearch/capi/blob/main/fsdp.py)：独立实现约 500 行代码的 FSDP
* [**thonking.ai**](https://www.thonking.ai/)：霍勒斯·何的一些博客文章——让显卡“嗡嗡”作响
* [**Aleksa's ELI5 Flash Attention**](https://gordicaleksa.medium.com/eli5-flash-attention-5c44017022ad)：Flash Attention 的简易解释
* [**TunibAI's 3D parallelism tutorial**](https://github.com/tunib-ai/large-scale-lm-tutorials)：使用 PyTorch 进行大规模语言建模教程。

## 附录

### A0: 并行编程速成课

在整篇博客文章中，我们将把大语言模型（LLM）的训练规模从一台 GPU 扩展到数百台 GPU。这将需要所有机器之间对权重、梯度和数据进行通信和同步。有一组分布式模式可以实现这一点，称为集体操作（collective operations）。在本节中，我们将对所有这些操作（如广播（Broadcast）、全规约（AllReduce）、分散（Scatter）等）进行一个简要的介绍。让我们开始吧！

一般的设置是，我们有若干个独立的节点，这些节点可以是 CPU 内核、GPU 或者计算节点。每个节点执行一些计算，然后我们想要将结果或其部分内容与其他节点通信，以便进行下一个计算步骤（t+1）。

![image.png|400](https://nanotron-ultrascale-playbook.static.hf.space/assets/images/a0_general.png)
也许我们需要将一个节点的结果发送到所有其他节点，或者我们需要将每个节点的所有中间结果求和以报告总体结果。通常，有一个具有较高地位的节点起着核心作用，在此用 root 表示，它是某些操作的源或目标。让我们从最简单的原语之一开始：广播操作。

#### 广播

一个非常常见的模式是，你在节点 1 上有一些数据，并希望将这些数据共享给所有其他节点，以便它们可以利用这些数据进行一些计算。广播操作正是实现这一目的的。

![image.png|400](https://nanotron-ultrascale-playbook.static.hf.space/assets/images/a0_broadcast.png)
PyTorch 原生提供了集体操作（collective operations），因此我们可以轻松编写一个小示例来演示广播（broadcasting）的工作原理。首先，我们需要使用 `dist.init_process_group` 初始化一个进程组，该函数会设置通信后端（稍后我们会讨论 NCCL），确定存在多少个工作进程（也称为节点），并为每个工作进程分配一个秩（rank，可以通过 `dist.get_rank` 获取）。最后，它会在工作进程之间建立连接。

为了展示 `dist.broadcast` 操作，让我们在 `rank=0` 上创建一个具有非零值的张量，并在其他工作节点上创建全零张量。然后我们使用 `dist.broadcast(tensor, src=0)` 将 `rank=0` 上的张量分发到所有其他工作节点。

```python
import torch
import torch.distributed as dist

def init_process():
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(dist.get_rank())
    
def example_broadcast():
    if dist.get_rank() == 0:
        tensor = torch.tensor([1, 2, 3, 4, 5], dtype=torch.float32).cuda()
    else:
        tensor = torch.zeros(5, dtype=torch.float32).cuda()
    print(f"Before broadcast on rank {dist.get_rank()}: {tensor}")
    dist.broadcast(tensor, src=0)
    print(f"After broadcast on rank {dist.get_rank()}: {tensor}")
    
init_process()
example_broadcast()
```

你可以使用 `torchrun --nproc_per_node=3 dist_op.py` 运行上述脚本（为此你需要 3 块 GPU，或者相应地更改 `nproc_per_node`），你应该会看到以下输出。

```python
Before broadcast on rank 0: tensor([1., 2., 3., 4., 5.], device='cuda:0')
Before broadcast on rank 1: tensor([0., 0., 0., 0., 0.], device='cuda:1')
Before broadcast on rank 2: tensor([0., 0., 0., 0., 0.], device='cuda:2')

After broadcast on rank 0: tensor([1., 2., 3., 4., 5.], device='cuda:0')
After broadcast on rank 1: tensor([1., 2., 3., 4., 5.], device='cuda:1')
After broadcast on rank 2: tensor([1., 2., 3., 4., 5.], device='cuda:2')
```

很好，看起来它按预期工作。请注意，由于我们无法控制哪个打印语句先执行（我们在这里为了便于阅读对它们进行了排序），排名消息可能会以乱序打印出来。现在让我们继续讨论 Reduce 和 AllReduce 模式！

#### Reduce & AllReduce

归约模式是分布式数据处理中最基本的模式之一。其核心思想是通过一个函数 `f()`（例如求和或求平均值）将每个节点上的数据进行合并。在归约范式中，结果仅发送到根节点；而在全归约（AllReduce）情况下，结果会被广播到所有节点。

![image.png|600](https://nanotron-ultrascale-playbook.static.hf.space/assets/images/a0_reduce_allreduce.png)

当然，不存在能够执行此操作的神奇“自由飞行”节点，通常每个节点都会在节点的环形或树形结构中进行部分计算。举个简单的例子：假设我们需要计算每个节点上的数字之和，并且我们的节点以环形模式相连。第一个节点将其数字发送给一个邻居节点，该邻居节点将自身的数字与接收到的数字相加，然后再将其转发给下一个邻居节点。在节点环的一轮传递结束后，第一个节点将接收到总和。

以下是运行一个简单归约操作的代码，用于对张量求和，我们通过 `op=dist.ReduceOp.SUM` 指定要使用的操作（你可以在 [Pytorch 文档](https://pytorch.org/docs/stable/distributed.html#torch.distributed.ReduceOp)中找到支持的更多操作信息）。

```python
def example_reduce():
    tensor = torch.tensor([dist.get_rank() + 1] * 5, dtype=torch.float32).cuda()
    print(f"Before reduce on rank {dist.get_rank()}: {tensor}")
    dist.reduce(tensor, dst=0, op=dist.ReduceOp.SUM)
    print(f"After reduce on rank {rank}: {tensor}")
    
init_process()
example_reduce()
```

请注意，在 Reduce 操作中，只有 `dst` 节点上的张量会被更新。

```python
Before reduce on rank 0: tensor([1., 1., 1., 1., 1.], device='cuda:0')
Before reduce on rank 1: tensor([2., 2., 2., 2., 2.], device='cuda:1')
Before reduce on rank 2: tensor([3., 3., 3., 3., 3.], device='cuda:2')

After reduce on rank 0: tensor([6., 6., 6., 6., 6.], device='cuda:0')
After reduce on rank 1: tensor([2., 2., 2., 2., 2.], device='cuda:1')
After reduce on rank 2: tensor([3., 3., 3., 3., 3.], device='cuda:2')
```

同样，我们可以执行一个 AllReduce 操作（在这种情况下我们不需要指定目标）。

```python
def example_all_reduce():
    tensor = torch.tensor([dist.get_rank() + 1] * 5, dtype=torch.float32).cuda()
    print(f"Before all_reduce on rank {dist.get_rank()}: {tensor}")
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    print(f"After all_reduce on rank {dist.get_rank()}: {tensor}")
    
init_process()
example_all_reduce()
```

在这种情况下，所有节点上都有结果可用。

```python
Before all_reduce on rank 0: tensor([1., 1., 1., 1., 1.], device='cuda:0')
Before all_reduce on rank 1: tensor([2., 2., 2., 2., 2.], device='cuda:1')
Before all_reduce on rank 2: tensor([3., 3., 3., 3., 3.], device='cuda:2')

After all_reduce on rank 0: tensor([6., 6., 6., 6., 6.], device='cuda:0')
After all_reduce on rank 1: tensor([6., 6., 6., 6., 6.], device='cuda:1')
After all_reduce on rank 2: tensor([6., 6., 6., 6., 6.], device='cuda:2')
```

现在让我们转向下一个分布式通信操作。在许多实际情况中，每个节点单独执行许多复杂的计算，我们需要在节点之间共享最终结果。Gather 和 AllGather 是我们在这种情况下想要使用的操作。让我们来看一看！

#### Gather & AllGather

“Gather” 和 “AllGather” 与 “Broadcast” 非常相似，因为它们都允许在节点之间分发数据而不进行修改。与“Broadcast”的主要区别在于，并不是需要从一个节点向所有其他节点共享一个值，而是每个节点都有各自的数据块，我们希望将这些数据块全部收集到一个节点上（在 “Gather” 的情况下），或者将所有数据块收集到所有节点上（在 “AllGather” 的情况下）。一图胜千言，让我们来看一下。

![image.png|600](https://nanotron-ultrascale-playbook.static.hf.space/assets/images/a0_gather_allgather.png)

请注意，虚线表示某些数据实际上根本不会移动（因为它已经存在于节点上）。

在 gather 操作的情况下，我们需要准备一个容器对象，在本例中是 `gather_list`，用于存储收集到的张量。

```python
def example_gather():
    tensor = torch.tensor([dist.get_rank() + 1] * 5, dtype=torch.float32).cuda()
    if dist.get_rank() == 0:
        gather_list = [
            torch.zeros(5, dtype=torch.float32).cuda()
            for _ in range(dist.get_world_size())
            ]
    else:
        gather_list = None
    print(f"Before gather on rank {dist.get_rank()}: {tensor}")
    dist.gather(tensor, gather_list, dst=0)
    if dist.get_rank() == 0:
        print(f"After gather on rank 0: {gather_list}")
    
init_process()
example_gather()
```

我们看到，`gather_list` 确实包含了所有 rank 的张量。

```python
Before gather on rank 0: tensor([1., 1., 1., 1., 1.], device='cuda:0')
Before gather on rank 1: tensor([2., 2., 2., 2., 2.], device='cuda:1')
Before gather on rank 2: tensor([3., 3., 3., 3., 3.], device='cuda:2')

After gather on rank 0: [tensor([1., 1., 1., 1., 1.], device='cuda:0'),
                         tensor([2., 2., 2., 2., 2.], device='cuda:0'),
                         tensor([3., 3., 3., 3., 3.], device='cuda:0')]
```

我们唯一需要为 AllGather 示例更改的是，每个节点都需要一个用于存储结果的占位符。

```python
def example_all_gather():
    tensor = torch.tensor([dist.get_rank() + 1] * 5, dtype=torch.float32).cuda()
    gather_list = [
        torch.zeros(5, dtype=torch.float32).cuda()
        for _ in range(dist.get_world_size())
        ]
    print(f"Before all_gather on rank {dist.get_rank()}: {tensor}")
    dist.all_gather(gather_list, tensor)
    print(f"After all_gather on rank {dist.get_rank()}: {gather_list}")
    
init_process()
example_all_gather()
```

确实我们可以看到，现在每个节点都拥有所有的数据。

```python
Before all_gather on rank 0: tensor([1., 1., 1., 1., 1.], device='cuda:0')
Before all_gather on rank 1: tensor([2., 2., 2., 2., 2.], device='cuda:1')
Before all_gather on rank 2: tensor([3., 3., 3., 3., 3.], device='cuda:2')

After all_gather on rank 0: [tensor([1., 1., 1., 1., 1.], device='cuda:0'),
                             tensor([2., 2., 2., 2., 2.], device='cuda:0'),
                             tensor([3., 3., 3., 3., 3.], device='cuda:0')]
After all_gather on rank 1: [tensor([1., 1., 1., 1., 1.], device='cuda:1'),
                             tensor([2., 2., 2., 2., 2.], device='cuda:0'),
                             tensor([3., 3., 3., 3., 3.], device='cuda:0')]
After all_gather on rank 2: [tensor([1., 1., 1., 1., 1.], device='cuda:2'),
                             tensor([2., 2., 2., 2., 2.], device='cuda:2'),
                             tensor([3., 3., 3., 3., 3.], device='cuda:2')]
```

那么，收集（gather）操作的逆操作是什么呢？在这种情况下，我们会有一个节点上拥有所有数据，并希望将其分配/切分到各个节点，可能还会进行一些中间处理。我们可以使用分散（Scatter）操作，或者在有中间处理操作的情况下，使用归约分散（Reduce Scatter）模式。

#### Scatter & ReduceScatter

正如其名称所隐含的意思，Scatter（分散）操作的目标是将一个节点上的数据切片后分发到所有其他节点。因此，它与 Broadcast（广播）操作不同，后者是复制数据而不进行切片；并且从逻辑上来说，它与 Gather（聚集）操作互为逆操作。

ReduceScatter 模式稍微复杂一些：想象一下，你应用的操作类似于 Reduce 情况中的操作，但与仅将结果移动到一个节点不同，我们还会将其均匀地分发到所有节点。

![image.png|600](https://nanotron-ultrascale-playbook.static.hf.space/assets/images/a0_scatter_reducescatter.png)
“Scatter” 操作在代码中写法与 “Gather” 相反：我们不是准备一个张量列表作为目标，而是将源数据准备成一个我们想要分发的张量列表。我们还需要指定源（`src`）。

```python
def example_scatter():
    if dist.get_rank() == 0:
        scatter_list = [
            torch.tensor([i + 1] * 5, dtype=torch.float32).cuda()
            for i in range(dist.get_world_size())
            ]
        print(f"Rank 0: Tensor to scatter: {scatter_list}")
    else:
        scatter_list = None
    tensor = torch.zeros(5, dtype=torch.float32).cuda()
    print(f"Before scatter on rank {dist.get_rank()}: {tensor}")
    dist.scatter(tensor, scatter_list, src=0)
    print(f"After scatter on rank {dist.get_rank()}: {tensor}")
    
init_process()
example_scatter()
```

因此，我们可以看到空张量是如何被 `scatter_list` 中的内容填充的。

```python
Rank 0: Tensor to scatter: [tensor([1., 1., 1., 1., 1.], device='cuda:0'),
                            tensor([2., 2., 2., 2., 2.], device='cuda:0'),
                            tensor([3., 3., 3., 3., 3.], device='cuda:0')]
Before scatter on rank 0: tensor([0., 0., 0., 0., 0.], device='cuda:0')
Before scatter on rank 1: tensor([0., 0., 0., 0., 0.], device='cuda:1')
Before scatter on rank 2: tensor([0., 0., 0., 0., 0.], device='cuda:2')

After scatter on rank 0: tensor([1., 1., 1., 1., 1.], device='cuda:0')
After scatter on rank 1: tensor([2., 2., 2., 2., 2.], device='cuda:1')
After scatter on rank 2: tensor([3., 3., 3., 3., 3.], device='cuda:2')
```

让我们创建更有趣的数据来演示 ReduceScatter 的逻辑：在每个节点上，我们创建一个包含 2 元素向量的列表，每个向量包含一个幂指数和一个基于节点等级的偏移函数（这有点难以想象，所以请看下面的示例）。

```python
def example_reduce_scatter():
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    input_tensor = [
        torch.tensor([(rank + 1) * i for i in range(1, 3)], dtype=torch.float32).cuda()**(j+1) 
        for j in range(world_size)
        ]
    output_tensor = torch.zeros(2, dtype=torch.float32).cuda()
    print(f"Before ReduceScatter on rank {rank}: {input_tensor}")
    dist.reduce_scatter(output_tensor, input_tensor, op=dist.ReduceOp.SUM)
    print(f"After ReduceScatter on rank {rank}: {output_tensor}")    
    
init_process()
example_reduce_scatter()
```

让我们打印出我们创建的数据模式。我们还立即看到了 ReduceScatter 模式：第一个秩接收了每个节点上第一个张量的总和，第二个秩包含了每个节点上第二个张量的总和，依此类推。

```python
Before ReduceScatter on rank 0: [tensor([1., 2.], device='cuda:0'),
											 tensor([1., 4.], device='cuda:0'),
											 tensor([1., 8.], device='cuda:0')]
Before ReduceScatter on rank 1: [tensor([2., 4.], device='cuda:1'),
                                 tensor([ 4., 16.], device='cuda:1'),
                                 tensor([ 8., 64.], device='cuda:1')]
Before ReduceScatter on rank 2: [tensor([3., 6.], device='cuda:2'),
                                 tensor([ 9., 36.], device='cuda:2'),
                                 tensor([ 27., 216.], device='cuda:2')]

After ReduceScatter on rank 0: tensor([ 6., 12.], device='cuda:0')
After ReduceScatter on rank 1: tensor([14., 56.], device='cuda:1')
After ReduceScatter on rank 2: tensor([ 36., 288.], device='cuda:2')
```

让我们快速了解一下一种常见的 AllReduce 实现，它使用 ReduceScatter 和 AllGather：环形 AllReduce。

#### Ring AllReduce

“环形全规约（Ring AllReduce）”是全规约（AllReduce）的一种具体实现方式，其针对可扩展性进行了优化。与所有设备直接相互通信（这可能会造成通信瓶颈）不同，“环形全规约”可分为两个关键步骤：规约分散（ReduceScatter）和全收集（AllGather）。其工作原理如下：

1. **ReduceScatter**

	- 每个设备将其数据（例如梯度）分割成块，并将其中一个块发送给其邻居设备。同时，每个设备从其另一个邻居设备接收一个块。
	- 当每个设备接收到一个块时，它会将相应的块与接收到的块相加（进行归约操作）。
	- 这一过程在环形网络中持续进行，直到每个设备持有一个部分归约的块，该块代表了所有设备中该块的梯度总和。

2. **AllGather**

	- 现在，每个设备都需要从其他设备收集完全规约后的数据块。
	- 各设备开始将规约后的数据块发送给相邻设备。
	- 每个设备转发其接收到的数据块，直到每个设备都拥有所有完全规约后的数据块，从而使每个设备获得完整的、汇总后的梯度。

让我们通过以下动图来说明这一点，这里有 5 块 GPU，每块 GPU 都有一个长度为 5 的张量。第一个动画展示了 ReduceScatter 步骤，在该步骤结束时，每块 GPU 都会接收到特定数据块（橙色矩形）的归约结果。

![image.png|600](https://nanotron-ultrascale-playbook.static.hf.space/assets/images/a0_reduce_scatter.gif)
下一个动画展示了 AllGather 步骤，在该步骤结束时，每个 GPU 都会获得 AllReduce 操作的完整结果。

![image.png|600](https://nanotron-ultrascale-playbook.static.hf.space/assets/images/a0_all_gather.gif)

你可能已经注意到，在 reduce-scatter 和 all-gather 步骤中，每个 GPU 都会发送和接收 $N-1$ 次数据。每次传输时，每个 GPU 发送 $\frac{K}{N}$ 个值，其中 $K$ 是跨 GPU 累加的数组的总值数。因此，每个 GPU 传输和接收的数据总量是 $2 \times (N-1) \times \frac{K}{N}$。当 $N$（GPU 的数量）很大时，每个 GPU 传输和接收的数据总量大约是 $2 \times K$，其中 $K$ 是参数的总数。

**在 AllReduce 中需要牢记两个关键要点：**

1. 当 $N$（GPU 的数量）较大时，AllReduce 的通信成本大约为 $2×K$。
2. AllReduce 操作可以分解为一个规约-分散（reduce-scatter）操作，后接一个全收集（all-gather）操作。这两个操作的通信开销是 AllReduce 的一半，大约为 $K$。

正如我们所看到的，这种实现方式即使在节点间带宽有限的情况下也能有效利用。

我们现在了解了分布式操作的主要构建模块，但在实际看到它们之前，让我们先来看一种用于同步的特殊操作：屏障（Barrier）。

#### Barrier

屏障（Barrier）是一种简单的操作，用于同步所有节点。在所有节点都到达屏障之前，屏障不会被解除。只有到那时，它们才被允许继续进行后续的计算。

![image.png|400](https://nanotron-ultrascale-playbook.static.hf.space/assets/images/a0_barrier.png)

我们可以通过在每个节点上设置不同的休眠时间来轻松模拟延迟节点，并观察它们全部通过屏障所需的时间。

```python
def example_barrier():
    rank = dist.get_rank()
    t_start = time.time()
    print(f"Rank {rank} sleeps {rank} seconds.")
    time.sleep(rank)  # Simulate different processing times
    dist.barrier()
    print(f"Rank {rank} after barrier time delta: {time.time()-t_start:.4f}")
    
init_process()
example_barrier()
```

我们可以看到，尽管排名第一的（对象）根本没有睡觉，但它通过障碍也花了 2 秒钟。

```python
Rank 0 sleeps 0 seconds.
Rank 1 sleeps 1 seconds.
Rank 2 sleeps 2 seconds.

Rank 0 after barrier time delta: 2.0025
Rank 1 after barrier time delta: 2.0025
Rank 2 after barrier time delta: 2.0024
```

我们需要谨慎对待像这样同步所有节点的操作，因为这违背了并行独立操作的初衷，可能会导致整个处理过程变慢。在许多情况下，如果一个速度快的节点已经开始处理下一个任务也没关系，因为该节点在下一次迭代中可能会变慢，从而在整个过程中平衡掉延迟。

在探讨实际的分布式训练实现之前，让我们先解开一个谜团：NCCL 到底是什么？

#### NCCL: 英伟达集合通信库

在多 GPU 上训练大型模型时，我们有时可能会取得突破，但总会遇到镍（或者 NCCL 🥁）！那是什么？

有几个实现了集体通信的库受 PyTorch 支持：有经典的消息传递接口（MPI），Meta 开发的 Gloo，最后还有英伟达集体通信库（NCCL）。它们在集体通信模式方面都提供了类似的功能，但针对不同的硬件配置进行了优化；NCCL 旨在高效地服务于 GPU - GPU 通信，而 MPI 和 Gloo 则适用于 CPU - CPU 或 CPU - GPU 通信。PyTorch 提供了一份很好的指南来帮助决定使用哪一个。

- GPU training: use NCCL
- CPU training: use Gloo

决策树中有一些细微之处，我们留给读者在上面提到的 PyTorch 指南中去探索。

现在我们已经讲解了分布式训练的基本操作，你现在应该能够轻松地阅读这篇博客文章了。

### A1: 分布式训练分析

#### Kernels

让我们先假设现在这些内核已经集成到 PyTorch 中了。作为简单示例，我们可以看看 PyTorch 中实现的 `torch.nn.functional.layer_norm` 层归一化函数。有几种方法可以对这个函数底层的内核进行性能分析。最直接的方法可能是使用 Python 的 `time` 模块。然而，由于 CUDA 操作是异步的，使用这种方法测量时间只会捕获在 Python 中启动内核所关联的开销，而不是内核本身的实际执行时间。

为了解决这个问题，我们可以利用 `torch.cuda.Event` 进行精确计时，并使用 `torch.cuda.synchronize()` 指令来确保等待内核执行完成。以下代码片段展示了这种方法。

```python
def profile_pytorch(func, input):
    # Create CUDA events to track time. CUDA operations are asynchronous,
    start = torch.cuda.Event(enable_timing=True)  # Event to mark the start time
    end = torch.cuda.Event(enable_timing=True)    # Event to mark the end time
    # Warmup to eliminate any overhead from the first run, which might not reflect 
    # the actual performance.
    for _ in range(10):
        func(input)
    # Record the start time before executing the function
    start.record()  
    func(input)  # Call the function we want to profile
    # Record the end time after the function has completed
    end.record()  
    # Synchronize the CUDA operations to ensure all operations are completed
    # before measuring the elapsed time.
    torch.cuda.synchronize()  
    # Calculate and return the elapsed time in milliseconds.
    return start.elapsed_time(end)
```

一种更有效的性能分析方法是使用之前提到的 PyTorch Profiler。例如，考虑以下代码：

```python
import torch
import torch.nn.functional as F

def pytorch_layer_norm(input):
    return F.layer_norm(input, input.size()[1:])

a = torch.randn(10000, 10000).cuda()

with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,  # Profile CPU activities
        torch.profiler.ProfilerActivity.CUDA,  # Profile CUDA activities
    ],
    # Define a schedule for the profiler
    schedule=torch.profiler.schedule(
        wait=1,      # Wait for 1 iteration before starting to profile
        warmup=3,    # Warm up for 3 iterations to stabilize performance
        active=2,    # Profile for 2 active iterations
        repeat=1,    # Repeat the profiling schedule once
    ),
    on_trace_ready=torch.profiler.tensorboard_trace_handler('.'),
    
) as p:
    for iter in range(10):
        pytorch_layer_norm(a)
        p.step()

# Print a table of the profiling results, sorted by total CUDA time, limited to the top 10 entries
print(p.key_averages().table(sort_by="cuda_time_total", row_limit=8))
```

这将打印按总 CUDA 时间排序的聚合分析结果，输出内容如下：

![image.png](https://nanotron-ultrascale-playbook.static.hf.space/assets/images/a1_kernels.png)

你也可以尝试按照我们之前所说的，在 chrome://tracing/ 中检查跟踪信息。

> [!tip]
> 如果你是初次使用此工具，可以通过左右箭头键浏览轨迹。此外，按住 Alt 键的同时用鼠标左右滚动可以放大或缩小。

放大后，您可以在此跟踪中观察到调用 `layer_norm` 时的操作流程。

![image.png](https://nanotron-ultrascale-playbook.static.hf.space/assets/images/a1_profile_trace.png)

The sequence begins in the CPU (the upper section) with `aten::layer_norm`, progressing to `aten::native_layer_norm`, and then transitioning to `cudaLaunchKernel`. From there, we move on to the GPU, where the `vectorized_layer_norm_kernel` kernel is called.

该序列从 CPU（上部）开始，以 `aten::layer_norm` 启动，接着进入 `aten::native_layer_norm`，然后过渡到 `cudaLaunchKernel`。从那里开始，我们继续到 GPU，在那里调用 `vectorized_layer_norm_kernel` 内核。

> [!NOTE]
> 您可以通过在分析器中将 `profile_memory` 设置为 `True` 来启用内存分析。然而，这可能会导致更复杂的跟踪记录。

虽然 PyTorch Profiler 能快速提供性能概览，但 NVIDIA Nsight Compute (ncu) 能更深入地洞察 GPU 性能，包括每个内核的详细执行时间和内存使用情况。运行该分析器非常简单：

```bash
ncu --set full python layer_norm.py
```

其中 `layer_norm.py` 是一个直接执行层归一化函数的简单文件。此命令将生成日志输出，但更有效查看结果的方法是通过设置输出标志：

```bash
ncu --set full -o output python layer_norm.py
```

并使用 Nsight Compute 打开 `output.ncu-rep` 文件，您将看到如下视图。

![image.png](https://nanotron-ultrascale-playbook.static.hf.space/assets/images/a1_ncu.png)

明确警告计算和内存利用率情况，以及如何让内核更好地平衡计算和内存以实现最大占用率。

#### CPP extension

如果你想要分析的内核尚未集成到 PyTorch 中，可以使用 PyTorch 的 `cpp_extension` 模块轻松编译和运行自定义 CUDA 代码。这个过程很简单——只需在 `.cu` 文件中创建你的 CUDA 内核，并使用 `cpp_extension` 模块中的 `load` 函数在 Python 中加载它。

对于一个简单的 `add` 内核，`.cu` 文件可能如下所示：

```clike
#include 
#include 
#include 

__global__ void add_kernel(float* x, float* y, float* output, int size) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size) {
        output[index] = x[index] + y[index];
    }
}

void add_cuda(torch::Tensor x, torch::Tensor y, torch::Tensor output) {
    int threads = 1024;
    int blocks = (x.size(0) + threads - 1) / threads;

    add_kernel<<>>(x.data_ptr(), y.data_ptr(), output.data_ptr(), x.size(0));
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("add_cuda", &add_cuda, "Vector addition (CUDA)");
}
```

python 文件加载内核：

```python
import torch
from torch.utils.cpp_extension import load

# Load and compile the CUDA extension
vector_add = load(
    name="vector_add",
    sources=["add_kernel.cu"],
    verbose=True
)

# Define input tensors
size = 10000
x = torch.randn(size, device='cuda')
y = torch.randn(size, device='cuda')
output = torch.empty(size, device='cuda')

# Run the CUDA kernel
vector_add.add_cuda(x, y, output)
```

使用这种方法，你可以像我们之前用 PyTorch 的分析器或 NVIDIA 工具展示的那样对自定义 CUDA 内核进行分析。

### A2: 典型的 LLM 训练规模

让我们来了解一下大型语言模型（LLM）训练中典型事物的规模大小。当我们谈论内存或计算量时，通常是在统计“元素”——可以将这些元素想象成张量中的数字。要得到以字节为单位的实际内存大小，你需要将元素数量乘以每个数字的大小（例如，bf16 格式每个数字占 2 字节，fp32 格式每个数字占 4 字节）。

以下是一些大致的估算数据

- 输入令牌（Input tokens）：对于每个批次，我们处理 seq·mbs 个令牌，其中 mbs 是微批次大小，seq 是序列长度。
- 激活值（隐藏状态）（Activations (hidden states)）：对于单层而言，隐藏状态张量的大小为 seq·mbs·h 个元素。
- 模型权重和梯度（Model weights and gradients）：模型中的每个权重矩阵（如线性层中的）大约有 h² 个元素。这是针对每个权重矩阵而言的。梯度的大小与权重相同。
- 优化器状态（Optimizer states）：对于每个权重矩阵（有 h² 个元素），如果你使用的是像 Adam 这样采用混合精度训练的优化器，它会以 fp32 精度（2·h²）保存动量和方差状态，另外还有 fp32 精度的主权重（h²）。因此，每个权重矩阵的总优化器状态大约为 (6·h²)。
- 模型总参数（Total model parameters）：对于每个Transformer 块：
    - 注意力参数（Attention parameters）：
        - QKV 投影（QKV projections）：3h² 个参数
        - 输出投影（Output projection）：h² 个参数
    - 带有门控线性单元（GLU）的多层感知机（MLP）参数（MLP parameters with GLU）：
        - 门控和上投影（Gate and up projections）：8h² 个参数（2 个大小为 h×4h 的矩阵）
        - 下投影（Down projection）：4h² 个参数（1 个大小为 4h×h 的矩阵）
    - 每个块的总参数：带有 GLU 的 MLP 时为 16h²，不带有 GLU 时为 12h²
    - 对于完整模型：带有 GLU 时为 16h²·num_layers
- 额外参数（Additional parameters）：
    - 输入嵌入（Input embeddings）：vocab_size·h
    - 语言模型头（LM head）：vocab_size·h（如果未与输入嵌入共享）
    - 位置嵌入（如果使用）（Positional embeddings (if used)）：max_seq_len·h
- 前向和反向传播计算量（FLOPs）（Forward and backward pass compute (FLOPs)）：对前向传播的计算量进行非常粗略的估计是 2·num_tokens·num_params。反向传播的计算量是前向传播的两倍，即 4·num_tokens·num_params。

### A3: 计算/通信重叠的数学原理

利用上一节中的公式，我们可以估算在分布式训练中计算和通信何时能够有效重叠。以数据并行（Zero-0）为例。

#### 数据并行通信分析

需要通信的总梯度大小为：

- Gradients = Parameters ≈ num_layers⋅16h2num_layers⋅16h2

在反向传播过程中，这些梯度以桶（默认大小为 25MB）为单位进行传递。对每个桶执行全规约操作的通信时间为：$$t_{comm} = t_{comm\_bucket} = \frac{bucket\_size \cdot 2(DP - 1)}{DP \cdot peak\_bw}
$$
在进行带宽计算时，我们使用 [NCCL 文档](https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md#summary)中的总线带宽公式。这些公式在计算GPU之间的有效带宽时会考虑特定的通信模式。

反向传播的计算时间：$$t_{compute} = \frac{4 \cdot num\_tokens \cdot num\_params}{peak\_flops}
$$
为了实现有效重叠，我们需要：$$\frac{t_{comm}}{t_{compute}} = \frac{num\_params}{2 \cdot num\_tokens} \cdot \frac{DP - 1}{DP} \cdot \frac{peak\_flops}{peak\_bw} \leq 1
$$
这个比率有助于确定通信是否会成为训练中的瓶颈。当该比率小于 1 时，通信可以与计算完全重叠。

#### ZeRO-3 (FSDP) 通信分析

对于 ZeRO-3，参数和梯度会在多个 GPU 上进行分片。让我们分析一个每个 Transformer 模块大小为 $16h^2$ 参数的模型的通信模式。

* 对于每个 Transformer 块的前向传播：
	* Allgather 参数：每个 rank 需要传输 $\frac{16h^2}{DP}$ 字节。
* 对于每个 Transformer 块的反向传播：
	- Allgather 参数：每个 rank 需要传输 $\frac{16h^2}{DP}$ 字节。
	- Reducescatter 梯度：每个 rank 需要传输 $\frac{16h^2}{DP}$ 字节。
- 每个块的总通信量：$3 \times \frac{16h^2}{DP}$ 字节。
- 整个模型的总通信量：$3 \times num\_layers \times \frac{16h^2}{DP}$ 字节。

Allgather 操作的通信时间为：$$
t_{comm} = \frac{16h^2 \cdot (DP - 1)}{DP \cdot peak\_bw}$$一个解码层前向传播的计算时间为：$$
t_{compute} = \frac{32 \cdot seq\_len \cdot mbs \cdot h^2}{peak\_flops}$$为了有效地重叠计算和通信，我们需要：$$
\frac{t_{comm}}{t_{compute}} = \frac{2 \cdot seq\_len \cdot mbs \cdot (DP - 1)}{DP \cdot peak\_flops \cdot peak\_bw} \leq 1$$当这个比率小于1时，下一层参数的通信可以隐藏在当前层的计算之后。

#### TP 通信分析

对于张量并行（TP），在线性层中激活值在 GPU 之间分片。让我们分析通信模式：

* 对于前向传播中的每个列线性层：
	- Allgather 激活：每个 rank 需要传输 $\frac{seq \cdot mbs \cdot h}{TP}$ 字节。
- 对于反向传播中的每个列线性层：
	- Reducescatter 梯度：每个 rank 需要传输 $\frac{seq \cdot mbs \cdot h}{TP}$ 字节。
- 对于行线性层也是如此。每个 Transformer 块有 2 个列线性层和 2 个行线性层。
- 每个块的总通信量：$\frac{8 \cdot seq \cdot mbs \cdot h}{TP}$ 字节。
- 整个模型的总通信量：$\frac{8 \cdot num\_layers \cdot seq \cdot mbs \cdot h}{TP}$ 字节。

让我们分析是否可以将一层的 allgather 通信与下一线性层的计算重叠。Allgather 操作的通信时间为：$$
t_{comm} = \frac{seq \cdot mbs \cdot h \cdot (TP - 1)}{TP \cdot peak\_bw}$$而下一线性层（参数为 $h^2$ ）的计算时间为：$$
t_{compute} = \frac{2 \cdot seq \cdot mbs \cdot h^2}{TP \cdot peak\_flops}$$为了有效重叠，我们希望通信时间小于计算时间：$$
\frac{t_{comm}}{t_{compute}} = \frac{TP - 1}{2 \cdot h} \cdot \frac{peak\_flops}{peak\_bw} \leq 1$$这个比率告诉我们是否可以成功地将 allgather 通信隐藏在下一线性层的计算之后。有趣的是，这个比率仅依赖于隐藏层大小 $h$ 和张量并行度 $TP$，而不依赖于序列长度或批次大小。

#### PP 通信分析

对于流水线并行（PP），激活值和梯度在流水线阶段之间进行通信。让我们分析通信模式：

* 对于前向传播中的每个微批次：
	- 接收和发送激活：$2 \cdot seq \cdot mbs \cdot h$ 字节。
- 对于反向传播中的每个微批次：
	- 接收和发送梯度：$2 \cdot seq \cdot mbs \cdot h$ 字节。
- 每个微批次的总通信量：$4 \cdot seq \cdot mbs \cdot h$ 字节。
- 对于梯度累积步骤（gas），总通信量为：$4 \cdot gas \cdot seq \cdot mbs \cdot h$ 字节。

让我们分析是否可以将激活/梯度的通信与下一个 Transformer 块的计算重叠。下一个流水线阶段中 Transformer 块的计算时间为：$$
t_{compute} = \frac{32 \cdot seq \cdot mbs \cdot h^2 \cdot num\_layers\_in\_next\_pp}{peak\_flops}$$而 P2P 传输的通信时间为：$$
t_{comm} = \frac{seq \cdot mbs \cdot h}{peak\_bw}$$为了有效重叠，我们希望：$$
\frac{t_{comm}}{t_{compute}} = \frac{peak\_flops}{32 \cdot h \cdot num\_layers\_in\_next\_pp \cdot peak\_bw} \leq 1$$与 TP 类似，这个比率不依赖于序列长度和批次大小。它依赖于隐藏层大小 $h$、下一个流水线阶段的层数，以及硬件的计算与 P2P 带宽能力比。

### 参考文献

2. Domino: Eliminating Communication in LLM Training via Generic Tensor Slicing and Overlapping  [[PDF]](http://arxiv.org/pdf/2409.15241.pdf)  
    Wang, G., Zhang, C., Shen, Z., Li, A. and Ruwase, O., 2024.
3. Striped Attention: Faster Ring Attention for Causal Transformers  [[PDF]](http://arxiv.org/pdf/2311.09431.pdf)  
    Brandon, W., Nrusimha, A., Qian, K., Ankner, Z., Jin, T., Song, Z. and Ragan-Kelley, J., 2023.
4. Breadth-First Pipeline Parallelism  [[PDF]](http://arxiv.org/pdf/2211.05953.pdf)  
    Lamy-Poirier, J., 2023.
5. DeepSeek-V3 Technical Report  [[PDF]](http://arxiv.org/pdf/2412.19437.pdf)  
    DeepSeek-AI, and others,, 2024.
6. Zero Bubble Pipeline Parallelism  [[PDF]](http://arxiv.org/pdf/2401.10241.pdf)  
    Qi, P., Wan, X., Huang, G. and Lin, M., 2023.
7. Mixtral of Experts  [[PDF]](http://arxiv.org/pdf/2401.04088.pdf)  
    Jiang, A.Q., Sablayrolles, A., Roux, A., Mensch, A., Savary, B., Bamford, C., Chaplot, D.S., Casas, D.d.l., Hanna, E.B., Bressand, F., Lengyel, G., Bour, G., Lample, G., Lavaud, L.R., Saulnier, L., Lachaux, M., Stock, P., Subramanian, S., Yang, S., Antoniak, S., Scao, T.L., Gervet, T., Lavril, T., Wang, T., Lacroix, T. and Sayed, W.E., 2024.
8. Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity  [[PDF]](http://arxiv.org/pdf/2101.03961.pdf)  
    Fedus, W., Zoph, B. and Shazeer, N., 2022.
9. A Survey on Mixture of Experts  [[PDF]](http://arxiv.org/pdf/2407.06204.pdf)  
    Cai, W., Jiang, J., Wang, F., Tang, J., Kim, S. and Huang, J., 2024.
10. GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding  [[PDF]](http://arxiv.org/pdf/2006.16668.pdf)  
    Lepikhin, D., Lee, H., Xu, Y., Chen, D., Firat, O., Huang, Y., Krikun, M., Shazeer, N. and Chen, Z., 2020.
11. FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness  [[PDF]](http://arxiv.org/pdf/2205.14135.pdf)  
    Dao, T., Fu, D.Y., Ermon, S., Rudra, A. and Ré, C., 2022.
12. FP8-LM: Training FP8 Large Language Models  [[PDF]](http://arxiv.org/pdf/2310.18313.pdf)  
    Peng, H., Wu, K., Wei, Y., Zhao, G., Yang, Y., Liu, Z., Xiong, Y., Yang, Z., Ni, B., Hu, J., Li, R., Zhang, M., Li, C., Ning, J., Wang, R., Zhang, Z., Liu, S., Chau, J., Hu, H. and Cheng, P., 2023.
13. torchao: PyTorch native quantization and sparsity for training and inference  [[link]](https://github.com/pytorch/ao)  
    maintainers, t. and contributors,, 2024.
14. Small-scale proxies for large-scale Transformer training instabilities  [[PDF]](http://arxiv.org/pdf/2309.14322.pdf)  
    Wortsman, M., Liu, P.J., Xiao, L., Everett, K., Alemi, A., Adlam, B., Co-Reyes, J.D., Gur, I., Kumar, A., Novak, R., Pennington, J., Sohl-dickstein, J., Xu, K., Lee, J., Gilmer, J. and Kornblith, S., 2023.

[^1]: An Empirical Model of Large-Batch Training  [PDF](http://arxiv.org/pdf/1812.06162.pdf)  McCandlish, S., Kaplan, J., Amodei, D. and Team, O.D., 2018.
[^2]: Mixed Precision Training  [PDF](http://arxiv.org/pdf/1710.03740.pdf)  Micikevicius, P., Narang, S., Alben, J., Diamos, G., Elsen, E., Garcia, D., Ginsburg, B., Houston, M., Kuchaiev, O., Venkatesh, G. and Wu, H., 2018.
[^3]: Reducing Activation Recomputation in Large Transformer Models  [PDF](http://arxiv.org/pdf/2205.05198.pdf)  Korthikanti, V., Casper, J., Lym, S., McAfee, L., Andersch, M., Shoeybi, M. and Catanzaro, B., 2022.
