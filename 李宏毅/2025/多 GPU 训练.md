
https://zhuanlan.zhihu.com/p/648924115


**Attention 层单步中间激活值显存表**

| 计算步骤                                  | 中间激活                   | 形状               | 占用显存             |
| ------------------------------------- | ---------------------- | ---------------- | ---------------- |
| $Q/K/V = xW_{Q/K/V}$                  | $x$                    | $[b, s, h]$      | $2bsh$           |
| $QK^T$                                | $Q, K$                 | $[b, a, s, h/a]$ | $2 \cdot 2bsh$   |
|                                       |                        | $[b, a, h/a, s]$ |                  |
| $\text{score} = \text{softmax}(QK^T)$ | $QK^T$                 | $[b, a, s, s]$   | $2bs^2a$         |
| $\text{dropout()}$                    | $\text{dropout\_mask}$ | $[b, a, s, s]$   | $bs^2a$          |
| $x = \text{score} \cdot V$            | $\text{score V}$       | $[b, a, s, s]$   | $2bs^2a + 2bsh$  |
|                                       |                        | $[b, a, h/a, s]$ |                  |
| $xW_o$                                | $x$                    | $[b, a, s, h/a]$ | $2bsh$           |
| $\text{dropout()}$                    | $\text{dropout\_mask}$ | $[b, s, h]$      | $bsh$            |
| $\text{layernorm()}$                  | $x$                    | $[b, s, h]$      | $2bsh$           |
| $\text{SUM}$                          |                        |                  | $13bsh + 5bs^2a$ |

**MLP 层单步中间激活值显存表**

| 计算步骤                          | 中间激活          | 形状            | 占用显存    |
|---------------------------------|-----------------|---------------|----------|
| $xW_{\text{up}}$              | $x$           | $[b, s, h]$ | $2bsh$  |
| $f_{\text{gelu}}(xW_{\text{up}})$ | $xW_{\text{up}}$ | $[b, s, 4h]$ | $8bsh$  |
| $xW_{\text{down}}$            | $x$           | $[b, s, 4h]$ | $8bsh$  |
| $\text{dropout}$              | $\text{dropout\_mask}$ | $[b, s, h] \text{ int}$ | $bsh$   |
| $\text{layernorm}$            | $x$           | $[b, s, h]$ | $2bsh$  |
| $\text{Sum}$                  |                 |               | $21bsh$ |

-----

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


