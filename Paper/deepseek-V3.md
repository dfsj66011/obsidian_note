
### 摘要 

我们推出了 DeepSeek-V3，这是一款强大的 MoE 语言模型，总参数量为 671B，每个 token 激活 37B 参数。为了实现高效推理和成本效益的训练，DeepSeek-V3 采用了 MLA 和 DeepSeekMoE 架构，这些架构已在 DeepSeek-V2 中经过充分验证。此外，DeepSeek-V3 开创了一种无辅助损失的负载平衡策略，并设定了多 token 预测训练目标以提升性能。我们在 14.8T 个多样且高质量的 tokens 上预训练 DeepSeek-V3，随后进行监督微调和强化学习阶段以充分发挥其能力。综合评估显示，DeepSeek-V3 优于其他开源模型，并达到了与领先的闭源模型相当的性能。尽管性能卓越，DeepSeek-V3 的完整训练仅需 2.788 M H800 GPU 时间。此外，其训练过程非常稳定。在整个训练过程中，我们没有遇到不可恢复的损失峰值，也没有进行任何回滚。模型检查点可在 https://github.com/deepseek-ai/DeepSeek-V3 获取。

<img src="https://arxiv.org/html/2412.19437v2/x1.png" width="500">**图 1：** DeepSeek-V3 及其对应模型的基准性能。


### 1、简介

近年来，LLMs 快速迭代和演变，逐步缩小与 AGI 之间的差距。除了闭源模型，开源模型也在取得显著进展，包括 DeepSeek 系列、LLaMA 系列、Qwen 系列和 Mistral 系列，努力缩小与闭源模型的差距。为了进一步推动开源模型能力的边界，我们扩展了模型规模，推出了 DeepSeek-V3，一个拥有671B 参数的大型 MoE 模型，其中每个 token 激活 37B 参数。

我们始终以前瞻性的视角追求强大的模型性能和经济成本。因此，在架构上，DeepSeek-V3 仍采用 MLA 以实现高效推理，并使用 DeepSeekMoE 进行成本效益高的训练。这两种架构在 DeepSeek-V2 中已被验证，证明了它们在保持强大模型性能的同时实现高效训练和推理的能力。除了基本架构，我们还实施了两种额外策略以进一步增强模型能力。首先，DeepSeek-V3 开创了一种无辅助损失策略以实现负载平衡，旨在最大限度地减少鼓励负载平衡对模型性能的不利影响。其次，DeepSeek-V3 采用多标记预测训练目标，我们观察到这提升了评估基准的整体性能。

为了实现高效训练，我们支持 FP8 混合精度训练，并对训练框架进行了全面优化。低精度训练已成为实现高效训练的有前途的解决方案，其演变与硬件能力的进步密切相关。在这项工作中，我们引入了 FP8 混合精度训练框架，并首次验证了其在超大规模模型上的有效性。通过支持 FP8 计算和存储，我们实现了加速训练和减少 GPU 内存使用。至于训练框架，我们设计了 DualPipe 算法以实现高效的流水线并行，通过计算-通信重叠减少流水线气泡并隐藏大部分训练过程中的通信。这种重叠确保了随着模型进一步扩展，只要我们保持恒定的计算与通信比率，就可以跨节点使用细粒度专家，同时实现接近零的全互通信开销。此外，我们还开发了高效的跨节点全互通信内核，以充分利用 InfiniBand（IB）和 NVLink 带宽。此外，我们精心优化了内存占用，使得在不使用昂贵的张量并行的情况下训练 DeepSeek-V3 成为可能。结合这些努力，我们实现了高训练效率。

在预训练期间，我们在 14.8T 高质量且多样化的标记上训练 DeepSeek-V3。预训练过程非常稳定。整个训练过程中，我们没有遇到不可恢复的损失峰值，也不需回滚。接下来，我们对 DeepSeek-V3 进行两阶段的上下文长度扩展。第一阶段将最大上下文长度扩展到 32K，第二阶段进一步扩展到 128K。随后，我们进行后训练，包括对 DeepSeek-V3 基础模型的 SFT 和 RL，以使其与人类偏好对齐并进一步释放其潜力。在后训练阶段，我们从 DeepSeek-R1 系列模型中提炼推理能力，同时仔细保持模型准确性和生成长度之间的平衡。

我们在一系列综合基准上评估 DeepSeek-V3。尽管其训练成本经济，综合评估显示，DeepSeek-V3-Base 已成为目前最强的开源基础模型，尤其在代码和数学方面。其聊天版本也优于其他开源模型，并在一系列标准和开放式基准上实现了与领先的闭源模型（包括 GPT-4o 和 Claude-3.5-Sonnet）相当的性能。

![[Pasted image 20250311104429.png|500]]
**表 1：** DeepSeek-V3 的训练成本（假设 H800 的租赁价格为每 GPU 小时 2 美元）。


最后，我们再次强调 DeepSeek-V3 的经济训练成本，详见表 1。通过优化算法、框架和硬件的协同设计实现。在预训练阶段，训练 DeepSeek-V3 每万亿个 token 仅需 180K H800 GPU 小时，即在我们 2048 个 H800 GPU 的集群上耗时 3.7 天。因此，我们在不到两个月内完成了预训练阶段，耗费 2664K GPU 小时。加上上下文长度扩展的 119K GPU 小时和后训练的 5K GPU 小时，DeepSeek-V3 的完整训练仅需 2.788M GPU 小时。假设 H800 GPU 的租赁价格为每 GPU小时 2 美元，我们的总训练成本仅为 5.576M 美元。请注意，上述成本仅包括 DeepSeek-V3 的正式训练，不包括与架构、算法或数据的前期研究和消融实验相关的成本。

我们的主要贡献包括：

**架构：创新的负载均衡策略和训练目标**

* 在 DeepSeek-V2 高效架构的基础上，我们首创了一种无辅助损失的负载均衡策略，最大限度地减少了因鼓励负载均衡而导致的性能下降。
* 我们研究了一种多标记预测（MTP）目标，并证明其对模型性能有益。它还可用于推理加速的投机解码。

**预训练：追求极致的训练效率**

* 我们设计了一个 FP8 混合精度训练框架，并首次验证了 FP8 训练在超大规模模型上的可行性和有效性。
* 通过算法、框架和硬件的协同设计，我们克服了跨节点 MoE 训练中的通信瓶颈，实现了几乎完整的计算通信重叠。这大大提高了我们的训练效率，降低了训练成本，使我们能够在不增加额外开销的情况下进一步扩大模型规模。
* 以仅 2.664M H800 GPU 小时的经济成本，我们完成了对 14.8T 标记的 DeepSeek-V3 预训练，生成了目前最强的开源基础模型。预训练后的后续训练阶段仅需 0.1M GPU 小时。

**后训练：从 DeepSeek-R1 中进行知识蒸馏**

- 我们引入了一种创新的方法，将长链思维（CoT）模型的推理能力，特别是来自 DeepSeek R1 系列模型之一的能力，蒸馏到标准大型语言模型中，尤其是 DeepSeek-V3。我们的流程巧妙地将 R1 的验证和反思模式整合到 DeepSeek-V3 中，显著提升了其推理性能。同时，我们也保持了对 DeepSeek-V3 输出风格和长度的控制。

**核心评估结果总结**

- **知识**：在教育基准测试中，如 MMLU、MMLU-Pro 和 GPQA，DeepSeek-V3 超越了所有其他开源模型，在 MMLU 上取得了 88.5 分，在 MMLU-Pro 上取得了 75.9 分，在 GPQA 上取得了 59.1 分。其表现与领先的闭源模型（如 GPT-4o 和 Claude-Sonnet-3.5）相当，缩小了开源和闭源模型在该领域的差距。在事实性基准测试中，DeepSeek-V3 在 SimpleQA 和 Chinese SimpleQA 上表现出色，领先于所有开源模型。虽然在英语事实知识（SimpleQA）上落后于 GPT-4o 和 Claude-Sonnet-3.5，但在中文事实知识（Chinese SimpleQA）上超越了这些模型，突显其在中文事实知识方面的优势。

- **代码、数学和推理**：在数学相关的基准测试中，DeepSeek-V3 在所有非长链思维的开源和闭源模型中达到了最先进的表现。值得注意的是，它在某些基准测试（如 MATH-500）中甚至超越了 o1-preview，展现了其强大的数学推理能力。在编码相关任务中，DeepSeek-V3 成为编码竞赛基准（如 LiveCodeBench）的表现最佳模型，巩固了其在该领域的领先地位。在工程相关任务中，尽管 DeepSeek-V3 略低于 Claude-Sonnet-3.5，但仍以显著优势超越其他所有模型，展示了其在各种技术基准中的竞争力。


在本文的剩余部分中，我们首先详细介绍 DeepSeek-V3 模型架构（第 2 节）。随后，我们介绍基础设施，包括计算集群、训练框架、FP8 训练支持、推理部署策略以及对未来硬件设计的建议。接下来，我们描述预训练过程，包括训练数据的构建、超参数设置、长上下文扩展技术、相关评估以及一些讨论（第 4 节）。之后，我们讨论后训练工作，包括 SFT、RL、相应的评估和讨论（第 5 节）。最后，我们总结此项工作，讨论 DeepSeek-V3 的现有局限性，并提出未来研究的潜在方向（第 6 节）。

### 2、架构

我们首先介绍 DeepSeek-V3 的基本架构，其特点是使用 MLA 以提高推理效率，以及 DeepSeekMoE 以降低训练成本。接下来，我们介绍 MTP 训练目标，我们观察到这能提高在评估基准上的整体性能。对于其他未明确提到的细节，DeepSeek-V3 遵循 DeepSeek-V2 的设置。
<img src="https://arxiv.org/html/2412.19437v2/x2.png" width="500">
**图 2.**  DeepSeek-V3 基本架构的示意图。继承 DeepSeek-V2，我们采用 MLA 和 DeepSeekMoE 以实现高效推理和经济训练。


#### 2.1 基础架构

DeepSeek-V3 的基本架构仍然基于 Transformer 框架。为了实现高效的推理和经济的训练，DeepSeek-V3 也采用了 MLA 和 DeepSeekMoE，这些在 DeepSeek-V2 中已经经过充分验证。与 DeepSeek-V2 相比，我们额外引入了一种无辅助损失的负载平衡策略，以缓解为确保负载平衡而导致的性能下降。图 2 展示了 DeepSeek-V3 的基本架构，我们将在本节中简要回顾 MLA 和 DeepSeekMoE 的细节。

##### 2.1.1 多头潜在注意力

对于注意力机制，DeepSeek-V3 采用了 MLA 架构。设 $d$ 为嵌入维度，$n_h$ 为注意力头的数量，$d_h$ 为每个头的维度，$\mathbf{h}_{t} \in \mathbb{R}^{d}$ 为在给定注意力层中第 $t$ 个标记的注意力输入。MLA 的核心是对注意力键和值进行低秩联合压缩，以减少推理过程中的键值（KV）缓存：$$
\begin{align}
    \boxed{\color{blue} \mathbf{c}_{t}^{KV}} &= W^{DKV} \mathbf{h}_{t}, \tag{1} \\[1.2ex]
    [\mathbf{k}_{t, 1}^{C};\mathbf{k}_{t, 2}^{C};...;\mathbf{k}_{t, n_{h}}^{C}] = \mathbf{k}_{t}^{C} &= W^{UK} \mathbf{c}_{t}^{KV},\tag{2} \\[1.2ex]
    \boxed{\color{blue}\mathbf{k}_{t}^{R}} &= \operatorname{RoPE}({W^{KR}} \mathbf{h}_{t}), \tag{3}\\[1.2ex]
    \mathbf{k}_{t, i} &= [\mathbf{k}_{t, i}^{C}; \mathbf{k}_{t}^{R}],\tag{4} \\[1.2ex]
    [\mathbf{v}_{t, 1}^{C};\mathbf{v}_{t, 2}^{C};...;\mathbf{v}_{t, n_{h}}^{C}] = \mathbf{v}_{t}^{C} &= W^{UV} \mathbf{c}_{t}^{KV}, \tag{5}
\end{align}$$其中 $\mathbf{c}_{t}^{KV} \in \mathbb{R}^{d_c}$ 是键和值的压缩潜在向量；$d_c (\ll d_h n_h)$ 表示 KV 压缩维度；$W^{DKV} \in \mathbb{R}^{d_c \times d}$ 是降维投影矩阵；$W^{UK},W^{UV} \in \mathbb{R}^{d_h n_h \times d_c}$ 分别是键和值的升维投影矩阵；$W^{KR} \in \mathbb{R}^{d_h^R \times d}$ 是用于生成带有旋转位置嵌入(RoPE) 的解耦键的矩阵；$\operatorname{RoPE}(\cdot)$ 表示应用 RoPE 矩阵的操作；$[\cdot;\cdot]$ 表示连接。

注意，对于 MLA，仅需在生成过程中缓存蓝色框中的向量（即 $\color{blue} \mathbf{c}_{t}^{KV}$ 和 $\color{blue}\mathbf{k}_{t}^{R}$），这显著减少了 KV 缓存，同时保持了与标准多头注意力（MHA）相当的性能。

对于注意力查询，我们也进行低秩压缩，以减少训练期间的激活内存：$$
\begin{align}
    \mathbf{c}_{t}^{Q} &= W^{DQ} \mathbf{h}_{t}, \tag{6} \\[1.2ex]
    [\mathbf{q}_{t, 1}^{C};\mathbf{q}_{t, 2}^{C};...;\mathbf{q}_{t, n_{h}}^{C}] = \mathbf{q}_{t}^{C} &= W^{UQ} \mathbf{c}_{t}^{Q}, \tag{7} \\[1.2ex]
    [\mathbf{q}_{t, 1}^{R};\mathbf{q}_{t, 2}^{R};...;\mathbf{q}_{t, n_{h}}^{R}] = \mathbf{q}_{t}^{R} &= \operatorname{RoPE}({W^{QR}} \mathbf{c}_{t}^{Q}), \tag{8} \\[1.2ex]
    \mathbf{q}_{t, i} &= [\mathbf{q}_{t, i}^{C}; \mathbf{q}_{t, i}^{R}],\tag{9}
\end{align}$$其中 $\mathbf{c}_{t}^{Q} \in \mathbb{R}^{d_c^{\prime}}$ 是查询的压缩潜在向量；$d_c^{\prime} (\ll d_h n_h)$ 表示查询压缩维度；$W^{DQ} \in \mathbb{R}^{d_c^{\prime} \times d}, W^{UQ} \in \mathbb{R}^{d_h n_h \times d_c^{\prime}}$ 分别是查询的降维和升维投影矩阵；$W^{QR} \in \mathbb{R}^{d_h^R n_h \times d_c^{\prime}}$ 是用于生成带有 RoPE 的解耦查询的矩阵。

最终，注意力查询（$\mathbf{q}_{t, i}$）、键（$\mathbf{k}_{j, i}$）和值（$\mathbf{v}_{j, i}^{C}$）结合起来，得到最终的注意力输出 $\mathbf{u}_{t}$：$$
\begin{align}
    \mathbf{o}_{t, i} &= \sum_{j=1}^{t} \operatorname{Softmax}_j\left(\frac{\mathbf{q}_{t, i}^T \mathbf{k}_{j, i}}{\sqrt{d_{h} + d_{h}^{R}}}\right) \mathbf{v}_{j, i}^{C}, \tag{10} \\[1.2ex]
    \mathbf{u}_{t} &= W^{O} [\mathbf{o}_{t, 1};\mathbf{o}_{t, 2};...;\mathbf{o}_{t, n_{h}}],\tag{11}
\end{align}$$其中 $W^{O} \in \mathbb{R}^{d \times d_h n_h}$ 表示输出投影矩阵。


##### 2.1.2 DeepSeekMoE 的无辅助损失负载均衡

**DeepSeekMoE 的基本架构**

对于 FFNs，DeepSeek-V3 使用了 DeepSeekMoE 架构。与 GShard 等传统 MoE 架构相比，DeepSeekMoE 使用更细粒度的专家，并隔离出一些专家作为共享专家。设 $\mathbf{u}_{t}$ 表示第 $t$ 个标记的 FFN 输入，我们计算 FFN 输出 $\mathbf{h}_{t}^{\prime}$ 如下：$$
\begin{align}
    \mathbf{h}_{t}^{\prime} & = \mathbf{u}_{t} + \sum_{i=1}^{N_{s}} {\operatorname{FFN}^{(s)}_{i}\left( \mathbf{u}_{t} \right)} + \sum_{i=1}^{N_r} {g_{i,t} \operatorname{FFN}^{(r)}_{i}\left( \mathbf{u}_{t} \right)}, \tag{12}\\[1.2ex]
    g_{i,t} & = \frac{g^{\prime}_{i,t}}{\sum_{j=1}^{N_r} g^{\prime}_{j,t}}, \tag{13}\\[1.2ex]
    g^{\prime}_{i,t} & = \begin{cases} 
    s_{i,t}, & s_{i,t} \in \operatorname{Topk} (\{ s_{j, t} | 1 \leq j \leq N_r \}, K_{r}), \tag{14}\\[1.2ex]
    0, & \text{否则}, 
    \end{cases} \\[1.2ex]
    s_{i,t} & = \operatorname{Sigmoid} \left( {\mathbf{u}_{t}}^{T} \mathbf{e}_{i} \right),\tag{15}
\end{align}$$其中 $N_{s}$ 和 $N_r$ 分别表示共享专家和路由专家的数量；$\operatorname{FFN}^{(s)}_{i}(\cdot)$ 和 $\operatorname{FFN}^{(r)}_{i}(\cdot)$ 分别表示第 $i$ 个共享专家和第 $i$ 个路由专家；$K_{r}$ 表示激活的路由专家数量；$g_{i,t}$ 是第 $i$ 个专家的门控值；$s_{i,t}$ 是标记到专家的亲和度；$\mathbf{e}_{i}$ 是第 $i$ 个路由专家的中心向量；$\operatorname{Topk}(\cdot, K)$ 表示在为第 $t$ 个标记和所有路由专家计算的亲和度分数中选取最高的 $K$ 个分数的集合。与 DeepSeek-V2 略有不同，DeepSeek-V3 使用 sigmoid 函数来计算亲和度分数，并在所有选定的亲和度分数中进行归一化以生成门控值。


**无辅助损失的负载平衡**

对于 MoE 模型，专家负载不平衡会导致路由崩溃，并在专家并行场景中降低计算效率。传统解决方案通常依赖辅助损失来避免负载不平衡。然而，过大的辅助损失会损害模型性能。为了在负载平衡和模型性能之间取得更好的平衡，我们开创了一种无辅助损失的负载平衡策略，以确保负载平衡。具体来说，我们为每个专家引入一个偏置项 $b_i$，并将其添加到相应的亲和分数 $s_{i,t}$ 中，以确定 Top-K 路由：$$
g^{\prime}_{i,t} = 
\begin{cases} 
s_{i,t}, & s_{i,t} + b_i \in \operatorname{Topk} (\{ s_{j, t} + b_j | 1 \leq j \leq N_r \}, K_{r}), \\[1.2ex]
0, & \text{otherwise}. 
\end{cases}\tag{16}$$注意，偏置项仅用于路由。将与 FFN 输出相乘的门控值仍然来自原始亲和分数 $s_{i,t}$。在训练期间，我们持续监控每个训练步骤中整个批次的专家负载。在每个步骤结束时，如果对应的专家过载，我们将减少偏置项 $\gamma$；如果对应的专家负载不足，我们将增加 $\gamma$，其中 $\gamma$ 是一个称为偏置更新速度的超参数。通过动态调整，DeepSeek-V3在训练期间保持专家负载平衡，并比通过纯辅助损失来鼓励负载平衡的模型取得更好的性能。


**序列级辅助平衡损失**

尽管 DeepSeek-V3 主要依赖无辅助损失的策略来实现负载平衡，为了防止单个序列内的极端不平衡，我们还采用了一种补充的序列级平衡损失：$$
\begin{align}
    \mathcal{L}_{\mathrm{Bal}} & = \alpha \sum_{i=1}^{N_r}{f_i P_i}, \tag{17}\\[1.2ex]
    f_i = \frac{N_r}{K_r T} \sum_{t=1}^{T} \mathbb{1} & \left( s_{i,t} \in \operatorname{Topk} ( \{ s_{j, t} | 1 \leq j \leq N_r \}, K_{r} ) \right), \tag{18}\\[1.2ex]
    s^{\prime}_{i,t} & = \frac{s_{i,t}}{\sum_{j=1}^{N_r} s_{j,t}}, \tag{19}\\[1.2ex]
    P_i & = \frac{1}{T} \sum_{t=1}^{T}{s^{\prime}_{i,t}},\tag{20}
\end{align}$$其中平衡因子 $\alpha$ 是一个超参数，对于 DeepSeek-V3 将被设定为一个极小值；$\mathbb{1}(\cdot)$ 表示指示函数；$T$ 表示序列中的 token 数量。序列级平衡损失鼓励每个序列上专家的负载保持平衡。


**节点限制路由**

类似于 DeepSeek-V2 使用的设备限制路由，DeepSeek-V3 也采用了一种受限的路由机制，以限制训练期间的通信成本。简而言之，我们确保每个 token 最多被发送到 $M$ 个节点，这些节点根据分布在每个节点上的专家的最高 $\frac{K_r}{M}$ 亲和分数之和进行选择。在此约束下，我们的 MoE 训练框架几乎可以实现完全的计算-通信重叠。


**无 token 丢弃**

由于有效的负载均衡策略，DeepSeek-V3 在整个训练过程中保持良好的负载均衡。因此，DeepSeek-V3 在训练期间不丢弃任何 token。此外，我们还实施了特定的部署策略以确保推理负载均衡，因此 DeepSeek-V3 在推理过程中也不丢弃 token。
<img src="https://arxiv.org/html/2412.19437v2/x3.png" width="600">
**图 3.** 我们的 MTP 实现示意图。我们为每个深度的每个 token 预测保留完整的因果链。


#### 2.2 多标记预测

受 meta_mtp 的启发，我们为 DeepSeek-V3 设定了一个 MTP 目标，将预测范围扩展到每个位置的多个未来标记。一方面，MTP 目标密集化了训练信号，可能提高数据效率。另一方面，MTP 可能使模型能够预先规划其表示，以更好地预测未来标记。图 3 展示了我们对 MTP 的实现。与 meta_mtp 不同的是，他们使用独立的输出头并行预测 $D$ 个附加标记，而我们则顺序预测附加标记，并在每个预测深度保持完整的因果链。我们将在本节中介绍 MTP 实现的细节。

**MTP 模块**

具体来说，我们的 MTP 实现使用 $D$ 个顺序模块来预测 $D$ 个额外的 token。第 $k$ 个 MTP 模块由一个共享的嵌入层 $\operatorname{Emb}(\cdot)$、一个共享的输出头 $\operatorname{OutHead}(\cdot)$、一个 Transformer 块 $\operatorname{TRM}_k(\cdot)$ 和一个投影矩阵 $M_k \in \mathbb{R}^{d \times 2d}$ 组成。对于第 $i$ 个输入 token $t_i$，在第 $k$ 层预测深度，我们首先将第 $k-1$ 层深度的第 $i$ 个 token 表示 $\mathbf{h}_i^{k-1} \in \mathbb{R}^{d}$ 和第 $i+k$ 个 token 的嵌入 $\operatorname{Emb}(t_{i+k}) \in \mathbb{R}^{d}$ 通过线性投影结合起来：$$
\mathbf{h}_i^{\prime k} = M_k [\operatorname{RMSNorm}(\mathbf{h}_i^{k-1}) ; \operatorname{RMSNorm}(\operatorname{Emb}(t_{i+k}))] \tag{21}$$其中 $[\cdot ; \cdot]$ 表示拼接。特别地，当 $k=1$ 时，$\mathbf{h}_i^{k-1}$ 指的是主模型给出的表示。注意，对于每个 MTP 模块，其嵌入层与主模型共享。组合后的 $\mathbf{h}_i^{\prime k}$ 作为第 $k$ 层深度 Transformer 块的输入，以生成当前深度的输出表示 $\mathbf{h}_{i}^{k}$：$$
\mathbf{h}_{1:T-k}^{k} = \operatorname{TRM}_k(\mathbf{h}_{1:T-k}^{\prime k}) \tag{22}$$其中 $T$ 表示输入序列长度，$_{i:j}$ 表示切片操作（包括左右边界）。最后，以 $\mathbf{h}_{i}^{k}$ 作为输入，共享的输出头将计算第 $k$ 个额外预测 token 的概率分布 $P_{i+1+k}^{k} \in \mathbb{R}^{V}$，其中 $V$ 是词汇表大小：$$
P_{i+k+1}^{k} = \operatorname{OutHead}(\mathbf{h}_{i}^{k}) \tag{23}$$输出头 $\operatorname{OutHead}(\cdot)$ 线性映射表示为 logits，然后应用 $\operatorname{Softmax}(\cdot)$ 函数来计算第 $k$ 个额外 token 的预测概率。此外，对于每个 MTP 模块，其输出头与主模型共享。我们保持预测因果链的原则与 EAGLE 相似，但其主要目标是推测解码，而我们利用 MTP 来改进训练。

**MTP 训练目标**

对于每个预测深度，我们计算交叉熵损失 $\mathcal{L}_{\text{MTP}}^{k}$：$$
\mathcal{L}_{\text{MTP}}^{k} = \operatorname{CrossEntropy}(P_{2 + k:T + 1}^{k}, t_{2 + k:T + 1}) = -\frac{1}{T} \sum_{i=2 + k}^{T + 1} \log P_i^k [t_i],$$其中 $T$ 表示输入序列长度，$t_i$ 表示第 $i$ 个位置的真实 token，$P_i^k [t_i]$ 表示第 $k$ 个 MTP 模块给出的 $t_i$ 的预测概率。

最后，我们计算所有深度上的 MTP 损失的平均值，并乘以一个权重因子 $\lambda$ 来获得整体 MTP 损失 $\mathcal{L}_{\text{MTP}}$，这作为 \dsviii{} 的附加训练目标：$$
\mathcal{L}_{\text{MTP}} = \frac{\lambda}{D} \sum_{k=1}^{D} \mathcal{L}_{\text{MTP}}^{k}.$$
**MTP 推理**

我们的 MTP 策略主要旨在提高主模型的性能，因此在推理过程中，我们可以直接丢弃 MTP 模块，主模型可以独立正常运行。此外，我们还可以重新利用这些 MTP 模块进行推测解码，以进一步提高生成速度。


### 3、基础设施

#### 3.1 计算集群

DeepSeek-V3 在配备有 2048 个 NVIDIA H800 GPU 的集群上进行训练。H800 集群中的每个节点包含 8 个通过 NVLink 和 NVSwitch 连接的 GPU。不同节点之间通过 InfiniBand（IB）互连以促进通信。

#### 3.2 训练框架

DeepSeek-V3 的训练由 HAI-LLM 框架支持，这是我们的工程师从头开发的高效轻量级训练框架。总体而言，DeepSeek-V3 应用了 16 路流水线并行（PP）、跨 8 个节点的 64 路专家并行（EP），以及 ZeRO-1 数据并行（DP）。

为了促进 DeepSeek-V3 的高效训练，我们进行了精细的工程优化。首先，我们设计了用于高效流水线并行的 DualPipe 算法。与现有的 PP 方法相比，DualPipe 具有更少的流水线气泡。更重要的是，它在前向和后向过程中重叠计算和通信阶段，从而解决了跨节点专家并行引入的高通信开销问题。其次，我们开发了高效的跨节点全对全通信内核，以充分利用 IB 和 NVLink 带宽，并保留专用于通信的流式多处理器（SMs）。最后，我们精心优化了训练过程中的内存占用，从而使我们能够在不使用昂贵的张量并行（TP）的情况下训练 DeepSeek-V3。

##### 3.2.1 DualPipe 和计算-通信重叠
<img src="https://arxiv.org/html/2412.19437v2/x4.png" width="600">
**图 4：** 单个前向和后向块对的重叠策略（Transformer 块的边界未对齐）。橙色表示前向，绿色表示“输入的后向”，蓝色表示“权重的后向”，紫色表示 PP 通信，红色表示屏障。全连接和 PP 通信都可以完全隐藏。


对于 DeepSeek-V3，跨节点专家并行引入的通信开销导致计算与通信的比率约为 1:1，效率较低。
为了解决这一挑战，我们设计了一种创新的流水线并行算法，称为 DualPipe，它不仅通过有效重叠前向和后向计算-通信阶段加速模型训练，还减少了流水线气泡。

DualPipe 的关键理念是重叠单个前向和后向块对内的计算和通信。具体来说，我们将每个块分为四个部分：`attention`、`all-to-all dispatch`、`MLP` 和 `all-to-all combine`。特别地，对于一个后向块，`attention` 和 `MLP` 都进一步分为两部分，`backward for input` 和 `backward for weights`，类似于 ZeroBubble。此外，我们还有一个 `PP communication` 组件。
如图 4 所示，对于一对前向和后向块，我们重新排列这些组件并手动调整用于通信与计算的 GPU SM 的比例。在这种重叠策略中，我们可以确保在执行期间全连接和 PP 通信都可以完全隐藏。鉴于这种高效的重叠策略，完整的 DualPipe 调度如图 5 所示。它采用双向流水线调度，同时从流水线的两端提供微批次，且大部分通信可以完全重叠。这种重叠还确保了随着模型的进一步扩展，只要我们保持恒定的计算与通信比率，我们仍然可以在节点间使用细粒度的专家，同时实现接近零的全连接通信开销。
<img src="https://arxiv.org/html/2412.19437v2/x5.png" width="600">
**图 5：** 8 个 PP 级别和 20 个微批次在双向中的 DualPipe 调度示例。反向微批次与正向微批次对称，因此为简化说明省略了它们的批次 ID。由共享黑色边框包围的两个单元格具有相互重叠的计算和通信。


即使在没有重通信负担的更一般场景中，DualPipe 仍然表现出效率优势。在表 2 中，我们总结了不同 PP 方法中的流水线气泡和内存使用情况。如表所示，与 ZB1P 和 1F1B 相比，DualPipe 显著减少了流水线气泡，而峰值激活内存仅增加了 $\frac{1}{PP}$ 倍。虽然 DualPipe 需要保留模型参数的两份副本，但由于我们在训练期间使用较大的 EP 大小，这并不会显著增加内存消耗。与 Chimera 相比，DualPipe 只要求流水线阶段和微批次可被 2 整除，而不要求微批次可被流水线阶段整除。此外，对于 DualPipe，随着微批次数量的增加，气泡和激活内存都不会增加。


**表 2**：不同流水线并行方法的流水线气泡和内存使用比较。$F$ 表示一个前向块的执行时间，$B$ 表示一个完整后向块的执行时间，$W$ 表示“权重后向”块的执行时间，而 $F\&B$ 表示两个相互重叠的前向和后向块的执行时间。


##### 3.2.2 跨节点全互联通信的高效实现

为了确保 DualPipe 的计算性能，我们定制了高效的跨节点全互联通信内核（包括分发和合并），以节省用于通信的 SM 数量。内核的实现与 MoE 门控算法和集群的网络拓扑共同设计。具体来说，在我们的集群中，跨节点 GPU 通过 IB 完全互联，节点内通信通过 NVLink 处理。NVLink 提供 160 GB/s 的带宽，大约是 IB（50 GB/s）的 3.2 倍。为了有效利用 IB 和 NVLink 的不同带宽，我们限制每个 token 被分发到最多 4 个节点，从而减少 IB 流量。对于每个 token，当其路由决策确定后，首先通过 IB 传输到目标节点上具有相同节点内索引的 GPU。一旦到达目标节点，我们将努力确保它通过 NVLink 即时转发到拥有其目标专家的特定 GPU，而不被后续到达的 token 阻塞。通过这种方式，IB 和 NVLink 的通信完全重叠，每个 token 可以在不增加 NVLink 额外开销的情况下有效选择每节点平均 3.2 个专家。这意味着，尽管 DeepSeek-V3 实际上只选择 8 个路由专家，它可以将这个数字扩展到最多 13 个专家（4 个节点 $\times$ 3.2 个专家/节点），同时保持相同的通信成本。总体而言，在这种通信策略下，仅需 20 个 SM 就足以充分利用 IB 和 NVLink 的带宽。

具体来说，我们采用了 warp 专业化技术，并将 20 个 SM 分为 10 个通信通道。在分发过程中，(1) IB 发送、(2) IB 到 NVLink 转发和 (3) NVLink 接收由各自的 warp 处理。分配给每个通信任务的 warp 数量根据所有 SM 的实际工作负载动态调整。同样，在合并过程中，(1) NVLink 发送、(2) NVLink 到 IB 转发和累加、(3) IB 接收和累加也由动态调整的 warp 处理。此外，分发和合并内核与计算流重叠，因此我们也考虑了它们对其他 SM 计算内核的影响。具体而言，我们采用定制的 PTX（并行线程执行）指令，并自动调整通信块大小，这显著减少了 L2 缓存的使用和对其他 SM 的干扰。

##### 3.2.3 极致内存节省，开销极小

为了在训练过程中减少内存占用，我们采用了以下技术。

**RMSNorm 和 MLA 上投影的重新计算**

我们在反向传播中重新计算所有 RMSNorm 操作和 MLA 上投影，从而消除了持续存储其输出激活的需要。虽然会有一些小开销，但这种策略显著减少了存储激活所需的内存。

**在 CPU 上的指数移动平均**

在训练期间，我们保留模型参数的指数移动平均（EMA），以便在学习率衰减后对模型性能进行早期估计。EMA 参数存储在 CPU 内存中，并在每次训练步骤后异步更新。此方法允许我们维护 EMA 参数，而不会产生额外的内存或时间开销。

**用于多 token 预测的共享嵌入和输出头**

使用 DualPipe 策略，我们将模型的最浅层（包括嵌入层）和最深层（包括输出头）部署在同一个 PP 排位上。这种安排使共享嵌入和输出头的参数和梯度在 MTP 模块和主模型之间实现物理共享。这种物理共享机制进一步提高了我们的内存效率。


#### 3.3 FP8 训练
<img src="https://arxiv.org/html/2412.19437v2/x6.png" width="600">
**图 6**：采用 FP8 数据格式的整体混合精度框架。为便于说明，仅展示了线性算子。

受到低精度训练的最新进展启发，我们提出了一种利用 FP8 数据格式的细粒度混合精度框架，用于训练 DeepSeek-V3。尽管低精度训练前景广阔，但通常受到激活、权重和梯度中异常值的限制。虽然在推理量化方面取得了显著进展，但在大规模语言模型预训练中成功应用低精度技术的研究相对较少。为应对这一挑战并有效扩展 FP8 格式的动态范围，我们引入了一种细粒度量化策略：$1 \times N_c$ 元素的平铺分组或 $N_c \times N_c$ 元素的块状分组。在我们的精度增加累积过程中，相关的反量化开销大大减少，这是实现准确的 FP8 通用矩阵乘法（GEMM）的关键。此外，为了进一步减少 MoE 训练中的内存和通信开销，我们以 FP8 缓存和分发激活，同时以 BF16 存储低精度优化器状态。我们在两个类似于 DeepSeek-V2-Lite 和 DeepSeek-V2 的模型规模上验证了所提出的 FP8 混合精度框架，训练约 1 万亿个标记（更多细节见附录 B.1）。值得注意的是，与 BF16 基线相比，我们的 FP8 训练模型的相对损失误差始终低于 0.25%，这一水平完全在训练随机性的可接受范围内。

##### 3.3.1 混合精度框架

基于广泛采用的低精度训练技术，我们提出了一种用于 FP8 训练的混合精度框架。在这个框架中，大多数计算密集型操作在 FP8 中进行，而一些关键操作则保留其原始数据格式，以平衡训练效率和数值稳定性。整体框架如图 6 所示。

首先，为了加速模型训练，核心计算内核（即 GEMM 操作）的绝大部分在 FP8 精度下实现。这些 GEMM 操作接受 FP8 张量作为输入，并生成 BF16 或 FP32 的输出。如图 6 所示，与线性算子相关的三个 GEMM 操作，即 Fprop（前向传播）、Dgrad（激活反向传播）和 Wgrad（权重反向传播），都在 FP8 中执行。理论上，这种设计将计算速度提高了一倍，相较于原来的 BF16 方法。此外，FP8 Wgrad GEMM 允许激活在反向传播中以 FP8 格式存储，大大减少了内存消耗。

尽管 FP8 格式在效率上有优势，但某些算子由于对低精度计算的敏感性，仍需要更高的精度。此外，一些低成本算子也可以利用更高的精度，而对整体训练成本的影响可以忽略不计。因此，经过仔细研究，我们为以下组件保留了原始精度（例如 BF16 或 FP32）：嵌入模块、输出头、MoE 门控模块、归一化算子和注意力算子。这些高精度的有针对性的保留确保了 DeepSeek-V3 的稳定训练动态。为了进一步保证数值稳定性，我们以更高的精度存储主权重、权重梯度和优化器状态。虽然这些高精度组件会带来一些内存开销，但在我们的分布式训练系统中，通过在多个 DP 排之间高效分片，其影响可以最小化。
<img src="https://arxiv.org/html/2412.19437v2/x7.png" width="600">
**图 7:** (a) 我们提出了一种细粒度量化方法，以减轻特征异常值引起的量化误差；为简化说明，仅展示了前向传播 (Fprop)。 (b) 结合我们的量化策略，我们通过在 CUDA 核心上每隔 128 个元素的 MMA 进行高精度累加，提升了 FP8 GEMM 的精度。

##### 3.3.2 提升量化与乘法的精度

基于我们的混合精度 FP8 框架，我们引入了多种策略来提高低精度训练的准确性，重点关注量化方法和乘法过程。

**细粒度量化**

在低精度训练框架中，由于 FP8 格式的动态范围有限（受限于其较少的指数位），溢出和下溢是常见的挑战。通常做法是通过将输入张量的最大绝对值缩放到 FP8 的最大可表示值来对齐输入分布。这种方法使低精度训练对激活异常值非常敏感，这会严重降低量化精度。为了解决这个问题，我们提出了一种细粒度量化方法，在更细的层级上应用缩放。如图 7(a) 所示： (1) 对于激活，我们在 1x128 的块基础上对元素进行分组和缩放（即，每个 token 每 128 个通道）；(2) 对于权重，我们在 128x128 的块基础上对元素进行分组和缩放（即，每 128 个输入通道每 128 个输出通道）。这种方法确保量化过程能够通过根据较小的元素组调整缩放来更好地适应异常值。在附录 B.2 中，我们进一步讨论了当我们在块基础上对激活进行分组和缩放时，训练不稳定性的问题。

我们方法的一个关键修改是在 GEMM 操作的内部维度上引入每组的缩放因子。标准的 FP8 GEMM 并不直接支持此功能。然而，结合我们精确的 FP32 累加策略，可以高效实现。

值得注意的是，我们的细粒度量化策略与微缩放格式的理念高度一致，而 NVIDIA 下一代 GPU（Blackwell 系列）的 Tensor 核心已经宣布支持具有更小量化粒度的微缩放格式。我们希望我们的设计可以为未来的工作提供参考，以跟上最新 GPU 架构的步伐。

**增加累加精度**

低精度 GEMM 操作通常受到下溢问题的困扰，其准确性在很大程度上依赖于高精度累加，这通常在 FP32 精度下进行。然而，我们观察到 NVIDIA H800 GPU 上 FP8 GEMM 的累加精度仅限于保留大约 14 位，这显著低于 FP32 累加精度。当内部维度 K 较大时，这一问题将更加突出，这是大规模模型训练中常见的情况，其中批量大小和模型宽度增加。以 K = 4096 的两个随机矩阵的 GEMM 操作为例，在我们的初步测试中，Tensor 核心中有限的累加精度导致最大相对误差接近 2%。尽管存在这些问题，有限的累加精度仍然是一些 FP8 框架的默认选项，严重限制了训练的准确性。

为了解决这个问题，我们采用了提升到 CUDA 核心以获得更高精度的策略。如图 7(b) 所示，在 Tensor 核心上执行 MMA（矩阵乘法-累加）时，中间结果使用有限的位宽进行累加。一旦达到 
$N_C$ 的间隔，这些部分结果将被复制到 CUDA 核心上的 FP32 寄存器中，在那里进行全精度的 FP32 累加。如前所述，我们的细粒度量化在内部维度 K 上应用每组缩放因子。这些缩放因子可以在 CUDA 核心上高效地进行乘法运算，作为去量化过程，几乎没有额外的计算成本。

值得注意的是，这一修改降低了单个 warpgroup 的 WGMMA（Warpgroup-level Matrix Multiply-Accumulate）指令问题率。然而，在 H800 架构上，通常两个 WGMMA 可以同时进行：当一个 warpgroup 执行提升操作时，另一个可以执行 MMA 操作。这种设计能够重叠这两个操作，保持 Tensor 核心的高利用率。根据我们的实验，设置 $N_C = 128$ 个元素，相当于 4 个 WGMMAs，代表了可以显著提高精度而不引入大量开销的最小累加间隔。

**尾数优于指数**

与之前工作采用的混合 FP8 格式不同，后者在 Fprop 中使用 E4M3（4 位指数和 3 位尾数），在 Dgrad 和 Wgrad 中使用 E5M2（5 位指数和 2 位尾数），我们在所有张量上采用 E4M3 格式以获得更高的精度。我们将这种方法的可行性归因于我们的细粒度量化策略，即块和块内缩放。通过在较小的元素组上操作，我们的方法有效地在这些分组元素之间共享指数位，减轻了动态范围有限的影响。

**在线量化**

延迟量化在张量级量化框架中被采用，它维护先前迭代中最大绝对值的历史记录以推断当前值。为了确保准确的缩放并简化框架，我们在线计算每个 1x128 激活块或 128x128 权重块的最大绝对值。基于此，我们得出缩放因子，然后在线将激活或权重量化为 FP8 格式。

##### 3.3.3 低精度存储和通信

在我们的 FP8 训练框架中，我们通过将缓存的激活和优化器状态压缩为低精度格式，进一步减少内存消耗和通信开销。

**低精度优化器状态**

我们采用 BF16 数据格式而不是 FP32 来跟踪 AdamW 优化器中的一阶和二阶矩，而不会导致明显的性能下降。然而，优化器存储的主权重和用于批量大小累积的梯度仍保持在 FP32，以确保训练过程中的数值稳定性。

**低精度激活**

如图 6 所示，Wgrad 操作在 FP8 中执行。为了减少内存消耗，自然选择是在 FP8 格式中缓存激活，以用于线性运算符的反向传播。然而，对于几个运算符，我们采取了一些特殊考虑，以实现低成本高精度训练：

1. 注意力运算符后的线性输入。这些激活也用于注意力运算符的反向传播，对精度非常敏感。我们采用了专门的 E5M6 数据格式用于这些激活。此外，这些激活将在反向传播中从 1x128 量化块转换为 128x1 块。为了避免引入额外的量化误差，所有的缩放因子都是整数的 2 次幂。

2. MoE 中 SwiGLU 运算符的输入。为了进一步降低内存成本，我们缓存 SwiGLU 运算符的输入，并在反向传播中重新计算其输出。这些激活也使用我们的细粒度量化方法存储在 FP8 中，在内存效率和计算精度之间取得平衡。

**低精度通信**

通信带宽是 MoE 模型训练中的关键瓶颈。为了解决这个问题，我们将 MoE 上投影之前的激活量化为 FP8，然后应用分派组件，这与 MoE 上投影中的 FP8 前向传播兼容。与注意力运算符后的线性输入类似，这些激活的缩放因子是 2 的整数幂。类似的策略应用于 MoE 下投影之前的激活梯度。对于前向和反向组合组件，我们保留它们在 BF16 中，以在训练管道的关键部分保持训练精度。


#### 3.4 推理和部署

我们在 H800 集群上部署了 DeepSeek-V3，每个节点内的 GPU 通过 NVLink 互连，整个集群的所有 GPU 通过 IB 完全互连。为了同时确保在线服务的服务级目标（SLO）和高吞吐量，我们采用了以下部署策略，将 *预填充* 和 *解码* 阶段分开。

##### 3.4.1 预填充

预填充阶段的最小部署单元由 4 个节点和 32 个 GPU 组成。 attention 部分采用 4 路张量并行（TP4）与序列并行（SP）结合 8 路数据并行（DP8）。其较小的 TP 大小为 4，限制了 TP 通信的开销。对于 MoE 部分，我们使用 32 路专家并行（EP32），确保每个专家处理足够大的批量，从而提高计算效率。对于 MoE 的全互通信，我们使用与训练相同的方法：首先通过 IB 在节点间传输令牌，然后通过 NVLink 在节点内的 GPU 之间转发。特别是，我们对浅层的稠密 MLP 使用 1 路张量并行，以节省 TP 通信。

为了在 MoE 部分的不同专家之间实现负载平衡，我们需要确保每个 GPU 处理的令牌数量大致相同。为此，我们引入了一种冗余专家的部署策略，复制高负载专家并冗余部署它们。高负载专家是基于在线部署期间收集的统计数据检测的，并定期调整（例如，每 10 分钟）。在确定冗余专家集合后，我们根据观察到的负载在节点内的 GPU 间仔细重新排列专家，尽量在不增加跨节点全互通信开销的情况下平衡 GPU 间的负载。对于 DeepSeek-V3 的部署，我们在预填充阶段设置了 32 个冗余专家。对于每个 GPU，除了其托管的原始 8 个专家外，还将托管一个额外的冗余专家。

此外，在预填充阶段，为了提高吞吐量并隐藏全互和 TP 通信的开销，我们同时处理两个具有类似计算工作负载的微批次，将一个微批次的 attention 和 MoE 与另一个的 dispatch 和 combine 重叠。

最后，我们正在探索一种专家的动态冗余策略，其中每个 GPU 托管更多的专家（例如 16 个专家），但在每个推理步骤中只激活 9 个。在每层全互操作开始之前，我们实时计算全局最优路由方案。鉴于预填充阶段的计算量很大，计算此路由方案的开销几乎可以忽略不计。

##### 3.4.2 解码

在解码过程中，我们将共享专家视为一个路由专家。从这个角度来看，每个令牌在路由时将选择 9 个专家，其中共享专家被视为高负载专家，总是会被选择。解码阶段的最小部署单元由 40 个节点和 320 个 GPU 组成。 attention 部分采用 TP4 和 SP，结合 DP80，而 MoE 部分使用 EP320。对于 MoE 部分，每个 GPU 仅托管一个专家，64 个 GPU 负责托管冗余专家和共享专家。 dispatch 和 combine 部分的全互通信通过 IB 上的直接点对点传输进行，以实现低延迟。此外，我们利用 IBGDA 技术进一步减少延迟并提高通信效率。

与预填充类似，我们在一定间隔内根据在线服务的统计专家负载定期确定冗余专家集。然而，我们不需要重新排列专家，因为每个 GPU 只托管一个专家。我们也在探索解码的动态冗余策略。然而，这需要更仔细地优化计算全局最优路由方案的算法，并与 dispatch 内核融合以减少开销。

此外，为了提高吞吐量并隐藏全互通信的开销，我们也在探索在解码阶段同时处理两个具有类似计算工作负载的微批次。与预填充不同，解码阶段 attention 消耗的时间更多。因此，我们将一个微批次的 attention 与另一个的 dispatch+MoE+combine 重叠。在解码阶段，每个专家的批量相对较小（通常在 256 个令牌以内），瓶颈在于内存访问而非计算。由于 MoE 部分只需加载一个专家的参数，内存访问开销很小，因此使用较少的 SM 不会显著影响整体性能。因此，为了避免影响 attention 部分的计算速度，我们可以只分配一小部分 SM 给 dispatch+MoE+combine。

#### 3.5 关于硬件设计的建议

基于我们对全互通信和 FP8 训练方案的实现，我们向 AI 硬件供应商提出以下芯片设计建议。

##### 3.5.1 通信硬件

在 DeepSeek-V3 中，我们实现了计算与通信的重叠，以隐藏计算过程中的通信延迟。这显著降低了对通信带宽的依赖，相较于串行计算和通信。然而，目前的通信实现依赖于昂贵的流处理器（SM）（例如，我们在 H800 GPU 中为此分配了 132 个 SM 中的 20 个），这将限制计算吞吐量。此外，使用 SM 进行通信会导致显著的效率低下，因为张量核心完全没有被充分利用。

目前，SM 主要为全互通信执行以下任务：

* 在 IB（InfiniBand）和 NVLink 域之间转发数据，同时聚合从单个 GPU 发往同一节点中多个 GPU 的 IB 流量。
* 在 RDMA 缓冲区（注册的 GPU 内存区域）和输入/输出缓冲区之间传输数据。
* 为 all-to-all combine 执行 reduce 操作。
* 在通过 IB 和 NVLink 域向多个专家传输分块数据时，管理细粒度的内存布局

我们希望未来的供应商能开发硬件，将这些通信任务从宝贵的计算单元 SM 中卸载出来，并作为 GPU 协处理器或类似于 NVIDIA SHARP 的网络协处理器。同时，为了减少应用编程的复杂性，我们希望这种硬件能够从计算单元的角度统一 IB（横向扩展）和 NVLink（纵向扩展）网络。通过这种统一接口，计算单元可以轻松完成诸如 read、write、multicast 和 reduce 等操作，在整个 IB-NVLink 统一域中通过基于简单原语的通信请求来实现。

##### 3.5.2 计算硬件

**Tensor Core 中更高的 FP8 GEMM 累积精度**

在 NVIDIA Hopper 架构的当前 Tensor Core 实现中，FP8 GEMM 的累积精度有限。在通过基于最大指数的右移对齐 32 个尾数乘积后，Tensor Core 仅使用每个尾数乘积的最高 14 位进行加法，并截断超出此范围的位。加法结果的累积到寄存器中也采用 14 位精度。我们的实现通过在 CUDA 核心中以 FP32 精度将 128 个 FP8×FP8 乘积的加法结果累积到寄存器中，部分缓解了这一限制。尽管在实现成功的 FP8 训练中有所帮助，但这仅仅是对 Hopper 架构在 FP8 GEMM 累积精度上硬件不足的妥协。未来的芯片需要采用更高的精度。

**支持按块和按块量化**

当前的 GPU 仅支持每张量量化，缺乏对我们按块和按块量化的原生支持。在当前实现中，当达到 $N_C$ 间隔时，部分结果将从 Tensor Core 复制到 CUDA 核心，乘以缩放因子，并添加到 CUDA 核心上的 FP32 寄存器中。尽管结合我们精确的 FP32 累积策略显著减轻了反量化的开销，但 Tensor Core 和 CUDA 核心之间频繁的数据移动仍限制了计算效率。因此，我们建议未来的芯片支持细粒度量化，使 Tensor Core 能够接收缩放因子并实现具有组缩放的 MMA。这样，整个部分和累积和反量化可以直接在 Tensor Core 内完成，直到生成最终结果，避免频繁的数据移动。

**支持在线量化**

尽管我们的研究证明了在线量化的有效性，但当前的实现难以有效支持在线量化。在现有过程中，我们需要从 HBM（高带宽内存）中读取 128 个 BF16 激活值（前一次计算的输出）进行量化，然后将量化后的 FP8 值写回 HBM，再次读取以进行 MMA。为了解决这种低效问题，我们建议未来的芯片将 FP8 转换和 TMA（张量内存加速器）访问集成到一个融合操作中，以便在激活从全局内存传输到共享内存的过程中完成量化，避免频繁的内存读写。我们还建议支持 warp 级转换指令以加速，这将进一步促进层归一化和 FP8 转换的更好融合。或者，可以采用近存计算方法，将计算逻辑放置在 HBM 附近。在这种情况下，BF16 元素可以在从 HBM 读入 GPU 时直接转换为 FP8，从而减少大约 50% 的片外内存访问。

**支持转置 GEMM 操作**

当前架构使得将矩阵转置与 GEMM 操作融合变得繁琐。在我们的工作流程中，前向传播期间的激活被量化为 1x128 FP8 块并存储。在反向传播期间，需要读取矩阵、反量化、转置、重新量化为 128x1 块，并存储在 HBM 中。为了减少内存操作，我们建议未来的芯片在 MMA 操作之前，支持从共享内存直接读取转置矩阵，对于训练和推理所需的精度。结合 FP8 格式转换和 TMA 访问的融合，这一增强将显著简化量化工作流程。


### 4、预训练

#### 4.1 数据构建

与 DeepSeek-V2 相比，我们优化了预训练语料库，增加了数学和编程样本的比例，并扩展了英语和中文以外的多语言覆盖范围。此外，我们精简了数据处理流程，以最大限度地减少冗余，同时保持语料库的多样性。受 xxx 的启发，我们实现了文档打包方法以确保数据完整性，但在训练期间没有采用跨样本注意力掩码。最终，DeepSeek-V3 的训练语料库由 14.8T 的高质量多样化标记组成。

在 DeepSeekCoder-V2 的训练过程中，我们观察到中间填充（FIM）策略在不影响下一个标记预测能力的同时，使模型能够根据上下文准确预测中间文本。与 DeepSeekCoder-V2 一致，我们在 DeepSeek-V3 的预训练中也采用了 FIM 策略。具体而言，我们使用前缀-后缀-中间（PSM）框架将数据结构化如下：
$$\begin{align}
\texttt{<|fim\_begin|>}f_{\text{pre}}\texttt{<|fim\_hole|>}f_{\text{suf}}\texttt{<|fim\_end|>}f_{\text{middle}}\texttt{<|eos\_token|>} . \nonumber
\end{align}$$
这种结构在文档层面作为预打包过程的一部分应用。FIM 策略的应用率为 0.1，与 PSM 框架一致。

DeepSeek-V3 的分词器采用字节级 BPE，词汇扩展到 128K 个标记。我们的分词器的预分词器和训练数据经过修改，以优化多语言压缩效率。此外，与 DeepSeek-V2 相比，新的预分词器引入了结合标点符号和换行符的标记。然而，这种方法在模型处理没有终止换行符的多行提示时，特别是少样本评估提示时，可能引入标记边界偏差。为了解决这个问题，我们在训练中随机拆分了一定比例的此类组合标记，使模型接触到更广泛的特殊情况，从而减轻这种偏差。

#### 4.2 超参数

**模型超参数**

我们设置 Transformer 层数为 61，隐藏维度为 7168。所有可学习参数均以标准差 0.006 随机初始化。在 MLA 中，我们将注意力头数 $n_h$ 设置为 128，每个头的维度 $d_h$ 为 128。KV 压缩维度 $d_c$ 设置为 512，查询压缩维度 $d_c^{\prime}$ 设置为 1536。对于解耦的查询和键，每个头的维度 $d_h^R$ 设置为 64。我们将除前三层外的所有 FFN 替换为 MoE 层。每个 MoE 层由 1 个共享专家和 256 个路由专家组成，其中每个专家的中间隐藏维度为 2048。在路由专家中，每个标记将激活 8 个专家，并确保每个标记最多发送到 4 个节点。多标记预测深度 $D$ 设置为 1，即除了准确的下一个标记外，每个标记还将预测一个额外的标记。与 DeepSeek-V2 一样，DeepSeek-V3 也在压缩的潜在向量后使用额外的 RMSNorm 层，并在宽度瓶颈处乘以额外的缩放因子。在此配置下，DeepSeek-V3 包含 671B 总参数，其中 37B 在每个标记时被激活。

**训练超参数**

我们使用 AdamW 优化器，其超参数设置为 $\beta_1=0.9$、$\beta_2=0.95$ 和 $\mathrm{weight\_decay}=0.1$。在预训练期间，我们将最大序列长度设置为 4K，并在 14.8T 标记上预训练 DeepSeek-V3。关于学习率调度，我们首先在前 2K 步中将其从 0 线性增加到 $2.2 \times 10^{-4}$。然后，我们保持 $2.2 \times 10^{-4}$ 的恒定学习率，直到模型消耗 10T 训练标记。随后，我们在 4.3T 标记中按照余弦衰减曲线逐渐将学习率衰减到 $2.2 \times 10^{-5}$。在最后 500B 标记的训练中，我们在前 333B 标记中保持 $2.2 \times 10^{-5}$ 的恒定学习率，并在剩余的 167B 标记中切换到另一个恒定学习率 $7.3 \times 10^{-6}$。梯度裁剪范数设置为 1.0。我们采用批量大小调度策略，在前 469B 标记的训练中，批量大小逐渐从 3072 增加到 15360，然后在剩余训练中保持 15360。我们利用管道并行性将模型的不同层部署在不同的 GPU 上，对于每一层，路由专家将在属于 8 个节点的 64 个 GPU 上均匀部署。关于节点限制路由，每个标记最多将被发送到 4 个节点（即 $M=4$）。对于无辅助损失的负载平衡，我们在前 14.3T 标记中将偏差更新速度 $\gamma$ 设置为 0.001，在剩余的 500B 标记中设置为 0.0。对于平衡损失，我们将 $\alpha$ 设置为 0.0001，以避免任何单个序列中的极端不平衡。MTP 损失权重 $\lambda$ 在前 10T 标记中设置为 0.3，在剩余的 4.8T 标记中设置为 0.1。
<img src="https://arxiv.org/html/2412.19437v2/x8.png" width="500">
**图 8**. 在 "大海捞针" (NIAH) 测试中的评估结果显示，DeepSeek-V3 在所有上下文窗口长度（最长达 128K）上表现良好。


#### 4.3 长上下文扩展

我们采用与DeepSeek-V2 类似的方法，使 DeepSeek-V3 具备长上下文能力。在预训练阶段之后，我们应用 YaRN 进行上下文扩展，并执行两个额外的训练阶段，每个阶段包含 1000 步，以逐步将上下文窗口从 4K 扩展到 32K，然后再到 128K。YaRN 的配置与 DeepSeek-V2 中使用的一致，仅应用于解耦的共享键 $\mathbf{k}^R_t$。两个阶段的超参数保持相同，比例 $s = 40$，$\alpha = 1$，$\beta = 32$，缩放因子 $\sqrt{t} = 0.1 \ln{s} + 1$。在第一阶段，序列长度设置为 32K，批量大小为 1920。在第二阶段，序列长度增加到 128K，批量大小减少到 480。两个阶段的学习率均设置为 $7.3 \times 10^{-6}$，与预训练阶段的最终学习率相匹配。

通过这两个阶段的扩展训练，DeepSeek-V3 能够处理长度达 128K 的输入，同时保持强大的性能。图 8 显示，经过监督微调后，DeepSeek-V3 在“Needle In A Haystack” (NIAH) 测试中表现出色，展示了在最长达 128K 的上下文窗口中的一致稳健性。

#### 4.4 评估

##### 4.4.1 评估基准

DeepSeek-V3 的基础模型在多语言语料库上进行预训练，其中英语和中文占主要部分，因此我们在一系列主要为英语和中文的基准测试以及一个多语言基准上评估其性能。我们的评估基于集成在 HAI-LLM 框架中的内部评估框架。

略

根据我们之前的工作，我们对包括 HellaSwag、PIQA、WinoGrande、RACE-Middle、RACE-High、MMLU、MMLU-Redux、MMLU-Pro、MMMLU、ARC-Easy、ARC-Challenge、C-Eval、CMMLU、C3 和 CCPM 在内的数据集采用困惑度评估，并对 TriviaQA、NaturalQuestions、DROP、MATH、GSM8K、MGSM、HumanEval、MBPP、LiveCodeBench-Base、CRUXEval、BBH、AGIEval、CLUEWSC、CMRC 和 CMath 采用生成评估。此外，我们对 Pile-test 进行基于语言模型的评估，并使用每字节比特数（BPB）作为度量标准，以确保使用不同分词器的模型之间的公平比较。

表格略

\subsubsection{Evaluation Results}

In Table~\ref{tab:main}, we compare the base model of \dsviii{} with the state-of-the-art open-source base models, including \dsvii{}-Base~\citep{dsvii} (our previous release), Qwen2.5 72B Base~\citep{qwen2_5}, and LLaMA-3.1 405B Base~\citep{llama3_1_405b}.
We evaluate all these models with our internal evaluation framework, and ensure that they share the same evaluation setting. 
Note that due to the changes in our evaluation framework over the past months, the performance of \dsvii{}-Base exhibits a slight difference from our previously reported results.
Overall, \dsviii{}-Base comprehensively outperforms \dsvii{}-Base and Qwen2.5 72B Base, and surpasses LLaMA-3.1 405B Base in the majority of benchmarks, essentially becoming the strongest open-source model.  

From a more detailed perspective, we compare \dsviii{}-Base with the other open-source base models individually.
(1) 
Compared with \dsvii{}-Base, due to the improvements in our model architecture, the scale-up of the model size and training tokens, and the enhancement of data quality, \dsviii{}-Base achieves significantly better performance as expected.
(2)
Compared with Qwen2.5 72B Base, the state-of-the-art Chinese open-source model, with only half of the activated parameters, \dsviii{}-Base also demonstrates remarkable advantages, especially on English, multilingual, code, and math benchmarks. 
As for Chinese benchmarks, except for CMMLU, a Chinese multi-subject multiple-choice task, \dsviii{}-Base also shows better performance than Qwen2.5 72B. 
(3)
Compared with LLaMA-3.1 405B Base, the largest open-source model with 11 times the activated parameters, \dsviii{}-Base also exhibits much better performance on multilingual, code, and math benchmarks. 
As for English and Chinese language benchmarks, \dsviii{}-Base shows competitive or better performance, and is especially good on BBH, MMLU-series, DROP, C-Eval, CMMLU, and CCPM. 

Due to our efficient architectures and comprehensive engineering optimizations, \dsviii{} achieves extremely high training efficiency. 
Under our training framework and infrastructures, training \dsviii{} on each trillion tokens requires only 180K H800 GPU hours, which is much cheaper than training 72B or 405B dense models. 

\begin{table}[h]
    \centering
    \footnotesize
    \setlength{\tabcolsep}{8pt}
    \begin{tabular}{@{}l c | c c | c c@{}}
    \toprule
    \multirow{2}{*}{\centering \textbf{Benchmark (Metric)}} & \multirow{2}{*}{\textbf{\# Shots}} & \textbf{Small MoE} & \textbf{Small MoE} & \textbf{Large MoE} & \textbf{Large MoE} \\
     & & \textbf{Baseline} & \textbf{w/ MTP} & \textbf{Baseline} & \textbf{w/ MTP} \\
    \midrule
    \# Activated Params {\tiny (Inference)} & - & 2.4B & 2.4B & 20.9B & 20.9B \\
    \# Total Params {\tiny (Inference)} & - & 15.7B & 15.7B & 228.7B & 228.7B \\
    \# Training Tokens & - & 1.33T & 1.33T & 540B & 540B \\
    \midrule
    Pile-test {\tiny (BPB)} & - & \textbf{0.729} & \textbf{0.729} & 0.658 & \textbf{0.657} \\
    BBH {\tiny (EM)} & 3-shot & 39.0 & \textbf{41.4} & 70.0 & \textbf{70.7} \\
    MMLU {\tiny (EM)} & 5-shot & 50.0 & \textbf{53.3} & \textbf{67.5} & 66.6 \\
    DROP {\tiny (F1)} & 1-shot & 39.2 & \textbf{41.3} & 68.5 & \textbf{70.6} \\
    TriviaQA {\tiny (EM)} & 5-shot & 56.9 & \textbf{57.7} & \textbf{67.0} & \textbf{67.3} \\
    NaturalQuestions {\tiny (EM)} & 5-shot & \textbf{22.7} & 22.3 & 27.2 & \textbf{28.5} \\
    HumanEval {\tiny (Pass@1)} & 0-shot & 20.7 & \textbf{26.8} & 44.5 & \textbf{53.7} \\
    MBPP {\tiny (Pass@1)} & 3-shot & 35.8 & \textbf{36.8} & 61.6 & \textbf{62.2} \\
    GSM8K {\tiny (EM)} & 8-shot & 25.4 & \textbf{31.4} & 72.3 & \textbf{74.0} \\
    MATH {\tiny (EM)} & 4-shot & 10.7 & \textbf{12.6} & 38.6 & \textbf{39.8} \\
    \bottomrule
    \end{tabular}
    \caption{
    Ablation results for the MTP strategy. 
    The MTP strategy consistently enhances the model performance on most of the evaluation benchmarks.
    }
    \label{tab:ablation_nextn}
\end{table}

\subsection{Discussion}

\subsubsection{Ablation Studies for Multi-Token Prediction}
\label{discussion:ablation_nextn}

In Table~\ref{tab:ablation_nextn}, we show the ablation results for the MTP strategy. 
To be specific, we validate the MTP strategy on top of two baseline models across different scales. 
At the small scale, we train a baseline MoE model comprising 15.7B total parameters on 1.33T tokens. 
At the large scale, we train a baseline MoE model comprising 228.7B total parameters on 540B tokens. 
On top of them, keeping the training data and the other architectures the same, we append a 1-depth MTP module onto them and train two models with the MTP strategy for comparison. 
Note that during inference, we directly discard the MTP module, so the inference costs of the compared models are exactly the same. 
From the table, we can observe that the MTP strategy consistently enhances the model performance on most of the evaluation benchmarks. 

\subsubsection{Ablation Studies for the Auxiliary-Loss-Free Balancing Strategy}
\label{discussion:ablation_noaux_tc}

In Table~\ref{tab:ablation_noaux_tc}, we show the ablation results for the auxiliary-loss-free balancing strategy.
We validate this strategy on top of two baseline models across different scales. 
At the small scale, we train a baseline MoE model comprising 15.7B total parameters on 1.33T tokens. 
At the large scale, we train a baseline MoE model comprising 228.7B total parameters on 578B tokens. 
Both of the baseline models purely use auxiliary losses to encourage load balance, and use the sigmoid gating function with top-K affinity normalization. 
Their hyper-parameters to control the strength of auxiliary losses are the same as \dsvii{}-Lite and \dsvii{}, respectively. 
On top of these two baseline models, keeping the training data and the other architectures the same, we remove all auxiliary losses and introduce the auxiliary-loss-free balancing strategy for comparison. 
From the table, we can observe that the auxiliary-loss-free strategy consistently achieves better model performance on most of the evaluation benchmarks. 

\begin{table}[t]
    \centering
    \footnotesize
    \setlength{\tabcolsep}{4pt}
    \begin{tabular}{@{}l c | c c | c c@{}}
    \toprule
    \multirow{2}{*}{\centering \textbf{Benchmark (Metric)}} & \multirow{2}{*}{\textbf{\# Shots}} & \textbf{Small MoE} & \textbf{Small MoE} & \textbf{Large MoE} & \textbf{Large MoE} \\
     & & \textbf{Aux-Loss-Based} & \textbf{Aux-Loss-Free} & \textbf{Aux-Loss-Based} & \textbf{Aux-Loss-Free} \\
    \midrule
    \# Activated Params & - & 2.4B & 2.4B & 20.9B & 20.9B \\
    \# Total Params & - & 15.7B & 15.7B & 228.7B & 228.7B \\
    \# Training Tokens & - & 1.33T & 1.33T & 578B & 578B \\
    \midrule
    Pile-test {\tiny (BPB)} & - & 0.727 & \textbf{0.724} & 0.656 & \textbf{0.652} \\
    BBH {\tiny (EM)} & 3-shot & 37.3 & \textbf{39.3} & 66.7 & \textbf{67.9} \\
    MMLU {\tiny (EM)} & 5-shot & 51.0 & \textbf{51.8} & \textbf{68.3} & 67.2 \\
    DROP {\tiny (F1)} & 1-shot & 38.1 & \textbf{39.0} & \textbf{67.1} & \textbf{67.1} \\
    TriviaQA {\tiny (EM)} & 5-shot & \textbf{58.3} & \textbf{58.5} & 66.7 & \textbf{67.7} \\
    NaturalQuestions {\tiny (EM)} & 5-shot & \textbf{23.2} & \textbf{23.4} & 27.1 & \textbf{28.1} \\
    HumanEval {\tiny (Pass@1)} & 0-shot & 22.0 & \textbf{22.6} & 40.2 & \textbf{46.3} \\
    MBPP {\tiny (Pass@1)} & 3-shot & \textbf{36.6} & 35.8 & 59.2 & \textbf{61.2} \\
    GSM8K {\tiny (EM)} & 8-shot & 27.1 & \textbf{29.6} & 70.7 & \textbf{74.5} \\
    MATH {\tiny (EM)} & 4-shot & \textbf{10.9} & \textbf{11.1} & 37.2 & \textbf{39.6} \\
    \bottomrule
    \end{tabular}
    \caption{
    Ablation results for the auxiliary-loss-free balancing strategy. 
    Compared with the purely auxiliary-loss-based method, the auxiliary-loss-free strategy consistently achieves better model performance on most of the evaluation benchmarks.
    }
    \label{tab:ablation_noaux_tc}
\end{table}

\subsubsection{Batch-Wise Load Balance VS. Sequence-Wise Load Balance}
\label{discussion:balance}

The key distinction between auxiliary-loss-free balancing and sequence-wise auxiliary loss lies in their balancing scope: batch-wise versus sequence-wise. 
Compared with the sequence-wise auxiliary loss, batch-wise balancing imposes a more flexible constraint, as it does not enforce in-domain balance on each sequence. 
This flexibility allows experts to better specialize in different domains. 
To validate this, we record and analyze the expert load of a 16B auxiliary-loss-based baseline and a 16B auxiliary-loss-free model on different domains in the Pile test set.
As illustrated in Figure~\ref{fig:expert_load}, we observe that the auxiliary-loss-free model demonstrates greater expert specialization patterns as expected. 

To further investigate the correlation between this flexibility and the advantage in model performance, we additionally design and validate a batch-wise auxiliary loss that encourages load balance on each training batch instead of on each sequence. 
The experimental results show that, when achieving a similar level of batch-wise load balance, the batch-wise auxiliary loss can also achieve similar model performance to the auxiliary-loss-free method.
To be specific, in our experiments with 1B MoE models, the validation losses are: 2.258 (using a sequence-wise auxiliary loss), 2.253 (using the auxiliary-loss-free method), and 2.253 (using a batch-wise auxiliary loss). 
We also observe similar results on 3B MoE models: the model using a sequence-wise auxiliary loss achieves a validation loss of 2.085, and the models using the auxiliary-loss-free method or a batch-wise auxiliary loss achieve the same validation loss of 2.080.

In addition, although the batch-wise load balancing methods show consistent performance advantages, they also face two potential challenges in efficiency: 
(1) load imbalance within certain sequences or small batches, and 
(2) domain-shift-induced load imbalance during inference. 
The first challenge is naturally addressed by our training framework that uses large-scale expert parallelism and data parallelism, which guarantees a large size of each micro-batch. 
For the second challenge, we also design and implement an efficient inference framework with redundant expert deployment, as described in Section \ref{sec:inference_deployment}, to overcome it.

\begin{figure}[!t]
\centering
\includegraphics[width=0.99\linewidth]{figures/relative_expert_load_multi.pdf}
\caption{
    Expert load of auxiliary-loss-free and auxiliary-loss-based models on three domains in the Pile test set. 
    The auxiliary-loss-free model shows greater expert specialization patterns than the auxiliary-loss-based one.
    The relative expert load denotes the ratio between the actual expert load and the theoretically balanced expert load. 
    Due to space constraints, we only present the results of two layers as an example, with the results of all layers provided in Appendix~\ref{app:detailed_expert_load}.
}
\label{fig:expert_load}
\end{figure}

\section{Post-Training}
\label{sec:alignment}

\subsection{Supervised Fine-Tuning}

We curate our instruction-tuning datasets to include 1.5M instances spanning multiple domains, with each domain employing distinct data creation methods tailored to its specific requirements.

\paragraph{Reasoning Data.} 
For reasoning-related datasets, including those focused on mathematics, code competition problems, and logic puzzles, we generate the data by leveraging an internal DeepSeek-R1 model. 
Specifically, while the R1-generated data demonstrates strong accuracy, it suffers from issues such as overthinking, poor formatting, and excessive length. 
Our objective is to balance the high accuracy of R1-generated reasoning data and the clarity and conciseness of regularly formatted reasoning data.

To establish our methodology, we begin by developing an expert model tailored to a specific domain, such as code, mathematics, or general reasoning, using a combined Supervised Fine-Tuning (SFT) and Reinforcement Learning (RL) training pipeline. 
This expert model serves as a data generator for the final model. 
The training process involves generating two distinct types of SFT samples for each instance: the first couples the problem with its original response in the format of <problem, original response>, while the second incorporates a system prompt alongside the problem and the R1 response in the format of <system prompt, problem, R1 response>.

The system prompt is meticulously designed to include instructions that guide the model toward producing responses enriched with mechanisms for reflection and verification. 
During the RL phase, the model leverages high-temperature sampling to generate responses that integrate patterns from both the R1-generated and original data, even in the absence of explicit system prompts. 
After hundreds of RL steps, the intermediate RL model learns to incorporate R1 patterns, thereby enhancing overall performance strategically.

Upon completing the RL training phase, we implement rejection sampling to curate high-quality SFT data for the final model, where the expert models are used as data generation sources. 
This method ensures that the final training data retains the strengths of DeepSeek-R1 while producing responses that are concise and effective.

\paragraph{Non-Reasoning Data.} 
For non-reasoning data, such as creative writing, role-play, and simple question answering, we utilize DeepSeek-V2.5 to generate responses and enlist human annotators to verify the accuracy and correctness of the data.

\paragraph{SFT Settings.} 
We fine-tune \dsviii{}-Base for two epochs using the SFT dataset, using the cosine decay learning rate scheduling that starts at $5 \times 10^{-6}$ and gradually decreases to $1 \times 10^{-6}$. 
During training, each single sequence is packed from multiple samples. 
However, we adopt a sample masking strategy to ensure that these examples remain isolated and mutually invisible.

\subsection{Reinforcement Learning}

\subsubsection{Reward Model}

We employ a rule-based Reward Model (RM) and a model-based RM in our RL process.

\paragraph{Rule-Based RM.} 
For questions that can be validated using specific rules, we adopt a rule-based reward system to determine the feedback. 
For instance, certain math problems have deterministic results, and we require the model to provide the final answer within a designated format (e.g., in a box), allowing us to apply rules to verify the correctness. 
Similarly, for LeetCode problems, we can utilize a compiler to generate feedback based on test cases. 
By leveraging rule-based validation wherever possible, we ensure a higher level of reliability, as this approach is resistant to manipulation or exploitation.

\paragraph{Model-Based RM.} 
For questions with free-form ground-truth answers, we rely on the reward model to determine whether the response matches the expected ground-truth. 
Conversely, for questions without a definitive ground-truth, such as those involving creative writing, the reward model is tasked with providing feedback based on the question and the corresponding answer as inputs. 
The reward model is trained from the \dsviii{} SFT checkpoints. 
To enhance its reliability, we construct preference data that not only provides the final reward but also includes the chain-of-thought leading to the reward. 
This approach helps mitigate the risk of reward hacking in specific tasks.

\subsubsection{Group Relative Policy Optimization}

Similar to \dsvii{}~\citep{dsvii}, we adopt Group Relative Policy Optimization~(GRPO)~\citep{deepseekmath}, which foregoes the critic model that is typically with the same size as the policy model, and estimates the baseline from group scores instead. 
Specifically, for each question $q$, GRPO samples a group of outputs $\{o_1, o_2, \cdots, o_G\}$ from the old policy model $\pi_{\theta_{old}}$ and then optimizes the policy model $\pi_{\theta}$ by maximizing the following objective:
\begin{equation}
\begin{split}
    \mathcal{J}_{GRPO}(\theta) &= \mathbb{E}{[q \sim P(Q), \{o_i\}_{i=1}^G \sim \pi_{\theta_{old}}(O|q)]}  \\
    & \frac{1}{G}\sum_{i=1}^G \left( \min \left( \frac{\pi_\theta(o_i |q)}{\pi_{\theta_{old}}(o_i |q)} A_i, \text{clip} \left( \frac{\pi_\theta(o_i |q)}{\pi_{\theta_{old}}(o_i |q)}, 1 - \epsilon, 1 + \epsilon \right)  A_i \right) - \beta \mathbb{D}_{KL}\left(\pi_{\theta} || \pi_{ref}\right)\right) ,
\end{split}
\label{eq:GRPO-obj}
\end{equation}
\begin{equation}
    \mathbb{D}_{KL}\left(\pi_{\theta} || \pi_{ref}\right) = \frac{\pi_{ref}(o_i|q)}{\pi_{\theta}(o_i|q)}- \log\frac{\pi_{ref}(o_i|q)}{\pi_{\theta}(o_i|q)} - 1,
\end{equation}
where $\epsilon$ and $\beta$ are hyper-parameters; 
$\pi_{ref}$ is the reference model; 
and $A_i$ is the advantage, derived from the rewards $\{r_1, r_2, \ldots, r_G\}$ corresponding to the outputs within each group:
\begin{equation}
    A_i = \frac{r_i - {\operatorname{mean}(\{r_1, r_2, \cdots, r_G\})}}{{\operatorname{std}(\{r_1, r_2, \cdots, r_G\})}}.
\end{equation}

We incorporate prompts from diverse domains, such as coding, math, writing, role-playing, and question answering, during the RL process. 
This approach not only aligns the model more closely with human preferences but also enhances performance on benchmarks, especially in scenarios where available SFT data are limited.

\subsection{Evaluations}

\subsubsection{Evaluation Settings}

\paragraph{Evaluation Benchmarks.}
Apart from the benchmark we used for base model testing, we further evaluate instructed models on IFEval~\citep{IFeval}, FRAMES~\citep{frames}, LongBench v2~\citep{bai2024longbench2}, GPQA~\citep{gpqa}, SimpleQA~\citep{simpleqa}, C-SimpleQA~\citep{csimpleqa}, SWE-Bench Verified~\citep{swe_verified}, Aider~\footnote{\url{https://aider.chat}}, LiveCodeBench~\citep{livecodebench} (questions from August 2024 to November 2024), Codeforces~\footnote{\url{https://codeforces.com}}, Chinese National High School Mathematics Olympiad (CNMO 2024)\footnote{\url{https://www.cms.org.cn/Home/comp/comp/cid/12.html}}, and American Invitational Mathematics Examination 2024 (AIME 2024)~\citep{AIME2024}. 

\paragraph{Compared Baselines.}
We conduct comprehensive evaluations of our chat model against several strong baselines, including \dsvii{}-0506, DeepSeek-V2.5-0905, Qwen2.5 72B Instruct, LLaMA-3.1 405B Instruct, Claude-Sonnet-3.5-1022, and GPT-4o-0513. 
For the \dsvii{} model series, we select the most representative variants for comparison. 
For closed-source models, evaluations are performed through their respective APIs.

\paragraph{Detailed Evaluation Configurations.}
For standard benchmarks including MMLU, DROP, GPQA, and SimpleQA, we adopt the evaluation prompts from the simple-evals framework\footnote{\url{https://github.com/openai/simple-evals}}. 
We utilize the Zero-Eval prompt format~\citep{Lin_ZeroEval_A_Unified_2024} for MMLU-Redux in a zero-shot setting. 
For other datasets, we follow their original evaluation protocols with default prompts as provided by the dataset creators.
For code and math benchmarks, the HumanEval-Mul dataset includes 8 mainstream programming languages (Python, Java, Cpp, C\#, JavaScript, TypeScript, PHP, and Bash) in total. 
We use CoT and non-CoT methods to evaluate model performance on LiveCodeBench, where the data are collected from August 2024 to November 2024. 
The Codeforces dataset is measured using the percentage of competitors. 
SWE-Bench verified is evaluated using the agentless framework~\citep{agentless}. 
We use the ``diff'' format to evaluate the Aider-related benchmarks. 
For mathematical assessments, AIME and CNMO 2024 are evaluated with a temperature of 0.7, and the results are averaged over 16 runs, while MATH-500 employs greedy decoding. 
We allow all models to output a maximum of 8192 tokens for each benchmark. 

\input{tables/chat_evaluation}

\subsubsection{Standard Evaluation}

Table~\ref{tab:chat} presents the evaluation results, showcasing that \dsviii{} stands as the best-performing open-source model. 
Additionally, it is competitive against frontier closed-source models like GPT-4o and Claude-3.5-Sonnet. 

\paragraph{English Benchmarks.}
MMLU is a widely recognized benchmark designed to assess the performance of large language models, across diverse knowledge domains and tasks. 
\dsviii{} demonstrates competitive performance, standing on par with top-tier models such as LLaMA-3.1-405B, GPT-4o, and Claude-Sonnet 3.5, while significantly outperforming Qwen2.5 72B. 
Moreover, \dsviii{} excels in MMLU-Pro, a more challenging educational knowledge benchmark, where it closely trails Claude-Sonnet 3.5. 
On MMLU-Redux, a refined version of MMLU with corrected labels, \dsviii{} surpasses its peers. 
In addition, on GPQA-Diamond, a PhD-level evaluation testbed, \dsviii{} achieves remarkable results, ranking just behind Claude 3.5 Sonnet and outperforming all other competitors by a substantial margin.

In long-context understanding benchmarks such as DROP, LongBench v2, and FRAMES, \dsviii{} continues to demonstrate its position as a top-tier model. 
It achieves an impressive 91.6 F1 score in the 3-shot setting on DROP, outperforming all other models in this category. 
On FRAMES, a benchmark requiring question-answering over 100k token contexts, \dsviii{} closely trails GPT-4o while outperforming all other models by a significant margin. 
This demonstrates the strong capability of \dsviii{} in handling extremely long-context tasks. 
The long-context capability of \dsviii{} is further validated by its best-in-class performance on LongBench v2, a dataset that was released just a few weeks before the launch of DeepSeek V3. 
On the factual knowledge benchmark, SimpleQA, \dsviii{} falls behind GPT-4o and Claude-Sonnet, primarily due to its design focus and resource allocation. 
\dsviii{} assigns more training tokens to learn Chinese knowledge, leading to exceptional performance on the C-SimpleQA. 
On the instruction-following benchmark, \dsviii{} significantly outperforms its predecessor, \dsvii{}-series, highlighting its improved ability to understand and adhere to user-defined format constraints.

\paragraph{Code and Math Benchmarks.}
Coding is a challenging and practical task for LLMs, encompassing engineering-focused tasks like SWE-Bench-Verified and Aider, as well as algorithmic tasks such as HumanEval and LiveCodeBench. 
In engineering tasks, \dsviii{} trails behind Claude-Sonnet-3.5-1022 but significantly outperforms open-source models. 
The open-source \dsviii{} is expected to foster advancements in coding-related engineering tasks. 
By providing access to its robust capabilities, \dsviii{} can drive innovation and improvement in areas such as software engineering and algorithm development, empowering developers and researchers to push the boundaries of what open-source models can achieve in coding tasks. 
In algorithmic tasks, \dsviii{} demonstrates superior performance, outperforming all baselines on benchmarks like HumanEval-Mul and LiveCodeBench. 
This success can be attributed to its advanced knowledge distillation technique, which effectively enhances its code generation and problem-solving capabilities in algorithm-focused tasks.

On math benchmarks, \dsviii{} demonstrates exceptional performance, significantly surpassing baselines and setting a new state-of-the-art for non-o1-like models. 
Specifically, on AIME, MATH-500, and CNMO 2024, \dsviii{} outperforms the second-best model, Qwen2.5 72B, by approximately 10\% in absolute scores, which is a substantial margin for such challenging benchmarks.
This remarkable capability highlights the effectiveness of the distillation technique from DeepSeek-R1, which has been proven highly beneficial for non-o1-like models.

\paragraph{Chinese Benchmarks.}
Qwen and DeepSeek are two representative model series with robust support for both Chinese and English. 
On the factual benchmark Chinese SimpleQA, \dsviii{} surpasses Qwen2.5-72B by 16.4 points, despite Qwen2.5 being trained on a larger corpus compromising 18T tokens, which are 20\% more than the 14.8T tokens that \dsviii{} is pre-trained on.

On C-Eval, a representative benchmark for Chinese educational knowledge evaluation, and CLUEWSC (Chinese Winograd Schema Challenge), \dsviii{} and Qwen2.5-72B exhibit similar performance levels, indicating that both models are well-optimized for challenging Chinese-language reasoning and educational tasks.

\begin{table}[t]
    \centering
    \begin{tabular}{c | c c}
    \toprule
    \textbf{Model} & \textbf{Arena-Hard} & \textbf{AlpacaEval 2.0} \\
    \midrule
    DeepSeek-V2.5-0905 & 76.2 & 50.5\\
    Qwen2.5-72B-Instruct & 81.2 & 49.1  \\
    LLaMA-3.1 405B & 69.3 & 40.5\\
    GPT-4o-0513 & 80.4 &  51.1 \\
    Claude-Sonnet-3.5-1022 & 85.2  & 52.0 \\
    DeepSeek-V3 & \textbf{85.5} & \textbf{70.0} \\
    \bottomrule
    \end{tabular}
    \caption{
    English open-ended conversation evaluations. 
    For AlpacaEval 2.0, we use the length-controlled win rate as the metric. 
    }
    \label{tab:open} 
\end{table}

\subsubsection{Open-Ended Evaluation}

In addition to standard benchmarks, we also evaluate our models on open-ended generation tasks using LLMs as judges, with the results shown in Table~\ref{tab:open}. 
Specifically, we adhere to the original configurations of AlpacaEval 2.0~\citep{alpaca2.0} and Arena-Hard~\citep{li2024crowdsourced}, which leverage GPT-4-Turbo-1106 as judges for pairwise comparisons. 
On Arena-Hard, \dsviii{} achieves an impressive win rate of over 86\% against the baseline GPT-4-0314, performing on par with top-tier models like Claude-Sonnet-3.5-1022. 
This underscores the robust capabilities of \dsviii{}, especially in dealing with complex prompts, including coding and debugging tasks. 
Furthermore, \dsviii{} achieves a groundbreaking milestone as the first open-source model to surpass 85\% on the Arena-Hard benchmark. 
This achievement significantly bridges the performance gap between open-source and closed-source models, setting a new standard for what open-source models can accomplish in challenging domains.

Similarly, \dsviii{} showcases exceptional performance on AlpacaEval 2.0, outperforming both closed-source and open-source models. 
This demonstrates its outstanding proficiency in writing tasks and handling straightforward question-answering scenarios. 
Notably, it surpasses DeepSeek-V2.5-0905 by a significant margin of 20\%, highlighting substantial improvements in tackling simple tasks and showcasing the effectiveness of its advancements.

\subsubsection{\dsviii{} as a Generative Reward Model}

We compare the judgment ability of \dsviii{} with state-of-the-art models, namely GPT-4o and Claude-3.5. 
Table~\ref{tab:rewardbench} presents the performance of these models in RewardBench \citep{lambert2024rewardbench}.
\dsviii{} achieves performance on par with the best versions of GPT-4o-0806 and Claude-3.5-Sonnet-1022, while surpassing other versions. 
Additionally, the judgment ability of \dsviii{} can also be enhanced by the voting technique. 
Therefore, we employ \dsviii{} along with voting to offer self-feedback on open-ended questions, thereby improving the effectiveness and robustness of the alignment process.

\begin{table}[t]
    \centering
    \begin{tabular}{c | c c c c c}
    \toprule
    \textbf{Model} & \textbf{Chat} & \textbf{Chat-Hard} & \textbf{Safety} & \textbf{Reasoning} & \textbf{Average} \\
    \midrule
    GPT-4o-0513 & 96.6 & 70.4 & 86.7 & 84.9 & 84.7\\
    GPT-4o-0806 & 96.1 & 76.1 & 88.1 & 86.6 & 86.7\\
    GPT-4o-1120 & 95.8 & 71.3 & 86.2 & 85.2 & 84.6\\
    \midrule 
    Claude-3.5-sonnet-0620 & 96.4 & 74.0 & 81.6 & 84.7 & 84.2\\
    Claude-3.5-sonnet-1022 & 96.4 & 79.7 & 91.1 & 87.6 & 88.7\\
    \midrule 
    \dsviii{} & 96.9 & 79.8 & 87.0 & 84.3 & 87.0\\
    \dsviii{} (maj@6) & 96.9 & 82.6 & 89.5 & 89.2 & 89.6\\
    \bottomrule
    \end{tabular}
    \caption{Performances of GPT-4o, Claude-3.5-sonnet and \dsviii{} on RewardBench.}
    \label{tab:rewardbench} 
\end{table}

\subsection{Discussion}

\subsubsection{Distillation from DeepSeek-R1} 

We ablate the contribution of distillation from DeepSeek-R1 based on DeepSeek-V2.5. 
The baseline is trained on short CoT data, whereas its competitor uses data generated by the expert checkpoints described above. 

Table~\ref{tab:distill} demonstrates the effectiveness of the distillation data, showing significant improvements in both LiveCodeBench and MATH-500 benchmarks. 
Our experiments reveal an interesting trade-off: the distillation leads to better performance but also substantially increases the average response length. 
To maintain a balance between model accuracy and computational efficiency, we carefully selected optimal settings for \dsviii{} in distillation.

Our research suggests that knowledge distillation from reasoning models presents a promising direction for post-training optimization. 
While our current work focuses on distilling data from mathematics and coding domains, this approach shows potential for broader applications across various task domains. 
The effectiveness demonstrated in these specific areas indicates that long-CoT distillation could be valuable for enhancing model performance in other cognitive tasks requiring complex reasoning. 
Further exploration of this approach across different domains remains an important direction for future research.

\begin{table}[t]
    \centering
    \begin{tabular}{c|cc|cc}
    \toprule
    \multirow{2}{*}{\textbf{Model}} & \multicolumn{2}{c}{\textbf{LiveCodeBench-CoT}} & \multicolumn{2}{c}{\textbf{MATH-500}} \\

     & Pass@1 & Length & Pass@1 & Length\\
    \midrule
     DeepSeek-V2.5 Baseline & 31.1 & 718 & 74.6 & 769\\
     DeepSeek-V2.5 +R1 Distill & 37.4 & 783 &83.2& 1510\\
    \bottomrule
    \end{tabular}
    \caption{
    The contribution of distillation from DeepSeek-R1. 
    The evaluation settings of LiveCodeBench and MATH-500 are the same as in Table~\ref{tab:chat}.
    }
    \label{tab:distill} 
\end{table}

\subsubsection{Self-Rewarding}

Rewards play a pivotal role in RL, steering the optimization process. 
In domains where verification through external tools is straightforward, such as some coding or mathematics scenarios, RL demonstrates exceptional efficacy.
However, in more general scenarios, constructing a feedback mechanism through hard coding is impractical. 
During the development of \dsviii{}, for these broader contexts, we employ the constitutional AI approach~\citep{bai2022constitutional}, leveraging the voting evaluation results of \dsviii{} itself as a feedback source. 
This method has produced notable alignment effects, significantly enhancing the performance of \dsviii{} in subjective evaluations. 
By integrating additional constitutional inputs, \dsviii{} can optimize towards the constitutional direction. 
We believe that this paradigm, which combines supplementary information with LLMs as a feedback source, is of paramount importance. 
The LLM serves as a versatile processor capable of transforming unstructured information from diverse scenarios into rewards, ultimately facilitating the self-improvement of LLMs. 
Beyond self-rewarding, we are also dedicated to uncovering other general and scalable rewarding methods to consistently advance the model capabilities in general scenarios.

\subsubsection{Multi-Token Prediction Evaluation} 

Instead of predicting just the next single token, \dsviii{} predicts the next 2 tokens through the MTP technique. 
Combined with the framework of speculative decoding~\citep{speculative_google,speculative_xhm}, it can significantly accelerate the decoding speed of the model. 
A natural question arises concerning the acceptance rate of the additionally predicted token. 
Based on our evaluation, the acceptance rate of the second token prediction ranges between 85\% and 90\% across various generation topics, demonstrating consistent reliability. 
This high acceptance rate enables \dsviii{} to achieve a significantly improved decoding speed, delivering 1.8 times TPS (Tokens Per Second).

\section{Conclusion, Limitations, and Future Directions}
\label{sec:conclusion}

In this paper, we introduce \dsviii{}, a large MoE language model with 671B total parameters and 37B activated parameters, trained on 14.8T tokens. 
In addition to the \dsattn{} and \dsmoe{} architectures, it also pioneers an auxiliary-loss-free strategy for load balancing and sets a multi-token prediction training objective for stronger performance.
The training of \dsviii{} is cost-effective due to the support of FP8 training and meticulous engineering optimizations. 
The post-training also makes a success in distilling the reasoning capability from the DeepSeek-R1 series of models.
Comprehensive evaluations demonstrate that DeepSeek-V3 has emerged as the strongest open-source model currently available, and achieves performance comparable to leading closed-source models like GPT-4o and Claude-3.5-Sonnet. 
Despite its strong performance, it also maintains economical training costs. 
It requires only 2.788M H800 GPU hours for its full training, including pre-training, context length extension, and post-training. 

While acknowledging its strong performance and cost-effectiveness, we also recognize that \dsviii{} has some limitations, especially on the deployment.
Firstly, to ensure efficient inference, the recommended deployment unit for \dsviii{} is relatively large, which might pose a burden for small-sized teams.
Secondly, although our deployment strategy for \dsviii{} has achieved an end-to-end generation speed of more than two times that of \dsvii{}, there still remains potential for further enhancement.
Fortunately, these limitations are expected to be naturally addressed with the development of more advanced hardware.

DeepSeek consistently adheres to the route of open-source models with longtermism, aiming to steadily approach the ultimate goal of AGI (Artificial General Intelligence).
In the future, we plan to strategically invest in research across the following directions.
\begin{itemize}
    \item 
    We will consistently study and refine our model architectures, aiming to further improve both the training and inference efficiency, striving to approach efficient support for infinite context length. 
    Additionally, we will try to break through the architectural limitations of Transformer, thereby pushing the boundaries of its modeling capabilities.
    \item 
    We will continuously iterate on the quantity and quality of our training data, and explore the incorporation of additional training signal sources, aiming to drive data scaling across a more comprehensive range of dimensions.
    \item 
    We will consistently explore and iterate on the deep thinking capabilities of our models, aiming to enhance their intelligence and problem-solving abilities by expanding their reasoning length and depth.
    \item 
    We will explore more comprehensive and multi-dimensional model evaluation methods to prevent the tendency towards optimizing a fixed set of benchmarks during research, which may create a misleading impression of the model capabilities and affect our foundational assessment.
\end{itemize}

\bibliography{main}

\newpage
\appendix

\section*{Appendix}

\section{Contributions and Acknowledgments}

\definecolor{damaiblue}{RGB}{0, 0, 100}
\definecolor{damaigreen}{RGB}{0, 100, 0}
\definecolor{damaired}{RGB}{100, 0, 0}

\begin{multicols}{2} % 开始两栏环境
\noindent
\textbf{\color{damaiblue} Research \& Engineering} \\
\color{damaiblue} Aixin Liu \\
\color{damaiblue} Bing Xue \\
\color{damaiblue} Bingxuan Wang \\
\color{damaiblue} Bochao Wu \\
\color{damaiblue} Chengda Lu \\
\color{damaiblue} Chenggang Zhao \\
\color{damaiblue} Chengqi Deng \\
\color{damaiblue} Chenyu Zhang* \\
\color{damaiblue} Chong Ruan \\
\color{damaiblue} Damai Dai \\
\color{damaiblue} Daya Guo \\
\color{damaiblue} Dejian Yang \\
\color{damaiblue} Deli Chen \\
\color{damaiblue} Erhang Li \\
\color{damaiblue} Fangyun Lin \\
\color{damaiblue} Fucong Dai \\
\color{damaiblue} Fuli Luo* \\
\color{damaiblue} Guangbo Hao \\
\color{damaiblue} Guanting Chen \\
\color{damaiblue} Guowei Li \\
\color{damaiblue} H. Zhang \\
\color{damaiblue} Han Bao* \\
\color{damaiblue} Hanwei Xu \\
\color{damaiblue} Haocheng Wang* \\
\color{damaiblue} Haowei Zhang \\
\color{damaiblue} Honghui Ding \\
\color{damaiblue} Huajian Xin* \\
\color{damaiblue} Huazuo Gao \\
\color{damaiblue} Hui Qu \\
\color{damaiblue} Jianzhong Guo \\
\color{damaiblue} Jiashi Li \\
\color{damaiblue} Jiawei Wang* \\
\color{damaiblue} Jingchang Chen \\
\color{damaiblue} Jingyang Yuan \\
\color{damaiblue} Junjie Qiu \\
\color{damaiblue} Junlong Li \\
\color{damaiblue} Junxiao Song \\
\color{damaiblue} Kai Dong \\
\color{damaiblue} Kai Hu* \\
\color{damaiblue} Kaige Gao \\
\color{damaiblue} Kang Guan \\
\color{damaiblue} Kexin Huang \\
\color{damaiblue} Kuai Yu \\
\color{damaiblue} Lean Wang \\
\color{damaiblue} Lecong Zhang \\
\color{damaiblue} Liang Zhao \\
\color{damaiblue} Litong Wang \\
\color{damaiblue} Liyue Zhang \\
\color{damaiblue} Mingchuan Zhang \\
\color{damaiblue} Minghua Zhang \\
\color{damaiblue} Minghui Tang \\
\color{damaiblue} Panpan Huang \\
\color{damaiblue} Peiyi Wang \\
\color{damaiblue} Qiancheng Wang \\
\color{damaiblue} Qihao Zhu \\
\color{damaiblue} Qinyu Chen \\
\color{damaiblue} Qiushi Du \\
\color{damaiblue} Ruiqi Ge \\
\color{damaiblue} Ruisong Zhang \\
\color{damaiblue} Ruizhe Pan \\
\color{damaiblue} Runji Wang \\
\color{damaiblue} Runxin Xu \\
\color{damaiblue} Ruoyu Zhang \\
\color{damaiblue} Shanghao Lu \\
\color{damaiblue} Shangyan Zhou \\
\color{damaiblue} Shanhuang Chen \\
\color{damaiblue} Shengfeng Ye \\
\color{damaiblue} Shirong Ma \\
\color{damaiblue} Shiyu Wang \\
\color{damaiblue} Shuiping Yu \\
\color{damaiblue} Shunfeng Zhou \\
\color{damaiblue} Shuting Pan \\
\color{damaiblue} Tao Yun \\
\color{damaiblue} Tian Pei \\
\color{damaiblue} Wangding Zeng \\
\color{damaiblue} Wanjia Zhao* \\
\color{damaiblue} Wen Liu \\
\color{damaiblue} Wenfeng Liang \\
\color{damaiblue} Wenjun Gao \\
\color{damaiblue} Wenqin Yu \\
\color{damaiblue} Wentao Zhang \\
\color{damaiblue} Xiao Bi \\
\color{damaiblue} Xiaodong Liu \\
\color{damaiblue} Xiaohan Wang \\
\color{damaiblue} Xiaokang Chen \\
\color{damaiblue} Xiaokang Zhang \\
\color{damaiblue} Xiaotao Nie \\
\color{damaiblue} Xin Cheng \\
\color{damaiblue} Xin Liu \\
\color{damaiblue} Xin Xie \\
\color{damaiblue} Xingchao Liu \\
\color{damaiblue} Xingkai Yu \\
\color{damaiblue} Xinyu Yang \\
\color{damaiblue} Xinyuan Li \\
\color{damaiblue} Xuecheng Su \\
\color{damaiblue} Xuheng Lin \\
\color{damaiblue} Y.K. Li \\
\color{damaiblue} Y.Q. Wang \\
\color{damaiblue} Y.X. Wei \\
\color{damaiblue} Yang Zhang \\
\color{damaiblue} Yanhong Xu \\
\color{damaiblue} Yao Li \\
\color{damaiblue} Yao Zhao \\
\color{damaiblue} Yaofeng Sun \\
\color{damaiblue} Yaohui Wang \\
\color{damaiblue} Yi Yu \\
\color{damaiblue} Yichao Zhang \\
\color{damaiblue} Yifan Shi \\
\color{damaiblue} Yiliang Xiong \\
\color{damaiblue} Ying He \\
\color{damaiblue} Yishi Piao \\
\color{damaiblue} Yisong Wang \\
\color{damaiblue} Yixuan Tan \\
\color{damaiblue} Yiyang Ma* \\
\color{damaiblue} Yiyuan Liu \\
\color{damaiblue} Yongqiang Guo \\
\color{damaiblue} Yu Wu \\
\color{damaiblue} Yuan Ou \\
\color{damaiblue} Yuduan Wang \\
\color{damaiblue} Yue Gong \\
\color{damaiblue} Yuheng Zou \\
\color{damaiblue} Yujia He \\
\color{damaiblue} Yunfan Xiong \\
\color{damaiblue} Yuxiang Luo \\
\color{damaiblue} Yuxiang You \\
\color{damaiblue} Yuxuan Liu \\
\color{damaiblue} Yuyang Zhou \\
\color{damaiblue} Z.F. Wu \\
\color{damaiblue} Z.Z. Ren \\
\color{damaiblue} Zehui Ren \\
\color{damaiblue} Zhangli Sha \\
\color{damaiblue} Zhe Fu \\
\color{damaiblue} Zhean Xu \\
\color{damaiblue} Zhenda Xie \\
\color{damaiblue} Zhengyan Zhang \\
\color{damaiblue} Zhewen Hao \\
\color{damaiblue} Zhibin Gou \\
\color{damaiblue} Zhicheng Ma \\
\color{damaiblue} Zhigang Yan \\
\color{damaiblue} Zhihong Shao \\
\color{damaiblue} Zhiyu Wu \\
\color{damaiblue} Zhuoshu Li \\
\color{damaiblue} Zihui Gu \\
\color{damaiblue} Zijia Zhu \\
\color{damaiblue} Zijun Liu* \\
\color{damaiblue} Zilin Li \\
\color{damaiblue} Ziwei Xie \\
\color{damaiblue} Ziyang Song \\
\color{damaiblue} Ziyi Gao \\
\color{damaiblue} Zizheng Pan \\

\noindent
\textbf{\color{damaigreen} Data Annotation} \\
\color{damaigreen} Bei Feng \\
\color{damaigreen} Hui Li \\
\color{damaigreen} J.L. Cai \\
\color{damaigreen} Jiaqi Ni \\
\color{damaigreen} Lei Xu \\
\color{damaigreen} Meng Li \\
\color{damaigreen} Ning Tian \\
\color{damaigreen} R.J. Chen \\
\color{damaigreen} R.L. Jin \\
\color{damaigreen} Ruyi Chen \\
\color{damaigreen} S.S. Li \\
\color{damaigreen} Shuang Zhou \\
\color{damaigreen} Tianyu Sun \\
\color{damaigreen} X.Q. Li \\
\color{damaigreen} Xiangyue Jin \\
\color{damaigreen} Xiaojin Shen \\
\color{damaigreen} Xiaosha Chen \\
\color{damaigreen} Xiaowen Sun \\
\color{damaigreen} Xiaoxiang Wang \\
\color{damaigreen} Xinnan Song \\
\color{damaigreen} Xinyi Zhou \\
\color{damaigreen} Y.X. Zhu \\
\color{damaigreen} Yanhong Xu \\
\color{damaigreen} Yanping Huang \\
\color{damaigreen} Yaohui Li \\
\color{damaigreen} Yi Zheng \\
\color{damaigreen} Yuchen Zhu \\
\color{damaigreen} Yunxian Ma \\
\color{damaigreen} Zhen Huang \\
\color{damaigreen} Zhipeng Xu \\
\color{damaigreen} Zhongyu Zhang \\

\noindent
\textbf{\color{damaired} Business \& Compliance} \\
\color{damaired} Dongjie Ji \\
\color{damaired} Jian Liang \\
\color{damaired} Jin Chen \\
\color{damaired} Leyi Xia \\
\color{damaired} Miaojun Wang \\
\color{damaired} Mingming Li \\
\color{damaired} Peng Zhang \\
\color{damaired} Shaoqing Wu \\
\color{damaired} Shengfeng Ye \\
\color{damaired} T. Wang \\
\color{damaired} W.L. Xiao \\
\color{damaired} Wei An \\
\color{damaired} Xianzu Wang \\
\color{damaired} Xinxia Shan \\
\color{damaired} Ying Tang \\
\color{damaired} Yukun Zha \\
\color{damaired} Yuting Yan \\
\color{damaired} Zhen Zhang \\

\end{multicols} % 结束两栏环境

Within each role, authors are listed alphabetically by the first name. 
Names marked with * denote individuals who have departed from our team. 

\section{Ablation Studies for Low-Precision Training}
\label{app:fp8}

\begin{figure}[!h]
\centering
\includegraphics[width=0.95\linewidth]{figures/fp8-v.s.-bf16.pdf}
\caption{
    Loss curves comparison between BF16 and FP8 training. 
    Results are smoothed by Exponential Moving Average (EMA) with a coefficient of 0.9.
}
\label{fig:fp8_vs_bf16}
\end{figure}

\subsection{FP8 v.s. BF16 Training}
\label{app:fp8_cp_bf16}
We validate our FP8 mixed precision framework with a comparison to BF16 training on top of two baseline models across different scales. At the small scale, we train a baseline MoE model comprising approximately 16B total parameters on 1.33T tokens. 
At the large scale, we train a baseline MoE model comprising approximately 230B total parameters on around 0.9T tokens. We show the training curves in Figure~\ref{fig:fp8_vs_bf16} and demonstrate that the relative error remains below 0.25\% with our high-precision accumulation and fine-grained quantization strategies.

\subsection{Discussion About Block-Wise Quantization}
\label{app:fp8_blockwise}
Although our tile-wise fine-grained quantization effectively mitigates the error introduced by feature outliers, it requires different groupings for activation quantization, i.e., \texttt{1x128} in forward pass and \texttt{128x1} for backward pass. 
A similar process is also required for the activation gradient. 
A straightforward strategy is to apply block-wise quantization per \texttt{128x128} elements like the way we quantize the model weights.
In this way, only transposition is required for backward.
Therefore, we conduct an experiment where all tensors associated with \texttt{Dgrad} are quantized on a block-wise basis. 
The results reveal that the \texttt{Dgrad} operation which computes the activation gradients and back-propagates to shallow layers in a chain-like manner, is highly sensitive to precision. 
Specifically, block-wise quantization of activation gradients leads to model divergence on an MoE model comprising approximately 16B total parameters, trained for around 300B tokens. 
We hypothesize that this sensitivity arises because activation gradients are highly imbalanced among tokens, resulting in token-correlated outliers~\citep{int4train}. 
These outliers cannot be effectively managed by a block-wise quantization approach.

\section{Expert Specialization Patterns of the 16B Aux-Loss-Based and Aux-Loss-Free Models}
\label{app:detailed_expert_load}
We record the expert load of the 16B auxiliary-loss-based baseline and the auxiliary-loss-free model on the Pile test set. 
The auxiliary-loss-free model tends to have greater expert specialization across all layers, as demonstrated in Figure~\ref{fig:detailed_expert_load}.

\begin{figure}[!t]
    \subfigure[Layers 1-7]{
        \includegraphics[width=0.95\linewidth]{figures/relative_expert_load_multi_1-6.pdf}
    }
\end{figure}

\begin{figure}[!t]
    \ContinuedFloat
    \subfigure[Layers 7-13]{
        \includegraphics[width=0.95\linewidth]{figures/relative_expert_load_multi_7-12.pdf}
    }
\end{figure}

\begin{figure}[!t]
    \ContinuedFloat
    \subfigure[Layers 13-19]{
        \includegraphics[width=0.95\linewidth]{figures/relative_expert_load_multi_13-18.pdf}
    }
\end{figure}

\begin{figure}[!t]
    \ContinuedFloat
    \subfigure[Layers 19-25]{
        \includegraphics[width=0.95\linewidth]{figures/relative_expert_load_multi_19-24.pdf}
    }
\end{figure}

\begin{figure}[!t]
    \ContinuedFloat
    \subfigure[Layers 25-27]{
        \includegraphics[width=0.95\linewidth]{figures/relative_expert_load_multi_25-26.pdf}
    }
    \caption{
        Expert load of auxiliary-loss-free and auxiliary-loss-based models on three domains in the Pile test set. 
        The auxiliary-loss-free model shows greater expert specialization patterns than the auxiliary-loss-based one.
        The relative expert load denotes the ratio between the actual expert load and the theoretically balanced expert load. 
    }
\label{fig:detailed_expert_load}
\end{figure}

\newpage

\end{CJK*}
\end{document} 

