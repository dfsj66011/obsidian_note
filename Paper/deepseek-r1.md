
DeepSeek-R1：通过强化学习激励大型语言模型的推理能力

#### 摘要

我们介绍了第一代推理模型，DeepSeek-R1-Zero 和 DeepSeek-R1。DeepSeek-R1-Zero 是一个通过大规模 RL 训练的模型，没有经过 SFT 作为初步步骤，展现了卓越的推理能力。通过 RL，DeepSeek-R1-Zero 自然涌现出许多强大且有趣的推理行为。然而，它也面临诸如可读性差和语言混用等挑战。为了解决这些问题并进一步提升推理性能，我们引入了 DeepSeek-R1，该模型在 RL 之前结合了多阶段训练和冷启动数据。DeepSeek-R1 在推理任务上的表现可与 OpenAI-o1-1217 相媲美。为了支持研究社区，我们开源了 DeepSeek-R1-Zero、DeepSeek-R1 以及从 DeepSeek-R1 基于 Qwen 和 Llama 蒸馏出的六个密集模型（1.5B、7B、8B、14B、32B、70B）。
<img src="https://arxiv.org/html/2501.12948v1/x1.png" width="500">
**图 1.** DeepSeek-R1 的性能基准


### 1、引言

近年来，LLMs 正在快速迭代和演化，逐步缩小与 AGI 之间的差距。

最近，后训练成为完整训练流程中的重要组成部分。研究表明，它可以在推理任务中提高准确性，对齐社会价值观，并适应用户偏好，同时与预训练相比所需的计算资源相对较少。在推理能力方面，OpenAI 的 o1 系列模型首次通过增加链式思维推理过程的长度引入了推理时的扩展。这种方法在数学、编码和科学推理等多种任务中取得了显著的改进。然而，如何有效地进行测试时扩展仍然是研究界的一个未解难题。此前的多项研究探索了各种方法，包括基于过程的奖励模型、强化学习，以及蒙特卡洛树搜索和束搜索等搜索算法。然而，这些方法都未能达到与 OpenAI 的 o1 系列模型相当的通用推理性能。

在本文中，我们首次尝试使用纯强化学习（RL）来提升语言模型的推理能力。我们的目标是探索 LLMs 在没有任何监督数据的情况下，通过纯 RL 过程自我进化来发展推理能力。具体而言，我们使用 DeepSeek-V3-Base 作为基础模型，并采用 GRPO 作为 RL 框架来提高模型在推理方面的表现。在训练过程中，DeepSeek-R1-Zero 自然涌现出许多强大而有趣的推理行为。经过数千步的 RL 训练，DeepSeek-R1-Zero 在推理基准测试中表现出色。例如，AIME 2024 的 pass@1 得分从 15.6\% 提升到 71.0\%，通过多数投票，得分进一步提高到 86.7\%，达到 OpenAI-o1-0912 的水平。

然而，DeepSeek-R1-Zero 面临可读性差、语言混杂等挑战。为了解决这些问题并进一步提高推理性能，我们引入了 DeepSeek-R1，它结合了少量的冷启动数据和多阶段训练流程。具体来说，我们首先收集数千条冷启动数据以微调 DeepSeek-V3-Base 模型。随后，我们进行类似 DeepSeek-R1-Zero 的面向推理的 RL。当 RL 过程接近收敛时，我们通过对 RL 检查点进行拒绝采样创建新的 SFT 数据，并结合 DeepSeek-V3 在写作、事实问答和自我认知等领域的监督数据，然后重新训练 DeepSeek-V3-Base 模型。在用新数据微调后，检查点进行额外的 RL 过程，考虑所有场景的提示。经过这些步骤，我们获得了一个称为 DeepSeek-R1 的检查点，其性能与 OpenAI-o1-1217 相当。

我们进一步探索从 DeepSeek-R1 到较小密集模型的蒸馏。使用 Qwen2.5-32B 作为基础模型，直接从 DeepSeek-R1 蒸馏的效果优于在其上应用RL。这表明，大型基础模型发现的推理模式对于提高推理能力至关重要。我们开源了蒸馏后的 Qwen 和 Llama 系列。值得注意的是，我们的蒸馏 14B 模型大幅超越了最新开源的 QwQ-32B-Preview，而蒸馏后的 32B 和 70B 模型在密集模型的推理基准测试中创下了新纪录。

#### 1.1 贡献

**后训练：基础模型的大规模强化学习**

* 我们直接将 RL 应用于基础模型，而不依赖于监督微调（SFT）作为初步步骤。这种方法允许模型探索链式思维（CoT）以解决复杂问题，从而开发出 DeepSeek-R1-Zero。DeepSeek-R1-Zero 展示了自我验证、反思和生成长链式思维的能力，这对研究界来说是一个重要的里程碑。值得注意的是，这是首次公开研究验证LLM的推理能力可以通过纯RL激励，而无需 SFT。这一突破为该领域的未来进步铺平了道路。
* 我们介绍了开发 DeepSeek-R1 的流程。该流程包括两个RL阶段，旨在发现改进的推理模式并与人类偏好对齐，以及两个 SFT 阶段，作为模型推理和非推理能力的种子。我们相信，这一流程将通过创建更好的模型来惠及行业。

**蒸馏：小模型也可以很强大**

* 我们证明了大型模型的推理模式可以被蒸馏到小模型中，结果显示其性能优于在小模型上通过 RL 发现的推理模式。开源的 DeepSeek-R1 及其 API 将有助于研究界在未来蒸馏出更好的小模型。
* 使用 DeepSeek-R1 生成的推理数据，我们微调了几个在研究界广泛使用的密集模型。评估结果表明，蒸馏后的较小密集模型在基准测试中表现出色。DeepSeek-R1-Distill-Qwen-7B 在 AIME 2024上达到 55.5\%，超越了 QwQ-32B-Preview。此外，DeepSeek-R1-Distill-Qwen-32B 在 AIME 2024 上得分 72.6\%，在 MATH-500 上得分 94.3\%，在 LiveCodeBench 上得分 57.2\%。这些结果显著超越了之前的开源模型，并与 o1-mini 相当。我们向社区开源了基于 Qwen2.5 和 Llama3 系列的 1.5B、7B、8B、14B、32B 和 70B 检查点。

#### 1.2 评估结果总结

* **推理任务：** (1) DeepSeek-R1 在 AIME 2024 上取得了 79.8\% 的 Pass@1 得分，略高于 OpenAI-o1-1217。在 MATH-500上，它获得了 97.3\% 的优异成绩，与 OpenAI-o1-1217 表现相当，显著超越其他模型。(2) 在编码相关任务中，DeepSeek-R1 在代码竞赛任务中表现出专家水平，Codeforces Elo 评分达到 2029，超过 96.3\% 的参赛人类。在工程相关任务中，DeepSeek-R1 表现略优于 DeepSeek-V3，这对开发人员在实际任务中有帮助。
* **知识：** 在 MMLU、MMLU-Pro 和 GPQA Diamond 等基准测试中，DeepSeek-R1 取得了卓越的成绩，显著超越 DeepSeek-V3，分别在 MMLU 上得分 90.8\%、MMLU-Pro 上得分 84.0\%、GPQA Diamond 上得分 71.5\%。虽然在这些基准测试中其表现略低于OpenAI-o1-1217，但 DeepSeek-R1 超越了其他闭源模型，展示了其在教育任务中的竞争优势。在事实基准 SimpleQA上，DeepSeek-R1 表现优于 DeepSeek-V3，显示出其处理基于事实查询的能力。在这一基准上也观察到类似趋势，OpenAI-o1 表现优于 4o。
* **其他：** DeepSeek-R1 在创意写作、一般问答、编辑、摘要等广泛任务中也表现出色。它在 AlpacaEval 2.0 上实现了 87.6\% 的长度控制胜率，在 ArenaHard 上实现了 92.3\% 的胜率，展示了其智能处理非考试导向查询的强大能力。此外，DeepSeek-R1 在需要长上下文理解的任务中表现出色，在长上下文基准测试中显著超越 DeepSeek-V3。

### 2、方法

#### 2.1 概述

以往的研究严重依赖大量的监督数据来提升模型性能。在本研究中，我们展示了即使不使用 SFT 作为冷启动，通过大规模 RL 也能显著提高推理能力。此外，结合少量冷启动数据可以进一步提升性能。在接下来的部分中，我们将介绍：(1) DeepSeek-R1-Zero，它在没有任何 SFT 数据的情况下直接对基础模型应用 RL；(2) DeepSeek-R1，它从使用数千个长链式思维（CoT）示例微调的检查点开始应用 RL；(3) 将 DeepSeek-R1 的推理能力提炼到小型稠密模型中。


#### 2.2 DeepSeek-R1-Zero：基础模型上的强化学习

强化学习在推理任务中表现出了显著的效果，我们的前期工作也证明了这一点。然而，这些工作严重依赖于监督数据，而这些数据的收集非常耗时。在本节中，我们探讨了 LLM 在**没有任何监督数据**的情况下发展推理能力的潜力，重点关注它们通过纯强化学习过程的自我进化。我们首先简要概述我们的 RL 算法，然后展示一些令人兴奋的结果，希望这能为社区提供有价值的见解。

##### 2.2.1 强化学习算法

**群体相对策略优化**

为了节省 RL 的训练成本，我们采用了群体相对策略优化（GRPO），该方法放弃了通常与策略模型大小相同的评论模型，而是从群体得分中估计基线。具体来说，对于每个问题 $q$，GRPO 从旧策略 $\pi_{\theta_{old}}$ 中采样一组输出 $\{o_1, o_2, \cdots, o_G\}$，然后通过最大化以下目标来优化策略模型 $\pi_{\theta}$：$$
\begin{equation}
\begin{split}
    \mathcal{J}_{GRPO}(\theta) &= \mathbb{E}{[q \sim P(Q), \{o_i\}_{i=1}^G \sim \pi_{\theta_{old}}(O|q)]}  \\[1.2ex]
    & \frac{1}{G}\sum_{i=1}^G \left( \min \left( \frac{\pi_\theta(o_i |q)}{\pi_{\theta_{old}}(o_i |q)} A_i, \text{clip} \left( \frac{\pi_\theta(o_i |q)}{\pi_{\theta_{old}}(o_i |q)}, 1 - \epsilon, 1 + \epsilon \right)  A_i \right) - \beta \mathbb{D}_{KL}\left(\pi_{\theta} || \pi_{ref}\right)\right) ,\\[1.5ex]
    &\mathbb{D}_{KL}\left(\pi_{\theta} || \pi_{ref}\right) = \frac{\pi_{ref}(o_i|q)}{\pi_{\theta}(o_i|q)}- \log\frac{\pi_{ref}(o_i|q)}{\pi_{\theta}(o_i|q)} - 1,
\end{split}\tag{1,2}
\end{equation}$$
其中，$\epsilon$ 和 $\beta$ 是超参数，$A_i$ 是优势，通过计算与每个群组内输出对应的一组奖励 $\{r_1, r_2, \ldots, r_G\}$ 得到：$$
\begin{equation}
    A_i = \frac{r_i - {\mathrm mean(\{r_1, r_2, \cdots, r_G\})}}{{\mathrm std(\{r_1, r_2, \cdots, r_G\})}}. \tag{3}
\end{equation}$$
```ini
========================================================================
A conversation between User and Assistant. The user asks a question, and the Assistant solves it. 
The assistant first thinks about the reasoning process in the mind and then provides the user 
with the answer.The reasoning process and answer are enclosed within <think> </think> and 
<answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> 
<answer> answer here </answer>. User: prompt. Assistant: 
========================================================================
```
表 1: DeepSeek-R1-Zero 的模板。训练期间，<font color="red">prompt</font> 将被替换为具体的推理问题。


##### 2.2.2 奖励建模

奖励是训练信号的来源，决定了 RL 的优化方向。为了训练 DeepSeek-R1-Zero，我们采用了基于规则的奖励系统，主要由两种类型的奖励组成：

* **准确性奖励**：准确性奖励模型评估响应是否正确。例如，在具有确定性结果的数学问题中，模型需要以指定格式（例如，放在一个框中）提供最终答案，从而可以可靠地基于规则验证正确性。同样，对于 LeetCode 问题，可以使用编译器根据预定义的测试用例生成反馈。
* **格式奖励**：除了准确性奖励模型外，我们还使用格式奖励模型，要求模型将其思考过程放在 `<think>` 和 `</think>` 标签之间。

在开发 DeepSeek-R1-Zero 时，我们没有应用结果或过程神经奖励模型，因为我们发现神经奖励模型在大规模强化学习过程中可能会遭遇奖励作弊问题，并且重新训练奖励模型需要额外的训练资源，复杂化了整个训练流程。

##### 2.2.3 训练模板

为了训练 DeepSeek-R1-Zero，我们首先设计了一个简单的模板，引导基础模型遵循我们指定的指令。如表 1 所示，该模板要求 DeepSeek-R1-Zero 先生成推理过程，然后给出最终答案。我们有意将约束限制在这种结构格式上，避免任何特定内容的偏见，例如强制反思性推理或提倡特定问题解决策略，以确保我们能够准确观察模型在强化学习过程中的自然进展。

##### 2.2.4 DeepSeek-R1-Zero 的性能、自我进化过程和顿悟时刻

**DeepSeek-R1-Zero 的性能**

![[Pasted image 20250315235616.png|500]]
**表 2**：DeepSeek-R1-Zero 与 OpenAI o1 模型在推理相关基准测试上的比较。

<img src="https://arxiv.org/html/2501.12948v1/extracted/6147501/figures/plot_aime_with_maj.png" width="500">
**图 2**：DeepSeek-R1-Zero 在训练过程中的 AIME 准确率。对于每个问题，我们采样 16 个响应并计算整体平均准确率，以确保评估的稳定性。


图 2 展示了 DeepSeek-R1-Zero 在 AIME 2024 基准测试中，整个强化学习训练过程中的性能轨迹。如图所示，随着强化学习训练的推进，DeepSeek-R1-Zero 的性能稳步提升。值得注意的是，AIME 2024 的平均 pass@1 得分显著提高，从初始的 15.6\% 增加到令人印象深刻的 71.0\%，达到了与 OpenAI-o1-0912 可比的性能水平。这一显著提升突显了我们的强化学习算法在优化模型性能方面的有效性。

表 2 提供了 DeepSeek-R1-Zero 与 OpenAI 的 o1-0912 模型在各种推理相关基准测试中的对比分析。结果显示，强化学习使 DeepSeek-R1-Zero 在无需任何监督微调数据的情况下获得了强大的推理能力。这是一个值得注意的成就，强调了模型仅通过强化学习就能有效学习和泛化的能力。此外，通过应用多数投票，DeepSeek-R1-Zero 的性能可以进一步提升。例如，当在 AIME 基准测试中使用多数投票时，DeepSeek-R1-Zero 的性能从 71.0\% 提升到 86.7\%，从而超越了 OpenAI-o1-0912 的表现。

DeepSeek-R1-Zero 能够在有无多数投票的情况下实现如此具有竞争力的性能，突显了其强大的基础能力及其在推理任务中进一步进步的潜力。
<img src="https://arxiv.org/html/2501.12948v1/extracted/6147501/figures/plot_length.png" width="500">
**图 3**：在强化学习过程中，DeepSeek-R1-Zero 在训练集上的平均响应长度。DeepSeek-R1-Zero 自然地通过更多的思考时间来解决推理任务。


**自我进化过程的 DeepSeek-R1-Zero**

DeepSeek-R1-Zero 的自我进化过程是一个展示强化学习如何驱动模型自主提升推理能力的精彩例子。通过直接从基础模型启动强化学习，我们可以在没有监督微调阶段影响的情况下，密切监控模型的进展。这种方法清晰地展示了模型随时间的演变，特别是在处理复杂推理任务方面的能力。

如图 3 所示，DeepSeek-R1-Zero 的思考时间在训练过程中持续改善。这种改善并非外部调整的结果，而是模型内部的内在发展。通过延长测试时间的计算，DeepSeek-R1-Zero 自然获得了解决日益复杂推理任务的能力。这种计算从生成数百到数千个推理标记不等，使模型能够更深入地探索和优化其思维过程。

这种自我进化过程中最引人注目的方面之一是随着测试时间计算的增加，复杂行为的出现。诸如反思（模型重新审视和重新评估其先前步骤）以及探索问题解决的替代方法等行为自发涌现。这些行为并非显式编程，而是模型与强化学习环境交互的结果。这种自发发展显著增强了 DeepSeek-R1-Zero 的推理能力，使其能够更高效、更准确地应对更具挑战性的任务。

**DeepSeek-R1-Zero 的“顿悟时刻”**

在 DeepSeek-R1-Zero 的训练过程中，观察到一个特别有趣的现象，即“顿悟时刻”。这一时刻，如表 3 所示，发生在模型的中间版本。在此阶段，DeepSeek-R1-Zero 通过重新评估其初始方法，学会为问题分配更多的思考时间。这种行为不仅证明了模型推理能力的提升，也展示了强化学习如何带来意想不到的复杂结果。

这个时刻不仅是模型的“顿悟时刻”，也是观察其行为的研究人员的顿悟时刻。它强调了强化学习的力量和美妙之处：我们不需要明确教模型如何解决问题，只需提供适当的激励，它就能自主发展出高级问题解决策略。“顿悟时刻”强有力地提醒我们，强化学习有潜力在人工系统中解锁新的智能水平，为未来更自主和自适应的模型铺平道路。

**DeepSeek-R1-Zero 的缺陷**

尽管 DeepSeek-R1-Zero 展现出强大的推理能力，并自主开发出意想不到且强大的推理行为，但它面临一些问题。例如，DeepSeek-R1-Zero 在可读性差和语言混杂等挑战上存在困扰。为了使推理过程更具可读性并与开放社区共享，我们探索了 DeepSeek-R1，这是一种利用人类友好的冷启动数据进行强化学习的方法。

![[Pasted image 20250316001329.png|500]]
**表 3**：DeepSeek-R1-Zero 中间版本的一个有趣的“顿悟时刻”。模型学会用拟人化的语气重新思考。这对我们来说也是一个顿悟时刻，让我们见证了强化学习的力量和美妙。

#### 2.3 DeepSeek-R1：带有冷启动的强化学习

受到 DeepSeek-R1-Zero 令人振奋的结果启发，我们提出两个自然的问题：1）通过引入少量高质量数据作为冷启动，是否能进一步提高推理性能或加速收敛？2）如何训练一个不仅能产生清晰连贯的思维链（CoT），而且具备强大通用能力的用户友好型模型？

为了解决这些问题，我们设计了一个训练 DeepSeek-R1 的流程。该流程包括以下四个阶段。

##### 2.3.1 冷启动

与 DeepSeek-R1-Zero 不同，为了避免基础模型在强化学习训练初期的不稳定冷启动阶段，对于 DeepSeek-R1，我们构建并收集了一小部分长思维链数据，以微调模型作为初始的强化学习执行者。为了收集这些数据，我们探索了几种方法：使用带有长思维链示例的少样本提示、直接提示模型生成带有反思和验证的详细答案、以可读格式收集 DeepSeek-R1-Zero 输出，并通过人工标注进行后处理和结果优化。

在这项工作中，我们收集了数千条冷启动数据来微调 DeepSeek-V3-Base 作为强化学习的起点。与 DeepSeek-R1-Zero 相比，冷启动数据的优势包括：

* 可读性：DeepSeek-R1-Zero 的一个主要限制是其内容往往不适合阅读。回复可能会混合多种语言或缺少用于突出答案的 Markdown 格式。相反，在为 DeepSeek-R1 创建冷启动数据时，我们设计了一种可读的模式，包括在每个回复末尾的总结，并过滤掉不适合阅读的回复。我们将输出格式定义为 `|special_token|<reasoning_process>|special_token|<summary>`，其中推理过程是查询的思维链，总结用于概括推理结果。
* 潜力：通过精心设计具有人工先验的冷启动数据模式，我们观察到相比 DeepSeek-R1-Zero 更好的性能。我们认为迭代训练是推理模型的更优方法。

##### 2.3.2 面向推理的强化学习

在对冷启动数据微调 DeepSeek-V3-Base 后，我们应用与 DeepSeek-R1-Zero 相同的大规模强化学习训练过程。此阶段重点增强模型的推理能力，特别是在涉及明确解决方案的推理密集型任务中，如编码、数学、科学和逻辑推理。在训练过程中，我们观察到思维链经常出现语言混杂的现象，特别是当强化学习提示涉及多种语言时。为减轻语言混杂问题，我们在强化学习训练中引入了语言一致性奖励，该奖励根据思维链中目标语言词汇的比例计算。尽管消融实验表明这种对齐会导致模型性能略微下降，但这种奖励符合人类偏好，使其更具可读性。最后，我们通过直接相加推理任务的准确性和语言一致性奖励来形成最终奖励。然后，我们对微调模型进行强化学习训练，直到其在推理任务上达到收敛。

##### 2.3.3 拒绝采样和监督微调

当面向推理的强化学习收敛时，我们利用生成的检查点来收集下一轮的 SFT（监督微调）数据。与主要关注推理的初始冷启动数据不同，此阶段结合了其他领域的数据，以增强模型在写作、角色扮演和其他通用任务中的能力。具体而言，我们按以下描述生成数据并微调模型。

**推理数据**

我们通过拒绝采样从上述强化学习训练的检查点中筛选推理提示，并生成推理轨迹。在前一阶段，我们仅包含可以使用基于规则的奖励进行评估的数据。然而，在这一阶段，我们通过引入额外的数据来扩展数据集，其中一些数据使用生成性奖励模型，通过将真实值和模型预测输入 DeepSeek-V3 进行判断。此外，由于模型输出有时混乱且难以阅读，我们过滤掉了混合语言、长段落和代码块的思维链。对于每个提示，我们采样多个回复，仅保留正确的。在总计，我们收集了大约 60 万条与推理相关的训练样本。

**非推理数据**

对于非推理数据，如写作、事实问答、自我认知和翻译，我们采用 DeepSeek-V3 的流程并重用其部分监督微调数据集。对于某些非推理任务，我们在回答问题之前通过提示调用 DeepSeek-V3 生成可能的思维链。然而，对于简单的查询，如“你好”，我们不提供思维链作为回应。最终，我们收集了大约 20 万条与推理无关的训练样本。

我们使用上述约 80 万条样本的精心挑选数据集对 DeepSeek-V3-Base 进行两轮微调。

##### 2.3.4 全场景强化学习

为了进一步使模型与人类偏好对齐，我们实施了一个次级强化学习阶段，旨在提高模型的有用性和无害性，同时优化其推理能力。具体来说，我们结合奖励信号和多样化的提示分布来训练模型。对于推理数据，我们遵循 DeepSeek-R1-Zero 中概述的方法，利用基于规则的奖励来指导数学、代码和逻辑推理领域的学习过程。对于一般数据，我们采用奖励模型来捕捉复杂和微妙场景中的人类偏好。我们基于 DeepSeek-V3 的流程，采用类似的偏好对和训练提示分布。在有用性方面，我们专注于最终总结，确保评估强调响应对用户的实用性和相关性，同时尽量减少对基础推理过程的干扰。在无害性方面，我们评估模型的整个响应，包括推理过程和总结，以识别和减轻生成过程中可能出现的任何潜在风险、偏见或有害内容。最终，奖励信号和多样化数据分布的整合使我们能够训练出一个在推理方面表现出色，同时优先考虑有用性和无害性的模型。

#### 2.4 蒸馏：赋予小模型推理能力

为了让更高效的小模型具备类似 DeepSeek-R1 的推理能力，我们直接使用 80 万条由 DeepSeek-R1 精选的数据对开源模型（如 Qwen 和 Llama）进行微调，具体细节见 2.3.3。我们的研究表明，这种简单的蒸馏方法显著增强了小模型的推理能力。我们使用的基础模型包括 Qwen2.5-Math-1.5B、Qwen2.5-Math-7B、Qwen2.5-14B、Qwen2.5-32B、Llama-3.1-8B 和 Llama-3.3-70B-Instruct。我们选择 Llama-3.3 是因为其推理能力略优于 Llama-3.1。

对于蒸馏模型，我们仅应用 SFT，未包括强化学习阶段，尽管引入强化学习可能会大幅提升模型性能。我们在此的主要目标是展示蒸馏技术的有效性，将强化学习阶段的探索留给更广泛的研究社区。

### 3、实验

**基准测试**

我们在以下基准上评估模型：MMLU、MMLU-Redux、MMLU-Pro、C-Eval、CMMLU、IFEval、FRAMES、GPQA Diamond、SimpleQA、C-SimpleQA、SWE-Bench Verified、Aider、LiveCodeBench（2024-08 至 2025-01）、Codeforces、中国全国高中数学奥林匹克（CNMO 2024）和美国邀请数学考试 2024（AIME 2024）。除了标准基准测试，我们还在开放式生成任务上使用大型语言模型作为评审进行评估。具体来说，我们遵循 AlpacaEval 2.0 和 Arena-Hard 的原始配置，使用 GPT-4-Turbo-1106 作为评审进行成对比较。在这里，我们仅向评估提供最终总结以避免长度偏差。对于蒸馏模型，我们报告在 AIME 2024、MATH-500、GPQA Diamond、Codeforces 和 LiveCodeBench 上的代表性结果。

**评估提示** 

按照 DeepSeek-V3 的设置，MMLU、DROP、GPQA Diamond 和 SimpleQA 等标准基准使用 simple-evals 框架中的提示进行评估。对于 MMLU-Redux，我们在零样本设置中采用 Zero-Eval 提示格式。至于 MMLU-Pro、C-Eval 和 CLUE-WSC，由于原始提示是少样本的，我们稍作修改以适应零样本设置。少样本的 CoT 可能会影响 DeepSeek-R1 的性能。其他数据集遵循其创建者提供的默认提示和原始评估协议。对于代码和数学基准，HumanEval-Mul 数据集涵盖八种主流编程语言（Python、Java、C++、C#、JavaScript、TypeScript、PHP 和 Bash）。LiveCodeBench 的模型性能使用 CoT 格式评估，数据收集时间为 2024 年 8 月至 2025 年 1 月。Codeforces 数据集使用来自 10 场 Div.2 比赛的问题以及专家制作的测试用例进行评估，之后计算预期评分和参赛者百分比。SWE-Bench 的验证结果通过 agentless 框架获得。AIDER 相关基准使用“diff”格式进行测量。DeepSeek-R1 的输出在每个基准中限制为最多 32,768 个标记。

**基线** 

我们与多个强大的基线模型进行全面评估，包括 DeepSeek-V3、Claude-Sonnet-3.5-1022、GPT-4o-0513、OpenAI-o1-mini 和 OpenAI-o1-1217。由于在中国大陆访问 OpenAI-o1-1217 API 存在困难，我们根据官方报告来报告其性能。对于蒸馏模型，我们还比较了开源模型 QwQ-32B-Preview。

**评估设置**

我们将模型的最大生成长度设置为 32,768 个标记。发现使用贪婪解码评估长输出推理模型会导致更高的重复率，并在不同的检查点上表现出显著的差异。因此，我们默认采用 pass@$k$ 评估，并使用非零温度报告 pass@1。具体来说，我们使用 0.6 的采样温度和 0.95 的 top-$p$ 值，为每个问题生成 $k$ 个响应（通常在 4 到 64 之间，取决于测试集大小）。然后计算 pass@1：$$
\text{pass@1} = \frac{1}{k} \sum_{i=1}^{k} p_i,$$其中 $p_i$ 表示第 $i$ 个响应的正确性。此方法提供更可靠的性能估计。对于 AIME 2024，我们还使用 64 个样本报告共识（多数投票）结果，记为 $\text{cons}@64$。


#### 3.1 DeepSeek-R1 评估

**表 4**：DeepSeek-R1 与其他代表性模型的比较。

在教育导向的知识基准测试中，如 MMLU、MMLU-Pro 和 GPQA Diamond，DeepSeek-R1 展现出优于 DeepSeek-V3 的性能。这一提升主要归功于在 STEM 相关问题上的准确性增强，通过大规模强化学习实现了显著的进步。此外，DeepSeek-R1 在 FRAMES 任务中表现出色，这是一项长上下文依赖的问答任务，展示了其强大的文档分析能力。这突显了推理模型在 AI 驱动的搜索和数据分析任务中的潜力。在事实性基准测试 SimpleQA 上，DeepSeek-R1 超越了 DeepSeek-V3，证明了其处理基于事实查询的能力。在这个基准上，OpenAI-o1 同样超过了 GPT-4o。然而，在中文 SimpleQA 基准上，DeepSeek-R1 的表现不如 DeepSeek-V3，主要原因是其在安全强化学习后倾向于拒绝回答某些查询。在没有安全强化学习的情况下，DeepSeek-R1 可以达到超过 70% 的准确率。

DeepSeek-R1 在 IF-Eval 基准测试中也表现出色，该测试旨在评估模型遵循格式指令的能力。这些改进与在 SFT 和 RL 训练的最后阶段加入指令跟随数据有关。此外，在 AlpacaEval 2.0 和 ArenaHard 上的出色表现表明 DeepSeek-R1 在写作任务和开放域问答中的优势。其显著超越 DeepSeek-V3 的表现强调了大规模 RL 的泛化优势，这不仅提升了推理能力，还改善了各个领域的表现。此外，DeepSeek-R1 生成的摘要长度简洁，ArenaHard 平均为 689 个标记，AlpacaEval 2.0 为 2,218 个字符。这表明 DeepSeek-R1 在基于 GPT 的评估中避免了长度偏差，进一步巩固了其在多任务中的稳健性。

在数学任务中，DeepSeek-R1 的表现与 OpenAI-o1-1217 相当，远超其他模型。在编码算法任务（如 LiveCodeBench 和 Codeforces）中，专注于推理的模型在这些基准上表现出色。在面向工程的编码任务中，OpenAI-o1-1217 在 Aider 上优于 DeepSeek-R1，但在 SWE Verified 上表现相当。我们相信，随着相关 RL 训练数据量的增加，DeepSeek-R1 的工程性能将在下一个版本中得到提升。

#### 3.2 蒸馏模型评估

**表 5**：DeepSeek-R1 蒸馏模型与其他可比模型在推理相关基准上的比较。

如表 5 所示，仅通过蒸馏 DeepSeek-R1 的输出，就能使高效的 DeepSeek-R1-7B（即 DeepSeek-R1-Distill-Qwen-7B，以下类似缩写）在各方面超越非推理模型，如 GPT-4o-0513。DeepSeek-R1-14B 在所有评估指标上超过了 QwQ-32B-Preview，而 DeepSeek-R1-32B 和 DeepSeek-R1-70B 在大多数基准测试中显著超越了 o1-mini。这些结果展示了蒸馏的强大潜力。此外，我们发现对这些蒸馏模型应用 RL 可以获得显著的进一步提升。我们认为这值得进一步探索，因此这里只展示了简单的 SFT 蒸馏模型的结果。

### 4、讨论


\subsection{Distillation v.s. Reinforcement Learning}
\input{tables/distill_vs_rl}

In Section \ref{sec:distilled_model_evaluation}, we can see that by distilling DeepSeek-R1, the small model can achieve impressive results. However, there is still one question left: can the model achieve comparable performance through the large-scale RL training discussed in the paper without distillation?


To answer this question, we conduct large-scale RL training on Qwen-32B-Base using math, code, and STEM data, training for over 10K steps, resulting in DeepSeek-R1-Zero-Qwen-32B. The experimental results, shown in Table \ref{tab:distill_vs_rl}, demonstrate that the 32B base model, after large-scale RL training, achieves performance on par with QwQ-32B-Preview. However, DeepSeek-R1-Distill-Qwen-32B, which is distilled from DeepSeek-R1,  performs significantly better than DeepSeek-R1-Zero-Qwen-32B across all benchmarks.

Therefore, we can draw two conclusions: First, distilling more powerful models into smaller ones yields excellent results, whereas smaller models relying on the large-scale RL mentioned in this paper require enormous computational power and may not even achieve the performance of distillation. Second, while distillation strategies are both economical and effective, advancing beyond the boundaries of intelligence may still require more powerful base models and larger-scale reinforcement learning.

\subsection{Unsuccessful Attempts}
In the early stages of developing \dsri{}, we also encountered failures and setbacks along the way. We share our failure experiences here to provide insights, but this does not imply that these approaches are incapable of developing effective reasoning models.

\paragraph{Process Reward Model (PRM)}
PRM is a reasonable method to guide the model toward better approaches for solving reasoning tasks~\citep{uesato2022solving, lightman2023let,mathshepherd}. However, in practice, PRM has three main limitations that may hinder its ultimate success. First, it is challenging to explicitly define a fine-grain step in general reasoning. 
Second, determining whether the current intermediate step is correct is a challenging task. Automated annotation using models may not yield satisfactory results, while manual annotation is not conducive to scaling up.
Third, once a model-based PRM is introduced, it inevitably leads to reward hacking~\citep{gao2022scalinglawsrewardmodel},  and retraining the reward model needs additional training resources and it complicates the whole training pipeline. In conclusion, while PRM demonstrates a good ability to rerank the top-N responses generated by the model or assist in guided search~\citep{snell2024scalingllmtesttimecompute}, its advantages are limited compared to the additional computational overhead it introduces during the large-scale reinforcement learning process in our experiments.

\paragraph{Monte Carlo Tree Search (MCTS)}
Inspired by AlphaGo~\citep{alphago} and AlphaZero~\citep{alphazero}, we explored using Monte Carlo Tree Search (MCTS) to enhance test-time compute scalability. This approach involves breaking answers into smaller parts to allow the model to explore the solution space systematically. To facilitate this, we prompt the model to generate multiple tags that correspond to specific reasoning steps necessary for the search. For training, we first use collected prompts to find answers via MCTS guided by a pre-trained value model. Subsequently, we use the resulting question-answer pairs to train both the actor model and the value model, iteratively refining the process.

However, this approach encounters several challenges when scaling up the training. First, unlike chess, where the search space is relatively well-defined, token generation presents an exponentially larger search space. To address this, we set a maximum extension limit for each node, but this can lead to the model getting stuck in local optima. Second, the value model directly influences the quality of generation since it guides each step of the search process. Training a fine-grained value model is inherently difficult, which makes it challenging for the model to iteratively improve. While AlphaGo's core success relied on training a value model to progressively enhance its performance, this principle proves difficult to replicate in our setup due to the complexities of token generation.

In conclusion, while MCTS can improve performance during inference when paired with a pre-trained value model, iteratively boosting model performance through self-search remains a significant challenge.




\section{Conclusion, Limitations, and Future Work}

In this work, we share our journey in enhancing model reasoning abilities through reinforcement learning. \dsro{} represents a pure RL approach without relying on cold-start data, achieving strong performance across various tasks. \dsri{} is more powerful, leveraging cold-start data alongside iterative RL fine-tuning. Ultimately, \dsri{} achieves performance comparable to OpenAI-o1-1217 on a range of tasks.

We further explore distillation the reasoning capability to small dense models. We use \dsri{} as the teacher model to generate 800K training samples, and fine-tune several small dense models. The results are promising: DeepSeek-R1-Distill-Qwen-1.5B outperforms GPT-4o and Claude-3.5-Sonnet on math benchmarks with 28.9\% on AIME and 83.9\% on MATH. Other dense models also achieve impressive results, significantly outperforming other instruction-tuned models based on the same underlying checkpoints.

In the future, we plan to invest in research across the following directions for \dsri{}.
\begin{itemize}[topsep=0pt]
    \item \textbf{General Capability:}
  Currently, the capabilities of \dsri{} fall short of DeepSeek-V3 in tasks such as function calling, multi-turn, complex role-playing, and JSON output. Moving forward, we plan to explore how long CoT can be leveraged to enhance tasks in these fields.
    \item \textbf{Language Mixing:}
\dsri{} is currently optimized for Chinese and English, which may result in language mixing issues when handling queries in other languages. For instance, \dsri{} might use English for reasoning and responses, even if the query is in a language other than English or Chinese. We aim to address this limitation in future updates.
 \item \textbf{Prompting Engineering:} When evaluating \dsri{}, we observe that it is sensitive to prompts. Few-shot prompting consistently degrades its performance. Therefore, we recommend users directly describe the problem and specify the output format using a zero-shot setting for optimal results.
\item  \textbf{Software Engineering Tasks:}
Due to the long evaluation times, which impact the efficiency of the RL process, large-scale RL has not been applied extensively in software engineering tasks. As a result, DeepSeek-R1 has not demonstrated a huge improvement over DeepSeek-V3 on software engineering benchmarks. Future versions will address this by implementing rejection sampling on software engineering data or incorporating asynchronous evaluations during the RL process to improve efficiency.
    
\end{itemize}

\bibliography{main}

\newpage
\appendix

\section*{Appendix}

\section{Contributions and Acknowledgments}

\definecolor{damaiblue}{RGB}{0, 0, 100}
\definecolor{damaigreen}{RGB}{0, 100, 0}
\definecolor{damaired}{RGB}{100, 0, 0}

\begin{multicols}{2} %
\noindent
\textbf{\color{damaired} Core Contributors} \\
\color{damaired} Daya Guo \\
\color{damaired} Dejian Yang \\
\color{damaired} Haowei Zhang \\
\color{damaired} Junxiao Song \\
\color{damaired} Ruoyu Zhang \\
\color{damaired} Runxin Xu \\
\color{damaired} Qihao Zhu \\
\color{damaired} Shirong Ma \\
\color{damaired} Peiyi Wang \\
\color{damaired} Xiao Bi \\
\color{damaired} Xiaokang Zhang \\
\color{damaired} Xingkai Yu \\
\color{damaired} Yu Wu \\
\color{damaired} Z.F. Wu \\
\color{damaired} Zhibin Gou \\
\color{damaired} Zhihong Shao \\
\color{damaired} Zhuoshu Li \\
\color{damaired} Ziyi Gao \\

\noindent
\textbf{\color{damaiblue} Contributors} \\
\color{damaiblue} 
\color{damaiblue} Aixin Liu \\
\color{damaiblue} Bing Xue \\
\color{damaiblue} Bingxuan Wang \\
\color{damaiblue} Bochao Wu \\
\color{damaiblue} Bei Feng \\
\color{damaiblue} Chengda Lu \\
\color{damaiblue} Chenggang Zhao \\
\color{damaiblue} Chengqi Deng \\
\color{damaiblue} Chong Ruan \\
\color{damaiblue} Damai Dai \\
\color{damaiblue} Deli Chen \\
\color{damaiblue} Dongjie Ji \\
\color{damaiblue} Erhang Li \\
\color{damaiblue} Fangyun Lin \\
\color{damaiblue} Fucong Dai \\
\color{damaiblue} Fuli Luo* \\
\color{damaiblue} Guangbo Hao \\
\color{damaiblue} Guanting Chen \\
\color{damaiblue} Guowei Li \\
\color{damaiblue} H. Zhang \\
\color{damaiblue} Hanwei Xu \\
\color{damaiblue} Honghui Ding \\
\color{damaiblue} Huazuo Gao \\
\color{damaiblue} Hui Qu \\
\color{damaiblue} Hui Li \\
\color{damaiblue} Jianzhong Guo \\
\color{damaiblue} Jiashi Li \\
\color{damaiblue} Jingchang Chen \\
\color{damaiblue} Jingyang Yuan \\
\color{damaiblue} Jinhao Tu \\
\color{damaiblue} Junjie Qiu \\
\color{damaiblue} Junlong Li \\
\color{damaiblue} J.L. Cai \\
\color{damaiblue} Jiaqi Ni \\
\color{damaiblue} Jian Liang \\
\color{damaiblue} Jin Chen \\
\color{damaiblue} Kai Dong \\
\color{damaiblue} Kai Hu* \\
\color{damaiblue} Kaichao You \\
\color{damaiblue} Kaige Gao \\
\color{damaiblue} Kang Guan \\
\color{damaiblue} Kexin Huang \\
\color{damaiblue} Kuai Yu \\
\color{damaiblue} Lean Wang \\
\color{damaiblue} Lecong Zhang \\
\color{damaiblue} Liang Zhao \\
\color{damaiblue} Litong Wang \\
\color{damaiblue} Liyue Zhang \\
\color{damaiblue} Lei Xu \\
\color{damaiblue} Leyi Xia \\
\color{damaiblue} Mingchuan Zhang \\
\color{damaiblue} Minghua Zhang \\
\color{damaiblue} Minghui Tang \\
\color{damaiblue} Mingxu Zhou \\
\color{damaiblue} Meng Li \\
\color{damaiblue} Miaojun Wang \\
\color{damaiblue} Mingming Li \\
\color{damaiblue} Ning Tian \\
\color{damaiblue} Panpan Huang \\
\color{damaiblue} Peng Zhang \\
\color{damaiblue} Qiancheng Wang \\
\color{damaiblue} Qinyu Chen \\
\color{damaiblue} Qiushi Du \\
\color{damaiblue} Ruiqi Ge* \\
\color{damaiblue} Ruisong Zhang \\
\color{damaiblue} Ruizhe Pan \\
\color{damaiblue} Runji Wang \\
\color{damaiblue} R.J. Chen \\
\color{damaiblue} R.L. Jin \\
\color{damaiblue} Ruyi Chen \\
\color{damaiblue} Shanghao Lu \\
\color{damaiblue} Shangyan Zhou \\
\color{damaiblue} Shanhuang Chen \\
\color{damaiblue} Shengfeng Ye \\
\color{damaiblue} Shiyu Wang \\
\color{damaiblue} Shuiping Yu \\
\color{damaiblue} Shunfeng Zhou \\
\color{damaiblue} Shuting Pan \\
\color{damaiblue} S.S. Li \\
\color{damaiblue} Shuang Zhou \\
\color{damaiblue} Shaoqing Wu \\
\color{damaiblue} Shengfeng Ye \\
\color{damaiblue} Tao Yun \\
\color{damaiblue} Tian Pei \\
\color{damaiblue} Tianyu Sun \\
\color{damaiblue} T. Wang \\
\color{damaiblue} Wangding Zeng \\
\color{damaiblue} Wen Liu \\
\color{damaiblue} Wenfeng Liang \\
\color{damaiblue} Wenjun Gao \\
\color{damaiblue} Wenqin Yu* \\
\color{damaiblue} Wentao Zhang \\
\color{damaiblue} W.L. Xiao \\
\color{damaiblue} Wei An \\
\color{damaiblue} Xiaodong Liu \\
\color{damaiblue} Xiaohan Wang \\
\color{damaiblue} Xiaokang Chen \\
\color{damaiblue} Xiaotao Nie \\
\color{damaiblue} Xin Cheng \\
\color{damaiblue} Xin Liu \\
\color{damaiblue} Xin Xie \\
\color{damaiblue} Xingchao Liu \\
\color{damaiblue} Xinyu Yang \\
\color{damaiblue} Xinyuan Li \\
\color{damaiblue} Xuecheng Su \\
\color{damaiblue} Xuheng Lin \\
\color{damaiblue} X.Q. Li \\
\color{damaiblue} Xiangyue Jin \\
\color{damaiblue} Xiaojin Shen \\
\color{damaiblue} Xiaosha Chen \\
\color{damaiblue} Xiaowen Sun \\
\color{damaiblue} Xiaoxiang Wang \\
\color{damaiblue} Xinnan Song \\
\color{damaiblue} Xinyi Zhou \\
\color{damaiblue} Xianzu Wang \\
\color{damaiblue} Xinxia Shan \\
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
\color{damaiblue} Y.X. Zhu \\
\color{damaiblue} Yanping Huang \\
\color{damaiblue} Yaohui Li \\
\color{damaiblue} Yi Zheng \\
\color{damaiblue} Yuchen Zhu \\
\color{damaiblue} Yunxian Ma \\
\color{damaiblue} Ying Tang \\
\color{damaiblue} Yukun Zha \\
\color{damaiblue} Yuting Yan \\
\color{damaiblue} Z.Z. Ren \\
\color{damaiblue} Zehui Ren \\
\color{damaiblue} Zhangli Sha \\
\color{damaiblue} Zhe Fu \\
\color{damaiblue} Zhean Xu \\
\color{damaiblue} Zhenda Xie \\
\color{damaiblue} Zhengyan Zhang \\
\color{damaiblue} Zhewen Hao \\
\color{damaiblue} Zhicheng Ma \\
\color{damaiblue} Zhigang Yan \\
\color{damaiblue} Zhiyu Wu \\
\color{damaiblue} Zihui Gu \\
\color{damaiblue} Zijia Zhu \\
\color{damaiblue} Zijun Liu* \\
\color{damaiblue} Zilin Li \\
\color{damaiblue} Ziwei Xie \\
\color{damaiblue} Ziyang Song \\
\color{damaiblue} Zizheng Pan \\
\color{damaiblue} Zhen Huang \\
\color{damaiblue} Zhipeng Xu \\
\color{damaiblue} Zhongyu Zhang \\
\color{damaiblue} Zhen Zhang \\

\end{multicols} %

Within each role, authors are listed alphabetically by the first name. 
Names marked with * denote individuals who have departed from our team. 


\setcounter{figure}{0}
\makeatletter 
\renewcommand{\thefigure}{A\@arabic\c@figure}
\makeatother

\setcounter{table}{0}
\makeatletter 
\renewcommand{\thetable}{A\@arabic\c@table}
\makeatother

\end{document} 
