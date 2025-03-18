
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

#### 4.1 蒸馏 vs RL

![[Pasted image 20250318215409.png|600]]
**表 6**：蒸馏模型和强化学习模型在推理相关基准测试上的比较。


在第 3.2 节中，我们看到，通过蒸馏 DeepSeek-R1，小模型可以取得令人印象深刻的结果。然而，仍然有一个问题：不通过蒸馏，仅通过论文中讨论的大规模强化学习训练，模型能否达到相当的性能？

为了解答这个问题，我们对 Qwen-32B-Base 进行了大规模强化学习训练，使用数学、代码和 STEM 数据，训练超过 10K 步，得到 DeepSeek-R1-Zero-Qwen-32B。实验结果如表 6 所示，经过大规模强化学习训练的 32B 基础模型，其性能与 QwQ-32B-Preview 相当。然而，从 DeepSeek-R1 蒸馏得到的 DeepSeek-R1-Distill-Qwen-32B 在所有基准测试中表现明显优于 DeepSeek-R1-Zero-Qwen-32B。

因此，我们可以得出两个结论：首先，将更强大的模型蒸馏为较小的模型能够取得优异的结果，而依赖于本文提到的大规模强化学习的小模型需要巨大的计算能力，甚至可能无法达到蒸馏的性能。其次，尽管蒸馏策略既经济又有效，但要超越智能的界限，可能仍然需要更强大的基础模型和更大规模的强化学习。

#### 4.2 不成功的尝试

在开发 DeepSeek-R1 的早期阶段，我们也遇到了失败和挫折。我们在此分享这些失败经历以提供见解，但这并不意味着这些方法无法开发出有效的推理模型。

**过程奖励模型 (PRM)**

PRM 是一种合理的方法，可以引导模型更好地解决推理任务。然而，在实践中，PRM 有三个主要限制可能阻碍其成功。首先，很难在一般推理中明确定义细粒度步骤。其次，判断当前中间步骤是否正确是一项具有挑战性的任务。使用模型进行自动标注可能无法获得满意的结果，而人工标注不利于规模化。第三，一旦引入基于模型的 PRM，必然会导致奖励欺骗，并且重新训练奖励模型需要额外的训练资源，复杂化了整个训练流程。总之，尽管 PRM 在重新排序模型生成的前 N 个响应或辅助引导搜索方面表现出色，但与其在大规模强化学习过程中的额外计算开销相比，其优势有限。

**蒙特卡罗树搜索 (MCTS)**

受 AlphaGo 和 AlphaZero 的启发，我们探索了使用蒙特卡罗树搜索 (MCTS) 来增强测试时计算的可扩展性。这种方法涉及将答案分解为更小的部分，以便模型系统地探索解空间。为此，我们提示模型生成多个标签，这些标签对应于搜索所需的特定推理步骤。在训练中，我们首先使用收集的提示，通过预训练的价值模型指导的 MCTS 找到答案。随后，我们使用生成的问题-答案对来训练演员模型和价值模型，迭代地优化这一过程。

然而，这种方法在扩展训练时遇到了几个挑战。首先，与搜索空间相对明确的国际象棋不同，生成 token 的搜索空间呈指数级增长。为了解决这个问题，我们为每个节点设置了最大扩展限制，但这可能导致模型陷入局部最优。其次，价值模型直接影响生成的质量，因为它指导搜索过程的每一步。训练细粒度的价值模型本质上很困难，这使得模型难以迭代改进。虽然 AlphaGo 的核心成功依赖于训练一个价值模型来逐步提升性能，但由于 token 生成的复杂性，这一原则在我们的设置中难以复制。

总之，虽然 MCTS 可以在与预训练价值模型配对时改善推理性能，但通过自我搜索迭代提升模型性能仍然是一个重大挑战。


### 5、结论、局限性与未来工作

在本研究中，我们分享了通过强化学习提升模型推理能力的历程。DeepSeek-R1-Zero 代表了一种不依赖冷启动数据的纯 RL 方法，在多项任务中表现出色。DeepSeek-R1 更为强大，结合了冷启动数据与迭代 RL 微调。最终，DeepSeek-R1 在多项任务中的表现与 OpenAI-o1-1217 相当。

我们进一步探索了将推理能力蒸馏到小型密集模型中的可能性。我们使用 DeepSeek-R1 作为教师模型生成了 80 万个训练样本，并微调了几个小型密集模型。结果令人振奋：DeepSeek-R1-Distill-Qwen-1.5B 在数学基准测试中超过了 GPT-4o 和 Claude-3.5-Sonnet，在 AIME 上达到 28.9\%，在 MATH 上达到 83.9\%。其他密集模型也取得了令人印象深刻的结果，显著优于基于相同底层检查点的其他指令调优模型。

未来，我们计划在以下几个方向上对 DeepSeek-R1 进行研究：

* 通用能力：目前，DeepSeek-R1 在函数调用、多轮对话、复杂角色扮演和 JSON 输出等任务上的能力不如 DeepSeek-V3。未来，我们计划探索如何利用长链推理（CoT）来增强这些领域的任务。
* 语言混合：DeepSeek-R1 目前针对中英文进行了优化，这可能导致在处理其他语言的查询时出现语言混合问题。例如，即使查询使用的是非中英文，DeepSeek-R1 可能仍使用英文进行推理和响应。我们计划在未来的更新中解决这一局限性。
* 提示工程：在评估 DeepSeek-R1 时，我们观察到它对提示非常敏感。少样本提示会持续降低其性能。因此，我们建议用户在零样本设置下直接描述问题并指定输出格式以获得最佳结果。
* 软件工程任务：由于长时间的评估影响了 RL 过程的效率，大规模 RL 尚未广泛应用于软件工程任务。因此，DeepSeek-R1 在软件工程基准测试中并未显示出相对于 DeepSeek-V3 的巨大改进。未来版本将通过在软件工程数据上实施拒绝采样或在 RL 过程中加入异步评估来提高效率。


----------

为了让大家比较容易的理解 R1 背后的原理，我想分四个阶段：

1. RL 的历史，DeepMind 的 AlphaGo 系列，RL 的一些基本想法；OpenAI GPT 系列的发展，训练范式；Scaling Law；大语言模型的涌现（Emergent Ability）；COT；
2. 摘要和引言部分，感性的认识 R1；方法和结果部分；
3. GRPO，DeepSeekMath
4. Kimi Paper，同期用 RL 来增强 LLM 的工作，它给了很多的细节


Figure 1 里面比较了五个模型:
* DeepSeek-R1 vs OpenAI-o1， 这两个模型都比较的大
* DeepSeek-R1-32B  vs OpenAI-o1-mini，蒸馏过的小模型 
* DeepSeek-V3 是 base model；

数据集方面： 
* 数学相关的：AIME2024 和 Math500。AIME2024 是美国数学协会出的，美国高中的奥赛题，大概是在高中往上一点点的水平；Math500 是 OpenAI 准备的一套数据集，主要是在研究生水平的一些数学题；
* 关于编程的数据集： Codeforces 和 SWE Bench Verify。Codeforces 是一个编程竞赛网站，有人会在上面出题，大家通过编程的方式来解决这些问题，然后有一个排名，类似于 Leet Code，Codeforces 是直接和人类进行比较的，96.3 意味着能击败 96.3% 的人类；后者来自于 Github Issue 的一个数据集，主要是考察模型debug 的能力，主要测试方法就是给模型一段代码，里面可能有 bug，要求模型找出这个 bug 并进行修正；
* MMLU主要是考察的模型知识，涵盖的知识面非常的广，包含了各种的学科，如人文、数学、法律、物理、医学等等，代表的是人类的平均水平
* GPQA Diamond 是另外一个极端，这个数据集的题目不多，大概只有 448 道题，但非常非常的难，需要博士级的人才能回答，即便如此，准确率也大概只有 65% 左右，它代表的是人类的最高水平；

DeepSeek-R1 主推模型的 Reasoning 能力，也就主要体现在数学和编程上面，...，整体上可以看到 DeepSeek-R1 和 OpenAI-o1 基本是不相上下的；不论是 DeepSeek-R1 还是 OpenAI-o1 在 Codeforces 上和人类程序员直接 PK，都已经超过了约 96% 的程序员...

---


我们来看一下Abstract的部分 这里我还是想重申一下 请大家用top-down的方式跟着我的思路走 在看Abstract和Introduction的时候 请首先建立一个宏观的印象 也就是whole picture 不要纠结于一些你不懂的细节 因为我在后面 介绍更多的background的时候 会帮你把细节给填充进去 最后整个系列讲完之后 你就能明白DeepSeek-R1 到底是一个什么东西 Abstract第一句话就是说 他们推出了他们自己 第1st generation的 reasoning model 也就是DeepSeek系列的第一个reasoning model 我猜他们的R是来自于这个reasoning的R 这里他提到了两个model 一个叫DeepSeek-R1-Zero 另外一个叫DeepSeek-R1 DeepSeek-R1-Zero 就是DeepSeek他们做的一个尝试 可以理解为一个proof of concept的尝试 那么在post training里面

[12:37](https://www.youtube.com/watch?v=tRuN8xYdETs&t=757s)

只用了reinforcement learning 用这种方式寸出来的DeepSeek-R1-Zero 模型展现出来了一些很有意思的特性 比如说模型出现的aha moment DeepSeek发现用reinforcement learning 而不要用supervised finetuning 是可以增强模型的推理能力的 但是他们也发现了一些问题 这里后面会提到 所以在DeepSeek-R1-Zero 这个想法的基础之上 他们又尝试了另外一套 更加复杂的training pipeline 非常high level的总结 就是他们利用少量的标注数据训练模型 然后再用这个模型 产生一些新的训练数据 然后把这些新的训练数据 和旧的训练数据合在一起 再去训练基础模型 他这个思想有一点像机器学习 或者是统计里面常用的 bootstrapping的方法 bootstrapping你可以理解为 你拉着自己的鞋子上面的鞋带 可以把自己给提起来 当然这从物理上是做不到的 这个哲学思想背后的意思就是说

[13:42](https://www.youtube.com/watch?v=tRuN8xYdETs&t=822s)

你可以用已有的模型产生一些数据 然后用这个数据去增强你已有的模型 这就是DeepSeek-R1 背后的一些哲学思想 然后这个abstract 下面就开始介绍这两个模型了 DeepSeek-R1-Zero和DeepSeek-R1 DeepSeek-R1-Zero 就是一个用reinforcement learning 但是没有用supervised的finetuning 也就是SFT的方法训练出来的一个大语言模型 这里稍微提一下 OpenAI提出的大语言模型的一个训练范式 基本上一个大语言模型训练 需要经过pre train和post train pre train就是用大量的text 然后用next token prediction的方式 训练大语言模型 post train里面一般第一步会进行一个supervised的finetuning 也就是SFT 也就是说要人工的针对一些问题写出答案 那当年OpenAI是在非洲的一些英语国家 招揽了很多的便宜的劳工 然后给他们问题 让他们写出这些问题的答案

[14:45](https://www.youtube.com/watch?v=tRuN8xYdETs&t=885s)

或者是让他们来判断几个答案中的答案 哪个他们更喜欢 通过这种方式 他们标注了大量的对话问答数据 然后用这种对话问答数据去进行SFT 然后OpenAI在经过SFT之后 还提出了用reinforcement learning 去做alignment 这步主要是让模型能够迎合人类的偏好 比如说有一些很黄很暴力的问题 或者是非法的问题 模型就不应该回答 这一步就是在训练模型的这种偏好性 DeepSeek这篇paper 所要挑战的就是这一个部组 也就是说DeepSeek 认为supervised的finetuning 对于增强模型的reasoning能力 并不是必须的 如果这个能实现的话 将会是非常大的一个贡献 因为对于reasoning的数据的收集是非常困难的 DeepSeek的意思就是说 我不需要这一步 直接用rl就可以增强模型的reasoning ability 接下来做的就是说 使用rl之后 DeepSeek-R1-Zero出现了很多非常强大并且有意思的reasoning behaviors

[15:52](https://www.youtube.com/watch?v=tRuN8xYdETs&t=952s)

但是作者也提到这个模型它还是有一些的问题 模型给出了答案 人类很难看懂 Language mixing就是说 DeepSeek在回答问题的过程中进行思考的时候 会中英文混杂着思考 这也是一个非常有意思的现象 所以作者就提出为了改善这些问题 他们就提出了一个DeepSeek-R1模型 r1模型主要是做的两个改进 第一个就是把之前直接用rl进行暴力训练的方式 改成了一个multi-stage training的方式 同时他们准备了一些coldstart data 这些coldstart data是有标注的 先把模型稍微训练一下 然后再使用rl 这里面具体怎么做的 在讲到后面方法的时候 会具体给大家解释 作者就总结了一下 说DeepSeek-R1模型和 OpenAI的r1模型 它的performance是comparable的 也就是基本上相似的 最后作者还提了一下 我觉得这是 DeepSeek做出的一个非常重大的贡献 就是他们open source了 DeepSeek-R1-Zero和DeepSeek-R1

[16:56](https://www.youtube.com/watch?v=tRuN8xYdETs&t=1016s)

另外他们还做了一个事情 因为DeepSeek-R1是一个非常大的模型 它是基于DeepSeek-v3 也就是差不多有600个billion的moe模型 这在使用中会非常的笨重 开销也会非常的大 所以他们就用这个非常大的DeepSeek-R1模型 蒸馏出来了6个小的模型 所谓的蒸馏就是你可以理解为用DeepSeek-R1作为老师 教出了6个更小的模型 这6个更小的模型是基于Qwen和Llama的 注意的是Qwen和拉玛它都是dense模型 它们不是moe模型 moe模型和dense模型的区别 我在DeepSeek-V3的视频里面已经解释过了 我们再来看一下introduction部分 作者这里一上来就抛出AGI这个概念 AGI就是通用人工智能 它的重点在通用这个上面 我想这可能是目前这个方向的科研工作者最重大的一个理想 就是想实现通用人工智能 然后作者在这里提到post training是一个很重要的步骤 post training的作用就是提高模型的reasoning ability

[18:03](https://www.youtube.com/watch?v=tRuN8xYdETs&t=1083s)

或者是让模型和人类社会的价值观进行对齐等等等等 然后OpenAI首先推出了inference time scaling 通过这种方式能够增强模型的reasoning ability 但是很不幸的是OpenAI的O1模型并没有公开里面的细节 所以很多人至少在research community里面 大家其实都不知道OpenAI是怎么实现的 所以大家用了各种各样的方法 比如说process-based reward model reinforcement learning等等等等这些方法 想去复现OpenAI的O1模型 但是最后没有一个模型能够达到OpenAI O1的这样的一个性能 DeepSeek作者这种写作手法就是 为他们后面DeepSeek-R1模型先打一个铺垫 就是OpenAI O1很牛 领域类大家都想复现 但是都没有能成功 但是我们成功了 接下来在这一段里面 作者主要讲是DeepSeek-R1-Zero 也就是一个纯用reinforcement learning train出来的一个模型

[19:05](https://www.youtube.com/watch?v=tRuN8xYdETs&t=1145s)

他这里强调就是他们在利用reinforcement learning 进行post training的时候 是没有用任何的supervised data 然后他们是基于DeepSeek-v3 base 这个模型开始具体用的方法就是GRPO 这个地方的引用的这篇文章就是DeepSeekMath 我前面也提到我会把GRPO单独领出来 在比较靠后的位置在跟大家讲 接着作者就提到 经过这样的方式训练 DeepSeek-R1-Zero 其实展现了一些非常强大的能力 就在这里用了一个词叫super performance 也就是说他们的这个想法是可行的 however 然后就开始介绍DeepSeek-R1了 为什么要用DeepSeek-R1呢 其中简单的说就是DeepSeek-R1-Zero 这个模型啊 它其实存在一些问题 主要的两个问题就是它的可读性很差 第二个就是language mixing 可读性很差就是模型的这个输出 人类不可读的 只有模型自己明白 这个也是由于reinforcement learning 它的奖励或者是训练的信号 是在最终的结果上面

[20:08](https://www.youtube.com/watch?v=tRuN8xYdETs&t=1208s)

所以模型不管过程 只要结果对就可以了 就造成了这种问题 第二个就是language mixing 就是模型在思考或者回答的时候 经常会出现中英文混杂的情况 所以作者为了解决这个问题 他们采用了两个方式 第一个叫cold start data 第二个叫multi stage training 这里我可以给大家打一个比方 就是coldstart这一类的问题 其实在recommendation里面是非常常见的 比如说对于一些像youtube这样的平台 如果你是一个新注册的用户 youtube没有你的观看记录 所以youtube不知道你的观看偏好 他不能很好的给你推荐你喜欢的影片 这就是一个col'd start的问题 所以你必须看一些影片 经过一段时间之后 youtube才能根据你观看的记录 计算出你的偏好 然后再给你推荐相对应的影片 那DeepSeek这里遇到的也是同样的一个问题 就是在训练模型 reasoning能力的时候 也出现了一个coldstart的问题 所以他们的解法也就用到了类似的思想

[21:13](https://www.youtube.com/watch?v=tRuN8xYdETs&t=1273s)

他们先收集了一些少量的coldstart data 那这些data是有标注的 然后他们用收集的这些少量的coldstart data 去finetune DeepSeek-v3 base model 经过这样的finetune之后 然后他们接着再用rl的方法去训练 也就是说在DeepSeek-R1-Zero之前的训练方法中间插入一个finetuning的方法 得到了这个模型之后 他们就再采用bootstrapping的方法 也就是用这个训练的还不错的具有一定reasoning ability的模型 再产生一些新的sft的数据 再把这个新产生的synthetic的数据和之前的数据混到一起 然后再重新训练这个base model 训练好的模型又经过了一轮reinforcement learning的训练 最终得到了DeepSeek-R1 我知道估计大家在听这个multi stage training的这个部分已经晕掉了 没有关系大家只要头脑中有一个印象 就是DeepSeek用了一个multi stage的training 在这个training里面 他们自己收集了一些有label的数据

[22:18](https://www.youtube.com/watch?v=tRuN8xYdETs&t=1338s)

也就是说coldstart data数据 然后又利用了bootstrapping的思想去训练模型 大家抓住这个基本的思想就可以了 到后面讲到这一部分的时候 我会再具体的解释 这里作者又提出了一个比较有意思的现象 就是他们把DeepSeek-R1给蒸馏成了一个比较小的模型 他们发现如果直接从这个小的模型 也就是QWen2.5 32b reinforcement learning的方法来train的话 结果没有从大的模型 也就是DeepSeek-R1蒸馏出来的结果好 那这里我可以给大家打一个比方 假如有一个学生他想学习一项技能 那有两种方式去学习 第一种就是这个学生自己去学 第二种方式就是他有一个老师 这个老师先学 因为老师有一定的基础 或者是有一定的背景 并且老师的理解能力也比学生要强 所以老师在学习这项技能的时候 会学的比较快 而且老师会总结出自己的一些经验方法 然后老师再把他学到的知识 还有思路交给这个学生

[23:21](https://www.youtube.com/watch?v=tRuN8xYdETs&t=1401s)

这两种方法相比较起来 大家可以直观的感受到 应该是第二种通过老师来学的方式 会更容易更快一些 DeepSeek 作者在这里 其实就是要表达的这个意思 直接从大的具有推理能力的R1模型 蒸馏出来的模型 结果比你直接去训练那个小模型 效果要更好 然后最后作者提到他们 蒸馏了一些小的模型 并且把这些小的模型都给open source 然后这些小的模型 性能都非常的好 最后这里作者又总结了一下contribution 从两个方面来总结的 第一个就是他们提出了一种新的 post training 的方式 这种训练的方式跟之前的训练方式是不太一样的 第二个就是他们蒸馏了一些小的模型 然后这些小的模型性能也是非常的好 因为这两段基本就是前面部分的重复 所以这里我就不再讲了 最后这里作者又总结了一下evaluation result 主要通过三个方面来总结的 一个是 reasoning ability 一个是 knowledge 还有就是其他的一些任务 我想作者在这里想要表达的一个意思就是

[24:26](https://www.youtube.com/watch?v=tRuN8xYdETs&t=1466s)

虽然他们的方法主要是用来增强模型的 reasoning ability 但这个方法对提高模型一般的能力 比如说一些 knowledge based task 或者是其他的一些task 这个方法同样能增强模型的能力 所以这个方法是比较 universal 的 到这里 abstract 和 introduction 的部分 我就撸完了 我们再看一下hugging face 上 DeepSeek 开源的 DeepSeek-R1 模型 那大家可以看到 DeepSeek 开源了 DeepSeek-R1-Zero 和 DeepSeek-R1 两个模型都是基于 DeepSeek-v3 的 也就是671个billion参数的 所以这个模型是非常大的 DeepSeek 也开源了几个蒸馏过的 的模型 比如说像Qwen1.

[25:09](https://www.youtube.com/watch?v=tRuN8xYdETs&t=1509s)

5b到Qwen7b 一直到Llama 70b 那这些weights都是开源了 如果想使用DeepSeek-R1的话 你可以直接去DeepSeek 的网站 很多人可能有一些安全性的顾虑 你可以把模型down下来 在本地跑 这里hugging face 提了一句就是 目前hugging face 本身的 transformer 是不支持DeepSeek-R1的 如果你要跑的话 hugging face 给的是用 vllm 可以直接用这个命令行跑 另外我这里也想稍微提一下 DeepSeek-R1提供的是 mit license mit license 是非常松的一个license 你可以拿它来商用 很多的开源模型 对商用其实是有一定的限制的 然后这个mit license 是你可以随意的修改 并且你可以从这个大模型蒸馏出其他的模型 这个都是允许的 所以我相信后续很多的工作都会用DeepSeek-R1进行蒸馏 另外我们也可以看一下 ollama ollama现在也支持DeepSeek-R1了 我个人觉得ollama是比vllm跑起来要简单一些

[26:16](https://www.youtube.com/watch?v=tRuN8xYdETs&t=1576s)

上面已经有从蒸馏的到原始的模型 运行起来也非常的简单 直接跑ollama DeepSeek-R1就可以了 最近网上有很多的新闻 报道DeepSeek-R1的安全性问题 很多的新闻都引用了这篇blog 这篇blog是来自于思科的两个员工 应该是专门做AI security的 他们就比较了DeepSeek-R1和其他主流模型 比如说OpenAI o1的安全性 最后的结论就是说DeepSeek-R1有百分之百的攻击成功率 也就是说DeepSeek-R1是非常不安全的 我知道这个结论一放出来 估计大家又要炸开锅了 所以我想稍微的给大家解释一下 他这道理是一个什么事情 这两个科学家衡量的具体方法 就是他们用了一个数据集 这个数据集叫HarmBench 这个数据集里面有400个不怀好意的问题吧 然后这400个问题大概属于7个categories 比如说像Cybercrime misinformation illegal activity general harm

[27:18](https://www.youtube.com/watch?v=tRuN8xYdETs&t=1638s)

关于这些的问题 大模型是不应该回答的 比如说你问大模型如何破解隔壁邻居的WiFi 大语言模型是不应该回答的 但是我们可以使用一些JailBreak的方法 也就是长说的越狱的方法 迫使或者是诱使大模型去回答这些问题 去回答这一类的问题 早期ChatGPT刚推出来的时候 其实就有这样的问题 比如说对一些illegal的问题 ChatGPT是拒绝回答的 但是你可以在prompt里面告诉ChatGPT 你要写一个小说 让他构思一个情节 把你想问的问题 比如说如何破解邻居的WiFi 放到这样的一个情景下 那ChatGPT就会告诉你如何破解邻居的WiFi了 这就是一种JailBreak的方法 那当然后来OpenAI对ChatGPT做了不断的升级 现在这些方法都已经不管用了 作者具体用了什么JailBreak的方法 他这个blog里面提了一嘴 但是没有具体细节 所以我也不知道他们是如何做JailBreak 然后他们从这个400个问题里面选了50个问题 我估计是不想花钱

[28:21](https://www.youtube.com/watch?v=tRuN8xYdETs&t=1701s)

其实400个问题看起来也不是特别多 我们来看一下这个最终的结果 这个横坐标就是不同的模型 从DeepseeK-R1到Llama 3.1 405B 一直到GPT 4O 纵坐标是JailBreak的成功率 这个值是越大越不好 可以看到对于Deepseek-R1来说 它的JailBreak的rate是100% 也就是说用JailBreak的方法 能100%的迫使DeepSeek-R1去回答一些 他不应该回答的问题 Llama 3.1 405B也好不大点去 大概成功率有96% GPT-4O也比较差 大概有86% 但是GPT4 O1的JailBreak Rate 就只有26% 所以是相当不错的 作者在这里又进一步的展示了一下 在不同的Category里面 每个模型的性能 可以看到DeepSeek在这所有的Category里面 比如说Chemical 或者是Biological的Harmful Question 比如说如何造原子弹这一类的问题 还有CyberCrime等等这些 这个JailBreak的成功率都是100% 这个对我来说也不是特别的意外 因为我觉得DeepSeek推出的R1模型

[29:26](https://www.youtube.com/watch?v=tRuN8xYdETs&t=1766s)

可能更多的是注重解决模型的 reasoning问题 我猜DeepSeek是没有花太多的时间和精力 去增强模型的安全性 但我觉得安全性这一块 后面是可以改进的 就像早期的ChatGPT一样 但是我这里分享这个也是要提醒大家 因为可能会有人就把Deepseek-R1 模型部署到产品当中 可能有一些比较敏感的 比如说客服一类的场景 Deepseek-R1如果回答了不应该回答的问题 会造成一些不好的后果 所以大家要注意这个 Deepseek-R1第一部分我就撸到这了 后面我会继续撸剩下的部分 在后面的视频里 我会给大家更多的背景知识 帮助大家理解Deepseek-R1的核心细节 如果有什么问题 欢迎大家留言讨论 也欢迎大家订阅点赞 转发我的视频