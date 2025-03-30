
DeepSeek-R1：通过强化学习激励大型语言模型的推理能力

#### 摘要

我们介绍了第一代推理模型，DeepSeek-R1-Zero 和 DeepSeek-R1。DeepSeek-R1-Zero 是一个通过大规模 RL 训练的模型，没有经过 SFT 作为初步步骤，展现了卓越的推理能力。通过 RL，DeepSeek-R1-Zero 自然涌现出许多强大且有趣的推理行为。然而，它也面临诸如可读性差和语言混用等挑战。为了解决这些问题并进一步提升推理性能，我们引入了 DeepSeek-R1，该模型在 RL 之前结合了多阶段训练和冷启动数据。DeepSeek-R1 在推理任务上的表现可与 OpenAI-o1-1217 相媲美。为了支持研究社区，我们开源了 DeepSeek-R1-Zero、DeepSeek-R1 以及从 DeepSeek-R1 基于 Qwen 和 Llama 蒸馏出的六个密集模型（1.5B、7B、8B、14B、32B、70B）。

> 1. *非常 high level 的总结就是，他们利用少量的标注数据训练模型，然后再用这个模型产生一些新的训练数据，并把这些新的训练数据和旧的训练数据合在一起，再次去训练基础模型，有点类似像 bootstrapping 的方法，这就是DeepSeek-R1 背后的一些哲学思想* ；
> 2. *DeepSeek 这篇论文而所要挑战的就是，它认为 SFT 对于增强模型的推理能力并不是必须的，如果这个能实现的话，将会是非常大的一个贡献，因为对于推理数据的收集是非常困难的* 


> [!question]
> Figure 1 里面比较了五个模型:
> * DeepSeek-R1 vs OpenAI-o1， 这两个模型都比较的大
> * DeepSeek-R1-32B  vs OpenAI-o1-mini，蒸馏过的小模型 
> * DeepSeek-V3 是 base model；
> 
> 数据集方面： 
> * 数学相关的：AIME2024 和 Math500。AIME2024 是美国数学协会出的，美国高中的奥赛题，大概是在高中往上一点点的水平；Math500 是 OpenAI 准备的一套数据集，主要是在研究生水平的一些数学题；
> * 关于编程的数据集： Codeforces 和 SWE Bench Verify。Codeforces 是一个编程竞赛网站，有人会在上面出题，大家通过编程的方式来解决这些问题，然后有一个排名，类似于 Leet Code，Codeforces 是直接和人类进行比较的，96.3 意味着能击败 96.3% 的人类；后者来自于 Github Issue 的一个数据集，主要是考察模型debug 的能力，主要测试方法就是给模型一段代码，里面可能有 bug，要求模型找出这个 bug 并进行修正；
> * MMLU主要是考察的模型知识，涵盖的知识面非常的广，包含了各种的学科，如人文、数学、法律、物理、医学等等，代表的是人类的平均水平
> * GPQA Diamond 是另外一个极端，这个数据集的题目不多，大概只有 448 道题，但非常非常的难，需要博士级的人才能回答，即便如此，准确率也大概只有 65% 左右，它代表的是人类的最高水平；
> 
> DeepSeek-R1 主推模型的 Reasoning 能力，也就主要体现在数学和编程上面，...，整体上可以看到 DeepSeek-R1 和 OpenAI-o1 基本是不相上下的；不论是 DeepSeek-R1 还是 OpenAI-o1 在 Codeforces 上和人类程序员直接 PK，都已经超过了约 96% 的程序员...

<img src="https://arxiv.org/html/2501.12948v1/x1.png" width="500">
**图 1.** DeepSeek-R1 的性能基准


### 1、引言

*后训练* 可以在推理任务中提高准确性，对齐社会价值观，并适应用户偏好，同时与预训练相比所需的计算资源相对较少。在推理能力方面，OpenAI 的 o1 系列模型首次通过增加链式思维推理过程的长度引入了推理时的扩展。此前的多项研究探索了各种方法，包括基于过程的奖励模型、强化学习，以及蒙特卡洛树搜索和束搜索等搜索算法。然而，这些方法都未能达到与 OpenAI 的 o1 系列模型相当的通用推理性能。**（TLDR：这些方法都达到 OpenAI 性能，复现不出来）**

在本文中，我们首次尝试使用纯 RL 来提升语言模型的推理能力。具体而言，我们使用 DeepSeek-V3-Base 作为基础模型，并采用 GRPO 作为 RL 框架来提高模型在推理方面的表现。在训练过程中，DeepSeek-R1-Zero 自然涌现出许多强大而有趣的推理行为。**（TLDR：看我的）**

然而，DeepSeek-R1-Zero 面临可读性差、语言混杂等挑战。为了解决这些问题并进一步提高推理性能，我们引入了 DeepSeek-R1，它结合了少量的*冷启动数据* 和 *多阶段训练流程*。具体来说，我们首先收集数千条冷启动数据以微调 DeepSeek-V3-Base 模型。随后，我们进行类似 DeepSeek-R1-Zero 的面向推理的 RL。当 RL 过程接近收敛时，我们通过对 RL 检查点进行拒绝采样创建新的 SFT 数据，并结合 DeepSeek-V3 在写作、事实问答和自我认知等领域的监督数据，然后重新训练 DeepSeek-V3-Base 模型。在用新数据微调后，检查点进行额外的 RL 过程，考虑所有场景的提示。经过这些步骤，我们获得了一个称为 DeepSeek-R1 的检查点，其性能与 OpenAI-o1-1217 相当。**（TLDR：人工收集数千条数据，微调；然后 RL；train 一波；数据合成；重新微调；重新 RL；得到 R1）**

我们进一步探索从 DeepSeek-R1 到较小密集模型的蒸馏。使用 Qwen2.5-32B 作为基础模型，直接从 DeepSeek-R1 蒸馏的效果优于在其上应用RL。这表明，大型基础模型发现的推理模式对于提高推理能力至关重要。我们开源了蒸馏后的 Qwen 和 Llama 系列。值得注意的是，我们的蒸馏 14B 模型大幅超越了最新开源的 QwQ-32B-Preview，而蒸馏后的 32B 和 70B 模型在密集模型的推理基准测试中创下了新纪录。**（TLDR：蒸馏的小模型比小模型直接 RL 强，蒸馏后的比原版大尺寸还好）**

### 2、方法

#### 2.1 概述

以往的研究严重依赖大量的监督数据来提升模型性能。在本研究中，我们展示了即使不使用 SFT 作为冷启动，通过大规模 RL 也能显著提高推理能力。此外，结合少量冷启动数据可以进一步提升性能。在接下来的部分中，我们将介绍：(1) DeepSeek-R1-Zero，它在没有任何 SFT 数据的情况下直接对基础模型应用 RL；(2) DeepSeek-R1，它从使用数千个长链式思维（CoT）示例微调的检查点开始应用 RL；(3) 将 DeepSeek-R1 的推理能力提炼到小型稠密模型中。


#### 2.2 DeepSeek-R1-Zero：基础模型上的强化学习

强化学习在推理任务中表现出了显著的效果，我们的前期工作也证明了这一点。然而，这些工作严重依赖于监督数据，而这些数据的收集非常耗时。在本节中，我们探讨了 LLM 在**没有任何监督数据**的情况下发展推理能力的潜力，重点关注它们通过纯强化学习过程的自我进化。我们首先简要概述我们的 RL 算法，然后展示一些令人兴奋的结果，希望这能为社区提供有价值的见解。

> [!warning]
> *关于不使用任何监督数据*，R1-Zero 从 V3 中训练出来的，而 V3 有 post-training，即有 SFT，只不过在 V3 使用 SFT 之后，已经非常的强了，所以在 R1-Zero 的时候，就可以不用任何 SFT 了？当然，还有一个可能，这里提到的 V3 base 和 V3 里面提到的不是一个 model 

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
===========================================================
A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer.The reasoning process and answer are enclosed within <think> </think> and 
<answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>. User: prompt. Assistant: 
============================================================
```
表 1: DeepSeek-R1-Zero 的模板。训练期间，<font color="red">prompt</font> 将被替换为具体的推理问题。

> [!warning]
> 要求有 reasoning process 以及放到 \<think\> 和 \</think\> 中间，在之前有一些工作中，为了让模型产生思维链 或者是做比如说 self-reflection 或者 self-evaluation，一般需要设计一些比较复杂的prompt，在 R1-Zero 的 prompt 中没有任何的部分要求模型去产生 self-reflection 或 self-evaluation，这跟后面的结果就形成了对比


##### 2.2.2 奖励建模

奖励是训练信号的来源，决定了 RL 的优化方向。为了训练 DeepSeek-R1-Zero，我们采用了 *基于规则* 的奖励系统，主要由 *两种类型* 的奖励组成：

* **准确性奖励**：准确性奖励模型评估响应是否正确。例如，在具有确定性结果的数学问题中，模型需要以指定格式（例如，放在一个框中）提供最终答案，从而可以可靠地基于规则验证正确性。同样，对于 LeetCode 问题，可以使用编译器根据预定义的测试用例生成反馈。
* **格式奖励**：除了准确性奖励模型外，我们还使用格式奖励模型，要求模型将其思考过程放在 `<think>` 和 `</think>` 标签之间。

在开发 DeepSeek-R1-Zero 时，我们*没有应用 outcome 或 process* 神经奖励模型，因为我们发现神经奖励模型在大规模强化学习过程中可能会遭遇 *reward hacking* 问题，并且重新训练奖励模型 *需要额外的训练资源*，复杂化了整个训练流程。

##### 2.2.3 训练模板

为了训练 DeepSeek-R1-Zero，我们首先设计了一个简单的模板，引导基础模型遵循我们指定的指令。如表 1 所示，该模板要求 DeepSeek-R1-Zero 先生成推理过程，然后给出最终答案。我们有意将约束限制在这种结构格式上，避免任何特定内容的偏见，例如强制反思性推理或提倡特定问题解决策略，以确保我们能够准确观察模型在强化学习过程中的自然进展。

##### 2.2.4 DeepSeek-R1-Zero 的性能、自我进化过程和顿悟时刻

**DeepSeek-R1-Zero 的性能**

![[Pasted image 20250315235616.png|500]]
**表 2**：DeepSeek-R1-Zero 与 OpenAI o1 模型在推理相关基准测试上的比较。

> [!warning]
> 这个结果直接就告诉我们，他们基本上把 o1 的结果复现出来了，R1-Zero 在很多结果上已经超过了o1-mini，和 o1 不相上下，除了在 coding 这一块要略差一些。

<img src="https://arxiv.org/html/2501.12948v1/extracted/6147501/figures/plot_aime_with_maj.png" width="500">
**图 2**：DeepSeek-R1-Zero 在训练过程中的 AIME 准确率。对于每个问题，我们采样 16 个响应并计算整体平均准确率，以确保评估的稳定性。

> [!warning]
> 在 AIME 数据集上的准确度，也就是在数学问题上 R1-Zero 的准确度，横坐标是训练时间，纵坐标是准确度，这里让模型产生 16 个 response，然后做 majority voting，在不做 majority voting 时，R1-Zero 的性能已经逼近 o1了，加上 majority voting 后，可超过 o1，所以通过这两个结果能够看出来，只用 RL 训练 LLM  产生 CoT 的方法是非常有潜力的

图 2 展示了 DeepSeek-R1-Zero 在 AIME 2024 基准测试中，整个强化学习训练过程中的性能轨迹。如图所示，随着强化学习训练的推进，DeepSeek-R1-Zero 的性能稳步提升。值得注意的是，AIME 2024 的平均 pass@1 得分显著提高，从初始的 15.6\% 增加到令人印象深刻的 71.0\%，达到了与 OpenAI-o1-0912 可比的性能水平。这一显著提升突显了我们的强化学习算法在优化模型性能方面的有效性。

表 2 提供了 DeepSeek-R1-Zero 与 OpenAI 的 o1-0912 模型在各种推理相关基准测试中的对比分析。结果显示，强化学习使 DeepSeek-R1-Zero 在无需任何监督微调数据的情况下获得了强大的推理能力。这是一个值得注意的成就，强调了模型仅通过强化学习就能有效学习和泛化的能力。此外，通过应用多数投票，DeepSeek-R1-Zero 的性能可以进一步提升。例如，当在 AIME 基准测试中使用多数投票时，DeepSeek-R1-Zero 的性能从 71.0\% 提升到 86.7\%，从而超越了 OpenAI-o1-0912 的表现。

DeepSeek-R1-Zero 能够在有无多数投票的情况下实现如此具有竞争力的性能，突显了其强大的基础能力及其在推理任务中进一步进步的潜力。
<img src="https://arxiv.org/html/2501.12948v1/extracted/6147501/figures/plot_length.png" width="500">
**图 3**：在强化学习过程中，DeepSeek-R1-Zero 在训练集上的平均响应长度。DeepSeek-R1-Zero 自然地*通过更多的思考时间来解决推理任务*。

> [!warning]
> 可能是这篇论文最重要的图之一，模型会 *自己学会* 使用更多的思考时间，横坐标是训练步，纵坐标是模型回复的程度，随着模型的训练，模型产生的答案越来越长，模型在没有给它任何要求的情况下，自己学会了产生长的思维链，去帮助更好的解决复杂的问题。
> 
> 稍微提一下，在实际应用过程中，并不是思维链越长越好，如果思维链太长，会导致的一个问题叫 overthinking，这会导致成本增加，同时也不满足人类的偏好。

**自我进化过程的 DeepSeek-R1-Zero**

DeepSeek-R1-Zero 的自我进化过程是一个展示强化学习如何驱动模型自主提升推理能力的精彩例子。通过直接从基础模型启动强化学习，我们可以在没有监督微调阶段影响的情况下，密切监控模型的进展。这种方法清晰地展示了模型随时间的演变，特别是在处理复杂推理任务方面的能力。

如图 3 所示，DeepSeek-R1-Zero 的思考时间在训练过程中持续改善。这种改善并非外部调整的结果，而是模型内部的内在发展。通过延长测试时间的计算，DeepSeek-R1-Zero 自然获得了解决日益复杂推理任务的能力。这种计算从生成数百到数千个推理标记不等，使模型能够更深入地探索和优化其思维过程。

这种自我进化过程中最引人注目的方面之一是随着测试时间计算的增加，复杂行为的出现。诸如反思（模型重新审视和重新评估其先前步骤）以及探索问题解决的替代方法等行为自发涌现。这些行为并非显式编程，而是模型与强化学习环境交互的结果。这种自发发展显著增强了 DeepSeek-R1-Zero 的推理能力，使其能够更高效、更准确地应对更具挑战性的任务。

**DeepSeek-R1-Zero 的“顿悟时刻”**

在 DeepSeek-R1-Zero 的训练过程中，观察到一个特别有趣的现象，即“顿悟时刻”。这一时刻，如表 3 所示，发生在模型的中间版本。在此阶段，DeepSeek-R1-Zero 通过重新评估其初始方法，学会为问题分配更多的思考时间。这种行为不仅证明了模型推理能力的提升，也展示了强化学习如何带来意想不到的复杂结果。

![[Pasted image 20250316001329.png|500]]
**表 3**：DeepSeek-R1-Zero 中间版本的一个有趣的 *“顿悟时刻”（aha moment）*。模型学会用拟人化的语气重新思考。这对我们来说也是一个顿悟时刻，让我们见证了强化学习的力量和美妙。

> [!question]
> 第二个最重要的图，这个图所想展示的，模型给定这样的一个数学问题，在思考链的中间，模型突然产生了 *wait wait wait, that's an aha moment I can flag here*，注意这一句话不是人类加上去的，是模型自己产生的，这个现象就非常有意思，猜测这可能有两个原因：
> 
> 1. 作为 base model 的 V3，训练数据当中可能会有类似的数据存在
> 2. 通过 RL 让模型探索未知，产生了一些训练数据中所没有提供的新的预测结果，

这个时刻不仅是模型的“顿悟时刻”，也是观察其行为的研究人员的顿悟时刻。它强调了强化学习的力量和美妙之处：我们不需要明确教模型如何解决问题，只需提供适当的激励，它就能自主发展出高级问题解决策略。“顿悟时刻”强有力地提醒我们，强化学习有潜力在人工系统中解锁新的智能水平，为未来更自主和自适应的模型铺平道路。

**DeepSeek-R1-Zero 的缺陷**

尽管 DeepSeek-R1-Zero 展现出强大的推理能力，并自主开发出意想不到且强大的推理行为，但它面临一些问题。例如，DeepSeek-R1-Zero 在 *可读性差* 和 *语言混杂* 等挑战上存在困扰。为了使推理过程更具可读性并与开放社区共享，我们探索了 DeepSeek-R1，这是一种利用人类友好的冷启动数据进行强化学习的方法。

> [!question]
> 从某种程度上来说，R1-Zero 只是一个 proof of concept 的实验，但结果已经让人非常激动了，用非常简单的训练框架，就能训练出一个和 o1 所媲美的模型

#### 2.3 DeepSeek-R1：带有冷启动的强化学习

> [!question]
> R1 的训练流程：在 post training 阶段，仍然是使用 SFT + RL，总共进行了两轮：
> * 第一轮：先做 SFT 初始化，再做 RL。RL 中可以把 LM 作为策略模型，直接用去 train 比较难，解决思路就是先做 SFT，初始化这个策略模型；然后再用 RL 训练，就会更容易一些
> 	* SFT 初始化怎么做？需要有 SFT 数据（冷启动数据），怎么获取？三个方法：
> 		* few-shot long CoT as an example，在 prompt 里面加入一些示例，这些示例是包含 CoT 思考过程的，这样模型在回答的时候就会产生一些 CoT；
> 		* 直接 prompt，让模型去产生带有 CoT 的答案，类似 zero-shot 的方法，比如可以在 prompt 中加入，let's think step by step，也可以使用更复杂的一些prompt，让模型产生一些更复杂的思维过程（这两个方法都是 prompt 工程）
> 		* 用 R1 zero 产生数据，但必须经过一些手动的整理，因为它产生的数据格式会比较乱 
> 	* 用冷启动数据，finetune V3
> 	* 然后 RL 训练，使用 GRPO，reward 和 r1-zero 类似，有 accuracy 奖励和 format 奖励，此外 *还加入一个 language consistent 奖励* ，鼓励模型产生更加一致的语言
> 	* 这一轮主要目的是增强模型 reasoning 的能力，但我们在使用的时候，不光希望模型有 reasoning 能力，还应该有一些比较 general 的能力，引入第二轮
> * 第二轮：增强模型的 reasoning 能力 *以及 general 的能力* ，仍是 SFT + RL
> 	* SFT：
> 		* *作者并没有用第一轮的 checkpoint 上继续微调*，而是在 V3 上面重新做 SFT
> 		* SFT 数据：首先产生了一些 reasoning 数据，又产生了一些 non-reasoning 数据，混合，对模型做 SFT，其中 reasoning 数据就是用前面训练好的模型产生大量的数据，然后做拒绝采样。
> 		* 如何做拒绝采样？把这些数据交给 V3 模型进行打分，通过这种方式，总共产生了 600k 的 reasoning 数据。
> 		* non-reasoning 数据如何产生？之前训练 V3 的 SFT 数据，以及让 V3 通过prompting 的方式，又产生了一些新的数据，混合，产生 200k 的 non-reasoning 数据
> 		* 总共有 800k 的 SFT 数据，对 V3 做 SFT
> 	* RL，这一步的目的，在增强模型 reasoning 能力的同时，也保证模型 general 能力，包括模型的 helpfulness 以及 harmlessness，这里就和 ChatGPT 使用的 RLHF 基本是一致的。
> 		* 奖励：除了前面的 rule-based 奖励，还使用人类偏好，rule-based 主要为了增强模型的 reasoning 能力，而人类偏好主要是为了模型的 general 能力
> 		* 作者稍微提了，他们还用了一些不同的 prompt，主要让模型有更好的泛化能力
> 

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

![[Pasted image 20250328225016.png|500]]

**表 4**：DeepSeek-R1 与其他代表性模型的比较。

> [!question]
> 首先整体上 V3 模型在 coding 和 math 上面已经比其他两个模型好了一大截，这就间接说明 R1 所使用的 V3 base 模型本身的推理能力就已经是非常强了，R1 做的就是进一步加强模型的推理能力，R1 对比 V3 有显著提高，对比 o1-mini 小模型几乎全面碾压，和 o1 基本持平。


在教育导向的知识基准测试中，如 MMLU、MMLU-Pro 和 GPQA Diamond，DeepSeek-R1 展现出优于 DeepSeek-V3 的性能。这一提升主要归功于在 STEM 相关问题上的准确性增强，通过大规模强化学习实现了显著的进步。此外，DeepSeek-R1 在 FRAMES 任务中表现出色，这是一项长上下文依赖的问答任务，展示了其强大的文档分析能力。这突显了推理模型在 AI 驱动的搜索和数据分析任务中的潜力。在事实性基准测试 SimpleQA 上，DeepSeek-R1 超越了 DeepSeek-V3，证明了其处理基于事实查询的能力。在这个基准上，OpenAI-o1 同样超过了 GPT-4o。然而，在中文 SimpleQA 基准上，DeepSeek-R1 的表现不如 DeepSeek-V3，主要原因是其在安全强化学习后倾向于拒绝回答某些查询。在没有安全强化学习的情况下，DeepSeek-R1 可以达到超过 70% 的准确率。

DeepSeek-R1 在 IF-Eval 基准测试中也表现出色，该测试旨在评估模型遵循格式指令的能力。这些改进与在 SFT 和 RL 训练的最后阶段加入指令跟随数据有关。此外，在 AlpacaEval 2.0 和 ArenaHard 上的出色表现表明 DeepSeek-R1 在写作任务和开放域问答中的优势。其显著超越 DeepSeek-V3 的表现强调了大规模 RL 的泛化优势，这不仅提升了推理能力，还改善了各个领域的表现。此外，DeepSeek-R1 生成的摘要长度简洁，ArenaHard 平均为 689 个标记，AlpacaEval 2.0 为 2,218 个字符。这表明 DeepSeek-R1 在基于 GPT 的评估中避免了长度偏差，进一步巩固了其在多任务中的稳健性。

在数学任务中，DeepSeek-R1 的表现与 OpenAI-o1-1217 相当，远超其他模型。在编码算法任务（如 LiveCodeBench 和 Codeforces）中，专注于推理的模型在这些基准上表现出色。在面向工程的编码任务中，OpenAI-o1-1217 在 Aider 上优于 DeepSeek-R1，但在 SWE Verified 上表现相当。我们相信，随着相关 RL 训练数据量的增加，DeepSeek-R1 的工程性能将在下一个版本中得到提升。

#### 3.2 蒸馏模型评估

**表 5**：DeepSeek-R1 蒸馏模型与其他可比模型在推理相关基准上的比较。

如表 5 所示，仅通过蒸馏 DeepSeek-R1 的输出，就能使高效的 DeepSeek-R1-7B（即 DeepSeek-R1-Distill-Qwen-7B，以下类似缩写）*在各方面超越非推理模型*，如 GPT-4o-0513。DeepSeek-R1-14B 在所有评估指标上超过了 QwQ-32B-Preview，而 DeepSeek-R1-32B 和 DeepSeek-R1-70B 在大多数基准测试中*显著超越了 o1-mini*。这些结果展示了蒸馏的强大潜力。此外，我们发现对这些蒸馏模型应用 RL 可以获得显著的进一步提升。我们认为这值得进一步探索，因此这里只展示了简单的 SFT 蒸馏模型的结果。

### 4、讨论

#### 4.1 蒸馏 vs RL

![[Pasted image 20250318215409.png|600]]
**表 6**：蒸馏模型和强化学习模型在推理相关基准测试上的比较。

在第 3.2 节中，我们看到，通过蒸馏 DeepSeek-R1，小模型可以取得令人印象深刻的结果。然而，仍然有一个问题：*不通过蒸馏，仅通过论文中讨论的大规模强化学习训练，模型能否达到相当的性能？*

为了解答这个问题，我们对 Qwen-32B-Base 进行了大规模强化学习训练，使用数学、代码和 STEM 数据，训练超过 10K 步，得到 DeepSeek-R1-Zero-Qwen-32B。实验结果如表 6 所示，经过大规模强化学习训练的 32B 基础模型，其性能与 QwQ-32B-Preview 相当。然而，从 DeepSeek-R1 蒸馏得到的 DeepSeek-R1-Distill-Qwen-32B 在所有基准测试中表现明显优于 DeepSeek-R1-Zero-Qwen-32B。

因此，我们可以得出两个结论：*首先，将更强大的模型蒸馏为较小的模型能够取得优异的结果，而依赖于本文提到的大规模强化学习的小模型需要巨大的计算能力，甚至可能无法达到蒸馏的性能*。其次，尽管蒸馏策略既经济又有效，但要超越智能的界限，可能*仍然需要更强大的基础模型和更大规模的强化学习。*

> [!question]
> 这也间接地解释了一个问题，为什么前人很多用 RL 去增强模型的推理能力的方法和 DeepSeek 基本类似，为什么却没有做出来？很重要的一个原因，如果直接在一个比较小的模型上，它的效果并不好，不论是 GRPO 还是 PPO，主要的因素就是模型的大小；
> 
> 其次，V3 模型本身是有很强的推理能力的。

#### 4.2 不成功的尝试

在开发 DeepSeek-R1 的早期阶段，我们也遇到了失败和挫折。我们在此分享这些失败经历以提供见解，但这并不意味着这些方法无法开发出有效的推理模型。

**过程奖励模型 (PRM)**

PRM 是一种合理的方法，可以引导模型更好地解决推理任务。然而，在实践中，PRM 有三个主要限制可能阻碍其成功。首先，*很难在一般推理中明确定义细粒度步骤*。其次，*判断当前中间步骤是否正确*是一项具有挑战性的任务。使用模型进行*自动标注* 可能无法获得满意的结果，而人工标注不利于规模化。第三，一旦引入基于模型的 PRM，*必然会导致奖励欺骗*，并且重新训练奖励模型需要额外的训练资源，复杂化了整个训练流程。总之，尽管 PRM 在重新排序模型生成的前 N 个响应或辅助引导搜索方面表现出色，但与其在大规模强化学习过程中的额外计算开销相比，其优势有限。

> [!question]
> 打脸 PRM，在 OpenAI、DeepMind 中有很多这方面的工作，当然包括 DeepSeek 自己， 作者就想表达，PRM 的方法其实有很多的局限性，OpenAI 和 DeepMind 的代表行工作，都是请人对每一步进行标注，然后训 PRM

**蒙特卡罗树搜索 (MCTS)**

受 AlphaGo 和 AlphaZero 的启发，我们探索了使用蒙特卡罗树搜索 (MCTS) 来增强测试时计算的可扩展性。这种方法涉及将答案分解为更小的部分，以便模型系统地探索解空间。为此，我们提示模型生成多个标签，这些标签对应于搜索所需的特定推理步骤。在训练中，我们首先使用收集的提示，通过预训练的价值模型指导的 MCTS 找到答案。随后，我们使用生成的问题-答案对来训练演员模型和价值模型，迭代地优化这一过程。

然而，这种方法在扩展训练时遇到了几个挑战。首先，与搜索空间相对明确的国际象棋不同，*生成 token 的搜索空间呈指数级增长*。为了解决这个问题，我们为每个节点设置了最大扩展限制，但这可能导致模型陷入局部最优。其次，价值模型直接影响生成的质量，因为它指导搜索过程的每一步。训练细粒度的价值模型本质上很困难，这使得模型难以迭代改进。虽然 AlphaGo 的核心成功依赖于训练一个价值模型来逐步提升性能，但由于 token 生成的复杂性，这一原则在我们的设置中难以复制。

总之，虽然 MCTS 可以在与预训练价值模型配对时改善推理性能，但通过自我搜索迭代提升模型性能仍然是一个重大挑战。

> [!question]
> MCTS 本质是一个搜索算法，可以在 inference 的时候，让模型产生很多的备选答案，然后用 MCTS 去搜哪一个答案最好。 但是在 LLM 里面有一个问题，每次生成的 token 可能性要远大于类似 AlphaZero 在棋盘上可以落子位置的可能性，搜索空间是巨大的。

### 5、结论、局限性与未来工作

在本研究中，我们分享了通过强化学习提升模型推理能力的历程。DeepSeek-R1-Zero 代表了一种不依赖冷启动数据的纯 RL 方法，在多项任务中表现出色。DeepSeek-R1 更为强大，结合了冷启动数据与迭代 RL 微调。最终，DeepSeek-R1 在多项任务中的表现与 OpenAI-o1-1217 相当。

我们进一步探索了将推理能力蒸馏到小型密集模型中的可能性。我们使用 DeepSeek-R1 作为教师模型生成了 80 万个训练样本，并微调了几个小型密集模型。结果令人振奋：DeepSeek-R1-Distill-Qwen-1.5B 在数学基准测试中超过了 GPT-4o 和 Claude-3.5-Sonnet，在 AIME 上达到 28.9\%，在 MATH 上达到 83.9\%。其他密集模型也取得了令人印象深刻的结果，显著优于基于相同底层检查点的其他指令调优模型。

未来，我们计划在以下几个方向上对 DeepSeek-R1 进行研究：

* 通用能力：目前，DeepSeek-R1 在函数调用、多轮对话、复杂角色扮演和 JSON 输出等任务上的能力*不如 DeepSeek-V3*。未来，我们计划探索如何利用长链推理（CoT）来增强这些领域的任务。
* 语言混合：DeepSeek-R1 目前针对中英文进行了优化，这可能导致在处理其他语言的查询时出现语言混合问题。例如，即使查询使用的是非中英文，DeepSeek-R1 可能仍使用英文进行推理和响应。我们计划在未来的更新中解决这一局限性。
* 提示工程：在评估 DeepSeek-R1 时，我们观察到它对提示非常敏感。少样本提示会持续降低其性能。因此，我们 *建议用户在零样本设置下直接描述问题并指定输出格式* 以获得最佳结果。
* 软件工程任务：由于长时间的评估影响了 RL 过程的效率，大规模 RL 尚未广泛应用于软件工程任务。因此，DeepSeek-R1 在软件工程基准测试中并*未显示出相对于 DeepSeek-V3 的巨大改进*。未来版本将通过在软件工程数据上实施拒绝采样或在 RL 过程中加入异步评估来提高效率。

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

![[Pasted image 20250324134718.png]]

 ------
 
看到蒸馏出来的这个小模型 比如说62B 它在GSM8K上的结果是五十七点四
甚至比这个大的模型540B 但是不使用它们的方法结果都要好
这个结果只有五十六点五 这跟DeepSeek-R1 paper里提到的 把一个具有推理能力的大模型蒸馏到一个小的模型上面
思路是非常一致的 我们可以看到google和uiuc的这篇paper已经从Jason Wei的早期的通过
prompt engineering的方法 增强cot进化到了通过训练模型
也就是在sft阶段让模型产生cot 并且sft的训练数据就是来自于模型本身
那很自然的一个想法就是 我是否能用reinforcement learning的方法使模型产生cot呢
我就用这篇paper给大家来介绍一下用reinforcement learning去增强模型cot
几个关键点 这篇paper是想从reinforcement learning的角度讨论如何复现o1的工作
这个图就对比了传统的reinforcement learning和在大语言模型下reinforcement learning
它的异同点 传统的reinforcement learning首先需要有一个agent和环境进行互动
这个agent会通过一个所谓的policy model产生一定的actions
这个policy model一般就是用neural network来代替 所以给定一个当前的状态
输入到这个policy model里面 这个policy model就会产生action 然后这个action给到这个environment之后会产生一些结果
这些结果再通过这个reward返还给这个agent 所以reinforcement learning本质就是要优化这个policy model
使得最终cumulative reward最大 如果我们放到大语言模型下面去讨论这个agent就可以把它换成一个大语言模型
这个大语言模型产生的cot的每一个步骤 比如说这里的step one
也就是cot里面的第一步 一致到cot里面的step t就可以看成一个policy model的输出
我们希望优化这个大语言模型 使得它能产生一系列的cot
通过这些 cot最终去maximize这个reward model 这个reward model如何定义就是一个难点
reward model有两种定义方式 一种叫outcome reward model
也就是缩写成orm 一种叫process reward model 也就是缩写成prm
orm model其实又可以分为两种 一种叫solution level的orm model
一种叫token level的orm model 后面我会在paper里面具体给大家介绍
这个model对整个cot也就是整个solution进行判断
它不能像右边的prm model一样对每个步骤进行判断
process reward model它是对于这个思维链的每一个步骤都去做验证
比如说看step one它是否正确 step two是否正确 一直到最后一步
所以prm model在reinforcementing框架里面可以提供更丰富的监督信号
下面我想给大家讲讲领域类关于cot reward model的一些文章
在开始讲之前 我想先给大家用top down的方式介绍一个大概
reward model本质是给定一个cot 判断整体或者是每一步这个cot的好坏
这个reward model其实有两个应用场景 如果这个reward model是放在reinforcement learning框架下提供
reward signal的话 那么它就叫做reward model 但是还有一种情况是在test time compute的时候使用
这种情况下就叫verifier 这个verifier具体是一个什么意思呢 就比如说如果我们用最简单的prompt engineering的方法
在测试的时候可以产生一系列的cot 这每个圆圈表示cot的step
比说这是step one step two一直到step n 我们可以用self consistent的方式产生很多这种平行的cot
比如这样 我们可以用majority voting的方式去选择最终的solution
也可以用一个verifier去选择最好的答案 这个verifier本质就是一个大语言模型
经过训练之后能够判断哪个cot更好 当然在这个基础之上
我们可以有很多的变种 比如说我们可以以树的形式 在每一步都用verifier去验证
从而得到最后的solution 当然这只是几个简单的例子 如何使用verifier也有很多很多的变种
因为DeepSeek在它的paper最后提到了一些关于reward model的讨论
而领域类几篇比较重要的paper是把reward model和verifier这两个概念混在一起
所以我就也把这两个概念混在一起给大家挑几篇paper讲解一下
但是大家在读paper的时候请稍微区分一下这两个概念 我们先来看OpenAI在2021年发表的一篇文章
这篇文章应该可以算是化石级的文章了 但我觉得这篇文章非常有开创性
因为很多人认为test time compute是在发表o1时候提出来的 其实不是
在2021年发表的文章OpenAI就已经提出了test time compu的这个概念
所以我觉得这篇文章对于理解后续的工作是一个基石 这篇文章主要的想法是提出训练verifier
去解决数学问题 它有几个主要的贡献点 第一个就是它提出了GSM8K这个数据集
这就是我们前面多次提到的那个数据集 这个数据集是一个小学水平的数据集
但与其他数据集不同的是 这个数据集提供了解决这些数学问题的解题思路
并且这些解题思路都是由人类所标注的 有了这个数据集 就极大地推动了cot这个领域的研究和发展
第二个我觉得比较重要的贡献就是这个paper提出了一种基于orm
也就是outcome reward model的方式去验证模型最后cot是否正确
从而实现test time compute 我觉得这三个贡献都非常的重要 我们就挨个儿的说一下
GSM8K总共有八千五百个grade school 也就是小学水平的数学题
这些数学题 里面的问题都是由人类去创建 这些问题所对应的答案也都是由人类去写
并且当时的OpenAI还把这整个数据集都给公开了 放到了github上面
然后OpenAI就提出了一种去训练verifier的方式 在这里有两个概念
一个叫generator 一个叫verifier 这两个东西在这篇paper里面都是同一个gpt模型
它们用的是GPT3 175B或者是6B generator就是我们想要训练的那个大语言模型
让它能够产生cot verifier就是给定一个cot
判断这个cot是正确还是不正确 如何训练这个verifier呢
OpenAI提出的方法分为三步 第一步先把训练数据中的question和solution
人工针对每个question写出来的答案一起送给大语言模型进行训练
所以这一步本质就是在做supervised finetuning 当训练好这个大语言模型之后
给定一个question 让这个大语言模型产生一百个不同的solution
也就是这里的s1到s100 然后让人类去标注这每个solution是否正确
所以针对每个solution都会有一个label 也就是这里的y1,y2一直到y100
所以这一块跟前面的paper思想非常像 也就是我们用cot标注数据去训练一个大语言模型
然后让这个大语言模型再产生更多的cot数据 用人工的方法去标注一下这些cot数据
这样就产生了大量的cot训练数据 并且这些训练数据带有label
这个label作用就是对应的cot是正确的还是错误的 然后把这产生的数据
也就是给定一个question 一个solution以及它的ground truth 这个solution是正确的还是不正确的这个verifier去做训练
所以这一步本质也是supervised learning 通过这种方式训练之后 我们就得到了一个verifier
这个verifier的作用就是给定一个question以及cot 预测它的label
也就是整个cot正确还是不正确 有了这个训练好的verifier
在inference的时候怎么使用呢 这也就是OpenAI首次提出的test time compute, 给定一个问题
首先它让这个generator GPT3 产生一百个带思维链的答案
solution1一直到solution100 然后把这每个答案经过这个verifier并预测它是否正确
所以针对这每个solution verifier都可以给出一个score 我就用p表示
然后再从中选出最好的那个score所对应的solution 因为这大部分的计算
比如产生一百个solution以及由verifier进行预测
都发生在inference time 也就是test time 所以OpenAI把这个方法称为test time compute
这个方法跟我们前面提到的self consistent方法非常的像 唯一的区别就是这里引入了一个verifier
而不是用majority voting的方法 作者在这里比较了一下finetune的方法和verification的方法哪个好
finetune的方法就是直接用人类的cot数据做supervised finetuning
verification就是利用训练好的verifier 继续做test time compute
左边是一个6B的GPT3模型 右边是一个175B的GPT模型
基于这种verification的方法都比supervised finetuning方法要好 这个图其实就表明了test time compute可以增强模型的性能
作者这里还做了一些有趣的实验 左边这个图的横坐标就是在inference时候generator产生多少个备选的solutions
从二十五一直到三千两百个 然后纵坐标是解题的成功率
在产生四百个候选的solution的时候 解题的成功率是最大的 为什么产生的候选solution大于
四百之后性能是在下降呢 OpenAI给出的解释是候选答案太多的时候
因为这个6B的verifier并不是特别的完美 使得这其中的某一些solution迷惑了这个比较小的verifier
让他觉得其中的某些solution可能是正确的 这就有点类似于我们之前提到的reward hacking的例子
所以导致性能的下降 接着OpenAI又做了一个实验 用verifier选出了一定的结果之后
然后再做majority voting 坐标表示的就是我用verifier选多少结果出来
做majority voting 相比较直接用verifier选出最好的solution
我们选出十个比较好的solutions 再用majority voting的话 可以进一步提高性能
所以OpenAI这里把verification的思想和majority voting的思想 又结合起来 我们再来看一下OpenAI是怎么去训练这个verifier
OpenAI的verifier就是一个大语言模型 和前面提到的generator是同样的架构
只不过它在这个大语言模型上面加入了一个scalar head 这个scalar head具体是干什么的呢
OpenAI把vocabulary里面的一个special token作为了
verifier prediction 模型对于这个special token会预测一个probability 它的ground truth对应的就是整个solution是正确还是错误
注意这里使用的ground truth是针对整个solution的 对于里面的每个步骤是否正确是没有ground truth
所以这里所有的token对应的ground truth都是一样的 如果这个solution是正确的
那所有token的ground truth就是正确 如果这个solution ground truth是错误的 那么这所有token对应的ground truth也都是错误
OpenAI使用这种special token去代表verifier prediction 有一个好处
就是它不影响这个大语言模型对于其他token的预测和使用
我们来看一下这个图 左边的这个图是在训练generator 右边的这个图是在训练verifier
左边这个图本质就是在做前面我们提到的supervised finetuning 然后使用的loss就是next token prediction
给定一个question以及人类标注的solution 就是要预测它的下一个token
所以这里有一个shift 同时又是因为我们想模型学习如何产生这种带cot的solution
所以prompt里面的question不应该进入lost的计算 所以这里灰掉了 当我们有了训练好的generator
产生了大量的solution数据 并且经过人工标注之后 我们就来训练verifier
这里的label就是来自于human的标注 所以给定一个question以及对应的solutions
如果这个solution整体是正确的 那么这里的label就全是一
如果这个solution整体是错误的 那么这里的label就全是零 通过这种方式
我们就可以训练一个verifier 而这里作者就设计了两个变种 一种叫solution level的verifier
另外一种就是叫token level的verifier token level的verifier比较容易理解
就是这个verifier 它对于每一个solution的token 它其实都可以有一个prediction
solution level的verifier 意思就是只使用这最后一个token 的预测
因为这个结果看到了所有token 也就是整个solution做出的prediction
所以它是一个solution level的prediction 作者在这个图里也比较了一下这两种verified差异
solution level的verifier 也就是这个橘色的线 在训练的早期是比token level verifier结果要好
但是当训练到一定的apple之后 这个token level的verifier性能就要超过solution level verifier
这里有一个可能的解释 就是对于token level的verifier 其实对于模型来说
这个任务更难 因为模型需要在cot比较早期的那些token 就要预测整个solution是否正确
这是非常难的一个事情 从另外一个角度 这个token level的verifier特别像alphago里面所使用的value function
在围棋中途棋还没有下完的时候 这个模型就要预测最终是赢还是输
因为这个token level的verifier结果比较好 所以OpenAI以及后续的工作 基本都是用的这个token level的verifier
这种token level的verifier还有一个好处 提供了可解释性 如果模型出错了
通过这种可视化 我们就能判断出来模型不确定的部分是哪里
OpenAI在随后二零二三年的时候又发表了一篇文章 叫做let's verify step by step
这是OpenAI最后一篇有技术细节的paper 所以很多人反复研读这篇paper
希望能找到o1的秘诀 我觉得这篇文章就是前面OpenAI那篇文章的继续
在前面那篇文章里面 OpenAI提出了一个GSM8K的数据集 在这个数据集上面
他们提出了一个orm model 也就是outcome reward model 这个model是对整体的思维链solution进行打分
好还是不好 在这篇paper里 OpenAI提出了一个更加细化的数据集 叫做PRM800K
我们来对比一下 在GSM8K里面 给定每一个(问题)数据集 由大语言模型
当时是GPT3 产生多个solution s1,s2 直到sn 然后由人来进行标注这个solution是正确的还是错误的
也就是label1一直到label n 然然后用这个数据train了一个orm model
在PRM800K里面 对于每一个问题 同样的用大语言模型
这里用的是GPT4 产生多个solution 但不同的是 对于这每个solution
OpenAI把它的cot分成了更细的steps 也就是推理的步骤 我这里用t来表示
从t1直到tm 后面会有具体例子 展示这每一个推理步骤是什么意思
然后OpenAI雇了很多人 每一个推理的步骤进行标注 也就是y11一直到y1m
然后以此类推 有了这样的数据之后 OpenAI就可以训练一个 process reward model
也就是这个模型可以对solution里面数列的每一个步骤进行判断是否正确
可以理解为prm的分辨率要比orm的分辨率更高 这里我想再跟大家强调一下这个reward model两种用法
以免大家产生混淆 reward model它可以用在reinforcement learning里面提供
reward signal 也可以在test time compute的时候作为一个verifier去选择哪个答案最好
在OpenAI这篇paper里面 他们讨论的是verifier这种用法 换句话说
在这篇paper里面还没有涉及到任何的reinforcement learning的部分 在后面我会给大家讲一篇DeepMind的paper
在那篇paper里面 reward model既作为reinforcement learning里面的 reward signal
也作为test time compute里面的verifier去选取最佳答案 在DeepMind那篇paper里面
它们的结论是使用orm model和使用prm model结果是差不多的
那OpenAI的这篇paper里面 它主要的结论是prm model其实要比orm model要好
所以这两篇paper的结论是有一点矛盾的 DeepSeek-R1的工作就是完全抛弃了这个reward model
在reinforcement learning阶段是用rule base的方法 在test time compute的时候使用的是最简单的majority voting
我们先来看一下fig 1这个PRM800K数据集是如何标注的 给定一个数学问题
这里有多个GPT4模型产生的推理步骤 这每一行就是
一个cot的step 然后OpenAI让人类的标注者对这每一步进行标注
分别是negative neutral和positive negative比较容易理解 就是这步推理是错误的
positive就是这步推理是对的 neutral的意思就是这步推理从技术上讲没有任何的问题
但是对整个答案其实没有太大的帮助 对于这个例子 人类的标注者认为这前面所有的推理步骤都是正确的
直到最后一步解方程的时候出了错误 所以有了这样的数据集 每个推理步骤都有人类的标注
我们就可以仿照前面orm model训练的方式 训练一个prm model
这个prm model的作用就是给定每一个推理步骤 可以判断出它是对还是错
这个prm model的训练过程跟前面的orm model类似 因为这部分内容对帮助我们理解DeepSeek-R1没有太多的帮助
所以这里我就略过 这个图就展示了 当我们有了一个训练好的prm model之后
我们对模型输出的cot solution可以进行打分 并且这种评价是可以到每个推理步骤上面的
比如对于左边这个问题 模型生成了一系列的推理步骤 prm认为所有的推理步骤都是正确的
但是对于右边这个问题 prm model认为前面的这些推理是正确的 但是到这一步
也就是红色高亮的部分推理步骤是错误的 还有后面具体解方程的这些步骤
prm也认为是错误的 这样的prm我们在inference的时候 对于一个问题
我们可以产生很多个备选的solution 然后用prm对这每一个solution进行打分
选出得分最高的那个答案作为最终的答案 这里有一个小的问题 prm的对于每一步的推理都有一个probability
p1p2 一直到pt OpenAI采取的方法是把这个proability整个都乘起来
这个乘积作为这整个solution的打分 也就是用这个proability去筛选备选的答案
这里和前面提到的self consistent方法不同的地方就是 这里用的是prm在选答案
self consistent是用的majority voting在选答案 这里展示的就是这篇paper最核心的一个结论部分
OpenAI在这里比较了三个方法 分别是prm orm和majority voting
横坐标是在inference时候让大语言模型产生多少个cot的备选答案
纵坐标是解题的成功率 可以看到这个prm这条局线明显是要好于orm的
并且这两个方法都要好于majority voting的 从上面这个数值也能看出 prm是要比orm好的
这就是OpenAI在这篇paper里面最核心的结论 注意这里OpenAI其实展示的是test time compute scaling law的结果
OpenAI在这篇文章里面没有非常直接的讨论这个概念 但我觉得完全可以从这个角度去理解
因为我们在inference的时候sample越多的备选答案 其实就是在增加test time compute
随着这个test time compute的增多 模型的性能是在不断的提高的 不论是prm orm还是majority voting
只不过这三个方法展现出来的scanning low是不一样的 prm这个方法随着test time compute增多
它的增大幅度是比另外两个方法要好 所以衍生出来的一个问题就是很多人猜测
OpenAI的o1模型会不会是一个在test time使用了prm这个方法的推理模型
我们再来看一下DeepMind在二零二二年发表的这篇文章 这篇文章其实在刚才那篇文章之前
但是我故意把这篇文章放在这儿讲 是因为逻辑上这样更连续 容易理解
DeepMind这篇文章主要就是比较了process和outcome base的两种方法
也就是我们前面提到的prm和orm 但是这篇文章和OpenAI前面两篇文章不同的是
他们把orm prm这个reward model既用到了reinforcement learning
里面作为reward signal的来源 也作为了test time compute的时候去选取最佳答案的verifier
但是DeepMind在这篇文章里面 它用了另外一个词叫decoding time compute
但是这两个其实说的都是一个东西 这片paper主要用的数据集是GSM8K
和前面OpenAI的做法类似 对于一个问题 DeepMind让大语言模型产生很多个不同的cot solution
然后对于这每个solution里面的推理步骤 DeepMind也是请人进行了标注
然后和前面的方法一样 用这个数据训练了prm 这个图展示的就是如何结合reward model
在reinforcement learning里面去使模型产生思维链 我们先看第一个final answer rl 这个其实就是DeepSeek-R1所采用的方法
也就是它不用任何的reward model 只用数学问题最后的标准答案作为reward去训练大语言模型
ORM-RL的意思就是这个时候我不使用这些标准答案作为reward
而是使用我们前面已经训好的reward model所提供的reward score来帮助rl训练
这里orm因为它是一个outcome base model 所以它只会对整个solution
整个思维链打分 同理我们也可以用这个prm进行reinforcement训练
prm可以对每一部思维链进行打分 这个表示就是思维链当中的不同的步骤
对于第一步 prm认为这个是最好的 基于这个结果可以再往下继续产生思维链
比如说在第二步的时候 这个结果prm认为是最好 以此类推 最终产生最后的答案
这就是使用reward model在reinforcement里面训练的基本思想 这个表格展示的是在GSM8K上面的error rate
DeepMind作者这里用了两种衡量标准 一种叫trace 一种叫final answer final answer比较容易理解
就是看模型最后的输出答案是否正确 注意这里用的是error rate
所以这一列的值越小越好 trace就是看它每一步的 推理过程是否正确
这里用的是error rate 所以这个也是越小越好 我们来看一下这个结果 sft加final answer rl和majority voting
这个其实就是DeepSeek-R1所采用的方法 final answer error rate是百分之二十点二
trace的error rate是十二点一通 同样的 我们再来看一下这两个结果
这两个结果就是DeepMind所提出来的 在reinforcement learning的时候使用reward model
以及在test time compute的时候使用reward model对答案进行ranking
然后选出最好的答案 可以看到 使用了reward model之后 这个error rate从百分之二十左右降到了百分之十二左右
这个trace error从百分之十二左右降到了百分之三左右 所以这改进是非常显著的
另外如果我们比较orm和prm这两个结果的话 可以发现它们的结果是非常类似的
不管是trace error还是final answer error 所以第一个结论就是使用orm还是使用prm
这两个结果是差不多的 但是后来OpenAI发表的paper 也就是我们刚刚介绍了一篇paper
就推翻了这个结论 认为prm是要比orm好的 但是不管是OpenAI的结论还是DeepMind的结论
都是使用reward model的 不论是在information阶段还是在test time compute阶段做ranking
但是很有意思的是 DeepSeek-R1却没有沿着前人的这些思路继续去探索reward model的使用
而是反其道而行之 直接把整个reward model都拿掉了 返璞归真 回到了最原始的方法
在这里就有一个矛盾了 为什么在DeepMind的这篇paper里面 他们展示的这种原始的方法error rate是比要用reward model error rate是要大的
也就是结果要差的 我觉得这里可能有两个因素 第一个就是这个reinforcement learning的算法
DeepMind在这片paper里面使用了一个叫expert iteration的方法 而DeepSeek在R1的paper里面使用的是叫grpo的方法
这是第一个不同 第二个不同 我觉得可能是最主要的一个原因就是这个base model的大小
大家注意看 在DeepMind这片paper里面 它的base model只有70B 我们回忆一下DeepSeek-R1的base model用的是DeepSeek-v3
有671B的参数量 所以比DeepMind这片paper所使用的base model大了差不多有十倍
我们前面给大家讲了scaling law emergent ability 有些模型的能力只有在参数量上升到一定的程度之后才会体现
所以很可能有一个原因就是DeepMind这篇paper使用的base model比较小 所以他们得出的结论就是
这种最简单的方法训练出来的模型推理能力 反而是没有用reward model训练出来的模型推理能力要好的
看到这里我也有一个想法 就是在科研领域 有的时候这些大牛
比如说像Google OpenAI 他们的结论也不一定完全正确 我们有的时候需要放到一定的context
下去思考 不能盲目的听从或者是相信前人的一些结论 有些结论我们可能要自己去验证
更多的时候 一个结论的成立是有很多的前提条件的 比如说这里的base model大小
rl的算法 所以科研就是在这种不断地矛盾 不断地争论当中向前发展
DeepSeek的研究思路其实也是连续的 在二零二四年的时候 他们就发表了一篇文章叫做Math Shepherd
他们其实在OpenAI以及DeepMind的思路上面还是做了一些的探索
这篇paper主要在探索的是如何能够更好地使用prm这个方法
prm这个方法最大的难点就是它需要人类去对推理步骤的每一步是正确还是错误进行人为的标注
这就极大的增加了这个方法的难度 所以这篇paper的核心就是他们提出了一种自动的方法去获得
这种process wise supervision data 这篇文章主体的部分和DeepMind的很像 它把这个prm用在了两个方面
一个是verification 一个是reinforcement learning DeepSeek在这篇文章里
是如何自动地给思维链的每一个步骤进行标注的呢 我们来看一下这个图
给定一个问题 可以用大语言模型产生一个思维链 也就是解题步骤
然后用人对每一个解题步骤进行标注 这种方法耗时耗力 dDeepSeek就提出
我们可以利用蒙特卡罗树搜索的思想 对每个步骤进行自动的标注
具体是什么意思呢 比如说对于这个问题 我们有了第一步解题思路
也就是s1 然后这里让一个大语言模型基于第一步的思维过程继续往下生成剩下的思维过程
也就是这里的s2 s3 当然我们不只是产生一种思维链 我们也产生好几种思维链
在这个图中是三种 这就好比在alphago里面下围棋 我们已经下了第一步
也就是这里的s1 然后我们用模型去模拟接下来可能发生的事情
让模型自己去产生可能剩下来的走法 比如说这里就有三种走法 然后再统计最后的输赢
第一种走法最终是胜利 第二种也是胜利 第三种是失败 然后我们就可以算两个ground truth label
第一种作者叫soft estimation 也就是算一个正确答案的百分比 这里有两个是正确的
所以是二除以三 第二种方法作者叫hard estimate 也就是这三个答案里面只要有一个答案是正确的
那么这个label就是1 当有了这样的一个label之后 我们就可以把这个label作为s1的这一步的ground truth label
s1这一步思考过程有三分之二的概率可以让接下来的思考过程通往正确的答案
或者从hard estimate的角度来讲 s1可以通往正确的答案 类似的我们也可以对s2做下面的这些计算
这样我就可以给思维链过程当中的每一步都打上了标签 然后再用这种每一步都有标签的思维链过程去训练prm model或者是orm model
然后类似于DeepMind那篇paper 一样可以把这个训练好的reward model放到reinforcement learning里面
也可以放到test time compute里面作为一个verifier去选取最佳的答案
因为这一部分的过程和DeepMind的paper非常像 所以这里我就不仔细解释了 我们直接来看一下结果
左边这个图是在GSM8K上面的准确度 横坐标是不同的大语言模型
黄色的部分表示使用了他们的方法训练出来的模型 可以看到 在GSM8K上面
这个结果已经达到了GPT4的水平 而且GSM8K上面的结果已经达到了百分之九十多
所以这个数据集基本上已经被刷爆了 我们可以回忆一下 OpenAI最早提出这个GSM8K数据集的时候
最好的结果也才百分之四十左右 短短两三年的时间 随着技术的发展
这个数据集就已经被刷爆了 所以在后续的paper里面 很少有paper在用这个GSM8K数据集
转而用这个MATH数据集 小学水平的数学问题对大语言模型已经不是一个难事
在这篇paper之后 并没有沿着prm这个思路往下走 我的猜测是他们可能遇到了一些问题
像他paper里面提到的 prm存在reward hacking的问题 所以经过一些研究之后
他们抛弃了prm这个思路 进而发展出来了后面的DeepSeek-R1
希望给大家铺垫了这么多 能帮助大家对DeepSeek-R1有一个更好的理解
今天的视频差不多就到这里了 我来给大家做一个小结 这个视频里我主要给大家介绍了cot技术
cot技术可以说是现在主流的大语言模型 比如说OpenAI的o1
DeepSeek的R1非常重要的一个基础 cot技术让大语言模型具备了一定的system two的思考能力
为了使模型能够产生cot 领域内目前有很多的方法
比如说早期的时候可以直接用prompt engineering的方法
let's think step by step 或者是也可以收集有cot标注的数据 直接用sft的方法去训练大语言模型
这样训练出来大语言模型就可以产生cot 或者是使用reinforcement learning的方法使大语言模型产生更好的cot
在这个视频里面 我们也花了很多的时间去讲cot里面的reward model和verifier
因为这个跟DeepSeek-R1的工作直接相关 领域内早期像OpenAI DeepMind其实都是在研究如何去构建一个更好的reward model
DeepSeek也在这方面有一些跟进的工作 但是最后DeepSeek还是放弃了这个思路
而在reinforcement里面直接用rule based reward 我也给大家提了一下test time compute
因为有了cot这个技术之后 我们可以让大语言模型在inference的时候产生很多的cot
然后用verifier或者是用majority voting的方法 从这众多的cot里面选一个最好的作为最后的答案
这种test time compute也符合scaling law 所以领域内现在逐渐的从training time compute开始转向test time compute
这也可能是OpenAI o1背后所使用的技术 我们再来回顾一下之前我提到的top down的讲解计划
我已经给大家介绍了reinforcement learning的background 以及OpenAI GPT整个系列的发展史
通过这个发展史介绍了训练范式 scaling law emergent ability 这个序列里面也给大家介绍了cot
所以大家有了这些background再去看DeepSeek-R1这个paper的时候 你就会觉得所有的那些知识都是那么的自然
并没有很难理解 唯一可能比较难理解的就是这个grpo的部分
我会放到第三个阶段给大家讲解 通过这一系列的背景介绍视频
我希望对大家理解DeepSeek-R1这篇paper有一定的帮助 也感谢大家的支持
有任何问题 欢迎你留言 也希望你能订阅、点赞、转发 让更多的人能够看到

