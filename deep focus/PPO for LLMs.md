
理解为我们带来现代 LLM 的复杂 RL 算法…

Oct 27, 2025

![](https://substackcdn.com/image/fetch/$s_!PJsw!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Ff8db8bc5-f39d-4d1a-be16-26e0c0eb01a7_2502x1398.png)

过去几年中，RL 一直是 LLM 研究中最具影响力的领域之一。早期研究利用 RL 将 LLM 与人类偏好对齐，而这项将 RL 应用于 LLM 的初步工作几乎完全依赖于近端策略优化（PPO）。这一选择使得 PPO 多年来成为 LLM 训练后默认的 RL 算法——*考虑到 LLM 研究的快速发展，这可谓统治地位相当持久！* 直到最近关于 LLM 推理的研究中，研究者们才开始使用 GRPO 等替代算法。

尽管 PPO 非常重要，但除了顶级研究实验室之外，人们对它的了解甚少。这种理解上的缺失是有充分原因的。*PPO 不仅是一种包含微妙实现细节的复杂算法*，而且其高昂的计算和内存开销使得在没有大量计算资源的情况下进行实验变得困难。要成功利用 PPO，既需要对算法有深刻的理解，也需要丰富的领域知识或实践经验。

本概述将从 RL 的基本概念入手，逐步深入理解 PPO 算法。在此基础上，我们将阐述使用 PPO 的关键实践要点，包括 PPO 的伪代码及其各个组成部分。最后，通过分析几项在 LLM 领域推广 PPO 的开创性研究，我们将把这些知识融会贯通。

## 一、强化学习（RL）基础

在深入了解 PPO 之前，我们需要先学习 RL 的基础知识。本节将介绍强化学习的基本问题设置和术语。此外，我们将推导一个简单的策略梯度表达式，这是 PPO 算法的基础。

### 1.1 问题设置与术语

在进行强化学习训练时，我们有一个 agent 在某个环境中执行 actions；如下图所示。

![|500](https://substackcdn.com/image/fetch/$s_!lQCe!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fd7117e42-c6ab-43c4-8878-5a88cb99c9ae_2203x870.png)

这些行为是由一个策略预测的——我们可以将策略视为智能体的大脑——通常这个策略是参数化的。例如，在训练 LLM 的背景下，策略就是 LLM 本身。我们可以将策略下给定行为的概率建模为 $π_θ(a_t | s_t)$。当策略输出一个行为时，环境的状态会根据转移函数进行更新，转移函数是环境的一部分。我们将转移函数表示为 $P(s_{t+1} | a_t, s_t)$。然而，转移函数对 LLM 来说不太相关，因为它们通常是直通的；也就是说，我们假设 $s_t = \{x, a_1, a_2, …, a_t\}$，其中 $x$ 是提示词。

最后，代理访问的每个状态都会从环境中获得一个奖励，可能是正数、负数或零（即无奖励）。如前图所示，我们的代理会迭代行动，每个动作（$a_t$）、奖励（$r_t$）和状态（$s_t$）都与时间步长 $t$ 相关联。将这些时间步长组合在一起就形成了一个轨迹；见下文。在这里，我们假设代理在这个特定轨迹中总共在环境中采取了 $T$ 步。

![|400](https://substackcdn.com/image/fetch/$s_!cjh1!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fbee11fdb-dee8-4d4e-8819-b97642a17129_2008x338.png)


利用概率的链式法则，我们还可以通过结合以下概率来计算完整轨迹的概率：

* 每个动作 $a_t$ 都由策略 $π_θ(a_t | s_t)$ 给出。
* 每个状态 $s_{t+1}$ 都由转移函数 $P(s_{t+1} | a_t, s_t)$ 给出

轨迹概率的完整表达式如下所示：

![|400](https://substackcdn.com/image/fetch/$s_!YCeT!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F52061751-cc8a-4f3e-a889-5d4e542b21bf_2092x770.png)

**强化学习目标**：在使用强化学习训练模型时，我们的目标是最大化整个轨迹上的累积奖励（即 $r_t$ 的总和）。然而，这一目标存在几种常见变体。具体而言，我们最大化的奖励可以是折现的或非折现的。通过引入折现因子 $γ$，我们鼓励策略尽早获得奖励而非延后获取。*换句话说，当下获得的奖励比未来获取更有价值*。

![|350](https://substackcdn.com/image/fetch/$s_!8D_n!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fbbfd6da8-2406-4197-b9d0-d3a1ec301b39_1496x876.png)

我们的目标通常被表述为预期累积奖励，其中期望值是对轨迹进行的。展开这个期望值可以得到一个按轨迹概率加权的总和。我们可以用连续或离散的方式来表述这一点。

![|350](https://substackcdn.com/image/fetch/$s_!45io!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F523baab0-10b4-438e-85d7-e7c5c0681209_1692x884.png)

**状态、价值和优势函数**：与强化学习目标相关，我们还可以定义以下函数集：

* *价值函数* $V(s)$：当从状态 $s$ 开始并根据当前策略 $π_θ$ 行动时，预期的累积奖励。
* *动作-价值函数* $Q(s, a)$：当你从状态 $s$ 开始，采取动作 $a$，然后根据策略 $π_θ$ 行动时，预期的累积奖励。
* *优势函数* $A(s, a)$：动作价值函数与价值函数之间的差值，即 $A(s, a) = Q(s, a) - V(s)$。

直观地说，优势函数通过计算在状态 $s$下采取动作 $a$ 后的预期回报与状态 $s$ 的一般预期回报之间的差值，来告诉我们某个动作 $a$ 有多大的用处。如果动作 $a$ 带来的回报高于预期，优势值将为正值，反之则为负值。优势函数在强化学习研究中扮演着重要角色——*它们被用来计算策略的梯度*。

> “在强化学习中，有时我们并不需要从绝对意义上描述一个动作有多好，而只需知道它平均而言比其他动作好多少。也就是说，我们想知道该动作的相对优势。我们通过优势函数来精确表达这一概念。”——摘自 [深度强化学习入门](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html)


### 1.2 LLMs 的强化学习公式

![|300](https://substackcdn.com/image/fetch/$s_!RBDE!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fd4b8b6b8-fe96-4b70-87d2-038a3b3511cf_1346x1134.png)

既然我们已经理解了强化学习的基础知识，现在需要将所学术语映射到 LLM 的训练场景中。具体对应关系如下（如上所示）：

* 我们的 *策略* 就是 LLM 本身。
* 我们的 *初始状态* 就是 prompt。
* LLM 的输出——无论是每个 token 还是整个完成内容——都是 action。
* 我们的 *状态* 是 prompt 与 LLM 输出的结合。
* LLM 的整个输出过程形成了一条 *轨迹*。
* *奖励* 来自验证器或奖励模型。

值得注意的是，在这个设置中没有转移函数，因为转移函数是完全确定性的。如果我们从一个提示 $x$ 开始，并且我们的 LLM 根据这个提示输入预测出 token $t_1$ 和 $t_2$，那么我们更新后的状态就简单地变为 $s_2 = \{x, t_1, t_2\}$。换句话说，*我们的状态只是 LLM 针对给定提示 $x$ 正在生成的运行完成内容。*

**MDP 公式化**：对于 LLMs，RL 可以通过两种关键方式进行公式化，这两种方式在如何建模动作方面有所不同。

1. *强盗式表述*：将 LLM 的整个完成或响应建模为单一动作。
2. *马尔可夫决策过程（MDP）建模*：将大语言模型输出的每个 token 视为独立动作。

我们在之前的概述中详细介绍了这两种方案的细节。不过，PPO 依赖于 MDP 方案，因此我们在此将主要关注 MDP 方案。正如我们所知，LLM 通过下一个 token 预测来生成输出；也就是说，通过依次生成输出补全中的每个 token。这个自回归过程如下图所示。

![|450](https://substackcdn.com/image/fetch/$s_!QUg4!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F5b1a8412-5cfb-481f-bd50-473f0a6fd9b5_1992x1037.png)

下一个 token 预测可以轻松映射到 RL 的设置中——我们可以*将每个 token 建模为一个动作*！这种设置被称为马尔可夫决策过程（MDP）框架。MDP 是一种用于建模决策的概率框架，包含状态、动作、转移概率和奖励——这正是我们迄今为止讨论的强化学习设置！用于强化学习的 MDP 框架如下所示。

![|400](https://substackcdn.com/image/fetch/$s_!KWz-!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F52f4f8de-4456-4cbd-935c-a945968b704d_1466x916.png)

在将 RL 建模为 LLMs 的 MDP 时，我们的初始状态是 prompt，而策略则通过预测单个 token 来执行。我们的 LLM 形成了一种（随机）策略，预测 token 的概率分布。在生成过程中，通过从该分布中选择一个 token 来执行动作——*每个 token 都是其自身的动作*。当一个 token 被预测出来后，它会被添加到当前状态中，并由 LLM 用于预测下一个 token ——这正是自回归的下一个 token 预测！最终，LLM 预测出一个停止 token（例如 `<|end_of_text|>` 或 `<eos>`）来完成生成过程，从而产生一个完整的轨迹。

### 1.3 策略梯度基础

在 RL 训练过程中，我们的目标是最大化目标函数——即累积（可能经过折扣的）奖励。为此，我们可以直接使用梯度上升法；具体方法如下。

![|400](https://substackcdn.com/image/fetch/$s_!slrY!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Ff3072897-d905-42be-b385-6186c24ae059_2390x302.png)

将这一点放在 LLM 的背景下，RL 训练遵循以下步骤序列。我们首先采样一批提示词，并用 LLM 或策略生成这些提示词的补全内容。然后，我们计算这些补全内容的奖励，并利用这些奖励来推导策略更新。*这最后的策略更新步骤正是使用梯度上升的地方*。

![|350](https://substackcdn.com/image/fetch/$s_!yR8D!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F20b7b374-8bee-45fb-b7ee-a26008aa7259_1267x843.png)

具体来说，我们利用完成情况和奖励来估算 RL 训练目标相对于策略参数的梯度——这被称为“*策略梯度*”。如果我们能计算出这个梯度，就可以通过梯度上升法来训练策略。但问题是：*我们该如何计算这个梯度呢？*

> _“强化学习的目标是为智能体找到一种最优行为策略，以获得最优奖励。策略梯度方法旨在直接建模和优化策略。”_ - [Lilian Weng](https://lilianweng.github.io/posts/2018-04-08-policy-gradient/)

**策略梯度**：几乎所有用于 LLM 训练的 RL 优化器（如 PPO、GRPO 和 REINFORCE）都属于策略梯度算法。这类算法的运作分为两步：*i)* 估算策略梯度；*ii)* 基于估算结果执行梯度上升。不同算法在策略梯度估算方法上各有差异，但其核心思想高度相似——我们只需根据具体技术微调细节。为深入理解策略梯度算法，我们将首先推导最基础的策略梯度形式，随后扩展这一思路，推导出更复杂的算法如信任域策略优化（TRPO）和近端策略优化（PPO）。

**Vanilla Policy Gradient（VPG）**：算法已被众多网络资源详细阐述。其他关于 VPG 的有用解释还包括：

- OpenAI 策略优化入门 [link](https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html)
- [Nathan Lambert](https://natolambert.com/) 的 RLHF 书籍 [link](https://rlhfbook.com/c/11-policy-gradients.html)
- [Lilian Weng](https://lilianweng.github.io/) 的策略优化算法 [link](https://lilianweng.github.io/posts/2018-04-08-policy-gradient/)

然而，为了完整性，我们将再次推导出一些简单的策略梯度形式。正如我们已经知道的，RL 的目标是最大化累积奖励。如果我们尝试计算这个目标相对于策略参数 $θ$ 的梯度，我们可以推导出以下结果：

![|500](https://substackcdn.com/image/fetch/$s_!GetI!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F1685ea69-1b2c-438c-87ed-dba51c4bee65_2406x1065.png)

log 求导步骤中，$\ln(y)'=\frac{y'}{y}$，所以 $y'=y \ln'(y)$，([source](https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html))

这个推导过程从 RL 训练目标（累积奖励）的梯度开始，最终得出策略梯度的基本表达式。上面列举了推导过程中使用的步骤。这里唯一复杂的步骤是对数导数技巧的使用以及最后一步，这一步利用了我们对轨迹概率的定义。在最后一步中，我们代入轨迹概率的定义，并观察到初始状态概率和转移函数相对于策略参数的梯度始终为零，因为它们都不依赖于策略；详见下文 ([source](https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html))。

![|500](https://substackcdn.com/image/fetch/$s_!Rkmm!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fb0f526be-55f2-4eae-abd8-fa4382d8335a_1564x432.png)

**实现基本的策略梯度**：到目前为止，我们推导出的基本策略梯度表达式是理论性的——它*涉及期望值*。如果我们想要在实际中计算这个梯度，就必须用样本均值来近似。换句话说，我们采样固定数量的轨迹（对于 LLM 来说，就是提示和补全），并对每个轨迹的策略梯度表达式取平均值。基本的策略梯度表达式包含两个我们已经知道如何计算的关键量：

* 奖励直接来自验证者或奖励模型。
* 动作的对数概率可以通过 LLM 计算得出（即这些只是 LLM 输出的 token 概率）。

为了使计算基本策略梯度的过程更加具体，下面提供了 PyTorch 伪代码的逐步实现。
<img src="https://substackcdn.com/image/fetch/$s_!PYzF!,w_1456,c_limit,f_webp,q_auto:good,fl_lossy/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F3e4bdafe-cd71-48b7-8a10-abdc895432f7_1920x1076.gif" width="600">

| 代码步骤                              | 理论推导（策略梯度定理）                              | 具体解释                                             |
| --------------------------------- | ----------------------------------------- | ------------------------------------------------ |
| `completions = LLM(prompts)`      | 采样轨迹 $\tau \sim \pi_{\theta}$             | 让当前策略（LLM）根据 prompts 生成文本（completions），这就是在采样轨迹。 |
| `rewards = RM(completions)`       | 计算轨迹回报 $R(\tau)$                          | RM 为生成的每条轨迹打分，得到回报 $R(\tau)$。                    |
| `token_logp = F.log_softmax(...)` | 计算 $\log \pi_{\theta}(a_t \mid s_t)$      | 计算每个生成步骤（状态 $s_t$ 下选择 token $a_t$）的策略对数概率。       |
| `loss = (- token_logp * rewards)` | 构建损失 $-\sum_t \log \pi(a_t\|s_t) R(\tau)$ | 将负的对数概率与回报相乘，损失最小化等价于策略梯度上升。                     |
| `loss.backward()`                 | 计算梯度 $\nabla_{\theta} J$                  | 反向传播自动计算近似策略梯度的期望。                               |
其中， `- token_logp * rewards`，这是核心。回顾理论，策略梯度是 $\mathbb{E} [ \sum_t \nabla_{\theta} \log \pi (a_t|s_t) \cdot R(\tau) ]$。在深度学习中，我们通常定义损失函数，然后通过最小化损失（梯度下降）来优化。由于 `token_logp` 是我们要增加的量的对数，所以加上负号将其变为损失。这样，*最小化这个损失就等价于最大化期望回报*（梯度下降变为了梯度上升）。

聚合损失：代码提供了几种选项，选项 1 是最常见的做法之一。它先对每个序列的 token 损失求和，然后除以每个序列的有效长度进行归一化，最后对所有批次内的序列求平均。这确保了不同长度的序列对损失的贡献是均衡的。

在上述实现中，我们需要注意的一个关键细节是：我们并非直接计算策略梯度，而是构建一个损失函数，使其梯度等于策略梯度，然后利用 PyTorch 的自动微分功能（通过 `loss.backward()` 实现）来间接计算策略梯度。用于计算策略梯度的具体损失函数如下所示。

![|400](https://substackcdn.com/image/fetch/$s_!TwP0!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fa4bb2d85-fdea-4cfc-a46b-e6c5f78ff4f4_1613x593.png)

理解这一区别非常重要，因为我们将通过损失函数而非直接策略梯度表达式来构建 PPO（以及TRPO）。

**基本策略梯度的问题**：基本策略梯度表达式虽然简单直接，但也存在几个显著问题：

* *高方差*：梯度估计可能具有高方差，导致训练不稳定。
* *不稳定的策略更新*：目前没有机制来防止策略发生可能破坏稳定的大规模更新。

由于方差较大，准确估计策略梯度通常需要在每次训练迭代中采样大量轨迹，这在计算上非常昂贵。我们必须使用 LLM 生成大量补全结果，并为所有这些补全计算奖励和 token 对数概率。

此外，这种高方差会增加训练不稳定的风险——*大而不准确的更新可能会对我们的策略造成重大损害*。为了解决这些问题，大多数策略梯度算法专注于减少策略梯度估计的方差，并在策略更新上强制执行信任区域（即限制策略在单次更新中可以改变的程度）。

> _“按照这个梯度迈出一步，会按比例提升每个动作的对数概率，比例因子为 $R(𝜏)$——即迄今为止获得的所有奖励之和.”_ - [Spinning up in Deep RL](https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html)

**Reward-to-go.**：例如，在我们基本的策略梯度中可以看到，我们基于轨迹的累积奖励来增加给定动作的概率。因此，我们可能会因为在该动作发生之前观察到的奖励而增加该动作的概率。

![|300](https://substackcdn.com/image/fetch/$s_!Ymws!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F6b14bade-8617-4bfa-9e4a-59811bbe8de7_1374x218.png)

这一简单观察促成了"奖励累计"策略梯度的诞生。这种改进的策略梯度表达式仅用动作后观察到的奖励总和替代了累积奖励。运用 [EGLP 引理](https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html#expected-grad-log-prob-lemma)，我们可以证明这种奖励累计公式是策略梯度的无偏估计量。此外，与之前的基础策略梯度表达式相比，奖励累计策略梯度被证明具有更低的方差。

![|450](https://substackcdn.com/image/fetch/$s_!s3m9!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F92c4ac85-74ac-4c12-8d51-c6c9b3bf22ba_2216x460.png)

**Baselines.** 为了进一步降低方差，我们还可以在策略梯度表达式中添加一个基线。与奖励累积策略梯度类似，我们可以利用 EGLP 引理证明，带基线的策略梯度版本是无偏的，且具有更低的方差。根据 EGLP 引理，该基线必须仅依赖于当前状态（否则将违反 EGLP 引理的假设，导致证明不再有效）。

![|400](https://substackcdn.com/image/fetch/$s_!QhBq!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fd4801db8-b3f3-4ec3-9d3f-624b8ffbd550_1774x344.png)

这个表达式与"奖励累计"策略梯度几乎完全相同——我们*只是从"奖励累计"项中额外减去一个基线*。在策略梯度估计中，可以使用多种可能的基线选择。*一个常见的基线是价值函数。使用价值函数作为基线，可以正向强化那些获得高于预期累积奖励的动作*。

_普通策略梯度算法的一个常见问题是梯度更新的高方差……为了缓解这一问题，人们采用了各种技术来对价值估计进行归一化处理，这些技术被称为基线。基线通过多种方式实现这一目标，有效地将状态价值相对于后续动作进行归一化（例如优势函数的情况，即 Q 值与状态价值之间的差值）。最简单的基线形式包括对奖励批次取平均值或使用移动平均值。 - [RLHF book](https://rlhfbook.com/c/11-policy-gradients.html)_

**通用策略梯度**：在文献 [3] 中，作者用一个更通用的策略梯度表达式总结了计算策略梯度的几种方法；

![|550](https://substackcdn.com/image/fetch/$s_!Vl-C!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F58aa8bae-6778-4ec0-ac53-3f8b8550390f_2137x836.png)

这个表达与我们目前所见的表达几乎完全相同。唯一的区别在于，我们将奖励项 $R(𝜏)$ 替换为一个通用的 $Ψ_t$ 项，它可以被设置为几种不同的表达式。例如，我们可以：

* 将 $Ψ_t$ 设为 $R(𝜏)$ 以恢复我们的基本策略梯度表达式。
* 将 $Ψ_t$ 设为时间 $t$ 之后获得的奖励，以恢复我们策略梯度的“奖励累积”变体。
* 将 $Ψ_t$ 设置为奖励的基线版本；例如，累积奖励 $R(𝜏)$ 与价值函数 $V(s_t)$ 之间的差值。
* 将 $Ψ_t$ 设为状态-动作 $Q$ 或优势函数 $A$。

尽管存在多种可能的表述方式，*PPO（以及几乎所有用于 LLM 领域的 RL 优化器）都致力于将 $Ψ_t$ 设定为优势函数 $A(s_t, a_t)$。这一设定被称为标准策略梯度（VPG）*。理论上，VPG 能产生方差最小的梯度估计。

![|350](https://substackcdn.com/image/fetch/$s_!1PL6!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F3dbd6ad6-4d9e-4085-b4a7-849b29789350_1662x470.png)

尽管 VPG 的方差较低，但在策略更新过程中仍缺乏强制信任区域的机制——*大规模破坏性策略更新仍可能使训练过程失稳*。PPO（近端策略优化）正是为解决这一问题而诞生。我们将看到，PPO 虽然沿用了基础策略梯度的表达式，但额外增加了对策略更新施加信任区域的机制。接下来我们将深入探讨 PPO 及其实现过程中涉及的诸多实践细节。

## 二、近端策略优化 (PPO)

既然我们已经理解了 RL 的基础知识，接下来我们将学习 PPO。这一部分的讲解将基于我们在上一节中推导出的 VPG 表达式，从 PPO 的前身——信任区域策略优化（TRPO）开始。TRPO 在稳定训练方面非常有效，但也相对复杂。PPO 作为一种更实用的替代方案被提出，具有类似的优势。在本节的最后，我们还将介绍广义优势估计（GAE），这是 PPO 中计算优势函数最常用的方法。

#### [Trust Region Policy Optimization (TRPO)](https://arxiv.org/abs/1502.05477) [6]

> _“TRPO uses a hard constraint rather than a penalty because it is hard to choose a single value of β that performs well across different problems—or even within a single problem, where the characteristics change over the course of learning.”_ - from [1]

Prior to learning about PPO, we need to take a look at its predecessor, Trust Region Policy Optimization (TRPO) [6]. The key motivation behind TRPO is creating an algorithm that is data efficient and does not require too much hyperparameter tuning. To do this, authors in [6] propose the constrained objective below, _which is guaranteed to monotonically improve our policy_. This objective enforces a trust region on the policy update, thus eliminating the risk of large and destructive policy updates that could destabilize training.

[

![](https://substackcdn.com/image/fetch/$s_!x5A5!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F1a9c1514-c3dd-4692-bb7a-d63644987d5e_1784x940.png)



](https://substackcdn.com/image/fetch/$s_!x5A5!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F1a9c1514-c3dd-4692-bb7a-d63644987d5e_1784x940.png)

Surrogate objective for TRPO (from [1])

**Surrogate objective.** This objective shown above is called the surrogate objective in TRPO. This naming stems from the fact that the surrogate objective is different from the standard RL training objective. In RL, we aim to maximize cumulative reward, but—_as we have seen in our discussion of the VPG_—directly maximizing this “true” objective of RL can lead to training instability. TRPO formulates the surrogate objective to maximize in place of the true objective.

There are a few noticeable differences between the above expression for TRPO and the VPG:

- Action probabilities in the current policy are normalized by the probability of that action in the old policy (i.e., the policy prior to training)—_this forms the policy ratio (also called an importance ratio)_. We also use probabilities in this formulation instead of log probabilities.
    
- There is a constraint placed on the objective to ensure that the expected KL divergence between the new and old policies is less than a threshold `δ`.
    

Otherwise, the TRPO loss function shares a similar structure to that of VPG—_it includes the advantage function and a sum over token-level probabilities in a trajectory_.

**Policy ratio.** The centerpiece of the TRPO loss function is the policy ratio, defined as shown below. The policy ratio tells us how much more likely a given action is in our current policy relative to the probability of that action before the training process started—_this is denoted as the “old” policy_.

[

![](https://substackcdn.com/image/fetch/$s_!IXsZ!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F4a7d1530-a2cc-48c6-9e95-8571b781ba35_1994x792.png)



](https://substackcdn.com/image/fetch/$s_!IXsZ!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F4a7d1530-a2cc-48c6-9e95-8571b781ba35_1994x792.png)

The policy (or importance) ratio

This quantity serves the purpose of assigning an importance to different actions within our trajectory. If the new policy assigns a higher probability to an action than the old policy did, this ratio is greater than one, increasing the influence of that action’s advantage in the objective. Conversely, if the new policy assigns a lower probability, the ratio is less than one, reducing the influence of that action. The policy ratio ensures that the policy update emphasizes actions that the new policy is making more likely—_especially if those actions have high advantage_—while suppressing actions that are becoming less likely under the new policy. By doing this, we ensure that the update is properly weighted according to how the new policy differs from the old, enabling stable and efficient policy improvement.

**Solving the surrogate objective.** Although this objective yields stable policy updates, solving it can be quite involved. By introducing an explicit constraint into our objective, we eliminate the ability to solve this objective with simple gradient ascent[3](https://cameronrwolfe.substack.com/p/ppo-llm#footnote-3-175107358). Instead, we have to solve this objective via the more complex [conjugate gradient algorithm](https://en.wikipedia.org/wiki/Conjugate_gradient_method). Alternatively, we could remove this constraint and instead add the KL divergence as a penalty into our loss function; see below. This unconstrained loss is simpler and can again be solved with basic gradient ascent.

[

![](https://substackcdn.com/image/fetch/$s_!fFIz!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F301f1d55-7e7c-4c2f-8138-67a3bc162338_1872x388.png)



](https://substackcdn.com/image/fetch/$s_!fFIz!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F301f1d55-7e7c-4c2f-8138-67a3bc162338_1872x388.png)

The penalty objective for TRPO

**From TRPO to PPO.** Formulating the constraint from TRPO as a penalty allows us to avoid complicated optimization techniques and rely upon basic gradient ascent. However, a new hyperparameter β is introduced to the optimization process that makes tuning difficult. Properly setting the value of β is essential for this objective to perform well, and finding a single value of β that generalizes to many domains is hard. As a result, both of the above objectives have their issues:

- The TRPO surrogate objective is too complex to solve in practice.
    
- The reformulated penalty objective is sensitive to the setting of β.
    

We want to develop an algorithm that retains the benefits of TRPO—_such as stability, data efficiency, and reliability_—while avoiding its complexity. Ideally, the algorithm should be broadly applicable and solvable using basic gradient ascent. These goals led to the proposal of PPO, which is largely inspired by TRPO. PPO’s objective is inspired by the TRPO surrogate objective but replaces the hard KL constraint with a clipping mechanism to enforce a trust region in a simpler way.

#### [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) [1]

> _“We propose a new family of policy gradient methods for RL, which alternate between sampling data through interaction with the environment, and optimizing a surrogate objective function using stochastic gradient ascent.”_ - from [1]

The VPG is simple to compute in practice, but it has poor data efficiency (i.e., the model must be trained over many samples to perform well) and high variance in the policy updates. These problems are largely solved by TRPO but at the cost of significant added complexity. PPO is an algorithm with the data efficiency and reliability benefits of TRPO that is still solvable with gradient ascent. In this way, PPO is a simpler algorithm compared to TRPO. As we will see, however, _PPO is still a complex algorithm with many implementation complexities of its own_.

[

![](https://substackcdn.com/image/fetch/$s_!S1nc!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fc38f9ea3-d07f-4240-898e-de3c75e66878_2264x786.png)



](https://substackcdn.com/image/fetch/$s_!S1nc!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fc38f9ea3-d07f-4240-898e-de3c75e66878_2264x786.png)

Update procedure in PPO (from [1])

**Training process.** Similarly to TRPO, PPO focuses upon optimizing a surrogate objective, but the objective in PPO has no constraint and has been slightly modified. As shown in the algorithm above, PPO performs more than a single policy update in each step, instead alternating between:

1. Sampling new data or trajectories from the policy.
    
2. Performing several epochs of optimization on the sampled data.
    

**The PPO surrogate objective** is again based upon the policy ratio between the current policy and the old model (i.e., the policy before any training is performed). To match notation in [1], we will denote the policy ratio as `r_t(θ)`, which is similar to the `r_t` notation used for the reward for time step `t`. However, _the policy ratio is unrelated to the reward_! To obtain the PPO objective, we start with the surrogate objective being maximized by TRPO with no KL constraint; see below.

[

![](https://substackcdn.com/image/fetch/$s_!fqSm!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F80447ac5-6fd2-4cbb-b33c-a4e385e7fc2c_1390x478.png)



](https://substackcdn.com/image/fetch/$s_!fqSm!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F80447ac5-6fd2-4cbb-b33c-a4e385e7fc2c_1390x478.png)

The unclipped PPO objective

We will call this formulation the “unclipped” objective. Because it does not have a constraint, this objective can be easily computed to derive the policy gradient by _i)_ estimating the advantage and _ii)_ computing the policy ratio. However, if we try to maximize this unconstrained objective, this will potentially lead to large and destructive policy gradient updates that make the training process unstable. To solve this issue, PPO introduces a novel clipping mechanism into the surrogate objective that helps us with maintaining the trust region; see below.

[

![](https://substackcdn.com/image/fetch/$s_!oHJG!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F7f6be9f2-f165-4e48-be0c-e63074454d2a_2003x338.png)



](https://substackcdn.com/image/fetch/$s_!oHJG!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F7f6be9f2-f165-4e48-be0c-e63074454d2a_2003x338.png)

The PPO surrogate objective

The main term in the objective is unchanged, but there is an added term with a clipped version of the policy ratio—_the policy ratio must fall in the range_ `[1 - ε, 1 + ε]`[4](https://cameronrwolfe.substack.com/p/ppo-llm#footnote-4-175107358). The clipping term disincentivizes the RL training process from moving the policy ratio away from a value of one. The PPO surrogate objective takes the minimum of clipped and unclipped objectives. In this way, _the PPO objective is a pessimistic (lower) bound for the original, unclipped objective_[5](https://cameronrwolfe.substack.com/p/ppo-llm#footnote-5-175107358).

[

![](https://substackcdn.com/image/fetch/$s_!ovlv!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F38769a7f-6549-4fed-ab3e-f829185b5069_1544x642.png)



](https://substackcdn.com/image/fetch/$s_!ovlv!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F38769a7f-6549-4fed-ab3e-f829185b5069_1544x642.png)

(from [1])

Depending upon whether the advantage is positive or negative, the behavior of clipping is slightly different; see above. The use of a minimum in the surrogate objective causes clipping to be applied in only one direction. In particular, we can arbitrarily _decrease_ surrogate objective by moving the policy ratio far away from a value of one, but clipping prevents arbitrarily _increasing_ the objective via the policy ratio. In this way, PPO de-incentivize large policy ratios so that our policy does not deviate too much from the old policy after training updates.

> _“With this scheme, we only ignore the change in probability ratio when it would make the objective improve, and we include it when it makes the objective worse.”_ - from [1]

To more deeply understand the clipping logic of PPO, we can consider each of the four possible cases that can arise when optimizing the surrogate objective:

- Case #1 [`A > 0`, `r_t(θ) ≤ 1 + ε`]: advantage is positive—_this is an action that we want to reinforce_. Our policy ratio is below `1 + ε`, so we perform a normal policy gradient update to increase the probability of this action.
    
- Case #2 [`A > 0`, `r_t(θ) > 1 + ε`]: advantage is positive again, but our policy ratio is greater than `1 + ε`. This means that this action is already more likely in the new policy relative to the old policy. The objective gets clipped, and the gradient with respect to further increases in the policy ratio is zero. This prevents the policy from making the action even more likely
    
- Case #3 [`A < 0`, `r_t(θ) ≥ 1 - ε`]: advantage is negative—_this is an action we want to negatively reinforce (i.e., decrease probability)_. Our policy ratio is above `1 - ε`, so we perform a normal policy gradient update to decrease the probability of this action.
    
- Case #4 [`A < 0`, `r_t(θ) < 1 - ε`]: advantage is negative again, but our policy ratio is less than `1 - ε`. This means that this action is already less likely in the new policy relative to the old policy. The objective gets clipped, and the gradient with respect to further decreases in the policy ratio is zero. This prevents the policy from making the action even less likely.
    

The policy ratio is computed between the current and old policies. The old policy is updated to match the current policy each time new data is sampled in PPO. In the context of LLMs, we perform 2-4 gradient updates (or sometimes more) [2] for each batch of data, _so_ _the old model is updated frequently_. The clipping operation in PPO, therefore, maintains a trust region for a particular batch of data.

**KL divergence.** When training LLMs with PPO, we usually incorporate the KL divergence between the current policy and a reference policy—_usually some policy from before RL training begins (e.g., the SFT model)_—into the training process. This added KL divergence term penalizes the policy from becoming too different from the reference policy, which has a regularizing effect. We compute KL divergence per token by comparing the token probability distributions outputted by the two LLMs for each token within the sequence. Details on how exactly the KL divergence is computed in practice can be found [here](https://cameronrwolfe.substack.com/i/167254905/kullback-leibler-kl-divergence).

[

![](https://substackcdn.com/image/fetch/$s_!MMrI!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fcc3d5004-2390-489f-995a-e0245c174535_2534x530.png)



](https://substackcdn.com/image/fetch/$s_!MMrI!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fcc3d5004-2390-489f-995a-e0245c174535_2534x530.png)

Incorporating KL divergence into the reward

There are two common ways of adding the KL divergence into PPO training. First, we can directly subtract the KL divergence from the reward in RL; see above. Alternatively, we can add the KL divergence as a penalty term to the RL training objective as shown below. In both cases, we simply want to maximize rewards without making our new policy too different from the reference.

[

![](https://substackcdn.com/image/fetch/$s_!kyeM!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fc7464e10-d669-4f6b-ab83-f1980b8918d4_2416x436.png)



](https://substackcdn.com/image/fetch/$s_!kyeM!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fc7464e10-d669-4f6b-ab83-f1980b8918d4_2416x436.png)

Incorporating a KL penalty into the RL training objective

Such a KL divergence term is almost universally used in RL training for LLMs, though the exact implementation varies. Both of the approaches outlined above have been used successfully. However, capturing the KL divergence via a penalty term in the training objective is probably more common (and a bit simpler).

**The critic.** Recall that the advantage function is defined as the difference between the state-action value function and the value function. In PPO, we estimate the state-action value function—_the expected reward for taking a specific action in a given state_—by using the actual reward observed for a trajectory. The value function, in contrast, is typically estimated using a learned model; see below.

[

![](https://substackcdn.com/image/fetch/$s_!noKQ!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F55141cda-9010-48ea-ba62-5cd56e9bd814_1772x629.png)



](https://substackcdn.com/image/fetch/$s_!noKQ!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F55141cda-9010-48ea-ba62-5cd56e9bd814_1772x629.png)

For example, we can create a separate copy of our policy, or—_for better parameter efficiency_—add a dedicated value head that shares weights with the policy to predict the value function. This learned value function is often referred to as a value model or critic. Taking a partial response as input, the critic predicts the expected final reward for every token position within the sequence; see below.

**Critic versus reward model.** In the context of LLMs, we predict the reward with a reward model. Additionally, most LLMs are trained using outcome supervision, meaning that a reward is only assigned after the model has generated a complete response (i.e., after the `<eos>` token has been outputted). The critic and reward model are similar in that they are both learned models—_usually another copy of our LLM policy_—that predict rewards. However, the critic predicts expected rewards given a partial completion as input, while the reward model typically predicts the reward received by an entire response; see below. Going further, the reward model is fixed throughout RL training, while the critic is continually updated.

[

![](https://substackcdn.com/image/fetch/$s_!fXOv!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Ffb8133ba-f772-44f5-bfbc-19e800a842cc_1732x570.png)



](https://substackcdn.com/image/fetch/$s_!fXOv!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Ffb8133ba-f772-44f5-bfbc-19e800a842cc_1732x570.png)

Value model versus reward model

**Critic training.** The value function is on-policy—_it is dependent upon the current parameters of our policy_. Unlike [reward models](https://cameronrwolfe.substack.com/p/reward-models) which are fixed at the beginning of RL training, the critic is trained alongside the LLM in each policy update to ensure that its predictions remain on-policy—_this is called an actor-critic setup_[6](https://cameronrwolfe.substack.com/p/ppo-llm#footnote-6-175107358). This is accomplished by adding an extra [mean-squared error (MSE) loss](https://en.wikipedia.org/wiki/Mean_squared_error)—_between the rewards predicted by the critic and actual rewards_—to the surrogate loss.

**PPO implementation.** To make each of these ideas more complete, we have implemented PPO in PyTorch pseudocode below. In this implementation, we see several of the key ideas we have discussed so far, such as:

- Computing the KL divergence between the current policy and a reference model, then directly subtracting this KL divergence from our reward.
    
- Using a learned critic to compute the advantage (and training this critic via an MSE loss alongside the policy itself).
    
- Computing the policy ratio with respect to the old model. The script below performs a single policy update, but PPO usually performs several (i.e., 2-4 in the case of LLMs [2]) policy updates for each batch of data. The “old” model in the policy ratio is the model from before the first update for a batch.
    
- Computing the full (clipped) PPO loss. We take the negative of this loss because PyTorch performs gradient descent (not ascent) by default.
    
- Aggregating or averaging the token-level PPO loss across a batch of sequences. There are many ways to aggregate the loss in a batch, and the approach used can significantly impact results [2][7](https://cameronrwolfe.substack.com/p/ppo-llm#footnote-7-175107358).
    

One interesting detail we see here is that—_despite the PPO loss using token probabilities and not log probabilities_—we choose to work with token log probabilities and exponentiate them instead of using raw probabilities when computing the policy ratio. This is a commonly-used numerical stability trick.

```
import torch
import torch.nn.functional as F

# constants
kl_beta = 0.1
critic_weight = 0.5
ppo_eps = 0.2

# sample prompt completions and rewards
with torch.no_grad():
    completions = LLM.generate(prompts)  # (B*G, L)
    rewards = RM(completions)  # (B*G, 1)

# create a padding mask from lengths of completions in batch
completion_mask = <... mask out padding tokens ...>

# compute value function / critic output
values = CRITIC(completions)  # (B*G, L) - predicted reward per token!

# get policy logprobs for each action
llm_out = LLM(completions)
per_token_logps = F.log_softmax(llm_out, dim=-1)  # (B*G, L)

# get reference logprobs for each action
ref_out = REF(completions)
ref_per_token_logps = F.log_softmax(ref_out, dim=-1)  # (B*G, L)

# compute KL divergence between policy and reference policy
kl_div = per_token_logps - ref_per_token_logps

# directly subtract KL divergence from rewards
# NOTE: KL div is per token, so reward becomes per token and reward
# for all tokens (besides last token) is just kl divergence.
# Reward for last token is sum of outcome reward and KL div.
rewards -= kl_beta * kl_div # (B*G, L)

# compute the advantage - simple approach
advantage = rewards - values.detach()  # (B*G, L)

# compute the policy ratio
# NOTE: old_per_token_logps must be persisted during first policy
# update for this batch of data and re-used in each subsequent update
policy_ratio = torch.exp(
    per_token_logps - old_per_token_logps,
)  # (B*G, L)
clip_policy_ratio = torch.clamp(
    policy_ratio,
    min=1.0 - ppo_eps,
    max=1.0 + ppo_eps,
)

# compute the ppo loss
ppo_loss = torch.min(
    advantage * policy_ratio,
    advantage * clip_policy_ratio,
)  # (B*G, L)
ppo_loss = -ppo_loss

# combine ppo loss and critic mse loss
critic_loss = ((rewards - values) ** 2)  # (B*G, L)
loss = ppo_loss + critic_weight * critic_loss

# aggregate the loss across tokens (many options exist here)
loss = ((loss * completion_mask).sum(axis=-1) /
        completion_mask.sum(axis=-1)).mean()

# perform policy gradient update
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

**Experiments.** The LLM setting is not considered in [1], as PPO was proposed during the heyday of [DeepRL](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html)—_well before the proliferation of LLMs_. Understanding the experimental results in [1] is nonetheless useful for gaining intuition on the mechanics of PPO. In these experiments, PPO is used to train fully-connected [multi-layer perceptrons](https://en.wikipedia.org/wiki/Multilayer_perceptron) (MLPs) from scratch on a variety of robotics and video game tasks. The policy and critic are kept separate (i.e., no parameter sharing).

First, authors use several simulated robotics tasks from the [OpenAI Gym](https://github.com/Farama-Foundation/Gymnasium) to test different formulations of the surrogate loss in PPO:

- The clipped objective (standard for PPO).
    
- The unclipped objective.
    
- The unclipped objective with (adaptive[8](https://cameronrwolfe.substack.com/p/ppo-llm#footnote-8-175107358)) KL divergence.
    

Unlike the typical RL training setup for LLMs, these experiments compute the KL divergence between the current policy and the old model, with the goal of testing whether this approach works better than the standard PPO clipping mechanism. Ordinarily, when training LLMs with PPO, the KL divergence is computed between the current policy and a reference model (e.g., the SFT model), not the old model[9](https://cameronrwolfe.substack.com/p/ppo-llm#footnote-9-175107358). However, in these experiments, using a reference model for the KL divergence is not possible because we are training models from scratch—_there is no pretrained model to serve as a reference_.

The results from testing these different objectives are outlined below—_the clipped objective for PPO stabilizes training and clearly outperforms the other options_.

[

![](https://substackcdn.com/image/fetch/$s_!CHQh!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fa1cc9a21-11e9-4c34-8d72-0576cde83e94_2086x894.png)



](https://substackcdn.com/image/fetch/$s_!CHQh!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fa1cc9a21-11e9-4c34-8d72-0576cde83e94_2086x894.png)

(from [1])

PPO is also tested on 49 games in the [Atari gameplay domain](https://arxiv.org/abs/1207.4708) and compared to strong baseline RL algorithms like [A2C](https://arxiv.org/abs/1602.01783) and [ACER](https://arxiv.org/abs/1611.01224). Performance is measured based on two metrics:

1. Average reward throughout training (favors faster learning).
    
2. Average reward over the last 100 training steps (favors final quality / reward).
    

For each of these metrics, we compute a “win rate”, which captures the number of times each algorithm achieves the top score across all Atari games. The results of these experiments are shown below, where we see that baseline algorithms like ACER perform similarly to or better than PPO but learn much slower. _PPO stabilizes training, performs well, and yields an improvement in sample complexity_[10](https://cameronrwolfe.substack.com/p/ppo-llm#footnote-10-175107358).

[

![](https://substackcdn.com/image/fetch/$s_!SgN4!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fc79fdf5d-6d9e-4f9c-b87e-885fe063de66_1814x499.png)



](https://substackcdn.com/image/fetch/$s_!SgN4!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fc79fdf5d-6d9e-4f9c-b87e-885fe063de66_1814x499.png)

(from [1])

#### [Generalized Advantage Estimation (GAE)](https://arxiv.org/abs/1506.02438) [3]

The advantage tells us how much better a given action is compared to the average action in a given state: `A(s_t, a_t) = Q(s_t, a_t) - V(s_t)`. The value function in this formulation is estimated by our critic, but we have not yet discussed in detail how the advantage function can be computed. In PPO, the advantage function is estimated on a per-token (or action) basis. There are two main approaches that can be used to compute the advantage, and these approaches form the basis for most other techniques.

**(1) Monte Carlo (MC).** An MC estimate of the advantage relies upon the actual reward observed for the full trajectory. Namely, the advantage is computed as the difference between the cumulative reward for the full trajectory `R(s_t)`[11](https://cameronrwolfe.substack.com/p/ppo-llm#footnote-11-175107358) and the value function for the current state `V(s_t)`, as predicted by the critic.

So far, our discussions of PPO have assumed an MC approach for estimating the advantage. The MC estimate has low bias because it relies on the actual reward observed for the trajectory (exact information), but MC estimates also have high variance. Therefore, we need to take many samples and make a sufficient number of observations to yield an accurate advantage estimate—_this can be expensive_.

**(2) Temporal Difference (TD).** The TD residual uses per-token value predictions from the critic to form a one-step estimate of the advantage, as shown below.

[

![](https://substackcdn.com/image/fetch/$s_!A4K-!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F4c1e98c7-da70-4da6-a365-3b2fe9cd2230_1723x896.png)



](https://substackcdn.com/image/fetch/$s_!A4K-!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F4c1e98c7-da70-4da6-a365-3b2fe9cd2230_1723x896.png)

Temporal difference (TD) residual

This TD residual analyzes how much the expected reward changes after predicting a single token and observing the actual reward for that action[12](https://cameronrwolfe.substack.com/p/ppo-llm#footnote-12-175107358). We subtract the value for the current state `V(s_t)` from the sum of:

1. The observed reward for the current state `r_t`.
    
2. The (discounted) value of the next state `V(s_{t+1})`.
    

Similarly to `V(s_t)`, the sum of these two terms captures the expected return at state `s_t`. However, the reward for the current state is captured via the actual observed reward `r_t` rather than being estimated by the critic. Therefore, the difference between these terms is capturing how much better the actual reward observed at state `s_t` is than expected—_this is the advantage_!

By using the actual reward `r_t`, we incorporate some exact information into our advantage estimate—_the terms in the estimate come partly from our critic and partly from real rewards_. Using such token-level rewards to estimate the advantage lowers the variance of the policy gradient. If our value function were exact, then the TD residual would also form an unbiased advantage estimate. Unfortunately, we do not have access to the ground truth value function, so we train a critic to estimate the value function[13](https://cameronrwolfe.substack.com/p/ppo-llm#footnote-13-175107358). Because accurately anticipating final rewards from a partial response is difficult, _the TD residual is biased._

**N-step estimators.** The TD residual analyzes the difference between actual and expected reward for a single step. However, we can generalize this idea to capture any number of steps. As shown below, an `N`-step advantage estimator has a similar structure to the TD residual, but it incorporates real rewards for `N` states, where `N` can be greater than one.

[

![](https://substackcdn.com/image/fetch/$s_!_U8s!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F18ae75ed-997b-4654-b383-dda56a8d9b2e_2298x716.png)



](https://substackcdn.com/image/fetch/$s_!_U8s!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F18ae75ed-997b-4654-b383-dda56a8d9b2e_2298x716.png)

`N`-step advantage estimators

Similarly to the single-step TD residual, advantage estimators with lower values of `N` have low variance but high bias. As we increase the value of `N`, however, we are incorporating more exact reward information into the advantage estimate, thus lowering the bias (and, in turn, increasing variance).

Taking this further, we can even recover an MC estimate by setting `N` equal to the total number of steps in the trajectory! This setting of `N` simply yields the difference between cumulative reward and the value of the current state `V(s_t)`. Therefore, different settings of `N` yield different tradeoffs in bias and variance, spanning all the way from the single-step TD residual (high bias, low variance) to an MC estimate (high variance, low bias).

_“GAE is an alternate method to compute the advantage for policy gradient algorithms that better balances the bias-variance tradeoff. Traditional single-step advantage estimates can introduce too much bias, while using complete trajectories often suffer from high variance. GAE works by combining two ideas – multi-step prediction and weighted running average (or just one of these).” - from [2]_

**Generalized Advantage Estimation (GAE)**, which is the most commonly-used approach for estimating the advantage with PPO, makes use of `N`-step advantage estimates. Instead of choosing a single value of `N`, however, GAE uses all values of `N` by taking an average of `N`-step advantage estimates with different values of `N`. This is done by introducing a mixing parameter `λ` for GAE as shown below[14](https://cameronrwolfe.substack.com/p/ppo-llm#footnote-14-175107358).

[

![](https://substackcdn.com/image/fetch/$s_!v3wn!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Ff11ed641-c3be-442a-ad17-b41072a721a8_2015x843.png)



](https://substackcdn.com/image/fetch/$s_!v3wn!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Ff11ed641-c3be-442a-ad17-b41072a721a8_2015x843.png)

GAE formulation

In this formulation, setting `λ = 0` yields a single-step TD residual because only the first term in the sum receives a non-zero weight. Additionally, a setting of `λ = 1` recovers the MC estimate. To see this, we can expand the definition of each TD residual in the sum, yielding the difference in cumulative discounted rewards and the value function of the current state `V(s_t)`; see below.

[

![](https://substackcdn.com/image/fetch/$s_!DRfY!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Ffdc295ca-a904-4885-85b2-59968c744cc0_2872x674.png)



](https://substackcdn.com/image/fetch/$s_!DRfY!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Ffdc295ca-a904-4885-85b2-59968c744cc0_2872x674.png)

The benefit of GAE is that the value of `λ ∈ [0, 1]` controls the bias variance tradeoff. As we increase the value of `λ`, more exact reward information is used in the advantage estimate, thus lowering the bias (but increasing variance). Similarly, we can use lower values of `λ` to reduce variance at the cost of higher bias.

**Outcome rewards.** When we are working with LLMs, we usually use an outcome reward setup, which simplifies GAE. The reward is always zero[15](https://cameronrwolfe.substack.com/p/ppo-llm#footnote-15-175107358), unless we are at the final step of the trajectory. In this scenario, most of the TD residual terms in our GAE summation are simply the difference in (discounted) value functions between two time steps `γV(s_{t + 1}) - V(s_t)`. The final term in the summation contains the actual outcome reward observed for the trajectory.

**GAE implementation.** To make the concept of GAE more concrete, let’s examine a real-world example adapted from AI2’s [OpenInstruct](https://github.com/allenai/open-instruct) library. The full PPO training script, available [here](https://github.com/allenai/open-instruct/blob/main/open_instruct/ppo2.py), is a great resource for learning the details of PPO in a production-grade training setting. The GAE component of this script is shown below with some additional comments for clarity. We can efficiently compute the GAE recursion by iterating through the sequence in reverse order.

```
import torch

# store advantages in reverse order while iterating thru sequence
advantages_reversed = []

# iterate backward to compute GAE recursion
lastgaelam = 0
gen_length = responses.shape[1]
for t in reversed(range(gen_length)):
    if t < gen_length - 1:
        # get value model prediction for time t + 1
        nextvalues = values[:, t + 1]
    else:
        # no values predicted beyond end of sequence
        nextvalues = 0.0

    # compute TD residual at time t    
    delta = rewards[:, t] + gamma * nextvalues - values[:, t]

    # add to the discounted sum of TD residuals for GAE    
    lastgaelam = delta + gamma * lam * lastgaelam

    # store the advantage for step t in our list
    advantages_reversed.append(lastgaelam)

# put the list of advantages in the correct order
advantages = torch.stack(advantages_reversed[::-1], axis=1)
```

## Using PPO for LLMs

[

![](https://substackcdn.com/image/fetch/$s_!CJn6!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fc0fd3791-df29-4a92-b185-21f6be4f2ddc_2176x642.png)



](https://substackcdn.com/image/fetch/$s_!CJn6!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fc0fd3791-df29-4a92-b185-21f6be4f2ddc_2176x642.png)

(from [7])

There are two different types of RL training that are commonly used to train LLMs (shown above):

- _[Reinforcement Learning from Human Feedback (RLHF)](https://cameronrwolfe.substack.com/p/the-story-of-rlhf-origins-motivations)_ trains the LLM using RL with rewards derived from a human preference [reward model](https://cameronrwolfe.substack.com/p/reward-models).
    
- _[Reinforcement Learning with Verifiable Rewards (RLVR)](https://cameronrwolfe.substack.com/i/153722335/reinforcement-learning-with-verifiable-rewards)_ trains the LLM using RL with rewards derived from rules-based or deterministic verifiers.
    

These RL training techniques differ mainly in how they derive the reward for training, but other details of the algorithms are mostly similar. As depicted below, they both operate by generating completions over a set of prompts, computing the reward for these completions, and using the rewards to derive a [policy update](https://cameronrwolfe.substack.com/p/policy-gradients-the-foundation-of)—_or an update to the LLM’s parameters_—with an RL optimizer (e.g., PPO).

[

![[animate output image]](https://substackcdn.com/image/fetch/$s_!uPv8!,w_1456,c_limit,f_auto,q_auto:good,fl_lossy/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F56eba05c-359c-400d-920f-38a36dd4690a_1920x1078.gif "[animate output image]")



](https://substackcdn.com/image/fetch/$s_!uPv8!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F56eba05c-359c-400d-920f-38a36dd4690a_1920x1078.gif)

Visual walkthrough of RL training for LLMs

RLHF was the original form of RL explored by LLMs like InstructGPT [8], the predecessor to ChatGPT. Early research on RLHF for LLMs used PPO as the default RL optimizer, which ultimately made PPO a standard choice for training LLMs with RL. RLVR was introduced [more recently](https://cameronrwolfe.substack.com/p/demystifying-reasoning-models), and most works in this space use [GRPO](https://arxiv.org/abs/2402.03300) as the underlying RL optimizer instead of PPO.

> _“PPO has been positioned as the canonical method for RLHF. However, it involves both high computational cost and sensitive hyperparameter tuning.”_ - from [9]

**Downsides of PPO.** Though it quickly became the default RL optimizer for RLHF, PPO is a complex actor-critic algorithm with high compute and memory overhead, as well as many low-level implementation complexities. The memory overhead of PPO is high because we keep four copies of the LLM in memory:

1. The policy.
    
2. The reference policy.
    
3. The critic.
    
4. The reward model (if we are using a reward model).
    

Additionally, we are updating the parameters of our critic alongside the policy itself and running inference for all of these models simultaneously, leading to high compute costs. Beyond memory and compute overhead, there are also many implementation details that we must carefully consider during PPO training:

- How do we initialize the critic and reward model? What training settings should we adopt for these models?
    
- What value of `ε` should we use for clipping in PPO?
    
- Which model should we use as our reference model for the KL divergence?
    
- How many policy updates should we perform for a batch of data?
    
- Do we add the KL divergence as a penalty to the loss or directly incorporate it into the reward function? What scaling factor `β` should we use?
    
- How should we weight the critic’s loss relative to the main PPO loss?
    
- Should we use GAE? What setting should we use for `λ`?
    

Each of these choices may impact the results of RL training! PPO is a sensitive algorithm that is prone to instability—_we may spend a lot of compute and time on training a model that ultimately performs poorly due to an incorrect hyperparameter setting_. For these reasons, simpler RL algorithms like [REINFORCE](https://cameronrwolfe.substack.com/p/reinforce) and [GRPO](https://arxiv.org/abs/2402.03300)—_or even RL-free techniques like [DPO](https://cameronrwolfe.substack.com/p/direct-preference-optimization)_—have become popular alternatives to PPO.

**PPO for LLMs.** In this final section, we will take what we have learned and study PPO specifically in the context of LLM training. We will focus particularly on the foundational works that were the first to use PPO for training LLMs [5, 8]—_this research laid the groundwork for the modern LLM boom shortly after_. While studying these papers, we will emphasize implementation details and practical lessons that are necessary to obtain a working PPO implementation.

#### **[Learning to Summarize from Human Feedback](https://arxiv.org/abs/2009.01325) [5]**

Abstractive summarization—_or using models to create a human-readable, concise summary of a piece of text—_has been studied for a long time. Prior to the rise of LLMs and RLHF, most papers on this topic trained language models using a [supervised learning](https://cameronrwolfe.substack.com/p/understanding-and-using-supervised) approach with human-written reference summaries and evaluated these models using traditional metrics like the [ROUGE score](https://cameronrwolfe.substack.com/i/138218863/evaluating-language-models-and-the-rouge-score).

These approaches can work well, but supervised learning and ROUGE are both proxies for what is actually desired—_a model that writes high-quality summaries_. In [5], authors solve this problem by replacing supervised learning with RLHF. Such an approach allows us to finetune language models to produce better summaries by directly using human feedback on model outputs as a training signal.

**PPO for summarization.** Authors in [5] are commonly credited with proposing the first RLHF framework for LLM finetuning. The proposed approach allows us to optimize an LLM based on the quality of its responses, as assessed by human annotators. Beginning with a pretrained LLM, we can iteratively:

1. Collect human [preference data](https://cameronrwolfe.substack.com/i/166169560/the-bradley-terry-model-of-preference).
    
2. Train a [reward model](https://cameronrwolfe.substack.com/p/reward-models) over this preference data.
    
3. Finetune our LLM with RL using this reward model.
    

Notably, authors in [5] adopt PPO as their underlying RL optimizer, which led PPO to become the common choice in subsequent RLHF research. With this RL training strategy, we can train an LLM to produce summaries that surpass the quality of human summaries and are even better than those produced by larger LLMs trained with a supervised learning approach; see below.

[

![](https://substackcdn.com/image/fetch/$s_!bjdU!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F377524f4-cff7-44f9-b717-ed1e842b50bb_1612x970.png)



](https://substackcdn.com/image/fetch/$s_!bjdU!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F377524f4-cff7-44f9-b717-ed1e842b50bb_1612x970.png)

(from [5])

**SFT stage.** In [5], the LLM is first trained using [supervised finetuning](https://cameronrwolfe.substack.com/p/understanding-and-using-supervised) over human reference summaries for a single epoch, producing a supervised baseline that is later finetuned via RLHF. The methodology for RLHF proposed in [5]—_as illustrated in the figure shown below_—is tailored to the summarization task.

[

![](https://substackcdn.com/image/fetch/$s_!oeIY!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fc713702e-ca1c-4759-bff4-b1dedfdf1bbf_1650x1016.png)



](https://substackcdn.com/image/fetch/$s_!oeIY!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fc713702e-ca1c-4759-bff4-b1dedfdf1bbf_1650x1016.png)

(from [5])

**Preferences and reward models.** In [5], a preference dataset is constructed by:

- Grabbing a textual input to summarize—_this is our prompt_.
    
- Producing many summaries of the input using several different policies—_these are different responses to the same prompt_.
    
- Sampling two summaries or responses for the prompt.
    
- Asking a human annotator to identify the better of the two summaries.
    

Authors in [5] collect this preference data in large batches. Once we have finished collecting a new batch of preference data, we train a reward model on the data such that it accurately predicts human preference scores given an LLM-generated summary. Then, we use this reward model to finetune our policy with PPO.

**A** **KL divergence** term is used for PPO in [5] to minimize divergence from the SFT model. Interestingly, authors in [5] were not the first to use this strategy—_it was actually adopted from [prior work](https://arxiv.org/abs/1907.00456)._ The KL divergence is directly subtracted from the rewards instead of being added to the PPO loss as a penalty term. We see in [5] that adding the KL divergence into RL training helps to prevent the model’s summaries from becoming too different from those seen during training.

[

![](https://substackcdn.com/image/fetch/$s_!ZjlA!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fc088796c-52eb-45e5-afbc-195116ec5d1f_1612x764.png)



](https://substackcdn.com/image/fetch/$s_!ZjlA!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fc088796c-52eb-45e5-afbc-195116ec5d1f_1612x764.png)

(from [5])

**Experiments.** In [5], large pretrained models matching the style of GPT-3 with 1.3B to 6.7B parameters are finetuned over the [TL;DR dataset](https://huggingface.co/datasets/openai/summarize_from_feedback). This dataset, which contains over three million posts from Reddit with author-written summaries, is filtered to only 120K high-quality examples; see above. Models are first trained using SFT—_these supervised models are also used as baselines across experiments_—and then further finetuned with RLHF. Given that summary length can impact the resulting quality score, the authors in [5] constrain generated summaries to 48 tokens and finetune the model accordingly.

Finetuning language models with human feedback outperforms a variety of strong English summarization baselines. Notably, the 1.3B summarization model outperforms a 10× larger model trained with SFT, and the 6.7B summarization model performs even better than the 1.3B model, revealing that summarization quality improves with model scale. Furthermore, we see that summarization models trained via RLHF generalize better to new domains. In particular, the models in [5] are applied to summarizing news articles—_a domain outside of the training data_—and found to perform well without further finetuning; see below.

[

![](https://substackcdn.com/image/fetch/$s_!HYOl!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fda0d4ac2-cee0-464b-ba5d-3b278f1b1b9c_1628x846.png)



](https://substackcdn.com/image/fetch/$s_!HYOl!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fda0d4ac2-cee0-464b-ba5d-3b278f1b1b9c_1628x846.png)

(from [5])

From here, summarization models are evaluated in terms of:

- _Coverage_: the summary covers all information from the original post.
    
- _Accuracy_: statements in the summary are accurate.
    
- _Coherence_: the summary is easy to read on its own.
    
- _Quality_: the overall quality of the summary is good.
    

When evaluated in this manner, we see that summarization models trained via RLHF benefit the most in terms of coverage, while coherence and accuracy are only slightly improved compared to supervised baseline models; see below.

[

![](https://substackcdn.com/image/fetch/$s_!d5Qe!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fd1f3213a-8fd2-4703-8987-b2cfcbc5880a_662x672.png)



](https://substackcdn.com/image/fetch/$s_!d5Qe!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fd1f3213a-8fd2-4703-8987-b2cfcbc5880a_662x672.png)

(from [5])

**Beyond summarization.** Although RLHF was explored only in the context of summarization in [5], the authors of this paper had an incredible amount of foresight about what was to come. The approach proposed in [5] later became a standard part of LLM post-training, as we will soon see with InstructGPT [8].

> _“The methods we present in this paper are motivated in part by longer-term concerns about the misalignment of AI systems with what humans want them to do. When misaligned summarization models make up facts, their mistakes are fairly low-risk and easy to spot. However, as AI systems become more powerful and are given increasingly important tasks, the mistakes they make will likely become more subtle and safety-critical, making this an important area for further research.”_ - from [1]

Interestingly, authors in [5] explicitly state their intent to leverage the proposed methodology to better align LLMs to human desires in the long term. This statement was made over two years prior to the proposal of ChatGPT! Work in [5] was a building block for major advancements in AI that were yet to come.

#### **[The N+ Implementation Details of RLHF with PPO](https://arxiv.org/abs/2403.17031) [4]**

[

![](https://substackcdn.com/image/fetch/$s_!Om25!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fbdf3dce4-738f-47c5-a5e3-f12c75887538_1864x1216.png)



](https://substackcdn.com/image/fetch/$s_!Om25!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fbdf3dce4-738f-47c5-a5e3-f12c75887538_1864x1216.png)

(from [4])

There are many moving parts in PPO training, including multiple copies of the LLM (i.e., policy, reference, critic, and reward model) and various hyperparameter settings that must be carefully tuned to ensure stable training. For these reasons—_and due to computational expense_—reproducing RL training results is difficult.

> _“It has proven challenging to reproduce OpenAI’s RLHF pipeline… for several reasons: 1) RL and RLHF have many subtle implementation details that can significantly impact training stability, 2) the models are challenging to evaluate… 3) they take a long time to train and iterate.”_ - from [4]

As a starting point for democratizing understanding of RL, authors in [4] focus on a simple setup—_OpenAI’s prior work on RLHF for summarization_ [5]. Though many details are already provided in the original work, authors in [4] fully reproduce these results while enumerating all implementation details needed to arrive at a working PPO implementation. The TL;DR summarization task is simple relative to most modern RLHF pipelines. However, this study—_based on Pythia models [10] with 1B, 2.8B, and 6.8B parameters_—provides a clear and comprehensive view of key practical considerations when training an LLM with PPO.

**Dataset considerations.** Authors in [4] enumerate around 20 practical details needed to obtain a working RLHF pipeline with PPO. Nearly half of these details are not related to PPO—_they focus on the training data_. For those who have worked with LLMs, this data emphasis should not come as a surprise: _data quality is the key determinant of success in all forms of LLM training, including RL_.

All experiments in [4] use the [TL;DR summarization dataset](https://huggingface.co/datasets/CarperAI/openai_summarize_tldr) from OpenAI, which contains both an SFT and preference dataset. Some notable remarks about the data used for PPO in [4] include:

- There is a misalignment in completion lengths between the SFT and preference portion of the TL;DR dataset—_the preference data tends to have longer completions_.
    
- Data must occasionally be truncated to fit within the fixed sequence length used in [4], but the authors choose to truncate at paragraph boundaries—_determined by newline characters_—instead of performing a hard truncation at the maximum sequence length.
    
- All completions are followed by an `<EOS>` token. Authors in [4] emphasize that this `<EOS>` token must be different than the padding token used by the LLM. Otherwise, the loss for the `<EOS>` token will be masked with the other padding tokens, preventing the model from learning to properly complete each sequence with an `<EOS>` token.
    

**Reward model.** Several choices exist for initializing the reward model in RLHF. In [4], we initialize with the weights of the SFT model, which matches settings used in [5]. A randomly-initialized linear head that is used to predict the reward is then added to the reward model’s architecture before the model is trained for a single epoch over the available preference data.

An outcome reward setting is used in [4]. To extract the reward, a forward pass is performed on the full sequence, and we extract the reward prediction from the `<EOS>` token only. To teach the policy to consistently output sequences of reasonable length with a corresponding `<EOS>` token, the **EOS trick** is used, which assigns a reward of -1 to any sequence with no `<EOS>` token.

> _“If the padding token does not exist, the extracted reward will then be logits corresponding to the last token of the sequence – if that token is not the EOS token, its reward won’t be used for PPO training”_ - from [4]

After the reward model is trained, authors follow the recommendation in [5] of **normalizing rewards** outputted by the model. Specifically, the reward model is used to predict rewards for the entire SFT dataset. Then, we compute the mean reward across this dataset and use this mean to center the average reward. In other words, this mean is subtracted as a bias from the reward model’s output, ensuring that rewards predicted over the SFT dataset have an average of zero. Normalizing the reward model’s output benefits training stability for PPO.

**Critic settings.** We must also choose how to initialize the critic. In [4], the critic is initialized with the weights of the reward model at the beginning of PPO training. After all, _the value model is effectively a reward model that predicts the reward on a per-token basis_. Authors observe in [4] that the reward model’s predictions are usually negative for all tokens except the `<EOS>` token; see below.

[

![](https://substackcdn.com/image/fetch/$s_!fBTb!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fd4cd7447-83f7-4f34-921a-41672d4c391c_1866x536.png)



](https://substackcdn.com/image/fetch/$s_!fBTb!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fd4cd7447-83f7-4f34-921a-41672d4c391c_1866x536.png)

(from [4])

Therefore, the value estimated by the critic is negative for nearly every token at the start of PPO training. However, we see in [4] that warm starting the critic in this way helps to improve the initial stability of gradients during training.

**Reward and advantage whitening.** In addition to normalizing rewards after training the reward model, many PPO implementations perform reward and advantage [whitening](https://joelouismarino.github.io/posts/2017/08/statistical_whitening/). An example implementation of the whitening operation is shown below, where the values can be a list of either rewards or advantages.

[

![](https://substackcdn.com/image/fetch/$s_!XoxA!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F9646db42-a84e-4dca-99a2-e585c053143c_1722x336.png)



](https://substackcdn.com/image/fetch/$s_!XoxA!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F9646db42-a84e-4dca-99a2-e585c053143c_1722x336.png)

(from [4])

When whitening rewards, we usually do not shift the mean (i.e., `shift_mean = False` in the above code) so that we can retain the magnitude and sign of the rewards. However, the mean is usually shifted when whitening advantages. Based on results in [4], _whitening rewards and advantages does not seem to have a huge positive or negative performance impact on the resulting policy_. However, whitening is a common implementation detail in PPO. Usually, whitening is applied over the set of rewards or advantages within a batch of data.

> _“Where normalization bounds all the values from the RM to be between 0 and 1, which can help with learning stability, whitening the rewards or the advantage estimates… can provide an even stronger boost to stability.”_ - from [2]

**Beware of dropout.** We must also be sure to avoid using dropout in PPO. Dropout adds noise to the model’s forward pass, making the computation of policy ratios and KL divergence unreliable. This implementation detail can cause optimization issues and tends to be impactful—_dropout is a perfect example of small but important practical details in PPO_. For example, the [OpenInstruct PPO script](https://github.com/allenai/open-instruct/blob/main/open_instruct/ppo2.py) explicitly disables dropout in the policy, critic, reference, and reward models.

**Final results.** After enumerating various practical choices and hyperparameter settings, the policies in [4] successfully replicate the original results of [5]. PPO models outperform those trained with SFT, and there are clear scaling trends that can be observed (i.e., larger models achieve better performance metrics) for SFT models, reward models, and the final RL policies. Additionally, the preference rate of the RL policies over human reference summaries—_as predicted by a GPT-3.5-based LLM judge_—scales predictably with model size; see below.

[

![](https://substackcdn.com/image/fetch/$s_!y_F0!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F63af44b0-f8ab-4b8a-9872-276a6d78726f_2462x820.png)



](https://substackcdn.com/image/fetch/$s_!y_F0!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F63af44b0-f8ab-4b8a-9872-276a6d78726f_2462x820.png)

(from [4])

#### **[Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) [8]**

Going beyond the summarization domain, authors in [8] explore the use of RLHF for language model [alignment](https://cameronrwolfe.substack.com/p/the-history-of-open-source-llms-imitation) by directly learning from human feedback. The resulting model, called InstructGPT, is the sister model and predecessor to ChatGPT. Since this model is outlined and explained in detail in [8], the work provides significant insight into how early LLMs at OpenAI were trained.

[

![](https://substackcdn.com/image/fetch/$s_!ZdHw!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F45180b88-a11e-42e8-8910-ceca2c3b447a_1618x980.png)



](https://substackcdn.com/image/fetch/$s_!ZdHw!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F45180b88-a11e-42e8-8910-ceca2c3b447a_1618x980.png)

(from [8])

Following an approach similar to [5], we start with a set of prompts that are either written by human annotators or collected from OpenAI’s API. We can then have annotators write responses to these prompts and finetune a pretrained LLM—_[GPT-3](https://cameronrwolfe.substack.com/i/88082618/language-models-are-few-shot-learners) in particular_—over these examples using SFT. Using this model, we can then collect comparison data by asking humans to select their preferred outputs from the LLM and apply the same RLHF process outlined in [5] for finetuning. As shown above, the resulting model is heavily preferred by humans and much better at following detailed instructions provided within the prompt.

> _“Making language models bigger does not inherently make them better at following a user’s intent.”_ - from [8]

**The alignment process.** Pretrained LLMs have a number of undesirable properties that we want to fix during post-training; e.g., hallucinations or an inability to follow detailed instructions. To fix these issues, we align the LLM in [8] according to the following set of criteria:

- _Helpful_: follows the user’s instructions and infers intention from [few-shot prompts](https://cameronrwolfe.substack.com/i/117151147/few-shot-learning) or other patterns.
    
- _Honest_: makes correct factual statements about the world.
    
- _Harmless_: avoids harmful outputs, such as those that denigrate a protected class or contain sexual/violent content.
    

Using RLHF, we can teach an LLM to reflect each of these qualities within its output. Specifically, this is done by constructing preference pairs where the preferred responses are chosen based upon adherence to these criteria.

[

![](https://substackcdn.com/image/fetch/$s_!ddkD!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F7ee233ce-ea11-4928-bcbc-131c5fdc2f2f_1732x930.png)



](https://substackcdn.com/image/fetch/$s_!ddkD!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F7ee233ce-ea11-4928-bcbc-131c5fdc2f2f_1732x930.png)

(from [8])

**More on RLHF.** Authors in [8] curate a team of 40 human annotators, who are screened with a test to judge their annotation quality, to collect preference data for the LLM. The approach for RLHF used in [8] matches the approach used in [5] almost completely. Using a pretrained LLM and a set of prompts for finetuning, the alignment process proceeds according to the following steps:

1. Collect human demonstrations of responses for each prompt.
    
2. Train the model in a supervised fashion over human demonstrations.
    
3. Collect preference data.
    
4. Train a [reward model](https://cameronrwolfe.substack.com/p/reward-models).
    
5. Optimize the underlying LLM or policy with PPO.
    
6. Repeat steps 3-5.
    

The distribution of prompts used for finetuning in [8] is outlined in the table below. For SFT, a dataset of over 13K prompt and response pairs is constructed. The reward model is trained over 33K prompts, while a dataset of size 31K is used for finetuning with PPO. Unlike [5], human annotators are shown 4-9 responses to a prompt (i.e., instead of two) when collecting comparison data, allowing them to quickly rank responses and generate larger amounts of comparison data more efficiently. However, _later work on RLHF largely abandoned this approach in favor of binary preferences_. The dataset used in [8] is also 96% English.

[

![](https://substackcdn.com/image/fetch/$s_!xMFU!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Ff9b979ad-bd64-47c4-bfe7-64890b661ba9_1660x724.png)



](https://substackcdn.com/image/fetch/$s_!xMFU!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Ff9b979ad-bd64-47c4-bfe7-64890b661ba9_1660x724.png)

(from [8])

Similarly to [5], a KL divergence term between the policy and the SFT model is directly subtracted from the reward to avoid drifting too far away from the SFT model. Additionally, extra pretraining updates are “mixed in” to the RLHF optimization process, which authors find to help with maintaining the model’s performance across various benchmarks. These pretraining updates, which use a supervised loss, are simply added to the PPO loss used during RL.

> _“We were able to mitigate most of the performance degradations introduced by our fine-tuning. If this was not the case, these performance degradations would constitute an alignment tax—an additional cost for aligning the model.”_ - from [2]

**Experimental findings.** In [8], authors train three models with 1.3B, 6B, and 175B (i.e., same as [GPT-3](https://cameronrwolfe.substack.com/p/language-model-scaling-laws-and-gpt)) parameters. From these experiments, we learn that human annotators prefer InstructGPT outputs over those of GPT-3, even for models with 10× fewer parameters; see below. This result is similar to observations in [5], where finetuning via RLHF enables much smaller models to outperform larger models trained in a supervised manner.

[

![](https://substackcdn.com/image/fetch/$s_!BTzq!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F08415ad7-db55-4f46-8415-2fb3da1c9ab6_1350x1348.png)



](https://substackcdn.com/image/fetch/$s_!BTzq!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F08415ad7-db55-4f46-8415-2fb3da1c9ab6_1350x1348.png)

(from [8])

Notably, outputs from InstructGPT-1.3B are preferred to those of GPT-3, which has 100× more parameters. Additionally, we see that InstructGPT-175B produces outputs that are preferred to GPT-3 85% of the time. Going further, InstructGPT models are found to more reliably follow explicit constraints and instructions provided by a human user within the model’s prompt; see below.

[

![](https://substackcdn.com/image/fetch/$s_!JB4X!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Ffc9280f9-a159-4e81-ab17-86faf28f47ba_1876x882.png)



](https://substackcdn.com/image/fetch/$s_!JB4X!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Ffc9280f9-a159-4e81-ab17-86faf28f47ba_1876x882.png)

(from [8])

Compared to pretrained and supervised models, InstructGPT is also found to be:

- More truthful.
    
- Slightly less toxic.
    
- Generalizable to instructions beyond the training dataset.
    

For example, InstructGPT can answer questions about code and handle prompts written in different languages, despite the finetuning dataset lacking sufficient data within this distribution. Although the model did not receive as much recognition as ChatGPT, InstructGPT was a major step forward in AI that introduced many core concepts used for training modern LLMs.

## Conclusion

PPO is one of the most widely used RL algorithms for LLMs that has—_through its key role in RLHF pipelines_—directly contributed to fundamental advancements in AI. As we learned, research on PPO was an important factor in the creation of models like InstructGPT and ChatGPT. These influential models catalyzed the ongoing boom in LLM research in which we currently find ourselves.

We cannot overstate the impact of PPO on LLM research, and PPO continues to play an important role in LLM post-training pipelines today. However, the barrier to entry for PPO is high due to its memory and compute overhead. Additionally, the results of PPO can vary based on a wide variety of practical implementation details and hyperparameter settings. For these reasons, most research on PPO has been centralized within top frontier labs. Only a small number of groups have sufficient compute resources to empirically tune and obtain a working PPO implementation at scale.

Nonetheless, understanding PPO is essential due to its fundamental role in AI research. The cost and complexity of PPO remains high, but RL researchers have recently expanded and improved upon ideas proposed by PPO. For example, REINFORCE and GRPO are simpler (and more stable) policy gradient algorithms that can be used to train LLMs, which use less memory than PPO by avoiding the critic. A working understanding of PPO makes understanding these new algorithms—_or even developing our own_—much simpler!

#### New to the newsletter?

Hi! I’m [Cameron R. Wolfe](https://cameronrwolfe.me/), Deep Learning Ph.D. and Senior Research Scientist at [Netflix](https://research.netflix.com/research-area/nlp-and-conversations). This is the Deep (Learning) Focus newsletter, where I help readers better understand important topics in AI research. The newsletter will always be free and open to read. If you like the newsletter, please subscribe, consider a paid subscription, share it, or follow me on [X](https://twitter.com/cwolferesearch) and [LinkedIn](https://www.linkedin.com/in/cameron-r-wolfe-ph-d-04744a238/)!

Subscribe

#### Bibliography

[1] Schulman, John, et al. “Proximal policy optimization algorithms.” _arXiv preprint arXiv:1707.06347_ (2017).

[2] Lambert, Nathan. “Reinforcement Learning from Human Feedback.” Online (2025). https://rlhfbook.com

[3] Schulman, John, et al. “High-dimensional continuous control using generalized advantage estimation.” _arXiv preprint arXiv:1506.02438_ (2015).

[4] Huang, Shengyi, et al. “The n+ implementation details of rlhf with ppo: A case study on tl; dr summarization.” _arXiv preprint arXiv:2403.17031_ (2024).

[5] Stiennon, Nisan, et al. “Learning to summarize with human feedback.” _Advances in neural information processing systems_ 33 (2020): 3008-3021.

[6] Schulman, John, et al. “Trust region policy optimization.” _International conference on machine learning_. PMLR, 2015.

[7] Lambert, Nathan, et al. “Tulu 3: Pushing frontiers in open language model post-training.” _arXiv preprint arXiv:2411.15124_ (2024).

[8] Ouyang, Long, et al. “Training language models to follow instructions with human feedback.” _Advances in neural information processing systems_ 35 (2022): 27730-27744.

[9] Ahmadian, Arash, et al. “Back to basics: Revisiting reinforce style optimization for learning from human feedback in llms.” _arXiv preprint arXiv:2402.14740_ (2024).

[10] Biderman, Stella, et al. “Pythia: A suite for analyzing large language models across training and scaling.” _International Conference on Machine Learning_. PMLR, 2023.

[1](https://cameronrwolfe.substack.com/p/ppo-llm#footnote-anchor-1-175107358)

As we can see, the discounted reward has an infinite horizon in this case. In other words, the total number of steps in the trajectory is infinite `T = ∞`. This is known as the infinite-horizon discounted return.

[2](https://cameronrwolfe.substack.com/p/ppo-llm#footnote-anchor-2-175107358)

The VPG was also partially covered in my overview of REINFORCE that was released a few weeks ago; see [here](https://cameronrwolfe.substack.com/p/reinforce).

[3](https://cameronrwolfe.substack.com/p/ppo-llm#footnote-anchor-3-175107358)

Specifically, if we wanted to solve a constrained optimization problem like this with gradient ascent, we would have to use constrained gradient ascent. However, this method requires that we project our solution into the space of valid solutions that satisfy the constraint after every optimization step, which would be computationally intractable for neural network parameters. The KL divergence is a very complex constraint for which to perform this projection!

[4](https://cameronrwolfe.substack.com/p/ppo-llm#footnote-anchor-4-175107358)

More specifically, if the policy ratio is greater than `1 + ε`, we set it equal to `1 + ε`. If the policy ratio is less than `1 - ε`, we set it to `1 - ε`. Otherwise, we keep the value of the policy ratio unchanged.

[5](https://cameronrwolfe.substack.com/p/ppo-llm#footnote-anchor-5-175107358)

The clipped objective will always be less than or equal to the unclipped objective due to the fact that we are taking the minimum of the unclipped and clipped objectives.

[6](https://cameronrwolfe.substack.com/p/ppo-llm#footnote-anchor-6-175107358)

The “actor” refers to the LLM—_or the model that is taking actions_—and the “critic” refers to the value model. The value model is called a critic due to the fact that it is predicting the reward associated with each action (i.e., effectively critiquing the action).

[7](https://cameronrwolfe.substack.com/p/ppo-llm#footnote-anchor-7-175107358)

For more details on loss aggregation in RL, see [this section](https://rlhfbook.com/c/11-policy-gradients.html#loss-aggregation) of the RLHF book, which provides concrete examples of different aggregation strategies and their impact.

[8](https://cameronrwolfe.substack.com/p/ppo-llm#footnote-anchor-8-175107358)

The adaptive KL divergence is explained in Section 4 of [1]. Instead of setting a fixed scaling factor for the KL divergence, authors propose dynamically adjusting this factor throughout training such that the KL divergence stays close to a target KL divergence `d_targ`. Put differently, instead of choosing the scaling factor, _we specify what we want our KL divergence to be and dynamically adjust the scaling factor throughout training to keep the KL divergence in this range_. This approach is not commonly used for recent LLMs, and it is much more common to set a fixed `β` coefficient for the KL divergence.

[9](https://cameronrwolfe.substack.com/p/ppo-llm#footnote-anchor-9-175107358)

The reference and old models are different models in PPO! The reference model is the policy parameters before any RL training is performed. For LLMs, the SFT model is usually the reference model. We usually perform multiple updates over a batch of data in PPO, _and the old model is the model before the first update_. The old model is updated each time a new batch of data is sampled, whereas the reference model is fixed.

[10](https://cameronrwolfe.substack.com/p/ppo-llm#footnote-anchor-10-175107358)

This means that less data is required to achieve a given level of performance (i.e., the learning process is faster).

[11](https://cameronrwolfe.substack.com/p/ppo-llm#footnote-anchor-11-175107358)

Specifically, we would use the cumulative reward after state `s_t`. However, for LLMs this distinction does not usually matter due to the use of outcome rewards.

[12](https://cameronrwolfe.substack.com/p/ppo-llm#footnote-anchor-12-175107358)

In fact, this is where the name for the TD residual comes from. We are computing the difference in value between two time steps.

[13](https://cameronrwolfe.substack.com/p/ppo-llm#footnote-anchor-13-175107358)

The critic is just a model that imperfectly estimates of the value function. The bias in the TD residual comes from the fact that the critic makes mistakes in estimating the value.

[14](https://cameronrwolfe.substack.com/p/ppo-llm#footnote-anchor-14-175107358)

To derive this expression, we begin with the original formula for the GAE showed in the top line, expand the definitions of the `N`-step advantage estimates, rearrange the terms, then use the [geometric series formula](https://en.wikipedia.org/wiki/Geometric_series) to derive the final expression.

[15](https://cameronrwolfe.substack.com/p/ppo-llm#footnote-anchor-15-175107358)

This statement assumes that the KL divergence is added to the loss and not directly incorporated into the reward.
