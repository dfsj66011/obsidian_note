
如何在不涉及 PPO 复杂性的情况下获得在线强化学习的好处...

RL 在 LLM 研究中正发挥着日益重要的作用。最初，RL 被用于通过 RLHF 等方法实现 LLM 的对齐。最近，它已成为训练强大大型推理模型（LRM）的基础。在使用 RL 训练 LLM 时，通常默认采用近端策略优化（PPO）等在线算法。然而，与 SFT 或 DPO 等替代方案相比，这些算法成本高昂且复杂。

- 必须将 LLM 的四个不同副本保存在内存中。
- 在线训练过程难以协调且可能不稳定。
- 有许多训练超参数需要适当调整。

PPO 的复杂性源于需要稳定在线训练过程。该算法诞生于早期研究阶段，当时主要关注从头开始训练神经网络来解决机器人运动控制和 Atari 游戏等任务。而 LLM 的 RL 环境则截然不同——我们是在对已具备强大先验知识的预训练模型进行微调。

> PPO（近端策略优化）已被确立为 RLHF 的标杆方法。然而，该方法不仅计算成本高昂，还需要敏感的超级参数调优。我们认为，推动 PPO 发展的动机原则在 RLHF 实践中并不那么关键，因此主张采用一种计算成本更低却能保持甚至提升性能的方法。

许多从业者因成本和复杂性而在训练 LLM 时避免使用在线强化学习。通过本文概述，我们将了解到在线强化学习其实不必如此困难！由于 LLM 领域的独特属性，我们可以使用更简单的算法——如 REINFORCE 或 REINFORCE 留一法（RLOO）—— 仍能获得与 PPO 相似的性能表现。因此，与其为了避免复杂性而选择无强化学习或离线替代方案，我们完全可以使用那些既能提供在线强化学习优势、又无需过度复杂的算法。

## LLMs RL 基础

我们将从 RL 的基础知识开始讲解。首先，我们会探讨 RL 中常见的问题设置和术语，以及这些形式化概念如何转化为 LLM 领域。在介绍完 RL 的基本原理及其在 LLM 中的应用后，本节的重点将放在策略优化上——通过推导 RL 中常用的标准策略梯度表达式，并概述这些训练算法最基础形式的具体实现方案。

#### RL 的问题设定与术语

在进行 RL 训练时，我们有一个 agent 在某个环境中执行 actions；具体如下。

![|500](https://substackcdn.com/image/fetch/$s_!lQCe!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fd7117e42-c6ab-43c4-8878-5a88cb99c9ae_2203x870.png)

这些行为由策略预测——我们可以将策略视为智能体的大脑——通常采用参数化形式（例如，在训练 LLM 时，策略就是 LLM 本身）。我们的策略可以是确定性的，也可以是随机的，但在本概述中，我们将假设策略是随机的。我们可以用π_θ(a_t | s_t)来表示在给定策略下某个行为的概率。

These actions are predicted by a **policy**—_we can think of the policy as the agent’s brain_—that is usually parameterized (e.g., the policy is the LLM itself in the context of training LLMs). Our policy can either be deterministic or stochastic, but in this overview we will assume the policy is stochastic[1](https://cameronrwolfe.substack.com/p/reinforce#footnote-1-173306894). We can model the probability of a given action under our policy as `π_θ(a_t | s_t)`.

When the policy outputs an action, the **state** of the environment will be updated according to a **transition function**, which is part of the environment. We will denote our transition function as `P(s_t+1 | a_t, s_t)`. However, transition functions are less relevant for LLMs because they are typically a pass-through; i.e., we assume `s_t = {x, a_1, a_2, …, a_t}`, where `x` is the prompt.

Finally, each state visited by the agent receives a **reward** from the environment that may be positive, negative, or zero (i.e., no reward). As shown in the prior figure, our agent acts iteratively and each action (`a_t`), reward (`r_t`), and state (`s_t`) are associated with a time step `t`. Combining these time steps together yields a **trajectory**; see below. Here, we assume that the agent takes a total of `T` steps in the environment for this particular trajectory.

[

![](https://substackcdn.com/image/fetch/$s_!cjh1!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fbee11fdb-dee8-4d4e-8819-b97642a17129_2008x338.png)



](https://substackcdn.com/image/fetch/$s_!cjh1!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fbee11fdb-dee8-4d4e-8819-b97642a17129_2008x338.png)

Using the chain rule of probabilities, we can also compute the probability of a full trajectory by combining the probabilities of:

- Each action `a_t` given by our policy `π_θ(a_t | s_t)`.
    
- Each state `s_t+1` given by the transition function `P(s_t+1 | a_t, s_t)`.
    

The full expression for the probability of a trajectory is provided below.

[

![](https://substackcdn.com/image/fetch/$s_!YCeT!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F52061751-cc8a-4f3e-a889-5d4e542b21bf_2092x770.png)



](https://substackcdn.com/image/fetch/$s_!YCeT!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F52061751-cc8a-4f3e-a889-5d4e542b21bf_2092x770.png)

Computing the probability of a trajectory

**RL objective.** When training a model with RL, our goal is to maximize the cumulative reward over the entire trajectory (i.e., the sum of `r_t`). However, there are a few variations of this objective that commonly appear. Specifically, the reward that we maximize can either be discounted or non-discounted[2](https://cameronrwolfe.substack.com/p/reinforce#footnote-2-173306894); see below. By incorporating a discount factor, we reward our policy for achieving rewards sooner rather than later. In other words, _money now is better than money later_.

[

![](https://substackcdn.com/image/fetch/$s_!8D_n!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fbbfd6da8-2406-4197-b9d0-d3a1ec301b39_1496x876.png)



](https://substackcdn.com/image/fetch/$s_!8D_n!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fbbfd6da8-2406-4197-b9d0-d3a1ec301b39_1496x876.png)

Our objective is usually expressed as an expected cumulative reward, where the [expectation](https://en.wikipedia.org/wiki/Expected_value) is taken over the trajectory. Expanding this expectation yields a weighted sum of rewards for each trajectory—_the weight is just the trajectory’s probability_. We can formulate this in a continuous or discrete manner; see below.

[

![](https://substackcdn.com/image/fetch/$s_!45io!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F523baab0-10b4-438e-85d7-e7c5c0681209_1692x884.png)



](https://substackcdn.com/image/fetch/$s_!45io!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F523baab0-10b4-438e-85d7-e7c5c0681209_1692x884.png)

We want to maximize this objective during training, which can be accomplished via [gradient ascent](https://en.wikipedia.org/wiki/Gradient_descent)[3](https://cameronrwolfe.substack.com/p/reinforce#footnote-3-173306894); see below. Given this setup, the lingering question that we have to answer is: _How do we compute this gradient?_ As we will see, much of the research on RL focuses on answering this question, and many techniques exist.

[

![](https://substackcdn.com/image/fetch/$s_!slrY!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Ff3072897-d905-42be-b385-6186c24ae059_2390x302.png)



](https://substackcdn.com/image/fetch/$s_!slrY!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Ff3072897-d905-42be-b385-6186c24ae059_2390x302.png)

Solving the RL objective with gradient ascent

**State, value, and advantage functions.** Related to RL objective, we can also define the following set of functions:

- _Value Function_ `V(s)`: the expected cumulative reward when you start in state `s` and act according to your current policy `π_θ`.
    
- _Action-Value Function_ `Q(s, a)`: the expected cumulative reward when you start in state `s`, take action `a`, then act according to your policy `π_θ`.
    
- _Advantage Function_ `A(s, a)`: the difference between the action-value and value function; i.e., `A(s, a) = Q(s, a) - V(s)`.
    

Intuitively, the advantage function tells us how useful some action `a` is by taking the difference between the expected reward after taking action `a` in state `s` and the general expected reward from state `s`. The advantage will be positive if the reward from action `a` is higher than expected and vice versa. Advantage functions play a huge role in RL research—_they are used to compute the gradient for our policy_.

> _“Sometimes in RL, we don’t need to describe how good an action is in an absolute sense, but only how much better it is than others on average. That is to say, we want to know the relative advantage of that action. We make this concept precise with the advantage function.**”**_ - [Spinning up in Deep RL](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html)

#### Markov Decision Process (MDP) versus Bandit Formulation

[

![](https://substackcdn.com/image/fetch/$s_!j0Id!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F00d609c2-0276-4102-914e-7de5d6a5326e_1404x1012.png)



](https://substackcdn.com/image/fetch/$s_!j0Id!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F00d609c2-0276-4102-914e-7de5d6a5326e_1404x1012.png)

RL terminology mapping for LLMs

Now that we understand RL basics, we need to map the terminology that we have learned to the setting of LLM training. We can do this as follows (shown above):

- Our **policy** is the LLM itself.
    
- Our **initial state** is the prompt.
    
- The LLM’s output—_either each token or the entire completion_—is an **action**.
    
- Our **state** is the combination of our prompt with the LLM’s output.
    
- The entire completion from the LLM forms a **trajectory**.
    

Notably, there is no transition function in this setup because the transition function is completely deterministic. If we start with a prompt `x` and our LLM predicts tokens `t_1` and `t_2` given this prompt as input, then our updated state simply becomes `s_2 = {x, t_1, t_2}`. In other words, _our state is just the running completion being generated by the LLM for a given prompt_ `x`.

**Markov decision process (MDP) formulation.** For LLMs, there are two key ways in which RL can be formulated that differ in how they model actions. We should recall that an LLM generates output via [next token prediction](https://cameronrwolfe.substack.com/i/136638774/understanding-next-token-prediction); i.e., by generating each token in the output completion sequentially. This autoregressive process is depicted below. As we can see, the next token prediction process maps very easily to an RL setup—_we can just model each token as an individual action_!

[

![](https://substackcdn.com/image/fetch/$s_!QUg4!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F5b1a8412-5cfb-481f-bd50-473f0a6fd9b5_1992x1037.png)



](https://substackcdn.com/image/fetch/$s_!QUg4!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F5b1a8412-5cfb-481f-bd50-473f0a6fd9b5_1992x1037.png)

The approach of modeling each token in the LLM’s output as an individual action is called the [Markov Decision Process (MDP)](https://en.wikipedia.org/wiki/Markov_decision_process) formulation. An MDP is simply a probabilistic framework for modeling decision-making that includes states, actions, transition probabilities and rewards—_this is exactly the setup we have discussed so far for RL_! The MDP formulation used for RL is shown below.

[

![](https://substackcdn.com/image/fetch/$s_!KWz-!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F52f4f8de-4456-4cbd-935c-a945968b704d_1466x916.png)



](https://substackcdn.com/image/fetch/$s_!KWz-!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F52f4f8de-4456-4cbd-935c-a945968b704d_1466x916.png)

When modeling RL as an MDP for LLMs, our initial state is the prompt and our policy acts by predicting individual tokens. Our LLM forms a stochastic policy that predicts a distribution over tokens. During generation, actions are taken by selecting a token from this distribution—_each token is its own action_. After a token is predicted, it is added to the current state and used by the LLM to predict the next token—_this is just autoregressive next token prediction_! Eventually, the LLM predicts a stop token (e.g., `<|end_of_text|>` or `<eos>`) to complete the generation process, thus yielding a complete trajectory.

**Bandit formulation.** In the above depiction of an MDP, we assume that a reward is provided for every time step, but the reward mechanism for an LLM is usually a bit different from this. Most LLMs are trained using outcome supervision[4](https://cameronrwolfe.substack.com/p/reinforce#footnote-4-173306894), meaning that a reward is only assigned after the model has generated a complete response (i.e., after the `<eos>` token has been outputted).

[

![](https://substackcdn.com/image/fetch/$s_!ZCyt!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Ff1f754f7-353f-4d3e-a79f-a85e5decbc73_1868x490.png)



](https://substackcdn.com/image/fetch/$s_!ZCyt!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Ff1f754f7-353f-4d3e-a79f-a85e5decbc73_1868x490.png)

Outcome versus process supervision for LLMs

In an outcome supervision setting, we may begin to question the utility of modeling each token as its own action. _How will we know whether any single action is helpful or not in this scenario?_ As an alternative, we could model the entire response as a single action that receives an outcome reward. This is the key idea behind the bandit formulation for RL training with LLMs; see below.

[

![](https://substackcdn.com/image/fetch/$s_!nAQM!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F3080d828-b154-42d7-9f6f-7ac24b2be0f4_2234x338.png)



](https://substackcdn.com/image/fetch/$s_!nAQM!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F3080d828-b154-42d7-9f6f-7ac24b2be0f4_2234x338.png)

This name comes from the idea of a [contextual bandit](https://en.wikipedia.org/wiki/Multi-armed_bandit) in probability theory. The bandit setup is simple: _our agent chooses an action, receives a reward and the episode ends_. Our complete trajectory is a single action and reward! For LLMs, our action is the full completion generated for a prompt, which receives an outcome reward.

**Which formulation should we use?** In the context of LLMs, we already know how to compute the probability of both individual tokens and the full completion for a prompt. Therefore, we have the ability to model RL using either an MDP or bandit formulation. Given that LLMs usually only receive outcome rewards, however, the bandit formulation—_despite being very simple_—is quite fitting for LLMs. As we will learn, both REINFORCE and RLOO adopt the bandit formulation, while algorithms like PPO use a per-token MDP formulation. In other words, _both RL formulations are viable and used for training LLMs_.

#### RL Training for LLMs

[

![](https://substackcdn.com/image/fetch/$s_!CJn6!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fc0fd3791-df29-4a92-b185-21f6be4f2ddc_2176x642.png)



](https://substackcdn.com/image/fetch/$s_!CJn6!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fc0fd3791-df29-4a92-b185-21f6be4f2ddc_2176x642.png)

Given the terminology and setup explained so far, we can now discuss how RL is actually used to train LLMs. There are two broad categories of RL training that are commonly used for LLMs today:

- _[Reinforcement Learning from Human Feedback (RLHF)](https://cameronrwolfe.substack.com/p/the-story-of-rlhf-origins-motivations)_ trains the LLM using RL with rewards derived from a human preference [reward model](https://cameronrwolfe.substack.com/p/reward-models).
    
- _[Reinforcement Learning with Verifiable Rewards (RLVR)](https://cameronrwolfe.substack.com/i/153722335/reinforcement-learning-with-verifiable-rewards)_ trains the LLM using RL with rewards derived from rules-based or deterministic verifiers.
    

These RL training techniques differ mainly in how they derive the reward for training, but other details of the algorithms are mostly similar. As depicted below, they both operate by generating completions over a set of prompts, computing the reward for these completions, and using the rewards to derive a [policy update](https://cameronrwolfe.substack.com/p/policy-gradients-the-foundation-of)—_or an update to the LLM’s parameters_—with an RL optimizer.

[

![[animate output image]](https://substackcdn.com/image/fetch/$s_!uPv8!,w_1456,c_limit,f_auto,q_auto:good,fl_lossy/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F56eba05c-359c-400d-920f-38a36dd4690a_1920x1078.gif "[animate output image]")



](https://substackcdn.com/image/fetch/$s_!uPv8!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F56eba05c-359c-400d-920f-38a36dd4690a_1920x1078.gif)

Visual depiction of RL for LLMs

The last step of this process is a gradient ascent step on the RL objective, just as we saw before. However, the actual objective used in RL training goes beyond maximizing cumulative reward. We try to maximize the reward while minimizing [KL divergence](https://cameronrwolfe.substack.com/i/167254905/kullback-leibler-kl-divergence) of our policy with respect to a reference policy—_usually an LLM checkpoint from the start of RL training_. We want to maximize reward without making our new model significantly different from the reference; see below.

[

![](https://substackcdn.com/image/fetch/$s_!kyeM!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fc7464e10-d669-4f6b-ab83-f1980b8918d4_2416x436.png)



](https://substackcdn.com/image/fetch/$s_!kyeM!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fc7464e10-d669-4f6b-ab83-f1980b8918d4_2416x436.png)

RL training objective with KL divergence

Computing the gradient of this objective with respect to the policy’s parameters is where most of the complexity lies in understanding RL. In the context of LLMs, we use policy gradient algorithms (e.g., PPO, GRPO, and REINFORCE) to compute this gradient. This overview will primarily focus on REINFORCE and its variants, but to learn how these algorithms work we need to first understand the simplest form of a policy gradient—_the vanilla policy gradient (VPG)_.

#### Deriving the Vanilla Policy Gradient (VPG)

We will cover the full derivation of the vanilla policy gradient (VPG) here for completeness. However, there are many existing overviews that explain VPG very well. A few great resources for further learning are as follows:

- Intro to Policy Optimization from OpenAI [[link](https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html)]
    
- RLHF Book from [Nathan Lambert](https://natolambert.com/) [[link](https://rlhfbook.com/c/11-policy-gradients.html)]
    
- Policy Optimization Algorithms from [Lilian Weng](https://lilianweng.github.io/) [[link](https://lilianweng.github.io/posts/2018-04-08-policy-gradient/)]
    

Additionally, the prior breakdown of VPG and policy optimization from this newsletter is linked below for easy reference. Our discussion in this section will largely be sampled from this more detailed exposition of policy gradients.

[

## Policy Gradients: The Foundation of RLHF

](https://cameronrwolfe.substack.com/p/policy-gradients-the-foundation-of)

[Cameron R. Wolfe, Ph.D.](https://substack.com/profile/29736521-cameron-r-wolfe-phd)

·

2023年10月2日

[![Policy Gradients: The Foundation of RLHF](https://substackcdn.com/image/fetch/$s_!1no5!,w_280,h_280,c_fill,f_auto,q_auto:good,fl_progressive:steep,g_auto/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F65814056-1bac-4066-8da6-4c323e676060_2408x1352.png)](https://cameronrwolfe.substack.com/p/policy-gradients-the-foundation-of)

A deep dive into policy gradients, how they are applied to training neural networks, and their derivation in the simplest-possible form.

[

Read full story

](https://cameronrwolfe.substack.com/p/policy-gradients-the-foundation-of)

**A basic policy gradient.** Our goal in policy optimization is to compute the policy gradient, or the gradient of our RL objective—_here we will assume our objective is cumulative reward_—with respect to the parameters of our policy. As a first step in computing the policy gradient, we can perform the derivation shown below.

[

![](https://substackcdn.com/image/fetch/$s_!GetI!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F1685ea69-1b2c-438c-87ed-dba51c4bee65_2406x1065.png)



](https://substackcdn.com/image/fetch/$s_!GetI!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F1685ea69-1b2c-438c-87ed-dba51c4bee65_2406x1065.png)

([source](https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html))

This derivation starts with the gradient of our RL training objective (cumulative reward) and ends with a basic expression for the policy gradient. To arrive at the policy gradient, we use mostly simple steps like _i)_ the definition of an expectation over a continuous random variable and _ii)_ the [log-derivative trick](https://andrewcharlesjones.github.io/journal/log-derivative.html).

The most complicated step of this derivation is the final step, which transforms the gradient of the log probability of a trajectory into a sum over the gradients of log probabilities of actions. This step uses our prior expression for the probability of a trajectory, converts the product into a sum (i.e., because we are working with [log probabilities](http://cuemath.com/algebra/properties-of-logarithms/)), and observes that the gradients of the initial state probability and transition function with respect to the policy parameters are always zero because neither of these components depend on the policy; see below.

[

![](https://substackcdn.com/image/fetch/$s_!Rkmm!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fb0f526be-55f2-4eae-abd8-fa4382d8335a_1564x432.png)



](https://substackcdn.com/image/fetch/$s_!Rkmm!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fb0f526be-55f2-4eae-abd8-fa4382d8335a_1564x432.png)

([source](https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html))

**Implementing a basic policy gradient.** The basic policy gradient expression that we derived above is actually pretty easy to compute. Specifically, this expression contains two key quantities that we already know how to compute:

- The reward comes directly from a verifier or reward model.
    
- Log probabilities of actions can be computed with our LLM (i.e., these are just the token probabilities from the LLM’s output).
    

To make the process of computing the basic policy gradient more concrete, a step-by-step implementation in PyTorch pseudocode has been provided below.

[

![[animate output image]](https://substackcdn.com/image/fetch/$s_!PYzF!,w_1456,c_limit,f_auto,q_auto:good,fl_lossy/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F3e4bdafe-cd71-48b7-8a10-abdc895432f7_1920x1076.gif "[animate output image]")



](https://substackcdn.com/image/fetch/$s_!PYzF!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F3e4bdafe-cd71-48b7-8a10-abdc895432f7_1920x1076.gif)

The core intuition behind the structure of this basic policy gradient is that we are increasing the probability of actions from trajectories with high rewards.

> _“Taking a step with this gradient pushes up the log-probabilities of each action in proportion to_ `R(𝜏)`_, the sum of all rewards ever obtained.”_ - [Spinning up in Deep RL](https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html)

This form of the policy gradient is simple, but it still appears in practice! For example, Cursor uses this exact expression in their [recent blog on online RL](https://cursor.com/blog/tab-rl). However, the expression in their blog assumes a bandit formulation, which causes the sum in the expression to be removed (i.e., because there is only one action).

[

![](https://substackcdn.com/image/fetch/$s_!yMfv!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F653838ee-1b3c-4740-be48-e51e04192c99_1366x380.png "Screenshot 2025-06-13 at 10.21.06 AM.png")



](https://substackcdn.com/image/fetch/$s_!yMfv!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F653838ee-1b3c-4740-be48-e51e04192c99_1366x380.png)

([source](https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html#baselines-in-policy-gradients))

**Reducing variance.** Our current policy gradient expression is simple, but it suffers from a few notable issues:

- The gradients can have high variance.
    
- There is no protection against large, unstable policy updates.
    

Most subsequent policy gradient algorithms aim to solve these problems by reducing variance of the policy gradient and enforcing a trust region on policy updates—_or, in other words, restricting how much we can change the model in a single update_. To do this, we usually replace the reward term in our policy gradient with a slightly different term; see below for some of the common options.

[

![](https://substackcdn.com/image/fetch/$s_!EZ-T!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fc797ee60-f90a-4c41-ae9b-b9da6d68096f_1804x716.png "Screenshot 2025-06-13 at 11.04.17 AM.png")



](https://substackcdn.com/image/fetch/$s_!EZ-T!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fc797ee60-f90a-4c41-ae9b-b9da6d68096f_1804x716.png)

(from [4])

As we can see, this expression is nearly identical to what we saw before. The only difference is that we have switched `R(𝜏)` with the generic `Ψ_t` term, which can be set equal to a couple of different things. For example, we can:

- Set `Ψ_t = R(𝜏)` to recover our basic policy gradient expression.
    
- Set `Ψ_t` equal to rewards received after time `t` (i.e., the reward-to-go policy gradient) to avoid crediting actions with rewards that came before them.
    
- Set `Ψ_t` to a [baselined](https://cameronrwolfe.substack.com/i/137421286/variants-of-the-basic-policy-gradient) version of the reward.
    
- Set `Ψ_t` equal to the state-action (`Q`) or advantage function (`A`).
    

A full overview of these choices and how they are derived can be found [here](https://cameronrwolfe.substack.com/i/137421286/variants-of-the-basic-policy-gradient). A common theme among these algorithms is the use of baselines, or extra terms—_which must only depend on the state_ `s_t`—that we subtract from the reward as shown below. Baselines serve the purpose of normalizing the reward (or value) for a state and can be shown to reduce the variance of policy gradients[5](https://cameronrwolfe.substack.com/p/reinforce#footnote-5-173306894).

[

![](https://substackcdn.com/image/fetch/$s_!LPFt!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fd202d1cb-9ee8-4540-9360-be8dea93a14b_2046x538.png)



](https://substackcdn.com/image/fetch/$s_!LPFt!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fd202d1cb-9ee8-4540-9360-be8dea93a14b_2046x538.png)

Adding a baseline to rewards in the policy gradient

A common problem with vanilla policy gradient algorithms is the high variance in gradient updates… In order to alleviate this, various techniques are used to normalize the value estimation, called _baselines_. Baselines accomplish this in multiple ways, effectively normalizing by the value of the state relative to the downstream action (e.g. in the case of Advantage, which is the difference between the Q value and the value). The simplest baselines are averages over the batch of rewards or a moving average. - [RLHF book](https://rlhfbook.com/c/11-policy-gradients.html)

Most of the algorithms we will see focus on setting `Ψ_t` equal to the advantage function—_this is known as the vanilla policy gradient (VPG) algorithm_. The advantage function is commonly used because it yields the lowest-variance policy gradient.

[

![](https://substackcdn.com/image/fetch/$s_!1PL6!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F3dbd6ad6-4d9e-4085-b4a7-849b29789350_1662x470.png)



](https://substackcdn.com/image/fetch/$s_!1PL6!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F3dbd6ad6-4d9e-4085-b4a7-849b29789350_1662x470.png)

The vanilla policy gradient

**Actor-critic.** We should recall that the advantage function is the difference between the state-action value function and the value function. In other words, _the VPG algorithm effectively uses the value function as a baseline in the policy gradient_. The value function is on-policy, meaning that it depends on the exact parameters of our policy in the current training iteration. Usually, we estimate the value function with a neural network. For LLMs, the value function is approximated with a separate value head[6](https://cameronrwolfe.substack.com/p/reinforce#footnote-6-173306894) (or model) that is initialized from the weights of the LLM and trained to predict the value function.

The LLM used to estimate the value function is referred to as a value model or critic. The critic predicts the value function—_or the expected reward starting from a given token or state_—for every token within a sequence. During RL training, the critic is actively updated alongside the LLM for each policy update—_this is referred to as an actor-critic setup_[7](https://cameronrwolfe.substack.com/p/reinforce#footnote-7-173306894). Unlike [reward models](https://cameronrwolfe.substack.com/p/reward-models) which are fixed at the beginning of RL training, the critic is dependent upon the current parameters of the policy. Therefore, to remain on-policy and avoid its predictions becoming stale, the critic must be updated along with the LLM itself. PPO is a notable example of a policy gradient algorithm that adopts such an actor-critic setup.

The critic is usually updated using a [mean-squared error (MSE) loss](https://en.wikipedia.org/wiki/Mean_squared_error) between the predicted and actual rewards. A pseudocode implementation of an actor-critic algorithm is provided below. Although this is a common setup, the use of a value model can be quite expensive—_this requires keeping an entire additional copy of the LLM in memory_! In fact, using a critic is part of the reason why PPO has high computational overhead. Next, we will learn about algorithms that adopt simpler and more efficient approaches for estimating the value function.

```
import torch
import torch.nn.functional as F

# sample prompt completions and rewards
with torch.no_grad():
    completions = LLM(prompts)  # (B*G, L)
    rewards = RM(completions)  # (B*G, 1)

# compute value function / critic output
values = CRITIC(completions)  # (B*G, L) - per token!
advantage = rewards - values.detach()

# get logprobs for each action
completion_mask = <... mask out padding tokens ...>
llm_out = LLM(completions)
token_logp = F.log_softmax(llm_out, dim=-1)

# loss includes a weighted combination of the policy gradient
# loss and the MSE loss for the critic
loss = (- token_logp * advantage) * completion_mask
loss += _beta * (0.5 * (values - rewards)**2)

# aggregate the loss (many options exist here)
loss = (loss.sum(axis=-1) /
        completion_mask.sum(axis=-1)).mean()

# gradient update
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

## REINFORCE and RLOO for LLMs

So far, we have learned about basic concepts in policy optimization and RL for LLMs. The basic policy gradient that we derived is easy to compute practically, but such a formulation leads to high-variance policy gradients and unstable training. To reduce variance, we need an RL optimizer that incorporates an advantage estimate into the policy gradient. However, popular algorithms like PPO accomplish this with a complicated actor-critic framework that introduces substantial overhead. Given this added complexity, we might wonder: _Should we just avoid online RL techniques altogether when training LLMs?_

> _“Recent works propose RL-free methods such as DPO or iterative fine-tuning approaches to LLM preference training. However, these works fail to question whether a simpler solution within an RL paradigm exists.”_ - from [3]

Although many [offline and RL-free training alternatives](https://cameronrwolfe.substack.com/p/online-rl) exist, there are also simple online RL algorithms that can be used to train LLMs. In this section, we will learn about REINFORCE and a slightly modified version of this algorithm called REINFORCE leave one out (RLOO). These online RL algorithms eliminate the need for a critic by estimating the value function with the average of rewards observed throughout training. In theory, such an approach yields higher-variance policy gradients compared to actor-critic algorithms like PPO. However, recent research [3, 5] has found that this increase in variance does not impact LLM training, _yielding easy-to-use and highly-performant options for online RL training_.

#### [REward Increment = Nonnegative Factor x Offset Reinforcement x Characteristic Eligibility (REINFORCE)](https://link.springer.com/article/10.1007/BF00992696) [1]

REINFORCE is a particular implementation of VPG that has low overhead, is simple to understand, and tends to be effective for training LLMs. The structure of the policy gradient used by REINFORCE is similar to the baselined policy gradient estimate we covered before. However, REINFORCE specifically uses the average of rewards observed during RL training as a baseline. This average can be computed in a few different ways; e.g., a moving average of rewards throughout training or an average of rewards present in the current batch.

[

![](https://substackcdn.com/image/fetch/$s_!yDWw!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fa8725dec-4bf1-4790-89cd-4287d7fbbf33_1978x448.png)



](https://substackcdn.com/image/fetch/$s_!yDWw!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fa8725dec-4bf1-4790-89cd-4287d7fbbf33_1978x448.png)

The expression for the policy gradient in REINFORCE is shown above. To compute a gradient update over a batch, we perform the following steps:

- Generate a completion for each prompt using the current policy.
    
- Store the log probabilities for the tokens in each completion
    
- Assign a reward to each completion (usually with a [reward model](https://cameronrwolfe.substack.com/p/reward-models)).
    
- Obtain a baseline by taking an average of rewards.
    
- Compute the advantage by subtracting the baseline from the reward.
    
- Compute the sum of log probabilities multiplied by the advantage for each completion, then average over the batch to form a [Monte Carlo](https://en.wikipedia.org/wiki/Monte_Carlo_method) estimate.
    

**What does the acronym mean?** The REINFORCE acronym is composed of three key components:

1. Reward Increment.
    
2. Non-negative factor.
    
3. Offset reinforcement.
    
4. Characteristic eligibility.
    

The first component is simply our update—_or increment_—to the policy’s parameters (i.e, the policy gradient), which is a product of the three other components. The manner in which these components are combined to form the policy gradient is shown below (top term). To clarify the meaning of each term, we also map the components of REINFORCE to the more familiar expression for a policy gradient. As we can see, these are the same terms we have learned about before (e.g., log probabilities, reward, and baseline)! Additionally, REINFORCE includes the learning rate—_a “non-negative factor” because we are performing gradient ascent and trying to maximize rewards_—within its expression.

[

![](https://substackcdn.com/image/fetch/$s_!VIDL!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F8bf32dae-2f08-46f4-ac67-0f9fbafd0a53_1730x916.png)



](https://substackcdn.com/image/fetch/$s_!VIDL!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F8bf32dae-2f08-46f4-ac67-0f9fbafd0a53_1730x916.png)

Mapping REINFORCE components to a familiar policy gradient expression

The term “offset reinforcement” is straightforward to understand. The baseline is directly subtracted from the reward in our policy gradient expression. In other words, the baseline is used to offset the reward, which is the reinforcement signal in RL (i.e., the reward determines whether actions are good or bad). _The baseline is, therefore, an offset to the reinforcement signal_. Unpacking the term “characteristic eligibility” requires a slightly deeper understanding of RL terminology.

> _“Characteristic Eligibility: This is how the learning becomes attributed per token. It can be a general value, per parameter, but is often log probabilities of the policy in modern equations.”_ - [RLHF book](https://rlhfbook.com/c/11-policy-gradients)

“Eligibility” is a jargon term in RL related to the [credit assignment problem](https://courses.csail.mit.edu/6.803/pdf/steps.pdf)—_or the problem of determining which specific actions contributed to the reward received by the policy_. Specifically, eligibility refers to whether a particular action taken by the LLM is actually responsible for a given reward. In the policy gradient expression, credit assignment is handled by the log probabilities of actions under the policy.

**Incorporating KL divergence.** As with most other RL training algorithms, we also incorporate the [Kullback-Leibler (KL) Divergence](https://cameronrwolfe.substack.com/i/167254905/kullback-leibler-kl-divergence) with respect to a reference policy—_usually a prior [SFT](https://cameronrwolfe.substack.com/p/understanding-and-using-supervised)-trained checkpoint of our model_—into REINFORCE. We have several different approaches for [approximating KL divergence](http://joschu.net/blog/kl-approx.html). A common approach is to approximate KL divergence as the difference in log probabilities between the policy and reference policy. Once we’ve made this approximation, the KL divergence is directly incorporated into the reward as shown below.

[

![](https://substackcdn.com/image/fetch/$s_!8KwS!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F193a7b06-64ee-4be1-b8b2-02849548b5bf_2186x566.png)



](https://substackcdn.com/image/fetch/$s_!8KwS!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F193a7b06-64ee-4be1-b8b2-02849548b5bf_2186x566.png)

This approach of subtracting the KL penalty from the reward varies depending on the RL training algorithm or implementation. For ex ample, [GRPO](https://arxiv.org/abs/2402.03300) incorporates the KL divergence into the loss function rather than directly into the reward. Adding the KL divergence into RL regularizes the training process and allows us to ensure that our policy does not deviate significantly from the reference policy.

**Efficiency & overhead.** Compared to algorithms like PPO, REINFORCE has reduced overhead, as it does not require the use of a value (or critic) model to compute the advantage estimate—_the average of rewards is used in place of the critic_. Therefore, there are only three LLM involved in the training process (i.e., policy, reference policy, and reward model), rather than four; see below. The downside of estimating the advantage in this way is higher variance. As we will see, however, the high variance of REINFORCE is not always a problem in the domain of finetuning LLMs—_this simple algorithm is actually quite effective in practice_.

[

![](https://substackcdn.com/image/fetch/$s_!1owh!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F01ef1923-61fa-4a60-8ab3-79aaca2573a4_2146x824.png)



](https://substackcdn.com/image/fetch/$s_!1owh!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F01ef1923-61fa-4a60-8ab3-79aaca2573a4_2146x824.png)

Key models involved in training with REINFORCE

**Modeling full completions.** There is one final detail missing from the image above: _How do we aggregate the log probabilities, KL divergences, and rewards to form the policy gradient update?_ One of the key distinguishing aspects of REINFORCE is that is uses a bandit formulation. The policy is trained by considering the full completion, rather than each token in the completion, as a single action.

> _“[REINFORCE] treats the entire model completion as a single action, whereas regular PPO treats **each completion token** as individual actions. Typically, only the EOS token gets a true reward, which is very sparse. Regular PPO would attribute a reward to the EOS token, whereas [REINFORCE] would attribute that EOS reward to the entire completion.”_ - from [5]

As we’ve learned, most LLMs are trained using an outcome reward setting, meaning that only the final `<eos>` token generated by the LLM is assigned a reward. However, the KL divergence is computed on the per-token basis, and—_as mentioned before_—the KL divergence is directly subtracted from the reward in REINFORCE. Therefore, we end up with a setup where the reward for all tokens in the completion is just the KL divergence, but the final token in the completion receives an additional reward from the reward model; see below.

[

![](https://substackcdn.com/image/fetch/$s_!QIGq!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F92525dd7-7c62-47e9-9131-3522e0d61864_1864x1286.png)



](https://substackcdn.com/image/fetch/$s_!QIGq!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F92525dd7-7c62-47e9-9131-3522e0d61864_1864x1286.png)

Bandit formulation in REINFORCE

We create a completion-level (bandit formulation) reward by summing per-token KL divergences and rewards over the sequence. Similarly, we can sum token-level log probabilities to get the log probability of the completion (or trajectory)[8](https://cameronrwolfe.substack.com/p/reinforce#footnote-8-173306894). As shown above, we can then use these completion-level components to compute the policy gradient similarly to before:

1. Subtract the baseline (average reward) from the completion-level reward.
    
2. Multiply this difference by the completion log probability.
    
3. Run a backward pass to compute the final policy gradient[9](https://cameronrwolfe.substack.com/p/reinforce#footnote-9-173306894).
    

This process computes the policy gradient for a single prompt and completion pair, but we generally average this gradient over a batch of completions.

**Pseudocode.** As a final step, we will make this discussion more concrete by implementing the computation of the policy gradient for REINFORCE in basic PyTorch[10](https://cameronrwolfe.substack.com/p/reinforce#footnote-10-173306894). We will assume that the baseline is computed by taking an average of rewards in the batch (i.e., rather than by using a moving average) so that the entire gradient update can be outlined within a single script; see below.

```
import torch

# constants
kl_beta = 0.1

# batch of two completions with three tokens each
per_token_logprobs = torch.tensor(
    [
        [-12.3, -8.3, -2.3],
        [-10.0, -7.0, -3.0],
    ],
    requires_grad=True,
)
reference_per_token_logprobs = torch.tensor([
    [-11.3, -8.4, -2.0],
    [-9.5, -7.2, -2.8],
])

# compute KL divergence approximation
kl_div = per_token_logprobs - reference_per_token_logprobs
kl_div = -kl_beta * kl_div

# get reward for each completion (e.g., from reward model)
score_from_rm = torch.tensor([1.0, 0.5])

# reward is attributed to final <eos> token
per_token_reward = kl_div.clone()
per_token_reward[range(per_token_reward.size(0)), -1] += score_from_rm

# compute REINFORCE update over full sequence
entire_completion_reward = per_token_reward.sum(dim=1)
baseline = entire_completion_reward.mean().detach()

# compute advantage
advantage = entire_completion_reward - baseline

# compute loss and gradient update
reinforce_loss = -per_token_logprobs.sum(dim=1) * advantage
reinforce_loss.mean().backward()
```

#### [REINFORCE Leave One Out (RLOO)](https://openreview.net/forum?id=r1lgTGL5DE) [2]

In REINFORCE, we generate a single on-policy completion per prompt during training and use the rewards from these completions to form our baseline via a moving average or an average of rewards in the batch. REINFORCE leave-one-out (RLOO) [2] changes this approach by:

1. Sampling multiple (`K`) completions per prompt.
    
2. Using these multiple completions to compute the reward average separately for each individual prompt.
    

Given `K` completions `{y_1, y_2, …, y_K}` to a prompt `x`, RLOO defines the baseline for completion `y_i` as shown below, which is simply an average over all rewards for completions to prompt `x` excluding the completion itself `y_i`. We “leave out” the reward of the completion for which the policy gradient is being computed and average over the rewards of other completions to the same prompt.

[

![](https://substackcdn.com/image/fetch/$s_!QwPe!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F6c5886bd-c7b0-461c-928b-d81e070fa7a2_1362x718.png)



](https://substackcdn.com/image/fetch/$s_!QwPe!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F6c5886bd-c7b0-461c-928b-d81e070fa7a2_1362x718.png)

Computing the baseline for RLOO

From here, we can compute the advantage estimate in RLOO by _i)_ computing this baseline for every completion in the batch and _ii)_ subtracting the baseline from the reward received by the completion; see below (first equation). To efficiently compute the baseline for RLOO, we can first compute a fixed average reward over the `K` completions and reformulate the advantage as in the second equation below. This approach allows us to compute the average reward once and avoid re-computing the leave one out average for all `K` completions to the prompt `x`.

[

![](https://substackcdn.com/image/fetch/$s_!B9wg!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Ff7e7191e-5a8f-4db8-ad1d-da6be1a66c54_1808x996.png)



](https://substackcdn.com/image/fetch/$s_!B9wg!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Ff7e7191e-5a8f-4db8-ad1d-da6be1a66c54_1808x996.png)

Advantage estimate in RLOO

This modified advantage estimate can be plugged into the same policy gradient expression used by REINFORCE. Similarly to REINFORCE, RLOO uses a per-completion—_as opposed to per-token_—loss, and we have no learned value model. However, the leave one out baseline used by RLOO lowers variance relative to the standard REINFORCE algorithm by using multiple samples per prompt to derive the policy gradient estimate. Compared to a single-sample approach, taking multiple samples per prompt benefits training stability, speed, and performance.

> _“The common case of sampling one prediction per datapoint is data-inefficient. We show that by drawing multiple samples per datapoint, we can learn with significantly less data, as we freely obtain a REINFORCE baseline to reduce variance.”_ - from [2]

**Practical usage.** After the popularization of RLOO for LLMs, a great blog on this topic was published by HuggingFace [5] exploring the implementation and practical performance of RLOO. This analysis extends authors’ prior work on correctly implementing and tuning PPO-based RLHF on summarization tasks [6]—_OpenAI’s [TL;DR summarization dataset](https://huggingface.co/datasets/openai/summarize_from_feedback) in particular._ In [5], these results are extended by training Pythia 1B and 6.9B models with RLOO, starting from the same SFT checkpoints and reward models from [6]. Models are evaluated by comparing their output to a reference summary with a GPT-4 judge; see below.

[

![](https://substackcdn.com/image/fetch/$s_!Fkqw!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F43a3270c-bd5c-4a96-9e87-019ea3d70082_1520x560.png)



](https://substackcdn.com/image/fetch/$s_!Fkqw!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F43a3270c-bd5c-4a96-9e87-019ea3d70082_1520x560.png)

(from [5])

As we can see, RLOO uses 50-70% less memory than PPO and runs 2-3× faster. These savings increase with the size of the model. In addition to these gains in efficiency, RLOO performs competitively to PPO and consistently outperforms offline algorithms like DPO. These results demonstrate the key value proposition of RLOO (and REINFORCE)—_these algorithms maintain the performance benefits of online RL algorithms while being simpler to implement and less costly to run_.

**Pseudocode.** To implement RLOO, we can modify our original REINFORCE example as shown below. Here, we assume that three completions are sampled per prompt (i.e., `K = 3`) and that our batch is composed of three prompts. For more production-ready code, both REINFORCE and RLOO are also supported within the volcano engine reinforcement learning (verl) library [7]; see [here](https://github.com/volcengine/verl).

```
import torch

# constants
K = 3  # completions per prompt
kl_beta = 0.1

# batch of three prompts with three completions each
per_token_logprobs = torch.tensor(
    [
        # prompt 1
        [
            [-12.3, -8.3, -2.3], # completion 1
            [-10.0, -7.0, -3.0], # completion 2
            [-10.5, -12.2, -9.1], # completion 3
        ],

        # prompt 2
        [
            [-11.0, -10.3, -1.3],
            [-11.1, -11.1, -0.8],   
            [-8.2, -11.9, -0.1],        

        ],
        
        # prompt 3
        [
            [-1.8, -2.1, -0.2],
            [-0.7, -3.5, -0.1],
            [-1.0, -2.2, -1.1],
        ],
    ],
    requires_grad=True,
)
reference_per_token_logprobs = torch.tensor([
    [
        [-11.8, -8.4, -2.3], 
        [-10.1, -7.2, -3.1],
        [-10.3, -12.9, -9.1],
    ],
    [
        [-11.8, -9.7, -1.3],
        [-12.3, -11.9, -0.2],
        [-8.1, -12.0, -0.5],
    ],
    [
        [-2.7, -2.0, -1.2],
        [-0.7, -3.6, -0.2],
        [-0.7, -1.2, -0.9],
    ],
])

# compute KL divergence approximation
kl_div = per_token_logprobs - reference_per_token_logprobs
kl_div = -kl_beta * kl_div

# reward for each completion (grouped by prompt)
score_from_rm = torch.tensor([
    [1, 2, 3], # rewards for completions to prompt 1
    [2, 3, 4], # rewards for completions to prompt 2
    [3, 4, 5], # rewards for completions to prompt 3
]).float()

# reward attributed to final <eos> token
per_token_reward = kl_div.clone()
per_token_reward[:, :, -1] += score_from_rm

# compute full sequence reward
entire_completion_reward = per_token_reward.sum(dim=-1)

# compute RLOO baseline in vectorized fashion
baseline = (
    entire_completion_reward.sum(dim=-1)[:, None]
    - entire_completion_reward
) / (K - 1)
baseline = baseline.detach()

# compute advantage and loss
advantage = entire_completion_reward - baseline
rloo_loss = -per_token_logprobs.sum(dim=-1) * advantage
rloo_loss.mean().backward()
```

#### **[Back to Basics: Revisiting REINFORCE Style Optimization for Learning from Human Feedback in LLMs](https://arxiv.org/abs/2402.14740) [3]**

> _“We posit that most of the motivational principles that led to the development of PPO are less of a practical concern in RLHF and advocate for a less computationally expensive method that preserves and even increases performance.”_ - from [3]

Although PPO is the de facto RL optimizer for RLHF, authors in [3] argue that the original motivations for PPO (i.e., avoiding large and unstable policy updates) are less relevant in the context of LLMs. Instead, we can use simpler RL optimizers—_REINFORCE in particular_—to save on compute and memory costs without sacrificing performance. In particular, we learn that aligning LLMS with a basic REINFORCE algorithm can achieve results that match or exceed those of PPO-based RLHF, as well as other algorithms like [DPO](https://cameronrwolfe.substack.com/p/direct-preference-optimization) and [RAFT](https://arxiv.org/abs/2304.06767). This paper was a key contribution that popularized the use of simpler RL optimizers for LLMs.

**LLMs versus DeepRL.** The crux of the argument in [3] revolves around the fact that LLM finetuning is a unique setting for RL that differs significantly from the [traditional DeepRL setting](https://spinningup.openai.com/en/latest/) in which algorithms like PPO were proposed. The most notable difference between these two settings is that LLMs are not trained with RL from scratch. Rather, _we are finetuning an LLM that has already undergone extensive pretraining_. This difference has two key implications:

- The risk of policy updates with catastrophically large variance is lower in LLM finetuning relative to the traditional DeepRL setting.
    
- The LLM finetuning setting has less of a need for regularizing the learning process relative to the traditional DeepRL setting.
    

We can concretely test this hypothesis by tweaking the settings of PPO. Namely, most implementations of PPO use [Generalized Advantage Estimation (GAE)](https://danieltakeshi.github.io/2017/04/02/notes-on-the-generalized-advantage-estimation-paper/) [4] to estimate the advantage function. The details of GAE are beyond the scope of this post. However, GAE contains the `λ ∈ [0.0, 1.0]` hyperparameter that can be used to control the tradeoff between bias and variance in the advantage estimate.

[

![](https://substackcdn.com/image/fetch/$s_!TXwn!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F7224cd65-a3fd-4593-8a48-0abcfeed30bf_1266x654.png)



](https://substackcdn.com/image/fetch/$s_!TXwn!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F7224cd65-a3fd-4593-8a48-0abcfeed30bf_1266x654.png)

(from [3])

Lowering `λ` reduces variance at the cost of increased bias, but this is a worthwhile tradeoff for domains—_like DeepRL_—with excessive variance in policy updates. As shown above, optimal performance in LLM alignment is achieved with a setting of `λ = 1.0`, which induces maximum possible variance in the policy gradient. Such a finding indicates that the level of variance in policy updates observed for LLM alignment is not detrimental to the LLM’s learning process.

> _“Large off-policy updates in our optimization regime are rare and do not have catastrophic effects on learning as they do in traditional DeepRL.”_ - from [3]

**Effective action space.** In addition to high variance, one complicating factor of RL training is the presence of a large action space. If there are many possible actions for the policy to take and rewards from these actions are noisy, learning a high-quality policy is difficult. Theoretically, the action space of an LLM is very large—_it includes all completions that the LLM can generate for a given prompt_.

[

![](https://substackcdn.com/image/fetch/$s_!7xYY!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fc836c3b8-7c81-4c3c-bf67-3682aad92a86_1274x748.png)



](https://substackcdn.com/image/fetch/$s_!7xYY!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fc836c3b8-7c81-4c3c-bf67-3682aad92a86_1274x748.png)

(from [3])

Practically speaking, however, the effective action space of an LLM—_the set of completions that the model is likely to generate_—is actually quite small. When an LLM is performing generation, this process is conditioned upon the prompt provided to the LLM, which is shown in [3] to be a strong conditioning.

[

![](https://substackcdn.com/image/fetch/$s_!7xYY!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fc836c3b8-7c81-4c3c-bf67-3682aad92a86_1274x748.png)



](https://substackcdn.com/image/fetch/$s_!7xYY!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fc836c3b8-7c81-4c3c-bf67-3682aad92a86_1274x748.png)

(from [3])

More specifically, we see in the figure above that probability mass in an LLM’s completions is highly-concentrated amongst a small number of tokens after the first step of the generation process (i.e., the first token that is outputted). Such an observation demonstrates that an LLM’s prompt provides strong conditioning for the generation process, _which makes the mode’s effective action space quite small_.

**From PPO to REINFORCE.** Given that variance is less of a concern for LLMs, authors in [3] perform RLHF experiments that use much simpler REINFORCE and RLOO algorithms as the RL optimizer in place of PPO. REINFORCE and RLOO make significant changes to the RL formulation used in PPO. Namely, PPO uses a per-token MDP formulation, while both REINFORCE and RLOO adopt a bandit formulation—_the entire completion is modeled as a single action_.

> _“We show that the modeling of partial sequences is unnecessary in this setting where rewards are only attributed to full generations… it is more appropriate and efficient to model the entire generation as a single action with the initial state determined by the prompt.”_ - from [3]

In addition to being simpler than the MDP formulation, modeling the full generation as a single action preserves the LLM’s performance and even speeds up learning, _indicating that formulating each token as its own action is an unnecessary complexity in an outcome reward setting_.

**Experimental setup.** Experiments in [3] are conducted on the [TL;DR summarize](https://huggingface.co/datasets/CarperAI/openai_summarize_tldr) and [Anthropic HH](https://huggingface.co/datasets/Anthropic/hh-rlhf) datasets with [Pythia-6.9b](https://huggingface.co/EleutherAI/pythia-6.9b) and [Llama-7b](https://huggingface.co/meta-llama/Llama-2-7b) models. Both reward models and policies are initialized using a model checkpoint obtained by running SFT on a curated dataset of high-quality completions for each respective dataset. During RL, training prompts are sampled from the SFT dataset. For evaluation, authors report each model’s average reward—_from the fixed reward model used for RL training_—on a hold out test set, as well as win-rates against GPT-4 using the [AlpacaFarm framework](https://arxiv.org/abs/2305.14387) (i.e., open-ended evaluation on chat-style prompts).

[

![](https://substackcdn.com/image/fetch/$s_!dMtu!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F3d5c4571-e684-4930-ad3f-e82ff9454f76_1582x894.png)



](https://substackcdn.com/image/fetch/$s_!dMtu!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F3d5c4571-e684-4930-ad3f-e82ff9454f76_1582x894.png)

(from [3])

**Is REINFORCE effective?** As shown above, both REINFORCE and RLOO—_in addition to being less memory intensive due to their lack of a learned critic model_—consistently outperform PPO, confirming that modeling partial sequences is unnecessary for the RLHF setting in [3]. RLOO is also found to be more sample efficient than the [RAFT algorithm](https://arxiv.org/abs/2304.06767) [9]—_given the same number of on-policy samples generated during training, RLOO tends to achieve better performance_; see below.

[

![](https://substackcdn.com/image/fetch/$s_!yEoc!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fdead7897-21dd-45ac-a63d-e8e25b922989_1666x744.png)



](https://substackcdn.com/image/fetch/$s_!yEoc!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fdead7897-21dd-45ac-a63d-e8e25b922989_1666x744.png)

(from [4])

This finding holds true for all models and data tested in [3]. The superior sample efficiency of RLOO makes intuitive sense given that all samples—_even those with poor or negative reward_—are used during training. In contrast, RAFT filters samples based on their reward and only trains on those with the best rewards.

When we evaluate models in terms of simulated win-rates on AlpacaFarm, many of the results above continue to be true, but we can compare the performance of each technique in a more human-understandable manner. As shown below, the best performance is consistently achieved with RLOO, and both REINFORCE and RLOO consistently outperform PPO. Notably, RLOO—_with four on-policy samples per prompt_—outperforms PPO by an absolute increase in win-rate of 10.4% and 14.5% for TL;DR and HH datasets. When used to align Llama, RLOO sees an even larger absolute win-rate improvement of 32.1% over PPO.

[

![](https://substackcdn.com/image/fetch/$s_!EFho!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fe75831a0-6467-4f19-9f48-793b78cdf1ee_2472x800.png)



](https://substackcdn.com/image/fetch/$s_!EFho!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fe75831a0-6467-4f19-9f48-793b78cdf1ee_2472x800.png)

(from [3])

**Improved robustness.** Authors in [3] conclude by studying the robustness of RLOO relative to RAFT in two areas:

- How does increasing the β term for KL divergence impact performance?
    
- How does adding noise to the reward estimate impact performance?
    

Interestingly, RLOO is found to be noticeably more robust to noise relative to RAFT; see below. When increasing β, RAFT performs worse than RLOO and produces a policy with a larger KL divergence relative to the reference policy. Additionally, the performance of RAFT sees a larger negative impact from noisy reward estimates relative to RLOO. Such degraded robustness to noise is caused by the fact that RAFT only trains on the highest-reward completions, _leading any perturbation to reward estimates to significantly impact training_.

[

![](https://substackcdn.com/image/fetch/$s_!YWO-!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F74163b82-ab60-4220-a908-1dfa70f2268b_1250x1204.png)



](https://substackcdn.com/image/fetch/$s_!YWO-!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F74163b82-ab60-4220-a908-1dfa70f2268b_1250x1204.png)

(from [3])

## Conclusion

We now have a foundational understanding of RL for LLMs that spans from basic terminology to functional implementations of online RL algorithms. Most work on RL training for LLMs uses actor-critic algorithms like PPO as the underlying optimizer. But, these algorithms introduce complexity and overhead to reduce the variance of policy gradients. In the context of LLMs, we have learned that much simpler online RL algorithms are available! REINFORCE and RLOO adopt a completion-level bandit setup for RL and normalize rewards using either:

- The average of rewards during training (for REINFORCE), or
    
- The average of rewards for other completions to a prompt (for RLOO).
    

Because they estimate the value function in this way, neither REINFORCE or RLOO require a learned critic, which reduces memory overhead and speeds up the training process. If we want to avoid the complexity of algorithms like PPO, these simpler online RL algorithms offer an effective alternative, rather than immediately turning to approaches that are completely offline or RL-free.

#### New to the newsletter?

Hi! I’m [Cameron R. Wolfe](https://cameronrwolfe.me/), Deep Learning Ph.D. and Senior Research Scientist at [Netflix](https://research.netflix.com/research-area/nlp-and-conversations). This is the Deep (Learning) Focus newsletter, where I help readers better understand important topics in AI research. The newsletter will always be free and open to read. If you like the newsletter, please subscribe, consider a paid subscription, share it, or follow me on [X](https://twitter.com/cwolferesearch) and [LinkedIn](https://www.linkedin.com/in/cameron-r-wolfe-ph-d-04744a238/)!

Subscribe

#### Bibliography

[1] Williams, Ronald J. “Simple statistical gradient-following algorithms for connectionist reinforcement learning.” _Machine learning_ 8.3 (1992): 229-256.

[2] Kool, Wouter, Herke van Hoof, and Max Welling. “Buy 4 reinforce samples, get a baseline for free!.” (2019).

[3] Ahmadian, Arash, et al. “Back to basics: Revisiting reinforce style optimization for learning from human feedback in llms.” _arXiv preprint arXiv:2402.14740_ (2024).

[4] Schulman, John, et al. “High-dimensional continuous control using generalized advantage estimation.” _arXiv preprint arXiv:1506.02438_ (2015).

[5] Costa Huang, Shengyi, et al. “Putting RL back in RLHF” [https://huggingface.co/blog/putting_rl_back_in_rlhf_with_rloo](https://huggingface.co/blog/putting_rl_back_in_rlhf_with_rloo) (2024).

[6] Huang, Shengyi, et al. “The n+ implementation details of rlhf with ppo: A case study on tl; dr summarization.” _arXiv preprint arXiv:2403.17031_ (2024).  
[7] Sheng, Guangming, et al. “Hybridflow: A flexible and efficient rlhf framework.” _Proceedings of the Twentieth European Conference on Computer Systems_. 2025.

[8] Lightman, Hunter, et al. “Let’s verify step by step.” _The Twelfth International Conference on Learning Representations_. 2023.

[9] Dong, Hanze, et al. “Raft: Reward ranked finetuning for generative foundation model alignment.” _arXiv preprint arXiv:2304.06767_ (2023).

[1](https://cameronrwolfe.substack.com/p/reinforce#footnote-anchor-1-173306894)

In other words, the output of our policy is not just a discrete action. Rather, it is a probability distribution over a set of possible actions. For example, LLMs output a probability distribution over the set of potential next tokens.

[2](https://cameronrwolfe.substack.com/p/reinforce#footnote-anchor-2-173306894)

Additionally, we can have a finite or infinite-horizon setup in this return. However, in the context of LLMs, we usually assume a finite-horizon setup (i.e., the LLM does not continue generating tokens forever).

[3](https://cameronrwolfe.substack.com/p/reinforce#footnote-anchor-3-173306894)

Here, we use gradient ascent (as opposed to descent) because we are trying to maximize a function. However, gradient ascent and descent are nearly identical. The only change is whether we subtract—_if minimizing a function in gradient descent_—or add—_if maximizing a function in gradient ascent_—the gradient to our model’s parameters.

[4](https://cameronrwolfe.substack.com/p/reinforce#footnote-anchor-4-173306894)

Process supervision is possible and has been explored in research on large reasoning models (LRMs), but it is less common than the outcome reward setting.

[5](https://cameronrwolfe.substack.com/p/reinforce#footnote-anchor-5-173306894)

Additionally, adding baselines to the policy gradient does not bias our gradient estimate. This fact can be proven by using the [EGLP lemma](https://cameronrwolfe.substack.com/i/137421286/variants-of-the-basic-policy-gradient), which also mandates that the baseline must only depend on the state `s_t`.

[6](https://cameronrwolfe.substack.com/p/reinforce#footnote-anchor-6-173306894)

By “head”, we mean an extra small layer added to the end of the LLM that is trainable.

[7](https://cameronrwolfe.substack.com/p/reinforce#footnote-anchor-7-173306894)

The “actor” refers to the LLM—_or the model that is taking actions_—and the “critic” refers to the value model. The value model is called a critic due to the fact that it is predicting the reward associated with each action (i.e., effectively critiquing the action).

[8](https://cameronrwolfe.substack.com/p/reinforce#footnote-anchor-8-173306894)

This stems from basic concepts in language modeling. Namely, we can take the product of probabilities for all tokens in a completion (or the sum of log probabilities) to get the probability of the full completion.

[9](https://cameronrwolfe.substack.com/p/reinforce#footnote-anchor-9-173306894)

Our policy gradient term contains the gradient of log probabilities, but we have access to log probabilities (not the gradient of log probabilities) in our example. So, we need to take the gradient of these log probabilities—_usually by running_ `loss.backward()` _in PyTorch_—to get the final policy gradient.

[10](https://cameronrwolfe.substack.com/p/reinforce#footnote-anchor-10-173306894)

This implementation, as well as our later implementation of RLOO, is just a modified version of the code from [this blog post](https://huggingface.co/blog/putting_rl_back_in_rlhf_with_rloo).

---

#### Subscribe to Deep (Learning) Focus

By Cameron R. Wolfe · Launched 3 years ago

I contextualize and explain important topics in AI research.

Subscribe

By subscribing, I agree to Substack's [Terms of Use](https://substack.com/tos), and acknowledge its [Information Collection Notice](https://substack.com/ccpa#personal-data-collected) and [Privacy Policy](https://substack.com/privacy).

[](https://substack.com/profile/397200345-sai-krishna)

[](https://substack.com/profile/219888043-suvrojyoti-biswas)

[](https://substack.com/profile/100781834-aykut-cayr)

[](https://substack.com/profile/174228865-emmanuel-maminta)

[](https://substack.com/profile/18106732-prarambh)

75 Likes∙

[6 Restacks](https://substack.com/note/p-173306894/restacks?utm_source=substack&utm_content=facepile-restacks)

[](https://cameronrwolfe.substack.com/p/reinforce/comments)

#### Discussion about this post

[](https://substack.com/profile/399762516-anshu?utm_source=comment)

[Anshu](https://substack.com/profile/399762516-anshu?utm_source=substack-feed-item)

[10月6日](https://cameronrwolfe.substack.com/p/reinforce/comment/163324169 "2025年10月6日 06:19")Edited

Thanks, Cameron for sharing your deep insights on RL. Really enjoying learning from them. A question regarding the contextual bandit section where you mention "Our complete trajectory is a single action and reward!": Does it mean this is in a single turn (single turn meaning single query-response generation) setting? If so, how could we extend this to a multi-turn conversational setting?

Like (1)

Reply

Share

[2 replies by Cameron R. Wolfe, Ph.D. and others](https://cameronrwolfe.substack.com/p/reinforce/comment/163324169)

[](https://substack.com/profile/30404185-joshua-mabry?utm_source=comment)

[Joshua Mabry](https://substack.com/profile/30404185-joshua-mabry?utm_source=substack-feed-item)

[9月30日](https://cameronrwolfe.substack.com/p/reinforce/comment/161164139 "2025年9月30日 01:10")

Thank you for posting such a clear and insightful explanation of how these RL algos work. The diagrams and pseudocode work well together.

Like (1)

Reply

Share

[1 reply by Cameron R. Wolfe, Ph.D.](https://cameronrwolfe.substack.com/p/reinforce/comment/161164139)

[7 more comments...](https://cameronrwolfe.substack.com/p/reinforce/comments)

[Decoder-Only Transformers: The Workhorse of Generative LLMs](https://cameronrwolfe.substack.com/p/decoder-only-transformers-the-workhorse)

[Building the world's most influential neural network architecture from scratch...](https://cameronrwolfe.substack.com/p/decoder-only-transformers-the-workhorse)

Mar 4, 2024 • 

[Cameron R. Wolfe, Ph.D.](https://substack.com/@cwolferesearch)

146

[

15

](https://cameronrwolfe.substack.com/p/decoder-only-transformers-the-workhorse/comments)

![](https://substackcdn.com/image/fetch/$s_!-1vf!,w_320,h_213,c_fill,f_auto,q_auto:good,fl_progressive:steep,g_center/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F6e3c9db5-400a-49de-a235-e09bc3aa3689_2392x1342.png)

[Demystifying Reasoning Models](https://cameronrwolfe.substack.com/p/demystifying-reasoning-models)

[Understanding reasoning models and their relation to standard LLMs...](https://cameronrwolfe.substack.com/p/demystifying-reasoning-models)

Feb 18 • 

[Cameron R. Wolfe, Ph.D.](https://substack.com/@cwolferesearch)

252

[

5

](https://cameronrwolfe.substack.com/p/demystifying-reasoning-models/comments)

![](https://substackcdn.com/image/fetch/$s_!mk9r!,w_320,h_213,c_fill,f_auto,q_auto:good,fl_progressive:steep,g_center/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F23d9c87e-b238-4fdd-996e-4ed4465b9931_2334x1282.png)

[AI Agents from First Principles](https://cameronrwolfe.substack.com/p/ai-agents)

[Understanding AI agents by building upon the most basic concepts of LLMs...](https://cameronrwolfe.substack.com/p/ai-agents)

Jun 9 • 

[Cameron R. Wolfe, Ph.D.](https://substack.com/@cwolferesearch)

339

[

24

](https://cameronrwolfe.substack.com/p/ai-agents/comments)

![](https://substackcdn.com/image/fetch/$s_!IitU!,w_320,h_213,c_fill,f_auto,q_auto:good,fl_progressive:steep,g_center/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fcee4a772-78a7-41b7-8cf1-4da233376ea6_2002x1122.png)

Ready for more?

Subscribe

© 2025 Cameron R. Wolfe

[Privacy](https://substack.com/privacy) ∙ [Terms](https://substack.com/tos) ∙ [Collection notice](https://substack.com/ccpa#personal-data-collected)

[Start your Substack](https://substack.com/signup?utm_source=substack&utm_medium=web&utm_content=footer)[Get the app](https://substack.com/app/app-store-redirect?utm_campaign=app-marketing&utm_content=web-footer-button)

[Substack](https://substack.com/) is the home for great culture