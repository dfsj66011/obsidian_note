
理解为我们带来现代 LLM 的复杂 RL 算法…

Oct 27, 2025

![](https://substackcdn.com/image/fetch/$s_!PJsw!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Ff8db8bc5-f39d-4d1a-be16-26e0c0eb01a7_2502x1398.png)

过去几年中，RL 一直是 LLM 研究中最具影响力的领域之一。早期研究利用 RL 将 LLM 与人类偏好对齐，而这项将 RL 应用于 LLM 的初步工作几乎完全依赖于近端策略优化（PPO）。这一选择使得 PPO 多年来成为 LLM 训练后默认的 RL 算法——*考虑到 LLM 研究的快速发展，这可谓统治地位相当持久！* 直到最近关于 LLM 推理的研究中，研究者们才开始使用 GRPO 等替代算法。

尽管 PPO 非常重要，但除了顶级研究实验室之外，人们对它的了解甚少。这种理解上的缺失是有充分原因的。*PPO 不仅是一种包含微妙实现细节的复杂算法*，而且其高昂的计算和内存开销使得在没有大量计算资源的情况下进行实验变得困难。要成功利用 PPO，既需要对算法有深刻的理解，也需要丰富的领域知识或实践经验。

本概述将从 RL 的基本概念入手，逐步深入理解 PPO 算法。在此基础上，我们将阐述使用 PPO 的关键实践要点，包括 PPO 的伪代码及其各个组成部分。最后，通过分析几项在 LLM 领域推广 PPO 的开创性研究，我们将把这些知识融会贯通。

## 强化学习（RL）基础

在深入了解 PPO 之前，我们需要先学习 RL 的基础知识。本节将介绍强化学习的基本问题设置和术语。此外，我们将推导一个简单的策略梯度表达式，这是 PPO 算法的基础。


Before learning more about PPO, we need to learn about RL in general. This section will cover basic problem setup and terminology for RL. Additionally, we will derive a simple policy gradient expression, which forms a basis for PPO.

#### **Problem Setup and Terminology**

When running RL training, we have an **agent** that takes **actions** within some **environment**; see below.

[

![](https://substackcdn.com/image/fetch/$s_!lQCe!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fd7117e42-c6ab-43c4-8878-5a88cb99c9ae_2203x870.png)



](https://substackcdn.com/image/fetch/$s_!lQCe!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fd7117e42-c6ab-43c4-8878-5a88cb99c9ae_2203x870.png)

Basic problem setup for RL

These actions are predicted by a **policy**—_we can think of the policy as the agent’s brain_—that is usually parameterized. For example, the policy is the LLM itself in the context of training LLMs. We can model the probability of a given action under our policy as `π_θ(a_t | s_t)`. When the policy outputs an action, the **state** of the environment will be updated according to a **transition function**, which is part of the environment. We will denote our transition function as `P(s_t+1 | a_t, s_t)`. However, transition functions are less relevant for LLMs because they are typically a pass-through; i.e., we assume `s_t = {x, a_1, a_2, …, a_t}`, where `x` is the prompt.

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

**RL objective.** When training a model with RL, our goal is to maximize the cumulative reward over the entire trajectory (i.e., the sum of `r_t`). However, there are a few variations of this objective that commonly appear. Specifically, the reward that we maximize can either be discounted or non-discounted[1](https://cameronrwolfe.substack.com/p/ppo-llm#footnote-1-175107358); see below. By incorporating a discount factor `γ`, we reward our policy for achieving rewards sooner rather than later. In other words, _money now is better than money later_.

[

![](https://substackcdn.com/image/fetch/$s_!8D_n!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fbbfd6da8-2406-4197-b9d0-d3a1ec301b39_1496x876.png)



](https://substackcdn.com/image/fetch/$s_!8D_n!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fbbfd6da8-2406-4197-b9d0-d3a1ec301b39_1496x876.png)

Our objective is usually expressed as an expected cumulative reward, where the [expectation](https://en.wikipedia.org/wiki/Expected_value) is taken over the trajectory. Expanding this expectation yields a sum over trajectories weighted by their probabilities. We can formulate this in a continuous or discrete manner; see below.

[

![](https://substackcdn.com/image/fetch/$s_!45io!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F523baab0-10b4-438e-85d7-e7c5c0681209_1692x884.png)



](https://substackcdn.com/image/fetch/$s_!45io!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F523baab0-10b4-438e-85d7-e7c5c0681209_1692x884.png)

**State, value, and advantage functions.** Related to RL objective, we can also define the following set of functions:

- _Value Function_ `V(s)`: the expected cumulative reward when you start in state `s` and act according to your current policy `π_θ`.
    
- _Action-Value Function_ `Q(s, a)`: the expected cumulative reward when you start in state `s`, take action `a`, then act according to your policy `π_θ`.
    
- _Advantage Function_ `A(s, a)`: the difference between the action-value and value function; i.e., `A(s, a) = Q(s, a) - V(s)`.
    

Intuitively, the advantage function tells us how useful some action `a` is by taking the difference between the expected reward after taking action `a` in state `s` and the general expected reward from state `s`. The advantage will be positive if the reward from action `a` is higher than expected and vice versa. Advantage functions play a huge role in RL research—_they are used to compute the gradient for our policy_.

> _“Sometimes in RL, we don’t need to describe how good an action is in an absolute sense, but only how much better it is than others on average. That is to say, we want to know the relative advantage of that action. We make this concept precise with the advantage function.**”**_ - [Spinning up in Deep RL](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html)

#### RL Formulation for LLMs

[

![](https://substackcdn.com/image/fetch/$s_!RBDE!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fd4b8b6b8-fe96-4b70-87d2-038a3b3511cf_1346x1134.png)



](https://substackcdn.com/image/fetch/$s_!RBDE!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fd4b8b6b8-fe96-4b70-87d2-038a3b3511cf_1346x1134.png)

RL terminology mapping for LLMs

Now that we understand RL basics, we need to map the terminology that we have learned to the setting of LLM training. We can do this as follows (shown above):

- Our **policy** is the LLM itself.
    
- Our **initial state** is the prompt.
    
- The LLM’s output—_either each token or the entire completion_—is an **action**.
    
- Our **state** is the combination of our prompt with the LLM’s output.
    
- The entire completion from the LLM forms a **trajectory**.
    
- The **reward** comes from a verifier or reward model (more details to follow).
    

Notably, there is no transition function in this setup because the transition function is completely deterministic. If we start with a prompt `x` and our LLM predicts tokens `t_1` and `t_2` given this prompt as input, then our updated state simply becomes `s_2 = {x, t_1, t_2}`. In other words, _our state is just the running completion being generated by the LLM for a given prompt_ `x`.

**MDP formulation.** For LLMs, there are two key ways in which RL can be formulated that differ in how they model actions:

1. _Bandit formulation_: the entire completion or response from the LLM is modeled as a single action.
    
2. _Markov Decision Process (MDP) formulation_: each token within the LLM’s output is modeled as an individual action.
    

We outlined the details for both of these formulations in a [prior overview](https://cameronrwolfe.substack.com/i/173306894/markov-decision-process-mdp-versus-bandit-formulation). However, PPO relies upon the MDP formulation, so we will primarily focus upon the MDP formulation here. As we should recall, an LLM generates output via [next token prediction](https://cameronrwolfe.substack.com/i/136638774/understanding-next-token-prediction); i.e., by generating each token in the output completion sequentially. This autoregressive process is depicted below.

[

![](https://substackcdn.com/image/fetch/$s_!QUg4!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F5b1a8412-5cfb-481f-bd50-473f0a6fd9b5_1992x1037.png)



](https://substackcdn.com/image/fetch/$s_!QUg4!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F5b1a8412-5cfb-481f-bd50-473f0a6fd9b5_1992x1037.png)

Autoregressive next token prediction with an LLM

Next token prediction maps easily to an RL setup—_we can model each token as an action_! This setup is called the [Markov Decision Process (MDP)](https://en.wikipedia.org/wiki/Markov_decision_process) formulation. An MDP is a probabilistic framework for modeling decision-making that includes states, actions, transition probabilities and rewards—_this is exactly the setup we have discussed so far for RL_! The MDP formulation used for RL is shown below.

[

![](https://substackcdn.com/image/fetch/$s_!KWz-!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F52f4f8de-4456-4cbd-935c-a945968b704d_1466x916.png)



](https://substackcdn.com/image/fetch/$s_!KWz-!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F52f4f8de-4456-4cbd-935c-a945968b704d_1466x916.png)

When modeling RL as an MDP for LLMs, our initial state is the prompt and our policy acts by predicting individual tokens. Our LLM forms a (stochastic) policy that predicts a probability distribution over tokens. During generation, actions are taken by selecting a token from this distribution—_each token is its own action_. After a token is predicted, it is added to the current state and used by the LLM to predict the next token—_this is just autoregressive next token prediction_! Eventually, the LLM predicts a stop token (e.g., `<|end_of_text|>` or `<eos>`) to complete the generation process, thus yielding a complete trajectory.

#### Policy Gradient Basics

During RL training, we want to maximize our objective—_the cumulative (possibly discounted) reward_. To accomplish this, we can just use [gradient ascent](https://en.wikipedia.org/wiki/Gradient_descent); see below.

[

![](https://substackcdn.com/image/fetch/$s_!slrY!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Ff3072897-d905-42be-b385-6186c24ae059_2390x302.png)



](https://substackcdn.com/image/fetch/$s_!slrY!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Ff3072897-d905-42be-b385-6186c24ae059_2390x302.png)

Solving the RL objective with gradient ascent

To put this in the context of LLMs, RL training follows the sequence of steps shown below. We first sample a batch of prompts and generate completions to these prompts with our LLM or policy. Then, we compute the rewards for these completions (more details to follow in later sections) and use these rewards to derive a policy update. _This final policy update step is where gradient ascent is used_.

[

![](https://substackcdn.com/image/fetch/$s_!yR8D!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F20b7b374-8bee-45fb-b7ee-a26008aa7259_1267x843.png)



](https://substackcdn.com/image/fetch/$s_!yR8D!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F20b7b374-8bee-45fb-b7ee-a26008aa7259_1267x843.png)

Key steps in RL training for LLMs

To be more specific, we use the completions and rewards to estimate the gradient of the RL training objective with respect to the parameters of our policy—_this is called the “policy gradient”_. If we can compute this gradient, then we can train our policy using gradient ascent. But, the question is: _How do we compute this gradient?_

> _“The goal of reinforcement learning is to find an optimal behavior strategy for the agent to obtain optimal rewards. The policy gradient methods target at modeling and optimizing the policy directly.”_ - [Lilian Weng](https://lilianweng.github.io/posts/2018-04-08-policy-gradient/)

**Policy gradients.** Nearly all RL optimizers used for LLM training (e.g., PPO [1], [GRPO](https://arxiv.org/abs/2402.03300), and [REINFORCE](https://cameronrwolfe.substack.com/p/reinforce)) are policy gradient algorithms, which operate by _i)_ estimating the policy gradient and _ii)_ performing gradient ascent with this estimate. These algorithms use different approaches for estimating the policy gradient, but the high-level idea behind all of them is quite similar—_we just tweak small details depending on the exact technique being used_. To understand policy gradient algorithms more deeply, we will first derive the simplest form of a policy gradient. Then, we will extend this idea to recover more intricate policy gradient algorithms like Trust Region Policy Optimization (TRPO) [6] and PPO [1].

The **Vanilla Policy Gradient (VPG)** has been extensively covered by many online resources. Other useful explanations of the VPG include:

- Intro to Policy Optimization from OpenAI [[link](https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html)]
    
- RLHF Book from [Nathan Lambert](https://natolambert.com/) [[link](https://rlhfbook.com/c/11-policy-gradients.html)]
    
- Policy Optimization Algorithms from [Lilian Weng](https://lilianweng.github.io/) [[link](https://lilianweng.github.io/posts/2018-04-08-policy-gradient/)]
    
- Policy Gradient Algorithms from this blog[2](https://cameronrwolfe.substack.com/p/ppo-llm#footnote-2-175107358) [[link](https://cameronrwolfe.substack.com/p/policy-gradients-the-foundation-of)]
    

However, we will again derive some simple forms of the policy gradient here for completeness. As we already know, our goal in RL is to maximize cumulative rewards. If we try to compute the gradient of this objective with respect to the parameters of our policy `θ`, we can derive the following:

[

![](https://substackcdn.com/image/fetch/$s_!GetI!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F1685ea69-1b2c-438c-87ed-dba51c4bee65_2406x1065.png)



](https://substackcdn.com/image/fetch/$s_!GetI!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F1685ea69-1b2c-438c-87ed-dba51c4bee65_2406x1065.png)

([source](https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html))

This derivation starts with the gradient of our RL training objective (cumulative reward) and ends with a basic expression for the policy gradient. The steps used in this derivation are enumerated above. The only complicated steps here are the use of the [log-derivative trick](https://andrewcharlesjones.github.io/journal/log-derivative.html) and the final step, which leverages our definition for the probability of a trajectory. In the final step, we substitute in our definition for the probability of a trajectory and observe that the gradients of the initial state probability and transition function with respect to the policy parameters are always zero because neither of them depend on the policy; see below.

[

![](https://substackcdn.com/image/fetch/$s_!Rkmm!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fb0f526be-55f2-4eae-abd8-fa4382d8335a_1564x432.png)



](https://substackcdn.com/image/fetch/$s_!Rkmm!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fb0f526be-55f2-4eae-abd8-fa4382d8335a_1564x432.png)

([source](https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html))

**Implementing a basic policy gradient.** The basic policy gradient expression we have derived so far is theoretical—_it involves an expectation_. If we want to actually compute this gradient in practice, we must approximate it with a sample mean. In other words, we sample a fixed number of trajectories—_or prompts and completions in the case of an LLM_—and take an average over the policy gradient expression for each of these trajectories. The basic policy gradient expression contains two key quantities that we already know how to compute:

- The reward comes directly from a verifier or reward model.
    
- Log probabilities of actions can be computed with our LLM (i.e., these are just the token probabilities from the LLM’s output).
    

To make the process of computing the basic policy gradient more concrete, a step-by-step implementation in PyTorch pseudocode has been provided below.

[

![[animate output image]](https://substackcdn.com/image/fetch/$s_!PYzF!,w_1456,c_limit,f_auto,q_auto:good,fl_lossy/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F3e4bdafe-cd71-48b7-8a10-abdc895432f7_1920x1076.gif "[animate output image]")



](https://substackcdn.com/image/fetch/$s_!PYzF!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F3e4bdafe-cd71-48b7-8a10-abdc895432f7_1920x1076.gif)

One key detail that we should notice in the above implementation is that we do not compute the policy gradient directly. Rather, we formulate a loss function for which the gradient is equal to the policy gradient then use [autodiff](https://en.wikipedia.org/wiki/Automatic_differentiation) in PyTorch to compute the policy gradient—_this happens during_ `loss.backward()`. The exact loss function used to compute the policy gradient is shown below.

[

![](https://substackcdn.com/image/fetch/$s_!TwP0!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fa4bb2d85-fdea-4cfc-a46b-e6c5f78ff4f4_1613x593.png)



](https://substackcdn.com/image/fetch/$s_!TwP0!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fa4bb2d85-fdea-4cfc-a46b-e6c5f78ff4f4_1613x593.png)

Creating a loss function for the policy gradient

This distinction is important to understand because we will formulate PPO (and TRPO!) via a loss function rather than a direct expression for the policy gradient.

**Problems with the basic policy gradient.** The basic policy gradient expression is straightforward, but it suffers from several notable issues:

- _High Variance_: The gradient estimates can have high variance, making training unstable.
    
- _Unstable Policy Updates_: There is no mechanism to prevent large, potentially destabilizing updates to the policy.
    

Due to the high variance, accurately estimating the policy gradient often requires sampling many trajectories per training iteration, which is computationally expensive. We must generate many completions with the LLM and compute the rewards and token log probabilities for all of these completions.

Additionally, this high variance increases the risk of training instability—_large and inaccurate updates could potentially cause significant harm to our policy_. To solve these issues, most policy gradient algorithms focus on reducing the variance of policy gradient estimates and enforcing a trust region on policy updates (i.e., limiting how much the policy can change in a single update).

> _“Taking a step with this gradient pushes up the log-probabilities of each action in proportion to_ `R(𝜏)`_, the sum of all rewards ever obtained.”_ - [Spinning up in Deep RL](https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html)

**Reward-to-go.** For example, we see in our basic policy gradient (copied below for reference) that we are increasing the probability of a given action based upon the cumulative reward of a trajectory. Therefore, we may increase the probability of an action due to rewards that were observed before the action even occurred!

[

![](https://substackcdn.com/image/fetch/$s_!Ymws!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F6b14bade-8617-4bfa-9e4a-59811bbe8de7_1374x218.png)



](https://substackcdn.com/image/fetch/$s_!Ymws!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F6b14bade-8617-4bfa-9e4a-59811bbe8de7_1374x218.png)

Basic policy gradient expression

This simple observation led to the creation of the “reward-to-go” policy gradient; see below. This modified policy gradient expression just replaces the cumulative reward with the sum of rewards observed after an action. Using the [EGLP lemma](https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html#expected-grad-log-prob-lemma), we can show that this reward-to-go formulation is an unbiased estimator of the policy gradient. Additionally, the reward-to-go policy gradient has provably lower variance compared to the basic policy gradient expression from before.

[

![](https://substackcdn.com/image/fetch/$s_!s3m9!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F92c4ac85-74ac-4c12-8d51-c6c9b3bf22ba_2216x460.png)



](https://substackcdn.com/image/fetch/$s_!s3m9!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F92c4ac85-74ac-4c12-8d51-c6c9b3bf22ba_2216x460.png)

The reward-to-go policy gradient

**Baselines.** To further reduce variance, we can also add a baseline to our policy gradient expression; see below. Similarly to the reward-to-go policy gradient, we can use the EGLP lemma to show that a baselined version of our policy gradient is unbiased and has lower variance. Due to the EGLP lemma, this baseline must only depend upon the current state (i.e., otherwise an assumption of the EGLP lemma is violated and the proofs are no longer valid).

[

![](https://substackcdn.com/image/fetch/$s_!QhBq!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fd4801db8-b3f3-4ec3-9d3f-624b8ffbd550_1774x344.png)



](https://substackcdn.com/image/fetch/$s_!QhBq!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fd4801db8-b3f3-4ec3-9d3f-624b8ffbd550_1774x344.png)

Adding a baseline to our policy gradient expression

This expression is nearly identical to the reward-to-go policy gradient—_we just subtract an additional baseline from the reward-to-go term_. There are many possible choices for baselines that can be used in policy gradient estimates. One common baseline is the value function. _Using the value function as a baseline positively reinforces actions that achieve a cumulative reward that is higher than expected._

_A common problem with vanilla policy gradient algorithms is the high variance in gradient updates… In order to alleviate this, various techniques are used to normalize the value estimation, called baselines. Baselines accomplish this in multiple ways, effectively normalizing by the value of the state relative to the downstream action (e.g. in the case of Advantage, which is the difference between the Q value and the value). The simplest baselines are averages over the batch of rewards or a moving average. - [RLHF book](https://rlhfbook.com/c/11-policy-gradients.html)_

**Generic policy gradient.** In [3], the options for computing the policy gradient were summarized with a more generic policy gradient expression; see below.

[

![](https://substackcdn.com/image/fetch/$s_!Vl-C!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F58aa8bae-6778-4ec0-ac53-3f8b8550390f_2137x836.png)



](https://substackcdn.com/image/fetch/$s_!Vl-C!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F58aa8bae-6778-4ec0-ac53-3f8b8550390f_2137x836.png)

(from [3])

This expression is nearly identical to expressions we have seen so far. The only difference is that we have changed our reward term `R(𝜏)` to a generic `Ψ_t` term, which can be set equal to several different expressions. For example, we can:

- Set `Ψ_t = R(𝜏)` to recover our basic policy gradient expression.
    
- Set `Ψ_t` equal to rewards received after time `t` to recover our reward-to-go variant of the policy gradient.
    
- Set `Ψ_t` equal to a baselined version of the reward; e.g., the difference between cumulative reward `R(𝜏)` and the value function `V(s_t)`.
    
- Set `Ψ_t` equal to the state-action (`Q`) or advantage function (`A`).
    

Despite the many possible formulations, PPO—_and nearly all of the RL optimizers used in the domain of LLMs_—focuses upon setting `Ψ_t` equal to the advantage function `A(s_t, a_t)`. _This setting is referred to as the vanilla policy gradient (VPG)_; see below. In theory, the VPG yields the lowest-variance gradient estimate.

[

![](https://substackcdn.com/image/fetch/$s_!1PL6!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F3dbd6ad6-4d9e-4085-b4a7-849b29789350_1662x470.png)



](https://substackcdn.com/image/fetch/$s_!1PL6!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F3dbd6ad6-4d9e-4085-b4a7-849b29789350_1662x470.png)

The vanilla policy gradient

Although the VPG has low variance, there is still no mechanism to enforce a trust region in the policy update—_a large and destructive policy update can still destabilize the training process_. PPO was created as a solution to this problem. As we will see, PPO resembles the basic policy gradient expressions we have seen but has added mechanisms for enforcing a trust region on the policy update. We will now learn more about PPO and the many practical details involved in its implementation.

## Proximal Policy Optimization (PPO)

Now that we understand RL basics, we will spend the next section learning about Proximal Policy Optimization (PPO) [1]. This explanation will build upon the VPG expression that we derived in the last section, beginning with Trust Region Policy Optimization (TRPO) [6]—_a predecessor to PPO_. TRPO is effective at stabilizing training, but it is also relatively complex. PPO was developed as a more practical alternative with similar benefits. To conclude the section, we will also cover Generalized Advantage Estimation (GAE) [3], which is the most common approach for computing the advantage function in PPO.

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

---

#### Subscribe to Deep (Learning) Focus

By Cameron R. Wolfe · Launched 3 years ago

I contextualize and explain important topics in AI research.

Subscribe

By subscribing, I agree to Substack's [Terms of Use](https://substack.com/tos), and acknowledge its [Information Collection Notice](https://substack.com/ccpa#personal-data-collected) and [Privacy Policy](https://substack.com/privacy).

[](https://substack.com/profile/159709240-marco-aurelio-sterpa)

[](https://substack.com/profile/322569007-yann)

[](https://substack.com/profile/5623511-shivaram-ys)

[](https://substack.com/profile/124988503-ashish-ibm)

[](https://substack.com/profile/174228865-emmanuel-maminta)

64 Likes∙

[4 Restacks](https://substack.com/note/p-175107358/restacks?utm_source=substack&utm_content=facepile-restacks)

[](https://cameronrwolfe.substack.com/p/ppo-llm/comments)

#### Discussion about this post

[](https://substack.com/profile/307819638-neocloud-deep-dives?utm_source=comment)

[NEOCLOUD DEEP DIVES](https://substack.com/profile/307819638-neocloud-deep-dives?utm_source=substack-feed-item)

[9h](https://cameronrwolfe.substack.com/p/ppo-llm/comment/171287800 "2025年10月29日 11:42")

This comprehensive guide brilliantly bridges the gap between theoretical RL and practical LLM implementation. The progression from basic policy gradients to GAE is particularly well structred. Your breakdown of the four clipping cases in PPO finally made that mechanism click for me - seeing how advantage sign determines when clipping activates is invaluable.

Like

Reply

Share

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

253

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