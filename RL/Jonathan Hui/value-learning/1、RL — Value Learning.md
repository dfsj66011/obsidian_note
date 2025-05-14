
> https://jonathan-hui.medium.com/rl-value-learning-24f52b49c36d


价值学习是 RL 中的一个基本概念。它是学习 RL 的入门点，就像深度学习中的全连接网络一样基础。价值学习用于估计到达某些状态或采取某些行动的好坏。虽然单独使用价值学习可能不足以解决复杂问题，但它是许多 RL 方法的关键构建块。在本文中，我们将通过示例来演示其概念。

假设我们计划从旧金山到圣何塞的旅行。假设你是一位专注的数据科学家，并在决策中考虑了许多因素。这些因素可能包括剩余距离、交通状况、道路条件，甚至是收到罚单的可能性。经过分析后，你为每个城市打分，并总是选择得分最高的下一条路线。

例如，当你在圣布鲁诺（SB）时，有两条可能的路线。根据它们的评分，SM 的评分更高，因此我们将前往 SM 而不是 WSM。

![|500](https://miro.medium.com/v2/resize:fit:700/1*GfmBUSLATQ_9d-Y2nTptdQ.png)

在强化学习（RL）中，价值学习方法基于类似的原则。我们估计处于某个状态的好坏程度，并采取能收集到最高总奖励的行动进入下一个状态。

### 1、价值函数

直观地说，价值函数 $V(s)$ 衡量在特定状态下的好坏程度。根据定义，它是遵循特定策略所能收集到的期望折扣奖励总和：$$V^{\pi}(s_t) = \sum_{t' = t}^{T} \mathbb{E}_{\pi_0} \left[ \gamma^{t' - t} r(s_{t'}, a_{t'}) \mid s_t \right]$$其中，$\gamma$ 是折扣因子。如果 $\gamma$ 小于 1，我们会以较低的当前价值看待未来的奖励。在这里的大多数例子中，为了简化说明，我们将 $\gamma$ 设为 1。我们的目标是找到一个能最大化期望奖励的策略。$$
V^*(s) = \max_{\pi} \mathbb{E} \left[ \sum_{t=0}^{H} \gamma^t R(s_t, a_t, s_{t+1}) \mid \pi, s_0 = s \right]$$通过价值学习，有许多方法可以找到最优策略。我们将在接下来的几个部分中讨论它们。



# **Value iteration**

First, we can use dynamic programming to calculate the optimal value of _V_ iteratively.

![](https://miro.medium.com/v2/resize:fit:700/1*q3PmocI0SGELSfNNFUjx8g.png)

Then, we can use the value function to derive the optimal policy.

When we are in SB, we have two choices.

![](https://miro.medium.com/v2/resize:fit:700/1*9N1LUeBN6DIIL80h_lqYKA.png)

The SB to SM route receives a -10 rewards because SB is further away. We get an additional -5 rewards for the SB to WSM route because we can get a speeding ticket easily in that stretch of the highway.

The optimal V*(SB) = max( -10 + V*(SM), -15 + V*(WSM)) = 60.

**Value Iteration Example**

Let’s get into a full example. Below is a maze with the exit on the top left. At every location, there are four possible actions: up, down, left or right. If we hit the boundary, we bounce back to the original position. Every single-step move receives a negative one reward. Starting from the terminal state, we propagate the value of _V_* outwards using:

![](https://miro.medium.com/v2/resize:fit:700/1*c2bAqQu5jcXy7JfmJZLxpw.png)

The following is the graphical illustration from iteration one to seven.

![](https://miro.medium.com/v2/resize:fit:700/1*FbAckmJ5lGMmK8QMolK65Q.png)

Modified from [source](https://mitpress.mit.edu/books/reinforcement-learning-second-edition)

Once it is done, for every location, we locate the neighbor with the highest _V_-value as our best move.

# **Policy evaluation**

The second method is the policy evaluation. A policy tells us what to do from a particular state.

![](https://miro.medium.com/v2/resize:fit:700/1*3qVw9kj0Unbh2SjScCt8wg.png)

We can evaluate a random policy continually to calculate its value functions. A random policy is simply a policy that take any possible action randomly. Let’s consider another maze with exits on the top left and the bottom right. The value function is calculated as:

![](https://miro.medium.com/v2/resize:fit:700/1*zM_XzpjfW1h6HK04nn1vWQ.jpeg)

For a random policy, each action has the same probability. For four possible actions, π(_a|s_) equals 0.25 for any state.

![](https://miro.medium.com/v2/resize:fit:700/1*0CQDoJrtngme2mj_ykvsgQ.png)

Modified from [source](https://mitpress.mit.edu/books/reinforcement-learning-second-edition)

For iteration 3 below, V[2, 2] = -2.9: we subtract one from each neighbor (negative one reward for every move), and take their average.

![](https://miro.medium.com/v2/resize:fit:700/1*hqExD01qgsj7eZv0qb4wSA.png)

As we continue the iteration, _V_ will converge and we can use the value function to determine the optimal policy again.

![](https://miro.medium.com/v2/resize:fit:700/1*RcRLrxZOctJ5cWXLh4V4nw.png)

# **Policy Iteration**

The third method is the policy iteration. Policy iteration performs policy evaluation and policy improvement alternatively:

![](https://miro.medium.com/v2/resize:fit:700/1*vTm5U-0UOb_HUqlD_2SsAA.png)

We continuously evaluate the current policy but we also refine the policy in each step.

![](https://miro.medium.com/v2/resize:fit:700/1*hdJrrav0zRMjzLsiMRFAjA.jpeg)

As we keep improving the policy, it will converge to the optimal policy.

![](https://miro.medium.com/v2/resize:fit:700/1*ha_omzqCR2X_p3Ti2frMUA.png)

[Source](https://mitpress.mit.edu/books/reinforcement-learning-second-edition)

Here is the example which we can find the optimal policy in four iterations, much faster than the policy evaluation.

![](https://miro.medium.com/v2/resize:fit:700/1*qQXcvHyr6ceu-yXmOH1TMg.png)

Modified from [source](https://mitpress.mit.edu/books/reinforcement-learning)

**Algorithm**

Let’s formulate the equations. The value-function at time step i+1 equals

![](https://miro.medium.com/v2/resize:fit:700/1*UuStnOOhxPGRce46VWjMHA.png)

Where **_P_** is the **model** (system dynamics) determining the next state after taking an action. The refined policy will be

![](https://miro.medium.com/v2/resize:fit:700/1*PLHPmIGB0XsZDBCmIlLgLA.png)

For a deterministic model**,** the equation can be simplified to:

![](https://miro.medium.com/v2/resize:fit:700/1*IO5MhCRyusHcelRP6LtwSQ.png)

![](https://miro.medium.com/v2/resize:fit:700/1*we4u87tMdNnuBc9boIYNlQ.png)

Here is the general flow of the algorithm:

![](https://miro.medium.com/v2/resize:fit:700/1*lHBPuSgDbh1fFKOWfVqZQQ.jpeg)

# **Bellman Optimality Equation**

In the previous section, we use the **dynamic programming** to learn the value iteratively. The equation below is often mentioned in RL and is called the Bellman equation constraint.

![](https://miro.medium.com/v2/resize:fit:700/1*CNjO1bWVXbPLaI8JpyGwsQ.png)

# **Value-Function Learning with S**tochastic Model

In the previous value iteration example, we spread out the optimal value _V_* calculation to its neighbors in each iteration

![](https://miro.medium.com/v2/resize:fit:700/1*593Fb9wOqfuibf7Nz59X2g.jpeg)

using the equation:

![](https://miro.medium.com/v2/resize:fit:700/1*LKjj0pA1q33s6TcFgOp5wg.png)

In those examples, the model _P_ is deterministic and is known. **_P_** is all zero except one state (_s’_) that is one. Therefore, it is simplified to:

![](https://miro.medium.com/v2/resize:fit:700/1*MiXJs48s2_VtLeNMuchMcg.png)

But for the stochastic model, we need to consider all possible future states.

![](https://miro.medium.com/v2/resize:fit:700/1*69YWFuR47FNsU3YypO4pkA.png)

![](https://miro.medium.com/v2/resize:fit:700/1*MROFEiyjAuyUKFmMkqD5wQ.png)

Let’s demonstrate it with another maze example using a stochastic model. This model has a noise of 0.2. i.e., if we try to move right, there is 0.8 chance that we do move right. But there is a 0.1 chance that we move up and 0.1 chance that we move down instead. If we hit a wall or boundary, we bounce back to the original position.

![](https://miro.medium.com/v2/resize:fit:700/1*-Uk5DOYXPn0sLfrXyJn0Lg.jpeg)

Modified from [source](https://drive.google.com/file/d/0BxXI_RttTZAhVXBlMUVkQ1BVVDQ/view)

We assume the discount factor γ will be 0.9 and we receive a zero reward for every move unless we hit the terminate state which is +1 for the green spot and -1 for the red spot above.

Let’s fast forward to iteration 5, and see how to compute V*[2, 3] (underlined in white below) from the result of iteration 4.

![](https://miro.medium.com/v2/resize:fit:700/1*dUWqCY3c_1eZATZXBXmljQ.jpeg)

Modified from [source](https://drive.google.com/file/d/0BxXI_RttTZAhVXBlMUVkQ1BVVDQ/view)

The state above [2, 3] has the highest V value. So the optimal action for V*[2, 3] is going up. The new value function is

![](https://miro.medium.com/v2/resize:fit:700/1*093wehxUYGRi1Of2_cQ5Mg.jpeg)

In each iteration, we will re-calculate V* for every location except the terminal state. As we keep iterating, _V*_ will converge. For example, V*[2, 3] eventually converges to 0.57.

![](https://miro.medium.com/v2/resize:fit:700/1*wg0I2CDjQvzooZUf6xBG-g.png)

**Algorithm**

Here is the pseudocode for the value iteration:

![](https://miro.medium.com/v2/resize:fit:700/1*kZL-KHzRcqpJLWPVy6hDwg.png)

[Source](https://drive.google.com/file/d/0BxXI_RttTZAhVXBlMUVkQ1BVVDQ/view)

# Model-Free

Regardless whether it is a deterministic or a stochastic model, we need the model **_P_** to compute the value function or to derive the optimal policy. (even though in a deterministic model, _P_ is all zero except one state which is one.)

![](https://miro.medium.com/v2/resize:fit:700/1*69YWFuR47FNsU3YypO4pkA.png)

![](https://miro.medium.com/v2/resize:fit:700/1*MROFEiyjAuyUKFmMkqD5wQ.png)

**Monte-Carlo method**

Whenever we don’t know the model, we fall back to sampling and observation to estimate the total rewards. Starting from the initial state, we run a policy and observe the total rewards (**_G_**) collected.

![](https://miro.medium.com/v2/resize:fit:700/1*M7sFNvua6g9VW88J2NxanA.png)

**_G_** is equal to

![](https://miro.medium.com/v2/resize:fit:700/1*hPKvFvVZ5rAcWtuCnRQXzQ.png)

If the policy or the model is stochastic, the sampled total rewards can be different in each run. We can run and reset the system multiple times to find the average of _V(S)_. Or we can simply keep a running average like the one below so we don’t need to keep all the previous sampled results.

![](https://miro.medium.com/v2/resize:fit:700/1*CtpnzBGgGHuRSiJOUHxWhA.png)

> ==Monte-Carlo method samples actions until the end of an episode to approximate total rewards.==

**Monte-Carlo control**

Even we can estimate _V(S)_ by sampling, how can we determine the action from one state to another?

![](https://miro.medium.com/v2/resize:fit:700/1*MROFEiyjAuyUKFmMkqD5wQ.png)

Without knowing the model, we don’t know what action can lead us to the next optimal state s’. For example, without the road signs (the model), we don’t know whether the left lanes or the right lanes of the highway lead us to SM or WSM?

![](https://miro.medium.com/v2/resize:fit:700/1*z3EZBCUMc-gKsoF-PDW7eg.png)

In the pong game below, we know what state we want to reach. But without the model, we don’t know how far (or how hard) should we push the joystick.

![](https://miro.medium.com/v2/resize:fit:700/1*rn8LnsP3f_rMT3M9yN7BJg.jpeg)

# **Action-value Learning**

This comes to the action-value function, the cousin of value function but without the need of a model. Instead of measuring how good a state **_V(s)_** is, we measure how good to take an action at a state **_Q(s, a)_**. For example, when we are at SB, we ask how good to take the right lanes or the left lanes on Highway 101 even though we don’t know where it leads us to. So at any state, we can just take the action with the highest _Q_-value. This allows us to work without a model at the cost of more bookkeeping for each state. For a state with _k_ possible actions, we have now _k Q_-values.

![](https://miro.medium.com/v2/resize:fit:700/1*OihK80yU3K8igPKRKEMFJg.png)

The **Q-value** (**action-value function**) is defined as the expected rewards for an action under a policy.

![](https://miro.medium.com/v2/resize:fit:700/1*yqdMSBJkwim8v_jAdvxuyg.png)

Similarly to the previous discussion, we can use the Monte-Carlo method to find **_Q_**.

![](https://miro.medium.com/v2/resize:fit:700/1*pI6fFYkmywG4jC3zREP2fg.png)

In our example, we will keep on the left lanes when we are in SB.

![](https://miro.medium.com/v2/resize:fit:700/1*uV_i2jpKzn6837kIIqOomg.png)

This is the Monte-Carlo method for _Q_-value function.

# **Policy Iteration with Q-value (model-free)**

We can apply the Q-value function in the policy iteration. Here is the flow:

![](https://miro.medium.com/v2/resize:fit:700/1*DYLVYv1MsgDqSlv6tepLDA.jpeg)

# **Issues**

The solution presented here has some practical problems. First, it cannot scale well for a large state space. The memory to keep track of _V_ or _Q_ for each state is impractical. Can we build a function estimator for value functions to save memory, just like the deep network for a classifier? Second, the Monte-Carlo method has very high variance. A stochastic policy may compute very different reward results in different runs. This high variance hurts the training. How can train the model with better convergence?

# Recap

Before looking into the solution. Let’s recap what we learn. We want to find an optimal policy that can maximize the expected discounted rewards.

![](https://miro.medium.com/v2/resize:fit:700/1*qw5vW_BrJeLQfKzYiIOFgw.png)

We can solve it by computing the value function or the Q-value function:

![](https://miro.medium.com/v2/resize:fit:700/1*__oKSGNMwgoo9cqD11ks7g.png)

![](https://miro.medium.com/v2/resize:fit:700/1*gTE6UacIuNGDHtHTpMqtEw.png)

And this can be solved using dynamic programming:

![](https://miro.medium.com/v2/resize:fit:700/1*Ur3ZXkd5-6EulriBChdigQ.png)

![](https://miro.medium.com/v2/resize:fit:700/1*YpBbf9JK5-p9yBipOw5AEQ.png)

or one-step lookahead which is also called **Temporal Difference** TD.

![](https://miro.medium.com/v2/resize:fit:700/1*XMgf7S8nfcbz2Mu9f4YXkw.png)

![](https://miro.medium.com/v2/resize:fit:700/1*h2eKK9sNQ_wK6XyeXEfQIw.png)

# Thoughts

Stay tuned for the next part where we will solve some of the problems mentioned before and apply them in practice.

