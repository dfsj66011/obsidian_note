
[The Danger of Overthinking: Examining the Reasoning-Action Dilemma in Agentic Tasks](https://arxiv.org/pdf/2502.08235)

2025.02，UC Berkeley，通过标题能看出来，与 agent、overthinking 有关系，

**问题：**

这篇 paper 要解决的问题是 agent overthinking 的问题，这里 *特指 reasoning-action 困境的现象*，与 R1 的 overthinking 不同，在 LLM 中指的是产生的思维链过长。这里提到的 overthinking 是指  agent 在解决实际问题的时候，可以花更多的资源在 reasoning 上，也可以花更多的资源在和环境进行互动上，而在计算资源一定的前提下，agent 如何做选择就是困境（dilemma）。

作者发现，现在的 LLM，无论是推理还是非推理模型，都更倾向于做内部的思考，而不是和环境的互动，取获取更新的信息，这一类现象他们称为 agent 的 overthinking。

**solution：**

提出 overthinking score，然后基于它，做比较全面的评估，包括主流非推理和推理模型，那如何解决 agent overthinking 的问题？

 提出了一个非常简单的方法，overthinking@k，类似于 pass@k，pass@k 意思是，让 agent 产生 k 个答案，只要有一个答案对，就认为 agent 是正确的。overthinking@k 是让 agent 产生 k 个答案，选择 score 最小的答案，可以很大的节省计算的成本。
 
paper 的主要内容就是这些。

-----------
**问题及现象：**


![[Pasted image 20250331150919.png|400]]
这个图是该论文的 framework，使用的数据集是 SWE-bench Verified，是基于 github issue 的数据集，给定一些真实的 github issue，让 agent 去解决这些跟编程有关的问题。

agent 有一个仿真的环境，可以做一些 planning，在仿真的环境里去测试哪些方案是可行的，当确定了方案之后，agent 可以采取这样的 action 与真实的环境进行互动，并且得到 observation，然后互动的历史，又会返还给 agent 进行下一轮的互动，这就是整个的 framework。

绿色部分，可以看成是 agent 的 internal reasoning / internal simulation，即 agent 自己的思考过程；蓝色部分，可以看成是实际的和环境的互动，也就是 environment interaction，所以就有 balance 的问题。

他们就发现，现在主流的 LLM，作为 agent 更倾向于去做 internal reasoning simulation 而不是去和环境做真实的互动，去收集有用的信息。

![[Pasted image 20250331152038.png|400]]

横坐标是 overthinking score，纵坐标是解决任务的百分比，值越高，说明更多的任务能够被解决，黄色的是推理模型，蓝色的是非推理模型，后缀 ”FC=function callig“，

1. 不论是推理还是非推理，overthinking score 越大，成功解决任务的百分比越低
2. 黄线往右偏移，即推理模型的 score 要更大一些，即推理模型作为 agent，更倾向 overthinking

作者就通过这个图说明 overthinking 现象还是非常普遍的，并且会导致 agent 的性能下降。

![[Pasted image 20250331152813.png|700]]

分析 agent 出问题的三种模式：

(a) 分析瘫痪：智能体耗费过多时间规划未来步骤，却未能在环境中取得实质性进展  
(b) 越界操作：未等环境做出响应，就执行后续操作，违反环境的顺序执行约束，太心急了
(c) 过早终止：智能体仅基于内部预测就提前终止任务，而非根据环境反馈做出决策

在定义了这三类问题之后，作者用 claude3.5 作为一个裁判，去评判其他 agent 产生的 internal reasoning，计算 overthinking score。

Claude 3.5 具体使用的 prompt，见附录 A，告诉 Claude3.5 做一个角色扮演，它是一个 AI judge，要判断一下其他的模型是更喜欢做 internal reasoning 还是跟环境产生互动，然后再把其他模型产生的 internal reasoning 的 CoT 放在标签里，要 claude3.5 去分析这三类问题，并且把问题细化，0-10 分，各个情况的分值标准，对于具体的每个问题应该如何分析、具体的步骤，以及一些实例作为few shot。最后再算一个平均值作为 overthinking score。

图 5 表明，Claude3.5 作为 judge 和人类算出的 overthinking score 是强烈正相关的。但绝对值还是有一些差别的，大概差了两倍（坐标轴具体数值）；

表 2 表明：推理模型的 overthinking score 更大一些，这也不奇怪，推理模型本身强调的就是多做推理。

图 6 比较的是 overthinking score 和模型大小的关系，也有一个强烈的负相关，模型越小，overthinking 越厉害一些；

-----

**解决思路：**

![[Pasted image 20250331155315.png|400]]

图 3，o1_high, o1_low，表示推理能力的设置高或低。

Lowest overthinking@k，在 agent 产生 internal reasoning 的时候，让它产生几个不同的备选方案 比如说 1-4，选择 overthinking score 最小的，纵轴比较的还是成功率。

1. 当只产生一种答案的时候， o1_low 的成功率大概在 21%，o1_high 在 29% 左右
2. 但如果让模型产生 2 个备选答案，o1_low 的 pass@k 就能到 32%，用 lowest overthinking 方法能够到 27.3%，这样做的好处是，减少成本。

o1_high 在这些问题上面的开销是 $1400, o1_low 只做一次是 $400，两次是 $800，所以用 lowest overthinking@k，节省 $600，这个方法 它的成本800美金，同时性能接近于 o1_high，那*为什么不能直接用 pass@k的结果*，pass@k 需要有 ground truth 去挑哪一个结果是正确的，lowest overthinking 是直接用最小的 score 作为结果，不需要 ground truth 。

---

作者最后还讨论了两个非常有意思的现象：

* 图 1 中，o1_high_FC 对比 o1_high，加入了 FC，overthinking score 要小很多，也就是说，FC 使得模型倾向于更多的去跟环境互动而不是去做 internal reasoning
* R1 是一个推理模型，但 overthinking score 却非常的小，甚至远小于 o1_high，接近于带 FC 的模型，作者的解释是很可能跟 deepseek 所使用的 RL 训练有一定的关系

**总结：**

作者总共提出了三种解决方案：

1. 用 smallest overthinking score，从 k 个结果里面进行选择
2. 加入 function calling 的能力
3. 使用 deepseek 这种 RL 的方法，去加强模型的推理能力

但最后两点，作者并没有做任何的实验，更多的是一种猜想。