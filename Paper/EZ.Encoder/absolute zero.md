
零数据训练 AI，自己出题，训练自己，RL 下新的 AI 学习范式，自我进化中的 AI 产生坏心思

[# Absolute Zero: Reinforced Self-play Reasoning with Zero Data](https://arxiv.org/abs/2505.03335)



------

* 在 deepseek R1 中，其实是使用了人类的数据 + 人类的标注(ground truth)，去掉了 CoT；R1 之前需要人类标注 CoT 进行训练，削弱了模型对于人类先验知识的依赖；
* TTRL：Test Time RL，去掉了 ground truth，test time 做多次的 sampling，然后进行 majority voting，对那些 majority voting 一致的结果当成 ground truth；
* Absolute Zero（AZR）更进一步，连数据都不需要

AZR 主要提出来一个 *思想是什么* 呢?  利用 LLM 自己产生数据，用这个产生的数据来训练它自己，但 AZR 所有的工作全部都是基于 coding 这一类的数据或者是 task 上面进行的，

**背景 Paper**：

Paper：Executable Code Actions Elicit(激发，引出) Better LLM Agents（2402），Agent（LLM（+memory） + tools）能与环境互动，

传统的方法  AI agent 要使用工具，都是用 JSON 的方式和环境进行互动，而 CodeAct 就是针对 tools，提出来了一个理念，不需要使用 JSON 形式， 而把所有和环境互动全部都用 code 的方式来实现，相当于把 tools 这一块给统一了。只要 AI agent 会写代码，理论上来说，它就可以使用或者是实现很多的工具了。因为在数字时代，很多的任务其实都可以通过代码来和环境进行交互的

json 的问题：生成的时候有可能格式会出错；其次可能会需要多个步骤

那同样的任务，只需要用大语言模型去生成一段这样的代码就全部搞定（参见图 1），图 1 下面两个图，左侧是成功率，在使用程序作为 action 的时候，成功率是最高的；右图展示程序所需步数最少

引言最后有介绍，让 agent 直接写代码有哪些好处？

1. 现在 Python 用的比较多，这种 AI agent 可以非常容易的和 Python 的环境进行整合；
2. agent 也可以用一些已经存在的 package，可以理解为是已经定义好的工具，可以直接调用；
3. 现在的 LLM 本身就已经有写 code 的能力，用代码作为 action 的方式可以很容易使用各种工具；
4. 代码本身就很容易对这种 workflow 进行控制，对于解决那些复杂问题，比 JSON 要好很多

HuggingFace 的 [smolagents 框架](https://huggingface.co/docs/smolagents/index)，里面就专门的提到了，对于 agent 的实现就是 code agent.


Paper: Visual Sketchpad: Sketching as a Visual Chain of Thought for Multimodal Language Models (2411)，让模型能够用视觉的草稿本，来产生视觉的思维链，这是对 codeact 的一个延伸，利用这种 visual sketchpad，本质是要这个 agent 去写程序来实现，github 有示例

---

**摘要：** 最主要的问题就是目前数据需要有人的 supervision，费时费力；AI 像超过人类，靠人类数据去训练是比较困难的。self-evolves 也就是可以自我进化，不需要外部的数据，自己训练自己；最后提到，在没有使用任何外部数据的情况下，已经能达到 SOTA 的性能了，甚至超过了那些需要使用人类数据的方法。

图 2 是目前的三种范式。Absolute Zero 就是在整个训练方法里面压根就没有人类，这个里面有两类AI agent，第一类 AI agent（proposer） 会产生学习的目标，第二类（solver）就去学习这一类的目标。

图 3 是该方法的示意图，大致就是 LLM 负责两个职责，左边负责出题，右边负责解题。各自产生一个 reward，就是 RL 的训练信号。

我们先来看一下这两个 *rewards 具体是怎么定义*的？这里其实有三个部分，除了上面两个，还有一个 format reward，与 R1 中类似。

* 对于 propose reward，最重要的就是题目不能出得太难，也不能出的太简单，公式 (4)，如果 solve 的解题率是 0 或 1，就是太难或太简单，则 propose reward = 0；如果学生能解决部分部分，得分就是 1-解题成功率。即 propose reward 可以理解为解题失败率或题目难度。
* solver reward 就是和 ground truth 对比
* 最后综合起来，考虑答案和格式

再来具体的看一下 *proposer 和 solver 具体是怎么学习的*，作者在这里把整个过程写得特别的复杂：

作者把这个训练的过程分成了三类：

* deduction（演绎）：给定 I,P -> O
* abduction （朔因）：给定 P,O -> I
* induction（归纳）：给定 I,O -> P

其实核心的思想就是，对于一个程序，可以把它分成一个 triplet，这里的 triplet 的意思是 

* 第一，这个程序一定会有一个输入 input，I
* 第二，一定会有一个程序本身，也就是 program，P
* 第三，一定会有一个 output，O

任何程序可以看作 \[I,P,O\], 整个训练过程，就是用其中任意两个来预测第三个

那这些 LLM 的输入和 LLM 的输出，也就是这个具体的训练数据该怎么产生？

------

deduction（演绎）：给定 I，P，预测 O，对于 proposer，这样的数据如何产生呢？首先还有 k 个reference examples，也就是 k 个例子，这 k 个例子可能是之前任务所产生的，我们可以用这 k 个例子作为 in-context learning，给 LLM，让 LLM 产生一个类似的 (p, i) pair，然后实际运行，如果没有任何语法错误，就会得到 O；对于 solver 来说，就可以利用 proposer 产生的这个 output 作为ground truth。（图 18 有具体的 propose deduction task example，图 23 是 solve deduction task example）

其他两类也是类似的。但在 abduction 的 solve 中，可能有不同的输入都可以得到这样的输出，实际是 slove 产生出 input 后，再次扔给 P，实际跑一遍，拿 output 与 propose 产生的 output 比较。（详见图 7）；

Induction 的 propose 是从前面 deduction 和 abduction 任务中生成的程序中采样，以及生成输入和 message（类似任务描述，因为只给定输入输出，可能不清楚是什么任务）并实际生成输出，propose 实际是产生了 10对 input-output；（详见图 21）；solve 则需要根据 message 以及 input-output（5对） 产生 code，然后用剩下的 5 对用于测试 code

然后再回头看图 4，就很容易理解了。

具体 RL 算法使用的 reinforce++，与 grpo 类似

-------

**表 1：** 我们来看一下结果，作者在这里比较了几类模型，第一类就是 base model，主要是Qwen 2.5 系列的各个模型，还有一类就是 Zero-style reasoners，其实就是 DeepSeek R1那一类的模型，zero-style 指的是没有思维链的标注，里面又分两个子类，第一类就是在coding 数据上面训练过的，第二类就是在math数据上面训练过的，所以这些模型它是有训练数据的，最后就是他们自己的模型，没有使用任何的训练数据。他们自己的方法也有两个子类，base 是从 Qwen 2.5 7B 这个 base model 用他们的方法训练出来的；coder 是从 Qwen 2.5 7B coder 模型加上他们的方法训练出来的。

这里的 benchmark 主要是两类数据集，和 coding 相关的和跟 math 相关的。

* 最后一列，用了 AZR 这个方法之后，相比较初始的 model 性能的提升分别增加了 7 和 10.2，self-play, self-involve 自己训自己的方式，就能把性能给提高这么多点；
* 可以看到从 coder 模型开始训练就最终的性能要好很多，这从侧面也反映出来这个模型的基本能力，尤其是 coding 的能力，对于他们这个方法非常的重要
* 还有一个很有意思的点，虽然他们没有用任何的人类的训练数据，但是他们自己生成的数据是在 coding 这个领域类的，结果对数学任务也能大幅涨点
	* 作为对比，zero-style 的这类，在 coding 上面训练，结果 math 任务涨点不多；同样 math 上训练，coding 上涨点不多，甚至下降


有几点值得讨论：

* human prior 的影响，这种人类的训练数据 coding data 或者 math data，其实都可以看成是一种 human prior，比如说在 coding data 上面训练过，再运用到其他的 domain，比如 math 上，跨领域的性能要么提高比较少，要么性能反而会下降；这也从侧面验证了The Era of Experience 里面提到的尽量少依赖 human；
* AZR这篇 paper 给我的一个感觉就是它似乎实现了 free lunch，在没有任何人类数据的情况下，它就能凭空的提升这个模型的性能？

Llama 3.1 8B，"Uh-oh moment"，这是一些不好的瞬间，不安全的，缺少人类监督，可能越来越邪恶。



