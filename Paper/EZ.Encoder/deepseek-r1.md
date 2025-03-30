
四个阶段：

* 阶段一：
	* RL 的历史，DeepMind 的 AlphaGo 系列；
	* LLM 训练，OpenAI GPT 系列的发展
		* 训练范式，DeepSeek-R1 一个比较大的创新点，就是从某种程度上颠覆了 OpenAI 的训练范式，但也没有完完全全的颠覆 OpenAI 的训练范式，还是借鉴了 OpenAI 的一些训练思想 
		* Scaling Law，R1 从某种程度上也挑战了大家普遍接受的 Scaling Law
		* 涌现能力
	* CoT -> Inference Time Compute，与 Reasoning 能力直接相关的
* 阶段二：DeekSeek-R1 的摘要和引言部分（前置讲解）；方法（暂时跳过 GRPO 部分）和结论部分
* 阶段三：GRPO / DeepSeekMath
* 阶段四：Kimi，同期论文，有很多细节，

---

R1 paper 主要做了三个事情：

1. 提出 R1-Zero 模型
	* pure RL -> LLM -> CoT -> 解决复杂问题（math, coding）
	* pure RL，没有 SFT，也没有 RM
2. 在 R1-Zero 的基础之上，提出 R1 模型 
	* SFT + RL -> LLM -> CoT -> 解决复杂 / general 问题
	* 把 CoT 数据和之前的 general 数据再做一轮 -> R1
	* 最开始的 CoT 数据起到了冷启动的作用
3. 蒸馏：
	* R1 -> CoT -> 蒸馏小模型


#### 1、关于 pure-RL

关于 R1 的一大贡献点，证明纯 RL 可以训练 LLM，Meta，2024.03, [Teaching Large Language Models to Reason with Reinforcement Learning](https://arxiv.org/abs/2403.04642)，更像一篇评测的文章，评测了不同的 RL 算法（Expert Iteration, PPO, Return-Conditioned RL），评测了不同的 reward（sparse、dense、RM） 

* dense  是指思维链每一步正确与否 
* sparse 是指最后的结果是否正确 
* reward model 是指训练模型去判断思维链是否正确 

此外，还做了一个比较是否使用 SFT 的评测，4.2 章节 “Results with no SFT Initialization”，这里把 SFT 作为 RL 的初始化阶段，

![[Pasted image 20250327204545.png|600]]
这里的稀疏 PPO 就是指不使用 SFT，不使用 RM，其本质就是 DeepSeek-R1-Zero，可以看到这里的结果也是相对最好的，但只是相对，绝对结果上，例如 deepseek 在 Math-Shepherd 上可以做到 90多，但这里只有 48，从绝对值上来看，其实是很差的。*所以这个思路之前就有人尝试过，但没有做出来而已*。

和 deepseek 的区别：PPO vs GRPO； 模型大小 13B vs 671B，**R1 的牛逼之处不在于他第一次提出了这一个方法，而是第一次能把别人没有做出来的这个方法给做出来。** 其中两个可能的原因就是具体的 RL 算法和模型的大小。

#### 2、训练 pipeline

R1 中采用类似 bootstrapping 的方法，其实在前面的工作里面也已经被广泛地使用了。

[LLama3 技术报告](https://arxiv.org/pdf/2407.21783)

![[Pasted image 20250327205713.png]]

Llama3 就采用了两阶段的训练方式： SFT + RL，具体怎么做 bootstrapping 的

* SFT 数据怎么来？先少量的标注的数据，类似 deepseek 提到的冷启动数据，*训练 RM*，目的是用来选数据；
* 当有了不错的 best model，就可以产生很多的思维链（k 个），用 RM 从中挑选，即 *拒绝采样*；在 R1 中，没有单独 train RM，而是直接使用 V3，因为 V3 已经非常强，直接判断 CoT 好不好即可；
* 在 Llama3 中，还加上了一些 specialized 的 SFT 数据，比如在 coding 的时候，如果有一些跟这个任务特别相关的 SFT 数据，就可以加进去，两部分 SFT 混合做微调。
* RL 阶段，此处使用 DPO，R1 使用 GRPO
* 当得到了一个最终的模型之后，又可以进行一个循环，Llama3 循环 6次，R1 循环 2 次

#### 3、蒸馏方面

[Large Language Models Can Self-Improve](https://arxiv.org/abs/2210.11610) Google, 2022, 作者在这里感兴趣的就是，如果一个模型已经有比较好的 CoT，能否将其蒸馏到一个小模型里面，5.3 章节

![[Pasted image 20250327211323.png|600]]
把 540B 具有比较好的思维链的大模型，蒸馏到 62B 的小模型上，在 GSM8K 上的结果是 57.4，比大模型的 56.5 还好。 但 540B 这个模型只是一个 pre-trained 的模型，不具有太好的思维链能力。

所以思维链的蒸馏，也不是一个什么新的想法。


#### 4、厉害之处

在于他能做出来，效果还非常不错。

#### 5、近期对 R1 的 follow up

[S1: Simple Test Time Scaling](https://arxiv.org/abs/2501.19393)，2025.02, 作者有李飞飞、Percy Liang 等，文章的评价不高 ，原因是方法比较简单，结果也不是特别好，但还是有启发的，

该论文抓住了 R1 最核心的几个要点，并加以延伸：

* R1 SFT 数据：使用了 800K 的数据，600K 推理 + 200K 非推理数据。
	* S1，不需要 600K，只需要 1K 的高质量数据就可以。*新的观点：质量大于数量*。
	* 但这 1000 个数据所涵盖的类别各种各样，而且很多都是非常难的问题，有些问题来自于斯坦福统计系的 qualify exam 
* R1：长的思维链是可以导致更好的结果，R1 是用 RL 方法，让模型自主产生比较强的思维链。
	* S1，简单粗暴，强行将模型生成思维链的过程中的终止 token，替换成 wait，强迫模型继续思考，让模型产生非常长的思维链，从而得到更好的结果，该方法称为 budget forcing。
	* 作者通过这种方式实现了 test time scaling，作者将该方法称为 sequential scaling，与 majority voting 这种 parallel scaling 是不一样的。
	* 比较了 sequential scaling 和 parallel scaling 哪个更有效一些，实验（图 4 b）显示前者好
	* **So，有没有可能结合起来？**

![[Pasted image 20250328234301.png|300]]

最后两行，就是 S1，分别是 SFT（1K）和 SFT+BF，base 是 Qwen2.5 这行的乎数据，对比而言 SFT 上提升明显，但 BF 在 SFT 上提升不大。此外，该方法和 o1-mini 等差距大，更不必和 o1 比，这也是很多人 diss 这篇论文的原因之一，结果太差。

##### 1、关于 SFT 阶段所使用的 reasoning 数据

[LIMO: less is more for reasoning](https://arxiv.org/abs/2502.03387)，2025.02，仅用 817 个高质量数据微调专门解决数学问题的模型，可大幅度提高它的性能。

模型预训练阶段，其实模型具备了很强的领域知识，要唤醒它的推理能力只需要一些非常少量的示例，但必须要非常的精确。总结下来就是两个因素：预训练强、示例要精确。 

![[Pasted image 20250328235829.png|600]]

在这个表格里面，比较了两种增强模型推理能力的基本范式，第一种是 RL，第二种 LIMO，可以简单的理解为 SFT，但使用高质量的训练数据。 在 RL 中，本质是通过搜索的方法，找到最优的推理路径，但在 LIMO 中，认为推理能力其实已经存在，但需要一些高质量示例去激活它。

基于不同的核心思想，表格接下来的内容也都会不一样，比如 RL 关注如何去找到最好的推理路径，就要使用 RL-based 的方法去进行探索，这个探索过程非常的重要；但 LIMO 最主要的问题是如何去创建这种高质量的推理数据；

在具体的实现上面也会有一些不同，基于 RL 的思想，用于计算的方法去尽可能的探索解空间；而基于 LIMO 的方法，就需要使用一些 cognitive principle 来指导如何去创建这些数据。

##### 2、关于 Long CoT 当中 overthinking 的问题 

[Do NOT Think That Much](https://arxiv.org/abs/2412.21187) 2025.02，例如问模型一个非常简单的问题，2+3 是多少？


提出了一个方法，类似于 bootstrapping 的方法，让这些模型产生很多的思维链答案，然后挑选出答案是正确，并且思维链最短的，然后去做 SFT，结果是，他们的模型产生的 token 更少但是准确率还略微的上升，思维的效率更高了


模型除了有 overthinking 的问题，还存在 underthinking 的问题，即，在思考的过程当中，可能会采取不同的思考路径，对于某些正确的思考路径，模型没有继续深入地思考下去而是浅尝则止。   这样导致的一个问题是，模型尝试了很多的方案，但最后的答案却是错的。

[Thoughts Are All Over the Place](https://arxiv.org/abs/2501.18585)，2025.02，这里他比较了不同模型在回答数学问题的时候，对于正确的问题和不正确的问题，它 token 的分布是什么情况，如果模型不带有推理能力，正确答案和错误答案 token 数目应该基本是差不多的。但对于有推理 能力的模型，对于不正确的答案，产生的 token 数目明显要比正确答案的 token 数要多很多，就说明模型可能在尝试，但是尝试很多次之后，答案仍然是错的 。

所以这间接的就表明对于这些错误的问题，模型可能在不断的尝试，但是没有深入的进去。

这篇文章里面提出的解决方案，非常的简单， 惩罚 “alternative” token （该词在错误尝试中，切换方案时会频繁出现） 的生成，在 decoding 阶段，修改它的 logits，使其概率降低，就会导致每一个思维路径，会更加的深入。



#### 6、GRPO

GRPO 本身理解并不难，它只是 PPO 算法的改进， 所以必须先要理解 PPO，PPO 也是之前 TRPO 算法的一个改进，

RL 主要有两类方法： Policy-Based 和 Value-Based ；


Value-Based：
* state -> NN -> good/bad，给定棋局，输出的是在此棋局上下棋能获胜的概率

Policy-Based：
* state -> NN -> action，以下棋为例，给定棋局，输出的是每个位置上落子的概率
* content -> LLM -> tokens
* 策略梯度
	* 在 pretrain / sft 中：θ <- θ - LR x gradient（梯度来自于 loss）
	* 在 RL 中：θ <- θ + LR x gradient x reward（reward 是指加强哪个动作，gradient 是指如何加强）
		* 例子：比如说练习投篮，先尝试各种动作，如果球进了，得到了 Reward，那相关的动作就应该被加强，所以投篮有没有进，决定了选择哪个 Action 去加强，如何加强就要靠 Gradient，让下一次投篮的时候这个动作会更多； 作为反例，假如投篮没有进，Reward 为 0，则这个 Action 不会被选中的，即使有Gradient，也不起作用了， 这就是 RL 最核心、最本质的思想 
	* 问题： RL 中的奖励是稀疏的，通过策略梯度去更新参数，最大的问题是高方差，比如赛车游戏中，稍微改变一下方向盘，车就会做非常大的改动，又想把方向盘转回来一点点，结果游戏里的赛车变化更大了，非常难控制
	* 为什么？以下棋为例，大部分时候奖励为 0，突然获得一个奖励信号，有时奖励信号很大，就会导致策略做出很大改变，网络更新方向很不稳定
* baseline 解决思路：引入 baseline，比如考试 92 分相对于平时 90 分，没有太大改进，这里的 baseline 可以理解为一种平均只，不再使用单独的 reward，而是使用相对值；
* actor-critic： 结合 Value-based，actor 就是 policy 模型，用来产生具体的动作，critic 是 Value 模型，帮助我们建立 baseline，实际训练当中两者会一起训练
	* 新的问题，二者都是 LLM，训练困难，计算量也很大
* advantage actor-critic（A2C）：advantage 可以理解为 baseline 的进阶，前面的 baseline 主要的目的是减少方差，但并不能很好的告诉策略模型哪一个具体的 action 比较好；这里的 advantage 引入了额外的信息，会告诉这个 action 相比较平均水平有多好，即在采取完某一个 action 之后，它的期望是多少，该值减去平均值，就可以得到这个 action 所对应的 advantage。
	* θ <- θ + LR x gradient x advantage（advantage 就是告诉我们哪个动作比平均水平要好，要去加强）
* TRPO：
	* 公式：ratio x advatage；这里的 ratio = π_new / π_old，也就是在训练过程当中，更新参数之后的 LLM 输出的概率和更新参数之前 LLM 的概率的比值
	* A2C 本质是 on-policy 的方法，即每做一次策略更新， 都必须用当前的策略去跑一遍环境，收集一批数据，然后用这个数据去评判这个策略好不好，从而更新策略参数，数据利用率很低
	* 想把前面的数据利用起来，有一个前提，新的策略和旧的策略，变化不能太大，那么它产生的数据可以理解为差别并不大，就可以利用起来。
	* ratio 的概念，就是在算新的策略和旧的策略它的差别是多少，其实就是一个修正值，如果用旧的策略产生的数据算 advantage，找出了某些行为可以进行加强，但是因为现在是要评估新的策略，之间会有一些差别，就乘以 ratio 去修正这个差别，这样我可以利用旧的策略产生的数据去评估新的策略 
	* 新的问题：如何保证新的策略和旧的策略差别不大呢？
	* TRPO 解决思路：引入了   KL divergence，$\text{KL}(\pi_{\text{old}}, \pi_{\text{new}}) \leq \delta$
* PPO：计算 KL 非常麻烦
	* 采用 clip 的方法限制 ratio 在 $[1-\epsilon, 1+ \epsilon]$
* GRPO：value 模型非常大，很占计算资源，砍掉
	* 拿掉 value 模型，怎么算 advantage？
	* 例如对于一个数学问题 Q，让模型产生非常多的 CoT 答案，64 个，分别算 reward，求均值， 再计算 advantage，就是 (reward - mean(r)) / sd(r)，处于标准差，相当于做归一化
	* GRPO的核心思想就是通过这种归一化之后，看哪些 action 是好的 action，从而把这些 action 拿出来作为 advantage


![[Pasted image 20250326233808.png]]
$$\begin{equation}
\begin{split}
    \mathcal{J}_{GRPO}(\theta) &= \mathbb{E}{[q \sim P(Q), \{o_i\}_{i=1}^G \sim \pi_{\theta_{old}}(O|q)]}  \\
    & \frac{1}{G}\sum_{i=1}^G\frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \left\{ \min \left[ \frac{\pi_\theta(o_{i,t} | q, o_{i,<t})}{\pi_{\theta_{old}}(o_{i,t} | q, o_{i,<t})} \hat{A}_{i,t}, \text{clip} \left( \frac{\pi_\theta(o_{i,t} | q, o_{i,<t})}{\pi_{\theta_{old}}(o_{i,t} | q, o_{i,<t})}, 1 - \epsilon, 1 + \epsilon \right)  \hat{A}_{i,t} \right] - \beta \mathbb{D}_{KL}\left[\pi_{\theta} || \pi_{ref}\right]\right\} ,
\end{split}
\end{equation}$$



#### 7、过度思考、重复生成

DeepSeek-R1，在回答 “树中两条路径之间的距离” 的问题是，耗时 430s， 产生了很多的思维链，但似乎有重复。

早在 gpt 系列，就已经发现存在重复序列的问题，这类问题称为文本退化（neural text degeneration），LLM 为什么会产生一些莫名其妙的重复序列，这个问题其实并没有一个明确的答案。领域类主要是从三个方面在讨论可能的原因和解决的方案

* data
* training
* decoding

R1 和早期 gpt 系列所展示的重续序列问题还不完全一样，R1 展现的问题是思维链过长 并且有一些疑似的重复思考过程，kimi k1.5 的论文里，同样使用了 RL 增强模型的推理能力，但不同的是 k1.5 的使用了 length penalty，鼓励模型产生比较短的思维链而惩罚长的思维链。

**为什么 LLM 会抽风？**

##### 7.1 可能原因一：data

如果数据当中有很多重复的序列，那么模型在这种重复的序列上面训练，自然就会产生很多重复的生成。

[The RefinedWeb Dataset for Falcon LLM](https://arxiv.org/abs/2306.01116) 2023，

![[Pasted image 20250329174700.png]]

先来了解一下 LLM 在准备数据的时候，大概是一个什么样的流程。这个图展示的是 common crawl  整个数据处理的流程，首先要把一些不好的网站去掉，然后提取文字信息，鉴别语种，灰色部分表示被去掉的数据部分，接下来很重要的一个点就是要做 “repetition removal”，比如同一个网页多次爬取，本身的内容有重复，不同网页之间有 copy 等；再然后会做进一步的 filtering，以及两个非常关键的 deduplication 步骤，fuzzy deduplication 和 exact deduplication。

repetition removal 会去掉很多的数据，deduplication 也会把数据几乎要砍掉一半，这就间接的说明从网上爬下来的数据里面，有非常多的重复的数据。deduplication 其实是很困难，因为很多的重复，它并不是每个字都完全一样的重复，想把训练数据当中的重复数据给清理干净并不是很容易的。

[Deduplicating Training Data Makes Language Models Better](https://arxiv.org/abs/2107.06499)，Google 2022，这篇文章的核心观点就是，对于 LM，如果我们能把训练数据做去重，这样的数据训练模型可以提高模型的性能。

因为数据特别大，去重工作非常耗时耗力，没有办法 100% 的去重，那如果在这种有重复的数据上面训练模型会有什么的影响呢？

[Repetition In Repetition Out](https://arxiv.org/abs/2310.10226) 

这篇文章就从数据的角度讨论了一下neural text degeneration产生的原因 它的标题取得非常有意思 叫repetition in 叫repetition out 这是仿照gapage in gapage out 我们先来看一下图一 左边这个是人类产生的文字 右边这个是模型产生的文字 人类的文字像这种蓝色高亮所表达的本身也就会存在一些重复的序列 如果模型在这样的数据上面训练的话
              
                  06:18
                  它产生的文字就会存在着这种重复 比如说像这句话就重复了两遍 然后在下面这个又重复了一遍 作者接下来就在这里 定量的分析了一下训练数据当中的重复内容对于模型产生文字有什么影响 横坐标表示的是人类的文字 然后这里作者创建了一个指标叫rep2 可以简单理解为重复的程度 这个值越大 表明文字当中重复的内容越多 纵坐标表示是模型在相应的人类文字上面训练之后产生的text 同样这里也是用rep2这个指标来衡量的 这里不同的颜色表示的是不同的数据集 比如像openweb、 pubmed、 arxiv 作为训练数据的人类文字当中的重复程度和模型生成的文字当中的重复程度是正相关的 换句话说 如果训练当中文字的重复内容越多 模型训练后生成的重复文字也越多 训练数据当中的重复内容会造成模型也产生重复内容
              
                  07:24
                  但是大家要注意一下 这里的scale是不一样的 对于人类的文字 最大的rep2只到十八 但是对于模型的生成的文字 最大的rep2能到八十 模型在有重复内容的人类文字上面训练之后 把这种重复给放大了 作者这里又做了一个非常有意思的实验 就是看了一下scaling law和模型生成重复内容之间的关系 横坐标是模型的大小 纵坐标是rep2 这个指标 可以看到模型越大 rep2的值是越来越低的 也就是说模型变大了之后 生成重复内容的程度是在降低的 也就是生成这种重复内容的问题在小模型当中更严重一些 但是同样 大家可以看到 即使把模型scale到七十个m 它的rep2的指标仍然在差不多五十左右 还是比人类文字的十八要大不少 所以基于这两个图 我们可以下一个结论 就是数据当中的重复序列会影响模型的生成 导致模型生成重复的序列 解决方案自然也是尽可能的去掉训练数据当中的重复序列
              
                  08:30
                  第二个解决方法就是把模型做大 像第二个图展示的 这篇paper作者提出的一个方案是使用dropout 在训练的时候加上一个mask 遇到重复序列的时候就把对于这些重复序列的attention给mask掉 就类似于在做causal attention的时候, 对attention所加的triangular mask 想法比较简单 我在这里就不具体展开了 除了训练数据里面有重复的内容会导致模型产生repetitive sequence以外 下面还有一个很重要的原因 就是这个decoding的过程 我先给大家一些基础知识 大语言模型是如何在做decoding的 假如说我们有一个大语言模型 给他一个context 比如说我想他产生欢迎来到ez.
              
                  09:19
                  encoder 如果我给他欢迎  这个大语言模型他的输出并不直接是来到的来这个字 这个大语言模型这里输出其实是一个probability distribution 什么叫probability distribution呢 就是对于你的vocabulary里面的每一个token 它都会预测生成一个probability 比如说这个时候产生来的probability是零点一 产生你这个token的probability是零点零五 类似的产生我这个 token的probability是零点零六等等等等 所以这整体就叫一个probability distribution over all the possible tokens  这个时候如何从这所有的token当中去选择一个token作为输出呢 最简单的方式就是用greedy decoding 就是你选取probability最大的那个token即可 假如说在这所有的token里面来的probability是最大的 那么我们就选来就可以了 这个就叫greedy decoding 那你也可以用pure sampling的方法 pure sampling的方法就是根据这每个token的probability
              
                  10:23
                  按照probability去选取 也就是说对来我有零点一的概率选到他 对于里有零点零五的概率选到他 这样我随机采样的时候就有一定的概率采到来 也有一定概率 虽然概率比较小 采到其他token 比如说你和我 还有一种方式叫topk的方法 topk就是基于greedy decoding和pure sampling中间的一种方法 因为greedy decoding可以看成k等于一 也就是每次都选probability最大的那个 pure sampling可以看成k等于the token size 也就是在所谓的token里面去采样 top k就比较容易理解了 就是我选取k个probability最大的tokens 然后在那个里面进行采样 举个例子 如果这个时候k等于二的话 并且最大的两个probability是来零点一和我零点零六的话 那我就是从这两个token里面进行随机采样 基于这种topk的方法 后面又有人提出了一种叫topp的方法 具体想解决 一个什么问题呢 就是topk里面这个k的选取的问题
              
                  11:27
                  这个k是一个人为指定的数字 但是在大语言模型进行decoding每一步的时候 它的这个distribution是不一样的 有的时候distribution可能会比较集中 大部分的probability都集中在少数几个token上面 有的时候在某些decoding的步骤当中 这个probability distribution可能会比较平 probability集中在大多数token上面 这个时候怎么选k就非常的关键了 对于这种probability distribution特别集中的情况下 如果你这个k选的太大 举个例子 比如说k取到一百 假如我的vocabulary size比较小 举个例子size等于一百五 这个时候你很有可能就会取到这些distribution tail上面的tokens 因为你的k取的太大了 这样会导致的一个后果就是你会取到那些probability其实非常低的头tokens 这样就会导致你生成的结果会出现一些不连贯 非常奇怪的tokens 如果decoding的时候probability distribution是比较平的 而这个k又选的太小的话 比如说这个时候选的是二二
              
                  12:32
                  那么你的decoding选取的token都会集中在这个probability最大的那几个token上面  这样就会导致产生的结果没有太多变化 很容易产生重复的序列 所以nucleus sampling的想法就是我动态的来产生一个k 在每一个decoding的过程当中 假如我有每个token的probability从p1一直到pn  从这个最大的probability开始累加算一个cumulative probability 当这个累加的值大于我们事先设好的阈值p的时候 我们就停止 然后选取这其中的k个token做采样 这样的话 如果你的distribution是上面这种情况比较集中的话 那么你算cumulative probability的时候很快就能超过你预先设好的这个p 所以这个时候k就会比较小  同理如果你在decoding的时候 probability distribution是比较平的 这个时候你在算cumulative probability的时候会选中比较多的tokens 所以这个时候的k就会比较大 这样就完美地解决了前面提到的top k的问题
              
                  13:38
                  使得能够动态地生成k 这就是nucleus sampling的核心思想 那前面这些都是基于sampling的方法做decoding 还有一种是基于search的方法去做decoding 叫beam search beam search就和前面sampling的方法不一样了 因为sampling的方法是有随机性的 那search的方法整体的结果是确定的 beam search具体是怎么做的呢 我们就以b等于二举例 假如我的token只有四个abcd 从start token开始 我生成了四个token abcd  beam size是等于二 这个时候我就选取probability最大的那两个token 假如说这个时候是a和c 那我就把其他的token都扔掉 在第二步生成的时候 基于前面生成的每一个结果 我都再生成abcd 基于c我也再生成abcd 这个时候我在这所有的path里面 也就是aa ab ac和ad 还有cacbcc和cd里面选取两个概率最大的path 这个概率是用log likelihood也就是probability的乘积来表示了
              
                  14:46
                  假如这个时候a b这个path和c c这个path log likelihood是最大的 那我们就把其他的path都丢弃掉  同理在基于这个之前的结果之上继续再做生成并且search 这个时候我的path就变成了aba abb  abc abd以及ac a acb acc和acd 然后我在这所有path里面选两个log likelihood最大的path 假如这里我选到了abc和abd 那么剩下的都可以抛弃掉了 这就是beam search的基本思想 这个参数b就是beam宽度 就好像我们拿着一个手电筒 这个手电筒能照出固定宽度的光束 我们就每走一步的时候用这个有固定宽度的手电筒看一下前方哪条路是最好 并且每次我们只能看一步 走完一步之后再用这种固定宽度的光束去走下一步 beam search一般用在translation,summary 这些需要稳定结果的任务上面 那对于这些sampling的方法 因为它每次产生的结果会变化 所以这类的方法一般用在聊天
              
                  15:50
                  chat或者是writing这些需要一定创造力的task上 这里稍微给大家做一个小结 因为后面会基于这些知识给大家讲一篇paper 大语言模型在做decoding的时候 有两种大的方法 第一种是做sampling这种方法产生的结果 会具有一定的随机性 第二类方法是做search 这类方法产生的结果基本是固定的 其中的代表方法就是beam search 在sampling的方法里面 根据这个k的大小 我们可以分为几种不同的方法 当k等于一的时候 我们叫greedy encoding 也就是我每次去产生的probability distribution里面 probability最大的那头 那如果我把这个k放宽一点 比如说我选top k 也就是说在probability distribution里面选k个probability最大的token  然后在那k个里面再做sampling 那如果我们再把k增大一点 让k等于所有token的大小 也就是我在所有的token里面 根据它产生的probability进行sampling 这个就叫pure sampling
              
                  16:53
                  pure sampling有一个比较大的问题 它会采集到那些probability特别低的token 因为虽然probability比较低 但并不代表它不能被sample到 这样造成的一个后果就是它产生的内容会非常的诡异不连续 在topk的基础之上 后续又有一篇paper我们马上就会介绍 提出了一种叫topp的方法 这个topk和topp的区别就是 topk的方法需要对这个k进行人为的设置 这相当于是一个超参 但是根据每一次decoding的时候probability distribution的形状不同 这个操餐不应该是固定的 我不再手动的设定这个k 而是设定一个阈值 这个阈值是用来算累积的probability 把所有token的probability从大到小进行排序 然后把它的probability进行累加 这个时候所选择出来的token就是我们下面要sample的token 这样做的一个好处就是它的这个k是dynamic的 是变化的 但是这里还有一个小的知识点叫temperature 对于一个probability distribution来说
              
                  17:57
                  可以用一个temperature参数去改变它的形状 我们就把这个temperature叫做t 如果t等于一的时候 是不改变这个probability distribution的形状的 如果t小于一的话 是会把这个probability distribution进行压缩 也就是让probability集中到更少的一些token上面去 如果t大于一的话 会把这个probability distribution拉的更平 这样让probability能够分散到更多的token上面去 注意这个对于temperature调整的这个方法和上面讲到的decoding的方法是 两个相对独立的方法 所以一般的做法是我们可以先调节temperature 也就是adjust这个distribution的形状 然后再做sampling 因为sampling是要基于probability distribution的 当我们用temperature的方式改变了distribution之后 这个sampling的结果也会随之改变 就是这两部可以联合起来使用 那我们就来看一下这篇发表在二零二零年的提出nucleus sampling的这篇文章 作者在这里展示了两个用gpt2去生成文字的例子
              
                  19:03
                  给定一个context 作者用了两种decoding的方式去生成文字 一种是beam search 另外一种用的是pure sampling beam search是一种比较稳定的decoding的方式  用这种decoding的方式会产生一些重复的序列 像这里蓝色所表示的 用这种pure sampling产生的结果 虽然没有重复的序列 但是产生的文字过于跳跃 缺少一定的连贯性 作者在这里就想表达 这两种decoding的方式都有各自的问题 beam search容易陷入局部最优 从而使生成的时候困在某些词语或短语上面 也就是容易产生这些重复序列 而对于这种比较奔放的pure sampling的方式 从所有的token里面进行采样 虽然避免了重复序列的问题 但是产生的序列缺少连贯性让人类看不懂 为什么大语言模型在beam search条件下会产生这些重复的序列 他们这里的做法是比较了一下beam search产生一段话的probability 和人类的实际的文字通过大语言模型之后的probability是多少
              
                  20:11
                  注意这里beam search产生的文字有大段的重复 这里横坐标就是每个token的index 纵坐标就是每个token所产生的probability 蓝色的就是beam search的probability beam search它产生的这个文字的probability其实一直都很高 基本在1附近 但是人类的文字通过大语言模型后 它的probability是浮动很大的 并且它的probability也不是最大的 这说明beam search作为一个搜索算法 它是起到了作用 因为它搜出来的答案确实是probability最大的 只不过人类产生的文字在大语言模型的心中probability并不是最大的 这个里面有一个潜在的原因 就是大语言模型它是用next token prediction的方法 训练出来的 所以大语言模型对于文字的理解本身就是具有一定的局限性 倾向于产生local的文字 人类产生的文字是连贯的 并且整段文字是有人类想要表达的中心思想 所以人类的句子更多是偏向global的 而大语言的这种local的性质
              
                  21:16
                  就会导致它容易产生一些repetitive的输出 作者在这里展示了一个图 系统的比较了一下不同的decoding方式所产生的结果的差异 给定一个人类数据的开始 也就是webtext 用这些不同的decoding方法去做生成 最后是人类数据下半部分 也就是相当于作为一个ground truth的reference 我们可以比较一下这些模型产生的文字和人类的文字 图中蓝色高亮的文字表示模型产生的文字是重复的 而红色高亮的文字表示这些文字是一些非常奇怪的文字 没有连续性 第一个beam search方法 也就是我们没有从distribution里面去做random sampling 这个方法生成的文字就是有很多的重复 也就是蓝色的这些 如果我们用pure sampling的话 也就是在每一个decoding的步骤当中 从所有的token里面进行采样的话 它会产生很多这些不连贯的 很奇怪的文字
              
                  22:19
                  如果我们接下来调整一下temperature 也就是用一个小于一的temperature 把原本的distribution进行压缩 让它的probability集中到更少数的token上面 这样做的一个目的就是让这些tail上面 也就是比较小probability的token它的probability更低 就不容易sample到那些很奇怪的token 可以看到 通过这样的方式所产生的文字的连贯性变得更好了一些 因为红色的部分变少 接下来就是用topk的方式进行decoding 这个地方k选的是六百四 也就是选六百四十个probability最大的tokens 从这六百四十个token里面做random sampling 这种sampling的方式是既有一些不连续性 也会有一些重复的序列产生 如果我们在这个基础之上把k变小 但同时把temperature也变小 备选sample的tokens变得更少 因为你的k就已经很少了 只有四十个 看到不连贯性有所降低 红色的变少了 但是这种重复的现象又变多了
              
                  23:22
                  前面的这些方法 要么就是有重复 要么就是有一些不连贯的输出 不论你如何调这个超参 结果都会或多或少的有一些问题 我们来看作者所提出的这种nucleus sampling的方法 这个输出的文字就跟ground truth比较接近 它没有重复的序列 并且不连贯的 也就是红色的部分非常的少 只有几个单词 所以这种nucleus sampling在现在的大语言模型decoding的时候用的挺多的 这篇paper作者就是想要改进这些前面的decoding方法 beam search主要是一个基于search的方法 其他这些都是基于sampling的方法 从这个结构我们就能看到 基于search的方法 它比较容易产生repetitive sequence 而基于sample的方法 它比较容易产生incoherent的问题 这篇paper所提出的nucleus方法 主要就是对这个topk方法一个简单的改进 这个图展示的主要就是topk方法所面临的问题 左边展示的是一个flat distribution 右边展示的是一个peak distribution 对于flat distribution
              
                  24:24
                  也就是它的probability分布在比较多的token上面 peak distribution也就是它的probability分布在少数的几个token上面 我们先看flat distribution 这里表示的是一个输入 然后这里表示的是所有decoding的token以及它对应的probability 如果我们用top k的方法 就是在这个里面选择probability最大的那k个token 然后做sampling 这里就有一个问题了 如果你的k取的比较小 假如说我这个时候k只取二的话 我只从这两个最有可能的token里面进行sample 这样导致的一个后果就是输出有很多的重复 因为他只能从这两个distribution里面采样  相反对于peak的distribution 如果我k的取得太大 比如说我k等于二十 那top k的方法会在这些token里面进行采样 那就有一定的概率会采样到这些probability比较低的token 比如说我就有可能会采样到这个burning 这样的话它产生的整个文字就是i ate the pizza where it was still burning  翻译过来就是当披萨还在着火的时候 我就吃了它
              
                  25:26
                  所以这是一个很奇怪的输出 原因就是因为k选的太大  我们采样到了这些小概率的token 也就是前面提到的incoherent的问题 nucleus sampling 或者也叫top p的方法 我在这里不设置一个人为的k 而是设置一个probability p的阈值 比如说t选择零点九 那么要做的事情就是把这所有token的probability从大到小进行排序 然后进行累加 一直加到这个阈值 比如说零点九 那么这种情况下有可能只有前面四个token被选中 所以接下来就只从这四个token里面进行采样 对于左边这个例子 因为每一个token的probability都相对来说比较小 这个时候我就需要累加到很多的token才能达到那个阈值 所以这个时候采样的token就会比较多 所以这样就达到了一个目的 在nucleus sampling里面 这个k是变化的 随着不同的distribution的形状 这个k可以大也可以小 这样就可以解决repetitive sequence以及incoherent的问题 这就是这篇paper的核心思想 那为什么这个方法叫nucleus sampling呢
              
                  26:29
                  你可以把这个累加的过程理解为就是在找probability mass probability集中在哪些tokens上面 可以把它理解为一个核 也就是nucleus的意思 这样的核给找出来 前面看了一个定性的结果 我们再来看一个定量的结果 这里我们主要看两列就可以了 perplexity和repetition perplexity衡量的是模型对这个结果的确定性 这个值越小越好 越小表明模型对这个结果越确定  repetition比较容易理解就是这个生成当中有多少的内容是重复的 我们可以看到人类的文字perplexity大概在十二左右 repetition是比较低的 只有百分之零点二八 对于greedy和beam search这两个方法 这个perplexity都是非常低的 只有一点五左右 说明模型是非常确定它的输出的 但这也带来一个问题 这样的输出里面的repetition非常的多 比如说对greedy的方法有百分之七十三的结果都是repetition beam search有百分之二十八的结果都是repetition 如果使用sampling的方法之后 repetition是极大的降低 但是同时也引入了一个问题
              
                  27:33
                  就是模型对生成的结果不是那么确定了 perplexity再升高 这也就是对应前面的定性的结果 模型生成的文字有一些连续性的问题 我们再来看一下这个nucleus的结果 注意这里我们是想它生成的结果跟人类生成的结果越接近越好 我们可以看到nucleus方法的perplexity和人类的结果是非常相似的 一个是十二 一个是十三 repetition也是非常相似的 大概在零点儿三左右 所以说明这个nucleus方法产生的文字和人类是非常接近的 就人类的这个结果再稍微展开讨论一下 这个结果的计算方法 就是把人类的文字输入到模型当中 让模型输出这些人类文字的对应的probability 然后通过这些probability来算perplexity 从而衡量人类文字对于模型来说它的确定性如何 可以看到人类的文字对模型来说这个 确定性并不是最低的 比如说相较于greedy或者是beam search的方法 但是我们在一般的机器学习的时候去最大化模型的输出的probability 比如说我们在做classification的时候
              
                  28:39
                  我们会选择probability最大的那个结果 而往往这样的做法是可行的 能够得到正确的答案 但是在文字生成这个任务上面 我们就发现这个思路似乎不可行了 因为对于模型来说 人类的文字probability就不是最大的 所以你再用去最大化probability的那些方法去做生成的话 那个结果就会很差 和人类的自然的文字就会有很多的不同 这是一个非常有意思的现象 大家可以思考一下人类文字为什么会这么的特殊 为什么人类在写作的时候选取的单词在模型看来不是那个probability最大的单词 之后我们再来看一下这个图 作者展示了一个非常有意思的现象 这里展示的是一个重复的句子i don't know 这个句子重复了两百次 我们就看这个图就好了 横坐标表示的是这个里面重复的token 纵坐标表示的是probability 所以这里表示的是第一个i don't know 这里表示的是第二个i don't know 这里表示的是第三个i don't know 我们就看这个i就好了 如果我们比较这三个i的话
              
                  29:43
                  可以看到它的probability是在依次上升的 也就是重复的越多 模型对于这个token的probability就越大 模型就越确定这是一个好的生成 类似的 对其他的token 我们也都能观察到类似的现象 比如说don't know的do 这个token也是在不断地变大了 这个现象就表明 一旦模型开始生成重复的序列之后 它的probability会越来越大 模型会越来越倾向于生成重复的序列 对于neural text degeneration的问题 我已经从data的角度给大家讨论了一下 也从decoding的角度给大家讨论了一下 接下来有一篇paper是从training的角度再解释这个问题 标题也是直接表达了这篇paper的核心就是如何learning to break the loop 就让模型学会如何去避免生成重复的序列 我们就直接来看一下这个图就好了 这个图的横坐标是重复的token的index 我们只看这个红色的柱状图就好了 纵坐标是这个token对应的probability 大家可以看到这个红色的bar
              
                  30:48
                  随着这个重复的不断的增多 这个probability是逐渐的增大的 也就是模型倾向于去重复前面生成过的token 并且在这个重复过程当中 模型会不断地增强这种倾向性 使得这个probability越来越大 所以这篇paper把这个现象取了个名字叫self reinforcement 这个现象就有点类似于模型在前面几步进行decoding的时候 一旦发现probability比较大的tokens 在后面的decoding的过程当中 就会倾向于继续使用那些token 并且将这些token的probability强化变大 有点类似于模型在偷懒 有比较确定的结果之后 后面就不断的重复使用  作者就在training的阶段提出了一个解决方案 在训练数据当中加入一些人造的重复数据 设计了一个loss去惩罚模型生成这些重复序列 在训练数据当中加入了一些反面教材 让模型不要去学习这些反面教材 通过这种在训练数据当中人为加入重复序列的方法
              
                  31:56
                  使得模型学会不要产生重复的序列 前面的几篇paper观察到了模型产生重复序列这一现象 并且也提供了一些解决方案 但是没有根本上去解释为什么模型会产生这些重复序列 这里有一篇二零二二年来自于anthropic的paper 因为paper的本意是想解释为什么现在的大语言模型可以做in context learning 所谓的in context learning就是可以在模型的prompt里面加入一些例子 模型可以直接从prompt里面你提供的这些例子学习 anthropic在这篇文章里面讨论的就是大语言模型里面attention部分有一些注意力头去专门负责做这种 in context learning的 他们把这些注意力头叫做induction heads 这个induction heads有一个特点就是他们会copy 所以我觉得这个很有可能是导致模型产生重复序列的一个重要原因 因为模型本身就有这样的机制存在 我们具体来看一下anthropic在这篇文章里面提到的induction heads是什么意思 这个induction heads它能做两个事情
              
                  33:02
                  第一个就是做这种prefix matching 第二个就是做copy  就是如果在你的prompt里面 你提供一些例子a b a b a可能是一个question b可能是一个answer 你提供很多这种question answer的例子放到你的prompt里面 然后在prompt最后你放上你你真正想问的问题 也就是这里的answer 然后模型自己会根据你前面提供的这些事例学习这个pattern 然后回答出b 那这里就涉及到了两个功能 也就是prefix matching以及copy 假如说这个是你提供的一个prefix 模型会先做一个prefix matching 因为这个地方的序列和这个地方的序列是完全一样的 所以当模型给定这个node要生成下一个单词的时候 他会先去通过attention机制找之前的prefix有没有类似的 他会发现有类似的 一旦找到之后 模型就会做copy 也就是把这个structure直接copy到这个地方作为生成 这就是prefix matching和copy的意思 所以antropic在这里给了一个例子
              
                  34:05
                  如果你的输入是the cat sat on the mat the cat 然后让模型生成下面的文字的话 因为模型有这个induction heads 所以模型会倾向于生成sat on the mat 也就是跟前面这一语是一样的 所以这里从某种程度上就解释了前面我们提到的self reinforcement 这里anthropic又进一步做了一些推广 这个induction heads不仅能做简单的copy 它还能够在语义空间做一些abstract representations 所以它copy的也不一定是 完全一模一样的例子或者是语句 比如说你提供的可能是a star b star 这个在语义空间上可能和a b比较相近 所以一旦你你提供给a之后 模型就会自己在语义空间里面进行copy 从而产生b 所以表面看起来a star b star和a b是不一样的 但它内在的工作机理还是一样的 还是使用了这个induction heads 所以我想用antropic这篇paper提到的induction heads去解释大约模型 产生重复序列的这个问题 我再给大家讲一篇paper 这篇paper是二零二五年二月份刚刚上线的
              
                  35:09
                  作者试图去找transformer里面repetition neurons 从而解释为什么language model会产生repetitions 这片paper和前面的antropic paper形成了互补 antropic那片paper可以从attention的角度解释模型为什么会产生重复的序列 这片paper从另外一个角度 也就是mlp层来解释这种重复序列的产生 这里作者主要研究的就是mlp层里面的neurons 我们就来看一下这个图就好了 这里展示的是一个重复序列 h i j k  h i j k一直重复 然后纵坐标表示的是mlp层里面的activation values 这个值越大 表示mlp层里面有些扭软被激活了 对于这些四个neurons来说 当模型开始生成重复序列的时候 它的activation是开始变大了 这就意味着在transformer里面有一些mlp的 neurons专门负责生成这些重复序列的 所以这就从非常底层的模型的架构上面解释了 模型内部就是有一些机制会产生重复序列
              
                  36:13
                  不论是在attention层还是在mlp层 上面的那些paper主要针对的是模型产生repetitive sequence的问题 R1中考思考的问题可能更多的是跟它的reinforcement learning有关系 我们就来探讨一下 我们先来简单的看一下DeepSeek-R1它这个思维链是怎么产生的 简单的说就是模型的思维链必须在这两个special token之间think以及反斜线think 所以模型在输出的过程当中 任何只要在这两个think token之间的 都会作为思考的思维链的内容 而这两个think special token是由模型自己产生的 换句话说 模型之所以会疯狂的思考 也就是模型不产生终止think的这个special token 为什么模型会不生成这个终止思考的special token呢 原因就藏在R1 paper里面的这个图里 DeepSeek作者想用这个图来表达 通过reinforcement learning 模型会自己学会生成思维链 而且是倾向于生成比较长的思维链
              
                  37:18
                  这横坐标是training的步骤 纵坐标表示的是模型生成答复的长度 可以看到 随着模型的训练 模型生成的这个答案也会越来越长 也就是他的thinking time会越来越长 但这不一定是一个好事 因为我们人类在思考的时候 其实讲究的是效率 这就好比你在考试的时候应该 是用更高效的方法去解答一个数学题 而不是追求更多的思考时间去解决一个数学题 因为考试的时间都是有限的 之所以R1会产生这个现象 是因为DeepSeek在训练R1的时候 reinforcement learning的reward并没有对生成思维链的长度做任何的限制 所以模型就钻了这个漏洞 使得产生的思维链越长越好 因为越长的思维链确实能够导致更准确的结果 但这不是我们人类真正想要的 在同时期发表的kimi k1.
              
                  38:10
                  5这篇paper里面 作者就考虑到了思维链的长度这个问题 作者就提出了一个length penalty的方法 他们观察到了一个所谓的overthinking phenomenon 也就是模型倾向于产生非常长的思维链 这样导致的一个后果就是 虽然这种长的思维链会得到比较好的结果 但过于长的思维链会增加training和inference the cost 因为它的token数实在是太多 而且这种overthinking并不是我们人类所喜欢的 所以作者在这里就提出了一个length reward 也就是在reinforcement learning里面加入一个新的reward signal 这个signal会鼓励模型产生更短的response 所以我认为DeepSeek-R1疯狂思考的问题 就是因为R1的reinforcement learning里面没有这种length penalty的reward signal   所以导致模型会倾向于生成非常长的思维链 而kimi这篇paper就观察到了这个现象 并且做了改进 大家可能会好奇 在视频最开始 我提到的让DeepSeek-R1疯狂思考的那个prompt到底是怎么发现  书中两条路径之间的距离这个问题就来自于北京大学的袁立老师的课题组
              
                  39:20
                  他们这里展示了一个图 但是没有详细的解释 我猜测他们是用了早期做prompt optimization的方法 早期chagpt刚刚出来的时候 因为模型各方面都不完善 所以需要做很多的prompt engineering 大部分的prompt engineering都是手工做的 也就是人类去不断地尝试各种prompt 看看哪个prompt能让模型产生更好的答案 有一类工作是用优化的方法 从众多的备选的prompt里面选择哪些prompt能够让目标函数最好 那我猜测这里可能也是用了一个类似的方法 简单来说就是把R1这个模型给固定住 然后来优化这个prompt 把它当成一个优化的问题 它这里提到优化的目标函数是maximizing long output sequence likelihood 也也就说这个优化问题是调整prompt使得模型最后输出的序列最长 通过这种方式找到哪些prompt能产生非常长的输出 从而就找到了前面我们提到的那个prompt,树中两条路径之间的距离 这个prompt可以让R1模型产生无止境的cot
              
                  40:29
                  直到这个cot的长度已经超过了这个模型本身的max token length constraint  最后再稍微的给大家提一下 在hugging face的generation function里面 也就是这个generation config里面有一个参数叫repetition penalty 这个参数就是用来 抑制模型生成的时候 产生重复序列的程度,1表明没有penalty 当把这个参数设为大于1的值的时候 就会惩罚模型生成的重复序列 它的原理就是如果你前面已经生成了某些token的话 在后面的生成的时候就会抑制这些poken的probability 具体的实现方式就是通过temperature那个参数 来改变对应 token probability distribution 但是大家要注意 这个参数也会引起一些问题 这个是在hugging face上面的一个讨论 这个人就举了一个例子 比如说如果你有一个句子叫united states吧啦吧啦吧啦 如果你设置了这个repetition penalty的话 在生成的时候 这个americans的这个里面的s就会被penalize
              
                  41:35
                  因为在前面us这个地方 这个s是一个单独的token 也就是s已经出现过了 所以在模型生成后面这个americas的时候 这个s又是一个新的token 并且也刚好是一个s token 这样当你使用repetition penalty这个参数的时候 这个s就会被抑制 所以你最后得到的结果就是largest country in the america 就会导致一些问题 所以大家在使用这个参数的时候要注意这个问题 今天内容差不多就到这里了 我来给大家做一个小结 在视频的开始 我给大家展示了两个大语言模型发疯的例子 第一个例子是R1产生永无止境的思考链 意识到超过模型最大的token输出序列 并且有些内容看起来是重复的 第二个例子是大语言模型 产生了很多repetitive sequence 然后我把这两个问题放在一起 撸了一些领域类关于neural text degeneration的paper 这些paper分别从data,training和decoding的方式 几个方面尝试在解释或者是解决这一类的问题 
              
                  42:41
                  最后针对R1的这种非常长的思维链问题 我们也简单的看了一下kimi k1.5里面的length penalty 我觉得这个内容非常有趣 所以我做了一期视频给大家讲解 也希望大家在看我这个视频的时候能有自己的思考 享受到思考的乐趣 那么你觉得语言模型为什么会产生这些奇怪的问题呢 你能否提出一些假设 欢迎你给我留言
              
            