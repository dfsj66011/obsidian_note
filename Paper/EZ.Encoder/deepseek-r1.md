
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


### 一、摘要和引言

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



              
                  15:01
                  这个也需要一些额外的计算量 作者接下来就介绍了一下 他们在训练R1-Zero时候使用的一个prompt 这个prompt总的来说就是非常的简单 就是要求这个模型提供reasoning process 并且放到这两个tag里 这两个tag是模型输出的special token 这里放这样一个表 他们想表达的一个含义就是说 我们的方法非常的简单 连这个prompt都是如此的简单 因为在之前有一些工作 为了让模型产生思维链 或者是做比如说self reflection 或者是self evaluation 一般是需要设计一些比较复杂的prompt 在R1-Zero的prompt里面 是没有任何的部分要求模型去产生self reflection 或者是self evaluation 这跟后面的结果就形成了对比 因为模型自己就产生了一个aha moment 接下来作者就在这里展示的是 R1-Zero和OpenAI的O1的一个比较 这个结果也是非常有深意的 因为领域内在R1这篇paper之前 大家都一直想复现O1的工作 这个结果直接就告诉读者
              
                  16:05
                  我们基本上把O1的结果复现出来了 因为R1-Zero在很多结果上已经超过了O1-mini 跟O1不相上下 我们就来简单的看一下这个结果 AIM和Math是两个数学相关的数据集 这两个是coding相关的数据集 GPQA是一个非常难的 需要人类顶级专家 比如说博士才能回答好的问题 可以看到在这些数据集上面 R1-Zero的表现是非常亮眼 基本上可以跟O1持平 除了在coding这一块要略差一些 然后作者就在这里展示了一个图 在AIME这个数据集上的准确度 也就是在数学问题上面 R1-Zero的准确度 横坐标是training的时间 纵坐标就是准确度 作者在这里用到了一个majority voting 然后是让模型产生16个response 然后做majority voting 在不做majority voting的时候 R1-Zero的性能已经逼近O1了 做完majority voting之后 R1-Zero的性能已经能超过O1了 所以通过这两个结果能够看出来
              
                  17:10
                  只用reinforcement learning去训练大语言模型 产生思维链 这个方法是非常有潜力的 在这篇paper里面做出这么炸裂的结果 并且能直接一举超过O1 这个是非常不容易的 接下来作者就展示了一个 可能是这篇paper最重要的一个图之一 也就是模型会自己学会 使用更多的思考时间 横坐标是training step 纵坐标是模型回复的程度 随着模型的训练 模型产生的答案是越来越长 所以模型在没有给它任何要求的情况下 自己学会了产生长的思维链 去帮助更好的解决复杂的问题 这里我也想稍微的提一下 在实际应用过程当中 并不是思维链越长越好 如果思维链太长的话 就会导致的一个问题叫overthinking 这会导致training和influence的cost增加 同时这也不满足人类的preference 接着作者又展示了一个aha moment 这也可能是这篇paper第二个最重要图
              
                  18:14
                  这个图所想展示的就是 模型给定这样的一个数学问题 在思考链的中间 模型突然产生了这样的一句话 wait wait, that's an aha moment I can flag here 注意这一句话不是人类加上去的 是模型自己产生的 这个现象就非常有意思 我猜测这可能有两个原因 第一个原因就是作为base model的v3模型 训练数据当中可能会有类似的数据存在 所以模型学会了这样的一些pattern 第二个可能原因就是通过reinforcement learning 让模型探索未知 产生了一些训练数据当中 所没有提供的新的预测结果 这就有一点类似于alphago 在跟李世石下棋的时候 alphago走出的那一步move 37 因为这一步在人类提供的训练数当中是没有 模型通过reinforcement learning 自己发现了这一步 所以我觉得这个跟aha moment有异曲同工之妙 但我也想提一下 大家不要过度的解读这个aha moment 觉得好像R1-Zero就有了人类的智慧
              
                  19:18
                  会像人一样思考和说话 这就像我在前面视频里面给大家解释的 大语言模型本质还是在学习 token之间的statistic correlation 这种aha moment其实也是这种 statistic correlation的一种表现 所以我不认为通过reinforcement learning训练之后 R1-Zero展现出来的人类的智慧 更别提独立的人格 作者最后在这里又提了一嘴 R1-Zero它所存在的问题 主要是两个问题 第一个就是可读性很差 第二个就是会存在language mixing的问题 他的思考经常会存在中英文混杂 并且这种思考过程人类有的时候是读不懂 所以从某种程度上来说 R1-Zero只是一个proof of concept的实验 因为它最终的结果由于这些问题 其实是没法直接用的 但仅仅是这个proof of concept的结果 已经让人非常激动了 因为它就用非常简单的训练框架 就能训练一出一个和O1所媲美的模型 R1-Zero的部分我就给大家介绍完了 在R1-Zero的基础之上 作者又提出了R1
              
                  20:22
                  如果说R1-Zero是一个proof of concept的概念车roadster 那R1就是一个真正能量产的model X 这个章节主要介绍了R1的训练流程 整个流程看起来特别的复杂 就用top down的方式给大家来讲一下 R1整个的训练流程 R1整体的训练思路 其实并没有跳出OpenAI提出的训练方式 也就是在post training的阶段 仍然是使用SFT加上reinforcement learning R1的训练总共进行了两轮 在第一轮里面 他们首先做了一个SFT initialization 然后再做的reinforcement learning 这个SFT initialization的意思就是 reinforcement learning里面 我们其实可以把language model 看成一个policy model 所以你如果直接用reinforcement learning 去train这个policy model的话是比较难的 一个方法就是你先用supervised finetuning的方法 初始化这个policy model 也就是这里说的SFT initialization 有了这个initialization之后
              
                  21:27
                  再去用RL训练就会更容易一些 我们接着按top down的方式 讲解一下这个SFT initialization 为了做这个SFT 首先我们就需要有SFT data 有了这些SFT data之后 我们才能做SFT 在R1的paper里面 作者就把这些SFT data叫做coldstart data 他们有了这些coldstart data之后 是拿这些数据finetune v3这个model 这些coldstart data又是怎么获得的呢 作者主要用了三个方法 第一个就是使用few-shot long CoT as an example 这个就有点类似于Jason Wei 在最早的那篇CoT的paper里面 所提到的few-shot的方法 也就是在prompt里面加入一些示例 这些示例是包含CoT思考过程 加入了这些范例之后 模型在回答的时候自己就会产生一些CoT 第二个就是直接prompt模型去产生带有CoT的答案 这也类似于我前面在CoT视频里面 给大家介绍的zero-shot的方法 比如说你可以在prompt里面加入一句 let's think step by step
              
                  22:30
                  当然也可以使用更复杂的一些prompt 让模型产生一些更复杂的思维过程 所以这两个方法也就是prompt engineering的方法 让模型产生思维链 这些方法在前面的视频里面 都已经给大家介绍过了 大家应该非常的熟悉 第三个方法就是用R1 zero产生的数据 但是必须经过一些手动的整理 因为R1 zero产生的数据格式会比较乱 通过这三种方式 我们就能获取一些coldstart的sft data 然后我们可以就用这些数据 在V3上面进行supervised finetuning 这一步做完了之后 就相当于对reinforcement learning里面的policy model 做了一个初始化 然后我们就可以去做reinforcement learning了 然后这里就是使用的GRPO 然后reward和R1 zero类似 有一个accuracy reward 有一个format reward 这里作者还加入了一个language consistent reward 加入这个reward的原因 是因为R1 zero它的输出经常会出现 比如中英混杂的情况 所以加入这个reward去鼓励模型 产生更加一致的语言
              
                  23:35
                  那round one这个就结束了 在round one这一步 主要的目的是增强模型reasoning的能力 但是我们在使用大模型的时候 希望模型不光有reasoning能力 还应该有一些比较general的能力 所以作者在这里又做了一轮训练round two 这里所要增强的 就是模型的reasoning能力 以及general的能力 具体的做法和前面也非常的类似 首先有一步sft 然后再做reinforcement learning 在做sft的时候 同样也有sft data的问题 有了这些data之后 就可以直接做sft了 在这个round two里面 作者并没有用前面的checkpoint继续fitting 而是在v3上面重新做sft 这个sft data是怎么产生的呢 作者首先产生了一些reasoning data 然后又产生了一些non-reasoning data 把这些reasoning data和non-reasoning data混在一起 对模型做sft 这样模型就会既有reasoning的能力 又有general的能力 这个reasoning的data 就是用前面训练好的模型
              
                  24:38
                  产生大量的数据 然后做rejection sampling 这个rejection sampling具体是怎么做的呢 他们用前面这个模型 产生了很多的reasoning数据之后 然后把这些数据交给v3模型进行打分 通过这种方式 他们总共产生了600k的reasoning data non-reasoning data是如何产生的呢 首先他们把之前训练v3的sft data给拿过来了 然后又让v3通过prompting的方式 又产生了一些新的数据 然后把这些数据合在一起 产生了200k的non-reasoning data 所以这一部分总共有800k的sft data 然后他们就用这800k的sft data 对v3进行supervised finetuning 这一步做完了之后 然后跟之前一样开始做reinforcement learning 那这一步的目的就是 在增强模型reasoning的能力的同时 也要保证模型general能力也很好 这就包括模型的helpfulness 以及模型的harmlessness 所以这一步就跟Chat GPT里面所使用的 RLHF基本是一致的
              
                  25:41
                  在这里reward用的是 除了前面rule-based的那些方法以外 还使用了human preference 那rule-based的方法主要是用 为了增强模型的reasoning能力 human preference主要是为了 模型的general能力 做了这里也稍微的提了一下 他们还用了一些不同的prompt 主要是让模型有更好的放缓能力 可以看到R1的训练 跟之前代表模型训练并没有太多的不同 也都是sft加入reinforcement learning sft加入reinforcement learning 然后R1这里做了两轮 第一轮主要是为了增强模型的reasoning能力 第二轮主要是为了增强模型的reasoning能力 以及模型的general能力 这种做多轮的方式 也不是R1第一次提出来的 像前面提到的Llama3就做了6轮 另外一点就是 在R1里面这个round1产生的这一步的模型 并没有继续做finetuning 这一步的模型主要是用来做数据的生成 也就是为了rejection sampling而使用的 所以第一步的最终的目的 就是为了产生这600k的高质量的reasoning data 有了这些data之后
              
                  26:47
                  他们直接在V3上面进行finetuning 而且这一步产生的600k的reasoning data 这个数量是相当庞大的 另外最近大家发现R1的安全性能比较差 很容易就被越狱了 主要就是因为DeepSeek在这里的RLHF 做的并不是特别的充分 可能DeepSeek在R1这篇工作里面 重点也不是RLHF 所以他们并没有花很多的功夫在这个地方 导致R1模型的安全性能不是特别的好 这就是R1一个整体的训练过程 希望大家有了一个基本的认识 稍微看一下R1的结果 这个表展现的就是各种不同的benchmark 包括一些英文的数据集 编程数学 还有中文的一些数据集 这一部分的模型就是不带reasoning能力的模型 这两个是OpenAI的O1模型 然后是R1模型 首先我想指出来的就是 这个V3模型其实就已经非常的强了 我们就看一下code和math 因为这两个已经能代表模型的reasoning能力了 整体上这个V3的模型 在coding和math上面
              
                  27:51
                  已经比其他两个模型好了一大截 比如说在math500上面 V3已经能到90了 GPT-4大概只能到74左右 好了不止一点半点 这就间接说明R1所使用的这个基模型 V3本身的推理能力就已经是非常强了 所以R1做的就是进一步加强 这个模型的reasoning能力 我们来看一下这一列 相比较V3 R1在其他的一些benchmark上面 性能进一步的提高 比如像math500从V3的90 提高到了R1的97.3 AIME从V3的39提高到了R1的79 这些提高也是非常显著的 我们再来对比一下 R1和OpenAI的两个O1模型 基本上R1是全面碾压 这个比较小的O1 mini模型 和O1模型基本上也是一个平手 我们只要看这个粗体的数字分布就可以了 第三个部分就是蒸馏 蒸馏也不是DeepSeek首先提出来的 在视频刚开始就给大家展现过一篇Google的paper Google就已经用具有推理能力的大模型
              
                  28:56
                  去蒸馏一个小模型 并且这个小模型的推理能力还不错 DeepSeek基本也是用的这样的一个思路 它们把这个比较大的R1 也就是有640几个B的参数的模型 蒸馏成几个小的模型 它们使用的是Qwen和Llama 我们就直接来看一下这个蒸馏的结果 这里模型的大小从小到大有1.5B的 7B的,一直到70B 然后这些模型全部都是Dense模型 这些蒸馏之后的模型性能也是相当不错的 比如我们就看一下Llama70B 在不少的数据集上面 性能都是最好的 甚至超过了O1 mini 所以这就给大家开了一扇窗户 如果你有一个非常强大的 具有推理能力的大模型 你把它蒸馏小了之后 这个模型的能力仍然非常强大 这样你就可以直接使用这个小模型 去做一些下游的任务 这是非常经济实惠的 后面大家基于这个想法 有很多后续跟进的工作 这里作者还做了一个实验 就是比较了一下蒸馏 和直接使用reinforcement Learning
              
                  29:59
                  我们就来看一下这个表格 这个模型表示的是一个base model 这个结果就表示直接使用reinforcement Learning 也就类似于R1-Zero的那个方法 这个模型表示的就是我使用蒸馏 这里的模型都是一模一样的 都是Qwen32B 所以可以排除模型的影响 我们可以看到这个蒸馏出来的模型 性能是比纯的reinforcement Learning的效果 是要好很多的 这也间接地解释了一个问题 为什么前人很多用reinforcement Learning的方法 去增强模型的推理能力 方法和DeepSeek这篇文章基本类似 为什么却没有做出来 很重要的一个原因就是 如果你直接在一个比较小的模型 比如说这里是32B 用reinforcement Learning的话 它的效果并不好 注意这里他们使用的是GRPO的方法 所以这也可以间接地排除 这个reinforcement Learning的算法 不论是GRPO还是PPO 可能就不是一个主要的因素 主要的因素就是这个模型的大小 所以你把这个模型换成V3的671B 它有可能就能train出来
              
                  31:01
                  第二个原因我认为非常重要的是 V3这个模型本身是有很强的推理能力的 从上面一个表格的结果我们就能看出来 所以你直接在V3的基础上 再用reinforcement Learning的方法去train 它才能有效果 这就好比于你让一个没有基础的小学生 去解决很难的问题 无论你如何push它 它的能力都不可能提高 只有它有了一定的基础能力之后 你才去push它 才会有一定的效果 第二个我觉得比较有用的结论就是 如果你的目的就是用一个比较小的模型 可能是由于你的生产环境的限制 你不能使用一个大模型 那么你就不用费时费力的去用reinforcement Learning 这种方式去增强模型的推理能力了 你应该直接用这种蒸馏的方式 换句话说就是你产生一些高质量的 reasoning数据 拿这些reasoning数据去finetune你的模型就好了 这样省时省力又省钱 所以后续有很多的paper 就是沿着这个思路继续拓展 所以这篇R1的paper主体部分就讲完了 作者在这个地方又写了一些unsuccessful attempts 我觉得对我来说这个也是挺重要的
              
                  32:06
                  DeepSeek作者其实就想打脸前面的那些文章 告诉大家前面那些方向可能都是不对的 第一个他们想打脸的就是PRM 这个在openAI在DeepMind有很多这方面的工作 当然包括DeepSeek自己 DeepSeek作者就想告诉大家 PRM的方法其实有很多的limitation 第一个就是比较难定义一个PRM 对每一步进行评价 尤其是有一些问题它是比较general的reasoning 所以你很难每一步都给出一个评价 第二个问题跟第一个问题其实是相关的 就是你如何去判断这每一步是否正确 其实也很难的 在我前面视频里面给大家介绍的 openAI和DeepMind的那两篇工作 他们都是请人对每一步进行标注 然后训了一个PRM 当然你也可以不使用人类的打分的方式 去使用这种automated的方法 比如像Math-Shepherd去自动的 生成一些中间步骤的标注 但这标注可能不是特别的准 第三个就是如果使用PRM的话 会导致reward hacking 原因就是PRM本身它也是一个neural network
              
                  33:12
                  并且是一个训练出来的neural network 一旦这种模型是通过数据训练出来的 它难免在某些场景下就会出现问题 这个时候在reinforcement learning的框架下 policy model就会尽可能的去 利用PRM的一些漏洞获得高的reward 但其实最后得到的结果都不是人类想要的 这里作者主要就是在说reward model 另外的一个用处 希望大家还记得我提过 它可以作为verifier来使用 作为verifier确实还不错 但是你要单独的训它 可能不是那么经济适用 作者这里还提到了一个失败的尝试 就是使用MCTS 我在reinforcement learning 那个视频里面也给大家提过 MCTS本质是一个搜索算法 所以你可以在inference的时候 让模型产生很多的备选答案 然后用MCTS的方法去搜哪一个答案最好 然后输出最好的答案就可以 但是在大语言模型里面有一个问题 alphago alphazero之所以能使用MCTS 是因为围棋下棋的可能性虽然很大 但还是是有限的
              
                  34:15
                  但是对于大语言模型 每次生成的token 可能性要远大于棋盘上可以落子的可能性 所以搜索空间是巨大的 想要使用MCTS这个方法 在大语言模型上面去搜索最好的答案 要困难很多很多 所以作者最后下了一个结论 就是通过MCTS这种方法 在inference的时候有一定的可能 但本身是非常challenging 最后作者在这里讲了一下 他们的limitations和future work 这几年某些deep learning的paper 后面都要提一下这个limitations 你submit paper的时候 很多conference journal 甚至强制你要填你的work的limitations是什么 如果写得太直白 会让reviewer觉得我们这个工作是有缺陷的 如果不写又不符合要求 但deepseek在这里还是非常有诚意的 提了一下R1这个模型几个重要的缺陷 第一个就是他们的general capacity 还是没有V3这个模型好 比如说像一些常用的function calling multi turn, complex role playing等等
              
                  35:20
                  这些都没有V3好 类似的在software相关的任务上面 R1的性能也不是特别的好 这个可能跟deepseek 他们没有特别的花精力去弄这一块有关系 像Llama3他们在后训练的时候 有很大一块就是搜集 跟software engineering相关的一些专有的数据 去增加模型这一块的性能 另外就是language mixing 就是R1仍然有一些language mixing的问题 尤其是当在处理其他的语言的时候 因为R1是在中文和英文上面进行训练 另外一个就是prompt engineering的一些问题 他们发现R1对于prompt非常的sensitive 也就是当你有一个问题的时候 你直接给R1形容这个问题 并且说明你要的output format是什么样子 尽量使用这种zero short的方法 而不要用这种few short的方法 因为你用few short的话 会在prompt里面加入一些例子 R1在这种情况下 它的performance会有一些的下降 所以R1这篇paper我也基本给大家撸完了 总的来说
              
                  36:24
                  我觉得R1 zero可能是这篇paper的一个亮点 R1虽然结果非常好 但它总的方法上来说 和前面比如像lLlama3的方法并没有太多的不同 这里做蒸馏的想法也不是什么新的想法 但是它的结果还是很有意义的 最后提一下DeepSeek的paper writing 我觉得DeepSeek的作者在写paper上面是非常厉害的 他们有一个很强的storytelling的能力 R1这篇paper给人的感觉要好于kimi k1.5的paper R1重点更加的突出 并且它有一个很强的逻辑线 它展示了用很简单的方法 就能训练出一个很强的模型 并且还放了一些非常吸引眼球的结果 在paper里面 比如说aha moment 如果让一个菜鸟来写这篇paper的话 可能就会把R1 zero的结果给藏起来了 因为R1 zero的结果其实并不是特别好 你直接展现R1的结果 其实更合乎情理一些 但是DeepSeek的作者从这些不好的结果当中 挖掘出了一些闪光点 写出来的paper让人觉得眼前一亮 同时R1这篇paper里面也没有放入过多的细节
              
                  37:29
                  paper里面细节太多 反而让人会抓不住重点 总的来说我觉得R1这篇paper写的是非常的棒 大家可以研究和模仿一下 它的这种storytelling的方式 第二个我想表达的观点就是 R1这篇工作虽然有很多地方 非常有启发性 但整体还是属于简单粗暴的方式 尤其是R1里面的reward设计 虽然简单容易训练 但也可能会引起一系列其他的问题 所以接下来我就想给大家再讨论几篇 近期领域类对R1的follow up工作 我也会穿插一些我自己的想法 给大家讲讲可能的新的研究方向 我们先来看一下这篇2025年2月上线的文章 标题叫S1 Simple Test Time Scaling 这篇文章在网上引起了很多的关注 主要是由李飞飞、Percy Liang这样的大老挂名 很多人对这篇文章的评价不高 原因是方法比较简单 结果也不是特别好 但对我来说这篇paper还是非常有启发的 我觉得我们在读paper的时候
              
                  38:32
                  不能只停留于表面 只看这个paper方法新不新 结果好不好 更多的应该是挖掘和学习这篇paper本质的东西 这些本质的东西能不能启发我们去创造一些新的东西 对我来说这篇paper是非常有启发的 我觉得这篇paper的作者是真正的把DeepSeek-R1那篇paper给读懂了 抓住了DeepSeek-R1最核心的几个要点 并且把这些要点加以延伸 一个R1的要点其实来自于SFT Data DeepSeek-R1之所以能那么强大 很大一部分原因就是因为R1使用了800K的数据 其中有600K是reasoning data 200K是non-reasoning 这600K高质量的reasoning data是提升R1 reasoning能力一个非常关键的步骤 这篇S1 paper的作者就抓住了这个点 他们提出来不需要600K的数据 只需要1K的高质量的数据就可以了 在这里提出了一个新的观点 质量大于数量 Quality大于Quantity
              
                  39:35
                  第二个我觉得R1的一个重点就是 长的思维链是可以导致更好的结果 R1是用的reinforcement learning的方法 使模型自主产生比较强的思维链 有没有更简单易行的方法呢 这就是S1这篇paper所提出来 在模型生成思维链的过程当中 把终止思考的那个token强行的给替换成wait这个token 我把这个思路用一个词来表示 就是欲罢不能 模型想要停止 但是人却不让 这样就会让模型产生非常长的思维链 从而得到更好的结果 这两步对我有启发 是因为我在读完R1的paper以及其他的paper之后 我把这些知识点都互相的联系起来 当我要外推产生新的知识的时候 S1这篇paper就给了我一些确信 原来我对R1的paper的理解 关于SFT data 关于long COT在某种程度上来说是正确的 因为S1就是沿着这个思路在往下走的 并且S1这篇paper告诉了我该如何继续往下走下去
              
                  40:42
                  比如对SFT data进行进一步的精简 比如说用更简单的方式去产生长的思维链 我有了这些确定之后 我可以进一步再外推 比如还有没有什么更好的方法 能够产生高质量的SFT data 对于长的思维链 有没有其他的Prompt Engineering的方式 让模型产生长的思维链呢 长的思维链和这个结果就一定是正相关吗 有没有可能长的思维链会导致不好的结果呢 等等这些 所以我在读S1这篇paper的时候 就好像在跟作者进行思想上的对话 他们告诉我我的思路是可行的 并且他们已经做了一些探索性的工作 这样鼓励我接着往这个方向探索 第三个我觉得对我比较有启发的思路就是 在没有资源的情况下如何做一些research 这篇paper就是一个很好的例子 因为在学术界并没有像工业界那样 很多的GPU去训练一些复杂的或者是大的模型 所以你只能做一些看起来比较简单的工作 S1这篇paper这些方法对GPU的要求都不高
              
                  41:47
                  那么我们在没有很多GPU资源的情况下 就可以做一些类似这样的研究工作 那我们来具体的看一下这篇paper的一些细节 首先作者在这里收集了一个数据集叫S1K 这个数据集只有一千个数据 但这一千个数据是high quality diverse并且是difficult question 这些数据所涵盖的类别各种各样 而且很多都是非常难的问题 有些问题来自于Stanford Statistic Department的qualify exam 作者是通过Gemini直接产生思维链 从而得到这些数据 然后对数据格式上进行一些筛选 把那些格式有问题的思维链数据给去掉 那这里可能会有一点点问题 作者说他们的数据是high quality 这个思维链当中有没有任何的错误 作者其实并没有进行任何的筛选 只是保证了格式上是高质量的 所以我觉得如何从模型当中筛选出高质量的CoT数据 可能是值得研究的一个方向 右边这个图展现的就是在不同的数据量下
              
                  42:52
                  Math500这个数据上的准确度 可以看到S1只用了一千多个数据集 已经能达到90多的准确度了 其他的模型虽然准确度也能达到90多 但是用的训练数据就要大很多 尤其是R1用到了将近800K的数据 他们用上面提到的数据进行supervised finetuning这个模型之后 在实际的推理过程当中 用了一个小的trick迫使模型产生更强的思维链 这个trick就是当模型在产生思维链的过程当中 一旦要产生终止思维链的那个token的时候 就人为的把那个token换成wait 比如说在这个例子当中 问题是Raspberry这个单词里面有多少个R 前面这些是模型产生的思维链 在这个地方其实模型已经给出了答案 有两个R 这个答案是错的 所以在这个之后 模型就会产生一个终止思维链的token 作者就把它换成了wait 这样模型在接下来进行生成的时候 就会自主的产生self-evaluation 从而得到最后正确的答案 在使用了这种作者叫budget forcing的方法之后
              
                  43:58
                  模型产生的思维链就会越来越长 也就是横坐标所表示的 同样的也能观察到 当思维链变长的时候 在不同数据集上的accuracy其实也在升高的 也就是这三个图所表示的 所以作者就通过这种方式实现了test time scaling 这种test time scaling作者称为sequential scaling 和前面所提到的majority voting 这种parallel scaling是不一样的 然后作者在这里比较了一下 sequential scaling和parallel scaling 哪个更有效一些 下面这条线就是做majority voting 时候的scaling curve 上面这条就是做budget forcing的时候 scaling curve 使用同样数目的token sequential scaling比parallel scaling要更有效一些 比如说在产生大概100k左右的token的时候 sequential scaling就能达到60%的accuracy了 majority voting这样的parallel scaling方法 只能大概到50%的accuracy 这种不同的scaling方式efficiency是不一样的
              
                  45:01
                  这对我来说也是一个很有意思的启发 有没有可能把这两种scaling的方式结合起来呢 会不会有一些不一样的好处 我们最后再来看一下这个结果 最下面两行是S1的结果 这里伪造的budget forcing 本质上来说就是只用了1k的数据 做supervised finetuning 下面的结果就是用了supervised finetuning 然后做了budget forcing 我们先来看一下这个budget forcing 它所带来的好处有多少 如果我们只比较这两行的话 budget forcing其实带来的改善并不是特别的多 再来看一下这一行 这个就是S1所使用的base model 从base model上面直接做sft之后 这个结果是有一个非常大的提高的 可以看出来这个方法最主要的贡献 其实是来自于sft 而不是来自于budget forcing 但是我们再跟O1或者O1 mini比一下 即使S1最好的结果跟O1或者O1 mini比 还是要差很多 更不用说这里的R1或者是R1 distill了 这也是很多人diss这篇paper的原因之一
              
                  46:09
                  就是他们提出的这个方法 结果实在是太差了 但不管怎么样 我觉得我们应该努力从每篇paper当中 去挖掘一些对我们有用的东西 这篇paper结果可能比较差 但是有一些思考思路 还是可以被我们所利用的 这篇S1 paper就为我打开了一个思考的窗户 它里面提到两个方向 引发了我更进一步的思考 第一个就是sft阶段所使用的reasoning data 第二个就是long CoT的问题 我就按照这两个方向 再给大家撸一些paper 来讨论一下这里面可以研究的一些问题 关于sft阶段所使用的reasoning data 我想给大家看一篇 2025年2月上线的文章 标题叫LIMO less is more for reasoning 比如你有100K的reasoning数据 你可以从中挑出1%都不到的 非常高质量的数据 比如说只有817个数据去训练你的模型 这里他们用的是一个 专门解决math问题的模型
              
                  47:12
                  拿这个数据去对模型做finetune之后 这个模型的能力就可以得到极大的提高 很有意思的问题就是 这个模型仅仅只用了817个 高质量的reasoning data 就可以大幅度提高它的性能 那到底是为什么呢 这篇paper就提出了一个 less is more reasoning hypothesis 在模型的pre-training阶段 其实模型已经具备了很强的domain knowledge 如果要唤醒模型的reasoning ability的话 其实只需要一些非常少量的demonstration 但是这个demonstration必须要非常的精确 这就好比一个人已经非常的聪明 有很强的能力 那么你交给他一个新的任务的时候 你只需要给他一些演示就可以了 所以这篇paper基于这个hypothesis 就提出了增强模型reasoning task 很重要的两个因素 就是模型的pre-training非常重要 要让模型尽可能的拥有 比较好的knowledge foundation 第二个就是在post-training阶段 要给模型提供这种认知的模板
              
                  48:17
                  这种认知的模板不需要太多 但是可以极大的提高模型 去使用在pre-training阶段所学会的那些知识 去解决复杂的推理问题 作者进一步在这个表格里面 比较了一下两种增强模型推理能力的基本范式 第一种就是reinforcement learning 第二种LIMO我们可以简单的理解为 supervised finetuning 但是使用高质量的训练数据 在reinforcement learning里面 本质上是要通过搜索的方法 去找到最优的推理路径 但是在LIMO这个方法里面 reasoning的能力其实已经存在了 只不过需要一些高质量的例子去激活它 所以这两个思路是有本质的不同 基于这个不同的核心思想 接下来的这些也都会不一样 比如说在reinforcement learning里面 如何去找到最好的推理路径 就要使用RL-based的方法去进行探索 所以这个探索过程非常的重要 但是如果是基于LIMO的思想的话 最主要的问题就是
              
                  49:22
                  如何去创建这种高质量的reasoning数据 在具体的实现上面也会有一些不同 基于RL的思想 用于计算的方法去尽可能的探索解空间 基于LIMO的方法 就需要使用一些cognitive principle 来指导如何去创建这些数据 那二者的区别也非常的明显 RL的方法是非常resource intensive LIMO的方法就是resource efficient 但我个人觉得这两个方法虽然有一些的不同 但也不代表它们不能被结合起来使用 其实R1本质上来说就是结合了LIMO和RL 只不过在R1里面 它们使用的reasoning data非常的大 所以也许可以把R1里面的reasoning data 进一步的精简 然后再结合RL的方法 我们具体来看一下这篇paper里面所展现的一个例子 对于这样的一个问题 作者比较了三个模型 分别是千万2.
              
                  50:17
                  532B DeepSeek-R1 和它们的LIMO方法 也就是只用817个sample进行finetune的 可以看到千万2.532B这个模型 这个答案是回答错了 并且它没有做self correction以及self verification 对于DeepSeek-R1它的答案是对了 并且里面有很多的self reflection wait let me confirm等等这种 比较有意思的是 这种仅用少量数据finetune出来的模型 已经具备了这种self reflection的能力 比如说在这里就出现了wait 还有let's verify that let me check again 说明模型在自我的反省 我们并不需要RL的内部 仅仅使用少量的高质量数据进行sft 就可以把模型当中已经潜在的reasoning能力给唤醒 所以我觉得这是一个很有意思的结论 可以进一步的研究 刚才跟大家讨论了sft阶段reasoning data的问题 我们现在再来讨论一下 Long CoT当中overthinking的这个问题 这篇文章是2025年2月上线的 如果你问模型一个非常简单的问题 2加3是多少
              
                  51:21
                  然后比较一下不同模型它产生的token数目 这些蓝色的全部都是不带reasoning能力的模型 红色的是带reasoning能力的模型 这些具有reasoning能力的模型 对于这些简单的问题产生了非常多的tokens 比如说像R1产生了987个tokens 仅仅是为了回答2加3这样的问题 O1虽然产生的token数更少一点 但也比其他的模型要更多 比如说GPT-4O只产生了7个token 然后这里作者就比较了一下token的数目和accuracy之间的关系 我们确实看到大概有一个正相关 也就是产生的token越多 它的accuracy越高 作者在这篇文章里面主要就是研究了一下overthinking的问题 确实在现在很多reasoning model里面普遍存在这样的问题 所以他们就提出了一个方法 就类似于我们前面提到的bootstrapping的方法 让这些模型产生很多的思维链答案 然后挑选出答案是正确 并且思维链最短的
              
                  52:24
                  整理出这些短思维链答案的数据集之后 去supervised finetuning之前的模型 比如说这里他们就finetune Qwen32B 最后得到的一个结果就是 他们的模型产生的token更少 但是accuracy还略微的上升 思维的效率更高了 模型除了有overthinking的问题 还存在underthinking的问题 所谓underthinking就是模型在思考的过程当中 可能会采取不同的思考路径 对于某些正确的思考路径 模型没有继续深入地思考下去 而是浅尝则止 这样导致的一个问题就是 模型是尝试了很多的方案 但是最后的答案却是错的 我们就来看一下这一篇paper 来自于同一个group 2025年2月上线的 这里他比较了一下不同模型 在回答数学问题的时候 对于正确的问题 也就是绿色的bar和不正确的问题 也就是红色的bar 它token的分布是什么情况 如果这个模型不带有reasoning能力 正确答案和错误答案
              
                  53:28
                  它token数目基本是差不多的 但是对于这些有reasoning能力的模型 对于不正确的答案 产生的token数目 明显要比正确答案的token数要多很多 就说明模型可能在尝试 但是尝试很多次之后 答案仍然是错的 对于这些错误的答案 模型尝试了有多少个思路 对于错误的答案 模型尝试的思路 要比正确的答案要多很多的 所以这间接的就表明 对于这些错误的问题 模型可能在不断的尝试 但是没有深入的进去 作者在这里就举了一个例子 对于比较复杂的数学问题 这个reasoning model 在这一个solution里面 采取了大概有20多种不同的思考路径 然后每次思考的时候 它并没有思考完 然后就换到了第二个思考路径 注意每次模型在切换 不同思考路径的时候 都用了这个词alternatively 在经过20几个尝试之后 模型最后得出的结论就是 这个问题太难了
              
                  54:31
                  它无法回答只能放弃 这篇文章里面提出的一个解决方案 也非常的简单 Penalize alternative这个词的生成 在decoding阶段 对于alternative这样的token 修改它的logits 使得生成这样token的probability降低 这样在模型生成的时候 不太容易生成alternative这个token 这就会导致每一个思维路径 会更加的深入 关于reasoning data 还有long CoT的问题 有很多后续的paper在讨论 就给大家简单的撸到这里 给大家做一个小结 R1这篇paper 我觉得整体的思路 是前续工作的一个延续 并没有太多非常出其不意的新的想法 最关键一点就是DeepSeek 真的能把R1给做出来 并且效果非常的好 能够达到甚至超过O1的水平 但是R1这篇工作 还是有些地方属于简单粗暴 可以进一步的改善 比如如何使用reasoning data 还有关于长思维链的生成 这些都可以再进一步的研究
              
                  55:34
                  今天视频就到这里了 欢迎大家点赞转发
              



                中文中文（繁体）
                
                  00:00
                  性能已经接近于GPT-4和Gemma3-Ultra了 最后作者得到了一个120个B Token的数学相关的数据 简单说就是把CoT和PoT给联系起来了 这一类的方法有一个最大的问题就是high variance GRPO之后就开始做减法 比如说PPO使用了更简单的clipping的方法 然后GRPO又把PPO里面的value function给拿掉了 所以后面这些不断地在简化前面的方法,大道至简 在这个notebook里面会使用GRPO训练一个1B的模型 这里训练就直接调用trainer.train() 就可以开始训练了 
                  
### DeepSeekMath

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


