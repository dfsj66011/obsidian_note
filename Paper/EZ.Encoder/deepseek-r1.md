
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
