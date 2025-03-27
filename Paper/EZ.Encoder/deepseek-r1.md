
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

 页效果如何
我们点一下智能体
它解释了这个智能体是什么
环境状态action
这个demo看起来就太简单太粗糙了
reward policy
总的来说我觉得在这个demo上面
deepSeek0324是吊打GPT 4.5
GPT 4.5这个实在是太粗糙了
下面这个demo我想用DeepSeek0324
分析一下DeepSeekMath这篇paper
所以我把PDF传给他了
然后我给了一些prompt
我想让DeepSeek分析这篇paper
并且帮我将内容总结成一个网页
这样做的目的就是
节省我去阅读这篇paper的时间
看看DeepSeek能不能帮我
把这篇paper主要的内容给总结下来
如果后面我需要展示结果的话
我也可以直接用这个网页去给大家展示
我们来跑的试一下
这里DeepSeek0324分析了DeepSeekMath这篇paper
然后写了一个html文件
也是非常的长
然后还做了一些说明
这个网页以专业美观的方式
呈现了DeepSeekMath论文的核心内容
使技术信息更易理解和消化
这就是我想达到的目的
0324生成的网页
首先他对这篇paper做了一个总结
DeepSeekMath-7b突破开源模型
在数学领域的极限
接近于GPT-4和Gemini Ultra
然后还总结了其他的一些结果
比如说在math基准数据集上
达到了51.7的准确率
还有他们是如何准备一个大规模训练数据的
以及推出了GRPO算法
这里DeepSeek0324还直接帮我做了一个可视化的图
在math基准数据集上的准确率
分别和GPT-4 Gemini Ultra
以及其他模型进行比较
还有在GSM-8K上面的性能
这里还总结了一个DeepSeekMath
在(中文)数学数据集上的结果
分别比的是pre-trained base模型
SFT的instruct模型和做完RL之后的模型
但是这里看起来好像有些问题
为什么高考数学填空这个结果是0
然后这里总结了一下具体的技术创新
比如说它是从common crawl里面
取出来的高质量的数学内容
这个图应该是一个幻觉
在paper里面应该是没有提到这个结果的
GRPO可以节省内存训练
提高训练速度
还有对应的百分比
这个在原始paper里面应该是没有的
原始的paper里面确实比较了
单阶段训练和两阶段训练
但是好像没有混合训练
所以这个可能是DeepSeek的一个幻觉
这里DeepSeek还把作者也放到了最后
但这个作者应该不全
这里DeepSeek只是把paper上面加星号的那几个人给列在这了
总的来说我觉得这个网页做的还是不错的
但是里面可能有一些幻觉
即使这样也可以极大的节省我的时间
我可以在这个模板基础之上进行修改
这样以后如果我想给别人讲解paper的话
我可以直接使用这个网页
比讲原始的paper可能更美观一些
GPT 4.5把DeepSeekmath这篇paper扔给他
加上一些prompt让他帮我们分析这篇paper
并做一个可视化的网页
这里GPT 4.5帮我生成了一个网页
我们来看一下GPT 4.5的结果
这个网页就这么简单的一页
而且看起来也非常的丑
他对摘要做了一个简短的总结
然后对这篇paper几个主要的贡献
比如说数据的预训练
强化学习
还有模型的对比
没有太大问题
但是太过于简单了
对我来说基本没有达到我的要求
我在这个视频里给大家快速的测试了一下
DeepSeek v3 0324这个最新的版本
和GPT 4.5做了一些比较
使用的一些例子都是我感兴趣的
如何能提高我的工作效率
比如说能否快速的帮我做一个
关于attention或者是reinforcement learning的demo
这样我讲解的时候可以直接用
或者是帮我直接分析一篇paper
做出可视化的结果
DeepSeek v3 0324在这三个例子上面
基本都是吊打GPT 4.5的
另外我在测试过程中也实验了
O3 mini high
结果也差强人意
所以我觉得DeepSeek v3 0324这个版本
是非常强大的
大家感兴趣可以试一下