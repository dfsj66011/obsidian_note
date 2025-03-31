
[Measuring AI Ability to Complete Long Tasks](https://arxiv.org/abs/2503.14499)

METR，2025.03.18  主要是评价目前 AI 在完成比较复杂的，尤其是长的任务方面和人进行对比，其性能如何？METR 是一个非盈利机构，成立于 2023 年，是专门评价模型能力以及安全性的，和很多大的公司，比如 OpenAI、Anthropic 都有合作，其中一个作者就来自于 Anthropic。

这篇文章要解决的一个问题是，目前对于 AI 的基准，缺少 real world meanings，即跟人类的能力进行比较，AI 到底是一个什么水平，他们提出了一个新的 metric，50%-task-completion time horizon。利用这个 metric，他们在创造出来的一些基准任务上进行比较，并有一些有趣的结论。

<img src="https://arxiv.org/html/2503.14499v1/extracted/6285858/images/methodology_new.png" width="500">
第一步：他们首先定义了多样化的任务套件(diverse task suite) ，这些 task 来自于三个数据集，基本上都是与 software、agent 相关的，每个数据集里面抽取了一些 task，每个 task 所需要的时间是不一样的，如 SWAA Suite 中的任务需要 1-30s 解决，当有了这些数据集之后

第二步，首先用人类去解决这些问题，人类有可能需要 2-3 个小时才能解决这类的问题，同样用 LLM 作为 agent 去跑同样的任务，记录成功率，分别以人类的消耗时间和 AI agent 的成功率

第三步：做图，就可以看到这样的一种负相关，取 50% 这个成功率，也就是说对于一个任务，如果模型能取得 50% 的成功率，这个时候对于人类所需要花费的时间是多少。作者就把这个时间称为horizon length，然后用 horizon length 和这个模型发布的时间画了一个图（右下角）。

<img src="https://arxiv.org/html/2503.14499v1/extracted/6285858/plots_non_github/headline-green.png" width="500">
这个图就是展示的模型发布的时间和人类解决这类任务需要花费的时间，早期的 GPT2 模型，可以以 50% 的成功率解决人类所需要 1-2s 解决的任务，随着时间的发展，模型越来越复杂，它能解决的任务也越来越复杂，比如对于 claude 3.7，它能解决人类需要大概 0.5-1h 才能解决的任务，这种 metric 就把模型的能力和人类的水平给联系起来了。人类解决任务所花费的时间，侧面其实反映的是这个任务的难易程度。

![[Pasted image 20250331103533.png|450]]

表 1 中有一些例子，展示了对于不同的任务，复杂程度大概是什么样子。

<img src="https://arxiv.org/html/2503.14499v1/extracted/6285858/plots/bar_chart_weighted_scores/headline.png" width="500">

这里作者展示的就是在不同的任务上面，模型的成功率是多少，claude 3.7 的性能是最好的，大概能到 60% 左右，但离 100% 还是有一定的距离。

<img src="https://arxiv.org/html/2503.14499v1/extracted/6285858/plots/success_rates/model_success_rate_vs_human_completion_time.png" width="350">
这里是模型的成功率和人类需要解决这类任务的时间，负相关的曲线，人需要的时间越长，意味着任务越难，模型陈功率越低。

![[Pasted image 20250331104315.png|500]]

表 3 这里具体的分析了一下，为什么现在这些 AI agent 在一些任务上面失败了，这里比较的是 GPT4和 o1，然后把失败类型分了 5 类，

* 第一类是模型 planning 能力比较差，或者对于工具的选择能力比较差，比如给出的 plan 不合适或工具的选择不合适
* reasoning 的过程不太对，它的逻辑推理能力还是有一定的缺陷，尤其是在比较复杂的任务上面 
* 过早放弃任务，与 under thinking 本质是一样的
* repeating failed actions ，经常会重复的执行前面失败的操作
* other

作者注释中提到，o1 其实成功的case更多一点，它的失败主要是因为在更难的这些任务上面，整体来说 o1 比 GPT-4 要好一些。比较有意思的是 o1 主要的失败集中在过早放弃任务上，也就是像 o1 这种具有推理能力的模型，反而更容易产生这种问题。而老模型主要失败在重复性动作上。当模型推理能力加强了之后，这种情况反而就减少了，可能与模型具有一定的思考能力有关系吧。

<img src="https://arxiv.org/html/2503.14499v1/extracted/6285858/plots/multiverse/boxplot.png" width="500">
接下来作者把图 1 做了一个外推， 这里是不同的评估方法，我们只看这最后一个 overall(2024-2025 trend) ，作者这里选的是如果把任务的难度外推到一个月，为什么要选一个月这个时间长度呢，作者特别解释有一个定义叫做 one month AGI，也就是说如果模型能解决人类需要大概一个月的时间去完成的任务，这样的 AI 基本就已经超过了人类在完成一些比较复杂的大型的软件应用上面，甚至是成立一个 startup 或者是比如说像一些新的科学发现。

根据作者的预测大概在 2027 年或者是 2028 年左右，one month AI 就可以达到了，也就是说大概两到三年以后，我们就有望能看到一个非常强大的 AI 能够自主地帮我们完成非常复杂的软件开发，帮助我们做科研。

<img src="https://arxiv.org/html/2503.14499v1/extracted/6285858/plots/cost/ratio_vs_length.png" width="500">
这里作者主要是比较了一下任务的难度，也就是人类完成这些任务大概需要多久，纵坐表表示 cost ratio，即模型 cost 和人类 cost 进行比较，值越小代表模型的 cost 越低。大部分都在 $10^0$ 这条线下面，表示模型比人类更便宜； 

对于特别简单的任务，人类大概只需要花一分钟，cost ratio 集中在 $10^{-2}$ 附近，1% 啊！这就意味着随着 AI 技术的发展，一旦 AI 能解决这一类的问题，那么它的成本将是非常低的。

对于比较复杂的任务，大概在 10% 到 1% 之间，所以对于我们人类来说，现在不要再去学习那些处于这种 domain 的技能了，应该尽量去学习那些更加复杂、AI 取代不了的技能，这样才可能让我们在这个社会当中生存而不被 AI 所取代。
