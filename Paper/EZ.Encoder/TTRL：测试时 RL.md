
两个点：

1. 提出的这个方法，可以让 Agent 自我进化，也就是 self evolution。 与 The Era of Experience 那篇 Paper 所提出来的，未来的 Agent 应该能够长期的自我学习进行进化，就有些不谋而合了；那这篇 Paper 并没有实现 Agent 的真正的自我进化，但提出的这个方法是有一定的潜力的。
2. 图一展示的效果很好，涨点实在是太明显了。 
<img src="https://github.com/PRIME-RL/TTRL/raw/main/figs/teaser.jpg" width="600">
Paper 来自于清华大学和上海 AI Lab。 主要的 Idea 其实就是三个小的 Idea 合起来。

* 第一个 Idea 就是 pseudo labeling，翻译过来就是伪标签。 
* 第二个 Idea 就是 Test time training TTT，也叫 TTA，测试时自适应
* 第三个 Idea 就是 reinforcement learning（GRPO/PPO）, 

------

**pseudo labeling 伪标签** 这个方法，基本思想非常简单。如果我们有很多的数据没有 label，可以用人进行标注，也可以直接用模型自己为这些数据预测一个标签。可以使用一种迭代的方法，假如刚开始的时候，有很少的数据有标签，可以先训练一个初始的 model，用初始的 model 对剩下来没有标签的数据进行预测，产生 label，这个 label 就叫 pseudo label, 然后用这个 pseudo label 和之前的数据一起再重新训练这个模型，然后用这个新训练的这个模型重新去预测标签，如此反复。

具体做法上，一开始这个预测的标签肯定是不准的，所以不能无脑的使用所有预测的标签。这里可以加一个过滤，比如说只用那些置信度最高的标签结果，就不会在标签里面引入过多的噪音。 也可以用模型预测的 probability 来训练模型，而不是用模型预测的标签结果来训练模型。 

早在 2022 年的时候，Google 就提出了 self-consistent 的方法。 而且这篇 TTRL 的方法跟 Google 这篇的方法非常的像。唯一的区别就是一个是在 training time，一个是在 test time。 Google 这篇 Paper 的思想就是让大语言模型自己产生数据的标注，这里的标注特指的是思维链，然后用大语言模型自己产生的这个思维链再去训练自己。 

如果 pseudo labeling 这么神奇的话，现在的大语言模型都是 generative model，那完全没必要再用人类标注的数据或者是人类的数据，直接让 generative model 生成大量的数据，自己训自己，为什么还要费这么大力气去收集数据、标注数据呢？

Nature 上面的一篇 Paper（AI Models collapse when trained on recursively generated data，当在递归生成的数据上进行训练时，人工智能模型会崩溃）。 这篇 Paper 主要的一个观点就是，如果模型在自己产生的数据上面进行训练的话，会出现一个现象叫 Model Collapse。 

作者这里的做法是这样的，有一批初始的数据（真实的数据）。然后 train 一个 model_0。 然后从这个 model 里面产生大量的数据，训练 model_1，然后如此往复得到 model_N。 作者比较了一下这些 model 的性能会有一些什么样的变化。 

<img src="https://media.springernature.com/lw685/springer-static/image/art%3A10.1038%2Fs41586-024-07566-y/MediaObjects/41586_2024_7566_Fig1_HTML.png?as=webp" width="600">
b 左侧图，横坐标是模型产生的数据的 perplexity，纵坐标是 probability。这个图其实表示的就是这个数据的一个 distribution，在 Generation 0 的时候，也就是真实的数据，它的分布是蓝色的部分，数据的困惑度是逐步左移的，而且长尾非常长（不好的数据）。所以这里就能看出两个结论：

1. 如果模型在自己产生的数据上面训练，模型在下面的迭代过程当中是能够产生和真实数据相似的数据。 这个 perplexity 主要的分布仍然是在这一块，和真实数据是十分接近的, 
2. 但是最主要的问题就是它会产生一些 long tail。 

右图展示的是模型的性能。纵坐标就是 perplexity，值越低越好。如果在真实的数据上面训练出来的模型，这个性能是最好的。 

做了两类实验，一类叫 no data preserve，就是每次模型在之前产生的数据上面进行训练的时候，不会使用更早之前产生的数据。 另外一类叫 10% data preserve，也就是每次在迭代训练的时候，会使用之前 10% 的数据。

这两类实验观察到的现象都是一致的。对于产生的数据，这个复杂度会越来越小，这也就是这篇 Paper 所提出的 model collapse 的意思，也就是这些产生的数据，它的 perplexity 过小了之后，就没有多样性了，但同时又会产生一些有毒的数据。 另外就是使用这些生成的数据，模型似乎是能学到一些东西的，并且随着这种迭代的次数增加，模型的性能基本上趋于稳定，但还是比真实数据上面训练出来的模型要差一些。 

所以本篇论文带来的一个结论就是，在自己生成的数据上面训练可行，但是有多少的好处值得怀疑。 

------

**第二个 test time training** 或者是 test time adaptation，也不是什么新的想法。

Test-Time Training on nearest neighbors for large language models，中的想法就特别的简单，对于一个大语言模型，如果有输入的话，就用这个输入去做 nearest neighbor 查询，从这个数据库里面去找出相关类似的数据，然后把这个数据作为训练数据去 fine tune 这个大语言模型，然后再把这个训练好的大语言模型直接应用到最原始的那个输入上面，产生一个输出。 
<img src="https://arxiv.org/html/2305.18466v3/x1.png" width="500">
MIT 在去年的时候也提出过一个 test time training 的方法（The Surprising Effectiveness of Test-Time Training for Abstract Reasoning），这篇 Paper 想要解决的一个问题就是用大语言模型去做 abstract reasoning，类似于智力测验，给一些成对的图案，推测出下面这个图案它应该是什么。 
<img src="https://arxiv.org/html/2411.07279v2/x4.png" width="500">
这篇 Paper 所使用的方法就是先做一些 data augmentation，比如说不做任何的变化、identity，或者是做 horizontal flip、vertical flip 等等。那这样也会遇到同样的一个问题，就是这个 ground truth 怎么办？这个时候作者就用了类似的 majority voting 的方法，让 LLM 产生一些答案，用 majority voting 的结果作为 ground truth 再去训练大语言模型。 

TTT 或者 TTA 主要的思想就是在测试的时候，根据测试样本去 fine tune 模型的参数，但往往测试样本是没有标注的。 如何在 inference 的时候做 fine tuning？TTRL 这篇 Paper 主要就是采用的 pseudo labeling 这样的一个思想。 

-----

**第三点 reinforcement learning**，略


总结一下这篇 Paper 主要就是在测试的时间，使用 pseudo labeling 的思想产生一些标注，然后用这些标注在 inference 的时候使用 RL fine tune 模型。

-----

*TTRL 的方法具体是怎么做的？*

TTRL 方法就是在测试数据上面，也就是 inference 的时候，利用这些 unlabel 的 data 继续对模型进行 fine tune。 

它的主要思路就是这样的，对于一个给定的测试数据，这个时候模型参数是固定住的。 然后让这个模型产生多个答案，然后对这个答案做 majority voting，这样产生一个 pseudo label。 然后用这个 pseudo label 对模型进行 RL training，这个时候模型的参数是要改变的，然后再输出最终的答案。
<img src="https://github.com/PRIME-RL/TTRL/raw/main/figs/overview.png" width="500">
这个图进一步的解释了这个思路。给定一个测试数据，通过大语言模型产生 M 个 prediction，然后对这 M 个 prediction 做一个 majority voting 产生最终的答案。然后把这个答案做一个 reward calculation，这样就会产生 RL 里面所需要的 reward。然后用这个 reward 去做 policy optimization, 去优化这个大语言模型的参数。 

绿色框内的部分就是 self-consistent 的方法，这一类方法可以把它理解为 test time scaling；如果额外的再把外部 training 的部分加上去，就是 test time training。 *所以 TTRL 就是把 test time scaling 和 test time training 给结合起来了*。 

关于 *投票*，这篇 Paper 用的是 RL 的 training，而不是用的 SFT。当模型产生了很多的答案之后，有两种做法，可以用 majority voting 之后选出了最好的答案直接做 SFT；那第二种方案就是把这个答案再转成 RL 的 reward，也就是这里的 0 和 1。 

提这个，是因为我觉得这个方法之所以能成功，并且有这么大的提高，且没有出现 model collapse，*很大一部分的原因可能就是来自于这里的监督信号是 RL 的 reward, 是 01*，而不是像 SFT 里面那样的监督信号。这种 RL 里面的监督信号相对来说更弱一些，所以如果产生的数据里面有毒，它可能受到的毒害不是那么的大。再加上 RL 算法本身有一定的自我探索机制在里面，所以我觉得这有可能是解释这篇 Paper 在自己的产生的 label 上面训练也能产生很好的结果的一个原因。 

实验部分主要比较的是 GRPO 和 PPO，然后用了两个不同的指标 accuracy 和 entropy。可以看到不管是 GRPO 还是 PPO，在 RL 的训练过程当中，性能似乎都是差不多的。但能观察到一个比较有意思的现象，对于 GRPO 来说，它这个曲线是非常的 bumpy，也就是不是那么的稳定，但是对于 PPO 来说，它这个曲线是更加稳定的。 *训练过程当中，PPO 要比 GRPO 更稳定一些*，很可能就是因为 PPO 当中有一个 value function。这个 value function 可以提供一些学习的信号，所以让这个训练更加的稳定。 
