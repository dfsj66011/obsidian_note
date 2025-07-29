
Author：Raj Dandekar

google docs: docs.google.com/spreadsheets/d/1GLAndnI1-PbFDXSa0qdbRaBLJiTQHdcZpmmfMbeRAqc/edit?gid=867380576#gid=867380576


（24.56）

DeepSeek 究竟有何特别之处？它是如何做到收费如此低廉的？又是如何在保持与 GPT-4 竞争的性能的同时，实现如此高的成本效益的？这里有四个主要方面需要讨论:

* 首先，DeepSeek 拥有创新的架构；
	* 采用了多头潜注意力机制（MLA）
	* 混合专家架构（MoE）
	* 多 token 预测（MTP）
	* 引入量化技术
	* RoPE
* 其次，其训练方法极具创造性和创新性；
	* 强化学习的兴起
	* 不再仅仅依赖人工标注的数据，而是利用大规模强化学习来教授模型进行复杂推理
	* 基于规则的奖励系统
* 第三，他们实现了多项 GPU 优化技巧；
	* 采用了 NVIDIA 的并行线程执行技术（PTX）
* 第四，构建了一个有利于蒸馏等技术的模型生态系统
	* 蒸馏至更小的模型（1.5B）


### 1、MLA

分为四个部分：

* LLMs 架构本身
* 自注意力
* 多头注意力
* KV 缓存


**1、当我们有了每个 token 的嵌入向量，为什么不能直接简单地用点积来计算注意力分数？**

直接使用 token 嵌入向量做内积确实在技术上是可行的，但通过 QKV 变换有几个重要的优势：

1. 表征空间的分离和专门化

```python
# 直接内积方式
attention_score = embedding_i @ embedding_j  # 单一表征空间

# QKV方式
Q = embedding @ W_q  # 查询空间：我在寻找什么？
K = embedding @ W_k  # 键空间：我能提供什么信息？
V = embedding @ W_v  # 值空间：我实际包含什么内容？
```

QKV将不同的语义角色分离到不同的子空间中，使模型能够学习更精细的注意力模式。

2. 增强表达能力和灵活性：直接内积只能捕获原始嵌入空间中的相似性，而QKV变换允许模型学习：
	* 非对称关系：Q 和 K 可以学习不同的变换，使得 attention(A,B) ≠ attention(B,A)
	* 任务特定的相似性：不同的注意力头可以关注不同类型的关系（语法、语义、位置等）

3. 多头注意力的实现，使得不同的注意力头能够捕获不同类型的依赖关系。

4. 梯度流和训练稳定性：QKV变换提供了额外的可训练参数，有助于更好的梯度传播，避免嵌入向量直接被注意力机制"绑架"，并提供更多的学习自由度

5. 维度控制：可以将高维嵌入投影到更适合的注意力计算维度

总的来说，虽然直接内积在某些简单场景下可能有效，但 QKV 变换提供了更强的表达能力、更好的可解释性和更灵活的学习机制，这是现代 Transformer 架构成功的关键因素之一。

-----------

**2、注意力机制中，需要对结果进行缩放，即除以 $\sqrt{ k }$ 的目的**

在注意力机制中，如果 softmax 分布过于尖锐，模型就会对某一个特定的键变得非常自信，而对其他键的置信度会非常低，这会导致训练过程非常不稳定。在 attention 计算公式中除以 $\sqrt{d_k}$ 的目的是 *防止点积结果过大，避免 softmax 函数进入饱和区域*。

1. 点积幅度问题：
	1. 当维度 $d_k$ 较大时，两个向量的点积 $Q \cdot K^T$ 的幅度会随维度增长
	2. 假设 $Q$ 和 $K$ 的元素是独立的随机变量，均值为 0，方差为 1
	3. 那么点积的方差约为 $d_k$，标准差约为 $\sqrt{d_k}$

2. Softmax 饱和问题：例如：`softmax([10, 1, 2]) ≈ [0.9999, 0.0000, 0.0001]`

3. 梯度消失：当 softmax 输出接近 0 或 1 时，其梯度接近 0，影响训练效果

----

3、**为什么是平方根，而不是平方或者直接 $d_{k}$ 或其他形式？**

选择 $\sqrt{d_k}$ 而不是其他形式有深刻的数学和统计学原理。

1. 方差分析：假设 $Q$ 和 $K$ 的每个元素都是独立同分布的随机变量，均值为 0，方差为 1：对于点积 $QK^T = \sum_{i=1}^{d_k} q_i k_i$：$$
\text{Var}(QK^T) = \text{Var}\left(\sum q_i k_i\right) = \sum \text{Var}(q_i k_i)$$由于 $q_i$ 和 $k_i$ 独立，且均值为 0：$$
\text{Var}(q_i k_i) = \mathbb{E}[q_i^2 k_i^2] - (\mathbb{E}[q_i k_i])^2 
= \mathbb{E}[q_i^2] \mathbb{E}[k_i^2] - 0
= \text{Var}(q_i) \times \text{Var}(k_i) = 1 \times 1 = 1$$因此：$$
\begin{align}
\text{Var}(QK^T) &= d_k \times 1 = d_{k}\\ \\
\text{Std}(QK^T) &= \sqrt{d_k}
\end{align}$$
2. 标准化目标：我们希望缩放后的点积方差保持为 1：$$
\text{Var}\left(\frac{QK^T}{\text{scale}}\right) = \frac{\text{Var}(QK^T)}{\text{scale}^2} = \frac{d_k}{\text{scale}^2} = 1$$
解得：$\text{scale} = \sqrt{d_k}$

3. 实验验证不同缩放方式的效果：

```python
import torch
import torch.nn.functional as F


def compare_scaling_methods(d_k=64, num_samples=1000):
    results = {}
    
    for _ in range(num_samples):
        Q = torch.randn(10, d_k)
        K = torch.randn(10, d_k)
        scores = Q @ K.T
        
        # 不同的缩放方式
        scales = {
            'no_scale': 1,
            'sqrt_dk': torch.sqrt(torch.tensor(d_k, dtype=torch.float)),
            'dk': d_k,
            'dk_squared': d_k**2
        }
        
        for name, scale in scales.items():
            scaled_scores = scores / scale
            if name not in results:
                results[name] = []
            results[name].append(scaled_scores.var().item())
    
    # 打印方差统计
    for name, variances in results.items():
        mean_var = torch.tensor(variances).mean()
        print(f"{name:12}: 平均方差 = {mean_var:.4f}")

compare_scaling_methods()
```

```
no_scale    : 平均方差 = 63.5230 
sqrt_dk     : 平均方差 = 0.9925 
dk          : 平均方差 = 0.0155 
dk_squared  : 平均方差 = 0.0000
```

4. 为什么其他缩放方式不好：

	1. 直接除以 $d_k$：$\text{Var}(QK^T / d_k) = d_k / d_k^2 = 1/d_k$；当 $d_k$ 很大时，方差过小，导致所有 attention 权重趋于均匀分布，失去选择性注意的能力
	2. 除以 $d_k^2$：$\text{Var}(QK^T / d_k^2) = d_k / d_k^4 = 1/d_k^3$；方差极小，几乎没有区分度
	3. 不缩放：高维时 softmax 饱和严重

--------

**掩码注意力机制：** 抹掉上三角部分的注意力分数，同时保证剩下的部分，每行归一化后总值仍为 1.

策略 1：先计算完整归一化后的注意力权重，然后实施掩码，对剩余部分的值做归一化处理，例如第 2 行抹掉后，只剩余 0.1, 0.4 两个值，归一化后值为 0.1/0.5,   0.4/0.5；这个策略的问题是在计算注意力得分时，实际上已经计算过一次归一化了，相当于做了两遍归一化，是否可以采用更智能的方法？

策略 2：基于未进行归一化后的注意力分数（$QK^T$）直接实施掩码，即将上三角部分赋值 $-\infty$，这么做的目的是，在 softmax 中，存在指数计算，而 $e^{-\infty}=0$

**Dropout**：随机将部分值抹掉为 0，例如抹掉 5% 的比例，则剩下的这部分需要适当放大，放大系数为 $1/0.95$

---------


所以如果你只使用一个头，你也会得到相同的11乘4的上下文向量矩阵，但它不会在一个头中包含两个视角。同样，你也会得到一个11乘4的上下文向量矩阵，但整个矩阵只会包含一个视角。而现在的优势在于，矩阵的大小保持不变，但它包含了两个视角。第一个视角由我的第一个头给出，我称之为P1；第二个视角由我的第二个头给出，称为P2。因此，我们从文本中提取了更多的信息。

当然，这样做的缺点在于，对于提取每个视角的信息，我们现在只有两个维度可用，这就是弊端所在。而之前我们实际上每个视角有四个维度，对吧？但现在每个视角可用的维度数量减少了。这就是多头注意力的主要缺点。关键弊端在于每个头的维度大小缩减了，对吧？正如你在这里看到的，每个头的维度实际上被压缩了，因为我们不得不将整个查询权重矩阵、键权重矩阵和值权重矩阵一分为二。

因此，每个注意力头的维度大小被缩小了。这意味着我们能捕捉的信息量有所减少，但能捕捉的视角数量却增加了。因此，每个注意力头能捕捉更多的视角。所以我的理解方式是分而治之。与其一次性攻克整个句子，不如将其分解为不同部分，然后每个部分从不同角度进行攻克。这就是我喜欢用来理解多头注意力的最简单方式。

那么，我们刚才看到的这个逐步流程，让我们快速回顾一下。我们从输入嵌入矩阵开始，艺术家用画笔描绘了一幅女性肖像画，我特意选择了这个可以从不同角度解读的句子作为例子，对吧？所以我们的操作是：首先处理输入嵌入矩阵，当它与可训练的查询键和值矩阵相乘时，我们会将这些可训练的权重矩阵分成两部分。我们将输出维度固定为4，并确定注意力头的数量。

既然我们有两个头，每个头实际上会得到两个维度。这被称为头维度，即d_out除以头的数量等于2。因此，wq1是8×2，wq2是8×2，以此类推。这些是头1的可训练查询、键和值权重矩阵，而这些是头2的可训练查询、键和值权重矩阵。好的，一旦我们有了w、q、k和v的多个副本，自然就会产生查询、键和值的多个副本。

因此，头1拥有一组q、k、v的副本，即q1、k1和v1，而头2拥有另一组q、k、v的副本，即q2、k2和v2。然后，我们为q1和k1计算第一个注意力分数矩阵，为q2和k2计算第二个注意力分数矩阵。第一个注意力分数矩阵来自头1，第二个来自头2。为什么我们需要两个注意力分数矩阵呢？因为每个头可能捕捉不同的视角，例如第一个头可能捕捉这种视角，第二个头可能捕捉另一种视角，等等。

因此，每个注意力头可能捕捉到不同的视角，这就是为什么这里有两个注意力分数矩阵。在我看来，这部分是最重要的步骤，因为在这里我们看到每个注意力分数矩阵捕捉了不同的视角，而这正是多头注意力机制的全部优势所在。之后，我们会遵循与自注意力机制相似的步骤。

然后，我们获取注意力分数矩阵，用键向量维度的平方根进行缩放，应用softmax函数，再应用因果注意力机制——这意味着我们会将注意力权重矩阵对角线上方的所有元素掩码为0。接着，如需提升模型泛化能力或防止过拟合，我们还可以应用dropout操作。至此，我们已计算出各注意力头的权重矩阵。随后，我们将每个注意力头的权重矩阵分别乘以其对应的值向量v1和v2，从而得到第一个注意力头的上下文向量矩阵和第二个注意力头的上下文向量矩阵。每个头的上下文向量矩阵代表着：现在我们有11行数据（如artist、painted等词汇），即模型从输入嵌入向量转化为了包含上下文信息的向量表示。


So now for artist, instead of just looking at the semantic notion of artist, the context vector for artist now captures information about how this artist relates to the other tokens. That's why this matrix is much more richer than the input embedding matrix. So this is the head 1 context vector matrix and this is the head 2 context vector matrix and in the last step what we do is that we merge the context vector matrices for both the heads and that leads to the final context vector matrix 11 by 4. The size of this is the same as what it would have been if we just used self-attention with a single head but the main advantage is that we have now two perspectives within this context vector matrix p1 and p2.

So hopefully we will capture richer representations in the text itself. The disadvantage is of course in each perspective we now get reduced number of dimensions to play with. So the expressivity in each perspective might be reduced but this is a trade-off which seems to work well in our favor because all the modern LLMs are based on the multi-head attention mechanism.

We know LLM just has a single head, we have multiple heads so that each head can capture a different perspective. So this is the whole step-by-step procedure of how we go from the self-attention mechanism to the multi-head attention mechanism which was the main purpose of today's lecture. Now what I want to show you is that I want to show you a very quick demonstration of how of visualization of these attention heads.

So what we are going to do is that we are going to take a pre-trained large language model. We are going to take a pre-trained LLM. So this is going to be a BERT model and it will have a bi-directional attention.

So causality is not implemented. So every token will look at previous tokens and also the tokens after that. So this is pre-trained which means it has already been optimized on a huge amount of data.

What we will do is that we will pass our input sentence to this pre-trained LLM and what's the input sentence, the artist painted the portrait of a woman with a brush. We will pass our input sentence to this pre-trained LLM and then what we will do is that we will peek into the different attention heads and we will see what every attention head essentially gives us. So remember that when you see the code there will be two parameters, there will be layer and the head.

So what layer essentially means is that an LLM architecture has multiple transformer blocks and each transformer block has multiple attention heads. So when we look at different layers it means different transformer block and when we look at head it means that which head we are in a particular layer. So for the purposes of demonstration we are only going to look at layer number 3 which is essentially the third transformer block and in this layer number 3 we are going to look at attention head number 3 and attention head number 8. What does it mean attention head number 3 and attention head number 8? The pre-trained LLM which we are looking at will have 11 attention heads.

So the output dimension is split into 11 different parts and then every attention head will essentially get D out divided by 11 and then we are going to look at the attention weights matrix such as this for each head and that is going to tell us that when we look at woman for example what is prioritized by different attention heads. So let's quickly jump into the demonstration right now. So the package which I have downloaded over here is birthwiz and then what I am simply doing is that I have loaded the pre-trained model over here and I am just showing a visualization for this sentence the artist painted the portrait of a woman with a brush and first I want to show you for layer number 3 and we will see for layer number 3 and essentially head number 3. So if you go into layer number 3 and head number 3 and if you hover on to woman let's see so if you hover on to woman you will see that the maximum attention is given to brush.

If you see on the right hand side the maximum attention if you trace this line you will see that the maximum attention is given to brush and we can also confirm this. So in this in this code I have essentially plotted the different attention scores which are given for woman and I have taken a screenshot over here. So if you look at layer 3 and head number 3 and if you take the query as woman here you can plot the tokens for which the maximum attention weight is given.

So the maximum attention weight of score is given to brush for layer 3 and head 3 but now let's go to layer 3 and head number 8. So if I go to layer 3 and head number 8 right now you will see that when you see woman the maximum attention is now given to portrait and that is again confirmed over here if you see layer number 3 and head number 8 maximum attention is given to portrait. So this might indicate that here the attention of the woman is given to brush right so that might mean that here we have this second visualization. So let me take a after this loads let me take a screenshot of that second visualization here.

So it seems that the attention between woman and brush is the maximum right. So it seems that head number 3 which we saw over here thinks of this interpretation or this perspective because it seems in the second perspective the woman holds a brush in her hand and the attention between woman and brush is the maximum in this head whereas if you look at head number 8 the attention between woman and brush is very low. So it seems that this head has recognized that maybe the woman is not holding the brush but the woman is just present in the portrait.

So it might mean that this second head thinks that this perspective is more strong. So it may be decodes this perspective. So here you can see this is the direct proof that a pre-trained transformer or a pre-trained LLM rather has different attention heads and each attention head can essentially uncover a different meaning the head number 3 which we saw over here uncovers this meaning that maybe the woman holds a brush in her hand whereas head number 8 uncovers this meaning that maybe it's just a portrait of a woman and the woman might not be holding a brush but the artist might just be painting the portrait of a woman.

So in this hands-on demonstration we just saw that different attention heads can capture different perspectives and that's the whole aim or the whole purpose of the multi-head attention mechanism. I hope I have been able to explain why what is the intuitive need that we need to go from the self-attention mechanism to the multi-head attention mechanism. This lecture was specifically dedicated to introducing the intuition behind multi-head attention mechanism.

In the next lecture what we are going to do is we are going to do actual calculation using mathematical numbers. So we are going to start with a given sentence we are going to assume some numbers and we are going to apply multi-head attention in practice. But I did not want to directly jump to this lecture without giving you an intuition for why we move from self-attention mechanism to multi-head attention mechanism.

This is the core building block of why LLMs work and it's also the core building block which DeepSeek figured out that we need to modify this block itself to make it a bit better. So although multi-head attention has a lot of advantages it does have some disadvantages in terms of storage space and computational efficiency which are mitigated or reduced by key value cache and further reduced by multi-head latent attention. So that's why we have this important milestone in our way.

We cannot understand KV cache or multi-head latent attention without understanding multi-head attention itself. So after the next lecture we will directly move to key value cache and then multi-head latent attention which is the first fundamental innovation in DeepSeek. So stay tuned till that time and I hope you are making notes alongside.

These lectures are a bit dense and I am deliberately making them a bit longer so that everything is explained to you. This won't be a lecture series of 2 to 3 lectures. I am planning to make it 35 to 40 videos of lecture series so that ultimately you really understand the nuts and bolts of how DeepSeek is constructed.

But to go to that stage it's important for us to be on the same page with the building blocks. Thanks a lot everyone and I look forward to seeing you in the next lecture. Hello everyone, my name is Dr. Raj Dhandekar.

I graduated with a PhD in Machine Learning from MIT in 2022 and I am the creator of the Build DeepSeek from Scratch series. Before we get started, I want to introduce all of you to our sponsor and our partner for this series, Invidio AI. All of you know how much we value foundational content, building AI models from the nuts and bolts.

Invidio AI follows a very similar principle and philosophy to that of us. Let me show you how. So here's the website of Invidio AI.

With a small engineering team, they have built an incredible product in which you can create high quality AI videos from just text prompts. So as you can see here, I have mentioned a text prompt, create a hyper-realistic video commercial of a premium luxury watch and make it cinematic. With that, I click on generate a video.

Within some time, I am presented with this incredible video, which is highly realistic. What fascinates me about this video is its attention to detail. Look at this, the quality and the texture is just incredible.

And all of this has been created from a single text prompt. That's the power of Invidio's product. The backbone behind the awesome video which you just saw is Invidio AI's video creation pipeline in which they are rethinking video generation and editing from the first principles.

To experiment and tinker with foundational models, they have one of the largest clusters of H100s and H200s in India and are also experimenting with B200s. Invidio AI is the fastest growing AI startup in India, building for the world. And that's why I resonate with them so much.

The good news is that they have multiple job openings at the moment. You can join their amazing team. I am posting more details in the description below.

Hello everyone, and welcome to this lecture in the build deep seek from scratch series. Today, we are going to have our second lecture on the multi head attention. If you remember in the previous lecture, we went through the conceptual overview of multi head attention and how the mechanism actually works, how we split the query key and value vectors into multiple different heads and how each of the heads eventually leads to a different attention score.

And that actually helps us in capturing two different perspectives with just the self attention mechanism, we can capture only one perspective of a given sentence or a given paragraph. But in some cases, we would like to essentially compute the concepts in a lot more detail or from a lot more perspectives, which is not possible through just a self attention mechanism. And that's why we have the multi head attention mechanism where each head allows us to compute a different perspective.

So we saw the whole process in the previous lecture. Today, what we are going to do is that today we are going to take a matrix, we are going to take an actual input embedding matrix, and we are going to see step by step about how the multi head attention works in practice. So this is going to be a mathematics based lecture.

The previous lecture was more about intuition. But here we are going to go step by step with respect to mathematics. And parallelly, I'm also going to show you the code for how the multi head attention function is written.

And by the way, what we are going to cover in today's class is exactly how the first multi head attention implementation works in practice. So whatever we have done until now in this lecture series, starting from self attention to causal attention to the first lecture of multi head attention, it all comes to a culmination point in this lecture, where we'll code out the entire multi head attention class in multi head attention class in Python. And I'll also show you the step by step mathematical derivation and every matrix multiplication in a lot of detail.

First, let's do a quick overview of what we learned in the previous lecture. In the previous lecture, we saw that if you have sentences such as this, the artist painted the portrait of a woman with a brush. This can be viewed in terms of two perspectives, either the artist painted the portrait of a woman using a brush or the artist painted the portrait of a woman with a brush.

So the woman had a brush in her hand, it can be two different perspectives, right? But a self attention mechanism can only capture one such perspective because there is only one attention scores matrix either we can have this or we can have this. So then the question is, can we somehow extend the self attention mechanism so that it can capture multiple perspectives? So what we did was a single head can get us one attention score. So what if we have multiple heads, right? So what if we have two self attention mechanisms instead of one, and then we saw how to implement this multi head attention step by step.

We started with an input embedding matrix, the artist painted the portrait of a woman with a brush. We saw that the first step which is done is that we decide the output dimension. And this is going to stay the same in today's lecture as well.

We have to decide the output dimension and we have to decide the number of heads which we want, the number of attention heads. So in our case, in the previous lecture, we had output dimension equal to 4 and number of attention heads equal to 2. So the head dimension is the output dimension divided by the number of attention heads that's equal to 2. This is the dimension of each head. So what actually happens when we start doing the calculation is that we split the trainable query matrix, the key matrix and value matrix into two parts.

And that eventually splits the query vectors, the key vectors and the value vectors into two parts. Why do we have two parts because there are two attention heads. So there is one copy of Q, K and V, which is the query key and value matrix for each head.

And now that we have two copies of the query key and value, it essentially leads to two attention scores. So this is the attention scores matrix calculated from the first head. And this is the attention score matrix calculated from the second head.

This right here is the most important point in the multi-head attention mechanism workflow because each attention scores matrix here essentially represents a different perspective. This was not possible with just the self-attention mechanism. Now we have two attention scores matrices, right? So each can capture a different perspective.

And that's the main advantage of multi-head attention. Then what we do is after we have the attention scores, we do the scaling with square root of the keys dimension, we apply softmax, causal attention. And if we want, we can do dropout also.

That leads to the attention weights, the head 1 attention weights and the head 2 attention weights. And then what we do is that we multiply the attention weights of the head 1 and head 2 with their corresponding value vectors. And then we get the context matrix for head 1 and head 2. We merge these context matrices and then we get the final context matrix, which now is a mixture of two perspectives.

So remember, the more the attention heads we have, the more the perspectives we can capture into our final context matrix. That's exactly how the multi-head attention actually works in practice. We also saw a cool visualization towards the end where we took a pre-trained large language model and we actually explored inside the attention heads.

So today, as I mentioned, we are going to do a mathematical calculation of exactly how the multi-head attention works. So I'm going to show you every single matrix multiplication from scratch. And then I'm also going to show you how it relates to the code for the multi-head attention.

So here if you see, this is the multi-head attention class. And if you just see the class without understanding the mathematical details, this all might seem a bit confusing to you, but I will take you today through the entire mathematical derivation and then I'll show you that it's actually directly mapped to the code and then understanding the code becomes that much easier. So let's get started.

Today, the example which we are going to consider is this example where we'll start with an input embedding matrix. So the input embedding matrix is X and there are three tokens essentially. Let me change my color here so that maybe I'll change it to this new color over here, which is orange.

So this is my input embedding matrix here. The way to visualize this matrix is that there are three tokens. This is token one that corresponds to the first row.

This is token two, which corresponds to the second row. And this is token three, which corresponds to the third row. And every token essentially has a certain dimension.

That's my input dimension. And this is going to be equal to six in this case. So there are three tokens and each token has six dimensions.

But as you see this, this is a tensor with three dimensions over here. So what's these three? The second is essentially just the number of tokens. The third is essentially the input dimension din.

So this is the tokens. This is the din. But what's the first dimension? The first dimension is essentially the batch, the batch size.

So we are going to just pass in one batch for the sake of simplicity. But if there are two batches which are passed, then this will be of a size two by three by six. If there are three batches which are passed together, then this will be of a size three by three by six, etc.

Right. But for the sake of simplicity, I am going to assume that just one batch is passed at one time. But remember that whenever you write the code for the multihead attention block, it always starts with the input embedding matrix with three dimensions.

So when you go to the code, you will see that we start the forward method of the multihead attention class with three dimensions, the batch size, the number of tokens and the input embedding dimension. That's the X dot shape, right? So this is my input embedding. So just to give you a quick recap of what this is, is if you remember the journey of the token through the.

So we had a lecture regarding the journey of a token through the LLM architecture, right? There is a whole data pre-processing step where every token essentially gets converted into something which is called as the input embedding, which is the summation of token embedding plus position embedding. And we call the input embeddings as uniform, essentially every token gets its own uniform. So the input embedding for every token, which I'm showing over here is actually that uniform.

So every token has its own row, right, which is a vector of six dimensions. That's its input embedding, which is the summation of token embedding plus position embedding. If you have not seen that lecture before, I highly recommend you to go through that lecture, which is titled journey of a token through the LLM architecture.

So now we have done the first step, which is essentially defining our input, which consists of three dimensions. All right, now next, let's go ahead to apply the multihead attention mechanism. As we saw in the previous lecture, we have to decide two things, we have to decide the output dimension and we also have to define the number of attention heads which we want.

Why do we need to decide these two dimensions? Because it eventually decides the dimension of each attention head, which I have, which is called as head dimension. Remember, the head dimension is going to be D out divided by n heads. So in this case, what I'm going to do is that I'm going to decide my output dimension equal to six, and I'm going to decide the number of heads to be equal to two.

So you can do a calculation of what each head dimension will be, it's going to be six divided by two. So each attention head is going to have three dimensions now. Okay, great.

The next step which I'm going to do is I'm going to multiply my input embedding matrix with my trainable query key and the value matrix. So first I have to initialize these matrices, right, I have to initialize. So remember my x now, which is this input has is one by three comma six.

So if you forget this first batch for now, it's three by six. So if you want to multiply it with trainable query matrix, trainable key matrix and trainable value matrix, the first the dimensions of this, the dimensions of this are D in comma D out. The dimensions of these trainable matrices are always D in comma D out.

And remember now we have finalized D in and D out, my D in is equal to my dimension in is equal to six and my D out is also equal to six. So my trainable weight matrices for the query key and value will all be six by six matrices which are initialized randomly. So that's what I'm going to do over here.

I'm initializing the trainable weight matrices for the key query and value. And these will be six by six matrices. This is for WQ, this is for WK and this is WV.

Remember, all the values inside these matrices are completely random right now. Our goal later through back propagation is to optimize these values so that the next token is predicted correctly. Once I have this trainable weight matrices, the next step what I'll do is that I'll take my input, I'll take my input and I'll multiply with these matrices.

So I'll multiply X with WK, X with WQ and X with WV. So remember my X is now one by three comma six. So if I multiply it with six by six that will lead to again a one by three comma six matrix.

So after this multiplication, I get the keys matrix, which is one by three by six. I get the queries matrix, which is one by three by six and I get the values matrix, which is one by three by six. I want you to pay very close attention to these three values over here.

Remember we started out with an X input whose dimensions are batch size, number of tokens and din. Now check the dimensions of these queries, keys and the values. The first is of course the batch size, the second is the number of tokens.

So these two remain the same, but the third is now dout. So instead of the input dimension, we are now in the output dimension space. So the keys, queries and values are all of the dimensions of batch size comma number of tokens comma output dimension.

And until now, we have not applied the multihead attention at all, but bear with me, we will apply it later. So let's go to the code right now. We have the X dot shape, which is B comma number of tokens comma the input dimension.

And then what we are going to do is that in the init method. So if you see, we have to initialize the trainable query key and value matrices, right? That's essentially done over here. The query, the key and the value matrices are initialized through a linear layer of a neural network and the bias term is equal to zero.

What this essentially means is that when you do self dot or when you pass X, which is the input through the trainable key weight matrix, it essentially takes a multiplication of the trainable key matrix multiplied by X. This is exactly what we wanted, right to get the keys, queries and the values. We just have to take the multiplication of the we have to take the multiplication of the input embedding matrix and the trainable query key and the value matrices. That's what's done in this step.

So here, although you cannot see a direct multiplication operation, your X is being passed X, which is my input is being passed as an input to this W key, which is the linear layer of a neural network. This essentially is a multiplication because the.

(该文件长度超过30分钟。 在TurboScribe.ai点击升级到无限，以转录长达10小时的文件。)

(转录由TurboScribe.ai完成。升级到无限以移除此消息。)

You see the bias terms are set to 0. The reason we use nn.linear instead of nn.parameter is that nn.linear just has an optimized initialization scheme for the weights. So that helps in our backpropagation later, but remember in this part what we are doing is that we are calculating the, so this is, this is calculation of the keys matrix which is represented by this key over here, this keys matrix. This part is the calculation of the queries which is the input multiplied by the trainable query matrix and the third part over here or the third line is the calculation of the values matrix which is the input multiplied by the trainable value matrix.

So until this point in the code we have obtained the keys, queries and the values and remember its dimensions are batch size, number of tokens and the output dimension. Now the real magic of multi-head attention starts after this point. Remember what we saw in the previous lecture.

What we saw in the previous lecture is that once we have the query key and the value matrices depending on the number of heads these matrices are split into parts right. So here if you see this whole, this whole trainable, this whole trainable query matrix which was there, it was essentially split into two parts because there are two heads. This is exactly what we are going to do in the code right now or this is exactly what we are going to do in the mathematical calculation right now.

If you see closely the last dimension is D outright, what we are going to do is that we are going to unroll this last dimension into two parts. So currently the dimensions of this keys, queries and the values they are B batch size, number of tokens and the output dimension right. We are going to unroll the output dimension into two parts.

What this means is that instead of having a 1 by 3, 6 we are going to unroll it to number of heads comma head dimension. Now check this out carefully. We have defined D out which is equal to 6 and we have defined D number of heads which is equal to 2 right.

That means that the head dimension is equal to 3. Essentially what this means is that each head has a dimension of 3. So if you have this query matrix now which is a 6 by 6 so 1, 2, 3, 4, 5, 6 and if I am going to repeat this 6 times let's look at my query matrix right now right. So actually my queries is 3 by 6 sorry 1 by 3 by 6 so I just have to have 3 rows. So now this is my query right.

But now instead of having 6 columns what I am saying is that there are 2 attention heads. So each head should get a dimension of 3. So I should split it into two parts. This should be my head number 1 and this should be my head number 2. So this entire dimension of 3 by 6 right so then essentially this will be 3 by 3 and then essentially this will be 3 by 3 or rather 1 by 3 by 3 and 1 by 3 by 3. So another way of saying this is that instead of having a 1 by 3 by 6 I will have a 4 dimensional tensor which is 1 by 3 by 2 by 3 where now D out is just replaced by number of heads and head dimension.

And there is a very easy way to actually visualize this. If you can easily visualize this right number of tokens comma D out that's essentially just 3 rows and 6 columns right. Number of tokens comma D out which was this is just 3 rows and 6 columns.

But now 3 by 2 comma 3 you can visualize it like this there are 3 so this is token 1, this is token 1, this is token 2 and this is token 3 right. And earlier each token essentially had 6 columns associated with it. The first token had these 6 columns associated with it because the output dimension was 6. But now you see these 6 columns have been unrolled into a 3 column and 3 column.

So, essentially what is done is that after you divide it into 2 parts this thing is essentially brought over here. This thing is essentially brought over here and this is essentially brought over here. So now what I have is that 1, 2, 3 and let me write it by brown 1, 2, 3. Then 1, 2, 3 and let me show this also by brown 1, 2, 3 and last I have 1, 2, 3 and brown again 1, 2, 3. So what is done is that my token 1, this is now my token 1, this is now my token 2 and this is now my token number 3 and token 1 instead of token 1 having all these 6 values together in a row.

The first row now corresponds to my head 1 which are these 3 values over here and the second row now corresponds to head number 2 which are these 3 values over here. Similarly, for token 2 this is my head 1 and this is my head 2. Similarly, for token number 3 this is my head 1 and this is my head number 2. So this is what it means when we say convert a 1 by 3, 6 or 3 by 6 to 3 by 2, 3. So now this is a 3 by 2, 3. So, this is exactly what I have written over here, a 1 by 3, 2, 3 looks like this. There are 3, why is it 3, 2, 3 because there are 3 tokens, 3 tokens, each token has 2 rows and 3 columns.

So token number 1 has, sorry each token has 2 rows and 3 columns. So token number 1 has 2 rows and 3 columns, token number 2 has 2 rows and 3 columns, token number 3 has 2 rows and 3 columns. This is exactly what can be seen over here, token number 1 has 2 rows and 3 columns, token number 2 has 2 rows and 3 columns, token number 3 has 2 rows and 3 columns and what does each row in each token correspond to, the first row corresponds to head number 1 and the second row corresponds to head number 2. So now imagine that one token, we are looking at the queries, so the first token had some sort of input embedding that is split into 2 parts, half of it goes to head number 1 which is the first row and half of it goes to head number 2 which is the second row and that's done in a similar way for token number 2 and token number 3. So this is the reshaped queries matrix.

So essentially reshaping just means splitting it into 2. So visually splitting the matrix into 2 looks easier right, but when in the code you see these unrolling parts, in the code when you see these unrolling parts it just gets very difficult to visualize, but here I am deliberately showing this visualization to you so that it's actually very easy when you, what does it mean to unroll the last dimension, to unroll the last dimension just means that you are at this full dimension which is 6 dimensional token, you split it into 2 and bring the second half below the first part. So that leads to a 1 by 3 by 1, 3, 2, 3 reshaped queries matrix, a 1, 3, 2, 3 reshaped keys matrix and a 1 by 3 1, 3, 2, 3 reshaped values matrix. So this is done in the code also, in the code what we have written here is that unroll the last dimension, so earlier we had b, number of tokens, d out and now this is changed to b, number of tokens, number of heads, head dimension, number of heads is equal to 2 and head dimensions is 3, so that's 1, 3, 2, 3 that's exactly what's written here, we are going to unroll this to b, number of tokens, number of heads, head dimension and the way this is done is that the keys matrix which was originally there right, we do keys.view b, number of tokens, number of heads, head dimension.

Similarly for values, we do values.view b, number of tokens, number of heads, head dimension. Similarly for the queries, we do queries.view b, number of tokens, number of heads, head dimension. So these are the keys, values and queries which are reshaped.

So until now we are at this part where we have reshaped the keys, queries and values to take into account multiple attention heads. Then what we are going to do is that, so the keys, queries and the values have been obtained. Then what we are going to do, we are going to group the matrices by the number of heads.

So you see the problem here is that we have grouped by the number of tokens right, we see the token 1 and within the token 1 there is the head 1 and head 2. But now what we have to do is that we have to group it by the heads. So instead of 1,3,2,3, I want 1,2,3,3 which means I want to interchange this and this. So essentially I want the dimensions of my matrices to be b, number of heads, number of tokens, head dimension.

So what this will do is that you see the queries matrix initially we grouped it with token 1, token 2, token 3. But now we will group it with heads, so now this is my head 1 and this is my head 2. Then what will happen is that within each head, within each head there is this token 1, token 2 and token 3 and each token now has head dimensions which is equal to 3, so this is 3 dimension. Similarly if you do, if you look at head number 2, the first row of head number 2 corresponds to token 1, token 2 and token 3. So you see the difference between this 1,3,2,3, here the grouping was with number of tokens but now we have grouped it with head 1 and head 2. Why do we do this? Because it's just easier to multiply, right? If you see, if you see over here, the advantage of these heads is that the queries, the keys and the values have been split into multiple heads. So we should clearly see the different copies, right? Now, now that we have done this type of a grouping, we can clearly see the copies.

This is the first copy which is the queries, this is Q1, this is Q1 which is the queries for the first head. This is Q2 which is the queries matrix for the second head. So the division just becomes very easy whereas if you group it with the number of tokens, my head 1 is here, my head 1 is here and my head 1 is here.

So part of my head 1 is in this first row, part is here and part is here. So it needs to be grouped together in one single place. So that's why we group it with the number of heads because later remember that we have to then take the dot product between, so let's say this is my Q1 now, this is my Q2 now, this is my K1 and this is the K2.

We have to take a dot product between Q1 and K1 transpose and we have to take a dot product between Q2 and K2 transpose. Remember this is exactly what we had done over here. We had taken a dot product between Q1 and K1 transpose and we had taken a dot product between Q2 and K2 transpose.

To take this dot product, all the, to take this dot product essentially head 1 needs to be in one place, Q1 needs to be in one place, Q2 needs to be in one place, K1 needs to be in one place, K2 needs to be one place. So that's why it's very important for us to group the matrices by the number of heads instead of grouping by the number of tokens. And this is what is done in the next part of the code.

The next part of the code, we just take the transpose of these dimensions number 1 and dimensions number 2. So keys.transpose 1, 2 just means that since Python starts with 0 indexing, this is index 0, this is index 1 and this is index 2. So we are going to take the transpose of these indices. These are going to be interchanged to number of heads comma number of tokens. So now we are going to be grouping by the number of heads.

So all the keys, queries and the values matrices are now transposed. And we have 1 comma 2 here because 1 is this index and 2 is this index. So these need to be interchanged now.

That's what's done in this part of the code. So until now we have the Q1, Q2, we have the K1, K2 and we have the V1, V2. So if we were to map it out to the steps which we had seen in the previous lecture, we have reached this part of the code where we have Q1, Q2, K1, K2 and V1, V2.

Now let's go to the next part where we actually compute, where we actually compute the attention scores. To compute the attention scores, what we have to simply do is that we have to look at Q1 and K1 and take their transpose, we have to look at Q2 and K2, take their transpose. That's it.

So that's exactly what we are doing here. We are going to take the queries and we are going to multiply it with keys dot transpose 2 comma 3. Why 2 comma 3? Because now we are grouping by the number of heads, right? So when we look at one head, we essentially, so first what we do is that we look at the first head. So we have to multiply this with the transpose of this matrix.

So what does it mean taking the transpose of this matrix? So it means that so now the rows here are T1, T2 and T3, right? And the columns are the dimensions. So the rows here are the number of tokens and the columns are head dimensions, which means we have to take the transpose of these two. So multiplying K1 and multiplying Q1 and K1 transpose essentially just means taking the transpose of these last two dimensions over here for K1.

Why the last two? Because we are already so each row here corresponds to first token, second token, third token and each column corresponds to the dimension which are essentially corresponding to the last two dimensions. So multiplying K1, multiplying Q1 with K1 transpose essentially just means queries multiplies by keys dot transpose 2 comma 3. So now we have this entire queries matrix, right? When we take this queries matrix and when we multiply it with keys dot transpose 2 comma 3, what will essentially happen is that first Q1 will be multiplied with K1 transpose and then Q2 will be multiplied with K2 transpose. So that's essentially what we are going to get over here.

So if you think about the dimensions of the resultant matrix, what we are now doing is that we have this matrix queries which is b comma number of heads comma number of tokens comma head dimension and we are multiplying it with keys dot transpose 2 comma 3 which means these two are interchanged. So we are multiplying this matrix with b comma number of heads comma head dim comma number of tokens, right? So what will the multiplication result in? It's the number of tokens comma head dim multiplied by the head dim comma number of tokens. So it's going to be b comma number of heads comma number of tokens comma number of tokens and that's just going to be 3 by 3. So if you take this multiplication, you will get this matrix which is of the size 1 comma 2 comma 3 comma 3. But now what this means is that this is head 1, this is actually head 1 attention scores.

This is actually head 1 attention scores and this is actually head 2 attention scores and since these are attention scores, of course, their dimensions have to be number of tokens, number of tokens multiplied by the number of tokens. So this is how we get the two attention scores in matrix multiplication which is exactly what was done in the code also. Once we have the, sorry, which was exactly what was done in our visual lecture.

Once we have q1 and k1, we take this dot product. Once we have q2 and k2, we take this dot product. So q1 multiplied by k1 transpose gives us the head 1 attention scores, q2 multiplied by k2 transpose gives us this head 2 attention scores.

But I want you to pay very careful attention to the dimensions over here because the dimensions are where people usually get confused, right? So you have this head 1, head 2, head 1, head 2, head 1 and head 2. So you have q1, q2, k1, k2, v1, v2. Then what you have to do is that you have to multiply q1 with k1 transpose, q2 with k2 transpose. And when you do that, you finally get this attention scores matrix whose dimensions now are b, which is the batch size, number of heads because you have grouped by the number of heads, this head 1 and this head 2. And why is this 3, 3? Because since it's attention scores, it has to be number of tokens multiplied by number of tokens because the attention scores are calculated among every token.

So this is now the 2 attention scores matrix which we have for the 2 heads. And this is exactly what is done in the code also. To get the attention scores, we have to multiply the queries and keys.transpose 2, 3. This 2, 3 is very important because it's the last 2 dimensions which get transposed and which get multiplied.

So the ultimate dimensions of the attention scores after we take the dot product for each head is b, number of heads, number of tokens, number of tokens. This step is also called taking the dot product for each head. Why? Because first we multiply q1 with k1 transpose, that's taking the dot product for head number 1. And then we multiply q2 with k2 transpose, that's essentially taking the dot product for head number 2. So until now we have found the attention scores matrix.

Now what we have to do is that we have to find the attention weights. So this is to get this what we had seen in yesterday's lecture was to get the attention weights we have to basically first scale it, then apply softmax, then do causal attention and if needed we can do dropout. So, now this is exactly what is done in the mathematical calculations also, so let me take you through that.

Ok, so we have the attention scores matrices now. What we will first do is that we will first mask the attention scores to implement causal attention. So to do this, so this is the head1 attention scores and these are the head2 attention scores.

What we do is that the elements above the diagonal are replaced with minus infinity. We saw this in the causal attention lecture also. And the elements above the diagonal in head number 2 are also replaced with minus infinity.

And what we will do is that we will also divide by the square root of head dimension. Remember in self-attention we divided by the square root of keys dimension. But now the keys dimension is equal to the head dimension.

Each key dimension is equal to the head dimension which is dout divided by number of heads which is 6 divided by 2. So we will scale it by the square root of 3, we will scale it by the square root of 3 and then we will apply softmax. What softmax will do is that it will make sure the elements with negative infinity are set to 0. Remember in causal attention we cannot peek into the future. So for each token we only get the attention scores corresponding to that token and the tokens which come before it.

And why do we divide by the square root of head dimension? This is just to make sure that the variance of the query is multiplied by the keys transpose does not blow up. Dividing by the square root of the head dimension makes sure that the variance of that dot product between queries and the keys transpose essentially stays closer to 1. And that's important for us when we are going to do back propagation etc. We don't want values to be widely different from each other.

So when we apply softmax we get the attention weights matrix and remember that the dimensions of the attention weight matrix are exactly same as the dimensions of the attention scores matrix. It's going to be batch size, number of heads, number of tokens and number of tokens. So the same thing here batch size 1, number of heads.

So this is head number 1, this is head number 2 and then 3, 3 because I have number of tokens equal to 3. These number of rows and number of columns also equal to the number of tokens. But the difference now between the attention weights and the attention scores is that the attention score in the attention weights if you see every row, every row essentially sums up to 1. So we can also implement dropout after this but I have not implemented it here for the sake of simplicity. So now what you can do is that you can go to the code and you will see that the same thing has been implemented here.

First what we do is that we create a mask of negative infinity above the diagonal which has been done over here. We create this mask of negative infinity above the diagonal. Then what we do is that we divide by the square root of the head dimension and then we take the softmax and if needed we can also apply the dropout.

So if you scroll up to the top we can set the dropout rate. By default I think the dropout rate we can set it to equal to 0 if we don't want any dropout but if you randomly want to turn off certain attention weights you can do that by applying the dropout rate of let's say 0.5. So this is how the until now we have calculated the attention weights and then apply dropout. Then what we do after we get the attention weights, remember the last step after getting the attention weights is that we have to multiply the head1, we have to multiply the head1 attention weights with the value 1 v1 and we have to multiply the head2 attention weights with v2.

So let's see how that is done now in matrix multiplication. Alright, so this is the attention weight matrix. So this is head number 1 and this is head number 2 and these are my values matrix right.

So my values is this is my v1 and this is my v1 and this is my v2. So h1 and h2. So what I'll do is that I'll simply multiply these two together.

Now take a look at the dimensions of what exactly is being multiplied over here. So b, number of heads and b, number of heads that's the same for both these matrices or both these four dimensional matrices they are both grouped by the number of heads. But what really we should check while multiplying is that this is number of tokens by number of tokens so that's going to be 3 by 3 and this is number of tokens by the head dimension.

So that's also 3 by 3. So when you multiply this again the product is now taken into the number of tokens comma head dimension space. So we have three tokens here and each head dimension is equal to 3. So when you multiply the attention weights with the values you get the context vector matrices. So the first row over here is the context vector matrix for head 1 and the second is the context vector matrix for head number 2. And we have three tokens over here so there are context vector for each tokens and the size of each context vector is equal to the head is equal to the head dimension which is equal to the number of which is equal to the last dimension over here which is equal to the head dimension.

Now if you scroll to the visual multi head attention this is exactly what we had obtained yesterday right. We had obtained the head 1 context matrix and we had obtained the head 2 context matrix. Here also there are 11 tokens and the size of each context vector was equal to 2 which was equal to the head dimension in this case.

This is the same thing as what is being done over here. We have the context vectors for head 1 and we have the context vectors for head number 2 and when you go inside each head the size is number of tokens and each token has the context vector of size equal to head dimension. This is done in this part where we multiply the attention weights multiplied with the values ok.

When we get the context vector. Now what we do is that when we get the context vector remember that our final aim is not directly to get two different context matrices but we have to merge the context matrix for head 1 and the context matrix for head 2. We have to merge these context matrices right. We don't have to keep them separate.

So that part is still remaining right and to merge these what we have to do is that we have to again group by the number of tokens. So that's why we need to reshape it again. Currently the dimension is b comma number of heads right.

It's grouped by the number of heads. So we need to reshape it again. Remember we did this step earlier once where what here what we did is we actually switched it.

So we deliberately brought the number of heads before. So we want to group by the number of heads but now we'll switch it back to the original configuration so that we group it by the number of tokens. So this is now token number 1. This is now token number 2 and this is now token number 3. The reason I want to group it by tokens is that eventually I want to merge the head 1 and the head 2 output for each token right.

So token 1 it has the head 1 context vector and it has a head 2 context vector. I don't want it to be separate. I want to merge.

So then I'll merge these two together. Similarly for token 2, I have the head 1 context vector and I have the head 2 context vector. I don't want these vectors to be separate so I'll merge these two.

For token number 3, I have the head 1 context vector and I have the head 2 context vector. I don't want these to be separate so I'll merge these two. And this merging is just easier if I group it by the number of tokens.

So that's why we actually switch these positions once more and that's the reason that there is one more transpose 1, 2 here. So once we get the context vector matrix, we'll again transpose 1, 2 so that we'll group again by the number of tokens. And once we group by the number of tokens, what we'll simply do is that we will merge for the token 1, we'll merge the first row and the second row.

So it leads to six values which are the first two rows merged. Then for token 2, we'll merge these two, head 1 context vector, head 2 context vector. That will give me these six values.

And for token number 3, I'll merge these two vectors. So that will give me these six values again. So ultimately the final resultant context vector matrix which I have will be batch size, number of tokens and the output dimension.

So you see what we did initially, we started out with initially we started out with b comma number tokens comma d in right. Then we went through a bunch of steps. And then ultimately we obtained the context vector which is b comma number of tokens, b comma number of tokens comma d out.

This is the final context vector matrix. And this is again the last part of my code. The last part of my code is this context vector.contiguous.view. What this will do is that this will just merge the first row, second row of token 1, first row, second row of token 2, first row, second row of token number 3. And it will give me an output context matrix of size 1 comma 3 comma 6. Remember now the beauty of this context vector matrix is that whenever someone looks at this size, they'll just see 1 comma 3 comma 6. But the way we have reached this is that we have we actually obtained two context vectors, right? We obtained two context vector matrices and then we merged them together into one.

So this one final context vector matrix actually contains two perspectives. It contains perspectives from the head 1 as well as head 2. So it's much richer than having just the self-attention mechanism producing a context vector matrix. Because now we actually had multiple context vector matrices and we merged them together.

If there were six attention heads, we would have six context vector matrices which should be merged together.

(该文件长度超过30分钟。 在TurboScribe.ai点击升级到无限，以转录长达10小时的文件。)

(转录由TurboScribe.ai完成。升级到无限以移除此消息。)

That is the beauty of multi-head attention, although the dimension looks the same as it would have when we did self-attention, but now the matrix is much more richer since it captures multiple perspectives. That is it. This is the last step of the multi-head attention and I want to thank you all for sticking through this entire lecture and seeing all the steps especially when we look at matrices and dimensions things can get a bit complicated and when you look at this code directly, you will think it's a bit complicated, right, but it's actually very simple.

If you understand the mathematics with respect to matrices, then the code actually makes a lot of sense. This is the main class, the multi-head attention, which powers all the major large language models out there. Of course, there were a lot of improvements after this such as KV caching, multi-head latent attention, flash attention, etc.

But if you understand these three dimensions, b, number of tokens, din, if you understand what this keys.vue, values.vue, queries.vue does, what transpose means, transpose 1, 2, how it relates to the handwritten exercise, you will find these things are not very difficult. So I highly encourage all of you to take a piece of paper and write all these things down as if you're following this lecture seriously. So once this class is actually defined, you can, what you can simply do is that you can just take an inputs vector, which is 1, 2, 3. So I have two, I have three rows over here, my three tokens, and each token is a six-dimensional vector, the same example which we saw on the notes.

And then we can just have, so here I'm having two batches though, I've stacked the inputs on top of the inputs. So remember, although in the handwritten notes, we just took one batch, the code is powerful enough to change the first dimension to even two. So that's what I'm considering over here.

I'm stacking these two inputs, one on top of each other to create a batch. And then I just pass this entire input to this multihead attention, that's it. And then I create the context vectors.

So you'll see the first context vector is 1 by 3, 6, this is exactly the context vector shape, which we had seen over here, 1 by 3, 6. And the second context vector is 1 by 3, 6. So this is the first batch, this whole thing is the first batch, and this whole thing is the second batch. That's why we have the size here, 2 by 3, 6, if we had three batches, it would have been 3, 3, 6. So just within 5 to 6 lines of code, we have implemented the multihead attention calculation. And here if you scroll above, these are 20 to 25 lines of code, which is the mechanism which powers or which is the brain behind how why large language models work so well.

These 25 lines of code actually encode the key advancement which happened in 2017, when the transformer block was introduced for the first time. And if someone just takes a look at this code, they'll find it difficult. But my main purpose of today's class was to link it to handwritten notes of mathematical derivation and also to an intuition which we looked at in the previous class.

Only then I showed you the code so that you don't get scared or intimidated by the code. But if you're seriously interested about developing your understanding and never forgetting how multihead attention works, take a piece of paper, write everything down so that you don't forget this at all. Now that we have completed this lecture, we have done multihead attention.

So we have finished these three parts. We are now fully ready to start learning about key value cache. Key value cache is that main mechanism which made multihead attention much more efficient.

And this serves as the bridge towards finally understanding multihead latent attention. That is the real key innovation which was implemented in the DeepSeq paper. But to understand key value cache and to understand multihead latent attention would have been very difficult for you to understand this if you if you did not understand today's lecture.

So that's why we had all these lectures on self-attention, causal attention, multihead attention. So I want to congratulate you and thank you for reaching this part. Please stay with me.

The later parts will be even more rewarding now that you have finished completing the lectures until here. So thanks a lot everyone. Please make notes along with me so that you learn the most.

Thanks everyone. I look forward to seeing you in the next lecture. Hello everyone.

My name is Dr. Raj Dhandekar. I graduated with a PhD in machine learning from MIT in 2022 and I am the creator of the Build DeepSeq from Scratch series. Before we get started, I want to introduce all of you to our sponsor and our partner for this series, NVIDIO AI.

All of you know how much we value foundational content, building AI models from the nuts and bolts. NVIDIO AI follows a very similar principle and philosophy to that of us. Let me show you how.

So here's the website of NVIDIO AI. With a small engineering team, they have built an incredible product in which you can create high quality AI videos from just text prompts. So as you can see here, I've mentioned a text prompt, create a hyper-realistic video commercial of a premium luxury watch and make it cinematic.

With that, I click on generate a video. Within some time, I am presented with this incredible video, which is highly realistic. What fascinates me about this video is its attention to detail.

Look at this. The quality and the texture is just incredible. And all of this has been created from a single text prompt.

That's the power of NVIDIO's product. The backbone behind the awesome video which you just saw is NVIDIO AI's video creation pipeline in which they are rethinking video generation and editing from the first principles to experiment and tinker with foundational models. They have one of the largest clusters of H100s and H200s in India and are also experimenting with B200s.

NVIDIO AI is the fastest growing AI startup in India building for the world. And that's why I resonate with them so much. The good news is that they have multiple job openings at the moment.

You can join their amazing team. I am posting more details in the description below. Hello everyone and welcome to this lecture in the build DeepSeek from scratch series.

Today, we make progress towards understanding one of the key innovations in the DeepSeek architecture and that key innovation is multi-head latent attention. DeepSeek, when they wrote the architecture of the model which was eventually going to transform the whole LLM landscape, one of the key innovations was this concept of multi-head latent attention. And through this innovation, they unlocked speed improvements, performance improvements in the transformer architecture itself.

Now to understand multi-head latent attention, we cannot go to understand this concept directly. We have to first understand the key value cache and the main purpose of today's lecture is to understand the key value cache. So I'm going to divide today's lecture in three parts.

First I'm going to explain to you what exactly happens during the inference time of a language model. Second, we are going to understand what is this key value cache and why do we really need it. And third thing which we will understand is that once we understand why we need the key value cache, how to implement the key value cache.

This lecture serves as the foundational building block towards finally understanding multi-head latent attention because latent attention would not have been developed if it were not for the key value cache. At the end of today's lecture, we will see that key value cache has some disadvantages. Of course, it leads to a huge number of improvements, but it comes with a dark side.

And to deal with this dark side of key value cache, latent attention was ultimately innovated. But to understand the dark side of the key value cache, we will need to first understand the concept in a lot of detail. I took a long time to prepare this lecture because I want all of us to understand every single thing from the very basics.

So most of the lecture is going to be on the whiteboard where as always I'll explain everything from scratch and then towards the end of the lecture we also have a coding component. So if you recall what all we have covered in this series so far, first we started with the concept of self-attention, then we learned about causal attention and in the previous lecture we learned about multi-head attention. If you have followed until this part, then you are aware of the concept of attention and now you are fully ready and equipped to understand the key value cache.

So without any further delay, let's get started with this very important concept. One more thing before we go ahead is that if you look at the OpenAI API and its pricing, let's look at the pricing for this GPT-4 which is mentioned over here. It's about $30 per million token.

Now we have another GPT-4 here with 32k that's $60 per million tokens. So why is there a price difference of two times? One thing to note here is that this 32k or 32,000 is the context size. For GPT-4 the context size is just 8,000 but here the context size is 32,000.

This clearly shows that if the context size actually increases, the price also increases proportionately and today we are going to learn that the key value cache is one very major reason for this price increase. Once we implement the key value cache, it comes with certain limitations and that's why those limitations depend on the context size and that's why as you increase the context size OpenAI charges you a lot more. This also will be clear once we reach the end of today's lecture.

So the first thing which we have to understand in this lecture is that key value cache or KV cache only comes into picture during the inference stage of LLMs and that's the most important thing to notice. So people who are watching this, stop and think whether your concept about pre-training and inference is clear. So when an LLM is used, it is actually decomposed into two parts.

First we have to pre-train the language model and then you get a pre-trained LLM. On the pre-trained LLM, then you perform inference which means you ask questions and then the LLM responds. To give an example, let's say if I go to chat GPT right now and then let me ask some questions such as make a travel plan for Italy.

If I ask this question right now, I am not doing pre-training here, I am in the inference stage because the model which is printing this response is already pre-trained but here the pre-trained model is used for inference which means for predicting one new token every single time and key value cache, the whole discussion which we are going to do in today's lecture only applies during the inference stage of the language model. This means that imagine your model is pre-trained, if the model has 175 billion parameters, all of those parameters are pre-trained and they are fixed. Now we are using that model to predict the next token.

So what actually happens during the inference stage is that it's a next token prediction task, right? So we are going to predict the next token during the inference stage. And this is the main purpose of inference. The model is now trained, all the parameters are fixed, we predict one token at a time.

So the first concept to understand is that key value cache only is applicable during the inference stage. So first let us understand what actually happens during the inference pipeline of language models. So let's say we ask or we give some sentence to the language model and here I have a simple demonstration for you where I have given some prompts such as the next day is bright and I am using GPT-2.

This is a pre-trained GPT-2, okay? I am going to pass this prompt to GPT-2 and then you will see that the model prints out one token each time, see over here what is happening. One new token is being printed each time during the inference stage. So the next day is bright and then the next token is and the next token is sunny, etc.

So I'll run this code again, just keep an eye out what is happening over here. So if you look at what is happening over here, you will see it proceeds very fast, but one new token is being printed every single time. And when you are interacting with chat GPT, the UI or the interface is such that you feel that everything is presented at once, but actually one new token is generated for you every single time, okay? So this is the main thing which is happening in the LLM inference and the way it works at a more deeper level is that let's say if the input is the next day is, it goes through the LLM architecture or the LLM pipeline, we'll see that in a moment.

The new token is predicted, let's say the new token is bright, that new token is then added back to the input. So the way it happens is that the next day is, okay? And let's say we are doing the inference and the next token predicted is bright, this bright is then added back over here and now the new input is the next day is bright. Now this whole sentence goes through my inference pipeline and the next token is predicted which is and this is now again appended back and now the new input sequence is the next day is bright and then it again goes through the inference pipeline and then the new token is predicted which is lovely, let's say and then this is again appended back to the input sequence.

This loop which I have shown over here is very important to understand the concept of key value cache. When an input sequence is given, it passes through the whole LLM architecture, the new token is predicted, that new token is appended back to the input sequence, that input sequence goes through the LLM architecture again, again a new sequence is or again a new token is predicted, that new token is again appended back to the input sequence. This goes on until we reach the specified limit of a maximum number of new tokens or until when the response is finished.

So keep this flow in mind. So now let us understand what happens when the input sequence goes through the LLM architecture itself to produce the next token. So if you remember what we have learnt in the previous lecture, this is the entire, this is the sequence through which the input tokens or input sequence actually passes.

They first go through the data pre-processing pipeline which is this whole, they are tokenized, then they are converted into token embeddings, positional embeddings are added, the input embeddings then go into the transformer block. In the transformer block, there are all of these different layers and then we finally come out of the transformer block, go through another layer of normalization, have an output layer and the next token is predicted. Now I want you to understand some very key things over here.

Let's say the input is the next day is. The input is the next day is and what I want to do is that after is, I want to predict the next token. What happens through this entire architecture now is that all of these are converted into input embeddings and they are passed to the transformer.

This much is fine. Now this, all of these input embeddings go through layer normalization, then they go through multi-head attention. Now this is that part where, so we have to predict what's the next token which comes after is right.

So in this part, we actually get the attention scores between is and all the other tokens and then what we do is that at this stage, we get the context vector for is. We get the context vector for is, which is the last token in my input sequence. We get the context vector for is and then the key thing to remember is that this context vector then goes through all of these other sequences and ultimately there is something like a logits matrix which comes out, which means that when is the context or when is we are focusing on is as the query and if the vocabulary size is 50,000, we get this vector of probabilities with length of 50,000 and then in this vector, we look at that index with the highest probability and that is going to be my next token, which is bright.

So, this is how the next token prediction task actually happens and let me explain this once more because I think this is the most important part to understand the key value cache and I don't think many people actually understand this in detail. So what I am trying to say over here is that this my task is the, let me increase the thickness a bit over here, let's say the input is the next day is that's the input right and what I want to do is that I want to pass this input through this entire architecture and I want to predict the next token. So the way this next token is predicted is that at the end of this architecture, I get something like a logits matrix and the logits matrix is actually computed for all the tokens the next day is and then this is a 50,000 vector, this is a 50,000 dimensional vector, this is a 50,000 dimensional vector, this is a 50,000 dimensional vector.

The size of all of these is 50,000 because that's my vocabulary size. One very important thing to remember here is that to predict the next token, I don't care about these at all. I don't care about this.

I only look at the logits vector for is, I only look at the logits vector for is and I look at that index which has the highest probability and then I decode that index and that predicts the next token. This insight is very important when this entire input sequence goes through this architecture and finally we get the logits matrix, only the last row of the logits matrix which is the vector corresponding to is that's the most important. Now what does this mean? This last row of the logits matrix ultimately depends on the context vector for is.

This last row of the logits matrix, so if we get the context vector for is over here, after this point the other tokens don't matter at all. The other tokens the next day only matter and after we come out of the attention block. Why? Because in the attention block we need to get the attention scores between is and all the other tokens.

So until this point all the other tokens are important but once we get out of the attention block at this point the only thing which matters is the context vector for is. This context vector for the last token then travels the rest of the pipeline and then we get this final row of the logits vector and then we can predict the next token. So what I am trying to say here is that when we are predicting the new token, we have to get the context vector for the last token which is is, to get that context vector we use the multi head attention block but after we come out of the multi head attention block and once we have that context vector for the is, then we don't need these other tokens at all.

We just need that one context vector to predict the next token. So first insight is that I only need, we only need, we only need the context vector, we only need the context vector for the last token in the input sequence to predict the next token. Please keep this in mind.

Why do we only need the context vector for the last token? Well because to predict the next token in the logits matrix, only the last row of the logits matrix actually matters and the last row of this logits matrix only depends on the context vector for the last token which is is. So we don't need all of these other tokens once we get the context vector for is. It's like using the other tokens till we get the context vector for is and then discarding them because now once I have the context vector for is, I know how is attains to all the other tokens and then I can predict my next token.

Keep this insight in mind, this is very very important to understand key value cache. So what I have written over here is that when a new token comes in, when a new token is there it is added back to the input sequence, the input sequence is converted into input embeddings and then we only need the context vector for the last token in the input sequence That context vector gives us the logits vector for the last token in the input sequence and that helps us to predict the next token. Keep this journey in mind, imagine the whole input sequence passing through the LLM architecture.

When the whole input sequence passes through the LLM architecture, finally we get an entire logits matrix like this. But in this logits matrix, all of these other, the first three tokens are irrelevant because we have to predict the token after is. So the only thing which is relevant over here is this last row, which means the only thing which is relevant is the context vector for is, that's all.

Now you must be thinking why is this insight needed, where is this used? It will be used later, just bear with me for a moment. Now think about the inference process which I mentioned, you get one sentence, you predict the next token, then you append that next token over here. So let me write it down, let's say you have the input sequence as the next day, that input sequence goes through this entire pipeline and I predict the next token which is bright.

Now this bright is appended to the input sequence again and my new input sequence again goes through my entire pipeline, my new sequence again goes through my entire pipeline and then I have the next token which is is. Now the new input sequence is the next day bright is, that again goes through my entire pipeline and I have my next token which is let's say and and you see after every inference the token is appended to the next input. So already you must be seeing here right, first this input sequence goes through the entire pipeline, now this input sequence goes through the entire pipeline, this input sequence then again goes through the entire pipeline.

It seems that a huge number of computations need to be performed during the inference and not just that, intuitively it seems that we might be performing the same calculations again and again right, which means the next day I already passed these three inputs through the entire architecture. Now do I have to again pass these three inputs through the entire architecture, then do I have to again pass these three inputs through the entire architecture, it seems that we are just performing the same computations again and again and again right, we are passing the same tokens in the input sequence through the entire architecture once a new token is generated. So just understand or intuitively imagine you have constructed the LLM and you are getting the inference but you think about the fact that actually a lot of in computations need to be performed during inference and maybe we are repeating the computations and what is the problem with repeating the number of computations, the main problem with repeating the number of computations is that for every piece of data stored in the memory we pay a price, think about data as houses right, imagine you have a huge land and there is a house for every land area occupied by the house you have to pay a price right, similarly think of the memory as that area which we have which is very precious to us, for every piece of that memory occupied with data we have to pay a price ok and the more the memory is occupied the higher the price we have to pay during inference.

The thing is the more number of computations you perform, the more number of repeated computations you perform, the more amount of data you need to store in the memory that leads to more amount of computations that leads to more price which you have to pay during the inference that is one reason intuitively why a higher context size is priced higher, we will come to why context size matters in a moment but just remember that more context size means more memory and every piece of data stored in the memory we pay a price. So now until now what we have seen is that many computations seem to be repeated during the inference that is number one, second insight which we have is that actually during the inference only the context vector for the last token matters, so maybe we can do something more efficient. So then the question is can we do something to reduce the memory storage during inference and once you start asking this question that once you intuitively realize that ok I seem to be doing so many repeated computations during the inference, can I do something to reduce the memory storage and this is where the key value cache or this is where the key value cache actually comes into the picture.

We will now see mathematically, so we will prove that we are actually repeating the computations currently I just intuitively showed you that we might be repeating some calculations right, now we are actually going to take a hands-on example and I am going to show to you that during inference we are actually repeating many computations, then we will see what to do to avoid those repeat repetitions and to implement that logic of avoiding repetitions we are going to use this intuition which we have seen that we only need the context vector for the last token. So I hope you are excited for the next part of this class where we are going to dive a bit into visual mathematics. Alright, now I want to prove to all of you that we are repeating calculations or repeating computations during inference and many things can be optimized further, let me prove that to you.

So first we have to look at the self-attention mechanism and we have seen this a lot in the previous classes. If you are not aware of this mechanism I encourage you to go through the previous classes. So here is what happens during the attention mechanism, let us say the input is the next day is.

So imagine now that when we are looking at the attention mechanism we are focusing on this block right now. So here is what happens, the input is the next day is that is the input sequence and we are considering an 8-dimensional input sequence and we have 4 tokens that is why the input embedding matrix is 4 by 8. We multiply it with the trainable query weight matrix, trainable key weight matrix and trainable value weight matrix.

(该文件长度超过30分钟。 在TurboScribe.ai点击升级到无限，以转录长达10小时的文件。)

(转录由TurboScribe.ai完成。升级到无限以移除此消息。)

The queries matrix, the keys matrix and the values matrix, all of these are 4 by 4. Then we multiply the queries with the keys transpose and that gives us the attention scores. Every row of the attention score corresponds to one query, which let's say is this and how it attends to the other tokens in the input sequence. That's why the attention scores is 4 by 4. The size of attention scores will always be number of tokens, number of tokens comma number of tokens.

Since we have 4 tokens here, the next day is it's a 4 by 4 matrix. Then once we have these attention scores, we scale it by square root of the keys dimension, we apply softmax and apply causality so that all the elements above the diagonal are 0. These are our attention weights and then we are going to multiply the values vector or we are going to multiply the attention weights with the values matrix and that gives us the context vector matrix. So once we come out of this attention block here, we have this context vector matrix.

So then we have context vectors for every or sorry the context vector for the next day and is. So these are the context vectors. They are much more richer than the input embedding vectors because each context vector captures information of the neighbors.

That's how the self-attention mechanism proceeds. Now we are doing inference, always keep this in mind, this entire lecture is in the domain of inference. We are not doing pre-training.

So now let's say we have to predict the next token. We have next day is and during this entire pipeline, once we have this context matrix, it goes through all of these rest of the layers, we get the logits matrix and we predict the next token. So the next day is goes through this entire transformer block during inference, we get out, we get the logits matrix and the next token is predicted.

Let's say the next token is bright. Now imagine or remember what I told you, the bright which is the next token will be appended to the previous input. So now during the next inference stage, my new input matrix becomes the next day is bright.

Correct? So now my input matrix is 5 by 8 because there are 5 tokens now and now this input matrix will again go through the entire attention mechanism. We will multiply it with the queries keys values. We get the queries keys values.

Now those are 5 by 4, 5 by 4, 5 by 4. The attention scores matrix is 5 by 5. The attention weights is 5 by 5 and the context matrix is now 5 by 4 because the first is for every or sorry the first row is for the next day is and the last row is for bright. Okay, so now take a look at these computations and take a look at the earlier computations which we did. Do you notice something repeating? So in the earlier computation, there are 4 tokens, right? The next day is right and in the next computation, there are 5 tokens.

The next day is bright. WQ, WK and WV are the same matrices in this step and in this step. So what is repeating when we computed in the previous inference step and when we computed the attention scores in the next inference step? What is exactly repeating? You can pause this video for a while here.

Okay, so now let's say if you take a very closer look at this new context matrix calculation, you will see that everything which I have shown in this black box is repeated. So these three which are marked in the black box, let's say these are 4 by 4, right? The first black box is 4 by 4, the second black box is 4 by 4 and the third black box is 4 by 4. We already computed these queries, keys and values 4 by 4 matrices in the previous inference step. That is the first repetition which you should be aware of.

Then let's look at the attention scores. In the attention scores, currently we have a 5 by 5, right? Actually let me remove this. We have this 5 by 5, correct? But now I am marking one 4 by 4. This 4 by 4 which I have marked over here, we have already computed this 4 by 4 attention score in the previous computation.

We have already computed this. Similarly when we go to the attention weights now, let me mark this. We have already computed this 4 by 4 in the previous attention weights calculation because previous attention weights was also 4 by 4. So you see the problem which we are, or the repetitions which we are doing here, we are again repeatedly calculating the same things again and again.

We are recalculating the queries, keys and values. We are recalculating the previous attention scores. We are recalculating the previous attention weights.

We are also recalculating these four values of the context vector matrix which was already computed before. So, why are we doing all of these re-computations? In fact, keep in mind that our whole goal of this task is to predict the next token, correct? Our whole goal of this task is to predict the next token and from our learning before to predict the next token after this, what do we need? To predict the next token, all we need is that to predict the next token, we only need the context vector for the last token which means that after doing all of these repetitions, the only thing which we are going to use is this context vector for bright because that's my last token currently. This context vector for bright is the only thing which we need to predict the next token.

So why are we computing all of these things unnecessarily? So now let us formalize this and try to quantify what are we computing again and again and what can we do to solve it? So think about this, right? It looks like we are unnecessarily doing a lot of repeated calculations. Can we optimize this? And already you can think that one way to optimize this is that these query or these keys and values matrices which I have computed previously, why do I need to compute again? Can I just store the keys and values matrices from my previous calculation? That's where the concept of caching comes into the picture. Caching is basically just storing the previously computed values so that you can use them in the next iteration.

So then you might be thinking, let me store the queries also, let me store queries, keys and values, let me store all of that. Now let's come to what exactly we need to store. Now we have understood that repeated calculations are happening and we need to store something in memory to make our computations more efficient as the next stages of inference proceed ahead.

Now we are going to see step by step what exactly we should store. The input is the next day is bright and our goal is to predict the next token. So the next day is bright will travel through this entire architecture and we have to predict the next token.

Remember what I told you earlier to get the next token prediction when we reach this final layer, when we reach this final layer, we will have a logits matrix, we will have a logits matrix for the next day is bright. This is the logits matrix when we reach the end of this computation. So when we reach at this part, we will have the logits matrix and the size for each will be my vocabulary size, all of these.

Now out of all of these vectors, I don't care about these first four at all. I don't care about this because I want to predict the next token. I only care about the logits vector for bright, then I am going to look at that index with the highest probability and predict the next token.

So I only care about the logits vector for bright. As a result, I only care about the context vector for bright, which is computed at this stage. I only care about the context vector for bright.

After this stage, all the other tokens do not matter to me at all. So after this point, once I get the logits vector for bright or sorry, once I get the context vector for bright, my context vector can go through all of these layers and only that context vector influences my logits vector. So I only need that.

I don't need other tokens after that point. So after this point, which is marked in orange right now, the earlier tokens are not needed. To generate a new token, we only need the hidden state or the context vector of the most recent token.

None of the other context vectors are needed. Keep that in mind. So when you look at this context vector matrix, when you look at this context vector matrix you had obtained earlier, we don't need the entire context vector matrix.

We only need the context vector which is corresponding to bright. That's the most important realization. Once you realize that you only need the context vector for bright, let's backtrack now.

So we only need this. That's what we have understood up till now. We have got the whole context vector, but I don't need these other context vectors at all because they don't influence this final logits vector.

So I only need the context vector for bright. Now let us backtrack and check what we actually really need. To get this context vector for bright, what do I need? How will this context vector for bright be calculated? I need the attention weights only for bright and I will multiply it with the entire values matrix.

That will get me the context vector for bright. So I need the attention weights for bright and I need the values matrix. How do I get the attention weights for brights? To get the attention weights for brights, I need the attention scores for bright, which means how bright relates to all the other tokens.

The next day is bright. I need to get these attention scores. These are these attention scores, 1,5.

So I need 5 values, 1 by 5. How do I get these attention scores? I get these attention scores because I need only the query vector for bright multiplied with the keys transpose. That's all. So, I will need the, so here is what I will actually need.

I will need my values matrix. That is what I will need and I will need my attention weights. To get the attention weights, I need attention scores.

To get attention scores, I need a query vector for bright multiplied by the keys transpose. Now how do I get the values vector? How do I get the values or how do I get the values matrix? Let's say this is my whole values matrix. So let's start from the top.

To get the context vector for bright, I need attention weights for bright multiplied by values matrix. What I have marked in my black box is the cached value matrix, which is the value matrix coming from the previous iteration. So the top 4 rows I will get from my cache.

My bottom row, which is the value vector corresponding to bright, I just take the input for bright multiplied with the trainable weight matrix for bright. And this is how I get the value vector for bright. This is the only new computation which I have to do.

For all these other value vectors, I can anyway cache them from my previous iterations. Then to get the attention scores, we need the query vector for bright multiplied by the keys transpose. To get the query vector for bright, I just take the input for bright multiplied with the trainable query weight matrix.

And if you look at the keys transpose, same to values, same as the values, the first 4 rows of this can be cached. I don't need to recompute this again. But only the last row, which is the keys corresponding to bright will be the input corresponding to bright multiplied by the trainable key matrix.

That's all. So if I zoom out a bit, these 3 boxes, number 1, number 2 and number 3 are the only 3 new computations I need to do for every inference. So take a look closely at these 3 boxes.

What are these boxes? These boxes are just the input vector for bright multiplied by the trainable query, the trainable key and the trainable value matrices. And these trainable matrices, this WQ, WK and WE are already fixed, because WK, WQ and WE are fixed during pre-training, we don't need to compute them again. So they are already fixed.

I don't even need to cache them, they are fixed values, they are fixed during pre-training. So I get my input token. So once a new token comes in, here's what I have to do.

Once a new token comes in, I find the query vector corresponding to the new token first by multiplying the input embedding for that token multiplied by WQ. Then I get the query vector for the new token. Then what I do is that I have already cached my previous keys.

Then I compute the new key vector for the new token. I compute the key vector for the new token by multiplying the input embedding with WK. This is the key vector for the new token.

I do not compute the key vectors for the previous tokens because they are cached. Then I augment my new key vector with my previous cache to get the whole keys matrix. Then I multiply the query vector with the keys transpose, I get my attention scores.

I scale them, apply softmax and causality to get my attention weights. Once I have my attention weights, I calculate the values vector for only the new token. By multiplying the input embedding with WE, I get the value vector for the new token and then to get the values matrix, I just append the new values vector with the cached values.

So I do not compute this cache again. And then I multiply the attention weights multiplied by this value vector and I get only the context vector for bright, that's it. I only get the context vector for bright.

I do not care about the other context vectors at all. So again if I zoom out, pay careful attention to how many caches I need. I only need to cache the keys and I only need to cache the values.

I do not need to cache the queries at all because I just need the new query vector for my new token. Since we only need to cache keys and values, this is called as key value cache. This is called as key value cache and sometimes it is also referred to as the KV cache.

So once we cache the keys and the values, every time a new token comes in, we do not need to recompute those previous keys and values. So again let me repeat what happens when a new token comes in. When a new token comes in, I first multiply the input embedding with WQ, WK and WV.

Only three computations need to be done. Then I get the query vector, the key vector and the value vector for the new token. Based on the query vector, I multiply the query with the keys matrix.

So to get the keys matrix, I use the keys cache and append my new keys vector. So then the query vector multiplied by the keys transpose gives me attention scores. Then I get the attention weights and then I use the cached value matrix and append the new value vector for the new token.

That's how I get my new values matrix. Then I multiply my attention weights with the new values matrix and that's how I get my context vector for the new token. That's it.

Then we get the context vector for bright and then this context vector at this stage then it passes through the rest of the architecture. When it comes out, we get the logits vector. We get the logits vector only for bright.

Then we look at this logits vector and we find that index with the highest probability and that gives us the next token. This is what is meant by key value cache. We just store the previous keys and values but to understand key value cache, it is very important to understand this intuition that to generate a new token, we only need the context vector of the most recent token.

In this case, it was bright and this insight actually helped us to know what to cache. Once we got this insight that I only need this new context vector, then you see how we backtrack. Then we know what all to cache.

Then it becomes very easy. We only need to cache the keys and the values. We don't need to cache queries.

That's all and only three new computations need to be done every time a new token comes in. The token embedding multiplied by WQ, WK and WE. That's all.

Then the keys are appended to the cache, the values are appended to the cache and we get the context vector for the new token. That saves a huge amount of computations during inference. We don't need to compute every single thing again and again like we saw over here.

These black boxes were re-computed again and again, right? All right. So until now, what we have learnt is that we need to cache the keys and the values matrices and this is called as the key value cache. We don't need to store or cache the queries matrix at all.

Now, what's the use of the key value cache? The main use of the key value cache is that as the number of input tokens increases, the amount of compute time increases linearly, which means that we can speed up the computations by a huge factor. Why do we speed up the computations? Because we are not doing repeating calculations again and again. Earlier, if you notice what was happening, what was happening here is that this the next day we get bright, bright is appended.

Then the next day is used again for the next calculation. Then it's used again for the next calculation. These repeated calculations lead to quadratic computations or quadratic complexity.

What quadratic complexity actually means is that as the number of tokens increases, as the number of tokens increases, if we don't do caching the amount of computations which we need to do increase in a quadratic manner. But once we do caching, so once we do caching, what actually happens is that if you see the same example, this once it's used is cached. It's not computed again.

It's cached. Then once this is used, it's cached. These keys and values are cached.

So we don't need to compute this again and again. As we saw the keys and values vector are cached. So we don't need to compute these black boxes again and again.

This caching helps us because it ultimately leads to a linear compute time. This caching leads to a linear compute time instead of a quadratic compute time. So as the input tokens increase, the compute time increases linearly, not quadratically.

So once we use the k-value cache, computations speed up because we don't recompute the same thing again and again and again. That's the whole advantage of using a k-value cache that we can just store the variables in memory and then we don't need to reuse or we don't need to recompute the same thing again and again. Remember, when we recompute the same thing, the number of computations increase.

Then the cost also increases. As we saw earlier, every single compute instance takes cost. So the moment we reduce the number of computations, we reduce the cost.

And that is the key advantage of k-value cache. So it seems that everything is amazing, right? This k-value cache, we can cache things in memory, we can store them and that reduces the number of computations. So it's good for us, right? What are the disadvantages? Well, the k-value cache, remember I told you that it comes with a dark side.

The k-value cache comes with a dark side and that is with respect to the size of the k-value cache. Whenever we store something in memory, it takes up space in the memory. So we have to pay more.

So speed is important, that's fine. But another consideration is memory, right? We are avoiding recomputations, but the disadvantage is that we are storing something in memory. Caching essentially means that we are storing something, right? And remember what we talked about data taking a footprint.

So every time we store something, it takes a footprint. It's like occupying land. So we have to pay the rent, we have to pay the cost.

So the more amount of space which is taken up by data, the more cost it will incur. So if you take a closer look at the size of the k-value cache, the way the size of the k-value cache is actually computed is that, let's say we have four tokens, right? The next day is every token will have a certain number of embedding dimension that is equal to the number of attention heads, the number of attention heads into the attention head dimension. This is the dimension of every input.

Now the thing is there are these many number of inputs and the number of inputs which will be there or the number of tokens in one sentence is decided by the context size. So if the context size is S, in this case the context size is 4, but in large models, the context size is 1000, even 10,000, 100,000, etc. So N into H into S and that is for one batch.

If you have multiple batches, it's N into H into S into B. So N into H is the dimension, S is the context size, B is the batch size. So you'll see this N into H into S into B. So in one transformer, the size of the cache for one transformer is this, because we have to save all of this. All these number of tokens we have to save and the dimensions, right? So if there are four dimensions, one, two, three, four, every single thing here carries some weight or rather it occupies some space.

So we have to pay for it. Put it another way. We take the same thing, the next day is right.

So in this case, we have used a context size of 4 and the dimension of 4. So N into H is 4, S is equal to 4 because that's the context size and the batch size is equal to 1. So for every single parameter, we have to pay. So how many parameters are there? Number of dimensions, which is N into H, number of tokens, which is given by the context size and number of batches. So here we have N into H into S into B. These many parameters we have to save for one transformer.

Now remember, when we have language models, there are multiple such transformers. So we have to take that factor also, which is L, which is the number of layers, right? So L into B into N into H into S. Now we have to have one cache for keys and one cache for values. So multiplied by 2. And then here we are assuming that every single thing is a floating point, which is essentially 16 bit.

So we are assuming 2 bytes per floating point. So this will be further multiplied by 2. So the size of the KV cache is given by L into B into N into H into S into 2 into 2. So keep in mind here that one key variable, which affects the size of the KV cache is context length. As the context length increases, the size of the KV cache increases.

So we have to store more memory or we have to store more parameters. So we have to pay more. And that is the reason why when we increase the context size, remember we saw this at the start, when we increase the context size, the amount of parameters which we need to store increases because the size of the cache also increases.

And that's one reason why more context size during inference, open-air charges more. So this is the size of the KV cache, right? And let's say if we have a 30 billion model whose number of transformer blocks is 48, the context size is 1024, number of dimensions is 7168 and batch size is 128. This leads to a KV cache size of 180 GB.

That's huge. If we consider DeepSeq R1 or their V3 model, V3 base model, they use number of transformer blocks as 61. If we use the batch size equal to 1 during inference, then the number of attention heads which they have is 128 and the size of each or the dimension of each head is 128.

And the context size is actually 100,000. So in this case, if you multiply all of this, the KV cache size becomes equal to 400 GB. That's huge.

This is a huge amount of size. And once that is stored in the memory, that reduces or that this much amount of parameter stored in memory will reduce the other computations also, will reduce the speed of other computations also. We'll have to pay more for this much amount of storage.

So then you might be wondering, how does DeepSeq charge so less during inference? It's because they figured out ways to deal with this. They don't use this variant of the KV cache at all. In fact, this plot also shows that as you, so here is GPT-3 small, then we have GPT-3 medium, GPT-3 large, GPT-3 XL.

So as we go from left to right on the x-axis, the GPT-3 size increases. And as you see, as the model size increases, the number of transformer blocks increase, the number of attention heads increase. There you'll see on the red, on the red, we have the KV cache size and the blue is the number of parameters.

So number of parameters, of course, increase. But if you look at this red curve, it's the KV cache size that increases by a huge amount. In fact, it increases in a quadratic or slightly accelerated manner as the size of the model increases.

That's one huge disadvantage of the KV cache. KV cache speeds up things, but it takes space. And this dark side of the KV cache is that it takes space and you have to pay more.

It reduces the speed of my other computations because my memory is now occupied. All of this needs to be solved. And to solve this multi-head latent attention is one mechanism.

And then eventually we'll understand multi-head latent attention once you understand this dark side of KV cache. But remember, this dark side is what motivated people to later have things like multi-query attention, group query attention, which we'll learn about. And then ultimately, deep-seek invented latent attention or multi-head latent attention to deal with this dark side.

All right. I hope now you have understood the KV cache advantages, disadvantages, etc. I just want to end this lecture by showing you a small code which I've implemented and that compares GPT-2 inference with and without KV cache.

So let's go over here right now. Let me run this once more. So if you see over here, what I've done is that I've done something very simple.

I've again used GPT-2. I have scrolled down a bit. Yeah, I've again used GPT-2, which is the model which I've used from Hugging Face.

Then the prompt is the next day is bright and I'm generating 100 new tokens. And in one case, I'm printing it with KV caching. And in another case, I'm printing it without KV caching.

So let's run this right now. When you run this. So these were my previous results, but I will also show it to you real time while running so that you have an understanding.

So now, actually, let me run this once more to show you exactly what is happening here. The with KV caching proceeds so fast that we can't even see it. So now, yeah, see this with KV caching is already completed.

Now it took two seconds and without KV caching, it's still running. So without KV caching, it's still printing right now as I'm speaking, and it has taken 6.7697 seconds. So see the difference here.

When we use cache equal to true, the time taken for GPT-2 to run this inference and to produce 100 new tokens is very low. That's the advantage of KV cache. Remember, we don't need to recompute.

So the inference time becomes quick. And when we use cache equal to false, the time taken increases by almost three times. So let me run this again.

With KV cache, it's done already. It took two seconds to print out the next 100 tokens and without KV cache, it's still running right now and it has taken around seven seconds. We already saw that this is the advantage of the KV cache, right? We don't do recalculations.

Everything is just.

(该文件长度超过30分钟。 在TurboScribe.ai点击升级到无限，以转录长达10小时的文件。)