
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


果传入两个批次，那么其尺寸将是2×3×6。如果同时传入三个批次，那么其尺寸将是3×3×6，以此类推。

没错。但为了简化起见，我将假设每次只传递一个批次。不过要记住，当你编写多头注意力块的代码时，总是从具有三个维度的输入嵌入矩阵开始。所以当你查看代码时，你会看到我们以三个维度开始多头注意力类的前向方法：批处理大小、标记数量和输入嵌入维度。这就是X的点形状，对吧？这就是我的输入嵌入。简单回顾一下，如果你还记得标记在其中的旅程。

所以我们上了一节关于令牌在LLM架构中旅程的讲座，对吧？有一个完整的数据预处理步骤，其中每个令牌基本上都被转换成所谓的输入嵌入，也就是令牌嵌入加上位置嵌入的总和。我们称这些输入嵌入为统一的，实际上每个令牌都有自己统一的输入嵌入。所以我在这里展示的每个令牌的输入嵌入，实际上就是那个统一的嵌入。

所以每个标记都有自己的行，对吧，这是一个六维向量。这是它的输入嵌入，也就是标记嵌入加上位置嵌入的总和。如果你还没看过那节课，我强烈建议你去学习一下，那节课的标题是“标记在LLM架构中的旅程”。

那么现在我们已经完成了第一步，也就是定义了我们的输入，它由三个维度组成。好的，接下来，让我们继续应用多头注意力机制。正如我们在上一讲中看到的，我们需要决定两件事：我们必须决定输出维度，还必须定义我们想要的注意力头的数量。

为什么我们需要确定这两个维度？因为这最终决定了每个注意力头的维度，也就是所谓的头维度。记住，头维度等于输出维度D_out除以头的数量n_heads。因此，在这种情况下，我将把输出维度设为6，头的数量设为2。

所以你可以计算每个注意力头的维度是多少，就是用6除以2。这样每个注意力头现在就会有3个维度了。好的，很好。接下来我要做的是将输入嵌入矩阵与可训练的查询键和值矩阵相乘。首先，我必须初始化这些矩阵，对吧？我必须初始化。记住，现在的x，也就是这个输入，是1×3,6的形状。

所以如果你暂时忘记这第一批数据，它是3乘6的。如果你想用可训练的查询矩阵、可训练的键矩阵和可训练的值矩阵来相乘，首先这些矩阵的维度是D_in乘以D_out。这些可训练矩阵的维度始终是D_in乘以D_out。

记住，现在我们已经确定了 D_in 和 D_out，我的 D_in 等于我的输入维度等于六，我的 D_out 也等于六。因此，我的可训练权重矩阵（查询、键和值）都将是随机初始化的 6x6 矩阵。这就是我接下来要做的。

我正在初始化可训练权重矩阵，用于键、查询和值。这些将是6x6的矩阵。这是WQ，这是WK，这是WV。记住，这些矩阵中的所有值目前都是完全随机的。我们稍后通过反向传播的目标是优化这些值，以便正确预测下一个标记。一旦我有了这些可训练的权重矩阵，下一步我将做的是获取我的输入，然后与这些矩阵相乘。

所以我会用X乘以WK，X乘以WQ，X乘以WV。记住我的X现在是1×3,6的矩阵。如果我用6×6的矩阵去乘它，结果还是一个1×3,6的矩阵。因此，经过这次乘法运算后，我得到了键矩阵，其维度是1×3×6。我还得到了查询矩阵，维度也是1×3×6，以及值矩阵，维度同样是1×3×6。我希望你们特别注意这里的这三个值。

记得我们一开始输入的X维度是批量大小、标记数量和din。现在来看看这些查询、键和值的维度。第一个当然是批量大小，第二个是标记数量。所以这两者保持不变，但第三个现在是dout。因此，我们现在处于输出维度空间，而不是输入维度。因此，键、查询和值都具有批大小、令牌数量和输出维度的维度。

到目前为止，我们还没有应用多头注意力机制，但请耐心听我说，稍后我们会应用它。现在让我们直接看代码。我们有X的形状，即B，标记数量，输入维度。然后我们要在初始化方法中做的是，你看，我们需要初始化可训练的查询、键和值矩阵，对吧？这基本上就是在这里完成的。查询、键和值矩阵通过神经网络的线性层进行初始化，偏置项设为零。

这本质上意味着，当你执行self.dot或传递X（即输入通过可训练的关键权重矩阵）时，它实质上是对可训练关键矩阵与X进行乘法运算。这正是我们想要的，对吧？为了得到键、查询和值，我们只需要对输入嵌入矩阵与可训练的查询键和值矩阵进行乘法运算。这就是这一步所做的。

所以在这里，虽然你看不到直接的乘法运算，但你的X被传递为X，也就是我的输入被作为输入传递给这个W键，它是神经网络的线性层。这本质上就是一个乘法，因为...

可以看到偏置项被设置为0。我们之所以使用nn.linear而不是nn.parameter，是因为nn.linear对权重采用了优化过的初始化方案。这有助于后续的反向传播过程。但请记住，这部分代码实现的是：首先计算键矩阵（即这里的keys矩阵），这部分是输入数据与可训练键矩阵的乘积；接着计算查询矩阵（即输入数据与可训练查询矩阵的乘积）；最后第三行代码计算的是值矩阵（即输入数据与可训练值矩阵的乘积）。

因此，在代码的这一部分，我们已经获得了键、查询和值，请记住它们的维度是批量大小、标记数量和输出维度。现在，多头注意力的真正魔力从这里开始。还记得我们在上一讲中看到的内容吗。我们在上一讲中看到，一旦有了查询、键和值矩阵，根据头的数量，这些矩阵会被分成若干部分。所以在这里，如果你看这个完整的、可训练的查询矩阵，它实际上被分成了两部分，因为有两个头。这正是我们现在要在代码中做的，或者说这正是我们现在要在数学计算中做的。

如果你仔细观察，最后一个维度是D。我们要做的是将这个最后一个维度展开成两部分。目前，这些键、查询和值的维度分别是批次大小B、标记数量和输出维度。我们将把输出维度展开成两部分。这意味着我们不是要处理一个1×3×6的张量，而是要将其展开为“头数, 头维度”的形式。现在仔细看这里：我们已经定义了D_out等于6，并且定义了头的数量D等于2，对吧。

这意味着头部维度等于3。本质上，这意味着每个头部的维度为3。所以，如果你现在有一个6乘6的查询矩阵，即1、2、3、4、5、6，如果我打算重复这个6次，让我们看看我现在的查询矩阵，对吧。实际上，我的查询是3乘6，抱歉，是1乘3乘6，所以我只需要有3行。所以现在这就是我的查询。

但现在我说的不是6列，而是有2个注意力头。因此每个头的维度应该是3。所以我应该把它分成两部分。这应该是我的头1，这应该是我的头2。所以整个3乘6的维度，实际上就会变成3乘3，然后这部分也会是3乘3，或者更准确地说1乘3乘3和1乘3乘3。换句话说，与其用1乘3乘6的张量，我将用一个四维张量1乘3乘2乘3，其中D_out现在被替换为头的数量和头的维度。

有一种非常简单的方法可以直观地理解这一点。如果你能轻松想象出正确数量的标记（tokens）和维度（D_out），那本质上就是3行6列的结构。标记数量（tokens）和维度（D_out）的组合，实际上就是3行6列的矩阵。

但现在3乘2，3，你可以这样想象：这里有3个，所以这是标记1，这是标记1，这是标记2，这是标记3，对吧。之前每个标记基本上有6列与之关联。第一个标记有这6列与之关联，因为输出维度是6。但现在你看到这6列被展开成了3列和3列。

所以，本质上所做的就是将它分成两部分后，这个东西基本上被移到这里。这个东西基本上被移到这里，而这个基本上被移到这里。所以现在我得到的是1、2、3，让我用棕色写下1、2、3。然后是1、2、3，让我也用棕色展示1、2、3，最后我有1、2、3，再次用棕色写下1、2、3。所以所做的就是我的标记1，现在这是我的标记1，这是我的标记2，这是我的标记3，而标记1不再是将这6个值连续排在一起。

第一行现在对应我的头部1，也就是这里的这三个值；第二行现在对应头部2，也就是这里的这三个值。同样地，对于标记2，这是我的头部1，这是我的头部2。同样地，对于标记3，这是我的头部1，这是我的头部2。这就是当我们说将1×3、6或3×6转换为3×2、3时的含义。所以现在这是一个3×2、3的结构。这正是我在这里所写的，1×3、2、3看起来像这样。为什么是3、2、3呢？因为有3个标记，每个标记有2行和3列。

所以令牌1（抱歉，每个令牌都有2行3列）。因此令牌1是2行3列，令牌2是2行3列，令牌3也是2行3列。这里可以清楚看到：令牌1是2行3列，令牌2是2行3列，令牌3是2行3列。每个令牌中的每一行对应什么呢？第一行对应头1，第二行对应头2。现在想象一个令牌——我们正在看查询部分——第一个令牌的输入嵌入被分成两部分：一半进入头1（即第一行），另一半进入头2（即第二行）。令牌2和令牌3也是同样处理。这就是重塑后的查询矩阵。

本质上，重塑操作就是将矩阵拆分成两部分。从视觉上看，将矩阵分成两部分似乎更容易理解，对吧？但在代码中看到这些展开部分时，就会觉得很难想象。不过，我特意在这里展示这个可视化过程，就是为了让你明白：展开最后一个维度究竟意味着什么？展开最后一个维度，就是指你面对这个完整的6维token时，把它分成两部分，并将后半部分挪到前半部分下方。这样一来，就得到了形状为1×3×1×3×2×3的重塑查询矩阵、1×3×2×3的重塑键矩阵，以及1×3×1×3×2×3的重塑值矩阵。

代码里也是这么实现的。我们在这里写的代码是“展开最后一个维度”——原本的维度是（批次数, token数, 输出维度），现在被改为（批次数, token数, 头数, 头维度）。这里头数为2，头维度为3，所以就是1×3×2×3，和代码里写的一模一样。我们要把它展开成（批次数, token数, 头数, 头维度），具体做法是：对原本的键矩阵执行`keys.view(b, token数, 头数, 头维度)`。

同样地，对于值（values），我们将其重塑为形状（b, token数量, 头数, 头维度）。对于查询（queries）也同样处理，重塑为（b, token数量, 头数, 头维度）。这些就是重塑后的键（keys）、值（values）和查询（queries）。

到目前为止，我们已经重塑了键、查询和值，以考虑多个注意力头。接下来我们要做的是，既然已经获得了键、查询和值，我们将按头的数量对矩阵进行分组。所以你看，这里的问题在于我们是按token数量分组的，对吧？我们看到token 1，在token 1里面有head 1和head 2。但现在我们需要做的是按heads来分组。因此，我想要的是1,2,3,3，而不是1,3,2,3，这意味着我想交换这个和这个。本质上，我希望我的矩阵维度是b（批次大小）、heads数量、tokens数量、head维度。

这样做的作用是，你最初看到的查询矩阵是按token 1、token 2、token 3分组的。但现在我们将按注意力头来分组，所以现在这是头1，这是头2。接下来会发生的是，在每个头内部，都有token 1、token 2和token 3，而每个token现在都有等于3的头维度，所以这是3维的。类似地，如果你查看头2，头2的第一行对应token 1、token 2和token 3。你可以看到1,3,2,3之间的区别，之前是按token数量分组的，但现在我们按头1和头2进行了分组。为什么要这样做？因为这更容易进行乘法运算，对吧？如果你看这里，这些头的优势在于查询、键和值被分割成多个头。所以我们应该能清楚地看到不同的副本，对吧？现在，既然我们完成了这种分组，就能清晰地看到这些副本了。

这是第一份查询副本，即Q1，这是Q1，也就是针对第一个头的查询。这是Q2，即第二个头的查询矩阵。这样一来，划分就变得非常简单，而如果你按照标记数量来分组，我的头1在这里，我的头1在这里，我的头1也在这里。

所以我的一部分注意力头1在第一行，一部分在这里，还有一部分在这里。因此需要将它们集中在一个地方进行分组。这就是为什么我们要按照注意力头的数量进行分组，因为稍后记得我们需要在它们之间进行点积运算。比如说，现在这是我的Q1，这是我的Q2，这是我的K1，这是我的K2。

我们需要计算Q1和K1转置的点积，以及Q2和K2转置的点积。记住这正是我们之前在这里所做的。我们计算了Q1和K1转置的点积，也计算了Q2和K2转置的点积。要进行这个点积运算，本质上需要将head1放在一个位置，Q1放在一个位置，Q2放在一个位置，K1放在一个位置，K2放在一个位置。因此，我们按头的数量而不是按标记的数量来分组矩阵是非常重要的。这也是代码下一部分所做的操作。

在代码的下一部分，我们只需对维度1和维度2进行转置操作。因此，keys.transpose(1, 2)意味着：由于Python的索引从0开始，这里索引0代表第一个维度，索引1是第二个维度，索引2是第三个维度。我们将对这两个维度的索引进行转置，使其变为（头数，令牌数）的排列。现在我们将按照注意力头的数量进行分组处理。

因此，所有的键、查询和值矩阵现在都被转置了。这里我们有1,2，因为1是这个索引，2是这个索引。所以现在需要将它们互换。这部分代码的作用就是如此。到目前为止，我们已经得到了Q1、Q2，K1、K2以及V1、V2。如果对照我们上节课看到的步骤，我们已经完成了代码的这一部分，即得到了Q1、Q2，K1、K2和V1、V2。

现在让我们进入下一部分，实际计算注意力分数。要计算注意力分数，我们只需要查看Q1和K1并取其转置，查看Q2和K2并取其转置。就是这样。这就是我们在这里所做的。我们将获取查询，并将其与键的点积转置（2,3）相乘。为什么是2,3？因为现在我们按头的数量进行分组，对吧？所以当我们看一个头时，本质上，首先我们做的是看第一个头。因此，我们必须将其与这个矩阵的转置相乘。

那么，对这个矩阵进行转置意味着什么呢？这意味着现在的行是T1、T2和T3，对吧？而列则是维度。这里的行代表标记的数量，列代表头维度，这意味着我们需要对这两者进行转置。因此，将K1与Q1相乘，再与K1的转置相乘，本质上只是对K1的最后两个维度进行转置。

为什么是最后两个维度？因为我们已经知道这里的每一行对应第一个token、第二个token、第三个token，而每一列对应特征维度——本质上就是对应最后两个维度。所以用K1转置矩阵乘以Q1，本质上就是让查询矩阵在第2、第3维度上与键矩阵进行点积转置运算。现在我们有了完整的查询矩阵对吧？当我们用这个查询矩阵去乘以keys.dot(transpose(2,3))时，实际发生的是：Q1会先与K1转置相乘，然后Q2会与K2转置相乘。这就是我们最终会在这里得到的结果。

因此，如果我们考虑结果矩阵的维度，现在的情况是：我们有一个查询矩阵，其维度为（批大小、注意力头数量、标记数量、头维度），然后将其与转置后的键矩阵（维度为批大小、注意力头数量、头维度、标记数量）相乘。那么矩阵乘法的结果会是什么？它是（标记数量、头维度）乘以（头维度、标记数量），最终得到的矩阵维度将是（批大小、注意力头数量、标记数量、标记数量），也就是一个3x3的矩阵。所以这个乘法的结果会得到一个尺寸为（1、2、3、3）的矩阵。但现在这意味着这是第一个注意力头的注意力分数。

这实际上是头1的注意力分数，这实际上是头2的注意力分数。由于这些都是注意力分数，它们的维度当然必须是标记数量乘以标记数量。这就是我们如何在矩阵乘法中得到两个注意力分数，这也正是代码中所做的操作。抱歉，这正是我们在可视化课程中所演示的内容。

一旦我们有了q1和k1，就进行点积运算。一旦我们有了q2和k2，也进行点积运算。因此，q1乘以k1的转置得到头1的注意力分数，q2乘以k2的转置得到头2的注意力分数。但我要你们特别注意这里的维度，因为维度往往是人们最容易混淆的地方，明白吗？你看这里有头1、头2、头1、头2、头1和头2。所以你有q1、q2、k1、k2、v1、v2。接下来你需要做的是将q1与k1转置相乘，q2与k2转置相乘。这样操作后，最终得到的注意力分数矩阵的维度就是b（批处理大小）、头数（因为你按头数进行了分组，即头1和头2）。为什么是3×3呢？因为既然是注意力分数，就必须是标记数乘以标记数，毕竟注意力分数是在每个标记之间计算得出的。

所以这就是我们现在得到的两个注意力分数矩阵，对应两个注意力头。代码中也是这么实现的。为了得到注意力分数，我们必须将查询和键进行转置（2, 3维度）。这里的2, 3非常重要，因为这是最后两个被转置并相乘的维度。

因此，在完成每个注意力头的点积运算后，最终的注意力分数维度为：批次大小、注意力头数量、令牌数量、令牌数量。这个步骤也被称为"为每个注意力头计算点积"。为什么呢？因为我们首先用q1乘以k1的转置，这是在计算第一个注意力头的点积；接着用q2乘以k2的转置，这本质上是在计算第二个注意力头的点积。至此，我们已经得到了注意力分数矩阵。

现在我们要做的是找到注意力权重。正如我们在昨天的讲座中所看到的，为了得到注意力权重，我们首先需要对其进行缩放，然后应用softmax，接着进行因果注意力处理，如果需要的话还可以进行dropout。这正是数学计算中所做的步骤，让我带大家一步步来看。

好的，现在我们有了注意力分数矩阵。首先我们要做的是对注意力分数进行掩码处理，以实现因果注意力机制。具体来说，这是头1的注意力分数，而这些是头2的注意力分数。我们所做的是将对角线以上的元素替换为负无穷大。这在因果注意力机制的课程中也有提及。此外，头号2中对角线以上的元素同样被替换为负无穷大。

我们将做的是，我们还会除以头维度的平方根。记得在自注意力机制中，我们除以了键维度的平方根。但现在键维度等于头维度。每个关键维度等于头维度，即dout除以头数6再除以2。因此，我们将用3的平方根进行缩放，用3的平方根进行缩放，然后应用softmax。softmax的作用是确保负无穷的元素被设置为0。记住，在因果注意力中，我们不能窥视未来。因此，对于每个标记，我们只能获得与该标记及其之前的标记相对应的注意力分数。

为什么我们要除以头维度的平方根呢？这主要是为了确保查询的方差与键的转置相乘时不会过大。通过除以头维度的平方根，可以保证查询与键转置之间的点积方差基本保持在接近1的水平。这一点对我们进行反向传播等操作非常重要，我们不希望数值之间差异过大。

因此，当我们应用softmax时，我们会得到注意力权重矩阵。请记住，注意力权重矩阵的维度与注意力分数矩阵的维度完全相同。它的维度将是批量大小、头数、标记数和标记数。所以这里也是一样的：批量大小为1，头数。

这是头1，这是头2，然后是3，3是因为我有3个标记。这些行数和列数也等于标记的数量。但现在注意力权重和注意力分数之间的区别在于，如果你看注意力权重中的每一行，每一行的总和实际上是1。所以在这之后我们也可以实现dropout，但为了简单起见，我在这里没有实现它。所以现在你可以做的是，你可以去看代码，你会发现同样的东西在这里已经实现了。

首先，我们创建一个对角线上方为负无穷的掩码，这一步已经在这里完成。我们生成这个对角线上方为负无穷的掩码。接着，我们将其除以头维度的平方根，然后进行softmax运算，如果需要的话，还可以应用dropout。

因此，如果你向上滚动到顶部，我们可以设置dropout率。默认情况下，我认为如果我们不想有任何dropout，可以将dropout率设置为0，但如果你想随机关闭某些注意力权重，可以通过设置dropout率（比如0.5）来实现。这就是到目前为止我们计算注意力权重并应用dropout的方法。然后，在得到注意力权重之后，我们还需要记住最后一步，即必须将head1的注意力权重与value1（v1）相乘，并将head2的注意力权重与v2相乘。

那么现在让我们看看矩阵乘法是如何进行的。好的，这是注意力权重矩阵。这是头1，这是头2，这些是我的值矩阵，对吧。我的价值观是，这是我的v1，这是我的v1，这是我的v2。所以h1和h2。我要做的就是将这两个相乘。

现在来看看这里相乘的具体维度。b代表注意力头的数量，这两个矩阵（或者说这两个四维矩阵）的头数相同，都是按注意力头数分组的。但我们在相乘时真正需要检查的是：这个矩阵是token数量乘以token数量，所以会是3×3的矩阵；而这个矩阵是token数量乘以每个头的维度。

所以这也是3乘3。当你再次进行乘法运算时，乘积现在被带入到token数量、头维度空间。我们这里有三个token，每个头的维度等于3。当你将注意力权重与值相乘时，就会得到上下文向量矩阵。这里的第一行是头1的上下文向量矩阵，第二行是头2的上下文向量矩阵。我们这里有三个token，所以每个token都有对应的上下文向量，每个上下文向量的大小等于头的维度，也就是这里的最后一个维度，即头的维度。

现在如果你滚动到可视化多头注意力部分，这正是我们昨天得到的结果，对吧？我们得到了头1的上下文矩阵，也获得了头2的上下文矩阵。这里同样有11个标记，每个上下文向量的大小等于2，也就是本例中的头维度。

这和这边做的事情是一样的。我们有头1的上下文向量，也有头2的上下文向量。当你深入每个头内部时，其大小等于标记数量，而每个标记的上下文向量尺寸等于头的维度。这部分就是通过将注意力权重与值相乘来实现的。

当我们得到上下文向量后。现在我们要做的是，在得到上下文向量时，记住我们的最终目标并不是直接得到两个不同的上下文矩阵，而是需要将头1的上下文矩阵和头2的上下文矩阵合并。我们必须合并这些上下文矩阵，对吧。我们不需要将它们分开保留。

所以这部分仍然保留着，而为了合并这些，我们需要做的是再次按令牌数量进行分组。这就是为什么我们需要重新调整它的形状。目前的维度是b，逗号，头的数量，对吧。它是按头的数量分组的。所以我们需要再次重塑它。记得我们之前做过这一步，当时我们实际上是将它进行了转换。


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