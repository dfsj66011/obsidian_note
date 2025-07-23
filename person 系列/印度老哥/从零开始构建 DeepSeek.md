
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


因为我们发现，对于每一个标记，我们甚至计算了该标记与之后出现的标记之间的注意力分数，而最终的上下文向量实际上是值向量与注意力分数矩阵的乘积。因此，每个标记的上下文向量本质上包含了关于未来的信息。但实际上，当我们预测下一个标记时，我们根本不会拥有关于未来的任何信息。

因此，理想情况下，当我们获取上下文向量时，应当完全阻止对未来信息的访问——对角线以上的所有元素本质上都应设为零。因为对于某个标记，我们只能访问位于它之前或等于当前标记的内容。这就是因果注意力机制的核心思想。这与自注意力机制形成鲜明对比，后者允许一次性访问整个输入序列。在计算注意力分数时，因果注意力机制确保模型仅考虑序列中当前标记及其之前的标记，这正是我们之前所见的实现方式。

因此，对于每个标记，我们只应考虑在该点出现的标记或在该当前标记之前出现的标记，以获得注意力分数。为了实现这一点，对于每个标记的处理过程，我们会屏蔽掉输入文本中在当前标记之后出现的未来标记。这是最重要的句子，这也是为什么因果注意力有时也被称为掩码保留。所以，如果你看一下双向注意力（即自注意力），每个标记基本上都会关注所有其他标记，对吧。

所以我们得到了当前标记前后注意力分数的信息，但在单向注意力机制中，这被称为自回归注意力、因果注意力或掩码保留——它有很多名称。我们本质上会屏蔽掉对角线上方的所有标记，也就是说，我们会将这些对角线以上的标记设为零。这正是接下来课程要讲的内容：我们如何具体将这些标记设为零。接下来的课程会有点数学化，但到目前为止，我希望你已经理解了我们要做的事情的直觉，对吧？

因此，我们将对角线上方的所有元素屏蔽掉，并记住之前见过的注意力权重矩阵中的这一特性。这里的注意力权重矩阵与注意力分数的主要区别在于，每一行的总和都等于一，这是我们希望保留的理想属性。问题是，如果随机将所有元素设为零，各行之和将不再等于一。因此，我们需要再次进行归一化处理，确保每一行的注意力权重总和恢复为一。

那么现在让我们来看看如何实际操作。任务是这样的：我们已经有了注意力分数，对吧？我们需要将注意力分数中对角线以上的所有元素都设为零。现在来看看怎么做。实际上有两种策略可以实现这一点。我希望你们在这里暂停一下，思考一下当前的问题。想象一下或者回顾一下之前的课程，在之前的课程中我们计算了这些注意力分数，但现在我告诉你们所有这些值都需要设为零。具体来说，所有这些对角线以上的值都需要设为零，所有这些值都需要设为零。

那么，你将如何从根本上计算注意力权重呢？或者换一种思路，你已经有了注意力权重，对吧？假设你把所有这些值都设为零，会发生什么呢？比如说，你现在把这些值都设为零，那么如果你把所有值都设为零，会发生什么？如果是这种情况，你会发现这些行的总和并不等于一。那么，你可以简单地做的是，剩下的部分。如果你把所有值都设为零，这个矩阵会变成类似这样：矩阵会是0.1 0 0 0，然后是0.1 0.5 0 0，接着是0.2。

所以实际上你需要设为零的真实数值是这几个：这个需要设为零，这三个、这两个以及这个——这些数值你都需要设为零。如果你直接把这个设为零，结果会是：这个变成0.1 0 0 0 0，这个变成0.1 0.5 0 0 0，这个变成0.05 0.2 0.2 0 0等等，剩下的两行也是同理。现在你发现问题了：每一行的数值加起来不等于一。

现在第一行的总和是0.1，第二行的总和是0.6，以此类推。那么在这种情况下，你能做些什么让每一行的总和都等于1呢？最简单的做法是：第一行的总和现在是0.1，对吧？所以你把这一行每个元素都除以0.1；第二行的总和现在是0.6，所以你把这一行每个元素都除以0.6；第三行的总和现在是0.2加0.2等于0.4，再加上0.05等于0.45，所以你把这一行每个元素都除以0.45。这样就能确保每一行的总和仍然等于1。

所以这实际上是实现因果注意力分数的第一种策略。第一种策略是这样的：你已经有了经过softmax处理后的注意力分数，也就是在自注意力机制中计算得到的注意力权重。然后你要做的就是在对角线以上添加零值，从而得到掩码注意力分数。但接下来你需要再次对每一行进行归一化，使其总和为一，最终得到掩码注意力权重。这就是我们刚才看到的策略。但你发现这个策略的问题了吗？让我们再看一下这里——这个策略的问题是...

从注意力分数到注意力权重，我们已经使用了softmax进行归一化，这里已经进行了一步归一化。然后，当我们除以行的总和时，又进行了一步归一化。因此，我们在这里不必要地进行了两步归一化。于是问题来了：我们能否采用更智能的归一化方式，从而只需应用一次softmax？事实证明这是可行的，具体做法是直接针对注意力分数进行操作。

让我来告诉你我们是怎么做的。我们的方法是直接针对注意力分数进行干预，而不是在注意力权重上进行干预。具体来说，我们现在是这样操作的：所有这些目前用圆圈标记的值，它们实际上位于对角线上方，最终需要被替换为零。而我们对这些值的处理方式是用负无穷来替换它们，我们会用负无穷来替换这些值。

那么我们来看看注意力分数矩阵现在变成了什么样子。现在，注意力分数矩阵的第一行变为0.6，然后是负无穷、负无穷、负无穷和负无穷。第二行变为0.1、0.1，接着是1.8，然后又是这三个负无穷值，所以我直接把它们复制粘贴到这里。第三行会有两个负无穷值，所以我只展示这三行的情况。第三行的值将是0.2、1.1和1.2。

我们为什么要这样做呢？因为请记住，当你应用softmax时，softmax的作用是对每个元素取指数。具体来说，对于第一行的每个元素，softmax会将其替换为e的x1次方除以总和，第二个元素则替换为e的x2次方除以总和，以此类推。我们正在对第一行进行这样的操作。让我在这里标记一下第一行。假设这是我的第一行，每个元素都将被替换为e的x1次方除以总和、e的x2次方除以总和等等。

现在我要你验证一下e的负无穷次方是多少，你会发现e的负无穷次方实际上等于0对吧？所以第二个元素会被替换为e的负无穷次方除以某个值——反正结果都是0，因为e的负无穷次方就是0。第三个元素会被替换为0，第四个元素会被替换为0，第五个元素也会被替换为0。本质上，所有出现负无穷的地方都会被替换为0，因为任何数的负无穷次方终究都等于0。

第一个元素将被替换为e的0.6次方除以e的0.6次方，因此它将被替换为1，所以总和为1。这里的第一个元素将被替换为e的0.1次方除以e的0.1次方加上e的0.8次方。第二个元素将被替换为e的1.8次方除以e的0.1次方加上e的1.8次方。因此，每一行的总和将为1，我们还将确保对角线上方的所有元素基本上为0。这将确保我们不会进行两个阶段的归一化。在这里，我们进行了两个阶段的归一化，即先进行softmax，然后对每一行进行归一化，但在这里我们只进行一次softmax归一化，就是这样。

这就是在对角线上方引入负无穷的技巧，这是一个非常强大的技巧，它为我们节省了计算量。因此，更高效的方式本质上是你有了注意力分数。更高效的方法是，你有了这些注意力分数，然后应用一种叫做上三角无穷掩码的东西。

这个面具本质上意味着你将上三角部分（即对角线以上的所有元素）替换为负无穷，然后你只需直接应用一次softmax。所以你看这里你只有一个softmax，而之前你需要一个softmax来获取注意力权重，然后还有另一层归一化。这样一来就有两次归一化。

另一方面，这是一种更高效的方式。接下来我们将通过代码来演示，以便你能理解我们在此要实现的目标。还记得在上节课中，我们从输入嵌入矩阵开始，那时你的旅程始于第一步。这里有六个输入，每个输入都有一个三维向量嵌入。在上一节课的最后，我们定义了这个自注意力类。本质上，它接收输入，找到键、查询和值，计算注意力分数，然后得到注意力权重，最后找到上下文向量。在因果注意力中，我们只需要对这一部分进行修改，使得对角线以上的所有元素都被屏蔽掉。那么，让我们来看看因果注意力中具体做了什么。

所以这一节的标题是“用因果注意力机制隐藏未来词汇”。我们从相同的输入开始，我这里只是打印出注意力权重，首先我要向你展示第一种方法。在第一种方法中，我们做的是从之前已经获得的注意力权重开始。

请记住，当我们展示这些注意力权重时，我们已经对注意力分数应用了softmax。在第一种方法中，我们所做的是仅取对角线以上的元素并将它们设为零。因此，这里的掩码实质上是将对角线以上的所有元素设为零，然后我们将这个掩码应用于注意力权重。当我们对这个注意力权重应用掩码时，你会看到我们得到了注意力权重矩阵，但对角线以上的所有元素现在都被设为零。

但这带来了一个问题，即每一行的总和不再等于一，这是一个问题。为了解决这个问题，我们所做的就是简单地将每一行除以其总和，这就是我们所看到的。实际上，这正是我们之前在黑板这里看到的，如果你还记得我们看到的第一个步骤的话。

只需将对角线以上的元素设为零，然后除以行的总和即可。当你运行这段代码时，它会给出掩码注意力权重，这正是因果注意力的主要目的。现在，我将向你展示第二种方法，这种方法实际上更有效。这就是注意力分数矩阵。还记得在第二种方法中，我们不像第一种方法那样从注意力权重开始，而是从注意力分数入手。我们首先计算注意力分数，然后这里是我们所使用的掩码。

所以我们有一个这样的掩码，对角线以上的部分都是1，然后我们拿这个掩码并用它来将对角线以上的所有元素替换为负无穷。一旦我们完成了这一步，接下来要做的就是只需要进行一次softmax运算。softmax会确保这里所有的负无穷实际上都被置为零。

这就是我现在屏幕上展示的内容，softmax函数还能确保每一行的总和等于1。现在如果你比较这些数值，比如1.5517.4483，你会发现它们实际上是完全相同的值。第三行是0.38.3097.3103，0.38.3097.3103。所以本质上两种方法得出的答案完全相同，但第二种方法实际上更高效，因为我们直接从注意力分数开始计算。我们将对角线上方的元素替换为负无穷，然后只需进行一次softmax运算。

我们不必像之前的方法那样先进行softmax再归一化。在实际编写因果注意力机制的代码之前，还有一个最终步骤叫做dropout。dropout实际上是一种深度学习技术，在这种技术中，我们有时会在训练神经网络时观察到有些神经元没有学到任何东西，这些神经元实际上变得懒惰，在学习过程中没有任何贡献。

假设这是我的第一个神经元，这是我的第二个神经元，这是我的第三个神经元，这是我的第四个神经元，这是我的第五个神经元，这是我的第六个神经元。假设这是我的第六个神经元。现在假设这六个神经元在学习过程中，我们观察到其中有实际上两个神经元无所作为，这意味着它们是懒惰神经元，而所有工作都由这四个神经元完成。

解决这个问题的一种方法是在训练过程中随机关闭一些神经元。例如，如果在训练过程中这个神经元和那个神经元被随机关闭，而这两个神经元原本承担了大部分工作，那么现在它们不存在了，剩下的两个神经元别无选择，只能自己学习一些东西。因此，dropout实际上解决了懒惰神经元的问题，即某些神经元根本不工作。

事实上，我们随机丢弃或使某些神经元失活时，其他神经元就必须加快步伐。这就像在做一个小组项目，我们总会遇到这样的情况：只有两个人在干活，其他人什么都不做。但如果这两个人突然生病了、无法参与，其他人就不得不接手工作。Dropout机制也是同样的道理。

所以我们实际的做法是，在因果注意力机制中应用了一种类似的dropout机制。具体来说，在计算完注意力权重后，假设我们得到了注意力权重矩阵——当然这个矩阵会呈现对角线以上元素全为0的特征。我们会随机将某些注意力权重置零，这里的"随机"指的是在每次迭代过程中，被置零的权重位置都会发生变化。

因此唯一固定的是丢弃率。如果丢弃率为50%，就意味着在大型语言模型的每次前向传播过程中，每一行中50%的注意力权重会被随机置为0，而且每次被选中的权重不会相同，因为这些权重是以随机方式选择的，但平均来说会有一半的注意力权重被置为0，这就是通过丢弃机制实现的。这张图实际上展示了这些灰色的神经元，或者更准确地说，这些灰色的注意力权重在这里被屏蔽掉了，或者说它们被随机关闭了。实施丢弃机制的原因在于它能提高泛化性能。

惰性神经元的问题在于，如果我们将这个神经网络应用于新问题时，惰性神经元仍然不会激活，因此泛化能力会不佳。而Dropout机制能确保所有神经元都能有效学习，从而避免对噪声的过拟合，或防止对数据的死记硬背——这些情况通常会导致过拟合和泛化问题。这就是Dropout的原理，接下来我将通过代码展示Dropout的具体实现方式。

假设你有一个这样的矩阵，为了简单起见，可以把它想象成注意力权重矩阵。当你对这样的矩阵应用dropout操作，比如torch.nn.dropout(0.5)，意味着平均而言每一行有50%的元素会被置零——注意这是平均概率，并不保证每行都恰好有一半元素归零。你看这里有些行完全没有被置零的元素，但有些行有五个元素归零，有些行四个，还有些行三个。

所以这是一个随机过程，但关键在于：当使用丢弃率为0.5的dropout时，未被置零的数值实际上会被放大2倍。如果采用0.4的丢弃率（即40%的节点被丢弃），那么剩余数值需要乘以1/0.4的系数进行补偿。这种缩放机制通过将矩阵中保留元素的值乘以1/0.5（即2倍）来抵消激活元素减少的影响。这种调整对于维持注意力权重的整体平衡至关重要，它能确保注意力机制在训练和推理阶段始终保持一致的平均影响力。

Dropout是一种非常简单的机制，你可以把注意力权重想象成灯泡。也就是说，如果这些是注意力权重，就把它们都看作灯泡，而dropout就是在每次迭代过程中灯泡随机熄灭或点亮，你可以指定dropout率。例如，如果dropout率设为50%，就意味着每一行中会随机将一半的灯泡熄灭（置为0）。理解这一点非常重要，它能防止过拟合并提升模型的泛化性能。现在我们已经掌握了这些知识，实际上可以开始编写因果注意力类了。

我们将从相同的输入开始，所以这里有六个标记。你的旅程始于一步，你会看到每个标记都是一个三维的输入嵌入向量。在这种情况下，我要做的是创建两个批次。这是第一个批次，记住每个批次有六个标记，每个标记有三个维度。所以当我把两个批次堆叠在一起时，我们会得到二，这是批次大小；六，这是这里的行数；三，这是列数，这本质上是输入嵌入的维度。

因此，这是我的输入嵌入向量在因果注意力机制中的应用。目标依然相同，我们最终会接收每个输入嵌入向量并将其转换为上下文向量。但现在唯一改变的是，假设我们得到了这些注意力分数，即查询向量与转置后的键向量相乘的结果。

唯一的变化是，我们将对角线上方的所有元素替换为负无穷大，然后进行softmax运算，实际上就是这样。注意力权重的计算方式如下：这将确保对角线上方的所有元素都等于零。接下来，我们应用dropout机制，这意味着在注意力权重的每一行中，权重会被随机置零。正如我们之前讨论的，这实际上提高了泛化能力。而上下文向量的计算方式保持不变。

我们只需获取注意力权重矩阵，并将其与值矩阵相乘，从而得到上下文向量矩阵。因此，如果您思考自注意力与因果注意力之间的区别，唯一的差异发生在注意力分数计算之后。在计算完注意力分数后，我们将对角线以上的元素替换为负无穷大，然后进行softmax运算，接着应用dropout。实际上这里有两处改动：第一处是负无穷大及随后的softmax处理，第二处则是引入dropout机制，它会随机将部分注意力权重置零。

你可能会注意到这里有一些我没有解释的内容，比如这个等于false的bias是什么意思。这基本上意味着我们只是将输入与键、查询和值相乘，而不添加任何偏置项，这就是为什么这个bias等于false。其次，这里的register buffer是什么？为什么要用self.register_buffer来创建这个mask。

所以主要观点是，虽然这不是绝对必要的，但缓冲区会随着我们的模型自动移动到适当的设备上，这在后续训练LLM时会很重要。这意味着我们不需要手动确保这些张量与模型参数位于同一设备上，从而可能避免设备不匹配的错误。目前你本身并不需要这个功能，但如果它存在的话，这只是一个更好的实践。但请记住，这三个维度将贯穿始终，即批大小、令牌数量和输入维度，也就是我们在这里看到的每个输入实际上都有三个维度，这一点非常重要。

好的，现在这个因果注意力类已经实现了，我们来测试一下。我假设输入维度d_in为3，输出维度d_out为2——这和之前示例中看到的输入输出维度一致。你们看这里（滚动屏幕），对，输入维度确实是3，但输出维度是2。这意味着可训练的查询、键、值权重矩阵会将每个输入向量投影到二维输出空间。正如我上节课提到的，在GPT和现代大语言模型中，输入输出维度通常保持一致，但为了演示方便，我们这里暂时采用不同维度。

那么我们接下来要做的就是定义批次，这里记住批次是2,6,3。如果你看的话，嗯，对的，批次就是2,6,3，这是我的输入。然后我只需要定义因果注意力。因果注意力类实际上需要四个输入，嗯，需要四个输入。这里我有我的d_in，我有我的d_out，我有我的上下文长度，最后一个是丢弃率。记住上下文长度是batch.shape[1]，也就是第一个索引。如果你看batch.shape，它是2,6,3，所以batch.shape索引为1的值就是6，因为我在序列中看的是6个元素，所以在这种情况下上下文长度是6。

所以上下文长度等于六，因果注意力类还需要什么？它需要d_in等于三，d_out等于二，上下文长度等于六，还需要一个dropout率。这里我刚刚提到dropout率等于零，所以你可以运行这个，最终看到这个，嗯是的，所以这里的dropout率等于零。在dropout之前你有这些注意力权重，嗯，实际上这部分我想我应该再运行一次。这是我的因果注意力类d_in，d_in，嗯是的，现在我打印了上下文向量，这些就是得到的上下文向量，对吧？所以这个大小是六乘二，这正是我们在这里看到的大小。

因此，上下文向量的尺寸将是6行2列（6x2）。但请记住，我们有两个批次的数据。在第一个批次中，我们有6个标记需要处理，这将生成一个6x2的上下文向量矩阵。当处理第二个批次时，同样会生成另一个6x2的上下文向量矩阵。这就是第二个批次的情况。现在请注意，我们这里传递的是两个批次的数据——如果你观察输入数据，会发现我们是将两个批次堆叠在一起的。

所以这里有两批数据，因此输出将是“2,6,2”。这里是“2,6”，第一批输出的是第一批的上下文向量矩阵，而这是第二批的上下文向量矩阵。因此输出尺寸是“2,6,2”。不要困惑为什么输出尺寸或输出维度不是“6×2”，原因是“2,6,2”表示我们有两批数据，每批的尺寸是“6×2”，对吧？这就是我的上下文向量矩阵，看起来是正确的。呃，这里我写了另一个函数，用于打印出应用 dropout 前后的注意力权重。这里我用 dropout 率为 0.5 运行了因果注意力类，并展示了应用 dropout 前后的打印结果。

所以你会看到，在dropout之前，比如说这里有四个权重是活跃的，对吧？但在dropout之后，只有其中一个保持活跃，但它的值被乘以了2，因为现在的dropout率是0.5。同样地，如果你看第三行，dropout前有三个活跃权重，但dropout后只有两个保持活跃。再看第五行，dropout前有五个活跃权重，但dropout后一个都不剩了——记住这是个随机过程。

平均而言，50%的权重会被关闭，但有时也可能所有权重都被关闭。你可以运行这部分代码亲自验证dropout的实际效果。至此，关于因果注意力的课程就结束了。请记住，因果注意力其实非常简单——只要理解了自注意力机制，其核心逻辑在于：对于给定token，我们无法获取未来信息，只能访问该token及其之前的上下文。

所以我们实际要做的是，需要获取注意力权重并将对角线以上的所有元素设为零。具体有两种实现方式：第一种是直接从先前通过注意力分数和softmax计算得到的注意力权重入手，将主对角线上方的所有元素置零，然后再次对行进行归一化——但这里你需要先做一次softmax，接着还要再做一次行归一化，相当于进行了两次归一化操作。而更高效的做法是：直接从应用softmax之前的注意力分数矩阵出发进行处理。

然后将对角线上方的所有元素设为负无穷大，这样在计算softmax时，系统会自动将这些负无穷大的元素归零，确保每一行的总和为一。这就是注意力权重的生成方式，也是如何使对角线上方的元素归零的原理。但在因果注意力机制中，我们并不止步于此——通常还会加入dropout操作，即随机屏蔽部分权重并将其设为零，以提升模型的泛化性能。每次前向传播时，被屏蔽的权重都会随机变化。

因此，这确保了没有“懒惰”的权重，每个权重在训练过程中实际上都在学习一些东西。然后我们实现了一个因果注意力类，如果你看的话，它实际上看起来像这样，大约15到20行代码。我希望你已经理解了这段代码的每一个方面。记住输入形状的这三个维度：批大小、标记数量和输入维度，正如你在每节课中一定已经注意到的那样。

我非常关注维度问题，因为归根结底，我认为理解注意力机制实际上就是理解矩阵运算。但人们往往对矩阵感到不自在，因为他们无法直观想象高维空间——这就是为什么我如此强调维度的重要性。下节课我们将继续深入：目前我们已经学完了自注意力机制和因果注意力机制，接下来就要开始学习多头注意力机制了。

那么我们将完全准备好理解e值缓存，最终我们将完全准备好理解多头潜在注意力机制。正如我一直强调的，在我讲解时一定要记笔记，这样你才能真正理解课程内容。如果你只是听课，可能会觉得自己理解了，但概念并不会得到强化。课程内容会越来越深入。

随着我们进入后续模块的学习，我衷心希望大家能坚持完成这门课程，听完所有讲座内容。请保持学习动力，做好课堂笔记，随时提出疑问，我们会为大家逐一解答。非常感谢各位的参与，期待下节课与大家再见！大家好，我是Raj Dandekar博士，2022年获得麻省理工学院机器学习专业博士学位，同时也是"从零构建Deep Seek"系列课程的创始人。

在我们开始之前，我想向大家介绍一下本系列的赞助商兼合作伙伴——InVideo AI。大家都知道我们有多重视基础内容建设，从零开始构建AI模型。InVideo AI遵循与我们非常相似的原则和理念。让我来展示一下。这是InVideo AI的网站，他们凭借一支精干的工程团队，打造了一款令人惊叹的产品，仅通过文本提示就能生成高质量的AI视频。

正如你所见，我输入了一段文字提示："创作一则超写实的高端奢华腕表广告视频，呈现电影级质感"。点击生成视频后，很快我就获得了这段令人惊叹的逼真视频。最让我震撼的是它对细节的极致把控——看这个材质纹理和光影效果简直不可思议，而这一切仅通过文字指令就实现了，这就是InVideo产品的魔力。

你刚才看到的惊艳视频背后，是InVideo AI基于第一性原理重构的视频创作流程。他们从底层模型开始实验与优化，彻底革新了视频生成与剪辑的技术范式。

他们在印度拥有最大的H100和H200集群之一，并且正在试验B200。在视频AI领域，他们是印度发展最快的AI初创公司，面向全球市场。这就是为什么我对他们如此有共鸣。好消息是，他们目前有多个职位空缺，你可以加入他们出色的团队。我会在下面的描述中发布更多详细信息。大家好，欢迎来到下一堂课。

在《从零开始构建深度学习》系列中，为了快速回顾，以下是目前为止我们已涵盖的内容：我们正在探索多头潜在注意力机制的旅程中。迄今我们已经理解了自注意力机制——我们从基础概念到代码实现全面讲解了这个模块。在上一讲中，我们掌握了因果注意力机制，而今天我们将要学习多头注意力机制的基础知识，这部分内容将在下一讲展开。


we are going to code multi-head attention but today the main thing which i want to convey to you is what what is the real need for having this multi-head attention and why can't we just stick with causal attention let's say the more i read in literature and the more resources i have seen very few people have explained what is the real need for having multiple heads in attention so without going into the code or without directly jumping into the code first it's very important for us to develop a very strong intuition of why we have multiple heads so today's lecture is titled going from self-attention to multi-head attention and i'm going to try to explain this to you in a very intuitive manner directly from the first principles so let's get started with today's lecture as we have seen before the main goal of the self-attention mechanism is essentially to take input embeddings and to transform them into something which is called as context embeddings remember that input embeddings on their own do not contain any information about neighbors so for example if i look at a single token in a sentence input embedding captures the semantic meaning of that token and its position but it does not capture how that token relates to the other tokens in the sequence and that's the additional mechanism which self-attention brings into the picture the context vector which is the main outcome of the self-attention mechanism contains information of how a given token attends to all its neighboring tokens and to go from the input embedding matrix to the context embedding matrix we have essentially four parts the first part is transformation to query key value matrices second part is calculation of the attention scores third part is calculation of the attention weights and fourth part is the context vector matrix calculation so let's quickly see these four parts in action right now so let me take you to that let me take you to that part of the notebook where we did those calculations here it is so here's the input embedding matrix which we have there are one two three four five tokens and each token is essentially an eight dimensional vector in the first step what we do is that we multiply this input embedding matrix with trainable query key and value weight matrices and then we get this query vector matrix the key vector matrix and the value vector matrix after this point we forget about the input embedding matrix completely and only operate on these queries keys and the values that's step number one step number two is we take a product of queries multiplied by the keys transpose and that essentially gives us the attention scores so every row of this corresponds to the attention score of that particular token so for example the second row of this matrix corresponds to the attention scores for next and how next actually relates to the next day is bright so these five attention scores are mentioned in this or are calculated in this second row similarly all other rows have attention scores of that particular token the main drawback of attention scores is that the rows don't sum up to one so we cannot make intuitive claims that when next is the query give 10 importance to the give 50 importance to next etc so to do that we have to normalize the attention scores and we have two steps in the normalization first what we do is that we take the attention scores and we divide them by the square root of the keys dimension this is done so that the variance of the queries multiplied by the keys transpose that stays equal to one and the variance does not become too large and then we apply the softmax function the softmax essentially ensures that when you look at a particular row all the entries of that row will sum up to one so this is normal self-attention where we look for a given token we look forward as well as backward in the causal attention mechanism what we do is that all the elements essentially about this diagonal all of these elements are essentially put to zero because for a given token we should not have access to all the tokens which come after that particular token so all of these elements which are marked in the green color right now they are effectively masked to zero that's what happens in the causal attention mechanism and then we can apply dropout to the attention weight so that we randomly mask out some elements to be zero this is to improve the generalization performance and to prevent overfitting the last step which is done here is that we take the attention weights matrix and we multiply with the value matrix to give us the context vector matrix and here you can see that every row of the context vector corresponds to my token so the first row is the context vector for the the second row is the context vector for next etc and if you look at the visualization of the context vector you will see that the context vector is much richer because it takes into account how much attention is paid to all the nearby or the neighboring tokens of next for example whereas the input embedding does not have that richness in information about the neighboring tokens so this was in a nutshell the self-attention mechanism and the causal attention mechanism also remember that when we saw the causal attention mechanism what we essentially did was we put the all the elements of the attention weights above this diagonal to be equal to zero and then after that we can also add dropout so we can further randomly turn off certain attention weights to zero that was causal attention okay so with that let's get started with today's lecture in which we will motivate why do we need to go from self-attention to multi-head attention so first remember that self-attention is awesome and self-attention has a lot of advantages because we finally start including or encoding information about the context which is present in the given sentence without self-attention mechanism language models would not be as good as understanding the meaning of sentences as they are today but for all the advantages of self-attention self-attention mechanisms come with one very major problem and multi-head attention solves that problem so let's see what this problem with the self-attention mechanism is and for that i want to start out with a simple illustration so what i want to start out with is that i want to start out with a sentence right and the sentence is the artist the artist painted the portrait of a woman with a brush let me read this sentence aloud once more the artist painted the portrait of a woman with a brush now try to think about what do you think this sentence means you can pause the video for a while and think about the interpretation of this sentence okay so if you have thought for some time you will have realized that the reason i have chosen this sentence is that it has two interpretations the first interpretation is that the artist so let's say if this is the artist the artist has painted the portrait of this woman

(该文件长度超过30分钟。 在TurboScribe.ai点击升级到无限，以转录长达10小时的文件。)

(转录由TurboScribe.ai完成。升级到无限以移除此消息。)

Using a brush. So, The Artist painted the portrait of a woman with a brush, which means that The Artist had a brush in their hand, and with that The Artist painted the portrait of a woman. That's the first interpretation, The second interpretation is that the artist has painted the portrait of a woman with a brush.

So, the painting is of a woman who has a brush in their hand and The Artist has painted this. So you see the difference in the first case, the painting is that of a woman who does not have a brush in her hand. The artist is painting the portrait with a brush.

But in the second case, it's the painting or the portrait of a woman with a brush in their hand. So this is the painting. So this is a woman with a brush in her hand and the artist has painted this.

So the first is a painting, painting a woman with a brush and the second is painting of a woman with a brush, right? So these are the two interpretations of this sentence. The problem of the self-attention mechanism comes into the picture over here. So what I want to now do is that I want you to focus on two attention scores matrices and which I'm going to put them together side by side over here now.

So take a look at the attention score matrix number one and so take a look at attention score number matrix number one and matrix number two. And what I'm also going to do is that I'm going to bring these corresponding images below it. Okay, so this is the first image.

And then I'm also going to bring the second image now. Here I have brought the second image now and let me just rub this arrow. Okay, now I want you to focus on these two attention score matrices.

Okay, the first attention score matrix, which is which I marked here as number one, it corresponds to this this image, right? And why is that? Because focus on the woman, if the woman is my query, the circles which I've marked in the red here correspond to the highest or the strongest attention scores. So you can focus on those red circles for now. So here we have a red circle between woman and a portrait.

Since it's a portrait of a woman, the attention score between woman and portrait is quite high. That is understandable. Whereas if you see over here in the second in the second attention scores matrix, the attention score between woman and the portrait is still high.

That is understandable because here also we have a portrait of a woman. But notice this red color over here, the attention score between woman and the brush is also very high. And why is the attention score between woman and the brush also high? Because in the second interpretation, the woman is holding a brush in her hand.

That is the difference between this row in the first attention score matrix and this row in the second attention score matrix. In the second attention scores, the attention score between woman and the brush is quite high. Whereas that's not the case in the first attention score because in the first attention score matrix, the woman is not holding a brush in her hand.

The second distinction which we see is when we look at, when we look at essentially artist. So when we look at artist in the first matrix, the artist has a high attention score with painted and the artist has a high attention score with brush. Why? Because the artist is holding a brush in her hand in the first interpretation.

Whereas if you look at the second interpretation, if you look at the second interpretation, the artist is actually not holding a brush in their hand and that's why the attention score between artist and the brush is actually not very high. That's the second difference between these two attention score matrices. And the third difference between these two is when you look at brush.

So when you look at brush, you'll see that the attention, if brush is the query, it has high attention scores with artist because the artist is holding the brush and it has high attention scores with, what is this, yeah, this should not be there. So the brush just has a high attention score with the artist because the artist seems to be holding the brush, right? Whereas here, if you see, brush has a high attention score with portrait because the portrait is that of a woman with a brush and brush also has a high attention score with woman. Why? Because the woman has a brush in her hand.

So brush has a high attention score with portrait and woman in this case, whereas in this case brush has a high attention score with artist. So, I hope I have convinced you now that in these two interpretations of the sentence, the attention scores matrix are completely different from each other, right? There can be two attention score matrices based on the interpretation which you have for the sentence. Now the main problem with the self-attention mechanism is that the self-attention mechanism can only capture a single perspective in a given input sequence.

It cannot capture multiple perspectives. So if you are looking at this sentence, the self-attention mechanism will either have this attention scores matrix or it will have this attention scores matrix and consequently the context vector will either be based on the first attention score matrix or the second attention score matrix. So it cannot take into account both of these perspectives at the same time and that's one major limitation of the self-attention mechanism that if you are given a piece of paragraph or a piece of sentences, self-attention can only capture one perspective.

So if you think about it more deeply, why would you want to capture multiple perspectives in a sentence? The reason you would want to capture multiple perspectives in a sentence or paragraph is that sometimes it is difficult to understand what the paragraph represents in just one form or meaning. One paragraph might have different angles, right? One paragraph might be expressed in multiple different ways and when we design our language model, we want our language model to be knowledgeable of multiple different perspectives present within a paragraph. An example here is that let's say we have a huge paragraph, right? And within that paragraph, we have a sentence which reads like this.

The government should regulate free speech. Now if you look at this sentence, the government should regulate free speech. The first perspective should be the government should impose restrictions on free speech.

So free speech should be curtailed, that's the first perspective. The second perspective should be the government should protect and preserve free speech. It's not really clear from this sentence which perspective is this paragraph referring to.

So it just says regulate free speech. So regulating free speech can either mean that free speech should be curtailed or free speech should be protected and preserved. What does this line really mean? So if we use the self-attention mechanism, again it will assign only one perspective to this.

But we actually want the attention mechanism to capture both of the perspectives which this sentence can include. And the advantage of capturing both perspectives is that our model just becomes more richer in its understanding. So when a user asks to summarize this document, we can summarize both the perspectives, whereas if we just use a self-attention mechanism and a user asks to summarize this document, we will only summarize one perspective and that is not good.

So we want to overcome this limitation of self-attention mechanism. So essentially what we want to do is that we want to have provision in our model so that we can have or we can capture different perspectives in my paragraph. So if one self-attention mechanism can only capture one perspective, can we somehow extend the self-attention mechanism so that it can capture multiple perspectives? So if one self-attention mechanism can essentially just capture one perspective, within the same architecture, what if I have multiple self-attention mechanisms? What if I have multiple self-attention mechanisms? So if one self-attention mechanism can capture one perspective, multiple self-attention mechanism can capture multiple perspectives.

So if let us say one self-attention mechanism gives me one context vector matrix, another self-attention mechanism gives me another context vector matrix, a third self-attention mechanism gives me a third context vector matrix, I will have these multiple context vector matrices, each of which capture different perspectives and then I will just merge them together so that the final context matrix which I have will be much richer in that it will capture multiple perspectives of the given sentence or the given paragraph. So the main idea is this right, you have instead of just having one self-attention mechanism, what you can do is that let us say you have the input, let us say you have the input embedding matrix right, you pass it through self-attention mechanism number 1, you pass it through self-attention mechanism number 2, let us say we look at 2 right now, so this will give me a context vector matrix 1, context vector matrix number 1 and this will give me a context vector matrix number 2 and then what I will do is that I will just merge these two context vector matrices together so that I will have one resultant context vector matrix which now consists of multiple perspectives, this is perspective number 1 and this is essentially perspective number 2 and when I say perspective, it does not necessarily have to do with meaning always, it can be something else like one context vector matrix essentially looks at or pays more attention to verbs in a given sentence, the second context vector matrix maybe pays more attention to the hidden meaning of a given sentence etc. So multiple representations might come out from different context vector matrix, so here we are just trying to expand our range of what we capture from a given sentence and this thing of converting the self-attention mechanism to multiple self-attention mechanism is essentially called as multi-head attention, multi-head attention, the reason this term head comes into the picture is that it is essentially like having multiple self-attention heads, so if you think of one self-attention as one person with one head, it is like one head is giving one attention, but now you have multiple heads, each of these heads is capturing a different perspective.

So think of these as multiple people, let us say this is the first person with the first perspective, second person with the second perspective, so this is just a nomenclature but the head comes from the fact that we are aggregating multiple self-attention blocks together and that is where the name multi-head attention actually has its origins. So I hope until now I have motivated the concept of why do we need multi-head attention and what are the limitations of the self-attention mechanism and now we are going to start looking at how does multi-head attention actually operate in the context of the matrices which we have seen before, the queries, keys, values etc., how exactly can we change that whole procedure so that we have multiple heads. So the main question here right now is that what if we have two self-attention mechanisms instead of one, so let us take this same sentence, the artist painted the portrait of a woman with a brush, now I am going to show you the step-by-step procedure of how we can have two self-attention mechanisms within this sentence and remember that the main purpose of this exercise is that if we have two self-attention mechanisms, we should have two attention scores matrices and we should have two context vector matrices because each has to capture a different perspective. 

So I will need to show you how this is exactly done in the query key value representation which we have. So what if we have two self-attention mechanisms instead of one, now by the way this two self-attention mechanism is also called having two heads and that is the origin of the term multi-head. So now what we are going to do is that we are going to take this sentence and we are going to see a step-by-step procedure of how we can implement a two-head attention on this input sequence and when you understand this step-by-step procedure, you will have a complete visual roadmap of how the multi-head attention mechanism works.

If you have understood self-attention multi-head attention is actually quite straightforward but I think in literature and in other videos it just explained in a complicated and reverse manner instead it's much easier to explain multi-head attention if you motivate it like this and then if you show the step-by-step visual matrix or visual matrices calculation. So let's see this step-by-step procedure right now. Okay, the first thing which we do is as always we start with an input embedding matrix and the input embedding matrix looks like this. 

We have these tokens the artist painted the portrait of a woman with a brush right. We have these 11 tokens and then every token is essentially an input embedding of 8 dimensions which we have considered over here this is also called as the input embedding dimension or DIN. So the dimensions of this entire matrix are we have 11 rows and we have 8 columns so the dimensions are 11 by 8 that's my input embedding matrix.

Remember the goal of this two-head attention now is to take this input embedding matrix and to convert it into two context vector matrices not just one we have to convert it into two so that each context vector matrix captures a different perspective. So this is the input embedding matrix which we have started with and then the first thing which I want to do is show you what we would have done if we just had a single attention head. So if we have a single attention head what would we have done we would have multiplied this input embedding matrix with the trainable query matrix which is an 8 by 4 dimension matrix a trainable key matrix which is an 8 by 4 dimension matrix and a trainable value matrix which is 8 by 4 and this multiplication would have resulted in a query vector matrix that's 11 by 4 a key vector matrix that's 11 by 4 and a value vector matrix that's 11 by 4. Now to extend this into a multi-head attention or a two-head attention in this case what we have to do is that we have to first decide on the output dimension which we want and here I am deciding the output dimension is equal to 4 that is something which is fixed at the start the input dimension is fixed the output dimension is fixed then what we do is that then we decide the number of heads which we want and here we are having two attention heads right so this output dimension is then split among these two attention heads so each attention head will have dimension equal to 2 and the way this visually looks right now is something like this so now my trainable query key and the value weight matrices earlier they looked like this right but now I am just going to divide them into two parts so my trainable query matrix is now for the first attention head it is this so see the size of this has been now it's 8 rows and 2 columns instead of 8 rows and 4 columns and for my second head the trainable query matrix is this which is again 8 rows and 2 columns so in terms of a nomenclature now instead of wq I have wq1 which corresponds to the first head and I have wq2 which corresponds to the second head so if dimensions understanding dimensions is a bit difficult just remember that to split this into two attention heads I have just divided the query into two parts the keys weight matrix into 2 and the value weight matrix into 2 that's it so this dimension over here which the trainable query weight matrix has the trainable query second for the second head this is called as the head dimension and the head dimension is just the output dimension divided by the number of heads so the output dimension in our case is equal to 4 and the number of heads is 2 so the head dimension is just 4 by 2 which is equal to 2 so what is the head dimension it's essentially the number of columns in each attention head and that's equal to 2 in our case and this same split happens for the trainable key matrix and the trainable value matrix as well so similar to what we did for the trainable query matrix we now have wk1 wv1 wk2 and wv2 so these are the trainable key matrices for the both heads and these are the trainable value matrices for the both heads so see what we are doing in this step is that we are creating multiple copies of the trainable query weight matrix the trainable key weight matrix and the trainable value weight matrix this is the main idea in the multi-head attention and if you think about it it's quite simple right in a single head we just had one matrix and remember that now that the d out is fixed we cannot change this so if we want two attention heads we just split the d out into two parts so this is my these are my trainable query weight matrices trainable key weight matrices and trainable value weight matrices that's step number three we essentially split or create multiple copies of wq wk and wv now now that we have multiple copies of wk wq wk and wv it will naturally create multiple copies of the query vectors the key vectors and the value vectors right because let's look at the query vectors first i will first take my input embedding matrix x and i will multiply it with wq1 so that's 11 by 8 multiplied by 8 by 2 and that will give me my first query vector matrix q1 which is 11 by 2 then i will take my input embedding matrix x and multiply it with wq2 that will give me my second query vector matrix that's q2 which is again 11 by 2 similarly i take my input embedding matrix multiply it with wk1 and wk2 and that gives me the two key vector matrices and i take my input embedding matrix and i multiply it with wv1 and multiply it with wv2 and then i get my two value vector matrices v1 and v2 now remember here what we have done simply is that instead of having one one query vector matrix one key vector matrix and one value vector matrix for single head since we have multiple heads now we have two query vector matrices q1 and q2 we have two key vector matrices k1 and k2 and we have two value vector matrices v1 and v2 and what are the dimensions of this these matrices the number of rows essentially remain the same so if you see for all of them the number of rows remains 11 why do the number of rows remain 11 because the number of tokens which i have the artist painted the portrait of a woman with a brush those are 11 tokens but the key thing to note here is that the number of columns which we have the number of columns now becomes equal to 2 because that's the head dimension remember the head dimension is just the d out divided by the number of heads which is equal to 4 divided by 2 which is equal to 2 so the number of columns in all of these matrices are equal to 2 again if you are getting confused just look at the head number 1 all of these matrices are the query key and the value matrices for head number 1 and all of these matrices are the query the key and the value matrices for head number 2 remember this head number 1 we have these matrices and head number 2 we have these matrices it's just that we have now created multiple copies so what happens is that we have the same vectors for a single head but now we have two copies and now that we have two copies we still have only four dimensions right so each copy has to have only two dimensions remember the d out is fixed at the start so in step number four we create multiple copies of the query the key and the value vector matrices which i have denoted over here right now q1 q2 k1 k2 v1 v2 now think about this right what is usually done in the next step usually we take the dot product of queries and the keys to get the attention score matrix but here we have two query matrices we have two key matrices so what will happen naturally we will have two attention score matrices right so that's what happened next we compute the attention scores for each attention head so this is q1 q2 k1 and k2 so for computing the head one attention scores what we simply do is we multiply q1 with the we multiply q1 with k1 transpose we multiply q1 with k1 transpose so that's 11 by 2 multiplied by 2 comma 11 and that gives us the attention score of the first head that's 11 by 11 then what we do is that to find the attention scores matrix of the second head we multiply q2 with k2 transpose so that's 11 by 2 multiplied by 2 by 11 that's 11 by 11 so now take a look here what is exactly happening when we looked at single head if we look at a single head attention we'll have an attention score matrix of 11 by 11 right if we just look at one head because there are 11 tokens here the cool thing which has happened or the amazing thing which has happened with multiple heads is that although the output dimension is getting split into two parts so the head dimension is equal to 2 the attention scores dimension remains the same for both the heads it's 11 by 11 for the first head and it's 11 by 11 for the second head and it would have been 11 by 11 if we just did a single head so essentially now what we have done is that we have two copies of the attention scores we have one 11 by 11 attention score and one 11 by 11 attention score why is it 11 by 11 because remember there are 11 tokens right the artist painted the woman painted the portrait of a woman with a brush etc and if you think about where we started with this is exactly what we wanted right instead of just getting one attention scores matrix we wanted to extend the self-attention mechanism so that we can get multiple attention scores matrices and that is exactly what has happened here since we had two copies of the queries since we had two copies of the queries and we had two copies of the keys we can essentially multiply these two copies and get two attention scores matrices so each attention score matrix essentially can capture a different perspective and that is the main advantage of multi-head attention this step here that although we have multiple heads and although the dimension of each head is now split into two so the head dimension is now equal to 2 which was 4 before so in single head attention it was 11 by 4 multiplied by 4 by 11 and that gave us the 11 by 11 attention score matrix but now it's 11 by 2 multiplied by 2 by 11 so although this dimension is reduced by half so although this dimension is essentially reduced by half the final attention score matrix still is 11 by 11 so this dimension is same in both of these cases that's the beauty of multi-head attention although the head each head has a reduced dimension when we take the dot product of the queries and keys transpose for both the heads it's we get two attention score matrices of dimensions 11 by 11 and each of these can now capture a different perspective essentially we have two copies of the attention score matrices now then what happens in the next step is the same since we have two copies of the attention scores now what we'll do is that we'll scale we'll scale by square root of the keys dimension we'll apply softmax and then we'll apply causal attention which means that we'll just make sure that all the elements above the diagonal are set to zero remember we cannot peek into the future and if needed we can also apply dropout so in this schematic i have assumed the dropout rate to be zero but after you get the attention weight matrix you can even have a dropout rate and randomly turn off different elements in the attention weight matrix so this is the head 1 attention weight matrix that is 11 by 11 and this is the head 2 attention weight matrix that is also 11 by 11 matrix now and what is the difference between attention weights and attention scores attention weights every row will just be normalized so if you look at every row it will be summed up to 1 and also remember that we are implementing causality here so we make sure that for both of these attention weight matrices the elements above the diagonal will essentially be equal to zero so just keep that in mind and then what we do in the last step is that now we have two we have an attention head matrix for both these heads right and remember earlier we had calculated the value matrices v1 and v2 v1 was 11 by 2 v2 was 11 by 2 so v1 is the value matrix for head 1 v2 is the value matrix for head 2 so what we will do in this last step is essentially we take the attention weight matrix of the first head we multiply it with the value matrix of the first head so that gives us the context vector matrix for head 1 which is 11 by 2 and for the second head we similarly take the attention weight matrices of the second head and we multiply it with the value vector for the second head so that is 11 by 11 multiplied by 11 by 2 and that gives us the head 2 context matrix so head 1 context matrix is 11 by 2 and head 2 context matrix is also 11 by 2 and now remember what we do after this point is that we have the context vector matrices from both the heads and remember what we had discussed at the start once we have the context once we have the context vector matrix for the head number one and once we have the context vector matrix for head number two we just merge these context vector matrices and that's exactly what we do in the last step in the last step what we do is that we have the first head one context matrix which is which you you can say as giving us the first perspective that is perspective one and we have the head sorry this should be head two so we have the head two context matrix and that essentially gives us the perspective two and when you merge these context vector matrices you will have the final context vector matrix which is of the size of 11 by 4 so to the left side of this is my first head to the right side of this is my second head so ultimately when i merge the results from both the heads together i'll have the context vector of vector of size 11 by 4 and remember now if you had just done a single head attention if you are just a single head attention without splitting into two heads the output dimension is four right so you would have also got the same context vector matrix size 11 by 4 but it would not have consisted of two perspectives in a single head if you had just used a single head there also you would have gotten the same context vector matrix of 11 by 4 but there the whole thing would have been just one perspective but now the advantage here is that

(该文件长度超过30分钟。 在TurboScribe.ai点击升级到无限，以转录长达10小时的文件。)

(转录由TurboScribe.ai完成。升级到无限以移除此消息。)

The size remains the same, but it consists of two perspectives. The first perspective given by my first head, which I am calling P1 and the second perspective given by my second head, which is called as P2. So we have just extracted more information from my text.

Of course, the disadvantage of this is that for extracting each perspective, we have only two dimensions to play with now, that is the drawback, whereas here we had essentially four dimensions for each perspective, right? But now we have a reduced number of dimensions to play with for each perspective. That is the drawback for multi-head attention. The main drawback is that the dimension size for each head reduces, right? As you see over here, the dimension for each head is effectively reduced because we have to split the whole query weight matrix, key weight matrix and value weight matrix into two.

So the dimension size for each head is reduced. So the amount of information we can capture is a bit reduced, but the number of perspectives we can capture is increased. So each head captures more perspective.

So the way I think about it is like divide and conquer. Instead of conquering the whole sentence at once, you divide into different parts and then each part conquers some different perspective. That's the simplest way I like to think about multi-head attention.

So this whole step-by-step procedure which we saw, let's recap it quickly. We start with the input embedding matrix, the artist painted the portrait of a woman with a brush and I have deliberately started with the sentence here which can be looked at from different perspectives, correct? So what we do here is that we start with the input embedding matrix and when we multiply it with the trainable query key and the value matrix, we split these trainable weight matrices into two parts. So we fix the output dimension equal to 4 and we decide the number of heads.

So since we have two heads here, each head will essentially get two dimensions. That's called as the head dimension, which is the d out 4 divided by the number of heads which is equal to 2. So wq1 is 8 by 2, wq2 is 8 by 2 etc. So these are the query key and the value trainable weight matrices for head number 1 and these are the trainable query key and the value weight matrices for head number 2. Alright, so once we have these multiple copies of w, q, w, k and w, v, naturally it leads to multiple copies of query key and value.

So head 1 has one copy of q, k, v which is q1, k1 and v1 and head 2 has another copy of q, k, v that's q2, k2 and v2. Then what we do is that for q1 and k1 we have the first attention scores matrix, for q2 and k2 we have the second attention scores matrix. The first attention scores matrix is from head 1, second attention scores matrix is from head 2. Why do we have two attention score matrices? Well, each head might be capturing a different perspective such as maybe the first head might be capturing this perspective, maybe the second head might be capturing this perspective etc.

So each head might be capturing different perspective and that's why we have two attention scores matrix here. In my view, this part is the most important step because here we see that each attention scores matrix captures a different perspective and that's the whole advantage of the multi-head attention mechanism. And then after that we follow similar steps which we have seen for self-attention.

We then take the attention score matrix, scale it by square root of keys dimension, apply softmax, apply causal attention which means we mask out all elements above the diagonal in the attention weights to be 0 and then if needed we can apply dropout to improve the generalization performance or to prevent overfitting. So until this point we have the attention weights which have been calculated and then what we do is we multiply the attention weights for every head into the value vector for that head v1 and v2 and then we get the context vector matrix for head number 1, context vector matrix for head number 2. What the context vector matrix for each head represents is that now we have 11 rows here right, the artist, painted etc. So we go from input embedding to a context vector.

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