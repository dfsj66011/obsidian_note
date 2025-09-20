
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

