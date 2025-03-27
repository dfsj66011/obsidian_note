
从基本原理出发，全面探索 Flash Attention，这不仅意味着我们要编写 Flash Attention 的代码，还要从理论上推导出它的实现过程，因此，我们假设 Flash Attention 这篇论文从未存在过，而是直接审视注意力计算本身及其存在的问题，并一步步尝试解决这些问题，通过这种方式，我们不仅能深入理解其工作原理，还能将理论与实践紧密结合，实践部分则通过编写代码来实现。为了编写 Flash Attention 的代码，我们需要为 GPU 编写一个内核程序，具体来说，是针对我们的需求定制一个内核，不过我们不会直接编写 C++ 代码，而是使用 Triton，它能够将 Python 代码直接转换为可在 GPU 上运行的 CUDA 内核程序，可以把 Triton 看作是一个编译器，它接收 Python 代码并将其转换为能在 GPU 上运行的程序。

本篇文章要讨论的主题包括：

* 多头注意力机制（MHA）
* safe softmax
* online softmax,
* 然后，我们将深入了解 GPU，因为我们要编写一个在 GPU 上运行的内核程序，因此，我们需要理解 CPU 和 GPU 之间的区别，什么事内核程序，以及它与 CPU 编写的普通程序有何不同，
* 我们将研究张量在内存中的布局方式，比如行优先布局、列优先布局，步幅等，
* 我们将探讨分块矩阵乘法，
* Triton 的软件流水线，以及 Triton 对我们代码所做的所有优化
* 最后，我们将能够编写 Flash Attention 的前向传播代码，当然，仅仅编写前向传播代码并不能让我们满足，
* 我们还希望编写反向传播代码，但要编写反向传播代码，我们还需要理解在自定义操作的情况下，自动微分 autograd 和梯度下降是如何工作的，
* 因此，我们需要理解什么是导数、梯度和雅可比矩阵，
* 然后计算我们在 Flash Attention 中使用的常见操作的梯度，
* 最终，我们将掌握足够的知识来编写反向传播代码，

### 0、先决条件

* 高中数学基础（导数）
* 线性代数的基础知识（矩阵乘法、矩阵转置等）
* 注意力机制
* 很多耐心

我们会从基本原理出发，从头开始推导一切。

### 1、多头注意力机制

快速回顾一下 MHA 是什么以及它是如何工作的，公式如下：$$\begin{align*}
\text{Attention}(Q, K, V) &= \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V \\
\text{MultiHead}(Q, K, V) &= \text{Concat}(\text{head}_1, \ldots, \text{head}_h)W^O \\
\text{head}_i &= \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)
\end{align*}$$
实际上，Flash Attention 主要关注的是 Attention 部分的优化，$Q,K,V$ 的线性层投影以及输出层投影都是常规的矩阵相乘，这一点在 GPU 中已经高度优化。


### 2、缩放点积注意力机制的局限性

#### 2.1 问题一、GPU I/O 限制

一个非常关键的问题是：我们为何需要改进注意力机制的实现方式，如果你查阅 [Flash Attention 1 论文](https://arxiv.org/pdf/2205.14135)，会注意到章节 2.2：

![[Pasted image 20250318151046.png|650]]

GPU 主要由两种内存构成，一种是 HBM，即动态随机存取存储器（DRAM），也就是 GPU 的内存，例如 A100 的 40GB 内存，这是 GPU 中容量最大的内存；此外还存在共享内存。

GPU 面临的问题是，访问 HBM（全局内存）与访问共享内存相比，速度极其缓慢，然而，与 HBM 相比，共享内存的容量要小得多，FlashAttention 论文中指出，*注意力机制的操作是 I/O 受限的*，这意味着，如果我们频繁访问全局内存，那么计算注意力机制的整体操作速度慢，这并不是因为计算这些操作本身慢，而是因为频繁访问速度较慢的全局内存导致的，因此我们可以将这类操作称为 I/O 受限型操作。

因此，改善这一状况的唯一方法是，在 GPU 的共享内存中计算注意力机制，尽管共享内存的容量要小得多，因为共享内存更靠近实际执行计算的 kernel，因此，我们需要将注意力计算拆分为更小的块，以便这些块能够放入共享内存中，然后在那里在那里计算输出矩阵的一部分，再将这部分复制到位于 HBM 中的输出矩阵中，并针对查询、键、和值矩阵划分的所有块，重复这一过程。

在论文中，他们称之为“分块（tiling）”，这是一种在编写 GPU 内核时常用的技术，尤其是在涉及矩阵乘法的情况下。现在我们了解了 FlashAttention 试图解决的核心问题。

#### 2.2 问题二、softmax

这种分块计算的最大难题在于 softmax，因为 softmax 需要访问整个 $S$ 矩阵的所有元素才能完成计算，因为需要计算归一化因子，这个因子是对所有元素逐行计算指数后的总和。


### 3、（Safe）Softmax

#### 3.1 softmax 计算上的问题

$$
\mathbf{S} = \mathbf{QK}^\top \in \mathbb{R}^{N \times N}, \quad \mathbf{P} = \text{softmax}(\mathbf{S}) \in \mathbb{R}^{N \times N}, \quad \mathbf{O} = \mathbf{PV} \in \mathbb{R}^{N \times d},$$
这里的 $QKV$ 都是经过相应线性层转换后的矩阵，$Q,K$ 的维度均为 $N \times d$，点积运算后，其输出矩阵 $S$ 的维度为 $N \times N$，softmax 操作按行处理，并不改变矩阵维度，其结果最后与 $V$ 相乘，输出维度为 $N \times d$。

softmax 操作的作用是什么呢？它会将这些点积结果进行转换，使得它们以某种方式变为一种概率分布，*按行计算*，这意味着每个数字都介于 0 到 1 之间，softmax 的定义如下：$$\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_{j=1}^{N} e^{x_j}}$$分母是向量所有维度指数的总和，这被称为归一化因子，为的是使所有这些数字介于 0 和 1 之间，使用 softmax 是因为我们希望这些数字都是正数（概率值），这是使用指数函数的原因，

但这里*存在一个问题*，问题在于，想象一下我们的输入向量由许多可能很大的数字组成，比如 100 的指数，会造成计算机结果上溢，即数值不稳定性，在计算机科学中，“数值不稳定性”意味着数字无法用我们现有的位数（通常是 32 位或 16 位）在固定表示形式中表示出来。

#### 3.2 解决方案

为了使这个 softmax 操作在数值上保持稳定，我们希望这些数字不会爆炸或变得太小以至于无法表示，我们需要找到一种解决方案，如下：$$\begin{align}\frac{e^{x_i}}{\sum_{j=1}^{N} e^{x_j}} 
&= \frac{c \cdot e^{x_i}}{c \cdot \sum_{j=1}^{N} e^{x_j}} = \frac{c e^{x_i}}{\sum_{j=1}^{N} c e^{x_j}} = \frac{e^{\log(c)} e^{x_i}}{\sum_{j=1}^{N} e^{\log(c)} e^{x_j}} \\[1.2ex]
&= \frac{e^{x_i + \log(c)}}{\sum_{j=1}^{N} e^{x_j + \log(c)}} = \frac{e^{x_i - k}}{\sum_{j=1}^{N} e^{x_j - k}} \quad \text{where } k = -\log(c)\end{align}$$
用因子 c 乘以分子和分母，通过上面推导过程，我们可以看到，如果我们巧妙地选择一个值插入到这个指数函数中，就能有效减少指数部分的计算量，我们将选择这个 $k$ 值等于输入向量中需要应用 softmax 的最大元素（$k=\max_i(x_i)$），这样一来，每个指数的参数要么为 0（当 $x_i$ 等于向量中的最大值时），要么小于 0，其指数结果介于 0 到 1 之间，这样的数值用 32 位浮点数就能轻松表示。

#### 3.3 safe softmax 算法

$$\text{softmax}(x_i) = \frac{e^{x_i - x_{\text{max}}}}{\sum_{j=1}^{N} e^{x_j - x_{\text{max}}}}$$
给定一个 $N \times N$ 的矩阵，对于每一行：

1. 寻找每一行的最大值
	* 时间复杂度：$O(n)$
	* 内存占用：$O(n)$
2. 计算分母归一化因子
	* 时间复杂度：$O(n)$
	* 内存占用：$O(n)$
3. 对向量中的每一个元素应用 softmax
	* 时间复杂度：$O(n)$
	* 内存占用：$O(n)$

伪代码如下：
```python
m_0 = -infty
for i=1 to N
    m_i = max(m_{i-1}, x_i)

l_0 = 0
for J=1 to N
    l_J = l_{J-1} + e^{x_J - m_N}

for K=1 to N
    x_K <- e^{x_K-m_N} / l_N
```

这段伪代码描述的算法相当慢，显而易见，这里存在 3 个 for 循环。所以接下来优化的思路就是寻找一种策略，合并其中的某些操作，减少循环次数。

### 4、Online Softmax

#### 4.1 online softmax

我们尝试将前两个操作融合到一个 for 循环中，这意味着我们只需要遍历数组一次，同时计算 $m_i$，并尝试计算 $l_j$，当然，我们无法在此刻计算 $l_j$，因为无法得知全局最大值，但我们可以尝试使用当前已知的局部最大值作为估算值来进行计算，即我们尝试用 $m_i$ 替代 $m_n$。

当后续迭代过程中发现更大值时，需要对过去计算项进行修正，实际上这个校正因子非常容易计算，以 $x=[3,2,5,1]$ 为例，在前两轮中最大值为 3，第三次迭代时，最大值为 5，即在第三轮迭代中，

* 错误迭代计算：$l_3 = l_2 + e^{5-5}=e^{3-3}+e^{2-3}+e^{5-5}$
* 正确修正方法：$l_3 = l^2 \cdot \textcolor{blue}{e^{3-5}} + e^{5-5}=(e^{3-3}+e^{2-3})\textcolor{blue}{e^{3-5}}+e^{5-5}$

显然这个修正因子的计算方法为过去的最大值与当前新的最大值之间的差。

因此，softmax 新算法如下：
```python
m_0 = -infty
l_0 = 0
for i=1 to N
    m_i = max(m_{i-1}, x_i)
    l_i = l_{i-1}*e^{m_{i-1} - m_i} + e^{x_i - m_i}

for K=1 to N
    x_K <- e^{x_K-m_N} / l_N
```

#### 4.2 数学归纳法证明


1. 证明对于大小为 $N=1$ 的向量，该命题成立：$$\begin{align}
m_1 &= \max(-\infty, x_1) = x_1 = \max_i(x_i) = x_{\max} \\[1.2ex]
l_1 &= 0 \times e^{-\infty} + e^{x_1 - x_1} = \sum_{j=1}^{N} e^{x_j - x_{\max}}\end{align}$$
2. 如果假设该命题对大小为 $N$ 的向量成立，证明它对大小为 $N+1$ 的向量也成立$$\begin{align}
m_{N+1} &= \max(m_N, x_{N+1}) = \max_i(x_i) \\[1.2ex]
l_{N+1} &= l_N \cdot e^{m_N - m_{N+1}} + e^{x_{N+1} - m_{N+1}} \\
&= \left(\sum_{j=1}^{N} e^{x_j - m_N}\right)e^{m_N-m_{N+1}} + e^{x_{N+1} - m_{N+1}} \\
&= \sum_{j=1}^{N} e^{x_j - m_{N+1}} + e^{x_{N+1} - m_{N+1}} \\
&= \sum_{j=1}^{N+1} e^{x_j - m_{N+1}} \end{align}$$
### 5、分块矩阵乘法

![[Pasted image 20250319151749.png|500]]

#### 5.1 忽略 softmax

目前我们先暂时忽略 softmax 的部分，即：$$\mathbf{S} = \mathbf{QK}^\top \in \mathbb{R}^{N \times N},  \quad \mathbf{O} = \mathbf{SV} \in \mathbb{R}^{N \times d},$$当然，这种做法是不正确的，但它简化了我们接下来要处理的内容，

![[Pasted image 20250320101609.png|500]]

现在，每个 query 是由 Q 矩阵中的两行组成的一个组，key 也做相应的分块，在此基础上做分块矩阵乘法，如下所示：

![[Pasted image 20250320101933.png|400]]

以 $S$ 中左上角第一个分块为例，$Q_1$ 的维度为 $(2, 128)$，$K^T$ 的维度是 $(128,2)$，即 $S_{11}$ 实际上是一个 $(2,2)$ 的小矩阵。接下来将 $S$ 矩阵与 $V$ 相乘，其结果也是显而易见的：
![[Pasted image 20250320102445.png|400]]
其运算结果为：

![[Pasted image 20250320102557.png|400]]


伪代码如下：
```python
FOR EACH BLOCK Q_i
    O_i = zeroes(2, 128)  // Output is initially zeroes
    FOR EACH BLOCK K_j
        O_i ← O_i + (Q_i * K_j^T) * V_j
    END FOR
END FOR
```

#### 5.2 softmax$^\star$

softmax$^\star$ 是去除了归一化的 softmax，$$\text{SOFTMAX}^*\left(S_{ij}\right) = \exp\left[S_{ij} - \text{rowmax}\left(S_{ij}\right)\right]
$$
将 softmax$^\star$ 应用于 $S$ 矩阵的每个块上，可以得到：
![[Pasted image 20250320142554.png|500]]

但是需要注意的是，理论上，应用于每个小块内部元素上，我们需要知道该行的最大值，但目前暂时无法获知，举例而言，假设 $S_{11}= [a\, b; c\, d]$，假设第一行的最大值是 $a$，第二行的最大值是 $d$，则 $P_{11}=[e^{a-a}\, e^{b-a}; e^{c-d}\, e^{d-d}]$，接下来，将 $P$ 矩阵与 $V$ 矩阵相乘，得到 $O$ 矩阵如下：
![[Pasted image 20250320152649.png|240]]

*再次强调：这里计算每个 softmax$^\star$ 的最大值，并不是 $S$ 矩阵这一行的全局最大值，而是每个块的局部最大值，这实际上是错误的；*

如何修正这个问题？仍然用前面介绍的 Online Softmax，我们的目标是设计一个算法，既能修正用于计算每个分块下的最大值，又能同时计算归一化因子，具体实现方法如下所述，

**初始化：**

1. $m_0 = \begin{bmatrix} -\infty \\ -\infty \end{bmatrix}$ （应该我们的分块中有两行，每行都有一个最大值）
2. $l_0 = \begin{bmatrix} 0 \\ 0 \end{bmatrix}$ （用于累积分母部分，归一化因子）
3. $O_0 = \begin{bmatrix} 0 & 0 & \cdots & 0 \\ 0 & 0 & \cdots & 0 \end{bmatrix}$ （2x128 矩阵）

**步骤 1：**

1. 计算 $m_1 = \max(\text{rowmax}(Q_1 K_1^T), m_0)$，<font color="blue">按行统计</font> $[a,b]$ <font color="blue">与</font> $m_0[0]$， $[c,d]$ <font color="blue">与</font> $m_0[1]$ <font color="blue">的最大值</font>
2. 计算 $S_1 = Q_1 K_1^T$，<font color="blue">可以将</font> $S_1$ <font color="blue">想象成一个</font> $(2 \times 2)$ <font color="blue">的矩阵</font> $[a,b; c, d]$
3. 计算 $l_1 = \text{rowsum}\left(\exp(S_1 - m_1)\right) + l_0 \cdot \exp(m_0 - m_1)$，<font color="blue">按行累积归一化因子值，最大值修正</font>
4. 计算 $P_{1,1} = \exp(S_1 - m_1)$，<font color="blue">每个元素都减去对应行中的最大值</font>
5. 更新 $O_1 = \text{diag}\left(\exp(m_0 - m_1)\right) O_0 + P_{1,1} V_1$，<font color="blue">第二项就是常规</font> $PV$ <font color="blue">计算</font>，<font color="red">第一项是对历史数据做最大值更新，同时</font> $\text{diag}$ <font color="red">是对角矩阵的含义，形式化为</font> $[m,0; 0,n] * (2 \times 128)$，<font color="red">则</font> $m$ <font color="red">仅参与和</font> $O_0$ <font color="red">的第一行计算中，这样可以保证每行的最大值仅参与该行的值更新</font>
 
注意：在该过程中暂时没有对 “softmax” 值进行归一化，此外，应该注意到这里实际是应用了两次 online softmax，分别用于在块内寻找局部最大值，并进行迭代更新，以及在块间寻找行内全局最大值，再次基于块整体迭代更新。

**步骤 2：**

1. 计算 $m_2 = \max(\text{rowmax}(Q_1 K_2^T), m_1)$
2. 计算 $S_2 = Q_1 K_2^T$
3. 计算 $l_2 = \text{rowsum}\left(\exp(S_2 - m_2)\right) + l_1 \cdot \exp(m_1 - m_2)$
4. 计算 $P_{1,2} = \exp(S_2 - m_2)$
5. 更新 $O_1 = \text{diag}\left(\exp(m_1 - m_2)\right) O_1 + P_{1,2} V_2$

继续进行该行下的步骤 3 和 步骤 4，直到最后一步，然后应用 “$l$” 归一化因子。

**步骤 5：**

1. 计算 $O_5 = \left[\text{diag}(l_4)\right]^{-1} O_4$，<font color="red">对角矩阵的逆，就是各个元素的倒数矩阵，相当于</font> $[1/m, 0; 0, 1/n]$<font color="red">，这样就实现了除以归一化因子的目的</font>

至此，对于 $Q_1$ 的注意力计算结束，可以看到，行内是逐块顺序执行的，而行间则是并行实现的，因此后续对 $Q_2$、$Q_3$ 等的计算可以和 $Q_1$ 并行实现
 
### 6、Flash Attention 前向传播
![[Pasted image 20250321163124.png|600]]

这是 Flash Attention 2 的前向传播过程，关于 Flash Attention 1 和 Flash Attention 2 之间的区别，会在后面解释。

1. 对 $Q, K, V$ 进行分块，分块大小取决于参数 $B_r$，因此每个块的大小为 $B_c \times d$
2. 初始化 $O, L$，然后接下来准备计算 softmax
3. line 3: 对 $Q_i$ 有个外层循环；line 6: 对 $K_j$ 有个内循环，与前面伪代码一致
4. line 12：计算 $O_i$，这与 5.2 中的步骤 5 完全一致
5. line 13：计算 $L_i$，这实际上是归一化因子的 $\log$，$$ \log\left(\sum_{i} \exp(x_i)\right) = x_{\text{max}} + \log\left(\sum_{i} \exp(x_i - x_{\text{max}})\right) $$

### 7、GPU、CUDA 简介

GPU 是我们购买的硬件单元，而 CUDA 是由 Nvidia 开发的软件堆栈，GPU 的任务不是同时处理多种不同的事情，而是专注于一件事或少数几件事，但处理的是海量数据，因此，我们在 GPU 上执行的操作需要大量计算，正因为如此，GPU 的大部分物理面积都用于计算单元

#### 7.1 向量加法示例

以向量加法为例：两个向量 A 和 B，各包含 8 个元素，

![[Pasted image 20250321174457.png|500]]

CUDA 的工作机制是，当我们要求它并行启动 n 个线程时，它会分配 n 个线程，并为每个线程分配一个唯一的标识符，在这个简单的示例中，我们可以这样理解，第一个线程会被分配索引 0，每个线程处理的数据项正好对应其线程索引号，

* line 14：通过代码，根据线程标识符，指定每个线程处理的数据项，
* line 15：if 语句，指定启动 8 个线程，为什么需要加 if ？在 CUDA 中，启动的线程数总是 32 的倍数，这是 CUDA 的一个基本单位（线程束，Wrap），它们共享一个控制单元，控制单元是 GPU 硬件的一部分，负责确定接下来执行哪条指令，这意味着，这组线程将始终执行相同的指令，也就是说它们会同时到达这里的 if。由于每个线程有自己的寄存器，执行时使用的数据可能各不相同，这种编程模型称为 SIMD（data），或 SIMT（thread）；因此一些通过 if 的线程执行加法，而未通过 if 的也“不得不进入”，因为它们共享同一控制单元，但需要解决这种控制流分叉问题。大致流程为：满足条件的正常执行，不满足条件的线程进入 for 循环，但啥也不干，所有的线程必须保持同步执行相同的指令，这个现象叫控制流分支分化，显然这种空闲状态会降低程序执行效率。因此应尽可能减少这种情况的发生。

> [!NOTE]
> 在 CUDA 中，控制流分支分化（branch divergence）是指同一个线程束（warp）中的不同线程执行不同控制流路径的情况。这会导致性能下降，因为 GPU 必须顺序执行每个分支路径，而不是并行执行。控制流分支分化的影响：
> 1. 线程束执行：当线程束中的线程遇到不同的条件分支（如 `if` 语句）时，GPU 会按顺序执行每个路径，直到所有线程完成。这意味着一些线程会处于空闲状态，等待其他线程完成。
> 2. 性能下降：分支分化会导致线程束内的线程不能完全并行执行，从而降低执行效率。

#### 7.2 向量分块示例

简单示例中 8 个元素，扩大到 1 M 个元素，一次分配 1百万个线程来执行任务，CUDA 会拒绝这样的请求，因为它超出了限制，当计算核心不足时，该如何管理并行计算呢？将输入向量划分为若干元素块，例如 GPU 由 32 个计算核，我们可以将输入向量划分为大小为 32 的块；而如果数据块大小为 32，GPU 有 64 个核，则一次可处理两个数据块，因此需要为 GPU 提供一定的工作粒度，需要增大数据的粒度，以便 GPU 能自主决定同时调度处理多少个数据块，这正是 CUDA 中引入块（blocks）概念的原因。

![[Pasted image 20250326093800.png]]

- grid：定义网格的维度（即线程块的数量和布局）。
- ​block：定义线程块（Block）的维度（即每个线程块中的线程数量和布局）。
- ​参数列表：传递给内核函数的参数（例如数组指针、标量值等）。

我们希望块的数量等于 $N / \text{block\_size}$，向上取整的值，因为 N 可能不是块大小的整数倍，接下来的问题是: 我们该如何将这些任务分配给每一个线程呢? 见 7.1 章节的图：

```c
int i = blockIdx.x * blockDim.x + threadIdx.x;
```

公式是：$\text{块 ID} \times \text{块大小} + \text{线程 ID}$

如果 GPU 有足够的空闲核心, 它可以选择同时运行一个或两个块，这就是为什么我们希望按块处理工作，因为这使得 GPU 在有足够核心的情况下，能够自主决定如何并行化操作，而且我们并不需要为 n 个元素的向量准备 n 个核心，我们可以将其划分为更小的块, 让 GPU 来管理调度。

#### 7.3 矩阵加法示例

![[Pasted image 20250326111258.png]]

公式是：$\text{（行 id）} \times \text{（一共几列）} + \text{（列 id）}$

在 C 或 C++ 中分配数组时，采用的是将所有行依次排列的扁平化数组结构， 因此我们需要依据行索引和列索引来定位数组中的元素，而这就是我们用来确定元素位置的公式.

![[Pasted image 20250326132814.png]]

grid 和 block 中都是三维尺寸，这里暂时用不到 Z 维度。

### 8、张量布局（Tensor Layouts）

当把一个矩阵或向量传递给 CUDA 或 CUDA 内核时, CUDA 并不会像 Python 那样一次性提供整个矩阵, 让你可以通过索引访问每个元素，而是只会给你一个指针，这个指针指向的是那个特定矩阵或向量的起始元素，然后，我们需要自己计算出所有剩余元素的内存地址。

不管是在 CPU 的内存中, 还是在 GPU 中，它将按照以下方式存储，假设第一个元素的起始地址是100，并且每个元素由一个 16 位的浮点数组成，这意味着每个元素将占用两个字节，因此，第二个元素的起始地址将是 102，第三个元素是 104 等，这正是在 C 语言中使用 malloc 分配向量或矩阵时所得到的结果，C 语言或内存分配器会分配足够的内存来存储所有元素，并会给你一个指向该内存起始地址的指针。

"步幅" (stride ) 属性告诉我们，在特定维度上需要跳过多少个元素才能到达下一个元素的位置。矩阵则会扁平化存储，按行扁平化处理的称为行主序布局，列主序布局的方式我们这里不讨论。




它在内存中的存储方式如下: 首先是第一行的元素, 紧接着是第二行的元素，假设第一个元素的内存地址是62.要访问下一个元素，我们需要将内存地址增加每个元素占用的字节数,这里是两个字节，因此，第二个元素的地址将是64, 第三个元素是66, 而第二行的无素会紧接在第一行之后开始存储.


---------

 e found in the previous for loop, so 5.
= to N 接着, 12将等于11加上当前元素的指数值,
then l2 will be equal to l1 plus the exponential of the, the current element,
= to N
e 即2减去最大值.
so it's 2 minus the maximum.
= to N 然后, 13将等于12加上当前元素的指数值.
then I3 will be, equal to I2 plus the exponential of the current element.
= to 5 minus 5, then I4 will be equal to the I3 plus exponential of 1 minus 5.
如果展开这个1,
if you expand this l,
它基本上就等于e 的三次方减五,
this will be basically equal to e to the power of three minus five
加上e 的二次方减五, 再加上e的五次方减五,
plus e to the power of two minus five plus e,
最后加上e 的一次方减五.
to the power of five minus five pl us e, to the power of one minus one minus five.
计算出归一化因子后,
After we have computed this normalization factor,
我们可以用它来归一化输入向量中的每个元素.
we can use it to normalize each element in the input vector, which means that the x -
也就是说, 新的x1(记为x1)将等于e的第一个元素3减去5的幂
new xl, so xl prime, let's say, will be equal to e,
再除以
to the power of what's the first element,
e 我们在前一个for 循环中计算得到的 L,
three minus five divided by L that we computed in the previous for loop,
Q 即第四次选代时的 L?
So the L at the fourth iteration.
新的x2.
the new x2.
而x3'则等于e的5减去5的幂除以14, 以此类推.
and x3 prime will be equal to the e to the power of 5 minus 5 divided by I4, etc.
等等.
etc.
对所有元素都如此操作.
for all the elements.
我知道这看起来非常简单, 但这将对我们后续的工作大有帮助.
i know this is super simple, but it will help us later.
因此, 在这个for 循环中, 我们需要遍历向量三次:
so in this for loop we have that we need to go through the vector three times,
第一次是为了计算这个for 循环, 第二次是计算另一个for 循环,
because first we need to compute this for loop, then we need to compute this for loop,
最后还需要再计算一次for 循环.
and then we need to compute another for loop.
我们不能打乱这个顺序执行, 因为要计算这个for 循环,
we can not do them not in this sequence, because in order to compute this for loop,
必须先知道最大值, 因为这里需要用到它.
we need to have the maximum element, because we need it here,
同样, 只有在完成前一个for 循环后, 才能计算下一个,
And we can not compute this for loop until we have computed the previous one,
因为归一化因子是必要的条件.
because we need to have the normalization factor.
然而, 我们坚持尝试将这两个操作融合到一个for 循环中
However, we are stubborn and let's try to fuse these two operations into one for loop,
这意味着我们只需遍历数组一次, 同时计算mi,
which means that we go through the array and simultaneously compute mi,
并在同一迭代中尝试计算j.
and in the same iteration we also try to compute lj.
当然, 我们无法在此刻计算1, 因为还未遍历完整个数组,
Of course, we will not be able to compute lj because we don't have the global maximum
也就无法得知全局最大值.
because we didn't go through all the array yet.
不过, 我们可以尝试使用当前已知的局部最大值作为估算值来进行计算.
however, let's try to use the locally, whatever estimate we have of the maximum so far.
于是, 我们尝试用mi 替代mn,
so let's try to use, instead of mn, let's try to use mi,
也就是用目前计算出的局部最大值来进行估算.
so the local maximum that we have computed so far.
如果我们以这种融合的方式对向量应用soft max,
so if we apply the soft max in this way, in this fused way, to this vector,
选代过程如下:这是我们的数组或向量
we will have the following iterations : So this is our array or vector,
STEP 2 第一步是计算mi.
max2=ma× (3, 2
and the first step is mi.
STEP 因此, m1将等于之前的最大值(即负无穷)
solml, will be equal to the previous maximum, which is minus infinity,
9.  负无穷与当前元素中的较大值等于3.
2-3
And L1 wilube equal'to the previous L, so LO, which starts from O, plus e,
STEP 2
2-3
2-3
STEP 2 因此暂时使用当前已知的最大值3.
2-3
STEP 2 在第二次迭代时,
max2= ma× (3, 2
Now at the second iteration,
2-3
2-3 人, 我们处理向量中的这个元素, 并计算当前的最大值.
we are at this element of the yector and we compu It e the maxim urm so. far.
Con to med
我们处理向量中的这介元素, 进计算当前的最大值.
wear e
Xecto weve "
q Qemento, Hen Re max e Qement
恩此当前的最大值是之前最大值与当前元素中的较大值.
So the maxi r
比e max e Q ement
Q ements, 归一化子等于之前的归一化因学
Qe me mts,
比e max e Qemen+
Qements, Hen
比e max e Qe men +
Lee
如果数组仅典这两介元素:(3和2)组成
lectoweve
Conto i me d e fiut
tuo
q Qemen ts, Hen We max e Qemen+
&em mts,
Con to imed he fiπ
q Qemento, Hen Ke max e Qement
01
ow因为找到的最发值3就是全局最大值
l our vector only Conto i me d 比e ft
q Qemento, Hen We max e Qement
10. 一化因字也是正确的
em
·因为每个指数值都是基于全筒最大值计算的
D
J第介元素的计算使用了3作为参数(即减去3)
Con to imed he fi π
q Qements, Hen Ke max e Qemen+
11. to i me d te fit q Qements, Hen We max e Qemen+
12. 向量的全局最关值
Rememts,
01
Con to imed he fiπt
q Qements, Hen We max e Qemen+
13. 向量
om mts,
Conto imed 比he fiπo
o Qemento, Hen Ke max e Qement
01
这也会导致我们的归一化因子出错
Rome mts,
l our vector omly
Contoimed e fiot
q Qements, Hen Ke max e Qe men +
因为我们处理到第等个元素, 也就是这里的5,
+然后重新计算最大值.
and we compute the maximum.
1,=最天值通过比较前一个最大值和当前元素得出,
so the maximum is the comparison of the previous maximum and the current eleme nt,
因此新的最大值变为5.
so the new maximum becomes five
归一化因子则是前一个归一化因子12 and the normalization factor is the previous normalization factor,
5 加上当前元素
so l2 plus the exponential of the current element,
2= 减去当前估计的最大值(即5)的指数结果.
minus the current estimate of the maximum, which is five.
however, If you look at this L3, this is wrong.
2-5
5-5
Com we x计
O因为 HL3 等于.? y Es!
Because L3 is equal to.
3-3
5-5
if you expand this summation 3-3
2-3
5-5
Com it will be equal to e to the power of three minus three plus e.
3-3
2-3
3-5
5-5
(2减3)次方再加上e的.
Iele I'to the power of two minus three plus e.
5减5)次方.
Hene il i to the power of five minus five.
Hene it in wx omg
这里的指数计算使用了3 作为全局最大值,
this, exponential here is using three as the global maximum
而这个也是以3作为全局最大值的.
I and this one is using'three as the global maximum.
Hene it in wx o my
所以前两个元素在计算时是以3作为全局最大值来处理的
so the first two elements have been computed thinking that the global maximum is three,
但实际上我们后来发现了一个更优的全局最大值, 也就是5,
but actually we later we found'a better global maximum, which is five,
Hene it in wx o my
这导致了这个归一化因子是错误的.
Here it in wx o my
我们能杏在第三次选代时修正前两次迭代中已经计算出的归一化结果
however, Can we fix at the third iteration'whatever normalization we have computed so far
Hene it in wx omg
呢?
Hene it in up to the'second iteration?
Hee it in wx omg
事实上, 我们是可以做到的.
Hene it iy wx Actlially we can.
Hee it in wx omg
On the fly? yes! 因为如果我们将其展开, 就像这里展示的那样, 我们已经完成了展开,
Because fe
5
这里我们需要的是减去5 FS!
2-5
5
因为这才是目前为止我们找到的全局最大值,
because that's actually the global maximum that we have found so far,
3-5
-5
5
而不是之前选代中使用的3!
3-5
-5
5
So, and here we ai so need to fix this, replace this minus three with minus five.
How can we do that?
5
其实, 如果我们用校正因子分别乘以这两项,
Well, if we multiply this one here and this one here with a correction factor that will.
2-5
5-5
couection
2-3
3-5 从而将新的最大值巧妙地引人到指数计算中, 问题就迎刃而解了.
sneak in a new maximum inside of this exponential, then we solve the problem.
cou ection
5-5
couection
3-5
, 实际上, 这个校正因子非常容易计算.
And actually this correction factor is very easy to calculate,
cou ection
courection
facton
2-3+3-5
5-5
co在第兰次迭代时, 如果我们用 L2
because'at the third iteration, if we multiply L2,
2-3+3-5
3-5
2-5
5-5
(即之前计算得到的归一化因子)乘以这个因子
so the previously coin puted normalization factor, with this factor here,
+3-5
2-3+3-5
3-S
2-5
5-5
couect 它是之前估计的最大值
2-3+3-5
3-52-5
5-5
co uection
facton
2-3+3-5
3-5
2-5
5-5
与当前估计的最大值(即5 )之差的指数函数
Minus the current estimate of the maximum,.
2-3+3-5
3-S
2-5
5-5
so five, we wilt see that by the properties of the exponential s
2-3+3-5
3-5
2-5
This one here will become e to the power of 3 minus 3 plus 3 minus 5.
这样一来, 这个负3就会与这个正3相互抵消
So this minus 3 will cancel out with this 3.
And also the second factor will have this 3, we cancel out with,
这个负3相互抵消.
CORRECT
this minus 3 will cancel out with this 3.
于是, 它们将分别变为e的3减5次方和é的2减5次方,
And they will become e to the power of 3 minus 5 and 2 to the power of,
e to the power of 2 minus 5, which is actually correct.
Because at the third iteration, we should be actually having,
we should be using minus 5 as the maximum of the array so far.
所以, 本质上,
Co Rrec T So, basically,
我们发现了一种方法, 可以在遍历数组的过程中
what we have found is a way to fix whatever normalization factor we have computed so far,
while iterating through the array
when we find a better maximum compared to what we have so far.
而当无需修正时, 这个公式依然适用
And when we don't need to fix anything, then the formula still stands,
因为我们在这里引入的校正因子(即这个表达式)起到了修正的作用
because what we did here as a correction factor, so this is the correction factor.
facton
3-3+3-5
2-3+3-5
+e
这个校正因子其实就是之前估计的最太值5-5
facton
3-3+3-5
so, the previous estimate-of the maximum P
facton
3-3+3-5
2-3+3-5
P
facton
2-5
5-5
facton
3-3+3-5
so the current maximum.
P
3-3+3~5
2-3+3-5
P
也就是说;一这里的是前一次送代中的最大值m(i-1),
So this is basically m of i minus one, and this is m of i,
而这是当前送代中的当前最大值"m(i).+
5-5
3-3+3-5
2-3+3-5
让我删掉它, 不然它会一直留在我的幻灯片里:
5-5
And let me delete it, otherwise it remains forever in my slides.
o R Rec T
基本上, 当我们处理到最后一个元素时,
So basically, when we arrive to the last element 10
基本上, 当我们处理到最后一个元素时
C
会发现最大值没有变化,
we will see that the maximum doesn't change,
因为我们将之前记录的最大值与当前元素进行比较,
NO NEED because we compare the previous maximum with the current element,
而当前元素小之前记录的最大值, 所以最大值保持不变.
which is less than the previous maximum, so the maximum doesn't change.
我们无需进行任何修正,
RE DO NOT NEED 因此, 因为之前的 L3 And we don't need to fix anything because the previous L3,
so the previously computed normalization factor is correct
because they have all been using the minus five.
M.= -∞
New meu do 因露, 当我们无需进行任何修正时,
So when we don't need to fix anything,
M.= -∞
we just multiply by e to the power of the previous maximum minus the current maximum,
而在这个例子中, 这个差值恰好为零, 所以e的幂次方就是e的零次方.
which is e to the power of zero in this case.
M.= -∞
So it's not fixing anything.
M.= -∞
New mu do co 此, 我们找到了一种方法,
So we have found a way to fix the previously computed normalization factor
可以在遍历数组的过程中修正之前计算得到的归一化因子,
while going through the array,
M.= -∞
even if at the current iteration, we don't have the global maximum yet.
M.= -0
so that every time the maximum change s we can fix, and every time it doesn't chan ge,
只需乘以的零次方, 这相当于乘以1, 对结果没有影响.
we just multiply with e to the power of zero, which is like multiplying with mol.
so the new algorithm that we have found for the soft max is the following :
e 首先, 我们将m0初始化为负无穷.
so we start with mo, equal to minus infinity.
e 接着, 我们将10初始化为零.
we start with lo, equal to zero.
e 我们遍历数组, 计算局部最大值, 即
we go through the array, We compute the local maximum, so the maximum so far,
x←
e 从第0个元素到当前迭代的第1个元素
from the Oth element to the ith element,
e 之间的最大值.
so to the element at which we are doing the iteration.
X 而之前计算的小i可以通过这个修正因子进行修正,
And the previously computed li can be fixed by using this correction factor,
e 修正因子为e的前一次最大值与当前最大值之差的幂次方.
which is e to the power of the previous maximum minus the current maximum.
e 再加上当前元素的值减去当前估计的最大值的指数.
plus the exponential of the current element minus the current estimate of the maximum.
e 通过这种方式, 我们只需遍历数组一次, 就能同时得到两个值:
In this way, we go through the array only once and we obtain two values,
e 全局最大值和归一化因子.
the global maximum and at the same time the normalization factor.
e 然后, 我们就可以利用这些值来计算 Softmax 了.
And then we can use it to compute the soft max.
这样一来, 我们将原本需要三次遍历数组的过程优化为仅需两次遍历
So we transform the three passes through the array into two passes through the array.
这一点至关重要
and this is very important -
我们稍后将看到如何实际运用它来推导出 Flash Attention.
and we will see how we actually use it to derive flash attention.
e
X-XMAx 到目前为止我举的这个例子并不能完全证明
the example that i have given you so far is nio t really a proof
我们的算法在所有情况下都有效,
that our algorithm will work in every case,
e
-XMAX 因为我们只是用了一个由四个元素组成的向量作为简单示例.
be cause we made a very simple example by using a vector made up of four elements.
那么, 我们的新算法是否适用于所有情况, 无论数值如何变化呢?
but does our hew-algorithm work in every single case, with whatever the numbers are?
这需要我们进行证明.
we need to prove that.
因此, 我们将通过数学归纳法来证明这一点.
So we will prove that by induction.
R
So, first of all, what are we trying to prove?
R 如你所见, 我们已经将前两个 for 循环融合成了一个for 循环.
We have fused the first two for loops into one for too p, as you can see here.
我们期望的是, 在这个for 循环结束时, Mn 一-
What we expect is that at the end of this for loop, this Mn -
也就是最后一次迭代时的 M一一家 实际上就是向量中的全局最大值
so the M at the last iteration-will be actually the global maximum in the vector,
而 Ln一一最后一次迭代时的
LL
and this Ln-so the L at the last iteration -
将等于向量中所有元素减去全局最大值
will be equal to the sum of all the exponential of all the elements
后的指数和.
minus the maximum element of the vector, so the global maximum of the vector.
我们需要对此进行证明, 因为之前我所做的只是一个示例,
And we need to prove that, because what I did before was an example,
并不能算作严格的证明.
and that was not really a rigorous proof.
我们将通过数学归纳法来证明这一点,
And the way we will prove it is by induction,
这是证明这类定理的典型方法.
which is a typical way of proving this kind of theorems.
现在, 归纳法的证明基本遵循以下步骤.
Now, proof by induction basically works in the following way.
我们需要证明我们的算法在基础情况下是有效的
We need to prove that our algorithm works for a base case,
例如当n等于1时, 然后我们假设
for example with n equal to one, and then we pretend,
we assy me that r the algorithm works on n and we need to prove X :
+
X N+ I"
XN+ I
X+1-m+1
e
如果这一 条件成立, 那么我们就证明了我们的算法对于所有可能的n 都是有效的
If this holds, t then we have proven our algorithm for every possible n,
因为它已经在基础情况下得到了验证.
Z because'it will work for the base case.
例如, 当等手1时成立, 然后通过归纳步骤我们可以说:
so, for example,(nequal'to1, and then by using the induction step we say :
XN+ I
X+1-m+1
如果它对n有效, 那么它对n加1也有效,
X N+ I
这意味着它对2 也有效.
then'it means that'it will also work for 2.
X N+ I
e
接着, 如果它对-2有效,
But then ifit woriks for 2, then it should also work for 3,
X N+ I
XN+ I 那么由于我们将证明的归纳步骤, 它对3也应该有效.
because of the induction step that we will prove.
X N+ I
XN+ I 如果它对3有效, 那么它对4也有效以此类推, 直到无穷大.
And if it works for 3, then it will also work for 4, etc, etc, up to infinity.
XN+1
e
让我们从基础情况开始证明, 即当n等于1时.
So'let's prove it far, the base tase,'which is n equal to 1.
e
+e
m NI=max (m, Xx+)
XN+1-m N+1
这非常简单.
max + kt's'very simple.
当n等于1时,"这个for循环只会执行一次迭代, 因此只涉及m1和11.
So at n equal tol, this'for loop will only have one iteration, so ml and I1.
m T将是前一个m的最大值, 即负无穷大
m1 will be the maximum of the previous m, which is minus infinity,
max
J=1
vectoonize
无论×1是什么值+x1. 通常不会等于负无穷太
x1 不可能等于负无究太, 因为密是一个固定表示形式的数值.
it can not be equal to minus infinity b because i it's a number in'd fixed representation.
因此, 由于当前只有一个元素x1, m1c既是这个元素的m值,
也是这介for循环中的最后一个m值
we me cve cie
m1 将等于仅电一个元素组成的向量的全局最大值
因此, 10乘以一个修正因子
so l0 multiplied by a correction factor, which will be,
e&rcrv eco ie, in this case'e to the power of minus infinity,
because the correction factor is the previous estimate of the max
eevec
, doe i+ old fon a vecto o nie +?
but the previous estimate of the max is minus infinity.
I, d ominus xi it is equal to minus infinity.
N, doep i+ hold fon a vecto o nize N+?
Iso this one will be okay, this will be cancelled out.
and then pl us e to the power of xi, minus the current maximum, which is xl, so m1.
And if this one r will be equal'to the sum of all the elements of the vector,
而向量仅由一个完素组成, 减去数组中的最大值×1.
which is made up of only one element minus the maximum element in the array, which is x1
I So we have proven that it works for n equal to one.
N, doep i+ old fon a vecto o nize N+?
Sme+h&orcrvectonie
I, d oo Now we assume that it works for n.
那么, 对于天小为n加1 的向量数组这个方法是否仍然适用呢?
Does it also work for an array with a vector of size n plus one?
P
x-m+
x-+1
XN+1-m+1 那么, 让我们来看看在第n加1次迭代时会发生什么.
So, let''see what happens at the n plus one iteration.
在第n加 1次迭代时, 我们将计算之前m的估计值(即第n次迭代时的 m
At the n plus one iteration, we will be doing the maximum of the previous estimate of m,
与当前元素×n加1的最大值.
which is the m at the nth iteration, and the current element, so x n of plus one.
e 这是根据最大值函数的性质得出的.
this by the properties of the max function.
实际上, 这等于全局向量直到n加1的最大值,
it will be actually equal to the maximum of the global vector, up to n plus one,
为最大值函数会在之前的估计值和当前的估计值之间选择较大的那个
because the maximum will choose whatever is the maximum between the previous estimate
= 而一n加1是归一化因子.
and the current estimate and I n plus one, which is the normalization factor.
在第n加 1次迭代时, I n加 1将等于 In.
at the n plus one iteration will be equal to the In.
因此, 前一个估计值(实际上是前一个归一化因子)
so the previous estimate, not previous estimate,
在迭代结束时
by the previous normalization factor at the end iteration,
= 乘以校正因子,
multiplied by the correction factor,
该校正因子是前一个最 大值减去当前最大值, 再加上当前元素×减去当前最大值估计值的指数
which is the previous maximum minus the current maximum, plus the exponential of x,
= 但最终结果仍然是 In.
the current element minus the current estimate of the maximum, But In.
我们假设这个性质, 民 即这个算法, 在n 次迭代时是成立的.
we assume that this property, so this algorithm, works up to n.
因此, In肯定等于向量中前n个元素的指数之和
So In is for sure equal to the sum of all the exponentials of the previous,
XN+1-m2+1
e
X-m+1
X+1-m+1 减去向量中前n个元素的局部最大值
of the vector up to n minus the local maximum of the vector up to the nth element,
XN+1-m2+1
e
J=
XN+1-m+1
J= 即 mn.
which is mn.
XN+1-m2+1
e
如果有需要校正的地方, 我们乘以校正因子,
We multiply by the correction factor, if there is something to correct,
该因子由前一个最大值减去当前最大值,
which will be the previous maximum minus the current maximum,
再加上当前元素的指数减去当前最大值的估计值组成.
plus the exponential of the current element, minus the current estimate of the maximum.
接下来, 通过
Now, by...
指数函数的性质.
the properties of the exponentials.
因此, 我们可以将其带入求和式中, 并会发现这个mn so we can bring this one inside of the summation and we will see that this mn
和这个mn 会相互抵消,
and this mn will cancel out,
因为最终表达式会是xj减去mn的指数, 再加上mn减去mn加 1
because it will be exponential of xj minus mn plus mn minus mn plus one.
所以, 这 个mn 和这个mn 会相互抵消, 我们最终得到这个结果加上这里的这个因子
so this mn and this mn will cancel out and we obtain this one plus this factor here
保持不变.
that remains unchanged.
然而,
However,
你可以看到,
you can see that
这里的内容正好是迭代n加1时求和项的精确参数.
this stuff here is exactly the argument of this summation at the iteration n plus 1.
所以, 这个表达式就是e的xj次方, 其中j从1到n, 减去mn加
So this one is e to the power of xj, where j is going from 1 to n, minus m n plus 1.
再加上e的x, n加1减去m, n加 1次方.
plus e to the power of x, n plus 1 minus m, n plus 1.
因此, j仅出现在这里, 其最大值为n,
so the j only appears here and it's equal maximum to n
而这类似于将i替换为n加1的情况.
and this is similar to being a j with n plus 1.
因此, 我们可以将这个求和的索引增加1, 结果将保持不变
so we can increase the index of this summation by 1 and it will be the same
最终得到的求和结果也相同.
and it will result in the same summation.
因此, 我们已经证明, 在n加1次迭代时,
so we have proven that also at the n plus one iteration we will have that
1仍然等于数组中所有元素的指数之和,
the I will be equal to the sum of all the elements of the array,
即数组中从第1个元素到第n加1个元素的指数,
the exponential of all the elements of the array up to the n plus one element,
减去这些元素中的最大值.
minus the maximum up to the n plus one element.
于 是, 我们已经证明了, 如果这个方法对n成立, 那么它对n加 1也同样适用
So we have proven that if it works for n, then it also works for n plus one.
+e 这足以证明该方法适用于任意大小的数组.
This is e hough to prove that it works for all size of arrays.
如果你没完全理解这个归纳证明, 也不用担心.
Don't worry if you didn't get the proof by induction.
如果你是第一次接触这类证明
If it's the first time you are seeing this kind of proof,
可能需要一些时间才能完全掌握.
it may take a little bit to get it.
如果你想进一步了解归纳证明
If you want to learn a little bit more about proof by induction,
我建议你观看一些其他的证明过程.
I recommend watching some other proof.
其实归纳证明很简单, 只需要调整到正确的思维方式.
It's very simple, it's just you need to get into the right mind set.
好了, 我们继续往下讲.
Anyway, let's move forward.
A. B
AB
AB
AB3
A. B 好的, 我们接下来讨论分块矩阵乘法
A2
llright, let'stal
A. B
ABg
An B2
An B
H
A. B
AB 我知道你可能想直接跳到代码部分 我们稍后就会讲到那里
I know that you want to j ur
li a tel
ly'and we will go there.
A. B
AB
AB
AB3
A. B 实际上, 我们还需要补充 点理论背景
H
We justne
AB3
A. B
AB
AB
AB3
A. B
AB 假设我们正在进行矩阵乘法运算
imagine w
A. B
AB
AB
A2. B
AB3
A. B 想要将其与矩阵 B相乘, 结果 生成
we want to multiply it with an'output matrix C.
AB
AB
AB
H
AB3
A B 假设第一个矩阵的维度是 M. 乘以
Tmaginethe
xare
rstmatri
A. B
AB
An B21
AB
H
A. B
AB 第二个矩阵的维度是
K 乘以
H
AB3
The
a
A. B
ABg
AB
H
AB3
A. B 结果将生成十个 M 乘以 出矩阵,
H
It will produ hat
A. B
AB
AB
H
A. B
AB3
A. B 现在, 假设我们希望并行化计算这个输出矩阵.
Now, imagine we want to
A. B
AB
An B2
AB
A. B
AB3
A. B
AB 我知道我还没提到 GPUI月 所以 人这里我们先不讨论 IGPU
GP
I not talk'about'GPus.
A. B
AB
AB
H
AB3
A. B
AB
AB 我们将讨论在多核 CPU情况下的并行化
H
on
eoi
A. B AB
AB
A. B
B
with which
A. B AB
An B
H
A. B
A. B
AB 因为如今在购买电脑时 通常
CPU
A2
because right now lin-now a
a'computer you have a cpu buy
A. B
AB
AB
A. B
AB3
A. B 而-CPU 又分为单核和多核 比如奴
and usually you can buy a core, four core,
sin a two
A. B
An B
H
A. B
AB3
AB 这些核心实际上就像是 部的小
CPU
H2
each of the these cores inside your pus Cpu
AB
AB
AB 能够并行执行操作
AB
H
that can
AB
AB
AB
H
A: B2
AB3
AB
AB 如何并行化矩阵乘法呢?
how to para
AB
AB
AB
An B2
An B
A. B
AB
AB
ABg 假设你需要并行化这个矩阵乘法
A2
imagine you have
AB
AB
ABg
B
AB
H
AB3
AB
AB
ABg 矩阵 C中的每个输出元素.
H2
each of the output element in t
dot
A. B
ABg 都是矩阵 A的一行与矩阵 的一列的 点积
H
A: B2
AB3
With m
atrix
A. B
AB
AB
AB
A. B
AB3
A. B 例如, 左上角的这个元素是 是矩阵 A的 第
for example, this element on
A. B
An B2
AB
H
AB3
A. B 与矩阵 B的第一列的点积结
H
b
A: B2
AB3
al
A. B
An B2
AB
H
A2. B
Aa B3
A. B 而矩阵 C右上角的这个元素, 则是矩阵 A·的第一行
this element on the top ri
vof a
AB
AB 写矩阵
B
15
A. B2
AB3
A. B
ABg
An B
H
A. B
AB3
A. B 左下角的这个元素是矩阵 A的最后一行与矩阵 B的第一列的点积
This element on the bot
A. B
AB 以此类推, 其他所有壳素的计算方式也是如此
and the first column of B or other elements.
A. B
AB1
An B
A. B
AB 现在, 为了并行化这一计算,
Now, to
A. B
AB
AB
AB3
A. B 如果我们希望完全并行化, 就需要与矩阵 C 中元素数量相当的核心数
we need as many cores as th
ents in
A. B
An B2
An B
A. B
AB3
A. B 因此, 如果
m和n的值很小, 或许我们现有的核心数量就足以应对
So if m and n are very be
A. B
AB
H
AB3
A. B 旦设想一下, 当
m 和n的数值相当大时
tel
A. B
AB 假设矩阵的规模达到
100 乘以100
H
We
100.
A. B2
AB3
by
A. B
AB
An B
H
AB3
A. B
AB 目前我们的 CPU并未配备
10000个核心
H
We don't have
ow in'the CPUs htn
A. B
ABg
A. B
A. B 那么, 如何在核心数少于矩阵元素数量的情况下
B
So how can
A. B
ABg
AB
AB
AB3
A. B 实现矩阵运算的并行化呢
by using less cores th itself?
A. B B
AB
A. B
AB
B 这便是我们引人分块矩阵乘
that's when we
A. B
ABg
An B
AB
A B2
ORi GINAL(, 8
ORi Gi NAL( B, 8)
ORi GWAL(8,)
BLOCK(2, )
BLOc K (2, 4)
BLOC以 (2, 2)
AB1 简而言之
ORi Gi NAL8, 8)
ORi GWAL(8,)
Bloc basicelly,
BLoc K (2, 4)
BLOCk (2, 2)
ORi GINAL (4, 8
ORi Gi NAL(8, 8)
oeicw L(分块矩阵乘法指的是将原始短陈(2")
block matrix mn ulti plication means that you can divide the original matrix
ORi GIN A(, 8)
ORi Gi NAL( B, 8)
ORi GWAL()
BLOck (2, 2)
BLoc K (2, 4)
BLocu (2, 4)
ORi GINAL (, 8 ORi Gi NAL (8, 8)
ORi GWAL() 划分为更小的元素块,
BLOCK (2, 2)
ORi GINAL (, 8)
ORi Gi NAL( B, 8)
ORi GWAL()
BLOCK (2, 2)
BLOCK (2, 4)
BLocu (2, 4)
ORi GINAL (, 8
ORi Gi NAL( B, 8)
oricw然后在这些块之间进行矩阵乘法的运算.
and then the operations of matrix multiplication can be computed between these blocks.
BLOCK MATRi X ORIGi NAL MATRi X
= BLOCK MATRi X 设想
1有一个8行
4列的矩阵.
have
a matrix that is 8 by 4.
BLOCK MATRi X ORi Gi NAL MATRi X D24
A21
= BLOCK MATRi X 这意味着它有8行和4列, 总共包含32个元素.
itmeansthat-i
thas8
and four whi
ch means that it has 32 elements.
BLOCK MATRi X ORi Gi NAL MATRi X
= BLOCK MATRi X 我们将它与另一个4行
8
8列的矩阵相乘
And are with another matrix that is four by eight.
= BLOCK MATRi X ORi Gi NAL MATRi X
= BLOCK MATRi X 也就是说, 这个矩阵有
4 行和8列
Sur
ght columns.
BLOCK MATRi X ORi Gi NAL MATRi X A21
= BLOCK MATRi X lem
ents.
= BLOCK MATRi X ORi Gi NAL MATRi X A21
= BLOCK MATRi X ei Ci NAL 取
have
64 elements.
= BLo CK MATRIX ORi Gi NAL MATRi X A21
= BLo CK MATRi X 我们的处理器并没有
5
res.
= BLo CK MATRIX ORi Gi NAL MATRi X D24
A21
= BLOCK MATRi X 那么小我们该如何实现并行化?
Gi NA L
pal
rall elize it?
= BLOCK MATRi X 假设我们仅有8个核心可用
ORIGi NAL
OATRi X
we have eight cores.
BLOCK MATRi X ORi Gi NAL MATRi X D24
A21
= BLOCK MATRi X 现在, 利用这8 个核心,:我们可以将原始矩阵 A划分为四个区块
inal matrix A into four blocks,
= BLo CK MATRIX ORi Gi NAL MATRi X A21
= BLOCK MATRi X 4行2列的元素.
W
of four by two elements.
= BLo CK MATRi X 该如何表达呢?
ORi Gi NAL MATRi X So
ow
ay?
A
B23
= BLOCK MATRIX ORi Gi NAL MATRi X A21
= BLOCK MATRix 左上角有8
14. 阵右上角的8个元素,
eight elements the
te
lements on the top right of this matrix,
Az
= BLOCK MATRi X 然后是左
on the bottom left
= BLOCK MATRi X 不角的
m
ight of this matrix.
= BLo CK MATRi X ORi Gi NAL MATRi X four blocks.
BLOCK MATRi X ORi Gi NAL MATRi X D24
= BLOCK MATRi X B
3也划分为八个区块
ma
itrix
into eight blocks,
= BLOCK MATRi X wh four elements.
= BLOCK MATRIX ORi Gi NAL MATRi X D24
A21
= BLOCK MATRi X B11. 代表原始矩阵中左上角的四个元素.
the. top.
eft
ents
in the original matrix.
BLOCK MATRi X ORi Gi NAL MATRi X A21
= BLOCK MATRi X 台矩阵中右 后上角的四个元素.
nen
ts in the original matrix.
BLOCK MATRIX ORi Gi NAL MATRi X 24
= BLo CK MATRIX = ORi Gi NAL 00 MATRi X
ne
um.
BLOCK MATRIX ORi Gi NAL MATRi X D24
A21
= BLOCK MATRi X 位于矩阵左下角的四个元素 以此类推.
ootto
S
th
legitimate, etc.
A2
BLOCK MATRIX ORi Gi NAL MATRi X D24
A21
= BLOCK MATRi X 我们如何进行这种分块矩阵乘法呢?
how
this
ma
trix
multiplication?
MATRi X 将这些矩阵视为仅由其分块组成的结构
ve can watch these matrices as made only by their blocks,
这里可以将这个矩阵看作仅由各个分块构成.
so we can view this matrix here as made up only by its blocks.
A21
我们可以将这个矩阵看作仅由其分块组成的结构.
we can view this matrix here as made up only by its blocks.
A
这种乘法的输出结果将是一个矩阵,
and the output of this multiplication will be a matrices
A21 其计算方式与原始矩阵相同,
that is computed in the same way as the original matrix
旦每个点积的输出不再是输出矩阵的单个元素
but where the output of each dot product will not be a single element of the output matrix
而是输出矩阵的一个元素块.
but it will be a block of elements of the output matrix.
A21
如, 这里的左上角分块是这个矩阵的第一行与这个矩阵的第一列的点积
for example, the top left block here is the dot product of the first row of this matrix,
其计算方式如下:
with this first column of this matrix, and it will be computed as follows :
它将由 a 11乘以 b11加上 a12乘以 b21得到,
so it will be a11 multiplied by b 11 plus a12 multiplied by b21,
而这个输出结果不再是一个单一标量, 而是一一嗯, 让我数一下.
and this output will not be a single scalar but it will be - uh, well, let me count.
MATRi X 应该是日 由八个元素组成.
it should be, u h, eight elements.
MATRi X 它应该是由四个元素组成.
so it should be four um made up.
MATRi X
MATRi X 个由四介或八个元素组成的块, 让我实际数一下.
be a bloc of four elements or eight elements, let me count actually.
因为我们有八个分块, 所以它应该由八个元素组成.
So because we have eight blocks and it should be made up of eight elements.
22 我们可以在这里看到这一点.
We can see that here.
MATRi X ORi Gi NAL MATRi X
BLOCK MATRi X 如何确定这个输出块的维度?:orci v AL
ORIGINAL 如何确定这个输出块的维度?
MATRi X sion so i
Fthis output block?
MATRi X = BLOCK ORi Gi NAL MATRi X A
MATRi X we MATRi X A
A11是一个4行2列的矩阵,
MATRi X
Al1isfour
two,
so i t"
ite
na smaller matrix mad ORIGINAL A
MATRi X AB
MATRi X ORIGi NAL A MATRi X
我们将其与 B11 相乘
MATRi X iultiplying it by B11
ORIGi NAL Ne
arel MATRi X A
MATRi X ORIGi NAL A MATRi X
MATRi X which isa to the or
A
MATRi X ORi Gi NAL A MATRi X
因此包含4 个元素
= BLo CK MATRi X ORIGi NAL A MATRi X
MATRi X ORIGi NAL A MATRi X
因此,
ATRIX
4 by 2, multiply by
ARi X
因此, 当我们将4行2列的矩阵与2行2列的矩阵相乘时
会得到一个
4行2列的输出块矩阵
it will produce a
Az B2
ORi GINAL (, 8
ORi Gi NAL(8, 8)
ORi GWAL (8,)
BLOCK BLoc K (2, 4)
BLOCK (2, 2)
= BLo CK
MATRi X
MATRi X
A kz 所以:如果我们逐块进行这个计算,
So if we do this computation here block by block,
B23
B24
将生成原始矩阵的一个输出元素块.
it will produce a block of output elements of the original matrix.
因此, 不是单个标量, 个输出块, 这使得并行化变得非常容易
So not a single scalar but a block of outputs which makes it very easy to parallelize,
因为如果我们只有火个 我们可以将每个输出块分配给一个核心
because if we have only eight cores, we can assign each output block to one core
生成原始矩阵的一个输出元素
and each core will not produce one output element of the original matrix,
而是会生成原始矩阵的八个元素, 形成一个4行2列的矩阵.
but it will produce eight elements of the original matrix as a four by two matrix.
而是 会生成原始矩阵
4 行
2列的矩阵
butitv
rod
ourbi
AB
A
A
22
12
ORi GINAL (, 8
ORi Gi NAL(, 8)
ORi GWAL(8,)
H 嗯,
BLOCK
= ORi Gi NAL MATRi X 块矩阵允许我们
matrix, allow us to, um,
ORi Gi NAL MATRi X
= ORi Gi NAL MATRi X 两种方式进行矩阵乘法:
:一种是逐元素相乘,
to do the matrix multi p
ication either by element by element,
ORIGi NAL MATRi X
= ORi Gi NAL MATRi X 就像我们进行普通矩阵乘法一样.
in the same way like we do normal matrix multiplication,
= ORi Gi NAL MATRi X 因为我们在块之间进行的矩阵乘法
I because the the matrix multiplication
与我们在原矩阵上进行的矩阵乘法方式相同
that we are doing between blocks is the same way
只是它生成的不是一个标量
as we do matrix multiplication with the original matrix
而是一个块.
and it will produce not a scalar but a block.
现在, 让我们来看看为什么这一点对我们来说非常重要.
and now let's see why this is very important for us.
= BLOC = ORi Gi NAL We kmo w
a+, k, V
MATRi X
129
BLOCK
128
128
= BLOC ORIGINAL w那么我们为什么要关注块矩阵乘法呢3
So why should we care about block matrix multi plicatior
129
BLOCK
128
128
= BLOCK = ORi Gi NAL We k mow MATRi X 129
128
BLOCK
128
128
= BLo C ORi Gi NAL We km 因为我们正试图计算以下操作
MATRi X Because we are trying to compute the following opera ti
129
2
Bioc K
128
128
= BLo C 具体来说, 就是将查询矩阵与键矩阵的转置相乘,
So the query multiplied by the transpose of the key
129
128
BLOCK
128
128
= BLo C = ORIGi NAL We kmow
MATRi X
129
BLOCK
128
128
= BLo C R RIGi NAL 然后对这个操作的结果应用soft max and then we should apply the soft max of this operation 129
BLOCK
128
128
= BLOC 最后将sof max 的输出与值矩阵相乘. rx ORi Gi NAL we and then we should multiply the output of the soft max with 129
128
BLOCK
128
128
= BLOC = ORIGi NAL We k mow MATRi X 129
128
BLOCK
128
129
= BLOC w 目煎:我们先时忽略soft max的部分i
ORIGi NAL for now, let's ignore the softmax.
129
BLo CK 1 ()
128
128
= BLo C w假设我们不打算应用任何rsoft max 区 函数 TRix ORi Gi NAL let's pretend that we are not going to apply any softmax
129
BLOCK
128
128
= BLo C R = ORIGi NAL We km ow Ha+ Q, k, V
MATRi X
129
BLOCK
128
128
= BLOC Ri Gi NAL 因此, 我们取查询矩阵与键矩阵转置相乘的结果,
so we take the output of the query multiplied by the. transpose c
keys
129
28
BLOCK
128
128
= BLo C 将其与值矩阵v 相乘, 得到注意力机制的输出. 当然, 这种做法是不正确的
and we just multiply it by v to obtain the output of the attention, V
iswrong
129
BLOCK
128
128
= BLOC 但它简化了我们接下来要处理的内容.
ORIGi NAL of course, but it simplifies our tract ation of what we are going next.
129
128
28
128
= BLOC = ORIGi NAL We km ow
R
MATRi X 129
128
BLOCK
128
128
ORIGi NAL 所以, 就目前而, 手 我们暂且假设不需要应用任何soft max 函数
so for for this moment, let's pretend that we are not gg
gto ap
softmax
129
BLOCK
128
128
Q2
So-we just do the query multiply b
Q4
[28
128
Q2
( V3)
Q3
an
d directly we multiply the result of this oper ration with W!
Q4
127
128
128
Q2
Q3
BLOCK4 ()
Q4
128
128
Q 2
Q3 这样得到的结果将是一个 个 nxd 的矩阵,
this will
Q4
128
127
[28
Q2 其中n代表 图一个d维
so n tokens each made up of an embedding of d dimensions,
So lowercase d dimensions.
Q4
127
128
128
Q2
Q3
BLOCK4 ()
Q4
128
128
Q2 我们知道, 查询,
(query ) 键(key )和值(value ) 本身也是nxd维的矩阵
Q4
128
123
128
ORi G( NAL =(8, 128) 也就是说, 这些. n个token是由d维的嵌入向量组成的.
so the um, n tokens which made up of an embedding of the dimensions.
ORi G( NAL=8, 128) 想 象一下, 我们有元个8×128的查询矩阵, 以及同样大小的键矩阵和值矩阵,
so imagine we have a query matrix and the key and the value matrix that are eight by 12:.
Ri G( NAL=8, 128)
we, so we have eight tokens.
ORi G( NAL =(8, 128)
1 L. c每个token 由128个维度组成.
each token is made up of 128 dimensions.
Ri G( NAL=8, 12%) 正 如我们所看到的,, 在进行矩阵乘法计算时, 我们可以将矩阵进行分块处理
we can divide, as we have seen, each when we compute a matrix multiplication.
ORi G( NAL=8, 12%)
1 L. 我们可以将矩阵划分成多个块.
we can divide our matrix into blocks.
Ri G( NAL=(8, 128) 如何选择分块方式完全取决于我们, 只要它们能够正常运算即可.
how we choose the blocks is up to us, as long as they operate.
ORi G( NAL =(8, 128) 在进行矩阵乘法时, 确保分块的形状能够相互匹配即可.
the shapes of the blocks match when doing the matrix multiplication.
ORi G( NAL =(8, 128) 例如,
so, for example,
在前面的例子中我们将矩阵 A 划分为块
in the previous case we divide ed our matrix A into blocks
SHOU2 D
WE
使得块矩阵
(即仅由这些块组成的矩阵)的形状与块矩阵 B 兼容,
so the matrix that is made up only of the blocks - is compatible with the block matrix B,
ORIGi N AO
MATRi X
2
CH
SHOU2 D
WE
从而确保这种操作能够顺利进行
so that this operation l is possible!
WE
因此, 这是我们在进行块矩阵乘法时
唯一需要注意的要求
when doing the block matr it mlti plication.
SHOU2 D
WE
块矩阵(即仅由块组成的矩阵)的形状
The shapes of the blocked matrix, so the matrix that is made only of the blocks,
CH
SHOU2 D
WE
在矩阵乘法中应相互匹配
CH
SHOU2 D
WE
除此之外, 具体如何划分并不重要.
SHOU2 D
WE
kmow
R
MATRi X 想象一下, 我们选择将查询矩阵按行划分为块
128
ORi G( NAL =(8, 12%) 这是完全可以做到的.
and we can do that.
ORi G( NAL =8, 128)
We don't have to necessarily divide also the columns.
O Ri G( NAL =8, 12%)
We can just divide the rows so that each Q is not a single row,
ORi G( NAL =(8, 12%)
Lc=(2而是由两行组成的一个组.
but it's a group of two rows.
ORi G( NAL = (8, 12%)
So Q1 is a group of the first two rows of the Q matrix, of the Q sequence.
ORi G( NAL = (8, 12%)
15. 中接下来的两行组成的组, 以此类推.
Q2 is the group of the second two rows of the Q sequence, et cetera, et cetera.
ORi G( NAL =(8, 12%)
And we do the same also for V.
ORi G( NAL =8, 12%) 对于 K, 我们不做这样的划分, 因为实际上我们要与 K的转置相乘
For K, we don't do it because we are actually going to multiply with K transpose,
BLOCu
4
Q4 所以直接在 K·的转置上进行这种细分
So we do this subdivision directly on, K transpose.
因此, 我们有被划分为若干行组的 Q So we have the Q, which has been divided into groups of rows,
ORi Gi NAL =(8, 128)
BLOCk BLOCK
ORi Gi NAL =(8, 128)
BLOCK 以及" K的转置,
128 这是一个108行8列的矩阵,
and then we have K transpose, which is a matrix that is 108 by 8,
=(4, 12 8)
BLOCK BLOCK
ORi Gi NAL =8, 128)
BLOCK 因为它是8行108列的键矩阵的转置.
because it's the transpose of the keys, which is 8 by 108.
ORi GINAL BLOCK =(4, 128)
BLOCK
ORi Gi NAL =8, 128)
BLOCK
128 我们决定将 K转置矩阵的每一列组划分为一个单独的块.
and we decide to divide each of the column group of columns of k into a single block.
ORi Gi NAL =(8, 128)
BLOCK
. 128
BLOCK 因此, k1是 K转置矩阵的前两列.
so the kl is the first two columns of k transpose.
ORi Gi NAL =8, 128)
BLOCK
k2则是 K转置矩阵中接下来的两列组成的组, 依此类推.
k2 is the second group of two columns in k transpose, etc.
=(4, 12 8)
BLOCK BLOCK 以此类推.
etc.
ORi Gi NAL =(8, 128)
BLOCK
128
BL直到 K4, 即 K转置矩阵中最后两列.
until k4, which is the last two columns in k transpose.
=(4, 128)
BLOCK BLOCK
. 128 BLOCK BLOCK 我们首先进行的操作是查询
the first operation that we do is the multiplication query
Q
Q 与键转置的乘法
ulti plied by the transpose of the keys,
Q
Q
这基本上意味着我们需要将每个查询与所有键相乘,
which basically means that we need to multiply each query with all the keys
·接着是第一个查询与所有键相乘, 以此类推.
And then the second with all the keys, et cetera, et cetera.
α现在, 每个查询并不是 Q 序列中的单一行.
Now, each ot a single row of the Q sequence.
QK QK
它是由@序列中两行组成的一个组.
It'sa
wo rows of the Q sequence.
而每个 K也不是 K转置中的单一列.
And each Ki single column of K transpose.
它是由 K转置中两列构成的一个组.
It'sa wo columns of Ktranspose.
Q, K, 1
. 伫这并不重要, 因为我们已经看到矩阵乘法:
But it doesn't matter je have seen that the matrix multiplication :
Q, KQK
如果将矩阵视为由块组成,
if we l
Write
Q, ka, ky
natrices as made up of blocks,
Q, K,
Q, K
我们只需按照常规矩阵乘法的方式进行计算即可.
we just compute it in the way when we do normal matrix multiplication.
Q, K, Q, k, Q, k Q, k]
所以我们正在将这个矩阵乘以那个矩阵, 就我们所知
so we are multiply i
trix by this matrix and, for what we know,
α. 这个矩阵由四行组成, 每行有128 个维度,
this matrix here is made up lo of fo l
ys with some dimensions, which is 128 dimensions,
Q."而那个矩阵则由多少行组成呢?
is made up of how many rows?
Qz K2
Q, K, %28 行和 4列.
2
and four columns.
QK
each row is made up of 128dimensions.
Qz K2
Don't worry, the result is correct any way.
Q, K,
QK
each row is made up of 128 dimensions.
eg anyway.
didn't draw the because it's too many to draw here,
但你需要想象每个向量都有很多维度:每个向量有128个维度.
s a lot of dimensions: 128 for each vector.
QK这重你需要想象它有128行.
And here to pretend that this is 128 rows.
当我们进行矩阵乘法时: 我们采用常规的矩阵乘法步骤,
When we do the matrix multip apply the normal matrix multiplication procedure,
QK 2
Q, K即每个输出元素.
is each output element.
Q, K,
QK
Q即每个输出元素.
Q, K,
which is each output element.
OG LAI AL
QK QK
ORi Gi NAL = (8, 8)
BLoc K =[4, 4)
because it's the
ORi Gi NAL = (8, 8)
BLoc K =[4, 4)
The first elem
ORi Gi NAL = C8, 8)
BLo CK =[4, 4)
QK,
QK?, 写那个向量的点积
with this vector here.
ORi Gi NAL =(818 )
Q, K,
ORi Gi NAL = (%, 8)
Q, K,
Q. 以第立个元素 也就是这个位置,
The second element, so this one here,
ORi Gi NAL =(83 )
Q, K,
Q. 将是这个向量与那个向量的点积.
wil be the dot product of this vector here with this vector here.
ORi Gi NAL =(813 )
Q, K,
ORi Gi NAL = (%, 8)
然而, 这并不是一个向量, 那个也不是一个向量.
However, this is. not vector and this is not a vector.
ORi GINAL =(818 )
Q, K,
Q, K 所以实际上这是矩阵乘法.
So it's actually a matrix multiplication.
ORi Gi NAL =(818 )
Q, K,
ORi Gi NAL =- (%, 8)
在这种情况下这里的这个元素不是一个标量,
Q, K, 它是一组输出矩阵的元素
It is a group of elements of the output matrix ORi GINAL =(818 )
Q, K,
Q. 因为我们正在进行块矩阵乘法.
because we are doing block matrix multiplication.
Q, K,
ORi Gi NAL = (8, 8)
Q, K, 以及它将包含多少个元素.
and how many elements it will be.
ORi Gi NAL =(83 )
我们知道, 原始的q1. 是一个2乘128的矩阵, k1是一个108乘2的矩阵
well, we know-that the original q1 is a 2 by 128, the k1 is 108 by 2,
ORi Gi NAL =(83 )
Qg ki
因此这将是输出矩阵中 组2乘2白 的元素
so it will
beagroup
of2by2
QK2
我们正在进行
g1与k1的矩阵乘法, 然后是q1与k2,
sowe are
QK,
ORi Gi NAL = (8, 8)
α. x接着是q1k与. k3, q1与k4, 以此类推,
then q1 with k3, q1 with k4, etc, etc,
ORi Gi NAL=(813)
这是第行的计算.. 接着第二行将是 Q2与所有 K的乘法,
for the first row-And then the second row will be Q 2 with all the Ks
ORi GINAL= C88)
然后是, Q3与所有 K的乘法, 以及 Q4与所有 K的乘法.
and the Q3-with all the Ks and Q 4 with all the Ks.
ORi Gi NAL= C83)
Q, K,
ORi Gi NAL = (8, 8)
正如你所见, 当我们进行矩阵乘法时,
Solas you can see, when we do matrix multiplication,
ORi Gi NAL = C818 )
Q, K,
ORi Gi NAL = (8, 8)
. K 我们并不关心底层是块、向量还是标量.
we don't even care if what is underlying is a block or a vector or a scalar.
ORi Gi NAL =(813 )
Q, K
Q. 我们與需遵循相同的步骤即可.
Q, K,
草先将黑色块矩阵的第一行
Q, K,
QK first row of the black block matrix,
气第个矩阵的第一列相乘,
[ Q, K
multiplication with the first column of the second matrix,
Q, K,
and then the first row with the second column, the first row with the third column,
Q, K,
QK et cetera, et cetera.
Q, K,
Q, K
ORi Gi NAL = C818)
BLo CK =[4, 4)
按照公式的要求,
BLOCK T et's then multiply,
ORi Gi NAL = C88)
ORi Gi NAL= B, 128) 我们需要将查询(auery)与键·(keys)的转置相乘
because the formula says that we need to multiply query with the transpose of the keys
ORGi NAL =(88)
ORi Gi NAL=(3, 128)
and then multiply by v.
ORi Gi NAL =(818 )
ORi Gi NAL =( B, 128)
BLo CK =(4, 4)
BLo CK =(4, 12g)
ORi Gi NAL =(88)
ORi Gi NAL=(3, 128)
Lo C=这些都是块矩阵:(, 124)
all of these are block matrices.
ORi Gi NAL =(818 )
ORi Gi NAL =( B, 128)
BLo CK =(4, 4)
BLo Ck =(4, 12g)
ORGi NAL =(88)
ORi Gi NAL=(3, 128) 现在, 正如你从我使用的颜色中可以看到的
now, as you can see from my using of colors,
ORi Gi NAL = C88)
ORi Gi NAL=( B, 128) 每当我提到原始矩阵时, 我使用蓝色,
every time i refer to the original matrix i use the blue color
ORi Gi NAL = C818 )
ORi Gi NAL =( B, 128)
Bb Lo CK =(4, 4)
BLo CK =(4, 124)
ORi Gi NAL = C88)
ORi Gi NAL= B, 128) 因此, 我们需要先将查询)(query)与键(keys)的转置相乘,
So we need to multiply the output of the query multiplied by the transpose of the key,
ORGi NAL -(8然后将结果与 AL-(3, 128) 相乘. 这里我们暂时跳过了softmax操作, 稍后会解释其中的原因
then by V, because we are skipping, for now, the soft max, and later we will see why.
ORi Gi NAL =(818 )
ORi Gi NAL =( B, 128)
BLo CK
=(4, 4)
BLo CK =(4, 12g)
ORi Gi NAL = C88)
ORi Gi NAL= C3, 28) 如果我们要进行这个乘法运算, 需要按照以下步骤操作.
So if we want to do this multiplication we need to do the following.
PSEUDOCODE
so it will be : this matrix is made up of blocks,
PSEUDOCODE
This should be 1 for the same reason as before PSEUDOCODE
而块矩阵乘法会忽略这一事实 器he same reason as before This should be PSEUDOCODE
直接将其视为普通矩阵进行乘法运算
he ame reason as before his should PSEUDOCODE
This should be 1 for the same reason as before PSEUDOCODE
所以, 我们先将第一行与第一列相乘, 然后将第一行与第二烈想乘
PSEUDOCODE
PSEUDOCODE
接着是第一行与第三列相乘, 以此类推.
then the third row, the first row. with the third column, etc.
PSEUDOCODE
PSEUDOCODE
那么, 在这个矩阵乘法的输出矩阵中,
So the first block of row
心第一行的第一个块是如何计算的呢?
how is going to be calculated in the output matrix of this matrix multiplication?
PSEUDO CODE
EACH BLOCK OF FOR EACH BLOCK Q :
THE OUT AITNATRIXO
FOR EACH 6 LOCK KJ
IS ACTUAL CY MADE UP
Well, it. will. be. the. first t row, so. the dot product of the first row.
ALL YE EROFS IS ACTUAL CYMA OE FOR EACH √
PSF UDO CODE EACH BLOCK OF FOREACH BCOCKQ
THE OUTATNATRIXO
FOR EACH BLOCK KJ IS ACTUAL CY MADE UP
IS ACTUAL CY MADE
实际上, 这是第一行的矩阵乘法, 但以类似于点积的方式进行,
FOR EACH IS ACTUAL CYA AOE FOR EACH
具体来说, 是与由 V
v2、v3和v4 纟 组成的第一列相乘.
let's say with. the first. co!
IS ACTUAL CYA ADE FOR EACH
PSEUDO CODE
EACH BLOCK OF FOREACH BCOCKQ
THE OUTATNATRIXO
FOR EACH BLOCK KJ IS ACTUAL CY MADE UP
此, 这个元素与v1相乘, 加上这个元素与v2相乘, 再加上这个元素与v3相乘
ISACTUALCYAADE
PSEUDO CODE
EACH BLOCK OF FOREACH BCOCKQ
TAEOUTATNATRIXO
FOR EACH BLOCK KJ IS ACTUAL CY MADE UP
最后加上这个元素写 V4相乘, 结果就是第个输出元素.
x2)x(2x128)=(2x128)
ISACTUAL CYA AOE FOR EACH
OF TWO ROWS!
ED Fo R END FGR
THE OUT PT MATRIX O EACH BLOCK OF FOREACH B COCK G
ISACTUAL CY MADE UP FOR EACH BLOCK KJ OF TWO ROW S!
ENDFOR END FOR
EACH BLDCKOF
C
FOREACHSCOCKG
The second output block will be this row with'this column.
PSEUDO CODE
THE OUT AT MATRi X O
EACH BLDCKOF
ISACTUALCY MADE UP FOR EACH 6 LOCKJ OF TWO ROWS!
END FOR
PSEUDO CODE 具体来说, 这个元素与1v1相乘,. 加上这个元素与y2. 相乘
OUTPT
PSEUDOCODE 再加上这个元素与v3相乘,. 最后加上这个元素与y4. 相乘.
OFTWOR
结果就是第二个输出块, 以此类推)
and this will-produce the second output block, etc.
等等.
END FGR etc.
OF TWO ROWS! 同样地, 第三个和第四个块的输出也是如此.
also for the third and the fourth block output.
我们来看看每个块是由什么组成的
let's look at what is each block made up of.
每个块由第一个元素组成, 即query 1与
g1相乘的结果,
so each block is made up of the the first element, so query 1 multiplied by q1,
THE OUT NTATRIXO
FOR EACH 6 LOCKKJ
ISACTUAL CY MADE UP OF TWO ROWS!
EAD FOR END FOR
THE OUT UTNATRIXO
FOR EACH6 LOCKKJ 因为这是query 与keys 的乘积
IS ACTUAL CY AAOE UP because it's the result of the query multiplied by the keys =
(x再与第二个矩阵的v1相乘得到的,
with. the vl of the second matrix,
EACH BLOCK OF FOREACH S COCK G :
plus the this element with this one, plus this element with this one,
EACH BLOCK OF FOR EACH S COCK Q THE OUTPUT MATRIX (@. y..) (plus'this element with this one r
ENDFOR
EACH BLDC KOF FOR EACH SLOCK G THE OUT AT MATRi X O FOR EACH BLOCK KJ OF TWO ROWS!
IS ACTUAL CY AADEUP
ENDFGR
生成这个输出的伪代码如下(虽然这并不完全是注意力机制,
so the pseudo code for generating this output, of this attention mechanism -
(a )因为我们跳过了soft max 步骤
which is not really attention mechanism because we skip the soft max,
但我想让你们习惯以块的思维方式来思考):
but i just want you to get into the habit of thinking in terms of blocks -
我们取每个query 块, 依次处理每个query.
is the following : so we take each query block, We go through each query and,
as you can see, let's-look at actually what this output is made up of.
它由query 1与 key 1相乘的结果再与v1 相乘组成,
it is made up of the query one multiplied by key one and the result multiplied by v1,
接着是
query 1与k3相乘的结果与v3相乘,
then the que
iery one with the k3 and the result multiplied by v3,
这基本上就是我们正在做的事情
this is basically what we are doing -
计算由块组成的这一行与这一列的点积.
is the dot product of this row with this column made up of blocks.
PSEUDOCODE
PSEUDOCODE
因此生成第一行的伪代码是:首先取query 1,
S. the Pseudo code for generating this first row is The query, is then query number one
然后依次遍历从1到4的key和value,
and then we iterate through the keys
并逐步累加结果.
and the values from one to four and we sum iteratively.
因此, 对于每个块一一 基本上是为了生成这个输出矩阵
So for each block - basically to generate this output matrix -
对于每一行, 我们会看到它是不同的query 与所有key 和value 的组合
and for each row we will see that it's a different query with all the keys and values,
然后这是query 3与所有key和value 的组合,
and then this will be the query number three with all the keys
这是query 4与所有key和value 的组合.
and values and this will be the query four with all the keys and values.
为了生成这个输出矩阵, 我们需要做以下操作:
(a k) Vu
So to generate this output matrix, we need to do :
遍历所有的 勺query, 每一行对应输出矩阵的一行,
we iterate through the queries and this will be one row of this output matrix,
然后对当前遍历的query i与第j个key 和value 进行迭代求和
and then we need to do this iterative sum of the query i that we are iterating through,
逐步累加结果,
multiplied by the jt h, k and v,
最终生成输出矩阵.
and we keep summing them iteratively and that would that will produce the output matrix,
你可以在这里看到这个过程.
or you can see here.
我 知道到目前为止我所做的内容对 Flash Attention 来说并不直接有用
I know that what I have done so far is not useless, not useful for flash attention,
但它对我们理解如何以块的方式计算这种乘积非常有帮助
but it's useful for us to get into the mindset of computing this product by blocks
因为后续我们还会结合soft max 来使用这种方法,
because later we will use it also with the soft max.
Qg K2
Qk Qk
33
34
QK
Q, Ka K
43
我知道我们自前计算的内容并不是真正的
soft ma
puted so far is not re
也不是完整的注意力机制 因为我们跳过丁
t M it's not really lan is m, because w
QK2
Qk Qk
33
34
Q, K
QK
Q, Ka K
α, K 因此; 我们需要以某种方式将其恢复进
34
how we need to restore it.
42
QK2
33
34
Q, K
QK
43
接下来的凡分钟我想大概10到20分钟 将会非常真有挑战性.
And the follow in minutes, are going to be
QK2
33
34
42
34
because I am goin
pperations that will
QK2
33
34
Q, K
QK
Q, KQK
43
Q多种指标:"莱法以及softmax 的各种变
34
and al
lotof
QK2
33
34
QK
QK
Q, Ka K
42
QK 所以可能会有些难以跟
132
33
34
be difficult to fol
low.
QK
QK
42
QK2
33
34
QK
QK
43
QK2 请不要放弃
R31
33
34
wever, don't give up P
QK
QK
43
QK2
33
34
QK
QK
43
QK2 你可以把这一部分多看几
33
34
YOU
this part twice, three tir
43
QK
QK
QK 每次观着都会有更深的理解
33
34
vill have a better t and in
QK2
33
34
Q, Ka K
42
我建议你先着到我们讲解完 Flash Attention 32
34
until we reach the fla
lashattentiol
thm
然后再回过头来重新观看这部分内容
33
34
Q, K,
43
QK2
33
34
Q, K,
QK
Q, Ka K
43
咽为当你着到 Flash 34
Attention 算法的讲解的
becau
W
reach the flash attention algor
QK2
Qk Qk
33
34
Q, K
QK
Q, KQK
42
33
34
and it
然后再重新观看这部分内容, 可以进- 一步加深你的
and th
henyo watch it to deepen your understan 1 G2
QK2
Qk Qk
33
34
Q, K
Q, KQK
42
"旁外, 我建议你拿起笔和纸, 把看到的操作步
34 Another thing that take pen and paper and write exactly the, operations
和每个块的形状都写下
33
34
write the shapes of each of these Q. KQK
QK2
Qk Qk
33
34
QK
Q, Ka K
42
尤其是这些参与矩阵乘法的元素的形状
33
34
ofthe
e
are part in this matrix multiplications
QK2
33
34
42
a这样能帮助你更好地理解正在发生的事情
34
So that you b
dwhat is happening and you bett
er remember
QK2
QKQk
33
34
QK
Q, KQK
42
也能在我提到某个特定元素或块时更容易记
34
whe
rticular element or a parti
cular
142
QK2
33
34
Q, K
Q, Ka K
42
好的在说完这段小小的激励之后, 我们
34
Okay,
aitel
is. small motivational
QK2
33
34
Q, K
42
到目前为止:我们所做的是将查询(query ) 导键
(keys ) 转置相乘
So, what we have don query multiplied by the 42
Qg K2
33
34
Q, K,
43
不过 这里的每个查询并不是查询序列中的单干
34
Howel
Q, K,
QK2
QK3 而是 一个查询块.
21
33
34
Q, K, Q, K2
42
QK2
33
34
QK,
QK
Q, Ka K
42
QK 2
33
34
ablockofrows.
QK,
QK2
42
Qg K2
33
34
在我们这个真体例子中:
Q1并不是查询序列中的单
In
our
s Q1 is not one row of the query sea uence.
QK2
33
34
rows of the query seq
uence
Q, K,
QK2
43
因为我们选择的块大小是两行为一
34 beca
usewe I as a block size a group of two
QK2
33
34
QK
Q, Ka K
42
同样; Kik transpose one ≠ 并不是 K 转置矩阵中的
And this k the K transpo
rix
QK2
33
34
QK2
Q, Ka K
42
"而是两列, 因为我们是这样选择的
32
33
34
it's two
colum
osematrix because like this.
QK2
Qk Qk
33
34
Q, K
Q, Ka K
42
QK如果你记不清宁 我们可以回头再看
33
34
emember, let'sgo
(4, (28)
128
Q y 这里我们选择 Kone, 是两列,
Here we have chosen K one is two columns √
128
ORi Gi NAL =(128, 8)
ORi Gi NAL=(8, 128)
BLOCK
BLOCK
=(4, 128)
128 而 Qone是原始查询矩阵的两行
Bock
128
ORi Gi NAL =((28, 8)
ORi Gi NAL=(8, 128)
BLOCK
BLOCK
=(4, 128)
每当我使用蓝色时, 指的是原始形状
And every time am referring to the original shape,
128
ORi Gi NAL =(88)
ORi Gi NAL= C88)
BLOCK BLOCK
而每当我使用粉色或紫色时, 无论哪种颜色
and every time I'm using the pink or violet, whatever it is,
BLOCK
ORi Gi NAL =(88)
ORi Gi NAL=(88)
BLOCK BLOCK
iv 我指的则是块矩阵.
ORi Gi NAL =(818)
BLOCK
I am referring to the block matrix. 二
所以它是原始矩阵中元素的块
So it's a block of elements of the original matrix.
BLOt K
ORi Gi NAL =(83) 好的,
ORi Gi NAL= C88)
BLOCK BLOCK Okay,
件事是将查询与键的转置相乘,
now the first thing that we have done was a query multiplied by the transpose of the keys.
BLOCK
ORi Gi NAL = C88)
ORi Gi NAL=(88)
BLOCK BLOCK
这会生成一个我们称之为-
And this produces a block matrix as output that we will call S, where each element sij BLOCK
BLOCK
BLOCK 也就是这个矩阵的 S11元素, 将是查询一与k转置一的乘积.
so the Si1 element of this matrix will be the query one with the k transpose one.
BLOCK
查询一与k转置二的乘积, S13则是查询一与k转置三的乘积, 以此类推
The S12 will be query 1 with k transpose 2, S13 will be query 1 with k transpose 3, etc,
L ·适用于所有行和所有列.
BLOCK etc, for all the rows and for all the columns.
BLOCK
接下来,=我们应该应用soft max,
BLOCK Then we should be applying the soft max
儿 L ·心因为如果你还记得公式,
BLOCK because if you remember the formula
ORi Gi NAL = (8, 8) 它是查询与键转置相乘后的sof tmax.
BLOCK
is softmax of the query multiplied by the transpose of the keys.
BLOCK
然而, 我想恢复soft max 操作, 但稍作调整, 这意味着
However, I want to restore the soft max operation, but with a twist which means
ORi Gi NAL = C88)
ORi Gi NAL = C818) 我们将应用简化版的soft max ) 并称之为soft max 星号?(softmaxstar )
that we will apply the simplified version of the soft max and we will call it soft max star,
ORi Gi NAL = (%18)
ORi Gi NAL = (8, 8)
BLo CK=(4,)
BLo CK=(4,)
ORi Gi NAL = C88)
ORi Gi NAL = C8, 8)
which is just the soft max without the normalization.
ORi Gi NAL = (8, 8)
ORi Gi NAL = (%, 8)
BLo CK
=(,)
BLo CK =(4,μ)
4
QK
lewrii
it for you what it means.
ORi Gi NAL = C88 )
QK2
34
Q. K
Q,. K
P
23
P
24
Q, K',
SOFTMAX
QK2
33
34
31
品
S=
SOFTMAX
2
24
QKQK2
32
33
34
S=
122 以, 如果你还记得, 如果我们还记得, 它就是softmax 星号
(softmax~star )
tmax
star
对于一个向量的两个元素, 我们逐个元素地应用它.
of two of a vector, we apply it element wise.
因此, 输出的一向量中的第i
so each element is modified according to the following formula :
个元素
so the ith element of the output I vector to
(我们对其应用soft max)等于输入向量中第i个元素的指数
which we are applying the soft max is equal to the exponential
··减去输人向量中的最大值,
of the ith element of the input vector minus the maximum element in the input vector
再除以一 一个归一化因子.
divided by a normalization factor.
这个归一化因子是根据以下求和公式计算得出的.
that is calculated according to this summation.
·即从j等于1到n的指数求和.
That is going from j equal to one up to n of the exponential.
of xi minus x max.
l 所以, 基本上我们是对每个元素减去×_max后取指数.
so basically, we are doing the exponential of each element minus this x max.
如果你记得清楚的话, 为什么要减去这个x_max呢?
and why are, if you remember correctly, why are we subtracting this x max b?
这是为了让指数计算在数值上稳定且可计算,
to make this exponential numerically stable, computable,
否则结果可能会爆炸式增长.
because otherwise it will explode.
因为我们将其应用于分子,
, 所以也需要在分母上进行同样的处理.
and because we are applying it to the numerator, we also need to apply to the denominator.
好的, soft max *操作与soft max 完全相同,
Okay, the soft max star operation is exactly like the soft max,
··但去掉了归一化部分,
but without the normalization part,
也就是说它只保留了
soft max 的分子部分.
which means that it's just the numerator of the soft max.
因此, 我们将根据这个公式修改应用soft max *的向量
So we will modify each element of the vector to
中的每个元素.
which we apply the soft max star according to this formula.
让我把它调整得更对齐一些, 像这样.
Let me move it more aligned, like this.
所以我们只需进行逐元素操作,
So we just do element wise operation,
·即对每个元素取指数后
that is the exponential of each element minus the maximum of the vector to
减去应用
soft max *的向量的最大值.
which we are applying soft max star.
fiu iot 好的, 那么我为什么要引人这个soft max *推 操作呢?
Okay, now why did I introduce this soft max star operation?
因为我们将把它应用到目前计算得到的矩阵上,
Because we will be applying it to the matrix that we have computed so far,
也就是这个 S矩阵
Q, K Which is this S matrix
ORi Gi NAL = C818)
BLo CK =(4, 4)
BLo CK = (4, 4)
BLOCK =4, 4) 因此, 我们对 S矩阵中的每个元素都应用了?
softmax
So we applied soft max star to each element of this S matrix,
ORi Gi NAL =(8, 8)
BLo CK =(4, 4)
BLo CK =(4,μ)
ORi Gi NAL = (%, 8) 但由于 S矩阵是块矩阵, 它的每个元素本身也是一个矩阵.
but each element of this S matrix is itself a matrix because it's a block matrix.
BLo CK =(4, 4)
BLo CK = (4, 4)
ORi Gi NAL = (%, 8)
·而这个 S矩阵的每个元素.
BLo CK =(4,)
And each element of this S matrix.
举个例子,
so, for examp
le, the element S11, is a 2x2 matrix,
举个例子, 元菱 S11是一个, 22矩阵, K
因为它来自两个矩阵的乘积
because it is co
Q, K
Q, ku
Q. k.
这两个矩阵分别是从 Q和 K
which are group of rov K
Q, k
Q, ku
Q. k.
那么, 举个例子, 这个
S11是什么呢?
Sofo
Q
S 那么, 举个例子, 这个 S11是什么呢?
So for example, this S11 is what?
S
S 让我们实际画出来看看吧
Let's draw it actually.
This S11
I will be, for example, made up of four elements.
Q我的暂且称它为 S11 的 A部分!
Let's call i t, I don't know, A of S11.
让我们选一个更好的命名方式, 比如
D吧, SC
let's choose better naming,
let'scall it, I don't know, A, B, C, an
d D
Q, 人这些都是通用的元素名称.
just the generic elements.
当我们对 S11位用
softmax
When we apply t
thesoftmaxstar to this S11,
it will result.
让我们来应用
Q 那么
Soft max *
SO let's apply the soft max star.
ak2
soft max 会生成一个矩阵人其中每个元素都 是原矩阵中 中对应元素的指数值
soft max star, it will result in a matrix that is each element,
of tmax 会生成一 其中每个元素都是原矩阵中对应元素的指数值
se
ement,
SOF TN
减去该行的最大值,
the expo i
lof each x imum for each row.
Q, k
现在我们并不知道哪 个是最大值 所以先随便选一个吧.
so let's choose one.
而这一行的最大值是 D mum for this row is D.
对块 S11
softmax 输出的第 第一个元素
The first e
r applied to this block
Q, k
将是其指数
S11
也就是a 减去
a的指数值 因为我们选择a作为这一行的最大值,
of a minus max kim um for this row.
第二个元素则是b减去
a的指数值,
alof b minus a,
因为 是这一行的最大值
Q, k
个元素是
C 减去d的指数值
h
lential
Iofcminusd,
因为
d 是底行的最大值;
hat's
bottom row,
Q, k
第二个元素则是d 减去 的指数值 这就是它的指数计算结果,
and this wi exp on that's the exponential.
Q, k
soft max 块矩阵中每个块的方式.
that'show kin this block matrix.
让我把这些内容删掉 不然它们会 一直留在我的幻灯片里
let me de
stn nt
I in my slides forever
之后我想把幻灯片分享给你们, 这样你们就可以用同样的内容了,
and later i want the bu can use my same slides
所以, 删掉, 删掉 再删掉.
SO delete
P
S=
Q, 所以, 删掉 删掉, 再删掉
so delete,
delete, delete.
SOFTMAX
QK2
Q, K Q, Ky
好的, 当我们 应用了
soft max 后
okay, after we have ap the elements in this sma
s我们会将其称为
P矩阵, 其中的每个元素, 比如p11,
we will call it the p matrix and each element, p11,
S仍然是一个2x2的块矩阵
SOFT MAX
will again be a block of two by two elements, um.
e XP
Si T SOFT MAX
因此* 就是经过
soft max *处理后的结果.
SOFT 斯,
so pl1 will be the soft max, so p11 will be the softmax star.
* Not e:
Pi= sof + max C @,
应用到s11上这里的s1是什么呢? 就是查询(query).
applied tos11.
, Where sll is what is a query?
Note : 的转置.
而p12则是
softmax* 应用到s12上的结果, 这里的 s12是什么呢? 也是查询(query )
and the p12 will be. the soft max star applied to s12, where s12 is what is a query?
1乘上k的转置2, 以此类推.
T one multiplied by k, transpose two, etc.
Thinina2x2 matrix etc, etc.
对于s的所有元素.
Thu, i,. x2 mfor all the elements of s.
好的, 现在我们已经应用了这个soft max *操作,
okay, now. that we have applied this soft max star operation,
Not e : ( P= sof +max *(@, K )
接下来按照注意力机制的公式, 我们应该做的操作是:
先对查询与键的转置相乘结果进行soft max计算,
is the soft max of the query multiplied by the transpose of the keys,
Not e : P= sof +max *(@, K )
¥ M然后将softma&的结果与值(v)相乘.
the h the result of the soft max multiplied by v.
Not e : ( P= sof +max *(@, K )
我知道我们用的不是真正的
soft max, 而是soft max*
i know that we dian'tapply the real soft max, we applied the soft max star,
Note : 他就是没有归一化的soft max.
x Not e : ( P= sof +max *(@ K )
稍后我们会看到如何弥补这个缺失的归一化步骤,
因为我们可以它放到最后再做, 这是完全可行的.
Not e : P= so+max*(@ K)
¥ N. e好的, 那么我们现在取这个 P矩阵,
I, : okay,'so we take this P matrix,
它是soft max *作用于 S矩阵的结果, 然后我们将它与 V相乘.
X'star'applied to this S matrix, and we multiply it by v.
ORi Gi NAL =(88 )
ORi Gi NAL =( B128)
BLOCK
BLo CK=(, 124)
PV,+ P2 V2+ Pa V3+ PV
W r ONG!
BLOCK HAS
我们该怎么做呢?
How do wedo it? o Nt!
PV+ P2 V2+ Pa V3+ PV
√
PV+ P2 V2+ Pa V3+ PV
W RONG!
嗯, 它是一个由矩阵块组成的块矩阵.
well, it's a block or it's a matrix made up of blocks of matrices.
PV+ P2 V2+ Pa V3+ PV
嗯, 月 所以p11 实际上不是一个标量
HAS
而是一个2x2的矩阵, 我们需要用它来乘以v.
HAS
但我们不是用原始的序列√来相乘, 而是用分块后的序列 V,
PV+ P2 V2+ Pa V3+ PV
就像之前一样, 这里的每个v不是v的一行, 而是v的一组行.
PV+ P2 V2+ Ps V+ PV
它有多少行呢?
BLOCK HAS
它是 V 的两行.
BLOCK HAS
PV,+ P2 V2+ PV3+ PV
W RONG!
GLOBAL FOR EACH ROW, BU THE 呢, 现在请完全忽略我在这里写的任何内容
TO EACH BLOCK TF ON L / WE PSEUDO COPE
HAD A WAy TO Fi X PSEUDOCODE FOR ER CHOCK G because we will use it later.
所以我们需要计算这个由块组成的矩阵的乘积.
Q Ri Gi NAL = C BLOCK
44
128 记住, 这个由块组成的矩阵实际北有四组行,
44
128
ORi Gi NAL =(88)
ORi Gi NAL=(3, 128)
BLOCK
=4, 4)
BLo CK=(h, 124)
44
128 这里的每一
44
128
ORi Gi NAL =(88)
ORi Gi NAL=(8, 128)
BLOCK
BLo CK=(, 124)
44
128
44
128
ORi Gi NAL =(88)
ORi Gi NAL=(3, 128)
BLOCK
BLo CK=(4, 124)
128 所以, 正如你所记得的,
44
128
when the algorithm f for computing the matrix multiplication is the same
44
128
ORi Gi NAL = C8, 3)
ORi Gi NAL=(3, 128)
=(4, 4)
BLo CK=(h, 124)
44
128
44
128
BLOCK except that we use blocks.
44
128
ORi Gi NAL =(8, 8)
ORi Gi NAL=(3, 128)
BLOCK
=4, 4)
BLOCK=(h, 124)
ORi Gi NAL =
ORi Gi NAL =(3, 128)
BLOCK
=4, 4)
所以, 我要做的操作是这样的.
ORi Gi NAL =3, 128
So what I am doing is, guys, the following operation.
OR GLAAL
ORi Gi NAL=3, 128)
So let's write it somewhere.
BLOCK BLo CK =(4, 124) 让我们把它写下来吧.
So let's write it somewhere.
U RONG!
ONE LOCAL TO EACH BLOCK 假设 O等于 P乘以 V, 明白吗?
PSEUDOCODE THE SOFT MAX..
ONE LOCAL TO EACH BLOCK 因此, 第一个输出行(由于它实际上不是一个行, 而是一个块行)
PSEUDOCODE THE SOFT MAX...
PSEUDOCODE FOR EACH SOCK G
W RONG!
EACH OF THE PT BLOC HAS BEEN WDi PENDENTLY CALCULATED,
SO THE MAX ELENE NT FOR EA CROW i S NOT THE
PV+ P2 V2+ Pa V3+ PVy
W RONG! 这个块矩阵的第一行与 V知
SOTHE MAX ELENE NT i S NOT THE
Pu V+ P2 V2+ Pa V3+ Pn V4
W RONG!
BLOC HAS BEEN WDi PENDENTLY CALCULATED,
SO THE MAX ELENE NT FOR EA CROW i S NOT THE
W RONG! 我们将其视为块矩阵,
BLOC HAS ○=
SO THEM AXE LENEN FOR EA CROW i S NOT THE
W RONG!
EACH OF THE P
BLOC HAS BEEN WDi PENDENTLY CALCULATED,
SO THE MAX ELENE NT FOR EA CROW i S NOT THE
PV+ P2 V2+ Pa V3+ PVy
W RONG! 因此密将等于p11乘以v1-加上p12:乘以v2
i S NOT THE
Pu V,+ P2 V2+ Pa V3+ Pn V4
W RONG! 因此它将等于p11乘以v1加上p T2乘以v2
so it will be pl1 multiplied by v1
plus p12 multiplied by'v2
BEENWDIPENDENT
WEACH BLOCK i S NOT THE THE
Pu V+ P2 V2+ Pa V3+ Pn Vy
W RONG!
BLOCH AS i S NOT THE BLOCK THE
PV+ P2 V2+ PV3+ PV
W RONG!
EACH OF THE BLOC HAS BEEN WDi PENDENTLY CALCULATED,
SO THE MAX ELENE NT FOR EA CROW i S NOT THE BLOCK THE
W EACH BLOCK GLOBAL FOR EACH ROW, BUT THE ONE LOCAL TO EACH BLOCK HAD A WAy TO Fi X
W EACH BLOCK This will produce the first output row of O,
HAD A WAY TO Fi X
W EACH BLOCK 但它实际上并不是一个行, 为它由两行组成.
FOR EACH ROW, BUT THE but it's not really a row because it's made up of two rows.
HAD A WAY TO Fi X
W EACH BLOCK GLOBAL FOR EACH ROW, BU THE So this stuff here is not one row, it is two row.
ADA WAy TO Fi X
我们可以证明这一点, 因为 P11是什么?
SO THE
And we can prove that because what is P11?
W EACH GLOBAL
P11是, 让我们把它写在这里.
P11 is, let's write it somewhere.
所以 P11是一个2x2的矩阵.
So P11 is a two by two matrix.
是的, 2x2.
Yeah, two by two.
然后我们将其与 V1相乘, V1是 V的一个由两行组成的块.
And we are multiplying it with v1, which is a block of two rows of v.
所以它是两行, 每行有128个维度.
So it is two rows by 128 dimensions.
因此, 它的尺寸是2x128.
So it is equal to two by 128.
所以这里的这个部分是2×128.
So this stuff here is two by 128.
所以这里的这个块,
So this block here,
也就是你正在计算的输出块
the output block that you are computing is a block of two rows
是我们正在计算的输出矩阵中的一个由两行组成的块.
of the output matrix that we are computing.
我知道这确实很难跟上, 因为我们涉及到了块的概念,
now this is really difficult to follow because we are involving blocks,
所以需要同时将矩阵想象成块的形式和原本的矩阵形式.
ne
d to visualize at the same time metrics as blocks and as the original metrics.
所以我强烈建议你暂停视频, 仔细思考,
at's why I highly recommend you to pause the video, think it through,
写下你需要记录的内容, 二
write down whatever you need to write down,
因为仅仅靠记忆这些形状是很难跟上的
because it's not easy to follow it just by memorizing the shapes.
你确实需要动笔写下来. 二
You actualy need to write down things
我们正在计算○矩阵的第一个输出块.
we are computing the first output block of the O matrix.
现在, 女 如果你还记得
Now, if you remember,
is N
IW EACH BLOCK GLOBAL FOR EACH ROW, B
ONE
LOCAL TO EACH BLOC
ONE
IW EACH BLOCK GLOBAL FOR EACH ROW, B
ONE
LOCAL TO EACH BLOC
is N 现在, 这个soft max 还没有应用到
GLOBAT Now,
ONE LOCAL TO EACH BLOC
is N 现在, 这个soft max还没有应用到 S
S短阵的整行上. ow, B
ONE
BLOCK vow max
Sis
基本上是为了计算这个soft max 星号
basically to compute this soft max star.
vow max
Sis
BLOCK: 我们之前所做的, 是独立于其他块来计算每个块的sof tmax 星号
what we did was to compute the soft max star at each Sir-vow rmax
_vow max SOFT MAX
-vow max So FT MA 这意味着我们用来计算每个soft max independently from the other blocks, which means that the maximum
yow ma X SOFT MAX 星号的最大值
that we are using to compute each soft max star
vow max SOFT MAX Ss exp
-vow max SOFT MA并不是 S矩阵这一行的全局最大值,
is not the global maximum for the row of this s matrix,
-vow max SOFT MAX
vow max SOFT MAX 而是每个块的局部最大值.
but the local maximum of each block.
-vow max SOFT MAX
. 这实 S 际上是错误的, 因为当我们计算soft max 时, 手 我们应用的是全局soft max and this is wrong, actually, because when we compute the soft max, we apply the soft max.
_vow max SOFT MAX
Sis
Sit-vow max SOFT MAX 我们应该使用整行的全局值.
e XP we should be using the global row.
i vow max SOFT MAX
Sis
Sit
iowmax SOFT N我想给你举个例子, 不使用分块的方式,
i want to give you an example, without using blocks,
IPv= sof + 震则可熊不太容易理解.
* Note :
because other wi vise i think it's not easy to follow.
QK
* Note : i Select
我们会有一个查询,
te ma x
QK
的转置相乘, 这会生成一个n×n的矩阵, 也就是序列长度乘以序列长度
multiplied by the transpose of the keys, this produces a matrix that. is n by n,
W
这个矩阵中的每个元素一一假设是3、4、5, 我也不确定具体有多少
so sequence by sequence, where each element of this matrix - so let's say. three, four,
--1、2、3、4、5、6, 对, 6、2、 3、4、5、6
five, I don't know how many - one, two, three, four, five, six, yeah, six, two, three,
应该是1、2、3、4、5、6.
four and five, six should be one, two, three, four, five, six. c.
好的, 这里的这个值应该是第一个查询与第一个键的点积.
okay, this one here should be the dot product of the first query, with the first um,
Select
让我用公式来说明, 因为这是查询一的转置乘以键
let me use, because query one transpose the key one.
Select
Select
这是因为, 正如我之前提到的, 当我们计算两个向量的乘积时
Select
我们总是将它们视为列向量.
we always treat them as column vectors.
Select
所以当你想要表示点积时.
so when you want to write the dot product.
Select
你不能直接对两个列向量进行相乘
you can not multiply two column vectors.
Select
你需要用一个行向量与一个列向量相乘.
you need to multiply one row vector with one column vector..
这就是为什么我们要对其中一个向量进行转置
that's why we transpose this one.
Select
Select
如果这让你感到困惑, 你也可以直接写成g1和k1
if it confuses you, you can also write ql, k1.
Select
这完全没有问题.
that's totally fine.
Select
Select
只是从数学符号的角度来看, 这样写不太规范
it's just wrong from a notation point of view.
Select
Select
总之, 第一个值将是查询向量与 的点积.
Select
第二个元素将是查询向量与 K2 的点积.
The second element will be the dot product of the query one with. the K2. x
Select
第三个元素是查询向量与 K3 的点积, 以此类推, 依此类推.
The third will be the query one with the K3, et cetera, et cetera, l et. cetera. x
因此, 这个结果是 Q1与 K1、 Q1与 K2、 Q1与 K3、 Q1与 K4的点积
So this is a Q1 with the K1、 Q1 with the K2, and the Q1 with the K3, Q1 with the, K4.
无论如何, 当我们计算soft max 时, 实际上是在这一整行中求最大值,
Anyway, When we do the soft max, we actually calculate the maximum on this entire row.
Select
然而, 我们实际在做的是块矩阵乘法.
However, what we are doing is we are actually doing a block matrix multiplication.
Select
正如你所记得的, 当我们分块计算时
And as you remember, when we do by blocks,
Select
Select
我们会将查询的行和键的行分组处理
we are grouping together rows of queries and rows of keys..
Select
在这个特定的例子中, 我们将两个查询向量归为一组, 形成一个查询块
and in this particular case, we are grouping the two queries together to. create one,
Select
同时将两个键向量归为一组, 形成一个键块.
one group of queries and two keys together to create one block of. keys. x
Select
因此, 我们需要这个块的另一行数据
so we need another row of this one.
Select
所以它是
so it's the.
Select
Select
让我选择:查询一与键一, 以及查询二与键
let me choose : query one k well, query two k one
Select
Select
这里应该是查询二与键一、查询二与键二、查询二与键三、 查询二与键四
this should be query two k one, query two k two, query two k three, query two k four,
Select
查询二与键五以及查询二与键六
query two k five and query-t we k six.
Select
Select
这里的每个块都在计算
Each of this block here is computing,
Select
SOFT MAX × Not e : ( Pi= sof +max (@, K )
Select
SOFTMAX 原矩阵中2x2的元素,
SOFT MAX Not e : P= sof +max *(@)
Select
如果我们从未应用块划分的话.
if we had never applied the blocks.
Select 口
SOFT MAX × Not e : ( P= sof + max *(a, K)
Select
因此, 它计算的是这里的这四个元素.
Tt
x So it'is computing these four elements here.
Select 口
如果我们在每个块上应用 Soft max *
And if we apply the soft max star to each of these blocks, s. cct
我们并没有使用这一行中的最大值元素.
we are not using the maximum element in this row.
Select
我们仅使用了每个块内的最大值元素,
We are only using the maximum element in each block,
Select
这意味着在与 V 度量的下游乘积中使用时,
which means that when we will use it in the downstream product with. v-metrics,
我们会累加错误的值,
we will be summing values that are wrong,
Select
因为这里的每个值都将基于一个非全局最大值的局部
because each of these values here will be based on a maximum
最大值.
that is not the global maximum for this row.
Select
这是该块的局部最大值.
It is the local maximum of this block here.
Select
Select
而这个块将使用其自身的局部最大值,
this block here will have the global, it will use the local maximum of this block here,
Select
这个块将使用其自身的局部最大值
and this block here will use the local maximum of this block here, et cetera, et cetera,
依此类推.
etcetera.
Select
Select
所以我想说的是, 当你将 P11与 V1相加时, P11可能有某个局部最大值
So what I'm trying to say is that when you sum P11 with V1, P11 may have some maximum,
Selec i
这个局部最大值与 P12 的局部最大值不同.
local maximum, that is different than from the local maximum of P12. 口
Select
P13可能有一个不同的局部最大值, 与 P11和
1 P12的局部最大值不同
And P13 may have a different maximum, Local maximum, that of p1, pi, p1 1 and p1 2.
因此, 我们需要找到一种方法来修正这里用于计算指数的最大值,
so we need to find a way to fix The maximum that was used to compute the exponential here
Selec 口
以防这里的最大值高于 P11 的局部最大值.
with the maximum found here, in case the maximum here is higher than the. one local to P11
Selec
因此, 如果我们在这里发现了一个
So if we have found, for example,
Select
比此处使用的最大值更高的最大值
here a maximum that is higher than the maximum used here here,
Selec
那么我们需要修正这个和这个,
then we need to fix this one and this one,
Select
Select
因为
soft max 中的最大值应该是整个行的最大值
because that maximum in the soft max should be the maximum for all the row,
而不是每个块的最大值
not the one belonging to each block.
Select 口
Select 口
这引导我们进入下一步
And this leads to our next ste l
p
Select 口
Select 口
如何修正这个问题?
How to fix this?
Select 口
Select 口
END FOR HOW CAW WE FIX THE PREVIOUS END FOR ITe RAti ON'S OUTPu T?
Select 口
这是一个输出块矩阵.
END FOR END FOR HOW CAN WE Select
END FOR HOW CAN WE FIX THE PREVIOUS END FOR ITe RATi ON'S OUTPUT?
Select 口
稍后我们将使用这个伪代码来调整
END FOR END FOR HOW
我们在某些块中产生的错误, 以防后续的块
ED Fo R END FOR that we have made in h some blocks in case the future blocks,
END FOR HOW CAW WE FIX THE PREVIOUS END FOR ITERATi ON'S OUTPUT?
Select 口
(如 P)有比 P或 P更好的最大值.
END FOR CAN END FOR HOW
HOW CAW WE FIX THE PREVIOUS END FOR Select 口
END FOR END So to compute this output, matrix O, we go through.
Select 口
END FOR HOW CAN WE FIX THE PREVOUS 我们选择:嗯, p11就是它本身
so, for example, tg compute the first row we choose :well, p1l is what. is is um,
让我们回到前面.
let's go back.
Select
p 11是.
m=(2x1)
P=(2x2)
pll is.
Select
128
让我也删除这个, 它已经不再需要了.
let me delete also this one, it's not needed anymore.
Select
Pu V+ P2 V2+ Pa V3+ Pa V4
W RONG!
BEEN WDi PENDENTLY CALCULATED.
Select CO THE MAX ELENE NT FOR EA CROW
BLOCK BLo CK vow rmax Select 口
BLo CK =(4, 4)
BLOCK
P11 is the softmax star c
of Q1, K1.
Select 口
BLOCK P12是 Q1和 K2的 Softmax*结果.
P12 is the softmax star
of Q1, K2.
Select 口
BLOCK BLOCK vow max Select ET MAX 口
BLOCK P13是 Q1和 K3的 Softmax*结果.
P13 is the softmaxstar
of Q1, K3.
Select 口
BLOCK BLOCK 这意味着
Select 口
BLOCK BLOCK vow max Select W
PSEUDOCODE THE OPT MAX...
Select 口
ITERATION'SOUTPUT?
DW CAN WE FIX THE PRE WOOS THE ONLi NE SOFT MAX THE IDEA IF WE CAN " AXTHE
10
Select 口
THE ONLINE SOFT MAX Select HE 口
要计算这里的这个块, 我们首先需要计算 P11 To compute this block here, we first need to compute the P11.
THE ONLINE SOFT MAX Select
THE ONIR1 N是什么? MA ×
What is P11?
Select
Well, P1l is the sof tma x star of a block of Q a
another block of. K, 口
END FGR HOW CAW WE Fi X THE ITERATi ON'S
Select
ITERATi ON'S OUT POT? 对于输出矩阵的第一行来说, 它表示的是 口
END FGR HOW CAN WE FIX THE PRE VOUS Iteration's out pot?
THE SOFT MAX Select ONLINE S 口
END Fo R END FGR
ie查询一写键一的 Softmax*结果
HOW CAW WE FIX THE PRE VOUS the soft max star of the query one with the, key one.
Select HE ONLINE?
END FGR HOW CAW WE FIX THE PRE VOUS THE SOFT MAX Select ONLINE S
EAD FOR ENDFGR
ie查询一写键三的 Softmax*结果
HOWCAWWEFIX THE PRE VOUS the soft max star of the query one with. the key three.
Select HE ONLINE? 口
END FGR HE ONLINE?
END FGR HOW CAN WE FIX THE PRE VOUS ITe RAti ON's OUt Po T?
THE SOFT MAX Select ONLINE S
EADFOR
ENDFGR
HOWCAWWEFIXTHE PREV OU S
IE遍所有键, 同时保持查询不变.
HE
END FGR HOW CAW WE FIX THE PRE VOUS ITERATi ON'S OUT POT?
THE SOFT MAX Select ONLINE S
END FOR END FGR HOW CAW WE FIX THE PREVOUS 我们需要进行 Softmax*计算得到 P11
So, to compute the first output row, we need to do the soft max star to produce P one one,
HE ONLINE S
END Fo R END FGR HOW CAN WE FIX THE PRE VOUS 即查询一与键一的 Soft max *结果, 并初始化为零
we need to do the soft max star of query one, K one, and we sum it initially to zero s,
HE
ON LINE'SOF
因为我们不知如何初始化输出, 所以就用零来初始化.
END FOR HOW CAW WE Fi X THE PRE VOUS because we don't um, we need to initialize our output somehow, 口
因为我们不知如何初始化输出, 所以就用零来初始化. 口
ENDFOR and we initialize it with zeros, then we sum the next p12, which is the query one,
ENDFOR with the k2, and then we sum the next p13, which is a query one, with. the k3, etc.
END FOR that's why we have this inner loop here, All right.
THE ONLi NE SOFT MAX Select
所以, 我们计算的这个输出是错误的, 因为正如我之前所说
so, however, this output that we are computing is wrong, because I. told you,
我们是用每个块的最大值统计量来计算 S of max * 的
END FOR EAD FOR Select ITERATION'SOUT POT?
而不是基于整个原始矩阵的每一行的最大值
END FOR EAD FOR Select ITERATION'SOUTPUT?
FOR EACH BLOCK KJ 来计算的.
EAD FOR
t is the overall row of the original matrix...
ITERATION'SOUTPUT
FOR EACH BLOCK KJ
EAD FOR END FOR HOW CAW WE FIX THE PRE VOUS Select ITe RAti On's OUt pot?
SOFT MAX WHILE IT ERAT i N Select THE SOFT MAXi S APPL
SOFT MAX WHILE IT ERATi NE
M:=max Cm-
xi-m
AROW, WE CAN ALS
BLOCKS OF ROUS
+e
Select THE SOFT MAXi S APPL.
Po= t N
SOFT MAX WHILE IT ERA Ti NE Select THE SOFT MAXi S APPL.
SOFT MAX WHILE IT ERAT i N E
m=ma x(m
AROW, WE CAN ALS
BLOCk S OF ROUS
. e +e
Select THE SOFT MAXi S APPL.
SOFT MAX WHILE IT ERAT i N 我们之前户经计算过一种算法,
A 叫做在线? Soft max.
We have comp uited before an algorithm called the'ohline soft max.
THE SOFT MAXi S APPL. 口
THE SOFt MAX i S APPLi ED 人我不确定之前是否提到过它叫在线 Soft max,∈ Ac
QN
但它确实叫做在线 Soft max.
but it's called the online soft max.
它允许我们在计算当前迭代时修正之前的迭代.
that allows to fix previous iterations when we are computing the current iteration.
基于什么原理呢?
based how?
让我们回顾一下在线 Soft Max.
Well, let's review the online soft max.
我们开始吧, 假设我们正在处理一个单一的向量.
We start, imagine we are working with one single vector.
我们有一个由n 个元素组成的向量.
So we are a vector made up of n elements.
我们做的是通过一个for 循环,
What we do is we do a for loop
选代计算到当前元素为止的最大值
where we compute iteratively the maximum up to the height element
修正之前迭代中计算的归一化因子.
and we fix the normalization factor computed in previous iteration
并在当前元素找到更大的最大值时
in case we found a better maximum at the current element.
如果这部分不清楚的话, 大家回去看看在线 S of max 的内容,
If this is not clear, guys go back and watch the online soft max,
因为这非常重要, 我们将用这种方法来修正 P1. 1
because this is very important, because this is what we are going to use to fix this P1. 1,
ITERATION'S OUTPUT?
THE ONLINE SOFT MAX
ITERATi ON'S OUTPUT? 和 P1. 2块, 以防在 P1. 3或 P1. 4等块中找到更大的最大值,
ITERATION'S OUT POT?
THE ONLINE SOFT MAX
那么, 让我们看看如何将在线 Soft max 应用到当前场景中, 以便进行计算
hee
EAD FOR END FOR HOW CAW WE FIX THE PREVIOUS ITERATi ON'SOUTPUT?
你可熊会想为什么要费这么大周折呢?
END FOR HOW CAN
EAD FOR END FOR HOW CAW WE FIX THE PRE WOUS ITe RATi ON'S OUTPUT?
我是说为什么呢?
END FOR END FOR HOW CAN WE FIX
END FOR END FOR HOW CAW WE FIX THE PRE VOUS ITe RATi ON'S OUTPu T?
FOR EACH BLOCK Q 真正的原因在于, 首先, 我们为什么要引入块矩阵乘法?
The real reason is when, first of all, why did we introduce block matrix multiplication?
END FOR END FOR
PV+ P2 V2+ Ps V3+ PV4 因为我们想要并行计算矩阵乘法
Pi T
BLOCHAS
O=
SO THE MAX ELENE NT FOR EA CROW
Pu V+ P2 V2+ Ps V3+ PV
EACH OF THE P:
BLOC HAS BEEN WDI PENDENTLY CALCULATED SO THE MAX ELENE NT FOR EA CROW
Pu V+ P2 V2+ Pa V3+ Pa V4 你可以认为 由于 P11等块彼此独立
BLOCK HAS O=
SOTHE MAX ELENE NT FOR EAC
你可以认为, 由于 P11等块彼此独立,
So you can think that each of these P11, because they are independent from each other CX ) RONG
并且每个块都使用各自块内的最大值,
and because each of them are using the maximum belonging to each block,
+ V
XRONG!
() RONG!
因此它们可以独立计算.
they can be computed independently from each other.
U RONG
( RONG!
然而, 我们需要以某种方式聚合它们的值.
Then, however, we need to somehow aggregate their value.
J RONG
( U ) RONG!
为了聚合这些值,
And to aggregate their value,
+ V
( U RONG!
我们需要修正那些独立计算的值,
we need to fix the Values that have been calculated independently X RONG!
因为在独立计算时我们没有这样做.
because we didn't when computing values independently.
+ V
URONG!
我们没有全局视角, 只有局部视角.
We don't have a global view, We have a local view.
+ V
X) RONG!
( X ) RONG!
因此, 我们计算局部块, 比如 P1, 1、 P1, 2、 P1, 3等.
so we compute a local blocks, a pl, l, pl, 2, pl, 3, etc.
RONG
等等.
etc.
( X ) RONG!
( X ) RONG!
然后, 当我们需要聚合这些值时, 必须对它们进行修正.
and then, when we aggregate these values, we need to fix them.
+ V
CX) RONG
( X ) RONG!
正因如此, 我们才尝试构建这套系统,
so That's why we are trying to come up with this system + V
XRONG!
FOR EACH B COCK Q FOR EACH BLOCK K 用于修正那些独立计算得出的值.
END FGOR HOW CAN WE FIX THE PRE
那么, 如何解决这个问题呢?
So, how to fix this?
让我们来看一下接下来的算法! c DE FOR EACH
S,= Q, k
We utl do it et heend.
Q,=vowsouexp( S,-m)+ Q-exr(mo-m)
P=exp( S-m,)
O, = diog(exp(m-m) O. + PV
S= Qk
Weutldo itcthe end. 首先, 正如我之前提到的
, 这里的○块是一个包含两行的块.
END FGR END Fo R HOW CAW WE FIX THE PREVIOUS IT e RAti O on's Out pot?
THE SOFT MAX
END FG R
EADFOR HOW CAW WE FIX THE PRE VOUS where each row is made up of 128 dimensions,
TAE SOFT MAX
这一点我们之 前通过检查 P1、1和的维度已经了解, 即 P1、1与 V1相乘的结果表明
ENDFOR
and we have seen that before by checking the dimensions of P1, 1 and V1, the result of p1,
THE
ONLLNE SOFT MAX
END FG R
EADFo R EFIXTHEPREVOUS 对于海茶输出块, 我们需要处理两个最大值
1, V1, which means that for each output block we need to take care of two maximums THE ONLLNE SO ET MAX
END FGR END Fo R HOW CAW WE FIX THE PRE WOUS ITERATi ON'S OUTPUT?
THE SOFT MAX
END FGR END Fo R HOW CAW WE FIX _ THE PREVIOUS and two normalization factors.
THE
END FGR END Fo R HOW CAW WE FIX THE PRE WOUS ITERATi ON'S OUTPUT?
THE SOFT MAX
END FG R
EADFOR FIX THE PRE VOUS so up to now I didn't use the normalization factor.
ONLINE SO ET MAX
我们说过, 芷在应用的是 Soft max *, 民
EAD Fo R FLX THE PRE WOUS 即未经归一化的 Soft max,
we said that we are applying soft max star, which is the soft max,
THE
END FG R
EADFOR without the normalization, but eventually we will need to compute this normalization.
THE
END FGR END Fo R HOW CAW WE FIX THE PRE WOUS ITe RATi ON'S OUTPUT?
THE SOFT MAX
EAD Fo R 因此,
so We want to create an algorithm that fixes the maximum use to compute each of this P11 THE
FOR EACH B COCK Q 最后再应用这个归一化因子.
FOR EACH BLOCK K
xa: K
EAD FOR THE PRE VOUS
FOR EACH B COCK Q 真体实现方法如下所述.
FOR EACH BLOCK KJ And the way we will do it is as follows.
EAD FOR
XTHE PREV IONS
END FOR HOW CAN WE FIX THE PRE WOOS ITERATION'S OUT POT?
AAA X
我们首先将最大值初始化为负无穷
We s Star f with initializing the maximum to minus infinity,
S,= Q, k
We ute doit at he end.
Q. =vow sou ex( S、 -(m)7 + Q·exr(mo-m.)
P=exp( S-m.)
O, = drag Cexp(m-m) O. + PV
S,= Q, k
We utl do it athe end.
Q,=vowsouexp( S、-m] 每处理一行就初始化一个.
P=exp( S-m,)
S,= Q, k.
Weufl doit at he end.
Q,=vow sou exp( S, 由于我们的输出块由两行组成,
P=exp( S-m
S,= Q, k
We utl do it at He end.
:因此需要为顶行和底行各设一个最大值
S,= Q, k
We utl do it ct he end.
Q, = vow suμ ex( S, -cm)7 + Qo·exr(m-m)
P=exp( S-m,)
O, = drag Cexp(m-m) O. + PV
S,= Q, k.
We wtl do it at he end.
Q, = vow sou [exp( S, 同时初始化归一化因子为0,
Pu = exp( S-m,)
S,= Q, k.
We utl do it at he end.
l,=因为自前尚未进行任何求和操作, 以及输出.
We utl do it athe end. 我们将输出初始化为全零, 因为目前还没有向这个输出添加任何内容
S,= Q, k
We utl do it at He end.
Q. =vow sou exr( S、 -m)7 + Q·exr(mo-m.)
P=exp( S-m.)
O, = drag Cexp(m-m) O. + PV
S,= Q, k
We ute do it at he end. 需要先进行计算.
P=exp( S
因此, 针对这里的输出块, 即这个输出块.
IF ONL / WE HAD A WAy TO Fi X PSEUDOCODE FOR EACH B COCK Q here.
√
IF ONLY WE HAD A WAY TO Fi X 我们需要遍历所有键来生成这些值:p1. 1mp12、p13、p14,
we :need ·to go through all the keys to produce this : p11, p12, p13, p14,
PSEUDc而查询则是第一个查询即查询块·1.
while ·the query is the. query number one, the query block number one.
第一步, 我们计算第个块p1i的最大值,
Kene..
so the first step that we do is we compute the maximum of the first block, p11,
即行最大值, 也就是块q1、k1中每行的最大值.
which is the row max, so the maximum for each row of the block, ql, k1.
这里实际上指的是s1, 而不是p11.
this is not pil, it's sl.
抱歉大家, 这里应该是s11.
sorry guys, this is s11.
因此, 我们计算它的最大值, 并称之为s1, 正如你在这里看到的
so we compute the maximum of this one and we call it actually sl, as you can see here.
接下来, 我们可以计算p11, 也就是 Softmax*
um, then we can calculate p11, which is the soft max star,
它是查询1与k1的指数乘积,
which is the exponential of the query, multiple query : 1, k1,
即s1减去局部组中的最大值s1, 然后将结果添加到我们的输出中
so s1 minus the maximum in the local group, s1, and we add it to our output.
目前, 输出初始化为零.
For now, the output is initialized with zero.
所以现在, 暂时忽略这部分.
So for now, ignore this part here.
稍后我会解释这部分.
I will explain it later.
因此, 目前 O1应该只等于 P11乘以 V1.
So for now, 0l should be equal only to P11 V1.
现在, 在第二步中, 我们 可能会在局部组 S12中找到更好的最大值, 这里的 S12就是新的最大值
Now, at the step number two, we may find in the local group S12, so this one is S12.
Pn = exp( S2 - m2)
O, = diag Cexp(m,-m) O, + PV2
Q2 =vow su [exp( S2 - m)7 + Q,· exp (m, -mz)
P,:我们需要更新顶部行和底部行的最大值.
we, may find a better maximum for the top row and the bottom row.
Pn = exp( S2 - m2)
O, = diag Cexp(m,-m) O, + PV2
P = exp( S2-m2) 这个最大值是 M2,
Pn = exp ( S2 - m2)
O, = diag Cexp(m,-m2) O, + Pi2 V2
Q2 = vow su [exp( S2 - m)7 + Q,· ex (m, -mz)
P,:x(它可能比之前这两个行的最大值更好,
which may be y better'than the'previous maximum for each of these two row,
P2 = exp( S2 - m2)
O, = dcag Cexp(m,-m2) O, + Pi2 V2
P = exp( S2 - m2) 但也可能不是.
我们需要找到一种方法, 在它更好的情况下进行调整,
Q2 = vow su [exp( S2 - m)7 + Q,· ex (m, -mz)
P,-而在它没有更好的情况下不做任何改动.
P n = exp( S2 - m2)
O, = diag Cexp(m,-m2) O, + PV2
P2=exp( S2-m2) 我们是这样做的.
drag Cexe(m, -m) O, + P2 V2 因此, 我们计算当前局部行查询2的新最大值
STEF
O, = dcag Cexp(m,-m) O, + Pi2 V2
LAST S
STEF AND So
N UNTi L HE
我们计算了p12, 它是s2的 Softmax*, 即s2减去局部最大值m2
we
0, = drag Cexp(m,-m) O, + P2 V2
LAST S
STEA AND SO ON UNTi L THE
O, = diag Cexp(m-m) O, + P2 V2 然后我们需要将其添加到输出中
STEL
O, = dlag Cexp(m,-m) O, + PV2
LAST S
STEF
AND So
UNTL THE
dia& Cexp(m,-m) O, + P2 V2
O. = dlag Cexp(m,-m) O, + Pi V2
LAST S
STEF AND So
ON UNTi L THE
diag Cexp(m,-m) O, + P22 那么如何修正o1呢? 它只使用了s1的局部最大值
S2 = Q, k 那么何修正1"呢? 它只使用了s1的局部最大值.
So how tofix the ol"which only used the maximum that was local to s1?
00+
S2 = Q, k
2 =vow suμ [exp( S2-m2)7 + Q,·exr(m, -m2)
P2 = exp( S2-m2)
S2 = Q, k 我们知道-可以通过使用指数函数来修正,
AND SO ON UNTi L THE LAST STEF HE NOR MALi ZATi ON FACt e APPLY
LAST STEF 没有归化 A becaus without the normalization,
AND So UML L. THE LAST S 因为我们应用的是 Soft max LATi ON FACT c APPLY T +because we are applying soft max sta
AND SO ON UNTi L THE LAST STEF NOR MALi ZATi ON FA C tc
APPLY THE
AND UNTi L THE LAST S
FA C tc
AND SO ON UNTi L THE LAST STEF HE NORMALi ZATi ON FACt c APPLY
UNTi L THE LAST STEf
2 进行乘法操作· Cc
A So basically we are saying that we multiply o1, which is a matrix.
AND SO ON UNTi L THE LAST STEF NORMALi ZATi ON FAct c APPLY THE
AND UNTi L. THE LAST S
FA C Tc
APPLY T + So let me show you what is this matri
STEA AND S 那么让我展示卡这个矩阵是什么.
FACt c APPLY
AND SO ON UNTi L THE LAST STEF APPLY THE NORMALi ZATi ON FACt c
P 2 = exp( S2-m2)
drag Cexp(m,-m) O, + Pi2 V2
LAST
STEA
P2=exp( S2
is a matrix made up of two rows.
LAST
STEA
P2 = exp( S2-m2)
diag Cexp(m,-m) O, + PV2
LAST
STEA
exp( S2-m2)
so, as you can see, here i have the shape of o1.
STEA
LAST
=vowsou ex( S2-m3+ Q·exr(m,-m2)
它是一个2行128列的矩阵.
it's a 2 by 128 matrix.
m2= max(rowmax( Qk22), m,)
S2 = Q, k
=vowsou ex( S2-m2)7+ Q·exp(m,-m2)
> M=max(rowmax 这是第一行.
S2 = Q, k
row max >m所以,
01_1, 01_2, 依此类推, 直到o1_128.
so oh one, one, oh'one, twg blah,. blah, until oh one one hundred twenty eight.
exp (
Mm2= max(rowmax( Qk22), m,)
S2 = Q, k
=vow sou exp( S2-m2)7 + Q,·exp(m, -m2)
>然后是02
21, 2_2, 依此类推, 直到02_128,
S2=
then oh two, one y
h two. two, blah, blah, and oh two,
exp(
>2
m= max(row max (
S2 = Q, k
vowsuu exp( S2-m)7 + Q,·exr(m, -m2)
需要修正这个值.
S2 = Q, k
exp(
S2 = Q, k
vow suu exp( S2-m)7 + Q,·exr(m, -m2)
如何修正呢?
S2 = Q, k
S2 = Q, k
vow suu exp( S2-m)7 + Q,·exr(m, -m2)
我们基举上就是使用之前在线 Soft max 中用过的指数函数
,=vow su μ
exp (
=vow soμ[exp( S2-m2) + Q,· exp(m, -m2)
P2=exp( S2-m2)
0, = diag Cexp(m,-m) O, + PV2
=vow sou [exp( S2-m2) + Q,· exp(m, -m2) 来修正.
P2=exp( S2-m2)
O, = drog (exp(m, -m) O, have, seen before.
=vow soμ[exp( S2-m2) + Q,· exp(m, -m2)
P2=exp( S2-m2)
0, = diag Cexp(m,-m) O, + P2 V2
OWSUM
exp( S2 -(m2)7 + Q,· exp(m, -m2) 因此, 如果我们用一个对角矩阵乘以这个矩阵, 对角矩阵的构造如下
Pn = exp( S2-m2)
= diag Cexp(m, -m) O, + P2 V2
UNTi L THE LAST S
STEP
exp( S2-m2
dae它是由两个元素组成的对角矩阵
STEP
P2 = exp( S2-m2) 因为m1减去·m2"的指数结果会是一个包含两个元素的向量
UNTL THE
P2 = exp( S2 - m2)
drag Cexp(m-m) O, + PV2
UNTi L THE LAST S
STEP
Pn=exp( S2-m2) 而元素级别的指数运算会生成另一个包含两个元素的向量.
UNTL. THE
Pn = exp ( S2 - m2)
= diag Cexp(m,-m) O, + PV2
UNTi L THE LAST S
STEP
exp( S2-m2)
da这里的 Diag基本上表示一个对角矩阵
STEP
UNTLTHE
Pn = exp( S2 - m2)
=diag Cexp(m,-m) O, + PV2
UNTi L THE LAST S
STEP
exp( S2-m2)
dcy(exr(m-m)其对角线上的元素
UNTLTHE
exp( S2-m2
d是我们应用 Diag操作的向量中的元素.
STEP
P2 = exp( S2 - m2)
=diag Cexp(m,-m) O, + PV2
UNTi L THE LAST S
STEP
Pn= exp( S2-m2) 也就是说这里的值将是m1第一个元素的指数.
UNTLTHE
exel So let me show you how to Write'it : exponential of m1 minus m2 minus m2,
所以第一个元素, 我们在这里称它为1.
so the first element, so let's call it one here.
这里有一个零, 这里也是零,
here is a zero, here will be zero,
让我们删除这个, 然后在这里写下另一个, 即m1减去m2的指数
and let's delete this one and we write another one here, exponential ml minus m2.
但这是向量的第二个元素.
but the second element of this vector.
> M2 = max(rowmax( Qk2), m,)
S2 = Q, k 简单来说, 这里的
S2 = Q, k
diag表示将向量分布到一个n xn的矩阵上,
diag means basically
take, the vector and distribute it over a n by n matrix,
O=rag(exp(m
S2 = Q, k
n是应用该操作的向量的大小,
wherenis. th
e size of the vector to which it is applied.
S 2 = Q, k
Q2 = vow su exp( S2-m2)7 + Q,-exr(m, -mz)
P=exp( S2-m2)
O, = drag Cexp(m,-m) O, + PV2
S2 = Q, k
Q2 = vow sou [exp( 矩阵中所有其他元素都应为零.
S2 = Q, k
Pn=exp( S2-m2)
K
0, = drag Cexp(m,-m) O, + PV2
S2 = Q, k
(m-m2) 这就是diag白 的含义.
Pn=exp( S2-m2)
S 2 = Q, k
Q2 = vow su [ex P( S2-m2) + Q,-exr(m,-mz)
Pn=exp( S2-m2)
O, = drag Cexp(m,-m) O, + PV2
S2 = Q, k
(-m)
Pn=exp( S2
S2 = Q, k
Pn=exp( S2-m2)
0, = drag Cexp(m,-m) O, + PV2
S2= Q, k 我们会看到这个乘法的输出将使用这个指数修正顶行的每个元素,
-m
S2 = Q, k
Q2 =vowsou exp( S2-m3) + Q,-exr(m,-m2)
Pn=exp( S2-m2)
O, = drag Cexp(m,-m) O, + PV2
S2 = Q, k 并使用这个指数修正底行的每个元素,
itial. and the bottom row with this exponential,
S 2 = Q, k
2=vowsouexp( S2-m)7+ Q·exp(m-m2)
Pn=exp( S2-m2)
0, = drog Cexp(m,-m) O, + PV2
S2= Q, k 这基本上会抵消之前迭代中计算的 M1,
out this M1 that was computed in the previous iteration
S2 = Q, k 并在·块矩阵的每个元素中引入
that we have computed in the current iteration
S2 = Q, k
Q2 = vow sou [ex P( S2-m2) + Q,-exr(m,-mz )
Pn=exp( S2-m2)
O, = drog Cexp(m,-m) O, + PV2
S2 = Q, k
Q2 = vow sou [ex P ( S2 - m 当前迭代中计算的 M2.
Pn=exp(
these elements in this O block matrix.
S2 = Q, k
Q2 = vow sou [ex P( S2-m2) + Q,-exr(m,-mz)
Pn=exp( S2-m2)
O, = drag Cexp(m,-m2) O, + PV2
AND SO ON UNTi L THE LAST STEP. THEN, WE
好的, 所以这个输出将是:这个元素将乘以这个,
2 ATON
所以它将用这里的这个因子修正011, 而021不会被修正
so'it will fix 011 with this factor here and 021 will not be fixed by -
"*将乘以零, 所以它不会对第一个输出元素有贡献.
o(exp(m,
所以这里的这个元素将只依赖于由 M1减去 M2的指数修正的011
So tihis élement here will only depend on O11 fixed by the exponential of M1 minus M2,
wsouexp( S2-m)+ Q·exr(m 但这是向量的第一个元素.
x P( S2-m2
but the first element of this vector.
然后也将由这里的这个指数修正, 而不是由这个修正.
[* S -"第一行的所有维度都将由这个指数修正.
L而这里的第二行的所有维度将由这里的这个指数修正
And afl'the dim
ensigns of the second row here will be fixed by this exponential here,
"[ S-这个棕量, 它是向量exp的第二个元素.
P( S2
this scalar here, which is the second element of the vector exp.
xp( S2-m
Okay, it was really challenging, this one.
所以我们正在做的是计算 P12,
P( S2
So what we are doing is we compute P12
ag(exp(m-m) O,+ PV
STEP 5 并通过乘以这里的这个矩阵,
and we fix all the elements in P1 by multiplying by this matrix here,
STEP 5 乘以这里的这个因子, 矩阵因子, 来修正 P1中的所有元素.
by multiplying by this factor here, matrix factor here.
STEP 5
And
STEP 5 当我们计算第三步时, 我们将修正第二步, 依此类推, 依此类推.
when we will compute step three, we will fix step two, et cetera, et cetera, et cetera.
STEP 5 现在让我们来谈谈归一化因子, 因为到目前为止我们一直忽略了它
Now let's talk about the normalization factor because for now we have been ignoring it.
Pn=exp( S2-m2)
O. = dlog Cexp(m,-m) O, + PV2
LAST STEP. THE N, W
AND SO ON UNTi L THE
归一化因子是我们在计算这些最大值时可以同时计算的
S2-m
AND SOON
因为它在我们之前看到的在线算法的伪代码中提供了,
because it is provided in the pseudo code of the online algorithm
用于 soft max.
that we have seen before for the soft max.
所以在计算最大值的同时,
So while computing the maximum,
我们实际上可以通过修正前一次迭代的归一化因子
we can actually compute the normalization factor
来计算归一化因子.
by fixing the normalization factor of the previous iteration.
M.= 这正是我们在这里所做的.
And this is exactly what we are doing here.
所以在第一次迭代中, 我们使用局部最大值计算了归一化因子.
So at the first iteration, we computed the normalization factor using the local maximum.
S2= Q, k
STEP 在第二次迭代中, 你现在可以忽略这个
S2 = Q, k
S因为我们没有用任何东西修正 LO, 因为 LO将是0.
S2 = Q, k
S2= Q, k
Q2=vow souexp( S-m) + Q-ex(m, -m)
Pn=exp( S2-m2)
O =dliag Cexp(m,-m2) O, + PV2
所以我们基本上只是在这里计算这个求和.
So-we are just'basically, we are just computing this summation here.
S2= Q, k
Pn=exp( S2-m2)
So Lo will be 0.
0 = dliag Cexp(m,-m) O, + PV2
=*(-以这里的这个因子将是0.
S2= Q, k
P2=
exp( S2-m2)
So this factor here will be 0.
S2 = Q, k
Q2 =vow sou [exp( S2-m)7 + Q-ex(m, -m)
Pn=exp( S2-m2)
0. = dliag Cexp(m,-m2) O, + PV2
AND
LAST STEP. THEN, W
A LAST _ STEP. THEN, W
AND 正是用来修正最大值的那个指数, 即 P1lo R
A pit's exactly the same exponential that fixes!
the maximum, the p11.
AND SO ON UNTi L THE LAST STEP. THEN, W APPLY THE ""
NOR m ALi ZATi ON FA Ct OR.
AND So e
5
LAST STEP. THE, W
AND
SOON 减去当前对最大值的估计,
AST. STEP. THEN, W
AND :
AND SO ON UNTi L THE LAST STEP. THEN, W APPLY THE ""
NOR MALi ZATi ON FA Ctor.
AND SOO 然启我们继续执行这一过程.
LAST STEP. THEN, W FActor.
APPLY THE 2 And we keep doing this job.
AND SO ON UNTi L THE LAST STEP. THEN, W APPLY THE NOR MALiz ATi ON FA ctor.
LAST STEP. THEN, W
AND S
O, = diag Cexp(m,-m) O, + PV2 最终, 我们将得到这个块的正确输出 F. THE, W
最终, 我们将得到这个块的正确输出,
At the end, we will obtain a correct output for this block here,
EAD FOR HOW CAN WE FIX THE PRE VOUS END FGR ITERATi ON'S OUTPUT?
END FGR
EAD FOR HOW CAW WE FIX THE PRE VOUS END FGR ITERATi ON'S OUTPUT?
EAD FOR END FGR Ite r Ation'How to apply the normalization?
归一化就是我们需要用归一化因子
END FG R
EADFOR HOW
EAD Fo R END FGR
EAD Fo R HOW CAW WE FIX THE PRE VOUS END FGR
但由于我们在遍历这四个循环时
END FOR EAD FOR HOW CAA
STEP 5 也在计算归一化因子.
we also calculate the normalization factor.
STEP 5 我们不断累积它, 直到迭代结束,
we keep accumulating it until we reach the end of the iteration
然后应用归一化因子.
and then we apply the normalization factor.
因此, 我们取最后的输出, 只需将其除以 L4,
so we take the last output and we just divide it by L4,
即第四次迭代计算出的归一化因子,
which is the normalization factor calculated as the fourth iteration,
这将修正 Soft max.
and that will fix the soft max.
Algorith i
好了, 各位.
All right guys.
Algorith i
Algorith i
现在我们已经推导出如何分块计算注意力模块输出的算法,
so now that we have derived the algorithm of how to compute this output
同时也在每个块中独立修正了 So max,
of the attention block-wise while also fixing the soft max,
Algorith i
我们知道
which is done independently in each single block,
Algor
Algorithm
归一化是在最后完成的.
we know that the normalization is done at the end.
Algorithm
我还想证明一下这一点.
I want to also prove it.
Algorithm
Algorithm
因此, 当我们引l 入这种在线计算 Soft max 的
So what we've done when we introduced this algorithm
算法时,
that computes the soft max in an online way,
AND SO ON UNTi L THE LAST STEP. THEN, WE APPLY THE
O = diog Cexp(m,-m) O, + PV2
AND SO ON UNTi L THE LAST STEP. THEN, WE APPLY THE NOR MALi ZATi ON FA cto R.
我们通过归纳法证明了该算法的正确性.
Pn=exp( S2
因此, 在这个算法结束时,
( M=max(rowmax( QK2), m
So at the end of this algorithm,
最后一次送代的 L 实际上就是
this L. of. the last. iteration will actually be the normalization factor
O, = diag Cexp(m-m) O. + PV 我们可以用来得到 Softmax的归一化因子.
that we can apply to get the soft max.
因此, 在在线计算输出时,
So we don't apply the normalization while computing this output in an online way,
我们不会在通过将查询与所有键块相乘的迭代过程中应用归一化.
iteratively way by multiplying the query with all the blocks of keys.
我们在这四次送代结束时应用它.
We apply it at the end of this four iteration.
在这四次迭代结束时, 我们将得到最后的输出.
And at the end of this four iteration, we will have the last output.
我们还知道, 最后的 L将包含我们需要应用于每行的确切归一化因子
And we also know that the last L will contain the exact normalization factor
因为这个0 是一个输出行块,
that we need to apply to each row, because this O is a block of output rows which is,
如果你还记得注意力机制
if you remember from the attention mechanism,
注意力的输出与输入查询向量的形状相同.
the output of the attention has the same shape as the input query vector.
这是一个标记序列.
which is a sequence of tokens.
因此, 这个○是一个我们需要应用归一化的标记序列.
So this O is a sequence of tokens that we need to apply the normalization to.
我们知道正确的因子是 L4.
And we know that the correct factor is L4.
那么, 让我们来证明这个简单的公式.
So let's prove this simple formula.
L4是一个向量, 它包含的元素数量与 O4中的行数相同.
L4 is a vector that contains as many elements as there are rowsin 04.
所以在这个口行块中.
So in this O block of rows.
假设它包含两行, 就像我目前描述的算法中那样,
suppose that it contains two rows, like in the algorithm that I have described so far,
我们假装将两行查询
in which we pretend that we are grouping two rows of queries
与两列键组合在一起.
with two columns of keys together.
因此, 输出 O, 即块○将包含两行输出,
So the output O, the block O will contain two rows of the output.
所以, 我们在这个 L4向量中会有两个归一化因子.
Sowe will have two normalization factor in this L4 vector here.
我们通过这个公式所做的是, 取这个 L4 向量
What we are doing with this formula is we are taking this L4 vector
并用它创建一个对角矩阵,
and we are creating a diagonal matrix with it
然后计算这个对角矩阵的逆矩阵.
and then we are computing the inverse of this diagonal matrix.
因此, L4是一个包含两个归一化因子的向量.
So L4 is a vector that contains two normalization factors.
因此, 它是 L, 我不确定, 让我们称之为 L4元素1和 L4元素2.
So i t's L, I don't know, let's call it L4 element 1 and L4 element 2.
这就是我们的 L4.
This is our L4.
向量.
vector.
然后我们有04.
then we have 04.
04是一个矩阵.
04 is a matrix.
从形状可以看出, 它是一个2乘128-的矩阵.
as you can see from the shape, is a 2 by 128 matrix.
所以○是一一实际上我们复制它吧, 哦不, 还是别复制了.
So o is- let's copy it actually, oh no, let's not copy it.
04是一个两行128列的矩阵,
o4 is a matrix that is two rows with 128 elements,
第一行有 128个元素, 第二行也有 128个元素.
so the first row with 128 dimensions and the second row with 128 dimensions.
我们对14 做的第一件事是
the first thing that we are doing with this I4 is
将其转换为一个对角矩阵,
we are converting it into a diagonal matrix,
它将是一个2乘2的对角矩阵, 因为它包含两个元素.
which will be a diagonal matrix, 2 by 2, because it contains two elements.
Algorithm 1 FLAs HATr ENTIo N-2forward pass Require : Matrices Q, K. V e RNxd in HBM, block sizes Be, Br.
因此它将变成类似这样的形式.
Algorithm 1 FLash Arrensoit will become something like this.
Require : Matrices Q, K. V e RNxd in HBM, block sizes Be, Br.
Algorithm 1 FLAs HATr ENTIo N-2forward pass Require : Matrices Q. K, V e RNxd in HBM, block sizes Be, Br.
所以它将是14, 14的第一个元素, 然后是0, 接着是0,
Algorithm 1 soitwilbe 14ythe first element of I4, 0 and then 0, 14,
Require : Matrices Q, K, V e RNxd in HBM, block sizes Be, Br.
Algorithm 1 FLAs HATr ENTIo N-2 forward pass Require : Matrices Q, K, V e RNxd in HBM, block sizes Be, Br.
以及这个向量的第二个元素14.
Algorithm 1 Flasn Arrerrther second element of this vector.
Require : Matrices Q, K, V e RNxd in HBM, block sizes Be, Br.
Algorithm 1 FLAs HATr ENTIo N-2 forward pass Require : Matrices Q. K. V e RNxd in HBM, block sizes Be, Br.
然后我们计算这个矩阵的逆矩阵.
Algorithm 1 Fithen-we are computing the inverse of this matrix.
Require : Matrices Q. K. V e RNxd in HBM, block sizes Be, Br.
Algorithm 1 FLAs HATr ENTION-2forward pass Require : Matrices Q. K. V e RNxd in HBM, block sizes Be, Br.
对角矩阵的逆矩阵就是将对角
Algorith r the inverse r of a diagonal matrix is just the diagonal matrix,
Require : Matrices Q, K, V e RNxd in HBM, block sizes Be, Br.
线上的每个元素取倒数后得到的对角矩阵.
Algor i with each element on the diagonal that becomes its reciprocal.
Require : Matrices Q, K. V e RNxd in HBM, block sizes Be, Br.
Algorithm 1 FLAs HATr ENTIo N-2forward pass Require : Matrices Q, K. V e RNxd in HBM, block sizes Be, Br.
1: Divide Qinto T,=
B
K...., KT. and V...., VT, of size B. xd each.
into T, blocks Li,..., LT, of size B, each.
3:for 1≤i≤ T, do Fo REACH BLOK Load Q ;from HBM to on-chip S RAM.
of size B. xdeach
3: this is from linear algebra x it's not i'm making it, i'm making this up.
into /
. blocks Z
otslze B. each
4:
Load Q:from HBM toon-chip S RAM.
K.
2: Div
so therinverse of this-matrix here is equal to uh, the same uh diagonal matrix,
into 7
ofsize B
4:
Load Q:from HBM to on-chip S RAM.
1: Divide Q into T,=
[blocks Q.... Qz, of size B, Xd each, and divide K, V in to T = 「blocks
B
K..., K. and V...., VT 恒每个元素是小4的倒数gh, anddivide the logsumexp L
2: Divide the output O e R
into T, blocks Li,..., LT, of size B, each.
3: for 1<i< T, do Forbut where each element is one 4:
Load Q;from HBMto on-chip SRAM.
1: Divide Q into T,=
B
K...., K. and V...., VT., of size Bxd each.
into T, blocks L..., LT, of size B, each.
3:for 1≤i≤ T, do Fo REACH : BLOCK 4:
Load Q;from HBM toon-chip SRAM.
即14的第一个元素的倒数0, 0, 以及14的第二个元素的倒数.
Kr. and
ofsize Bxdeach.
over l4, the first element of l40s0 and 1 over I4, the second element of I4.
into
blocks L
4:
Load Q:from HBM to on-chip S RAM.
1: Divide Qinto T,=
B
K...., K. and V...., V, of size Bx d each.
into T, blocks Li,..., LT, of size B, each.
3:for 1≤i≤ T, do Fo REACH : BLOCK 4:
Load Q:from HBM to on-chip SRAM.
1: Divide Qinto T,=
K...., KT. and V... 然后我们将这些内容进行相乘dide thologsumxp L
2: Divide the output O into T, blocks L,...
ofsize B, each
3: for 1 <i< T, do and then we are multiplying this stuff here.
4:
Load Q;from HBM to on-chip SRAM.
1: Divide Q into T,=
Br
K...., KT. and V...., VT, of size BXd each into T, blocks Li,..., LT, of size B, each.
3: for 1 <i< T, do Fok Eso let me delete some stuff.
4:
Load Q;from HBMto on-chip SRAM.
1: Divide Qinto T,=
[blocks Q
each. and divide K. V in to T=
blocks
2: Divide the output O e RNxd into Tblocks Ou
Qof size Bxd gach, and divide the logsumexp L
3:for 1≤i≤ T, do Fo REACH BLO
Algorithm 1 FLAs H ATTENTION-2forwardpass
Reauire
B.
所以这里的内容将与○相乘, 0是一个矩阵, 是一个2乘128的矩阵
so this stuff here is-getting. multiplied by o, which is a matrix, that is a 2 by 128.
eauire
Beau ire
所以我们在进行这个乘法运算.
Algorithm 1 FLasn Arrenr So-we are. doing this multiplication.
R
现在进行乘法运算.
Reauir
ck sizes Bc. B,
Reau ir Br
现在这是输出的结果.
Beau ire
所以这是二.
Beau ir
Beau ire
让我写 下来:2乘2矩阵与2乘128矩阵相乘的结果将是一个2乘128的矩阵
let me write it : two. by. two. multiplied by-two-by 128-will be a matrix, that is two by 128,
其中运算结果的第一行第一维
where. the first dimension of the first row of the output of this operation R
Algorithm 1 FLAs HATr ENTION-2forward pass Beau ire
将是这个的点积.
Algorithm 1 FLAs HATr ENTION-
. Wil. be, the dot product of this.
Beau ir
R
Algorithm 1 FLAs HATr ENTION-2forward pass Reau ire
将这一行与第一列相乘.
Algorithm 1 Fias Ar Gall this row-here with the first column.
R
Algorithm 1 FLAs HATr ENTION-2forward pass Reau ire
所以基本上我们在这里用14 的第一个元素来除这个元素.
Algorithm 1 FLAs HATr ENTION-2forward pass Reau ire
这里的第二个输出元素将是这一行
Algorithm 1 FLAs HATr ENTION-2forward pass Reau ire
与第二列的点积.
Reau ir sizes Bc. B
Algorithm 1 FLAs HATTENTIo N-2forward pass Reau ire
所以我们只进行乘法运算.
sizes Bc. B
我们在这里将输入向量的第二个元素
A Wfe are dividing. the, the. the second element here of this input vector,
R
除以 L的第一个元素,
R
因为第二行的所有元素都将乘以0,
for because the all the elements of the second row-will-be multiplied by 0,
R
Algorithm 1 FLAs HArr ENTIo N-2 forward pass
所以它们不会对这一输出行产生贡献,
B
Algorithm 1 FLAs HATr ENTION-2forward pass Reau ire
而第二个输出行将是点积的结果.
R
Algorithm 1 FLAs HATTENTION-2forward pass Reau ire
这个元素将是这一行与第一列的点积.
this element here will be the. dot product of this row with the first column.
R
这里的第一个元素乘以0, 所以它不会对这个输出产生贡献.
the first Element here is multiplied by O, so it will-not contribute to this output.
所以只有第二个元素, 即第二行的第一个元素.
Algorith nio it's only the second element, the first row-of the second.
Algorithm 1 FLAs HATr ENTION-2forward pass Reau ire
输入矩阵第二行的第一个元素将除以 L 的第二个元素,
this first element of the second row-of the input matrix here will be divided by L for 2,
这基本上将应用于第二行的所有元素
and So basically. this will-be. applied, will divide all the elements in the second row
而第一行的所有元素将除以 L的第一个元素,
andthis will. divide all the. elements in the first row-in producing this one here,
R
从而生成我们这里需要的归一化结果.
B
我们需要应用这个归一化因子,
R
Algorithm 1 FLAs HATr ENTION-2forward pass Reau ire
这应该能帮助你更好地理解为什么这个操作最终会归一化输出向量
and this should help you better visualize why this-operation is normalizing the vectors
并且仍然得到相同的结果.
Algor it un? f the. output at the. end and still obtaining the same result.
R
Algorithm 1 FLAs HATr ENTIo N-2forward pass Reau ire
现在让我们继续深入.
Reau ir
R
16:end for
15:
17: Return the output O and the log sume 好了, 各位,
All right, guys, finally,
15:
16:end for 我们 终于可以看看 Flash Attention 的前向传播过程了, 同时也可以将其与
we are ready to see the Flash Attention Forward Pass by also comparing it with
16:end for
15:
17: Return the output 我们自前推导的内容进行比较.
what we have derived so far.
15:
16:endfor So if you look at the Flash Attention paper, first of all,
15:
16:endfor
17: Return 这是 Flash Attention 2 的前向传播过程.
this is the Flash Attention 2 Forward Pass.
16:end for
15:
17: Return the output O and the logs umg 稍后我会解释
And later I will explain
15:
16:endfor Flash Attention T 和'Flash Attention 2 之间的区别
what are the differences between the Flash Attention 1 and the Flash Attention 2.
15:
16:endfor
17: 我不想直接跳到这个前向传播过程, 因为我相信,
didn't want to jump directly to this forward pass because I believe that,
15:
16:endfor
17: Return the output O and 即使推导过程有点难懂,
even if the derivation was a little difficult to follow,
15:
16:endfor
17: 它也能让你对正在发生的事情有一些直观的理解.
I believe that it gave you some intuition into what is happening.
15:
16:endfor 所以即使你只理解了其中的50%, 也足够了, 因为稍后我们还会编写代码
So even if you understand 50% of it, that's enough because later we will also code it
15:
16:endfor
17: Return the output 你应该能达到90%的理解程度.
and you should reach like 90% of understanding.
15:
16:endfor 因此, 每当我们引入一些新的信息时, 它都应该能提高你的理解.
So every time we introduce some new information, it should improve your understanding.
15:
16:endfor 基本上, 我们以
So, Basically, in Flash Attention, what we are, Flash Attention 2 especially, we take our,
15:
16:endfor as input, we have our query key and values, which are sequence of tokens.
15:
16:endfor 每个token由-个d维向量组成, 其中d是小写的d维.
Each token is made up of a vector of d dimensions and d, lowercase d dimensions.
14:
Write O;to HBMas the i-th block of O.
15:
Write L;to HBA 然后我们将这个查询分成若干块.
16:end for
17: Return the output O andthe And we divide this query, guess what, into blocks.
log sum exp L
14:
Write O; to HBM as the i-th block of O.
15: 分成多少块呢?
16:end for
17: Return the output O and the log sum exp L.
In how many blocks?
14:
Write O; to HBM as the i-th block of O.
15:
Write L;to HBM as the i-th block 这取决于参数br,
16:end for
17: Return the output O and the log sum exp L Well, depending on this parameter br,
14:
Write O; to HBM as the i-th block of O.
15:
16:endfor
17: Return the output O and the logs u which is the size of the query block that we want to choose.
14:
Write O; to HBM as the i-th block of O.
15:
16: 也就是说
So how many rows of query we want to group together into one block.
17 :
Return the output O and the log sum exp L
14:
Write O; to HBM as the i-th block of O.
15:
16:end for
17: Return the output O and the log sum exp L And we also do it with K and V,
14:
Write O; to HBM as the i-th block of O.
15:
Write L;to HBM
16:endfor
and we divided that into blocks depending on this parameter BC.
14:
Write O; to HBM as the i-th block of O. 也就是我们想要生成的输出.
Then we also initialize the output, which is the output that we want to produce.
14:
Write O; to HBM as thei-th block of O.
15:
Write
o HBMasthe-thblock of L flash attention 到底在计算什么呢?
17 : Return the output O and the log sum exp So what is the flash attention computing?
14:
Write O;to HBM as the i-th block of O.
15:
flash attention 计算的是以下内容.
Well, the flash attention is computing the following.
它计算的是soft max.
So it's computing the soft max.
具体来说 就是用查询乘以键的转置, 再除以归一化因子, 得到soft max 值
the soft max of the query, multiplied by the transpose of the keys,
最后乘以 V,
divided by the normalization factor, multiply that by V,
这就是它的计算过程, 而且是以这种方式来计算的.
and so that's what it's going to compute, and it's going to compute it this way.
首先, 我们有一个针对查询的外层循环,
First of all, there is an outer loop through the queries,
这与之前看到的伪代码是一致的
which corresponds to the same pseudo code that we have seen before,
因为我们要并行计算
because we want to compute each block of the output matrix in parallel
输出矩阵的每个块.
with respect to the others.
AND SOON UNTi L THE LAST STEP. THEN, WE 说白了
, 我们希望独立计算这个输出块和那个输出块.
So basically we want to ;compute this output block and this output block independently.
0
HOW CANWEFIXTHEPREWOOS
ITERATION'SOUTPUT?
THE ONLINE SOFT MAX
这里的这个输出块依赖于第一个查询和所有的键.
THE ONLi NE SOFT MAX
HOW CAN WE FIX THE PRE WOOS ITERATION'SOUTPUT?
THE ONLi NE SOFT MAX
而这个输出块则依赖于第二个查询和所有的键.
THE ONLINE SOFT MAX
HOW CAN WE FIX THE PRE WOOS ITERATION'S OUTPUT?
THE ONLi NE SOFT MAX
这个输出块依赖于第三个查询和所有的键
THE ONLINE SOFT MAX
HOW CAN WE FIX THE PRE WOOS ITERATION'SOUTPUT?
THE ONLi NE SOFT MAX
这里的查询1 不是指第一个查询
THE ONLINE SOFT MAX
而是指第一组查询或第一个查询块;查询2 也不是指第二个查询
OUTPUT THE ONLi NE SOFT MAX
HOW CAN WE FIX THE PRE WOOS ITERATION'S OUTPUT?
THE ONLi NE SOFT MAX
而是指查询矩阵的第二个块,
OUTPUT THE ONLi NE SOFT MAX
以此类推
HOW CAN WE FIX T
THE
ONLi NE SOFT MAX
HOW CAN WE FIX THE PREV ONS ITERATION'S OUTPUT?
THE ONLINE SOFT MAX
正因如此, 我们需要在所有块之间进行外层迭代,
So that's why we have this outer iteration among all the blocks,
这样才能并行计算输出矩阵的所有块.
because we want to compute all those blocks of the output matrix in parallel.
但要计算每一个输出块
But to compute each of these output block,
17 : Return the output O and the log sum exp L. 我们需要遍历所有的键.
we need to go to an iteration among all the keys.
这就是为什么我们在键上设置了一个内层循环.
That's why we have an inner loop on the keys.
我们执行的正是之前已经完成的操作.
And we do exactly the same operation that we have done so far.
手动操作.
by hand.
首先我们计算s矩阵,
sofirst we compute the s matrix,
它是每个查询块与对应键块的乘积结果.
which is what the each block of query with the corresponding block of the keys.
17 : Return the output O and the log sum exp L. 接着, 我们计算当前s块的局部最大值.
then we compute the local maximum to the current s block.
17 : Return the output O and the log sum exp L 这是局部最大值, 我们会将其与前一次送代的最大值进行比较
this is the local maximum and we compare it with the maximum of the previous iteration,
因为这是在线 Soft max 中的标准做法.
because that's what we do in the online soft max.
17 : Return the output O and the log sum exp L. 随后, 我们计算p块, 即 s块的 Soft max *值
then we compute the p, the p block, which is the soft max star of the s block,
减去s 块的局部最大值.
minus the local maximum of the S block.
接下来, 我们计算归一化因子.
Then we compute the normalization factor.
归一化因子是什么?
What is the normalization factor?
它是所有 Soft max *指数值的总和
It is the summation of all the exponential of the soft max star,
但需要结合上一步的归一化因子进行调整.
but by fixing the normalization factor of the previous step.
我们知道如何调整归一化因子
and we know how to fix the normalization factor
只需乘以一个指数值
because we just multiply by an exponential,
即前一次最大值与当前最大值的差值.
which is the previous maximum minus the current maximum.
这就是这个因子的作用
That's what this factor is.
接着, 我们使用之前提到的修正因子
and then we compute the output exactly using the same correction factor
精确计算输出,
that we have seen before,
这个因子是一个对角矩阵, 其对角线上的元素由这个向量构成,
which is the diagonal matrix, made up of the diagonal,
该向量是前一次最大值与当前最大值的差值
where on the diagonal you have the elements of this vector here,
取指数后的结果.
which is the exponential of the previous maximum minus the current maximum,
我们将这个结果乘以上一步的输出, 团 因为需要修正上一步的计算结果
multiplied by the output of the previous step, because we want to fix the previous step,
它是基于上一步的局部最大值得出的
because it was based on the previous p,
再加上当前基于局部最大值的 PV 值
which was using the maximum of the local previous p, plus the current PV,
而这一结果将在下一次送代中被修正.
which is based on the current local maximum, and it will be fixed by the next iteration.
好的, 最终, 在我们遍历完所有的键之后,
Okay, and at the end, after we have gone through all the keys,
我们已经计算出了所有输出块, 但尚未应用归一化因子.
so we have computed all the output block, but we didn't apply the normalization factor.
归一化因子在最后应用, 是因为在处理每个键的过程中
and it's applied at the end because while going through each key,
我们正在计算 Soft max 的 L 归一化因子
we are calculating the L normalization factor for the soft max,
而在这个循环内部, 我们仅计算了 Soft max *
because inside of this for loop we are just computing the soft max star,
因此并未对每个值进行归一化.
so we are not normalizing each value.
因此, 最后需要对结果进行归一化处理, 具体操作如下:
So, at the end, someone has to normalize it, and it will be this, this instruction here,
使用我们在所有送代中计算得到的归一化因子
which is : use the normalization factor that we have computed over all the iterations
并将其应用于o 的每个元素. 因为 Soft max *与实际 Soft max 的区别
and apply it to each element of o, because the difference between the soft max star
仅在于是否除以了归一化因子.
and the actual soft max is just the division by the, the normalization factor.
这里的操作实际上是将每个 O And this instruction here is actually dividing each o
除以对应的归一化因子,
with the corresponding normalization factor,
每 个因子对应输出块中的一行, 也就是我们正在计算的输出矩阵的每一行
one for each row of the block, each row in the output block that we are computing.
稍后, 我们还将探讨接下来的操作步骤.
Later, we will see also what do we do?
SRAM 是什么?
What is this SRAM?
HBM 是什么?
What is the HBM?
现在, 我只需要你专注于我们正在进行的操作.
For now, I just want you to concentrate on the operations that we are doing.
这些操作与我们今为止所做的完全一致.
And they're exactly the same operations that we have done so far.
稍后我们会详细讲解.
later we will see.
此外, 我们还需要了解为什么要在这里保存这些内容等等.
also, why do we need to save this stuff here and etc.
等等.
etc.
但目前, 你应该已经掌握了足够的知识, 能够理解
but for now, you should have enough knowledge to be able to follow
15:
Write L;to HBMas the i-thblockof L
16:end for what is written in the flash attention paper.
15:
Write L;to HBMas the i-th blockof L.
16:end for 因为, 就前向传播算法而言,
17: Return the output O and for, with respect to the forward pass algo


让我们引入一个属性，步幅(stride)，步幅是什么呢，步幅告诉我们在每个维度上需要跳过多少个元素，才能到达该维度的下一个元素，举个例子"假设我们想要访问某个元素，此如第一行的所有元素. 我们把这里的张量称为t，因此, t[0] 表示获取第一行的所有元素，在第一行中, 只选择第一行, 并返回该行的所有元素，这种索引方式是如何工作的呢?

从指向第一个元素的指针开始，它会选择第一行, 然后逐个移动索引, 依次访问每个元素，因此, 它会依次选择第一个、第二个、第三个元素，它是如何知道需要逐个元素移动的呢? 因为在这个维度上, 步幅(stride)为1，因此;步幅告诉我们在这个维度上需要跳过多少个元素才能到达下一个元素。现在假设我们想要获取 t 的 [0, 1]部分,那么, 在这种情况下, 假设我们想要获取 T 的第一行(即索引为1的行)的所有元素,首先, 它需要跳过第一维度中的一些元素. 它需要跳过索引为 0 的元素, 因为我们并不选择它. 我们只想选择第一维度中索引为1的元素, 也就是索引为1的那一行，因此, 由于它将从指向第一个元素的指针开始, 它需要知道要跳过多少个元素, 而这个跳过的元素数量是由步幅(stride)决定的. 所以，步幅告诉我们, 要到达第一维度的下一个元素, 需要跳过多少个元素. 因此 在这种情况下, 它会从指向第一个元素的指针开始, 跳过三个元素，这样就从第二行开始，然后, 在这一行内部，它会根据第二维度的索引，（步幅为 1）继续遍历，所以它会一个接一个地遍历, 最终只返回内存中的这一部分.

总结一下，步幅(stride)就是一个数值，它告诉我们，在每个维度中需要跳过多少个元素才能到达该维度的下一个索引位置，这意味着要从一行跳到另一行, 我们需要跳过三个元素，要从一列跳到一列, 我们只需要跳过一个元素. 步幅为什么有用呢？

步幅之所以有用，是因为它让我们能够轻松地重塑张量，而无需进行任何计算，让我们来看看吧，好的, 假设我们想重塑一个矩阵，假设最初这个矩阵的形状是2行3列, 也就是2行乘以3列, 我们按照以下方式计算步幅，这意味着, 要从一行跳到下一行, 你需要跳过3个元素，而要从一列跳到下一列，你只需要跳过 1 个元素，因此, 你需要每次跳过一个元素, 如果我们想将其重塑为3行2列的形状，也就是3行乘以2列，我们可以通过改变步幅来重塑它, 而无需实际改变其内存布局，因为观察张量的物理配置，我们可以通过相同的物理视图将其访问为这种形状或这种形状，因为要从一行跳到下一行, 步幅是 3，所以我们需要跳过 3 个元素，这意味着第二行的起始元素由起始指针加上3 个元素给出，因此第二行将正好从这里开始，而第二行的每个元素都紧挨着，因为第二维的步幅是 1，所以你可以看到，要获取第二行，我们可以从这里开始，然后一个接一个的获取所有这些元素，这就是第二行，假设我们想从这个视图、这个形状、这个重塑后的矩阵中获取第二行，该怎么做呢?

让我们来看看步幅?现在行方向的步幅是 2，这意味着要从一行跳到下一行，我们需要跳过 2 个元素，所以如果我们想选择这一行，就从内存的起始点开始，也就是这个起始指针，我们跳过前两个元素，因为步幅表明要从一行跳到下一行需要跳过 2 个元素，于是我们达到这里，然后正好选择两个连续的元素，因为第二维的步幅是 1，步幅让我们能够在不改变张量在内存中实际存储方式的情况下，对其进行重塑，此外,步幅还使我们能够在不改变矩阵在内存中存储形状的情况下，获取其转置矩阵，也就是说，无需改变元素在内存中的排列方式，这非常酷, 因为我们可以在不改变内存中任何内容的情况下，将同一矩阵既视为未转置的版本，又视为转置后的版本，因此, 只需通过操作索引和步幅, 就能免费获得这一特性，

因此，要在两个维度上对矩阵进行转置，我们只需交换这两个维度上的步幅即可，举个例子, 假设我们想得到这个矩阵的转置,只需要交换步幅即可。那么, 如果我们想获取转置矩阵的第二行，该如何操作呢? 我们始终持有指向张量存储的第一个元素的指针，也就是张量在内存中存储的起始位置.这表明，要从一行跳到下一行，我们需要跳过一个元素，这没错, 因为正你所见，第二个元素在内存中正好也是第二个元素，因此, 我们只需跳过一个元素, 就能到达第二行的起点，然后, 要在同一行中从一个元素跳到下一个元素, 我们需要跳过三个元素，因此, 第二行的第二个元素将在第一行的第一个元素之后, 跳过三个元素的位置，所以, 在第二个元素之后, 我们需要跳过三个元素. 于是我们跳过这个元素, 再跳过这个元素最终到达这个元素, 即 8, 它正好是第二行第二列的元素，由此可见, 步长(stride )基本上能让我们实现两件事，一是它允许我们重塑张量，而无需在内存中重新分配其存储结构，二是，它让我们能够在不重新排列内存中元素的情况下转置矩阵, 这非常棒, 因为移动内存数据的开销很大.

由于重新排列内存数据代价高昂，所以这种无需额外操作就能实现的功能非常实用，另外，举个例子，在 PyTorch 中，有两种方法可以用来重塑张量，一种叫做 reshape 方法，另一种叫做 view 方法，在通过交换两个维度的步幅来转置矩阵后，就无法再免费重塑张量了，因为张量的步幅本质上是什么呢? 步幅是如何计算的呢? 步幅其实就是所有后续维度形状的乘积, 让我用一个具体例子来说明.

因此, 第零维的步幅就是，后续维度形状中所有元素的乘积，所以第零维的步幅就是从索引 1 开始的所有形状的乘积，在二维矩阵中不太容易着出来·因为元素数量不够多，所以我们用三维矩阵来说明.这是一个具有三个维度的张量，它的形状是2、4、3, 这意味着我们有两个矩阵. 每个矩阵由四行三列组成，步幅的计算方式如下：第 0 维的步幅就是 4 乘以 3 的乘积，这里的 3 就是 3 乘以 1 的结果，因为 3 之后没有其他维度，因此, 当我们进行转置时, 这种步幅特性就会丢失, 在通过交换步幅完成矩阵转置后,我们无法再进行进一步的形状重塑操作，所以, 从根本上说, 张量在逻辑上不是连续的，因此, 这是一个非常高级的特性，了解与否并不重要, 但如果你对此感到好奇，基本上, 在 PyTorch 中, 你无法在张量转置后对其进行视图操作，因为 PyTorch 在转置张量时只是交换了两个步幅，但失去了步幅特性, 这本质上是步幅将不再等于后续形状的乘积，因此, 这里不再是2, 比如说这里应该是2, 而这里应该是1. 但在转置后, 这一特性就丢失了, 因此如果你想在转置后重新调整张量的形状, 实际上需要重新分配张量，记住这一点并不重要, 这只是个有趣的知识点.


那么, 这个转置到底是什么呢? 步幅是用来做什么的呢？步幅的作用是什么呢？步幅主要有两个用途，首先, 它用于理解如何索引这个张量，因此, 只要有了指向这个张量起始地址的指针, 我们就可以按照自己的需求来索引这个张量，这样我们就能访问任意行、任意列的数据，此外, 它还能让我们在不改变内存中元素排列的情况下, 免费重塑这个张量的形状，第三，它使我们能够通过简单地交换想要转置的两个维度的步幅, 随心所欲的转置张量，既然我们已经了解了张量是如何存储在内存中的，现在终于可以进入 Triton 的世界, 看看一些具体的例子了. 好了，各位, 既然我们已经了解了张量的工作原理以及 CUDA的工作方式，现在我们可以来看一些 Triton 内核的示例, 看看 Triton 与 CUDA 有何不同，现在，如果你访问 Triton 的官方网站，你会发现一些教程，就像这里展示的这一部分.，让我们一起学习一个教程，来理解 Triton 与 CUDA 的区别，如果你进入教程部分, 会发现里面有很多示例，首先, 我将为 Flash Attention 编写的代码是基于这里的一个教程，融合注意力机制, 你可以在这里看到，但做了一些修改, 因为我大幅简化了代码. 例如, 我去掉了 FP8的实现部分，此外，我还，比如, 这里的融合注意力代码仅适用于反向传播过程, 并且只针对因果注意力机制，而我的代码则同时适用于因果和非因果注意力机制. 我做的第二个修改是, 他们没有使用这里提到的指数工具来加速运算。因为指数工具是通过一个更快的单元实现的，我使用的是 Flash Attention 的原始实现，它使用的是以e为底的指数函数等.，等等，因此, 我尽可能简化了代码, 使其易于理解, 而不是一味追求优化. 所以我的代码肯定比这里看到的融合注意力要慢一些，但它应该更易懂, 更容易理解. 无论如何，让我们开始向量加法的教程吧，如果你去看向量加法的教程，里面有一些关于如何使用 Triton 进行向量加法的示例. 这应该能帮助你进入使用 Triton 编写内核的思维模式，与其先编写内核再调用它, 不如我们反其道而行之。那么，我们来看看如何调用这个内核, 并探索它是如何工作的，我已经从网站上复制了教程中的向量加法代码，首先, 让我们看看我们想要实现什么，我们有一个输入向量×和一个输入向量，我们想要计算向量加法，这意味着使用torch 时, 我们希望执行以下操作，同时我们也希望通过调用这个名为 add 的方法, 用 triton 实现相同的操作，然后我们想比较这两个向量的输出结果, 它们应该是相等的，或者至少差异应该非常非常小，因为在使用浮点数时，总会存在一些舍入误差，这个向量的大小是98, 000个元素, 我们想要以分块的方式进行处理，正如你之前所记得的，使用 CUDA 时, 可以通过启动大量线程来实现向量加法。每个线程执行一个操作，但当线程数量不足以处理所有数据时。就需要将输入向量划分为多个块，这正是我们接下来要做的，那么, 让我们来看一下这个add 方法, 这个add方法基本上会首先为输出向量分配所需的内存, 然后计算启动网格. 启动网格会告诉 Triton, 就像在 CUDA中一样, 我们想要启动多少个线程块，也就是要启动多少个线程块，如果你还记得, 在 CUDA内核中，我们会指定需要多少个线程块, 以及每个线程块中有多少个线程，在 Triton的情况下, 我们只需指定需要多少个线程块，而不强制规定要启动多少个线程，Triton 会自行决定启动多少个线程，我们只需告诉每组线程应该执行什么任务，以这个例子来说,"我们将元素数量 N(这里是98, 000)，划分为大小为1, 024的块, 这个大小被初始化为1, 024，这基本上是说, 为了计算网格大小, 你需要进行向上取整的除法，也就是说, 用元素数量n除以块大小, 然后向上取整，这就是这个表达的含义, 因此，我们需要多少块，现在, 每个块应该执行的任务都在内核中定义,
ensor([1. 3713, 1. 3876, 0. 4948,
Now, what each block should do is inside of the kernel.
> TIMELINE OUTLINE
tensor ([1. 3713, 1. 3076, 0. 4940,
/5. 6s
., 0. 6724, 1. 2141, 0. 9733], device =′cuda:0′)
ensor([1. 3713 接下来, 我们转到内核部分, 当启动内核时
Solet'sgo to the kernel, and when we launch the kernel,
> TIMELINE OUTLINE
tensor ([1. 3713, 1. 3076, 0. 4940,
tensor([1. 3713, 1. 3076,
.., e. 6724, 1. 2141, 0. 97331, device =*cuda:0′)
The naximus difference 可以在方括号中指定启动网格
10. 4948,
11. 2141, 8. 97331,
we can specify the launch grid in this square parenthesis > TIMELINE OUTLINE
tensor ([1. 3713, 1. 3076, 0. 4948,
tensor ([1. 3713,
..., 0. 6724, 1. 2141, 0. 97331, device =cuda:0′)
The naximu 然后在圆括号中指定这个内核的参数,
12. 6724, 1. 2141, 8. 97331,
and then in the round parenthesis, we specify the arguments of this kernel > TIMELINE OUTLINE
tensor ([1. 3713, 1. 3076, 0. 4940,
tensor ([1. 3713, 1. 3076, 0. 4940,
, 0. 6724, 1. 2141, 0. 97331, device=cuda:0′)
The maximum difference b 现在, 让我们进入内核部分,
So let'sgo to the kernel > TIMELINE OUTLINE []
n_elenents =output. numel () 我们发现,: Python 和 Triton 并未直接让我们访问张量x.
we see that python, tri tony will not give us access to the tensor x.
> TIMELINE OUTLINE 4
# In this case, we use a 1 D grid where the size is the grid bda meta:(triton. cdiv(n_elenents, meta ['BLo CK_ SIz E']),) 它提供的是指向该张量第一个元素的指针
it will give. us a pointer to the first element of this tensor > TIMELINE OUTLINE
这让我们回想起张量的布局方式.
and. this. takes us back. to the tensor layouts.
> TIMELINE OUTLINE
this case 我们之所以研究张量布局、
Tuple [int ]
so the reason we studied the tensor layouts > TIMELINE OUTLINE
步幅等概念, 是因为这段代码 即add 内核将在 GPU 上运行
and the strides and all the stuff is because triton, this code, uh,
> TIMELINE OUTLINE 4
n _elements =output. numel ()
# In this case, we use a 1 D grid w
# It is analogous to Cu DA launch grids 而 GPU 并不能
meters ) Tuple [int ].
this add kernel will run on the gpu and the gpu can not, um, does not,
grid =
la nbda meta:(triton. cdiv(n_elen
> TIMELINE OUTLINE
It is analog this ca s 像 Py Torch 那样通过多维索引
index tensors like py torch by using all the dimensions > TIMELINE OUTLINE
以及广播等高级功能直接操作张量.
and with the broadcasting and all this fancy stuff.
> TIMELINE OUTLINE
GPU 只会提供指向内存中张量首元素的指针
the gpu wil I just give you the pointer to the first element of this tensor in the memory > TIMELINE OUTLINE
至于如何计算你想要访问的所有元素的索引
and then it's up to you to compute all the indexes > TIMELINE OUTLINE
analogous 则完全由你自己负责.
> Tuple [int ]
of all the elements that you want. to access.
> TIMELINE OUTLINE
n _elements =output. numel () 而
yp tr则是指向向量y中第一个元素的指针.
this y pointer is the first, the pointer to the first element of the y vector.
> TIMELINE OUTLINE
此外, 我们还有一个指向输出向量的指针
then we have the pointer to the output vector > TIMELINE OUTLINE
n _elements =output. numel ()
It is analogous this case, 用于存储矩阵加法的结果.
> Tuple [int ]
where we want to store the result of this matrix addition.
> TIMELINE OUTLINE
我们指定了向量包含多少个元素以及块大小
we specify h how many ·elements our vectors have and what is the block size,
> TIMELINE OUTLINE
即每个块应处理多少数据项)
so how ·many items each block should process,
> TIMELINE OUTLINE
但这并不一定与每个内核拥有的线程数相对应.
which may not correspond to how many threads each kernel will have.
> TIMELINE OUTLINE
你可能会感到困惑, 因为在 Triton 和 CUDA 中
You may be confused because, okay, in. Triton, in Cu DA.
> TIMELINE OUTLINE
我们确实定义了每个块应包含多少线程.
we specified how many threads ·each block should have.
> TIMELINE OUTLINE
因此, 我们管理的粒度是在线程层面上.
So the granularity that we manage is the thread level.
> TIMELINE OUTLINE
而在这里, 我们是指一组线程共同处理一定量的数据.
Here we are saying it's a group of thread. that should work with this quantity of data.
> TIMELINE OUTLINE
n _elements = output. numel () 至于 Triton 实际使用多少线程来优化执行, 则由它自行决定.
Then it's up to Triton to optimize. the number of threads that it will actually use.
> TIMELINE OUTLINE
n _elements =output. numel ()
It is analogous to Cu DA launch grids. It # In this case, we use a 1 D grid where the Callable (meta parameters )-> Tuple [int ].
grid =lambda meta :(triton. cd iv (n_elenents,
OUTLINE Each torch. tensor object is in plicitly co Actually In a callable GPU kernel.
first element > TIMELINE
assert x. is _cuda and y. is _cuda and output. is _cuda output = torch. en pty _like (x ) 通过指定数据项的数量, 我们可以间接控制实际使用的线程数
there are ways to say how many threads we actually want by specifying the number of words,
> TIMELINE OUTLINE
过这一点我们稍后再详细讨论.
but we will see that later.
OUTLINE torch.
> TIMELINE
is analog this cas 现在只需记住, 这个内核线程
for-now, just remember that. this thread, this kernel here,
> TIMELINE OUTLINE
将处理输人向量中的一定数量的元素:具体是多少呢?
will process a number of elements in the input vectors :how many number?
> TIMELINE OUTLINE
多少元素、块大小以及元素数量.
how many elements, block ·size, number of elements.
> TIMELINE OUTLINE
首先, 我们需要确定当前处理的是哪个块.
first of all, we need to identify which block we are.
> TIMELINE OUTLINE
n _elements = output. nunel() 在 CUDA中我们使用名为blockid:x的变量来标识块的编号
In Cu DA, we use the variable called block id. x·to identify the identifier of the block > TIMELINE OUTLINE
它指示我们应该处理哪一组元素.
which tells us which group of elements we should be working with.
> TIMELINE OUTLINE
n _elements =output. numel () 在 Triton 中, 您可以通过使用program id 实现相同的功能.
In Triton, you do. the same by using program id.
> TIMELINE OUTLINE
在 CUDA 中块 ID 可以沿着 X、y和 Z轴分布.
And in cu DA, the block id can be along the x, y, and z axis.
> TIMELINE OUTLINE
n _elements =output. numel() 而在 Triton 中, 这些被称为维度0、1和2.
In Triton, these are called the dimensions 0, 1, and 2.
> TIMELINE OUTLINE
n _elements =output. numel () 这里我们处理的是单维数据, 因此只需使用一个轴来指定块的索引.
Here we have one-dimensional data, so we only use one axis to specify the block index.
> TIMELINE OUTLINE
于是我们得到块的索引, 也就是 PID.
So we get the block index, which is the Pi D > TIMELINE OUTLINE
在 Triton 中, 这被称为程序1 D.
In. Triton, this is called the program ID > TIMELINE OUTLINE
更直观的理解是将其视为程序, 就像这个程序
It's more intuitive to think of it as the program, like this is a kind of a program > TIMELINE OUTLINE
与其他具有不同程序 ID 的程序并行运行一样.
that is running in parallel with other programs that will have different program ID.
> TIMELINE OUTLINE
n _elements =output. numel ()
It In
n this case, we use a 1 D grid
is analogous to Cuo A launch grids 根据程序 ID
Le(meta parameters )
> Tuple [int ]
grid =la mbda ne ta:(triton. cdiv(n_e
And based on the program. ID,
OUTLINE Each torch. ten triton. jit''edf > TIMELINE
我们可以确定这个程序应该从哪个元素开始工作.
we can understand what is the starting element this program should work with.
> TIMELINE OUTLINE
因此, 这个蓝色线程块应该处理的是这些元素.
Ca this So this blue block of threads should work with.
> TIMELINE OUTLINE
要得到这个起始位置, 只需将 PID? 乘以块大小即可.
And to get that is just the Pi D. multiplied by the block size.
> TIMELINE OUTLINE
n _elements = output. numel() 所以, PID为0的程序应从第0个元素开始处理
Sothe Pi DOshould be working with. the element-that starts from the element O.
> TIMELINE OUTLINE
n _elements =output. numel()
PID为2的程序则从第2048个元素开始处理
Andthe Pi D2-should start-from the element 2048.
> TIMELINE OUTLINE
这意味着它需要跳过前2048个元素从索引为2048的元素开始处理
Soit should skip the first 2048 elements and start with the element within dex2048.
> TIMELINE OUTLINE
_elements =output. num el() 接下来我们定义如何根据×和y向量中的指针
next, we define how to load these ·elements-based on the pointer in > TIMELINE OUTLINE
. In this case, we use a 10 gri 来加载这些元素.
grid=lanbda meta: (triton.
which of the x and they vector.
OUTLINE Each torch. ten so triton. j it'`ed
> TIMELINE Don't forget to pas
n _elements =output. nun el()
It
In this case, we use a 1 D grid where the size is analogous to Cu DA launch grids. It can or Call able (meta parameters )> Tuple [in t].
8
grid= lanbda meta:(triton. cdiv (n_elenents, meta OUTLINE Each torch. tensor object is implicitly co triton. j it''ed functions can be indexed to do that,
a callable GPU kernel.
first element > TIMELINE
我们需要指定相对于起始地址的偏移量列表, 以确定要加载的数据位置
we specify a list of offsets with respect to the starting address that we want to load.
> TIMELINE OUTLINE
_elements =output. numel () 因此, 由于 Triton 中的每个程序都处理一组数据块
so, because each program in triton works with a group of u h, um of data,
> TIMELINE OUTLINE
n _elements =output. numel ()
The SP HD launch grid denotes the It is analogous to Cu DA launch 而不是单个元素
> Tuple [int ]
In this case, we use a 1 D grid
bda meta:(triton. cdiv
so not one single element but a block of elements,
> TIMELINE OUTLINE
this 我们需要明确要加载哪些元素.
we mean we ·need to understand which elements to load.
> TIMELINE OUTLINE
n_elenents =output. numel ()
In this case,
is analogous 也就是这些元素的偏移量.
Tuple [int ]
grid =lambda net a :
OUTLINE triton. jit''ed
ach torch. te
so the offset of these elements.
> TIMELINE
n _elements =output. numel() 对于程序1 D为0的情况, 它将加载从索引0开始的数据块
in the case of the program i d O, it will load the block start, so0,
> TIMELINE
OUTLINE
4
直到但不包括索引1024, 而程序1 D为1则处理下一个数据块.
plus the elements from index 0to. 1001024, excluded with the program elementum one.
> TIMELINE OUTLINE
n _elements =output. numel ()
The SPMD launch grid denotes It is analogous to Cu DA laun 这样操作的结果是
arameters )> Tuple [int ]
In this case, we use a 1 Dg
this basically will result in a vector. that is uh, well,
> TIMELINE OUTLINE
的向量将从1024开始, 接着是102510261027, 依此类推, 直到204
the program start with pidequalt0 one, will be 1024, then1025, 1026, 1027, etc, etc,
> TIMELINE OUTLINE
It is analogous In this case, 而对于程序1 D为2的情况
Tuple[int]
until 2047, um, with the program number, let's say two, this, uh,
> TIMELINE OUTLINE
output =torch. enpty_like(x)
output 偏移量对应的元素将是20482049, 以此类推
this-offsetswillbetheelements2048, 2049, blah, blah,
> TIMELINE OUTLINE
output = torch. enpty_like(x)
the output. 直到接近3000左右的数值
OUTLINE grid=
bda neta:
blahuntil3oo0 and something um.
> TIME LINE
robject is implicitly rted into a pointer to its first element.
We need to pre allo output 您可能还记得, 当我们启动一个网格时
now we also as you remember, when we create um, when we launch a grid > TIMELINE OUTLINE or object is implicitly er ted into a pointer to its first elem e
线程的数量并不总是基于数据块中的元素数量
the number of threads is not always-based on the number of elements in the block > TIMELINE OUTLINE
或向量中的元素数量来决定的.
or. the-number of elements in your vector.
> TIMELINE OUTLINE
基础数的倍数, 这个基础数一般取32 这意味着网格的规模会据此调整
it is always a multiple of a base number, which is usually 32, which means that the grid > TIMELINE OUTLINE
n _elements =output. numel () 因此 这个程序可能会有超出实际需要的线程数量,
this program may have more. threads that it needs,
> TIMELINE OUTLINE
assert x. is _cuda and y. is _cuda and output. is _cuda output =torch. en pty _like (x ) 那些多余的线程应保持闲置状态,
So some threads should not be doing anything,
> TIMELINE OUTLINE
assert x. is _cuda and y. is _cuda and output. is _cuda output = torch. en pty _like (x ) 不加载任何数据, 也不参与任何计算求和的操作,
so should not be loading any data and should not be computing any summation.
> TIMELINE OUTLINE
assert x is _cuda andy is _cuda and output. is _cua output = torch. en pty _like (x)
n_elements = output. numel () 正因如此, 我们需要借助这个掩码来实现上述控制.
So. this is why we need this mask > TIMELINE OUTLINE
rit on. j it`'ed
assert x. is _cuda and y. is _cuda and output. is _cuda output = torch. en pty _like (x) 这意味着我们加载的所有偏移量最多只能涵盖n个元素
This means that all these offsets-that we are loading should be at most up tonelements > TIMELINE OUTLINE
设想一下, 如果你的向量共有2060 个元素.
because imagine you-have a vector of 2060 elements.
> TIMELINE OUTLINE
也就是说, 该内核第三个程序对应的偏移量将加载
Which means that this offset for the third program of this kernel will load > TIMELINE OUTLINE
In this case,
analog 2048、2049到2060,
> Tuple[int]
grid=lan bd a neta:
OUTLINE ach torch.
the offset that go from 2048 > TIMELINE
以及之后的20612062等元素.
2049, 2060, and then als02061, 2062, etc., etc.
> TIMELINE OUTLINE
但我们之前提到, 实际只有2060 个元素存在.
Butwe·said-thatweonlyhave2, 060elements.
> TIMELINE OUTLINE
所以, 从20612062 开始直到3000 多的那些元素
n_elements=output. numel()
So all the elements of 2, 061, 2, 062, et cetera, until 3, 000 and something,
> TIMELINE OUTLINE
n _elements =output. nunel()
# In this case, we use a 1 D grid It is analogous to Cu DA launch grid 其实并不存在
grid = lambda meta :(triton. cd iv (n_ele
they don't exist.
OUTLINE Each torch. tensor object is in plicit triton. j it'ed functions can be inde
oble GPU kernel.
> TIMELINE Don't forget to pass net a-parameters as
this 因此我们需要以某种方式告知
So we need to tell somehow > TIMELINE OUTLINE triton. jit `'edf
assert x is _cuda and y. is _cuda and output. is _cuda output = torch. en pty _like (x ) 处理这些不存在的元素的线程, 它们不应加载任何数据.
that all the threads that are working with. these elements should not load anything.
> TIMELINE OUTLINE
这就是我们需要这个掩码的原因.
That's why we need this mask > TIMELINE OUTLINE
rit on. j it''ed func
这个掩码的作用是, 在所有的偏移量中
this. mask-tells load, among all the offsets,
> TIMELINE OUTLINE
指示当前线程块仅处理那些实际存在
that this block should work with only those elements > TIMELINE OUTLINE
n _elements =output. numel ()
is analogous to CUDA 且掩码值为真的元素.
-> Tuple [int ]
that actually exist for which this mask is true.
net a :(tri to > TIMELINE OUTLINE
接下来我们加载当前程序对应的元素组
then we. load the elements of. this. current program,
> TIMELINE OUTLINE
这些元素由指定的偏移量定义,
up le [int ]
which is a group of elements defined by these offsets,
> TIMELINE OUTLINE
但仅加载那些掩码值为真的元素.
and only the one for ·which this mask is true.
> TIMELINE OUTLINE
_elements =output. numel () 也就是说, 只有实际存在的元素才会被加载.
So only the one that actually exists > TIMELINE OUTLINE
rit on. jit `
而其余的元素则应当被忽略. lnt All the others should be ignored.
> TIMELINE OUTLINE triton. j it`'ed
assert x is _cuda and y. is _cuda and output. is _cuda output =torch. en pty _like (x ) 此外, 我们还可以指定当掩码为假时应该加载什么内容.
And we can also specify what it should load in case the mask is false.
> TIMELINE OUTLINE
n _elements =output. numel () 虽然可以通过另一个参数来设置, 但在这里我们并未看到相关内容
with another parameter, but we're not seeing that here.
> TIMELINE OUTLINE
我们同样加载了y 向量中的元素组
We also. load the group of elements of they vector > TIMELINE OUTLINE
然后计算输出×加y 的结果.
and then we compute the output x plus y.
> TIMELINE OUTLINE
如果您还记得之前在 CUDA 中我们是这样操作的
So if you remember previously in Cu DA we did something like this,
> TIMELINE OUTLINE
output = to rch. enpty_like(x) 比如output[i]=x[i]+y[i]o.
like the output ofi is equal to the xofi plus the yof i.
> TIMELINE OUTLINE callable GPU kernel
output = torch. en pty _like (x)
assert x. is_cuda and y. is_cuda and output. is _cuda 那时我们是一次处理一个元素, 因为每个线程只负责一个索引.
so we did it one element at a time because each thread was working with one index.
> TIMELINE OUTLINE callable GPU kernel.
output = torch. en pty _like (x)
assert x. is_cuda and y. is_cuda and output. is _cuda 而在这里, 我们是以一组元素为单位进行处理的
here we are working with a group of elements.
> TIMELINE OUTLINE
rit on. jit callable GPU kernel.
output = torch. en pty _like (x ) 这里 的×是一个元素组, 确切地说是一个元素块, 其大小最多为block size so this x is a group of elements, is a block of elements, at most of size block size,
> TIMELINE OUTLINE all able GPU kernel
output = torch. en pty _like (x ) 实际上就是block size 这么大o actually of size block size, and it's this.
> TIMELINE OUTLINE callable GPU kernel.
output = torch. en pty _like (x)
assert x. is_cuda and y. is_cuda and output. is _cuda y是来自y向量的一个元素组, 我们正在逐组计算输出
y is a group of elements from the y vector and we are computing the output group by group.
> TIMELINE OUTLINE
output =torch. en pty _like (x)
assert x. is_cuda and y. is_cuda and output. is _cuda _elements =output. numel ()
# It is analogous to Cu DA launch grids.
In this case, we use a 1 D grid where 因此
meters )-> Tuple [int ].
grid =lanbda meta:(triton. cdiv(n_elenents,
so this,
OUTLINE NOTE :
Each torch. tensor object is implicitly c or its first element.
> TIMELINE obtain callable GPU kernel.
output = torch. en pty _like (x) 是将×的一组元素与y中对应的组相加
this is summing a group of elements of x with the corresponding group i n y
> TIMELINE OUTLINE callable GPU kernel
output =torch. en pty _like (x ) 并将结果写人output or -> Tuple [int ].
and writing it in output.
> TIMELINE OUTLINE Each torch. ter triton. j it'
sor object is
output = torch. en pty _like (x ) 接下来, 我们需要保存这个输出结果将其存储到输出张量output 中
then we need to restore this output, we need to store it in the output tensor, output,
> TIMELINE OUTLINE
output = torch. en pty _like (x)
assert x. is_cuda and y. is_cuda and output. is _cuda 你可以看到这里的ptr 是指向输出向量第一个元素的指针,
ptr, that you can see here, which is a pointer to the first element of the output vector.
> TIMELINE OUTLINE callable GPU kernel
output = torch. en pty _like (x ) 我们得确定这个输出向量该存放在哪里, 它的尺寸
and we say'that where should we store this output vector which is of size,
> TIMELINE OUTLINE callable GPU kernel
output =torch. en pty _like (x)
assert x. is_cuda and y. is_cuda and output. is _cuda It is analogous to Cu DA 和形状是怎样的呢?
grid =
this case,
lanbda meta:(triton. cdiv(n_el
shape of this vector?
> TIMELINE OUTLINE Each torch. tensor object is imp triton. jit `
callable GPU kernel
output = torch. en pty _like (x)
assert x. is_cuda and y. is_cuda and output. is _cuda 这里的大小是block size. plein t.
grid =
NOTE :
here is block size.
> TIMELINE OUTLINE Each torch. tensor object is implici
tri to n. jit`
output = torch. en pty _like (x )
uda and output. is _cuda 我们应该把它保存在哪里呢?
Where should'we save it?
> TIMELINE OUTLINE
output = torch. en pty _like (x ) 应该保存在我们加载×的相同偏移位置
Well, in the same offset to where which we loaded x.
> TIMELINE OUTLINE callable GPU kernel.
output = to rch. enpty_like(x) 如果这个程序处理的是索引20482049等位置的数据
So if this program work with the index 2048, 2049, etc., etc.,
> TIMELINE OUTLINE callable GPU kernel.
output = to rch. enpty_like(x) 那么所有的输出结果也应该写入相同的偏移位置, 民 即从2048
then all this output should be written in the same offset 2048, 2049, etc.,
> TIMELINE OUTLINE callable GPU kernel
output =to rch. enpty_like(x) 到大约3000多的地方.
Tuple[int]
(triton. cd
up to 3000 and something.
> TIMELINE OUTLINE object callable GPU kernel
output = torch. en pty _like (x) 同时
y. is _cuda and output. is _cuda 还要使用掩码
(mask )a. 因为我们并不想写入整个block size 大小的值
using the mask as well, because we don't want to write all the values of this block size,
> TIMELINE OUTLINE callable GPU kernel
output =torch. en pty _like (x )
uda and output. is _cuda 毕竟可能没有那么多元素.
Tuple [int ].
because maybe we don't have enough elements,
> TIMELINE OUTLINE GPU kernel
output = torch. enpty_like(x)
tx. is_cuda and y. is_cuda and output. is _cuda 所以, 只写入向量中实际存在的部分即可
so only write the one that are actually present in the vector.
> TIMELINE OUTLINE
output = torch. en pty _like (x ) 之所以需要这个掩码, 是因为 CUDA 会启动一定数量的线程
so the reason we need this mask is because cu DA will launch a number of threads.
> TIMELINE OUTLINE
output =torch. en pty _like (x)
assert x. is_cuda and y. is_cuda and output. is _cuda 而这些线程的数量始终是某个基本单元的倍数
that is always a multiple of a base unit > TIMELINE OUTLINE triton. jit `
output = torch. en pty _like (x ) 但这个倍数可能并不完全匹配我们正在处理的向量大小.
that may not be a multiple of the vector size that we are working with.
> TIMELINE OUTLINE callable GPU kernel.
output = torch. en pty _like (x)
assert x. is_cuda and y. is_cuda and output. is _cuda 因此, 我们需要找到一种方法, 告诉某些线程无需执行任何操作
so we need to find a way to tell some threads to not do anything,
> TIMELINE OUTLINE
output = torch. en pty _like (x ) 尤其是当数据不可用时.
Tuple [int ].
for those that the data is not available.
> TIMELINE OUTLINE
output = torch. en pty _like (x ) 那么 让我们回顾一下目前看到的内容.
So let's rehearse what you have seen so far.
> TIMELINE OUTLINE callable GPU kernel
output =torch. en pty _like (x)
assert x. is_cuda and y. is_cuda and output. is _cuda 在
CUDA 中我们编写的程序是以线程为单位的
In Cu DA, the program that we write is at the thread level.
> TIMELINE OUTLINE callable GPu kernel
output = torch. en pty _like (x ) 因此 每个线程需要明确自己应该执行什么操作.
So each thread, what it should do.
> TIMELINE OUTLINE a callable GPU kernel.
output =torch. en pty _like (x)
assert x. is_cuda and y. is_cuda and output. is _cuda 而在 Triton 中, 操作的单位是数据块
In Triton, it's this block of data.
> TIMELINE OUTLINE callable GPU kernel
output = torch. en pty _like (x ) 我们是以线程块为单位进行操作的.
We work with a block of threads.
> TIMELINE OUTLINE callable GPU kernel.
output = torch. enpty_like(x)
assert x. is _cud a
a and output. is_cuda 这个线程块应该处理哪些数据:nd
What data this block of thread should work with.
> TIMELINE OUTLINE callable GPU kernel
15:
Write L;to HBMas the i-th blockof L.
16:end for 好了备位, 乡 终于到了这一刻.
17: Return the output O All right, guys, finally the moment has come.
15:
Write L;to HBMas thei-thblock of L.
So we are going to code the flash attention for our path right now in Triton,
15:
Write L;to HBMas the i-th blockof L.
16:end for 但在此之前, 让我们先复习一下算法.
17: Return the out pl but let's rehearse the algorithm.
15 : 意力机制的目标, 特别是在 Triton 和 Flash Attention 中
Write L ;to HBM as the i-thblock of L. 注意
So the goal of the attention mechanism, specifically in Triton, in flash attention,
15:
Write L;to HBMas the i-th blockof L.
16:end for
is to compute the attention output,
16:end for
15:
Write L;to HBMas the i-thblockof L.
17: Return the which is we want to compute the output of the following formula.
具体来说就是将查询(query )与键
15:
Write L;to HBMas the i-thblock of L.
(key )的转置相乘
So the query multiplied by the transpose of the key,
15:
Write L;to HBMas thei-thblock of L. 然后除以买维度.(head dimension)白 的平方根, 接着应用sof tmax 函数
divided by the square root of the head dimension, all multiply,
15:
Write L;to HBMas the i-th blockof L.
16:end for
we apply the soft max and then all multiplied by v.
15:
Write L;to HBMas the i-thblockof L.
16: 在本视频中我们将编写前向传播和反向传播的代码. 太视频中
17:
now we in this video we will be coding the forward pass and also the backward pass.
15:
Write L;to HBMas thei-th block of L.
but before coding the backward pass, we need to understand how the auto grad works.
我们需要了解什么是梯度、什么是雅可比矩阵,
we need to understand what is the gradient, what is the jacobian,
15:
Write L;to HBMas thei-thblock of L.
16:end for
17: Rctun theoutpu如何推导矩阵乘法操作的梯度等等.
how to derive the gradient of the matrix multiplication operation, etc.
15:
Write L; to HBM as the i-th block of L.
16:endfor
17: Return the output O and the log sum exp L 等等
etc.
15:
Write L;to HBMas the i-thblockof L.
16:endfor
1: Rctu这部分内容将在视频的另一个部分中详细讲解
so that is going to be another part of the video.
15:
Write L;to HBM as thei-thblock of L.
16:end for 1: Rctum theou现在, th让我们专注于前向传播的实现
For now let's concentrate on the forward pass.
15:
Write L;to HBMas the i-thblockof L.
16:end for
17: Return the output 冒前, 我们已经掌握了一些工具
Right now, we have some tools.
15:
Write L;to HBMas the i-thblockof L.
16:endfor
17: Return the output O and the log sun 我们知道, GPU So we know that we have this thing called the GPU
15:
Write L;to HBM as the i-thblock of L.
16:end for
1: Rectun the output 可以在多个核心之间并行化操作
that can parallelize operation among multiple cores.
15:
Write L;to HBM as the i-thblockof L.
16:endfor 在
CUDA中我们可以通过编写程序来定义每个线程应执行的操作
We know that in cu DA we can parallelize operations by writing a program
15:
Write L;to HBMas the i-thblockof L.
16:end for
17: Return the output O and the logs u 从而实现并行化.
that is the definition of what each thread should do.
15:
Write L;to HBMas the i-thblock of L.
1 Rctun thoou或者我们可以遵循 Triton编程模式
16:end for or we can follow the Triton programming mode,
15:
Write L;to HBMas the i-thblockof L.
16:endfor
17: Return the ou 用· Python 指定每组线程应执行的任务
which is telling in Python what each group of threads should do.
15:
Write L;to HBM as the i-thblockof L.
16:endfor 每个线程应执行的操作
17: Return the output O and the The mapping between what each thread should do
15:
Write L;to HBMas the i-thblockof L. 及其处理的数据元素之间的映射关系, 由我们程序员来决定.
16:endfor
and which element that should thread work with is up to us, to the programmers.
15:
Write L;to HBMas the i-thblockof L.
16:endfor
17: Return the output O and the 在su Triton 中也是如此.
And the same happens in Triton.
15:
Write L;to HBMas the i-thblockof L.
16:end for
17: Return the output O and 我们指定所需的线程块数量
We tell how many blocks of threads we want,
15:
Write L;to HBMas the i-thblockof L.
16:endfor
17: Return the output 以及每个线程块应处理的数据量
how much data each block of thread should process,
15:
Write L;to HBMas the i-thblockof L.
16:endfor 这就是我们在向量加法中看到的块大小.
17: Return the ou 是
so that's the block size that we saw in the vector addition.
15:
Write L;to HBM as the i-thblockof L. 但是 向量元素与每组线程标识(即我们看到的程序 ID)之间的映射关系
16:endfor
But then the mapping between the elements of the vector and the identity of each group of
15:
Write L;to HBM as the i-thblockof L.
17: Return the output O and the logsume由我们决定.
16:end for threads, so the program ID that we saw, is up to us.
15:
Write L;to HBM as the i-thblockof L.
16:end for
: R在实现· Flash Attention 时, 也会遵循同样的原则
And the same will happen when we record flash attention.
15:
Write L;to HBM as thei-thblock of L. 让我们来看看在 Flash Attention 中, 有哪些部分可以进行并行化
16:endfor Let's see what can we parallelize in this flash attention.
15:
Write L;to HBMas thei-thblock of L. 首先您看到的这段代码是 Flash Attention 的前向传播过程,
16:endfor
So, first of all, this code that you see, the forward pass of the flash attention,
15:
Write L to HBM as the i-th block of /它接收的输
16:end for 入是查询
(query)"p键(key)和值(value), 这些都是大小为 Nx D的矩阵
takes as input query key and value that is matrices of N by D.
15:
Write L;to HBMas the i-thblock of L.
16:end for 然而, 通常在 Transformer网络中
17: Return the outp However, usually in a transformer network,
10:
11:
endfor 我们不仅仅处理一个由d维组成的序列,
12:
Onchip, c
13:
On
14:
Write L;to HBMas the i-thblockof L.
15:
10:
11:
endfor
12:
13:
Write O;to HBMas thei-thblockof O.
14:
Write L;to HBMas the i-thblockof L.
15:
10:
On chip, compute O) = diag(em]-
11:
endfor 而是处理多个这样的序列.
12:
On chip, compute O;=
13:
On chip, c
14:
Write L;to HBMas the i-thblockof L.
15:
10:
11:
On chip, compute O; = diag(c( T)-1o( T)
endfor
12:
13:
Write O;to HBMasthe i-th blockof O.
14:
Write L;to HBMas the i-thblockof L.
15:
10: 这里的小写字母d代表每个注意力头(head)所分配的维度数
15:
Write L;to HBMas the i-thblock of L.
10:
11:
m而我们并不只有一个注意力头, 而是有多个.
endfor
12:
13:
y one head, we have multiple heads.
14:
Write L;to HBMas the i-thblockof L.
15:
10:
11:
On chip, compute O; = diag(c(c)-1o( T)
endfor
12:
13:
Write O;to HBMasthe i-thblockof O.
14:
Write L;to HBMas thei-thblockof L.
15:
10: 因此, 您这里看到的算法是每个注意力头需要执行的任务.
ndfor
13:
14:
15:
Write L;to HBMas the i-thblock of L.
15:
Write L;to HBMas thei-thblock of L 也就是说每个批次中的每个注意力头都应该执行这一算法.
So each head of each batch should do.
16:endfor外
15:
Write L;to HBMas the i-thblock of L. 论分
Moreover, we have seen before when talking about block matrix multiplication,
我们可以并行化输出的计算过程, A " FIX THE X :
A ROW. WE CAN ALSO Fi X
IF ONLY / WE HAD A WAY TO Fi X PSF 因为这里的输出块依赖于查询块以及所有的键块.
because. this. output block here depends on the query one and all the keys.
FOR EACH BLOCK KJ
IF ONL / WE HAD A WAy TO Fi X P 这个输出块依赖于查询组块中的查询与所有键块,
This on je here depends-on the query group block of query two with all the keys,
FOR EACH BLOCK K
IF ONLY WE HAD A WAY TO Fi X 而那个输出块则依赖于查询组块中的查询三与所有键块, 以此类推.
FOR EACH BLOCK J
IF ONL / WE HAD A WAy TO Fi X PSF 由于这个输出块仅依赖于查询组块中的查询一
so'because, this one only depends on query the group, the block query one,
FOR EACH BLOCK KJ
IF ONL / WE HAD A WAy TO Fi X P SFU 而那个输出块仅依赖于查询组块中的查询二
FOR EACH B COCK FOR EACH BLOCK
IF ONL / WE HAD A WAy TO Fi X 它们可以相互独立地工作, 当然它们共享相同的键块.
they can work independently from each other by sharing, of course work, the keys.
FOR EACH BLOCK KJ
我们还需要理解另一关键概念一一共 共享内存.
Another thing that we need to understand about Triton is the shared memory.
13: 在 GPU中, 我们拥有蒿带宽内存, 它类似于计算机中的 RAM.
Soin. the, GPu. we have the, high, bandwidth memory, which is kind of the RAM.
15:
Writ e L;to HBM as the i-thblockof L. 他们会告诉你它拥有40 GB 的显存
so when you buy an A1o0, they tell you that it has 40 GB.
15:
Write L; to HBM as the i-thblock of L.
16:
17:
that's the amount of memory in the high bandwidth memory, so the DRAM.
15:
Write L; to HBM as the i-th block of L.
16:end for
Solet'slook at actually the structure of the Gp U.
如图所示, 这里我们有 DRAM, T 它是 GPU 所具备的大容量内存.
which is here, we have this s DRAM, which is the big memory that the GPU has.
THROUGHPUT 接着, 每个流式多处理器
(我们可以称之为一个线程块)
And then each streaming multiprocessor, so let's call it a block of threads,
FOL M( EED
M( EED edmemory.
实际上,
streaming multiprocessors,
拥有一块称为共享内存的区域
f memory called the shared memory,
它比 DRAM 小得多, 真的非常非常小.
Fo r
FOL M( EED
这两种内存之间有何不同呢?
M( EED
Fo What changes between the s two memories?
FOL M( EED
访问 DRAM 的速度非常慢,:而访问共享内存的速度则极其
to the shared memory is very, very,
FOL M( EED
very fast.
FOL M( EED
CUDA 与 Triton 的一个区别在于
M( EED
Fo Iso one thing that is different between CUDA
M( EED
FOL
当你在 CUDA 中加载某些数据时
information in Cu DA,
个 你是直接从全局内存中加载的
FOL M( EED
因为当我们启动一个 CUD A内核时, 首先, 正如你所记得的
M
Because When we launch a CUDA k
rst of all, as you remember,
在我的 C++代码中, 我们先将张量或向量从 CPU复制到 GPU,
rsfrom, or the vectors from,
它们会驻留在, GPU. 的全局内存中.
global memory of the Gpu.
FOL M( EED
然后我们直接从全局内存中加载这些元素.
from the global memory.
FOL M( EED
个 然而, 访问全局内存通常要慢得多.
个 因此, Flash Attention 的计算
即我们使用torch 实现的原始版本注意力计算
M( EED
Fin its the attention Comput a
n its naive version -
FOL M( EED
非常慢
M ( ZED the one that we can do with the torch -
FOL
FOL M( EED
原因在于访问全局内存的速度极其缓慢.
I memory is very slow.
FOL M( EED
所以我们希望尽可能多地利用共享内存.
le the shared memory.
因此, 我们希望将那些从全局内存加载到共享内存中的元素重复利用
obal memory into the shared memory
Fo R FOL M( EED
这样就不必每次访问全局内存
A
来加载向量或矩阵中的元素了
M( EED
Fto load elements from the vect tors or the matrices.
FOL M( EED
这也是 Triton 中所发生的情况.
M( EED
Fo I And this is what happens also in Triton.
个 在 Triton 中,
M( EED
Fo lso in Triton, whenever
实际上是将信息从全局内存复制到共享内存中.
I memory to the shared memory.
FOL M( EED
is done on the shared memory.
FOL M( EED
而当你存储信息时
FOL M( EED
M you are copying the data from the sh memory to the global memory.
M( EED
FOL
个 这一过程大大提升了速度
M( EED
faster.
因此, 我们始终操作的是那些已加载到共享内存中的元素.
loaded in the shared memory.
FOL M( EED
这块共享内存, 本质上是由同一线程块内的所有线程共享的
FO FO
这块共享内存, 本质上是由同一线程块内的所有线程共享的
this shared memory basically is shared for all th ads that belong to the same block.
FOL M( EED
在 Triton 中, 我们有一个抽象层,
M( EED
traction'level
使我们不必直接与线程打交道
M( EED
而是始终与属于同一线程块
hat belong to the same block,
共享同一块内存的线程组协作.
M( EED
dmemory.
FOL M( EED
因此在 Triton 中, 我们先将数据从全局内存复制到共享内存,
So in Triton, we are copying information I'the global memory to the shared memory,
to re back to the global memory.
FOL M( EED
with Flash Attention.
FOL M( EED
Personal 10 Nov 2024
10 Nov 2024
9 Nov 2024 现在, 让我们来回顾一下 P
Flash Attention B Now let's review the algorithm of flash attention.
15:
Write L;to HBM as the i-thblock of L.
So in flash attention, we have to go an out or for loop, that is among all the,
15:
Write L; to HBM as the i-thblock of L.
16:endfor
between all the query blocks and then an inner loop that is through all the key block.
15:
Write L;to HBM as the i-thblock of L. 在最初的. 后lash. Attention 算法, 即 Flash Attention 1 中
In the original Flash Attention algorithm, the Flash Attention 1,
15:
Write L;to HBM as the i-thblock of L.
the outer block was on the keys and the inner block was on the queries.
15:
Write L; to HBM as the i-thblock of L.
16:end for
17: Return the outpu
This made it less parallelizable.
15:
Write L;to HBMas the i-th blockof L.
16:end for
17: Return the output O and the log sum exp L why Why?
15:
Write L;to HBM as the i-thblock of L.
Because the outer loop is on the gu eries and we have seen before
注意力机制的输出可以针对每个查询块独立计算,
that the output of this attention can be computed independently for each block of queries,
因此并行化变得容易得多=
so it's much easier to parallelize."fx "'te on =to N
SOF TMAX WHILE ITERATi NG OW
因此, 对于这个外循环, 实际上我们并不需要真正运行一个循环,
So this l outer for loop, actually we don't have to'run a for loop,
SOFT MAX WHILE ITERATi NG OW
而是可以启动多个内核, 每个内核负责处理外循环的一次送代.
we just spawn many kernels, each working with one iteration of this outer for loop.
也就是说, 每个内核会处理外循环中不同的查询块.
So each working with a different query block of this outer for loop.
而内循环则是我们必须遍历的部分,
And the inner for loop is something that we have to iterate through,
因此每个 Triton 内核将负责处理一个查询块
so each Triton kernel will work with one query block
并逐一遍历所有的键块.
and then iterates through all the key blocks.
在键块内部, 我们将执行
And inside of this key block, we have already seen the operations that we are going to do,
之前探讨过的那些操作.
which we explored before.
在循环结束时,
And at the end of this for loop,
我们需要将输出结果存储回高带宽内存中
we need to store back the output in the high bandwidth memory.
这就是我们的实现方式.
And this is how we are going to work.
另外需要注意的是, 这里的查询、键和值都是n乘d维的矩阵.
Another thing that we should notice is that this query key and value are n by d.
8:
9:
On chip, compute m
max(m
(j-1)
, row max ( S( D) ∈ RBr, P(j)
(pointwise), c(i)
=em
10:
j-1)
8: 如我之前提到的在transformer模型中y通常处理的不仅仅是一个序列
So as I said before p but usually in a transformer model y wer don't have only one sequence.
10 :
8:
9:
On chip, compute m
max(m
(j-1)
, rowmax( S() ∈ RBr, P(j)
(pointwise ), e(i)
=em
10:
j-1)
10j-1)+ P(j)v;
8:
On chip, compute Si) = Q; KT e RB, x Bc.
9:
(i-1)
10:
So we can also para li elize on the numbe f of sequences that we Rave in the batch 10:
=diag(e
8:
On chip, compute Si) = Q; KT e RB, x Bc.
9:
because each batch-can'work independently from each other.
10:
10
8:
9:
On chip, compute m
max(m
, rowmax( S() ∈ RBr, P(j)
(pointwise ), c(i)
=em
10:
=diag(em
j-1)
10 Jj-1)+ P()v;
and inside each ium and each head, each sequence has multiple heads,
10 :
8:
9:
On chip, compute m
max(m
, rowmax( S() ∈ RBr, P(j)
(pointwise),
=em
10:
j-1)
9:
so each head also can'work independently fr6m each Rother,
10:
=diag(em
8:
On chip, compute S) = Q; KT e RB, x Bc.
9:
On chip, compute m
10:
8:
On chip, compute Si) = Q; KT e RB, x Be.
9:
that's what's the meaning of head, that's what's the meaning of muiti-head attention,
10:
8:
9:
On chip, compute m
ma x(m
, rowmax( S()) ∈ RBr, P(j)
(pointwise), t(i)
=em
10:
j-1)
8:
On chip, compute Si) = Q; KT e RB, x Be.
9:
so that each i head can compute'the attention independently from each other.
10 :
9:
(soiwerwill also parallelize along the head dimension.
10 :
8:
9:
On chip, compute m
ma x(m
, rowmax( S() ∈ RBr, P(j)
(pointwise), t(i)
=em
10:
j-1)
8:
On chip, compute Si) = Q; KT e RBrx Bc.
9:
And, moreover, if you look at this definition of the query block,
10 :
9 :
And, moreover, if you look at this definition'of the guery block,
RB
10:
On chip, compute O) = diag(e"i
-m
(pointwise ),
10:
0n我们会发现查询也可以被分割成多个块.
11:
endfor
we can also split the query into blocks 12:
On chip, compute O;=diag(
(pointwise ),
=emj--m
10:
On chip, comp每个查询块都能独立正作,+ Pvj.
12:
Onchip, compute O;=diag(l
(pointwise ), e(i)
10:
11:
endfor
12:
10:
On chip, compute0生成ag个输出块-1)+ P()vj
11:
end for 12:
Onchip, compute
diag(l;
(pointwise ), e(i)
10:
11:
endfor
12:
(pointwise ),
10:
Onchip, c这就是我们实现并行化的方式. v
11:
endfor This is how we are going to parallelize.
12:
On chip, compute O=diag(
( T)
(pointwise ), 6 因此,, 我们将对批次中的每个序列进行并行化.
10:
11:
t We are going to parallelize each sequence in the batch.
12 :
(pointwise ), e(i)
10:
11:
endfor
12:
(pointwise ), 在每个序列内部我们将对每个头进行并行化;而在每个头内部
inside of eacl ience, we are going to parallelize each head and inside of each head,
12 :
(pointwise ), e(i)
Onchi我们则会对每个查询块进行并行化?
10:
11:
endfor
we are going to parallelize each query block.
12:
On chip, compute=iag()
10
(pointwise ), e(i)
10:
11:
endfor
12:
(pointwise ), e
10: 那么i我们最多能有多少个程序同时并行运行呢?
11:
So
how'many programs we will have working in parallel at most?
12 :
5:
(0)
=(-0∞) BERBr
6:
for 1≤j≤ Tcdo Fo REACH KBLOCK
7:
Load Ky, V; from HBM to on-chip SRAM.
8:
i
5:
6:
7
It will be the number of batches, so the number of sequences in the batch,
(j-1)
5:
(0)
=(-0∞) B, ∈ RBr
6:
7:
Load K, V; from HBM t OPH
8:
On chip, compute s( =so the batch size.
(j-1)
5:
(0)
=(-∞o) B, E RBr
6:
for 1 ≤j≤ Tc do Fo Q EACH Ks BLOCK
7:
Load Kj, V; from HBM to on-chip SRAM.
8:
(i-1)
B
5:
(0)
=(-∞o) B, E RBr
6:
7:
Load K, V
8:
It will be. the batch size, rmultiplied by the number of heads,
(j-1)
Br
5:
(0)
=(-0∞) B. E RBr
6:
for 1 ≤ j≤ Tc do Fo R EACH Ks BLOCK
7:
Load Kj, V; from HBM to on-chip SRAM.
8:
Br
5:
6:
7
multiplied by the number of blocks thats we. will divide the query sequence into.
(j-1)
B
T. d OFOR EACH Q : BLOCK from HBM to on-chip SRAM.
i< Tcdo FOR EACH KBLOCK
Ky, V;from HBM to on-chip S RAM.
T, d OFOR EACH Q : BLOCK I ≤ I. do so let's call it the-til don't know - block size q, the block size q.
Ky, V;from HBM to on-chip SRAM.
T, d OFOR EACH Q : BLOCK from HBM to on-chip SRAM.
i< Tc do FOR EACH Ks BLOCK
Ky, V; from HBM to on-chip SRAM.
T. d OFOR EACH Q : BLOCK 接下来就让我们实际动手编写代码吧
j < I, dall right, now that we have seen this one, let's go actually code it.
K;, V;from HBM toon-chip S RAM.
4:
Load Q from HBM to on-chip SRAM.
5:
for 1≤j≤ Tdo Fo REACH KBLOCK
6:
Load K, V;from HBM to on-chip S RAM.
8:
On chip. compute S)= QK e RBx Be.
9:
4:
Load Q from HBM to on-chip SRAM.
=(-∞) B,∈ RBr
5:
6:
Load K, V;from HBM toon-h
8:
On chip, computes d'I-have already introduced a little bit 9 :
14:
Write O;to HBMas the i-thblock of O.
15:
Write L;to HBMasthei-thblock of L.
Flash Attention 与 Triton 文档中版本的一些区别.
the differences between my implementation of the Flash Attention and the one that you can
14:
Write O to HBM as the i-th block of O.
15:
Write L; to HBM as the i-th block of L.
16:end for 首先,
17: Return the output O and the log sum exp find on the Triton documentation, which is, first of all,
14:
Write O;to HBMasthei-thblock of O.
15:
Write L;to HBMasthei-thblock of L. 因为我认为这对我们的解释来说并不必要,
I don't work with Fp8 because I believe this is unnecessary for our explanation.
14:
Write O;to HBMas thei-thblock of O.
15:
Write L;to HBMas the i-thblockof L
FP8. 3 因为最新的 GPU也支持 FP8.
It's, of course, much faster because the recent Gpus also support Fp8.
14:
Write O;to HBMasthe i-th block of O.
5: 第正个区别
Write L;to HBMasthei-thblock of L. 在 Trit on 网站上的 Flash Attention 中
Second difference is that in the flash attention on the Triton website
14:
Write O to HBM as the i-th block of O.
15:
Write L;to HBMas the i-thblockof L.
16:endfor 反向传播仅针对因果注意力机制实现
17: Return the output U the backward pass is only implemented for the causal attention,
14:
Write O;to HBMasthe i-th block of O.
15:
Write L;to HBMas thei-thblock of L.
16: 的实现中 我同时支持因果和非因果注意力机制
but in my case I implement it for the causal and the non-causal attention,
14:
Write O to HBM as the i-th block of O.
15:
Write L; to HBM as the i-th block of L. 尽管速度会慢一些
16:endfor
17: Return the output O and the log sum exp even if it's slower.
14:
Write O;to HBMasthei-th block of O.
15:
Write L;to HBMas thei-thblock of L. 留
And later, actually, I want to give you an exercise on how to improve it.
14:
Write O; to HBM as the i-th block of O.
15:
Write L; to HBM as the i-th block of L. 第三个主要区别是,
16:end for
17: Return the output O and the log sum exp
And the third difference,
14:
Write O; to HBM as the i-th block of O.
15:
Write L;to HBM as the i-thblock of L.
16:endfor
softmax缩放因子.
17: Return the output O and the log sum exp main difference is that I make explicit use of the soft max scale.
14:
Write O;to HBMasthe i-th block of O.
15:
Write L;to HBMas thei-thblock of L.
So I actually use the scale when needed.
14:
Write O;to HBMasthei-th block of O.
15:
Write L;to HBMas the i-thblockof L.
Triton 7 在线计算 Flash Attention 时,
Another difference is that in the online Triton computation of the flash,
14:
Write O to HBM as the i-th block of O.
15:
Write L; to HBM as the i-thblock of L.
16:endfor
X 并非
e 的× 次方, 而是2的×次方,
attention is this : x is not really e to the power of x, but it's 2 to the power of x,
14:
Write O to HBM as the i-th block of O.
15:
Write L;to HBM as the i-thblock of L 然后通过使用对数来进行补偿.
16:end for
and then they compensate it by using the logarithm.
14:
Write O; to HBM as the i-th block of O.
15:
Write L; to HBM as the i-th block of L.
16:endfor 不过
17: Return the output O and the log sum exp L
However,
14:
Write O to HBM as the i-th block of O.
15:
Write L;to HBM as the i-thblockof L. 这可能是因为2的×次方的实现
16:endfor
17:
because probably the implementation of 2 to the power of x is faster
14:
Write O; to HBM as the i-th block of O.
15:
Write L; to HBM as the i-th block of L 的×次方更快.
16:endfor 比e B
17: Return the output O and the log sum exp L than the e to the power of x.
14:
Write O to HBM as the i-th block of O.
15:
Write L;to HBMasthei-thblock of L.
16:end for 但在我的实现中 我保留了原始的指数计算,
but in my case I retain the original exponential
14:
Write O to HBM as the i-th block of O.
15:
Write L;to HBMas the i-thblockof L 因为我想遵循原算法, 使得代码
16:endfor
17: Return the output O and the log sum exp because I want to follow the original algorithm to make it simpler
14:
Write O;to HBMasthe i-th blockof O.
15: 与 Flash Attention 论文中的算法对照起来更直观易懂.
Write L;to HBMas thei-thblockof L.
16:
to visualize the code along with the algorithm as in the flash citation paper.
14:
Write O to HBM as the i-th block of O.
15:
Write L;to HBMasthe i-thblock of L 那么让我们开始吧.
I know I have created a lot of hype, so let's do it.
我知道我已经吊足了大家的胃口, 那么让我们开始吧.
I know I have created a lot of hype, so let's do it.
OUTLINE TIME LIN
让我们从创建一个新文件开始.
Let'sstart by creating a new file.
OUTLINE TIMELINE
我们将其命名为 program. py.
Let's call it program. py.
OUTLINE TIMELINE
就像我之前介绍 Triton 时那样,
Just like before when I introduced Triton,
TIME LIN
我会先编写使用我们内核的代码
I will start by coding first the code that will use our kernel OUTLINE TIME LIN
然后再编写内核本身.
and then we code the kernel.
OUTLINE TIMELINE
我们这次只编写内核的前向传播部分.
And we will only be coding the forward pass of the kernel.
那么, 我们先导入所需的库
So let'sstart by importing what we need to import,
OUTLINE
也就是 torch 和 Triton.
which is just the torch and the Triton.
OUTLINE
其次, 让我们开始吧, 我先确认一下, 好的.
And secondly, let's start by, let me check, okay.
OUTLINE TIME LIN
Co-pilot已经关闭了, 所以我不用担心这个问题.
Co-pilotis already off, so I don't have to worry about that.
OUTLINE
让我们开始编写代码, 用于测试我们的 Triton 实现
Let'sstart to implement the code that will test our implementation of the triton
并将其与朴素实现的注意力机制进行比较.
and compare it with the naive implementation of the attention mechanism OUTLINE
我们创建用于测试的查询、键和值序列, 其中, 如果你还记得的话,
So we create our query key and value sequence for testing, which is, if you remember,
OUTLINE
查询是批量大小的.
a query is the batch size.
OUTLINE
它的维度是批量大小, 因为我们有多个序列.
It has the dimension batch size because we have multiple sequences.
OUTLINE TIMELINE
每个序列包含多个头, 并且由 SQL en 个标记组成
Each sequence has a number of heads, and it's made up of s QL en tokens,
OUTLINE TIMELINE
每个标记由一个头维数的维度标识.
and each token is identified by a head dim number of dimensions.
OUTLINE TIMELINE
这是因为我们已经将每个标记分割成更小的标记
This is because we have already split each token into smaller tokens,
OUTLINE TIMELINE
每个小标记都有自己的头维度.
each with its own head dimension.
OUTLINE TIMELINE
如果你去掉num Heads 维度, 那么你重新拼接回
If you remove the num Heads dimension, then you put back,
OUTLINE TIMELINE
所有头维度的部分.
you concatenate all the dimensions of this head dim.
TIMELINE OUTLINE
我们使用正态分布来初始化查询、键和值序列.
We initialize the query key and the value sequence by using a normal distribution.
OUTLINE TIMELINE
这段代码我已经从 Triton 的教程中获取, 所以没有什么不同.
This code I already took from the tutorial of Triton, so it's nothing different.
OUTLINE TIMELINE
我们需要梯度
And we require the gradient > TIMELINE OUTLINE
因为我们想计算关于查询、键和值的梯度.
because we want to compute the gradient with respect to query key and value.
OUTLINE TIMELINE
稍后我们会看到原因, 因为我们还想测试反向传播
And we will see later why, because we want to test also the backward pass,
OUTLINE TIMELINE
尽管我们现在不会编写它的代码.
even though we will not be coding it now.
> TIMELINE OUTLINE
所以我们做的第一件事是定义我们的soft max 比例.
So the first thing that we do is we define our soft max scale.
OUTLINE TIMELINE
正如你所记得的, 公式是查询乘以键的转置
which is, as you remember, the formula is query, multiplied by the transpose of the keys TIMELINE OUTLINE
然后除以头维度的平方根.
and then divided by the square root of head dimension.
OUTLINE TIMELINE
所以 dk 或 d head, 有时这么称呼, 然后我们需要
so dkord head, sometimes it's called, and then we need to,
OUTLINE TIMELINE
计算这个值.
so we need to compute this one.
OUTLINE TIMELINE
我们已经可以计算它了.
we can already compute it.
OUTLINE TIMELINE
就是这个:它是头维度平方根的倒数.
it's this : this is one over the square root of the head dimension, um.
OUTLINE TIMELINE
然后我们还定义了do, 稍后我们会看到这是什么.
and then we also define do, and later we will see what is this.
OUTLINE TIMELINE
但这基本上是我们将在反向传播中需要的
but this is basically we will be needed, needed for the backward pass um.
OUTLINE TIMELINE
别担心, 女 如果你不明白do 是什么, 稍后我们会看到它.
don't worry, if you don't understand what is do, later we will see it.
OUTLINE TIMELINE
让我们实现一个简单的注意力机制, 这非常简单:
let's do the naive implementation of the attention, which is very simple, which is :
OUTLINE TIMELINE
首先我们定义掩码
first we define the mask and we use this mask only > TIMELINE OUTLINE
并且只有当我们要计算的注意力是因果注意力时才会使用这个掩码
if the attention we are computing is causal.
> TIMELINE OUTLINE
所以, 正如你所看到的, 我们传递了一个名为causal 的参数
so, as you can see, we pass this parameter called the causal > TIMELINE OUTLINE
它告诉我们是想要计算因果注意力还是非因果注意力
that tells if we want to compute the causal attention or the not causal attention,
> TIMELINE OUTLINE
以及 D类型, 即float16
And the D type, which is float16,
OUTLINE TIMELINE
因为我们希望直接使用16 位浮点数,
because we want to work directly with 16-bitfloating point numbers.
TIMELINE OUTLINE
我们不会使用 FP8, 只是因为我们不想实现它.
We will not be working with FP8 just because we don't want to implement it.
> TIMELINE OUTLINE
我的实现实际上没有 Triton 网站教程中的那么快
implementation is actually not as fast as the one in the tutorial of the Triton website,
> TIMELINE OUTLINE
但我相信它更容易理解
but I believe it's much more easier to comprehend OUTLINE TIMELINE
于是我们定义了掩码.
So we define the mask.
> TIMELINE OUTLINE
我们计算乘积,
We compute the product,
> TIMELINE OUTLINE
即查询(query )与键(key )转置的乘积
the query multiplied by the transpose of the key > TIMELINE OUTLINE
再除以头维度
(head dimension) E 的平方根
divided by the square root of the head dimension.
> TIMELINE OUTLINE
这就是为什么我们要乘以 soft max 的原因
So that's why we are multiplying by soft max.
> TIMELINE OUTLINE
如果我们要计算 的注意力是因果的
(causal ), 那么我们就使用这个已经计算好的掩码
If the attention we are computing is causa, then we use this mask that we have computed.
> TIMELINE OUTLINE
因此, 我们将所有掩码等于零的点积
So we replace all the points,
OUTLINE TIMELINE
替换为负无穷大.
all the dot products where this mask is equal to zero with minus infinities.
TIMELINE OUTLINE
然后, soft max 会将这些负无穷大替换为零,
And then the soft max will replace this minus infinities with zeros.
OUTLINE TIMELINE
因为我们接下来要应用soft max, 而soft max 是按行应用的
because then we are applying the soft max, and the soft max is applied by rows,
OUTLINE TIMELINE
就像普通的注意力机制一样.
just like the normal attention.
OUTLINE TIMELINE
我们继续计算.
we compute, okay.
OUTLINE TIMELINE
接下来我们要做的第二件事是
the second thing that we do is we want to um,
OUTLINE TIMELINE
输出结果是 soft max 的输出与 v 的乘积.
sothe output is the product of the output of the soft max with the v.
OUTLINE TIMELINE
这就是在朴素实现的 Flash Attention 机制中
so this is the reference output on the naive implementation of um flash,
TIMELINE OUTLINE
参考输出的结果.
of the attention mechanism.
> TIMELINE OUTLINE
然后, 我们还需要计算
then we want to compute,
> TIMELINE OUTLINE
输出相对于输入
we want to also derive the gradients of the output with respect to the inputs,
TIMELINE OUTLINE
即 v 、k和q)的梯度.
and in this case it's the, the, the v, the k and the q.
TIMELINE OUTLINE
稍后我们会详细解释这里的具体操作.
later we will see what are we doing here.
TIMELINE OUTLINE
接下来, 我们还需要
then we want also to,
TIMELINE OUTLINE
将这个参考实现与我们的 Triton 实现进行对比.
we want to compare this reference implementation with our triton implementation.
> TIMELINE OUTLINE
那么, 我们开始吧.
so let's do it.
OUTLINE TIMELINE
我们的 Triton 实现将封装在一个名为 Triton Attention 的类中
so our triton implementation will be implemented as a class called triton attention OUTLINE TIMELINE
45
tri_out= Triton A
ention. apply ( Q, K, V, causal, soft max _scale ). half ()
tri _out. backward (do)
tri_dv, V. grad= V. grad. clone (), None
18
tri_d K, K. grad= K. grad. clone(),
8
> TIMELINE OUTLINE
tri _out = Tri to ention apply ( Q, K, V, causal, soft max _scale ). half () 并通过个名为apply 的方法来调用
that we will call using this method called apply,
> TIMELINE OUTLINE
us al, softmax _scale ). half () 稍后我们会详细介绍这个方法, 以及如何传入query 、key 和value 参数
and later we will see what is this method in which we pass the query key and value.
> TIMELINE OUTLINE
tri _dv,
tri _d K 如果我们想计算因果注意力机制
if we want to compute the causal attention,
> TIMELINE OUTLINE
它应该使用soft max 的缩放因子, 并生成一些输出
the soft max scale that it should be using and it should produce some output > TIMELINE OUTLINE
a x_scale). half () 即soft max 的结果与v 相乘后的值. 同时, 我们还可以运行反向传播
which is the output of the softness multiplied by v, then we can run also the backward > TIMELINE OUTLINE
tri_dv,
tri_d K,
V. grad
K. grad 而这个反向传播的计算方式
and this backward will be the same backward > TIMELINE OUTLINE
tri _dv,
tri _d K 将与 Triton Attention 中的一致.
that we will compute with the tritone attention.
TIMELINE OUTLINE
45
46
tri_out. backward (do)
tri_dv, V. grad= V. grad. clone(), None
tri_d K, K. grad= K. grad. clone(),
8
> TIMELINE OUTLINE
接下来, 我们就可以对比我们实现的结果了.
and then we compare, Okay, and then we can compare the result of our implementation.
OUTLINE TIMELINE
tri_d V, V. grad
V. grad. clone(),
None 那么, 这个
Triton Attention. apply 方法与参考实现(即这里展示的这个)进行对比
so this triton attention. apply with the reference implementation, which is this one here.
> TIMELINE OUTLINE
assert torch. all close (ref _o, tri_out,
assert to rch. all atol=atol, rtol=rtol) 这里我们会用到all Close 函数
And this should be, we use the function all close,
TIMELINE OUTLINE
53
52
atol=1e-2
55
assert torch. all close (ref _o, tri_out assert torch.
assert torch 它的作用是逐元素比较两个张量
which basically compares the elements of two tensors > TIMELINE OUTLINE
53
atol=1e-2
55
assert torch. all close (re f_o, tri_out, atol=atol, rtol=rtol 并确保它们的绝对差值不超过设定的值.
and make sure that their absolute difference is no more than this one.
> TIMELINE OUTLINE
53
52
atol=
1e-2
54
assert torch. all close (ref _o, tri_out,
assert torch. all close (ref assert torch. all close (re
assert torch. all close (ref 我们没有采用相对距离
We are not using the relative distance,
> TIMELINE OUTLINE
55
54
assert torch. all close (re f_o, tri_out, atol=atol, rtol=rtol)
assert torch. all close (ref _dk, tri _dk,
assert torch. all close(ref_d Q, tri_d Q, ato 而是直接使用
we are just using the absolute distance between the two elements,
> TIMELINE OUTLINE
torch. all close (re f_o, tri_out,
atol=atol, rtol=rtol) 两个向量中对应元素之间的绝对距离.
corresponding elements of two vectors.
> TIMELINE OUTLINE
atol=atol, rtol=rtol 我们即将构建的这个实现既适用于因果注意力机制
This implementation that we will build will work with the causal attention > TIMELINE OUTLINE
tri _d Q, Q. grad
i_d K,
= Q. grad. clone(),
50 也适用于非因果注意力机制
rto l=e. 0
54
53
atol=1e-2
OUTLINE
55
assert torch. all clos TIMELINE assert
tri _d Q, Q. grad = Q. grad. clone (), N
i_d K,
K. grad= K. grad. clone(), 而之前在 Triton 官网上看到的那个实现仅适用于
while the one. that we saw in. the. website of Triton only works with the.
TIMELINE OUTLINE t torch. all close (ref do. tri do. atol=atol, rtol=rtol)
rch. all close (ref _d V,
前向传播部分实际上同时支持因果和非因果注意力机制
The forward pass actually works with the causal and non-causal.
> TIMELINE OUTLINE
se(ref d Q, tr i d Q. ato l =atol, rtol=rtol)
rtot=rtol
而后向传播部分仅适用于因果注意力机制的情况.
while the backward pass only Works in the case of the causal attention.
> TIMELINE OUTLINE all close(ref do. tri do. ato l=atol, rtol=rtol)
tri_d V, atol=a
rtol=rtol
tri _d K,
tri_d Q, Q. grad = Q. grad. clone(),
K. grad = K. grad. clone (), 确实如此, 官网上的实现经过了高度优化.
to Okay, but its highly optimized, the one online,
> TIMELINE OUTLINE re f_d V, tri_dv, ato l=ato t,
d0. tri do, atol=atol, rtolar to l )
tri _dk,
tri_d Q, Q. grad = Q. grad. clone(),
K. grad = K. grad. clone (), 如果你想学习更多优化 Triton 内核的技巧
so if you want to. learn a little more. tricks on how to optimize Triton kernels > TIMELINE OUTLINE t torch. all close (ref do, tri do. atol=atol, rtol=rtol)
h. all close (re f_d V, tri_dv, atol=atol, rtol=rtol
tri _d Q, Q. grad = Q. grad. clone(),
tri_d K, K. grad= K. grad. clone(),
50
#compare 那里蕴含着丰富的知识.
52
rto l=0. 0
54
53
atol=1e-2
OUTLINE
55
assert torch. all close (re f_dv, tri_dv, atol=atol, rtol=rtol)
> TIMELINE assert ssert torch. all close (re f do. tri do. atol=atol, rtol=rtol)
tri_dk, 好的各位,
tri_d Q, Q. grad= Q. grad. clone (), 现在我们试着来实现这个 Triton Attention, 至少先完成前向传播部分
Anyway guys, now. let's'try to. implement. this Triton Attention, at least the forward pass.
> TIMELINE OUTLINE
ref_d V, tri_d V
d Q. tri do. atol=atol, rtol=rtol)
atol=atol, rtol=rtol
tri _d K
tri_d Q, Q. grad = Q. grad. clone(),
K. grad = K. grad. clone (), 那么, 我们就开始动手实现这个 Triton Attention 类吧
So. lets. go to implement this Triton Attention class.
> TIMELINE OUTLINE torch. all clo all close (re f do, tri do. atol=atol, rtol=rtol)
triton 4
tri_out = Triton Attention. apply ( Q, K, V, causal, soft max _scale ). half ()
tri _dv, V. grad = V. grad. clone(), None
tri_d K, K. grad = K. grad. clone (),
None 8
49
tri_do, Q. grad= Q. grad. clone(), None > TIMELINE OUTLINE 51
52
rtol =0. 0
re f_d K, K. grad = K. grad. clone(),
ref_d Q, Q. grad = Q. grad. clone (), None #triton implementation tri _out = Triton Attention. apply ( Q, K, V, causal, soft max _scale ). half ()
tri _out. backward (do)
8
49
trldv, vgrad= V. grad. clone (), None tri _d K, K. grad = K. grad. clone (),
> TIMELINE OUTLINE 52
tri_d Q, Q. grad= Q. grad. clone), None None
re f_dk, K. grad = K. grad. clone(),
ref_d Q, Q. grad = Q. grad. clone (), None 好的, 这里需要注意的是, 每次你想在 Torch 中引入一个新的操作时
Okay, here, every. timeyou want to introduce a new operation into Torch,
> TIMELINE OUTLINE
re f_dk, K. grad = K. grad. clone(),
ref_d Q, Q. grad = Q. grad. clone (), None #triton implementation 都需要继承..
tri _out = Triton Attention. apply ( Q, K, V, ca
tri _out. backward (do)
tri_dv, V. grad= V. grad. clone(), None
Nyou need to derive the.
OUTLINE tri _d Q, Q. grad= Q. grad. clone(), Nor
tri_d K, K. grad= K. grad. clone(),
> TIMELINE
52
re f_dk, K. grad = K. grad. clone(),
ref_d Q, Q. grad = Q. grad. clone (), None 你需要通过继承auto grad. function 类来实现你的自定义操作
you need to implement your operation by deriving from this auto grad. function class.
> TIMELINE OUTLINE
ref_d K,
ref_d Q, Q. grad = Q. grad. clone (), None K. grad = K. grad. clone (), 因此, To rch 中的每一个操作 无论是softma XRe LU、 Z
Zui Glu还是其他任何操作
So every operation in Torch-actually, if it's the soft max or it's the - I don't know -
> TIMELINE OUTLINE
ref_d Q, Q. grad= Q. grad. clone(), None
#triton tri _out 实际上都是通过继承自function 的类
the Re LU or the Zu i Glu or whatever there is, it is always implemented as a function,
> TIMELINE OUTLINE
re f_d Q, Q. grad = Q. grad. clone(), Noe
ref_d K, K. grad = K. grad. clone (),
#triton implementation 来实现的.
trl d. xras a class that derives from this function.
tri _dv, V. grad= V. grad. clone(), Non
tri_out. backward (do )
OUTLINE TIMELINE 52
tri_d Q, Q. grad
re f_dk, K. grad = K. grad. clone(),
ref_d Q, Q. grad = Q. grad. clone (), None #triton implementation tri _out = Triton Attenti 并且它需要提供两个方法:
tri _dv, V. grad tri _out. backward (do )
And it should provide two methods.
OUTLINE tri_d K, K. grad=
tri_d Q, Q. grad
TIMELINE
14
ref_do, Q. grad= Q. grad. clone (), None
ref _d K, K. grad = K. grad. clone (), 称为前向传播
(forward pass )另一个称为反向传播
(backward pass )
one called ·the forward pass and one called the backward pass.
> TIMELINE OUTLINE
re f_dk, K. grad = K. grad. clone(),
ref_d Q, Q. grad = Q. grad. clone (), None triton in ple 前向传播负责生成该操作的输出
OUTLINE forward should produce the output of this operation > TIMELINE
re f_dk, K. grad = K. grad. clone(),
ref_d Q, Q. grad = Q. grad. clone (), None #triton in plementation 而反向传播则计算损失函数
and the backward should compute the gradient of the loss ard(d0)
> TIMELINE OUTLINE
re f_dk, K. grad = K. grad. clone(),
ref_d Q, Q. grad = Q. grad. clone (), None #triton implementation tri _out = Triton At tent 相对于该操作输入的梯度.
tri k. krwith respect to the input of that function.
tri _dv, V. grad tri _out. backward (do )
> TIMELINE OUTLINE tri_d Q, Q. grad
re f_dk, K. grad = K. grad. clone(),
ref_d Q, Q. grad = Q. grad. clone (), None triton tri _out 稍后我们会详细子解它的工作原理. 的
tri _out.
tri _dv,
OUTLINE tri_dk,
tri_d Q, Q. grad
K. grad
and later we will see how that works.
> TIMELINE
ref_d K,
ref_d Q, Q. grad= Q. grad. clone (), None K. grad = K. grad. clone (), 现在我们先把注意力集中在前向传播上
ri_d
for now s let'sconcentrate on the forward pass.
> TIMELINE OUTLINE
ref_d K,
ref_do, Q. grad= Q. grad. clone (), None K. grad = K. grad. clone (), 要实现前向传播我们需要创建一个名为forward 的静态方法
to implement the forward pass, we need to create a static method that is called forward,
> TIMELINE OUTLINE
ref_o. backward(do) 它接收个名为context的输入参数.
49
twhich. takes as. input-one. thing called the context.
> TIMELINE OUTLINE tri_dv, V. grad= V. grad. clone(), No
ref _o. backward (do ) 正如你所知;在自动求导(auto grad )中, 训练神经网络时,
so, as you know, in the auto grad, in when training neural networks,
> TIMELINE OUTLINE tri_dv, V. grad= V. grad. clone(), None
ref_o=torch. matmul( P, V)
ref_o. backward (do) 我们会涉及前向传播和反向传播两个过程.
trlt we have the forward pass and the backward.
> TIMELINE OUTLINE tri_dv, V. grad= V. grad. clone(), No
ref_dv, V. grad = V. grad. clone(),
ref_o. backward (do )
re f_d Q, Q. grad = Q. grad. clone()
ref_d K,
K. grad= K. grad. clone 在计算反向传播时
49
trlot-tritttwhen computing the backward pass,
#triton in plement ati > TIMELINE OUTLINE 5
tri_dv, V. grad = V. grad. clone (), None
tri _out. backward (do
ref _o. backward (do ) 我们需要复用前向传播过程中每个计算节点的激活值
we need to reuse the :activations of each of the computation nodes during the forward pass > TIMELINE OUTLINE tri_d V, V. grad
V. grad. clone(), No
ref _o. backward (do ) 而这个context 的作用正是让我们保存这些激活值的信息
and this context basically allow us to save the information > TIMELINE OUTLINE tri_dv, V. grad
V. grad. clone(), N
ref_dv, V. grad= V. grad. clone(
ref_o. backward (do )
ref _d K, K. grad = K. grad. c
ref_d Q, Q-grad= Q. grad. c 以便在反向传播时使用.
to for the necessary activations that we will need during the backward pass.
> TIMELINE OUTLINE V. grad V. grad. clone (), N
ref_0=torch. natmul( P, V)
ref_o. backward (do ) 稍后我们会在 Flash Attention 算法中看到
And later we will see in the flash attention algorithm > TIMELINE OUTLINE tri_dv, V. grad = V. grad. clone(), No
ref _o. backward (do ) 为了计算反向传播, 我们需要保存哪些信息.
what information we need to save-in order to compute the backward pass.
> TIMELINE OUTLINE tri_d V,
V. grad
V. grad. clone(), No
ref _o. backward (do )
ref _dv, v. grad
ef_d K,
K. grad 例如;我们需要保存哪些内容.
ref_d Q, Q-grad
50
49
#triton impl For example, what we will need to save.
OUTLINE tri _out = Triton tri _out. backwar > TIMELINE tri_dv, V. grad= V. grad. clone(),
ref_o=torch. matmul( P, V)
ref_o. backward (do ) 在反向传播过程中, 我们需要实时重新计算
During. the backward pass we will need to recompute on the fly > TIMELINE OUTLINE, V. grad = V. grad. clone (), N
ref _o. backward (do ) 每个块的查询与键的转置相乘结果
the query multiplied by'the'transport of the keys for each block,
> TIMELINE OUTLINE
ref _o. backward (do ) 但我们不希望重新计算每一行的归一化因子或最大值
but we don't want to recompute the normalization factor or the maximum value for each row > TIMELINE OUTLINE tri_dv, V. grad
V. grad. clone(), No
ref _o. backward (do)
ref_dv, v. grad
ref_dk, K. grad
V. grad. clon 因此我们会保存这两个值.
ref_d Q, Q. grad= Q. gr
50
49
#triton implementation so we will save those two values.
OUTLINE tri _out = Triton Attenti o
tri_out. backward (do )
> TIMELINE tri_d V, V. grad= V. grad. clone(),
ref _o. backward (do )
ref_dv, v. grad 实际上, 我们不会保存两个值
ref_d Q, Q-grad
trilot-tr it And actually we will not save two values,
triton imp > TIMELINE OUTLINE tri_d V, V. grad= V. grad. clone(),
ref_0=torch. matmul( P, V)
ref_o. backward (do ) 而是通过一 种称为log Sum E×p的技巧来保存一个值, 稍后我们会详细讲解这个技巧
we will save one value with a trick called the log Sum Exp trick that we will see later.
> TIMELINE OUTLINE tri_d V, V. grad= V. grad. clone(), No
ref _o. backward (do ) 总之:这context 就像是一个存储区域
Anyway, this context is just a kind of a storage area > TIMELINE OUTLINE tri_dv, V. grad
V. grad. clone(), N
ref _o. backward (do ) 我们可以在这里保存一些必要的数据, 以便在反向传播时重新计算使用
where we can save :some stuff that will be necessary for us to recompute the backward.
> TIMELINE OUTLINE tri_d V, V. grad
V. grad. clone(),
ref _o. backward (do)
ef_dv, 而且你可以根据需要保存任何数据,
49
#triton implem
And you can save whatever you like.
OUTLINE tri _out = Triton Att tri _out. backward (do )
> TIMELINE tri_dv, V. grad= V. grad. clone(), N
ref_o=torch. matmul( P, V)
ref_o. backward (do ) 接下来是这个操作的输入, 包括查询、 键和值
then we have. the input of this operation, which is the query key and value,
> TIMELINE OUTLINE tri_dv, V. grad
V. grad. clone(), Nor
ref_o=torch. natmul( P, V)
ref_o. backward (do ) 它们都是张量;并且带有因果性(causal )属性.
trl t-tri to which is. three ·tensors, with the causal.
triton imple > TIMELINE OUTLINE tri _dv, V. grad = V. grad. clone(), No
tri_out. backward (
ref_o=torch. matmul( P, V)
ref_o. backward (do ) 如果我们要计算因果注意力(causal attention )
trio if we are going to compute the causal attention > TIMELINE OUTLINE 52
tri_dv, V. grad = V. grad. clone (), None
ref_o. backward(do) 还需要指定softmax的缩放因子
50 的
triot :and the soft max scale ·that we should apply,
TIMELINE OUTLINE tri _dv,. grad= V. grad. clone(),
tri_out. ba
ref _o. backward (do ) 这个缩放因子通常是头维度:(head dimension )的平方根的倒数
based on the one over the square root of the head dimension,
> TIMELINE OUTLINE tri_dv, V. grad = V. grad. clone(), None
rtol=0. 0
atol=1e-2 实际上, 我们也可以通过检查张量的形状动态计算这个值
which we could also compute it on the fly, actually, by the way,
> TIMELINE OUTLINE
P=torch. softmax( P. float(), din=-1). half ()
re f_o= torch. matmul( P, V)
ref_dv, V. grad = V. grad. clone (),
ref _o. backward (do )
ref _d K, K. grad = K. grad. clone (), 不过这并不是重点.
by checking the shape of this, but okay, it doesn't matter Anyway.
ref_d Q, Q. grad= Q. grad. clone(),
TIMELINE OUTLINE
tri _out. backward (do ) 因此, 我们首先要做的是提取这些张量的形状
so the first thing'that we are going to do is to extract the shapes of these objects > TIMELINE OUTLINE
tri _out. backward (do)
tri_dv, V. grad
tri_dk,
tri_d Q, Q. 并确保它们的形状符合我们的预期
and make sure all the shapes are what we expect them to be.
> TIMELINE OUTLINE assert torch. all close (re f_o, tri_out, atol=atol, rtol=rtol)
ref_o=torch. matmul( P, V)
ref_o. backward (do)
ref_dv, V. grad= V. grad. clo 查询、 键和值的形状是:
49
#triton in plementation ref_do, Q. grad= Q. grad. clo
So the shape of the query key > TIMELINE OUTLINE 52
tri_out = Triton Attention tri _out. backward (do )
#reference implementation torch. trit(torch. ones (( SEQ _ LEN, SEQ_ LEN), device=cuda) 批次大小×头数量×序列长度×头维度.
K. t
se(2. 3)
and value is a batch size by number of heads by sequence length by head dimension.
> TIMELINE OUTLINE grad. clone ()
reference implementation MASK = torch. tril(torch. ones(( SEQ_ LEN
EQ_ LEN), device 我们需要确保查询、键和值的头维度是一致的
We make sure that the head dimension matches for the query key and value.
> TIMELINE OUTLINE
MASK = torch. tril (torch. ones (( SEQ _ LEN, SEQ_ LEN), dev #reference implementation 它们必须匹配因为每个向量的尺寸应该相同
K. tra
se(2. 3))
> TIMELINE OUTLINE. grad. clone (),
#reference implementation ones (( SEQ _ LEN MASK =torch. trit(torch.
SEQ _ LEN ), device ="cuda ") 接着, 我们预先分配输出向量的空间, 用于存储计算结果.
> TIMELINE OUTLINE ref _d K,
d0= torch. randn_like( Q) # Needed for the backwards pass MASK =torch. tril (torch. ones (( SEQ _ LEN )
reference in plementation P =torch. matmul( Q, K. transpose(2, 3 正如你所记得的
49
18
if causal:
P[,, MASKm0]float("-inf
50
P=torch. so ftmax( P. float(), din=-1).
Soas you remember,
> TIMELINE OUTLINE 52
ref_o. backward (do )
reference impler 注意力机制中的输出与查询
=torch. matmul( Q,
fcausal :
> TIMELINE OUTLINE ref _o. backward (do )
d0 = torch. randn_like( Q) # Needed for the backwards pass MASK = torch. tril (torch. on
#reference implementation 47
P=torch. matmul( Q, K. tran 建和值序列的形状相同.
49
18
if causal :
OUTLINE P =. torch. soft na x( P. float (), dim key and value sequence.
> TIMELINE 52
ref_o. backward (do )
这里需要提醒的是, 查询、 键和值序列
where the guer y ·key and value sequence, i want to remind you,
> TIMELINE OUTLINE ref _o. backward (do )
并不是注意力机制输入中的原始序列(即token序列)
sis not. the guer y key and value of the input, of the attention > TIMELINE OUTLINE ref _o. backward (do )
而是已经经过w qwk 和wv 变换后的输出
which is a sequence of ·tokens y but it's the output already of the wq, wk and wv,
> TIMELINE OUTLINE ref _o. backward (do )
因为 Flash Attention 并不关注优化这些矩阵乘法的计算
because flash attention is not concerned with optimizing those metrics, multiplication,
> TIMELINE OUTLINE ref _o. backward (do )
而只关注wqwk和wv变换后的输出
P trhrbut only-the output of the wq, wk and wb.
> TIMELINE OUTLINE 52
ref_o. backward (do )
因此我们预先分配一个输出张量来存储结果
So we pre-allocate the output tensor where we will store this output > TIMELINE OUTLINE ref _o. backward (do )
这个张量的形状与查询、键和序列矩阵相同
which has the same shape as the query, key and sequence matrix.
> TIMELINE OUTLINE ref_o. backward(d0)
d0 = torch. randn_like( Q) # Needed for the backwards pass
14
#reference implementation MASK =torch. tril (to rch. ones(( SEQ_
P=torch. matmul( Q, K. transpose ( 实际上, 并非如此.
49
if causal:
50
P=torch. sof tmax( P. float(), din=-1)
Actually, no, not true.
> TIMELINE OUTLINE 51
52
ref_o. backward (do )
MASK =torch. tril refere
nce imple 实际上, 它的形状与查询相同
f causal :
Actually, it has the same shape as the query,
> TIMELINE OUTLINE 51
5
ref_o. backward (do )
reference imple r MASK = to rch. tril(t
torch. matmul( Q, 但可能与键和值的形状不同
49
fcausal :
but it may not be the same as the key and value.
> TIMELINE OUTLINE ref _o. backward (do )
d0 = torch. randn_like( Q) # Needed for the backwards pass 4
MASK= torch. trit(torch. ones(( SEQ _ LEN, SEO_ LEN), devic #reference implementation 48
47
P=torch. matmul( Q, K. transpose (2, 3))*softmax_scale
if causal:
99
P =torch. soft max( P. float (), din=-1). half()
P[:,, MASK mm θ]=float(-inf ")
Why?
> TIMELINE OUTLINE 51
52
ref_o =torch. matmul( P, V)
ref_o. backward (do )
因为存在一种称为交叉注意力的机制, 其中查询、键和值来自不同的来源
Because there is this thing called cross-attention where the Query, key and value are.
> TIMELINE OUTLINE ref _o. backward (do )
MASK = torch. tril (torch #reference in plementation P =torch. mat m ul( Q, K. t 它们的转置形式各不相同
49
50
if causal:
P[:,, MASKm0]=float (
transposition are different.
OUTLINE P =torch. soft nax( P. float(),
> TIMELINE
52
ref_o. backward (do )
通过
W Q、 WK、 WV 进行的投影并非来自同一个输入序列, 而是两个不同的序列
projection through W Q, WK, WV,'not of the same input sequence but of two sequences.
> TIMELINE OUTLINE ref _o. backward (do )
因此, 当查询来自一个序列, 而键和值来自另一个序列
So cross-attention happens-when we have a query that comes from one sequence > TIMELINE OUTLINE ref _o. backward (do )
MASK = torch. tri l(tor reference in plem 并且它们分别通过各自的 WK
torch. matmul( Q,
fcausal :
and the key and value come from r another sequence and they pass through their own Wk > TIMELINE OUTLINE ref _o. backward (do )
WV 进行投影时, 就会发生交叉注意力. 此时, 它们的序列长度可能并不相同
and w V and ·they may not have the same sequence length.
> TIMELINE OUTLINE ref _o. backward (do )
reference in ple
MASK= torch. tril(to 因此注意力机制输出的形状
50
49
f causal :
So the shapes of the output of the attention > TIMELINE OUTLINE 52
ref_o. backward (do )
MASK = torch. tril (to rc reference in plement at i 仅取决于查询序列的形状
torch. matmul( Q, K
fcausal:
49
only depends on the shape of the query sequence,
> TIMELINE OUTLINE ref _o. backward (do )
reference imple 47
MASK=torch. tril(t 而与键和值序列的形状无关.
99
48
lf causal :
P torch. soft max p. rnot of ·the key and value sequence.
> TIMELINE OUTLINE 52
ref_o. backward (do )
reference impl
MASK=torch. tril 这种情况发生在交叉注意力中
99
f causal :
P[:,, MASK OUTLINE P = torch. softmax( P.
ref_o=torch. natmu
this happens during cross attention,
> TIMELINE 52
ref_o. backward (do )
但在语言模型中, 我们通常处理的是自注意力机制
but usually in ·language models we always work with the self attention,
> TIMELINE OUTLINE ref _o. backward (do )
因此这种情况不会发生至少在有因果关系的语言模型中不会出现
so that should not happen, at least in the causal language models.
> TIMELINE OUTLINE ref _o. backward (do )
接下来我们进入这个阶段, 稍后我们会详细探讨这个阶段的具体内容.
> TIMELINE OUTLINE ref _o. backward (do )
soft max_scale=1/( HEAD_ DIM**θ. 5)# QK~t/sqrt( HEAD_ DIM)
d0= torch. randn_like( Q)# Needed for the backwards pass 单来说, 这个阶段只是一个标识数字, 用于指示我们接下来要执行的操作
basically, the stage, it's just a number that tells if the operation > TIMELINE OUTLINE torch. nat mul ( P, V)
Soft max_scale=1/( HEAD_ DIM**θ. 5)# QK~t/sqrt( HEAD_ DIM)
d0= torch. randn_like( Q)# Needed for the backwards pass 是用于因果注意力还是非因果注意力.
that we are going to do later is for the causal attention or for the not causal attention,
> TIMELINE OUTLINE nat mul ( P, V)
Soft max_scale=1/( HEAD_ DIM**θ. 5)# QK~t/sqrt( HEAD_ DIM)
d0= torch. randn_like( Q)# Needed for the backwards pass #reference imple torch. tril 然后我们需要定义启动网格.
99
if causal
torch. matmu
And then we need to define our launch grid.
OUTLINE > TIMELINE ref_o=
torch. matmul( P, V)
d0= torch. randn_like( Q)# Needed for the backwards pass 启动网格告诉我们 Triton 需要启动多少个并行进程.
The launch grid tells us-how many parallel process we need to be launched by Triton.
> TIMELINE OUTLINE ref_o=torch. natmul( P, V)
req 实际上, 这些进程将由 CUDA 启动
soft na x_scale=
Actually ;they will be launched by Cu DA > TIMELINE OUTLINE
们始终将 Triton 作为与 CUDA 交互的接口, 因此可以说是由 Triton 来启动
but we always work with Triton as an interface to cu DA, so by Triton.
> TIMELINE OUTLINE
正如我之前提到的, 在 Triton 中, 我们希望沿着批次维度进行并行化处理
So in Triton, as I said before, we want to parallelize along the batch dimension.
> TIMELINE OUTLINE
因此, 批次中的每个序列都应该独立运行, 互不干扰.
So each batch, each sequence in the batch should work independently from each other.
> TIMELINE OUTLINE
不仅如此, 批次中每个序列内部的每个注意力头
d Not only each inside of each sequence in the batch > TIMELINE OUTLINE
normal _(mean=o. 0, s
. requires _grad _ 也应该独立运行, 互不影响.
each head should work independently from each other.
> TIMELINE OUTLINE
因此, 至少我们有批次大小乘以注意力头数量的程序在运行.
So at least we have a batch size multiplied by number of heads programs.
> TIMELINE OUTLINE
对于每一个这样的程序, 我们还引入了另一个维度
And for each r of this program, we have another dimension called the,
> TIMELINE OUTLINE
normal _(mean=0. 8, std
. requires _grad_ 即将查询划分为多个查询块.
99
softnax_scale=1
rwedivide the query into blocks of queries.
OUTLINE
d0=torch.
> TIMELINE
正如您所记得的, 在讨论块矩阵乘法时
So as you remember, when talking about a block matrix multiplication,
TIMELINE OUTLINE
我们并不直接操作原始的查询矩阵.
we don't ·work with the query as the original matrix query matrix > TIMELINE OUTLINE
处理每个查询
(即单个向量或单个标记), 而是以查询组为单位进行操作
So where each-query is one vector or one token, we work with group of queries.
> TIMELINE OUTLINE
因此, 每个查询块代表查询序列中的一组标记.
So each ·block of queries is a group of tokens in the query sequence.
TIMELINE OUTLINE
这意味着我们计划在两个维度上启动多个内核
So we are'saying that we want to launch a number of kernels OUTLINE TIMELINE
normal _(mean=@. 0,
requires _grad _ 或线程块, 也就是一组线程.
or blocks of threads or a group of threads along two dimensions.
> TIMELINE OUTLINE
正如 CUDA 内核可以沿着 X和 Y两个维度启动一样
So just like the Cu DA kernel can be launched a long two dimensions X and Y,
> TIMELINE OUTLINE
这里我们也是沿着两个维度来启动程序.
here we-are launching programs along two dimensions.
TIMELINE OUTLINE
其中一个维度指明了我们将要处理的是哪个批次
soft max _scale One dimension that tells us which batch > TIMELINE OUTLINE
normal _(mean=o. 8, std=0. 5)
. requires _grad _() 的哪个注意力头.
which head of which batch we are going to work with.
> TIMELINE OUTLINE
requ 即具体是哪个批次的哪个注意力头.
5oftmax_scale=1/( HEAD
Sowhich head of which batch.
OUTLINE ndn _ Like ( Q )
> TIMELINE
我们接下来要处理的是哪个批次的元素?
soft nax _scale
rbatch element are we going to work with?
OUTLINE
d0=torch. r
> TIMELINE
( BATCH _ SIZE, NUM _ HEADS, SEO _ LEN, HEAD _ IM), dtype 在这里面, 我们会确定:好的, 这是一个序列,
and inside this we are going to say :okay, this is a sequence,
> TIMELINE OUTLINE
requ. 接下来我们要处理的是哪一组查询?
OUTLINE
0
which group of queries are we going to work with?
TIMELINE
14
normal_(mean=0. 0,
requires _grad_() 我们要处理的是哪一部分?
8
99
48
softnax_scale=1/
are we going to going to work with?
TIMELINE OUTLINE 52
总的来说, 查询组具体指的是什么呢?
of tna x So-overall and the group of gu eries is what?
OUTLINE TIMELINE
序列长度是否除以了我们想要分组在一起的查询数量.
Is the sequence length divided by the number of queries that we want to group together.
> TIMELINE OUTLINE
因此, 块大小的立方告诉我们每个查询块中有多少个查询.
So the block size cube tells us how many queries are there in each block of queries.
> TIMELINE OUTLINE
( BATCH _ SIZE, NUM _ HEADS, SEQ_ LEN, HEAD_ DIM) 因此, 这里的cdiv指的是向上取整的除法运算
softnax_scale =
So this c div is just the ceiling division.
> TIMELINE OUTLINE 51
require 所以它等于, 让我在这里写下来,
49
softnax_scale=1
Soit is equal to, let me write it here.
> TIMELINE OUTLINE
or ch. enpty (
( BATCH _ SIZE, NUM _ HEADS, SEQ_ LEN, HEAD_ DIM), dtype=dtype, 这等于序列长度除以块大小 Q后向上取整的结果
This is equal to'ceiling of sequence length divided by the block size Q.
> TIMELINE OUTLINE
torch. en pty (
( BATCH _ SIZE, NUM_ HEADS, SEQ_ LEN, HEAD_ DIM), dtype normal _(mean requires _grad 这告诉我们有多少个 Q的块.
sof tax sca This tells us how many blocks of Q we have.
> TIMELINE OUTLINE
d0=torch.
44
( BATCH_ SIZE, NUM_ EADS, SEO_ LEN, HEAD _ DI M), dtyp. normal _(mean=0. 0,
requires_grad_() 那么, 让我们来复习一下.
8
99
48
So, let'srehearse.
OUTLINE
sof tnax_scale=1/( HEAD_ DIM**0. 5)# QK~t
d0= torch. randn _like ( Q )# Needed for > TIMELINE 52
h. enpty (
( BATCH _ SIZE, NUM _ HEADS, SEQ_ LEN, HEAD_ DIM), dty P 我们有个张量 Q, 其维度为批量大小乘以头数
We have a tensor that is Q, that is batch size by number of heads,
> TIMELINE OUTLINE
( BATCH _ SIZE, NUM_ HEADS, SEQ_ LEN, HEAD_ DIM), dtype =dtype, 每个 Flash Attention 算法将处理以下内容
TIMELINE OUTLINE
45
normal_(mean=e. e, std=0. 5)
( BATCH_ SIZE, NUM_ HEADS, SEQ_ LEN, HEAD_ DIM), dtyped type,
requires _grad_() 序列长度和头维度.
49
50
48
softmax_scale=1
the sequence length head dimension.
TIMELINE OUTLINE 52
d0=torch. randn_like
( BATCH _ SIZE, NUM_ HEADS, SEQ_ LEN, HEAD_ DIM), dtype =dtype, 此外, 我们已经了解到 Flash Attention 包含两个循环
Moreover, we have seen that the flash attention has two loops.
> TIMELINE OUTLINE
torch. en pty (
( BATCH _ SIZE, NUM_ HEADS, SEQ_ LEN, HEAD_ DIM), dtype=dtype 另一个是遍历所有键块的内层循环.
sof tmax s one is the innerloop along all the key block > TIMELINE OUTLINE
( BATCH _ SIZE, NUM _ HEADS, SEQ_ LEN, HEAD_ DIM), dty
. enpty 我们已经知道, 查询块之间可以独立工作
We have seen that the query block can work independently from each other,
> TIMELINE OUTLINE
( BATCH _ SIZE, NUM _ HEADS, SEQ_ LEN, HEAD_ DIM), dtyp 因此我们可以启动与查询块数量相同的并行程序
so we can spawn :as many programs in parallel as there are number of blocks of queue > TIMELINE OUTLINE
( BATCH _ SIZE, NUM _ HEADS, SEQ_ LEN, HEAD_ DIM), dtyp
46
45
. normal _(mean =0. e, std 因为它们能够并行处理
48
requires_grad_()
99
sof tua sale because they can work in parallel.
OUTLINE TIMELINE 52
d0=torch. randn_like( Q)
( BATCH _ SIZE, NUM _ HEADS, SEQ _ LEN, HEAD _ DIM ), 因此, 这个网格告诉我们有多少个程序可以并行运行.
So this grid tells us how many programs there are that can work in parallel.
OUTLINE TIMELINE
torch. en pty (
( BATCH _ SIZE, NUM_ HEADS, SEQ_ LEN, HEAD_o IM), dtype =dtype,
nor nal _(me a requires _gra 随后,(
GPU 将根据其资源情况
Then-it will be the GPU that, based on its resources,
> TIMELINE OUTLINE
torch. en pty (
( BATCH _ SIZE, NUM_ HEADS, SEQ_ LEN, HEAD_ DIM), dtype 决定实际并行运行多少个程序.
requires _gra will decide how many programs actually to work in parallel > TIMELINE OUTLINE
如果资源充足,"能够使所有程序并行运行, 那再好不过.
> TIMELINE OUTLINE
( BATCH _ SIZE, NUM _ HEADS, SEQ_ LEN, HEAD_ DIM), dtyp 如果资源不足以支持所有程序并行运行
If it doesn't have enough resources to make them work in parallel,
> TIMELINE OUTLINE
torch. en pty (
( BATCH _ SIZE, NUM_ HEADS, SEQ_ LEN, HEAD_ DIM), dtype =dtype, device GPU 则会按顺序逐一启动这些程序
:it will-launch them sequentially, one after another.
> TIMELINE OUTLINE
( BATCH _ SIZE, NUM _ HEADS, SEQ_ LEN, HEAD_ DIH), dtype 最后-个维度类似于 CUDA启动网格中的z维度
8
soft max _scale =1/( HEAD_ DIM**. 5)# QK~t/sqrt( HEAD_ DIM)
and > TIMELINE OUTLINE 52
d0= torch. randn_like( Q)# Needed for the backwards pass
( BATCH _ SIZE, NUM _ HEADS, SEQ_ LEN, HEAD_ DIH), dtype 最后-个维度类似于 CUDA启动网格中的z维度
the last dimension is :this is like the z dimension in the CUDA, in the CUDA launch grid.
> TIMELINE OUTLINE
( BATCH _ SIZE, NUM _ HEADS, SEQ_ LEN, HEAD_ DIM), dtyp 但我们并不打算使用它, 因为不需要额外的并行层级.
and we don't want-to use it because we don't want an additional level of parallelism.
> TIMELINE OUTLINE
( BATCH _ SIZE, NUM _ HEADS, SEQ _ LEN, HEAD _ DIM ) 好的, 这就是我们的启动网格. 我们将启动一定数量的程序, 即
all right, this is our launch grid, so we will launch a number of programs, that is.
> TIMELINE OUTLINE
这个数值, 代表并行程序或并行内核的数量.
this one, a number of programs, of parallel programs, or number of parallel kernels,
> TIMELINE OUTLINE soft nax_scale
*0. 5)# QK~t/sqrt( HEAD_ DIM)
Triton 中每个内核工作由一个线程组完成
and each kernel in Triton work is a group of threads,
> TIMELINE OUTLINE
Sof tmax _scale = 1 /( HEAD_ DIM**0. 5)# QK^t/sqrt( HEAD_ DIM)
( BATCH _ S 其大小等于批量大小乘以头数
which is batch size multiplied by number of heads,
> TIMELINE OUTLINE soft nax_scale=
**0. 5)# QK^t/sqrt( HEAD_ DIM)
( BATCH _ SIZE, NUM_ HEADS 再乘以查询块的数量
49
. requires normal _(me
0. 0, std=0. 5
OUTLINE multiplied by number of blocks of queue.
TIMELINE soft max _scale =
这就是我们将查询序列划分成的块数
So how many blocks we divided the queue sequence into.
> TIMELINE OUTLINE
sof tna x_scale
45
BATCH_ SIZE, NUM_ HEADS, SE_ 好的,"我们继续.
47
48
normal_(mean=0. 8, std=0. 5)
99
. requires _grad _()
Okay, let'scontinue.
> TIMELINE OUTLINE 52
softnax_scale=
1 /( HEAD_ DIM**θ. 5)# QK~t/ Sqrt ( HEAD _ DIM )
接下来, 我们会具体了解这个部分.
8
Sowe will see what is this one.
> TIMELINE OUTLINE HEAD _ DIM**0. 5)# QK~t/sqrt( HEAD_ DIM)
个 M是我们需要的另一个矩阵
Sothis Mis another matrix > TIMELINE OUTLINE AD_ DIM**0. 5)# QK~t/sqrt( HEAD_ DIM)
1. 8, std=0. 5) 它用于反向传播中的logsumexp计算,
thatwev
Ineed and it's the log sum expo for the backward pass,
OUTLINE WI TIMELINE requires _grad _()
我们稍后会在前向传播结束时
99
BATCH_ SIZE, NUM
> TIMELINE OUTLINE 51
52
. requires _grad _()
不是视频的结尾, 而是前向传播部分的结尾 一一了解它的具体用途
not at the end of this video but at the end of the forward pass-what it's needed for.
> TIMELINE OUTLINE requires _grad _()
简单来说, 你可以把它理解为每一行的最大值.
> TIMELINE OUTLINE requires _grad _())
44
43
normal_(mean =o. 8, std=0. 5)
45
. requires_grad_()
99
48
torci
( BATCH_ SIZE, NUM_ HEADS, SEO_ LEN, HEAD_o IM), dtype=dtype, device=cuda
npty (
> TIMELINE OUTLINE 51
52
normal_(mean=o. 0, std=0. 5)
. requires _grad _()
1. 0, std=0. 5) 为了在反向传播中重新计算查询乘数
um you, we t
BATCH_ SIZE, NUM_ HEAD to, tore compute the query multiplier,
> TIMELINE OUTLINE 51
52
. requires _grad _()
1. 0, std=0. 5)
requires _grad 和键值, 我们也需要这些信息.
> TIMELINE OUTLINE 51
52
requires _grad _()
如果我们不想在反向传播时重新计算每一行的最大值
> TIMELINE OUTLINE. requires _grad _()
10. 8, std=0. 5) 以及softmax的归一化因子, 我们需要保存两个信息:
and the normalization factor of the soft max, we should save two things :
> TIMELINE OUTLINE. requires _grad _(
1. 0, std=0. 5) 是每一行的最大值, 二是归一化因子
> TIMELINE OUTLINE requires _grad _()
1. 0, std=0. 5) 不过, 通过使用 月logsub×的技巧, 我们只需要保存一个值,
however, by using the log sub x trick, we can only save one value, which is the,
> TIMELINE OUTLINE requires _grad _()
n=0. 8, std=0. 5) 正如你在 Flash Attention 算法中看到的那样
【 BATCH _ SIZE, NUM _ HEADS, SEQ _ LEN, HEAD _ DIM )
> TIMELINE OUTLINE 52 requires _grad _()
1. 8, std=0. 5) 就是在 Flash Attention 中用到的东西, 具体来说, 就是这里展示的内容
attention, it's this stuff here which is,
let's see here, it's this stuff here.
> TIMELINE OUTLINE requires _grad _()
14:
Write O;to HBM
thblock of O
15:
Write L;to HBMasthei-thblock of L 它代表了每一行的最大值加上归一化因子的对数.
so this li, which is the maximum for each row, plus the logarithm of the um,
14:
Write O to HBM as the i-th block of O.
15:
Write L;to HBMas the i-thblockof L 在计算反向传播时,
16:end for
17: Return the output O and the log sum ex of the normalization factor, And basically, when computing the backward pass,
14:
Write O to HBM as the i-th block of O.
15:
Write L;to HBMas the i-thblockof L. 我们需要动态地重新计算这个模块的内容.
16:endfor
17: Return the output Oa we need to recompute on the fly this block here.
14:
Write O to HBM as the i-th block of O.
15:
Write L;to HBM as the i-thblockof L. 这里的查询与度矩阵的转置相乘.
16:endfor 因此
So this query multiplied by the transpose of degree.
14:
Write O to HBM as the i-th block of O.
15:
Write L;to HBM as the i-thblock of L
16:endfor 但正如你所记得的, 要应用softmax,
17: Return the output O and the Tog sum ex But to apply the soft max, as you remember,
14:
Write O to HBM as the i-th block of O.
15:
Write L;to HBMasthe i-thblockof L
16:end for 我们需要每一行的最大值和归一化因子.
17: Return the output O an we need to have the maximum for each row and the normalization factor.
14:
Write O;to HBMasthei-th block of O. 由
15:
Write L;to HBMasthei-thblock of L. 我们在反向传播时无需重新计算
So we don't recompute them during the backward
17 : Return the output O and the log sum exp L. 直接使用保存的信息即可.
because we have already computed them during the forward, so we save this information.
17 : Return the output O and the log sum exp L. 但我们并不需要分别保存这两个信息.
But we don't need to save these two information separately.
17 : Return the output O and the log sum exp L 我们可以将它们合并为一个值, 称为li,
We can aggregate it into one single value called li,
17 : Return the output O and the log sum exp L. 稍后我们会看到如何使用它.
and later we will see how we can use it.
17 : Return the output O and the log sum exp L 好的, 民 既然我们已经定义了这个, 现在可以继续往下进行了
All right, so we have defined also this one and we can proceed further.
好的, 既然我们已经定义了这个, 现在可以继续往下进行了.
All right, so we have defined also this one and we can proceed further.
> TIMELINE OUTLINE requires _grad _()
. 8, std=0. 5) 现在: 我们启动网格
(grid) 和内核 (kernel)
Sonowwelaunch our grid, our kernel.
> TIMELINE OUTLINE 51
52
requires _grad _()
std=0. 5) 别担心,! 虽然会有点长, 但请继续往下看.
> TIMELINE OUTLINE. requires _grad _()
我们 通过定义启动网格(launch grid )来启动前向传播的内核(kernel )
So we are launching the kernel for the forward pass by defining what is the launch grid.
> TIMELINE OUTLINE ( BATCH _ SIZE, NUM_ HEADS, SEQ_ LEN, HEAD_ DIM), dtype =dtype, device ="cuda
也就是确定最多可以并行运行多少个这样的程序.
so how many of this program should run in parallel at most.
> TIMELINE OUTLINE ( BATCH _ SIZE, NUM_ HEADS, SEQ_ LEN, HEAD_ DIM), dtype =dtype, device ="cuda
我们传入查询(query )、键(key )、值(values )
> TIMELINE OUTLINE ( BATCH _ SIZE, NUM_ HEADS, SEO_ LEN, HEAD_o IM), dtype =dtype, device ="cuda
55
BATCH_ SIZE= Q
UM
SEQ_ LEN= Q.
soft max 的缩放因子(soft max scale ), 以及 m we are passing the soft max scale, the m,
> TIMELINE OUTLINE 54
torch. enpty
( BATCH_ SIZE, NUM_ HEADS, SEO _ LEN, HEAD _ DI M), dtype =dtype, device =cuda
这些是需要为反向传播保存的信息.
which is the information that we need to save for the backward pass.
> TIMELINE OUTLINE ( BATCH _ SIZE, NUM _ HEADS, SEO _ LEN, HEAD _ DI M), dtype =dtype, device =cuda
实际上,"它就是 Flash Attention 算法伪代码中的l.
it's actually the l in the code of the pseudo code of the flash attention algorithm.
> TIMELINE OUTLINE ( BATCH _ SIZE, NUM_ HEADS, SEQ_ LEN, HEAD_ DIM), dtype =dtype, device ="cuda
这里我称它为m, 因为在原始代码中它也被称为m2, 还有0
> TIMELINE OUTLINE ( BATCH _ SIZE, NUM_ HEADS, SEQ_ LEN, HEAD_ DIM), dtype =dtype, device ="cuda
60 这是我们的内核应该保存其输出的地方.
63
62
test_op( BAT
where the our kernel'should save its output.
> TIMELINE OUTLINE ( BATCH _ SIZE, NUM_ HEADS, SEQ_ LEN, HEAD_ DIM), dtype =dtype, device ="cuda
55
NUM_ HEADS= Q. sha
BATCH_ SIZE = Q. sh
58
59
HEAD_ DIM= HEAD_ DIH
STAGE =stage,
SEQ_ LEN= Q. shape[2] 然后, 正如你所记得的
60
61
62
_op( BATCH _ SIZE, NUM _ HEAD And then, as you remember,
> TIMELINE OUTLINE 63
torch. enpty(
( BATCH_ SIZE, NUM_ HEADS, SEQ _ LEN, HEAD _ DI M), dtype =dtype, device ="cuda
我们无法像在 Torch 中那样通过索引l 张量来获取所有方便的轴.
we don't get all'the nice axes by indexing a tensor, like we are used to in Torch.
> TIMELINE OUTLINE BATCH _ SIZE, NUM_ HEADS, SEQ_ LEN, HEAD_ DIM), dtype =dtype, device ="cuda
60 我们只能获取指向 Q起始元素的指针、
6
Weonly get a pointer to the starting element of Q > TIMELINE OUTLINE ( BATCH _ SIZE, NUM_ HEADS, SEO_ LEN, HEAD_o IM), dtype =dtype, device ="cuda
55
SEQ_ LEN= 指向 K起始元素的指针以及指向 V起始元素的指针,
a pointer to the starting element of K and to the starting element of v,
> TIMELINE OUTLINE ( BATCH _ SIZE, NUM_ HEADS, SEQ_ LEN, HEAD_ DIM), dtype =dtype, device ="cuda
然后必须自行计算内存中其他元素的索引位置.
and then we have to figure out all'the index in the memory of the other elements.
> TIMELINE OUTLINE ( BATCH _ SIZE, NUM _ HEADS, SEO _ LEN, HEAD _ DI M), dtype =dtype, device ="cuda
HEAD _ DIM = HEAD _ DIH_ K,
SEQ_ LEN= Q. shape[2],
60
59
STAGE =stage, 如何计算索引.
61
62
t_op( BATCH_ SIZE, NUM_ HEADS how to calculate'the index.
> TIMELINE OUTLINE 63
torch. enpty(
( BATCH_ SIZE, NUM_ HEADS, SEQ _ LEN, HEAD _ DI M), dtype =dtype, device ="cuda
57
SEQ_ LEN= Q. shape[2]
[1],
58
59
HEAD _ DIM = HEAD STAGE =stage, 我们需要步幅(stride )
60
63
62
t_op( BATCHSIZE, NUM_ HEADS, SEO _ LEN, H we need'the stride,
> TIMELINE OUTLINE torch. en pty (
( BATCH _ SIZE, NUM _ HEADS, SEO _ LEN, HEAD _ DI M), dtype =dtype, de Vice =cuda
因为 步幅告诉我们在从一个维度移动到另一个维度时需要跳过多少个元素
de ftes t_op BATCH _ SIZE, NUM _ HEADS, SEQ_ LEN, HEAD_ IH, cau
float16):
OUTLINE torch. en pty (
because > TIMELINE ( BATCH _ SIZE, NM _ HEADS,, SEQ _ LEN, HEAD _ IM), dtype =dtype, de Vice =cuda
因为 步幅告诉我们在从一个维度移动到另一个维度时需要跳过多少个元素
the stride tells us how many elements to skip to go from one dimension to the other,
> TIMELINE OUTLINE ( BATCH _ SIZE, NUM_ HEADS, SEQ_ LEN, HEAD_ DIM), dtype =dtype, device ="cuda
这就是为什么我们要为每个张量的每个维度传递步幅
and that's why we are passing the stride for each dimension of each tensor.
> TIMELINE OUTLINE ( BATCH _ SIZE, NUM_ HEADS, SEQ_ LEN, HEAD_ DIM), dtype =dtype, device ="cuda
SEQ_ LEN= Q. shape[2] 实际上, 在我们的案例中, 我们处理的q
actually, in our case, we are only working with q,
> TIMELINE OUTLINE ( BATCH _ SIZE, NUM_ HEADS, SEQ_ LEN, HEAD_ DIM), dtype =dtype, device ="cuda
k和√具有相同的数据类型和形状
k and v that are actually of the same d type and of the same shape,
> TIMELINE OUTLINE BATCH _ SIZE, NUM_ HEADS, SEQ_ LEN, HEAD_ DIM), dtype =dtype, device ="cuda
因此理论上不需要为每个张量传递所有步幅
so we should not need actually to pass all all the strides for each of these tensors > TIMELINE OUTLINE ( BATCH _ SIZE, NUM_ HEADS, SEQ_ LEN, HEAD_ DIM), dtype =dtype, device ="cuda
STAGE =
HEAD DIM 因为它们应该具有相同的步幅.
60
62
test_op( BA
because they should have the same strides.
> TIMELINE OUTLINE 63
64
( BATCH_ SIZE, NUM_ HEADS, SEQ_ LEN, HEAD _ DI M), dtype =dtype, device ="cuda
55
JUM
[1], 不过, 在 原始代码中, 我记得它们确实传递了这些步幅, 所以我也保留了这一做法
however, in the original code i'believe they they were passing it, so i kept it.
> TIMELINE OUTLINE ( BATCH _ SIZE, NUM_ HEADS, SEO_ LEN, HEAD_o IM),, dtype =dtype, device ="cuda "
SEQ _ LEN 因此, 步幅让我们能够通过这些指针进行索引, 从而理解
so the stride allow will allow us to index these pointers, to understand um,
> TIMELINE OUTLINE ( BATCH _ SIZE, NUM_ HEADS, SEQ_ LEN, HEAD_ DIM), dtype =dtype, device ="cuda
HEAD _ DIM= HEAD_ DIH_ K,
SEQ_ LEN= Q. shape[2],
STAGE =stage, 并访问张量的元素,
to access the ete ments of of this tensor, just by using its starting, uh,
> TIMELINE OUTLINE ( BATCH _ SIZE, NUM_ HEADS, SEQ_ LEN, HEAD_ DIM), dtype =dtype, device ="cuda
只需使用指向起始元素的指针以及步幅
the pointer to the starting element, and then the strides,
> TIMELINE OUTLINE BATCH _ SIZE, NUM_ HEADS, SEQ_ LEN, HEAD_ DIM), dtype =dtype, device ="cuda
SEQ _ LEN = 我们就能索引到张量中任意所需的元素.
we will be able to index any element we want in the tensor.
> TIMELINE OUTLINE ( BATCH _ SIZE, NUM_ HEADS, SEQ_ LEN, HEAD_ DIM), dtype =dtype, device ="cuda
SEQ _ LEN 接着, 我们传递这些形状的信息, 包括批大小、注意力头的数量、
Then we pass the information of these shapes, so the batch size, the number of heads,
> TIMELINE OUTLINE ( BATCH _ SIZE, NUM_ EADS, SEO_ LEN, HEAD_ DIM), dtype =dtype, device =cuda
test _op ( BATCH _ SIZE,
=(
torch. en pty ( 序列长度以及每个头的维度.
( BATCH _ SIZE the sequence length and the head dimension.
> TIMELINE OUTLINE 68
det te St_op( BAT OH SIZE, NUM _ HEADS, SEQ _ LEN, HEAD _ DIM, C
torch. float16):
BATCH 这些信息对所有张量都是相同的
66
requires gr And which is the same for all of them.
> TIMELINE OUTLINE
7
58
SEQ_ LEN= Q. shape[2]
HEAD _ DIM-HEAD_ DIH_ K
NUM_ HEADS= Q. shape[1],
50
STAGE =stage, 然后是阶段参数.
62
test_op( BATCH_ SIZE, NUM_ HEADS, SEQ_ L
64
( BATCH_ SIZE, NUM_ HEADS, SEO _ LEN,
And then. the. stage.
> TIMELINE OUTLINE 65
56
阶段参数用于指示我们是否要计算因果注意力(causal attention )
The stage indicates if we are going to compute the causal attention > TIMELINE OUTLINE test o{ BATCH SIZE. NUM HEADS. SEO LEN. HEAD DIH. causal. dtv De=torch. float161:
那么, 让我们先不实现它, 继续编写这个方法.
So let's not implement it and let's continue writing this method.
> TIMELINE OUTLINE e st op( BATCH SIZE. NUM HEADS. SEO LEN. HEAD DIM. causal. dtv De=torch. float161:
接下来*我们需要保存一些在后向传播中会用到的信息
So then we need to save some information'that we will be needed for the backward pass,
> TIMELINE OUTLINE 8, std=0. 5
ctx. grid =grid _scale =soft max _scale 也就是我之前提到的这个上下文变量,
which is this context variable that I told you before.
> TIMELINE OUTLINE ( BATCH _ SIZE, NUM _ HEADS, SEO _ LEN, HEAD _ DI M), dtype =dtype, device ="cuda
ale =soft max 播保存了一些信息, 包括查询 间(query )、键(key )和值(value )这些张量
> TIMELINE OUTLINE ( BATCH _ SIZE, NUM _ HEADS, SEO _ LEN, HEAD _ DI M), dtype =dtype, device ="cuda
tx. soft scale =soft max _scale 它们是在后向传播过程中需要计算梯度的对象.
which are the tensor for which we want to compute the gradient during the backward pass.
OUTLINE TIMELINE ( BATCH _ SIZE, NUM_ HEADS, SEQ_ LEN, HEAD_ DIM), dtype =dtype, device ="cuda
ctx. soft ma scale = soft max _scale 此外我们还需要存储这个 M张量和 O张量.
And'then we need'to store also this M'tensor and this O tensor.
> TIMELINE OUTLINE ( BATCH _ SIZE, NUM_ HEADS, SEQ_ LEN, HEAD_ DIM), dtype =dtype, device ="cuda
52
(paraneter)ctx: Any
Ctx. HEAD _ DIM=
ctx. causal= 嗯 然后我们就可以继续了.
return0
test_op( BATCH _ SIZE, NUM _ HEADS, SEQ _ LEN, HEAD um, then we can.
OUTLINE 70
71
torch. enpty(
> TIMELINE ( BATCH _ SIZE, NUM _ HEADS, SEO _ LEN, HEAD _ DI M), dtype =dtype, device ="cuda
scale = soft max _scale 我们还需要存储因果变量(causal variable )
def test _op( BA
atl we need to also store the causal variable.
> TIMELINE OUTLINE ( BATCH _ SIZE, NUM _ HEADS, SEO _ LEN, HEAD _ DI M), dtype =dtype, device =cuda
ctx. grid=grid
ard( Q, k, V, O, M 因为
ctx. sof tn scale = soft max _scale 如果我们在前向传播过程中计算了因果注意力(causal attention )
so because if we computed'the causal attention during the fourth forward pass,
> TIMELINE OUTLINE ( BATCH _ SIZE, NUM_ HEADS, SEQ_ LEN, HEAD_ DIM), dtype =dtype, device ="cuda
ctx. grid = grid ctx. soft scale =soft max _scale 那么在后向传播时也需要这些信息
then during the backward pass we need to have this information > TIMELINE OUTLINE ( BATCH _ SIZE, NUM _ HEADS, SEO _ LEN, HEAD _ DI M), dtype =dtype, device ="cuda
ctx. grid =grid K _scale = soft max _scale ctx. soft 以便屏蔽掉那些我们不希望影响梯度的部分.
because we need to mask out the things that we don't want to contribute to the gradient.
> TIMELINE OUTLINE ( BATCH _ SIZE, NUM _ HEADS, SEO _ LEN, HEAD _ DI M), dtype =dtype, device ="cuda
scale =soft max _scale 不过, 这部分内容我们会留到计算后向传播时再详细讨论
but we will'see that later when computing the backward pass.
> TIMELINE OUTLINE ( BATCH _ SIZE, NUM_ HEADS, SEQ_ LEN, HEAD_ DIM), dtype =dtype, device ="cuda
scale =soft max _scale 现在, 让我们专注于这个前向传播的注意力机制实现.
for now, let'sconcentrate on this attention forward.
> TIMELINE OUTLINE ( BATCH _ SIZE, NUM_ HEADS, SEQ_ LEN, HEAD_ DIM), dtype =dtype, device ="cuda
大 此, 我们需要实现这个前向传播的内核函数, 也就是你看到的这部分代码
> TIMELINE OUTLINE
tx 这个方法是
attention _forward, 用于实现前向传播的逻辑.
So underscore attention, underscore forward method.
> TIMELINE OUTLINE Q =
Triton 内核实际上就是个带有特定装饰器(@triton. jit )的 Python 方法
now a triton kernel is just a python method with a particular decorator called triton. git > TIMELINE OUTLINE
stride (1), 因此我们直接复制并粘贴方法签名即可.
99
copy and paste the signature.
> TIMELINE OUTLINE 51
stride_0_head-0. stride(1),
HEAD _ DIM_ V= V. shape[1]
HEAD_ DIM_ Q, HEAD_ DIM_ K= Q. shape[-1], K. shape[-1] 正是 这个装饰器
(@trit onjit)将一个普通方法转变为 Triton 内核函数.
so this is what makes a try a method become a triton kernel.
> TIMELINE OUTLINE
HEAD _ DIH_ V= V. shape[-1] 如你所见, 这里我们传入查询矩阵(query )
and as you can see, here we pass the query,
> TIMELINE OUTLINE 51
= Q. stride(2) 键矩阵
(key)值矩阵
(value)以及其他信息和m矩阵
key and value-matrix along with other information and the m matrix.
> TIMELINE OUTLINE
stride_ V_seq= V. stride(2),
请注意, 不要将这里的m矩阵与我们将要应用的掩码(mask )混淆.
so please don't confuse them matrix with the mask that we will apply.
> TIMELINE OUTLINE
SEQ_ LEN= Q. shape[2]
84
stride0 因 是在运行时动态生成的.
87
BATCH_ SIZE= Q. shape 01,
stride_0_dim=0. stride(3),
um on the fly.
> TIMELINE OUTLINE 90
NUM_ HEADS= Q. shape [1],
SEQ_ LEN= Q. shape[2 ],
我们会在运行时动态生成它
87
88
BATasze-oshapeowe will generate it on the fly
stride
0_dim=0. stride(3),
> TIMELINE OUTLINE 89
NUM_ HEADS= Q. shape [1]
SEQ_ LEN= Q. shape[2 ],
stride 因为在这个场景下, 我们只关注是否使用因果注意力
because we are only concerned in this case with a causal attention > TIMELINE OUTLINE
SEQ_ LEN= Q. shape[2]
品
causal attention )
85
87
88
BATCH_ SIZE= Q. shape [0],
stride_0_dim=0. stride(3),
or not causal attention.
> TIMELINE OUTLINE 89
NUM_ HEADS= Q. shape [1],
SEO_ LEN= Q. shape[2 ],
这里我们不支持自定义掩码(mask)
stride_0_dim=0.
> TIMELINE OUTLINE
SEQ_ LEN= Q. shape[2),
HEAD _ DIM_ V = V. shape[1]
HEAD _ DIM_ Q, H
HEAD_ DIM_ K = Q. shape[-1], K. s 嗯, 接着我 们传入这些张量的步幅(strides )批次大小、头数(number of head s)
um, then we pass. the strides of all these tensors, the batch size, the number,
> TIMELINE OUTLINE
HEAD _ DIH_ V= V. shape[-1] 序列长度、头维度(head dimension, 即每个张量的形状)
number of heads, the sequence length, the head dimension,
> TIMELINE OUTLINE
HEAD _ DIM V = V. shape [-1]
HEAD_ DIM_ Q.
HEAD_ DIM_ K= Q. shape[-1], K. shape[-1]
BATCH_ SIZE 以及g的块大小和kv的块大小.
assert which is the shape of each of these tensors, and the block size q and the block size kv.
> TIMELINE OUTLINE
HEAD _ DIM_ V= V. shape[-1]
a 的块大小表示我们希望将多少个查询(queries )组合在一起
the block size g indicates how many queries we want to group together > TIMELINE OUTLINE
43
HEAD_o IM_ Q, HEAD_ DIM_ K= Q. shape [-1], K. shape[1]
HEAD_ DIM_ V = V. shape[1]
BATCH _ SIZE, NUM _ HEADS, 形成 Q矩阵的一个块;
49
assert HEAD_ DIH_ O
50
otrchept yu to make one block of the Q matrix,
> TIMELINE OUTLINE 52
52
stage=3 if causal
43
HEAD_ DIM_ V = V. shape[1] 而
KV的块犬小则表示我们希望将多少个键(keys)
torch. c pty-and. the KV indicates how many keys > TIMELINE OUTLINE 51
52
stage=3ifc
HEAD _ DIHLV= V. shape[-1] 和值(values)组合在一起, 形成 K和 V矩阵的一个块.
and values we want-to put together to make one block of the K and V matrix > TIMELINE OUTLINE
3
HEAD_ DIM_ V= V. shape[1] 这正是我们在进行块矩阵乘法时所采用的做法.
which is what we do when we do block matrix multiplication.
> TIMELINE OUTLINE
3
HEAD_ DIM_ V = V. shape[1] 这个参数是 一个数字, 用于指示我们当前使用的是因果注意力
(causal attention )
This stage is a number that indicates if it's a causal > TIMELINE OUTLINE 52
HEAD _ DIM_ V = V. shape[1] 还是非因果注意力(non-causal attention )
stat rah cpt or not causal attention we are doing,
> TIMELINE OUTLINE 52
stage=3 if ca
0=torch. empty _tike(0)
tage=3:i T causal et 如果采用因果注意力它的值为3;如果采用非因果注意力, 它的值则为
so itwill be three in case it's a causal and one in case it's not causal.
> TIMELINE OUTLINE
=torch. empty _like ( Q ) 好的, 我们首先做的事情是验证一些信息.
Okay, the first thing that we do is to verify some information.
> TIMELINE OUTLINE
assert HEAD _ DIM _ Q = HE AD_ DIM_ K and HEAD _ DIM_ K == HEAD_ DIM_ V 因此, 我们验证 KV的块大小是否小于或等于头维度(headdimension )
So we verify that the block size of the KV is less than or equal to the head dimension.
> TIMELINE OUTLINE
assert HEAD _ DIH_ Q == HEAD_ DIM_ K and HEAD _ DIM_ K == HEAD_ DIM_ V
51
52
0=torch. empty _like ( Q)
stage =3 if causal else 1 说实话,
8
55
grid = lambda args :(
#cei L ( SEQ _ LEN / BLOCK _ SIZE _ O)=
57
triton. cdiv( SEQ_ LEN, args BLOCK _ SIZE _
work with?
work with?
> TIMELINE OUTLINE 1,# Zin the CUDA launch grid to
assert HEAD _ DI M_ Q == HEAD_ DIM_ K and HEAD_ DIM_ K == 我 觉得在我的代码中不需要这个验证, 因为我已经移除了大部分限制条件
Idon't think we need it with my code because. l removed most of the constraints.
> TIMELINE OUTLINE
assert HEAD _ DIM_ Q == HEAD _ DIM _ K and HEAD_ DIM_ K == HEAD_ DIM_ V 这个检查在原始代码中也存在, 所以我保留了它.
Sothis check was also present in the. original code, so I kept it.
> TIMELINE OUTLINE
assert HEAD _ DIM_0 = HEAD _ DIM_ K and HEAD_ DIH_ K == HEAD_ DIM_ V
0=torch. empty_like( Q) 不过, 这完全取决于我们的具体实现方式.
56
cei L( SEQ_ LEN
> TIMELINE OUTLINE 57
1,# Z in the Cu DA launch grid
assert HEAD _ DIM_0 = HEAD _ DIM_ K and HEAD_ DIH_ K == HEAD_ DIM_ V
0=torch. empty_like( Q)
stage 稍后我们会了解自动调优的过程
Later we will see what is the auto-tuning process > TIMELINE OUTLINE 1,# Z in the Cu DA launch grid
assert HEAD _ DIM_ Q == HEAD _ DIM _ K and HEAD_ DIM_ K == HEAD_ DIM_ V
0=torch. empty_like( Q)
stage =3 if 看看需要为哪些变量进行调优
and later we will see what variables we are going to autotune for > TIMELINE OUTLINE
assert HEAD _ DIM_ Q == HEAD _ DIM _ K and HEAD_ DIM_ K == HEAD_ DIM_ V torch. empty _like ( Q ) 选择多少阶段(stages ), L 以及选择多少warp 等细节.
and how many stages we will choose, how many warps we will choose, et cetera, et cetera.
OUTLINE TIMELINE
assert HEAD _ DIH_ Q == HEAD _ DIM_ K and HEAD_ DIM_ K == HEAD_p IM_ V 所以, 我们先把这个问题留到后面再讨论
56
triton. cdiv( SEQ_ LEN, args [
BATCH _ SIZE * NUM _ HEADS,
So let'sleave it for later :
vork with?
> TIMELINE OUTLINE 57
1,# Z in the CUDA launch grid
assert HEAD _ DIM _ Q == HEAD _ DI M_ K and HEAD _ DIM_ K == HEAD_ DIM_ V 你可以选择注释掉这部分, 也可以保留它.
cei L( SEQ_
> TIMELINE OUTLINE 57
1,# Z in the CUDA launch grid
assert HEAD _ DIM _ Q == HEAD _ DI M_ K and HEAD _ DIM_ K == HEAD_ DIM_ V
51
52
0=torch. empty _like ( Q)
stage =3 if causal el
53 这应该不会有什么影响,
54
grid = lambda args :(
# Cei L( SEQ_ LEN / BLOCK_ SIZE_ Q)
56
BATCH _ SIZE * NUM _ HEADS,# Which triton. cd iv ( SEQ _ LEN, args [ BLOCK It'shouldn't matter. w work with?
> TIMELINE OUTLINE 58
1,# Z in the CUDA launch grid
BATCH _ SIZE, NUM_ HEADS, SEQ_ LEN, HEAD_ DIM= Q. shape 正如我之前提到的, 我们首先会启动一个网格(grid )
the first thing that we do, as, l said before, we launch a grid.
> TIMELINE OUTLINE triton. cd iv ( SEQ _ LEN, args [ BLo CK _ SIZE _ Q "]),# which group of queries are we going to
( BATCH _ SIZE, NUM _ HEADS, SEQ _ LEN ), device = Q. devic 网格
. float32
(grid )是一系列程序的集合, 其中每个程序都有其标识符. 正如在 CUDA
soagrid is a series of programs where we will have some identifiers, like in the Cu DA > TIMELINE OUTLINE
D=0
( BATCH _ SIZE, NUM _ HEADS, SEO _ LEN ), device = Q. device, dtype=torch. float32 我们通过×轴和y轴上的标识符来区分不同的线程块(blocks )
we had an identifier for the blocks on the x-axis and on the y-axisin Triton > TIMELINE OUTLINE
( BATCH _ SIZE, NUM _ HEADS, SEQ _ LEN ), device = Q. device, dtype=torch. float32 在 Triton 中, 我们通过类似的标识符来区分启动的程序:
we get. this identifier for the programs we launched :
> TIMELINE OUTLINE
块大小(block size )得到 Q, 这是沿第零轴
I(zeroth axis ) E 的程序数量
sequence length divided by block size, Q, number of programs along the zeroth axis > TIMELINE OUTLINE 0=0
( BATCH _ SI 而批次大小(batch size ) 乘以 注意力头数
(number of heads )则是沿第一轴(first axis )的 的程序数量
and the batch size multiplied by. number of heads along the first axis of the launch grid.
> TIMELINE OUTLINE
D=0
( BATCH _ SIZE, NUM _ HEADS,, SEO _ EN ), device Q. device, d type =torch. float32
67
attn_fwd[grid]( 这些标识符将帮助我们确定
69
Q= Q
70
71
K= K
V= V
maxscaesotmaxsc Which will help us identify OUTLINE 72
softm
> TIMELINE
0=0
( BATCH _ SIZE, NUM _ HEADS, SEQ _ LEN ), device = Q. device, dtype=torch. float32 在这个程序(即内核)中需要处理哪一部分查询(query )
which part of. the query we are going to work with in this program,
> TIMELINE OUTLINE 0=0
static met hoc 以及该程序应处理哪个批次(batch )和哪个注意力头(head )
in this kernel, and also in which batch. and on which head this program should work with.
OUTLINE TIMELINE
这就是我们接下来要做的事情.
assert HEAD _
=torch. emp
ty_like (
So that's what we are going to do now.
> TIMELINE OUTLINE 55
BATCH _ SIZE, NUM _ HEADS, SEQ _ LEN, HEAD _ DIM
assert HEAD_ DIH_ Q HEAD_ DIH_ K and HEAD _ 我们正试图
We are trying to understand what part of the input we should work with =torch. empty _like ( Q )
> TIMELINE OUTLINE
49
48
BATCH_ SIZE, NUM_ HEADS, SEO_ LEN,
assert HEAD _ DIH_ Q = HEAD_ DIHLK and 根据程序的 ID
0=torch. empty _like ( Q)
stage=3if causal based on the IDs of the program,
> TIMELINE OUTLINE 55
grid = lambda args:(
(类似于(
CUD A~中的块 ID)来理解应该处理输入的哪一部分.
stae-which corresponds to the block ID in Cu DA.
0=torch. empty_like( Q)
> TIMELINE OUTLINE
BATCH _ SIZE, NUM_ HEADS 因此, 程序1 D为0表示我们正在处理这部分内容
So, the program I DO indicates, it's this stuff here, tells us which part of the queries,
> TIMELINE OUTLINE
它告诉我们当前程序将处理查询(gu eries )白 的哪个块(block ).
so which block of the queries we are going to work with.
> TIMELINE OUTLINE
62 为什么我们需要对查询(guery)分块处理呢?
Why do we have a block on the query?
> TIMELINE OUTLINE attn _fwd [grid ]
Q= Q,
1,# Z in the Cu DA launch grid
63
64
# Mis the logsumexp 因为正如我们之前所见
65
torch. e=pty (
( BATCH _ SIZE, NUM _ HE Because as we saw before,
OUTLINE 89
attn_fwd[grid](
> TIMELINE
Q= Q,
每个查询块(query block )白 的输出可以独立计算
the output can be computed independently for each block of the queries,
> TIMELINE OUTLINE
而每个查询块需要遍历所有的键(keys )和值(values )
while each block of the query has to iterate through all the key and values.
> TIMELINE OUTLINE
62
63 因此, 这将告诉我们
64
65
# Mis the logsumexp for the =torch. e =pty (
( BATCH _ SIZE, NUM _ HEADS, SEV So this will tell us > TIMELINE OUTLINE _attn _fwd [grid ](
61
triton. cdiv( SEQ_ LEN, args[ BLoc K _ SIZE _o"l),# Which group of queries are we going to BATCH _ SIZE * NUM _ HEADS,# Which head of which batch element are we going to work with? 当前程序中正在处理的是查询块
(query block) 的
swhat is the index of the block of the queries > TIMELINE OUTLINE 65
61
triton. cdiv( SEQ_ LEN, args[ BLo CK _ SIZE _o "]),# which group of queries are head of which batch element are we going to work with?
1,# Zin the CUDA launch grid 哪个索引位置.
M is the log
xp for the bac
OUTLINE
65
we are going to work with in this particular program.
> TIMELINE
triton. cd iv ( SEQ _ LEN, args [" BLo CK _ SIZE _o"l),#which group of qu BATCH _ SIZE * NUM _ HEADS,# Which head of which batch element are we going to work with?
62 接下来, 我们还可以理解这个程序关联的是哪个批次(batch)
Then we can understand also which index, which batch.
> TIMELINE OUTLINE
triton. cd iv( SEQ_ LEN, args BLo CK_ SIZE_o"1),# Which group of queries are BATCH _ SIZE * NUM _ HEADS,# Which head of which batch element are we going to work with? 以及哪个注意力头(head)
7
and which head-this program is associated with.
OUTLINE 69
68
> TIMELINE
0= torch. empty _like ( Q)
stage =3if causal else 1 程序 ID为1的 值是批次大小(batch size )与注意力头数量(number of heads )的乘积
The program ID number one is the product of the batch size and the number of heads.
> TIMELINE OUTLINE
attn _fwd [grid ]( 这意味着在第一轴上, 我们将运行
It means that we will have as many programs on the axis number one OUTLINE TIMELINE
71
M=torch. empty (
( BATCH _ SIZE, NUM 与此乘积数量相同的程序.
167
75
attn_fwd[grid]
asthere are indicated by this product.
> TIMELINE OUTLINE 78
V= V.
因此, 通过这个乘积, 我们可以明确当前程序具体关联的是哪个批次
So this product lets us understand, this product will tell us which batch > TIMELINE OUTLINE
72
71
# M is the logsumexp for the backward pass torch. empty (
( BATCH _ SIZE, NUM _ HEADS, 和哪个注意力头.
and which'head this particular program is associated with.
> TIMELINE OUTLINE
70
71
72
73
M= torch. empty (
( BATCH _ SIZE, NUM _ HEADS, S 要获取批次的 ID
8
74
attn_fwd[grid](
Soto get the l D of the batch.
> TIMELINE OUTLINE 78
只需将这个数除以注意力头的数量, 结果即为批次索引.
we just divide this number by the number of heads and it will give us the head index.
TIMELINE OUTLINE
而要获取该批次内的注意力头索引, r 只需对这个数取模(modulus )
And to get the head index inside this batch, we just do this number here, modulus,
> TIMELINE OUTLINE
70
71
72
73
M= torch. empty (
( BATCH _ SIZE, NUM_ HEADS 注意力头的数量即可
74
75
167
attn_fwd[grid ](
the number of heads.
> TIMELINE OUTLINE 78
V= V.
69
71
74
13
BATCH_ 好的, 接下来我们需要做的是.
167
Okay, the next thing that we need to do.
> TIMELINE OUTLINE 78
ftnax_scale,
83
softmax_scale 首先, 当我们传递一个张量时, 因为正如你在这里看到的
we need to, okay, first of all, when we pass a tensor, because, as you can see here,
> TIMELINE OUTLINE
stride_ K_head= K. stride(1),
101
102 传递给注意力机制前向方法的queue参数是一个张量
the queue parameter to this attention forward method is a tensor > TIMELINE OUTLINE ctx. grid =grid
stride _0_head= Q. stride(1), 它是此函数(前向函数)的输入.
because-it's the input of this function, forward function,
> TIMELINE OUTLINE
stride_ V_seq= V. stride(2),
stride stride _ Q_seq=
ead= Q. stride(1)
. stride(2), 当我们调用attention. apply 丶时, 会触发这个前向函数
and this forward function is called here when we do attention. apply > TIMELINE OUTLINE
stride_ V_seq= V. stride(2)
tri _d K, K. grad= K. grad. clone(),
tri_d Q, Q. grad= Q. grad. clone(), None 而这里的
gueue参数已经被创建为一个张量.
and it's this queue stuff here, and'this queue stuff here has been created as a tensor.
> TIMELINE OUTLINE torch. all close (re f_d Q, tri_d Q, atol=atol, rtol=rtol
attn _fwd [grid ]( 因此, 当我们把一个张量传递给 Triton 内核时, 它实际上并不是一个张量
So when we pass a tensor to a triton kernel, it's not really a tensor,
> TIMELINE OUTLINE
stride_ Q_seq=0. stride121,
# This indicates which head and batch to process. Each 而是指向该张量在内存中第一个元素的指针.
it is a pointer to the first element of that tensor in the memory.
> TIMELINE OUTLINE
This indicates which head and batch top index _batch _head = tl. program _id(1) 现在我们需要明确的是, 既然已经知道要处理哪个批次
now we need to understand, because now we know which batch we are going to work with > TIMELINE OUTLINE
61
BATCH_ SIZE, NUM_ HEADS, SEO_ LEN, HEAD _ DIM = Q. shape assert HEAD _ 以及该批次中的哪个注意力头
54
0=torch. empty
66
stage and which head we are going to work with > TIMELINE OUTLINE grid=
ce1 L( SEQ LBN/ BLOCK SIZE Q)= How m
any blocks of Q we have
BATCH _ SIZE, NUM _ HEADS, SEQ _ LEN, HEAD _ DIM 接下来就需要通过索引来从张量中正确选择对应的批次
we need to index this tensor to select the right batch > TIMELINE OUTLINE grid =
ceil ( SEQ LEN / BLOCK SIZE Q )= How
61
BATCH_ SIZE, NUM_ HEADS, SEO_ LEN, HEAD assert HEAD _p IHQ HEAD 及其内部的注意力头.
65
54
0=torch. empty _like ( Q )
stage and'the'right head inside of the right batch.
> TIMELINE OUTLINE grid =
da args :
ny blocks of Q we have
HEAD _ DIM_ V= V. shape[-1]
HEAD _ DIM_ Q, HEAD_ DIM_ K= Q. shape[-1], K. shape[-1] 也就是说, 我们本质上拥有这个queue 张量.
which means. that basically we have this queue tensor.
> TIMELINE OUTLINE q rid =lambda ar as :(
# This indicates which head and batch to process. Each pro
ndex_batch_head= tl. program_id(1) 因此我们需要做一些类似这样的操作
so we need to do some some sort of like,
> TIMELINE OUTLINE
# This indicates which head and batch to process. Each prog index _batch _head = tl. program _id(1)
Th1 比如通过queue 的索引来定位批次
some stuff like this like a queue of the index batch > TIMELINE OUTLINE 51
This indicates which head and batch to
ndex_batch_head=tl. program_id(1) 而注意力头的数量则指示我们要处理的是哪一个头,
and the number of heads indicates which head we are going to work with.
> TIMELINE OUTLINE
# This indicates which head and batch to process. Each pro index _batch _head = tl. program _id(1)
44
45
# This indicate which 46
index _batch =index_ba 所以这里应该是头的索引.
99
index_head =index _batch This indicate so it should be index of head.
> TIMELINE OUTLINE 51
52
# This indicates which head and batch to process. Each n dex_batch_head= tl. program_id(1) 然后, 我们需要选中这些索引范围内的所有内容.
And we need to select everything that is inside these indices.
> TIMELINE OUTLINE
# This indicates which head and batch to index _batch _head = tl. program _id(1) 因此, 我们需 要找到张量中正确的位置即该批次和该注意力头对应的特定序列长度
Th1
Sowe need to enter the tensor at the right location where the particular sequence length > TIMELINE OUTLINE
# This indicates which head and batch to process. Each index _batch _head = tl. program _id(1)
# This indicate which batch this ndex _batch =index _batch _head 和头维度的起始处,
and head dimension for this batch and for this head starts.
> TIMELINE OUTLINE
# This indicates which head and batch to pr index _batch _head = tl. program _id(1) 为此, 我们需要生成一个偏移量, 以便将张量移动到正确的位置
For that we need to generate an offset in which we need to move this tensor from > TIMELINE OUTLINE
# This indicates which head and batch to prc index _batch _head = tl. program _id(1) 因为当前指针指向的是整个张量的起始位置.
because this pointer is pointing at the beginning of the entire tensor.
> TIMELINE OUTLINE
# This indicates which head and batch to index _batch _head = tl. program _id(1) 因此, 我们需要在批次大小维度和注意力头数量维度上进行移动.
Sowe need to move in the batch size dimension and in the number of heads dimension.
TIMELINE OUTLINE
54
lass Triton Attention (torch. auto grad. Function ): 为此, 我们生成以下偏移量, 它将告诉我们这个特定批次
To do that we generate the following offset which will tell us where this particular batch > TIMELINE OUTLINE
lass Triton Attention (torch. auto grad. Function ):
57 @static method def forward (ctx,
HEAD_ DIM_ Q,
HEAD_ DIM_ V 和特定头在张量中的起始位置.
9
and where this particular head starts in this tensor.
> TIMELINE OUTLINE a SSert HEAD _ DIM_ Q== HEAD_ DIM_ K
EAD_ DIA_
57
def forward(ctx, Q, K, V
@static method HEAD _ DIM_ V = V. sh
HEAD_ DIM_ Q, HEAD 为此, 我们需要计算步长.
61
62
> TIMELINE OUTLINE 63
assert HEAD_ DIM_ Q= HEAD_ DIM_ K and HEAD _ OI H_ K
lass Triton Attention (torch. auto grad. Function ):
static methoc 我们需要利用步长来实现这一目的
61
> TIMELINE OUTLINE 62
assert HEAD_ DIH_ O= HEAD_ DIM_ K and HEAD _ OI H_ K =
HEAD _ DIM _
因此,*我们将创建一个 Q KV 偏移量
So what we are. going to. do is we are going to create the Q k V offset.
> TIMELINE OUTLINE
57 class Triton Attention (to rch. autog 这应该是序列长度
59
61
@static method > TIMELINE OUTLINE HEAD _ DIM_ V = V. shape[-1]
52 class Triton Attention (to rch. a 即批次索引乘以步长.
59
Which will'be. the index batch multiplied by the stride.
> TIMELINE OUTLINE HEAD _ DIM_ V= V. shape[-1]
57 class Triton Attention (to rch. autograd. Func 对于批次维度
59
@static method 61
62
> TIMELINE
OUTLINE
63
它将告诉我们需要跳过多少个元素才能到达下一个批次.
which will tell us how many elements we need to skip to get to the next batch.
> TIMELINE OUTLINE
lass Triton Atte 然后将其乘以目标批次的索引值.
And we multiply it by the index of the batch that we want.
> TIMELINE OUTLINE HEAD _ DIM _ V
57 class Triton Attention (to rch 因此, 对于第0个批次
59
@static method 61
62
HEAD_ DIM_ Q, HEAD_ DIH_ K= Q. shap > TIMELINE OUTLINE 63
HEAD_ DILV= V. shape[-1]
我们无需跳过任何元素, 因为已经指向该批次的第一个元素.
we don't skip anything because we. are already pointing to the first element of that batch.
> TIMELINE OUTLINE
DIM_ V= V. shape[-1]
但若处于第1 个批次, 我们将跳过相应数量的元素,
But if we are at the batch one we will skip that many number of elements.
> TIMELINE OUTLINE HEAD _ DIM_ V = V. shape[-1]
class Triton Atte 此外还需要跳过一些注意力头
59
@static method 61
62
HEAD_ DIM_ Q, H
> TIMELINE OUTLINE 63
HEAD_ DIM_ V= V. shape[-1]
根据当前处理的注意力头, 决定需要跳过多少个这样的头.
How many heads we. need to skip based on which head we are going to work with.
> TIMELINE OUTLINE HEAD _ DIM_ V = V. shape[-1]
那么, 是什么告诉我们如何从一个注意力头跳转到下一个呢?
And what tells :us how-to go from one head to the next?
> TIMELINE OUTLINE HEAD _ DIM_ V = V. shape[-1]
是头维度(head dimension )白 的步长(stride )
def forward (ctx, Q, K,
ferduc, u The stride of the head dimension.
> TIMELINE OUTLINE HEAD _ DIM_ V = V. shape[-1]
因此,*我们将当前应处理的注意力头索引
So we multiply. the index. head so the head that we should be working with > TIMELINE OUTLINE
57 class Triton Attention (to rch. autogr 乘以头维度的步长.
59
8
61
@static method with the-stride queue head > TIMELINE OUTLINE HEAD _ DIHLV = V. shape[-1]
好的.
61 class Triton Attention (torch. auto grad. Function ):
All right.
OUTLINE @static method def forward (ctx, Q, K, V, causal, soft max _scale ):
> TIMELINE HEAD _ DIM _ Q, HEAD_ DIH_ K= Q. shape[-1], K. shape[-1]
然后, 我们选择
59 class Triton Attention (to rch. autograd. Func
61
@static method Then we select..
> TIMELINE OUTLINE def forward (ctx, Q, K, V, causal, soft max _scale ):
HEAD _ DIM _ Q, HEAD_ DIH_ K = Q. shape[-1], K. shape[-1]
Triton 提供了一个新功能, 我觉得这是最近才有的.
Triton helps us with a new function that I think it was quite recent.
> TIMELINE OUTLINE HEAD _ DIM _ Q, HEAD _ DIM _ K = Q. shape[-1], K. shape[-1]
ard(ctx, Q, K, V, causal,
这个功能让我们能够轻松索引张量中的元素
estatic that helps us index elements inside of a tensor > TIMELINE OUTLINE def forward (ctx, Q, K, V,
HEAD_ DIM_ Q, HEAD_ DIM_ K= Q. shape [-1], Kshape[-1]
57 而无需处理那些对初学者来说
without having to deal with all the complex indexing maths > TIMELINE OUTLINE HEAD _ DIM _ Q, HEAD_ DIM_ K = Q. shape[-1], K. shape[-1]
57 可能令人困惑的复杂索引计算.
61
lass Triton Attention that can be confusing for beginners.
> TIMELINE OUTLINE 63 @static method def forward (ctx, Q, K, V, causal, soft max _scale ):
HEAD _ DIM _ Q, HEAD_ DIH_ K = Q. shape[-1], K. shape[-1]
因此, 我将使用几个方法来帮助我们完成这种索引操作.
So I will be using a few methods to help us with this indexing.
> TIMELINE OUTLINE ef forward (ctx, Q, K, V,
HEAD_ DIM_ Q, HEAD_ DIM_ K = Q. shape [-1], K. shape[-1]
这个函数名为make _block _pointer `, 它的用法如下:简单来说
and this function-is. called make block pointer and it's this following :so basically > TIMELINE OUTLINE HEAD _ DIM_ Q, HEAD_ DIM_ K= Q. shape [-1], K. shape[-1]
ard(ctx, Q, K, V, cau
strides =(
stride _ K _dim,
stride _ K _seq,
make _block pointer 接收一个指针(而不是向量)作为输入.
this make block pointer takes as input a vector, and sorry, a pointer, not a vector.
> TIMELINE OUTLINE e _block _ptr g
78
79
strides=(
stride_ K_dim,
81
),# We invert the strides stride _ K_seq,
8.
offsets=(0,θ), 它以指针作为输入.
order =(θ, 1),
block she
[ HEAD_ DIH, BLOCK
85
it takes as input a pointer.
TIMELINE OUTLINE 87
0_block _ptr =tl. make_block_ptr(
avk offset.
strides =(
stride _ K _dim,
stride _ K _seq, 在这种情况下, 我们指定:创建一个具有以下形状的块, 即
in this case, we are saying : create a block that has the following shape, that is,
> TIMELINE OUTLINE =tl. ma
80
79
stride _ K_seq,
stride_ K_dim,
81
),# We invert the stride offset s=(0,θ), 序列长度乘以头维度.
84
83
order=(0, 1),
block _shape ( HEAD _ DIM,
sequence length by head dimension.
TIMELINE OUTLINE 87
79
strides=(
stride_ Kdim, 那么 让我一步步来完成这个操作吧
So let me do it one by one, actually.
OUTLINE TIMELINE
o _block _pt r=tl. make _block _ptr (
shape =( SEQ _ LEN, HEAD _ DIM ), 我不想一下子把所有内容都堆在一起, 免得大家感到困惑, 好吗?
don't want to confuse you guys with all this stuff all together, okay?
> TIMELINE OUTLINE. Function l :
stage =3if causal else 1
0=torch. empty _like ( Q) 首先, 这里有一个指针它当前指向的是 Q加上 QKV 偏移量的位置
So start, there is a pointer that is right now pointing at Q plus Q KV offset > TIMELINE OUTLINE
stage =3if causalelse 1
0=torch. empty _like ( Q)
grid 现在, 它并不是指向第一个批次,
83
Soright now it is not pointing at the first batch > TIMELINE OUTLINE 86
HEAD _ DIM_ V= V. shape[-1]
HEADDIM_ O,
D_ DIM_ K= Q. shape[-1], K. shape[-1]
72
75
BATCH_ SIZE, 而是准确地指向了我们的批次
76
assert HEAD_ D
77
78
0=torch. empty but it's pointing exactly to our batch > TIMELINE OUTLINE 79
stage =3ifcausa
HEAD _ DIM _ V= V. shape [-1]
HEADDIM_ O,
(= Q. shape[-1], K. shape[-1] 也就是这个特定程序应该处理的批次, 并且在这个批次内
so the batch that this particular program should be working with and inside this batch > TIMELINE OUTLINE
HEAD _ DIM_ V= V. shape[1]
HEAD DIM_ Q, HEAD DIM_ K
= Q. shape[-1], K. shape[-1]
BATCH_ 指向了这个程序应该处理的特定头.
to the particular head that this program should be working with > TIMELINE OUTLINE
HEAD _ DIM_ V= V. shape [-1]
HEAD DIM_ Q. HEA
K= Qshape[-1], K. shape[-1]
"这基本上意味着我们正指向一个张量.
which ·is basically saying that we are pointing to a tensor.
> TIMELINE OUTLINE
72
71
HEAD _ DIM_ V= V. shape[-1]
HEAD_ DIM_ Q, HEAD_ DIM_ K= Q. shape[-1], K. shape[-1]
74
BATCH _ SIZE, NUM_ HEADS, SEO_ LEN, HEAD_ DIM 具体如下.
76
assert HEAD _p IM_ O HEAD_ DIM_ K a
77
78
0= torch. empty _like ( Q )
that is as follows.
> TIMELINE OUTLINE 79
stage =3if causal else 1
HEAD _ DIM_ V= V. shape[-1]
HEAD DIM_ O, HEAD DIMK
= Q. shape[-1], K. shape[-1] 因此, 我们指向的是以下张量, 民 即正确的头
So we are pointing to the following tensors, which is the right head,
> TIMELINE OUTLINE
70
HEAD _ DIM_ V= V. shape[-1]
HEAD DIM_ O, HE
Q. shape[-1], K. shape[-1]
72 和正确的批次, 然后选择其中的所有内容.
the right batch ·and-the right head, and then we are selecting everything inside.
> TIMELINE OUTLINE
HEAD _ DIM_ V= V. shape[-1]
HEAD DIM_ O. HEAD DIM_ K
= Q. shape[-1], K. shape[-1] 因此, 它指向的是这个特定张量的第一个元素.
Soit'spointing to the first element of this particular tensor.
> TIMELINE OUTLINE
HEAD _ DIM _ V= V. shape [-1]
HEADDIM_ Q. H
= Q. shape[-1], K. shape[-1] 这个特定张量是因为我们已经选择了批次和头.
This particular tensor because we have already selected the batch and the head.
> TIMELINE OUTLINE
HEAD _ DIM_ V= V. shape[-1]
HEAD DIM_ O, HEA
DIM_ K
K= Q. shape[-1], K. shape[-1]
BATCH_ SIZE, N 它是一个二维张量, 形状如下
it is a. two-dimensional tensor with the following shape,
> TIMELINE OUTLINE
HEAD _ DIM_ V= V. shape [-1]
HEADDIM_ O, H
= Q. shape[-1], K. shape[-1]
72
BATCH_ 因为其维度为:序列长度和头维度.
because the following dimensions are : sequence length and head dim.
> TIMELINE OUTLINE
lass Triton Attention (torch. auto grad. Function ): 因此, 我们表示:取这个指针, 它包含一个具有以下形状的张量:
So we are saying : take this pointer which contains a tensor of the following : shape,
> TIMELINE OUTLINE
class Triton Attention (torch. auto grad. Function ):
70
def forward(ctx, Q, K, v, causa
astatic method 71
HEAD_ DIM_ V= V. shape[-1] 序列长度和头维度
8
74
BATCH _ SIZE, N "sequence, length and head dimension,
> TIMELINE OUTLINE 1516
assert HEAD_ DIH_ Q
HEAD_ DIM_ K and
lass Triton Attention (torch. auto grad. Function ): 并且我还提供了这个指针中这些维度的步幅.
and i'm also giving you the'strides of these dimensions that are in this pointer.
> TIMELINE OUTLINE
lass Triton Attention (torch. auto grad. Function ):
70 因此我们需要的两个维度是序列维度
sothe the, thetwo dimensions that are that we need are the sequence dimension and > TIMELINE OUTLINE
class Triton Attention (torch. auto grad. Function ):
70
def forward(ctx, Q, K, v, causal, soft max _sca a static method 71
HEAD_ DIM_ Q, HEAD_ DIH_ K= Q. shape [-1],
HEAD_ DIM_ V= V. shape[-1] 和头维度
8
72
74
75
BATCH _ SIZE, NUM _ HEADS, SEO _ LEN the head dim dimension,
> TIMELINE OUTLINE 76
assert HEAD_p IH_ Q
HEAD_ DIM_ K and HE AD_ DIH_ K
HEAD _ DIM _ V
对于查询张量
q tensor 来说, 头维度就是这一个. 在这个查询张量中
which is this one for'the g tensor, a and um, And in this query tensor,
> TIMELINE OUTLINE HEAD _ DIM _ K
Triton Attention (torch. auto grad. Function ): 我们希望根据程序应处理的查询块来选择
we want to select a block of queries based on the block of queries > TIMELINE OUTLINE
class Triton Attention (torch. auto grad. Function ):
70
def forward(ctx, Q, K, v, causal, soft max_scale
71
HEAD _ DI_ V= V. shape[-1]
HEAD_ DIMQ, HEAD_ DIM_ K= Q. shape[-1], K 查询块
73
74
OUTLINE assert HEAD _ DIH _ HEAD _ DIM _ K > TIMELINE
Triton Attention (torch. auto grad. Function ):
static me
HEAD_ D 我觉得可能得用i Pad来辅助理解
74
So Tthink I need'to may be probably use the i Pad,
> TIMELINE OUTLINE 35 Sert HEAD_ DIH_ Qm HEAD_ DIM_ K and HEAD _ DIM _ K
lass Triton Attention (torch. auto grad. Function ):
70
def forward(ct
astatic method HEAD _ DIM 否则光靠想象可能会让人很困惑.
HEAD_ DIM_ V
74
73
BAT
otherwise it can be very confusing to visualize > TIMELINE OUTLINE 75
76
assert HEAD_ DIH_ Q
HEAD_ DIM_ K and HEAD _o IM _ K m HEAD _
class Triton Attention (torch. auto grad. Function ):
70
71
def forward(ctx, Q, K, V, caus 那就实际操作一下吧.
72
HEAD_ DILV= V. shape[-1]
8
73
74
BATCH _ SIZE, NUM _ HEADS, SEO _ LEN, H So let's do it actually.
> TIMELINE OUTLINE 75
76
a SSert HEAD_ DIH_ Q HEAD_ DIM_ K and HEAD _ DIM _ Km HEAD _ DIM _
ass Triton Attention (torch. auto grad. Function ): 我看看能不能再创建一个窗口, 然后用i Pad 来操作
Let me see if T can create another here and let'suse the i Pad.
> TIMELINE OUTLINE assert HEAD _ DIM _ Q HEAD _ DIM _ K
好的. 目 Rule crd All right.
Scan Templates OUT L
好的, 我们有一个 Q向量, 也就是 Q张量, 因为这个结构
Okay, so we have a Q vector, a Q'tensor, because this construct,
我们会用于所有其他张量,
we will be using it for all the other tensors.
所以如果你理解了一个张量, 就等于理解了所有其他的张量.
so if you understand it for one tensor, you understand it for all the others.
我们有一个 Q 张量, 它的维度依次是批量大小、头数、
We have a Q tensor that is a batch by number of heads, number of heads,
序列长度, 以及头维度.
then the sequence length, and then the head dimension.
用下面这行代码:就是这一行.
With the following line : so this line here.
ass Triton Attention (torch. auto grad. Function ):
70
def forward(ctx,
astatic method HEAD _ DIM_ Q,
HEAD_ DIM_ V 用下面这行代码:就是这一行.
8
74
BATCH_ SIZE,
With the following line :so this line here.
OUTLINE as Sert HEAD _ DIH_ Q
HEAD_ DIH_ K and HEAD_ DIM_ K
> TIMELINE
lass Triton Attention (torch. auto grad. Function ):
70
staticmetho
72 我们创建 Q加上 QKV偏移量时
74
> TIMELINE OUTLINE 76
HEAD_ DIM_
lass Triton Attention (torch. auto grad. Function ):
70 经正确选择了批次维度和头维度
we are already selecting the right batch dimension and already the right head dimension,
> TIMELINE OUTLINE
已经正确选择了批次维度和头维度
we are already selecting the right batch dimension and already the right head dimension,
这意味着我们的 Q 不再指向第一个批次
which means that we have already forwarded our Q to not point to the first batch
和第一个头,
and the first head,
而是指向当前程序正在处理的特定批次
but to point to the exact batch that this program is working with
和特定头.
and the exact head that this program is working with.
也就是说,
which basically
现在它指向的是一个由这两个维度组成的张量.
means that right now it is pointing at a tensor that is made up of these two dimensions.
现 在, 在这个张量内部, 我们还需要选择当前程序应该处理的正确的查询块
now, inside of this tensor, we also need to select the right block of query
以及这个维度.
that this program should work with and this dimension here.
序列维度包含了所有的查询.
so the sequence dimension is all the queries.
因此, 我们需要选择正确的查询, 也就是说, 我们需要跳过一些查询.
so we need to select the right queries, so we need to skip some queries.
如何跳过一些查询呢?
how to skip some queries?
Triton Attention (torch. auto grad. Function ): 我们只需要跳过:thread l dx*乘以块大小, 民 即g-查询数量的部分
well, we say'that we need to skip block index multiplied by block size, q -
> TIMELINE OUTLINE EAD_ DIM_ Q
HEAD_ DIM_ K
Triton Attention (torch. auto grad. Function ): 因为这些查询将由另一个程序处理
number of queries, because they will be processed by another,
> TIMELINE OUTLINE HEAD _ DIM _
ass Triton Attention (torch. auto grad. Function ):
def forward (ctx,
a static method HEAD _ DIM_ Q,
HEAD_ DIM_ V 该程序会使用这里的这个数字.
74
by another program that will have this number here.
> TIMELINE OUTLINE 75
HEAD_ DIM_ Q
HEAD_ DIM_ K
class Triton Attention (torch. auto grad. Function ):
71
70
def fowa rd(ctx, Q, K, V, causa
astatic method 程序 D将有所不同.
72
HEAD_ DIM_ V= V. shape[-1]
8
74
73
BATCH _ SIZE, NUM _ HEAD The program ID will be different.
> TIMELINE OUTLINE 76
assert HEAD_ DIH_ Q
lass Triton Attention (torch. auto grad. Function ): 因此, 通过这行代码, 我们不仅选择了队列中
8
So we are selecting with this line, not only inside of the queue,
> TIMELINE OUTLINE HEAD _ DIM_ O
HEAD_ DIM_ K
HEAD_ DIM_
正确的索引和头维度, 还确定了序列维度中的精确位置
the right index and the head, t but also the right position in this dimension,
> TIMELINE OUTLINE as Sert HEAD _ DIM_ Qm
HEAD_ DIM_ K
class Triton Attention (torch. auto grad. Function ):
70
def forward(ctx, Q, K, V, causal, soft m
astatic method 71
HEAD_ DIM_ V= V. shape[1] 这个位置指向了
in the sequence line dimension that will point to the exact,
> TIMELINE OUTLINE HEAD _ DIM _
ass Triton Attention (torch. auto grad. Function ):
70
def forward(ctx, Q, K
astatic method HEAD _ DIM_ Q, HEA 当前特定程序应处理的查询块
74
73
wro to the starting point of the exact query block > TIMELINE OUTLINE 1516
class Triton Attention (torch. auto grad. Function ):
70
def forward(ctx, Q, K, v, causal, soft ma x_s
astatic method HEAD _ DI_ V = V. shape [-1]
HEAD_ DIM_ Q, HEAD_ DIM_ K= Q. shape[-1], 的起始点
that this particular p program should be working with.
> TIMELINE OUTLINE assert HEAD _ DI H_ Q
HEAD _ DIM _ K
class Triton Attention (torch. auto grad. Function ):
70
def forward(ctx, Q. K, V, c
astatic method 71
HE AD_ DIM_ V= V. shape[1]
HEAD_ DIM_ Q, HEAD_ DIM_ 这就是正在发生的事情.
74
73
Aro, sze, mwwos, se T his'iswhat is happening.
> TIMELINE OUTLINE 75
76
assert HEAD_ DIH_ Q
HEAD_ DIM_ K and HEAD _o I M_ K
同时, 我们也在创建这个块, 稍后我们将看到它如何被使用
And we are also creating this block basically, later we will see how it can be used.
TIMELINE OUTLINE HEAD _ DIM _
lass Triton Attention (torch. auto grad. Function ):
71
def forward(ct
astatic method HEAD _ DIM 为了创建具有特定形状的块
72
73
HEAD_ DIM_
1516
74
BATCH _ SIZE,
um'to uh to create a block of the shape.
> TIMELINE OUTLINE
class Triton Attention (torch. auto grad. Function ):
70
estatic method def forward (ctx,
71
HEAD_ DIM_ Q,
HEAD_ DIM_ V 我们正在指定这个张量的大小.
74
We are telling what is the the size of this tensor.
OUTLINE 1516
assert HEAD_ DIH_ Q HEAD_ DIM_ K
> TIMELINE HEAD _ DIM _ K HEAD _ DIM _
class Triton Attention (torch. auto grad. Function ):
70
71
def forward(ctx, Q, K,
astatic method 因为这个张量有两个维度
73
74
BATCH_ SIZE, NUM_ HEA
sothis tensor has two dimensions > TIMELINE OUTLINE 1516
assert HEAD _ DIH_ Q HEAD_ DIM_ K and HEAD_ DIM_ K m HEAD_ DIMV
grad. Function ):
70 我们正指向正确查询序列的起始位置.
because we are pointing to the beginning of the right query sequence.
> TIMELINE OUTLINE
因此它仅有两个维度:序列维度和头维度.
so it has only two dimensions :the sequence dimension and the head dim dimension.
> TIMELINE OUTLINE
class Triton Attention (torch. auto grad. Function ):
70
def forward(ctx, Q, K, v
astatic method 71
72
HEAD _o IH_ V= V. shap
HEAD_ DIM_ Q, HEAD 因此, 它是最后一个维度.
73
74
> TIMELINE OUTLINE 1516
assert HEAD _ DIH_ QHEAD_ DIM_ K and HEAD_ DIM_ K
HEAD_ DIM_ V
lass Triton Attention (torch. auto grad. Function ):
70 因 由于我们已经跳过了一些查询
HEAD
um, and we are already pointing to the right beginning of the sequence dimension > TIMELINE OUTLINE
lass Triton Attention (torch. auto grad. Function ): 所以现在指向的是序列维度的正确起始位置.
74
75
because we have already skipped some queries.
> TIMELINE OUTLINE a SSert HEAD _ DIM_ Q
HEAD_ DIM_ K and HEAD_ DIM_ K
ass Triton Attention (torch. auto grad. Function ):
71
70
def forward(ctx,
astatic method HE AD_ DIM_ Q, 我们为什么要跳过一些查询呢?
72
73
HEAD_ DIM_ V
OUTLINE
74
assert HEAD _ DIH_ QHEAD_ DIM_ K and HEAD_ DIM_ K
> TIMELINE
ass Triton Attention (torch. auto grad. Function ):
HEAD 因为这些查询将由另一个程序处理
because these queries will be handled by another program > TIMELINE OUTLINE HEAD _ DIM _ K
lass Triton Attention (torch. auto grad. Function ): 该程序会将thread ldx队列指向其他值.
8
that will have a block index queue to some other values.
> TIMELINE OUTLINE E AD_ DIM_ K
class Triton Attention (torch. auto grad. Function ):
70
def forward(ctx, Q, K, V, causal,
a static method 71
HEAD_ DIM_ V= V. shape[-1] 嗯, 还有这个顺序.
8
73
74
BATCH _ SIZE, NUM _ HEADS, SEO _ LEN, HEAD um, and this order.
OUTLINE assert HEAD _ DIH_ QHEAD_ DIM_ K and HEAD_ DIMK
> TIMELINE m HEAD _ DIM _
Triton Attention (torch. auto grad. Function ): 呢, 其实我不太清楚这个顺序是什么.
8
74
> TIMELINE OUTLINE assert HEAD _ DIH_ QHEAD_ DIM_ K
HEAD_ DIM_
lass Triton Attention (torch. auto grad. Function ):
70
def forward(ctx, Q,
astatic method HEAD _ DIM_ Q, 你可以尝试输入零一和一二
8
72
73
HEAD_ DIH_ V
74
BATCH_ SIZE,
you can'try to put zero one and one two.
> TIMELINE OUTLINE as Sert HEAD _ DIH_ Q mm HEAD_ DIM_ K ar
AD_ DIM_ K
Triton Attention (torch. auto grad. Function ): 我认为这是 Triton 进行的一些优化.
1516
74
wi think it's'some optimization that triton does.
> TIMELINE OUTLINE assert HEAD _ DIM_ Q mm HEAD_ DIM_ K and
HEAD_ DIM
我查阅了在线文档, 但没找到相关内容.
i have read the online documentation and i couldn't find anything about it.
> TIMELINE OUTLINE
lass Triton Attention (torch. auto grad. Function ): 所以这是我需要进一步研究的内容.
74 > TIMELINE OUTLINE assert HEAD _ DIH_ Q
HEAD_ DIM_ K ar
HEAD_ DIM_
lass Triton Attention (torch. auto grad. Function ): 但实际上即使你输入零一, 也没什么关系.
but actually,'
even if you put a zero one, it doesn't matter.
> TIMELINE OUTLINE E AD_ DIM _ K anc
lass Triton Attention (torch. auto grad. Function ): 所以 我认为这是你告诉 Triton 的一种方式, 表明你是想要这个块的转置版本
So I think it's something that you tell Triton if you want the transposedofthisblock
> TIM ELINE
OUTLINE
ssert HEAD_ DIM
class Triton Attention (torch. auto grad. Function ):
71
70
def forward(ctx, Q, K, V, causal, sof
HEAD _ DIM _ Q, HEAD _ DI H_ K= Q. sha 还是非转置版本.
HEAD_ DIM_ V= V. shape[-1]
74
3
or you want'the not'transposed version of this block.
> TIMELINE OUTLINE
Sert HEAD_ DIM_ Qm HEAD_ DIM_ K
class Triton Attention (torch. auto grad. Function ):
70
def forward(ctx, Q, K,
HEAD _ DIM_ Q, HEAD 稍后我们会看到, 实际上
HEAD_ DIM _ V= V. sh
And later we will see actually how we can transpose the key block > TIMELINE OUTLINE HEAD _ DIM _ Q HEAD _ DIM _ K
lass Triton Attention (torch. auto grad. Function ): 我们如何在不进行任何转置操作的情况下对键块进行转置.
without doing any transpose operation, actually.
> TIMELINE OUTLINE 76
assert HEAD_ DIH_ Qm HEAD_ DIM_ K HEAD _ DIM _ K
Triton Attention (torch. auto grad. Function ): 我们只需像之前看到的那样改变步幅即可.
8 We wil I just change the strides like we have seen before.
> TIMELINE OUTLINE assert E AD_ DIM_ K
lass Triton Attention (torch. auto grad. Function ): 虽然这个make block pointer 并不是必需的
Now this make block pointer is not something that is necessary,
> TIMELINE OUTLINE AD_ DIM_ Q
但它会在我们索引这个特定指针时让事情变得更简单.
8 but it makes our life easier when we will index this particular pointer.
> TIMELINE OUTLINE assert HEAD _ DIM _ Q HEAD _ DIM _ K
class Triton Attention (torch. auto grad. Function ): 因此, 我们
static met l 可以像处理 Py Torch 中的张量那样, 以几乎相同的方式来操作这个指针
BATCH _ SIZE, NUM_ HEADS, SEO_ LEN, HEAD_ DIMQ. shape So > TIMELINE OUTLINE 76
75
HEAD_ DIM_ V
ass Triton Attention (torch. auto grad. Function ): 因此, 我们
a static method 可以像处理 Py Torch 中的张量那样, 以几乎相同的方式来操作这个指针
we can treat this pointer nearly in the same way when we work with the tensor in Py Torch > TIMELINE OUTLINE
Triton Attention (torch. auto grad. Function ): 我们将能够在某一维度上直接增加索引
We wit l be able to increase one index in one dimension > TIMELINE OUTLINE assert HEAD _ DIM _ Q HEAD _ DIM _ K
class Triton Attention (torch. auto grad. Function ):
70
def foward(ctx, Q, K, V, caus
astatic method HEAD _ DIM _ Q, HEAD _ DIM_ K 而无需手动计算步幅.
72
HEAD_ DIM_ V= V. shape[-1]
8
74
without having to do the computation of the strides.
> TIMELINE OUTLINE assert HEAD _ DIH_ Q
HEAD_ DIM
EAD_ DIM_ K
class Triton Attention (torch. auto grad. Function ):
70
def forward(ctx, Q, K, V,
astatic method HEAD _ DIH_ Q, HEAD_ DIH 稍后在执行反向传播时.
72
8
74
> TIMELINE OUTLINE assert HEAD _p IH_ O
HEAD_ DIM_ K and HEAD_ DIH_ K
ass Triton Attention (torch. auto grad. Function ): 我将不会使用这个, 而是手动完成所有的指针索引操作.
I will not use this one and do all the pointer indexing by hand > TIMELINE OUTLINE assert HEAD _ DIM _ Q HEAD _ DIM _ K
rad. Function ): 因此你可以通过比较使用make block pointer so you can'check'the differences of indexing a tensor by using make block pointer > TIMELINE OUTLINE
lass Triton Attention (torch. auto grad. Function ):
70
def forward(ctx,
astatic method 71
HEAD_ DIM_ Q,
HEAD_ DIM_ V 和不使用它来索引张量的差异.
73
74
75
BATCH _ SIZE NUM_ EADS, SEO _ LEN, HEAD and'not by using it.
> TIMELINE OUTLINE 76
assert HEAD_p IH_ Q
HEAD_o IM_ K and H HEAD _ DIM _ K HEAD _ DIM _
lass Triton Attention (torch. auto grad. Function ): 总之, 为了回顾一下, 我们正在创建什么?
74
75
> TIMELINE OUTLINE
class Triton Attention (torch. auto grad. Function ): 我们正在创建一个指针, 它指向批次维度中的正确索引
We are creating'a pointer to the right index in the batch,
> TIMELINE OUTLINE assert HEAD _ DI H_ Q *
class Triton Attention (torch. auto grad. Function ):
71
70
def forward(ctx, Q, K, V, causa a static method 头维度中的正确索引
HEAD _ DIM_ V= V. shape[-1]
74
73
> TIMELINE OUTLINE 1516
assert HEAD _ DIH_ Q
HEAD_ DIM HEAD _ DIM _
ass Triton Attention (torch. auto grad. Function ): 并且我们根据thread ldx 队列已经跳过了一些查询
and we'are already skipping some queries based on the block index queue.
> TIMELINE OUTLINE
class Triton Attention (torch. auto grad. Function ):
70
def forward(ctx, Q, K,
astatic method 72
HEAD_ DIM_ Q, HEAD
HEAD_ DIM_ V= V. sha 因此, 这个指针已经指向了
so this pointer is already pointing to the right block of queries > TIMELINE OUTLINE EAD_ DIM_ QH
HEAD_ DIM_ K
ass Triton Attention (torch. auto grad. Function ):
70
astatic method HEAD _ DIM_
HEAD_ DIM_ V 当前程序应该处理的正确查询块.
74
that this particular program should be working with.
> TIMELINE OUTLINE assert HE AD_ DIM_ O
HEAD _ DIM _ K
class Triton Attention (torch. auto grad. Function ):
70
estatic method def forward (ctx,
71
HEAD_ DIM_ O,
HEAD_ DIM_ V 现在让我们来看看v和k块.
8
74
73
> TIMELINE
OUTLINE
1516
assert HEAD _ DIM_ Q HEAD _ DIM _ K
EAD_ DIM_ K
HEAD_ DIM_
现在, 让我们复制 V 块, 它与查询类似, 但我们不会深入其中
so let's copy the v block now, which is'similar to the query, but we are not going inside.
> TIMELINE OUTLINE assert HEAD _ DIM_ Q
EAD_ DIM_ K
EAD_ DIM_
offset s=(g. 0) 我们只通过批次索引和头索引来进行索引操作.
we are only. indexing by the index batch and the index head > TIMELINE OUTLINE @static method
69
offsets=(θ,θ),
72
block _shae=( BLOCK_ SIZE _ KV, HEAD _ DIM ),
order =(1, 0), 那又怎样?
so what?
OUTLINE
76
class Triton Attention (torch. auto grad. Function ):
> TIMELINE @static method
offset s=(0,θ),
e=( BLOCK_ SIZE_ KV, 这个实际上 让我在这里写一下一一已经在跳过了.
this-one actually-let me write i t here-is already skipping.
> TIMELINE OUTLINE @static method
72 block _shape =( BLOCK _ SIZE order =(1, 0),
offsets=(0,θ), 所以, 这些查询的数量.
73
75
76
class Triton attention torch. auto grad So this amount of queries.
> TIMELINE OUTLINE 7
astatic method
offsets =(θ,θ) 这正是我们使用块指针进行索引的内容.
This is what we are indexing with this make block pointer.
> TIMELINE OUTLINE @static method
offset s=(0,θ) 因此, 我们处于正确的批次和头中, 并且跳过了一些查询.
So we are in the right batch, in the right head, and we are skipping some queries.
> TIMELINE OUTLINE @static method
offset s=(0, 这里我们只是按批次和头进行索引.
here we are just indexing by batch and by head > TIMELINE OUTLINE static method
offset s=(0,θ) 因此, 我们正在处理索引批次和索引头的 V, 而没有进行选择,
so we are doingv of index batch, index head and we are not selecting.
> TIMELINE OUTLINE @static method
59 我们没有跳过任何内容, 因为你看
lass Tritwe are. not. skipping anything because, you see,
> TIMELINE OUTLINE static method
offsets =(el θ) 在第一维度和第二维度中, 这个偏移量都等于零.
this offset is equal. to zero in :the first dimension, in the second dimension,
> TIMELINE OUTLINE @static method
offsets =(θ,θ) 因此, 在序列长度方面我们没有跳过任何内容
so we are not skipping anything on the sequence length > TIMELINE OUTLINE @static method
在头维度方面我们也没有跳过任何内容.
we are not skipping anything in the head dimension, dimension, head dimension, dimension.
OUTLINE TIMELINE @static method
strides =(stride _ V_seq, stride__dim),
offsets=(0, 好的, 那么让我们来看一下k块指针.
ass trium, all right so let's look at the k block pointer.
> TIMELINE OUTLINE @static method
7 block _shape =( BLOCK _ SIZE _ KV, HEAD_ DIM),
order=(1, 0), 这与之前不同, 因为你知道, 在计算 Flash Attention 算法时,
and this is different because, as you know, when computing the flash attention algorithm,
> TIMELINE OUTLINE @static method
block _shape =( BLOCK _ SIZE _ KV, HEAD_ DIM),
order=(1, 0), 我们需要访问查询块以及所有转置后的键块.
we need to have access. to the block of gu eries and all the block of the key transposed > TIMELINE OUTLINE @static method
block _shape =( BLOCK _ SIZE _ KV, HEAD_ DIM),
order=(1, 0), 因此, 在访问键时, 我们不能像访问 Q那样直接访问它.
sowhen accessing the key we shouldn't access it like we are accessing TIMELINE OUTLINE @static method
block _shape =( BLOCK _ SIZE _ KV, HEAD_ DIM),
order=(1, 0)
75 我们需要反转我们想要转置的两个维度
we should invert the. two. dimensions that we want to transpose for > TIMELINE OUTLINE @static method
secs而使用make_block_ptr
order=(1, 0) 可以非常简单实现这一点, 你可以在这里看到具体操作.
and that's very simple with make block PTR, and you can see it here.
> TIMELINE OUTLINE @static method
我们指定要指向正确的索引和正确的头
We say that we want. to point to the right index and to the right head OUTLINE TIMELINE @static method
83
block _shape=( HEAD_ DIH, BLOCK_ SIZE offset s=(0,θ),
order=(0, 1), 以及其中的张量.
8
and the tensor inside of it,
OUTLINE class Triton Attention (torch. auto grad > TIMELINE @static method
offsets =(θ, 让我在这里写下来, 以便稍后我可以逐行解释.
so let me write it here so later I can explain it line byline OUTLINE TIMELINE @static method
offset s=(0, 我们在这里的操作是:找到 K张量, 选择正确的批次
So what we are doing here is go to the K tensor, select the right batch,
OUTLINE TIMELINE @static method
选择正确的头, 并选择其中的所有内容.
select the. right head, select everything that is inside TIMELINE OUTLINE @static method
81
stride_ K_seq,
"block _shape " parameter of each pointers block.
offsets =(θ,θ) 因此,"这是一个二维张量,
&ed by the param so it's a tensor of two dimensions with the sequence ·length and the head dim,
> TIMELINE OUTLINE @static method
offsets =(θ, 因为你可以在这里看到序列长度、头维度等信息.
because you can see here :sequence length and head dims, etcetera.
OUTLINE TIMELINE @static method
offset s=(0, 但我们不希望先是序列长度, 然后是头维度
but we don't want first sequence length and then head dim,
> TIMELINE OUTLINE @static method
offset s=(0, 而是希望先有头维度, 再有序列长度.
We want-first head dim and then segue nce length TIMELINE OUTLINE static method
81
offsets=(0,θ)
order=(θ, 1)
block_shap 所以我们需要对其进行转置.
lass Triton Atte tion r torch. auto gra So We Want to transpose it.
> TIMELINE OUTLINE static method
block _shape =( HEAD _ DIH, BLOCK _ SIZE order =(θ, 1),
offsets=(0,θ) 如何实现转置呢?
OUTLINE class Triton Attention (torch. auto grad. Funct how to transpose it?
> TIMELINE
offset s=(0, 我们只需说明, 你需要以转置后的两个步幅来读取这个张量.
wejust say that you need. to read this tensor with the two strides transposed > TIMELINE OUTLINE @static method
offsets =(θ, 因此, 我们的意思是:首先使用维度维度的步幅
so we are saying :first use the stride of the dimension dimension > TIMELINE OUTLINE @static method
offset s=(0,θ)
order=(θ, 1), 然后使用序列维度的步幅
and then use the stride of the sequence dimension OUTLINE TIMELINE static method
offsets =( 而这个张量的形状不再是序列长度乘以头维度
and the shape of this uh tensor is not sequence head dim,
OUTLINE TIMELINE @static method
ter ) 而是头维度乘以序列长度, 它是一个键值对的块.
class Tri tit'sahead dim ·sequence and it's a block of kvs.
OUTLINE TIMELINE @static method
81
offsets=(0,
e strides w. r. t Q, sc 为什么我们不直接在这里使用序列维度呢?
why we are not putting directly the sequence dimension here?
TIMELINE OUTLINE @static method
the strides w. r. t Q, so we 因为我们后续想要按块逐块地跳过.
because we want to skip block byblock later.
OUTLINE ass Tri to > TIMELINE @static method
offset s=(0, 因此, 我们并未在序列维度上选择整个序列长度
sowe are not selecting all. the sequence length in the sequence dimension,
> TIMELINE OUTLINE @static method
81
offsets=(0,θ)
order=(e, 1),
block_sha 而是仅选取了一个键值对块
Class Triton Attention (to rc We are just selecting a block of kvs > TIMELINE OUTLINE static method
后续将通过另一种方法跳转到下一个块.
and later we-will use another method to go to the next block.
TIMELINE OUTLINE @static method
offset s=(0,θ)
order=(θ, 1) 希望通过这样的索引展示方式
Sol hope. that by showing you the indexing like this,
> TIMELINE OUTLINE @static method
offset s=(0, 能让大家更容易理解索引的过程.
it's. a. little easier to follow the indexing.
> TIMELINE OUTLINE ass Triton Atter static method
offset s=(0, 因此, 对于每个张量, 我们都会准确地定位到正确的批次和头维度
So for each tensorywe are going into the right batch in the right head dimension.
> TIMELINE OUTLINE @static method
offset s=(0, 0) 而对于查询部分, 我们会跳过一些查询块
And for the query y we are skipping some query blocks > TIMELINE OUTLINE @static method
offset s=(0, 因为每个程序将处理一个不同的较小的查询块.
because each ·program will work with a small different query block.
> TIMELINE OUTLINE @static method
offset s=(0, 但对于键和值, 每个程序都需要遍历所有的键和值.
But for the key and value y each program needs to iterate through all the key and value.
> TIMELINE OUTLINE @static method
offset s=(0,θ)
block_sh 因此, 我们只需将其指向第一个键
Sowejust point it to the first key OUTLINE lass Triton Attention (torch TIMELINE static method
81
offsets=(0,θ),
the strides w. r. t Q, so block _shape =( HEAD _ DIM, BLOCK _ SIZE _ KV),
order=(θ, 1), 和值块,
and value block OUTLINE class Triton Attention (torch. auto grad. Function ):
TIMELINE @static method
offset s=(0, 随后在即将进行的for循环中, 逐块向前推进
and then we will advance one block by one during the for loop that we will do later.
TIMELINE OUTLINE @static method
offset s=(0, 同样地, 在输出部分, 我们也可以构建一个块张量.
Then in the. output, also we can make a block tensor.
TIMELINE OUTLINE @static method
offsets =(block _in de 这基本上创建了一个指针, 就像在查询、键和值的情况下一样,
And this basically creates a pointer, just like in the query key and value case,
TIMELINE OUTLINE @static method
offsets =(block _in de 我们通过它选择正确的批次索引.
OUTLINE iss Triton Atten in which we select the right index batch TIMELINE static method
因此, 我们所做的是按批次进行索引.
So what. we are doing is we are indexing by batch OUTLINE TIMELINE @static method
offsets =(block _index _q * BLoc K_ SIZl
order=(1, 0), 我们按头维度进行索引
s Triton Attention (torch. auto grad. We are indexing by head > TIMELINE OUTLINE static method
offsets =(block _in de 哦不, 我们并没有选择内部的所有内容.
OUTLINE ass Trit Oh no, we. are not selecting everything inside.
> TIMELINE @static method
offsets = 在这种情况下, 我们还跳过了部分查询块.
We are skipping also, in this case, some blocks of queries.
> TIMELINE OUTLINE @static method
offsets =(block _index _q 因为, 正如我之前所说, 输出的形状与查询相同.
8
Because, asl said before, the output has the same shape as the query.
> TIMELINE OUTLINE @static method
block _shape =( BLOCK _ SIZE order =(1, 0),
offsets=(block_index_q * 因此, 这个特定的程序
Sothis particular block OUTLINE lass Triton Attention (torch. auto grad. Fu > TIMELINE @static method
拥有这个特定thread ldx 队列的程序
This particular program that we that will have this particular block index queue,
OUTLINE TIMELINE @static method
strides (stride _ O _seq, stride _ Q _din),
offsets=(block _index_q* BLoc K _ SIZE _ Q, 0),
shape( SEQ_ LEN, HEAD_ DIM ),
block_shape =( BLOCK_ SIZE_0)
4
order=(1, 0), 将只处理一个查询块
V_block_ptr
> TIMELINE
OUTLINE
59
strides=(stride_ V_seq, stride_ V_dim),
shap
EOLEN.
offsets =(block _ind block _shape =
order =(1, 0) 这将只生成输出矩阵的一个块.
which-will produce only one block of the output matrix,
> TIMELINE OUTLINE strides =(stride _ V _seq, stride _ V _dim ),
stride _ K _dim,
stride _ K _seq,
offset s=(0, 我们需要精确地选择那个块
84
order=(θ, 1)
lock_shz
85
and we need to select exactly that one > TIMELINE OUTLINE 87
_block_pt
base=0+qvk_offset,
offsets =(block 以便将指针准确地指向我们应开始写入的位置.
So we can point this pointer exactly to the point where we should start writing.
> TIMELINE OUTLINE @static method
offsets =(block 此, 在这种情况下;我们也跳过thread ldx 队列乘以块大小(即队列行数)
So let's skip also in. this. case, a block index queue multiplied by block size, queue rows,
> TIMELINE OUTLINE @static method
们就能精确地选择出我们的程序
(确切地说是这个特定程序)将生成的块
So we select exactly the block. that our program, this particular program, will produce.
> TIMELINE OUTLINE @static method
offsets =(block _index _q
order=(1, 0), 当我提到这个特定程序时
ss Triton
When-lspeak :about this particular program,
TIMELINE OUTLINE static method
V _block _pt r = tl. make _block _ptr(# V[ind
base= V+qvk_offset,#
shape =( SEQ _ LEN, HEAD _ DIM ),
strides =(stride _ V_seq, stride_ V_ 我指的是在x0轴
Imean the program that is identified by this program IDin thex O axis
offsets=(θ, 0)
> TIMELINE OUTLINE
和第一轴上通过这个程序1 D标识的那个程序.
offsets=(0,θ)
( BLO
and this program ID in the first axis.
OUTLINE 72
order=(1, 0),
> TIMELINE
73
65
base= V+qvk_of
shape=( SEQ_ LEN, 因为这些程序有望并行运行
strides =(stride Because each of these programs will run in parallel, hopefully offset s=(0,
> TIMELINE OUTLINE
每一个程序都会有各自不同的block index U 和index Batch Head 值
and each of them wil have a different value for the block index U and index Batch Head > TIMELINE OUTLINE
好了现在我们已经将指针指向了正确的位置
Okay, now we have pointed our pointers to the right position > TIMELINE OUTLINE
class Triton Attention (torch. auto grad. Function ):
@static method 无论是读取信息
where they should either read some information > TIMELINE OUTLINE
还是写入信息它们都应该在这里进行操作.
Barc or, they. should eith. er write some information.
> TIMELINE OUTLINE
通过使用make block pointer, 这些指针也可以直接作为张量来处理
By using make block pointer, these. pointers can also be treated directly as tensors.
> TIMELINE OUTLINE
97
lass Triton Attention (torch. auto grad. Function ): 正因如此我们才指定了这些张量的形状
So that's why we specify the shapes of these tensors,
> TIMELINE OUTLINE
lass Triton Attention (torch. auto grad. Function ): 因为 Triton 目前提供了一些方法, 可以直接操作数据块
because Triton right now provides. some methods to work directly with blocks > TIMELINE OUTLINE
97
199
@static method def forward (ctx, Q, K, V, 就像直接操作指针一样,
> TIMELINE OUTLINE
97
199
@static method 如同我们在访问张量一般
def forward (ctx, Q, K,
BATCH _ SIZE, NUM _ HEADS,
like we. are. accessing tensors.
> TIMELINE OUTLINE
这样一来,"我们就能像索引张量那样来索引它们了, 明白了吧.
OUTLINE uros So we, can, index. them like tensors, all right.
> TIMELINE
简而言之它主要是基于步幅
(strides )为你做一些计算,
so basically, just try to doing some calculation for you based on the strides,
> TIMELINE OUTLINE
这样你就不必手动进行这些计算了,
@static me def forward (ctx, Q, K,
HEAD_ DIM_ Q, HEA
> TIMELINE OUTLINE RATCH ST7 F
97 但在稍后执行反向传播时
@static method def forward (ctx, Q, K,
> TIMELINE OUTLINE
我们将避免使用大型块指针, 届时会看到手动完成的索引操作.
we will avoid using big block'pointer and we will see the indexing done by hand.
> TIMELINE OUTLINE
好的, 如你所知, 我们正在处理一个查询块.
All right, as you know, we are processing a single block of queries.
> TIMELINE OUTLINE
那么, 让我们回到算法上来吧, 不然我们就偏离了正在做的事情.
So let's go back to the algorithm, otherwise we lose the sight of what we are doing.
> TIMELINE OUTLINE
RAT CH STZF.
那么, 我们到这里来, 让我展示一下我的i Pad.
103
HEAD_ DIM_ O,
TIMELINE OUTLINE 185
17 : Return the output O and the log sum exp L. 好的, 如你所知, 对于每个程序,
All right, so as you know, each program,
17 : Return the output O and the logsumexp L.
16:end for 我们将沿着查询块维度进行并行化处理.
we will parallelize along the query block dimension.
17 : Return the output O and the log sum exp L. 因此, 每个程序将处理不同的查询块.
So each program will work with a different query block.
16:end for
17: Return the output O and the log sum exp L. 接着, 我们需要对所有键和值块进行一个for 循环遍历
And then we need to do a for loop on all the key and value blocks.
17 : Return the output O and the logsumexp L.
16:end for 现在开始.
Right now.
17 : Return the output O and the log sum exp L. 目前, 我们已将指针移动到正确的位置,
we just moved our pointers to the right position
17 : Return the output O and the log sum exp L. 选择出需要操作的查询块,
to select the right query block that we should work with,
17 : Return the output O and the logsumexp L.
16:end for 并将指针定位到待处理的键和值块的起始位置.
and to the beginning of the keys and values block that we should work with,
17 : Return the output O and the logsumexp L.
16:end for 以便根据当前程序应处理的索引和头部信息
based on which index and which head this particular program should be working with.
HEAD _ DIM _ Q,
HEAD _ DIM _ 好的现在我们已经将指针指向了
BATCH _ SIZE All right, now that we have pointed our pointers to the right position in > TIMELINE OUTLINE stage =3if causalelse 1
104
103
HEAD_ DIM_ V= V. shape[-1]
BATCH _ SIZE, NUM _ HEADS, SEQ 程序应当处理的位置
which our program should be working it, inside of the big pointers that are,
0 > TIMELINE OUTLINE stage =3if causal else 1
103
defforward(ctx, Q, K, V, causal, softmax_scale):
HEAD_ DIM
104 这些位置位于包含批次维度、头部数量维度
inside of the big tensors that have the batch dimension, the number of heads dimension,
> TIMELINE OUTLINE
def forward (ctx, Q, K, V, causal, softnax_scale ):
HEAD _ DIM_ Q
Q. shape[-1], K. shape[-1] 序列长度维度和头部维度的大张量中
the sequence length dimension and the heading dimension.
> TIMELINE OUTLINE =3if causal else 1
stage :
EAD_ DIM_ Q,
pe[-1], K. shape[-1] 由于我们已经指向了正确的批次和头部, 所以现在可以开始处理数据了.
we have because we are pointing to the right batch and we are pointing to the right head > TIMELINE OUTLINE stage =3if causalelse1
102
defforward(ctx, Q, K, V, causal, soft max _scale ): 这些张量
HE AD_ DIM_ O, 已经变成了二维张量, 因此它们现在只针对序列维度和头维度进行操作
these tensors have become two-dimensional tensors, so they only work on the.
> TIMELINE OUTLINE stage =3if causal else 1
EAD_ DIM
shape[-1], K. shape[-1] 它们现在仅涉及序列长度和头部维度的张量.
they are only tensors on the sequence length and on the head dimension.
> TIMELINE OUTLINE stage =3if causal else 1
102
def forward(ctx, Q, K, V, causal, softmax_scale): 接
10
HEAD_ DIM_ Q, 下来, 我们需要获取一些额外的信息, 这些信息将在后续步骤中派上用场
now we need some more, some more information that we will use later.
> TIMELINE OUTLINE stage =3if causal else1
102
def forward(ctx, Q, K, V, causal, softmax_scale):
shape[-1], K. shape[-1] 首先, 我们需要获取当前查询块中每个查询的偏移量
The first information that we need is the offsets of each query > TIMELINE OUTLINE 111
stage =3if causal else 1
HEAD _ DIM_ Q,
Q. shape[-1], K. shape[-1] 这些偏移量用于确定当前程序应处理的查询位置,
8
108
assert HEAD _ DIM_ Q
EAD DIM_ K
inside OUTLINE 189
0= torch. empty_like( Q)
> TIMELINE 111
stage=3if causalelse1
102
defforward(ctx, Q, K, V, causal, soft max _scale ):
EAD_ DIM_ Q,
shape[-1], K. shape[-1] 这些偏移量用于确定当前程序应处理的查询位置.
of the current block of queries that this particular program should be working with.
> TIMELINE OUTLINE 3if causalelse1
HEAD _ DIM _ Q,
Q. shape[-1], K. shape[-1]
HEAD_ DIM_ V 这可以通过以下代码行来实现.
BATCH _ SIZE,
assert HE AD_ D
And that is given by the following line.
> TIMELINE OUTLINE 0=torch. empty _like ( Q)
stage=3if causalelse1
183
HEAD_ DIM_ Q, HEAD_ DIM_ K
Q. shape [-1], K. shape [-1]
HEAD_ DIM_ V= V. shape 让我复制并粘贴这段代码.
BATCH _ SIZE, NUM _ HEA assert HEAD DI H Q=
So let me copy and paste.
> TIMELINE OUTLINE 0=torch. empty _like ( Q)
stage=3if causal else1
@static method def forward (ctx, Q, K, V, causal, soft max _scale ):
HEAD _ DIM _ Q, HEAD_ DIH_ K= Q. shape[-1],
HEAD_ DIM_ V = V. shape [-1] 就是这段代码.
8
assert HEAD_ DIM= HEAD_ DIHK and HE A Which o i S. thi S One BATCH _ SIZE, NUM _ HEADS, SEQ _ LEN,
> TIMELINE OUTLINE 0=torch. empty_like( Q)
102 @static method ef forward (ctx, Q, K, V, 那么, 查询的偏移量具体有多少个呢? 首先, 我们需要明确它们的数量
So the offsets of the queries are, first of all, they are, how many of them?
> TIMELINE OUTLINE 0=torch. empty_like( Q)
102 @static method ef forward (ctx, Q, K, V,
_scale ): 因为每个查询块由"block size queue "指定数量的查
Block size queue, because-each block of queries is made up of blocksize queue number > TIMELINE OUTLINE 0= torch. empty_like( Q)
def forward (ctx, Q, K, V, causal, soft max _scale ):
HEAD _ DIM_ V = V. shape[1] 询组成.
BATCH_ SIZE, NUM_ HEADS, SEQ _ LEN, HEAD _ DI > TIMELINE OUTLINE 0=torch. empty_like( Q)
def forward (ctx, Q, K, V, causal, soft max _scale ):
11
HEAD_ DIM_ V = V. shape[-1] 它是一个token.
BATCH _ SIZE, NUM _ HEADS, SEQ _ LEN,
assert HEAD _ DIM = HEAD _ DI MK and HEAD _ DI It'S am t Oken.
> TIMELINE OUTLINE 0= torch. empty_like( Q)
102 @static method 这里的"dimension " 并不是指整个token 的嵌入
dimension is the dim dimension-is not the all the embedding of the token,
> TIMELINE OUTLINE 0=torch. empty_like( Q)
@static method def forward (ctx, Q, K, V,
HEAD_ DIM_ Q, 而是每个token嵌入的一部分
HEAD_ DIM_ V
but a part of the embedding-of each token u which part-the part corresponding to the head BATCH _ SIZE > TIMELINE OUTLINE 0= torch. empty_like( Q)
@static method efforward(ctx, Q, K, V, 具体是与当前程序处理的"head"相对应的那部分.
108
that this particular program is going to work with.
OUTLINE 110
109
> TIMELINE
111
0=torch. empty _like ( Q )
@static method def forward (ctx, Q, K, V, cal
HEAD _ DIM_ O, HEAD
scale ):
HEAD _ DIH_ V= V. sh 因此, 我们正在生成偏移量
BATCH_ SIZE, NUM_
OUTLINE assert HEAD _ DIM _ Q =
So we are generating the offsets > TIMELINE 0=torch. empty_like( Q)
@static method def forward (ctx, Q, K, V,
HEAD DII
cale)
HEAD_ D 用于从包含所有查询的大张量中加载
that will load this particular number of this particular queries > TIMELINE OUTLINE 0=torch. empty_like( Q)
def forward (ctx, Q, K, V, causal, s
HEAD_ DIM_ V = V. shape[-1] 特定数量的特定查询.
BATCH _ SIZE, NUM _ H
aser from the big tensor that contains all queries.
> TIMELINE OUTLINE 0=torch. empty_like( Q)
102 @static method def forward (ctx, Q, K, V,
scale): 我们知道, 我们的查询从threadldx Q乘以块大小 Q开始
And we know that our queries start at the block index Q multiplied by block size Q > TIMELINE OUTLINE 0=torch. empty_like( Q)
103
@staticmethod
104
105 因此, 如果这是程序编号0, 假设块大小为4.
So if this is the program. number O, they will imagine block size is equal to 4.
> TIMELINE
OUTLINE
0=torch. empty _like ( Q )
@static method def forward (ctx, Q, K, V,
HEAD
HEAD 这些查询的索引1将是 0、 1、 2 和 3.
s They will be the query with index 0, 1, 2, and 3
> TIMELINE
OUTLINE
0=torch. empty _like ( Q )
def forward (ctx, Q, K, V, c
HEAD _ DIH_ Q, HEAD_ DIH
HEAD_ DIM_ V= V. shap 但假设我们是程序编号3
BATCH_ SIZE, NUM_ H
OUTLINE assert H But imagine we are. the program number 3
> TIMELINE
0=torch. empty _like ( Q )
@static method 这意味着我们需要跳过3乘以4, 民 即12.
107
which-means that we need to skip3multipliedby4, so12
TIMELINE OUTLINE 111
1
0=torch. empty_like( Q)
102
@staticmethod
103
defforward (ctx, Q, K, V 因此, 它将指向查询编号13、14、15和16, 依此类推.
So it will point to the query number 1314, 15, and 16, et cetera, et cetera, et cetera.
> TIMELINE OUTLINE 0= torch. empty_like( Q)
@static method 好的, 我们对键和值也进行同样的操作.
All right,-and we do the same for the key and values.
> TIMELINE OUTLINE 0= torch. empty_like( Q)
def forward (ctx, Q, K, V, causal, soft max _scale ):
HEAD _ DIM_ O, HEAD_ DIH_ K= Q. shape [-1], K. shape[-1]
HEAD_ DIM_ V = V. shape[-1] 最初,
BATCH_ SIZE, NUM_ HEADS, SEO _ LEN, HEAD _
assert HEAD _ DIM= HEAD_ DIMK and HEAD_ DIK =nitially > TIMELINE OUTLINE 0= torch. empty_like( Q)
@static method 键和值是我们每次迭代所需的一系列键和值.
the key and values is a-range of keys and values that we need at each iteration.
> TIMELINE OUTLINE 0= torch. empty_like( Q)
class Triton Attention (torch. auto grad. Function ):
deffonardctx, Q, K, V, causal, s
@static method 由于我们的 K HEAD _ DIM _ Q,
because our pointer for the K > TIMELINE OUTLINE BATCH _ SIZE, NUM _ F
class Triton Attention Ctor ch. au
@static method def forward (ctx, Q, K, 和 V指针指向这个特定批次
107
109
108
and Vis pointing to the beginning of the sequence of key > TIMELINE OUTLINE 110
111
class Triton Attention (torch. a @static method def forward (ctx, Q, 和特定头的键值序列的开头.
and value for this particular b HEAD _ DIM _ Q,
batch and for this particular head > TIMELINE OUTLINE
class Triton Attention (to rc @static method def forward (ctx, Q, 我们指向键和值的第一个块.
HEAD _ DIM _ Q,
pointing to the first block of key and value.
OUTLINE we are p
ATCH_ SIZE,
> TIMELINE
102
ad. Function): 因此, 在查询的情况下, 我们并没有跳过任何内容.
108
sowe are not skipping anything in the query case.
OUTLINE > TIMELINE
我们之所以跳过, 是因为我们的程序每次只处理一个查询块.
we are skipping because our program will only work with one single block of queries.
> TIMELINE OUTLINE
rad. Function ): 在这种情况下 我们不会跳过任何内容
8
HEAD_ DIM_ V= V. sha
in'this case we don't skip anything > TIMELINE OUTLINE BATCH _ SIZE, NUM _ HEA
class Triton Attention (torch. 因为我们需要遍历这些键和值.
@static method def forward (ctx,
8
because we need to iterate through this key and values.
HEAD _ DIM _ Q,
> TIMELINE OUTLINE
class Triton Att @static method def forward (ct 因此, 我们指向键值的第一个块.
so we are pointing to the first block of key values.
> TIMELINE OUTLINE
lass Triton Attention (torch. auto grad. Function ): 假设块大小:kv 等于四, 那么这里的值将分别是零、一
so imagine block size : Kv is equal to four, so this stuff here will be equal to zero, one,
> TIMELINE OUTLINE
class Triton Attention (torch. auto grad. Function ):
@static method HEAD _ DIM_ Q, HEAD_ DIH_ K= Q. shape [-1], K. shape
HEAD _ DIH_ V= V. shape[-1]
twoa
and three.
> TIMELINE OUTLINE BATCH _ SIZE, NUM_ HEADS, SEQ_ LEN, HEAD_ DIM =
Q. sh
好的. 现在我们需要, 正如你所记得的
HEAD _ DIM _
All'right, now we need, as you remember,
> TIMELINE OUTLINE BATCH _ SIZE, NUM _ HE
在 Flash Attention 算法内部, 我们需要计算一个查询块
inside of the Flash Attention Algorithm we need to compute a block of query > TIMELINE OUTLINE
class Triton Attention (torch. 与键的转置相乘的结果
@static method def forward (ctx,
HE AD_ DIM_ O,
multiplied by the transpose of the keys.
> TIMELINE OUTLINE
103 对于每一个这样的块, 我们需要应用softmax*号操作.
And to each of this block, we need to apply the soft max star.
> TIMELINE OUTLINE 111
如果你还记得, 什么是soft max *号操作?
> TIMELINE OUTLINE
class Triton Attention (to rch. auto 它就是softmax函数
187
@static method def forward (ctx, Q, K, V, causal, soft max OUTLINE HEAD _ DIM_ V= V. shape[-1]
HEAD _ DIM_ Q, HEAD_ DIM_ K
kas. shapt-. klt. isithe soft max.
> TIMELINE
class Triton Attention (to rch. auto 但不包含归一化步骤.
187
@static method OUTLINE HEAD _ DIM _ V= V. shape[-1]
HEAD_ DIM_ Q, HEAD_ DIM
o. without-the normalization.
> TIMELINE
183 class Triton Attenti 因此在计算soft max *号时
@static method def forward (ctx,
HEAD _ DIM _ Q,
> TIMELINE OUTLINE HEAD _ DIM_ V= V. shape[-1]
我们实际上也计算了归一化因子, 但并未立即应用它
we also actually compute. the-normalization factor without applying it > TIMELINE OUTLINE
lass 而是在最后一步才应用这个归一化因子
and we apply the normalization factor at the end.
> TIME LINE OUTLINE
因此, 对于每一个查询块与键转置相乘的结果
So for each bio ck of query-multiplied by transpose of the keys,
> TIMELINE OUTLINE
我们需要找到该特定块中每一行的最大值
we need to have the maximum for each row in this particular block > TIMELINE OUTLINE
183 class Triton Attention (to rcl 以及每一行的归一化因子.
@static method > TIMELINE OUTLINE
102 这就是为什么我们需要以下这两个统计量的原因
109
108
So that's why we need these two following statistics.
> TIMELINE OUTLINE 110
111
183 class Triton Attention (torch. auto grad. Function ): 也就是这个.
@static method def forward (ctx, Q, K, V, causal, soft max OUTLINE HEAD _ DIM _ V= V. shape [-1]
HEAD_ DIM_ Q, HEAD_ DIM_ K
kashti, which is this one.
> TIMELINE
这基本上是一个数字块, 其大小取决于
8
stat
And this is basically a block of numbers based on > TIMELINE OUTLINE HEAD _ DIM_ V = V. shape[-1]
103
tl. zeros([ BLock SIZE_ Q), dtype =tl. float32)-float("inf")
104 我们查询块中包含的查询数量.
187
how many queries we have in our block of queries.
> TIMELINE OUTLINE
103
mi:the runn
tl. zeros([ BLo CK _ SIZE _ Q), dtypentl. float32)-float("inf") 每个值都初始化为负无穷, 就像我之前展示的算法中那样.
Each one initialized with minus infinity, just like in my algorithm that I showed before.
> TIMELINE OUTLINE HEAD DIM_ V= V. shape[-1]
让我回到幻灯片, 以防我们忘记了.
sta So let me go back to the slides in case we forgot > TIMELINE OUTLINE HEAD _ DIM_ V = V. shape[-1]
16:end for
17: Return the output O and the log sum exp L. 或者实际上, 你也可以查看 Flash Attention 算法
or actually you can also check the flash attention algorithm -
17 : Return the output O and the log sum exp L. 我们确实是用负无穷来初始化的
we initialize it with minus infinities.
17 : Return the output O and the log sum exp L. 到目前为止, 我们正在创建这些东西.
so so far we are creating this stuff here.
16:end for
17: Return the output O and the log sum exp L 因此, 我们正在初始化当前所在的m_i, 接下来会初始化! i,
so we are initializing the m, i we are in, we will be initializing the l,
17 : Return the output O and the log sum exp L. 然后是○, 随后在这里展示内层循环,
i and we will initializing the o and then we will show the inner loop here,
这正是我们之前见过的算法.
Alg and this is exactly-the algorithm that we have seen before.
Q=vowsoexp( S-m)+ Q-exp(mo-m)
P=exp( S-m, 所以我们用负无穷来初始化m.
O=drag Cex p(m-
So we initialize m with minus infinities.
P=exp( S-m)
我们同样需要初始化l.
O=drag (exp (m-m,)
now we initialize also the I's.
P=exp( S-m) 让我回到代码部分.
O=dlog(exp(m-m) O.+ PV
so let me go back to the code.
183
_1=tl. zeros([ BLOCK_ SIZE_ Q],
16 让我回到代码部分.
@static method def forward (ctx, Q, K, V, ca
csolet me go back to the code.
> TIMELINE OUTLINE 111
HEADD
tl. zeros ([ BLOCK _ SIZE _ Q ),
dtype=tl. float32)-float("inf") 好的 Ls是用这里的这个数字初始化的.
all right, So. the Ls are initialized with this number here.
> TIMELINE OUTLINE 111
HEAD DIM_ V= V. shape[-1]
103
m_i=tl. zeros([ BLo CK_ SIZE_ Q], dtype =tl. float 32)
float("in")
105
104 所以在这里.
108
107
Triton Attention (torch. auto grad. Fun so here.
OUTLINE 109
110
@static method > TIMELINE 111
def forward(ctx, Q, K, V, causal,. soft na x_scale):
105 在
O块中, 正如我们从 Flash迭代算法中看到的
In the O blocks, as we can see from the flash iteration algorithm,
> TIMELINE OUTLINE @static method
#1_i: the running sum. We
0块是以零值初始化的
the Oblock is initialized with zeros.
OUTLINE 111
112
class Triton Attention (to r
> TIMELINE
113
@static method
#l_i: the running sum. We have one for each query (as we sum the attention scores by row s)
o_block =
acc:t 这就是我们初始化一个块的原因
110
OUTLINE
113
112
class Triton tte tion tor So. thats why we initialize a block.
TIMELINE 114 @static method
#1_i: the running sum. We have one for each query (as we sum the attention scores by rows )
l _i = tl. zeros ({ BLOCK _ SIZE _ Q), dtype =tl. float 32)+1. 0
0_block=tl. zeros([ BEOCK_ SIZE _ Q, HEAD_ DIH), dtype=t1. float32)
acc :the accumulator for the output, which 1s a group of rows of the 0matrix
110
OUTLINE
112
113
class Triton Attention (torch. auto grad. Function ):
> TIMELINE 114 @static method
o _block =
acc : the 这是该特定程序将计算的输出块
This is the output block. that this particular program will compute,
> TIMELINE OUTLINE @static method
1_i: the running sum. We have one for each query (as we sum the attention 其值取决于批次中的位置和索引中的位置.
which is based on. the position in the batch and the position in the index.
> TIMELINE OUTLINE @static method
因此, 它是大小为 Q 块尺寸的一个块.
OUTLINE 112
113
class Trit on Atte
Soit. is one block of the size of blocksize Q
> TIMELINE
114
@static method
那么, 这个块中有多少个查询取决于头维度的大小.
So how many ·queries there are in this block by head dimension,
> TIMELINE OUTLINE @static method
#1_i: the running sum. We have one for each query (as we sum the attention scores by rows )
l _i = tl. zeros ({ BLOCK _ SIZE _ Q), dtype =tl. float32)+1. 0
0_block =tl. zeros([ BLOCK_ SIZE _0, HEA_ DIH], dtype=tl. float32)
acc:the accumulator for the output, which is a group of rows of the O matrix
110
111
OUTLINE
113
class Triton Attention (torch. auto grad. Function ):
> TIMELINE 114 @static method
1_i: the running sum. We have one for each query (as we sum the attention 如果你想直观地理解这点, 让我们回到幻灯片.
which if you want to visualize it, let'sgo back to the slides > TIMELINE OUTLINE @static method
18 它等于这里这个矩阵的一个块.
11
ltisegualto one block of this matrix here.
OUTLINE 113
class Triton Atte
> TIMELINE 114 @static method
107 它等于这里这个矩阵的一个块
CODE OUTLINE 113
112
class Tr
> TIMELINE
114
@static me
PSEUDO CODE
THE SOFT MAX.. 因此, 它是输出矩阵的一个块.
FOR EACH so it's one block of the output matrix.
L
EPREVIOUS
PSEUDOCODE THE SOFT MAX.. 它是一行块的集合, 或一块行的组合.
+ RV s
HEPRE VOUS
PSEUDOCODE THE SOFT MAX...
FOR EACH B THE PRE VOUS
好的, 现在让我们回到代码部分.
END2
EF
Ite okay, so let's go back to the code now.
Ho W
好了, 现在我们在这里已经初始化了一些内容.
110
111
All. right, so. now we have initialized a little stuff here.
OUTLINE 112
113
> TIMELINE
114
@static method
#1_i: the running sum. We have one for each query (as we sum the attention scores by rows)
_1=tl. zeros([ BLOCK _ SIZE_ Q),
0_block=tl. zeros ([ BLo CK acc : the accumulator for 因此, 输出中的mi和li
110
OUTLINE
113
class Triton Attention (torch. auto gr So. the output, the miand li
> TIMELINE
114
@static method
113 分别表示这个特定查询块中每一行的最大值
(mi)
8
wheremiis'the maximum for each row in this particular query block OUTLINE TIMELINE ATCH _ SIZE, NUM _
116 @static method 以及查询块中每一行的归一化因子
117
efforward(ct
8
and the li is the normalization factor for each of the items in the query,
TIMELINE OUTLINE BATCH _ SIZE, NUM
113 class Triton Attention (torch. auto grad. Function ):
115
116
@static method def forward (ctx, Q, K, V, car
_scale):
HEAD_ DIM_ V
for each of the rows in our query block > TIMELINE OUTLINE 121
BATCH_ SIZE, NUM_ HEADS, SEQ_ LEN, HEAD _ DIM =
113 现在我们需要在 Elash Attention 算法中执行内层循环, 即for 循环,
now we need to do the for loop, the innerloop, in the flash attention algorithm.
> TIMELINE OUTLINE BATCH _ SIZE, NUM _ HEADS, SEQ _ LEN = Q. shape
114
115
116 我们将创建一个独立的方法来运行内层循环.
1
we will'create a separate method that will run the innerloop.
OUTLINE TIMELINE
114
113
116
@staticmethod 好的, 让我把它复制到这里
117
def forward (ctx,
so let'sletme copy'it here,
and i am following the same structure of the code > TIMELINE OUTLINE BATCK
= Q. shap
113 我遵循的是你在 Triton 网站教程中看到的代码结构
8
119
that you'see in the tutorial of the triton website.
OUTLINE TIMELINE 121
BATCH_ SIZE, NUM_ HEADS, SEQ_ LEN, HEAD _ DIM
SEQ _ LEN, 基本上, 无论我们是否运行
> TIMELINE OUTLINE def forward (ctx, Q, K, V, causal, soft max _scale ):
offs _kv,
SEQ_ LEN, 因果注意力机制
or even if we are not running the causal attention,
> TIMELINE
OUTLINE
def forward(ctx, Q, K, V, causal, soft max _scale ):
of fs_kv,
130
SEQ_ LEN, 我们都会先构建这个for循环, 然后再构建另一个for循环.
we make this for loop and then we will make another for loop.
> TIMELINE
OUTLINE
def forward(ctx, Q, K, V, causal, soft max _scale ):
SEQ _ LEN,
offs _kv, 接下来我会解释其中的原因
> TIMELINE OUTLINE @static method
那么, 让我先把代码写出来.
class Triton tte tion to rc. auto grad. Fun So. let me first write it.
> TIMELINE OUTLINE
off s_q, 然后我们再一起分析.
OUTLINE class Triton Attention (torch. auto grad. Funct And then we will see.
> TIMELINE @static method
off s_q, 所以, 这里的这个函数就是内层循环.
OUTLINE class Triton Atte So. this function here will be the inner loop.
> TIMELINE @static method
_block 142 在内层循环中, 需要逐一遍历所有的键和值块,
This inner loop. needs to go through all key and value blocks one by one.
> TIMELINE OUTLINE SEQ LEN,
K _block _ptr,
_block _ptr,
block _index _q soft max _scale, 对于每一个查询块和值块
BLOCK _ SIZE _ Q,
BLOCK _ SIZE _ KV,
And for each query and value block OUTLINE off s_q,
offs_kv,
> TIMELINE
49
SEQ LEN,
K _block _ptr
_block
141
_block_ptr 算法都需要对之前计算出的softmax *形块进行调整和修正.
it needs to fix the s previous calculated block of the previous soft max star block.
> TIMELINE OUTLINE SEQ LEN,
141
K_block_ptr 简而言之 我们在这里需要实现的功能大致如下
So basically what we rare doing here, we will need to create a function as the following.
> TIMELINE OUTLINE SEQ LEN,
offs _kv
简而言之, 我们在这里需要实现的功能大致如下.
So basically what we are doing here, we will need to create a function as the following.
在此过程中, 我们将遍历所有的键值块.
where we are going to iterate on all the key value block.
我们需要利用程序中固定的查询块,
We will need to compute the query multiplied by the transpose of the keys
以及当前遍历的键块
using the query block that is fixed for this program
来计算查询与转置后的键的乘积.
and the key block is the one that we are iterating it through.
针对每一个查询, 我们必须计算出每一行的最大值.
And for each of these queries, we need to calculate what is the maximum for each row.
我们需要计算soft max *形值, 民 即未进行归一化因子处理的soft max.
We need to compute the soft max star, so the soft max without the normalization factor.
我们需要维护统计量 L, 它作为归一化因子
We need to keep the statistics L, which is the normalization factor
将在for 循环迭代结束时被应用
that we will apply at the end of the iteration of the for loop.
同时, 我们还需要更新输出结果.
And at the same time, we need to update the output.
正如您所记得的, 输出结果是 P11乘以 V1加上 P12乘以 V2,
So as you remember, the output is P11 multiplied by V1 plus P12 multiplied by V2,
PSEUDO CODE 不过在此之前, 我们需要对之前的" P11 进行修正.
but we need to fix the previous Pit1 *
PSEUDO CODE 因此, 为了修正这一点, 每次我们向 P11 或输出结果累加时,
都需要调整前一次迭代的输出, 然后引入当前迭代的
we need to fix the output of the previous iteration and then we introduce the p
和v块.
and v block of the current iteration.
因此, 在您看到的 Triton 网站上的代码中, 作者
So here the author of the code for the one
Q _block,
block _ptr
141
_block_ptr 因此 在您看到的 Triton网站上的代码中, 作者
146
So here the author of the code for the one OUTLINE 148
147
ofts_kv,
> TIMELINE
L49
SEQ LEN,
决定将这个for 循环拆分为两个步骤,
that you see on the Triton website decided to split this for loop into two steps.
> TIMELINE OUTLINE
46 因为在因果注意力机制中, 当我们应用因果注意力时
Because in the causal attention, we need to, when we have a causal attention,
> TIMELINE OUTLINE
BLOCK _ SIZE _ KV, 我们不希望查询关注那些位于它之后的键.
we have a group of, we don't want the query to attend the keys that come after it.
> TIMELINE OUTLINE
146 而在非因果注意力机制中, 我们允许所有查询关注所有键.
While in the non-causal attention, we let all the queries attend to all the keys.
> TIMELINE OUTLINE
145
146
BLOCK_ SIZE_ KV, 这也 意味着我们需要在遍历所有键和值的for 循环内部加入某种条件判断
which also means that we will need to have some kind of if statement inside of this, if > TIMELINE OUTLINE
BLOCK _ SIZE _ KV,
offs_q,
offs_kv, 即在执行因果注意力时
149
SEO_ LEN,
in the side of this for loop through all the key and values in which we need to check > TIMELINE OUTLINE
BLOCK _ SIZE _ KV, 需检查当前处理的查询是位于键和值之前
if the this particular query that we are working with comes before > TIMELINE OUTLINE
of fs_kv,
offs_q, 还是之后.
SEO_ LEN,
or after the key and value,
in case we are doing the causal attention.
> TIMELINE OUTLINE
因此, 不同于一次性遍历所有键和值
So instead of iterating through all the key and values,
> TIMELINE OUTLINE
146 特别是在因果注意力机制下, 通过将其拆分为两步操作, 我们首先
also in the case of the causal attention, by splitting it into two steps, we are saying,
> TIMELINE OUTLINE
仅遍历那些索引小于当前查询块的所有键
first, let's iterate through all the key and values for which the index is smaller than > TIMELINE OUTLINE
off s_q,
offs_kv, 和值.
SEQ_ LEN,
OUTLINE
153
class Triton Attention C torch. auto the current queries block.
> TIMELINE
SEQ_ LEN,
offs_kv,
offs_q, 为此, 我们需要分别计算因果
and for this we need to compute the attention in the case of the causal > TIMELINE OUTLINE
of fs_kv,
offs_q, 和非因果情况下的注意力.
SEO_ LEN,
and non-causal case.
> TIMELINE OUTLINE 153 class Triton Attention (to rc
接着, 对于位于该块右侧的所有元素
then for all the elements on the right of this block > TIMELINE OUTLINE
145
BLOCK_ SIZE_ KV,
4 即键索引大于查询索引的情况, 在因果注意力机制下
so for which the key index is more than the Q index, in the case of causal attention,
> TIMELINE OUTLINE
我们无需进行任何计算, 因为它们将被掩码处理
we don't need to compute anything because it will be masked out,
> TIMELINE OUTLINE
在soft max 中会变为零, 不会对输出产生贡献
because in the soft max it will become zeros, so it will not contribute to the output,
> TIMELINE OUTLINE
of fs_kv,
offs_q, 因此我们根本无需计算它们.
SEO_ LEN,
sowe don't even have to compute it.
> TIMELINE OUTLINE 153 class Triton Attention (t
正因如此, 我们将这个for 循环分成了两步进行.
152
151
This is why. we. split this for loop into two steps > TIMELINE OUTLINE 153
154
class Triton A
首先, 我们遍历查询与键矩阵相乘结果
So first wei iterate to all the parts that are left to the diagonal of the query > TIMELINE OUTLINE
of fs_q,
offs_kv, 对角线左侧的所有部分.
SEOLEN,
multiplied by the key matrix.
> TIMELINE OUTLINE 153 class Triton Attention (torch.
offs 即所有查询索引小于键索引的值.
So for all the values for which the query index is less than the key index.
> TIMELINE OUTLINE
然后, 在应用因果掩码的情况下
And then we skip all the parts to the right of this diagonal > TIMELINE OUTLINE 153
offs 我们跳过对角线右侧的所有部分.
OUTLINE lass Triton Att in case we are working with a causal mask > TIMELINE
of fs_q,
offs_kv, 但在非因果掩码的情况下
SEO_ LEN,
But in case of the non-causal mask > TIMELINE OUTLINE class Triton Attention C toro
我们需要计算对角线左侧和右侧的所有部分.
we compute the left part and the right part of this diagonal > TIMELINE OUTLINE
别担心, 当我们详细讲解这个for 循环时, 一切都会变得更加清晰明了
All right, don't worry, when we record this for loop, it will be more clear.
> TIMELINE OUTLINE
因此, 我在此先做一个简短的介绍.
So I just wanted to give a little introduction.
> TIMELINE OUTLINE lass Triton At
接下来, 让我们着手编写这个内层循环的代码吧.
OUTLINE 152
153
So let'sgocode this inner loop.
> TIMELINE 154
这个内层循环具体要执行哪些操作呢?
What will this innerloop do?
> TIMELINE OUTLINE lass Triton Attention (torch
139
_block 它将处理我们找到的特定查询块, 也就是这个队列块.
bl
It will work with this particular query block that we have found, so this queue block.
> TIMELINE OUTLINE of fs_q.
_1,
_1
0block 为什么我看不到这个队列块呢?
osiltwill, why I don't see the queue block?
BLOCK _
> TIMELINE OUTLINE of fsa.
_i,
K_block _ptr Q _block,
_block _ptr 因为我还没有加载它, 没错.
soft max scal Because I didn't load it, well, yeah.
OUTLINE BLOCK _ SIZE _ KV,
> TIMELINE of fs a.
K _block _ptr,
V _block _ptr, 那我们就把它加载进来吧.
BLOCK _ SIZE _ Q,
BLOCK _ SIZE _ KV,
soft max _scale,
Let's load it.
> TIMELINE OUTLINE 147
offs a,
_1 实际上, 我们需要加载的是查询块.
Soweneedtoloadthegueryblock, actually > TIMELINE OUTLINE of fsa.
0_block, l_i, m_i=_attn_fwd_inner(
o_block,
_i, 我们忘记加载它了.
Q_block
V _block _ptr,
We forgot to load it.
> TIMELINE OUTLINE 147
BLOCK SIZEQ.
0_block, l_i, m_i=
"_attn_fwd_in 正如你所记得的
0. block, 在 Triton 中 我们通过使用load语句将数据从高带宽内存加载到 SRAM So as you remember in triton, we load data from the high bandwidth memory to the SRAM > TIMELINE OUTLINE BLOCK SIZE Q.
0_block,
o_block,
L1
Q_block, 也就是共享内存中
K_block_ptr _block _ptr so to the shared memory by using the load statement.
> TIMELINE OUTLINE 147
BLOCK SIZE KV,
BLOCK_ SIZE_ O,
o _block, 我们正在指示加载当前应处理的查询块
And we are telling load the query block that we should be working with.
143 TIMELINE OUTLINE BLOCK SIZE KV BLOCK _ SIZE
因为 Q block _ptr 这个指针已经指向了
because this i pointer Q block PTR is already pointing to the right block 138 > TIMELINE OUTLINE K block ptr,
base=0+qvk_offset shape=( SEQ_ LEN, HEAD_ 我们应当操作的正确块.
91
offsets =(block _in de block _shape =( BLOCK _
strides =(stride _0_
OUTLINE
93
94
order=(1, 0),
that we should be working with.
> TIMELINE 95
因此, 它已经跳过了其他程序需要处理的所有块
So it's already skipping all the blocks that other programs should be working with.
> TIMELINE OUTLINE
O _block =tl. zeros ([ BLo CK _ SIz E _
acc :the accumulator for the output, which is a grou 它将 加载一个大小为block size Q 和head _dim 的张量, 也就是正确的查询块
and it will load a tensor of blocksize Q, head dim, so the right block of queries.
> TIMELINE OUTLINE a us al attention or for the blocks to the left of the dia q onal in the This step
13
f STAG 我们将它传递到这个内部循环中, 并将输出传递进去.
And we pass it to this innerloop to which we pass the output.
> TIMELINE OUTLINE _block _ptr,
block _ptr
因此, 它应该在这里写入输出: Li和 Mi
so where it should write this output :the Li and Mi > TIMELINE OUTLINE 143
K_block_ptr,
_block_ptr,
135 这些是行的统计数据, 即每个查询每行的最大值
which are'the statistics for the rows, so the maximum for each row of each query,
> TIMELINE OUTLINE _block _ptr,
block _ptr
以及 Li, 它是每个查询的归一化因子
and the. Li, which is the normalization factor for each query.
TIMELINE OUTLINE V _block _ptr,
K _block p
o_block, l_i, 以及查询块.
o_block,
1
and the query block.
> TIMELINE OUTLINE K_block _ptr,
V_block_ptr,
Q_block,
135 这个程序应当从键和值块的指针起始处开始处理
this program should be working with the beginning of the key and the value block pointer OUTLINE TIMELINE _block _ptr,
_block _ptr
This o_block, i,
o_block, 因为我们需要遍历它们.
1
because we need to iterate through them.
> TIMELINE OUTLINE 143
K_block_ptr
V_block_ptr,
STAG 因此, 我们只需将其指向起始位置, 然后在内部for 循环中
so we just point it to the beginning and then inside the for -
> TIMELINE OUTLINE V _block _ptr,
block _pt
3:
ins for the block s 遍历它们.
o_block,
inner for loop we will iterate through the m.
1
> TIMELINE OUTLINE K _block _ptr V _block _ptr,
135
f STAG 接下来是计算查询时应使用的softmax缩放因子
then the soft max scale that we should use when computing query,
> TIMELINE OUTLINE _block _ptr block _ptr
f STAG
# This
3
o_block,_
o_block, 它乘以键的转置, 以及块大小
multiplied by the transpose of the keys, the block size,
1 > TIMELINE OUTLINE V _block _ptr,
o _block, 即每个 Q块中包含的查询数量
sohow many queries we have in each block of Q > TIMELINE OUTLINE 143
K_block_ptr
V_block_ptr,
以及每个 KV 块中包含的键和值的数量.
and how. many key and value we have in each block of Kv.
> TIMELINE OUTLINE 142
K_block_ptr
V_block_ptr,
这一阶段用于判断我们处于对角线的左侧
This is a stage that tells us if we are on the left side of the diagonal > TIMELINE OUTLINE V _block _ptr,
3:
ins for the blocks to
o_block, 还是右侧,
or on the right side of the diagonal OUTLINE K _block _ptr,
Q _block,
> TIMELINE V _block _ptr,
STAGE 从而确定是否需要根据当前位置应用因果掩码
so it will tell us if we need to apply the causal mask or not, based on where we are,
> TIMELINE OUTLINE K _block _ptr v _block _ptr,
STAGE o_block, 以及是否确实需要应用该掩码
1
and if we need to apply the causal mask > TIMELINE OUTLINE 143
K_block_ptr
V_block_ptr,
134
135
STAGE 偏移量和 KV偏移量表示的是每个 Q块和 KV块内部
The offset Q and the offset KV are just the offsets of the query > TIMELINE OUTLINE _block _ptr,
3:
o_block,_i, m_1
o_block, 查询和键的偏移位置
and key inside of each Q and KV block OUTLINE Q _block,
K _block _ptr,
> TIMELINE 143
V_block_ptr,
135
f STAG 它们是由索引组成的列表, 用于指示我们有多少个查询.
which is a. list of indices that tells us how many queries we have.
> TIMELINE OUTLINE _block _ptr _block _ptr,
接着是序列长度, 即整个序列的长度
141 and then. the sequence length, the entire sequence length,
> TIMELINE OUTLINE K _block _ptr _block _ptr,
134
STAG 因为在for 循环中, 我们需要逐块遍历整个序列长度.
because in the for loop we need to iterate to all the sequence length block byblock.
TIMELINE OUTLINE K _bloc l
V_block _ptr,
if STAG 于是我们依次处理 KV块、 KV块、 KV块
so block of kv, block of kv, blockofkv.
> TIMELINE OUTLINE 143
K_block_ptr
V_block_ptr,
好的, 现在我们来编写这个映射方法
all right, let'swrite this map, let'swrite this method TIMELINE OUTLINE 143
V_block_ptr,
@static met! 稍后我们还需要继续完善这个方法.
and later we actually need to continue this method again.
> TIMELINE OUTLINE 163
我们开始吧, 让我继续往下进行, 好的.
BATCH _ SIZE, N "so let's go and let me go here, all right.
> TIMELINE OUTLINE 164
assert HEAD _o IM_ O
HEAD_o IM_ K and HEAD_o IH_ K m= HEAD_ DIM_ V
这个方法, 我们已经看过它的签名部分了.
So this method we have already seen the signature.
> TIMELINE OUTLINE
block _index _q=tl. program _id(e) 这实际上就是另一个内核函数, 它可以被第一个内核调用
so it's just another kernel, so it can be called by the first kernel,
> TIMELINE OUTLINE
stride _ V_dim,
stride_
stride_o_
stride_0 这种操作在 CUDA中也是可行的
s Hand this is something you can also do in Cu DA.
BATCH SIZ I > TIMELINE OUTLINE HEAD _ DIM: tl. constexpr,
SEO_ LEN:
stride _ V _dim, 实际上, 你可以从一个 CUDA 内核中调用另一个 CUDA 内核.
you can actually call one Cu DA kernel from another Cu DA kernel.
> TIMELINE OUTLINE HE AD_ DIN: tl. const expr,
stride _ V _seq stride _ V _dim, 然后, 根据这个内部循环的阶段, 我们决定需要执行哪些操作.
And then we, based on the stage of this inner loop, we decide what we need to do.
> TIMELINE OUTLINE HEAD _ DI H:tl. const expr,
stride _ Q _head,
stride _
stride 因此, 当我们使用因果注意力机制时
tride
50
49
stride_ K_seq,
So when we are using a causal attention,
OUTLINE stride _ K _dim,
stride _ V _batch > TIMELINE stride _ V _head,
f STAGE==3: 我们只希望对那些索引小于或等于键的查询
soweonly want to apply the attention to the queries for > TIMELINE OUTLINE 173
block_index_q
f STAGE # This step runs for the blocks to == 3:
0_block, l_i, m_i=_attn_fwd_inn
o_btock, 应用注意力.
which t
Q_block the index is less than or equal to the key > TIMELINE OUTLINE block _index _q
164
165
f STAGE==3:
# This step runs for the 所以我们只希望查询能够感知或关注到出现在它之后的键和值.
So we only want the query to know or attend to key and value that come after it.
block > TIMELINE OUTLINE block _index _q,
f STAGE==3: 接着, 我们将阶段参数的值树传递进去.
then we pass the value tree for the stage parameter.
> TIMELINE OUTLINE block _index _q,
164
165
STAGE ==3:
_block 现在, 在因果情况下, 这会变成4减3, 结果等于1.
now, when we in the causal case this will become 4minus3, itisequalto 1.
> TIMELINE OUTLINE block _index _q
t 七stat it _as ert( BLOCK_ SIZE_ KV
HEAD_ IN) 因此, 结果是, 我们只处理从0到当前g块范围内的键
sowhat will happen is that we will only work with the range of keys > TIMELINE OUTLINE
52
stride_v_batch,
stride_ V_head,
53
stride _v_seq,
stride_v_dim,
stride_o _batch, 和值,
stride _o _head,
and values that are from O up to the current block of q, so all the keys,
> TIMELINE OUT LINE
NUM_ HEAD S:tl. const expr
也就是说所有索引小于我们正在处理的查询索引的键.
that whose index is-less than or less than the index of the queries we are working with > TIMELINE OUTLINE tl. const exp
52
53
stride_ V_seq,
stride_ V_head,
stride _v_dim, 即因果掩码左侧的部分.
57
stride_o_seq,
stride_o_head,
stride _o _dim,
BATCH _ SIZE,
So to the left part of the causal mask.
> TIMELINE OUT LINE
NUM_ HEAD S: tl. const expr,
52
53
stride_v_seq,
stride_ V_head,
stride _v _dim, 我来画一下.
8
57
stride_o_seq,
stride_o_din,
stride _o _head,
Let me draw it.
OUTLINE 59
NUM_ HEADS:tl. constexpr,
BATCH _ SIZE,
> TIMELINE 60
stride _ V_batch
stride _ V_head,
53
stride
tride 不然的话, 我觉得理解起来会很困难,
Otherwise, I think it's going to be very difficult to follow.
> TIMELINE OUTLINE
52
stride_ V_batch
stride_ V_head,
53
54
stride _v_seq,
stride _v_dim,
stride_o_batch 那我们就实际操作一下吧.
57
stride_o_seq,
stride_0_head
59
58
stride_o_dim,
BATCH_ SIZE,
So let's do it actually.
> TIMELINE OUTLINE 50
NUM_ HEADS: tl. const expr,
F0 LFN: t1. constexnr
stride _ V _batch tri de_ V_head,
stride 我们新建一个窗口, 然后跳转到这里.
tride_0_seq
tride_0_din,
So let's open a new one and let's go here.
> TIMELINE OUT LINE
NUM_ HEAD S:tl. const expr BATCH _ SIZE,
好的.
All right.
之前我们已经用过这个了, 所以可以再操作一遍.
So we have been using this one before, so we can do it again.
清空页面.
Clear page.
Select
好的, 现在我希望你们把这个矩阵想象成一个分块矩阵,
All right, now I want you to think of the following matrix as a block. matrix, x
我们用粉色来画它, 因为之前我都是用粉色画的
so let's draw it in pink because I have been drawing it all in pink.
我们知道, 在这个查询矩阵的行中, 乘以键矩阵的转置
we know that in the rows of this query, multiplied by the transpose of. the keys,
我们得到的是查询块.
we have a the queries blocks of queries.
Select
所以我们并不是在看单个块, 而是在同时观察所有块,
so we are not watching one single block, we are watching all the blocks. right now.
这是查询块一, 这是查询块二.
so this is the query block one, this is the query block two. sc ce
这是查询块三.
this is the query block three.
Select
每个查询块都是由多个查询token 组成的,
each of this query block is made up of multiple tokens of queries
然后我们还有键块.
and then we have the key, the key blocks.
Select
我们这样做吧.
Let's do it like this.
Select
虽然画得不太好看, 但没关系.
Very ugly, but okay.
Select
嗯, 键块一、键块二、键块三、键块四,
uh, key one, key block two, key block three, key block four',
当你在计算因果注意力时, 会用到这些键块.
when apply calculating the attention, when you calculate the causal attention.
所以, 在使用因果掩码时
so, um, like with the causal mask,
你希望每个查询只关注它之前的键.
you want only the query to attend to keys that come before it.
当我们应用因果掩码时, 这部分会被置为零,
so when we apply the causal mask, this stuff here will be made up of. zeros,
这部分也会被置为零, 还有这部分、
this stuff here will be made up of zeros, this stuff here will be made up. of zeros,
这部分以及这部分, 全部都会被置为零.
and this stuff here and this stuff here and this stuff here, all made up. of zeros.
在这种特定情况下, 我们完全不需要进行任何掩码操作, 因为
we never have to mask out anything when we are in this case, because, welly,
在这个具体场景中, 确实
when we are in this particular scenario, actually in this particular scenario,
不需要对任何内容进行掩码处理.
we don't need to mask out anything for sure.
why?
why?
因为在这个键块中, 所有的键
because all the key keys in this block, so in this block of keys.,-
对应的索引都会小于相应查询的索引,
will have an index that is smaller than the index of the corresponding queries,
前提是查询和键的块大小是匹配的
in case the the block size of the query and the key matches.. c
因此, 每个查询块由三个子查询组成.
So each block of query is made up of three queries.
Select
所以, 这是查询编号零、一和二.
So this is the query number zero, one, and two.
Select
这是查询编号三、四、五.
This is the query number three, four, five.
Select
三、四、五, 没错.
Three, four, five, yeah.
Select
接下来是编号六、七和八.
This will be the number six, seven, and eight.
Select
而这是查询编号九、十和十一.
And this will be the query number nine, 10, and 11.
Select
总计, 我们有十二个查询.
In total, we have 12 queries.
Select
如果我们为块选择相同的大小,
We will have the same indices also for the keys in case Select
键的索引也将保持一致.
we choose the same size for the blocks.
Select
Select
因此, 这里的这个键块将对应键编号零、一和二.
So this key block here will be the key number zero, one and two.
Select
个
Q 这个则是键编号三、四、五.
This will be the key number three, four, five.
Select
Select
接下来是键编号六、七、八, 以此类推, 依此类推.
This will be the six, seven, and eight, et cetera, et cetera, et cetera.
Select
现在的情况是, 正如你所见
Now, what happens is that in this case, as you can see,
Select
键的索引总是小于查询的索引.
the key indices of the keys are always smaller than the indices of the. queries.
Select
因此, 即使是在因果掩码的情况下, 我们也不需要屏蔽任何内容.
So we don't need to mask out anything, even in the case of the causal mask.
Select
因为我们确信, 在这种情况下, 所有这些点积都不会被屏蔽.
because we are sure that in this case, all of these dot products will never be masked out.
同样地, 在这种情况下, 所有这些点积都不会被屏蔽, 而且在这种情况下
Also, in this case, all these dot products will never be masked out and also. in this case,
也永远不会被屏蔽, 永远不会被屏蔽, 永远不会被屏蔽.
will never be masked out, will never be masked out and will never be masked out.
然而, 在这种情况下, 沿着对角线, 一些查询的索引会大于键的索引
and in this case, however, along the diagonal, some of the queries will. be more :
Select
而另一些则不会.
will have an in
ndex that is l bigger than that of the keys and some of them. will not be.
引不会大于键的索引.
ex that is bigge
rthan that of the keys. sci cct
因为这些都是查询块和键块
ar bloc keys.
其中一些需要被屏蔽, 而另一些则不需要被屏蔽.
some of them need to be masked out and some of the mc don't need to be. masked out
因此, 我们将循环分为多个步骤来处理.
so we are dividing our for loop into multiple steps.
Select
Select
我们首先处理的是对角线左侧的所有部分,
the first step that we are doing is all to the left of this diagonal in
这些区域无需进行任何屏蔽操作.
which we don't need to mask out anything.
Select
Select
接下来,
then we will see another step Select
我们会看到另一部分需要屏蔽的区域
here in which we we need to mask out, and then everything to the right of. this will be :
Select
而在因果注意力机制中, 对角线右侧的部分我们甚至不会计算
we will not even compute in the case of causal attention, sccct
因为我们知道它们全为零, 所以不会参与运算.
because we already know it's made up of zero, so it will not come.
Selec i
因此, 在应用soft max 后, 查询与键转置的乘积结果
so the product query multiplied by the transpose of the keys after the. soft max
将全为零.
will be made up of zeros.
Select
Select
所以, 如果你查看 Flash Attention 算法,
so if you look at the flash attention algorithm, so this stuff -..
Personal 0005-Introto GPU& CUDA
0006-Tensor Layouts 0007-Software Pipelining 这里的贡献值会是零, 因为我们用零乘以 V,
9 Nov 2024
here the contribution will be zero-because we are multiplying zero with v,
结果仍然是零, 因此我们无需更改输出.
it will be zero, so we don't need to change the output.
既然我们已经知道这部分矩阵不会对输出产生任何贡献
so why even compute this part of the matrix
那为什么还要费劲去计算它呢?
if we already know it's not going to contribute to the output?
大 此, 我们直接跳过所有这些迭代步骤, 这也是我们将循环分割处理的原因
so we just skip all those iterations and this is why we are splitting the for loop.
希望现在一切都更加清晰明了了
i hope now it's much more clear.
好了, 让我们回到之前的内容.
All right, so let's go back.
stride _ V _head,
t ride _. 好的, 现在我们来到了第一阶段中对角线左侧的部分.
OK, so we are now'to the left part of the diagonal in case of the stage number one.
> TIMELINE OUT LINE
NUM_ HEAD S:tl. const expr
F0 LFN:tl. constexn
stride _ V _head, 在第二阶段中, 我们正好位于对角线上
In the case of the stage number two, it's the part exactly on the diagonal in > TIMELINE OUT LINE
NUM_ HEAD S:tl. const exp
stride _ V _head, 这里需要计算一些点积, 而另一些点积则无需进行.
which we need. to do some dot products and some other dot products we don't need to do.
OUTLINE TIMELINE
53
52
stride_v_seq,
stride_ V_head,
stride _o _batch stride _v_dim, 而在非因果注意力机制中
55
stride _ O _head,
57
58
stride_o_seq,
stride _o _dim,
And then for the non-causal attention,
OUTLINE 59
BATCH_ SIZE,
NUM_ HEADS :
> TIMELINE
stride _ V _head,
tri de_ V_seq 我们只需从零到序列长度一次性完成, 无需分多步进行.
we just go from zero to the sequence length without doing this multi-step.
> TIMELINE OUT LINE
NUM_ HEAD S:tl. const expr
F0 LFN: tl. constexn
53
52
stride _v_dim,
stride_ V_seq,
stride_ O_batch
stride_0_head 因为我们无需屏蔽任何内容.
57
58
sres because we don't need to mask out anything.
stride _o _seq,
> TIMELINE OUT LINE
NUM_ HEAD S: tl. const expr
51
stride_v_batch
stride_ V_head,
53
stride _ V_seq,
stride _v_dim,
stride_ Obat 正因如此, 我们才有了这个阶段,
57
58
stride_o_seq,
stride_o_dim,
stride_0_he
So this is why we have this stage.
OUT LINE
NUM_ HEAD S: tl. const expr,
BATCH _ SIZE,
> TIMELINE F0 LFN:tl. constexn
tri de_ V_head, 这告诉我们, 当前特定阶段应处理的关键块的索引范围
This tells us what is the lower and higher index of the key block that this > TIMELINE OUT LINE
NUM_ HEAD S: tl. const expr
F0 LFN: tl. constexn
53
52
stride_ V_seq,
stride_ V_head,
stride _v_dim, 即从低到高的具体位置.
55
stride _o _head,
57
58
stride_o_seq,
stride _o _dim,
particular stage should be working with.
OUTLINE BATCH _ SIZE,
NUM _ HEADS :
> TIMELINE
好的, 现在这个名为"multiple of "的函数
str aes All right, now this function here, multiple of,
BATCH _ SIZE > TIMELINE OUT LINE
NUM_ HEAD S:tl. const expr
2
stride_ K_dim,
tride_ V_batch, 是用来告诉 Triton, 这里的这个数字是另一个数字的倍数
is just telling Triton that this number here is a multiple of this number.
> TIMELINE OUTLINE st ri
TCH STZF
stride _ Kd in,
stride _ K_seq, 这样一来, Triton就能进行一些优化了.
stride_o_seq,
tride_0_head So Triton can make some optimizations.
> TIMELINE OUTLINE
stride_o_din,
BATCH SI7 F
51
stride_ K_seq
stride_ Kdin, 因此当我们处理因果注意力时, 会进入第一阶段
So the stage one happens when we are doing a causal attention,
> TIMELINE OUTLINE stride _o _dim
o_block, l_i, m_1= 即该函数中的第三阶段, 而四减三的结果就是一
so stage number three in this function, and four minus three will become one.
> TIMELINE OUTLINE
sof tmax _scale,
_block, 设想我们正处于因果注意力的场景中
So imagine we are in the causal attention.
> TIMELINE OUTLINE
sof tmax _scale
101
10g
stride s=(stride_ V_seq, stride_v_dm),
offsets=(0, 8)
102
103 我们将遍历位于对角线左侧的关键块和值块
104
> TIMELINE OUTLINE strides =
# BATCH _ SIZE, NUM _ HEADS, SEQ _ LEN, HEAD _ DIM BATCH _ SIZE, NUM _ HEADS, SEQ _ LEN stride t ride 这些块相对于正在处理的查询块而言.
with respect to the query block that we are working with > TIMELINE OUTLINE stride _ K _dim,
0,# BATCH_ SIZE, NUM_ HEAD S, SEQ_ LEN, HEAD_ DIM BATCH _ SIZE, NUM _ HEADS, SEQ _ LEN 在我们首次调用内部函数时, 并未采用因果注意力机制.
In the case we are doing not causal attention in this first call to the inner function.
> TIMELINE OUTLINE stride _ K _dim,
当前阶段为一, 因此四减去阶段数等于三
175 the stage will be one, so the four minus stage will be equal to three.
> TIMELINE OUTLINE
因此, 我们将执行if 语句的这一部分, 从而遍历所有的键和值.
so we will execute. this part of the if statement so we will go to all the key and values.
TIMELINE OUTLINE
stride _ V_batch
565
54
stride _ V_seq,
57
stride_v_din, 仅针对因果注意力机制的情况.
58
stride_0_head,
In case For the causal attention only OUTLINE stride _o _dim,
BATCH _ SIZE,
> TIME LINE
NUM_ HEAD S:tl. const expr,
BLOCK _ SIZE _ Q,
BLOCK _ SIZE _ KV,
SEQ _ LEN, 如您所见,
as you can see here,
> TIMELINE OUTLINE
177
178
BLOCK _ SIZE _ Q,
BLOCK_ SIZE_ KV,
soft max _scale 我们将在此处进行另一次迭代, 这次仅沿对角线进行,
we will do another iteration here that will only be done along the diagonal > TIMELINE OUTLINE
BLOCK _ SIZE _ O,
BLOCK _ SIZE _ KV, 其中我们需要屏蔽掉某些部分.
SEQ _ LEN,
in which we need to mask out something.
> TIMELINE OUTLINE
177
178
BLOCK _ SIZE _ Q,
BLOCK_ SIZE_ KV 因为在每个块内部, 部分键的索引会低于查询的索引
185
because inside of each blocks there will be some keys > TIMELINE OUTLINE
BLOCK _ SIZE _ Q,
BLOCK _ SIZE _ KV,
SEQ _ LEN, 而另一部分
that have the index below the index of the query > TIMELINE OUTLINE Triton Attention (to
BLOCK _ SIZE _ O,
BLOCK _ SIZE _ KV, 则会高于查询的索引.
SEO _ LEN,
and some that have above the index of the query > TIMELINE OUTLINE 186
178
BLOCK_ SIZE_ KV, 因此, 仅在因果注意力机制下, 我们会两次调用此函数.
Soonly in the causal attention we will call this function twice.
> TIMELINE OUTLINE
BLOCK _ SIZE KV 第一次调用时, 阶段参数设为1;
The first time with the stage equal to one > TIMELINE OUTLINE cent ion (torch. auto grad. Function ):
index _batch. to tl. int61*strise_ Q_batch
+1hdex_head. to it L. Inf64|*stride_0_head
Q_block_ptr 第一次调用时, 阶段参数设为2.
andi the second time with the stage equal to two.
HEAD _ OIM TIMELINE OUTLINE LDCK_ SZE_ OHEAD_ DIM}
第二次调用时我们仅遍历那些恰好位于矩阵对角线上的键值块组
And the second i time we will only iterate through the group of key v blocks > TIMELINE OUTLINE BLOCK _ SIZE _ KV:tl. const expr,
stride _ O _batch tri de 该矩阵由查询与转置后的键相乘得到
8
NUM_ HEADS
64
63
SEQ_ LEN: tl. const expr,
that > TIMELINE OUTLINE BLOCK _ SIZE _ KV:tl. const expr,
BLOCK _ SIZE _ Q: tl. const expr,
stride _ O _batch 该矩阵由查询与转置后的键相乘得到
are exactly on the diagonal of the matrix query multiplied by the transpose of the keys,
> TIMELINE OUTLINE BLOCK _ SIZE _ KV:tl. const expr
57
stride_o_batch
59
stride_ O_seq
stride_o_dim,
BATCH_ SIZE, 即由所有块构成的大矩阵,
63
othe big matrix that is made up of all the blocks.
NUM _ HEADS :
SEQ _ L > TIMELINE OUTLINE 6g
BLOCK _ SIZE _ KV:tl. constexpr,
BLOCK_ SIZE_ Q:tl.
stride _o _batch 好的, 现在这一点应该已经清楚了, 让我们继续深入.
All right now. that this should be clear, let'sproceed further.
> TIMELINE OUTLINE BLOCK _ SIZE _ KV:tl. const expr,
stride _ V _seq,
stride _ V _dim, 那么, 日 由于我们需要执行for 循环, 民 即 Flash 算法中的内层循环.
So let's, because we. need to do the for loop, the inner for loop of the flash.
> TIMELINE OUTLINE HEAD _ DI M: tl. const expr,
stride _ V _dim, 接下来, 让我们加载第一个键和值的块
attention, let's go and load the first blocks of key and values,
> TIMELINE OUTLINE HEAD _o I M: tl. const expr,
stride _ V _head,
stride _ V _batch 58
stride_ V_seq,
stride_ V_dim,
60 也就是当前键和值块指针所指向的
which is exactly the one that the key and V blocks are currently pointing at,
stride _
> TIMELINE OUTLINE SEQ _ L EN:tl. const expr
soft na x_scale,
# BATCH _ SIZE, NUM _ HEADS, SEQ _ LEN, HEAD _ DIM BATCH _ SIZE, NUM _ HEADS, SEQ _ LEN, HEAD _ DIH BATCH _ SIZE, NUM_ HEADS, SEO_ LEN
00 块.
stride_ Q_batch,
BATCH _ SIZE, NUM _ HEADS, SEO _ LEN, HEAD _ DIM
50
49
stride_0_seq,
stride_ Q_head,
which is the oo block.
> TIMELINE OUTLINE 52
stride_ Kbatch,
stride_ O_dim,
BATCH SIZE UM _ HEADS,
SEQ _ LEN,
HEAD _ DIM
BATC 因此, 我们基本上定义了这些指针.
SIZE
47
softnax
50
49
BATCH _ SIZE,
So we define the pointers basically OUTLINE stride _ Q _batch stride _ O _head,
> TIMELINE 52
stride_ Q_seq,
stride _ V _batch stride _ V _head,
stride _ V_seq 我们将键和值块指向这个
61
62
stride_ V_dim,
stride_
63
stra W e point the key and value blocks to the first key
strs
OUTLINE
64
65
strid
> TIMELINE BATCH _ SIZE,
stride _ K dim
stride _ V_bat ch
58
59
60
61
stride tride
for : 循环应该处理的第一个键和值块
and value block that this for loop should be working with > TIMELINE OUTLINE BATCH _ SIZE,
stride _ K_dim
stride _ V_batch
58
59
stride_ V_head,
stride_ V_seq,
stride V din 具体取决于当前的阶段.
62
63
stride_o _head,
which will be based on the stage.
OUTLINE 65
stride_o_seq,
stride_o_dim,
> TIMELINE 66
BATCH_ SIZE,
stride _ V _batch, 因此, 如果这是对该函数的第一次调用
stride _o _head,
So if it's the first call to this function,
OUTLINE 64
65
stride_o_seq,
stride_o_dim,
> TIMELINE BATCH _ SIZE,
BATCH _ SIZE, NUM _ HEADS, SEQ _ LEN, HEAD _ DIM 无论是因果还是非因果情况, 它们都将指向第一个块.
they will be pointing to the first block in the case of the causal and then on-causal.
> TIMELINE OUTLINE stride _ K _batch
stride _0_seq,
stride_ O_head,
54
stride _ Q_dim,
stride_ K_batch 如果这是对该函数的第二次调用
55
stride_ K_head,
stride_ K_seq,
8
stride_ K_dim,
If it's the second call to this function,
OUTLINE 58
stride_ V_batch
stride_ V_head,
> TIMELINE stride _ V _seq,
BATCH _ SIZE,
NUM _ HEADS, SEQ _ LEN, HEAD _ DIM (这种情况仅发生在因果注意力机制中)
which only happens in the case of the causal attention,
> TIMELINE OUTLINE stride _ Q _seq,
BATCH _ SIZE SEQ _ LEN, HEAD _ DIM 它们将精确指向对角线上的键和值块.
they will be-pointing exactly to the key and value block to the diagonal.
> TIMELINE OUTLINE stride _ O _seq,
好的, 接下来我们需要构建这个for 循环.
Eoc sz Alright, then we need to make the for loop.
> TIMELINE OUTLINE
55
stride_ K_batch, 那么, 让我们遍历所有需要执行的for循环, 现在就开始吧.
strl So. let's loop over all the for loops, so let's do it.
> TIMELINE OUTLINE 53
stride_v_dim,
55
stride_o_head,
56
57
stride_ O_dim,
stride _ K_bat stride _ K _he a
stride_ K_seq 接下来, 对键和值进行循环处理.
8
61
60
stride_ K_dim,
stride_ V_batch So loop over the key and value.
OUTLINE stride _ V_seq,
stride_ V_head,
> TIMELINE
63
stride _ O _head,
stride _ Q_seq,
stride_ Q_dim, 我们要做的就是, 没错, 按计划进行.
61
tride_ K_din,
And what we do is okay.
OUTLINE stride _ V _seq,
stride _v _head,
> TIMELINE 53
ide V dim
54
55
stride_ Q_head,
stride_ Q_seq,
6
stride_0_dim 我们让编译器知道, 这里的start KV值
weletthe'compiler know that this number here, the start Kv > TIMELINE OUTLINE
54
55
stride_ O_batch,
stride_ O_head,
stride _ O _seq,
stride _o_dim,
58
stride _ K_batch, 始终是 KV块大小的倍数
60
stride_ K_seq
stride_ K_head,
61
strdewill always be a multiple of the block size KV,
> TIMELINE OUTLINE 63
stride_v_head,
stride _ Q _batch stride _ Q_head,
stride_ Q_seq, 因为我们是一个接一个地从当前 KV块移动到下一个 KV块.
because we will be moving from one KV block to the next KV block, block byblock > TIMELINE OUTLINE
stride _ Q_seq,
stride_ Q_head, 因此, 我们让编译器明白, 这里的startkv值是 KV块大小的整数倍.
we let the compiler know that this number-here start k v-isamultiple of blocksize, kv > TIMELINE OUTLINE
54
55
stride_ Q_batch,
stride_ Q_head, 从逻辑角度来看, 这一点并不会带来任何变化.
It doesn't change anything from a logic point of view.
> TIMELINE OUTLINE
54
55
stride_0_batch,
stride_ O_head,
56
57
stride _ O_seq,
stride _ Q_dim,
stride_ K _batch, 我们只是
8
59
60
stride_ K_seq,
stride_ K_head,
62
stride_ V_batch
stride_ K_dim,
We are just telling,
> TIMELINE OUTLINE 63
stride_ V_head,
stride _ Q _batch,
stride _ Q _head, 向编译器提供一些提示, 以便它能执行 Triton 所具备的其他优化操作
giving some hint to the compiler so it can do some other optimization that triton does.
> TIMELINE OUTLINE
stride _ Q _head,
stride _ Q _batch, 在 Flash Attention 算法中, 首 首先映入眼帘的任务是
Now the first thing that we see in the flash attention algorithm is > TIMELINE OUTLINE
55
stride_ Q_head,
stride_ Q_batch,
stride _ Q _seq,
stride _ Q_dim, 计算查询的点积.
58
stride _ K _batch
60
61
stride_ K_head,
> TIMELINE OUTLINE 63
stride _0_head,
stride_ Q_batch 因此 这是我们当前迭代中正在处理的查询块
So this is the : particular block of the query that we are working with,
> TIMELINE OUTLINE
54
55
stride _ Q_batch,
BATCH_ SIZE, NUM_ HEADS, SEO_ LEN, HEAD_ IH
56
stride_ Q_head,
stride_ Q_seq,
stride_ Q_dim, 与 KV 块的组合.
60
59
stride_ K_head,
stride_ K_batch
61
stride_ K_dim,
with the current kv block in this iteration.
> TIMELINE OUTLINE 63
56
stride_ Q_head,
stride_ Q_batch,
stride _ Q _seq,
stride _o_dim, 那么, 让我们开始吧.
58
59
stride _ K _batch,
61
60
stride_ K_seq,
stride _ K _head,
So let's do it.
> TIMELINE OUTLINE 63
stride_ K_dim,
stride_ V_batch
stride _ Q _batch stride _ Q_head, 接下来, 我们计算 K和√的值, 这意味着我们需要加载相应的数据
stride_ K_dim,
stride_ K_seq,
stride _ K _head,
so we compute kan dv, so we load the.
> TIMELINE OUTLINE 63
stride_v_batch,
ide V head.
stride _ Q _batch,
stride _ Q _head, 查询部分的数据已经由该函数的调用者预先加载好了.
the query have already been loaded by the caller of this function.
> TIMELINE OUTLINE
我们已经在此处完成了数据的加载.
stride _ K _seq We invert the strides we have loaded it here.
TIMELINE OUTLINE block _shape =( HEAD _ DIH. BLOCK _ SIZE _ KV )
offsets =(θ, B ),
Stage :3if 我们已经加载了查询数据
if STAGE
o_block,
# This step c here we have already loaded the query.
> TIMELINE OUTLINE Li,
i
strides =(stride 不过我们还需要加载当前这一块的 K值数据.
107
but we. need to. load the current block of k.
OUTLINE _btock_ptr
basen V+qvk_uffset.
> TIMELINE shape =( SEO _ LEN, HEAD _ DIM )
61
stride_ K_batch,
tride_ K_head, 于是我们加载由k指针指向的当前块 K值数据
so we load the current block of k indicated by the k pointer and we multi > TIMELINE OUTLINE
61
52
stride_ Q_dim,
stride_ K_batch, 接着进行矩阵乘法操作, 即将当前块的查询数据
we do the matrix multiplication of the current block of query, the, the block of query,
> TIMELINE OUTLINE
stride _ K _batch,
stride _ K _head,
stride _ K_sec
stride_ K_dim,
stride_ V_b 与已转置的当前块键值数据相乘,
with. the current block of key which is already transposed > TIMELINE OUTLINE
61
stride _ Q_dim,
stride_ K _batch,
stride_ K_head, 因为在加载 K值数据时, 当我们定义 K块指针的时候
because when we loaded this k, k, when we defined the kb lock pointer,
> TIMELINE OUTLINE tride_0_head
dex _batch head =tl. prog associated wixh leach batehas ium_ ADS eads]
ndex_batch 就已经调整了其步长(stride)
92
we defined it already with the stride changed > TIMELINE OUTLINE qvk_ofts
snoex_barch. to irt. ant64)stride_0_hatch
strides =(stride _0_seq, stride_o_dim),
ffsets=(block in de 因此, 我们读取的张量已经是转置后的形式.
> TIMELINE OUTLINE 142
stride _ K _dim,
stride _v _batch, 因此, 我们正在进行的操作是查询数据与转置后的键值数据相乘.
so we are doing the query multiplied by the transpose of the keys.
> TIMELINE OUTLINE BATCH _ SIZE
stride _ O _dim, 简而言之:好, 现在让我们在这里进行操作.
stride _ V _batch,
basically : Okay, now let'sdo here.
OUTLINE 67
stride_ V_seq,
stride_ V_head,
> TIMELINE 59
stride_ V_dim,
stride _ Q _head,
stride _ Q _seq, 而这里的这部分代码基本上在说:如果当前阶段是第二阶段.
and this part here basically saying okay, if the stage is two.
> TIMELINE OUTLINE stride _ V _seq,
60
_attn_fwd(
NUM_ HEADS, SEQ_ LEN, HEAD_ DIM 当阶段为二时, 意味着我们正好处于对角线上.
8
when the stage is two is when we are exactly on the diagonal.
> TIMELINE OUTLINE stride _ Q _seq,
SEQ _ LEN,
HEAD _ DIM 我们知道, 部分查询的索引会大于键的索引
we know that some-of the queries will have an index that is bigger than that of the keys OUTLINE TIMELINE stride _ Q _seq,
UM _ HEADS, SEQ _ LEN, HEAD DIM 而另一些查询的索引则会小于键的索引.
and some of them will have an index that is smaller than that of the keys.
> TIMELINE OUTLINE stride _ Q _seq,
SEQ _ LEN, HEAD DIM 因此我们仅需在这种情况下应用因果掩码
So we need to apply the causal mask only in this case > TIMELINE OUTLINE stride _ Q _seq,
_attn_fwd(
SEQ_ LEN, HEAD DIM
2 所以我们的基本做法是定义需要应用的掩码
So basically what we do is we define the mask that we should be applying.
> TIMELINE OUTLINE stride _ Q _seq,
60
_attn_fwd(
SEQ_ LEN, HEAD_ DIM 因此, 掩码会屏蔽掉所有不符合条件的值.
So the mask will mask out all the values for which this mask is not true.
> TIMELINE OUTLINE stride _ O _seq,
60
_attn_fwd(
_ SIZE, NUM_ HEADS, SEQ_ LEN, HEAD_ DIM
61
62
BATCH_ SIZE, NUM_ HEADS, SEO_ LEN
# BATCH
63
64
softnax_scale,
BATCH _ SIZE, NUM_ HEADS, SEO_ 当掩码条件成立时,
65
BATCH _ SIZE, NUM _ HEADS,
BATCH SIZE,
NUM HEADS,
OUTLINE stride _ Q _batch,
stride _ Q _head,
So when this mask is true,
> TIMELINE 59
stride_ O_seq,
SEQ _ LEN, HEAD DIM soft na 即查询的索引大于键和值的索引时.
when the index of the gu ery is more than the index of the kand vis > TIMELINE OUTLINE stride _ Q _seq,
NUM _ HEADS,
SEQ _ LEN, HEAD _ DIM
EOLEN
DIM
54 然后, 我们应用
softmax进行缩放处理
OUTLINE
67
stride a bat And we, okay, we apply the soft max scale.
stride _ O _head > TIMELINE 59
stride_ Q_seq,
NUM _ HEADS,
SEQ _ LEN, HEAD DIM 正如您所记得的, 这里我们仅计算了查询与键转置的乘积
> TIMELINE OUTLINE stride _ O _seq,
# BATCH _ SIZE NUM _ HEADS, SEQ_ LEN, HEAD_ DIM
62 但还需要除以头维度的平方根.
EO LEN.
HEAD
sof tna x_scale,
but you also need to divide by the square root of head dimension.
> TIMELINE OUTLINE stride _ O _seq,
60
_attn_fwd(
SEQ_ LEN, HEAD_ DIM
61
62
# BATCH_ SIZE,
# BATCH
SIZE
NUM_ HEADS,
64
63
softnax_scale, 我们在此处完成这一步骤.
65
66
# BATCH _ SIZE,
NUM _ HEADS, SEQ _ LEN, HEAI
OUTLINE
62
stride_ Q_batch stride _ Q _head,
And we do it here.
> TIMELINE 69
stride_ O_seq,
61
BATCH
SIZE
SEQ_ LEN, HEAD _ DIM
62
# BATCH_ SIZE, 由于我们已经计算出了乘积
OLEN.
Um, and then we, because we already computed the, um, uh, the, the, the product,
> TIMELINE OUTLINE stride _ Q _seq,
60
_attn_fwd(
NUM_ HEADS, SEQ_ LEN, HEAD_ DIM
61
62
# BATCH_ SIZE BATCH SIZE OLEN
63
64
softnax_scale, 现在可以为每一行求出最大值.
66
65
OUTLINE 68 stride we can calculate the maximum for each row.
stride _ Q_he
> TIMELINE
69
stride_ O_seq,
_attn_fwd(
SEQ_ LEN, HEAD DIM
EAD 接着, 我们进行减法操作, 因为在 Flash Attention 算法后续步骤中
And then we, we we subtract because when later in the flash attention algorithm,
> TIMELINE OUTLINE stride _ Q _seq,
60
_attn_fwd(
NUM_ HEAD S, SEQ_ LEN, HEAD_ DIM ADS. SEO LEN 还有一项我称之为"sof tmax *"的运算.
we have another operation, which is the, which I call the soft max star.
> TIMELINE OUTLINE stride _ Q _seq,
_attn _fwd (
BATCH NUM _ HEADS, SEQ _ LEN, HEAD _ DIM
HEADS. SEO LEN.
AD 正如您所知,
softmax*需要对 S矩阵的每一行、每个元素执行操作.
> TIMELINE OUTLINE stride _ Q _seq,
正如您所知, soft max *需要对 S矩阵的每一行、每个元素执行操作
And as you remember, the soft max star needs to do each row, each element of the S matrix
即查询与键转置的乘积减去每一行的最大值.
So the query multiplied by the transport of the keys minus the maximum for each row.
因此, 我们已经可以计算出每一行的最大值了.
So we can already compute the maximum for each row.
60
def
_attn_fwd(
SEQ_ LEN, HEAD_ DIM 因此我们已经可以计算出每一行的最大值了.
So we can already compute the maximum for each row.
> TIMELINE OUTLINE stride _ Q _seq,
60
_attn_fwd(
# BATCH_ SIZE, NUM_ HEADS, SEO _ LEN, HEAD _ DI H
61
62
# BATCH _ SIZE, NUM _ HEAD S
63
# BATCH_ SIZE, NUM_ HEAD 在计算每行最大值之前
OUTLINE And we can a also, before computing the maximum for each row,
> TIMELINE stride _ Q _seq,
60
_attn_fwd(
61
62
BATCH_ SIZE,
# BATCH SIZE NUM _ HEADS,
SEQ _ LEN, HEAD _ DIM soft na x_scale
BATCH _ SIZE, 我们还需要屏蔽掉第二阶段
we need to mask out all the elements that will be masked out in the stage number two,
> TIMELINE OUTLINE stride _ Q _seq,
60
_attn_fwd(
NUM_ HEADS, SEQ_ LEN, HEAD_ DIM
61
62
# BATCH_ SIZE SIZE 64
softnax_scale, 沿对角线将被掩码的所有元素.
65
66
which is along the diagonal.
OUTLINE 62
stride_ Q_batch,
stride_ Q_head,
> TIMELINE 69
stride_0_seq,
61
62
BATCH_ SIZE, NUM_ HEADS, SE_ LEN, HEAD _ DIM BATCH _ SIZE, NUM _ HEADS, SEO_ EN
EAD DII
64
63
softnax_scale,
BATCH _ SIZE, NUM_ HEADS, SEO_ L 以及如何进行屏蔽,
55
BATCH_ SIZE, NUM_ HEADS, SEQ _ LEN,
BATCH _ SIZE, NUM _ HEADS, SEQ OUTLINE 67
62
stride_ Q_batch,
stride_ Q_head,
and how to mask out.
> TIMELINE 69
stride_0_seq,
# BATCH _ SIZE, NUM _ HEADS, SEQ _ LEN, HEAD _ DIM # BATCH _ SIZE, NUM_ HEADS,
62
63
BATCH_ SIZE, NUM_ HEADS 在应用soft max 之前
soft na x_scale,
we just replace with minus infinity, before applying the soft max,
OUTLINE TIMELINE stride _ Q _seq,
我们只需将所有对应掩码为假的值替换为负无穷大,
I the values for which the mask is false OUTLINE 68
stride_ Q_head
stride_ Q_batc
TIMELINE stride__seq,
HEADS,
SEQ _ LEN, HEAD _ DIM 嗯, 现在我们已经完成了什么计算呢?
> TIMELINE OUTLINE 62
stride_0_seq,
60
NUM_ HEADS, SEQ_ LEN, HEAD_ DIM 我们已经计算了查询与键转置的乘积.
we have computed the query multiplied by transport of the keys.
> TIMELINE OUTLINE stride _ Q _seq,
60
_attn_fwd(
62 我们已经进行了必要的掩码处理
we have masked out in case we need to mask and when we need to mask > TIMELINE OUTLINE stride _ O _seq,
SIZE,
SEQ_ LEN, HEAD _ DIM 并且仅在沿对角线的情况下才需要进行这样的掩码操作.
0,# BATCH_ SIZE,
only when we are along the diagonal.
OUTLINE 68
stride_ Q_batch,
stride_ Q_head,
> TIMELINE 69
stride_0_seq,
SEQ _ LEN, HEAD DIM 在所有其他情况下, 我们不需要屏蔽任何内容.
all-the other cases we don't need to mask out anything OUTLINE in > TIMELINE 59
stride_0_seq,
我们只需乘以softmax的缩放因子, 然后减去mij.
SEO LEN. HEAD DIM
we just multiply by the soft max scale and then we, um, we subtract the mij.
> TIMELINE OUTLINE stride _ Q _seq,
61
62
# BATCH _ SIZE, NUM _ HEADS, S
# BATCH
SIZE,
NUM_ HEADS, SEQ_ LEN, HEAD_ DIM
SEO LEN
63
64
soft na x_scale,
# BATCH _ SIZE, NUM_ H 代表每一行的最大值
65
66
BATC
SIZE OUTLINE 68
stride_obathemij
stride_ Q_head is the maximum value for each row,
> TIMELINE 69
stride_ O_seq,
6
ATCH
NUM_ HEAD S, SEQ_ LEN, HE AD_ DIM
EO LEN. HEAD DII 因为我们需要计算soft max *号运算
OUTLINE because v we need to compute the soft max star operation,
> TIMELINE stride _ Q _seq,
51
SEQ_ LEN, HEAD DIM SEO LEN. HEAD DIM 即不带归化的soft max. 在 Flash Attention 算法中
which is the soft max without the normalization, which, in the flash attention algorithm,
> TIMELINE OUTLINE stride _ Q _seq,
即不带归一化的soft max. 在 Flash Attention 算法中
which is the soft max without the normalization, which, in the flash attention algorithm,
这一运算正是生成pij 的过程.
is exactly this operation which will produce the pij.
61
62
# BATCH
BATCH_ SIZE,
SIZE
SEQ_ LEN,
HEAD _ DIM
EA D
OIM
63
64
softnax_scale, 好的, 那我们继续往下看.
65
66
# BATCH _ SIZE,
NUM _ HEADS, SEQ _ LEN,
OUTLINE 62
stride_ Q_batch
stride_ Q_head,
okay, so let'sgo here.
> TIMELINE 69
stride_ O_seq,
otrir
on. jit 现在我们可以计算pij块了, 它位于这里
so now we can compute the pij block, which is this stuff here,
> TIMELINE OUTLINE stride _ Q _batch,
是查询和 KV 块变量的指数运算结果
which is the exponential of the query, Kv block variable here,
> TIMELINE OUTLINE BATCH _ SIZE, NUM _ HEADS, SEO _ LEN, HEAD _ DIM
60
62
63
61
@trito n. jit
_attn_fwd( 其中已经减去了 M.
64
65
# BATCH_ SIZE, NUM_ HEADS,
NUM HEADS
66
BATCH_ SIZE,
which have already subtracted the M OUTLINE 67
68
softnax_scale,
# BATCH _ SIZE > TIMELINE 69
D.
BATCH_ SIZE,
NUM_ HEADS, SEO_ LEN, HEAD _ DIM
在之前的步骤中, 我们已经减去了这个 MI.
have already subtracted this Ml at the previous instruction.
OUTLINE So we > TIMELINE BATCH _ SIZE, NUM _ HEADS, SEO _ LEN, HEAD _ DIM
在之前的步骤中, 我们已经减去了这个 MI.
So we have already subtracted this Ml at the previous instruction.
因此, 现在我们可以直接进行指数运算了
So now we can just apply the exponential.
因此, 现在我们可以直接进行指数运算了
soft max sau So now we can just apply the exponential > TIMELINE OUTLINE BATCH _ SIZE, NUM _ HEADS, SEO _ LEN, HEAD _ IH
61
@triton. jit
64
63
_attn_fwd(
# BATCH _ SIZE 这就是我们在这里所做的操作.
66
65
BATCH_ SIZE,
# BATCH SIZE OUTLINE 67
68
softnax_scale,
BATCH _ SIZE,
And this is what we are doing here.
> TIMELINE 69
D.
# BATCH_ SIZE,
NUM_ HEADS, SEO_ LEN, HEAD _ DIM
60
62
61
63
@triton. jit
_attn_fwd(
64
65
Q
BATCH_ SIZE, NUM_ HEAD S, SEO_ LEN, HEAD_ DIM # BATCH _ SIZE, NUM _ HEADS, SEQ _ LEN, HEAD _ DIM
8
66
OUTLINE
67
68
softnax_scale,
BATCH _ SIZE, NUM _ HEADS, SEO _ LEN Okay > TIMELINE 59
D.
# BATCH_ SIZE, NUM_ HEADS, SEO_ LEN, HEAD _ OIH
接下来我们需要计算各行的总和, 以确定归一化因子.
Then we need to compute the sum of the rows for the normalization factor.
> TIMELINE OUTLINE # BATCH _ SIZE, NUM _ HEADS, SEO _ LEN
Compute the sum by
_ij= tl. sum[ P_block, 1]
t:attention scores 因此, 对于当前的块, 我们将获得一个列表
So for the current block, we will have a list of,
> TIMELINE OUTLINE BATCH _ SIZE, NUM _ HEADS, SEQ _ LEN, HEAD _ DIM
_ij=tl. sum[ P_block, 1]
of the attention scores 其中包含当前 kv 块对应的 pij 块.
66
etriton. jit
_attn_fwd(
OUTLINE 67
62
we have the pij block for the current kv block.
> TIMELINE BATCH _ SIZE, NUM _ HEADS, SEQ _ LEN, HEAD _ DI
_ij =tl. sun[ P_block, 1
of the attention scores 为了计算softmax 的归一化因子
66
to n. jit
OUTLINE
67
Tocompute the normalization factor for the soft max,
> TIMELINE # BATCH _ SIZE, NUM _ HEADS, SEQ _ LEN, HEAD _ DIM
Compute the sum by rows
_ij=tl. sum P_block, 1]
of the attention scores 我们需要持续累加这些指数运算的结果.
66 we need to keep summing up these exponentials.
OUTLINE 67 > TIMELINE # BATCH _ SIZE, NUM _ HEADS, SEQ _ LEN, HEAD _ DIM
# Compute the sum by r
_ij= tl. sum[ P_block, 1]
of the attention scores 随后, 我们将调整这些指数运算的结果
66
65
otrit
_attn_fwd(
def
# BATCH_ SIZE,
And later we will fix the exponentials > TIMELINE OUTLINE 62
59
BATCH_ SIZE, NUM_ HEADSSEO_ LEN, HEAD _ DI BATCH _ SIZE,
No quick fixes availlable =tl. sum( P_block, 1)
attention scores
61 以及上一步计算出的归一化因子.
on. jit
TIMELINE OUTLINE BATCH _ SIZE, NUM _ HEADS, SEQ _ LEN, HEAD _ DIM
=tl. sum( P_block, 1)
No quick fixes available attention scores 不过, 这部分工作我们稍后再进行.
66
65
etriton. jit
_attn_fwd(
67
def Q. BATCH _ SIZE, NUM _ HEADS, SEO _
But we will do that later.
> TIMELINE OUTLINE 62
59
BATCH_ SIZE,
BATCH_ SIZE, NUM_ HEADS, SEO _ LEN, HEAD _ DIM NUM _ HEADS,
=tl. sum( P_block, 1)
No quick fixes available attention scores 现在, 我们刚刚计算了当前块的归一化因子
So now we just computed the normalization factor for the current block,
> TIMELINE OUTLINE BATCH _ SIZE, NUM HEADS, SEQ _ LEN, HEAD _ DIM
=tl. sum( P_ock, 1) 它就是单行中所有数值的总和.
66
which is just the sum of all the values on a single row.
to n. jit
> TIMELINE OUTLINE BATCH _ SIZE, NUM _ HEADS, SEQ _ LEN, HEAD _ DIM
Compute the Sum by
i]=tl. sun( P_block, 13
of the attention scores 正如这里所示, 这与我们之前所做的操作相同
Which is the same as what we did before here, as you can see here.
> TIMELINE OUTLINE BATCH _ SIZE, NUM _ HEADS, SEQ _ LEN, HEAD _ DIM
在介绍算法时, 对于每个块, 我们都会对 P矩阵进行行求和操作
Q
正如这里所示.
of the P matrix.
P 矩阵是什么呢?
What is the P matrix?
它是 S减去 M后的指数运算结果.
It's the exponential of the S min us M.
目前, 我们尚未对前一个块进行修正处理.
And for now we didn't apply the correction to the previous block.
就是这样.
That's it.
Compute the sum
_ij=tl. sun[ P_block, 1]
the attention scores 因此, 我们已经计算出了当前 K 和 V 块的 LIJ So we computed the Li J for the current Kan d V block.
> TIMELINE OUTLINE # BATCH _ SIZE, NUM _ HEADS, SEQ _ LEN, HEAD _ DIM
Compute the sum byr
_ij=tl. sum( P_block, 1)
of the attention scores 接着, 我们计算前一个块的修正因子
And then we compute the correction factor for the previous block.
> TIMELINE OUTLINE BATCH _ SIZE, NUM_ HEADS, SEO_ LEN, HEAD_0 IM
60 # Compute the sun by rows of the attention scores l_ij=tl. sum( P_block, 1)
63 前一个块的修正因子
8
etrito n. jit So the correction factor for the previous block -
> TIMELINE OUTLINE BATCH _ SIZE, NUM _ HEADS, SEO _ LEN, HEAD _ DIM
# Compute the sum by rows of the attention scores _ij=tl. sum( P_block, 1) 如果你还记得论文中的公式一一是这样的
8
if you remember the formula from the paper-is this :
> TIMELINE OUTLINE # BATCH _ SIZE, NUM _ HEADS, SEO _ LEN, HEAD _ DIM
如果你还记得论文中的公式一一是这样的:
if you remember the formula from the paper - is this :
1 乘以指数部分, 即先前估计的最大值
one is the exponential of the previous estimate of the maximum,
减去当前估计的最大值, 正好就是这一项.
minus the current estimate of the maximum, which is exactly this one.
# Compute the sum by rows of the attention scores l_ij=tl. sum( P_block, 1)
52 减去当前估计的最大值, 正好就是这一项.
minust
the current estimate of the maximum, which is exactly this one.
> TIMELINE OUTLINE def _attn _fwd (
# Compute the sum by rows of the attention scores _ij =tl. sum( P_block, 1)
61 即先前估计的最大值减去当前估计的最大值.
So the previous. estimate of the maximum minus the current estimate of the maximum.
> TIMELINE OUTLINE def _attn _fwd (
# Compute the sum by rows
_ij=tl. sum( P_block, 1)
of the attention scores 稍后我们会解释为什么 MI 代表先前估计的最大值
We will see later why Ml is the previous estimate of the maximum TIMELINE OUTLINE def _attn _fwd (
# Compute the sun by rows of the attention scores l_ij=tl. sum( P_block, 1)
This 而 MIJ则代表当前估计的最大值
and why Mi J is the current estimate of the maximum,
TIMELINE OUTLINE 9
def _attn_fwd(
# Compute the sum by rows
_ij=tl. sum( P_block, 1)
of the attention scores 因为后者源自我们正在计算的当前块,
because it is coming from the current block that we are computing TIMELINE OUTLINE _attn _fwd (
# Compute the sum by rows of the attention scores l_ij =tl. sum( P_block, 1)
61
62
# This is the correction factor for the MI可以说是
64
65
66
Mlis the, let'ssay, the one that,
OUTLINE 6>
@triton. jit
> TIMELINE
69
def _attn _fwd (
Compute the sum by rows
l_ij=tl. sum( P_block, 1)
of the attention scores 前一次迭代的结果, 因为之后我们会用 MIJ 来覆盖 MI.
it is the one of the previous iteration because later we will override Ml with Mi J.
> TIMELINE OUTLINE def _attn _fwd (
l_ij=tl. sum( P_block, 1)
of the attention scores 不过到目前为止我只是在按 Flash Attention 算法的步骤进行.
But I'm just following the flash attention algorithm so far > TIMELINE OUTLINE 9
def _attn_fwd(
# Compute the sum by rows
_ij=tl. sum( P_block, 1)
of the attention scores 52 因此我正在计算前一个 LI的修正因子
54
Solamcomputing the correction factor of the previous Li,
> TIMELINE OUTLINE def _attn _fwd (
Compute the sun
_ij=tl. sum( P_block, 1)
of the attention scores 这 在 Flash Attention 算法中是这样的:让我在这里展示一下这部分内容
which in the flash attention algorithm is :let me show you this stuff here.
> TIMELINE OUTLINE def _attn _fwd (
15: 在 Flash A
Attention 算法中是这样的: 让我在这里展示一下这部分内容,
which-in the flash attention algorithm-is : let me show you this stuff here.
17 :
Return the output O and'the log sum exp
15:
Write Lto HBMasthe i-thblockof L.
16:end for
17: Return the output O and the log sum exp L.
15:
Write L;to HBMasthei-thblock of L 所以就是这里的这一部分, 对, 就是这个, 然后我们应用它.
so it is this stuff here, this one here, okay, and then we apply it.
l_ij= tl. sum( P_block, 1)
of the attention scores 所以就是这里的这一部分, 对, 就是这个, 然后我们应用它.
so it is this stuff here, this one here, okay, and then we apply it > TIMELINE OUTLINE def _attn _fwd (
60 # Compute the sum by rows of the attention scores l_ij= tl. sum( P_block, 1) 于是, 我们应用这个修正因子, 就这样进行了应用
soapply the correction factor, so we apply it.
> TIMELINE OUTLINE 59
@triton. jit
def _attn_fwd(
2 因此, 我们将前一次的 Li与修正因子相加, 再加上当前的 Li
Sowe apply the previous Li with the correction factor plus the current Li > TIMELINE OUTLINE _attn _fwd (
This is the correction factor for the previous 1_i
=tl. math. exp(m_i
i*alpha correct io 后者来自当前的 P块
which is the one coming from the current Pblock > TIMELINE OUTLINE @triton.
def _attn _fwd (
This is the cor 是我们通过当前 KNV? 在当前迭代中计算得出的
the one. that we computed with the current K NV, with the current iteration.
> TIMELINE OUTLINE def _attn _fwd (
# This is the correction factor for the previous l_i
_ij)
67 目前, 我们正在执行这一操作.
70
And right now we are doing this operation.
> TIMELINE OUTLINE @triton. jit def _attn _fwd (
15:
Write Lto HBMasthe i-thblockof L
16:end for 前, 我们正在执行这一操作.
17: Return the output O and the log sum exp
And right now we are doing this operation.
15:
Write L;to HBM asthe i-thblock of L.
Li等于前一次的 Li乘以修正因子.
17: Return the output O and the log sum exp So Li is equal to the previous Li multiplied by the correction factor.
15:
Write Lto HBMasthe i-thblockof L.
16:endfor
17: Return the output O and the log sum exp L
alpha 好的, 那么接下来我们需要做什么呢?
69
70
all right, and then what we need to do?
> TIMELINE OUTLINE @triton. jit def _attn _fwd (
59 好的, 如您所记得的, 公式是这样的
okay, we need to, as you remember, the formula is um,
> TIMELINE OUTLINE
我们计算 P块, 然后需要将其与 V块相乘
calculate the pblock and then we need to multiply by the v block.
OUTLINE we um > TIMELINE _attn _fwd
69 所以我们需要加载√块.
70
sowe need to load the v block OUTLINE 72
@triton. jit
> TIMELINE def _attn _fwd (
67 # Apply the correction factor _i=l_i*alpha+l_ij
69 那就让我们加载它吧,
8
70
so let's load it.
OUTLINE 72
@triton. jit
> TIMELINE def _attn _fwd (
我们根据指向 V块的指针 V所指示的位置来加载 V块.
8
Weload
> TIMELINE OUTLINE 3
@triton. jit
def _attn_fwd(
我们根据指向 V块的指针 V所指示的位置来加载 V块.
the V block based on the pointer of the V block to which the pointer V is pointing to.
> TIMELINE OUTLINE _attn _fwd
在此次迭代开始时, 如果我们处于第三阶段
At the beginning of this iteration, in case we are in stage number three,
TIMELINE OUTLINE _attn _fwd (
of the attention scores 也就是说, 如果我们正在执行非因果注意力机制
so in case we are doing, for example, not causal attention,
> TIMELINE OUTLINE
_ij=tl. su
the attention scores # This is t 那么指针将指向第一个 KV块.
65
64
a Lpha it will be pointing. to the first K V block.
OUTLINE
66
> TIMELINE
然后, 好的, 这里只是一个类型转换
> TIMELINE OUTLINE
71
P_block = P_block. to(t1. float16] 确保数据以16位浮点数格式存储
75
@triton. jit
77
_attn_fwd(
so we make sure this is in floating point 16.
> TIMELINE OUTLINE BATCH _ SIZE, NUM _ HEADS, SEO _ LEN, HEAD _ DIM BATCH
71
P_block = P_block. to (tl. float16)
74 接着, 我们计算输出块.
76
And then we compute the output block.
OUTLINE 77
78
def_attn_fwd(
etriton. jit
TIMELINE BATCH _ SIZE, NUM _ HEADS, SEO _ LEN, HEAD _ DIM
71
P_block = P_block. to (tl. float16)
74 因此, 我们正在计算以下内容.
75
76
So we are computing the following.
OUTLINE 77
@triton. jit
def_attn_fwd(
> TIMELINE # BATCH _ SIZE, NUM _ HEADS, SEQ _ LEN, HEAD _ DIM
P_block= P_block. to(tl. float16) 于是, 我们只需取 V和 P, 将其与 V相乘, 并将结果累加到输出中
Sowejust take V, P, multiply it by V and we add it to the output.
> TIMELINE OUTLINE BATCH _ SIZE, NUM _ HEADS, SEQ _ LEN, HEAD _ DIM
# This 0_block= 这正是我们在此处所执行的操作
75
o_block
OUTLINE
78
@triton. jit And this is what we are doing here.
> TIMELINE def _attn _fwd (
7
P_block= P_block. to(tl. float16) 我们取 P;将其与 V相乘, 并将结果累加到 O块中
We take P, we multiply it by V and add it to the O block.
> TIMELINE OUTLINE def _attn _fwd (
P_block= P_block. to (tl. float16)
73
This co
block= 让我们逐行深入解析这段代码.
75
OUTLINE 78
@triton. jit
Let'sgoactually to this line one by one.
> TIMELINE def _attn _fwd (
P_block= P_block. to(tl. float16) 首先, 我们需要利用修正因子来调整先前的输出块.
So first of all, we need to fix the previous output block with the correction factor.
> TIMELINE OUTLINE
P_block= P_block. to(tl. float16)
74
73
# This comp block = O _block *alsha 这里提到的修正因子.
75
o_block =tl. dot( P_block
Correction factor that we have here.
> TIMELINE OUTLINE @triton. jit def _attn _fwd (
P_block= P_block. to(tl. float16) 因此;我们可以通过这里的alpha 项来修正前一个块
So we can fix the previous block with this alpha term here,
> TIMELINE OUTLINE
P_block= P_block. to(tl. float16) 这个alpha 项就是针对前一个块的修正因子
which is the correction factor for the previous block OUTLINE 8
gtriton. jit
> TIMELINE def _attn _fwd (
71
P_block = P_block. to (tl. float16) 目前, 我们仅修正了前一个块, 但尚未加入新的 PV值.
ter] o_block: An
And so we just fixed the previous block for now, but we didn't add the new PV.
> TIMELINE OUTLINE _attn _fwd (
P_block= P_block. to (tl. float16) 为了加人新的 PV值, 我们需要计算 P 和 V的点积.
Sotoadd the new PV, we do the dot product of Pan d V.
> TIMELINE OUTLINE
P_block= P_block. to(tl. float16) 这里的第三个参数指示矩阵乘法一一注意, 这不是点积
And this third argument tells the dot, this not dot product,
> TIMELINE OUTLINE def _attn _fwd (
71
70
P_block = P_b
)->tensor
75
o_block =tl. dot( PLlock, V_bloc
O_block=0b 而是矩阵乘法
7
78
it's actually the matrix multiplication,
> TIMELINE OUTLINE @triton. jit def _attn _fwd (
P_block = P_b
)-tensor
73
# This cc 使用此处的元素作为累加器,
75
o_block=tl. d
tell this matrix multiplication to use this element here as the accumulator.
> TIMELINE OUTLINE
P_block= P_block. to(tl. float16)
73 这完全等同于将 P块与 V块相乘
So this is exactly the same as doing Pblock multiplied by the V block OUTLINE TIMELINE
P_block = P_block. to(tl. float16)
O块加上 P块与 V块的乘积结果
Oblockplus equal to P block multiplied by the V block.
> TIMELINE OUTLINE
P_block= P_block. to(tl. float16)
75 这里之所以进行优化, 是因为无论如何,
This is just optimized because anyway,
> TIMELINE OUTLINE 78
@triton. jit
def _attn_fwd(
P_block= P_block. to(tl. float16) 这个点积函数都需要一个地方来存储中间结果
this dot function here needs some place where to store the intermediate results.
> TIMELINE OUTLINE
7
P_block= P_block. to(tl. float16) 那么, 何不直接将其存储在最终应该存放的位置呢?
So why not just store it where it should actually go?
TIMELINE OUTLINE @triton. jit def _attn _fwd (
P_block = P_block. to(tl. float16)
O_block=
block= 由于矩阵乘法本质上就是点积
And because the dot, the matrix multiplication,
TIMELINE OUTLINE gtriton. jit
lef_attn_fwd(
P_block= P_block. to(tl. float16) 点积又是一系列相加的重复过程
is just a dot product and the dot product is just a repeated sum,
> TIMELINE OUTLINE
P_block= P_block. to(tl. float16)
73 这个累加器会持续将结果加到这个块上
this accumulator will be, this dot will keep summing the result to this block here,
TIMELINE OUTLINE
71
P_block = P_block. to (tl. float16)
# This computes the following : O_new= Px
0_block= O_block*alpha[:, None ] 最终的效果
75
o_block =tl. dot( P_block, V_block, o_btod)
which will exactly result in this instruction,
TIMELINE OUTLINE @triton. jit def _attn _fwd (
P_block= P_block. to(tl. float16) 与我们单独进行矩阵乘法后再加到0块上完全一致.
like we have done the matrix multiplication separately and we added it to the O block.
> TIMELINE OUTLINE f_attn _fwd (
P_block= P_block. to(tl. float16) 正因如此 这个参数被称为累加器
So this is, that's why this argument is called the accumulator.
> TIMELINE OUTLINE _attn _fwd (
71
P_block = P_block. to (tl. float16) 好的如此一来我们不仅计算出了输出结果
All right, so we have also computed the output > TIMELINE OUTLINE @triton. jit def _attn _fwd (
P_block = P_block. to(tl. float16)
75 还更新了当前送代中的最大值估计值
and then we save the new estimation of the maximum for the current iteration > TIMELINE OUTLINE f_attn _fwd (
73
72
# This computes the following :0_
74
0_block =0_block*alpha[:,
0_block=tl. dot( P_block, V_block
77
n_i ==_ij 并将其记为 MI.
79
and it becomes Ml.
> TIMELINE OUTLINE @triton. jit def _attn _fwd (
这样, 在下一轮送代中, 我们就可以利用它来计算校正因子了.
So at the next iteration, we can use it to calculate the correction factor.
TIMELINE OUTLINE def _attn _fwd (
72 当前块的处理至此告一段落, 接下来我们可以继续处理下一个块了
And then we have finished for the current block and then we can move on to the next block OUTLINE TIMELINE
77 于是, 我们将 K和 V的指针各自向前移动一个 K和 V块的距离
Soweadvanceour Kand V pointers by one block of Kan d V.
> TIMELINE OUTLINE def _attn _fwd (
77 由于我们清楚 V块指向的是一个特定形状的张量
Weadvance it differently because we know that > TIMELINE OUTLINE @triton. jit def _attn _fwd (
因此我们对它们的移动方式做了区分处理
83
the Vblock is a pointer to a tensor of shape OUTLINE @triton. jit > TIMELINE def _attn _fwd (
77
n_i = =_1j
# Move to the next block of K and V
_block _pt r=tl. advance (v _block _ptr,
( BLOCK _ SIZE _ KV, 0))
Kblock _ptr=tl. advance( Kblock_ptr,(0, BLOCK_ SIZE_ KV)]
8. 2
83
> TIMELINE OUTLINE @triton. jit def _attn _fwd (
77
n_i = =_1j
80
_block_ptr=tl. advance [v_blc Move to the next block of K 让我在这里写一下.
K_block_ptr=tl. advance( K_bl
83
Let me write it here.
OUTLINE 84
@triton. jit
> TIMELINE def _attn _fwd (
77
n_i = =_1j
# Move to the nex
v_block _ptr=tl. 这是一个形状为... 的张量
81
K_block_ptr=t
83
84
This is a tensor of shape.
> TIMELINE OUTLINE @triton. jit def _attn _fwd (
77
n_i = =_ij
79
80
Move to the next block of K
_block_ptr=tl. advance(v_b 序列长度,"头维度.
83
sequence length, head dim.
OUTLINE
84
@triton. jit
> TIMELINE def _attn _fwd (
77
n_i = =_ij
# Move to the next block of K and V
V_block _ptr =tl. advance(v_block _ptr,
( BLOCK_ SIZE _ KV, 0))# V[ SEQ_ LEN, HEAD_ DIM]
81
K_block _pt r= tl. advance ( K_btock_ptr,(0, BLOCK_ SIZE_ KV))]
83
> TIMELINE OUTLINE 84
@triton. jit
def_attn_fwd(
77
n_i = =_ij 因此, 我们需要将序列长度增加一个 KV块的大小
Sowe need to increase the sequence length by one block size, kv,
> TIMELINE OUTLINE _attn _fwd (
n_i = =_ij Move to the _block _ptr 而 K块实际上是 K的转置块
_block pt
8
83
whilethekblockis actually the k transpose block OUTLINE g triton. jit > TIMELINE def _attn _fwd (
n_i = n_ij 所以我们 需要这样做, 并且由于我们交换了步幅和形状, 它实际上是一个转置操作
so we need to and it is a transpose because we have exchanged the strides and the shape,
> TIMELINE OUTLINE
n_i = =_1j
V_block_ptr=
Move to the ne 因此它现在代表的是头维度
81
K_block_ptr
83
84
so itishead dimension.
> TIMELINE OUTLINE @triton. jit def _attn _fwd (
77
n_i = =_ij
81
K_block _ptr
_block_ptr=tl. advance( V_block 头维度序列长度.
83
head dimension sequence length.
> TIMELINE OUTLINE 84
@triton. jit
def _attn_fwd(
n_i ==_ij _block _ptr =t Move to the ne 因此我们无需改变头维度
81
K_block_ptr
83
Sowe don'tchange the head dimension,
OUTLINE 84
@triton. jit
> TIMELINE def _attn _fwd (
n_i = n_ij 只需将序列长度按 KV 块的大小递增
we just advance these g uence length by segue nce block size kv.
TIMELINE OUTLINE _attn _fwd (
n_i = =_ij 简单来说:我们只需指向下个 K块和下一个 V 块.
So basically we are just going to point to the next block of k and to the next block of v.
> TIMELINE OUTLINE lef _attn _fwd (
77
76
n_i = =_ij
78
# Move to the next block of K and V
Be
K_block_ptr= tl. advance ( K_btock_ptr,(e, BLOCK _ SIZE_ KV))
V_block_pt r=tl. advance ( V _block _ptr,
( BLOCK _ SIZE _ KV, 0))
84
83
> TIMELINE OUTLINE @triton. jit def _attn _fwd (
love to the next block of K and V
tl. advance ( V _block _ptr,( BLo CK _ SIZE _ KV, o)) 希望您能理解 Flash Attention 算法的工作原理.
I hope you were able to follow the algorithm of flash attention.
TIMELINE OUTLINE _attn _fwd (
Move to the next block of K and V
K_block_ptr
( BLOCK_ SI ZE_ KV, 0)) 我尽量保持了相同的命名方式.
85
Itried to use the same names.
> TIMELINE OUTLINE @triton. jit def _attn _fwd (
Move to the next block of K and V
K_block_ptr=tl. ad
_block_ptr=tl. advance ( V_block_ptr,( BLo CK_ SIZE_ KV, 0)) 我尽量采用了相似的逻辑
85
Itried to use more or less the same logic > TIMELINE OUTLINE @triton. jit def _attn _fwd (
( BLOCK _ SIZE _ KV, 0)) 并在每一步都标注了所涉及的公式.
and always writing the formula that I am referring to.
> TIMELINE OUTLINE def _attn _fwd (
# Move to the next block of K and V
K_block_ptr=tl. advance (
( BLOCK _ SIZE _ KV, 0)) 希望您没有感到困惑.
85
So hopefully you didn't get lost.
> TIMELINE OUTLINE @triton. jit def _attn _fwd (
( BLOCK _ SIZE _ KV, 0)) 我认为论文中的 Flash Attention 算法
I think the only difference > TIMELINE OUTLINE @triton. jit def _attn _fwd (
# Move to the next block of K and V
K_block_ptr 与这段代码的唯一区别可能在于
( BLOCK_ SIZE_ KV, 0))
that there is between the flash attention algorithm as written on the paper > TIMELINE OUTLINE
# Move to the next block of K and V
K_block_ptr
( BLOCK_ SIZE_ KV, 0)) 这个alpha 值, 它是修正因子.
and this code is probably this alpha, which is the correction factor.
> TIMELINE OUTLINE f_attn _fwd (
_block 我希望这一点是易于理解的
( BLOCK _ SIZE _ KV, 0)) 品 不过,
85
but Ihopeit'seasily understandable > TIMELINE OUTLINE @triton. jit def _attn _fwd (
_blo
( BLOCK_ SIZE_ KV, 0)) 总之, 我们最终只需返回○块即可.
85
Anyway, then we just return the O block > TIMELINE OUTLINE @triton. jit def _attn _fwd (
79
( BLOCK_ SIZE_ KV, 0))
82
K_block_ptr= tl. advance ( K _block _ptr,
(0, BLOCK _ SIZE_ KV)) 所以,〇块.
84
returl
8
85
So O block.
OUTLINE 87
@triton. jit
def_attn_fwd(
> TIMELINE # BATCH _ SIZE, NUM _ HEADS, SEQ _ LEN, HEAD _ DIH
79 # Move to the next block ( BLOCK _ SIZE _ KV, 0))
81
K_block _ptr =tl. advance( K_block _ptr,
(, BLOCK_ SIZE _ KV))
84 是当前输出块(也是一个 Q块)中
Ll, which is the normalization factor for each row in the current output block > TIMELINE OUTLINE BATCH _ SIZE, NUM _ HEADS, SEQ _ LEN, HEAD _ DIH
80
V_block_ptr=tl. advance ( V_btock_ptr,( BLOCK_ SIZE_ KV, 0))
# Move to the next block of 82
K_block_ptr=tl. advance ( K_block _ptr,(0, BLoc K_ SIZE _ KV))
84
83
return o_block,_i, 每一行的归一化因子
85
which is also a Q block OUTLINE 87
@triton. jit
_attn_fwd(
> TIMELINE # BATCH _ SIZE, NUM _ HEADS, SEO _ LEN, HEAD _ DIH
V _block _pt r=tl. advance ( V_block_ptr,( BLOCK_ SIZE_ KV, 0))
# Move to the next block K _block _pt r=tl. advance ( K_block_ptr,(0, BLoc K_ SIZE_ KV)) 因为我们正在独立处理一个 Q块, 与其他程序无关:
because we are working with one Q block independently from the other programs,
> TIMELINE OUTLINE BATCH _ SIZE, NUM _ HEADS, SEO _ LN, HEAD _ I
v _block _pt r=tl. advance ( V_btock_ptr,( BLOCK_ SIZE_ KV, 0))
# Move to the next block of K and V
81
K_block_ptr =tl. advance ( K_block_ptr,(0, BLo CK_ SIZE_ KV))
84
return o _block, l_i, 而 MI则是每一行的最大值
and Ml is the maximum value for each row.
OUTLINE 87
@triton. jit
_attn_fwd
> TIMELINE # BATCH _ SIZE, NUM _ HEADS, SEO _ LEN, HEAD _ DIM
Vblock _ptr =t1. advance( V_block _ptr,( BLOCK_ SIZE _ KV, 0))
K_block_ptr=tl. advance ( K_block_ptr,(0, BLoc K_ SIZE_ KV))
84 这些值在反向传播过程中会派上用场
which will be needed for the backward pass.
OUTLINE g triton. jit attn > TIMELINE # BATCH _ SIZE, NUM _ HEADS, SEQ _ LEN, HEAD _ DIH
79
_block _pt r=tl. advance( K_block _ptr,(e, BLoc K_ SIZE _ KV ) 因为在反向传播时, 我们会动态计算gk 查询乘数转移
because when, in. the backward pass, we will compute the qk query multiplier transfer,
TIMELINE OUTLINE BATCH _ SIZE, NUM _ HEADS, SEQ _ LEN, HEAD _ DIM
V_block _ptr =t1. advance V_block _ptr,( BLOCK_ SIZE _ KV,)
# Move to the next block of K and V
K_block _ptr=tl. advance( K_block _ptr,(0, BLoc K_ SIZE _ Kv )) 和键块, 同时还需要应用soft max 函数,
the u key block on the fly-we need to also apply the soft max.
> TIMELINE OUTLINE BATCH _ SIZE, NUM _ HEADS, SEQ _ LEN, HEAD _ DIM
79
V_block_ptr=t1. advance( V_btock _ptr,( BLOCK _ SIZE _ KV,))
# Move to the next bloc
K_block _ptr=tl. advance( K_block _ptr,(0, BLoc K _ SIZE _ Kv )) 与其在反向传播时重新计算这些已在正向传播中完成的内容
but instead of recomputing this stuff, which we already computed during the forward pass,
OUTLINE TIMELINE BATCH _ SIZE, NUM _ HEADS, SEQ _ LEN, HEAD _ DIM
79
V_block _ptr =tl. advance( V_block _ptr,( BLOCK_ SIZE _ KV, 0))
K_block_ptr=tl. advance ( K_block_ptr,(e, BLo CK_ SIz E_ KV))
84 这样可以节省一些计算量.
which will save us some computation.
OUTLINE 87
@triton. jit
def_attn_fwd(
> TIMELINE # BATCH _ SIZE, NUM _ HEADS, SEQ _ LEN, HEAD _ DIH
79
V_block _ptr =tl. advance( V_block _ptr,( BLo CK_ SIz E _ KV,θ))
K_block _pt r=tl. advance ( K _block _pt r,(e, BLoc K _ SIZE _ KV ))
, 现在我觉得是时候讨论一下对数求和技巧了
um, now i know it's time to talk about the log sum x trick,
> TIMELINE OUTLINE BATCH _ SIZE, NUM _ HEADS, SEQ _ LEN, HEAD _ DIM
K _block _ptr=tl. avance( K_block _ptr,(0, BLo CK_ SIZE _ KV )) 因为我们将要用到它. 所以, 让我们先回顾一下旧方法.
because we are going to use it, so Let'sgo back to the old method.
> TIMELINE OUTLINE BATCH _ SIZE, NUM _ HEADS, SEQ _ LEN, HEAD _ DIM
SEQ _ LEN,
If STAGE 那么, 我们继续往下讲
D_block, f, f=_att_fwd_innerl
OUTLINE
0_blork,
So let's go here.
> TIMELINE
0=torch. empty_like( Q)
stage=3if causal else 1
grid= lambda args:( 好的
#cei L( SEQ_ LEN / BLOCK _ SIZE _ Q )=
triton. cd iv( SEQ_ LEN, args[ BLOCK_ SIZE_ O"1),
All right,
going to work with?
OUTLINE 1,# Z in the CUDA launch grid BATCH _ SIZE * NUM _ HEADS,# Which head of whic going to work with?
> TIMELINE
0=torch. empty_like( Q) 已经对这个函数进行了两次调用计算.
sowe have computed two. calls of this function > TIMELINE OUTLINE
Q_block,
K_block_ptr, 如果我们在处理因果注意力机制的情况下
_block_ptr
8
in :case we are working with the causal attention.
BLOCK _ SIZE _ KV > TIMELINE OUTLINE offs _kv,
Q _block,
K _block _ptr,
V _block _ptr, 在计算因果注意力时
BLOCK _ SIZE _ Q,
SIZEKV In case we are computing causal attention,
> TIMELINE OUTLINE offs _kv,
BLOCK _ SIZE _ O,
BLOCK _ SIZE _ KV, 我们调用这个函数一次
we call this function once to work with all the query blocks that are to the left side 229 > TIMELINE OUTLINE
_block _ptr,
225 以处理位于查询键矩阵对角线左侧的所有查询块.
236
offs_kv
> TIMELINE OUTLINE 33
_block _ptr,
lock _in de
sof tm BLOCK_ ST 接着, 我们再次调用这个函数
Then we do another call of this function to work only with those blocks of keys
229
> TIMELINE OUTLINE
SEO_ LEN, 专门处理那些恰好位于查询键矩阵对角线上的键块.
237
3
that exactly lie on the diagonal of the guer y key matrix.
> TIMELINE OUTLINE 238
239
estatic method
SEQ _ LEN, 因为在这种情况下
> OUTLINE class Triton Attention (to rc because in this case,
> TIMELINE 239 @static method
SEQ _ LEN, 有些数值需要被掩码处理, 而有些则不需要.
some of the values need to be masked out and some of them do not need to be masked out.
OUTLINE TIMELINE @static method
soft max _scale,
BLOCK _ SIZE _ O,
BLOCK _ SIZE _ KV,
offs_q,
offs_kv,
2. 此外
SEQ_ LEN,
Moreover,
> TIMELINE OUTLINE
soft max _scale,
BLOCK _ SIZE _ Q,
BLOCK_ SIZE_ KV,
offs_q,
2, 这样做可以避免
offs_kv,
by doing this we can avoid computing the dot products for all those values > TIMELINE OUTLINE
soft max _scale,
BLOCK _ SIZE _ O, 在因果情况下为那些键索引高于查询索引的值
SE_ LEN,
in the causal case for > TIMELINE OUTLINE
soft max _scale,
BLOCK _ SIZE _ O,
BLOCK_ SIZE_ KV,
offs_q, 计算点积
offs_kv,
which the key is index of the key is higher than the index of the query,
> TIMELINE OUTLINE
soft max _scale,
BLOCK _ SIZE _ O, 从而节省一些计算量 因为无论如何, 在soft max 之后, 这些值都会变为零
saving some computation because any way they will be resulting, after the soft max,
> TIMELINE OUTLINE
soft max _scale,
BLOCK _ SIZE _ Q,
BLOCK _ SIZE _ KV,
offs_q,
2, 不会对输出产生贡献.
offs_kv,
inzerosand they will not contribute to the output > TIMELINE OUTLINE
soft max _scale BLOCK _ SIZE _ O,
BLOCK_ SIZ
offs_q, 因此, 这样的操作应该会更快.
offs_kv,
SEQ_ LEN,
So it should be faster.
> TIMELINE OUTLINE
嗯好的现在我们回到这个调用方法上来
uh,
okay, now let's go back to. the this method here, so calling method,
static meth c > TIMELINE OUTLINE HEAD _ DIM_ V= V. shape[-1]
234 还有最后一件事需要完成, 那就是计算log Sum Exp.
and there is one last thing. that we need to do, which is we need to compute the log Sum Exp,
OUTLINE TIMELINE HEAD _ DIM_ V= V. shape[-1]
接下来, 我将向大家展示这是如何操作的
static method > TIMELINE OUTLINE 242
HEAD_ DIM_0, HEAD_ DIM_ Ke Q. shape I-1l; K. shape[-1]
_i+=tl. math. log ( 为了在反向传播过程中能够重新计算soft max,
So in order for the backward pas s torecompute the soft max > TIMELINE OUTLINE
_i+= tl. math. log ( 而无需再次计算每行的归一化因子
without. having to recalculate the normalization factor > TIMELINE OUTLINE 242
35 和最大值, 我们实际上需要保存两个不同的内容.
and the maximum value for each row, we should be actually saving two different stuff.
> TIMELINE OUTLINE
m_i+= tl. math. log (
# This is nee 个是查询块中每行的最大值
One is the maximum for each row in the query block > TIMELINE OUTLINE 242
另一个是查询块中每个查询的归一化因子.
and one is. the normalization factor for each query in the query block.
> TIMELINE OUTLINE
_i+= tl. math. log (
This is needed to c 然而, 这里有个小技巧.
However, there is a trick.
> TIMELINE OUTLINE 242 class Triton Attention (torch
这个技巧, 其实并不叫做log Sum Exp技巧
38
And the trick is, okay, it's not really called log Sum Exp Trick > TIMELINE OUTLINE 4
_i+=tl. math. log( 因为1og Sum Exp技巧通常用于其他目的.
because the log Sum Exp Trick is used for another purpose.
> TIMELINE OUTLINE 242
不过,
"我们暂且称之为"1og Sum Exp技巧第二版"吧
39
But let'scall it log Sum Exp Trick number two.
OUTLINE 241
40
class Triton > TIMELINE 242
那么, 这个"1og Sum E×p技巧第二版"大致是这样的
Sothelog Sum Exp Trick number two is something like this.
> TIMELINE OUTLINE 42
m_i+= tl. math. log (
# This is needed to con
So let me open the.
> TIMELINE OUTLINE 242 class Triton Attention (torch. au
幻灯片.
slides.
因此, 当我们进行查询与转置后的键相乘时,
so when we do um - query multiplier, transpose of the keys -
会得到一个由点积构成的矩阵.
we get a matrix that is made up of dot products.
因此, 像这样, 像这样, 就是一个点积, 我们可以称之为查询一
so something like this, like this, is one dot product, so let's call it query one -
转置键一.
transpose the key one.
查询一:转置键二.
query one : transpose the key two.
这是查询二-转置键二.
and this is query two - transpose the key two.
接下来我们需要应用soft max 函数, 对吧?
Then we need to apply the soft max right?
那么soft max 是什么呢?
So the soft max is what?
让我们来写一下soft max 的公式.
Let's write the formula of the soft max.
对于每一个向量, 即这是一个向量, 这也是一个向量
For each of these vectors, so this is a vector and this is a vector,
因为我们按行应用了它.
because we applied it by rows.
对于每一个向量, 它将按如下方式逐个元素地修改每个元素.
For each of these vectors, it will modify element-wise each element as follows.
所以, xi 的soft max等于xi的指数减去 一一哎呀,
So the soft max of xi is equal to the exponential of xi minus, oh my God,
我没留够空间, 让我们把这些内容往后挪一下.
I didn't leave enough space, so let's move this stuff here back.
然后把这些内容稍微往左移一点.
and this stuff here, little left.
好的,
All right,
soft max 的计算方式是:对向量中的每个元素取指数,
it will be the soft max of the exponential of each element
然后减去当前向量中元素的最大值
minus the maximum for the current vector to
(xmax ), 再除以归一化因子.
which we are applying the soft max divided by the normalization factor,
这个归一化因子是所有可能的)(在这个例子中n等于2,
which is the summation over all possible j's, where n, in this case,
因为每个向量由两个元素组成)
is equal to two because we have each vector is made up of two elements
的指数(xi-xmax )的总和.
of the exponential of xi minus xmax.
现在, 假设我们已经有了×max, 并且也已经计算出了这个总和.
Now, imagine we already have x max, and we already have this summation.
在 Flash Attention 算法的前向传播过程中, 这里的这部分被称为
In the flash attention algorithm in the forward pass, this stuff here is called li,
而这里的这部分则称为mi.
and this stuff here is called mi.
在代码中, 我们实际保存的
What we are going to save in the code you can see here,
m_i+=tl. math. log (
This is neede 在代码中, 我们实际保存的
What we are going. to save in the code you can see here,
> TIMELINE OUTLINE 242
_i+= tl. math. log (
This is need e 并不是mi和li分开的值
we are saving actually not mi and li separately > TIMELINE OUTLINE 242
class Trito
This is needed to 而是mi加上li的对数.
class Triton A we will be saving mi plus the logarithm of li.
> TIMELINE OUTLINE 242
因此, 我们将保存m i加上 Ii的对数值.
So we are going to save mi plus the log of li.
这样一来, 在计算反向传播时,
so what will happen is that when we will um compute the um, compute the backward pass,
我们需要动态地重建这个矩阵,
we need to recreate this matrix here on the fly,
这意味着我 们需要重新计算查询向量(query ), 并将其与键向量(keys )的转置相乘
which means that we need to recompute the query, multiply by the transpose of the keys,
然后再应用soft max 操作.
and we to um, and then we should apply the soft max.
要应用soft max, 我们需要这里的这部分和那部分,
to apply the soft max, we should need this stuff and this stuff here,
但我们手头只有这部分数据.
but we have only this stuff here.
那么, 这里的这个就是m吗?
So this is the m?
i加上一的对数?
i plus the logarithm of I?
因此, 在计算soft max 时, 我们会按照以下步骤进行:
so when we're computing the soft max, we will compute the following :
我们将定义一个新的soft max,
so we will compute the soft max as follows : we will define - let's call it a new soft max,
这里我用另一种颜色来表示.
so let me use another color here.
我们将按照以下方式应用soft max:设xi的sof tmax 为soft max we will apply the soft max as follows : so soft max of xi, let's call it soft max two,
因为我不想混淆, soft max because it's-I don't want to confuse-soft max -
二等于每个元素减去.. 后的指数值.
is equal to the exponential of each element minus.
我们将减去这里的这个值,
we will subtract this value here,
它对应于我们当前应用soft max 的那一行.
the one corresponding to the current row to which we are applying the soft max.
因此, 它将是xi 减去mi再减去li的对数的指数.
So it will be the exponential of xi minus mi minus the log of li.
如果我们展开这个表达式, 它将变成指数形式,
If we expand this expression, this will become the exponential of,
因为两个数之和的指数
because the exponential, the sum of two, the exponential of the sum,
等于这两个数指数的乘积.
is equal to the product of the two exponentials.
我们也可以这样写:
we can also write it like this :
它将是xi减去mi的指数, 除以 Ii对数的指数,
so it will be the exponential of xi minus mi divided by the exponential,
猜猜看结果是什么?
the exponential of the log of I i, which guess what?
它等于xi 减去mi的指数除以li,
it is equal to the exponential of xi minus mi divided by li,
这正是归一化因子.
which is exactly the normalization factor.
此外, 我们还有mi.
and we also have mi.
因此, 我们不再保存两个值, 而是只保存一个值. 当我们应用它时
so instead of saving two values, we save only one value and when we apply it,
指数的性质会自动确保
the exponentials properties will take care of actually also normalizing each value
每个应用到的值都被正确归一化.
to which we apply it.
如果你忘记了指数的性质, 这其实很简单.
if you don't remember the properties of the exponential, it is very simple.
因此, a加b的指数等于a的指数
so the exponential of a multi plus b is equal to the exponential of a
乘以b 的指数,
multiplied by the exponential of b and the exponential of a -
而a减b的指数
not exponential, it's the exponential a min us b-
等于a的指数除以b的指数,
is equal to the exponential of a divided by the exponential of b,
这就是我们使用的技巧.
And this is the trick that we're using.
这就是为什么我们不需要保存两个不同的值.
So that's why we don't need to save two different values.
我们只需要保存一个值.
We just need to save one value.
然后当我们应用它时,
And then when we apply it, it will automatically be taken care,
由于指数的性质, 它会自动处理归一化.
will take care of normalizing because of the properties of the exponential.
# This is nee 好的,"我们继续往下讲
All right, let'smove forward.
> TIMELINE OUTLINE 242 class Triton Attention (torch.
_i+=tl. math. log ( 我们还创建了这个值, 它将在反向传播过程中使用
So we have also created this value that we will use during the backward pass.
> TIMELINE OUTLINE
m_i += tl. math. log (
This is neede to 现在, 正如您所记得的
now, as you remember,
> TIMELINE OUTLINE 242 class Triton Attention (to to grad. Function ):
235
_i+=tl. math. log( 在 Flash Attention 算法中, 我们在计算每个块时并不进行归一化处理
in the flash attention algorithm we don't normalize each block while computing it.
> TIMELINE OUTLINE lass Triton Attention (torch
_i+=tl. math. log ( 我们会在最后对输出进行归一化, 而这正是我们接下来要做的
we normalize the output at the end, and this is exactly what we are going to do here.
> TIMELINE OUTLINE
L i+=tl. math. log ( 因此, 在计算出当前输出块 所属所有行所需的归一化因子后, 我们才在最后对该块进行归一化处理
so we normalize the block at the end, after we have computed all the normalization factors > TIMELINE OUTLINE
m_i += tl. math. log(
_1 并保存这个mi值.
o_block= O_block/ Li:, None ]
that we need for all the rows that belong to the current output block, We save this mi.
> TIMELINE OUTLINE
m_i+= tl. math. log(
This is 所以我们保存它.
o_block =0_block/i[:, None ]
so we save it.
> TIMELINE OUTLINE 242 class Triton Attention (torch. auto grad. Function ):
这个m1值就是每行的归一化因子和最大值
238
this miiswhat is the normalization factor and the maximum for each row > TIMELINE OUTLINE
_1+=tl. math. log(
# This is needed 0_block =0_block 我们在反向传播时会用到它.
that we will need for the backward pass.
> TIMELINE OUTLINE class Triton Attenti
_1+= tl. math. log(
235 因此, 我们需要将其保存在一个张量中, 以便在反向传播时使用
sowe need to save it in a tensor that we will use during the backward pass.
> TIMELINE OUTLINE
_1+=tl. math. log( 那么我们需要弄清楚这个张量具体是哪一个?
sowe need to understand which tensor is this?
> TIMELINE OUTLINE class Triton
146
shape=| SE(_ LEN, HEAD_ DIM) 这个张量就是我们称为m的张量, 它具有批量大小
and it's the tensor. that we called m, which is a tensor of a batch size.
> TIMELINE OUTLINE
# This indicates which head and batch to process. Each program index _batch _head =tl. program _id(1)
This indicate which 头数和序列长度这三个维度.
This indicate index _batch =index index _he a
num heads and sequence length dimensions > TIMELINE OUTLINE qv k _offset =(
# This a
# This indicates which head and batch to process index _batch _head =tl. program _id(1)
126 因此我们需要在这个张量中选择正确的位置
so we need to select the right point in this tensor,
> TIMELINE OUTLINE 131
132
qvk_offset =(
This indicate 以确定该将mi 值保存到哪里
index _batch =i This to select to where we should save this mi values.
> TIMELINE OUTLINE qv k _offset =(
因此, 我们需要选择正确的批量大小索引和头数索引.
so we need. to select'the right batch size index and the right number of head index.
> TIMELINE OUTLINE qv k _offset =
strides (stride _0_seq, stride_ O_dim),
shape( SEO_ LEN, HEAD _ DIM), 们将这个指针按以下偏移量向前推进, 目 即 m 加上 index _batch _head 的值
so we advance this pointer by. the following offset, which is m plus the index batch head > TIMELINE OUTLINE
as Sert HEAD _ DIM _ Q = HEAD _ DI M_ K and HEAD _ DIM_ K == HEAD_ DIM_ V
stage=3 这是因为index _batch _head 一-
rid =
because each index-okay, the-index batch head -
> TIMELINE OUTLINE
=_1+=tl. math. log(
o_block =0_block/l_il:,
# This is needed to comp 它表示当前程序的索引
_ptr s= M+index _batch _head is what is the index of the current program > TIMELINE OUTLINE 242 class Triton Attention (torch. auto grad. Function ):
=_1+= tl. math. log(
_i 包含了我们正在处理的头数
237
o_block =0_block/_1 that includes information about which head we are working with 3 > TIMELINE OUTLINE Triton Attention (torch. auto grad. Function ):
_1+=tl. math. log(
o_block =0_block/l_il:, None ]
compute the logsumex 和批次信息
238
and which batch we are working with, because each of this um,
> TIMELINE OUTLINE ass Triton Attention (torch. auto grad. Function ):
BLOCK _ SIZE _ KV:tl. const expr,
STAGE :tl. const expr, 对于每一个批次和每一个头数, 我们都对应着一个序列长度
for each batch and for each head, we have a sequence length.
> TIMELINE OUTLINE = tl. program _id(1)
115
BLOCK_ SIZE_ KV:tl. const expr,
STAGE :tl. const expr, 我们可以根据当前的索引值, 跳过相应数量的序列长度.
we can skip a number of sequence length based on which index is okay.
> TIMELINE OUTL INE
d= tl. program _id(1)
BLOCK _ SIZE _ KV:tl. const expr,
STAGE :tl. const exp r,
tl. stat 实际上,"我们正在进行的是跳过操作.
olock _
what we are doing is basically we are skipping.
> TIMELINE OUTLINE This n dex_batch_head = tl. program_id(1)
116
STAGE:tl. constexpr, 对于每一个批次和每一个头数, 我们都有一个序列长度
for each batch and'for each head we will have a sequence length > TIMELINE OUTLINE = tl. program _id(1)
STAGE :tl. constexpr, 因为序列中的每个token都有一个最大值
126
2
because each token in the sequence has a maximum value > TIMELINE OUTLINE 2
=tl. program_id(1)
STAGE :tl. constexpr, 同时每个token也会有一个归一化值
8
and each token in the sequence will have a normalization value.
> TIMELINE OUTLINE tl. program _id(1)
STAGE :tl. const exp r,
tl. static _ 因此, 根据当前的批次和头数组合
So, based on the current combination of batch and head > TIMELINE OUTLINE 123
2
tl. program_id(1)
1
STAGE:tl. constexpr, 我们可以跳过其他程序将要处理的一部分序列长度.
we can skip a number of sequence length that other programs will process.
> TIMELINE OUTLINE =tl. program _id(1)
116
BLOCK_ SIZE_ KV:tl. constexpr,
117
STAGE:tl. constexpr,
1 由于在这个张量中, 序列长度是最后一个维度
So, because in this tensor we have the sequence length as the last dimension > TIMELINE OUTLINE =tl. program _id(1)
tl. static _assert ( BLOCK _ SIZE _ KV <= HEAD_ DIH)
120 而批次大小和头数大小的组合索引也包含在内
and we have what is the combined index of the batch size and number of head size,
> TIMELINE OUTLINE
dex _batch = index _batch _head // NUM _ HEADS
tl. static _assert ( BLOCK _ SIZE _ KV <= HEAD _ DIH )
block _index _q= tl. program_id(0)
# This indicate which block in the 我们可以根据
123 # This indicates which head and we can skip a number of sequence of length based on the combined index L. progra r > TIMELINE OUTLINE ndex _batch =index _batch _head // NUM _ HEADS
127 index _batch =index _batch _head // NUM _ HEADS 程序索引号为
1(即此处提到的index batch_head)提供的组合索引l
132
qvk_offset which is given by the program index number one,
OUTLINE 13
133
> TIMELINE Live Sh
# This indicate the position index _batch =index _batch _head // NUM _ HEADS index _head =index batc 跳过相应数量的序列长度.
# This allows to get which is the index batch head that we have here.
> TIMELINE OUTLINE
lead the blocks of Q:lt wit T stpy
SHAM thraugnoup
o_block =tl. loadf Q_block_ptr) 这便是我们在此处进行跳过操作的原因
o_btock,
And this is why we skip here.
> TIMELINE OUTLINE
ceil( SEQ 序列长度的数值乘以索引中的批次头数.
261
Asequence length number multiplied by the index batch head.
> TIMELINE OUTLINE ( BATCH _ SIZE, NUM _ HEADS, SEQ _ LEN ), device = Q. device, dtype=torch. float32
cei L ( SEQ _ LEN 这个 M 指向整个张量的第一个元素
This Mis pointing to the first element of the entire tensor.
> TIMELINE OUTLINE 265
264
( BATCH_ SIZE, NUM_ HEADS, SEO _ LEN ), device = Q. device, dtype=torch. float32
257
grid= labda a 因此我们根据这个特定程序
cei L( SEQ 正在使用的组合索引(即index _batch _head )来跳过相应的头数和批次
so we are skipping the heads and the batch based on the combined index,
> TIMELINE OUTLINE ( BATCH _ SIZE, NUM _ HEADS, SEO _ LEN ), device = Q. device, dtype=torch. float32
cei L( SEQ_ LEN
BATCH_ SIZ 接着我们会遇到一个偏斜问题
6
index batch head that this particular program is working with, and then we have of skew.
> TIMELINE OUTLINE ( BATCH _ SIZE, NUM _ HEADS, SEO _ LEN ), device = Q. device, dtype=torch. float32
256
cei L( SEQ_ LEN
257 产生偏斜的原因是,:每个这样的核心 即注意力机制的前向传播方法
of skew is because each of these kernels, the attention forward method,
> TIMELINE OUTLINE ( BATCH _ SIZE, NUM _ HEADS, SEQ _ LEN ), device = Q. device, dtype=torch. float32
class Triton Attention (torch. auto grad. Function ):
@static method def forward (c tx, Q, K, v, caus 都会处理一个查询块.
HEAD _ DIM V= V. shape[-1]
HEAD DIM_ O, HEAD DIMK
BATc Hsize, mm e Aoswill work with one query block.
> TIMELINE OUTLINE assert HEAD _ DIM _ HE AD_ DIM_ K and HEAD_ DIH_ Kmm HEAD_ DIM_ V
249
BATCH_ SIZE, NUM_ HEADS, SEO_ LEN, HEAD _ OIM 每个查询块都包含一些索引, 这些索引精确地指向它所包含的查询
Each query block has some indices for the exact queries it includes.
> TIMELINE OUTLINE ceil ( SEQ _ LEN / BLo CK _ SIZE _ Q )= How many blocks of Q we have
class Triton Attention (torch. auto grad. Function ): 这由你在这里看到的off _skew 变量给出
And this is given by off-skew variable that you can see here,
> TIMELINE OUTLINE 251
a SSert HEAD DIM_ Q==
HEAD_ DIM_ Kand HEAD_ DIM_ K== HEAD_ DIM_ V
Q _block,
K _block _ptr, 它表示我们需要跳过多少个查询块
which is how many blocks of queries we need to skip > TIMELINE OUTLINE offs _kv,
Q _block,
K _block _ptr,
_block _ptr,
lock soft ma 因为这些块将由其他程序处理
because they will be processed by other program s,
OCK_ SIZE KV > TIMELINE OUTLINE offs _kv,
Q _block,
_block _ptr, 再加上一 个特定查询块所拥有的查询范围
207
LOCK_ SIZE_ KV
plus the range of queries that a particular block of queries has.
> TIMELINE OUTLINE offs _kv,
Q_block,
K_block_ptr, 假设这个特定程序正在处理从12到16的查询
Imagine this particular program is working with the queries that go from, I don't know,
> TIMELINE OUTLINE offs _kv,
Q_block,
K_block_ptr, 那么这些查询就是12、13、14、15.
_block_ptr
from 12 to 16, then this will be 12, 13, 14, 15
CK_ SIZEKV
> TIMELINE OUTLINE offs _kv,
Q _block,
K _block _ptr,
_block _ptr 因此, 归一化因子和每行的最大值
So the normalization factor and the maximum value for each row,
> TIMELINE OUTLINE offs _kv,
Q_block,
K_block_ptr, 我们只针对这些查询索引(即12、13、14和15)进行计算.
we only have that for these indices of query, So 12, 13, 14, and 15.
LOCK_ SIZ
> TIMELINE OUTLINE 211
offs_kv,
Q _block,
_block _ptr, 这就是为什么我们还需要跳过这个特定程序
_block_ptr
208
209
And that's why we need to also skip the number of queries > TIMELINE OUTLINE 211
offs_kv,
attn_fwd[grid](
V= V, 处理的查询数量
=0
stric
that this particular program works with,
> TIMELINE OUTLINE
stride_ Q_
. stride(2)
. stride(3)
ard pass, one for each query 这个数量已经包含在这个偏移量(即offset _query 变量)中了.
which-is-already included in this offset, offset queue variable.
> TIMELINE OUTLINE V= V,
好的, 现在我们可以存储 MI了
att_lgriu All right, so now we can store the Ml,
> TIMELINE OUTLINE
因为我们有了指向它应该保存位置的指针.
because we have the pointer to where it should be saved > TIMELINE OUTLINE 270
Q= Q
= K
# Z in the CUDA launch grid 我们还可以存储由内部for 循环计算得到的输出.
And we can also store the output, which was computed by our inner for loop.
> TIMELINE OUTLINE attn _fwd [grid ](
Q= Q,
263
# Z in the CUDA launch grid 各位, 这就是注意力机制的前向传播步骤, 民 即 flash attention.
And this,
guys, is the forward step of the attention, flash attention.
> TIMELINE OUTLINE Q= Q,
wd [grid ]
# Z in the CUDA launch grid
269 现在我们应该继续向前,? 也就是计算反向传播
now we should go forward, which is, we should compute the backward pass > TIMELINE OUTLINE = Q,
in the Cu DA launch grid
266 我们已经具备了计算反向传播的所有要素
wealso have all the ingredients for computing the backward pass,
> TIMELINE OUTLINE attn _fwd [grid ](
Q= Q,
262
1,# Z in the Cu DA launch grid 因为我们已经见识过这个技巧一一 即 log sum x 技巧
because we have already seen this trick, which is the logsumxtrick
float3z
> TIMELINE OUTLINE attn _fwd [grid ](
= Q,
1,# Z in the Cu DA launch grid
# Mis the Lo 所以我们已经知道如何使用它:
( BATCH _ SIZE, N so we already know what, how to use it :
NUM _ H > TIMELINE OUTLINE 271
attn_fwd[grid](
Q= Q,
在反向传播过程中动态计算查询键块
to compute the query key block during the backward pass on the fly
6
> TIMELINE OUTLINE attn _fwd [grid ]
= Q,
为了理解反向传播, 我们还缺什么?
'What we miss to understand the backward pass?
SIZE, NUM _ HE AI > TIMELINE OUTLINE 271
attn_fwd[grid](
Q= Q,
in the Cu DA Launch grid 首先,"我们需要理解什么是反向传播?
Well, we need to understand what is the, first of all, what is the backward pass?
267
> TIMELINE OUTLINE attn _fwd [grid ]
1,# Z in the Cu DA launch grid
# M is the Logs 我们为什么需要反向传播呢?
M =torch. empty BATCH _ SIZE,
JUM _ HE Why do we even need a backward pass?
> TIMELINE OUTLINE 271
attn_fwd[grid](
Q= Q,
1, # Z in the CUDA Launch grid 我们需要理解 Py Torch 中的自动求导机制是什么?
We heed to understand what is the auto grad of Py Torch?
> TIMELINE OUTLINE 27e
271
attn_fwd[grid](
Q= Q,
1, # Z in the CUDA launch grid
# M is the log sum ex for the backward pass M = torch. empty ( 如何计算?
( BATCH _ SIZE, NUM _ HEADS, SEQ _ LEN), d
dev ic
ce= Q. device, dtype
. float3z
how to compute?
> TIMELINE OUTLINE 271
_attn_fwd[grid](
Q= Q,
1,# Z in the Cu DA launch grid 在反向传播过程中计算梯度时, 雅可比矩阵是什么?
what is the jacobian when computing the gradient on the backward pass?
> TIMELINE OUTLINE attn _fwd [grid ](
= Q,
1,# Z in the CUDA launch grid
# M is the Log sum e
M=torch. empty ( 我们真的需要计算这个吗?
( BATCH _ SIZE, NUM _
do we even need to compute that?
> TIMELINE OUTLINE attn_fwd[grid](
271
Q= Q,
6 因此, 我们需要手动推导反向传播的所有公式.
sowe need to derive all the formulas of the backward pass by hand > TIMELINE OUTLINE Q= Q
263
1,# Z in the Cu DA launch grid 所以, 如果你准备好迎接挑战, 我们就继续吧. 女 好的
so if you are in for the challenge, let'scontinue All right.
> TIMELINE OUTLINE 270
fwd[grid]
Q= Q,
所以, 如果你准备好迎接挑战, 我们就继续吧. 好的.
so if you are in for the challenge, let's g continue All right.
->o
那么现在, 在深入探讨 Flash Attention 算法的反向传播之前,
so now, before looking at the flash, attentions backward pass at the algorithm,
¥(× 长+
->o
我们需要先理解为什么我们需要反向传播.
we need to understand why we even need a backward pass. 长+
因此, 在探讨 Py Torch 的自动求导机制之前,
so before looking at the a auto grad'of Py Torch,
->o
我们应当先理解什么是导数、梯度以及雅可比矩阵.
这样, 当我们讨论导数、梯度和雅可比矩阵时, 就不会感到迷茫.
->o
因此, 我将快速回顾一下这些主题的内容.
So I will do a very fast, let's say, rehearsal of what these topics are.
->o
那么, 什么是导数呢?
((x) - c 当你有一个函数, 它以实数值作为输入并输出实数值时,
when you have a function that takes as input a real value and outputs a real value,
我们讨论的就是导数, 其定义如下.
we talk about, deriv ati yes, which is defined as follows.
->o
√函数关于其变量×的导数被定义为
The derivative of the function with respect to its variable,
X
->o
当步长h 超近于零时,
函数在x加h处的值减去函数在x处的值, 再除以h的极限.
that goes to zero of the function evaluated at x plus h.
即×加上步长h处的函数值减去×处的函数值, 再除以步长h.
so x plus the step size, minus f, evaluated at x divided by the step size.
->o
直观上讲
So intuitively we are saying i
is the ratio of how much
((x) - c 导数的意义是函数输出值随输入值微小变化的变化率.
the output change for a smal change, for, how much the input has changed in the function.
☆->o
这也让你直观地理解了为什么导数能告诉你
This also gives yoy the intuition of why the derivative tells you ¥(×
)人函数在某一点处切线的斜率.
the inclination of the tangent, line to the function at the point in which it's evaluate d.
->o
我还将使用以下符号来表示导数.
I will use also the following notation to denote the derivative.
y cox icma tel
EW 因此, 导数.
<—>x+sothe deriva
ytoxicmatel
我习惯这样写:f(x)
I am used to write it as like this'so f- prime of xy -
LD 但也可以写成df(x)/dx, 或者dy/dx
but it's also possible to write-it as d of f, of x'with respect to dx, or d of y,
(其中y 是函数的输出)
where y is the output Of the function : with r respect to x,
M
它们都表示相同的东西人也就是上面定义的导数.
and they are all equal to the'same thing, which is the definition above.
×i chon ped
如果我们把这个公式倒过来,
M 也可以写成以下形式
mroxic natel
So if we want to evaluate the function at the position x plus h,
cyto x ioma tel
也可以将其表示为f(x)乘以h
cytoxiomatel
即函数在x点的导数乘以步长h
so the derivative of the function in the point x multiplied by h,
再加上
which is the step size plus f ofx.
Jx
M 这实际上也是我们推导求解微分方程的欧拉法则的方法,
Tibis
不过这不是今天的主题."
cyto xi matel
所以这个h, 我们也可以称之为 Ax. x
LD
因此,
f(x+ L
△ X)大致等于这个表达式 P?
X Sof of x plus delta x is more or less,
因为这里有一个极限, 表示只有当h 非常非常小时, 这个近似才成立
这就是为什么我们在这里用"约等于"来表示.
so that's why we put this more or less approximately.
sof of x plus delta x is more or less equal to f,
cyycoximatel
再加上f(x)
cyp toxic matel
"你也可以这样理解这个公式:如果
this you can also read it as follows : that if, by inverting this formula,
M
发生了一个微小变化, 这个微小变化就是△x,
If'x changes by a little amount, and this little amount is delta x,
那么y 会变化多少呢?
how much y will change?
的变化量就是这个精确的数值, 也就是y对×的导数, 即dy/dx,
Y will change by this exact amount, which is the derivative of y with respect to x,
乘以×的变化量.
so dy with respect to dx, multiplied by how much x has changed.
因此dy/dx告诉我们, 当×发生微小变化时, y会变化多少,
So this dy dx tells us how much y will change with a small change of x,
if we multiply with the actual change of x.
it will tell us why, how exactly y will be affected.
嗯, 我不想过多停留在这个话题上,
um, i don't want to use stay too much on this,
但我想用这种直观理解来引入链式法则,
but i would like to use this intuition to introduce the chain rule,
M 因为想象一下, 如果我们有一个复合函数.
because imagine we have a function of a function.
2
2+2 假设我们有z等于f(g(x).
so imagine we have z is equal to f, of g, of x.
2
2+2 一我们可以认为×通过函数. g·映射到变量y,
We can think of x being mapped into a variable y through the function g
2+2 然后y通过函数
f映射到变量z.
and the n y being mapped into a variable z through the function f.
2
×发生微小的变化, 这里说的微小变化指的是△x, 那么y会变化多少呢?
If x changes by a little bit, and by a little bit I mean delta x, how much y will change?
2 嗯, y会变化 Ay.
Well, y will change by delta y.
2
Ay是什么呢?
What is delta y?
2+2 一△y就是y对×的导数乘以x的步长.
Delta y is the derivative of y with respect to x multiplied by the step size of x.
2
如果y 发生变化, 它也会影响到z因为y. 和z之间存在直接的映射关系.
but if y changes it will also affect z, because there is a direct mapping between y and z.
2
2
2+ 一那么, y发 发生微小变化时, z会变化多少呢?
so how much z will change for a small change in y?
2+2
IV 让我们来看看.
let's see.
2+2 所以, 如果y从原来的值变化了一个微小的 Ay,
so if y changes from the old y by a small step delta y,
2
+2 那么z也会相应地变化△z,
then z will also change by some delta z,
2+2 一而这个△z就是dz对dy. 的导数乘以 Ay.
and this delta z is the dz of on dy multiplied by delta y.
2
2+2 如果我们用上面计算得到的 Ay来替换这里的 Ay,
if we replace this delta by with the delta y
2 就能推导出链式法则.
that we have computed in the expression above, we arrive to the chain rule.
2 这样就能告诉我们z会受到怎样的影响.
It will tell us how z will be affected.
J这就是△z, 也就是×发生微小变化时对z 的影响.
So this is delta z, what is the effect on z for a small change on x.
y 它由两个导数的乘积组成,
And it's the product of the two derivatives,
一个是y对x的导数, 另一个是z对y的导数.
one for y with respect to s and one z with respect to y.
M 这就是我们在高中学习的链式法则.
And this is the chain rule that we study in high school.
M
y 确实如此.
So it is.
如果你想计算dz对dx的导数, 那就是dz对dy的导数乘以dy对dx的导数
if you want to compute dz on dx, it is dz on dy multiplied by dy, dx,
这非常直观.
which is very intuitive.
举个例子,
if you think about the following example,
你可以把z 看作汽车的价格,×看作石油的价格
so you can think of z as the price of cars and x as the price of the oil -
石油价格的微小变化会对汽车价格产生多大影响呢?
How much will a small change in the price of oil affect the price of a car?
石油价格的微小变化会影响另一个变量y,
Well, the small change in the price of the oil will affect, for example, a variable y,
比如电力的价格.
y which could be the price of electricity.
So how much of the price of electricity will affect the price of a car?
It's through the derivative of the price of the car with respect to the electricity.
要计算石油价格对汽车价格的影响
To get the effect of the price of oil on the price of the car,
我们只需将这两个效应相乘即可.
we just multiply the two effects.
2 这正是链式法则背后的直观理解.
And this is the intuition behind the chain rule.
([x )
Anyway, let's talk about gradients.
当我们有宁个以向量为输太并输出标量的函数时,
So, when we have a function that as input takes a vector and produces a scalar,
M
. N
我们不再谈论导数, 而是讨论梯度.
we talk not any more about derivatives, we talk about gradients.
+
M
想象一下, 我们有一个函数, 它接收一个由两个维度
so imagine we have a function that takes as input a vector made up of two dimensions,
+
(通常为n 维)组成的向量作为输入, 并输出一个标量.
but n dimension in general, and it produces a scalar.
+
M
我们 在什么时候需要处理这种类型的函数呢? 例如, 损失函数就是典型的例子
when do we have to deal with this kind of function, for example loss functions?
?
M
M
损失函数总是输出一个标量值
loss functions are something that are always a scalar as output M +
而它们的输入则是张量
+
M
举个例子, 想象一下交叉熵损失函数.
+
它会接收一系列标记, 每个标记都有自己的1ogits,
It will take a'seguence of tok kens, each token with its own log its,
+
然后计算出一个单一的数字, 即损失值.
and it will compute one single number, which is the loss.
+
M
那么, 在这种情况下, 如何观察输入对输出的影响呢?
So, how to view the effect on the output with respect to. the input in this case?
+
如果×发生微小变化,
、而这个微小变化不再是一个数字, 而是一个向量.
iv a dive
port col
因此, 如果×发生变化, 即旧×加上△x (这是一个向量加法)
那么y 也会受到影响.
受到什么影响呢?
by what?
port col
vadive
port col
y的变化将由dy对dx的导数乘以△x决定.
ywill
+
port col
然而, 这里的△x 不再是个单一的数字, 而是一个向量,
ative
因为×1可能会有微小变化, x2也会有微小变化,
port co R
x3、x4, 依此类推,
live
直到xn 都会有微小变化.
port col
因此这实际上是这两个向量的点积.
port col
为什么是点积呢?
port co R Why a dot product?
port col
因为y的变化会受到x1 变化的影响 也会受到
x2. 变化的影响
fected by the change in x2,
Nole : he chaim rule aplceo im He pome wey
同样会受到x3直到xn变化的影响
N. e : eit wiube affected by the change in x3 up to xn.
oomewoy
Nole : He chaim rule appeceo im He pome wey
而x1对y的影响程度, 是由y对x1的偏导数
乘以×1的变化量来决定的
pommewey
No le: e Chaim rule aplceo im He pome wey
以此类推, 直到xn 的最后一个影响分量
Nole : He chaim rule apneceo im He pome wey
在这种情况下, 链式法则同样适用, 其原理与标量情况下的链式法则一致.
化因此, 公式保持不变.
so the formula does not change.
这里的链式法则世同样适用.
also for the change rule here.
deriva dive 我只是想提醒你, 这里我们讨论的是梯度
梯度实际上是一个茁输出相对于输入向量中每个变量的偏导数组成的
pont co R live
当我们讨论一个以向量为输入并生成向量的函数时,
就不再使用"梯度"这一概念, 而是转而讨论"雅可比矩阵"
ec tar
yectar 因此, 如果我们的输入×一一民 即该函数的输火 发生微小变化
so if our input x, the input x of this function, chan
product and this delta x is a vector-
N)x( Nx1)=( Mx1)
那么输出y也会随之改变, 且这种变化表现为 Ay.
ec tar
△x 是一个向量.
delta x is a vector.
因此, 这个结果也必须是一个向量.
so this one has to be a vector.
所以, 这里的这个量必须是一个矩阵, 而这个矩阵被称为雅可比矩阵
it has this one here has to be a matrix, and this matrix is called the jacobian.
它是一个矩阵, 其行数(稍后我们会讨论符号表示)
it is a matrix that has as many rows - later we will talk about the notations -
等于输出变量的数量,
so it has as many rows as there are output variables
列数则等于输入变量的数量.
and as many columns as there are input variables,
第一行是第一个输出变量
the first row will be the partial derivative of the first output variable
对于所有输入变量的偏导数,
with respect to all the input variables,
第二行是第二个输出变量
the second row will be the partial derivative of the second output variable
对所有输入变量的偏导数,
with respect to all the input variables,
最后一行则是最后一个输出变量
and the last row will be the partial derivatives of the last output variable
对输入向量中所有输入变量的偏导数.
with respect to all the input variable in the input vector.
现在我们来讨论一下符号表示.
Now let's talk about the notations.
这里 所写的雅可比矩阵是按照分子布局(numerator layout )的方式表示的.
The Jacobian data I have written here is written according to the numerator layout.
这种 表示方式称为分子约定(numerator convention ), 还有一种约定叫做
This is called the numerator layout and there is another convention called the,
抱歉, 不是布局, 而是分子约定.
not layout, sorry guys, it's called the numerator convention.
还有另一种约定, 称为分母约 定 (denominator convention )或分母表示法(denominator notation )
And there is another convention, called denominator convention or notation,
在这种表示法中, 行并不是
in which the rows are not the.
行数并不等于输出变量的数量,
the number of rows is not the equivalent to the number of output variables,
而是等于输入变量的数量.
but equal to the number of input variables.
因此, 我们选择以这种方式书写雅可比矩阵是基于某种约定的.
So the fact that we choose to write the Jacobian as follows is based on a convention.
你也可以按照分母约定来书写雅可比矩阵,
you can also write the the jacobian according to the denominator convention,
只需将这里的雅可比矩阵进行转置,
just by transposing this jacobian here,
同时链式法则的公式也会相应地发生变化.
and also the formula for the chain rule changes accordingly.
目前, 我希望保持链式法则的公式与标量情况下的形式一致,
for now i want to keep the formula for the chain rule,
因此我在这里使用了这种表示法.
just like the one for the scalar case, so that's why i am using this notation here,
不过, 稍后我们只需通过转置操作就能在不同表示法之间切换.
but later we can change between one notation to the other just by doing a transposition.
好的, 现在我们回顾了导数、
Okay, now that we have reviewed what is a derivative,
梯度和雅可 比矩阵的概念, 接下来让我们讨论一下当我们对张量求导时会发生什么
what is a gradient and what is a Jacobian, let's talk about what happens
when
we take derivatives with respect to tensors of a tensor with respect to another tensor.
然是雅可比矩阵, 但它被称为广义雅可比矩阵(generalized Jacobian )
In this case, we talk about the Jacobian, but it's called the generalized Jacobian.
因此, 如果我们有一个函数, 其输入是六个维度为dx 的张量,
So if we have the function, that is, as input takes a tensor of dx dimensions,
其中第个形状emo io mol 这是张量的形状描述
where the first shape - this is kind of the shape of the tensor,
个元素是n1 二
so the first element of the shape is n1.
tew
个具有这种形状的输出张量.
dx and it produces an output tensor that has this shape.
tew
即输出的张量形状为m1, m2,.., m_dy.
So ml, m2, blah, blah, m, d, y.
dim e
DM-
in this case the formula for the chain rule doesn't change.
如果x 发生微小变化, 即变化量为deltax (这是一个张量)
and if x changes by a little amount, so by delta x, which is a tensor,
那么y会受到多大的影响呢?
y will also be affected by how much?
y的变化量dy 等于dy/dx乘以deltax, 这是一个张量乘积, 结果将是一个雅可比矩阵.
by dy on dx, multiplied by delta x, and this is a tensor product, it will be a jacobian.
这被称为广义雅可比矩阵, 其形状如下:
this is called generalized jacobian, with the following shape :
输出的所有维度乘以输入的所有维度.
so all the dimensions of the output multiplied by all the dimensions of the input.
好的, 目前这部分内容还非常抽象.
All right, this is very abstract for now.
接下来, 我们将通过一个具体案例来理解这个概念.
We will see actually a concrete case of this one,
我们将推导矩阵乘法输出的梯度,
because we will be deriving the gradient of the output of a matrix multiplication,
也就是在反向传播过程中, 计算损失
the gradient of the loss
相对于矩阵乘法操作中
when computing backward pass with respect to each of the input
每个输入的梯度.
of the matrix multiplication operation.
同时, 我们还会对
soft max 函数进行计算, 并进一步分析注意力机制(attention )的梯度.
And we will do it also for the soft max and we will do it also for the attention.
因此, 我不想一下子涉及太多主题.
So I don't want to jump to too many topics.
我只是想让我们先进入正确的思维模式.
I just wanted us to get into the right mind set.
我们知道, 在处理标量函数时, 导数就是其变化率.
So we know that derivatives, when we have scalar functions.
当输出为标量而输入为向量时, 我们讨论的是梯度.
Gradients, when the output is a scalar, input is a vector.
当输入和输出都是向量时, 我们面对的是雅可比矩阵.
Jacobian, when the input and output are both vectors.
而当输入和输出都是张量时, 我们则需使用广义雅可比矩阵的概念.
Generalized Jacobian, when the input and the output are tensors.
链式法则在任何情况下都以相同的方式适用.
The chain rule always works in the same way.
好的, 接下来我们来讨论自动求导(auto grad )
(a w, +e.
我将从标量情况开始讲解之后再扩展到张量情况.
I will do the scalar case, and then we will extend it to the tensor case.
(a w, +e.
想象一下, 我们有一个非常简单的计算图.
So-imagine we have a very simple computation graph.
为什么我们需要计算图呢?
Why we have computation graph?
因为我们讨论的是神经网络,
Because we are talking about neural networks,
而神经网络本质上就是计算图 一一它们由输入、
and neural networks are nothing more than computation graphs where we have some input,
参数以及对这些输入和参数执行的操作构成.
we have some parameters and we do some operations with these input and parameters.
假设你有一个输入a, 这个输入a乘以一个标量参数weight,
Suppose that you have an input a, and this input a is multiplied by a parameter weight,
然后生成输出y1.
it's just a scalar, and it produces an output y1.
这个y1随后与另一个数b1相加, 生成y2.
This yl is then summed up with another number called b1, and it produces y2.
小这个y2接着被平方(即输入z的平方)
This y2 is then raised to the power of two, so this is z to the power of two,
生成y3.
it's just the power of two of the input, and it produces y3.
这个y3就是我们的损失函数, 它是一个标量,
And this y3 becomes our loss function, so it's a scalar.
为了应用梯度下降法
为了应用梯度下降法,
what we want to do to apply gradient descent is
我们需要计算损失函数
we want to compute the gradient of the loss function with respect to each of the input
Jw
, 相对于这个计算图中每个输入的梯度,
of this computation graph,
也就是计算图中每个叶节点的梯度.
so each of the leaves of this computation graphs.
什么是叶节点呢?
what are the leaves?
Jw
J 就是这些节点, 即参数节点和输入节点.
it's this node here, so the parameter nodes and the input nodes.
Jw
要做到这一点, 有两种方法.
and to do that there are two ways.
一种方法是, 如果你掌握了直接关联输入与输出(即损失)的表达式
one is if you have access to the expression that relates directly the input to the output,
那么你可以直接计算梯度,
so the. to'the loss, then'you can directly compute the, the gradient,
在这里是导数, 因为涉及的是标量对向量的关系, 而非向量对向量的梯度
the derivative in this case,
M
假设在这种情况下,
= 1. 2"),'so in-this case,
你想要计算损失相对于 W1 的导数.
imagine you want to compute the'derivative of the loss with respect to w1.
设想我们拥有直接关联w1与phi
Imagine we have access. to the exact expression that relates the wl to the phi,
(即我们的损失)自 的确切表达式.
1. 2", which-is our loss. =
我们可以按照以下方式计算它.
We tan compute it as follows.
因此, 我们只需对这个关于w1的表达式求导,
So we just derive this expression with fespect to w1, which is 2 times,
由于这是函数的二次方, 所以结果是2 倍.
because this is the power to f 2 of a function.
即2乘以该函数,
: So it is2 multiplied by the function,
再乘以该函数内部
multiplied by the derivative of the content of this function
相对于我们正在求导的变量的导数.
with respect to the variable that we are deriving.
因此, 表达式将变为如下形式:还有另一种方法,
so it will become the following expression :there is another way,
那就是利用链式法则.
which. is by using the chain rule.
于是我们可以这样计算:
L. 2"),'so We can use ::
· 于是我们可以这样计算:
phi对yw1的导数等于phi对y3(即前一个节点的输出)的导数
再乘以y3 which is the previous output of the previous node,
2
对前一个节点输出的导数.
then the derivative of phi3 with respect to the previous, the output of the previous node.
接着乘以y2
so, and then multiplied by the derivative of y2
with respect to the output of the previous node,
最后再莱以y对 W1白 的导数.
and then the derivative of yl with respect to w1.
如果我们完成这一连串的乘法运算, 将会得到相同的结果.
If we do all this chain of fn ulti plication, we will obtain the same result.
你可以看到, 这里的这一部分正好等于这里的这一部分.
And you can see that here, this stuff here is exactly equal to this stuff here.
=2α=
2a(aw+.
通过这个方法? 我们会注意到一些事情.
By doing this procedure here, we will note something.
= 2αg= 2a(aw+.)
也就是说, 我想稍微放大一下视野, 好吧.
That is, I want to zoom out a little bit, okay.
To compute the, the derivative of phi with respect to w1,
我们进行了这一连串的乘法运算.
we are doing all this chain of multiplication.
但是, 这一连串乘法中的每一项、每一个因子究竟是什么呢?
But what is each item in, what is each factor in this sequence of multiplications?
其实, 这里的内容不过是phi对y2的导数.
Well, this stuff here is nothing more than the derivative of phi with respect to y2.
这些乘法运算本质上就是
These multiplications here are nothing more than the derivative of phi with respect to w,
phi对y1 的导数.
'twith respect to yl.
而所有这些结合起来, 就是phi对w1的导数.
And all of them combined are the derivative of phi with respect to wl.
中
Py Torch 会怎么做呢?
What Py Torch will do?
它会执行以下操作.
It will do the following.
Py Torch 会执行反向传播,
Py Torch will do the backward pass,
因为它知道输出相关的计算图.
because Py Torch knows what is the computation graph that relates the output.
也就是这里的损失函数,
so the loss function in this case,
20 人
=2ag2=2a(aw+e.)
以及我们想要计算梯度的变量.
and the variable for which we want to compute the gradient.
=2ag2=2a(aw+e.)
现在我们讨论的是导数:所以不是梯度,
Right now we are talking about derivatives, so it's not gradient,
但机制是完全一样的
but the mechanism is exactly the same.
=2ag2=2a(aw+e)
所以 Py Torch. 会说, 它会,
So Py Torch will say, it will,
=2ay=2a(aw+e)
=2ay=2a(aw+)
Py Torch就像一个人敲开这个操作的门, 说:"嘿,
Py Torch is like a person that knocks the door of this operation and says, Hey,
=2ag=2a(aw+e.) 平方操作.
operation power of two.
=2ag=2a(aw+e.)
=2ay=2a(aw+.) 如果我给你损失函数关于 Y3的梯度, 也就是1,
If I give you the gradient of the loss with respect to Y3, which is one,
=2ag=2a(aw+e.) 因为损失和 Y3实际上是一样的,
because loss and Y3 are actually the same,
=2ag =2a(aw+e.)
= 2ay2=2a(aw+e.) 你能给我损失函数关于 Y2的梯度吗?
can you give me the gradient of the loss with respect to Y2?
=2ag=2a(aw+e.)
=2ay=2aaw+ C.) 因为 Py Torch 实际上并没有实现一个自动求导系统,
Because Py Torch actually does not implement an auto grad system in the sense
= 2ay= 2a(aw+e.) 它并不知道导致输出的符号操作.
that it does not know the symbolic operations that led to the output.
=2ay=2a(aw+e.)
=2ay=2a(aw+ C.) 它只知道计算输出的函数是什么.
It just knows what are the functions that computed the output.
=2ag2=2a(aw+e.) 而每个函数都有一个函数.
and each function has a function.
=2ag=2a(aw+e.)
= 2ay =2a(aw+.) 每个函数都是 Python 中的一个类, 它实现了两个方法.
each function is a class in python that implements two methods.
=2ag=2a(aw+e.)
=2ay=2a(aw+.) 一个是前向传播步骤, 另一个是反向传播步骤.
one is the forward step and one is the backward step.
=2αg=2a(aw+e.)
=2ay=2a(aw+.) 前向传播步骤接收输入, 在这个例子中是y2, 并计算输出 一一y3.
the forward steps takes the input, so in this case y2, and computes the output- y3.
=2ag=2a(aw+e.)
=2ay=2a(aw+e.) 反向传播步骤将接收损失函数关于其输出的梯度,
the backward step will take the gradient of the loss with respect to its output
=2αg=2a(aw+e.)
=2ay2=2 aaw+ G.) 并需要计算损失函数关于其输入的梯度.
and needs to compute the gradient of the loss with respect to its input.
=2ag=2a(aw+e.)
=2ay=2a(aw+.) 我们该如何实现这一点呢?
how can we do that?
=2ag=2a(aw+e.)
=2ag=2a(aw+e.) 其实这很简单, 因为 Py Torch会像敲门一样自动处理这些
well, It's very simple, because a pie torch will knock the door, as -
让我在这里复制一下相关的代码和内容.
let me copy it and this stuff here.
否则, 来回切换就不那么容易了.
otherwise it's not easy to go back and forth.
好的, 我们把它放在这里.
Okay, and let's place it here.
Py Torch 会"敲"这个函数的门, 然后问:"嘿,
Py Torch will knock the door of this function here and will say, hey,
XW UPSTREAM GNADEN T
如果我把损失函数关于你输出的梯度给你,
你能给我损失函数关于你输入的梯度吗?
UP STRE AG NAD TENT
是的, 这个函数能够做到.
why?
= XW GRAD TENT
式法则的存在, 这里的这个操作符, 或者说这个函数, 完全可以做到这一点
UPSTREAM G NAD TENT
它接收损失函数关于其输出的梯度,
take the loss, the gradient of the loss function with respect to its output,
UPS TREM MG NAD TENT
乘以其输出关于输入的雅可比矩阵,(在这里就是导数)
multiply it'by the jacobian, or in this case,
结果就等于损失函数
the derivative of its ou it put with respect to its input,
关于其输入的梯度.
and it will be equal to the gradient of the loss with respect to its input.
UPSTREAM 6 NADENT
接着, Py Torch 会拿着这个结果, 志"敲"下一个操作符的门,
也就是这个求和操作, 然后问:"嘿,
which is this one, this'summation, and we'll say, hey,
REV
UP STRE AG NAD TENT
如果我把损失函数关于你输出的梯度给你,
if i give you the gradient of'the loss with respect to your output,
UP STRE AG NAD TENT
你能给我损失函数关于你输入的梯度吗?
UPSTREAM GINA DENT
Yes, this operator can do it, because this operator just needs to apply the chain rule.
XW
于是, 它会接收 Py Torch提供的损失函数关于 Y2的梯度
UP STRE AG NAD TENT
并通过与雅可比矩阵相乘来完成计算.
and by multiplying it with the, the Jacobian.
在这里, 雅可比矩阵就是其输出关于输入的导数.
in this case it's the derivative, the derivative of its output with respect to its input.
这样, 它就能计算出损失函数关于其输入的梯度.
it can compute the gradient'of the loss with respect to its input.
UPSTREAM MG NAD TENT
y =x W
接着, Py Torch 会拿着这次反向传播的输出,
The n Py Torch will take this output of this backward pass
去"敲"下一个操作符的门,
62 ADENT 也就是这个乘积操作.
and will knock the door of the next operator, which is this product.
y=x W
UPSTREMMGNADENT
同样地, 我们再次提出那个问题.
62 ADENT And we ask again the same question.
嘿, 如果我把损失函数关于你输出的梯度给你,
62 ADENT
Hey, if I give you the gradient of the loss with respect to your output,
你能给我损失函数关于你输入的梯度吗?
can you give me the gradient of the loss with respect to your input?
y =x W
y=x W
UPSTREMM6 NADENT
yes.
Yes.
y =xw
这个操作符也会完成同样的任务.
62 ADENT
S This will do the same exact job.
它会拿损失函数关于输出的梯度,
62 ADENT It will take the gradient of the loss with respect to the output
y=x W
UPSTREMMGNADENT
乘以输出关于输入的雅可比矩阵,
62 ADENT multiplied by the Jacobian of the output with respect to the input,
从而得到损失函数关于输入的梯度.
62 ADEN and obtain the gradient of the loss with respect to the output.
y=xw
UPSTREAM6 NADENT
62 ADENT 输入.
)yinput.
y =xw UPSTREAM MGN A DENT
这就是 Py Torch执行反向传播的流程.
62 ADENT
and this is how py torch runs the backward step.
UPSTREAM GNADEN T
它会沿着计算图从后往前, 依次访问每个操作符,
G2 ADENT
it runs one operator at a, time backwards in the computation graph,
UPS TREM MGN A DENT
并不断问筒一个问题:
knocking the door of each operator and asking always the same question :
"如果我把损失函数关于你输出的梯度给你,
62 ADENT
if i give you the output, the gradient of the loss with respect to your output,
UPSTREAM GNADEN T
你能给我损失函数关于你输入的梯度吗?
G2 ADENT can you give me the gradient of the loss with respect to your input?
每个操作符都会运用链式法则
62 ADEN and each operator will jy st apply the chain rule to to get this gradient,
计算出 Py Torch 所需的梯度.
to calculate this gradient that Py Torch needs.
y =xw
为什么 Py Torch不能自己完成这个任务呢?
62 ADENT
Why Py Torch can not do it by itself?
因为 Py Torch 不做符号数学运算.
62 ADEN
Because Py Torch, does not do symbolic mathematics.
它无法直接获取每个函数的具体数学表达式.
G2 ADENT It does not have access to the exact expression that each function is computing.
UPSTREAM 6 RADENT
它只是将函数当作一个黑箱, 负责前向计算和反向传播.
62 ADENT It just uses the function as a black box that computes forward and backward.
y=xw
UPSTREMMGNADENT
然而, 使用雅可比矩阵时, 我们会遇到一个问题.
However, with it he Jacobian, we have a problem.
y=xw
UPSTREMMGNADENT
让我们来看着这个问题是什么.
G2 ADENT And let is see what is the problem.
y=xw
UPSTREMMGNADENT
好的.
All right.
到目前为止, 我们一直在处理由标量组成的计算图.
s. up to now, we have been working with a computation graph that is made up of scalars.
但之前讨论的内容不仅适用于标量场景,
But the things that we have said, they work in the scalar case,
也适用于张量场景.
but also in the tensor case.
让我们回到计算图, 看看它的结构.
So let's go back, see what is our computation graph.
我们已经知道, Py Torch 会逐运算符处理, 每次都提出相同的问题
We have seen that Py Torch will go operator by operator, asking always the same question.
如果给出损失函数相对于你输出的梯度,
If I give you the gradient of the loss with respect to your output,
你能计算出损失函数相对于你输入的梯度吗?
can you compute me the gradient Of the loss with respect to your input?
每个运算符只需应用链式法则即可完成这一计算.
And each operator can'just apply the chain rule to compute that.
现在想象一下,
Imagine now that all of
所有这些运算符处理的不是标量, 而是张量.
these operators are working not with'scalars, but are working with tensors,
这意味着每个运算符的输出相对于输入的导数
which means that the derivative of the output with respect to the input of each operator
a 不再是简单的导数:
Cis not a derivative :
由于输出和输入都是张量
it wit l be a Jacobian because
这个导数将表现为雅可比矩阵/( Jacobian ), 即一种广义的雅可比矩阵
the output will be a tensor, a generalized Jacobian, and the input will be a tensor.
这也意味着, 这里的量
which me ahs also that this quantity here -
即损失函数相对于输入的导数
so the derivative'of the loss with respect to the input,
在这种情况下不再是普通的导数, 而是一个梯度. 因为输出,
in this case will not be a derivative, it will be a gradient, because the output,
即损失函数, 始终是一个标量, 而输入(在这里是y1 )则是一个张量,
the loss, is a number always,'while the input, in this case yl, will be a tensor.
因此, 输出是标量, 而输入是张量.
so number output, input is a tensor.
这时我们讨论的就是梯度了.
then'we talk about gradients.
因此, 这将是一个梯度,
so this will be a gradient, the,
我们称之为运算符需要计算的下游梯度.
and we will call it the downstream gradient that the operator needs to compute.
这将是 Py Torch 提供给每个运算符的上游梯度.
This will be the upstream gradient that Py Torch will give to each of these operators.
即损失函数相对于每个运算符输出的梯度.
So the gradient of the loss'with respect to the output of each operator.
每个运算符需要通过雅可比矩阵计算出相应的下游梯度.
And each operator needs to come up with this downstream gradient by using the Jacobian.
然而, 雅可比矩阵存在一个问题.
However,'the Jacobian has a problem.
假设我们正在实现一个简单的矩阵乘法运算.
So imagine we are implementing a simple operation that is the matrix multiplication.
而矩阵乘法的定义是
And the matrix multiplication is...
Do WNST REA M
A它接收一个 X 张量作为输入,
takes as input a X tensor,
DOWNST REAM 将其与由参数组成的 W矩阵相乘, 并生成一个 Y矩阵作为输出,
it multiplies it by a W matrix made up of parameters and produce s a Y matrix as output.
Do WN ST REAM 假设是一个 N× D的矩阵, W是一个 D× M的矩阵,
Suppose that x is, let's call it, N by D matrix, w is, let's say, D by M matrix,
DOWNSTREAM CLAE那么 Y将是一个 Nx M的矩阵.
an d so Y will be a N by M matrix.
D OWNST REAM 通常输人×是一系列向量, 每个向量都具有 D个维度.
Usually the input x is a sequence of vectors, each with D dimensions.
LG CAL JACOBI AN
DOLWN ST REAM
So you can think of it as a sequence of tokens.
DOC WN STREAM
Each token is a vector made up of D dimensions.
DO LWN STREAM
LG CAL JACOBIAN DOC WN ST REAM
通常, 我们会处理许多这样的标记
Usually we have many tokens.
DOC UN STREAM
LG CAL JACOBIAN DOC WN ST REAM
因此, 假设 N通常至少为1024, 至少在最近的语言模型中,
So suppose that N usually is at least 1024, at least in the most recent language models,
DO LWN ST REAM
we even have millions of tokens actually.
DO CUN STREAM
LG CAL JACOBIAN DOC WN ST REAM
CAL
OOLWN STREAM
3 CAL 因此这个也是
1024.
DOCWNSTREAM
LG CAL JACOBIAN DOC WN ST REAM
LGCAL 嗯, d和m
um d and m.
JACOBIAN DOC WN STREAM
LG CAL JACOBIAN DOC WN ST REAM
OCAL
m也至少是/1024, 所以实际上可以是2020,
m is als0 at least 1024, s0 we can actually become 2020, 2048, let's say so i.
DOCUNSTREAM
l G CAL JACOBIAN DOC WN ST REAM
R CAL
i like the powers of two, by the way.
DO LUN STREAM
LG CAL JACOBIAN DOC WN ST REAM
so the problem of the jacobian is this : if we compute,
DOLWN ST REAM
想通过将上游梯度与雅可比矩阵相乘
want to compute this downstream gradient by multiplying the upstream gradient
Do WNST REAM
GRADi ENt 来计算下游梯度,
with the jacobian,
DOWNST REAM
C 这个雅可比矩阵会非常大. 看看这里的维度,
this jacobian matrix is huge because look at the dimensions here,
DOWNSTREAM 这将是个矩阵, 具体来说, n乘以m的乘积,
this will be a matrix that is, So it will be well, n by m multiplied,
所以
JACOBIAN 它将是一个形状为 n、m 的张量
so it will t
JACOBIAN
嗯, 它将有1024乘以 M(即2048), 再乘以1024,
Well, it will have 1024 multiplied by M, which is 2048, multiplied by 1024,
GRADi ENt 再乘以 D(也是1024)
multiplied by D, which is 1024.
JACOBIAN So it is
JACOBIAN 因为它的体积太大了.
DO UN ST REAM
JACOBIAN Do WN ST 得是, 我们需要计算这个下游梯度,
JACOBIAN
JACOBIAN So how can we proceed?
GRADi ENt
JACOBIAN The first thin
我想向你展示为什么它实际上是一个极其、极其、极其稀疏的矩阵
And I want to show you why it is actually a super, super, super sparse matrix.
因为如果你看看输入, 输入对输出的影响是什么,
Because if you look at the input, what is the effect of the input on the output,
输入是一系列标记.
the input is a sequence of tokens.
这是第一个标记.
So this is the token number one.
它是一个具有1024维的向量
It's a vector of some dimensions, 1024 dimensions.
然后我们有另一个标记作为输入.
Then we have another token as input.
我们还有另一个标记作为输入, 然后我们有另一个标记作为输入,
we have another token says input then we have another tokens as input
我们乘以由一些列组成的 W矩阵,
and we multiply by the w matrix which is made up of some columns some columns
所以这个矩阵是n乘以d, 对吗?
So this one is n by d Right?
GRADi ENT
而 W是 D乘以 M的矩阵.
And w is D by M.
所以是 D 乘以 M.
So D by M.
这将生成一个 N乘以 M的矩阵.
This will produce a matrix that is N by M.
因此, 它也将是一系列标记, 每个标记由 M个维度组成.
So it will be also a sequence of tokens, each made up of m dimensions.
因此, 它将是一个这样的矩阵.
So it will be a matrix like this.
而这将是第四个输出标记.
And this will be the fourth output token.
现在, 这里的输出行是输入行与所有列的点积结果.
Now this output row here is the dot product of this input row with all the columns.
因此, 这些维度相对于其他所有标记的维度的导数
So the derivative of each of these dimensions with respect to the dimensions
将为零,
of all the other tokens will be zero
因为它们对这个输出没有贡献.
because they do not contribute to this output.
因此
So the Jacobian will have zeros every time
每当我们计算第一个维度
we are calculating the derivative of this first dimension
相对于其他标记任何元素的导数时, Jacobian 矩阵的对应位置都将为零
with respect to any other element of other tokens.
这就是为什么我们总能找到一个更好的公式
That's why we always can come up with a better formula
--来计算这个下游梯度
for computing this downstream gradient
而不需要实际生成 Jacobian 矩阵
that does not involve the materialization of the Jacobian,
因为 Jacobian 本身是稀疏的.
because the Jacobian itself is sparse.
那么, 让我们看看在矩阵乘法的情况下
So let'ssee how we can optimize this computation
如何在不实际生成 Jacobian 的情况下优化这个计算,
without materializing the Jacobian in the case of matrix multiplication
因为这对 Flash Attention 来说是必需的.
because we need it for flash attention.
好的, 各位.
All right guys.
在继续讨论反向传播之前,
so before proceeding to the backward,
我们先来看一下 Flash Attention 反向传播的公式.
watch the formulas of the backward pass of the flash.
在讨论 Flash Attention 之前,
attention, let's look at how to compute
我们先来看看如何计算矩阵乘法操作相对于其输入的梯度.
the gradient of the matrix multiplication operation with respect to its input.
假设我们已经知道, Py Torch 实际上已经提供了
So imagine we are okay Py Torch already have
如何通过损失函数
actually how to compute the gradient of the inputs of the matrix multiplication
相对于矩阵乘法输入的梯度
with the gradient of the loss with respect to the input
来计算矩阵乘法输入的梯度的方法.
of the matrix multiplication operation.
但在 Flash Attention 中, 我们正在创建一个自定义内核,
But in flash attention, we are creating a custom kernel,
这意味着这个自定义内核将多个操作融合为一个操作.
which means that the custom kernel is f using multiple operations into one operation.
因此, 当 Py Torch 调用我们的操作符时, 它会询问我们的操作符
so when py torch will knock the door of our operator, it will ask the our operator,
也就是我们构建的 Triton Attention 操作符
which is the triton attention operator that we have built,
损失函数相对于q 、k和v的梯度是多少
what is the gradient of the loss function with respect to q, k and v,
因为这些都是我们函数的输入.
because that's the input of our function.
如果我们看一下目前已经构建的代码,
so if we look at the code that we have built so far,
stride _0_head= Q. stride(1), 如果我们看一下目前已经构建的代码
soif we look at the code that we have built so far,
> TIMELINE OUTLINE
stride_ V_head= V. stride(1),
278
stride h= Q. stride(θ) 你会发现我们的 Triton 操作将在计算图中作为一个节点
You can'see that our Triton rotation will be a node in the computation graph > TIMELINE OUTLINE
stride_ V_head= V. stride(1),
d= Q. stride (1),
stride(o)
= Q. stride(2),
stride(e) 它接收 Q、
stride_ V_batch= V. stride(0)
that takes as input Q,
> TIMELINE OUTLINE
stride_ V_head= V. stride(1),
stride _0_head= Q. stride(1),
stride(2)
K和 V作为输入, 并生成一个输出,
> TIMELINE OUTLINE
stride_ V_head= V. stride(1),
stride _ Q_head= Q. stride (1),
stride
_batch =q. stride (0),
stride_0_seq 然后,!
Py Torch 会提供损失函数相对于该输出的梯度
Then Py Torch will give us. the gradient of the loss with respect to that output.
> TIMELINE OUTLINE
stride_ V_head= V. stride(1),
278
stride_0_head= Q. stride(1), 因此, Py Torch 会给我们一个d O (即损失函数相对于输出的梯度)
284
283
stride_ K_seq= K. stride(2)
So itwillgiveusa D O
> TIMELINE OUTLINE stride _ V _head = V. stride (1),
stride_ V_batch= V. stride(0)
stride _0_head= Q. stride (1),
stride
Q_batch= Q. stride(0), 也就是损失函数相对于输出 O的导数.
So the derivative of the loss with respect to O > TIMELINE OUTLINE
stride_ V_head= V. stride(1),
stride _0_head= Q. stride (1),
stride
Q_batch=q. stride(0),
279 接下来, 我们会要求这个类 一一 即 Triton Attention -
And then well ask this class here, so try to attention,
> TIMELINE OUTLINE 285
286
stride _ V_head= V. stride(1 ),
stride _0_head= Q. stride(1), 计算损失函数相对于 Q、 K和 V的梯度
to compute the gradient of the loss with respect to Q, K, and B.
28
> TIMELINE OUTLINE
stride_ V_head= V. stride(1),
stride _0_head= Q. stride(1), 因为我们是将多个操作融合在一起执行.
Becau'se we are f using multiple operations together.
> TIMELINE OUTLINE
stride_ V_head= V. stride(1),
stride _ Q_head= Q. stride(1), 因此, 我们是在实时计算
(2)
stride_ Kheac stride _ K_bat stride _ V _batch = V. s
so :we are computing on the fly > TIMELINE OUTLINE
stride_ V_head= V. stride(1),
stride _ Q _head = Q. stride (1),
stride_ Q_seq= Q. stride(2) 查询
(query )与键
(key) 转置相乘后的softmax the soft max of query multiplied by the transport of the key,
> TIMELINE OUTLINE
stride_ V_head= V. stride(1)
stride _ O_
stric. stride (θ) 接着再进行soft max 操作, 并将其与值( V )相乘, 从而得到输出结果
and then multiplying doing the soft max and multiplying it by V to compute the output.
> TIMELINE OUTLINE
stride_ V_head= V. stride(1)
stride. stride (e) 我们需要在内部计算这个梯度,! 以便计算输入( Q、 K、 V)白 的梯度.
we need to compute this gradient internally, to compute this gradient of the inputs > TIMELINE OUTLINE
stride _0_head= Q. stride(1),
stride 因此, 由于我们正在将这些操作融合在一起执行
so, because in these operations that we are doing, fusing together,
> TIMELINE OUTLINE
stric
Q. stride(0), 其中包括矩阵乘法 我们需要手动推导矩阵乘法的梯度
there is a matrix multiplication ;we need to derive by hand the matrix multiplication, uh,
> TIMELINE OUTLINE
stride _0_head= Q. stride (1),
stride
h= Q. stride(0 ), 即损失函数相对于矩阵乘法操作
the gradient of the :of the loss function with respect to the input > TIMELINE OUTLINE
stride_ V_head= V. stride(1),
stri
ad= Q. stride (1),
Q. stride (o),
K. stride(θ) 输入的梯度,
stride_ K_head= K. stride(1)
stride_ V_
of. the matrix multiplication operation > TIMELINE OUTLINE
stride_ V_head= V. stride(1),
stride _0_head= Q. stride (1),
strid
h= Q. stride(0), 以便能够将其提供给 Py Torch.
dso that we can provide it top y torch.
> TIMELINE OUTLINE
stride _0_head= Q. stride (1),
stride
0_batch= Q. stride(0), 这正是我们需要推导这个公式的原因
> TIMELINE OUTLINE
stride_ V_head= V. stride(1),
strid
ch= Q. stride(0), 我会用一种非常简单的方式来推导它
will. derive it in the simp in a very simple way, and > TIMELINE OUTLINE
stride_ V_head= V. stride(1)
stride _0_head= Q. stride(1), 然后我们也会对softmax进行同样的推导
And then we will do it for the soft max as well,
> TIMELINE OUTLINE
stride_ V_head= V. stride(1),
stride _0_head= Q. stride(1), 因为这两部分是我们需要手动推导的
because t these are the two things that we need to derive by hand > TIMELINE OUTLINE
stride_ V_head= V. stride(1),
stride _0_head= Q. stride (1),
stride
Q_batch= Q. stride(0), 以便得出 Flash Attention 反向传播的公式.
to derive the formula of the flash attentions backward pass.
> TIMELINE OUTLINE
stride_ V_head= V. stride(1)
那么, 我们开始吧.
So let's start.
假设我们在计算图中有一个节点称为矩阵乘法,
Imagine we have a node in the computation graph called the matrix multiplication,
这个节点正在执行矩阵乘法操作.
and this node in the computation graph is doing a matrix multiplication.
也就是说, 它正在计算以下操作:y等于×乘以 W.
So it is computing the following operation, y is equal to x multiplied by w.
现在, 当 Py Torch 计算这个节点的反向传播时, 它会提供什么作为输入呢?
Now what Py Torch will give us as input when computing the backward pass of this node.
Py Torch 会提供损失函数的梯度,
Py Torch will give us the gradient of the loss,
也就是损失函数相对于这个节点输出的梯度, 即d中/dy,
so it will give us d phi with respect to dy, so the output of this node,
并要求我们计算损失函数相对于输入×和参数w的梯度,
and will ask us to compute the gradient of the loss function,
和 do /dw.
and the gradient of the loss function with respect to dw.
我将演示其中最简单的一个,
the easiest one to work with and the one that i will be showing,
另一个则不会在视频中展示,
and the other one i will not show in the video,
但我会附上 PDF 幻灯片来说明它是如何计算的,
but i will attach the pdf slide on how it is computed,
因为它们的计算方式非常相似,
because they are very similar in the way they are computed,
我不想因为不必要的重复而使视频变得过长.
so i don't want to make the video too long for unnecessary reasons.
现在, 我们来计算损失函数相对于输入
Let's compute the gradient of the loss function with respect to the input,
×的梯度.
so with respect to x.
那么, 如何在不显式构建雅可比矩阵的情况下手动计算呢?
All right, so how to do that by hand without materializing the Jacobian?
因为, 正如 我们所看到的, 我们不能直接通过显式构建雅可比矩阵来使用链式法则
Because, as we have seen, we can not just use the chain rule by materializing the Jacobian,
虽然这是最简单的方法,
which would be the easiest way,
但雅可比矩阵是一个非常大的矩阵, 甚至无法放入 GPU 的内存中
because the Jacobian is a very big matrix that can not even fit in the memory of the Gpu.
因此, 我们需要找到一种更巧妙的方法.
so we need to find a smarter way.
我们利用了雅可比矩阵稀疏的特性,
we exploit the fact that the jacobian is sparse,
希望最终能得到一个不需要显式构建庞大
so hopefully we will get formula that does not involve the materialization of a very big,
稀疏雅可比矩阵的公式.
sparse jacobian.
让我们来看看具体怎么做.
let's see so.
嗯, 让我们看看, 嗯, 让我们换个角度.
uh, let's see um, let's change.
在处理这类推导时, 我总是建议从一些具体的例子入手:
when dealing with this kind of derivations, i always recommend to make some exam ple:
比如张量.
tensors.
假设×是一个大小为... 的张量.
so suppose that that x is a tensor of size.
假设x是一个大小为nxd 的张量, 其中n设为 1, d设为
let's say n by d, where N, let's say N, is equal to one and D is equal to, let's say,
three.
而 W 也是一个张量, 或者说矩阵, 形状为dxm,
And w is a tensor also, or a matrix, with the shape, let's say, D by M,
其中m设为4.
where M is equal to, let's say, four.
因此, Y的形状将是nx m.
And Y will have, as a consequence, the shape N by M.
所以它的形状将是1×4.
So it will have the shape, well, one by four.
Py Torch 将为我们提供以下结果.
What Py Torch will give us, Py Torch will give us the following quantity.
它将生成这个结果.
So it will give us this stuff here.
即损失函数相对于该操作符输出
So the gradient of the loss function with respect to the output of this operator,
Y 的梯度.
which is Y.
因此, 它将生成一个维度为n×m的向量
So it will give us a vector or a tensor actually with the following dimension,
或张量.
which is N by M.
我们需要计算损失函数相对于×的梯度
and we need to compute the gradient of the loss function with respect to x,
这将是一个形状为n xd 的张量. 因为在处理梯度时,
which should be a tensor of shape n by d, because when dealing with the gradient,
它总是与输入变量的形状相同,
it always has the shape of the input variable,
这是由于输出是一个标量, 相对于输入中的每个元素而言,
because it's the output which is a scalar with respect to each element in the input,
因此梯度的形状与分母一致.
so it has the same shape as the denominator.
好的.
All right.
因此, 在处理这类问题时, 我通常建议先创建示例矩阵,
so when dealing with this kind of problems, I always recommend to create example matrices
观察输出结果的变化, 然后再尝试推导出梯度矩阵.
and then work out what happens to the output and then try to work out the gradient matrix
那么, 我们就开始吧.
So let's do it.
让我们来看看输出是如何计算的.
So let's see that how is the output computed?
输出将是一个1×4的矩阵.
Well, the output will be a matrix that is one by four.
其计算过程如下.
computed as follows.
输入是一个1×3的矩阵.
It will be the input, so one by three.
我们将输入记为 X11、 X12、 X130
So let's call the input x one on e, x one two, x one three.
这个输入将与一个3×4维的矩阵 W相乘.
It will be multiplied by another matrix, w, that it has dimension three by four.
矩阵 W有3行4列.
So it will be three rows by four columns.
矩阵 W的元素记为 W11、 W12、 W13、 W140
So it will be w one, one w one, two w one, three w one, four.
W340 then w21, w22, w23, w24, w31, w32, w33, w34.
如果我们进行矩阵乘法运算, 结果将会是这样的.
if we do this matrix multiplication, it will be well.
运算后将生成如下矩阵, 这是正确的.
it will produce the following matrix : that is okay.
这是一个1行3列的矩阵,
this is one row by three columns.
这是一个3行4列的矩阵
this is a three column, three rows by four columns.
因此, 输出的矩阵将是1行4列的.
So the output will be a matrix that is a one by four.
即1行4 列的矩阵.
So one row by four columns.
为了便于展示, 我将用较小的字体书写, 否则这里可能放不下.
So it will be, let me write it with a smaller because otherwise it will never fi t here.
那么, 我们就这么处理吧.
So let's do it like this.
结果将是 X1乘以 W11, 加上 X12乘以 W21,
It will be X one one multiplied by W one one plus X one two multiplied by W21
XW+ XW2 再加上 X3乘以 W310
plus X13 multiplied by W31.
X W + X 这将是输出矩阵的第一个元素.
And this will be the first element of the output.
输出矩阵的第二个元素将是×11乘以 W12,
The second element of the output will be Xi1 with w12,
加上 X12乘以 W22, 再加上 X13乘以 W320
X11 with w12 plus X12 with w22 plus x one, three with w, three, two.
这将是输出矩阵的第二个元素.
this will be the second element of the output matrix.
输出矩阵的第三个元素将是 一一让我把这些内容移到左边
the third element of the output matrix will be-let me move this stuff on the left,
否则可能放不下.
otherwise it will never fit.
好了, 现在应该能放下了.
so okay, i think now it can fit.
这个元素将是×1乘以 W13, 加上 X12乘以 W23,
this will be x- i need also to watch this one - so x one, one with w one, three x one,
再加上 X13乘以 W330
x11with w13plusx12withw23plusx13withw33,
接 着, 我们将同一行与最后一列相乘, 得到 X11乘以 W14, 加上 X12乘以 W24
and then we multiply the same row with the last column, so it will be x11, w14 plus x12,
再加上 X13乘以 W340
w24 plus x13, w34.
这就是矩阵乘法得到的输出 Y.
This will be the output Y if we do the matrix multiplication.
这就是 Py Torch 会提供给我们的结果.
What Py Torch will give us.
它会给出损失函数的梯度.
it will give us the gradient of the loss.
它会给出delta phi 相对于deltay 的梯度, 因为这就是梯度的含义.
it will give us delta phi with respectto delta y, because it's a gradient.
它的形状与分母相同, 因此其形状为1×4.
it has the same shape as the denominator, so it has a shape that is one by four.
我们暂且称之为未知值, 因为目前还不清楚这个数值会是多少.
let'scall it because we don't know what this value will-be.
这些值将由 Py Torch 提供给我们.
they will be provided to us by pi torch.
我们暂且给它们起个通用名称, 比如dy 11、dy 12、
let's just give them generic name, like d y one one, d y one two,
dy13和dy14. 现在,
d y one three and d y one four, like this : Now,
为了计算我们需要提供给 Py Torch 的下游梯度,
to compute the downstream gradient that we need to provide to P y Torch,
我们应该构建 Jacobian 矩阵, 也就是
we should be computing the, we should be materializing the Jacobian, which is um,
which is uh.
好的, 让我们写下链式法则的公式.
okay, let's write the chain, the chain rule formula.
因此, 我们需要提供delta phi 相对于delta x 的梯度,
so we need to provide delta phi to with respect to delta x,
这等于delta phi 相对于deltay (由 Py Torch提供)
which is equal to delta phi with respect to delta y- this is provided by pi torch -
乘以 Jacobian 矩阵, 即deltay 相对于deltax的梯度.
multiplied by the jacobian, which is delta y with respect to delta x.
现在, 与其直接构建这个 Jacobian 矩阵, 我们不妨尝试另一种方法.
Now, instead of materializing this Jacobian, let's try to do this.
现在, 我们尝试将其具体化,
let's materialize it now
并对这两个量进行乘法运算, 看看是否能简化某些部分.
and let's do the multiplication of these two quantities to see if something simplifies.
那么这里的部分就是dy 相对于dx 的导数,
So this stuff here will be dy with respect to the x,
也就是每个输出y 相对于每个输入×的导数.
which means the derivative of every output y with respect to every input x.
我们有多少个输出呢?
How many output we have?
我们有四个元素作为输出, 也就是这里的这些.
We have four elements as the output, which is this stuff here.
而输入矩阵×中有三个元素.
and we have a three element as input in the x matrix.
因此, 结果将如下所示:我
so it will be as follows : i -
无法直接复制它, 因为我的屏幕不够大
i don't know how to let me copy it because my screen is not big enough -
我记得x是x1、x2和x3.
and i remember that x is x, 1, 1 and xx 2.
因此, dy 相对于dx 的导数将包含以下项:
so delta y with respect to delta x will have the following entries :
y1相对于x11的导数一一可以看到,
so the y1 with respect to x11 - and, as you can see,
y1中只有一个x11与w11相乘.
y1 only has one xi1 appearing as multiplied by w11.
因此, 相对于x11的导数将是w11, 然后是y11.
So the derivative with respect to x11 will be w11, then y11.
这就是结果.
so this is stuff.
相对于x12, 它将是w21, 然后是×.
with respect to x12, it will be w21, Then X.
Y11相对于×13的导数将是 W31.
Y11 with respect to X13 will be W31.
这个矩阵的第二行将是
The second row of this matrix will be
第二个输出 Y2 the derivative of the partial derivative of the second output,
相对于所有×输入的偏导数,
so Y2, with respect to all the X inputs, which will be the derivative,
也就是这里的这些项相对于每个×的偏导数, 分别是 W12、 W22
partial derivatives of this stuff here, with respect to every X which is W12,, W22,,
和 W32.
I guess, and w32.
现在让我检查一下我做的对不对.
Now let me check if what I'm doing is correct.
是的, 因为我已经做过了, 所以我可以随时复查一下.
Yes, because I've already done it, so I can always double check.
然后我们有 W, 这里的这些项相对于所有×的偏导数
And then we have W, the partial derivatives of this stuff here with respect to all the X,
分别是 W13、 W23和 W33.
which is W13, W23, and w33.
然后是最后一个输出y4 相对于所有×的偏导数,
Then the partial derivatives of the last output, so y4, with respect to all the x,
分别是w14、w24和w34.
which will be w14, w24andw34.
我们得到了如下的 Jacobian 矩阵.
We obtain the following Jacobian.
但这个 Jacobian 矩阵, 如你所见, 其实就是w 的转置.
But this Jacobian, as you can see, is just equal to w transposed.
所以我们不需要具体化这个 Jacobian 矩阵.
So we don't need to materialize the Jacobian.
我们只需要将 Py Torch 提供的梯度
We can just do the multiplication of whatever gradient Py Torch is giving us,
与 W 的转置相乘, 就能得到下游的梯度.
multiply it by w transpose, and we will get the downstream gradient.
那么让我重写一下, 这样我们就清楚自己在做什么了.
So let me rewrite so we know what we are doing.
will be sparse, but with a "repeating pattern "
will be sparse, but with a "repeating pattern ".
Then Jacob jan worked eut for N=lis there p
因此, dΦ/dx等于dΦ/dy乘以dy/dx.
So d phi on dx is equal to d phi with respect to y multiplied by dy on dx.
大 但我们已经看到, dy/dx其实就是w的转置.
but we have seen that dy on dx is just equal to w transposed.
所以, d/dx等于dΦ/dy乘以w的转置,
so this is equal to d phi on dx, dy multiplied by w transpose,
这样我们就得到了下游的梯度.
and this gives us the downstream gradient.
因此, 为了提供 Py Torch 所需的下游梯度,
so in order to provide the downstream gradient that py torch need,
我们只需将 Py Torch 提供的梯度与w的转置相乘
we just need to take whatever gradient py torch will give us multiplied by w transpose,
大 就能得到损失函数
and it will give us the gradient of the loss function with respect to the input x
关于矩阵乘法输入×的梯度.
of the matrix multiplication.
同样地,
in the same way,
我们也可以写出损失函数关于 W 的梯度公式
we can also write the formula for the gradient of the loss function with respect to w,
它等于x的转置乘以d中/dy关于dw的部分.
and it is equal to x transpose multiplied by d phi with respect to dw dy.
如何记住这些公式呢?
How to remember these formulas?
这里有一个记忆法则, 那就是
These are, there is a mnemonic rule, which is,
这些是唯一能让这个公式符合×的形状,
these are the only possible ways for this to have the shape of x
那个公式符合 W的形状的方式.
and this to have the shape of w.
因为这里的这个部分, 会与 Y 的形状相同.
Because this ones, this stuff here, will have the same shape of Y.
因此它的形状将是n乘以m.
So it will be n by m.
而这里的这个部分, 形状将与 W 的转置相同.
this stuff here will have shape of w transpose.
W 的尺寸是d乘以m,
w is d by m,
所以w的转置应该是m乘以d, 而矩阵乘法
so w transpose should be m by d and the resulting operation of this matrix multiplication,
或张量乘法的结果将是/n乘以d, 这与×的形状完全一致.
or tensor multiplication, will be n by d, which is exactly the same shape as x.
在这种情况下, xt 是t的转置, 尺寸为n乘以d,
In this case we will have that xt is the transpose of t and it is n by d,
因此是d乘以n再乘以d中/dy, 这是一个梯度,
so it's d by n multiplied by d phi with respect to dy, which is a gradient,
所以它与分母的形状相同.
so it has the same shape as the denominator.
因此, 它的尺寸是n乘以m, 而输出的形状将是d乘以m, 这正好与
So it has n by m, uh, and the output will have um shape d by m, which is exactly the um,
W 的形状一致.
the shape of w.
所以如果你要记住这些关系, 这是唯一能让形状匹配的方式,
so if you, if to remember them, this is the only way this shape work out,
否则就无法成立.
otherwise they don't work out.
因此, 这是一个用于记忆
so this is a mnemonic formula on how to remember
如何根据矩阵乘法输出的损失梯度,
how to compute the gradient of the inputs of a matrix multiplication,
计算输入梯度的记忆公式.
given the gradient of the loss with respect to the output of the matrix multiplication.
而矩阵乘法的输入是输入矩阵
and the inputs to the matrix multiplication are the input matrix
和参数矩阵 W
and the parameter matrix w.
现在我们需要推导soft max 输出
Now we need to derive the gradient of the output of the soft max
相对于其输入的梯度,
with respect to the input of the soft max,
因为这是我们在融合注意力机制中进行的另一个操作.
because that's another operation that we do in our fused attention,
我们将多个操作融合在一起
because we a re fusing many operations together,
包括矩阵乘法和soft max.
which is matrix multiplication and the soft max.
因此, 这是理解 Flash Attention 反向传播
So this is the second ingredient
所需的第二个关键要素.
that we need to understand the backward pass of flash attention.
那么, 让我们开始吧.
So let's do it.
在进行这个推导时, 我将
I will use, to make this derivation,
采用与 Flash Attention 论文中相同的符号表示.
I will use the same notation as in the flash attention paper.
首先, 让我们为这部分内容写上标题:
So, first of all, let's write the title of this stuff,
通过soft max 的梯度计算.
which is the gradient through the soft max.
在计算注意力机制时, 我们首先进行的操作是
The first operation that we do during computation of the attention is
计算查询向量与键向量转置的乘积.
we compute the product of the query multiplied by the transpose of the keys.
我们以分块的方式进行计算, 即逐块处理,
We do in a block wise ways, means that we do it block by block,
但这并不影响最终结果, 因为最终效果是相同的
but it doesn't matter because the end result is the same.
因此, 我们可以将 S表示为 Q与键转置的乘积.
So we can write S equal to Q multiplied by the transpose of the keys.
接着, 我们对这个结果应用soft max 函数.
And then we apply the soft max to this operation.
将这一操作的结果称为 P, 也就是 S的soft max 输出.
to the result of this operation and we call this output P, which is the soft max of s.
在应用soft max 之后, 我们取其输出
and after we have applied the soft max, we take the output of the soft max,
并将其与 V 相乘, 从而得到最终的结果.
we multiply it by V to obtain the output.
因此, 输出等于 P乘以 V.
so the output is equal to P multiplied by V.
现在我们需要理解如何进行计算, 因为正如我之前提到的
Now we need to understand how to because, as I said before,
Py Torch 的 Auto Grad 机制是按照以下方式工作的.
Py Torch Auto Grad works in the following way.
Py Torch 会将我们的注意力计算视为一个黑箱.
Py Torch will treat our attention computation as a black box.
因此, 我们将得到如下的计算图.
So we will have a computation graph like the following.
我们将有一个 查询输入、一个键输入和一个值输入, 这些都是由一系列标记组成的序列
We will have a query input, a key input, and a value input, which are sequences of tokens,
每个标记都有一定的嵌入维度.
each one with some embedding dimension.
这些输入被送入一个称为注意力的黑箱
these are fed to some black box called the attention,
这是我们自己实现的注意力机制,
which is our implementation of the attention,
也就是我们之前开始编写的那个函数.
which is the function that we started coding before.
这些输入将作为计算图中这个节点的输入,
this will be fed as input to this node in the computation graph,
而计算图将输出一个张量0.
and the computation graph will output a, an output tensor o.
Py Torch 会给我们什么?
what py torch will give us a?
Py Torch 会提供损失相对于输出的梯度.
py torch will give us the gradient of the loss with respect to the output.
正如你所记得的, Py Torch 会逐个访问每个运算符并询问:
So, as you remember, Py Torch knocks the door at each operator and says :
如果我将损失相对于你输出的梯度提供给你,
if I give you the gradient of the loss with respect to your output,
你能返回给我损失相对于你输入的梯度吗?
can you give me the gradient of the loss with respect to your inputs?
这正是我们需要解决的问题.
And this is what we need to figure out.
因此, 在已知损失相对于输出的梯度的情况下,
So given the gradient of the loss with respect to the output,
我们需要弄清楚如何计算损失相对于 WQ 的梯度.
we need to understand how to compute the gradient of the loss with respect to w Q.
以及损失相对于 WK 的梯度, 和损失相对于 WV 的梯度.
the gradient of the loss with respect to w K, the gradient of the loss with respect to wv.
然而, 由于存在两个中间操作,
However, there is no direct connection between Q and O or K and 0
Q 与 O之间或 K与 O之间并没有直接的连接.
because there are two intermediate operations.
首先是一个矩阵乘法, 接着是soft max 操作
So one, there is a first matrix multiplication, then there is a soft max,
然后再进行另一个矩阵乘法.
then there is an additional matrix multiplication.
然而,
However,
我们拥有工具能够帮助我们理解
we have tools that allow us to understand
梯度是如何通过这些操作传播的.
how the gradient propagates through multiple operations
当多个操作依次应用时,
when they are applied in sequence.
这就是所谓的链式法则.
And that's called the chain rule.
然而,
However,
我们已经看到, 如果以最直接的方式应用链式法则
we have seen that applying the chain rule in its naive way
并具体化雅可比矩阵, 实际上是不可行的.
while materializing the Jacobian is infeasible.
因此, 我们需要理解如何在不具体化雅可比矩阵的情况下应用链式法则
So we need to understand how to apply the chain rule without materializing the Jacobian.
这正是我们将要
And that's what we are going to figure out
针对注意力计算中的一个操作
for one of the operations inside of this attention computation,
-soft max 一- 去解决的问题.
which is the soft max.
这就是为什么我们要进行这次推导,
And that's why we are going to do this derivation,
我保证这是我们将要做的最后一次推导.
which I promise is the last one that we will do.
然后, 我们最终将着手编写flash attention 的反向传播代码.
And then we will finally go to code the backward pass of flash attention.
我们无法直接着手编写flash 的反向传播代码.
We can not proceed directly to coding the backward pass of the flash.
如果我们直接看attention 的计算公式,
attention, because if we look at the formulas on how it is computed,
是无法理解推导过程是如何得出的.
we will not understand how the derivation comes out.
好的, 现在我们可以开始了.
Okay, now we can start.
让我把这些内容删掉吧.
So let me delete this stuff.
删除.
delete.
为了简化理解, 假设我们现在对s 矩阵逐行应用soft max,
and imagine for simplicity : now we apply the soft max to a row wise to this s matrix,
也就是说每一行都独立地进行soft max 计算.
so each row is soft maxed independently from the others.
那么, 让我们看看矩阵的某一行会发生什么变化, 为了简化,
so let's see what happens to one single row of this matrix and for simplicity,
我将这一行称为 S.
i will call it s.
所以, S代表s矩阵中的某一行.
so s is a single row of the s matrix.
我也可以称它为s的第i行
i could also call it s of i,
但如果这样表示, 我们就得一直带着这个索引1了.
but If I do it like this we will have to carry over the index.
好吧, 伙计们, 动手做吧.
Okay, guys, just do it.
我们会带着这个索引一起推导.
We will carry over the index.
好的, 我们就用 SI来表示 S矩阵中的某一行吧.
All right, so let's call Sl one row of the S matrix.
因此, SI的表达式, 用张量表示法一一也就是 Py Torch 的张量表示法
So Sl is equal to, let's say it's the, in tensor notation, Py Torch tensor notation,
会是这样的.
it will be like this.
也就是说, 从矩阵 S或者说张量 S中, 我们取出第i行以及所有的列.
So from the matrix S, from the tensor S, we take the i th row and all the columns.
这就是 SI 的定义.
This is the definition of Sl.
我知道这个表示法看起来不太美观, 但它有助于你们理解.
I know it's very ugly notation, but it helps you understand.
这是一个具有特定大小和维度的向量.
and this is a vector of size and dimensions.
我们在 这个向量上应用soft max 函数, 将会得到一个输出向量, 我们称之为 Pl.
We apply the soft max to this vector and we will obtain an output vector and we call it Pl.
PI等于 SI的soft max.
Pl is equal to the soft max of Sl.
正如我们所见, soft max 操作不会改变输入的形状,
So, as we have seen, the soft max operation does not change the shape of the input,
它只是逐个元素地改变数值.
it just changes element-wise each number.
因此, 输出也将是一个大小为r的n次方的向量.
So, the output will also be a vector of size r to the power of n.
那么, 什么是soft max 呢?
Now, what is the soft max?
s of max 的定义如下.
So, the soft max is defined as follows.
即 pij 的 soft max.
The soft max of well pij.
因此, p-i向量的第j个元素等于
So the j-th element of the p-i-th vector is equal to
s-i向量的第j个元素的指数,
the exponential of the j-th element of the s-i-th vector
除以一个归一化因子, 这个因子按如下方式计算:
divided by a normalization factor that is computed as follows with :
这里不用j, 我们用k来表示.
let's say not j, let's use k in this case.
不用k, 我们用 I来表示.
not k, let's use I.
等于从 1到n的e的si次方的和.
is equal to one up to n of e to the power of s i l.
好的, 首先, 你可能会疑惑:
all righ t, so uh, first of all, you may be wondering : the soft max that we are up,
我们在计算注意力机制的前向传播过程中使用的soft max,
that we apply during the forward pass of the computation of the attention,
并不是这个原始的soft max.
is not really this soft max,
因为如果你 还记得之前我们应用的方法, 我们实际上使用的是经过调整的soft max because in, if you remember what we applied before, we were applying the soft max,
其中每个指数函数的参数都减去了
where each of the argument of the exponential is reduced
该向量中的最大值元素.
by the maximum element in the vector to which we apply the soft max.
所以它大致是这样的.
So it was more or less like this.
即 Sij 减去 Simax.
So Sij minus Simax.
也就是 Sij 向量中的最大值元素.
So the maximum element in the Si j, Si vector.
同时, 分母中的参数也减去了 Simax.
And also the argument of the denominator was reduced by Simax.
然而,
However,
我们也证明了这里的操作
We also proved that this stuff here is equivalent to the standard sof tmax
与不进行参数减法的标准soft max 是等价的,
without this reduction in the argument,
因为这种参数减法只是
because this reduction in the argument is only added
为了确保数值计算的安全性.
because we want to make it numerically safe to compute,
但从数学角度来看, 不进行减法的计算方式也是等价的.
but it's equivalent to do it without from a mathematical point of view.
在计算机上, 当然, 这样做可能会导致数值不稳定,
On the computer, of course, it will become numerically unstable,
但从数学的角度来看, 它们是相同的.
but from a mathematical point of view, it is the same thing.
这也意味着无论你如何计算前向传播过程, 结果都是一样的.
which also means that it doesn't matter how you compute the forward pass.
如果它与另一个数学定义等价,
if it's equivalent to another mathematical definition,
你总是可以使用那个数学定义来计算反向传播.
you can always use the other mathematical definition to compute the backward pass.
最终得到的结果将是相同的.
it will result in the same value.
如果你没听懂我刚才说的, 让我用一个更简单的例子来说明:
if you didn't understand what i said, let me give you a more simple example, which is :
想象你有一个a.
imagine you have a.
你还记得高中时学过的那个公式吗?
do you remember the formula from high school?
就是这个:cos2x+ sin2x=1.
this one : so cosine cosine of squared of x plus sine squared of x is equal to one.
现在, 假设我们计算一个输出:y=cos2x.
now imagine we compute an output : y is equal to cosine squared of x.
然后我们需要计算y 对×的导数.
and then we need to compute the derivative of y with respect to x.
无论你是将cosx 对×求导,
it doesn't
还是将1-Sin2x对x求导
or if you compute it as the derivative of one minus sine squared of x with respect to x,
结果都会完全相同,
because they will result in exactly the same result,
因为这两个定义是等价的.
because the two definitions are equivalent.
正因为如此, 我们无需在指数部分额外添加这个因子,
And this is why we don't need to add this factor in the exponential
因为从数学上讲, 这两种定义是等价的.
because the two definitions are equivalent mathematically.
我们只需采用数值上更安全的那个定义, 因为在计算机上进行计算时
We just use the numerically safe one because when computed on the computer,
我们需要确保数值稳定性.
we need something that is numerically stable.
这样就不会出现溢出问题.
that will not overflow.
好的, 那么, 我们想要得到什么呢?
All right, now, what do we want to obtain?
所以, 我们希望在已知损失函数关于soft max 输出(即 Pi向量)的梯度的情况下
So we
所以, 我们希望在已知损失函数关于soft max 输出(即 Pi 向量)的梯度的情况下
want to obtain the gradient of the loss with respect to the input vector of the soft max,
计算出损失函数
which is the Si vector,
关于soft max 输入向量(即 Si 向量)
given the gradient of the loss with respect to the output of the soft max,
的梯度.
which is the Pi vector.
通过链式法则, 我们可以得到这个结果.
multi, and we can obtain that with the chain rule.
将其乘以 Pi关于 Si的雅可比矩阵.
multiply that by the jacobian pi with respect to si.
现在, 链式法则始终是有效的.
now we the chain wheel is always valid.
让我们来看看这个雅可比矩阵是什么样子的.
let's see what does this jacobian look like?
嗯, 好的, 那么这个雅可比矩阵就是 DPI 关于delta SI的导数.
um, all right, so this jacobian will be DPl with respect to delta Sl.
嗯, 我们需要完成这个计算.
Well, we need to do it.
让我们来仔细看看这个雅可比矩阵中每个元素的具体形式.
Let'slook at what each element in this Jacobian will look like.
那么, 第j个元素相对于第k个元素的偏导数.
So the jth element with respect to the, let'ssay the kth element.
我们正在计算, 或者说,
So we are, we are computing the, the,
我们正在研究这个雅可比矩阵中每个元素的具体形式,
we are looking at what each element in this jacobian will look like,
也就是雅可比矩阵到底是什么.
which is what is the jacobian?
它指的是雅 可比矩阵中输出(分子)的每个元素相对于输入(分母)的每个元素的偏导数
it's each element in the output in the numerator of the jacobian
也就是这个分数形式中的每个分量.
derived
也就是这个分数形式中的每个分量.
with respect to each element in the denominator of the jacobian In this fraction here.
也就是说, 我们正在分析输出向量中的每一个元素
so we are saying for each element in the output vector
相对于输入向量中每一个元素的偏导数.
Derived with respect to each element in the input vector.
这就是我们在这里写的内容. 那么, 输出向量是如何得到的呢?
This is what we are writing here So what is how is the output vector obtained?
嗯, Pij, 我们知道它等于.
Well, Pij, we know that it is equal to.
根据soft max 的定义, 它是通过以下方式得到的.
by the definition of the soft max is obtained as follows.
即e 的 Si次方除以归一化因子, 我们称之为 L,
so e to the power of Sij Divided by The normalization factor, let's call it L,
等于 1到 n.
is equal to 1 to n.
e的 Si L 次方, 全部对 Sik 求导.
e to the power of s i L all derived withrespect to S i k.
i k.
i k.
所以, 我们正在尝试做的是, 我们知道 P 向量是
so what we are trying to do is we know that the p vector is -
) Sik
假设它是一个包含三个元素的向量, 那么这是p1,
) Sik
RW ) Sik
这是p11, p12 和p13.
this is a well plone one one one, p one two and p one three.
) Sik
s向量也将是一个包含三个元素的向量, 因此它将包含s11、
J Sik
S12 和 s13.
s ohe two and s one three.
我们的标是计算 Jacobian 矩阵,
) Sik
然后
Jacobian 矩阵的第二行将是
这个向量对每一个输入元素的导数.
) P :
the derivative of this one with respect to each of this input element.
J Sik
接着, Jacobian 矩阵的第三行将是这里的内容
Then the third row of the Jacobian will be this stuff here ) Sik
对 S向量中每一个输入元素的导数.
t to each of the input element of the S vector.
) Sik
我们正在努力理解这一点.
) Pes W
Ve are trying to understand.
这个 Jacobian矩阵中的通用元素是什么样子的.
) Sik
RW
基于输出向量的第j个元素,
) Sik
因此 这个j索引指的是输出向量, 而k索引指的是输入向量中的第k个元素
) Sik
好的, 当我们计算这个 Jacobian 时, 可能会出现一种情况.
all right, so what can happen when we do this jacobian is that we have a.
这里的这个表达式是两个函数的商的导数,
this one here is the derivative of a fraction of two functions,
我们从高中就知道, 两个函数的商的导数
and we know from high school that the derivative of the fraction of two functions
如下所示.
is as follows.
所以, f(x)对g(x)的导数一一让我这样写
so the derivative of the derivative-let me write like this -
关于×的导数等于
of f of x with respect to g of x, prime is equal to, with respect to x, by the way,
f(x)乘以g(x)减去g'(x)乘以f(x), 除以g(x)的平方.
is equal to f prime, oops - of x multiplied by g of x minus g prime of x.
f(x)除以g(x)的平方.
f of x all divided by the g of x to the power of two.
就像这样.
like this.
现在让我们在这里应用它.
Now let's apply it here.
所以这将变成, 这里我们会有两种情况.
So this will become, here we will have two cases.
要么我们求导的变量
Either the variable that we are deriving with respect to,
即这个s_i_k, 与被求导的变量具有相同的索引.
so this s i k has the same index as the variable being derived.
所以我们要么是在计算 P11对 S11的导数,
So either we are doing P11 with respect to S11
) Pes
要么是在计算 P11对另一个不同索引的变量的导数.
) Sik
比如 P11对 S12或 S13的导数.
So like Pi1 with respect to S12 or S13.
J Sik
因此, 我们需要考虑两种情况.
So there are two cases that we need to consider.
假设我们在计算 P11对 S11的导数,
Suppose that we are deriving P11 with respect to S11
e
或者 P12对 S12. 的导数, 或者 P13对 S13的导数.
or we are deriving. P12 with respect to S12 or we are deriving P13 with respect to S13.
Sie
也就是说, 我们在计算输出向量中某个元素
So we are y deriving the element of the output
对输入向量中具有相同索引的元素的导数.
with respect to the same element in the input with the same index.
e
在这种情况下, 这个导数会呈现如下形式:
so in this case, the this um, this derivative will look like the following :
它是函数f
so it's the derivative of f,
对分母中具有相同索引的变量的导数.
so the numerator with respect to the denominator that has the same index.
也就是说, 在这种情况下, j等于k.
so we are saying that in this case j is equal to k.
因此, 分子对 Sij的导数, 也就是e的 Sij次方对 Sij的导数,
so The numerator with respect to Sij, with respect to e, to the power of Sij,
结果就是e的 Sij 次方.
with respect to Sij, will be e to the power of sij.
因为e的x1次方对x1的导数就是e的x1次方.
Sobecauseeto the power of xl with respect to xl will bee to the power of xl.
所以这等于, 我现在要缩小一下尺寸.
So this is equal to, I am reducing the size now.
e的 Siji次方, 然后我们需要将其乘以分母
e to the power of Sij, then we need to multiply that by the denominator of the fraction,
也就是这里的这个求和项.
which is this summation here.
所以对所有可能的 L求和e的 Sij 次方,
So the summation over all possible L of e to the power of Sij,
再减去分母对所求导变量的导数.
minus the derivative of the denominator with respect to the variable being derived.
因此, 这个分母是所有输入元素的指数之和.
so this denominator is the sum of all the exponentials of all the input elements.
如果我们对某一个特定的输入元素求导,
if we derive it with respect to one particular input element,
至少会有一项包含该输入元素,
there will be at least one term that contains that input element,
而其他所有项的结果都会是零.
and so the all the other terms will result in zero.
因此, 唯一剩下的导数将是e的sik 次方
so the only derivative that will survive will be the e to the power of s i k
对sik 的导数.
with respect to s i k.
所以我们写成减去e的sik 次方, 乘以分子,
So we write min us e to the power of sik, multiplied by the numerator,
也就是e的sij 次方.
which is e to the power of sij.
所有这些除以分母的平方.
All this divided by the denominator to the power of two.
也就是这里的这个求和.
which is this summation here.
所以 I从1到n, e的sil 次方, 全部平方.
So I equal to one up to n e to the power of s i l, all to the power of two.
而这里的这一部分将等于, 我们可以看到, 这两个项,
And this stuff here will be equal to, well, we can see that this two term,
这个和这个有一个共同因子, 即e 的sij 次方.
this one and this one have a one term factor in common, which is e to the power of s ij.
所以我们可以提取出来.
So we can collect that.
所以e的sii次方乘以求和减去e的sik次方,
So e to the power of s ij multiplied by the summation min us e to the power of sik
所有这些除以分母
All this divided by the denominator
也就是这里这个东西的平方.
Which is the power of two of this stuff here?
那么让我复制粘贴一下, 同时旋转一下
So let me just copy and paste it, which is, let me rotate it also,
因为我不知道为什么我总是写得小小的
because I don't know why I always write little Little : Yeah, all right,
好了, 这里的东西等于, 嗯, 我们可以把这两项分开.
and this stuff here is equal to Well, we can separate the two terms.
所以我们可以把这里的这一项和这一项分开.
so we can separate this term here and this term here,
因为分母是平方,
because the denominator is to the power of two.
所以我们可以这样写:e的 Sij 次方除以分母,
So we can write it also as: e to the power of Sij divided by the denominator
即 I从1到n的e的 Sil次方的和,
which is summation of I equal one to n e to the power of Sil
乘以这里的东西.
multiplied by this stuff here.
所以这里的东西除以相同的分母,
so this stuff here divided by the same denominator,
即 I从1到n的e的 Sil次方的和,
so there's summation of L equal 1 up to N,
减去e的 Sik 次方,
e to the power of s i L minus e to the power of s i K-
再除以相同的分母, Sil.
I am S i K divided by the same denominator, S i L.
现在这个可以写成:这里的东西不过是输出元素 Pij,
Now this one can be written as : this stuff here is nothing more than the output element,
因为这个只是应用于 Sij 元素的soft max, 我们都知道.
Pij, because this one is just the soft max applied to the Sij element, which we know.
因为应用于 Sij 元素的soft max 输出称为 Pij,
that the output of the soft max applied to the Sij element is called Pij
因为它是我们称为 P 的输出向量的一个元素.
because it's one element of the output vector which we call the P.
所以这里的东西等于 Pij,
So this stuff here is equal to Pij,
乘以这里的东西将等于1 减去这里的东西.
multiplied by this stuff here will be equal to one minus this stuff here.
这里的东西是应用于 SIK 元素的soft max 输出,
What is this stuff here is the output of the soft max applied to the Si K element,
所以它将是 PIK.
so it will be Pi K.
所以它等于1减去 PIK.
So it is equal to one minus PIK.
好的, 这是在这种情况下.
okay, and this is in the case.
我们求导的变量与分子具有相同的索引.
the variable with respect to which we derive has the same index as the numerator.
在这个分数中, 在这个导数中.
in this fraction here, in this derivative here.
另一种情况是当两个变量, 即输出的索引
the other case is when the two variables, so the output,
与输入的索引, 不相同的时候.
the index of the output with respect to the index of the input, are not the same.
在这种情况下, 我们会有另一种情况.
in this case we will have another case.
那么我们会得到丨吗?
so we will have that j?
嗯, 让我再写一遍.
uh, let me write it again.
所以这里的内容, 我希望我能全部复制下来而不出错.
so this stuff here i hope i can copy it all without.
在另一种情况下, 即当s不等于j时,, s是什么?
in the other case, in which s is not equal to j, uh, what s?
S ik 是j不等于k, 所以j不等于k.
it's J not equal to k, so j is not equal to k.
在这种情况下会发生什么?
what happens In this case?
嗯, 这将是分子的导数,
it will be well, the derivative of the numerator,
因为我们需要再次应用这里的这个公式.
because we need to apply again this formula here.
所以, 分子相对于不同变量的导数
So, derivative of the numerator with respect to something that is not the same variable,
P
将为零,
it will be zero,
因为这就像计算e 的×一次方
because it's like computing the derivative of e to the power of x one
相对于×二的导数,
with respect to x two,
结果会是零.
it will be zero.
所以它将为零.
So it will be zero.
因此, 无论g(x)是什么, 这里的整个第一项都会变成零
So all the first term here will become zero,
再减去这个分数
no matter what is g of x minus the derivative of the denominator
分母相对于变量sik 的导数
of this fraction here with respect to the variable s i k.
g, 关于 sik 的导数
%, prime of si k.
所以这是输入中的所有变量
So this is all the variable in the input e
而我们正在对输入中的一个特定变量求导.
and we are deriving it with respect to one particular variable of the input.
e
因此, 在求和过程中只有一项会保留下来, 那就是sik 这一项.
So only one item in the summation will survive, so it will be the item s i k.
e
所以它将是e的sik 次方乘以f(x),
so it will be e to the power of s i k multiplied by f of x,
也就是这个分数的分子, 即e的幂次.
P
哦, 我们漏掉了减去e的 sij 次方这一项.
oh, we forgot a minus e to the power of s i j.
让我看看是否遗漏了什么.
let me see if i forgot something.
所有这些除以这里分数的分母的平方,
all divided by the denominator of this fraction here to the power of two,
所以它等于从 I等于1到n, e的sil 次方的总和
so it is equal to the summation l, equal one up to n of e to the power of s i l,
再整体平方.
all to the power of two.
嗯, 我想我没有遗漏任何东西, 那么我们继续吧.
uh, i believe i didn't for get anything, so let's continue.
所以这里我们可以看到, 这部分是因为一一, 好吧, 我们分开来看
So here also, we can see that this one here is because - uh, okay, let's separate it -
减去e的sik 次方, 除以从 I等于1到n, e的sil次方的总和,
minus e to the power of s i k, divided by the summation l,
再乘以e的sij 次方,
equal one up to n of e to the power of s i l multiplied by e to the power of s i j,
除以从 I 等于1到n, e的sil 次方的总和.
divided by the summation I equal one up to n of e to the power of s i l,
这里的这些东西不过是对 SI向量的第k个元素应用soft max 的结果,
This
这里的这些东西不过是对 SI向量的第k个元素应用soft max 的结果.
stuff here is nothing more than the soft max applied to the kth element of the Sl vector.
这里的这一项不过是对 SI向量的第j个元素应用soft max 的结果
This
这里的这一项不过是对 SI向量的第i个元素应用soft max 的结果
one here is nothing more than the soft max applied to the jth element of the Sl vector.
所以我们知道这些是什么了.
So we know what these are.
我们知道我们称之为 p减去 pik乘以 pij.
We know that we call the m p minus pik pij.
所以最终我们有两种情况.
So in the end we have two cases.
一种是这里这个东西的导数.
One is the derivative of this stuff here.
看起来如下.
looks like the following.
雅可比矩阵中的每一项看起来如下.
Each item in the Jacobian looks like the following.
当分子和分母具有相同索引时.
When the numerator and the denominator have the same index.
即j等于k.
so j equal to k.
这里的东西现在等于.
this stuff here is equal to now.
这里的符号表示有误, 我不应该用等号来书写.
this notation here is wrong, so I shouldn't be writing it with equal sign.
不过没关系, 朋友们, 我们只是稍微讨论一下.
but it doesn't matter, guys, we are doing a little.
e
J Sik
) Sik
Okay, so-pij, PI J multiplied by 1, minus P I K.
让我检查一下.
let me check.
P是的, 一种情况是当」不等于 K时.
那么这里的内容一一让我这样写
then this stuff here-let me write it like this -
将等于负的 PIK 乘以 PIJ will be equal to minus P I K multiply the P I J.
既然我们已经了解了雅可比矩阵的两种典型情况,
Now that we know what the two typical cases of this Jacobian look like,
现在让我们来看看这个雅可比矩阵在矩阵形式中的具体表现.
let's actually look at what this Jacobian look like in the matrix form.
所以这个雅可比矩阵将如下所示.
So this Jacobian will look like the following.
它将是一个大致如下的矩阵.
It will be a matrix that is more or less like the following.
它将是一个nx n的矩阵, 其中n是输入向量和输出向量的大小.
It will be an n by n matrix where n is the size of the input vector and the output vector.
这里是雅可比矩阵的第一个元素.
A t here the first element of the Jacobian.
如你所见, 如你所记得的
as you saw, as you remember,
按照分子布局, 雅可比矩阵的第一行
the first row of the Jacobian in the numerator convention
是第一个输出对所有输入的导数.
is the derivative of the first output with respect to all the input.
因此, 这里的第一个项将是 P11对 S11的导数.
So this first term here will be the derivative of P11 with respectto S11.
因此, 在这种情况下, J和 K匹配, 所以我们知道它将等于 P.
So in this case, J and K match, so we know that it will be egual to P.
P11乘以1减去 P11
one one multiplied by one, minus p one one.
这个元素右边的第二个元素.
the second element to the right of this one.
所以元素 P12将是 P12对... 的导数.
so the element one two will be uh, the derivative of p one two with respect to, uh.
抱歉, 是 P11对 S12的导数.
sorry, p one one with respect to s one two.
J和 K不匹配, 因此我们将处于这种情况, 所以它将是负的 P11
the j and k do not match, so we will be in this case here, so it will be minus p one one,
乘以 P12.
p one two.
第三个元素.
The third element.
你可以自己验证一下.
you can check it by yourself.
它将等于负的 P 11乘以 P13, 依此类推, 直到最后一项
It will be minus P 1, 1, P 1, 3, blah, blah, blah until the end,
是负的 P11乘以 P1n.
which will be minus P 1 1, P 1 n.
雅可比矩阵的第二行将如下所示:
The second row of this, Jacobian, will be, will look like this :
因此, 它将是 P12对 S11的导数.
So it will be the derivative of P 1: 2 with respect to S 1: 1.
J和 K不匹配, 所以我们处于这种情况.
the J and K do not match, So we are in this case here.
因此, 它将等于负的p12乘以p11.
so it will be minus p12, p11.
然后是下一个元素:它将是p12对s12的导数.
then the next element : it will be the derivative of p12 with respect to s12.
因此, j和k匹配, 我们处于第一种情况.
so j and k match, so we are in the first case.
因此, 它将等于p12乘以1减去p12.
so it will be p12 multiplied by 1 minus p12.
那么这里的部分将等于.
Then this stuff here will be equal to.
接着, 第三个元素将是p12对p13的导数, 以此类推.
then the third element will be minus p12 with respect to p13, blah, blah, blah.
直到我们到达最后一个元素, 即负的p12乘以p1n,
And until we arrive to the last one, which is minus p12 with respect to p1n,
而不是 p12对 p1n的导数.
not with respect to, multiplied by pin.
所有元素都是如此, 直到最后一行.
and all the elements like this until the last row.
最后一行将是.....
the last row will be the.
最后一行的第一个元素将是
the first element of the last row will be the derivative of the uh,
最后一个输出元素对第一个输入元素的导数.
last output element with respect to the first input element.
因此, 它将是p1n对s11的导数.
so it will be the derivative of pin with respect to s11.
所以, 这两个索引并不匹配.
so, um, the two indices do not match.
所以我们处于第二种情况.
so we are in the second case.
所以它将是负的p1n乘以p11.
so it will be minus p 1 n, p 1 1.
这将依次为负的p1n乘以p12, 依此类推.
this will be minus p 1 n, p l, 2, etc, etc, etc.
既然我们在这里, 我也来算一下第三个元素吧.
let me do also the third element, since we are here.
所以是负的p1n乘以p13, 以此类推.
so minus p 1 n, p 1, 3, etc, etc, etc.
直到最后一行的最后一个元素, 我想应该是负的p1n乘以p1n.
until the last element of the last row, which will be minus p 1 n, p 1 n, i guess.
哦不, 这样不对, 因为这两个索引是匹配的.
Oh no, that's wrong, guys, because the two indices match.
所以应该是 P1n乘以 1减去 P1n.
so it should be P1n multiplied by one, minus P1n.
这就是雅可比矩阵的样子.
This is what the Jacobian will look like.
让我们看看能否通过模式识别找到更好的方法
Let's see if we can find a better,
来生成这个雅可比矩阵.
how to generate this Jacobian with some pattern recognition.
让我们换一种方式来写.
Let's write it in a different way.
首先, 我们可以注意到这个雅可比矩阵是对称的.
First of all, the first thing that we can notice is that this Jacobian is symmetric.
因此, 你可以看到这个元素等于这个元素.
So you can see that this element is equal to this element.
如果你展开第三行, 你会发现它等于这个元素.
If you expand the third row, you will see that it's equal to this element.
右上角的这个元素等于左下角的那个元素.
This one on the top right corner is equal to the one in the top bottom left corner.
所以这个矩阵是对称的.
So this matrix is symmetric.
其次, 我们可以注意到
the second thing that we can notice is that
只有对角线上的元素是不同的.
only the element in the diagonal are different.
它们有一个额外的项, 因为你可以看这里的这个元素, 让我写出来
they have an additional term because you can look at this element here, so let me write :
这里的这个元素也可以写成p11减去p11乘以p11.
this element here can also be written as p11 minus p11 multiplied by p11.
第二行中的第二个元素,
the second element here in the second row,
也就是这个矩阵的第二个对角线元素, 是p12减去p12乘以p12.
so the second diagonal element of this matrix is p12 minus p12 multiplied by p12
P
Pn (t-PN
所以对角线上的这个元素实际上看起来和其他元素一样.
PQ so this element on the diagonal actually look like just like the other elements.
它们只是多了一个额外的项, 第一个对角线元素中是 P11 they just have an additional term, which is P11 in the first diagonal element,
-s
-Pw P2
第二个对角线元素中是 P12.
P12 in the second diagonal element.
因此我们也可以说这里的这个矩阵是
So we can also say that this matrix here is the product
所有可能的 Pij与 Pik 组合的乘积,
of all the possible combinations of Pij with Pik
这些组合可以通过外积获得,
which we can obtain with an outer product
甚至可以通过一列与其转置的乘积得到.
or even with the product of one column with the transpose of the same column.
所以, 如果你取一个列向量一一例如, 假设 P 是一个列向量
So if you do one column vector- for example, imagine P is a column vector-
然后你计算 P乘以 P的转置
and you do P multiplied by PT,
你会得到这两个向量所有可能的乘积组合,
you obtain all the possible combinations of products of these two vectors,
因为这将是一个.
because this will be one.
我可以举一个简单的例子.
I can do a simple case.
所以 P11, P1, 我们称它为 P2, P3.
so P11, P1, let's call it P2, P3.
乘以行向量p1, p2, p3.
multiplied by the row vector, pl, p2, p3.
这将生成p1与p之间所有可能的乘积组合,
this will generate all the possible uh combinations of products between pl and the p.
因为这将是一个3乘1的矩阵.
the first vector and the second vector, because this will be a 3 by 1.
这是一个1乘3的矩阵, 因此 会生成一个3 乘3的矩阵, 它将等于p1乘以p1, p1乘以 p2, p1乘以 p3
this is 1 by 3, so it will be generated 3 by 3 vector and it will be equal to p1, p1, p1,
等等.
p2, pl, p2, pl, p3, etc.
等等, 依此类推.
etc, etc.
此外, 我们可以看到在矩阵的对角线上有额外的项,
moreover, we can see that in the diagonal of the matrix we have this additional term,
第一个对角线元素中是p1,
this additional term p1, in the first diagonal element, p1,
第二个对角线元素中是p12, 第三个对角线元素中是p13.
p12 in the second diagonal element, p13 in the third diagonal element.
我实际上称它为p1.
i actually call it pl.
这是错误的, 因为我应该称它为pi.
it's wrong because i should call it pi.
这就是为什么我不想引入i 这个索引.
that's why i didn't want to bring the i indices.
所以实际上不应该是p1, 而应该是pi, pi, pi,
so it's not really p1, it should be pi, Pl, Pl, Pl,
因为我们是在为通用的第i个pi向量进行这个操作.
because we are doing it for the generic it h Pl vector.
-Pw Pr2
Pw(-PN
让我来修正一下索引.
So let me fix the indices.
-Pw P2
Pw(-Pi N
Ping, P13 P,
" Pi N, Pi3.
这是一个
This is one.
2
-Pw P2
Pw(i-Pi N
PI和 Pl.
PI and Pl.
好的, 所以这是 PI, Pl.
Okay, so this is Pl, Pl.
p i, p i, p i, p i, p i, p i.
pi, p i, pi, pi, pi, p i.
好的, 我们可以得到, 嗯, 所以我们也可以将这个雅可比矩阵写成
okay, we can obtain, uh, so we can write this, um,
一个对角矩阵,
this jacobian here also as the diagonal matrix,
其对角线上的元素都是p 的元素?
that in the diagonal has all the element of the p?
i向量减去p向量, 再乘以其自身的转置.
i vector minus the p vector multiplied by the transpose of itself.
所以与其自身相乘, 但需要转置.
So with itself but transposed.
因为我们需要所有元素都是 P 的一个元素
Because we need all the elements to be kind of a combination of one element of p
与另一个元素的某种组合,
with another element of p,
而在对角线上, 我们还需要一些额外的项,
plus, only on the diagonal we need some of this additional term,
即p 的元素,
which are the elements of p
所有这些p的输出元素与p的转置相乘后都会被取负.
and all the elements of the output of this p multiplied by p transposed are negated.
这就是为什么我们需要这个减号.
that's why we need this minus sign.
所 以如果你看一下 Flash Attention 的论文, 他们会在这里给你这个公式
So if you look at the flash attention paper, they give you this formula here.
他们说如果 Y 等于×的soft max,
They say that if Y is equal to the soft max of x,
那么雅可比矩阵将如下所示, 会是一个对角矩阵.
then the Jacobian will look like the following, will be diagonal.
y的对角矩阵减去y 乘以y 的转置, 其中y 是一个列向量. 好了, 各位
of y minus y y transposed where y is um the is a column vector All right, guys,
我知道这已经很长了, 所以我们先暂停一下, 现在终于要开始写代码了
I know this has been long, so let's take a pause and we are going to now code finally.
首先, 让我们来验证一下 Flash Attention 反向传播的数学推导.
First of all, let's check the mathematics of the backward pass of a flash attention.
我们会简要地看一下.
We will see it briefly.
我不会再做任何推导, 但我会解释一下.
I will not do any more derivation, but I will explain it.
然后我们最终转向编写代码.
And then we finally switch to coding it.
那么, 我们开始吧.
So let's go.
B. 2
Memory-efficient backward pass We derive the backward pass of attention and show that it can also be computed with linear memory. Rabe
and Staats 66|suggests that the backward pass can be done without quadratic extra memory by applying gradient checkpointing to the memory-efficient forward pass. We instead derive the backward pass explicitly and show how it can be computed in a memory-efficient manner.
(where d O denotes that there
B. 2
Memory-efficient backward 终于可以看看 Flash Attention. 的反向传播 好了,
and Staat s66]sug
at the backward pass where d O denotes
B. 2
2 Memory-efficient backward pass We derive the backward pass of attention and show that it can also be computed with linear memory. Rabe
and Staats66|suggests that the backward pass can be done without quadratic extra memory by applying gradient checkpointing to the memory-efficient forward pass. We instead derive the backward pass cx plicitly and show how it can be computed in a memory-efficient manner.
(where d O denotes
B. 2
and Staats66|sugggs
sts that the b
extra memory by applying where do denotes
B. 2
2 Memory-efficient backward pass We derive the backward pass of attention and show that it can also be computed with linear memory. Rabe
and Staats 66|suggests that the backward pass can be done without quadratic extra memory by applying gradient checkpointing to the memory-efficient forward pass. We instead derive the backward pass explicitly and show how it can be computed in a memory-efficient manner.
(where do denotes
B. 2
Memory-efficient backward pass backward pass emory by applying And if you i look at the appendix ot the flash attention paper, y
where do denotes
From Eq.(2). we have that d P= d Ov T, and so : 你会看到 B. 2部分, 1
you will'see this part B. 2 where they. derive the backward pass. step. by step.
O=f V
(in matrix notation )d V= P"d O. Thus: 现在, 我不想一步步推导整个过程, 因为那会太完长
dr=dov
From Eq.(2). we have that d P =d O v T, and so :
gotm rough Thee radio mts
但我想提供所有必要的工具, 养 帮助你们理解它
but'r want. to give you all the tools neces OR abe
y=xw and show how it can be computed in a memory-efficient manner.
B. 2
Memory-efficient backward pass We derive the backward pass of attention and show that it can also be computed with linear memory. Rabe
and Staats66|suggests that the backward pass can be done without quadratic extra memory by applying gradient checkpointing to the memory-efficient forward pass. We instead derive the backward pass explicitly y =xw and show how it can be computed in a memory-efficient manner.
B. 2
Weder Rl
ory. Rabe
y=xw
and show how it can be computed in a memory-efficient manner.
B. 2
Memory-efficient backward pass We derive the backward pass of attention and show that it can also be computed with linear memory. Rabe
and Staats 66|suggests that the backward pass can be done without quadratic extra memory by applying gradient checkpointing to the memory-efficient forward pass. We instead derive the backward pass explicitly y =xw and show how it can be computed in a memory-efficient manner.
B. 2
We derive the Mar memory. Rabe
and Staats|66|sugges
amemory by applying y =xw and show how it can be computed in a memory-efficient manner.
B. 2
Memory-efficient backward pass We derive the backward pass of attention and show that it can also be computed with linear memory. Rabe
and Staats66|suggests that the backward pass can be done without quadratic extra memory by applying gradient checkpointing to the memory-efficient forward pass. We instead derive the backward pass explicitly y =xw and show how it can be computed in a memory-efficient manner.
B. 2
Memory-efficient back wag 需要复习
We derive the backward pa compu ed with linear memory. Rabe
y=xw and show how it can be computed in a memory-efficient manner.
B. 2
Memory-efficient backward pass We derive the backward pass of attention and show that it can also be computed with linear memory. Rabe
and Staats66|suggests that the backward pass can be done without quadratic extra memory by applying gradient checkpointing to the memory-efficient forward pass. We instead derive the backward pass explicitly y =xw and show how it can be computed in a memory-efficient manner.
B. 2
Memory-efficient baskwa 个矩阵的命名
and Staats 66|suggests We derive the backward pass of at ter M or P ai M computed with linear memory. Ra be
adati extra memory by applying the'
y=xw and show how it can be computed in a memory-efficient manner.
B. 2
Memory-efficient backward pass We derive the backward pass of attention and show that it can also be computed with linear memory. Rabe
and Staats 66|suggests that the backward pass can be done without quadratic extra memory by applying gradient checkpointing to the memory-efficient forward pass. We instead derive the backward pass explicitly y =xw and show how it can be computed in a memory-efficient manner.
Therefore the forward pass can, be 正如你所知, 在前向注意力机制中, 在前向传播过程中
Compute As you know, in the forward attention, in the forward pass,
extra memory 2. Compute o; for alli according to Eq.(2), which takes O(d) extra memory.
vj. Therefore the forward pass can be computed with O(n) extra memory :
2. Compute L;for all i according to Eq.(1), which takes O(n) extra memory.
3. Compute o;for alli according to Eq.(2), which takes O(d)extra memory.
Therefore. the forward pass can computed with O(n)
extra memory 我们将查询
(query ) 与键
(key ) 的转置相乘
we do the query multiplied by the transpose of the key which takes O(n)extra memory 2. Compute o;for all i according to Eq.(2), which takes O(d)extra memory.
v. Therefore t be computed with O(n)extra memory : 这个输出的结果我们称之为 S
4. Compute L;for all according to Eq.
which takes O(n)extra memory.
and the output of this we call it S.
5. Compute o;for alliaccording to Eq.(2), which takes O(d)extra memory.
Therefore the. forward pass can be computed with O(n)extra memory 然后我们对这个
S 矩阵应用
Soft max函数 它就变成了
P矩阵.
Then we apply the soft max to this S matrix and it becomes the P matrix.
Compute
6. Compute o; for all i according to Eq.(2), which takes O(d) extra memory.
vj. Therefore the forward pass can be computed with O(n)extra memory :
Softmax
7. Compute L;foralliaccording to Eq.(1);which takes O(n) extra memory.
The soft max is applied by rows.
8. Compute o;for alli according to Eq.(2), which takes O(d)extra memory.
forward be computed wi th O(n)
extra memory : 与√矩阵相乘 从而得到注意力的输出
9. Compute for
al Tiaccording
whichitakes O(n)extramemory.
and we multiply by a V matrix to obtain the output of the attention.
10. Compute o; foral i according to Eq.(2), which takes O(d) extra memory.
Li
vj. Therefore the forward pass can be computed with O(n)extra memory :
11. Compute L;for all i according to Eq. (1), which takes O(n) extra memory.
12. Compute o;for all i according to Eq.(2), which takes O(d)extra memory.
Compute o;for alli according to Eq.(2), which takes O(d)extra memory. 让我们看看, 例如,
Memory-efficient backward pas Let's look at, for example, how the computation le rive the backward pass of attention and show that it can also be computed with linear memory.
B. 2
of the ith. row of. theoutput is. computed, based. on, the. P matrix. and. the, V matrix Rabe gradient checkpointing to the memory-efficient forward pass. We instead derive the backward pass explicitly
B. 2
ndshow that i gradient checkpointing to the memory-efficient forward pass. We instead derive the backward pass explicitly
B. 2
Memory-efficient backward pass We derive the backward pass of attention and show that it can also be computed with linear memory. Rabe
and Staats|66|suggests that the backward pass can be done without quadratic extra memory by applying gradient checkpointing to the memory-efficient forward pass. We instead derive the backward pass explicitly
B. 2
We derive the backward pass of attention and show that it can also be computed with linear memory. Rabe gradient checkpointing to the memory-efficient forward pass. We instead derive the backward pass explicitly
B. 2 按照惯例
because. when we write in, linear algebra, whenever we. write the. name of aye cto We derive. the backward gradient checkpointing to the memory-efficient forward pass. We instead derive the backward pass explicitly
B. 2
Memory-efficient backward pass We derive the backward pass of attention and show that it can also be computed with linear memory. Rabe
and Staats66|suggests that the backward pass can be done without quadratic extra memory by applying gradient checkpointing to the memory-efficient forward pass. We instead derive the backward pass explicitly
B. 2
We derive the backwar
Dass fat tent ipn and show that it canals q be computed with linear memory. Rabe gradient checkpointing to the memory-efficient forward pass. We instead derive the backward pass explicitly
B. 2
Memory-efficient backward pass We derive the backward pass of attention and show that it can also be computed with linear memory. Rabe
and Staats66|suggests that the backward pass can be done without quadratic extra memory by applying gradient checkpointing to the memory-efficient forward pass. We instead derive the backward pass explicitly
B. 2
M但这个特定向量的来源实际上是输出矩阵的一行.
But the. origin. of. this particular y ector is actually i a row. of the. output matrix. g We derive the back way nand show that it can als
obe computed Babe gradient checkpointing to the memory-efficient forward pass. We instead derive the backward pass explicitly
B. 2
Memory-efficient backward pass We derive the backward pass of attention and show that it can also be computed with linear memory. Rabe
and Staats|66|suggests that the backward pass can be done without quadratic extra memory by applying gradient checkpointing to the memory-efficient forward pass. We instead derive the backward pass explicitly
B. 2
Let's. try to. understand what is the output row. of a matrix in a matrix multiplication.
We derive the backward pas
ionand show that it cay
Ruted with ear Rab gradient checkpointing to the memory-efficient forward pass. We instead derive the backward pass explicitly
mory-efficient backward pass backward pass of attention and show that it can also be computed with linear memory. Rabe
66|suggests that the backward pass can be done without quadratic extra memory by applying ck pointing to the memory-efficient forward pass. We instead derive the backward pass explicitly
ng to Eq. (1), which takes O(n) extra memory. 现在hi c这样我们就能理解如何从这里到这里了.
ng to Eq.
Now, so that we can understand how to go from here to here.
pass
ng to Eq.(1), which takes O(n) extra memory.
ng to Eq. (2), which takes O(d) extra memory.
ng to Eq.(1), which takes O(n) extra memory. 比如一个 A 矩阵.
So, let's write a generic matrix multiplication, for example, an A matrix.
nass
but ed with O(n) extra memory : 假设它如下所示.
v hich takes O(n) ere trg say that it. is the following.
but ed with O(n) extra memory :
which takes O(n) extra memory.
but ed with O(n) extra memory : 我们只写一行.
v hich takes O(n) ext and we only r write one row.
but ed with O(n) extra memory :
v hich takes O(n) extra memory.
but ed with O(n) extra memory : 实际上让我再放大一下, 我想写小一点, 这样我们就有足够的空间
At tually let me zoom again and l want to write smaller so we have enough space.
th out extra memory by repeatedly sum n
所以我们做一个有一行的矩阵.
th out extras q we make am atrbthathasprowtedly
sumn
th out extra memory by repeatedly sum n
让我们称它为a1, a2, a3.
thout extra
let'ecall it ron e, atwoyathreeatedly
sumn
th out extra memory by repeatedly sum n
然后我们进行乘法运算.
th out extra me and then we lmyltiplpeatedly
sumn
th out extra memory by repeatedly sum n
这将是一个具有多行的矩阵, 就像这个一样,
hou t exthis will be r matrix with many rows like this one l y
sumn
因为我们只想研究一行的效果, 并将其与另一个矩阵相乘.
because we want to study the effect only of one row and we mu ltiply it by another matrix.
让我们称这个矩阵为 A, 它有, 比方说
h Out let'scal it this one is the matrix a and it has i don't know-SUmm
n行3列.
thout extra let'seayncwsby three polumestedly
sumn
然后我们应该有另一个矩阵 B,
hout extr then we should have another matrix bed ly
summ
th out extra memory by repeatedly sum n
它有3 列和一定数量的行,
with three. columns and some number of three rows and some number gf columns,
比方说4列.
thout extra meetseyfoulcolumns peated ly
sumn
th out extra memory by repeatedly sum n
因此, 我们称第一行为, 让我再放大一点, 所以它是 B11, B12, B13, B14
So we calt the first row, let'scall it let me zoom ymore. soit's B1l, B12s B13: B14.
然后这个应该是 B21, B22, B23, B24, 这个应该是 B31, B32, B33, B34
then this one should be B21: B22 B23, B24y This should be B3l1. B3. 2 B3:3: B3:
l out extra memory by repeatedly sum mi
我知道我的符号表示不够严谨.
l out extr akno re am not very rigorous ipmaotatily
summi
l out extra memory by repeatedly sum ml
我应该用大写字母 A
l Out should have all ed a U these d lements with the capital letter mmi
l out extra memory by repeatedly sum ml
和大写字母 B来称呼所有这些元素.
lout extra men nd the capital letepeatedly
summl
这就是你在引用矩阵的单个元素时使用的符号.
So this is the notation that you use when refering to a singl item-of a matri rxr
out extra memory by repeatedly sum ml
但请原谅我这一点.
l out extra me rtp las efogive meptatedly
summl
l out extra memory by repeatedly sum ml
这个矩阵乘法的输出将是另一个n×4的矩阵,
The output of this matrix multiplication wil be another matrix that is n byi
l out extra memory by repeatedly sum ml
也就是说, 输出的每一行都会有4列.
O totit wilbenby 4ewe wil have 4 columns for each row of ths out pr tni
l out extra memory by repeatedly sum ml
我想用另一种方式来表达输出.
lou t extrd want tm rite the output in ederantwayy
summl
l out extra memory by repeatedly sum ml
因此, 我想这样写, 仅作为一个向量, 将第一行输出作为一个向量
So J want te write it as folle ws asa yector only, So the first output row a sa yector.
并想了解这个向量的每个维度是什么.
l Out and want to rnd ert and what is each di pension ef this vector m mi
l out extra memory by repeatedly sum ml
因为否则我在这里没有足够的空间来书写.
l Out Se because other wi sf I don't have enough space to writ e itherin mi
那么首先, 让我们把它写下来.
So the first, let's write it.
no
让我们称之为口.
So let's call it O.
我想写下○的第一个元素, 即输出的第一行,
I want to write what is O of one, which is the first row of the output,
ut extra mer但以列向量的形式呈现tedly summing but written as a column vector.
extra memor
ut extra memory by repeatedly summing Era
ut extra n所以o的第b个元素将在这里lly summing So O of one will be here.
ey tra.
我们应该用小写字母. 表示第一个元素, 它应该是一个向量,
I ) e X we should use the small letter O of one should be a vector
th out extra memory by repeatedly sum r 其中第一个维度是这里这些内容的点积.
n
Ewhere the first dimension is the dot product of this stuff here.
即 A矩阵的第一行与 B矩阵的第一列的点积.
n)s
sothe first row of the A matrix with the first column of the B matrix.
extra memory :
th out extra memory by repeatedly sumr 那么第一个维度就是 A1与 B11的点积.
n)
xt r So the fiesta let'ssay, dimension will be A1 with B11.
n)
extra memory :
我也应该把这个称为 A11, 实际上是 A12.
n)
EXtra Ishould also call this one Ai1, A12 actually.
hout 我也应该把这个标为 X1hy实除 P是atedly
sun
I should also call this one A11, A12 actually.
extra memory :
Oi without extra memory by repeatedly
以及 A13, 即 A13.
Oi
without IC and A one three so A one th rey!
repeatedly
Oi without extra memory 1by repeatedly
由于 A 矩阵有多行, 让我使用正确的命名方式
Because we have many rows in the A Matri so let me use the correct ham ing.
Oi without extra memory 1by repeatedly
因此, 这将是 A11与 B11的乘积, A11乘以 B11加上 A12乘以 B21
Oso this l Win be A one, on et with B bneone, Aone, oneyb ohe, bne plus A one y
vithoutextra再加kr Al3r乘以5 B3repeatedly sun two multiplied by B two, one plus'A one, three with B three, one,
vith out extra memory by repeatedly sun
vithout这将是输出短阵第v行的第rep维度tedlysun
And this will be the first dimension of the first row of the output matrix O.
vith out extra memory by repeatedly sun
vith out extra memory by repeatedly su m 输出矩阵○第一行的第二个维度将是
( n The second dimension of the first row of the output matrix O will be
vith out extra memory by repeatedly sum
0(n)
extra memory :
vith out extra memory by repeatedly sum A矩阵的这一行与 B矩阵第二列的点积.
the dot product of this row of the A matrix with the second column of the B matrix.
vith out extra memory by repeatedly sum
0(n)
extra memory :
vith out extra memory by repeatedly sum 让我在这里写上 B.
0(n)
extra merndlet me write here B.
vith out extra memory by repeatedly sum
0(n)
) extra memory :
vith out extra memory by repeatedly sum 因此, 它将是 A11乘以 B12加上 A12乘以 B22再加上 A13乘以 B32.
( n) ex t So it will be A1b B12 plus A12, B22 plus a13, b32.
vith out extra memory by repeatedly sum O(n) extra memory :
vith out extra memory by repeatedly sum 第三个维度将是 A11乘以 B13加上 A12乘以 B23再加上 A13乘以 B33
( n the third dimension will be a11, b13 plus a12, b23 plus a13, b33.
vith out extra memory by repeatedly sum 第四个维度将是 A11乘以 B14加上 A12乘以 B24
the fourth dimension will be all, uh, b, one, four plus a one, two, b, two,
D(n) extra memory :
再加上 A13乘以 B34.
D(n) extra memory :
现在, 这是输出矩阵○的第一行输出,
0(n)
Cnow this is theo ntput, the first output row of the o matrix,
称为向量01, 其中:这是该向量的第一个维度,
and it's a vector called bone, and these are : this is the first dimension of this vector,
这是第二个, 这是第三个, 而这是第四个维度,
这里的每一项都是一个标量.
D(n) extra memory :
a
memo因此, 输出o1, 即输出矩阵的第一行,
So the output O1, which is the first row of the output matrix,
a memory : 也可以表示为第一个元素.
can also be written as the first element.
如你所见0它是多个向量的和, 其中第一个元素是 A11乘以.
as you can see, is a sum of many vectors where the first element is A 11 multiplied.
ame 让我用一个更小的, 这个, 但我想用一个更小的.
let me use a smaller, this one, but I want to use a smaller.
a memory : 我无法改变这里的大小.
I can't change the size here.
a memory : 好的, 没关系.
Okay, it doesn't matter.
am所以如你所见, 这里 A1每次乘以一个不同的 B数.
So as you can see here, there is A 1 multiplying a different B number every time.
a mem0r所以这是 B11, B12, B13, B14.
So this is B11, B12, B13, B14.
a mem0ry B11、 B12、 B13、 B14 是什么?
What is B11, B12, B13, B14?
a memory : 这是 B矩阵的第一行.
It is the first row of the B matrix.
amemor所以它等于 B1和第一行的所有维度.
So it is equal to B1 and all the dimensions of the first row.
Then plus, then we have the element A12, multiplied by b21, b22, b23, etc.
memory : 这是 B矩阵的第二行.
and this is the second row of the B matrix.
so we use the tensor notation of Py Torch to describe this row,
memory : 即b2及其所有维度.
which is b2 and all the dimensions of b2.
So this is a vector scalar product and plus A13 multiplied by B3
memory : 及其所有维度.
and all the dimensions of B3.
这个式子也可以写成对所有可能的i This one can also be written as the summation over all possible i
从 1到3的求和,
that go from one to three,
其中 1到3表示 A矩阵中有多少列, 即 Aij, 嗯, A1.
where one to three is how many columns there are in the A matrix, of A ij, well, A1.
by repeatedly summing
让我们把这个j称为j, 抱歉.
Let's call this one j, actually, sorry.
by repeatedly s summing
by repeatedly 让我们称客湘j Let's call it j.
by repeatedly summing
equal to one, and let's call this the generic i th row of the output matrix, will be a?
y by repeatedly 1 summing
cy by repeatedly Si2m有ng
i one, a i two and a?
y by repeatedly 1 summing
by rep is a每个都莱以br矩阵的相应行,
i three, each one multiplied by the corresponding row in the b matrix,
y by repeatedly 1 summing
所以我们可以写成a ii 乘以b, 其中bj是b的行.
Ca t sowe eah waite ithsa in multi pluie d by b, where bj is the row of b.
by repeatedly 1 summing
or y by repeatedly summing 我们也可以这样写, 以表明这是一个向量.
We can also write it like this to indicate that this is a vector.
mory by repeatedly summing 这正是他们在这里所做的.
And this is exactly what they do here.
vith O(n) extra memory : 这正是他们在这里所做的.
akes O(n)extra merr And this is exactly what they do here.
akes O(
nemory
according to Eq.(1), which takes O(n) extra memory.
ccording to Eq. (2), which takes O(d) extra memory.
memory ccording to Eq. (2), which talostheutpturematri &ry.
according to Eq. (1), which takes O(n) extra memory.
ccording to Eq.(2), which takes O(d) extra memory.
当我们进行乘法运算, 即 P 乘以 V时, 输出矩阵的第i行,
htakes when'we do the multiplication k, p Multiplied by v, the ith row of the output matrix,
or all i according to Eq. (1), which takes O(n) extra memory.
or all i according to Eq.(2), which takes O(d) extra memory.
for all i according to Eq.(1), which takes O(n) extra memory. 我们称之为 它是一个向量, 但在表示上是 个列向量
or afl i according to ake s
O(extra memory we call it Ol, which is a vector, but by notation it is a'column vector,
or all i according to Eq.(1), which takes O(n) extra memory.
or all i according which takes Otd )extra memory where the elements of this column vector are actually the elements of the ith row of o.
for all i according to Eq. (1), which takes O(n) extra memory.
or all i according to Eq.(2), which takes O(d) extra memory.
for all i according to Eq. (1), which takes O(n) extra memory. 这只是表示方式, 各位 等于 P的第i行.
this is only'by notation, guys - is equal to the ith row of P.
for all i according to Eq. (1), which takes O(n) extra memory. 因此,
or all i according to Eq.
so the ith row of the matrix that is on the left in the matrix,
or all i according to Eq. (1), which takes O(n) extra memory. 乘以. V 矩阵的所有列
or all i according. to Eq.(2), which multiplication multiplied by all the columns of the V matrix,
or all i according to Eq. (1), which takes O(n) extra memory.
or all i according to Eq.(2), which takes O(d) extra memory.
for all i according to Eq.(1), which takes O(n) extra memory. 的第i 行所有元素的和
or all i according to Eq.(2
whichtakes O(aext
cramemor v which can also be written as the summation over all the elements of the ith row of p.
or all i according to Eq. (1), which takes O(n) extra memory. 即左边矩阵的第1行的所有元素
oralliaccordingto
Eq.
2.'which takes Ode x so all the elements of the ith row of the first matrix, the one on the left in the matrix,
to Eq.(1), which takes
to Eq. (2), vhultiplication'multiplied by'each vector in the V matrix
ordingto Ea其中v Vi中的第j(个矩阵是 V矩阵的每一行.
ording to Ewhere the jth:matrizx hereiny iseach, row of the V matrix.
e the forward pass can be computed with O(n) extra memory :
for all i according to Eq. (1), which takes O(n) extra memory.
for all i according to Eq.(2), which takes O(d) extra memory.
for all i according for all i according t An l pij ean'&liso be written as y pij is'what?
for all i according to Eq. (1
which takes ( 是soft max 的输出
for all i according to Eq. 3thebiutputof the'softhaxnemory.
for all i according to Eq. (1), which takes O(n) extra memory.
for all i according to Eq.(2), which takes O(d) extra memory.
extra memory.
for all i according to Eq. (2), whg6hasyo@ Rh6wextra memory.
for all i according to Eq.
soft max 的输出是输入完素的指数函数.
the output of the'soft maxi se to the pb wer of the element input of the soft max.
for all i according to Eq. e(1), whicht2
for all i according what is the ele hent input of the'soft max?
for all i according to 是查询与键的转置相乘的结巢?
emory.
for all i accorqs'the qdery Multiplied by the transpose 6f the keys.
for all i according 齿此, 它是个询与个键的点
for all i accor sort'sa Hot produkt between one query and one key.
for all i according to Eq. (1), which takes O(n) extra memory.
for all i according to Eq.(2), which takes O(d) extra memory.
for all iac cord 这就是为什么崔指数菌数中会有这些内容.
for all i ac And that's Why you'ha ive this'stuff here in ther exponential.
for all i according to Eq. (1), which takes O(n) extra memory.
for all i according to Eq.(2), which takes O(d) extra memory.
put e o;for all i according to Eq.(2), which takes O(d) extra memory.
I emory-efficient 所以这是理解这个推导过程的第一步.
66|suggcsts that the backward pass can be done without quadratic extra memory by applying
B. 2
Memory-efficient backward pass and Staats66l suggests that the backward pass can be done without quadratic extra memory by applying We derive the backward pass of attention and show that it can also be computed with linear memory. Rabe gradient checkpointing to the memory-efficient forward pass. We instead derive the backward pass explicitly and show how it can be computed in a memory-e ficient manner.
B. 2
Memory-efficient hack ward 我们还学习
tew h linear memory. Rabe gradient checkpointing to the men i or ye H i cient and show how it can be computed in a memory-e ficient manner.
Dack ward pass explicitly
respectively ).
(in matri so far is how to derive the backward path of the ;matrix multiplication and of the softmax.
-doi.
(3)
d P =do vj Recall that P := soft max( S:). Using the fact that the Jacobian of y = soft max(x) is(diag (y)-yy
we have that d S:=(diag( P)-PP)d P:= Pod Pi
-( Pd P) Pi:
NX1
Nx1
d P =do vj we have that d S:=(diag( B)
( Pd P) Pi:
NX1
d P =do vj Recall that P := soft max( S:). Using the fact that the Jacobian of y = soft max(x) is(diag (y)-yy
we have that d S :=(diag ( P)-PPT)d P:= Pod P:-( Pd P) P
NX1
Nx1
NX1
Recall that P =soft max( S:). Using the fact Jacobian of softmax(x)is(diag(y)-yy
we
havth在矩阵乘法中 让我们回顾一
In the matrix multiplication, let's rehearse'the'formula.
dig (e:)d R:
aru fied
O= PV
O= PV 假设给定一个矩阵乘法, 即y等于×乘以w, 我们知道,
So if, given a Mmatrik multiplication that is y equal to x multiplied by w, We know that,
O= PV
O= PV 给定损失函数相对于y(即该操作的输出)的梯度,
Fgiven the gradient of the loss function with respect to y,
O= PV 我们可以
so the output of this operation,
O= Pv 推导出损失相对于
We derive the gradient of the loss with respect to one of the input
O= Pv 该函数的一个输入
of this function,
O= Pv
(x或w)的梯度.
which is the x or w.
O= Pv
O= PV 为了得到相对于×的梯度, 我们需要取上游梯度
To get the gradient with respect to x, we need to take the upstream gradient,
O= PV
(即相对于输出的梯度)乘以 W的转置.
ient with respect to the output, multiplied by the transpose of wt.
O= Pv
O= Pv 为了得到相对于 W的梯度, 我们需要将输入的转置
And to get the gradient with respect to w, we need to do the xt,
O= Pv
O= Pv 乘以上游梯度.
the
e input transposed multiplied by the upstream gradient.
O= Pv
O= Pv 这个公式我们没有推导,
this one is the formula that we didn't derive,
O= PV 但它们的推导过程完全相同.
but how to derive them is exactly the same procedure.
O= Pv
Since we already comput
在注意力机制中,
Sip rt tent ioe already computed
我们最后的乘积操作是adr等手p乘以yuted Li, d
P
are
Since we already computed Li, d
The gradients d Q and d K are
小在反向传播过程中
what Py Torch will give us as input during the backward pass The gradients d Q and dk are
损失相对于输出的梯度
computed Li, d
is the gradient of the loss with respect'to the output.
Since we already computed Li, d
The gradients d Q and d K are
And
Since we already computed Li, d
The gradients d Q and d K are
来推导出损失相对于!
Y computed Li, d
to derive the gradient of the loss with respect to Q,
Since we already computed Li, d
The gradients d Q and d K are
The gradients d Q and d K are
这样它就可以在反向传播过程中
lite d Li. d
The gradients d Q and dk are
Since we already computed Li, d
The gradients d Q and d K are
被计算图中的前序操作符使用
Om puted Li, d
in the computation graph, in the operations before.
Since we already computed Li, d
The gradients d Q and d K are
好的, 但为了得到相对于
guer y..
key 和value 的梯度
_我们需要先推导出每个中间操作的梯度
Bi it ed.
d
Since we already computed Li, d
The gradients d Q and d K are
Li, d
Ihe gradients d Q
Land dk are
Since we already computed Li, d
The gradients d Q and d K are
因此, 在已知损失相对于◎的梯度的情况
Li. d
So the gradient with respect to, of the loss with respect to v The gradient s d Q
2anddk
are
From Eq. (2), we have that d P =
From Ea._(2)..
. we, have that d P = 损失相对于√的梯度的计算方式,
given the gradient of the loss with respect to o,
From Eq. (2), we have that d P =
From Eg. (2)..
. we have that d P 与矩阵乘法中损失相对于×的梯度的计算方式
it is exactly like computing the gradient of the loss with respect to x
-与矩阵乘法中损失相对于axr的梯度的计算方式d Li, d
df=dov
Since we already computed Li, d
The gradients d Q and d K are
Sinc 完金相圆ready computed Li, d
f ne
Since we already computed Li, d
The gradients d Q and d K are
我们知道官等于ept by computed Li, d
P =dov
我们知道官等于ept ly computed Li, d
And we knp w that it is equal to pt d Q and d K are
The gradient s d Q and d K are
From Eq. (2), we have that d P =
所以, 各位e这景是类比. d Q and d K are
d=d0 v
Pjn by analogy guv we have that d P =
大家应该能理解这里的类比关系 Qandd Kare
dl= do vt
The gradient s d Q and d K are
From Eq. (2), we have that d P =
也就是左侧矩阵的转置乘以t 游梯度. and d K are
The gradient s d Q and d K are
From Eq. (2), we have that d P =
在论文中 T l他们是这样写的d Q and d K are
因此, dv等于pt乘以dogr这就是你在这里看到的公式rea
From Eg.(2),
Since we already computed Li, dv ; can The gradients d Q and d K are a little From Eq. (2), we have that d P = d Ov T
另一个推导是如何求出相对于dp 的梯度.
The other derivation is how to derive the gradient with respect to dp
V
Recall that Pi : = soft max( S:). Using the fact that the J
havethat
Nx1
Is ing the fact that the J
P
Nx1
左侧矩阵的梯度.
d Pi j = do vj that is on the left side of the matrix multiplication.
Recall that soft max c Using the fact t that the
d Pi j = do vj Recall that Using the fact that t
the
因此, 这就像在参考公式中推导损失相对于×的梯度
So it is just like deriving the gradient of the loss with respect to x do vj Recall that soft max( S.).
Using the fact that the
From Eq. (2), we have that d P = d Ov T, and so:
d Pij = do vj
Recall that Using the fact that f the
一样.
d Pi j = do vj in the reference formulas.
Recall that soft max (
Using the fact that the
d Pi j = do vj Recall that Using the fact that f the
它等于上游梯度乘以另一个矩阵的转置,
which is equal to the upstream gradient multiplied by the transpose of the other matrix,
do Recall that =soft max Using the fact that th
d Pi j = do vj Recall that Using the fact that f the
在论文的符号表示中, 他们将其写为dp, 等于d?
which in the notation of the paper they write it as dp, is equal to d?
Recall that soft max Using the fact that t
○乘以v的转置.
d Pij=dovj
O multiplied by v transposed.
Recall that so tt max Using the fact that the
From Eq. (2), we have that d P = d Ov T, and so:
d Pij = do vj
Recall that soft max Using the fact that t
the
就是这个公式.
d Pi j = do vj and it's this formula here.
Recall that l =soft max( S.
Using the fact that t
the
d Pi j = do vj Recall that Using the fact t the
d Pi j= dovj.
call that P=sa
that how they compute this stuff here is exactly as above.
respec hi y ely )
so, as this derivation here they call yj the j th row of the v matrix im matrix notation dvj
we have that d P = d Ov T, and so:
d Pij = do T vj.
we have that d P = d Ov T, and so: 并将其写为pij乘以do.
and they write it as pij multiplied by'do.
d Pi = do vj Recall that P := soft max( S:). Using the fact that the Jacobian of y = soft max(x)
ave that
如何得出这个公式呢?
Recall that Pi : = soft r how to arrive to this formula here? obian of y = softmax(x)
avethat
d Pi j=dovj Recall that P := soft max( S:). Using the fact that the Jacobian of y = soft max(x)
ave that
好的, 我们开始推导吧.
Recall that P: = softmax( S:). Wellgletlsflotithat the Jacobian of y = soft max (
ave that
d Pi i =do
让我写下来.
so let me write.
From Eq. (2), we have that d P = d Ov T, and
From 好的, 2理论上我们可以人这个推导中得知ndso:
okay, theoretically we know that from this derivation here.
d Pi j =
ack ward pass of attention and show that it can also be computed with linear memory. Rabe suggests that the backward pass can be done without quadratic extra memory by applying the backward pass explicitly it can be computed in a memory-efficient manner to compute the input gradients d Q, d K. d V
ack ward pass of attention and show that it can also be computed with linear memory. Rabe suggests that the backward pass can be done without quadratic extra memory by applying so in ting to theme mg it can be computed i name m
t the we know that. the. i through. rho, of. the. output in a. matrix multiplication.
efficient, manner.
ao:ak:ay
Pij do ;=
dvi Li
do;
mahd Vaccumu
Qecaweweconue
Since we already computed L, dv;can be computed without extra memory by repeated s un
d P=dov The gradients d Q and d K are a little more complicated. We go through the gradients d D
首先我们简化一下操作:"每次着到转置符号
Recall that soft max(x)is(
如果你不喜欢在矩阵乘法中处理转置,
and you don't like work with the transpose in a matrix multiplication,
we already computed Li, dvi can be "cor
不妨给它换个名字, 用新名字进行推导,
just give it a different name and then work with the different name and after,
"we already computed Li, dv can be cor
we already computed Li, dv ; can be cor
等公式得出后, 再把转置操作代回去.
when you have deri yed the formula, you re substitute the transpose operation.
we already computed Li, dv can be cor
we already computed Li, dv ; can be cor
在这个例子中, 我们计算的是:dv等于p的转置乘以d?
in this case we are doing : dv is equal to p transpose multiplied by d?
we already computed av can Be cor
we already computed Li, dv ; can be cor
我们把 P 的转置称为.
we already Let's call P transpose.
computed Li,
dvi can be cor
我们给它起一个之前没用过的名字.
we already Let's give it a name that we didn't use so far.
computed Li, dv can be cor
那就叫它 F 吧.
So let's call it F.
we already computed " Li,
. dvi can be cor
we already computed Li, dv ; can be cor
有空的时候, 我总是用 F.
we already I always use F when it's available.
I computed Li, dv can be cor
于是我们称之为 DV.
So we call DV.
we already computed Li,
. dvi can be cor
we already computed Li, dv ; can i be cor
等于f 和 do 的乘积.
we already complite d
is egual to f, d o.
. dvi can
be cor
we already computed Li, dv ; can be cor
从上面的推导可知,
we know from above here, from this derivation here, or this derivation here is equivalent -
we already computed can be cor
d=dovt
ndd K_are a little more complicated From Eq.
, and so :
we know from above here, from this derivation here, or this derivation here is equivalent -
d Pi j=do
Since we already _ computed Li, dv ; can be computed without we know from above here, from (this derivation here, or this derivation :here i s eguivalent -
or y-efficient backward pass ack ward pass of attention and show that it can also be computed with linear suggests that the backward pass can be done without quadratic extra mem
he gradients d Q and dk are a ittle more complicated. We go through and so :
The gradients d Q and d K are a little more complicated. We go through From Eq.(2), wehave th即矩阵cddv T的第ljs行,
so the out ith row of the let's know the j th row, let's call-it the jth row d v, j,
d Pij
=dovj.
The g radients d Q 即矩阵dav的第t行, more complicated. W
Orso the out ith row bfthe lets know thegth row, let's call it the jth row d v, j,
The gradient s d Q and d K are a lit
From Eq. (2). we have that d P = d Ov
等于第gr个短阵f第j 行元素的总和are ali is equal. to a summation of each element of the j th row of f of. the first matrix From Eg.
we haye that d P =dov
Since we already computed Li, dv; cai
nory-efficient forward pass. We instead derive the backward pass explicit l
d in a memory-efficient manner.
(where d O den input gradients d Q, d K. d V ∈ Rnxd
(where d Q, d K, d Vdenote
show that it can also be computed with linear memory. Rabe ass can be done without quadratic extra memory by applying forward pass. We instead derive the backward pass explicitly
ass can be done without quadratic ra memory "
by applying ier
n and show that it can also be computed with linear memory. R a be
ardpass can be dor 让我们来看看接下来怎么做nory by applying memory-efficient manner.
the backward pass can be done without quadratic extra memory by ap p
uted ina memory-effic is ot we :do the let'ssee here.
lar loss function. and let th (where d O
a scalar loss function Φ, and let the output gradient be d O e Rnxd (where d
ute theinputgradi让我们来看看接下来怎么做ed Q, d K, d Vde not so we do the let's see her e.
asy to sce. Applying reverse-mode auto diff by hand (aka the chain rule )
那就按i 来操作吧. d P;= do Tvj.
So let's do it by i.
Recall that P ;: = soft max( S:). Using the fact that the Jacobian of y
From Eq. (2), we h那就按a来操作吧 Ov T, and so:
so let's do it by i.
这是对第 H介矩阵第i行中所有d元素的求和l K are it's the sum over all possible i of the ith element in the j th row of the first matrix.
From Eg.
we have that d P
The gradient s d Q and d K are
From Eg. (2), we have that d P = (
所以是 F的第li行第s个无素and d K are From Eq. (), Fwe have that d P = c
The gradient s d Q and d K are
From Eg. (2), we have that d P =(
T 乘以点积d而不是点积 Qand d K are multiplied dot product, not dot product.
这是一个标量向量, 乘以一个向量的乘法, 也就是一
this is a scalar vector, multiplication multiplied by a vector, that is -
Recall that Pi : = soft max ( Si :). Using the fact that
memory-efficient forward pass.
Wejnsteadderiyethe back war
plicitl
utedinam让我确认下公式 是另一个矩阵的第j行.
lalet'me check'what'was the formulae so it Was the j throw Of the t other matrix.
he input gradients d Q, d K, d V e Rnxd
(where d Q, d K, d Vdenote
suggests that the backward pass can be done without quadratic extra memory by applying it can be coputeuamemlory-et ftientmarmie
t let me check what was the formular so it was. the j th row:of the other matrix. es
to compute the input gradients d Q. d K_d V
whered Q. d K. d Vdenote
suggests that the backward pass can be done without quadratic extra memory by applying oint ing to the memory-efficient forward pass. We instead derive the backward pass explicitly t can be computed in a memory-efficient manner.
t there is a scalar loss function Φ, and let the output gradient be d O e Rnxd (where d O denotes
nt checkpointing to the memory-efficient forward pass. We instead derive the backward pass ow how it can be computed in a memory-efficient manner.
p pose that there is a scalar loss function Φ, and let the output gradient be d O e Rnxd (where d0
We want to compute the input gradients d Q, d K, d V ∈ Rnxd (where d Q, d K, d V denote
ent checkpointing to the memory-efficient forward pass. We instead derive the backward pass 即o的第i行, 其中
Tunction o. and let the output gradient be d Oe where d
这是第i行的第个元素, a这是v矩阵的第1行, 不过我们并不需要
this is the ith row ofp this isthej throw of the va trix a ume and but also. we don
我们知道+并不是我们它有的矩阵! 它实除是p的转置, j
we know that fs not a matrix that we have lite actualy the trns pose of p.
Since we 这o赚着dv等手pputed Li, dv j
Twhich means the tfi will b equal tpu d K are a
Sin 因为在矩阵转置中y两个索引会直换.
Li, dv j
because Tnh matriy trans pe sition you wye rt the two indice are a.
Since we already computed Li, dvj The gradients d Q and d K are a
所以这是对 P 的所有可能的1进行求和, 不是和而是1和乘以的第个元素
So this is the summation oyer all Bossiblei's ofpnotili but ultiplied byo.
Since we already computed Li, dvj The gradients d Q and d K are a
Since 这应该写若边着到的公式相同ted Li, dv j
i, and this shoul peegualt theame fomuhthat you see ch the right here.
Since we 这应该写看边看到的公式相同. Li, dv ; can
om Eq. (2), we have that d P = d Ov T, and so:
D
The gradients d Q and d K are a little more complica Dm Eq.(2这样你就可以计算v矩阵中的一行输出so:
this allows you to compute one output row in the v matrix, okay,
dp
From Eq2)
d P
SO:
d Si:=(diag( Pi:)-Pi: PT)d Pi:= Pi:od Pi:-( PTd Pi) Pi:, 而我们知道 直pij其实就是soft max 的输出.
and we know that pij isj
sa XThis con le eor
d Si:=dr
Cone ever u fred
and so :
soft max 的输出是
and the soft output of the soft max is the input of the soft max
and s Q :
soft max 输入的指数值,
to the exponential of the input of the soft max
t le
e more complicated. We go through th 除以与该行相关的归一化因子.
divided by the normalization factor associated with the that row.
and so :
t le
e more complicated. We go through th 因为我们正在遍历i的行, 所以
so because we are iterating through the row of i, it will be the height,
so because we are iterating through the row of i, it will be the height,
cc all that P : = soft max ( Si :). Using the fact that the Jacobian of y = softmax(x)
hat
d St: = (diag( Pi:) - Pi: PT)d Pi: = Pi: o d Pi: -( PT d Pi:) Pi:,
cc all that P:= soft归x化因子将与eoic的行高度相美an of y = softmax(x)
hat the normalization factor associated with that row of um of oi.
e call that Pi : = soft max ( Si :). Using the fact that the Jacobian of y = softmax(x)
hat
d Si: = (diag( Pi:) - Pi: PT)d Pi: = Pi: o d Pi: -( PT d Pi:) Pi:,
where o denotes pointwise multiplication.
Define
所以我们知道p的公式等于s的soft max.
wh'so. we know'that'the'formula for the p is equal to the soft max of s.
Define eq k
where o denotes pointwise multiplication.
Define
现在p的第i行将是s的第i行的soft max,
where : now the ith row of p'will be the soft max of the i th row of s,
Define
这就是这里所写的内容.
where o denotes pointwise m and this is what is written here.
Define
where o denotes pointwise multiplication.
Define
我们从推导中得知, 关于soft max 操作的雅可比矩阵
we know'from our derivation r that'the jacobian with respect to the soft max operation -
Define
如果我们有一个输入×, 车 输出是y,
so if we have'an'input ix and the output is y, of the soft max operation,
De line
where o denotes pointwise multiplication.
Define
那么y关于×的雅可比矩阵等于对角矩阵y.
" The, Jacob iain of the y'with respect to the x is equal to the diagonal y.
Define
它是一个由向量y的元素组成的对角矩阵, 减去y乘以y的转置
it's a diagonal matrix of the element of the vector y minus y multiplied by y transposed.
而且我们之前也看到过, 这个矩阵是对称的.
And we have also seen before that this matrix is symmetric.
where o denotes pointwise multiplication.
Define
D= Pd P=
do;v=do!
V=dooi
(4)
L
then
然而, 你可能不理解这里的公式, 因为我们在链式法则中看到
do:0
4
then
我们总是这样写.
in the chain rule, we always write it like this.
我们总是写下游梯度, 比如dx的dphi,
We always write that the downstream gradient, so the d phi of, let's say, dx,
应该等于上游梯度,
should be equal to the upstream gradient,
即dphi关于dy乘以dy关于dx.
so d phi with respect to dy multiplied by dy and with respect to dx.
这只有在你把这个矩阵放在分子约定中时才能奏效.
this only works if you make this matrix here as a in the numerator convention.
分子约定是生成雅可比矩阵的两种约定之一.
the numerator convention is one of the two convention in which you can create a jacobian.
到目前为止, 我们一直将其写为分子约定.
we so far we have always written it as the numerator convention.
如果你使用分子约定, 这是一个行向量, 这也是一个行向量.
If you use the numerator convention, this is a row vector and this is a row vector.
然而, 如果你想将这里的量视为列向量
However, if you want to treat this stuff here as a column vector,
那么你需要取其转置,
then you need to take the transposed
或者需要在分母约定中生成雅可比矩阵.
or you need to make the Jacobian in the denominator convention.
如何得到这个公式呢?
How to get this formula here?
where o denotes pointwise multiplication.
Doi 因为这个公式基本上是将雅可比矩阵
her
where o denotes pointwise multiplication. 而不是梯度上游梯度乘以雅可比矩阵.
then
where o denotes pointwise multiplication. 这只是因为在这里我们将其视为列向量.
then
NX I
NXN 当你想要将行向量转换为列向量时,
And when you do the, you want to transform a row vector into a column vector,
你需要对方程两边都进行转置.
you take the transpose of both sides of the equation.
让我们实际操作一下.
And let's do it actually.
我们对等式两边都应用转置.
So we apply the transpose to the both side of the equation.
Ok?
Okay?
在矩阵乘法中, 如果你对 AB 进行转置,
In a matrix multiplication, if you do A B transposed,
它会变成 B转置乘以 A转置.
it become B transposed multiplied by A transposed.
因此, 转置操作会独立地应用于矩阵乘法的每个输入,
So the transposed is applied independently to each input of the matrix multiplication,
但我们会反转矩阵乘法的顺序.
but we invert the matrix multiplication.
如果你还记得, 矩阵乘法是不可交换的.
And if you remember, the matrix multiplication is not commutative.
所以我们在这里的做法是, 我们说, 好吧, 这将是dx的dphi,
So what we do here is that we say, okay, it will be the d phi of dx,
这里他们称之为dsi.
and here they call it, here they call it dsi.
因此, 它基本上就变成了 dx上的 d phi.
so it will basically just become d phi on dx.
如果你把这个当作列向量
if you treat this one as a column vector,
那么这个列向量将等于dy 在dx 上的列向量
so this one as a column vector will be equal to dy on dx as a column vector,
也就是分母布局下的雅可比矩阵, 在这种情况下
as a jacobian o in in denominator layout, in this case,
乘以dv 在dy 上的列向量, 这个也是列向量
multiplied by d v on d y as a column vector, this one is a column vector.
这是一个列向量, 这也是你在这里看到的
this is a column vector and this is what you see here.
这就是为什么雅可比矩阵位于上游梯度的左侧.
that's why the jacobian is on the left side of the upstream gradient.
The Jacobian is symmetric (we saw it before when deriving it ).
So the expression of the Jacobian doesn't change when wet rsns pose it.
The Jacobian is symmetric (we saw it before when deriving, it ).
So the expression of the Ja &obi an, dogs n'tchange )whewe trsnsposeitk j.
(5) 需要什么?
Similarly,
what else we need?
(dovj-D)qi
(6
The Jacobian is symmetric (we saw it before when deriving it ).
So the expression of the Ja &obi an, dogs n'tchange )when we trsnsposeitk, 我知道这个推导过程中有很多内容,
(5)
Welt,'i know that there is a lot of things here in this derivation,
(dov-D)qi
(6
嗯,
Dk
(5)
weit,'i'i know that there is a lot of things here in this derivation,
dk j =>d Sjqi=> Pij(d Pij-Di)qi=
(dov-Di)qi
(6)
13. Compute dv; for all j according to Eq. (3). which takes O(d) extra memory.
14. Compute D:但我更倾向于直接看代码ramemory.
:. Corut i prefer actdaniy going ditrety to the'cod e,
15. Compute dk; for all j according to Eq. (6), which takes O(d) extra memory.
16. Compute dv; for all j according to Eq. (3). which takes O(d) extra memory.
17. Compute D; for all i according to Eq. (4). which takes O(n) extra memory.
18. Compute dq;for all i according to Eq.(5). which takes O(d)extra memory.
19. Compute dk; for all j according to Eq. (6), which takes O(d) extra memory.
20. Comput e dv; for all j according to Eq. (3), which takes O(d) extra memory. 那么, 我们直接来看代码吧在编写代码的过程中"我会回头参考公式
So let's go to the'cod e, and whiie'writing the code, T go'back'to the formulas in
21. Compute dk; for all j according to Eq. (6), which takes O(d) extra memory.
22. Compute dv; for all j according to Eq.(3), which takes O(d) extra memory. 这样我们就能将实际操作与论文中的公式对应起来.
which we can find the'association'dr what We are doing and the formula in the paper.
23. Compute dk; for all j according to Eq. (6), which takes O(d) extra memory.
24. Compute dv;for all j according to Eq.(3), which takes O(d) extra memory.
25. Compute D for 我认为这是最好的方式
extra memory.
extra memory.
26. Compute dk; for all j according to Eq. (6), which takes O(d) extra memory.
So let's proceed further.
19:
On chip, compute D;=rowsum(d OO)∈ RB 好了,
T
2xs: Alright guys, now we can finally code the backward pass.
22:
Onchip. computed Kd K+rd SQR
25:endfor
24:
Writed Kd Kd Vd Vto HBM.
19:
On chip, compute D;=rowsum(d OO)∈ RB
20:
21:
22:
Onchip. computed Kd K+rd SQ
23:
Write d Kd K, d Vd V;to HBM1.
end for 24:
25:
endfor
20:
21:
Write d Qd Q+rd SKB, xo HBA
let's look at the algorithm of the backward pass as written in the paper.
21:
rited K
26: Reni d Q. d R. d V
22:
On chip, compute d Kd K+rd SQBxd
24:
23: 这就是 Flash " Attention 1的论文
endfor
26: Retun d Q. d K
25:
end W
This is the paper Flash Attention 1.
We see that similar to the forward pass, the backward pass perforns O( N2) FLOPs and only requires gradients
22:
On chi p, compute d Kd K +rd SQ; Bxd
23:
24:
Write d Kd K, d V;d V to HBAI
end for 25:end for
26: Return d Q, d K. d V.
O( N) extra memory beyond inputs, output, output gradient, and input gradients.
We see that similar to the forward pass, the backward pa
performs O( N2) FLOPs and only requires
22
On chip, compute dkd K+rd SQ. xd
23:
endfor
Titon网站上代码的结构来进行. 立
because we will follow the structure of the code that is present on the Triton website.
22:
On chip, compute d Kd K +rd SQ Bxd
23:
end for 所以 文样拆分: 但我简化了它
so it's not my idea to split it like this, l
but I simplified it.
O( N) extra memory beyond inputs, output, output gradient, and input gradients.
22:
On chip, compute dkd K+rd SQxd
23:
end for
I simplified it So it's different than the one that you can find on line,
O( N) extra memory beyond inputs, output, output gradient, and input gradients.
We sce thats
22:
On chip. compute d Kd K+rd SQBxd
23:
24:
end for Write d K 因为我的版本是简化版,
25:end for
26: Return d Q. d K. d V
because mine is a simplified version O( N) extra memory beyon d inputs, output, output gradient, and input gradients.
We see that similar to the forward pass, the backward pass performs O( N-) FLOPs and only requires
22:
On chip. compute dkdk+rd SQxd
23:
24:
end for
25: 并直适用于茵巢和非因果注意力机制
and mine works with the causal and non-causal attention.
O( N) extra memory beyond inputs, output, output gradient, and input gradients.
Ne sce thatsimilar t
otihe forward pass, the backward passperforms O( N) FLOPs and only requires
22:
On clhip. ompute d Kd K+rd SQBxd
23:
24:
end for
Write d K 首先, 如巢你看这个算法,
26: Return d Q, d K.
25:end for
So first, if you look at this algorithm,
O( N) extra memory beyond inputs, output, output gradient, and input gradients.
Hd only requires
On chip. compute dkdk+rd SQxd 你会发现我们有一个外部循环遍历所有的 K和 V块,
end for
you
I can see that we have an outer loop through all the K
O( N) extra memory beyond inputs, output, output gradient, and input gradients.
22:
On chip, compute d Kd K+rd SQ. xd
21:
23:
end for
25: 一个内部循环遍历所有的查询块.
26:
and v blocks and an inner loop through all the query blocks.
We sce that
22:
On chip, compute d Kd K+rd SQBxd
23:
end for 如你所着到的, 为了计算 DQ,
to compute the DQ,
O( N) extra memory beyond inputs, output, output gradient, and input gradients.
We see that similar t
23:
24:
Write d Kd K. d Vd V;to HBAI.
end for 26: Retur d Q. d K. d V
25:end for
O( N) extra memory beyond inputs, output, output gradient, and input gradients.
We see that similar to the forward pass, the backward pass performs O( N2) FLOPs and only requires
24:
23:
end for
26: Returh
which is the downstream gradient of the loss with respect to the Q matrix,
O( N) extra memory beyond inputs, output. output gradient, and input gradients.
We see that similar t
22:
On chip. compute d Kd K +rd SQBxd
23:
24:
end for Write d K
25:end for
26: Return d Q, d K. d V
we need to have an iteration through all the Ks.
O( N) extra memory beyond inputs, output, output gradient, and input gradients.
We see that similar to the forward pass
22:
On chip, compute dkd K+rd SQ. xd
23:
end for 因
or
So if we follow the loop like it is.
O( N) extra memory beyond inputs, output, output gradient, and input gradients.
s and only requires
On chip, compute dkd K + rd SQ e Bxd
3
Write d K, d Kd V,
end for
end for
O( N) extra memory beyond inputs, output, output gradient, and input gradients.
We see that similar to tie forward pass, the back ward pass performs O( N2) FL. OPs and only requires
end for it would involve writing to the high bandwidth memory, so to the DRAm of the GPU.
22:
On clhip. compute d Kd K+rd STQRB. xd
23:
24:
Write d Kd Kd V
end for 这可能会影响效率.
26: Return d Q, d K. d V
25:end for at every inner i iteration, and that could be also that that is not so efficient, um,
O( N) extra memory beyond inputs, output, output gradient, and input gradients.
22
On chip, compute dkd K+rd SQ. xd 如果我们不想写人数据, 京
23:
end for 就需要在块之间进行某种同步 此外,
and also, if we don't want to write, it would require some sort of inter,
O( N) extra memory beyond inputs, output, output gradient, and input gradients.
22:
On chip. compute d Kd K+d STQe B. xd
23:
24:
Write d Kd Kd V
end for 这同样效率不高.
26: Return d Q. d K. d V
25:end for
some sort of synchronization between blocks, which is also not very efficient.
22:
On chip, compute d Kdk+rd SQxd
23:
24:
end for
25:
end大
so we split, we will split this four into two parts,
O( N) extra memory beyond inputs, output, output gradient, and input gradients.
22:
On chip, compute d Kd K+rd SQ. xd 因为可以看到每个dg的计算依赖于对 K的循环,
23:
end for
beca use we can see that each dq depends on a loop over the case
O( N) extra memory beyond inputs, output. output gradient, and input gradients.
We sce that si in ilar to the forward pass
22:
On chip, compute d K;
23:
21:
Write d Kd K, d V;d V;to HIBAI
end for
26: Return d Q. d K. d V.
25: end for
O( N) extra memory beyond inputs, output, output gradient, and input gradients.
We see that similar to the forward pass, the backward pass performs O( N2) FLOPs and only requires
22:
On chip, compute d Kd K+rd SQ. xd
23:
end for
a nd each dk depends on a loop over all the q's.
O( N) extra memory beyond inputs, output, output gradient, and input gradients.
22:
On chip, compute d Kd K +rd SQBxd
23:
end for 我们将固定第k个块, 为了计算 并遍历所有的q 块.
so to compute dk, we will fix the kth block and iterate through all the q blocks.
O( N) extra memory beyond inputs, output, output gradient, and input gradients.
22:
On chip, compute d Kd K +rd SQ Bxd 我们将进行另一轮迭代, 不
23:
end for 接着, 在这轮迭代中固定q块
then we will do another iteration in which we fix the q block
O( N) extra memor y beyon d inputs, output, output gradient, and input gradients.
Wescethatsimilarto
he lorwardpass
ackwardpassperlorms O( N) FLOPs and only re ui res
22:
On chip. compute d Kd K+rd SQB. xd
21:
23:
end for
25:end for
26: Return d
and iterate through all the kv blocks to compute the dq.
backward passperforms O( N-) FLOPs and only requires
22:
On chip, compute d Kd K+ rd SQ;e Bxd
23:
21:
end for
2:end这京 这个思路
26: Retr
this is what we are going to follow and this is an idea
O( N) extra memory beyond inputs, output, output gradient, and input gradients.
22
On chip, compute d Kd K+rd SQ. xd
23:
end for
Titon网站上的原始实现中借鉴的
that i took from the original implementation that is present on triton website.
O( N) extra memory beyond inputs, output, output gradient, and input gradients.
e that the for war tl pass, the
22:
On chip, comput d Kd K+rd SQBxd
end for
Another thing that we can notice here is: where is it?
O( N) extra memory beyond inputs, output, output gradient, and input gradients.
We sce that similar
where o denotes pointwise Define multiplication D= Pd P :=
o2v;=do
=dooi
(4)
then
d S= Pod P:-DP
the n
da=ds Pds
d S= Pd P
P(d P-D)
L
We describe the full details of FLAs H AT r ENT Io N forward pass. Given input sequences Q. K. V e RNx a want to compute the attention output O e RN xd
T. compute the dq and the dk, so the dq vector and the dk vector, we need this element,
dropout ( P. paro the input to-a quences in the batch don't
pute dq ; for all i according to Eq.(5). which takes O(d)extra memory. 这里称为di 的信息它是两者共享的.
pute dk ;for all jac K this information here called di, and it's shared between the two.
LASH ATTENTION : Forward Pass
27. Compute dq; for all i according to Eq.(5). which takes O(d) extra memory.
28. Compute dk; for all j according to Eq. (6), which takes O(d) extra memory.
B. 3
FLASH ATTENTION : Forward Pass We describe the full details of FLAs H ATTENTION forward pass. Given input sequences Q. K. V e RNxd, we
29. Compute dq; for all i according to Eq. (5). which takes O(d) extra memory. 因此, 我们可以预先计算它, 然后将其复用于qi 向量,
So we can prez compute it and then we can reuse it for the qi vector,
We de se rib the full details of FLAs H ATTENTION forward pass. Given input sequences Q KVRx d we
30. Compute dq; for all i according to Eq. (5). which takes O(d) extra memory.
31. Compute dk;for 以计算qi向量和dk向量
B. 3
Fl Ast or compute the rq ir vector and the dk vector.
We de seri be the full details of FLASH ATTENTION forward pass. Given input sequences Q. K. V RNxd we
32. Compute dq; for all i according to Eq.(5), which takes O(d) extra memory.
33. Compute dk; for all j according to Eq. (6), which takes O(d) extra memory.
B. 3
FLASH ATTENTION : Forward Pass We describe the full details of FLAs H ATTENTION forward pass. Given input sequences Q, K. V e RNxd, we
34. Compute dq; for all i according to Eq. (5). which takes O(d) extra memory.
35. Compute dk; for all j acco 这个di是什么呢? memory.
B. 3
FLASH ATTENTIon : For w What is this d?
36. Compute dq; for all i according to Eq. (5). which takes O(d) extra memory.
dl is introduced here and it's ther dot product of a vector that is the dol vector We describe the full details of FLAs H ATTENTION forward pass. Given input sequences Q. K, V e RNxd, we
Therefore the backward pass can also be computed with O(n) extra memory :
Therefore the backward pass can also be computed with O(n) extra memory:
37. Compute dv ; for all j accordi ng to Eq. (3), which takes O(d) extra memory.
38. Compute D;for all i according to Eq.(4), which takes O(n)extra memory.
Therefore the backward pass can also be computed with O(n) extra memory : 所以我们要做的第一件事是遍历9
39. Computedv
lemorv
So the. first, thing that. we. wil. do is doa loop oyer all the. yectors in 0
和d O中的所有向量, 并计算它们的点积, 以得到这个di元素.
Therefore the backward pass can also be computed with O(n) extra memory 2. and, do and do their dot products. to compute this. dl. element.
Therefore the backward pass can also be computed with O(n) extramemory:
40. Compute dv; for all j accordi ng to Eq. (3), which takes O(d) extra memory.
41. Compute D;for all i according to Eq.(4), which takes O(n) extra memory.
然后我们将使用这个d 元素, 实际上, 让我想想.
Therefore the backward pass can also be computed with O(n) extra memory :
, 是的.
42. Then, we, ywill use this d! element and actually. let me see, Yeah.
Therefore the backward pass can also be computed with O(n) extramemory:
43. Compute dv ; for all j accordi ng to Eq. (3), which takes O(d) extra memory.
44. Compute D;for all i according to Eq.(4), which takes O(n)extra memory.
接着, 我们会利用这个di元素来更新并计算, DQ 、和 DK.
Therefore the backward pass can also be computed with O(n) extra memory :
And then, we, will use. this R! element, to. update, to. compute DQ and DK.
45. Compute D; for all i according to Eq.(4), which takes O(n) extra memory.
46. Compute dq;for all i according to Eq.(5), which takes O(d) extra memory.
47. Compute dk; for all j according to Eq.(6). which takes O(d) extra memory.
48. Compute D; for all i according to Eq. (4), which takes O(n) extra memory. 此外, 我们还将进行另外两个循环, 一个循环中固定查询
J (queue And we'will also have another two loops s one in which we fix the queue
49. Compute D; for all i according to Eq.(4), which takes O(n) extra memory.
50. Co并遍历所有键r(keys)5, 另一个循环中固定键
and we iterate through all the keys ah done in which we fix the keys
51. Compute D; for all i according to Eq. (4), which takes O(n) extra memory.
52. Compute dq;for alli acco并遍历所有查询 O(d)extra memory.
53. Compute dk; for anditeratetthrough r all'the queues tra memory.
54. Compute D; for all i according to Eq.(4), which takes O(n)extra memory.
55. Compute dq;for all i according to Eq.(5), which takes O(d) extra memory.
56. Compute dk; for all j according to Eq.(6), which takes O(d) extra memory.
57. Compute D; for all i according to Eq. (4), which takes O(n) extra memory.
58. Com现在, r我们对代码的结构有了大致的了解.
Sonow that we know more'or less'the structure of the'code'that we're with.
. requires _grad _()
normal _(mean =0. 0, std=0. 5)
torch. enpty(
( BATCH_ SIZE, NUM _ HEADS, SEO _ LEN, HEAD _0 IM), 好的
normal_(mean=o. e, std=0. 5)
. requires _grad _()
All right.
> TIMELINE OUTLINE
normal _(mean =0. 0, std=0. 5)
requires_grad_()
317 因此我们首先在这里编写这个反向传播函数.
Sowe start by writing this backward function here.
> TIMELINE OUTLINE 323
normal _(mean=e. 8, std=0. 5)
requires _grad _() 让我确认一下, 女 好的.
=(
torch. en pty (
( BATCH _ SIZE, NUM_ HEADS, SEQ_ LEN, HEAD_ DIM), dtype =
Let me check, yeah OUTLINE noral_(meane. 8, std= B. 5)
> TIMELINE. requires _grad _()
return 0 @static method def backward 好的, 还记得这个保存的张量吗?
OUTLINE def
testop Okay, so do you remember this saved tensor?
Q =(
> TIMELINE torch. en pty (
return 0 这些都是我们在前向传播过程中保存的信息
gst
deftestop BATCs Iz E, NM_ HE As These are all the information > TIMELINE OUTLINE 314
313
torch. enpty
return 0
308
@static method def backward (ctx, do): 以便计算反向传播
389
10
that we save ·during s the forward pass to compute the backward pass.
> TIMELINE OUTLINE torch. en pty (
return 0
306 现在, 为了优化内存利用率, 我们采用 Flash Attention算法.
311
Now-to optimize the memory utilization in flash.
OUTLINE 312
313
> TIMELINE
314
torch. enpty (
return 0
3e6 在注意为机制中我们不保存查询与键矩阵转置相乘的结果
attention, we don't save the-query multiplied by the transpose of the key matrix,
> TIMELINE OUTLINE torch. en pty (
return 0 因为这会生成一个序列乘以序列的矩阵, 体积过于庞大.
because that would be a sequence by sequence matrix that is too big.
> TIMELINE OUTLINE torch. en pty (
中 我们将其保存到 HBM(高带宽内存)中, 位于
39
388
DRAM(动态随机存取存储器)内,
to save into the HB My in the DRAM during the forward pass,
> TIMELINE OUTLINE 314
torch. enpty
return 0 @static method def backward ( 随后再从 HBM 取回至本地内存.
Q, K, V,
and then-re-get it back from the HBM into the local memory.
> TIMELINE OUTLINE 314
torch. enpty
return 0 我想提醒您, 在 Triton中, 与 CUDA相比
Because l want-to. remind you that in Triton, compared to CUDA > TIMELINE OUTLINE torch. en pty
385
return0 我们采取的做法是将数据从高带宽内存加载到共享内存
in Triton what we do is wes load stuff from-the high bandwidth memory in the shared memory,
> TIMELINE OUTLINE torch. en pty (
return 0 也就是 SRAM(静态随机存取存储器)中
testop BATCHSIZE, NUMHEADS, SEQ _ LEN, HEAD _ DIHSOt SRAM
OUTLINE
314
def
Q=(
> TIMELINE torch. en pty (
385
return0 我们在共享内存中完成所有运算操作后, 再调用存储方法
wedoall the operations there and then, after, when we call the store method.
> TIMELINE OUTLINE torch. en pty (
return 0
@staticme 将结果从共享内存保存至高带宽内存.
we save the element from the shared memory into the high bandwidth memory.
> TIMELINE OUTLINE torch. en pty (
return 0 @static method def backward( 为了避免完整地生成这个 S矩阵
Q, K, V,
32
Soin order to not materialize. this S matrix in its entirety,
> TIMELINE OUTLINE 314
313
torch. enpty(
return 0
306 将其保存到 HBM后再重新取回一一这一过程可能非常耗时.
saveitto-the-HBMeand then re-get-it back, which could be very slow.
> TIMELINE OUTLINE torch. en pty (
return 0 @static method def backward (ctx, do): 其次, 实际上,
31
> TIMELINE OUTLINE 313
Q=(
torch. enpty(
return 0
@staticmethod
defbackwa 这样做成本很高, 因为当前我们通常
Q, K,
it is very. expensive because usually right now OUTLINE 313
def
test_o
Q=(
> TIMELINE torch. en pty (
return 0 是在成干上万的token上计算注意力.
we are computing attention on. thousands and thousands of tokens.
> TIMELINE OUTLINE torch. en pty (
return 0 试想一下, 保存一个5000乘以5000大小的矩阵,
311
312
def test
Sor imagine saving a matrix ·that is5, 0o0by5, 000.
> TIMELINE OUTLINE 314
313
torch. enpty(
return 0 对于每个批次、每个注意力头来说, 保存如此庞大的矩阵都是不小的负担
That's a big matrix to save-for each batch, for each batch and for each head.
> TIMELINE OUTLINE 314
torch. enpty(
return 0 因此,:保存这样的矩阵确实会消耗过多资源
So that would be really too ·expensive to save.
OUTLINE 313
def test
> TIMELINE
314
torch. enpty (
return 0 Flash Attention 的核心思路在于, 在反向传播过程中实时重新计算
@static method 311
312
dettestp Sothe idea in Flash Attention is to recompute > TIMELINE OUTLINE 314
313
Q=(
torch. enpty
return 0 那些可以即时得出的结果, 因为无论如何
what we can compute on the fly during the backward pass because anyway,
> TIMELINE OUTLINE torch. en pty
return 0 如果我们选择加载这些数据, 都会受到内存 I/0的限制.
311
312
dettif we were to load-it it would ·be memory Io bound > TIMELINE OUTLINE 314
313
torch. enpty
return 0 因此, 相比从内存中保存和恢复数据, 即时重新计算反而更加高效.
So it's faster ·too recompute than-to save. it and restore it from the memory.
> TIMELINE OUTLINE torch. en pty
return 0 这便是 Flash Attention 的设计理念
test ope acsi z, u This is the *idea of Flash : Attention.
OUTLINE 313
def
Q=(
> TIMELINE torch. en pty
return 0
309 好的,:我们在前向传播过程中保存了一些数据
16
313
Okayy-so wesaved some stuff during the forward pass > TIMELINE OUTLINE torch. en pty (
return 0 @static method 现在在反向传播时可以访问这些数据
and now z we can-access it back during the backward pass > TIMELINE OUTLINE 314
313
torch. enpty(
return 0 @static method defbackvard(ctx, 这些数据被存储在上下文中
Q, K, V, O, M
31
and. this stuff is saved in. the context OUTLINE 313
def test_op( BATCH_ SIZE,
Q=(
> TIMELINE torch. en pty (
return 0 就像是由 Py Torch提供的一个字典一样.
and it's kind of a dictionary that is made available by Py Torch.
> TIMELINE OUTLINE torch. en pty (
return 0 没
386 错, 这样我们就能取回查询(query)、键(key)和 和值(value)这些数据
311
All right, sos we get back the query key and values.
> TIMELINE OUTLINE 313
torch. enpty(
def backward (ctx, do ):
@static method Q, K, V, O, M=ctx. saved_tensors 众所周知
test_op BATCH _ SIZE, NUM _ HEADS, SEQ _ LEN, HEA And as you know,
> TIMELINE OUTLINE Q=(
tor ch. enpty (
( BATCH _ SIZE, NUM _ HEADS, SEO _ LEN, HEAD _ DI M), dtype =dtype, device ="cuda
@static method def backward (ct x, do):
Q, K, V, O, M=ctx. saved_ter
311 在自动求导过程中
Py Torch during the auto grad wit r just give us the gradient of the loss > TIMELINE OUTLINE ( BATCH _ SIZE, NUM_ HEADS, SEQ_ LEN, HEAD_ DIM), dtype =dtype, device ="cuda
@static method def backward (ctx, do ):
Py Torch 会直接提供损失函数相对于我们实现的注意力机制输出结果的梯度.
with respect to the output of our implementation of the attention, of our attention.
> TIMELINE OUTLINE ( BATCH _ SIZE, NUM _ HEADS, SEO _ LEN, HEAD _ DI M), dtype =dtype, device ="cuda
@static method def backward (ctx, do ): 这就是 Triton实现的注意力机制.
e
test _op ( BATCH _ SIZE, NUM _ HEADS,
So'this'is'Triton attention.
> TIMELINE OUTLINE Q=(
tor ch. enpty (
( BATCH _ SIZE, NM_ HEADS, SEQ_ EN, HEAD_ IM), dtyp=dtype, device =cuda
@static method def backward (ct x, do):
Q, K, V, 0, M= 接下来, 我们需要仅利用
test_op( BATCH_ SIZE
Q=(
And'then we need'to compute dq, dk,
> TIMELINE OUTLINE torch. en pty (
( BATCH _ SIZE, NM _ HEADS, SEQ _ LEN, HEAD _ IM), dtype =dtype, device =cuda
@static method def backward (ctx, do): 损失函数相对于输出的梯度, 来计算查询(dq)、键(dk)和值(dv)白 的
and dvby-using only the gradient of the output with respect to the loss > TIMELINE OUTLINE ( BATCH _ SIZE, NUM_ HEADS, SEQ_ LEN, HEAD_ DIM), dtype =dtype, device ="cuda
def backward (ctx, do ):
@static method Q, K, V, O, M=ctx. saved_tensors 梯度.
test_op( BATCH _ SIZE, NUM _ HEADS,
With respect to the output.
> TIMELINE OUTLINE Q=(
tor ch. enpty (
( BATCH _ SIZE, NUM _ HEADS, SEQ _ LEN, HEAD _ IM), dtype =dtype, device =cuda
@static method def backward (ctx,
do ): 我们还需要进行一些验证检查
let
test_op( BATCH _ SIZE, NUM _ HEADS, SEQ _
Q =(
We do for some checks.
> TIMELINE OUTLINE torch. en pty (
( BATCH _ SIZE, NUM _ HEADS, SEO _ LEN, HEAD _ DI M), dtype =dtype, device =cuda
def backward (ctx, do ):
@static method Q, K, V, O, M=ctx. save d_tensors 那么在这里
def
test_op( BATCH _ SIZE, NUM _ HEADS, SEQ _ LEN, HEAD _ DIM, Ca
So here...
ch. float16):
> TIMELINE OUTLINE Q=(
tor ch. enpty (
( BATCH _ SIZE, NM _ HEADS, SEO _ LEN, HEAD _ IM), dtype =dtype, device =cuda
@static method def backward (ctx, do): 我明白这段代码还有优化空间, 比如
I knowl could optimize this code and make it even smaller by, for example,
312 > TIMELINE OUTLINE ( BATCH _ SIZE, NM _ HEADS, SEQ _ LEN, HEAD _ IM), dtype =dtype, device =cud
ctx. save_for_ba 以检查一下这里使用的步幅 富(stride), 这样能让代码变得更简洁高效
307
@static m checking that here, the stride that I am using
> TIMELINE OUTLINE 30 B
defbackward(ctx, do):
HEAD _ DIM = HEAD _ DI H_ K,
STAGE =stage, 实际上, 在代码内部, 我总是假设步幅是相同的
I actually, inside of. the code, I always pretend that the stride is the same,
> TIMELINE OUTLINE return o
assert Q. stride ()m K. stride ()m V. stride ()
assert do. is _contiguous ()
4. stride()
d0. stride()
def
test_op( BATCH _ SIZE 不过这一点并不影响整体功能.
torch. en pty ( BATCH _ SIZE, NUM_ EADS, SEO _ LEN, HEAD_
normal_(mean=e. 8, std=0. 5)
but it doesn't matter.
> TIMELINE OUTLINE. requires _grad _()
312
assert Q. stride ()mm K. stride()
assert do. is _contiguous ()
= V. stride ()
1. stride()
do. stride 我只是从 Triton 的代码中提取出来, 并尝试将其简化.
319
318
just. take. the code from Triton and try to simplify it.
> TIMELINE OUTLINE requires _grad _(
assert do. is _contiguous ()
Q. stride (
K. stride() == V. stride()
1. stride() do. stride (
def 我的自标是简化代码, 而不是优化它.
BATCH _ SIZE, NUM _ HE > TIMELINE OUTLINE
assert Q. stride ()=m K. stride() == V. stride() = 0. stride() m do. stride ()
e test 好的我们创建了一些向量和张量
All right, we create the *vectors, the tensors,
> TIMELINE OUTLINE normal _(mean=0. e, std=0. 5)
assert do. is _contiguous ()
assert Q. stride ()
1. stride()
d0. stride() 用来存储反向传播的结果, 也就是dgq、dk和 dv.
in which we will store the result of this backward pass, which is the dq, dk, and dv.
> TIMELINE OUTLINE BATCH _ SIZE,
d Q = torch. empty _like ( Q )
torch torch. en pty _like ( K ) 正如我们从梯度定义中所了解的
And, as you know from what we have seen of the definition of the gradient 318 > TIMELINE OUTLINE ( BATCH _ SIZE, NUM_ HEADS, SEO_ LEN, HEAD_o IM), dtype =dtype, device ="cuda
d Q = torch. empty _ Like ( Q)
=torch. enpty_like( K)
317 梯度向量的输出大小
the size of the output of the gradient vector is the size of the vector 18 > TIMELINE OUTLINE ( BATCH _ SIZE, NUM_ HEADS, SEQ_ LEN, HEAD_ DIM), dtype =dtype, device ="cuda
d Q = torch. empty _like ( Q)
=tor ch. enpty =torch. empty _like ( K ) 与计算梯度的向量大小相同
With respect to which we calculate the gradient,
> TIMELINE OUTLINE ( BATCH _ SIZE, NUM_ HEADS, SEQ_ LEN, HEAD_ DIM), dtype =dtype, device ="cuda "
d Q = torch. empty _like ( Q )
=torch. empty _like (
=torch. empty _like ( K) 因为分子总是一个标量
test_op( BAT Q =(
because in the numerator is always a scalar > TIMELINE OUTLINE ( BATCH _ SIZE, NUM_ HEADS, SEO_ LEN, HEAD_o IM), dtype =dtype, device ="cuda
d Q= torch. empty_ Like( Q)
317 而我们要对输入向量中的所有元素计算梯度.
and we compute the gradient with respect to all the elements in the input vector.
> TIMELINE OUTLINE ( BATCH _ SIZE, NUM_ HEADS, SEQ_ LEN, HEAD_ DIM), dtype =dtype, device ="cuda *
314
torch. enpty_like( K 因此 梯度本身的输出是一个与计算梯度的元素大小相同的
so the output, the gradient itself, is a vector of the same size of the element by which > TIMELINE OUTLINE ( BATCH _ SIZE, NUM_ HEADS, SEQ_ LEN, HEAD_ DIM), dtype =dtype, device ="cuda
=to rch. enpty_like ( K)
=torch. enpty_ Like( V) 向量
test_op( BATCH _ SIZE,
Q =(
We compute the gradient with respect to.
> TIMELINE OUTLINE torch. en pty ( BATCH _ SIZE, NUM_ EADS, SEO_ LEN, HEAD_ DIM), dtype =dtype, device =cuda
or ch. enpty _like ( K) 所以我们得到了一些关于批量大小的信息.
def
test_op( BA So we get some information on the bed size.
> TIMELINE OUTLINE ( BATCH _ SIZE, NUM_ EADS, SEO_ LEN, HEAD_ DIM), dtype =dtype, device =cuda
d Q = torch. empty _like ( Q)
d K= torch. enpty_like( K)
d V= torch. empty _like (v ) 等等, 等等, 等等.
blah, blah, blah.
> TIMELINE OUTLINE def
test_op( BATCH _ SIZE, NUM _ HEADS, SEQ _ LEN, HEAD_ OIM, Ca USa
Q=(
m. float16):
314
orch. enpty_like( K) 稍后我们会了解这个warp数量和stage 数量是什么意思.
And later we will see what is this number of warps and the number of stages.
> TIMELINE OUTLINE
BATCH _ SIZE, NUM _ HEADS, S
NUM_ STAGE 我现在先不解释这个,
BLOCK _ SIZE _ MICRO, BK
I will not explain it now.
> TIMELINE OUTLINE def
test_op BATCH _ SIZE, NUM_ HEADS, SEQ_ LEN, HEAD_ DIM
Q=(
BATCH _ SIZE, NUM _ HEAD S
UM_ WARPS, NUM _ STAGE
SEO_ LEN= Q. shape[:3]
D
y Torch中的warp数量
BLOCK _ SIZE _ MICRO,
BLRC It's how Py Torch number of parts warps > TIMELINE OUTLINE
def
test_op( BATCH_ SIZE, NUM_ HEA
Q=(
决定了我们希望在网格中启动多少线程
is an indication on how many threads we want to launch in our grid.
2
> TIMELINE OUTLINE
316 而
stage 数量实际上指的是软件流水线中使用的阶段数.
and number of stages is actually the number of stages that is used in software pipelining.
> TIMELINE OUTLINE
稍后讨论自动调优时, 我们会了解什么是软件流水线,
We will see later what is software pipelining when we talk about the auto tuning.
> TIMELINE OUTLINE
BATCH _ SIZE, NUM _ HEADS,
NUM _ WARPS,
NUM _ STAGES 然后我们定义了一些块.
BLOCK _ SIZE _ MICRO, BLOC Then we define some blocks.
> TIMELINE OUTLINE def test_op BATCHSIZEUM_ HEADSSEQ_ NHA
Q=(
310
BATCH_ SIZE, NUM_ HEADS
Q. shape [:3] 在原始代码中, 我想它们被称为 KV1、 KV2、(
Q1 和 Q2 块
In the original code, 1 think they call it the block KV1, KV2, Q1, and Q2
> TIMELINE OUTLINE
BATCH _ SIZE, NUM_ HEADS, SEO_ LEN = Q. shape[:3]
BLOCK _ SIZE _ MICRO BLOC 我觉得这有点让人困惑
I think it was confusing.
> TIMELINE OUTLINE def
test_op BATCH _ SIZE, NUM_ EADS, SEQ_ LEN, H
Q=(
BATCH _ SIZE, NUM _ HEADS, SEO_ LEN = Q. shape[:3]
BLOCK_ SIZE_ MICRO, 我称之为宏块和微块, 因为
2
call it thel block macro and block micro because the thing > TIMELINE OUTLINE 2
318 我们将固定的部分和送代的部分分别对应查询的不同处理方式.
that we will fix and the things that we will iterate from will be one is the query.
> TIMELINE OUTLINE
BATCH _ SIZE, NUM _ HEADS, SEO_ LEN = Q. shape[:3]
NUM_ W 因此, 我们固定查询块, 遍历所有键
sowefixthe query block and we iterate through all the keys > TIMELINE OUTLINE
BATCH _ SIZE, NUM_ HEADS, SEO_ LEN = Q. shape[:3] 然后固定键和值块, 再重新遍历查询.
and then we will fix the keys and values block and reiterate through the queries.
> TIMELINE OUTLINE
BATCH _ SIZE, NUM _ HEADS, SEO_ LEN = Q. shape[:3]
NUM_ WAI 我们遍历的是微块, 固定的是宏块.
the one that we iterate on is the micro one and the one that we fix is the macro one.
> TIMELINE OUTLINE
BATCH _ SIZE, NUM _ HEADS, SEO_ LEN = Q. shape[:3]
NUM_ W
IZE_ MACR0 =32, 1 这是我
, 我使用的命名方式.
this is my-uh, t
the naming that i am using.
> TIMELINE OUTLINE test _op ( BATCH _ SIZE, NUM _ HEAD
BLOCK _ SIZE _ MICR NUM _ WARPS, 接下来, 正如我之前提到的
Then we, as I said before,
> TIMELINE OUTLINE def
test_op BATCH _ SIZE, NUM_ HEADS, SEQ_ LEN, HEAD_ DIM,
Q=(
BATCH _ SIZE, NUM _ HEADS, SEO_ LEN = Q. shape[:3]
318 我们需要预先计算之前在论文中看到的 DI元素.
we need to pre-computethe Dl elements that we saw in the paper before.
> TIMELINE OUTLINE
BATCH _ SIZE, NUM_ HEADS, SEO_ LEN= Q. shape[:3]
NUM_ WARPS,
BLOCK_ SIZ 这就是我们要启动的第一个内核.
So that's the first kernel that we are going to launch.
> TIMELINE OUTLINE
BLOCK _ SIZE _ 这个内核会有自己的启动网格
And this kernel will have its own launch grid > TIMELINE OUTLINE def Q =(
BATCH _ SIZE, NUM_ HEADS, SEO_ LEN = Q. shape[:3] 因为之后我们想要优化这个内核的调优.
because later we want to optimize the tuning of this kernel.
2 > TIMELINE OUTLINE
I UM_ W 稍后我们会讨论针对其自身参数的调优.
Later we will talk about the tuning with respect to its own parameters.
322 > TIMELINE OUTLINE
BATCH _ SIZE, NUM _ HEADS, SEO _ LEN = Q. shape [:3]
NUM_ WARPS, NUM_ STAGES = 4, 3
BLOCK _ SIZE _ MICRO, BLOCK _ SIZE _ MACRO=32, 1 让我想想.
So let me see > TIMELINE OUTLINE deftes t_op( BATCH _ SIZE, NUM _ HEADS, SEQ _ LEN, HEAD _ DIH, casal, dtype=
Q=(
rch. float16):
BATCH _ SIZE, NUM_ HEADS, SEO_ LEN = Q. shape[:3]
BLOCK _ SIZE _ MICRO, BLOC 我们接下来要做什么呢?
what are we going to do so here?
> TIMELINE OUTLINE def
test_op BATCH _ SIZE, NUM _ HEADS, SEQ _ LEN, H
BATCH _ SIZE, NUM_ HEADS, SEO_ LEN = Q. shape[:3]
19 我们要启动的第一个内核就是这个预处理内核.
so the first kernel that we are going to launch is this pre-processedkernel.
> TIMELINE OUTLINE
SEQ _ LEN= SEQ _ LEN, 这个预处理内核会预先计算我们需要计算的所有di 元素,
this pre-processedcamera will p pre-computeall the di elements that we need to compute.
OUTLINE TIMELINE
BLOCK _ SIZE _ O = BLOCK _ SIZE _ MACRO SEQ _ LEN= SEQ_ LEN,
HEAD_ DIM= Ct X. HEAD_ DIM, 我记得 dk 和 dv.
iremember dk and dv > TIMELINE OUTLINE det
test_op BATCH _ SIZE, NUM _ HEADS, SEQ_ LEN, HEAD_ DIM,
Q=(
causal, dtype =torc
SEQ _ LEN= SEQ_ LEN, 如果考虑dq和dk, 这个di元素仅依赖于o和do
if i dg and dk and this di element depends only on o and do so,
> TIMELINE OUTLINE
329
SEQ_ LEN= SEQ_ LEN, 那么我们就来实现它, 并创建另一个名为backward preprocessor 的函数
let's do it and let's create another function called the backward preprocessor.
> TIMELINE OUTLINE
D= D,
SEQ _ LEN= SEQ_ LEN,
BLOCK _ SIZE _ Q= BL
HEAD DIM = Ct X. 预处理网格的流程是什么?
what is the process pre-process grid?
> TIMELINE OUTLINE test _op ( BATCH _ SIZE,
SEO _ LEN= SEQ_ LEN,
BLOCK _ SIZE _ O= E
HEAD DIM 这是该函数或内核的启动网格
this is the launch grid of this function, of this kernel > TIMELINE OUTLINE
def
est_op( BATCH _ SIZE,
SEQ _ LEN= SEQ_ LEN 它将针对每个批次和每个头独立启动
and this will be launched on a independently, for each batch and for each head and.
3 > TIMELINE OUTLINE
SEQ_ LEN= SEQ_ LEN, 它将处理向量0的块大小
OUTLINE moreover,
it will work with a block size of vectors of o > TIMELINE
D= D
SEO _ LEN= SEQ_ LEN,
BLOCK _ SIZE _ Q= BI
HEAD DIM= Ct X.
O 向量的数量是多少?
What is this number of vectors of o?
> TIMELINE OUTLINE def
test_op( BATCH _ SIZE, N
Q=(
SEO_ LEN= SEQ_ LEN, 它将是块大小宏, 因此是128个0向量
It will be the block size macro,
soon128vectorsof0 > TIMELINE OUTLINE
BLOCK _ SIZE _ O = BLOCK _ SIZE _ MACRO,
HEAD _ DIM =ctx. HEAD _ DIH, 那么, 让我复制这个函数的签名.
So let me copy the signature of this function.
> TIMELINE OUTLINE det
test_op( BATCH _ SIZE,
BLOCK _ SIZE _ Q BLOCK _ SIZE _ MACRO,
HEAD _ DIM =ctx. HEAD _ DIH, 就在这里.
This is here.
> TIMELINE OUTLINE def
test_op BATCH _ SIZE, NUM_ HEADS, SEQ_ LEN, HEAD_ DIM,
Q=(
@static method def forward (ctx, Q.
HEAD_ DD_ O, 那么, 我们在这里写下来吧.
HEADDIM_ V
BATCH_ SIZE, NUH_ HEAOS, SEO_ LEN,
Solet'swrite it here.
> TIMELINE OUTLINE 254
assert HEAD_ IH_ QHEAD_o IM_ K
tl. store (0_block_ptr, 0_block. to(0. type. elenen t_ty))
class Triton Attention (torch. aut 我觉得这样就可以了.
@static method > TIMELINE OUTLINE 249
HEAD_ DIM_ V= V. shape[-1]
BATCH _ SIZE, NUM _ HEADS, SEQ _ LEN, HEAD _ DIM assert HEAD _ DIH_ Q
HEAD_o IM_ K and HEAD_ DIM_ K
0=torch. empty _like ( Q)
stage=3 if causal else1
grid=lambda args :(
Okay > TIMELINE OUTLINE 267
#cei L( SEQ_ LEN/ BLOCK_ SIZE_ Q)=
HEAD _ DIM_ V= V. shape[-1]
BATCH_ SIZE,
assert HEAD 这个函数接收矩阵0作为输入.
stage-3 if causa This function takes the matrix O.
0=torch. empty_like( Q)
> TIMELINE OUTLINE grid=la
bda args :
HEAD _ DIM_ V= V. shape[-1]
HEAD _ DIM_ Q, HEAD_ DIH_ K= Q. shape[-1], K. shape[-1]
BATCH_ SIZE,
NUM_ HE
EQ_ LEN, H
assert HEAD 因此它是指向矩阵 O的指针.
0= torch. empty _like (
So it's a pointer to the matrix O > TIMELINE OUTLINE grid =la a mbda args :(
259
HEAD_ DIM_ V= V. shape [-1]
BA TCH_ SIZE, NUM 它是指向矩阵 DO的指针,-同时也是一个指向矩阵 D的指针
266
It'sapointer to the DO and it's a pointer to the matrix D pty _like ( Q > TIMELINE OUTLINE 267
grid
HEAD _ DI M_ V= V. shape[-1] 我们将在这个矩阵-D中存储这些 DI元素.
stae-3ifcwherewewill store this D Ielements.
0=torch. empty_like( Q)
> TIMELINE OUTLINE
HEAD _ DIM_ V= V. shape[-1]
HEAD _ DIM_ Q, HEAD_ DIM_ K= Q. shape[-1], K. shape[-1]
BATCH_ SIZE, NUM _ HE 而且, 我们在输出中为每个向量都准备了一个.
st And we have one for each vector in the output.
0=torch. empty_like(0
> TIMELINE OUTLINE bda args :(
259 这就是为什么这个 D的形状是批量大小、头数、序列长度.
that's why the'shape of this D is batch size, number, head sequence length.
> TIMELINE OUTLINE grid =
这意味着, 在注意力机制的输出中, 每个输出元素都对应一个这样的 D it means it's one for each of the output element in the output of the attention.
> TIMELINE OUTLINE
这个 DI实际上位于哪里呢?
stride _0
BATCH_ SIZE= Q. shape[0],
stride,
this Dl, where is it actually?
> TIMELINE OUTLINE NUM_ HEADS= Q. shape[1],
不是这个, 而是那个, 对, 就像 M 一样
it's not this one, it's this one, yeah, like M.
> TIMELINE OUTLINE requires _grad _()
343 因此, 它的形状与 M相同, 正如你所见, 它的大小就是这样的
> TIMELINE OUTLINE
d= Q. stride(1), 所以 它的维度是批量大小、头数和序列长度.
293
297
So batch size, number of heads, and sequence length.
> TIMELINE OUTLINE 296
stride_ V_seq= V. stride(2),
326 BLOCK _ SIZE _ MICRO, BLOCK _ SIZE _ MACRO=32, 128
M. 如果你还记得, 是我们在前向传播过程中保存的矩阵
M, if you remember ;is the matrix that we saved during the forward pass,
> TIMELINE OUTLINE
requires _grad_()
355 它包含子softmax的归一化因子以及最大元素
which includes the normalization factor of the soft max and also the maximum element > TIMELINE OUTLINE cor ch. enot v
-requires _grad _() 但采用的是log-sum-exp 形式.
BAT CI. requires _grad _()
but in log-sum-exp format,
> TIMELINE OUTLINE
requires _grad_()
355 因此, 当我们应用它时,"它会自动为每一行应用最大元素
so that when we apply it will automatically apply the maximum element for each row > TIMELINE OUTLINE torch. e not v (
354-requires _grad _() 并同时进行归一化, 这一点我之前应该已经证明过了.
and also normalize at the same time, which I think I proved previously > TIMELINE OUTLINE torch. e not v
nts Di
0=0 那么, 让我来操作一下
SEQ_ LEN-SEQ_ LEN,
D= D,
OUTLINE BLOCK _ SIZE _ Q BLOCK _ SIZE _ MACRO,
HEAD _ DIM ctx. HEAD _ DIM,
So, let me do it.
> TIMELINE
stage =3 if causal else 1
grid= lambda args :(
#cei L ( SEQ _ LEN / BLOCK_ SIZ 于是, 我们这样写
triton. cdiv( SEQ_ LEN, args
BATCH _ SIZE * NUM_ HEADS,#
kwith
1,# Z in the CUDA launch So, we write it like this.
> TIMELINE OUTLINE M is the log sum exp for the backward pass, one for each query
0= torch. empty _like ( Q)
stage=3if causal else1
grid=lambda args :( 于是, 我们提取出
triton. cdiv( SEQ_ LEN, args
cei L( SEQ_ LEN/ BLOCK_ S
or k with?
BATCH _ SIZE * NUM_ HEADS,# W
1,# Z in the CUDA launch grid So, we extract the...
work with?
> TIMELINE OUTLINE
0=torch. empty _like ( Q)
stage=3if causal else 1
grid =lambda args :( 这个程序的索引
triton. cdiv( SEQ_ LEN, args[
cei L( SEQ_ LEN/ BLOCK _ SIZE BATCH _ SIZE * NUM _ HEADS the index of this program.
> TIMELINE OUTLINE
0=torch. empty _like ( Q)
stage=3 if causal else 1 因此这个程序有两个标识符, 类似于索引.
so'this program has two uh index like identifier.
> TIMELINE OUTLINE
265
0=torch. empty_like( Q) 这相当于 CUDA的标识符, 并且它是沿着第零轴的
8
this is equivalent to the cuda identifier and this is along the axis zero.
> TIMELINE OUTLINE
0=torch. empty _like ( Q)
3ifcausalelse 1 那么, 让我们看看我们在第零轴上启动了什么.
solet'ssee what we uh, what we, what did we launch on the axis zero.
> TIMELINE OUTLINE
HEAD _ DIM =ct X. HEAD _ DIM, 在这个启动网格的第零轴上
so on the axis zero of this launch grid we defined what is the block of vectors of theo > TIMELINE OUTLINE
我们定义了该特定程序将处理的向量块,
that this particular wil program will work with > TIMELINE OUTLINE
而第二轴则决定了该程序将处理哪个批次
ral mean. And the second axis is which batch > TIMELINE OUTLINE. requires _grad _()
test _op ( BATCH _ SIZE, N 以及批次中的哪个注意力头.
tor ch. enpty (
and which head inside-of each batch this particular program will work with.
> TIMELINE OUTLINE
因此这标识了 Q 的 thread ldx So this. identifies the block index of Q,
> TIMELINE OUTLINE stride _ O_head= O. strideili
stride(2)
HEAD _ DIM_ V= V. shape[-1] 即该特定程序将处理 O矩阵中的哪一组向量
so which group of vectors in the O matrix this particular program will work with.
> TIMELINE OUTLINE
HEAD _ DIM_ V = V. shape[1]
HEAD DIMQ,
HEAD_ DIM_ K= Q. shape[-1], K. shape[-1]
BATCH_ SIZE, NUM _ HEADS, SE 这里之所以称之为 Q
assert HEAD _ DIM_ Q=
stage =3if causal else
0=torch. empty _like ( Q )
Here it's called Q, I believe,
> TIMELINE OUTLINE rae.
HEAD _ DIM_ V= V. shape[-1] 是因为我直接从原始代码中复制过来的, 他们称之为 Q.
because l copied it from the original code where they call it Q > TIMELINE OUTLINE
HEAD _ DIM _ V= V. shape[-1]
HEAD DIM_ Q, HEAD DIM K
Q. shape[-1], K. shape[-1]
BATCH_ SIZE 不过, 我其实也可以把它称为0.
assert but l could have eventually also called it O > TIMELINE OUTLINE
stag
HEAD _ DIM_ V = V. shape[1] 简而言之, 这意味着我们正针对这个程序进行定义.
Sowe define so basically, this means that we are for this program.
> TIMELINE OUTLINE
HEAD _ DIM_ V = V. shape[1]
HEAD DIM Q.
HEAD DIM_ K
Q. shape[-1], K. shape[-1] 我们需要跳过一些查询向量, 这些向量要么已经被
we need to skip some query vectors that have been already > TIMELINE OUTLINE
HEAD _ DIM_ V = V. shape[1]
HEAD DIM Q, HEAD DIM_ K
Q. shape[-1], K. shape[-1] 其他并行程序处理过, 要么即将被处理.
or that will be. or have been already processed by other programs in parallel.
> TIMELINE OUTLINE
HEAD _ DIM_ V = V. shape[1] 因此, 我们只会处理 O中具有以下索引的查询
6
Sowe will only block with a number of guer y vectors inside of o > TIMELINE OUTLINE
HEAD _ DIM_ V = V. shape[1]
HEAD DIM Q, HEAD DIMK
= Q. shape[-1], K. shape[-1]
BATCH_ SIZE, NUM_ HEADS, SEO_ LEN, HEAD_ DIM = Q. 向量块
assert HEAD _ DIH_ Q
HEAD_ DIM_ K and HEAD_ DIH
> TIMELINE OUTLINE arid -
HEAD _ DIM_ V= V. shape[-1] 假设查询块的大小是128, 这是我们之前的定义方式.
So imagine. that the query block size is, I think it's 128, the way we have defined it.
> TIMELINE OUTLINE
HEAD _ DIM _ V = V. shape[1]
HEADDIM_ Q,
HEAD DIM K
= Q. shape[-1], K. shape[-1]
BATCH_ SIZE, 但为了简化理解, 假设它是4.
assert HE AD_ D
stra. py but. suppose it's a four for simplicity.
stage =3if cat
> TIMELINE OUTLINE
HEAD _ IM_ V = V. shape[1]
HEAD DIM Q, HEAD DIM K
= Q. shape[-1], K. shape[-1]
BATCH_ SIZE, NUM_ HEADS, SEQ _ LE 所以这个值将会是.
assert HEAD _ DIM _ Q HEAD _ DIM
stage=3if causal else 1
0=torch. empty _like ( Q)
so this one will be.
> TIMELINE OUTLINE arid -
HEAD _ IM_ V = V. shape[1]
HEAD DIM Q
HEAD DIMK
Q. shape[-1], K. shape[-1]
BATCH_ SIZE, 那么查询向量的数量是多少呢?
assert HEAD > TIMELINE OUTLINE stage
HEAD _ DIM_ V = V. shape[1]
HEAD DIM_ Q.
HEAD DIM
Q. shape[-1], K. shape[-1] 序列长度即查询向量的数量, 我们可以想象这些查询向量是.
sequence length, number of query vectors we have so some of imagine the query vectors are.
> TIMELINE OUTLINE
HEAD _ DIM_ V = V. shape[1]
HEAD DIM_ Q, HEAD DIMK
Q. shape[-1], K. shape[-1]
BATCH_ SIZE 总的来说, 真体数量我也不确定.
assert H
stor h. ptyuk in total they are, i don't know.
> TIMELINE OUTLINE
HEAD _ DIM _ V = V. shape [1]
HEAD DIM_ Q,
HEAD DIMK Q. shape [-1], K. shape[-1] 假设总共有64个, 其中32个由其他程序处理
265
0
let'ssay64. and 32 will be managed by other programs,
> TIMELINE OUTLINE 266
267
HEAD _ DIM_ V= V. shape[-1]
HEADDIM_ Q,
HEAD DIM
Q. shape[-1], K. shape[-1] 那么这一特定批次的索引将会是33、34、35和36.
so this particular of skew will be equal to 33, 34,, 35 and 36.
> TIMELINE OUTLINE
HEAD _ DIM_ V= V. shape[-1]
HEAD DIM_ Q,
HEAD DIM_ K
Q. shape[-1], K. shape[-1]
BATCI 这表示在输出矩阵0的所有向量中
This tells me which query vectors or which vectors in the output O matrix > TIMELINE OUTLINE
HEAD _ DIM_ V = V. shape[1]
HEAD DIMQ,
HEAD DIM_ K
Q. shape[-1], K. shape[-1] 当前程序将处理哪些查询向量或具体哪些向量
among all the vectors in'the O matrix this particular program is going to work with.
> TIMELINE OUTLINE
HEAD _ DIM_ V= V. shape[-1] 接下来我们还要提取批次的索引, 这告诉我们当前程序将处理哪个批次
Okay, so then we. extract also the index of the batch, which tells us which batch > TIMELINE OUTLINE stage =3if causalelse1
HEAD _ IM_ Q, HEAD_ DIH_
Q. shape[-1], K. s
pe[-1]
BATCH _ SIZE 以及每个批次中的哪个注意力头
and which head. in. each batch this particular program is going to work with > TIMELINE OUTLINE
def forward (ctx, Q, K, V, causal, soft max _scale ):
HEAD _ DIM _ Q, HEAD _ DIH_ K
HEAD_ DIM_ V
V= V. shape[-1]
BATCH 这正是我们启动网格的第一个维度.
.- which is the dimension one of our launch grid assert > TIMELINE OUTLINE stage =3if causal else 1
def forward (ctx, Q, K, V, causal, soft max _scale ):
HE AD_ DIM_ Q, HEAD _ DIH_
Q. shape[-1], K. sh
ape[-1]
HEAD_ DIM _ V BATCH _ SIZE, 接着, 我们定义维度的偏移量
And then we define the offset of the dimension > TIMELINE OUTLINE stage =3 if causal else 1
@static method def forward (ctx, Q, K, V, causal, soft max _scale ):
HEAD _ DIM _ Q,
HEAD _ 因为需要加载每个向量的所有维度.
BATCH because we need to load all'the dimensions of each vector.
> TIMELINE OUTLINE 67
0=torch. empty_like( Q)
258
259
@static method def forward (ctx, Q, K, V, causal, soft max _scale ): 因此, 这是一个向量, 指示我们需要从每个向量加载哪些维度
these are the it's a'vector that tells which dimensions we need to load from each vector > TIMELINE OUTLINE 0=torch. empty_like( Q)
def forward (ctx, Q, K, V, causal, soft max _scale ):
static method HEAD _ DIL_ V= V. sh
HEAD_ DIM_ Q, HEAD_ DIH_ K 而我们将加载所有这些维度,
BATCH _ SIZE, NUM _
assert HEAD _ DI H_ Q= HEAD and we will load all of them.
> TIMELINE OUTLINE 0=torch. e=pty_like( Q)
258
259
@static method def forward (ctx, Q, K, V, causal, soft max _scale ): 因此, 我们不在注意力头维度上进行划分, 而是仅在序列长度维度上划分
So we don't divide on the head dimension, We just divide on the sequence length dimension,
> TIMELINE OUTLINE 0=torch. empty_like( Q)
def forward (ctx, Q, K, V, causal, soft max _scale ):
HE AD_ DIM_ O,
HEAD _ DI_ V 由多个程序共同分担加载任务.
BATCH _ SIZE,
assert m E opm the load among multiple programs.
> TIMELINE OUTLINE 0= torch. empty_like( Q)
@static method 在这部分视频中你会看到, 当我们编写反向传播时
You will see in this part. of. the video, so when we are writing the backward pass,
> TIMELINE OUTLINE
@static method 不会像前向传播那样使用make _block _pointer.
that we will not be using the. make block pointer like we did during the forward pass.
> TIMELINE OUTLINE
K. block_ptr,
obtock
_btock_ptr, 所以这里的这个函数.
soft max BLOCK _ SIZE _ O,
OUTLINE BLOCK _ SIZE _ XV,
STAGE So this function here.
TIMELINE off s_q.
the offsets for the to l
ns in the K and V sequence to process 我们将直接通过使用步幅来进行索引操作.
We will work. directly with indexing by using the strides.
> TIMELINE OUTLINE 187
_attn _bwd _preprocess (
SEO_ LEN,
D, 那么, 我们开始吧
8
HEAD_ DIH:tl. const expr,
BLOCK _ SIZE _ Q:tl. const expr,
block _index _qtl, progro=_1d(e)
So let's do it.
> TIMELINE OUTLINE 252
offs_q=block _1ndexq+ BLOCK_ SIZE _ O +tl. arange(θ, BLOCK_ SIZF_0)
class Triton Attention (torch. auto grad. Function ):
"现在, 我们来加载○的一个行块.
@static metho HEAD _ DIM BATCH _siz, So let's. load-a single block of rows of O.
HEAD _ DIM _ V= V. shape > TIMELINE OUTLINE assert HEAD _ DIH_ Q == HEAD _o IM_ K and HEAD_ DIH_ K == HEAD_ DIM _ V
class Triton Attention (torch. auto grad. Function ):
def forward (ctx @static method 的形状与 Q 相同
HE AD_ DIM_ O,
which L want. to remind you has the same shape as Q HEAD _ DIM _ V
V. shape [-]
> TIMELINE OUTLINE as Sert HEAD _ DIM_ Q = HEAD _ DIM _ K and HEAD_ DIM_ K == HEAD_ DIM_ V
259 index _batch _head * HEAD _ DIM * SEQ_ LEN 正因如此, 我们可以称这个块的大小为 Q块大小
264
class Triton At tent and that's why we can call it block size Q.
> TIMELINE OUTLINE 266
67
@static method def forward (ctx, Q, K, V, causal, soft max _scale ):
index _batch _head * HEAD _ DIM * SEQ_ LEN 我们正在加载的 O块就是 O本身
So the O block that we are loading is 0,
> TIMELINE OUTLINE @static method def forward (ctx, Q, K, V, causal, soft max _scale ):
index _batch _head * HEAD _ DIM * SEQ _ LEN 这里的加载函数接收一个指向待加载内容的指针.
so the load function accepts a pointer to what it should load.
> TIMELINE OUTLINE ef forward (ctx, Q, K, V, causal, soft max _scale ):
259 实际上, 它接收的不是单一指针, 而是一个指针数组
6
So actually not a pointer, it accepts an array of pointers > TIMELINE OUTLINE def forward (ctx, Q, K, V, causal, soft max _scale ):
或者说是多维指针数组, 以便于加载多维数据.
or a multi dimensional array of pointers in case you want to load a multidimensional data.
> TIMELINE OUTLINE ef forward (ctx, Q, K, V, causal, soft max _scale ):
实际上,
Toad 函数也支持加载二维数据
So actually load also allows you to load two dimensional data.
> TIMELINE OUTLINE forward (ctx, Q, K, V, causal, soft max _scale ):
259 在此例中, 我们将加载二维数据, 即 O的一个行块.
In this case, we are going to load two-dimensional data, which is a block of rows of o,
> TIMELINE OUTLINE ef forward (ctx, Q, K, V, causal, soft max _scale ):
Batch _head * HEAD _ DIM * SEQ _ LEN 这个行块是一个张量, 其形状为块大小 Q which should be a block, a tensor of the shape, block size Q, in this case,
> TIMELINE OUTLINE forward (ctx, Q, K, V, causal, soft max _scale ):
+index _batch _head * HEAD _ DIM * SEQ _ LEN +off s_q[:, None]* HEAD_ D】
offs_din
ne,:] 与头维度相乘的结果.
265
264
multiplied by the other dimension being head dimension.
> TIMELINE OUTLINE def forward (ctx, Q, K, V, causal, soft max _scale ):
index _batch _head * HEAD _ DI H * SEQ_ LEN 但我们需要告诉函数在◎矩阵中具体哪个位置找到这个行块.
But we need to tell it where in this O matrix it needs to find this one.
> TIMELINE OUTLINE def forward (ctx, Q, K, V, causal, soft max _scale ):
index _batch_head * HEAD_ DIM * SEQ_ LEN 首先, 我们需要根据其他程序将要处理的批次和头数
8
First of all, we need to skip some batches and some heads based on > TIMELINE OUTLINE ef forward (ctx, Q, K, V, causal, soft max _scale ):
+index _batch _head * HEAD IM * SEQ_ LEN
(parameter ) HEAD _ DIM : const expr 262
+offs_q[:, None]* HEAD_ DIH
offs_din N 跳过相应的部分,
what the head and the batch that will be processed by other programs.
> TIMELINE OUTLINE def forward (ctx, Q, K, V, causal, soft max _scale ):
(parameter ) HEAD _ DIM :const expr index _batch _head * HEAD _ DIM * SEQ _ LEN 因此依据本程序要处理的批次和头索引
So based on the index that this program will process of the batch and the head,
> TIMELINE OUTLINE def forward (ctx, Q, K, V, causal, soft max _scale ):
index _batch head * HEAD _ DIM * SEQ _ LEN 我们需要略过所有其他的批次和头.
We need to skip all the other batches and heads.
> TIMELINE OUTLINE def forward (ctx, Q, K, V, causal, soft max _scale ):
+index _batch _head * HEAD _ DIM * SEQ_ LEN
+offs_q:, None
+offs_din 我们来写出这个张量的形状.
Triton Attention (to Let'swrite the shape of this tensor.
> TIMELINE OUTLINE def forward (ctx, Q, K, V, causal, soft max _scale ):
@static method
index _batch _head * HEAD _ DIH * SEQ _ LEN 因此, 张量的形状为:批次大小、头数、序列长度
So the O tensor has a shape, batch size, number of heads, then sequence length,
> TIMELINE OUTLINE def forward (ctx, Q, K, V, causal, soft max _scale ):
+index _batch _head * HE AD_ DIH* SEQ_ LEN
+offs _q[:, None]* HEAD_ DIM +offs_dim[ None,:] 以及头维度.
Triton Attention (torch. auto gr ac and'then head dimension.
> TIMELINE OUTLINE @static method def forward (ctx, Q, K, V, causal, soft max _scale ):
每个块和每个头将包含序列长度乘以头维度数量的元素.
Each block and each head will have sequence length multiplied by head dim number of items.
> TIMELINE OUTLINE f forward (ctx, Q, K, V, causal, soft max _scale ):
index _batch _head * HEAD _ DIH SEQ _ LEN 那么根据我们的索引, 需要跳过多少元素呢?
Sobased on our index, we skip how many items?
> TIMELINE OUTLINE 267
def forward(ctx, Q, K, V, causal, soft max _scale ):
(parameter ) HEAD _ DIM :const expr index _batih_head* HEAD_ DIM * SEQ_ LEN 我们的索引值乘以头维度再乘以序列长度.
our index multiplied by head dimension multiplied by sequence length > TIMELINE OUTLINE forward (ctx, Q, K, V, causal, soft max _scale ):
+ Index _batch _head * HEAD _ DIM * SEQ _ LEN +of fs_q[:, None]* HEAD_ DIM
+offs_dim[ None,:] 具体来说
s Triton Attention (torch. auto grad. Fu
So'what I mean is this,
> TIMELINE OUTLINE @static method def forward (ctx, Q, K, V, causal, soft max _scale ):
+index _batch _head * HEAD _ DIH* SEQ_ LEN
+offs_q:, No 批次0和头0将包含序列长度
the batch zero and the head zero will have a sequence length > TIMELINE OUTLINE def forward (ctx, Q, K, V, causal, soft max _scale ):
+index _batch _head * HEAD _ DIH* SEQ_ LEN
offs_din N 乘以头维度数量的元素.
Triton Attention (
multiplied by head dimension items.
> TIMELINE OUTLINE 267 @static method def forward (ctx, Q, K, V, causal, soft max _scale ):
+index _batch _head * HEAD _ DIH * SEQ_ LEN
+offs_q[:, None]
offs_din No 批次0 的头 2同样如此.
and'the batch 0 and head 2 will also have the same number of head > TIMELINE OUTLINE forward (ctx, Q, K, V, causal, soft max _scale ):
index _batch _head * HEAD _ DIH* SEQ_ LEN 那么, 我们需要从○张量的起始位置跳过多少元素呢?
8
so how many items-sequence length multiplied by head dimension -
> TIMELINE OUTLINE def forward (ctx, Q, K, V, causal, soft max _scale ):
+index _batch _head * HEAD _ DIH* SEQ_ LEN
D
+offs_q:, Non 答案是序列长度乘以头维度.
dowe need to skip from the starting of the O tensor?
> TIMELINE OUTLINE def forward (ctx, Q, K, V, causal, soft max _scale ):
+1ndex_batch_head * HEAD_ IM* SEQ_ LEN
D 这等于当前批次和头指示器的索引值
it is equal to the index of the current batch and head indicator.
> TIMELINE OUTLINE forward (ctx, Q, K, V, causal, soft max _scale ):
index _batch head * HEAD _ DIH * SEQ_ LEN 由于这个索引同时标识了批次中的头
264
265
sobecause this index indicates both the head in the batch > TIMELINE OUTLINE 6
def forward(ctx, Q, K, V, causal, soft max _scale ):
+index at ch_head * HEAD_ DIH * SEQ_ LEN
+offs_q[:, None ]* HE AD_ D]
+offs _dim [ None,:] 和每个批次内部的头
( BLOCK _ SIZE Q, HEAD and ’the head inside of each batch,
> TIMELINE OUTLINE def forward (ctx, Q, K, V, causal, soft max _scale ):
@static method
+index atch _head * HE AD_ DIH * SEQ_ LEN
262
+offs_dim[ N
+offs_q:, 因为它已经是头和批次的乘积
because it's already the product of the head and the batch, so how many we skip,
> TIMELINE OUTLINE def forward (ctx, Q, K, V, causal, soft max _scale ):
index _batc lhead* HEAD_ DIH * SEQ_ LEN 所以根据这个索引, 我们需要跳过多少元素呢?
264
265
class Triton Attention (torch.
indicated by the this index?
> TIMELINE OUTLINE 266 @static method def forward (ctx, Q, K, V, causal, soft max _scale ):
266
index_batch_head* HEAD_ DIM* SEQ _ LEN 当我们定位到当前批次和当前头的起始位置后
and after we p point to this starting point of the current batch and the current head > TIMELINE OUTLINE ef forward (ctx, Q, K, V, causal, soft max _scale ):
(parameter) SEQ _ LEN : Any of fs_din [ No 需要选择一个二维张量,
Triton We need to select a two-dimensional tensor > TIMELINE OUTLINE @static metho
def forward (ctx, Q, K, V, causal, soft max _scale ):
+index _batch _head * HEAD _ DI H* SEQ _ LEN
(parameter) SEQ_ LEN: Any
+offs_ql:,
offs_din 其中行的偏移量由offsq指示
Where the offsets are indicated for the rows by off sq,
> TIMELINE OUTLINE def forward (ctx, Q, K, V, causal, soft max _scale ):
258
Index_batch_head * HEAD_ DIH* SEQ _ LEN 这就是为什么会有这个一一手 我不确定该怎么称呼它.
and that's why we have this one, the - I don't know what this is called.
> TIMELINE OUTLINE def forward (ctx, Q, K, V, causal, soft max _scale ):
258 index _batch _head * HEAD _ DIM* SEQ_ LEN 这是索引l, 分号索引, 它指示了ofsg中的所有向量
this is the index, semicolon index that tells all these vectors in of sq,
> TIMELINE OUTLINE def forward (ctx, Q, K, V, causal, soft max _scale ):
并额外增加了一个列维度, 这些列将由of sd im 表示.
with an additional dimension for the columns, and these columns will be the ofs dim.
> TIMELINE OUTLINE ef forward (ctx, Q, K, V, causal, soft max _scale ):
index _batch _head * HEAD _ DIM * SEQ _ LEN 简而言之, 这将选择一个具有以下形状的张量:
so basically, this will'select a tensor of the following shape :
> TIMELINE OUTLINE def forward (ctx, Q, K, V, causal, soft max _scale ):
在这个包含批次大小和头数量的大张量内部
inside of this big tensor that includes pet size and number of heads > TIMELINE OUTLINE def forward (ctx, Q, K, V, causal, soft max _scale ):
+index _batch _head * HEAD _ DI H* SEQ_ LEN 这就是我们正在做的操作.
Triton Attention (torch. au This is what we are doing.
> TIMELINE OUTLINE def forward (ctx, Q, K, V, causal, soft max _scale ):
@static method
+index _batch _head * HE AD_ DIH * SEQ_ LEN
+offs _q[:, None]* HEAD_ DIM +offs_dim[ None,:] 也就是说
( BLOCK _ SIZE Q, HEAD_ DIH)
8
Triton Attention (torch. auto grad. Functi or So we are saying :
> TIMELINE OUTLINE 267 @static method def forward (ctx, Q, K, V, causal, soft max _scale ):
259
index_batch_head * HEAD_ DIH * SEQ _ LEN 我们要在一个由四个维度组成的张量中, 选择这样一个大小的张量
select a tensor of this size inside of one that is made up of four dimensions,
> TIMELINE OUTLINE def forward (ctx, Q, K, V, causal, soft max _scale ):
Index _batch _head * HEAD _ DI H* SEQ_ LEN 跳过其他程序将处理的所有批次和头的元素.
skipping the elements of all the batch and heads that will be processed by other programs.
> TIMELINE OUTLINE def forward (ctx, Q, K, V, causal, soft max _scale ):
59
index_batch_head * HEAD_ DIH * SEQ _ LEN 我总是用"程序"来表述, 因为在 Triton 中它们被称为程序
I always talk in terms of programs because in Triton these are called programs,
> TIMELINE OUTLINE def forward (ctx, Q, K, V, causal, soft max _scale ):
+index _batch _head * HEAD _ DIH * SEQ_ LEN 而在 CUDA中, 你会称它们为内核
in CUDAyou would refer to them as kernels.
> TIMELINE OUTLINE @static metho
def forward (ctx, Q, K, V, causal, soft max _scale ):
+1ndex_batch_head * HEAD _p IH* SEQ_ LEN 没错, 所以这一部分已经完成了,
8
Triton Attention (torch. auto gra right, so this one is done.
> TIMELINE OUTLINE def forward (ctx, Q, K, V, causal, soft max _scale ):
@static method
+off s_ql:,
+off s_di n[ N 希望我已经解释得足够清楚了.
ihope it is decently clear.
> TIMELINE OUTLINE @static method def forward (ctx, Q, K, V, causal, soft max _scale ):
6
Lndex_batch_head * HEAD _ DIH* SEQ_ LEN 嗯 好的, 那么我们也加载一个单独的d块吗?
264
265
um, all right, so then we also load a single block of d?
> TIMELINE OUTLINE 266
6
def forward(ctx, Q, K, V, causal, soft max _scale ):
+index _batch _head * HEAD _ DIH* SEQ_ LEN
+offs_q[:, None]* HEAD_ DIH
+offs_dim [ No ne,:] 哦, 同样的方式
o in the same way,
> TIMELINE OUTLINE def forward (ctx, Q, K, V, causal, soft max _scale ):
@static method
index _batch _head * HEAD _ DIM * SEQ _ LEN 因为我们要从整个序列长度中加载一组向量, 也包括d吗?
because we are going to load a group of vectors from all the sequence length, also from d?
> TIMELINE OUTLINE def forward (ctx, Q, K, V, causal, soft max _scale ):
+index _batch _head * HEAD _ DIM * SEQ _ LEN
270
). to(tl. float32] 哦, 还有d呢?
271
272
class Triton Attention (torch. auto grad.
O, and the d?
> TIMELINE OUTLINE 274
275
@static method def forward (ctx, Q, K, V, causal, soft max _scale ):
index _batch _head * HEAD _ DIM * SEQ_ LEN 的形状与. 相同, 而. 的形状又与q相同
271
272
ohas the same shape as o, which has the same shape as q,
> TIMELINE OUTLINE 273
274
75
def forward(ctx, Q, K, V, causal, soft max _scale ):
index _batch _head * HEAD _ DIM * SEQ_ LEN 这就是为什么我们可以使用threadldx.
Trito
and that's why we can use the The block index.
> TIMELINE OUTLINE def forward (ctx, Q, K, V, causal, soft max _scale ):
@static m
index _batch _head * HEAD _ DIM * SEQ_ LEN 我们称之为 Q, 因为它们是等价的, 具有相同的形状.
we call it Q, because it's equivalent, because they have the same shape.
> TIMELINE OUTLINE forward (ctx, Q, K, V, causal, soft max _scale ):
+index batch _head *
HEAD _ DIH * SEQ _ LEN 好的, 那么如何计算这个di 元素呢
Okay, and how to compute this di element.
> TIMELINE OUTLINE 275 @static method def forward (ctx, Q, K, V, causal, soft max _scale ):
+index _batch _head * HEAD _ DIH * SEQ_ LEN
+offs_q[:, None]
+offs_din [ Non
* HEAD_ DIH
278
. to(tl. float32) 嗯, 这在论文中有详细说明
OUTLINE 275
274
@static method Well, it's written in the paper.
> TIMELINE def forward (ctx, Q, K, V, causal, soft max _scale ):
所以如果我们深入探讨这个问题, 具体是什么呢?
274
273
class Triton Attention So if we go in the inthe What is it man?
> TIMELINE OUTLINE 275 @static method def forward (ctx, Q, K, V, causal, soft max _scale ):
21 :
Write d Qd Q+rd SKRB, xd to HBM.
22:
On chip. compute d K
d K+d SQ∈ RBxd
23:
end for
24:
Write d K←d K, d V
26: Return d Q. d K. d V
25:end for
if we go here.
21 :
Write d Qd Q+rd SKRB, xd to HBM.
22: 示了如何根据给定的do块和. 块来计算每个di.
it shows you'how. to. compute the di of each given a block of do and a block of o.
21 :
Write d Qd Q+d SKRB, xd to HBM.
22:
On chip. compute dkdk +rd SQe RB. xd
23:
end for
25:end for
24:
Writed K←d K, d V;d V;to HBMI.
26: Return d Q. d K. d V
21 :
Write d Qd Q+d SKRB, xd to HBM.
22: 也就是行求和, 就是 即按行进行累加
21:
Write d Qd Q+rd SKRB, xd to HBM.
22:
23:
Writedkdk付
end for 24:
25:end for
21 :
Write d Qd Q+rd SKRB, xd to HBM.
22
On chip. compute d K 对于○知 矩阵中的每一 个向量 我们都会得到一个逐元素乘积的和
21:
Write d Qd Q+rd SK RB, xd to HBM.
2:
Onchip. compute d Kd K+rd SQ∈ RBxd
21 :
Write d Qd Q+d SKRBxd to HBM.
22:
On chip, compute dkdk +rd SQe B. xd
23:
end for
25:end for
24:
Writed Kd K, d Vd V;to HBM.
26: Return d Q. d K. d V
21 :
Write d Qd Q+rd SKRB, xd to HBM.
22:
On chip. compute dk 是矩阵乘法, 而是逐元素相乘
23:
endf好
24:
21:
Write d Qd Q+rd SKRBxd to HBM.
22:
On chip. compute dkdk +rd SQe RB. xd
23:
end for
25:end for
24:
Writed Kd K, d Vd V;to HBM.
26: Return d Q. d K. d V
21 :
Write d Qd Q+rd SKRBxd to HBM.
22: 一个矩阵的每个元素
23:
endfor 24:
Writed Kd K. d V
25:end for
21 :
Write d Qd Q+rd SKRBxd to HBM.
22:
On chip. compute dk
23:
endfor 二个矩阵的对应元素相乘.
24:
Writed K
25:endf
21 :
Write d Qd Q+d SKRB, xd to HBM.
22:
On chip. compute dkdk +rd SQe B. xd
23:
end for
25:end for
24:
Writed Kd Kd V d Vto HBMI.
26: Return d Q. d K. d V
21:
Write d Qd Q+d SKRB, xd to HBM.
22:
23:
endfor
24:
Writed K
21 :
Write d Qd Q+rd SKRB, xd to HBM.
22:
On chip,. compute dkdk+rd SQB. xd
23:
end for
25:end for
24:
Writed K;d K, d V;d V;to HBMI.
26: Return d Q. d K. d V
21:
Write d Qd Q+rd SKRB, xd to HBM.
22:
23:
end for 24:
Write d K
25:end for
index _batch _head * HEAD _ DIM * SEQ _ LEN +off s_q:,
* HEAD _ DIM
). to(tl. f 好的, 接下来我们计算这个 DI块.
OUTLINE
275
274
@static method Okay, so we compute this Dl block.
> TIMELINE def forward (ctx, Q, K, V, causal, soft max _scale ):
267
+index_batch_head * HEAD_ DIM * SEQ _ LEN 它的形状将是 Q 的块大小, 因为每个向量都会对应一个求和结果.
which will have shape block size Q, because we will have one sum for each vector.
> TIMELINE OUTLINE def forward (ctx, Q, K, V, causal, soft max _scale ):
+of fs_dim[
. to(tl. float32) 接下来我们需要将结果存储在某个位置
8
class Triton Atte Then, well, we need to store it somewhere,
> TIMELINE OUTLINE def forward (ctx, Q, K, V, causal, soft max _scale ):
@static method
+of fs_dim[
. to(tl. float32)
272
273 因此需要计算它在 D矩阵中的具体存储位置.
8
so we need to calculate where to store it inside of the D matrix.
> TIMELINE OUTLINE def forward (ctx, Q, K, V, causal, soft max _scale ):
+of fs_dim[
). to(tl. float32)
_block 我记得 D矩阵的形状与 M相同
Well, t
the D matrix is, Tremember correctly, has the same shape as M > TIMELINE OUTLINE def forward (ctx, Q, K, V, causal, soft max _scale ):
DIM * SEQ _ LEN 因此它的大小应该与批次大小一致.
Dbtock=tlsudobtock*obu So it should be batch size.
> TIMELINE OUTLINE ass Triton Attention (torch. auto grad. Function ):
+of fs_q:,
DIH* SEQ_ LEN
). to(tl. float32)
+off s_din 包含头数和序列长度两个维度.
ek t. anumber of. heads and a sequence length.
> TIMELINE OUTLINE lass Triton Attention (torch. auto grad. Function ):
HEAD _ DIH* SEQ_ LEN 因此我们需要根据当前的threadldx立方体
273
272
bc=tdo so. we. need to. select. the right batch > TIMELINE OUTLINE rad. Function ):
* SEQ _ LEN +off s_ql:,
+offs_dim 在序列长度中选择正确的批次,
270
. to(tl. float32)
and the right. head and also. the right position inside of the sequence length 271 > TIMELINE OUTLINE Triton Attention (torch. auto grad. Function ):
+off s_q:, None DIM* SEQ_ LEN
). to(tl. float32)
+offs_dim[ N 正确的头以及正确的位置.
based on. the-block index cube that we have.
OUTLINE
D_block > TIMELINE lass Triton Attention (torch. auto grad. Function ):
D _block class Triton Attention (to r 好的接下来让我来索引.
@static method > TIMELINE OUTLINE HEAD _ DIM _ V= V. shape[-1]
HEAD_ DIM_ Q, HEAD_ DIM_ K
271
272
Compute the D block 好的, 没
(d0 block*0_block, axis=1) 问题, 既然我们已经完成子这一步, 就像之前一样, 我们可以直接跳过继续
okay, All right, because we already, so we skip again, just like before.
> TIMELINE OUTLINE HEAD _ DIM_ V= V. shape[-1]
Compute the D block Store the D blo 我们知道· D·的大小是这样的
D_block _ptr s= D
class Triton Attention (torch We know that the D isof this size.
> TIMELINE OUTLINE 280
def forward(ctx, Q, K, V, causal, soft max _scale ):
@static method I-11
每个批次和每个头都会有序列长度数量的元素.
Each batch and each head will have sequence length number of elements.
> TIMELINE OUTLINE def forward (ctx, Q, K, V, causal, soft max _scale ):
pute the D block 因此, 我们需要从张量的起始位置跳过的元素数量
So how many number of elements we need to skip from the starting of the tensor > TIMELINE OUTLINE def forward (ctx, Q, K, V, causal, soft max _scale ):
274 是序列长度乘以批次大小与头编号的组合索引.
issequence length multiplied by the combined index batch size head number.
> TIMELINE OUTLINE
272
271 此外;我们还需要根据threadldx队列跳过一些查询
And plus, we. need to also skip some queries based on our block index queue,
> TIMELINE OUTLINE def forward (ctx, Q, K, V, causal, soft max _scale ):
272
271 个跳过操作已经在偏移队列中完成了, 月 所以只需加上偏移队列的值即可
and this skipping is already done inside of off queue, so we add off queue.
> TIMELINE OUTLINE _scale ):
272 然后, 当我们计算出应该存储这个 DI块的索引位置时
8
And then, once we have computed the index where we should store this Dl block,
> TIMELINE OUTLINE
def fonwardctx, Q, K, V, causal, soft (max _scale ):
Store the D bloch 我为什么要称它为 D块呢?
D_block_ptrs= D+
class Triton Attention (torch. auto grad. Functi why did I even call it D block?
> TIMELINE OUTLINE 280
def forward(ctx, Q. K, V, causal, soft max scale ):
@static method
Compute the D block D_block=tl. sum(do_block *0_block, axis=1)# Sha Store the ( BLOCK _ SIZE _ Q,
D_block _ptrs 让我们把它存储起来, 月 所以让我
Triton Attention (torch. auto grad.
Let's store it, so let me..
> TIMELINE OUTLINE def forward (ctx, Q, K, V, causal, soft max _scale ):
@static method
Compute the D block
273
_block=tl. s 我并没有称它为 D块, 我想这是原代码中已有的命名, 但这里指的是 DI I didn't call it D block, I think it was already in the original code, but this is Dl.
> TIMELINE OUTLINE def forward (ctx, Q, K, V, causal, soft max _scale ):
Compute the D block Store (do_block*0_block, axis=1)# Shape :( BLOCK _ SIZE _ Q,)
_block 这个大矩阵 D*实际上包含了所有 DI,
And this big matrix Dis actually the matrix that includes all the Dl.
> TIMELINE OUTLINE ef forward (ctx, Q, K, V, causal, soft max _scale ):
D_block
block*0_block, axis=1) Shape :( BLOCK _ SIZE _ Q,) 每个 Dl 对应序列长度中的一个token.
one for each'token in the sequence length.
> TIMELINE OUTLINE @static meth def forward (ctx, Q, K, V, causal, soft max _scale ):
Compute the block D _block =tl. s
Store the bloc
m(do_block *0_block, axis=1)# Shape :( BLOCK _ SIZE _0,)
tl. store( D_bloc
_block ptrs= 好了, 预处理工作已经完成
lass Tri to All right, so the pre-processing has been done.
> TIMELINE OUTLINE @static met def forward (ctx, Q, K, V, causal, soft max _scale ):
Compute the block # Store the
D _block =tl
(do_block*0_block, axis=1)# Shape :( BLOCK _ SIZE _ Q,)
_block ptr 现在我们需要准备两个for 循环.
ass Triton A Now we need to prepare the two for loops.
> TIMELINE OUTLINE @static methoc
def forward(ctx, Q, K, V, causal, soft max _scale ):
_block 正如我之前提到的, 我们将执行两个for 循环:
As you remember I said before, we will be doing two for loops :
> TIMELINE OUTLINE forward (ctx, Q, K, V, causal, soft max _scale ):
_block lock*0_block, axis=1) Shap 个循环中, 我们固定查询(guery)并遍历所有键(key )和值(value ):
one in which we fix the query and we iterate through all the keys and values,
> TIMELINE OUTLINE def forward (ctx, Q, K, V, causal, soft max _scale ):
273
D_block =
Compute the block *0_block,
axis=1)
( BLOCK_ SIZE_0, 第二个循环中;我们固定键值块(key-value block )并遍历所有查询.
and one in which we fix the key and value block and we iterate through all the queries > TIMELINE OUTLINE def forward (ctx, Q, K, V, causal, soft max _scale ):
273
block
( BLOCK_ SIZE_0, 在编写代码的过程中, 我会一直展示论文中的公式, 所以不用担心.
And while coding it, I will always show you the formula from the paper, so don't worry.
> TIMELINE OUTLINE ef forward (ctx, Q, K, V, causal, soft max _scale ):
d ef
test_op( BATCH _ SIZE, NUM _ HEADS, SEQ_ LEN, HEAD_ DIM, causal, dtype=torch. float16):
torch. enpty(
( BATCH_ SIZE, N
normal_(mean =0. 0, 让我们开始下一次迭代吧.
. requires_grad_()
Let'sstartwiththenextiteration.
> TIMELINE
OUTLINE
torc
( BATCH SIZE. NUM HEADS. SEO LEN. HEAD DIM). dtve=dtvoe. device="cuda
npty (
test _op ( BATCH _ SIZE, NUM _ HEADS, SEQ _ LEN, HEAD _ DIM, causal, dtype=torch. float16): 首先 我们为下一次送代创建启动网格.
So first we create the launch grid for the next iteration.
> TIMELINE OUTLINE 380
BATCH SIZE. NUM HEADS. SEO LEN. HEAD DIM). dtv De=dtv De. device =cuda
d ef
test_op( BATCH _ SIZE, NUM _ HEADS, SEQ_ LEN, HEAD_ DIH, causal, dtype=torch. f Loat16):
torch. enpty(
( BATCH_ SIZE, NUM_ HE
normal_(mean =0. 0, 启动网格始终保持不变
. requires_grad_()
The launch grid is always the same,
> TIMELINE
OUTLINE
torcl
( BATCH SIZE. NUM HEADS. SEO LEN. HEAD DIM). dtv De=dtv De. device="cuda
npty (
因为我们需要固定一个块, 同时遍历所有其他块.
because we need to keep one block fixed and iterate through all the other blocks.
> TIMELINE OUTLINE
op ( BATCH _ SIZE, NUM _ HEADS, SEQ_ LEN, HEAD DIM, Cau Sal, dty P 我们固定的块将决定并行运行的程序数量
The block that we keep fixed will define how many programs we have that run in parallel > TIMELINE OUTLINE
7 而固定的块包含一个由宏定义的块大小数量的元素.
and the block that is fixed has a block size macro number of elements.
> TIMELINE OUTLINE
373
def test_op( BATCH_ SIZE, NUM_ HEADS, SEQ _ LEN, HEAD _ DIM, causal, dtype
torch. float16): 这就是为什么我们创建了序列长度除以块大小宏的块数
That's why we create a sequence length divided by block size macro number of blocks.
> TIMELINE OUTLINE
de f
test_op( BATCH _ SIZE, NUM _ HEADS, SEO _ LEN, HEAD _ DIH, causal, d type =torch. float16):
=(
torch. enpty( 在这一轴上创建线程块或程序.
BATCH_ S
> TIMELINE OUTLINE requires _grat
372
73
torch. float16): 在这个网格中, 轴2一一我也可以同样随意地使用轴1.
the axis2inthisgridis-icould have used also the axis1, indifferently > TIMELINE OUTLINE
我认
73
def test_op( BATCH_ SIZE, NUM_ HEADS, SEQ_ LEN, HEAD_ DIM,
loat16): 为在原始代码中已经完成了这一部分 它将指示我们将处理哪个批次
i think it Was already done here in the original code-it will indicate which batch 福
> TIMELINE OUTLINE
de f test_op( BATCH _ SIZE, NUM _ HEADS, SEQ _ LEN, HEAD _ DIH, causal, d type =torch. float16):
=(
torch. enpty( 以及每个批次中的哪个头.
and which head inside of each batch we are going to work with.
377
BATCH_ SIZE,
> TIMELINE OUTLINE
373
def test_op( BATCH_ SIZE, NUM_ HEADS, SEQ _ LEN, HEAD _ DIM, causal, dtype=
=torch. float16): 因此, 与正向传播类似, 我们也将使用一个名为stage的变量
so, and just like the forward pass, we will also use a variable called stage,
> TIMELINE OUTLINE
373 如果我们要计算的是因果注意力,! 则其值为三:
that if the attention that we are computing is causal, it will be equal to three,
> TIMELINE OUTLINE
如果是非因果注意力则其值为一.
77 and if we are computing a non-causal attention, then it will be equal to one.
> TIMELINE OUTLINE
在第一次迭代中我们将固定 K和 V块
( The first iteration :we'will fix K and V blocks > TIMELINE OUTLINE normal _(mean=0. 0, std=0. 5)
reauires arad(
375 并遍历所有块:这些 Q块的大小为块大小
and we witliterate through "all'the Q blocks in size of block size,
> TIMELINE OUTLINE
test _op ( BATCH _ SIZE, NUM _ HE 即查询向量的微观数量
torch. en pty (
( BATCH _ SIZE, NUM _ H micro number of query vectors.
> TIMELINE OUTLINE normal _(mean=0. 6, std=0. 5)
. reauires arad ()
HEAD _ DIH=ctx. HEAD_ DIH,
BLOCK _ KV= BLOCK_ SIZE _ MACRO STAGE =stage 那么, 让我们来看一下这个函数的签名.
So let'slook at the signature.
> TIMELINE OUTLINE def
test_op BATCH _ SIZE, NUM _ HEADS, SEQ _ LEN,
HEAD _ DIM =ctx. HEAD _ DIM,
STAGE =stage, 因此, 我们以网格启动的方式调用它
So we pass, we launch it as a launch grid because -
> TIMELINE OUTLINE
HEAD _ DIM=ctx. HEAD_ DIM,
393
STAGE =stage 因为我们已经定义了程序的数量, 所以我们也知道了会有多少个k V 块
and we have defined how many. programs we have, so we have how many kv blocks we will have OUT LIN TIMELINE
HEAD _ DIM =ctx. HEAD _ DIM,
STAGE =stage,
_stages = NUM 即序列长度除以宏块大小
it's the sequence land divided by the block size macro,
> TIMELINE OUTLINE 401
est_op( BATCH_ SIZE
HEAD _ DIM =ctx. HEAD _ DIH,
393
STAGE=stage, 因为在这个函数的for循环中, 我们将保持这些块固定不变,
399
because > TIMELINE OUTLINE 400
01
Q=(
HEAD _ DIM=ctx. HEAD_ DIH,
393
STAGE =stage, 因为在这个函数的for 循环中, 我们将保持这些块固定不变.
that's the the block that we will keep fixed in this uh for loop, in this function.
> TIMELINE OUTLINE
HEAD _ DIM=ctx. HEAD_ DIH,
BLOCK _ KV= BLOCK_ SIZE _ MACRO STAGE =stage 接着, 我们遍历所有大小为微块(我将其定义为32 )的查询块
and then we go through all the query blocks in size of block size micro,
> TIMELINE OUTLINE
NUM _ HEADS = NUM _ HEADS,
SEQ _ LEN= SEQ_ LEN,
BLOCK _ KV = BLOCK_ SI HEAD _ DIM =ctx. HEAD 稍后我们会讨论自动调优
num _stages = NUM _ STAGES,
num _warps WARPS,
which i defined itas32 > TIMELINE OUTLINE
stride _dim Q. stride(3),
NUM_ HEADS = NUM_ HEADS,
BLOCK _ Q BLOCK _ SIZE _ MICF
SEQ_ LEN-SEO_ LEN, 以及如何调整这些值.
HEAD_ DIM=ctx. HEAD BLOCK _ KV-BLOCK _ SIZE and later we will talk about autotuning and how to tune these values.
> TIMELINE OUTLINE
num _warps = NUM _ WARPS,
num _stages = NUM _ STAGES 好的, 那么我传入查询向量、键向量和值向量.
All right, so T pass the query vector, the key vector and the V vector.
> TIMELINE OUTLINE ( BATCH _ SIZE, NUM _ HEADS, SEO _ LEN, HEAD _ IM), dtype =dtype, device =cuda
_stages = NUM _ STAGES 抱歉, 不是向量, 而是张量.
Q =(
> TIMELINE OUTLINE torch. en pty (
( BATCH _ SIZE, NUM _ HEADS, SEO_ LEN, HEAD_ DIM), dtypdtype, device =cuda
um _stages = NUM _ STAGES, 现在传入的是查询张量
de f
test_op( BATCH _ SIZE, NUM _ HEADS, SEQ _ LE
Q=(
Now the query tensor,
> TIMELINE OUTLINE torch. en pty (
( BATCH _ SIZE, NUM _ HEADS, SEO _ LEN, HEAD _ DI M), dtype =dtype, device cuda
张量和 V张量, 它们指向张量的起始位置
K tensor and V tensor and they are pointing to the beginning of the tensor,
tride_batch= Q. stride(0)
> TIMELINE OUTLINE
stride_dim= Q. stride(3),
attn bwd dk dv Igrid]
Q= Q, 这意味着它们从张量的第一个批次
which means that they are beginning to the first batch > TIMELINE OUTLINE 382
d V=d V,
attn_bwd dk_dv Igrid]
Q= Q 个头第一个标记以及第一个维度开始.
and the first head and the first token and the first dimension of the tensors.
> TIMELINE OUTLINE
NUM _ HEADS = NUM _ HEADS,
SEQ _ LEN= SEQ_ LEN, 接着我们传入soft max 的缩放因子
Then we pass the soft max scale.
> TIMELINE OUTLINE
SEQ_ LEN= SEQ_ LEN,
BLOCK 我们传入了 O、 DQ、 DK 和 DV.
_warps=
Wepassthe O, DQ, DK, and DV.
> TIMELINE OUTLINE
SEQ _ LEN= SEQ_ LEN,
BLOCK _ Q = BLOCK_ SIZ M是计算过程中所需的关键项
Mis the one that is needed to compute.
> TIMELINE OUTLINE
389
SEQ_ LEN= SEQ_ LEN, 正如之前提到的, 我们没有将 P矩阵存储在 HBM中
BLOCK _ Q = BLOCK as you remember, from what we said before, we did not save the P matrix in the HBM > TIMELINE OUTLINE
389
SEQ_ LEN= SEQ_ LEN, 因为我们的目标是在反向传播过程中动态地重新计算它.
because we want to recompute it on the fly during the backward pass.
> TIMELINE OUTLINE
389
SEQ_ LEN= SEQ_ LEN, 因此 查询与键的转置相乘会生成一个非常大的矩阵
BLOCK _ Q= E
394
Sothe-query multiply by transpose of the keys,
OUTLINE 396
395
> TIMELINE
397
NUM _ HEADS = NUM_ HEADS,
SEQ_ LEN= SEQ_ LEN, 若将其存储在 HBM中并恢复, 会占用大量资源
it's a very-big matrix to save in the HBM and restore it,
> TIMELINE OUTLINE 396
397
SEQ _ LEN= SEQ_ LEN, 为此我们选择在需要时动态计算这个矩阵
so we want to compute it on the fly.
> TIMELINE OUTLINE
SEQ _ LEN= SEQ_ LEN,
BLOCK _ KV = BLOCK BLOCK _ Q = BLOCK_ SIZ 但无需重新计算归一化因子
MICI STAGEs stage HEAD DIM =ct X But we-don-tneed to recompute the normalization factor > TIMELINE OUTLINE 396
397
SEQ _ LEN= SEQ_ LEN,
BLOCK _ Q = 和每行的最大元素来应用soft max,
and the maximum element for each row to apply the softmax
9
> TIMELINE OUTLINE
SEQ _ LEN= SEQ_ LEN, 因为这些已在正向传播过程中计算完毕
BLOCK that was-already computed during the forward pass > TIMELINE OUTLINE
SEQ _ LEN= SEQ_ LEN,
BLOCK _ Q-BLOCK _ SIZE _ MI BLOCK _ KV = BLOCK _ SIZE 并保存到了矩阵 M 中.
STAGE =stage,
HEAD DIM = Ct X. HEAD u _warps OUTLINE um _stages = NUM _ STA and saved into this matrix M,
> TIMELINE
SEQ _ LEN= SEQ_ LEN,
M 包含了每行最大值的指数对数求和
which includes the log sum exp of the maximum of each row 94
> TIMELINE
OUTLINE
97
stride _batch = Q. stride (e)
stride _head = Q. stride (1),
stride _seq= Q. stride(2),
stride_dim-Q. strid
SEO_ LEN-SEO_ LEN, 以及归一化因子的对数.
plus. the logarithm of the normalization factor.
> TIMELINE OUTLINE
385
stride _batch-Q. stride(0)
stride_head = Q. stride(1), 借助指数 对数求和技巧, 我们只需直接应用它, 便能同时实现每个值的归一化处理
Withthelogsume
exp trick. we can just apply it and it will also normalize each value.
> TIMELINE OUTLINE
BLOCK _ Q = BLOCK _ SIZE _ MICRO,
SEQ _ LEN= SEQ _ LEN 接着, 我们得到了这里计算出的 D张量, 其中包含了所有 DI 值
Then we have the D tensor that we computed here with all the Dl values,
> TIMELINE OUTLINE
每个 DI值对应 O张量中的一个向量.
one for each vector in the O tensor.
> TIMELINE OUTLINE
BLOCK _ Q = BLOCK _ SIZE _ MICRO, 接下来, 我们需要传入头的数量、序列长度
Then we need to pass the number of heads, the sequence length.
> TIMELINE OUTLINE
BLDC E_ Q= BLOCK _ SIZE _ MICRO,
BLOCK _ KV = BLOCK _ SIZE _ MACRO 以及我们想用于 KV 的块大小, 即宏块大小
the block size that we want to use for the KV, which is the macro block size,
> TIMELINE OUTLINE
BLOCK _ KV = BLOCK 而微块大小则是我们始终迭代处理的那个.
and the micro block size is always the one that weiterateon.
395
> TIMELINE OUTLINE
390
391 我认为采用这种命名方式, 应该更容易理解我们在迭代哪一个
I think using this name, it should be easier to understand which one we are iterating > TIMELINE OUTLINE
BLOCK _ Q = BLOCK_ SZE _ MICRO,
SEQ _ LEN= SEQ_ LEN,
BLOCK _ KV = BLOCK 以及我们希望保持固定的是哪一个
and which we want to keep fixed.
> TIMELINE OUTLINE
BLOCK _ Q = BLOCK_ SZE _ MICRO,
BLOCK _ KV = BLOCK 因此, 固定的是宏块, 而迭代的是微块.
So the fixed one is macro and the iterating one is the micro.
> TIMELINE OUTLINE
BLOCK _ Q = BLOCK _ SIZE _ MICRO 嗯还有头维度(head dimension )
BLOCK _ KV = BLOCK _ SIZE _ M
um, head dimension, um,
> TIMELINE OUTLINE
9 稍后我们会明白为何要使用不同的块大小进行迭代
and later we will see why we would use a different block size to iterate from,
> TIMELINE OUTLINE
BLOCK _ O= B'
(parameter )ctx: A
BLOCK_ KV=
393
394 因为这涉及到 Triton 能够利用软件流水线技术
because this is related to the number of stages that triton can divide your for loop into > TIMELINE OUTLINE
BLOCK _ Q= B'
(paraneter) ctx: 将你的for循环划分为多个阶段的原因.
BLOCK_ KV=
thanks to software pipelining.
> TIMELINE OUTLINE
BLOCK _ Q= B'然后我们还有头维度
(head dimension )
then we have head dimension.
> TIMELINE OUTLINE
391
BLOCK_ KV= E 阶段
(stage)参数指示了我们在前向传播中计算注意力时
397
9
the stage indicates if the attention > TIMELINE OUTLINE 398
BLOCK _ Q BLOCK _ SIZE _ MICRO,
SEQ _ LEN= SEQ_ LEN,
BLOCK variable )stage : Literal [3, 1 是否采用了因果性(causal )机制.
that we computed in the forward pass was causal or not causal, um,
> TIMELINE OUTLINE
BLOCK _ Q = BLOCK _ SIZE _ MICRO,
391
BLOCK _ KV= BLOCK_ SIZE _ MAC 此外 还有我们设定为固定的warp 数量和流水线阶段数.
the number of warps and the number of stages which we defined as fixed.
> TIMELINE OUTLINE
stride_seq = Q. stride (2),
stride_dim= Q. stride(3), 不过, 稍后我们会讨论自动调优(autotuning )白 的相关内容.
but later we will talk about auto tuning.
> TIMELINE OUTLINE 395
394
+
num_stages = NUM_ STAGES
Live Shan
HEAD _ DIM =ctx. HEAD_ DIM,
STAGE=stage,
393 所以我有时候会反复重复同样的内容, 这一点我应该改进一下.
so sometimes i repeat the same stuff over and over, so i should change that.
> TIMELINE OUTLINE
393
HEAD_ DIM=ctx. HEAD_ DIM,
STAGE =stage, 嗯, 好的, 让我们先写出这个函数的签名, 然后把它放在这里
um, okay, let's write the signature of this function and let's put it here.
> TIMELINE OUTLINE
BLOCK _ Q :tl. constexpr,
BLOCK_ KV :tl. constexpr
HEAD_ DIH:t1. c
STAGE:tl. co 我们已经描述了这个函数的签名
301
Wealreadydescribedwhatisthesignatureofthisfunction.
OUTLINE
SO
> TIMELINE
def forward(ctx, Q, K, V, causal, soft max _scale ):
29
BLOCK_ KV:tl. constexpr,
HEAD_ DI M: tl. const expr,
STAGE :tl. const expr, 让我们直接进入正题吧,
let's go directly to the meet.
OUTLINE 384
385
@static method > TIMELINE 86
def forward(ctx, Q, K, V, causal, soft max _scale ):
BLOCK _ KV :
EAD_ DIM: 首先, 我们需要理解的是, 确定如何调整
so The first thing that we need to do is understand the offset by
> TIMELINE
OUTLINE
def forward(ctx, Q, K, V, causal, soft max _scale ):
HEAD _ DI M: tl. const expr, 查询(query )、 键(key )和值(value )的偏移量
384
class T
which we need to move this query, key and value.
> TIMELINE OUTLINE 305
def forward(ctx, Q, K, V, causal, soft max _scale ):
a stati
tl. int64
offset_batch_headgseq This is the offset that allows 扁移量首先取决于
310
ass Triton Attent
And the offset is given by the, first of all,
OUTLINE 313
311
@static method > TIMELINE def forward (ctx, Q, K, V, causal, soft max _scale ):
tl. int64 定位到正确的批次(batch)以及每个批次中正确的注意力头(head )
we need to enter the right batch and the right head inside of each batch.
> TIMELINE OUTLINE def forward (ctx, Q, K, V, causal, soft max _scale ):
offset _batch _head =(stride _batch *index _batch + stride _head *index _head ). to (
tl. int64 我们通过将程序索引(program index )
We compute the index of the batch just like during the forward pass > TIMELINE OUTLINE forward (ctx, Q, K, V, causal, soft max _scale ):
385
tl. int64 即头索引
(head index )与批次索引(batch index )的乘积一一进行划
311
310
by dividing the program index,
> TIMELINE OUTLINE 312
313
def forward(ctx, Q, K, V, causal, soft max _scale ):
@static method
304 offset _batch _head =(stride _batch *index _batch +stride _head *index _head ). to (
385
tl. int64 来计算批次的索引这与前向传播中的做法一致.
which is a multiplication of the index of the head and of the batch.
TIMELINE OUTLINE def forward (ctx, Q, K, V, causal, soft max _scale ):
tl. int64 我们将其除以头的数量, 以确定当前程序正在处理哪个批次.
wedivide it by the number of heads to get which batch this program is working with.
> TIMELINE OUTLINE ef forward (ctx, Q, K, V, causal, soft max _scale ):
tl. int64 的头, 我们只需进行取模运算, 就像在单次遍历的for循环中所做的那样
and to get the head we just do the modulus, just like in the for loop, for one pass.
> TIMELINE OUTLINE def forward (ctx, Q, K, V, causal, soft max _scale ):
tl. int64 批次和头的偏移量指示了 让我确认一下它的具体用途
the offset batch head indicates - let me check what is it for.
> TIMELINE OUTLINE 313
def forward(ctx, Q, K, V, causal, soft max _scale ):
tl. int64 好的, 它帮助我们定位到正确的批次和正确的头
311
310
okay, it enters the right batch and the right head.
> TIMELINE OUTLINE 312
313
def forward(ctx, Q, K, V, causal, soft max _scale ):
305
tl. int64
offset _batch _head _seq =(index _batch _head* SEQ_ LEN). to (tl. int64)
us to select the right seq given the batch and head.
class Triton Attention (torch. auto grad. Function ):
OUTLINE 313
311
@static method > TIMELINE def forward (ctx, Q, K, V, causal, soft max _scale ):
offset _batch _head =(stride _batch *in de batch +stride _head *index _head ). to (
tl. int64 那么这里的步幅(stride )是什么呢?
Triton Attention (torch. auto grad. Funct so what is the stride?
> TIMELINE OUTLINE def forward (ctx, Q, K, V, causal, soft max _scale ):
@static method
tl. int64
offset _batch _head _seq This is the offset thz 如果你还记得的话,
class Triton Attention (torch. auto g if you remember correctly > TIMELINE OUTLINE 313
def forward(ctx, Q, K, V, causal, soft max _scale ):
@static method
385
tl. int64 步幅(stride )告诉我们在该维度上需要跳过多少个元素
the stride tells us how many items you need to skip in that dimension > TIMELINE OUTLINE def forward (ctx, Q, K, V, causal, soft max _scale ):
tl. int64
This is 才能到达同一维度中的下一个索引.
to arrive to the next index in the same dimension.
> TIMELINE OUTLINE def forward (ctx, Q, K, V, causal, soft max _scale ):
tl. int64 们想要跳过批次的索引数量,? 只需将其乘以批次步幅
(stride batch)
so if we want to skip index number of batch, we multiply it by the stride batch,
> TIMELINE OUTLINE def forward (ctx, Q, K, V, causal, soft max _scale ):
offset _batch _head =(stride _bitch *index _batch +stride _head *index _head ). to (
tl. int64 即到达下一个批次需要跳过的元素数量.
which is how many elements you need to skip to arrive to the next batch.
> TIMELINE OUTLINE def forward (ctx, Q, K, V, causal, soft max _scale ):
tl. int64 此外, 我们还需要定位到正确的头(head)
311
Triton Atte
Plus we also need to enter the right head.
> TIMELINE OUTLINE 313 @static method def forward (ctx, Q, K, V, causal, soft max _scale ):
tl. int64
offset _batch _head _seq =(index _batch _head* SEQ_ LEN). to(tl. int64)
This is the offset that allows riah t seq
given the batch and head.
8 class Triton Attention (torch. auto grad. Function ):
OUTLINE 311
313
@static method > TIMELINE def forward (ctx, Q, K, V, causal, soft max _scale ):
305
+
tl. int64 因此, 我们将头的索引乘以头的步幅(stride of the head )
So we multiply the index of the head multiplied by the stride of the head.
> TIMELINE OUTLINE ef forward (ctx, Q, K, V, causal, soft max _scale ):
tl. in t64 这样, 我们就能准确地定位到 QK 和 V矩阵中对应的头的位置
toenterexactlyinthathead, inthetensor, foreachofthe Q, Kand V matrices.
> TIMELINE OUTLINE def forward (ctx, Q, K, V, causal, soft max _scale ):
tl. int64 此外我记得这个步幅也会用于 M和 D
Plus, we also have this will be used for, if I remember, for M and D > TIMELINE OUTLINE ef forward (ctx, Q, K, V, causal, soft max _scale ):
tl. int64 因为 M和 D没有头的维度它们仅与批次大小相关
because M and D only don't have the head dimension, so they are only batch size.
> TIMELINE OUTLINE ef forward (ctx, Q, K, V, causal, soft max _scale ):
tl. int64
offset _batch _head_sed This is the offset that all o 头的数量, 序列长度.
number of heads, sequence length.
> TIMELINE OUTLINE 313
def forward(ctx, Q, K, V, causal, soft max _scale ):
@static method
tl. int64 因此我们只需将批次索引乘以序列长度
so We just use the index batch multiplied by sequence length > TIMELINE OUTLINE def forward (ctx, Q, K, V, causal, soft max _scale ):
305
tl. int64 因为对于每个批次和每个头, 我们都会有序列长度的数据项.
because for each batch and on each head we will have sequence length, number of items.
> TIMELINE OUTLINE def forward (ctx, Q, K, V, causal, soft max _scale ):
tl. int64 你可以将其理解为从一个批次的头移动到下一个批次的头
so you can think of it at the stride to move from one batch head to the next batch head > TIMELINE OUTLINE def forward (ctx, Q, K, V, causal, soft max _scale ):
tl. int64
offset _batch _head _seq =(index _batch _head* SEDg EN). to(tl. int64)
This is the offset that allows us to select the right sequ given the batch and head.
8 class Triton Attention (torch. auto grad. Function ):
> TIMELINE OUTLINE 313 @static method def forward (ctx, Q, K, V, causal, soft max _scale ):
tl. int64
offset _batch _head _seq =(index _batch _h This is the offset thz 的步幅
class Triton Attention (torch. auto grad. Function )
or to the yeah.
> TIMELINE OUTLINE 313 @static method def forward (ctx, Q, K, V, causal, soft max _scale ):
tl. int64 接下来, 我们移动指针, 操作就是这样,
so let's move the pointers, And this was so.
> TIMELINE OUTLINE @static meth def forward (ctx, Q, K, V, causal, soft max _scale ):
315
d0 += offset _batch _head 我们通过偏移量 batch head 移动指针 Q、 K 和 V
320
321
We move the pointer Q, Kand V by the offset batch head > TIMELINE OUTLINE 322
323
def forward(ctx, Q, K, V, causal, soft max _scale ):
315
ffset_batch_heat 因为我们希望在大张量中定位到正确的批次和正确的头,
because we want to enter the right batch and the right head inside of this big tensors.
> TIMELINE OUTLINE def forward (ctx, Q, K, V, causal, soft max _scale ):
315
314
0+=
ffset_batch_head
D
DK 和 DV 也进行相同的操作, 因为它们的形状与 Q、 K和 V一致
And we do it also for D O, DQ, D K and D v because they have the same shape as a Q.
> TIMELINE OUTLINE def forward (ctx, Q, K, V, causal, soft max _scale ):
+=offset _batch _head +=offset batch _
offset batch 而 D O 的形状也与 Q 相同
K and V and D'o also has the same shape as > TIMELINE OUTLINE @static m def forward (ctx, Q, K, V, causal, soft max _scale ):
315
d0+=offset_batch_head 因此,
318 它们的形状相同, 所以我们使用相同的偏移量来移动
So they have the same shape, so we move by the same offset.
> TIMELINE OUTLINE def forward (ctx, Q, K, V, causal, soft max _scale ):
offset _batch _head 好的, 接着我们移动 m和d, 将它们定位到
All right, so And then we move m and d to move them to the right starting point on
321
> TIMELINE OUTLINE def forward (ctx, Q, K, V, causal, soft max _scale ):
d V += offset _batch _head += offset 当前批次和当前头的序列起始点.
which the sequence of the current head and the current batch and the current head start s.
323
TIMELINE OUTLINE ef forward (ctx, Q, K, V, causal, soft max _scale ):
d V += offset _batch _head 这样, 它们就指向了专属于当前批次
So they are pointing to the first vector of the sequence dedicated to the current batch
323
> TIMELINE OUTLINE ef forward (ctx, Q, K, V, causal, soft max _scale ):
d V += offset _batch _head offset bat c 和当前头的序列的第一个向量.
s Triton Attention (torch. auto grad. F and the current head > TIMELINE OUTLINE @static method def forward (ctx, Q, K, V, causal, soft max _scale ):
320
Make sure the 同样地,
q、"k、v*以及do、dq、dk 和v也遵循相同的逻辑.
And the same is true for g, k, and vand the do, dg, dk, and.
> TIMELINE OUTLINE def forward (ctx, Q, K, V, causal, soft max _scale ):
好的, 接下来我们加载一些其他数据.
OUTLINE class Trito n Attention(t
Okay then we load some other stuff > TIMELINE @static method
因为在这个迭代中, 我们固定了kv class ri to natte tion t Because here we fix, in this iteration,
> TIMELINE OUTLINE @static method
D += offset _batch _head _se 并通过q进行循环迭代.
in this method we are going. to. do a for loop in which we fix kv and we iterate through q.
325
> TIMELINE OUTLINE @static method
因此, 我们首先需要加载kv 的这个分块,
OUTLINE class Triton At So we first need to load this divs block of kv.
> TIMELINE @static method
ein the right place w. r. t batch, 我们按照以下步骤进行操作.
> TIMELINE OUTLINE @static method
offs _kv = start _kv+tl. arange (θ, BLoc K _ KV)
d_btock=tl. zer( BLOCKKV, HEAD _ DIM ],
d V _block =tl. zeros ({ BLOCK _ KV, HEAD_ DIM), dtype 如下所示.
OUTLINE
33
class Triton Attention (torch. auto grad. Function ):
as follows.
> TIMELINE @static method
offs _kv = start _kv +tl. arange (θ, BLo CK _ KV)
d V_block =t1. zeros({ BLOCK_ KV, HEAD_ IH], dtypent1. float32)
8
> TIMELINE OUTLINE 33 class Triton Attention (torch. auto grad. Function ):
@static method
offs _kv =start _kv+tl. arange (θ, BLo CK _ KV ) 因此我们知道需要加载一个二维张量
class Triton Attenti So We know we need to load a2d tensor,
> TIMELINE OUTLINE @static method
329
offs_kv = start _kv+tl. arange(e, BLo CK_ KV)
336 所以需要定义每个向量( K和v)在第二维度上的范围
so we need to define what ·are. the ranges in the second dimension of each vector, kan dv,
> TIMELINE OUTLINE @static method
offs _kv =start _kv+tl. arange (θ, BLo CK _ KV)
d V_bl ock= tl. zeros (
d K_block=tl. zeros 而这个范围由这个因子决定
that we need. to load, and it's defined by the this, by this factor.
> TIMELINE OUTLINE @static method
offs _kv =start _kv+tl. arange(θ, BLo CK_ KV) 接下来, 我们需要明确当前程序将要处理的 KV块是哪一个.
then we want to understand. which. KV block this particular program is going to work with.
> TIMELINE OUTLINE @static method
offs _kv = start _kv+tl. arange(θ, BLo CK_ KV) 因此这个特定程序会跳过一些 KV块
So this particular program is going to skip some KVs > TIMELINE OUTLINE @static method
offs _kv =start _kv +tl. arange (θ, BLo CK _ KV ) 这些块将由可能并行运行的其他程序处理.
that will already be managed by other programs that may be running in parallel.
> TIMELINE OUTLINE @static method
offs _kv = start _kv + t. a range (θ, BLo CK _ KV)
333
d V_block = t1. zeros({ BLOCK _ KV, HEAD _o IM),
d K_block=tl. zeros([ BLOCK_ KV, HEAD_ DIM ], 如何根据程序
and how to understand what this program should be working with 334 > TIMELINE OUTLINE @static method
offs _kv = start _kv + tl. arange (θ, BLo CK _ KV)
d V_block= t1. zeros( BLOCK_ KV, HEAD _ DIM], dtype=t1. f
d K_block= tl. zeros ([ BLOCK _ KV, HEAD _ DI M), dtype 的索引
class Triton atter in based on the index of the program zero,
> TIMELINE OUTLINE 33 @static method
329
offs_kv= start_kv+tl. arange (e, BLo CK _ KV )
(由序列长度除以块大小宏定义) 来确定该程序应处理的内容.
which is defined on. sequence length divided by the block size macro.
> TIMELINE OUTLINE @static method
STAGE =stage,
453 如果你还记得, 块大小宏是我们设定的固定值.
And if you remember, block size macro is the thing that we fix.
> TIMELINE OUTLINE
STAGE =stage,
num _warps = NI
um_stage 因此, 程序 ID 为零会告诉我们
So it's telling us this program ID zero will tell us > TIMELINE OUTLINE 459
def test
Q=(
STAGE =stage, 有多少块大小宏的 KV已经被其他程序管理
455
howmanyb
block size macro KV are already being managed by other programs,
> TIMELINE OUTLINE
HEAD _ DIH =ctx. HEAD _ DIH,
nu_stages= NUM_ STAGES, 我们无需关心它们.
458
so we shouldn't. care about them.
> TIMELINE OUTLINE 459
def test_op( BATCH_ SIZE, NUM
Q =(
因此, 我们跳过这些部分,
stride _0_dim
=0. stride(3),
. stride(2
So we skip them.
> TIMELINE OUT LINE
NUM_ HEADS = Q. shape[1],
BATCH_ SIZE= Q. shape(e),
of fs_kv= start_kv+tl. arange(0, BLOCK _ KV)
d V_block=tl. zeros([ BLOCK_ KV, 让我们回到这里.
d K_btock= tl. zeros({ BLOCK_ KV,
336
class Triton Attention (tr ch. auto grad. Function Let'Sgo back here > TIMELINE OUTLINE 337
338
@static method
def forward (ctx, Q, K, V, causal, soft max _scale ):
shape [-1], K. shape[-1] 这是我们需要跳过的向量数量. 因此, 其他 KB 从起始 KB 开始
And this is the number of vectors that we need to skip So other KB start from start KB > TIMELINE OUTLINE stage =3 if causalelse1
def forward (ctx, Q, K, V, causal, soft max_scale):
Q. shape[-1], K. shape[-1] 我们需要加载多少取决于块 KB的大小
and how many we need'to load them Bell depends on what is the block KB?
344
> TIMELINE OUTLINE stage =3if causal else 1
def forward (ctx, Q, K, V, causal, soft max _scale ):
HEAD _ DIM_ Q, HEAD_ DIM_ K
= Q. shape [-1], K. shape[-1]
HE AD_ DIM_ V= V. shap 这个块 KB等于块大小宏.
BATCH_ SIZE, NUM_ HE
assert HE This block KB is equal to block size macro.
> TIMELINE OUTLINE stage =3if causal else 1
d V _block = tl. zeros ([ BLoc K _ KV
d K_block= tl. zero 因此, 它将包含128个向量.
class Triton Attention (t OUTLINE @static method So it will be 128 vectors.
> TIMELINE
32 因此, 我们定义了二维张量, 并将其存储在 SRAM中
So we define our tensors, two-dimensional tensors that we will store in the SRAM,
> TIMELINE OUTLINE HE AD_ DIM_ O,
331
332
offs_kv =start_kv+tl. arange (θ, BLo CK _ KV)
d V_block= tl. zeros([ BLoc K_ KV 因为在 Triton 中, 每次加载数据时都是从 HBM 加载到 SRAM 的
because in Triton every time you load something you load it from the HBM into the SRAM.
TIMELINE OUTLINE HEAD _ DIM _ Q, H
d V _block =tl. zeros ([ BLOCK _ K V, HEAD_ DIM], dtype=tl. float32)
d Kblock =tl. zeros [ BLOCK _ KV, HEAD_ DIM],
dtype=tl. float32)
class Triton Attention(torch. autograd. Function):
OUTLINE
def forward(ctx, Q, K, V, causal, soft max _scale ):
@static method > TIMELINE
332
offs_kv = start_kv+tl. arange (θ, BLo CK _ KV)
d V_block=tl. zeros([ BLo CK_ KV,
HEAD _ DIM ], dtype=tl. float32 因此, 我们定义了它们在 SRAM中的存储位置
338
337
So we define where they should be saved in the SRAM > TIMELINE OUTLINE 339
ape[-1], K. shape[-1]
d V_block=tl. zeros([ BLoc K_ KV,
d K_block
HEAD_ DIM], dtype=tl. float32 初始值为零, 现在开始加载它们.
and they are initially zeros and now we load them.
> TIMELINE OUTLINE @static method
d V _block =tl. zeros ([ BLo CK_ KV,
d K_block= tl. zeros( 我们按如下方式加载它们.
OUTLINE class Triton Attention (torch. auto gr So we load them as follows.
> TIMELINE @static method
K+offs_kv[:, None]*stride_seq+offs_dim [ None,:]*stride _din )# Shape :( BLOCK _ KV1, HEAD_ DIH)
_block=tl. load(
)# Shape :( BLOCK _ KVg HEAD_ DIH
V+offs_kv[:, None]*stride_seq +offs _dim [ None,:]*stride _dim
OUTLINE
345
344
class Triton Attention (torch. auto grad. Function ):
> TIMELINE @static method
)# S hap( LOCK_ KV1, HEAD_ DIH)
K+offskv:, None ]stride_seq +off s_di N one,]*stride din
_block=tl. load(
V+offs_kv[:, None ]*stride_seq +offs _dim [ None,:]*stride _dim )# Shape :( BLOCK _ KV1, HEAD_ DIH)
OUTLINE
345
344
class Triton Attention (torch. auto grad. Function ):
> TIMELINE @static method
Koffs_kv[, None]*stride_seq+offs_dim [ None,:]*stride _dim
V_block=tl. load
V+offs_kv[:, 我们确认在 K张量指针中
class ir We say. that. okay in the K, in the K tensor pointer,
> TIMELINE OUTLINE @static method
+of fs_kv:, None ]*stride _seq +offs_dim[ None,:] * stride_dim
338
CK_ KV1, HEAD_ DIM) 它已经指向了正确的索引、正确的批次和正确的头
which is already pointing. to the right index, to the right batch and to the right head.
> TIMELINE OUTLINE @static method
+
#5hape:( BLOCKKV 1, HEAD_ DIM)
+offs_kv[:, None ]* stride_seq +offs _dim [ None,:]*stride _din
V_block=tl. load(
V+offs_kv[:,
None ]*stride_seq +offs _dim [ None,:]* stride _dim
)# Shape:( BLDCKKV1; HEAD_ DIH)
> TIMELINE OUTLINE 345 class Triton Attention (torch. auto grad. Function ):
@static method
+of fs_kv:, None ]*stride _seq +offs_dim[ None,:]* stride_din
K_ KV1, HEAD_ DIM) 因为这是我们在前面步骤中已经处理好的
OUTLINE
345
class Triton Atte because. that's something that we did here.
> TIMELINE 346 @static method
V_block=tl. load (
V+of fs_kv [:, 我们声明需要加载正确的键序列, 它应该从键的起始位置开始
we say we need. to load the right sequence of keys, which should start from of key,
> TIMELINE
OUTLINE
def forward(ctx, Q, K, V, causal, soft max _scale ):
339
V_block=tl. load(
( BLOCK_ KV1, HEAD _ DIM )
V +offs _kv [:, 因为这里已经包含了在序列长度维度上应该跳过的数量,
because this already includes how many we should skip in the sequence length dimension.
TIMELINE OUTLINE
339
_block=tl. load( 对于每个向量, 我们需要加载头维度中的所有维度
and for each of these vectors, we need to load all the dimensions in the head dimension,
TIMELINE
339
V_block=t1. load(
Shape :
( BLOCK _ KV 1, HEAD_ DIM) 因为键
(如果我想提醒您的话)白 的结构是批次、头数量、
because. the key, if I want to remind you, is batch number of heads,
> TIMELINE
OUTLINE
def forward(ctx, Q, K, V, causal, soft max _scale ):
V block =tl. load (
# Shape :( BLOCK _ KV1, HEAD _ DIM)
)# Shape:( BLOCK_ KV1, HEAD_ DIM)
V+off s_kv[:, None]* stride _seq +offs_di n[ N
* stride _di 序列长度和头维度.
class Triton Attention (torch. a Sequence Length and head Dim.
rad. Function ):
> TIMELINE OUTLINE 347
defforward(ctx, Q, K, V, causal, soft max _scale ):
@static method
V _block =tl. load ( 现在, 通过使用这行代码, 我们跳转到了正确的批次和正确的头.
Now, by using. this line, we are skipping to the right B and to the right head.
> TIMELINE OUTLINE def forward (ctx, Q, K, V, causal, soft max _scale ):
V _block =tl. load(
# Shape
( BL0 CK_ KV 1, HEAD_ DIM) 这意味着我们已经在这里进行了索引, 并在此处选择了特定的索引.
so it'slikewe alreadyindexedhereandherewealreadyselected an index.
> TIMELINE
OUTLINE
def forward(ctx, Q, K, V, causal, soft max _scale ):
V _block =tl. load ( 因此, 当前 K 指向一个二维张量的起始位置, 而我们明确表示:
so right now this K is pointing to the beginning of a tensor of two dimension and we tell :
> TIMELINE OUTLINE def forward (ctx, Q, K, V,
al, soft max _scale ):
_block =tl. load (
# Shape :( BLOCK _ KV1, HEAD_ DIM) 我们不需要整个序列, 只需要序列的某一部分.
okay,
We don't want all the sequence, we want some part of the sequence.
> TIMELINE OUTLINE
V _block =tl. load (
# Shape :
( BLOCK _ KV1, HEAD _ DIM)
# Shape:( BLOCK_ KV1, HEAD_ DIH)
V+offs _kv[:, None]* stride_seq+offs_dim [ None,:] 哪一部分呢?
class Triton Attention (torch. auto grad. Function ):
which part?
> TIMELINE OUTLINE 347
def forward(ctx, Q, K, V, causal, soft max _scale ):
@static method
V _block =tl. load (
Shape:
( BLOCK_ KV 1, HEAD_ DIM)
*stride_seq + offs_dim None
:]*stride_d 由这个start KV 指示的那一部分.
Triton Attention(torch. autogr
The one that is indicated by this start Kv.
> TIMELINE
OUTLINE
@staticmethod
def forward(ctx, Q, K, V, causal, soft max _scale ):
V _block =tl. load (
# Shape :
( BLOCK _ KV1, HEAD_ DIM)
+offs_kv[:, 以及我们想要序列长度中的多少部分.
And how many of in these g uence length we want > TIMELINE OUTLINE def forward (ctx, Q, K, V, causal, sof
V_block=tl. load(
+offs_kv :, 嗯, 我们想要 我觉得这样写会更清晰, 我们可以这样表达
well, we want-I
IThinkit'seasytowriteitlikethis, sowecanwriteit
TIMELINE
OUTLINE
def forward(ctx, Q, K, V, causal, soft max _scale ):
V _block =tl. load (
# Shape :
( BLOCK _ KV1, HEAD_ DIM)
348
+offs_kv:,
start'KV 开始到 start KV 加上 block KV 结束的部分.
344
class Triton Atter that from start KV to start KV plusblock KV.
OUTLINE
346
@staticmet
> TIMELINE
def forward(ctx, Q, K, V, causal, soft max _scale ):
V _block =tl. load ( 因此, 我们想要这个数量的张量, 正好位于这个位置. 那么, 对于头维度
so we want this. number often sor, e
exactly at this location and for head dimension,
> TIMELINE OUTLINE f forward (ctx, Q, K, V, causal, soft max _scale ):
V_block = tl. load(
V+offs_kv[:, None]* stride # Shape :( BLOCK _ KV1, 我们想要选择什么呢?
class Triton Attention (torch. auto gra what do we want to select?
> TIMELINE OUTLINE 347
def forward(ctx, Q, K, V, causal, soft max _scale ):
@static method
V _block =tl. load (
# Shape :
( BLOCK _ KV1, HEAD_ DIM)
# Shape:( BLOCK_ KV1,
V+offs _kv [:, None ]* stride _ 我们想要选择所有的维度.
Triton Attention (torch. auto grad. F we want to select all the dimensions.
> TIMELINE OUTLINE 347
def forward(ctx, Q, K, V, causal, soft max _scale ):
@static method
_block =tl. load (
( BLOCK _ KV1, HEAD _ DIH) 因此
+offs _kv :,
*stride_seq +offs_dim 我们可以这样表示:从零到head dimension 的范围, 也就是这个:offs dim
so we say that we want from zero to head dimension, which is exactly this: offs dim.
Triton
> TIMELINE
OUTLINE
def forward(ctx, Q, K, v, causal, soft max _scale ):
V_block=tl. load(
V+offs_kv:, 好的, 我们对k块执行这个操作, 同时也对v块执行同样的操作.
44
okay, we do it for the k block and we do it for the v block.
Triton Attenti > TIMELINE OUTLINE 346
347
def forward(ctx, Q, K, V, causal, soft max _scale ):
V_block=tl. load(
V+offs_kv[:, None]* stride_seq +off s_din [ None,:]* stride _dim ( BLOCK _ 这里我觉得我没有修改注释.
OUTLINE class Triton Atten t
herej think i didn't change the comment.
> TIMELINE 348 @static method
V +offs _kv l:, 应该是blockkv, i 而这里应该是 block KB, 之前它被称为 block KB1
this should be block kv and this should be block KB before it was called block KB1,
> TIMELINE OUTLINE @static method
# Shape :( BLOCK V, HEAD _ D
V+offs _kv [:, None ]*stride_seq +off s_din [ None,:]* stride _din 就像原始代码中那样
OUTLINE class Triton Attention (torch. auto grad. Fl like in the original code > TIMELINE 348 @static method
# Shape :( BLOCK _ KV, HEA
V+offs_kv[:, None]* stride_seq +off s_din [ None,:]*stride _dim 我对命名做了一些简化.
class ritoattetiontorcol simplified a little bit the naming.
> TIMELINE OUTLINE @static method
]*stride_seq +offs _dim [ None,:]*stride _din 我觉得这个版本更好, 更容易理解
think this one is better, easier to follow > TIMELINE OUTLINE 4
+off s_dim[ None,:]*stride_din
343 因为在原始代码中他们也是用了两个for循环
345
because in the original code they also do two for loops,
TIMELINE OUTLINE 347
lone ]*stride_seq +offs _dim [ None,:]* stride _dim 但在第二个循环中他们用了倒序的方式
but in the second for loop they do it backward TIMELINE OUTLINE lass Triton
V+offs_kv[:, None]*stride_seq+offs_dim [ None,:]*stride _din ( BLOCK 只是为了不改变循环的结构
just to not change the structure of the loops.
> TIMELINE OUTLINE 348 class Triton
但我觉得我的版本虽然更详细, 但更容易理解.
but I think mine is more verbose but easier to understand > TIMELINE OUTLINE
343 而且可能效率更低, 我的版本效率确实低得多
344
and probably less efficient, mine is much less efficient.
> TIMELINE OUTLINE 347
V +offs _kv [:, None ]* stride_seq +off s_din [ None,:]*stride _din 接着我们有off sq,
Then we have off sq,
> TIMELINE OUTLINE 48 class Triton Attention (torch.
+of fs_kv[:, None ]*stride _seq +offs_dim [ None
:]*stride_dim 因为我们需要理解每个查询块需要加载多少向量
because we need to understand for each block of queries how many vectors we need to load,
OUTLINE TIMELINE
V +offs _kv [:, None ]*stride_seq +off s_din [ None,:]* stride _dim 这由. off sg 表示, 并说明它们的数量
and it's indicated by this off sq and how many are them.
> TIMELINE OUTLINE
V +offs _kv [:, None ]* stride_seq +off s_din None,:]*stride _din
offs_q=tl. arange(0, BLOg_ Q) 它是一个 blockq.
it's a block q.
> TIMELINE OUTLINE 348 class Triton Attention (torch. au
d V _block =tl. zeros ([ BLOCK _ K V, HEAD_ DIM], dtype=tl. float32)
d K_block =tl. zeros( BLOCK_ KV, HEAD_ DIM),
dtype =tl. float32 这个方法中的block g 颜色表示的是块大小-micro, t 也就是32个向量
blockq in the color of this method was block size - micro, so it is 32 vectors.
> TIMELINE OUTLINE ( BLOCK _ KV, HEAD _ DIN
# Shape :
l. load(
( BLOCK_ KV, HEAD_ DIM) 好的, 现在我们需要访问 Q向量和 O向量, 不对
Okay, now we need to access, Q vectors an d O vectors, no,
> TIMELINE OUTLINE 4
of fs_q=tl. arange(0, BLOCK_ Q) 是 Q向量, 但它已经转置了.
OUTLINE def forward (ctx, Q, K,
@static method Q vectors but already transposed.
> TIMELINE HEAD _ DIM _ O, HEAD_ DIH_ K= Q. shape[-1], K. shape[-1]
of fs_q=tl. arange(0, BLOCK_ Q) 还有 O向量, 我们也需要访问它们,
OUTLINE estatic deffo And the O vectors, also we need to access them > TIMELINE HEAD _ DIM _ O, HEAD_ DIM_ K= Q. shape[-1], K. shape[-1]
of fs_q=tl. arange[e, BLocx Q] 因为我们要遍历查询和 O向量, 为什么呢?
because we are going. to iterate through queries an d O vectors actually also, why?
TIMELINE OUTLINE
343
342
offs_q=tl. arangee, BLOCK Q 因为让我们看看这里, 看看论文中的公式:为了计算vj Because let's look at here, let's look at the formulas in the paper :to compute vj.
TIMELINE OUTLINE
Because let's-look at here, let's. look at the ·formulas-in ·the paper : to compute vj,
Standard attention ( Algorithm O ) backward pass requires O( Nd+ N2) HBM accesses. uhile FLAs H ATTENTION backward pass ( Algorithm 4)requires O( N2d2 M-1) HBMaccesses
We analyze th
9heorem2)
so to compute the dvj-that's what we are trying ·to compute-here -
Standard attention ( Algorithm O )backward pass requires O( Nd+ N2) HBMaccesses. while FLASH ATTENTION backward pass
I We see that similar to the o rward pass, the backward pass performs O( N2) FLOPs and only requires Weanalyze the IO-com
Wald pass( Theorem 2)
Theorem 5. Let Nweneedto. iteratethroughallthed? AMuithd≤ M≤ Nd.
Standardattention ( Algorithm O) backward passrequires O( Nd+ N) HBM accesses, uh ile FLAs H ATTENTION
We see that similar to the forward pass, the backward pass performs O( N2) FLOPs and only requires
Weanalyze the IO-complexityof thebadau
Theorem5. Let N be the sequence length, d l M@t Sgion, and M be sizeof SRAM with d≤ M≤ Nd.
Standard attention ( Algorithm O )backward pass requires ( Nd+ N2) HBMaccesses. while FLAs H ATTENTION backward pass ( Algorithm 4)requires O( N2d2 M-1) HBMaccesses
We see that similar to the forward pass, the backward pass performs O( N2) FLOPs and only requires Wea Hu
Thand to compute dky we need to iterate through all the q? vd.
Standard attention ( Algorithm O )backward pass requires O( Nd+ N2) HBMaccesses, uhile FLAs H ATTENTION
We see that similar to the forward pass, the backward pass performs O( N2) FLOPs and on ly requires
O( N)extra memo r向量uts
o Theorem2)
Theorem 5. ivectors y because the qi is a block of vectors td≤ M≤ Nd.
Standard attention ( Algorithm O )backward pass requires O( Nd+ N2) HBMaccesses, while FLAs H ATTENTION
的原因
So that's-why we needy and why do. we need to access q as a transpose?
Standard attention ( Algorithm O )backward pass requires O( Nd+ N2) HBMaccesses, while FLAs H ATTENTION
of fs_q=t1. arange(θ, BLOCK_ O) 这就是为什么我们需要, 以及为什么我们需要以转置形式访问g的原因
So that's why we need, and why do we need to access q as a transpose?
> TIMELINE OUTLINE E AD_ DIM_ O,
342
offs_q=tl. arange, BLoc K Q
343 因为我们需要计算, 让我在这里展示一下, pij转置.
Because we need to compute, let me show you here, pij transpose.
> TIMELINE OUTLINE HEAD _ DIM_ Q, HEAD_ DIM_ K
pe[-1], K. shape[-1]
We see that similar to the forward pass, the backward pass performs O( N2) FLOPs and only requires Because we need to compute, let me show you here, pij transpose.
Standard attention ( Algorithm O ) backward pass requires O( Nd+ N2) HBMaccesses, while FLAs H ATTENTION backward pass
We see that similar to the forward pass, the backward pass performs O( N2) FLOPs and on ly requires
O( N) extra memory beyond inp We analyze the IO-complex i 亿计算
or ward pass ( Theorem 2).
Theorem5. Let Nbethe sequ Toi compute pij tran Spose ize of SRAM with d≤ M≤ Nd.
Standard attention ( Algorithm O )backward pass requires ( Nd+ N2) HBMaccesses. while FLAs H ATTENTION backward pass ( Algorithm 4)requires O( N2d2 M-) HBM accesses.
We see that similar to the forward pass, the backward pass performs O( N2) FLOPs and only requires forward pass ( Theorem 2)
we need q. transpose because pij would be the soft max of the-query Standard a i tention ( Algorithm O )backward pass requires O( Nd+ N2) HBMaccesses. while FLAs H ATTENTi ON backward pass ( Algorithm A )requires O( N2d2 M-) HBMaccesses
结里
Thcorem 5. Let multiplied by. the transpose of the keys. with d ≤ M ≤ Nd.
Standard attention ( Algorithm O )backward pass requires ( Nd+ N2) HBMaccesses, while FLAs H ATTENTION
p 的转置
Standard attention ( Algorithm O ) backward pass requires ( Nd+ N2) HBM accesses, while FLAs H ATTENTION backward pass ( Algorithm 4)requires O( N2d2 M-) HBMaccesses
We see that similar to the forward pass. the backward pass performs O( N2) FLOPs and only requires Qi ill then you ·need to do query. transpose :k multiplied by query transpose.
Standard attention ( Algorithm O )backward pass requires O( Nd+ N2) HBMaccesses, while FLAs H ATTENTION
这就是为什么我们访问的是查询转置而不是查询本身
We see that similar to the forward pass, the backward pass performs O( N2) FLOPs and only requires 就是
so that's why we accessed query transpose instead of queries,
Standard attention ( Algorithm O )backward pass requires ( Nd+ N2) HBMaccesses, while FLAs H ATTENTION backward pass ( Algorithm 4)requires( N2d2 M-) HBMaccesses
而我们访问查询转置的方式就是通过调整步幅来实现的
and the way we access. query. transpose is just by playing with the stride.
Standard attention ( Algorithm O )backward pass requires O( Nd+ R2) HBMaccesses, while FLAs H ATTENTION
342
343 而我们访问查询转置的方式就是通过调整步幅来实现的
and the way we access query transpose is just by playing with the stride.
TIMELINE OUTLINE HEAD _ DIM _ O, HEAD_ DIH_ K= Q. shape[-1], K. shape[-1]
343
342
offs_q=tl. arange[e, BLOCK _ Q ] 那 么我们就这么做吧, 我也在代码中写了注释, 角 解释为什么我们可以这样做
so let'sdo it like this, and i have also written the comment on why we can do it.
> TIMELINE OUTLINE E AD_ DIM_ O, HEAD _ DIM _ K
#q _ptr s= Q+off s_ql:, None ]*stride_seq +offs _dim [ None,:]* stride _dim This is equivalent to doing :
#q T_ptrs =tl. trans(q_ptrs)
q T_ptrs= Q+offs_q[ Non
d0_ptrs=d0+offs_ql:, 所以这相当于访问查询
So this is equivalent to accessing the query.
> TIMELINE OUTLINE class Triton At tent
g T _ptr s= Q+ 首先, 好吧, 这个操作是什么?
at each iteration First, okay, what is this operation?
> TIMELINE OUTLINE
#q_ptrs= Q+off s_q:, None]* stride_seq +offs_din [ None,]*stride di
#q T_ptrs=tl. trans(q_ptrs)
We point to the first BLO
d0+offs_q:, 这里的这个操作是什么?
What is this operation here?
> TIMELINE OUTLINE class Triton Attention (torch. a
#q _ptr s= Q+off s_ql:, None ]* stride_seq +offs _dim [ None,:]*stride _din 这段代码的意思是:定位到查询的起始指针
This is saying : go to the query, starting pointer,
> TIMELINE OUTLINE
这个指针已经指向了正确的批次和正确的注意力头
to the guer y which is already ight batch and to the right head for OUTLINE pointing to the i > TIMELINE
#_ptr s= Q+off s_g[:, None ]* stride_seg +off s_din [ None,:]* stride _din 当前程序应该处理的就是这部分数据.
ter) Q : An which this particular program should work with > TIMELINE OUTLINE
do _ptrs = do +off s_gl:, None ]*stride_seg +off s_din [ None,:]*stride _din > TIMELINE OUTLINE class Triton Attention (torch. auto grad. Function ):
348
#q_ptrs = Q+offs_g[:, None]* stride_seq +offs_din[
q T_ptrs= 然后选择一个二维向量, 在这个向量中, 沿着列方向重复查询的起点
and select a two-dimensional vector where you repeat the query starting point along the,
> TIMELINE OUTLINE
#q T_ptrs= tl. trans(q_ptrs)
# We point to the first BLoc K _ Q rows of Q for both the q T and do pointers, inside the for loop we will move forward by BLoc K _ Q rows at each iteration q T_ptrs = Q+offs_q[ None,:1 stride_seq +offs _dim [:, None ]* stride _dim do _ptrs = do +off s_ql:, None ]* stride_seq +off s_din [ None,:]* stride _din > TIMELINE OUTLINE class Triton Attention (torch. auto grad. Function ):
#q _ptr s= Q+off s_ql:, None ]* stride_seq +off s_dim[ None,:]* stride_dim
q T_ptrs
_ Q rows at each iteration 但实际上我们应该沿着行方向重复
in this case along the columns,
> TIMELINE OUTLINE class Tri to
#q _ptr s= Q+off s_ql:
# This is equivalent to doing :
* stride_seq + off s_dim[ None,:]* stride_dis
q T_ptrs= Q+offs 因为我们想要选择查询的行.
but we should be repeating it along the rows because we want to select rows of queries.
TIMELINE OUTLINE
348
q T_ptrs 不过, 如果我们想要选择查询的转置, 只需将两个维度互换即可,
However, if we want to select the query transpose, we just invert the two dimensions.
> TIMELINE OUTLINE
#q T_ptrs=tl. trans(q_ptrs)
# We point to the first BLoc K _ Q rows of Q for both the q T and do pointers, inside the for loop we will move forward by BLoc K _ Q rows at each iteration q T_ptrs = Q+offs_q INone,:1 stride_seq +offs _din I:, None]* stride_din do_ptrs=do+offs_ql:, None l*stride _seq +off s_din [ None,:]*stride _din > TIMELINE OUTLINE class Triton Attention (torch. auto grad. Function ):
#q _ptr s = Q+off s_ql:, None ]* stride_seq + offs _dim[ N
348
q T_ptrs=tl. tras(q_ptrs) 所以, 让我在不进行查询转置的情况下展示给你看,
So this is, let me actually show you without doing the query transpose,
> TIMELINE OUTLINE
#q T_ptrs=tl. trans(q_ptrs)
We point to the first BLo CK rard by BLo CK_ Q rows at each iteration.
do _ptrs=do+offs_ql:,
q T_ptr
offs_q No 我们可以简化成这样.
so let'sdo it simplified like this.
> TIMELINE OUTLINE class Triton Attention (torch. a
q T_ptr
0_ptrs 要访问查询的指针而不进行转置,
soto access the query um,
352
the query pointers, without transposition,
> TIMELINE OUTLINE
ptr s= Q+off s_ql:, None ]*stride_seq +offs_dim[ No
e,:]*stride_dim 我们可以这样做:定位到查询张量, 创建一个二维张量
we can just do like this :go to the query tensor and create a2 D tensor > TIMELINE OUTLINE
qptrs= Q+offs j[:, None]* stride_seq+offs_dim [ None,:]* stride _dim
q T_ptrs=tl. trans(q_ptrs)
We point to the first BLoc K _ Q rows of Q for both the q T and do pointers, inside the for loop we will move forward by BLoc K _ Q rows at each iteration q T_ptrs = Q+off s_q None,
do_ptrs= do+offs_q[:, None]*stride_seq+offs_din [ None, 1]* stride _din
:]*stride_seq +offs _dim [:, None ]stride _dim > TIMELINE OUTLINE class Triton Attention (torch. auto grad. Function ):
在行中放置你想要获取的每个查询的起始点
wherein the rows you put the starting point of each query that you want to get > TIMELINE OUTLINE
qptrs= Q+os_q[:, None]*stride_seq+offs _dim [ None,:】*stride _dim # We point to the first BLoc K _ Q rows of Q for both the q T and d0 pointers, inside the for loop we will move forward by BLoc K _ Q rows at each iteration.
q T _ptrs=tl. trans(q_ptrs)
d0_ptrs =d0+off s_q:, None ]*stride_seq +off s_din [ None,:]*stride _din
q T_ptrs= Q+offs_q None,
:]* stride_seq +offs _dim [:, None ]*stride _di > TIMELINE OUTLINE class Triton Attention (torch. auto grad. Function ):
Q_ptrs= Q+ors_q[:, None]*stride_seq +offs _dim [ None,:】*stride _im 并将每个这样的指针在列上也复制一份.
and replicate each of this pointer also on the column.
> TIMELINE OUTLINE
q_ptrs= Q+fs_q[:, None]*stride_seq+of fs_dim[ None,:】*stride_dim
q T_ptrs 这就是添加这个" None"维度的含义.
That's the meaning of adding this dimension none.
> TIMELINE OUTLINE
q_ptrs= Q+offs_q[:, None 这相当于在 Py Torch中使用unsqueeze操作.
s This is equivalent to when you do in Py Torch the un squeeze.
> TIMELINE OUTLINE
q_ptrs= Q+off s_q[:, None* st ride_seq+offs_dim None,:】*stride_dim
348
T_ptrs = tl. trans(q_ptrs) 就像你在调用unsqueeze函数时, 我想是用了参数1.
354
like you are calling offs q, unsqueeze, I think, 1
> TIMELINE
OUTLINE
355
class Triton At
q ptr s= Q+off s_ql:, None ]*stride_seq +offs _dim [ None,:]*stride _dim # We point to the first BLo CK _ Q rows of Q for both the q T and do pointers, inside the for loop we will move f onward by BLoc K_ Q rows at each iteration.
q T_ptrs=tl. trans(q_ptrs)
d0_ptrs =do +off s_ql:, None ]*stride_seq +off s_din [ None,:]* stride _din
#q T_ptrs = Q+offs_q None,:]*stride_seq +offs_diml:, None]*stride_dim
> TIMELINE OUTLINE class Triton Attention (torch. auto grad. Function ):
off s_q. unsqueeze(1]
aptrs= Q+offs_ql:,
stride_seq +offs_dim[ No
:]*stride_dim 因此 这等同于给这个张量增加一个列维度
So this is equivalent to adding the column dimension to this tensor.
> TIMELINE OUTLINE
of fs_q. unsqueeze(2)
Qptrs= Q+offs_q I:, None ]*stride_seq +offs _dim [ None,:】*stride _din # We point to the first BLoc K _ rows of Q for both the q T and do pointers, inside the for loop we will move forward by BLOCK _ Q rows at each iteration.
q T_ptrs=tl. trans(q_ptrs)
d0_ptrs =do +off s_ql:, None ]* stride_seq +off s_din [ None,:]* stride _din
#q T_ptrs = Q+offs_q[ None,:]* stride_seq +offs _dim :, None ]*stride _dim > TIMELINE OUTLINE class Triton Attention (torch. auto grad. Function ):
q T_ptrs=tl. trans(q_ptrs)
at each iteration q T _pt 并将列上的所有值重复一遍,
and repeating all the values that are on the columns.
> TIMELINE OUTLINE
# We point to the first BLoc K _ Q rows of Q for both the q T and do pointers, inside the for loop we will move forward by BLo CK _ Q rows at each iteration.
:]*stride_seq +offs _dim [:, None ]* stride _di
do_ptrs = do +off s_q:, None 】*stride_seq +off s_din [ None,:]*stride _in > TIMELINE OUTLINE class Triton Attention (torch. auto grad. Function ):
qptrs= Q+offs _ql:, Nog】*stride_seq+offs_dim [ None,:】*stride _dim
q T_ptrs= tl. trans(q_ptrs)
# We point to the first BLo CK _ Q row
ard by BLo CK_ Q rows at each iteration d0_ptrs=d0+offs_q[:,
q T_ptrs
+offs_g[ None 会有多少列呢?
How many columns will be there?
> TIMELINE OUTLINE class Triton Attention (torch.
q _ptr s = Q+off s_ql:, None ]* stride_seq +offs _dim [ Non ne,:]*stride_dim
T_ptrs =tl. trans(q_ptrs) 当我们将其与这里的这个张量相加时, 它会进行广播操作.
It will be broadcasted when we sum it with this tensor here.
> TIMELINE OUTLINE
q ptr s= Q+off s_ql:, None ]*stride_seq +offs _dim None,:]*stride dim # We point to the first BLoc K _ Q rows of Q for both the q T and do pointers, inside the for loop we will move forward by BLoc K _ Q rows at each iteration.
q T_ptrs=tl. trans(q_ptrs)
d0_ptrs = do +off s_q[:, None ]* stride_seq +offs _din IN one,:]* stride _din
q T_ptrs = Q+offs_q[ None,
:]*stride_seq +offs _dim [:, None ]*stride _di > TIMELINE OUTLINE class Triton Attention (torch. auto grad. Function ):
q_ptrs= Q+offs _q:, None】*stride_seq+offs_dim [ None,:]*stride dim 这是un squeeze 和广播操作的结合.
This is a combination of un squeezing and broadcasting.
> TIMELINE OUTLINE 355
A_ptrs = Q+offs _q[:, None]*stride_seq +offs_dim [ Nor ne,:]*stridedim
tl. trans(q_ptrs) 因此, 我们正在提取由offsg指定的查询向量, 然后
So we are taking the query vectors indicated by off sq and then we are,
> TIMELINE OUTLINE
qptrs= Q+offs _q[:, None]*stride_seq+fs_dim [ None,:】*stride _dim
q T_ptrs = tl. trans(q_ptrs)
# We point to the first BLoc K _ Q rows of Q for both the q T and do pointers, inside the for loop we will move forward by BLo CK _ Q rows at each iteration.
d0_ptrs=do+offs_q:, None]*stride_seq +off s_din [ None,:]*stride _din
q T_ptrs= Q+offsq None,
:]* stride_seq offs _dim [:, None ]stride _di > TIMELINE OUTLINE 356 class Triton Attention (torch. auto grad. Function ):
347
348
q_ptrs= Q+offs_g[:, None]*stride_seq+fs_dim[ None,:]* stride_dim 对于每个查询向量, 我们都在选择由dim指示的所有头维度.
for each query vector, we are selecting all the head dimensions indicated by dim.
> TIMELINE OUTLINE
q _ptr s= Q+offs_ql:, None]*stride_seq+fs_din [ Non one,:]*stride _dim
AT_ptrs
=tl. trans(q_ptrs)
q T_p 如果你反转这个广播操作, 就会生成
if you invert this broadcasting, it will create the transpose of the,
352 > TIMELINE OUTLINE
q_ptrs= Q+offs_ql:, Yone]*stride_seq +off s_din [ None,:】*stride _dim # We point to the first BLoc K _ Q rows of Q for both the q T and d0 pointers, inside the for loop we will move forward by BLoc K _ Q rows at each iteration.
q T_ptrs=tl. trans(q_ptrs)
d0_ptrs =do +off s_q:, None ]*stride_seq +off s_din [ None,:]*stride _din #q T_ptrs= Q+offs_q INone,
:]*stride_seq +offs _dim [:, None ]*stride _dis > TIMELINE OUTLINE class Triton Attention (torch. auto grad. Function ):
q_ptrs= Q+offs _ql:, one]* stride_seq+offs_dim [ None,:】*stride _dim
q T_ptrs = tl. trans(q_ptrs
at each iteratior
q T_ptr
d0_ptrs 你想要访问的查询向量的转置.
the query vector that you are trying to access.
> TIMELINE OUTLINE class Triton A
tl. trans (q _ptrs ) 所以这里的这些操作等同于这两行代码
so this stuff here is equivalent to the, these two lines,
> TIMELINE OUTLINE
348
q_ptrs= Q+offs_ql:, Nor
*stride_seq.+ offs _dim [ No 即访问查询向量然后进行转置这是你可以实现的操作.
query and then transposing, And it's something that you can do OUTLINE so accessing > TIMELINE
#q T_ptrs=tl. trans(q_ptrs)
We_point to the first BLoc K _ Q rows of Q for both the q T and do pointers, inside the for loop we will move forward by BLoc K _ Q rows at each iteration g Tp&r= Q+offs _q[ None,:]*stride_seq+offs_dim :, None ]*stride _dim do _ptrs = do +off s_ql:, None ]* stride_seq +off s_din [ None,:]*stride _din > TIMELINE OUTLINE class Triton Attention (torch. auto grad. Function ):
q T_ptrs 我可以从指针层面来阐述这个过程
354
could write down what is happening at the pointer level > TIMELINE OUTLINE 355
q T_ptrs=tl. trans(q_ptrs) 基本上你需要将off-skew 视为一个指针向量.
so basically you need to think of off-skew as being a vector of pointers.
> TIMELINE OUTLINE
#q T_ptrs=tl
# We point to
+
q T_ptrs = Q+off s_g[ None,:]* stride_seq +offs _dim [:, None ]* stride _dim
d0_ptrs = do+offs_q:, None]*stride_seq +off s_din [ None,:]* stride _din > TIMELINE OUTLINE class Triton Attention (torch. auto grad. Function ):
#q_ptrs= Q+off s_g[:, None]* stride_seq+offs_din [ None,:]* stride _dim 我们乘以了序列步长(segue ncest ride variable ) offs _
We multiplied by the sequence stride > TIMELINE OUTLINE 355
347
348 这告诉我们从一个查询向量到下一个需要跳过多少个元素
which tells us how many element we need to skip to go from one query vector to the next,
> TIMELINE OUTLINE
#q _ptr s = Q+off s_ql:, None ] *stride_seg +off s_din m[ None,:]* stride _dim 因为每个步长队列的步长将等于这个值
because each stride queue will be, the stride will be equal to > TIMELINE OUTLINE
#q T_ptrs =tl. trans(q_ptrs)
seq: Any
at each iteration q T_ptrs= Q+offs_g 在头维度为128的情况下
in the case that the head dimension is 128 > TIMELINE OUTLINE class Triton At tent
#q_ptrs= Q+off s_q[:, None]*stride_seq +offs_din [ None,:]* stride _din
#q T_ptrs =tl. trans(q_ptrs)
at each iteration q T_ptrs= Q+offs_g 序列维度的步长将是128
the stride of the sequence dimension will be 128 > TIMELINE OUTLINE
#q_ptrs= Q+off s_q:, None]*stride_seg+offs_din [ None,:]* stride _din
q T_ptrs 这意味着从一个查询向量到下一个, 你需要
itmeanst
352
that to go from one query vector to the next, you need to > TIMELINE OUTLINE
#q T_ptrs= tl. trans(q_ptrs)
q T_ptrs= Q+offs_q[ None,:]*stde_seq+offs_dim[:, None]*stride _din
d0_ptrs =do+offs_ql:, None]*stride_seq +off s_din [ None,:]* stride _din > TIMELINE OUTLINE class Triton Attention (torch. auto grad. Function ):
#q_ptrs = Q+of fs_q:, None] *stride_seg+offs_dinm[ None,:]* stride_dim 向前移动128个元素, 因为我想提醒你
you need to go forward by128 elements, b because i want to remind you > TIMELINE OUTLINE
#q_ptrs = Q+off s_q:, None] *stride_seq+offs_din [ None,:]* stride _dim 在内存中张量总是以扁平化的方式存储
that in the memory the t tensors are always stored like flattened > TIMELINE OUTLINE
#q_ptrs= Q+off s_q[:, None]*stride_seg+offs_din [ None,:]* stride _din 即每个维度都与下个维度扁平化存储在一起.
eter) strid
353
like each dimension is flattened with the next dimension.
> TIMELINE OUTLINE 355
#q T_ptrs
tl. trans(q_ptrs)
at each it e ratio 想象一下你有一个三行四列的矩阵
so imagine you have three rows and four columns,
> TIMELINE OUTLINE
但首先你会存储前三行, 然后是第一行.
but the first you will have the first three rows, then the sorry, t the first row.
> TIMELINE OUTLINE
347
348 也就是先存储前四列, 接着是接下来的四列, 然后是再接下来的四列
sothe first four columns,
then the next four columns, then the next four columns,
> TIMELINE OUTLINE
#q T_ptrs= tl. trans(q_ptrs)
# We point to the first BLo CK _ Q will move forward by BLo CK _ Q rows at each iteration q T_ptrs= Q+offs_q[ None,:]*strde_seq
do _ptrs = do +off s_q[:, None 】*stride 逐行存储.
row after row.
> TIMELINE OUTLINE class Triton Attention (torch. auto grad. Function ):
#q T_ptrs =tl. trans(q_ptrs)
at each iteration a T _ptr s
Q+offs 不写下来确实很难直观想象.
It's difficult to visualize until you write it down.
> TIMELINE OUTLINE class Triton k
#q T_ptrs =tl. trans(q_ptrs)
We point to the first BLo CK _0 那要怎么写下来呢?
(paraneter) stride_
ard by BLo CK _ Q rows at each iteration do_ptrs= do+offs_q[, Nor
q T_ptrs= Q+offs_g[ None,:]
So how to write it down?
> TIMELINE OUTLINE class Triton Attention (torch. au
#q_ptrs = Q+offs _q:, None] * stride_seq +offs_dim [ Non # This is equivalent to doing :
q T _ptrs 那么一开始的偏移量是多少呢?
at each iteration So what is off-skew at the beginning?
> TIMELINE OUTLINE class Triton Attention (t
q T_ptrs 这是一个范围 从0到100不对是从0到
32. 具体是0、
3
It'sarange that is from here, from 0to100, no, 0to32, 0, 1, 2, 3,
4
> TIMELINE OUTLINE
#q T_ptrs =tl. trans(q_ptrs)
at each iteration T_ptrs= Q+offs_q
5678以此类推
four, five, six, seven, eight, etc, etc.
> TIMELINE OUTLINE class Triton Attention (to re
#q_ptrs= Q+off s_q[:, None]*stride_seq+offs_din [ None,:*stride _dim #q T_ptrs
tl. trans(q_ptrs)
T_ptrs 我们将每个元素与序列的步幅相乘
we are multiplying each one with the stride of the sequence,
35 Z > TIMELINE OUTLINE
#q _ptr s= Q+ off s_ql:, None ] * stride_seq + off s_din [ None,:]* stride _din
#q T_ptrs = tl. trans(q_ptrs) 这样就不会跳过任何元素.
at each iteration q T_ptrs = Q+offs_q[ Nc
sothis will not skip any element.
> TIMELINE OUTLINE class Triton Attention (torch
这样正好跳过, 意味着健康维度是128 this will skip exactly implying that the health dimension is 128 > TIMELINE OUTLINE
#q _ptr s = Q+ off s_ql:, None ] * stride_seq +off s_din m[ None, :] * stride _din
#q T_ptrs =tl. trans(q_ptrs)
at each iteration a T_ptrs 这样会跳过两倍的128个元素
this will skip two times 128 elements.
> TIMELINE OUTLINE class Triton Attention (
347
q_ptrs= Q+offs_ql:, None]
stride_seq + offs_dim[ N
348 然后我们还要给这个向量添加另一个维度, 月 所以它将成为一个多维向量
and then we are adding. also another dimension to this vector, so this will be a vector.
> TIMELINE OUTLINE
#q _ptr s= Q+off s_ql:, None ]* stride_seq +of fs_di INone,:]* stride_din
q T_ptrs 接着, 你会在头部维度列数上进行广播
then you broadcast it on head dimension, number of columns,
> TIMELINE OUTLINE
#q _ptr s= Q+ off s_ql:, None ]* stride_seq +off s_di IN one,:]* stride _din
#q T_ptrs =tl. trans(q_ptrs)
We point to the first le)offs_din:
q T_ptrs= Q+offs_q[ No 并向每一个添加一个数值.
and to each of them you add one number.
> TIMELINE OUTLINE class Triton Atten
#q_ptrs= Q+off s_q[:, None ]* stride _seq+offs_di[ None,:]*stride_din 因此它会变成一个类似于fall 的向量
so it will become a vector like fall.
> TIMELINE OUTLINE
348
q T_ptrs=tl. trans(q_ptrs) 好的, 各位, 我还是直接演示下吧, 不然我觉得这可能会让人太困惑
okay, let me just do it,
guys, otherwise i think it's too confusing > TIMELINE OUTLINE
348 啊, 明白了, 那么我们得到的向量是这样的:首先是0, 接着是128
ah, okay, sowe have a vector that is as follows :so zero, then we have a128.
> TIMELINE OUTLINE *stride_seq + offs _dim [:, None ]* stride _dir
4 然后是两倍的128, 再是三倍的128, 以此类推
then we have two times 128, then we have three times of 128,
etc.
> TIMELINE OUTLINE T_ptrs
:]* stride_seq +offs_dim[:, N
None ]* stride _din
3*128
# We access the Qas a transposed array, so tha
of fs_din as a row vector
q_ptrs= Q+off s_ql:, None ]* stride_seq +off s_din OUTLINE #q T_ptrs= tl. trans(q_ptrs)
# We point to the first BLoc K _o rows of Q for both the q T etc.
cers, inside the for loop we will move forward by BLoc K _ Q rows at each iteration > TIMELINE q T_ptrs= Q+offs_q[ None,:]*stride_seq +offs _dim [:,
]*stride _dim
347
#2*128
348
3*128 我们正在添加由stream 指示的列数所以它们各自有多少列.
we are adding how many columns indicated by of steam, so of them has how many columns.
> TIMELINE OUTLINE ne,:]*stride_seq +offs_dim[:, N
he]*stride_din
This is equiv 因此, 它已经包含了列的数目
vecto
q_ptrs
q T_ptrs
soit has had. the. number of columns.
by BLo c K_ O rows at each iteration.
OUTLINE
do_ptrs =do +offs > TIMELINE
为了简化起见 我们假设这不是128 维的
please, for simplicity, let's. pretend it's not 128 dimensions.
ch iterati
> TIMELINE OUTLINE
# This is equivalent to access the row vector 让我们假设它是四维的
q T_ptrs = tl. trans(q_pt
ard by BLo CK _ Q rows at each iteration a T _ptr s= Q+off s_q
> TIMELINE OUTLINE
# This is equivalent to doing :
access the Q as a tra
off s_din as a row vector q_ptrs= Q+offs_ql:, None]
q T_ptrs=tl. trans (q _ptrs ) 那么, 这里就是四
We point to the first BLoc K _ O
q T_ptrs= Q+offs_q None,:]*stride will move forward by BLoc K _ Q rows at each iteration > TIMELINE OUTLINE
这里将是四的两倍, 而这里则是四的三倍
this. will be. two. times :four, this. will be three times four.
ach iter atl > TIMELINE OUTLINE
我们正在添加另一个维度, 即dim 维度
355
ch iteratio OUTLINE we > TIMELINE
# This is equiva len #q_ptrs= Q+offs_ 每一个都乘以dim的步幅
vecto
q T_ptrs
We point by BLo CK_ Q rows at each iteration q T_ptrs= Q
do_ptrs=do+
Each. one multiplied by. the stride of dim,
> TIMELINE OUTLINE
这个步幅将为一, 因为它是最后一个维度
which will be one because it's. the last dimension.
a ch iterat i > TIMELINE OUTLINE
步幅dim, 那么我们正在添加多少列呢?
Stride dim, so we are adding how many columns?
> TIMELINE OUTLINE
# This is equivalent to doing :
We access the Q as a tran sposed array, so that
ans off s_din as a row vector
#q_ptrs Q+offs_ql: None ]* stride_seq +offs_
#q T_ptrs =tl. trans(q_ptrs) 四列
# We point to the first BLoc K _ Q rows
q T_ptrs = Q+offs_q[ None,:]*stride_seq +offs _dim[:
sof Q for the for loop we will move for vard by BLo CK_ Q rows at each iteration.
> TIMELINE OUTLINE
d O_ptrs =do +off s_ql:, None ]* stride_seq +offs_din[ No
59
所以,
351 treat of fs_q as a col 我们正在添加一、零一 三, 我猜是这样的
356
357
Sowe. are adding one, zero, one, two, three, I guess.
> TIMELINE OUTLINE 358
# This is equivalent to 对吧?
row vector #q T_ptrs = tl. trans(q_ptr
ard by BLo CK _ Q rows at each iteration.
OUTLINE
do_ptrs =do +off s_q[:,, None Zero, one, two, three, right?
> TIMELINE
同样地, 我们还要加上这个, 哦, 天哪
Also to'this one, we. are adding, oh my God.
at each iteration > TIMELINE OUTLINE 0_ptrs
# This is equivalent to doing :
#q_ptrs= Q+offs_q:, None]*str
We access the Q as a transposed array 枣
#q T_ptrs=tl. trans(q_ptrs)
# We point to the first BLoc K _ Qrow
q T_ptrs = Q+offs_q[ None,:]*strid > TIMELINE OUTLINE
同样地我们还要加上这个, 零、一、二、
And also. to this. one, we. are adding, Zero, one, two, three a ch iterat i > TIMELINE OUTLINE
351 好的
. 然后我们还要在这个上面加上零、一、二、 二
Okay, ahd then also to this one we. are adding zero, one, two, three.
> TIMELINE OUTLINE
# This is equivalent to ccess the Qa
row vecto 那么, 这会选择什么呢?
q T_pt rs = tl. trans(q_pt
We point to the first B
ard by BLoc K_ Q rows at each iteration do _ptrs =do+offs_q[:, None ] *5
q T_ptrs Q+offs_q[ None,:]
So what this. will select?
> TIMELINE OUTLINE
这将从指针 Q 的起始点开始选择
This will select from. the starting point of the pointer each iteration > TIMELINE OUTLINE 0_ptrs
offs_din[ No
ne,:]*stride
它将选择第零个元素, 接着是第一个元素
it will select the element zero, then the element one,
> TIMELINE OUTLINE T_ptrs
+offs_din[:, None]* stride_
ows at each iteration.
d0+ offs al:,
然后是第二个元素, 最后是第三个元素.
then the element two. and. then. the element three,
> TIMELINE OUTLINE T_ptrs= Q+offs_q
:]*stride _seq +offs_din[:, N
]*stride_dir
s at each iteration
351 这些正是我们应当选取的第一个向量的头部维度.
which is exactly the head dimension of the. first vector that we should be selecting.
> TIMELINE OUTLINE stride _dir :]* stride din
353 接着:它会从向量的起始点选择第四个元素
Then it will select,'the. element 4 from. the. starting point of. the vector.
> TIMELINE OUTLINE stride _dir
access the Q as a 让我写下这个操作的结果.
q_ptrs
#q T_ptrs
Let me write the result of this operation.
> TIMELINE OUTLINE a T _ptrs of fs_q INo
:]* stride _seq +offs_din[:,
ne]* stride_dir ard by BLoc K_ Q rows at each iteration d0+offs a:,
*stride seq +offs din[ None.:]*stride din
这个结果将是0、12、
#q_ptrs= Q+offs_ql:,
#q T_ptrs=tl. trans (q_ptrs)
This 1s equivalen
Thisone will be 0, 1, 2, 3
> TIMELINE OUTLINE q T_ptrs= Q+offs_q[ None,:]*stride_seq+offs_dim[:,
doptrs=do+offs a:. N
We point to the first BLoc K _ Q
然后, 它会选择第4567 个元素.
Thenit. will. selecttheelement4, 5, 6, 7.
> TIMELINE OUTLINE q T_ptrs
Q+offs_q N
+offs_din I:, None stride _dir
351 接下来, 它会选择第8个元素, 我猜还有910、11.
358
Then it will select. the element 8, lguess, 9, 10, 11
> TIMELINE OUTLINE T_ptrs Q+offs_q
*stride_seq +offs_di n I:,
ne,:]*stride din
*stride_dir
0ptrs=d0+offsa:.
*stride seq + off s din No
之后, 它会选择第1213、1415个元素
Then it will select the element 12, 13, 14, 15.
BLOCK_ Q rows
at each iteration > TIMELINE OUTLINE q T_ptrs = Q+off s_q Nc
:]*stride_seq +offs_din [:,
*stride seq +of fs din[ None.:]*stride din
ptrs=d0 +of fs a[:.
因此, 从这个队列指针指向的起始位置开始
So, from the starting. point of. where. this queue is pointing,
each iteration > TIMELINE OUTLINE *stride _dir +off s al:,
offs din IN
它会选择紧邻队列之后的第一个元素、
it will select the first element right after. this queue,
rows at each iteration > TIMELINE OUTLINE q T_ptrs=
+ offs_q
stride_
+offs din[ N
# We access the as a transposed array, so that's why we treat offs _gas a column #ptrs =g +offs_g(;, None ] * stride_seg +offs _dim [ None,:] * stride _dim This is equivalent to doing OUTLINE W point to the first BLoc K rows of for both the g T and do pointers, inside the for loop we wilmove fonvard by BLoc K_grows at each iteration g T_ptrs=tl. trans(g_ptrs)
> TIMELINE q T_ptrs=q+offs_q Nor
+offs al:.
ne,:]*stride_seq +offs _din [:,
+off s din [ None,:]*stride din one ]* stride _din
access the gas 第二个元素、第三个元素,
the second element right after. this queue, the third element right. after this queue,
355
> TIMELINE
OUTLINE
q T_ptrs =
:]*stride din stride _dir
We access the Q as a transposed array, s 依此类推
# This is equivalent to doing #q T_ptrs=tl. trans(q_ptrs)
> TIMELINE OUTLINE e,:]*stride din
*stride_dim
59
d0+offsal:.
你可以看到, 这将成为第一个查询向量.
And this will be. the, you. can see. that this will. be the first. query vector.
> TIMELINE OUTLINE stride_seq + off s_din[:, None]
*stride_di
d0 +offs al:.
这将成为第一个查询向量
This is equivalent q T _ptrs = tl. tr
This. will be. the second query vector.
ward by BLoc K_ Q rows at each iteration > TIMELINE OUTLINE q T_ptrs= Q+offs_q Nc
这将成为第三个查询向量
This is equivalen
ters This. will be. the. third query. vector.
e forward by BLoc K _ Q rows at each iteration > TIMELINE OUTLINE q T_ptrs= Q+off s_q Non
,:]*stride_seq +offs_din [:, None :]*stride din
351 这是第四个查询向量, 因为它们在内存中是连续存储的
This is the fourth query. vector because in. the. memory, they are stored. one. after another.
> TIMELINE OUTLINE
:]* stride_5
stride_dir
它们被展平了
# This is equivalent to doing :
#q T_ptrs =tl. trans(q_ptrs)
# We point to the first BLoc K _ Q rows of They are flattened.
loop we will move forward by BLoc K _ Q rows at each iteration > TIMELINE OUTLINE q T_ptrs= Q+offs_q[ None,:]*stride_seq +off s_din[:, N
lone ]*stride _dim :]*stride din d0 +offs a[:.
*stride seq + offs din No
因此, 在内存中, 它们是这样存储的.
So in the. memory, they are stored like this.
> TIMELINE OUTLINE a T_ptrs
+offs_q Nc
ne.:]*stride din stride _din at each iteration +of fs din No
它们的存储方式如下
#q_ptrs= Q+offs_q:,
This is equivalent to do in q T _ptr s =tl. tran They are. stored. like. the. following > TIMELINE OUTLINE q T_ptrs = Q+offs_q Nor
:]*stride din stride _dir ve forward by BLoc K _ Q rows at each iteration d0+offs al:.
#(0, 1, 2, 3)
# We access the Qa 它们一个接一个地这样存储.
This is equivale
q T_ptrs
> TIMELINE OUTLINE q T_ptrs= Q+offs_q
:]*stride_seq + offs_din [:, None ]* stride _dil
d0+offs al:.
*stride seq + offs din None,:]*stride din
# We access the Q as a 因此, 它将选择所有这些
#q_ptrs= Q+offs_ql:,
#q T_ptrs= tl. trans (q _ptrs )
This is equivalent So. it will select all of. them.
OUTLINE We point to the first BLo CK will move forward by BLoc K _ Q rows at each iteration > TIMELINE do +off sa:. N
:]*stride_seq +off s_din[:, N
stride _dir
351
#(0, 1, 2, 3) 同时, 它还会创建一个具有我们想要可视化形状的虚拟张量.
(q_ptrs)
And OUTLINE 358
357
first BLoc K_ Q rows of Q for both the q T
> TIMELINE
q T_ptrs Q +offs_q None,:]*stride_seq +offs_din[:
0ptrsd0+offsa:. N
one ]* stride _dir :]*stride din
351
#(0, 1, 2, 3) 同时, 它还会创建一个具有我们想要可视化形状的虚拟张量.
it also create a virtual tensor with. the. right shape that we want. to. visualize. it. into.
> TIMELINE OUTLINE :]*stride din
351
#(0, 1, 2, 3)(4, 5g6, 7) 正如我们之前所见, 当你在内存中处理张量布局时
So as we sawbefore, when you. work. with. a. tensor layout in memory,
TIMELINE OUTLINE *stride _dir +off s din [ No
总是可以根据所需的形状将其视为任意形态
you can always view it as whatever shape you like based on. the shape. that you want.
> TIMELINE OUTLINE
#(0, 1, 2, 3)(4, 56, 7)
We access the Q as a 而重塑操作总是零成本的
And. the reshaping is always free.
> TIMELINE OUTLINE q T_ptrs= Q+offs _q Nc
:]*stride_seq +offs_dim :
stride _din
它并不涉及改变内存中元素的排列方式
Doesn't involve changing the arrangement of the elements in. the memory 355
> TIMELIN E
OUTLINE
q+off s_din :
+offs din[ None.:]* stride din
#(0, 1, 2, 3)(4, 5g6, 7) 希望现在变得更清楚了
#q_ptrs= Q+offs_q:, N
This is equivalent to #q T_ptrs=tl. trans(q_ptrs)
> TIMELINE OUTLINE q T_ptrs = Q+offs_q None,:]*stride _
off s_din [:, None ]*stride _din
s at each iteration :]*stride din
#(0, 1, 2, 3)(4, 5g6, 7) 那么现在我们可以继续深入探讨了?
So now. we can proceed further.
> TIMELINE OUTLINE q T_ptrs
:]*stride_seq +offs_din I:,
one,:]*stride din each iteration.
天啊,"这真是相当复杂呢."
q_ptrs
q T_ptrs
oh. my god, it was quite complicated > TIMELINE OUTLINE ne,:]*stride din stride _dir ard by BLoc K_ Q rows at each iteration d0+offs al:.
We point to the first BLoc K _ Q rows of Q for both the q T and do pointers, inside the for l
351
0_ptrs=d0 +offs_g[:, None]*stride _sec 所以每当遇到难题时, 我就会动手画图, 我也建议你这么做
so whenever i get stuck, i just draw things, and i think you should do it too TIMELINE OUTLINE le)
lape[-1]
pe[-1]. K. sh
# We point to the first BLoc K _ Q rows of Q for both the q T and do pointers, inside the for
d0_ptrs =d0 +off s_q[:, N
ne]*stride_seq +offs_din [ None,:]*stride _din 因为这是学习的唯一途径.
because. that's the only way to learn.
TIMELINE OUTLINE tati c method
We point to the first BLoc K _ Q rows of Q for both the q T and do pointers, inside the for
351
]*stride_seq +offs_din[ None,:]*stride _din 如果试图在脑海中想象所有内容, 往往很困难.
if you. try. to. imagine. everything in your head, it's always difficult,
TIMELINE OUTLINE a static method
# We point to the first BLoc K _ Q rows of Q for both the q T and do pointers, inside the for loop
q T_ptrs= Q+offs_q[ None,:]*stride_seq +offs _dim [:, None ]*stride _din
0_ptrs
d0+offs_q[:, None】*stride_seq +off s_din [ None,:]*stride _din 对于 O向量, 我们也采用同样的方法处理,
class Triton Atten And. we. do. the same job for the O vectors.
TIMELINE OUTLINE 358
351
q T_ptrs = Q+offs_q[ None,:]*stride_seq +offs_din [:,
0_ptrs=d0 +offs_g[:, No
ne]* stride _seg +offs_din [ N
tride_din
stride_din 处理 O向量时, 我们不需要以转置形式访问它, 因为转置在此处并非必需
Inthe O vectors, we. dont. access it as a transpose because we don't need it in transpose > TIMELINE OUTLINE static method
# We point to the first BLoc K _ Q rows of Q for both the q T and do pointers, inside the for loop
a T_ptrs = Q+offs_q[ None,:]*stride_seq +off s_din [:, None ]*stride _din
do_ptrs =do+offs_q[:, None】*stride_seq+offs_din[ None,:]* stride_din 只有 Q向量, 我们需要以转置形式处理
lass Triton Attention (to Only. the. Q, we need it in transpose.
> TIMELINE OUTLINE 358 static method
# We point to the first BLoc K _ Q rows of Q for both the q T and do pointers, inside the for loo
d0+offs_q:, N
ne]* stride_seq +offs_din[ None,:]*stride_din 好的, 它沿着查询的序列维度快速遍历.
55
Okay, it. race. through the sequence dimension of the query.
> TIMELINE OUTLINE a static method
q T_ptrs = Q+offs_q[ None,:]* stride_seq +offs _dim [:, None ]* stride _dim one ]* stride_seq +off s_din [ None,:]* stride _din
do_ptrs =d0+offs_ql:, N
Iterates over 于是, 我们从第零个查询开始.
So we start from the query number zero > TIMELINE OUTLINE class Triton Atter
d O _ptrs =do +off s_ql:, None 】*stride_seq +of fs_din[ None,:]*stride_din
curr_q = q
Iterates over the seq u
nce di me 在当前情况下.
in the current um.
> TIMELINE OUTLINE class Triton Attention (torch. aut
q T_ptrs = Q+offs_q[ None,:]* stride_seq +off s_din [:, None ]*stride _din None ]* stride_seq +off s_din [ None,:]* stride _din
d0_ptrs =d0+offs_ql:, N 在查询中, 我们需要遍历整个序列长度维度
well, in the query we need to go through all the sequence length dimension > TIMELINE OUTLINE
do _ptrs = do +off s_ql:, None ]* stride_seq +off s_din [ None,:]* stride _din
q T_ptrs= Q+offs_q[ None,:]*stride_seq +off s_dim[:, None]*stride_dim
iterati
351 因为只有在键中, 我们才能选择想要处理的正确键.
because only the key we select the right key that we want to work with.
TIMELINE OUTLINE
351
do_ptrs=d0+off s_q[:, None ]*stride _seq+offs_din [ N
one,:]*stride_din 这里我想提醒大家, 我们固定了键, 并遍历所有查询
so i want to remind you, here we fix the key and we go through all the queries.
TIMELINE OUTLINE
q T_ptrs = Q+offs_q[ None,:]* stride_seq +off s_din [:, None ]*stride _dim
do_ptrs = do+offs_q[:, None】*stride_seq +off s_din [ None,:]*stride _din 而查询需要从零开始, 直到序列长度结束.
but the query we need to start from zero until sequence length 5 TIMELINE OUTLINE
358
351
d0_ptrs =do+offs_q[:, Nor
e]* stride_seq +off s_din [ N
one,:]*stride _din 因此 这个for 循环的步数将是序列长度除以查询块大小
so the number of steps of this for loop will be a sequence length divided by block queue.
TIMELINE
如果序列中有. 1000个元素, 而查询块大小为32
So ifwehave1, 0ooelements in the sequence and the block queue is 32.
> TIMELINE OUTLINE
Iterates over t
curr_q=8 那么步数就是1000除以32.
num_steps = SEQ_ LER
it will be 1, 000dividedby32.
> TIMELINE OUTLINE class Triton Atte
do _ptrs =do +off s_q[:, None ]* stride_seq +off s_din [ None,:]*stride _din
q T_ptrs= Q+offs_q[ None,:]*stride_seq+offs_din[:, None]
352 选择1000不太合适, 应该是1024, 召 否则就无法整除了
Bad choice of 1, 000, itshouldbe1, 024, otherwise it's not divisible.
TIMELINE OUTLINE
352 于是, 在这个for循环中, 我们遍历每个块, 并加载一个查询块
So then we go through each block in this for loop and we load a block of queue,
> TIMELINE OUTLINE
# Iterates over the seq
curr_q=0
nce dimension of the query for blk_idxin
um_steps= SEQ_ LEN 第一个块由我们的指针指示.
the first one indicated by our pointer.
> TIMELINE OUTLINE class Triton Attention (to
355
curr_q=θ
: SEQ_ LEN// BLOCK_ Q 在迭代结束时, 我们将指针移动到下一个查询块,
And at the end of. the iteration, we will move it to the next block of queue.
TIMELINE OUTLINE @static method
curr_q=θ
ps= SEQ_ LEN// BLOCK_ Q 好的, 我们还会加上存储在 M矩阵中的logsum exp 值
Okay, we'll add also. the. log sum exp values that are stored in the M matrix > TIMELINE OUTLINE @static method
T _block =tl. load 因为我们想要实时计算 PT.
OUTLINE class Triton Att because. we want to compute on the fly PT > TIMELINE @static method
PT 是查询与键相乘后经过soft max 的转置
PT is the transpose of the soft max of query multiplied by the keys,
> TIMELINE OUTLINE @static method
但我们希望避免先计算查询与键的转置相乘
but we want to not take query multiplied by the transpose of the key > TIMELINE OUTLINE @static method
# Load a block of Q
q T _block =tl. loa d(q T_ptrs)
offs_q= curr_q+tl. arange # Load the logs u values for 再进行转置操作.
OUTLINE class Triton Attention (torch. auto gra and. then do the transpose.
> TIMELINE @static method
off s_q 由于我们已经以转置形式访问 Q
OUTLINE class Triton At tent We just already access Q as a transpose,
> TIMELINE 365 @static method
因此可以直接计算 PT而无需先计算 P再进行转置操作
so we can already compute PT :instead of computing P and then transposing it.
> TIMELINE OUTLINE @static method
357 因此, 我们从logsumexp矩阵(即前向传播过程中计算的 M矩阵) 中
So we load the offsets. of. the elements that we need from this log sum exp matrix.
> TIMELINE OUTLINE @static method
# Load a block of Q
offs_q 加载所需元素的偏移量
361
which is. the M. matrix that we computed during the forward pass.
TIMELINE OUTLINE @static method
# Load a block of Q
offs_q 并且我们每次访问一个查询块
OUTLINE
364
365
class Triton At tent And we access a block of queue at a time,
> TIMELINE @static method
q T_block= tl. load(q T_ptrs)
offs_q 即当前迭代中正在处理的那个块.
the one we are currently working with in the iteration.
> TIMELINE OUTLINE @static method
# Load the logs
OT1 接着, 我们访问一个查询.
> TIMELINE OUTLINE @static method
xp va tues 由于键已经转置, 我们直接进行相关计算
class Triton atte tion key o transposed already, so we do the.
> TIMELINE OUTLINE @static method
ies in the # This gives QK_ T_bloc 如果你想得到 PT, P应该是.
class Triton Attention (
if you. want to get the pt, p should be.
> TIMELINE OUTLINE @static method
=tl. load( M+offs_q) 实际上这不是 P, 因为我们还没有进行softmax操作, 它实际上是 ST.
this is actually not. p because we didn't do the soft max, it's actually st.
> TIMELINE OUTLINE @static method
361
=t1. load( M+offs_q) 不过, 如果你想要得到 PT, 你需要对 ST进行softmax 操作
but okay, if you want. to get the pt, you need to get the soft max of st.
> TIMELINE OUTLINE @static method
对 ST 进行soft max 操作的结果就是它
lass Triton Attention (torch. auto g The Soft max of st is what it's > TIMELINE OUTLINE 368 static method
off s_q=curr _q+tl. a range (0, BLOCK_0)
n= tl. load( M+of fs_q)
QK_ T_block= softmax _scale *tl. dot
# This gives us ( QKT) T =( KT) T(
A是 S的转置.
class Triton attention torch. auto grad. Functi ca :is transposed of s.
> TIMELINE OUTLINE @static method
m=tl. load ( M +off s_q)
S是查询query 广乘以它的转置
what is s is-a query multiplied by transport of this.
TIMELINE OUTLINE @static method
361
=tl. load( M+offs_q) 因此, 要得到 ST, 你需要对键( Key)进行转置, 而不是键乘以查询的转置
so to get st you need to do key transposed, no key multiplied by query transposed > TIMELINE OUTLINE @static method
off s_q=curr_q+tl. arange(0, BLOCK_0)
n=tl. load( M+offs_q)
QK_ T_blo
# This giv 正如你所记得的, 在矩阵乘法中
class r so as you t remember, in the matrix multiplication,
> TIMELINE OUTLINE @static method
off s_q=curr _q+tl. a range (0, BL0 CK_0)
n=tl. load( M +offs_q)
QK_ T_block This gives u 如果你对矩阵乘法进行转置
if you. transpose the matrix multiplication OUTLINE 367
368
class Triton Atter
> TIMELINE @static method
=tl. load ( M +off s_q) 你也需要将矩阵乘法中的两个元素进行交换.
you need to also invert the two element in the matrix multiplication.
TIMELINE OUTLINE @static method
361
=tl. load( M+offs_q) 这就是为什么我们要进行键乘以查询转置的操作.
so that's why we are doing a key multiplied by query transposed > TIMELINE OUTLINE @static method
ies in the current blocl
n=tl. load( M+offs_q)
QK_ T_block This gives 这样我们就能得到 S 的转置,
OUTLINE class Triton Attention (torch.
this wil give us s transposed.
> TIMELINE a static method
36)
=tl. load( M+offs_q) 在应用之前我们还会用softmax的缩放因子对其进行缩放
we are also scaling it with the soft max scale Before we apply the.
> TIMELINE OUTLINE @static method
361
=tl. load( M+offs_q) 要应用softma×我们只需对每个元素取指数, 减去其最大值
to apply the soft max we just need to do the exponential of each element minus its maximum TIMELINE OUTLINE @static method
off s_q=curr_q+t1. arange(θ, BLOCK_0)
n= tl. load ( M+of fs_q)
# This gives us ( QKT) T
QK_ T_block= softmax _scale 再除以归一化值.
divided. by the normalization value.
OUTLINE class Triton Attention (tor > TIMELINE @static method
16
:tl. load( M+offs_q) 但使用对数求和技巧时我们只需将每个元素减去m值
but with the log sum X trick we just need to each element subtracted by them value,
> TIMELINE OUTLINE @static method
这个m值已经包含了归一化因子
ass Tr which already. includes the normalization factor.
> TIMELINE OUTLINE 376 static method
This give Q K_ T_block 我想我已经推导过这部分了
i think i already did the the derivation of this.
OUTLINE class Triton > TIMELINE @static method
# This gives Q K_ T_block P_ T_block apply 所以我们不需要再重复一遍.
class Triton atte So we dont need to go through that again.
> TIMELINE OUTLINE 370 @static method
QK_ T_block x_scale*tl. dot( K_block, q T_block) 好的, 现在我们实际上有了 pt块
OUTLINE
3723
class Triton k okay, so. now we have the pt block actually.
> TIMELINE @static method
所以在这个公式中, 我其实应该写成st.
so in this-formula i should have written st actually > TIMELINE OUTLINE @static method
好的, 那么在实现因果注意力机制时, 我们还需要屏蔽掉一些值
okay, then when doing the. causal attention, we also need to mask out some values, so,
> TIMELINE OUTLINE @static method
# We apply the soft max by P_ T_block =tl. math. exp 正如你在这里所看到的
OUTLINE
371
class Triton Attention t torch. auto grad. Funct ira S you Can See here > TIMELINE @static method
off s_q[ Nc
,:]>= offs_kv[:, None]
373 因此, 在这种情况下, 因果掩码是在计算完so ftmax之后应用的
soin this case the causal mask is applied after the soft max has been computed.
> TIMELINE OUTLINE @static method
of fs_q[ Non
one,:]>= offs_kv[:, None]
375
Replace all the masked this case we 因为在这个步骤中
because during. this. one is you ·are used to compute the apply the soft, the causal mask
7
OUTLINE TIMELINE @static method
off s_q[ None,:]>= of fs_kv[:, None]
373 你通常是在计算softmax之前应用因果掩码的
378
before. computing the soft max detection.
OUTLINE 379
380
lass Triton Attent
> TIMELINE 81 @static method
of fs_q[ Nc
he,:]>= offs_kv[:, None] 但这实际上是在前向传播过程中进行的
OUTLINE class Triton but this is actually during the forward pass,
TIMELINE @static method
of fs_q[ Nc
e,:]>= offs_kv[:, None]
373 因为你不希望归化因子受到那些本应为零的元素
because you don't want the normalization factor to be affected by the element TIMELINE OUTLINE @static method
off s_q[ None, :]>= offs _kv [:, None ]
# Replace at ll the masked values with 0.
)# Shape:( BLOCK_ KV1, BLOCK_ Q1)
In this case we do not need to 的影响
ited the normalization factors (stored in "m")
P_ T_block =tl. here mask _block, P _ T_bl
OUTLINE class Triton Attention torch. auto grad. Functi that should be zero.
> TIMELINE @static method
of fs_q[ Nc
ne,:]>= offs_kv[:, None ] 但我们已经计算了归一化因子, 所以它不会再受到影响了.
But we already computed. the. normalization factor, so it can not be affected anymore.
> TIMELINE OUTLINE @static method
of fs_q[ No
one,:]>= offs_kv[:, None]
373 因此, 我们可以在应用:soft max 之后进行掩码操作
OUTLINE 379
class Trit Sowecanmaskout after applying the software > TIMELINE 81 @static method
of fs_q[ Nc
e,:]>= offs_kv[:, None]
373 因为归化因子已经基于我们之前应用的掩码计算
because the normalization factor has already been calculated OUTLINE TIMELINE @static method
off s_q[ None,:]> offs _kv [:, None ]
)# Shape :( BLOCK _ KV1, BLOCK_ Q1)
P_ T_block=tl. where [mask _block, P_ T_b
In this case we do not need t factors (stored in "m ")
based on. the :fact that we applied the mask OUTLINE 379
380
class Triton A TIMELINE @static method
of fs_q INc
e,:]>= offs_kv[:, None] 这就是为什么我们可以在应用softmax之后再进行掩码操作.
And that's why we can apply it after applying the soft max.
> TIMELINE OUTLINE @static method
of fs_q[ Nc
ne,:]>= offs_kv[:, None] 嗯, 所以掩码始终是相同的
factors (stored in "m ")
OUTLINE class Triton Attention (to r
uhso the mask is always the same > TIME LINE static method
off s_q
:]>= offs _kv [:, None ] 因此, 如果查询的索引超过了某个值
so if the query. is more than the, the index of the query > TIMELINE OUTLINE @static method
of fs_ql
:]>= offs_kv[:, None]
373 那么在这种情况下:掩码对于所有不需要被屏蔽的值来说都是有效的
so the mask is true in. this-case for all the values that do not need to be masked.
> TIMELINE OUTLINE @static method
374 所以;所有不需要被屏蔽的值就是这里的这些
soall the values-that. do not need to be masked are these ones here,
TIMELINE OUTLINE @static method
of fs_q[ Nc
ne,:]>= offs_kv:, None 而其他所有值都将被替换为零
and all the other value will be um, with replaced with the zeros.
TIMELINE OUTLINE static method
374 好的, 所以在我们已经对pt块进行掩码操作之后, 京 就可以计算dv了:
altright, so after we have-the pt block already masked, we can calculate dv, dv.
> TIMELINE OUTLINE @static method
# Replace all the masked values with @.
# Shape :( BLOCK _ KV1, BLOCK_ Q1)
P_ T_bloc
In this 我会在论文中指出正确的公式.
iwill write, i will point to the right formula in the paper.
TIMELINE OUTLINE @static method
do _block =tl. lc
ording 所以我们加载一个d的块吗?
dv_block
8
class Triton Attention torch. auto grad. Fu SO We load a block of d?
TIMELINE OUTLINE @static method
do _block =t l. loa(do_ptrs)
nintarisasino
dby pointer d V _block +=tl. dot P_ T_block. to(tl. float16), do_b1
According to the formula :
=d V _old +
the matrix multiplication OUTLINE class Triton Attention (torch. auto grad. Function ):
TIMELINE @static method
do _block 为什么我们不加载一个d的块呢?
dv_block class Triton Attention torch. a Why u do : We not load a block of d?
TIMELINE OUTLINE 389 static method
d V _block +=tl. dot P_ T_block. to (
According to the formula : 让我们看看论文.
class ri to natte tion to rca to grad. Fun let's look at the paper.
TIMELINE OUTLINE @static method
P_ T_block Return a tensor of data whe
d V_block +=tl. dot[ P_
cording to the fo 那么如何计算dv块呢?
sohow to compute the dv block OUTLINE class Triton Attention (torch.
TIMELINE @static method
I We see that similar to the forward pass, the backward pass performs O( N2) FLOPs and only requires We analyze the IO-com
dpass ( Theorem 2)
Theorem5. Let Nbethsohow to. compute the. dv blocks r AM with d≤ M≤ Nd.
Standard attention ( Algorithm O )backward pass requires O( Nd+ N2) HBMaccesses, while FLAs H ATTENTION
O( N) extra memory beyond inputs, output, output gradient, and input gradients.
We see that similar to the forward pass, the backward pass performs O( N2) FLOPs and only requires We analyze the IO-complexity of the backward pass, similar to the forward pass ( Theorem 2).
Theorem 5. Let N be the scquence length, d be the head dimension, and M be size of SRAM with d < M≤ Nd.
Standard attention ( Algorithm O )backward pass requires O( Nd+ N2) HBMaccesses, while FLAs H ATTENTION backward pass ( Algorithm 4)requires O( N2d2 M') HBMaccesses.
s. the backward pass performs O( N2) FLOPsand only requires 它是一个累加和
so the dv block is. computed as the old dv plus y so arepeated sum ; as you can see Standard attention ( Algorithm O ) backward pass requires O( Nd+ N2) HBM accesses. uhile FLAs H ATTENTi ON
P_ T_block Re tua tensor of data whose values are loaded d V _block b=tl. dot[ P_ T_block. to(tl. f 日的dv加上 PT.
asyou can see it's here-plus equal The old DV plus PT > TIMELINE OUTLINE @static method
end for 旧的dv加上 PT.
Write d Kas youltan see jt's here, plus lequal The old DV plus P T.
endfor
end for Write d K; ← d K;, d V;← d V; to HBM.
endfor
en 所以这里 PT. 的下标表示应用了dropout 之后的 PIJ.
Writ so here PT dropped j indicates the Pi D after applying the drop out.
end for
end for Write d K; ← d K;, d V;← d V; to HBM.
endfor
end for 在这个实现中, 我们并不支持dropout,
Write d K in this l implementation, we don't support the drop out end for
er 而且实际上很少有模型会在注意力机制中使用dropout.
Wr and also very few m6dels actually use the drop out in the attention.
end for
end for Write d K; ← d K;, d V;← d V; to HBM.
endfor
end for 所以 PT 乘以doi.
Write d K ; ← d K ;, d V ;so-ptl Myltiplie Bby. doi.
end for
end for Write d K; ← d K;, d V;← d V; to HBM.
endfor
end for 所以doi 的块和doi 是同一个块.
Write d K ; s Ua blbck ofdoivanddoiislthe same block.
end for
On chip, compute Write d Qi
Write d Q; ← d Q; + rd St; K; e RBrxd to HBM.
endf
Write d Q;←d Q;+那应该也是;嗯. HBM.
On chip, compute dthat shouild bealsolum. e RBexd
end for
Write d Q; ← d Q; + rd S;; K; e RBrxd to HBM.
endfor
Wridoic和-kid Qgi始终指向各自张量中的同一行块.
On chip, compute d K; ←- dkgof rd SQi e RBexd.
endfor
Write d Q; ← d Q; + rd S;; K; e RBrxd to HBM.
endfor
这就是原因, 因为内层迭代中的i 表示g的 F个块和 O的一个块
that's why, l because this inner iteration i indicates, a block of g and a block of O,
end for
Write d Q;
d Q; + rd S;; K; e RBxd to HBM.
Write d Q but they are always referring
Write d Q; - d Q; + rd S;; K; e RB, xd to HBM.
On chip, co 因为 DO和 DQ的形状是相同的
Bx B
Writed Q
because On chip, compute dk j
ER
Write d Q; - d Q; + rd S;; K; e RBxd to HBM.
RBX B 因此,
so we go through the l blocks of yan dl ously because one is needed for dv.
需要 DO而计算 PK则需要
RBx B
Q.
sofor DV
2:
On chip, compute d K; ← d K; + rd S
Qi ∈ RB cxd
3:
end for
4:
Write d K; ← d K;, d V;
d V; to HBM.
P_ T_block
->tensor
377 这就是为什么我们要按照论文中的方法计算 DV
And that's why we. compute the DV as follows, just like from the paper.
> TIMELINE OUTLINE @static method
P_ Tblock =tl. where (mask_block, P_ T_block, e. 0)
In this case we do not need to mask do _block =tl. load (
According to the 所以用 PT块乘以 DO.
d V_block tl. dot( P_ Tb
So PTblockmultiplied by Do OUTLINE 384 class Triton Attention (torch. aut > TIMELINE @static method
2:
3:
end for
4:
Write d K ; - dk So PT block multiplied by. Do.
2:
3:
end for
4:
Writeyo
can see, it's a P transpose multiplied by do block.
P_ T_block =tl. where (mask _block, P_ T_block, 0. 0)
do_block=t
Accord in 这样, 我们就计算好了 DO块
Load
dv_block=
OUTLINE
384
class Triton Attention (tor
Sowe have computed the Do block TIMELINE static method
do _block =tl. load (do _ptr s) 接下来, 我们需要加载预先计算好的 DI元素.
Then we need. to. load the : Dl element that we pre-computed initially TIMELINE OUTLINE static method
第一次调用函数时, 称为注意力反向预处理器
with the first call. to the function called the attention backward preprocessor > TIMELINE OUTLINE
d0_hlock =tt. load(d0_ptrs)
assert HEAD _0 IH_ Q = HEAD _ DIM_ K and HEAD_ DIH_ K == HEAD_p IM_ V
0=torch. empty_like( Q 因为计算dk时需要用到它,
OUTLINE triton. cdiv( SEQ_ LEN
BATCH_ SIZE* NUM_ HE
because we will need it for dk > TIMELINE 1,# Z in the CUDA laur
h gric
BATCH _ SIZE, NUM _ HEADS, SEQ _ LEN, HEAD _ DIM assert HEAD _ DIM_ Q = HEAD_ DIM_ K and H
0=torch. empty _like ( Q) 让我们看看.
stage =3if causal else 1
grid= lambda args :(
Let's see.
> TIMELINE OUTLINE #cei L ( SEQ _ LEN / BLOCK _ SIZE _ Q )= How
BATCH _ SIZE, NUM_ HEADS, SEQ_ LEN, HEAD_ DIM = Q. shape assert HEAD _ DIM_ Q = HEAD_ DIM 我们加载了多少个?
grid-= tabda arg How many of them we are loading?
stage > TIMELINE OUTLINE triton. cd iv ( SEQ _ LEN, args BLOCK _ SIZE _0"1),#which
cei L( SEQ_ LE
ueries are we going to work with?
BATCH _ SIZE, NUM_ HEADS, SEQ_ LEN, HEAD_ DIM = Q. shape assert HEA 加载的数量与查询数量完全相同
Exactly the same number of query > TIMELINE OUTLINE triton. cd iv ( SEQ _ LEN, args BLo CK _ SIZE _o ]),#which group of queries are we going to work with?
cei L ( SEQ _ LB
@static method 因为我们总是加载相同数量的块大小微向量.
def forward (ctx
394
that we load because we load always. the same number of blocksize micro number of vectors.
> TIMELINE OUTLINE
@static method def forward (ctx, 好的, 我会复制一些内容并逐步解释.
Okay, l will copy some-stuff and explain it step by step.
> TIMELINE OUTLINE 0=torch. empty_like( Q)
static method 391
def forward(ctx, 那么, 接下来我们需要进行的操作是计算这个dk.
8
So, the next operation that-we need to do is to compute this dk TIMELINE OUTLINE 0=torch. empty_like( Q)
q T_ptrs+= BLOCK _ Q *stride_seq
do_ptrs += BLOCK_ Q*stride_seq
curr_q+= BLOCK_ Q
> TIMELINE OUTLINE class Triton Attention (torch. auto grad. Function ):
@static method
pointers.
T_ptrs+
BLOCI 为了计算dk, 我们需要dst.
class Triton tte tion to rc. auto g To compute dk, we need dst.
> TIMELINE OUTLINE @static method
# Increment pointers.
curr_q+= BLOCK_ Q
do_ptrs+= BLOCK_ Q*stride_seq > TIMELINE OUTLINE class Triton Attention (torch. auto grad. Function ):
@static method
curr_.
T_ptrs 为了计算dst, 我们需要得到 dp.
class Triton Attention to r To compute dst, we need to get dp.
> TIMELINE OUTLINE 403
gstatic method
q T_ptrs+= BLo CK _ Q*strie
d0_ptrs+= BLOCK_ Q*str 所以我们一步一步来.
OUTLINE class Triton Attention (torch. auto grad. Fun So let'sgo one by one.
> TIMELINE @static method
d K _block+=sof tmax _scale *tl. dot (d S_ T_block, tl. tra 我们从这些公式的末尾开始, 逐步回溯到它们的起点, 这样我们就能理解
Let'sgo from the end to the beginning of these formulas so we don't understand > TIMELINE OUTLINE @static method
curr _q a T _ptrs 每个部分是如何被使用和创建的
where everything is used to where everything is created > TIMELINE OUTLINE 40 Z @static method
curr _q a T _ptrs 那么, 我们从decay 开始吧.
class Triton Attention (torch. auto grad.
So. let'sstart from decay.
> TIMELINE OUTLINE @static method
q T_ptrs+= BLOCK _ Q *stride_seq
d O_ptrs+= BLOCK_ Q*stride_seq
curr_q+= BLOCK_ Q
OUTLINE class Triton Attention (torch. auto grad. Function ):
> TIMELINE @static method
curr_q+= BLOCK_0
q T_ptrs+= BLOCK_ Q*stride_seq 如果你看论文,
do_ptrs+= BLOCK_ Q*stride_seq
class Triton Attention (torch. auto grad. F u
lfyou look at the paper,
> TIMELINE OUTLINE @static method
O( N)extra memory beyond inputs, ou ale and input gradients.
We analyze the IO-complexity of y du lool at the paper he forward pass ( Theorem 2).
The orem5. Let Nbethesequence length, dbe the head dimension, and Mbe size of SRAM with d≤ M≤ Na
We see that similar to the forward pass, the backward pass performs O( N2) FLOPs and on ly require
O( N)extra memory beyond inputs, output, output gradient, and input gradients.
We analyze the IO-complexity of the backward pass. similar to the forward pass ( Theorem 2)
The orem5. Let Nbethesequence length, dbe the head dimension, and Mbe size of SRAM with d≤ M≤ Na
decay 等守旧的decay 加上ds的转置乘以o的=个块yrquire
detayisegualito'the old decay plus ds transpose t multiplied by'a block of Q.
The orem5. Let Nbethesequence length, dbe the head dimension, and Mbe size of SRAM with d≤ M≤ Na
24:
Write d K; ← d K;, d V; ← d V; to HBM.
25:
end for
26: Return d Q, d K, d V.
24:
Write d K;
d K;, d V;. d V; to HBM.
25:
end for 嗯, 这就是这里所写的内容.
26: Return d Q, d K, ymand this is what is written here.
curr _q+= BLOCK _ Q
q T_ptrs+=
BLOCK 嗯这就是这里所写的内容.
d0_ptrs += BLOCK
class Triton attention to rum an do this is what is written here.
> TIMELINE OUTLINE @static method
所以这里的加等号基本上就是旧的加上新的
so it is plus equal means basically just the old plus the new,
TIMELINE OUTLINE @static method
curr _q += BLOCK _ Q
q T_ptrs+= BLOCK_ Q 这是一种增量累加的方式.
d0_ptrs +=
OUTLINE class Triton Attention (to rcl some-its an incremental addition > TIMELINE @static method
所以用这里的新内容来更新旧的k.
so increment the. old. k with some new stuff, which is this stuff here.
99 > TIMELINE OUTLINE @static method
所以soft max 的缩放因子(这里用tau 表示)
so the soft max scale multiplied because also there is a soft max scale, this tau here,
> TIMELINE OUTLINE @static method
24:
Write d K;
d K, d V; - d V; to HBM
25:end 所以soft max 的缩放因子(这里用tau 表示)
so the soft max scale multiplied, because also there is a soft max scale, this tau here,
24:
Write d K; ← d K;, d V; d V to HBM.
25:
end for
26: Return d Q, d K, d V.
# Increment pointers.
curr_q+= BLOCK_ Q
q T_ptrs+= BLOCK_ Q*stride_seq
d0_ptrs+= BLOCK_ Q*stride_seq
OUTLINE class Triton Attention (torch. auto grad. Function ):
> TIMELINE 483 @static method
curr _q+= BLOCK_ Q 会乘以 dst块
do_ptrs+= BLOCK_ Q*stride_seq class Tri to attenti o multiplied by the matrix multiplication > TIMELINE OUTLINE @static method
q T _ptrs += BLOCK _ Q 与 Q转置的矩阵乘法结果
between dstblock and the transpose of um and andq And you Q > TIMELINE OUTLINE @static method
24:
Write d K;
d K,, d V- d V; to HBM.
25:
end for 与 Q转置的矩阵乘法结果.
26:
Rbetween dst blockand the transpose of um and and q And you Q.
24:
Write d K;
d K, d V ← d V ; to HBM.
25:
26: Return dc you can see here this Q, but we don't have a Q.
d K _block+= sof tmax _scale *tl. dot (ds_ T_block, tl. trans (q T _block ))
According to the form 你可以在这里看到这个 Q, 但我们目前还没有 Q
class Tri to you can see here this Q, but we don't have a Q
OUTLINE
401
> TIMELINE @static method
395
d K_block += softmax_scale*tl. dot (d S_ T_block, tl. trans (q T _block 我们有一个( 的转置所以我们对 Q转置再取一次转置, 它就变回 Q了
We have a Q transpose So. we. take the transpose of Q transpose and it becomes back Q now.
OUTLINE TIMELINE @static method
395 让我们来看看这个 Dst块. 根据论文中的公式, Dst的计算方式如下
Let's look at this D st block Dstu is calculated as follows so in the formula of the paper.
> TIMELINE OUTLINE @static method
curr_q+= BLOCK_0
# Increment pointers.
q T_ptrs += BLOCK _ Q *stride_seq do_ptrs+= BLOCK_ Q*stride_seq 我们有 Ds.
We have Ds.
OUTLINE class Triton Attention (torch. auto grad. Function ):
> TIMELINE @static method
24:
Write d K; ← d K;, d V; ← d V; to HBM.
25:
end for
26: Return d Q, d K, d V.
24:
Write d K;
d K, d V; ← d V, to HBM.
25:
end for 这里的 Ds等于:对,
26: Return d Q, d K, d Ds is here, it is equal to : yeah,
24:
Write d K;, d K ;, d V← d V; to HBM.
25:
endf就是这里, 它等于一个块pij 逐个元素乘以
26:
Reitis here, and jtisequal to a block p i j multiplied element wise
24:
Write d K; ← d K;, d V; ← d V; to HBM.
25:
end for
dpi 减去 di.
26: Return d Q, d K, d V. with dpi minus d i now,
24 :
Write d K; d K;, d V; ← d V ; to HBM
25:
end不过, 我们不需要’s, 我们需要的是s的转置.
26: Return um, we don't need the s, we need the s transposed.
不过我们不需要s, 我们需要的是s的转置
umwe don't need the s, we need thes transposed > TIMELINE OUTLINE @static method
所以计算:的转置时, 这是一个逐元素乘法
so to compute. the s. transpose, this is an element wise multiplication,
> TIMELINE OUTLINE @static method
q T_ptrs += BLOCK _ Q *stride_seq do_ptrs+= BLOCK_ Q*stride_seq 而不是矩阵乘法.
class Tr to natte tion torch at ogr not a : matrix multiplication,
> TIMELINE OUTLINE @static method
# Incr 这意味着当你对这个操作取转置时
which means that when you take the transport of this operation > TIMELINE OUTLINE static method
# Increment pointers.
curr_q+= BLOCK_ Q
q T_ptrs+= BLOCK_ Q*stride 不需要反转任何东西
class Triton Attention (torch c you don't need to invert anything,
> TIMELINE OUTLINE @static method
curr_q
q T_ptrs 只需对两个操作数取转置即可.
you just ·need to take the transpose of the two operands.
> TIMELINE OUTLINE @static method
395
Incre 因此, 为了计算st我们对p取转置, 得到pt, 这我们已经有了
so to compute the st we take the transpose of p, which is apt, and we already have that > TIMELINE OUTLINE @static method
a T _ptrs 然后对括号内的所有内容取转置,
and then the transpose of everything that is inside of the parentheses.
> TIMELINE OUTLINE @static method
# In cr 么, 这个dpt减去di 在这里我们将行与列互换, 所以这个dpt是什么呢
sothisdptminusdi, where. we invert the rows with the columns, so this dpt is what?
> TIMELINE OUTLINE @static method
Inc re 在论文中我们知道dp的公式:dp在这里, 它等于d.
well, in the paper, we. know. the formula for dp: dp is here and it is equal to d.
> TIMELINE OUTLINE @static method
24:
Write d K;_d K;d V ; - d V ; to HBM.
25: 在论文中, 我们知道 dp的公式:dp在这里, 它等于d.
well pin the paper d we know the formula for dp: dp is here and it is equal to d.
24:
Write d K;
d K;, d V; ← d V; to HBM.
25:
end for
26: Return d Q, d K, d V.
24:
Write d K;. d K;, d V; ←, d V; to HBM.
25:
endf等一下, 这里的dp等于 Do乘以 V的转置.
26:
Rewe wait. dphere and it is equal to Do multiplied by V transpose.
下这里的dp 等于 DO乘以 V的转置.
we wait dp here and it is equal to Do multiplied by V transpose.
> TIMELINE OUTLINE @static method
但我们不需要 DP, 我们需要的是 DPT.
OUTLINE class Triton k But We dont need the DP, we need the DPT.
> TIMELINE @static method
在这种情况下这不是逐元素乘法, 而是矩阵乘法.
And in this case, its not an element-wise multiplication, it is a matrix multiplication.
> TIMELINE OUTLINE @static method
d K _block += sof tmax _scale *tl. dot (d S_ T_block, tl. trans (q T_block))
curr_
a T_ptr 因此为了得到 DPT而不是 DP
Soin. order to get not a DP, but DPT,
OUTLINE class Triton Attention (to > TIMELINE @static method
24:
Write d K;. d K ;, d V ;←d V ; to HBM.
25:
endfor 因此, 为了得到 DPT而不是 DP,
26: Return d Q, d KSo in order to get not a DP, but DPT,
24:
Write. d K; ← d K;, d V; d V;to HBM
25:
5:end 我们需要对这个矩阵乘法的两个操作数取转置.
we need to take the transpose of these two operands of this matrix multiplication.
24:
Write d K; d K; d V
d V; to HBM.
25:
end for 在矩阵乘法中, 当你取转置时,
26: Re And in the matrix multiplication, when you take the transpose,
24:
Write d K;
d K; d V= d V to HBM
25:
endfor 还需要交换两个操作数的顺序.
26: Ret urryauneed, to also invert the order of the two operands.
24:
Write d K; ← d K;, d V;
d V; to HBM.
25:
end for
26: Return d Q, d K, d V.
24 :
Write. d K;
d K;, d V; d V ; to HBM. 因此,. 我们需要对 VT取转置, 它就变成了 V, 也就是 V块,
26: Sowe need to take the VT transpose, which becomes V, so the V block,
因此, 我们需要对 V丁取转置, 它就变成了 V, 也就是 V块
So we need to take the VT transpose, which becomes V, sothe V block.
> TIMELINE OUTLINE @static method
24:
Write d K;. d K ;, d V,
d V; to HBM.
25:
en再与另一个操作数进行矩阵乘法, 即 DOI的转置,
26:
Retumatrix, multiplied by the other operand, so Dol transpose,
d K _block += sof tmax _scale *tl. dot (d S_ T_block, tl. trans (q T_block)) 再与另一个操作数进行矩阵乘法, 民 即 DOI的转置
OUTLINE
401
matrix multiplied by the other operand, so Dol transpose,
TIMELINE @static method
这就是为什么我们要对 DO 进行转置操作
ass rr and thats why we are doing the transpose of Do > TIMELINE OUTLINE 403 @static method
现在我不打算逐一讲解所有指针
Right now I'm ·not going through all the single pointers > TIMELINE OUTLINE @static method
因为我已经告诉过你们如何检查指针指向的内容
because l already. told you how to check what a pointer is pointing to > TIMELINE OUTLINE @static method
curr_q+= BLOCK _ Q # Increme
q T_ptrs+= BLOCK_ Q 以及偏移量所指的是什么.
class Triton Attention (torch and what an offset is referring to.
> TIMELINE OUTLINE @static method
395 希望现在你们对 Triton中指针的工作原理有了更深入的理解
Ihope that now you have a better understanding on how these pointers work in Triton,
> TIMELINE OUTLINE @static method
其实它们在 CUDA 中的工作方式也是一样的
which is also the same way in which they work in Cu DA > TIMELINE OUTLINE @static method
Incre pointers. 因为在(
GPU 中我们只能获取到张量起始地址的指针
because in the GPu we only get a pointer to the starting address of the tensor,
> TIMELINE OUTLINE @static method
然后需要自己计算出所有的索引.
class Tr it and. then we need to work out all these indices.
> TIMELINE OUTLINE 403 static method
我们已经计算完了dk 块, 现在转到下一个查询
we have computed. the dk block so we now go to the next query,
> TIMELINE OUTLINE @static method
q T_ptrs+= BLOCK_ Q*str id
BLOCK_ Q*str 也就是下一个查询块,
to the next block of queries, and so the next block of queries,
TIMELINE OUTLINE @static method
Inc curr _q += BLOCK _ Q pointers. 因为我们固定了k和v块, 并遍历所有查询
because we are fixing k and v blocks and we are iterating through all the queries,
> TIMELINE OUTLINE ef forward(ctx,
Q, K, V, cau
curr _q += BLOCK _ Q # Increment pointers. 所以需要移动查询, 通过步长序列转置指针
so we need to move the query, transpose the pointers by stride sequence,
> TIMELINE OUTLINE def forward (ctx, Q, K, V, causal, soft max _scale ):
curr_q+= BLOCK_ Q
9 这意味着我们如何从一个查询转到下一个查询.
which means that how can we go from one query to the next TIMELINE OUTLINE
curr _q+= BLOCK _ Q
pointers.
398
q T_ptrs += BLOCK_ Q*strideyseq
do_ptrs+= BLOCK_ Q*stri 我们与当前块g相乘
and we multiply with the current block q,
TIMELINE OUTLINE @static method def forward (ctx, Q, K, V, causal, soft max _scale ):
curr _q + BLOCK _ Q
# Incren
398
q T_ptrs+=
do_ptrs+=
BLOCK _ Q*stride_seq
BLCK_q*stride_seq 是一个向量,
class Triton Attention (torch. auto grad. Functi which is a vector which > TIMELINE OUTLINE def forward (ctx, Q, K, V, causal, soft max _scale ):
@static method
curr_q
399 指示我们正在访问的g中当前元素的指针
indicates the pointers to the current element in q that we are accessing,
> TIMELINE OUTLINE forward (ctx, Q, K, V, causal, soft max _scale ):
对. 也进行同样的操作, 并使用块 g 作为元素和步长 q, 因为 d、
and we do it also for theo and we use the block gas element and the stride g, be caused TIMELINE OUTLINE
curr_q+a BLOCK _ Q
q T_ptrs+=
BLOCK
* stride_s
do _ptrs += 和g都具有相同的形状,
o and q all have the same shape.
auto grad. F > TIMELINE OUTLINE @static method def forward (ctx, Q, K, V, causal, soft max _scale ):
+= BLOCK _ Q
397 在我们运行完所有查询的for循环后, 就可以存储这个dk和dv块了
After we have run the for loop of all the queries, we can store this dk and dv block.
> TIMELINE OUTLINE
curr _q
q T_ptrs LOCK_ Q* stride_seq
do_ptrs 因此, 我们按以下方式将其写回
OUTLINE class Triton Attention (torch.
So we write it back as follows.
> TIMELINE static method
0_pt
BLOCK_ Q* stride_seq 这就是我们函数的结尾了, 伙计们.
OUTLINE class Triton At tent And this is the end of our function, guys.
> TIMELINE @static method
403
tl. store(dv_block _ptrs, dv_block)
* stride _ 因此 我们将dv 块准确地保存在当前dv已经指向的位置
So we save the dv block exactly in the position inside of the current okay dv is already,
> TIMELINE OUTLINE @static method
我相信它已经指向了正确的批次和头
believe ;pointing to the right batch and to the right head OUTLINE 378
> TIMELINE
d0_block=tl. load(do _ptrs )
343
offs_q=tl. arange(0, BL0 CK_ Q)#0, 128, 2*128, 3*128, 4, 5, 6, 7, 因为我们在这里以及dk的情况下都对其进行了递增
because we incremented it here. and also in the case of dk each iteration > TIMELINE OUTLINE 351
dp T_block=tl. dot( V_block, tl. trans (do _block )). to (tl. float32)
Licatior 然后我们需要在序列维度中告诉它应该在何处保存这个k和v块
then we need to tell it in. the sequence dimension where they should save this block of k > TIMELINE OUTLINE
tl. store (d K_block_ptrs, d K_block) 这是由此指示的.
41
class Triton Attention (to rc @static method and v and this is indicated by this one.
> TIMELINE OUTLINE 412
def foward(ctx, Q, K, V, causal, sof
Write back d K.
tl. store(d K _block _ptrs 我们像之前那样创建指针,
we say and we create the, the pointers, just like before.
> TIMELINE OUTLINE 412
def forward (ctx, Q, K, V, causal, sof
scale):
e I-11
d K _block _ptr s=
tl. store(d K _bloc Write back d K. 伙计们, 别让我再重复一遍了.
class Triton Attention (to r don't make me do it again.
OUTLINE 412 @static method def forward (ctx, Q, K, v
guys,
scale ):
TIMELINE
d Kblock_ptrs=d K+offs_kv:, None】*stride_seq +off s_din [ None,:】* stride _d in
tl. store(d K _block _ptrs, d K_block)
Write back d K.
41
class Triton Attention (torch. auto grad. Function ):
@static method > TIMELINE OUTLINE 412
def forward(ctx, Q, K, V, causal, soft max _scale ):
HF AD DTM O.
Write back d K.
tl. store (d K_block_ptrs, d K_block) 这真的很简单.
41
class Triton Attention (torch. auto grad. Functio @static method it's a really easy.
> TIMELINE OUTLINE 412
def forward(ctx, Q, K, V, causal, soft max _scale ):
Write back d K 如果你像写这个键和值指针向量一样写下来
if you write it. down like you write this vector of key and values pointers,
TIMELINE OUTLINE forward (ctx,
484
Write back d K 实际上它们不是指针, 而是你需要从序列维度中提取的键和值范围
which. is. not pointers, actually, they are a range of them of key > TIMELINE OUTLINE f forward (ctx,
cale )
Write back d K.
tl. store(d K_block _ptrs, d K_block) 你只需添加另一个维度
and value that you need. to take from the sequence dimension, you add another dimension,
TIMELINE def forward (ctx, Q, K, V, causal, sof
scale)
[-1]
d K _block _ptrs Write back d K 也就是列维度, 然后在列中重复每个值
that is. the column, S
so you repeat each value in the columns > TIMELINE OUTLINE 1
forward(ctx, Q,
Write back d K.
tl. store(d K _block _ptrs, 最后在这里添加头维度,
and then. you add the dimension here for the header dimension.
> TIMELINE OUTLINE for w
[-11
484
Write back d K 总之, 在我们计算完指针后, 也就是确定dk和dv应该存储的位置后
anyway, after we have. calculated the pointers, where we should store the dk and the dv,
> TIMELINE OUTLINE
tl. store (dk_block_ptrs, d K_ble 我们将它们存储起来,
41
@static method we store them in the um.
> TIMELINE OUTLINE 412
def foward(ctx, Q, K, V, causal, 5
tmax_scale):
-11
# Write back d K.
d K_bloc k_ptrs
tl. store(d K_bloc 我们将这些指针存储在dv 中
the pointers of um, we store them in the dv um.
> TIMELINE OUTLINE 412
def forward(ctx,
ax_scale )
04
d K_block_ptrs
Write back d K 我的意思是, 我们将它们存储在 DV 张量和 DK张量中
Imean. we store them in the DV tensor and the DK tensor.
TIMELINE OUTLINE 411
412
scale)
e I-11
Write back dk.
tl. store (dk_block_ptrs, d K_block) 我们保存了什么?
410
class Triton Attention (torch. auto grad. Func @static method What do we save?
> TIMELINE OUTLINE 412
def forward(ctx, Q, K, V, causal, softmax_scale):
[-11
d K _block _ptrs =
Write back dk
tl. store(dk_block 我们保存了 DV 块和 DK 块
410
class Triton Attention @static method We save the DV block and the DK block > TIMELINE OUTLINE 412
def fowardctx, Q, K, V, causal, soft
[-11
398
q T_ptrs += BLo CK_ Q* stride_seq 也就是我们在编写的for 循环中逐步修改的部分.
which is the one that. we were incrementally changing in the for loop that we have written.
TIMELINE OUTLINE offs _kv [
offs _dim [
d S_ T_block d S_ T_block
P_ T_block *(dp T _block-Di[ None,:]) 好的, 现在我们完成了这部分
d K_block
q T_ptrs+=
O. kay, now that we finished this one,
> TIMELINE OUTLINE
do_ptrs +=
d S_ T_block = P_ T_block *(dp T _block-Di[ None,:])
d S_ T_block=
d S_ T_block. to (tl. float16) 接下来可以进人下一个函数, 它将处理另一个for循环.
we can go to the. next function that will do the other for loop.
> TIMELINE OUTLINE += BLOCK _ Q *stride _
d S_ T_block = P_ T_block*(dp T _block-Di[ None,:])
d S_ T_block=d S_ T_block. to (tl. float16) 那就开始吧.
d K_block+= softmax_scale *tl. dot (d S_ T
# According to the formula on the curr_q+= BLOCK_ Q
q T_ptrs += BLo CK_ Q* stride_seq So let's do it.
> TIMELINE OUTLINE
d O_ptrs += BLOCK _ Q *stride_seq
24:
Write d K;d K, d V←d V; to HBM.
25:end for
26: Return d Q, d K, d V.
538 好的, 现在我们来处理迭代的第二部分, 也就是这一块内容.
Okay, so now We do the second part of the iteration, which is this one.
> TIMELINE OUTLINE
538 我先复制一下这段代码然后再详细说明它的功能
543
544
So-let me just copy it and then we describe it.
> TIMELINE OUTLINE 545
( BATCH _ SIZE, NUM_ EADS,
normal_(mean =0.θ, std=0. 5 我们把它写在这里吧.
. requires_grad_()
Let's write it here.
> TIMELINE
OUTLINE
548
( BATCH_ SIZE, NUM_ EADS, SEQ_ LEN, HEAD_ DIM), dtype=dtype, device=cuda
npty
HEAD _ DIM =ctx. HEAD _ DIM,
BLOCK _ KV = BLOCK _ SIZE _ MICRO STAGE =stage 好的, 我们沿用之前的启动网格配置.
Okay, we use the same launch grid as before.
> TIMELINE OUTLINE 5
test_op( BATCH_ SIZE
HEAD _ DIM =ctx. HEAD _ DIM,
BLOCK _ KV = BLOCK _ SIZE _ MICRO STAGE =stage,
_sta qe 我们得先声明这个函数.
Of course, we need to declare this function.
> TIMELINE OUTLINE 5
test_op( BATCH_ SIZE, N
HEAD _ DIM =ctx. HEAD _ DI,
BLOCK_ KV = BLOCK _ SIZE _ MICRO STAGE =stage, 再次强调, 由于网格是根据块大小宏定义的
And again, because the grid is defined for the block size macro for > TIMELINE OUTLINE
attn_bwd 而我们要保持这一部分固定不变,
V= V
softma
sawhat is the thing that we keep fixed.
> TIMELINE OUTLINE dodo,
528 接着, 在四重迭代的内部, 我们以微块大小为步长进行处理.
And then in the side of the-four iteration, we do, steps of blocksize micro 福
> TIMELINE OUTLINE
test _op ( BATCH _ SIZE, NUM _ HEADS, SEQ _ LEN, HEAD _ DIM, causal, d type =torch. float16): 这里我们固定 Q不变而遍历 K和 V
OUTLINE
564
ni
this case we are fixing Q and we are iterating through K > TIMELINE 565
test _op BATCH _ SIZE, NUM _ HEADS, SEQ _ LEN, HEAD _ DIM, causal, d type =torch. float 16):
Q( 因为需要计算 DQ.
torch. enpty(
( BATCH_ SIZE, NUM_ HEADS,
requ res grand V because we need to compute D Q > TIMELINE OUTLINE 565
test _op ( BATCH _ SIZE, NUM _ HEADS, SEQ _ LEN, HEAD _ DIM, causal, dtypeatorch. float16): 目前我们已经计算出了 DK和 DV.
Right now we have computed DK and DV > TIMELINE OUTLINE
HEAD _ DIM =ct X. HEAD _ DIM,
STAGE =stage, 好的, 我认为参数设置与之前相同
Okay, I believe the arguments are the same as before,
> TIMELINE OUTLINE def test-opl
stride _head = Q. stride (1),
stride_seq= Q. stride(2), 事实上这也是 Triton 官网原始实现中
HE or and actually this is also the reason why,
BLOCK > TIMELINE OUTLINE STAGE =stage,
hum _warps = NUM _w ARPS,
stride _head =
stride_seq = Q. stride (2)
Q. stride(1)
544 作者决定使用相同for循环但不同参数的原因
547
the original implementation on the Triton website,
OUTLINE 55
int
STAGE =stage > TIME LINE
num_warps = NUM _w ARPS,
stride _head Q. stride (1)
NUM_ HEADS= NU
stride_dim=0 我觉得那样有点让人困惑
SEQ _ LEN= SEQ_
the author decided to use the same for loop but with different arguments > TIMELINE OUTLINE
stride _head = Q. stride (1)
stride_seq= Q. stride(2)
stride_dim= Q. stride(3)
NUM_ HEADS = NUM _ HEADS, 所以我把它们分开了.
SEQ _ LEN= SEQ_ LEN,
and I believe i it was a. little confusing, so that's why I just separated them.
BLOCK _ Q = BLOCK _ SIZE _
> TIMELINE OUTLINE STAGE =stage,
UM _ WARPS
stride _head Q. stride (1)
stride_din 我只是把代码重复写了两遍
HEApp IM-ctx HEAool. just repeated the code twice.
BLOCK KV = BLOCK S
> TIMELINE OUTLINE STAGE =stage,
hum _warps = NUM _w ARPS,
stride _head Q. stride(1), 这个视频的目标是尽可能易于理解
The goal of this video is to be as easy to understand as possible,
547 > TIMELINE OUTLINE warp S = NUM _ WARPS
stride _head = Q. stride (1)
stride_seq= Q. stride(2),
stride_dim= Q. stride(3)
NUM_ HEADS = NUM _ HEADS, 而不是追求最高效率.
SEQ_ LEN= SEQ_ LEN,
Haow-ckanot'to be as efficient as possible.
BLOCK _ Q = BLOCK _ SIZE _ MA BLOCK _ KV = BLOCK > TIMELINE OUTLINE STAGE =stage,
num _warps = NUM _w ARPS,
num _war ps= N
Cut Copy 我们回到这里, 让我再复制一下函数签名
Let'sgo here, so let me copy the signature again.
> TIMELINE OUTLINE torch. en pty (
P
STAGE =stage,
num _warps = NUM _ WARPS,
num _stages = NUM _ STAGES, 我来定义这个函数.
OUTLINE def
test_op( BATCH _ SIZE, NUM _ HEADS,
Q =(
Let me define-this-function.
> TIMELINE torch. en pty (
class Triton Attention (torch. auto grad. Function ):
def forward (ctx, Q, K, V, causal, soft max _
@static method 哒--哒一哒.
HEAD _ DIH _ V= V. shape [-1]
HEAD_ DIM_ Q, HEAD_ DIM_ K= Q. shape[-1],
BATCH _ SIZE, NM _ HEADS, SEO_ LE, HEAD_ DIM = o. Pra-pa-pam.
> TIMELINE OUTLINE 418
assert HEAD _0 IH_ Q = HEAD_ DIM_ K and HEAD_ OIH_ K == HEAD_ DIM_ V
427
BLOCK_ KV:tl. constexpr,
428
429
STAGE:tl. constexpr,
HEAD _ DIM: tl. constexpr,
431
8
432
class Triton Attention (torch. auto grad. Function ):
Okay OUTLINE 433
434
@static method > TIMELINE 435
def forward(ctx, Q, K, V, causal, soft max _scale ):
427 同样地, 我们需要先将查询、 键和值移动到正确的指针位置
so again, we need to first move the query key and value to the right pointers, 福
> TIMELINE OUTLINE def forward (ctx, Q, K, V, causal, soft max _scale ):
def forward (ctx, Q, K, V, causal, soft max_scale):
HEAD DIMV
= V. shape[-1] 这些指针将指向当前程序正在处理的批次
which will point. to. the exact batch and the exact head that we are working > TIMELINE OUTLINE stage 3if causal else
def forward (ctx, Q, K, V, causal, soft max _scale ):
HEAD _ DIM_ Q, HEAD_o IM_ K= Q. shape [-1], K. shape[-1]
HEAD_ DIM_ V= V. shape[-1]
BATCH_ SIZE, NUM_ HEADS, SEQ _ LEN, HEAD 和注意力头.
assert HEAD _ DIH_ Q == HEAD_p IM_ K and with in this program.
OUTLINE 0=torch. empty _like ( Q)
stage =3if causal else 1
> TIMELINE
ctx. causal =causal return O
@staticnatbed 那么, 我们开始吧
0. KV. 0. M=ctx saved _ten assert do. is _contiguous ()
So. let's do it.
> TIMELINE OUTLINE 592
d0torch. enpty_tike(0)
assert 0. stride()= K. stridet1 Vstrioe(
=torch. empty _like ( H )# Shape :( BATCH _ SIZE, NUM _ HEADS, SEQ _ LEN )
pute all the elements D1 让我看看代码在哪里.
= D
> TIMELINE OUTLINE HEAD _ DIM = Ctx. HEAD _ DIM
第一部分和我们之前写的其他for 循环完全一样.
And the first part-is exactly-the same as the other for loop that we have written.
> TIMELINE OUTLINE stage =3if ctx. causal else 1
45a
455
M=to rch. empty
Mis tne togsueexe for ne packvara pass, one for eacn query BATCH _ SIZE, NUN_ HEADS, SEq_ LEN), 我们回到这里, 实际上我只是复制了代码, 所以它完全一样.
So let'sgo here, and really, I just copied, so it's exactly the same.
> TIME LINE OUTLINE
d V += offset _batch _head 我们检查当前的批次和注意力头索引
Triton Att So we check what is the index batch head,
> TIMELINE OUTLINE @static method def forward (ctx, Q, K, V, causal, soft max _scale ):
d V += offset _batch _head 将查询键和值的指针移动到正确的位置.
We move the query key value pointers to the right place.
> TIMELINE OUTLINE def forward (ctx, Q, K, V, causal, soft max _scale ):
将查询q、 键
(k)、值(v)白 的指针移动到正确 的位置, 同时将掩码(m)和归一化因子(d)的指针也移动到正确的位置
the do, the q, the k, thev point'to the right place, the m and d to the right place,
> TIMELINE OUTLINE
+offset _batch _h class Triton Attention (to rch. 和之前的操作完全一样.
8
@static method def forward (ctx, Q, K, V, causal,, soft max _scale )
HEAD _ DIM _ V = V. shape [1]
HEAD_ DIM_ Q, HEAD_ DIM_ K= Q. shape[-1]
exactly like before.
> TIMELINE OUTLINE
D += offset _batch _head_seq 所以我觉得没有必要再重复解释了.
8
lass Triton
estmu So. don't. think. I need to explain that again.
def for war > TIMELINE OUTLINE
452
15
D+=offset_batch_head_ 然后我们加载-个查询(g):的块, 这个块在后续计算中会保持不变
And then we load a. block of g, the one that we wil. keep fixed > TIMELINE OUTLINE
452
D+= offset _batch _head_seq 所以, 查询(q) 实际上我在这里加载了很多内容.
8
458
457
estt'so. the q. Let. me load a lot of stuff here actually.
> TIMELINE OUTLINE 459
Q. shape[-1], K. shape[-1]
好的, 我们定义了在头维度上加载键
( K )
Okay, we defined the offset that we will need to load the blocks of K
478
TIMELINE OUTLINE rad. Function ):
474 和值( V)块所需的偏移量, 因为我们将对键和值进行迭代.
and Pin the head dimension because we are going to iterate in the Kan d V
> TIMELINE OUTLINE
VT_ptrs=
T_ptrs= K+o
Di= tl. load 我们将以转置块的形式访问它们.
TIMELINE OUTLINE 481
d. Function ):
VT_ptrs= V+offs_kv[ None,:1*stride_seq +off s_din :, None ]*stride _din
k T_ptrs= K+offs_kv[ None,:]* stride_seq +off s_din [:, None ]*stride _din
Di=tl. load( D+offs_q)
curr_kv = 8
> TIMELINE OUTLINE lass Triton Attention (torch. auto or ad. Function ):
num _steps = SEQ _ LEN // BLOCK _ KV
因此, 我们
474 不直接访问键( K)和值( V) 而是以转置后的形式访问它们, 民 即 KT和
Soinstead of accessing. them directly as K and V, we access them as KT and VT.
> TIMELINE OUTLINE too rad. Function l :
472
473
access theand V as
74 您知道, 这只需要通过改变步幅
(strides ) 就可以实现.
And you know that that's possible just by changing the strides.
TIMELINE OUTLINE to arad. Function ):
VT_ptrs= V+offs_kv[ None,:]*stride_seq +off s_din :, None ]*stride _din
k T_ptrs= K+offs_kv[ None,:]*stride_seq +off s_din [:, None ]* stride _din
Di=t1. load( D+offs_q)
curr_kv = 8
num _steps = SEQ _ LEN // BLOCK _ KV > TIMELINE OUTLINE lass Triton Attention (torch. autour ad. Function ):
472
473
7
ofls kv 在这种情况下, 因为我们将它们视为二维向量来处理.
In this case, u because we are treating them as 2 D vectors.
TIMELINE OUTLINE 18
ad. Function ):
k T_ptrs = K+offskv[ None,:1 *stride_seq +off s_din :, None ]*stride _din
v T_ptrs= V+offsky[ None,:1*stride_seq +off s_din [:, None ]*stride _din We access the K and V as transposed blocks Di=tl. load( D+offs_q)
curr_kv = 8
> TIMELINE OUTLINE 482 class Triton Attention (torch. auto or ad. Function ):
num _steps = SEQ _ LEN // BLOCK _ KV
474 当我们想访问键(k) E 时, 我们将偏移量kv视为未转置的k.
we treat the. offset kv, when you want to access k, asjust not transpose but k > TIMELINE OUTLINE rad. Function l :
474
k T_ptrs = K+of_kv[ None,:] *stride_seq +off s_din [:, None ]* stride _din # We access the K and V as transposed blocks v T_ptrs= V+offs_kv[ None,:]*stride_seq +off s_din :, None ]* stride _din
Di=tl. load( D+offs_q)
curr_kv = 8
num _steps = SEQ _ LEN // BLOCK _ KV > TIMELINE OUTLINE class Triton Attention (torch. auto or ad. Function ):
473
474 您将这个偏移量k V视为一个行向量, 抱歉, 是列向量,
you treat this offset kv as a row vector, sorry, a columnvector.
> TIMELINE OUTLINE
474 因此, 您需要在行上重复, 每次访问所需的k偏移量
soyou repeat on. the rows, each k offset that you want to access.
> TIMELINE OUTLINE oo rad. Function l :
473
472
474 在这种情况下, 我们将其视为行向量, 因此它将在行上重复.
In this case, we are. treating it as a row vector, so it will be repeated on the rows.
> TIMELINE OUTLINE
v T_ptrs= V+offs_kv[ None,:1 * stride_seq +off s_din [:, None ]* stride _din
k T_ptrs = K+offs_kv[ None, 1 * stride_seq +off s_din [:, None ]*stride _din
Di=tl. load( D+offs_q)
curr_kv = 8
> TIMELINE OUTLINE class Triton Attention f torch. auto or ad. Function ):
num _steps = SEQ _ LEN // BLOCK _ KV
k T_ptrs
Di= tl. load 抱歉, 它将在列维度上进行广播
478
47
umsorry it will be broadcasted on the column dimension TIMELINE OUTLINE 481
rad. Function ):
474
k T_ptrs = K+of fs_kv[ N
v T_ptrs= V+offs_kv
Di=tl. load( D+ 这样您就可以访问转置后的k
478
479
and'that's. howyou can access the transposed version of k
OUTLINE
480
TIMELINE
481
VT_ptrs= V+off s_kv[ None, 以及转置后的
Di=tl. load( D+offs_q)
and how you can access the transposed version of v > TIMELINE OUTLINE 482
我们还在加载g 向量, 这个向量的位置
another thing t that we are doing is we are loading the g vector, which vector, well > TIMELINE OUTLINE ad. Function l :
474 取决于偏移量.(skew) 而偏移量文基于起始队列(start queue )
based. on, of skew, which is based on the start queue,
> TIMELINE OUTLINE 481
rad. Function ):
474
VT_ptrs= V+offs_kv INoe,:1*strid e
Di=tl. load (o+offsq ) 这进一步取决于
which. is. based on the exact starting point in curr _kv =
> TIMELINE OUTLINE ad. Function l :
473
472
474
k T_ptrs = K+offs_kv[
VT_ptrs= V+offs_kv 程序开始处理的具体起点.
Di=tl. load( D+off
which. this particular program should be working with TIMELINE OUTLINE ad. Function l :
T_ptrs 这是因为该程序是以二维方式运行的
478
because. this particular program works as two dimensions.
TIMELINE OUTLINE arad. Function ):
T_ptrs
/ T_ptrs 第一个维度表示程序应处理哪个批次
Di=tl
curr_kv=a
the first dimension indicate which batch TIMELINE OUTLINE 481
ad. Function ):
473
We access the 474
VT_ptrs= V+of
477
Di=tl. load( D 和哪个注意力头, 而第二个维度
and which head this. program should be working with, and the second dimension,
7 > TIMELINE OUTLINE rad. Function ):
475 即程序索引1号0)则指示在所有的序列长度中
which is the program. index number zero indicates which, among all the sequence length.
TIMELINE OUTLINE rad. Function ):
474
k T_ptrs= K+offs _kv[
v T_ptrs= V+offs
We access the Di=tl. load( D+ 该程序具体要处理哪个查询.
479
which query. this particular program is going to work with.
> TIMELINE OUTLINE 481
rad. Function ):
这一点由索引块
(index block )来指示.
urr_kv=8
urstes = se. ew this vis indicated by the index block.
> TIMELINE OUTLINE 481
rad. Function ):
v T_ptrs= V 在这个情况下, 实际上应该是q.
Di = tl. load(
curr_kv = B
this should be actually gin this case.
OUTLINE 481
num_steps = SEQ_ LEN
> TIMELINE ad. Function ):
我忘记改名字了, 所以让我现在改一下
forgot to change the name, so actually let me change it > TIMELINE OUTLINE 481
oarad. Function ):
所以它是索引1 Q, 因为我们跳过了部分 Q
urr_kv=8
So, it! svindex Q, because we skip some Q > TIMELINE OUTLINE
475
474
v T_ptrs= V+offs_kv IN one,:]
476
Di=tl. load( D+offs_q) 我们跳过了多少 Q?
curr_kv =8
um_steps= SEQ_ LEN// BLOCK_ KV
How many Q we skip?
> TIMELINE OUTLINE 48.
475
474
VT_ptrs = V+of fs_kv[
k T_ptrs= K+offs_kv[ N 根据当前程序的索引
476
Di=tl. load( D +offs_q)
curr_kv = 8
um_steps Based on the index of the current program TIMELINE OUTLINE 481
KT_ptrs
VT_ptrs 乘以前序程序已处理的块数来决定.
multiplied by how many blocks have already been processed by the previous programs.
478 > TIMELINE OUTLINE ad. Function l :
k T_ptrs = K+offs_kv
VT_ptrs = V
Di= tl. load 这将告诉我们在序列长度范围内
curr _kv =
This will tell us inside of the sequence length > TIMELINE OUTLINE 482
ad. Function ):
474
v T_ptrs= V+offs_
k T_ptrs= K+offs_kv[
We access the Di=tl. load ( D+ot 当前程序需要选择哪些查询.
what are. the queries that this one needs to select > TIMELINE OUTLINE ad. Function ):
474 因此, 我们使用起始查询加上块队列的范围来确定.
So that's why we. use the start query plus the range that is block queue.
> TIMELINE OUTLINE rad. Function ):
472
473
474 假设在所有序列长度中, 这个程序的起始查询是100
So imagine the starting query for this program among all the sequence length is 100,
> TIMELINE OUTLINE u to arad. Function ):
474 那么它将加载查询行100、101、102, 依此类推
then this will load the query row100,, 101, 102,, blah, blah,
> TIMELINE OUTLINE rad. Function ):
VT_ptrs= V
T_ptrs
Di= tl. loac 直到100加上块队列减一的位置
curr_kv= B
blahuntil1ooplusblock queue minus one.
> TIMELINE OUTLINE 481
um_steps
74 这就是我们在当前程序中要加载的查询向量的范围
This is the range of. the query vectors that we will load in this program.
> TIMELINE OUTLINE
472
473
474 我们通过使用 Q加上列方向重复的偏移量来加载 Q的块,
weloadthe block of Q by using Q plus the offset repeated on the columns.
> TIMELINE OUTLINE
472
473
174 因此, 我们将其视为列向量, 但在行向量上重复广播
sowe treat it as a columnvector but we, repeat, broadcast it on the rows vector > TIMELINE OUTLINE ad. Function ):
T_ptr 其中每一列将是一个头维度乘以步幅.
where each column will be one head dimension multiplied by the stride 78 > TIMELINE OUTLINE rad. Function ):
474
475
]*stride_din 476 在这种情况下, 我们实际上也可以不乘以步幅
inthis case we actually can also not multiply by the stride > TIMELINE OUTLINE rad. Function ):
472
473
474 因为在维度维度中的步幅, 即批次的最后一个维度的步幅是一
because the stride in. the. dimension dimension, so the last dimension of the batch is one,
> TIMELINE OUTLINE rad. Function ):
474
VT_ptrs
k T_ptrs 因为从一个元素移动到下一个元素
478
Di=tl.
because. to go from one, actually the stride, how it is defined.
TIMELINE OUTLINE rad. Function ):
474
VT_ptrs= V+of fs
k T_ptrs= K+offs_kv[ N
Di=tl. load( D+ 你只需移动一个元素的位置
> TIMELINE OUTLINE 481
ad. Function ):
k T _ptrs = K+0
VT_ptrs= V 因此最后一个维度的步幅总是一
Di= tl. load(
curr_kv=8
because > TIMELINE OUTLINE 481
num_steps= SEQ_ LEN// BLOCK_ KV
ass Triton Attention torch. auto or ad. Function ):
474
VT_ptrs = V
k T_ptrs= K+0
We access the Di= tl. load 因此最后一个维度的步幅总是一
to go one element to. the next element, you should move to move it to by one element 78 > TIMELINE OUTLINE rad. Function ):
474 因此, 我们加载了dq, 即本次迭代中要计算的内容
soweload thedqwhich is. the stuff that we are going to compute in this iteration,
> TIMELINE OUTLINE
472
473
474 然后我们需要加载do, 并且do使用与g相同的偏移量
and then we have a do. that we need to load and the do we use the same offset as q
> TIMELINE OUTLINE rad. Function ):
472
473
474 因为(
do. 和dg具有相同的形状并且以相同的方式工作.
because the do and. the dg have the same shape and they work in the same way.
TIMELINE OUTLINE rad. Function ):
472
473
174 因此, 我们加载了q的一个块, 并加载了对应的 DO块(在本例中)
so we load a block of q and we load the corresponding block of o, of Do in this case,
> TIMELINE OUTLINE rad. Function ):
474
DO的形状与 O相同, 而 O的形状又与 Q相同
and the Do has the same shape as O, which has the same shape as Q > TIMELINE OUTLINE rad. Function ):
472
473
474 此外, 我们还需要加载 M归一化因子, 这些因子位于 M矩阵中
Plus, we need to load also the M normalization factors which are in the M matrix,
> TIMELINE OUTLINE rad. Function ):
4756
474
v T_ptrs= V+oft
477
Di=tl. load ( D 对应于我们将在本程序中处理的
which one the one. corresponding to this particular group of queries > TIMELINE OUTLINE rad. Function ):
474
We access the K and V as transp
k T_ptrs = K+offs_kv[ None,:]* stride _seq
VT_ptrs= V+offs_kv[ None,:]*stride 特定查询组
478
Di= tl. load( D +offs_q)
479
that we are going to work within this particular program.
> TIMELINE OUTLINE rad. Function ):
474
k T_ptrs= K+offs _kv[ N
VT_ptrs = V+offs_kv[ N
We access the K and V as trans Di=tl. load( D +offs_q) 我们从偏移量开始.
478
479
curr_kv=8
hum_steps = SEQ_ LEN // BLOCK_ KV
We start with the offsets.
> TIMELINE OUTLINE
472
473
474 如您所见, 偏移量是从零位置开始的第一个 KV块.
Asyou can see, the offsets are the first block of KV starting from the zero position.
> TIMELINE OUTLINE rad. Function ):
472
473
474 因此, 国 因为我们将会遍历所有的 KV, 月 所以从零位置的 KV开始.
Sobecause we will iterate through all the KVs and we start from the zero KV.
> TIMELINE OUTLINE rad. Function ):
即从第零个键向量和第零个值向量开始.
So the. key vector zero and the V vector zero.
OUTLINE TIMELINE ad. Function ):
472
473
474 然后, 在每次送代中, 我们将按 KV块中的向量数量向前移动.
And then we will move byblock KV number of vectors forward at each iteration.
> TIMELINE OUTLINE
47
Di =tl. load( D+offs_q )
curr_kv =θ
num_steps= SEQ_ LEN// BLOCK_ K 希望我没有讲得太快
Ihope Ididn'tgotoofast
OUTLINE
484
485
@staticmethod
> TIMELINE
def forward(ctx, Q, K, V, causal, soft max _scale ):
Di=tl. load( D+of fs_q )
curr_kv =8
num_steps= SEQ_ LEN// 因为这里写的大部分内容与
481
because most of the things that are written here are very similar to
18
TIMELINE
OUTLINE
ef forward(ctx, Q, K, V, causal, soft max _scale ):
Di = tl. load( D+offs_q)
478 我们在另一个for循环中已经完成的内容非常相似
483
What we have already done in the other for loop,
> TIMELINE OUTLINE 485
484
dffowrd(ctx, Q, K, V, causal, soft max _scal):
ostatic n
Di= tl. load( D +offs_q)
curr_kv=θ
num_steps = SEQ _ LEN // BLOCK_ KV 所以我不想过多重复.
lass Triton Att
so T don't want to repeat myself too much.
OUTLINE
484
485
@staticmetho
> TIMELINE
def forward(ctx, Q, K, V, causal, soft max _scale ):
Di= tl. load( D+offs_q)
curr_kv=θ
um_steps 真正重要的是我们将使用的公式
What did matter is actually the formulas that we will use,
> TIMELINE OUTLINE 485
ef forward(ctx, Q, K, V, causal, soft max _scale ):
Di = tl. load ( D +offs_q )
num_steps = SEQ_
curr_kv =θ 这些公式与论文中的完全一致.
s Triton Attention(
whichisexactlytheonein thepaper.
> TIMELINE
OUTLINE
@staticmethod
def forward(ctx, Q, K, V, causal, soft max _scale ):
Di = tl. load( D +of fs_q )
num_steps = SEQ_ LEN // BLOCK_ K
curr_kv =θ 我们遍历这些 KV 块.
Triton Attention(tor
Wegothroughtheseblocksof Kv.
> TIMELINE
OUTLINE
485
@staticmethod
def forward(ctx, Q, K, V, causal, soft max _scale ):
curr_kv= B 我们加载第一个 K转置块和 V转置块
Weload the first block of K transposed and V transposed > TIMELINE OUTLINE 486
def forward(ctx, Q, K, V, causal, soft max _scale ):
479
478
Di = tl. load( D +offs_q)
curr_kv = θ
num _steps = SEQ _ LEN// BLOCK_ KV for blk_idx in range(num_steps 像往常一样这样加载
class Triton Attention(torc
which isloadedlikethisasusual.
TIMELINE
OUTLINE
487
@staticmethod
def forward(ctx, Q, K, V, causal, soft max _scale ):
curr _kv = θ
num _steps = SEQ _ LEN// BLOCK_ KV 483
_ T_block
_ T_bloc 以及另一个要加载的元素指针
andwhat are the pointers of another element that you want to load
TIMELINE
OUTLINE
ef forward(ctx, Q, K, V, causal, soft max _scale ):
curr_kv = θ
SEQ _ LEN// BLOCK_ KV Triton就会将你请求的块加载到 SRAM中
and it will load the block that you are asking Triton to load inside of the SRAM.
> TIMELINE
OUTLINE
ef forward(ctx, Q, K, V, causal, soft max _scale ):
curr_kv =θ
SEQ _ LEN// BLOCK_ KV 因此 这些数据都存储在 SRAM 中, Q 和
So thisstuffallresidesinthe SRAMandalso Qresidesinthe SRAM
TIMELINE
OUTLINE
forward(ctx, Q, K, V, causal, soft max _scale ):
curr_kv = θ
hum_steps= SEQ _ LEN// BLOCK_ KV V_ T_block
_ T_block
DO也同样存储在 SRAM中
485
class Triton Attention(torcl
andalso DOresidesinthe SRAM
TIMELINE
OUTLINE
@staticmethod
def forward(ctx, Q, K, V, causal, soft max _scale ):
接着, 我们计算查询与键转置的乘积
Then we compute the query multiplied by the transpose of the keys TIMELINE OUTLINE forward (ctx, Q, K, V, causal, soft max _scale ):
for bi k_idxin range (num _steps ):
K. T_block = tl. load(k T_ptrs)
V_ T_block=tl. load(v T_ptrs) 因为需要计算 P块.
Triton Atter
because we need to compute the Pblock > TIMELINE OUTLINE 90 @static method def forward (ctx, Q, K, V, causal, soft max _scale ):
483
KTblock =tl. load ( KT_pts)
484
V_ T_block = tl. load(v T_ptrs)
486
K_block= 因此, qk块就是
8
487
488
ss Triton Attention (torch. auto grad. Function ):
So the query,
TIMELINE OUTLINE 491 @static method def forward (ctx, Q, K, V, causal, soft max _scale ):
KT_block = tl. load( KT_ptrs)
V_ T_block=tl. load(v T_ptrs)
QKblock= sof tmax_scale *tl 当前查询块中的查询
487
488
the qk block is just the query in the current query block TIMELINE OUTLINE 491
def forward(ctx, Q, K, V, causal, soft max _scale ):
KT _block = tl. load( KT_ptrs)
485
484 与当前键块中的k转置相乘的结果
V_ T_block =tl. load(v T_ptrs)
487
488
lass Trito With the k transposed in the current key block.
TIMELINE OUTLINE def forward (ctx, Q, K, V, causal, ftmax_scale :
@static met
KT _block = tl. load ( KT_ptrs)
483
/_ T_block=tl. load(v T _ptrs ) 但由于我们访问键时已经进行了转置, 所以不需要再次转置.
But we access the keys already as a transposed, so we don't need to transpose it.
> TIMELINE OUTLINE def forward (ctx, Q, K, V, causal, soft max _scale ):
KT_block= tl. load( KT_ptrs)
483
484
V_ T_block=tl. load(v T_ptrs)
OK_block=sof tmax_scale *tl. dot( Q_ 即使需要转置,
487
488
And anyway, even if we need to transpose it > TIMELINE OUTLINE 489
498
491
@staticmeth
def forward (ctx, Q, K, V, causal, soft max _scale ):
483
KT_block = tl. load( KT_ptrs)
V_ T_block= tl. load(v T_ptrs)
486
K_block=sof tmax_scale*t 也不需要任何计算
487
488
it doesn't require any computation to transpose matrix,
OUTLINE 489
> TIMELINE
491
def forward (ctx, Q, K, V, causal, soft max _scale ):
KT_block =tl. load( KT_ptrs)
V_ T_block= tl. load(v T_ptrs)
486
485 只需以不同的方式访问它即可.
487
488
we just access it in a different way.
OUTLINE 489
49
@static method > TIMELINE def forward (ctx, Q, K, V, causal, soft max _scale ):
KT _block = tl. load (k T_ptrs)
_ T_block=tl. load(v T_ptrs) 因为在内存布局中它总是以扁平数组的形式存储.
because in the memory layout it's always stored kind of as a flattened array.
OUTLINE TIMELINE def forward (ctx, Q, K, V, causal, soft max _scale ):
" P_block" is not accessed Py lan [variable ) P_block: tensor 接着, 我们计算p ^块, 它是soft max 的输出
Then we compute the p blockwhich is the output of the soft max.
OUTLINE TIMELINE def forward (ctx, Q, K, V, causal, soft max _scale ):
482
KT_block =tl. load(k T_ptrs)
V_ T_block=tl. load(v T_ptrs) 对于每个查询键我们减去该查询块的logsumexp值
so each of the query key we subtract the log sum exp value for this block of queries.
> TIMELINE OUTLINE ef forward (ctx, Q, K, V, causal, soft max _scale ):
482
KT_block= tl. load( KT_ptrs)
483
V_ T_block=tl. load(v T_ptrs) 这就是为什么我们在加载m块时使用正在加载的查询的偏移量
that's why for loading the m block we use the offsets of the queries that we are loading.
> TIMELINE OUTLINE def forward (ctx, Q, K, V, causal, soft max _scale ):
KT _block = tl. load ( K T_ptrs)
/_ T_block=tl. load(v T_ptrs) 如您所知m块巴经包含了归一化因子
and, as you remember, the m block already includes also the normalization factor,
> TIMELINE OUTLINE def forward (ctx, Q, K, V, causal, soft max _scale ):
48
KT_block = tl. load( KT_ptrs)
V_ T_block=tl. load(v T_ptrs)
_block
K_block 因为每个m实际上是每行的最大值
because each m is actually the maximum value for each row > TIMELINE OUTLINE ef forward (ctx, Q, K, V, causal, soft max _scale ):
483
KT_block = tl. load( KT_ptrs)
V_ T_block=tl. load(v T_ptrs)
QK_block= sof tmax_scal
_block = tl. math. 加上归一化因子的对数
488
plus the logarithm of the normalization factor that > TIMELINE OUTLINE def forward (ctx, Q, K, V, causal, soft max _scale ):
KT _block = tl. load ( KT_ptrs )
V_ T_block=tl. load(v T_ptrs) 当您应用指数属性时, 它会进入分母.
when you apply with the properties of the exponential, it goes into the denominator.
18 > TIMELINE OUTLINE def forward (ctx, Q, K, V, causal, soft max _scale ):
KT_block =tl. load(k T_ptrs)
V_ T_block=tl. load(v T_ptrs) 好的, 然后我们再次应用自回归掩码
487
OUTLINE okay,
and then we apply again the autoregressive masking.
TIMELINE def forward (ctx, Q, K, V, causal, soft max _scale ):
P_block =tl. math. exp( QK_bloc
K_ T_block) 哎呀, 我做了什么?
Oops, what did i do?
> TIMELINE OUTLINE @static method def forward (ctx, Q, K, V, causal, soft max _scale ):
P_block=tl. math. exp( QK_ 让我回到这里的代码.
ittention(torc
Let me go back to the code here.
> TIMELINE OUTLINE @static method def forward (ctx, Q, K, V, causal, soft max _scale ):
QK _block= sof tmax_scale *tl. dot( Q_block, K_ T_block)
P_block=tl. math. exp( QK_ 所以我们有这个阶段.
8
ittention (torch So we have the stage, this one.
> TIMELINE OUTLINE @static method def forward (ctx, Q, K, V, causal, soft max _scale ):
QK _block= sof tmax P_block=tl
*tl. dot( Q_block, K_ T_block) 因此, 当我们启动反向传播时
8
Trito n Atter So when we launched the backward pass,
> TIMELINE OUTLINE @static method def forward (ctx, Q, K, V, causal, soft max _scale ):
if STAGE =3: 阶段三表示在前向传播中我们计算了因果注意力
stage three indicates that in the forward pass we computed the causal attention TIMELINE OUTLINE ef forward (ctx, Q, K, V, causal, soft max _scale ):
if TAGE==3: 如果我们在前向传播中计算了因果注意力
Soif we computed the causal attention in the forward pass,
TIMELINE OUTLINE def forward (ctx, Q, K, V, causal, soft max _scale ):
if AGE==3: 那么在反向传播中也需要屏蔽这些元素
we
also need to mask out these elements in the backward pass.
> TIMELINE OUTLINE forward (ctx, Q, K, V, causal, soft max _scale ):
if STAGE==3:
otgs_kv =
mask_block 因此, 我们创建了一个掩码
P_block
Sowecreatethemask
> TIMELINE
OUTLINE
@staticmethod
def forward(ctx, Q, K, V, causal, soft max _scale ):
if STAGE==3:
offs.
ask 这个掩码仅在查询索引大于键索引
which tells us which index this mask is true only for the elements for TIMELINE OUTLINE forward (ctx, Q, K, V, causal, soft max _scale ):
if STAGE ==3:
# Autoregress
49
mask block =of gs_q[:, None ]
_block 的元素时为真.
which the query index is more than the key index.
> TIMELINE OUTLINE def forward (ctx, Q, K, V, causal, soft max _scale ):
if STAGE==3: 如果条件为真则不进行掩码处理, 否则进行掩码处理.
And if this is true, then we don't mask, otherwise we mask.
> TIMELINE OUTLINE def forward (ctx, Q, K, V, causal, soft max _scale ):
ask block =of fs_q l:,
=offs_kv [ N 接下来我们计算下一个操作, 即计算dp和ds.
and let's compute the next operation, which is to compute dp and ds.
> TIMELINE
OUTLINE
ef forward(ctx, Q, K, V, causal, soft max _scale ):
+tl. arange (e, BLo CK _ K V)
9
ask block=offs_q l:,
offs_kv [ 实际上, 我们直接计算dk,: 然后像之前一样进行解释,
actually, ilet'scomputedirectlydkand thenweexplainitlikebefore.
> TIMELINE
OUTLINE
forward(ctx, Q, K, V, causal, soft max _scale ):
an ge(e, BLOCK _ KV)
ask block=of fs_q l:,
offs_kv [ 我们从末尾开始, 逐步回溯到计算所需的部分,
sowestartfromtheendandwegotowherethisstuff, whatisneededtocomputeit.
> TIMELINE
OUTLINE
fforward(ctx, Q, K, V, causal, soft max _scale ):
d S _block =ds_block. to(tl. float16) 如果你查看公式让我确认一下这个.
so if you look at the formula, let me check This one.
> TIMELINE OUTLINE def forward (ctx, Q, K, V, causal, soft max _scale ):
ds _block = P_block *(d P _block-Dil :, None l)
d S_block =ds_block. to(tl. float16)
NOTE : We need to de-scale dq
# Compute d Q. 我认为我们不需要.
501
d Q_block+= sof tmax_scale
502
Triton Attention (torch. auto grad. Fun We don't need, I think.
> TIMELINE OUTLINE 585
def forward(ctx, Q, K, V, causal, soft max _scale ):
@static method
ds _block = P_block*(d P _block-Di[:, None])
d S_block = ds_block. to(tl. float16)
d Q_block
# Compute 好的, 我们转到i Pad 上来继续.
class Triton At tent Okay, let's go here to the i Pad.
OUTLINE @static method def forward (ctx, Q, K, V, ca
> TIMELINE = Q. shape[-1], K. shape[-1]
24:
Write d K;← d K;, d V;←d V; to HBM.
Okay, let's go here to the i Pad.
We see that similar to the forward pass. the backward
24:
Write d K; ← d K, d V;← d V; to HBM.
25:end for
26: Return d Q, d K, d V.
24:
Write d K;← d K;, d V;← d V; to HBM.
Okay, what we are trying to compute here is dq.
We see that similar to the forward pass. the backward pass performs O( N2)
d S _block =d S _block. to (tl. float16)
d Q_block+ 好的, 我们这里要计算的是dq.
sta
Okay, what we are trying to compute here is dq.
> TIMELINE OUTLINE 05
efforward (c tx
HEADDIM_ Q,
Q. shape[-1], K. shape[-1]
24:
Write d K;←d K, d V;←d V; to HBM.
25:end for
26: Return d Q, d K, d V.
24:
Write d K; ← d K;, d V;← d V; to HBM.
25: 论文所示. dq等于原有的dq 加上 tau
26:
So dq, as you can see in the paper, is dq is equal to the old dq plus tau,
24:
Write d K;← d K;, d V;←d V; to HBM.
25:end for
26: Return d Q, d K, d V.
24:
Write d K; ← d K;, d V;← d V; to HBM.
softmax 缩放因子, 也就是这里的这个值)
end for which is the soft max scale,
We see that similar to the forward pass. the backward pass performs O( N2)
d S _block = P_block *(d P _block-Di[:, None])
ds_block =ds _block. to(tl. float 16)
93
497 即
softmax 缩放因子, 也就是这里的这个值)
502
503
@static method which is the soft max scale,
> TIMELINE OUTLINE 584
05
def forward(ctx, Q, K, V, caus a
HEAD_ DIM _ Q,
d S _block = P_block*(d P _block-Di[:, None l)
ds_block =ds_block. to(tl. float16)
d Q_block+= sof tmax _scale *tl. do t(ds_block,
# Compu te d Q. 再乘以ds
estatuwhich is this stuff here multiplied by the matrix > TIMELINE OUTLINE
24:
Write d K; ← d K, d V; ← d V; to HBM.
25:end for 再乘以ds
26: Return d Q, d K, d V.
which is this stuff here multiplied by the matrix performs O( N2)
24:
Write d K;←d Kj, d V;←d V; to HBM.
25:end for multiplication between the ds and the kb lock.
We see that similar to the forward pass. the backward erforms O( N2
24:
Write d K;←d K;, d V←d V; to HBM.
25:end for
26: Return d Q, d K, d V.
24:
Write d K; ← d K;, d V;← d V; to HBM. 这里的ds块就是、ds块, 而k块则是kt块的转置.
nd, for so the ds block is here and the k block is the transpose of the kt block.
We see that similar to the forward pass. the backward pass performs O( N2)
d S _block = P_block *(d P_block-Di[:, None])
9
ds_block=ds_block. to (tl. float16) 这里的ds块就是ds块, 而k块则是kt块的转置.
so theds block is here and the k block is the transpose of the kt block.
TIMELINE OUTLINE
d S _block d S _block. to (tl. float16) 因为我们已经以转置块的形式访问了k.
501
503
502
because we are accessing k already as a transpose block > TIMELINE OUTLINE 584
HEADDIM_ Q, H
HEAD_ DIH_ K= Q. shape [-1], K. shape[-1]
ds _block = P_block*(d P _block-Di[:, None])
ds_block =dsblock. to(tl. float16)
d Q_block+= sof tmax _scale *tl. dot(ds_block, tl. trans(kg_block))
# Compute d Q.
class Triton Attention (torch. auto grad. Function ):
> TIMELINE OUTLINE @static method def forward (ctx, Q, K, V, causal, soft max _scale ):
HEAD _p IM _ Q, HEAD_ DIM_ K= Q. shape[-1], K. shape[1]
ds _block = P_block *(d P _block-Di[:, None])
ds_block =ds_block. to(tl. float16) 我们也可以通过反转来直接访问未转置的k块
we could also access a k directly as not transposed block by inverting.
TIMELINE OUTLINE = Q. shape[-1], K. shape[-1]
497
ds_block =ds_block. to (tl. float16) 如果你不想以转置块的形式访问它, 就像这样操作, 像这里一样, 不转置
if you don't want to access it as a transpose block, just do like this, like here, none.
> TIMELINE OUTLINE HEAD _ DIM_ Q, H
= Q. shape[-1], K. shape[-1]
d S _block =d S_block. to(tl. float16) 这样会将其视为行向量, 并在列方向上广播
this will treat it as a row vector and the broadcast along the columns otherwise.
> TIMELINE OUTLINE HEAD _ DIM _ Q, H
HEAD_ DIM_ K= Q. shape[-1], K. shape[-1]
ds _block =ds _block. to (tl. float16)
d S_block = P_block*(d P_block-Di[:, None ])
d Q _block+= sof tmax _sca 因此, 这里也需要调整
class Triton Attention (torch. a OUTLINE @static method so this one you should need to change,
> TIMELINE HEAD _ DIM _ Q, HEAD_ DIM_ K= Q. shape[-1], K. shape[1]
ds _block = P_block *(d P _block-Di[:, None])
d S_block =d S_block. to(tl. float16)
d Q_block 因为需要将其视为列向量来处理.
because this one you need to treat it as a column vector.
> TIMELINE OUTLINE 584
505
ds _block = P_block *(d P _block-Di[:, None])
ds_block =ds_block. to(tl. float16)
d Q_block+= sof tmax_scale *tl. dot(ds_block, tl.
class Triton Attention (torch. auto grad. Function ):
OUTLINE @static method def forward (ctx, Q, K, V, causal, soft max _scale )
the dimensions.
> TIMELINE HEAD _ DIM _ Q, HEAD_ DIH_ K= Q. shape[-1], K. shape[-1]
d S _block = P_block*(d P _block-Di[:, None])
d S_block = ds_block. to(tl. float16)
497 但如果你想以 K转置的形式访问, 只需反转这两个操作即可.
But if you want to access it as a K transpose, then you just invert these two operations.
> TIMELINE OUTLINE HEAD _ DIM _ Q, HEAD_ DIM_ K = Q. shape[-1], K. shape[-1]
ds _block = P_block *(d P _block-Di[:, None])
d S_block =d S_block. to(tl. float16) 希望我没有搞错什么, 我们继续往下推进吧.
hopeldidn't mess up anything, so let'smove forward.
> TIMELINE OUTLINE 505
= Q. shape[-1], K. shape[-1]
497
ds_block =d S_block. to (tl. float16) 好的,"我们知道 DQ的计算公式与论文中完全一致
So, okay, we know that the formula for the DQ is exactly the same as the paper one,
> TIMELINE OUTLINE HEAD DIM _ Q, H
HEAD_ DIM_ K= Q. shape[-1], K. shape[-1]
ds _block =ds _block. to (tl. float16)
d S_block = P_block *(d P_block-Di[:, None ])
d Q _block+= sof tmax _sca # Compute d Q. 但这个 DS块是什么呢?
class Triton Attention (torch. a but what is this Ds block?
OUTLINE @static method def forward (ctx, Q, K, V, causal,
> TIMELINE HEAD _ DIM _ Q, HEAD_ DIH_ K= Q. shape[-1], K. shape[1]
ds _block =ds _block. to(tl. float 16)
d Q _block+= softmax_scale *t1
# Compute d Q. 让我们来看看论文.
class Triton Attention (torch. auto grad. F Let's look at the paper.
OUTLINE @static method def forward (ctx, Q, K, V, causal, soft max _scale )
> TIMELINE HEAD _ DIM _ Q, HEAD_ DIM_ K= Q. shape[-1], K. shape[1]
ds _block =
ds_block. to(tl. float16) 这个 DS块正是源自这里的这些内容.
et This Ds block is coming from this stuff here.
> TIMELINE OUTLINE
24:
Write d K;← d K;, d V;d V; to HBM.
enc
DS. 块正是源自这里的这些内容.
This Ds block is coming from this stuff here.
We see that similar to the forward pass. the backward pass
24:
Write d K; ←d K, d V;←d V; to HBM.
25:end for
26: Return d Q, d K, d V.
24:
Write d K;←d K;, d V;←d V; to HBM.
so this, i believe, this stuff here, ds, which is a pi, the p block element,
pass performs O( N2)
ds _block =d S_block. to(tl. float16) 因此, 我认为这里的 DS块, 是一个π(pi)元素
sothis, i believe, this stuff here, ds, which is api, the pblock element,
> TIMELINE OUTLINE HEAD DIM _ Q, HEAD _ DIM _ K
24:
Write d K;← d K;, d V;←d V; to HBM.
so this, i believe, this stuff here, ds, which is a pi, the p block element,
pass performs O( N2)
24:
Writ d K;←d Kj, d V;←d V; to HBM.
wise multiplication with the dpi minus di, which is dpi minus di.
9 通过逐元素乘法与dπ减去di结合, 即dπ减去di.
wisemultiplication with the dpi minus di, which is dpi minus di > TIMELINE OUTLINE HEAD _ DIM _ Q, HEAD_ DIM_ K = Q. shape[-1], K. shape[-1]
d S _block = P_blok*(d P_block-Di(:, None])
d S _block = ds_block. to (tl. float16)
d Q_block += softmax 那么这个p块是什么呢?
class Triton Attention (to ro now, what is the this p block?
OUTLINE @static method def forward (ctx, Q, K, V, causal, sof TIMELINE HEAD _ DIM _ Q, HEAD_ DIH_ K= Q. shape[-1], K. shape[1]
d S _block = P_block *(d P _block-Di[:, None])
ds_block =ds _block. to (tl. float16)
d P_block=
497
p块实际上就是softmax 的输出, 而这个输出我们已经得到了
the pblock is exactly the output of the soft max, which we already have.
> TIMELINE OUTLINE HEAD _ DIM_ K= Q. shape[-1], K. shape[-1]
ds _block =d S _block. to (tl. float16)
d S_block = P_block*(d P_block-Di[:, None ])
d Q _block+= sof tmax _scale *tl. dot
Compute d Q.
dp块是什么呢?
8
583
class Triton Attention (torch. auto grad. Fu what is the dp block?
OUTLINE @static method def forward (ctx, Q, K, V, causal, soft max _scale ):
> TIMELINE 505
HEAD_ DIM_ Q, HEAD_ DIM_ K= Q. shape [1], K. shape[1]
497
d S_block= P_block*(d P_lock-Di(:, None ))
ds _block =ds_block. to(ti. float16]
dp块实际上就是d O乘以 V的转置, 其中d O我们已经加载好了
well, the dp block is exactly do multiplied by V transposed, which is do,
> TIMELINE OUTLINE HEAD _ DIM _ Q, HEAD_ DIM_ K = Q. shape[-1], K. shape[-1]
24:
Write d K;←d K, d V←d V; to HBM.
well, the dp block is exactly do multiplied by V transposed, which is do,
We see that similar to the forward pass. the backward pass performs O( N2)
dp 块实际上就是d O乘以 V的转置, 其中d O我们已经加载好了
well, the dp. block is exactly do multiplied by V transposed, which is do > TIMELINE OUTLINE HEAD _ DIM _ Q, HEAD_ DIM_ K= Q. shape[-1], K. shape[-1]
d S _block = P_block(d P _block-Di[:, None])
d S_block =ds_block. to(tl. float16)
d Q_block+= sof tmax _scale *tl. dot (
# Compute d Q. 而 V也是以转置
which we already loaded, and it's here, multiplied by the transpose of the V > TIMELINE OUTLINE HEAD _ DIM _ Q, H
ds _block = P_block*(d P _block-Di[:, None])
d S_block =ds_block. to(tl. float16)
d Q_block+= sof tmax _scale *tl. do t(ds_b
# Compute d Q. 形式加载的
class Triton Attention (torch. a OUTLINE @static method which we already load as transposed > TIMELINE
ds _block =ds _block. to (tl. float16)
ds_block = P_block*(d P_block-Di[:, None ])
d Q _block += soft 这就是我们计算d Q 的方式
class Triton Attention (to r OUTLINE @static method And this is how we computed Q > TIMELINE HEAD _ I M_ Q, HEAD_ DIM_ K= Q. shape[-1], K. shape[1]
24:
Write d K;←d K, d V;←d V; to HBM.
25:end for
26: Return d Q, d K, d V.
ds _block =ds_block. to(tl. float16)
d Q_block += sof tmax _scale *tl. dot(ds_block, tl. trans( K_ T_block))
# Compute d Q.
8
583
class Triton Attention (torch. auto grad. Function ):
> TIMELINE OUTLINE 505 @static method def forward (ctx, Q, K, V, causal, soft max _scale ):
d S _block = P_block*(d P _block-Di[:, None])
d S_block =ds_block. to(tl. float16) 接下来,"我们当然需要移动到下一个键值块(key Vs)
Then, of course, we need to move to the next block of key Vs,
> TIMELINE OUTLINE def forward (ctx, Q, K, V, causal, soft max _scale ):
d S _block =ds_block. to(tl. float16)
d Q_block+= 因此我们像之前一样递增指针.
so we increment the pointers just like before.
> TIMELINE OUTLINE @static me def forward (ctx, Q, K, V, causal, soft max _scale ):
97
ds_block =ds_block. to (tl. float16) 于是, 我们移动到下个键值块(keys和values)
503
02
So we move to the next block of keys and values.
> TIMELINE OUTLINE 504
def forward(ctx, Q, K, V, causal, soft max _scale ):
a static r
d Q _block+= sof tmax _scale *tl. dot(ds_block, tl. trans( K_ T_block)) 同时, 我们也像之前一样移除指针.
and also remove the pointers, just like before.
> TIMELINE OUTLINE @static me def forward (ctx, Q, K, V, causal, soft max _scale ):
curr_kv +=
/ T_ptrs 接着, 我们需要存储 DQ的 的结果
Andthenweneedtostoretheresultof DQ
TIMELINE
OUTLINE
def forward(ctx, Q, K, V, causal, soft max _scale ):
k T_ptrs
/ T_ptrs += 通过像下面这样分割for循环
and this way we only need to do one write to the HBM > TIMELINE OUTLINE def forward (ctx, Q, K, V, causal, soft max _scale ):
BLOCK _ KV *stride_seq 我们只需对 HBM进行一次写入操作.
by dividing the for loop like the following.
> TIMELINE
OUTLINE
@staticmethor
ef forward(ctx, Q, K, V, causal, soft max _scale ):
VT _ptrs += BLOCK _ KV * stride _seq
do_block_ptrs=d Q+offs_q:, None]*stride_seq+offs_dim None,:*stride_dim
t1gstore (do_block_ptrs, do_block]
508
class Triton Attention (torch. auto grad. Function ):
OUTLINE 509
510
@static method > TIMELINE 511
def forward(ctx, Q, K, V, causal, soft max _scale ):
VT_ptrs += BLOCK _ KV * stride_seq
d Q_block_ptrs=d Q+offs_ql:, 如果你看原始算法
509
508
Triton Attenti
So if you look at the original algorithm,
> TIMELINE
OUTLINE
510
511
@staticmethod
def forward(ctx, Q, K, V, causal, soft max _scale ):
24:
Write d K;←d K, d V;←d V; to HBM.
25:end for So if you look at the original algorithm,
We see that similar to the forward pass. the backward pass performs O( N2)
26: Return d Q, d K, d V
We see that similar to the forward pass, the backward pass performs O( N2) FLOPs and only 1
O( N)extra memory beyond inputs, output, output gradient, and input gradients.
We analyze the IO-complexity of the backward pass, similar to the forward pass ( Theorem 2 ).
23:
endfor
24:
Writed K←d K, d Vd Vto H BM
25:end for
26: Return d Q, d K, d V.
23:
endfor
24:
Write d K
d Kd V
d Vo HBM
25:end for 我不确定它是否真的对应
I don't know if the original algorithm actually corresponds to the implementation 26:
Return d Q. d K. dv
Wesee that similar to the forward pass. the backward pass performs O( N2) FLOPsand only
26: Return d Q, d K, d V. 他们在 CUDA 中的实现
performs O( N2) FLOPsand only requires We analyze the IO-complexity of the backward pass. similar to the forward pass ( Theorem 2 ).
22:
23:
end for
24:
25:endf
26:
but'I'don't think so because it would not be so optimized.
Return d Q. d K. d V
We see that similar to the forward pass, the backward pass performs O( N2) FLOPs and only requires
25:end for
24:
Write d K;d Kd V;d V;to HBM
26: Rqturn d Q. d K. d V. 但在原始算法的论文中
We see th
O( N) extra memory beyond inputs, output, output gradient, and input gradients.
24:
Writed Kd K, d V;d V;to HBM
25:
26:
they say that you need. to. go through all. the. keys. and. then, while. going. through the keys,
O( N) extra memory beyond inputs, output, output gradient, and input gradients.
25:end for
24:
Write d K;d Kd V;d V;to HBM.
26: Rqturn d Q. d K. d V. 还需要访问所有队列
you need. to. go. to. all the queues and for. each queue that you visit O( N) extra memory beyond inputs, output, output gradient, and input gradients.
24:
Write d K;
d K, d V;d V;to HBM.
25:end for
O( N) extra memory beyond inputs, output, output gradient, and input gradients.
25:end for
24:
Write d K;d Kd V;d V; to HBM. 这种方式并不优化.
26: Rqturn d Q. d K. d V
( N) FLOPsand on lyrequires
O( N) extra memory beyond inputs, output, output gradient, and input gradients.
24:
Write d K;
d Kd V;d V;to HBMI
25:end for
O( N) extra memory beyond inputs, output, output gradient, and input gradients.
因为每介键值的更新仅依赖于特定的队列块
24:
Write d K;d K, d V;d V;to HBMI.
because each. key is. upda ted depends on ly on. a particular block. of queue,
O( N) extra memory beyond inputs, output, output gradient, and input gradients.
25: end for
24:
Write d K;
d K, d V;d V;to HBMI
26: Rqturn d Q. d K. d V 抱歉
pass performs O( N2) FLOPs and on ly requires
O( N) extra memory beyond inputs. output, output gradient, and input gradients.
24:
Write d K;d K, d V;d V; to HBMI.
And then, we. fix. the gu eries and. we,. iterate through. all. the. keys O( N) extra memory beyond inputs, output, output gradient, and input gradients.
24:
Write d Kd K, d V;d V;to HBMI.
2end 因为 的一个块依赖于 Ks 的所有块
because. on e. block. of. Q. depends. on all the blocks. of Ks..
O( N) extra memory beyond inputs, output, output gradient, and input gradients.
v T_ptrs += BLOCK _ KV *stride_seq
tlgstore(do_block_ptrs, do_block]
d Q_block_ptrs=d Q+offs_q[:, None]*stride_seq+offs_dim[ None,:]*stride_dim
class Triton Attention(torch. autograd. Function):
> TIMELINE
OUTLINE
def forward(ctx, Q, K, V, causal, soft max _scale ):
@static method
503
(mof Tuatrs += BLOCK_ KV * stride_seq 因此我们进行了分割, 这就是我们写的第二个循环.
Andthisiswhywesplitandthisisthesecondloopthatwehavewritten.
> TIMELINE
OUTLINE
def forward(ctx, Q, K, V, causal, soft max _scale ):
507
508
class Triton Attention (torch. auto grad. Function ): 现在我们已经写好了实现 Flash Attention 所需的所有内容
Now we have written everything that we need to flash attention.
> TIMELINE OUTLINE
class Triton Attention (torch. auto grad. Function ):
511
def forward(ctx, Q, K,
@static method 包括前向传播和反向传播,
HEAD _ DIM _ Q,
HEAD _ DIM _
> TIMELINE OUTLINE BATCH _ SIZE,
static m 所以, 我们应该准备好启动内核了.
So we should be ready to launch the kernel > TIMELINE OUTLINE BATCH _ SIZE, NUM
V= V
scale =ctx. soft max _scale 希望我在复制代码时没有出错.
hope I didn't make any mistake in copying the code.
> TIMELINE OUTLINE 641
stride_head= Q. stride(1),
ide_batch = Q. stride (o )
V= V
_scale =ctx. soft max _scale 所以我不打算尝试启动它
V=d V
So I don't think I will try to launch it > TIMELINE OUTLINE 641
= Q. stride(1),
stride_he
633
softmax_scale =ctx. softmax_scale
V= V, 如果有任何错误, 我会直接使用我已经写好的参考代码
638
and if there is any error I wil I just use my reference code,
TIMELINE OUTLINE stride _head = Q. stride (1),
stride_batch= Q. stride(θ),
V= V
soft max _scale =ctx. soft max _scale 这些代码我之前已经用作复制
which I have already written that I used as a copy.
> TIMELINE OUTLINE 641
stride _head= Q. stride(1),
stride_batch= Q. stride(0)
633
V= V,
soft ma
ax_scale=ctx. softmax _scale 到目前为止, 我的参考代码和我们刚刚写的代码之间唯一的区别
The oh ily difference up to now between my reference code > TIMELINE OUTLINE 648
stride_head= Q. stride(1),
batch = Q. stride (θ)
V= V
scale =ctx. soft max _scale 是自动调优, 这一点我还没有解释.
and the one that we have written is the auto tuning, which I didn't explain.
637 > TIMELINE OUTLINE stride _head = Q. stride (1),
stride_batch= Q. stride(o )
V= V
scale =ctx. soft max _scale 那么, 我们来谈谈自动调优吧.
So let's talk about the auto tuning.
> TIMELINE OUTLINE 641
ead= Q. stride(1),
stride_he
V= V
scale =ctx. soft max _scale, 自动调优功能在原始论文中就已经存在
So the autotuning is also something that was already present in the original paper 637 > TIMELINE OUTLINE stride _head = Q. stride (1),
tride_batch= Q. stride(o )
V= V
x_scale=ctx. soft max _scale, 我直接保留了它, 没有做任何改动.
and I kept it as is.
> TIMELINE OUTLINE 641
stride _batch= Q. stride(0),
stride_head = Q. stride(1),
37a
373
Autoregr 我移除了反向传播的自动调优功能, 但在前向传播中, 如果你仔细看
Iremoved the autotuning for the backward pass, But in the forward pass, if you check > TIMELINE OUTLINE
109
110
stride _ Q_seq,
stride_ O_dim, 这里有一段代码, 它指示了用于 Triton的自动调优配置.
111
there is this code here that indicates the auto-tuning configuration for Triton. 福
> TIMELINE OUTLINE stride _v _dim,
110
stride_ O_dim, 因此 Triton 基本上无法预先知道最佳的块大小
11
8
So Triton basically can not know beforehand what is the best block size,
> TIMELINE OUTLINE stride _v _dim,
110
stride _ O_dim,
stride_ K_batch
112
stride_ K_head, 也无法确定查询
113
114
stride_ K_seq,
stride_ K_dim,
8
stridehor what is the best block size for the query,
> TIMELINE OUTLINE 118
stride_ V_seq
stride_v_dim,
110
stride_ O_dim,
stride_ K_batch stride _ K _seq,
stride _ K _head, 键和值的最佳块大小
stride _ K_din,
or what is the best block size for the key and values,
> TIMELINE OUTLINE 118
stride_ V_dim,
stride _0_dim,
113
stride _ K
stride_ K 或者我们其他维度中的最佳块大小.
8
114
stride_ K
or what is the best block size for another dimension that we have.
> TIMELINE OUTLINE stride _v _dim,
stride _o_dim,
stride _ K _batcl
113
stride_ K_head,
stride_ K_seq, 我们需要根据运行的硬件条件
114
stride_ K_din,
We need to try, based on the hardware that we are running on,
> TIMELINE OUTLINE stride _ V _dim,
stride _ Q_seq,
stride_ Q_dim,
stride_ K_batch stride _ K _head,
stride _ K_seq,
SRAM 的可用性
8
114
stride_ K_dim
OUTLINE stride _v _head,
stride _ V _seq,
based on the availability on the SRAM,
TIMELINE 118
stride_ V_dim,
stride _ Q_dim, 以及 Triton能够应用的线程粗化策略来进行尝试.
11
8
116
based on the thread coarsening that Triton can apply.
TIMELINE OUTLINE 117
118
stride_ V_dim,
stride _ Q_seq,
stride_ Q_dim, 另外, 我还没提到线程粗化这个概念.
8
So I didn't talk also about thread coarsening OUTLINE stride _v > TIMELINE 118
stride_v_dim,
110
stride_ Q_dim, 简单来说, 在 CUDA中, 你可以选择每个线程执行一个原子操作
Basically, in cu DA, you can choose if each thread does one atomic operation.
> TIMELINE OUTLINE stride _v _dim,
110
stride _ Q_dim,
stride_ K_batch 112
stride_ K_head, 例如, 在矩阵加法中
113
114
stride_ K_seq,
stride_ K_dim,
8
stride_ V_batch
For example, in a matrix addition,
OUTLINE 117
stride_ V_head,
> TIMELINE 118
stride_v_dim,
109
110
stride_ O_seq,
stride_ Q_dim, 每个线程要么负责输出矩阵中一个特定元素的加法运算
each thread is doing one addition of one particular element of the output matrix,
> TIMELINE OUTLINE stride _ V_din
stride _ O _dim,
stride _ K _batch stride _ K_seq,
stride_ K_head, 要么管理多个元素的计算.
8
stride_ V_batch stride _ K _dim OUTLINE stride _v _head,
stride _ V _seq,
or it's managing multiple elements > TIMELINE 118
stride_v_dim,
stride _ Q _dim,
stride _ K_heat
stride_ K_seq, 这就是所谓的线程粗化, 我认为.
stride_ K_din stride _v _
stride vh This is called thread coarsening and I think > TIMELINE OUTLINE 118
stride_ V_dim,
110
stride _ O_dim,
stride _ K_batch
113
stride_ K_seq,
stride_ K_head, 我没有查阅文档
8
stride_ V_batch
ididn'tcheck the documentation,
OUTLINE stride _v _head,
stride _ V _seq,
> TIMELINE 118
stride_v_dim,
stride _ O_dim,
114
113 但我相信 Triton会根据你指定的块大小
8
but i believe triton does it for you based on the block size that you give it > TIMELINE OUTLINE stride _v _dim,
stride _ Q_seq,
stride _ Q_dim,
trid 和所需的warp数量自动处理线程粗化.
stride_ V _head,
t ride _
and the number of warps that you want.
> TIMELINE OUTLINE 118
stride_v_dim,
stride_ V_seq
110
stride_ Q_dim,
stride_ K_batch
113
stride_ K_seq,
stride_ K_head,
war p数量指的是什么?
114
stride_v_batch,
stride_ K_dim,
OUTLINE stride _ V _head,
stride _ V _seq,
the number of warps is what is?
> TIMELINE 118
stride_v_dim,
110
stride _ Q_dim,
stride_ K_batch 112
stride_ K_hea d, 一组线程块?
113
114
stride_ K_seq,
stride_ K_dim,
stride_ V_batch
a block of threads?
OUTLINE stride _v _head,
stride _ V _seq,
> TIMELINE 118
stride_v_dim,
110
109
stride _ Q_seq,
stride_ Q_dim, 组由32个线程组成的协作单元, 它们同时运行相同的指令,
11
32 threads that work cooperatively, running the same instruction, always at the same time.
> TIMELINE OUTLINE stride _ V_din
110
stride_ Q_dim,
stride_ K_batch
113
stride_ K_seq,
stride_ K_head, 阶段数量则更为有趣,
8
114
stride_ K_dim,
stride_ V_bat
OUTLINE stride _v _head stride _ V _seq,
the number of stages is more interesting > TIMELINE 118
stride_v_dim,
stride _0_dim,
stride_ K_batch
stride_ K_head,
stride _ K_seq, 这是 Triton进行的一项优化.
8
stride _ K_dim
stride_ V _batch it's an optimization that triton does.
OUTLINE stride _v _head,
stride _ V _seq,
> TIMELINE 118
stride_ V_dim,
110
stride_ Q_dim,
stride_ K_batch
113
stride_ K_head,
stride_ K_seq, 本质上, 这并不是循环展开.
8
114
stride_ V_batch
stride_ K_dim,
OUTLINE stride _v _head,
stride _ V _seq,
basically, it is not loop unrolling.
> TIMELINE 118
stride_ V_dim,
110
stride_ Q_dim, 那么, 我们实际上来讨论一下一一, 来讨论一下软件流水线
so actually let's talk about - uh, let's talk about software pipelining,
> TIMELINE OUTLINE stride _ V _dim,
stride _ Q _dim, 因为这是我们理解这段代码的最后一部分
because this is. the last part that we need to understand from this code,
114 > TIMELINE OUTLINE stride _v _dim,
110
stride _ Q_dim,
stride _ K_batch
113
stride_ K_seq,
stride_ K_head, 也就是自动调优.
8
115
114
stride_ K_din,
stride _v_batch,
116
stride_v_head,
which is the auto tuning.
> TIMELINE OUTLINE 118
stride_ V_seq,
stride_v_dim,
stride _ Q _dim,
stride,
stride 所以我认为这里最有趣的部分并不是
stride stride > TIMELINE OUTLINE 118
stride_v_dim,
110
stride _ Q_dim,
stride _ K_batch
113
stride _ K_seq,
stride_ K_head, 选择 Q 和 K 的块大小
8
stride_ K_dim,
stride,
OUTLINE stride _
choosing the block size Q and the block size K > TIMELINE 118
stride_ V_dim,
110
stride_ Q_dim,
stride_ K_batch
113
stride_ K_seq,
stride_ K_head, 因为这其实只是基本操作.
8
114
stride_ V_batch,
stride_ K_dim,
OUTLINE stride _v _head,
stride _ V _seq,
because that is just kind of.
> TIMELINE 118
stride_ V_dim,
110
stride_ Q_dim, 你只需要根据运行时间,
111 尝试各种配置, 找出效果最好的那个.
you try whatever configuration works best, based on the timing.
> TIMELINE OUTLINE stride _ V _dim,
stride _ O_dim,
riton实际上会为你运行所有这些配置.
tride K bat
Triton will actually run all these configurations for you.
> TIMELINE OUTLINE 118
stride_ V_dim,
stride _ Q_dim,
stride _ K stride_ K 每当序列长度或头维度发生变化时
114
stride_k
every. time the sequence length or the head dimension changes,
> TIMELINE OUTLINE stride _ V _dim,
stride _ Q_dim,
tride
trid 对于每一对头维度和序列长度的组合
114
115
and for every pair of head dimension and sequence length.
> TIMELINE OUTLINE 118
stride_ V_din
stride _ O_dim,
tride_
tride 它都会选择运行时间最短的最佳配置
113
114
it will choose the best configuration that runs in the least amount of time.
> TIMELINE OUTLINE stride _v _dim,
stride _ Q _dim,
stride _
stride _ 这实际上能为你带来最佳的吞吐量.
stride stride OUTLINE stride _ V_h
That gives you the best throughput actually > TIMELINE 118
stride_ V_dim,
110
stride_ Q_dim, 那么, 我们来看看这个numstages, 它是什么以及它是如何工作的
So let'slook at this num stages, what is it and how it works.
> TIMELINE OUTLINE 118
stride_v_dim,
110
stride _ K_batch
stride_ O_dim,
113
stride_ K_seq,
stride_ K_head, 那么, 我们开始吧
8
114
stride_ K_dim,
stride_ V_batch,
OUTLINE stride _v _head,
stride _ V _seq,
So let's do it.
> TIMELINE 118
stride_v_dim,
motcce tat et
wimg
Hhe
WR
RD
we Diety 好的, 软件流水线化是在你有一个类似for 循环的情况下使用的.
TO
RORO
mot cce tat et
nertolly
wimg
Hhe
WR
RD
, 你有一个顺序操作,
WR
o 在这个操作中每次送代都不依赖于前一次迭代.
which each iteration does not depen c
nd on the previous iteration.
RO
net olly 因此 你在一次送代中执行的操作与
RO
nto 你在前一次选代中所做的操作是独立的
what you have done in the [
previous iteration,
WR RO
emot cet 这或多或少与我们之前在for 循环中所做的工作类似.
nertcolly
wimy
Hhe
WR
RO
RD
nor toll y "实际上 我相信在某些情况下,
WR
RO
wim g 这一点并非必须成立.
WR
RORO
也就是说. 即使操作之间存在依赖关系, 你仍然可以进行软件流水线化,
nertcolly
wimg
Hhe
WR
RD
no toll 比如想象你肴一个媚下的for 循环:
RD RD
nertcolly
wimg
Hhe
WR
WR
TO
RD
Dietu 从1到的循环, 首先你加载一些数据,
for loop that rose from one to n, and d first you load some data,
R O
RD
nertcolly
wimg 然后加载另一些数据,
WR
RO
ot接着进行矩阵乘法, 最后存储一些数据.
RO RO
nertcolly
wimg
Hhe
WR
RD
RD
WR
emotce at t
Dietu
ot. 这里你在读取数据, 这里你也在读取数据,
RORO
not. 这里你在进行计算, 而这里你在写入数据.
RD RD
mot cce Hat et
nertcolly
wimg
Hhe
WR
RD
RD
如果我们观察每次迭代中发生的事情, 会看到以下情况.
RORO
Comale untt
ltecat'om 2
Iteration
Imagine our Gp urs'imade up of a compute unit
Comale cunt
ltecatcon 2
Iteration
and a unit that r is dedicated to loading stuff,
这个单元负责从内存中读取数据或向内存写入数据.
so reading from the memory or writing to the memory.
Comale unt
ltecat'on 2
Itecation
Comule 从时间轴上看, 在第一次送代中2
Co mule un tt 首先我们会 读取一些数据, 而此时计算单元处于空闲状态, 因 因为它需要等待这些数据.
first we are reading some data and the compute unit is idle because we need this data.
Comyule unt
ltecato 2
Itecation
Co male unt 接着我们会读取更多数据, 计算单元依然空闲, 因为它需要等待这些数据
Then we are reading some more data and the compute u unit is idle because we need this data.
Comale unit
ltecaton 2
Iteration
Coom yale un tt 最后, 当我们有了足够的数据时, 计算单元就可以执行操作了,
Then finally we have enough data and then we can compute this operation
Com yule un tt 而读取单元此时则处于空闲状态.
and the reading unit is idle.
Comale untt
ltecatc'on 2
Iteration
Co male unt 然后我们会将一些数据写回内存, 而计算单元再淡进入空闲状态.
And then we are writing some data back to the memor
I and the compute unit is again idle.
Coop u le untt And then it will be idle for another two time steps
ltecaton 2
Itecation!!
Com male cuntt 直到它获得足够的数据才能继续执行计算.
until it has enough data to run the computation.
ltecatcon 2
Iteration U!!
Coo ale unit 正如你所见, 这种方式效率并不高, 因为在往荷时刻
So as you can see, this is not very efficient be
cause at any point in time,
there is only one unit working and the other is sitting idle.
因此, 优化这个循环的一种方法是采用软件流水线技术.
So one way to optimize this for loop is to do software pipelining.
你可以通过 告诉 Triton 你希望有多少个流水线阶段, 让它为你的循环实现这一优化
And
接下来, 我们来看看它是如何工作的.
So let'ssee how it works.
Prologue Aelunt, werlinp
Epiloque
WR 因此, 对循环进行流水线化意味着首先
Pr. lo S o to pipeline a for loop means that, fl first of all lo yue
需要将所有操作转换为异步
wim g
WR
Memory Fo
RD RD
WR And in CUDA, a at least in the Gpu of NVi DIA,
RD
存在异步内存加载和异步内存写入操作.
WR there are the async loading from the
nertiolly
wimg
Hhe
WR
Memory FO
RO
RD
WR Me more y
nertiolly
wimg 它是否已完成.
WR
wimg
nertiolly
WR
RD
Memory Fo RO
WR
wimg
nertiolly
C
WR
RD
Memory FO
RD
nertolly
wimg
Hne 并继续执行下一条指令
WR
and
Hhe
nertiolly
wimg
C
WR
Memory FO
RD RD
o 在这里, 我会启动一个加载迭代, 它会立即返回
Mom or cy
wimg
nertiolly
WR
RD
Memory FO RO
nertolly
wimg 并继续执行下一条指令.
WR
and
nartiolly
wimg
WR
RD
Memory FO
RD
nertiolly
wimg 然后我可以进行计算
WR
Memory FO
我会先检查这两个操作是否已完成
but before computing I just ch
WR
Momorcy
RD
因此, 我可以立即启动两个读取操作, 然后只需检查它们是否已完成.
WR
wimg
Hhe
WR
RD
Memory FO
通过软件流水线技术
So with the software pipe lin
20 H
R0 A 我们将不同送代的接 操作整合到一个选代中
lining operations MA
MM
MM
MM
M
MM
Prologue Epilogue of different iterations.
into a single iteration.
on uch more
Prologue A el unit wen limp Epilogue "hoed "
much more
So first, basically what we will do. is we will read the first matrix
that we need for computin s this. matrix multiplication..
Prologue A el unit won limp Epilogue any mc operat on "hoed "
much more
Prologue 接下来, 在第二次迭代中
Epilogue Then after this next iteration,
much more
Prologue ARe l unt werlin p Epilogue any mc opera tom "hoed "
much more
P. lg我们读取第二次选代的第个矩阵pil we read the we read the first
Prologue A el unit wen limp Epilogue "hoed "
much more
and also read the second matrix, of the first iteration.
Prologue A el unit won limp Epilogue "hoed "
much more
因此, 我将其称为读取 A和读取· B, 4 A表宗读取我们需要的第一个矩阵
So I call itread A and read B,
muh
Prologue A el unit wen limp Epilogue "hoed "
much more
that we need and B means read
Prologue A el unit wor limp Epilogue "hoed "
much more
Prologue 所有这些操作都是异步的
Epilogue All these operations are asynchronous.
much more
Prologue A el unit won limp Epilogue any mc One ratio w s
"hoed"
much more
然后, 我在第三次送代中启动另个鼻步操作内容是:
Prologue A el unit wen limp Epilogue "hoed "
much more
P. 读取第空次迭代的第不矩阵,
Epilogue read the first matrix of the. third iteration much more
Prologue A el unit won lin p Epilogue "hoed "
much more
and then read the second
Prologue A el un two r limp Epilogue "hold "
much more
Prologue 随后进行矩阵乘法计算
Epilogue and then compute ths
因为在第三次迭代时, 这两个操作应该已经完成了
Prologue A el unit won limp Epilogue "hoed "
much more
但在计算矩阵乘法的同时, 我不会让加载单元闲置
but while computing the matrix multi pl
because they are still computing. this, this and this load.
much in ore
Prologue A el unit won limp Epilogue "hoed "
much more
这只有在能够启动异步操作的情况下才能实现.
This can only work if you can spawn async operations.
mauk more
Prologue A el unit won limp Epilogue "hoed "
much more
Prologue 因此在第三次送代中
Epilogue So at the third iteration, I can compute 1
is matrix multiplication by using this one
and this one because I should t have finished.
Prologue Epilogue "hold "
much more
P."然而, 在计算矩阵乘法的同时,
Epilo pue But while I'm computing he matrix multiplication,
m one
Prologue 我已经启动了一些异步操作
Epilogue I already spawned some async operations much more
Prologue A el un two r limp Epilogue "hoed "
much more
因此, 在第四次迭代中, A我将启动第西次迭代的数据加载操作
so at the fourth iteration i will spawn t
ng of the data for the fourth iteration,
muh
Prologue A el unit won limp Epilogue anymc onerakom
"hoed"
much more
Prologue 同时加载第三次迭代的数据
Epilogue loading the data for the third iteration,
much more
Epil op ue because they should have alr ready. completed, by now.
muk more
Prologue A elwntworlimp
Epilogue
"hoed"
much more
P ·在编程语言或 CUDA 语言中有一些原语"
P. 可以用来检查操作是否完成.
Epilogue to check if the operation has complete d.
omuch more
因此:实际上在进行乘法运算之前, 我们会先检查
so actually, before doing the mul
Prologue A el unit won lin p Epilogue "hold "
much more
Prologue 异步操作是否已经完成
Epilogue if the async operation has. finished.
much more
so it's not like we just expect. it to be finished.
Prologue A el unit won linp
Epilogue
aymconeradows
"hoed"
much more
从时间的角度来看这就像在 Jv a Script 中一样.
with respect to time,
his. is. like in Java Script.
的东西pi logue you have these things called the. prom. promise,
Prologue A el unit wen limp Epilogue "hoed "
much more
你可以在真正需要它们之前等待 Promise 完成.
and you can wait for the promise to be ished. before you, actual
But you can spawn as many promise'as you, want.
much more
Prologue A el unit won limp Epilogue "hoed "
much more
P在 C#中, 我想它们被称为任务
( Tasks )lo yue in C sharp, I think,
ey. are called'the tasks.
on uch more
Prologue A el unit wen limp Epilogue "hoed "
much more
因此你可以生成任意数量的住务 然后在需要时
Prologue A el unit won lin p Epilogue "hoed "
much more
Prologue 只需等待你所需的那一个
Epilogue Then you just wait for the m,
muk more
P. 而其他任务仍在后台算步运行.
Epilogue while the other are still running i in the background Asynchronously.
Prologue ARl unit won limp Epilogue "hoed "
much more
Epil op ue this is the whole idea of.
software pip
这就是软件流水线的核心理念.
much more Note :
正如你所见, 软件流水线
M one vet,
akow
menary
Honwe
her a kom much more me nary Hon we
Note :+
any mc one radon s much more HRe wrologue
memary
Hon weconconoume dutianp
N. 我们可能拥有足够的数据来执行前两次送代
menary
Not e:t
any mc onerakom
much more
HRe wrologue me mary Hon wecon conoume dutiaup
No l:e以及第三次送代的一半数据"
eradko w
omemary
N 因此我们需要增
SRAM 的内存需求
more me nary
Note :+
muh more HRewrologue
Hon wecon conoume dutia up
好的, Triton 会为你处这种软件流水线操作
era kom Okay, and Triton wil memory
Not e:+
M它会将所有加载存储操作"
menary
more Hi on we into'async operations me nary
N.:并为你完成这种流水线处理.
you memory
Note :
much more Re mcologue me mary Hon wecon conoume dutiaup
Note :+
o M com nee memory
e rakow much more me mary
人因为我们在模型训练中它经采用了类似的做法.
me mary Hon we
这种方法被称为流水线并行.
It is called pipeline parallelism.
流水线并行的工作方式如下.
So in pipeline parallelism, it works as follows.
我们有一个非常大的神经网络, 无法完全放入单个 GPU 中.
We have a very big neural network that does not fit in a single GPu.
假设这个神经网络由三层组成, 分别是第一层、
So imagine this neural network is made up of three layers, layer one,
第二层和第三层.
layer two and layer three.
但由于模型规模过大, 无法完全容纳在一个 GPU 中.
But this is so big and it does not fit entirely in one single GPU.
于是, 一种解决方案是将每一层分别放入一个 GPU 中.
So one way would be to put this each layer into one GPU.
例如, 我们将第一层放入 GPU1.
So we put, for example, layer one into Gpu one.
第二层放入 GPU2, 第三层则放入 GPU3.
a layer two into GPU two, layer three into GPU number three.
假设我们为这个神经网络提供了一个输入.
So imagine we have an input for this neural network.
于是, 我们将其送入第一个 GPU.
So we put it to the first GPU.
GPU1会处理第一层
The GPu one will process the layer one
并生成一些输出, 这些输出随后会被传输到 GPU2.
and generate some output which will be transferred to the GPu two.
GPU2将计算其自身的输出并传输到 GPU3,
which will calculate its own output and transfer it to the GPu3,
GPU3 再计算其自身的输出
which will compute its own output,
最终我们就能得到神经网络的最终输出
and finally we will have the output of the neural network.
问题在于, 当你将 GPU1的输出发送到 GPU 2,
The problem is when you send the output of the GPU1 to the GPU2
以便 GPU2执行其自身任务时
for the GPU2 to do its own thing,
GPU1此刻便空闲了.
the GPu1 now is free.
这样会造成资源浪费.
So it is a waste of resources.
我们应始终让 GPU 保持忙碌状态
We always should keep the Gp Us busy.
因此, 我们可以采取的一种方法是, 不将所有数据一次性发送到 GPU So one thing that we can do is instead of sending all the mega batch to the Gp U1,
而是分批发送多个较小的数据块.
we send many smaller batches.
它是如何工作的呢?
How does it work?
想象一下, 我们将批次0, 也就是第0批数据, 发送到 GPU1.
Imagine that we send the batch number O, so batch O, to the GPu 1.
GPU1将计算其输出并发送到 GPU2.
The GPU 1 will compute its output and send it to the GPU 2.
此时, GPU2正在处理第0批数据.
So now the GPu 2 is computing the batch number 0.
此时, 第0批数据已不在此处.
So now the batch o is not here anymore.
但此时 GPU1已空闲, 因此我们发送另一个称为批次1的微批次数据
But now the Gp U 1 is free, so we send another micro-batch called batch 1.
接着, BGPU2将完成对批次0的处理, 并将其发送至 GPU3.
Then BGPU2 will finish processing the batch zero and will send it to the GPU number three.
此时, GPU3已接收批次0的数据, 而 GPU2则处于空闲状态.
So now the Gp U three has the batch number zero and the Gp U two now is free.
于是我们进行了数据传输, 同时希望 GPU1也已完成了任务.
So we transferred and hopefully also Gpu one has finished.
因此, 我们将批次1的数据从 GPU1传输至 GPU2,
So we transferred the batch number one from GPU one to GPu two,
随后 GPU1将恢复空闲状态.
and then the BGPU one will be free.
于是我们进行了数据传输,
So we transfer.
此时, GPU 1变为空闲状态, 而 GPU2开始处理新的任务.
Here becomes one and now this one is free.
由于 GPU1现已空闲, 我们可以开始处理另一个批次的数据.
So because it's Gpu one is free, we can introduce another batch.
接下来是批次2的数据, 以此类推, 不断循环这一过程
So batch number two, et cetera, et cetera, et cetera.
因此, 每当我们将一个批次的数据从一个 GPU 转移到另一个 GPU 时, 我们都会在流水线的起始端引 I 入一个新的批次
So we always introduce, while moving one batch from one GPu to the other,
并在每次送代中使各个批次
we introduce a new batch at the beginning of the pipeline
向前移动一个位置.
and they shift by one position at every iteration.
这种操作方式确保了 GPU 始终处于忙碌状态.
This will keep the GPUs always busy.
流水线并行技术存在一个问题, 即所谓的"气泡效应"
there is only one problem of the pipeline parallelism, which is the this bubbling effect,
BATCH. 这是因为在流水线初始阶段, 需要一定的时间来填满流水线.
because to create this pipeline, you, at the beginning of this - um, okay,
A< CH2
BATCH. 此外, 流水线并行还面临反向传播步骤的挑战.
actually in the pipeline parallelism, you also have the problem of the backward step.
BACCH2
BATCH. 在反向传播过程中
so the backward step has to run exactly in reverse,
BA< CH2
BATCH. 必须严格按照接收微批次的顺序逆向执行.
in the order in which you receive the micro batches, while in triton,
而在 Triton 中实现软件流水线时, 多 会遇到前导和尾声的问题.
when doing software pipelining, you have the problem of the prologue and the epilogue,
Rim lon
这是因为需要先建立流水线才能开始流水线处理
because you need to create this pipe omm
C
omm
并且在流水线结束时
and at the end of the'pipeline you need to use all the stuff
必须处理完流水线中所有的数据.
A that is currently'in'the pipeline.
0m
00
因此, 只有在for 循环的初始阶段
W So only in the beginhing step
0m
00.
和最后阶段
and in the last'step of
fthis for loop omm
C
om
GPU 的所有计算单元可能不会同时工作.
all the units of this Gpu may not'b'e working simultaneously.
C
om
这意味着什么?
W ewhich, what does it mean?
这意味着, 为了有效利用流水线技术
需要确保for 循环的迭代次数远大于
将迭代过程划分的阶段数
bigger than the number of stages in which your iteration is divided into.
omm
在这个例子中, 我们有四个阶段, 这些阶段被称为流水线阶段.
In this case we have'four stages these are called stage s.
Rimilo n
omm
Comale untt
Itecatcon2
Iteration (!!
Com gul e cnt 因此, 我们希望迭代次数要远失手
So you want the number of iterations to be much more,
Comyule umt
Itecatcon2
Iteration
Co male unt to be much larger than the number of stages.
Comal e untt
Iteration
Itecatcon 2
Coo mule un tt 好了各位, 我终于完成了这个视频.
Alright guys, finally I have completed the video.
Commale untt
Itecatcon 2
Iteration
Co male unt
I hope that you learned a lot from this video.
Cooma le untt 我相信我们可以运行 Triton 代码, 那么让我们实际运行一下.
I believe that we can run the Triton code, so let's run it actually.
110
stride_ O_dim, 我相信我们可以运行 Triton 代码, 那么让我们实际运行一下.
believe that we can run the Triton code, so let'srun it actually.
> TIMELINE OUTLINE stride _v _dim,
stride _ O _dim,
stride _ K _batch stride _ K _seq,
stride _ K_head, 我已经把代码都复制好了.
114
stride_ K_dim,
stride_ V_batch OUTLINE stride _ V _head,
stride _ V _seq,
I copied everything.
> TIMELINE 118
stride_ V_dim,
of tse to atch _head _sea 我相信我们也放了代码来测试, 但还没放主方法
I believe we also put the code to test it, but we didn't put the main method,
TIMELINE OUTLINE
现在可以复制过来.
which we can copy right now.
> TIMELINE OUTLINE
真的希望没有错误.
I really hope there is no error.
> TIMELINE OUTLINE
真心希望如此.
I really hope.
> TIMELINE OUTLINE
TIMELINE > OUTLINE o F ile Save d
o Fille Saved 16 mins 让我确认一下是否在正确的机器上.
o File Saved
17 mins
Let me check if i am in the right machine O File Saved
18 mins
19 mins
Undo/ Redo
TIMELINE > OUTLINE 是的, 没问题.
o File Save d
O File Saved 16 mins
o Fille Saved
17 mins
O File Saved 18 mins
19 mins
Iam.
Undo/ Redo
> OUTLINE TIMELINE o File Save d 直接运行程序吧.
o Fille Saved O File Saved
17 mins
16 mins
O File Saved
18 mins
19 mins
Let's just run program.
Undo / Redo
TIMELINE > OUTLINE 祈祷吧.
O Fille Save d
O File Saved
16 min s
o File Saved
17 mins
Pray.
O File Saved
18 mins
19 mins
Undo/ Redo
TIMELINE OUTLINE O Fille Save d
O File Save d 如果有错误, 我就直接复制自己的参考实现
o File Saved If there is an error, I wil I just copy my own reference implementation,
O File Saved
TIMELINE > OUTLINE O File Save d
O File Saved 16 min 但我希望它能正常运行, 否则可能是我漏掉了什么
O File Saved
17 mins
but I hope it works because otherwise I forgot something.
O File Saved
18 mins
19 mins
Undo/ Redo
TIMELINE OUTLINE o Fille Save d
O File Saved 我正在 H100上运行我的代码, 因为公司有 H100.
o File Saved
So I'mrunning mycodeonan H1oo because my company has H1o0
O File Saved
TIMELINE OUTLINE o File Saved 如果你用的是较小的 GPU, 可以尝试缩短序列长度,
O File Save d
OFile Saved File Saved If you have a smaller GPU, what you can do is you can reduce the sequence length
TIMELINE > OUTLINE o File Save d
O File Saved 16 mins 你可以减小批次大小.
o File Saved
17 mins
You can reduce the batch size.
o F ile Saved
18 mins
Undo/ Redo
TIMELINE OUTLINE o File Save d
O File Saved 我觉得它已经是一了.
8
o File Saved
17 mins
Ithink it's already one.
OF ile Saved
18 mins
Undo/ Redo
TIMELINE > OUTLINE O File Saved 当我们调用它时, 哎呀, 最佳大小没了
O File Saved
17 mins
16 mins
O File Saved
O File Saved
18 mins
when we call it, oh no, the best size.
Undo/ Redo
19 mins
TIMELINE OUTLINE o File Saved O File Saved 你可以减小批次大小、头数或序列长度.
8
o File Saved
you can reduce the best size, the number of heads, these g uence length OF ile Saved
TIMELINE OUTLINE o File Saved O File Saved 你甚至可以将头维度设为8, 序列长度设为16.
O File Saved
you can even put head dimension equal to eight and the sequence length equal to 16.
DFile Saved
TIMELINE > OUTLINE 我们来看看.
O Fille Save d
o File Saved 16 mins
File Saved
17 mins
Let's check.
O File Saved
18 mins
19 mins
Undo / Redo
TIMELINE OUTLINE O Fille Save d
o File Save d 运行反向传播, 尝试执行反向传播, 返回的梯度数量不正确
O File Saved Run backward, try to run backward, returned an incorrect number of gradient O File Saved Undo / Redo
TIMELINE OUTLINE O File Saved 预期五个, 实际得到一个
o File Saved
17 mins
16 mins
o File Saved
O File Saved
18 mins
expected five gotone.
19 mins
TIMELINE OUTLINE o File Saved 我们可能漏掉了一些返回语句, 我觉得是这样,
o File Saved
17 mins
o File Saved
OF ile Saved
18 mins
Weprobably forgot some return statement, i believe.
Undo/ Redo
19 mins
TIMELINE OUTLINE o File Save d
o File Saved 没错, 我在这里忘记写返回语句了.
o File Saved
17 mins
Yes, so Iforgot the return statement here.
O File Saved
18 mins
19 mins
Undo/ Redo
TIMELINE OUTLINE o File Save d
o File Save d 因此, 在运行完最后一个循环后
o File Saved
o File Saved
17 min s
So in the backward pass after running the last for loop,
Undo / Redo
TIMELINE OUTLINE o Fille Saved 我们需要返回计算得到的结果.
o File Saved
17 mins
O File Saved
OF ile Saved
18 mins
we need to return the stuff that we have computed Undo/ Redo
19 mins
TIMELINE OUTLINE o File Save d
o File Saved
1 min 再次祈祷好运.
o File Saved
17 mins
Cross finger again.
O File Saved 19 mins
File Saved
TIMELINE OUTLINE Tri
o File Save d
O File Saved 好了, 通过了, 所以由 To rch计算的反向传播
o File Saved
17 min
Okay, passed,
so the backward pass that is computed by Torch O File Saved File Saved
TIMELINE OUTLINE Lon _engin
o File Saved (learn-triton )(base )(wo RKER ) umargip-10-1-4-40:~/projects /flash-attention-from-first-principles s
o File Save d
o File Saved 17 mins 1 min
O File Saved 19 mins
File Saved
-triton /lib/python3. 12/site-pac
. py:768:
TIMELINE OUTLINE o File Save d 与我们的反向补丁在10的负二次方的绝对误差范围内
8
o File Saved O File Saved
O File Saved
it is equivalent to our backward patch up to 10 to the power of minus two error,
File Saved
TIMELINE OUTLINE int execu Lon _engin
o File Saved (learn-triton )(base )(wo RKER )umargip-101-4-4:~/projects /flash-attention-from-first-principles s
o File Save d
o File Saved 17 mins 1min
O File Saved 19 mins
File Saved
TIMELINE OUTLINE PASS Eur h
Lon _engi 是等价的
O Fille Saved (learn-triton )(base )(w ORKER )umargip-18-1-4-40:~/project o File Save d
o File Saved 17 mins 1min
O File Saved
19 mins
absolute error.
File Saved
TIMELINE OUTLINE Lon _engin
o File Saved (lem-triton (base )w RKERar@ip11:/prjects/flah-attention fr first-rinciples s
o File Save d
o File Saved 17 mins 1 min
O File Saved 19 mins
File Saved
TIMELINE OUTLINE PASS EUr h
O Fille Saved (le arm-triton )(base )( WORKER )umar 8ip-10-1-4 所以, 正如你所见
o File Saved
17 mins
1 min
o File Saved
O File Saved
19 mins
Sowhen you, as you can see,
File Saved
TIMELINE OUTLINE Lon _engin o Fle Save d
o File Save d
o File Saved 17 mins 1 min
O File Saved 19 mins
File Saved
n3. 12/site-p
TIMELINE OUTLINE PAS CE D
O Fille Saved
o File Saved 我们在这里运行的反向传播与之前的不同
8
o File Saved
this backward that we run here is different than the backward that we run here O File Saved File Saved
triton / Lib/python3. 12/site-packages /torch /auto grad /graph. py:768: Useriarning: Atte
TIMELINE OUTLINE PASSE Un ngine to run the backward pass
o Fille Saved (learn-triton )(base 因为当你应用triton attention 时
o File Saved
17 mins
1 min
o File Saved
O File Saved
19 mins
because When you apply triton attention,
File Saved
3. 12/site-
. py:768:
TIMELINE OUTLINE o File Saved o File Saved 会在我们张量的计算图中引人一个新的计算图
8
o File Saved
it will introduce a new computation graph in the computation graph of our tenso rs
O File Saved
TIMELINE OUTLINE Lon _engin
o File Saved (learn-triton )(base )(wo RKER )uar@ip-181-4:/projects /flash-attention-from-first-principles s
o File Save d
o File Saved 17 mins 1min
O File Saved 19 mins
File Saved
TIMELINE OUTLINE PAC SEO Tri
o File Saved
o File Saved
1 min 其中会包含这个triton attention 操作符.
o File Saved O File Saved
17 mins
19 mins
that will include this triton attention operator.
File Saved
TIMELINE OUTLINE on _engi n
o File Saved (learn-triton )(base )(wo RKER )umaraip-101-4-40:/projects /flash =attention-from-first-principles s
o File Save d
o Fille Saved 17 mins 1min
o File Saved 19 mins
File Saved
TIMELINE OUTLINE PASS Eur
o File Saved (learn-triton )(ba s 而当 Py To rch想要计算反向传播时
o File Save d
17 mins
1min
o File Saved
o File Saved
19 mins
Andwhen Py Torchwant to compute the backward pass,
File Saved
TIMELINE OUTLINE o File Saved o File Saved 它会直接调用这个trito nattention的反向函数来进行计算.
o File Saved
itwilljustcallthebackwardfunctionofthistrito nattentiontocomputeit.
o File Saved File Saved
TIMELINE OUTLINE Lon _engin
o File Save d
o File Save d
o File Saved 17 mins 1min
O File Saved 19 mins
File Saved
TIMELINE OUTLINE o Fille Save d
o File Saved min 它会为所有作为这个triton attention 输入的张量
O File Saved O File Saved
19 mins
17 mins
And it will populate the grad value of all the tensors File Saved
TIMELINE OUTLINE (learn-triton )(base )(wo RKER ) unaraip-101-4-:~/projects /flash a tent in-from-first-principles s
Lon _engin
o File Save d
o File Saved Fille Saved 17 mins 1min
o File Saved 19 mins
File Saved
TIMELINE OUTLINE PASS EUr n
ion_engine O File Saved (learn-triton )(base )( WORKER )unar@ip-18-1-440: 填充grad值
o File Saved Fille Saved 17 mins 1 min
o File Saved
19 mins
that are the input to this triton attention.
File Saved
TIMELINE OUTLINE Lon _engin
o F ile Save d
(learm-triton) (base) (wo RKER) umargip10-1-4-4:~/prjects/flash-atention-froa-first-principles
o File Saved o File Saved
17 mins 1 min
O File Saved 19 mins
File Saved
TIMELINE OUTLINE O Fille Saved (lea 这就是 Py Torch自动求导的工作原理, 各位
o File Saved
17 mins 1 min
o File Saved
O File Saved
19 mins
And this is how Py Torch auto grad works, guys.
File Saved
TIMELINE OUTLINE Lon _engin
o File Saved (learn-triton )(base )(wo RKER ) umaraip-101-4-40:/projects /flash-attention-from-first-principles s
o File Save d
o File Saved 17 mins 1 min
O File Saved 19 mins
File Saved
Atte TIMELINE OUTLINE PASS Eur Calls into O Fille Saved (learn-triton )(base )(w ORKER )unargip- 感谢大家观看我的视频
o File Saved
17 min s
1 min
o File Saved
O File Saved
19 mins
Thankyou for watching my video, guys
File Saved
TIMELINE OUTLINE PASSE un Calls into O Fille Saved (learn-triton )(base ) 这真是非常、非常、非常有挑战性
o File Saved
17 mins
1 min
o File Saved
O File Saved
19 mins
It has been super, super, super demanding File Saved
TIMELINE OUTLINE Lon _engin
o File Saved (learn-triton )(base )(wo RKER )umar@ip-10-440:/projects /flash-attention-from-first-principles s
o File Save d
o Fille Saved 17 mins 1 min
O File Saved 19 mins
File Saved
ck ages /to rch/aut
tograd/graph. py:768:
TIMELINE OUTLINE o File Save d
o File Saved min 我花了好几个月的时间自学 Triton 、
o File Saved
O File Saved
17 min
Spent many months first of all to learn myself about the Triton File Saved
n-triton/lib/python3. 12/site-packages /torch /auto grad /graph. py:768: Userlarning:
TIMELINE OUTLINE PASS Eur
o Fille Saved (learn-triton )(base )( WOR CUDA 、flash attention 等等
o File Saved 17 mins
1 min
o File Saved
O File Saved
19 mins
about CUDA about flash attention,
File Saved
TIMELINE OUTLINE PASS Eur ard ( Calls into o Fille Saved (learn-triton )(base )( Wo RKER ) 而且, 我还有一份全职工作
o File Saved 17 mins
1 min
o File Saved
O File Saved
19 mins
etc Also, Ihave a full-time job File Saved
4. 12/site -
TIMELINE OUTLINE o Fille Save d
o File Save d 所以制作这样的视频真的很难,"我需要投入大量的时间, 比如
o File Saved So it is really hard to make videos like this, like I need to dedicate, You know,
O File Saved File Saved
TIMELINE OUTLINE PASS Eur
o File Saved (learn-triton )(base )(wo RKER )unar@ip-101-4 晚上、早晨和周末
o File Saved File Saved 17 mins 1min
O File Saved
19 mins
the nights, the mornings, the weekends.
File Saved
TIMELINE OUTLINE PASSED a rd ( Calls into O File Saved (learn-triton )(base )( 我花了三天时间才录完这个视频
o File Saved
17 mins
1 min
O File Saved
O File Saved
19 mins
Ispent three days just to record this video File Saved
triton /lib /python 3. 12/site-p
ckages/torch/au
TIMELINE OUTLINE O File Saved 因为有时候我不满意自己的解释方式, 有时候会出错
8
OFile Saved File Saved
because sometimes I don't like how I explain something, sometimes I make mistake,
D File Saved
TIMELINE OUTLINE Calls int
o Fille Save d
o File Save d 或者有时候需要重录, 因为我的做法不对等等
o File Saved sometimes I need to restart because what I'm doing is wrong, etc o File Saved or File Saved
-py:768: User warning :
OUTLINE TIMELINE o File Save d
o File Save d 而且我相信到目前为止我所做的内容应该没有大的错误.
O File Saved and I believe there should be no big errors in what I have done so far.
O File Saved File Saved
TIMELINE OUTLINE Calls int
o File Save d
o File Saved min 但可以肯定的是,*我的符号表示可能不太规范
o File Saved o File Saved
17 mins
19 mins
but for sure, my notation is completely bad,
File Saved
OUTLINE nternall
n3. 12/site-p
TIMELINE (learn-triton )(base )
PAS CE D 因为我所有的数学知识都是自学的
o File Save d
o File Saved min
o File Saved o File Saved
17 mins
19 mins
because all the mathematics I know has been self-taught by.
File Saved
TIMELINE OUTLINE PASS Eur h
ion _engin Call sir
o File Saved (le arm-triton )(base )(w ORKER )una r@ip-18-1-440: 我都是自己学的.
o File Saved
17 min s
1 min
O File Saved
O File Saved
19 mins
I learned it by myself.
File Saved
TIMELINE OUTLINE Lon _engin
o File Saved (le arm-triton )(base )(wo RKER ) umar@ip-10-1-44e:~/prjects/flash-attention-from-first-principles s
o File Save d
o Fille Saved 17 mins 1min
O File Saved 19 mins
File Saved
TIMELINE OUTLINE PASS Eur Calls into O Fille Saved (learn-triton ) 所以, 因为我不是在学术环境中学习的
o File Saved
17 mins
1 min
o File Saved
O File Saved
19 mins
Sobecause I didn't learn it in academia.
File Saved
TIMELINE OUTLINE Lon _engin
o File Saved (learn-triton )(base )(wo RKER ) umar@ip-101-4-40:/projects /flash-attention-from-first-principles s
o File Save d
o Fille Saved 17 mins 1min
o File Saved 19 mins
File Saved
12 /site -
TIMELINE 我有些不好的习惯, 正在努力改正, 所以我用的符号表示可能不太标准
O Fille Saved
O File Saved 8
O Fi le Saved
Ihave bad habits and I'm trying to get rid of them, so I use the very bad notation.
O File Saved File Savec
triton /lib /python 3. 12/site-p
ckages/torch/au
TIMELINE OUTLINE o File Saved o Fi le Saved 有时候我用大写字母表示, 有时候用小写字母
8
OFile Saved sometimes Icallwiththecapitalletter, sometimeswiththelowercase,
O File Saved File Saved
TIMELINE OUTLINE internally a t
O File Saved (learn-triton )(base )(wo RKER ) 有时候甚至忘了加下标等等
o File Saved
17 mins
1 min
O File Saved
O File Saved
19 mins
sometimes Ii us t forget the index, etc File Saved
TIMELINE OUTLINE Lon _engine. run _
o File Saved (learn-triton )(base )(wo RKER )umar@ip-10-1-4-40:~/projects /flash-attention-from-first-principles s
o File Save d
o Fille Saved 17 mins 1 min
o File Saved 19 mins
File Saved
internally at
n-triton/ Lib/python3. 12/site-packages /torch /auto grad /graph. py:768: Userlarning: A
Atte
TIMELINE OUTLINE PASS EUr Calls into t
O File Saved (learn-triton )(base )( WORK 所以我正在努力解决这些问题.
o File Saved
17 mins
1 min
o File Saved
O File Saved
19 mins
So I'mtrying to solve these problems.
File Saved
TIMELINE OUTLINE Lon _engine. run _
o File Saved (learn-triton )(base )(wo RKER )umar@ip-101-440:/projects /flash-attention-from-first-principles s
o File Save d
o Fille Saved 17 mins 1 min
o File Saved 19 mins
File Saved
TIMELINE OUTLINE O Fille Saved
o File Saved 1 min 我觉得我已经解释清楚了,? 所以应该没问题
o File Saved
O File Saved
17 mins
19 mins
Ibelieve I have explained everything, so I should be.
File Saved
TIMELINE OUTLINE Lon _engin
o File Saved now (le arm-triton )(base )(wo RKER ) umaraip-10-1-4-40:/projects /flash-attention-from-first-principles s
o File Save d
o File Saved 17 mins 1 min
O File Saved 19 mins
File Saved
TIMELINE OUTLINE O Fille Save d
o File Saved 你应该已经掌握了推导 Flash Attention 论文中所有公式
o File Saved
17 mins
you should have all the knowledge to derive all the formulas O File Saved 19 mins
File Saved
TIMELINE OUTLINE PASS EUr h
o File S aved
(leam-triton)(base)(w ORx ER)uma@ip-10-1-440:~/proj 所需的知识,
incipless
o File Saved
17 mins
1min
o File Saved
O File Saved
19 mins
that you see in the paper of the flash attention,
File Saved
TIMELINE OUTLINE int execu Lon _engin
o File Saved (leam-triton (base )(wo RKER )umar@ip1-144:/projects /flash-attention-fro first-principles s
o File Save d
O Fille Saved 17 mins 1 min
O File Saved 19 mins
File Saved
OUTLINE TIMELINE PASS Eur
O Fille Saved (learn-triton )(base )(wo RKER )umar@ip-10-1-4-40:~/pro 同时也应该对
o File Save d
O File Save d
O File Saved and you should also have an internal image on how the back, the, the,
File Saved
TIMELINE OUTLINE ion_engin
o File Saved I OW
(lem-triton )(base )(wo RKER )umar@ip-1-144:~/prjects/flash-attention-fr first-principles s
o File Save d
o File Saved 17 mins 1 min
O File Saved 19 mins
File Saved
triton /lib/python3. 12/site-p
TIMELINE OUTLINE O File Save d
o File Save d 注意力计算是如何逐块进行的有了一个清晰的内部理解,
O File Saved O File Saved
19 mins
17 mins
the attention calculation is working block by blocks.
File Saved
TIMELINE OUTLINE ion_engin
o File Saved (learn-triton )(base )(wo RKER )umar@ip-101-4-40:/projects /flash-attention-from-first-principles s
o File Save d
o File Saved 17 mins 1min
O File Saved 19 mins
File Saved
ograd/graph. py:768: User warning OUTLINE TIMELINE o File Save d
o File Saved 我知道我本可以花20个小时把事情讲得更清楚
O File Saved
OFile Saved
19 m know that i could have spent 2o hours explaining things better,
File Saved
12/site-
-py:768:
TIMELINE 我也有自己的生活, 还有妻子要陪伴, 所以我不可能做出100小时的视频
o File Saved OFi le Sa ved
8
Fi le Saved
butialso have a life andialso have awife, so iiicannotmakea 1oohoursvideos.
OFile Saved
TIMELINE OUTLINE Lon _engin
o File Saved (learn-triton )(base )(w ORKER ) umar@ip-101-4-40:~/projects /flash-attention-from-first-principles s
o File Save d
o Fille Saved 17 mins 1min
O File Saved 19mins
File Saved
TIMELINE OUTLINE Calls into o File Saved (learn-triton )( 另外, 制作这些视频时还有一些干扰
o File Saved
17 mins
min
O File Saved
o File Saved
19 mins
Also, there were some interruptions making these videos File Saved
TIMELINE OUTLINE Lon _engin
o File Saved (learn-triton )(base )(wo RKER )umar@ip-101-4-4:/projects /flash-attention-from-first-principles s
o File Save d
o Fille Saved 17 mins 1 min
o File Saved 19 mins
File Saved
TIMELIN E
OUTLIN E
PASSEurn
io n_engi
o File S aved
(learn-triton)(base)(wo RKER)unar@ip-18-1-440 我拔了几颗智齿,
o File Saved
17 mins
1 min
O File Saved
o File Saved
19 mins
Iremoved some wisdom teeth File Saved
to grad/graph. py:768: User warning TIMELINE OUTLINE Calls int
o File Saved o File Saved 恢复至少花了一周多的时间, 因为实在太疼了.
8
O File Saved
so it took me at least more than one week to recover because it was so painful o File Saved File Saved
TIMELINE OUTLINE Lon _engin
o File Saved (learn-triton )(base )(wo RKER ) umar@ip10-1-4-40:~/projects /flash-attention-from-first-principles s
o File Save d
o File Saved 17 mins 1 min
O File Saved 19 mins
File Saved
internally at
n-triton/lib/python3. 12/site-packages /torch /auto grad /graph. py:768: Userarning: Atte
TIMELINE OUTLINE Calls into o Fille Saved (learn-triton )(base )( WORK 所以,"感谢大家观看我的视频.
o File Saved
17 mins
1 min
o File Saved
O File Saved
19 mins
So, thank you guys for watching my video.
File Saved
TIMELINE OUTLINE Lon _engin
o File Saved (learn-triton )(base )(wo RKER ) umangip-101-4-40:/projects /flash-attenti om-from first-principles s
o File Save d
o File Saved 17 mins 1 min
O File Saved 19 mins
File Saved
-triton /lib /python 3. 12/site-pac
eckages/to rch/autograd /graph. py:768: Useriarning:
TIMELINE OUTLINE Calls int
o Fille Saved (learn-triton )(base )( Wo RKER ) 希望这次你们也学到了很多.
o File Saved
17 mins
1 min
o File Saved
O File Saved
19 mins
Ihope you learned a lot also this time.
File Saved
TIMELINE OUTLINE Lon _engin
o File Saved (learn-triton )(base )(wo RKER )umargip10-1-4-40:/projects /flash-attention-from-first-principles s
o File Save d
O File Saved 17 mins 1 min
O File Saved 19 mins
File Saved
OUTLINE PASS Eur n internally at TIMELINE (learn-triton )(base ) 正如你所见, Triton 是一个新事物
o Fille Saved
o File Saved 1 min
o File Saved
O File Saved
17 mins
19 mins
Asyou can see, Triton is something new.
File Saved
ck ages /to rch/autograd/graph. py:768: User warning : Atte TIMELINE OUTLINE PASS Eur
o Fille Saved (learn-triton )(base )(wo RKER )unar@ip 相关的文档资料并不多.
o Fle Saved
17 mins
1 min
O File Saved
O File Saved
19 mins
There is not much documentation.
File Saved
OUTLINE TIMELINE Calls into o Fille Save d
o File Saved 因此, 我所说的关于 Triton 的内容可能并不完全准确
O File Saved So, something that I have said about Triton may not be totally correct
O File Saved File Saved
internally at
triton/lib/python3. 12/site-packages /to rch/autograd/graph. py:768: User warning :
TIMELINE OUTLINE o File Saved (learn-triton )(base )( WORK 因为确实缺乏足够的文档参考
o File Saved
17 mins
1 min
o File Saved
O File Saved
19 mins
because really there is very little documentation.
File Saved
TIMELINE > OUTLINE o F ile Saved (learn-triton )(base )(wo RKER ) unargip-10-1-4-40:~/projects /flash-attention-from-first-principles s
o File Saved Fille Saved 17 min s
1 min
O File Saved 19 mins
OFile Saved
-triton /lib/python3. 12/site-packages /torch /aut TIMELINE OUTLINE Calls int
o File Saved o File Saved 我所掌握的 Triton知识, 都是通过研究他人编写的代码
8
O File Saved
all the Triton that I have learned is by looking at the code written by other s
O File Saved File Saved
TIMELINE OUTLINE PASS EUr h
o File S aved
(learm-triton)(base)(w ORKER)umareip 并试图理解它而获得的
o File Saved
17 mins
1 min
o File Saved
O File Saved
19 mins
and try to understand it.
File Saved
TIMELINE > OUTLINE Lon _engin
o File Saved (learn-triton )(base )(wo RKER )umar@ip-10-1-4-4:/projects /flash-attention-from-first-principles s
o File Save d
o Fille Saved 17 mins 1min
o File Saved 19 mins
File Saved
/fsx/umar/miniconda3/envs/learn-triton / Lib/python3. 12/site-packages /torch /auto grad/graph. py:768: Useriarnin
TIMELINE OUTLINE PASS Eur n
o Fille Saved (le arm-triton )(base )(wo RKER )umar eip 我想就到这里吧, 各位
o File Saved
17 mins
1 min
O File Saved
O File Saved
19 mins
And I think that's it, guys.
File Saved
