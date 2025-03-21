
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

#### 7.1 简单示例

以向量加法为例：两个向量 A 和 B，各包含 8 个元素，

![[Pasted image 20250321174457.png|500]]

CUDA 的工作机制是，当我们要求它并行启动 n 个线程时，它会分配 n 个线程，并为每个线程分配一个唯一的标识符，在这个简单的示例中，我们可以这样理解，第一个线程会被分配索引 0，每个线程处理的数据项正好对应其线程索引号，

* line 14：通过代码，根据线程标识符，指定每个线程处理的数据项，
* line 15：if 语句，指定启动 8 个线程，为什么需要加 if ？在 CUDA 中，启动的线程数总是 32 的倍数，这是 CUDA 的一个基本单位（线程束，Wrap），它们共享一个控制单元，控制单元是 GPU 硬件的一部分，负责确定接下来执行哪条指令，这意味着，这组线程将始终执行相同的指令，也就是说它们会同时到达这里的 if。由于每个线程有自己的寄存器，执行时使用的数据可能各不相同，这种编程模型称为 SIMD（data），或 SIMT（thread）；因此一些通过 if 的线程执行加法，而未通过 if 的也“不得不进入”，因为它们共享同一控制单元，但需要解决这种控制流分叉问题。大致流程为：满足条件的正常执行，不满足条件的线程进入 for 循环，但啥也不干，所有的线程必须保持同步执行相同的指令，这个现象叫控制流分支分化，显然这种空闲状态会降低程序执行效率。因此应尽可能减少这种情况的发生。

> [!NOTE]
> 在 CUDA 中，控制流分支分化（branch divergence）是指同一个线程束（warp）中的不同线程执行不同控制流路径的情况。这会导致性能下降，因为 GPU 必须顺序执行每个分支路径，而不是并行执行。控制流分支分化的影响：
> 1. 线程束执行：当线程束中的线程遇到不同的条件分支（如 `if` 语句）时，GPU 会按顺序执行每个路径，直到所有线程完成。这意味着一些线程会处于空闲状态，等待其他线程完成。
> 2. 性能下降：分支分化会导致线程束内的线程不能完全并行执行，从而降低执行效率。

#### 7.2 复杂示例

简单示例中 8 个元素，扩大到 1 M 个元素，超过了 GPU 线程数，这就是在 cuda 中引入 blocks 概念的原因，接下来假设数据元素是 8 个，GPU 只有 4 个核，
