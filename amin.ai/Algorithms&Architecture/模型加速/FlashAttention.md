
标准的自注意力机制在序列长度 $N$ 增加时表现不佳，因为它需要计算完整的 $N×N$ 注意力矩阵，并进行 $O(N^2·d)$ 量级的运算和内存存储。尤其在长上下文模型中，计算量和内存消耗会急剧膨胀。现有的近似方法（如 Linformer、Performer）往往需要牺牲准确性，或由于 GPU 效率问题而无法在实际应用中实现运行时间的改进。

下图 ([source](https://huggingface.co/docs/text-generation-inference/en/conceptual/flash_attention)) 对比了标准注意力机制与 FlashAttention，展示了 FlashAttention 如何通过将操作融合为更少的内存事务来减少内存读写次数。

![|600](https://aman.ai/primers/ai/assets/flashattention/flash-attn.jpg)



**FlashAttention‑1:**

* 每个注意力头/层使用单个融合的CUDA内核，将QKT计算、掩码处理、softmax归一化、dropout随机失活及输出乘法运算整合为一体，以最大限度减少高带宽内存（HBM）与静态随机存储器（SRAM）之间的数据传输。
* 将Q、K、V分块成 SRAM 大小的块；为数值稳定性重新计算每块的 softmax 归一化。
* 并行性主要体现在批次和头数上；序列长度并发的使用有限。
* I/O 最优设计——在实际 SRAM 容量下，经证明需要 O(N⋅d) 内存流量的下限。

**FlashAttention‑2**：

* 通过在多个线程块上分割头部计算，实现了序列长度维度的并行处理。
* 通过延迟 softmax 缩放操作来减少非 GEMM 浮点运算量——从而消除跨块冗余归一化。
* 使用 CUTLASS 和 CuTe 实现，旨在提高占用率和线程块协调性。
* 增强型 warp 组分区以减少共享内存同步开销。

**FlashAttention‑3**:

* 专为 NVIDIA Hopper（H100）硬件设计，利用 warp专业化和异步调度技术：部分 warp 执行WGMMA GEMM 运算，其他 warp 执行 softmax/scaling 操作，实现计算重叠。
* 以乒乓方式在每个块内跨线程束组进行流水线 GEMM 和 softmax 操作，以最大化利用张量核心和张量内存加速器（TMA）。
* 引入分块式 FP8 量化技术，采用非连贯处理与动态异常值处理机制，以最大限度减少数值误差。
* 利用 Hopper 的 WGMMA 和 TMA 指令，在低精度和标准 FP16/BF16 模式下均能保持高吞吐量。

**性能比较**：

|**Version**|**Target GPU**|**Forward Speedup**|**Peak Throughput**|**Backward Speedup**|**Numerical Accuracy (Low‑prec)**|
|---|---|---|---|---|---|
|FlashAttention‑1|Ampere / A100|~3× over PyTorch on GPT‑2 (seq=1K)|~30–50% utilization|~ similar to baseline|Full FP16/BF16 accuracy; exact attention|
|FlashAttention‑2|Ampere / A100|~2× over v1|~225TFLOPs/s (~72%)|~2–4× over naive backward|Same full precision accuracy|
|FlashAttention‑3|Hopper / H100|~1.5–2× over v2 (FP16)|~740TFLOPs/s (~75% BF16); ~1.2–1.3PFLOPs/s (FP8)|~1.5–1.75× over v2|FP8 RMSE ~2.6× lower than baseline FP8; full precision accuracy preserved.|

# Umar 讲解


### 1、多头注意力机制

MHA 工作原理：$$\begin{align*}
\text{Attention}(Q， K， V) &= \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V \\
\text{MultiHead}(Q， K， V) &= \text{Concat}(\text{head}_1， \ldots， \text{head}_h)W^O \\
\text{head}_i &= \text{Attention}(QW_i^Q， KW_i^K， VW_i^V)
\end{align*}$$
实际上，Flash Attention 主要关注的是 Attention 部分的优化，$Q，K，V$ 的线性层投影以及输出层投影都是常规的矩阵相乘，这一点在 GPU 中已经高度优化。

### 2、缩放点积注意力机制的局限性

#### 2.1 问题一、GPU I/O 限制

一个非常关键的问题是：我们为何需要改进注意力机制的实现方式，查阅 [Flash Attention 1 论文](https://arxiv.org/pdf/2205.14135)，章节 2.2：

![[Pasted image 20250318151046.png|650]]

GPU 主要由两种内存构成，一种是 HBM，即动态随机存取存储器（DRAM），也就是 GPU 的内存，例如 A100 的 40GB 内存，这是 GPU 中容量最大的内存；此外还存在共享内存。

GPU 面临的问题是，访问 HBM（全局内存）与访问共享内存相比，速度极其缓慢，然而，与 HBM 相比，共享内存的容量要小得多，FlashAttention 论文中指出，*注意力机制的操作是 I/O 受限的*，这意味着，如果我们频繁访问全局内存，那么计算注意力机制的整体操作速度慢，这并不是因为计算这些操作本身慢，而是因为频繁访问速度较慢的全局内存导致的，因此我们可以将这类操作称为 I/O 受限型操作。

因此，改善这一状况的唯一方法是，在 GPU 的共享内存中计算注意力机制，尽管共享内存的容量要小得多，但共享内存更靠近实际执行计算的 kernel，因此，我们需要将注意力计算拆分为更小的块，以便这些块能够放入共享内存中，然后在那里计算输出矩阵的一部分，再将这部分复制到位于 HBM 中的输出矩阵中，并针对查询、键、和值矩阵划分的所有块，重复这一过程。

#### 2.2 问题二、softmax

分块计算的最大难题在于 softmax，因为 softmax 需要访问整个 $S$ 矩阵的所有元素才能完成计算，因为需要计算归一化因子，这个因子是对所有元素逐行计算指数后的总和。

### 3、（Safe）Softmax

#### 3.1 softmax 计算上的问题
$$
\mathbf{S} = \mathbf{QK}^\top \in \mathbb{R}^{N \times N}， \quad \mathbf{P} = \text{softmax}(\mathbf{S}) \in \mathbb{R}^{N \times N}， \quad \mathbf{O} = \mathbf{PV} \in \mathbb{R}^{N \times d}，$$
这里的 $QKV$ 都是经过相应线性层转换后的矩阵，$Q，K$ 的维度均为 $N \times d$（$N$ 是序列长度，$d$ 是每个 head 中 token 的嵌入维度，已完成多头切分），点积运算后，其输出矩阵 $S$ 的维度为 $N \times N$，softmax 操作按行处理，并不改变矩阵维度，其结果最后与 $V$ 相乘，输出维度为 $N \times d$。

softmax 操作的作用是将这些点积结果进行转换，使得它们以某种方式变为一种概率分布，*按行计算*，定义如下：$$\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_{j=1}^{N} e^{x_j}}$$分母是向量所有维度指数的总和，这被称为归一化因子，为的是使所有这些数字介于 0 和 1 之间，使用 softmax 是因为我们希望这些数字都是正数（概率值），这是使用指数函数的原因，

但这里*存在一个问题*，想象一下我们的输入向量由许多可能很大的数字组成，比如 100 的指数，会造成计算机结果上溢，即数值不稳定性。

#### 3.2 解决方案

为了使数值上保持稳定，我们需要找到一种解决方案，如下：$$\begin{align}\frac{e^{x_i}}{\sum_{j=1}^{N} e^{x_j}} 
&= \frac{c \cdot e^{x_i}}{c \cdot \sum_{j=1}^{N} e^{x_j}} = \frac{c e^{x_i}}{\sum_{j=1}^{N} c e^{x_j}} = \frac{e^{\log(c)} e^{x_i}}{\sum_{j=1}^{N} e^{\log(c)} e^{x_j}} \\[1.2ex]
&= \frac{e^{x_i + \log(c)}}{\sum_{j=1}^{N} e^{x_j + \log(c)}} = \frac{e^{x_i - k}}{\sum_{j=1}^{N} e^{x_j - k}} \quad \text{where } k = -\log(c)\end{align}$$
用因子 $c$ 乘以分子和分母，通过上面推导过程可以看到，如果我们巧妙地选择一个值插入到这个指数函数中，就能有效减少指数部分的计算量，我们将选择这个 $k$ 值等于输入向量中需要应用 softmax 的最大元素（$k=\max_i(x_i)$），这样一来，每个指数的参数要么为 0（当 $x_i$ 等于向量中的最大值时），要么小于 0，其指数结果介于 0 到 1 之间，这样的数值用 32 位浮点数就能轻松表示。

#### 3.3 safe softmax 算法

$$\text{softmax}(x_i) = \frac{e^{x_i - x_{\text{max}}}}{\sum_{j=1}^{N} e^{x_j - x_{\text{max}}}}$$
给定一个 $N \times N$ 的矩阵，对于每一行：

1. 寻找每一行的最大值，时间复杂度：$O(n)$，内存占用：$O(n)$
2. 计算分母归一化因子，时间复杂度：$O(n)$，内存占用：$O(n)$
3. 对向量中的每一个元素应用 softmax，时间复杂度：$O(n)$，内存占用：$O(n)$

伪代码如下：
```python
m_0 = -infty
for i=1 to N
    m_i = max(m_{i-1}， x_i)

l_0 = 0
for J=1 to N
    l_J = l_{J-1} + e^{x_J - m_N}

for K=1 to N
    x_K <- e^{x_K-m_N} / l_N
```

这里存在 3 个 for 循环。所以接下来要寻找一种策略，合并其中的某些操作，减少循环次数。

### 4、Online Softmax

#### 4.1 online softmax

尝试将前两个操作融合到一个 for 循环中，这意味着我们只需要遍历数组一次，同时计算 $m_i$，并尝试计算 $l_j$，当然，我们无法在此刻计算 $l_j$，因为无法得知全局最大值，但我们可以尝试使用当前已知的局部最大值作为估算值来进行计算，即我们尝试用 $m_i$ 替代 $m_n$。

当后续迭代过程中发现更大值时，需要对过去计算项进行修正，实际上这个校正因子非常容易计算，以 $x=[3，2，5，1]$ 为例，在前两轮中最大值为 3，第三次迭代时，最大值为 5，即在第三轮迭代中，

* 错误迭代计算：$l_3 = l_2 + e^{5-5}=e^{3-3}+e^{2-3}+e^{5-5}$
* 正确修正方法：$l_3 = l^2 \cdot \textcolor{blue}{e^{3-5}} + e^{5-5}=(e^{3-3}+e^{2-3})\textcolor{blue}{e^{3-5}}+e^{5-5}$

显然这个修正因子的计算方法为过去的最大值与当前新的最大值之间的差。

因此，softmax 新算法如下：

```python
m_0 = -infty
l_0 = 0
for i=1 to N
    m_i = max(m_{i-1}， x_i)
    l_i = l_{i-1}*e^{m_{i-1} - m_i} + e^{x_i - m_i}

for K=1 to N
    x_K <- e^{x_K-m_N} / l_N
```

#### 4.2 数学归纳法证明

1. 证明对于大小为 $N=1$ 的向量，该命题成立：$$\begin{align}
m_1 &= \max(-\infty， x_1) = x_1 = \max_i(x_i) = x_{\max} \\[1.2ex]
l_1 &= 0 \times e^{-\infty} + e^{x_1 - x_1} = \sum_{j=1}^{N} e^{x_j - x_{\max}}\end{align}$$
2. 如果假设该命题对大小为 $N$ 的向量成立，证明它对大小为 $N+1$ 的向量也成立$$\begin{align}
m_{N+1} &= \max(m_N， x_{N+1}) = \max_i(x_i) \\[1.2ex]
l_{N+1} &= l_N \cdot e^{m_N - m_{N+1}} + e^{x_{N+1} - m_{N+1}} \\
&= \left(\sum_{j=1}^{N} e^{x_j - m_N}\right)e^{m_N-m_{N+1}} + e^{x_{N+1} - m_{N+1}} \\
&= \sum_{j=1}^{N} e^{x_j - m_{N+1}} + e^{x_{N+1} - m_{N+1}} \\
&= \sum_{j=1}^{N+1} e^{x_j - m_{N+1}} \end{align}$$
---------------------------------- 视频 47:28 ----------------------------------

### 5、分块矩阵乘法

![[Pasted image 20250319151749.png|500]]

#### 5.1 忽略 softmax

目前我们先暂时忽略 softmax 的部分，即：$$\mathbf{S} = \mathbf{QK}^\top \in \mathbb{R}^{N \times N}，  \quad \mathbf{O} = \mathbf{SV} \in \mathbb{R}^{N \times d}，$$这种做法是不正确的，但它简化了我们接下来要处理的内容，

![[Pasted image 20250320101609.png|500]]

现在，每个 query 是由 Q 矩阵中的两行组成的一个组，key 也做相应的分块，在此基础上做分块矩阵乘法，如下所示：

![[Pasted image 20250320101933.png|400]]

以 $S$ 中左上角第一个分块为例，$Q_1$ 的维度为 $(2,128)$，$K^T$ 的维度是 $(128, 2)$，即 $S_{11}$ 实际上是一个 $(2, 2)$ 的小矩阵。接下来将 $S$ 矩阵与 $V$ 相乘，其结果也是显而易见的：
![[Pasted image 20250320102445.png|400]]
其运算结果为：

从宏观上看，S 矩阵 $(4,4)$，V 矩阵 $(4, 1)$，所以 O 矩阵大小 $(4,1)$，以 $O_{11}$ 为例：$$O_{11}=(Q_{1}K_{1}^T)V_{1}+(Q_{1}K_{2}^T)V_{2}+(Q_{1}K_{3}^T)V_{3}+(Q_{1}K_{4}^T)V_{4}$$这里的 $(Q_{1}K_{1}^T)=(2,128)\times(128,2)=(2,2)$，$V_{1}=(2,128)$，因此整体还是 $(2,128)$

伪代码如下：

```python
FOR EACH BLOCK Q_i
    O_i = zeroes(2， 128)                // Output is initially zeroes
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

但是需要注意的是，理论上，应用于每个小块内部元素上，我们需要知道该行的最大值，但目前暂时无法获知，举例而言，假设 $S_{11}= [a \quad b;  c \quad d]$，假设第一行的最大值是 $a$，第二行的最大值是 $d$，则 $P_{11}=[e^{a-a}\quad e^{b-a}; e^{c-d}\quad e^{d-d}]$，接下来，将 $P$ 矩阵与 $V$ 矩阵相乘，得到 $O$ 矩阵。 

*再次强调：这里计算每个 softmax$^\star$ 的最大值，并不是 $S$ 矩阵这一行的全局最大值，而是每个块的局部最大值，这实际上是错误的；*

#### 5.3 修正后的 softmax

目标是设计一个算法，既能修正用于计算每个分块下的最大值，又能同时计算归一化因子，具体实现方法如下所述，

**初始化：**

1. $m_0 = \begin{bmatrix} -\infty \\ -\infty \end{bmatrix}$ （我们的分块中有两行，每行都有一个最大值）
2. $l_0 = \begin{bmatrix} 0 \\ 0 \end{bmatrix}$ （用于累积分母部分，归一化因子）
3. $O_0 = \begin{bmatrix} 0 & 0 & \cdots & 0 \\ 0 & 0 & \cdots & 0 \end{bmatrix}$ （2x128 矩阵）

**步骤 1：**

1. 计算 $S_1 = Q_1 K_1^T$，<font color="blue">可以将</font> $S_1$ <font color="blue">想象成一个</font> $(2 \times 2)$ <font color="blue">的矩阵</font> $[a，b; c， d]$
2. 计算 $m_1 = \max(\text{rowmax}(Q_1 K_1^T)， m_0)= \begin{bmatrix}a \\ d \end{bmatrix}$，<font color="blue">按行统计的最大值</font>
3. 计算 $l_1 = \text{rowsum}\left(\exp(S_1 - m_1)\right) + l_0 \cdot \exp(m_0 - m_1)$，<font color="blue">按行累积最大值修正后的归一化因子值</font>
4. 计算 $P_{1,1} = \exp(S_1 - m_1)=[e^{a-a}\quad e^{b-a}; e^{c-d}\quad e^{d-d}]$，<font color="blue">每个元素都减去对应行中的最大值</font>
5. 更新 $O_1 = \text{diag}\left(\exp(m_0 - m_1)\right) O_0 + P_{1,1} V_1$，<font color="blue">第二项就是常规</font> $PV$ <font color="blue">计算</font>，<font color="red">第一项是对历史数据做最大值更新，同时</font> $\text{diag}$ <font color="red">是对角矩阵的含义，形式化为</font> $[m，0; 0，n] * (2 \times 128)$，<font color="red">则</font> $m$ <font color="red">仅参与和</font> $O_0$ <font color="red">的第一行计算中，这样可以保证每行的最大值仅参与该行的值更新</font>，这一点在步骤 2 中更明显。

注意：在该过程中暂时没有对 “softmax” 值进行归一化，此外，应该注意到这里实际是应用了两次 online softmax，分别用于在块内寻找局部最大值，并进行迭代更新，以及在块间寻找行内全局最大值，再次基于块整体迭代更新。

**步骤 2：**

1. 计算 $S_2 = Q_1 K_2^T[x，p; q， y]$
2. 计算 $m_2 = \max(\text{rowmax}(Q_1 K_2^T)， m_1)= \begin{bmatrix}x \\ y \end{bmatrix}$
3. 计算 $l_2 = \text{rowsum}\left(\exp(S_2 - m_2)\right) + l_1 \cdot \exp(m_1 - m_2)$
4. 计算 $P_{1,2} = \exp(S_2 - m_2)=[e^{x-x}\quad e^{p-x}; e^{q-y}\quad e^{y-y}]$
5. 更新 $O_1 = \text{diag}\left(\exp(m_1 - m_2)\right) O_1 + P_{1,2} V_2$，这里的 $O_{1}$ 是在此前最大值 $a$ 和 $d$ 基础上计算的，现在需要更新到最大值 $x$ 和 $y$ 上。

继续进行该行下的步骤 3 和 步骤 4，直到最后一步，然后应用 “$l$” 归一化因子。

**步骤 5：**

1. 计算 $O_5 = \left[\text{diag}(l_4)\right]^{-1} O_4$，<font color="red">对角矩阵的逆，就是各个元素的倒数矩阵，相当于</font> $[1/m， 0; 0， 1/n]$<font color="red">，这样就实现了除以归一化因子的目的</font>

至此，对于 $Q_1$ 的注意力计算结束，可以看到，行内是逐块顺序执行的，而行间则是并行实现的，因此后续对 $Q_2$、$Q_3$ 等的计算可以和 $Q_1$ 并行实现
 
---------------------------------- 视频 01:44:06 ----------------------------------

### 6、Flash Attention 前向传播
![[Pasted image 20250321163124.png|600]]

这是 *Flash Attention 2 的前向传播过程*，关于 Flash Attention 1 和 Flash Attention 2 之间的区别，会在后面解释。

1. 对 $Q， K， V$ 进行分块，分块大小取决于参数 $B_r$，因此每个块的大小为 $B_c \times d$
2. 初始化 $O， L$，然后接下来准备计算 softmax
3. line 3: 对 $Q_i$ 有个外层循环；line 6: 对 $K_j$ 有个内循环，与前面伪代码一致
4. line 12：计算 $O_i$，这与 5.2 中的步骤 5 完全一致
5. line 13：计算 $L_i$，这实际上是归一化因子的 $\log$，$$ \log\left(\sum_{i} \exp(x_i)\right) = x_{\text{max}} + \log\left(\sum_{i} \exp(x_i - x_{\text{max}})\right) $$
6. line 17：返回注意力 $O$，以及归一化因子的 $\log$ 值 $L$，$L$ 值用于反向传播使用。交叉熵对 logits 的梯度为 $\frac{\partial L}{\partial z_i}=y_i - t_i$，而这里的 $y_{i}=\frac{e^{s_i}}{\sum_j e^{s_j}}$，而 $L = \log \sum_j e^{s_j}$，所以 $y_i = e^{s_i - L}$

### 7、GPU、CUDA 简介

GPU 是我们购买的硬件单元，而 CUDA 是由 Nvidia 开发的软件堆栈，GPU 的任务不是同时处理多种不同的事情，而是专注于一件事或少数几件事，但处理的是海量数据，因此，我们在 GPU 上执行的操作需要大量计算，正因为如此，GPU 的大部分物理面积都用于计算单元。

#### 7.1 向量加法示例

以向量加法为例：两个向量 A 和 B，各包含 8 个元素，

![[Pasted image 20250321174457.png|600]]

CUDA 的工作机制是，当我们要求它并行启动 n 个线程时，它会分配 n 个线程，并为每个线程分配一个唯一的标识符，在这个简单的示例中，我们可以这样理解，第一个线程会被分配索引 0，每个线程处理的数据项正好对应其线程索引号，

* line 14：通过代码，根据线程标识符，指定每个线程处理的数据项，
* line 15：if 语句，指定启动 8 个线程，为什么需要加 if ？在 CUDA 中，启动的线程数总是 32 的倍数，这是 CUDA 的一个基本单位（线程束，Wrap），它们共享一个控制单元，控制单元是 GPU 硬件的一部分，负责确定接下来执行哪条指令，这意味着，这组线程将始终执行相同的指令，也就是说它们会同时到达这里的 if。由于每个线程有自己的寄存器，执行时使用的数据可能各不相同，这种编程模型称为 SIMD（data），或 SIMT（thread）；因此一些通过 if 的线程执行加法，而未通过 if 的也“不得不进入”，因为它们共享同一控制单元，但需要解决这种控制流分叉问题。大致流程为：满足条件的正常执行，不满足条件的线程进入 for 循环，但啥也不干，所有的线程必须保持同步执行相同的指令，这个现象叫控制流分支分化，显然这种空闲状态会降低程序执行效率。因此应尽可能减少这种情况的发生。

> [!NOTE]
> 在 CUDA 中，控制流分支分化（branch divergence）是指同一个线程束（warp）中的不同线程执行不同控制流路径的情况。这会导致性能下降，因为 GPU 必须顺序执行每个分支路径，而不是并行执行。控制流分支分化的影响：
> 1. 线程束执行：当线程束中的线程遇到不同的条件分支（如 `if` 语句）时，GPU 会按顺序执行每个路径，直到所有线程完成。这意味着一些线程会处于空闲状态，等待其他线程完成。
> 2. 性能下降：分支分化会导致线程束内的线程不能完全并行执行，从而降低执行效率。

#### 7.2 向量分块示例

简单示例中 8 个元素，扩大到 1 M 个元素，一次分配 1 百万个线程来执行任务，CUDA 会拒绝这样的请求，因为它超出了限制，当计算核心不足时，该如何管理并行计算呢？将输入向量划分为若干元素块，例如 GPU 由 32 个计算核，我们可以将输入向量划分为大小为 32 的块；而如果数据块大小为 32，GPU 有 64 个核，则一次可处理两个数据块，因此需要为 GPU 提供一定的工作粒度，需要增大数据的粒度，以便 GPU 能自主决定同时调度处理多少个数据块，这正是 CUDA 中引入块（blocks）概念的原因。

![[Pasted image 20250326093800.png]]

- grid：定义网格的维度（即线程块的数量和布局）。
- ​block：定义线程块（Block）的维度（即每个线程块中的线程数量和布局）。
- ​参数列表：传递给内核函数的参数（例如数组指针、标量值等）。

我们希望块的数量等于 $N / \text{block\_size}$，向上取整的值，因为 N 可能不是块大小的整数倍，接下来的问题是: 我们该如何将这些任务分配给每一个线程呢? 见 7.1 章节的图：

```c
int i = blockIdx.x * blockDim.x + threadIdx.x;
```

公式是：$\text{块 ID} \times \text{块大小} + \text{线程 ID}$

如果 GPU 有足够的空闲核心， 它可以选择同时运行一个或两个块，这就是为什么我们希望按块处理工作，因为这使得 GPU 在有足够核心的情况下，能够自主决定如何并行化操作，而且我们并不需要为 n 个元素的向量准备 n 个核心，我们可以将其划分为更小的块， 让 GPU 来管理调度。

#### 7.3 矩阵加法示例

![[Pasted image 20250326111258.png]]

公式是：$\text{（行 id）} \times \text{（一共几列）} + \text{（列 id）}$

在 C 或 C++ 中分配数组时，采用的是将所有行依次排列的扁平化数组结构， 因此我们需要依据行索引和列索引来定位数组中的元素，而这就是我们用来确定元素位置的公式.

![[Pasted image 20250326132814.png]]

grid 和 block 中都是三维尺寸，这里暂时用不到 Z 维度。

### 8、张量布局（Tensor Layouts）

#### 8.1 扁平化存储

当把一个矩阵或向量传递给 CUDA 或 CUDA 内核时， CUDA 并不会像 Python 那样一次性提供整个矩阵， 让你可以通过索引访问每个元素，而是只会给你一个指针，这个指针指向的是那个特定矩阵或向量的起始元素，然后，我们需要自己计算出所有剩余元素的内存地址。

不管是在 CPU 的内存中， 还是在 GPU 中，它将按照以下方式存储，假设第一个元素的起始地址是 100，并且每个元素由一个 16 位的浮点数组成，这意味着每个元素将占用两个字节，因此，第二个元素的起始地址将是 102，第三个元素是 104 等，这正是在 C 语言中使用 malloc 分配向量或矩阵时所得到的结果，C 语言或内存分配器会分配足够的内存来存储所有元素，并会给你一个指向该内存起始地址的指针。

矩阵会扁平化存储，按行扁平化处理的称为行主序布局，列主序布局的方式我们这里不讨论。行主序布局意味着，矩阵在内存中的存储方式为，先存储第一行的元素， 紧接着是第二行的元素。

#### 8.2 步幅属性

```text
1  2  3
5  8  13           # shape: [2, 3]   stride: [3, 1]

在内存中的实际存储样子：   1   2   3   5   8   13
           address:    62  64  66  68  70  72
```

步幅属性告诉我们，在每个维度中需要跳过多少个元素才能到达该维度的下一个索引位置，以上图为例，从一行跳到相邻另一行，需要跳过 3 个元素，从一列跳相邻另一列，只需跳过 1 个元素。*那步幅为什么有用呢？*

##### 8.2.1 矩阵重塑（Reshape）

步幅之所以有用，是因为它让我们能够轻松地重塑张量，而无需进行任何计算，

```text
1   2   3                          1   2
5   8   13           --->          3   5
                                   8   13

stride: [3,1]                      stride: [2, 1]

在内存中的实际存储样子：   1   2   3   5   8   13
           address:    62  64  66  68  70  72
```

将一个 (2,3) 的矩阵，重塑为一个 (3,2) 的矩阵，可以通过改变步幅来重塑，而无需实际改变其内存布局。

##### 8.2.2 矩阵转置（Transpose）

```text
1   2   3                          1   5
5   8   13           --->          2   8
                                   3   13

stride: [3,1]                      stride: [1, 3]

在内存中的实际存储样子：   1   2   3   5   8   13
           address:    62  64  66  68  70  72
```

可以在不改变内存中任何内容的情况下，将同一矩阵既视为未转置的版本，又视为转置后的版本，只需交换这两个维度上的步幅即可。

由此可见， *步长基本上能让我们实现两件事*，一是它允许我们重塑张量，而无需在内存中重新分配其存储结构，二是能够在不重新排列内存中元素的情况下转置矩阵，这非常棒，因为移动内存数据的开销很大。在 PyTorch 中，有两种方法可以重塑张量，`reshape` 和 `view`，在通过交换两个维度的步幅来转置矩阵后，就无法再免费重塑张量了，

> [!tip]
> 在 PyTorch 中，`reshape` 和 `view` 都用于重塑张量，但它们在某些情况下的行为有所不同。
> 
> - `view` 需要原始张量是连续的（即内存中数据的存储顺序没有变化），否则可能会失败。
> - `reshape` 更灵活，可以在必要时自动创建张量的副本。
> 
> 当通过改变步幅（如转置操作）来交换两个维度时，张量在内存中的存储顺序发生了变化，数据不再是连续的。这种情况下，`view` 可能无法正常工作，因为它依赖于数据的连续性。而 `reshape` 则可以处理这种情况，因为它会在需要时创建数据的副本来满足要求。

张量的步幅本质上是什么呢? 步幅是如何计算的呢? 步幅其实就是所有后续维度形状的乘积，$$\text{stride}[i] = 
\begin{cases} 
\displaystyle\prod_{j=i+1}^{N} \text{shape}[j]， & \text{if } i < N \\[1.5ex]
1， & \text{if } i = N 
\end{cases}$$
例如，一个 3D 矩阵的形状是 `(2, 4, 3)`，则步幅属性值为 `(12, 3, 1)`，转置后，例如变为 `(12, 1, 3)`，就破坏了这种规律，失去了步幅特性。

当我们进行转置时， 这种步幅特性就会丢失，在通过交换步幅完成矩阵转置后我们无法再进行进一步的形状重塑操作，从根本上说，是因为张量在逻辑上不是连续的。基本上，在 PyTorch 中，无法在张量转置后对其进行视图操作，因为 PyTorch 在转置张量时只是交换了两个步幅，但失去了步幅特性，这本质上是步幅将不再等于后续形状的乘积。

---------------------------------- 视频 02:40:49 ----------------------------------

### 9、进入 Triton

[Triton 官方文档](https://triton-lang.org/main/index.html)，本篇教程参考的[教程](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html#sphx-glr-getting-started-tutorials-06-fused-attention-py)，

[官网向量加法教程](https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html)

*在 CUDA 中编写的程序是以线程为单位的*，每个线程需要明确自己应该执行什么操作，而*在 Triton 中操作的单位是数据块*，我们是以线程块为单位进行操作的。

### 10、Flash Attention 中的并行化

![[Pasted image 20250321163124.png|600]]

这段代码是 FlashAttention 的前向传播过程，我们可以并行化输出的计算过程， 因为输出块依赖于查询块以及所有的键块，不同的输出块依赖于不同的查询块与所有键块，它们可以相互独立地工作，当然它们共享相同的键块。

#### 10.1 共享内存

在 GPU 中，我们拥有高带宽内存，类似于计算机中的 RAM，如 A100 的 40GB 高带宽内存（HBM）的容量，也就是 DRAM。

![[Pasted image 20251216182042.png|500]]

如图所示，最下面是 DRAM，它是 GPU 所具备的大容量内存，接着，每个流式多处理器 (线程块)，实际上还拥有一块共享内存，它比 DRAM 小得多，区别在于访问 DRAM 的速度非常慢，而访问共享内存的速度则极快。

CUDA 与 Triton 的一个区别在于，当在 CUDA 中加载某些数据时，是直接从全局内存中加载的，例如，在使用 torch 实现的原始版本注意力计算中，当我们启动一个 CUDA 内核时，我们先将张量或向量从 CPU 复制到 GPU，它们会驻留在 GPU 的全局内存中，然后我们直接从全局内存中加载这些元素。

在 Triton 中，每当加载一些数据时，实际上是将数据从全局内存复制到共享内存中，然后，所有操作都在共享内存上完成，而当存储数据时，再将数据从共享内存复制回全局内存，这一过程大大提升了速度。因此，我们始终操作的是那些已加载到共享内存中的元素，本质上是由同一线程块内的所有线程共享的。

#### 10.2 FlashAttention 算法回顾

**FlashAttention-1 vs. FalshAttention-2：**

在 FlashAttention-2 中，有一个外循环，用于遍历所有的 $Q$ 块，以及一个内循环，用于遍历所有的 $K$ 块，而在 FlashAttention-1 中，外循环处理的是 $K$ 块，而内循环处理的是 $Q$ 块，这种设计降低了算法的并行化能力，为什么？

由于注意力机制的输出可以针对每个 $Q$ 块独立计算，因此，对于 $Q$ 的外循环，实际上并不需要真正运行一个循环，而是可以启动多个内核并行处理，而内循环 $K$ 块则是我们必须遍历的部分，因此每个 Triton 内核将负责处理一个 $Q$ 块，并逐一遍历所有的 $K$ 块，在 $K$ 块内部，我们将执行之前探讨过的那些操作，在循环结束时，我们需要将输出结果存储回 HBM 内存中。

另外需要注意的是，这里的 $Q$、$K$ 和 $V$ 都是 $n \times d$ 维的矩阵，而通常处理的是一个 batch 的序列，因此，我们还可以在批次中的序列上进行并行化处理，每个批次可以独立工作。在每个序列内部，每个注意力头可以独立工作，每个小 $Q$ 块也能独立正作，这就是我们实现并行化的方式。那么我们最多能有多少个程序同时并行运行呢？它等于 `batch_size * head_num * q_block_num`

### 11、Triton 编码实现

**自定义实现与 Triton 文档版本区别：**

1. 没有使用 FP8，这对我们的解释来说并不必要，当然，使用 FP8 
2. Triton 网站上的 FlashAttention 中，反向传播仅针对因果注意力机制实现，而我们会同时支持因果和非因果注意力机制，尽管速度会慢一些
3. Triton 中显式的使用了 softmax 缩放因子，而我们实际上在需要时才应用这个缩放因子
4. Triton 在线计算 FlashAttention 时，使用的是 $2^x$ 并非 $e^x$，然后通过使用对数进行补偿

### 12、从导数到雅可比矩阵

导数：$$f'(x)=\lim_{ h \to 0 } \frac{f(x+h)-f(x)}{h}=\frac{\partial f(x)}{\partial x}=\frac{\partial y}{\partial x}$$
稍加整理：$$\begin{align*}
f(x + h) &\approx f'(x) \cdot h + f(x) \\[0.5em]
f(x + \Delta x) &\approx f'(x) \cdot \Delta x + f(x) \\[0.5em]
f(x + \Delta x) &\approx \frac{\mathrm{d}y}{\mathrm{d}x} \cdot \Delta x + f(x) \\[0.5em]
y^{\text{NEW}} &\approx \frac{\mathrm{d}y}{\mathrm{d}x} \cdot \Delta x + y^{\text{OLD}}
\end{align*}$$
所以，当 $x$ 改变了 $\Delta x$，$y$ 将近似的改变 $\frac{\mathrm{d}y}{\mathrm{d}x}\cdot \Delta x$

#### 12.1 链式法则

假设：$z=f(g(x))$

由 $x^{\text{new}}=x^{\text{old}}+\Delta x$  推导出  $y^{\text{new}} \approx \frac{\mathrm{d}y}{\mathrm{d}x} \cdot \Delta x + y^{\text{old}}$

由 $y^{\text{new}}=y^{\text{old}}+\Delta y$  推导出  $z^{\text{new}} \approx \frac{\mathrm{d}z}{\mathrm{d}y} \cdot \Delta y + z^{\text{old}}$，

将上式 $\Delta y$ 部分带入可得，$z^{\text{new}} \approx z^{\text{old}} +\frac{\mathrm{d}z}{\mathrm{d}y} \cdot \frac{\mathrm{d}y}{\mathrm{d}x} \cdot \Delta x$

于是，可得：$\frac{\partial z}{\partial x}=\frac{\partial z}{\partial y}\cdot \frac{\partial y}{\partial x}$
 
#### 12.2 梯度

梯度：函数的输入是向量，输出是标量。$f$：$R^{N}\to R$

例如：$f\!\left(\begin{bmatrix} x_1 \\ x_2 \end{bmatrix}\right) = y$，$y^{\text{new}} \approx y^{\text{old}} + \nabla f \cdot \Delta x$，这里的 $\Delta x$ 也不再是标量，是向量，将其与梯度点积

梯度定义：$\nabla f = \begin{pmatrix} \displaystyle \frac{\partial y}{\partial x_1}, & \displaystyle \frac{\partial y}{\partial x_2}, & \dots \end{pmatrix}$，所以有：$y^{\text{new}} \;\approx\; y^{\text{old}} + \frac{\partial y}{\partial x_1} \Delta x_1 + \frac{\partial y}{\partial x_2} \Delta x_2 + \dots$

梯度实际上是一个由输出相对于输入向量中，每个变量的偏导数组成的。

#### 12.3 雅可比矩阵

雅可比：函数的输入是向量，输出也是向量。$f$：$R^{N}\to R^M$

例如：$f\left(\begin{bmatrix} x_1 \\ x_2 \end{bmatrix}\right) =\begin{bmatrix} y_1 \\ y_2  \\ y_3 \end{bmatrix}$，$x^{\text{old}} \to x^{\text{old}} + \Delta x$，有 $y^{\text{new}} \;\xrightarrow{\approx}\; y^{\text{old}} + \frac{\partial y}{\partial x} \Delta x$

雅可比矩阵：$$\text{Jacobian} = 
\begin{bmatrix}
\displaystyle \frac{\partial y_1}{\partial x_1} & \displaystyle \frac{\partial y_1}{\partial x_2} & \cdots & \displaystyle \frac{\partial y_1}{\partial x_N} \\[1.5em]
\displaystyle \frac{\partial y_2}{\partial x_1} & \displaystyle \frac{\partial y_2}{\partial x_2} & \cdots & \displaystyle \frac{\partial y_2}{\partial x_N} \\[1.5em]
\vdots & \vdots & \ddots & \vdots \\[1.5em]
\displaystyle \frac{\partial y_M}{\partial x_1} & \displaystyle \frac{\partial y_M}{\partial x_2} & \cdots & \displaystyle \frac{\partial y_M}{\partial x_N}
\end{bmatrix}$$
其中 $\frac{\partial y}{\partial x} \Delta x$ 是矩阵-向量乘法，$(m,n) \times (n,1)=(m,1)$ 

#### 12.4 广义雅可比矩阵

广义雅可比矩阵：函数输入是张量，输出也是张量。$f: \mathbb{R}^{N_1 \times \cdots \times N_{D_x}} \to \mathbb{R}^{M_1 \times \cdots \times M_{D_y}}$

例如：$f\bigl( D_x\text{-dimensional tensor } \mathbf{x} \bigr) = D_y\text{-dimensional tensor } \mathbf{y}$

其中 $\frac{\partial y}{\partial x} \Delta x$ 是张量乘法，$(M_1 \times \cdots \times M_{D_y})\times(N_1 \times \cdots \times N_{D_x})$ 

#### 12.5 自动求导（Autograd）

```
(a)   -----  (x)  -----  (+)  -----  (z^2)  ------  (phi)
              |           |
              |           |
             (w_1)       (b_1)
```

表达式：$\phi=y_{3}=(y_{2})^2=(y_{1}+b_{1})^2=(aw_{1}+b_{1})^2$

对 $w_{1}$ 求偏导：$\frac{\partial\phi}{\partial w_{1}}=2(aw_{1}+b_{1})(a)=2a(aw_{1}+b_{1})$

链式求导：$\frac{\partial\phi}{\partial w_{1}}=\frac{\partial\phi}{\partial y_{3}} \cdot \frac{{\partial y_{3}}}{\partial y_{2}} \cdot \frac{{\partial y_{2}}}{\partial y_{1}} \cdot \frac{{\partial y_{1}}}{\partial w_{1}}$


#### 12.6 雅可比矩阵稀疏性

由于输入 $X$ 的维度 $(N,D)$ 以及权重 $W$ 的维度 $(D,M)$ 都非常大，会导致雅可比矩阵非常的大，然而这个雅可比矩阵及其稀疏，

```
--------------              - - - - - - - -               = = = = = = = 
--------------              |             |               = = = = = = =
--------------      x       |             |        =      = = = = = = =
--------------              |             |               = = = = = = =
--------------              |--------------               = = = = = = =
  X (N,D)                       W (D,M)                       Y (N,M)
```

以图示为例，$X$ 每一行是一个 token，维度是 $D$，输出中每一行是该 token 的某种表示，以 $Y$ 中的第一行值为例，$Y$ 中的第一行是由 $X$ 的第一行与 $W$ 的每一列乘积得到的，也就是说与 $X$ 的其他行无关，其偏导自然为 0，这就产生大量的稀疏性。不过我们一般无需对 $X$ 求导，对 $W$ 求导一样的。
 
#### 12.7 如何不实际生成雅可比矩阵进行优化

假设一个具体的示例：$x$ 是大小为 $(n,d)$ 的张量，假设 $n=1,d=3$，而 $w$ 大小 $(d,m)$，其中 $m=4$，则 $y$ 的大小为 $(n,m)=(1,4)$，PyTorch 提供损失函数相对于该操作符输出 $y$ 的梯度，也是 $(n,m)$ 大小，我们需要计算损失函数相对于 $x$ 的梯度，这将是一个形状为 $(n,d)$ 的张量，在处理梯度时，它总是与输入变量的形状相同。
$$y = \begin{bmatrix} x_{11} & x_{12} & x_{13} \end{bmatrix} \begin{bmatrix} w_{11} & w_{12} & w_{13} & w_{14} \\ w_{21} & w_{22} & w_{23} & w_{24} \\ w_{31} & w_{32} & w_{33} & w_{34} \end{bmatrix}$$
现在，PyTorch 提供$$\frac {\partial \phi} {\partial y} = \begin{bmatrix} \mathrm{d}y_{11} & \mathrm{d}y_{12} & \mathrm{d}y_{13} & \mathrm{d}y_{14} \end{bmatrix}$$
当求解 $\partial \phi/\partial x$ 时，常规做法是利用链式法则，通过上式乘以雅可比矩阵 ${\partial y}/{\partial x}$ 计算。

$y$ 的结果中有 4 项：$(x_{11}w_{11}+x_{12}w_{21}+x_{13}w_{31})$、$(x_{11}w_{12}+x_{12}w_{22}+x_{13}w_{32})$、$(x_{11}w_{13}+x_{12}w_{23}+x_{13}w_{33})$ 和 $(x_{11}w_{14}+x_{12}w_{24}+x_{13}w_{34})$，

根据这 4 项，如果我们计算雅可比矩阵，可以发现 $\frac{\partial y}{\partial x}=w^T$

所以：$$\frac{\partial \phi}{\partial x}=\frac{\partial \phi}{\partial y}\cdot w^{T} \qquad \frac{\partial \phi}{\partial w}=x^{T}\cdot \frac{\partial \phi}{\partial y} $$
这个公式的记忆方法是根据各个矩阵的维度，例如后一项，左侧应该与 $w$ 维度相同 $(d,m)$，而 $X^T$ 的维度为 $(d,n)$，而最后一项维度是 $(n,m)$

#### 12.8 推导通过 softmax 的梯度

$$S=QK^{T} \qquad P=\text{softmax}(S) \qquad O=PV$$
按行处理 softmax 等，假设 $S_{i}$ 是其第 $i$ 行，$S_i = S[i, :] \in R^N$，$P_{i}=\text{softmax}(s_{i}) \in R^N$，其中：$$\text{softmax}(P_{ij})=\frac{e^{S_{ij}-S_{i\max}}}{\sum_{l=1}^{N}e^{S_{il}-S_{i\max}}}$$前文中，我们介绍的是为了保证数值稳定，包含了减去最大值的操作，但从数学上讲结果一样，减不减最大值操作不影响求导结果。我们需要计算：$$\frac{\partial \phi}{\partial S_{i}}=\frac{\partial \phi}{\partial P_{i}}\cdot\frac{\partial P_{i}}{\partial S_{i}}$$其中我们需要计算最后一项$$\frac{\partial P_{ij}}{\partial S_{ik}}=\frac{\partial \left[\frac{e^{S_{ij}}}{\sum_{l=1}^{N}e^{S_{il}}}\right]}{\partial S_{ik}}$$
举个具体示例，比如 $S= \begin{matrix}[S_{11}&S_{12}&S_{13}]\end{matrix}$，$P= \begin{matrix}[P_{11}&P_{12}&P_{13}]\end{matrix}$，这个雅可比矩阵的第一行是 $P_{11}$ 分别对 3 个 $S$ 项的偏导，所以上式求导，分两种情况：

分式求导法则：
$$\left[\frac{f(x)}{g(x)}\right]'=\frac{g(x)f'(x)-f(x)g'(x)}{g^2(x)}$$

1. $j=k$ 情况下：分子部分：$e^{s_{ij}}\left( \sum \right)-e^{s_{ij}}e^{s_{ik}}$，分母部分是 $\left( \sum \right)^2$，所以结果是：$$\frac{e^{s_{ij}}\left( \sum^N_{l=1}e^{s_{il}}-e^{s_{ik}} \right)}{\left( \sum^N_{l=1} e^{s_{il}}\right)^2}=\frac{e^{s_{ij}}}{ \sum^N_{l=1} e^{s_{il}}} \cdot \frac{\sum^N_{l=1}e^{s_{il}}-e^{s_{ik}} }{ \sum^N_{l=1} e^{s_{il}}}=P_{ij}(1-P_{ik})$$
2. $j\neq k$ 情况下：分子第一项将是 0，所以结果是：$$\frac{e^{s_{ij}}(-e^{s_{ik}})}{\left( \sum^N_{l=1} e^{s_{il}}\right)^2}=\frac{e^{s_{ij}}}{ \sum^N_{l=1} e^{s_{il}}} \cdot \frac{-e^{s_{ik}} }{ \sum^N_{l=1} e^{s_{il}}}=P_{ij}(-P_{ik})$$
雅可比矩阵：$$\frac{\partial P_{ij}}{\partial s_{ik}} = \begin{bmatrix}
P_{i1}(1-P_{i1}) & -P_{i1}P_{i2} & -P_{i1}P_{i3} & \cdots & -P_{i1}P_{iN} \\
-P_{i2}P_{i1} & P_{i2}(1-P_{i2}) & -P_{i2}P_{i3} & \cdots & -P_{i2}P_{iN} \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
-P_{iN}P_{i1} & -P_{iN}P_{i2} & -P_{iN}P_{i3} & \cdots & P_{iN}(1-P_{iN})
\end{bmatrix}$$
这是个对称矩阵，可以写为 $\text{diag}(P_{i})-PP^T$，所以 $y=\text{softmax}(x)$，$\frac{\mathrm{d}y}{\mathrm{d}x}=\text{diag}(y)-yy^T$


