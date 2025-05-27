
[An In-depth exploration of Rotary Position Embedding (RoPE)](https://aiexpjourney.substack.com/p/an-in-depth-exploration-of-rotary-position-embedding-rope-ac351a45c794)


旋转位置编码（RoPE）是一种广泛使用的位置编码技术，被许多大型语言模型所采用，例如Llama、Llama2、PaLM、CodeGen 等。

最近，我仔细研究了关于 RoPE 的论文，并推导了其中的公式。我想在这里分享这些内容，希望能帮助读者理解这一巧妙的想法。

本文主要由三部分组成，包括对底层原理的介绍、可视化图解以及对 Llama 模型中 RoPE 代码的分析。

### 为什么我们需要位置编码技术？

Transformer 模型的卓越性能归功于其核心的注意力机制，该机制会计算输入序列中每个标记之间的注意力权重。

假设一个序列有 $N$ 个标记。第 $m$ 个标记的嵌入是 $x_m$，第 $n$ 个标记的嵌入是 $x_n$。

在不向词嵌入添加位置信息的情况下，我们可以将它们转换为查询 $q_m$、键 $k_n$ 和值 $v_n$，如方程 (1) 所示：
$$\begin{align}
\mathbf{q}_m &= W_q^m \mathbf{x}_m \\
\mathbf{k}_n &= W_k^n \mathbf{x}_n \\
\mathbf{v}_n &= W_v^n \mathbf{x}_n
\end{align} \tag{1}$$
查询和键用于计算注意力权重，而输出则计算为值的加权和，如公式（2）所示：

$$\begin{align}
a_{m,n} &= \frac{\exp\left(\frac{\mathbf{q}_m^\top \mathbf{k}_n}{\sqrt{d}}\right)}{\sum_{j=1}^{N} \exp\left(\frac{\mathbf{q}_m^\top \mathbf{k}_j}{\sqrt{d}}\right)} \\
\mathbf{o}_m &= \sum_{n=1}^{N} a_{m,n} \mathbf{v}_n
\end{align} \tag{2}
$$
我们发现，当不包含位置信息时，标记 $x_m$ 和 $x_n$ 之间的注意力权重 $a(m, n)$ 无论它们的位置如何都保持不变。换句话说，注意力权重 $a(m, n)$ 与位置无关，这与我们的直觉相悖。例如，“狗咬猫” 和 “猫咬狗” 的含义明显不同。

此外，当两个标记之间的距离较近时，我们希望它们之间的注意力权重较大；反之，当距离较远时，注意力权重应较小。

为了解决这个问题，我们可以为模型引入位置编码。这使得每个词嵌入都能包含其在输入序列中的位置信息。我们定义一个函数 $f$，将位置信息 $m$ 融入词嵌入 $x_m$ 中，得到 $q_m$。同样地，将位置信息 $n$ 融入词嵌入 $x_n$ 中，得到 $k_n$ 和 $v_n$，如公式 (3) 所示：

$$\begin{align}
\mathbf{q}_m &= f_q(\mathbf{x}_m, m) \\
\mathbf{k}_n &= f_k(\mathbf{x}_n, n) \\
\mathbf{v}_n &= f_v(\mathbf{x}_n, n)
\end{align} \tag{3}
$$
在加入位置信息后，我们可以将方程（3）代入方程（2），从而在注意力机制中引入位置信息。

### 旋转位置嵌入（RoPE）的核心思想

RoPE 旨在将相对位置信息 $(m - n)$ 融入方程（3）中 $q_m$ 和 $k_n$ 的内积运算。

我们如何判断它是否包含位置信息？只需将 $q_m$ 和 $k_n$ 的内积表示为 $x_m$、$x_n$ 和 $m-n$ 的函数 $g(x_m, x_n, m-n)$，其中 $m-n$ 代表两个向量之间的相对位置信息。因此，我们的建模目标就变成了寻找满足以下方程 (4) 的函数 $f$：
$$\mathbf{q}_m^\top \mathbf{k}_n = \langle f_q(\mathbf{x}_m, m), f_k(\mathbf{x}_n, n) \rangle = g(\mathbf{x}_m, \mathbf{x}_{n}, m-n)\tag{4}$$
在注意力机制中，token 之间的交互隐含在 query 和 key 的点积运算中。如果我们定义 $q_m$ 和 $k_n$ 的点积为 $m-n$ 的函数，就可以通过函数 $f$ 实现绝对位置编码，从而为每个 token 赋予位置信息。

### RoPE 如何找到一个满足条件的函数 $f$

*目前已知的唯一信息是方程（3）和（4），除此之外一无所知。*

在广阔的函数空间中，找到一个满足给定条件的函数 $f$ 并非易事。*面对难题时，常见的解决思路是尝试将其简化：先考虑简单明晰的情形，再推广至更复杂的场景*。

#### Step 1: 假设嵌入维度为 2，简化问题。

大语言模型的嵌入维度显然远大于 2，但我们可以从这个简单案例进行推广。

在二维情况下，$q_m$ 和 $k_n$ 都是二维向量。*对于二维向量，我们可以将其视为复平面上的复数*。因此，$q_m$ 和 $k_n$ 可以用复数形式表示，包含各自的模和幅角。同样，我们也可以用复数形式表示内积函数 $g$，其中 $R$ 和 $\theta$ 分别代表模和幅角。由此得到方程 (5)：

$$\begin{align}
\mathbf{q}_m &= f_q(\mathbf{x}_m, m) = R_q(\mathbf{x}_m, m) e^{i \Theta_q(\mathbf{x}_m, m)} \\
\mathbf{k}_n &= f_k(\mathbf{x}_n, n) = R_k(\mathbf{x}_n, n) e^{-i \Theta_k(\mathbf{x}_n, n)} \\
g(\mathbf{x}_m, \mathbf{x}_n, m - n) &= R_g(\mathbf{x}_m, \mathbf{x}_n, m - n) e^{i \Theta_g(\mathbf{x}_m, \mathbf{x}_n, m - n)}
\end{align}  \tag{5}
$$

> [!NOTE]
> 公式 (5) 中间 $\mathbf{k}_n$ 应该取共轭复数表示，这样可以保证 $q_{m} \cdot k_{n}=R_{q}R_{k}e^{i(\theta_{q}-\theta_{k})}$


#### Step 2: 将方程（5）代入方程（4）

我们可以得到以下关系：

$$\begin{align}
R_q(\mathbf{x}_m, m) R_k(\mathbf{x}_n, n) &= R_g(\mathbf{x}_m, \mathbf{x}_n, m - n) \tag{6}\\[1.2ex]
\Theta_q(\mathbf{x}_m, m) - \Theta_k(\mathbf{x}_n, n) &= \Theta_g(\mathbf{x}_m, \mathbf{x}_n, m - n)\tag{7}\end{align}
$$

#### Step 3: 根据方程（6）计算函数 $f$ 的模数

对于方程（6），令 $m = n$，我们得到方程（8）：
$$
R_q(\mathbf{x}_m, m) R_k(\mathbf{x}_n, m) = R_g(\mathbf{x}_m, \mathbf{x}_n, 0) = R_q(\mathbf{x}_m, 0) R_k(\mathbf{x}_n, 0) = \|\mathbf{q}\| \|\mathbf{k}\|  \tag{8}$$
这表示在不同位置编码下，模长保持不变。

式 (8) 中第二个等号成立的原因是，对于式 (6)，我们可以设 $m = n = 0$。最后一个等号成立是因为两个向量夹角为 0，两个向量内积等于两个向量的模相乘，$\mathbf{a} \cdot \mathbf{b} = \|\mathbf{a}\| \|\mathbf{b}\| \cos \theta$

由于方程（5）的初始条件（$m = 0，n = 0$），方程（8）的最终等式成立，如方程（9）所示：

$$\begin{align}
\mathbf{q} = f_q(\mathbf{x}_m, 0) = \|\mathbf{q}\| e^{i\theta_q} = R_q(\mathbf{x}_m, 0) e^{i\Theta_q(\mathbf{x}_m, 0)} \\[1.2ex]
\mathbf{k} = f_k(\mathbf{x}_n, 0) = \|\mathbf{k}\| e^{i\theta_k} = R_k(\mathbf{x}_n, 0) e^{i\Theta_k(\mathbf{x}_n, 0)}
\end{align}\tag{9}
$$

由方程 (8) 可知，函数 $f$ 的模仅与 $q_m$ 和 $k_n$ 的模有关，而与 $m$ 的值无关。因此，我们直接采用最简单的关系给出一个解：

$$\begin{align}
R_q(\mathbf{x}_m, m) = \|\mathbf{q}\| \\[1.2ex]
R_k(\mathbf{x}_n, m) = \|\mathbf{k}\|
\end{align}\tag{10}
$$
这样，我们就得到了函数 $f$ 的模。接下来，我们需要找到函数 $f$ 的幅角。


#### Step 4: 根据方程（7）确定函数 $f$ 的参数

对于方程（7），设 $m = n$，我们得到方程（11）：

$$\Theta_q(\mathbf{x}_m, m) - \Theta_k(\mathbf{x}_n, m) = \Theta_g(\mathbf{x}_m, \mathbf{x}_n, 0) = \Theta_q(\mathbf{x}_m, 0) - \Theta_k(\mathbf{x}_n, 0) = \theta_q - \theta_k
\tag{11}$$

方程（11）中第二个等号成立的原因是，对于方程（7），我们可以设 $m = n = 0$。最终等式成立是由于方程（9）。

根据方程（11）重新排列：

$$\Theta_q(\mathbf{x}_m, m) - \theta_q = \Theta_k(\mathbf{x}_n, m) - \theta_{k}\tag{12}$$

观察方程（12），它解释了一个重要问题。*方程（12）两边的值仅与 $m$ 相关，而与 $x$ 无关*。无论 $x = x_m$ 还是 $x = x_n$，结果都保持不变。方程（12）的左边可以表示为：

$$\phi(m) = \Theta_q(\mathbf{x}_m, m) - \theta_{q} \tag{13}$$

观察 $ϕ(m+1)$ 与 $ϕ(m)$ 之间的关系：

$$\begin{align*}
\phi(m+1) - \phi(m) &= \Theta_q(\mathbf{x}_m, m+1) - \theta_q - (\Theta_q(\mathbf{x}_m, m) - \theta_q) \\
&= \Theta_q(\mathbf{x}_m, m+1) - \Theta_q(\mathbf{x}_m, m) \\
&= \Theta_g(\mathbf{x}_m, \mathbf{x}_m, 1) \quad (\because \text{eq(7)})
\end{align*} \tag{14}$$

可以看出，$ϕ(m)$ 是 $m$ 的函数，而 $ϕ(m+1) - ϕ(m)$ 的值与 $m$ 无关。这表明 $ϕ(m)$ 应该是关于 $m$ 的等差数列。
$$\phi(m) = m\theta + \gamma \tag{15}$$
可以看出，步骤 4 是为了证明 ${ϕ(m)}$ 是一个等差数列。

#### Step 5: 寻找函数 $f$

结合方程（10）和（15），我们发现函数 $f$ 的模和幅角已经确定，这意味着我们已经找到了函数 $f$。

具体而言，将方程（15）（为简化起见，设 $γ = 0$）和方程（10）、（13）代入方程（5）：

$$\begin{align*}
\mathbf{q}_m &= f_q(\mathbf{x}_m, m) = R_q(\mathbf{x}_m, m) e^{i \Theta_q(\mathbf{x}_m, m)} \\
&= R_q(\mathbf{x}_m, m) e^{i(m\theta + \theta_q)} \\
&= ( \|\mathbf{q}\| e^{i\theta_q} ) e^{im\theta} \\
&= \mathbf{q} e^{im\theta} \quad (\because \text{eq(9)})
\end{align*} \tag{16}$$

#### Step 6: 确定 $q$ 和最终结果

方程（3）的一个典型选择是：

$$\mathbf{q}_m = f_q(\mathbf{x}_m, m) = \mathbf{W}_q (\mathbf{x}_m + \mathbf{p}_{m)} \tag{17}$$

其中 $p_m$ 是一个取决于标记 $x_m$ 位置的向量。

回顾方程（9）中 $q$ 的定义，它是在 $m=0$ 的情况下定义的。这里，我们假设当 $m=0$ 时没有位置信息，这样做也是为了与方程（17）兼容。我们直接将其定义为：

$$\mathbf{q} = f_q(\mathbf{x}_m, 0) = \mathbf{W}_q \mathbf{x}_{m} \tag{18}$$
所以最终的结果是：

$$\begin{align*}
\mathbf{q}_m &= f_q(\mathbf{x}_m, m) = (\mathbf{W}_q \mathbf{x}_m) e^{im\theta} \\[1.2ex]
\mathbf{k}_n &= f_k(\mathbf{x}_n, n) = (\mathbf{W}_k \mathbf{x}_n) e^{in\theta}
\end{align*}\tag{19}$$

我们可以将方程（19）代入方程（10）来验证其同样成立。感兴趣的读者可以自行计算。

将方程（19）表示为二维矩阵形式，其中 $W_q$ 为 $2 \times 2$ 矩阵，$x_m$ 和 $q$ 为二维向量：

$$\begin{align}
\mathbf{q}_m &= f_q(\mathbf{x}_m, m) = 
\begin{pmatrix}
\cos(m\theta) & -\sin(m\theta) \\
\sin(m\theta) & \cos(m\theta)
\end{pmatrix}
\begin{pmatrix}
\mathbf{W}_q^{(11)} & \mathbf{W}_q^{(12)} \\
\mathbf{W}_q^{(21)} & \mathbf{W}_q^{(22)}
\end{pmatrix}
\begin{pmatrix}
x_m^{(1)} \\
x_m^{(2)}
\end{pmatrix} \\[1.1ex] \\
&=
\begin{pmatrix}
\cos(m\theta) & -\sin(m\theta) \\
\sin(m\theta) & \cos(m\theta)
\end{pmatrix}
\begin{pmatrix}
q^{(1)} \\
q^{(2)}
\end{pmatrix}
\end{align}$$

*这是一个向量旋转函数，意味着通过将向量旋转 $mθ$ 角度，我们可以为向量添加绝对位置信息。这就是旋转位置编码的由来。数学之美令人惊叹。*


### 视觉呈现

为了更好地理解 RoPE 中的位置编码，以下描述结合图形来说明如何将位置编码分配给二维嵌入。

假设二维嵌入 $q = (1, 0)$，且方程 (20) 中的 $θ$ 为常数，本例中设 $θ = 1$。当标记位于位置 $m = [0, 1, 2, 3, 4, 5]$ 时，可为其分配对应的位置信息，如图所示：

![|400](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fa5b191a5-0104-499c-b48f-ea7ce14e5f74_800x598.png)

### 提升至高维空间

前文介绍了如何为二维向量分配位置信息，这可以通过旋转一定角度来实现。但在实际应用中，嵌入的维度通常高达数百甚至数千。现在的问题是，如何将二维情况推广到多维情形。

论文中提出的方法相当直接。通常，嵌入维度是偶数。因此，我们将高维向量分解成对并分别旋转它们。高维向量的旋转可以表示为以下方程：

$$\mathbf{q}_m = f_q(\mathbf{x}_m, m) = \mathbf{R}_{\theta, m}^d \mathbf{W}_q \mathbf{x}_{m}\tag{21}$$

$$\mathbf{R}_{\theta, m}^d =
\begin{pmatrix}
\cos(m\theta_1) & -\sin(m\theta_1) & 0 & 0 & \cdots & 0 & 0 \\
\sin(m\theta_1) & \cos(m\theta_1) & 0 & 0 & \cdots & 0 & 0 \\
0 & 0 & \cos(m\theta_2) & -\sin(m\theta_2) & \cdots & 0 & 0 \\
0 & 0 & \sin(m\theta_2) & \cos(m\theta_2) & \cdots & 0 & 0 \\
\vdots & \vdots & \vdots & \vdots & \ddots & \vdots & \vdots \\
0 & 0 & 0 & 0 & \cdots & \cos(m\theta_{d/2}) & -\sin(m\theta_{d/2}) \\
0 & 0 & 0 & 0 & \cdots & \sin(m\theta_{d/2}) & \cos(m\theta_{d/2})
\end{pmatrix} \tag{22}$$

这里 $θ$ 都是常数，在论文中直接赋值，其灵感可能来自正弦位置编码：
$$
\theta_i = 10000^{-2(i-1)/d}, \quad i \in [1, 2, \ldots, d/2]
\tag{23}$$
其中 $d$ 是嵌入维度。

下图展示了处理高维情况的方法：


![|500](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F40b600d4-9286-42ab-9b2a-c7a145809569_800x456.png)

### Llama 中的 RoPE 实现

以下代码片段均来自[同一文件](https://github.com/facebookresearch/llama/blob/main/llama/model.py)。我在关键代码部分添加了注释。

#### precompute_freqs_cis 函数

*该函数的目的就是计算了公式 (23) 备用*，`torch.outer` 用于计算两个一维张量的外积的函数。外积是指将一个向量中的每个元素与另一个向量中的每个元素相乘，结果是一个矩阵，例如 `[1,2,3]` 与 `[4,5]` 的外积是 `[[4,5], [8,10], [12, 15]]` 相当于互相组合起来；`torch.polar` 用于将极坐标形式的张量转换为复数形式。给定幅度和相位角，`torch.polar` 返回对应的复数。

```python
def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """
    预计算给定维度的复指数（cis）频率张量。

    此函数使用给定维度 'dim' 以及结束索引 'end' 计算带有复指数的频率张量
    参数 'theta' 用于缩放频率。

    Args:
        dim (int): Dimension of the frequency tensor.
        end (int): End index for precomputing frequencies.
        theta (float, optional): Scaling factor for frequency computation. Defaults to 10000.0.

    Returns:
        torch.Tensor: Precomputed frequency tensor with complex exponentials.
    """
    # 每组包含嵌入的两个分量，计算每组对应的旋转角度 theta_i。
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    # 生成 token 序列索引 m = [0, 1, ..., sequence_length - 1]
    t = torch.arange(end, device=freqs.device)  # type: ignore
    # 计算 m * theta_i
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cisp
```

这个函数仍然相当抽象。让我举个例子来说明，当 `dim = 4`（嵌入维度为 4）且序列长度为 3 时，生成的 `freqs_cis` 将是：

```
tensor([[ 1.0000+0.0000j,  1.0000+0.0000j],
        [ 0.5403+0.8415j,  0.9999+0.0100j],
        [-0.4161+0.9093j,  0.9998+0.0200j]])
```

从方程 (24) 可以看出：

* `freqs_cis` 有 3 个分量，对应序列长度为 3。
* 每个组件由两个复数组成。

$$\begin{align*}
[ \cos(0 \cdot \theta_1) + \sin(0 \cdot \theta_1)j, & \cos(0 \cdot \theta_2) + \sin(0 \cdot \theta_2)j ] \\
[ \cos(1 \cdot \theta_1) + \sin(1 \cdot \theta_1)j, & \cos(1 \cdot \theta_2) + \sin(1 \cdot \theta_2)j ] \\
[ \cos(2 \cdot \theta_1) + \sin(2 \cdot \theta_1)j, & \cos(2 \cdot \theta_2) + \sin(2 \cdot \theta_2)j ]
\end{align*} \tag{24}$$
其中，$\theta_1 = 10000^{-2(1-1)/4} = 1, \quad \theta_2 = 10000^{-2(2-1)/4} = 0.01$

为什么需要预先计算这个表格，你将在下面的 `apply_rotary_emb` 函数中看到。

#### apply_rotary_emb 函数

该函数是将 RoPE 应用于输入张量。它首先将 $x_q$ 按每组两个分量进行重塑，然后将其转换为复数形式$x_{q\_}$。 $x_{q\_}$ 随后通过复数乘法与 `freqs_cis` 相乘。

```python
def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    对输入张量应用旋转嵌入，使用给定的频率张量。

	此函数利用提供的频率张量 'freqs_cis'，对给定的查询 'xq' 和键 'xk' 张量应用旋转嵌入。
	输入张量被重塑为复数形式，频率张量则被重塑以实现广播兼容性。
	最终生成的张量包含旋转嵌入，并以实数张量形式返回。

    Args:
        xq (torch.Tensor): Query tensor to apply rotary embeddings.
        xk (torch.Tensor): Key tensor to apply rotary embeddings.
        freqs_cis (torch.Tensor): Precomputed frequency tensor for complex exponentials.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.
    """
    # 将 xq 和 xk 重塑并转换为复数
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    # 应用旋转操作，然后将结果转换回实数。
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)
```

为了解释使用复数乘法的原因，让我们回顾之前的高维旋转矩阵（方程（22）），该旋转矩阵被分解为 $d/2$ 组，每组仅包含两个分量。这里，我们以 $d=4$ 为例。

$$
R_{\theta,m}^d = \left(
\begin{array}{cc|cc}
\fcolorbox{red}{white}{$\begin{array}{cc}
\cos(m\theta_1) & -\sin(m\theta_1) \\
\sin(m\theta_1) & \cos(m\theta_1)
\end{array}$} & \begin{array}{cc}0 & 0 \\ 0 & 0\end{array} \\
\hline
\begin{array}{cc}0 & 0 \\ 0 & 0\end{array} & \fcolorbox{green}{white}{$\begin{array}{cc}
\cos(m\theta_2) & -\sin(m\theta_2) \\
\sin(m\theta_2) & \cos(m\theta_2)
\end{array}$}
\end{array}
\right)
\quad
q = \left(
\begin{array}{c}
\fcolorbox{red}{white}{$\begin{array}{c} q^{(1)} \\ q^{(2)} \end{array}$} \\[3ex]
\fcolorbox{green}{white}{$\begin{array}{c} q^{(3)} \\ q^{(4)} \end{array}$}
\end{array}
\right)   \tag{25}
$$

在四维情况下，`apply_rotary_emb` 的计算方法如下：红色框之间执行复数乘法运算，绿色框之间同样执行复数乘法运算。旋转矩阵的复数形式由预先计算的 `freqs_cis` 提供，而 `q` 的复数形式则由 `xq_` 提供。

为什么复数的乘法运算有效？

![|600](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F8b5d1d32-1dc5-4c77-8c9d-355ff38b9b60_800x458.png)

如图所示，这是因为红框之间的乘法结果由（不失一般性，这里以红框为例）方程（26）给出：
$$\left( q^{(1)} \cos(m\theta_1) - q^{(2)} \sin(m\theta_1), \, q^{(1)} \sin(m\theta_1) + q^{(2)} \cos(m\theta_1) \right) \tag{26}$$
方程 (26) 的复数形式是通过将 `xq_` 和预先计算的 `freqs_cis` 分别提供的两个复数相乘而得到的。
$$\left( q^{(1)} + q^{(2)} i \right) \cdot \left( \cos(m\theta_1) + \sin(m\theta_1) i \right) \tag{27}$$
同样，绿色方框之间的复数乘法运算生成了第一个标记的 $q_m$ 的最后两个维度。结合方程（26），它形成了第一个标记的查询嵌入 $q_m$，如方程（28）所示：
$$\begin{align*} \mathbf{q}_m = & \left( q^{(1)} \cos(m\theta_1) - q^{(2)} \sin(m\theta_1), \, q^{(1)} \sin(m\theta_1) + q^{(2)} \cos(m\theta_1), \right. \\ & \left. q^{(3)} \cos(m\theta_2) - q^{(4)} \sin(m\theta_2), \, q^{(3)} \sin(m\theta_2) + q^{(4)} \cos(m\theta_2) \right) \end{align*} \tag{28}
$$
可以看出，`precompute_freqs_cis` 和 `apply_rotary_emb` 通过复数运算及复数与实数间的转换，巧妙地实现了高维 RoPE 位置编码。


#### Attention:: forward

然后，在 `Attention` 类的前向函数中使用 `apply_rotary_emb` 来计算 RoPE。

```python
class Attention(nn.Module):
    """Multi-head attention module."""

    ...
    ...

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        """
        Forward pass of the attention module.

        Args:
            x (torch.Tensor): Input tensor.
            start_pos (int): Starting position for caching.
            freqs_cis (torch.Tensor): Precomputed frequency tensor.
            mask (torch.Tensor, optional): Attention mask tensor.

        Returns:
            torch.Tensor: Output tensor after attention.

        """
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        # Calculate RoPE
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)
        ...
        ...
```

### 结论

值得一提的是，RoPE 是在 2021 年提出的，当时其应用并不广泛。例如，Transformer 使用了正弦位置编码，后来代表性模型 BERT 采用了可学习的位置嵌入。

当基于 RoPE 的大型模型（如 Llama）被广泛使用时，人们发现 RoPE 可以通过旋转矩阵将位置编码外推到预训练长度之外。这提高了模型的泛化能力和鲁棒性，这是以往的位置编码方法无法实现的。因此，RoPE 得到了广泛应用。

总的来说，RoPE 巧妙地将向量旋转的思想应用于大语言模型的位置编码中，并通过复数运算实现。这是数学思维在人工智能领域熠熠生辉的典范。

