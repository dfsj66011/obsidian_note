
Paper：[YaRN: Efficient Context Window Extension of Large Language Models](https://arxiv.org/pdf/2309.00071)


RoPE 通过旋转查询 $Q$ 和键 $K$ 向量来注入位置信息。对于维度为 $|D|$ 的向量，其在位置 $m$ 的旋转操作可以表示为：
$$f(\mathbf{x}_m, m) = R_m \mathbf{x}_m$$
其中：$$
R_m(x_j) = x_j \odot \cos(m\theta) + (x_j \text{ with first half swapped with second half}) \odot \sin(m\theta)$$
$R_m$ 是一个旋转矩阵，其旋转角度由位置 $m$ 和频率 $\theta_d$ 决定。频率 $\theta_d$ 的计算方式为：
$$
\theta_d = b^{-2d/|D|}
$$
- $b$ 是一个预设的基数（通常为 10000）。
- $|D|$ 是头的维度。
- $d$ 是成对的维度索引，从 $0$ 到 $|D|/2 - 1$。

后续方法都旨在通过修改位置 $m$ 或频率 $\theta_d$ 来实现上下文扩展。论文中将这些修改统一为以下形式：
$$
f(\mathbf{x}_m, m, \theta_d) \rightarrow f(\mathbf{x}_{m}, g(m), h(\theta_d)) \quad \cdots \quad (\text{Eq. 12})
$$
其中 $g(m)$ 是对位置索引的变换，而 $h(\theta_d)$ 是对频率的变换。

---

### 0、PI (Position Interpolation) 位置插值法回顾

PI 方法的核心是 *线性缩放位置索引*。如果训练长度是 $L$，目标长度是 $L'$，那么对于位置 $m$，新的位置索引是 $m' = m \cdot \frac{L}{L'}$。这相当于将所有位置的旋转速度都 *均匀地减慢* 了 $\frac{L'}{L}$ 倍。

**PI 代码实现：**

```python
import torch
import math
import matplotlib.pyplot as plt
import numpy as np

def rotate_half(x):
    x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)

def apply_rope(x, freqs):
    # x: (batch_size, seq_len, head_dim)
    # freqs: (seq_len, head_dim)
    cos = freqs.cos()
    sin = freqs.sin()
    return x * cos + rotate_half(x) * sin

def precompute_freqs_cis(dim: int, end: int, base: float = 10000.0):
    # freqs: (dim // 2)
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    # t: (end, 1)
    t = torch.arange(end, device=freqs.device)
    # freqs: (end, dim // 2)
    freqs = torch.outer(t, freqs).float()
    # freqs: (end, dim)
    freqs_cis = torch.cat((freqs, freqs), dim=-1) # Duplicate for cos and sin
    return freqs_cis

def rope_pi(x, seq_len, head_dim, train_len, target_len, base=10000.0):
    # PI: Scale position indices
    scale_factor = target_len / train_len
    scaled_seq_len = int(seq_len / scale_factor) # Simulate original length for freq computation

    # Precompute freqs for the *scaled* sequence length
    # The key is that the 't' in precompute_freqs_cis is effectively scaled
    # by passing a smaller 'end' value.
    freqs = precompute_freqs_cis(head_dim, scaled_seq_len, base)

    # If x is longer than scaled_seq_len, we need to interpolate freqs
    if seq_len > scaled_seq_len:
		# 这是一个简化的插值方法。实际应用中，您需要为目标长度生成频率，然后在应用 RoPE 时缩放位置索引。
		# 为了清晰起见，让我们为实际的目标长度生成频率并缩放位置。
		# 这是更常见的PI实现方式：
		
		# freqs_for_target_len = precompute_freqs_cis(head_dim, target_len, base)\
		# scaled_positions = torch.arange(seq_len) * (train_len / target_len)
		# freqs = freqs_for_target_len[scaled_positions.long()]
		
		# 通过缩放 precompute_freqs_cis 中的 't' 来实现 PI 的更直接方法
		# 这实际上会均匀地减慢所有频率的旋转速度。
        
        t_scaled = torch.arange(seq_len, device=x.device) * (train_len / target_len)
        freqs_pi = 1.0 / (base ** (torch.arange(0, head_dim, 2)[: (head_dim // 2)].float() / head_dim))
        freqs_pi = torch.outer(t_scaled, freqs_pi).float()
        freqs_pi = torch.cat((freqs_pi, freqs_pi), dim=-1)
        
        return apply_rope(x, freqs_pi)
    else:
        # If seq_len is within original train_len, use original freqs
        freqs = precompute_freqs_cis(head_dim, seq_len, base)
        return apply_rope(x, freqs)

# Example Usage for PI
head_dim = 128
train_len = 1024
target_len = 4096
seq_len = 4096 # Current sequence length for inference

# Dummy input tensor (batch_size, seq_len, head_dim)
x_dummy = torch.randn(1, seq_len, head_dim)

# Apply PI RoPE
x_pi_out = rope_pi(x_dummy, seq_len, head_dim, train_len, target_len)
print(f"PI RoPE output shape: {x_pi_out.shape}")
```

**PI 的目的与效果分析：**
- **目的：** 缓解位置编码在长序列上的外推问题，通过线性缩放位置索引，使得模型在长序列上看到的“相对位置”与训练时相似。
- **效果：**
    - **优点：** 实现简单，无需额外训练，对困惑度有一定改善。
    - **缺点：** 均匀缩放所有频率，可能导致高频信息（对局部关系敏感）被过度压缩，从而损失模型对局部细节的感知能力。这在高频分量（快速旋转的维度）上表现尤为明显，因为它们在训练长度内可能已经完成了多个周期，具有较好的外推能力，但被强制减速。

---

### 1. "NTK-aware" 插值法

该方法旨在解决位置插值（PI）中存在的高频信息损失问题。PI 方法通过均匀缩放所有维度，损害了模型分辨邻近 Token 的能力。NTK-aware 方法则通过非均匀缩放，对高频维度影响较小，对低频维度影响较大。

"NTK-aware" 插值法通过修改 RoPE 的基数 $b$ 来实现。
$$\begin{align}
g(m) &= m \\[1.2ex]
h(\theta_d) &= {b'}^{-2d/|D|}
\end{align}
$$
其中，新的基数 $b'$ 计算如下：
$$
b' = b \cdot s^{\frac{|D|}{|D|-2}}
$$


- $g(m) = m$ 表示位置索引 $m$ 保持不变。
- 核心改动在于 $h(\theta_d)$，它使用了新的基数 $b'$。
- $b$ 是原始基数（如 10000）。
- $s$ 是尺度因子，即 $s = L' / L$（目标长度 / 原始训练长度）。
- $|D|$ 是注意力头的维度。

*核心思想*：整个基数 $b$ 被一个**与维度无关**的常数 $s^{\frac{|D|}{|D|-2}}$ 所缩放。这个新的基数 $b'$ 被用于计算所有维度的频率。这导致所有频率都受到了影响，但由于指数关系，它实现了对低频维度的更大程度压缩。

*缺点*：这种方法并非纯粹的插值，某些维度会被轻微外推，导致在微调时效果不如 PI。此外，实际的上下文扩展倍数通常需要设置比理论值更高的 $s$。

#### 示例演示

**设定参数：**

-   原始基数 $b = 10000$
-   注意力头维度 $|D| = 128$
-   原始训练长度 $L = 1024$
-   目标扩展长度 $L' = 4096$
-   尺度因子 $s = L' / L = 4096 / 1024 = 4$

首先，计算 $b'$ 的缩放因子：
$$
\text{缩放因子} = s^{\frac{|D|}{|D|-2}} = 4^{\frac{128}{128-2}} = 4^{\frac{128}{126}} \approx 4^{1.015873} \approx 4.0632
$$
然后计算新的基数 $b'$：
$$
b' = b \cdot \text{缩放因子} = 10000 \cdot 4.0632 = 40632
$$

在这个例子中，原始基数 $b=10000$ 变成了 $b' \approx 40632$。*新的基数 $b'$ 比原始基数 $b$ 大了约 4.06 倍。*

**频率变化分析：**

频率 $\theta_d$ 决定了旋转的速度，频率越高，旋转越快。选择以下维度 $d$（注意 $d$ 是成对维度索引，从 $0$ 到 $|D|/2 - 1 = 63$）：

-   **$d=0$**: 理论上的最高频（作为参考）
-   **$d=5$**: 较高频维度
-   **$d=31$**: 中频维度
-   **$d=50$**: 较低频维度
-   **$d=63$**: 最低频维度

**1. 原始频率 $\theta_d = b^{-2d/|D|}$：**

-   $d=0$: $\theta_0 = 10000^{-0/128} = 10000^0 = 1$
-   $d=5$: $\theta_5 = 10000^{-10/128} \approx 10000^{-0.078125} \approx 0.591$
-   $d=31$: $\theta_{31} = 10000^{-62/128} \approx 10000^{-0.484375} \approx 0.0109$
-   $d=50$: $\theta_{50} = 10000^{-100/128} = 10000^{-0.78125} \approx 0.00075$
-   $d=63$: $\theta_{63} = 10000^{-126/128} \approx 10000^{-0.984375} \approx 0.00014$

**2. "NTK-aware" 新频率 $\theta_d' = {b'}^{-2d/|D|}$：**

-   $d=0$: $\theta_0' = 40632^{-0/128} = 40632^0 = 1$
-   $d=5$: $\theta_5' = 40632^{-10/128} \approx 40632^{-0.078125} \approx 0.580$
-   $d=31$: $\theta_{31}' = 40632^{-62/128} \approx 40632^{-0.484375} \approx 0.0054$
-   $d=50$: $\theta_{50}' = 40632^{-100/128} \approx 40632^{-0.78125} \approx 0.00025$
-   $d=63$: $\theta_{63}' = 40632^{-126/128} \approx 40632^{-0.984375} \approx 0.000035$

**频率变化对比：**

| 维度 $d$   | 原始频率 $\theta_d$ | 新频率 $\theta_d'$ | 频率变化倍数 ($\theta_d' / \theta_d$)             |
| :------- | :-------------- | :-------------- | :------------------------------------------ |
| 0 (最高频)  | 1               | 1               | $1 / 1 = 1$ (不变)                            |
| 5 (较高频)  | 0.591           | 0.580           | $0.580 / 0.591 \approx 0.981$ (轻微减小)        |
| 31 (中频)  | 0.0109          | 0.0054          | $0.0054 / 0.0109 \approx 0.495$ (约减半)       |
| 50 (较低频) | 0.00075         | 0.00025         | $0.00075 / 0.00025 =0.3$ (约减三分之二)           |
| 63 (最低频) | 0.00014         | 0.000035        | $0.000035 / 0.00014 \approx 0.250$ (约减四分之三) |
对于 $d=0,5$ 这样的较高频维度，频率仅**轻微减小**（约 1.9%）。这意味着这些维度在扩展后的序列中，其旋转速度几乎保持不变。由于这些高频维度在训练长度内已经完成了足够多的旋转周期，它们本身就具有较强的外推能力，能够有效编码局部、细微的位置关系。因此，保持其原始或接近原始的旋转速度，有助于模型在长序列中继续利用这些高频信息进行“外推”。

对于低频维度 ($d=50, 63$)：频率**显著减小**（约 66~75%）。这意味着这些最慢的旋转维度，其旋转速度被大幅减慢。低频维度在训练长度内可能只完成了很少的周期，甚至不足一个周期，因此它们的外推能力很差。通过大幅减慢其旋转速度，使得在更长的序列中，这些维度也能像在训练长度内一样，完成相似的旋转角度范围，从而实现“低频插值”，让模型在长距离上也能有效编码位置信息。

中频维度 ($d=31$)：频率减小的幅度介于高频和低频之间（约 50%）。这表明 NTK-aware 方法对所有维度都进行了调整，但调整的程度是**非均匀的**，越是低频的维度，其频率被压缩得越厉害。

这种非均匀的频率缩放，正是 "NTK-aware" 插值法实现“高频外推、低频插值”的关键机制。它避免了 PI 方法对所有频率一刀切的均匀缩放，从而更好地保留了模型在长序列中对局部和全局位置信息的感知能力。

#### 代码实现

```python
import torch
import math

# --- 辅助函数 (所有方法共用) ---
def rotate_half(x):
    x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)

def apply_rope(x, freqs_cis):
    # x: (batch_size, seq_len, head_dim)
    # freqs_cis: (seq_len, head_dim)
    cos = freqs_cis.cos()
    sin = freqs_cis.sin()
    return x * cos + rotate_half(x) * sin
# ------------------------------------

def precompute_freqs_cis_ntk_aware(dim: int, end: int, train_len: int, base: float = 10000.0, scale: float = 1.0):
    """
    严格按照论文公式 2.1 实现 "NTK-aware" 插值
    b' = b * s^(|D|/(|D|-2))
    """
    # 计算新的基数 b'
    # s 是尺度因子 scale
    # |D| 是维度 dim
    power = dim / (dim - 2)
    base_scaled = base * (scale ** power)
    
    # 使用新的基数 b' 计算频率
    # freqs_i = (b')^(-2i/|D|)
    freqs = 1.0 / (base_scaled ** (torch.arange(0, dim, 2).float() / dim))
    
    # 计算所有位置的频率
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.cat((freqs, freqs), dim=-1)
    
    return freqs_cis

def rope_ntk_aware(x, seq_len, head_dim, train_len, target_len, base=10000.0):
    scale = target_len / train_len
    freqs = precompute_freqs_cis_ntk_aware(head_dim, seq_len, train_len, base, scale)
    return apply_rope(x, freqs)

# --- 示例 ---
head_dim = 128
train_len = 2048
target_len = 8192
seq_len = 8192
x_dummy = torch.randn(1, seq_len, head_dim)

x_ntk_aware_out = rope_ntk_aware(x_dummy, seq_len, head_dim, train_len, target_len)
print(f"'NTK-aware' RoPE 输出形状: {x_ntk_aware_out.shape}")
```

---

### 2. "NTK-by-parts" 插值法

该方法旨在解决 "NTK-aware" 存在的问题，即所有维度都被同一种策略处理。它认为，高频维度（波长远小于上下文长度）已经具备良好的相对位置编码能力，不应被插值；而低频维度（波长接近或超过上下文长度）则需要被插值以避免外推。

"NTK-by-parts" 方法通过一个斜坡函数 $\gamma(r)$ 来混合两种策略：不插值和线性插值（PI）。
$$
\begin{align}
g(m) &= m \\[1.2ex]
h(\theta_d) &= \Big(1-\gamma\big(r(d)\big)\Big) \frac{\theta_d}{s} + \gamma\big(r(d)\big) \theta_d
\end{align}
$$
其中，斜坡函数 $\gamma(r)$ 定义为：
$$
\gamma(r) = 
\begin{cases}
    0, &\text{if } r < \alpha\\
    1, &\text{if } r > \beta\\
    \dfrac{r - \alpha}{\beta - \alpha}, &\text{otherwise}
\end{cases}
$$
而 $r(d)$ 是原始上下文长度 $L$ 与波长 $\lambda_d$ 的比率，正比于 $L * θ_d$。

- $g(m) = m$ 表示位置索引不变。
- $h(\theta_d)$ 是对频率 $\theta_d$ 的修改。
- $\theta_d / s$ 是 *线性插值（PI）* 后的频率。
- $\theta_d$ 是**原始**频率。
- $r=\frac{L}{\lambda}$，原始上下文长度 $L$ 与波长 $\lambda$ 的比值
 
*核心思想*：通过 $\gamma(r)$ 函数，将频率在“完全插值”和“完全不插值”之间进行平滑过渡。

* 当 $r(d) < \alpha$（低频），$\gamma(r)=0$，新频率为 $\theta_d / s$，即 *完全进行 PI 插值*。
* 当 $r(d) > \beta$（高频），$\gamma(r)=1$，新频率为 $\theta_d$，即 *完全不插值，保留原始频率*。
* 当 $\alpha \le r(d) \le \beta$（中频），在两种策略之间进行线性过渡。

$\alpha, \beta$ 是超参数，需要针对模型进行调整（论文建议 Llama 模型使用 $\alpha=1, \beta=32$）。

#### 示例演示

**设定参数：**

-   原始基数 $b = 10000$
-   注意力头维度 $|D| = 128$
-   原始训练长度 $L = 1024$
-   目标扩展长度 $L' = 4096$
-   尺度因子 $s = L' / L = 4096 / 1024 = 4$
-   超参数 $\alpha = 1.0$
-   超参数 $\beta = 32.0$


**计算 $r(d)$ 和 $\gamma(r)$：** 首先，我们需要计算每个维度 $d$ 的原始频率 $\theta_d$，然后计算 $r(d)$（我们使用 $L \cdot \theta_d$ 作为 $r(d)$ 的代理，因为它们是正比关系，且论文中也提到 $r(d) = L / \lambda_d = L / (2\pi b'^{2d/|D|})$，而 $\theta_d = b^{-2d/|D|}$，所以 $r(d)$ 正比于 $L \cdot \theta_d$）。

**1. 原始频率 $\theta_d = b^{-2d/|D|}$：**

-   $d=0$: $\theta_0 = 10000^{-0/128} = 1$
-   $d=5$: $\theta_5 = 10000^{-10/128} \approx 0.591$
-   $d=31$: $\theta_{31} = 10000^{-62/128} \approx 0.0109$
-   $d=50$: $\theta_{50} = 10000^{-100/128} \approx 0.00078$
-   $d=63$: $\theta_{63} = 10000^{-126/128} \approx 0.00014$

**2. 计算 $r(d)$ (代理值 $L \cdot \theta_d$)：**

-   $d=0$: $r(0) = 1024 \cdot 1 = 1024$
-   $d=5$: $r(5) = 1024 \cdot 0.591 \approx 605.2$
-   $d=31$: $r(31) = 1024 \cdot 0.0109 \approx 11.16$
-   $d=50$: $r(50) = 1024 \cdot 0.00078 \approx 0.799$
-   $d=63$: $r(63) = 1024 \cdot 0.00014 \approx 0.143$

**3. 计算 $\gamma(r)$：**

-   $d=0$ ($r=1024$): $r > \beta=32 \implies \gamma(1024) = 1$
-   $d=5$ ($r=605.2$): $r > \beta=32 \implies \gamma(605.2) = 1$
-   $d=31$ ($r=11.16$): $\alpha=1 \le r \le \beta=32 \implies \gamma(11.16) = \frac{11.16 - 1}{32 - 1} = \frac{10.16}{31} \approx 0.328$
-   $d=50$ ($r=0.799$): $r < \alpha=1 \implies \gamma(0.799) = 0$
-   $d=63$ ($r=0.143$): $r < \alpha=1 \implies \gamma(0.143) = 0$

---

**计算 "NTK-by-parts" 新频率 $h(\theta_d)$：**

新频率公式：$h(\theta_d) = (1-\gamma) \frac{\theta_d}{s} + \gamma \theta_d$

-   **$d=0$ (高频，$\gamma=1$)：**
    $h(\theta_0) = (1-1) \frac{\theta_0}{4} + 1 \cdot \theta_0 = 0 \cdot \frac{1}{4} + 1 \cdot 1 = 1$
    **频率变化倍数：$1 / 1 = 1$ (不变)**

-   **$d=5$ (高频，$\gamma=1$)：**
    $h(\theta_5) = (1-1) \frac{\theta_5}{4} + 1 \cdot \theta_5 = 0 \cdot \frac{0.591}{4} + 1 \cdot 0.591 = 0.591$
    **频率变化倍数：$0.591 / 0.591 = 1$ (不变)**

-   **$d=31$ (中频，$\gamma \approx 0.328$)：**
    $h(\theta_{31}) = (1-0.328) \frac{0.0109}{4} + 0.328 \cdot 0.0109$
    $= 0.672 \cdot 0.002725 + 0.003575$
    $= 0.00183 + 0.003575 = 0.005405$
    **频率变化倍数：$0.005405 / 0.0109 \approx 0.496$ (约减半)**

-   **$d=50$ (低频，$\gamma=0$)：**
    $h(\theta_{50}) = (1-0) \frac{\theta_{50}}{4} + 0 \cdot \theta_{50} = 1 \cdot \frac{0.00078}{4} + 0 = 0.000195$
    **频率变化倍数：$0.000195 / 0.00078 = 0.25$ (减小四分之三)**

-   **$d=63$ (低频，$\gamma=0$)：**
    $h(\theta_{63}) = (1-0) \frac{\theta_{63}}{4} + 0 \cdot \theta_{63} = 1 \cdot \frac{0.00014}{4} + 0 = 0.000035$
    **频率变化倍数：$0.000035 / 0.00014 = 0.25$ (减小四分之三)**

---

**频率变化对比总结：**

| 维度 $d$ | $r(d)$ (代理值) | $\gamma(r)$ | 原始频率 $\theta_d$ | 新频率 $h(\theta_d)$ | 频率变化倍数 ($h(\theta_d) / \theta_d$) | 策略 |
| :------- | :-------------- | :---------- | :------------------ | :------------------- | :------------------------------------ | :--- |
| 0        | 1024            | 1           | 1                   | 1                    | $1$ (不变)                            | 不插值 |
| 5        | 605.2           | 1           | 0.591               | 0.591                | $1$ (不变)                            | 不插值 |
| 31       | 11.16           | 0.328       | 0.0109              | 0.005405             | $0.496$ (线性过渡)                    | 混合 |
| 50       | 0.799           | 0           | 0.00078             | 0.000195             | $0.25$ (完全 PI 插值)                 | PI 插值 |
| 63       | 0.143           | 0           | 0.00014             | 0.000035             | $0.25$ (完全 PI 插值)                 | PI 插值 |

**与 "NTK-aware" 的主要区别：**

-   **"NTK-aware"**：通过修改基数 $b$ 来实现**所有维度**的非均匀缩放，但即使是高频维度，其频率也会有轻微变化（除非 $d=0$）。
-   **"NTK-by-parts"**：通过 $\gamma(r)$ 函数，明确地将高频维度**完全排除在插值之外**，使其频率保持不变。这使得高频信息的保留更为彻底和精确。对于低频维度，它则采用与 PI 相同的均匀缩放。

这种分段策略使得 "NTK-by-parts" 能够更精细地控制不同频率维度的行为，从而在长上下文任务中取得更好的性能。

#### 代码实现

```python
def precompute_freqs_cis_ntk_by_parts(dim: int, end: int, train_len: int, base: float = 10000.0, scale: float = 1.0, alpha: float = 1.0, beta: float = 32.0):
    """
    严格按照论文公式 2.2 实现 "NTK-by-parts" 插值
    h(θ_d) = (1-γ) * (θ_d/s) + γ * θ_d
    """
    # 计算原始频率 θ_d
    freqs_original = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    
    # 计算 r(d)，即 L / λ_d。这正比于 L * θ_d。
    # 我们使用 num_rotations = L * θ_d 作为 r(d) 的代理。
    num_rotations = train_len * freqs_original
    
    # 计算斜坡函数 γ(r) 的值
    gamma_vals = torch.zeros_like(num_rotations)
    
    # Case 1: r > β (高频) -> γ = 1
    mask_high_freq = num_rotations > beta
    gamma_vals[mask_high_freq] = 1.0
    
    # Case 2: r < α (低频) -> γ = 0 (已经是默认值)
    
    # Case 3: α <= r <= β (中频) -> γ = (r - α) / (β - α)
    mask_transition = (num_rotations >= alpha) & (num_rotations <= beta)
    gamma_vals[mask_transition] = (num_rotations[mask_transition] - alpha) / (beta - alpha)
    
    # 计算 PI 插值后的频率 θ_d / s
    freqs_pi = freqs_original / scale
    
    # 根据公式计算最终频率 h(θ_d)
    freqs_new = (1.0 - gamma_vals) * freqs_pi + gamma_vals * freqs_original
    
    # 计算所有位置的频率
    t = torch.arange(end, device=freqs_new.device)
    freqs = torch.outer(t, freqs_new).float()
    freqs_cis = torch.cat((freqs, freqs), dim=-1)
    
    return freqs_cis

def rope_ntk_by_parts(x, seq_len, head_dim, train_len, target_len, base=10000.0, alpha=1.0, beta=32.0):
    scale = target_len / train_len
    freqs = precompute_freqs_cis_ntk_by_parts(head_dim, seq_len, train_len, base, scale, alpha, beta)
    return apply_rope(x, freqs)

# --- 示例 ---
x_ntk_by_parts_out = rope_ntk_by_parts(x_dummy, seq_len, head_dim, train_len, target_len)
print(f"'NTK-by-parts' RoPE 输出形状: {x_ntk_by_parts_out.shape}")
```

---

### 3. "Dynamic NTK" 动态缩放

这并非一种新的频率计算方法，而是一种在推理时应用上述方法的**策略**。它解决了在处理短序列时，固定尺度因子 $s$ 会导致不必要性能损失的问题。

在每个前向传播中，动态地更新尺度因子 $s$：
$$
s = \max(1, l' / L)
$$

- $L$ 是原始训练长度。
- $l'$ 是**当前**推理序列的长度。

*核心思想*：当当前序列长度 $l'$ 不超过训练长度 $L$ 时，$s=1$，不进行任何插值，使用原始 RoPE，避免性能损失。当 $l'$ 超过 $L$ 时，$s = l'/L$，插值被激活，且尺度因子恰好对应当前所需的扩展比例。当这个动态策略与 "NTK-aware" 插值法结合时，被称为 "Dynamic NTK"。

#### 代码实现

```python
def rope_dynamic_ntk(x, seq_len, head_dim, train_len, base=10000.0):
    """
    实现 Dynamic NTK，即在推理时动态计算 scale，并应用 "NTK-aware" 插值
    """
    # 动态计算 scale
    scale = max(1.0, seq_len / train_len)
    
    # 调用 "NTK-aware" 的频率计算函数，传入动态 scale
    freqs = precompute_freqs_cis_ntk_aware(head_dim, seq_len, train_len, base, scale)
    
    return apply_rope(x, freqs)

# --- 示例 ---
# 短序列
x_dummy_short = torch.randn(1, 1024, head_dim)
x_dynamic_short_out = rope_dynamic_ntk(x_dummy_short, 1024, head_dim, train_len)
print(f"Dynamic NTK (短序列) 输出形状: {x_dynamic_short_out.shape}")

# 长序列
x_dummy_long = torch.randn(1, 8192, head_dim)
x_dynamic_long_out = rope_dynamic_ntk(x_dummy_long, 8192, head_dim, train_len)
print(f"Dynamic NTK (长序列) 输出形状: {x_dynamic_long_out.shape}")
```

---

### 4. YaRN (Yet another RoPE N-gram)

YaRN 是论文提出的最终方法，它将 "NTK-by-parts" 插值法、动态缩放策略与 *注意力温度缩放* 相结合，以达到最佳效果。

YaRN 在 "NTK-by-parts" 的基础上，引入了对注意力分数的温度 $t$ 缩放：
$$
\text{Attention}(\mathbf{q}_m, \mathbf{K}) = \text{softmax}\left(\frac{\mathbf{q}_m \mathbf{K}^T}{t\sqrt{|D|}}\right)
$$
为了方便实现且不修改注意力代码，这等价于将 $\mathbf{q}$ 和 $\mathbf{k}$ 向量在应用 RoPE 之后乘以一个因子 $\sqrt{1/t}$。论文给出了一个经验公式来计算这个因子：
$$
\sqrt{1/t} = 0.1 \ln(s) + 1
$$

*核心思想*：YaRN 认为，在扩展上下文时，注意力分布的熵会发生变化。引入温度 $t$ 可以调整 softmax 的尖锐程度，使其更接近原始模型的注意力分布特性。

*实现技巧*：不直接修改注意力中的除法，而是将 Q 和 K 向量乘以 $\sqrt{1/t}$。由于 RoPE 是线性操作，这等价于在 RoPE 操作之后对结果进行缩放。

#### 代码实现

```python
def rope_yarn(x, seq_len, head_dim, train_len, base=10000.0, alpha=1.0, beta=32.0):
    """
    实现完整的 YaRN 方法
    结合了 NTK-by-parts, Dynamic Scaling, 和 Attention Temperature Scaling
    """
    # 1. Dynamic Scaling: 动态计算 scale
    scale = max(1.0, seq_len / train_len)
    
    # 2. NTK-by-parts: 使用动态 scale 计算频率
    freqs = precompute_freqs_cis_ntk_by_parts(head_dim, seq_len, train_len, base, scale, alpha, beta)
    
    # 应用 RoPE
    x_rotated = apply_rope(x, freqs)
    
    # 3. Attention Temperature Scaling: 计算并应用温度因子
    if scale > 1.0:
        # 论文公式: sqrt(1/t) = 0.1 * ln(s) + 1
        temp_scale_factor = 0.1 * math.log(scale) + 1.0
        x_rotated *= temp_scale_factor
        
    return x_rotated

# --- 示例 ---
x_yarn_out = rope_yarn(x_dummy, seq_len, head_dim, train_len)
print(f"YaRN RoPE 输出形状: {x_yarn_out.shape}")
```

---

### 5、总结与对比

| 方法 | 核心思想 | 优点 | 缺点 (据论文) |
| :--- | :--- | :--- | :--- |
| **"NTK-aware"** | 统一缩放 RoPE 的基数 `b`，实现非均匀频率压缩。 | 优于 PI，更好地保留高频信息。 | 存在轻微外推，微调效果不佳；尺度因子 `s` 不精确。 |
| **"NTK-by-parts"** | 根据波长分段处理：高频保留，低频插值，中频过渡。 | 避免了不必要的插值和外推，在微调和非微调场景下均表现出色。 | 引入了需要调整的超参数 $\alpha, \beta$。 |
| **Dynamic Scaling** | 推理时根据当前序列长度动态调整尺度因子 `s`。 | 避免了对短序列的性能损害，使模型能平滑地处理超长上下文。 | 是一种策略，需与具体插值法结合；对 KV Cache 实现有要求。 |
| **YaRN** | 结合 "NTK-by-parts"、动态缩放和注意力温度缩放。 | 综合了所有方法的优点，在微调和非微调场景下均达到最佳性能。 | 是一个复合方法，但实现上只需修改 RoPE 生成部分，兼容性好。 |

### 6、后续改进方向

该论文提出的 YaRN 方法本身就是对之前方法的一系列改进的集大成者，代表了当前（论文发表时）在 RoPE 上下文扩展领域的先进水平。未来的改进可能会围绕以下几点：

1.  **自适应温度**：YaRN 的温度缩放因子依赖于一个经验公式。更先进的方法可能会让模型学习或根据注意力分布的统计特性动态调整温度。
2.  **超越 RoPE**：虽然这些方法极大地扩展了 RoPE 的能力，但社区也在探索全新的位置编码方法，例如 ALiBi 等，它们可能从设计上就具备更好的外推性。
3.  **更精细的插值策略**：NTK-by-parts 的分段线性插值已经很有效，但可能存在更优的非线性插值函数来平滑过渡高频和低频部分。