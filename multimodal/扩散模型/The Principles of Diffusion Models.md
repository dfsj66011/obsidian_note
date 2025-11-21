
From Origins to Advances

------

## 摘要

本专著聚焦于塑造扩散模型发展的原理，追溯其起源，并展示不同的公式如何源于共同的数学思想。

扩散模型首先定义一个将数据逐步转化为噪声的 “*前向破坏过程*”。该前向过程通过定义一系列连续的中间分布，将数据分布与简单的噪声分布联系起来。扩散模型的核心目标是构建一个反向运行的逆过程，在恢复前向破坏过程所定义的相同中间分布的同时，将噪声转化为数据。

我们描述了三种互补的方式来形式化这一理念。受变分自编码器启发的 *变分视角*，将扩散视为逐步学习去除噪声的过程，通过解决一系列小的去噪目标，共同教导模型如何将噪声还原为数据。基于能量模型的 *基于分数视角*，学习数据分布演化的梯度，该梯度指示如何将样本推向更可能的区域。与归一化流相关的 *流视角*，将生成过程视为在学习的速度场下，沿着一条平滑路径将样本从噪声移动到数据的过程。

这些观点共享一个共同的核心：一个学习到的时间依赖速度场，其流动将简单先验转化为数据。基于此，采样相当于求解一个微分方程，该方程沿着连续的生成轨迹将噪声演化为数据。在此基础上，本专著讨论了可控生成的 *引导* 方法、高效采样的 *先进数值求解器*，以及受扩散启发的 *流映射模型*，这些模型学习沿此轨迹任意时间点之间的直接映射关系。

本书专为具备深度学习基础的读者撰写，旨在提供关于扩散模型的清晰、概念化且数学基础扎实的理解。书中阐明了理论基础，解释了不同公式背后的推理逻辑，并为这一快速发展领域的进一步学习和研究奠定了坚实基础。它既是研究人员的原理性参考指南，也是学习者易于入门的起点读物。

--------

# 一、深度生成模型

```text
What I cannot create, I do not understand.     —— Richard P. Feynman
```

深度生成模型（DGMs）是一种神经网络，它学习高维数据（如图像、文本、音频）的概率分布，从而能够生成与数据集相似的新样本。我们用 $p_{\phi}$ 表示模型分布，用 $p_{\mathrm{data}}$ 表示数据分布。给定一个有限的数据集，我们通过最小化衡量 $p_{\phi}$ 与 $p_{\mathrm{data}}$ 之间距离的损失函数来拟合 $\phi$。训练完成后，生成过程相当于运行模型的采样程序来抽取 $\mathrm{x} \sim p_{\phi}$（密度 $p_{\phi}(\mathrm{x})$ 是否可直接计算取决于模型类别）。模型质量的评判标准包括生成样本及其汇总统计量与 $p_{\mathrm{data}}$ 的匹配程度，以及特定任务或感知指标的表现。

本章将构建这些概念背后的数学和理论基础。我们在第 1.1 节中形式化该问题，在第 1.2 节中展示代表性模型类别，并在第 1.3 节中总结实用分类法。

## 1.1 什么是深度生成模型？

DGMs（深度生成模型）以从未知且复杂的真实数据分布 $p_{\mathrm{data}}$ 中抽取的大量现实样本（如图像、文本）作为输入，输出一个经过训练的神经网络，该网络参数化了一个近似分布 $p_{\phi}$。其目标具有双重性：

1. 逼真生成：生成与真实数据无法区分的新颖、逼真样本。
2. 可控生成：实现对生成过程的细粒度和可解释性控制。

本节介绍了DGMs 的基本概念和背后的动机，为深入探讨其数学框架和实际应用做好准备。

### 1.1.1 数学设置

我们假设可以访问一组有限的样本，这些样本是从一个潜在的、复杂的数据分布 $p_{\mathrm{data}}(\mathbf{x})$ 中独立同分布（i.i.d.）抽取的。

**DGM 的目标**   DGM 的主要目标是从有限数据集中学习一个易于处理的概率分布。这些数据点被视为从未知且复杂的真实分布 $p_{\mathrm{data}}(\mathbf{x})$ 中采样的观测值。由于 $p_{\mathrm{data}}(\mathbf{x})$ 的形式未知，我们无法直接从中抽取新样本。因此，核心挑战在于构建一个能充分逼近该分布的模型，从而生成新的、逼真的样本。

为此，DGM 采用深度神经网络对模型分布 $p_{\phi}(\mathrm{x})$ 进行参数化建模，其中 $\mathbf{\phi}$ 代表网络的可训练参数。其训练目标是找到最优参数 $\mathbf{\phi}^\ast$，使模型分布 $p_{\phi}(\mathrm{x})$ 与真实数据分布 $p_{\mathrm{data}}(\mathbf{x})$ 之间的散度最小化。从概念上讲，$$
p_{{\mathbf{\phi}}^\ast}(\mathrm{x}) \approx p_{\mathrm{data}}(\mathrm{x}).$$
当统计模型 $p_{{\mathbf{\phi}}^\ast}(\mathrm{x})$ 能够很好地逼近数据分布 $p_{\mathrm{data}}(\mathrm{x})$ 时，它就可以作为生成新样本和评估概率值的代理。这种模型 $p_{\phi}(\mathrm{x})$ 通常被称为 *生成模型*。

**DGM 的能力**   一旦获得了数据分布的代理模型  $p_{\phi}(\mathrm{x})$ ，我们就可以使用蒙特卡洛采样等方法从  $p_{\phi}(\mathrm{x})$  中生成任意数量的新数据点。此外，通过计算  $p_{\phi}(\mathrm{x'})$ 的值，我们可以确定任何给定数据样本 $\mathrm{x'}$ 的概率（或似然）。

![[dgm-learning.pdf]]

**图 1.1 DGM 的目标图示**  训练一个 DGM 本质上是在最小化模型分布 $p_{\phi}$ 与未知数据分布 $p_{\mathrm{data}}$ 之间的差异。由于无法直接获取 $p_{\mathrm{data}}$，必须通过从该分布中抽取的有限独立同分布（i.i.d.）样本 $\mathrm{x}_{i}$ 来高效估计这种差异。

**DGM 训练**   我们通过最小化差异度 $\mathcal{D}(p_{\mathrm{data}},p_{\phi})$ 来学习模型族 $\{p_{\phi}\}$ 的参数 $\phi$。$$\begin{align}
  \phi^\ast \in \arg\min_{\phi}\; \mathcal{D}(p_{\mathrm{data}},p_{\phi}).\tag{1.1.1}
\end{align}$$由于 $p_{\mathrm{data}}$ 未知，$\mathcal{D}$ 的实际选择必须允许从 i.i.d. 样本中对 $p_{\mathrm{data}}$ ​进行高效估计。在容量足够的情况下，$p_{\phi^\ast}$可以很好地逼近 $p_{\mathrm{data}}$。


**前向 KL 与最大似然估计（MLE）**   标准选择是（前向）Kullback-Leibler 散度：$$\begin{align*}
\mathcal{D}_{\mathrm{KL}} \big(p_{\mathrm{data}}\|p_{\phi}\big)
  := &\int p_{\mathrm{data}}(\mathrm{x})\,\log\frac{p_{\mathrm{data}}(\mathrm{x})}{p_{\phi}(\mathrm{x})}\,\mathrm{d}\mathrm{x} \\
  = &\mathbb{E}_{\mathrm{x}\sim p_{\mathrm{data}}} \big[\log p_{\mathrm{data}}(\mathrm{x})-\log p_{\phi}(\mathrm{x})\big].
\end{align*}$$这是不对称的，即，$$
\mathcal{D}_{\mathrm{KL}}(p_{\mathrm{data}} \| p_{\phi}) \neq \mathcal{D}_{\mathrm{KL}}(p_{\phi} \| p_{\mathrm{data}}).$$重要的是，最小化 $\mathcal{D}_{\mathrm{KL}}(p_{\mathrm{data}} \| p_{\phi})$ 会促进 *模式覆盖*：如果存在一个正测度集 $A$，其中 $p_{\mathrm{data}}(A)>0$，但对于 $\mathrm{x}\in A$ 有 $p_{\phi}(\mathrm{x})=0$，那么被积函数在 $A$ 上包含 $\log \big(p_{\mathrm{data}}(\mathrm{x})/0\big)=+\infty$，因此 $\mathcal{D}_{\mathrm{KL}}=+\infty$。因此，最小化前向 KL 迫使模型在数据有支持的地方分配概率。

虽然数据密度 $p_{\mathrm{data}}(\mathrm{x})$ 无法显式计算，但前向 KL 散度可以分解为：$$\begin{align*}
\mathcal{D}_{\mathrm{KL}} \big(p_{\mathrm{data}}\|p_{\phi}\big)
  &= \mathbb{E}_{\mathrm{x} \sim p_{\mathrm{data}}} \left[\log \frac{p_{\mathrm{data}}(\mathrm{x})}{p_{\phi}(\mathrm{x})}\right] \\[0.5em]
  &= -\,\mathbb{E}_{\mathrm{x} \sim p_{\mathrm{data}}} \big[\log p_{\phi}(\mathrm{x})\big] 
     + \mathcal H \big(p_{\mathrm{data}}\big),
\end{align*}$$这里的 $\mathcal H \big(p_{\mathrm{data}}\big):= -\,\mathbb{E}_{\mathrm{x} \sim p_{\mathrm{data}}} \big[\log p_{\mathrm{data}}(\mathrm{x})\big]$ 是数据分布的熵，相对于 $\phi$ 是常数。这一观察结果意味着以下等价关系：

> [!引理 1.1.1]
>
>最小化 KL $\Leftrightarrow$ MLE
> $$\begin{align}
>     \min_{\phi}\, \mathcal{D}_{\mathrm{KL}} \big(p_{\mathrm{data}} \,\|\, p_{\phi}\big)
>     \;\Longleftrightarrow\;
>     \max_{\phi}\, \mathbb{E}_{\mathrm{x}\sim p_{\mathrm{data}}} \big[\log p_{\phi}(\mathrm{x})\big].\tag{1.1.2}
> \end{align}$$

换句话说，最小化前向 KL 散度等同于执行最大似然估计（MLE）。

在实践中，我们用从独立同分布样本 $\{\mathrm{x}^{(i)}\}_{i=1}^N \sim p_{\mathrm{data}}$ 中得到的蒙特卡洛估计来替代总体期望，从而得到经验最大似然估计目标：$$\begin{align*}
  \hat{\mathcal{L}}_{\mathrm{MLE}}(\phi)
  := -\frac{1}{N}\sum_{i=1}^N \log p_{\phi} \big(\mathrm{x}^{(i)}\big),
\end{align*}$$通过小批量随机梯度进行优化；无需评估 $p_{\mathrm{data}}(\mathrm{x})$。


**Fisher 散度**    Fisher 散度是（基于分数的）扩散建模的另一个重要概念（第三章）。对于两个分布 $p$ 和 $q$，其定义为$$\begin{align}
    \mathcal D_{\mathrm F}(p \|  q):=\mathbb{E}_{\mathbf{x}\sim p} \left[
\left\|\nabla_{\mathbf x}\log p(\mathbf x)-\nabla_{\mathbf x}\log q(\mathbf x)\right\|_2^{2}
\right].\tag{1.1.3} \end{align} $$
它衡量的是评分函数 $\nabla_{\mathbf x}\log p(\mathbf x)$ 和 $\nabla_{\mathbf x}\log q(\mathbf x)$ 之间的差异，这些向量场指向概率更高的区域。简而言之，$\mathcal D_{\mathrm F}(p \|  q)\ge 0$，当且仅当 $p=q$ 几乎处处成立时，等式成立。它对归一化常数具有不变性，因为分数仅取决于对数密度的梯度，并构成了 *分数匹配* 的基础：一种通过学习对数密度梯度进行生成的方法（基于分数的模型）。在此设定中，数据分布 $p=p_{\mathrm{data}}$ 作为目标，而模型 $q=p_{\phi}$ 的训练目标是使其分数场与数据的分数场对齐。


**Beyond KL**    虽然 KL 散度是衡量概率分布差异最广泛使用的指标，但它并非唯一选择。不同的散度度量捕捉了差异的不同几何或统计概念，进而影响学习算法的优化动态。一个广泛的家族是 *$f$ 散度*：

$$\begin{align}
    \mathcal{D}_f(p\|q)
=\int q(\mathrm{x}) f \left(\frac{p(\mathrm{x})}{q(\mathrm{x})}\right)\mathrm{d}\mathrm{x},
\qquad f(1)=0,
\end{align}$$

这里的 $f:\mathbb{R}_+ \to\mathbb{R}$ 是一个凸函数。通过改变 $f$，我们得到了许多著名的散度：$$
\begin{aligned}
f(u)&=u\log u &&\Rightarrow&& \mathcal{D}_f=\mathcal{D}_{\mathrm{KL}}(p\|q)\quad\text{(forward KL)},\\
f(u)&=\tfrac12 \left[u\log u-(u+1)\log \tfrac{1+u}{2}\right] &&\Rightarrow&& \mathcal{D}_f=\mathcal{D}_{\mathrm{JS}}(p\|q)\quad\text{(Jensen--Shannon)},\\
f(u)&=\tfrac12|u-1| &&\Rightarrow&& \mathcal{D}_f=\mathcal{D}_{\mathrm{TV}}(p,q)\quad\text{(total variation)}.
\end{aligned}$$
为清楚起见，明确的形式是$$
\mathcal{D}_{\mathrm{JS}}(p\|q)=\tfrac12 \mathcal{D}_{\mathrm{KL}}\big(p\| \tfrac12(p+q)\big)+\tfrac12 \mathcal{D}_{\mathrm{KL}}\big(q\| \tfrac12(p+q)\big),$$和$$\mathcal{D}_{\mathrm{TV}}(p,q)=\tfrac12 \int_{\mathbb{R}^D
} |p-q|\mathrm{d} \mathrm{x}
=\sup_{A\subset \mathbb{R}^D } |p(A)-q(A)|.$$
直观地说，JS 散度提供了一种平滑且对称的度量方法，能够平衡两种分布并避免 KL 散度的无界惩罚（我们稍后将看到，这有助于解释生成对抗网络（GAN）框架），而总变差距离则捕捉了两种分布之间最大的可能概率差异。

另一种观点来自最优传输理论（参见第 7 章），其代表是 Wasserstein 距离。它衡量将概率质量从一个分布转移到另一个分布的最小成本。与比较密度比的 $f$-散度不同，Wasserstein 距离依赖于样本空间的几何结构，即使在 $p$ 和 $q$ 的支撑集不重叠时仍然具有意义。

每一种差异都体现了分布之间不同的接近概念，从而引发不同的学习行为。在本专著的生成建模过程中，当这些差异自然出现时，我们将重新审视它们。


### 1.1.2 建模分布中的挑战

为了模拟复杂的数据分布，我们可以使用带有参数 $\phi$ 的神经网络对概率密度函数 $p_{\mathrm{data}}$ 进行参数化，从而创建一个我们表示为 $p_{\phi}$ 的模型。为了使 $p_{\phi}$ 成为一个有效的概率密度函数，它必须满足两个基本属性：

\begin{enumerate}
    \item[(i)] \textbfs{Non-Negativity:} $p_{\bm{\phi}}(\rvx) \ge 0$ for all $\rvx$ in the domain.
    \item[(ii)] \textbfs{Normalization:} The integral over the entire domain must equal one, i.e., $\int p_{\bm{\phi}}(\rvx) \diff \rvx = 1$.
\end{enumerate}

A network can naturally produce a real scalar $E_{\bphi}(\rvx) \in \R$ for input $\rvx$. To interpret this output as a valid density, it must be transformed to satisfy conditions (i) and (ii).  
A practical alternative is to view $E_{\bphi}\colon\mathbb{R}^D\to\mathbb{R}$ as defining an \emph{unnormalized} density and then enforce these properties explicitly.


\paragraph{Step 1: Ensuring Non-Negativity.}
We can guarantee that our model's output is always non-negative by applying a positive function to the raw output of the neural network $E_{\bm{\phi}}(\rvx)$, such as $\abs{E_{\bm{\phi}}(\rvx)}$, $E^2_{\bm{\phi}}(\rvx)$. A standard and convenient choice is the exponential function. This gives us an unnormalized density, $\tilde{p}_{\bm{\phi}}(\rvx)$, that is guaranteed to be positive:
\[
\tilde{p}_{\bm{\phi}}(\rvx) = \exp(E_{\bm{\phi}}(\rvx)).
\]

\paragraph{Step 2: Enforcing Normalization.}
The function $\tilde{p}_{\bm{\phi}}(\rvx)$ is positive but does not integrate to one. To create a valid probability density, we must divide it by its integral over the entire space. This leads to the final form of our model:
\[
p_{\bm{\phi}}(\rvx) = \frac{\tilde{p}_{\bm{\phi}}(\rvx)}{\int \tilde{p}_{\bm{\phi}}(\rvx') \diff \rvx'} = \frac{\exp(E_{\bm{\phi}}(\rvx))}{\int \exp(E_{\bm{\phi}}(\rvx')) \diff \rvx'}.
\]
The denominator in this expression is known as the \emph{normalizing constant} or \emph{partition function}, denoted by $Z(\bm{\phi})$:
\[
Z(\bm{\phi}) := \int \exp(E_{\bm{\phi}}(\rvx')) \diff \rvx'.
\]
While this procedure provides a valid construction for $p_{\bm{\phi}}(\rvx)$, it introduces a major computational challenge. For most high-dimensional problems, the integral required to compute the normalizing constant $Z(\bm{\phi})$ is intractable. This intractability is a central problem that motivates the development of many different families of deep generative models.


In the following sections, we introduce several prominent approaches of DGM. Each is designed to circumvent or reduce the computational cost of evaluating this normalizing constant.
\newpage



\begin{figure}[th!]
\centering
\resizebox{\linewidth}{!}{%
\begin{tikzpicture}[
  ->, >=Stealth, thick, font=\normalsize,
  x=2.6cm, y=2.0cm,
  every node/.style={outer sep=0pt}
]
% ============= Styles (single source of truth) =============
\tikzset{
  state/.style={
    rectangle, rounded corners=3pt,
    draw=black, fill=gray!10,
    minimum height=10mm, minimum width=14mm, % same for all variables
    align=center, inner sep=2pt,
    font=\Large % <-- enlarge text in ALL variable nodes (x, x', x_t, z, ...)
  },
  stategap/.style={ % invisible box with SAME size as state, for equal arrow lengths
    rectangle, rounded corners=3pt,
    draw=none, fill=none,
    minimum height=10mm, minimum width=14mm,
    align=center, inner sep=2pt
  },
  processor/.style={
    trapezium, trapezium stretches=true,
    trapezium left angle=80, trapezium right angle=80,
    draw=black, fill=gray!20,
    minimum height=12mm, minimum width=28mm, align=center, inner sep=2pt
  },
  ovalval/.style={
    draw, ellipse, fill=gray!10,
    minimum height=7mm, minimum width=20mm, align=center
  },
  flowblock/.style={ % NF function blocks
    rectangle, rounded corners=3pt,
    draw=black, fill=gray!20,
    minimum height=12mm, minimum width=28mm, align=center, inner sep=3pt
  }
}

% ============= Shared column anchors (perfect row alignment) =============
\coordinate (L) at (0,0);    % left column
\coordinate (M) at (2.0,0);  % (kept for reference)
\coordinate (R) at (4.8,0);  % right column
\coordinate (C) at ($ (L)!0.5!(R) $); % true midpoint between L and R

% Vertical offsets for rows
\def\rowA{0}       % Row 1: EBM
\def\rowB{-1.3}    % Row 2: AR
\def\rowC{-3.0}    % Row 3: VAE
\def\rowD{-4.7}    % Row 4: NF
\def\rowE{-7.1}    % Row 5: GAN (two floors)
\def\rowF{-8.5}    % Row 6: DM

% Row tags
\node[anchor=east] at ($(L)+(-0.5,\rowA)$) {\textbfs{EBM}};
\node[anchor=east] at ($(L)+(-0.5,\rowB)$) {\textbfs{AR}};
\node[anchor=east] at ($(L)+(-0.5,\rowC)$) {\textbfs{VAE}};
\node[anchor=east] at ($(L)+(-0.5,\rowD)$) {\textbfs{NF}};
\node[anchor=east] at ($(L)+(-0.5,\rowE)$) {\textbfs{GAN}};
\node[anchor=east] at ($(L)+(-0.5,\rowF)$) {\textbfs{DM}};

% ====================== Row 1: EBM (unchanged) ======================
\node[state]   (ebm-x)   at ($(L)+(0,\rowA)$) {$\rvx$};
\node[ovalval] (ebm-val) at ($(C)+(0,\rowA)$) {value};
\coordinate (midE) at ($ (ebm-x.east)!0.5!(ebm-val.west) $);
\node[processor, shape border rotate=270] (ebm-energy) at (midE)
  {\footnotesize\textbfs{Energy}\\[-2pt]\footnotesize $E_{\bm\phi}(\rvx)$};
\draw (ebm-x.east) -- (ebm-energy.west);
\draw (ebm-energy.east) -- (ebm-val.west);

% ====================== Row 2: AR (unchanged) =======================
\coordinate (Lb) at ($(L)+(0,\rowB)$);
\coordinate (Rb) at ($(R)+(0,\rowB)$);
\node[state]   (ar-x)    at ($ (Lb)!0/6!(Rb) $) {$\rvx$};
\node[state]   (ar-x0)   at ($ (Lb)!1/6!(Rb) $) {$\rvx_{0}$};
\node[state]   (ar-x1)   at ($ (Lb)!2/6!(Rb) $) {$\rvx_{1}$};
\node[state]   (ar-x2)   at ($ (Lb)!3/6!(Rb) $) {$\rvx_{2}$}; % centered at C
\node[stategap](ar-gap)  at ($ (Lb)!4/6!(Rb) $) {};
\node[state]   (ar-xLm1) at ($ (Lb)!5/6!(Rb) $) {$\rvx_{L-1}$};
\node[state]   (ar-xL)   at ($ (Lb)!6/6!(Rb) $) {$\rvx_{L}$};
\node at (ar-gap) {$\boldsymbol{\cdots}$};
\draw (ar-x.east)    -- (ar-x0.west);
\draw (ar-x0.east)   -- (ar-x1.west);
\draw (ar-x1.east)   -- (ar-x2.west);
\draw (ar-x2.east)   -- (ar-gap.west);
\draw (ar-gap.east)  -- (ar-xLm1.west);
\draw (ar-xLm1.east) -- (ar-xL.west);
% Optional arcs
\draw (ar-x.south)    to[out=-35, in=-145] (ar-x2.south);
\draw (ar-x0.south)   to[out=-40, in=-140] (ar-x2.south);
\draw (ar-gap.south)  to[out=-40, in=-140] (ar-xL.south);
\draw (ar-xLm1.south) to[out=-55, in=-125] (ar-xL.south);

% ====================== Row 3: VAE (unchanged) =======================
\node[state] (vae-x)   at ($(L)+(0,\rowC)$) {$\rvx$};
\node[state] (vae-z)   at ($(C)+(0,\rowC)$) {$\rvz$};
\node[state] (vae-xp)  at ($(R)+(0,\rowC)$) {$\rvx'$};
\coordinate (midEnc) at ($ (vae-x.east)!0.5!(vae-z.west) $);
\node[processor, shape border rotate=270] (vae-enc) at (midEnc)
  {\footnotesize\textbfs{Encoder}\\[-2pt]\footnotesize $q_{\bm\theta}(\rvz|\rvx)$};
\coordinate (midDec) at ($ (vae-z.east)!0.5!(vae-xp.west) $);
\node[processor, shape border rotate=90] (vae-dec) at (midDec)
  {\footnotesize\textbfs{Decoder}\\[-2pt]\footnotesize $p_{\bm\phi}(\rvx|\rvz)$};
\draw (vae-x.east)   -- (vae-enc.west);
\draw (vae-enc.east) -- (vae-z.west);
\draw (vae-z.east)   -- (vae-dec.west);
\draw (vae-dec.east) -- (vae-xp.west);

% ====================== Row 4: NF (unchanged) =======================
\node[state] (nf-x)  at ($(L)+(0,\rowD)$) {$\rvx$};
\node[state] (nf-z)  at ($(C)+(0,\rowD)$) {$\rvz$};
\node[state] (nf-xp) at ($(R)+(0,\rowD)$) {$\rvx'$};
\coordinate (midFwd) at ($ (nf-x.east)!0.5!(nf-z.west) $);
\coordinate (midInv) at ($ (nf-z.east)!0.5!(nf-xp.west) $);
\node[flowblock] (nf-forward) at (midFwd)
  {\footnotesize\textbfs{Forward}\\[-1pt]\footnotesize $\rvf_{\bm\phi}(\rvx)$};
\node[flowblock] (nf-inverse) at (midInv)
  {\footnotesize\textbfs{Inverse}\\[-1pt]\footnotesize $\rvf_{\bm\phi}^{-1}(\rvz)$};
\draw (nf-x.east)       -- (nf-forward.west);
\draw (nf-forward.east) -- (nf-z.west);
\draw (nf-z.east)       -- (nf-inverse.west);
\draw (nf-inverse.east) -- (nf-xp.west);

% ====================== Row 5: GAN (two floors; unchanged) =======================
\def\gansep{0.9} % vertical gap between lower and upper floors
\node[state] (gan-z)  at ($(L)+(0,\rowE)$) {$\rvz$};
\node[state] (gan-xp) at ($(C)+(0,\rowE)$) {$\rvx'$};
\coordinate (midGen) at ($ (gan-z.east)!0.5!(gan-xp.west) $);
\node[processor, shape border rotate=90] (gan-gen) at (midGen)
  {\footnotesize\textbfs{Generator}\\[-2pt]\footnotesize $\rmG_{\bm\phi}(\rvz)$};
\node[state]   (gan-x)   at ($(C)+(0,\rowE+\gansep)$) {$\rvx$};
\node[ovalval] (gan-val) at ($(R)+(0,\rowE+\gansep)$) { 0/1};
\coordinate (midDisc) at ($ (gan-x.east)!0.5!(gan-val.west) $);
\node[processor, shape border rotate=270] (gan-disc) at (midDisc)
  {\footnotesize\textbfs{Discriminator}\\[-2pt]\footnotesize $D_{\bm\zeta}$};
\draw (gan-z.east)   -- (gan-gen.west);
\draw (gan-gen.east) -- (gan-xp.west);
\draw (gan-x.east)   -- (gan-disc.west);
\draw (gan-xp.east)  -- (gan-disc);
\draw (gan-disc.east) -- (gan-val.west);

% ====================== Row 6: DM (aligned with AR; equal-length arrows) =======================
% Use the same 7 centers as AR, anchored at L and R
\coordinate (Lf) at ($(L)+(0,\rowF)$);
\coordinate (Rf) at ($(R)+(0,\rowF)$);

\node[state]    (dm-x)     at ($ (Lf)!0/6!(Rf) $) {$\rvx$};
\node[state]    (dm-x0)    at ($ (Lf)!1/6!(Rf) $) {$\rvx_{0}$};
\node[state]    (dm-x1)    at ($ (Lf)!2/6!(Rf) $) {$\rvx_{1}$};
\node[state]    (dm-x2)    at ($ (Lf)!3/6!(Rf) $) {$\rvx_{2}$}; % exactly the same x as AR's x_2 and C
\node[stategap] (dm-gap)   at ($ (Lf)!4/6!(Rf) $) {};
\node[state]    (dm-xLm1)  at ($ (Lf)!5/6!(Rf) $) {$\rvx_{L-1}$};
\node[state]    (dm-xL)    at ($ (Lf)!6/6!(Rf) $) {$\rvx_{L}$};

% Draw the dots on top of the invisible gap box (no effect on spacing)
\node at (dm-gap) {$\boldsymbol{\cdots}$};

% Offsets for top (dashed) and bottom (solid) channels
\def\dmshift{4.5pt}

% Forward noising (top dashed; equal visible lengths)
\draw[dashed] ([yshift=\dmshift]dm-x.east)     -- ([yshift=\dmshift]dm-x0.west);
\draw[dashed] ([yshift=\dmshift]dm-x0.east)    -- ([yshift=\dmshift]dm-x1.west);
\draw[dashed] ([yshift=\dmshift]dm-x1.east)    -- ([yshift=\dmshift]dm-x2.west);
\draw[dashed] ([yshift=\dmshift]dm-x2.east)    -- ([yshift=\dmshift]dm-gap.west);
\draw[dashed] ([yshift=\dmshift]dm-gap.east)   -- ([yshift=\dmshift]dm-xLm1.west);
\draw[dashed] ([yshift=\dmshift]dm-xLm1.east)  -- ([yshift=\dmshift]dm-xL.west);

% Reverse denoising (bottom solid; equal visible lengths)
\draw ([yshift=-\dmshift]dm-x0.west)  -- ([yshift=-\dmshift]dm-x.east);
\draw ([yshift=-\dmshift]dm-x1.west)  -- ([yshift=-\dmshift]dm-x0.east);
\draw ([yshift=-\dmshift]dm-x2.west)  -- ([yshift=-\dmshift]dm-x1.east);
\draw ([yshift=-\dmshift]dm-gap.west) -- ([yshift=-\dmshift]dm-x2.east);
\draw ([yshift=-\dmshift]dm-xLm1.west)-- ([yshift=-\dmshift]dm-gap.east);
\draw ([yshift=-\dmshift]dm-xL.west)  -- ([yshift=-\dmshift]dm-xLm1.east);

\end{tikzpicture}}%
\vspace{1.0cm}
\caption{\textbfs{Computation graphs of prominent deep generative models.} Top to bottom: \textbfs{EBM} maps an input $\rvx$ to a scalar energy; \textbfs{AR} generates a sequence $\{\rvx_\ell\}$ left to right with causal dependencies; \textbfs{VAE} encodes $\rvx$ to a latent $\rvz$ and decodes to a reconstruction $\rvx'$; \textbfs{NF} applies an invertible map $\rvf_\bphi$ between $\rvx$ and $\rvz$ and uses $\rvf_\bphi^{-1}$ to produce $\rvx'$; \textbfs{GAN} transforms noise $\rvz$ to a sample $\rvx'$ that is judged against real $\rvx$ by a discriminator $D_{\bm\zeta}$; \textbfs{DM} iteratively refines a noisy sample through a multi-step denoising chain $\{\rvx_\ell\}$. Boxes denote variables, trapezoids are learnable networks, ovals are scalars; arrows indicate computation flow.}
\label{fig:ebm-ar-vae-nf-gan-dm-unified}
\end{figure}
\newpage

\section{Prominent Deep Generative Models}\label{sec:examples-dgm}





A central challenge in generative modeling is to learn expressive probabilistic models that can capture the rich and complex structure of high-dimensional data. Over the years, various modeling strategies have been developed, each making different trade-offs between tractability, expressiveness, and training efficiency. In this section, we explore some of the most influential strategies that have
shaped the field, accompanied by a comparison of their computation graphs in
\Cref{fig:ebm-ar-vae-nf-gan-dm-unified}.



% \ys{Avoid using fancy words. It’s not that you can’t use “advanced” terms like tapestry, but these words are rare for a reason: they’re hard to use well and should only appear when truly appropriate, which isn’t the case here.}

% \se{why this order? i would start with the easiest, so either AR or EBM}

% \se{this VAE paragraph is full of undefined jargon and will make little to no sense to a non-expert reader}



\paragraph{Energy-Based Models (EBMs).}
EBMs~\citep{ackley1985learning,lecun2006tutorial} define a probability distribution through an energy function $ E_{\bm{\phi}}(\rvx) $ that assigns lower energy to more probable data points. The probability of a data point is defined as:
\begin{equation*}
    p_{\bm{\phi}}(\rvx) := \frac{1}{Z({\bm{\phi}})} \exp(-E_{\bm{\phi}}(\rvx)),
\end{equation*}
where 
\[Z({\bm{\phi}}) = \int \exp(-E_{\bm{\phi}}(\rvx)) \diff\rvx\] 
is the partition function. Training EBMs typically involves maximizing the log-likelihood of the data. However, this requires techniques to address the computational challenges arising from the intractability of the partition function. In the following chapter, we will explore how Diffusion Models offer an alternative by generating data from \emph{the gradient of the log density}, which does not depend on the normalizing constant, thereby circumventing the need for partition function computation.


% \begin{figure}[tbh!]
% \centering
% \resizebox{0.6\linewidth}{!}{%
% \begin{tikzpicture}[>=Latex, thick, font=\normalsize, node distance=0.8cm]

% \tikzset{
%   state/.style={
%     rectangle, rounded corners=3pt,
%     draw=black, fill=gray!10,
%     minimum height=28pt, minimum width=18pt, align=center,
%     outer sep=0pt, inner sep=2pt
%   },
%   encoderblock/.style={
%     trapezium, trapezium stretches=true,
%     trapezium left angle=80, trapezium right angle=80,
%     shape border rotate=270,
%     draw=black, fill=gray!20,
%     minimum height=6pt, minimum width=12pt, inner sep=2pt, align=center
%   },
%   decoderblock/.style={
%     trapezium, trapezium stretches=true,
%     trapezium left angle=80, trapezium right angle=80,
%     shape border rotate=90,
%     draw=black, fill=gray!20,
%     minimum height=6pt, minimum width=12pt, inner sep=2pt, align=center
%   }
% }

% % --- EBM nodes (smaller) ---
% \node[state] (x) {$\mathbf{x}$};
% \node[decoderblock, right=of x, minimum height=34pt, minimum width=56pt, inner sep=2pt] (energy)
%   {\footnotesize\textbfs{Energy Function}\\[-3pt]\footnotesize $E_{\bm{\phi}}(\mathbf{x})$};
% \node[draw, ellipse, fill=gray!10, right=of energy,
%       minimum height=16pt, minimum width=32pt, align=center, inner sep=1.5pt] (val)
%   {\footnotesize value};

% % arrows
% \draw[->] (x) -- (energy);
% \draw[->] (energy) -- (val);

% \end{tikzpicture}}
% \caption{\textbfs{Energy-based model (EBM).} The energy function $E_{\bm{\phi}}(\mathbf{x})$ maps an input $\mathbf{x}$ to a scalar energy; lower energy implies higher likelihood $p_{\bm{\phi}}(\mathbf{x}) \propto \exp \big(-E_{\bm{\phi}}(\mathbf{x})\big)$.}
% \label{fig:ebm-graph}
% \end{figure}



\paragraph{Autoregressive Models.}
Deep autoregressive (AR) models~\citep{frey1995does,larochelle2011neural,uria2016neural} 
% \se{not the right citation, there is a lot of much earlier work} 
factorize the joint data distribution $p_{\mathrm{data}}$ into a product of conditional probabilities using the \emph{chain rule of probability}:
\begin{equation*}
    p_{\mathrm{data}}(\mathbf{x}) = \prod_{i=1}^D p_{\bm{\phi}}(x_i |\mathbf{x}_{<i}),
\end{equation*}
where $\mathbf{x} = (x_1, \ldots, x_D)$ and $\mathbf{x}_{<i} = (x_1, \ldots, x_{i-1})$.

Each conditional $p_{\bm{\phi}}(x_i |\mathbf{x}_{<i})$ is parameterized by a neural network, such as a Transformer, allowing flexible modeling of complex dependencies. Because each term is normalized by design (e.g., via softmax for discrete or parameterized Gaussian for continuous variables), global normalization is trivial.

Training proceeds by maximizing the exact likelihood, or equivalently minimizing the negative log-likelihood,
% \begin{figure}[tbh!]
%     \centering
%     \resizebox{\linewidth}{!}{%
% \begin{tikzpicture}[
%     ->, >=Stealth, thick,
%     node_dist/.style={right=1.7cm of #1},
%     font=\small
%   ]

%   % Node style
%   \tikzset{state/.style={
%       rectangle, rounded corners=3pt,
%       draw=black, fill=gray!10,
%       minimum height=35pt, minimum width=25pt
%     }
%   }

%   % Nodes
%   \node[state] (x)   {$\rvx$};
%   \node[state] (z1)  [node_dist=x]   {$\rvx_0$};
%   \node[state] (z2)  [node_dist=z1]  {$\rvx_1$};
%   \node[state] (z3)  [node_dist=z2]  {$\rvx_2$};
%   \node        (dots)[right=0.7cm of z3] {$\boldsymbol{\cdots}$};
%   \node[state] (zLm1)[right=0.7cm of dots] {$\rvx_{L-1}$}; % match gap with z3 -> dots
%   \node[state] (zT)  [node_dist=zLm1] {$\rvx_L$};

%   % Forward path (top, dashed)
%   \path[dashed] ([yshift=4.5pt]x.east)    edge ([yshift=4.5pt]z1.west);
%   \path[dashed] ([yshift=4.5pt]z1.east)   edge ([yshift=4.5pt]z2.west);
%   \path[dashed] ([yshift=4.5pt]z2.east)   edge ([yshift=4.5pt]z3.west);
%   \path[dashed] ([yshift=4.5pt]z3.east)   edge ([yshift=4.5pt]dots.west);
%   \path[dashed] ([yshift=4.5pt]dots.east) edge ([yshift=4.5pt]zLm1.west);
%   \path[dashed] ([yshift=4.5pt]zLm1.east) edge ([yshift=4.5pt]zT.west);

%   % Reverse path (bottom, solid)
%   \path ([yshift=-4.5pt]z1.west)  edge ([yshift=-4.5pt]x.east);
%   \path ([yshift=-4.5pt]z2.west)  edge ([yshift=-4.5pt]z1.east);
%   \path ([yshift=-4.5pt]z3.west)  edge ([yshift=-4.5pt]z2.east);
%   \path ([yshift=-4.5pt]dots.west)edge ([yshift=-4.5pt]z3.east);
%   \path ([yshift=-4.5pt]zLm1.west)edge ([yshift=-4.5pt]dots.east);
%   \path ([yshift=-4.5pt]zT.west)  edge ([yshift=-4.5pt]zLm1.east);

% \end{tikzpicture}
% }
% \caption{\textbfs{Illustration of DDPM.} Dashed arrows: forward noising; solid arrows: learned reverse denoising.}
% \label{fig:vdm-grap}
% \end{figure}


While AR models achieve strong density estimation and exact likelihoods, their sequential nature limits sampling speed and may restrict flexibility due to fixed ordering. Nevertheless, they remain a foundational class of likelihood-based generative models and key approaches in modern research. 
% \se{calling it a baseline is not very accurate}

% \begin{figure}[tbh!]
% \centering
% \resizebox{\linewidth}{!}{%
% \begin{tikzpicture}[
%     ->, >=Stealth, thick, font=\normalsize,
%     node_dist/.style={right=1.2cm of #1}
% ]
%   % Node style
%   \tikzset{
%     state/.style={
%       rectangle, rounded corners=3pt,
%       draw=black, fill=gray!10,
%       minimum height=35pt, minimum width=25pt, align=center
%     }
%   }

%   % Nodes (sequence)
%   \node[state] (x)   {$\rvx$};
%   \node[state] (x0)  [node_dist=x]   {$\rvx_{0}$};
%   \node[state] (x1)  [node_dist=x0]  {$\rvx_{1}$};
%   \node[state] (x2)  [node_dist=x1]  {$\rvx_{2}$};
%   \node        (dots)[right=0.7cm of x2] {$\boldsymbol{\cdots}$}; % gap A
%   \node[state] (xLm1)[right=0.7cm of dots] {$\rvx_{L-1}$};       % gap B = gap A
%   \node[state] (xL)  [node_dist=xLm1] {$\rvx_{L}$};

%   % Chain arrows (left-to-right)
%   \draw[->] (x.east)   -- (x0.west);
%   \draw[->] (x0.east)  -- (x1.west);
%   \draw[->] (x1.east)  -- (x2.west);
%   \draw[->] (x2.east)  -- (dots.west);
%   \draw[->] (dots.east) -- (xLm1.west);
%   \draw[->] (xLm1.east) -- (xL.west);

%   % Autoregressive conditioning (curved, illustrative)
%   \draw[->] (x.south)  to[out=-30, in=-150] (x2.south);
%   \draw[->] (x0.south) to[out=-40, in=-140] (x2.south);
%   % \draw[->] (x1.south) to[out=-50, in=-130] (dots.south);

%   \draw[->] (dots.south) to[out=-40, in=-140] (xL.south);
%   \draw[->] (xLm1.south) to[out=-55, in=-125] (xL.south);
%   % \draw[->] (x1.south) to[out=-20, in=-160] (xL.south);
% \end{tikzpicture}}
% \caption{\textbfs{Autoregressive (AR) model.}
% Generation proceeds left-to-right, with each token conditioned on all previous ones:
% $p(\rvx)=\prod_{t=0}^{L} p(\rvx_t|\rvx_{<t})$.}
% \label{fig:ar-graph}
% \end{figure}


\paragraph{Variational Autoencoders (VAEs).}
VAEs~\citep{kingma2013auto} extend classical autoencoders by introducing latent variables $\rvz$ that capture hidden structure in the data $\rvx$. 
Instead of directly learning a mapping between $\rvx$ and $\rvz$, VAEs adopt a probabilistic view: they learn both an \emph{encoder}, $q_{\btheta}(\rvz|\rvx)$, which approximates the unknown distribution of latent variables given the data, and a \emph{decoder}, $p_{\bphi}(\rvx|\rvz)$, which reconstructs data from these latent variables. 
To make training feasible, VAEs maximize a tractable surrogate to the true log-likelihood, called the Evidence Lower Bound (ELBO):
\begin{equation*}
    \mathcal{L}_{\text{ELBO}}(\btheta, \bphi; \rvx) 
    = \mathbb{E}_{q_{\btheta}(\rvz | \rvx)} \left[ \log p_{\bphi}(\rvx | \rvz) \right] 
    - \mathcal D_{\mathrm{KL}} \left( q_{\btheta}(\rvz | \rvx) \,\|\, p_{\mathrm{prior}}(\rvz) \right).
\end{equation*}
Here, the first term encourages accurate reconstruction of the data, while the second regularizes the latent variables by keeping them close to a simple prior distribution $p_{\mathrm{prior}}(\rvz)$ (often Gaussian).  

VAEs provide a principled way to combine neural networks with latent-variable models and remain one of the most widely used likelihood-based approaches. 
However, they also face practical challenges, such as limited sample sharpness and training pathologies (e.g., the tendency of the encoder to ignore latent variables). 
Despite these limitations, VAEs laid important foundations for later advances, including diffusion models.



% \se{this VAE paragraph is full of undefined jargon and will make little to no sense to a non-expert reader}
% \ys{It would be useful to cycle back and answer how VAEs get around the chanllenge of intractable normalizing constants. Same for the sections below on normalizing flows, GANs and autoregressive models.} \jcc{Revised!}



\paragraph{Normalizing Flows.}
Classic flow-based models, such as Normalizing Flows (NFs)~\citep{rezende2015variational} and Neural Ordinary Differential Equations (NODEs)~\citep{chen2018neural}, aim to learn a bijective mapping $\rvf_{\bm{\phi}}$ between a simple latent distribution $ \rvz $ and a complex data distribution $ \rvx $ via an invertible operator. This is achieved either through a sequence of bijective transformations (in NFs) or by modeling the transformation as an Ordinary Differential Equation (in NODEs). These models leverage the ``change-of-variable formula for densities'', enabling MLE training:
\begin{equation*}
    \log p_{\bm{\phi}}(\rvx) = \log p(\rvz) + \log \left| \det \frac{\partial \rvf_{\bm{\phi}}^{-1}(\rvx)}{\partial \rvx} \right|,
\end{equation*}
where $ \rvf_{\bm{\phi}} $ represents the invertible transformation mapping $ \rvz $ to $ \rvx $. NFs explicitly model normalized densities using invertible transformations with tractable Jacobian determinants. The normalization constant is absorbed analytically via the change-of-variables formula, making likelihood computation exact and tractable.

Despite their conceptual elegance, classic flow-based models often face practical limitations. For instance, NFs typically impose restrictive architectural constraints to ensure bijectivity, while NODEs may encounter training inefficiencies due to the computational overhead of solving ODEs. Both approaches face challenges when scaling to high-dimensional data. In later chapters, we will explore how Diffusion Models relate to and build upon these classic flow-based methods.


% \begin{figure}[tbh!]
% \centering
% \resizebox{0.85\linewidth}{!}{%
% \begin{tikzpicture}[>=Latex, thick, font=\sffamily, node distance=1.0cm]

% \tikzset{
%   state/.style={
%     rectangle, rounded corners=3pt,
%     draw=black, fill=gray!10,
%     minimum height=45pt, minimum width=24pt, align=center,
%     outer sep=0pt
%   },
%   latent/.style={
%     rectangle, rounded corners=3pt,
%     draw=black, fill=gray!10,
%     minimum height=45pt, minimum width=24pt, align=center,
%     outer sep=0pt
%   },
%   flowblock/.style={
%     rectangle, rounded corners=3pt,
%     draw=black, fill=gray!20,
%     minimum height=70pt, minimum width=120pt, align=center,
%     inner sep=6pt
%   }
% }

% % nodes
% \node[state] (x) {$\Large \mathbf{x}$};

% \node[flowblock, right=of x] (forward) {\Large\textbf{Forward}\\[-2pt]\Large\textbf{Functions}\\[2pt]
%   $\Large \rvf_{\bm{\phi}}(\mathbf{x})$};

% \node[latent, right=1.0cm of forward] (z) {$\mathbf{z}$};

% \node[flowblock, right=1.0cm of z] (inverse) {\Large\textbf{Inverse}\\[-2pt]\Large\textbf{Functions}\\[2pt]
%   $\Large \rvf_{\bm{\phi}}^{-1}(\mathbf{z})$};

% \node[state, right=of inverse] (xprime) {$\Large \mathbf{x}'$};

% % arrows (edge-to-edge)
% \draw[->] (x) -- (forward);
% \draw[->] (forward) -- (z);
% \draw[->] (z) -- (inverse);
% \draw[->] (inverse) -- (xprime);

% \end{tikzpicture}}
% \caption{\textbfs{Illustration of a normalizing flow.} A bijective forward map $\rvf_{\bm{\phi}}$ transforms data $\mathbf{x}$ to a latent $\mathbf{z}$, and the inverse $\rvf_{\bm{\phi}}^{-1}$ maps back to data space.}
% \label{fig:flow-graph}
% \end{figure}

\paragraph{Generative Adversarial Networks (GANs).}
GANs~\citep{goodfellow2014generative} consist of two neural networks, a generator $ \rmG_{\bm{\phi}} $ and a discriminator $ D_{\bm{\zeta}} $, that compete against each other. The generator aims to create realistic samples $ \rmG_{\bm{\phi}}(\rvz) $ from random noise $ \rvz \sim p_{\mathrm{prior}}$, while the discriminator attempts to distinguish between real samples $ \rvx $ and generated samples $ \rmG_{\bm{\phi}}(\rvz) $. The objective function for GANs can be formulated as:
\begin{equation*}
    \min_{\rmG_{\bm{\phi}}} \max_{D_{\bm{\zeta}}} \underbrace{\mathbb{E}_{\rvx \sim p_{\mathrm{data}}(\rvx)}[\log D_{\bm{\zeta}}(\rvx)]}_{\text{real}} + \underbrace{\mathbb{E}_{\rvz \sim p_{\mathrm{prior}}(\rvz)}\left[\log(1 - D_{\bm{\zeta}}\left(\rmG_{\bm{\phi}}(\rvz))\right)\right]}_{\text{fake}}.
\end{equation*}
GANs do not define an explicit density function and therefore bypass likelihood estimation entirely. 
Instead of computing a normalization constant, they focus on generating samples that closely mimic the data distribution. 

From a divergence perspective, the discriminator implicitly measures the discrepancy between the true data distribution $p_{\mathrm{data}}$ 
and the generator distribution $p_{\rmG_{\bphi}}$, where $p_{\rmG_{\bphi}}$ denotes the distribution of generated samples $\rmG_{\bphi}(\rvz)$ obtained from noise $\rvz \sim p_{\mathrm{prior}}$. 
With an optimal discriminator for a fixed generator $\rmG_{\bphi}$ computed as
\begin{equation*}
     \frac{p_{\mathrm{data}}(\rvx)}{p_{\mathrm{data}}(\rvx) + p_{\rmG_{\bphi}}(\rvx)},
\end{equation*}
the generator’s minimization reduces to
\begin{equation*}
    \min_{\rmG_{\bphi}} \; 2\,\mathcal{D}_\mathrm{JS} \left(p_{\mathrm{data}} \,\|\, p_{\rmG_\bphi}\right) - \log 4.
\end{equation*}
Here, $\mathcal{D}_\mathrm{JS}$ denotes the Jensen–Shannon divergence, defined as
\begin{equation*}
    \mathcal{D}_\mathrm{JS}(p \,\|\, q) 
    := \tfrac{1}{2} \mathcal{D}_\mathrm{KL} \left(p \,\middle\|\, \tfrac{p+q}{2}\right) 
    + \tfrac{1}{2} \mathcal{D}_\mathrm{KL} \left(q \,\middle\|\, \tfrac{p+q}{2}\right).
\end{equation*}
This shows that GANs implicitly minimize $\mathcal{D}_\mathrm{JS}(p_{\mathrm{data}} \,\|\, p_{\rmG_\bphi})$. 
More broadly, extensions such as $f$-GANs~\citep{nowozin2016f} generalize this view by demonstrating that adversarial training can minimize a family of $f$-divergences, 
placing GANs within the same divergence-minimization framework as other generative models.


% \begin{figure}[tbh!]
% \centering
% \resizebox{0.85\linewidth}{!}{%
% \begin{tikzpicture}[>=Latex, thick, font=\sffamily, node distance=0.9cm]

% \tikzset{
%   state/.style={
%     rectangle, rounded corners=3pt,
%     draw=black, fill=gray!10,
%     minimum height=45pt, minimum width=25pt, align=center,
%     outer sep=0pt
%   },
%   latent/.style={
%     rectangle, rounded corners=3pt,
%     draw=black, fill=gray!10,
%     minimum height=28pt, minimum width=28pt, align=center,
%     outer sep=0pt
%   },
%   encoderblock/.style={
%     trapezium, trapezium stretches=true,
%     trapezium left angle=80,
%     trapezium right angle=80,
%     shape border rotate=270, % wide->narrow (discriminator)
%     draw=black, fill=gray!20,
%     minimum height=8pt, minimum width=15pt, inner sep=4pt, align=center
%   },
%   decoderblock/.style={
%     trapezium, trapezium stretches=true,
%     trapezium left angle=80,
%     trapezium right angle=80,
%     shape border rotate=90,  % narrow->wide (generator)
%     draw=black, fill=gray!20,
%     minimum height=8pt, minimum width=15pt, inner sep=4pt, align=center
%   }
% }

% % --- GAN nodes ---
% \node[latent] (z) {$\mathbf{z}$};
% \node[decoderblock, right=of z, minimum height=48pt, minimum width=70pt] (gen)
%   {\footnotesize\textbf{Generator}\\[-2pt]\footnotesize $\mathrm{G}_{\bm{\phi}}(\mathbf{z})$};
% \node[state, right=of gen] (xprime) {$\mathbf{x}'$};
% \node[state, above=0.9cm of xprime] (x) {$\mathbf{x}$};

% \node[encoderblock, right=1.2cm of x, minimum height=44pt, minimum width=70pt] (disc)
%   {\footnotesize\textbf{Discriminator}\\[-2pt]\footnotesize $D_{\bm{\zeta}}$};

% \node[draw, ellipse, fill=gray!10, right=of disc, minimum height=22pt, minimum width=40pt, align=center, inner sep=2pt] (val)
%   {\footnotesize 0/1};

% % --- arrows ---
% \draw[->] (z) -- (gen);
% \draw[->] (gen) -- (xprime);
% \draw[->] (x) -- (disc);
% \draw[->] (xprime) -- (disc);
% \draw[->] (disc) -- (val);

% \end{tikzpicture}}
% \caption{\textbfs{Generative Adversarial Network (GAN).} The generator $\mathrm{G}_{\bm{\phi}}$ maps noise $\mathbf{z}$ to a sample $\mathbf{x}'$, while the discriminator $D_{\bm{\zeta}}$ receives either real $\mathbf{x}$ or generated $\mathbf{x}'$ and outputs a binary decision.}
% \label{fig:gan-graph}
% \end{figure}



% \se{might be nice to tie this to the divergence discussion from previous sections}

Although GANs are capable of generating high-quality data, their min-max training process is notoriously unstable, often requiring carefully designed architectures and engineering techniques to achieve satisfactory performance. However, GANs have since been revived as an auxiliary component to enhance other generative models, particularly Diffusion Models.






\newpage
\section{Taxonomy of Modelings}\label{sec:taxonomy}

% \ys{We can think of explicit and implicit models as two ways to parameterize a probabilistic distribution. Explicit models focus on directly modeling the probability density or mass function, while implicit models define the distribution indirectly through a sampling process. This distinction concerns the modeling approach only and does not address the training objective. Therefore, it is not at a high enough level to categorize “approaches” to generative modeling.} \jcc{Revised!}

As we have seen, DGMs span a wide spectrum of modeling strategies. A fundamental distinction lies in how these models \emph{parameterize} the underlying data distribution, that is, whether they specify $p_\bphi(\mathbf{x})$ \emph{explicitly} or only \emph{implicitly}, irrespective of the training objective.

\begin{itemize}
  \item \textbfs{Explicit Models:}  
  These models directly parameterize a probability distribution $p_\bphi(\mathbf{x})$ via a tractable or approximately tractable density or mass function. Examples include ARs, NFs, VAEs, and DMs, all of which define $p_\bphi(\mathbf{x})$ either exactly or through a tractable bound.

  \item \textbfs{Implicit Models:}  
  These models specify a distribution only through a sampling procedure, typically of the form $\mathbf{x} = \rmG_\bphi(\mathbf{z})$ for some noise variable $\mathbf{z} \sim p_{\mathrm{prior}}$. In this case, $p_\bphi(\mathbf{x})$ is not available in closed form and may not be defined at all.
\end{itemize}


The table in \Cref{tb:comparison-explicit-implicit} offers a concise summary of these contrasting approaches.


\begin{table}[th]
  \caption{Comparison of Explicit and Implicit Generative Models}
  \small
  \centering
  \resizebox{\textwidth}{!}{
  \begin{tabular}{cccc}
    \toprule
    & \multicolumn{2}{c}{\textbfs{Explicit}} & \textbfs{Implicit} \\
    \cmidrule(lr){2-3}
    & \textbfs{Exact Likelihood} & \textbfs{Approx.\ Likelihood} & \\
    \midrule
    \textbfs{Likelihood}
      & Tractable
      & Bound/Approx.
      & \makecell{Not Directly Modeled/\\Intractable} \\

    \textbfs{Objective}
      & MLE
      & ELBO
      & Adversarial \\

    \textbfs{Examples}
      & NFs, ARs
      & VAEs, DMs
      & GANs \\
    \bottomrule
  \end{tabular}
  }
  \label{tb:comparison-explicit-implicit}
\end{table}


\paragraph{Connection to Diffusion Models.}
Taken together, these classical families of DGMs illustrate complementary strategies for modeling complex distributions.  
Beyond their standalone importance, they also provide guiding principles for understanding diffusion models.  
Diffusion methods inherit ideas from several of these perspectives: they connect to VAEs through variational training objectives, to EBMs through score-matching approaches that learn gradients of the log-density (closely tied to energy functions), and to NFs through continuous-time transformations.  

To lay the groundwork for the diffusion methods discussed in later chapters, we will focus on three central paradigms: VAEs (\Cref{sec:vae}), EBMs (\Cref{sec:ebm}), and NFs (\Cref{sec:flow-based-method}).  
This exploration provides a foundation for the core principles that underlie modern diffusion-based generative modeling, which will be developed further in the chapters that follow.


% \rmkb{\jc{We remark that the training objective of diffusion models, from a variational perspective, constitutes an upper bound on the negative log-likelihood (i.e., an ELBO), and thus serves as a proxy for maximum likelihood estimation, which is characteristic of explicit models.}
% }
% \ys{Not sure I buy the argument that diffusion models are implicit generative models.}
% \jcc{Revised!}
% \se{i would cut this remark, it won't make much sense unless you already know this stuff (in which case it's not useful)}

% \se{i would end with some text explaing the reader why all of this stuff is relevant to diffusion models, which is what (presumably) they care about.. something high level like there are connections to ebms, vaes, nf, and looking at it from these perspectives reveals useful/interesting properties of diffusion models}


\newpage
\section{Closing Remarks}\label{sec:ch1_cr}


This chapter has established the foundational concepts of deep generative modeling. We begin by defining the primary objective: to learn a tractable model distribution $p_{\text{model}}$ (parametrized by $\bm{\phi}$) that approximates an unknown, complex data distribution $p_{\text{data}}$. A central challenge is the computational intractability of the normalizing constant, or partition function $Z(\bm{\phi})$, which is required to define a valid probability density.



To circumvent this problem, various families of deep generative models have been developed, each employing a distinct strategy. We surveyed several prominent approaches, including Energy-Based Models (EBMs), Autoregressive Models (ARs), Variational Autoencoders (VAEs), Normalizing Flows (NFs), and Generative Adversarial Networks (GANs). These models can be broadly categorized into explicit models, which define a tractable density, and implicit models, which define a distribution only through a sampling procedure.





While each of these classical frameworks is significant, three in particular serve as the conceptual origins for the diffusion models that are the focus of this monograph: VAEs, EBMs, and NFs. In the chapters that follow, we will trace the evolution of diffusion models from these three foundational paradigms:
\begin{enumerate}
    \item Part B will begin by exploring the variational perspective (\Cref{ch:variational}), showing how (the hierarchical latent variable structure of) VAEs leads naturally to the formulation of Denoising Diffusion Probabilistic Models (DDPMs).
    \item Next, we will examine the score-based perspective (\Cref{ch:score-based}), which originates from EBMs and score matching, and develops into Noise Conditional Score Networks (NCSN) and the more general Score SDE framework (\Cref{ch:score-sde}).
    \item Finally, we will investigate the flow-based perspective (\Cref{ch:flow-based}), which builds upon the principles of Normalizing Flows to frame generation as a continuous transformation, generalized by the concept of Flow Matching.
\end{enumerate}

By understanding these origins, we will build a coherent framework for interpreting the diverse formulations of diffusion models and uncovering the deep principles that unify them.

