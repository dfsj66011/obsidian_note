
> [!NOTE]
> [Paper: LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/pdf/2302.13971)
> 
> Meta AI,    2023.02
> 
> **摘要**：我们推出 LLaMA，这是一个包含 *7B 至 65B 参数* 的基础语言模型系列。我们在数万亿 token 上训练这些模型，并证明仅使用公开可获取的数据集即可训练出最先进的模型，而无需依赖专有或难以获取的数据集。具体而言，LLaMA-13B 参数版本在多数基准测试中超越 GPT-3（175B 参数），而 LLaMA-65B 参数版本可与当前最佳模型 Chinchilla-70B 和 PaLM-540B 相媲美。


### 1、引言

基于海量文本训练的 LLMs 已展现出通过文本指令或少量示例执行新任务的能力（GPT-3）。这种小样本学习特性最初在模型规模达到一定程度时显现，由此催生了一系列专注于进一步扩展模型规模的研究。这些研究基于一个*假设：更多参数会带来更好性能*。然而 chinchilla 的最新研究表明，在*给定计算预算下，最佳性能并非由最大模型实现，而是通过更多数据训练的较小模型获得*。

Hoffmann 等人（2022）提出的缩放定律旨在确定如何针对特定训练计算预算最优地扩展数据集和模型规模。然而，这一目标 *忽略了推理预算* 的重要性——当大规模部署语言模型时，推理预算就变得至关重要。在此背景下，给定目标性能水平，理想的模型并非训练速度最快者，而是 *推理速度最快者*。虽然训练大型模型可能以更低成本达到特定性能水平，但经过更长时间训练的小型模型最终在推理阶段成本更低。例如，尽管 Hoffmann 等人建议用 200B tokens 训练 10B 参数模型，但我们发现 *7B 参数模型的性能在训练 tokens 超过 1T 后仍持续提升*。

本工作的重点是通过训练比通常使用更多的 tokens，训练一系列在不同推理预算下实现最佳性能的语言模型。这些模型名为 LLaMA，参数量从 7B 到 65B 不等，与现有最佳大语言模型相比具有竞争力。例如，LLaMA-13B 在大多数基准测试中表现优于 GPT-3，尽管其规模小了 10 倍。我们相信该模型将有助于普及大语言模型的使用和研究，因为它可以在单个 GPU 上运行。在高端规模上，我们的 65B 参数模型也与 Chinchilla 或 PaLM-540B 等最佳大语言模型具有竞争力。

与 Chinchilla、PaLM 或 GPT-3 不同，我们仅使用公开可获取的数据，这使得我们的工作与开源兼容，而现有大多数模型依赖的数据要么未公开，要么缺乏文档记录（例如 “Books—2TB” 或 “Social media conversations”）。虽然存在一些例外，如 OPT、GPT-NeoX、BLOOM 和 GLM，但它们都无法与 PaLM-62B 或 Chinchilla 相媲美。

在本文的剩余部分，我们将概述对 Transformer 架构所做的改进，并介绍我们的训练方法。随后报告模型性能，与其它大语言模型在标准测试集上进行对比。最后，我们通过责任人工智能领域的最新基准测试，揭示模型内部编码的部分偏见与毒性内容。


### 2、方法

我们的训练方法类似于先前研究（GPT3, PaLM）中描述的方法，并受到 Chinchilla 缩放定律的启发。我们使用标准优化器在大量文本数据上训练大型 Transformer 模型。

#### 2.1 预训练数据
$$
\begin{array}{|l|c|c|c|}
\hline
\textbf{数据集} & \textbf{采样比例} & \textbf{训练轮数} & \textbf{磁盘占用} \\ \hline
\text{CommonCrawl} & 67.0\% & 1.10 & 3.3\,\text{TB} \\ \hline
\text{C4} & 15.0\% & 1.06 & 783\,\text{GB} \\ \hline
\text{Github} & 4.5\% & 0.64 & 328\,\text{GB} \\ \hline
\text{Wikipedia} & 4.5\% & 2.45 & 83\,\text{GB} \\ \hline
\text{Books} & 4.5\% & 2.23 & 85\,\text{GB} \\ \hline
\text{ArXiv} & 2.5\% & 1.06 & 92\,\text{GB} \\ \hline
\text{StackExchange} & 2.0\% & 1.03 & 78\,\text{GB} \\ \hline
\end{array}
$$
**表 1：预训练数据**。用于预训练的数据混合比例，针对每个子集我们列出了采样比例、在1.4T tokens 训练时对该子集执行的训练轮次以及磁盘占用空间。基于 1T tokens 的预训练运行采用相同的采样比例。


我们的训练数据集由表 1 所示的多种数据源混合而成，涵盖多个不同领域。大部分情况下，我们复用了其他大语言模型训练所使用的数据源，但仅限于采用公开可用且符合开源许可的数据。由此得到以下训练数据组成及其在训练集中的占比：

**英语 CommonCrawl 数据集**：我们对 2017 至 2020 年间的五份 CommonCrawl 转储文件进行了预处理，采用 CCNet 流水线。该处理流程实现了行级去重，通过 fastText 线性分类器进行语言识别以移除非英文页面，并利用 n-gram 语言模型过滤低质量内容。此外，我们训练了一个线性模型来区分维基百科引用页面与随机采样页面，最终剔除了未被归类为引用页面的数据。

**C4**：在探索性实验中，我们观察到使用多样化的预处理 CommonCrawl 数据集能提升性能。因此我们将公开可用的 C4 数据集纳入了训练数据。C4 的预处理流程同样包含去重和语言识别步骤：与 CCNet 的主要区别在于质量过滤环节，其主要依赖启发式规则（如标点符号的存在性、网页中单词和句子的数量等）。

**Github**：我们使用了 Google BigQuery 上公开的 GitHub 数据集。仅保留采用 Apache、BSD 和 MIT 许可证的项目。此外，基于行长度或字母数字字符比例等启发式方法过滤低质量文件，并通过正则表达式移除模板化内容（如文件头声明）。最后，我们对结果数据集进行文件级别的精确去重处理。

**维基百科**：我们添加了 2022 年 6 月至 8 月期间的维基百科转储文件，涵盖 20 种使用拉丁或西里尔字母的语言。我们对数据进行了处理，移除了超链接、注释和其他格式化的样板内容。

**Gutenberg 与 Books3 数据集**：我们的训练数据包含两个书籍语料库：由公共领域书籍构成的古登堡计划，以及来自 ThePile 公开大语言模型训练数据集的 Books3 部分。我们进行了书籍级别的去重处理，删除了内容重叠率超过 90% 的书籍。

**ArXiv**：我们通过处理 arXiv 的 LaTeX 文件来为数据集添加科学数据。我们删除了第一节之前的所有内容以及参考文献部分。同时，我们还移除了 .tex 文件中的注释，并内联扩展了用户编写的定义和宏，以提高论文间的一致性。

**Stack Exchange**：我们纳入了 Stack Exchange 的数据转储，这是一个涵盖计算机科学到化学等多领域的高质量问答网站。我们保留了 28 个最大站点的数据，清除了文本中的 HTML 标签，并按照评分从高到低对答案进行了排序。

$$
\begin{array}{|l|c|c|c|c|c|c|}
\hline
\textbf{Params} & \textbf{Dimension} & \textbf{N Heads} & \textbf{N Layers} & \textbf{Learning Rate} & \textbf{Batch Size} & \textbf{N Tokens} \\ \hline
6.7B & 4096 & 32 & 32 & 3.0 \times 10^{-4} & 4M & 1.0T \\ \hline
13.0B & 5120 & 40 & 40 & 3.0 \times 10^{-4} & 4M & 1.0T \\ \hline
32.5B & 6656 & 52 & 60 & 1.5 \times 10^{-4} & 4M & 1.4T \\ \hline
65.2B & 8192 & 64 & 80 & 1.5 \times 10^{-4} & 4M & 1.4T \\ \hline
\end{array}
$$
**表 2**：模型大小，结构和优化器超参数


**分词器**：我们采用字节对编码（BPE）算法对数据进行分词，具体实现使用 SentencePiece 工具。值得注意的是，我们将所有数字拆分为单独的数字字符，并对未知的 UTF-8 字符回退到字节级别进行分解。

总体而言，经过分词处理后，我们的整个训练数据集包含约 1.4T 个词元。在大部分训练数据中，每个词元在训练期间仅使用一次，但维基百科和书籍领域的资料除外——我们对这两类数据进行了约两个训练周期的重复学习。

#### 2.2 架构

基于近期大规模语言模型的研究成果，我们的网络采用 Transformer 架构。我们融合了后续提出的多项改进方案，这些方案曾应用于 PaLM 等不同模型。以下是与原始架构的主要差异及相应改进的灵感来源（括号内标注）：

**预归一化（GPT3）**： 为提高训练稳定性，我们对每个 Transformer 子层的输入进行归一化处理，而非归一化其输出。此处采用 RMSNorm 归一化函数。

**SwiGLU 激活函数（PaLM）**： 我们将 ReLU 非线性激活函数替换为 SwiGLU 激活函数，并采用 $\frac23 4d$ 维而非 PaLM 原论文中的 $4d$ 作为隐藏层维度。


\paragraph{Rotary Embeddings [GPTNeo].}\hspace{-3pt}We remove the absolute positional embeddings, and instead, add rotary positional embeddings (RoPE), introduced by \citet{su2021roformer}, at each layer of the network.

The details of the hyper-parameters for our different models are given in Table~\ref{tab:architecture}.

\begin{figure}
\centering
\includegraphics[width=\linewidth]{figures/train_loss.pdf} % 
\caption{
\textbf{Training loss over train tokens for the 7B, 13B, 33B, and 65 models.}
\model-33B and \model-65B were trained on 1.4T tokens. The smaller models were trained on 1.0T tokens. All models are trained with a batch size of 4M tokens.
\label{fig:trainincurves}
}
\end{figure}

\subsection{Optimizer}

Our models are trained using the AdamW optimizer~\citep{loshchilov2017decoupled}, with the following hyper-parameters: $\beta_1 = 0.9, \beta_2 = 0.95$.
We use a cosine learning rate schedule, such that the final learning rate is equal to 10\% of the maximal learning rate.
We use a weight decay of $0.1$ and gradient clipping of $1.0$.
We use $2,000$ warmup steps, and vary the learning rate and batch size with the size of the model (see Table~\ref{tab:architecture} for details).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%






\subsection{Efficient implementation}
We make several optimizations to improve the training speed of our models.
First, we use an efficient implementation of the causal multi-head attention to reduce memory usage and runtime.
This implementation, available in the \texttt{xformers} library,\footnote{https://github.com/facebookresearch/xformers} is inspired by~\citet{rabe2021self} and uses the backward from~\citet{dao2022flashattention}.
This is achieved by not storing the attention weights and not computing the key/query scores that are masked due to the causal nature of the language modeling task.


\begin{table*}[t!]
  \centering
  \setlength{\tabcolsep}{5pt}
  \begin{tabular}{lrccccccccc}
  \toprule
  & & BoolQ & PIQA & SIQA & \hspace{-0.3cm} HellaSwag \hspace{-0.2cm} & \hspace{-0.2cm} WinoGrande \hspace{-0.3cm} & ARC-e & ARC-c & OBQA \\
  \midrule
  GPT-3        & 175B & 60.5 & 81.0 & -    & 78.9 & 70.2 & 68.8 & 51.4 & 57.6 \\
  Gopher       & 280B & 79.3 & 81.8 & 50.6 & 79.2 & 70.1 & -    & -    & -    \\
  Chinchilla   & 70B  & 83.7 & 81.8 & 51.3 & 80.8 & 74.9 & -    & -    & -    \\
  PaLM         & 62B  & 84.8 & 80.5 & -    & 79.7 & 77.0 & 75.2 & 52.5 & 50.4 \\
  PaLM-cont    & 62B  & 83.9 & 81.4 & -    & 80.6 & 77.0 & -    & -    & -    \\
  PaLM         & 540B & \tbf{88.0} & 82.3 & - & 83.4 & \tbf{81.1} & 76.6 & 53.0 & 53.4 \\
  \midrule
  \multirow{4}{*}{\model}
     & 7B   & 76.5 & 79.8       & 48.9 & 76.1 & 70.1 & 72.8       & 47.6       & 57.2 \\
     & 13B  & 78.1 & 80.1       & 50.4 & 79.2 & 73.0 & 74.8       & 52.7       & 56.4 \\
     & 33B  & 83.1 & 82.3 & 50.4 & 82.8 & 76.0 & \tbf{80.0} & \tbf{57.8} & 58.6       \\
     & 65B  & 85.3 & \tbf{82.8}  & \tbf{52.3}  &  \tbf{84.2}    &  77.0    & 78.9  & 56.0  &   \tbf{60.2} \\
  \bottomrule
  \end{tabular}
  \caption{
  \textbf{Zero-shot performance on Common Sense Reasoning tasks.}
  \label{tab:commonsense}
  }
\end{table*}

To further improve training efficiency, we reduced the amount of activations that are recomputed during the backward pass with checkpointing.
More precisely, we save the activations that are expensive to compute, such as the outputs of linear layers.
This is achieved by manually implementing the backward function for the transformer layers, instead of relying on the PyTorch autograd.
To fully benefit from this optimization, we need to reduce the memory usage of the model by using model and sequence parallelism, as described by \citet{korthikanti2022reducing}. Moreover, we also overlap the computation of activations and the communication between GPUs over the network (due to \texttt{all\_reduce} operations) as much as possible.

When training a 65B-parameter model, our code processes around 380 tokens/sec/GPU on 2048 A100 GPU with 80GB of RAM.
This means that training over our dataset containing 1.4T tokens takes approximately 21 days.


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



\section{Main results}
Following previous work~\citep{brown2020gpt3}, we consider zero-shot and few-shot tasks, and report results on a total of 20 benchmarks:
\begin{itemize}
\item\textbf{Zero-shot.}
We provide a textual description of the task and a test example.
The model either provides an answer using open-ended generation, or ranks the proposed answers.
\item\textbf{Few-shot.} 
We provide a few examples of the task (between 1 and 64) and a test example. The model takes this text as input and generates the answer or ranks different options.
\end{itemize}

We compare \model with other foundation models, namely the non-publicly available language models GPT-3~\citep{brown2020gpt3}, Gopher~\citep{rae2021goepher}, Chinchilla~\citep{hoffmann2022chinchilla} and PaLM~\citep{chowdhery2022palm}, as well as the open-sourced OPT models~\citep{zhang2022opt}, GPT-J~\citep{gptj}, and GPT-Neo~\citep{black2022gpt}.
In Section~\ref{sec:instruct}, we also briefly compare \model with instruction-tuned models such as OPT-IML~\cite{iyer2022opt} and Flan-PaLM~\cite{Chung2022ScalingIL}.

We evaluate \model on free-form generation tasks and multiple choice tasks.
In the multiple choice tasks, the objective is to select the most appropriate completion among a set of given options, based on a provided context.
We select the completion with the highest likelihood given the provided context.
We follow \citet{eval-harness} and use the likelihood normalized by the number of characters in the completion, except for certain datasets (OpenBookQA, BoolQ), for which we follow \citet{brown2020gpt3}, and select a completion based on the likelihood normalized by the likelihood of the completion given ``Answer:'' as context: ${\scriptstyle P ( \mathtt{completion} \mid \mathtt{context})/P(\mathtt{completion} \mid ``Answer:" ) }$.

\begin{table}[h]
  \centering
  \setlength{\tabcolsep}{4pt}

   {
  \begin{tabular}{@{}l@{} r cccc@{}}
    \toprule
           && 0-shot & 1-shot & 5-shot & 64-shot \\
    \midrule
    GPT-3      & 175B & 14.6 & 23.0 & -    & 29.9 \\
    Gopher     & 280B & 10.1 & -    & 24.5 & 28.2 \\
    Chinchilla & 70B  & 16.6 & -    & 31.5 & 35.5 \\
    \midrule
    \multirow{3}{*}{PaLM}
               & 8B   & 8.4  & 10.6 & - & 14.6\\
               & 62B   & 18.1  &26.5 & - & 27.6\\
               & 540B & 21.2 & 29.3 & -    & 39.6\\
    \midrule
    \multirow{4}{*}{\model}
                & 7B   & 16.8 &	18.7 &	22.0 &	26.1 \\
                & 13B  & 20.1 &	23.4 &	28.1 &	31.9 \\
                & 33B  & \bf{24.9} &	28.3 &	32.9 &	36.0 \\
                & 65B  & 23.8 &	\bf{31.0} &	\bf{35.0} &	\bf{39.9} \\
    \bottomrule
  \end{tabular}}
  \caption{
  \textbf{NaturalQuestions.} Exact match performance.
  \label{tab:nqa}
  }
\end{table}

\subsection{Common Sense Reasoning}

We consider eight standard common sense reasoning benchmarks: BoolQ~\citep{clark2019boolq}, PIQA~\citep{bisk2020piqa}, SIQA~\citep{sap2019socialiqa}, HellaSwag~\citep{zellers2019hellaswag}, WinoGrande~\citep{sakaguchi2021winogrande}, ARC easy and challenge~\citep{clark2018think} and OpenBookQA~\citep{mihaylov2018can}.
These datasets include Cloze and Winograd style tasks, as well as multiple choice question answering.
We evaluate in the zero-shot setting as done in the language modeling community.

In Table~\ref{tab:commonsense}, we compare with existing models of various sizes and report numbers from the corresponding papers.
First, \model-65B outperforms Chinchilla-70B on all reported benchmarks but BoolQ.
Similarly, this model surpasses PaLM-540B everywhere but on BoolQ and WinoGrande.
\model-13B model also outperforms GPT-3 on most benchmarks despite being 10$\times$ smaller.


\subsection{Closed-book Question Answering}


We compare \model to existing large language models on two closed-book question answering benchmarks: Natural Questions~\citep{kwiatkowski2019natural} and TriviaQA~\cite{joshi2017triviaqa}.
For both benchmarks, we report exact match performance in a closed book setting, i.e., where the models do not have access to documents that contain evidence to answer the question.
In Table~\ref{tab:nqa}, we report performance on NaturalQuestions, and in Table~\ref{tab:tqa}, we report on TriviaQA.
On both benchmarks, \model-65B achieve state-of-the-arts performance in the zero-shot and few-shot settings.
More importantly, the \model-13B is also competitive on these benchmarks with GPT-3 and Chinchilla, despite being 5-10$\times$ smaller. 
This model runs on a single V100 GPU during inference.


\begin{table}[h]
  \centering
  \setlength{\tabcolsep}{4pt}
   {
  \begin{tabular}{@{}l@{} r cccc@{}}
    \toprule
&& 0-shot & 1-shot & 5-shot & 64-shot\\
    \midrule
    Gopher     & 280B & 43.5 & - & 57.0 & 57.2 \\
    Chinchilla & 70B  & 55.4 & - & 64.1 & 64.6 \\
    \midrule
    \multirow{4}{*}{\model}
                & 7B   &	50.0 & 53.4 &	56.3 &	57.6 \\
                & 13B  &	56.6 & 60.5 &	63.1 &	64.0 \\
                & 33B  &	65.1 & 67.9 &	69.9 &	70.4 \\
                & 65B  &    \bf{68.2} & \bf{71.6} &	\bf{72.6} &	\bf{73.0}   \\
    \bottomrule
  \end{tabular}}
  \caption{
  \textbf{TriviaQA.} Zero-shot and few-shot exact match performance on the filtered dev set.
  \label{tab:tqa}
  }
\end{table}


\subsection{Reading Comprehension}
We evaluate our models on the RACE reading comprehension benchmark~\citep{lai2017race}.
This dataset was collected from English reading comprehension exams designed for middle and high school Chinese students.
We follow the evaluation setup from~\citet{brown2020gpt3}
and report results in Table~\ref{tab:readingcomprehension}.
On these benchmarks, \model-65B is competitive with PaLM-540B, and, \model-13B outperforms GPT-3 by a few percents.



\begin{table}[t]
  \center
  \begin{tabular}{@{}lrcc@{}}
  \toprule
  & & RACE-middle & RACE-high \\
  \midrule
  GPT-3  & 175B & 58.4 & 45.5 \\
  \midrule
  \multirow{3}{*}{PaLM}
         & 8B   & 57.9 & 42.3 \\
         & 62B  & 64.3 & 47.5 \\
         & 540B & \tbf{68.1} & 49.1 \\
  \midrule
  \multirow{4}{*}{\model}
         & 7B & 61.1 & 46.9 \\
         & 13B  & 61.6 & 47.2 \\
         & 33B  & 64.1 & 48.3 \\
         & 65B  & 67.9 & \tbf{51.6}    \\
  \bottomrule
  \end{tabular}
  \caption{
  \textbf{Reading Comprehension.} Zero-shot accuracy.
  \label{tab:readingcomprehension}
  }
\end{table}


\subsection{Mathematical reasoning}

We evaluate our models on two mathematical reasoning benchmarks: MATH~\cite{hendrycks2021measuring} and GSM8k~\cite{cobbe2021training}.
MATH is a dataset of 12K middle school and high school mathematics problems written in LaTeX.
GSM8k is a set of middle school mathematical problems.
In Table~\ref{tab:math}, we compare with PaLM and Minerva~\cite{lewkowycz2022solving}.
Minerva is a series of PaLM models finetuned on 38.5B tokens extracted from ArXiv and Math Web Pages, while neither PaLM or \model are finetuned on mathematical data.
The numbers for PaLM and Minerva are taken from~\citet{lewkowycz2022solving}, and we compare with and without \texttt{maj1@k}.
\texttt{maj1@k} denotes evaluations where we generate $k$ samples for each problem and perform a majority voting~\cite{wang2022Self}.
On GSM8k, we observe that \model-65B outperforms Minerva-62B, although it has not been fine-tuned on mathematical data.

\begin{table}[h] %
  \center
  \setlength{\tabcolsep}{2pt}
  \begin{tabular}{@{}l@{}r@{} cc c cc@{}}
  \toprule
&       & MATH & \footnotesize{+\texttt{maj1@k}} && GSM8k & \footnotesize{+\texttt{maj1@k}} \\
  \midrule
  \multirow{3}{*}{PaLM}
  	& 8B	& 1.5  & - && 4.1  & - \\
	& 62B	& 4.4  & - && 33.0 & - \\
	& 540B	& 8.8  & - && 56.5 & - \\
 \midrule
 \multirow{3}{*}{Minerva}
        & 8B	& 14.1 & 25.4 && 16.2 & 28.4 \\
	& 62B	& 27.6 & 43.4 && 52.4 & 68.5 \\
	& 540B	& \bf 33.6 & \bf 50.3 && \bf 68.5 & \bf 78.5 \\
\midrule
\multirow{4}{*}{\model}
        & 7B	& 2.9  & 6.9  && 11.0 & 18.1 \\
	& 13B	& 3.9  & 8.8  && 17.8 & 29.3 \\
	& 33B	& 7.1  & 15.2 && 35.6 & 53.1 \\
	& 65B	& 10.6 & 20.5 && 50.9 & 69.7 \\
  \bottomrule
  \end{tabular}
  \caption{
  \textbf{Model performance on quantitative reasoning datasets.} For majority voting, we use the same setup as Minerva, with $k=256$ samples for MATH and $k=100$ for GSM8k (Minerva 540B uses $k=64$ for MATH and and $k=40$ for GSM8k). \model-65B outperforms Minerva 62B on GSM8k, although it has not been fine-tuned on mathematical data.
  \label{tab:math}
  }
\end{table}


\subsection{Code generation}
\label{sec:codegen}
We evaluate the ability of our models to write code from a natural language
description on two benchmarks: HumanEval~\cite{chen2021Evaluating} and MBPP~\cite{austin2021program}.
For both tasks, the model receives a description of the program in a few sentences, as well as a few input-output examples. 
In HumanEval, it also receives a function signature, and the prompt is formatted as natural code with the textual description and tests in a docstring. 
The model needs to generate a Python program that fits the description and satisfies the test cases.
In Table~\ref{tab:code}, we compare the pass@1 scores of our models with existing language models that have not been finetuned on code, namely PaLM and LaMDA~\cite{thoppilan2022lambda}. PaLM and \model were trained on datasets that contain a similar number of code tokens.

As show in Table~\ref{tab:code}, for a similar number of parameters, \model outperforms other general models such as LaMDA and PaLM, which are not trained or finetuned specifically for code.
\model with 13B parameters and more outperforms LaMDA 137B on both HumanEval and MBPP. 
\model 65B also outperforms PaLM 62B, even when it is trained longer. 
The pass@1 results reported in this table were obtained by sampling with temperature 0.1. The pass@100 and pass@80 metrics were obtained with temperature 0.8. We use the same method as~\citet{chen2021Evaluating} to obtain unbiased estimates of the pass@k.

It is possible to improve the performance on code by finetuning on code-specific tokens.
For instance, PaLM-Coder~\citep{chowdhery2022palm} increases the pass@1 score of PaLM on HumanEval from 26.2\% for PaLM to 36\%.
Other models trained specifically for code also perform better than general models on these tasks~\citep{chen2021Evaluating,nijkamp2022codegen,fried2022incoder}.
Finetuning on code tokens is beyond the scope of this paper.


\begin{table}[h!]
\setlength{\tabcolsep}{4pt}
\centering
\begin{tabular}{lrcccc}
\toprule
 & Params & \multicolumn{2}{c}{HumanEval} &  \multicolumn{2}{c}{MBPP}\\
pass@ &  & @1 & @100 &  @1 & @80 \\
\midrule
LaMDA & 137B & 14.0 & 47.3 & 14.8 & 62.4\\
PaLM  & 8B & 3.6$^*$  & 18.7$^*$ & 5.0$^*$ & 35.7$^*$\\
PaLM  & 62B & 15.9  & 46.3$^*$ & 21.4 & 63.2$^*$\\
PaLM-cont & 62B & 23.7  & - & 31.2 & -\\
PaLM  & 540B & \textbf{26.2} & 76.2 & 36.8 & 75.0\\
\midrule
\multirow{4}{*}{\model}
                        & 7B    & 10.5 &  36.5 & 17.7 & 56.2\\
                        & 13B   & 15.8 & 52.5 & 22.0 & 64.0\\
                        & 33B   & 21.7 & 70.7 & 30.2 & 73.4\\
                        & 65B   & 23.7  & \textbf{79.3} & \textbf{37.7} & \textbf{76.8}\\
\bottomrule
\end{tabular}
\caption{
\textbf{Model performance for code generation.}
We report the pass@ score on HumanEval and MBPP. HumanEval generations are done in zero-shot and MBBP with 3-shot prompts similar to~\citet{austin2021program}. The values marked with $^*$ are read from figures in~\citet{chowdhery2022palm}.
\label{tab:code}
}
\end{table}


\begin{table*}[t!]
    \center
    \begin{tabular}{lrcccccc}
        \toprule
         &  & Humanities & STEM & Social Sciences & Other & Average\\
        \midrule
        GPT-NeoX   & 20B   & 29.8 & 34.9 & 33.7 & 37.7 & 33.6 \\
        GPT-3      & 175B  & 40.8 & 36.7 & 50.4 & 48.8 & 43.9 \\
        Gopher     & 280B  & 56.2 & 47.4 & 71.9 & 66.1 & 60.0 \\
        Chinchilla & 70B   & 63.6 & 54.9 & 79.3 & \tbf{73.9} & 67.5\\
        \midrule
        \multirow{3}{*}{PaLM}
                   & 8B       & 25.6 & 23.8 & 24.1 & 27.8 & 25.4 \\
                   & 62B      & 59.5 & 41.9 & 62.7 & 55.8 & 53.7 \\
                   & 540B     & \tbf{77.0} & \tbf{55.6} & \tbf{81.0} & 69.6 & \tbf{69.3} \\
        \midrule
        \multirow{4}{*}{\model}
                   & 7B & 34.0 & 30.5 & 38.3 & 38.1 & 35.1 \\
                   & 13B  & 45.0 & 35.8 & 53.8 & 53.3 & 46.9 \\
                   & 33B  & 55.8 & 46.0 & 66.7 & 63.4 & 57.8 \\
                   & 65B  & 61.8 & 51.7 & 72.9 & 67.4 & 63.4  \\
        \bottomrule
    \end{tabular}
    \caption{
    \textbf{Massive Multitask Language Understanding (MMLU).} Five-shot accuracy.
    \label{tab:mmlu}
    }
\end{table*}
\subsection{Massive Multitask Language Understanding}
The massive multitask language understanding benchmark, or MMLU, introduced by \citet{hendrycks2020measuring} consists of multiple choice questions covering various domains of knowledge, including humanities, STEM and social sciences. 
We evaluate our models in the 5-shot setting, using the examples provided by the benchmark, and report results in Table~\ref{tab:mmlu}.
On this benchmark, we observe that the \model-65B is behind both Chinchilla-70B and PaLM-540B by a few percent in average, and across most domains.
A potential explanation is that we have used a limited amount of books and academic papers in our pre-training data, i.e., ArXiv, Gutenberg and Books3, that sums up to only 177GB, while these models were trained on up to 2TB of books.
This large quantity of books used by Gopher, Chinchilla and PaLM may also explain why Gopher outperforms GPT-3 on this benchmark, while it is comparable on other benchmarks.

\begin{figure*}[t]
\centering
\includegraphics[width=\linewidth]{figures/all_evals.pdf}
\caption{
\textbf{Evolution of performance on question answering and common sense reasoning during training.}
\label{fig:evals}
}
\end{figure*}

\subsection{Evolution of performance during training}

\looseness=-1 During training, we tracked the performance of our models on a few question answering and common sense benchmarks, and report them in Figure~\ref{fig:evals}.
On most benchmarks, the performance improves steadily, and correlates with the training perplexity of the model (see Figure~\ref{fig:trainincurves}). 
The exceptions are SIQA and WinoGrande.
Most notably, on SIQA, we observe a lot of variance in performance, that may indicate that this benchmark is not reliable.
On WinoGrande, the performance does not correlate as well with training perplexity: the \model-33B and \model-65B have similar performance during the training. 

\section{Instruction Finetuning}
\label{sec:instruct}

In this section, we show that briefly finetuning on instructions data rapidly leads to improvements on MMLU.
% 
Although the non-finetuned version of \model-65B is already able to follow basic instructions, we observe that a very small amount of finetuning improves the performance on MMLU, and further improves the ability of the model to follow instructions.
Since this is not the focus of this paper, we only conducted a single experiment following the same protocol as~\citet{Chung2022ScalingIL} to train an instruct model, \model-I.



\begin{table}[h]
\centering
\begin{tabular}{lrc c}
\toprule
OPT  & 30B  & 26.1 \\
GLM  & 120B & 44.8 \\
PaLM & 62B  & 55.1 \\
PaLM-cont   & 62B & 62.8 \\
Chinchilla & 70B & 67.5 \\
\model & 65B & 63.4 \\
\midrule
OPT-IML-Max         & 30B & 43.2 \\
Flan-T5-XXL     & 11B & 55.1 \\
Flan-PaLM       & 62B & 59.6 \\
Flan-PaLM-cont & 62B & 66.1 \\
\model-I       & 65B & \bf 68.9 \\
\bottomrule
\end{tabular}
\caption{
\textbf{Instruction finetuning -- MMLU (5-shot).}
Comparison of models of moderate size with and without instruction finetuning on MMLU.
\label{tab:instruct}
}
\end{table}


In Table~\ref{tab:instruct}, we report the results of our instruct model \model-I on MMLU and compare with existing instruction finetuned models of moderate sizes, namely, OPT-IML~\cite{iyer2022opt} and the Flan-PaLM series~\cite{Chung2022ScalingIL}. 
All the reported numbers are from the corresponding papers.
Despite the simplicity of the instruction finetuning approach used here, we reach 68.9\% on MMLU.
\model-I (65B) outperforms on MMLU existing instruction finetuned models of moderate sizes, but are still far from the state-of-the-art, that is 77.4 for GPT \texttt{code-davinci-002} on MMLU (numbers taken from~\citet{iyer2022opt}).
The details of the performance on MMLU on the 57 tasks can be found in Table~\ref{tab:mmluapp} of the appendix.

%%%%%%%%%%%% bias and toxicity section

\input{bias}

\begin{table*}[t]
    \centering
    \begin{tabular}{l ccccc}
    \toprule
         & \multirow{2}{*}{GPU Type} & GPU Power &  \multirow{2}{*}{GPU-hours} & Total power  & Carbon emitted\\
         &  & consumption &       &  consumption & (tCO$_2$eq)\\
\midrule
OPT-175B    & A100-80GB & 400W & 809,472 & 356 MWh & 137 \\
BLOOM-175B  & A100-80GB & 400W & 1,082,880 & 475 MWh & 183\\
\midrule
\model-7B   & A100-80GB & 400W & 82,432 & ~36 MWh & ~14 \\
\model-13B  & A100-80GB & 400W & 135,168 & ~59 MWh & ~23 \\
\model-33B  & A100-80GB & 400W & 530,432 & 233 MWh & ~90 \\
\model-65B & A100-80GB & 400W & 1,022,362 & 449 MWh & 173 \\       
     \bottomrule
    \end{tabular}
    \caption{
    \textbf{Carbon footprint of training different models in the same data center.}
    We follow~\citet{wu2022sustainable} to compute carbon emission of training OPT, BLOOM and our models in the same data center. For the power consumption of a A100-80GB, we take the thermal design power for NVLink systems, that is 400W. We take a PUE of 1.1 and a carbon intensity factor set at the national US average of 0.385 kg CO$_2$e per KWh.
    \label{tab:cf}
    \vspace{-0.5em}
    }
\end{table*}


\section{Carbon footprint}

The training of our models have consumed a massive quantity of energy, responsible for the emission of carbon dioxide. 
We follow the recent literature on the subject and breakdown both the total energy consumption and the resulting carbon footprint in Table~\ref{tab:cf}.
We follow a formula for \citet{wu2022sustainable} to estimate the Watt-hour, Wh,  needed to train a model, as well as the tons of carbon emissions, tCO$_2$eq.
For the Wh, we use the formula:
$$\textrm{Wh} = \textrm{GPU-h}\times(\textrm{GPU power consumption}) \times \textrm{PUE},$$ where we set the Power Usage Effectiveness (PUE) at $1.1$.
The resulting carbon emission depends on the location of the data center used to train the network. For instance, BLOOM uses a grid that emits 0.057 kg CO$_2$eq/KWh leading to 27 tCO$_2$eq and OPT a grid that emits 0.231 kg CO$_2$eq/KWh, leading to 82 tCO$_2$eq.
In this study, we are interested in comparing the cost in carbon emission of training of these models if they were trained in the same data center.
Hence, we do not take the location of data center in consideration, and use, instead, the US national average carbon intensity factor of 0.385 kg CO$_2$eq/KWh.
This leads to the following formula for the tons of carbon emissions:
$$\textrm{tCO}_2\textrm{eq}=\textrm{MWh}\times0.385.$$
We apply the same formula to OPT and BLOOM for fair comparison.
For OPT, we assume training required 34 days on 992 A100-80B (see their logs\footnote{\url{https://github.com/facebookresearch/metaseq/tree/main/projects/OPT/chronicles}}).
Finally, we estimate that we used 2048 A100-80GB for a period of approximately 5 months to develop our models.
This means that developing these models would have cost around 2,638 MWh under our assumptions, and a total emission of 1,015 tCO$_2$eq.
We hope that releasing these models will help to reduce future carbon emission since the training is already done, and some of the models are relatively small and can be run on a single GPU.



\section{Related work}
\paragraph{Language models} are probability distributions over sequences of words, tokens or characters~\citep{shannon1948mathematical,shannon1951prediction}.
This task, often framed as next token prediction, has long been considered a core problem in natural language processing~\citep{bahl1983maximum,brown1990statistical}.
Because \citet{turing1950computing} proposed to measure machine intelligence by using language through the ``imitation game'', language modeling has been proposed as a benchmark to measure progress toward artificial intelligence~\citep{mahoney1999text}.

\paragraph{Architecture.} Traditionally, language models were based on $n$-gram count statistics~\citep{bahl1983maximum}, and various smoothing techniques were proposed to improve the estimation of rare events~\citep{katz1987estimation,kneser1995improved}.
In the past two decades, neural networks have been successfully applied to the language modelling task, starting from feed forward models~\citep{bengio2000neural}, recurrent neural networks~\citep{elman1990finding,mikolov2010recurrent} and LSTMs~\citep{hochreiter1997long,graves2013generating}.
More recently, transformer networks, based on self-attention, have led to important improvements, especially for capturing long range dependencies~\citep{vaswaniAttention2017,radford2018improving,dai2019transformer}.

\paragraph{Scaling.} There is a long history of scaling for language models, for both the model and dataset sizes.
\citet{brants2007large} showed the benefits of using language models trained on 2 trillion tokens, resulting in 300 billion $n$-grams, on the quality of machine translation.
While this work relied on a simple smoothing technique, called \emph{Stupid Backoff}, \citet{heafield2013scalable} later showed how to scale Kneser-Ney smoothing to Web-scale data.
This allowed to train a 5-gram model on 975 billions tokens from CommonCrawl, resulting in a model with 500 billions $n$-grams~\citep{buck2014n}.
\citet{chelba2013one} introduced the \emph{One Billion Word} benchmark, a large scale training dataset to measure the progress of language models.

In the context of neural language models, \citet{jozefowicz2016exploring} obtained state-of-the-art results on the Billion Word benchmark by scaling LSTMs to 1 billion parameters.
Later, scaling transformers lead to improvement on many NLP tasks.
Notable models include BERT~\citep{devlin2018bert}, GPT-2~\citep{radford2019language}, Megatron-LM~\citep{shoeybi2019megatron}, and T5~\citep{raffel2020exploring}.
A significant breakthrough was obtained with GPT-3~\citep{brown2020gpt3}, a model with 175 billion parameters.
This lead to a series of \emph{Large Language Models}, such as Jurassic-1~\citep{lieber2021jurassic}, Megatron-Turing NLG~\citep{smith2022using}, Gopher~\citep{rae2021goepher}, Chinchilla~\citep{hoffmann2022chinchilla}, PaLM~\citep{chowdhery2022palm}, OPT~\citep{zhang2022opt}, and GLM~\citep{zhengGLM2022}.
\citet{hestness2017deep} and \citet{rosenfeld2019constructive} studied the impact of scaling on the performance of deep learning models, showing the existence of power laws between the model and dataset sizes and the performance of the system.
\citet{kaplan2020scaling} derived power laws specifically for transformer based language models, which were later refined by \citet{hoffmann2022chinchilla}, by adapting the learning rate schedule when scaling datasets.
Finally, \citet{wei2022emergent} studied the effect of scaling on the abilities of large language models.

\section{Conclusion}

\looseness=-1 In this paper, we presented a series of language models that are released openly, and competitive with state-of-the-art foundation models. Most notably, \model-13B outperforms GPT-3 while being more than 10$\times$ smaller, and \model-65B is competitive with Chinchilla-70B and PaLM-540B. Unlike previous studies, we show that it is possible to achieve state-of-the-art performance by training exclusively on publicly available data, without resorting to proprietary datasets.
We hope that releasing these models to the research community will accelerate the development of large language models, and help efforts to improve their robustness and mitigate known issues such as toxicity and bias.
Additionally, we observed like \citet{Chung2022ScalingIL} that finetuning these models on instructions lead to promising results, and we plan to further investigate this in future work.
Finally, we plan to release larger models trained on larger pretraining corpora in the future, since we have seen a constant improvement in performance as we were scaling.



\section*{Acknowledgements}
\looseness=-1 We thank Daniel Haziza, Francisco Massa, Jeremy Reizenstein, Artem Korenev, and Patrick Labatut from the xformers team.
We thank Susan Zhang and Stephen Roller for their support on data deduplication.
We thank Luca Wehrstedt, Vegard Mella, and Pierre-Emmanuel Mazaré for their support on training stability.
We thank Shubho Sengupta, Kalyan Saladi, and all the AI infra team for their support.
We thank Jane Yu for her input on evaluation.
We thank Yongyi Hu for his help on data collection.


\bibliography{custom}
\bibliographystyle{acl_natbib}

\input{appendix.tex}


\end{document}
