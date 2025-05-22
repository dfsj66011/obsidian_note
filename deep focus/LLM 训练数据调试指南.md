
### 一、LLM 开发生命周期

![|600](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Ffa43347c-41d6-473e-8a46-095192476264_1934x682.png)

借助这些自动化指标，我们能在两次人工评估之间完成更多次模型迭代，从而更快提升模型质量（如下图所示）。

![|500](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F37f8440c-a1aa-4252-af9f-e8047f7dcf09_1354x654.png)

在自动评估方面，通常采用两种主要技术——*基准式评估和 LLM 评判*（详见下文）。这两种策略分别用于测试模型在封闭式任务和开放式任务中的表现。

![|500](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F00b3ca29-cb6d-472e-b4f4-69ad7875d503_938x418.png)

基准测试式评估（如选择题形式或问答对形式）在 NLP 研究史上一直被广泛使用。现代针对 LLM 的此类基准测试包括 MMLU 或 GPQA Diamond 等。这些基准测试具有封闭式答案，但大语言模型生成的开放式输出往往难以评估。目前最流行的开放式评估技术是 "LLM即评委" 方法，以及其他相关技术（如奖励模型、微调评委或验证器）。

**调整数据**。一旦建立了评估框架，我们便可以开始训练新模型并衡量其性能。针对每个新模型，我们会实施某些干预措施（期望能）提升 LLM 的表现。传统上，AI 研究者对算法和架构兴趣浓厚，有时我们确实会调整这些细节！例如，Llama-4 对其训练后流程进行了重大改动，许多大语言模型正在将 RLVR 等新算法纳入训练流程以增强推理能力。然而尽管存在这些新进展，*绝大多数干预措施仍与数据相关*。我们会调整训练数据，保持其他要素不变，重新训练（或继续训练）模型，进而观察新数据是否提升了模型性能。

从概念上讲，最直接的数据干预方式就是收集更多训练数据。在 LLM 开发过程中持续收集更多数据是常见做法。例如 Llama-2 技术报告指出，模型会经历多个阶段的后续训练，每个阶段都会为新的后续训练收集更多数据。虽然数据收集在概念上看似简单，但数据标注实际上是个极其复杂且充满细节的领域，需要正确的策略（通常还需*依赖前期经验*）才能成功实施，更多细节可参阅[此处](https://lilianweng.github.io/posts/2024-02-05-human-data-quality/)和[此处](https://eugeneyan.com/writing/labeling-guidelines/)。

> _““从人类数据中获取最大价值，需要迭代训练模型、不断演进且高度精细的数据指令、通过数据代工企业进行转化，以及其他诸多挑战。” —— [RLHF book](https://rlhfbook.com/c/06-preference-data.html)

![|500](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F6a8502ae-348d-47be-bd17-888dceb16c60_1598x826.png)

鉴于提升 LLM 质量的干预措施大多与数据相关，数据整理成为至关重要的议题——例如，已有数家初创公司和大量优秀论文聚焦这一领域。然而，尽管数据问题是 LLM 训练的基础环节，相关研究在 AI 领域却长期处于边缘地位。数据优化并非光鲜热门的话题，*但它往往是 LLM 训练成败的关键分水岭*。

#### 1.1 我们如何筛选数据？

简而言之，我们有两种整理数据的方法：

1. 直接查看数据。​ 
2. 利用模型输出来调试训练数据。

例如，我们可以通过人工检查或基础搜索与启发式方法来筛选和调试数据。此外，还可以借助另一个模型分析数据，例如打标签、分类、分配质量分数等。这些策略均与待构建的下游模型无关——我们直接针对训练数据本身进行操作。然而，在完成模型训练后，我们还能通过以下方式调试 LLM 的输出来进一步优化数据筛选流程：

* 识别低质量模型输出
* ​定位导致这些输出的（潜在）数据问题​
* 通过干预手段修复数据​
* 重新训练模型

**调试策略概述**。在本文中，我们将上述两种策略称为数据导向型与模型导向型数据优化。虽然可以用不同术语表述这些概念（当前命名也并非完美——例如数据导向型优化仍可能涉及模型使用，区别在于我们是用模型分析数据而非用数据训练模型），但为保持论述清晰一致，我们将沿用这套术语体系。

![|500](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fefd00ea7-16a3-49ba-9cd8-de68803dc606_1596x376.png)

在讨论这些观点时，我们需谨记：以数据为中心的调试和以模型为中心的调试并非互斥关系。事实上，我们几乎总应同时运用这两种方法。数据导向的整理工作无需训练任何模型，这在 LLM 开发的早期阶段极具价值。*经验丰富的科学家们总是在建模前投入大量时间分析与理解数据*。

随着时间的推移，我们会持续进行此类以数据为核心的分析，但一旦训练出模型，新的分析途径便成为可能。为了调试和改进我们的 LLM，我们必须开发一种多维度的方法，从而更深入地理解模型、数据以及二者间的关联。

### 二、以数据为核心的策展：聚焦数据

为了深入理解我们的数据，我们将从手动查看数据开始。通过人工检查数据，我们将逐渐发现并 *修复数据* 中的重要问题和模式。然而，为了将这种数据整理过程扩展到我们自身判断之外，我们需要使用基于启发式方法或其他机器学习模型的自动化技术。

**人工检查**。调试 LLM 的第一步就是仔细检查模型的训练数据。*这项工作应在模型训练开始前进行，并贯穿整个模型开发周期*。虽然人工数据检查非常耗时（且往往枯燥乏味），但这是 LLM 开发不可或缺的环节。通过人工检查数据，我们能更深入地理解数据特性，进而更好地把握模型表现。任何 LLM 研究者都会坦言，他们大部分时间都花在人工检查数据上。这项看似不起眼的工作，实则是成功训练 LLM 的关键要素——*既无法回避，也不应回避！*

人工数据检查的主要局限在于一个简单的事实：它不具备可扩展性——*研究人员能手动检查的数据量是有限的*。一旦我们通过足够的人工检查充分理解了数据，就需要制定更好的策略来扩展数据检查的规模。

**启发式过滤**。人工检查能揭示数据中的许多问题和有趣模式。例如，我们可能注意到某些词汇被异常频繁地重复使用([案例](https://www.reddit.com/r/ClaudeAI/comments/1fyk8ql/claude_ignores_its_own_system_prompts_with/))。为确保模型不会反映数据中这些次优模式，我们可以通过启发式方法匹配这些模式的训练样本并进行过滤（或修改）。例如，通过简单的字符串匹配就能发现重复使用相同词汇集的数据。这里我们运用基础启发式方法来解决数据中显著的缺陷。

在数据检查和筛选方面，我们还可以考虑许多其他启发式方法。例如，我们可能注意到某些数据源的质量更高或比其他数据源更具实用特性。为此，我们可以在训练过程中重点利用这些数据，甚至从该来源获取更多数据。同样，我们可能发现数据子集中存在可通过正则表达式识别或修复的格式问题。根据人工检查阶段的观察结果，训练数据集可能需要应用近乎无限数量的启发式检查或修正措施。

**基于模型的过滤**。若观测到的问题无法通过启发式方法解决，则可借助机器学习模型进行修正。fastText 分类器因其高效性被广泛用于 LLM 数据过滤——其甚至 *能胜任预训练规模的数据处理*。fastText 模型应用于 LLM 数据过滤的具体案例包括：语言识别（如过滤非英语数据）或识别有毒内容。此外，定制化的 fastText 模型能轻松训练以应对各类定制过滤任务，只需：i) 在目标数据样本上训练模型；ii) 使用模型识别此类数据；iii) 对识别出的数据执行保留或移除操作。

我们也可以使用其他类型的模型进行数据过滤。例如，LLM-as-a-Judge 类模型通常既用于数据过滤，也用于生成合成数据。Constitutional AI 就是利用 LLM 裁判生成合成偏好对的典型范例，而Llama-4 则采用 LLM 裁判从监督微调数据集中剔除简单样本。我们可以运用类似方法，以 *较高准确度识别数据* 中的任意属性和模式，从而实现过滤目的。

这类大型模型相较于 fastText 模型效率低得多，因此仅限于较小规模的应用场景（通常用于训练后阶段）。以比现代最大规模语言模型小约 10,000 倍的 BERT-base 模型为例，其与 fastText 模型在效率和硬件需求上的差距极为显著。尽管如此，开发更精密的数据整理方法和模型，仍是当前人工智能研究中最具影响力的课题之一。



### 三、模型导向的优化：调试 LLM 的输出

一旦我们开始基于自身数据训练 LLM，就能利用这些模型来调试训练数据集中的问题。模型导向的数据优化理念很简单，只需：

1. 识别模型生成的问题或错误输出；
2. 搜索可能导致这些输出的训练数据实例。

问题输出的识别通过我们的评估系统进行处理。我们可以通过人工检查（甚至我们自己动手！）手动识别低质量输出，或借助自动评估机制高效发现错误或低分输出。一旦这些有问题的输出被定位，调试 LLM 就转化为搜索问题——*我们需要找出可能与这些不良输出相关的训练样本*。本节将详细介绍几种常用方法，最终重点介绍由 AI2 团队最新开发的低成本高效数据溯源技术 OLMoTrace。

#### 3.1 在训练数据中搜索

![|500](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F407d89cc-a925-4b68-9d9b-5c0a2f563fec_1654x516.png)

寻找相关训练数据与其他搜索问题类似；可参考前文所述。唯一的区别在于，我们的查询来自 LLM 的输出，而非手动输入搜索框的内容。但所有相同的搜索技术均可用于解决此问题。本节将简要介绍搜索的核心概念及其在追溯训练数据中的应用。

**词汇检索**。在深度学习普及之前的许多年里，大多数搜索引擎都纯粹基于词汇匹配，这意味着它们依赖关键词（或 n-gram）匹配来查找与查询相关的文档。为了高效实现这种匹配，我们使用一种称为倒排索引的数据结构。通过统计查询与文档之间的匹配次数，并考虑所匹配的每个 n-gram 的独特性，我们可以为每个文档计算出一个相关性分数。最常用的算法是 BM25，其计算公式如下所示。

![|500](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F82315c90-36c0-46e7-92b9-7b77a34a5280_2250x644.png)

```python
from transformers import AutoTokenizer
from rank_bm25 import BM25Okapi

tok = AutoTokenizer.from_pretrained(<your tokenizer>)

corpus = [
    "Here is a training example",
    "Here is another training example...",
]

tokenized_corpus = [doc.split(" ") for doc in corpus]

bm25 = BM25Okapi(tokenized_corpus)
```

**语义搜索**。尽管词汇搜索功能强大且高效，但该技术仍依赖于关键词匹配——*这种框架无法捕捉语义匹配（即不同词汇表达相似含义）*。若要处理语义匹配问题，我们需要采用某种形式的向量搜索技术，具体方法如下所述。

![|600](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fd19f706e-65a8-4236-a652-d1bd5958e61c_2124x358.png)

在向量搜索中，我们使用嵌入模型为每个待检索文档生成嵌入向量。随后，将这些嵌入向量存储至向量数据库，借助 HNSW 等算法即可高效搜索相似向量。此时只需将查询语句嵌入，就能在索引中检索语义相似的文档——这正是 RAG 技术的工作原理：通过检索相关文本片段来增强 LLM 的上下文理解。

![|500](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F635a986a-9e9a-4a3a-8fd8-860017df9770_1802x786.png)

上文所述的语义搜索系统采用双编码器架构，该架构会为文档和查询分别生成独立的嵌入向量——*这些向量通过余弦相似度分数进行匹配*。此外，我们还可以使用交叉编码器，这种模型将文档和查询同时作为输入，直接输出单一的相似度分数。这两种策略的差异已在上图中进行可视化展示。目前公开代码库中提供了多种预训练的双编码器与交叉编码器模型，用户既可直接开箱使用，也可进行微调适配；更多细节请参阅此[链接](https://sbert.net/)。

现代搜索系统综合运用了所有这些技术。首先通过结合双编码器与（BM25）词法搜索的混合方法，高效检索出与查询最相关的文档。随后利用交叉编码器对检索到的文档进行细粒度排序，*将最相关的文档提升至列表顶部*。所有这些组件都可以根据搜索引擎使用过程中收集的数据进行微调，从而持续提升其准确性。

![|500](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fb72ab1d3-a9ee-42ea-ba81-8e03fa5f841e_1720x446.png)

**将搜索技术应用于调试**。既然我们已经理解了搜索系统的基础知识，现在也可以将这些思路应用于调试 LLM 的输出。然而，调试 LLM 输出时有两个独特的考量因素，使其不同于标准搜索应用场景： 

* LLM 的训练数据集可能极其庞大（数万亿 token 量级），这会限制某些技术的使用。  
* 根据具体应用场景，LLM 的输出及其训练文档可能非常冗长。

如果要追踪大规模数据集，使用向量搜索等技术 *虽然并非不可行*，但可能既耗时又昂贵。我们首先需要为整个数据集生成嵌入向量，然后将这些向量存储在向量数据库中以实现可搜索性。这一过程需要大量准备工作（包括构建大规模的数据管道！），因此准入门槛较高。

进一步而言，由于 LLM 的输出内容和训练文档可能非常冗长，我们需要以不同方式处理这一搜索问题。与其将完整输出作为搜索查询，我们更应关注输出中较短的片段，并在训练数据中寻找相似片段。理想情况下，我们希望开发一种训练数据追溯技术，需满足以下条件：

- 配置相对简单
- 能高效处理大规模数据集
- 可在较短片段层级上运行

#### 3.2 Infini-gram：将无限 n-gram LM 扩展至万亿级 tokens

> “我们没有采用预先计算n元组计数表的方法（这种方法的计算成本极高），而是开发了一个名为infini-gram的引擎——该引擎基于后缀数组构建——能够以毫秒级延迟计算∞-gram（以及任意n值的n-gram）概率。”

要理解如何高效追踪海量数据集，我们首先需要了解"无限元组"(infini-gram)的概念[1]。简而言之，无限元组是将n元组推广到任意大N值的广义形式。正如我们将看到的，用于计算无限元组概率的数据结构，同样能（极其高效地）定位和统计海量数据集中任意长度的文本片段。*这一特性对于以模型为中心的数据筛选和调试非常有用*！

**什么是n元语言模型？** n元语法（n-gram）本质上是由N个有序标记（或单词）组成的集合。给定一段文本序列，我们可以如上所示将其切分为n元组（本例中设定N=3）。若将整个文本数据集分解为n元组，通过统计特定n元组在数据集中出现的频次，即可直接计算出该n元组的概率，具体方法如下所示。

![|500](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F5a6e7f4c-5367-4a52-a0e7-53de8f0b45f6_2366x622.png)

所有这些计数通常会被预先计算并存储在计数表中，这使得我们可以快速查找 n-gram 概率并评估上述表达式。实际上，我们可以利用 n-gram 概率构建一个简单的语言模型！要使用 n-gram 预测序列中的下一个词元，只需：

1. 查看序列中最后 N-1 个词元。​
2. 获取每个可能 n-gram 在给定前 N-1 个词元时的概率。​
3. 与其他语言模型类似，对下一个词元进行采样。

**n元语言模型的局限性**。从实际应用来看，n元语言模型在文本生成方面表现欠佳——*仅依靠统计n元词频无法构建强大的聊天机器人*。虽然这对任意N值都成立，但制约n元模型性能的核心问题在于：n元统计表的规模会随N值呈（近似）指数级增长。因此大多数n元模型仅能采用较小N值（例如*常用设定N=5*），这导致其捕捉长距离语义关联的能力较弱，具体表现如下所述。

![|600](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F2286afc0-b048-419f-95f1-2e152ff94137_1884x942.png)

此外，n元语言模型还面临稀疏性问题。某些n元组可能未在训练数据中出现，迫使我们回退到更小的n元组来计算概率——*这一概念通常被称为n元"回退"*。在实际回退到更小n元组时，如何构建有效的概率估计其实相当复杂。

**让n-gram模型重焕新生**。作者提出了一种n-gram语言模型的变体——称为infini-gram（或∞-gram）——能更好地适配现代大语言模型。相较于标准n-gram模型，infini-gram进行了两大关键改进：

1. 像其他现代大语言模型一样基于海量文本数据（数万亿token）训练，从而缓解数据稀疏性问题；
2. 在计算n-gram概率时，N值可无限增大，从而捕捉数据中更具意义的概率分布。

**什么是∞-gram（无限元组）**？通过上述改进，infini-gram模型解决了前文讨论的传统n元语言模型的两大核心缺陷。其运作原理如下：假设存在文本序列w，要计算其中第i个标记的infini-gram值时，我们会考察该标记之前的所有标记（如下图所示）。

![|500](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fc2004290-7717-47cd-896a-46acc539811f_1882x698.png)

在这个等式的左侧，无限元概率是以序列的整个先前上下文为条件的，这与之前不同。然而，等式的右侧与n元概率完全一致！*n元与无限元的关键区别在于我们如何选择N的值*。

对于n元语法而言，N是一个（固定的）超参数。而无限元语法则采用回退机制动态选择N值。具体来说，我们先用最大可能的N值（即序列中所有前置词元）测试该表达式的分母，并持续递减N值直至分母不为零；详见下文说明。

> “一旦分母变为正数，我们便停止回退，此时分子可能仍为零……实际有效n值等于1加上提示词在训练数据中出现的最长后缀长度。”

如果将w’定义为w的子序列（包含第i-1个标记），那么这种回退方法就是寻找数据集中存在的w’的最长后缀。然后，我们使用通过回退找到的N值，按照之前的标准n-gram概率表达式来计算infini-gram概率。

计算∞-gram概率。为了计算无限元语法概率，我们无法像之前那样预先计算计数并将其存储在表格中。由于N值无界，且无限元语法是在[1]中基于LLM规模数据集训练的——*这样的计数表规模将极其庞大*。为此，我们采用一种称为后缀数组的数据结构，构建高效计算无限元语法概率的引擎。

![|300](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F069cd044-9bb4-4bcf-bb11-511d82c54341_729x583.png)

上图展示了后缀数组的概念。给定长度为L的文本序列w，构建后缀数组的步骤如下：

1. 提取该序列的所有后缀（共L个）。
2. 按字典序对这些后缀进行排序。
3. 将排序后每个后缀的原始索引存入列表——此列表即为后缀数组！

假设w’是w中任意一个从第i个词到第j个词构成的子串（i < j）。由于后缀数组是按字典序排序的，任何以w’开头的后缀都会连续存储在该数组中。利用这一特性，我们可以高效计算w’在w中的出现次数：只需找到后缀数组中第一个和最后一个以w’为前缀的后缀索引，两者之差即为w’的出现频次。通过计算w’的频次，我们就能推导任意无限元组概率——*该操作既可用于确定N值，也能计算无限元组概率表达式中的分子与分母频次！*

![|400](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F25d05a53-38fd-4c7d-a477-0b8bd256443d_1053x575.png)

**面向大语言模型的∞-gram方法。​**​ 在大语言模型的语境中，我们的序列w是整个经过分词的训练数据集，其中文档边界通过固定的分隔符标记（如上所述）。这一序列规模庞大——现代大语言模型的训练数据可达数万亿token量级——但后缀数组能有效处理此类规模的数据。

> “在推理过程中，整个无限元索引可以完全驻留在磁盘上，从而最大限度地减少所需的计算资源（无需GPU，且CPU/内存占用极低）……我们最优化后的无限元引擎能在平均20毫秒内完成给定n元组的频次统计。对于n元语言模型，它可在40毫秒内计算出概率及下一词分布；若采用∞元模型，则耗时200毫秒。”

例如，文献[1]中构建在一个5万亿token数据集上的后缀数组消耗了约35TB内存。构建该后缀数组耗时约48小时，且整个后缀数组创建后（即使计算无限元组概率时）均可存储在磁盘中。由此产生的无限元组引擎能够为超过两万亿个独特n元组计算概率。然而，在此规模的数据集上检索给定n元组的计数仍仅需约20毫秒！

在实践中使用∞-gram模型。要完全理解infini-gram背后的理念需要一些时间。幸运的是，整个infini-gram项目——与Ai2的其他项目一样——是完全开源的！Python中有大量开源工具可用于处理infini-gram。详见项目官网获取完整信息。

```python
%pip install infini_gram 
python -m infini_gram.indexing 
    --data_dir <path to data>
    --save_dir <path to save index>
    --tokenizer llama  # also supports gpt2 and olmo
    --cpus <cpus available>
    --mem <memory available (in Gb)>
    --shards 1  # increase if N > 500B
    --add_metadata 
    --ulimit 1048576
```

与此概述最相关的工具是inifini-gram Python包。该包已预置了多个开源LLM训练数据集，同时我们也能通过上述命令为自定义数据集构建infini-gram索引。索引就绪后，即可使用该Python包高效运行多种搜索与计数操作（示例见下文，更多细节请参阅此处链接）。

```python
from infini_gram.engine import InfiniGramEngine
from transformers import AutoTokenizer

# instantiate tokenizer (must match tokenizer used for indexing)
tokenizer = AutoTokenizer.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    add_bos_token=False,
    add_eos_token=False,
)

# connect to infini-gram engine
engine = InfiniGramEngine(
    index_dir=<path to index>,
    eos_token_id=tokenizer.eos_token_id,
)

# sample n-gram / sequence
inp = "This is my sample n-gram sequence."
inp_ids = tokenizer.encode(inp)

# find matching n-grams in dataset
result = engine.find(input_ids=input_ids)

# n-gram count
result = engine.count(input_ids=inp_ids)

# n-gram probability
result = engine.prob(
    prompt_ids=inp_ids[:-1],
    cont_id=inp_ids[-1],
)

# next token distribution
result = engine.ntd(prompt_ids=inp_ids)

# infini-gram probability
result = engine.infgram_prob(
    prompt_ids=inp_ids[:-1],
    cont_id=inp_ids[-1],
)
```

#### 3.3 OLMoTrace: Tracing Language Model Outputs Back to Trillions of Training Tokens
![|500](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F5aedfd0f-700f-41bd-85b3-3eff8c3ae7dd_1232x810.png)

OLMoTrace [2]开创了一种创新方法，能高效地将大语言模型的输出结果溯源至其训练数据中的具体示例。该技术已部署于Ai2 playground平台（如上图所示），可在数秒内完成追踪，检索出与大语言模型输出相关的训练文档。考虑到大语言模型的训练数据规模极其庞大，人们或许会质疑这种实时追踪如何实现。幸运的是，答案早已揭晓——正是infini-grams技术！

> “OLMOTRACE的目的是为用户提供一个工具，用以探索语言模型可能从何处习得生成特定词序列的能力，其重点在于通过逐字匹配来建立模型输出与训练数据之间最直接的联系。“

**追踪策略**。OLMoTrace 的核心思想是寻找模型输出与其训练数据集中共有的长且独特的 token 序列实例。给定提示词和 LLM 生成的响应作为输入，OLMoTrace 将返回以下内容：

- 一组在 LLM 响应中发现的显著文本片段。
- 与每个响应片段相关联的、来自 LLM 训练数据的最相关文档片段列表。

与向量搜索不同，模型输出与训练数据之间的匹配必须完全一致。通过后缀数组可以快速识别完全相同的词元匹配，正如上一节所讨论的。然而，要确保识别并返回最佳匹配文档，需要在标准无限元语法功能基础上构建一个四步算法。

**（步骤1）最大匹配文本段**。在对 LLM 的响应进行分词后，我们找出该响应中所有满足以下三个属性的文本段：

1. *存在性*：该文本段在训练数据中存在完全匹配项。
2. *最大性*：该文本段不是其他匹配文本段的子段。
3. *完整性*：该文本段不残缺；例如，不以残缺单词开头或结尾，或中间不包含断开的标点符号。

这些特性在下图中得以展示。如图所示，图中存在三个匹配的文本片段。然而，*除标绿* 的一个片段外，其余片段均因不符合以下任一条件而被移除：i) 非最大覆盖范围；ii) 非自包含结构。

![|500](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F57e984b7-bba7-44d3-a877-c3d74693180d_2178x642.png)

“直接计算最大跨度在算法上是低效的，但文献[2]的作者提出了一种更高效的算法，该算法依赖于infini-gram索引中的find操作。给定一个token序列作为输入，find操作会返回：

- 索引中匹配跨度的计数。
- 一个可用于查找匹配数据跨度的段范围。”

然而，如果返回的计数为零——*表明我们的数据中不存在与该序列完全匹配的项*——查找操作仍会返回一个（空的）分段范围。由于后缀数组是按字典序排列的，该范围的索引对应于数据集中与该序列最长匹配的前缀。

```python
# run find operation with infini-gram engine
result = engine.find(input_ids=inp_ids)

"""
### .find() output example (match): 
    {
        'cnt': 10,
        'segment_by_shard': [(13693395, 13693405)],
    }

### .find() output example (no match):
    {
        'cnt': 0,
        'segment_by_shard': [(85267640, 85267640)],
    }
"""

# lookup training documents from .find()
rank_start, rank_end = result['segment_by_shard'][0]
ranks = [r for r in range(rank_start, rank_end)]
for r in ranks:
    docs = engine.get_doc_by_rank(
        s=0,  # assumes suffix array has a single shard
        rank=r,
        max_disp_len=len(inp_ids) * 5,  # size of doc chunk
    )
    doc_text = [tokenizer.decode(d['token_ids']) for d in docs]
    print(f'Number of documents: {len(docs)}')
    print(f'Matching document: {doc_text[0]}')
```

[2]利用find操作的这一特性构建了一种高效的跨度匹配算法。如下图所示，该算法通过为输入序列的每个后缀运行一次find操作，从而得到每个后缀对应的最长匹配前缀。当所有匹配跨度被识别后，我们可再次遍历该列表，剔除非最大或非自包含的匹配跨度。

![|500](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fb435fcab-0df9-4a07-b6f4-fc7c07e646d7_2314x804.png)

**（步骤2）跨度过滤**。如果按照上述方法计算出的最大跨度列表过长，我们需要采用某种策略来识别其中最有用且相关的跨度。为此，文献[2]的作者根据跨度的单字概率（越低越好）——即跨度内各词条单字概率的乘积——对跨度进行评分。特定词条的单字概率通常已预先计算并存储在缓存中，其计算方法如下所示。

![|200](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Feb6f7bb6-632c-40d1-bfbf-db4f585a4969_1090x466.png)

在文献[2]中，作者根据跨度的单字概率对跨度进行排序，并仅保留该列表中前K个跨度，其中K = ceil(0.05 x L)，L表示序列长度。

**（步骤3-4）合并文本片段并获取文档**。为避免信息冗余，OLMoTrace会将重叠的文本片段进行合并，并为每个最终合并后的片段检索相关文档。但由于每个片段关联的文档数量可能过多，需进行二次筛选；例如文献[2]的作者会为每个片段保留十份文档。为确定最相关的文档，可通过计算大语言模型输出与检索文档之间的BM25评分进行排序。

> _“为了优先展示最相关的文档，在文档面板中我们按BM25分数降序对所有文档进行排序。每篇文档的BM25分数是通过将检索到的文档集合视为语料库，并将用户提示与语言模型响应的拼接作为查询来计算的。”——引自[2]

![|500](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F250f798c-fb39-46d7-83c4-da76cbbeccda_2150x1062.png)

**示例实现**。OLMoTrace的推理流程如上图所示。为了更好地理解其工作原理，我们将使用Python中的infini-gram包（快速）实现核心功能。要构建infini-gram索引，需要将所有LLM训练数据放入单一目录中。infini-gram包要求数据格式为一个或多个.jsonl文件，每个文件包含text和metadata字段（如下所示）。该.jsonl文件的每一行对应训练数据集中的一个独立文档。

```json
{
    'text': 'This is a training sequence for our LLM...',
    'metadata': {
        'source': <url>,
        'category': 'general',
        'year': 2025,
        ...
    },
}
```

当数据完成格式化后，我们即可按照前述方法构建infini-gram索引。此外，OLMoTrace还要求预先计算所有标记的单字概率。这两个步骤的具体实现如下所示。本代码默认使用Llama 2分词器进行追踪，且假设infini-gram索引仅需单个分片。实际应用中可根据需求更换底层分词器，若处理超大规模数据集（即超过5000亿标记）时，则可能需要支持多分片索引。

```python
import os
import json
from collections import Counter
import tempfile

from transformers import AutoTokenizer

# load tokenizer / data
enc = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", add_bos_token=False, add_eos_token=False)
data_rows = [{'text': 'here is some training data'}, ...]

# compute / save unigram probabilities
all_toks = []
for x in data_rows:
    all_toks.extend(enc.encode(x['text']))
total_toks = len(all_toks)
tok_count = Counter(all_toks)
unigram_probs = {}
for tid in tok_count:
    cnt = tok_count[tid]
    unigram_probs[tid] = cnt / total_toks
with open(<save path>, 'w') as json_file:
    json.dump(unigram_probs, json_file, indent=4)

# build infinigram index
data_dir = <path to data>
save_dir = <save index here>
temp_dir = tempfile.TemporaryDirectory()
command = (
    f"python -m infini_gram.indexing --data_dir {data_dir} "
    f"--temp_dir {temp_dir.name} --save_dir {save_dir} "
    f"--tokenizer llama --cpus 12 --mem 64  --shards 1 "
    f"--add_metadata --ulimit 100000 "
)
print(command)
os.system(command)
temp_dir.cleanup()
```

既然已构建完成infini-gram索引，我们便能按照[2]中OLMoTrace提出的算法，在训练数据集上追踪文本序列——如下方代码所示。该代码不仅返回一组文本片段，还会返回这些片段在训练语料库中关联的文档及其元数据。

```python
import ast
import math
import random

from infini_gram.engine import InfiniGramEngine
from transformers import AutoTokenizer

def compute_longest_prefix(query, doc):
    """helper function for computing longest prefix of query that exists
    within a document"""

    def shared_prefix_length(list1, list2):
        prefix_length = 0    
        for elem1, elem2 in zip(list1, list2):
            if elem1 == elem2:
                prefix_length += 1
            else:
                break
        return prefix_length

    first_id = query[0]
    start_idx = [index for index, value in enumerate(doc) if value == first_id]
    longest_prefix = 0
    for si in start_idx:
        longest_prefix = max(
            longest_prefix,
            shared_prefix_length(query, doc[si:]),
        )
    return longest_prefix

# setup
enc = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", add_bos_token=False, add_eos_token=False)
engine = InfiniGramEngine(index_dir=<path to index>, eos_token_id=enc.eos_token_id)
unigram_probs = {1: 0.5, 2: 0.5} # load pre-computed probabilities

# LLM output / query to search
generation = 'Here is the output of the LLM that we want to search for in our data.'
gen_ids = enc.encode(generation)


"""
Step One: find maximal matching spans
"""
L = len(gen_ids)
max_doc_toks = len(gen_ids) * 2  # size of spans to retrieve in documents

# find longest prefix match for every suffix in the query
spans = []
for start in range(len(gen_ids) - 1):
    _suffix = gen_ids[start:]
    _suff_res = engine.find(input_ids=_suffix)

    # if no match, get the longest matching prefix using find result
    if _suff_res['cnt'] == 0:
        _shards = _suff_res['segment_by_shard']
        assert len(_shards) == 1  # assume only one shard
        _doc_ids = engine.get_doc_by_rank(
            s=0,  # assume only one shard
            rank=_shards[0][0],
            max_disp_len=max_doc_toks,
        )['token_ids']
        matched_toks = compute_longest_prefix(_suffix, _doc_ids)  # get longest matching prefix
    elif _suff_res['cnt'] > 0:
        matched_toks = len(_suffix)
    spans.append((start, start + matched_toks))

# remove partial and non-self-contained spans
full_spans = []
for start, end in spans:
    span_ids = gen_ids[start: end]
    span_text = enc.decode(span_ids)

    # check for internal punctuation
    has_internal_punc = False
    punc_chars = "!.?\n"
    for ch in span_text[:-1]:
        if ch in punc_chars:
            has_internal_punc = True
            break
    if has_internal_punc:
        continue

    # check if first token is a continuation of a word
    first_tok_id = span_ids[0]
    first_tok = enc.convert_ids_to_tokens(first_tok_id)
    if first_tok[0] != '▁':  # assumes Llama 2 token format
        continue

    # no sub-token follows the last token
    if end < len(gen_ids) and tokenizer.convert_ids_to_tokens(gen_ids[end])[0] != "▁":
        continue
    full_spans.append((start, end, span_ids, span_text))    

# remove non-maximal spans
maximal_spans = []
max_end_pos = -1
full_spans = sorted(full_spans)
for start, end, ids, text in full_spans:
    if end > max_end_pos:
        maximal_spans.append((start, end, ids, text))
        max_end_pos = end


"""
Step Two: filter to keep long / unique spans
"""
K = math.ceil(0.05 * L)
assert K > 0
filt_spans = []
for start, end, ids, text in maximal_spans:
    span_uni_prob = [unigram_probs.get(_id) for _id in ids]
    span_uni_prob = math.prod(span_uni_prob)
    filt_spans.append((start, end, ids, text, span_uni_prob))
filt_spans = sorted(filt_spans, key=lambda x: x[-1])
filt_spans = filt_spans[:K]
filt_spans = sorted(filt_spans)  # sort based on start position again


"""
Step Three: retrieve Enclosing Docs
"""
docs_per_span = 10
span_to_docs = defaultdict(list)
for i, (start, end, ids, text, uni_prob) in enumerate(filt_spans):
    # run retrieval in infinigram index to get documents
    span_res = engine.find(input_ids=ids)
    assert span_res['cnt'] > 0
    assert len(span_res['segment_by_shard']) == 1  # assume only one shard

    rank_start, rank_end = span_res['segment_by_shard'][0]
    ranks = [r for r in range(rank_start, rank_end)]
    if len(ranks) > docs_per_span:
        # retrieve fixed number of documents for each span
        ranks = sorted(random.sample(ranks, docs_per_span))

    # NOTE: we can instead rank documents by BM25 score here!
    for r in ranks:
        _doc = engine.get_doc_by_rank(
            s=0,
            rank=r,
            max_disp_len=max_doc_toks,
        )
        _doc_meta = ast.literal_eval(_doc['metadata'])['metadata']
        _doc_text = enc.decode(_doc['token_ids'])
        _doc_data = {
            "text": _doc_text,
            **_doc_meta
        }
        span_to_docs[i].append(_doc_data)

        
"""
Step Four: merge overlapping spans
"""
# get indices of spans to merge together
merged_spans = [[0]]
curr_idx = 0
curr_start = filt_spans[0][0]
curr_end = filt_spans[0][1]
for i, next_span in enumerate(filt_spans[1:]):
    start = next_span[0]
    end = next_span[1]
    if start < curr_end:
        curr_end = max(curr_end, end)
        merged_spans[curr_idx].append(i + 1)
    else:
        curr_start, curr_end = start, end
        curr_idx += 1
        merged_spans.append([i + 1])
        assert len(merged_spans) == curr_idx + 1

# merge spans into a final set
final_spans = []
for ms in merged_spans:
    all_docs = []
    docs_per_merged_span = math.ceil(docs_per_span / float(len(ms)))  # subsample docs for spans being merged
    for i in ms:
        # take top docs from each span being merged
        all_docs.extend(span_to_docs[i][:docs_per_merged_span])
    _spans = [filt_spans[i] for i in ms]
    start = min([x[0] for x in _spans])
    end = max([x[1] for x in _spans])
    text = enc.decode(gen_ids[start: end])
    final_spans.append({
        "start": start,
        "end": end,
        "text": text,
        "docs": all_docs,
    })


"""
Step Five: observe tracing results
"""
docs_to_print = 5
print(f'Query Text: {enc.decode(gen_ids)}')
for i, sp in enumerate(final_spans):
    print("\n" + "="*20 + f" SPAN {i + 1} / {len(final_spans)} " + "="*20)
    print(f"Span Text: {sp['text']}\n")
    for j, doc in enumerate(sp['docs']):
        print("-"*10 + f" Document {j + 1} / {len(sp['docs'])} " + "-"*10)
        for k in ['text', 'movie_id', 'src_lang', 'start_frame', 'end_frame']:
            if k == 'text':
                v = doc[k].replace('\n', ' ')
            else:
                v = doc[k]
            print(f"- {k} --> {v}")
```

由此可见，OLMoTrace的核心功能并不复杂——大部分复杂代码已被infini-gram包抽象封装！对于感兴趣的读者，我强烈建议您在自己的模型和数据上测试这段代码，亲身体验它能返回哪些类型的结果！

![|500](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F332e82aa-8b1d-4c48-8baf-13d820ba8e81_1840x432.png)

**OLMoTrace的应用**

OLMoTrace专长于发现大语言模型输出与其训练数据之间完全匹配的长且独特的片段。这些精确匹配是定位可能影响模型特定输出的训练数据的重要依据。文献[2]探讨了多种应用场景：

- 事实核查：将模型生成的事实陈述与训练数据中的类似陈述进行比对。
- 创意表达：验证模型的"创意"输出是真正原创，还是直接从训练数据复制而来。
- 推理能力：检测模型是否沿用了训练数据中可验证问题（如数学题）的解题推理过程。

在以上每种情况中，我们都能通过追踪模型的输出来定位训练数据中具有显著逐字匹配的区域，从而对大型语言模型有新的认识。

#### 3.4 推理模型与未来研究
![|600](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F1f3ea8fb-4672-4580-b9e5-6f9520114cf0_2344x498.png)

**推理模型的扩展。​**​ 如上所述，LLM 通常经过多个阶段的训练，每个阶段采用独特的数据形式：

* 监督微调（SFT）：通过具体的提示-响应对示例训练模型，使其学会复现给定的响应模式。
- 基于人类反馈的强化学习（RLHF）：利用偏好对（即同一提示下两个响应中标注出更优选项的数据）调整模型行为。
- 可验证奖励的强化学习（RLVR）：采用纯强化学习机制，当模型通过基于规则（通常为确定性）的验证函数正确解决问题时给予奖励。

尽管存在这些独特的数据格式，我们只需最小改动即可将OLMoTrace应用于训练的每个阶段！我们可以轻松地在监督学习样本和偏好对上构建无限元语法索引（但可能需要区分处理偏好对中的正负补全结果）。然而对于RLVR任务，*我们可能需要更深入地思考数据追踪的实现方式*。

![|500](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Ffb865992-1eee-4fdb-b98a-165f4d555e11_1774x608.png)

在使用RLVR训练大语言模型时，我们拥有一个包含可验证解决方案的问题数据集，例如已知答案的数学问题或带有测试用例的编程问题。我们可以轻松判断模型是否正确解决了这些问题（通过字符串匹配或更鲁棒的方法），如前文所述。随后，模型通过大规模强化学习驱动的自我进化过程，自主掌握解决这类问题的能力，这一机制已在DeepSeek-R1 中得到实证。

> “我们探索大型语言模型在无监督数据条件下发展推理能力的潜力，重点关注其通过纯强化学习过程实现自我进化的能力。”

在强化学习训练过程中，我们从文献[7]中发现，大型语言模型会输出复杂的思维链——有时长达数千个标记！——以此提升推理能力。然而若要对这些推理轨迹建立索引，就会遇到一个有趣的问题：这些推理轨迹实际上并非训练数据的一部分，而是大型语言模型在强化学习训练过程中自行生成的。

![|500](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F36e006bb-5959-485b-bb4a-d45b235a8a9d_1800x1004.png)

类似地，大型语言模型生成的补全结果会通过奖励模型进行排序，并用于强化学习人类反馈（RLHF）阶段的策略更新（具体说明可参见此处）。若需捕捉强化学习训练过程中习得的模式（包括RLHF和RLVR），就必须持续记录训练期间模型生成的所有补全内容。获得这些补全数据后，我们可以像处理常规训练数据一样为其建立索引，将其纳入无限元文法索引体系，并通过OLMoTrace工具进行追踪。

**相关（及未来）研究**。尽管OLMoTrace具有实用性，但精确匹配并不能保证因果关系——大语言模型生成某个输出可能存在多种原因。仅因为我们找到了与大语言模型输出相似的训练数据，并不意味着这些数据必然导致了该输出。

为深入理解大语言模型的输出机制，多个并行研究方向正在探索可解释性的替代策略。例如，近期有多篇论文研究如何教导大语言模型在生成内容时引用来源[8,9,10]，具体如下。

![|500](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Ff79a885a-b083-4dc6-b86d-33001a12fd90_1278x838.png)

这种引用来源的能力可以被整合进大语言模型的标准训练流程中——例如预训练[8]或基于人类反馈的强化学习（RLHF）[9]——从而使模型学会何时以及如何为答案提供依据。然而，仍然无法保证这些引用能真实解释输出结果的生成过程。

机制可解释性领域致力于研究神经网络的内部结构，以理解其产生特定输出的原因。尽管深度神经网络通常被描述为黑箱系统，但在微观层面（即小规模权重组合）进行研究时，我们可以发现这些网络中存在着大量重复的电路结构和特征单元。例如，视觉神经网络往往包含专门用于检测曲线、边缘等视觉特征的独立功能单元。

![|500](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F16d01ddd-a81c-442b-bccd-0f8af4d3c5ca_2200x1660.webp "Abstract Feature Examples")
机械可解释性这一课题主要由Anthropic公司推动普及。在最近的一份研究报告中，学者们通过字典学习方法对Claude Sonnet模型中的特征进行了大规模研究。如上所述，该研究发现了数百万个对应高级概念的特征，例如人物、地点、代码错误等。

>“我们已经识别出数百万个概念在Claude Sonnet（我们已部署的大型语言模型之一）内部是如何表征的。这是首次对现代生产级大型语言模型内部结构进行详细解析。”

此外，作者们分析了特征之间的“距离”，并发现了一些有趣的性质；例如，金门大桥的特征与恶魔岛的特征非常接近。这类研究虽然尚处于起步阶段，但可以说是真正理解大语言模型为何及如何产生特定输出的最有前景的途径。

### 四、结论

正如我们所知，优化训练数据集是大语言模型训练过程中最具影响力和最重要的环节之一。为了有效筛选和调试数据，我们应当首先从数据本身入手——而非直接训练模型！第一步需要人工检查数据，充分理解其各项属性、规律及特性。为扩展人工检查的规模，我们可以结合启发式方法（在可行时）与机器学习模型（如fastText或大语言模型评判器）。这种以数据为核心的优化流程，着重于在训练大语言模型之前修复问题并提升数据质量。


> “我注意到一个规律：优秀的AI研究者都愿意手动检查大量数据。不仅如此，他们还会构建能快速手动检查数据的基础设施。虽然不够光鲜，但手动检查数据能提供对问题有价值的直觉。”——Jason Wei

一旦我们开始训练大语言模型（LLMs），就可以利用模型的输出来发现数据中的问题。具体而言，我们可以：

1. 通过评估框架识别有问题的模型输出；​ 
2. 将这些输出追溯到训练数据中的对应部分。​

虽然我们可以使用标准搜索技术（如词法搜索或向量搜索）来追踪数据，但像OLMoTrace[2]这类专门为大型语言模型开发的追踪技术更为适用。这类技术部署简单快捷、信息量丰富，且能扩展到任意规模的数据集，使其成为调试LLM训练数据集时极具实用价值的选择。


