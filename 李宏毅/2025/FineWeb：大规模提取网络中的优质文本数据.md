
* Authors：Guilherme Penedo, Hynek Kydlíček, Loubna Ben Allal, Anton Lozhkov, Colin Raffel, Leandro Werra, Thomas Wolf
* Published：May 31, 2024
* Reading time: 45 min

LLM 的性能在很大程度上取决于其预训练数据集的质量和规模。然而，像 Llama-3 和 Mixtral 这样的最先进开源 LLM 的预训练数据集并未公开，而且关于它们是如何创建的知之甚少。

最近，我们发布了 [🍷FineWeb](https://huggingface.co/datasets/HuggingFaceFW/fineweb)，这是一个用于 LLM 预训练的新的大规模（*15T tokens，44TB*）数据集。FineWeb 源自 96 个 CommonCrawl 快照，并且*比其他开放预训练数据集能产生性能更好的 LLM*。为了在机器学习领域提供更清晰的认知，并推进关于如何训练高质量 LLM 的开放性理解，我们仔细记录并分析了 FineWeb 中使用的所有设计选择，包括对去重和过滤策略的深入探究。本长篇报告深入探讨了如何为 LLM 预训练创建一个大规模且高质量的网络规模数据集。

在本报告中，我们还介绍了 [**📚 FineWeb-Edu**](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)，它是 FineWeb 的一个子集，是利用可扩展的自动化高质量注释构建而成，具有教育价值，并且在 MMLU、ARC 和 OpenBookQA 等多个教育基准测试中的表现优于所有可公开获取的网络数据集。[**📚 FineWeb-Edu**](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) 有两种规模/过滤级别：*1.3T（教育内容极高）和 5.4T（教育内容高）tokens*。

**摘要：** 这篇博客讨论了大规模处理和评估数据质量的相关内容，🍷 FineWeb 配方，以及创建其 📚 FineWeb-Edu 子集所遵循的过程。

## 一、网络数据

### 1.1 寻找原始数据

关于用于训练大语言模型的网络数据集，人们常问的一个常见问题是：“他们究竟从哪里获取这么多数据？”。通常有两种选择：

- 要么你自己抓取数据，就像 OpenAI 或 Anthropic（以及其他一些公司）所做的那样（见[此处](https://platform.openai.com/docs/bots)和[此处](https://darkvisitors.com/agents/claudebot)）；
- 要么使用公共的网络爬取网页库，比如由非营利组织 [CommonCrawl](https://commoncrawl.org/) 维护的那个。

为了构建 🍷 FineWeb，我们遵循了许多 LLM 训练团队过去所采用的方法，以 CommonCrawl（CC）作为起点。自 2007 年以来，Common Crawl 非营利组织一直在对网络进行爬取，并且通常每隔 1 到 2 个月就会发布一次新的爬取结果，其中包含通过自动网络爬取获得的 200 到 400T 的文本内容。

例如，最新的 CC 抓取（2024 年 4 月）包含 27 亿个网页，总计 386T 未压缩的 HTML 文本内容[^1]。自 2013 年以来已发布了 96 次抓取，2008 年至 2012 年期间发布了 3 次抓取，这些抓取采用不同的（较旧的）格式[^2]。

### 1.2 大规模处理

鉴于所涉及的数据量巨大，我们必须克服的主要挑战之一是拥有一个模块化、可扩展的代码库，以便我们能够快速处理我们的处理决策并轻松尝试新想法，同时适当地并行化我们的工作负载，并提供对数据的清晰洞察。

为此，我们开发了 [`datatrove`](https://github.com/huggingface/datatrove)[4]，这是一个开源数据处理库，使我们能够将过滤和去重设置无缝扩展到数千个 CPU 核心。创建 🍷 FineWeb 所涉及的所有数据处理步骤都使用了这个库。您将在 [datatrove 存储库](https://github.com/huggingface/datatrove/blob/main/examples/fineweb.py)中找到我们使用的确切脚本。

### 1.3 什么是优质数据？

这可能是创建数据集时需要牢记的主要问题。在大多数情况下，特别是在大型语言模型预训练的背景下[^3]，“高质量”并不是一个定义非常明确的术语[5] [6]，甚至不是仅通过直接的人类观察就能始终清晰感知到的文档属性[7]。

在给定被认为是“干净”的语料库（通常是维基百科[^4]）上训练模型，并使用它来检查我们试图整理的数据集的困惑度，这种方法仍然很常见[8]。不幸的是，这并不总是与我们在一系列感兴趣的下游任务上性能的提升相关[9]，因此，另一种常用的方法是在我们数据集的一个代表性子集上训练小型模型 [^5]，并在一组评估任务上对它们进行评估。之所以使用小型模型，是因为训练成本和时间与模型大小成正比。在第二种方法中，选择一组多样且有代表性的数据集-评估任务非常重要，并且尽量不要对任何一个单独的基准测试过度拟合，因为这可能会损害预训练后获得的 LLM 的通用性。

比较不同数据集的另一种方法是在每个数据集上训练一个模型，然后让人类对模型的生成结果进行评分和比较（就像在 [LMSYS 聊天机器人竞技场](https://lmarena.ai/)中那样）[10]。可以说，这种方法在代表真实模型使用情况方面能提供最可靠的结果，但遗憾的是，通过这种方式获取消融实验结果既昂贵又耗时。而且，这通常还要求模型经过指令微调阶段以获得对话能力，因为预训练模型并非直接设计用于遵循指令，因此对提示细节更为敏感[11]。

在这项工作中，我们采用了训练小型模型并在一组“早期信号”基准任务上对其进行评估的方法。我们认为，在牢记上述关于在评估基准上过拟合的注意事项的前提下，这是对用于训练这些模型的数据质量的一个合理代理指标。

### 1.4 消融实验与评估设置

为了比较某一特定处理步骤的影响，我们在数据集的两个版本上训练了两个模型，一个版本经过额外步骤处理（即我们希望评估的步骤），另一个版本则去除了该步骤（进行了删减/移除）。除了数据之外，这两个模型在其他方面完全相同：参数数量相同、架构超参数相同，并且都在每个版本的数据中随机抽取相同数量的 tokens 上进行单轮训练 —— 因此唯一的区别就在于训练数据。然后，我们在相同的任务集上对每个模型进行评估，并比较平均分数。

我们的消融模型是使用 [`nanotron`](https://github.com/huggingface/nanotron) 训练的。我们的“消融模型”有 1.82B 个参数（包括嵌入），采用了 Llama 架构，序列长度为 2048，全局批量大小约为 2M tokens，并使用了 GPT2 分词器。对于大多数消融实验，我们在约 28B tokens 上进行了训练（大致是该模型规模的 Chinchilla[12] 最优训练规模）。为了确认每一步过滤后的相对性能提升，我们按照下文进一步说明，在 350B tokens 上进行了更长时间的训练运行。

（我们将在 Nanotron 上尽快提供用于复现这些消融模型的配置。）

我们使用 [`lighteval`](https://github.com/huggingface/lighteval/) 对模型进行了评估。我们通过挑选那些在较小规模下（仅在“几百亿” tokens 上训练的“小”模型）能提供良好信号的基准测试，精心选定了一组用于消融研究的基准测试集。通常，我们会依据以下标准，在 `lighteval` 提供的所有基准测试中挑选这些基准测试。

- 在对同一数据集的不同采样进行训练时，方差较小：我们希望在对数据子集进行的训练能够代表整个数据集，并且在可能的限度内，得到的分数对确切数据点选择的敏感度要低于对我们过滤器的效果的敏感度。
- 训练过程中性能单调（或接近单调）递增：理想情况下，随着看到的标记数量增加，在高信号基准测试中的性能不应下降（这表明在小规模下结果不可靠）。
- 对于此任务，性能至少比随机基线高出几个标准差：鉴于我们规模较小的消融模型和训练，我们通常不会在任何基准测试中达到极高的分数，但我们希望确保得到的分数高于随机噪声。

经过考虑，我们选择了以下基准测试列表：

- CommonSense QA[13]
- HellaSwag[14]
- OpenBook QA[15]
- PIQA[16]
- SIQA[17]
- WinoGrande[18]
- ARC[19]
- MMLU[20]

为确保我们的检查点评估在有限的时间内完成，我们将较长的基准测试样本数量限制在 1000 个（在单个 8 GPU 节点上的墙钟评估时间少于 5 分钟——与训练并行进行）。

你可以在[这里](https://huggingface.co/datasets/HuggingFaceFW/fineweb/blob/main/lighteval_tasks.py)找到我们所使用的全部任务和提示的列表。

## 二、🍷 FineWeb 配方

在接下来的小节中，我们将解释制作 FineWeb 数据集所采取的每个步骤。

![|600](https://huggingfacefw-blogpost-fineweb-v1.static.hf.space/dist/assets/images/fineweb-recipe.png)

你可以在[此处](https://github.com/huggingface/datatrove/blob/main/examples/fineweb.py)找到一个完全可复现的  `datatrove` 配置。

### 2.1 起点：文本提取

CommonCrawl 数据主要有两种格式：WARC 和 WET。**WARC**（网络档案格式）文件包含抓取的原始数据，包括完整的页面 HTML 和请求元数据。**WET**（WARC 封装文本）文件提供了这些网站的纯文本版本。

大量数据集以 WET 文件作为起点。根据我们的经验，Common Crawl 用于创建这些 WET 文件的默认文本提取方式对于大型语言模型（LLM）预训练的目标而言并非最优，而且有多种开源库能够提供更好的文本提取方法。我们使用 trafilatura 库 [21] 从 WARC 文件中提取文本内容，通过直观检查结果发现，与其他库相比，该库能提供高质量的提取效果。

你可以在[此处](https://github.com/scrapinghub/article-extraction-benchmark/blob/master/README.rst)找到一个比较几个文本提取库的基准测试。

为了验证这一决定，我们直接使用 WET 文件处理了 2019-18 数据转储，并使用 trafilatura 从 WARC文件中提取的文本进行处理。我们对每个数据集应用了相同的处理流程（我们的基础过滤+MinHash，详见下文），并训练了两个模型。虽然 WET 数据的结果数据集大约大 25%（约 254B tokens），但其质量比使用 trafilatura 从 WARC 文件中提取文本的数据集（约 200B tokens）差得多。对一些样本的视觉检查证实，WET 文件上的许多额外标记是不必要的页面样板内容。

然而，需要注意的是，文本提取是我们处理过程中成本最高的步骤之一，因此我们认为，对于预算较低的团队来说，使用现成的WET数据可能是一个合理的权衡。

[交互图]

### Base filtering

Filtering is an important part of the curation process. It consists in removing part of the data (be it words, lines, or even full documents) that lowers the performance of the model and is thus deemed to be “lower quality” in our eval-driven process of dataset crafting.

As a basis for our filtering we used part of the setup from RefinedWeb

[22]

. Namely, we:

- Applied URL filtering using a [blocklist](https://dsi.ut-capitole.fr/blacklists/) to remove adult content

- Applied a [fastText language classifier](https://fasttext.cc/docs/en/language-identification.html)
    
    [23]
    
    [24]
    
     to keep only English text with a score ≥ 0.65

- Applied quality and repetition filters from MassiveText
    
    [25]
    
     (using the default thresholds)

After applying this filtering to each of the text extracted dumps (there are currently 96 dumps) we obtained roughly 36 trillion tokens of data 8 .

### Deduplicating the data

Deduplication is one of the most important steps when creating large web datasets for LLM pretraining. Methods to deduplicate datasets attempt to identify and remove redundant/repeated data from the dataset.

#### Why deduplicate?

The web has many aggregators, mirrors, templated pages or just otherwise repeated content spread over different domains and webpages. Sometimes, these duplicated pages can even be introduced by the crawler itself, when different links point to the same page.

Removing these duplicates (deduplicating) has been correlated with improvements in model performance

[26]

 and a reduction in memorization of pretraining data

[27]

, which might allow for better generalization. Additionally, the performance uplift obtained through deduplication can be equated to increased training efficiency: by removing duplicated content, a model can reach the same performance level with fewer training iterations – or equivalently, for a given number of training tokens, a model will have seen more diverse data.

[28]

[29]

There are different ways to identify and even define duplicated data. Common approaches rely on hashing techniques to speed up the process, or on building efficient data structures to index the data (like suffix arrays). Methods can also be “fuzzy”, by using some similarity metric to mark documents as duplicates, or “exact” by checking for exact matches between two documents (or lines, paragraphs, or whatever other granularity level being used) 9 .

#### Our deduplication parameters

Following RefinedWeb

[22]

, we decided to apply MinHash, a fuzzy hash based deduplication technique that scales efficiently to many CPU-nodes and allows us to tune similarity thresholds (by controlling the number and size of buckets) as well as the length of the subsequences considered (by controlling the n-gram size). We chose to collect each document's 5-grams 10 and compute minhashes using 112 hash functions in total, split into 14 buckets of 8 hashes each — targeting documents that are at least 75% similar. Documents with the same 8 minhashes in any bucket are considered a duplicate of each other.

This would mean that for two documents with a similarity (s) of 0.7, 0.75, 0.8 and 0.85, the probability that they would be identified as duplicates would be 56%, 77%, 92% and 98.8% respectively (1-(1-s^8)^{14}). See the plot below for a match probability comparison between our setup with 112 hashes and the one from RefinedWeb, with 9000 hashes, divided into 450 buckets of 20 hashes (that requires a substantially larger amount of compute resources, as each individual hash must be computed, stored and then compared with hashes from other documents):

00.20.40.60.8100.20.40.60.81

MinHash parametersFineWeb: 1-(1-s^8)^14RefinedWeb: 1-(1-s^20)^450Document similarity (s)Matched as dups probability

[](https://plotly.com/)

While the high number of hash functions in RefinedWeb allows for a steeper, more well defined cut off (documents with real similarity near the threshold are more likely to be correctly identified), we believe the compute and storage savings are a reasonable trade off.

It should also be noted that intra-document deduplication is already handled by our repetition filter, which removes documents with many repeated lines and paragraphs.

#### More deduplication is always better, right?

Initially, we were operating under the assumption that _more deduplication is always better_, so our first approach was to take the entire dataset (all 90+ dumps) and deduplicate them together as one big dataset using MinHash.

We did this in an iterative manner: starting with the most recent dump (which at the time was 2023-50) and proceeding chronologically until we reached the oldest crawl. We deduplicated each dump not only within itself, but removing any document matching any other documents in the previously processed dumps.

For instance, for the second most recent dump (2023-40 at the time), we deduplicated it against the most recent one in addition to within itself. As a result, the older the dumps, the larger the number of dumps it was deduplicated against and the more data we removed from it (indeed, in the oldest dumps, the deduplication step removed more than 90% of the base filtered data).

Deduplicating the dataset in this manner resulted in 4 trillion tokens of data, but, quite surprisingly to us, when training on a randomly sampled 350 billion tokens subset, our ablation models showed next to no improvement over a model trained on the non deduplicated data, scoring far below its predecessor RefinedWeb on our aggregate of tasks (see graph below).

01002003000.380.40.420.440.460.48

Dedup across all dumps does not improve performanceRefinedWebFineWeb filtered onlyFineWeb full MinHashTraining tokens (billions)Aggregate Score

[](https://plotly.com/)

Metric:Aggregate ScoreHellaSwagARCMMLUOpenBook QACommonsense QAPIQASocial IQAWinoGrande

Rolling window:

5

This challenged our assumption that more deduplication would inevitably result in higher benchmark scores, so we decided to take a closer look at one of the oldest dumps, dump 2013-48:

- pre deduplication, this dump had ~490 billion tokens

- after our iterative MinHash, ~31 billion tokens remained (94% of data had been removed)

As an experiment, we tried training two models on 28 billion tokens sampled from the following data from 2013-48:

- the fully deduplicated remaining ~31 billion tokens (_originally kept data_)

- 171 billion tokens obtained by individually deduplicating (without considering the other dumps) the ~460 billion tokens that had been removed from this dump in the iterative dedup process (_originally removed data_) 11

010200.340.360.380.40.42

The originally removed data outperforms the kept dataOriginally removed dataOriginally kept dataTraining tokens (billions)Aggregate Score

[](https://plotly.com/)

Metric:Aggregate ScoreHellaSwagARCMMLUOpenBook QACommonsense QAPIQASocial IQAWinoGrande

Rolling window:

0

These results show that, for this older dump taken in isolation, the data that was kept (10% of the original data) was actually _worse_ than the 90% of data we removed 12 . This is also confirmed by visual inspection: _originally kept data_ contains far more ads, lists of keywords and generally badly formatted text than _originally removed data_.

#### Taking a step back: individual dump dedup

We decided to experiment with an alternative approach: we deduplicated each dump with MinHash individually (independently of the other dumps). This resulted in 20 trillion tokens of data.

When training on a random sample from this dataset we see that it now matches RefinedWeb’s performance (see curves below):

01002003000.380.40.420.440.460.48

Independent dedup outperforms dedup across dumpsFineWeb independent MinHashRefinedWebFineWeb filtered onlyFineWeb full MinHashTraining tokens (billions)Aggregate Score

[](https://plotly.com/)

Metric:Aggregate ScoreHellaSwagARCMMLUOpenBook QACommonsense QAPIQASocial IQAWinoGrande

Rolling window:

5

We hypothesize that the main improvement gained from deduplication is the removal of very large clusters that are present in every single dump (you will find some examples of these clusters in the RefinedWeb paper, each containing _hundreds of thousands_ of documents) and that further deduplication for clusters with a low number of duplicates (less than ~100 i.e. the number of dumps) actually harms performance: data that does not find a duplicate match in any other dump might actually be worse quality/more out of distribution (as evidenced by the results on the 2013-48 data).

While you might see some performance improvement when deduplicating a few dumps together, at the scale of the entire dataset (all the dumps), the effect from this upsampling of lower quality data side effect seems to be more impactful.

One possibility to consider is that as filtering quality improves, this effect may not be as prevalent, since the filtering might be able to remove some of this lower quality data. We also experimented with applying different, and often “lighter”, deduplication approaches on top of the individually deduplicated dumps. You can read about them further below.

#### A note on measuring the effect of deduplication

Given the nature of deduplication, its effect is not always very visible in a smaller slice of the dataset (such as 28B tokens, the size we used for our filtering ablations). Furthermore, one must consider the fact that there are specific effects at play when deduplicating across all CommonCrawl dumps, as some URLs/pages are recrawled from one dump to the next.

To visualize the effect of scaling the number of training tokens on measuring deduplication impact, we considered the following (very extreme and unrealistic regarding the degree of duplication observed) theoretical scenario:

- there are 100 CommonCrawl dumps (roughly accurate)

- each dump has been perfectly individually deduplicated (every single document is unique in this dump)

- each dump is a perfect copy of each other (maximum possible duplication across dumps, effectively the worst case scenario)

- each dump has 200 billion tokens (for a total of 20 trillion, the resulting size of our individual dedup above)

- each dump is made up of documents of 1k tokens (200M documents per dump)

We then simulated uniformly sampling documents from this entire dataset of 20 trillion tokens, to obtain subsets of 1B, 10B, 100B, 350B and 1T tokens. In the image below you can see how often each document would be repeated.

1B10B100B350B1T00.20.40.60.81

Sampling from 1000 identical buckets with 200B tokens each# duplicates16-328-164-8321Sample sizeDataset fraction

[](https://plotly.com/)

For 1B almost all documents would be unique (#duplicates=1), despite the fact that in the entire dataset each document is repeated 100 times (once per dump). We start seeing some changes at the 100B scale (0.5% of the total dataset), with a large number of documents being repeated twice, and a few even 4-8 times. At the larger scale of 1T (5% of the total dataset), the majority of the documents are repeated up to 8 times, with some being repeated up to 16 times.

We ran our performance evaluations for the deduplicated data at the 350B scale, which would, under this theoretical scenario, be made up of a significant portion of documents duplicated up to 8 times. This simulation illustrates the inherent difficulties associated with measuring deduplication impact on the training of LLMs, once the biggest duplicate clusters have been removed.

#### Other (failed) global approaches

To build on top of our newly found method (independently deduplicating each dump). We attempted to improve the performance by further deduplicating the independently minhash deduped 20 trillion tokens of data with alternative global (over all dumps) deduplication methods. We explored the following approaches:

- URL deduplication, where we only kept one document per normalized (lowercased) URL (71.5% of tokens removed, 5.6T left) — _FineWeb URL dedup_

- Line deduplication:
    
    - remove all but 1 (randomly chosen) occurrence of each duplicated line (77.8% of tokens dropped, 4.4T left) — _FineWeb line dedup_
    
    - same as above, but only removing duplicate lines with at least 10 words and dropping documents with fewer than 3 sentences after deduplication (85% of tokens dropped, 2.9T left) — _FineWeb line dedup w/ min words_
    
    - remove all but 1 occurrence of each span of 3 duplicated lines with each number treated as 0 when finding duplicates, (80.9% of tokens removed, 3.7T left) — _FineWeb 3-line dedup_

The performance of the models trained on each of these was consistently worse (even if to different degrees) than that of the original independently deduplicated data:

01002003000.360.380.40.420.440.460.48

Attempting to further globally dedup worsened perfFineWeb independent MinHashRefinedWebFineWeb line dedup w/ min wordsFineWeb URL dedupFineWeb line dedupFineWeb 3-line dedupFineWeb full MinHashFineWeb filtered onlyTraining tokens (billions)Aggregate Score

[](https://plotly.com/)

Metric:Aggregate ScoreHellaSwagARCMMLUOpenBook QACommonsense QAPIQASocial IQAWinoGrande

Rolling window:

5

### Additional quality filtering

By this point we had reached the same performance of the previous work we attempted to reproduce and extend: RefinedWeb, using our base filtering and independent MinHash. Still, on our aggregate of tasks, another heavily filtered dataset, the C4 dataset

[30]

, still showed stronger performances on some benchmarks of our evaluation suite.

We therefore set out to find new filtering steps that would, at first, allow us to match the performance of C4 and, at a second stage, surpass it. A natural starting point was to look into the processing of C4 itself.

#### C4: A dataset that has stood the test of time

The [C4 dataset](https://huggingface.co/datasets/c4) was first released in 2019. It was obtained from the `2019-18` CommonCrawl dump by removing non english data, applying some heuristic filters on both the line and document level, deduplicating on the line level, and removing documents containing words from a word blocklist.

Despite its age and limited size for current standards (around 175B gpt2 tokens), this dataset is, to this day, a common sub-set of typical LLM training, being used in models such as the relatively recent Llama1

[31]

. This success is due to the strong performance that models trained on this dataset exhibit, excelling in particular on the Hellaswag benchmark 

[14]

, one of the benchmarks in our “early signal” group with the highest signal-to-noise ratio. We experimented applying each of the different filters used in C4 to a baseline of the independently deduped FineWeb 2019-18 dump:

051015200.30.350.40.45

C4 filtering effect on HellaSwagAll filtersC4All filters except terminal_punctterminal_punct filterword_lengths filtercurly_bracket filterbaselineTraining tokens (billions)HellaSwag

[](https://plotly.com/)

Metric:Aggregate ScoreHellaSwagARCMMLUOpenBook QACommonsense QAPIQASocial IQAWinoGrande

Rolling window:

3

- applying “All filters” (drop lines not ending on punctuation marks, mentioning javascript and cookie notices + drop documents outside length thresholds, containing “lorem ipsum” or a curly bracket, `{`) allows us to match C4’s HellaSwag performance ("All filters" vs "C4" curves, respectively).

- The curly bracket filter, and the word lengths filter only give a small boost, removing 2.8% and 4.3% of tokens, respectively

- The terminal punctuation filter, by itself, gives the biggest individual boost, but removes _around 30%_ of all tokens (!)

- The lorem_ipsum, javascript and policy rules each remove <0.5% of training tokens, so we did not train on them individually

- "All filters except the (very destructive) terminal_punct" performs better than terminal_punct by itself, while removing less in total (~7%)

We decided to apply all C4 filters mentioned above except the terminal punctuation one. We validated these results with a longer run, which you will find in a plot in the next section.

#### A statistical approach to develop heuristic filters

To develop new heuristic filters and select their thresholds we devised a systematic process:

1. we started by collecting a very large list of high level statistics of our datasets (over **fifty** different metrics) ranging from common document-level metrics (e.g. number of lines, avg. line/word length, etc) to inter-document repetition metrics (inspired by MassiveText), on both a high quality and a lower quality web dataset;
2. we selected the metrics for which the Wasserstein distance between the two distributions (of the metric computed on each dataset) was larger;
3. we inspected the histograms of the two distributions and empirically chose a threshold that would make the lower quality dataset more closely resemble the higher quality one on this metric;
4. we validated the resulting filter (metric-threshold pair) by using it on a reference dataset and running small ablations.

Due to our (new) assumption that global MinHash greatly upsamples lower quality data in the oldest dumps, we computed metrics on both the independently MinHashed and the (worse quality) global MinHashed versions of the 2013-48 and 2015-22 crawls (two older crawls). We then compared the statistics at a macro level, by looking at the distribution of these metrics for each one.

Perhaps not too surprisingly given our findings for deduplication, we found significant disparities in most of the metrics for the two deduplication methods. For instance, the `line-char-duplicates` metric (nb. of characters in duplicated lines / nb. characters), roughly doubled from the independent dedup (0.0053 for 2015-22 and 0.0058 for 2013-48), to the global dedup (0.011 for 2015-22 and 0.01 for 2013-48), indicating that the latter had higher inter-document repetition.

Following the process listed above for these datasets yielded **seventeen** candidate metric-threshold pairs. In the image below, you can see three of these histograms:

00.20.40.60.8100.020.040.060.080.10.120.14

Histograms of selected metricsFull MinHash CC-MAIN-2013-48Independent MinHash CC-MAIN-2013-48Fraction of lines ended with punctuationDocument FrequencyFiltered out

[](https://plotly.com/)

Metric:Lines Ended With PunctuationLines CharsShort Lines

As an example, we inspected the histograms of "fraction of lines ending with punctuation" (see the image above) and observed an increased document density of global MinHash at around 0.12. We then filtered with this threshold and found that the removed data had a higher amount of short lists or consisted of only document layout text ("Home", "Sign up", etc).

We then assessed the effectiveness of these seventeen newly created filters, by conducting several of our _28 billion tokens_ ablation runs on the _2019-18 crawl_. Out of all those runs, we identified **three** filters (the ones based on the histograms above) that demonstrated the most significant improvements on the aggregate score:

- Remove documents where the fraction of lines ending with punctuation ≤ 0.12 (10.14% of tokens removed) — vs the 30% from the original C4 terminal punct filter

- Remove documents where the fraction of characters in duplicated lines ≥ 0.1 (12.47% of tokens removed) — the original MassiveText threshold for this ratio is ≥ 0.2

- Remove documents where the fraction of lines shorter than 30 characters ≥ 0.67 (3.73% of tokens removed)

- When applying the three together, ~22% of tokens were removed.

051015200.360.370.380.390.40.410.420.43

Custom filters PerformanceFilters combinedPunctuation filterLine duplicates filterShort lines filterBaselineTraining tokens (billions)Aggregate Score

[](https://plotly.com/)

Metric:Aggregate ScoreHellaSwagARCMMLUOpenBook QACommonsense QAPIQASocial IQAWinoGrande

Rolling window:

3

These filters allowed us to further improve performance and to, notably, surpass the C4 dataset performance while providing a much larger dataset at the same time.

### The final 🍷 FineWeb dataset

The final [🍷 FineWeb](https://huggingface.co/datasets/HuggingFaceFW/fineweb) dataset comprises 15T tokens and includes the following previously mentioned steps, in order, each providing a performance boost on our group of benchmark tasks:

- base filtering

- independent MinHash deduplication per dump

- a selection of C4 filters

- our custom filters (mentioned in the previous section)

01002003000.380.40.420.440.460.48

The different FineWeb processing stepsFineWeb: id mh + C4 + custom filtersFineWeb: id mh + C4 filtersFineWeb: independent MinHash (id mh)FineWeb: base filtering onlyTraining tokens (billions)Aggregate Score

[](https://plotly.com/)

Metric:Aggregate ScoreHellaSwagARCMMLUOpenBook QACommonsense QAPIQASocial IQAWinoGrande

Rolling window:

5

#### Comparisons with other web-scale datasets

We compared [🍷 FineWeb](https://huggingface.co/datasets/HuggingFaceFW/fineweb) with the following datasets that are usually considered the highest quality openly accessible web-scale datasets (we also indicate for each the approximate number of tokens in the public version of the dataset):

- [RefinedWeb](https://huggingface.co/datasets/tiiuae/falcon-refinedweb) (500B tokens)
    
    [22]
    

- [C4](https://huggingface.co/datasets/allenai/c4) (172B tokens)
    
    [30]
    

- [Dolma v1.6](https://huggingface.co/datasets/allenai/dolma) (3T tokens) (the CommonCrawl part) 
    
    [32]
    
     13

- [The Pile](https://huggingface.co/datasets/EleutherAI/pile) (340B tokens) 
    
    [33]
    

- [SlimPajama](https://huggingface.co/datasets/cerebras/SlimPajama-627B) (627B tokens) 
    
    [34]
    

- [RedPajama2](https://huggingface.co/datasets/togethercomputer/RedPajama-Data-V2) (20T tokens) 
    
    [35]
    
     (deduplicated)

- and our new [🍷 FineWeb](https://huggingface.co/datasets/HuggingFaceFW/fineweb) (15T tokens) (this report)

You will find the 350B-tokens-trained ablation models openly accessible and gathered in [this collection](https://huggingface.co/collections/HuggingFaceFW/ablation-models-662457b0d213e8c14fe47f32). We have uploaded checkpoints at every 1000 training steps. You will also find our full [evaluation results here](https://huggingface.co/datasets/HuggingFaceFW/fineweb/blob/main/eval_results.csv).

01002003000.360.380.40.420.440.460.48

Dataset ablationsFineWeb (ours)RefinedWebC4DolmaSlimPajamaRedPajama2The PileTraining tokens (billions)Aggregate Score

[](https://plotly.com/)

Metric:Aggregate ScoreHellaSwagARCMMLUOpenBook QACommonsense QAPIQASocial IQAWinoGrande

Rolling window:

5

🍷 FineWeb is thus – to the best of our knowledge – the open dataset leading to the current highest model performances while allowing to train on several trillion tokens.

## 📚 FineWeb-Edu

1002003000.380.40.420.440.460.480.5

Dataset ablationsFineWeb-EduFineWebRefinedWebC4DolmaSlimPajamaRedPajama2The PileTraining tokens (billions)Aggregate Score

[](https://plotly.com/)

Metric:Aggregate ScoreHellaSwagARCMMLUOpenBook QACommonsense QAPIQASocial IQAWinoGrande

Rolling window:

5

📚 FineWeb-Edu outperforms 🍷 FineWeb and all other open web datasets on our group of evaluation tasks.

[📚 FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) is an additional development of FineWeb that we are excited to introduce in this tech report and openly release. 📚 FineWeb-Edu is based on a new approach that has recently emerged for filtering LLM training datasets: using synthetic data to develop classifiers for identifying educational content. This technique was notably used in the trainings of Llama 3

[1]

 and Phi3

[36]

, but its large-scale impact on web data filtering has, in our opinion, thur far not been publicly explored to its full potential.

The popular Phi3 models were trained on 3.3 and 4.8 trillion tokens, with the paper

[36]

 stating:

> Our training data consists of heavily filtered publicly available web data (according to the 'educational level') from various open internet sources, as well as synthetic LLM-generated data.

Similarly, Llama 3 blog post

[37]

 notes:

> We found that previous generations of Llama are good at identifying high-quality data, so we used Llama 2 to help build the text-quality classifiers that are powering Llama 3.

However, these classifiers and filtered datasets are not publicly available. To further enhance 🍷 FineWeb's quality, we developed an educational quality classifier using annotations generated by [Llama-3-70B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-70B-Instruct) to create [**📚 FineWeb-Edu**](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu).

### Annotating for educational quality at scale

We used [Llama-3-70B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-70B-Instruct) to annotate 500k samples from 🍷 FineWeb, scoring each for their educational quality on a scale from 0 to 5.

We explored various prompt formats to automatically extract an educational score using an LLM and found that the additive scale by Yuan et al.

[38]

 worked best. This scale allows the LLM to reason about each additional point awarded, unlike the single-rating Likert scale which fits samples into predefined boxes. Then, to avoid the LLM favoring highly technical pages like arXiv abstracts and submissions, we focused on grade-school and middle-school level knowledge. By setting a threshold of 3 (on a scale of 0 to 5) during the filtering process, we were able to also retain some high-level educational pages.

![Prompt for LLM annotation](https://cdn-uploads.huggingface.co/production/uploads/61c141342aac764ce1654e43/fjZQ4izIj1rx1xQnBTKKr.png)

Prompt used for Llama3 annotations of the educational score, also available [here](https://huggingface.co/HuggingFaceFW/fineweb-edu-classifier/blob/main/utils/prompt.txt).

In terms of open-weight models to use for annotating the data, we experimented with several models including [Mixtral-8x7B-Instruct](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1) and [Mixtral-8x22B-Instruct](https://huggingface.co/mistralai/Mixtral-8x22B-Instruct-v0.1), [Llama-3-70B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-70B-Instruct) as well as a jury gathering the scores from these three models

[39]

. In our experiments we found that using Llama3 alone gave the most reliable results.

### Training a classifier

To scale our annotations to the trillions of tokens in FineWeb, we used the Llama3-70B annotations to train a small classifier. The model we used was a [Snowflake-arctic-embed](https://huggingface.co/Snowflake/snowflake-arctic-embed-m) embedding model with a classification head with a single regression output on top of it. We trained this model on the 450,000 Llama 3 annotations for 20 epochs with a learning rate of 3e-4, freezing the embedding and encoder layers. We saved the checkpoint with the highest F1 score on our held-out validation set of 45k samples, treating Llama 3 annotations as ground-truth. After training, we rounded the scores to integers from `0` to `5`.

We then converted the problem to a binary classification task by using a fixed threshold to determine if a file is educational. With a threshold of `3`, the model achieved an F1 score of 82% on the validation set, indicating strong performance in distinguishing high-quality educational content.

The classifier is available at: [HuggingFaceFW/fineweb-edu-classifier](https://huggingface.co/HuggingFaceFW/fineweb-edu-classifier). The training and inference code is available on [GitHub](https://github.com/huggingface/cosmopedia/tree/main/classification).

### Filtering and results

We applied the classifier to the 15T tokens of 🍷 FineWeb, a process that required 6,000 H100 GPU hours. We investigated the impact of using different thresholds for the filtering and found that using a threshold of `3` gave the best overall results. Although using a threshold higher than `3` improves performance on knowledge and reasoning intensive benchmarks, it significantly degrades performance on HellaSwag and PIQA. The plot below shows the performance of each threshold compared to FineWeb on six different benchmarks; it uses a 1.82B model trained on 8B tokens.

FW-Edu-threshold=4FW-Edu-threshold=3FW-Edu-threshold=2FineWeb (FW)0.240.260.280.30.32

FineWeb-Edu thresholdingDatasetMMLU

[](https://plotly.com/)

Metric:HellaSwagARCMMLUOpenBook QAPIQASocial IQAWinoGrande

**Note:** this ablation was conducted on 8B tokens from the 2024-10 dump for both the FineWeb and FineWeb-Edu subsets, which might not be representative of the entire dataset. The next ablation shows that the findings for threshold 3 hold on a longer run of 350B tokens from all FineWeb dumps, except for HellaSwag, where we noticed a slight performance degradation.

We built 📚 FineWeb-Edu by filtering out samples with scores lower than 3. This removed 92% of the dataset, leaving us with 1.3 trillion educational tokens. To evaluate the effectiveness of this filtering at a larger scale, we conducted an ablation using a 1.82B model trained on 350 billion tokens, similar to the FineWeb filtering ablation mentioned above:

C4DolmaFineWebRedPajama2RefinedWebSlimPajamaThe PileFineWeb-Edu0.250.30.350.4

Evaluation results at 350B tokensDatasetMMLU

[](https://plotly.com/)

Metric:Aggregate ScoreHellaSwagARCMMLUOpenBook QAPIQASocial IQAWinoGrande

Here are the key highlights of the ablation results above:

- 📚 FineWeb-Edu **surpasses 🍷 FineWeb and all other open web datasets, with remarkable improvements on educational benchmarks** such as MMLU, ARC, and OpenBookQA.
- It achieves the same performance with significantly less data, requiring 10x fewer tokens compared to C4 and Dolma to match MMLU results.
- This demonstrates the effectiveness of using classifiers trained on LLM annotations for large-scale data filtering.

Given that a threshold of 2 also demonstrated strong performance while retaining more data, we are releasing an additional dataset filtered with this threshold, containing 5.4 trillion tokens under [HuggingFaceFW/fineweb-edu-score-2](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu-score-2).

You can find the two datasets along with the classifier used for the filtering in this [collection](https://huggingface.co/collections/HuggingFaceFW/fineweb-edu-6659c3f3d399d0e1d648adfd).

## Bonus: CommonCrawl over time

> Just like fine wine, not all crawls are created equal.

While ablating filtering steps, we noticed that certain crawls outperformed others by a significant margin. We decided to investigate this phenomenon.

### Benchmark performance by crawl

For each crawl, we trained two 1.8B models on 27 billion tokens randomly sampled from that crawl's data (after the base filtering and MinHash deduplication steps), where each run had a different random 27BT sampling of this data. We trained 192 such models, totaling over 60 thousand H100 GPU-hours. We subsequently took the last 3 checkpoints for both runs and plotted the average of these 6 data points per crawl.

The plot below clearly shows that some dumps perform far worse than others. Each year has a different color, and the number of crawls per year also varies.

2013201420152016201720182019202020212022202320240.420.4250.430.435

Score by dumpYearAggregate Score

[](https://plotly.com/)

Metric:Aggregate ScoreHellaSwagARCMMLUOpenBook QACommonsense QAPIQASocial IQAWinoGrande

We investigated possible causes for this behaviour such as changes in the most common URLs of each dump, as well as potential benchmark contamination, but could not find any conclusive explanation. We leave further investigation for future work.

### Synthetic data

We wondered if the strong performance of the last few crawls could be, in part, attributed to the presence of a larger quantity of synthetic data (data generated by LLMs). Such a change would not be surprising due to the recent increase in popularity of LLMs, notably of ChatGPT.

Since, to the best of our knowledge, there is no foolproof method to detect synthetic data, we opted to use a proxy metric: we measured the frequency of the following words in each crawl: `"delve", "as a large language model", "it's important to note", "rich tapestry", "intertwined", "certainly!", "dive into"`, all of which are commonly used by ChatGPT.

It is important to note that not all samples containing one of these phrases were necessarily generated by ChatGPT (and also that many ChatGPT generated samples do not contain any of these phrases), but assuming that the amount of synthetic data were to not change across crawls, one would expect these frequencies to remain approximately constant over time.

The results are shown in the following plot:

2021-042021-102021-172021-212021-252021-312021-392021-432021-492022-052022-212022-272022-332022-402022-492023-062023-142023-232023-402023-502024-102024-1805μ10μ15μ20μ0.4240.4260.4280.430.4320.4340.4360.438

Synthetic Data ContaminationYearSynthetic proxy Words RatioAggregate ScoreChat-GPT Release

[](https://plotly.com/)

While the frequency remained approximately constant until 2023-14 (ChatGPT was released at the end of 2022), we find a steep increase of our proxy metric in recent crawls. While this simple test is not enough to conclude that ChatGPT completions and other synthetic data is improving the quality of the most recent crawl, it at the very least does not seem to drastically harm it.

We expect to continue seeing increasing quantities of synthetic data on new CC crawls. However, while for relatively small trainings this data does not seem to harm performance (and might actually improve it), it is not clear that this holds for much larger trainings.

## Conclusion and looking forward

Through our open science efforts we hope to keep shining a light on the black box that is the training of high performance large language models as well as to give every model trainer the ability to create state-of-the-art LLMs. We are excited to continue iterating on FineWeb and to release increasingly better filtered subsets of web data, in a fully open and reproducible manner.

In the short term, we are looking forward to applying the learnings from (English) FineWeb to other languages. While English currently dominates the LLM landscape, we believe that making high quality web data in other languages as accessible as possible would be incredibly impactful.

In a nutshell: the future is bright and exciting for studying the science of creating datasets at scale and in the open 🤗.

### Citation

For attribution in academic contexts, please cite this work as

Penedo, et al., "The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale", 2024.

BibTeX citation

@inproceedings{
penedo2024the,
title={The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale},
author={Guilherme Penedo and Hynek Kydl{\'\i}{\v{c}}ek and Loubna Ben allal and Anton Lozhkov and Margaret Mitchell and Colin Raffel and Leandro Von Werra and Thomas Wolf},
booktitle={The Thirty-eight Conference on Neural Information Processing Systems Datasets and Benchmarks Track},
year={2024},
url={https://openreview.net/forum?id=n6SCkn2QaG}
}

### Footnotes

1. Note that the size changes from crawl to crawl. Note also that we use "dump" or "crawl" interchangeability in this report.[[↩]](https://huggingfacefw-blogpost-fineweb-v1.static.hf.space/dist/index.html#d-footnote-1)
2. We have not processed these 3 older crawls.[[↩]](https://huggingfacefw-blogpost-fineweb-v1.static.hf.space/dist/index.html#d-footnote-2)
3. Note that this report is focused on the special field of web-scale datasets ("web-scale" typically meaning >100 billion tokens obtained from the web) used to pretrain a Large Language Model (by pretraining we mean the very first step in the training of a model, starting from random weights). We don't pretend to cover any other field of dataset creation nor that the lessons or hypothesis we develop in this document can extend to any field besides this specific field.[[↩]](https://huggingfacefw-blogpost-fineweb-v1.static.hf.space/dist/index.html#d-footnote-3)
4. Even though as we mentioned above the notion of "clean" is so ill-defined that it should probably not been seen as equivalent to wikipedia-type of text[[↩]](https://huggingfacefw-blogpost-fineweb-v1.static.hf.space/dist/index.html#d-footnote-4)
5. "Small" in comparison to standard sizes of today's LLMs, i.e. small in comparison to 7-70 billion parameters. In this work "small" means about 1-2 billion parameters[[↩]](https://huggingfacefw-blogpost-fineweb-v1.static.hf.space/dist/index.html#d-footnote-5)
6. In particular we suspect that it keeps too much boilerplate content and navigation menus.[[↩]](https://huggingfacefw-blogpost-fineweb-v1.static.hf.space/dist/index.html#d-footnote-6)
7. We used trafilatura default options with `favour_precision=True`.[[↩]](https://huggingfacefw-blogpost-fineweb-v1.static.hf.space/dist/index.html#d-footnote-7)
8. As everywhere in this report: this is the number of tokens when tokenized with the `gpt2` tokenizer[[↩]](https://huggingfacefw-blogpost-fineweb-v1.static.hf.space/dist/index.html#d-footnote-8)
9. Note that here, even when we discuss "fuzzy" deduplication, we are only employing methods that operate on character/word matches, aka surface-level text. A more complex concept of deduplication is concerned with "semantic" deduplication: comparing/removing texts which are relative to the same concepts and use for instance synonyms or paraphrasing. We don't discuss these topics here but note that they can be important in the field of large-scale synthetic data generation for instance (see our [Cosmopedia release](https://huggingface.co/blog/cosmopedia) on this topic)[[↩]](https://huggingfacefw-blogpost-fineweb-v1.static.hf.space/dist/index.html#d-footnote-9)
10. Our units are "words", computed in the [MinHash processing function](https://github.com/huggingface/datatrove/blob/e9963f69f1fbab1a61339bd1b497f6e138b9f47f/src/datatrove/pipeline/dedup/minhash.py#L196) with a [language-specific word tokenizer](https://github.com/huggingface/datatrove/blob/e9963f69f1fbab1a61339bd1b497f6e138b9f47f/src/datatrove/utils/word_tokenizers.py#L323).[[↩]](https://huggingfacefw-blogpost-fineweb-v1.static.hf.space/dist/index.html#d-footnote-10)
11. While there may be documents in _originally kept data_ similar to documents in _originally removed data_, we estimate the overlap to be small (around 4 billion tokens)[[↩]](https://huggingfacefw-blogpost-fineweb-v1.static.hf.space/dist/index.html#d-footnote-11)
12. Note that these ablation models are trained only on data from this dump so it's considered independently of all the other dumps.[[↩]](https://huggingfacefw-blogpost-fineweb-v1.static.hf.space/dist/index.html#d-footnote-12)
13. There is a newer version of Dolma, v1.7, which is smaller[[↩]](https://huggingfacefw-blogpost-fineweb-v1.static.hf.space/dist/index.html#d-footnote-13)

### References

1. Language Models are Unsupervised Multitask Learners  
    Radford, A., Wu, J., Child, R., Luan, D., Amodei, D. and Sutskever, I., 2019.
2. DataTrove: large scale data processing  [[link]](https://github.com/huggingface/datatrove)  
    Penedo, G., Kydlíček, H., Cappelli, A., Sasko, M. and Wolf, T., 2024. GitHub repository. GitHub.
3. Measuring Data  
    Mitchell, M., Luccioni, A.S., Lambert, N., Gerchick, M., McMillan-Major, A., Ozoani, E., Rajani, N., Thrush, T., Jernite, Y. and Kiela, D., 2023.
4. A Pretrainer's Guide to Training Data: Measuring the Effects of Data Age, Domain Coverage, Quality, & Toxicity  
    Longpre, S., Yauney, G., Reif, E., Lee, K., Roberts, A., Zoph, B., Zhou, D., Wei, J., Robinson, K., Mimno, D. and Ippolito, D., 2023.
5. CCNet: Extracting High Quality Monolingual Datasets from Web Crawl Data  
    Wenzek, G., Lachaux, M., Conneau, A., Chaudhary, V., Guzmán, F., Joulin, A. and Grave, E., 2019.
6. Dolma: an Open Corpus of Three Trillion Tokens for Language Model Pretraining Research  
    Soldaini, L., Kinney, R., Bhagia, A., Schwenk, D., Atkinson, D., Authur, R., Bogin, B., Chandu, K., Dumas, J., Elazar, Y., Hofmann, V., Jha, A.H., Kumar, S., Lucy, L., Lyu, X., Lambert, N., Magnusson, I., Morrison, J., Muennighoff, N., Naik, A., Nam, C., Peters, M.E., Ravichander, A., Richardson, K., Shen, Z., Strubell, E., Subramani, N., Tafjord, O., Walsh, P., Zettlemoyer, L., Smith, N.A., Hajishirzi, H., Beltagy, I., Groeneveld, D., Dodge, J. and Lo, K., 2024.
7. Chatbot Arena: An Open Platform for Evaluating LLMs by Human Preference  
    Chiang, W., Zheng, L., Sheng, Y., Angelopoulos, A.N., Li, T., Li, D., Zhang, H., Zhu, B., Jordan, M., Gonzalez, J.E. and Stoica, I., 2024.
8. Training language models to follow instructions with human feedback  
    Ouyang, L., Wu, J., Jiang, X., Almeida, D., Wainwright, C.L., Mishkin, P., Zhang, C., Agarwal, S., Slama, K., Ray, A., Schulman, J., Hilton, J., Kelton, F., Miller, L., Simens, M., Askell, A., Welinder, P., Christiano, P., Leike, J. and Lowe, R., 2022.
9. Training Compute-Optimal Large Language Models  
    Hoffmann, J., Borgeaud, S., Mensch, A., Buchatskaya, E., Cai, T., Rutherford, E., Casas, D.d.L., Hendricks, L.A., Welbl, J., Clark, A., Hennigan, T., Noland, E., Millican, K., Driessche, G.v.d., Damoc, B., Guy, A., Osindero, S., Simonyan, K., Elsen, E., Rae, J.W., Vinyals, O. and Sifre, L., 2022.
10. CommonsenseQA: A Question Answering Challenge Targeting Commonsense Knowledge  [[link]](https://aclanthology.org/N19-1421)  
    Talmor, A., Herzig, J., Lourie, N. and Berant, J., 2019. Proceedings of the 2019 Conference of the North {A}merican Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pp. 4149--4158. Association for Computational Linguistics. [DOI: 10.18653/v1/N19-1421](https://doi.org/10.18653/v1/N19-1421)
11. HellaSwag: Can a Machine Really Finish Your Sentence?  [[link]](https://aclanthology.org/P19-1472)  
    Zellers, R., Holtzman, A., Bisk, Y., Farhadi, A. and Choi, Y., 2019. Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, pp. 4791--4800. Association for Computational Linguistics. [DOI: 10.18653/v1/P19-1472](https://doi.org/10.18653/v1/P19-1472)
12. Can a Suit of Armor Conduct Electricity? A New Dataset for Open Book Question Answering  
    Mihaylov, T., Clark, P., Khot, T. and Sabharwal, A., 2018. EMNLP.
13. PIQA: Reasoning about Physical Commonsense in Natural Language  
    Bisk, Y., Zellers, R., Bras, R.L., Gao, J. and Choi, Y., 2019.
14. SocialIQA: Commonsense Reasoning about Social Interactions  
    Sap, M., Rashkin, H., Chen, D., LeBras, R. and Choi, Y., 2019.
15. WinoGrande: An Adversarial Winograd Schema Challenge at Scale  
    Sakaguchi, K., Bras, R.L., Bhagavatula, C. and Choi, Y., 2019.
16. Think you have Solved Question Answering? Try ARC, the AI2 Reasoning Challenge  
    Clark, P., Cowhey, I., Etzioni, O., Khot, T., Sabharwal, A., Schoenick, C. and Tafjord, O., 2018.
17. Measuring Massive Multitask Language Understanding  
    Hendrycks, D., Burns, C., Basart, S., Zou, A., Mazeika, M., Song, D. and Steinhardt, J., 2021.
18. Trafilatura: A Web Scraping Library and Command-Line Tool for Text Discovery and Extraction  [[link]](https://aclanthology.org/2021.acl-demo.15)  
    Barbaresi, A., 2021. Proceedings of the Joint Conference of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing: System Demonstrations, pp. 122--131. Association for Computational Linguistics.
19. The RefinedWeb Dataset for Falcon LLM: Outperforming Curated Corpora with Web Data, and Web Data Only  
    Penedo, G., Malartic, Q., Hesslow, D., Cojocaru, R., Cappelli, A., Alobeidli, H., Pannier, B., Almazrouei, E. and Launay, J., 2023.
20. Bag of Tricks for Efficient Text Classification  
    Joulin, A., Grave, E., Bojanowski, P. and Mikolov, T., 2016. arXiv preprint arXiv:1607.01759.
21. FastText.zip: Compressing text classification models  
    Joulin, A., Grave, E., Bojanowski, P., Douze, M., Jegou, H. and Mikolov, T., 2016. arXiv preprint arXiv:1612.03651.
22. Scaling Language Models: Methods, Analysis & Insights from Training Gopher  
    Rae, J.W., Borgeaud, S., Cai, T., Millican, K., Hoffmann, J., Song, F., Aslanides, J., Henderson, S., Ring, R., Young, S., Rutherford, E., Hennigan, T., Menick, J., Cassirer, A., Powell, R., Driessche, G.v.d., Hendricks, L.A., Rauh, M., Huang, P., Glaese, A., Welbl, J., Dathathri, S., Huang, S., Uesato, J., Mellor, J., Higgins, I., Creswell, A., McAleese, N., Wu, A., Elsen, E., Jayakumar, S., Buchatskaya, E., Budden, D., Sutherland, E., Simonyan, K., Paganini, M., Sifre, L., Martens, L., Li, X.L., Kuncoro, A., Nematzadeh, A., Gribovskaya, E., Donato, D., Lazaridou, A., Mensch, A., Lespiau, J., Tsimpoukelli, M., Grigorev, N., Fritz, D., Sottiaux, T., Pajarskas, M., Pohlen, T., Gong, Z., Toyama, D., d'Autume, C.d.M., Li, Y., Terzi, T., Mikulik, V., Babuschkin, I., Clark, A., Casas, D.d.L., Guy, A., Jones, C., Bradbury, J., Johnson, M., Hechtman, B., Weidinger, L., Gabriel, I., Isaac, W., Lockhart, E., Osindero, S., Rimell, L., Dyer, C., Vinyals, O., Ayoub, K., Stanway, J., Bennett, L., Hassabis, D., Kavukcuoglu, K. and Irving, G., 2022.
23. Deduplicating Training Data Makes Language Models Better  
    Lee, K., Ippolito, D., Nystrom, A., Zhang, C., Eck, D., Callison-Burch, C. and Carlini, N., 2022.
24. Quantifying Memorization Across Neural Language Models  
    Carlini, N., Ippolito, D., Jagielski, M., Lee, K., Tramer, F. and Zhang, C., 2023.
25. Scaling Data-Constrained Language Models  
    Muennighoff, N., Rush, A.M., Barak, B., Scao, T.L., Piktus, A., Tazi, N., Pyysalo, S., Wolf, T. and Raffel, C., 2023.
26. Scaling Laws and Interpretability of Learning from Repeated Data  
    Hernandez, D., Brown, T., Conerly, T., DasSarma, N., Drain, D., El-Showk, S., Elhage, N., Hatfield-Dodds, Z., Henighan, T., Hume, T., Johnston, S., Mann, B., Olah, C., Olsson, C., Amodei, D., Joseph, N., Kaplan, J. and McCandlish, S., 2022.
27. Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer  
    Raffel, C., Shazeer, N., Roberts, A., Lee, K., Narang, S., Matena, M., Zhou, Y., Li, W. and Liu, P.J., 2023.
28. LLaMA: Open and Efficient Foundation Language Models  
    Touvron, H., Lavril, T., Izacard, G., Martinet, X., Lachaux, M., Lacroix, T., Rozière, B., Goyal, N., Hambro, E., Azhar, F., Rodriguez, A., Joulin, A., Grave, E. and Lample, G., 2023.
29. Dolma: an Open Corpus of Three Trillion Tokens for Language Model Pretraining Research  
    Soldaini, L., Kinney, R., Bhagia, A., Schwenk, D., Atkinson, D., Authur, R., Bogin, B., Chandu, K., Dumas, J., Elazar, Y., Hofmann, V., Jha, A.H., Kumar, S., Lucy, L., Lyu, X., Lambert, N., Magnusson, I., Morrison, J., Muennighoff, N., Naik, A., Nam, C., Peters, M.E., Ravichander, A., Richardson, K., Shen, Z., Strubell, E., Subramani, N., Tafjord, O., Walsh, P., Zettlemoyer, L., Smith, N.A., Hajishirzi, H., Beltagy, I., Groeneveld, D., Dodge, J. and Lo, K., 2024. arXiv preprint.
30. The {P}ile: An 800{GB} dataset of diverse text for language modeling  
    Gao, L., Biderman, S., Black, S., Golding, L., Hoppe, T., Foster, C., Phang, J., He, H., Thite, A., Nabeshima, N. and others,, 2020. arXiv preprint arXiv:2101.00027.
31. SlimPajama: A 627B token cleaned and deduplicated version of RedPajama  [[link]](https://huggingface.co/datasets/cerebras/SlimPajama-627B)  
    Soboleva, D., Al-Khateeb, F., Myers, R., Steeves, J.R., Hestness, J. and Dey, N., 2023.
32. RedPajama: an Open Dataset for Training Large Language Models  [[link]](https://github.com/togethercomputer/RedPajama-Data)  
    Computer, T., 2023.
33. Phi-3 technical report: A highly capable language model locally on your phone  
    Abdin, M., Jacobs, S.A., Awan, A.A., Aneja, J., Awadallah, A., Awadalla, H., Bach, N., Bahree, A., Bakhtiari, A., Behl, H. and others,, 2024. arXiv preprint arXiv:2404.14219.
34. Our responsible approach to Meta AI and Meta Llama 3  [[link]](https://ai.meta.com/blog/meta-llama-3-meta-ai-responsibility/)  
    Meta,, 2024.
35. Self-rewarding language models  
    Yuan, W., Pang, R.Y., Cho, K., Sukhbaatar, S., Xu, J. and Weston, J., 2024. arXiv preprint arXiv:2401.10020.
36. Replacing Judges with Juries: Evaluating LLM Generations with a Panel of Diverse Models  
    Verga, P., Hofstatter, S., Althammer, S., Su, Y., Piktus, A., Arkhangorodsky, A., Xu, M., White, N. and Lewis, P., 2024. arXiv preprint arXiv:2404.18796.