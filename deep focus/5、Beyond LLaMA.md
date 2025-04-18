

通过将其输出作为训练目标，我们可以将一些来自更大网络的信息提炼到正在训练的较小的“学生”网络中。有关知识蒸馏及其众多变体的更多信息，请查看下面的链接。

[Knowledge Distillation Survey](https://arxiv.org/abs/2006.05525)

### 其他东西……

在整个概述中，我们还将提及 OpenAI 目录中的一些特定模型的名称（例如 text-davinci-003）。有关 OpenAI API 中提供的模型（及相关描述）列表，请参阅此处。

### [Alpaca: An Instruction-following LLaMA model](https://crfm.stanford.edu/2023/03/13/alpaca.html) 

> _“在学术界对指令遵循模型进行研究一直很困难，因为没有容易获取的在能力上接近OpenAI的text-davinci-003等闭源模型的模型。” 

阿尔帕卡（Alpaca）[3] 是LLaMA - 7B [1] 大型语言模型（LLM）的微调版本，其性能与OpenAI的text - davinci - 003（即GPT - 3.5）相似。阿尔帕卡的微调过程基于Self - Instruct [2]，在此过程中，从性能更高的LLM（即text - davinci - 003）收集遵循指令的数据，并用于监督微调（SFT）。简单来说，阿尔帕卡表明，在遵循指令的情境下，小型开源LLM的质量可以通过在高质量数据上进行微调得到极大提升。此外，阿尔帕卡的整个微调过程仅花费600美元（包括数据收集和微调），这使得此类遵循指令的LLM在研究目的下易于且低成本复制。

![|600](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fd507e047-bfb6-4d05-a575-07d4ffb591f1_1434x618.png)

**方法**   要通过监督微调（SFT）创建一个遵循指令的大型语言模型（LLM），我们需要 i) 一个高质量的预训练语言模型以及 ii) 用于监督微调的遵循指令的数据。幸运的是，LLaMA 的近期发布提供了易于获取的预训练语言模型。获取遵循指令的数据则稍微复杂一些，但有一种方法是自我指令[2]。从高层次上讲，自我指令通过引导大型语言模型生成的输出来进行进一步训练。在 Alpaca 的案例中，我们使用 text-davinci-003 通过以下方式生成遵循指令的数据：

1. 从self-instruct的种子集中选取 175 个指令和输出对作为起始。
2. 通过向大型语言模型（LLM）提供种子集作为少样本学习的上下文示例，提示其生成更多指令。

[3]的作者还采用了一些技巧（例如，修改提示以及更高效的解码/生成过程），与原始的self-instruct [2]相比，使数据生成过程成本更低且更高效。总体而言，通过OpenAI API生成指令遵循数据的成本为52K条指令遵循示例不到500美元。

然后，使用基于 HuggingFace 的训练框架，在这些数据上对 LLaMA-7B 模型进行微调。通过使用全分片数据并行（FSDP）和混合精度训练技术，微调过程在 8 个 A100 GPU 上缩短至3小时，成本不到100美元。用于创建Alpaca的代码/数据可在线获取。然而，禁止商业使用Alpaca，原因如下：i) Alpaca所基于的LLaMA具有非商业许可证；ii) OpenAI禁止使用其模型来训练竞争性LLM。

[Alpaca Code](https://github.com/tatsu-lab/stanford_alpaca)

**结果**   阿尔帕卡（Alpaca）在用于自我指令（self-instruct）的评估集指令（即主要与电子邮件、社交媒体和生产力相关的任务）以及作者手写的开放域指令上进行了评估。在这些任务中，发现阿尔帕卡的表现与text-davinci-003相似（即在测试的大约180个案例中，有50%的情况下表现最佳）。尽管这种评估的范围显然有限，但考虑到阿尔帕卡比GPT - 3.5小得多且相对容易复制，其表现仍然相当令人印象深刻。

![|600](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fb76be6ec-f52f-4687-9b88-5db3455053c6_1420x740.png)

与text-davinci-003类似，Alpaca的输出通常比ChatGPT的输出短。换句话说，该模型的风格反映了用于生成微调所用的指令遵循数据的LLM的风格。

### [Vicuna: An Open-Source Chatbot with 90% ChatGPT Quality](https://vicuna.lmsys.org/) [4]

![|500](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F239ec346-e0c2-45c0-bc3c-c4e571a35348_1532x1386.png)

像ChatGPT这样的信息检索对话代理（或聊天机器人）非常出色，但这类模型的训练框架和架构却不为人知，这阻碍了开源研究的发展。为了解决这个问题，[4]的作者提出了Vicuna，这是一个通过微调LLaMA-13B[1]（即一个与GPT-3性能相当的小型语言模型）创建的开源聊天机器人。Vicuna的微调数据是与ChatGPT的用户对话示例，整个微调过程可以以不到300美元的成本复制，从而使聊天机器人更易于用于研究目的。与Alpaca相比，Vicuna与ChatGPT更为相似，并能生成更详细、更有结构的答案。

![|500](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F3478758a-f597-4bee-a07c-0dd4e3cda9b2_1024x458.png)

**方法**   用于与Vicuna进行监督微调（SFT）的数据是通过公共应用程序编程接口（API）从ShareGPT下载的，ShareGPT是一个允许用户分享其与ChatGPT对话的平台。在微调之前，作者会过滤掉不适当和低质量的数据，并将较长的对话分割成适合LLaMA - 13B最大上下文长度的较短片段。总共收集了7万个对话。与Alpaca类似，该模型在8个A100 GPU上使用全量梯度分散并行（FSDP）进行训练（进行了一些修改以降低成本并处理长序列），这大约需要一天时间；详见上文。作者公开提供了用于训练和托管Vicuna的代码。

[Vicuna Code](https://github.com/lm-sys/FastChat)

下表提供了Vicuna与开源大型语言模型LLaMA和Alpaca更全面的比较。接下来我们将讨论如何对Vicuna进行评估。

![|500](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F0742a316-aae9-4929-921c-0cff93187e7d_1524x1248.png)

**结果**   准确评估聊天机器人相当困难，而且随着聊天机器人质量的提高，评估难度会更大。例如，[4]中的作者声称，用于评估Alpaca的自指令评估集已被最近的聊天机器人有效解决，这使得模型之间的差异难以辨别。鉴于现有基准测试的局限性以及创建新的全面评估集的难度，[4]中的作者选择了不同的策略：使用大型语言模型进行评估。

> _“随着GPT - 4的最新进展，我们好奇其能力是否已达到类人水平，从而能够实现一个用于基准测试生成和性能评估的自动化评估框架。”_ 

在这一点上，我们可能会认为这实际上是不可能奏效的。聊天嵌套？然而，令人惊讶的是，基于最近提出的GPT - 4模型[6]构建一个评估框架效果很好。首先，[4]的作者设计了八类问题（例如，角色扮演场景和数学任务）。然后，提示GPT - 4在每个类别中生成一系列多样化的问题。有趣的是，发现GPT - 4能够生成让最近的聊天机器人难以回答的难题。

特别是，GPT - 4被用于在每个类别中生成十个问题，并对五个不同聊天机器人（即LLaMA - 13B、Alpaca - 13B、Vicuna - 13B、Bard和ChatGPT）的输出进行评估。更进一步说，通过让GPT - 4根据详细程度、有用性、相关性和准确性对答案的质量进行评级来判断每个模型输出的质量。尽管以这种方式进行评估似乎有些牵强，但GPT - 4相当一致地对模型进行了排名，甚至还解释了其推理过程。

![|500](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F61e91dff-1f9c-4eb0-9b0f-c02e08fa50ae_900x416.png)

根据GPT - 4的判断，Vicuna的输出质量相对于ChatGPT达到了92%；详见上文。这个比例是通过让GPT - 4为每个模型的输出打分得到的。然后，通过计算所有问题的总质量得分，可以评估模型之间的相对性能。尽管这种评估方法并不严谨，但它相当有趣，相对一致，并且促使我们思考大型语言模型（LLM）领域未来发展的有趣方式。

![|500](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fc78507d2-9ac5-4419-b566-de6f73bbe101_840x480.png)

与其他开源模型相比，我们发现GPT - 4倾向于选择Vicuna的输出。此外，在45%的问题上，Vicuna生成的结果质量超过或与ChatGPT相当。对于一个仅需300美元就能进行微调的模型来说，这种质量水平相当令人印象深刻！

### [Koala: A Dialogue Model for Academic Research](https://bair.berkeley.edu/blog/2023/04/03/koala/) [5]

> _“如果使用精心收集的数据进行训练，体积小到可以在本地运行的模型也能在很大程度上展现其更大规模同类模型的性能。”_ 

在这一点上，我们可能会开始怀疑我们是否会有用尽可供以命名大型语言模型（LLMs）的动物名称的一天。不过，考拉（Koala）与小羊驼（Vicuna）和阿尔帕卡（Alpaca）类似，因为它依然致力于缩小专有大型语言模型和开源大型语言模型之间的质量差距。更具体地说，考拉（Koala）是LLaMA-13B的一个版本，它已经针对来自多种来源的对话数据进行了微调，这些来源包括公共数据集以及互联网上可获取的与其他高质量大型语言模型的对话。

![|500](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F62a48c6e-7b89-43d1-8e09-145f47b3632a_1512x296.png)

当在实际提示上进行评估时，发现Koala-13B与ChatGPT相比表现具有竞争力，甚至超过了相关的Alpaca模型。因此，Koala的研究结果继续支持我们在所有基于LLaMA的工作中看到的趋势。即我们看到，给定用于微调的正确数据，较小的模型也能达到令人印象深刻的质量。这样的发现可能会让我们思考：我们是否过于关注模型规模，而对数据质量的关注不够？

**方法**  考拉（Koala）使用来自公共数据集和互联网的对话数据进行微调。然而，[5]中的作者们着重强调了为微调精心策划高质量数据集的重要性。用于微调考拉（Koala）的数据大致可分为基于蒸馏的数据（即来自其他大型语言模型的对话）或开源数据（即存在于公共数据集中的数据），其中包括来自ShareGPT、HC3、OIG、Anthropic HH以及OpenAI WebGPT/摘要的数据。此外，微调集甚至包括用于训练Alpaca[3]模型的数据。

![|500](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F322813ba-a456-42e4-bb7a-31e983abf754_1970x686.png)

所有这些数据都是基于对话的。然而，值得注意的是，一些数据集针对每个问题包含多个被评为好或不好的对话或回复。有趣的是，我们可以利用之前的技术[8]将这些信息纳入到大型语言模型（LLM）的微调过程中。具体而言，这是通过条件训练来实现的，在这种训练中，我们可以简单地用人工偏好标记对用于训练LLM的数据进行条件设定（例如，只需追加关于对话是好还是不好的文本信息）；见上文。这种方法能够提升性能，并使我们甚至可以使用质量较低的对话来进行模型训练。

文献[5]的作者将用于训练和托管Koala的框架公开可用。该模型使用八个V100 GPU训练了两个轮次，大约需要6小时。总体而言，在可抢占/现货实例（假设我们可以使用可抢占/现货实例）的情况下，训练这个模型的计算成本不到100美元，这意味着Koala是我们迄今为止所见到的模型中复现成本最低的！

[Koala Code](https://github.com/young-geng/EasyLM)

**结果**  文献[5]中的作者训练了两种不同类型的Koala模型：

- Koala-distill：仅在蒸馏数据（即来自其他聊天机器人的对话示例）上进行微调
- Koala-all：使用上述所有数据进行微调

基于人体试验和反馈，将这些考拉模型的质量与阿尔帕卡（Alpaca）和ChatGPT的质量进行比较。为了进行评估，使用了阿尔帕卡[3]评估集中的问题以及来自互联网的一组真实用户查询。作者选择向评估集中添加更多问题，因为阿尔帕卡的评估集与其训练数据非常相似（即，二者均源自self-instruct[2]）。

![|500](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F540fe427-9302-45ca-8be6-16b63eac5f62_1702x864.png)

当人类从质量和正确性的角度评判不同大型语言模型（LLM）的输出时，发现“Koala-all”的表现常常超过“Alpaca”，并且在大量情况下其质量与“ChatGPT”相当甚至更优。此外，我们发现“Koala-distill”的表现实际上优于“Koala-all”。鉴于“Koala-distill”的微调数据集更小（即仅包含来自“ChatGPT”的示例对话），这一点有些违反直觉，但这告诉我们用于微调的数据类型和质量极其重要。也就是说，使用由更大、更好的大型语言模型生成的对话进行微调非常有效。

> _“构建强大的对话模型的关键可能更多地在于精心策划具有多样化用户查询的高质量对话数据。”_ 


### 更进一步……

尽管LLaMA提出的时间并不长，但Alpaca、Vicuna和Koala并非仅有的受LLaMA推动（或启发）的重要模型。下面我们可以看到近期发布的其他一些开源语言模型的列表。

- • Lit-LLaMA：在Apache-2.0许可证下开源的LLaMA复现版本（允许商业用途）。
- • ChatLLaMA：使用LLaMA、您自己的数据以及尽可能少的计算资源制作ChatGPT的个性化版本。
- • FreedomGPT：一个开源的聊天机器人（基于Alpaca），强调无审查。
- • ColossalChat：一个开源的ChatGPT复制品，附带一个完全实现的（并且公开的）基于LLaMA的RLHF管道（包括数据收集、监督微调、奖励模型训练和强化学习微调；见下文）。
- • StackLLaMA：为生产强大的聊天机器人提供基于RLHF的微调的开源实现和讨论（特别是以LLaMA为起点）。
- • GPT4All：基于LLaMA和GPT-J训练开源LLMs的演示、数据和代码（具有Apache-2.0许可证！）。
- • Baize：一个基于LLaMA的开源聊天机器人，使用LoRA（一种参数高效的微调方法）进行微调。
- • Galpaca：Galactica（科学领域的语言模型）的一个版本，已经在与Alpaca相同的数据集上进行了微调。
- • Dolly 2.0：该模型不基于LLaMA，而是一个经过指令微调达到ChatGPT类似质量并且开放商业使用的开源聊天机器人。
- • Open Assistant：一个开源聊天机器人（可与ChatGPT相媲美），能够理解任务、与第三方系统交互并检索信息。

![|500](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F718f0b30-eaa8-4952-be9d-7b7bc428d449_1618x1172.png)

除了提出的各种模型之外，由于LLaMA的出现，大型语言模型（LLM）的研究和使用也变得更加容易。LLaMA - 13B已经可以仅使用单个GPU来运行，但现在我们甚至可以在本地（例如，在苹果笔记本电脑上）进行运行！

- • Alpaca.cpp：在本地运行Alpaca的开源复现版本。
- • GPTQ - 4 - LLaMA：LLaMA的4位量化版本。
- • LLaMA.cpp：对几种开源大型语言模型进行4位量化推理，这使得可以在本地托管（例如，在苹果笔记本电脑上）。  
    似乎大型语言模型很快将比以往任何时候都更广泛地供人们使用。

## 要点

我们可以从这项工作中推断出的主要观点是：i) LLaMA激发了大量开源大型语言模型（LLM）的研究；ii) 由于LLaMA的存在，围绕大型语言模型的研究/使用变得更容易获取。如果一个月前有人告诉我，我能在我的苹果笔记本电脑上运行一个性能接近ChatGPT的大型语言模型，我不会相信的。这是一个令人兴奋的时代，我很感激能成为这样一个了不起的社区的一小部分！下面列出了一些基本的收获。

大型语言模型（LLMs）适用于所有人。如果我们之前对此还有所质疑，那么现在我们知道研究界确实可以对大型语言模型展开有价值的研究。几周前，我们大多数人都认为由于对数据和计算能力的极高要求，大型语言模型很难被广泛使用。然而，现在我们只需花费几百美元就能训练出具有ChatGPT水平（或至少接近其水平）的模型，甚至还能在笔记本电脑上使用这些模型进行对话！

较小的模型就足够了吗？长期以来，模型规模（连同大规模预训练数据集一起）一直是高性能大型语言模型（LLMs）的一个重要组成部分。然而，像Koala和Vicuna这样的模型向我们展示了较小的LLMs实际上可以表现得非常出色（甚至在某些情况下可以与ChatGPT等强大的LLMs相媲美）。这样的发现凸显了数据质量的重要性。在我们在这里看到的工作中，最有效的技术往往使用较大LLMs的输出作为训练数据，这表明知识蒸馏可能是创建小而强大的LLMs的一个重要组成部分。

商业上可行？尽管这些技术中的许多都很酷，但在商业应用中使用它们却很困难。例如，OpenAI禁止使用ChatGPT（或任何其他API模型）来训练竞争模型，从而阻止了基于OpenAI API的知识蒸馏方法。此外，即使是LLaMA本身也禁止商业使用。因此，像Alpaca、Koala和Vicuna这样的模型仅从研究角度而言是有趣的，它们的方法不能用于任何商业使用的模型。然而，有了像Lit-LLaMA这样的提议，这些模型的商业可行版本似乎可能会慢慢出现。

