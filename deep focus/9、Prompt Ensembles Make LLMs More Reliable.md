
任何使用过大语言模型（LLMs）的人都知道，提示工程是一个非正式且困难的过程。对提示进行细微改动可能会导致模型输出发生巨大变化，很难（甚至在某些情况下不可能）预知改变提示会产生什么影响，而且提示行为高度依赖于所使用的模型类型。当我们考虑用大语言模型构建应用程序时，提示工程的脆弱性是一个残酷的现实。如果我们无法预测模型的行为，又如何围绕这个模型构建一个可靠的系统呢？尽管大语言模型能力极强，但这个问题让它们在许多实际场景中的应用变得复杂。

> “提示工程是一个脆弱的过程，其中对提示的微小修改可能会导致模型预测出现较大变化，因此需要投入大量精力来设计一个极为完美的任务提示。” —— 引自[2]

鉴于大型语言模型（LLMs）的脆弱性，寻找使这些模型更准确、更可靠的技术最近已成为一个热门研究课题。在本概述中，我们将特别关注一种技术——提示集成（prompt ensembles）。简单来说，提示集成就是一组旨在解决同一问题的多样化提示。为了提高 LLM 的可靠性，我们可以通过使用多个不同的输入提示向 LLM 提问，并在推断最终答案时考虑模型的每个响应来生成问题的答案。正如我们将看到的，关于这个话题的一些研究相当技术性。然而，这些技术背后的基本思想很简单，并且可以极大地提高 LLM 的性能，使得提示集成成为提高 LLM 可靠性的首选方法。

### 什么是可靠性？

在我们研究可靠性时，为这一概念提供一个精确的定义是有用的。正如前文提到的，大语言模型可能相当脆弱。如果我们稍微改变一下输入，可能会得到截然不同的输出。通常情况下，大语言模型的输出既难以预测又不准确。Chip Huyen 在一篇近期的博客文章中对这些问题进行了广泛讨论。下面的引文概述了与传统编程任务相比，基于大语言模型构建应用程序的难度。

>“编程语言大多是精确的……而在提示工程中，指令是用自然语言编写的，自然语言比编程语言灵活得多……大语言模型生成响应中的歧义可能会成为致命问题。”

可靠性是解决方案。从高层次来看，可靠性指的是系统减轻噪声并抽象或避免大语言模型（LLM）出现不一致行为的能力。这可能意味着让大语言模型更加准确，或是提高模型行为的一致性或可预测性。如果我们想最大化大语言模型的效用，就必须找到方法让其行为更可靠，以便围绕它们构建应用程序，而不会出现破坏系统的意外“状况”。从实际操作层面来看，这意味着我们必须：

- 采用更严格/系统的方法进行提示工程。
- 寻找使大语言模型更可预测和更准确的技术。
- 当大语言模型未能符合我们期望的格式时，实施保障措施/设定边界。

上述每一点都是提高大语言模型可靠性的一个步骤。简单来说，我们只是想找到让大语言模型在应用中表现更一致的方法，这样终端用户就不会那么困惑，能获得更理想的体验。如果我们致力于以更严谨的方式运用大语言模型，就完全有可能降低提示工程的脆弱性，并减轻大语言模型的整体模糊性。在本概述中，我们将主要关注上述第二点——让大语言模型更具可预测性和准确性的技术。

### 用大型语言模型解决棘手问题

![|500](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F6d3436fa-babf-49f2-8973-4e19cbb80ea7_2556x1438.png)

尽管大型语言模型（LLMs）可以借助少样本学习等技术解决许多任务，但在解决多步问题或需要推理的问题时往往力不从心[15]。为此，近期研究探索了思维链（CoT）提示等技术[3]，包括自洽性[4]等几项显著的扩展方法，以提高大型语言模型的推理能力。通过这些研究我们了解到，语言模型本身已经具备解决复杂（基于推理的）问题的能力——我们只需采用正确的提示方法！

> 大型预训练语言模型具备内在的推理能力，但需要特定的提示来释放其潜能。——引自[1]

自洽性。给定一种思维链（CoT）提示方法，自洽性[4]可以通过以下方式提高大语言模型（LLM）的准确性：i) 从同一模型生成多个不同的输出；ii) 对每个输出的答案进行多数投票，并将多数票结果作为最终答案；具体如下所示。这种技术通过汇总一组多样化输出的结果来提高LLM的准确性。自洽性既简单又有效，这表明提高LLM可靠性的实用技术可能并非遥不可及。因此，我们可能会想：如何进一步采用这种方法？是否有其他更简单的、效果更好的技术？

![|300](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F646fdbc0-75b7-485f-aba4-041ea791afa8_910x1308.png)


提示集合（prompt ensembles）。自洽性的有效性源于在形成最终答案时所考虑的生成输出的多样性。然而，这项技术有一个关键细节值得我们注意——所有输出都是用同一个提示生成的。为了增加生成输出的多样性，我们可以考虑使用一组多样的多个提示来解决同一个问题。

>“人们思考方式不同，[但]不同的想法往往能得出相同的正确答案。” ——引自[1]

这种被称为提示集成的方法，与自洽性相比，可以用来生成更多样化的模型输出，从而进一步提高大型语言模型应用的可靠性。此外，提示集成易于理解，并且可以在无需大量实施工作的情况下自动构建。在本文中，我们将探讨有关提示集成的最新研究，重点关注使大型语言模型更有效的实用工具。

### 其他重要概念

除了迄今为止所涵盖的观点之外，还有一些在概述后面提到的小概念和术语，理解它们可能会有所帮助。

bootstrapping。这是一个在更广泛的计算机科学领域常用的通用术语。它指的是利用现有资源来做一些新的或有用的事情的理念。在本概述中，我们将“引导启动”用于描述将一个现有的预训练大语言模型作为系统的一个组件，用于生成用于集成学习的新提示。

弱监督。训练机器学习模型有许多不同的方法。弱监督是一种介于有监督学习和无监督学习之间的技术。它不像有监督学习那样完全依赖标记数据，但它确实使用某种形式的“标签”作为训练信号。例如，我们可能会使用某种启发式方法生成“伪标签”，甚至在训练过程中结合使用标记数据和未标记数据。

杰卡德指数（Jaccard Index）。在机器学习（ML）社区中，杰卡德指数通常被称为交并比（Intersection over Union，IoU），用于计算两个有限集合之间的相似度。要计算杰卡德指数的值，我们需要找出两个集合之间相交元素的数量，然后用这个数量除以两个集合的并集的大小。例如，如果我们有两个集合分别为{a, b, c}和{b, c, d}，那么杰卡德指数将是0.5（即两个元素相交，并且两个集合之间共有四个独特元素）。

### 提示集合研究

先前关于思维链（CoT）提示和自洽性的研究向我们表明，巧妙的提示策略可以大幅提高大语言模型可靠解决难题的能力。现在，我们将超越这些简单的基线方法，来看看近期关于将提示集合用于大语言模型的研究。这类研究提供了大量关于最佳实践的实际知识，我们可以采用这些最佳实践来提高大语言模型的可靠性。

### 多样化推理步骤验证器（DiVeRSE）[1]

>“人们思考方式不同，但不同的想法往往能得出相同的正确答案。” ——引自[1]

[1]中的作者探索了对思维链（CoT）和自洽性提示技术[3, 4]的扩展，这种扩展提高了在复杂的多步推理任务上的性能。这项被称为DiVeRSE的技术，使用提示集合（即旨在解决同一问题的不同提示的集合）来增强生成推理路径的多样性，然后训练一个验证模块来推断每个输出的正确性；

![|500](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F2c01f2cd-0ad1-4199-a91e-48ca1eb14bf7_1328x554.png)

自洽性和 DiVeRSE 都会生成多个推理路径，并将这些路径组合以形成最终答案。然而，自洽性是从大型语言模型（LLM）中使用相同的提示采样多个推理路径，而 DiVeRSE 则为解决同一问题构建多样化的提示集合，并从每个提示中采样多个推理路径。此外，自洽性仅通过对推理路径进行多数投票来形成其最终答案。DiVeRSE 采用了一种更复杂的方法：

1. 训练一个验证器/分类器来预测每个推理路径的正确性。
2. 根据正确性对推理路径进行加权平均。

简而言之，DiVeRSE 通过以下两方面提升大型语言模型（LLM）的推理能力：一是增强生成推理路径的多样性；二是在构建最终答案时，对可能正确的推理路径赋予更高权重。

![|500](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F849ebb29-b667-4835-9741-156c06268d72_2380x952.png)

构建提示集合。DiVeRSE 的主要优势之一在于它利用提示集合来最大化生成输出的多样性。但是，生成这些提示集合成本高吗？我们能自动构建提示集合吗？特别是考虑到思维链（CoT）提示，我们可以通过两种主要方式生成提示集合（见上图）：

1. 重新采样：给定一个包含问题、答案和用于少样本学习的 K 个理由的 CoT 提示，我们可以通过从全部理由中随机抽取 R < K 个示例来生成独特的提示。
2. 自举法：如果我们的 CoT 提示中没有足够的少样本示例来进行重新采样，我们可以简单地提示另一个大语言模型生成伪推理路径，以便在执行重新采样时纳入其中。

利用这些技术，我们可以自动生成 DiVeRSE 的提示集合，而无需大量的人工努力。

> 因果语言模型没有机制来纠正早期步骤中的先前错误，这会迅速导致混乱的结果。——引自[1]

验证模块。为了形成最终答案，DiVeRSE 使用一个验证模块来预测其生成的每条推理路径的正确性，然后根据这些预测取加权平均值。该验证器是一个二元分类器（例如，基于 BERT [5] 或 DeBERTa [6]），它是在由底层大语言模型生成的正确和错误推理路径的数据集上进行训练的。值得注意的是，需要标记数据来生成该验证器的数据集，因为这些标签用于确定任何推理路径的最终答案是否真正正确。

在测试时，该验证器用于为 DiVeRSE 生成的每条推理路径生成一个正确性得分，正确性得分较低的路径在最终答案中所占的投票权重较小。有趣的是，我们在[1]中发现，特别是执行逐步验证（即训练验证器预测每个单独推理步骤的正确性，而不是整个路径）可以显著提高推理性能；详见下文。

![|500](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F8cbc29f7-52fa-4a3b-9f91-55498d4ae5f8_1818x1224.png)

它的表现如何？研究人员使用多种不同的LLM（如OpenAI API中的davinci（GPT - 3）、text - davinci - 002（GPT - 3.5）和code - davinci - 002），将DiVeRSE与贪婪解码和自洽性等基线技术进行了比较。在八项不同的推理任务中，这些任务涉及算术推理、常识推理和归纳推理，DiVeRSE相比自洽性始终有提升；具体见下文。最值得注意的是，使用code - davinci - 002的DiVeRSE在总共六个基准测试上达到了最先进的性能，甚至超过了强大的、拥有5400亿参数的PaLM模型[7]。

![|500](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F7d5824fb-99ca-4e35-813f-d2a04fae7c91_2144x1038.png)

进一步研究表明[1]：其一，提示集成有助于提升推理性能；其二，当集成提示数量达到一定阈值后，推理性能趋于平稳；其三，使用验证器（尤其是分步验证器）的效果优于多数投票法（尽管多数投票法简单得多）。详见下文。

![|500](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F87dee7f9-8f6d-49a8-b699-5f702dbb3c85_1716x1366.png)


### 随便问（AMA）[2]

![|500](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fecb580cd-2079-4e1e-9732-1415100c0361_2482x1220.png)

[2]中的作者探索了有效构建和使用提示集的实用技术。从高层次来看，所提出的被称为“问我任何问题”（AMA）的技术，其动机在于消除构建“完美”提示的需求。相反，我们可以通过生成一组不完美（但仍然有效）的提示并汇总它们的结果，来设计一种有效且可靠的提示策略。但是，我们必须聪明地选择如何汇总这些提示的结果（即，简单多数投票的效果并不够好！）。此外，我们不能随意使用任何一组提示！特别是，我们在[2]中发现，最有效的提示集利用了那些鼓励开放式生成的提示。

虽然这听起来很棒，但我们可能有一些疑问。这对所有任务都适用吗？收集提示词成本高吗？我们应该如何汇总结果？[2]中提出的方法旨在兼具可扩展性和通用性，这意味着它可以用于提高任何模型或任务的性能。这种效率和效果源于三个主要理念：

1. 提示词结构：AMA强调使用开放式提示词，而非限制输出中标记的提示词。
2. 可扩展的提示词生成：AMA没有使用人工手动编写开放式提示词集合，而是使用大型语言模型（LLM）来生成和回答提示词，从而减少了所需的人力。
3. 弱监督：由于多数投票的效果不佳，AMA使用弱监督来学习提示词之间的依赖关系，并将大型语言模型的输出汇总成一个准确的最终答案。

为什么多数投票效果不佳？[2]中的研究工作不仅致力于提高集成中提示的质量和结构，还源于这样一个事实：通过多数投票的方式综合提示集成中大型语言模型（LLM）的输出以生成最终答案（例如自洽性方法[4]）效果很差。但为什么会这样呢？有趣的是，[2]中的作者给出了一个相当清晰直观的答案。

> “我们发现准确率的平均变化为9.5%，并且错误情况下的杰卡德指数比提示错误为独立同分布（i.i.d.）时高出69%。多数投票（MV）是先前工作中主要的无监督聚合策略，但它没有考虑任一特性，这使得它不可靠。” —— 来自[2]

换一种说法，大型语言模型（LLMs）所犯的错误并非随机分布！相反，针对多个不同提示词给出的模型输出可能会围绕一个错误的答案聚集。这对于多数投票机制而言是一个巨大的问题，因为错误的答案实际上可能会成为多数票的结果！为了解决这个问题，我们需要一种更复杂的策略，通过建模集成中提示词输出之间的准确性和依赖关系，来检测并处理此类情况。

构建一个出色的提示词集合。作为第一步，[2]中的作者研究了构成最有效集合的提示词类型。如下图所示，考虑了三种不同的提示技术类别。

![|500](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F586a7d7c-9a93-42f3-9b82-a14d15c6d945_2178x1126.png)

当对这些提示策略进行比较时，我们发现开放式提示格式（即完形填空式和自由形式）的表现优于要求大语言模型输出特定一组标记的限制性提示格式。更进一步而言，在自由形式提示中让问题更加精确或具体也能在一定程度上提高准确性。为什么自由形式提示的效果更好呢？答案并不完全清楚，但自由形式生成更贴近用于预训练大多数大语言模型的下一个标记预测任务，从直觉上讲，这意味着这类模型可能更擅长以这种格式解决问题。

![|500](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F5f91e3b2-8f57-44cf-9539-113c8f747434_2238x792.png)

受开放式提示有效性的启发，AMA通过生成一组关于给定输入的问题来构建提示集合；详见上文。这些问题采用自由格式，并强调输入的不同方面，可能提供有用且互补的信息。然而，手动创建这些问题可能代价高昂！为了避免这种情况，我们可以直接使用大型语言模型（LLM）！在文献[2]中可以看到，通过少样本学习，LLM可以用来生成关于所需主题的有用问题；详见下文。

![|500](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F87087f1d-9455-4fb7-a8e4-2beb1ee706e3_1400x396.png)

通过在AMA中改变所使用的上下文示例，并采用一组经过实证检验表现良好的提示模板，[2]中的作者完全自动化了提示集合的构建！

>“为了可靠地聚合提示预测结果，我们运用了弱监督工具，这是一种强大的方法，可在没有标注数据的情况下，从较弱的信号源中学习高质量模型。” —— 引自[2]

聚合结果。我们可以构建一个提示集合，但还有一个问题尚未解决：如何聚合每个提示下大语言模型的输出？尤其是对于自由形式的提示，从大语言模型的输出中提取正确答案可能相当困难。[2]中的聚合方法借鉴了弱监督和图模型[8, 9, 10]方面的前期工作。其高层次思路是利用弱监督来学习和预测不同提示之间的依赖关系以及每个提示的准确性。我们可以利用这些信息来聚合提示，并推断出最有可能的最终答案。与DiVeRSE不同，这种方法无需标记数据，并且解决了多数投票法的常见失败情况（例如，大语言模型在不同提示下产生相同的错误）。

AMA的表现如何？在[2]中，研究人员使用多种大语言模型（即四个不同的模型系列，包括BLOOM、OPT、EleutherAI和T0）在20个不同的基准测试上对AMA方法进行了测试，这些模型的参数规模从1.25亿到1750亿不等。该分析的目的是确定AMA是否是一种通用方法，可以适用于许多不同的设置。分析结果相当积极。使用AMA，我们发现小型开源模型（特别是GPT - J - 6B）的性能可以优于像GPT - 3这样的大型模型；具体情况见下文。

![|500](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F46a5251c-399f-4c84-a445-f108cbb00eac_1388x996.png)

在对所有不同模型进行测试后，我们发现中等规模的模型（即参数量在60亿到200亿之间）从自适应混合注意力（AMA）机制中获益最大，详见下文。与基线模型相比，少样本提示技术而言，AMA在所有模型和任务中的性能实现了约10%的绝对提升。因此，这是一种通用的方法，可以可靠地提升几乎所有大语言模型的性能。

![|500](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fb4bc9508-41cf-4af5-975f-fb1fcc80448d_2200x786.png)

AMA提供了一种构建提示集合的富有洞察力的方法。该出版物充满了实用的技巧，用于引导预训练的大型语言模型（LLM）编写有效的提示。我们在[2]中看到，用于聚合提示集合上LLM响应的方法论极其重要——简单多数投票是不够的！[2]中提出的聚合方法在技术上较为复杂，可能需要付出一定的实现努力，但它效果良好，并且不需要监督标签。通过采用AMA等方法，我们可以通过提高任何LLM的准确性和一致性来增强其可靠性。

> “我们希望AMA（提问与回答环节）及未来的工作能够通过提高在不够完美的提示下进行处理的能力，并支持使用小型、私有和开源的大语言模型，来帮助解决使用大语言模型时的痛点。” —— 引自[2]

### 结束语

希望我们现在应该明白，提示集成易于使用且具有巨大的潜力。要利用这项技术，我们只需做到以下几点：i) 构建一组旨在解决同一问题的多样化提示；ii) 使用每个提示生成多个大型语言模型输出；iii) 汇总这些输出以形成最终答案。正如我们所见，汇总过程可能有些复杂（即，简单多数投票通常是不够的）。然而，提示集成的构建和使用很简单，这使其成为提高大型语言模型可靠性的有力工具。以下是一些主要要点。

可靠性至关重要。要在现实世界中使用大型语言模型（LLMs），我们需要围绕它们构建软件系统。然而，为了构建围绕 LLMs 的软件系统，我们需要降低这些模型不可预测/模棱两可的特性带来的影响。提示集成（Prompt ensembles）提供了一种相当直接的方法来提高 LLMs 的准确性和可靠性。通过鼓励 LLM 针对解决特定问题生成一组多样化的输出，我们可以研究这些响应之间的关系，并开发自动技术以生成更高质量、最终的结果。

跨大型语言模型（LLMs）的泛化能力。通常，提示工程策略较为脆弱。如果我们调整提示，可能会得到截然不同的结果。同样，如果保持提示不变而更换模型，也会出现这种情况。如果我们构建了一个基于大型语言模型的应用程序，后来决定更换底层模型，那么很可能需要更改大部分使用的提示。然而，像AMA [2]这样的技术表明，提示集合可以缓解这一问题，因为它们能在多种不同模型上持续提升性能。因此，提示集合通过降低对底层模型的敏感性来提高可靠性。

聚合是困难的。在了解了自洽性之后，我曾乐观地认为通过简单的提示技巧可以让大语言模型（LLMs）变得更加可靠。正如我们在本概述中所看到的，并非总是如此。我们可以轻松地使用大语言模型为任何给定问题生成多个不同的输出，但我们聚合这些响应的方式至关重要。不幸的是，DiVeRSE和AMA提出的方法相当复杂，很可能需要付出大量的实施努力。然而，我们清楚地看到，仅仅采用多数投票法的效果不如更复杂的技术。希望很快会有更简单的聚合技术被提出。

局限性。尽管提示集成非常出色，但它并非完美无缺。像 DiVeRSE 和 AMA 这样的技术依赖于为每个回答的问题使用大语言模型生成大量输出。我们会使用多个提示词，甚至可能为每个提示词生成多个响应——这对大语言模型来说意味着大量的推理工作！因此，提示集成在成本上可能很高，无论是从资金方面还是从延迟角度来看都是如此。如果我们想在现实世界的应用程序中利用提示集成，就必须非常谨慎地应用它，因为它可能会极大地改变应用程序的成本和效率。
