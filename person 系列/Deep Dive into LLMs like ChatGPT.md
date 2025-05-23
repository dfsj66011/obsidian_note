
大家好，其实我一直想做这个视频。这是一个面向大众、全面介绍 ChatGPT 等大语言模型的视频。我希望通过这个视频，能帮助大家建立起理解这类工具运作方式的思维模型。

"它在某些方面显然充满魔力且令人惊叹。有些事它做得非常出色，另一些则不尽如人意，同时还有许多需要注意的尖锐问题。那么这个文本框背后究竟是什么？你可以输入任何内容并按下回车，但我们应该输入什么？这些生成的文字又是从何而来？它的运作原理是什么？你实际上是在和什么对话？我希望通过这个视频探讨所有这些话题。"

我们将完整梳理这类系统是如何构建的整个流程，但我会尽量让讲解通俗易懂，适合大众理解。首先我们来看看像 ChatGPT 这样的产品是如何打造的，在这个过程中我也会探讨这些工具背后涉及的认知心理学原理。好，现在让我们开始构建 ChatGPT。

### Step 1: download and preprocess the internet

整个流程将分为多个按顺序排列的阶段。*第一阶段称为预训练阶段*，而预训练的第一步是下载并处理互联网数据。为了让大家对此有个大致概念，我建议查看这个链接（FineWeb）。

有一家名为 HuggingFace 的公司收集并精心整理了一个名为 FineWeb 的数据集，他们在这篇博客文章中详细介绍了构建 FineWeb 数据集的过程。所有主要的 LLM 提供商，如 OpenAI、Anthropic、Google 等，内部都会有类似 FineWeb 这样的数据集。那么我们在这里试图实现什么目标呢？我们试图从公开可用的来源获取大量互联网文本。

因此我们正致力于获取大量质量极高的文档资料。同时我们也追求文档内容的广泛多样性，因为这些模型需要吸纳海量知识。简而言之，我们既需要大批优质文档，又要求这些文档覆盖领域足够宽广。实现这一目标相当复杂。正如你在这里看到的，需要多个阶段才能做好。接下来，我们稍微看看其中一些阶段的具体情况。

目前，我只想指出，以 FineWeb 数据集为例——它相当能代表生产级应用中的典型数据规模——实际占用的磁盘空间仅为 44TB 左右。如今，一个 U 盘就能轻松存储 1TB 数据，甚至几乎可以装进一块单独的硬盘里。因此归根结底，这并非海量数据。尽管互联网规模极其庞大，但我们处理的是文本数据，并且进行了严格筛选。在此案例中，最终获得的数据量约为 44TB。

那么让我们来看看这些数据的大致样貌，以及其中一些阶段的具体内容。许多这类工作的起点，同时也是最终贡献大部分数据来源的，是来自 Common Crawl 的数据。Common Crawl 是一个自 2007 年起就在持续抓取互联网数据的组织。

截至 2024 年，例如 Common Crawl 已索引了 27 亿个网页。他们让众多网络爬虫在互联网上持续抓取。其基本运作原理是：从少量种子网页出发，然后追踪所有链接进行爬取。你只需不断追踪链接，持续索引所有信息，久而久之就会积累大量互联网数据。因此，这通常是许多此类工作的起点。不过，Common Crawl 的数据相当原始，需要经过多种方式的过滤处理。

因此，他们在此记录——这是同一张图表——略微阐述了这些阶段中发生的处理流程。首先是一个名为 URL 过滤的环节。这里指的是存在一些屏蔽列表，本质上就是你不希望从中获取数据的域名或 URL 清单。因此，通常这包括恶意软件网站、垃圾邮件网站、营销网站、种族主义网站、成人网站等类似内容。在这一阶段，我们会剔除大量此类网站，因为我们不希望它们出现在数据集中。第二部分是文本提取。

你必须记住，所有这些网页，这些爬虫保存的就是这些网页的原始 HTML 代码。所以当我在这里检查时，这就是原始 HTML 实际呈现的样子。你会注意到它包含所有这些标记，比如列表之类的东西，还有 CSS 以及所有这些内容。所以这几乎就是这些网页的计算机代码。但我们真正想要的只是这段文本，对吧？我们只需要网页的文本内容，而不需要导航栏之类的东西。因此，需要大量的过滤、处理和启发式方法，才能充分筛选出这些网页中的优质内容。

下一阶段是语言过滤。例如，FineWeb 会使用语言分类器进行过滤。他们会尝试猜测每个网页所使用的语言，然后只保留英语内容占比超过 65% 的网页。因此你可以理解，这就像不同公司可以自行决定的设计策略。我们要在数据集中涵盖多大比例的不同类型语言？举例来说，如果过滤掉所有西班牙语数据，你可能会想到，后续模型在处理西班牙语时表现不佳，因为它从未接触过足够多的该语言数据。不同公司对多语言性能的重视程度可以有所差异，这便是一个例子。

FineWeb 主要专注于英语。因此，如果他们后续训练语言模型，该模型在英语方面会非常出色，但在其他语言上可能表现不佳。经过语言过滤后，还会进行其他几步筛选步骤，包括去重等处理。例如以去除个人身份信息（PII）作为收尾工作。这类信息包括地址、社保号码等敏感数据。我们需要在数据集中检测此类内容，并将含有这类信息的网页过滤剔除。这里有很多步骤，我就不详细展开了，但这是预处理中相当重要的一部分。最终你会得到像 FineWeb 这样的数据集。点击进入后，你可以看到一些实际处理后的样例展示。

任何人都可以在 Hugging Face 网页上下载这些内容。以下是最终进入训练集的文本示例，这是一篇关于 2012 年龙卷风的文章。2012 年发生了一些龙卷风事件，接下来要讲的内容有点特别——你知道吗？人体内有两个像 9 伏黄色小电池大小的肾上腺。好吧，这算是一篇有点奇怪的医学文章。你可以把这些内容想象成互联网上经过各种方式筛选后仅保留文字的网页。

而现在我们拥有了海量文本数据，足足 40TB。这些数据正是当前阶段下一步工作的起点。此刻，我想让大家直观地了解我们目前所处的阶段。

于是我提取了这里的前 200 个网页——要知道我们手头有海量数据——把所有文本内容抓取出来拼接在一起。最终得到的就是这个：一段未经处理的原始文本，最原生态的网络文本。这段文本数据包含了所有这些模式。而我们现在要做的是开始在这些数据上训练神经网络，以便神经网络能够内化并建模文本的流动方式。

### Step 2: tokenization

所以我们手头有了这么一大段文本素材，现在需要构建能模仿它的神经网络。不过在将文本输入神经网络之前，我们必须先确定*如何表示这些文本以及如何输入*。当前我们的技术方案是：这些神经网络需要接收一维的符号序列，并且要求这些符号来自有限的预设集合。

因此我们必须先确定符号体系，再将数据表示为这些符号的一维序列。目前我们拥有的是一维文本序列——它从这里开始，延伸至此，接着转向此处，如此往复。

虽然这段文字在我的显示器上是以二维方式排列的，但它实际上是一个一维序列，从左到右、从上到下阅读，对吧？这就是一个一维的文本序列。既然是计算机处理，自然存在底层的数据表示。如果我用 UTF-8 编码这段文本，就能获得计算机中对应这些文本的原始比特数据。

它的呈现形式是这样的。举个例子，这里的第一根柱状图实际上代表前八位二进制数据。那么这到底是什么呢？从某种意义上说，这正是我们要寻找的数据表现形式。我们只有两种可能的符号，0 和 1，并且有一个非常长的序列，对吧？但实际上，这个序列长度在我们的神经网络中是一种非常有限且宝贵的资源，我们并不希望只有两个符号却生成极其冗长的序列。相反，我们需要在符号集（即词汇表）的大小与最终序列长度之间做出权衡。因此，我们不想仅用两个符号却得到超长的序列。

我们需要更多的符号和更短的序列。那么，一种简单的压缩或缩短序列长度的方法是：将连续的比特位（例如八位）组合成一个称为“字节”的单元。由于这些比特位只有开或关两种状态，如果我们取八位一组，实际上只存在 256 种可能的开关组合。因此，我们可以将这个序列重新表示为字节序列。这样字节序列的长度将缩短为原来的八分之一，但我们现在有 256 种可能的符号。这里的每个数字范围都是从 0 到 255。现在，我真心建议你们不要把这些当作数字，而是看作独特的 ID 或符号。或许更恰当的做法是……用独特的表情符号来替换每一个数字。这样你就会得到类似这样的结果。所以我们基本上有一个表情符号序列，共有 256 种可能的符号。你可以这样理解。但事实证明，在生产最先进的语言模型时，你实际上需要超越这个范围。

你希望继续缩短序列长度，因为这同样是一种宝贵的资源，用以换取词汇表中更多的符号。实现这一目标的方法是运行所谓的字节对编码算法。其工作原理是，我们本质上是在寻找那些频繁出现的连续字节或符号。例如，我们发现序列 116 后接 32 的情况非常普遍且频繁出现。因此，我们将把这组配对合并为一个新符号。具体来说，我们将创建一个ID为 256 的符号，并将所有 116、32 的配对替换为这个新符号。

然后我们可以根据需要多次迭代这个算法。每次生成新符号时，都会减少序列长度并增大符号规模。实践证明，将词汇表大小设定为约 10 万个可能的符号是较为理想的选择。具体而言，GPT-4 使用了 100,277 个符号。这种将原始文本转换为这些符号（我们称之为标记）的过程就称为*标记化(tokenization)*。现在让我们来看看 GPT-4 是如何执行标记化的——如何将文本转换为标记，又如何将标记转换回文本，以及这一过程实际呈现的效果。

有一个我常用来探索这些标记表示的网站叫 TickTokenizer。在这里的下拉菜单中选择 CL100K Base，这是 GPT-4 基础模型的标记器。左侧可以输入文本，它会显示该文本的标记化结果。例如，“hello world” 实际上恰好由两个 token 组成：一个是 ID 为 15,339 的 “hello” token，另一个是 ID 为 1,917 的 “ world” token。因此，“hello world” 就表示为 “hello world”。

现在，如果我要将这两个部分合并，比如，我会再次得到两个标记，但这次是标记 “h” 和 “elloworld”。如果我在 hello 和 world 之间加入两个空格，又会得到不同的标记化结果。这里会出现一个新标记 220。你可以自己尝试调整，看看会发生什么变化。另外请注意，这是区分大小写的。所以如果是大写的 H，那就是另一个东西了。或者如果是 "HELLO WORLD"，实际上这会变成三个token，而不仅仅是两个 token。没错，你可以通过这个工具来把玩体验，直观感受这些标记（token）的工作原理。视频后面我们还会再回过头来深入讲解标记化（tokenization）的部分。现在，我只是想先带大家看看这个网站。

我想向你展示的是，这段文字归根结底可以这样理解：例如，如果我在这里选取一行，GPT-4 将会这样解析它。这段文本将是一个长度为 62 的序列，具体序列如下所示。这就是文本块与这些符号的对应关系。同样地，这里共有 100,277 种可能的符号。现在我们得到的是这些符号的一维序列。

好的，我们稍后会再回到分词这个话题，但现在我们就讲到这里。那么，我现在所做的是：我选取了数据集中这段文本序列，并用我们的分词器将其表示为一个标记序列。这就是它现在的样子。

例如，当我们回到 FindWeb 数据集时，他们提到这不仅占据了 44TB 的磁盘空间，而且该数据集包含约 15T 个标记序列。这里展示的只是该数据集前几千个标记中的一小部分。但请记住，整个数据集实际上包含 15T 个标记。

### Step 3: neural network training

请再次记住，所有这些都代表小的文本片段。它们就像是这些序列的原子。这里的数字没有任何意义。它们只是唯一的标识符。好了，现在我们进入有趣的部分——*神经网络训练*。

这里正是训练神经网络时大量计算工作发生的核心环节。在此步骤中，我们的目标是建模这些标记在序列中如何相互跟随的统计关系。具体操作是：我们提取数据中的标记窗口进行分析。因此，我们会从这些数据中随机抽取一个 token 窗口。窗口的长度可以从零个 token 开始，一直延伸到我们设定的某个最大值。例如，在实际操作中，你可能会看到 8000 个 token 的窗口。

理论上，我们可以使用任意长度的 token 窗口。但处理非常长的窗口序列在计算上会非常昂贵。因此，我们通常会选择一个合适的数字，比如 8000、4000 或 16000，并在此处进行截断。在这个例子中，我将选取前四个标记以确保内容整齐呈现。我们将截取这四个标记——"I"、"View"、"ing" 和 " Single"（即这些标记 ID）作为一个四标记窗口。现在我们要做的，本质上是尝试预测这个序列中下一个即将出现的标记。

#### 3.1 网络 I/O

所以接下来是 3962，对吧？我们现在在这里做的就是将其称为上下文。这四个标记就是上下文，它们会输入到神经网络中。这就是神经网络的输入。现在，我将稍后详细讲解这个神经网络内部的构造。目前，重要的是理解神经网络的输入和输出。输入是可变长度的标记序列，长度范围从零到某个最大值，比如 8,000。

现在的输出是对接下来内容的预测。由于我们的词汇表包含 100,277 个可能的标记，神经网络将输出恰好对应这些标记数量的数值，每个数值都代表该标记作为序列中下一个出现的概率。因此，它是在对接下来出现的内容进行预测。

最初，这个神经网络是随机初始化的。稍后我们会具体解释这意味着什么。但本质上，它是一次随机变换。因此，在训练的最初阶段，这些概率值也具有一定的随机性。这里我举了三个例子，但请记住实际数据集中包含 10 万个数字。当前神经网络给出的概率显示，这个 " Direction" 标记的出现概率暂时被预测为 4%。11,799 的概率为 2%。而此处，3962（" Post"）的概率为 3%。当然，我们已从数据集中对该窗口进行了抽样。

因此我们已知道接下来的数字。我们知道——这就是标签——正确的答案是序列中接下来实际出现的数字是 3962。现在我们所掌握的是这个用于更新神经网络的数学运算过程。我们有办法调整它。稍后我们会详细讨论这一点。但基本上，我们知道这里 3% 的概率，我们希望这个概率能更高一些。我们希望所有其他标记的概率都更低。因此，我们有一种数学方法来计算如何调整和更新神经网络，使正确答案的概率略微提高。如果我现在对神经网络进行一次更新，下次当我将这组特定的四个标记序列输入神经网络时，经过微调的神经网络可能会将 " Post" 的概率输出为 4%。

" Case" 的概率可能是 1%。而 " Direction" 可能会变为 2% 或类似数值。因此，我们有一种微调方法，可以稍微更新神经网络，使其对序列中下一个正确标记给出更高的概率预测。现在我们需要记住的是，这一过程不仅仅发生在这里的这一个标记上——即这四个输入预测出这一个的情况。实际上，该过程会同时作用于整个数据集中所有标记。因此在实践中，我们会采样小窗口，即分批处理这些小窗口数据。

接着，在每一个这样的标记处，我们都要调整神经网络，使得该标记出现的概率略微提高。这一切都是在大批量标记的并行处理中完成的。这就是训练神经网络的过程。这是一个不断更新的过程，目的是让模型的预测结果与训练集中实际发生的统计数据相匹配。其概率分布会逐步调整，以符合数据中这些标记(token)相互跟随的统计规律。

#### 3.2 网络内部结构

接下来让我们简要探讨这些神经网络的内部机制，以便你对它们的运作原理有个基本认识。

神经网络内部机制。如前所述，我们接收的输入是 token 序列。本例中虽然只展示 4 个输入token，但实际可处理范围从零到约 8,000 个 token 不等。理论上，这可以是一个无限数量的标记。只是处理无限数量的标记在计算上过于昂贵。因此我们仅将其截断至特定长度，这就成为该模型的最大上下文长度。

现在，这些输入 $x$ 与神经网络参数（或称权重）在一个庞大的数学表达式中混合运算。此处我展示了六个参数示例及其设定值，但实际上现代神经网络会拥有数十亿个这样的参数。最初，这些参数是完全随机设定的。现在，在参数随机设置的情况下，你可能会预期这个神经网络会做出随机预测，事实也确实如此。一开始，它的预测完全是随机的。

但正是通过这种不断更新网络的迭代过程——我们称之为神经网络的训练——这些参数的设置得以调整，从而使神经网络的输出与训练集中观察到的模式保持一致。你可以把这些参数想象成 DJ 调音台上的旋钮。当你转动这些旋钮时，针对每个可能的标记序列输入，都会得到不同的预测结果。

训练神经网络，其实就是寻找一组与训练集统计数据相吻合的参数。现在，我举个实例展示这个庞大数学表达式的样貌，让你有个直观感受。现代神经网络很可能是包含数万亿项的巨型表达式。不过，让我在这里给你看一个简单的例子。它看起来大概是这样的。我是说，这些就是那种表达式，只是想告诉你它并不可怕。

我们有一些输入 $x$，比如 $x_1$、$x_2$，在这个例子中有两个示例输入，它们会与网络的权重 $w_0$、$w_1$、$w_2$、$w_3$ 等混合。这种混合涉及简单的数学运算，如乘法、加法、指数运算、除法等。神经网络架构研究的主题就是设计具有诸多便利特性的有效数学表达式。

它们表现力强、可优化、可并行化等等。但归根结底，这些并非复杂的表达式。本质上，它们通过将输入与参数混合来进行预测。我们正在优化这个神经网络的参数，以使预测结果与训练集保持一致。现在，我想向你们展示一个实际生产级别的神经网络示例。为此，我建议大家访问这个网站，它提供了这些神经网络非常直观的可视化效果。这就是您在本网站将了解到的内容。而这里应用于生产环境的神经网络具有这种特殊结构。该网络被称为 Transformer。

以这个具体模型为例，它大约包含 85,000 个参数。现在我们来看顶部结构：输入层接收词元序列作为输入，信息随后在神经网络中向前传播，最终输出层会生成经过 softmax 处理的逻辑值。但这些预测针对的是接下来会出现什么，即下一个标记是什么。在这里，有一系列转换过程，以及在这个数学表达式中生成的所有中间值，它们共同作用以预测后续内容。举例来说，这些标记会被嵌入到所谓的分布式表示中。

因此，每个可能的标记在神经网络内部都有一个代表它的向量。首先，我们将这些标记进行嵌入。然后，这些值会以某种方式在这个图中流动。这些单独来看都是非常简单的数学表达式。比如我们有层归一化、矩阵乘法、softmax 函数等等。这里大致就是这个 Transformer 中的注意力模块。

然后信息会流入多层感知器模块，依此类推。这里的这些数字都是表达式的中间值。你几乎可以把它们想象成这些合成神经元的放电频率。

但我得提醒你，别太把它当神经元来理解，因为这些与你大脑中的神经元相比极其简单。你体内的生物神经元是具有记忆等功能的非常复杂的动态过程，而这个表达式里可没有记忆这回事。这是一个从输入到输出固定不变的数学表达式，没有记忆功能，完全无状态。因此，与生物神经元相比，这些神经元非常简单。

但你可以大致将其视为一种人造的脑组织——如果你倾向于这样理解的话。信息流经所有这些神经元，直到我们得出预测结果。不过，我不会过多纠结于这些转换过程中精确的数学细节。老实说，我认为深入探讨这一点并不那么重要。真正需要理解的是，这是一个数学函数。它由一组固定参数所定义，比如大约 85,000个。

这是一种将输入转化为输出的方式。当我们调整参数时，会得到不同类型的预测结果。随后我们需要找到这些参数的合适配置，使预测结果能够与训练集中观察到的模式大致吻合。这就是 Transformer 模型。好了，我已经向你们展示了这个神经网络的内部结构，我们也稍微讨论了训练它的过程。

### Step 4: inference

接下来我想介绍使用这些网络的另一个主要阶段，那就是被称为推理的阶段。

因此在推理过程中，我们所做的是从模型中生成新数据。我们本质上想观察模型通过其网络参数内化了哪些模式。从模型生成数据的过程相对直接。我们从一些基本作为前缀的标记开始，比如你想要的起始内容。假设我们希望以标记 91 开头，那么我们就将其输入网络。

记住，网络给出的是概率，对吧？它在这里给出的是这个概率向量。所以我们现在可以做的，本质上就是抛一个有偏见的硬币。也就是说，我们可以基于这个概率分布来采样一个标记。

So for example, token 860 comes next. So 860 in this case, when we're generating from model, could come next. Now, 860 is a relatively likely token. 

It might not be the only possible token in this case. There could be many other tokens that could have been sampled. But we could see that 860 is a relatively likely token as an example. 

And indeed, in our training example here, 860 does follow 91. So let's now say that we continue the process. So after 91, there's 860. 

We append it. And we again ask, what is the third token? Let's sample. And let's just say that it's 287, exactly as here.

Let's do that again. We come back in. Now we have a sequence of three. 

And we ask, what is the likely fourth token? And we sample from that and get this one. And now let's say we do it one more time. We take those four. 

We sample. And we get this one. And this 13659, this is not actually 3962 as we had before. 

So this token is the token article instead. So viewing a single article. And so in this case, we didn't exactly reproduce the sequence that we saw here in the training data. 

So keep in mind that these systems are stochastic. We're sampling. And we're flipping coins. 

And sometimes we luck out. And we reproduce some small chunk of the text in the training set. But sometimes we're getting a token that was not verbatim part of any of the documents in the training data. 

So we're going to get sort of like remixes of the data that we saw in the training. Because at every step of the way, we can flip and get a slightly different token. And then once that token makes it in, if you sample the next one and so on, you very quickly start to generate token streams that are very different from the token streams that occur in the training documents. 

So statistically, they will have similar properties, but they are not identical to training data. They're kind of like inspired by the training data. And so in this case, we got a slightly different sequence. 

And why would we get article? You might imagine that article is a relatively likely token in the context of bar, viewing, single, et cetera. And you can imagine that the word article followed this context window somewhere in the training documents to some extent. And we just happen to sample it here at that stage. 

So basically, inference is just predicting from these distributions one at a time, we continue feeding back tokens and getting the next one. And we're always flipping these coins. And depending on how lucky or unlucky we get, we might get very different kinds of patterns, depending on how we sample from these probability distributions. 

So that's inference. So in most common scenarios, basically, downloading the internet and tokenizing it is a preprocessing step.

(该文件长度超过30分钟。 在TurboScribe.ai点击升级到无限，以转录长达10小时的文件。)


(转录由TurboScribe.ai完成。升级到无限以移除此消息。)

And then, once you have your token sequence, we can start training networks. And in practical cases, you would try to train many different networks of different kinds of settings and different kinds of arrangements and different kinds of sizes. And so you'd be doing a lot of neural network training.

And then once you have a neural network and you train it, and you have some specific set of parameters that you're happy with, then you can take the model and you can do inference. And you can actually generate data from the model. And when you're on ChatGPT and you're talking with a model, that model is trained and has been trained by OpenAI many months ago, probably.

And they have a specific set of weights that work well. And when you're talking to the model, all of that is just inference. There's no more training.

Those parameters are held fixed. And you're just talking to the model, sort of. You're giving it some of the tokens, and it's kind of completing token sequences.

And that's what you're seeing generated when you actually use the model on ChatGPT. So that model then just does inference alone. So let's now look at an example of training and inference that is kind of concrete and gives you a sense of what this actually looks like when these models are trained.

Now the example that I would like to work with and that I'm particularly fond of is that of OpenAI's GPT2. So GPT stands for Generatively Pre-trained Transformer. And this is the second iteration of the GPT series by OpenAI.

When you are talking to ChatGPT today, the model that is underlying all of the magic of that interaction is GPT4, so the fourth iteration of that series. Now GPT2 was published in 2019 by OpenAI in this paper that I have right here. And the reason I like GPT2 is that it is the first time that a recognizably modern stack came together.

So all of the pieces of GPT2 are recognizable today by modern standards. It's just everything has gotten bigger. Now I'm not going to be able to go into the full details of this paper, of course, because it is a technical publication.

But some of the details that I would like to highlight are as follows. GPT2 was a transformer neural network, just like the neural networks you would work with today. It had 1.6 billion parameters, right? So these are the parameters that we looked at here.

It would have 1.6 billion of them. Today, modern transformers would have a lot closer to a trillion or several hundred billion, probably. The maximum context length here was 1,024 tokens.

So it is when we are sampling chunks of windows of tokens from the data set, we're never taking more than 1,024 tokens. And so when you are trying to predict the next token in a sequence, you will never have more than 1,024 tokens kind of in your context in order to make that prediction. Now this is also tiny by modern standards.

Today, the context lengths would be a lot closer to a couple hundred thousand or maybe even a million. And so you have a lot more context, a lot more tokens in history. And you can make a lot better prediction about the next token in a sequence in that way.

And finally, GPT2 was trained on approximately 100 billion tokens. And this is also fairly small by modern standards. As I mentioned, the fine web data set that we looked at here, the fine web data set has 15 trillion tokens.

So 100 billion is quite small. Now, I actually tried to reproduce GPT2 for fun as part of this project called LLM.C. So you can see my write-up of doing that in this post on GitHub under the LLM.C repository. So in particular, the cost of training GPT2 in 2019 was estimated to be approximately $40,000.

But today, you can do significantly better than that. And in particular, here, it took about one day and about $600. But this wasn't even trying too hard.

I think you could really bring this down to about $100 today. Now, why is it that the costs have come down so much? Well, number one, these data sets have gotten a lot better. And the way we filter them, extract them, and prepare them has gotten a lot more refined.

And so the data set is of just a lot higher quality. So that's one thing. But really, the biggest difference is that our computers have gotten much faster in terms of the hardware.

And we're going to look at that in a second. And also, the software for running these models and really squeezing out all the speed from the hardware as it is possible, that software has also gotten much better as everyone has focused on these models and trying to run them very, very quickly. Now, I'm not going to be able to go into the full detail of this GPT2 reproduction.

And this is a long technical post. But I would like to still give you an intuitive sense for what it looks like to actually train one of these models as a researcher. Like, what are you looking at? And what does it look like? What does it feel like? So let me give you a sense of that a little bit.

OK, so this is what it looks like. Let me slide this over. So what I'm doing here is I'm training a GPT2 model right now.

And what's happening here is that every single line here, like this one, is one update to the model. So remember how here we are basically making the prediction better for every one of these tokens. And we are updating these weights or parameters of the neural net.

So here, every single line is one update to the neural network, where we change its parameters by a little bit so that it is better at predicting next token and sequence. In particular, every single line here is improving the prediction on 1 million tokens in the training set. So we've basically taken 1 million tokens out of this data set.

And we've tried to improve the prediction of that token as coming next in a sequence on all 1 million of them simultaneously. And at every single one of these steps, we are making an update to the network for that. Now, the number to watch closely is this number called loss.

And the loss is a single number that is telling you how well your neural network is performing right now. And it is created so that low loss is good. So you'll see that the loss is decreasing as we make more updates to the neural net, which corresponds to making better predictions on the next token in the sequence.

And so the loss is the number that you are watching as a neural network researcher. And you are kind of waiting. You're twiddling your thumbs.

You're drinking coffee. And you're making sure that this looks good so that with every update, your loss is improving. And the network is getting better at prediction.

Now, here you see that we are processing 1 million tokens per update. Each update takes about 7 seconds roughly. And here we are going to process a total of 32,000 steps of optimization.

So 32,000 steps with 1 million tokens each is about 33 billion tokens that we are going to process. And we're currently only about 420, step 420 out of 32,000. So we are still only a bit more than 1% done.

Because I've only been running this for 10 or 15 minutes or something like that. Now, every 20 steps, I've configured this optimization to do inference. So what you're seeing here is the model is predicting the next token in the sequence.

And so you sort of start it randomly. And then you continue plugging in the tokens. So we're running this inference step.

And this is the model sort of predicting the next token in the sequence. And every time you see something appear, that's a new token. So let's just look at this.

And you can see that this is not yet very coherent. And keep in mind that this is only 1% of the way through training. And so the model is not yet very good at predicting the next token in the sequence.

So what comes out is actually kind of a little bit of gibberish, right? But it still has a little bit of like local coherence. So since she is mine, it's a part of the information. Should discuss my father, great companions, Gordon showed me sitting over it, and et cetera.

So I know it doesn't look very good. But let's actually scroll up and see what it looked like when I started the optimization. So all the way here at step 1. So after 20 steps of optimization, you see that what we're getting here looks completely random.

And of course, that's because the model has only had 20 updates to its parameters. And so it's giving you random text because it's a random network. And so you can see that at least in comparison to this, the model is starting to do much better.

And indeed, if we waited the entire 32,000 steps, the model will have improved the point that it's actually generating fairly coherent English. And the tokens stream correctly. And they kind of make up English a lot better.

So this has to run for about a day or two more now. And so at this stage, we just make sure that the loss is decreasing. Everything is looking good.

And we just have to wait. And now, let me turn now to the story of the computation that's required. Because of course, I'm not running this optimization on my laptop.

That would be way too expensive. Because we have to run this neural network. And we have to improve it.

And we need all this data and so on. So you can't run this too well on your computer. Because the network is just too large.

So all of this is running on a computer that is out there in the cloud. And I want to basically address the compute side of the story of training these models and what that looks like. So let's take a look.

Okay, so the computer that I'm running this optimization on is this 8xh100 node. So there are eight h100s in a single node or a single computer. Now, I am renting this computer.

And it is somewhere in the cloud. I'm not sure where it is physically, actually. The place I like to rent from is called Lambda.

But there are many other companies who provide this service. So when you scroll down, you can see that they have some on-demand pricing for sort of computers that have these h100s, which are GPUs. And I'm going to show you what they look like in a second.

But on-demand 8xh100 GPU. This machine comes for $3 per GPU per hour, for example. So you can rent these.

And then you get a machine in the cloud. And you can go in and you can train these models. And these GPUs, they look like this.

So this is one h100 GPU. This is kind of what it looks like. And you slot this into your computer.

And GPUs are this perfect fit for training neural networks because they are very computationally expensive. But they display a lot of parallelism in the computation. So you can have many independent workers kind of working all at the same time in solving the matrix multiplication that's under the hood of training these neural networks.

So this is just one of these h100s. But actually, you would put multiple of them together. So you could stack eight of them into a single node.

And then you can stack multiple nodes into an entire data center or an entire system. So when we look at a data center, we start to see things that look like this, right? We have one GPU goes to eight GPUs, goes to a single system, goes to many systems. And so these are the bigger data centers.

And they, of course, would be much, much more expensive. And what's happening is that all the big tech companies really desire these GPUs so they can train all these language models because they are so powerful. And that is fundamentally what has driven the stock price of NVIDIA to be $3.4 trillion today, as an example, and why NVIDIA has kind of exploded.

So this is the gold rush. The gold rush is getting the GPUs, getting enough of them so they can all collaborate to perform this optimization. And what are they all doing? They're all collaborating to predict the next token on a dataset like the FindWeb dataset.

This is the computational workflow that basically is extremely expensive. The more GPUs you have, the more tokens you can try to predict and improve on. And you're going to process this dataset faster.

And you can iterate faster and get a bigger network and train a bigger network and so on. So this is what all those machines are doing. And this is why all of this is such a big deal.

And for example, this is an article from about a month ago or so. This is why it's a big deal that, for example, Elon Musk is getting 100,000 GPUs in a single data center. And all of these GPUs are extremely expensive, are going to take a ton of power.

And all of them are just trying to predict the next token in the sequence and improve the network by doing so. And get probably a lot more coherent text than what we're seeing here a lot faster. Okay, so unfortunately, I do not have a couple 10 or 100 million of dollars to spend on training a really big model like this.

But luckily, we can turn to some big tech companies who train these models routinely and release some of them once they are done training. So they've spent a huge amount of compute to train this network. And they release the network at the end of the optimization.

So it's very useful because they've done a lot of compute for that. So there are many companies who train these models routinely. But actually, not many of them release these what's called base models.

So the model that comes out at the end here is what's called a base model. What is a base model? It's a token simulator, right? It's an internet text token simulator. And so that is not by itself useful yet.

Because what we want is what's called an assistant. We want to ask questions and have it respond to answers. These models won't do that.

They just create sort of remixes of the internet. They dream internet pages. So the base models are not very often released because they're kind of just only a step one of a few other steps that we still need to take to get an assistant.

However, a few releases have been made. So as an example, the GPT-2 model released the 1.6 billion, sorry, 1.5 billion model back in 2019. And this GPT-2 model is a base model.

Now, what is a model release? What does it look like to release these models? So this is the GPT-2 repository on GitHub. Well, you need two things basically to release model. Number one, we need the Python code usually that describes the sequence of operations in detail that they make in their model.

So if you remember back this transformer, the sequence of steps that are taken here in this neural network is what is being described by this code. So this code is sort of implementing the, what's called forward pass of this neural network. So we need the specific details of exactly how they wired up that neural network.

So this is just computer code and it's usually just a couple hundred lines of code. It's not that crazy. And this is all fairly understandable and usually fairly standard.

What's not standard are the parameters. That's where the actual value is. Where are the parameters of this neural network? Because there's 1.6 billion of them and we need the correct setting or a really good setting.

And so that's why in addition to this source code, they release the parameters, which in this case is roughly 1.5 billion parameters. And these are just numbers. So it's one single list of 1.5 billion numbers.

The precise and good setting of all the knobs such that the tokens come out well. So you need those two things to get a base model release. Now, GPT-2 was released, but that's actually a fairly old model as I mentioned.

So actually the model we're going to turn to is called LLAMA3. And that's the one that I would like to show you next. So LLAMA3, so GPT-2 again was 1.6 billion parameters trained on 100 billion tokens.

LLAMA3 is a much bigger model and much more modern model. It is released and trained by Meta. And it is a 405 billion parameter model trained on 15 trillion tokens.

In very much the same way, just much, much bigger. And Meta has also made a release of LLAMA3. And that was part of this paper.

So with this paper that goes into a lot of detail, the biggest base model that they released is the LLAMA3.1 405 billion parameter model. So this is the base model. And then in addition to the base model, you see here foreshadowing for later sections of the video, they also released the instruct model.

And the instruct means that this is an assistant. You can ask it questions and it will give you answers. We still have yet to cover that part later.

For now, let's just look at this base model, this token simulator, and let's play with it and try to think about, you know, what is this thing and how does it work? And what do we get at the end of this optimization if you let this run until the end for a very big neural network on a lot of data? So my favorite place to interact with the base models is this company called Hyperbolic, which is basically serving the base model of the 405B LLAMA3.1. So when you go into the website, and I think you may have to register and so on, make sure that in the models, make sure that you are using LLAMA3.1 405 billion base. It must be the base model. And then here, let's say the max tokens is how many tokens we're going to be generating.

So let's just decrease this to be a bit less just so we don't waste compute. We just want the next 128 tokens and leave the other stuff alone. I'm not going to go into the full detail here.

Now, fundamentally, what's going to happen here is identical to what happens here during inference for us. So this is just going to continue the token sequence of whatever prefix you're going to give it. So I want to first show you that this model here is not yet an assistant.

So you can, for example, ask it, what is two plus two? It's not going to tell you, oh, it's four. What else can I help you with? It's not going to do that. Because what is two plus two is going to be tokenized.

And then those tokens just act as a prefix. And then what the model is going to do now is just going to get the probability for the next token. And it's just a glorified autocomplete.

It's a very, very expensive autocomplete of what comes next. And depending on the statistics of what it saw in its training documents, which are basically web pages. So let's just hit Enter to see what tokens it comes up with as a continuation.

OK, so here it kind of actually answered the question and started to go off into some philosophical territory. Let's try it again. So let me copy and paste.

And let's try again from scratch. What is two plus two? OK, so it just goes off again. So notice one more thing that I want to stress is that the system, I think every time you put it in, it just kind of starts from scratch.

So it doesn't, the system here is stochastic. So for the same prefix of tokens, we're always getting a different answer. And the reason for that is that we get this probability distribution and we sample from it.

And we always get different samples. And we sort of always go into a different territory afterwards. So here in this case, I don't know what this is.

Let's try one more time. So it just continues on. So it's just doing the stuff that it's on the internet, right? And it's just kind of like regurgitating those statistical patterns.

So first things, it's not an assistant yet. It's a token autocomplete. And second, it is a stochastic system.

Now, the crucial thing is that even though this model is not yet by itself very useful for a lot of applications just yet, it is still very useful because in the task of predicting the next token in the sequence, the model has learned a lot about the world. And it has stored all that knowledge in the parameters of the network. So remember that our text looked like this, right? Internet web pages.

And now all of this is sort of compressed in the weights of the network. So you can think of these 405 billion parameters as a kind of compression of the internet. You can think of the 405 billion parameters as kind of like a zip file, but it's not a lossless compression.

It's a lossy compression. We're kind of like left with kind of a gestalt of the internet and we can generate from it, right? Now we can elicit some of this knowledge by prompting the base model accordingly. So for example, here's a prompt that might work to elicit some of that knowledge that's hiding in the parameters.

Here's my top 10 list of the top landmarks to see in Paris. And I'm doing it this way because I'm trying to prime the model to now continue this list. So let's see if that works when I press enter.

Okay, so you see that it started the list and it's now kind of giving me some of those landmarks. And I noticed that it's trying to give a lot of information here. Now you might not be able to actually fully trust some of the information here.

Remember that this is all just a recollection of some of the internet documents. And so the things that occur very frequently in the internet data are probably more likely to be remembered correctly compared to things that happen very infrequently. So you can't fully trust some of the things that is some of the information that is here because it's all just a vague recollection of internet documents.

Because the information is not stored explicitly in any of the parameters. It's all just the recollection. That said, we did get something that is probably approximately correct.

And I don't actually have the expertise to verify that this is roughly correct. But you see that we've elicited a lot of the knowledge of the model. And this knowledge is not precise and exact.

This knowledge is vague and probabilistic and statistical. And the kinds of things that occur often are the kinds of things that are more likely to be remembered in the model. Now I want to show you a few more examples of this model's behavior.

The first thing I want to show you is this example. I went to the Wikipedia page for Zebra. And let me just copy paste the first, even one sentence here.

And let me put it here. Now when I click Enter, what kind of completion are we going to get? So let me just hit Enter. There are three living species, et cetera, et cetera.

What the model is producing here is an exact regurgitation of this Wikipedia entry. It is reciting this Wikipedia entry purely from memory. And this memory is stored in its parameters.

And so it is possible that at some point in these 512 tokens, the model will stray away from the Wikipedia entry. You can see that it has huge chunks of it memorized here. Let me see, for example, if this sentence occurs by now.

Okay, so we're still on track. Let me check here. Okay, we're still on track.

It will eventually stray away. Okay, so this thing is just recited to a very large extent. It will eventually deviate because it won't be able to remember exactly.

Now, the reason that this happens is because these models can be extremely good at memorization. And usually this is not what you want in the final model. And this is something called regurgitation.

And it's usually undesirable to cite things directly that you have trained on. Now, the reason that this happens actually is because for a lot of documents, like for example, Wikipedia, when these documents are deemed to be of very high quality as a source, like for example, Wikipedia, it is very often the case that when you train the model, you will preferentially sample from those sources. So basically the model has probably done a few epochs on this data, meaning that it has seen this web page like maybe probably 10 times or so.

And it's a bit like you, like when you read some kind of a text many, many times, say you read something a hundred times, then you will be able to recite it. And it's very similar for this model. If it sees something way too often, it's going to be able to recite it later from memory.

Except these models can be a lot more efficient like per presentation than a human. So probably it's only seen this Wikipedia entry 10 times, but basically it has remembered this article exactly in its parameters. Okay, the next thing I want to show you is something that the model has definitely not seen during its training.

So for example, if we go to the paper and then we navigate to the pre-training data, we'll see here that the dataset has a knowledge cutoff until the end of 2023. So it will not have seen documents after this point. And certainly it has not seen anything about the 2024 election and how it turned out.

Now, if we prime the model with the tokens from the future, it will continue the token sequence and it will just take its best guess according to the knowledge that it has in its own parameters. So let's take a look at what that could look like. So the Republican Party kid, Trump.

Okay, President of the United States from 2017. And let's see what it says after this point. So for example, the model will have to guess the running mate and who it's against, et cetera.

So let's hit enter. So here are things that Mike Pence was the running mate instead of JD Vance. And the ticket was against Hillary Clinton and Tim Kaine.

So this is kind of an interesting parallel universe potentially of what could have happened according to the alarm. Let's get a different sample. So the identical prompt and let's resample.

So here the running mate was Ron DeSantis and they ran against Joe Biden and Kamala Harris. So this is again a different parallel universe. So the model will take educated guesses and it will continue the token sequence based on this knowledge.

And we'll just kind of like all of what we're seeing here is what's called hallucination. The model is just taking its best guess in a probabilistic manner. The next thing I would like to show you is that even though this is a base model and not yet an assistant model, it can still be utilized in practical applications if you are clever with your prompt design.

So here's something that we would call a few-shot prompt. So what it is here is that I have 10 words or 10 pairs and each pair is a word of English colon and then the translation in Korean. And we have 10 of them.

And what the model does here is at the end we have teacher colon and then here's where we're gonna do a completion of say just five tokens. And these models have what we call in-context learning abilities. And what that's referring to is that as it is reading this context, it is learning sort of in place that there's some kind of an algorithmic pattern going on in my data and it knows to continue that pattern.

And this is called kind of like in-context learning. So it takes on the role of a translator and when we hit completion, we see that the teacher translation is sunsaenim, which is correct. And so this is how you can build apps by being clever with your prompting even though we still just have a base model for now.

And it relies on what we call this in-context learning ability and it is done by constructing what's called a few-shot prompt. Okay, and finally, I want to show you that there is a clever way to actually instantiate a whole language model assistant just by prompting. And the trick to it is that we're gonna structure a prompt to look like a webpage that is a conversation between a helpful AI assistant and a human.

And then the model will continue that conversation. So actually to write the prompt, I turned to ChatGPT itself, which is kind of meta, but I told it, I want to create an OLM assistant, but all I have is the base model. So can you please write my prompt? And this is what it came up with, which is actually quite good.

So here's a conversation between an AI assistant and a human. The AI assistant is knowledgeable, helpful, capable of answering a wide variety of questions, et cetera. And then here, it's not enough to just give it a sort of description.

It works much better if you create this few-shot prompt. So here's a few terms of human assistant, human assistant. And we have a few turns of conversation.

And then here at the end is we're gonna be putting the actual query that we like. So let me copy paste this into the base model prompt. And now let me do human column.

And this is where we put our actual prompt. Why is the sky blue? And let's run assistant. The sky appears blue due to the phenomenon called ray light scattering, et cetera, et cetera.

So you see that the base model is just continuing the sequence. But because the sequence looks like this conversation, it takes on that role. But it is a little subtle because here it just, you know, it ends the assistant and then just, you know, hallucinates the next question by the human, et cetera.

So it'll just continue going on and on. But you can see that we have sort of accomplished the task. And if you just took this, why is the sky blue? And if we just refresh this and put it here, then of course we don't expect this to work with the base model, right? We're just gonna, who knows what we're gonna get.

Okay, we're just gonna get more questions. Okay, so this is one way to create an assistant even though you may only have a base model. Okay, so this is the kind of brief summary of the things we talked about over the last few minutes.

Now, let me zoom out here. And this is kind of like what we've talked about so far. We wish to train LLM assistants like ChatGPT.

We've discussed the first stage of that, which is the pre-training stage. And we saw that really what it comes down to is we take internet documents, we break them up into these tokens, these atoms of little text chunks, and then we predict token sequences using neural networks. The output of this entire stage is this base model.

It is the setting of the parameters.

(该文件长度超过30分钟。 在TurboScribe.ai点击升级到无限，以转录长达10小时的文件。)


(转录由TurboScribe.ai完成。升级到无限以移除此消息。)

And this base model is basically an internet document simulator on the token level. So it can just, it can generate token sequences that have the same kind of like statistics as internet documents. And we saw that we can use it in some applications, but we actually need to do better. 

We want an assistant, we want to be able to ask questions, and we want the model to give us answers. And so we need to now go into the second stage, which is called the post training stage. So we take our base model, our internet document simulator, and hand it off to post training. 

So we're now going to discuss a few ways to do what's called post training of these models. These stages in post training are going to be computationally much less expensive. Most of the computational work, all of the massive data centers, and all of the sort of heavy compute and millions of dollars are the pre-training stage. 

But now we go into the slightly cheaper, but still extremely important stage called post training, where we turn this LLM model into an assistant. So let's take a look at how we can get our model to not sample internet documents, but to give answers to questions. So in other words, what we want to do is we want to start thinking about conversations.

And these are conversations that can be multi-term. So there can be multiple terms, and they are in the simplest case, a conversation between a human and an assistant. And so for example, we can imagine the conversation could look something like this. 

When a human says what is 2 plus 2, the assistant should respond with something like 2 plus 2 is 4. When a human follows up and says what if it was star instead of a plus, assistant could respond with something like this. And similar here, this is another example showing that the assistant could also have some kind of a personality here, that it's kind of like nice. And then here in the third example, I'm showing that when a human is asking for something that we don't wish to help with, we can produce what's called a refusal. 

We can say that we cannot help with that. So in other words, what we want to do now is we want to think through how an assistant should interact with a human. And we want to program the assistant and its behavior in these conversations. 

Now, because this is neural networks, we're not going to be programming these explicitly in code. We're not going to be able to program the assistant in that way. Because this is neural networks, everything is done through neural network training on datasets. 

And so because of that, we are going to be implicitly programming the assistant by creating datasets of conversations. So these are three independent examples of conversations in a dataset. An actual dataset, and I'm going to show you examples, will be much larger. 

It could have hundreds of thousands of conversations that are multi-turn, very long, et cetera, and would cover a diverse breadth of topics. But here I'm only showing three examples. But the way this works basically is assistant is being programmed by example. 

And where is this data coming from? Like 2 times 2 equals 4, same as 2 plus 2, et cetera. Where does that come from? This comes from human labelers. So we will basically give human labelers some conversational context, and we will ask them to basically give the ideal assistant response in this situation. 

And a human will write out the ideal response for an assistant in any situation. And then we're going to get the model to basically train on this and to imitate those kinds of responses. So the way this works then is we are going to take our base model, which we produced in the pre-training stage, and this base model was trained on internet documents. 

We're now going to take that dataset of internet documents, and we're going to throw it out, and we're going to substitute a new dataset. And that's going to be a dataset of conversations. And we're going to continue training the model on these conversations, on this new dataset of conversations. 

And what happens is that the model will very rapidly adjust, and will sort of like learn the statistics of how this assistant responds to human queries. And then later during inference, we'll be able to basically prime the assistant and get the response, and it will be imitating what the human labelers would do in that situation, if that makes sense. So we're going to see examples of that, and this is going to become a bit more concrete. 

I also wanted to mention that this post-training stage, we're going to basically just continue training the model, but the pre-training stage can in practice take roughly three months of training on many thousands of computers. The post-training stage will typically be much shorter, like three hours for example, and that's because the dataset of conversations that we're going to create here manually is much, much smaller than the dataset of text on the internet. And so this training will be very short, but fundamentally we're just going to take our base model, we're going to continue training using the exact same algorithm, the exact same everything, except we're swapping out the dataset for conversations.

So the questions now are, where are these conversations, how do we represent them, how do we get the model to see conversations instead of just raw text, and then what are the outcomes of this kind of training, and what do you get in a certain psychological sense when we talk about the model. So let's turn to those questions now. So let's start by talking about the tokenization of conversations. 

Everything in these models has to be turned into tokens, because everything is just about token sequences. So how do we turn conversations into token sequences, is the question. And so for that we need to design some kind of encoding, and this is kind of similar to maybe if you're familiar, you don't have to be with for example the TCP IP packet on the internet. 

There are precise rules and protocols for how you represent information, how everything is structured together so that you have all this kind of data laid out in a way that is written out on a paper and that everyone can agree on. And so it's the same thing now happening in LLMs. We need some kind of data structures, and we need to have some rules around how these data structures, like conversations, get encoded and decoded to and from tokens. 

And so I want to show you now how I would recreate this conversation in the token space. So if you go to TickTokenizer, I can take that conversation, and this is how it is represented for the language model. So here we are iterating a user and an assistant in this two-turn conversation, and what you're seeing here is it looks ugly, but it's actually relatively simple.

The way it gets turned into a token sequence here at the end is a little bit complicated, but at the end, this conversation between the user and assistant ends up being 49 tokens. It is a one-dimensional sequence of 49 tokens, and these are the tokens, okay? And all the different LLMs will have a slightly different format or protocols, and it's a little bit of a wild west right now, but for example, GPT-4.0 does it in the following way. You have this special token called IM underscore start, and this is short for imaginary monologue of the start. 

Then you have to specify... I don't actually know why it's called that, to be honest. Then you have to specify whose turn it is. So for example, user, which is a token 1428. 

Then you have internal monologue separator, and then it's the exact question, so the tokens of the question, and then you have to close it. So IM end, the end of the imaginary monologue. So basically, the question from a user of what is two plus two ends up being the token sequence of these tokens. 

And now the important thing to mention here is that IM start, this is not text, right? IM start is a special token that gets added. It's a new token, and this token has never been trained on so far. It is a new token that we create in a post-training stage, and we introduce. 

And so these special tokens, like IM set, IM start, et cetera, are introduced and interspersed with text so that they sort of get the model to learn that, hey, this is the start of a turn for... Who is it the start of the turn for? The start of the turn is for the user. And then this is what the user says, and then the user ends. And then it's a new start of a turn, and it is by the assistant. 

And then what does the assistant say? Well, these are the tokens of what the assistant says, et cetera. And so this conversation is now turned into this sequence of tokens. The specific details here are not actually that important. 

All I'm trying to show you in concrete terms is that our conversations, which we think of as kind of like a structured object, end up being turned via some encoding into one-dimensional sequences of tokens. And so because this is a one-dimensional sequence of tokens, we can apply all this stuff that we applied before. Now it's just a sequence of tokens, and now we can train a language model on it. 

And so we're just predicting the next token in a sequence, just like before, and we can represent and train on conversations. And then what does it look like at test time during inference? So say we've trained a model, and we've trained a model on these kinds of datasets of conversations, and now we want to inference. So during inference, what does this look like when you're on ChatsGPT? Well, you come to ChatsGPT, and you have, say, like a dialogue with it. 

And the way this works is basically, say that this was already filled in. So like, what is 2 plus 2? 2 plus 2 is 4. And now you issue what, if it was times, imend. And what basically ends up happening on the servers of OpenAI or something like that is they put an imstart, assistant, imsep, and this is they end it right here. 

So they construct this context, and now they start sampling from the model. So it's at this stage that they will go to the model and say, okay, what is a good first sequence? What is a good first token? What is a good second token? What is a good third token? And this is where the LLM takes over and creates a response, like, for example, a response that looks something like this. But it doesn't have to be identical to this, but it will have the flavor of this if this kind of a conversation was in the dataset. 

So that's roughly how the protocol works, although the details of this protocol are not important. So again, my goal is just to show you that everything ends up being just a one-dimensional token sequence. So we can apply everything we've already seen, but we're now training on conversations, and we're now basically generating conversations as well. 

Okay, so now I would like to turn to what these datasets look like in practice. The first paper that I would like to show you, and the first effort in this direction, is this paper from OpenAI in 2022. And this paper was called InstructGPT, or the technique that they developed. 

And this was the first time that OpenAI has kind of talked about how you can take language models and fine-tune them on conversations. And so this paper has a number of details that I would like to take you through. So the first stop I would like to make is in section 3.4, where they talk about the human contractors that they hired, in this case from Upwork or through ScaleAI, to construct these conversations. 

And so there are human labelers involved whose job it is professionally to create these conversations. And these labelers are asked to come up with prompts, and then they are asked to also complete the ideal assistant responses. And so these are the kinds of prompts that people came up with. 

So these are human labelers. So list five ideas for how to regain enthusiasm for my career. What are the top 10 science fiction books I should read next? And there's many different types of kind of prompts here. 

So translate the sentence to Spanish, etc. And so there's many things here that people came up with. They first come up with the prompt, and then they also answer that prompt, and they give the ideal assistant response. 

Now how do they know what is the ideal assistant response that they should write for these prompts? So when we scroll down a little bit further, we see that here we have this excerpt of labeling instructions that are given to the human labelers. So the company that is developing the language model, like for example OpenAI, writes up labeling instructions for how the humans should create ideal responses. And so here, for example, is an excerpt of these kinds of labeling instructions. 

On a high level, you're asking people to be helpful, truthful, and harmless. And you can pause the video if you'd like to see more here. But on a high level, basically just answer, try to be helpful, try to be truthful, and don't answer questions that we don't want kind of the system to handle later in chat GPT. 

And so roughly speaking, the company comes up with the labeling instructions. Usually they are not this short. Usually they are hundreds of pages, and people have to study them professionally. 

And then they write out the ideal assistant responses following those labeling instructions. So this is a very human heavy process, as it was described in this paper. Now the data set for InstructGPT was never actually released by OpenAI. 

But we do have some open source reproductions that were trying to follow this kind of a setup and collect their own data. So one that I'm familiar with, for example, is the effort of Open Assistant from a while back. And this is just one of, I think, many examples.

But I just want to show you an example. So these were people on the internet that were asked to basically create these conversations, similar to what OpenAI did with human labelers. And so here's an entry of a person who came up with this prompt. 

Can you write a short introduction to the relevance of the term monopsony in economics? Please use examples, et cetera. And then the same person, or potentially a different person, will write up the response. So here's the assistant response to this. 

And so then the same person or different person will actually write out this ideal response. And then this is an example of maybe how the conversation could continue. Now explain it to a dog. 

And then you can try to come up with a slightly simpler explanation or something like that. Now this then becomes the label, and we end up training on this. So what happens during training is that, of course, we're not going to have full coverage of all the possible questions that the model will encounter at test time during inference. 

We can't possibly cover all the possible prompts that people are going to be asking in the future. But if we have a data set of a few of these examples, then the model during training will start to take on this persona of this helpful, truthful, harmless assistant. And it's all programmed by example. 

And so these are all examples of behavior. And if you have conversations of these example behaviors, and you have enough of them, like 100,000, and you train on it, the model sort of starts to understand the statistical pattern, and it kind of takes on this personality of this assistant. Now it's possible that when you get the exact same question like this at test time, it's possible that the answer will be recited as exactly what was in the training set. 

But more likely than that is that the model will kind of do something of a similar vibe, and it will understand that this is the kind of answer that you want. So that's what we're doing. We're programming the system by example, and the system adopts statistically this persona of this helpful, truthful, harmless assistant, which is kind of reflected in the labeling instructions that the company creates.

Now I want to show you that the state of the art has kind of advanced in the last two or three years since the InstructGPT paper. So in particular, it's not very common for humans to be doing all the heavy lifting just by themselves anymore. And that's because we now have language models, and these language models are helping us create these datasets and conversations.

So it is very rare that the people will literally just write out the response from scratch. It is a lot more likely that they will use an existing LLM to basically come up with an answer, and then they will edit it, or things like that. So there's many different ways in which now LLMs have started to kind of permeate this post-training stack. 

And LLMs are basically used pervasively to help create these massive datasets of conversations. So I don't want to show, like UltraChat is one such example of like a more modern dataset of conversations. It is to a very large extent synthetic, but I believe there's some human involvement. 

I could be wrong with that. Usually there will be a little bit of human, but there will be a huge amount of synthetic help. And this is all kind of like constructed in different ways. 

And UltraChat is just one example of many SFT datasets that currently exist. And the only thing I want to show you is that these datasets have now millions of conversations. These conversations are mostly synthetic, but they're probably edited to some extent by humans. 

And they span a huge diversity of sort of areas and so on. So these are fairly extensive artifacts by now. And there are all these like SFT mixtures, as they're called. 

So you have a mixture of like lots of different types and sources, and it's partially synthetic, partially human. And it's kind of like gone in that direction since. But roughly speaking, we still have SFT datasets. 

They're made up of conversations. We're training on them, just like we did before. And I guess like the last thing to note is that I want to dispel a little bit of the magic of talking to an AI. 

Like when you go to ChatGPT and you give it a question, and then you hit enter, what is coming back is kind of like statistically aligned with what's happening in the training set. And these training sets, I mean, they really just have a seed in humans following labeling instructions. So what are you actually talking to in ChatGPT? Or how should you think about it? Well, it's not coming from some magical AI, like roughly speaking. 

It's coming from something that is statistically imitating human labelers, which comes from labeling instructions written by these companies. And so you're kind of imitating this. You're kind of getting, it's almost as if you're asking a human labeler. 

And imagine that the answer that is given to you from ChatGPT is some kind of a simulation of a human labeler. And it's kind of like asking, what would a human labeler say in this kind of a conversation? And it's not just like, this human labeler is not just like a random person from the internet, because these companies actually hire experts. So for example, when you are asking questions about code and so on, the human labelers that would be involved in creation of these conversation datasets, they will usually be educated, expert people. 

And you're kind of asking a question of like a simulation of those people, if that makes sense. So you're not talking to a magical AI, you're talking to an average labeler. This average labeler is probably fairly highly skilled, but you're talking to kind of like an instantaneous simulation of that kind of a person that would be hired in the construction of these datasets.

So let me give you one more specific example before we move on. For example, when I go to ChatGPT and I say, recommend the top five landmarks you see in Paris, and then I hit enter. Okay, here we go. 

Okay, when I hit enter, what's coming out here, how do I think about it? Well, it's not some kind of a magical AI that has gone out and researched all the landmarks and then ranked them using its infinite intelligence, et cetera. What I'm getting is a statistical simulation of a labeler that was hired by OpenAI. You can think about it roughly in that way.

And so if this specific question is in the post-training dataset somewhere at OpenAI, then I'm very likely to see an answer that is probably very, very similar to what that human labeler would have put down for those five landmarks. How does the human labeler come up with this? Well, they go off and they go on the internet and they kind of do their own little research for 20 minutes and they just come up with a list, right? So if they come up with this list and this is in the dataset, I'm probably very likely to see what they submitted as the correct answer from the assistant. Now, if this specific query is not part of the post-training dataset, then what I'm getting here is a little bit more emergent because the model kind of understands that statistically the kinds of landmarks that are in this training set are usually the prominent landmarks, the landmarks that people usually want to see, the kinds of landmarks that are usually very often talked about on the internet. 

And remember that the model already has a ton of knowledge from its pre-training on the internet. So it's probably seen a ton of conversations about pairs, about landmarks, about the kinds of things that people like to see. And so it's the pre-training knowledge that has been combined with the post-training dataset that results in this kind of an imitation. 

So that's roughly how you can kind of think about what's happening behind the scenes here in the statistical sense. Okay, now I want to turn to the topic of LLM psychology, as I like to call it, which is where sort of the emergent cognitive effects of the training pipeline that we have for these models. So in particular, the first one I want to talk to is, of course, hallucinations. 

So you might be familiar with model hallucinations. It's when LLMs make stuff up. They just totally fabricate information, et cetera. 

And it's a big problem with LLM assistants. It is a problem that existed to a large extent with early models for many years ago. And I think the problem has gotten a bit better because there are some mitigations that I'm going to go into in a second. 

For now, let's just try to understand where these hallucinations come from. So here's a specific example of three conversations that you might think you have in your training set. And these are pretty reasonable conversations that you could imagine being in a training set. 

So like, for example, who is Tom Cruise? Well, Tom Cruise is a famous actor, American actor and producer, et cetera. Who is John Barrasso? This turns out to be a US senator, for example. Who is Genghis Khan? Well, Genghis Khan was blah, blah, blah. 

And so this is what your conversations could look like at training time. Now, the problem with this is that when the human is writing the correct answer for the assistant, in each one of these cases, the human either, like, knows who this person is or they research them on the internet, and they come in and they write this response that kind of has this, like, confident tone of an answer. And what happens basically is that at test time, when you ask for someone who is, this is a random name that I totally came up with, and I don't think this person exists, as far as I know.

I just tried to generate it randomly. The problem is when we ask who is Orson Kovats, the problem is that the assistant will not just tell you, oh, I don't know. Even if the assistant and the language model itself might know inside its features, inside its activations, inside of its brain, sort of, it might know that this person is, like, not someone that it's familiar with.

Even if some part of the network kind of knows that in some sense, the saying that, oh, I don't know who this is, is not going to happen because the model statistically imitates its training set. In the training set, the questions of the form who is blah are confidently answered with the correct answer. And so it's going to take on the style of the answer and it's going to do its best.

It's going to give you statistically the most likely guess. And it's just going to basically make stuff up. Because these models, again, we just talked about it, is they don't have access to the internet. 

They're not doing research. These are statistical token tumblers, as I call them. It's just trying to sample the next token in the sequence.

And it's going to basically make stuff up. So let's take a look at what this looks like. I have here what's called an inference playground from Hugging Face.

And I am on purpose picking on a model called Falcon 7B, which is an old model. This is a few years ago now. So it's an older model. 

So it suffers from hallucinations. And as I mentioned, this has improved over time recently. But let's say who is Orson Kovats? Let's ask Falcon 7B instructor. 

Run. Oh, yeah. Orson Kovats is an American author and science fiction writer.

Okay. This is totally false. It's a hallucination. 

Let's try again. These are statistical systems, right? So we can resample. This time Orson Kovats is a fictional character from this 1950s TV show.

It's total BS, right? Let's try again. He's a former minor league baseball player. Okay. 

So basically the model doesn't know. And it's given us lots of different answers because it doesn't know. It's just kind of like sampling from these probabilities.

The model starts with the tokens who is Orson Kovats assistant. And then it comes in here. And it's getting these probabilities. 

And it's just sampling from the probabilities. And it just comes up with stuff. And the stuff is actually statistically consistent with the style of the answer in its training set. 

And it's just doing that. But you and I experience it as a made up factual knowledge. But keep in mind that the model basically doesn't know. 

And it's just imitating the format of the answer. And it's not going to go off and look it up. Because it's just imitating, again, the answer. 

So how can we mitigate this? Because, for example, when we go to chat GPT and I say, who is Orson Kovats? And I'm now asking the state of the art model from OpenAI. This model will tell you. Oh. 

So this model is actually is even smarter. Because you saw very briefly, it said searching the web. We're going to cover this later. 

It's actually trying to do tool use. And kind of just like came up with some kind of a story. But I want to just, who is Orson Kovats did not use any tools. 

I don't want it to do web search. There's a well known historical public figure named Orson Kovats. So this model is not going to make up stuff. 

This model knows that it doesn't know. And it tells you that it doesn't appear to be a person that this model knows. So somehow we sort of improved hallucinations. 

Even though they clearly are an issue in older models. And it makes totally sense why you would be getting these kinds of answers if this is what your training set looks like. So how do we fix this? Okay. 

Well, clearly we need some examples in our data set that where the correct answer for the assistant is that the model doesn't know about some particular fact. But we only need to have those answers be produced in the cases where the model actually doesn't know. And so the question is, how do we know what the model knows or doesn't know? Well, we can empirically probe the model to figure that out.

So let's take a look at, for example, how Meta dealt with hallucinations for the Lama 3 series of models as an example. So in this paper that they published from Meta, we can go into hallucinations, which they call here factuality. And they describe the procedure by which they basically interrogate the model to figure out what it knows and doesn't know, to figure out sort of like the boundary of its knowledge. 

And then they add examples to the training set where for the things where the model doesn't know them, the correct answer is that the model doesn't know them, which sounds like a very easy thing to do in principle. But this roughly fixes the issue. And the reason it fixes the issue is because remember that the model might actually have a pretty good model of its self-knowledge inside the network. 

So remember, we looked at the network and all these neurons inside the network. You might imagine there's a neuron somewhere in the network that sort of like lights up for when the model is uncertain. But the problem is that the activation of that neuron is not currently wired up to the model actually saying in words that it doesn't know. 

So even though the internals of the neural network know, because there's some neurons that represent that, the model will not surface that. It will instead take its best guess so that it sounds confident, just like it sees in the training set. So we need to basically interrogate the model and allow it to say, I don't know, in the cases that it doesn't know. 

So let me take you through what MetaRoughly does. So basically what they do is, here I have an example. Dominik Hasek is the featured article today, so I just went there randomly. 

And what they do is basically they take a random document in a training set and they take a paragraph and then they use an LLM to construct questions about that paragraph. So for example, I did that with chat-gpt here. So I said, here's a paragraph from this document. 

Generate three specific factual questions based on this paragraph and give me the questions and the answers. And so the LLMs are already good enough to create and reframe this information. So if the information is in the context window of this LLM, this actually works pretty well. 

It doesn't have to rely on its memory, it's right there in the context window. And so it can basically reframe that information with fairly high accuracy. So for example, it can generate questions for us like, for which team did he play? Here's the answer. 

How many cups did he win? Et cetera. And now what we have to do is we have some question and answers, and now we want to interrogate the model. So roughly speaking, what we'll do is we'll take our questions and we'll go to our model, which would be, say, Llama in meta. 

But let's just interrogate Mistral7b here as an example. That's another model. So does this model know about this answer? Let's take a look.

So he played for Buffalo Sabres, right? So the model knows. And the way that you can programmatically decide is basically we're going to take this answer from the model and we're going to compare it to the correct answer. And again, the models are good enough to do this automatically. 

So there's no humans involved here. We can take basically the answer from the model and we can use another LLM judge to check if that is correct according to this answer. And if it is correct, that means that the model probably knows. 

So what we're going to do is we're going to do this maybe a few times. So, okay, it knows it's Buffalo Sabres. Let's try again.

Buffalo Sabres. Let's try one more time. Buffalo Sabres. 

So we asked three times about this factual question and the model seems to know. So everything is great.

(该文件长度超过30分钟。 在TurboScribe.ai点击升级到无限，以转录长达10小时的文件。)


