
大家好，其实我一直想做这个视频。这是一个面向大众、全面介绍 ChatGPT 等大语言模型的视频。我希望通过这个视频，能帮助大家建立起理解这类工具运作方式的思维模型。

它在某些方面显然充满魔力且令人惊叹。有些事它做得非常出色，另一些则不尽如人意，同时还有许多需要注意的尖锐问题。那么这个文本框背后究竟是什么？你可以输入任何内容并按下回车，但我们应该输入什么？这些生成的文字又是从何而来？它的运作原理是什么？你实际上是在和什么对话？我希望通过这个视频探讨所有这些话题。

我们将完整梳理这类系统是如何构建的整个流程，但我会尽量让讲解通俗易懂，适合大众理解。

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

记住，网络给出的是概率，对吧？它在这里给出的是这个概率向量。所以我们现在可以做的，本质上就是抛一个有偏见的硬币。也就是说，我们可以基于这个概率分布来采样一个标记。例如，接下来是 token 860。因此，在这种情况下，当我们从模型生成时，860 可能会紧随其后。现在，860 是一个相对可能的 token。在这种情况下，它可能不是唯一可能的标记。可能还有许多其他标记可以被采样。但我们可以看到，860 作为一个例子是一个相对可能的标记。

事实上，在我们这个训练示例中，860 确实紧跟在 91 之后。那么现在让我们继续这个过程。也就是说，91 之后就是 860。我们附加它。我们再次询问，第三个 token 是什么？让我们取样。假设它是 287，就像这里一样。让我们再来一次。我们重新开始。现在我们有三个连续的 token。我们问，第四个可能的 token 是什么？然后我们从中抽样得到这个。现在假设我们再重复一次。我们取这四个。我们取样。然后我们得到了这个。而这个 13659，实际上并不是我们之前得到的 3962。所以这个 token 实际上是 token "Article"。因此，在这种情况下，我们并没有完全重现训练数据中看到的序列。

所以请记住，这些系统是随机的。我们在采样，我们在掷硬币。有时我们运气不错，能重现训练集中的一小段文本。但有时我们得到的标记并非训练数据中任何文档的逐字内容。所以我们将会得到类似于训练数据中的某种混音版本。因为在每个步骤中，我们都可以进行掷硬币，得到稍微不同的标记。一旦这个标记被采用，如果你继续采样下一个标记，以此类推，你很快就会开始生成与训练文档中出现的标记流完全不同的标记流。

从统计学上看，它们会具有相似的特性，但并不与训练数据完全相同。它们更像是受到训练数据的启发。因此在这个案例中，我们得到了一个略有不同的序列。

我们为什么会得到 “article” 这个词呢？你可能会认为，在 “|”、“viewing”、“single” 等词的上下文中，“article” 是一个相对可能出现的词。某种程度上，你可以想象在训练文档的某个地方，“article” 这个词跟随了这个上下文窗口。而我们恰好在这个阶段采样到了它。

简单来说，推理就是依次从这些概率分布中进行预测，我们不断反馈标记并获取下一个标记。整个过程就像不断掷硬币。根据我们运气的好坏，从这些概率分布中采样时，可能会得到截然不同的模式。这就是推理。

在大多数常见情况下，下载互联网数据并进行分词处理实际上是一个预处理步骤。然后，一旦你有了 token 序列，我们就可以开始训练网络了。在实际应用中，你会尝试训练许多不同的网络，它们有不同的设置、不同的排列方式和不同的大小。因此，你会进行大量的神经网络训练。

当你有了一个神经网络并训练它，并且得到一组令你满意的特定参数后，你就可以使用这个模型进行推理。实际上，你可以从模型中生成数据。当你在 ChatGPT 上与模型对话时，那个模型已经训练完成，可能是 OpenAI 在几个月前训练的。它们有一套特定的权重效果很好。当你与模型对话时，这一切都只是推理。不再有训练过程。这些参数是固定的。你基本上只是在和模型对话。你给它一些标记，它就会完成标记序列。这就是你在 ChatGPT 上实际使用该模型时所看到的内容生成过程。因此，该模型仅执行推理任务。

### GPT-2：训练和推理

那么现在让我们来看一个具体的训练和推理示例，这样你就能了解这些模型在训练时的实际运作情况。

现在我想讨论的例子，也是我特别喜欢的，就是 OpenAI 的 GPT2。GPT 代表 Generatively Pre-trained Transformer。这是 OpenAI 推出的 GPT 系列的第二代版本。今天当你与 ChatGPT 对话时，支撑这一神奇交互的底层模型正是 GPT4，也就是该系列的第四代版本。而 GPT2 则是 OpenAI 在 2019 年发表的成果，就是我现在手里拿的这篇论文。我之所以钟爱 GPT2，是因为它首次完整呈现了可辨识的现代技术架构。

按照现代标准来看，GPT2 的所有组件至今仍具有辨识度。只不过一切都变得更庞大了。当然，由于这是篇技术论文，我无法在此详述其全部细节。但我想重点强调的一些细节如下。GPT2 是一个 Transformer 神经网络，就像你今天会使用的神经网络一样。它有 1.6B 个参数，对吧？这些就是我们在这里看到的参数。如今，现代 Transformer 模型的参数量可能已接近万亿或数千亿。这里的最大上下文长度是 1,024 个标记。因此，当我们从数据集中采样 token 窗口的片段时，我们永远不会取超过 1,024 个 token。因此，当你试图预测序列中的下一个 token 时，你的上下文中永远不会有超过 1,024 个 token 来进行预测。这在现代标准下也是微不足道的。

如今，上下文长度已经大幅提升至数十万甚至可能达到百万级别。这样一来，历史记录中可容纳的上下文信息更多，标记数量也大幅增加。通过这种方式，你能够更准确地预测序列中的下一个标记。最后，GPT2 的训练数据大约是 100B 个标记。按现代标准来看，这个规模也相当小。正如我提到的，我们在这里研究的 fineweb 数据集有 1.5T 个标记，所以 100B 其实很少。实际上，我为了好玩，在这个名为llm.c 的项目中尝试复现 GPT2。你可以在 GitHub 上的 llm.c 仓库里看到我写的相关文章。具体来说，2019 年训练 GPT2 的成本估计约为 4 万美元。

但如今，你能做得比这好得多。具体来说，这次只花了一天时间和大约 600 美元。而且这还没怎么费劲。我觉得今天你可以把价格降到 100 美元左右。为什么成本下降这么多呢？首先，这些数据集的质量大幅提升。其次，我们筛选、提取和准备数据的方式也变得更加精细。因此，数据集的质量要高得多。这是一方面。但最大的不同在于，我们的计算机硬件速度大幅提升。

我们稍后会详细讨论这一点。此外，用于运行这些模型并尽可能从硬件中榨取所有速度的软件，随着大家都专注于这些模型并试图以极快的速度运行它们，这些软件也有了很大的改进。现在，我无法详细介绍这个 GPT2 的复现过程。这是一篇技术性很强的长文。但我想让你直观地感受一下，作为一名研究人员实际训练这些模型是什么样子。比如，你会看到什么？看起来是怎样的？感觉如何？让我来为你简单描绘一下。

好的，这就是它的样子。让我把这个滑过去。我现在正在做的是训练一个 GPT2 模型。这里发生的情况是，这里的每一行，比如这一行，都是对模型的一次更新。所以请记住，我们基本上是在为每一个标记改进预测。同时我们也在更新这些神经网络的权重或参数。

因此，这里的每一行都是对神经网络的一次更新，我们通过微调其参数，使其能更准确地预测下一个标记和序列。具体来说，这里的每一行都在提升对训练集中 1M 个标记的预测能力。也就是说，我们实际上是从这个数据集中提取了 1M 个标记来进行优化。我们试图同时改进对所有 1M 个标记的下一个标记的预测。在每一个步骤中，我们都会对网络进行相应的更新。现在，需要密切关注的一个数字就是这个叫做损失的值。

损失值是一个单一的数字，它告诉你神经网络当前的表现如何。这个数值的设计初衷是越低越好。因此，你会看到随着我们对神经网络进行更多更新，损失值在不断下降，这意味着对序列中下一个标记的预测会越来越准确。

因此，损失值就是作为神经网络研究者的你所关注的数字。你只能耐心等待，百无聊赖地消磨时间。你正在喝咖啡。同时你也在确保一切看起来不错，这样每次更新时，你的损失都在减少，网络的预测能力也在不断提高。现在，你可以看到我们每次更新处理 1M 个标记。每次更新大约需要 7 秒钟。这里我们将总共进行 32,000 步优化处理。所以，32,000 步，每步 1M 个标记，总共大约要处理 33B 个标记。而我们目前才进行到第 420 步，也就是 32,000 步中的 420 步。所以，我们只完成了略多于 1% 的工作量。

因为我只运行了大概 10 到 15 分钟。现在，每 20 步我都会配置这个优化进行推理。所以你现在看到的是模型在预测序列中的下一个标记。于是你有点随机地开始了。然后你继续填入标记。所以我们正在运行这个推理步骤。而这个模型就是在预测序列中的下一个标记。每次你看到有东西出现，那就是一个新的标记。让我们来看看这个。

你可以看出这还不够连贯。请记住，这只是训练进度的 1%。因此，模型在预测序列中的下一个标记方面还不够熟练。所以输出的内容其实有点像是胡言乱语，对吧？但它仍然保留了一些局部的连贯性。既然她属于我，那就是信息的一部分。应该讨论我的父亲、伟大的伙伴们，戈登让我坐在上面等等。

所以我知道这看起来不太好。但让我们向上滚动一下，看看我开始优化时的样子。回到第一步这里。经过 20 步优化后，你会发现我们得到的结果看起来完全是随机的。当然，这是因为模型只更新了 20 次参数。所以它给出的文本是随机的，因为它是一个随机网络。由此可见，至少与此相比，模型的表现正在变得更好。事实上，如果我们等待完整的 32,000 步训练过程，模型的改进程度会达到生成相当连贯英语的水平。生成的词汇流准确无误，整体英语表达也显得更加自然流畅。所以现在还需要再运行一两天。在这个阶段，我们只需要确保损失在减少。一切看起来都很顺利。

而我们只能等待。现在，让我来谈谈所需的计算过程。当然，我并不是在我的笔记本电脑上运行这个优化。那会太贵了。因为我们需要运行这个神经网络。而且我们还得改进它。我们需要所有这些数据等等。所以你在自己的电脑上无法很好地运行这个程序。因为网络实在太庞大了。这一切都在云端的计算机上运行。我想主要谈谈训练这些模型的计算方面及其具体表现。让我们来看看。

好的，我现在运行这个优化程序的电脑是一个 8xh100 节点。也就是说，一台节点或者说一台电脑里有八个 h100。目前这台电脑是我租来的。它就在云端的某个地方。实际上，我也不确定它的物理位置在哪里。我喜欢租用的地方叫 Lambda。但提供这项服务的公司还有很多。往下滑动页面，你会看到他们针对配备 H100 这类 GPU 的计算机提供了一些按需定价方案。稍后我会展示这些 GPU 的外观。但按需提供 8xh100 GPU。例如，这台机器每小时每个 GPU 收费 3 美元。因此，你可以租用这些设备。然后你在云端获得一台机器。你可以进入并训练这些模型。这些 GPU 看起来是这样的。所以这是一块H100 GPU。它大概长这样。你可以把它插进电脑里。

而 GPU 非常适合用于训练神经网络，因为它们需要极高的计算量。但这类计算能展现出高度的并行性。因此，你可以让许多独立的工作单元同时运作，共同解决神经网络训练背后涉及的矩阵乘法运算。所以这只是其中一块 H100 芯片。但实际上，你会把多块组合在一起。比如可以把八块堆叠成一个节点。然后，你可以将多个节点堆叠成整个数据中心或整个系统。因此，当我们观察数据中心时，就会开始看到类似这样的结构，对吧？从一个 GPU 扩展到八个 GPU，再到单个系统，再到多个系统。这些就是规模更大的数据中心。

当然，它们的价格会高得多。目前的情况是，所有大型科技公司都非常渴望获得这些 GPU，因为它们功能强大，可以用来训练各种语言模型。这从根本上推动了英伟达股价飙升至如今的 3.4 万亿美元，也是英伟达股价暴涨的原因。所以这就是淘金热。淘金热就是争抢 GPU，获取足够多的 GPU 让它们能够协同工作来完成这种优化。那它们都在做什么呢？它们都在协作预测像 FindWeb 数据集这样的数据集上的下一个标记。

这是极其昂贵的计算流程。GPU 越多，就能尝试预测和改进更多 token，处理数据集的速度也会更快。你可以更快地进行迭代，获得更大的网络，训练更大的网络，以此类推。这就是所有这些机器正在做的事情。这就是为什么这一切如此重要。

例如，这是一篇大约一个月前的文章。这就是为什么像埃隆·马斯克在一个数据中心获得 10 万块 GPU 这样的事如此重要。所有这些 GPU 都非常昂贵，将消耗大量电力。它们都只是在试图预测序列中的下一个标记，并通过这样做来改进网络。而且可能会比我们在这里看到的要快得多地生成更加连贯的文本。好吧，遗憾的是，我没有几千万或几亿美元来训练一个像这样真正庞大的模型。

但幸运的是，我们可以求助于一些大型科技公司，它们会定期训练这些模型，并在训练完成后发布部分模型。因此，它们投入了大量计算资源来训练这个网络，并在优化结束时发布该网络。因此这非常有用，因为他们为此进行了大量计算。所以有很多公司会定期训练这些模型。但实际上，其中发布所谓基础模型的公司并不多。

所以最终呈现的这个模型被称为基础模型。什么是基础模型？它本质上是个标记模拟器，对吧？一个互联网文本标记模拟器。就其本身而言，它目前还不具备实用价值。因为我们想要的是一种所谓的助手。我们希望能提出问题并得到回答。这些模型无法做到这一点。他们只是对互联网进行某种混音创作。他们梦想着网页。因此，基础模型并不经常发布，因为它们只是我们迈向智能助手所需的几个步骤中的第一步。

不过，已经有几个版本发布了。举个例子，GPT-2 模型在 2019 年发布了 1.5B 参数的版本。这个GPT-2 模型是一个基础模型。那么，什么是模型发布？发布这些模型是什么样的？这是 GitHub 上的 GPT-2 仓库。基本上，发布模型需要两样东西。第一，我们通常需要 Python 代码，详细描述模型中执行的操作序列。

所以如果你还记得这个 Transformer，这段代码描述的就是这个神经网络中所采取的步骤序列。这段代码实际上是在实现这个神经网络的所谓前向传播过程。因此我们需要确切了解他们是如何连接这个神经网络的具体细节。所以这只是一段计算机代码，通常也就几百行代码。没什么大不了的。这些代码都相当容易理解，而且通常相当标准。

不标准的是参数。真正的价值就在那里。这个神经网络的参数在哪里？因为有 1.5B 个参数，我们需要正确的设置或非常好的设置。因此，除了源代码之外，他们还发布了参数，在这个例子中大约是 1.5B 个参数。这些参数只是一串数字，也就是一个包含 1.5B 个数字的单一列表。

所有旋钮的精确和良好设置，以确保 token 输出良好。因此，你需要这两样东西来发布一个基础模型。现在，GPT-2 已经发布了，但正如我提到的，这实际上是一个相当老的模型。

### LLaMA-3.1 基础模型推理


实际上，我们要转向的模型叫做 LLaMA3。接下来我想向大家展示的就是它。GPT-2 的参数规模是 1.6B，训练数据量是 10B tokens。LLaMA3 是一个规模更大、更现代化的模型。它由 Meta 发布并训练，是一个拥有 405B 参数、基于 15T 标记训练的模型。同样地，只是规模要大得多。Meta 还发布了 LLAMA3，这也是这篇论文的一部分。

这篇论文详细介绍了他们发布的最大基础模型—— LLaMA3.1 405B 参数模型。这是基础模型。除此之外，正如视频后面部分会提到的，他们还发布了指令模型。指令意味着这是一个助手。你可以向它提问，它会给你答案。我们稍后还会讲到那部分。

目前，我们不妨先看看这个基础模型——这个标记模拟器，来把玩一番，试着思考：这究竟是什么？它是如何运作的？如果让它在海量数据上运行到极致，训练出一个庞大的神经网络，最终我们能得到什么？我个人最喜欢与基础模型互动的平台是 Hyperbolic 公司，他们主要提供 405B 参数的 LLaMA3.1 基础模型。当你进入网站时（可能需要注册等操作），请务必在模型选项中确认你选用的是 LLaMA3.1-405B 基础版，必须是基础模型。这里有个参数叫 "max tokens"，它决定了我们将生成多少个标记。

所以我们就稍微减少一点，以免浪费计算资源。我们只需要接下来的 128 个标记，其他的就不用管了。这里我就不详细解释了。现在，从根本上说，这里发生的事情与我们推理过程中发生的事情是一样的。所以这只会继续你给它任何前缀的标记序列。因此，我想先向你们展示，这里的这个模型还不是一个助手。

例如，你可以问它“二加二等于几？”，它不会直接告诉你“哦，等于四。还有什么可以帮你的吗？”，它不会这么做。因为“二加二等于几”会被分词处理。然后这些标记就充当了前缀。接下来模型要做的就是获取下一个标记的概率。说白了就是个高级的自动补全。这不过是一个非常、非常昂贵的自动补全功能，用于预测接下来的内容。它基于训练文档（基本上是网页）中看到的统计数据进行预测。所以，我们只需按下回车键，看看它会生成什么样的续写标记。

好的，实际上这里已经回答了问题，并开始进入一些哲学领域。让我们再试一次。我来复制粘贴一下。让我们从头再来一次。二加二等于几？好吧，它又自动重启了。所以我想再强调一点，这个系统每次输入后似乎都会从头开始运行。所以它不会，这里的系统是随机的。对于相同的 token 前缀，我们总是得到不同的答案。原因在于我们得到的是概率分布，并从中进行采样。

而我们总是得到不同的样本。之后我们总是会进入一个不同的领域。所以在这种情况下，我不知道这是什么。让我们再试一次。所以它就这样继续下去。它只是在做互联网上的那些事情，对吧？而且它有点像是在重复那些统计模式。首先，它还不是一个助手。它只是一个标记自动完成工具。其次，它是一个随机系统。

现在，关键之处在于，尽管这个模型本身对许多应用来说还不太实用，但它仍然非常有用，因为在预测序列中下一个标记的任务中，模型已经学到了很多关于世界的知识。所有这些知识都存储在网络的参数中。还记得我们的文本是这样的吗？互联网网页。

而现在，这一切在某种程度上都被压缩进了网络的权重中。所以你可以把这 405B 个参数看作是对互联网的一种压缩。你可以把这 405B 个参数想象成某种压缩文件，但它并不是无损压缩。这是一种有损压缩。我们留下的更像是互联网的完形，我们可以从中生成内容，对吧？现在，我们可以通过适当提示基础模型来引出其中一些隐藏的知识。例如，这里有一个提示，可能有助于引出隐藏在参数中的某些知识。

以下是巴黎必看十大景点的清单。我之所以这样做，是想引导模型继续完成这个列表。让我们看看按下回车键后是否有效。好的，你看到它已经开始列出清单，现在正在给我一些地标信息。我注意到它在这里试图提供大量信息。不过，你可能不能完全相信这里的一些信息。请记住，这一切只是对部分网络资料的回忆。因此，在网络数据中频繁出现的内容可能比那些极少出现的内容更容易被准确记住。所以你不能完全信任这里的一些信息，因为它只是对网络资料的模糊回忆。因为这些信息并未明确存储在任何参数中。一切都只是回忆。不过，我们确实得到了一些可能大致正确的东西。我其实没有专业知识来验证这是否大致正确。但你可以看到我们已经引出了模型的很多知识。而这些知识并不精确和准确。

这种知识是模糊的、概率性的和统计性的。经常发生的事情往往更容易在模型中被记住。现在，我想再展示几个这个模型的行为示例。

首先，我想向你展示这个例子。我去了维基百科的斑马页面。让我把第一句话复制粘贴到这里。让我把它放在这里。现在当我点击回车键时，我们会得到什么样的补全结果呢？让我按一下回车键。有三个现存物种，等等，等等。该模型在这里生成的内容是对维基百科条目的精确复述。它纯粹依靠记忆来背诵这段维基百科内容。而这些记忆都储存在它的参数中。因此，在这 512 个标记的某个时刻，模型可能会偏离维基百科的条目。你可以看到它在这里记住了大量的内容。让我看看，比如说，现在是否出现了这句话。好的，我们还在正轨上。让我确认一下。好的，我们还在正轨上。它最终会偏离。好吧，这东西在很大程度上只是被背诵。它最终会偏离，因为它无法准确记住。

现在，这种情况发生的原因是这些模型可能非常擅长记忆。通常，这并不是你在最终模型中想要的。这种现象被称为"反刍（regurgitation）"。而且通常不建议直接引用你训练过的内容。实际上，这种情况发生的原因在于，对于许多文档（比如维基百科），当这些文档被视为非常高质量的信息来源时，在训练模型的过程中往往会优先从这些来源采样。也就是说，模型很可能已经对这些数据进行了多次训练周期，这意味着它可能已经看过这个网页大约 10 次左右。

这有点像你，比如当你反复阅读某段文字很多很多次，比如说读了一百遍，那么你就能背诵它。这个模型也是如此。如果它看到某样东西太多次，它就能凭记忆背诵出来。但这些模型的效率可能比人类高得多，比如每次演示时。所以它可能只看了 10 次这个维基百科条目，但基本上它已经准确地把这篇文章记在了参数里。

好了，接下来我要展示的是这个模型在训练过程中绝对没有见过的东西。例如，如果我们查阅这篇论文并浏览预训练数据部分，会发现数据集的知识截止日期是 2023 年底。这意味着它不会包含此后发布的任何文档。当然，它也没有关于 2024 年选举及其结果的信息。现在，如果我们用未来的标记来启动模型，它将延续标记序列，并根据其自身参数中的知识做出最佳猜测。那么，让我们来看看这可能是什么样子。共和党的小子，特朗普。

好的，2017 年的美国总统。让我们看看接下来会说什么。例如，模型必须猜测竞选搭档以及对手是谁，等等。让我们按下回车键。这里列出了迈克·彭斯作为竞选搭档而非 J.D.万斯的情况。而当时的竞选对手是希拉里·克林顿和蒂姆·凯恩。所以这可能是警报预示的另一个有趣的平行宇宙。让我们换个样本试试。同样的提示，重新采样。所以在这里，竞选搭档是罗恩·德桑蒂斯，他们与乔·拜登和卡玛拉·哈里斯竞争。这又是一个不同的平行宇宙。因此，模型会做出有根据的猜测，并根据这些知识继续生成标记序列。

我们在这里看到的这些现象，其实就是所谓的"幻觉"。模型只是以概率的方式做出最佳猜测。接下来我想展示的是，尽管这只是一个基础模型，还不是一个助手模型，但如果你在提示设计上足够巧妙，它仍然可以应用于实际场景。

这就是我们所说的“few-shot 提示”。具体来说，我这里有 10 个单词或 10 对词组，每对都是一个英文单词后接冒号，然后是它的韩语翻译。我们总共有 10 组这样的对应关系。该模型的作用是，在最后我们会看到“teacher:”，然后在这里我们将进行一个仅包含五个标记的补全。这些模型具备我们所说的上下文学习能力。这指的是，当模型在读取这段上下文时，它会在过程中学习到数据中存在某种算法模式，并知道要继续遵循这种模式。这就是所谓的上下文学习。它扮演了翻译者的角色，当我们点击完成时，会看到老师被翻译为 "xxx"，这是正确的。由此可见，即使目前我们仅拥有基础模型，通过巧妙地设计提示词，也能构建出实用的应用程序。

它依赖于我们所谓的上下文学习能力，这是通过构建所谓的少样本提示（few-shot prompt）来实现的。最后，我想告诉大家，其实有一种巧妙的方法，仅通过提示就能实例化一个完整的语言模型助手。其诀窍在于，我们将设计一个看起来像网页的提示，其中包含一位乐于助人的 AI 助手与人类之间的对话。然后模型会继续这段对话。实际上，为了撰写提示词，我直接求助了 ChatGPT 本身，这有点 "meta" 的感觉——我告诉它：我想创建一个 LLM 助手，但我只有基础模型。所以你能帮我写提示词吗？这就是它给出的方案，说实话相当不错。

以下是 AI 助手和人类之间的一段对话。这位 AI 助手知识渊博、乐于助人，能够回答各种各样的问题，等等。然而，仅仅给出这样的描述是不够的。如果你创建这个少量示例提示，效果会好得多。这里有一些人类助手的术语，人类助手。我们还有一些对话轮次。最后在这里，我们将放入我们喜欢的实际查询。让我把这个复制粘贴到基础模型提示中。现在让我来做人类列部分。

这就是我们放置实际提示的地方。为什么天空是蓝色的？让我们运行助手。天空呈现蓝色是由于一种称为瑞利散射的现象，等等，等等。

所以你看，基础模型只是在延续序列。但由于这个序列看起来像对话，它就扮演了那个角色。不过这里有点微妙，因为它只是...你看，它结束了助手的部分，然后就开始幻想人类的下一个问题，诸如此类。所以它会一直持续下去。但你可以看到我们已经某种程度上完成了任务。如果你就拿这个来说，为什么天空是蓝色的？如果我们刷新一下并放在这里，当然我们不指望基础模型能处理这个，对吧？我们只是，谁知道会得到什么结果。

好的，我们还会遇到更多问题。那么，这是一种创建助手的方法，即使你可能只有一个基础模型。好了，这就是我们刚才几分钟讨论内容的简要总结。

现在，让我把视角拉远一点。这大致就是我们目前讨论的内容。我们希望训练像 ChatGPT 这样的大型语言模型助手。

### 预训练到后训练

我们已经讨论了第一阶段，即预训练阶段。我们看到，实际上这一阶段的核心在于：我们获取互联网文档，将其分解为这些标记（token）——这些小文本片段的基本单元，然后利用神经网络预测标记序列。整个这一阶段的输出就是这个基础模型。

这是在设置参数。而这个基础模型本质上是一个基于词元级别的互联网文档模拟器。因此，它能够生成具有与互联网文档相似统计特性的词元序列。我们发现它可以应用于某些场景，但实际上我们还需要做得更好。



It is the setting of the parameters. And this base model is basically an internet document simulator on the token level. So it can just, it can generate token sequences that have the same kind of like statistics as internet documents. And we saw that we can use it in some applications, but we actually need to do better. 

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


