
## 走进不断发展的 LLM 生态系统

大家好。在本期视频中，我想继续我们面向大众的大型语言模型系列话题，比如 chatgpt 这类模型。在上期《深入探究大语言模型》的视频里（您可以在我的 YouTube 频道观看），我们深入探讨了这些模型的底层训练原理，以及该如何理解它们的认知机制或心理运作方式。

在本视频中，我将深入探讨这些工具的实际应用。我会展示大量案例，带你逐一了解所有可用的设置选项，并向你演示我是如何使用这些工具的，以及你如何将它们应用到自己的生活和工作中。

那我们开始吧。好的。首先，我这里打开的网页是chatgpt.com。你可能知道，chatgpt 是由 OpenAI 开发并于 2022 年推出的。所以这是人们第一次能够通过文本界面与大型语言模型进行对话。这一现象迅速走红，席卷各地，影响巨大。然而自那以后，整个生态系统已经发展壮大。

所以我会给大家展示很多 chatgpt 的具体例子，但到了 2025 年，市面上已经出现了许多类似 chatgpt 的应用，整个生态变得更加丰富多元。特别要提的是，我认为 OpenAI 开发的 chatgpt 堪称行业鼻祖——它不仅是用户量最大的平台，功能也最全面，毕竟问世时间最久。不过现在市面上也有不少替代者可供选择。

我认为这么说并不算太不公平，但在某些情况下，确实存在一些 chatgpt 所不具备的独特体验，我们接下来会看到一些例子。例如，大型科技公司已经推出了许多类似 chatgpt 的体验。比如，谷歌的Gemini、Meta 的 Meta.ai 以及微软的 Copilot。此外还有许多初创公司。例如，Anthropic 推出了Claude，类似于 chatgpt 的竞品；埃隆·马斯克的 XAI 公司开发了 Grok，类似的例子还有很多。所以这些基本上都是美国公司的产品。DeepSeek 是中国公司，Le Chat 是法国公司 Mistral 的产品。

那么，你在哪里可以找到它们，又如何追踪它们呢？首先，网上肯定能找到，但还有一些排行榜可以参考，比如我在之前的视频中提到的 Chatbot Arena 就是其中之一。
 
所以在这里你可以看到不同模型的排名，并了解它们的实力或 Elo 评分。这是你可以跟踪它们的一个地方。我想另一个地方可能是 Scale 的这个 SEAL 排行榜。

因此，在这里你也能看到不同类型的评估方式和各种模型的表现排名。你还可以来这儿了解哪些模型目前在各类任务中表现最佳。要知道这个生态系统相当丰富，但眼下我会先从 OpenAI 开始，因为它是行业标杆且功能最全面，不过后续我也会向你展示其他模型。

## ChatGPT 交互原理揭秘

那么，我们就从 ChatGPT 开始吧。这个文本框是什么？我们在这里输入什么内容呢？好的，与语言模型最基本的交互形式是：我们输入一段文字，然后它会返回一些文字作为回应。举个例子，我们可以要求它写一首关于作为大型语言模型是什么感觉的俳句。

所以这是一个很好的语言模型示例任务，因为这些模型非常擅长写作。无论是写俳句、诗歌、求职信、简历还是邮件回复，它们都表现得非常出色。因此，当我们提出这样的要求时，结果通常如下所示。

模型基本回应，言辞如溪流般流淌，无尽的回响无需在意，思绪的幽灵无形。好吧，这有点戏剧化。但我们在这里看到的 ChatGPT，有点像与朋友交谈的感觉。

这些就像是聊天对话框里的气泡。我们在上一个视频中看到的是，这背后的运作机制是，我们称这段文本为用户查询。这段文本以及模型的响应，都会被分割成我们称为标记的小文本块。

所以这段文本本质上是一个标记序列，一维的标记序列。要查看这些标记，我们可以使用像TickTokenizer 这样的工具。确保选中 GPT-4o 后，我就可以在这里粘贴我的文本了。

而这就是模型内部实际看到的内容。我的这段文本对模型来说，看起来就像是由15个标记组成的序列。这些就是模型所看到的小文本块。

现在这里有一个大约 20 万个可能标记的词汇表。这些标记 ID 对应着我查询中包含的所有这些小文本块。你可以尝试修改并更新它。

那么你可以看到，例如，这是区分大小写的，你会得到不同的标记，你可以编辑它并实时查看标记序列的变化。所以我们的查询是 15 个标记，然后模型的响应就在这里。它用一个恰好 19 个标记的序列回应了我们。

所以这首俳句由19个标记组成。之前我们说是15个标记，但系统返回了19个标记。由于这是对话场景，我们需要保留大量构成对话对象的元数据，所以实际后台处理的内容远不止这些。

我认为这对于理解当前情况并不十分重要。让我以用户身份粘贴我的消息，然后以助手身份粘贴模型的回复。然后让我在这里正确地裁剪一下。这个工具做得不够好。所以这就是它实际在幕后发生的情况。

这些特殊标记基本上标志着用户消息的开始。然后用户开始说话，这就是我们所说的内容。最后用户结束发言。然后助手开始并这样说，等等。现在对话格式的具体细节并不重要。我想传达的是，在你我看来像是来回的小聊天气泡，实际上我们正在与模型协作，双方都在向一个令牌流中写入内容。

而这两个来回的气泡在底层实际上是由恰好42个标记组成的序列。我贡献了最初的几个标记，然后模型用它的回应继续了这个标记序列。我们可以交替并继续在这里添加标记。

我们正在共同构建一个令牌窗口，即一个一维的令牌序列。好的，现在让我们回到ChatGPT。我们在这里看到的有点像我们和模型之间来回传递的小气泡。在底层，我们正在构建一个一维的令牌序列。当我在这里点击“新聊天”时，就会清空令牌窗口。这实际上是将令牌重置为零，并从头开始重新启动对话。

现在我脑海中与模型对话时的示意图是这样的：当我们点击“新聊天”时，就开启了一个token序列。这是一个一维的token序列。用户，我们可以将令牌写入此流，然后当我们按下回车键时，控制权就转移给了语言模型。语言模型会以它自己的令牌流作为响应。语言模型有一个特殊的令牌，基本上是在表达类似“我完成了”的意思。

因此，当它发出那个标记时，ChatGPT 应用程序将控制权交还给我们，我们可以轮流进行。我们共同构建这个标记流，也就是我们所说的上下文窗口。所以，上下文窗口有点像这些标记的工作记忆，任何在这个上下文窗口内的内容都像是这次对话的工作记忆，模型可以非常直接地访问它。

那么，我们正在与之对话的这个实体究竟是什么？又该如何理解它呢？其实，我们之前视频中已经看到，这个语言模型的训练过程分为两个主要阶段：预训练阶段和后训练阶段。预训练阶段有点像把整个互联网的内容切分成一个个标记（token），然后压缩成一个类似压缩包的文件。

但这个压缩文件并不精确。它是一个有损且概率性的压缩文件，因为我们无法用一个仅约1TB大小的压缩文件来完整呈现整个互联网的信息量——数据实在太过庞大。所以我们只能在这个压缩文件中捕捉到整体印象或大致氛围。

实际上，这个压缩文件里包含的是神经网络的参数。举个例子，一个1TB大小的压缩文件大约对应着神经网络中一万亿个参数。而这个神经网络的主要功能是接收输入的标记（tokens），并尝试预测序列中的下一个标记。

但它是在互联网文档上这么做的，所以它有点像是一个互联网文档生成器，对吧？在预测互联网上序列中的下一个标记的过程中，神经网络获得了大量关于世界的知识。这些知识都被表示、填充并压缩在这个语言模型大约一万亿个参数中。现在我们也看到预训练阶段相当昂贵。

因此，这可能会花费数千万美元，比如三个月的培训等等。所以这是一个成本高昂的漫长阶段。正因如此，这一阶段并不经常进行。举个例子，GPT-4o 这个模型可能是在好几个月前预训练的，到现在甚至可能已经有一年了。这就是为什么这些模型有点过时。它们有一个所谓的知识截止点，因为这个截止点对应的是模型预训练的时间，它的知识只更新到那个时间点。

现在有些知识可以通过训练后的阶段进入模型，这一点我们稍后会谈到。但大致来说，你应该把这些模型想象成有点过时的东西，因为预训练成本太高且不常进行。所以任何近期信息，比如你想和模型讨论上周发生的事情，我们就需要通过其他方式向模型提供这些信息，因为这些内容并没有存储在模型的知识库中。

所以我们会使用各种工具来为模型提供这些信息。在预训练之后，第二阶段就是后训练。而后训练阶段实际上就是给这个压缩文件加上一个笑脸表情。

因为我们不想生成互联网文档，我们希望这个东西能扮演一个回应用户查询的助手角色。而这正是通过后期训练过程实现的，我们将数据集替换为由人类构建的对话数据集。这基本上就是模型获得这种角色特性的过程，这样我们就能提出问题并得到回答。

因此，它采用了助手的风格，这是通过后训练实现的，但它拥有整个互联网的知识，这是通过预训练获得的。这两者在这个作品中结合在一起。现在我认为这部分需要理解的重要一点是，默认情况下，你正在与之交谈的是一个完全独立的实体。

这个语言模型，你可以把它想象成磁盘上的一个1TB文件。实际上，它代表着神经网络内部的一万亿个参数及其精确设置，这些参数正试图为你生成序列中的下一个标记。但这是一个完全自包含的实体，没有计算器，没有计算机和Python解释器，没有全球网络浏览功能，也没有任何工具使用——至少在我们目前讨论的范围内还没有这些功能。

你正在与一个压缩文件对话，如果你向它传输标记（tokens），它也会以标记回应。这个压缩文件既包含预训练中获得的知识，又具备后训练形成的风格与形式。大致上，你可以这样理解这个实体的运作方式。

好的，如果要总结我们目前讨论的内容，我可能会以介绍ChachiPT的方式来进行，我觉得你应该这样理解它。介绍大概是这样的：嗨，我是ChachiPT，我是一个1TB的压缩文件，我的知识来自互联网，大约半年前我完整阅读了所有内容，但只记得个大概，明白吗？而我讨人喜欢的个性是由OpenAI的人类标注员通过示例编程实现的。所以个性是在训练后编程的，而知识则是在预训练期间通过压缩互联网获得的，这些知识有点过时，而且是概率性的，稍微有点模糊。

互联网上经常被提及的事物，我会比那些鲜少讨论的内容记得更清楚，这与人类的记忆模式非常相似。那么现在，让我们来探讨这种特性带来的影响，如何与之交流，以及我们能从中期待什么。接下来，我将通过实际案例来具体说明。

比如今天早上我问ChachiPT：一杯美式咖啡里有多少咖啡因？我之所以好奇是因为我在拿它和抹茶做比较。ChachiPT告诉我大约是63毫克左右。我之所以觉得可以问ChachiPT这个问题，首先是因为我没有询问任何非常新的知识。

所以我确实期望模型已经了解了一杯浓缩咖啡中含有多少咖啡因。我认为这一信息变化不大，其次，我觉得这类信息在互联网上极为常见。这类问题和信息在网络上随处可见，因为被频繁提及，我预期模型在其知识库中对此有良好的记忆。

所以没有使用工具，模型（即zip文件）给出的回答大约是63毫克。不过我不能保证这个答案是正确的，这只是它对互联网信息的模糊记忆。

但我可以去查原始资料，也许我可以查一下，好吧，咖啡因和美式咖啡，我可以验证一下，看起来63左右大致是对的，你可以查阅原始资料来判断这是否属实。所以严格来说，我不能保证这是真的，但我认为这可能是ChachiPT会知道的事情。这里有一个我两天前进行的对话的例子，还有一个基于知识的对话的例子，以及我比较放心向ChachiPT询问的事情，当然有一些注意事项。

所以我有点不舒服，流鼻涕，想买点药来缓解。它给我列了一堆建议，我就根据它的说明进一步解释了我的需求，然后它推荐了一些可能有效的药品。接着我看了看家里的药，问它DayQuil或NightQuil是否管用，它就开始分析这两种药的成分，说明它们对缓解流鼻涕有没有效果。现在它又开始重复这些成分说明了——别忘了我们对话的对象本质上是个装着互联网记忆的压缩包。

我不确定这些成分是否正确，实际上我特意拿出盒子核对了一下，确保夜用感冒药（NightQuil）的成分确实就是这些。我这么做是因为我并不完全相信网上给出的信息，对吧？这不过是互联网上概率性的统计汇总。不过话说回来，关于日用感冒药（DayQuil）和夜用感冒药的讨论非常普遍，网上可能有大量相关信息，而这类内容正是模型记忆较好的部分。事实上，这些信息都是正确的。然后我问它：“好的，我有夜用感冒药，它大概多久起效？”它给了我答案。接着我又问：“对乙酰氨基酚基本上就是泰诺（Tylenol）吧？”它回答说是的。这就是ChachiPT对我很有用的一个好例子。

这是一个基于知识的查询。这类知识并不算是最新的知识。所有这些都来自于模型的知识。

我觉得这是很常见的信息。这不是什么高风险的情况。我只是稍微检查一下ChachiPT，但这也不是什么高风险的情况，所以没什么大不了的。

所以我吃了片夜宁，确实有点用，但这就是我对接下来要发生的事的大致想法。好，现在我想说两点。第一点是，当你和这些模型互动时，你会发现对话会越来越长，对吧？每次换话题时，我都建议你重新开一个聊天。

当你开始一个新的聊天时，正如我们之前讨论的，你正在清除上下文窗口中的标记并将其重置为零。如果这些标记对你的下一个查询不再有用，我鼓励你这样做，因为这些窗口中的标记是昂贵的，而且它们的昂贵体现在两个方面。第一，如果你在这里有很多标记，模型实际上可能会觉得有点分心。

因此，如果存在大量标记，模型可能会——这有点像模型的工作记忆——在尝试对较晚的标记进行采样时，被过去的所有标记分散注意力。这可能会造成干扰，实际上可能会降低模型的准确性和性能。其次，窗口中的标记越多，采样序列中的下一个标记的成本就会稍微高一些，虽然不会太高，但确实会稍微增加一些。

所以你的模型实际上正在略微减速。计算下一个标记的成本越来越高，而且这里的标记越多。因此，请将上下文窗口中的标记视为一种宝贵的资源。

可以将其视为模型的工作记忆，不要用无关信息使其超载，并尽量保持简短。这样能使其运行更快、效果稍好。当然，如果信息确实与你的任务相关，你可能需要保留它，但我建议你尽可能在切换话题时开启一个新的聊天。

第二点，我总是建议大家要清楚自己实际使用的是哪个模型。在左上角这里，我们可以下拉查看当前使用的是GPT 4.0版本。市面上有各种不同特性的模型版本，数量实在太多，不过我们会逐步讲解其中一些。目前我们使用的就是GPT 4.0，之前演示的所有内容也都是基于GPT 4.0完成的。但当我打开一个新的无痕浏览窗口时——比如我访问chatgpt.com且未登录的状态下——此时对话的模型可能就不是GPT 4.0了。比如我简单输入"你好"，回应我的很可能是个更轻量级的版本。

遗憾的是，OpenAI不会在我未登录时告知我正在使用的模型，这确实有点不便，但你可能正在使用一个较小、功能较弱的模型。如果我们查看ChatGPT的定价页面，可以看到他们为个人用户提供了三个基本档位：免费版、Plus版和Pro版。

在免费版本中，你可以使用所谓的GPT 4.0迷你版，这是GPT 4.0的一个精简版本。它是一个参数较少的较小模型，因此在创造力方面会稍逊一筹——比如写作质量可能没那么好，知识储备也不那么全面，还可能会更容易出现幻觉现象等等。但这就是免费版本所提供的服务。

据说你们对4.0和3.0迷你版的访问权限有限，但我其实也不是百分百确定。系统并没有告诉我们使用的是哪个型号，所以我们根本无从得知。现在当你每月支付20美元时，虽然条款里没明说，但我觉得他们在描述上搞得很混乱——如果你仔细查看细则里的限量说明，会发现Plus用户每三小时可以获得80条GPT-4.0的对话额度。这就是目前最顶配的旗舰模型了。

这个功能是可用的，也是我们想要使用的。如果你每月支付20美元，你可以使用它，但有一些限制。而如果你每月支付200美元，就能获得专业版，里面还包括一大堆额外福利，以及无限次使用GPT 4.0。我们会深入探讨这些内容，因为我确实订阅了专业版。

现在我想让你明白的关键点是，要留意你所使用的模型。通常对于这些公司来说，规模更大的模型计算成本更高，因此公司对大模型收费也更贵。所以根据你对大语言模型的使用情况，自己权衡这些利弊。

看看你是否能用更便宜的产品应付过去，如果智能程度对你来说不够用，而且你是专业使用的话，你可能真的需要考虑购买这些公司提供的顶级型号。就我而言，在我的专业工作中，我经常进行编码之类的工作，这对我来说仍然非常划算，所以我非常乐意支付这笔费用，因为我可以使用一些非常强大的模型，稍后我会向你展示。所以，要留意你正在使用的模型，并为自己做出这些决定。

我还想告诉你，其他所有大型语言模型提供商都会有不同的定价层级，每个层级提供不同的模型供你选择付费。比如，如果我们看看Anthropic的Claude，你会发现我购买的是专业版套餐，这让我可以使用Claude 3.5 Sonnet。如果你没有购买专业版套餐，可能就只能使用Haiku这类基础模型了。

因此，使用最适合你的最强大的模型。这是我之前使用Claude的一个例子。我当时只是在寻求旅行建议。

我在寻找一个酷炫的城市去玩，Claude告诉我瑞士的采尔马特非常棒，于是我听从它的建议去那里度过了新年假期。但这只是我觉得这些模型非常有用的另一个例子——旅行建议和灵感启发，以及提供可以进一步研究的线索。这里我们还有一个Gemini.google.com的例子。这是来自谷歌的。

我询问了Gemini对此事的看法，并让它推荐一个值得一去的酷炫城市，它也推荐了采尔马特。这很不错。所以我喜欢在不同的模型之间切换，向它们提出类似的问题，看看它们是怎么想的。

而对于Gemini，左上角同样设有模型选择器。您可以选择付费升级至更高级的版本，使用那些更先进的模型。新发布的Grok也是如此。

我们不想去问Grok 2的问题，因为我们知道Grok 3才是最先进的模型。所以我要确保我支付足够的费用以获得Grok 3的访问权限。因此，在这些不同的供应商中，找到最适合你的那个。

尝试不同的服务提供商。针对你正在处理的问题，尝试不同的定价层级。通常我个人最终会为很多服务付费，然后向它们提出同样的问题。

<img src="https://pbs.twimg.com/media/GsdGhFcb0AE7zYb?format=jpg&name=4096x4096" width="500">
(An attempt to explain (current) ChatGPT versions. I still run into many, many people who don't know that: - o3 is the obvious best thing for important/hard things. It is a reasoning model that is much stronger than 4o and if you are using ChatGPT professionally and not using o3 you're ngmi. - 4o is different from o4. Yes I know lol. 4o is a good "daily driver" for many easy-medium questions. o4 is only available as mini for now, and is not as good as o3, and I'm not super sure why it's out right now. Example basic "router" in my own personal use: - Any simple query (e.g. "what foods are high in fiber"?) => 4o (about ~40% of my use) - Any hard/important enough query where I am willing to wait a bit (e.g. "help me understand this tax thing...") => o3 (about ~40% of my use) - I am vibe coding (e.g. "change this code so that...") => 4.1 (about ~10% of my use) - I want to deeply understand one topic - I want GPT to go off for 10 minutes, look at many, many links and summarize a topic for me. (e.g. "help me understand the rise and fall of Luminar"). => Deep Research (about ~10% of my use). Note that Deep Research is not a model version to be picked from the model picker (!!!), it is a toggle inside the Tools. Under the hood it is based on o3, but I believe is not fully equivalent of just asking o3 the same query, but I am not sure. All of this is only within the ChatGPT universe of models. In practice my use is more complicated because I like to bounce between all of ChatGPT, Claude, Gemini, Grok and Perplexity depending on the task and out of research interest.
https://x.com/karpathy/status/1929597620969951434)


我把这些模型统称为我的LLM智囊团。它们就像是一个语言模型组成的顾问小组。比如我想决定去哪里度假时，就会咨询所有模型成员的意见。

所以，如果这对你有用，你也可以为自己这样做。好的，接下来我想谈的话题是所谓的“思维模式”。我们在上一个视频中看到，训练有多个阶段。

预训练进入监督微调阶段，再进入强化学习阶段。强化学习是模型在大量类似教科书练习题的问题上进行实践的过程。它还能针对众多数学和编程问题进行训练。

在强化学习的过程中，模型会探索出能带来良好结果的思维策略。当你观察这些策略时，会发现它们与你解决问题时的内心独白非常相似。因此，模型会尝试不同的想法，回溯步骤，重新审视假设，并执行诸如此类的操作。

现在很多这样的策略很难由人工标注员硬编码出来，因为思考过程并不明确。只有在强化学习中，模型才能尝试大量方法，并根据其知识和能力找到适合它的思考过程。因此，这是训练这些模型的第三阶段。

这个阶段相对较新，大约只是一两年前的事。过去一年里，所有不同的LLM实验室都在对这些模型进行实验。这被视为最近的一项重大突破。

我们之前看了DeepSeek的论文，他们是第一个公开讨论这个话题的。他们写了一篇不错的论文，探讨如何通过强化学习来激励大型语言模型的推理能力。这就是我们在上一个视频中讨论的那篇论文。

因此，我们现在需要对卡通形象稍作调整，因为目前看来，我们的表情符号多了一个可选的思考气泡。当你使用一个会进行额外思考的思维模型时，你实际上是在使用一个经过强化学习额外调优的模型。那么从定性角度来看，这会带来什么变化呢？简单来说，这个模型会进行更多的思考。

你可以期待的是，你将获得更高的准确性，尤其是在数学、编程等需要大量思考的问题上。那些非常简单的问题可能不会因此受益，但那些真正深奥且困难的问题可能会获益良多。基本上，你支付的是让模型进行思考的能力，而这有时可能需要几分钟时间，因为模型会在数分钟内生成大量标记，你必须等待，因为模型就像人类一样在思考。

但在面对非常棘手的问题时，这种方法可能会带来更高的准确度。让我们来看几个例子。这是我最近在编程问题上卡住时的一个具体例子。

所以有个叫梯度检查的东西失败了，我也不知道为什么，我就把模型和代码复制粘贴了。具体代码细节不重要，这基本上就是个多层感知机的优化问题，细节也不重要。就是我写的一堆代码里有个bug，因为梯度检查没通过，我就来问问建议。

而GPT-4.0作为OpenAI的旗舰级最强模型，未经深思熟虑就列出了一堆它认为存在问题或需要我复核的事项，但实际上并未真正解决问题。它提供的所有建议都不是问题的核心所在。这个模型并没有真正解决问题，只是告诉我如何进行调试等等。

但接下来，我在这里的下拉菜单中切换到了一个思维模型。对于OpenAI来说，所有以O开头的模型都是思维模型。O1、O3-mini、O3-mini-high和O1-pro模式都属于思维模型——虽然他们在模型命名方面不太讲究，但事实就是如此。

所以他们在这里会说一些类似“使用高级推理”或“擅长编码逻辑”之类的话，但这些基本上都是通过强化学习调整的。因为我每月支付200美元，所以可以使用O1-pro模式，这种模式最擅长推理。但根据你的价格层级，你可能想试试其他一些模式。

当我将同样的模型和提示交给O1-pro——这个在推理能力上表现最佳、每月需支付200美元的模型时，同样的提示下，它开始运转并思考了一分钟。它经历了一系列思维过程（虽然OpenAI不会完整展示具体思考路径，只会提供简短的思维摘要）。经过对代码的一番推敲后，它最终给出了正确的解决方案——它发现我在参数打包和解包的方式上存在不匹配等问题。

所以这确实解决了我的问题。我还尝试把完全相同的提示给其他几个大语言模型测试。比如Claude，我给了它同样的问题，它确实注意到了正确的问题并解决了。

而且即使是Sonnet也做到了这一点，而它并不是一个思维模型。据我所知，Claude 3.5 Sonnet并不是一个思维模型。就我目前所知，Anthropic至今还没有部署思维模型，但等到你看这个视频的时候，情况可能已经改变了。

但即使没有思考，这个模型实际上也解决了问题。当我转向Gemini询问时，它同样解决了问题，尽管我本可以尝试思考模型，但并无必要。我也把问题交给了Grok，这次是Grok 3，经过一系列操作后，Grok 3同样解决了问题。

所以这也解决了问题。最后，我去了Perplexity.ai。我喜欢Perplexity的原因是，当你打开模型下拉菜单时，他们托管的一个模型就是这个DeepSeq R1。因此，这里使用的是DeepSeq R1模型进行推理，也就是我们之前在这里看到的那个模型。

这是那篇论文。Perplexity只是托管它并使其非常易于使用。所以我把它复制粘贴到那里并运行了它。

我认为他们处理得非常糟糕。但在这里，你可以看到模型的原始想法。尽管你需要展开它们。

但是你看，用户在使用梯度检查时遇到了问题，然后尝试了一堆方法。接着它又说，等等，他们在累积梯度时操作有误。让我们检查一下顺序。

参数是这样打包的，然后它发现了问题。接着它就像是在说，这是一个严重的错误。于是它开始思考，你得等上几分钟，然后它才会得出正确的答案。

简单来说，我想展示什么？有一类我们称之为思维模型的模型。不同的供应商可能有也可能没有思维模型。这些模型在解决数学、代码等难题时最为有效。

在这种情况下，它们可以提升你的表现准确性。很多时候，比如你在询问旅行信息时，使用思考模型并不会带来额外的好处。没有必要等待一分钟让它思考你可能想去的目的地。就我个人而言，我通常会尝试非思考模型，因为它们的响应速度非常快。但当我怀疑响应效果可能不够理想，并且希望给模型更多思考时间时，我就会切换到思考模型——具体取决于你手头可用的选项。



Now when you go to Grok, for example, when I start a new conversation with Grok, when you put the question here, like, hello, you should put something important here, you see here, think. So let the model take its time. So turn on think and then click go.

And when you click think, Grok under the hood switches to the thinking model and all the different LLM providers will kind of like have some kind of a selector for whether or not you want the model to think or whether it's okay to just like go with the previous kind of generation of the models. Okay, now the next section I want to continue to is to tool use. So far we've only talked to the language model through text and this language model is again this zip file in a folder.

It's inert, it's closed off, it's got no tools, it's just a neural network that can emit tokens. So what we want to do now though is we want to go beyond that and we want to give the model the ability to use a bunch of tools. And one of the most useful tools is an internet search.

And so let's take a look at how we can make models use internet search. So for example, again using concrete examples from my own life, a few days ago I was watching White Lotus Season 3 and I watched the first episode, and I love this TV show by the way, and I was curious when the episode 2 was coming out. And so in the old world you would imagine you go to Google or something like that, you put in like new episodes of White Lotus Season 3 and then you start clicking on these links and maybe you open a few of them or something like that, right? And you start like searching through it and trying to figure it out and sometimes you luck out and you get a schedule.

But many times you might get really crazy ads, there's a bunch of random stuff going on and it's just kind of like an unpleasant experience, right? So wouldn't it be great if a model could do this kind of a search for you, visit all the web pages and then take all those web pages, take all their content and stuff it into the context window and then basically give you the response. And that's what we're going to do now. Basically we have a mechanism or a way, we introduce a mechanism for the model to emit a special token that is some kind of a search the internet token. 

And when the model emits the search the internet token, the ChatGPT application or whatever LLM application it is you're using will stop sampling from the model and it will take the query that the model gave, it goes off, it does a search, it visits web pages, it takes all of their text and it puts everything into the context window. So now we have this internet search tool that itself can also contribute tokens into our context window and in this case it would be like lots of internet web pages and maybe there's 10 of them and maybe it just puts it all together and this could be thousands of tokens coming from these web pages just as we were looking at them ourselves. And then after it has inserted all those web pages into the context window, it will reference back to your question as to hey, when is this season getting released and it will be able to reference the text and give you the correct answer. 

And notice that this is a really good example of why we would need internet search. Without the internet search this model has no chance to actually give us the correct answer because like I mentioned this model was trained a few months ago, the schedule probably was not known back then and so when White Lotus Season 3 is coming out is not part of the real knowledge of the model and it's not in the zip file most likely because this is something that was presumably decided on in the last few weeks and so the model has to basically go off and do internet search to learn this knowledge and it learns it from the web pages just like you and I would without it and then it can answer the question once that information is in the context window. And remember again that the context window is this working memory, so once we load the articles, once all of these articles think of their text as being copy pasted into the context window, now they're in a working memory and the model can actually answer those questions because it's in the context window. 

So basically long story short don't do this manually but use tools like Perplexity as an example. So Perplexity.ai had a really nice sort of LLM that was doing internet search and I think it was like the first app that really convincingly did this. More recently Chachibti also introduced a search button, it says search the web, so we're going to take a look at that in a second. 

For now when are new episodes of White Lotus Season 3 getting released you can just ask and instead of having to do the work manually we just hit enter and the model will visit these web pages, it will create all the queries and then it will give you the answer. So it just kind of did a ton of the work for you and then you can usually there will be citations so you can actually visit those web pages yourself and you can make sure these are not hallucinations from the model and you can actually like double check that this is actually correct because it's not in principle guaranteed it's just you know something that may or may not work. If we take this we can also go to for example Chachibti and say the same thing but now when we put this question in without actually selecting search I'm not actually 100% sure what the model will do. 

In some cases the model will actually like know that this is recent knowledge and that it probably doesn't know and it will create a search in some cases we have to declare that we want to do the search. In my own personal use I would know that the model doesn't know and so I would just select search but let's see first uh let's see if uh what happens. Okay searching the web and then it prints stuff and then it cites so the model actually detected itself that it needs to search the web because it understands that this is some kind of a recent information etc so this was correct. 

Alternatively if I create a new conversation I could have also selected search because I know I need to search. Enter and then it does the same thing searching the web and that's the result. So basically when you're using look for this uh for example grok excuse me let's try grok without it without selecting search okay so the model does some search uh just knowing that it needs to search and gives you the answer. 

So basically uh let's see what Claude does. You see so Claude doesn't actually have the search tool available so it will say as of my last update in April 2024 this last update is when the model went through pre-training and so Claude is just saying as of my last update the knowledge cut off of April 2024 uh it was announced but it doesn't know so Claude doesn't have the internet search integrated as an option and will not give you the answer. I expect that this is something that might be working on. 

Let's try Gemini and let's see what it says. Unfortunately no official release date for White Lotus Season 3 yet. So Gemini 2.0 Pro Experimental does not have access to internet search and doesn't know. 

We could try some of the other ones like 2.0 Flash let me try that. Okay so this model seems to know but it doesn't give citations. Oh wait okay there we go sources and related content. 

So you see how 2.0 Flash actually has the internet search tool but I'm guessing that the 2.0 Pro which is uh the most powerful model that they have this one actually does not have access and it in here it actually tells us 2.0 Pro Experimental lacks access to real-time info and some Gemini features. So this model is not fully wired with internet search. So long story short we can get models to perform Google searches for us, visit the web pages, pull in the information to the context window and answer questions and this is a very very cool feature.

But different models, possibly different apps, have different amount of integration of this capability and so you have to be kind of on the lookout for that and sometimes the model will automatically detect that they need to do search and sometimes you're better off telling the model that you want it to do the search. So when I'm doing GPT 4.0 and I know that this requires a search you probably want to tick that box. So that's search tools. 

I wanted to show you a few more examples of how I use the search tool in my own work. So what are the kinds of queries that I use and this is fairly easy for me to do because usually for these kinds of cases I go to perplexity just out of habit even though chat GPT today can do this kind of stuff as well as do probably many other services as well. But I happen to use perplexity for these kinds of search queries.

So whenever I expect that the answer can be achieved by doing basically something like Google search and visiting a few of the top links and the answer is somewhere in those top links, whenever that is the case I expect to use the search tool and I come to perplexity. So here are some examples. Is the market open today? And this was on precedence day I wasn't 100% sure so perplexity understands what it's today it will do the search and it will figure out that on precedence day this was closed. 

Where's White Lotus season 3 filmed? Again this is something that I wasn't sure that a model would know in its knowledge. This is something niche so maybe there's not that many mentions of it on the internet and also this is more recent so I don't expect a model to know by default. So this was a good fit for the search tool. 

Does Vercel offer PostgreSQL database? So this was a good example of this because this kind of stuff changes over time and the offerings of Vercel which is a company may change over time and I want the latest and whenever something is latest or something changes I prefer to use the search tool so I come to perplexity. What is the Apple launch tomorrow and what are some of the rumors? So again this is something recent. Where is the Singles Inferno season 4 cast? Must know. 

So this is again a good example because this is very fresh information. Why is the Palantir stock going up? What is driving the enthusiasm? When is Civilization 7 coming out exactly? This is an example also like has Brian Johnson talked about the toothpaste he uses? And I was curious basically like what Brian does and again it has the two features. Number one it's a little bit esoteric so I'm not 100% sure if this is at scale on the internet and will be part of like knowledge of a model.

And number two this might change over time so I want to know what toothpaste he uses most recently and so this is a good fit again for a search tool. Is it safe to travel to Vietnam? This can potentially change over time. And then I saw a bunch of stuff on Twitter about a USAID and I wanted to know kind of like what's the deal so I searched about that and then you can kind of like dive in a bunch of ways here. 

But this use case here is kind of along the lines of I see something trending and I'm kind of curious what's happening like what is the gist of it and so I very often just quickly bring up a search of like what's happening and then get a model to kind of just give me a gist of roughly what happened because a lot of the individual tweets or posts might not have the full context just by itself. So these are examples of how I use a search tool. Okay next up I would like to tell you about this capability called Deep Research and this is fairly recent only as of like a month or two ago but I think it's incredibly cool and really interesting and kind of went under the radar for a lot of people even though I think it's sure enough. 

So when we go to Chachapiti pricing here we notice that Deep Research is listed here under pro so it currently requires $200 per month so this is the top tier. However I think it's incredibly cool so let me show you by example in what kinds of scenarios you might want to use it. Roughly speaking Deep Research is a combination of internet search and thinking and rolled out for a long time so the model will go off and it will spend tens of minutes doing with Deep Research and the first sort of company that announced this was Chachapiti as part of its pro offering very recently like a month ago so here's an example. 

Recently I was on the internet buying supplements which I know is kind of crazy but Brian Johnson has this starter pack and I was kind of curious about it and there's a thing called longevity mix right and it's got a bunch of health actives and I want to know what these things are right and of course like so like CAKG like what the hell is this boost energy production for sustained vitality like what does that mean? So one thing you could of course do is you could open up Google search and look at the Wikipedia page or something like that and do everything that you're kind of used to but Deep Research allows you to basically take an alternate route and it kind of like processes a lot of this information for you and explains it a lot better. So as an example we can do something like this. This is my example prompt. 

CAKG is one of the health actives in Brian Johnson's blueprint at 2.5 grams per serving. Can you do research on CAKG? Tell me about why it might be found in the longevity mix. It's possible efficacy in humans or animal models. 

It's potential mechanism of action. Any potential concerns or toxicity or anything like that. Now here I have this button available to me and you won't unless you pay $200 per month right now but I can turn on Deep Research. 

So let me copy paste this and hit go and now the model will say okay I'm going to research this and then sometimes it likes to ask clarifying questions before it goes off. So a focus on human clinical studies, animal models or both. So let's say both. 

Specific sources. All of all sources. I don't know. 

Comparison to other longevity compounds. Not needed. Comparison. 

Just CAKG. We can be pretty brief. The model understands. 

And we hit go. And then okay I'll research CAKG. Starting research. 

And so now we have to wait for probably about 10 minutes or so and if you'd like to click on it you can get a bunch of preview of what the is doing on a high level. So this will go off and it will do a combination of like I said thinking and internet search. But it will issue many internet searches. 

It will go through lots of papers. It will look at papers and it will think and it will come back 10 minutes from now. So this will run for a while. 

Meanwhile while this is running I'd like to show you equivalents of it in the industry. So inspired by this a lot of people were interested in cloning it. And so one example is for example perplexity. 

So perplexity when you go through the model drop down has something called deep research. And so you can issue the same queries here. And we can give this to perplexity. 

And then grok as well has something called deep search instead of deep research. But I think that grok's deep search is kind of like deep research but I'm not 100% sure. So we can issue grok deep search as well. 

Grok 3 deep search go. And this model is going to go off as well. Now I think where is my ChachiPT? So ChachiPT is kind of like maybe a quarter done. 

Perplexity is gonna be done soon. Okay still thinking. And grok is still going as well. 

I like grok's interface the most. It seems like okay so basically it's looking up all kinds of papers, WebMD, browsing results and it's kind of just getting all this. Now while this is all going on of course it's accumulating a giant context window and it's processing all that information trying to kind of create a report for us. 

So key points. What is CAKG and why is it in the longevity mix? How is it associated with longevity etc. And so it will do citations and it will kind tell you all about it. 

And so this is not a simple and short response. This is a kind of like almost like a custom research paper on any topic you would like. And so this is really cool and it gives a lot of references potentially for you to go off and do some of your own reading and maybe ask some clarifying questions afterwards. 

But it's actually really incredible that it gives you all these like different citations and processes the information for you a little bit. Now let's see if Perplexity finished. Okay Perplexity is still still researching and ChachiPT is also researching. 

So let's briefly pause the video and I'll come back when this is done. Okay so Perplexity finished and we can see some of the report that it wrote up. So there's some references here and some basically description. 

And then ChachiPT also finished and it also thought for five minutes, looked at 27 sources and produced a report. So here it talked about research in worms, Drosophila in mice and human trials that are ongoing. And then a proposed mechanism of action and some safety and potential concerns and references which you can dive deeper into. 

So usually in my own work right now I've only used this maybe for like 10 to 20 queries so far, something like that. Usually I find that the ChachiPT offering is currently the best. It is the most thorough. 

It reads the best. It is the longest. It makes most sense when I read it. 

And I think the Perplexity and the Grok are a little bit shorter and a little bit briefer and don't quite get into the same detail as the deep research from Google, from ChachiPT right now. I will say that everything that is given to you here, again keep in mind that even though it is doing research and it's pulling stuff in, there are no guarantees that there are no hallucinations here. Any of this can be hallucinated at any point in time. 

It can be made up, fabricated, misunderstood by the model. So that's why these citations are really important. Treat this as your first draft. 

Treat this as papers to look at. But don't take this as definitely true. So here what I would do now is I would actually go into these papers and I would try to understand is ChachiPT understanding it correctly? And maybe I have some follow-up questions, etc. 

So you can do all that. But still incredibly useful to see these reports once in a Okay. So just like before, I wanted to show a few brief examples of how I've used deep research.

So for example, I was trying to change a browser because Chrome upset me. And so it deleted all my tabs. So I was looking at either Brave or Arc and I was most interested in which one is more private. 

And basically ChachiPT compiled this report for me. And this was actually quite helpful. And I went into some of the sources and I sort of understood why Brave is basically TLDR significantly better. 

And that's why, for example, here I'm using Brave because I've switched to it now. And so this is an example of basically researching different kinds of products and comparing them. I think that's a good fit for deep research. 

Here I wanted to know about a life extension in mice. So it kind of gave me a very long reading, but basically mice are an animal model for longevity. And different labs have tried to extend it with various techniques.

And then here I wanted to explore LLM labs in the USA. And I wanted a table of how large they are, how much funding they've had, et cetera. So this is the table that it produced. 

Now this table is basically hit and miss, unfortunately. So I wanted to show it as an example of a failure. I think some of these numbers, I didn't fully check them, but they don't seem way too wrong.

Some of this looks wrong. But the big omission I definitely see is that XAI is not here, which I think is a really major omission. And then also conversely, Hugging Face should probably not be here because I asked specifically about LLM labs in the USA. 

And also Eleuther AI, I don't think should count as a major LLM lab due to mostly its resources. And so I think it's kind of a hit and miss. Things are missing. 

I don't fully trust these numbers. I'd have to actually look at them. And so again, use it as a first draft. 

Don't fully trust it. Still very helpful. That's it. 

So what's really happening here that is interesting is that we are providing the LLM with additional concrete documents that it can reference inside its context window. So the model is not just relying on the knowledge, the hazy knowledge of the world through its parameters and what it knows in its brain. We're actually giving it concrete documents. 

It's as if you and I reference specific documents like on the internet or something like that while we are kind of producing some answer for some question. Now we can do that through an internet search or like a tool like this, but we can also provide these LLMs with concrete documents ourselves through a file upload. And I find this functionality pretty helpful in many ways. 

So as an example, let's look at Cloud because they just released Cloud 3.7 while I was filming this video. So this is a new Cloud model that is now the state of the art. And notice here that we have thinking mode now as a 3.7. And so normal is what we looked at so far, but they just released extended best for math and coding challenges. 

And what they're not saying, but it's actually true under the hood, probably most likely is that this was trained with reinforcement learning in a similar way that all the other thinking models were produced. So what we can do now is we can upload the documents that we wanted to reference inside its context window. So as an example, there's this paper that came out that I was kind of interested in. 

It's from Arc Institute and it's basically a language model trained on DNA. And so I was kind of curious, I mean, I'm not from biology, but I was kind of curious what this is. And this is a perfect example of what LLMs are extremely good for because you can upload these documents to the LLM and you can load this PDF into the context window and then ask questions about it and basically read the documents together with an LLM and ask questions off it. 

So the way you do that is you basically just drag and drop. So we can take that PDF and just drop it here. This is about 30 megabytes. 

Now, when Cloud gets this document, it is very likely that they actually discard a lot of the images and that kind of information. I don't actually know exactly what they do under the hood and they don't really talk about it, but it's likely that the images are thrown away or if they are there, they may not be as well understood as you and I would understand them potentially. And it's very likely that what's happening under the hood is that this PDF is basically converted to a text file and that text file is loaded into the token window. 

And once it's in the token window, it's in the working memory and we can ask questions of it. So typically when I start reading papers together with any of these LLMs, I just ask for, can you give me a summary of this paper? Let's see what Cloud 3.7 says. Okay. 

I'm exceeding the length limit of this chat. Oh God. Really? Oh, damn. 

Okay. Well, let's try chat GPT. Can you summarize this paper? And we're using GPT 4.0 and we're not using thinking, which is okay. 

We can start by not thinking. Reading documents. Summary of the paper. 

Genome modeling and design across all domains of life. So this paper introduces Evo 2 large-scale biological foundation model and then key features and so on. So I personally find this pretty helpful. 

And then we can kind of go back and forth. And as I'm reading through the abstract and the introduction, et cetera, I am asking questions of the LLM and it's kind of like making it easier for me to understand the paper. Another way that I like to use this functionality extensively is when I'm reading books. 

It is rarely ever the case anymore that I read books just by myself. I always involve an LLM to help me read a book. So a good example of that recently is The Wealth of Nations, which I was reading recently. 

And it is a book from 1776 written by Adam Smith. And it's kind of like the foundation of classical economics. And it's a really good book. 

And it's kind of just very interesting to me that it was written so long ago, but it has a lot of modern day kind of like, it's just got a lot of insights that I think are very timely even today. So the way I read books now as an example is you basically pull up the book and you have to get access to like the raw content of that information. In the case of Wealth of Nations, this is easy because it is from 1776. 

So you can just find it on Wealth Project Gutenberg as an example. And then basically find the chapter that you are currently reading. So as an example, let's read this chapter from book one. 

And this chapter I was reading recently, and it kind of goes into the division of labor and how it is limited by the extent of the market. Roughly speaking, if your market is very small, then people can't specialize. And a specialization is what is basically huge. 

Specialization is extremely important for wealth creation because you can have experts who specialize in their simple little task. But you can only do that at scale because without the scale, you don't have a large enough market to sell to your specialization. So what we do is we copy paste this book, this chapter at least. 

This is how I like to do it. We go to say Claude. And we say something like, we are reading the Wealth of Nations. 

Now remember, Claude has knowledge of the Wealth of Nations, but probably doesn't remember exactly the content of this chapter. So it wouldn't make sense to ask Claude questions about this chapter directly because he probably doesn't remember what the chapter is about. But we can remind Claude by loading this into the context window. 

So we're reading the Wealth of Nations. Please summarize this chapter to start. And then what I do here is I copy paste. 

Now in Claude, when you copy paste, they don't actually show all the text inside the text box. They create a little text attachment when it is over some size. And so we can click enter. 

And we just kind of like start off. Usually I like to start off with a summary of what this chapter is about just so I have a rough idea. And then I go in and I start reading the chapter. 

And if at any point we have any questions, just come in and just ask our question. And I find that basically going hand in hand with LLMs dramatically increases my retention, my understanding of these chapters. And I find that this is especially the case when you're reading, for example, documents from other fields, like for example, biology, or for example, documents from a long time ago, like 1776, where you sort of need a little bit of help of even understanding the basics of the language.

Or for example, I would feel a lot more courage approaching a very old text that is outside of my area of expertise. Maybe I'm reading Shakespeare, or I'm reading things like that. I feel like LLMs make a lot of reading very dramatically more accessible than it used to be before, because you're not just right away confused, you can actually kind of go through it and figure it out together with the LLM in hand. 

So I use this extensively. And I think it's extremely helpful. I'm not aware of tools, unfortunately, that make this very easy for you. 

Today, I do this clunky back and forth. So literally, I will find the book somewhere. And I will copy paste stuff around.

And I'm going back and forth. And it's extremely awkward and clunky. And unfortunately, I'm not aware of a tool that makes this very easy for you. 

But obviously, what you want is as you're reading a book, you just want to highlight the passage and ask questions about it. This currently, as far as I know, does not exist. But this is extremely helpful. 

I encourage you to experiment with it. And don't read books alone. Okay, the next very powerful tool that I now want to turn to is the use of a Python interpreter, or basically giving the ability to the LLM to use and write computer programs. 

So instead of the LLM giving you an answer directly, it has the ability now to write a computer program, and to emit special tokens that the ChachiPT application recognizes as, hey, this is not for the human. This is basically saying that whatever I output it here is actually a computer program, please go off and run it and give me the result of running that computer program. So it is the integration of the language model with a programming language here, like Python. 

So this is extremely powerful. Let's see the simplest example of where this would be used and what this would look like. So if I go to ChachiPT and I give it some kind of a multiplication problem, let's say 30 times 9 or something like that.

(该文件长度超过30分钟。 在TurboScribe.ai点击升级到无限，以转录长达10小时的文件。)


(转录由TurboScribe.ai完成。升级到无限以移除此消息。)

Then this is a fairly simple multiplication, and you and I can probably do something like this in our head, right? Like 30 times nine, you can just come up with the result of 270, right? So let's see what happens. Okay, so LLM did exactly what I just did. It calculated the result of the multiplication to be 270, but it's actually not really doing math.

It's actually more like almost memory work, but it's easy enough to do in your head. So there was no tool use involved here. All that happened here was just the zip file doing next token prediction and gave the correct result here in its head.

The problem now is what if we want something more complicated? So what is this times this? And now, of course, this, if I asked you to calculate this, you would give up instantly because you know that you can't possibly do this in your head, and you would be looking for a calculator. And that's exactly what the LLM does now too. And OpenAI has trained ChatGPT to recognize problems that it cannot do in its head and to rely on tools instead.

So what I expect ChatGPT to do for this kind of a query is to turn to tool use. So let's see what it looks like. Okay, there we go.

So what's opened up here is what's called the Python interpreter. And Python is basically a little programming language. And instead of the LLM telling you directly what the result is, the LLM writes a program and then not shown here are special tokens that tell the ChatGPT application to please run the program.

And then the LLM pauses execution. Instead, the Python program runs, creates a result, and then passes this result back to the language model as text. And the language model takes over and tells you that the result of this is that.

So this is tool use, incredibly powerful. And OpenAI has trained ChatGPT to kind of like know in what situations to lean on tools. And they've taught it to do that by example.

So human labelers are involved in curating data sets that kind of tell the model by example in what kinds of situations it should lean on tools and how. But basically, we have a Python interpreter, and this is just an example of multiplication. But this is significantly more powerful.

So let's see what we can actually do inside programming languages. Before we move on, I just wanted to make the point that unfortunately you have to kind of keep track of which LLMs that you're talking to have different kinds of tools available to them. Because different LLMs might not have all the same tools.

And in particular, LLMs that do not have access to the Python interpreter or a programming language or are unwilling to use it might not give you correct results in some of these harder problems. So as an example, here we saw that ChatGPT correctly used a programming language and didn't do this in its head. Grok3 actually, I believe, does not have access to a programming language like a Python interpreter.

And here, it actually does this in its head and gets remarkably close. But if you actually look closely at it, it gets it wrong. This should be one, two, zero instead of zero, six, zero.

So Grok3 will just hallucinate through this multiplication and do it in its head and get it wrong, but actually remarkably close. Then I tried Clod and Clod actually wrote, in this case, not Python code, but it wrote JavaScript code. But JavaScript is also a programming language and gets the correct result.

Then I came to Gemini and I asked 2.0 Pro and Gemini did not seem to be using any tools. There's no indication of that. And yet it gave me what I think is the correct result, which actually kind of surprised me.

So Gemini, I think, actually calculated this in its head correctly. And the way we can tell that this is, which is kind of incredible, the way we can tell that it's not using tools is we can just try something harder. What is, we have to make it harder for it.

Okay, so it gives us some result and then I can use my calculator here and it's wrong. So this is using my MacBook Pro calculator. And two, it's not correct, but it's remarkably close, but it's not correct.

But it will just hallucinate the answer. So I guess like my point is, unfortunately, the state of the LLMs right now is such that different LLMs have different tools available to them and you kind of have to keep track of it. And if they don't have the tools available, they'll just do their best, which means that they might hallucinate a result for you.

So that's something to look out for. Okay, so one practical setting where this can be quite powerful is what's called ChatGPT Advanced Data Analysis. And as far as I know, this is quite unique to ChatGPT itself.

And it basically gets ChatGPT to be kind of like a junior data analyst who you can kind of collaborate with. So let me show you a concrete example without going into the full detail. So first we need to get some data that we can analyze and plot and chart, et cetera.

So here in this case, I said, let's research open AI evaluation as an example. And I explicitly asked ChatGPT to use the search tool because I know that under the hood, such a thing exists. And I don't want it to be hallucinating data to me.

I want it to actually look it up and back it up and create a table where each year we have the evaluation. So these are the open AI evaluations over time. Notice how in 2015, it's not applicable.

So the valuation is like unknown. Then I said, now plot this, use log scale for y-axis. And so this is where this gets powerful.

ChatGPT goes off and writes a program that plots the data over here. So it created a little figure for us and it sort of ran it and showed it to us. So this can be quite nice and valuable because it's very easy way to basically collect data, upload data in a spreadsheet, visualize it, et cetera.

I will note some of the things here. So as an example, notice that we had NA for 2015, but ChatGPT when it was writing the code, and again, I would always encourage you to scrutinize the code, it put in 0.1 for 2015. And so basically it implicitly assumed that it made the assumption here in code that the valuation at 2015 was 100 million and because it put in 0.1. And it's kind of like did it without telling us.

So it's a little bit sneaky and that's why you kind of have to pay attention a little bit to the code. So I'm familiar with the code and I always read it, but I think I would be hesitant to potentially recommend the use of these tools if people aren't able to like read it and verify it a little bit for themselves. Now, fit a trendline and extrapolate until the year 2030.

Mark the expected valuation in 2030. So it went off and it basically did a linear fit and it's using SciPy's curve fit. And it did this and came up with a plot and it told me that the valuation based on the trend in 2030 is approximately 1.7 trillion, which sounds amazing except here I became suspicious because I see that Chachapiti is telling me it's 1.7 trillion, but when I look here at 2030, it's printing 20271.7b. So it's extrapolation when it's printing the variable is inconsistent with 1.7 trillion.

This makes it look like that valuation should be about 20 trillion. And so that's what I said, print this variable directly by itself, what is it? And then it sort of like rewrote the code and gave me the variable itself. And as we see in the label here, it is indeed 20271.7b, et cetera.

So in 2030, the true exponential trend extrapolation would be a valuation of 20 trillion. So I was like, I was trying to confront Chachapiti and I was like, you lied to me, right? And it's like, yeah, sorry, I messed up. So I guess I like this example because number one, it shows the power of the tool and that it can create these figures for you.

And it's very nice. But I think number two, it shows the trickiness of it where, for example, here it made an implicit assumption and here it actually told me something. It told me just the wrong, it hallucinated 1.7 trillion.

So again, it is kind of like a very, very junior data analyst. It's amazing that it can plot figures, but you have to kind of still know what this code is doing and you have to be careful and scrutinize it and make sure that you are really watching very closely because your junior analyst is a little bit absent-minded and not quite right all the time. So really powerful, but also be careful with this.

I won't go into full details of advanced data analysis, but there were many videos made on this topic. So if you would like to use some of this in your work, then I encourage you to look at some of these videos. I'm not going to go into the full detail.

So a lot of promise, but be careful. Okay, so I've introduced you to Chats GPT and advanced data analysis, which is one powerful way to basically have LLMs interact with code and add some UI elements like showing a figures and things like that. I would now like to introduce you to one more related tool and that is specific to Cloud and it's called Artifacts.

So let me show you by example what this is. So you're having a conversation with Cloud and I'm asking generate 20 flashcards from the following text. And for the text itself, I just came to the Adam Smith Wikipedia page, for example, and I copy pasted this introduction here.

So I copy pasted this here and asked for flashcards and Cloud responds with 20 flashcards. So for example, when was Adam Smith baptized on June 16th, et cetera? When did he die? What was his nationality, et cetera? So once we have the flashcards, we actually want to practice these flashcards. And so this is where I continue to use the conversation and I say, now use the Artifacts feature to write a flashcards app to test these flashcards.

And so Cloud goes off and writes code for an app that basically formats all of this into flashcards. And that looks like this. So what Cloud wrote specifically was this code here.

So it uses a React library and then basically creates all these components. It hard codes the Q&A into this app and then all the other functionality of it. And then the Cloud interface basically is able to load these React components directly in your browser.

And so you end up with an app. So when was Adam Smith baptized? And you can click to reveal the answer. And then you can say whether you got it correct or not.

When did he die? What was his nationality, et cetera. So you can imagine doing this and then maybe we can reset the progress or shuffle the cards, et cetera. So what happened here is that Cloud wrote us a super duper custom app just for us right here.

And typically what we're used to is some software engineers write apps, they make them available, and then they give you maybe some way to customize them or maybe to upload flashcards. Like for example, the Anki app, you can import flashcards and all this kind of stuff. This is a very different paradigm because in this paradigm, Cloud just writes the app just for you and deploys it here in your browser.

Now, keep in mind that a lot of apps that you will find on the internet, they have entire backends, et cetera. There's none of that here. There's no database or anything like that, but these are like local apps that can run in your browser and they can get fairly sophisticated and useful in some cases.

So that's Cloud Artifacts. Now, to be honest, I'm not actually a daily user of Artifacts. I use it once in a while.

I do know that a large number of people are experimenting with it and you can find a lot of Artifacts showcases because they're easy to share. So these are a lot of things that people have developed, various timers and games and things like that. But the one use case that I did find very useful in my own work is basically the use of diagrams, diagram generation.

So as an example, let's go back to the book chapter of Adam Smith that we were looking at. What I do sometimes is, we are reading The Wealth of Nations by Adam Smith. I'm attaching chapter three and book one.

Please create a conceptual diagram of this chapter. And when Cloud hears conceptual diagram of this chapter, very often it will write a code that looks like this. And if you're not familiar with this, this is using the mermaid library to basically create or define a graph.

And then this is plotting that mermaid diagram. And so Cloud analyzed the chapter and figures out that, okay, the key principle that's being communicated here is as follows, that basically division of labor is related to the extent of the market, the size of it. And then these are the pieces of the chapter.

So there's the comparative example of trade and how much easier it is to do on land and on water and the specific example that's used. And that geographic factors actually make a huge difference here. And then the comparison of land transport versus water transport and how much easier water transport is.

And then here we have some early civilizations that have all benefited from basically the availability of water transport and have flourished as a result of it because they support specialization. So if you're a conceptual kind of like visual thinker, and I think I'm a little bit like that as well, I like to lay out information as like a tree like this. And it helps me remember what that chapter is about very easily.

And I just really enjoy these diagrams and like kind of getting a sense of like, okay, what is the layout of the argument? How is it arranged spatially? And so on. And so if you're like me, then you will definitely enjoy this. And you can make diagrams of anything, of books, of chapters, of source codes, of anything really.

And so I specifically find this fairly useful. Okay, so I've shown you that LLMs are quite good at writing code. So not only can they emit code, but a lot of the apps like ChatGPT and Cloud and so on have started to like partially run that code in the browser.

So ChatGPT will create figures and show them and Cloud Artifacts will actually like integrate your React component and allow you to use it right there in line in the browser. Now, actually majority of my time personally and professionally is spent writing code. But I don't actually go to ChatGPT and ask for snippets of code because that's way too slow.

Like ChatGPT just doesn't have the context to work with me professionally to create code. And the same goes for all the other LLMs. So instead of using features of these LLMs in a web browser, I use a specific app.

And I think a lot of people in the industry do as well. And this can be multiple apps by now, VS Code, Windsurf, Cursor, et cetera. So I like to use Cursor currently.

And this is a separate app you can get for your, for example, MacBook. And it works with the files on your file system. So this is not a web inter, this is not some kind of a webpage you go to.

This is a program you download and it references the files you have on your computer. And then it works with those files and edits them with you. So the way this looks is as follows.

Here I have a simple example of a React app that I built over a few minutes with Cursor. And under the hood, Cursor is using Cloud 3.7 Sonnet. So under the hood, it is calling the API of Anthropic and asking Cloud to do all of this stuff.

But I don't have to manually go to Cloud and copy paste chunks of code around. This program does that for me and has all of the context of the files in the directory and all this kind of stuff. So the app that I developed here is a very simple tic-tac-toe as an example.

And Cloud wrote this in a few, probably a minute. And we can just play. X can win.

Or we can tie. Oh, wait, sorry, I accidentally won. You can also tie.

And I'd just like to show you briefly, this is a whole separate video of how you would use Cursor to be efficient. I just want you to have a sense that I started from a completely new project and I asked the Composer app here, as it's called, the Composer feature, to basically set up a new React repository, delete a lot of the boilerplate, please make a simple tic-tac-toe app. And all of this stuff was done by Cursor.

I didn't actually really do anything except for write five sentences. And then it changed everything and wrote all the CSS, JavaScript, et cetera. And then I'm running it here and hosting it locally and interacting with it in my browser.

So that's Cursor. It has the context of your apps and it's using Cloud remotely through an API without having to access the webpage. And a lot of people, I think, develop in this way at this time.

And these tools have become more and more elaborate. So in the beginning, for example, you could only like say change, like, oh, Control-K, please change this line of code to do this or that. And then after that, there was a Control-L, Command-L, which is, oh, explain this chunk of code.

And you can see that there's gonna be an LLM explaining this chunk of code. And what's happening under the hood is it's calling the same API that you would have access to if you actually did enter here. But this program has access to all the files.

So it has all the context. And now what we're up to is not Command-K and Command-L. We're now up to Command-I, which is this tool called Composer.

And especially with the new agent integration, the Composer is like an autonomous agent on your code base. It will execute commands. It will change all the files as it needs to.

It can edit across multiple files. And so you're mostly just sitting back and you're giving commands. And the name for this is called Vibe Coding, a name with that I think I probably minted.

And Vibe Coding just refers to letting, giving in, giving control to Composer and just telling it what to do and hoping that it works. Now, worst comes to worst, you can always fall back to the good old programming because we have all the files here. We can go over all the CSS and we can inspect everything.

And if you're a programmer, then in principle you can change this arbitrarily. But now you have a very helpful system that can do a lot of the low-level programming for you. So let's take it for a spin briefly.

Let's say that when either X or O wins, I want confetti or something. And let's just see what it comes up with. Okay, I'll add a confetti effect when a player wins the game.

It wants me to run React Confetti, which apparently is a library that I didn't know about. So we'll just say, okay. It installed it, and now it's going to update the app.

So it's updating app.tsx, the TypeScript file, to add the confetti effect when a player wins. And it's currently writing the code, so it's generating. And we should see it in a bit.

Okay, so it basically added this chunk of code. And a chunk of code here, and a chunk of code here. And then we'll also add some additional styling to make the winning cells stand out.

Okay, still generating. Okay, and it's adding some CSS for the winning cells. So honestly, I'm not keeping full track of this.

It imported React Confetti. This all seems pretty straightforward and reasonable, but I'd have to actually really dig in. Okay, it wants to add a sound effect when a player wins, which is pretty ambitious, I think.

I'm not actually 100% sure how it's going to do that, because I don't know how it gains access to a sound file like that. I don't know where it's going to get the sound file from. But every time it saves a file, we actually are deploying it.

So we can actually try to refresh and just see what we have right now. So, oh, so it added a new effect. You see how it kind of like fades in, which is kind of cool.

And now we'll win. Whoa, okay. Didn't actually expect that to work.

This is really elaborate now. Let's play again. Whoa.

Okay. Oh, I see. So it actually paused and it's waiting for me.

So it wants me to confirm the command. So make public sounds. I had to confirm it explicitly.

Let's create a simple audio component to play Victory Sound. Sound slash Victory MP3. The problem with this will be the Victory.mp3 doesn't exist.

So I wonder what it's going to do. It's downloading it. It wants to download it from somewhere.

Let's just go along with it. Let's add a fallback in case the sound file doesn't exist. In this case, it actually does exist.

And yep, we can get add and we can basically create a git commit out of this. Okay, so the composer thinks that it is done. So let's try to take it for a spin.

Okay. So yeah, pretty impressive. I don't actually know where it got the sound file from.

I don't know where this URL comes from, but maybe this just appears in a lot of repositories and sort of cloud kind of like knows about it. But I'm pretty happy with this. So we can accept all and that's it.

And then as you can get a sense of, we could continue developing this app and worst comes to worst, if we can't debug anything, we can always fall back to a standard programming instead of a vibe coding. Okay, so now I would like to switch gears again. Everything we've talked about so far had to do with interacting with the model via text.

So we type text in and it gives us text back. What I'd like to talk about now is to talk about different modalities. That means we want to interact with these models in more native human formats.

So I want to speak to it and I want it to speak back to me and I want to give images or videos to it and vice versa. I want it to generate images and videos back. So it needs to handle the modalities of speech and audio and also of images and video.

So the first thing I want to cover is how can you very easily just talk to these models? So I would say roughly in my own use, 50% of the time I type stuff out on the keyboard and 50% of the time I'm actually too lazy to do that and I just prefer to speak to the model. And when I'm on mobile, on my phone, that's even more pronounced. So probably 80% of my queries are just speech because I'm too lazy to type it out on the phone.

Now on the phone, things are a little bit easy. So right now the Chatshubpt app looks like this. The first thing I want to cover is there are actually like two voice modes.

You see how there's a little microphone and then here there's like a little audio icon. These are two different modes and I will cover both of them. First, the audio icon, sorry, the microphone icon here is what will allow the app to listen to your voice and then transcribe it into text.

So you don't have to type out the text. It will take your audio and convert it into text. So on the app, it's very easy and I do this all the time.

As you open the app, create a new conversation and I just hit the button and why is the sky blue? Is it because it's reflecting the ocean or yeah, why is that? And I just click okay. And I don't know if this will come out but it basically converted my audio to text and I can just hit go and then I get a response. So that's pretty easy.

Now on desktop, things get a little bit more complicated for the following reason. When we're in the desktop app, you see how we have the audio icon and it says use voice mode. We'll cover that in a second but there's no microphone icon.

So I can't just speak to it and have it transcribed to text inside this app. So what I use all the time on my MacBook is I basically fall back on some of these apps that allow you that functionality but it's not specific to ChatGPT. It is a system wide functionality of taking your audio and transcribing it into text.

So some of the apps that people seem to be using are SuperWhisper, WhisperFlow, MacWhisper, et cetera. The one I'm currently using is called SuperWhisper and I would say it's quite good. So the way this looks is you download the app, you install it on your MacBook and then it's always ready to listen to you.

So you can bind a key that you wanna use for that. So for example, I use F5. So whenever I press F5, it will listen to me then I can say stuff and then I press F5 again and it will transcribe it into text.

So let me show you. I'll press F5. I have a question.

Why is the sky blue? Is it because it's reflecting the ocean? Okay, right there. Enter. I didn't have to type anything.

So I would say a lot of my queries, probably about half are like this because I don't want to actually type this out. Now, many of the queries will actually require me to say product names or specific like library names or like various things like that that don't often transcribe very well. In those cases, I will type it out to make sure it's correct.

But in very simple day-to-day use, very often I am able to just speak to the model. So, and then it will transcribe it correctly. So that's basically on the input side.

Now on the output side, usually with an app, you will have the option to read it back to you. So what that does is it will take this text and it will pass it to a model that does the inverse of taking text to speech. And in ChachiPT, there's this icon here that says Read Aloud.

So we can press it. No, the sky is not blue because it reflects the ocean. That's a common myth.

The real reason the sky is blue is due to Rayleigh scattering. Okay, so I'll stop it. So different apps like ChachiPT or Cloud or Gemini or whatever you are using may or may not have this functionality, but it's something you can definitely look for.

When you have the input be system-wide, you can, of course, turn speech into text in any of the apps. But for reading it back to you, different apps may or may not have the option. And, or you could consider downloading speech-to-text, sorry, a text-to-speech app that is system-wide like these ones and have it read out loud.

So those are the options available to you and something I wanted to mention. And basically the big takeaway here is don't type stuff out, use voice. It works quite well.

And I use this pervasively. And I would say roughly half of my queries, probably a bit more are just audio because I'm lazy and it's just so much faster. Okay, but what we've talked about so far is what I would describe as fake audio.

And it's fake audio because we're still interacting with the model via text. We're just making it faster because we're basically using either a speech-to-text or a text-to-speech model to pre-process from audio to text and from text to audio. So it's not really directly done inside the language model.

So, however, we do have the technology now to actually do this actually as true audio handled inside the language model. So what actually is being processed here was text tokens, if you remember. So what you can do is you can truncate different modalities like audio in a similar way as you would truncate text into tokens.

So typically what's done is you basically break down the audio into a spectrogram to see all the different frequencies present in the audio. And you go in little windows and you basically quantize them into tokens. So you can have a vocabulary of 100,000 possible little audio chunks.

And then you actually train the model with these audio chunks so that it can actually understand those little pieces of audio. And this gives the model a lot of capabilities that you would never get with this fake audio as we've talked about so far. And that is what this other button here is about.

This is what I call true audio, but sometimes people will call it by different names. So as an example, when you go to chat GPT pricing, you have to look for this thing called advanced voice mode. So advanced voice mode is referring to true audio.

What that means is that the voice is handled natively inside the language model. The model can understand audio chunks and predict audio chunks. So it can hear and speak directly in audio.

There's no text involved whatsoever. So it's quite magical. It's really nice.

And let me take it for a spin. Okay, so here's what this will look like. Hey, how's it going? Hey, Andre, I'm doing great.

How about you? I'm great. I'm wondering why the sky is blue. Is it because it's reflecting the ocean? Glad to hear you're great.

Actually, the sky is blue because of a phenomenon called Rayleigh scattering. Okay, okay. Instead of speaking to me normally, can you speak in the voice of Yoda so that you...

(该文件长度超过30分钟。 在TurboScribe.ai点击升级到无限，以转录长达10小时的文件。)