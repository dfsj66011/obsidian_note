
Author：Karina Nguyen (Claude Instant @ Anthropic)



我叫卡琳娜。最近我一直在研究 Claude，它是由 Anthropic 训练的大型语言模型。最近我主要致力于减少幻觉现象，如何让 Claude 自我修正答案，以及为 Claude2 发布准备的许多其他功能。因此，今天我要谈谈针对任务优化的提示工程写作原则，并希望能为想要使用 Claude API 的你提供一些指导，分享我认为最有效的最佳实践和技巧。

> [!NOTE]
> 为什么提示这么难？为什么它有效？
> 1. 这些模型根据前面的词来估计每个后续词的概率。精心设计的提示可以提高生成所需和准确词/短语的概率。
> 2. 由于注意力机制，模型可以专注于输入文本的特定部分。有效的提示确保注意力被适当地引导以获得期望的输出。在提示中加入特定任务的关键词和上下文可以激活模型知识的相关部分，从而提高性能。
> 3. 高效的提示可以在不需要额外计算资源或模型重新训练的情况下带来更好的结果。

首先，我想谈谈为什么提示编写如此困难。要理解这一点，我们首先需要明白提示究竟是什么。因此，这些模型会根据前面的单词来估计每个后续单词的概率。所以，精心设计的提示可以在某种程度上增加生成理想且准确短语的概率。由于大型语言模型中的注意力机制，模型可以专注于输入文本的特定部分。因此，有效的提示能确保注意力被正确引导以获得期望的输出。在提示中加入任务特定的关键词、上下文和示例非常重要，这样可以激活模型内部知识的相关部分。最后，提示能带来更好的结果，因为无需额外的计算资源或模型训练。

> [!NOTE]
> 为什么提示这么难？ 人类不擅长提示的原因在于：
> 
> 1. 他们知道自己想要什么，但不知道如何从模型中获得最佳表现
> 2. 他们模糊地知道自己想要什么，因此不知道如何向模型解释这一点
> 3. 他们根本不知道自己想要什么

这样你就可以利用推理时间测试计算来实现这一点。那么为什么提示这么难呢？根据我与客户和开发者的交流，我认为提示之所以困难，主要有三个原因：

* 首先，人们——人类——知道自己想要什么，但他们不知道如何从模型中获得最佳性能。我认为这就是我们今天要重点讨论的内容。
* 第二个原因是，他们隐约知道自己想要什么，但不知道如何向模型做出最佳解释。因此，模型对人类想要从任务中获得什么感到困惑。
* 第三个原因是他们不知道自己想要什么，所以人类也不知道自己想要什么，这相当糟糕。模型也很难理解。

> [!NOTE]
> 他们隐约知道自己想要什么，却不知道如何向模型解释清楚 + 他们其实并不清楚自己想要什么
> 
> 1. 提供大量包含边界案例的示例
> 2. 尝试用给五岁小孩讲解的方式来解释
> 3. 不断迭代、优化、再优化

所以基本策略是，如果你发现自己对任务只有模糊的了解，那就提供一堆例子。这个模型将擅长根据示例推断出你想要做什么。这些示例可能需要多样化，并涵盖各种典型情况。试着用非常简单的方式解释，就像你要向一个五岁小孩解释那样。我认为，我发现的是，你必须能够反复尝试，并投入大量时间在提示上。


> [!NOTE]
> 写作原则  
> 
> 过去，我对"提示词"这个词的主要体验仅限于创意写作课程。我们常常忘记，给语言模型提供提示本身就是一种创意写作行为。我看到很多人因为提示词"就是不起作用"而感到恼火。在大多数情况下，我认为这只是意味着人们缺乏原创性和想象力，无法创造性地思考如何让它发挥作用。


根据我的经验，比如作为一名研究工程师，我大部分时间都在与人们结对合作，共同完善提示。过去，我一直以为我对“提示”这个词的主要体验仅限于创意写作课程。我毕业于伯克利大学，上过一些创意写作课程。我们经常做一些类似提示练习的作业。所以我们常常忘记，给语言模型提供提示其实也是一种创意写作行为。我看到人们会感到恼火，为什么提示就是不起作用。但在大多数情况下，我认为这只是意味着人们缺乏某种原创性或创造力，无法想出新颖的方法让它发挥作用。

> [!NOTE]
> 提示模型已成为每位研究人员和工程师日常参与的一种写作形式。这种写作需要形成假设、验证假设，并根据新的见解进行修订。

我最近写了一篇关于写作文化的博客文章，其中我提到的一个观点是，提示正在成为一种新的写作形式，对于每天从事研究工作的工程师和科学家来说都是如此。这种写作需要形成假设。所以你必须问模型，它能否做到这一点。而你想要对此进行测试。该模型能否自我纠正其回答？是或否。然后你开始尝试形成假设。

接下来你要验证关于这个模型的一些假设。随着不断迭代，你会获得新的见解，比如发现模型在某个特定方面表现不错，但在完成这项任务的其他方面则不太理想。这样你就能更清楚地了解它的优势和劣势所在。

> [!NOTE]
> 写作原则 
> 
> 目标是撰写能够清晰传达任务目标的提示词，同时提供恰到好处的约束和引导，使模型能够生成相关且高质量的输出。
> 
> 清晰性 - 在提示中使用简单、明确的语言。避免可能混淆模型的复杂句法或模糊措辞。  
> 简洁性 - 保持提示简短且重点突出。仅包含模型所需的关键信息，删除不必要的词语。  
> 连贯性 - 逻辑清晰地构建提示，开头提供上下文，结尾明确任务。  
> 一致性 - 对于特定模型，保持提示的格式、术语和语气一致。

我想先从写作的基本原则说起，因为归根结底，提示词设计本质上就是写作。我们的目标是写出能清晰传达任务目标的提示词，同时提供恰到好处的约束和引导，使模型能够生成高质量且相关的输出。

因此，我发现有大约四到六种写作指导原则特别有效，尤其是在使用 Claude 时。或许需要先说明一下 Claude 与 GPT 的不同之处。我认为使用 Claude 时，你几乎需要把它当作另一个人来对待。所以你必须像给五岁小孩解释事情那样，或者说你必须详细说明，我会分享更多关于如何做到这一点的例子。但我认为这是 GPT 模型与 Claude 的一个显著区别。

第一个原则就是清晰。在提示中使用简单且明确的语言。避免使用可能让模型困惑的复杂句法或模糊表达。

其次是简洁性。保持提示简短且重点突出。只包含模型所需的关键信息。

第三是连贯性。在提示的开头提供上下文，并在结尾明确任务，使结构合乎逻辑。保持格式的一致性。如果使用XML标签，请在提示中始终如一地使用。如果你使用某些术语，最好不要像那样基本上套用分销模式。所以要保持一致。方向应提供类型、长度、风格或任何类似的指导方针。

指导模型回应的方向。通过示例、来源来具体化提示，让模型引用自身内容，例如当上下文中有长文档时，可以引用文档内容来支持论点。或者帮助模型利用搜索结果或其他支持性的上下文信息来构建论证依据。

通过多样化的边缘案例示例进行互动。对未来提示非常有用。现在，我将介绍一些我认为有趣的任务，并看看如何在具体任务中运用 Claude。

所以你可以说像詹姆斯·邦德女郎那样，你可以选择礼服风格，某种程度上你能得到类似詹姆斯·邦德风格的礼服效果。或者你也可以选择未来感空灵套装，这更像是基于氛围的搜索，你可以去店铺里看看。哦对了。

Clip是由OpenAI训练的一种对比语言到图像的模型。它是开源的，基本上他们提供了文本和图像的嵌入表示，这样你就可以——它的工作原理是这样的：你可以嵌入图像和文本结果，然后你可以通过余弦相似度来根据用户查询在你的数据库中找到最相似的项目。不知道这样说是否清楚。

有问题请告诉我。我想你可以看看 Clip。是的。

如果你感兴趣的话。我当时在想，好吧，我该如何在这个项目中使用Claude，根据用户的请求来筛选相关推荐，这就是任务。所以某种程度上，你有用户的输入。

比如说，穿一件艾玛·张伯伦风格的西装外套，就像《了不起的盖茨比》电影里那样，打造一套未来感十足的造型去参加Met Gala。另一方面，你有一个图像转文字的数据库，里面有图片和它们的标签，这些标签可以由原始来源提供，也可以用多模态模型根据图片生成标签。所以Claude的任务就是根据图片的标签来筛选，判断这件单品是否相关。

我应该向用户推荐这个吗？这个准确吗？是否符合用户的喜好？我可以个性化推荐吗？

因此，如果你能看一下这个非常简单的提示策划策略，你就可以像零样本一样使用它，比如我需要你判断这个项目是否与用户查询相关。这是用户查询。这是项目描述。

该商品是否相关，或者是否应根据用户的查询推荐给用户。请回答是或否。请在答案标签中写出答案。

让我们来剖析一下。首先，Claude真的非常喜欢XML标签，简直爱不释手。我认为这一点是人们最容易忽视的首要特征，而非错误。

他们不喜欢把任何东西放在XML标签里，所以他们的性能并不是很好。而我确实很喜欢XML标签，所以你应该把所有东西都放在XML标签里，并且使用XML标签时要保持一致。那么什么是用户查询项呢？你可以描述得非常详细。

稍后我可以分享更多例子。这里，你喜欢与Claude互动的方式，就像你可以在这里看到语言一样。我需要你来判断这个项目是否相关。

感觉就像在和真人交谈一样。这个XML功能最近才在Entropiq官网上线，大概也就一两个月前的事。这是你们特意训练的功能，还是在预览时偶然发现的？我想应该是...我们当初尝试过...XML格式是我们最早进行微调的格式之一。

后来我们发现，客户需要支持Markdown格式，或者像Claude需要使用JSON格式。所以我们基本上是从客户那里学习到的，但最初它用的是XML格式。这主要是因为微调的原因，还是因为你们的训练网站里有很多XML内容？我觉得两者都有关系。

你不需要关闭XML标签。关闭XML标签？是啊，我不太清楚。哦对，抱歉。

我犯了个错误，是的。我……嗯，XML标签的一个好处就是它真的很容易提取，对吧？比如里面的字符串。而Claude在这方面挺擅长的……我可以告诉它不要在XML标签里放任何东西。

有时候Claude会像这样巴拉巴拉说一堆信息。但如果你要求它把答案写在特定的标签里，它就不会添加任何额外内容。这真是语言模型最烦人的地方之一。

这是我通过Claude.ai界面得到的结果。你看，这件商品相关吗？结果显示不相关。

这个项目相关吗？是的。但我不认为这是一个100%完美的系统。所以它非常像零样本。

基本上你可以进行迭代。我们会在这方面尝试更多的迭代。策略二就是当你让模型花点时间思考某个项目是否相关时。

根据上述标准进行思维标签化。你有点像是在让模型通过推理进行更深入的思考。这基本上就是所谓的思维链。

您还可以添加类似的标准。因此，作为您评价的一部分，请考虑以下标准。如果您想引导模型思考物品是否符合用户请求的具体属性？比如帮助模型思考向用户推荐物品意味着什么？物品是否符合季节？或者用户查询中的天气条件是否匹配？例如，在夏季不应推荐冬季外套。

所以在标准中，你可以提供更多的例子，更详细的例子。另一件你需要反复思考的事情是，不仅仅是给出"是"或"否"的答案。而是可以根据你的评分标准来决定是否推荐该物品。

1表示最不推荐，10表示强烈推荐。并将最终分数放在score标签中。那么它是如何运作的呢？比如这里用户查询詹姆斯·邦德西装外套单品，我是从某个品牌那里拿来的。

克劳德想用“思想标签”来开始。总的来说，这看起来非常相关。最终得分是九分。

再举一个例子。我想穿成《了不起的盖茨比》电影里的风格。这件编织绳短款马甲就是不错的选择。

批评的核心在于，根据查询中的上下文线索，该物品并不符合用户的需求。它似乎并不符合《了不起的盖茨比》电影的特质。它试图给出一些理由，但显得牵强。

所以比分是二。你可以再详细一点。这就像是非常简单的迭代。

你们有什么问题吗？是的。我看到一件有趣的事情。我是说那些被括起来的XML标签。

我的朋友不是以英语为母语的人。他的提示总是用那种有点搞笑的英语写的。但他组织得非常好，尽管英语不正确，效果却非常好。

对。为什么？为什么会这样？我认为这些模型在语言间的知识迁移方面表现得相当出色，或者说它们能很好地推断用户的意图。

是的，我也没有一个非常明确的答案。所以那个有一点语法错误的文本看起来已经足够接近了。

在这两种情况下，真实答案的概率都很接近。没错，就像是我不知道一样。

在这个具体例子中，我很好奇你是否发现评分存在任何偏差。换句话说，如果你查看评分分布的话，它会是一个正态分布吗？是的。这是个有趣的问题。

就像这是我们在研究环境中提出的一个问题。就像我们试图理解的一件事。就像我们有一个名为“社会影响”的研究小组。

我们现在正在努力理解的一件事是，当你总结新闻文章并试图评估其偏见时，偏见的分布情况是怎样的。我觉得这是一个活跃的研究领域。是的，我认为这取决于具体的任务。

我真的没有测试过这个。确切地说，就是昨天的事。提示。

好的。第二个任务。众所周知，Claude 以 10 万上下文长度著称。

《了不起的盖茨比》的整本书可以放入上下文中，你可以让模型总结这本书。比如基于大量上下文提出一些任务。这基本上是一个时间测试计算的事情。

因此，这是一个很长的上下文。让我想想。你可以用不同的方式使用长上下文。

一种方法是你可以放入多个文档，并尝试总结或基于这些文档检索信息。另一种利用长上下文的方式是准备大量、庞大的少样本提示。正如你所知，思维链技术依赖于明确的推理过程。

忠实地反映模型的实际推理过程。然而，在我们最近的一篇论文中发现，情况并非总是如此。简单来说，这意味着当你要求模型进行链式思考时，它可能并不一定会通过这种思考链来生成最终答案。

它可能会直接忽略掉，或者根本不当回事。所以我们说它基本上是不靠谱的，不太可靠。

因此，我们在本文中提出，基于分解的方法实际上可以在特定的问答任务上实现强劲的性能。有时在提高忠实度的同时，接近思维链的表现。大家有什么问题吗？忠实度，是的，是指对提示的忠实度。

是的，让我来解释一下什么是分解。这是论文中的图表。我们采用了三种方法。

首先是一种思维链方法，就像这里有一个问题：你能钻进袋鼠的育儿袋里吗？有两个选项。A，可以。B，不。思维链提示会说让我们一步步思考。给出推理过程。

人类根据上述内容提出了后续问题。最可能的单一答案选项是什么？模型说的是正确答案是B，对吧？思维链分解是指当你分解一个问题时。你可以要求模型将一个问题分解成多个子问题。

这样每个子问题之间就相对独立了。因为在思维链条中，比如你有第一、第二、第三点，它们之间会相互影响，对吧？而在分解过程中，你会把每个子问题放在独立的上下文中。所以在某种程度上，这减少了偏差。

因此，在这个问题上，让我们看看这里有一个子问题一。Scooby-Doo是什么类型的动物？模型的回答是：Scooby-Doo是一个虚构的角色。

另一个给助手Claude的子问题是：袋鼠的育儿袋平均有多大？你可以看到，每个子问题都是相对独立、自成一体的，是非常原子化、独立的问题。

因此，你会有多个类似这样的子问题。然后你要做的就是重新组合。也就是说，你把子问题、答案、子问题、答案、子问题、答案整合成一个上下文，然后根据这个上下文询问模型，最可能的单一答案选项是什么。

正确答案是B。是的。在思维链中，就像我们提到的系统一样，逐步减少内容。你知道的，输入模型并让它分解成多个步骤。

是的。我可以在接下来的几张幻灯片中分享这个提示。是的。

还有其他问题吗？这张图表？好的。我们来看看提示。虽然很难看清，但我会分享幻灯片。

我将给你一个法律背景，也就是一个法律问题。假设你有一个法律问题要问我们，比如在协议条款下，哪种论点最能说服人们认为某人对债权人负有责任，这就是问题的背景。基本上这就是问题所在。

所以你可以选择不同的模型。这就像是一个多选题。在此之前，你会看到一段简短的提示。

基本上，正如所说，我来这里是为了回答你的问题。我会给你一个问题，希望你能将其分解成一系列子问题。每个子问题都应该自成一体，包含所有必要的信息。

这真的很重要。等等，等等。确保不要分解超出必要的部分。

简洁为上。废话少说。请将每个子问题放入这样的标签中，但要包含与每个标签对应的数字。

模型回答：“是的，我明白了。您有一个问题。请选择答案。”


And the model provides sub questions for you. And then what you do is that you try to answer the first sub question and you give it to the model. You try to answer the second sub question, you give it to the model.

Third sub question, you give it to the model. And then later you say, like, based on everything above, like you give all the context. Answer me the question.

The correct answer is C. Yeah. And so this is very similar in the legal context. You have sub questions like what is consideration of contract law? Blah, blah, blah.

And you can have another model to sample here. You can have another model to answer this. It doesn't necessarily should be one model.

And then there's another sub question. And here's the answer. Yeah.

Do you guys have any questions? The second thing that I want to talk about is how to use Claude to do evaluations. Like evaluating, like, Claude on, like, long context ability. Let's say you have a lot of, like, documents and you want to understand how good Claude is answering questions based on the document.

Or is it able to answer, like, the questions not just, like, from its pertain knowledge, but, like, based on the document itself. And so I'm going to give you an example that we did at Anthropic. Multiple choice QA evaluation design.

So our goal was to evaluate techniques to maximize Claude's chance to correctly recalling a specific piece of information from a long document. And so the document that we chose was a government document that contains, like, a bunch of, like, meeting transcripts, different departments. And we also chose the one that was, like, from this year, July 13th, which is, like, way after Claude's training data cut off.

So that you don't, like, you have the document that does not have in the pertain knowledge or something. And so what you're trying to do is, like, now you want to use Claude to generate question-answer pairs. In a way, like, you create, like, data set-based.

You use language models to create, like, data sets. And so the way you do that is that you split the document into sections and use Claude to generate, like, five multiple choice questions for each section. Each with three wrong answers and one right answer.

And if you do that, you then reassemble, like, randomized sets of those sections into, like, long documents that you could pass them to Claude and test its recall of their contents. This is very meta. Let me know if you have questions.

So here's a prompt to generate multiple choice questions. We ask, like, please write five actual questions for this. Some guidelines at the end.

And basically, we test different strategies, prompting strategies. Just asking Claude, give Claude two fixed examples of correctly answered general knowledge that are unrelated to the government document. Providing two examples and providing five examples of correctly answered questions.

And we tested the strategies on different settings, like one is containing the answer positioned at the beginning, the end, or the middle and the input. And we tested with, like, 70K and 95K token documents. You can look at the prompt and more specific how we did this in our blog post.

But basically, the result is this. Here we see that...

(转录由TurboScribe.ai完成。升级到无限以移除此消息。)

Yeah, the metric was to, let's see, like basically how many, how many times like Claude has correctly answered the question, right, and so, yeah, sorry, basically what we find is that like for document Q&A, asking the question at the end of the prompt performs a lot better than asking at the beginning. You can see it here. Pulling relevant quotes into like critique or like thoughts tags is helpful.

It's like the small cost to latency but improves accuracy and we tested on like both Claude and Claude instant, and it seems like you can boost way better performance with Claude instant than Claude two. Basically the idea is that like if you want to use long doc Q&A, put the instructions at the end of your prompt. Yeah, that's like the result of this.

Oh yeah, like you just ask the model to like put thoughts in like thoughts tags before answering the question so it has like more reasoning based approach. Yeah. Sorry, I'm.

But like the outcome was basically putting it at the end matters more than all the other Yeah. Optimizing. Right.

Yeah. Yeah. In a way this is like an example to show like how to use Claude to generate a data set that you can like evaluate and like you can use it for like evaluation basically.

Yeah. So this has to do with putting your instruction at the end of the prompt. First, are there theories on like why specifically the instruction should be at the end and are there any things like, do we have any understanding of like, are there certain things at the beginning of the prompt that still might be weighted or is it like this sliding scale that like the further in the beginning of the prompt, like the less attention it gets? Yeah, I think that's basically the hypothesis.

It's like the, you know, it's like the distance. It's like the model attends more to the end of the prompt than the beginning. I think there was a paper saying like it just forgets in the middle or something.

Yeah. I think this is the problem with like long context that you're trying to fix or something. Yeah.

So what you're saying is for Claude to do best if you give it at the end. Yeah. So that paper doesn't apply to Claude? Um, I did not read that paper.

No, I'm just curious. What you're saying is you're finding it, at least for Claude, The end part that gets more attention. Yeah.

For like a specific task is like a long context, Q&A for like long documents. Yeah. Um, yeah, we have not tested on other tasks to my knowledge.

Um, so the prompts you showed were using regular prose and then the XML tags. Um, I think that's also what's in the Anthropic docs. Have you guys ever done experiments on like that kind of format versus markdown versus everything is in XML? Do you have any thoughts on that? Yeah.

Uh, so, um, in general, I think, I think it's because markdown was kind of like, there's not that much of like, I don't know. It's like best in XML tags. Like I'm thinking I've like tried to Claude to like, you know, use like JSON now or like, uh, use markdown.

But sometimes it's like, you know, it's not as good as like XML with XML. It's almost a hundred percent accuracy. Yeah.

Let's see. Yeah. Um, let's go to another task.

Um, which is like, you can use language models to like auto label basically anything. Um, so one of the examples that we did last year, um, we asked Claude to categorize the labels for the clusters. And so, um, this was for the paper, but the approach was very simple.

We have a bunch of like, you know, texts and we embed them in UMAP. Um, and we do like KNN clustering and for each cluster, sorry, K-means clustering and for each cluster select like for each cluster aggregate all the, you know, little like statements or claims. And we ask the model to come up with the category for this cluster.

Um, so that's the approach. And you can look at the other labels. Uh, labels here are not super good because we use Claude 1.3 at that time.

Claude 2 is supposed to be like way better at this. Um, but this is like, you know, cash. It was like last year.

Um, where's my slides? And so one thing that you can do with this kind of task, we call it self-consistency. You can generate N samples for the question. Um, so let's say you have a question like, how do you label this cluster? And you generate independently N times.

And you can ask just like, come up with like one category. Um, well this method is mostly useful for like quantitative. Like if you have like a math question and you sample like different, like sample multiple times and come up with the answer.

Um, like the most common answer is, uh, the one that you select for the final answer. And this is called the majority of load. Another technique that you can use is like, um, have like two generated samples and ask another model to evaluate whether those samples are consistent or not.

And if the samples are consistent, well you gain more confidence that this is correct, right? And if it's not consistent, you just like deselect. Um, another thing that you want to do with Claude is, uh, if you, if Claude is kind of like misses the nuance, especially for like categorizing a lot of labels and you have like a lot of like categorizations, um, you can add contrasting conceptual distinctions in your instruction. And you can do it in multiple ways.

One way to do it is like you provide bad example. Let's say like here's a very bad category and you should never come up with it because this is like too narrow or like too general and this is not what I want. Like give like contrastive like examples.

Vary the context. Use examples in different contexts and settings. Um, not just like, just like have like more diversity.

Like diversity is like, um, the more diverse like future prompt examples, the better. Use analogies and metaphors. Um, if the concept is like too hard to understand what the model tried to like decompose and like bring analogy, um, point out like common misconceptions.

Um, especially for like categorizing like let's say what is false presupposition, right? Like, uh, point out the common misconception and like clarify like why this is like incorrect. Like provide examples that explicitly show why a common misconception is wrong. Uh, yes. 

Do you guys have any questions? Yep. Um, I cannot hear but, um. So the goal here is you're running a classifying algorithm with a group.

Yeah. Like come up with a category. Like, um, yeah, come up with a category or like classify, uh, like label that cluster basically.

Um, so here's like very basic like tips and strategies with Cloud API. Um, number one is formatting. Um, like human assistant is like what Cloud loves and if you miss this, you miss it. 

Like you'll get like very terrible results. Uh, new line, new line human. And new line, new line assistant.

Um, yeah. Uh, you can also put words in Cloud's mouth to like kind of like say like, do you understand it? And you can like put in the Cloud's mouth, yes, I understand it. In a way to like, you know, put the model into this mode. 

Have Cloud repeat instructions back. Um, you can say like, do you understand the instructions? Um, and you can put like assistant, yes, I understand instructions, blah, blah, blah. Uh, to reduce hallucinations, like let Cloud hedge and like say like I don't know or like I don't have enough information or like context to answer the question.

Um, here's another thing. Um, if you have like generate direct quotes, if you have like a document or like um, a long document in the context, um, make Cloud to say find appropriate quotes, but also say like, um, if there are no quotes in this document that seems relevant to this question, please just say I don't find any relevant quotes so that it doesn't make up or fabricate new quotes. Uh, how to give good examples? Um, are the examples similar to the ones you need to classify? Are the examples diverse enough for Cloud not to overfit to the specifics? Equally distributed among answer types, don't always choose option A, but like you kind of like have the diversity.

Um, yeah. I got a lot of questions. Yeah.

Formatting in a way, oh here? Um, I think they didn't put like new line, new line. Pretty sure. Oh, sorry, here, you put like human assistant inside the XML tags. 

You only need, you only have to use human assistant as like tokens to like sample, but you should never put it in like like inside the context itself. Either use like user and like other like, um, you know, um, words like user AI or something or like H or A, but you should never use human assistant. Human assistant is like very special, special words.

Um, it would be okay, but you would like make, you should like have human and then assistant in between and then human assistant and then another human and assistant basically. Uh, yeah, formatting is like human assistant, human assistant, you should never have like human, human, assistant, assistant or something. Uh, that's bad. 

Um, yeah. I think we have like more extensive uh, explanations in the API docs if you can look at it. Yeah, I get a lot of questions like what the future of prompt engineering is. 

Um, and I think the answers are pretty clear. Like prompting will stay. We'll just ask like more complicated nuanced questions or like tasks for the model.

Um, prompt engineering is a we will like, we're moving towards the world where we'll have like more and more synthetic data generation and so I'm pretty optimistic about like using models to like generate like diverse sets of like datasets. Um, you can also use language models to write like evaluations. Um, so you use prompting to do that.

Um, reinforcement learning from AI feedback, um, is an alternative to like reinforcement human, from human feedback, which is like a little more scalable, but basically you ask the model to revise its own responses in the process. So you give the model, you like ask the model to like self reflect or like self revise. Um, and so you use prompting in that process to do this.

And especially like prompt engineering will become like a standard part of like product development. I feel like um, things that we did in cloud products such as like auto generating titles, like this things was never like done before, like before like large language models. And so you can like create delightful mini UX experiences, uh, such as like that using just prompting or something.

And you can like have personalization. Maybe you can embed all the users conversations and like suggest like new topics for the conversation. Um, and you can use models to do that. 

Um, and the most like interesting thing is like, uh, finding most optimal prompts for specific tasks. Maybe you want to like minimize the number of tokens to get the highest accuracy for the task. Um, yeah.

Uh, here are some resources. Uh, we just uh, launched Antarctic Cookbook with like certain like demos on research, on retrieval and search. Um, we have prompt design guide in API, uh, book. 

Um, you can also read out the papers that we publish. Uh, oftentimes we have like appendix with like all the prompting that we do. Yeah. 

Thank you so much. And, uh, if you have any questions, let me know. Yeah.

Yeah. Yeah. I think I'm most, um, experienced with cloud because I use it like every day. 

Um, less experience with GPT. Uh, I did not look carefully to be honest at their like API docs. Um, but it seems like the strategy is a little bit different.

Yeah. They don't have like formatting as you are, let's say. Yeah. 

I think, uh, that's, that's actually one of the directions to like, um, don't remember how, what was that paper called? Like LLM optimizers I think. Right. Um, but yeah, I guess like, um, in a way there are like certain tasks that the models are like not good at currently, like for example, like self-correction, like the models are not really good at like self-correcting their like answers and like, can you find like a prompt that was like pretty good at it or like, um, other tasks that you want. 

Yeah. Yeah. I think, um, depending on the task, sometimes we just like have to look manually, qualitatively uh, at outputs. 

Um, sometimes you, let's say, um, you want to evaluate, you know, how much does the model refuses and if it refuses in a relevant context or not. And so, uh, you use, you know, generated answers and you categorize refusals in different categories and use the model to categorize that. And so you just like see the rate. 

Um, yeah. I can think of that example. Yeah. 

It depends on the task. Some tasks are like, you know, um, for like hallucinations, you actually have to like look yourself or something. Yeah.

Was OpenAI? Yeah, yeah. I think, uh, um, I won't say too much about this, but I actually have not like excessively used function calling from OpenAI like other models. Yeah. 

Yes. Yeah. I think, uh, one, actually, this is an interesting question. 

Like I worked on the auto-generating titles for Cloud.AI and, um, one thing that I asked Cloud is to be like an editor, like have an editorial taste. And what we did is actually we took previous titles and we put in the context to generate a new title. And so in a way it's like a little bit more consistent to, um, what the style of the user is. 

Yeah. Uh, yeah. I'm not sure if I can share that.

Uh, yeah, we, we use this in production. I can like show you like Cloud.AI interface. And, uh, one thing that we changed recently is that like, if you have like pretty like short, like, um, you know, sometimes you don't have, you don't need like LLM to come up with a title.

You just take, if the prompt is like very short, you just like, um, use the, like the first like words. Um, but here, yeah, like, I don't know. Um, let's see.

Hey's introduction. But then like recommend some books. Um, yeah, I don't know.

Yeah. Um, no, but I can tell. Uh, so Cloud So let's look at the, is there some docs on this? When did we announce? August 9th? Um, basically Cloud 2 is a larger model.

It's a little bit smarter, it's like smarter than Cloud Instant. Cloud Instant is way cheaper and way faster. But Cloud Instant is better than Cloud Instant 1 in like more like reasoning based tasks. 

So it's way better at math. It's way better at code. Um, other benchmarks are like pretty similar.

But I think we specifically trained Cloud Instant to be good at like math and code. Um, and it's way better at like red teaming, um, like automated red teaming relations. So it's more robust to like jailbreaks.

Um, yeah, I really like this model. You guys should use it. Yeah.

Yes. Uh, yeah. Yeah. 

Red teaming is an interesting concept. It's um, basically you like the models are pretty like vulnerable to like jailbreaks. Um, so sometimes let's say like a very simple example, like can the model give you instructions how to build a bomb? And so we consider it as a jailbreak.

And so the goal is to like uh, in that cases, like the model should like refuse or do not like provide any additional information in case of like, um, unsafe, like prompts or something like this. And so this is like the internal evaluation that we have. Um, you can read in the model card that we launched in Cloud 2 how we specifically do that. 

Um, but it's basically the amount of like, um, how robust the model is to like those jailbreaks. Yeah. Cool.

Yeah. Thank you.

(转录由TurboScribe.ai完成。升级到无限以移除此消息。)