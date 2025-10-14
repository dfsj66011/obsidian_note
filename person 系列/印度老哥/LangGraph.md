
## 一、引言

大家好，我叫 Nitesh，欢迎来到我的 YouTube 频道。我非常高兴地宣布，在这个 YouTube 频道上，我们将开启一个新的播放列表，主题是“基于 LangGraph 的 Agentic AI”。

老实说，这个话题很空洞，过去三四个月里我收到了无数相关的消息，都是你们这边发来的。你们一直跟我说："老师，请您做一个关于 LangGraph 的播放列表吧。"其实早在三四个月前，我就决定要好好做一个关于这个主题的播放列表。这几个月我一直在深入研究这个主题，花了很多时间制定课程大纲，然后围绕这个大纲做了大量准备工作来准备内容。在整个过程中，我做了很多文档研究。今天的视频之所以特别，是因为如果你想完整跟随这个播放列表，今天的视频就非常重要。因为在今天的视频里，我要告诉你我的整个思考过程，我会一步一步向你解释这个播放列表的所有内容。

首先是时机问题。我觉得现在正是学习 Agentic AI 的最佳时机。为什么呢？因为现在无论你打开哪个平台——YouTube、Twitter 还 是Instagram，都会不断看到这个术语。全世界都在大力炒作这个概念，而且我认为这种炒作是有道理的。想想看，2022 年 ChatGPT 来了，自从 ChatGPT 出现后，计算机科学中生成式 AI 开启了一条全新的轨迹。现在生成式 AI 工具已经变得如此成熟，借助它们真的可以构建出非常强大的代理。接下来的五年里，这些 AI 代理将在未来创造巨大的价值。因此，全球所有的大领导、大公司的 CEO 以及处于非常有影响力位置的人们都预见到，这是一项可以彻底改变世界的事物。如果你学习如何构建代理应用程序，那么在未来你将处于一个极具价值的位置。所以时机就是现在。

第二个关键点是需求。正如我在视频开头所说，过去三四个月里，我频道上每三个评论中就有一个是：“先生，请放下一切，在 LangGraph 上创建一个播放列表吧。”因为行业内对此讨论非常多。所以从你们那里传来了非常强烈的需求。第二个关键点就是这个。

而第三个关键点是积累。如果你现在看这个频道，我们一直在按顺序覆盖很多内容。我主要努力以非常系统的方式深入覆盖内容，然后继续前进。所以在这个频道上，我们先做了机器学习，然后做了深度学习，接着我们开始了 LangChain 等等。当我们开始生成式 AI 时，我个人觉得我们已经学习了很多，差不多准备好去学习和理解 LangGraph 以及如何构建 AI 代理了。这就是第三个原因。基于这三个原因，我认为我们现在处于一个应该开始这个特定播放列表并学习相关内容的位置。

好了，现在让我们继续讨论我启动这个播放列表背后的愿景是什么？每当你做任何事情时，背后都有一个强烈的愿景。所以我想与大家分享我的愿景，即通过这个播放列表我希望实现什么。如果我要诚实地告诉你，当 LangGraph 进入市场，并逐渐从你们的网站收到消息说“先生，请教授 LangGraph”时，我做的第一件事就是上 YouTube 搜索，看看目前有哪些关于 LangGraph 的现有内容。

我注意到 YouTube 上有两种类型的内容。第一种是直接使用 LangGraph 教授如何创建项目的内容，这是一种类型。然后还有第二种类型的内容，教授的是 LangGraph 非常基础的基本原理。在这两种内容中，我发现了一个缺陷：在教授如何创建项目的地方，没有讨论基本原理；而在专注于基本原理的地方，视频非常简短，很快就结束了。简而言之，我在 YouTube 上没有找到任何全面的播放列表。没有看到任何关于 LangGraph 的内容，如果有人从开始到结束完成它，那么通过 LangGraph 的帮助，他就能获得完整的端到端知识。于是，我决定创建一个包含大约 30、40、50 个视频的播放列表，并以这样的方式制作这个播放列表：如果有人从头到尾观看它，那么他将学会如何创建代理应用程序，并且完全掌握 LangGraph。

所以，我对于这个播放列表有三个明确的目标：我的第一个目标是制作一个简单易懂的播放列表。我想让任何人，即使是初学者，也能通过跟随这个播放列表轻松学会创建代理应用程序，并且在创建任何类型的代理时都不会遇到任何困难。其次，我的目标是让你通过这个播放列表以这样的方式掌握 LangGraph 的基础知识，从而对 LangGraph 有很强的掌控力。第三个愿景是，我希望通过这个播放列表给你提供如此深刻的概念性理解，以至于即使明天 LangGraph 被其他新框架取代，你也能凭借这种概念深度轻松掌握那个新框架。那么，这三个就是我希望通过这个播放列表实现的可操作目标。好了，我们已经讨论了为什么要开始这个播放列表，也讨论了关于这个播放列表我的目标和愿景是什么。现在我要告诉你最重要的事情：在这个播放列表中我们将遵循什么样的课程安排。

在开始讨论课程之前，我想先给大家一个免责声明：我即将介绍的课程内容并非最终版本。未来可能会出现一些目前尚未纳入课程的新内容，也可能我现在讨论的部分最终不会出现在正式课程中。这背后的原因很简单——AI 技术正在飞速发展，每天都在更新迭代，旧知识不断被取代。因此，我们今天制定的课程大纲，三个月后是否仍然适用尚未可知。这就是我想事先说明的免责声明。这就是为什么我不会告诉你具体的逐项课程安排，而是会告诉你我是如何将整个课程划分为不同的模块的。

好了，让我们从第一个模块开始。第一个模块将围绕 “Agentic AI 的基础” 展开，这将是我们整个播放列表的前 5-6 个视频。在这 5-6 个视频中，我的目标是为你提供一个关于 Agentic AI 及其相关术语的非常深入的概述。我们将在这里讨论什么是 Agentic AI。AI 代理与 Agentic AI 有什么区别，生成式 AI 与 Agentic AI 又有什么区别？除此之外，我们还将讨论 Agentic RAG 是什么，传统 RAG 与 Agentic RAG 有何不同。所有这些内容我们都将在这里进行探讨。我们还将讨论有哪些顶级框架可以帮助您构建 Agentic AI 应用程序。这将是一个包含 5-6 个视频的系列，如果您观看这些视频，您将获得一个非常好的高层次概述，了解我们在这个播放列表中将要涵盖的内容。所以这是一个非常重要的模块。

好的，接下来是模块 2，我们将开始我们的 LangGraph 之旅。在这个特定的模块中，我的目标是教你们 LangGraph 的基础知识，比如 graph 是如何构建的，state 的概念是什么，nodes 是什么，edges 是什么，conditional edges 是什么。我们将通过这些基础概念来学习，并且借助这些概念，我会教你们如何构建一些非常流行的 AI 工作流程。这就是我们模块 2 的计划。

接下来是模块3，我们将学习高级的 LangGraph 概念。如果你掌握了基础知识，并借助这些知识学会创建 AI 工作流程，你就会获得很大的信心。之后，就是深入了解 LangGraph 的合适时机。LangGraph 为你提供了许多概念，借助这些概念，你可以构建行业级的 AI 代理。例如持久性、内存的概念、人在循环中、断点、检查指针、时间旅行的概念，这些都是非常高级的概念。通过在你的 AI 代理中实现这些概念，你可以使它们真正达到行业级水平。所有这些内容我们都会在这个特定的模块中涵盖。

接下来是这个播放列表中最有趣的模块，在那里我们将学习如何创建 AI 代理。到目前为止，我们已经相当详细地介绍了 LangGraph，现在是时候利用这些知识来构建各种不同类型的人工智能代理了。在这个特定的模块中，首先我会给大家讲解关于人工智能代理的理论知识，以及目前行业中流行的设计模式。然后，我们将逐步学习创建不同类型的人工智能代理。我们将从反应代理开始，之后学习创建反思设计模式，接着是一个名为“自我提问求助”的模式，我们将在此基础上展开工作，进行规划。

我们将在此基础上展开工作，完成这些内容后，还将学习构建多智能体系统。这些核心内容会在本系列课程的第四模块中涵盖。当您完成这个模块时，就能掌握创建各类 AI 智能体的技能。之后我们将转向不同方向，学习构建智能 RAG 应用。您之前在 LangChain 中学习过传统 RAG 制作，而智能 RAG 是其进阶版本——我们将 AI 智能体与 RAG 概念相融合。这里涉及多种架构类型，我们将探讨标准RAG、自优化 RAG 等各种不同的 RAG 架构体系。

我们将在这里涵盖这些内容，然后进入播放列表的最后一个模块，在那里我们将学习产品化。到目前为止，我们在播放列表中学到的所有内容都将帮助我们构建一个项目。我们将以这样的方式构建这个项目，以便你可以将其添加到简历中，并在面试中展示给他人。因此，在这里我们将为我们的代理提供用户界面，添加调试支持，增加可观察性，集成 Langsmith，最后我们还将学习如何部署它。所以，整个播放列表中将有 6 个模块。我现在不想与你们详细讨论具体的主题，因为它们一直在变化，但播放列表的大致结构已经在你们的屏幕上显示出来了。

我希望你喜欢这个课程，它组织得非常系统，让你觉得很有逻辑性。如果你有任何反馈，请在评论中告诉我。如果我认为你的反馈非常非常有价值，我会将其整合到这个课程中。好了，现在我们来谈谈先决条件。很多人在开始这个播放列表时都会有疑问，那就是我们是否准备好开始这个播放列表了。如果你想学习、阅读或观看这个播放列表，你应该掌握三件事。首先，你需要懂 Python。这次我不会说“基础 Python 也行”，因为在这个播放列表中，你需要一些中级水平的 Python 知识。我们会涉及一些基础 Python 不涵盖的内容。我们会大量使用面向对象编程（OOP），也就是说 OOP 的原则是必须的。此外，你还会用到 typing 模块、pydantic 等。如果你不具备这些知识，你将无法很好地跟上这个播放列表。这是第一点。

第二点是你需要对大型语言模型（LLMs）有一定的熟悉度。你应该对 LLMs 有一点了解。如果你看过我的 Langchain 播放列表，那么这个问题对你来说就不会是问题。第三点是 Langgraph，它是建立在 Langchain 之上的。所以在这个播放列表中，你会经常看到，每当我们编写任何代码时，都会有一些 Langchain 的依赖。所以如果你一点都没有学过 Langchain，那么这个播放列表对你来说就会很难理解。所以我强烈建议你一定要看一下我的 Langchain 播放列表，里面有大约 18 个视频。

我有一些比较长的视频，但如果你看了它们，这个特定的播放列表会对你很有帮助，好吗？从先决条件的角度来看，你需要掌握这三样东西。现在我想回答一些更重要的问题。第一个问题是很多人经常问的，所以我在这里说一下：这个播放列表里总共有多少个视频？关于这一点，我无法给你确切的数字。下一个问题是上传视频的频率会是怎样的？说实话，我会尽力每周给你带来三个视频，好吗？对我来说，超过三个是不可能的。如果有时候我制作的视频少于三个，那背后一定是有原因的。

因为你知道每个人都有自己的个人生活，其中可能会有各种事情，所以请你稍微理解一下。但我想从我这边给你一个承诺，我会尽量每周上传三个视频，好吗？剩下的计算就交给你了，看看整个播放列表完成需要多长时间。是的，我觉得这两个问题很多人都会问，所以我提前回答一下。除此之外，如果你还有其他问题，可以在评论区提问，我和我的团队会尽力回答。另外，如果你想学习 LangGraph，请关注这个播放列表，我已经向自己承诺要构建它。

在 LangGraph 上创建最佳播放列表，接下来的 3-4 个月里，我会全力以赴打造这个播放列表，真心希望这个播放列表在未来会受到大家的喜爱。对这个播放列表超级兴奋，希望你们也是！如果你们喜欢这个视频，也喜欢我们将要做的事情，请给这个视频点赞，分享给那些想学习 LangGraph 的朋友们。如果你们还没有订阅这个频道，请务必订阅。下个视频见，拜拜！

## 二、

大家好，我叫 Nitesh，欢迎来到我的 YouTube 频道。今天的视频我们将开启一个新的播放列表，这个列表名为《基于 LangGraph 的 Agentic AI》。如果你看过我的上一个视频，应该记得我在其中介绍了我们将如何规划整个播放列表的内容框架，并讨论了课程大纲以及我对这个系列视频的愿景目标。

现在我们要开始这个列表的第一个视频了，今天的主题是《生成式 AI vs 代理式 AI》。我知道你们现在在想什么——"我们连 Agentic AI 是什么都不知道，怎么能理解它与生成式 AI 的区别呢？"你脑海中浮现的想法是正确的。现在，如果不了解 Agentic AI，你就无法理解这两者之间的区别。当我制定这个课程计划时，原本打算先教你什么是 Agentic AI，然后再教你生成式 AI 与 Agentic AI 的区别。在回顾课程时，我意识到内心的老师告诉我，如果我先向你们解释生成式 AI 与代理式 AI 之间的区别会更有趣。

先了解一下区别，然后在下一个视频中我们将正式涵盖 Agentic AI 的主题，为了解释这两种技术的区别，我做了一个实际的场景来演示，我们讨论生成式 AI 作为一种解决方案，然后使用代理式 AI 改进它，了解一场彻底的进化即将如何发生。

#### 什么是生成式 AI

如果你让我总结一下生成式 AI，我会说生成式 AI 是一项强大的变革性技术，它仅仅在三年前才问世，而在这短短三年间，它已经彻底改变了世界。如今，如果你去接触任何个人并询问他们对生成式 AI 的看法，你会发现他们内心会呈现出两种情绪：一种情绪是这个人可能会对生成式 AI 感到兴奋。因为我们都知道生成式 AI 非常强大，可以做很多事情，但同时个人可能会有点害怕，担心这项技术变得过于强大，以至于抢走我们的工作。

如果我们正式讨论一个定义的话，这里我写了一个定义，让我们来读一下：生成式 AI 指的是一类可以创造新内容的 AI 模型。例如文本、图像、音频、代码或视频等类似人类创造的数据。

这非常美丽地捕捉了生成式 AI 的精髓。简单来说，生成式 AI 是 AI 的一个分支，在这里你构建的模型能够在不同模态中创建新数据。模态指的是基于文本、图像或视频的数据。我们的生成式 AI 模型可以在这些任何模态中创建新数据。而生成式人工智能最棒的部分在于，它新创造的数据感觉完全像是人类创作的一样，对吧？这就是生成式人工智能的定义。

现在，如果我们谈论生成式人工智能的发展历程，它已经存在了大约三年。在这三年中，出现了许多非常成功的生成式人工智能产品。如果我要举一些例子，那么第一个例子肯定是 ChatGPT。ChatGPT 是那个标志着生成式人工智能旅程开始的产品，大约三年前这个聊天机器人进入了我们的生活。我想在当前的趋势下，它已经取代了我们所有人生活中的谷歌，不仅仅是 ChatGPT，除此之外还有很多其他聊天机器人，比如谷歌的 Gemini，或者 Claude，或者 Grok。这些都是非常强大的聊天机器人，它们完全具备像人类一样生成文本的能力。不仅如此，显然它们还很智能。因此，基于 LLM 的应用程序是生成式人工智能的第一个真正例子。

第二个是图像生成模型，基于扩散的模型，比如 DALL·E 或 MidJourney，这些也非常流行。现在的情况是，你只需提供一个简单的文本描述，说明你想要什么样的图像，这些模型就能迅速为你生成完全符合要求的图像。

第三个例子是代码生成大型语言模型（LLMs），许多大型语言模型都经过微调，使其能够编写软件代码，例如 CodeLLama 就是一个例子。因此，这也是一类软件模型。

第四类是文本转语音（TTS）模型，这类模型的特点是，你提供一个描述或文本，这些模型会将该描述转换为语音，而且生成的语音听起来就像真人发声一样。当真人说话时，声音是这样的，对吧？这就是一个很好的例子，Elevenlabs。我相当确定你在某个地方听过这个名字。

最后是像 sora 这样的视频生成模型，你再次提供一个文本描述，而这个模型反过来为你制作一个短视频片段，对吧？

所以这些都是产品，它们在过去三年中出现，并在各自的领域中相当成功。那么，你屏幕上看到的这张幻灯片，非常准确地告诉你什么是生成式AI。过去三年 Gen AI 的发展历程如何？如果你想真正理解 Gen AI 的力量，我们可以将其与传统 AI 系统进行比较。现在你可能在想，我所说的传统 AI 指的是什么？我在 AI 领域已经工作了大约 8-9 年，所以在我看来，我们在前生成 AI 时代所做的工作，我称之为传统 AI。我的意思是，如果你过去从事过经典机器学习或构建过深度学习模型，那么我将它们统称为传统 AI，传统人工智能正在召唤我们。

问题是，我们要做的是真正理解生成式人工智能有多强大或有何不同。我们会将其与我们过去构建的传统人工智能系统进行比较。在过去，你有基础数据，数据中有一些输入和一些输出，而传统人工智能模型的工作就是找出模式，或者更准确地说，识别输入和输出之间的关系，以便在未来如果有新的输入，我们可以根据它生成一个输出。

例如，在传统人工智能中，我们主要处理分类问题。如果让我给你一个分类问题的例子，假设我们正在构建一个系统，其任务是查看一封新收到的邮件，并判断这封邮件是否是垃圾邮件。这就是一个分类问题的例子。或者再举一个例子，通过检查患者的 X 光片，我们可以判断该患者是否患有癌症，这也属于分类问题。

现在让我们谈谈传统 AI 系统如何解决这类问题。首先，它们会收集数据，数据中既包含输入也包含输出。例如，胸部图像以及标注信息（是否癌症患者）。我们的传统 AI 模型会做什么呢？它会研究数据、寻找规律，并试图基于这些规律理解输入与输出之间的关系。一旦掌握了这种关系，模型就能轻松地为任何新给定的输入生成相应的输出。

同理，如果你在处理回归问题，回归问题是指你不将数据分类到某个类别中，而是预测一个连续的输出。例如，根据过去的数据，你需要预测今天的温度会如何，或者根据过去的数据，你需要预测某家给定公司的股票价格今天会是多少。对于这类问题，当你构建传统的人工智能模型时，它的工作方式完全相同。它会查看数据，尝试理解其中的模式，并试图找出输入和输出之间的数学关系。而且，当基于这种关系有新的输入时，我们的模型会预测并给出一个输出，对吧？所以这就是传统 AI 系统的工作方式。

相比之下，生成式 AI 从根本上就不同，因为生成式 AI 在您提供数据时，它不会去寻找输入和输出之间的关系，而是试图理解整个数据的分布。理解分布意味着试图理解数据的本质，或者可以说，它试图理解数据的性质。例如，如果你有一个生成式 AI 模型，你给它大量猫的图像，那么你的生成式 AI 模型会尝试理解真实生活中猫看起来是什么样子，猫的分布情况如何。一旦生成式 AI 模型理解了你的数据分布，即理解了现实世界中猫的样子，它就能很容易地从相同的分布中生成一个新的样本给你。生成新样本的意思是就是让它能够从中生成一个新的样本。

我希望通过这件事你能理解在基础层面上，传统 AI 系统与我们过去学习的那些有何不同，以及生成式 AI 的一个巨大优势在于它能够提供如此精致和完美的输出，以至于其输出感觉完全像是人类创造的。由于这一事实，生成式 AI 如今在许多领域得到了应用，并且我们看到了非常好的结果。

那么接下来我们将讨论哪些领域最常应用生成式 AI。首先，生成式 AI 的第一个应用案例就是创意和商业写作，正如我们刚才讨论的那样。尽管生成式人工智能可以围绕多种模态生成内容，但最初的用例出现在文本数据上。当 ChatGPT 问世时，它已经能够像人类一样生成文本。因此，从那时起，生成式人工智能工具在创意和商业写作中得到了广泛应用。

假设你想写一篇博客，那么你可以很容易地做什么呢？你可以先定义一个博客的大纲，然后使用像 ChatGPT 这样的工具来生成整篇博客，对吧？或者你在写一封商务邮件，希望里面没有任何语法错误，而且邮件的语气非常正式，那么你可以先写一个草稿，然后粘贴到 ChatGPT上，生成一个非常正式的版本。目前，在行业中可以看到很多这样的用例。事实上，我注意到在电子邮件中，任何流行的工具和应用程序都能让你轻松阅读任何邮件的摘要，并立即起草新的回复。所有这些功能你已经可以看到了。

第二个应用场景是在软件开发中。过去，我们编写的所有代码都是手动完成的，对吧？有时会出现一些错误或漏洞，我们就得手动调试，或者去 StackOverflow 这样的网站上寻找解决方案。但自从生成式 AI 出现后，首先，你不再需要手动编写完整的代码了。所以自动补全工具更进一步，它们能自动预测如果你写了这段代码，接下来你想做什么，并在此基础上生成整个代码。此外，如果你的代码运行时出现错误，你可以轻松地将这些错误输入到像 ChatGPT 这样的工具中，找出错误的原因。因此，软件开发是 GenAI 被非常积极使用的一个领域。

第三是客户支持。世界上每家公司都需要客户支持，因为无论你销售什么产品，100 个客户中可能有 2 个会遇到问题。这些客户会回头联系你的客户支持团队，打电话或发消息。如果你是一家拥有数百万用户的公司，那么在这样的规模下，为每个特定用户分配一个客户主管有点困难。在这种情况下，公司通常会创建一个聊天机器人来解决这类问题。聊天机器人基本上是基于生成式人工智能的聊天机器人，它会尝试根据用户的查询来解决问题。如果能够解决，那就很好；如果无法解决，那么它会将投诉转发给人工客服。如今，随便拿一家大公司来说，比如 Uber、Zomato、Swiggy，这些大规模运营的公司都有自己的聊天机器人。

第四，在线教育正在发生巨大的变革。比如，当你在 YouTube 上观看视频时，你可以借助 Gen AI 工具来解答疑问。如果你对视频中的某个部分有疑问，你可以直接在那里写下你的问题，比如“我在这个特定时间点有疑问”，然后你的 Gen AI 工具可以解释如何解决这个疑问。或者，如果你想学习一项新技术，你可以轻松地使用 ChatGPT，根据自己的需求定制个性化的学习计划。如果你对某个主题不理解，你可以粘贴相关内容，生成一个简化版的总结。因此，教育领域充满了无限的可能性。事实上，在过去三年里，由于 Gen AI 工具的出现，我们所有人的学习方式都发生了一些变化。

正如我所说，除了文本之外，还有其他模式，Gen AI 可以在这些模式下生成内容。其中一种是图像，第二种是视频。比如说，你是一名平面设计师，你的日常工作就是制作图形。假设你为某个 YouTuber 工作，你的职责是制作缩略图。那么与其完全靠自己制作整个缩略图，你现在可以做的就是提供一点描述，通过 AI 工具生成缩略图，然后逐步改进它。

再假设你在某家公司担任社交媒体实习生，你的工作是介绍他们的产品。你必须制作精美的信息图表，以前人们是用软件手动制作这些信息图表的，现在你可以使用这些 AI 工具，它们会为你生成信息图表。明白吗？或者假设你在广告公司工作，你的工作是为公司制作广告。以前人们制作广告需要付出很多努力，必须进行拍摄和整个广告的编辑。现在你可以怎么做呢？用 AI 工具快速完成。你可以使用像 Sora 和 Runway 这样的工具生成视频片段，然后将这些片段拼接起来，就能制作出一个简短的小广告，对吧？这些只是部分应用领域，我并没有全部列出，但你可以理解这种能够模仿人类创造力的质量，这是一个非常强大的特性，正因为如此，Gen AI 正在各行各业迅速被采用。

关于 Gen AI 最棒的部分是它还在不断进化和改进。举个例子，当图像生成模型刚出现时，它们真的很糟糕。如果你生成一张图片，里面的拼写会非常错误或乱七八糟。但现在你看看最近的模型，你会发现现在生成的图片里面的文本已经没有拼写错误了，对吧？所以你可以从这个例子中看到，仅仅在过去 3 年里，生成式人工智能已经极大地进化和改进了。而且今后这一点也将继续成立，那么未来很有可能无论你在生活中使用什么应用程序，你都会看到 Gen AI 的某种整合，对吧？

所以我希望在过去 10-15 分钟内，我能给你一个快速的回顾，告诉你 Gen AI 是什么，Gen AI 如何工作，与传统 AI 有何不同，以及 Gen AI 在哪些应用领域有用。所有这些内容我都已经在快速回顾中告诉你了。现在我们已经有了这个介绍，现在我们要做什么呢？我们要选择一个实际的场景，一个问题，然后借助 Gen AI 来解决这个问题，好吗？

首先，我来告诉你我们要解决的问题是什么。想象一下，你在某家公司担任人力资源招聘人员的职位，好吗？你的工作就是每当公司需要招聘新人时，你就负责招聘，好吗？目前，你被分配了一个任务，就是为公司招聘一名后端工程师。好的，那么现在你需要确保你的公司找到一个优秀的后端工程师。那么，现在如果你把这个任务分解一下，看看为了雇佣一个好的后端工程师需要执行哪些步骤，我们来看看这些步骤是什么。

首先，第一步是作为人力资源招聘人员，你需要起草一份职位描述。你需要创建一个文件，在其中非常详细地列出所有的要求，明确你需要什么样的后端工程师。你想让他做什么，以及他应该是什么样的，比如他应该具备哪些技能，是否有任何资格标准，你会给他多少薪水，以及你对那个员工的期望是什么，所以你必须首先创建一个职位描述，

任务编号2，一旦职位描述创建好了，你就必须把它放在某个工作平台上，比如 job.com，一旦职位描述发布到平台上，之后你会收到所有的申请，你必须对这些申请进行筛选，比如说有 1000 个人申请了你的职位，显然你不能面试 1000 个人，所以你要根据他们的简历从这些人中筛选出 10、20 或 25 个人。

你会研究每个人的简历，将符合你要求的进行比较，然后筛选出 25 名候选人进行面试，对吧？之后就是面试环节，你需要面试这 25 名候选人，以了解谁最适合。一旦你找到合适的人选，接下来你会做什么？你会为那个人发放录用通知书。最后，当他接受了录用通知书，你就需要负责他的整个入职流程，对吧？

那么这就是你的任务，对吧，你需要执行它，而现在我们要做的是在整个任务执行过程中实施 Gen AI，我们会看看 Gen AI 如何应用于整个任务的执行过程中，非常简单，让我们暂时想象一下我们公司已经给了我们一个聊天机器人，对吧，而且那个聊天机器人基本上是一个基于 LLM 的聊天机器人，我可以去那里聊天，解决我所有的疑问，我可以去问聊天机器人，如果我需要任何帮助，我就可以去问它要。这是一个简单的聊天机器人，我的公司给我提供的。好的，那么我们现在开始我们的整个招聘流程。

第一步是我需要起草一份职位描述，对吧？所以我会去这个聊天机器人那里，然后我会对它说，看，我想招聘一个有 2 到 4 年经验的后端工程师，好吗？然后聊天机器人会说，告诉我，我该如何帮助你？然后你会说，请为这个特定的职位制作一份职位描述给我。那么，既然这是一个基于 LLM 的聊天机器人，它会为你生成一份职位描述（JD），就像这里写的那样：我们正在寻找一名有 2 到 4 年后端开发经验的远程后端工程师。你提供的所有其他细节，你的聊天机器人都会一一填入这里，比如薪资是多少，要求有哪些，所有内容都会被安排在这份 JD 里，对吧？所以，在完成第一步的过程中，我们的聊天机器人已经帮了我们。

现在，我们继续下一步。下一步就是，我们手头已经有了 JD。如果我们要把它发布到某个招聘门户网站上，我会去问我的聊天机器人，问它这个职位描述可以发布在哪里。聊天机器人可以根据它的训练知识回复说，有一些平台你会得到很好的回应，比如试试 LinkedIn 和 Naukri。然后我会说好的，谢谢，我来做这个。接着我会手动去 LinkedIn 上发布这个职位描述，手动去 Naukri.com 上粘贴这个职位描述。

现在进入第三步，第三步是筛选。已经过去了一段时间，我们的求职门户上也收到了一些申请。现在，我们有 8 位申请者提交了申请。那么现在，我可以去我的聊天机器人那里说，兄弟，我收到了 8 份申请，你能帮我筛选一下吗？我的聊天机器人可以根据职位描述和一些通用建议给我一些指导，比如雇佣那些有 Python 和云计算经验的人，以及那些有创业公司工作经验的人，并且雇佣那些有领导项目经验的人，那么他们会根据我创建的职位描述给我一个通用的建议，然后我会再次表示感谢指导，我会去做，然后我会执行整个流程，我会逐一查看简历，并根据职位描述筛选候选人。

现在筛选的下一步是安排面试，对吧，那么我会再次去找聊天机器人，并说看，八个人中有两个人我喜欢，我需要面试这两个人。你能起草一封邮件，让我可以邀请那些人参加面试吗？那么聊天机器人会为你起草一封邮件，你可以拿起那封邮件发给那两位候选人，对吧，安排时间的步骤也完成了，这是一封示例邮件。

好的，现在来到下一步，你需要面试那些候选人，为了面试他们，你显然需要问他们一些问题，在这里你的聊天机器人也可以帮助你，你可以去找聊天机器人，然后说看看这个职位描述，并在此基础上告诉我。关于可以问哪些问题，那么再次强调，你的聊天机器人基于其训练数据，会给你一些指导，比如兄弟，可以询问他们在后端开发方面的经验，他们使用过哪些框架，围绕解决问题你可以提问。所以你可以对Jat说，帮我生成一个问题库，那么你的聊天机器人会通过Jat为你生成一个后端问题库。现在你可以轻松地用这些确切的问题去询问候选人，对吧。

那么面试的步骤也完成了，现在你找到了合适的人选，接下来你需要做的就是给那个人发一份 offer。然后你再次去找聊天机器人，告诉它：“看，我已经确定了一个候选人，你能帮我起草一份 offer letter吗？”你的聊天机器人会立即为你生成一份 offer letter，你只需要拿起这份 offer letter，然后通过邮件发送给你的候选人，对吧？就是这样。我想你已经看到了，在整个任务的每一步中，你的聊天机器人是如何协助你的。

如果你回到前生成式 AI 时代，如果我们谈论 2015-2018 年，那时候你必须自己处理所有这些事情，你必须自己创建 JD，你必须自己找出面试问题，所有你需要发送的邮件内容都必须自己写。但现在，Gen AI 已经渗透到所有这些环节中，它非常轻松地优化了所有这些步骤，让我们需要做的工作少了一些，而且输出的结果也更好了一些。所以，Gen AI 显然在帮助我们解决这个问题，但这是否是最好的解决方案呢？或者这个整体方法中仍然存在一些问题，答案是肯定的，确实存在某些问题，我们可以解决它们，这些问题是什么，我们一个一个来讨论。

第一个问题是，整个聊天机器人与人类的互动过程，人类不断过来告诉它现在要做什么，而聊天机器人则在瞬间解决这些查询，这整个过程是被动的，被动是什么意思呢？当我向聊天机器人发出提示时，它只是被动地做出反应并为我提供解决方案，而不是主动的。它无法自行理解流程应该是什么，下一步应该做什么，聊天机器人内部缺乏这种理解能力，它是被动的。所以这是一个问题，作为人类，我不得不处理大量的流程，然后聊天机器人在某些地方帮助我，这是第一个问题。

第二个问题是我们的聊天机器人目前没有任何记忆功能，这意味着它不具备上下文感知能力。如果我今天让它生成一份职位描述，三天后我再回来询问关于那份职位描述的内容，它不会记得，我必须再次向它展示职位描述的内容，对吧。


तो memory का issue है, उसके बाद जो भी



वो बहुती जेनेरिक अडवाईज है। जैसे मैंने उससे पूछा की मुझे एक जेडी बना के दो बाक एंड इंजीनियर की। तो वो जो जेडी बना के देगा ना वो एक जेनेरिक सा जेडी होगा जो अक्रोस कंपनी कहीं भी यूज़ किया जा सकता है। बट अच्छा होता अगर वो जेडी मेरी कंपनी के हिसाब से स्पेसिफिकली बन के आता। मेरी कंपनी के डि.ए.ने के हिसाब से। राइट, बट करेंट जो इंप्लिमेंटेशन है वो जेनरिक एडवाइस ही दे सकता है। हमारी कंपनी के स्पेसिफिक कोई इंफॉर्मेशन उसके पास नहीं है। राइट, तो ये भी एक प्रॉब्लम है। और प्रॉब्लम मंबर फौर इस कि हमारा जो चाट बॉट है अभी वो खुछ से कोई आक्शन्स नहीं ले सकता। मतलब ये हुआ कि भले वो मुझे जेडी बना के दे सकता है बट खुछ से वो जा करके नौकरी.com पे उसको पोस्ट नहीं कर सकता। वो मुझे इमेल का कॉंटेंट बना के दे सकता है बट खुछ से इमेल नहीं कर सकता। अच्छा होता अगर हमारे चाट बॉट के पास ये capabilities भी होती। तो अब हमने आपको problem भी दिखा दी और उसका एक solution भी दिखाया और ये भी हमने discuss किया कि उस solution में क्या-क्या problems है। अब क्या करते हैं अब एक-एक करके इन problems को ये जो problems आपके सामने लिखी हुई हैं इनको एक-एक करके solve करते हैं और अपने Gen AI solution को improve करते हैं। जैसे कि next what I will do is किसी तरीके से हम ये make sure करेंगे कि हमारा Gen AI चाट बॉट generic advice देने के बदले specific to our company ऐसा advice दे। तो हम अपने Gen AI चाट बॉट को इस तरीके से modify करेंगे कि उसके पास हमारी company का information भी होगा और going forward जब भी हम उससे कुछ भी response मांगेंगे तो वो हमारी company का information ध्यान में रख करके response generate करेगा so हम अपने Gen AI चाट बॉट को improve करने के लिए so that वो generic advice ना दे करके हमारी company के हिसाब से tailor made advice दे इसके लिए उस चाट बॉट को हम अपने company के knowledge बेस से connect करदेंगे so basically what we will do is कि हम उस चाट बॉट को हमारी company के बहुत सारे documents provide करदेंगे so that future में जब भी हम चाट बॉट से questions पूछेंगे so that our chatbot can refer these documents to us and give us an answer and since it will read our company's documents and give us an answer so the replies that will come will be tailor made for our company so the question that comes is exactly what documents we will provide to our chatbot so you can provide many types of documents like first of all you can take all the JD templates that have been used in your company in the past you can take all of them and feed them to the chatbot like past all the jobs that have been hired in your company you have provided the JD of all those jobs you have also given the example of those JDs which have been high performing means such JDs on which many applications have come and apart from this whatever variations are there in JD like the variation of remote vs in office or the variation of junior vs senior you have also fed all of this to your chatbot apart from this what you can do is the hiring strategy of your company or you can call it hiring playbook you can also feed that to your chatbot what will be there in this? there will be information that which is the best platform for your company to do hiring what are the best practices that your company uses in order to do effective hiring what is the internal salary band of your company how much salary do you pay on this particular experience level there can also be this information that when you are shortlisting a profile what all pointers do you see here all the interviews that have been done in the past what all questions have been asked in those interviews they also have a question bank so you are feeding all these things to your chatbot apart from this whatever documents are required for onboarding you can also upload all that like the templates of the offer letter or the templates of the welcome email or what are the employee policies you can feed all of this to your chatbot so when you give all this information to your chatbot so now your chatbot will not give generic advice it will give the output in reply according to your company now I guess if you have read a little about generative AI in the past if you have read about lang chain then you might be able to identify that this chatbot is called RAG based chatbot RAG as in Retrieval Augmented Generation it is a very famous GNI concept so now what we did we improved our simple LLM based chatbot and made it RAG based chatbot and now what we will do we will execute the same task with this RAG based chatbot like first of all we have to make a GD for our job so again I will go to the chatbot and I will tell it that I want to hire a back-end engineer so now this chatbot since it knows everything about my company so it can give me a lot of tailor-made solutions like as soon as I tell it that I want to hire a back-end engineer it will understand that our tech stack is python or django so it will automatically mention it in GD if I tell it that I want to hire at 2-4 years experience level so my chatbot will automatically understand that how much salary it should be paid so all this will automatically come in GD I don't need to specify it explicitly so this step 1 of drafting now our RAG based chatbot has become more useful than before so GD is made now we have to post GD so again I will ask that where should I post I am a new HR recruiter so your chatbot will tell you that based on past hiring processes these are the platforms which have given us the best results so put it on LinkedIn, put it on Naukri, put it on AngelList you will say thank you I will post it and you will manually go to LinkedIn, Naukri and post that job after that comes the shortlisting so again you have 8 applications and you went to chatbot and you said that I have so many applications help me in shortlisting so again since chatbot knows how people have been shortlisted in the past so based on their profile it will give you customized pointers to hire people who have Python Django experience who have AWS experience who have prior startup experience who have project lead experience and if you upload resumes yourself then it will manually figure out that these are the 2 or 3 candidates whose skill set is matching the most you should shortlist them so in the process of shortlisting this chatbot is helping me more than before after shortlisting comes scheduling again for scheduling I will go to the chatbot and I will say that I have shortlisted 2 people can you draft an email again it will go to its knowledge base there it will see how to schedule an interview how to use a template for email in that template it will generate an email and you will send that email to the candidate after that comes the interviewing step you will go and tell your chatbot that tell me for this profile what kind of questions have been asked in the past in this company and again your chatbot will go to its knowledge base and search and it will tell you that these are the questions that have been asked in the past at the experience level of 2-4 years from a back end engineer you will say that do one thing remove all the questions that have been asked in the past and it will remove all the questions from the interview question bank and you will conduct the interview on that basis and finally if you like someone then you will go to your chatbot and you will say can you draft me an offer letter so again your company's style or the format in that format an offer letter will be generated and you will mail that to your candidate right so now you can already see that the previous implementation of a simple LLM based chatbot versus this implementation of a RAG based chatbot there is a sure shot improvement earlier where I was getting generic advice now I am getting advice specific to my needs and clearly the RAG based chatbot is much more useful right but have we tackled all the problems the answer is still no if we go back to the problem slide then still there are certain problems which we have not solved like still my RAG based chatbot is reactive which means that I am going and asking the question and it is answering me it is never happening that the chatbot is coming and taking initiative and it is telling me that I should do this next right so still our chatbot is reactive still it does not have context awareness it will forget what we talked about 3 days ago I will have to remind it again this one thing we have solved earlier where I was getting generic advice which was not that useful now I am getting advice specific to my company so we have solved this particular problem and lastly still our chatbot is not able to take actions by itself though it is able to draft JD as per my company but it is not able to post it on LinkedIn so we have moved forward but still there is scope for improvement so let's do one more thing our existing RAG based chatbot let's improve this how? let's discuss next so the next improvement our chatbot does is that we want our chatbot to be able to take actions by itself mostly the chatbots you might have used they can only give textual replies right? we ask a question chatbot's job is to flip and give contextual reply but what if our chatbot along with giving replies can take some actions by itself for example not only it can draft mail but at the same time it can also send mail not only it can draft JD but at the same time it can post JD on LinkedIn so our work can be easier so to achieve this we will integrate some tools with our chatbot for example we will integrate our chatbot with LinkedIn API tool so that our chatbot will be able to communicate with LinkedIn we can also connect our chatbot to a resume parser tool with the help of which our chatbot will be able to understand the content of the PDF we can also integrate our chatbot to a calendar tool with the help of which it will be able to see when we are occupied when we are busy and accordingly it will be able to make a schedule for us we can also integrate our chatbot to our mail API so that it can send and receive emails and lastly we can connect our chatbot to our human resource management software with the help of which it can do a lot of work so this is our next improvement what we did is we took our chatbot and integrated all these tools so now not only our chatbot is able to give contextual replies but at the same time we can also make it do all these tasks and this type of chatbot is called tool augmented chatbot it is a chatbot but it has additional powers you have given it that it has the power to access tools so now we will again see the same flow but this time with a tool augmented chatbot so again i came to my chatbot i said that i need a back end engineer i have 2 to 4 years of experience so here my chatbot can assist me with the drafting process it will make a JD for me based on company information we are not using any tool here we are just using the aspect of our chatbot and generating a JD in the next step we can use a tool when we have to post a job so here i will tell my chatbot JD is made can you post it on various platforms so your chatbot will say based on past hiring processes following platforms performed the best which are LinkedIn and Naukri, now what it will do it will not wait for you it will automatically hit the LinkedIn and Naukri API and post the JD so we have an API through which we are posting jobs on LinkedIn and Naukri automatically so now as a recruiter my job is saved next step suddenly we realized that that many people did not apply for the job so i will go and ask the chatbot check and tell how many applications we have received so again since our chatbot has LinkedIn access it will say connecting to LinkedIn and it will tell you that we have just received one application so now you will say what can be done so there again your company's hiring playbook it will give you some solutions that in the past few applications have been read so in the company you can do 2-3 things first you can broaden the JD rather than writing backend engineer write full stack engineer similarly you can boost the post on LinkedIn as in you can run ads so here again you will tell your chatbot to do 2 things 1st revise the JD means write full stack engineer instead of backend engineer so your chatbot will revise the JD go to LinkedIn and update the post 2nd you will tell it do one thing i am giving so many credits go and boost it on LinkedIn so again it will go to LinkedIn and boost the post so you can again see if something is going wrong so in that also our chatbot is assisting us next step is shortlisting if we get some applications so I will go and tell my chatbot can you help me shortlist the candidates so here my chatbot will say sure, I have a resume parser tool so it will go to LinkedIn it will download the resume of all 8 applicants it will study those resumes it will study JD, it will match both and it will select the best candidates and give it to you it will say 2 candidates shortlisted now you will say mail me their profiles so since again my chatbot has the power to mail it will pick up those resumes and mail it to you so you can see in the shortlisting process because you have a resume parser tool so as a recruiter my work is getting easier right next is scheduling I have to schedule an interview so I will say mail the 2 candidates you shortlisted and schedule the interview so now our chatbot will hit our calendar API and will check when I am free and it will ask me are you free this friday can I keep your interview on friday so I will say yes and it will send a email to the candidate and me as well and your chatbot will say I am crafting a mail, mail is ready I have sent it to both of you so you can see in scheduling by using the calendar API our task is getting easier after that comes interviewing so here again the flow will be the same you will ask again what kind of questions to ask it will go to your company's database it will make a list of all the questions asked in the past and send it to you and again you selected a candidate you will say can you draft an offer letter it will say yes review it you will say yes so since it has the power to mail it will mail the offer letter to the candidate and when the candidate accepts your chatbot your chatbot can help you you will say to the chatbot the candidate has accepted the offer letter send him a welcome email so again it will draft a welcome email it will mail it to you now you will say the next thing to do is trigger the onboarding process so since your chatbot has access to your HR management software it will again trigger the onboarding process in which there will be a lot of things first of all the employment contract will be generated the new employee's official email id will be created he will be assigned a laptop he will have a KT session plan and all this you are doing with the HRM software so again you can see as an HR recruiter my work is getting easier when I am using a tool augmented chatbot so I hope you can see how we are bringing improvement in our solution but have we solved all the problems the answer is still no let's discuss again what problems are left problem number 1 our chatbot is still reactive which means I am taking the initiative I am telling him what to do he is not telling me so our chatbot is still reactive he still does not have context awareness he does not have memory so this is also a problem he is giving me specific advice because it is a drag based software so this is happening and now we have solved this problem that our chatbot can take actions by itself I don't have to mail myself I don't have to post myself I don't have to check when I am free this week all these things are done by my chatbot I don't have to manually run the process everything is done by my chatbot right there is one more problem I want to tell you our current chatbot cannot adapt by itself which means when we told him guess what many applicants are not applying so after we told him he understood that this is a problem and to solve it these things can be done but he did not strategize by himself he did not understand that this is a problem so he is not able to adapt if there is a problem then our chatbot is not able to adapt this is also a problem so in a nutshell we have solved 2 problems our chatbot is still not reactive context aware and cannot adapt by itself so now let's do one thing let's do one last improvement and through that last improvement we will solve these 3 problems so in a nutshell we have to improve our chatbot in such a way that first of all instead of being reactive it becomes proactive it takes initiatives by itself and it has context awareness which means it remembers what it has done in the past and what it has to do in the future and also it has adaptability which means when a flow of action is not executing properly then it can choose alternate paths so this type of chatbot which is proactive which is reactive which is context aware and which is adaptable this type of chatbot you can call as AI agent right so we basically want that when we tell our chatbot that I want to hire a backend engineer then it will understand this whole goal that ok I need to hire a remote backend engineer who has 2-4 years of experience and not only it understands this goal but to execute this goal it can also plan it will understand by itself that to execute this task I have to make a JD I have to put it on a platform I have to continuously monitor the job to see how many people are applying if needed I have to adjust the strategy I have to screen the candidates I have to take interviews I have to send the offer letter I have to handle their onboarding so now you can see that this chatbot is not reactive it is proactive you just told it an end goal that I want to hire a remote engineer backend engineer and after that it automatically planned the whole path right so we call this type of chatbot agentic AI chatbot ok and now we will see that if you have this type of chatbot then how this task is executed ok since we have told it that we need a backend engineer so it understood the goal it made the plan and now it will execute the plan one by one so first of all it will tell you that I will first start with drafting the JD and in that process I am taking the help of the company's documents and now I have drafted the JD once you review it if you need any changes then let me know so you saw it and you told that this is absolutely fine now after that your chatbot will tell you that you liked the JD so now what I am doing based on past data I am posting this JD on two platforms LinkedIn and Naukri since this chatbot has access to tools so it will quickly use LinkedIn and Naukri's API to post the job you will say thank you and your chatbot will tell you that the job is posted now I will continuously monitor how many people applied ok by the way this whole process is happening behind the scenes through the API as I told you earlier in the tool augmented chatbot now suddenly you get a message from your chatbot and your chatbot is telling you that the job posting has received just two applications so far, much below our expectation and it identified the problem itself and started telling the solution it says that the suggested action is to broaden the JD hire full stack engineers and promote the job on LinkedIn do you think these two things are right if yes, then I can proceed so you said yes, do both so it went quickly revised the JD and ran ads on LinkedIn and it said I will keep monitoring after that some applications are there on your job posting since it is continuously monitoring it will notify you that 8 applications received and I have not only monitored I have also screened i.e. using the resume parser tool downloaded the resume analyzed it and based on my analysis I can see 2 strong candidates out of 8 3 partial matches 3 weak matches so should I schedule an interview for 2 strong candidates you will say ok go ahead now what it will do it will check your availability from your calendar API and it will tell you that you are free on Friday do you want me to schedule the interview on Friday, you will say fine it will create an invitation mail it will send the candidate and you too after that there is a process of interviewing it will send you a reminder on the day of the interview that today your 2 interviews are lined up you will say thank you for reminding it will say that I have mailed you a document which has a list of interview questions which you can ask the candidate I have taken these questions from past hiring so you will say thanks, I will check them out you will take the interview you will like someone you will go and tell the chatbot I have finalized one candidate can you make an offer letter for that candidate it will go to past documents and see how the offer letter looks it will make an offer letter for you it will ask you to review you will say fine it will send the offer letter it will not only send it it will also track the reply after that as soon as the candidate accepts the offer letter your chatbot will notify you again that the candidate has accepted the offer and I have triggered the onboarding I have sent him a welcome email I have submitted the request for IT access and for that candidate the laptop provisioning is also done can I set up an intro meeting for both of you you will say yes and it will set up an intro meeting for you and now you can just see how magical it is if this whole thing is executed in this way then you can see how much autonomy is there in the system and as a recruiter I just have to monitor this whole thing and where I have to give approvals I am just giving approvals rest of the heavy lifting this system is doing for me and this is an agentic AI system now if we go back to those problems which we identified in the beginning now our chatbot is not reactive it is proactive it is identifying the goal by itself it is planning to complete that goal and every step of that planning it is executing by itself because it is context aware memory is obviously with our chatbot that's why it is context aware it remembers which step it did last and which step it has to do next so this is also a very important improvement it is giving specific advice according to our company because there is a rag element here it is able to take actions by itself because it has tools integration and lastly it can adapt as well a while ago when it was monitoring how many people applied and only 2 people applied then it identified this problem that more people should have applied so it identified the problem it gave the solution you just had to approve it did all the work by itself so the concept of adaptability is also implemented here so i hope i was able to explain to you how an agentic AI chatbot will solve this problem in this scenario so now let's conclude this discussion we took a problem statement that we are a HR recruiter and we saw 4 ways to solve that problem starting with generative AI then we improved it and added the concept of rag then we improved it and added the concept of tools and finally we solved that problem with the help of agentic AI so let's do one thing let's conclude what we understood from this discussion so you must have understood that generative AI focuses on creating content textual, image, video any kind of content generative AI's end goal is to generate some content whereas agentic AI is something very different agentic AI's end goal is that you have been given a goal and you have to achieve that goal at any cost you get a goal you plan for it and then step by step you execute those plans so this is the biggest difference second difference is











## 概述

LangChain 是开始使用大型语言模型（LLM）进行开发的最简单方式，只需不到 10 行代码，就能让你开始基于 OpenAI、Anthropic、Google 等平台构建智能代理。

LangChain 代理构建在 LangGraph 之上，旨在提供持久执行、流式处理、人工干预、持久化等功能。对于基本的 LangChain 代理使用，您无需了解 LangGraph。


### 核心优势

* 不同的提供商拥有独特的 API 与模型交互，包括响应格式。LangChain 标准化了您与模型的交互方式，使您能够无缝切换提供商并避免被锁定。
* LangChain 的代理抽象设计得易于上手，让您用大约 10 行代码就能构建一个简单的代理。但它也提供了足够的灵活性，让您随心所欲地进行上下文工程。
* LangChain 的代理构建在 LangGraph 之上，这使我们能够利用 LangGraph 的持久执行、人在环路支持、持久化等功能。
* 通过可视化工具深入洞察复杂代理行为，追踪执行路径、捕捉状态转换并提供详细的运行时指标。



--------

原始 LLMs 应用：

* 需要人类提供大量的指令，被动响应式回答问题
* 没有记忆
* 给出的建议都非常通用，如不能结合公司内部规则给出针对性策略
* 不能做 actions，如发送邮件

RAG 改进：

* 仍然是反应式（Reactive）被动的，仍然缺乏上下文感知能力
* 仍然没有记忆
* 可以给出针对性具体意见（主要改进点）
* 还是不能做动作

工具增强：例如引入一些外部 API 等，可以解析 PDF，可以自动发邮件等

* 聊天机器人仍然是被动的，是我在告诉它该做什么，而不是它在指导我
* 仍然缺乏上下文意识，没有记忆功能
* 可以给出针对性具体意见（RAG 改进）
* 可以执行某些操作（工具增强改进）
* 不能自适应某些特殊情况，如果中间出现任何问题，我们的聊天机器人无法适应

所以简而言之，我们需要以这样的方式改进我们的聊天机器人：

1. 首先，它应该是主动的而不是被动的，意味着它应该采取一些主动行动，
2. 并且具备上下文意识，这意味着它应该记住之前做了什么以及接下来需要做什么。
3. 同时，它还应具备适应性，这意味着当一个行动流程无法正确执行时，它能够选择替代路径。

*AI Agent：具有主动的、具备上下文意识的和适应性的，你只需要告诉它一个最终目标*，比如我需要招聘一个后端工程师。

**生成式 AI 和代理式 AI 的区别：**

1. 任何内容生成式 AI 的最终目标是让你获得一些生成的内容，而代理式 AI 则完全不同。在代理式 AI 中，最终目标是给你一个目标，你必须不惜一切代价实现这个目标。你获得目标，为其制定计划，然后逐步执行这些计划。所以这是最大的区别。

2. 第二个区别是生成式 AI 是被动的，你作为人类在每一步指导生成式 AI 该做什么，而生成式 AI 做出反应；而代理式 AI 是主动的、自主的。一旦你给它设定了一个目标，它就会自动完成所有后续工作。它甚至会将人类纳入循环中，但人类的工作大多围绕审批展开。

3. 第三点也是最重要的一点是，生成式 AI 是代理 AI 的基础构件。代理 AI 是一个更广泛的术语，涵盖了许多正在进行的元素，包括工具的概念、规划和推理等。为了实现这些功能，代理 AI 应运而生。生成式 AI 是一种能力，而代理 AI 是一种行为

----

代理型 AI 是一种能够从用户那里接受任务或目标，然后在最少人为指导下自行完成任务的 AI。它会进行规划、采取行动、适应变化，并且只在必要时寻求帮助。简单来说，代理型 AI 是一种软件范式，在这种范式中，你向系统提供一个目标，然后系统开始自行思考如何实现这个目标，在实现该目标的过程中，所有需要的规划和执行工作都由代理型 AI 系统自动完成。

代理式 AI 系统的特征：

* 自主的；
* 目标导向的；
* 能够进行一些规划；
* 能够进行一些推理；
* 能够自适应；
* 是情境感知的。


任何代理型 AI 系统的核心高级组件：

* 代理型 AI 系统的大脑；（LLMs）
* 编排器（orchestrator）；（决定什么时候、哪个步骤、如何执行等，指的是框架，如 langgraph）
* 工具；
* 记忆；
* 监督者（让你的代理和人类一起工作的组件）

---------

### LangChain vs LangGraph

LangChain 是一个开源库，旨在简化基于 LLM 的应用程序的构建过程。

核心组件：
* model 组件，它提供了一个统一的接口，借助该接口可以与任何 LLM 提供商的 LLM 进行交互。这样就不需要在代码中做太多修改了。
* prompt 组件，通过提示的帮助，可以进行任何类型的提示工程。
* 检索器组件，能帮助从任何向量存储或知识库中获取相关文档。

LangChain 最大的优势、最突出的特点正是它的“链”（chain），LangGraph 是一个编排框架，它利用LLM 提供有状态的、多步骤和事件驱动的工作流程。它非常适合设计单智能体和多智能体的 AI 应用。


现在LLM工作流中可以有很多任务。一个任务可能是调用LLM，第二个任务可能是调用某个工具，第三个任务可能是做决策。所以基本上，Langraph在这里做了什么？它理解了你的工作流，然后将这个工作流转化为一个图的形式。这个图的构建方式是，每个节点都是你整个工作流中的一个子任务。最棒的是，所有这些节点都通过边相互连接，而这些边告诉我们，在执行完一个特定节点或任务后，接下来应该执行哪个任务。简而言之，Lang Graph 正在做什么？它为您提供了一个功能，通过这个功能的帮助，您可以将任何 LLM 工作流以流程图的形式先表示出来，然后再执行。一旦您创建了这个图表，接下来您需要做的就是为第一个节点提供输入，并触发这个工作流或图表。然后，所有节点将按照正确的顺序自动执行，您的工作流就完成了。这里确切地写着：Lang Graph 是一个用于构建智能的编排框架。

stateful and multi-step LLM workflows 好吧，现在Lang Graph不仅仅局限于创建图表，它还为您提供了额外的功能。比如在这里，如果您愿意，可以并行执行任务，正如您在这里看到的，一个节点之后，接下来的两个节点会同时执行。您还可以实现循环的概念，在这个节点之后，您可以回到前面的节点，并且可以在循环中完成这个操作。您还可以进行分支，在这个节点之后，根据某个条件，要么这个节点会执行，要么那个节点会执行。同时，您还可以获得记忆功能，可以记录这里正在执行的所有任务和发生的所有对话。此外，您还可以获得恢复能力的功能，如果将来某个任务中您的整个工作流程中断了，您可以从那个点恢复。因此，结合所有这些核心功能，您可以说...

