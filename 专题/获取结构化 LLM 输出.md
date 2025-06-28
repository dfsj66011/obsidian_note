
https://www.deeplearning.ai/short-courses/getting-structured-llm-output

欢迎来到由 DotTxt 合作推出的"获取结构化 LLM 输出"课程。在传统聊天界面中使用 LLM 时，非结构化文本输出尚可接受，但若用于软件开发，解析和依赖自由格式文本输出将变得异常困难。这正是 JSON 等结构化输出至关重要的原因——它们为机器提供了清晰可读的处理格式。例如，若需要 LLM 分析产品评论并生成产品名称及情感倾向（积极/消极）两个字段，以 JSON 格式输出这两个字段就能确保下游软件可靠处理这些信息。本课程将指导你掌握多种获取结构化输出的方法。

基本思路是，你可以告诉 LLM 你希望数据以特定格式呈现。如果你能清晰地描述这种格式——特别是使用计算机科学家所称的正则表达式（regex）——那么这将是一种高效的方法，确保 LLM 可靠地生成符合该格式的输出。因为正则表达式可用于对下一个生成的 token 施加约束，逐个标记地进行控制。

很高兴向大家介绍我们的讲师：威尔·库尔特（Will Kurt）和卡梅伦·菲佛（Cameron Pfiffer）。威尔是 DotTxt 的创始工程师，卡梅伦则是开发者关系工程师。他们将帮助像你这样的开发者构建可靠的结构化格式，并将其集成到 LLM 应用中。

谢谢安德鲁的介绍。在本课程中，你将首先从支持结构化响应的模型中获得结构化输出。你还将了解这种方法的局限性。为了解决其中一些限制，我们将使用 Instructor 这个开源库，它会重新提示模型，直到生成有效的 JSON 结构。

你还将学习约束解码的工作原理。这是 Outlines（我们将在本课程中使用的开源库）背后的核心概念。在学习本课程所有概念的同时，你将通过多个精彩案例进行实践，包括一个社交媒体分析智能体。该智能体能够读取用户帖子、识别情感倾向，并判断是否需要回复。若发现问题，它甚至能自动生成客户支持工单。

所有的 JSON 输出。在使用基于重试的方法后，你将学习如何通过 get-go 获取结构和输出，这是一个通过拦截模型分配给每个下一个 token 的概率（logits）在 token 级别约束模型输出的库——Outlines 会阻止任何不符合你定义的 schema 或格式的 token。这确保了无需重试即可获得有效输出。

在最后一课中，你将学习如何生成超越常规 JSON 的内容。例如，你将学会如何生成有效的电话号码、电子邮件地址，甚至是 ASCII 井字棋棋盘；基本上任何能用正则表达式表示的格式都可以生成。许多人参与了这个课程的创作。

我要感谢 DotTxt 的 Rémi Louf 和 Andrea Pessl。来自 DeepLearning.AI 的 Esmaeil Gargari 也为本课程做出了贡献。

第一课将介绍结构化输出生成。听起来很棒。让我们继续观看下一个视频，开始结构化之旅吧。

--------------

在本课程中，我们将概述结构化输出。了解它们的重要性以及生成这些输出的不同方法。你将看到结构化输出如何使基于 LLM 的软件开发更具可扩展性。你将学习各种结构化输出方法如何引导你从提示技巧转向真正的 AI 工程。让我们开始吧。

首先，我们需要明确什么是结构化输出。我们与一个 LLM 合作，通常该模型的输出是某种自由形式的文本。这种文本没有特定的结构，仅仅是文字而已。结构化输出意味着模型的输出遵循某种结构，在这里指的是 JSON 格式。JSON 是一种非常常见的结构化输出形式，但还有很多其他形式。当你使用结构化输出时，首先想到的一个明显问题是：为什么我们需要使用这些结构化输出呢？

如果你长期使用大语言模型，可能对常见的聊天界面非常熟悉。在这种模式下，模型与我们的互动方式就像人类在和我们对话。通常，助手会有一个提示，我们接着提出问题，然后得到回答。在这个例子中，我要求它对我收到的一条社交媒体消息做出回复，模型随后给出了答案。作为人类，我们很容易解析这些回复，理解模型在告诉我们什么，并利用这些信息。

让我们来看看在编程环境中如何实现这一点。在代码中使用大语言模型时，我们使用一种非常相似的界面，称为指令界面。这与聊天界面类似，但略有不同。因此，我们实际上是在向模型发送请求并获取响应。这与上一张幻灯片中的内容相同，只是格式不同。我们再次从模型获得响应。这使得我们能够在代码中与模型交互。但我们仍需处理模型返回的响应。如何解析模型返回的这些信息？作为人类，这很容易做到，但我们需要以编程方式实现。

让我们简单讨论一下如何用大语言模型构建系统，以便更好地理解结构化输出的必要性，以及它们如何帮助我们构建可扩展的软件。

让我们构建一个由 LLM 驱动的社交媒体代理。这个代理将获取我们在社交媒体上收到的回复，并自动生成回应，以实现社交媒体账号管理的自动化流程。我们设计了一个非常简单的应用架构：首先通过 LLM 接收预设提示词，然后附加用户消息，接着获取 LLM 生成的回复，最后通过社交媒体 API 将回复发布出去。但核心问题在于：如何处理这些原始数据？正如前文所述，我们需要建立数据提取机制。最基础的解决方案是直接添加解析层——这是使用 LLM 处理此类问题时常见的初级方案，即编写简单规则来解析输出答案。

遗憾的是，这极其耗时。如果你曾手动编写过类似的解析器，你会发现这需要花费大量时间，部分原因在于该过程极易出错。稍有不慎就可能遗漏细节、出现失误，或遇到意料之外的输出结果。想要准确无误地完成这项工作相当棘手。正因如此，这种方法的可扩展性较差。为了说明这一点，让我们看看智能体的另一个版本。

假设产品经理提出需求，要求我们修改社交媒体智能体——当用户回复属于投诉内容时，需要将该投诉转发至客户支持部门。我们不想让模型自动回复那些遇到问题的人。我们希望用人来确保我们喜欢这个回复。但对于其他所有评论或发给我们的消息，我们可以让代理继续回复它们。这需要对我们的流程进行简单的分支处理。这看起来很容易实现，但使用之前的技术，我们发现实际上会变得相当混乱。现在我们需要解析出它是否是投诉，以及如果需要的话，还要从回复本身解析出我们最初的内容。显然，这并不是一个可扩展的解决方案来使用 LLM 编写软件。

但如果我们能在输出中获得可预测的 JSON 呢？在这种情况下，实现这一点非常简单。我们有一个投诉字段和一个响应字段。如果投诉字段为真（我们可以像检查任何其他 JSON 一样进行检查），我们可以将其发送给客户支持。如果不是投诉，我们可以继续将响应传递给我们的社交媒体 API。API 可以轻松通过 JSON 提取这些信息。

接下来的问题是，我们如何获取结构化输出？开始使用结构化输出的最简单方法是利用你可能已经在使用的专有 API。每个推理提供商都提供了不同的结构化输出解决方案。我们将稍作讨论。其中一种方法称为基于逻辑的方法或受限解码。这种方法通过修改模型本身，使其仅生成符合结构要求的标记。我们将在课程末尾更详细地讨论这一点。

另一种常见做法称为函数调用或工具使用。在进行函数调用时，模型会获得一个可调用或响应的函数列表，这些通常以 JSON 格式处理。还有一种称为 JSON 模式的功能，即模型经过微调专门返回 JSON 数据，并在收到相应提示时执行。当然，背后可能还存在我们尚未理解的"魔法"。这正是专有 API 存在的问题之一——因为我们无法始终了解其后台运作机制。

现在让我们探讨使用专有 API 结构化输出的利弊：优势在于我们使用的是 JSON 格式，这非常便利。我们之前处理的是非结构化文本，而现在有了 JSON。在使用大语言模型创建可靠系统方面，这是一个巨大的飞跃。如果你已经在使用这些主要供应商之一，这些技术也很容易上手。如果你已经在应用程序中使用 OpenAI，那么添加对结构化的支持并不需要太多额外的工作或代码。另一个好处是这些供应商一直在研发新技术，提高输出的质量。从 OpenAI 获取 JSON 在过去一年里已经有了巨大的改进。

当然，这些方法也有一些缺点。主要问题之一是你的代码现在与特定的模型提供商紧密绑定。这意味着更换提供商可能需要进行大规模重构——例如，如果你的代码原本针对 OpenAI 编写，而你想尝试改用 Instructor，就可能需要重写大量结构化代码。另一个问题是结果不一致性。当然这取决于提供商，但某些提供商返回的结构并不总是符合预期，这对于构建可扩展系统可能造成严重障碍。第三个问题是输出质量影响不明确。许多评估案例表明，有开发者发现使用某些提供商的结构化输出反而会降低这些评估任务的性能表现。这并不是普遍现象，也未必得到证实，但这是人们一直怀疑的问题，而且确实可能是个问题。此外，你能使用的数据结构类型也很有限。通常你只能使用标准 JSON 的一个子集。你可以获取名称和字段，但无法对字段进行详细的正则表达式验证，比如确保日期符合特定格式。

解决这些问题的一个方法是使用所谓的 "re-promoting"。专有模型只能与特定 API 配合使用，而 re-promoting 库的设计初衷是与任何主流大语言模型供应商兼容。这类工具的例子包括 Instructor 和 LangChain。

re-promoting 非常有趣。它的工作方式是，我们首先向 LLM 进行一次常规传递，提供一个提示，模型就会产生一个输出。此外，我们还会为模型提供一个验证器。这只是一个关于我们期望 JSON 看起来是什么样子的描述。如果模型的输出符合我们的验证器，数据就会直接发送给我们。这很棒，但如果输出有问题怎么办？在这种情况下，我们看到生成了一个尖括号，而我们期望的是一个花括号。这不是一个有效的输出。

re-promoting 库会自动将解析失败的原因附加到提示中，并再次尝试。这种方法有很多优点。一是现在你可以使用任何 LLM API。这在开发可复用的优秀软件方面是一个巨大的进步。这使得你编写的代码结构可以跨 API 使用。通常，更换提供商只需要更改密钥和提供商名称。与大多数专有提供商相比，我们在结构上也获得了更大的灵活性。所以这仍然只是JSON格式，但我们可以对字段使用正则表达式来强制约束条件，比如特定的日期格式，或者确保用户的电子邮件地址是有效的。

这种方法也有一些缺点。最大的缺点是重试可能会非常昂贵，无论是在金钱还是时间方面。对于许多 LLM 应用程序的开发者来说，时间是一个更大的考虑因素，因为你让用户等待结果。由此而来，甚至没有成功的保证。如果在一定的重试次数后，库没有成功，它就会简单地失败。再次强调，虽然我们对结构有更多的控制，但我们仍然只有 JSON 作为可行的输出。

结构生成，也称为约束解码，是一种直接与模型合作以获得我们想要的输出的方法。我们使用“结构化生成”这一术语，是因为我们实际上是在控制 token 的生成过程，以最终获得结构化的输出结果。目前有许多支持这一功能的库，包括 .txt的 Outlines、SGLang、微软的 Guidance 以及 XGrammar 等。

为了理解这些工具的工作原理，让我们先简单回顾一下 LLM 是如何生成标记的。整个过程始于一个 prompt，这个提示会被转换成一连串的标记，然后一次性输入到大语言模型中进行处理。然后，LLM 将这些标记转换为下一个标记可能出现的概率分布。这些权重被称为 logits，用于描述每个给定标记出现的可能性。接着，系统会根据这些权重采样一个标记，并将其附加到提示词之后。持续进行标记采样，直到遇到语句结束标记为止——这就是结构化生成背后的运作原理。

假设我们要将结构定义为仅允许字符按顺序排列的字符串？我们将在课程稍后部分讨论如何具体实现这一点。但目前，我们暂且假设这就是现有的约束条件。符合此结构的有效字符串包括 ABC、AABC、AC 和 BBB。所有这些例子中，每个字母都按顺序排列。无效的例子则有 BAC 和 CCCA。这两个例子中的字符顺序混乱，因此不符合我们结构的规则。现在让我们一步步解析其实际运作原理。请看第一个字符的 logits 分布情况——实际上此时没有任何约束，因为任何字符都可以作为有效的首字符。假设我们采样得到字母 b。当 b 被追加到提示词后，系统会重新运行流程并生成新的 logits。但这些 logits 并不符合我们的约束条件，因为其中有部分概率分配给了字母 A。根据结构规则，A 在此处是无效字符。结构化生成就是这样运作的。

实际上，它会修改那些逻辑值，移除无效的部分，并对允许的部分重新加权。我们再次激活另一个标记，一个 C，再次得到新的逻辑值，并再次面临同样的问题。现在，A 和 B在序列中不再是有效的下一个标记。因此，结构化生成再次简单地将这些从可能的采样中移除。读取过程会规范化概率，我们继续这个过程，直到最终生成一个保证符合我们结构规则的字符串。


让我们来谈谈结构化生成的优缺点。这项技术有很多版本。它可以与任何现有的开放 LLM 一起使用。因此，如果你使用 HuggingFace 的模型（包括视觉模型等任何模型），它们都能与结构化生成协同工作。由于我们直接与模型交互，处理速度极快。推理过程中的成本几乎为零。事实上，有研究表明，通过利用标记中固有的可跳过结构，它甚至能缩短推理时间。它能提供更高质量的结果。详情可参阅我们的博客。但经过反复评估对比结构化生成与非结构化生成时，我们发现结构化生成能显著提升基准测试表现。

在资源受限的环境中，这种方法同样表现优异。由于它极为轻量且高效，即使在搭载微型大语言模型的极小设备上，你依然可以采用结构化生成技术来确保模型输出质量。我们能够生成极其丰富的结构类型——除了 JSON 格式外，还包括正则表达式，甚至是语法完全正确的代码。

不过结构化生成也存在一个局限：由于我们直接操作 logits，这意味着你必须拥有对这些 logits 的控制权。所以，这意味着你要么使用开源模型，要么自己托管专有模型并控制输出概率。

我们讨论的内容也可以这样理解：这是从提示词破解到 AI 工程的演进之路。当我们仅使用基于聊天的界面时，本质上是在进行最典型的提示词破解。我们不断修改发送给大语言模型的信息，解读返回结果，并持续迭代这个过程直至获得满意输出。但这种方式既不可扩展，甚至难以复现。因此，我们离真正的软件还有很远的距离。转向专有的 Json API 为我们打开了全新的可能性世界。我们现在可以开始编写真正的软件了。我们的模型能够提供可预测的响应，这使得我们可以将使用 LLM 的代码集成到软件的其他部分中。

但有一个问题，我们大部分实现这一功能的代码都依赖于单一的供应商。通过重新提示库，我们可以创建可重用的软件库，这些库可以使用各种不同的 LLM 供应商。这是在创建可扩展的、与 LLM 协同工作的软件方面迈出的巨大一步。然而，我们的代码和模型本身之间仍然存在很大的差距。我们正依靠这种巧妙但有限的提示技巧来获取所需结果，而这正是结构化生成真正迈向人工智能工程的地方。我们直接与模型协作，精准获取预期成果。我们能编写出易于理解、扩展和改进的代码。

本节课我们讲解了结构化输出的基础知识，了解了其价值所在以及如何助力构建可扩展的软件系统。我们还探讨了各类结构化输出形式，包括供应商提供的 API 接口、重新提示库以及结构化生成技术。最后，我们见证了这些技术如何协同作用，使得基于大语言模型开发卓越软件成为可能。

下节课中，你将通过使用 OpenAI 结构化输出 API 构建社交媒体代理程序，亲身体验这套机制的实际运作。敬请期待，下节课见。

--------------

在本课程中，你将使用 OpenAI 的结构化输出 API 构建一个简单的社交媒体管理代理。你还将学习使用 Pydantic 指定所需输出结构的基础知识。好了，让我们深入探讨。结构化输出通常使用所谓的 JSON 模式来强制执行输出结构。JSON 模式或标准描述了 JSON 消息的形状。因此，这些模式提供了结构化输出所需的类型信息。

例如，这里有一个包含多莉·帕顿（Dolly Parton）姓名、年龄和电子邮件地址的 JSON 消息。从这里我们可以看到，左侧是键（如姓名、年龄和电子邮件），右侧是对应的值（多莉·帕顿、79岁和 dolly@test.com）。让我们来看一个描述像多莉·帕顿这样的人的 JSON 模式示例。

这里我们同样可以看到这也是 JSON 格式，包含了我们所有的键，比如姓名和电子邮件，以及这些键的类型。例如，title 等于 name，type 等于 string。让我们稍微谈谈 Pydantic。手动维护 JSON 模式既困难又容易出错。




So we can see here we have keys
on the left such as name, age and email followed by their values
on the right, Dolly Parton, aged 79, and dolly@test.com. Let's take a look at an example of a Json schema
that describes a person like Dolly Parton. Here we can see this is also Json
and includes all of our keys such as name is an email
as well as the types of those keys. Such as title equals
name and type equals string. Let's talk a little bit about pydantic. Maintaining JSON schemas
by hand is difficult and prone to error. 



As you saw, they're pretty complicated and for anything that's non-trivial
they can be an enormous pain to work with. So, AI engineers often use pydantic
to describe model output structure. Pydantic is an open source
data validation library where you can flexibly define data
structures for use in your application. It's also feature-rich. You should go check out all the things
that it supports in the documentation. We won't be able to cover
all of the tools that it has available. Using pydantic makes language
model outputs programmable. As we'll see in just a second. So here is how you would write
a user class inside of pydantic. Inheriting from base model is all
you need to make a user a pydantic class. Next, you're going to use type annotations
to define the structure that your model should generate. For example,
name can be any string like Dolly Parton. Age can be any integer like 79, and email
can be either nothing or a string. Note, however, that OpenAI does not allow you
to enforce the formatting of email or i.e. always in the form of test, at test.com . Outlines and Instructor,
and a few other tools do. So let's build a social media customer
support agent. We have a lot of users who like to tweet at us
saying at Tech Corp your app is amazing. The new design is perfect. And then we're going to put this inside
of our customer support agent, and then we're going to output some structured
data telling us what product it's about, what the sentiment is from the user,
whether or not we should respond, and a support ticket
if anything happens to be broken. All right. So, let's go to the code. We're going to start
with a little bit of housekeeping here. So these two lines just filter out
some warnings that we may not want to see while we're learning. And here's how you load the OpenAI API key
that we're going to use. These three lines
go and set up your OpenAI API key that DeepLearning.AI
has configured. Here's how you initialize
the OpenAI client using the OpenAI key. So let's see an example
of how we define structure with pydantic. We'll start by pasting in this user class
that we added from the slides. So let's generate a user object
using OpenAI. Here, we're going to use the client
that we defined. And then we're going to call dot Beta dot
chat dot completions dot parse. We're going to select our model, here we're going to use GPT-4o-mini. Then we're going to provide
some messages. Here, the first message is the system prompt
which says you're a helpful assistant. And then we're going to pass the user
prompt that you might type into ChatGPT. And then we're just going to say
make up a user. Then in response format
we specify the user class we just made. When we run this, it'll take a second. And then OpenAI will provide us an user. Let's look at this user
by taking it out of our completion object. Normally, OpenAI returns a list of choices
when you request multiple values. But here, we've only requested
one completion. Then, we're going to extract the message. And then the parse object inside of dot parsed. Let me run this. We see
we got a user named John Doe, age 30, with an email John Doe at example.com. Let's build
our social media management agent. Anytime you're working with structured
output, you typically define the structure that you want to see before you start
working with the rest of your code. In this case, we've added a mention
which is a pydantic class that has product and sentiment. And you'll see here
that these are literals, which means that the language model has to choose
between app, website or not applicable. Same thing with sentiment
has to be positive, negative or neutral. Next, the model can choose to respond to the user
so it is allowed to choose true or false. whether or not
the company needs to respond. And then it has a response
which is an optional string. So that can be either a string or nothing. This is the response
that we would provide to the user. Lastly, if the model determines
that a support ticket needs to be open, the model can write a description
for the developers. And this may also be either
nothing or string. Now, let's give ourselves
some example data. I'm going to call this the mentions list
and it includes a few examples. So we have a positive review of our app. They like the design. Next, there's a user
letting us know that the website is down and that we need to fix it. And then lastly we have "Hey at Tech Corp
you're so evil", which we probably don't want to respond to though. We might wish to generally address
concerns about our evilness. So here's how you add an analyze
mentioned function. This is how we take the user messages
that they sent to us, and then we construct this mention object
that we just designed. So here we're going to define a function
called analyze mention. We're going to provide the post
that the user sent to us. And then an optional keyword argument
for the personality, which we'll talk about in just a second. By default, this is friendly. As before we're going to use dot
beta, dot chat, dot completions, dot parse. Select our model
and provide a few messages. In this case, we've provided the system prompt
which says what we expect from the model. We want the model to provide the product mention, the mentioned sentiment, whether or not to respond. We let the model know that it should not respond
to inform inflammatory messages or bait, and we ask it for a customized message
to send to the user. Lastly, we ask the model
to provide an optional support ticket description
to upload to our ticket management system. Then, finally we let the model know
what its personality is. And then inside of the user prompt
we just add the mention. Lastly, we request a mention object. Then we're just going to return
the parsed object at the end. So let's go ahead and use this
analyze mention function. We're going to use the first mention
which if you remember was the positive review about our app. We're going to pass that to the are analyze mention function
and put it in processed mention. When we run this we'll see. Hey Techcorp, your app so amazing. And then we see the models
provided us with product app. So it's identifying that it's about
the app as well as positive sentiment. The model has flagged
that there needs to be a response. And then it includes a response that
we can choose to send back to the user. We're thrilled to hear that
you love the new design. And then lastly, there's no support ticket because there's no obvious issues
to address with our software. If you would like to adjust the personality of this model,
you can simply change the personality. I've changed the personality here
to be rude. In which case the model says
thanks for the praise. We're glad you love the new design,
but don't get too comfy. We're constantly improving, which could be rude or try playing around
with the different personalities. And for those who are curious, the underlying content that was generated
by the model is just Json. So when we print these out
using pedantic model dump JSON method, we can see what was generated
by the language model before it was passed into the pydantic object
that we have. In this case, we can see the raw text. Here's an example of something
you can play with on your own. In this example, you're going to create a user post
which has just a message inside of it. And then you're going to ask a language
model to act as if it was a person, and then provide tweets to techcorp. We describe roughly what's going on there. And then we can see the output. Hey, just wanted to shout out to TechCorp
for their amazing app. Now, we can provide that output
from the user and put it back into our analyze
mentioned function. When we see this, we're going to get app
positive sentiment. Thanks so much for your kind words. And of course the language model just seems to like
saying thanks so much for your kind words. So try adjusting this user post class.
In this example, I've added a user post with extras class
which is also a pydantic class that includes a couple of extra fields such as user mood which has to be between
awful, bad, and evil. Product also has to be app website
not applicable as in the mentioned class. And then we can also include
a internal monologue which includes a list of strings. Now when we run this,
we'll get user post with extras. User mood is awful. Sentiment is negative. Can't believe how buggy the app is, etc. And then the user just says it's
very frustrated with the latest version of your app. And then of course we can pass this back
into our social media analyzer and see what it says. And as we can see
here, the social media analysis bot detects that it's about the app,
that the sentiment is negative. Provides an example response
that hopefully placate the user. And then adds a support ticket
where we note that the user has reported constant crashes
with the latest version of the app. So the purpose of structured output
is to turn language model responses into programable data. So here's an example of a way that we can
program with our language model output. Here, we're going to construct a list to
contain all of the rows that we generate. We're going to iterate across dimensions
that our users have provided. We're going to analyze those mentioned
by creating these mentioned objects. If there's a response that is needed,
we can print out that we're responding to that user message
using the standard attributes from any Python class,
such as processed mention dot sentiment. If there's a support ticket,
we can print that out as well. Let's take a look
at how we program with our mentions. Here, we're going to iterate through each
of the mentions that we have available. We're going to call the LLM
to get a mention object that we can program with, right here. Then we're going to print out some
information about that process mentioned that's in the helper file. If you want to take a look. We're going to
convert our process data to a dictionary using pydantic tools in this case
the model dump method. Next, we're going to store this
original message in a data frame row. So we'll basically just be adding it
to this processed dict object. Then we're going to add that dictionary
to our list of rows. And then print out a blank row
so that we can read everything that gets printed out. So let's run
this and see what the result looks like. As we go through, we can see the model
responding to positive.= app feedback.
The negative website feedback. It will describe its responses and add support tickets if relevant
such as right here. The user has reported our website is down. And then of course
the user who has specified hey, at Tech Corp, you're so evil,
we've chosen not to respond to it. And then lastly, the advantage
of working with structured output is that you know,
that it follows an exact format, which means that you can put it into various other data formats
such as a data frame. So in here we're going to import pandas and construct
a data frame using the dictionaries. Here, we can see that
it's extremely simple. We have the product, the website the sentiment, whether it needs
a response, the original mention, and whatever other features
we might wish to add in the future. In this example, you've learned
how to define a model's output structure using pydantic
how to use OpenAI's structured output API and basic program
with language model outputs. Next, you're going to use Instructor
to get structured output from various other inference providers.