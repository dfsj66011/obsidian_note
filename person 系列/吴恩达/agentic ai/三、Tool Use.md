
## 3.1 What are tools？

在本模块中，您将学习 LLM 的工具使用，这意味着让您的 LLM 自行决定何时需要调用某个函数来执行操作、收集信息或完成其他任务。就像我们人类借助工具能比徒手完成更多事情一样，LLM 通过调用工具也能实现更强大的功能。不过我们提供的不是锤子、扳手或钳子这类实体工具，而是可供 LLM 调用的函数——正是这些工具让 LLM 具备了更广泛的能力。

让我们来看看。如果你问一个可能几个月前训练过的 LLM，现在几点钟了？这个训练过的模型并不知道确切的时间，所以它应该会回答：抱歉，我无法获取当前时间。但如果你写一个函数，并让大语言模型能够访问这个函数，它就能给出更有用的答案。当我们让 LLM 调用函数，或者更准确地说，让 LLM 请求调用函数时，这就是我们所说的工具使用。这些工具其实就是我们提供给 LLM 的函数，LLM 可以请求调用它们。

具体来说，工具使用是这样运作的：在这个例子中，我将把上一张幻灯片中展示的`getCurrentTime` 函数提供给 LLM。当你提问"现在几点"时，LLM 可以决定调用 `getCurrentTime` 函数。该函数会返回当前时间，这个时间随后会被反馈到对话历史中，最终 LLM 可以输出，比如说，"下午3 点 20 分"。因此，步骤的顺序是这样的：首先有一个输入提示。在这个例子中，LLM 会查看可用的工具集（虽然这个例子中只有一个工具），然后决定调用这个工具。这个工具是一个函数，它会返回一个值，这个值会被反馈给 LLM，最后 LLM 生成它的输出。

现在，工具使用的一个重要方面是，我们可以让 LLM 自行决定是否使用任何工具。因此，在同样的设置下，如果我问它 “绿茶中含有多少咖啡因”，LLM 不需要知道当前时间就能回答这个问题，所以它可以直接生成答案——绿茶通常含有这么多咖啡因——而且无需调用`getCurrentTime` 函数。

在我的幻灯片中，我会在 LLM 上方使用这种带虚线框的符号来表示，我们正在为 LLM 提供一组工具，供 LLM 在认为合适时选择使用。

这与你在之前视频中看到的一些例子不同，那些例子中，我作为开发者硬编码了一些操作，比如在研究代理的某个固定节点总是执行网页搜索。相比之下，`getCurrentTime` 函数调用并非硬编码——是否调用该函数完全由 LLM 自主决定。我们将继续使用这种虚线框标注法，来表示我们向大语言模型提供一个或多个工具时，由模型自行决定是否调用（以及调用哪些）工具。

以下是一些工具使用可能帮助基于 LLM 的应用程序生成更好答案的例子。如果你问它：“你能找到加利福尼亚州山景城附近的一些意大利餐厅吗？”如果它有一个网络搜索工具，那么 LLM 可能会选择调用网络搜索引擎进行查询，搜索“加利福尼亚州山景城附近的餐厅”，并使用获取的结果来生成输出。

或者，如果你经营一家零售店，并且希望能够回答诸如“给我看看买了白色太阳镜的顾客”这样的问题，如果你的 LLM 被授权访问查询数据库工具，那么它可能会查找销售表格，看看哪些条目售出了一副白色太阳镜，然后利用这些信息生成输出。

最后，如果你想进行利率计算，比如我存入 500 美元，10 年后，利率为5%，我会得到多少钱？如果你恰好有一个利息计算工具，那么它可以调用利息计算功能来计算。或者你会发现，之后的一个方法是让 LLM 编写代码，比如像这样写一个数学表达式，然后进行计算，这将是让 LLM 计算出正确答案的另一种方式。

因此，作为开发者，你需要仔细思考你希望应用程序真正实现哪些功能，然后创建必要的函数或工具，使这些功能可供 LLM 使用，让它能够选择合适的工具来完成诸如餐厅推荐、零售问题解答或财务助手等任务。根据你的应用程序需求，你可能需要实现并提供不同的工具给你的 LLM。

到目前为止，我们讨论的大多数示例只向 LLM 提供了一种工具或一个功能。但在许多实际应用中，你可能需要为 LLM 提供多种工具或功能，让它自行选择调用哪一个。例如，如果你正在开发一个日历助手代理，你可能希望它能够处理类似这样的请求：请在我的日历中查找周四的空闲时段，并为我和爱丽丝安排一个约会。

因此，在这个例子中，我们可以向 LLM 提供一个工具或功能来安排预约，即发送日历邀请，检查日历以查看我何时有空，以及如果它想取消现有的日历条目，还可以删除预约。根据这组指令，LLM 首先会决定在可用的不同工具中，它应该使用的第一个工具可能是检查日历。

因此，调用一个检查日历的函数，它会返回我周四的空闲时间。根据这些反馈给 LLM 的信息，它可以决定下一步是选择一个时间段，比如下午 3 点，然后调用预约函数，向 Alice 发送日历邀请，并将其添加到我的日历中。

最终输出的结果（希望是日历条目已成功发送的确认信息）会反馈给 LLM，然后大语言模型可能会告诉我："您与 Alice 的约会已安排在周四下午3点。" 能够让你的大语言模型使用工具是一件非常重要的事情。这将使你的应用程序变得更加强大。在下一个视频中，我们将看看如何编写函数，如何创建工具，然后让它们可供你的大语言模型使用。让我们继续观看下一个视频。

## 3.2 Creating a Tool

LLM 决定调用函数的过程起初可能显得有些神秘，毕竟 LLM 的训练目标只是生成输出文本或文本标记。那么这究竟是如何实现的呢？在本视频中，我将带您逐步剖析 LLM 真正实现函数调用的完整流程。让我们开始探索吧。工具其实就是 LLM 可以请求执行的代码或函数，比如我们在前一个视频中看到的这个 getCurrentTime 函数。

如今，领先的 LLMs 都已直接训练为能够使用工具。但我想带你们看看，如果必须自己编写提示词来告诉模型何时使用工具会是什么样子——这正是在 *LLMs 被直接训练使用工具之前的早期阶段*，我们不得不采用的方式。尽管 *如今我们不再完全沿用这种方法*，但这有望帮助你更好地理解整个过程。我们将在下一个视频中介绍更现代的语法结构。

如果你已经实现了这个获取当前时间的函数 getCurrentTime，那么为了将这个工具提供给 LLM，你可能会编写如下提示词。你可以这样告诉 LLM，你有一个名为 getCurrentTime 的工具可以使用。要使用它，我希望你打印出以下文本。先打印全大写的 FUNCTION，然后打印 getCurrentTime。如果我看到这段文字，全大写的 FUNCTION 后面跟着 getCurrentTime，我就知道你是想让我为你调用 getCurrentTime 函数。当用户问现在几点时，LLM 就会意识到它需要调用或请求调用  getCurrentTime 函数。于是 LLM 就会输出它被告知的内容，输出全大写的 FUNCTION: getCurrentTime。

现在，我必须编写代码来查看 LLM 的输出，看看是否有这个全大写的函数。如果有，那么我需要提取这个 getCurrentTime 的参数，以确定 LLM 想要调用哪个函数。然后，我需要编写代码来实际调用 getCurrentTime 函数，并提取输出，比如说，上午 8 点。接着，是开发者编写的代码，也就是我的代码，必须将上午 8 点这个时间反馈给 LLM，作为对话历史的一部分。

当然，对话历史包括最初的用户提示、请求是一个函数调用等事实。最后，LLM 知道之前发生了什么：用户问了一个问题，请求了一个函数调用，然后我调用了该函数并返回了上午 8 点。最终，LLM 可以查看所有这些信息并生成最终响应，即现在是上午 8 点。

*需要明确的是，为了调用一个函数， LLM 并不直接调用该函数。* 相反，它会输出特定格式的内容，比如这样，告诉我需要调用 LLM 的函数，然后告诉 LLM 我所请求函数的输出结果。在这个例子中，我们只给 LLM 提供了一个函数，但你可以想象，如果我们给它三四个函数，我们可以让它用全大写字母输出函数名，然后是它想要调用的函数名称，甚至可能包括这些函数的一些参数。实际上，现在让我们来看一个稍微复杂一点的例子，其中 getCurrentTime 函数接受一个参数，用于指定你希望获取当前时间的时区。

在这个第二个例子中，我编写了一个函数来获取指定时区的当前时间，这里的时区是 getCurrentTime 函数的输入参数。为了让 LLM 能够使用这个工具来回答诸如"新西兰现在几点？"这样的问题——因为我的回答就在那里，所以在打电话给她之前，我确实会查一下新西兰的时间。为了让 LLM 能够使用这个工具，你可以修改系统提示，告诉它可以使用 getCurrentTime 工具来查询特定时区的时间。使用时，我会输入 getCurrentTime，然后加上时区信息。这是一个简化的提示。实际操作中，你可能会在提示中加入更多细节，比如这个函数是什么、如何使用等等。

在这个例子中，LLM 会意识到它需要获取新西兰的时间，因此会生成类似这样的输出：FUNCTION: getCurrentTime(Pacific/Auckland)。这是新西兰的时区，因为奥克兰是新西兰的主要城市。然后我需要编写代码来搜索 LLM 输出中是否出现了这个全大写的函数，如果出现了，那么我需要提取出这个函数来调用。最后，我将使用 LLM 生成的参数（即 Pacific/Auckland）调用 getCurrentTime 函数，可能会返回凌晨 4 点。然后像往常一样，我将这个输入给 LLM，LLM 输出结果。现在是新西兰凌晨 4 点。

总结一下，以下是让 LLM 使用工具的过程。首先，你需要向 LLM 提供工具，实现函数，然后告诉 LLM 这个工具是可用的。当 LLM 决定调用一个工具时，它会生成一个特定的输出，让你知道需要为 LLM 调用这个函数。然后你调用函数，获取其输出，将刚刚调用的函数的输出反馈给 LLM，LLM 接着利用这个输出继续执行它决定要做的下一步操作。在我们这个视频的例子中，LLM 只是生成了最终的输出，但有时它甚至可能决定下一步是调用另一个工具，这个过程会继续下去。

现在，我们发现这种全大写的函数语法有点笨拙。这是我们过去在 LLM 尚未被训练出自主调用工具能力时的做法。对于现代 LLM 而言，你不再需要指示它输出全大写函数、再搜索全大写函数等等。相反，LLM 经过训练会使用特定语法来清晰地表达何时需要调用工具。在下一个视频中，我想向大家展示现代 LLM 请求调用工具的实际语法是什么样的。让我们继续观看下一个视频。

## 3.3 Tool syntax

让我们来看看如何编写代码让你的大语言模型调用工具。这是我们之前不带时区参数的 getCurrentTime 函数。我来演示一下如何使用 aisuite 开源库让你的大语言模型调用工具。顺便说一句，从技术上讲，正如你在上一个视频中看到的，大语言模型并不会直接调用工具，它只是请求你去调用工具。但在开发自动化工作流程的程序员群体中，我们经常会说"大语言模型调用工具"，虽然技术上并不准确，但这种说法更简洁。

这里的语法与调用这些 LLM 的 OpenAI 语法非常相似，不同之处在于这里我使用的是 aisuite 库。这是一个开源包，由我和一些朋友共同开发，可以轻松调用多个 LLM 提供商。所以代码语法看起来可能有点复杂，但不用担心。在代码实验室中你会看到更多相关内容。简单来说，这与 OpenAI 的语法非常相似：你会看到类似 response = client.chat.completions.create 这样的代码，然后选择模型（这里我们使用的是 OpenAI 的 GPT-4o 模型），messages=messages（假设你已经将要传递给 LLM 的消息放入了一个数组中），接着是 tools=，后面列出你希望 LLM 能够使用的工具列表。

在这种情况下，只有一个工具，即获取当前时间，然后不必过于担心最大轮次参数。这个参数的存在是因为在一个工具调用返回后，LLM 可能会决定调用另一个工具，而在那个工具调用返回后，LLM 可能又会决定调用另一个工具。因此，最大轮次只是一个上限，表示你希望 LLM 在停止之前连续请求一个又一个工具的次数，以避免可能的无限循环。实际上，除非你的代码在做一些异常雄心勃勃的事情，否则几乎不会达到这个限制。

所以我不会太担心最大轮次参数。通常我直接设为 5，但实际上这个设置影响不大。事实证明，在 aisuite 中，get_current_time 函数会自动以恰当的方式向大语言模型描述，使其知道何时调用该功能。因此你无需手动编写冗长的提示词来告知大语言模型——在 aisuite 中，这种语法会自动完成这个工作。为了让这个过程看起来不那么神秘，实际上它是通过查看 "get_current_time" 函数关联的注释字符串，来确定如何向大语言模型描述这个功能的。 

为了说明其工作原理，这里再次展示该函数，以及使用 aisuite 调用 LLM 的代码片段。在后台运行过程中，系统会创建一个详细描述该函数的 JSON 模式结构。右侧显示的内容实际上是传递给 LLM 的信息——具体会提取函数名称 "get_current_time"，同时从文档字符串中提取函数功能描述，这些信息能让 LLM 理解该函数的作用并决定何时调用它。

有些 API 需要你手动构建这个 JSON 模式，然后将该 JSON 模式传递给 LLM，但 aisuite 包会自动为你完成这一过程。举个稍微复杂些的例子，如果你有一个更复杂的获取当前时间的工具，它还有一个输入时区参数，那么 aisuite 会创建这个更复杂的 JSON 模式，像之前一样，它会提取函数名称（即获取当前时间），从文档字符串中提取描述，然后还会识别参数是什么，并根据左侧显示的文档向 LLM 描述这些参数。

这样在生成调用工具的函数参数时，它就知道应该是类似 "America/New York"、"Pacific/Auckland" 或其他时区的格式。如果你在左下角执行这段代码，它将使用 OpenAI 的 GPT-4o 模型，判断大语言模型是否需要调用函数。如果需要，就会调用函数获取输出结果，再反馈给大语言模型，最多重复五次交互后返回最终响应。

请注意，如果 LLM 请求调用获取当前时间函数，aisuite 或此客户端将为您调用获取当前时间，因此您无需自己显式执行此操作。所有这些都在您需要编写的这个单一函数调用中完成。只需注意，其他一些 LLM 接口的实现需要您手动执行此步骤，但对于这个特定的包，所有这些都封装在这个 client.chat.completions.create 函数调用中。

现在你已经了解了如何让 LLM 调用函数，希望你能在实验环节玩得开心。当你给 LLM 提供几个函数后，它会决定采取行动、获取更多信息来满足你的需求，这种感觉真的很神奇。如果你还没尝试过，相信你会发现这非常酷。在所有可以给 LLM 的工具中，有一个比较特别，那就是代码执行工具。事实证明它非常强大。如果你告诉 LLM 它可以编写代码，并且你有一个工具可以帮它执行这些代码，因为代码能做很多事情。我们让 LLM 灵活地编写并执行代码，这成了赋予 LLM 的一个极其强大的工具。所以代码执行很特别。让我们进入下一个视频，讨论 LLM 的代码执行工具。


## 3.4 Code execution

In a few agentic applications I've worked on, I gave the LLM the option to write code to then carry out the task I wanted it to. And I've been a few times now, I've been really surprised and delighted by the cleverness of the code solutions it generated in order to solve various tasks for me. So if you haven't used code execution much, I think you might be surprised and delighted at what this will let your LLM applications do. Let's take a look. Let's take an example of building an application that can input math word problems and solve them for you. So you might create tools that add numbers, subtract numbers, multiply numbers, and divide numbers. And if someone says, please add 13.2 plus 18.9, then it triggers the add tool and then it gets you the right answer. But what if someone now types in, what is the square root of two? Well, one thing you could do is write a new tool for a square root, but then maybe some new thing is needed to carry out exponentiation. And in fact, if you look at the number of buttons on your modern scientific calculator, are you going to create a separate tool for every one of these buttons and the many more things that we would want to do in math calculation? So instead of trying to implement one tool after another, a different approach is to let it write and execute code. To tell the LLM to write code, you might write a prompt like this. Write code to solve the user's query. Return your answer as Python code delimited with execute Python and closing execute Python tags. So given a query like what is the square root of two, the LLM might generate outputs like this. You can then use pattern matching, for example, a regular expression to look for the start and end execute Python tags and extract the code in between. So here you get these two lines of code shown in the green box, and you can then execute this code for the LLM and get the output, in this case, 1.4142 and so on. Lastly, this numerical answer is then passed back to the LLM and it can write a nicely formatted answer to the original question. There are a few different ways you can carry out the code execution step for the LLM. One is to use Python's exec function. This is a built-in Python function which will execute whatever code you pass in. And this is very powerful for your LLM to really write code and get you to execute that code, although there are some security implications which we'll see later in this video. And then there are also some tools that will let you run the code in a safer sandbox environment. And of course, square root of two is a relatively simple example. An LLM can also accurately write code to, for example, do interest calculations and solve much harder math calculations than this. One refinement to this idea, which you sort of saw in our section on reflection, is that if code execution fails, so if for some reason the LLM had generated code that wasn't quite correct, then passing that error message back to the LLM to let it reflect and maybe revise this code and try another one or two times. That can sometimes also allow it to get a more accurate answer. Now, running arbitrary code that an LLM generates does have a small chance of causing something bad to happen. Recently, one of my team members was using a highly agentic coder and it actually chose to remove star.py within a project directory. So this is actually a real example. And eventually that agentic coder did apologize. It said, yes, that's actually right, that was an incredibly stupid mistake. I guess I was glad that this agentic coder was really sorry, but I already deleted a bunch of Python files. Unfortunately, the team member had it backed up on GitHub repo, so there was no real harm done, but it would have been not great if this arbitrary code, which made the mistake of deleting a bunch of files, had been executed without the backup. So the best practice for code execution is to run it inside a sandbox environment. In practice, the risk for any single line of code is not that high. So if I'm being candid, many developers will execute code from the LLM without too much checking. But if you want to be a bit safer, then the best practice is to create a sandbox so that if an LLM generates bad code, there's a lower risk of data loss or leakage of sensitive data and so on. So sandbox environments like Docker or E2B as a lightweight sandbox environment can reduce the risk of arbitrary codes being executed in a way that damages your system or your environment. It turns out that code execution is so important that a lot of trainers of LLMs actually do special work to make sure that code execution works well on their applications. But I hope that as you add this as one more tool for you to potentially offer to LLMs or let you make your applications much more powerful. So far and what we've discussed, you have to create tools and make them available one at a time to your LLM. It turns out that many different teams are building similar tools and having to do all this work of building functions and making them available to the OMs. But there is recently a new standard called MCP, Model Context Protocol, that's making it much easier for developers to get access to a huge set of tools for LLMs to use. This is an important protocol that more and more teams are using to develop LLM based applications. Let's go learn about MCP in the next video.

