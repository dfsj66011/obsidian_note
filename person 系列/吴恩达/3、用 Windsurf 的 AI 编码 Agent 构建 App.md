
https://www.deeplearning.ai/short-courses/build-apps-with-windsurfs-ai-coding-agents/


既然你已经更深入地了解了代理的实际运作，现在让我们学习几个心智模型，用来剖析和理解代理背后的工作原理。我们开始吧。本节课的主要目的是理解Cascade这个协作代理的核心组件为何如此强大，而不依赖于所有用户体验和附带的花哨功能。

许多人可能都见过这个经典代理循环的某种版本。本质上，这个代理系统会从开发者那里获取一些输入提示。然后它会访问一个充当大脑的大型语言模型，该模型会基于输入提示进行推理，从代理所拥有的各种工具中判断应该使用哪个工具。这个工具可能是诸如grep或嵌入搜索之类的搜索工具。

它可以是用来编辑文件的工具，也可以是用来建议终端命令的工具。你在演示中看到了所有这些工具。然后，在工具采取一些行动后，大型语言模型会再次进行推理，好吧，根据输入和刚刚调用的工具，我的下一步行动应该是什么？应该是调用另一个工具吗？在这种情况下，它会继续这个循环，直到最终那个充当大脑的大型语言模型决定，嘿，我们不需要再调用工具了。

是时候进入最终状态了，我的操作可以返回给用户了。这就是经典的智能体循环，其中大型语言模型被用作工具调用的推理代理。此时可以清晰地看到，一个智能体系统存在两个重要组成部分。首先是工具。可以采取的所有操作有哪些，以及这些工具在每个步骤中的威力如何。另一半则是推理模型。

而如今，决定调用哪些工具、如何调用这些工具（即向这些工具输入什么信息），在综合考虑大型语言模型掌握的所有相关信息后，最先进的做法是使用OpenAI和Anthropic等模型提供商提供的这些通用基础模型。但这些模型实际上并未涵盖所有可用于改善像Cascade这样的智能代理体验的维度。我想提到的第一个维度是情境感知的概念。

我们之前稍微提到过这一点，当时我们讨论了如何通过改善知识的获取来提高和巩固人工智能系统的结果。但有几种不同的方式来思考问题的上下文感知。首先是信息来源。对于手头的任务来说，所有相关的知识来源是什么？比如编写新代码这样的任务，当然包括私有代码库中的现有代码，但也可能包括文档、工单或Slack消息等内容。

这些都是可以帮助代理在其决策过程中建立基础的各种信息来源。另一半是解析。能够获取数据固然很好，但如何真正对这些数据进行推理？是否存在某种结构或隐含信息，能够实际提高从大量信息库中检索相关信息的能力？

我们将在接下来的几张幻灯片中更详细地讨论这一点。最后是访问权限问题。这与提高质量关系不大，但需要特别关注上下文感知的访问权限，尤其是考虑到大型组织可能对某些知识有访问控制。人工智能不应提供用户原本无法获取的知识访问权限。

智能代理系统的另一个组成部分并未完全体现在代理循环中，那就是对人类行为的理解。开发者采取的行动能让代理系统隐式地理解需要完成的任务。这真正实现了协作式代理体验或类似心流般的体验。如果情境感知能调动所有相关的显性知识。

跟踪开发者是否打开文件、执行某些导航或IDE操作，或在文件中进行编辑的想法，所有这些我们从这些操作中获得的隐含信息，也可以作为输入提供给LLM，以推理接下来需要调用什么工具，或者我们是否已经完成了工具的调用。通过观察不仅仅是代理系统的其他AI模式，我们实际上可以看到上下文感知和人类行为跟踪相结合的价值。

例如，我们可以看看经典的自动补全功能，AI会在光标位置建议几行代码来完成输入，这样作为开发者的你就不必从头开始输入。这在编写模板代码时帮助很大。这张图展示了一系列不同的实验，我们改变了模型可以访问的上下文级别、感知能力和人类行为追踪。

因此，在所有这些例子中，模型是完全相同的。我们唯一改变的是提供给它的输入类型。我们将以仅使用当前打开的文件作为LLM上下文来生成自动补全建议作为基准性能。如果你开始结合意图（这可能包括IDE中打开的其他文件和标签），自动补全结果的质量实际上提高了11%。

这清楚地表明，通过整合意图和人类行为追踪，即使使用相同的模型，也能创造出更好的结果。如果我们尝试另一个实验，例如，对整个代码库进行嵌入处理，简单地分块，然后在整个仓库中进行基于嵌入的简单检索，其结果也比基线要好。但有一点需要注意，可能有点反直觉的是，它的表现实际上比仅使用打开文件和标签的意图要差。

这实际上非常关键地指出了上下文感知系统的解析能力。因为如果你将解析方式从简单的分块和仅基于嵌入的检索，转变为使用抽象语法树解析（AST解析）、自定义代码解析器、更智能的分块以及更高级的检索方法——不再仅依赖基于嵌入的检索，而是考虑启发式方法和代码库的其他结构（如导入或邻近文件）——这样的改变能显著提升系统性能，远超基准水平。

因此，这实际上是为了强调，对于代理系统来说，不仅仅是工具和模型非常重要，上下文意识和对开发者意图的理解同样至关重要。总的来说，上下文意识工具和人类行为成为不同基因体验相互区别的轴心。

尤其是因为不同工具之间的推理模型相对相同。当然，稍微讨论一下这些工具也很重要，因为工具是智能体在推理步骤之间可以采取的行动。如果你有更高质量的工具，你就能采取更高质量的行动，从而更快地获得更好的结果。

这将分为三大类工具，这也是剖析智能代理系统构成的另一种思维模型。第一类是搜索与发现工具。其核心理念在于：你首先需要获取所有相关信息，才能着手进行任何改动。

第二类是能让我们改变世界状态的工具。最后是验证环节，确保对状态的任何改变确实能提升整体系统，并让我们更接近手头的任务。仔细想想，人类的工作方式其实也差不多，对吧？当我们开始一项任务时，会寻找所有相关信息，无论是在代码库、网上还是其他地方。

我们将进行一些修改，然后编译代码，运行代码并查看结果，看看是否符合预期。如果不符合，我们就重新开始整个过程。同样地，我们可以将这些工具归类为代理工具。因此，将这些工具重新组合在一起，正是这种组合使得这种协作式代理体验感觉非常自然。

比如说你想做个修改，你对一个类做了修改，现在你想在同一个目录下的其他类里也做同样的修改。这算是个相对模板化的任务，但如果我身边有个同行程序员或人类同行程序员一直在观察我的工作，我就可以直接告诉他们：嘿，在这个目录里类似的地方也这么改。这时候再把这个协作智能体想象成同行程序员，你就能看出这些不同组件是如何协同实现这个功能的。

首先，代理能够观察到开发者最近的操作，使其能够理解“同样的事情”指的是我刚才所做的最近编辑。通过工具访问，代理可以在这个目录中找到相关文件，随后利用类似编辑的功能进行修改。此外，上下文感知能力使代理能够判断其他类是否确实处于与我刚修改的类“相似的位置”的实体。

因此，需要这三个要素共同作用，才能创造出与代理合作的体验，类似于与人类同行程序员合作的感觉。最后总结几点：情境感知带来了明确的知识。人类的行为会带来隐含的意图。工具随后会采取行动。结合这种显性知识与隐含意图。这一过程将贯穿搜索与发现、状态变更与验证。最终，大语言模型被用于整合所有这些信息，以选择在适当的时间进行哪些正确的工具调用。但这是当今不同代理系统的共同特点。

-----------

本节课我们将深入探讨AI智能体的搜索与发现问题——这实际上需要全新的范式、对基准测试的重新审视以及创新方法，以最终最大化智能体的影响力。现在让我们开始吧。我们将花些时间讨论搜索与发现机制，因为在这个智能体主导的世界里，这确实是个非常有趣的问题，也是我们自己已实现若干创新的领域。

首先，让我们开始，为什么搜索和发现对代码来说很有趣？嗯，代码是非常分散的，因为框架、库、抽象的存在，并非所有你需要的信息都恰好存在于你正在编辑的文件中。它通常是不完整的。编写新代码需要的不仅仅是现有的代码。你需要文档，你需要工单。你需要搜索网络。而且它并不精确。我们都知道，做同一件事有很多种方法。因此，如果你想在你的特定代码库中找到构建某物的正确方法，那么一个千篇一律的方法和回应可能并不适用。

搜索和发现作为一个问题的真正关键在于，如果我们只检索到不正确或无价值的信息，那么从我们的AI代理系统中也只能得到不正确和无价值的结果。目前最先进的技术是检索增强生成（RAG）。在最基本的层面上，我们从用户的问题或提示开始。有一个检索器会检索出必要的相关上下文，与问题和任务一起输入到大语言模型中，以获得响应。

这就是我们在非常抽象的层面上通常对检索的理解，但仔细想想，这一切都是为了类似副驾驶或助手的系统，我们只能对大型语言模型进行一次调用。而代理方法实际上从根本上改变了这一点。因为与其在检索上反复迭代以使其更复杂、更准确（因为你只有一次机会调用大型语言模型，因此也只有一次检索机会），代理方法提供了不同的解决方案。

多步骤代理检索方法意味着我们实际上可以进行多次检索尝试。这与人类的行为非常相似。当我们外出寻找可能相关的信息，却发现并不真正相关时，我们会继续搜索，不断尝试，直到收集到所有相关信息后再采取行动。这正是我们应该思考的方式，即以代理方法来解决搜索和发现问题。

也许检索器不必完美无缺，但我们需要能够迭代人工智能。同样重要的是，并非所有检索器都必须相同。在整个搜索和发现问题中，我们可以针对可能需要完成的不同任务类型配备不同的检索器。那么我们需要具备哪些类型的任务和工具呢？有些任务中，我们明确知道自己想从语料库信息中获取什么。

有一条关于检索的规则，类似于grep。在某些搜索和发现任务中，我们大概知道自己需要什么，但就是不知道如何获取。比如，我可能需要做一次网页搜索，因为我知道网上有个联系表单对象的示例，我只是需要想办法找到它。第三类通常比较模糊，我只是想要完成任务所需的所有相关信息。

不太确切知道它到底是什么，但我知道我的总体目标是在这种情况下构建一个新的联系表单对象。现在，我或许可以详细讨论每一个部分，但为了本课程的目的，我将重点介绍我们在第三类别中创新并添加的一些新工具的方法。那么，让我们先更深入地了解一下第三类别。

如果我只是想构建一个新对象，并且只想获取所有相关信息，那么在实际操作中我需要些什么呢？例如，尝试为我的特定代码库构建一个联系表单时，我可能需要从内部实用工具库中提取信息。我可能需要查阅外部包和文档。还有代码库内外的其他示例。风格指南。显然，我需要综合大量不同的信息片段，才能为我的特定代码库获得真正高质量的响应。

因此，考虑到这一点，让我们来谈谈当今的最新技术。针对这类问题，目前最先进的技术是嵌入搜索。简单来说，它的工作原理是：你有一个嵌入模型，能够将对象（比如一段代码）转换为一串数字，即嵌入向量。你可以对代码库中已有的所有代码片段进行这种转换。
 
然后，在检索时，你只需将当前的工作、正在处理的代码上下文，使用相同的嵌入模型将其转换为自己的嵌入向量。接着，你将这个生成的嵌入向量与所有现有的嵌入向量进行比较，看看在这个高维嵌入空间中哪些其他向量是相近的。嵌入模型的基本思想是将看起来相似的文本片段转换为在向量空间中也彼此接近的嵌入向量序列。因此，如果操作得当，当你进行检索时，你会获取到一堆至少与你当前工作相似、理想情况下相关的代码片段。这就是基本的方法。

当然，这种基于嵌入的方法并不完美。因为我们操作的是嵌入向量而非原始文本，所以会丢失大量原始文本片段的细微差别。尽管我们尝试了越来越大的嵌入模型和各种方法，但嵌入在检索效果上似乎遇到了某种瓶颈。而这正是Windsurf和Cascade中针对该问题的一些创新和独特方法发挥作用的地方。

因为我们首先问自己的问题是，这里的基准测试结果真的有用吗？因为现实情况是，很多检索基准并不完全适用于代码问题。这些基准测试的工作原理是：有一个信息库，它就像大海捞针的问题。我是否从整个现有的信息库中检索到了非常特定的信息片段。

实际上，正如我们讨论过构建联系表单对象所需考虑的所有不同部分一样，我们确实需要大量信息片段来综合它们以得到正确的响应。因此，"大海捞针"的方法实际上并不是我们特别感兴趣的基准。我们更感兴趣的是这样的概念：如果我检索50个对象，其中有多少真实且相关的对象会实际出现在这50个中。

如果我能在其中保持较高的召回率，至少意味着我掌握了在一个庞大得多的信息库中所需的所有相关信息，这些信息对于完成我的特定任务尤为关键。当然，像这样的基准我们需要构建。那么，我们该如何构建呢？我们查看了GitHub仓库的公共语料库，意识到每一次提交信息都对应着跨多个文件的一系列深度修改。

这实际上是一种匹配关系，介于可能从提交信息中衍生出的查询内容，与我们需要检索的所有相关代码片段（即该提交中包含的所有差异变更）之间。通过从代码库中提取并整理这些信息，我们可以针对提交信息进行查询，并验证我们的检索方法是否找到了所有修改过的文件——这些文件正是所有相关的信息片段。

如果我想提前真正执行那个提交或查询。再看看基于嵌入的方法在这类基准测试中的结果，我们会发现实际上没有任何方法能超过50%，这意味着基于嵌入的方法误报率很高，尤其是在越来越大的代码库上。而且一开始就只检索到做出变更所需全部信息的一半。因此，显然我们希望构建一个比这更好的工具。

我们的方法是什么？我们使用了更多的计算资源。针对这个问题，我们的方法被称为“Riptide”。其核心理念是摆脱嵌入向量，因为一旦进入嵌入空间，我们就失去了那些细微差别。相反，我们会获取查询请求，提取代码库中的每一段代码片段，然后运用大型语言模型来提问：这段代码与我当前的具体查询有多相关？

我们并行运行所有这些类型的查询，然后根据每个查询的相关性响应，对代码库中的所有代码片段进行重新排序。正如你所意识到的，这种级别的重新排序从未在嵌入空间中被应用过。这一切都是通过基于LLM的语义搜索检索器完成的。毫不意外，在我们的recall 50基准测试中，这种方法的表现优于基于嵌入的方法。再次强调，这只是多步骤检索过程中的一个工具。

虽然还不够完美，但通过打造一款检索质量显著提升的工具，并将其与多步骤检索范式相结合，我们现在拥有了一种搜索与发现的方法，能让我们的智能代理系统在大规模代码库中运行。最后总结几点关键启示：既然讨论的是智能代理，请务必考虑采用多步骤检索而非单步骤检索。

思考所有不同的潜在优化工具，因为你无需局限于单一的检索方法。可以采用多种检索途径。质疑现有基准和约束条件，从而开发新的检索方式来提升你的智能代理系统性能。既然我们已经初步探讨了搜索与发现的运作机制（尤其是针对级联系统和代码代理的场景），现在让我们将其应用到一个大型代码库中，执行几项不同的任务。我们现场见。

-------------

在本课程中，您将运用并见证搜索与探索方法的力量，当使用AI代理在大型生产代码库上完成几项不同任务时。让我们开始吧。很好。在这个演示中，我们将前往另一个代码库获取另一种语言，仅为了展示Cascade所能做到的广度，特别是在大型代码库上，使用我们刚刚讨论过的一些搜索和探索工具。这是一个包含大量Java代码的代码库，大约有10万行代码，包含许多文件，它使用了多种不同的框架和包，比如Groovy和Spring Boot。

假设我是一个刚接触这个代码库的新开发者。通常这需要花些时间，但借助我们之前讨论的那些强大的搜索和发现工具，Cascade 能帮助我们加快速度。现在我要打开 Cascade，向它提问：请解释我的标签代码和整个源目录（大约 300 多个文件）的作用。让我们看看 Cascade 会怎么做。Cascade 将运用我们讨论过的多种搜索和发现工具。

首先，它会使用LS工具来实际确定所有顶级目录是什么。在这里，它进行了一次riptide调用。如你所见，搜索并评估了近3000个不同的代码片段，以了解它们与理解这个代码库问题的相关性。然后，它特别进入了测试目录，在那里又进行了一次riptide调用，以进一步理解。再次强调，这里你看到了多步检索的过程。在我们尝试给出答案之前，已经进行了多次检索调用。

看，这里解释得很简洁明了。它涵盖了我需要了解的这个代码库的所有重要部分，甚至还提供了一些不错的代码链接供我查阅。当然，我随时可以继续提问。比如我正在浏览这里，觉得书籍管理功能很有意思，想了解更多细节——请详细解释书籍管理的运作机制。这个对话会结合之前的交流轨迹和上下文，但正如你所见，我们会针对这个新问题再做一次深入分析，重点考察它与当前查询的相关性。

所以你可以看到，我们正在多次使用这些搜索和发现工具。在得出问题的答案之前。再次强调，全面的解释附带大量引用。我可以点击并阅读代码库中的内容，自己理解。这就是这些强大的搜索和发现工具在实际生产代码库中的威力。不过，我要展示的是搜索和发现在另一种工具中的应用。假设我真的想对此进行迁移。

让我们在Spring Boot上实现这个功能。我会询问Cascade当前使用的Spring Boot版本是什么，以及最新版本是多少？这个问题的前半部分需要理解代码库，后半部分可能需要在网上查找。所以系统会进行分析。首先确定当前使用的Spring Boot版本。现在你可以看到正在进行网络搜索。我们正在通过Cascade访问网络。它会搜索最新的Spring Boot版本。然后获取搜索结果，生成一个概要，这样我既可以查看页面内容。

或者我可以在编辑器中准确地执行文本分析，并获取预训练模型中不存在的信息。它甚至会提示我：“嘿，这里存在一个需要多次版本变更的重大版本差距”，并开始引导我进行迁移流程。虽然我不会完成整个迁移过程，但这个演示突出展示了多种不同的搜索和发现工具。

我们看到每次都在多步骤方法中使用LS工具、Riptide工具和网络搜索。正是这种方式让智能代理系统能够处理这些更复杂、长期运行的任务。希望了解这些强大的搜索和发现工具如何在生产代码库中使用是令人兴奋的。不过让我们转换话题，开始构建一个新应用，以探索Cascade提供的更多功能。

------------

在本课程中，你将使用协作代理开始构建一个更大的维基百科分析应用的数据分析部分。在此过程中，你将学习如何指导和观察AI代理工作的基础知识。让我们开始吧。在深入构建应用之前，我们将介绍最后一个概念。在之前的幻灯片中，我们讨论了很多关于构成协作AI代理（如Cascade）内在强大功能的不同部分和组件，比如上下文感知、工具和人类操作。

但同样重要的是，要考虑那些独立于底层功能、能提升用户体验的维度。即便上述因素保持不变，工具本身仍有许多方面能让用户体验变得极其出色。需要特别指出的是，在我们开始构建维基百科分析应用时，我会频繁提及这些要点。首先是功能维度——虽然我们会大量使用Cascade，但像Windsurf这类AI原生IDE提供的智能代理功能并非其全部能力。

那么，何时适合使用这些其他功能呢？有很多指导性的地方。虽然Cascade有能力独立进行、深入、退出并检索正确的信息，但事实证明，你越擅长给AI提供一定程度的指导，就越能将注意力集中在正确的地方。就像同行程序员一样，他们会表现得更好。那么，如何有效地在Cascade中做到这一点呢？可观察性。当然，现在的AI能力远超仅仅提供几行自动补全建议或回复一条聊天消息。
 
因此，如果你擅长随时观察AI的操作，你就能跟上AI的节奏，保持状态，确保自己能高效完成越来越大的任务。最后，还有很多细节优化。有很多实用的插件和快捷方式，确保你能以最佳方式与AI进行原生交互。所以我们决定开发IDE界面并非随意之举。我们认为其中有许多可以出彩的细节优化和用户体验设计。

那么，我们就直接开始吧。好的。我们将从一个完全空白的项目开始，叫做“维基百科分析”。我们将从零开始构建这个项目。在这个视频中，我们将进行一些数据分析。这个应用的灵感来源于我和办公室的一群朋友玩的一个叫做“钓鱼”的游戏。这个游戏很简单。你会得到一个维基百科页面的所有类别，然后你需要猜出这些类别对应的维基百科页面。

这让我开始思考。如果我要尝试学习一种新的信息类别，我需要做些什么？在真正理解这个主题之前，我首先需要学习哪些基本术语？因此，我们要做的是尝试利用维基百科的信息，选取一个类别，并了解属于该类别的所有页面中出现的单词频率。

这将为我提供一个良好的起点来开始我的工作。所以我打算从这样一个提示开始。它就在这里复制。我会大声读出来。编写一个脚本，该脚本将维基百科类别作为命令行参数输入，并输出该类别中所有页面上非常见单词的累积频率，然后在一个特定类别上运行它。我最喜欢的一个类别是大语言模型。我还指定了使用Mediawiki API。在运行之前，这里有几件事需要指出。请注意，我对输入和输出的具体形式描述得相当详细。我指定了想要使用的某些API。

对AI保持清晰的表达非常重要，尤其是在没有任何背景信息的情况下。如果你一开始就给出非常模糊的陈述，AI可能会做出虽然正确但出乎意料的事情。再次强调，这就像与程序员同事合作一样。如果你想让程序员同事帮你完成某些工作，你需要明确告诉它该做什么？在开始工作之前，关于Cascade面板还有几点需要说明，正如你所见，我目前正在使用Claude 3.5 Sonnet作为我的推理模型。

实际上你可以选择多种不同的可用模型，但让我们先开始吧。我点击回车键让Cascade开始运行。很好。在我点击requirements.txt的安装命令之前，先指出Windsurf中的几个功能来帮助你观察AI在做什么。第一个是我们之前已经用过的功能，就是"打开差异对比"按钮，因为编辑可能涉及多个文件。能够点击"打开差异对比"并查看Cascade所做的更改是非常有用的。

仍需仔细检查AI所做的修改。你还会注意到这里有一个我们称之为“级联栏”的功能。这是一种直接在编辑器中构建审阅流程的方式。因此，如果文件中有多处修改，你可以使用上下箭头来浏览。而左右箭头则可以在有编辑的文件之间切换。我可以在单个区块级别、文件级别或所有文件中接受这些更改。

但现在我们已经做了一些可观测性工作，让我们继续在这里工作。所以我首先需要安装requirements.txt。所以我点击接受。请注意，我的环境实际上使用的是Python 3而不是Python，pip3而不是pip。因此，Cascade能够识别这一点，并相应地纠正。但在它继续的同时，也许这是我想要Cascade在我工作时一直知道的事情，我不希望它总是发现我用的是pip3而不是pip。

所以我在这里展示第一种指导方法：如果你点击右下角的“Windsurf设置”选项卡，会看到许多不同的设置选项。我要进入这个名为“设置全局AI规则”的选项。这些是Cascade在你工作时会遵循的规则，无论你在哪里工作。当然，仓库规则是针对特定仓库的，但我可以在这里直接指定使用Python3而不是Python，用pip3替代pip之类的规则。现在每次我让Cascade执行操作时，它都应该能遵循这些指令了。

我们稍后会验证这一点。但现在让我们继续看看Cascade在做什么。它正在提示：好的，现在让我们在大型语言模型类别上运行它。开始执行。如你所见，它已经开始运行了。它正在开始处理维基百科上所有属于大型语言模型类别的不同页面。你可能注意到，由于空间限制我们截断了输出内容，但实际上Cascade使用的是集成开发环境自带的终端。

所以，点击这个名为“转到终端”的小按钮，你就能在编辑器的终端面板中调出完整的执行记录。这样我就能查看，虽然只能看到最后几行，但不出所料，一些最常见的词汇包括“model”（模型）、“models”（模型）、“language”（语言）、“OpenAI”、“ChatGPT”。所以，我们已经可以大致了解到，是的，如果我要学习大语言模型，这些可能是我需要熟悉的一些术语。

这太棒了。它成功构建了一个完全符合我需求的数据分析功能。我将继续操作并接受所有更改。重要的是要随时接受更改，而不是反复累积修改。所以我现在就接受所有更改。关闭终端。让我们继续检查我之前设置的全局规则。为此，我将使用另一个不同的功能。

我本可以在当前的级联对话中要求检查这一点，但重要的是要在不同的对话之间保持一定程度的分离。因此，我将开始一个新的级联对话。在这里，我可以像之前需要的那样发出相同的命令，安装requirements.txt。如果它没有学到任何东西，它就会使用pip而不是pip3，但现在它使用了我们指定的全局规则，直接使用pip3。

所以这就是一个例子，说明只要设置一些规则，Cascade就能始终遵循关于环境或工作方式的一般最佳实践。当然，在这个例子中，一切都已经设置好了。回到我最初的对话，我过去常常回到过去的对话中，然后返回，现在我又回到了最初的对话。

在本课程中，您使用Cascade构建了应用程序的整个分析部分，通过维基百科API获取所有相关数据，进行相应解析，最终得出频率结果。在此过程中，您运用了Windsurf ID中的多项功能来引导Cascade并观察其运行情况。下一节中，我们将为系统添加缓存功能，因为我不希望在每次迭代应用程序时都重新运行此分析。那我们下节课见。

-----------

在本课程中，您将为维基百科分析应用添加缓存功能，但更重要的是，您将学习如何让AI代理重回正轨，即使它做出了一些稍显意外或非预期的行为。让我们开始吧。这就是我们上节课结束时的进度。我们已经完成了所有的频率分析。但在继续构建应用程序的其他部分之前，我想先添加一些缓存功能，因为实际浏览与大型语言模型相关的数百个页面确实花费了不少时间。

在与AI代理愉快合作的过程中，偶尔会遇到它做出一些稍显意外或非预期行为的情况。让我们深入探讨。首先回顾上节课的收尾部分——我们已经完成了所有频率分析工作。但在继续构建应用程序其他模块前，我想先添加缓存功能，因为处理与大语言模型相关的数百页内容确实耗费了大量时间。

所以我只希望结果能被缓存。如果我使用相同的输入，那么当我修改应用程序的其他部分时就不会有问题。为此，我将使用这个提示：创建一个本地缓存，这样我们就不必重新运行检索逻辑。同样的类别，正是我们刚才提到的。让我们看看Cascade做了什么。很好。看起来Cascade进行了多次编辑。我可以使用Cascade栏来逐个查看差异，并阅读和理解代码变更。让我们运行这个。当然，第一次运行时实际上需要再次处理每个文件。

让我们来看看缓存。好吧，看来它并没有完全按照我的意愿执行。它所做的只是找出每个不同页面的页面ID。这实际上并不是我想要的。我原本想缓存我脚本的所有结果。如果我回到我的查询部分，你会发现我在这方面有点含糊，对吧？从技术上讲，它确实缓存了关于该类别页面结果的某些内容，但并不是我真正想要的。我想要的是缓存这个脚本整体处理的结果。

所以从这里开始，我有几个选择。即使我已经接受了文件中的所有更改，我们仍然可以采取两种行动。我们可以进入Cascade，要求它撤销所有更改，然后尝试进行一些新的编辑来修复问题，但这可能会让我们陷入一点小麻烦。或者，我们可以使用一个便捷的功能——当你将鼠标悬停在消息上时，可以选择回退到某个步骤。这个回退操作会还原整个对话历史，并撤销过程中对文件所做的所有更改。所以如果我点击回退...

维基分类分析文件中的所有缓存逻辑都被移除了。这基本上可以帮助保持对话历史的简洁，避免大量来回尝试让代理恢复正常，否则这些内容可能会在后续对话中作为上下文让它更加困惑。因此，回退功能确实能有效帮你摆脱困境。现在让我删除这个缓存，我们不再需要它了。再稍微修改一下这个提示语，让它更符合我想要实现的具体目标。

所以我现在很清楚，我想要脚本结果的本地缓存。我只是不想再重新运行它。即使是针对同一类别的处理逻辑。好的。首先，它要求我创建缓存目录。这没问题。好吧。那么做了一些更改。让我们再试一次。它还是得先遍历所有不同的文件进行第一轮处理。我们来看看现在的缓存是什么样子。很好。

缓存里确实有我想要的东西，也就是脚本的运行结果。所以如果我重新编写，结果会明显更快地出现。因此，我们得以构建一些缓存逻辑。此外，我还可以使用级联栏来仔细检查所有更改，确保一切都符合我的预期，并且我理解代码并在每个单独的代码块层面接受它。

我可以从文件层面接受这一点。但更重要的是，在这节课中，你学会了当AI做出一些你不太预期的行为时，如何让自己摆脱困境。所以，好好利用这一点，保持你与AI结对编程伙伴之间健康的对话历史。在下一节课中，我们将把已完成的所有工作可视化，通过添加前端来以美观的方式展示所有这些结果和频率。到时候见。

--------------

在本课程中，你将构建一个全栈应用程序，用于展示你在维基百科分析应用中所完成的数据分析结果。在此过程中，你将学习到许多技巧、窍门和最佳实践，以最大限度地提升与AI助手的互动效果。让我们开始吧。好了，现在到了最有趣的部分。我们将把之前为构建频率分析和缓存所做的所有工作，通过网页实现可视化展示。

那就切换到Windsurf吧。这是我们上次停下的地方。我要做的是为这个任务开启一个新的对话，因为我们在构建前端部分。我打算使用这个提示：创建一个网页，如果结果缓存可用就从那里加载数据（当然），否则从头计算词频并显示一个词云图，其中单词的大小与其出现频率成正比。再次说明，我指定要使用HTML、JavaScript和Flask来完成这个任务。让我们看看Cascade能做什么。好的，看起来Cascade已经添加并编辑了一堆文件。

再次，我可以使用开放差异工具Cascade栏，来观察和调查所有已做的更改。让我们去运行应用程序。好的。看起来应用程序已经上线了。让我们去看看。现在我们来到了localhost 5000，也就是它被部署的地方。让我输入我的大语言模型类别。生成词云，砰。看起来速度够快。似乎是在缓存中。对我来说看起来很棒。让我们回到Windsurf，我再展示一些东西。

首先我想展示的是，正如你所见，目前Cascade正在运行一个终端进程。你会注意到输入提示区域上方出现的这个状态栏。这里会显示所有你需要审阅的变更，以及所有需要检查的文件。同时你还能看到当前正在运行的所有终端进程。

你也可以从这里或直接从命令实际运行的地方结束和取消任何进程。那么让我们开始吧。在进行审查时，我可以逐个文件查看。我可以在块级别接受更改，也可以在文件级别接受更改。我还可以直接从这条工具栏接受所有更改。因此，有多种不同的方式来接受或拒绝级联的更改。到目前为止一切顺利。让我们再四处看看。我这里有一个HTML文件，看起来很不错。

显然，这里有很多JavaScript代码，正如所要求的。也许我对JavaScript还不够专业。我有点想了解一下发生了什么。Windsurf编辑器中还有一些其他方便的功能，隐藏在各处，使这类操作变得非常容易。例如，如果我查看这个函数，这是一个用JavaScript生成词云的函数，我对此并不专业。

你会注意到文本编辑器顶部出现了许多有用的代码透镜。所以我只需点击“解释”即可。这样做的结果是，Cascade会拉取所有相关信息到Cascade中进行分析，并向我解释发生了什么。这非常有用，尤其是当我使用Cascade和这些代理处理不熟悉的语言或框架时。我还可以用它来准确解释发生了什么，而我们能够检测到这一点并仅拉取相关信息，这要归功于我们在前面课程中提到的基于代码库构建的复杂解析逻辑。

因此，这始终是理解现有代码的一个有用工具。接下来，我想在我的应用程序中添加不同的颜色。我想让它看起来非常漂亮。为了实现这一点，我将实际展示一些与Cascade或代理无关的功能。再次强调，这是一个AI原生的IDE，因此它不仅仅只有代理体验。为此，我将转到我的文件资源管理器。

我将创建一个名为Colorpalette.py的新文件，专门用来定义调色板。现在我要暂时关闭Cascade。我要展示的第一个功能叫做“命令”。这是一种直接用自然语言向AI发出指令的能力。在文本编辑器中，其快捷键是command I。如你所见，命令输入框出现了。这次，我只需给命令一个指令：创建一个包含六个十六进制颜色的调色板类。常见的调色板等等。

 让我们看看它是如何运作的。如你所见，它正在快速为我生成大量代码，这很棒，甚至可能比我手动提示Cascade时还要快，这就是为什么你应该始终使用代理体验来处理一切。不过它很棒的地方在于，你还可以使用更多功能。例如，你已经可以看到自动补全正在尝试建议额外的样板代码。所以看看这个，也许我想要你的RGB调色板类。

我只需自动补全，就能快速生成多行代码。我可以在这里继续我的工作。我可以说"定义函数来获取所有调色板"，它就会找到方法直接返回所有调色板。这很棒，但你会发现——让我调整下换行以便显示清楚——这不仅仅是命令或自动补全的功能。在文本编辑区域里，还存在着许多被动的编辑体验。

例如，我可能会直接到这里，把这个默认调色板重命名。你会发现我们的AI已经开始建议远离我光标位置的后续编辑内容。这就是在文本编辑器内部拥有强大被动体验的优势——无需依赖智能体作为每次修改的辅助工具，就能快速生成大量模板代码。虽然这很有帮助，但现在我需要将创建的调色板类真正整合到应用程序中。

因此，针对这一点，我将切换回Cascade。我要让Cascade做的是将这个“获取所有调色板”方法集成到我现有的应用中。为此，我会使用一个名为“应用提及”的功能——这是我们之前没提过的功能——但为了展示它的实际作用，这个功能能让你作为开发者，在已经明确AI需要关注的地方，以非常轻量的方式引导AI。

 再次强调，由于Cascade具备自主性，它确实有能力自行解决这个问题。但就像与同事程序员协作时一样，如果你能温和地引导他们走向正确的方向，就能帮他们节省时间。因此，请以类似的方式思考这个@提及功能。你可以@提及文件、目录，也可以@提及我们稍后将用到的独立方法和函数。你甚至能@提及网络上一系列常见的第三方API公共文档。

那么，让我们将这个调色板集成到我们的词云应用中。为此，我将使用刚才提到的那个函数，即获取所有调色板的功能，将其应用到词云应用中。再次强调最佳实践。让我们稍微描述一下。我们希望调色板可以在下拉菜单中选择，并显示正在使用的颜色。让我们看看会发生什么。

如你所见，通过使用提到的应用程序，代理能够直接前往需要查看的地方，然后它将利用这种代理搜索和发现能力来查看其他相关内容。开始进行一些修改。你可以在我操作时使用开放差异。看到一些正在进行的更改。让我们看看效果如何。看起来相当不错。那么让我们尝试生成类别名称。当然这是我的责任。哦，我不小心拼错了。没关系。很好。看起来我们得到了使用这种材质调色板的词云。让我们开始更改调色板。更改深色调色板。

似乎生成的词云有所不同。太棒了。看起来运行正常。基本符合我的预期。我注意到一点，当选择不同的调色板时，它会自动更改词云，而无需我点击“生成词云”按钮。也许我喜欢这样。也许我实际上想稍微调整一下。再次强调，这是与AI代理或任何AI工具合作的迭代过程。我必须回到Windsurf并明确表示，确保我需要点击生成按钮来创建新的词云。即使我更改了调色板的选择。
 
或者只是某种指示性的操作，我们来看看它会有什么效果。看起来这并不是一个很大的变化。这样就整理好了所有的变更。让我们回到应用程序。刷新一下。输入。类别。好的。现在如果我选择，你称之为词云变化。但词云并没有改变。所以让我们看看。给我加载屏幕。太棒了。再次强调，这些都是小细节，但这是使用Cascade迭代应用程序的方式。

在这节课中，我们做了很多工作。我们不仅使用Cascade构建了整个全栈应用，还利用命令和自动补全功能快速生成了一堆样板代码，并将其整合到我们的整体应用中。正如你所看到的，使用Windsurf不仅仅是通过代理进行迭代。它是一种与AI的完整互动体验，旨在最大限度地提高你开发所需功能的速度。这很有趣！我们将在接下来的一节课中结束这个部分。所以让我们继续吧。

------------

在本课程中，你将通过利用AI代理的多模态能力添加一些进一步的功能，你还将学习一些功能和最佳实践，以便继续构建和定制你的维基百科分析。让我们继续。你已经完成了最后一课。在这最后一课中，我们将使用一些多模态输入再添加一个功能。

 那么，我们切换到Windsurf。很好。这个应用看起来不错。你会注意到这里有一个功能，可以在Cascade中插入图片。我们将利用这个功能。接下来我要做的是回到我的应用程序。应用程序看起来不错。我现在在这里，我该做什么呢？让我缩小一点。我实际上要截个图。让我打开截图。很好。我实际上只是要给这个添加一些东西。

那么让我在左边这里添加一个矩形。一些文字。假设我还想让你知道视觉呈现很棒。但实际上我还想同时看到原始频率。所以我要放上原始频率。让我们再添加一个文本框。词一。词二。其实我就在这里放一些文字吧。这大概就是我做的，大约30秒的草图。

现在，我不需要再详细解释布局的所有细节，只需保存这张图片即可。让我们回到Cascade。现在我将使用这个图片输入。我已经将图片添加到Cascade中。接下来我只需要说：按照图片所示，将原始频率添加到应用中。就这么简单。Cascade可以利用这些推理模型的多模态特性。它已经注意到这些原始频率需要放在词云左侧的方框中。

我从没跟Cascade提过这事。好了，改动似乎已生效。我们返回页面刷新看看。大语言模型。把画面稍微拉近些。左侧方框里显示的是原始频率数据，就像我图中展示的那样，连红色轮廓上的红字标注都保留着。这就是通过多模态技术提升开发效率的简单应用案例。

所以这是我们这次演示中要添加的最后一个功能。但我很期待看到你们能在此基础上构建出什么。为了帮助你们更好地使用，我会指出Cascade和Windsurf中一些在这次演示中没能展示的其他功能。

首先，Windsurf设置面板上有许多选项，通过多个分界面来定制您的体验。我们之前没有展示的一个功能是“记忆”概念。记忆的理念类似于规则，它们是Cascade可以持续回溯和引用的信息片段。

因此，Cascade能够随着时间的推移，逐步构建关于你如何开展工作以及对你而言重要事项的状态记录。这些记忆既可以被明确提及（即那些规则），也可以自动生成。这就是Cascade生成的记忆功能所实现的——它会在你工作时自动创建这些记忆并发送通知，或者你可以直接告诉Cascade"记住这个"，它就会将这些记忆存入其记忆库中。

与规则类似，您可以随时返回并编辑这些内容。但这种方式能让Cascade自动学习，而无需您通过规则明确指定所有细节。这里还有许多控制您被动体验的元素，以及其他一些不错的用户体验小细节。设置面板中还有其他功能，例如如果我搜索"Windsurf"，您会看到许多其他可编辑的字段选项。

我最喜欢的功能之一其实是Cascade命令中的全部执行和拒绝列表。如你所见，Cascade建议的每条命令我都必须接受。但有些命令我很乐意自动执行，而有些命令我永远不想自动运行。所以在最新版本中，你实际上可以设置命令白名单和黑名单。甚至还有涡轮模式，可以自动为你运行所有命令，如果你想完全沉浸式地与你的代理一起编程的话。

本节课和整个课程就到这里。正如我们所看到的，Windsurf和Cascade拥有众多功能和特性，让开发工作充满乐趣。希望您已经掌握了如何通过这些AI协作助手优化开发工作流的一两个技巧。当然，这只是当前的一个阶段。Windsurf和Cascade只会不断进步，越来越好。

人工智能将变得更智能。其功能、完善度、引导性和可观测性都将持续扩展。请通过我们的文档或更新日志随时了解最新动态，掌握Windsurf和Cascade中的最新最优功能，从而最大化您作为开发者的人工智能体验。

----------

恭喜你完成了这门课程。在本课程中，你将学习人工智能代码辅助的历史，如何区分一些炒作与现实，如何看待这些人工智能代码助手，以及这些人工智能代理如何工作。我们深入探讨了搜索和发现这一具体而重要的范式，在此过程中，你使用windsurf和协作代理构建了许多不同的应用程序。我期待看到你用windsurf构建的作品。

--------------


## 💻 &nbsp; Prompts for lessons 7-10:

**📚 &nbsp; L7: Wikipedia Analysis App – Data Analysis**

Write a script that takes in a wikipedia category as a command line argument and outputs the cumulative frequency of non-common words across all of the pages in that category, and then run it on "Large_language_models." Use the MediaWiki API.

**📚 &nbsp; L8: Wikipedia Analysis App – Caching**

[First prompt] Create a local cache of page results so that we aren't rerunning the retrieval logic for the same category
[Second prompt] Create a local cache of the script's results so that we aren't rerunning the processing logic for the same category

**📚 &nbsp; L9: Wikipedia Analysis App – Fullstack App**

Create a webpage that loads from the results cache (if available, otherwise computes the frequencies from scratch) and displays a word cloud where the sizes of the words are proportional to frequency. Use HTML, javascript, and Flask.
[Command] create a ColorPalette class that would contain 6 hex colors, and a bunch of common color palettes that extend the class

Integrate @get_all_color_palettes into the word cloud app. Make the color palettes selectable in a drop down and display the colors being used.

**📚 &nbsp; L10: Wikipedia Analysis App – Polish**

Add raw frequencies to the app as shown in the image.


## 🧑‍💻 &nbsp; Repos used in the course

**📚 &nbsp; L3 Repo – Fixing Tests Automatically**

- https://github.com/Exafunction/windsurf-demo

**📚 &nbsp; L6 Repo – Understanding Large Codebases**

- https://github.com/ddd-by-examples/library

------------------------------------------------

欢迎来到与Arize AI合作打造的AI智能体评估课程。假设您正在开发一个AI编程助手，要生成优质代码可能需要执行规划、工具调用、反思等诸多步骤。而采用评估驱动开发流程，将大幅提升您的开发效率。在本课程中，您将学会如何为基于智能体的应用程序添加可观测性。这意味着您可以实时追踪每个步骤的执行情况，从而对各个组件进行精准评估，并高效推动组件级优化。

然后在整个系统层面上也是如此。如果你在问自己一些问题，比如是否应该在最后一步更新提示，是否应该更新工作流程的逻辑，或者是否应该更改你正在使用的大型语言模型？拥有一个纪律严明、以评估为导向的过程将极大地帮助你以系统化的方式做出这些决策，而不是随机尝试很多事情，看看哪种方法有效。如果你听说过错误分析的概念，这是机器学习中的一个关键概念，那么这教会你如何在代理工作流程开发过程中进行错误分析。

 如果你没听说过错误分析，那也没关系。但本课程将围绕这一重要概念体系展开，教你如何高效构建自主工作流程。本课程由开发者关系负责人约翰·吉尔胡利和Arize AI产品总监阿曼·汗共同授课。很高兴能与你们共同完成这门课程。谢谢。安德鲁。我们非常期待教授这门课程。谢谢。假设你正在构建一个研究智能体，它能进行网络搜索、识别信息来源、收集内容、总结研究发现，如果发现输出结果存在缺陷还会自动迭代优化。

在构建这个复杂系统时，你需要评估每个步骤输出的质量。例如，在文献筛选环节，你可以创建一个测试集，其中包含研究主题和对应的预期文献集合，然后统计智能体选择正确文献的百分比。对于摘要生成这类开放式任务，你可以调用另一个大型语言模型进行评判，或者采用我们称为"大语言模型评审"的方法，以此评估文本摘要这类开放式输出的质量。

除了测试和改进代理输出的质量外，您还可以评估代理所采取的路径，以确保它不会陷入循环或不必要地重复步骤。因此，在本课程中，您将学习如何构建评估体系，以迭代和改进代理的输出质量和所采取的路径。您将通过创建一个基于代码的代理来实现这一点，该代理作为数据分析器运行。

该代理将配备一套工具，使其能够连接数据库并进行分析。其中包括一个用于识别使用何种工具的路由器，以及一个用于记录聊天历史的内存。你将收集并评估代理处理查询时所采取的步骤痕迹，并对收集到的数据进行可视化。然后，你将学习如何使用不同类型的评估器来评估代理工作流程中的每个工具。

 你还需要评估路由器是否根据用户的查询选择了正确的工具，以及是否提取了正确的参数来执行工具，并评估代理所采取的轨迹。最后，你将所有的评估器整合到一个结构化的实验中，以便迭代和改进你的代理。虽然本课程重点在于开发过程中应用评估。

你还将学习如何在生产环境中监控你的智能体。本课程的诞生凝聚了许多人的心血。我要特别感谢来自Arize AI的Mikyo King、Xander Song和Aparna Dhinakaran。同时来自DeepLearning.AI的Hawraa Salami也为本课程做出了贡献。John和Aman都是评估AI智能体工作流这一重要课题的专家。现在让我们进入下一个视频，希望你能享受这门课程，并从John和Aman身上学到很多。


------------------

AI代理是由大语言模型（LLMs）驱动的软件应用。在第一课中，你将学习如何评估基于LLM的系统与传统软件测试的不同之处。接着，我们将探讨代理的含义，并讨论在评估代理时需要考虑的因素。让我们开始吧。一般来说，评估通常分为两个层面。在左侧，是大语言模型评估。这主要关注大语言模型在执行特定任务时的表现。

你可能见过像MMLU这样的基准测试，涵盖数学、哲学、医学等多个领域的问题，或者用于代码生成任务的人类评估。供应商经常利用这些基准来展示他们的基础模型表现如何。在右侧，你会看到LLM系统或应用程序评估。这部分着重评估你的整体应用程序（LLM只是其中的一部分）的表现如何。这里用于评估的数据集是通过手动、自动或使用真实世界数据合成的方式创建的。

当你将大型语言模型（LLM）整合到一个更广泛的系统或产品中时，你会希望了解整个系统（包括提示词、工具、记忆和路由组件）是否达到了预期的效果。你可能对传统软件测试有所了解，这些系统在很大程度上是确定性的。你可以将其想象成一列在轨道上行驶的火车。通常有一个明确的起点和终点，检查每个部分（火车或轨道）是否正常运行通常很简单。

另一方面，基于大语言模型的系统更像是在繁忙的城市中驾驶汽车。环境多变，系统具有不确定性。在软件测试中，你依赖单元测试来检查系统的各个部分，并通过集成测试确保它们按预期协同工作。结果通常是相当确定的。与传统软件测试不同，当你多次向大语言模型输入相同的提示时，可能会看到略有不同的回应。

就像城市交通中司机的行为会有所不同一样。你经常需要处理更多定性或开放式的指标，比如输出的相关性或连贯性，这些可能并不完全适合严格的通过/失败测试模型。对于LLM系统，有几种常见的评估类型。首先是幻觉问题：LLM是在准确使用提供的上下文，还是在编造内容？其次是检索相关性：如果系统检索了文档或上下文，它们是否真的与查询相关？然后是问答准确性：回答是否符合用户需要的真实情况？最后是毒性问题。

大语言模型是否输出了有害或不良语言？以及整体性能如何。系统在实现其目标方面的表现如何？有许多开源工具和数据集可以帮助你衡量这些方面，同时也能帮助你开发自己的评估方法。我们将在课程后面介绍其中的一些内容。一旦你从基于大语言模型的应用转向智能体，你就增加了一层额外的复杂性。
 
智能体利用大语言模型进行推理，但它们还会通过选择工具、API或其他功能来代表你采取行动。智能体是一种基于软件的系统，能够运用推理能力代表用户执行操作。一个智能体通常包含三大核心组件：首先是推理能力，由大语言模型驱动；其次是路由决策，负责选择要使用的工具或技能；最后是执行环节，完成工具调用、API调用或代码运行。你可能已经接触过智能体的应用实例，比如帮你做笔记或转录信息的个人助理型智能体。

基于桌面或浏览器的代理程序，可帮助自动化重复性任务。用于数据抓取和摘要的代理，以及能够进行搜索和整理研究的代理。让我们用一个示例用例来说明代理的实际工作原理。假设您希望代理预订一次前往旧金山的旅行。幕后有很多工作在进行。首先，代理必须根据您的请求确定应该调用哪个工具或API。它需要理解您真正需要什么，以及哪些资源会有所帮助。

接下来，它可能会调用搜索API来查询可用航班或酒店，并决定向你提出后续问题或优化其构建该工具请求的方式。最终，你希望返回一个友好且准确的响应，最好包含正确的行程细节。现在，让我们思考如何逐步评估：代理是否一开始就选择了正确的工具？当它形成搜索或预订请求时，是否以正确的参数调用了正确的函数？它是否准确使用了你的联系方式，例如日期、偏好和地点？

最终回复看起来如何？语气是否恰当，内容是否准确？在系统中，有很多地方可能会出错。也许代理返回的是飞往圣地亚哥的航班，而不是旧金山。对某些人来说可能没问题。但如果有人想去旧金山却最终得到圣地亚哥的航班，他们就会不高兴。这凸显了为什么你不仅需要评估大语言模型的原始输出，还要评估代理在每一步如何决定每个动作。你可能会遇到诸如代理调用错误的工具、误用上下文，甚至采用尖酸或不恰当语气等问题。

有时用户还会尝试破解系统，这可能会产生更多意想不到的结果。为了评估这些因素中的每一个，你可以使用人类反馈或人类参与循环，或者在较小程度上使用LLM本身作为评判者，来评估代理的最终响应是否真正满足你的要求。我们将在本课程后面更深入地探讨LLM作为评判者在代理评估中的作用。最后，请记住，即使是对提示或代码的小改动也可能产生意想不到的连锁反应。
 
例如，添加一句简单的提示语，如“记得礼貌回应”，可能有助于改进多个使用场景，但也可能导致你意想不到的测试案例出现倒退。这就是为什么你需要维护一组具有代表性的测试案例或数据集，以反映你的关键使用场景。每次调整系统时，你都可以在这些数据集上重新运行评估，以发现倒退问题，并随着时间的推移不断构建新的代理能力。这种方法对于开发稳健的代理评估至关重要。

就像使用传统软件一样，您需要对智能体的性能进行迭代优化。然而，您不能仅仅依赖确定性检测。智能体本身具有不确定性，会采取多种路径，可能在一个场景中表现退步，同时在另一个场景中有所改进。为了应对这种情况，您需要一套覆盖不同用户场景的稳定测试方案，并在每次修改时运行这些测试。

这些测试通常从生产数据中循环反馈，比如真实世界的查询和真实用户的互动。然后回到开发阶段，你可以在提示、工具或方法上进行优化。这可以帮助你在代理部署到生产环境时捕捉到回归问题，并持续扩展测试覆盖范围，从而逐步改进系统。本课程将涵盖的一些工具包括：

跟踪检测工具，用于了解代理底层运行情况；包含LLM作为评判者的评估运行器；可用于重复实验的数据集；用于收集人工标注和生产反馈的功能；以及一个提示词实验场，可用于数据迭代优化。

 在本课中，您全面了解了LLM模型评估与系统评估的对比，为什么基于LLM的应用程序需要不同于传统软件的测试方法，以及智能体在推理、路由和行动方面带来的额外复杂性。常见问题包括智能体在上下文使用中错误选择工具，以及使用同一套工具从开发到生产进行迭代测试的重要性。您将看到所有这些内容在实际中的应用，并很快深入实践代码。在接下来的课程中，您将更仔细地研究如何评估智能体，收集哪些数据，以及如何构建这些评估，以确保您的智能体在现实世界中保持正轨。


---------------

在本课程中，您将从零开始构建一个AI代理，并学习如何评估它。本节课，我们将深入探讨这个代理结构的细节，然后通过一个能够执行数据分析的代理示例进行讲解。接着，您将学习如何用Python编写这样的代理。好了，我们开始吧。代理通常由三种主要类型的组件组成：路由器、技能，以及记忆和状态。

右侧图像展示了一个智能体示例：黑色部分代表路由器，底部深紫色区域表示状态，以及该智能体可调用的三项技能——包裹追踪、商品查询和问题解答。这些组件都能与状态进行交互，路由器则负责处理用户输入并将结果反馈给用户。首先来看路由器，它是整个智能体的核心调度中枢。

某种程度上可以说是代理的大脑。路由器负责决定代理将调用哪个技能或功能来回答用户的问题。因此，当路由器接收到用户输入或来自不同技能的响应时，路由器将决定接下来调用哪个其他技能。路由器可以采取几种不同的形式。它们可以是一个配备了函数调用的LLM（我们将在本课程中使用），也可以是一个更简单的NLP分类器，甚至只是一个基于规则的代码。

一般来说，你使用的路由器类型越简单，性能表现就越好，运行也会更稳定，但路由器的功能范围也会相应缩小。像具备函数调用功能的LLM（大语言模型）这类工具，虽然功能范围非常广泛，但相比基于规则的代码，其可靠性稍逊一筹。而这正是评估可以发挥作用、帮你缩小差距的地方。

有些智能体不会采用单一的路由步骤，而是将这种逻辑分散在整个智能体中。采用这种方式的流行框架包括LineGraph和OpenAI Swarm。它们仍然具备路由逻辑，但不是通过一个单独的路由步骤，而是将这一职责分散在智能体本身中。接下来，技能是智能体拥有的独立逻辑块和能力。因此，技能使智能体能够真正执行任何任务，比如通过API与外部世界连接、调用数据库，或者完成智能体能够实现的各种不同任务。

每个代理都将拥有一项或多项技能。没有任何技能的代理实际上无法执行任何操作，这并非实际应用场景。技能由多个独立步骤组成，包括大语言模型调用、应用程序代码、API调用或任何其他您希望使用的代码。一个非常常见的技能示例是RAG（检索增强生成）技能。在这种情况下，您的代理可以具备检索增强生成能力，处理从嵌入到从向量数据库查找数据，再到利用检索到的上下文进行大语言模型调用的全过程。

而这一切都被囊括在一个单一的RAG技能中。因此，你可以看到技能如何能涵盖多个不同的步骤，而整个LLM应用程序在代理的上下文中可以被视为技能。一旦技能完成，在大多数代理上下文中，它们也会返回到路由器，以便可以选择是返回给用户还是从那里调用另一个技能。代理还利用记忆和状态来存储信息，这些信息可以被代理内的每个组件访问。

Typically, memory and state are used
to store things like retrieved-context, configuration variables, or very commonly,
a log of previous agent execution steps. This last one is probably the most common
that you'll see. Many LLM APIs actually rely on being passed in a whole dictionary or list of messages about what the agent has done previously,
before choosing the next step to call. The OpenAI function calling router
that you'll use throughout the course uses this approach, so you'll get
very familiar with that approach. Turning to your example agent that you're
going to build throughout this course. The example agent that you're going to
create is a data analysis assistant that can help you understand
and ask questions over a sales database that you have. That agent has a few different skills. It has a data lookup skill that can query
information from that attached database. It has a data analysis skill
that can draw conclusions from that data, spot trends and make calculations. And then finally,
it has a data visualization skill that can generate Python code
and create graphs through that code and visualizations. Looking
at your example agent visually here you see you have a user who's
sending in queries to a router in this case
GPT-4o-mini call with function calling. And then that router will call
one of three different tools here, the lookup sales data tool, data
analysis tool or data visualization tool. And then those tools will return
back to the router, which then decides to either return to the user
or call another tool from there. Now we're using the word tool here
because that's what's expected by GPT-4o-mini. That's analogous to skills in this case. So your lookup sales data tool would be
analogous to a lookup sales data skill. Diving into each of these skills
a little bit more deeply, you see that each
skill has a few different steps that it can actually go through
to accomplish its task. The Lookup sales data tool, for example,
first prepares a database so it loads in a local database
and make sure that it's ready to query that database. Then it will generate SQL using another
LLM call to query that database, and then finally execute that SQL
and return the result all the way back to the router. Similarly, the data analysis tool makes a single call
to generate an analysis. So it just makes a single LLM call
and then returns that response back to the router. And finally, the data visualization tool actually
makes two sequential LLM calls. First, to generate a chart config and then to generate Python code
based on that config. The reason for the two calls in that case
is that while you could just ask an LLM to generate code straight away
and do both steps at once, you'll get more unreliable responses
that way. And because chart visualizations in Python
are somewhat formulaic, it helps to first generate this chart
config with a few key variables and then generate the Python code
based on that chart config. So splitting up the task into two simpler
tasks instead of asking for the LLM
to complete one more difficult task. In this lesson, you've learned about the major components
of agents, routers, skills, and memory, and you've examined the example agent
that you're going to build. In the next video, you're going to go through a notebook
and actually implement and build this example agent.

---------------


In this notebook, you're going to build
the agent that's going to be used
throughout the rest of this course. As you've seen, this is an example agent
that uses a few different tools and a router to answer questions
about a data file that's attached. So first thing you want to do is import a few different libraries
that are going to be used by this agent. So you have an OpenAI
library, Pandas, Json, and then a library called duckDB,
which gets used for some of the SQL code that you're going to run, as well as pydantic,
some markdown displaying, and then some helper methods
that are used to gather the OpenAI key and a few other kind of things
specific to your environment here. And then next thing you're going to do
is you'll want to set up your OpenAI client
that will be used throughout this agent. In this case, using GPT-4o-mini. And then you're ready
to start defining the tools. So as mentioned there's three tools
that the agent is going to have. The first is a database
lookup tool that it will use. And so there's a local file
in the notebook that is store sales price
elasticity promotions data file. So, here's a path to that file
that's going to be used. And this is a parquet file
which is just a file that's going to be used to simulate the database
that your agent can interact with. And this file contains a bunch of
transaction data of different store sales. So you can imagine every time there's a sale in a particular store,
one new entry gets added to this file with the product skew
and with the any promotions attached to it, as well as the
price and the cost of that sale. So now,
there's a few steps to this first tool. First, the tool needs to generate SQL that can be run on this database
that you have locally. And then it needs to actually run
that SQL command and get the result back. So to generate
the SQL you're going to use a prompt here. So you'll need a prompt
to actually generate the SQL code. And so you can see that here
generate a SQL query based on a prompt. Do not reply with anything
besides the SQL query. And you can see the prompt
is going to be passed in as a variable, as are the columns for the table
and the table name, so that the LLM knows what available columns there are. What the table is,
which is critical when it creates the SQL code that's going to run. So now you can set up a method
to actually generate the SQL code itself, using that GPT-4o-mini model
that you established. So here is the generated
SQL method that you might use. So it takes in a prompt list of columns
and then table name. And then it formats
this SQL generation prompt that you have up here
with those different variables. So you'll see the prompt, columns
and table name passed in. And then that formatted prompt
will be sent over to GPT-4o-mini as a user message. And then that response will get returned.
And now you need to incorporate that SQL generation code into a tool
that can be used by the agent to actually run the code
after it's been generated. And so for that you'll find another method
here. In this case, this is going to
be called lookup sales data. And so what this method will do
is it takes in a prompt. And then it will try to first create the database from that local parquet file. So you can see it creates a data frame
here using pandas with that local file. And then it uses the DuckDB library here
to say create this table
if it doesn't exist already, which will create a SQL database table
based on that data frame that you have. And then next, you'll use the generate SQL
query method that you just created
to create an SQL query here. And that can then be slightly modified
here to remove any extra trending white spaces
or any other extra characters here, and then run
using duckDB to receive a result. This step right here can be useful
because oftentimes you'll have the LLM respond with SQL at the top
to kind of format the response there. It doesn't just respond with code. Sometimes it includes that extra SQL bit, which is why you're removing
some of those characters here. DuckDB here is used to take the data frame that you have
and turn it into, a corner database in memory
that you can access. And that will make it easier to run
any kinds of, queries across that database and will work
fairly efficiently and very quickly. And it also allows you to use generated
SQL code and query that database
using SQL code in a very easy way. Now that you have your tool defined here, it's always good to make sure
that the tool is actually working. So you can put in an example query here,
print out the result, say example data and call your method
with show me the sales data for store 1320 on November 1st, 2021,
and you can run that. It might take a second to run
and you should get a response back. Should get a response
back that looks something like this. So you see data that's been retrieved
from your SQL database. Great.
That's one tool down. Two more to go. So next tool is a data analysis tool. And so this is a tool
that can take in data and then makes an LLM call to conduct analysis
and capture trends from that data. So, similar to before you're going to want to have a prompt
for that particular tool. In this case, it's pretty straightforward. It's analyze the following data. And there's a variable for data. And then your job
is to answer the following question. And there's a variable for the prompt. Now you'll have a method
to actually make that call. So you can have an analyze sales data
method here that takes in a prompt and some data. And then similar to before, you're going to format the prompt
you just created with those parameters, make a call to your OpenAI client and get a response back as the analysis. And then you can always do
some little bit of error checking here
to see if the analysis doesn't come back for whatever reason,
then you can still return within no analysis could be generated. This kind of thing
is especially helpful in agents because if you have a response
that breaks halfway through your agent, you want to make sure that that doesn't
cascade and break everything else
inside of your agent. So, good error checking is almost more important
in agents than it can be in other tools. And now, with your analyze sales data method defined, you can test
and make sure that that's working. So you can call analyze
sales data the prompt: What trends do you see in this data? And you can pass in the example data
that you retrieved with the previous tool. And you should get back
something that looks like this, which is some markdown
formatted response from your model. And moving on to the third and final tool
that the agent will use here. This third tool
is a data visualization tool. And if you think back to the previous
lesson, you'll remember that
this works using two different steps. So at first generates a chart config
or what we're calling a chart config. And then it will use that chart
config to generate some Python code
to generate a visualization. So starting with the chart config here. You're going to again have a prompt that
you want to use to generate that config. You'll see it says generate a chart
config based on some data. Your goal is to show a visualization goal. And then here, you're going to use a tool
called pydantic along with a feature of OpenAI
called Structured Output. So that you can ensure that when you make
a call to generate a chart config, it matches a very specific format
that you're defining here. So first thing to do that is define
the actual format that you want to use. So in this case this visualization config is a set of a dictionary
with four different variables here. Chart type x axis, y axis, and title. And then you can see there's a description of what
each of those different variables are. And you'll use that in just one
sec here with your OpenAI call in order to make sure that the output
matches that format exactly. So now you can define a method for extracting the chart config
and creating that chart config. It'll take in some data
as well as a visualization goal. And then again, it's going to format the chart
config prompt that you set up above. And then there's going to be slight
difference in this call that's made here. Because now you're using that structured output feature
to tell OpenAI that it should respond in a format that matches that visualization
config that you just created. So one important thing to note is there's
a slightly different message string here to allow you to include that response
format variable for visualization config. And again
what this will do is it will make it so that when OpenAI responds,
when GPT-4o-mini responds, it will be in the format
that matches your visualization config. So you'll be able to access things
like the content dot chart type
or content dot x axis, or y axis, or title. And the broader goal of this approach
is you're really scoping down the first thing you ask the model to do to say
instead of just generating Python code, it really just has to pull out, okay,
what's the chart type I should create? What's the x axis, y axis and title
that I should give to my chart? And then you can pass that into an LLM
in the next step of this tool. Do you have a more defined goal
for that LLM to generate code for. So moving on to the next step within
this third tool is to actually create the chart code
itself. So similar to before, you're going
to create a prompt to start off with. It says write Python code
to generate a chart based on this config. And it's passing in this config here. And then you can define a method to use that particular prompt
to generate Python code. So you'll see this create chart
method here takes in a dictionary which is that config. So that's what it was
generated by the last step there. And then it formats
your create chart prompt here. With that config. Makes your standard
OpenAI call gets its response back. And then similar to what you did
with the SQL code, it's always a good idea to make sure that, any sort of prefix here
that gets added by the model like this Python one you see here, is getting removed and replaced so that
you're only left with a runnable code. And now the last thing to do with
this tool is just compose those two different steps
together into one method. So now this generate visualization method is going to take in some data
and a visualization goal. It's first going to create a config
with the extract chart config method. And then it will create some Python code
using the create chart code. And finally, you can test and make sure
that this is actually all working correctly. So you can say generate visualization
with some example data and then a visualization goal. In this case, that's similar to the prompt the user might have. And we'll print out the response as well. And you should now see some Python code
that's been printed out here that will generate
a particular visualization for you. And you can actually take it
one step further and run the code here using the command execute code. And you'll see a visualization. In this case it looks like
the code was created correctly. But the visualization
looks slightly off here. So, this is where evaluation
is going to come into play later on because you'll be able to catch cases
where maybe the visualization is slightly off. One important note is that you always have the option to include
the ability for your agent to run code, but it's something you have to be
a little bit careful with because if you just run code that's been
created by an LLM, it could have errors. It could do things that you don't expect
it to. So it's always good to run code
within an enclosed environment if you're giving your agent
that capability. In this example agent, you left that
capability out just to keep things safe. But something good to know if you're incorporating the ability
to run code into future agents. Now, with all of your tools defined,
the next step is to put those tools into a format
that can be understood by your router. In this case, you're using an OpenAI,
GPT-4o-mini router with function calling. So there's a very specific format that
OpenAI expects to be pass those tools in. You can add that in here. And what you'll see is that each object
within this array of tools has a specific Json format
that's been defined. So in this first one, this first tool here
is your lookup sales data tool. So this is a method
that you defined earlier. And you'll see there's a description
in here of what that tool does. Or it can do lookup data
from this particular file. And then you'll see
that certain properties have been defined for things like the prompt parameter
that needs to be passed into this tool. And so this is what tells the router
how to choose this function. And if it chooses this function to call
what parameters need to be passed into it. So these descriptions are really critical both for the description of the function
as well as descriptions of the parameters. Because if you get these wrong, the router
may choose not to call your function. It may not understand
what that function does, or it may think the function can do things
that it actually can't do. So this is often
something that you end up changing a lot of times when building an agent
is getting the description and the parameter
descriptions exactly right. So there'll be an entry here
for each of your different tools. As well as a mapping at the bottom here to say this name here
matches this particular function, which I got used in a second. Awesome. You're now ready to define your router. To define your router
you can use a few different approaches. Let's walk through the code here. So in this case, this run agent function
that you have it's going to be what runs your router. Oftentimes, people will set up routers
to either work recursively where you continually call a function
that represents your router. Or in this case, you could go for something
even simpler and just use a while loop here that you break out of
when the agent completes its task. Looking through the code that you see
here, you have the messages object
that's passed into this function here. And then first we'll make sure that that messages object
matches the right format that we expect. So in this case
you want to have it as a user message. And this is a format that's expected
by OpenAI in this case. So you want to check for cases
where maybe you've passed in just a string as opposed to a dictionary of
these messages and correct for that case. And then if the system prompt has not actually been added
into this already, then you'll want to make sure
that that's been added. We'll define
that system prompt in just one moment. And then to actually run your agent,
it's going to follow a similar loop here where you first make a call to OpenAI
with your tools object that you just set up up there,
as well as the messages, and then you'll receive a response back to look for if there's any tool
calls inside that response. And then if there are, you're
gonna want to handle those tool calls. So you'll need to define
what that means in just a second. And then if there's no more tool calls
then you can return the response back to the user. So there's a couple more pieces
you need here for the router. First, you need this system prompt variable. So your router needs
some sort of system prompt. So you can define that
a very basic version of this is just you are a helpful assistant that can
answer questions about this data set. And the last piece you'll need is a way to handle the tool
calls that get made by your router. And so you can add in a method
to handle those tool calls. So here this is just looping through
each tool call that gets made. And then for that tool call
looking up the corresponding function name parsing out any arguments
that have been included in the tool call, and then calling that function
with those arguments and appending the result
back to the messages object. One other important thing to note here
is that OpenAI specifically,
as well as some other models, rely on a behavior where if they tell you
to call a particular tool, if that's the response you get back
from your router, then in the next call that you make to them, you need to include
a response of that particular tool. So they'll give you a tool call ID, and if you don't include back a tool message here
with that particular tool call ID, then they'll actually error and say,
hey, you need to call this particular tool and you need to give me a response for
that tool call that I asked you to make. And now with each of these
you are ready to go and run your agent. Pull in an example here. You can ask it a question like "show
me the code for a graph of sales by store. In November 2021
and tell me what trends you see." And you'll see it start to run, and you'll start to get
some print statements from your router there
where it's making calls to OpenAI, it receives some tool calls back,
it's going to process those tool calls. Then go back to the router
and you'll see that loop complete a few times as the code
runs here. And once you see a message saying that
there's no more tool calls to make and it's returned, the final response,
you can print out your result. And you should see in this case some code to generate that graph. As well as some of the trends that were
gathered by the data analysis tool. Congratulations. You now have a working agent
that can answer questions based on this local database
of sales data that you captured.


----------------


Now that
you understand your agent example, the next step is to add
observability to your agent code. This will give you insight
into the trajectory or sequence of steps your agent takes to respond
to a user's query. You'll learn how to instrument your agent, which means adding code
to track your agent steps, and then visualize the collected
information or traces in the Phoenix UI. All right. Let's have some fun. When it comes to observing your agent, there are a few key concepts
that are important to know. The first is observability. Observability is a general software
concept that refers to having complete visibility
into every layer of your application. In the context of LLM applications, that's
often tracking things like the prompt, the response, token usage,
and then any calls that happen around the calls
that you're making to LLMs. Observability
is typically made up of traces and spans. Traces refer to full run
through of your application, so a single trace would be one and end run of your application from an input
all the way through to an output. And then traces are comprised
of multiple spans. Spans are individual steps of calls to LLMs or code, or lookups to databases
or whatever might be. So one trace is made of multiple spans,
and they're typically presented in this hierarchical way where spans
can be nested underneath each other to show that they're operating
within another span. And then a set of spans
will make up one trace hierarchy. Spans are typically presented
in this hierarchical manner here where you have individual spans
that are nested within each other,
all making up one trace. And you might have spans for LLM calls
as you see in the image here. You might have spans for tools. You might have chain spans which are just general logic steps
that happen within a trace. Traces and spans come from a framework
called OpenTelemetry. OpenTelemetry is often
abbreviated to OTEL, and it's one of the most widely used
standards for application observability,
even outside of LLMs and AI agents. OTEL includes the idea of traces and spans
that are captured within your application, as well as some standards
for how to capture those traces and spans. Often this process is referred to as instrumentation,
which you'll learn more about in a second. And then OpenTelemetry also includes
the concept of collectors and processors to receive those traces and spans,
and then enumerate them in a platform that allows you to visualize those traces
and run later evaluation on them. Throughout this course,
you're going to use a tool called Arize Phoenix,
which serves as one of those collectors to allow you to receive,
visualize, and evaluate traces. This is
what a trace looks like in Phoenix. You're going to see a lot more of this
throughout the notebooks as you start
implementing some of these techniques. But you'll notice
there's one trace here of Agent Run in the image
that's made up of multiple spans. You have an LLM span in orange. You have your chain spans in blue. And then your tool spans in yellow. Traces and spans are captured
in your application through a process called instrumentation. Instrumentation refers
to the practice of marking which functions or code blocks within your code you want to track as spans
and what attributes you want to attach to those spans. You can do this
manually with clauses or decorators, as you'll see
in the next notebook as well. However, tools like Phoenix do automate
some of this process for you, especially if you're using popular libraries
like OpenAI or LlamaIndex or LangChain. Observability
is important for a few different reasons. First, when you're getting off the ground
and building your application, observability
really simplifies the debugging process. As you're building. It's much easier to debug
by going through a nice visual trace than it is to pore over print statements
and logs in your application, so it makes it easier to get going. But then, as you start
launching your application to multiple different users, you start moving towards
production or testing even. Then, it gives you a detailed log
of all of the calls made to your application and all of the different
inputs that you're receiving. So it gives you a large database
of information of all of your different application runs,
so you can monitor how it's performing. And then these traces will become the
bedrock that is used to run evaluation. So later notebooks will have you export
data from Phoenix that you can then use for evaluations at scale across
multiple runs of your application. Finally, putting these together, observability helps
you really start to understand and eventually control some of the more
unpredictable behavior of LLMs. LLMs are by their very nature,
unpredictable in some cases, and the best way to deal with that is by monitoring them and later
evaluating them within your application. In this lesson, you've learned
what observability is, how it works, and why it's important, as well as some of the basic building
blocks of observability like traces and spans, and how those traces in spans
are collected. In the next notebook, you'll implement
some of the instrumentation techniques that you saw in these slides, and you'll set up your first Phoenix
instance here to collect those traces.


---------------

In this notebook, you're going to add
tracing and observability into the agent that you previously built
in the last notebook. So you'll start from the agent
that you've already built. And then you're going to add a few things in here to allow you to trace
that agent using Arize Phoenix, and then get some better observability
into what your agent's doing. So as before, you're going to start out
by adding some new imports here that will be used for Phoenix. So importing here Phoenix
as well as some basic libraries, and then importing a few things
from Phoenix here to set up some of that tracing observability registering here as well as some libraries
from OpenTelemetry, which you just learned about in the previous slides,
and then a library called Open Inference. Open Inference is a library
that's used and created by the Arize team, and it helps translate some concepts
in OpenTelemetry to work better with LLMs. So those are your libraries there
that you're going to be using. Go ahead and import those. And then as before, you're going to set up
your OpenAI model and client that you want to use. And now you can launch and connect
to an instance of Phoenix. So Phoenix is an application
that can receive the traces that you're going to send
from your agent here. And then can visualize those in a UI. And so in this environment you've already
got an instance of Phoenix running. There are a few ways that you can
launch Phoenix. You can either use this command
to launch Phoenix inside of a notebook. Alternatively,
you can run it on your own local machine, or you can access it through a web
app on the Arize website. In this case, in these notebooks,
Phoenix has been launched for you, so you don't actually have to run anything
here. You've already got an instance
that's running. Now, Phoenix has a concept of projects, and so projects are used to separate out
some of the traces that you're sending. Maybe you're tracing multiple
different kinds of applications or agents. You might want to group some of the
tracing into different projects. So here
you want to define the project name. And then you can use the register
method from Phoenix to connect your application
here to your Phoenix instance. So this register method here will take in
the project name as well as an endpoint. This endpoint is just
where your Phoenix instance is running. And that will be
where any traces get sent to. So again
you have it running in your notebook. And so this helper method here
get Phoenix endpoint. It's in the utils function. You can take a look
and see what that's doing if you want to. And that will give you an output that looks something like this where you can see OpenTelemetry
tracing has been set up. And so now any openTelemetry traces
that you capture or you instrument will be sent through
into your Phoenix project. So, next step that you'll go through here
is you've made this connection to Phoenix, but you still need to instrument
your application and mark which calls and which methods should be sent
through into Phoenix. And with what attributes. You can do this manually. However, libraries like Phoenix also have tools to do
some of this automatically for you. So this OpenAI instrument or for example, is part of the open inference library
that we mentioned. And what this will do is if I run that,
it will take in the tracer provider
that we set up here, and it will actually make it
so that any calls you make to OpenAI's library from this point on in the project
will get sent through into Phoenix traced properly. So if you're just using OpenAI calls
this would be all you would need to do to actually set
up tracing for those calls. In this example agent,
you have some stuff beyond that as well too. So you have tool calls that get made. You have some other logic
that's happening. So you're actually going to combine
this automatic instrumentor here with some more manual tracing. And so, to set up your manual tracing,
the first thing you want to do here is you want to get a tracer object from that tracer provider. And then this is what's going to be used
to mark any of those methods that you want to send through into Phoenix
as well. Now with all of that set up, make sure that you've also run
the different cells of your agent so that you can then run and test
your agent as we trace it. Now when it comes to setting up manual tracing for your agent
or really any project, it's always helpful to start from the outermost
layer that you want to trace. The outermost tracer span,
and then work your way down in terms of the detail
that you want to go to. This just helps make sure that you're
capturing the right information, and you give yourself a holistic view
of what's going on. So in this case,
you have your run agent method that's going to be used here
to start your agent. However, this method has a loop in it. And as we mentioned
in the previous notebook, sometimes you might want to use more of a
recursive call here where you're calling this method multiple times
for one given run of your agent. So it's helpful
sometimes to create an actual another outer layer
that you can use, or outer method that you can start
as that very top level trace or span here. So in this case
you're going to create a new method. In this case the start main span could just be called start agent
or whatever you want to use there. And really all this method is doing aside
from some of the tracing stuff, is just calling this run
agent method that you have already. So it's just calling that run
agent method. And again,
if you had that as a recursive method, this would be the initial call
that you would make to that method. And then beyond that it's adding
in some calls to set up tracing here. So this is your first manual tracing call
that's being added. So you'll add in using this tracer object. You'll start the span here with the name
Agent Run. You can make the name whenever you want. And then the open inference span
kind, agent. The span kind is just
what kind of category is the span. It'll map to some of the colors and things like that in the Phoenix
UI that you'll see in a second. So in this case, agent is the example. You'll see some for
tools and for chain later on. And then
that will start that current span. And anything within that with clause block
will be treated as part of that span. You can also set some attributes
for the span. So in this case setting the input to
what's the input value for that given span. In this case it'll be the messages object. And then when the agent completes
you can set the output here. This will be the value
of the return of the agent. And then you can set a status code
if you want to to say this was a completed call correctly.
There was not an error on this call. So you can define that method. And then from there you can start to work your way down
into the agent adding tracer. So next step can be done here is going into your run agent method. And there's maybe a few places
that you might want to start spans here. The most obvious one
is that each time you call your router, you may want to have a span
representing that router call. So here's what you can add to do that. So there's been a couple updates
to the method here. Everything up here is the same. No changes
there as you get down to your while loop. Here's the first change that you'll see. So adding that same call from before
tracer start this current span. This time calling it router call and then
adding in the open inferred span kind. This time adding in chain as the span kind.
Chain is just a basic logic step. There's no LLM call or tool call or it's
not an agent, it's just a chain. Chain is almost the default in a way. So you can start that
and then add in the input. Once again you'll keep your OpenAI call the same here. Add in your status as well. And then you can add in the output either
the tool calls or if you have all the tool calls completed and you're
going back to the user, you can add the eventual response of your agent here
to final response. So you can update that method here.
And then you can add a couple more pieces and then take a look
at what this looks like in Phoenix. So one other area
that you might want to add another span is when you go to handle the tool
calls, you might want to have a span
that captures that particular method and tracks that method. So here's how you can do that. You can go up to your handle tool
call method. And in this case you can use a different
technique for actually marking a span. So instead of using the with clause
you have another option which is to use a decorator
for this method. And this is really useful here
because really this method is totally self-contained. If you just said everything in this
method can be one span and that would work well for the instrumentation
you want to set up. So here you can just mark @tracer.chain. You could do @tracer agent
or @tracer tool for the different kinds of tool spans
or spans that you would want to track. And then what that will do is
it will treat any calls to this method
right here as a single span in Phoenix. And then it will take the inputs to this
method as the inputs, and then it will take any return from this
method as the output value. So if you saw before you were marking
the input in the output manually, this is a kind of convenience way
that makes it so that you don't have to define
those manually. It'll just happen automatically. And so now's a good moment. You've added a few different kinds
of instrumentation spans and traces. It's a good opportunity to take a pause,
run your agent and see what that looks like in Phoenix. So if you're going to run your agent here,
make sure that you're using your newly created start main span method and you can run that. But if you switch over into Phoenix, it should now see
you have some data in Phoenix to review. So if you open up Phoenix
and to open up Phoenix, there'll be a URL provided in the notebook
that you're working in. You open up Phoenix,
you'll start on this projects page. And if you remember,
you set the project name of Tracing Agent. So you should see there's now one trace inside
that you can click into that project. And you should see something that looks like this,
where one row here is your trace. You've named it Agent Run. You can click into this and see a few
different spans that have been marked. I'm going to expand this so you can see those. So each row here within
your trace is one span. And so your first one is this agent Run. So this was what was set with the with
clause inside of start main span. And then you have
the router call as a span. And so this was what was set with the
with clause inside of your start agent. And then you'll also have this tool
called chain span here which is the one that you created using that decorator
on your handle tool calls method. You'll also see these orange calls here,
which are LLM calls that are made to OpenAI. And so these are all coming
from the automatic instrumentor that you used at the very beginning of this video,
where you added the OpenAI instrumentor. So those are automatically
getting captured without doing any of
the manual instrumentation. And if you expand this a little bit you can see
some of the values that are captured here. So, starting in your agent run you have your input and output values
that are coming there. Similarly for router
you'll have your input and output as well. And then if you look at your LLM spans
you'll have a little bit more information. You'll have things like the user prompt,
the system prompt, the output. And then you also have
all the tools that were supplied. So you can see the tool definitions
that were supplied to your agent as well. And in any of these spans you can also
always go into the attribute section and see all of the information
that was sent to that particular span. So this is where you have all of the info. If you want to really
deep dive into some of these. Great. So, so far you've got your agent router and your handle tool calls method, as well
as some of your LLM calls being tracked. But you also have some tools
inside of this agent. And it would be helpful to understand when those tools are actually being called
and what the responses are there. So you can add in some tracing
for those tools as well. So to add tracing to some of your tools,
you have a few different options here. First, you can use those decorators
that you used for your handle tool calls method.
And that's probably the simplest way. So usually the recommended way
to go about doing that. So starting here
you have your lookup sales data method which is your first tool. And so you can add in a decorator
at the top to say tracer dot tool. And then again that will treat this as a single span
whenever this method is called. And then it will take the input
as this prompt value and the output as whatever the return of this method is. Now this particular tool actually calls
some other kinds of methods inside of it. It calls this generate SQL query method. And so it might be helpful to have
that as a separate span to say okay, first you want to have a span
for generating the SQL code and then a separate
one for running the SQL code. So what you can do is
if there's part of a method that you want to actually track
as an individual span, you can use the with clause
that you saw before. So you can add in here
before your SQL query is run. You can add in a with clause to say
start_as_current_span. In this case
you can call it execute SQL query and set the open inference
span kind as chain once again. And then you'll want to make sure
that this is indented so that it's part of that span
that you've created there. And then it's would be good to add
in some of the input and output values that go along with that as well. So, right before this
you can add in the input value. And so you can do span dot set input. And in this case the SQL query that's going to be called
would make sense as a good input. And then after your call is made you
You can say span set output and set status. With the result of that SQL query. And in this case you're defining it
using the parameter name. You can also just define it
by sending the parameter itself whichever way is easier for you to use. And then so now if you run this then that tool will have two spans
that it creates whenever it gets run. And we'll have one for the method itself
using this decorator here. And then it will have a separate span
within that other span nested below for specifically
the SQL query running itself. And it makes sense to do that in this case
because you might generate correct SQL, but then there might be some connection issue or something else
that prevents it from being run correctly. And it's helpful
to see where that error might be. So if you see that the internal span
here fails but the external one succeeds, then you know that there was an issue
with connecting to your SQL database in this case. And so finally you can add in similar
tracing to your remaining tools. And then that'll probably cover all the
tracing that you need for the agent here. And so this is fairly straightforward
to add in here. Your next method here
is pretty simple in this tool. So you can just use the decorator
to say trace this as a tool. Again
it will grab the input and output for you. Nothing really more that you need to do in
this case to trace that particular tool. And then for your data visualization tool, again, you can use the decorators as well. This is another case
where you have a two-step tool. But in this case you have multiple steps that are split
across two different methods. So you can keep things simple
and use the decorators because those contain
all of the logic for that particular step. So here for this extract chart config. This in itself is not a tool. This is just part of a tool. So it makes sense to call it chain here. And so you can say @tracer dot
chain as opposed to @tracer dot tool. For this extract chart config method. And then you can do the same kind of thing for the other method that gets called here
which is your create chart method in your @tracer dot chain. And also make sure that you run
any of these cells again as you added things to them
just to update that. And then finally
you can go to your last method here this generate visualization
and mark this as a tool. So you've marked three different tool
spans for your three different tools. You've marked a few different chain spans
within those tools. And you've marked your agent
and router spans as well. Now you've added
all the tracing that you need to so you can go back down
and run your agent again, and you should see some
more detailed information. And so
now if you jump back over into Phoenix, you should see you have a new trace
that's been run. So you should now have a second entry. And if you click into that, it'll look
slightly different than the one before. You'll have a few more spans here. Notably, you'll have this yellow
one of your lookup sales data tool. So that was one that you captured with
that decorator on your lookup sales data. And then within that you'll also have
a chain span for executing the SQL query. And look at the inputs
and outputs of those. this is the one that's been captured
using the with clause inside of your lookup sales data method. And if you ran more complicated queries
that you used other tools, you would see those tools
appearing as well too. So now you have all of the information
that you need to understand what's happening
inside of your agent at any given step.


--------------


0:00 现在是时候开始设置您的评估了。 0:02 在本课中，你将重点评估每个代理的技能， 0:06 以及路由器的能力 0:07 根据用户的要求选择正确的工具并正确执行。 0:11 你将了解三种类型的评估器：基于代码的 0:14 名 评估员、一名法学硕士作为评委，以及人工注释。 0:17 然后，你将对你的代理应用适当的评估器 0:20 的 例子。我们开始吧。 0:23 运行评估主要使用三种技术。 0:26 第一个是基于代码的评估。 0:28 第二是法学硕士和评委评估。 0:30 最后是人工注释。 0:33 每种技术都可以应用于代理来衡量 0:36 评估代理的不同部分， 0:38 但它们也可以应用于其他基于 LLM 的应用程序。 0:41 所以这些技术对两个代理都有效 0:44 以及更传统的 LLM 应用程序。 0:47 代码库评估是最简单的评估类型 0:51 在 LLM 或代理评估方面，它们是最相似的 0:54 进行传统的软件集成测试或评估。 0:58 基于代码的评估 0:59 涉及在应用程序的输出上运行某种代码。 1:03 一些常见的例子是检查 1:06 如果您的输出与某个正则表达式匹配。 1:08 所以也许你希望回复只包含数字 1:12 或无字母数字字符。 1:13 您可以使用正则表达式进行检查，以确保您的响应与该过滤器匹配。 1:18 您可能还想确保您的响应是 Json 可接受的。 1:21 或者说你经常看到的一个，特别是在聊天机器人中， 1:23 正在检查聊天机器人的响应是否包含某些关键字。 1:27 大多数情况下，公司不希望他们的聊天机器人提到竞争对手， 1:31 所以你可以进行基于代码的评估 1:33 查看该竞争对手的名称是否出现在代理的响应中。 1:37 但也许最常见的基于代码的评估类型是使用这些评估 1:41 将您的应用程序输出与预期输出进行比较。 1:44 如果你有关于输入预期输出的真实数据， 1:49 那么使用基于代码的评估来直接 1:52 将应用程序的输出与预期输出进行比较， 1:56 或者使用余弦相似度或余弦距离 2:00 在这两个不同的值之间进行更多的语义匹配。 2:04 下一个技术 2:05 运行评估被称为 LLM 作为法官。 2:08  LLM 是法官，顾名思义，涉及 2:11 使用单独的 LLM 来测量应用程序的输出。 2:15 通常的做法是抓住 2:18 应用程序的输入和输出，以及可能的其他一些关键 2： 一次运行应用程序可获得 23 条信息。 2:26 构建一个单独的提示来评估 2:29 根据这些输入和输出制定的具体标准。 2:33 将该提示发送给单独的法官或评估员 LLM， 2:37 然后获得法学硕士学位，分配一个标签 2:40 对该响应进行具体访问。 2:43 再举一个例子来说明它是如何运作的。 You can examine this case
of evaluating the relevance of a document that was retrieved in a RAG system. The retrieval span in this case
is made up of a user query. And then the documents
that were retrieved for that given query. That query and then the reference documents are then sent into the separate,
eval template that you see here and then populated
so as to ask another LLM, "Are these reference retrieved documents
relevant to the user's question?" And then that separate
LLM will either say those documents are relevant or irrelevant. An LLM as-a-Judge is really powerful
because you're able to run large-scale evaluations across
both quantitative and qualitative metrics. However, it's important to keep a few
things in mind when using LLM as a judge. The first is
that only really the best models actually
align closely with human judgment. So if you're going to run LLM as a judge,
you often need to use a GPT-4o or Claude3.5 Sonnet
or similarly high-end model to do so. Even with these, though, a LLM as a judge
is never going to be 100% accurate. There will always be some margin of error,
because at the end of the day, you are using an LLM
to assign that particular label. You can mitigate some of this by tuning
either your LLM judge prompt or even your LLM judge model, and you'll learn a little bit more about
these in later modules in this course. Finally, it's always important to remember
to use discrete classification labels as opposed to undefined scores when
setting up the outputs of your LLM judge. So you should always use things like
correct versus incorrect, relevant versus irrelevant, and never a measure like "score
this response on a scale from 1 to 100." The reason for that is that LLMs
don't have a great sense of what constitutes an 83 out of 100 versus
a 79 out of 100, especially when they're evaluating
each case independently. So always use discrete
classification labels wherever you can. The third evaluation technique that you can use
is using annotations or human labels. The idea here is that you can use tools
like Phoenix or other observability platforms to construct a queue of lots
of runs of your application to construct an annotation queue,
and then have human labelers work their way
through that queue and attach feedback, or judge
the responses of your application. The other method that you can use
is actually gathering feedback from your end users. So you might have seen cases in the wild
where LLM systems have a thumbs up, thumbs down response system
where you can rate the responses of that LLM system. And you can use the same technique
inside your application to gather some feedback or evaluation metrics
about how your app is performing. When you
have these few different techniques it can be tricky to decide
which one to use for a given evaluation. One way to think about
this is how qualitative versus quantifiable or quantitative is the metric
that you're trying to measure. If you have something like evaluating
the quality of a summarization or evaluating the clarity of analysis,
that's a very hard metric to assign, quantitative
or code-based measurement to. So that's where you might want to rely on
LLM as a judge or human labels
to understand that qualitative metric. If you have something that's a little bit
more flexible or can be defined in code, like whether an output matches
a certain regex, then a code-based evaluation will work
in that case. The other way to think about these
is whether the evaluation needs to be 100% accurate, or if it can afford
to be less than 100% accurate. If you remember, LLM
as a judge is never going to be 100% accurate
or 100% deterministic kind of technique. So if you need 100% accuracy,
then you'll need to rely on either human labels or code-based evals. What you might notice here is that human labels
is sort of the best version of evaluation. In this case, it's
flexible and it's deterministic. However, in practice it can be hard
to get human labels at scale because it's a very labor
intensive process to do lots of labeling on your data. And if you rely on end users to provide
feedback, there is some selection bias there over
who chooses to supply feedback to your application.
So it's generally not advisable to use that as a large scale
evaluation technique. Now that you understand
that the techniques that you can use to evaluate an agent or really
an LLM application, now you'll learn about the different pieces of your agent
that you can apply these techniques to to run these evaluations. This lesson we'll go through evaluating the router
and the skills. And you'll have a future lesson in this course that goes into evaluating
the path in detail. Starting off with a router. Routers are typically evaluated
in two different ways. First, you'll evaluate the function
calling choice, aka did the router make the right choice
and choosing a function to call. And if you're using an NLP-based classifier
as opposed to an LLM with function calling, you can still evaluate
the function calling choice. And you still use this
sort of evaluation metric. The second kind of thing that gets evaluated in routers is typically
the parameter extraction. So once the router chooses
which function to call, does it extract the right parameters
from the question to pass those into whatever function
it's decided to call. One way to evaluate
a router is by using an LLM as-a-Judge. This is an example of the prompt,
the template that you would use for your LLM as a judge
to evaluate a router in this case. So you'll notice at the top
you have some instructions for the LLM judge
to say what it's going to be evaluating. You have some places where data will be
added into the prompt, in this case the user's question,
and then the tool call that was chosen. And then you have some instructions
saying the LLM judge should respond with a single word,
either correct or incorrect. And then you have some more details on what incorrect means
and what correct means in this case. And finally,
you have some tool definitions information so that the LLM judge knows
all of the possible options that your application had
when choosing which tool to call. Looking at the example here,
you might have a case where a user asks your agent, can you help me
check on the status of my order? Number 1234. And the agent says
"definitely" makes a tool call. In this case, choosing to call an order
status check method and then extracts that one two, three,
four and says this is an order number. So it passes it through as an order number
to that order status check method. And this all looks good. And it gets back a response
saying the order has been shipped. It goes back to
the user says your order has been shipped, and the user follows up and says
"okay, when will arrive?" And the agent decides
to make another call here, this time using the shipping and status
check method. That all looks good so far,
and then it takes that same one, two three, four and this time
as it has a shipping tracking ID in this case that's an order number
and not a shipping tracking ID. So here the agent actually failed
in the parameter extraction task, even though it succeeded in the function
calling task that it had. Now, when it comes to evaluating skills,
you can use a few different techniques to evaluate skills either
standard LLM-based evaluations or LLM as-a-Judge evaluations
or code-base evals. And one important thing to call out here
is that skills themselves are really either just other LLM applications
that have been composed into a skill. The agent can use, or other software
applications like API calls or application code that have again
been compressed into a skill. So all the techniques that you can use to
evaluate skills are really just the same techniques
that you can use to evaluate either standard software
applications or LLM applications. So you might use an LLM as-a-Judge
to evaluate things like relevance, hallucination, question
answer correctness, generated code
readability, or summarization. And then you might use code
based evaluators to evaluate regex matches Json parse ability in your response
or comparison against ground truth. Looking at your example agent,
you have three different skills here. You have a database
lookup skill, a data analysis skill, and a data visualization code gen skill. I invite you to pause the video here
and think of a few different evaluations that can be used to evaluate
each of these three different skills, whether the entire skill
or one single part of the skill. As in the case of this lookup sales
data tool, you have a step to prepare the database,
generate SQL, execute SQL. So you might have evaluations
on the full lookup sales data tool, or on one step of that process. So here are a set of evaluators
that you could use to evaluate each of those different tools. So for the database lookup tool
you could use either LLM as-a-Judge or a code-based eval
to do SQL generation correctness and see if the SQL that's generated
is correct. For the data analysis tool, you could use an LLM as-a-Judge
to check for clarity in the analysis, and make sure all of the entities
that are mentioned in that analysis are correct and match back to entities in the input
or other sections of their data. And for the data visualization
code gen tool, you can use a code-based eval to make sure
the code is runnable. Fairly straightforward eval
that you can run there. Now, it's important to mention
that you could have come up with different evals to use here. And again, eval is sometimes
more of an art than a science. And so if you came up with different ideas
don't be discouraged. Those could be just as correct
or in some cases more correct here too. The last thing
to mention is that you have SQL generation correctness being run with either LLM
as-a-Judge or code-based evals. You'll see what both
those look like in future notebooks. You can either use an LLM
as-a-Judge to judge whether this equals correctly generated, or you can use a code-based
eval to compare the SQL that's generated
against some ground truth data, or the result of that SQL
against some ground truth data. In this lesson, you've learned three
different techniques to run Agent evals: 12:23 法学硕士作为评委，基于代码的评估和人工注释。 12:27 你还学到了 12:28 使用每种技术运行的常见评估类型。 12:31 最后，你的代理应该在哪些部分应用这些技术 12：34 给你评价。 12:36 在 12:36 下一个笔记本，你将申请法学硕士作为法官和基于代码的评估 12:40 了解这些如何与您的示例代理详细合作。


-------------

In this lesson, you're going to take the agent
that you've already built and traced, and now add some evaluations
to measure its performance. Similar to before, you're going to start
by importing some libraries that'll be used here. So there's some that you've seen
before like Phoenix and some other libraries here. And then you're adding in some new ones. So, specifically from Phoenix evals, importing some things like an LLM
as-a-Judge prompt template that's going to be used,
and some methods like LMM classify that are going to help you run
LLM as-a-Judge, evaluations. One other one that's important
to call out here is this nest_asyncio you might see.
This is used to run multiple calls asynchronously and simultaneously
to speed up some of your evaluation. You'll see where
that comes in in a second. Now again
in Phoenix there's a concept of projects. And so just to separate things out
from a previous notebook, you may want to use a different project
name here. So in this case
you could use evaluating agent. And then in this notebook
just to keep things simple, what's been provided for you
is the entire agent that you run that you set up previously is in the utils file
that's attached to this notebook. And so you're going to import
a few different methods from there. But if you run this you'll see
the same output that you saw before in the previous notebook saying "tracing
has been connected and set up there." Again this is all the code
that you created in the last notebook. It's just a pretty long notebook already, so it's been separated out
into a utils file for you. But this has been set up with your agent to run and connected
to the Evaluating Agents project. So here, anything that you call to run
an agent, start main spans either of those methods will be traced and
sent through into the Evaluating Agents project within Phoenix. It just lives in that separate
utils file to keep things clean here. You've also imported tools which are going to be used
for one of your evaluations as well too. So now when you're evaluating your agent, you can go for two different
kind of approaches to start evaluating it. One starts from a data set of examples. You can run through your agent and you'll learn more about that in the next modules
where you'll cover experiments. The other way is to run
real-world examples through your agent. Trace those examples
and then evaluate the performance of the agent using those traces. So, in order to do that, you can set up
some basic questions for your agent. Things like what was the most popular
product skew, total revenue across stores, some other kind of queries
that are going to come through there. And then you can loop through your agent
and call the start main span method that you imported above
with each of those questions. So this will just loop through each of those questions to your agent
and have those traced in Phoenix. So this cell will take a minute or two to run. Go ahead and kick off. Great. Now once your calls have all finished
running, you can jump back in to Phoenix and see the traces
for all of those different runs. If you start from the projects view, you should now have a new project
that says Evaluating Agent. And you got six traces in there. So you'll see a row
for each run of your agent. And you could click into those and see
more details on what that agent executed for that given step, including some calls that are using multiple different tools
that you can trace. Now, in order to evaluate your agent,
there's a few different pieces of the agent that make sense to evaluate
as you learned in the previous slides. So, one of those is the router. So you could evaluate this router call
that's being made where it's deciding
which particular tool to call. So that would be one thing to evaluate. Or you can evaluate
some of the skills like the sales lookup data skill
or the generate visualization skill. And now the pattern of evaluation
in Phoenix typically involves exporting spans
from your Phoenix instance, either using LLM as-a-Judge
or a code-based eval to add labels to each row of those spans
and then uploading them back into Phoenix. So let's start
by evaluating the router itself and using an LLM as-a-Judge to do that
router evaluation. So you're going to want to export
some of these spans from Phoenix. You can do that using code. But just to show how you would filter down
to the spans that you care about. For the router here
you really care about within the router call this first LLM call
that's being made. So if you look at this first
LLM calling, you can see
this is where you have a users question. And then you have the response here, is one of these tools. So in this case the lookup sales data tool. That was the response
back from the router. So this is all the information that you want to be able
to grab and export from Phoenix. And you can actually look at some of
the attributes here to see the information associated with these. The first is that you can see that there's
an open inference span kind of LLM. So that's one way
that you could filter by this. Or you could use any of the other
information inside of the attributes here to filter for this particular
kind of span that you want to use. Now back in your notebook to evaluate
your router using an LLM as-a-Judge. There's a template that's provided
as part of the Phoenix Library for this kind of evaluation, because this is a very common evaluation
that oftentimes people want to run. So this tool calling Prompt template is an
LLM as-a-Judge template that you can use to evaluate your function
calling of your router. So if you look through
what this is looking for, you can see all of the instructions
given to the judge. And one thing that you see here
is that there's two variables in this case up here, and one variable down below
that need to be passed in to the template. So, first is the question the user asked
and then the tool call that was made. It also looks for the tool definitions, which is a set of
all of the possible tools and their definitions
that could be called so the LLM can compare against that. So, next you can export the required spans
from Phoenix using some of those filters
you just saw in Phoenix. So there's a few things
that you can do here. So, first off you can use this span query method to set some sort of filter
that will filter down the spans in your project and export
any spans that match the filter. Here, you can use span kind of equals LLM. And this will give you all of the LLM spans
for both the router calls as well as other LLM spans. So you're going to do a little bit
more filtering in a second there. And then you can choose
which columns or attributes you want to export as part of that call. And so here you want to export import dot value and LLM dot tools. And you want to export them
using the headers question and tool call. Those headers are important
because they match, you scroll up a bit, they match what's going to be looked for
by your template. So question and tool call and then ask for
the input dot value and LLM tools. Those are attribute
names inside of Phoenix. So if you want to quickly jump
in over to Phoenix one more time, you can see in this case in your attributes. You can see input here dot value will give you the full messages
object that was input there. And if you scroll down further, you can see LLM dot tools here, you can see the tools
that that were chosen. So now you can export
those different values. Run this query on your Phoenix client and that will export
those particular spans. You can see
they're also passing in the project name. And then one other thing
that's being done here to filter down. Because again this will pull up all of the
LLM spans, even those that are not to do with tool calling or routing in this case. So you can also just use this dropna
method to remove any spans that were exported
that don't have a tool call. So they have no value here for LLM tools. That's an easy way
to just filter down to just the LLM spans that have to do with routing. So if you run the cell now you should see your tool calls
data frame. It's been exported. Should have two columns
question and tool call. And again those match what
the template above is going to look for. And then you'll have the question from the user
as well as the tool calls that were made. You'll also have this context dot span
ID column. This is what corresponds
to the particular span inside of Phoenix. And it's going to be what allows you
to upload the data back into Phoenix and have it
match up in a little bit here. So now you have this data frame. The next step is to add labels
to each row of that data frame based on your template above. So here's how you can do that. There's a few things going on here. First, the LLM classify method is a method
that's supplied by Phoenix that will take each row of your data frame
and run it through some provided template. So in this case
your tool calling prompt template. And then it will take a response. And in this case it will snap it to either
a correct or incorrect value. That just ensures that you have really
specific labels that are being used. Sometimes the LLM might respond
with correct with a capital C as opposed to lowercase. And so this rails function here
will snap it. So all the values are consistent. And then providing a model that's going
to be used to actually run this prompt. So this is your judge model
in this case using GPT-4o. And then this variable here will allow to provide an explanation
for why it shows that particular label. There's a couple other things
going on inside the code here. One is that this suppress tracing
tag up top. This has been added
because if you remember, your agent is set up to trace
any calls that are made to OpenAI. So if you didn't have this method saying, hey, turn off tracing
for everything happening inside here, then you would actually see some spans
tracked to your project for all of the evaluation calls
that are being made. So you want to make sure your LLM
as-a-Judge spans themselves don't get traced. So you're suppressing training there. And then on the template row here, there's
some kind of wizardry going on here to add in and replace the tool definitions
with your tools object. That's the idea here. So if you quickly
see your template from earlier, it's being passed in this tool's
definition, which is a definition of all of the possible tools
that could be called. And you want to make sure that that's
being replaced with the Json dictionary of tools
that your agent is using, so that your LLM judge knows
all the possible tools that could have been called,
so it can make its judgment appropriately. And finally, one other thing
that's happening here after your LLM judge runs, is you're also adding
in a score column that is just one if the LLM judge
label is correct, zero otherwise, that's used to have a numeric value
attached to your LLM judge, which is used in
some of the visualization. So now you can go ahead and run that and you'll see some LLM classified progress
bar being made. These calls are made asynchronously
because of the nest_asyncio library you imported all the way
at the beginning of this notebook. And then you should see a response
that looks like this, where you have some labels
as well as explanations. And then if you scroll to the right you'll also see
you have some scores, values 1 or 0. So now you've appended some labels onto your span IDs. The last thing to do
here is upload these back into Phoenix so you can visualize the data there. So you can use this log evaluations method
which will take a couple things here. It'll take the name of the eval
that you want to use as well as the data frame that you just created with
all of those labels and context dot span IDs. Now, if you jump back into Phoenix,
you may need to refresh your project, and then you'll see there's a new value
at the top here of your tool calling eval. And so you'll see a numeric value. Again
because you added that score and it's able to have a numeric
and a percentage value here. And then if you start
going into any of your runs and you click on your router span right here, you'll now see that
the feedback tab here has a new entry. And you'll see it's now labeled either
correct or incorrect. And then it has an explanation as to why
that particular label was chosen. And so you can use these to get some more information about what's
happening inside of your application. And you can actually filter down as well
if you want to go to spans as opposed to traces,
to see all of your spans in Phoenix. And then you can set a filter
for your evals to be tool calling eval, as a name of your eval. And then let's look at just
the incorrect ones. You can look at... And now you'll see there's cases
where your router was incorrect there. And you could click into those
and see the feedback as to why that particular run
was incorrect. Awesome. So now you've evaluated
your router using LLM as-a-Judge. You can add in some evaluations
for your skills as well. One that might make sense to do
would be evaluating your generated code
for your generate visualizations. And so one way you could do this
you can say okay I've got my tool here,
called Generate visualization. Why not export all of the spans for that Generate visualization. And then you can evaluate the code
that's been generated there. So you'll follow the same kind of pattern
as before. Start by exporting the spans
that you want to evaluate. And in this case
you can just use the name of the span equals generate visualization. Super easy there. And then you'll want to export
the output of that span. So you can run this. You probably just have one entry here
given the examples that have been run. And then instead of using LLM
as-a-Judge here, you can define a
code-based evaluator instead. So in this case you
can do a very simple evaluation of just: Is the generated code runnable? And so you could say okay
here is a method that will check for that. It will take some generated code. And it will do some quick checks
to make sure it doesn't have any extra string values inside of it. And then it will try to execute that code
returning true or false
depending on if there were exceptions. And now you can take that method and apply it to your code
gen data frame that you exported above. So you can apply the code as runnable method
here to your generated code column, and then map to the return values of true to runnable and false to not runnable. And then again, you want to add a score. So you can say that one is runnable
and zero is not runnable. And now if you run that method
you'll add in some labels there. You can
take a look at your generated data frame. And now you have some labels and scores. And finally you can upload this data back into Phoenix
using that same log evaluation method. And now if you look in Phoenix
you'll have a new entry at the top this time for runnable code. And it looks like the generated code
wasn't runnable. So in this case it's good eval
caught that for us. So you could use this at scale to run
more checks of your runnable code, this time using a code-based evaluator
as opposed to an LLM as-a-Judge evaluator. And there's a couple other tools that
you had that might make sense to evaluate. Another good example would be the data
analysis tool that you're using to generate analysis on top of any data
that's been exported. And so this one's another good one to use
LLM as a judge for. However, there's not actually a template
built into Phoenix for clarity of analysis. So you might wanna actually
define your own template in this case. So you can actually just define
your own template as text using whatever variables make sense here. And so in this case
you can create your own template and then follow the same pattern
as before. Export the information you care about. And in this case, one way you could export
relevant information is that you could look
for all of your agent spans, which are that top-level span,
and just look at the overall output value, because that's what you can evaluate
whether or not that was, correctly and clearly communicated. So you can export your top-level
agent spans. And then again make sure that when you're doing this,
you're matching any of the names here or the column headers here to whatever
is expected by your LLM as-a-Judge prompt. And now you can use the same LLM classify method, with tracing
suppressed this time using your manually defined
prompt to run your LLM as-a-Judge. And then once that completes, you'll have your new data frame
that you can upload back into Phoenix. You can call this one
a response clarity eval. And you should see that appear in Phoenix
under response clarity. In this case, it looks like 100% out 100%. And you had one third and final tool, which is your SQL code
generation and database lookup tool. So you can add a similar evaluation
for that. Again, defining an LLM as-a-Judge prompt. So in this case
you can see a SQL eval gen prompt. And then you can export relevant spans. So again in this case
you can export your LLM spans. And then you can later
filter down a little bit more specifically here to filter whether the question
that's being asked contains this: generate
a SQL query based on a prompt method. Oftentimes the hardest part of setting up
the evals is filtering down the right set of spans
and coming up with the right criteria. So just know that you can filter
based on anything inside of the spans, including string manipulation here,
or just looking to see if the prompt contains
a certain kind of value. Export your data there. Once again, run your LLM as-a-Judge prompt. And then finally upload
that data into Phoenix. And so now you should have four different
evaluators in Phoenix. You have one evaluator for your tool
calling and your router. You'll have one evaluator
for your SQL code gen, your response clarity
and your generated code visualizations. So at this point, you now have at least one evaluator
set up for each main part of your agent. And you can use these
to get some directional indications of how your agent is performing
across different types of queries.

-----------

In addition to testing your agents
skills and router capabilities, you need to ensure that your agent can respond to a user's query
in an efficient number of steps. In this lesson,
you'll learn how to assess this by computing a convergent score
and apply the test to your agent example. Let's go. Now what is an agent trajectory? A trajectory is simply the path
the agent takes through different router, tool,
and other logic steps for a given input. Looking at your example agent,
you can examine a few different trajectories
to get a sense of how these work. First, you can examine a query like:
"Which store had the most sales in 2021?" In this case, the agent would go from
the user to the router and then would go from that router to,
in this case, the lookup sales data tool,
as well as the data analysis tool. Back to the router
and then finally back to the user. Now one important thing to call out here
is that in this case, the agent has been built to allow for
multiple tool calls from one router step. So that's why you see both the Lookup
sales data tool and the data analysis tool being called together
in the one step between routers. Some agents limit their routers
to only call one tool at a time. That's just an architectural decision that
you can make when building your agent. To look at a more complicated trajectory. You might have a query like plot sales
data volume over time. In this case, your agent would go through
a user to the router to your to look up sales data and data analysis tools
back to the router, which then decides to call the data visualization tool
to generate that visualization code that would then return back to the router
and finally go back to the user. So these are pretty straightforward
trajectories so far. All told. But you can imagine that trajectories
get very complicated very quickly. Your example agent has three tools. But agents in production can have ten, 20,
30 different tools that they use. And even some systems will start
to compose multiple agents together. So those agents can work together
with each other to accomplish some sort of goal. In these multi-agent systems you can have trajectories getting
very complicated very, very quickly. Now you might be asking
if your agent's output is correct, does the trajectory really matter
all that much? And the answer is yes,
because efficiency matters in this case. Now, you might have a certain use case,
like if you're working on a hobby project or you're doing some research
where the efficiency of your agent doesn't matter.
That might be okay. But in most production agents or real world
agents, some amount of efficiency matters. If you can answer a user's question
in six steps as opposed to 11 steps,
that means fewer LLM calls. That means less variability. That means lower cost
and lower latency for the user. So how can you track and measure
trajectories? One way to do
that is using a tool called Convergence. Convergence as a measurement
for how closely your agent follows an optimal path for a given query. So you can think about this
as a measurement of how much your agent has converged towards an optimal path
for a certain type of query. So how do you test for convergence? Well, one way to do
it is using the following technique: You can run your agent
on a set of similar queries. In this case it could be
a bunch of questions that ask your agent to retrieve sales data from November 21,
and then construct a set of different graphs
or visualizations on that data. And the idea here is
you want questions that are similar enough where your agent really should take
the same path through each of them, but different enough
that there could be some variation that you can improve upon
within your agent. Next, you run each of these queries
through your agent and record the number of steps
taken for each of those queries, and then find the length
of the optimal path taken. In this case,
that's the minimum number of steps that your agent
took to get through one of these queries. Then you
can use all of those different numbers to calculate your convergence score,
which is a numeric representation of how often your agent
is taking that optimal path. Another way to think about the convergence
score is what percent of the time is your agent taking the optimal path
for that given set of inputs? So a convergence score of one means
that your agent's taking the optimal path 100% of the time, and your convergence
score will always be between 0 and 1. There's a couple things to keep in mind
when running convergence evaluations. First is that convergence evaluations
will typically not catch situations where your agent takes an unnecessary step every time
for every query in your test set. And the reason being here
is that the optimal path in a convergence eval is typically the minimum number of steps
the agent took to get through one of the queries. So again, if you have an unnecessary step
that's taken by all the runs of your agent, typically convergence evals
won't catch that situation. Second, you want to make sure
when running convergence evaluators that you're only running them
using four completed runs of your agent. If your agent takes three steps
to get into a problem and then errors out, you don't want to track those three steps
because that will skew your convergence evaluation data. In this lesson, you've learned
what trajectory and convergence are, why they're important
to track and measure. And then one way to measure
and calculate convergence. In the next notebook, you'll implement
this convergence measurement and evaluation technique.

-------------


So now that you've added some evaluations
to your router and skills for your agent, the next step is to evaluate the path or understand if it's taking an efficient
path for a given query. So in order to do that, again,
you're going to start by importing a few libraries
that will be useful here. Most of the things
you've already seen for Phoenix. And then now you've got some new things
from Phoenix for what's called experiments, which
you'll learn about in just a second here. So you can import those libraries.
And one other thing to call out is that, again, the agent's been imported
from your utils method here. So, you have the same agent that you've
been working with the whole time. You're importing that run agent method. So just know that that code
is sitting in the background. And then here you're still going
to need to connect to the Phoenix client. So in this case I'll just save that
Phoenix client variable. And then the way that you can evaluate
your trajectory as you learned in the previous slides, is by running
a set of queries through your agent and then tracing the number of steps
taken by each of those queries. So in order to evaluate the trajectory
here, you have to actually compare multiple
runs of the agent together. So in order to do that, you're going to use a tool
called a run experiment within Phoenix, which allows you to run
multiple different runs of your agent and then compare those
to each other in certain ways. So here's a good point to pause
and dive a little more into experiments and how they work. Experiments in Phoenix
are made up of a few different steps. First, it's taking a data
set of different test cases, then sending those into a particular task
or job to run and then evaluating
the results of that task. So a data set of test cases
is going to have a bunch of different queries or questions
that you might run through your agent. And typically you'll have an input value
inside of those. examples
as well as maybe an expected output. In this first case,
you won't have an expected output. You'll just have an input. Sometimes you have expected outputs,
sometimes you don't. And that dictates some of the evaluators
you can use later on. So you have a set of examples here. And then you'll run those
through a version of your agent. So in this case just going to use
one single version of your agent. You'll get the outputs
from that particular round. And you'll see actually that you made
some slight modifications to your agent to actually track the number of steps
taken as well as part of that task. So you can define the task
for your experiment. And then once you've collected
all the results of each example through your task, you can
then send those to a set of evaluators. And those evaluators could be
the ones you set up in previous rounds. Or in this case you can do
more of a comparative evaluator, comparing different runs of your agent. And so in this case, there's another variable
that will be added to each example, which is the output of that
particular example through the task. And you'll learn a lot more
about experiments in the next lesson as well too. This is sort of
just your primer for them in this case. So step one there is creating a dataset of test cases. So the way that you can do that
is you have a set of questions. In this case convergence questions. And if you think back to the slides, one of the ways that you test
for convergence here is you send a lot of different
variations of a similar type of query through your agent
and then track the number of steps taken. So notice
that each of these different examples here is all about the number or the average
quantity of items per transaction. So you see the average quantity sold per
transaction mean number of items for sale. Calculate
the typical quantity for transactions. You see these are all kind of variations
on the same question. So the agent should take the same path
for each of these. But sometimes there's variation. So you can take this list of questions,
create a data frame out of them and then upload
that data frame into Phoenix. So now you have a data
set that lives in Phoenix. If you want to,
you can quickly visualize it here Now, if you want to look in your Phoenix window, you can actually see
under the datasets tab here, you'll now have an entry for the data
set that you have just uploaded. You can click into that. And you can see all of the different
examples that you've uploaded. And now you can use this data
set to start running experiments. So for the next step here you have to
define the task for your agent. Now you could just run these examples
through your agent. But again you want to have some count
of the number of steps that were taken. You might want to format
some of the messages so you can make slight tweaks
to your agent as well. When you're setting it up as a task. So in this case, I do a couple of things
here. One is actually create
a method that will format some of the message steps
which you can come back to in a second. And then you'll create a task here
that is run agent and track path. So starting with this task
you'll see it takes in an example. So if you remember each row of that data
set is an example. So you can see in this case
you're taking in an example variable. And then it's calling. It's taking the input value
from that example. And then calling the run agent method
on that particular example. And then finally calling
that format messages step. So the return from this run
agent and track path, it's going to be the length of the messages
inside the agent as the path length. And then the actual message is object. And then this format message step really
just is going to go through all of the messages
in your agent's log. And it's going to format them in
a little bit easier to read sort of way. So you can see the two calls that are made
and make it a little bit easier to compare here. So now with your task ready and your data set defined,
you can start an experiment. So to do that
you'll call this run experiment method. And then what this will do is it'll take
in your data set as well as your task or your function that gets applied
to each row of that data set. And then you can come up with a name. So in this case convergence eval
and a description. So if you run that now,
each row of your dataset will be run through the run
agent and track path method. And you'll get results back. That'll take a second to run because
it's equivalent to 17 runs of your agent. So give that a second to complete. Now you should have some results
that look like this. You'll see all those runs completed. And you can actually click
and see the results of that experiment inside of Phoenix. So if you click into that,
what you'll see is in your dataset, you now have an entry for experiments. And you have this first run
of your experiment that you named convergence eval. And you can actually click and see
all of the outputs for you to run there. In this case looks fairly successful
on each of our runs. And so now what you can do is
you can actually go back into the code and you can apply evaluators to each of
those different experiment runs. And so here's where you can essentially
implement your convergence evaluator. And so because you just run
your experiment, you have written that experiment has something that
you can apply and access in your code. So you can always view the results
of that experiment as a data frame where you have the output, input
and other various columns there. And so that's one thing that you can actually use
to run your convergence eval as well. And so in order to calculate the convergence
first you need to calculate the minimum number of steps that was taken
by all the runs of your agent. And so to do that add in code here to first take your experiment as a data
frame like you've just done above. And then look at the output column
and turn all of those different, variables into values that you can access,
and then calculate the minimum or optimal path by using the minimum
function on the path length variable
within each of those different outputs. So this is just going to give you a number
for the minimum path length taken. So if you run that you should get something says the optimal path length is
you should probably see five in here. That's a that is the optimal path length
that's being used. And one important thing to note
is just the way that this has been set up so far is it's counting. Every message has the path length. So it is including things like
the system message and the user message. That's fine in this case because you're
comparing a bunch of different examples that all include both those variables
or both those messages. You just want to make sure
that you're consistent. So if you include the user message
in the system message again that's fine. You just got to make sure that you do it
on every example that you're testing with. So now you can create a method to use as your evaluator. And in this case you can use
this, evaluate path length method which is going to take an output
and compare it, the path length of that output to the minimum or optimal
path length that you calculated earlier. You're also you're going to use this
create evaluator decorator here. This is totally optional but it allows you to name the evaluator
that you're going to run here and see. Or it'll mark
how it shows up inside of Phoenix. And so now you can take the experiment
that you've already run and use this evaluate experiment
method to take in the experiment. And then this evaluate path length
method that you run above. And that will take all of the results
from your experiment and run them through any evaluators that you add in here. AKA in this case, the evaluate path length and it will give
you a score at the end of it. That'll go pretty quick
because it's just a basic code-based eval. And if you jump over into Phoenix now, you'll see that your experiment,
if you go back to your dataset, you'll now have a column here
for convergence eval. It's named that because of the decorator
that you attached. And in this case
we've got a perfect score of one. Our agent has taken the correct path. Every example here. You might see a different value here. Every time we've run
this we've gotten a different value. So you may see a different value
here for one. And that'll give you an idea of whether or not your agent
is converging towards the correct path.

--------------


So far you've learned
how to evaluate your agents skills, router and path convergence. You'll now
learn how to combine your evaluators into a structured experiment to iterate on
and improve your agent. Let's dive in. Evaluation-driven development is a concept
that involves using evaluations and measurements from your agent
to help guide where you spend your time improving and iterating on
and developing your agent. Evaluation-driven development
is made up of a few different steps. First, you curate a data set of test cases
or examples that you can send through
different variants of your agent. Next, you send each of those different test
cases through those different variants of your agent,
each time changing things like the model you're using, the prompts you're using,
or the agent logic. And then once you run those different test
cases through those different variations of your agent,
you take the results of all of those different experiments
and run them through your evaluators. Those evaluators
will give you a set of scores that you can then use
to compare the different iterations of your agent on an apples to apples
basis. Now, this is presented
linearly in the visualization here, but in practice
this tends to be more of a cycle, especially as you move
your agent into production. LLM apps require kind of iterative
improvement as you're working on them. And so typically what will happen is
you'll you'll go through this full process, your release or agent or you'll
get it into some other people's hands. And then you'll realize that you wanted to add in different test cases
or different evaluators. And so you can always update
and change each of these different pieces as you're going. And then you can create a flywheel here,
where you're incorporating information from production back into your development process
using evaluation-driven development. Diving into each of these steps
in a little bit more detail. You start by curating a data set test
cases. And the idea here is to be
more comprehensive than exhaustive. You can have a set of examples that are representative of the inputs
that you expect your agent to receive. And you just really need 1 or 2 examples from each of the different types of inputs
that you might get. You don't need to have hundreds
and hundreds of examples, especially if there's
similar types of examples. And these examples can come from either
live runs of your agent or can be constructed
beforehand, manually, or even in some cases,
and generated using another model. In practice,
you often start by constructing the examples yourself,
and then add to them from live data. As you release your agent. And then wherever possible, it's always good to include expected outputs
along with your test cases. You may not always have expected outputs,
but if you can include those, it unlocks different types of evaluations
that you can run. LLM as-a-Judge, evals can be used
even if you don't have expected output. But certain code-based evals do require
an expected output to compare against. With your data set in place
and your test case is in place. Then you can start to make changes to your
agent and track each of those changes. So, some tests that you can often do are changing things
like your prompts used by your agent, changing the tool definitions
that you're passing into your router, changing the router logic itself,
changing some of the skills or the skill structure, or just swapping
in a new model that you want to test out. And the
practice of sending each of your test cases through a variation of your agent
is often referred to as an experiment. We are experimenting
with a certain version of your agent. And you can use those experiments
to record and measure results from each of those different runs. Once you have all your experiment
data collected, then you can use all of your evaluators from the previous lessons
that you've built and apply them
to the results of those experiments. So you can use your code-based evaluators,
your comparison against ground truth, or checking if generated code is runnable,
as well as the convergence evaluators that you run. And you can use your LLM
as-a-Judge evaluators, like function calling,
analysis, clarity, and entity correctness. Now, how can you apply this to your agent? As a quick reminder, your agent is set up
to use this OpenAI router with function calling and then three
different skills that you can evaluate. First, taking an example of how you would set up an experiment
around with a router. You might have an example test case that looks something like this
where you have an input, which is which stores
have the best sales performance in 2021 and then an expected output. In this case, the database lookup tool is
the tool should be chosen by the router. And then you might want to experiment
with trying different tool descriptions. So if you think back to the Json object
that you pass into your LLM router, you might want to modify the description
that's given to each tool to see if that improves
your router performance. And then once you have
those experiments run, you can evaluate the results
using either code based comparison against ground truth. Because here you have
that expected output. You have that ground truth data. Or you can use a function
calling LLM as-a-Judge. Taking another component of your agent. You have your database
lookup tool or skill, and you might have a test case
that looks something like this, where you have the same input
from the previous round. And then in this case, the expected output
is some SQL code that you see there. And so if you notice here that SQL code is an intermediate step
within your database lookup tool. First, it generates SQL and then it runs
that SQL and gets an output. So that's one thing to keep in mind here
is that again you can evaluate just the SQL generation
part of your database lookup tool. Invite you
to pause the video here for a second, and see if you can think of an experiment
and some evaluations that you could run on this database lookup tool. So one experiment you could do would be testing
different SQL generation prompts. You could also test different models
or other pieces of that as well. And then you can use your code based
comparison against ground truth evaluator, similar to what you had looked at in
previous notebooks as well, because you have that ground truth
to compare against. Next, you look at the database
analysis tool. In this case, you might have a test case
like the one you see here where you have a message from a user,
and then you actually have some retrieved data. Because the database analysis tool takes in both a user question
as well as some data to analyze. So you'd have to have both
those pieces in your test case, and then you don't actually
have an expected output in this test case. So once again,
I invite you to pause the video and see if you can think of an experiment
and then an evaluation that you can run in this case. So here you want to experiment could be testing different
LLM models. You could also test different prompts
or other kind of logic changes there too. Then you can use the analysis
clarity and entity correctness LLM as-a-Judge evaluations from previous slides and notebooks
that you've seen to evaluate the results. And as you structure
all of these together and start to run multiple iterations of your agent
with multiple different evaluators and create a whole process around this,
you can end up with, a dashboard or a heads-up display
that looks something like this. Each row has one run of your agent, and then all of your evaluators,
those different columns here, and you can measure, okay,
what are the effects of each change that I'm going to make to my agent across
not just one part of the agent, but each part of the agent holistically. And then again, as you
start to move your agent into production, you'll find that you're going to come up
with new test cases, new evaluations, and new changes
that you want to make using some of that production
monitoring data that you have. And then you can bring that back
into your testing and development process. So this whole experimentation framework
and this whole evaluation driven development framework enables you to not just create
a strong application in development, but also incorporate production learnings
that you have into your development process
and create a large flywheel here that you can use to create a better
and better agent over time. In this lesson, you learned the purpose
of evaluation-driven development, what it is and how it works,
and you learned how you can structure experiments around your evals
to scale them as you continually improve your agent. In the next notebook, you're going
to implement some of these techniques and create the visualization
that you saw just a few slides ago by adding lots of evaluations to the agent
you've been working on so far, and structuring them into an experiment.

---------------

In this final notebook here, you can take everything
that you've learned and compose it all together
to create one large experiment, to be able to test out all the different parts
of your agent at once. So you'll combine some of the evaluators
that you created in previous notebooks, along with the experiment structure here, to create an easy
way to iterate on your agent. As before, import some different libraries
that you're going to use. These should look familiar
at this point in your LLM classify by some of the experiment functions
that you might use. And then from utils some functions
including the agent that you've been working on so far. Now, you're going to run a few different
evaluators. So it's helpful to define a few eval model
at the beginning saying this is the evaluator LLM judge model
you're going to use throughout. And then you're also going
to be accessing the Phoenix client. So you can go ahead and add
in the Phoenix client. Now, you're going to use one big experiment
to actually structure this code here. And so the first step is creating the data set that will be run
through for that experiment. So to start out with
you can create a data set here. and this is a pretty big data set. Here you have the questions
that are being inputted. You also have some ground truth data
like the SQL results that you expect for those questions, as well as some ground truth data
about the SQL generated for some of those questions. Because you're combining multiple
different evaluators here, you have some multiple examples
of ground truth data that's being added. But again, you'll use the same syntax
as you used previously to create a data frame out of those examples
and then upload that data frame into Phoenix
under the name Overall Experiment Inputs. Now, one of the thing that's important
to note is because you have multiple keys this time, when you upload a data
set into Phoenix, you define which of those keys are input values, so in this case just the question. And then which of them are output values. So these are your expected outputs. So your SQL result and SQL generated keys. Run that and your
data set will be uploaded into Phoenix. You can take a look in Phoenix
within the data sets section. You may need to refresh your Phoenix
to have that appear. And then you'll now see you have this
overall experiments input data set. And if you click into any of these examples you'll see that
each one has input keys for question and then expected output keys
for SQL result and SQL generated. Now you might be asking why we just have ground truth data
for the SQL result in SQL generated. The reason for that
is that you can still use LLM as-a-Judge when you don't have expected ground
truth data to compare against. So for some of the other tools
you can use LLM as-a-Judge approach where you don't need that expected value. So on the subject of LLM as-a-Judge, now it's time to define all of your different
evaluators that you're going to use. And there's two different LLM as-a-Judge
prompts that you might want to use here. And these should look familiar
from the previous notebooks that you run. So if one is your LLM as-a-Judge
for clarity and your response. And then another here is for what
we're calling entity correctness. So in this case,
this is another type of LLM as-a-Judge that you can run on the output
of your model or your data analyzer step to check to make sure that any, any variables that are mentioned,
any SQL columns or anything like that are mapped correctly in the input,
the data throughout the run, and then the output. This is basically making sure
that your agent doesn't start referring to SKU or skew columns as store IDs, or something
along those lines. Now you can define
your evaluator functions. And so you can go one by one for these
starting with the router. So, for your router
you're doing a function calling eval. And because you're using this
in the context of an experiment, you can expect that
this is going to be sent an input. And then an output. In some
cases you also have expected output. In this case you don't have ground truth
data for function calling. So you just have an input
and then the output. This is the original input that agent got. And then this is going to be your output
from running your agent. So what you'll do is from that output grab
some of the tool calls that were made. That'll be one of the entries
in the output. And then you can construct together a data frame using the question as one key. And then any of the different function
calls as the other rows here. And the reason that you're constructing
a data frame is one so that it works nicely
with the LLM classify method. And two, because you might actually have
multiple tool calls per question here. So you might have a case where a question came in and your agent
decided to call two different tools. So this structure here
will create a data frame that's two rows with the question
and the tool call column. And then you're going to use your LLM
as-a-Judge, LLM classify method here again
using that data frame. The tool calling prompt template as before
exactly the same code that you ran. Previously to run your LLM
as-a-Judge for function calling. A couple notebooks I got there. So you can define that method, and
then here you're returning back the score. This is going to be the mean score here. Because again, you could have
two different LLM as a judge or two different functions
that were called by your agent. So you're going to have to have some way
of reporting back the overall score for this run. Now, you can start evaluating your tools. So one of those tools is going to be
your database lookup tool. And so you can grab a method to evaluate your database
lookup tool. And so in this case
it's evaluating the SQL result. And now if you think back to the data
frame, you'll remember that you have some ground
truth data. You have some expected responses
for SQL generation. So in this case you will still have
the output of your agent. But then you're also going to ask
for the expected value. So these are going to be
the expected ground truth responses that you have
that you defined in your data frame. And so first it's
checking to make sure the outputs there. And then it's grabbing
any kind of SQL result from the actual output that was run. And in this case it's going to grab
all the two responses from the output. And then this code here will actually loop
through all the different tools that were called
in that particular run of the agent. And then it's going to look for one
that matches the name lookup sales data. And the idea there is
you want to make sure that you're just looking at the tool
call responses here that are for that lookup sales data
method. You don't want to look
for your other two tools in this case. And then once you have
that as your SQL result, then you're going to grab the actual response
from that particular tool. So all of this code here is just getting
you to a point where the SQL result equals the SQL that was generated by your agent
when it ran. And now you can do one more step here,
which is to pull the numbers, specifically the numerical values
out of that SQL result. And then also out of the expected
target SQL result. So this here is your expected value
that's in your data set. And the reason to pull out
just the numbers is because in SQL you can define columns in your responses
and things like that using different names. And this just allows you
to make sure that you're not having a case where your agent use a slightly different
column name in its response. And still got
the correct numerical result. And then this will return true or false,
depending on if those numbers match. Now, one thing you'll notice here
is that in this case, you're using a code-based evaluator
to evaluate your SQL generation. If you think back to a couple notebooks
ago, you actually used an LLM as a judge to evaluate your SQL response. Either of those approaches can be used.
If you think back to the slides, LLM as-a-Judge will work. It may not be 100% accurate. So the method you're looking at here
to compare against ground truth is generally going to be a more accurate
kind of evaluation. But at the same time it relies on
you having some of that ground truth data. So it relies on
you having those expected values which you may not have in large amounts
or large scales. Okay. Now moving on to your second tool. You have your data analysis tool. And there's actually
two different evaluations you can run for your data analysis tool
that match the two prompts you saw above. One is to evaluate
the clarity of the response. And so in this case you can use
an evaluate clarity method here that takes in the output of your agent
and the original input to your agent. And then it will construct a quick data frame out of those
using the query and response columns, because that's what's expected by your
LLM as-a-Judge prompt. And then it will call the LLM classify
method. And classify the result of that particular
run of your agent. So this one is exactly the same
as the previous LLM as-a-Judge that you ran
a couple notebooks ago for clarity. It's just now put into this function
that you can call that knows to expect things like the input in the
output from different steps. Similarly, you can add another evaluator here
for entity correctness. Again, same exact kind of flow. So you'll have a method. You'll create a data frame
out of the input and the output. And then you'll call LLM classify
using your LLM as-a-Judge prompt. So now you've got evaluators
for your router your tool one. Your tool two. Finally, the last piece here is your tool
three. Which again you can use one that you've used previously here to check
if the generated code is runnable. To evaluate tool three
you can use a code is runnable evaluation here
this time you don't need the input. You don't have any expected values
to compare against. You just have the output. So what you can do is
you can take that output and you can again
pull out any tool call responses. This is similar to what you did
for the database lookup tool. So all of this code here is just getting
you to a point where generated code equals the code that was generated
when the tool name equals generate visualization. And then you can take that and make sure that you have stripped away
any extra characters that are used there. And then at the end here, you can just run
your execute on that generated code. If it executes correctly, return true. If not, return false. So this is the same evaluation
that you used a couple of notebooks ago, just with some extra manipulation to get it to look for the specific output
here. Great. And now you have your data set, as well as all the evaluators
that you want to run on your experiment. The last piece that you need is your task. And so you can define a task method here. In this case run agent task. And it's going to take that example. Grab the input question from it
and then run your agent on that. Now you can define that. And one other thing to call out here
is there's this process messages method that's called. This is also in your utils function. And it's just going to take the output
of your agent and process the messages to match
a little bit cleaner of a format to read. Now with your data set and your task
and all of your evaluators defined, you're ready
to run your giant experiment here. So you can use the run experiment method. You're passing in the data
set the run agent task. And for evaluators, you can pass in all
your different evaluator functions here. And what will happen is now each row of that data
set will be run through that agent task. And then the outputs of that will be run
through all of your different evaluators. Go ahead and run that. And that will take a second to run. So once that completes
you'll see that you run seven different tasks through your agent. And then you'll see a bunch of output here
around all of the different evaluations. So you'll see again, you're running
35 evaluations here total because it's five evaluations
for each seven case. So you'll have quite a bit of output here. That's all normal
and it's all looking good. You'll have some graphs and then you'll finally have some scores at the bottom
for some of those different runs. And now if you jump over into Phoenix
you'll see a detailed view there as well. To make sure you go
to your overall experiments input here. And then you'll see a bunch of different
columns for all those different evals. So it looks like all the code is runnable. The evaluation analysis is clear. The entities are correct. It looks like some issues with
SQL generation and then function calling. So you could jump into this experiment
and see it in more detail. So you could see all the tool calls and all the format of response that
you have here for some of these pieces. So you could click into those and see more
if you wanted to. And now the question becomes
what do you do with this information? How do you improve your agent? And so you've got really two options here
for how to improve your agent. One is in code. You can go in and start
making some changes to your agent and running different versions
of your agent using the experiment. So SQL generation was a little bit
tough there. So why not start with changing up
the SQL generation prompt. So this is your base SQL generation prompt
that's being used by our agent. We can add something here like think
before you respond. You might make bigger changes to it. This is just a pretty basic change. But one you could try out. And then we have a helper function here to just update that SQL generation prompt. In practice,
you probably go into your agent code. You would actually make the change
in the agent code. This way, we'll do that
same kind of thing here. And then you can rerun your experiment and you'll because you've updated
your agent, that task is now a new task. So you might want actually change
this experiment name to say V2 and evaluating the overall experiment with changes to SQL prompt. And then you can run that experiment again
and you'll get a comparable line. And so now if you scroll
through those results, you'll see similar kinds of outputs before. And then you can continue going down,
see all the results that were run. And you'll have your same scores
from before. You can jump over into Phoenix
to see those in detail. And now you may have to refresh your page. You should see a new entry for your
in second experiment for your V2
and then changes to the SQL prompt. And so it looks like
if you look at the scores there, looks like SQL didn't
really have the change that you wanted. So you might have to have
some more changes from there. And one of the things it looks like
some of the responses that came back are less clear or correct
this time. That could be due to the change you made. It could be due to just the fact
that you're running with seven test cases here,
which is a sort of small amount. So it's always good to do
multiple runs of each experiment to get some more statistically significant kind of data back as you're
testing your agent's performance. So you can continue to make changes
inside of your agent, either to one part of your agent or multiple parts of your agent
to try to boost these scores over time. And you could continue
to make those code changes until you get to the results
that you want with your agent. There is one other approach that you can
use as well too, which is in Phoenix, there's a tool called playground
where you can jump into the playground button here, and that will bring your data
set into a playground environment where you can make whatever queries
you want over your data set. And so you could actually take that
same SQL generation prompt and run it directly here on your data
if you want it to. So you could copy over your prompt
from your notebook and paste that in here. And then in this case,
you might want to have it look for specific variables
from your inputs and outputs. So in this case if I use curly braces
you can add these curly braces. And then you'll actually pull
in the question from that particular row. So you could use this. And if you then ran this you would then have results for each row
that you could look at. You can use the compare button that will
add a second iteration of the prompt. In this case you're using GPT-4o-mini. So you can switch that to 4o-mini. And then similarly over here
have consistency there. And so you could actually make your changes
to your prompts directly in the UI here. And then this will give you a side by side
comparison of the responses. So you could always make the changes
to your agent in your code itself and then run them as a new experiment. Alternatively, you can use this
prompt playground tool to run changes directly in the UI and have a little bit faster
of an iteration cycle as you change them. No matter if you
use either of these different techniques, you can make changes to your agent and use
your new experiment that you've created to have a really easy apples
to apples way to compare what are the actual facts
of each of those changes, so that you're not just moving
things around without understanding what the effect of each change situation
might be. So you now have a structure
that you can use to iterate on your agent and compare
all of those different iterations to each other in a structured way.

---------------


For this lesson, you'll dive into
how you can improve your LLM evaluators and LLM as-a-Judge
evaluations within your agent. The idea here is that as you use
LLM as-a-Judge, you may want to improve the evaluators themselves, as
those are never going to be 100% accurate as you've learned in previous lessons. So this serves as a bonus lesson here
where you can learn how to improve those evaluators as you go through the process
of improving your agent itself. So if you think back to the previous slides lesson,
you might have noticed a case where you used both a code-based comparison
against ground truth and an LLM as-a-Judge function calling evaluation
to measure the performance of your router. And you might be thinking, why would you need to use two different
evaluators to evaluate the same thing? Especially because the code based
evaluator is going to be 100% accurate. So why would you add in the LLM as-a-Judge evaluator? Well, the answer there is
that you can use this technique to actually measure how effective
and how closely your LLM as-a-Judge evaluator aligns to your 100% accurate
code base comparison against ground truth. Because as you remember, LLM as-a-Judge is not 100%
accurate method. However, it can scale a lot further beyond
the comparison against ground truth. You could apply an LLM as-a-Judge eval
to all of the runs of your application if you want it to,
but it pays to know how accurate that LLM as-a-Judge is compared
to your 100% accurate evaluation method. So you can use experiments to actually
improve your LLM as-a-Judge itself. The experiments that you just learned about in a previous
lesson, can still be applied this time to your LLM as-a-Judge,
as opposed to to your agent itself. It's a little bit meta, but you can use the same techniques
to judge your judge in this case. So you could set up an experiment on the function
calling LLM judge. And you might have an example test case
that looks something like we see here where you have an input. That's which stores
have the best sales performance in 2021 and then an output in this case
the database lookup. And the important thing is that everything in red
there is the input to your LLM judge, because you're having your LLM judge
judge the performance of that agent. So everything in red there is the input. And then you'd have your expected output
in this case is correct. So that expected output
is your ground truth data. And then you could set up the experiment
to test different versions of your LLM as-a-Judge prompt. You might change
some of the wording there. Or you could add things
like few shot examples. So you could add in some examples
of previous judgments made that were correct to help align
the judge into that prompt as well. And then you can evaluate
the performance of your LLM as-a-Judge using the code-based comparison
against ground truth. In this case, comparing to that correct
or incorrect expected output label. And you can use the same approach
for another LLM as-a-Judge that you're using, in this case
the one you're using to evaluate analysis clarity. And so in that example, you might have an input
that looks something like you see here: in 2021 the stores that perform the best,
yada yada. And then the expected output of the
analysis is clearer because of x, y and z. And this time
you might judge different types of models that are being used to create that
judge label. You could also test different LLM as a judge prompts similar
to what you did in the previous example. Now the question here though becomes
how do you evaluate that output? Because while you have expected output,
it's no longer a clean just correct or incorrect label. It's this expected output of the analysis
is clear because of something. And so what if your LLM as-a-Judge
comes back with the analysis is easy to understand
because of x, y, and z. That should be correct. But you can't just compare
those strings exactly. That's where you can use something
like semantic similarity to compare the meaning
of each of those strings in a numeric way, as opposed to doing a direct comparison
between output, unexpected output. So in this lesson, you've learned
how you can measure and improve your LLM as-a-Judge evals using structured experiments. And your next and final lesson,
you'll learn about moving your agents into production
and monitoring those agents in production, as well as wrap up all the lessons
that you've learned here in the course.

---------------


After
you develop your agent and improve it until it's production ready,
you need to apply the same techniques of tracing evals and experimentation
to your agent in production. During production, things like code changes or model updates can degrade the performance of your agent. You can use your evals and run experiments
to continuously monitor and improve your agent. Let's get to it. As a quick recap, you've covered
four main steps that take you from an initial agent prototype all the way
through to a production-ready system. Choosing the right architecture, you need to decide on an agent framework
that matches your use case. Deciding which eval to use. Figuring out which metrics really matter
for your system, such as accuracy, latency,
and convergence. Building your evaluation structure,
which includes the prompts, tools, and data
you'll use to measure performance. Iterating with your data,
you keep refining your agent by analyzing results, adjusting prompts
or logic, and testing again. This cycle repeats as you move
from development to production and back again, so you can catch issues early
and refine based on real-world feedback. When you actually reach production,
you may find that the simple agent designs you use
initially need to scale up and complexity. You might end up
adopting a multi-agent system where agents can call
other specialized agents, or a multimodal system where your agent
handles different types of data, like text or images or audio,
or a continuous improving system where your agent learns
from user interactions in real time, either through manual updates
or automated processes. So, what's different about production? Well, you can often discover new failure
modes. Users might ask you questions
that you never saw in development or they reference something
that your system doesn't know about yet, like a brand new product. If your agent now calls out
to additional APIs or other agents, there are more chances for errors
or unexpected outputs. You might also try A/B testing
or different model strategies that introduce surprising regressions that you didn't anticipate
in a controlled environment. The encouraging part here is that a lot of the tools
you use in development, like instrumentation and feedback
loops, are just as valuable in production. You'll continue to collect metrics
that you define for evaluation, and you'll rely on continuous integration
and continuous delivery flows and ongoing experiments
to keep tabs on your agent's performance. In other words, you can use those
same tracing and annotation methods you had in development, only
now, they're enriched by real user data. You'll gather
feedback from actual interactions, label any problematic outputs, and identify
where your agent might be struggling. Collecting user feedback
for evals is crucial in production. You can take real usage data, for example,
every user query or interaction, and attach human-labeled annotations
to highlight issues or successes. If your eval metrics disagree
with what real users say, that's a sign you might need to recheck your system
or your eval. Maybe you're measuring the wrong metric. Or maybe there's a deeper logic
flaw and the agent's flow. You also want to keep track of metrics over time to understand efficiency
or execution dependencies. For example,
if you're using convergence evals, you can see how many steps it takes
for the agents to reach a correct answer. If that number grows after you tweak
your prompts, it can indicate a regression since you're likely calling external
LLM APIs, you should track those calls, too. They can have a significant impact on latency and costs,
which can impact your end user experience. Your choice of model, perhaps a large reasoning model versus
a smaller, faster one may vary based on the complexity of tasks
in production. Keeping an eye on those decisions
and their downstream effects is part of robust production
monitoring. When you gather human feedback
and run your evals on real traffic, you'll gain a clearer picture
of how changes like swapping out a model or updating a prompt, impact
the overall system. You can rerun the same experiments
you used in development, but now on new data or new failure modes
that you've discovered in production. This is why you'll want to maintain
consistent data sets and keep augmenting them
with production samples. One effective approach is curating golden
data sets that capture your most critical use cases
and known failure modes. Each time you push a change,
like adjusting a prompt or logic, you can recheck these data sets to be sure
you haven't broken something you've already solved. Experiments can act like gates
for shipping changes, so you can decide whether to roll forward
or roll back a change. Let's imagine you have a self-improving
agent that collects feedback automatically whenever a user interacts
with your system. You can add those user examples,
both successful and failed interactions,
to a continuously updated data set. Then you run
CI/CD experiments on this data set, checking
if new versions of the agent do better or worse on the very latest
real-world scenarios. When you refine the agent logic
or tweak prompts, you can also incorporate few shot examples
from the newly collected data. This lets your system learn from mistakes
and gradually converge towards better performance
on exactly the tasks that matter most in a self-improving and automated manner. What you're doing here
is essentially applying evaluation driven development in production. You're watching for new failure modes, monitoring how well your agent
handles them, and automating feeding production
feedback back into your evals. In this lesson, you've learned what to watch out for new queries,
complicated architectures, and unexpected user behavior,
and how to keep your agent on track by continuously measuring
and refining its performance. You've also seen how the tools and data
you used in development carry seamlessly over to production,
and how you might scale those approaches for even more robust agent systems, by using CI/CD pipelines
with consistent experiments, you can ensure that every update
to your agent maintains or improves on the quality
you've already achieved. Now, you have a solid foundation
for monitoring your agents in production and continuously refining their
performance based on real-world data.

-------------


Throughout this course, you learned how to trace, evaluate,
and improve a code-based agent. You can now take what you've learned
and apply it to any other agent framework. Please check out the resource section
that we've included after this lesson. And I can't wait to see what
you'll build on your own.

-----------


