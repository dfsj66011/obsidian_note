
https://www.deeplearning.ai/short-courses/build-apps-with-windsurfs-ai-coding-agents/


在本课程中，你将使用AI代理来分析并解决一些JavaScript代码中的问题，在这个过程中，你会看到更多关于代理如何工作的细节。让我们开始吧。在这个演示中，我们将使用诸如jest等测试框架，在一个JavaScript代码库上完成任务。不用担心你不熟悉JavaScript或这些框架。

这次演示的目的是展示你实际上可以和一个代理一起迭代你可能不熟悉的代码库。所以，正如你所看到的，这里有一些代码，我将请Cascade帮我调试并修复代码库中的一些问题。

 首先，我必须询问Cascade是修复还是运行此仓库中的所有jest测试。接下来我将指出Cascade正在做的几件不同的事情。随着深入我们会补充更多细节。第一件事是Cascade正在遍历整个仓库，试图理解我们为何需要运行这些测试。它深入其中，分析现有知识。也就是说它对现有代码库具有认知能力。它拥有诸如建议终端命令等功能工具。

这样我就可以出去运行测试了。他们能够分析堆栈跟踪，然后说，嘿，有一个失败的测试。你想调查一下吗？所以我当然可以和它对话，就像你和任何聊天体验一样，只不过在这种情况下，它是一个已经独立采取了多个步骤的代理。

所以我只说“好的”。现在Cascade将采取多个步骤和使用多种工具，就像人类一样。它会再次利用上下文感知能力来分析测试代码和影响测试的代码。它会推理问题可能出在哪里。然后使用工具实际修改文件。在这个例子中，它意识到问题实际上出在测试用例本身。所以当它再次使用编辑工具时，还会建议我运行测试的命令。

现在所有的测试都通过了。所以在大约一分钟内，我们并不十分了解代码库的情况下，这个代理已经帮助我们解决了测试中的错误。我打算接受这些更改，但我想再做一件事来展示Windsurf中Cascade的另一个功能。也许我会转到定义文件，然后说，实际上，我想让这个函数名更具描述性。

我就说ForCells吧。现在我可以回到Cascade，直接说继续更新所有调用点。注意我并没有具体说明我做了哪些修改，但由于Cascade会关注我在编辑器其他部分的操作，它能够察觉我刚才所做的更改并相应采取行动。Cascade已经启动，会遍历文件、更新所有调用点，现在将对多个文件进行多处编辑——这是许多仅支持单次LLM调用的AI编程助手无法实现的功能。

完成所有编辑后，Cascade会建议我们重新运行测试，以验证测试是否仍然通过，并且所有调用点都已相应更新。虽然这只是一个非常简短的演示，但它突出了使Cascade极其强大的许多核心要素。这包括对现有代码库的理解。

他们使用一套完全不同的工具来采取行动、调查和验证其工作，同时也能清晰地理解我作为开发者在与文本编辑器交互时的一些意图。接下来，我们将以这个例子为基础，剖析并理解这类自主系统运作的部分心智模型。

-----------

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


Again, I can use the open diff tool,
the Cascade bar, in order to observe and investigate
all the changes that have been made. Let's go and run the app. Okay. Seems like the app is live. Let's go check it out. So here we are at localhost 5000
where it said it was deployed. Let me type in my large language models
category. Generate the word cloud and boom. Seems like it's fast enough. Seems like it's in the cache. Looks great to me. Let's go back to Windsurf
and I'll show a few more things. The first thing I want to show
is that, as you can see, currently Cascade is running a terminal process. You'll notice here in this bar that appears above the input prompt area. You can see all the changes
that you need to review over, and all the files
that you need to look into. And you can also see all the terminal
processes that are currently running. You can also end and cancel
any of the processes from here or directly
from where the command was actually run. So let's go in. When making reviews. Again I can go file to file. I can accept changes at block level. I can accept changes at file level. I can accept all the changes
directly from this bar. So a lot of different ways in order
to accept or reject changes from cascade. So, so far so good. Let's do a little bit more looking around. I have the HTML file here which looks great. There's obviously
a lot of JavaScript here as asked. Maybe I'm just not expert on JavaScript. And I kind of want to understand
a little bit of what's going on. There's some nice other handy features
hidden around the Windsurf editor to make these kinds of operations
really easy. For example, if I go to the function,
this is a function generate word cloud in JavaScript,
not an expert on this. You'll notice that a number of helpful
code lenses appear at the top above the text editor. So I could just click explain. And what this will do is Cascade will pull
in all that relevant information into Cascade. Perform analysis and explain to me
what's going on. This is really helpful,
especially as I use Cascade and these agents
in unfamiliar languages or frameworks. I can also use it to explain
exactly what's going on, and our ability to be able to detect this and pull in just the relevant information
is due to all that complex parsing logic that we built on top of code bases
that we mentioned in earlier lesson. So this is always a helpful utility
in order to understand code that exists. The next thing I want to add
though to my app is different colors. I want to make it look very pretty. And to do this, I'm actually going to demonstrate
a number of the functionalities that have nothing to do with Cascade
or the agent. Again, this is an AI native IDE,
so has much more than just the agentic experience. To do this, I'm
going to go to my File Explorer. I'm going to create a new file
called Colorpalette.py just to define the color palette. I'm going to close Cascade for now. The first functionality that I'm going
to show is what's called command. This is the ability to give instructions
to the AI directly in natural language. In the text editor,
the shortcut for that is command I. As you can see,
the command entry comes up. In this case,
I'll just give command an instruction to create a color palette class
that contains six hex colors. Punch of common color palettes, etc. Let's see how it works. As you can see, it's going to rapidly
generate a lot of code for me, which is great even faster than I can
probably do it while prompting Cascade, which is why you should always use
agentic experience for everything. But it's nice because you can also
use further functionalities. For example, you can already see the autocomplete is trying to suggest
additional boilerplate code. So looking at this maybe
I want like your class RGB color palette. I can just autocomplete and multiple
lines of code can be generated rapidly. I can continue my work here. I can say define function to just get all color palettes,
find a method just to return all of them. That's great, but you'll notice here let me actually word-wrap
to make it visible. But it's not just the ability
to command or autocomplete. There's a lot of the passive experience
that exists with editing in the text area. For example, I might just go here and I might just rename this default
color palette. What you'll notice is that our
AI is already suggesting the next edits that appear
far away from my cursor position. So, this is the benefit of having
a really strong pasive experience inside the text editor itself. You can rapidly generate
a lot of boilerplate code without really need to rely on the agent
as a crutch for every edit that you want to make. So this is helpful, but now actually need to incorporate this color palette class
that I created into my application. So for this, I'm
going to switch back to Cascade. What I'm going to ask Cascade to do
is I'm going to integrate this get all color palettes
method into my existing app. To this I'm going to use a functionality
called App Mentions, something we haven't mentioned so far,
but just to show what it really does, is this a way for you as the developer
to guide the AI in a very light lift manner in places that you already know
what the AI needs to look at. Again, because it is agentic, Cascade
does have the capability to go and figure this out by themselves. But just like if you're working
with a peer programmer, if you can just nicely nudge them in the right direction,
you're just going to save them some time. So think about it that mentions
in the similar kind of manner. You're able to @
mention files, directories. You can mention @ individual methods
and functions which we'll use in a moment. You could even
@ mentioned the web were a bunch of third party public docs for third party
APIs that are common. So let's integrate this color palette
into our Word Cloud app. To do that, I'll just use the
that mentioned, as we just discussed, to tag in the
get all color palettes function. Second to the word cloud app. Again best practices.
Let's be a little descriptive. What we want to see
made the color palettes selectable in a dropdown and display the colors being used. Let's see what happens. As you can see, by using an app mentioned, the agents able to directly go to where it needs to look, and then it will use this agentic search and discovery capabilities
to look at other relevant pieces. Start making some modifications. You can use the open diff as I go along. See some of the changes being made. Let's see how this looks. Looks pretty good to me. So let's try to generate category name. Of course that's on me. Oh, I accidentally misspelled it. That's fine. Cool. So seems like we got word cloud
using this material color palette. Let's start changing the color palette. Change the dark color palette. Seems
to generate the word cloud differently. Awesome. Seems to be working. Kind of like how I'd expect it to work. One thing I noticed that
when selecting a different color palette, it would automatically change
the word cloud without me clicking the Generate Word cloud button. Maybe I like it. Maybe I actually want to change this
a little bit. Again, this is the iteration part of
working with an AI agent or any AI tools. I must go back to Windsurf
and just say, make sure that I need to click the generate button to create a new word cloud. Even if I change the selection of the color palette. Or just some kind of instructed action
and we'll see what it does. Seems like it wasn't a huge change. That sorts all changes. Let's go back to the application. Refresh it. Put in. Category. Okay. Now if I select
and you call it word cloud change. But the word cloud didn't change. So let's see. Give me the loading screen. Awesome. So again these are minor details but way to iterate on your applications
using Cascade. We did quite a lot in this lesson. We not only built the entire full
stack up using Cascade, but we also used command and autocomplete to rapidly
build a bunch of boilerplate code and then incorporate it
into our overall application. As you can see, using Windsurf
is more than just iterating with an agent. It is a full experience
with AI in order to maximally improve your velocity in developing
the features that you want to. This is fun! We're going to wrap it up
with one more lesson right after this. So let's continue.

------------

In this lesson,
you will add some further functionalities by leveraging the multi-modality
capabilities of the AI agent, and you will also learn about
a number of features and best practices so that you can continue to build
and customize your Wikipedia analysis up. Let's continue. You made it to the final lesson. In this last lesson, we're just gonna add one more functionality
using some multi-modality inputs. So, let's switch over to Windsurf. Great. So the app looks good. One thing you'll notice is down here
there's the ability to insert images into Cascade. So we're going to utilize that. So what I'm going to do
is I'm going to go back to my application. And the application looks good. What do I do? Now that I'm here my application
let me zoom out a little bit. I'm actually going to take a screenshot. Let me open the screenshot. Great. I'm actually
just going to add some stuff to this. So let me add a rectangle over here on the left. Some text. Let's say I also want to you know
the visual representation is great. But I actually also want to see
the raw frequencies while I'm at it. So I'm gonna put raw frequencies. Let's add another text box. Word one. Word two. Let me actually just put in some text
here. This is kind of just roughly
what I do, about 30s kind of sketch. Now, instead of having to explain all of the details of the layout,
I'm just going to save that image. Let's go back to Cascade. So I'm going to use this image input now. So I just added the image to Cascade. And now I'll just say: add raw frequencies to the app as shown in the image. Something as simple as that. Cascade can use the multimodality aspects of these reasoning models. Already, it noticed
that these raw frequencies had to go to the box
on the left of the word cloud. I never mentioned that to Cascade. All right. Changes seem to be live. So let's go back. Let's refresh. Large language models. Zoom back in a little. Okay. The raw frequencies
are there on the box on the left. Even using the red text on the red
outlines like I showed in the image. So this is a simple usage
of using multi-modality in order
to increase your development velocity. So this is the last feature that we were going to actually add
as part of this demonstration. But I'm excited to see
all that you can build on top of this. And just help you along the way, I'll point out a couple of other things
in Cascade and a Windsurf that I wasn't able to
as part of this demo. The first is there's a lot of settings
on the Windsurf settings panel, many cutaways
in order to customize your experience. One of the things that we didn't show
was the idea of memories. The idea behind memories is to, similarly
to rules, be pieces of information that Cascade
can continuously go back and reference. So allows Cascade to essentially
build state of how you do work and what's important to you as time goes on. These memories can either
be explicitly mentioned, those are the rules,
or they can even be auto-generated. So that's what
these Cascade generated memories does. Either Cascade will automatically generate
these memories as you're doing work, and it'll give you a notification. Or you can just tell Cascade,
remember this for me and it will insert memories
into his memory bank. Similar to rules, you can go back and edit
those at any point in time. But it's a way for Cascade
to automatically learn things without you having to explicitly mention
everything in terms of a rule. There's a lot of pieces
that control your passive experience and other kind of nice little UX
pieces over here as well. There are other functionalities
in the settings panel, so if I search up Windsurf in the settings
panel, you'll see there's a lot of other different kinds
of editable fields for you. One of my favorite is actually the Cascade
commands all Out and deny lists. As you can see, every single command
that Cascade suggested I had to accept. But sometimes there are certain commands
that I'm totally fine for auto-running or the certain commands
I never wanted to auto-run. So you can actually
whitelist and blacklist commands in more recent releases. There's even turbo mode, which will just
automatically run everything for you. If you just want to go full vibe
coding with your agent. That's all for this lesson and course, as we saw,
there are a lot of functionalities and features that make working with Windsurf
and Cascade a lot of fun. And hopefully you learned a thing or two
about how to best improve your development workflows
with these AI collaborative agents. Of course, this is just a moment in time. And Cascade and Windsurf, we're only going
to continue to get better. The AI is going to get smarter. The functionalities and polish and guidance and observability
will only continue to expand. So keep up to date with all the changes
on our docs, or using our change logs to see what is the latest
and greatest within a Windsurf and Cascade so you can maximize your experience
as a developer with AI.

----------

Congratulations on completing this course. In this course,
you'll learn about the history of AI code assistance,
how to separate some hype from reality, how to think about these AI code
assistants, and how these AI agents work. We deep dive into search and discovery
as a specific, important paradigm, and along the way you built a number of different apps using windsurf
and the collaborative agent. I look forward to seeing
what you build with windsurf.

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

