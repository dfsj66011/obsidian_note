
https://www.deeplearning.ai/short-courses/getting-structured-llm-output

本课程由 DotTxt 合作推出，基本思路是，你可以告诉 LLM 你希望数据以特定格式呈现。讲师：威尔·库尔特（Will Kurt）和卡梅伦·菲佛（Cameron Pfiffer）。威尔是 DotTxt 的创始工程师，卡梅伦则是开发者关系工程师。

--------------

### 1、概述结构化输出

使用 Re-prompting 库，如 Instructor 和 LangChain 等。

Re-promoting 的工作方式是，我们首先向 LLM 进行一次常规传递，提供一个提示，模型就会产生一个输出。此外，我们还会为模型提供一个验证器。这只是一个关于我们期望 JSON 看起来是什么样子的描述。如果模型的输出符合我们的验证器，数据就会直接发送给我们。

Re-promoting 库会自动将解析失败的原因附加到提示中，并再次尝试。这种方法有很多优点。一是现在你可以使用任何 LLM API。这在开发可复用的优秀软件方面是一个巨大的进步。这使得你编写的代码结构可以跨 API 使用。通常，更换提供商只需要更改密钥和提供商名称。与大多数专有提供商相比，我们在结构上也获得了更大的灵活性。所以这仍然只是JSON格式，但我们可以对字段使用正则表达式来强制约束条件，比如特定的日期格式，或者确保用户的电子邮件地址是有效的。

这种方法也有一些缺点。最大的缺点是重试可能会非常昂贵，无论是在金钱还是时间方面。对于许多 LLM 应用程序的开发者来说，时间是一个更大的考虑因素，因为你让用户等待结果。由此而来，甚至没有成功的保证。如果在一定的重试次数后，库没有成功，它就会简单地失败。再次强调，虽然我们对结构有更多的控制，但我们仍然只有 JSON 作为可行的输出。

=========

结构生成，也称为约束解码，是一种直接与模型合作以获得我们想要的输出的方法。我们使用“结构化生成”这一术语，是因为我们实际上是在控制 token 的生成过程，以最终获得结构化的输出结果。目前有许多支持这一功能的库，包括 .txt的 Outlines、SGLang、微软的 Guidance 以及 XGrammar 等。

假设我们要将结构定义为仅允许字符按顺序排列的字符串，符合此结构的有效字符串包括 ABC、AABC、AC 和 BBB。所有这些例子中，每个字母都按顺序排列。无效的例子则有 BAC 和 CCCA。这两个例子中的字符顺序混乱，因此不符合我们结构的规则。第一个字符的 logits 分布情况——实际上此时没有任何约束，因为任何字符都可以作为有效的首字符。假设我们采样得到字母 b。当 b 被追加到提示词后，系统会重新运行流程并生成新的 logits。但新的 logits 可能不符合我们的约束条件，因为其中有部分概率分配给了字母 A。根据结构规则，A 在此处是无效字符。实际上，它会修改那些逻辑值，移除无效的部分，并对允许的部分重新加权，继续这个过程，直到最终生成一个保证符合我们结构规则的字符串。

它可以与任何现有的开放 LLM 一起使用。因此，如果你使用 HuggingFace 的模型（包括视觉模型等任何模型），它们都能与结构化生成协同工作。由于我们直接与模型交互，处理速度极快。推理过程中的成本几乎为零。事实上，有研究表明，通过利用标记中固有的可跳过结构，它甚至能缩短推理时间。它能提供更高质量的结果。

在资源受限的环境中，这种方法同样表现优异。由于它极为轻量且高效，即使在搭载微型大语言模型的极小设备上，你依然可以采用结构化生成技术来确保模型输出质量。我们能够生成极其丰富的结构类型——除了 JSON 格式外，还包括正则表达式，甚至是语法完全正确的代码。

不过结构化生成也存在一个局限：由于我们直接操作 logits，这意味着你必须拥有对这些 logits 的控制权。所以，这意味着你要么使用开源模型，要么自己托管专有模型并控制输出概率。

=============

#### Use Structured Outputs

OpenAI 接口的 [Structured Outputs](https://platform.openai.com/docs/guides/structured-outputs?api-mode=responses)

--------

#### Retry-based Structured Output

Instructor 的使用，提示输入到语言模型中，得到某种形式的输出。然后检查它是否是有效的语法。如果是，我们就将其返回给用户。如果不是，我们将输出和错误反馈给语言模型，并尝试再次获取输出，直到得到有效的输出为止。

**优缺点**：Instructor 使用起来很简单。它支持广泛的提供商，如 Anthropic、OpenAI 等。它在不同提供商之间提供一致的API，因此可以避免供应商锁定，并且支持 pydantic的所有功能。缺点方面，Instructor 不能完全强制执行结构，因此会出现解析失败的情况。重试可能会意外增加金钱和时间成本。每次解析失败时，都必须将整个提示重新发送回模型。Instructor 在重新提示和/或重试方面不太透明。你需要稍微深入研究内部机制才能理解发生了什么。

------

#### Structured Generation with Outlines

基于 logits 的结构化生成是一种更高效灵活的输出结构化方法，也称为约束解码或逻辑值调控法。实际上，我们是通过拦截模型中的逻辑值并对其进行修改，从而影响模型输出的概率分布。

----

#### Beyond JSON!

Outline 在底层使用了正则表达式来模拟我们想要使用的结构。这使我们能够定义比单纯 JSON 更广泛的结构范围。正则表达式与有限状态机之间还存在一种有趣的关系。实际上，我们利用这一点，在遍历模型时采用了一种轻松高效处理结构的方法。


