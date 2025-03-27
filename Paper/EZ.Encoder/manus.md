
<img src="https://arxiv.org/html/2309.02427v3/x1.png" width="500">
AI Agent：本质还是 LLM，但会跟环境有互动，从而处理非常复杂的任务，并不像 ChatGPT 那样，进行几轮对话就可以解决的。 

Agent 需要有三个重要的能力：

1. reasoning / planning 的能力
2. 使用 tools 的能力
3. 需要有 memory 

[REACT](https://arxiv.org/pdf/2210.03629) Google Brain，2023 的论文，之前的工作，大家主要关注的是 LLM 的 reasoning 能力，比如 COT，而忽略了 LLM 可以和环境互动，从环境产生一些信息，可以帮助 LLM 更好的完成任务。RE 就是 reasoning，ACT 也就是和环境的互动。**（github 有 [ipynb](https://github.com/ysymyth/ReAct/blob/master/hotpotqa.ipynb)）**

![[Pasted image 20250327163230.png]]


给定一个比较复杂的问题，需要额外的信息，并且还需要一定的推理
* 如果用 LLM 直接回答，答案是错的 
* 即使用了 COT，只强调 reasoning，模型虽然能有一些推理，答案仍然是错的
* Act-Only，赋予模型上网搜索的功能，这个答案也是错的
* ReAct，首先模型会产生一个 thought 或 planning，去搜、推理等，答案对了

它跟之前 LLM 不同的点在于，它鼓励 LLM 除了有自己的 thought 以外，还基于 thought 去和环境互动，利用搜索引擎去搜集更多的信息，随后又产生新的 reasoning 以及下面要采取的行动。

*如何教会模型采用这种 react 的方式*  few-shot learning；ReAct 的实现非常的简单，已经具备现在 AI Agent 的雏形， 但*缺少 memory 的功能* 

[Reflexion](https://arxiv.org/abs/2303.11366)，随后 2023.10 提出来一种方法，让模型自己做反思，把反思作为模型的长期记忆，从而帮助模型去进行思考规划，作者让模型对前面的整个解题过程进行了一个总结，然后把这个总结作为一种形式的 memory 放到了第二次的尝试当中 **（github 有 [ipynb](https://github.com/langchain-ai/langgraph/blob/main/docs/docs/tutorials/reflexion/reflexion.ipynb)）**

![[Pasted image 20250327165335.png|600]]

框架：Agent 和环境互动，互动过程中，每一步 Agent 要用到两种 memory，
* short-term memory 是模型之前的思考过程，即 LLM 的 context
* long-term memory 是模型对前序方法产生的总结，即 reflective text

这个方法在 ReAct 的基础上，引入了一个 self-reflection module，本质也是 LLM，让 LLM 做一些阶段性的小节，类似于错题本，将这些经验教训类化成一个长期的记忆，

前面的论文主要是通过提示工程的方法，让模型学会使用工具，可否 *直接训练模型使用工具* 呢？

[Toolformer](https://arxiv.org/abs/2302.04761) ，meta，2023.02，用 sft 去训练模型使用工具，*难点* 是如何产生 sft 数据，*思路* 是self-supervised 的方法，通过 bootstrapping 让模型自己产生很多 API call 的数据，然后经过 filter 之后，反过来去 finetune 模型。

[GAIA](https://arxiv.org/abs/2311.12983)，来自于 meta、huggingface、autogpt，提出了一个 benchmark，测试 AI assistance 能力，展示开发一个能实际解决问题的 AI agent 有多难


Manus 的官网没有放出任何的技术细节，但是根据网上的信息，Manus主要是使用的 claude 3.5 以及 Qwen 模型，并没有训练一个自己的 AI agent，主要还是在工程方面，因为根据复杂的任务，如何使用 LLM 去调用相应的工具，制定计划分析问题，这是一个很复杂的流程，所以网上有人评价 Manus 就是一个缝合怪，缝合的就是对各种不同工具的调用。

最近很火的一个概念叫 MCP，是 Antropic 2024.11 提出来的，Model Context Protocol，提出 MCP 的主要原因是，目前不同厂商的 LLM 对于调用工具的接口定义是不一样的，调用不同的工具 API 的定义也可能是不一样的，MCP 把各种各样不同的 application 通过 MCP 和 LLM 连接起来

 [Glama](https://glama.ai/mcp/servers) 平台里面就有很多开发者创造的 Open Source MCP Server，例如 Arxiv MCP Server，允许 AI assistant 能搜索并且访问 arxiv 的 research papers，不需要自己再单独写这样的 API，只需调用统一的 MCP Server 即可。[Smithery](https://smithery.ai/)网站上面也有很多 MCP Server
    