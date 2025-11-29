
核心组件：

* Agents
* Models
* Messages
* Tools
* Short-term memory
* Streaming
* Structured output



***

## Advanced topics



### Log probabilities

Certain models can be configured to return token-level log probabilities representing the likelihood of a given token by setting the `logprobs` parameter when initializing the model:

```python  theme={null}
model = init_chat_model(
    model="gpt-4o",
    model_provider="openai"
).bind(logprobs=True)

response = model.invoke("Why do parrots talk?")
print(response.response_metadata["logprobs"])
```

### Token usage

A number of model providers return token usage information as part of the invocation response. When available, this information will be included on the [`AIMessage`](https://reference.langchain.com/python/langchain/messages/#langchain.messages.AIMessage) objects produced by the corresponding model. For more details, see the [messages](/oss/python/langchain/messages) guide.

<Note>
  Some provider APIs, notably OpenAI and Azure OpenAI chat completions, require users opt-in to receiving token usage data in streaming contexts. See the [streaming usage metadata](/oss/python/integrations/chat/openai#streaming-usage-metadata) section of the integration guide for details.
</Note>

You can track aggregate token counts across models in an application using either a callback or context manager, as shown below:

<Tabs>
  <Tab title="Callback handler">
    ```python  theme={null}
    from langchain.chat_models import init_chat_model
    from langchain_core.callbacks import UsageMetadataCallbackHandler

    model_1 = init_chat_model(model="gpt-4o-mini")
    model_2 = init_chat_model(model="claude-haiku-4-5-20251001")

    callback = UsageMetadataCallbackHandler()
    result_1 = model_1.invoke("Hello", config={"callbacks": [callback]})
    result_2 = model_2.invoke("Hello", config={"callbacks": [callback]})
    callback.usage_metadata
    ```

    ```python  theme={null}
    {
        'gpt-4o-mini-2024-07-18': {
            'input_tokens': 8,
            'output_tokens': 10,
            'total_tokens': 18,
            'input_token_details': {'audio': 0, 'cache_read': 0},
            'output_token_details': {'audio': 0, 'reasoning': 0}
        },
        'claude-haiku-4-5-20251001': {
            'input_tokens': 8,
            'output_tokens': 21,
            'total_tokens': 29,
            'input_token_details': {'cache_read': 0, 'cache_creation': 0}
        }
    }
    ```
  </Tab>

  <Tab title="Context manager">
    ```python  theme={null}
    from langchain.chat_models import init_chat_model
    from langchain_core.callbacks import get_usage_metadata_callback

    model_1 = init_chat_model(model="gpt-4o-mini")
    model_2 = init_chat_model(model="claude-haiku-4-5-20251001")

    with get_usage_metadata_callback() as cb:
        model_1.invoke("Hello")
        model_2.invoke("Hello")
        print(cb.usage_metadata)
    ```

    ```python  theme={null}
    {
        'gpt-4o-mini-2024-07-18': {
            'input_tokens': 8,
            'output_tokens': 10,
            'total_tokens': 18,
            'input_token_details': {'audio': 0, 'cache_read': 0},
            'output_token_details': {'audio': 0, 'reasoning': 0}
        },
        'claude-haiku-4-5-20251001': {
            'input_tokens': 8,
            'output_tokens': 21,
            'total_tokens': 29,
            'input_token_details': {'cache_read': 0, 'cache_creation': 0}
        }
    }
    ```
  </Tab>
</Tabs>

### Invocation config

When invoking a model, you can pass additional configuration through the `config` parameter using a [`RunnableConfig`](https://reference.langchain.com/python/langchain_core/runnables/#langchain_core.runnables.RunnableConfig) dictionary. This provides run-time control over execution behavior, callbacks, and metadata tracking.

Common configuration options include:

```python Invocation with config theme={null}
response = model.invoke(
    "Tell me a joke",
    config={
        "run_name": "joke_generation",      # Custom name for this run
        "tags": ["humor", "demo"],          # Tags for categorization
        "metadata": {"user_id": "123"},     # Custom metadata
        "callbacks": [my_callback_handler], # Callback handlers
    }
)
```

These configuration values are particularly useful when:

* Debugging with [LangSmith](https://docs.smith.langchain.com/) tracing
* Implementing custom logging or monitoring
* Controlling resource usage in production
* Tracking invocations across complex pipelines

<Accordion title="Key configuration attributes">
  <ParamField body="run_name" type="string">
    Identifies this specific invocation in logs and traces. Not inherited by sub-calls.
  </ParamField>

  <ParamField body="tags" type="string[]">
    Labels inherited by all sub-calls for filtering and organization in debugging tools.
  </ParamField>

  <ParamField body="metadata" type="object">
    Custom key-value pairs for tracking additional context, inherited by all sub-calls.
  </ParamField>

  <ParamField body="max_concurrency" type="number">
    Controls the maximum number of parallel calls when using [`batch()`](https://reference.langchain.com/python/langchain_core/language_models/#langchain_core.language_models.chat_models.BaseChatModel.batch) or [`batch_as_completed()`](https://reference.langchain.com/python/langchain_core/language_models/#langchain_core.language_models.chat_models.BaseChatModel.batch_as_completed).
  </ParamField>

  <ParamField body="callbacks" type="array">
    Handlers for monitoring and responding to events during execution.
  </ParamField>

  <ParamField body="recursion_limit" type="number">
    Maximum recursion depth for chains to prevent infinite loops in complex pipelines.
  </ParamField>
</Accordion>

<Tip>
  See full [`RunnableConfig`](https://reference.langchain.com/python/langchain_core/runnables/#langchain_core.runnables.RunnableConfig) reference for all supported attributes.
</Tip>

### Configurable models

You can also create a runtime-configurable model by specifying [`configurable_fields`](https://reference.langchain.com/python/langchain_core/language_models/#langchain_core.language_models.chat_models.BaseChatModel.configurable_fields). If you don't specify a model value, then `'model'` and `'model_provider'` will be configurable by default.

```python  theme={null}
from langchain.chat_models import init_chat_model

configurable_model = init_chat_model(temperature=0)

configurable_model.invoke(
    "what's your name",
    config={"configurable": {"model": "gpt-5-nano"}},  # Run with GPT-5-Nano
)
configurable_model.invoke(
    "what's your name",
    config={"configurable": {"model": "claude-sonnet-4-5-20250929"}},  # Run with Claude
)
```

<Accordion title="Configurable model with default values">
  We can create a configurable model with default model values, specify which parameters are configurable, and add prefixes to configurable params:

  ```python  theme={null}
  first_model = init_chat_model(
          model="gpt-4.1-mini",
          temperature=0,
          configurable_fields=("model", "model_provider", "temperature", "max_tokens"),
          config_prefix="first",  # Useful when you have a chain with multiple models
  )

  first_model.invoke("what's your name")
  ```

  ```python  theme={null}
  first_model.invoke(
      "what's your name",
      config={
          "configurable": {
              "first_model": "claude-sonnet-4-5-20250929",
              "first_temperature": 0.5,
              "first_max_tokens": 100,
          }
      },
  )
  ```
</Accordion>

<Accordion title="Using a configurable model declaratively">
  We can call declarative operations like `bind_tools`, `with_structured_output`, `with_configurable`, etc. on a configurable model and chain a configurable model in the same way that we would a regularly instantiated chat model object.

  ```python  theme={null}
  from pydantic import BaseModel, Field


  class GetWeather(BaseModel):
      """Get the current weather in a given location"""

          location: str = Field(..., description="The city and state, e.g. San Francisco, CA")


  class GetPopulation(BaseModel):
      """Get the current population in a given location"""

          location: str = Field(..., description="The city and state, e.g. San Francisco, CA")


  model = init_chat_model(temperature=0)
  model_with_tools = model.bind_tools([GetWeather, GetPopulation])

  model_with_tools.invoke(
      "what's bigger in 2024 LA or NYC", config={"configurable": {"model": "gpt-4.1-mini"}}
  ).tool_calls
  ```

  ```
  [
      {
          'name': 'GetPopulation',
          'args': {'location': 'Los Angeles, CA'},
          'id': 'call_Ga9m8FAArIyEjItHmztPYA22',
          'type': 'tool_call'
      },
      {
          'name': 'GetPopulation',
          'args': {'location': 'New York, NY'},
          'id': 'call_jh2dEvBaAHRaw5JUDthOs7rt',
          'type': 'tool_call'
      }
  ]
  ```

  ```python  theme={null}
  model_with_tools.invoke(
      "what's bigger in 2024 LA or NYC",
      config={"configurable": {"model": "claude-sonnet-4-5-20250929"}},
  ).tool_calls
  ```

  ```
  [
      {
          'name': 'GetPopulation',
          'args': {'location': 'Los Angeles, CA'},
          'id': 'toolu_01JMufPf4F4t2zLj7miFeqXp',
          'type': 'tool_call'
      },
      {
          'name': 'GetPopulation',
          'args': {'location': 'New York City, NY'},
          'id': 'toolu_01RQBHcE8kEEbYTuuS8WqY1u',
          'type': 'tool_call'
      }
  ]
  ```
</Accordion>

***

<Callout icon="pen-to-square" iconType="regular">
  [Edit the source of this page on GitHub.](https://github.com/langchain-ai/docs/edit/main/src/oss/langchain/models.mdx)
</Callout>





-----------

```python
from langchain_core.prompts import PromptTemplate

template = PromptTemplate(template="""xxx{a}yyy{b}zzz{c}""")
input_variables = ["a", "b", "c"]

prompt = template.invoke({
	"a": a
	"b": b
	"c": c
})
```

这段代码完全可以用 `f-string` 来写，为什么需要使用这个 prompt 模板类呢？主要三个理由：

1. 通过提示模板类的帮助来创建提示，会默认获得一些验证。也就是说，假设你在这里使用了三个占位符——style_input、paper_input、length_input——但不小心在代码中的输入变量里忘记了提供 length_input。这时候，有一个参数叫做 `validate_template`，如果将其设为 true，当运行这段代码时，它会自动验证这里所有的占位符是否都被提及了。如果没有，就会自动触发一个错误，代码也不会运行。当然多个其他参数值，比如 name 也不行。
2. 可以通过 save 保存为 json 文件
	```python
	template.save("template.json")                # save
	template = load_prompt("template.json")         # load
	```

3. 提示模板与 Chain 的整个生态系统紧密耦合。
	```python
	# 上面代码中相当于调用了两次 invoke，一次 template.invoke，一次 model.invoke
	# 组合为 chain
	chain = template | model
	result = chain.invoke({
		"a": a
		"b": b
		"c": c
	})
	```



-------------

**Output Parsers**

LangChain 中的输出解析器帮助将原始大语言模型响应转换为结构化格式，如 JSON、CSV、Pydantic 模型等。它们确保应用程序中的一致性、验证和易用性。


四种最常用的输出解析器：

* StrOutputParser：是 LangChain 中最简单的输出解析器，用于解析 LLM 的输出并将其作为纯字符串返回。基本上就是 `result.content`，好处就是组 chain，比如
	```python
	parser = StrOutputParser()
	chain = template1 | model | parser | template2 | model | parser
	chian.invoke(...)
	```
* JSONOutputParser
	调用 `getFormatInstructions` 函数时，这个 Parser 就知道应该发送哪些格式指令，之所以称之为格式指令的 `partial_variables`，是因为它在运行时不会被填充，用户不会告诉我们，它在运行之前就已经被填充了，但 JSON 输出解析器的特性会接收大量的 JSON 字典，你无法强制执行任何模式，也就是说你不能指定你的 JSON 对象
	```python
	parser = JsonOutputparser()
	
	tremplate = PromptTemplate(
		template="Give me the name, age and city of a fictional person \n {format_instruction}",
		input_variables=[],
		partial_variables={'format_instruction': parser.get_format_instructions()}
	)
	c
	result = model.invoke(prompt)
	final_result = parser.parser(result.content)
	
	--------------
	# chain 形式调用
	chain = template | model | parser
	result = chain.invoke({})
	print(result)
	```
* StructuredOutputParser：它帮助根据预定义的字段模式从 LLM 响应中提取结构化的 JSON 数据，同样，结构化输出解析器的最大缺点是，无法进行数据验证，只能指定需要的 JSON 结构
	```python
	schema = [
		ResponseSchema(name="fact1", description="Fact 1 about the topic"),
		ResponseSchema(name="fact2", description="Fact 2 about the topic")
		ResponseSchema(name="fact3", description="Fact 3 about the topic")
	]
	parser = StructuredOutputParser.from_response_schemas(schema)
	
	tremplate = PromptTemplate(
		template="Give 3 fact about {topic} \n {format_instruction}",
		input_variables=["topic"],
		partial_variables={'format_instruction': parser.get_format_instructions()}
	)
	prompt = template.invoke({'topic': "xxxx"})
	```
* PydanticOutputParserpa
	```python
	parser = PydanticOutputParser(pydantic_object=Person)
	template = ....    # 同上
	```


除此之外还有很多其他的解析器，比如有逗号分隔列表（CSV）、列表输出、Markdown、编号列表、XML输出解析器等等，还有很多可用的解析器。所以这些都包括在内，比如枚举输出解析器、日期时间解析器、输出修复解析器等。

-----------

三种不同类型的 Chain：

1. 顺序链
2. 并行链：`RunnableParallel({"chain1": xx|yy|zz, "chain2: x|y|z"})`
3. 条件链：`RunnableBranch((lambda x: x.sentiment == 'positive', chain1), (...), (lambda x: "could not find sentiment"))`


---------

**Runnables：**

* 为什么：重新构建，确保所有组件都是标准化的，并且能够无缝连接。

* 是什么：所有可运行对象都应该有共同的方法，其中最重要的方法是 `invoke`。那么，我们如何确保我们所有的组件都具有相同的方法呢？在面向对象编程中，有一个非常可靠的方法来确保这一点，我们称之为抽象。也就是说，我将创建一个名为 `runnable` 的抽象类，然后我所有的其他组件类都将继承它。

```python
def invoke(self, input_data):
    for runnable in self.runable_list:
        input_data = runnable.invoke(input_data)
    return input_data
```


可运行组件分为两大类：特定任务的可运行组件，可运行原语。

特定任务可运行组件：这些是已转换为可运行组件的 LangChain 核心模块，以便它们可以在管道中使用。执行特定任务的操作，如 LLM 调用、提示词生成、检索等。例如：`ChatOpenAI`、`PromptTemplate`、`Retriever` 

可运行原语：这些是构建 AI 工作流执行逻辑的基本构件。通过定义不同可运行对象之间的交互方式（顺序、并行、条件判断等），它们有助于协调执行流程。示例：
* `RunnableSequence` - 按顺序执行步骤（使用 | 运算符）
* `RunnableParallel` - 同时执行多个步骤。
* `RunnableMap` - 将同一输入映射到多个函数。
* `RunnableBranch` - 实现条件执行（if-else逻辑）。
* `RunnableLambda` - 将自定义 Python 函数封装为可运行对象。
* `RunnablePassthrough` - 直接将输入作为输出传递（充当占位符）。


大家好，我叫Nitesh，欢迎来到我的YouTube频道。在这个视频中，我们将继续我们的LangChain播放列表。我原本的计划是在覆盖完链和可运行内容后，再给大家讲解memory组件。但在稍作研究后，我发现LangChain团队正在逐步将memory组件从LangChain中移除，并将其整合到LangGraph中。因此，我决定在开始LangGraph的播放列表时，再为大家讲解memory组件。很可能这件事会在下个月完成。

所以在今天的视频中，我们将开始一个全新的内容，那就是使用LangChain来构建基于RAG的应用程序。没错，一个基于RAG的应用程序包含多个组件，今天我们将学习其中的一个组件，叫做文档加载器。在开始视频之前，我还想分享一个有趣的更新。YouTube推出了一个新功能，作为创作者，你可以在视频中开启配音功能。这基本上意味着，作为学生或观众，你会在视频上方看到一个选项，可以更改视频的音频语言。

我的视频是印地语的，但如果你想看英语版本，你可以通过更改音频设置来观看整个视频的英语版本。最近这个功能刚刚推出，我也在尝试使用。我发现这个更改音频的功能的翻译质量非常好，也就是说，如果有人听不懂印地语，他们可以轻松地通过英语理解整个视频。唯一的问题是，视频的音频声音与我的声音非常不同，可能会让你觉得有点滑稽。所以在这个视频中，我将再次尝试这个功能。

我想在你们面前做个实验，请大家试着用英语观看这个视频。如果你们觉得合适，请在评论区留言告诉我们。这样我们以后的所有视频都会默认启用英语音频。所以这是一个有趣的更新，现在让我们开始视频吧。在开始这个视频之前，我想给你们一个快速回顾，看看我们在这个播放列表中已经涵盖了哪些内容。到目前为止，我们在这个播放列表中重点讨论了两个重要的事情：首先，我们关注了长链的重要组件。

在上面我们看到了不同的组件，比如长链中的模型组件、提示组件或链式组件。所以，我不仅详细地向你们讲解了这些组件，同时我还亲自编写了实际的代码，以便你们能够很好地理解所有这些组件。其次，我在这个播放列表中非常关注另一个方面，那就是长链的核心概念。我确保向你们深入讲解了长链中使用的所有核心概念，比如可运行的概念，我已经通过两三个视频详细地讲解了这一点。

因此，在接下来的学习中，如果你看到这个概念，并且对它感到适应，那么总的来说，可以说在这一点上，我们对长链的基本原理已经清楚了，我们已经准备好使用长链来构建任何基于LLM的应用程序。接下来，在这个播放列表中，我们将学习如何使用长链来构建基于RAG的应用程序，接下来的4-5个视频中，这将是我们关注的重点。如果你在这一点上脑海中浮现出“RAG是什么”这个问题，那么首先让我给你一个快速的介绍，什么是基于RAG的应用程序，RAG一般是什么。到目前为止，生成式AI的浪潮中，最大的用例之一就是聊天机器人，我们可以在任何网站上使用它。

你可以去和它的聊天机器人聊天，就像你和ChatGPT聊天一样。ChatGPT可能是目前最受欢迎的生成式AI软件。在ChatGPT上，你会怎么做呢？你会打开网站，在那里输入你的文本，输入你的问题，然后按回车键，你就会得到它的回复。现在，这在大多数情况下都能正常工作，但在某些特定情况下，像ChatGPT这样的软件可能无法帮助你。比如，假设你去ChatGPT，问它关于时事的问题。

很有可能ChatGPT是基于过去的数据训练的，它没有关于当前日常生活中发生的事情的信息，比如今天或昨天发生的事情。在这种情况下，你就得不到答案。其次，如果你问关于你的个人数据的问题，比如问你过去一周收到的邮件相关的问题，显然ChatGPT无法给你答案。

因为ChatGPT没有看过那些数据。同样地，如果你是一名程序员，你向ChatGPT询问关于你公司文档的问题，显然ChatGPT无法给你答案，因为它没有看过那些数据。所以在那些ChatGPT没有数据来回答你的情况下，基于RAG（检索增强生成）的应用程序就能帮上忙。在RAG应用中，你本质上做的是为你的大型语言模型提供一个外部知识库。现在这个外部知识库...

任何东西都可以，可能是你公司的数据库，可能是大量的PDF文件，也可能是你的个人文档。你以某种方式将这个外部知识库与这个LLM连接起来。现在，每当有用户来问一个这个LLM不知道的问题时，这个LLM可以迅速去这个知识库查找，找出这个问题的答案是什么，并借助这个外部知识库的帮助为你提取出答案。这里写的正是这个东西。

所以，RAG是一种将信息检索与语言生成相结合的技术。那么，信息检索是从这个外部知识库进行的，而语言生成是在LLM的帮助下完成的。该模型从知识库中检索相关文档，然后将其作为上下文生成准确且基于事实的回应。那么，使用RAG的最大好处就是，你可以从任何LLM中获取最新的信息。

第二，你还能获得极大的隐私保护。想象一下，如果你需要对你的个人文档提出一些问题，那么一个选择是什么？你会在ChatGPT上上传这个文档。现在，如果那是非常机密的信息，那么你在ChatGPT上上传它就不是一件好事。那么在RAG中你会怎么做？你可以上传你的文档，并在上面提问，而且没有文档大小的限制。现在假设你有一个1GB的文档，你想完整地上传到ChatGPT上，但ChatGPT的上下文长度有限。

你可能不会指望它能完整阅读整个文档并给出答案，但即便如此，RAG最佳应用程序也能帮助你。在RAG中，你可以将整个文档分成若干部分，轻松处理。这些都是你使用RAG最佳应用程序后能获得的好处，这也是为什么RAG最佳应用程序目前在行业中是一个非常强大的趋势。所以，我希望你现在对RAG最佳应用程序以及RAG是什么有了一个大致的概念。接下来，我们将更详细地讨论这个问题。现在我的计划是，我不会直接教你如何一次性创建RAG或RAG最佳应用程序，而是……

我要做的是首先教你RAG（检索增强生成）应用程序中最重要的组件，一旦你熟练掌握这些组件后，我就会教你抛开所有这些组件，最终学会构建一个完整的RAG应用程序。RAG最重要的组件包括：文档加载器、文本分割器、向量数据库和检索器。只要组合这四个组件，你就能构建任何基于RAG的应用程序，无论架构多么复杂，大多数情况下都是由这四个组件构成的。今天的视频将会讲解文档加载器，它能帮助你从任何来源加载文档。所以在今天的视频中，我们将探讨不同的来源...

他们将学习如何加载文档到长链中，然后在接下来的视频中，我们还将涵盖顺序文本分割器、向量数据库和检索器。一旦这四部分内容都覆盖完毕，我们最终将学习如何创建基于RAG的应用程序。现在，说到文档加载器，长链中有数百种文档加载器，不可能让你逐一学习每种加载器。因此，我想到了一个方法，就是向你介绍文档加载器最重要的原则。

今天视频中我要教的是最重要的概念，除此之外还有四个你们可能会最常用的文档加载器，我会教你们如何使用它们。我会教你们文本加载器、PDF加载器、网页加载器和CSV加载器。另外，我还会稍微指导一下其他存在的文档加载器以及如何使用它们。好的一点是，尽管这些文档加载器处理的是不同的东西，但它们的基本原理是完全相同的。

所以如果你能很好地理解一个文档加载器，那么你基本上可以轻松地与任何人合作。那么，简而言之，如果我要总结整个讨论，现在我们大家在很好地学习了Lang Chain之后，将开始构建基于RAG的应用程序。而在Lang Chain中构建基于RAG的应用程序，有各种不同的组件，其中第一个组件就是文档加载器，我们今天就要学习它。文档加载器中有很多种文档加载器，我会在这个视频中向你展示四个最重要的。好了，这就是总结。

我真的希望你现在对整个大局有了清晰的认识。现在让我们开始视频吧。现在，伙计们，我们来讨论一下文档加载器到底是什么。看这里，有一个定义。首先我会读一下这个定义，然后我会向你解释。所以，文档加载器是LangChain中的组件，用于从各种来源加载数据到标准化格式，通常是文档对象，然后可以用于分块、嵌入和检索。

而生成基本概念是，目前LangChain的开发者们注意到，要构建基于RAG（检索增强生成）的应用程序，你需要加载数据。这些数据可能存在于不同的来源中，比如数据可能是PDF文件、文本文件，也可能在某个数据库中，或者存储在某个云服务提供商那里。因此，数据可以从多个来源获取，而且有数百种来源。我们需要做的是确保无论我们从哪个来源获取数据，这些数据都能转换为一种统一的格式。

他们创建了一种标准化格式，这样你就可以轻松地将其与Lang Chain的任何其他组件一起使用。在这里，这些人建立了一种标准化格式，他们将其命名为“document”。你会注意到，当我使用文档加载器从任何来源获取数据时，它总是以document的格式呈现给我。document的格式是这样的：每个document对象包含两个部分，一个是page content，即实际的数据内容；另一个是围绕它的metadata，比如来源是什么，文件来自哪里，创建时间，最后修改时间等。

作者名称 这类信息您可以在元数据中找到。那么，文档加载器基本上是LangChain中的这些实用工具，其工作是从不同的数据源获取数据并将其转换为标准化格式。这个标准化格式是什么呢？就是文档对象。好了，现在您知道文档加载器是什么了，接下来我将逐一为您介绍一些最重要的文档加载器。

让我们从文本加载器开始，如果我们要讨论文本加载器，那么文本加载器是Lang Chain中最简单的文档加载器之一，它的工作非常简单，就是读取文本文件并将其作为文档对象引入Lang Chain中，就是这样，非常简单。理想情况下，当你需要处理任何类型的日志文件、代码片段或类似YouTube视频转录这样的内容时，你会使用它。

那么你会为它使用文本加载器，我给你看看代码是怎么工作的，这样任何类型的文档加载器你都可以在Lang Chain中找到，Lang Chain下划线社区包里有所有的文档加载器，你都可以在Lang Chain下划线社区包中找到。所以在这里你需要写的是Lang Chain社区.documentloaders，然后从这里你可以导入任何类型的文档加载器。目前我们在这里导入的是文本加载器。在这里你需要做的是，首先创建一个加载器对象，这是一个文档加载器对象，而且是文本加载器类的对象。在这里初始化这个对象的时候，你可以发送不同的参数。

最重要的参数是，您需要在这里指定文本文件的路径。比如，我现在有一个名为cricket.txt的文本文件，这实际上是我用ChatGPT生成的一首关于板球的诗，然后我把它保存为一个文本文件并放在我的文件夹里。所以我要做的就是加载这个文件，在这里我会给出它的路径：cricket.txt，这就是路径。其次，您还可以在这里指定文本文件使用的编码格式。在我的情况下，它是utf-8，主要是因为有一些特殊字符，所以我需要在这里指定编码。

很有可能你不需要在文本文件中提供这种编码，所以我创建了自己的文档加载器对象。现在，这个文档加载器对象有一个名为load的函数，你只需要调用load函数即可。它会做什么呢？它会将你的文本文件作为文档加载到内存中。因此，我们将其存储在一个变量中，并称之为docs。现在，我将打印docs。如果我运行它，你可以看到这是输出结果。首先，如果我展示docs的类型是什么的话...

那么你会注意到docs实际上是一种Python列表类型。无论你使用哪种文档加载器——现在我们用的是文本加载器，之后我们会用PyPDF加载器，或者任何其他加载器——你总会发现一件事：LangChain的每个文档加载器在加载文档时，都会将其作为文档列表加载，明白吗？这意味着你的文档实际上被分割成了多个部分，并以列表形式存储后提供给你。希望你能理解这一点。

今后无论我们读取多少文档加载器，您都会注意到，每个人的输出都是一个文档列表，对吧？所以，如果我在这里向您展示这个特定列表中有多少文档，您会发现，在当前这个特定情况下只有一个文档，对吧？您如何提取它呢？您可以使用这个简单的Python代码来提取它，即我们取出列表的第一个项目，对吧？如果我运行这段代码，那么现在您可以看到，

你现在看到的这一切，如果我把docs zero的类型提取出来，你就会开始理解这些东西了。看，这是docs的zeroth item的数据类型，它是一个document，明白吗？我希望你能理解。既然这是一个document，我之前告诉过你，它里面会有两样东西：第一个是page content，第二个是metadata。让我们检查一下这两样东西是否都在这里。所以我在做什么呢？我再次打印docs zero，你可以直接在这里看到metadata。如果你稍微往上滚动一点，实际上这是一个相当大的文本文件。

我快速滚动到这里，你看，这个页面内容即使被称为page content，你也看到了一件事——页面内容始终会存在于文档中，metadata也始终存在。而且你可以分别提取这两部分，比如你可以在这里写.page content，在这里你可以写docs[0].metadata，这两部分你都可以单独提取。我再次运行，你可以看到，这部分是metadata，而上面这整个部分是你的主文本。好的，现在你可以以任何方式使用它，比如假设我可以在这里形成一个链式操作：我正在提取这段文本，并将其发送给一个LLM。

那么我要做的是快速写下lanchain openai，从lanchain core.output_parsers导入chat openai，从lanchain core.prompts导入prompt template，从.env导入load_env并调用它。创建一个模型，模型将是chat openai。创建一个提示，这将是prompt template类的对象，模板将是“为以下诗写一个摘要/n”，然后这里会放入我的诗，输入变量中会放入诗。然后我可以创建一个解析器，解析器将是string output parser的对象，好的。

现在我所能做的就是在这个地方我可以形成一个链条，而链条将由提示、模型和解析器组成。现在我所能做的就是可以调用chain.invoke，并且在chain.invoke中，我可以用文档的第0页内容代替诗歌发送，好吧。然后这个东西我现在可以打印，所以保存了，运行了，那么现在诗歌已经打印出来了，元数据也打印出来了，现在摘要也会打印出来，看这个，摘要也出来了，好吧。所以你可以看到这有多简单，整个流程变得多么简单。如果你愿意的话，这个函数loader.load和你正在从中提取页面内容的东西，你可以把它包装成一个可运行的lambda，然后它也可以成为这个链条的一部分。

那么有多少灵活性呢，好的，那么你学到的基本概念是，当你使用文本加载器时，或者就此而言，当你使用任何文档加载器时，你会得到一个文档对象列表，对吧，你可以从中提取文档对象，每个文档对象都附带两样东西，一个是它的页面内容，另一个是它的元数据，你可以根据自己的需要使用这两样东西，好的，所以我希望你已经理解了文本加载器是如何工作的，现在我要向你展示你有一个包含25页的PDF。

那么，您只需将其发送到PDF文档加载器中，您有一个包含25页的PDF文件，当您将其发送到PDF文档加载器中时，它会为您生成25个文档对象，基本上处理后，您将得到一个包含25个文档对象的列表，明白吗？请看这里，如果您有一个PDF文件，那么每一页都会对应一个文档对象，每个文档对象都有自己的页面内容和元数据，其中包括页码和来源等信息。

你会得到它的，好吗？我希望你能理解。现在，pypdf加载器内部使用pypdf库来读取PDF文件，好吗？这就是为什么这个特定的文档加载器在处理扫描的PDF或复杂布局时效果不佳。如果你有一些非常简单的文本PDF文件，你可以使用pypdf加载器来处理它们。如果你有一些像照片转换成的PDF那样的文件，那么对于这类情况，你还有其他文档加载器可供选择，我稍后会向你展示。

我来做一件事，我会写代码给你看如何与这个特定的文档加载器一起工作。在展示代码之前，我先给你看我有这个PDF文件，这就是那个PDF文件，基本上是我们的深度学习课程大纲。你可以注意到它有23页。我们要做的是用pypdf加载器来加载这个特定的PDF文件。首先我们需要做的是进入langchain.community.documentloaders，这次我们要导入pypdf加载器，同时你要确保你的机器上已经安装了pypdf，否则这段代码不会运行。

我已经安装了Alldie，这就是代码能运行的原因，好吧，现在这是LangChain最棒的部分，你可以使用任何文档加载器，它的使用格式是完全相同的，所以在这里你也会创建一个加载器对象，这将是PyPDF加载器的对象，在这里你给出你的文件路径，所以我这里有dlcurriculum.pdf，所以我会写dlcurriculum.pdf，好吧，然后我们会做什么，我们会调用loader.load函数，这样我们就能得到我们的文档对象集合，现在我要做什么，我要打印docs，只是为了展示给你看。

我们得到了一个文档对象列表，大家可以看到，这是输出结果，列表里包含了所有内容。如果我来展示列表的长度，保存、运行，看这里，总共有23个文档对象，因为我们的PDF有23页。基本上，你的PDF有多少页，我们就会把每一页转换成一个文档对象。现在，我来提取并展示第一个文档对象。

所以我要做的就是简单地写print docs का 0 और dot page content，我会把它的page content提取出来给你看，同时我也会向你展示它的metadata是什么，dot metadata，保存了，运行了，看这个伙计们，你现在看到的就是你第一页的内容，对吧，注意看，你在PDF中看到的正是这个东西，campus x深度学习课程，人工神经网络以及如何改进它们，对吧，right，最后如果你去看，你会看到metadata，

好的，这里写了制作人的名字，创作者的姓名也标注了，对吧？我希望你能完全理解这部分内容。在LangChain中加载PDF就是这么简单。在讲解PDF加载器之前，我提到过我们正在学习的PDF加载器叫做PyPDF Loader。PyPDF Loader并不是LangChain中唯一的PDF加载器，还有很多其他可用的PDF加载器。PyPDF Loader适用于主要包含文本数据的PDF文件，当你需要加载文本时可以使用它。但PyPDF Loader在很多场景下效果并不理想，比如你的PDF是由扫描图片组成的。

那么PyPDF Loader就不会那么好用，所以对于这类用例，LangChain中还提供了其他一些PDF加载器。虽然我无法一一演示所有加载器的用法，但我可以为您指引正确的资源方向。这里列举了一些您应该了解的其他PDF加载器：例如，如果您处理的PDF包含大量表格结构，并且需要提取表格数据，可以使用名为PDF Plumber Loader的加载器；如果想处理扫描图像类的PDF，可以选择unstructured PDF Loader或Amazon Textract PDF Loader。

好的，如果你的PDF中有很多布局，那么PyMuPDF是一个选择，而对于结构提取，你有unstructured PDF Loader。我会告诉你一个很好的教程，你可以参考它。如果你正在提取布局或从PDF中提取图像数据，那么你可以在Lanchain的文档中找到这个特定的教程，它会告诉你如何提取结构。你需要访问这个特定的链接，我会在描述中给你这两个链接。所以这些都是你的PDF加载器。

在Lanchain中，有PyPDF，我们曾与之合作，还有unstructured PDF loader，Amazon Textract，PDF Plumber，PyMU PDF，你只需点击其中任何一个，就能看到它的文档和用例。同样的功能，加载、延迟加载，所有这些你都能看到。所以我不建议你去全部阅读，因为没必要研究所有内容。

这个项目会根据不同的项目有所变化，你只需要掌握基本概念。当你开始做一个项目时，如果在这个项目中需要用到这些资源，你可以随时来参考它们。到目前为止，我们已经学到的内容已经足够了。现在我们已经学会了，如果你有一个单独的文本文件或一个单独的PDF文件，如何在Lang Chain中加载它。但是如果你有一个文件夹，并且这个文件夹中有多个PDF文件呢？

要把所有这些一起加载到链中，你会怎么做？这个问题的答案是目录加载器。那么，目录加载器是做什么的呢？它是一个文档加载器，帮助你从一个目录中加载多个文档，明白吗？让我来展示给你看。那么，在这个点上，我的项目文件夹里有一个名为“books”的文件夹，里面有三本关于机器学习的书。这是第一本书，大约有326页；这是第二本书，有392页；这是第三本书，有468页。现在我要做的是写一段代码。

借助它的帮助，我将这三本书一起加载到land chain中，好吗？所以，为此我们需要做什么，再次从land chain社区内部，从文档加载器内部，必须加载，目录加载器，同时还有pypdf加载器，因为本质上这三者都是pdf文件，对吧？现在这里主要的工作开始了，现在再次创建一个加载器对象，而这个加载器对象将是目录加载器的对象。现在在这个目录加载器中，你需要指定两件事情，首先你要指定你的目录路径，在我的情况下，在我的项目文件夹内部，

有一个名为books的目录或文件夹，其次，我需要说明，在这个文件夹内，我需要加载哪些文件。为此，有一个名为globe的参数，你可以在这里指定一个模式，所有满足这个模式的文件都会被选中。例如，我在这里给出的模式是asterisk.pdf，这意味着在books文件夹内，所有PDF文件都会被加载。现在，你可以在这里提供各种不同的模式，我在笔记中也已经创建了一些示例。例如，如果你提供了这个模式，那么它的意思是，

这意味着你想从所有的子文件夹中加载所有的文本文件，这意味着你想从根目录中加载所有的PDF文件，这意味着你想从数据文件夹中加载所有的CSV文件，这意味着你想从所有的子文件夹中加载所有的文件。你可能在以前使用DOS时见过这些模式。然后，你需要告诉你的加载器是哪个类，在我们的例子中，假设所有的文件都是PDF文件。

我们将使用的加载器类是pypdf-loader，好了，现在从这里开始我们的工作就开始了，我们将调用loader.load，并且无论从那里返回什么，我们都会将其存储在docs中，然后我会再次打印出来给你看，首先我会让你看到有多少文档正在加载，好了，保存并运行，你可以看到，总共有1186个文档返回给我，为什么会有1186个呢？回答这个问题非常简单，你有3个PDF文件，一个文件有326页，一个有392页，一个有468页，当你把这3个数字加起来时，

那么你会得到1186，好的，所以基本上发生了什么，你正在将所有PDF的每一页，以一个文档的格式呈现，好的，所以如果你想看第一个PDF的第一页，你会写docs0.pagecontent，如果你想看元数据是什么，你会写docs0.metadata，现在如果你运行这个，看这里，那么在第一个页面中，只有这么多内容，写着“一”，然后看元数据，元数据中你会看到很多东西，什么时候创建的，什么时候最后修改的，除此之外在source中，你会看到书的名称，这是哪个PDF的一部分，里面有多少页，所以第一个PDF有326页。

这是它的哪个页面呢？如果我直接加载第326页，那将是第一本书的最后一页。如果我运行这个，看，这是最后一页，这是它的内容。实际上，我们去了第326页，然后第二本书的第一页就开始了，因为这里显示的是《Python机器学习入门》PDF，它有392页，这是它的第一页。这意味着如果我在写325，那将是第一本书的最后一页，因为我们的索引是从0开始的，0到325就是326页，明白吗？所以通过这种方式，你可以在一个目录中加载任意数量的PDF文件，不仅如此，你还可以加载文本文件。

这个直接加载器可以与其他所有文档加载器一起使用，所以接下来我教你的那些文档加载器，你也可以和目录加载器一起使用，明白了吗？所以我希望你已经理解了如何从一个目录中加载所有文件。现在我不知道你是否注意到了，但上次我们写的那个目录加载器的代码，当我们运行它的时候，运行起来花了一点时间，对吧？

我再次向你展示，所以如果我运行这段代码，你会注意到这段代码在运行和输出时花费了一点时间，原因是我们在使用目录加载器同时加载三个PDF文件。现在想象一下，如果加载三个PDF文件就花了这么多时间，大约10-12秒，那么如果这个特定文件夹中有100个同样大小的PDF文件，会发生什么情况？我希望你能理解，那将花费相当多的时间。这段代码的第二个问题是什么？就是你同时在内存中加载这三个PDF文件。

这样你就可以在其上运行操作了。现在这是三个PDF文件，你可以一次性将它们加载到内存中，但如果有一百个PDF、五百个PDF，那么一次性将它们全部加载到内存也是不可能的。所以这两个问题——加载需要时间，以及所有内容同时加载到内存——为了解决这两个问题，LangChain中有一个解决方案，那就是懒加载。它的工作原理是什么呢？我刚才已经告诉过你，任何文档加载器你都可以使用，它们都有一个加载函数，通过这个函数你可以加载文档。看这里，loader.load在这里也有，这里也有，这里也有。你随便选一个文档加载器都可以。

它里面一定会有这个加载函数，但同时每个文档加载器中也有一个懒加载函数，它的功能虽然相同，都是将文档加载到内存中，但工作方式不同。那么加载函数做什么呢？它会进行文档的急切加载，这基本上意味着如果有一个500页的PDF文件，加载函数会做什么呢？它会将这500页一次性全部加载到内存中，创建500个文档，并将所有这些文档同时加载到内存中，然后返回一个文档列表，你可以根据需要对其进行操作。而懒加载的工作方式则略有不同。

如果你给懒加载一个500页的PDF，它不会直接给你一个文档列表，而是返回一个生成器——你可能在Python里也读到过——这是一个文档生成器。那么它的特别之处是什么呢？就是借助它，你一次只在内存中加载一个文档，运行你需要的操作，然后这个文档就会从内存中移除；接着加载第二个文档，执行相应的操作，之后它也会被移除，以此类推。

那么这就是load和lazy load的主要区别，看这里，写得非常好，load函数执行的是eager loading，一次性将所有内容加载到内存中，它会返回一个文档列表，一个文档对象列表，并且一次性将所有文档加载到内存中。当你拥有的文档数量较少或体积较小时，比如只有一两个PDF文件，这时你应该使用load函数，因为你需要一次性将所有内容加载到内存中。而lazy loading的特点是它按需加载，一次只加载一个文档。

它会返回一个文档对象生成器，文档不会一次性全部加载，而是根据需要逐个获取。当你需要处理大量文档或文件时，或者需要进行流式处理而不占用太多内存时，就应该使用这种方式。好了，希望你能理解基本概念，即立即加载与延迟加载的区别。如果你需要处理大量文档，就应该使用延迟加载。

因为这里发生的情况是，你得到的不是一个列表而是一个生成器对象，你可以在这个生成器上运行循环，一次处理一个文档。我很快向你展示差异在哪里。看，这是我们有的目录加载器代码，我们有3个PDF文件。现在我先运行加载给你看，然后再运行懒加载给你看。所以在这里，我有1100个文档，它们一次性加载到内存中。现在我正在做的是在这个列表上运行一个循环，`for document in docs, print`，每个文档的元数据。看，我运行了这个代码。

现在看看这段代码如何运行，一开始很长时间什么都不会发生，代码正在内存中创建所有文档，然后你会得到一个列表，接着你在列表上运行循环，并打印所有文档的元数据，看这里，现在我们终于看到了输出，现在看看我将运行懒加载而不是加载，你会看到区别，你不需要一开始就等待，看这里，已经开始打印了，所以发生了什么，每次内存中有一个文档进来，我们打印它的元数据，删除它，然后下一个进来，打印它的元数据，删除它，我们这样做了1100次。

这就是为什么你看，保持一致的时间让这一切发生，好吧，那么这就是加载和懒加载之间的主要区别，简而言之，你只需要记住这一点，如果你有很多文档，不可能一次性全部加载到内存中，那么你就有懒加载这个选项，好吧，这就是你在这里唯一需要学习的东西，接下来我们要讨论一个非常有趣的文档加载器，它的名字叫基于网络的加载器，从名字你可能已经有点理解了，这个加载器的特点是什么，就是你可以借助它的帮助，抓取任何网页的内容，然后加载到链中。

然后你可以就它提出问题，这里写的是，基于网络的加载器是Land Chain中的一个文档加载器，用于从网页加载和提取文本内容。假设你有一个网页，我们在Flipkart上搜索了MacBook Air，这里有很多文本，而且文本分布在很多不同的地方，比如这里有文本，除此之外下面还有其他不同的文本。而我们这个加载器在内部是如何工作的呢？它使用了两个Python库，一个是requests，借助它你可以向那个网页发送HTTP请求；另一个是Beautiful Soup，借助它你可以理解那个网页的HTML结构。

并将其转换为文本格式，带到LangChain中。通常，当您处理静态网站时，会使用此加载器。如果您的网站主要是博客、新闻文章或任何公共网站，您可以通过这个特定的加载器在LangChain中加载它。如果谈到限制，它在静态页面上表现更好。如果您的网页包含大量JavaScript，即用户操作时会触发很多动态内容，那么在这种设置下，此加载器的效果就不那么理想。针对这种情况，有一个单独的加载器，称为Selenium URL加载器。这个加载器在静态网页上表现非常好，尤其是HTML内容较多的页面。好吧，我来做一件事。

我给你看一段代码，我会在这个特定的网页上加载它，然后问它一些问题，好的，所以从代码的角度来看，一切都会非常相似，你需要再次从lanchain community.documentloader导入，这次是基于web的loader，我们将创建一个基于web的loader类的loader对象，这里你需要一个URL，好的，所以我们在URL中传递的是这个URL，嗯，所以这是URL，现在就像任何其他文档加载器一样，你调用loader.load类，然后无论你得到什么，你都把它存储在docs中，我给你看一下docs的长度，然后我也会给你打印docs，实际上我会给你展示docs的第0项的页面内容。

好的，那么我们运行了这段代码，你可以看到，你能够看到网页的内容，对吧。借助Beautiful Soup的帮助，你学会了如何去除HTML标签，提取里面的文本。这里还有一点需要注意，你得到的文档数量只有一个，这意味着当你使用基于网络的加载器加载单个URL的内容时，你会得到一个单一的文档。但这里你有一个灵活性，你可以传递一个URL列表而不是单个URL。比如，如果你想一次性加载四五个不同网页的文档。

那么你可以把所有的URL放在一个列表里发送，那样的话，你会得到每个URL对应的文档对象，但我们这里不这么做，好吗？现在我们有了这个页面，我们来做一件事，通过问一些问题来看看。为此，我们需要使用这个设置，就是文本加载器。所以我要做的是，我会复制这么多代码，然后在这里粘贴。现在我要做的是，我会在这里写上“回答以下问题\n”，这是“来自以下文本的问题”，然后这里就出现了“文本”。输入变量中会有两个东西，一个是文本，另一个是问题，好了。解析器准备好了，URL准备好了，加载器准备好了，现在不是打印。

让我们形成一个链条，链条将会形成，提示、模型、解析器，然后我们将调用chain.dot.invoke，这里我们将发送问题，这里我们将发送文本，文本中将会包含我们的文档0.page.content，而问题中我们可以问，这个产品的峰值亮度是多少，类似这样的问题，然后让我们打印，无论我们返回什么，保存了，运行了，所以它说文本没有提供关于产品峰值亮度的信息，让我们把问题改成我们正在谈论的产品是什么，所以我们正在谈论的产品是Apple MacBook Air M2，这是产品规格，所以我不知道这里有什么内容。

我刚才提了这些问题，所以你可以轻松加载任何网页并在上面提问。我突然想到一个非常有趣的项目创意：如果我们开发一个Chrome插件，每当任何网站打开时，我们就在那里启动这个插件，然后用户可以实时通过该插件就特定页面进行聊天。这可能会是一个非常棒的项目创意。要实现它，你需要开发一个Chrome插件，后台需要运行一个API，这个API通过文档加载器与LLM进行交互。所以这又是一个不错的项目创意，而且并不太复杂。

但这其中有一个令人惊叹的因素，所以你可以尝试一下。在本视频中，我们要介绍的最后一个文档加载器叫做CSV加载器。从名字上你就能明白，这个特定的文档加载器是用来在Lang Chain中加载CSV文件的。如果你有一个CSV文件，并想通过大型语言模型（LLM）来提问，那么你就可以使用这个加载器。接下来，我会快速向你展示它是如何工作的。我已经创建了一个文件，实际上我已经在里面写好了代码。

我这里有一个CSV文件，名为“社交网络广告”，其中有5列：用户ID、性别、年龄、预估薪资、是否购买，总共有400行。这个特定的加载器是如何工作的呢？它为每一行创建一个不同的文档对象。所以如果我创建一个加载器对象，在这里我会给出我的CSV文件的路径，然后执行loader.load。实际上这里称为docs，如果我要展示docs的长度，你会看到输出是400。基本上，数据中的每一行都会对应一个文档对象。现在我要打印一个单独的文档对象。

所以我正在打印0份文档，我已经运行了它，你可以看到，这是第一行，这里的页面内容是这么多，所以你得到的是一个字符串，其中给出了每个列名和值，在元数据中你有来源，并且给出了行的值，如果我在这里输入1，然后运行，那么我会看到第二行，这是第二行，所以这么多是页面内容，这是你的元数据，所以这就是csv加载器的工作原理，你可以用它进行懒加载，如果你有一个大的csv文件。

你可以通过生成器遍历每一行，并对其执行任何类型的操作，比如你可以轻松地询问某一特定列的最大值是什么。这也是一个加载器，考虑到你正在从事数据分析工作，将来可能会用到它。以上就是最常用的4种加载器，除此之外，LangChain中还有许多其他文档加载器，让我为你快速概述一下。

所以如果你访问这个网址，你就能获取关于LangChain中所有文档加载器的信息，这些加载器被清晰地分类，比如网页类加载器，它们能帮助你加载网页内容，包括基于网页的加载器、非结构化加载器、站点地图加载器、超浏览器、AgentQL等。还有很多针对PDF的加载器，以及与云服务相关的加载器，比如用于数据加载的S3、Azure、Dropbox、Google Drive等。此外，还有针对社交平台、消息服务、生产力工具（如GitHub）的众多数据加载器。

以及许多常见的文件类型，有许多数据加载器，如CSV加载器、JSON加载器，这里你会看到一个完整的列表，LangChain中存在的每一种数据加载器，比如YouTube转录，你必须理解它是如何工作的，点击这个链接，这里你会看到它的使用案例和相关代码，但我再次建议，你不需要阅读所有的文档加载器，根据项目来学习。

所以，如果你的下一个项目确实需要加载YouTube字幕，那么你可以从文档中查阅这个特定的加载器。否则，阅读所有的加载器并没有太大用处。我已经教了你那些重要的、基础的知识，当你需要的时候，你可以从文档中查阅。在结束视频之前，还有最后一件事我想和你讨论。假设你正在进行一个项目，而你的数据源是这样的一个数据源，LangChain中没有现成的文档加载器可用。在这种情况下，你可以创建自己的自定义文档加载器，在其中定义...

加载函数如何工作，或者懒加载函数如何工作，LangChain为你提供了这个功能，你可以创建自己的自定义数据加载器。所以在这里，你基本上需要创建一个类，并且必须从基础加载器类继承该类，然后在这里，你可以添加自己的懒加载和加载函数，通过编写自定义逻辑，你可以制作自己的自定义数据加载器。事实上，LangChain中已经存在的所有这些数据加载器都是这样制作的。

因为LangChain提供了这个功能，你可以制作自己的自定义数据加载器，所以在LangChain社区中，许多程序员为了自己的使用场景，制作了不同类型的数据加载器，然后它们被添加到了LangChain中。这就是为什么随着时间的推移，LangChain拥有了一大批数据加载器、文档加载器，你也可以使用它们。这就是为什么所有这些文档加载器都在langchain_community包中，因为它们都是由社区开发的。

所以如果你遇到这种情况，找不到符合需求的文档加载器，你可以轻松阅读本页内容，然后创建自己的自定义数据加载器或文档加载器。好了，以上就是本视频的全部内容，希望我所讲解的内容对你有所帮助。如果你喜欢这个视频，请点赞支持；如果还没订阅频道，记得订阅哦。下期视频再见，拜拜。

