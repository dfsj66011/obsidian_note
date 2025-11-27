
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

### Rate limiting


<Accordion title="Initialize and use a rate limiter" icon="gauge-high">
  LangChain in comes with (an optional) built-in [`InMemoryRateLimiter`](https://reference.langchain.com/python/langchain_core/rate_limiters/#langchain_core.rate_limiters.InMemoryRateLimiter). This limiter is thread safe and can be shared by multiple threads in the same process.

  ```python Define a rate limiter theme={null}
  from langchain_core.rate_limiters import InMemoryRateLimiter

  rate_limiter = InMemoryRateLimiter(
      requests_per_second=0.1,  # 1 request every 10s
      check_every_n_seconds=0.1,  # Check every 100ms whether allowed to make a request
      max_bucket_size=10,  # Controls the maximum burst size.
  )

  model = init_chat_model(
      model="gpt-5",
      model_provider="openai",
      rate_limiter=rate_limiter  # [!code highlight]
  )
  ```

  <Warning>
    The provided rate limiter can only limit the number of requests per unit time. It will not help if you need to also limit based on the size of the requests.
  </Warning>
</Accordion>

### Base URL or proxy

For many chat model integrations, you can configure the base URL for API requests, which allows you to use model providers that have OpenAI-compatible APIs or to use a proxy server.

<Accordion title="Base URL" icon="link">
  Many model providers offer OpenAI-compatible APIs (e.g., [Together AI](https://www.together.ai/), [vLLM](https://github.com/vllm-project/vllm)). You can use [`init_chat_model`](https://reference.langchain.com/python/langchain/models/#langchain.chat_models.init_chat_model) with these providers by specifying the appropriate `base_url` parameter:

  ```python  theme={null}
  model = init_chat_model(
      model="MODEL_NAME",
      model_provider="openai",
      base_url="BASE_URL",
      api_key="YOUR_API_KEY",
  )
  ```

  <Note>
    When using direct chat model class instantiation, the parameter name may vary by provider. Check the respective [reference](/oss/python/integrations/providers/overview) for details.
  </Note>
</Accordion>

<Accordion title="Proxy configuration" icon="shield">
  For deployments requiring HTTP proxies, some model integrations support proxy configuration:

  ```python  theme={null}
  from langchain_openai import ChatOpenAI

  model = ChatOpenAI(
      model="gpt-4o",
      openai_proxy="http://proxy.example.com:8080"
  )
  ```

  <Note>
    Proxy support varies by integration. Check the specific model provider's [reference](/oss/python/integrations/providers/overview) for proxy configuration options.
  </Note>
</Accordion>

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




现在我该做什么呢？我要创建我的链条，首先我会创建一个笑话生成器链条。笑话生成链条是怎么构建的呢？它将通过可运行序列的帮助来构建，我会在其中发送三样东西：我会发送我的提示1，我会发送我的模型，还会发送我的解析器，明白了吗？

这已经成为了我的笑话生成器链，这部分我已经完成了。现在我需要做的是创建这个并行链。并行链将通过可运行并行来构建，在字典中我会写下笑话，这将是一个可运行的传递，然后我会写下解释，这本身就是一个链。所以这将是一个可运行的序列，然后会到达提示到模型和解析器，好的。

那么我希望你能理解，现在我们创建了这个部分，这里面有一个路径，这是另一个路径，对吧，现在我们需要做的是将这两者相互连接起来，画出这个连接。所以我们会写 final chain is equal to joke generation chain。好的，为了连接这两个，我们也会使用 runnable sequence，并且我们会使用 joke generation chain 和 parallel chain，这样我的 final chain 就完成了。

现在我要写final chain dot invoke，然后这里我的主题会是let's say cricket，无论结果是什么我都会打印出来，好的，那么现在返回给我的是一个dictionary，我们先运行一次，你可以看到这是the output，这就是我们的dictionary，其中第一部分是joke，这部分是joke，然后explanation部分是分开的，好的，现在你可以从这个dictionary中单独提取joke和explanation。

接下来我们将要学习的Runnable Primitive叫做Runnable Lambda，这也是一个非常实用的Primitive。通过它的帮助，你可以将任何Python函数转换为一个Runnable。如果你已经将某个Python函数转换为Runnable，那就意味着它可以与其他Runnables一起工作，就像你在处理一个任务一样。

你在进行一项操作，从公司的数据库中加载客户的评论，然后将它们发送给一个LLM（大型语言模型），LLM会告诉你这些评论的情感倾向。但你意识到，数据库中这些客户的评论并不十分干净，也就是说，这些文本中存在各种各样的问题，比如里面夹杂着HTML标签。

里面有标点符号，里面有你的表情符号、emoji，而你的LLM在这个数据上表现不够理想。所以理想情况下，你应该向LLM发送干净的数据。那么你可以做的是，在这里定义一个函数，它的任务就是进行预处理。在这里你可以定义各种预处理步骤，比如转换为小写字母、删除标点符号、执行词形还原，无论你在NLP中学到了什么，所有这些预处理都可以在这个函数中完成。

发生了什么，现在你做了什么，借助可运行 lambda 的帮助，将其转换为一个可运行对象。由于现在它已经转换为一个可运行对象，你可以直接将其输出连接到你的 LLM 可运行对象上，然后 LLM 可运行对象又与解析器相连。所以基本上，你的这个预处理部分已经成为整个工作流程的自动部分了。

因为现在这不是一个普通的Python函数，而是变成了可运行的。所以我希望你能明白，用简单的语言来说，借助可运行的lambda，你可以将任何Python函数转换为可运行的。一旦它被转换为可运行的，之后它就可以成为任何链的一部分。那么我们来举一个小例子，通过这个例子我们将学习可运行的lambda是如何工作的。假设我们拿同一个笑话。

让我们举个例子：我们向用户询问一个主题，然后根据该主题生成一个笑话。好的，但现在在打印笑话的时候，我不仅想打印笑话的内容，还想同时打印这个笑话中总共有多少个单词。而且我不想向LLM询问单词数量，因为通常LLM在这方面表现不太好。那么我要怎么做呢？我会创建一个流程，现在我会在屏幕上写出来。我们要做的是首先形成一个提示，这个提示我们会发给LLM。

我们将发送给LLM，它会生成自己的输出，我们将通过字符串输出解析器接收它。之后，我将创建一个并行链。在这个并行链中，一方面我会做一个直通操作，这样我的笑话就能原样打印在这里。而在这里，我将创建一个可运行的lambda，它将接收这个笑话，然后在这里计算这段文本中有多少个单词，然后在这里给你单词的数量。

他会告诉我，然后我会把这两个加起来打印结果，所以基本上这里有一部分是顺序链，这里有一部分是并行链，而在并行链内部，一边是直通，一边是可运行的lambda函数。所以我们现在就要做这个，在做之前，我先工作一下，我会很快地写下来，可运行的lambda函数是如何工作的。所以我要做的是从lang chain.schema.runnable导入可运行的lambda。

而我要做的是创建一个名为“word counter”的函数，它需要一个文本来完成它的工作，当它得到文本时，它会返回text.split，这样整个文本就会变成一个列表，然后我会计算这个列表的长度，以找出里面有多少个单词，对吧？所以这是一个普通的函数，对吧？现在我需要把它转换成可运行的代码。

所以我要做的是可运行的单词计数器，我正在创建一个变量，然后写一个可运行的lambda单词计数器。在这个步骤中，我已经将这个函数转换为一个可运行的对象。现在这个东西是一个可运行的，这意味着它内部有一个调用函数。在调用函数内部，我正在发送这个字符串，并且我正在运行这段代码。我没有打印任何东西，所以我现在打印一下。保存并运行，你可以看到答案是5。

我希望你现在已经理解了这一点，现在我们将创建一个应用程序，在那里我们会打印笑话，并且与笑话一起，我们还会打印出该笑话中有多少个单词。所以我会稍微复制一下这段代码，事实上我可以复制到这里的所有代码。在这里我已经导入了所有内容，在这里我还会导入一些其他东西，我会导入runnable lambda，我会导入runnable pass through。

我会运行并行导入，这是我的提示。目前我只需要一个提示，所以我将其命名为“prompt”。写一个关于主题模型和解析器的笑话。现在看这里，我将首先构建一个生成笑话的链式结构，这将通过可运行序列的帮助完成，其中我将发送三个东西：提示、模型。

解析器好了，现在我要创建我的并行链，所以会有并行链可运行并行，这里会有笑话，它将是一个可运行的直通，这里会有字数统计，它将是一个可运行的lambda，在这个可运行的lambda里面，你会发送一个函数，所以我在这里创建函数。

所以定义单词计数函数，获取文本并返回文本长度的分割点。然后单词计数，我在这里发送。好的，这是一种方法。它的第二种方法是你可以直接在这里发送lambda函数，像这样lambda x x点的分割长度。这段代码也会给出完全相同的结果。你明白吗？它的名字叫可运行lambda，因此被称为。

因为你可以在这里发送lambda函数，好的 现在我正在创建一个最终的链，它是由笑话生成链和并行链组成的 所以在可运行序列内部，我会传递笑话生成链和并行链以及最终的链，点调用主题中我会传递AI，然后我会打印无论结果是什么，保存了运行了

你可以看到这里有一个笑话，这里是字数统计。好的，现在你可以用更好的方式来展示它，比如你已经将结果存储在一个变量中，现在你在做什么呢？你在写“final result is equal to”，然后在这里你放上“/word count.format”，在这里你传入结果的“joke”。

而在这里，您正在传递结果的字数统计，然后打印最终结果保存并运行，您可以看到我们的内容正在正确显示。好的，我希望您已经完全理解了可运行lambda的使用方式，这是一个非常强大的工具。如果在任何工作流程中，您觉得需要添加一些自定义逻辑到链中。


如果您的邮件是一般查询，您可能希望您的聊天机器人直接回复它，那么您可以这样做：关于这封邮件，我们想要执行不同的操作，您可以设置这样的流程：首先，您会收到邮件，然后您需要创建一个提示，在其中您会说：“请分析这封邮件的内容，并将其归类到特定类别中，例如是投诉、一般查询还是退款请求。”然后您发送这个提示。


LLM会告诉你投诉是关于退款还是一般查询。现在你在这里设置一个可运行的分支，然后你可以创建多个免费分支：一个分支处理投诉，一个分支处理退款请求，一个分支处理一般查询，一个分支处理一般查询，具体取决于LLM。

是否将该电子邮件放入类别中，这些分支中的任何一个都会被触发，对吧？那么，在Natshell中，基本概念是这样的：如果你的基础中有任何条件逻辑，其中某些事情的发生依赖于其他事情，那么你就在那里使用可运行的分支。为了更好地理解这一点，我们将举一个小例子。我们将向用户请求一个主题，并在该主题的基础上，借助提示的帮助，让LLM生成一份报告。

我们会解析它并将其带到这里。现在，如果这份报告超过500字，我们会再次告诉LLM进行总结；如果报告少于500字，我们会原样打印出来。我希望你能理解。如果你的报告超过500字，我们会再次告诉LLM进行总结；如果报告少于500字，我们会再次告诉LLM。

我们会说直接按原样打印这个。我希望你能理解，那么让我们写一段代码看看它是如何工作的。所以在这里我已经创建了一个文件，所有的导入我都已经完成了。首先我需要创建一个提示，所以我们的提示模板就准备好了。这是第一个提示。除此之外，我们还需要另一个提示，当字数超过500的限制时触发。所以在这里我会提供它，并在这里我会把文本放入输入变量中。所以这两个提示就准备好了。我们选择了模型，即OpenAI的聊天模型，以及解析器。我正在处理字符串输出部分。我正在处理字符串输出部分。

