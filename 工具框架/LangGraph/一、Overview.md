
LangGraph 是一个低级别的编排框架和运行时环境，专为构建、管理和部署长期运行且具有状态的智能体而设计。包括 Klarna、Replit、Elastic 等正在塑造智能体未来的公司都对它信赖有加。

LangGraph 是一个非常底层的框架，完全专注于智能体编排。在使用 LangGraph 之前，我们建议您先熟悉构建智能体所需的一些组件，从 *模型* 和 *工具* 开始。

在文档中，我们将频繁使用 LangChain 组件来集成模型和工具，但使用 LangGraph 并不强制要求使用 LangChain。如果你是智能体开发的新手，或希望获得更高层次的抽象框架，我们推荐您使用 LangChain 提供的 agents 架构——它已经为常见的 LLM 调用和工具循环场景预置了现成解决方案。

LangGraph 专注于对智能体编排至关重要的底层能力：持久执行、流式处理、人机交互等。

## <Icon icon="download" size={20} /> Install

hello world:

```python  theme={null}
from langgraph.graph import StateGraph, MessagesState, START, END

def mock_llm(state: MessagesState):
    return {"messages": [{"role": "ai", "content": "hello world"}]}

graph = StateGraph(MessagesState)
graph.add_node(mock_llm)
graph.add_edge(START, "mock_llm")
graph.add_edge("mock_llm", END)
graph = graph.compile()

graph.invoke({"messages": [{"role": "user", "content": "hi!"}]})
```

## 核心优势

LangGraph 为任何长时间运行的有状态工作流或代理提供底层支持基础设施。LangGraph 不会抽象提示或架构，并提供以下核心优势：

* 持久执行：构建能够经受故障并长期运行的代理，可以从中断处恢复继续执行。
* 人在回路：通过随时检查和修改代理状态，引入人工监督。
* 全面记忆：创建具备短期工作记忆（用于持续推理）和跨会话长期记忆的有状态代理。
* 使用 LangSmith 进行调试：通过可视化工具深入洞察复杂的代理行为，追踪执行路径、捕捉状态转换并提供详细的运行时指标。
* 生产就绪的部署：借助专为处理有状态、长时间运行工作流独特挑战而设计的可扩展基础设施，自信部署复杂的智能体系统。


## LangGraph 生态

虽然 LangGraph 可以单独使用，但它也能与任何 LangChain 产品无缝集成，为开发者提供构建智能代理的完整工具套件。为了提升您的 LLM 应用开发效率，建议将 LangGraph 与以下工具搭配使用：

* LangSmith — 有助于代理评估和可观测性。调试性能不佳的 LLM 应用运行，评估代理轨迹，在生产中获得可见性，并随时间推移提升性能。
* LangGraph —— 通过专为长时间运行的有状态工作流设计的部署平台，轻松部署和扩展智能代理。跨团队发现、复用、配置和共享代理 —— 并利用 Studio 中的可视化原型设计快速迭代。
* LangChain - 提供集成和可组合组件，以简化 LLM 应用程序开发。包含基于 LangGraph 构建的代理抽象。

# 二、Quickstart

本快速入门指南展示了如何使用 LangGraph 图 API 或函数式 API 构建一个计算器代理。

* [Use the Graph API](#use-the-graph-api) 如果您更倾向于将您的代理定义为节点和边的图。
* [Use the Functional API](#use-the-functional-api) 如果你更倾向于将你的代理定义为一个单一函数。

对于这个示例，你需要创建一个 Claude（Anthropic）账户并获取 API 密钥。然后，在你的终端中设置 ANTHROPIC_API_KEY 环境变量。

----------
（以下内容使用 Graph API）

## 1. 定义工具和模型

在本示例中，我们将使用 Claude Sonnet 4.5 模型，并定义加法、乘法和除法的工具。

```python  theme={null}
    from langchain.tools import tool
    from langchain.chat_models import init_chat_model


    model = init_chat_model(
        "claude-sonnet-4-5-20250929",
        temperature=0
    )


    # Define tools
    @tool
    def multiply(a: int, b: int) -> int:
        """Multiply `a` and `b`.

        Args:
            a: First int
            b: Second int
        """
        return a * b


    @tool
    def add(a: int, b: int) -> int:
        """Adds `a` and `b`.

        Args:
            a: First int
            b: Second int
        """
        return a + b


    @tool
    def divide(a: int, b: int) -> float:
        """Divide `a` and `b`.

        Args:
            a: First int
            b: Second int
        """
        return a / b


    # Augment the LLM with tools
    tools = [add, multiply, divide]
    tools_by_name = {tool.name: tool for tool in tools}
    model_with_tools = model.bind_tools(tools)
    ```

## 2. 定义状态

该图的状态用于存储消息和 LLM 调用的次数。状态在 LangGraph 中贯穿代理的整个执行过程。带有 `operator.add` 的 `Annotated` 类型确保新消息会被追加到现有列表中，而不是替换它。

```python  theme={null}
    from langchain.messages import AnyMessage
    from typing_extensions import TypedDict, Annotated
    import operator


    class MessagesState(TypedDict):
        messages: Annotated[list[AnyMessage], operator.add]
        llm_calls: int
    ```

## 3. 定义模型节点

模型节点用于调用 LLM 并决定是否调用工具。

```python  theme={null}
    from langchain.messages import SystemMessage


    def llm_call(state: dict):
        """LLM decides whether to call a tool or not"""

        return {
            "messages": [
                model_with_tools.invoke(
                    [
                        SystemMessage(
                            content="You are a helpful assistant tasked with performing arithmetic on a set of inputs."
                        )
                    ]
                    + state["messages"]
                )
            ],
            "llm_calls": state.get('llm_calls', 0) + 1
        }
    ```

## 4. 定义工具节点

工具节点用于调用工具并返回结果。

```python  theme={null}
    from langchain.messages import ToolMessage


    def tool_node(state: dict):
        """Performs the tool call"""

        result = []
        for tool_call in state["messages"][-1].tool_calls:
            tool = tools_by_name[tool_call["name"]]
            observation = tool.invoke(tool_call["args"])
            result.append(ToolMessage(content=observation, tool_call_id=tool_call["id"]))
        return {"messages": result}
    ```

## 5. 定义结束逻辑

条件边缘函数用于根据 LLM 是否调用了工具来路由到工具节点或结束。

```python  theme={null}
    from typing import Literal
    from langgraph.graph import StateGraph, START, END


    def should_continue(state: MessagesState) -> Literal["tool_node", END]:
        """Decide if we should continue the loop or stop based upon whether the LLM made a tool call"""

        messages = state["messages"]
        last_message = messages[-1]

        # If the LLM makes a tool call, then perform an action
        if last_message.tool_calls:
            return "tool_node"

        # Otherwise, we stop (reply to the user)
        return END
    ```
 
## 6. 构建并编译代理

该代理是使用 `StateGraph` 类构建的，并通过 `compile` 方法进行编译。

```python  theme={null}
    # Build workflow
    agent_builder = StateGraph(MessagesState)

    # Add nodes
    agent_builder.add_node("llm_call", llm_call)
    agent_builder.add_node("tool_node", tool_node)

    # Add edges to connect nodes
    agent_builder.add_edge(START, "llm_call")
    agent_builder.add_conditional_edges(
        "llm_call",
        should_continue,
        ["tool_node", END]
    )
    agent_builder.add_edge("tool_node", "llm_call")

    # Compile the agent
    agent = agent_builder.compile()

    # Show the agent
    from IPython.display import Image, display
    display(Image(agent.get_graph(xray=True).draw_mermaid_png()))

    # Invoke
    from langchain.messages import HumanMessage
    messages = [HumanMessage(content="Add 3 and 4.")]
    messages = agent.invoke({"messages": messages})
    for m in messages["messages"]:
        m.pretty_print()
    ```

恭喜！您已使用 LangGraph 图 API 构建了您的第一个代理。

```python
      # Step 1: Define tools and model

      from langchain.tools import tool
      from langchain.chat_models import init_chat_model


      model = init_chat_model(
          "claude-sonnet-4-5-20250929",
          temperature=0
      )


      # Define tools
      @tool
      def multiply(a: int, b: int) -> int:
          """Multiply `a` and `b`.

          Args:
              a: First int
              b: Second int
          """
          return a * b


      @tool
      def add(a: int, b: int) -> int:
          """Adds `a` and `b`.

          Args:
              a: First int
              b: Second int
          """
          return a + b


      @tool
      def divide(a: int, b: int) -> float:
          """Divide `a` and `b`.

          Args:
              a: First int
              b: Second int
          """
          return a / b


      # Augment the LLM with tools
      tools = [add, multiply, divide]
      tools_by_name = {tool.name: tool for tool in tools}
      model_with_tools = model.bind_tools(tools)

      # Step 2: Define state

      from langchain.messages import AnyMessage
      from typing_extensions import TypedDict, Annotated
      import operator


      class MessagesState(TypedDict):
          messages: Annotated[list[AnyMessage], operator.add]
          llm_calls: int

      # Step 3: Define model node
      from langchain.messages import SystemMessage


      def llm_call(state: dict):
          """LLM decides whether to call a tool or not"""

          return {
              "messages": [
                  model_with_tools.invoke(
                      [
                          SystemMessage(
                              content="You are a helpful assistant tasked with performing arithmetic on a set of inputs."
                          )
                      ]
                      + state["messages"]
                  )
              ],
              "llm_calls": state.get('llm_calls', 0) + 1
          }


      # Step 4: Define tool node

      from langchain.messages import ToolMessage


      def tool_node(state: dict):
          """Performs the tool call"""

          result = []
          for tool_call in state["messages"][-1].tool_calls:
              tool = tools_by_name[tool_call["name"]]
              observation = tool.invoke(tool_call["args"])
              result.append(ToolMessage(content=observation, tool_call_id=tool_call["id"]))
          return {"messages": result}

      # Step 5: Define logic to determine whether to end

      from typing import Literal
      from langgraph.graph import StateGraph, START, END


      # Conditional edge function to route to the tool node or end based upon whether the LLM made a tool call
      def should_continue(state: MessagesState) -> Literal["tool_node", END]:
          """Decide if we should continue the loop or stop based upon whether the LLM made a tool call"""

          messages = state["messages"]
          last_message = messages[-1]

          # If the LLM makes a tool call, then perform an action
          if last_message.tool_calls:
              return "tool_node"

          # Otherwise, we stop (reply to the user)
          return END

      # Step 6: Build agent

      # Build workflow
      agent_builder = StateGraph(MessagesState)

      # Add nodes
      agent_builder.add_node("llm_call", llm_call)
      agent_builder.add_node("tool_node", tool_node)

      # Add edges to connect nodes
      agent_builder.add_edge(START, "llm_call")
      agent_builder.add_conditional_edges(
          "llm_call",
          should_continue,
          ["tool_node", END]
      )
      agent_builder.add_edge("tool_node", "llm_call")

      # Compile the agent
      agent = agent_builder.compile()


      from IPython.display import Image, display
      # Show the agent
      display(Image(agent.get_graph(xray=True).draw_mermaid_png()))

      # Invoke
      from langchain.messages import HumanMessage
      messages = [HumanMessage(content="Add 3 and 4.")]
      messages = agent.invoke({"messages": messages})
      for m in messages["messages"]:
          m.pretty_print()

      ```


--------------

# 三、Run a local server

本指南将向您展示如何在本地运行 LangGraph 应用程序。

## 先决条件

在开始之前，请确保您具备以下条件：

* LangSmith 的 API 密钥 - 免费注册

## 1. 安装 LangGraph CLI

```bash pip theme={null}
  # Python >= 3.11 is required.
  pip install -U "langgraph-cli[inmem]"
```

  ```bash uv theme={null}
  # Python >= 3.11 is required.
  uv add 'langgraph-cli[inmem]'
  ```

## 2. 创建 LangGraph app

从 [`new-langgraph-project-python` template](https://github.com/langchain-ai/new-langgraph-project) 模板创建一个新应用。该模板展示了一个单节点应用程序，你可以用自己的逻辑进行扩展。

```shell
langgraph new path/to/your/app --template new-langgraph-project-python
```

> **其他模板**​ 如果你使用 `langgraph new` 而不指定模板，将会出现一个交互式菜单，让你从可用模板列表中选择。

## 3. 安装依赖项

在你的新 LangGraph 应用的根目录中，以编辑模式安装依赖项，以便服务器使用你的本地更改：


In the root of your new LangGraph app, install the dependencies in `edit` mode so your local changes are used by the server:

<CodeGroup>
  ```bash pip theme={null}
  cd path/to/your/app
  pip install -e .
  ```

  ```bash uv theme={null}
  cd path/to/your/app
  uv sync
  ```
</CodeGroup>

## 4. Create a `.env` file

You will find a `.env.example` in the root of your new LangGraph app. Create a `.env` file in the root of your new LangGraph app and copy the contents of the `.env.example` file into it, filling in the necessary API keys:

```bash  theme={null}
LANGSMITH_API_KEY=lsv2...
```

## 5. Launch Agent server

Start the LangGraph API server locally:

```shell  theme={null}
langgraph dev
```

Sample output:

```
INFO:langgraph_api.cli:

        Welcome to

╦  ┌─┐┌┐┌┌─┐╔═╗┬─┐┌─┐┌─┐┬ ┬
║  ├─┤││││ ┬║ ╦├┬┘├─┤├─┘├─┤
╩═╝┴ ┴┘└┘└─┘╚═╝┴└─┴ ┴┴  ┴ ┴

- 🚀 API: http://127.0.0.1:2024
- 🎨 Studio UI: https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024
- 📚 API Docs: http://127.0.0.1:2024/docs

This in-memory server is designed for development and testing.
For production use, please use LangSmith Deployment.
```

The `langgraph dev` command starts Agent Server in an in-memory mode. This mode is suitable for development and testing purposes. For production use, deploy Agent Server with access to a persistent storage backend. For more information, see the [Platform setup overview](/langsmith/platform-setup).

## 6. Test your application in Studio

[Studio](/langsmith/studio) is a specialized UI that you can connect to LangGraph API server to visualize, interact with, and debug your application locally. Test your graph in Studio by visiting the URL provided in the output of the `langgraph dev` command:

```
>    - LangGraph Studio Web UI: https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024
```

For an Agent Server running on a custom host/port, update the `baseUrl` query parameter in the URL. For example, if your server is running on `http://myhost:3000`:

```
https://smith.langchain.com/studio/?baseUrl=http://myhost:3000
```

<Accordion title="Safari compatibility">
  Use the `--tunnel` flag with your command to create a secure tunnel, as Safari has limitations when connecting to localhost servers:

  ```shell  theme={null}
  langgraph dev --tunnel
  ```
</Accordion>

## 7. Test the API

<Tabs>
  <Tab title="Python SDK (async)">
    1. Install the LangGraph Python SDK:
       ```shell  theme={null}
       pip install langgraph-sdk
       ```
    2. Send a message to the assistant (threadless run):
       ```python  theme={null}
       from langgraph_sdk import get_client
       import asyncio

       client = get_client(url="http://localhost:2024")

       async def main():
           async for chunk in client.runs.stream(
               None,  # Threadless run
               "agent", # Name of assistant. Defined in langgraph.json.
               input={
               "messages": [{
                   "role": "human",
                   "content": "What is LangGraph?",
                   }],
               },
           ):
               print(f"Receiving new event of type: {chunk.event}...")
               print(chunk.data)
               print("\n\n")

       asyncio.run(main())
       ```
  </Tab>

  <Tab title="Python SDK (sync)">
    1. Install the LangGraph Python SDK:
       ```shell  theme={null}
       pip install langgraph-sdk
       ```
    2. Send a message to the assistant (threadless run):
       ```python  theme={null}
       from langgraph_sdk import get_sync_client

       client = get_sync_client(url="http://localhost:2024")

       for chunk in client.runs.stream(
           None,  # Threadless run
           "agent", # Name of assistant. Defined in langgraph.json.
           input={
               "messages": [{
                   "role": "human",
                   "content": "What is LangGraph?",
               }],
           },
           stream_mode="messages-tuple",
       ):
           print(f"Receiving new event of type: {chunk.event}...")
           print(chunk.data)
           print("\n\n")
       ```
  </Tab>

  <Tab title="Rest API">
    ```bash  theme={null}
    curl -s --request POST \
        --url "http://localhost:2024/runs/stream" \
        --header 'Content-Type: application/json' \
        --data "{
            \"assistant_id\": \"agent\",
            \"input\": {
                \"messages\": [
                    {
                        \"role\": \"human\",
                        \"content\": \"What is LangGraph?\"
                    }
                ]
            },
            \"stream_mode\": \"messages-tuple\"
        }"
    ```
  </Tab>
</Tabs>

## Next steps

Now that you have a LangGraph app running locally, take your journey further by exploring deployment and advanced features:

* [Deployment quickstart](/langsmith/deployment-quickstart): Deploy your LangGraph app using LangSmith.

* [LangSmith](/langsmith/home): Learn about foundational LangSmith concepts.

* [SDK Reference](https://reference.langchain.com/python/langsmith/deployment/sdk/): Explore the SDK API Reference.

***

<Callout icon="pen-to-square" iconType="regular">
  [Edit this page on GitHub](https://github.com/langchain-ai/docs/edit/main/src/oss/langgraph/local-server.mdx) or [file an issue](https://github.com/langchain-ai/docs/issues/new/choose).
</Callout>

<Tip icon="terminal" iconType="regular">
  [Connect these docs](/use-these-docs) to Claude, VSCode, and more via MCP for real-time answers.
</Tip>


---

> To find navigation and other pages in this documentation, fetch the llms.txt file at: https://docs.langchain.com/llms.txt





我的目标是以非常有条理的方式把事情做好。我们首先在这个频道学习了机器学习，然后学习了深度学习，接着又开始了Lang Chain等，还涉足了生成式AI。到了这个阶段，我个人觉得我们已经学得足够多，差不多准备好学习和理解Lang Graph以及如何构建AI代理了，这就是第三个原因。接下来，我想谈谈我启动这个播放列表背后的愿景。无论你做什么事情，背后都应该有一个强大的愿景。

如果可能的话，我想与您分享我的愿景，即通过这个播放列表我希望实现什么目标。如果我坦白告诉您，当Landgraff进入市场并逐渐从您的网站收到消息说“先生，请教授Landgraff”时，我做的第一件事就是上YouTube搜索目前有哪些关于Landgraff的现有内容可用，而我注意到有两种类型的内容。

在YouTube上，第一种内容是通过使用Derick Le Landgraff来教授如何创建项目的。这是一种类型的内容。然后还有第二种类型的内容，主要是教授Landgraff非常基础的基础知识。在这两种内容中，我发现了一个缺陷：在教授创建项目的地方，基础知识没有得到充分讨论；而在专注于基础知识的地方，视频又太短了。


