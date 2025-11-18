
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

# Quickstart

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
    </Accordion>
  </Tab>

  <Tab title="Use the Functional API">
    ## 1. Define tools and model

    In this example, we'll use the Claude Sonnet 4.5 model and define tools for addition, multiplication, and division.

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

    from langgraph.graph import add_messages
    from langchain.messages import (
        SystemMessage,
        HumanMessage,
        ToolCall,
    )
    from langchain_core.messages import BaseMessage
    from langgraph.func import entrypoint, task
    ```

    ## 2. Define model node

    The model node is used to call the LLM and decide whether to call a tool or not.

    <Tip>
      The [`@task`](https://reference.langchain.com/python/langgraph/func/#langgraph.func.task) decorator marks a function as a task that can be executed as part of the agent. Tasks can be called synchronously or asynchronously within your entrypoint function.
    </Tip>

    ```python  theme={null}
    @task
    def call_llm(messages: list[BaseMessage]):
        """LLM decides whether to call a tool or not"""
        return model_with_tools.invoke(
            [
                SystemMessage(
                    content="You are a helpful assistant tasked with performing arithmetic on a set of inputs."
                )
            ]
            + messages
        )
    ```

    ## 3. Define tool node

    The tool node is used to call the tools and return the results.

    ```python  theme={null}
    @task
    def call_tool(tool_call: ToolCall):
        """Performs the tool call"""
        tool = tools_by_name[tool_call["name"]]
        return tool.invoke(tool_call)

    ```

    ## 4. Define agent

    The agent is built using the [`@entrypoint`](https://reference.langchain.com/python/langgraph/func/#langgraph.func.entrypoint) function.

    <Note>
      In the Functional API, instead of defining nodes and edges explicitly, you write standard control flow logic (loops, conditionals) within a single function.
    </Note>

    ```python  theme={null}
    @entrypoint()
    def agent(messages: list[BaseMessage]):
        model_response = call_llm(messages).result()

        while True:
            if not model_response.tool_calls:
                break

            # Execute tools
            tool_result_futures = [
                call_tool(tool_call) for tool_call in model_response.tool_calls
            ]
            tool_results = [fut.result() for fut in tool_result_futures]
            messages = add_messages(messages, [model_response, *tool_results])
            model_response = call_llm(messages).result()

        messages = add_messages(messages, model_response)
        return messages

    # Invoke
    messages = [HumanMessage(content="Add 3 and 4.")]
    for chunk in agent.stream(messages, stream_mode="updates"):
        print(chunk)
        print("\n")
    ```

    <Tip>
      To learn how to trace your agent with LangSmith, see the [LangSmith documentation](/langsmith/trace-with-langgraph).
    </Tip>

    Congratulations! You've built your first agent using the LangGraph Functional API.

    <Accordion title="Full code example" icon="code">
      ```python  theme={null}
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

      from langgraph.graph import add_messages
      from langchain.messages import (
          SystemMessage,
          HumanMessage,
          ToolCall,
      )
      from langchain_core.messages import BaseMessage
      from langgraph.func import entrypoint, task


      # Step 2: Define model node

      @task
      def call_llm(messages: list[BaseMessage]):
          """LLM decides whether to call a tool or not"""
          return model_with_tools.invoke(
              [
                  SystemMessage(
                      content="You are a helpful assistant tasked with performing arithmetic on a set of inputs."
                  )
              ]
              + messages
          )


      # Step 3: Define tool node

      @task
      def call_tool(tool_call: ToolCall):
          """Performs the tool call"""
          tool = tools_by_name[tool_call["name"]]
          return tool.invoke(tool_call)


      # Step 4: Define agent

      @entrypoint()
      def agent(messages: list[BaseMessage]):
          model_response = call_llm(messages).result()

          while True:
              if not model_response.tool_calls:
                  break

              # Execute tools
              tool_result_futures = [
                  call_tool(tool_call) for tool_call in model_response.tool_calls
              ]
              tool_results = [fut.result() for fut in tool_result_futures]
              messages = add_messages(messages, [model_response, *tool_results])
              model_response = call_llm(messages).result()

          messages = add_messages(messages, model_response)
          return messages

      # Invoke
      messages = [HumanMessage(content="Add 3 and 4.")]
      for chunk in agent.stream(messages, stream_mode="updates"):
          print(chunk)
          print("\n")
      ```
    </Accordion>
  </Tab>
</Tabs>

***

<Callout icon="pen-to-square" iconType="regular">
  [Edit the source of this page on GitHub.](https://github.com/langchain-ai/docs/edit/main/src/oss/langgraph/quickstart.mdx)
</Callout>

<Tip icon="terminal" iconType="regular">
  [Connect these docs programmatically](/use-these-docs) to Claude, VSCode, and more via MCP for    real-time answers.
</Tip>
