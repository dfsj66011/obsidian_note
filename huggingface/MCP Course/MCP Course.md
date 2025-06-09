
| 章节￼ | 主题￼            | 描述￼                                            |
| --- | -------------- | ---------------------------------------------- |
| 0   | 入门准备￼          | 为您配置将要使用的工具和平台                                 |
| 1   | MCP 基础、架构与核心概念 | 解释模型上下文协议（MCP）的核心概念、架构及组件。通过一个简单用例展示 MCP 的应用。  |
| 2   | 端到端用例：MCP 实战   | 构建一个简单的端到端 MCP 应用程序，可与社区共享。                    |
| 3   | 部署用例：MCP 实战    | 利用 Hugging Face 生态系统及合作伙伴服务，构建一个已部署的 MCP 应用程序。 |
| 4   | 附加单元           | 额外单元助您从课程中获得更多，包括与合作伙伴的库和服务协作。￼￼               |

## 模型上下文协议（MCP）简介

欢迎来到 MCP 课程的第一单元！在本单元中，我们将探讨模型上下文协议的基础知识。

**你将学到什么**

在本单元中，你将：

- 了解什么是模型上下文协议（MCP）及其重要性
- 学习与 MCP 相关的核心概念和术语
- 探索 MCP 解决的集成挑战
- 了解 MCP 的主要优势与目标
- 通过实际案例观察 MCP 集成的简单示例

完成本单元后，你将扎实掌握 MCP 的基础概念，并准备好进入下一单元深入学习其架构与实现。

**MCP 的重要性**

人工智能生态系统正在迅速发展，大型语言模型（LLMs）和其他人工智能系统变得越来越强大。然而，这些模型往往受限于其训练数据，无法获取实时信息或专业工具。这一限制阻碍了人工智能系统在许多场景中提供真正相关、准确和有用回答的潜力。

这就是模型上下文协议（MCP）的用武之地。MCP 使 AI 模型能够连接外部数据源、工具和环境，实现 AI 系统与更广泛的数字世界之间信息和能力的无缝传输。这种互操作性对于真正有用的 AI 应用程序的发展和采用至关重要。

**第一单元概述**

以下是本单元内容概览：

1. 什么是模型上下文协议？——我们将首先定义 MCP 并探讨其在 AI 生态系统中的作用。
2. 核心概念——深入解析与 MCP 相关的基础概念和术语。
3. 集成挑战——剖析 MCP 旨在解决的问题，特别是 "M×N 集成难题"。
4. 优势与目标——讨论 MCP 的关键优势与目标，包括标准化、增强 AI 能力和互操作性。
5. 简单示例——最后通过一个 MCP 集成实例演示其实际应用。

现在，让我们共同探索模型上下文协议的精彩世界！

-----

## 关键概念与术语

在深入探讨模型上下文协议（MCP）之前，理解构成 MCP 基础的关键概念和术语非常重要。本节将介绍支撑该协议的基本理念，并为整个课程中讨论 MCP 实现提供通用词汇。

MCP 常被称作 “AI 应用的 USB-C”。正如 USB-C 为连接各种外设与计算设备提供了标准化的物理和逻辑接口，MCP 则为 AI 模型与外部能力对接提供了一致的协议。这种标准化让整个生态系统受益：

- 用户可以在各种 AI 应用中享受更简单、更一致的体验
- AI 应用开发者可以轻松集成不断增长的工具和数据源生态系统
- 工具和数据提供商只需创建一次实现，即可适配多个 AI 应用
- 整个生态系统将从增强的互操作性、创新力提升和减少碎片化中受益

**集成问题**

M×N 集成问题指的是在没有标准化方法的情况下，将 M 个不同的人工智能应用程序连接到 N 个不同的外部工具或数据源所面临的挑战。

**无 MCP（M×N 问题）**

如果没有像 MCP 这样的协议，开发者将需要创建 M×N 个定制集成——即每个 AI 应用程序与外部功能的每一种可能配对都需要一个集成。

![Without MCP|500](https://huggingface.co/datasets/mcp-course/images/resolve/main/unit1/1.png)

每个 AI 应用都需要与每个工具/数据源单独集成。这是一个非常复杂且昂贵的过程，给开发者带来了很多阻碍，并导致高昂的维护成本。

一旦我们拥有多个模型和多个工具，集成的数量就会变得难以管理，每个集成都有其独特的接口。

![Multiple Models and Tools|500](https://huggingface.co/datasets/mcp-course/images/resolve/main/unit1/1a.png)

**采用 MCP（M+N 解决方案）**

MCP 通过提供一个标准接口，将这一问题转化为 M+N 问题：每个 AI 应用只需实现一次 MCP 的客户端，每个工具/数据源只需实现一次服务器端。这极大地降低了集成复杂性和维护负担。

![With MCP|500](https://huggingface.co/datasets/mcp-course/images/resolve/main/unit1/2.png)

每个 AI 应用只需实现一次 MCP 的客户端，而每个工具/数据源也只需实现一次 MCP 的服务端。

### 核心 MCP 术语 

既然我们已经了解了 MCP 解决的问题，接下来让我们深入探讨构成 MCP 协议的核心术语和概念。

MCP 是一种类似于 HTTP 或 USB-C 的标准协议，旨在将人工智能应用与外部工具及数据源相连接。因此，使用标准术语对于确保 MCP 有效运作至关重要。

在记录我们的应用程序和与社区交流时，应使用以下术语。

**组件**

就像 HTTP 中的客户端服务器关系一样，MCP 也有客户端和服务器。

![MCP Components|500](https://huggingface.co/datasets/mcp-course/images/resolve/main/unit1/3.png)

* **Host：** 终端用户直接与之交互的面向用户的 AI 应用程序。示例包括 Anthropic 的 Claude 桌面应用、Cursor 等 AI 增强型集成开发环境（IDE）、Hugging Face Python SDK 等推理库，或基于LangChain 或 smolagents 等库构建的自定义应用程序。Host 负责启动与 MCP 服务器的连接，并协调用户请求、LLM 处理和外部工具之间的整体流程。
* **客户端：** 宿主应用程序中的一个组件，负责管理与特定 MCP 服务器的通信。每个客户端与单个服务器保持 1:1 的连接关系，处理 MCP 通信的协议级细节，并充当宿主逻辑与外部服务器之间的中介。
* **服务端：** 通过 MCP 协议对外提供能力（工具、资源、提示）的外部程序或服务。

很多内容将“客户端”和“宿主应用”混为一谈。严格来说，宿主应用是面向用户的应用程序，而客户端是宿主应用中负责与特定 MCP服 务器通信的组件。

**能力**

当然，应用的价值在于其所提供的功能总和。因此，功能是您应用中最重要的部分。MCP 可以与任何软件服务连接，但有一些常见功能被许多 AI 应用所使用。




| 功能￼￼    | 描述￼                                                            | 示例                         |
| ------- | -------------------------------------------------------------- | -------------------------- |
| **工具**  | AI 模型可以调用的可执行功能，用于执行操作或检索计算数据。通常与应用程序的用例相关。                    | 天气应用程序的工具可能是一个返回特定位置天气的函数。 |
| **资源**  | 只读数据源，提供上下文而无需大量计算。                                            | 研究助手可能有一个科学论文的资源。          |
| **提示**  | 预定义的模板或工作流程，指导用户、AI 模型和可用功能之间的交互。                              | 摘要提示。                      |
| **采样￼** | ￼￼￼￼服务器发起的请求，要求客户端/主机执行 LLM 交互，从而实现递归操作，LLM 可以审查生成的内容并做出进一步决策。 | 写作应用程序审查自己的输出并决定进一步改进。￼￼   |

在下图中，我们可以看到应用于代码代理用例的集成能力。

![collective diagram|500](https://huggingface.co/datasets/mcp-course/images/resolve/main/unit1/8.png)

该应用程序可能会以以下方式使用其 MCP 实体：

| 实体  | 名称    | 描述￼                         |
| --- | ----- | --------------------------- |
| 工具￼ | 代码解释器 | 一种可以执行 LLM 编写代码的工具。         |
| 资源  | 文档    | ￼￼￼￼包含应用程序文档的资源。            |
| 提示  | 代码风格  | 指导 LLM 生成代码的提示。             |
| 采样  | 代码审查  | ￼￼允许 LLM 审查代码并做出进一步决策的采样。￼￼ |

理解这些关键概念和术语为有效使用 MCP 奠定了基础。在接下来的章节中，我们将基于此基础，探索构成模型上下文协议的架构组件、通信协议及其功能。


-----

## MCP 的结构组件

在上一节中，我们讨论了 MCP 的核心概念和术语。现在，让我们更深入地探讨构成 MCP 生态系统的架构组件。

**主机、客户端和服务器**

模型上下文协议（MCP）基于客户端-服务器架构构建，可实现 AI 模型与外部系统之间的结构化通信。

![MCP Architecture|500](https://huggingface.co/datasets/mcp-course/images/resolve/main/unit1/4.png)

MCP 架构由三个主要组件构成，每个组件都有明确的职责分工：主机（Host）、客户端（Client）和服务器（Server）。我们在前文已简要提及这些组件，现在让我们深入剖析每个组件的具体职责。

**Host**

Host 是面向用户的 AI 应用程序，终端用户可以直接与之交互。

例如：

- AI聊天应用，如 OpenAI 的 ChatGPT 或 Anthropic 的 Claude 桌面版
- 增强 AI 功能的集成开发环境（IDE），如 Cursor，或与 Continue.dev 等工具的集成
- 使用 LangChain 或 smolagents 等库构建的自定义 AI 代理和应用

Host 的职责包括：

- 管理用户互动和权限
- 通过 MCP 客户端启动与 MCP 服务器的连接
- 协调用户请求、LLM 处理和外部工具之间的整体流程
- 以连贯的格式将结果呈现给用户

在大多数情况下，用户会根据自身需求和偏好选择 host 应用。例如，开发者可能因其强大的代码编辑功能而选择 Cursor，而领域专家则可能使用基于 smolagents 构建的定制应用。

**Client**

客户端是 host 应用程序中的一个组件，负责管理与特定 MCP 服务器的通信。其主要特点包括：

- 每个客户端与单个服务器保持 1:1 连接
- 处理 MCP 通信的协议级细节
- 充当 host 逻辑与外部服务器之间的中介

**Server**

服务端是一个外部程序或服务，通过 MCP 协议向 AI 模型提供功能。服务端：

- 提供对特定外部工具、数据源或服务的访问
- 作为现有功能的轻量级封装
- 可在本地（与主机同一台机器）或远程（通过网络）运行
- 以标准化格式公开其功能，供客户端发现和使用

**沟通流程**

让我们来看看这些组件在典型的 MCP 工作流程中是如何相互作用的：

在下一节中，我们将通过实际示例深入探讨使这些组件得以运行的通信协议。

1. 用户交互：用户与 host 应用程序互动，表达意图或查询。
2. host 处理：host 处理用户的输入，可能使用 LLM 来理解请求并确定可能需要哪些外部能力。
3. 客户端连接：host 指示其客户端组件连接到适当的服务器。
4. 能力发现：客户端查询服务器以发现其提供的能力（工具、资源、提示）。
5. 能力调用：根据用户需求或 LLM 的判定，host 指示客户端从服务器调用特定能力。
6. 服务器执行：服务器执行请求的功能并将结果返回给客户端。
7. 结果整合：客户端将这些结果传回 host，host 将其整合到 LLM 的上下文中或直接呈现给用户。

这种架构的一个关键优势在于其模块化。一个 host 可以通过不同的客户端同时连接到多个服务器。新的服务器可以轻松添加到生态系统中，而无需对现有 host 进行任何更改。不同服务器之间的功能可以轻松组合。

正如我们在前一节中所讨论的，这种模块化设计将传统的 M×N 集成问题（M 个 AI 应用程序连接到 N 个工具/服务）转化为更易管理的 M+N 问题，其中每个 host 和服务器只需实现一次 MCP 标准。

该架构看似简单，但其强大之处在于通信协议的标准化以及组件间职责的清晰划分。这种设计形成了一个紧密的生态系统，使得人工智能模型能够与日益丰富的外部工具和数据源实现无缝对接。

**结论**

这些交互模式遵循几个关键原则，这些原则塑造了 MCP 的设计和演进。该协议通过提供统一的 AI 连接协议来强调标准化，同时通过保持核心协议的简洁性（同时支持高级功能）来维持简单性。安全性通过要求用户对敏感操作进行明确授权得到优先保障，而可发现性则实现了能力的动态发现。该协议在设计时考虑了可扩展性，支持通过版本控制和能力协商进行演进，并确保在不同实现和环境中的互操作性。

在下一部分，我们将探讨使这些组件有效协作的通信协议。

-------

## 通信协议

MCP 定义了一种标准化的通信协议，使客户端和服务器能够以一致、可预测的方式交换消息。这种标准化对于实现社区间的互操作性至关重要。在本节中，我们将探讨 MCP 中使用的协议结构和传输机制。

我们正在深入研究 MCP 协议的细节。虽然使用 MCP 进行开发时不需要了解所有这些内容，但了解它的存在及其工作原理是很有帮助的。

**JSON-RPC：基础**

MCP 的核心在于使用 JSON-RPC 2.0 作为客户端与服务器之间所有通信的消息格式。JSON-RPC 是一种基于 JSON 编码的轻量级远程过程调用协议，这使得它：

- 易于人类阅读和调试
- 与语言无关，支持在任何编程环境中实现
- 成熟稳定，具有明确的规范和广泛的采用
![message types|500](https://huggingface.co/datasets/mcp-course/images/resolve/main/unit1/5.png)
该协议定义了三种类型的消息：

1. **请求**

从客户端发送到服务器以启动操作。请求消息包括：

- 一个唯一标识符（`id`）
- 要调用的方法名称（例如，`tools/call`）
- 方法的参数（如果有）

示例请求：

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "tools/call",
  "params": {
    "name": "weather",
    "arguments": {
      "location": "San Francisco"
    }
  }
}
```

2. **回复**

从服务器发送到客户端以回复请求。响应消息包括：  
​
* 与对应请求相同的 `id`  
* 要么是 `result`（表示成功），要么是 `error`（表示失败）

示例成功响应：

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "temperature": 62,
    "conditions": "Partly cloudy"
  }
}
```

错误响应示例：

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "error": {
    "code": -32602,
    "message": "Invalid location parameter"
  }
}
```

3. **通知(Notifications)**

不需要回复的单向消息。通常由服务器发送给客户端，用于提供事件更新或通知。

示例通知：

```json
{
  "jsonrpc": "2.0",
  "method": "progress",
  "params": {
    "message": "Processing data...",
    "percent": 50
  }
}
```

**传输机制**

JSON-RPC 定义了消息格式，但 MCP 还规定了这些消息在客户端和服务器之间如何传输。支持两种主要的传输机制：

* stdio（标准输入/输出）：
	* stdio 传输用于本地通信，客户端和服务器在同一台机器上运行：
		* 宿主应用程序将服务器作为子进程启动，并通过写入其标准输入(stdin)和读取其标准输出(stdout)与之通信。
		* 这种传输方式的用例包括本地工具，如文件系统访问或运行本地脚本。
		* 这种传输方式的主要优点是操作简单，无需网络配置，并且由操作系统安全地隔离运行。
* HTTP + SSE（服务器发送事件）/可流式 HTTP：
	* HTTP+SSE 传输用于远程通信，客户端和服务器可能位于不同的机器上：
		* 通信通过 HTTP 进行，服务器使用服务器发送事件（SSE）通过持久连接向客户端推送更新。
		* 该传输的用例包括连接远程 API、云服务或共享资源。
		* 这种运输方式的主要优势在于它可以在网络间运行，能够与网络服务集成，并且兼容无服务器环境。

MCP 标准的最新更新引入并完善了“可流式 HTTP”，该功能通过允许服务器在需要时动态升级至 SSE （服务器发送事件）以实现流式传输，从而提供了更大的灵活性，同时保持与无服务器环境的兼容性。

**交互生命周期**

在上一节中，我们讨论了客户端（💻）与服务器（🌐）之间单次交互的生命周期。现在，让我们来看看在 MCP 协议背景下，客户端与服务器之间完整交互的生命周期。

**初始化**

客户端连接到服务器，双方交换协议版本和功能支持情况，服务器随后返回其支持的协议版本和功能。

|     |                    |     |
| --- | ------------------ | --- |
| 💻  | →  <br>initialize  | 🌐  |
| 💻  | ←  <br>response    | 🌐  |
| 💻  | →  <br>initialized | 🌐  |

客户通过通知消息确认初始化完成。

**探索**  

客户请求了解可用能力的信息，服务器则回应以可用工具的列表。

|   |   |   |
|---|---|---|
|💻|→  <br>tools/list|🌐|
|💻|←  <br>response|🌐|

这个过程可以针对每个工具、资源或提示类型重复进行。

**执行**

客户端根据 host 的需求调用功能。

|   |   |   |
|---|---|---|
|💻|→  <br>tools/call|🌐|
|💻|←  <br>notification (optional progress)|🌐|
|💻|←  <br>response|🌐|

**终止**

当不再需要时，连接会优雅地关闭，服务器会确认关闭请求。

|   |   |   |
|---|---|---|
|💻|→  <br>shutdown|🌐|
|💻|←  <br>response|🌐|
|💻|→  <br>exit|🌐|

客户端发送最终退出消息以完成终止。

**协议演进**

MCP 协议设计具有可扩展性和适应性。初始化阶段包含版本协商机制，确保协议演进时保持向后兼容性。此外，能力发现功能使客户端能够适配不同服务器提供的特性，从而实现在同一生态系统中兼容基础版和高级版服务器。

----

## 了解 MCP 功能

MCP 服务器通过通信协议向客户端提供多种功能。这些功能主要分为四大类，每类都具有独特的特性和应用场景。让我们深入探讨构成 MCP 功能基础的这些核心要素。

在本节中，我们将以语言无关的函数形式展示示例代码。这样做的目的是聚焦于核心概念及其协作方式，而非特定框架的复杂性。

在接下来的单元中，我们将展示这些概念如何在 MCP 特定代码中实现。

**工具**

工具是 AI 模型可以通过 MCP 协议调用的可执行函数或操作。

* 控制：工具通常由模型控制，即 AI 模型（LLM）根据用户请求和上下文决定何时调用它们。  
* 安全性：由于这些工具能够执行具有副作用的行为，其运行可能存在风险。因此，通常需要用户明确批准。  
* 使用场景：发送消息、创建工单、查询 API、执行计算。


**Example**: A weather tool that fetches current weather data for a given location:

python

javascript

Copied

def get_weather(location: str) -> dict:
    """Get the current weather for a specified location."""
    # Connect to weather API and fetch data
    return {
        "temperature": 72,
        "conditions": "Sunny",
        "humidity": 45
    }

## [](https://huggingface.co/learn/mcp-course/unit1/capabilities#resources)Resources

Resources provide read-only access to data sources, allowing the AI model to retrieve context without executing complex logic.

- **Control**: Resources are **application-controlled**, meaning the Host application typically decides when to access them.
- **Nature**: They are designed for data retrieval with minimal computation, similar to GET endpoints in REST APIs.
- **Safety**: Since they are read-only, they typically present lower security risks than Tools.
- **Use Cases**: Accessing file contents, retrieving database records, reading configuration information.

**Example**: A resource that provides access to file contents:

python

javascript

Copied

def read_file(file_path: str) -> str:
    """Read the contents of a file at the specified path."""
    with open(file_path, 'r') as f:
        return f.read()

## [](https://huggingface.co/learn/mcp-course/unit1/capabilities#prompts)Prompts

Prompts are predefined templates or workflows that guide the interaction between the user, the AI model, and the Server’s capabilities.

- **Control**: Prompts are **user-controlled**, often presented as options in the Host application’s UI.
- **Purpose**: They structure interactions for optimal use of available Tools and Resources.
- **Selection**: Users typically select a prompt before the AI model begins processing, setting context for the interaction.
- **Use Cases**: Common workflows, specialized task templates, guided interactions.

**Example**: A prompt template for generating a code review:

python

javascript

Copied

def code_review(code: str, language: str) -> list:
    """Generate a code review for the provided code snippet."""
    return [
        {
            "role": "system",
            "content": f"You are a code reviewer examining {language} code. Provide a detailed review highlighting best practices, potential issues, and suggestions for improvement."
        },
        {
            "role": "user",
            "content": f"Please review this {language} code:\n\n```{language}\n{code}\n```"
        }
    ]

## [](https://huggingface.co/learn/mcp-course/unit1/capabilities#sampling)Sampling

Sampling allows Servers to request the Client (specifically, the Host application) to perform LLM interactions.

- **Control**: Sampling is **server-initiated** but requires Client/Host facilitation.
- **Purpose**: It enables server-driven agentic behaviors and potentially recursive or multi-step interactions.
- **Safety**: Like Tools, sampling operations typically require user approval.
- **Use Cases**: Complex multi-step tasks, autonomous agent workflows, interactive processes.

**Example**: A Server might request the Client to analyze data it has processed:

python

javascript

Copied

def request_sampling(messages, system_prompt=None, include_context="none"):
    """Request LLM sampling from the client."""
    # In a real implementation, this would send a request to the client
    return {
        "role": "assistant",
        "content": "Analysis of the provided data..."
    }

The sampling flow follows these steps:

1. Server sends a `sampling/createMessage` request to the client
2. Client reviews the request and can modify it
3. Client samples from an LLM
4. Client reviews the completion
5. Client returns the result to the server

This human-in-the-loop design ensures users maintain control over what the LLM sees and generates. When implementing sampling, it’s important to provide clear, well-structured prompts and include relevant context.

## [](https://huggingface.co/learn/mcp-course/unit1/capabilities#how-capabilities-work-together)How Capabilities Work Together

Let’s look at how these capabilities work together to enable complex interactions. In the table below, we’ve outlined the capabilities, who controls them, the direction of control, and some other details.

|Capability|Controlled By|Direction|Side Effects|Approval Needed|Typical Use Cases|
|---|---|---|---|---|---|
|Tools|Model (LLM)|Client → Server|Yes (potentially)|Yes|Actions, API calls, data manipulation|
|Resources|Application|Client → Server|No (read-only)|Typically no|Data retrieval, context gathering|
|Prompts|User|Server → Client|No|No (selected by user)|Guided workflows, specialized templates|
|Sampling|Server|Server → Client → Server|Indirectly|Yes|Multi-step tasks, agentic behaviors|

These capabilities are designed to work together in complementary ways:

1. A user might select a **Prompt** to start a specialized workflow
2. The Prompt might include context from **Resources**
3. During processing, the AI model might call **Tools** to perform specific actions
4. For complex operations, the Server might use **Sampling** to request additional LLM processing

The distinction between these primitives provides a clear structure for MCP interactions, enabling AI models to access information, perform actions, and engage in complex workflows while maintaining appropriate control boundaries.

## [](https://huggingface.co/learn/mcp-course/unit1/capabilities#discovery-process)Discovery Process

One of MCP’s key features is dynamic capability discovery. When a Client connects to a Server, it can query the available Tools, Resources, and Prompts through specific list methods:

- `tools/list`: Discover available Tools
- `resources/list`: Discover available Resources
- `prompts/list`: Discover available Prompts

This dynamic discovery mechanism allows Clients to adapt to the specific capabilities each Server offers without requiring hardcoded knowledge of the Server’s functionality.

## [](https://huggingface.co/learn/mcp-course/unit1/capabilities#conclusion)Conclusion

Understanding these core primitives is essential for working with MCP effectively. By providing distinct types of capabilities with clear control boundaries, MCP enables powerful interactions between AI models and external systems while maintaining appropriate safety and control mechanisms.

In the next section, we’ll explore how Gradio integrates with MCP to provide easy-to-use interfaces for these capabilities.

[<>Update on GitHub](https://github.com/huggingface/mcp-course/blob/main/units/en/unit1/capabilities.mdx)

The Communication Protocol

[←The Communication Protocol](https://huggingface.co/learn/mcp-course/unit1/communication-protocol)[MCP SDK→](https://huggingface.co/learn/mcp-course/unit1/sdk)

[Understanding MCP Capabilities](https://huggingface.co/learn/mcp-course/unit1/capabilities#understanding-mcp-capabilities)[Tools](https://huggingface.co/learn/mcp-course/unit1/capabilities#tools)[Resources](https://huggingface.co/learn/mcp-course/unit1/capabilities#resources)[Prompts](https://huggingface.co/learn/mcp-course/unit1/capabilities#prompts)[Sampling](https://huggingface.co/learn/mcp-course/unit1/capabilities#sampling)[How Capabilities Work Together](https://huggingface.co/learn/mcp-course/unit1/capabilities#how-capabilities-work-together)[Discovery Process](https://huggingface.co/learn/mcp-course/unit1/capabilities#discovery-process)[Conclusion](https://huggingface.co/learn/mcp-course/unit1/capabilities#conclusion)


----

[![Hugging Face's logo](https://huggingface.co/front/assets/huggingface_logo-noborder.svg)Hugging Face](https://huggingface.co/)

- [Models](https://huggingface.co/models)
- [Datasets](https://huggingface.co/datasets)
- [Spaces](https://huggingface.co/spaces)
- Community
    
- [Docs](https://huggingface.co/docs)
- [Pricing](https://huggingface.co/pricing)

- ---
    
- ![](https://huggingface.co/avatars/5718fc9db9d5ef597ef85560419fd2ea.svg)
    

# MCP Course

🏡 View all resourcesAgents CourseAudio CourseCommunity Computer Vision CourseDeep RL CourseDiffusion CourseLLM CourseMCP CourseML for 3D CourseML for Games CourseOpen-Source AI Cookbook

Search documentation

⌘K

EN

 [548](https://github.com/huggingface/mcp-course)

0. Welcome to the MCP Course

[Welcome to the MCP Course](https://huggingface.co/learn/mcp-course/unit0/introduction)

1. Introduction to Model Context Protocol

[Introduction to Model Context Protocol (MCP)](https://huggingface.co/learn/mcp-course/unit1/introduction)[Key Concepts and Terminology](https://huggingface.co/learn/mcp-course/unit1/key-concepts)[Architectural Components](https://huggingface.co/learn/mcp-course/unit1/architectural-components)[Quiz 1 - MCP Fundamentals](https://huggingface.co/learn/mcp-course/unit1/quiz1)[The Communication Protocol](https://huggingface.co/learn/mcp-course/unit1/communication-protocol)[Understanding MCP Capabilities](https://huggingface.co/learn/mcp-course/unit1/capabilities)[MCP SDK](https://huggingface.co/learn/mcp-course/unit1/sdk)[Quiz 2 - MCP SDK](https://huggingface.co/learn/mcp-course/unit1/quiz2)[MCP Clients](https://huggingface.co/learn/mcp-course/unit1/mcp-clients)[Gradio MCP Integration](https://huggingface.co/learn/mcp-course/unit1/gradio-mcp)[Unit 1 Recap](https://huggingface.co/learn/mcp-course/unit1/unit1-recap)[Get your certificate!](https://huggingface.co/learn/mcp-course/unit1/certificate)

2. Use Case: End-to-End MCP Application

3. Use Case: Advanced MCP Development

Bonus Units

# [](https://huggingface.co/learn/mcp-course/unit1/sdk#mcp-sdk)MCP SDK

The Model Context Protocol provides official SDKs for both JavaScript, Python and other languages. This makes it easy to implement MCP clients and servers in your applications. These SDKs handle the low-level protocol details, allowing you to focus on building your application’s capabilities.

## [](https://huggingface.co/learn/mcp-course/unit1/sdk#sdk-overview)SDK Overview

Both SDKs provide similar core functionality, following the MCP protocol specification we discussed earlier. They handle:

- Protocol-level communication
- Capability registration and discovery
- Message serialization/deserialization
- Connection management
- Error handling

## [](https://huggingface.co/learn/mcp-course/unit1/sdk#core-primitives-implementation)Core Primitives Implementation

Let’s explore how to implement each of the core primitives (Tools, Resources, and Prompts) using both SDKs.

python

javascript

Copied

from mcp.server.fastmcp import FastMCP

# Create an MCP server
mcp = FastMCP("Weather Service")

# Tool implementation
@mcp.tool()
def get_weather(location: str) -> str:
    """Get the current weather for a specified location."""
    return f"Weather in {location}: Sunny, 72°F"

# Resource implementation
@mcp.resource("weather://{location}")
def weather_resource(location: str) -> str:
    """Provide weather data as a resource."""
    return f"Weather data for {location}: Sunny, 72°F"

# Prompt implementation
@mcp.prompt()
def weather_report(location: str) -> str:
    """Create a weather report prompt."""
    return f"""You are a weather reporter. Weather report for {location}?"""

# Run the server
if __name__ == "__main__":
    mcp.run()

Once you have your server implemented, you can start it by running the server script.

Copied

mcp dev server.py

This will initialize a development server running the file `server.py`. And log the following output:

Copied

Starting MCP inspector...
⚙️ Proxy server listening on port 6277
Spawned stdio transport
Connected MCP client to backing server transport
Created web app transport
Set up MCP proxy
🔍 MCP Inspector is up and running at http://127.0.0.1:6274 🚀

You can then open the MCP Inspector at [http://127.0.0.1:6274](http://127.0.0.1:6274/) to see the server’s capabilities and interact with them.

You’ll see the server’s capabilities and the ability to call them via the UI.

![MCP Inspector](https://huggingface.co/datasets/mcp-course/images/resolve/main/unit1/6.png)

## [](https://huggingface.co/learn/mcp-course/unit1/sdk#mcp-sdks)MCP SDKs

MCP is designed to be language-agnostic, and there are official SDKs available for several popular programming languages:

|Language|Repository|Maintainer(s)|Status|
|---|---|---|---|
|TypeScript|[github.com/modelcontextprotocol/typescript-sdk](https://github.com/modelcontextprotocol/typescript-sdk)|Anthropic|Active|
|Python|[github.com/modelcontextprotocol/python-sdk](https://github.com/modelcontextprotocol/python-sdk)|Anthropic|Active|
|Java|[github.com/modelcontextprotocol/java-sdk](https://github.com/modelcontextprotocol/java-sdk)|Spring AI (VMware)|Active|
|Kotlin|[github.com/modelcontextprotocol/kotlin-sdk](https://github.com/modelcontextprotocol/kotlin-sdk)|JetBrains|Active|
|C#|[github.com/modelcontextprotocol/csharp-sdk](https://github.com/modelcontextprotocol/csharp-sdk)|Microsoft|Active (Preview)|
|Swift|[github.com/modelcontextprotocol/swift-sdk](https://github.com/modelcontextprotocol/swift-sdk)|loopwork-ai|Active|
|Rust|[github.com/modelcontextprotocol/rust-sdk](https://github.com/modelcontextprotocol/rust-sdk)|Anthropic/Community|Active|
|Dart|[https://github.com/leehack/mcp_dart](https://github.com/leehack/mcp_dart)|Flutter Community|Active|

These SDKs provide language-specific abstractions that simplify working with the MCP protocol, allowing you to focus on implementing the core logic of your servers or clients rather than dealing with low-level protocol details.

## [](https://huggingface.co/learn/mcp-course/unit1/sdk#next-steps)Next Steps

We’ve only scratched the surface of what you can do with the MCP but you’ve already got a basic server running. In fact, you’ve also connected to it using the MCP Client in the browser.

In the next section, we’ll look at how to connect to your server from an LLM.

[<>Update on GitHub](https://github.com/huggingface/mcp-course/blob/main/units/en/unit1/sdk.mdx)

Understanding MCP Capabilities

[←Understanding MCP Capabilities](https://huggingface.co/learn/mcp-course/unit1/capabilities)[Quiz 2 - MCP SDK→](https://huggingface.co/learn/mcp-course/unit1/quiz2)

[MCP SDK](https://huggingface.co/learn/mcp-course/unit1/sdk#mcp-sdk)[SDK Overview](https://huggingface.co/learn/mcp-course/unit1/sdk#sdk-overview)[Core Primitives Implementation](https://huggingface.co/learn/mcp-course/unit1/sdk#core-primitives-implementation)[MCP SDKs](https://huggingface.co/learn/mcp-course/unit1/sdk#mcp-sdks)[Next Steps](https://huggingface.co/learn/mcp-course/unit1/sdk#next-steps)

---

[![Hugging Face's logo](https://huggingface.co/front/assets/huggingface_logo-noborder.svg)Hugging Face](https://huggingface.co/)

- [Models](https://huggingface.co/models)
- [Datasets](https://huggingface.co/datasets)
- [Spaces](https://huggingface.co/spaces)
- Community
    
- [Docs](https://huggingface.co/docs)
- [Pricing](https://huggingface.co/pricing)

- ---
    
- ![](https://huggingface.co/avatars/5718fc9db9d5ef597ef85560419fd2ea.svg)
    

# MCP Course

🏡 View all resourcesAgents CourseAudio CourseCommunity Computer Vision CourseDeep RL CourseDiffusion CourseLLM CourseMCP CourseML for 3D CourseML for Games CourseOpen-Source AI Cookbook

Search documentation

⌘K

EN

 [548](https://github.com/huggingface/mcp-course)

0. Welcome to the MCP Course

[Welcome to the MCP Course](https://huggingface.co/learn/mcp-course/unit0/introduction)

1. Introduction to Model Context Protocol

[Introduction to Model Context Protocol (MCP)](https://huggingface.co/learn/mcp-course/unit1/introduction)[Key Concepts and Terminology](https://huggingface.co/learn/mcp-course/unit1/key-concepts)[Architectural Components](https://huggingface.co/learn/mcp-course/unit1/architectural-components)[Quiz 1 - MCP Fundamentals](https://huggingface.co/learn/mcp-course/unit1/quiz1)[The Communication Protocol](https://huggingface.co/learn/mcp-course/unit1/communication-protocol)[Understanding MCP Capabilities](https://huggingface.co/learn/mcp-course/unit1/capabilities)[MCP SDK](https://huggingface.co/learn/mcp-course/unit1/sdk)[Quiz 2 - MCP SDK](https://huggingface.co/learn/mcp-course/unit1/quiz2)[MCP Clients](https://huggingface.co/learn/mcp-course/unit1/mcp-clients)[Gradio MCP Integration](https://huggingface.co/learn/mcp-course/unit1/gradio-mcp)[Unit 1 Recap](https://huggingface.co/learn/mcp-course/unit1/unit1-recap)[Get your certificate!](https://huggingface.co/learn/mcp-course/unit1/certificate)

2. Use Case: End-to-End MCP Application

3. Use Case: Advanced MCP Development

Bonus Units

# [](https://huggingface.co/learn/mcp-course/unit1/quiz2#quiz-2-mcp-sdk)Quiz 2: MCP SDK

Test your knowledge of the MCP SDKs and their functionalities.

### [](https://huggingface.co/learn/mcp-course/unit1/quiz2#q1-what-is-the-main-purpose-of-the-mcp-sdks)Q1: What is the main purpose of the MCP SDKs?

 To define the MCP protocol specification To make it easier to implement MCP clients and servers To provide a visual interface for MCP interactions To replace the need for programming languages

Submit

### [](https://huggingface.co/learn/mcp-course/unit1/quiz2#q2-which-of-the-following-functionalities-do-the-mcp-sdks-typically-handle)Q2: Which of the following functionalities do the MCP SDKs typically handle?

 Optimizing MCP Servers Defining new AI algorithms Message serialization/deserialization Hosting Large Language Models

Submit

### [](https://huggingface.co/learn/mcp-course/unit1/quiz2#q3-according-to-the-provided-text-which-company-maintains-the-official-python-sdk-for-mcp)Q3: According to the provided text, which company maintains the official Python SDK for MCP?

 Google Anthropic Microsoft JetBrains

Submit

### [](https://huggingface.co/learn/mcp-course/unit1/quiz2#q4-what-command-is-used-to-start-a-development-mcp-server-using-a-python-file-named-serverpy-)Q4: What command is used to start a development MCP server using a Python file named server.py ?

 python server.py run mcp start server.py mcp dev server.py serve mcp server.py

Submit

### [](https://huggingface.co/learn/mcp-course/unit1/quiz2#q5-what-is-the-role-of-json-rpc-20-in-mcp)Q5: What is the role of JSON-RPC 2.0 in MCP?

 As a primary transport mechanism for remote communication As the message format for all communication between Clients and Servers As a tool for debugging AI models As a method for defining AI capabilities like Tools and Resources

Submit

Congrats on finishing this Quiz 🥳! If you need to review any elements, take the time to revisit the chapter to reinforce your knowledge.

[<>Update on GitHub](https://github.com/huggingface/mcp-course/blob/main/units/en/unit1/quiz2.mdx)

MCP SDK

[←MCP SDK](https://huggingface.co/learn/mcp-course/unit1/sdk)[MCP Clients→](https://huggingface.co/learn/mcp-course/unit1/mcp-clients)

[Quiz 2: MCP SDK](https://huggingface.co/learn/mcp-course/unit1/quiz2#quiz-2-mcp-sdk)[Q1: What is the main purpose of the MCP SDKs?](https://huggingface.co/learn/mcp-course/unit1/quiz2#q1-what-is-the-main-purpose-of-the-mcp-sdks)[Q2: Which of the following functionalities do the MCP SDKs typically handle?](https://huggingface.co/learn/mcp-course/unit1/quiz2#q2-which-of-the-following-functionalities-do-the-mcp-sdks-typically-handle)[Q3: According to the provided text, which company maintains the official Python SDK for MCP?](https://huggingface.co/learn/mcp-course/unit1/quiz2#q3-according-to-the-provided-text-which-company-maintains-the-official-python-sdk-for-mcp)[Q4: What command is used to start a development MCP server using a Python file named server.py ?](https://huggingface.co/learn/mcp-course/unit1/quiz2#q4-what-command-is-used-to-start-a-development-mcp-server-using-a-python-file-named-serverpy-)[Q5: What is the role of JSON-RPC 2.0 in MCP?](https://huggingface.co/learn/mcp-course/unit1/quiz2#q5-what-is-the-role-of-json-rpc-20-in-mcp)


------------

[![Hugging Face's logo](https://huggingface.co/front/assets/huggingface_logo-noborder.svg)Hugging Face](https://huggingface.co/)

- [Models](https://huggingface.co/models)
- [Datasets](https://huggingface.co/datasets)
- [Spaces](https://huggingface.co/spaces)
- Community
    
- [Docs](https://huggingface.co/docs)
- [Pricing](https://huggingface.co/pricing)

- ---
    
- ![](https://huggingface.co/avatars/5718fc9db9d5ef597ef85560419fd2ea.svg)
    

# MCP Course

🏡 View all resourcesAgents CourseAudio CourseCommunity Computer Vision CourseDeep RL CourseDiffusion CourseLLM CourseMCP CourseML for 3D CourseML for Games CourseOpen-Source AI Cookbook

Search documentation

⌘K

EN

 [548](https://github.com/huggingface/mcp-course)

0. Welcome to the MCP Course

[Welcome to the MCP Course](https://huggingface.co/learn/mcp-course/unit0/introduction)

1. Introduction to Model Context Protocol

[Introduction to Model Context Protocol (MCP)](https://huggingface.co/learn/mcp-course/unit1/introduction)[Key Concepts and Terminology](https://huggingface.co/learn/mcp-course/unit1/key-concepts)[Architectural Components](https://huggingface.co/learn/mcp-course/unit1/architectural-components)[Quiz 1 - MCP Fundamentals](https://huggingface.co/learn/mcp-course/unit1/quiz1)[The Communication Protocol](https://huggingface.co/learn/mcp-course/unit1/communication-protocol)[Understanding MCP Capabilities](https://huggingface.co/learn/mcp-course/unit1/capabilities)[MCP SDK](https://huggingface.co/learn/mcp-course/unit1/sdk)[Quiz 2 - MCP SDK](https://huggingface.co/learn/mcp-course/unit1/quiz2)[MCP Clients](https://huggingface.co/learn/mcp-course/unit1/mcp-clients)[Gradio MCP Integration](https://huggingface.co/learn/mcp-course/unit1/gradio-mcp)[Unit 1 Recap](https://huggingface.co/learn/mcp-course/unit1/unit1-recap)[Get your certificate!](https://huggingface.co/learn/mcp-course/unit1/certificate)

2. Use Case: End-to-End MCP Application

3. Use Case: Advanced MCP Development

Bonus Units

# [](https://huggingface.co/learn/mcp-course/unit1/mcp-clients#mcp-clients)MCP Clients

Now that we have a basic understanding of the Model Context Protocol, we can explore the essential role of MCP Clients in the Model Context Protocol ecosystem.

In this part of Unit 1, we’ll explore the essential role of MCP Clients in the Model Context Protocol ecosystem.

In this section, you will:

- Understand what MCP Clients are and their role in the MCP architecture
- Learn about the key responsibilities of MCP Clients
- Explore the major MCP Client implementations
- Discover how to use Hugging Face’s MCP Client implementation
- See practical examples of MCP Client usage

In this page we’re going to show examples of how to set up MCP Clients in a few different ways using the JSON notation. For now, we will use _examples_ like `path/to/server.py` to represent the path to the MCP Server. In the next unit, we’ll implement this with real MCP Servers.

For now, focus on understanding the MCP Client notation. We’ll implement the MCP Servers in the next unit.

## [](https://huggingface.co/learn/mcp-course/unit1/mcp-clients#understanding-mcp-clients)Understanding MCP Clients

MCP Clients are crucial components that act as the bridge between AI applications (Hosts) and external capabilities provided by MCP Servers. Think of the Host as your main application (like an AI assistant or IDE) and the Client as a specialized module within that Host responsible for handling MCP communications.

## [](https://huggingface.co/learn/mcp-course/unit1/mcp-clients#user-interface-client)User Interface Client

Let’s start by exploring the user interface clients that are available for the MCP.

### [](https://huggingface.co/learn/mcp-course/unit1/mcp-clients#chat-interface-clients)Chat Interface Clients

Anthropic’s Claude Desktop stands as one of the most prominent MCP Clients, providing integration with various MCP Servers.

### [](https://huggingface.co/learn/mcp-course/unit1/mcp-clients#interactive-development-clients)Interactive Development Clients

Cursor’s MCP Client implementation enables AI-powered coding assistance through direct integration with code editing capabilities. It supports multiple MCP Server connections and provides real-time tool invocation during coding, making it a powerful tool for developers.

Continue.dev is another example of an interactive development client that supports MCP and connects to an MCP server from VS Code.

## [](https://huggingface.co/learn/mcp-course/unit1/mcp-clients#configuring-mcp-clients)Configuring MCP Clients

Now that we’ve covered the core of the MCP protocol, let’s look at how to configure your MCP servers and clients.

Effective deployment of MCP servers and clients requires proper configuration.

The MCP specification is still evolving, so the configuration methods are subject to evolution. We’ll focus on the current best practices for configuration.

### [](https://huggingface.co/learn/mcp-course/unit1/mcp-clients#mcp-configuration-files)MCP Configuration Files

MCP hosts use configuration files to manage server connections. These files define which servers are available and how to connect to them.

Fortunately, the configuration files are very simple, easy to understand, and consistent across major MCP hosts.

#### [](https://huggingface.co/learn/mcp-course/unit1/mcp-clients#mcpjson-structure)mcp.json Structure

The standard configuration file for MCP is named `mcp.json`. Here’s the basic structure:

This is the basic structure of the `mcp.json` can be passed to applications like Claude Desktop, Cursor, or VS Code.

Copied

{
  "servers": [
    {
      "name": "Server Name",
      "transport": {
        "type": "stdio|sse",
        // Transport-specific configuration
      }
    }
  ]
}

In this example, we have a single server with a name and a transport type. The transport type is either `stdio` or `sse`.

#### [](https://huggingface.co/learn/mcp-course/unit1/mcp-clients#configuration-for-stdio-transport)Configuration for stdio Transport

For local servers using stdio transport, the configuration includes the command and arguments to launch the server process:

Copied

{
  "servers": [
    {
      "name": "File Explorer",
      "transport": {
        "type": "stdio",
        "command": "python",
        "args": ["/path/to/file_explorer_server.py"] // This is an example, we'll use a real server in the next unit
      }
    }
  ]
}

Here, we have a server called “File Explorer” that is a local script.

#### [](https://huggingface.co/learn/mcp-course/unit1/mcp-clients#configuration-for-httpsse-transport)Configuration for HTTP+SSE Transport

For remote servers using HTTP+SSE transport, the configuration includes the server URL:

Copied

{
  "servers": [
    {
      "name": "Remote API Server",
      "transport": {
        "type": "sse",
        "url": "https://example.com/mcp-server"
      }
    }
  ]
}

#### [](https://huggingface.co/learn/mcp-course/unit1/mcp-clients#environment-variables-in-configuration)Environment Variables in Configuration

Environment variables can be passed to server processes using the `env` field. Here’s how to access them in your server code:

python

javascript

In Python, we use the `os` module to access environment variables:

Copied

import os

# Access environment variables
github_token = os.environ.get("GITHUB_TOKEN")
if not github_token:
    raise ValueError("GITHUB_TOKEN environment variable is required")

# Use the token in your server code
def make_github_request():
    headers = {"Authorization": f"Bearer {github_token}"}
    # ... rest of your code

The corresponding configuration in `mcp.json` would look like this:

Copied

{
  "servers": [
    {
      "name": "GitHub API",
      "transport": {
        "type": "stdio",
        "command": "python",
        "args": ["/path/to/github_server.py"], // This is an example, we'll use a real server in the next unit
        "env": {
          "GITHUB_TOKEN": "your_github_token"
        }
      }
    }
  ]
}

### [](https://huggingface.co/learn/mcp-course/unit1/mcp-clients#configuration-examples)Configuration Examples

Let’s look at some real-world configuration scenarios:

#### [](https://huggingface.co/learn/mcp-course/unit1/mcp-clients#scenario-1-local-server-configuration)Scenario 1: Local Server Configuration

In this scenario, we have a local server that is a Python script which could be a file explorer or a code editor.

Copied

{
  "servers": [
    {
      "name": "File Explorer",
      "transport": {
        "type": "stdio",
        "command": "python",
        "args": ["/path/to/file_explorer_server.py"] // This is an example, we'll use a real server in the next unit
      }
    }
  ]
}

#### [](https://huggingface.co/learn/mcp-course/unit1/mcp-clients#scenario-2-remote-server-configuration)Scenario 2: Remote Server Configuration

In this scenario, we have a remote server that is a weather API.

Copied

{
  "servers": [
    {
      "name": "Weather API",
      "transport": {
        "type": "sse",
        "url": "https://example.com/mcp-server" // This is an example, we'll use a real server in the next unit
      }
    }
  ]
}

Proper configuration is essential for successfully deploying MCP integrations. By understanding these aspects, you can create robust and reliable connections between AI applications and external capabilities.

In the next section, we’ll explore the ecosystem of MCP servers available on Hugging Face Hub and how to publish your own servers there.

## [](https://huggingface.co/learn/mcp-course/unit1/mcp-clients#tiny-agents-clients)Tiny Agents Clients

Now, let’s explore how to use MCP Clients within code.

You can also use tiny agents as MCP Clients to connect directly to MCP servers from your code. Tiny agents provide a simple way to create AI agents that can use tools from MCP servers.

Tiny Agent can run MCP servers with a command line environment. To do this, we will need to install `npm` and run the server with `npx`. **We’ll need these for both Python and JavaScript.**

Let’s install `npx` with `npm`. If you don’t have `npm` installed, check out the [npm documentation](https://docs.npmjs.com/downloading-and-installing-node-js-and-npm).

### [](https://huggingface.co/learn/mcp-course/unit1/mcp-clients#setup)Setup

First, we will need to install `npx` if you don’t have it installed. You can do this with the following command:

Copied

# install npx
npm install -g npx

Then, we will need to install the huggingface_hub package with the MCP support. This will allow us to run MCP servers and clients.

Copied

pip install "huggingface_hub[mcp]>=0.32.0"

Then, we will need to log in to the Hugging Face Hub to access the MCP servers. You can do this with the `huggingface-cli` command line tool. You will need a [login token](https://huggingface.co/docs/huggingface_hub/v0.32.3/en/quick-start#authentication) to do this.

Copied

huggingface-cli login

python

javascript

### [](https://huggingface.co/learn/mcp-course/unit1/mcp-clients#connecting-to-mcp-servers)Connecting to MCP Servers

Now, let’s create an agent configuration file `agent.json`.

Copied

{
    "model": "Qwen/Qwen2.5-72B-Instruct",
    "provider": "nebius",
    "servers": [
        {
            "type": "stdio",
            "config": {
                "command": "npx",
                "args": ["@playwright/mcp@latest"]
            }
        }
    ]
}

In this configuration, we are using the `@playwright/mcp` MCP server. This is a MCP server that can control a browser with Playwright.

Now you can run the agent:

Copied

tiny-agents run agent.json

In the video below, we run the agent and ask it to open a new tab in the browser.

The following example shows a web-browsing agent configured to use the [Qwen/Qwen2.5-72B-Instruct](https://huggingface.co/Qwen/Qwen2.5-72B-Instruct) model via Nebius inference provider, and it comes equipped with a playwright MCP server, which lets it use a web browser! The agent config is loaded specifying [its path in the `tiny-agents/tiny-agents`](https://huggingface.co/datasets/tiny-agents/tiny-agents/tree/main/celinah/web-browser) Hugging Face dataset.

When you run the agent, you’ll see it load, listing the tools it has discovered from its connected MCP servers. Then, it’s ready for your prompts!

Prompt used in this demo:

> do a Web Search for HF inference providers on Brave Search and open the first result and then give me the list of the inference providers supported on Hugging Face

## [](https://huggingface.co/learn/mcp-course/unit1/mcp-clients#next-steps)Next Steps

Now that you understand MCP Clients, you’re ready to:

- Explore specific MCP Server implementations
- Learn about creating custom MCP Clients
- Dive into advanced MCP integration patterns

Let’s continue our journey into the world of Model Context Protocol!

[<>Update on GitHub](https://github.com/huggingface/mcp-course/blob/main/units/en/unit1/mcp-clients.mdx)

MCP Clients

[←Quiz 2 - MCP SDK](https://huggingface.co/learn/mcp-course/unit1/quiz2)[Gradio MCP Integration→](https://huggingface.co/learn/mcp-course/unit1/gradio-mcp)

[MCP Clients](https://huggingface.co/learn/mcp-course/unit1/mcp-clients#mcp-clients)[Understanding MCP Clients](https://huggingface.co/learn/mcp-course/unit1/mcp-clients#understanding-mcp-clients)[User Interface Client](https://huggingface.co/learn/mcp-course/unit1/mcp-clients#user-interface-client)[Chat Interface Clients](https://huggingface.co/learn/mcp-course/unit1/mcp-clients#chat-interface-clients)[Interactive Development Clients](https://huggingface.co/learn/mcp-course/unit1/mcp-clients#interactive-development-clients)[Configuring MCP Clients](https://huggingface.co/learn/mcp-course/unit1/mcp-clients#configuring-mcp-clients)[MCP Configuration Files](https://huggingface.co/learn/mcp-course/unit1/mcp-clients#mcp-configuration-files)[mcp.json Structure](https://huggingface.co/learn/mcp-course/unit1/mcp-clients#mcpjson-structure)[Configuration for stdio Transport](https://huggingface.co/learn/mcp-course/unit1/mcp-clients#configuration-for-stdio-transport)[Configuration for HTTP+SSE Transport](https://huggingface.co/learn/mcp-course/unit1/mcp-clients#configuration-for-httpsse-transport)[Environment Variables in Configuration](https://huggingface.co/learn/mcp-course/unit1/mcp-clients#environment-variables-in-configuration)[Configuration Examples](https://huggingface.co/learn/mcp-course/unit1/mcp-clients#configuration-examples)[Scenario 1: Local Server Configuration](https://huggingface.co/learn/mcp-course/unit1/mcp-clients#scenario-1-local-server-configuration)[Scenario 2: Remote Server Configuration](https://huggingface.co/learn/mcp-course/unit1/mcp-clients#scenario-2-remote-server-configuration)[Tiny Agents Clients](https://huggingface.co/learn/mcp-course/unit1/mcp-clients#tiny-agents-clients)[Setup](https://huggingface.co/learn/mcp-course/unit1/mcp-clients#setup)[Connecting to MCP Servers](https://huggingface.co/learn/mcp-course/unit1/mcp-clients#connecting-to-mcp-servers)[Connecting to MCP Servers](https://huggingface.co/learn/mcp-course/unit1/mcp-clients#connecting-to-mcp-servers)[Next Steps](https://huggingface.co/learn/mcp-course/unit1/mcp-clients#next-steps)


------------


[![Hugging Face's logo](https://huggingface.co/front/assets/huggingface_logo-noborder.svg)Hugging Face](https://huggingface.co/)

- [Models](https://huggingface.co/models)
- [Datasets](https://huggingface.co/datasets)
- [Spaces](https://huggingface.co/spaces)
- Community
    
- [Docs](https://huggingface.co/docs)
- [Pricing](https://huggingface.co/pricing)

- ---
    
- ![](https://huggingface.co/avatars/5718fc9db9d5ef597ef85560419fd2ea.svg)
    

# MCP Course

🏡 View all resourcesAgents CourseAudio CourseCommunity Computer Vision CourseDeep RL CourseDiffusion CourseLLM CourseMCP CourseML for 3D CourseML for Games CourseOpen-Source AI Cookbook

Search documentation

⌘K

EN

 [548](https://github.com/huggingface/mcp-course)

0. Welcome to the MCP Course

[Welcome to the MCP Course](https://huggingface.co/learn/mcp-course/unit0/introduction)

1. Introduction to Model Context Protocol

[Introduction to Model Context Protocol (MCP)](https://huggingface.co/learn/mcp-course/unit1/introduction)[Key Concepts and Terminology](https://huggingface.co/learn/mcp-course/unit1/key-concepts)[Architectural Components](https://huggingface.co/learn/mcp-course/unit1/architectural-components)[Quiz 1 - MCP Fundamentals](https://huggingface.co/learn/mcp-course/unit1/quiz1)[The Communication Protocol](https://huggingface.co/learn/mcp-course/unit1/communication-protocol)[Understanding MCP Capabilities](https://huggingface.co/learn/mcp-course/unit1/capabilities)[MCP SDK](https://huggingface.co/learn/mcp-course/unit1/sdk)[Quiz 2 - MCP SDK](https://huggingface.co/learn/mcp-course/unit1/quiz2)[MCP Clients](https://huggingface.co/learn/mcp-course/unit1/mcp-clients)[Gradio MCP Integration](https://huggingface.co/learn/mcp-course/unit1/gradio-mcp)[Unit 1 Recap](https://huggingface.co/learn/mcp-course/unit1/unit1-recap)[Get your certificate!](https://huggingface.co/learn/mcp-course/unit1/certificate)

2. Use Case: End-to-End MCP Application

3. Use Case: Advanced MCP Development

Bonus Units

# [](https://huggingface.co/learn/mcp-course/unit1/gradio-mcp#gradio-mcp-integration)Gradio MCP Integration

We’ve now explored the core concepts of the MCP protocol and how to implement MCP Servers and Clients. In this section, we’re going to make things slightly easier by using Gradio to create an MCP Server!

Gradio is a popular Python library for quickly creating customizable web interfaces for machine learning models.

## [](https://huggingface.co/learn/mcp-course/unit1/gradio-mcp#introduction-to-gradio)Introduction to Gradio

Gradio allows developers to create UIs for their models with just a few lines of Python code. It’s particularly useful for:

- Creating demos and prototypes
- Sharing models with non-technical users
- Testing and debugging model behavior

With the addition of MCP support, Gradio now offers a straightforward way to expose AI model capabilities through the standardized MCP protocol.

Combining Gradio with MCP allows you to create both human-friendly interfaces and AI-accessible tools with minimal code. But best of all, Gradio is already well-used by the AI community, so you can use it to share your MCP Servers with others.

## [](https://huggingface.co/learn/mcp-course/unit1/gradio-mcp#prerequisites)Prerequisites

To use Gradio with MCP support, you’ll need to install Gradio with the MCP extra:

Copied

pip install "gradio[mcp]"

You’ll also need an LLM application that supports tool calling using the MCP protocol, such as Cursor ( known as “MCP Hosts”).

## [](https://huggingface.co/learn/mcp-course/unit1/gradio-mcp#creating-an-mcp-server-with-gradio)Creating an MCP Server with Gradio

Let’s walk through a basic example of creating an MCP Server using Gradio:

Copied

import gradio as gr

def letter_counter(word: str, letter: str) -> int:
    """
    Count the number of occurrences of a letter in a word or text.

    Args:
        word (str): The input text to search through
        letter (str): The letter to search for

    Returns:
        int: The number of times the letter appears in the text
    """
    word = word.lower()
    letter = letter.lower()
    count = word.count(letter)
    return count

# Create a standard Gradio interface
demo = gr.Interface(
    fn=letter_counter,
    inputs=["textbox", "textbox"],
    outputs="number",
    title="Letter Counter",
    description="Enter text and a letter to count how many times the letter appears in the text."
)

# Launch both the Gradio web interface and the MCP server
if __name__ == "__main__":
    demo.launch(mcp_server=True)

With this setup, your letter counter function is now accessible through:

1. A traditional Gradio web interface for direct human interaction
2. An MCP Server that can be connected to compatible clients

The MCP server will be accessible at:

Copied

http://your-server:port/gradio_api/mcp/sse

The application itself will still be accessible and it looks like this:

![Gradio MCP Server](https://huggingface.co/datasets/mcp-course/images/resolve/main/unit1/7.png)

## [](https://huggingface.co/learn/mcp-course/unit1/gradio-mcp#how-it-works-behind-the-scenes)How It Works Behind the Scenes

When you set `mcp_server=True` in `launch()`, several things happen:

1. Gradio functions are automatically converted to MCP Tools
2. Input components map to tool argument schemas
3. Output components determine the response format
4. The Gradio server now also listens for MCP protocol messages
5. JSON-RPC over HTTP+SSE is set up for client-server communication

## [](https://huggingface.co/learn/mcp-course/unit1/gradio-mcp#key-features-of-the-gradio--mcp-integration)Key Features of the Gradio <> MCP Integration

1. **Tool Conversion**: Each API endpoint in your Gradio app is automatically converted into an MCP tool with a corresponding name, description, and input schema. To view the tools and schemas, visit `http://your-server:port/gradio_api/mcp/schema` or go to the “View API” link in the footer of your Gradio app, and then click on “MCP”.
    
2. **Environment Variable Support**: There are two ways to enable the MCP server functionality:
    

- Using the `mcp_server` parameter in `launch()`:
    
    Copied
    
    demo.launch(mcp_server=True)
    
- Using environment variables:
    
    Copied
    
    export GRADIO_MCP_SERVER=True
    

3. **File Handling**: The server automatically handles file data conversions, including:
    
    - Converting base64-encoded strings to file data
    - Processing image files and returning them in the correct format
    - Managing temporary file storage
    
    It is **strongly** recommended that input images and files be passed as full URLs (“http://…” or “https://…”) as MCP Clients do not always handle local files correctly.
    
4. **Hosted MCP Servers on 🤗 Spaces**: You can publish your Gradio application for free on Hugging Face Spaces, which will allow you to have a free hosted MCP server. Here’s an example of such a Space: [https://huggingface.co/spaces/abidlabs/mcp-tools](https://huggingface.co/spaces/abidlabs/mcp-tools)
    

## [](https://huggingface.co/learn/mcp-course/unit1/gradio-mcp#troubleshooting-tips)Troubleshooting Tips

1. **Type Hints and Docstrings**: Ensure you provide type hints and valid docstrings for your functions. The docstring should include an “Args:” block with indented parameter names.
    
2. **String Inputs**: When in doubt, accept input arguments as `str` and convert them to the desired type inside the function.
    
3. **SSE Support**: Some MCP Hosts don’t support SSE-based MCP Servers. In those cases, you can use `mcp-remote`:
    
    Copied
    
    {
      "mcpServers": {
        "gradio": {
          "command": "npx",
          "args": [
            "mcp-remote",
            "http://your-server:port/gradio_api/mcp/sse"
          ]
        }
      }
    }
    
4. **Restart**: If you encounter connection issues, try restarting both your MCP Client and MCP Server.
    

## [](https://huggingface.co/learn/mcp-course/unit1/gradio-mcp#share-your-mcp-server)Share your MCP Server

You can share your MCP Server by publishing your Gradio app to Hugging Face Spaces. The video below shows how to create a Hugging Face Space.

Now, you can share your MCP Server with others by sharing your Hugging Face Space.

## [](https://huggingface.co/learn/mcp-course/unit1/gradio-mcp#conclusion)Conclusion

Gradio’s integration with MCP provides an accessible entry point to the MCP ecosystem. By leveraging Gradio’s simplicity and adding MCP’s standardization, developers can quickly create both human-friendly interfaces and AI-accessible tools with minimal code.

As we progress through this course, we’ll explore more sophisticated MCP implementations, but Gradio offers an excellent starting point for understanding and experimenting with the protocol.

In the next unit, we’ll dive deeper into building MCP applications, focusing on setting up development environments, exploring SDKs, and implementing more advanced MCP Servers and Clients.

[<>Update on GitHub](https://github.com/huggingface/mcp-course/blob/main/units/en/unit1/gradio-mcp.mdx)

MCP Clients

[←MCP Clients](https://huggingface.co/learn/mcp-course/unit1/mcp-clients)[Unit 1 Recap→](https://huggingface.co/learn/mcp-course/unit1/unit1-recap)

[Gradio MCP Integration](https://huggingface.co/learn/mcp-course/unit1/gradio-mcp#gradio-mcp-integration)[Introduction to Gradio](https://huggingface.co/learn/mcp-course/unit1/gradio-mcp#introduction-to-gradio)[Prerequisites](https://huggingface.co/learn/mcp-course/unit1/gradio-mcp#prerequisites)[Creating an MCP Server with Gradio](https://huggingface.co/learn/mcp-course/unit1/gradio-mcp#creating-an-mcp-server-with-gradio)[How It Works Behind the Scenes](https://huggingface.co/learn/mcp-course/unit1/gradio-mcp#how-it-works-behind-the-scenes)[Key Features of the Gradio <> MCP Integration](https://huggingface.co/learn/mcp-course/unit1/gradio-mcp#key-features-of-the-gradio--mcp-integration)[Troubleshooting Tips](https://huggingface.co/learn/mcp-course/unit1/gradio-mcp#troubleshooting-tips)[Share your MCP Server](https://huggingface.co/learn/mcp-course/unit1/gradio-mcp#share-your-mcp-server)[Conclusion](https://huggingface.co/learn/mcp-course/unit1/gradio-mcp#conclusion)


--------------


[![Hugging Face's logo](https://huggingface.co/front/assets/huggingface_logo-noborder.svg)Hugging Face](https://huggingface.co/)

- [Models](https://huggingface.co/models)
- [Datasets](https://huggingface.co/datasets)
- [Spaces](https://huggingface.co/spaces)
- Community
    
- [Docs](https://huggingface.co/docs)
- [Pricing](https://huggingface.co/pricing)

- ---
    
- ![](https://huggingface.co/avatars/5718fc9db9d5ef597ef85560419fd2ea.svg)
    

# MCP Course

🏡 View all resourcesAgents CourseAudio CourseCommunity Computer Vision CourseDeep RL CourseDiffusion CourseLLM CourseMCP CourseML for 3D CourseML for Games CourseOpen-Source AI Cookbook

Search documentation

⌘K

EN

 [548](https://github.com/huggingface/mcp-course)

0. Welcome to the MCP Course

[Welcome to the MCP Course](https://huggingface.co/learn/mcp-course/unit0/introduction)

1. Introduction to Model Context Protocol

[Introduction to Model Context Protocol (MCP)](https://huggingface.co/learn/mcp-course/unit1/introduction)[Key Concepts and Terminology](https://huggingface.co/learn/mcp-course/unit1/key-concepts)[Architectural Components](https://huggingface.co/learn/mcp-course/unit1/architectural-components)[Quiz 1 - MCP Fundamentals](https://huggingface.co/learn/mcp-course/unit1/quiz1)[The Communication Protocol](https://huggingface.co/learn/mcp-course/unit1/communication-protocol)[Understanding MCP Capabilities](https://huggingface.co/learn/mcp-course/unit1/capabilities)[MCP SDK](https://huggingface.co/learn/mcp-course/unit1/sdk)[Quiz 2 - MCP SDK](https://huggingface.co/learn/mcp-course/unit1/quiz2)[MCP Clients](https://huggingface.co/learn/mcp-course/unit1/mcp-clients)[Gradio MCP Integration](https://huggingface.co/learn/mcp-course/unit1/gradio-mcp)[Unit 1 Recap](https://huggingface.co/learn/mcp-course/unit1/unit1-recap)[Get your certificate!](https://huggingface.co/learn/mcp-course/unit1/certificate)

2. Use Case: End-to-End MCP Application

3. Use Case: Advanced MCP Development

Bonus Units

# [](https://huggingface.co/learn/mcp-course/unit1/unit1-recap#unit1-recap)Unit1 recap

## [](https://huggingface.co/learn/mcp-course/unit1/unit1-recap#model-context-protocol-mcp)Model Context Protocol (MCP)

The MCP is a standardized protocol designed to connect AI models with external tools, data sources, and environments. It addresses the limitations of existing AI systems by enabling interoperability and access to real-time information.

## [](https://huggingface.co/learn/mcp-course/unit1/unit1-recap#key-concepts)Key Concepts

### [](https://huggingface.co/learn/mcp-course/unit1/unit1-recap#client-server-architecture)Client-Server Architecture

MCP follows a client-server model where clients manage communication between users and servers. This architecture promotes modularity, allowing for easy addition of new servers without requiring changes to existing hosts.

### [](https://huggingface.co/learn/mcp-course/unit1/unit1-recap#components)Components

#### [](https://huggingface.co/learn/mcp-course/unit1/unit1-recap#host)Host

The user-facing AI application that serves as the interface for end-users.

##### [](https://huggingface.co/learn/mcp-course/unit1/unit1-recap#client)Client

A component within the host application responsible for managing communication with a specific MCP server. Clients maintain 1:1 connections with servers and handle protocol-level details.

#### [](https://huggingface.co/learn/mcp-course/unit1/unit1-recap#server)Server

An external program or service that provides access to tools, data sources, or services via the MCP protocol. Servers act as lightweight wrappers around existing functionalities.

### [](https://huggingface.co/learn/mcp-course/unit1/unit1-recap#capabilities)Capabilities

#### [](https://huggingface.co/learn/mcp-course/unit1/unit1-recap#tools)Tools

Executable functions that can perform actions (e.g., sending messages, querying APIs). Tools are typically model-controlled and require user approval due to their ability to perform actions with side effects.

#### [](https://huggingface.co/learn/mcp-course/unit1/unit1-recap#resources)Resources

Read-only data sources for context retrieval without significant computation. Resources are application-controlled and designed for data retrieval similar to GET endpoints in REST APIs.

#### [](https://huggingface.co/learn/mcp-course/unit1/unit1-recap#prompts)Prompts

Pre-defined templates or workflows that guide interactions between users, AI models, and available capabilities. Prompts are user-controlled and set the context for interactions.

#### [](https://huggingface.co/learn/mcp-course/unit1/unit1-recap#sampling)Sampling

Server-initiated requests for LLM processing, enabling server-driven agentic behaviors and potentially recursive or multi-step interactions. Sampling operations typically require user approval.

### [](https://huggingface.co/learn/mcp-course/unit1/unit1-recap#communication-protocol)Communication Protocol

The MCP protocol uses JSON-RPC 2.0 as the message format for communication between clients and servers. Two primary transport mechanisms are supported: stdio (for local communication) and HTTP+SSE (for remote communication). Messages include requests, responses, and notifications.

### [](https://huggingface.co/learn/mcp-course/unit1/unit1-recap#discovery-process)Discovery Process

MCP allows clients to dynamically discover available tools, resources, and prompts through list methods (e.g., `tools/list`). This dynamic discovery mechanism enables clients to adapt to the specific capabilities each server offers without requiring hardcoded knowledge of server functionality.

### [](https://huggingface.co/learn/mcp-course/unit1/unit1-recap#mcp-sdks)MCP SDKs

Official SDKs are available in various programming languages for implementing MCP clients and servers. These SDKs handle protocol-level communication, capability registration, and error handling, simplifying the development process.

### [](https://huggingface.co/learn/mcp-course/unit1/unit1-recap#gradio-integration)Gradio Integration

Gradio allows easy creation of web interfaces that expose capabilities to the MCP protocol, making it accessible for both humans and AI models. This integration provides a human-friendly interface alongside AI-accessible tools with minimal code.

[<>Update on GitHub](https://github.com/huggingface/mcp-course/blob/main/units/en/unit1/unit1-recap.mdx)

Unit1 recap

[←Gradio MCP Integration](https://huggingface.co/learn/mcp-course/unit1/gradio-mcp)[Get your certificate!→](https://huggingface.co/learn/mcp-course/unit1/certificate)

[Unit1 recap](https://huggingface.co/learn/mcp-course/unit1/unit1-recap#unit1-recap)[Model Context Protocol (MCP)](https://huggingface.co/learn/mcp-course/unit1/unit1-recap#model-context-protocol-mcp)[Key Concepts](https://huggingface.co/learn/mcp-course/unit1/unit1-recap#key-concepts)[Client-Server Architecture](https://huggingface.co/learn/mcp-course/unit1/unit1-recap#client-server-architecture)[Components](https://huggingface.co/learn/mcp-course/unit1/unit1-recap#components)[Host](https://huggingface.co/learn/mcp-course/unit1/unit1-recap#host)[Server](https://huggingface.co/learn/mcp-course/unit1/unit1-recap#server)[Capabilities](https://huggingface.co/learn/mcp-course/unit1/unit1-recap#capabilities)[Tools](https://huggingface.co/learn/mcp-course/unit1/unit1-recap#tools)[Resources](https://huggingface.co/learn/mcp-course/unit1/unit1-recap#resources)[Prompts](https://huggingface.co/learn/mcp-course/unit1/unit1-recap#prompts)[Sampling](https://huggingface.co/learn/mcp-course/unit1/unit1-recap#sampling)[Communication Protocol](https://huggingface.co/learn/mcp-course/unit1/unit1-recap#communication-protocol)[Discovery Process](https://huggingface.co/learn/mcp-course/unit1/unit1-recap#discovery-process)[MCP SDKs](https://huggingface.co/learn/mcp-course/unit1/unit1-recap#mcp-sdks)[Gradio Integration](https://huggingface.co/learn/mcp-course/unit1/unit1-recap#gradio-integration)

---


[![Hugging Face's logo](https://huggingface.co/front/assets/huggingface_logo-noborder.svg)Hugging Face](https://huggingface.co/)

- [Models](https://huggingface.co/models)
- [Datasets](https://huggingface.co/datasets)
- [Spaces](https://huggingface.co/spaces)
- Community
    
- [Docs](https://huggingface.co/docs)
- [Pricing](https://huggingface.co/pricing)

- ---
    
- ![](https://huggingface.co/avatars/5718fc9db9d5ef597ef85560419fd2ea.svg)
    

# MCP Course

🏡 View all resourcesAgents CourseAudio CourseCommunity Computer Vision CourseDeep RL CourseDiffusion CourseLLM CourseMCP CourseML for 3D CourseML for Games CourseOpen-Source AI Cookbook

Search documentation

⌘K

EN

 [548](https://github.com/huggingface/mcp-course)

0. Welcome to the MCP Course

[Welcome to the MCP Course](https://huggingface.co/learn/mcp-course/unit0/introduction)

1. Introduction to Model Context Protocol

[Introduction to Model Context Protocol (MCP)](https://huggingface.co/learn/mcp-course/unit1/introduction)[Key Concepts and Terminology](https://huggingface.co/learn/mcp-course/unit1/key-concepts)[Architectural Components](https://huggingface.co/learn/mcp-course/unit1/architectural-components)[Quiz 1 - MCP Fundamentals](https://huggingface.co/learn/mcp-course/unit1/quiz1)[The Communication Protocol](https://huggingface.co/learn/mcp-course/unit1/communication-protocol)[Understanding MCP Capabilities](https://huggingface.co/learn/mcp-course/unit1/capabilities)[MCP SDK](https://huggingface.co/learn/mcp-course/unit1/sdk)[Quiz 2 - MCP SDK](https://huggingface.co/learn/mcp-course/unit1/quiz2)[MCP Clients](https://huggingface.co/learn/mcp-course/unit1/mcp-clients)[Gradio MCP Integration](https://huggingface.co/learn/mcp-course/unit1/gradio-mcp)[Unit 1 Recap](https://huggingface.co/learn/mcp-course/unit1/unit1-recap)[Get your certificate!](https://huggingface.co/learn/mcp-course/unit1/certificate)

2. Use Case: End-to-End MCP Application

3. Use Case: Advanced MCP Development

Bonus Units

# [](https://huggingface.co/learn/mcp-course/unit1/certificate#get-your-certificate)Get your certificate!

Well done! You’ve completed the first unit of the MCP course. Now it’s time to take the exam to get your certificate.

Below is a quiz to check your understanding of the unit.

If you’re struggling to use the quiz above, go to the space directly [on the Hugging Face Hub](https://huggingface.co/spaces/mcp-course/unit_1_quiz). If you find errors, you can report them in the space’s [Community tab](https://huggingface.co/spaces/mcp-course/unit_1_quiz/discussions).

[<>Update on GitHub](https://github.com/huggingface/mcp-course/blob/main/units/en/unit1/certificate.mdx)

Unit1 recap

[←Unit 1 Recap](https://huggingface.co/learn/mcp-course/unit1/unit1-recap)[Introduction to Building an MCP Application→](https://huggingface.co/learn/mcp-course/unit2/introduction)

[Get your certificate!](https://huggingface.co/learn/mcp-course/unit1/certificate#get-your-certificate)


------------


[![Hugging Face's logo](https://huggingface.co/front/assets/huggingface_logo-noborder.svg)Hugging Face](https://huggingface.co/)

- [Models](https://huggingface.co/models)
- [Datasets](https://huggingface.co/datasets)
- [Spaces](https://huggingface.co/spaces)
- Community
    
- [Docs](https://huggingface.co/docs)
- [Pricing](https://huggingface.co/pricing)

- ---
    
- ![](https://huggingface.co/avatars/5718fc9db9d5ef597ef85560419fd2ea.svg)
    

# MCP Course

🏡 View all resourcesAgents CourseAudio CourseCommunity Computer Vision CourseDeep RL CourseDiffusion CourseLLM CourseMCP CourseML for 3D CourseML for Games CourseOpen-Source AI Cookbook

Search documentation

⌘K

EN

 [548](https://github.com/huggingface/mcp-course)

0. Welcome to the MCP Course

[Welcome to the MCP Course](https://huggingface.co/learn/mcp-course/unit0/introduction)

1. Introduction to Model Context Protocol

[Introduction to Model Context Protocol (MCP)](https://huggingface.co/learn/mcp-course/unit1/introduction)[Key Concepts and Terminology](https://huggingface.co/learn/mcp-course/unit1/key-concepts)[Architectural Components](https://huggingface.co/learn/mcp-course/unit1/architectural-components)[Quiz 1 - MCP Fundamentals](https://huggingface.co/learn/mcp-course/unit1/quiz1)[The Communication Protocol](https://huggingface.co/learn/mcp-course/unit1/communication-protocol)[Understanding MCP Capabilities](https://huggingface.co/learn/mcp-course/unit1/capabilities)[MCP SDK](https://huggingface.co/learn/mcp-course/unit1/sdk)[Quiz 2 - MCP SDK](https://huggingface.co/learn/mcp-course/unit1/quiz2)[MCP Clients](https://huggingface.co/learn/mcp-course/unit1/mcp-clients)[Gradio MCP Integration](https://huggingface.co/learn/mcp-course/unit1/gradio-mcp)[Unit 1 Recap](https://huggingface.co/learn/mcp-course/unit1/unit1-recap)[Get your certificate!](https://huggingface.co/learn/mcp-course/unit1/certificate)

2. Use Case: End-to-End MCP Application

3. Use Case: Advanced MCP Development

Bonus Units

# [](https://huggingface.co/learn/mcp-course/unit2/introduction#building-an-end-to-end-mcp-application)Building an End-to-End MCP Application

Welcome to Unit 2 of the MCP Course!

In this unit, we’ll build a complete MCP application from scratch, focusing on creating a server with Gradio and connecting it with multiple clients. This hands-on approach will give you practical experience with the entire MCP ecosystem.

In this unit, we’re going to build a simple MCP server and client using Gradio and the HuggingFace hub. In the next unit, we’ll build a more complex server that tackles a real-world use case.

## [](https://huggingface.co/learn/mcp-course/unit2/introduction#what-youll-learn)What You’ll Learn

In this unit, you will:

- Create an MCP Server using Gradio’s built-in MCP support
- Build a sentiment analysis tool that can be used by AI models
- Connect to the server using different client implementations:
    - A HuggingFace.js-based client
    - A SmolAgents-based client for Python
- Deploy your MCP Server to Hugging Face Spaces
- Test and debug the complete system

By the end of this unit, you’ll have a working MCP application that demonstrates the power and flexibility of the protocol.

## [](https://huggingface.co/learn/mcp-course/unit2/introduction#prerequisites)Prerequisites

Before proceeding with this unit, make sure you:

- Have completed Unit 1 or have a basic understanding of MCP concepts
- Are comfortable with both Python and JavaScript/TypeScript
- Have a basic understanding of APIs and client-server architecture
- Have a development environment with:
    - Python 3.10+
    - Node.js 18+
    - A Hugging Face account (for deployment)

## [](https://huggingface.co/learn/mcp-course/unit2/introduction#our-end-to-end-project)Our End-to-End Project

We’ll build a sentiment analysis application that consists of three main parts: the server, the client, and the deployment.

![sentiment analysis application](https://huggingface.co/datasets/mcp-course/images/resolve/main/unit2/1.png)

### [](https://huggingface.co/learn/mcp-course/unit2/introduction#server-side)Server Side

- Uses Gradio to create a web interface and MCP server via `gr.Interface`
- Implements a sentiment analysis tool using TextBlob
- Exposes the tool through both HTTP and MCP protocols

### [](https://huggingface.co/learn/mcp-course/unit2/introduction#client-side)Client Side

- Implements a HuggingFace.js client
- Or, creates a smolagents Python client
- Demonstrates how to use the same server with different client implementations

### [](https://huggingface.co/learn/mcp-course/unit2/introduction#deployment)Deployment

- Deploys the server to Hugging Face Spaces
- Configures the clients to work with the deployed server

## [](https://huggingface.co/learn/mcp-course/unit2/introduction#lets-get-started)Let’s Get Started!

Are you ready to build your first end-to-end MCP application? Let’s begin by setting up the development environment and creating our Gradio MCP server.

[<>Update on GitHub](https://github.com/huggingface/mcp-course/blob/main/units/en/unit2/introduction.mdx)

Building an End-to-End MCP Application

[←Get your certificate!](https://huggingface.co/learn/mcp-course/unit1/certificate)[Building the Gradio MCP Server→](https://huggingface.co/learn/mcp-course/unit2/gradio-server)

[Building an End-to-End MCP Application](https://huggingface.co/learn/mcp-course/unit2/introduction#building-an-end-to-end-mcp-application)[What You’ll Learn](https://huggingface.co/learn/mcp-course/unit2/introduction#what-youll-learn)[Prerequisites](https://huggingface.co/learn/mcp-course/unit2/introduction#prerequisites)[Our End-to-End Project](https://huggingface.co/learn/mcp-course/unit2/introduction#our-end-to-end-project)[Server Side](https://huggingface.co/learn/mcp-course/unit2/introduction#server-side)[Client Side](https://huggingface.co/learn/mcp-course/unit2/introduction#client-side)[Deployment](https://huggingface.co/learn/mcp-course/unit2/introduction#deployment)[Let’s Get Started!](https://huggingface.co/learn/mcp-course/unit2/introduction#lets-get-started)


-------------


[![Hugging Face's logo](https://huggingface.co/front/assets/huggingface_logo-noborder.svg)Hugging Face](https://huggingface.co/)

- [Models](https://huggingface.co/models)
- [Datasets](https://huggingface.co/datasets)
- [Spaces](https://huggingface.co/spaces)
- Community
    
- [Docs](https://huggingface.co/docs)
- [Pricing](https://huggingface.co/pricing)

- ---
    
- ![](https://huggingface.co/avatars/5718fc9db9d5ef597ef85560419fd2ea.svg)
    

# MCP Course

🏡 View all resourcesAgents CourseAudio CourseCommunity Computer Vision CourseDeep RL CourseDiffusion CourseLLM CourseMCP CourseML for 3D CourseML for Games CourseOpen-Source AI Cookbook

Search documentation

⌘K

EN

 [548](https://github.com/huggingface/mcp-course)

0. Welcome to the MCP Course

[Welcome to the MCP Course](https://huggingface.co/learn/mcp-course/unit0/introduction)

1. Introduction to Model Context Protocol

[Introduction to Model Context Protocol (MCP)](https://huggingface.co/learn/mcp-course/unit1/introduction)[Key Concepts and Terminology](https://huggingface.co/learn/mcp-course/unit1/key-concepts)[Architectural Components](https://huggingface.co/learn/mcp-course/unit1/architectural-components)[Quiz 1 - MCP Fundamentals](https://huggingface.co/learn/mcp-course/unit1/quiz1)[The Communication Protocol](https://huggingface.co/learn/mcp-course/unit1/communication-protocol)[Understanding MCP Capabilities](https://huggingface.co/learn/mcp-course/unit1/capabilities)[MCP SDK](https://huggingface.co/learn/mcp-course/unit1/sdk)[Quiz 2 - MCP SDK](https://huggingface.co/learn/mcp-course/unit1/quiz2)[MCP Clients](https://huggingface.co/learn/mcp-course/unit1/mcp-clients)[Gradio MCP Integration](https://huggingface.co/learn/mcp-course/unit1/gradio-mcp)[Unit 1 Recap](https://huggingface.co/learn/mcp-course/unit1/unit1-recap)[Get your certificate!](https://huggingface.co/learn/mcp-course/unit1/certificate)

2. Use Case: End-to-End MCP Application

3. Use Case: Advanced MCP Development

Bonus Units

# [](https://huggingface.co/learn/mcp-course/unit2/gradio-server#building-the-gradio-mcp-server)Building the Gradio MCP Server

In this section, we’ll create our sentiment analysis MCP server using Gradio. This server will expose a sentiment analysis tool that can be used by both human users through a web interface and AI models through the MCP protocol.

## [](https://huggingface.co/learn/mcp-course/unit2/gradio-server#introduction-to-gradio-mcp-integration)Introduction to Gradio MCP Integration

Gradio provides a straightforward way to create MCP servers by automatically converting your Python functions into MCP tools. When you set `mcp_server=True` in `launch()`, Gradio:

1. Automatically converts your functions into MCP Tools
2. Maps input components to tool argument schemas
3. Determines response formats from output components
4. Sets up JSON-RPC over HTTP+SSE for client-server communication
5. Creates both a web interface and an MCP server endpoint

## [](https://huggingface.co/learn/mcp-course/unit2/gradio-server#setting-up-the-project)Setting Up the Project

First, let’s create a new directory for our project and set up the required dependencies:

Copied

mkdir mcp-sentiment
cd mcp-sentiment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install "gradio[mcp]" textblob

## [](https://huggingface.co/learn/mcp-course/unit2/gradio-server#creating-the-server)Creating the Server

> Hugging face spaces needs an app.py file to build the space. So the name of the python file has to be app.py

Create a new file called `app.py` with the following code:

Copied

import gradio as gr
from textblob import TextBlob

def sentiment_analysis(text: str) -> dict:
    """
    Analyze the sentiment of the given text.

    Args:
        text (str): The text to analyze

    Returns:
        dict: A dictionary containing polarity, subjectivity, and assessment
    """
    blob = TextBlob(text)
    sentiment = blob.sentiment
    
    return {
        "polarity": round(sentiment.polarity, 2),  # -1 (negative) to 1 (positive)
        "subjectivity": round(sentiment.subjectivity, 2),  # 0 (objective) to 1 (subjective)
        "assessment": "positive" if sentiment.polarity > 0 else "negative" if sentiment.polarity < 0 else "neutral"
    }

# Create the Gradio interface
demo = gr.Interface(
    fn=sentiment_analysis,
    inputs=gr.Textbox(placeholder="Enter text to analyze..."),
    outputs=gr.JSON(),
    title="Text Sentiment Analysis",
    description="Analyze the sentiment of text using TextBlob"
)

# Launch the interface and MCP server
if __name__ == "__main__":
    demo.launch(mcp_server=True)

## [](https://huggingface.co/learn/mcp-course/unit2/gradio-server#understanding-the-code)Understanding the Code

Let’s break down the key components:

1. **Function Definition**:
    
    - The `sentiment_analysis` function takes a text input and returns a dictionary
    - It uses TextBlob to analyze the sentiment
    - The docstring is crucial as it helps Gradio generate the MCP tool schema
    - Type hints (`str` and `dict`) help define the input/output schema
2. **Gradio Interface**:
    
    - `gr.Interface` creates both the web UI and MCP server
    - The function is exposed as an MCP tool automatically
    - Input and output components define the tool’s schema
    - The JSON output component ensures proper serialization
3. **MCP Server**:
    
    - Setting `mcp_server=True` enables the MCP server
    - The server will be available at `http://localhost:7860/gradio_api/mcp/sse`
    - You can also enable it using the environment variable:
        
        Copied
        
        export GRADIO_MCP_SERVER=True
        

## [](https://huggingface.co/learn/mcp-course/unit2/gradio-server#running-the-server)Running the Server

Start the server by running:

Copied

python app.py

You should see output indicating that both the web interface and MCP server are running. The web interface will be available at `http://localhost:7860`, and the MCP server at `http://localhost:7860/gradio_api/mcp/sse`.

## [](https://huggingface.co/learn/mcp-course/unit2/gradio-server#testing-the-server)Testing the Server

You can test the server in two ways:

1. **Web Interface**:
    
    - Open `http://localhost:7860` in your browser
    - Enter some text and click “Submit”
    - You should see the sentiment analysis results
2. **MCP Schema**:
    
    - Visit `http://localhost:7860/gradio_api/mcp/schema`
    - This shows the MCP tool schema that clients will use
    - You can also find this in the “View API” link in the footer of your Gradio app

## [](https://huggingface.co/learn/mcp-course/unit2/gradio-server#troubleshooting-tips)Troubleshooting Tips

1. **Type Hints and Docstrings**:
    
    - Always provide type hints for your function parameters and return values
    - Include a docstring with an “Args:” block for each parameter
    - This helps Gradio generate accurate MCP tool schemas
2. **String Inputs**:
    
    - When in doubt, accept input arguments as `str`
    - Convert them to the desired type inside the function
    - This provides better compatibility with MCP clients
3. **SSE Support**:
    
    - Some MCP clients don’t support SSE-based MCP Servers
    - In those cases, use `mcp-remote`:
        
        Copied
        
        {
          "mcpServers": {
            "gradio": {
              "command": "npx",
              "args": [
                "mcp-remote",
                "http://localhost:7860/gradio_api/mcp/sse"
              ]
            }
          }
        }
        
4. **Connection Issues**:
    
    - If you encounter connection problems, try restarting both the client and server
    - Check that the server is running and accessible
    - Verify that the MCP schema is available at the expected URL

## [](https://huggingface.co/learn/mcp-course/unit2/gradio-server#deploying-to-hugging-face-spaces)Deploying to Hugging Face Spaces

To make your server available to others, you can deploy it to Hugging Face Spaces:

1. Create a new Space on Hugging Face:
    
    - Go to huggingface.co/spaces
    - Click “Create new Space”
    - Choose “Gradio” as the SDK
    - Name your space (e.g., “mcp-sentiment”)
2. Create a `requirements.txt` file:
    

Copied

gradio[mcp]
textblob

3. Push your code to the Space:

Copied

git init
git add app.py requirements.txt
git commit -m "Initial commit"
git remote add origin https://huggingface.co/spaces/YOUR_USERNAME/mcp-sentiment
git push -u origin main

Your MCP server will now be available at:

Copied

https://YOUR_USERNAME-mcp-sentiment.hf.space/gradio_api/mcp/sse

## [](https://huggingface.co/learn/mcp-course/unit2/gradio-server#next-steps)Next Steps

Now that we have our MCP server running, we’ll create clients to interact with it. In the next sections, we’ll:

1. Create a HuggingFace.js-based client inspired by Tiny Agents
2. Implement a SmolAgents-based Python client
3. Test both clients with our deployed server

Let’s move on to building our first client!

[<>Update on GitHub](https://github.com/huggingface/mcp-course/blob/main/units/en/unit2/gradio-server.mdx)

Building the Gradio MCP Server

[←Introduction to Building an MCP Application](https://huggingface.co/learn/mcp-course/unit2/introduction)[Using MCP Clients with your application→](https://huggingface.co/learn/mcp-course/unit2/clients)

[Building the Gradio MCP Server](https://huggingface.co/learn/mcp-course/unit2/gradio-server#building-the-gradio-mcp-server)[Introduction to Gradio MCP Integration](https://huggingface.co/learn/mcp-course/unit2/gradio-server#introduction-to-gradio-mcp-integration)[Setting Up the Project](https://huggingface.co/learn/mcp-course/unit2/gradio-server#setting-up-the-project)[Creating the Server](https://huggingface.co/learn/mcp-course/unit2/gradio-server#creating-the-server)[Understanding the Code](https://huggingface.co/learn/mcp-course/unit2/gradio-server#understanding-the-code)[Running the Server](https://huggingface.co/learn/mcp-course/unit2/gradio-server#running-the-server)[Testing the Server](https://huggingface.co/learn/mcp-course/unit2/gradio-server#testing-the-server)[Troubleshooting Tips](https://huggingface.co/learn/mcp-course/unit2/gradio-server#troubleshooting-tips)[Deploying to Hugging Face Spaces](https://huggingface.co/learn/mcp-course/unit2/gradio-server#deploying-to-hugging-face-spaces)[Next Steps](https://huggingface.co/learn/mcp-course/unit2/gradio-server#next-steps)


------------


[![Hugging Face's logo](https://huggingface.co/front/assets/huggingface_logo-noborder.svg)Hugging Face](https://huggingface.co/)

- [Models](https://huggingface.co/models)
- [Datasets](https://huggingface.co/datasets)
- [Spaces](https://huggingface.co/spaces)
- Community
    
- [Docs](https://huggingface.co/docs)
- [Pricing](https://huggingface.co/pricing)

- ---
    
- ![](https://huggingface.co/avatars/5718fc9db9d5ef597ef85560419fd2ea.svg)
    

# MCP Course

🏡 View all resourcesAgents CourseAudio CourseCommunity Computer Vision CourseDeep RL CourseDiffusion CourseLLM CourseMCP CourseML for 3D CourseML for Games CourseOpen-Source AI Cookbook

Search documentation

⌘K

EN

 [548](https://github.com/huggingface/mcp-course)

0. Welcome to the MCP Course

[Welcome to the MCP Course](https://huggingface.co/learn/mcp-course/unit0/introduction)

1. Introduction to Model Context Protocol

[Introduction to Model Context Protocol (MCP)](https://huggingface.co/learn/mcp-course/unit1/introduction)[Key Concepts and Terminology](https://huggingface.co/learn/mcp-course/unit1/key-concepts)[Architectural Components](https://huggingface.co/learn/mcp-course/unit1/architectural-components)[Quiz 1 - MCP Fundamentals](https://huggingface.co/learn/mcp-course/unit1/quiz1)[The Communication Protocol](https://huggingface.co/learn/mcp-course/unit1/communication-protocol)[Understanding MCP Capabilities](https://huggingface.co/learn/mcp-course/unit1/capabilities)[MCP SDK](https://huggingface.co/learn/mcp-course/unit1/sdk)[Quiz 2 - MCP SDK](https://huggingface.co/learn/mcp-course/unit1/quiz2)[MCP Clients](https://huggingface.co/learn/mcp-course/unit1/mcp-clients)[Gradio MCP Integration](https://huggingface.co/learn/mcp-course/unit1/gradio-mcp)[Unit 1 Recap](https://huggingface.co/learn/mcp-course/unit1/unit1-recap)[Get your certificate!](https://huggingface.co/learn/mcp-course/unit1/certificate)

2. Use Case: End-to-End MCP Application

3. Use Case: Advanced MCP Development

Bonus Units

# [](https://huggingface.co/learn/mcp-course/unit2/clients#building-mcp-clients)Building MCP Clients

In this section, we’ll create clients that can interact with our MCP server using different programming languages. We’ll implement both a JavaScript client using HuggingFace.js and a Python client using smolagents.

## [](https://huggingface.co/learn/mcp-course/unit2/clients#configuring-mcp-clients)Configuring MCP Clients

Effective deployment of MCP servers and clients requires proper configuration. The MCP specification is still evolving, so the configuration methods are subject to evolution. We’ll focus on the current best practices for configuration.

### [](https://huggingface.co/learn/mcp-course/unit2/clients#mcp-configuration-files)MCP Configuration Files

MCP hosts use configuration files to manage server connections. These files define which servers are available and how to connect to them.

The configuration files are very simple, easy to understand, and consistent across major MCP hosts.

#### [](https://huggingface.co/learn/mcp-course/unit2/clients#mcpjson-structure)mcp.json Structure

The standard configuration file for MCP is named `mcp.json`. Here’s the basic structure:

Copied

{
  "servers": [
    {
      "name": "MCP Server",
      "transport": {
        "type": "sse",
        "url": "http://localhost:7860/gradio_api/mcp/sse"
      }
    }
  ]
}

In this example, we have a single server configured to use SSE transport, connecting to a local Gradio server running on port 7860.

We’ve connected to the Gradio app via SSE transport because we assume that the gradio app is running on a remote server. However, if you want to connect to a local script, `stdio` transport instead of `sse` transport is a better option.

#### [](https://huggingface.co/learn/mcp-course/unit2/clients#configuration-for-httpsse-transport)Configuration for HTTP+SSE Transport

For remote servers using HTTP+SSE transport, the configuration includes the server URL:

Copied

{
  "servers": [
    {
      "name": "Remote MCP Server",
      "transport": {
        "type": "sse",
        "url": "https://example.com/gradio_api/mcp/sse"
      }
    }
  ]
}

This configuration allows your UI client to communicate with the Gradio MCP server using the MCP protocol, enabling seamless integration between your frontend and the MCP service.

## [](https://huggingface.co/learn/mcp-course/unit2/clients#configuring-a-ui-mcp-client)Configuring a UI MCP Client

When working with Gradio MCP servers, you can configure your UI client to connect to the server using the MCP protocol. Here’s how to set it up:

### [](https://huggingface.co/learn/mcp-course/unit2/clients#basic-configuration)Basic Configuration

Create a new file called `config.json` with the following configuration:

Copied

{
  "mcpServers": {
    "mcp": {
      "url": "http://localhost:7860/gradio_api/mcp/sse"
    }
  }
}

This configuration allows your UI client to communicate with the Gradio MCP server using the MCP protocol, enabling seamless integration between your frontend and the MCP service.

## [](https://huggingface.co/learn/mcp-course/unit2/clients#configuring-a-mcp-client-within-cursor-ide)Configuring a MCP Client within Cursor IDE

Cursor provides built-in MCP support, allowing you to connect your deployed MCP servers directly to your development environment.

### [](https://huggingface.co/learn/mcp-course/unit2/clients#configuration)Configuration

Open Cursor settings (`Ctrl + Shift + J` / `Cmd + Shift + J`) → **MCP** tab → **Add new global MCP server**:

**macOS:**

Copied

{
  "mcpServers": {
    "sentiment-analysis": {
      "command": "npx",
      "args": [
        "-y", 
        "mcp-remote", 
        "https://YOURUSENAME-mcp-sentiment.hf.space/gradio_api/mcp/sse", 
        "--transport", 
        "sse-only"
      ]
    }
  }
}

**Windows:**

Copied

{
  "mcpServers": {
    "sentiment-analysis": {
      "command": "cmd",
      "args": [
        "/c", 
        "npx", 
        "-y", 
        "mcp-remote", 
        "https://YOURUSENAME-mcp-sentiment.hf.space/gradio_api/mcp/sse", 
        "--transport", 
        "sse-only"
      ]
    }
  }
}

### [](https://huggingface.co/learn/mcp-course/unit2/clients#why-we-use-mcp-remote)Why We Use mcp-remote

Most MCP clients, including Cursor, currently only support local servers via stdio transport and don’t yet support remote servers with OAuth authentication. The `mcp-remote` tool serves as a bridge solution that:

- Runs locally on your machine
- Forwards requests from Cursor to the remote MCP server
- Uses the familiar configuration file format

Once configured, you can ask Cursor to use your sentiment analysis tool for tasks like analyzing code comments, user feedback, or pull request descriptions.

[<>Update on GitHub](https://github.com/huggingface/mcp-course/blob/main/units/en/unit2/clients.mdx)

Building the Gradio MCP Server

[←Building the Gradio MCP Server](https://huggingface.co/learn/mcp-course/unit2/gradio-server)[Building an MCP Client with Gradio→](https://huggingface.co/learn/mcp-course/unit2/gradio-client)

[Building MCP Clients](https://huggingface.co/learn/mcp-course/unit2/clients#building-mcp-clients)[Configuring MCP Clients](https://huggingface.co/learn/mcp-course/unit2/clients#configuring-mcp-clients)[MCP Configuration Files](https://huggingface.co/learn/mcp-course/unit2/clients#mcp-configuration-files)[mcp.json Structure](https://huggingface.co/learn/mcp-course/unit2/clients#mcpjson-structure)[Configuration for HTTP+SSE Transport](https://huggingface.co/learn/mcp-course/unit2/clients#configuration-for-httpsse-transport)[Configuring a UI MCP Client](https://huggingface.co/learn/mcp-course/unit2/clients#configuring-a-ui-mcp-client)[Basic Configuration](https://huggingface.co/learn/mcp-course/unit2/clients#basic-configuration)[Configuring a MCP Client within Cursor IDE](https://huggingface.co/learn/mcp-course/unit2/clients#configuring-a-mcp-client-within-cursor-ide)[Configuration](https://huggingface.co/learn/mcp-course/unit2/clients#configuration)[Why We Use mcp-remote](https://huggingface.co/learn/mcp-course/unit2/clients#why-we-use-mcp-remote)


----------

[![Hugging Face's logo](https://huggingface.co/front/assets/huggingface_logo-noborder.svg)Hugging Face](https://huggingface.co/)

- [Models](https://huggingface.co/models)
- [Datasets](https://huggingface.co/datasets)
- [Spaces](https://huggingface.co/spaces)
- Community
    
- [Docs](https://huggingface.co/docs)
- [Pricing](https://huggingface.co/pricing)

- ---
    
- ![](https://huggingface.co/avatars/5718fc9db9d5ef597ef85560419fd2ea.svg)
    

# MCP Course

🏡 View all resourcesAgents CourseAudio CourseCommunity Computer Vision CourseDeep RL CourseDiffusion CourseLLM CourseMCP CourseML for 3D CourseML for Games CourseOpen-Source AI Cookbook

Search documentation

⌘K

EN

 [548](https://github.com/huggingface/mcp-course)

0. Welcome to the MCP Course

[Welcome to the MCP Course](https://huggingface.co/learn/mcp-course/unit0/introduction)

1. Introduction to Model Context Protocol

[Introduction to Model Context Protocol (MCP)](https://huggingface.co/learn/mcp-course/unit1/introduction)[Key Concepts and Terminology](https://huggingface.co/learn/mcp-course/unit1/key-concepts)[Architectural Components](https://huggingface.co/learn/mcp-course/unit1/architectural-components)[Quiz 1 - MCP Fundamentals](https://huggingface.co/learn/mcp-course/unit1/quiz1)[The Communication Protocol](https://huggingface.co/learn/mcp-course/unit1/communication-protocol)[Understanding MCP Capabilities](https://huggingface.co/learn/mcp-course/unit1/capabilities)[MCP SDK](https://huggingface.co/learn/mcp-course/unit1/sdk)[Quiz 2 - MCP SDK](https://huggingface.co/learn/mcp-course/unit1/quiz2)[MCP Clients](https://huggingface.co/learn/mcp-course/unit1/mcp-clients)[Gradio MCP Integration](https://huggingface.co/learn/mcp-course/unit1/gradio-mcp)[Unit 1 Recap](https://huggingface.co/learn/mcp-course/unit1/unit1-recap)[Get your certificate!](https://huggingface.co/learn/mcp-course/unit1/certificate)

2. Use Case: End-to-End MCP Application

3. Use Case: Advanced MCP Development

Bonus Units

# [](https://huggingface.co/learn/mcp-course/unit2/gradio-client#gradio-as-an-mcp-client)Gradio as an MCP Client

In the previous section, we explored how to create an MCP Server using Gradio and connect to it using an MCP Client. In this section, we’re going to explore how to use Gradio as an MCP Client to connect to an MCP Server.

Gradio is best suited to the creation of UI clients and MCP servers, but it is also possible to use it as an MCP Client and expose that as a UI.

We’ll connect to the MCP server we created in the previous section and use it to answer questions.

## [](https://huggingface.co/learn/mcp-course/unit2/gradio-client#mcp-client-in-gradio)MCP Client in Gradio

### [](https://huggingface.co/learn/mcp-course/unit2/gradio-client#connect-to-an-example-mcp-server)Connect to an example MCP Server

Let’s connect to an example MCP Server that is already running on Hugging Face. We’ll use [this one](https://huggingface.co/spaces/abidlabs/mcp-tools2) for this example. It’s a space that contains a collection of MCP tools.

Copied

from smolagents.mcp_client import MCPClient

with MCPClient(
    {"url": "https://abidlabs-mcp-tools2.hf.space/gradio_api/mcp/sse"}
) as tools:
    # Tools from the remote server are available
    print("\n".join(f"{t.name}: {t.description}" for t in tools))

Output

### [](https://huggingface.co/learn/mcp-course/unit2/gradio-client#connect-to-your-mcp-server-from-gradio)Connect to your MCP Server from Gradio

Great, now that you’ve connected to an example MCP Server, let’s connect to your own MCP Server from Gradio.

First, we need to install the `smolagents`, Gradio and mcp-client libraries, if we haven’t already:

Copied

pip install "smolagents[mcp]" "gradio[mcp]" mcp fastmcp

Now, we can import the necessary libraries and create a simple Gradio interface that uses the MCP Client to connect to the MCP Server.

Copied

import gradio as gr
import os

from mcp import StdioServerParameters
from smolagents import InferenceClientModel, CodeAgent, ToolCollection, MCPClient

Next, we’ll connect to the MCP Server and get the tools that we can use to answer questions.

Copied

mcp_client = MCPClient(
    {"url": "http://localhost:7860/gradio_api/mcp/sse"} # This is the MCP Server we created in the previous section
)
tools = mcp_client.get_tools()

Now that we have the tools, we can create a simple agent that uses them to answer questions. We’ll just use a simple `InferenceClientModel` and the default model from `smolagents` for now.

It is important to pass your api_key to the InferenceClientModel. You can access the token from your huggingface account. [check here.](https://huggingface.co/docs/hub/en/security-tokens), and set the access token with the environment variable `HF_TOKEN`.

Copied

model = InferenceClientModel(token=os.getenv("HF_TOKEN"))
agent = CodeAgent(tools=[*tools], model=model)

Now, we can create a simple Gradio interface that uses the agent to answer questions.

Copied

demo = gr.ChatInterface(
    fn=lambda message, history: str(agent.run(message)),
    type="messages",
    examples=["Prime factorization of 68"],
    title="Agent with MCP Tools",
    description="This is a simple agent that uses MCP tools to answer questions."
)

demo.launch()

And that’s it! We’ve created a simple Gradio interface that uses the MCP Client to connect to the MCP Server and answer questions.

## [](https://huggingface.co/learn/mcp-course/unit2/gradio-client#complete-example)Complete Example

Here’s the complete example of the MCP Client in Gradio:

Copied

import gradio as gr
import os

from mcp import StdioServerParameters
from smolagents import InferenceClientModel, CodeAgent, ToolCollection, MCPClient

try:
    mcp_client = MCPClient(
        {"url": "http://localhost:7860/gradio_api/mcp/sse"} # This is the MCP Server we created in the previous section
    )
    tools = mcp_client.get_tools()

    model = InferenceClientModel(token=os.getenv("HUGGINGFACE_API_TOKEN"))
    agent = CodeAgent(tools=[*tools], model=model)

    demo = gr.ChatInterface(
        fn=lambda message, history: str(agent.run(message)),
        type="messages",
        examples=["Prime factorization of 68"],
        title="Agent with MCP Tools",
        description="This is a simple agent that uses MCP tools to answer questions.",
    )

    demo.launch()
finally:
    mcp_client.disconnect()

You’ll notice that we’re closing the MCP Client in the `finally` block. This is important because the MCP Client is a long-lived object that needs to be closed when the program exits.

## [](https://huggingface.co/learn/mcp-course/unit2/gradio-client#deploying-to-hugging-face-spaces)Deploying to Hugging Face Spaces

To make your server available to others, you can deploy it to Hugging Face Spaces, just like we did in the previous section. To deploy your Gradio MCP client to Hugging Face Spaces:

1. Create a new Space on Hugging Face:
    
    - Go to huggingface.co/spaces
    - Click “Create new Space”
    - Choose “Gradio” as the SDK
    - Name your space (e.g., “mcp-client”)
2. Update MCP Server URL in the code:
    

Copied

mcp_client = MCPClient(
    {"url": "http://localhost:7860/gradio_api/mcp/sse"} # This is the MCP Server we created in the previous section
)

3. Create a `requirements.txt` file:

Copied

gradio[mcp]
smolagents[mcp]

4. Push your code to the Space:

Copied

git init
git add server.py requirements.txt
git commit -m "Initial commit"
git remote add origin https://huggingface.co/spaces/YOUR_USERNAME/mcp-client
git push -u origin main

## [](https://huggingface.co/learn/mcp-course/unit2/gradio-client#conclusion)Conclusion

In this section, we’ve explored how to use Gradio as an MCP Client to connect to an MCP Server. We’ve also seen how to deploy the MCP Client in Hugging Face Spaces.

[<>Update on GitHub](https://github.com/huggingface/mcp-course/blob/main/units/en/unit2/gradio-client.mdx)

Gradio as an MCP Client

[←Using MCP Clients with your application](https://huggingface.co/learn/mcp-course/unit2/clients)[Building Tiny Agents with MCP and the Hugging Face Hub→](https://huggingface.co/learn/mcp-course/unit2/tiny-agents)

[Gradio as an MCP Client](https://huggingface.co/learn/mcp-course/unit2/gradio-client#gradio-as-an-mcp-client)[MCP Client in Gradio](https://huggingface.co/learn/mcp-course/unit2/gradio-client#mcp-client-in-gradio)[Connect to an example MCP Server](https://huggingface.co/learn/mcp-course/unit2/gradio-client#connect-to-an-example-mcp-server)[Connect to your MCP Server from Gradio](https://huggingface.co/learn/mcp-course/unit2/gradio-client#connect-to-your-mcp-server-from-gradio)[Complete Example](https://huggingface.co/learn/mcp-course/unit2/gradio-client#complete-example)[Deploying to Hugging Face Spaces](https://huggingface.co/learn/mcp-course/unit2/gradio-client#deploying-to-hugging-face-spaces)[Conclusion](https://huggingface.co/learn/mcp-course/unit2/gradio-client#conclusion)


---------


[![Hugging Face's logo](https://huggingface.co/front/assets/huggingface_logo-noborder.svg)Hugging Face](https://huggingface.co/)

- [Models](https://huggingface.co/models)
- [Datasets](https://huggingface.co/datasets)
- [Spaces](https://huggingface.co/spaces)
- Community
    
- [Docs](https://huggingface.co/docs)
- [Pricing](https://huggingface.co/pricing)

- ---
    
- ![](https://huggingface.co/avatars/5718fc9db9d5ef597ef85560419fd2ea.svg)
    

# MCP Course

🏡 View all resourcesAgents CourseAudio CourseCommunity Computer Vision CourseDeep RL CourseDiffusion CourseLLM CourseMCP CourseML for 3D CourseML for Games CourseOpen-Source AI Cookbook

Search documentation

⌘K

EN

 [548](https://github.com/huggingface/mcp-course)

0. Welcome to the MCP Course

[Welcome to the MCP Course](https://huggingface.co/learn/mcp-course/unit0/introduction)

1. Introduction to Model Context Protocol

[Introduction to Model Context Protocol (MCP)](https://huggingface.co/learn/mcp-course/unit1/introduction)[Key Concepts and Terminology](https://huggingface.co/learn/mcp-course/unit1/key-concepts)[Architectural Components](https://huggingface.co/learn/mcp-course/unit1/architectural-components)[Quiz 1 - MCP Fundamentals](https://huggingface.co/learn/mcp-course/unit1/quiz1)[The Communication Protocol](https://huggingface.co/learn/mcp-course/unit1/communication-protocol)[Understanding MCP Capabilities](https://huggingface.co/learn/mcp-course/unit1/capabilities)[MCP SDK](https://huggingface.co/learn/mcp-course/unit1/sdk)[Quiz 2 - MCP SDK](https://huggingface.co/learn/mcp-course/unit1/quiz2)[MCP Clients](https://huggingface.co/learn/mcp-course/unit1/mcp-clients)[Gradio MCP Integration](https://huggingface.co/learn/mcp-course/unit1/gradio-mcp)[Unit 1 Recap](https://huggingface.co/learn/mcp-course/unit1/unit1-recap)[Get your certificate!](https://huggingface.co/learn/mcp-course/unit1/certificate)

2. Use Case: End-to-End MCP Application

[Introduction to Building an MCP Application](https://huggingface.co/learn/mcp-course/unit2/introduction)[Building the Gradio MCP Server](https://huggingface.co/learn/mcp-course/unit2/gradio-server)[Using MCP Clients with your application](https://huggingface.co/learn/mcp-course/unit2/clients)[Building an MCP Client with Gradio](https://huggingface.co/learn/mcp-course/unit2/gradio-client)[Building Tiny Agents with MCP and the Hugging Face Hub](https://huggingface.co/learn/mcp-course/unit2/tiny-agents)

3. Use Case: Advanced MCP Development

Bonus Units

# [](https://huggingface.co/learn/mcp-course/unit2/tiny-agents#building-tiny-agents-with-mcp-and-the-hugging-face-hub)Building Tiny Agents with MCP and the Hugging Face Hub

Now that we’ve built MCP servers in Gradio and learned about creating MCP clients, let’s complete our end-to-end application by building an agent that can seamlessly interact with our sentiment analysis tool. This section builds on the project [Tiny Agents](https://huggingface.co/blog/tiny-agents), which demonstrates a super simple way of deploying MCP clients that can connect to services like our Gradio sentiment analysis server.

In this final exercise of Unit 2, we will walk you through how to implement both TypeScript (JS) and Python MCP clients that can communicate with any MCP server, including the Gradio-based sentiment analysis server we built in the previous sections. This completes our end-to-end MCP application flow: from building a Gradio MCP server exposing a sentiment analysis tool, to creating a flexible agent that can use this tool alongside other capabilities.

![meme](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/tiny-agents/thumbnail.jpg)

Image credit https://x.com/adamdotdev

## [](https://huggingface.co/learn/mcp-course/unit2/tiny-agents#installation)Installation

Let’s install the necessary packages to build our Tiny Agents.

Some MCP Clients, notably Claude Desktop, do not yet support SSE-based MCP Servers. In those cases, you can use a tool such as [mcp-remote](https://github.com/geelen/mcp-remote). First install Node.js. Then, add the following to your own MCP Client config:

Tiny Agent can run MCP servers with a command line environment. To do this, we will need to install `npm` and run the server with `npx`. **We’ll need these for both Python and JavaScript.**

Let’s install `npx` with `npm`. If you don’t have `npm` installed, check out the [npm documentation](https://docs.npmjs.com/downloading-and-installing-node-js-and-npm).

Copied

# install npx
npm install -g npx

Then, we need to install the `mcp-remote` package.

Copied

npm i mcp-remote

typescript

python

For JavaScript, we need to install the `tiny-agents` package.

Copied

npm install @huggingface/tiny-agents

## [](https://huggingface.co/learn/mcp-course/unit2/tiny-agents#tiny-agents-mcp-client-in-the-command-line)Tiny Agents MCP Client in the Command Line

Let’s repeat the example from [Unit 1](https://huggingface.co/learn/mcp-course/unit1/mcp-clients.mdx) to create a basic Tiny Agent. Tiny Agents can create MCP clients from the command line based on JSON configuration files.

typescript

python

Let’s setup a project with a basic Tiny Agent.

Copied

mkdir my-agent
touch my-agent/agent.json

The JSON file will look like this:

Copied

{
	"model": "Qwen/Qwen2.5-72B-Instruct",
	"provider": "nebius",
	"servers": [
		{
			"type": "stdio",
			"config": {
				"command": "npx",
				"args": [
					"mcp-remote",
					"http://localhost:7860/gradio_api/mcp/sse" // This is the MCP Server we created in the previous section
				]
			}
		}
	]
}

We can then run the agent with the following command:

Copied

npx @huggingface/tiny-agents run ./my-agent

Here we have a basic Tiny Agent that can connect to our Gradio MCP server. It includes a model, provider, and a server configuration.

|Field|Description|
|---|---|
|`model`|The open source model to use for the agent|
|`provider`|The inference provider to use for the agent|
|`servers`|The servers to use for the agent. We’ll use the `mcp-remote` server for our Gradio MCP server.|

We could also use an open source model running locally with Tiny Agents. If we start a local inference server with

Copied

{
	"model": "Qwen/Qwen3-32B",
	"endpointUrl": "http://localhost:1234/v1",
	"servers": [
		{
			"type": "stdio",
			"config": {
				"command": "npx",
				"args": [
					"mcp-remote",
					"http://localhost:1234/v1/mcp/sse"
				]
			}
		}
	]
}

Here we have a Tiny Agent that can connect to a local model. It includes a model, endpoint URL (`http://localhost:1234/v1`), and a server configuration. The endpoint should be an OpenAI-compatible endpoint.

## [](https://huggingface.co/learn/mcp-course/unit2/tiny-agents#custom-tiny-agents-mcp-client)Custom Tiny Agents MCP Client

Now that we understand both Tiny Agents and Gradio MCP servers, let’s see how they work together! The beauty of MCP is that it provides a standardized way for agents to interact with any MCP-compatible server, including our Gradio-based sentiment analysis server from earlier sections.

### [](https://huggingface.co/learn/mcp-course/unit2/tiny-agents#using-the-gradio-server-with-tiny-agents)Using the Gradio Server with Tiny Agents

To connect our Tiny Agent to the Gradio sentiment analysis server we built earlier in this unit, we just need to add it to our list of servers. Here’s how we can modify our agent configuration:

typescript

python

Copied

const agent = new Agent({
    provider: process.env.PROVIDER ?? "nebius",
    model: process.env.MODEL_ID ?? "Qwen/Qwen2.5-72B-Instruct",
    apiKey: process.env.HF_TOKEN,
    servers: [
        // ... existing servers ...
        {
            command: "npx",
            args: [
                "mcp-remote",
                "http://localhost:7860/gradio_api/mcp/sse"  // Your Gradio MCP server
            ]
        }
    ],
});

Now our agent can use the sentiment analysis tool alongside other tools! For example, it could:

1. Read text from a file using the filesystem server
2. Analyze its sentiment using our Gradio server
3. Write the results back to a file

### [](https://huggingface.co/learn/mcp-course/unit2/tiny-agents#deployment-considerations)Deployment Considerations

When deploying your Gradio MCP server to Hugging Face Spaces, you’ll need to update the server URL in your agent configuration to point to your deployed space:

Copied

{
    command: "npx",
    args: [
        "mcp-remote",
        "https://YOUR_USERNAME-mcp-sentiment.hf.space/gradio_api/mcp/sse"
    ]
}

This allows your agent to use the sentiment analysis tool from anywhere, not just locally!

## [](https://huggingface.co/learn/mcp-course/unit2/tiny-agents#conclusion-our-complete-end-to-end-mcp-application)Conclusion: Our Complete End-to-End MCP Application

In this unit, we’ve gone from understanding MCP basics to building a complete end-to-end application:

1. We created a Gradio MCP server that exposes a sentiment analysis tool
2. We learned how to connect to this server using MCP clients
3. We built a tiny agent in TypeScript and Python that can interact with our tool

This demonstrates the power of the Model Context Protocol - we can create specialized tools using frameworks we’re familiar with (like Gradio), expose them through a standardized interface (MCP), and then have agents seamlessly use these tools alongside other capabilities.

The complete flow we’ve built allows an agent to:

- Connect to multiple tool providers
- Dynamically discover available tools
- Use our custom sentiment analysis tool
- Combine it with other capabilities like file system access and web browsing

This modular approach is what makes MCP so powerful for building flexible AI applications.

## [](https://huggingface.co/learn/mcp-course/unit2/tiny-agents#next-steps)Next Steps

- Check out the Tiny Agents blog posts in [Python](https://huggingface.co/blog/python-tiny-agents) and [TypeScript](https://huggingface.co/blog/tiny-agents)
- Review the [Tiny Agents documentation](https://huggingface.co/docs/huggingface.js/main/en/tiny-agents/README)
- Build something with Tiny Agents!

[<>Update on GitHub](https://github.com/huggingface/mcp-course/blob/main/units/en/unit2/tiny-agents.mdx)

Building Tiny Agents with MCP and the Hugging Face Hub

[←Building an MCP Client with Gradio](https://huggingface.co/learn/mcp-course/unit2/gradio-client)[Coming Soon→](https://huggingface.co/learn/mcp-course/unit3/introduction)

[Building Tiny Agents with MCP and the Hugging Face Hub](https://huggingface.co/learn/mcp-course/unit2/tiny-agents#building-tiny-agents-with-mcp-and-the-hugging-face-hub)[Installation](https://huggingface.co/learn/mcp-course/unit2/tiny-agents#installation)[Tiny Agents MCP Client in the Command Line](https://huggingface.co/learn/mcp-course/unit2/tiny-agents#tiny-agents-mcp-client-in-the-command-line)[Custom Tiny Agents MCP Client](https://huggingface.co/learn/mcp-course/unit2/tiny-agents#custom-tiny-agents-mcp-client)[Using the Gradio Server with Tiny Agents](https://huggingface.co/learn/mcp-course/unit2/tiny-agents#using-the-gradio-server-with-tiny-agents)[Deployment Considerations](https://huggingface.co/learn/mcp-course/unit2/tiny-agents#deployment-considerations)[Conclusion: Our Complete End-to-End MCP Application](https://huggingface.co/learn/mcp-course/unit2/tiny-agents#conclusion-our-complete-end-to-end-mcp-application)[Next Steps](https://huggingface.co/learn/mcp-course/unit2/tiny-agents#next-steps)


----

