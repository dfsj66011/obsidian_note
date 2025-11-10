
https://www.pinecone.io/learn/series/langchain/

LangChain 库赋能开发者利用 LLM 创建智能应用。它正在革新各行各业与技术领域，彻底改变我们与技术的每一次互动。

## 引言

LangChain 的核心是一个围绕 LLM 构建的框架。我们可以将其用于聊天机器人、生成式问答（GQA）、文本摘要等多种应用场景。

让我们开始吧！

## 一、LangChain：入门指南

LLMs 随着 2020 年 OpenAI 发布 GPT-3 而登上世界舞台。此后，它们的受欢迎程度稳步增长。

直到 2022 年底，情况才有所改变。人们对 LLMs 以及更广泛的生成式人工智能领域的兴趣激增。这背后的原因可能是 LLM 持续取得重大进展的上升势头。

我们看到了关于谷歌“有感知能力”的 LaMDA 聊天机器人的戏剧性新闻。首个名为 BLOOM 的高性能开源 LLM 发布。OpenAI 推出了他们的新一代文本嵌入模型以及下一代 “GPT-3.5” 模型。

在 LLM 领域取得一系列重大突破后，OpenAI 发布了 ChatGPT，使 LLM 技术成为万众瞩目的焦点。

LangChain 大约在同一时间出现。其创始人哈里森·蔡斯（Harrison Chase）于 2022 年 10 月下旬进行了首次提交。在陷入 LLM 浪潮之前，仅进行了短短几个月的开发。

尽管该库还处于早期阶段，但它已经包含了围绕 LLM 核心构建出色工具的众多强大功能。在本文中，我们将介绍这个库，并从 LangChain 提供的最简单组件——LLM 开始。

### LangChain

LangChain 本质上是一个围绕 LLM 构建的框架。我们可以将其用于聊天机器人、生成式问答（GQA）、文本摘要等多种场景。

该库的核心思想是我们可以将不同组件“链式”组合在一起，围绕 LLM 构建更高级的用例。链可以由来自多个模块的多个组件组成：

* 提示模板：提示模板是针对不同类型提示的模板。例如“聊天机器人”风格的模板、ELI5 问答模板等。
* LLM：如 GPT-3、BLOOM 等
* Agents：代理利用 LLM 来决定应采取哪些行动。可以使用网络搜索或计算器等工具，所有这些都被封装在一个逻辑操作循环中。
* 记忆：短期记忆，长期记忆。

我们将在 LangChain手册后续章节中更详细地逐一探讨这些内容。

目前，我们将从提示模板和 LLMs 的基础知识开始。我们还将探索库中提供的两种 LLM 选项，使用来自 Hugging Face Hub 或 OpenAI 的模型。

### 提示模板

输入到 LLM 的提示词通常以不同方式构建，以便获得不同的结果。对于问答场景，我们可以将用户的问题重新格式化以适应不同的问答风格，比如传统问答形式、答案要点列表，甚至是与给定问题相关的摘要总结。

#### 在 LangChain 中创建 prompt

让我们来构建一个简单的问答提示模板。首先需要安装 `langchain` 库。

```bash
!pip install langchain
```

从这里，我们导入 `PromptTemplate` 类并像这样初始化一个模板：

```python
from langchain import PromptTemplate

template = """Question: {question}

Answer: """
prompt = PromptTemplate(
        template=template,
    input_variables=['question']
)

# user question
question = "Which NFL team won the Super Bowl in the 2010 season?"
```

使用这些提示模板与给定问题时，我们将得到：

```
Question: Which NFL team won the Super Bowl in the 2010 season? Answer:
```

目前，这就是我们所需要的。我们将在 Hugging Face Hub 和 OpenAI LLM 生成中使用相同的提示模板。

### Hugging Face Hub LLM

LangChain 中的 Hugging Face Hub 端点连接到 Hugging Face Hub，并通过其免费推理端点运行模型。我们需要一个 Hugging Face 账户和 API 密钥来使用这些端点。

一旦你获得了 API 密钥，我们就将其添加到 `HUGGINGFACEHUB_API_TOKEN` 环境变量中。我们可以像这样用 Python 来实现：

```python
import os

os.environ['HUGGINGFACEHUB_API_TOKEN'] = 'xxx'
```

接下来，我们必须通过 Pip安装 `huggingface_hub` 库。

```
!pip install huggingface_hub
```

现在我们可以使用 Hub 模型生成文本。我们将使用 `google/flan-t5-x1`。

Hugging Face Hub 默认的推理 API 并未使用专用硬件，因此运行速度可能较慢。它们也不适合运行大型模型，例如 `bigscience/bloom-560m` 或 `google/flan-t5-xxl`（注意 `xxl` 与 `xl` 的区别）。

```python
from langchain import HuggingFaceHub, LLMChain

# initialize Hub LLM
hub_llm = HuggingFaceHub(
	repo_id='google/flan-t5-xl',
    model_kwargs={'temperature':1e-10}
)

# create prompt template > LLM chain
llm_chain = LLMChain(
    prompt=prompt,
    llm=hub_llm
)

# ask the user question about NFL 2010
print(llm_chain.run(question))
```