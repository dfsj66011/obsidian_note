
DSPy 是一个用于构建模块化 AI 软件的声明式框架。它让你能基于结构化代码快速迭代，而非脆弱的字符串操作，并提供算法将 AI 程序编译为针对语言模型的高效提示词与权重参数——无论你是在开发简单分类器、复杂的 RAG 流程，还是智能体循环系统。

DSPy（声明式自优化 Python 框架）让你无需费力编写提示词或训练任务，就能用自然语言模块构建 AI 软件，并通用化地将其与不同模型、推理策略或学习算法组合。这使得 AI 软件在跨模型和策略时更可靠、易维护且具备可移植性。

简单来说，DSPy 就像 AI 编程的高级语言，类似于从汇编语言转向 C 语言，或从指针运算转向 SQL。欢迎加入社区、寻求帮助，或通过 GitHub 和 Discord 开始贡献。

**入门指南一：安装 DSPy 并设置语言模型**

```shell
> pip install -U dspy
```

你可以通过设置 OPENAI_API_KEY 环境变量或在下方传递 api_key 进行身份验证。

```python
import dspy
lm = dspy.LM('openai/gpt-4o-mini', api_key='YOUR_OPENAI_API_KEY')
dspy.configure(lm=lm)
```

**直接调用 LM**

地道的 DSPy 编程涉及使用模块，我们将在本页后续内容中定义这些模块。不过，你仍然可以轻松直接调用上面配置的语言模型。这为你提供了统一的 API，并让你能够受益于自动缓存等实用功能。

```python
lm("Say this is a test!", temperature=0.7)  # => ['This is a test!']
lm(messages=[{"role": "user", "content": "Say this is a test!"}])  # => ['This is a test!']
```

**1）模块帮助你将 AI 行为描述为代码，而非字符串。**

To build reliable AI systems, you must iterate fast. But maintaining prompts makes that hard: it forces you to tinker with strings or data _every time you change your LM, metrics, or pipeline_. Having built over a dozen best-in-class compound LM systems since 2020, we learned this the hard way—and so built DSPy to decouple AI system design from messy incidental choices about specific LMs or prompting strategies.

DSPy shifts your focus from tinkering with prompt strings to **programming with structured and declarative natural-language modules**. For every AI component in your system, you specify input/output behavior as a _signature_ and select a _module_ to assign a strategy for invoking your LM. DSPy expands your signatures into prompts and parses your typed outputs, so you can compose different modules together into ergonomic, portable, and optimizable AI systems.

Getting Started II: Build DSPy modules for various tasks

Try the examples below after configuring your `lm` above. Adjust the fields to explore what tasks your LM can do well out of the box. Each tab below sets up a DSPy module, like `dspy.Predict`, `dspy.ChainOfThought`, or `dspy.ReAct`, with a task-specific _signature_. For example, `question -> answer: float` tells the module to take a question and to produce a `float` answer.

[Math](https://dspy.ai/#__tabbed_2_1)[RAG](https://dspy.ai/#__tabbed_2_2)[Classification](https://dspy.ai/#__tabbed_2_3)[Information Extraction](https://dspy.ai/#__tabbed_2_4)[Agents](https://dspy.ai/#__tabbed_2_5)[Multi-Stage Pipelines](https://dspy.ai/#__tabbed_2_6)

|   |   |
|---|---|
|[1](https://dspy.ai/#__codelineno-18-1)<br> [2](https://dspy.ai/#__codelineno-18-2)<br> [3](https://dspy.ai/#__codelineno-18-3)<br> [4](https://dspy.ai/#__codelineno-18-4)<br> [5](https://dspy.ai/#__codelineno-18-5)<br> [6](https://dspy.ai/#__codelineno-18-6)<br> [7](https://dspy.ai/#__codelineno-18-7)<br> [8](https://dspy.ai/#__codelineno-18-8)<br> [9](https://dspy.ai/#__codelineno-18-9)<br>[10](https://dspy.ai/#__codelineno-18-10)<br>[11](https://dspy.ai/#__codelineno-18-11)|`def evaluate_math(expression: str):     return dspy.PythonInterpreter({}).execute(expression) def search_wikipedia(query: str):     results = dspy.ColBERTv2(url='http://20.102.90.50:2017/wiki17_abstracts')(query, k=3)     return [x['text'] for x in results] react = dspy.ReAct("question -> answer: float", tools=[evaluate_math, search_wikipedia]) pred = react(question="What is 9362158 divided by the year of birth of David Gregory of Kinnairdy castle?") print(pred.answer)`|

**Possible Output:**

`[](https://dspy.ai/#__codelineno-19-1)5761.328`

Using DSPy in practice: from quick scripting to building sophisticated systems.

[](https://github.com/stanfordnlp/dspy/issues)

## 2) **Optimizers** tune the prompts and weights of your AI modules.

DSPy provides you with the tools to compile high-level code with natural language annotations into the low-level computations, prompts, or weight updates that align your LM with your program’s structure and metrics. If you change your code or your metrics, you can simply re-compile accordingly.

Given a few tens or hundreds of representative _inputs_ of your task and a _metric_ that can measure the quality of your system's outputs, you can use a DSPy optimizer. Different optimizers in DSPy work by **synthesizing good few-shot examples** for every module, like `dspy.BootstrapRS`,[1](https://arxiv.org/abs/2310.03714) **proposing and intelligently exploring better natural-language instructions** for every prompt, like `dspy.MIPROv2`,[2](https://arxiv.org/abs/2406.11695) and **building datasets for your modules and using them to finetune the LM weights** in your system, like `dspy.BootstrapFinetune`.[3](https://arxiv.org/abs/2407.10930)

Getting Started III: Optimizing the LM prompts or weights in DSPy programs

A typical simple optimization run costs on the order of $2 USD and takes around 20 minutes, but be careful when running optimizers with very large LMs or very large datasets. Optimization can cost as little as a few cents or up to tens of dollars, depending on your LM, dataset, and configuration.

[Optimizing prompts for a ReAct agent](https://dspy.ai/#__tabbed_3_1)[Optimizing prompts for RAG](https://dspy.ai/#__tabbed_3_2)[Optimizing weights for Classification](https://dspy.ai/#__tabbed_3_3)

This is a minimal but fully runnable example of setting up a `dspy.ReAct` agent that answers questions via search from Wikipedia and then optimizing it using `dspy.MIPROv2` in the cheap `light` mode on 500 question-answer pairs sampled from the `HotPotQA` dataset.

|   |   |
|---|---|
|[1](https://dspy.ai/#__codelineno-22-1)<br> [2](https://dspy.ai/#__codelineno-22-2)<br> [3](https://dspy.ai/#__codelineno-22-3)<br> [4](https://dspy.ai/#__codelineno-22-4)<br> [5](https://dspy.ai/#__codelineno-22-5)<br> [6](https://dspy.ai/#__codelineno-22-6)<br> [7](https://dspy.ai/#__codelineno-22-7)<br> [8](https://dspy.ai/#__codelineno-22-8)<br> [9](https://dspy.ai/#__codelineno-22-9)<br>[10](https://dspy.ai/#__codelineno-22-10)<br>[11](https://dspy.ai/#__codelineno-22-11)<br>[12](https://dspy.ai/#__codelineno-22-12)<br>[13](https://dspy.ai/#__codelineno-22-13)<br>[14](https://dspy.ai/#__codelineno-22-14)|`import dspy from dspy.datasets import HotPotQA dspy.configure(lm=dspy.LM('openai/gpt-4o-mini')) def search_wikipedia(query: str) -> list[str]:     results = dspy.ColBERTv2(url='http://20.102.90.50:2017/wiki17_abstracts')(query, k=3)     return [x['text'] for x in results] trainset = [x.with_inputs('question') for x in HotPotQA(train_seed=2024, train_size=500).train] react = dspy.ReAct("question -> answer", tools=[search_wikipedia]) tp = dspy.MIPROv2(metric=dspy.evaluate.answer_exact_match, auto="light", num_threads=24) optimized_react = tp.compile(react, trainset=trainset)`|

An informal run like this raises ReAct's score from 24% to 51%, by teaching `gpt-4o-mini` more about the specifics of the task.

What's an example of a DSPy optimizer? How do different optimizers work?

## 3) **DSPy's Ecosystem** advances open-source AI research.

Compared to monolithic LMs, DSPy's modular paradigm enables a large community to improve the compositional architectures, inference-time strategies, and optimizers for LM programs in an open, distributed way. This gives DSPy users more control, helps them iterate much faster, and allows their programs to get better over time by applying the latest optimizers or modules.

The DSPy research effort started at Stanford NLP in Feb 2022, building on what we had learned from developing early [compound LM systems](https://bair.berkeley.edu/blog/2024/02/18/compound-ai-systems/) like [ColBERT-QA](https://arxiv.org/abs/2007.00814), [Baleen](https://arxiv.org/abs/2101.00436), and [Hindsight](https://arxiv.org/abs/2110.07752). The first version was released as [DSP](https://arxiv.org/abs/2212.14024) in Dec 2022 and evolved by Oct 2023 into [DSPy](https://arxiv.org/abs/2310.03714). Thanks to [250 contributors](https://github.com/stanfordnlp/dspy/graphs/contributors), DSPy has introduced tens of thousands of people to building and optimizing modular LM programs.

Since then, DSPy's community has produced a large body of work on optimizers, like [MIPROv2](https://arxiv.org/abs/2406.11695), [BetterTogether](https://arxiv.org/abs/2407.10930), and [LeReT](https://arxiv.org/abs/2410.23214), on program architectures, like [STORM](https://arxiv.org/abs/2402.14207), [IReRa](https://arxiv.org/abs/2401.12178), and [DSPy Assertions](https://arxiv.org/abs/2312.13382), and on successful applications to new problems, like [PAPILLON](https://arxiv.org/abs/2410.17127), [PATH](https://arxiv.org/abs/2406.11706), [WangLab@MEDIQA](https://arxiv.org/abs/2404.14544), [UMD's Prompting Case Study](https://arxiv.org/abs/2406.06608), and [Haize's Red-Teaming Program](https://blog.haizelabs.com/posts/dspy/), in addition to many open-source projects, production applications, and other [use cases](https://dspy.ai/community/use-cases).

 Back to top

[

Next

Learning DSPy

](https://dspy.ai/learn/)

© 2025 [DSPy](https://github.com/stanfordnlp)

Made with [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/)

[](https://github.com/stanfordnlp/dspy "github.com")[](https://discord.gg/XCGy2WDCQB "discord.gg")

Ask AI