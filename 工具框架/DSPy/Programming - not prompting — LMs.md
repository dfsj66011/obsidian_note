


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

------

# Learning DSPy: An Overview

DSPy exposes a very small API that you can learn quickly. However, building a new AI system is a more open-ended journey of iterative development, in which you compose the tools and design patterns of DSPy to optimize for _your_ objectives. The three stages of building AI systems in DSPy are:

1) **DSPy Programming.** This is about defining your task, its constraints, exploring a few examples, and using that to inform your initial pipeline design.

2) **DSPy Evaluation.** Once your system starts working, this is the stage where you collect an initial development set, define your DSPy metric, and use these to iterate on your system more systematically.

3) **DSPy Optimization.** Once you have a way to evaluate your system, you use DSPy optimizers to tune the prompts or weights in your program.

We suggest learning and applying DSPy in this order. For example, it's unproductive to launch optimization runs using a poorly-design program or a bad metric.

-----

# Programming in DSPy

DSPy is a bet on _writing code instead of strings_. In other words, building the right control flow is crucial. Start by **defining your task**. What are the inputs to your system and what should your system produce as output? Is it a chatbot over your data or perhaps a code assistant? Or maybe a system for translation, for highlighting snippets from search results, or for generating reports with citations?

Next, **define your initial pipeline**. Can your DSPy program just be a single module or do you need to break it down into a few steps? Do you need retrieval or other tools, like a calculator or a calendar API? Is there a typical workflow for solving your problem in multiple well-scoped steps, or do you want more open-ended tool use with agents for your task? Think about these but start simple, perhaps with just a single `dspy.ChainOfThought` module, then add complexity incrementally based on observations.

As you do this, **craft and try a handful of examples** of the inputs to your program. Consider using a powerful LM at this point, or a couple of different LMs, just to understand what's possible. Record interesting (both easy and hard) examples you try. This will be useful when you are doing evaluation and optimization later.

----

# Language Models

The first step in any DSPy code is to set up your language model. For example, you can configure OpenAI's GPT-4o-mini as your default LM as follows.

|   |   |
|---|---|
|[1](https://dspy.ai/learn/programming/language_models/#__codelineno-0-1)<br>[2](https://dspy.ai/learn/programming/language_models/#__codelineno-0-2)<br>[3](https://dspy.ai/learn/programming/language_models/#__codelineno-0-3)|``# Authenticate via `OPENAI_API_KEY` env: import os; os.environ['OPENAI_API_KEY'] = 'here' lm = dspy.LM('openai/gpt-4o-mini') dspy.configure(lm=lm)``|

A few different LMs

[OpenAI](https://dspy.ai/learn/programming/language_models/#__tabbed_1_1)[Gemini (AI Studio)](https://dspy.ai/learn/programming/language_models/#__tabbed_1_2)[Anthropic](https://dspy.ai/learn/programming/language_models/#__tabbed_1_3)[Databricks](https://dspy.ai/learn/programming/language_models/#__tabbed_1_4)[Local LMs on a GPU server](https://dspy.ai/learn/programming/language_models/#__tabbed_1_5)[Local LMs on your laptop](https://dspy.ai/learn/programming/language_models/#__tabbed_1_6)[Other providers](https://dspy.ai/learn/programming/language_models/#__tabbed_1_7)

You can authenticate by setting the `OPENAI_API_KEY` env variable or passing `api_key` below.

|   |   |
|---|---|
|[1](https://dspy.ai/learn/programming/language_models/#__codelineno-1-1)<br>[2](https://dspy.ai/learn/programming/language_models/#__codelineno-1-2)<br>[3](https://dspy.ai/learn/programming/language_models/#__codelineno-1-3)|`import dspy lm = dspy.LM('openai/gpt-4o-mini', api_key='YOUR_OPENAI_API_KEY') dspy.configure(lm=lm)`|

If you run into errors, please refer to the [LiteLLM Docs](https://docs.litellm.ai/docs/providers) to verify if you are using the same variable names/following the right procedure.

## Calling the LM directly.

It's easy to call the `lm` you configured above directly. This gives you a unified API and lets you benefit from utilities like automatic caching.

|   |   |
|---|---|
|[1](https://dspy.ai/learn/programming/language_models/#__codelineno-10-1)<br>[2](https://dspy.ai/learn/programming/language_models/#__codelineno-10-2)|`lm("Say this is a test!", temperature=0.7)  # => ['This is a test!'] lm(messages=[{"role": "user", "content": "Say this is a test!"}])  # => ['This is a test!']`|

## Using the LM with DSPy modules.

Idiomatic DSPy involves using _modules_, which we discuss in the next guide.

|   |   |
|---|---|
|[1](https://dspy.ai/learn/programming/language_models/#__codelineno-11-1)<br>[2](https://dspy.ai/learn/programming/language_models/#__codelineno-11-2)<br>[3](https://dspy.ai/learn/programming/language_models/#__codelineno-11-3)<br>[4](https://dspy.ai/learn/programming/language_models/#__codelineno-11-4)<br>[5](https://dspy.ai/learn/programming/language_models/#__codelineno-11-5)<br>[6](https://dspy.ai/learn/programming/language_models/#__codelineno-11-6)|``# Define a module (ChainOfThought) and assign it a signature (return an answer, given a question). qa = dspy.ChainOfThought('question -> answer') # Run with the default LM configured with `dspy.configure` above. response = qa(question="How many floors are in the castle David Gregory inherited?") print(response.answer)``|

**Possible Output:**

`[](https://dspy.ai/learn/programming/language_models/#__codelineno-12-1)The castle David Gregory inherited has 7 floors.`

## Using multiple LMs.

You can change the default LM globally with `dspy.configure` or change it inside a block of code with `dspy.context`.

Tip

Using `dspy.configure` and `dspy.context` is thread-safe!

|   |   |
|---|---|
|[1](https://dspy.ai/learn/programming/language_models/#__codelineno-13-1)<br>[2](https://dspy.ai/learn/programming/language_models/#__codelineno-13-2)<br>[3](https://dspy.ai/learn/programming/language_models/#__codelineno-13-3)<br>[4](https://dspy.ai/learn/programming/language_models/#__codelineno-13-4)<br>[5](https://dspy.ai/learn/programming/language_models/#__codelineno-13-5)<br>[6](https://dspy.ai/learn/programming/language_models/#__codelineno-13-6)<br>[7](https://dspy.ai/learn/programming/language_models/#__codelineno-13-7)|`dspy.configure(lm=dspy.LM('openai/gpt-4o-mini')) response = qa(question="How many floors are in the castle David Gregory inherited?") print('GPT-4o-mini:', response.answer) with dspy.context(lm=dspy.LM('openai/gpt-3.5-turbo')):     response = qa(question="How many floors are in the castle David Gregory inherited?")     print('GPT-3.5-turbo:', response.answer)`|

**Possible Output:**

`[](https://dspy.ai/learn/programming/language_models/#__codelineno-14-1)GPT-4o: The number of floors in the castle David Gregory inherited cannot be determined with the information provided. [](https://dspy.ai/learn/programming/language_models/#__codelineno-14-2)GPT-3.5-turbo: The castle David Gregory inherited has 7 floors.`

## Configuring LM generation.

For any LM, you can configure any of the following attributes at initialization or in each subsequent call.

|   |   |
|---|---|
|[1](https://dspy.ai/learn/programming/language_models/#__codelineno-15-1)|`gpt_4o_mini = dspy.LM('openai/gpt-4o-mini', temperature=0.9, max_tokens=3000, stop=None, cache=False)`|

By default LMs in DSPy are cached. If you repeat the same call, you will get the same outputs. But you can turn off caching by setting `cache=False`.

## Inspecting output and usage metadata.

Every LM object maintains the history of its interactions, including inputs, outputs, token usage (and $$$ cost), and metadata.

|   |   |
|---|---|
|[1](https://dspy.ai/learn/programming/language_models/#__codelineno-16-1)<br>[2](https://dspy.ai/learn/programming/language_models/#__codelineno-16-2)<br>[3](https://dspy.ai/learn/programming/language_models/#__codelineno-16-3)|`len(lm.history)  # e.g., 3 calls to the LM lm.history[-1].keys()  # access the last call to the LM, with all metadata`|

**Output:**

`[](https://dspy.ai/learn/programming/language_models/#__codelineno-17-1)dict_keys(['prompt', 'messages', 'kwargs', 'response', 'outputs', 'usage', 'cost'])`

### Advanced: Building custom LMs and writing your own Adapters.

Though rarely needed, you can write custom LMs by inheriting from `dspy.BaseLM`. Another advanced layer in the DSPy ecosystem is that of _adapters_, which sit between DSPy signatures and LMs. A future version of this guide will discuss these advanced features, though you likely don't need them.

-----


# Signatures

When we assign tasks to LMs in DSPy, we specify the behavior we need as a Signature.

**A signature is a declarative specification of input/output behavior of a DSPy module.** Signatures allow you to tell the LM _what_ it needs to do, rather than specify _how_ we should ask the LM to do it.

You're probably familiar with function signatures, which specify the input and output arguments and their types. DSPy signatures are similar, but with a couple of differences. While typical function signatures just _describe_ things, DSPy Signatures _declare and initialize the behavior_ of modules. Moreover, the field names matter in DSPy Signatures. You express semantic roles in plain English: a `question` is different from an `answer`, a `sql_query` is different from `python_code`.

## Why should I use a DSPy Signature?

For modular and clean code, in which LM calls can be optimized into high-quality prompts (or automatic finetunes). Most people coerce LMs to do tasks by hacking long, brittle prompts. Or by collecting/generating data for fine-tuning. Writing signatures is far more modular, adaptive, and reproducible than hacking at prompts or finetunes. The DSPy compiler will figure out how to build a highly-optimized prompt for your LM (or finetune your small LM) for your signature, on your data, and within your pipeline. In many cases, we found that compiling leads to better prompts than humans write. Not because DSPy optimizers are more creative than humans, but simply because they can try more things and tune the metrics directly.

## **Inline** DSPy Signatures

Signatures can be defined as a short string, with argument names and optional types that define semantic roles for inputs/outputs.

1. Question Answering: `"question -> answer"`, which is equivalent to `"question: str -> answer: str"` as the default type is always `str`
    
2. Sentiment Classification: `"sentence -> sentiment: bool"`, e.g. `True` if positive
    
3. Summarization: `"document -> summary"`
    

Your signatures can also have multiple input/output fields with types:

1. Retrieval-Augmented Question Answering: `"context: list[str], question: str -> answer: str"`
    
2. Multiple-Choice Question Answering with Reasoning: `"question, choices: list[str] -> reasoning: str, selection: int"`
    

**Tip:** For fields, any valid variable names work! Field names should be semantically meaningful, but start simple and don't prematurely optimize keywords! Leave that kind of hacking to the DSPy compiler. For example, for summarization, it's probably fine to say `"document -> summary"`, `"text -> gist"`, or `"long_context -> tldr"`.

You can also add instructions to your inline signature, which can use variables at runtime. Use the `instructions` keyword argument to add instructions to your signature.

`[](https://dspy.ai/learn/programming/signatures/#__codelineno-0-1)toxicity = dspy.Predict( [](https://dspy.ai/learn/programming/signatures/#__codelineno-0-2)    dspy.Signature( [](https://dspy.ai/learn/programming/signatures/#__codelineno-0-3)        "comment -> toxic: bool", [](https://dspy.ai/learn/programming/signatures/#__codelineno-0-4)        instructions="Mark as 'toxic' if the comment includes insults, harassment, or sarcastic derogatory remarks.", [](https://dspy.ai/learn/programming/signatures/#__codelineno-0-5)    ) [](https://dspy.ai/learn/programming/signatures/#__codelineno-0-6))`

### Example A: Sentiment Classification

`[](https://dspy.ai/learn/programming/signatures/#__codelineno-1-1)sentence = "it's a charming and often affecting journey."  # example from the SST-2 dataset. [](https://dspy.ai/learn/programming/signatures/#__codelineno-1-2) [](https://dspy.ai/learn/programming/signatures/#__codelineno-1-3)classify = dspy.Predict('sentence -> sentiment: bool')  # we'll see an example with Literal[] later [](https://dspy.ai/learn/programming/signatures/#__codelineno-1-4)classify(sentence=sentence).sentiment`

**Output:**

`[](https://dspy.ai/learn/programming/signatures/#__codelineno-2-1)True`

### Example B: Summarization

`[](https://dspy.ai/learn/programming/signatures/#__codelineno-3-1)# Example from the XSum dataset. [](https://dspy.ai/learn/programming/signatures/#__codelineno-3-2)document = """The 21-year-old made seven appearances for the Hammers and netted his only goal for them in a Europa League qualification round match against Andorran side FC Lustrains last season. Lee had two loan spells in League One last term, with Blackpool and then Colchester United. He scored twice for the U's but was unable to save them from relegation. The length of Lee's contract with the promoted Tykes has not been revealed. Find all the latest football transfers on our dedicated page.""" [](https://dspy.ai/learn/programming/signatures/#__codelineno-3-3) [](https://dspy.ai/learn/programming/signatures/#__codelineno-3-4)summarize = dspy.ChainOfThought('document -> summary') [](https://dspy.ai/learn/programming/signatures/#__codelineno-3-5)response = summarize(document=document) [](https://dspy.ai/learn/programming/signatures/#__codelineno-3-6) [](https://dspy.ai/learn/programming/signatures/#__codelineno-3-7)print(response.summary)`

**Possible Output:**

`[](https://dspy.ai/learn/programming/signatures/#__codelineno-4-1)The 21-year-old Lee made seven appearances and scored one goal for West Ham last season. He had loan spells in League One with Blackpool and Colchester United, scoring twice for the latter. He has now signed a contract with Barnsley, but the length of the contract has not been revealed.`

Many DSPy modules (except `dspy.Predict`) return auxiliary information by expanding your signature under the hood.

For example, `dspy.ChainOfThought` also adds a `reasoning` field that includes the LM's reasoning before it generates the output `summary`.

`[](https://dspy.ai/learn/programming/signatures/#__codelineno-5-1)print("Reasoning:", response.reasoning)`

**Possible Output:**

`[](https://dspy.ai/learn/programming/signatures/#__codelineno-6-1)Reasoning: We need to highlight Lee's performance for West Ham, his loan spells in League One, and his new contract with Barnsley. We also need to mention that his contract length has not been disclosed.`

## **Class-based** DSPy Signatures

For some advanced tasks, you need more verbose signatures. This is typically to:

1. Clarify something about the nature of the task (expressed below as a `docstring`).
    
2. Supply hints on the nature of an input field, expressed as a `desc` keyword argument for `dspy.InputField`.
    
3. Supply constraints on an output field, expressed as a `desc` keyword argument for `dspy.OutputField`.
    

### Example C: Classification

`[](https://dspy.ai/learn/programming/signatures/#__codelineno-7-1)from typing import Literal [](https://dspy.ai/learn/programming/signatures/#__codelineno-7-2) [](https://dspy.ai/learn/programming/signatures/#__codelineno-7-3)class Emotion(dspy.Signature): [](https://dspy.ai/learn/programming/signatures/#__codelineno-7-4)    """Classify emotion.""" [](https://dspy.ai/learn/programming/signatures/#__codelineno-7-5) [](https://dspy.ai/learn/programming/signatures/#__codelineno-7-6)    sentence: str = dspy.InputField() [](https://dspy.ai/learn/programming/signatures/#__codelineno-7-7)    sentiment: Literal['sadness', 'joy', 'love', 'anger', 'fear', 'surprise'] = dspy.OutputField() [](https://dspy.ai/learn/programming/signatures/#__codelineno-7-8) [](https://dspy.ai/learn/programming/signatures/#__codelineno-7-9)sentence = "i started feeling a little vulnerable when the giant spotlight started blinding me"  # from dair-ai/emotion [](https://dspy.ai/learn/programming/signatures/#__codelineno-7-10) [](https://dspy.ai/learn/programming/signatures/#__codelineno-7-11)classify = dspy.Predict(Emotion) [](https://dspy.ai/learn/programming/signatures/#__codelineno-7-12)classify(sentence=sentence)`

**Possible Output:**

`[](https://dspy.ai/learn/programming/signatures/#__codelineno-8-1)Prediction( [](https://dspy.ai/learn/programming/signatures/#__codelineno-8-2)    sentiment='fear' [](https://dspy.ai/learn/programming/signatures/#__codelineno-8-3))`

**Tip:** There's nothing wrong with specifying your requests to the LM more clearly. Class-based Signatures help you with that. However, don't prematurely tune the keywords of your signature by hand. The DSPy optimizers will likely do a better job (and will transfer better across LMs).

### Example D: A metric that evaluates faithfulness to citations

`[](https://dspy.ai/learn/programming/signatures/#__codelineno-9-1)class CheckCitationFaithfulness(dspy.Signature): [](https://dspy.ai/learn/programming/signatures/#__codelineno-9-2)    """Verify that the text is based on the provided context.""" [](https://dspy.ai/learn/programming/signatures/#__codelineno-9-3) [](https://dspy.ai/learn/programming/signatures/#__codelineno-9-4)    context: str = dspy.InputField(desc="facts here are assumed to be true") [](https://dspy.ai/learn/programming/signatures/#__codelineno-9-5)    text: str = dspy.InputField() [](https://dspy.ai/learn/programming/signatures/#__codelineno-9-6)    faithfulness: bool = dspy.OutputField() [](https://dspy.ai/learn/programming/signatures/#__codelineno-9-7)    evidence: dict[str, list[str]] = dspy.OutputField(desc="Supporting evidence for claims") [](https://dspy.ai/learn/programming/signatures/#__codelineno-9-8) [](https://dspy.ai/learn/programming/signatures/#__codelineno-9-9)context = "The 21-year-old made seven appearances for the Hammers and netted his only goal for them in a Europa League qualification round match against Andorran side FC Lustrains last season. Lee had two loan spells in League One last term, with Blackpool and then Colchester United. He scored twice for the U's but was unable to save them from relegation. The length of Lee's contract with the promoted Tykes has not been revealed. Find all the latest football transfers on our dedicated page." [](https://dspy.ai/learn/programming/signatures/#__codelineno-9-10) [](https://dspy.ai/learn/programming/signatures/#__codelineno-9-11)text = "Lee scored 3 goals for Colchester United." [](https://dspy.ai/learn/programming/signatures/#__codelineno-9-12) [](https://dspy.ai/learn/programming/signatures/#__codelineno-9-13)faithfulness = dspy.ChainOfThought(CheckCitationFaithfulness) [](https://dspy.ai/learn/programming/signatures/#__codelineno-9-14)faithfulness(context=context, text=text)`

**Possible Output:**

`[](https://dspy.ai/learn/programming/signatures/#__codelineno-10-1)Prediction( [](https://dspy.ai/learn/programming/signatures/#__codelineno-10-2)    reasoning="Let's check the claims against the context. The text states Lee scored 3 goals for Colchester United, but the context clearly states 'He scored twice for the U's'. This is a direct contradiction.", [](https://dspy.ai/learn/programming/signatures/#__codelineno-10-3)    faithfulness=False, [](https://dspy.ai/learn/programming/signatures/#__codelineno-10-4)    evidence={'goal_count': ["scored twice for the U's"]} [](https://dspy.ai/learn/programming/signatures/#__codelineno-10-5))`

### Example E: Multi-modal image classification

`[](https://dspy.ai/learn/programming/signatures/#__codelineno-11-1)class DogPictureSignature(dspy.Signature): [](https://dspy.ai/learn/programming/signatures/#__codelineno-11-2)    """Output the dog breed of the dog in the image.""" [](https://dspy.ai/learn/programming/signatures/#__codelineno-11-3)    image_1: dspy.Image = dspy.InputField(desc="An image of a dog") [](https://dspy.ai/learn/programming/signatures/#__codelineno-11-4)    answer: str = dspy.OutputField(desc="The dog breed of the dog in the image") [](https://dspy.ai/learn/programming/signatures/#__codelineno-11-5) [](https://dspy.ai/learn/programming/signatures/#__codelineno-11-6)image_url = "https://picsum.photos/id/237/200/300" [](https://dspy.ai/learn/programming/signatures/#__codelineno-11-7)classify = dspy.Predict(DogPictureSignature) [](https://dspy.ai/learn/programming/signatures/#__codelineno-11-8)classify(image_1=dspy.Image.from_url(image_url))`

**Possible Output:**

`[](https://dspy.ai/learn/programming/signatures/#__codelineno-12-1)Prediction( [](https://dspy.ai/learn/programming/signatures/#__codelineno-12-2)    answer='Labrador Retriever' [](https://dspy.ai/learn/programming/signatures/#__codelineno-12-3))`

## Type Resolution in Signatures

DSPy signatures support various annotation types:

1. **Basic types** like `str`, `int`, `bool`
2. **Typing module types** like `List[str]`, `Dict[str, int]`, `Optional[float]`. `Union[str, int]`
3. **Custom types** defined in your code
4. **Dot notation** for nested types with proper configuration
5. **Special data types** like `dspy.Image, dspy.History`

### Working with Custom Types

`[](https://dspy.ai/learn/programming/signatures/#__codelineno-13-1)# Simple custom type [](https://dspy.ai/learn/programming/signatures/#__codelineno-13-2)class QueryResult(pydantic.BaseModel): [](https://dspy.ai/learn/programming/signatures/#__codelineno-13-3)    text: str [](https://dspy.ai/learn/programming/signatures/#__codelineno-13-4)    score: float [](https://dspy.ai/learn/programming/signatures/#__codelineno-13-5) [](https://dspy.ai/learn/programming/signatures/#__codelineno-13-6)signature = dspy.Signature("query: str -> result: QueryResult") [](https://dspy.ai/learn/programming/signatures/#__codelineno-13-7) [](https://dspy.ai/learn/programming/signatures/#__codelineno-13-8)class Container: [](https://dspy.ai/learn/programming/signatures/#__codelineno-13-9)    class Query(pydantic.BaseModel): [](https://dspy.ai/learn/programming/signatures/#__codelineno-13-10)        text: str [](https://dspy.ai/learn/programming/signatures/#__codelineno-13-11)    class Score(pydantic.BaseModel): [](https://dspy.ai/learn/programming/signatures/#__codelineno-13-12)        score: float [](https://dspy.ai/learn/programming/signatures/#__codelineno-13-13) [](https://dspy.ai/learn/programming/signatures/#__codelineno-13-14)signature = dspy.Signature("query: Container.Query -> score: Container.Score")`

## Using signatures to build modules & compiling them

While signatures are convenient for prototyping with structured inputs/outputs, that's not the only reason to use them!

You should compose multiple signatures into bigger [DSPy modules](https://dspy.ai/learn/programming/modules/) and [compile these modules into optimized prompts](https://dspy.ai/learn/optimization/optimizers/) and finetunes.


-------


# Modules

A **DSPy module** is a building block for programs that use LMs.

- Each built-in module abstracts a **prompting technique** (like chain of thought or ReAct). Crucially, they are generalized to handle any signature.
    
- A DSPy module has **learnable parameters** (i.e., the little pieces comprising the prompt and the LM weights) and can be invoked (called) to process inputs and return outputs.
    
- Multiple modules can be composed into bigger modules (programs). DSPy modules are inspired directly by NN modules in PyTorch, but applied to LM programs.
    

## How do I use a built-in module, like `dspy.Predict` or `dspy.ChainOfThought`?

Let's start with the most fundamental module, `dspy.Predict`. Internally, all other DSPy modules are built using `dspy.Predict`. We'll assume you are already at least a little familiar with [DSPy signatures](https://dspy.ai/learn/programming/signatures), which are declarative specs for defining the behavior of any module we use in DSPy.

To use a module, we first **declare** it by giving it a signature. Then we **call** the module with the input arguments, and extract the output fields!

`[](https://dspy.ai/learn/programming/modules/#__codelineno-0-1)sentence = "it's a charming and often affecting journey."  # example from the SST-2 dataset. [](https://dspy.ai/learn/programming/modules/#__codelineno-0-2) [](https://dspy.ai/learn/programming/modules/#__codelineno-0-3)# 1) Declare with a signature. [](https://dspy.ai/learn/programming/modules/#__codelineno-0-4)classify = dspy.Predict('sentence -> sentiment: bool') [](https://dspy.ai/learn/programming/modules/#__codelineno-0-5) [](https://dspy.ai/learn/programming/modules/#__codelineno-0-6)# 2) Call with input argument(s).  [](https://dspy.ai/learn/programming/modules/#__codelineno-0-7)response = classify(sentence=sentence) [](https://dspy.ai/learn/programming/modules/#__codelineno-0-8) [](https://dspy.ai/learn/programming/modules/#__codelineno-0-9)# 3) Access the output. [](https://dspy.ai/learn/programming/modules/#__codelineno-0-10)print(response.sentiment)`

**Output:**

`[](https://dspy.ai/learn/programming/modules/#__codelineno-1-1)True`

When we declare a module, we can pass configuration keys to it.

Below, we'll pass `n=5` to request five completions. We can also pass `temperature` or `max_len`, etc.

Let's use `dspy.ChainOfThought`. In many cases, simply swapping `dspy.ChainOfThought` in place of `dspy.Predict` improves quality.

`[](https://dspy.ai/learn/programming/modules/#__codelineno-2-1)question = "What's something great about the ColBERT retrieval model?" [](https://dspy.ai/learn/programming/modules/#__codelineno-2-2) [](https://dspy.ai/learn/programming/modules/#__codelineno-2-3)# 1) Declare with a signature, and pass some config. [](https://dspy.ai/learn/programming/modules/#__codelineno-2-4)classify = dspy.ChainOfThought('question -> answer', n=5) [](https://dspy.ai/learn/programming/modules/#__codelineno-2-5) [](https://dspy.ai/learn/programming/modules/#__codelineno-2-6)# 2) Call with input argument. [](https://dspy.ai/learn/programming/modules/#__codelineno-2-7)response = classify(question=question) [](https://dspy.ai/learn/programming/modules/#__codelineno-2-8) [](https://dspy.ai/learn/programming/modules/#__codelineno-2-9)# 3) Access the outputs. [](https://dspy.ai/learn/programming/modules/#__codelineno-2-10)response.completions.answer`

**Possible Output:**

`[](https://dspy.ai/learn/programming/modules/#__codelineno-3-1)['One great thing about the ColBERT retrieval model is its superior efficiency and effectiveness compared to other models.', [](https://dspy.ai/learn/programming/modules/#__codelineno-3-2) 'Its ability to efficiently retrieve relevant information from large document collections.', [](https://dspy.ai/learn/programming/modules/#__codelineno-3-3) 'One great thing about the ColBERT retrieval model is its superior performance compared to other models and its efficient use of pre-trained language models.', [](https://dspy.ai/learn/programming/modules/#__codelineno-3-4) 'One great thing about the ColBERT retrieval model is its superior efficiency and accuracy compared to other models.', [](https://dspy.ai/learn/programming/modules/#__codelineno-3-5) 'One great thing about the ColBERT retrieval model is its ability to incorporate user feedback and support complex queries.']`

Let's discuss the output object here. The `dspy.ChainOfThought` module will generally inject a `reasoning` before the output field(s) of your signature.

Let's inspect the (first) reasoning and answer!

`[](https://dspy.ai/learn/programming/modules/#__codelineno-4-1)print(f"Reasoning: {response.reasoning}") [](https://dspy.ai/learn/programming/modules/#__codelineno-4-2)print(f"Answer: {response.answer}")`

**Possible Output:**

`[](https://dspy.ai/learn/programming/modules/#__codelineno-5-1)Reasoning: We can consider the fact that ColBERT has shown to outperform other state-of-the-art retrieval models in terms of efficiency and effectiveness. It uses contextualized embeddings and performs document retrieval in a way that is both accurate and scalable. [](https://dspy.ai/learn/programming/modules/#__codelineno-5-2)Answer: One great thing about the ColBERT retrieval model is its superior efficiency and effectiveness compared to other models.`

This is accessible whether we request one or many completions.

We can also access the different completions as a list of `Prediction`s or as several lists, one for each field.

`[](https://dspy.ai/learn/programming/modules/#__codelineno-6-1)response.completions[3].reasoning == response.completions.reasoning[3]`

**Output:**

`[](https://dspy.ai/learn/programming/modules/#__codelineno-7-1)True`

## What other DSPy modules are there? How can I use them?

The others are very similar. They mainly change the internal behavior with which your signature is implemented!

1. **`dspy.Predict`**: Basic predictor. Does not modify the signature. Handles the key forms of learning (i.e., storing the instructions and demonstrations and updates to the LM).
    
2. **`dspy.ChainOfThought`**: Teaches the LM to think step-by-step before committing to the signature's response.
    
3. **`dspy.ProgramOfThought`**: Teaches the LM to output code, whose execution results will dictate the response.
    
4. **`dspy.ReAct`**: An agent that can use tools to implement the given signature.
    
5. **`dspy.MultiChainComparison`**: Can compare multiple outputs from `ChainOfThought` to produce a final prediction.
    

We also have some function-style modules:

1. **`dspy.majority`**: Can do basic voting to return the most popular response from a set of predictions.

A few examples of DSPy modules on simple tasks.

Try the examples below after configuring your `lm`. Adjust the fields to explore what tasks your LM can do well out of the box.

[Math](https://dspy.ai/learn/programming/modules/#__tabbed_1_1)[Retrieval-Augmented Generation](https://dspy.ai/learn/programming/modules/#__tabbed_1_2)[Classification](https://dspy.ai/learn/programming/modules/#__tabbed_1_3)[Information Extraction](https://dspy.ai/learn/programming/modules/#__tabbed_1_4)[Agents](https://dspy.ai/learn/programming/modules/#__tabbed_1_5)

|   |   |
|---|---|
|[1](https://dspy.ai/learn/programming/modules/#__codelineno-8-1)<br>[2](https://dspy.ai/learn/programming/modules/#__codelineno-8-2)|`math = dspy.ChainOfThought("question -> answer: float") math(question="Two dice are tossed. What is the probability that the sum equals two?")`|

**Possible Output:**

`[](https://dspy.ai/learn/programming/modules/#__codelineno-9-1)Prediction( [](https://dspy.ai/learn/programming/modules/#__codelineno-9-2)    reasoning='When two dice are tossed, each die has 6 faces, resulting in a total of 6 x 6 = 36 possible outcomes. The sum of the numbers on the two dice equals two only when both dice show a 1. This is just one specific outcome: (1, 1). Therefore, there is only 1 favorable outcome. The probability of the sum being two is the number of favorable outcomes divided by the total number of possible outcomes, which is 1/36.', [](https://dspy.ai/learn/programming/modules/#__codelineno-9-3)    answer=0.0277776 [](https://dspy.ai/learn/programming/modules/#__codelineno-9-4))`

## How do I compose multiple modules into a bigger program?

DSPy is just Python code that uses modules in any control flow you like, with a little magic internally at `compile` time to trace your LM calls. What this means is that, you can just call the modules freely.

See tutorials like [multi-hop search](https://dspy.ai/tutorials/multihop_search/), whose module is reproduced below as an example.

|   |   |
|---|---|
|[1](https://dspy.ai/learn/programming/modules/#__codelineno-18-1)<br> [2](https://dspy.ai/learn/programming/modules/#__codelineno-18-2)<br> [3](https://dspy.ai/learn/programming/modules/#__codelineno-18-3)<br> [4](https://dspy.ai/learn/programming/modules/#__codelineno-18-4)<br> [5](https://dspy.ai/learn/programming/modules/#__codelineno-18-5)<br> [6](https://dspy.ai/learn/programming/modules/#__codelineno-18-6)<br> [7](https://dspy.ai/learn/programming/modules/#__codelineno-18-7)<br> [8](https://dspy.ai/learn/programming/modules/#__codelineno-18-8)<br> [9](https://dspy.ai/learn/programming/modules/#__codelineno-18-9)<br>[10](https://dspy.ai/learn/programming/modules/#__codelineno-18-10)<br>[11](https://dspy.ai/learn/programming/modules/#__codelineno-18-11)<br>[12](https://dspy.ai/learn/programming/modules/#__codelineno-18-12)<br>[13](https://dspy.ai/learn/programming/modules/#__codelineno-18-13)<br>[14](https://dspy.ai/learn/programming/modules/#__codelineno-18-14)<br>[15](https://dspy.ai/learn/programming/modules/#__codelineno-18-15)<br>[16](https://dspy.ai/learn/programming/modules/#__codelineno-18-16)<br>[17](https://dspy.ai/learn/programming/modules/#__codelineno-18-17)<br>[18](https://dspy.ai/learn/programming/modules/#__codelineno-18-18)|`class Hop(dspy.Module):     def __init__(self, num_docs=10, num_hops=4):         self.num_docs, self.num_hops = num_docs, num_hops         self.generate_query = dspy.ChainOfThought('claim, notes -> query')         self.append_notes = dspy.ChainOfThought('claim, notes, context -> new_notes: list[str], titles: list[str]')     def forward(self, claim: str) -> list[str]:         notes = []         titles = []         for _ in range(self.num_hops):             query = self.generate_query(claim=claim, notes=notes).query             context = search(query, k=self.num_docs)             prediction = self.append_notes(claim=claim, notes=notes, context=context)             notes.extend(prediction.new_notes)             titles.extend(prediction.titles)         return dspy.Prediction(notes=notes, titles=list(set(titles)))`|

Then you can create a instance of the custom module class `Hop`, then invoke it by the `__call__` method:

`[](https://dspy.ai/learn/programming/modules/#__codelineno-19-1)hop = Hop() [](https://dspy.ai/learn/programming/modules/#__codelineno-19-2)print(hop(claim="Stephen Curry is the best 3 pointer shooter ever in the human history"))`

## How do I track LM usage?

Version Requirement

LM usage tracking is available in DSPy version 2.6.16 and later.

DSPy provides built-in tracking of language model usage across all module calls. To enable tracking:

`[](https://dspy.ai/learn/programming/modules/#__codelineno-20-1)dspy.settings.configure(track_usage=True)`

Once enabled, you can access usage statistics from any `dspy.Prediction` object:

`[](https://dspy.ai/learn/programming/modules/#__codelineno-21-1)usage = prediction_instance.get_lm_usage()`

The usage data is returned as a dictionary that maps each language model name to its usage statistics. Here's a complete example:

`[](https://dspy.ai/learn/programming/modules/#__codelineno-22-1)import dspy [](https://dspy.ai/learn/programming/modules/#__codelineno-22-2) [](https://dspy.ai/learn/programming/modules/#__codelineno-22-3)# Configure DSPy with tracking enabled [](https://dspy.ai/learn/programming/modules/#__codelineno-22-4)dspy.settings.configure( [](https://dspy.ai/learn/programming/modules/#__codelineno-22-5)    lm=dspy.LM("openai/gpt-4o-mini", cache=False), [](https://dspy.ai/learn/programming/modules/#__codelineno-22-6)    track_usage=True [](https://dspy.ai/learn/programming/modules/#__codelineno-22-7)) [](https://dspy.ai/learn/programming/modules/#__codelineno-22-8) [](https://dspy.ai/learn/programming/modules/#__codelineno-22-9)# Define a simple program that makes multiple LM calls [](https://dspy.ai/learn/programming/modules/#__codelineno-22-10)class MyProgram(dspy.Module): [](https://dspy.ai/learn/programming/modules/#__codelineno-22-11)    def __init__(self): [](https://dspy.ai/learn/programming/modules/#__codelineno-22-12)        self.predict1 = dspy.ChainOfThought("question -> answer") [](https://dspy.ai/learn/programming/modules/#__codelineno-22-13)        self.predict2 = dspy.ChainOfThought("question, answer -> score") [](https://dspy.ai/learn/programming/modules/#__codelineno-22-14) [](https://dspy.ai/learn/programming/modules/#__codelineno-22-15)    def __call__(self, question: str) -> str: [](https://dspy.ai/learn/programming/modules/#__codelineno-22-16)        answer = self.predict1(question=question) [](https://dspy.ai/learn/programming/modules/#__codelineno-22-17)        score = self.predict2(question=question, answer=answer) [](https://dspy.ai/learn/programming/modules/#__codelineno-22-18)        return score [](https://dspy.ai/learn/programming/modules/#__codelineno-22-19) [](https://dspy.ai/learn/programming/modules/#__codelineno-22-20)# Run the program and check usage [](https://dspy.ai/learn/programming/modules/#__codelineno-22-21)program = MyProgram() [](https://dspy.ai/learn/programming/modules/#__codelineno-22-22)output = program(question="What is the capital of France?") [](https://dspy.ai/learn/programming/modules/#__codelineno-22-23)print(output.get_lm_usage())`

This will output usage statistics like:

`[](https://dspy.ai/learn/programming/modules/#__codelineno-23-1){ [](https://dspy.ai/learn/programming/modules/#__codelineno-23-2)    'openai/gpt-4o-mini': { [](https://dspy.ai/learn/programming/modules/#__codelineno-23-3)        'completion_tokens': 61, [](https://dspy.ai/learn/programming/modules/#__codelineno-23-4)        'prompt_tokens': 260, [](https://dspy.ai/learn/programming/modules/#__codelineno-23-5)        'total_tokens': 321, [](https://dspy.ai/learn/programming/modules/#__codelineno-23-6)        'completion_tokens_details': { [](https://dspy.ai/learn/programming/modules/#__codelineno-23-7)            'accepted_prediction_tokens': 0, [](https://dspy.ai/learn/programming/modules/#__codelineno-23-8)            'audio_tokens': 0, [](https://dspy.ai/learn/programming/modules/#__codelineno-23-9)            'reasoning_tokens': 0, [](https://dspy.ai/learn/programming/modules/#__codelineno-23-10)            'rejected_prediction_tokens': 0, [](https://dspy.ai/learn/programming/modules/#__codelineno-23-11)            'text_tokens': None [](https://dspy.ai/learn/programming/modules/#__codelineno-23-12)        }, [](https://dspy.ai/learn/programming/modules/#__codelineno-23-13)        'prompt_tokens_details': { [](https://dspy.ai/learn/programming/modules/#__codelineno-23-14)            'audio_tokens': 0, [](https://dspy.ai/learn/programming/modules/#__codelineno-23-15)            'cached_tokens': 0, [](https://dspy.ai/learn/programming/modules/#__codelineno-23-16)            'text_tokens': None, [](https://dspy.ai/learn/programming/modules/#__codelineno-23-17)            'image_tokens': None [](https://dspy.ai/learn/programming/modules/#__codelineno-23-18)        } [](https://dspy.ai/learn/programming/modules/#__codelineno-23-19)    } [](https://dspy.ai/learn/programming/modules/#__codelineno-23-20)}`

When using DSPy's caching features (either in-memory or on-disk via litellm), cached responses won't count toward usage statistics. For example:

`[](https://dspy.ai/learn/programming/modules/#__codelineno-24-1)# Enable caching [](https://dspy.ai/learn/programming/modules/#__codelineno-24-2)dspy.settings.configure( [](https://dspy.ai/learn/programming/modules/#__codelineno-24-3)    lm=dspy.LM("openai/gpt-4o-mini", cache=True), [](https://dspy.ai/learn/programming/modules/#__codelineno-24-4)    track_usage=True [](https://dspy.ai/learn/programming/modules/#__codelineno-24-5)) [](https://dspy.ai/learn/programming/modules/#__codelineno-24-6) [](https://dspy.ai/learn/programming/modules/#__codelineno-24-7)program = MyProgram() [](https://dspy.ai/learn/programming/modules/#__codelineno-24-8) [](https://dspy.ai/learn/programming/modules/#__codelineno-24-9)# First call - will show usage statistics [](https://dspy.ai/learn/programming/modules/#__codelineno-24-10)output = program(question="What is the capital of Zambia?") [](https://dspy.ai/learn/programming/modules/#__codelineno-24-11)print(output.get_lm_usage())  # Shows token usage [](https://dspy.ai/learn/programming/modules/#__codelineno-24-12) [](https://dspy.ai/learn/programming/modules/#__codelineno-24-13)# Second call - same question, will use cache [](https://dspy.ai/learn/programming/modules/#__codelineno-24-14)output = program(question="What is the capital of Zambia?") [](https://dspy.ai/learn/programming/modules/#__codelineno-24-15)print(output.get_lm_usage())  # Shows empty dict: {}`


-----


# Evaluation in DSPy

Once you have an initial system, it's time to **collect an initial development set** so you can refine it more systematically. Even 20 input examples of your task can be useful, though 200 goes a long way. Depending on your _metric_, you either just need inputs and no labels at all, or you need inputs and the _final_ outputs of your system. (You almost never need labels for the intermediate steps in your program in DSPy.) You can probably find datasets that are adjacent to your task on, say, HuggingFace datasets or in a naturally occuring source like StackExchange. If there's data whose licenses are permissive enough, we suggest you use them. Otherwise, you can label a few examples by hand or start deploying a demo of your system and collect initial data that way.

Next, you should **define your DSPy metric**. What makes outputs from your system good or bad? Invest in defining metrics and improving them incrementally over time; it's hard to consistently improve what you aren't able to define. A metric is a function that takes examples from your data and takes the output of your system, and returns a score. For simple tasks, this could be just "accuracy", e.g. for simple classification or short-form QA tasks. For most applications, your system will produce long-form outputs, so your metric will be a smaller DSPy program that checks multiple properties of the output. Getting this right on the first try is unlikely: start with something simple and iterate.

Now that you have some data and a metric, run development evaluations on your pipeline designs to understand their tradeoffs. Look at the outputs and the metric scores. This will probably allow you to spot any major issues, and it will define a baseline for your next steps.

If your metric is itself a DSPy program...

If your metric is itself a DSPy program, a powerful way to iterate is to optimize your metric itself. That's usually easy because the output of the metric is usually a simple value (e.g., a score out of 5), so the metric's metric is easy to define and optimize by collecting a few examples.

 Back to top

[

Previous

Modules



](https://dspy.ai/learn/programming/modules/)[

Next

Data Handling

](https://dspy.ai/learn/evaluation/data/)

© 2025 [DSPy](https://github.com/stanfordnlp)

Made with [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/)

[](https://github.com/stanfordnlp/dspy "github.com")[](https://discord.gg/XCGy2WDCQB "discord.gg")

Ask AI


-------


# Data

DSPy is a machine learning framework, so working in it involves training sets, development sets, and test sets. For each example in your data, we distinguish typically between three types of values: the inputs, the intermediate labels, and the final label. You can use DSPy effectively without any intermediate or final labels, but you will need at least a few example inputs.

## DSPy `Example` objects

The core data type for data in DSPy is `Example`. You will use **Examples** to represent items in your training set and test set.

DSPy **Examples** are similar to Python `dict`s but have a few useful utilities. Your DSPy modules will return values of the type `Prediction`, which is a special sub-class of `Example`.

When you use DSPy, you will do a lot of evaluation and optimization runs. Your individual datapoints will be of type `Example`:

`[](https://dspy.ai/learn/evaluation/data/#__codelineno-0-1)qa_pair = dspy.Example(question="This is a question?", answer="This is an answer.") [](https://dspy.ai/learn/evaluation/data/#__codelineno-0-2) [](https://dspy.ai/learn/evaluation/data/#__codelineno-0-3)print(qa_pair) [](https://dspy.ai/learn/evaluation/data/#__codelineno-0-4)print(qa_pair.question) [](https://dspy.ai/learn/evaluation/data/#__codelineno-0-5)print(qa_pair.answer)`

**Output:**

`[](https://dspy.ai/learn/evaluation/data/#__codelineno-1-1)Example({'question': 'This is a question?', 'answer': 'This is an answer.'}) (input_keys=None) [](https://dspy.ai/learn/evaluation/data/#__codelineno-1-2)This is a question? [](https://dspy.ai/learn/evaluation/data/#__codelineno-1-3)This is an answer.`

Examples can have any field keys and any value types, though usually values are strings.

`[](https://dspy.ai/learn/evaluation/data/#__codelineno-2-1)object = Example(field1=value1, field2=value2, field3=value3, ...)`

You can now express your training set for example as:

`[](https://dspy.ai/learn/evaluation/data/#__codelineno-3-1)trainset = [dspy.Example(report="LONG REPORT 1", summary="short summary 1"), ...]`

### Specifying Input Keys

In traditional ML, there are separated "inputs" and "labels".

In DSPy, the `Example` objects have a `with_inputs()` method, which can mark specific fields as inputs. (The rest are just metadata or labels.)

`[](https://dspy.ai/learn/evaluation/data/#__codelineno-4-1)# Single Input. [](https://dspy.ai/learn/evaluation/data/#__codelineno-4-2)print(qa_pair.with_inputs("question")) [](https://dspy.ai/learn/evaluation/data/#__codelineno-4-3) [](https://dspy.ai/learn/evaluation/data/#__codelineno-4-4)# Multiple Inputs; be careful about marking your labels as inputs unless you mean it. [](https://dspy.ai/learn/evaluation/data/#__codelineno-4-5)print(qa_pair.with_inputs("question", "answer"))`

Values can be accessed using the `.`(dot) operator. You can access the value of key `name` in defined object `Example(name="John Doe", job="sleep")` through `object.name`.

To access or exclude certain keys, use `inputs()` and `labels()` methods to return new Example objects containing only input or non-input keys, respectively.

`[](https://dspy.ai/learn/evaluation/data/#__codelineno-5-1)article_summary = dspy.Example(article= "This is an article.", summary= "This is a summary.").with_inputs("article") [](https://dspy.ai/learn/evaluation/data/#__codelineno-5-2) [](https://dspy.ai/learn/evaluation/data/#__codelineno-5-3)input_key_only = article_summary.inputs() [](https://dspy.ai/learn/evaluation/data/#__codelineno-5-4)non_input_key_only = article_summary.labels() [](https://dspy.ai/learn/evaluation/data/#__codelineno-5-5) [](https://dspy.ai/learn/evaluation/data/#__codelineno-5-6)print("Example object with Input fields only:", input_key_only) [](https://dspy.ai/learn/evaluation/data/#__codelineno-5-7)print("Example object with Non-Input fields only:", non_input_key_only)`

**Output**

`[](https://dspy.ai/learn/evaluation/data/#__codelineno-6-1)Example object with Input fields only: Example({'article': 'This is an article.'}) (input_keys=None) [](https://dspy.ai/learn/evaluation/data/#__codelineno-6-2)Example object with Non-Input fields only: Example({'summary': 'This is a summary.'}) (input_keys=None)`

----


# Metrics

DSPy is a machine learning framework, so you must think about your **automatic metrics** for evaluation (to track your progress) and optimization (so DSPy can make your programs more effective).

## What is a metric and how do I define a metric for my task?

A metric is just a function that will take examples from your data and the output of your system and return a score that quantifies how good the output is. What makes outputs from your system good or bad?

For simple tasks, this could be just "accuracy" or "exact match" or "F1 score". This may be the case for simple classification or short-form QA tasks.

However, for most applications, your system will output long-form outputs. There, your metric should probably be a smaller DSPy program that checks multiple properties of the output (quite possibly using AI feedback from LMs).

Getting this right on the first try is unlikely, but you should start with something simple and iterate.

## Simple metrics

A DSPy metric is just a function in Python that takes `example` (e.g., from your training or dev set) and the output `pred` from your DSPy program, and outputs a `float` (or `int` or `bool`) score.

Your metric should also accept an optional third argument called `trace`. You can ignore this for a moment, but it will enable some powerful tricks if you want to use your metric for optimization.

Here's a simple example of a metric that's comparing `example.answer` and `pred.answer`. This particular metric will return a `bool`.

`[](https://dspy.ai/learn/evaluation/metrics/#__codelineno-0-1)def validate_answer(example, pred, trace=None): [](https://dspy.ai/learn/evaluation/metrics/#__codelineno-0-2)    return example.answer.lower() == pred.answer.lower()`

Some people find these utilities (built-in) convenient:

- `dspy.evaluate.metrics.answer_exact_match`
- `dspy.evaluate.metrics.answer_passage_match`

Your metrics could be more complex, e.g. check for multiple properties. The metric below will return a `float` if `trace is None` (i.e., if it's used for evaluation or optimization), and will return a `bool` otherwise (i.e., if it's used to bootstrap demonstrations).

`[](https://dspy.ai/learn/evaluation/metrics/#__codelineno-1-1)def validate_context_and_answer(example, pred, trace=None): [](https://dspy.ai/learn/evaluation/metrics/#__codelineno-1-2)    # check the gold label and the predicted answer are the same [](https://dspy.ai/learn/evaluation/metrics/#__codelineno-1-3)    answer_match = example.answer.lower() == pred.answer.lower() [](https://dspy.ai/learn/evaluation/metrics/#__codelineno-1-4) [](https://dspy.ai/learn/evaluation/metrics/#__codelineno-1-5)    # check the predicted answer comes from one of the retrieved contexts [](https://dspy.ai/learn/evaluation/metrics/#__codelineno-1-6)    context_match = any((pred.answer.lower() in c) for c in pred.context) [](https://dspy.ai/learn/evaluation/metrics/#__codelineno-1-7) [](https://dspy.ai/learn/evaluation/metrics/#__codelineno-1-8)    if trace is None: # if we're doing evaluation or optimization [](https://dspy.ai/learn/evaluation/metrics/#__codelineno-1-9)        return (answer_match + context_match) / 2.0 [](https://dspy.ai/learn/evaluation/metrics/#__codelineno-1-10)    else: # if we're doing bootstrapping, i.e. self-generating good demonstrations of each step [](https://dspy.ai/learn/evaluation/metrics/#__codelineno-1-11)        return answer_match and context_match`

Defining a good metric is an iterative process, so doing some initial evaluations and looking at your data and outputs is key.

## Evaluation

Once you have a metric, you can run evaluations in a simple Python loop.

`[](https://dspy.ai/learn/evaluation/metrics/#__codelineno-2-1)scores = [] [](https://dspy.ai/learn/evaluation/metrics/#__codelineno-2-2)for x in devset: [](https://dspy.ai/learn/evaluation/metrics/#__codelineno-2-3)    pred = program(**x.inputs()) [](https://dspy.ai/learn/evaluation/metrics/#__codelineno-2-4)    score = metric(x, pred) [](https://dspy.ai/learn/evaluation/metrics/#__codelineno-2-5)    scores.append(score)`

If you need some utilities, you can also use the built-in `Evaluate` utility. It can help with things like parallel evaluation (multiple threads) or showing you a sample of inputs/outputs and the metric scores.

`[](https://dspy.ai/learn/evaluation/metrics/#__codelineno-3-1)from dspy.evaluate import Evaluate [](https://dspy.ai/learn/evaluation/metrics/#__codelineno-3-2) [](https://dspy.ai/learn/evaluation/metrics/#__codelineno-3-3)# Set up the evaluator, which can be re-used in your code. [](https://dspy.ai/learn/evaluation/metrics/#__codelineno-3-4)evaluator = Evaluate(devset=YOUR_DEVSET, num_threads=1, display_progress=True, display_table=5) [](https://dspy.ai/learn/evaluation/metrics/#__codelineno-3-5) [](https://dspy.ai/learn/evaluation/metrics/#__codelineno-3-6)# Launch evaluation. [](https://dspy.ai/learn/evaluation/metrics/#__codelineno-3-7)evaluator(YOUR_PROGRAM, metric=YOUR_METRIC)`

## Intermediate: Using AI feedback for your metric

For most applications, your system will output long-form outputs, so your metric should check multiple dimensions of the output using AI feedback from LMs.

This simple signature could come in handy.

`[](https://dspy.ai/learn/evaluation/metrics/#__codelineno-4-1)# Define the signature for automatic assessments. [](https://dspy.ai/learn/evaluation/metrics/#__codelineno-4-2)class Assess(dspy.Signature): [](https://dspy.ai/learn/evaluation/metrics/#__codelineno-4-3)    """Assess the quality of a tweet along the specified dimension.""" [](https://dspy.ai/learn/evaluation/metrics/#__codelineno-4-4) [](https://dspy.ai/learn/evaluation/metrics/#__codelineno-4-5)    assessed_text = dspy.InputField() [](https://dspy.ai/learn/evaluation/metrics/#__codelineno-4-6)    assessment_question = dspy.InputField() [](https://dspy.ai/learn/evaluation/metrics/#__codelineno-4-7)    assessment_answer: bool = dspy.OutputField()`

For example, below is a simple metric that checks a generated tweet (1) answers a given question correctly and (2) whether it's also engaging. We also check that (3) `len(tweet) <= 280` characters.

``[](https://dspy.ai/learn/evaluation/metrics/#__codelineno-5-1)def metric(gold, pred, trace=None): [](https://dspy.ai/learn/evaluation/metrics/#__codelineno-5-2)    question, answer, tweet = gold.question, gold.answer, pred.output [](https://dspy.ai/learn/evaluation/metrics/#__codelineno-5-3) [](https://dspy.ai/learn/evaluation/metrics/#__codelineno-5-4)    engaging = "Does the assessed text make for a self-contained, engaging tweet?" [](https://dspy.ai/learn/evaluation/metrics/#__codelineno-5-5)    correct = f"The text should answer `{question}` with `{answer}`. Does the assessed text contain this answer?" [](https://dspy.ai/learn/evaluation/metrics/#__codelineno-5-6) [](https://dspy.ai/learn/evaluation/metrics/#__codelineno-5-7)    correct =  dspy.Predict(Assess)(assessed_text=tweet, assessment_question=correct) [](https://dspy.ai/learn/evaluation/metrics/#__codelineno-5-8)    engaging = dspy.Predict(Assess)(assessed_text=tweet, assessment_question=engaging) [](https://dspy.ai/learn/evaluation/metrics/#__codelineno-5-9) [](https://dspy.ai/learn/evaluation/metrics/#__codelineno-5-10)    correct, engaging = [m.assessment_answer for m in [correct, engaging]] [](https://dspy.ai/learn/evaluation/metrics/#__codelineno-5-11)    score = (correct + engaging) if correct and (len(tweet) <= 280) else 0 [](https://dspy.ai/learn/evaluation/metrics/#__codelineno-5-12) [](https://dspy.ai/learn/evaluation/metrics/#__codelineno-5-13)    if trace is not None: return score >= 2 [](https://dspy.ai/learn/evaluation/metrics/#__codelineno-5-14)    return score / 2.0``

When compiling, `trace is not None`, and we want to be strict about judging things, so we will only return `True` if `score >= 2`. Otherwise, we return a score out of 1.0 (i.e., `score / 2.0`).

## Advanced: Using a DSPy program as your metric

If your metric is itself a DSPy program, one of the most powerful ways to iterate is to compile (optimize) your metric itself. That's usually easy because the output of the metric is usually a simple value (e.g., a score out of 5) so the metric's metric is easy to define and optimize by collecting a few examples.

### Advanced: Accessing the `trace`

When your metric is used during evaluation runs, DSPy will not try to track the steps of your program.

But during compiling (optimization), DSPy will trace your LM calls. The trace will contain inputs/outputs to each DSPy predictor and you can leverage that to validate intermediate steps for optimization.

`[](https://dspy.ai/learn/evaluation/metrics/#__codelineno-6-1)def validate_hops(example, pred, trace=None): [](https://dspy.ai/learn/evaluation/metrics/#__codelineno-6-2)    hops = [example.question] + [outputs.query for *_, outputs in trace if 'query' in outputs] [](https://dspy.ai/learn/evaluation/metrics/#__codelineno-6-3) [](https://dspy.ai/learn/evaluation/metrics/#__codelineno-6-4)    if max([len(h) for h in hops]) > 100: return False [](https://dspy.ai/learn/evaluation/metrics/#__codelineno-6-5)    if any(dspy.evaluate.answer_exact_match_str(hops[idx], hops[:idx], frac=0.8) for idx in range(2, len(hops))): return False [](https://dspy.ai/learn/evaluation/metrics/#__codelineno-6-6) [](https://dspy.ai/learn/evaluation/metrics/#__codelineno-6-7)    return True`


----

# Optimization in DSPy

Once you have a system and a way to evaluate it, you can use DSPy optimizers to tune the prompts or weights in your program. Now it's useful to expand your data collection effort into building a training set and a held-out test set, in addition to the development set you've been using for exploration. For the training set (and its subset, validation set), you can often get substantial value out of 30 examples, but aim for at least 300 examples. Some optimizers accept a `trainset` only. Others ask for a `trainset` and a `valset`. For prompt optimizers, we suggest starting with a 20% split for training and 80% for validation, which is often the _opposite_ of what one does for DNNs.

After your first few optimization runs, you are either very happy with everything or you've made a lot of progress but you don't like something about the final program or the metric. At this point, go back to step 1 (Programming in DSPy) and revisit the major questions. Did you define your task well? Do you need to collect (or find online) more data for your problem? Do you want to update your metric? And do you want to use a more sophisticated optimizer? Do you need to consider advanced features like DSPy Assertions? Or, perhaps most importantly, do you want to add some more complexity or steps in your DSPy program itself? Do you want to use multiple optimizers in a sequence?

Iterative development is key. DSPy gives you the pieces to do that incrementally: iterating on your data, your program structure, your assertions, your metric, and your optimization steps. Optimizing complex LM programs is an entirely new paradigm that only exists in DSPy at the time of writing (update: there are now numerous DSPy extension frameworks, so this part is no longer true :-), so naturally the norms around what to do are still emerging. If you need help, we recently created a [Discord server](https://discord.gg/XCGy2WDCQB) for the community.


------


# DSPy Optimizers (formerly Teleprompters)

A **DSPy optimizer** is an algorithm that can tune the parameters of a DSPy program (i.e., the prompts and/or the LM weights) to maximize the metrics you specify, like accuracy.

A typical DSPy optimizer takes three things:

- Your **DSPy program**. This may be a single module (e.g., `dspy.Predict`) or a complex multi-module program.
    
- Your **metric**. This is a function that evaluates the output of your program, and assigns it a score (higher is better).
    
- A few **training inputs**. This may be very small (i.e., only 5 or 10 examples) and incomplete (only inputs to your program, without any labels).
    

If you happen to have a lot of data, DSPy can leverage that. But you can start small and get strong results.

**Note:** Formerly called teleprompters. We are making an official name update, which will be reflected throughout the library and documentation.

## What does a DSPy Optimizer tune? How does it tune them?

Different optimizers in DSPy will tune your program's quality by **synthesizing good few-shot examples** for every module, like `dspy.BootstrapRS`,[1](https://arxiv.org/abs/2310.03714) **proposing and intelligently exploring better natural-language instructions** for every prompt, like `dspy.MIPROv2`,[2](https://arxiv.org/abs/2406.11695) and **building datasets for your modules and using them to finetune the LM weights** in your system, like `dspy.BootstrapFinetune`.[3](https://arxiv.org/abs/2407.10930)

What's an example of a DSPy optimizer? How do different optimizers work?

## What DSPy Optimizers are currently available?

Optimizers can be accessed via `from dspy.teleprompt import *`.

### Automatic Few-Shot Learning

These optimizers extend the signature by automatically generating and including **optimized** examples within the prompt sent to the model, implementing few-shot learning.

1. [**`LabeledFewShot`**](https://dspy.ai/api/optimizers/LabeledFewShot): Simply constructs few-shot examples (demos) from provided labeled input and output data points. Requires `k` (number of examples for the prompt) and `trainset` to randomly select `k` examples from.
    
2. [**`BootstrapFewShot`**](https://dspy.ai/api/optimizers/BootstrapFewshot): Uses a `teacher` module (which defaults to your program) to generate complete demonstrations for every stage of your program, along with labeled examples in `trainset`. Parameters include `max_labeled_demos` (the number of demonstrations randomly selected from the `trainset`) and `max_bootstrapped_demos` (the number of additional examples generated by the `teacher`). The bootstrapping process employs the metric to validate demonstrations, including only those that pass the metric in the "compiled" prompt. Advanced: Supports using a `teacher` program that is a _different_ DSPy program that has compatible structure, for harder tasks.
    
3. [**`BootstrapFewShotWithRandomSearch`**](https://dspy.ai/api/optimizers/BootstrapFewshotWithRandomSearch): Applies `BootstrapFewShot` several times with random search over generated demonstrations, and selects the best program over the optimization. Parameters mirror those of `BootstrapFewShot`, with the addition of `num_candidate_programs`, which specifies the number of random programs evaluated over the optimization, including candidates of the uncompiled program, `LabeledFewShot` optimized program, `BootstrapFewShot` compiled program with unshuffled examples and `num_candidate_programs` of `BootstrapFewShot` compiled programs with randomized example sets.
    
4. [**`KNNFewShot`**](https://dspy.ai/api/optimizers/KNNFewShot/). Uses k-Nearest Neighbors algorithm to find the nearest training example demonstrations for a given input example. These nearest neighbor demonstrations are then used as the trainset for the BootstrapFewShot optimization process.
    

### Automatic Instruction Optimization

These optimizers produce optimal instructions for the prompt and, in the case of MIPROv2 can also optimize the set of few-shot demonstrations.

1. [**`COPRO`**](https://dspy.ai/api/optimizers/COPRO): Generates and refines new instructions for each step, and optimizes them with coordinate ascent (hill-climbing using the metric function and the `trainset`). Parameters include `depth` which is the number of iterations of prompt improvement the optimizer runs over.
    
2. [**`MIPROv2`**](https://dspy.ai/api/optimizers/MIPROv2): Generates instructions _and_ few-shot examples in each step. The instruction generation is data-aware and demonstration-aware. Uses Bayesian Optimization to effectively search over the space of generation instructions/demonstrations across your modules.
    

### Automatic Finetuning

This optimizer is used to fine-tune the underlying LLM(s).

1. [**`BootstrapFinetune`**](https://dspy.ai/api/optimizers/BootstrapFinetune): Distills a prompt-based DSPy program into weight updates. The output is a DSPy program that has the same steps, but where each step is conducted by a finetuned model instead of a prompted LM.

### Program Transformations

1. [**`Ensemble`**](https://dspy.ai/api/optimizers/Ensemble): Ensembles a set of DSPy programs and either uses the full set or randomly samples a subset into a single program.

## Which optimizer should I use?

Ultimately, finding the ‘right’ optimizer to use & the best configuration for your task will require experimentation. Success in DSPy is still an iterative process - getting the best performance on your task will require you to explore and iterate.

That being said, here's the general guidance on getting started:

- If you have **very few examples** (around 10), start with `BootstrapFewShot`.
- If you have **more data** (50 examples or more), try `BootstrapFewShotWithRandomSearch`.
- If you prefer to do **instruction optimization only** (i.e. you want to keep your prompt 0-shot), use `MIPROv2` [configured for 0-shot optimization to optimize](https://dspy.ai/api/optimizers/MIPROv2#optimizing-instructions-only-with-miprov2-0-shot).
- If you’re willing to use more inference calls to perform **longer optimization runs** (e.g. 40 trials or more), and have enough data (e.g. 200 examples or more to prevent overfitting) then try `MIPROv2`.
- If you have been able to use one of these with a large LM (e.g., 7B parameters or above) and need a very **efficient program**, finetune a small LM for your task with `BootstrapFinetune`.

## How do I use an optimizer?

They all share this general interface, with some differences in the keyword arguments (hyperparameters). Detailed documentation for key optimizers can be found [here](https://dspy.ai/api/optimizers/vfrs), and a full list can be found [here](https://dspy.ai/api/optimizers/BetterTogether/).

Let's see this with the most common one, `BootstrapFewShotWithRandomSearch`.

`[](https://dspy.ai/learn/optimization/optimizers/#__codelineno-0-1)from dspy.teleprompt import BootstrapFewShotWithRandomSearch [](https://dspy.ai/learn/optimization/optimizers/#__codelineno-0-2) [](https://dspy.ai/learn/optimization/optimizers/#__codelineno-0-3)# Set up the optimizer: we want to "bootstrap" (i.e., self-generate) 8-shot examples of your program's steps. [](https://dspy.ai/learn/optimization/optimizers/#__codelineno-0-4)# The optimizer will repeat this 10 times (plus some initial attempts) before selecting its best attempt on the devset. [](https://dspy.ai/learn/optimization/optimizers/#__codelineno-0-5)config = dict(max_bootstrapped_demos=4, max_labeled_demos=4, num_candidate_programs=10, num_threads=4) [](https://dspy.ai/learn/optimization/optimizers/#__codelineno-0-6) [](https://dspy.ai/learn/optimization/optimizers/#__codelineno-0-7)teleprompter = BootstrapFewShotWithRandomSearch(metric=YOUR_METRIC_HERE, **config) [](https://dspy.ai/learn/optimization/optimizers/#__codelineno-0-8)optimized_program = teleprompter.compile(YOUR_PROGRAM_HERE, trainset=YOUR_TRAINSET_HERE)`

Getting Started III: Optimizing the LM prompts or weights in DSPy programs

A typical simple optimization run costs on the order of $2 USD and takes around ten minutes, but be careful when running optimizers with very large LMs or very large datasets. Optimizer runs can cost as little as a few cents or up to tens of dollars, depending on your LM, dataset, and configuration.

[Optimizing prompts for a ReAct agent](https://dspy.ai/learn/optimization/optimizers/#__tabbed_1_1)[Optimizing prompts for RAG](https://dspy.ai/learn/optimization/optimizers/#__tabbed_1_2)[Optimizing weights for Classification](https://dspy.ai/learn/optimization/optimizers/#__tabbed_1_3)

This is a minimal but fully runnable example of setting up a `dspy.ReAct` agent that answers questions via search from Wikipedia and then optimizing it using `dspy.MIPROv2` in the cheap `light` mode on 500 question-answer pairs sampled from the `HotPotQA` dataset.

|   |   |
|---|---|
|[1](https://dspy.ai/learn/optimization/optimizers/#__codelineno-1-1)<br> [2](https://dspy.ai/learn/optimization/optimizers/#__codelineno-1-2)<br> [3](https://dspy.ai/learn/optimization/optimizers/#__codelineno-1-3)<br> [4](https://dspy.ai/learn/optimization/optimizers/#__codelineno-1-4)<br> [5](https://dspy.ai/learn/optimization/optimizers/#__codelineno-1-5)<br> [6](https://dspy.ai/learn/optimization/optimizers/#__codelineno-1-6)<br> [7](https://dspy.ai/learn/optimization/optimizers/#__codelineno-1-7)<br> [8](https://dspy.ai/learn/optimization/optimizers/#__codelineno-1-8)<br> [9](https://dspy.ai/learn/optimization/optimizers/#__codelineno-1-9)<br>[10](https://dspy.ai/learn/optimization/optimizers/#__codelineno-1-10)<br>[11](https://dspy.ai/learn/optimization/optimizers/#__codelineno-1-11)<br>[12](https://dspy.ai/learn/optimization/optimizers/#__codelineno-1-12)<br>[13](https://dspy.ai/learn/optimization/optimizers/#__codelineno-1-13)<br>[14](https://dspy.ai/learn/optimization/optimizers/#__codelineno-1-14)<br>[15](https://dspy.ai/learn/optimization/optimizers/#__codelineno-1-15)|`import dspy from dspy.datasets import HotPotQA dspy.configure(lm=dspy.LM('openai/gpt-4o-mini')) def search(query: str) -> list[str]:     """Retrieves abstracts from Wikipedia."""     results = dspy.ColBERTv2(url='http://20.102.90.50:2017/wiki17_abstracts')(query, k=3)     return [x['text'] for x in results] trainset = [x.with_inputs('question') for x in HotPotQA(train_seed=2024, train_size=500).train] react = dspy.ReAct("question -> answer", tools=[search]) tp = dspy.MIPROv2(metric=dspy.evaluate.answer_exact_match, auto="light", num_threads=24) optimized_react = tp.compile(react, trainset=trainset)`|

An informal run similar to this on DSPy 2.5.29 raises ReAct's score from 24% to 51%.

## Saving and loading optimizer output

After running a program through an optimizer, it's useful to also save it. At a later point, a program can be loaded from a file and used for inference. For this, the `load` and `save` methods can be used.

`[](https://dspy.ai/learn/optimization/optimizers/#__codelineno-6-1)optimized_program.save(YOUR_SAVE_PATH)`

The resulting file is in plain-text JSON format. It contains all the parameters and steps in the source program. You can always read it and see what the optimizer generated. You can add `save_field_meta` to additionally save the list of fields with the keys, `name`, `field_type`, `description`, and `prefix` with: `optimized_program.save(YOUR_SAVE_PATH, save_field_meta=True).

To load a program from a file, you can instantiate an object from that class and then call the load method on it.

`[](https://dspy.ai/learn/optimization/optimizers/#__codelineno-7-1)loaded_program = YOUR_PROGRAM_CLASS() [](https://dspy.ai/learn/optimization/optimizers/#__codelineno-7-2)loaded_program.load(path=YOUR_SAVE_PATH)`


---------

