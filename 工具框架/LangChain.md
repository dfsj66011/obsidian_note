
核心组件：

* Agents
* Models
* Messages
* Tools
* Short-term memory
* Streaming
* Structured output


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


会遇到两种类型的 LLM：

* 默认就能生成结构化输出的 LLM，比如 OpenAI 的 GPT 模型，使用 `with_structured_output` 函数即可，极大简化；
* 根本不具备这种能力的 LLM，它们无法为你生成结构化输出。解析器 `Output Parsers`

接下来的内容主要考虑 *第一种情况*，共有 3 种方式：

1. 使用 TypedDict：静态类型提示（供 IDE 检查），无验证，零运行开销
2. 使用 Pydantic：运行时数据验证，自动数据转换，原生支持 json、dict 互转等，有运行开销
3. 直接使用 JSON 模式

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


**RAG 最重要的组件**：

* 文档加载器
* 文本分割器
* 向量数据库
* 检索器

**文档加载器**：

常用的有：文本加载器、PDF加载器、网页加载器和 CSV 加载器。

我们需要做的是确保无论我们从哪个来源获取数据，这些数据都能转换为一种统一的格式。他们将标准化格式命名为 “document”，每个 document 对象包含两个部分，一个是 `page_content`，即实际的数据内容；另一个是围绕它的`metadata`，比如来源是什么，文件来自哪里，创建时间，最后修改时间等。

*文本加载器*：读取文本文件并将其作为文档对象引入 LangChain 中，
```python
loader = TextLoader("xxx.txt", encoding="utf-8")
docs = loader.load()    # List[Document]
docs[0].page_content
```

*PDF 加载器*：如果有一些非常简单的文本 PDF 文件，可以使用 `PyPDFLoader` 来处理它们。如果有一些像照片转换成的 PDF 那样的文件，那么对于这类情况，还有其他文档加载器可供选择。
```python
loader = PyPDFLoader("xxx.pdf")
docs = loader.load()      # 按页码，比如 23 个
```

`PyPDFLoader` 在很多场景下效果并不理想，比如 PDF 是由扫描图片组成的，例如，如果处理的 PDF 包含大量表格结构，并且需要提取表格数据，可以使用 `PDFPlumberLoader` 的加载器；如果想处理扫描图像类的 PDF，可以选择`unstructuredPDFLoader` 或 `AmazonTextractPDFLoader`。如果 PDF 中有很多布局，那么 `PyMuPDFLoader` 是一个选择，而对于结构提取，有 `unstructuredPDFLoader`。

但是如果我们有一个文件夹，这个文件夹中有多个 PDF 文件，怎么处理？

答案是 *DirectoryLoader*，帮助从一个目录中加载多个文档。
```python
loader = DirectoryLoader(
	path="books", 
	glob="*.pdf",
	loader_cls=PyPDFLoader
)

docs = loader.load()
```

一次性加载太多的文件，内存溢出，且耗时长，LangChain 中提供 Lazy load，返回的是生成器。

*WebBaseLoader*，借助它的帮助，抓取任何网页的内容，然后加载到链中。它使用了两个库，一个是 requests，借助它你可以向那个网页发送 HTTP 请求；另一个是 Beautiful Soup，借助它你可以理解那个网页的 HTML 结构，并将其转换为文本格式，带到 LangChain中。如果网页包含大量 JavaScript，针对这种情况，有一个单独的加载器，称为 `SeleniumURLLoader`。
```python
url = 'xxx'
loader = WebBaseLoader(url)
docs = loader.load()
```

当使用 `WebBaseLoader` 加载单个 URL 的内容时，会得到一个单一的文档。但这里你有一个灵活性，可以传递一个 URL 列表而不是单个 URL。

*CSVLoader*：
```python
loader = CSVLoader(file_path="xxx.csv")
docs = loader.load()
```

它为每一行创建一个不同的文档对象。

---------

**文本分割器**


把内容分成小块，可以按页面、按段落分成块。

第一个原因，分割使我们能够处理那些原本会超出模型上下文限制的文档。
第二个原因，构建过程中执行的各种任务，比如嵌入任务、语义搜索任务、文本摘要任务，在这些所有类型的任务中，文本分割通过提供更好的结果来执行。
第三个原因是为了优化计算资源。

四种常见的文本分割方法：

* 基于文档长度的分割；
* 基于文本结构的分割；
* 基于文档结构的分割；
* 基于语义意义的分割

*基于长度的文本分割*：最简单、最快速的文本分割方法。需要预先决定好每个块的大小。比如决定每个块的大小是 100 个字符；

但它的最大缺点是，这种特定方法在分割文本时，既不看你的文本的语言结构，也不看语法，甚至不考虑语义。简单地说，如果要在100个字符处停止，它就会在100个字符处停止。因此尽管这种方法非常快，但并不经常使用。

```python
text = "xxx"
splitter = CharacterTextSplitter(
	chunk_size=100,
	chunk_overlap=0,   # 10%-20% 合适
	separator=""
)
result = splitter.split_text(text)

---------

docs = ....   # pdf loader
result = splitter.split_documents(docs)
```


*Text-Structured Based 文本分割*：首先将文本组织成段落，然后在段落内部将文本组织成句子，在句子内部再将文本组织成单词。这就是结构层次。递归字符文本分割（recursive character text splitting），这是最常用的文本分割技术之一。

这里发生的是，首先定义一些分隔符，比如对于段落你有两个斜杠，然后对于换行你有一个斜杠，而对于空格你有一个空格。最后，如果什么都没找到，那基本上也可以按字符拆分。

```python
splitter = RecursiveCharacterTextSplitter(
	chunk_size=300,
	chunk_overlap=0,
)
```

*Document-Structured Based*：如果有一个不是纯文本的文档，就像一段代码，不能像普通纯文本那样分割它。它不是按段落组织的，也不是按句子组织的，它是通过某些关键词以不同的方式组织的。比如有一个叫做 “class” 的结构，有函数，有循环等。同样，也可以将其应用于 Markdown 等，它们有自己独特的分隔符类型。

```python
splitter = RecursiveCharacterTextSplitter.from_language(
	language=Language.PYTHON,
	chunk_size=300,
	chunk_overlap=0,
)
```

*Semantic Meaning Based*

如果这两个话题谈论的是类似的主题，它们之间的相似度就会很高；但如果它们谈论的是截然不同的主题，相似度就会很低，当突然发现某对句子的相似度非常低时，那个点就是在向你表明（某种变化或分界）。

基于语义的文本分割器正是按照这个原则工作的，它们采用滑动窗口的方法，逐个比较句子之间的相似性和语义含义，当它们发现相似性突然急剧下降时，就会意识到，这里的含义发生了变化。

但它还处于*实验阶段*，根据我的经验，无论我如何使用它，它的表现看起来都不是很准确。需要搭配嵌入模型，生成句子嵌入。

---------

**向量存储**

1. 第一个挑战是生成嵌入向量，需要为每个数据生成这些嵌入向量；
2. 第二个挑战是存储问题。
3. 第三个挑战是语义搜索，在这些向量中进行搜索，找出哪个向量最相似。

向量存储是一种旨在存储和检索以数值向量表示的数据的系统，有四个关键特性：

1. 第一个是存储，这是任何向量存储最基本的特性，
2. 相似性搜索，有助于检索与查询向量最相似的向量，
3. 索引功能，用于加速搜索过程，一种数据结构或方法，能够对高维向量进行快速相似性搜索。例如聚类方法，著名的近似最近邻查找技术等。
4. 通过它的帮助，可以执行所有的 CRUD 操作

LangChain 提供知名向量存储（如 Supabase、Pinecone、Chroma、Qdrant、Weaviate 等）的组件。在 LangChain 中所有主要的向量存储，可通过一个通用接口的帮助来设计所有内容，这样即使将来你移除一个组件并替换为另一个，也无需对代码进行重大修改。

Chroma 是一个轻量级的开源向量数据库，特别适合本地开发和中到小规模的生产需求。在 Chroma DB 中，最顶层的是租户（tenant），你可以理解为一个用户、一个组织或一个团队，这些都位于最顶层。现在这个用户可以创建多个数据库，然后在数据库内可以创建多个集合，在每个集合内部可以存储多个文档。现在文档中有两样东西，一个是嵌入向量，另一个是关于该向量的元数据。

[Google Colab Demo](https://colab.research.google.com/drive/1VwOywJ9LPSIpKWKj9vueVoexSCzGHXNC?usp=sharing)，很简单。


-----------

**检索器（Retrievers）**

它根据用户的查询从数据源中获取相关文档，数据源可能是向量存储，也可能是某个 API 或其他任何东西，这里存放了你所有的数据。检索器内部会深入数据源，扫描你所有的文档，并尝试理解哪些文档对于给定的查询是最相关的。

所有检索器都是 Runnable，就像模型、提示等一样。这意味着可以使用检索器来形成链，或者将检索器插入现有的链中。

可以基于数据源分类：不同的检索器与不同的数据源一起工作。比如，有一个名为维基百科检索器的检索器。这个检索器的特点是，它会接收查询，然后去维基百科进行搜索；基于向量存储检索器的数据源，这个特定的检索器会去向量存储中搜索哪些文档最相关。

可以基于检索的搜索策略进行分类：不同的检索器使用不同的机制来搜索文档，例如，有一种检索器叫做 MMR（最大边际相关性），还有一种叫做多查询检索器；另一种叫做上下文压缩检索器。

可以从两个方面进行考虑：一是该检索器处理的是哪种数据源，其次是他搜索时使用什么策略。

[Colab](https://colab.research.google.com/drive/1vuuIYmJeiRgFHsH-ibH_NUFjtdc5D9P6?usp=sharing)


*MMR 的核心哲学* 是：我们如何挑选出不仅与查询相关，而且彼此不同的结果。所以每当 MMR 获取并返回结果时，它总是尝试做到这一点。不仅那些翻出来的文件要与搜索查询相关，同时它们之间也要有很大的不同。MMR 是一种信息检索算法，旨在减少检索结果中的冗余，同时保持与查询的高度相关性。

首先，它会挑选出最相关的文档。接着，它挑选的下一个文档不仅相关，而且与之前选出的文档非常不同。它会一直这样进行下去，最终以这种方式为你获取并呈现文档。这就是 MMR 背后的核心理念，不仅应该提供相关结果，同时还应呈现多样化的结果。

*多查询检索器*（Multi Query Retriever）的工作原理中有一个简单的核心哲学，它试图以某种方式消除用户查询中的模糊性。当用户发送一个查询时，如果这个查询有点模糊，首先，会把这个查询发送给一个 LLM，它会从这个查询中生成多个不同的查询。

*上下文压缩检索器*：一个单一文档中同时提到 A、B 两件事，当用户查询 A 时，它会忽略掉 B 方面的内容，不会返回整个文档。

但为什么会有这样的文件呢？这是可能的，当处理非常大的文本时，当应用文本分割器时，并不能完全控制如何分割你的文本。有时候你的文本可能会以这种方式被分割，以至于段落中间被分割开了。

-----------

**RAG**

在某些特定情况下，LLMs 无法提供帮助：

1. 询问有关我的私人数据的问题
2. 询问训练数据截止日期之后的问题
3. 幻觉

有一种方法可以帮助我们解决这三个问题，*微调*，但微调的劣势显而易见，耗时耗力且技术性强，且需要频繁更新。

另一种技术叫做“上下文学习”，通过提示中的示例学习解决任务，而无需更新其权重。这种上下文学习正是 LLMs 的一种涌现特性。涌现特性是指在一个系统中突然出现的行为或能力。

现在，与其进行少量提示，不如在提示中直接发送解决某个任务的完整上下文。为了解决一个查询所需的所有上下文，将这些完整的上下文注入到我们的提示中，这样我们模型的参数化知识就在某种程度上得到了增强。这就是 *RAG*

如果从非常高的层面来说，RAG 由两个概念组成：信息检索与文本生成。所以广义上讲，RAG 可以分为四个步骤，索引、检索、增强、生成。

创建外部知识库的过程称为*索引*；尝试在整个外部知识库中寻找那些能够帮助回答这个查询的片段或部分的过程称为 *检索*；将用户的查询和检索到的上下文结合起来，这个 prompt 的创建就是 *增强*；LLM 最终文本 *生成*。

*索引的构建*：

1. 文档摄取，将源知识加载到内存中，LangChain 中文档加载的作用
2. 文本分块，将大文档分解成更小的、有语义意义的块
3. 嵌入生成
4. 向量存储

*检索过程*：从预先构建的索引（即向量存储）中实时查找与用户问题最相关的信息片段的过程。LangChain 检索器角色，

1. 为查询生成嵌入向量（相同的嵌入模型）
2. 检索策略，语义？MMR？上下文压缩？
3. rank，如基于余弦相似度等指标，或更高级的 re-rank 算法，但核心思想是对结果进行排序，提取排名靠前的文本片段，构成上下文。

*增强*，创建 prompt，不仅包括查询，也注入上下文，将查询和上下文结合起来，创建一个提示，如 “你是一个乐于助人的助手，仅从提供的上下文中回答问题，这是我发送的XXX上下文知识。如果上下文知识内容不足，只需说你不知道，以避免幻觉。”

*生成阶段*：LLM 读取 prompt 和上下文，并生成正确的回答。

[YouTube 视频解读完整示例](https://colab.research.google.com/drive/1pat55z_iiLqzInsLi3sWS2wekFCXprQW?usp=sharing)

**如何改进这个简单的 RAG 系统？** 

* 可以做一些基于用户界面的增强，比如做成 chrome 插件；
* 评估方面。怎么知道我们的系统是否在正常工作？一个非常流行的库是ragas。它会在多个指标上评估 RAG 系统，比如 faithfulness（忠实度），意思是最终生成的答案是否与你的上下文相关。此外还有答案的相关性，即生成的答案是否与问题正确相关，以及上下文精确度。还有一个类似的库叫 LangSmith，你可以用它来进行追踪。
* 改进索引部分，例如此处转录的文本是自动生成的，里面会有各种各样的错误，同样文本分割也有优化空间，此处的 FAISS 是非常基础的向量存储系统。可以使用类似 Pinecone 的解决方案。
* 检索部分，第一个阶段是预检索，使用 LLM 来重写它，很多时候用户的查询有点模糊，或者不够有意义；其次可以进行领域感知路由，例如可以设置了多个检索，领域感知路由会根据查询触发其中的一个检索，作用是进行路由选择；在检索过程中还可以进行多种操作，比如使用 MMR 这样的搜索策略。
* 检索后重新排序、上下文压缩等。
* 增强部分，Prompt 模板，上下文窗口优化
* LLM 生成，每次回答的时候，同时也要说明是从上下文的哪一部分得出的答案，称之为引用。
* 护栏（guardrailing），基本作用是防止 LLM 输出任何混乱或不恰当的内容。
* 多模态 RAG 系统。
* Agent RAG，如联网检索。
* 带记忆的等

--------------

**Tools**

工具只是一个 Python 函数，它以 LLM 能够理解的方式打包，并在需要时调用。因此，LLM 会自己思考何时需要哪个工具，然后根据自己的判断调用该工具并输入参数。工具执行完自己的工作后，会反馈给 LLM，LLM 会告诉你任务已完成。这就是整个流程。

两种工具：内置工具，自定义工具。这些工具也是 Runnables

[Colab Code](https://colab.research.google.com/drive/1GHHGsDFB5266Cc0xDsZ6OWzkB5GGSxFW?usp=sharing)

-------------

**Tool Binding**：

工具绑定是将工具与大语言模型注册在一起的步骤。为了让 LLM 知道有哪些工具可用，它还需要了解每个工具的功能以及应该使用什么输入格式。

[Colab Note](https://colab.research.google.com/drive/1-xMYU9ExZqoySEX-XHAvEaE17PCWvc9H?usp=sharing)

你可能会想：原以为工具调用意味着 LLM 会直接调用这个工具，也就是激活这个工具并获取答案——但事实并非如此。非常重要的说明："*LLM 实际上并不运行工具，它只是推荐工具和输入参数*，真正的执行是由 LangChain 或程序员处理的。"

`InjectedToolArg` 作为一个非常重要的参数注解，核心作用是安全地将某些工具参数的控制权从大语言模型手中剥离，转而在代码运行时由应用程序逻辑动态注入。即该参数在 LLM 生成的函数调用参数列表中不出现。需要后面自行添加，例如：

```python
@tool
def convert(
	base_currency_value: int, 
	conversion_rate: Annotated[float, InjectedToolArg]
) -> float:
"""
given a currency conversion rate this function calculates 
the target currency value from a given base currency value
"""
return base_currency_value * conversion_rate

ai_message.tool_calls

[{'name': 'get_conversion_factor',
  'args': {'base_currency': 'INR', 'target_currency': 'USD'},
  'id': 'call_PKL8v7zwmphzNel0MnvjjGvY',
  'type': 'tool_call'},
 {'name': 'convert',
  'args': {'base_currency_value': 10},     # 看不到 conversion_rate 这一项
  'id': 'call_vRdld30yHTKFGcTQPumgzH5u',
  'type': 'tool_call'}]
  
-------------------

if tool_call['name'] == 'convert':
	# fetch the current arg
	tool_call['args']['conversion_rate'] = conversion_rate   # 自行填补
	tool_message2 = convert.invoke(tool_call)
	messages.append(tool_message2)
```

-------------

**AI Agents（下文内容过时，-> LangGraph）：**

AI 代理是一种智能系统，它从用户那里接收一个高级目标，并通过使用外部工具、API 或知识源，自主地计划、决定并执行一系列动作，同时在整个过程中保持对多步骤的上下文推理，并适应新信息。

[Colab Note](https://colab.research.google.com/drive/1O7cdBtiP_GNXgL9Iz4LPzYfvTKtMtv25?usp=sharing)

react = reasoning + action

`hwchase17/react` standard ReAct agent prompt

```
Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}
```

React 是 AI 代理中使用的一种设计模式，它允许 LLM 在一个结构化的多步骤过程中将内部推理与外部行动交织在一起。在 React 内部，实际上做了三件事：首先是思考，其次是行动，第三是观察。

React代理的工作方式：它持续运行着“思考-行动-观察”的循环，直到它认为自己获得了最终答案时，循环才会中断，用户就能得到答案。这个完整的思维轨迹你也能看到。所以你随时都能理解代理在想什么，代理在做什么。

那么，这个完整的循环是由谁来协调的呢？AgentExecutor 负责运行这个循环。
