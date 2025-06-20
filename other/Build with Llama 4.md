
欢迎来到《基于 Llama 4 的开发》课程。本课程由 Meta 合作打造，由 Meta 人工智能团队合作伙伴工程总监 Amit Sangani 主讲。Llama 系列开源模型已助力全球开发者构建众多 AI 应用。如今借助 Llama 4 的专家混合模型，你将发现部署比以往更简单，还能通过多图提示实现更先进的多模态理解，甚至执行图像定位任务。

全新 Llama 4 模型具备更大的上下文窗口——Maverick 模型支持百万级 token，而 Scout 模型更可处理高达千万级 token，这对分析大型代码库等场景尤为实用。课程还将介绍 Llama 4 配套的新软件工具：提示优化器和合成数据生成器。没错 Andrew，Llama 4 的模型原生支持多模态，在 Scout 模型中更能处理长达千万 token 的超长上下文。

你将通过 Meta 官方 API 及其他推理服务商获得 Llama 4 的实战经验：构建能解析视觉内容、检测物体、精准回答图像定位问题的应用；学习无需分块处理即可解析整本书籍和研究论文的长文本技术。 Meta 还推出了 Llama Tools 开源工具集，本课程将带你使用其中两大最新工具：基于 DSPy 优化器的提示自动优化工具（需指定问答评估指标），以及支持多格式数据采集/生成/管理的合成数据工具包（大幅减少微调数据集制作的手动工作量）。

感谢 Meta 的 Jeff Tang、Justin Lee、Sanyam Bhutani、Kshitiz Malik，以及 DeepLearning.AI 的 Esmaeil Gargari 和 Brandon Brown 的贡献。下节课将由 MetaAI 研究团队的 Kshitiz Malik 解析 Llama 4 架构，重点讲解专家混合模型的技术原理——为何仅需激活少量权重参数即可高效运行。请继续观看下一视频深入学习。

-------

在本课程中，你将了解 Llama 模型如何从 Llama 2 发展到最新发布的 Llama 4 模型。我们还将邀请 Meta 人工智能研究团队的成员来讲解 Llama 4 的架构。好了，让我们开始吧。

Llama 最初是 FAIR 实验室的一个快速推进的研究项目，最初专注于形式数学，但团队很快发现了经过良好训练的小型模型的潜力，这促成了 Llama 模型的发布，自此推动了研究和行业领域的重大创新。新一代 Llama 模型在行业基准测试中表现出色，并引入了强大的新功能。

![[Pasted image 20250620093928.png|600]]

Meta 在 Llama 3.2 版本中新增了支持图像推理的多模态模型及可在边缘设备运行的轻量级模型，包括 11B 和 90B 参数的视觉模型，以及仅 1B 和 3B 参数的小型纯文本模型。Llama 3.3 版本进一步提升了效率，其 70B 参数的 Instruct 模型达到了 Llama 3.1 版本 405B 参数大模型的性能水准。如今 Llama 4 开启了全新纪元，包含两大专家混合模型：17B 参数的 Llama 4 Scout 采用 16 个专家模块，主打速度与成本优势；同为 17B 参数的 Llama 4 Maverick 配备 128 个专家模块，提供顶尖性能表现。

Llama 4 模型通过早期融合设计实现了*原生多模态能力*，这意味着它从一开始就能在单一统一模型中接收文本和视觉输入。实际上，这种设计将文本标记和视觉衍生标记在输入层就进行结合，并由同一个 Transformer 主干网络联合处理，而非采用独立的编码器-解码器路径。早期融合是一项重大进步，因为它使我们能够利用大量未标注的文本、图像和视频数据对模型进行联合预训练。我们还改进了 Llama 4 的视觉编码器。

![[Pasted image 20250620094717.png|600]]

这是一页关于 Llama 3 和 Llama 4 模型的对比。Llama 4 的主要改进包括：Llama 4 Maverick 的上下文长度达到了 1M，Llama 4 Scout 更是达到了 100M，而 Llama 3.1 及后续 Llama 3 模型的上下文长度仅为 128K。Llama 4 支持的语言已从 Llama 3 的 8 种扩展到 12 种。

![[Pasted image 20250620094951.png|600]]

这是 Llama 4 的模型卡，显示了其官方支持的 12 种语言。17B 个活跃参数、专家数量、活跃和非活跃专家的总参数以及最大上下文长度。

**Llama 4 架构**

Llama 4 采用了混合专家（MoE）架构。MoE 是大语言模型中常用的架构设计。随着模型参数数量或容量的增加，我们通常会看到性能提升，因为模型可以对 token 进行更复杂的转换处理。但在密集型模型中，参数增加也意味着训练和推理阶段的成本会更高。

MoE 模型通过使用条件计算来提高模型质量，同时保持计算成本可控。它们仅为每个标记激活总参数的一小部分，从而减少了单个标记所需的计算量，同时仍保留模型容量。MoE 模型的关键部分是门控网络，也称为路由器。路由器决定哪些专家被激活来处理给定的标记。

路由机制的选择对模型性能起着核心作用。Transformer 架构包含两种主要层类型：注意力层和前馈网络层（简称 FFN）。计算成本通常由 FFN 层主导。与常见做法一致，Llama 4 混合专家模型仅在 FFN 层采用条件计算，注意力层仍保持与密集模型相同。该 MoE 层和 Llama 4 配置包含若干路由专家及一个共享专家——所有标记都会经过共享专家处理，同时每个标记还会精确通过其中一个路由专家。Scout 有 16 个路由专家，而 Maverick 有 128 个。在 Scout 中，所有 FFN 层都是 MoE（混合专家），而 Maverick 则交替使用密集层和 MoE 层，以减少模型中的总参数数量。

让我们以 Llama 4 中的 MoE 层如何处理一系列 token 为例。假设我们有一个共享专家和两个路由专家。假设序列有四个 token：The quick brown fox。

第一步，tokens 会进入路由器，以决定每个 token 将被分配到哪个路由专家。我们为每个 token 和路由专家的组合计算一个路由器亲和度分数。token 会被发送到路由器分数最高的路由专家那里，同时 token 的激活值也会乘以路由器亲和度分数。在这个例子中，token：The、brown 和 fox 会进入路由专家 1，而 quick 则进入路由专家 0。

所有 token 也会传递给共享专家。最终，共享专家和路由专家的输出会被相加，从而生成 MoE 层的最终输出。以上是对 Llama 4 架构的简要概述。如果你有兴趣了解更多，可以查看我们的博客。

LlamaAPI 提供了一种简单快捷的方式来使用和构建 Llama 模型。为了便于集成，我们还提供了 Python 和 TypeScript 的轻量级 SDK，使开发者能够快速将 API 接入他们的应用程序。通过 Llama API，你可以利用最新的 Llama 模型进行开发，包括 Llama for Maverik、Scout、Llama 3.3 8b 和 70B 模型等。

我们还推出了两款新的 Llama 工具。首先是提示优化工具，它能自动为 Llama 模型优化提示词。该工具可将适用于其他大语言模型的提示词转化为专为 Llama 模型优化的版本，从而提升性能和可靠性。你将在第六课中看到具体操作演示。本课程中你将使用的另一个工具是 Meta 的合成数据工具包。你将学习如何使用该工具创建自己的高质量数据集。当你需要微调或测试模型却缺乏理想数据集时，这个工具特别有用。在后续课程中，你还能用它生成问答对、推理步骤以及其他格式的数据。下一节课我们将开始使用 Llama API 进行开发，我们课堂上见！

----

在本课程中，你将学习如何通过 Meta 官方 API 快速上手 Llama 4。让我们开始吧。Llama 4 API 无需自行搭建基础设施，提供两种调用方式：REST 风格 API 或 Meta 官方 Python 客户端。该 API 还兼容 OpenAI 客户端库，可实现不同 API 间的无缝切换。你还能立即访问 Scout 和 Maverick 等最新版Llama模型。

在笔记本中，你将首次与 Llama API 进行互动。你将学习如何设置 API 客户端、发送提示词，并通过不同的文本和图像提示示例查看结果。你还将构建一个能支持 Llama 4 所有 12 种语言的翻译聊天机器人。好了，让我们开始吧。首先导入我们的 API keys 和库。

在本课程中，你将使用 Llama API。要使用 Llama API，我们需要 llama_api_key 和 llama_base_url。在 deeplearning.ai 平台上，所有 key 都已为你设置好，所以你无需进行任何操作。此外，你还需要导入 Llama API 客户端。在本课程中，你将多次调用 Llama API，因此创建一个 Llama 辅助函数会很有帮助，以便我们可以重复使用它。这个 llama4 函数接收你的提示、图片的URL，以及默认设置为 Llama 4 Scout 的模型。

然后，根据你传递的提示和图像 URL，内容被生成，接着创建使用 Llama API 客户端的客户端。随后，基于该内容的消息被生成并传递给客户端。最后，接收并返回响应。现在让我们调用该函数，要求它用三句话简要介绍 AI 的历史。以下是响应内容。

Llama API 还兼容 OpenAI 等流行库。这次我们使用 OpenAI 兼容库来创建相同的 Llama 4 函数。为此，我们导入 OpenAI，然后这次不使用 Llama API 客户端，而是使用 OpenAI。传入 Llama API 密钥和基础 URL。其余部分保持不变。我们调用这个函数并传入相同的提示。由于温度设置为零，我们得到了相同的响应。你也可以将图像传递给 Llama 4 函数，并向 Llama 询问关于图像的问题。让我们通过一个例子来看看。 

首先，我们来实现这个显示图像的功能，它可以获取图像 URL 并显示图像。然后我们向 Llama 提问关于这张图片的内容。我们可以将图像 URL 和提示语"这张图片里有什么？"传递给 Llama 4 功能模块，并得到响应。图像描述显示有三只羊驼以及关于图像内容的更多信息。与 Llama 3.2 不同，Llama 4 原生支持多张图像输入，在最多五张图像的评估中表现良好。

现在让我们使用这第二张图像，在将其传递给 Llama 之前，我们先同时显示这两张图片。所以我们有两种不同的羊驼图像。现在你可以通过将图像的 URL 作为列表传递，让 Llama 比较两张图片。以下是响应结果：两张图片描绘的都是羊驼。这些是主要差异和一些相似之处。在下一课中，你将在更多图像用例上使用 Llama。

Llama 4 Maverick 和 Scout 分别支持最多 1M 和 10M 的上下文长度标记数，相比之前的 Llama 模型有了大幅提升。让我们以《双城记》这本约 19.3 万标记数的免费电子书为例。这里有一个问题：书的最后一句话是什么？以及这句话之前的段落内容是什么？让我们用 Llama 4 Maverick 来回答这个问题。

在后续课程中，你将处理更多长文本用例。Llama 4 的另一项主要能力是其对 12 种语言的文本理解。让我们问 Llama“你懂多少种语言？”，并要求它用它所掌握的所有语言回答。以下是 Llama 4 能够理解的 12 种语言列表。

让我们构建一个快速聊天机器人，充当实时翻译器。这就是我们的多语言聊天机器人，它能获取源语言和目标语言，并在此处形成系统消息。我们使用 OpenAI 客户端库，传递 Llama API 密钥和基础URL，获取客户端。然后，在这里使用传递的系统提示内容形成消息，接收响应后返回。让我们以英语为源语言，法语为目标语言来构建这个翻译器。开始聊天时说“你好”吧。

如你所见，系统识别出当前语言为英语，并将 "hello" 翻译为目标语言法语。随后，针对第一人称发出的问候，第二人称用英语回应。让我们再次用第二个问题测试这一流程。同样，系统先识别语言，将问题翻译成目标语言法语，再用识别出的语言回答问题。欢迎尝试用其他语言测试这款翻译聊天机器人。以上就是我们对 Llama API 的快速概览。下节课中，你们将实践几个有趣的图像理解与图像定位应用案例。好的，我们课堂上见。

------

在本课程中，你将使用 Llama 4 进行图像理解和推理。我们将通过几个实际案例展开学习——从识别图像中的物体，到基于截图编写用户界面。现在就开始吧。Llama 4 的 Scout 和 Maverick 模型能够同时处理语言和视觉输入，这意味着它们对图像的理解推理能力与处理文本时同样出色。这一特性开启了广阔的应用场景：无论是解答关于图像的疑问、描述场景内容，还是识别物体都不在话下。图像定位（Grounding）是 Llama 4 的突出功能之一，它能将提示词的不同部分与图像中的特定区域建立关联。这样可以提供更准确的答案，尤其是在问题涉及特定内容时，比如图片右上角有什么，或者众多工具图片中的测量工具在哪里。Llama 4能够找到这些物体并返回边界框的坐标。

在笔记本中，你将处理几个图像推理的用例。你将对一张包含多种工具的图片进行图像定位。你将分析 PDF 文件中的表格。你将根据用户界面的截图生成代码。你将解决一个数学谜题，并分析电脑屏幕。

好了，让我们开始吧。首先加载我们的 API 密钥。在这个实验中，除了 Lama API，我们还将使用 TogetherAI 上的 Llama 4。为此，你需要拥有 Llama 基础 URL 和 TogetherAI API密钥。所有这些在DeepLearning.AI平台上都已为您设置好，因此您无需密钥即可运行这些笔记本。现在让我们加载两个实用函数：Llama 4 API和Llama 4 together。Llama 4 together函数与您在上节课中已经实现的Llama 4函数非常相似。




For this, you need to have Llama base URL and TogetherAI API key. All of this is already set up for you
on DeepLearning.AI platform, so you do not need to have a key
to run these notebooks. Now let's load our two utils function Llama 4 API and Llama 4 together. Llama 4 together function is very similar to the Llama 4
function you already implemented in previous lesson. 


The difference is that instead of Llama
API client, you import Together, you create TogetherAI client, and pass your TogetherAI API key. The rest stays the same. Most of the inference providers
like TogetherAI often expect image data to be embedded as base64 strings
instead of request payload. This function will get the image back and will return base64
encoding of the image. Image grounding is a fundamental task
in computer vision and natural language processing that involves
identifying the specific objects or regions in an image that correspond
to a given text description. Let's see how Llama 4 does image grounding on this picture of many tools. This is our prompt. Which tools in the image
can be used for measuring length? Provide
bounding boxes for every recognized item. You can convert the tools PNG image to base64 and pass it
along with the prompt to Llama 4 function. And here are two items
that are used for measuring length, and the coordinates for the bounding boxes
for each of the items are returned. Note that the bounding marks values represent the normalized coordinates
of a bounding box in an image. To get the actual pixel coordinates, you will need to multiply
the normalized values by the width and height of the image.
In the utils file, we have two helper functions:
parse output and draw bounding boxes. The first one
will get the output of the Llama and will parse the coordinates
of the bounding boxes. And the second function
will draw bounding boxes on the image
so that you can visualize the result. Let's pass the image and prompt to Llama again
and save the result, and then pass the output, which is the result from the model to parse underscore output and get the tools
coordinates and your description. And then finally pass it
along with the original image to draw bounding boxes
and to see the result. Here is our image with bounding boxes
around two of the tools that are used
for measuring, the name of each tool, and the coordinates for the bounding box
for each of the tools. Let's work on another use case. This time you will use Llama
to analyze the table in a PDF document. We can do this in two ways: One, convert the table into text
and ask Llama questions based on that text. And second, we can take the image
of the table and use it to prompt Llama. Let's see both and compare them. Here is the PDF file.
And this is the table we want to ask questions about. To convert PDF to text, we can use this helper function
that gets the file and extracts and returns its text. Let's pass our PDF file to the PDF
to text function and get its text. Let's see the converted text for this table in the report. We can search for fourth quarter and full year 2024 financial, which is this text and display
part of the report that will have that table. Here is the extracted text
from that table. Let's now ask Llama about the 2024 operating
model using the text of the report. Here is the response operating margin for 2024 was 42%
which is correct. Now, let's repeat this using the table saved as an image. We have saved it in this file. And here is the image. We can convert the image to base64
and pass it along with the prompt to Llama 4. And here is the response.
Which is again the same 42% you got using the text of the report. Please note that although the answers in
both of these cases were the same, the reasoning is different,
and in some tricky situations you can get better
and more accurate results using the image, as the Llama will have a much better
understanding of the overall structure of the table
compared to its plain text version. Now let's use Llama to code a user interface
using the image of that interface. For this, we are going to take
a screenshot of this frame from a video on Meta's website,
and we will call it using Llama 4. This is the image of that screenshot. Let's first ask a question
to see if Llama understands this image. If I want to change the temperature
on the image, where should I click? The temperature is this slider here
and it is currently set to 0.6. And here's Llama's response: to change the temperature on the image,
you should click on the slider next to temperature and its current
value is also given. Okay, let's see
if Llama can code this interface for us. We prompted it: write a Python script that uses Gradio
to implement the chatbot UI in the image. This is the response with the instructions
on what to install and the final Python code. Let's copy and run this code. Here is the code we got from Llama, and by running it we get this interface with the sliders all implemented
and even the chatbox that you can type your message
and interact with the interface. Note that this is just the interface
that we asked Llama to implement, and this doesn't include
all the functionality that needs to be added later
for this to fully work. Now let's use Llama to solve
a math problem. Here is the problem we want to solve. To solve this,
Llama should understand the problem and how to solve it. Here is our prompt: Answer the question in the image. We pass the base64 underscore math image
in addition to the prompt to Llama. And here's the response. Llama is giving all the steps
it takes to solve this problem, and the final answer
is calculated to be 40, which is correct. Let's work on another use case where
you use Llama to analyze computer screen. Here is the image
you are going to work with. This might look familiar
as this is an image of our previous course on Llama 3.2 on DeepLearning.AI
platform. Let's ask Llama
to describe this screenshot in detail, including browser, URL and tabs. Here is a detailed analysis of this image. Even with the list of all the icons
at the bottom of the screen, let's say you have a browser agent
that wants to automatically go to the next lesson. Let's ask Llama this question. If I want to go to the next lesson,
what should I do? We display the image again,
so we have the result and image together. And here is the response to proceed
to the next lesson: Click
on the red button labeled "Next lesson" which is here located at the bottom right
corner of the screen. This use case shows
how you can use Llama in computer use applications
and in building browser AI agents. In this lesson,
you used Llama in several image reasoning and grounding use cases and applications. In the next lesson, you're going to go
deep into prompt format of Llama 4. All right. See you there.

