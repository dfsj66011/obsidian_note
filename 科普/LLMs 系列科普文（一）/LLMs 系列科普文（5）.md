
在前文中，我们讲述了什么是基础模型，并重点以 LLaMA 3.1 基础模型为例，向大家演示了它可以做什么，有哪些问题或有趣的现象。

在进入新的主题内容之前，我们再次对 *基础模型* 做一些总结：

- 这是一个基于 token 级别的互联网文档模拟器
- 它具有随机性/概率性——每次运行都会生成不同的内容
- 它能够"幻想"出互联网文档
- 它也能逐字背诵记忆中的某些训练文档（"机械复述"）
- 模型的参数有点像互联网的有损压缩文件  
    => 大量有用的世界知识被存储在网络参数中
- 通过巧妙设计提示词，你已能将其用于实际应用（例如翻译）
    - 例如：构建"少量示例"提示词并利用"上下文学习"能力，实现英语→韩语翻译应用
    - 例如：使用对话式提示词打造能回答问题的助手

但我们还能做得更好，我们希望训练像 ChatGPT 这样的大型语言模型助手 ...

## 五、后训练（Post-Training）

我们已经讨论了第一阶段，即预训练(Pre-Training) 阶段，这一阶段的核心在于：我们获取互联网文档，将其分解为这些 token，然后利用神经网络预测 token 序列。整个这一阶段的输出就是这个基础模型。

我们发现它可以应用于某些场景，但实际上我们还需要做得更好。我们需要一个助手，能够提出问题并得到模型的回答。因此，我们现在需要进入第二阶段，即所谓的*后训练阶段*。于是，我们将基础模型——我们的互联网文档模拟器——交给后训练阶段进行处理。

接下来我们将探讨几种方法，用于对这些模型进行所谓的"后训练处理"。这些后训练处理阶段的*计算成本将大幅降低*。大部分计算工作、所有大型数据中心、以及所有重型计算设备和数百万美元的开支，都集中在预训练阶段。

后训练阶段虽然成本降低，但该阶段依然极其重要，它的目标是将这个大语言模型转化为真正的助手。让我们来看看如何让模型不再简单检索网络文档，而是学会回答问题。换句话说，我们的目标是要开始构建对话思维。这些对话可以是多轮次的。

![[post_data.png|300]]

举个例子，我们可以想象对话可能是这样的。当人类问 “2+2=?” 时，助手应该回答“2+2=4”。如果人类接着问“如果把 "+" 换成 "$\times$" 会怎样”，助手可以做出相应的回答，同样地，这个例子也展示了助手可以带有某种个性，显得友善。而当人类提出模型不愿协助的请求时，模型可以给出所谓的拒绝回应，模型可以说对此无能为力。换句话说，我们现在想做的是思考一个助手应该如何与人类互动。我们想要在这些对话中编程助手及其行为。

现在，由于这是神经网络，我们不会在代码中明确编程这些内容，这一切都是通过对数据集进行神经网络训练来完成的，正因如此，我们将通过*创建对话数据集* 来隐式地训练这个助手。这里展示的是数据集中三个独立的对话示例。而实际的数据集规模要大得多。它可以进行成千上万次多轮、冗长的对话，涵盖广泛的话题。其基本运作方式是通过示例来编程助手。

但就像图例中的这些数据是从哪里来的呢？它们*来自人类标注员*。我们基本上会给人类标注员一些对话上下文，然后让他们给出在这种情况下理想助手应该给出的回答。人类会为助手在各种情境下写出理想的回答。然后我们将让模型以此为基础进行训练，模仿这类回答。具体实操上，我们将*采用预训练阶段生成的基础模型*。

### 后训练数据集（对话）

我们将舍弃现有的互联网文档数据集，转而采用一个全新的数据集——对话数据集，我们将基于这个全新的对话数据集继续训练模型。实际上，模型会迅速调整，并大致学会应如何响应人类查询的统计规律。然后在后续推理过程中，我们基本上可以引导助手获得响应，它会模仿人类标注员在这种情况下会采取的行动。

在后训练阶段，我们基本上会继续训练模型，但预训练阶段实际上可能需要大约三个月的时间，在数千台计算机上进行训练。而后训练阶段通常会短得多，比如三个小时，这是因为我们将手动创建的对话数据集比互联网上的文本数据集要小得多。因此，这个训练时间会非常短，**但从根本上说，我们只是拿基础模型，继续使用完全相同的算法、完全相同的所有东西进行训练，只不过我们把数据集换成了对话**。

那么现在的问题是，这些对话在哪里，我们如何表示它们，如何让模型看到的是对话形式而不仅仅是原始文本，然后这种训练的结果是什么，当我们谈论模型时，从某种心理意义上你能得到什么。现在让我们转向这些问题。

#### 对话的 tokenizer

让我们从对话的分词开始讨论。这些模型中的所有内容都必须转化为 token，因为一切都与 token 序列有关。那么问题来了，我们如何将对话转化为 token 序列？

为此，我们需要设计某种编码方式，这有点类似于——如果你熟悉的话——比如互联网上的 TCP/IP 数据包（当然不了解也没关系）。信息的呈现方式、所有内容的组织结构都有精确的规则和协议，这样才能确保各类数据以书面形式清晰呈现，并获得所有人的认可。如今同样的情况也发生在大型语言模型中。我们需要某种数据结构，还需要制定规则来规范这些数据结构（比如对话）如何编码为 token，又如何从 token 中解码还原。

![[tokenizer_post_data.png|600]]

我们再次打开 TickTokenizer，选择 gpt-4o 模型，这就是这段对话在模型中的表示形式。现在我们正在迭代用户和助手的两轮对话，虽然看起来有点杂乱，但实际上相当简单。

这段内容最终被转换为 token 序列的方式有点复杂，但最终，用户与助手之间的这段对话会被编码为 49 个 token。这是一个由 49 个 token 组成的一维序列，不同的模型会采用略微不同的格式或协议，目前这方面还比较混乱，但以 GPT-4o 为例，它是这样处理的：使用一个 `<|im_start|>` 的特殊 token，关于 im 是什么意思，貌似有一些不同的解释，如 "imaginary monologue of the start"（起始假想独白）的缩写，或者 input message 的简写，老实说，我也不知道为什么叫这个么东西。然后你必须指定轮到谁了，即对话角色，比如说用户，token ID 为 1428。然后有一个内部独白分隔符 `<|im_sep|>`，接着是具体的问题，然后你需要结束它 `<|im_end|>`，即想象独白的结束。

现在要提到的重要一点是，`<|im_start|>`  等符号并不是文本内容，它是一个额外添加的特殊 token，这是一个全新的 token，从未出现在预训练阶段的字典表中，即迄今为止从未参与过训练。它是我们在后训练阶段创建并引入的新 token。因此，这些特殊 token，需要被引入并与文本交错排列，以便让模型学会识别：嘿，这是一轮对话的开始...是谁的回合开始呢？是用户的回合开始。然后是用户说的话，接着用户结束发言。然后是新的一轮对话开始，这次是助手的回合。然后助手会说什么呢？巴啦巴啦，于是这段对话就被转化成了这一连串的 token。

我们原本视为某种结构化对象的对话，最终会通过这种编码转化为一维的 token 序列。因此，我们依然只是在预测序列中的下一个 token，就像之前一样，同时我们也能对对话进行建模和训练。

那么在推理阶段的测试过程中，这会是什么样子呢？假设我们已经训练好了一个模型，并且是在这类对话数据集上训练的，现在我们要进行推理。那么在推理过程中，当你在同 ChatGPT 交互时，这会是什么样子呢？实际上我们将对话文本发送给 chatgpt，其内部也会按这种格式进行内部处理，将我们的问题以及可能存在的上下文对话编码为相同的格式，所以这就是协议的大致工作原理，协议的细节并不重要。再次强调，我的目标只是向你展示，最终一切都归结为一维的 token 序列。因此，我们可以应用之前讲到的所有内容，但现在是在对话上进行训练，并且基本上也在生成对话。

#### 早期对话数据集的构建过程

接下来，我们简单介绍下这些数据集在实际中的应用，OpenAI 在 2022 年发表的 InstructGPT 论文[^1]是在这个方向上的首次尝试.

![[instructgpt.png|300]]

这是 OpenAI 首次谈到如何利用语言模型并通过对话对其进行微调。这篇论文包含了许多细节，值得和大家分享一下。

![[instructgpt34.png|600]]

在 3.4 章节，这部分讲述了他们雇佣的外包团队——这些人员来自 Upwork 或通过 ScaleAI 招募——来构建这些对话内容。因此，这里会有人工标注员参与，他们的专业工作就是创建这些对话，这些标注员被要求提出 prompt（基本上是一些问题），然后还要完成理想的助手回复。其中在附录 A2.1 章节罗列了一些示例：
![[instructgptA21.png|400]]
可以看到这里会有各种类型的，如头脑风暴（接下来我应该读哪10本科幻小说？）、分类问题（按文本讽刺程度打分，1-10分）、信息提取（从表格中提取课程名称）、文本生成类（为某种产品撰写一则创意广告，在 Facebook 上投放，目标受众为家长）等，这里有很多不同类型的提示。

他们首先想出了这些提示，然后他们还需要回答这些提示，给出了理想的助手回应。那么他们如何知道针对这些提示应该写出怎样的理想助手回应呢？

在附录 B.3 章节就能看到这里提供给人工标注人员的标注指南节选。开发语言模型的公司，比如 OpenAI，会撰写标注指南，指导人类如何创建理想的回应。这里展示的就是这类标注指南的一个节选片段。
![[instructgptB3.png|600]]

> 翻译：你收到了一份用户提交的基于文本的任务描述。该任务描述可能以明确的指令形式呈现（例如“写一个关于聪明青蛙的故事”）。任务也可能以间接方式指定，例如通过提供多个期望行为的示例（例如给定一系列电影评论及其情感倾向，然后是一条没有情感倾向的评论，你可以假设任务是预测最后一条评论的情感倾向），或者通过生成期望输出的开头（例如给定“从前有一只聪明的青蛙叫朱利叶斯”，你可以假设任务是继续这个故事）。你还将获得几个文本输出，旨在帮助用户完成任务。你的工作是评估这些输出，确保它们有帮助、真实且无害。对于大多数任务，*真实和无害* 比 *有帮助* 更重要。

从高层次来看，你是在要求人们乐于助人、诚实守信、避免伤害。如果你想了解更多，可以定位到论文 B.3 章节详细阅读。但概括来说，基本就是回答问题时要尽力提供帮助，力求真实，不要回答那些我们不希望系统在后续 ChatGPT 对话中处理的问题。

因此，大致来说，公司会制定标签说明。通常这些说明不会这么简短，可能会有数百页之多，人们需要专业地研究它们。然后他们根据这些标注说明写出理想的助手回应。正如这篇论文所述，这是一个非常依赖人工的过程。目前 OpenAI 实际上从未发布过 InstructGPT 的数据集。

但我们确实有一些开源项目在尝试遵循这种设置并收集自己的数据。比如，一个例子是之前 Open Assistant[^2] 所做的努力，这只是众多例子中的一个。

![[OpenAssistant.png|500]]

这些是网上被要求创作这类对话的人，类似于 OpenAI 让人类标注员做的工作。这里展示的是某人想出的这个提示的条目。你能写一篇关于“买方垄断”在经济学中相关性的简短介绍吗？请使用例子等说明。然后由同一个人，或可能是另一个人，来撰写回应。以下是助手对此的回复。然后，同一个人或不同的人会实际写出这个理想的回应。接着，这可能是对话如何继续的一个例子。现在，向一只狗解释它。然后你可以尝试想出一个稍微简单一点的解释或类似的东西。现在这就变成了标签，我们最终会基于此进行训练。

当然我们无法覆盖模型在推理测试时会遇到的所有可能问题，也就无法涵盖人们未来可能提出的所有提示。但如果我们拥有一些这样的示例数据集，那么在训练过程中，模型就会开始呈现出这种乐于助人、诚实无害的助手形象。这一切都是通过示例编程实现的。因此，这些都是行为示例。如果你针对这些示例行为进行对话，并且数量足够多，比如 10 万条，然后进行训练，模型就会开始理解其中的统计模式，并逐渐呈现出这种助手般的个性。当然，在测试时遇到完全相同的问题时，答案可能会一字不差地复述训练集中的内容。

但更有可能的是，模型会做出类似感觉的回应，并理解这就是你想要的答案类型。这就是我们正在做的事情。我们通过示例来编程系统，系统在统计上采用了这种乐于助人、诚实无害的助手角色，这在一定程度上反映在公司创建的标注指南中。

#### 现代对话数据集的构建过程

现在我想告诉你的是，自 InstructGPT 论文发表以来的这两三年里，前沿技术已经有了相当的进步。具体来说，现在已经不太常见人类完全靠自己来完成所有繁重的工作了。这是因为我们现在有了语言模型，这些语言模型正在帮助我们创建这些数据集和对话。因此，*现在人们很少会完全从零开始写出回答*。

更常见的情况是，他们会使用现有的 LLMs 来生成答案，然后进行编辑或类似的处理。甚至不仅仅是生成答案，也可以让其生成问题，自问自答，当然问题和答案可以来自同一个 LLMs，也可以是不同的 LLMs。

因此，现在 LLMs 已经开始以多种不同的方式渗透到这一后训练流程中。而 LLMs 基本上被普遍用于帮助创建这些庞大的对话数据集。我不想具体展示，但像 UltraChat 就是一个更现代的对话数据集的例子，其包含数百万次对话。

![[ultrachat.png|400]]

这份数据集在很大程度上是合成的，但通常也会有少量人工参与，而且这些都是以不同方式构建的。上图是这份数据集主题的一个概览，它们涵盖了极其多样化的领域，基本每一个小的颜色块都是一个主题，像这类数据集，我们称为 SFT（监督微调）数据集，而 UltraChat 只是目前众多 SFT 数据集中的一个例子。

如今借助 LLMs，你可以很便捷的拥有有各种各样的类型和来源的数据混合，它们部分是合成的，部分是人类生成的，之后的发展方向大致如此。

最后我想说的是，我希望稍微消除一些与 AI 对话的神秘感。就像你去 ChatGPT 那里问一个问题，按下回车键后，返回的内容在某种程度上与训练集中的数据统计对齐。而这些训练集，说白了，不过是人类按照标注指令播下的一粒种子。**那么，你实际上在和 ChatGPT 的什么对话呢？或者说，你该如何理解它？直白点说，它并非来自某种神奇的 AI**。

*这源于一种在统计上模仿人类标注者的机制*，而这些标注者又遵循由这些公司编写的标注指南。因此，你某种程度上是在模仿这个过程。你几乎就像是在向人类标注者提问一样。想象一下，ChatGPT 给你的回答就像是对人类标注员的一种模拟。这有点像在问：在这种对话中，人类标注员会说什么？而且，这个*人类标注员可不是随便从网上找来的普通人*，因为这些公司实际上会聘请专家。比如，当你询问有关代码等问题时，参与创建这些对话数据集的人类标注员通常都是受过教育、具备专业知识的专家。你实际上是在向那些人的模拟版本提问。所以你不是在和一个神奇的 AI 对话，而是在和一个普通的标注员交流。这个普通的标注员可能相当熟练，但你实际上是在和那种在构建这些数据集时会被雇佣的人的即时模拟版本对话。

在我们继续之前，让我再举一个具体的例子。比如，当我打开 ChatGPT，输入“推荐巴黎最值得看的五个地标”，然后按下回车。它巴啦巴啦给出输出一些内容，我该怎么理解它呢？其实这并不是某种神奇的 AI，它并没有出去研究所有的地标，然后用它无限的智慧给它们排名等等。我得到的是 OpenAI 雇佣的一个标注员的统计模拟。你可以大致这样理解。

因此，如果这个具体问题恰好出现在 OpenAI 的后训练数据集中，我很可能会看到一个答案——这个答案大概率与人类标注员为那五个地标写下的内容高度相似。那么人类标注员是如何得出这些答案的呢？他们会去网上进行一段时间的自主调研，然后列出一份清单。如果这份清单被收录进数据集，我看到的助手回复很可能就是他们提交的"标准答案"。

但若该查询不在后训练数据集中，此刻生成的回答就更具 *涌现性*——因为模型通过统计规律已理解到：训练集中出现的地标通常是知名景点、人们常去的热门地标，也是网络上被频繁讨论的地标类型。请记住，模型在互联网预训练阶段已经掌握了海量知识。它很可能见识过大量关于地标以及人们喜闻乐见事物的对话内容。

**正是这种预训练知识与后续训练数据集的结合，才造就了这种模仿能力。所以，这就是你大致可以从统计学角度理解模型背后运作原理的方式。**


[^1]: InstructGPT Paper: https://arxiv.org/pdf/2203.02155

[^2]: Open Assistant: https://huggingface.co/datasets/OpenAssistant/oasst1
	
