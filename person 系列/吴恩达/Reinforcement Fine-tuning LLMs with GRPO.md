
欢迎来到"利用 GPRO 进行强化微调大语言模型"课程，本课程由 Predibase 合作开发。在这门课程中，你将深入探索强化微调（RFT）技术——这是一种运用强化学习来提升大语言模型在需要多步推理任务（如数学运算或代码生成）中表现能力的训练方法。通过激发大语言模型逐步推理问题的能力，强化微调能引导模型自主发现复杂任务的解决方案，而非像传统监督学习那样依赖既有示例。这种方法让你只需极少量的训练数据（通常仅需监督微调所需样本量的几十分之一）就能使模型适应复杂任务。

我很荣幸向你介绍本课程的导师：Predibase 联合创始人兼首席技术官 Travis Addair，以及该公司高级机器学习工程师兼机器学习负责人 Arnav Garg。两位专家都曾运用 RFT 技术为众多客户解决实际商业问题。

"感谢 Andrew 的介绍。我们非常期待本次教学。"在本课程中，你将通过一个趣味案例——训练小型语言模型玩流行字谜游戏 Wordle（玩家需在六次尝试内猜出五个字母的单词）来探索 RFT 的工作原理。你将首先引导 Qwen-2.5-7B 模型进行游戏，分析其表现并开发奖励函数来帮助模型持续优化。这个奖励函数正是 deepseek 公司开发的"群组相对策略优化"（GRPO）算法的核心组件，该算法专为推理任务的强化学习而设计。

GRPO 的独特之处在于：大语言模型对单个提示生成多个响应后，系统会基于可验证指标（如正确格式或可运行代码）通过奖励函数进行评分。这与 PPO 或 DPO 等依赖人类反馈或多模型系统的传统强化学习算法形成鲜明对比。

完成 Wordle 案例的奖励函数设计后，你还将掌握适用于各类问题的通用奖励函数设计原则，并学习防范"奖励破解"现象——即模型为最大化奖励而采取与问题解决无关的行为。接下来课程将深入解析 RFT 过程中的损失计算技术细节，你会发现 GRPO 算法中看似复杂的裁剪操作、KL 散度和损失函数等环节，在代码实现时远比想象中简单。

课程最后，你将学习如何通过 Predibase API 使用自定义数据和奖励函数实施 RFT。在此我要感谢 Predibase 的 Michael Ortega 和 DeepLearning.AI 的 Tommy Nelson 对本课程开发的贡献。

具备优秀推理能力的大语言模型是智能体系统的核心组件，而 RFT 技术能让较小模型在智能体工作流中发挥出色。当前业界对大语言模型的这项能力充满期待，而强化学习本身作为一项强大却仍被多数人视为神秘的技术，现在正是学习如何运用它来定制推理模型的绝佳时机。相信掌握这些知识将会让你获益匪浅。接下来让我们进入下个视频，了解 RFT 与监督微调的核心差异。

----------
让我们从探索强化学习如何帮助大语言模型（LLM）通过实验和结果反馈来学习新任务开始。你将了解这一过程与监督微调的区别，并直观理解最重要强化学习算法的工作原理。现在让我们深入探讨。

传统上，我们通过称为监督微调的过程来教授LLM完成任务，如分类、命名实体识别和代码生成。首先，我们收集一个由提示和响应对组成的标注数据集，这些数据展示了我们希望LLM学习的行为。然后，在训练过程中，每个样本都会经历两个步骤：在前向传播中，模型为给定提示生成输出；在后向传播中，我们将模型的输出与正确答案进行比较，计算误差并更新模型权重以减少误差。当我们在数千个类似样本上重复这些步骤时，模型就学会了期望的行为。

监督微调的关键在于它通过示范来教导模型。例如，我们可以向模型展示一组数学题及其最终答案，模型将学会生成这些输出的模式，即使遇到以前未见过的类似数学题也能应对。对于更复杂的任务，你可以在答案旁边包含推理轨迹和思考标记。通过这样构建数据集，你可以同时教导模型两个方面：第一是输出格式，即如何使用标记将思考过程与最终答案分开；第二是逐步推理的能力，这是通过教导模型如何生成从提示到期望解决方案的逻辑链来实现的。

然而，尽管监督微调擅长许多任务，它确实存在一些局限性。要获得良好的质量改进，通常需要数千个高质量的标注样本供模型学习，而这些样本的收集可能更加困难和昂贵。另一个常见问题是过拟合现象，即模型过于完美地学习训练数据的模式，在面对未见过的样本时表现不佳。这些局限性表明我们需要一种训练方法，既能减少对大量标注的依赖，又能缓解过拟合，同时仍能引导模型达到期望行为。

强化学习就是这样一种替代方案，在这种方法中，模型通过与环境交互并优化奖励信号来学习，而不是模仿固定的标注样本。为了更好地理解这个概念，让我们仔细看看这个例子：在这个例子中，你的小狗可以采取许多不同的行动——它可以选择坐在一个地方，可以选择打滚，也可以在你扔出棍子时选择去捡。小狗通过学习明白，在所有可能的行动中，只有当它实际捡回棍子时才能获得奖励（零食）。在本课程后续内容中，我们将继续探讨这一机制。





Let's get started by exploring
how reinforcement learning can help an LLM learn a new task by experimenting
and receiving feedback on the results. You'll see how this process differs
from supervised fine-tuning, and gained intuition about how the most important reinforcement
learning algorithms work. Let's dive in. Traditionally, we teach LLMs tasks
such as classification, named entity
recognition, and code generation through a process called supervised
fine-tuning. First, we assemble a label data
set of prompt and response pairs that demonstrate the behavior
we want the LLM to learn. Then, during training, each example goes
through two steps: In the forward pass, the model generates
an output for the given prompt. Then, in the backward pass,
we compare the model's output to the correct response, compute the error and update
the model's weights to reduce that error. When we repeat these steps
across thousands of similar examples,
the model learns the desired behavior. The key aspect of supervised
fine-tuning is that it teaches the model using demonstrations. For example, we can show the model a set
of math problems and the final answer, and it will learn the patterns
to produce these outputs, even for similar math problems
that is not seen before. For more complex tasks,
you can include reasoning traces and think tags alongside your answers.
By structuring your data set, this way, you can teach the model
two aspects simultaneously. The first, is the output format. That is how to use tags to separate
thoughts from the final answer. And the second is its ability
to do step-by-step reasoning. This is done by teaching the model
how to produce the chain of logic that leads from the prompt
to the desired solution. However, while SFT is good at many tasks,
it does have some limitations. To see good quality improvements,
you typically need thousands of high quality labeled examples
for the model to learn from. Which can be more difficult
and expensive to collect. Another common problem that you may run
into is the phenomenon of overfitting, where the model learns the patterns
and the training data too well, it does not show the same performance on examples
it has not seen before. These limitations point towards
the need for a training approach that can reduce our reliance on needing
extensive labeling and mitigate overfitting, while still guiding the model
towards a desired behavior. One such alternative is reinforcement
learning, where the model learns by interacting
with its environment and optimizing for the reward signal, rather
than mimicking fixed labeled examples. To understand this idea better,
let's take a closer look at this example. In this example, your puppy has
many different actions that it can take. It can choose to sit in one place. It can choose to roll over, or it can choose to fetch the stick
when you actually throw it. The puppy learns that out of all the actions that it can take,
it does get a treat, which is its reward, when it actually fetches a stick
and returns that back to you. Compared to sitting in the same place. So in this example,
the puppy is the agent. Fetching
a stick is an action that the puppy takes, and the treat is a reward
received from the environment. The observation
is that the puppy receives a treat for bringing the stick
rather than other actions. Now how does this idea
actually translate to LLM training? Well, we can start with an example
such as a prompt which comes from the environment,
and we can feed it to an LLM, which is the agent.
The LLM then takes an action by generating a sequence of tokens
as its response. We can evaluate this response
and provided a score that will serve as a reward
for the action it took. This score can be based on quality,
human preference, or an automated metric like accuracy. The model can then use this reward
as feedback to adjust its weights, so that it can learn to maximize
its reward for different input prompts. This process can be repeated
on new examples or even the same ones, and the model will continue to refine
its weights to get higher rewards. So how do we actually go about
implementing such a training process? One approach that's proven extremely effective is reinforcement
learning with human feedback or RLHF. And this is the very process
that powers ChatGPT. The RLHF workflow has four steps. In step one,
we send a prompt to the LLM and sample multiple candidate responses
using temperature based sampling. In step two, we ask annotators to rank these responses
for the prompt from best to worst. This produces a preference ranking
data set. In step three, we train a separate reward model to learn
to predict these human preferences. It takes a prompt and response pair
as input and output the score to indicate
how good this responses. Finally, in step four,
we find in the original LLM with the reinforcement learning algorithm like PPO. For each prompt,
the LLM generates a response, the reward model scores
it, and the LLM's weights are updated to increase the likelihood
of producing high-scoring outputs. As you repeat this
step over hundreds of prompts, it learns to generate responses
that will produce high scores and align with human preferences. Another reinforcement learning algorithm that has gained popularity is Direct
Preference Optimization, or DPO. Like RLHF,
it uses human preference data. But instead of first training a separate reward
model, it directly fine-tunes the LLM on human preference pairs. Let's see how it does this. We start with the same process as RLHF, where we pass a prompt
the LLM and sample candidate responses. However, in this case, we will just sample
two different responses A and B. Next, we can get human feedback
by asking annotators to tell us which of the two responses
they prefer more. This is often done using thumbs up
or thumbs down and various apps, but there are alternate ways
of collecting it as well. These preferences are then used to create a preference data
set that consists of a prompt, the chosen response, and the rejected
response for the same prompt. Finally, we can use the DPO algorithm
to update the model's weights to generate responses
with higher human preference. The idea behind the training
algorithm itself is very simple. For each prompt, you compare the model's
probability distribution for the preferred response to the rejected response and see
which one it is more likely to generate. Then we adjust the weights
so that the model's probability for the like response goes up, and the probability
for the dislike response goes down. Both our RLHF and DPO rely on human
preference labels instead of ground truth answers, but they differ in label,
format, cost, and risk. RLHF requires
full rankings of the many candidate responses to obtain a reward model,
and also requires multiple copies of the model's
weights to be loaded into memory, resulting in very high compute and memory
overhead. DPO, in contrast, uses simple preference
pairs, reducing computational load by not requiring a reward model,
but still demands large numbers of annotated comparisons to learn
fine-grained nuances and preferences. However, neither method teaches
the model entirely new tasks. They simply guide the model towards
human preferred behaviors. To get around the limitations
of large preference data sets, the DeepSeek team proposed a new alternative
method called Group
Relative Policy Optimization, or GRPO. The algorithm behind DeepSeek R1. The GRPO algorithm sidesteps
the need for any human preference labels by leaning on programable reward functions
that we can define. Its core training loop has three steps. Like RLHF, we first send a prompt to the LLM
and sample multiple candidate responses. Next, we can write one
or more programable reward functions that take each prompt and response pair's
input and emits a score. For example, you can get the format
of the output or its correctness. If these functions are written well, the generated responses
will receive a range of scores. GRPO algorithm then treats each
candidate's reward as a training signal. It pushes up the probability of producing
responses with above-average scores within this group, and pushes down
those responses with below-average scores. By repeating this loop, GRPO fine-tunes the model directly
on the reward function to care about without ever collecting preference data,
and thus unlocks reinforcement fine-tuning even when human labels
are scarce or costly. There are many more details
on reward functions that we'll cover, along with the GRPO training algorithm,
throughout the rest of this course.