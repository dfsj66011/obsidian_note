
paper：[Welcome to the Era of Experience](https://t.co/Y6m4jLRjnh)
authors: David Silver, Richard S. Sutton*

我们正处于人工智能新时代的门槛上，这一时代有望实现前所未有的能力水平。新一代智能体将通过主要从经验中学习来获得超人类的能力。本说明探讨了将定义这个即将到来的时代的关键特征。

### 人类数据时代

近年来，AI 通过在海量人类生成的数据上进行训练，并根据专家的人类示例和偏好进行微调，取得了显著进展。LLMs 就是这种方法的典型代表，它们已经达到了一种极为广泛的通用性水平。如今，单个大型语言模型能够执行从写诗、解答物理问题到诊断医疗问题和总结法律文件等多种任务。

然而，尽管模仿人类足以在很多能力上达到相当的水平，但单靠这种方法并不能且可能永远无法在许多重要主题和任务中实现超人类智能。在诸如数学、编程和科学等关键领域，从人类数据中提取的知识正迅速接近极限。大多数高质量的数据源——那些能够真正提升强大智能体性能的数据源——要么已经被利用殆尽，要么很快就会耗尽。仅靠从人类数据进行监督学习所推动的进步速度明显放缓，这表明需要一种新的方法。此外，诸如新定理、新技术或科学突破等有价值的新见解，超出了当前人类理解的边界，无法通过现有的人类数据来获取。

### 体验时代

要取得更大的进展，就需要新的数据来源。这些数据必须以一种随着智能体变得更强而不断改进的方式生成；任何用于合成数据的静态程序都会很快被超越。这可以通过让智能体不断从自身经验中学习来实现，即由智能体与环境交互产生的数据。人工智能正处于一个新的时期的边缘，在这个时期，经验将成为改进的主要媒介，并最终使当今系统中使用的人类数据的规模相形见绌。

这种转变可能已经开始了，即使对于体现以人类为中心的人工智能的大型语言模型而言也是如此。一个例子是在数学能力方面。AlphaProof 最近成为首个在国际数学奥林匹克竞赛中获奖的程序，超越了以人类为中心的方法的表现。AlphaProof 的 RL 算法最初接触了大约十万条由人类数学家多年创建的形式化证明，随后通过与形式化证明系统的持续交互，又生成了一亿条证明。这种对交互体验的关注使 AlphaProof 能够探索超出现有形式化证明范围的数学可能性，从而发现新颖且有挑战性问题的解决方案。非正式数学也通过用自生成的数据取代专家生成的数据取得了成功；例如，DeepSeek 的近期工作“凸显了强化学习的力量和魅力：我们无需明确地教模型如何解决问题，只需为其提供正确的激励，它就能自主开发出高级的问题解决策略。”

我们认为，一旦充分挖掘体验式学习的潜力，将会出现令人难以置信的新能力。这个体验时代可能会以智能体和环境为特征，它们除了能从大量体验数据中学习外，还将在其他几个维度突破以人类为中心的人工智能系统的局限：

* 智能体将融入体验流，而非短暂的交互片段。
* 他们的行动和观察将深深植根于环境之中，而非仅仅通过人类对话进行交互。
* 他们的回报将基于他们对环境的体验，而非来自人类的先入为主的判断。
* 他们将针对经验进行规划和/或推理，而非仅仅以人类的方式进行推理。

我们认为，借助恰当选择的算法，当今的技术已经为实现这些突破提供了足够强大的基础。此外，人工智能领域对这一目标的追求将推动这些方向上的新创新，使人工智能迅速朝着真正超人类的智能体方向发展。

### Streams 

经验型智能体可以在一生中持续学习。在人类数据时代，基于语言的人工智能大多侧重于短暂的交互片段：例如，用户提出一个问题，（可能在经过几个思考步骤或工具使用动作后）智能体作出回应。通常，从一个片段到下一个片段几乎没有或根本没有信息传递，无法实现随时间的适应。此外，智能体的目标完全局限于当前片段内的结果，比如直接回答用户的问题。相比之下，人类（和其他动物）存在于一个持续多年的行动和观察的连续流中。信息贯穿整个流，并且它们的行为会根据过去的经验进行自我修正和改进。此外，目标可以以远在未来流中的行动和观察来指定。例如，人类可能会选择行动来实现长期目标，如改善健康、学习一门语言或取得科学突破。

强大的智能体应该拥有自己的经验流，就像人类一样在较长的时间尺度上不断发展。这将使智能体能够采取行动以实现未来的目标，并随着时间推移不断适应新的行为模式。例如，一个与用户可穿戴设备相连的健康管理智能体可以监测数月之久的睡眠模式、活动水平和饮食习惯。然后，它可以提供个性化的建议、鼓励，并根据长期趋势和用户的特定健康目标调整其指导建议。同样，一个个性化教育智能体可以跟踪用户学习新语言的进度，找出知识漏洞，适应他们的学习风格，并在数月甚至数年的时间里调整其教学方法。此外，一个科学智能体可以追求宏伟的目标，比如发现一种新材料或减少二氧化碳排放。这样的智能体可以在较长时间内分析现实世界的观测数据，开发并运行模拟实验，提出现实世界中的实验或干预措施建议。

在每种情况下，智能体都会采取一系列步骤，以便在指定目标方面最大化长期成功。单个步骤可能不会带来任何即时利益，或者甚至在短期内可能是有害的，但总体上仍可能有助于长期成功。这与当前的 AI 系统形成鲜明对比，后者仅对请求提供即时响应，而没有任何能力去衡量或优化其行动对环境的未来后果。

### 行动与观察

体验时代的智能体将在现实世界中自主行动。以人类数据为重点的时代的大语言模型主要关注人类特权的行动和观察，这些行动和观察会向用户输出文本，并将用户的输入文本传回到智能体中。这与自然智能有显著不同，在自然智能中，动物通过运动控制和传感器与环境进行交互。虽然动物，尤其是人类，可能会与其他动物进行交流，但这种交流是通过与其他感觉运动控制相同的接口进行的，而不是通过特权通道。

长期以来，人们已经认识到，大型语言模型（LLMs）也可能在数字世界中调用动作，例如通过调用应用程序编程接口（API）（例如，参见[43]）。最初，这些能力主要来自人类使用工具的示例，而不是来自智能体的经验。然而，编码和工具使用能力越来越多地依赖于执行反馈[17, 7, 12]，即智能体实际运行代码并观察发生的情况。最近，新一代原型智能体开始以更通用的方式与计算机进行交互，使用的是人类操作计算机所用的相同界面[3, 15, 24]。这些变化预示着从专属于人类的通信方式向更加自主的交互方式的转变，在这种交互方式中，智能体能够在世界中独立行动。这样的智能体将能够积极探索世界，适应不断变化的环境，并发现人类可能永远想不到的策略。

这些更为丰富的交互方式将提供一种自主理解和控制数字世界的手段。智能体可以采用诸如用户界面这类“对人类友好”的操作和观测方式，这自然有助于与用户进行沟通和协作。智能体还可以采取“对机器友好”的操作，比如执行代码和调用应用程序编程接口（API），从而能够为了实现自身目标而自主行动。在体验时代，智能体还将通过数字接口与现实世界进行交互。例如，一个科学智能体可以监测环境传感器、远程操作望远镜，或者在实验室中控制机械臂来自主开展实验。

### 奖励

如果体验式智能体能够从外部事件和信号中学习，而不仅仅是从人类偏好中学习，会怎样呢？

以人类为中心的大型语言模型（LLMs）通常针对基于人类先入为主的判断的奖励进行优化：专家观察智能体的行动，并决定这是一个好的行动，或者在多个替代方案中挑选出最佳的智能体行动。例如，专家可能会评判健康助手的建议、教育助手的教学或者科学家助手建议的实验。这些奖励或偏好是由人类在其后果缺失的情况下确定的，而不是衡量这些行动对环境的影响，这意味着它们并非直接基于现实世界。以这种方式依赖人类的先入为主的判断通常会导致智能体的性能遇到难以突破的上限：智能体无法发现那些被人类评分者低估的更好的策略。要发现远远超出现有人类知识的新想法，有必要使用基于现实的奖励：源于环境本身的信号。例如，健康助手可以将用户的健康目标转化为基于静息心率、睡眠时长和活动水平等信号组合的奖励，而教育助手可以利用考试成绩为语言学习提供基于现实的奖励。同样地，一个以减缓全球变暖为目标的人工智能助手可能会使用基于对二氧化碳水平的实证观察的奖励，而一个以发现更强材料为目标的人工智能助手的奖励可能会基于材料模拟器的测量结果组合，例如抗拉强度或杨氏模量。

基于实际情况的奖励可能来自作为智能体环境一部分的人类。例如，人类用户可以报告他们是否觉得蛋糕好吃、运动后有多疲劳或者头痛的疼痛程度，从而使辅助型智能体能够提供更好的食谱、完善其健身建议或者改进其推荐的药物。这类奖励衡量的是智能体在其环境中的行为所产生的后果，最终应该会比预先评判所提出的蛋糕食谱、锻炼计划或治疗方案的人类专家提供更好的帮助。

如果不是来自人类数据，奖励又来自哪里呢？一旦智能体通过丰富的动作和观测空间（见上文）与世界相连，就会有大量基于实际情况的信号为奖励提供依据。实际上，世界充满了诸如成本、错误率、饥饿程度、生产力、健康指标、气候指标、利润、销售额、考试成绩、成功、访问量、产量、股票、点赞数、收入、愉悦/痛苦程度、经济指标、准确性、功率、距离、速度、效率或能耗等量化指标。此外，还有无数其他信号源于特定事件的发生，或者源于从原始的观测和动作序列中提取的特征 。

原则上，人们可以创建出多种不同的智能体，每个智能体都以一个基于实际情况的信号作为其奖励目标来进行优化。有一种观点认为，即便只有一个这样的奖励信号，并且能对其进行极为有效的优化，或许也足以催生出具备广泛能力的智能[34]。这是因为在复杂环境中实现一个简单目标，往往需要掌握各种各样的技能。

然而，追求单一的奖励信号表面上似乎并不符合能够可靠地引导至任意用户期望行为的通用人工智能的要求。那么，自主优化基于非人类的奖励信号是否与现代人工智能系统的要求相悖呢？我们认为情况并非必然如此，我们通过勾勒出一种可能满足这些期望的方法来进行论证；当然也可能存在其他方法。

其理念是根据基于实际情况的信号，以用户引导的方式灵活调整奖励。例如，奖励函数可以由一个神经网络来定义，该神经网络将智能体与用户和环境的交互作为输入，并输出一个标量奖励。这使得奖励能够根据用户的目标，以特定方式选择或组合来自环境的信号。例如，用户可能会设定一个宽泛的目标，如 “提高我的健康水平”，此时奖励函数可能会返回用户心率、睡眠时长和运动步数相关的函数。或者用户可能设定 “帮我学习西班牙语” 的目标，那么奖励函数可以返回用户的西班牙语考试成绩。

此外，用户可以在学习过程中提供反馈，例如他们的满意度水平，这可以用于微调奖励函数。然后，奖励函数可以随时间适应，以改进其选择或组合信号的方式，并识别和纠正任何不对准的情况。这也可以理解为一个双层优化过程，将用户反馈作为顶级目标进行优化，并在低层次上优化来自环境的可靠信号。通过这种方式，少量的人类数据可能有助于大量的自主学习。

### 规划与推理

体验时代会改变智能体规划和推理的方式吗？最近，在使用能够推理或用语言“思考”的大型语言模型（LLMs）[23, 14, 10]方面取得了重大进展，这些模型在输出响应之前会遵循思维链[16]。从概念上讲，大型语言模型可以充当通用计算机30：大型语言模型可以将标记追加到其自身的上下文中，从而使其能够在输出最终结果之前执行任意算法。

在人类数据时代，这些推理方法被明确设计用于模仿人类的思维过程。例如，大型语言模型（LLMs）已被引导产生类似人类的思维链[16]，模仿人类思维的痕迹[42]，或强化与人类示例相匹配的思维步骤[18]。推理过程可能会进一步微调，以产生与人类专家确定的正确答案相匹配的思维痕迹[44]。

然而，人类语言极不可能是通用计算机的最佳实例。更高效的思维机制肯定存在，它们使用非人类语言，例如可能利用符号、分布式、连续或可微计算。原则上，一个自学习系统可以通过学习如何从经验中思考来发现或改进这些方法。例如，AlphaProof 学会了以一种与人类数学家截然不同的方式正式证明复杂的定理[20]。

此外，通用计算机的原理仅涉及代理的内部计算；它并未将其与外部世界的现实联系起来。一个被训练用于模仿人类思维甚至匹配人类专家答案的代理可能会继承深深嵌入这些数据中的错误思维方式，例如有缺陷的假设或固有偏见。例如，如果一个代理接受了使用5000年前的人类思维和专家答案进行推理的训练，它可能会用泛灵论的方式来推理物理问题；1000年前，它可能会用有神论的方式推理；300年前，它可能会用牛顿力学的方式推理；50年前则可能用量子力学的方式推理。超越每种思维方式都需要与现实世界互动：提出假设、进行实验、观察结果并相应地更新原理。同样，代理必须基于现实世界的数据才能推翻错误的思维方式。这种基础提供了一个反馈回路，使代理能够将其继承的假设与现实进行对比，并发现不受当前主流人类思维模式限制的新原理。没有这种基础，无论多么复杂的代理都会成为现有人类知识的回音室。要超越这一点，代理必须积极与世界互动，收集观察数据，并利用这些数据迭代地完善其理解，这在许多方面反映了推动人类科学进步的过程。

将思维直接与外部世界建立联系的一种可能方法是构建一个世界模型[37]，该模型能够预测智能体的行为对世界的后果，包括预测奖励。例如，健康助手可能会考虑推荐当地的健身房或健康播客。智能体的世界模型可能会预测用户的心率或睡眠模式在此行为之后可能随之发生的变化，同时也会预测未来与用户的对话。这使得智能体能够直接根据其自身行为及其对世界的因果效应来进行规划[36, 29]。随着智能体在其经验流中不断地与世界进行交互，其动态模型会不断更新以纠正其预测中的任何错误。给定一个世界模型，智能体可以应用可扩展的规划方法来提高预测的智能体性能。

making a recommendation for a local gym or a health podcast. The agent’s world model might predict how a user’s heart rate or sleep patterns might subsequently change following this action, as well as predicting future dialogue with the user. This allows the agent to plan [36, 29] directly in terms of its own actions and their causal effect upon the world. As the agent continues to interact with the world throughout its stream of experience, its dynamics model is continually updated to correct any errors in its predictions. Given a world model, an agent may apply scalable planning methods that improve the predicted performance of the agent. 

Planning and reasoning methods are not mutually exclusive: an agent may apply internal LLM computations to select each action during planning, or to simulate and evaluate the consequences of those actions. WhyNow? Learning from experience is not new. Reinforcement learning systems have previously mastered a large number of complex tasks that were represented in a simulator with a clear reward signal (c.f., approximately, the “era of simulation” in Figure 1). For example, RL methods equalled or exceeded human performance 5Figure 1: A sketch chronology of dominant AI paradigms. The y-axis suggests the proportion of the field’s total effort and computation that is focused on RL. through self-play in board games such as backgammon [39], Go [31], chess [32], poker [22, 6] and Stratego [26]; video games such as Atari [21], StarCraft II [40], Dota 2 [4] and Gran Turismo [41]; dextrous manipulation tasks such as Rubik’s cube [1]; and resource management tasks such as data center cooling [13]. Furthermore, powerful RL agents such as AlphaZero [33] exhibited impressive and potentially unlimited scalability with the size of the neural network, the quantity of interactive experience, and the duration of thinking time. However, agents based on this paradigm did not leap the gap between simulation (closed problems with singular, precisely defined rewards) to reality (open-ended problems with a plurality of seemingly ill-defined rewards). The era of human data offered an appealing solution. Massive corpuses of human data contain examples of natural language for a huge diversity of tasks. Agents trained on this data achieved a wide range of competencies compared to the more narrow successes of the era of simulation. Consequently, the methodology of experiential RL was largely discarded in favour of more general-purpose agents, resulting in a widespread transition to human-centric AI. However, something was lost in this transition: an agent’s ability to self-discover its own knowledge. For example, AlphaZero discovered fundamentally new strategies for chess and Go, changing the way that humans play these games [28, 45]. The era of experience will reconcile this ability with the level of taskgenerality achieved in the era of human data. This will become possible, as outlined above, when agents are able to autonomously act and observe in streams of real-world experience [11], and where the rewards may be flexibly connected to any of an abundance of grounded, real-world signals. The advent of autonomous agents that interact with complex, real-world action spaces [3, 15, 24], alongside powerful RL methods that can solve open-ended problems in rich reasoning spaces [20, 10] suggests that the transition to the era of experience is imminent. 6Reinforcement Learning Methods Reinforcement learning (RL) has a rich history that is deeply rooted in autonomous learning, where agents learn for themselves through direct interaction with their environment. Early RL research yielded a suite of powerful concepts and algorithms. For example, temporal difference learning [35] enabled agents to estimate future rewards, leading to breakthroughs such as superhuman performance in backgammon [39]. Exploration techniques, driven by optimism or curiosity, were developed to help agents discover creative new behaviors and avoid getting stuck in suboptimal routines [2]. Methods like the Dyna algorithm enabled agents to build and learn from models of their world, allowing them to plan and reason about future actions [36, 29]. Concepts like options and inter/intra-option learning facilitated temporal abstraction, enabling agents to reason over longer timescales and break down complex tasks into manageable sub-goals [38]. The rise of human-centric LLMs, however, shifted the focus away from autonomous learning and towards leveraging human knowledge. Techniques like RLHF (Reinforcement Learning from Human Feedback) [9, 25] and methods for aligning language models with human reasoning [44] proved incredibly effective, driving rapid progress in AI capabilities. These approaches, while powerful, often bypassed core RL concepts: RLHF side-stepped the need for value functions by invoking human experts in place of machine-estimated values, strong priors from human data reduced the reliance on exploration, and reasoning in human-centric terms lessened the need for world models and temporal abstraction. However, it could be argued that the shift in paradigm has thrown out the baby with the bathwater. While human-centric RL has enabled an unprecedented breadth of behaviours, it has also imposed a new ceiling on the agent’s performance: agents cannot go beyond existing human knowledge. Furthermore, the era of human data has focused predominantly on RL methods that are designed for short episodes of ungrounded, human interaction, and are not suitable for long streams of grounded, autonomous interaction. The era of experience presents an opportunity to revisit and improve classic RL concepts. This era will bring new ways to think about reward functions that are flexibly grounded in observational data. It will revisit value functions and methods to estimate them from long streams with as yet incomplete sequences. It will bring principled yet practical methods for real-world exploration that discover new behaviours that are radically different from human priors. Novel approaches to world models will be developed that capture the complexities of grounded interactions. New methods for temporal abstraction will allow agents to reason, in terms of experience, over ever-longer time horizons. By building upon the foundations of RL and adapting its core principles to the challenges of this new era, we can unlock the full potential of autonomous learning and pave the way to truly superhuman intelligence. Consequences The advent of the era of experience, where AI agents learn from their interactions with the world, promises a future profoundly different from anything we have seen before. This new paradigm, while offering immense potential, also presents important risks and challenges that demand careful consideration, including but not limited to the following points. On the positive side, experiential learning will unlock unprecedented capabilities. In everyday life, personalized assistants will leverage continuous streams of experience to adapt to individuals’ health, educational, or professional needs towards long-term goals over the course of months or years. Perhaps most transformative will be the acceleration of scientific discovery. AI agents will autonomously design and conduct experiments in fields like materials science, medicine, or hardware design. By continuously learning from the results of their own experiments, these agents could rapidly explore new frontiers of knowledge, leading to the development of novel materials, drugs, and technologies at an unprecedented pace. However, this new era also presents significant and novel challenges. While the automation of human 7capabilities promises to boost productivity, these improvements could also lead to job displacement. Agents may even be able to exhibit capabilities previously considered the exclusive realm of humanity, such as longterm problem-solving, innovation, and a deep understanding of real world consequences. Furthermore, whilst general concerns exist around the potential misuse of any AI, heightened risks may arise from agents that can autonomously interact with the world over extended periods of time to achieve long-term goals. By default, this provides fewer opportunities for humans to intervene and mediate the agent’s actions, and therefore requires a high bar of trust and responsibility. Moving away from human data and human modes of thinking may also make future AI systems harder to interpret. However, whilst acknowledging that experiential learning will increase certain safety risks, and that further research is surely required to ensure a safe transition into the era of experience, we should also recognise that it may also provide some important safety benefits. Firstly, an experiential agent is aware of the environment it is situated within, and its behaviour can adapt over time to changes in that environment. Any pre-programmed system, including a fixed AI system, can be unaware of its environmental context, and become maladapted to the changing world into which it is deployed. For example, a critical piece of hardware may malfunction, a pandemic might cause rapid societal change, or a new scientific discovery may trigger a cascade of rapid technological developments. By contrast, an experiential agent could observe and learn to circumvent malfunctioning hardware, adjust to rapid societal change, or embrace and build upon new science and technology. Perhaps even more importantly, the agent could recognise when its behaviour is triggering human concern, dissatisfaction, or distress, and adaptively modify its behaviour to avoid these negative consequences. Secondly, the agent’s reward function may itself be adapted through experience, for example using the bilevel optimisation described earlier (see Rewards). Importantly, this means that misaligned reward functions can often be incrementally corrected over time by trial and error. For example, rather than blindly optimising a signal, such as the maximisation of paperclips [5], the reward function could be modified, based upon indications of human concern, before paperclip production consumes all of the Earth’s resources. This is analogous to the way that humans set goals for each other, and then adapt those goals if they observe people gaming the system, neglecting long-term well-being, or causing undesired negative consequences; although also like human goal-setting, there is no guarantee of perfect alignment. Finally, advancements relying on physical experience are inherently constrained by the time it takes to execute actions in the real world and observe their consequences. For example, the development of a new drug, even with AI-assisted design, still requires real-world trials that cannot be completed overnight. This may provide a natural brake on the pace of potential AI self-improvement. Conclusion The era of experience marks a pivotal moment in the evolution of AI. Building on today’s strong foundations, but moving beyond the limitations of human-derived data, agents will increasingly learn from their own interactions with the world. Agents will autonomously interact with environments through rich observations and actions. They will continue to adapt over the course of lifelong streams of experience. Their goals will be directable towards any combination of grounded signals. Furthermore, agents will utilise powerful non-human reasoning, and construct plans that are grounded in the consequences of the agent’s actions upon its environment. Ultimately, experiential data will eclipse the scale and quality of human generated data. This paradigm shift, accompanied by algorithmic advancements in RL, will unlock in many domains new capabilities that surpass those possessed by any human. 8Acknowledgements The authors would like to acknowledge helpful comments and discussion from Thomas Degris, Rohin Shah, Tom Schaul and Hado van Hasselt.








我觉得其实真正推动 AI 发展的往往是那些有前瞻性的、有思想的文章，他们可能不一定解决点前的具体问题，却能让我们看清未来该哪里走，比如 [The Bitter Lesson](http://www.incompleteideas.net/IncIdeas/BitterLesson.html) 那篇文章，其实就是 OpenAI 发展 ChatGPT 整个系列的一个指导思想，

The Bitter Lesson（V1.0），提出的一个主要观点是，我们应该少施加一些人类的先验知识到算法的设计、训练等过程中，让算法模型自己去探索，也就是 less structure，然后再把这类的方法 scale up；

这篇 V2.0 版本里面，David Silver 和 Rich Sutton 说的其实还是同一个事情，只不过把这个观点放到了当前 agent 的框架下，如果我们 high level 来看这篇 paper 的话，它主要就提的一个观点，如果我们想实现 superhuman capacity，即 AGI，目前主要有两类方法：

* The Era of Human Data
* The Era of Experience


 