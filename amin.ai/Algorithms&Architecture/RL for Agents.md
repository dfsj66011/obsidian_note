

# Primers • Reinforcement Learning for Agents

- [Background: Teaching Agents Tool-Calling with RL](https://aman.ai/primers/ai/RL-for-agents/#background-teaching-agents-tool-calling-with-rl)
    - [Environment, MDP Formulation, and Action Space](https://aman.ai/primers/ai/RL-for-agents/#environment-mdp-formulation-and-action-space)
        - [The MDP for “When / Which / How”](https://aman.ai/primers/ai/RL-for-agents/#the-mdp-for-when--which--how)
        - [State (st)](https://aman.ai/primers/ai/RL-for-agents/#state-s_t)
        - [Structured, Factored Action Space](https://aman.ai/primers/ai/RL-for-agents/#structured-factored-action-space)
        - [Action Type 1: `ANSWER(final_text)`](https://aman.ai/primers/ai/RL-for-agents/#action-type-1-answerfinal_text)
        - [Action Type 2: `CALL(tool_name, Args_json)`](https://aman.ai/primers/ai/RL-for-agents/#action-type-2-calltool_name-args_json)
        - [Structured Action Encoding](https://aman.ai/primers/ai/RL-for-agents/#structured-action-encoding)
        - [Episode Dynamics](https://aman.ai/primers/ai/RL-for-agents/#episode-dynamics)
        - [Handling Invalid/Malformed Actions](https://aman.ai/primers/ai/RL-for-agents/#handling-invalidmalformed-actions)
        - [Integrating “When / Which / How” of Tool-Calling Into the Action Space](https://aman.ai/primers/ai/RL-for-agents/#integrating-when--which--how-of-tool-calling-into-the-action-space)
    - [Annotation Sources for Reward Components (“When”, “Which”, and “How”)](https://aman.ai/primers/ai/RL-for-agents/#annotation-sources-for-reward-components-when-which-and-how)
        - [Reward Component: Call (Deciding “When” a Tool Should be Invoked)](https://aman.ai/primers/ai/RL-for-agents/#reward-component-call-deciding-when-a-tool-should-be-invoked)
            - [Rule-based Supervision](https://aman.ai/primers/ai/RL-for-agents/#rule-based-supervision)
            - [Discriminative Reward Model](https://aman.ai/primers/ai/RL-for-agents/#discriminative-reward-model)
            - [Generative Reward Model (LLM-as-a-Judge)](https://aman.ai/primers/ai/RL-for-agents/#generative-reward-model-llm-as-a-judge)
        - [Reward Component: Tool Selection (Choosing “Which” Tool)](https://aman.ai/primers/ai/RL-for-agents/#reward-component-tool-selection-choosing-which-tool)
            - [Rule-based Supervision](https://aman.ai/primers/ai/RL-for-agents/#rule-based-supervision-1)
            - [Discriminative Reward Model](https://aman.ai/primers/ai/RL-for-agents/#discriminative-reward-model-1)
            - [Generative Reward Model](https://aman.ai/primers/ai/RL-for-agents/#generative-reward-model)
        - [Reward Component: **Tool-Syntax Correctness**](https://aman.ai/primers/ai/RL-for-agents/#reward-component-tool-syntax-correctness)
            - [Rule-based](https://aman.ai/primers/ai/RL-for-agents/#rule-based)
            - [Discriminative Reward Model](https://aman.ai/primers/ai/RL-for-agents/#discriminative-reward-model-2)
            - [Generative Reward Model](https://aman.ai/primers/ai/RL-for-agents/#generative-reward-model-1)
        - [Reward Component: **Tool-Execution Correctness**](https://aman.ai/primers/ai/RL-for-agents/#reward-component-tool-execution-correctness)
            - [Rule-based](https://aman.ai/primers/ai/RL-for-agents/#rule-based-1)
            - [Discriminative Reward Model](https://aman.ai/primers/ai/RL-for-agents/#discriminative-reward-model-3)
            - [Generative Reward Model](https://aman.ai/primers/ai/RL-for-agents/#generative-reward-model-2)
        - [Reward Component: Argument Quality (Deciding “How” to Call a Tool)](https://aman.ai/primers/ai/RL-for-agents/#reward-component-argument-quality-deciding-how-to-call-a-tool)
            - [Rule-based](https://aman.ai/primers/ai/RL-for-agents/#rule-based-2)
            - [Discriminative Reward Model](https://aman.ai/primers/ai/RL-for-agents/#discriminative-reward-model-4)
            - [Generative Reward Model](https://aman.ai/primers/ai/RL-for-agents/#generative-reward-model-3)
        - [Reward Component: **Final Task Success**](https://aman.ai/primers/ai/RL-for-agents/#reward-component-final-task-success)
            - [Rule-based](https://aman.ai/primers/ai/RL-for-agents/#rule-based-3)
            - [Discriminative Reward Model](https://aman.ai/primers/ai/RL-for-agents/#discriminative-reward-model-5)
            - [Generative Reward Model](https://aman.ai/primers/ai/RL-for-agents/#generative-reward-model-4)
        - [Merged Preference-Based Rewards (For “Call”, “Which”, and “How”)](https://aman.ai/primers/ai/RL-for-agents/#merged-preference-based-rewards-for-call-which-and-how)
        - [Unified Reward Formulation](https://aman.ai/primers/ai/RL-for-agents/#unified-reward-formulation)
        - [Asymmetric Rewards in Tool-Calling RL](https://aman.ai/primers/ai/RL-for-agents/#asymmetric-rewards-in-tool-calling-rl)
            - [Why Asymmetry is Required](https://aman.ai/primers/ai/RL-for-agents/#why-asymmetry-is-required)
            - [Reward Table: Positive and Negative Rewards by Category](https://aman.ai/primers/ai/RL-for-agents/#reward-table-positive-and-negative-rewards-by-category)
                - [Reward Values for “When / Which / How” and Outcome-Level Components](https://aman.ai/primers/ai/RL-for-agents/#reward-values-for-when--which--how-and-outcome-level-components)
            - [Worked Example with Asymmetric Rewards](https://aman.ai/primers/ai/RL-for-agents/#worked-example-with-asymmetric-rewards)
                - [Trajectory A: Imperfect but Reasonable Exploration](https://aman.ai/primers/ai/RL-for-agents/#trajectory-a-imperfect-but-reasonable-exploration)
                - [Trajectory B: Full Correct Behavior](https://aman.ai/primers/ai/RL-for-agents/#trajectory-b-full-correct-behavior)
            - [How Asymmetry Stabilizes PPO/GRPO](https://aman.ai/primers/ai/RL-for-agents/#how-asymmetry-stabilizes-ppogrpo)
            - [Takeaways](https://aman.ai/primers/ai/RL-for-agents/#takeaways)
    - [RL Optimization Pipeline: Shared Flow + PPO vs. GRPO](https://aman.ai/primers/ai/RL-for-agents/#rl-optimization-pipeline-shared-flow--ppo-vs-grpo)
        - [Shared RL Training Flow](https://aman.ai/primers/ai/RL-for-agents/#shared-rl-training-flow)
        - [PPO: Losses and Update Rules](https://aman.ai/primers/ai/RL-for-agents/#ppo-losses-and-update-rules)
            - [Surrogate Objective](https://aman.ai/primers/ai/RL-for-agents/#surrogate-objective)
            - [Value Loss](https://aman.ai/primers/ai/RL-for-agents/#value-loss)
            - [KL/Entropy Penalty](https://aman.ai/primers/ai/RL-for-agents/#klentropy-penalty)
            - [Full PPO Loss](https://aman.ai/primers/ai/RL-for-agents/#full-ppo-loss)
            - [Implementation Notes](https://aman.ai/primers/ai/RL-for-agents/#implementation-notes)
        - [GRPO: Losses and Update Rules](https://aman.ai/primers/ai/RL-for-agents/#grpo-losses-and-update-rules)
            - [Group Sampling & Relative Advantage](https://aman.ai/primers/ai/RL-for-agents/#group-sampling--relative-advantage)
            - [GRPO Surrogate](https://aman.ai/primers/ai/RL-for-agents/#grpo-surrogate)
            - [Value Loss](https://aman.ai/primers/ai/RL-for-agents/#value-loss-1)
            - [KL/Entropy Penalty](https://aman.ai/primers/ai/RL-for-agents/#klentropy-penalty-1)
            - [Full GRPO Loss](https://aman.ai/primers/ai/RL-for-agents/#full-grpo-loss)
            - [Implementation Notes](https://aman.ai/primers/ai/RL-for-agents/#implementation-notes-1)
        - [Integrating the Unified Reward](https://aman.ai/primers/ai/RL-for-agents/#integrating-the-unified-reward)
    - [Curriculum Design, Evaluation Strategy, and Diagnostics for Tool-Calling RL](https://aman.ai/primers/ai/RL-for-agents/#curriculum-design-evaluation-strategy-and-diagnostics-for-tool-calling-rl)
        - [Curriculum Design Overview](https://aman.ai/primers/ai/RL-for-agents/#curriculum-design-overview)
        - [Stage 0: Pure Supervised Bootstrapping (SFT)](https://aman.ai/primers/ai/RL-for-agents/#stage-0-pure-supervised-bootstrapping-sft)
        - [Stage 1: Binary Decision Curriculum (Learning **When**)](https://aman.ai/primers/ai/RL-for-agents/#stage-1-binary-decision-curriculum-learning-when)
        - [Stage 2: Tool-Selection Curriculum (Learning **Which**)](https://aman.ai/primers/ai/RL-for-agents/#stage-2-tool-selection-curriculum-learning-which)
        - [Stage 3: Argument-Construction Curriculum (Learning **How**)](https://aman.ai/primers/ai/RL-for-agents/#stage-3-argument-construction-curriculum-learning-how)
        - [Stage 4: Multi-Step Tool Use (Pipelines)](https://aman.ai/primers/ai/RL-for-agents/#stage-4-multi-step-tool-use-pipelines)
        - [Stage 5: Open-Domain Free-Form Tasks](https://aman.ai/primers/ai/RL-for-agents/#stage-5-open-domain-free-form-tasks)
        - [Diagnostics and Monitoring](https://aman.ai/primers/ai/RL-for-agents/#diagnostics-and-monitoring)
            - [Process-Level Metrics](https://aman.ai/primers/ai/RL-for-agents/#process-level-metrics)
            - [Outcome-Level Metrics](https://aman.ai/primers/ai/RL-for-agents/#outcome-level-metrics)
        - [Detecting Skill Collapse](https://aman.ai/primers/ai/RL-for-agents/#detecting-skill-collapse)
        - [Curriculum Scheduling (Putting It All Together)](https://aman.ai/primers/ai/RL-for-agents/#curriculum-scheduling-putting-it-all-together)
        - [Final Note](https://aman.ai/primers/ai/RL-for-agents/#final-note)
    - [Reinforcement Learning and the Emergence of Intelligent Agents](https://aman.ai/primers/ai/RL-for-agents/#reinforcement-learning-and-the-emergence-of-intelligent-agents)
    - [The Role of Reinforcement Learning in Self-Improving Agents](https://aman.ai/primers/ai/RL-for-agents/#the-role-of-reinforcement-learning-in-self-improving-agents)
    - [Environments for Reinforcement Learning in Modern Agents](https://aman.ai/primers/ai/RL-for-agents/#environments-for-reinforcement-learning-in-modern-agents)
- [The Three Major Types of Reinforcement Learning Environments](https://aman.ai/primers/ai/RL-for-agents/#the-three-major-types-of-reinforcement-learning-environments)
    - [Single-Turn Environments (SingleTurnEnv)](https://aman.ai/primers/ai/RL-for-agents/#single-turn-environments-singleturnenv)
    - [Tool-Use Environments (ToolEnv)](https://aman.ai/primers/ai/RL-for-agents/#tool-use-environments-toolenv)
    - [Multi-Turn, Sequential Environments (MultiTurnEnv)](https://aman.ai/primers/ai/RL-for-agents/#multi-turn-sequential-environments-multiturnenv)
    - [Implications](https://aman.ai/primers/ai/RL-for-agents/#implications)
- [Reinforcement Learning for Web and Computer-Use Agents](https://aman.ai/primers/ai/RL-for-agents/#reinforcement-learning-for-web-and-computer-use-agents)
    - [Background: Policy-Based and Value-Based Methods](https://aman.ai/primers/ai/RL-for-agents/#background-policy-based-and-value-based-methods)
    - [Background: Process-Wise Rewards vs. Outcome-Based Rewards](https://aman.ai/primers/ai/RL-for-agents/#background-process-wise-rewards-vs-outcome-based-rewards)
    - [Reinforcement Learning from Human Feedback (RLHF) and Direct Preference Optimization (DPO)](https://aman.ai/primers/ai/RL-for-agents/#reinforcement-learning-from-human-feedback-rlhf-and-direct-preference-optimization-dpo)
    - [Why These Algorithms Matter for Web & Computer-Use Agents](https://aman.ai/primers/ai/RL-for-agents/#why-these-algorithms-matter-for-web--computer-use-agents)
    - [Key Equations](https://aman.ai/primers/ai/RL-for-agents/#key-equations)
        - [Advantage Estimation & Value Networks](https://aman.ai/primers/ai/RL-for-agents/#advantage-estimation--value-networks)
        - [KL-penalty / Trust Region](https://aman.ai/primers/ai/RL-for-agents/#kl-penalty--trust-region)
        - [Preference Optimization (DPO)](https://aman.ai/primers/ai/RL-for-agents/#preference-optimization-dpo)
        - [Sample Efficiency & Off-policy Corrections](https://aman.ai/primers/ai/RL-for-agents/#sample-efficiency--off-policy-corrections)
    - [Agentic Reinforcement Learning Via Policy Optimization](https://aman.ai/primers/ai/RL-for-agents/#agentic-reinforcement-learning-via-policy-optimization)
        - [Milestone-Based Reward System](https://aman.ai/primers/ai/RL-for-agents/#milestone-based-reward-system)
            - [Example Milestones by Task Category](https://aman.ai/primers/ai/RL-for-agents/#example-milestones-by-task-category)
        - [Example Reward Function](https://aman.ai/primers/ai/RL-for-agents/#example-reward-function)
        - [Example Instantiation](https://aman.ai/primers/ai/RL-for-agents/#example-instantiation)
    - [Agent Training Pipeline](https://aman.ai/primers/ai/RL-for-agents/#agent-training-pipeline)
- [Environment Interaction Patterns for Agent Design](https://aman.ai/primers/ai/RL-for-agents/#environment-interaction-patterns-for-agent-design)
    - [Environment Design in Reinforcement Learning for Agents](https://aman.ai/primers/ai/RL-for-agents/#environment-design-in-reinforcement-learning-for-agents)
    - [Single-Turn Environments (SingleTurnEnv)](https://aman.ai/primers/ai/RL-for-agents/#single-turn-environments-singleturnenv-1)
    - [Tool-Use Environments (ToolEnv)](https://aman.ai/primers/ai/RL-for-agents/#tool-use-environments-toolenv-1)
    - [Multi-Turn Sequential Environments (MultiTurnEnv)](https://aman.ai/primers/ai/RL-for-agents/#multi-turn-sequential-environments-multiturnenv-1)
    - [Designing Rewards for Complex Agent Environments](https://aman.ai/primers/ai/RL-for-agents/#designing-rewards-for-complex-agent-environments)
    - [Implications for Agent Design and Evaluation](https://aman.ai/primers/ai/RL-for-agents/#implications-for-agent-design-and-evaluation)
    - [Comparative Analysis](https://aman.ai/primers/ai/RL-for-agents/#comparative-analysis)
- [Reward Modeling](https://aman.ai/primers/ai/RL-for-agents/#reward-modeling)
    - [The Role of Reward Modeling](https://aman.ai/primers/ai/RL-for-agents/#the-role-of-reward-modeling)
    - [Process-Wise and Outcome-Based Reward Integration](https://aman.ai/primers/ai/RL-for-agents/#process-wise-and-outcome-based-reward-integration)
    - [Tool-Augmented Reward Modeling (TARM)](https://aman.ai/primers/ai/RL-for-agents/#tool-augmented-reward-modeling-tarm)
        - [Motivation and Background](https://aman.ai/primers/ai/RL-for-agents/#motivation-and-background)
        - [Structure and Workflow of Tool-Augmented Reward Models](https://aman.ai/primers/ai/RL-for-agents/#structure-and-workflow-of-tool-augmented-reward-models)
        - [Role of Supervised Fine-Tuning and Reinforcement Learning](https://aman.ai/primers/ai/RL-for-agents/#role-of-supervised-fine-tuning-and-reinforcement-learning)
        - [The Tool-Augmented Reward Dataset (TARA)](https://aman.ai/primers/ai/RL-for-agents/#the-tool-augmented-reward-dataset-tara)
        - [Empirical Results and Observations](https://aman.ai/primers/ai/RL-for-agents/#empirical-results-and-observations)
        - [Connection to Reinforcement Learning for Agents](https://aman.ai/primers/ai/RL-for-agents/#connection-to-reinforcement-learning-for-agents)
    - [Feedback Alignment and Human Preference Modeling](https://aman.ai/primers/ai/RL-for-agents/#feedback-alignment-and-human-preference-modeling)
    - [Multi-Objective Reward Modeling](https://aman.ai/primers/ai/RL-for-agents/#multi-objective-reward-modeling)
    - [Evaluation Frameworks for RL-Based Agents](https://aman.ai/primers/ai/RL-for-agents/#evaluation-frameworks-for-rl-based-agents)
        - [Key Evaluation Metrics Include](https://aman.ai/primers/ai/RL-for-agents/#key-evaluation-metrics-include)
    - [Takeaways](https://aman.ai/primers/ai/RL-for-agents/#takeaways-1)
- [Search-Based Reinforcement Learning, Monte Carlo Tree Search (MCTS), and Exploration Strategies in Multi-Step Agents](https://aman.ai/primers/ai/RL-for-agents/#search-based-reinforcement-learning-monte-carlo-tree-search-mcts-and-exploration-strategies-in-multi-step-agents)
    - [Motivation: Exploration vs. Exploitation in Complex Agentic Systems](https://aman.ai/primers/ai/RL-for-agents/#motivation-exploration-vs-exploitation-in-complex-agentic-systems)
    - [Monte Carlo Tree Search (MCTS) in RL-Based Agents](https://aman.ai/primers/ai/RL-for-agents/#monte-carlo-tree-search-mcts-in-rl-based-agents)
    - [Neural-Guided Search: Policy Priors and Value Models](https://aman.ai/primers/ai/RL-for-agents/#neural-guided-search-policy-priors-and-value-models)
    - [Integration of Search with Reinforcement Learning and Fine-Tuning](https://aman.ai/primers/ai/RL-for-agents/#integration-of-search-with-reinforcement-learning-and-fine-tuning)
    - [Process-Wise Reward Shaping in Search-Based RL](https://aman.ai/primers/ai/RL-for-agents/#process-wise-reward-shaping-in-search-based-rl)
    - [Integration of Search with Reinforcement Learning and Fine-Tuning](https://aman.ai/primers/ai/RL-for-agents/#integration-of-search-with-reinforcement-learning-and-fine-tuning-1)
    - [Exploration Strategies in Web and Computer-Use Environments](https://aman.ai/primers/ai/RL-for-agents/#exploration-strategies-in-web-and-computer-use-environments)
    - [Planning and Value Composition Across Multiple Environments](https://aman.ai/primers/ai/RL-for-agents/#planning-and-value-composition-across-multiple-environments)
    - [Summary and Outlook](https://aman.ai/primers/ai/RL-for-agents/#summary-and-outlook)
- [Memory, World Modeling, and Long-Horizon Credit Assignment](https://aman.ai/primers/ai/RL-for-agents/#memory-world-modeling-and-long-horizon-credit-assignment)
    - [The Need for Memory and Temporal Reasoning](https://aman.ai/primers/ai/RL-for-agents/#the-need-for-memory-and-temporal-reasoning)
    - [Explicit vs. Implicit Memory Architectures](https://aman.ai/primers/ai/RL-for-agents/#explicit-vs-implicit-memory-architectures)
    - [World Modeling: Learning Predictive Environment Representations](https://aman.ai/primers/ai/RL-for-agents/#world-modeling-learning-predictive-environment-representations)
    - [Temporal Credit Assignment and Advantage Estimation](https://aman.ai/primers/ai/RL-for-agents/#temporal-credit-assignment-and-advantage-estimation)
    - [Hierarchical Reinforcement Learning (HRL)](https://aman.ai/primers/ai/RL-for-agents/#hierarchical-reinforcement-learning-hrl)
    - [Memory-Augmented Reinforcement Learning (MARL)](https://aman.ai/primers/ai/RL-for-agents/#memory-augmented-reinforcement-learning-marl)
    - [Long-Horizon Planning Via Latent Rollouts and Model Predictive Control](https://aman.ai/primers/ai/RL-for-agents/#long-horizon-planning-via-latent-rollouts-and-model-predictive-control)
    - [Takeaways](https://aman.ai/primers/ai/RL-for-agents/#takeaways-2)
- [Evaluation, Safety, and Interpretability in Reinforcement-Learning-Based Agents](https://aman.ai/primers/ai/RL-for-agents/#evaluation-safety-and-interpretability-in-reinforcement-learning-based-agents)
    - [Why Evaluation and Safety Matter in RL-Based Agents](https://aman.ai/primers/ai/RL-for-agents/#why-evaluation-and-safety-matter-in-rl-based-agents)
    - [Core Dimensions of Agent Evaluation](https://aman.ai/primers/ai/RL-for-agents/#core-dimensions-of-agent-evaluation)
        - [Task Performance](https://aman.ai/primers/ai/RL-for-agents/#task-performance)
        - [Behavioral Efficiency](https://aman.ai/primers/ai/RL-for-agents/#behavioral-efficiency)
        - [Robustness and Generalization](https://aman.ai/primers/ai/RL-for-agents/#robustness-and-generalization)
        - [Alignment and Ethical Compliance](https://aman.ai/primers/ai/RL-for-agents/#alignment-and-ethical-compliance)
        - [Interpretability and Transparency](https://aman.ai/primers/ai/RL-for-agents/#interpretability-and-transparency)
    - [Safety Challenges in RL Agents](https://aman.ai/primers/ai/RL-for-agents/#safety-challenges-in-rl-agents)
    - [Interpretability and Traceability in Agent Behavior](https://aman.ai/primers/ai/RL-for-agents/#interpretability-and-traceability-in-agent-behavior)
    - [Safety-Aware RL Algorithms](https://aman.ai/primers/ai/RL-for-agents/#safety-aware-rl-algorithms)
    - [Human-in-the-Loop (HITL) Evaluation and Oversight](https://aman.ai/primers/ai/RL-for-agents/#human-in-the-loop-hitl-evaluation-and-oversight)
    - [Benchmarking Frameworks for Safe and Transparent Evaluation](https://aman.ai/primers/ai/RL-for-agents/#benchmarking-frameworks-for-safe-and-transparent-evaluation)
    - [Toward Aligned, Interpretable, and Reliable Agentic Systems](https://aman.ai/primers/ai/RL-for-agents/#toward-aligned-interpretable-and-reliable-agentic-systems)
- [Tool-Integrated Reasoning](https://aman.ai/primers/ai/RL-for-agents/#tool-integrated-reasoning)
    - [Foundations and Theoretical Advancements in TIR](https://aman.ai/primers/ai/RL-for-agents/#foundations-and-theoretical-advancements-in-tir)
    - [Practical Engineering for Stable Multi-Turn TIR](https://aman.ai/primers/ai/RL-for-agents/#practical-engineering-for-stable-multi-turn-tir)
    - [Scaling Tool-Integrated RL from Base Models](https://aman.ai/primers/ai/RL-for-agents/#scaling-tool-integrated-rl-from-base-models)
    - [Code-Interleaved Reinforcement for Tool Use](https://aman.ai/primers/ai/RL-for-agents/#code-interleaved-reinforcement-for-tool-use)
    - [Tool-Augmented Evaluation Agents](https://aman.ai/primers/ai/RL-for-agents/#tool-augmented-evaluation-agents)
    - [Synthesizing Trends in TIR + RL Integration](https://aman.ai/primers/ai/RL-for-agents/#synthesizing-trends-in-tir--rl-integration)
    - [Synthesis: Beyond Individual Tool Use](https://aman.ai/primers/ai/RL-for-agents/#synthesis-beyond-individual-tool-use)
    - [Unifying RL and TIR: Process vs. Outcome Rewards](https://aman.ai/primers/ai/RL-for-agents/#unifying-rl-and-tir-process-vs-outcome-rewards)
    - [Synthesis and Outlook](https://aman.ai/primers/ai/RL-for-agents/#synthesis-and-outlook)
- [Citation](https://aman.ai/primers/ai/RL-for-agents/#citation)

## 概览

强化学习（RL）为教导人工智能代理如何通过与环境互动并从行动结果中学习来制定决策提供了一个正式框架。该学习过程由马尔可夫决策过程（MDP）控制，其定义为元组(S,A,P,R,γ)，其中S表示所有可能状态的集合，A表示可用动作的集合，P(s′∣s,a)是决定环境如何变化的转移概率函数，R(s,a)是向代理提供反馈的奖励函数，γ是控制代理相对于即时奖励对未来奖励重视程度的折扣因子。代理旨在学习一个策略π(a∣s)，表示在状态s下选择动作a的概率，以最大化预期累积奖励：
$$J(\pi)=\mathbb{E}_{\pi}\left[\sum_{t=0}^{\infty} \gamma^{t} R\left(s_{t}, a_{t}\right)\right]
$$
强化学习与监督学习的不同之处在于，它不会直接提供正确答案（标签）。相反，智能体必须探索不同的行动，观察其后果，并根据获得的奖励调整其策略。这种试错过程使得强化学习特别适合在复杂的数字环境（如网络、桌面系统和软件工具）中运行的智能体。

## 背景：为什么工具调用代理需要 RL 而非 SFT

训练语言模型可靠地调用工具（API、计算器、搜索引擎等）仅靠监督学习是不够的。虽然监督微调（SFT）可以教会模型模仿示例轨迹，但它无法教会模型在动态交互环境中决定何时、调用哪个工具或如何调用工具。具体如下：

**SFT 缺乏对工具调用的决策权**：

工具调用不仅仅是生成一个正确的 JSON 片段；它还需要在上下文中决定 *是否* 适合进行工具调用。监督式微调（SFT）仅模仿示范行为——其目标是最大化：$$\operatorname{SFT}(\theta)=-\sum_{t} \log p_{\theta}\left(a_{t}^{\text {expert }} \mid s_{t}\right)$$
…不依赖于结果或未来后果。在使用工具的环境中，调用工具的成本（延迟、计费、上下文切换）必须被考虑在内——SFT 无法对此进行编码。相比之下，强化学习（RL）可以优化累积回报。$$J(\pi) = \mathbb{E}_{\tau \sim \pi} \left[ \sum_{t=0}^{T} \gamma^{t} R(s_t, a_t) \right]$$……从而学会何时应避免调用工具。

**SFT 无法教授工具选择：**

 当存在多种工具（搜索、计算器、地图 API 等）时，模型必须学会一种选择策略。监督式微调（SFT）仅能学习复制演示中所做的选择，但无法理解选择错误工具时的权衡或后果。而强化学习（RL）会对错误选择给予负面奖励，从而教会模型区分不同工具。

**SFT 无法整合工具输出反馈：**

即使 SFT（监督式微调）教会了正确的参数格式，它也无法获得关于执行成功与否、工具输出质量或返回值如何影响最终答案的反馈。而在强化学习（RL）中，奖励机制可以涵盖语法正确性、执行成功率、参数质量以及最终答案的准确性——这些都是 SFT 所无法捕捉的。

**SFT 在多步骤工作流程和停止条件方面表现不佳：**

许多工具使用任务需要多次顺序调用、条件逻辑，以及决定何时停止调用工具并给出答案。监督式微调（SFT）只能看到固定长度的演示，无法推广到动态长度或停止决策。而强化学习（RL）通过分幕回报和学习的策略来处理这个问题，即在“回答”动作与进一步“调用”动作之间做出选择。

**SFT 无法对工具的滥用、过度使用或使用不足进行处罚：**

不必要的工具调用（会增加成本/延迟）或缺少必要的工具调用（会降低准确性）需要明确的惩罚措施。监督式微调（SFT）无法编码这类成本信号，因为训练损失函数仅奖励与演示标记匹配的行为。而强化学习（RL）则直接将成本纳入奖励函数中。

**SFT 在示范分布之外泛化能力不佳：**

新工具、新的论证模式、未见过的查询或动态上下文在工具使用系统中很常见。监督式微调（SFT）容易对示范动作的固定分布产生过拟合。而通过探索和回报优化的强化学习（RL），则能帮助模型发现新行为并适应变化后的情境。

**SFT 无法优化多组件目标**：

工具的使用需要协调多个不同的子技能：决定何时调用工具、选择适合的工具、构建参数、格式化JSON、确保工具执行成功、保证最终答案的正确性，以及最小化工具的成本和延迟。

SFT 提供的单一整体损失函数无法区分这些组件。它不能有选择性地惩罚时间安排、选择、参数结构、模式字段或步骤效率方面的错误。相比之下，强化学习（RL）能够实现细粒度的奖励塑造，其中每个组件都会为整体目标贡献自己的奖励项。这使得可以分别奖励正确的工具时间安排和正确的工具选择，分别奖励参数正确性和执行成功，以及分别奖励最终答案和中间步骤。

### 什么是模仿学习？为什么在强化学习之前使用监督微调

在应用强化学习教授工具调用行为之前，现代大语言模型系统几乎总是从模仿学习开始。在大语言模型的背景下，模仿学习是通过监督微调（SFT）来实现的——训练模型复制专家编写的正确工具使用示例。

本节将阐述：(i) 模仿学习的定义，(ii) 为何监督式微调(SFT)是其特例，以及(iii) 在工具使用场景中，模仿学习为何是强化学习(RL)的必要预热阶段。

#### 什么是模仿学习？

模仿学习通过直接复制专家行为来训练策略，而非通过试错来学习。无需奖励、无需探索、无需环境优化——仅需从状态到行为的监督映射。

形式上，给定示范轨迹 τ=(s0,a0),(s1,a1),…,(sT,aT)，模仿学习会最大化专家行为的似然性：$$\mathrm{IL}(\theta)=-\sum_{t=0}^{T} \log p_{\theta}\left(a_{t}^{\text {expert }} \mid s_{t}\right)$$
这与标准的监督学习非常接近，但在机器人技术和强化学习理论中，它被称为行为克隆，是最简单的模仿学习方法之一。

#### 为什么 SFT 正是模仿学习

在训练大语言模型生成推理轨迹、工具调用 JSON 或使用标注示例生成最终答案时，SFT 直接应用上述损失函数。模型不会进行探索，不会观察工具输出，也不会因正确的长期行为获得奖励。

在工具调用情境下，SFT 教导：工具调用的外观（语法），人类调用工具的粗略模式，典型的参数结构，最终答案的格式。这是模仿，而非策略优化。


#### 为什么在强化学习之前模仿学习是必不可少的

强化学习在原始文本上应用时并不稳定。动作空间庞大，语法结构脆弱，初始随机探索会产生无效的工具调用。因此，所有有效的工具使用强化学习系统都会通过监督微调（SFT）进行预热启动，为模型提供基础能力：

- 基本工具语法和模式素养：如果没有监督微调（SFT），模型在强化学习过程中会产生格式错误的JSON，导致持续的错误和嘈杂的梯度。
- 一个最小化的“何时/哪个/如何”先验知识：监督微调（SFT）样本至少为模型提供了关于工具使用时机、工具选择及参数形成的启发式模式。
- 降低探索负担：从零开始进行强化学习需要大量探索才能采样到正确的工具调用。监督微调（SFT）极大地缩小了搜索空间。
- 早期强化学习训练的稳定性和安全性：随机初始化的强化学习会导致：失控的工具调用循环，参数格式错误，没有成功的回合，退化的策略。
- SFT 通过将初始模型锚定在合理行为上来防止这种崩溃。

#### 为什么仅靠模仿学习是不够的（回顾）

SFT 赋予你的是**能力**，而非对政策的精通。经过 SFT 后，模型仍存在以下不足：

- 决定何时避免不必要的工具调用，
- 在多个工具之间根据权衡进行选择，
- 根据执行反馈调整参数，
- 多步骤规划，
- 最小化工具使用成本，
- 在收集到足够信息时停止。

模仿学习提供了起点，而强化学习则提供了实现真正工具使用熟练度所需的决策优化。

## 背景：使用强化学习教授智能体工具调用

### 动机

近年来，基于大型语言模型（LLMs）的工具增强推理范式逐渐受到关注：例如，秦等人（2023）发表的 [Tool Learning with Foundation Models](https://arxiv.org/abs/2304.08354) 系统性地阐述了基础模型如何通过选择和调用外部工具（如API）来解决复杂任务。

￼​教会大语言模型使用工具本质上是一个包含三个部分的学习问题：​ ​ 
1. 何时调用工具：决定对于给定的查询，工具调用是必要的、可选的还是不必要的。​ 
2. 调用哪个工具：在多个可用工具中选择正确的工具。​ 
3. 如何调用工具：生成有效且结构正确的参数，使工具能够成功执行。​ ​ ​

这三个类别分别对应决策层面、选择层面和参数层面的能力，每个层面都需要不同的监督和奖励信号。

冯等人（2025年）的 [ReTool: Reinforcement Learning for Strategic Tool Use in LLMs](https://arxiv.org/abs/2504.11536)  以及钱等人（2025年）的 [ToolRL: Reward is All Tool Learning Needs](https://arxiv.org/abs/2504.13958)  等先前研究表明，对工具学习问题进行细粒度分解能显著提升强化学习的稳定性和策略质量，尤其是在将调用工具的决策与实际调用机制分离时效果更为突出。

本文提出了一种完整的端到端强化学习方案，其中通过PPO或相关算法优化单一策略，同时学习：何时调用工具，选择哪个工具，以及如何构建正确的参数。

#### 为什么“何时/哪个/如何”分解是必要的

**时机：** 工具使用的时机决定了解决方案的效率和正确性。过度调用会导致不必要的成本和延迟，而调用不足则会导致答案不完整或不正确。因此，工具时机的选择形成了一个二元或多类策略决策，必须明确学习。

**选择工具：** 即使适合调用工具，模型也必须从 API 库中选择正确的工具。这是一个分类问题，需要结构化的动作空间和工具选择奖励。

**方法：** 工具参数必须是有效的 JSON 格式，与模式一致且语义正确。这是一个结构化生成问题，需要对语法、可执行性和参数质量进行奖励。

即使在同一条策略中，这些决策也需要不同的监督信号，而强化学习通过分离其奖励项使模型能理解轨迹优劣的原因，从而获益。

诸如 [ToolRL](https://arxiv.org/abs/2504.13958) 等研究表明，将这些不同能力分解为奖励组成部分，可以提高奖励信号的清晰度，减少信用分配的难度，并产生更可控的执行时行为。

### Recipe

以下是主要实施阶段的总结：

1. 定义一个环境与动作空间，需支持：时机决策（使用工具与否），工具选择决策，参数生成决策。
2. 为每个学习轴标注或推导标签：时间标签（when-labels）：ywhen ∈ {0,1}；工具标签（which-labels）：ywhich ∈ {1,…,K}（共K种工具）方法标签（how-labels）：参数模式示例或参考轨迹
3. 通过监督微调（模仿学习）引导大语言模型（LLM），使策略初步掌握以下基础能力：工具使用时机的判断，工具的选择方法，有效参数格式的识别
4. 设计一个多组件的奖励函数，包括：
	1. 时机奖励：用于正确的使用工具/不使用工具决策，
	2. 选择奖励：用于正确的工具选择
	3. 执行奖励：用于语法有效性、可执行性及参数质量，
	4. 最终任务成功奖励
5. 使用 PPO（或 GRPO）对具有组合奖励的轨迹进行训练：计算回报Rt，计算优势At（例如使用GAE），使用 KL 正则化更新策略和价值模型，以监督备用策略为基准。
6. 课程设计：从简单的监督轨迹逐步过渡到复杂的多步骤工作流程，模型需要在这些流程中交错做出“何时”、“选择哪个”以及“如何操作”的决策。
7. 诊断与评估：分别跟踪各维度的指标：时间准确性，内容准确性，参数正确性，可执行率，以及最终任务准确性。

### 环境、MDP 公式化和动作空间

工具增强型大语言模型在推理过程中必须做出三个决策：
1. 是否应该调用工具（何时）
2. 哪个工具是合适的（哪个）
3. 如何构建有效且高效的参数（如何）。

这种分解方式反映了Schick等人（2023年）在Toolformer系统中采用的行为因子分解方法，以及Yao等人（2022年）在ReAct中展现的结构化规划思路。同时，它也契合了Feng等人（2025年）在ReTool和ToolRL（2025年）等最新强化学习方法中的策略设计理念——将工具选择建模为多阶段决策过程。

#### “何时/哪个/如何” 的 MDP

我们将工具使用建模为一个马尔可夫决策过程（MDP）：$=(S,A,P,R,γ)$

……采用分解动作空间的方式，明确捕捉“何时/何种/如何”的结构。

#### State ($s_t$)

每个状态编码包括：

- 用户的查询
- 正在进行的推理步骤
- 过去的工具调用和输出
- 系统指令
- 可选的片段记忆（短期轨迹）

完整状态被序列化为一个结构化文本提示，输入到大型语言模型中，类似于 ReAct 风格的推理轨迹。

#### 结构化、分解的动作空间

动作空间分解为：何时调用工具、调用哪个工具（在调用的条件下）、如何构建参数（在选定工具的条件下），这产生了两种不相交的高级动作类型：

* 动作类型 1： `ANSWER(final_text)`，当模型确定不再需要工具调用时使用。
* 动作类型 2： `CALL(tool_name, Args_json)`，进一步分解为：
    - 何时：决定调用工具而非自行回答
    - 选择：从可用工具集中选取工具
    - 方式：为该工具生成有效的参数 JSON

这种分解通过确保强化学习梯度反映工具使用中的不同子技能，从而提升学习效果。

#### 结构化行动编码

为了稳定强化学习训练，每个动作都按照 ReTool 和 Toolformer 中的做法，以严格的机器可读 JSON 格式进行格式化。

`CALL action` 示例：

```json
<action>   
{     
	"type": "call",     
	"when": true,     
	"which": "weather_api",     
	"how": { 
		"city": "Berlin", 
		"date": "2025-05-09" 
	}   
} 
</action>
```

`ANSWER action` 示例：

```json
<action>   
{     
	"type": "answer",     
	"when": false,     
	"content": "It will rain in Berlin tomorrow."   
} 
</action>
```
 
显式或隐式地包含 when 标志；显式包含有助于调试和信用分配。

#### 剧集动态

一集的流程如下：

1. 大语言模型接收初始状态 $s_0$。
2. 大语言模型生成一个包含“何时/哪个/如何”的结构化动作 $a_0$。
3. 环境解析该动作：
    - 如果是 `ANSWER`→ 任务结束。
    - 如果是 `CALL`→ 执行工具，将输出添加到上下文，生成下一个状态 $s_1$。    
4. 根据“何时”、“哪个”、“如何”的正确性及最终答案质量计算奖励。
5. 持续该过程直到触发 ANSWER 或达到最大步数限制。

这种多步结构支持 ReAct 中使用的多跳推理，并与 Toolformer 中的任务设置保持一致。

#### 处理无效/格式错误操作

无效的“何时/哪个/如何”选择不应终止情节。相反：

* 分配负面语法或有效性奖励​
* 向模型返回错误信息​
* 允许代理继续

这与 Christiano等人（2017年）在《基于人类偏好的深度强化学习》中提出的奖励塑造策略一致。

#### 将工具调用的“何时/哪个/如何”整合到行动空间中

在强化学习优化过程中，策略梯度是针对整个结构化动作计算的，但奖励会沿着三个决策轴进行分解，PPO 或 GRPO 算法提供稳定的更新（如 ReTool 和 ToolRL 所示）

因此，策略会同步学习：何时使用工具是合适的，应该选择哪个工具，如何构建高质量的参数，这种模块化设计也让奖励工程变得更容易，因为每个组件都可以独立训练和调试。

### 奖励成分的注释来源（“何时”、“哪个”和“如何”）

本节说明如何为强化学习系统中的所有奖励组件生成监督信号，反映工具使用行为分解为：

- 何时 → 决定是否以及何时使用工具
- 哪个 → 选择调用哪个工具
- 如何 → 通过正确构造的参数构建调用方式

为支持这一点，奖励被分解为以下组成部分：
1. 调用（何时调用）：是否应调用工具。
2. 工具选择：是否选择了正确的工具（选哪个）。
3. 工具语法正确性：工具调用的格式是否正确。
4. 工具执行正确性：工具是否成功执行。
5. 参数质量：参数是否恰当（如何设置）。
6. 最终任务成功：整个流程是否产生了正确答案。
7. 基于偏好/生成式评估：更高层次的判断（LLM作为评判者）。

每个奖励维度可以通过以下方式的组合进行监督：
* 基于规则的启发式方法
* 基于人类数据训练的判别式奖励模型
* 生成式奖励模型（如Guo等人（2025）在 DeepSeek-R1 中提出的 LLM-as-a-Judge 方法）

#### 奖励组件：调用（决定“何时”应调用工具）

该组件支持“时机”维度：在推理过程中的当前节点，是否适合/有必要调用工具？

##### 基于规则的监督

使用确定性规则和意图检测器，灵感来源于 Schick 等人（2023年）的 Toolformer 等作品：
- 天气问题 → 需要天气 API
- 数学表达式 → 需要计算器
- “定义X / 解释Y” → 无需工具
- 事实查询 → 搜索工具
- 可执行任务（如预订） → 使用相应领域工具

这将生成二进制或分级标签 ycall∈0,1。

##### Discriminative Reward Model

- Train a classifier fϕ(x) predicting P(ycall=1∣x) using human-labeled examples indicating if/how strongly the query requires tool use.
- This mirrors methodology from RLHF as in [InstructGPT](https://arxiv.org/abs/2203.02155) by Ouyang et al. (2022).

##### Generative Reward Model (LLM-as-a-Judge)

- Use a judge model (e.g., DeepSeek-V3 per [DeepSeek-R1](https://arxiv.org/abs/2501.12948)):
    
- Prompt: “Given this user query and available tools, should the agent call a tool at this stage? Provide yes/no and reasoning.”
    
- Extract a scalar reward from the generative verdict.
    
- This can capture nuanced timing requirements over multiple steps.
    

#### Reward Component: Tool Selection (Choosing “Which” Tool)

- This component supports the **which** dimension: Given that a tool is to be called, was the _correct_ tool chosen?

##### Rule-based Supervision

- If rules map tasks to a specific tool or tool category, then:
    
    - If the predicted tool matches the rule → +reward
    - Otherwise → −reward
- This is similar to mapping tool types in [ReAct](https://arxiv.org/abs/2210.03629) by Yao et al. (2022).
    

##### Discriminative Reward Model

- Train a classifier fψ(st,at) that judges whether the selected tool matches human expectations for that state.

##### Generative Reward Model

- Ask a judge LLM: “Was TOOL_X the best tool choice for this request at this step?”
    
- Score the answer and normalize.
    

#### Reward Component: **Tool-Syntax Correctness**

- Supports the **how** dimension partially, focusing on _format_:
    
    - JSON validity
    - Required argument fields
    - Correct schema shape

##### Rule-based

- JSON parse success
- Schema validation
- Argument-type validation
    
- **Reward:**
    
    rsyntaxt={+1if JSON + schema valid −1otherwise
    
- This echoes structured action enforcement in [ReAct](https://arxiv.org/abs/2210.03629).

##### Discriminative Reward Model

- Classify correct vs. incorrect tool-call formats.

##### Generative Reward Model

- Ask an LLM judge whether the formatting is correct (1–10), normalize to reward.

#### Reward Component: **Tool-Execution Correctness**

- Did the tool run without error?

##### Rule-based

- HTTP 200 or success flag → +reward
- Errors / exceptions → −reward

##### Discriminative Reward Model

- Trained to predict execution feasibility or correctness.

##### Generative Reward Model

- Judge evaluates based on logs and outputs.

#### Reward Component: Argument Quality (Deciding “How” to Call a Tool)

- This is the core of the **how** dimension: constructing appropriate arguments.

##### Rule-based

- For numeric or structured problems:

rargst=−|apred−agold|

- For strings, use embedding similarity or fuzzy match.

##### Discriminative Reward Model

- Trained to identify argument errors (bad city name, missing date, etc.).

##### Generative Reward Model

- LLM-as-a-Judge evaluates argument plausibility/fit to the query.

#### Reward Component: **Final Task Success**

- Whether the overall trajectory produced a correct answer.

##### Rule-based

- Unit test pass
- Exact match
- Tolerance-based numeric match

##### Discriminative Reward Model

- Using preference modeling as in [Deep RL from Human Preferences](https://arxiv.org/abs/1706.03741) by Christiano et al. (2017), train:

RM=−logerϕ(τA)erϕ(τA)+erϕ(τB).

##### Generative Reward Model

- Judge LLM compares model prediction with ground truth (as in [DeepSeek-R1](https://arxiv.org/abs/2501.12948)).

#### Merged Preference-Based Rewards (For “Call”, “Which”, and “How”)

- You can construct pairs of trajectories differing in:
    
    - timing of tool calls (call),
    - choice of tool (which), and
    - argument construction (how)
- Let the judge or human annotator choose the better one.
    
- Train a preference RM to provide combined signals.
    

#### Unified Reward Formulation

- All reward signals—process and outcome—are merged into one scalar:
    
    R=wcallrcallwhen+(wtoolrtool)which+(wsyntaxrsyntax+wexecrexec+wargsrargs)how+(wtaskrtask+wprefrpref)outcome-level
    
    - where:
        
        - The **when** group controls _whether_ a tool is invoked.
        - The **which + how** group supervises _tool choice_ and _argument construction_.
        - The **outcome-level** group ensures the final result is correct and aligns with human/judge preferences.
- This single scalar reward R is what enters the RL optimizer (e.g., PPO or GRPO).
    
- Weights w are tuned to balance shaping vs. final correctness.
    

#### Asymmetric Rewards in Tool-Calling RL

- This section explains why tool-calling RL systems use **asymmetric rewards** (positive rewards much larger than negative rewards), how this stabilizes PPO/GRPO, and how asymmetry applies across the **when / which / how** components. A full worked example and a comprehensive reward table are included.
    
- Asymmetric reward schedules are used in practical tool-use RL systems such as ReTool, ToolRL, DeepSeek-R1, and RLHF pipelines. They ensure that:
    
    - Success is highly rewarded.
    - Failure incurs penalties but not catastrophic ones.
    - Exploration does not collapse into inert policies (e.g., “never call tools”).
    - The hierarchy — deciding **when** to call tools, **which** tool to call, and **how** to construct correct arguments — all receive stable and interpretable feedback.

##### Why Asymmetry is Required

- Because tool-calling introduces many potential failure points (incorrect timing, wrong tool, malformed arguments, bad final answer), symmetric rewards would cause massive early negative returns. The policy would quickly learn the degenerate strategy: “Never call any tool; always respond directly.”
    
- Asymmetric rewards avoid this by:
    
    - Using **large positive** rewards for correct full trajectories.
    - Using **mild or moderate negative** rewards for mistakes.
    - Ensuring that exploratory attempts are only _slightly_ penalized.
    - Allowing the policy to differentiate between “bad idea but learning” vs. “excellent behavior.”
- This encourages exploration in the factored action space and prevents PPO/GRPO from collapsing into trivial policies.
    

##### Reward Table: Positive and Negative Rewards by Category

- Below is a consolidated table representing **typical** asymmetric reward magnitudes for each component. These values are illustrative and are often tuned per domain.

###### Reward Values for “When / Which / How” and Outcome-Level Components

|**Reward Component**|**Description**|**Positive Reward Range**|**Negative Reward Range**|
|---|---|---|---|
|**When** (call decision)|Correctly calling a tool when needed|+0.5 to +1.5|−0.2 (tool required but not called)|
||Correctly not calling a tool|+0.3 to +1|−0.2 (tool called when unnecessary)|
|**Which** (tool selection)|Selecting correct tool|+0.5 to +2.0|−0.3 to −0.7 (wrong tool)|
|**How: Syntax**|JSON validity and schema correctness|+0.3 to +1.0|−1.0 (malformed JSON or wrong schema)|
|**How: Execution**|Tool executes successfully (HTTP 200, etc.)|+0.5 to +1.0|−1.0 to −2.0 (execution error)|
|**How: Argument Quality**|High-quality arguments (correct fields, values)|+0.5 to +2.0|−0.5 to −1.5 (missing/incorrect/poor arguments)|
|**Outcome: Final Task Success**|Producing correct final answer using tool output|+8.0 to +15.0|−0.3 to −1.0 (incorrect final answer)|
|**Outcome: Preference/Judge Score**|Judge or LLM-as-a-critic evaluation of final output|+1.0 to +5.0|−0.1 to −1.0|

- This table reflects the following structural principles:
    
    - The **largest rewards** are reserved for correct _end-to-end_ solution quality.
    - The **largest penalties** correspond only to errors that break execution (syntax, runtime failure).
    - Small errors in timing, selection, or argument quality incur **light penalties**.
    - Rewards across “when / which / how” are significantly **lower** than final-task success, ensuring shaping rewards guide early learning but final correctness dominates late learning.

##### Worked Example with Asymmetric Rewards

- Consider the user query: “What’s the weather in Paris tomorrow?”
    
- Correct behavior requires:
    
    1. Deciding a tool is required (**when**).
    2. Selecting the weather API (**which**).
    3. Providing correct arguments in JSON (**how**).
    4. Producing the correct final answer using the tool output.
- Below are two trajectories demonstrating asymmetry.
    

###### Trajectory A: Imperfect but Reasonable Exploration

1. **When** decision correct → +1.0
2. **Which** tool wrong → −0.5
3. JSON syntax valid → +0.5
4. Tool executes (but irrelevant) → 0
5. Final answer wrong → −0.5

- Total reward:

RA=1.0−0.5+0.5+0−0.5=0.5

- Even though the overall answer is wrong, the trajectory gets a _small positive_ reward because several subcomponents were correct. This prevents the model from concluding that tool use is too risky.

###### Trajectory B: Full Correct Behavior

1. Correct **when** → +1.0
2. Correct **which** → +1.5
3. Correct JSON arguments → +1.0
4. Successful tool execution → +1.0
5. Correct final answer → +10.0

- Total reward:

RB=1.0+1.5+1.0+1.0+10.0=14.5

- The tremendous difference between +14.5 and +0.5 clearly guides PPO/GRPO toward producing the full correct behavior.

##### How Asymmetry Stabilizes PPO/GRPO

- Advantages are computed via:

At=Rt−V(st)

- With asymmetric rewards:
    
    - Failed trajectories receive slightly negative or slightly positive returns.
    - Successful trajectories receive large positive returns.
    - Advantage variance stays manageable.
    - Exploration does not collapse into “never call tools.”
    - The policy improves steadily across “when / which / how” dimensions.
- If rewards were symmetric (e.g., +10 vs. −10), then most exploratory episodes would produce extreme negative advantages, instantly pushing the model toward refusing all tool calls. Asymmetry prevents this collapse.
    

##### Takeaways

- Asymmetric rewards are essential for training LLM tool-calling policies because they:
    
    - Preserve exploration.
    - Deliver stable gradients for PPO/GRPO.
    - Avoid trivial degenerate strategies.
    - Properly balance shaping rewards (for “when / which / how”) with outcome-level rewards.
    - Distinguish partial correctness from catastrophic failure.
    - Encourage correct final answers without over-penalizing small mistakes.
- The reward table and examples above provide a practical blueprint for implementing and tuning asymmetric rewards in your own RL tool-calling system.
    

### RL Optimization Pipeline: Shared Flow + PPO vs. GRPO

- This section describes how to take the unified reward from Section 3 and plug it into a full reinforcement learning (RL) pipeline—including both Proximal Policy Optimization (PPO) by [Schulman et al., 2017](https://arxiv.org/abs/1707.06347) and Group Relative Policy Optimization (GRPO) by [Shao et al., 2024](https://arxiv.org/abs/2402.03300). We present first the shared components, then algorithm‐specific losses and update rules.
- A detailed discourse of preference optimization algorithms is available in the [Preference Optimization](https://aman.ai/primers/ai/preference-optimization) primer.

#### Shared RL Training Flow

1. **Rollout Generation**:
    
    - Use the policy πθ (based on the LLM) to interact with the tool‐calling environment defined in Section 2.
    - At each step t you have state st, select action at (`CALL` tool or `ANSWER`), observe next state st+1, and receive scalar reward rt (from the unified reward).
    - Repeat until terminal (ANSWER) or maximum steps T.
    - Collect trajectories τ=(s0,a0,r0),…,(sT−1,aT−1,rT−1),(sT).
2. **Return and Advantage Estimation**:
    
    - Compute discounted return:
        
        Rt=∑k=tTγk−t,rk
        
    - Estimate value baseline Vψ(st) (for PPO) or compute group‐relative statistics (for GRPO).
        
        - Advantage (for PPO):
            
            At=Rt−Vψ(st)
            
            - Use Generalized Advantage Estimation (GAE) if desired (as typically done in PPO):
                
                A(λ)t=∑l=0∞(γλ)lδt+l,δt=rt+γVψ(st+1)−Vψ(st)
                
3. **Policy Update**:
    
    - Use a surrogate objective (dependent on algorithm) to update θ (policy), and update value parameters ψ where needed.
    - Optionally include a KL-penalty or clipping to ensure policy stability.
4. **Repeat**:
    
    - Collect new rollouts, update, evaluate. Monitor metrics such as tool‐call decision accuracy (“when”), correct tool selection (“which”), argument correctness (“how”), and final task success.

#### PPO: Losses and Update Rules

##### Surrogate Objective

- For PPO the objective is using clipped surrogate:

LPPO(θ)=𝔼s,a∼πθold[min(rt(θ)At,clip(rt(θ),1−ϵ,1+ϵ)At)]

- where:

rt(θ)=πθ(at∣st)πθold(at∣st)

- … and ϵ≈0.1−0.3.

##### Value Loss

Lvalue(ψ)=𝔼st∼π[(Vψ(st)−Rt)2]

##### KL/Entropy Penalty

- Often a term is added:

LKL(θ)=β,𝔼st,at∼πθ[logπθ(at|st)πref(at|st)]

- … to keep the policy close to either the old policy or a reference SFT policy.

##### Full PPO Loss

LtotalPPO=−LPPO(θ)+cv,Lvalue(ψ)+cKL,LKL(θ)

- … with coefficients cv,cKL.

##### Implementation Notes

- Use mini-batches and multiple epochs per rollout.
- Shuffle trajectories, apply Adam optimizer.
- Clip gradients; log metrics for tool decisions and argument quality.

#### GRPO: Losses and Update Rules

##### Group Sampling & Relative Advantage

- In GRPO [Shao et al., 2024] you sample a group of G actions (a1,…,aG) under the same state s. Compute each reward r(s,aj). Then define group mean and standard deviation: μ,σ. Advantage for each is:

AGRPO(s,aj)=r(s,aj)−μσ

##### GRPO Surrogate

LGRPO(θ)=1G∑j=1G𝔼s,a1:G∼πθold[min(rj(θ)AGRPO(s,aj),clip(rj(θ),1−ϵ,1+ϵ)AGRPO(s,aj))]

- … with the same ratio definition rj(θ)=πθ(aj∣s)/πθold(aj∣s).

##### Value Loss

- GRPO typically **omits** a parametric value estimator—baseline derived via group statistics.

##### KL/Entropy Penalty

- Same form as in PPO if desired.

##### Full GRPO Loss

LtotalGRPO=−LGRPO(θ)+cKLLKL(θ)

##### Implementation Notes

- At each state draw multiple candidate tool/answer actions, compute rewards, form group.
- This is particularly suited for LLM tool-calling contexts where you can generate multiple alternate completions.
- GRPO reduces reliance on value network.

#### Integrating the Unified Reward

- Given the unified reward R from the prior step, each step’s rt is used in return and advantage estimation. The policy thus simultaneously learns “when/which/how” tool calling by maximizing return:

J(θ)=𝔼τ∼πθ[∑t=0Tγtrt]

- Both PPO and GRPO approximate gradient ascent on J(θ) under stability constraints.

### Curriculum Design, Evaluation Strategy, and Diagnostics for Tool-Calling RL

- This section describes how to structure training so the model reliably learns **when**, **which**, and **how** to call tools, and how to evaluate progress during RL. Curriculum design is crucial because tool-calling is a hierarchical skill; introducing complexity too early destabilizes learning, and introducing it too late yields underfitting.

#### Curriculum Design Overview

- Curriculum design gradually increases difficulty along three axes:
    
    1. **When** → recognizing tool necessity vs. non-necessity
    2. **Which** → selecting the correct tool
    3. **How** → providing high-quality arguments
- Each axis has its own progression. The curriculum alternates between breadth (many domains/tools) and depth (multi-step workflows).
    
- This staged approach mirrors the structured curricula seen in code-generation RL (e.g., unit-tests → multi-step tasks) in works like [Self-Refine](https://arxiv.org/abs/2303.17651) by Madaan et al. (2023).
    

#### Stage 0: Pure Supervised Bootstrapping (SFT)

- Before RL begins, do supervised fine-tuning on a dataset that explicitly includes:
    
    - Examples requiring a tool,
    - Examples that must _not_ use a tool,
    - Examples mapping queries to correct tool types,
    - Examples showing valid argument formats.
- The SFT initializes:
    
    - An approximately correct “when → which → how” policy,
    - JSON formatting reliability,
    - Stable tool-calling syntax.
- This prevents “flailing” during early RL where the model might emit random tool calls.
    

#### Stage 1: Binary Decision Curriculum (Learning **When**)

- **Focus:** detect whether a tool is required.
    
- **Task mix:**
    
    - 50% queries that require a specific tool (weather/math/search)
    - 50% queries that must be answered without tools
- **Goal:** learn the call/no-call boundary.
    
- **Metrics:**
    
    - Call precision
    - Call recall
    - False-positive rate (unnecessary calls)
    - False-negative rate (missed calls)
- **Reward emphasis:**
    
    - Increase (w_{\text{call}})
    - Reduce penalties for syntax/execution errors early on

#### Stage 2: Tool-Selection Curriculum (Learning **Which**)

- Add tasks that require choosing _between_ tools:
    
- **Task examples:**
    
    - Weather vs. news
    - Search vs. calculator
    - Translation vs. summarization (if tools exist)

**Goal:** learn discriminative mapping from task intent → tool identity.

- **Curriculum trick:**
    
    - For ambiguous queries, include diverse examples so the RL agent learns to think (internal chain-of-thought) before issuing tool calls.
- **Metrics:**
    
    - Tool-selection accuracy
    - Confusion matrix across tool categories
    - Average number of tool attempts per query
- **Reward emphasis:**
    
    - Shift weight from (w_{\text{call}}) → (w_{\text{which}})
    - Introduce penalties for repeated incorrect tool choices

#### Stage 3: Argument-Construction Curriculum (Learning **How**)

- Introduce tasks with argument complexity:
    
    - **Task examples:**
        
        - Weather(city, date)
        - Maps(location, radius)
        - Calculation(expressions with multiple steps)
        - API requiring nested JSON fields
    - **Training strategy:**
        
        - Start with minimal arguments (one field)
        - Add multi-argument calls
        - Introduce noisy contexts (typos, ambiguity)
    - **Metrics:**
        
        - Argument correctness (string similarity or numeric error)
        - Schema completeness
        - Tool execution success rate
    - **Reward emphasis:**
        
        - Increase wargs
        - Tighten penalty for malformed JSON or missing fields

#### Stage 4: Multi-Step Tool Use (Pipelines)

- Introduce tasks requiring **multiple sequential tool calls**, e.g.:
    
    1. Search for restaurants
    2. Get the address
    3. Query weather at that address
    4. Produce a combined answer
- Here the agent must plan sequences and must choose when to stop calling tools.
    
- **Metrics:**
    
    - Number of steps per episode
    - Optimality of tool sequence
    - Rate of premature or redundant tool calls
- **Reward emphasis:**
    
    - Add step penalties
    - Strengthen outcome reward since multi-step tasks dominate final task success

#### Stage 5: Open-Domain Free-Form Tasks

- Finally, mix in diverse real-world questions with unconstrained natural-language variety.
    
- **Goal:** produce a robust “universal” tool-use agent.
    
- **Metrics:**
    
    - Overall episodic return
    - Win-rate vs. evaluator models (LLM-as-a-Judge)
    - Human preference win-rate
    - Task success accuracy in open benchmarks

#### Diagnostics and Monitoring

##### Process-Level Metrics

- Aligned with the **when → which → how** decomposition:
    
    - **When:**
        
        - Call precision/recall
        - Unnecessary call rate
        - Missed call rate
        - Call timing consistency
    - **Which:**
        
        - Tool selection accuracy
        - Error matrix across tools
        - Repeated incorrect tool selection episodes
    - **How:**
        
        - Argument correctness scores
        - JSON validity rate
        - Execution success rate

##### Outcome-Level Metrics

- **Final answer accuracy:**
    
    - Exact match
    - Tolerance-based match
    - Semantic similarity
    - Pass rate vs. LLM-judge (DeepSeek-V3, GPT-4, etc.)
- **Task efficiency:**
    
    - Number of steps per solved task
    - Number of tool calls per successful episode
    - Reward per timestep
- **User-facing metrics:**
    
    - Latency per episode
    - Number of external API calls

#### Detecting Skill Collapse

- **Red flags include:**
    
    - Spike in JSON errors → syntax collapse
    - Rising unnecessary tool use → call collapse
    - Tool-selection deterioration → “which” collapse
    - Rising tool execution failures → argument collapse
    - Flat final-task accuracy → plateau due to overfitting on shaping rewards
- **Solutions:**
    
    - Adjust reward weights w⋅
    - Reintroduce supervised examples
    - Increase entropy regularization
    - Add KL penalties to keep model close to reference

#### Curriculum Scheduling (Putting It All Together)

- **A typical recipe:**
    
    1. **Stage 0 (SFT):** 30k–200k examples
    2. **Stage 1 (When):** 1–5 RL epochs
    3. **Stage 2 (Which):** 3–10 RL epochs
    4. **Stage 3 (How):** 5–20 RL epochs
    5. **Stage 4 (Pipelines):** 10–30 RL epochs
    6. **Stage 5 (Open-domain):** continuous RL/adaptation
- **Dynamic curriculum:** shift task sampling probabilities based on evaluation metrics—for example, increase argument-focused tasks if argument correctness stagnates.
    

#### Final Note

- A well-designed curriculum ensures the policy does not simply memorize tool-call structures but truly internalizes:
    
    - **when** tool use is warranted,
    - **which** tool to call,
    - **how** to call it correctly,
    - … and how to combine tools into multi-step workflows to solve real tasks.

### Reinforcement Learning and the Emergence of Intelligent Agents

- With the rise of Large Language Models (LLMs) and multimodal foundation models, RL has become a critical mechanism for developing autonomous, reasoning-capable agents. Early efforts demonstrated that LLMs could act as agents that browse the web, search for information, and perform tasks by issuing actions and interpreting observations.
    
- One of the first large-scale examples was **[WebGPT](https://arxiv.org/abs/2112.09332)** by Nakano et al. (2022), which extended GPT-3 to operate in a simulated text-based browsing environment. The model was trained through a combination of imitation learning and reinforcement learning from human feedback (RLHF).
    
    - WebGPT introduced a **text-based web interface** where the model interacts via discrete commands such as _Search_, _Click_, _Quote_, _Scroll_, and _Back_, using the Bing Search API as its backend. Human demonstrators first generated browsing traces that the model imitated through **behavior cloning**, after which it was fine-tuned via **PPO** against a **reward model** trained on human preference data. The reward model predicted human judgments of factual accuracy, coherence, and overall usefulness.
    - Each browsing session ended when the model issued “End: Answer,” triggering a synthesis phase where it composed a long-form response using the collected references. The RL objective included both a terminal reward from the reward model and a per-token KL penalty to maintain policy stability. Empirically, the best 175B “best-of-64” WebGPT model achieved human-preference rates of **56% over human demonstrators** and **69% over Reddit reference answers**, showing the success of combining structured tool use with RLHF.
    - The following figure ([source](https://arxiv.org/abs/2112.09332)) shows the text-based browsing interface used in WebGPT, where the model issues structured commands to retrieve and quote evidence during question answering.
    
    ![](https://aman.ai/primers/ai/assets/RL-for-agents/WebGPT.jpg)
    
- Subsequent systems expanded these capabilities. **[Agent Q](https://arxiv.org/abs/2408.07199)** by Putta et al. (2024) introduced a hybrid RL pipeline that integrates **Monte Carlo Tree Search (MCTS)** with **Direct Preference Optimization (DPO)**.
    - Agent Q formalizes decision making as a **reasoning tree**, where each node represents a thought–action pair and edges correspond to plausible continuations. MCTS explores multiple reasoning branches guided by a value model estimating downstream reward. During training, preference data between trajectories is used to train a DPO objective, directly optimizing the policy toward preferred rollouts without relying on an explicit reward scalar.
    - This setup enables **off-policy reuse** of exploratory trajectories: the model learns from both successes and failures by evaluating them through a learned preference model. Empirically, this led to substantial gains in reasoning depth and factual accuracy across multi-step question answering benchmarks, demonstrating that structured search and preference-based policy updates can yield stronger reasoning alignment than gradient-only PPO approaches.
- More recent advancements such as **[OpenWebVoyager](https://arxiv.org/abs/2410.19609)** by He et al. (2024) brought these ideas into the multimodal realm. OpenWebVoyager extends open-source multimodal models (Idefics2-8B-Instruct) to perform real-world web navigation using both **textual accessibility trees** and **visual screenshots**. The training process unfolds in two phases:
    
    1. **Imitation Learning (IL)**: The model first learns from expert trajectories collected with GPT-4o via the WebVoyager-4o system. Each trajectory contains sequences of _thoughts_ and _actions_ derived from multimodal observations (screenshot + accessibility tree). The IL objective jointly maximizes the log-likelihood of both action and reasoning token sequences:
        
        JIL(θ)=E(q,τ)∼DIL∑t[logπθ(at|q,ct)+logπθ(ht|q,ct)]
        
    2. **Exploration–Feedback–Optimization Cycles**: After imitation, the agent autonomously explores the open web, generating new trajectories. GPT-4o then acts as an _automatic evaluator_, labeling successful trajectories that are retained for fine-tuning. Each cycle introduces newly synthesized tasks using the **Self-Instruct** framework, ensuring continuous policy improvement. Iteratively, the task success rate improves from **19.9% to 25.8%** on WebVoyager test sets and from **6.3% to 19.6%** on cross-domain Mind2Web tasks.
        
    
    - The following figure ([source](https://arxiv.org/abs/2410.19609)) shows the overall process of OpenWebVoyager, including the Imitation Learning phase and the exploration–feedback–optimization cycles.
    
    ![](https://aman.ai/primers/ai/assets/RL-for-agents/OpenWebVoyager.jpg)
    
    - The following figure ([source](https://arxiv.org/abs/2410.19609)) shows the model architecture of OpenWebVoyager. The system uses the most recent three screenshots and the current accessibility tree to guide multimodal reasoning, ensuring temporal grounding across page transitions.
    
    ![](https://aman.ai/primers/ai/assets/RL-for-agents/OpenWebVoyager2.jpg)
    
- Alongside real-environment exploration, a complementary approach is to scale policy learning with synthetic but reasoning-grounded interaction data. **DreamGym**, proposed in ([Scaling Agent Learning via Experience Synthesis](https://arxiv.org/abs/2511.03773) by Chen et al. (2025)), formalizes this by training a reasoning-based _experience model_ that serves as both a generative teacher and an adaptive simulator. This model produces synthetic task curricula and consistent next-state transitions, enabling closed-loop reinforcement learning at scale.
    
    - The framework introduces _experience synthesis_ as a core principle—training a language-conditioned simulator capable of generating realistic interaction traces that preserve reasoning consistency and causal coherence. By jointly optimizing the policy and the experience model under trust-region constraints, DreamGym maintains stability and theoretical convergence guarantees: if the model error and reward mismatch remain bounded, improvements in the synthetic domain provably transfer to real-environment performance.
    - The result is a unified infrastructure that decouples exploration (handled by the experience model) from policy optimization, dramatically reducing real-environment sample costs while preserving fidelity in reasoning tasks. Empirically, DreamGym demonstrates significant gains in multi-tool reasoning, long-horizon planning, and web navigation.
    - The following figure illustrates that compared to the traditional agent learning paradigm, DreamGym provides the first scalable and effective RL framework with unified infrastructure.
    
    ![](https://aman.ai/primers/ai/assets/RL-for-agents/DreamGym1.jpg)
    
- **Early Experience**, proposed in ([Agent Learning via Early Experience](https://arxiv.org/abs/2510.08558) by Zhang et al. (2025)), establishes a two-stage curriculum—implicit world modeling and self-reflection over alternative actions—that uses only language-native supervision extracted from the agent’s own exploratory branches, before any reward modeling or PPO/GRPO.
    
    - The first stage, _implicit world modeling_, trains the agent to predict environmental dynamics and next states, effectively learning the structure of interaction without any external reward. The second stage, _self-reflection_, asks the agent to introspectively compare expert and non-expert behaviors, generating rationale-based preferences that bootstrap value alignment.
    - These objectives serve as pre-RL signals that warm-start the policy, leading to faster and more stable convergence once reinforcement learning begins. In empirical evaluations, the Early Experience framework significantly improves downstream success rates across both web-based and software-agent benchmarks, and integrates seamlessly with later RL fine-tuning methods like PPO or GRPO.
    - The following figure shows the progression of training paradigms. (Left:) The Era of Human Data relies on expert demonstrations, where supervision comes from human-/expert-curated actions; it is reward-free (i.e., does not require the environment to provide verifiable reward) but not data-scalable. (Right:) The envisioned Era of Experience builds upon environments with verifiable rewards, using them as the primary supervision for reinforcement learning; however, many environments either lack such rewards (Xue et al., 2025) or require inefficient long-horizon rollouts (Xie et al., 2024a). Center: Our Early Experience paradigm enables agents to propose actions and collect the resulting future states, using them as a scalable and reward-free source of supervision
    
    ![](https://aman.ai/primers/ai/assets/RL-for-agents/EarlyExperience1.jpg)
    

### The Role of Reinforcement Learning in Self-Improving Agents

- RL serves as the foundation of _self-improving_ artificial agents. These agents do not depend solely on human-provided supervision; instead, they learn continuously from their own experiences.
    
- A representative example of this approach is [Large Language Models Can Self-improve at Web Agent Tasks](https://arxiv.org/abs/2405.20309) by Patel et al. (2024), which introduced a looped learning process where an agent repeatedly performs tasks, evaluates its own performance, and fine-tunes itself on the best results. In their experiments, agents improved their web-navigation success rates by over 30% without any additional human data, demonstrating that RL can bootstrap the agent’s progress over time.
    
- The following figure shows ([source](https://arxiv.org/abs/2405.20309)) the _self-improvement loop_ used in Patel et al. (2024), illustrating how the agent collects trajectories, filters low-quality outputs, fine-tunes itself, and iterates for continual improvement.
    

![](https://aman.ai/primers/ai/assets/RL-for-agents/WebArena.jpg)

- Synthetic-experience RL closes the loop for self-improving agents by letting a reasoning experience model synthesize adaptive rollouts and curricula matched to the current policy, yielding consistent gains in both synthetic and sim-to-real settings; theory further bounds the sim-to-real gap by reward-accuracy and domain-consistency errors, rather than strict pixel/state fidelity metrics (cf. [Scaling Agent Learning via Experience Synthesis](https://arxiv.org/abs/2511.03773) by Chen et al. (2025)).
    
- This iterative process typically follows these stages:
    
    1. **Data Collection:** The agent generates task trajectories by interacting with the environment.
    2. **Filtering and Evaluation:** The system automatically assesses each trajectory, discarding low-quality samples.
    3. **Fine-Tuning:** The agent is retrained using successful examples, effectively reinforcing good behavior.
    4. **Re-evaluation:** The improved agent is tested, and the cycle repeats.
- This form of continual self-improvement makes RL a key enabler for developing general-purpose, autonomous web and software agents.
    

### Environments for Reinforcement Learning in Modern Agents

- To support these learning processes, researchers have developed structured environments that simulate the complexity and variety of real-world digital interactions. One comprehensive framework is [AgentGym](https://arxiv.org/abs/2406.04151) by Xi et al. (2024), which defines a unified interface for training and evaluating LLM-based agents across 14 environment types—ranging from academic reasoning and games to embodied navigation and web interaction.
    
- The following figure ([source](https://arxiv.org/abs/2406.04151)) shows the _AgentGym framework_, illustrating the standardized environment interface, modular design, and integration of various environment types for LLM-driven agent training.
    

![](https://aman.ai/primers/ai/assets/RL-for-agents/AgentGym.jpg)

- In AgentGym, an agent’s experience is modeled as a trajectory consisting of repeated _thought–action–observation_ cycles:
    
    τ=(h1,a1,o1,...,hT,aT)∼πθ(τ|e,u)
    
    - where ht represents the agent’s internal reasoning (its “thought”), at the action it takes, ot the resulting observation, and e,u the environment and user prompt respectively.
- This approach bridges the symbolic reasoning capabilities of LLMs with the sequential decision-making framework of RL, forming the basis for modern interactive agents.
    

## The Three Major Types of Reinforcement Learning Environments

- Modern RL environments for language-based and multimodal agents are generally organized into three broad categories. Each category captures a distinct interaction pattern and optimizes the agent for a different type of intelligence or capability.

### Single-Turn Environments (SingleTurnEnv)

- These environments are designed for tasks that require only a single input–output interaction, where the agent must produce one decisive response and then the environment resets. Examples include answering a question, solving a programming challenge, or completing a math problem.
    
- In this setting, the reward signal directly evaluates the quality of the single output. Training methods usually combine supervised fine-tuning with RL from human or synthetic feedback (RLHF). For instance, in coding problems or reasoning benchmarks, the agent’s response can be automatically graded using execution correctness or symbolic validation. Such setups are ideal for optimizing precision and factual correctness in domains where each query is independent of the previous one.
    
- SingleTurnEnv tasks are computationally efficient to train because there is no need to maintain long-term memory or context. They are commonly used to bootstrap an agent’s basic competencies before moving to more complex, multi-step environments.
    

### Tool-Use Environments (ToolEnv)

- Tool-use environments focus on enabling agents to perform reasoning and decision-making that involve invoking external tools—such as APIs, search engines, calculators, code interpreters, or databases—to complete a task. These environments simulate the agent’s ability to extend its cognitive boundaries by interacting with external systems.
    
- In [Tool Learning with Foundation Models](https://doi.org/10.1145/3704435) by Qin et al. (2024), the authors surveyed a wide range of approaches where foundation models learn to select, call, and integrate the outputs of external tools into their reasoning processes. This kind of training allows the model to perform symbolic computation, factual verification, and data retrieval in ways that pure text-based reasoning cannot.
    
- The following figure shows ([source](https://doi.org/10.1145/3704435)) the _conceptual overview of tool learning with foundation models_, where models dynamically decide when and how to invoke tools such as web search and other APIs to solve complex problems.
    

![](https://aman.ai/primers/ai/assets/RL-for-agents/ToolLearningOverview.jpg)

- A related innovation is [Tool-Augmented Reward Modeling](https://arxiv.org/abs/2310.01045) by Li et al. (2024), which enhanced RL reward models by giving them access to external APIs such as search engines or translation systems. This modification made reward models not only more accurate but also more interpretable, as each decision could be traced through explicit tool calls.
    
- The following figure ([source](https://arxiv.org/abs/2310.01045)) shows illustrates the pipeline of (a) Vanilla reward models (RMs); (b) Tool-augmented RMs, namely Themis; (c) RL via proximal policy optimization (PPO) on above RMs; (d) Examples of single or multiple tool use process in the proposed approach.
    

![](https://aman.ai/primers/ai/assets/RL-for-agents/Themis.jpg)

- Tool-use environments test the agent’s ability to decide _when_ and _how_ to use a tool, what input arguments to provide, and how to interpret the returned results. This capability is crucial for building practical software assistants and web agents that interact with real systems.

### Multi-Turn, Sequential Environments (MultiTurnEnv)

- Multi-turn environments represent the most complex and realistic category of RL settings. In these environments, an agent engages in extended, multi-step interactions where each decision depends on the evolving context and memory of previous steps. Examples include navigating a website, writing and revising code iteratively, managing files on a computer, or executing multi-phase workflows such as online booking or document editing.
    
- Agents operating in these environments must reason about long-term goals, plan multiple actions in sequence, and interpret feedback dynamically. Systems such as WebArena, WebShop, [Agent Q](https://arxiv.org/abs/2408.07199) by Putta et al. (2024), and [OpenWebVoyager](https://arxiv.org/abs/2410.19609) by He et al. (2024) exemplify this paradigm. They train agents through multi-step RL using trajectory-based feedback, where each complete sequence of actions and observations contributes to the learning signal.
    
- These environments are optimized for developing autonomy and adaptability. The agent must not only predict the next best action but also understand how that action contributes to the overall task objective. MultiTurnEnv scenarios are thus the closest analogs to real-world usage, making them essential for training general-purpose digital agents.
    

### Implications

- Agentic RL, which is the evolution of RL for agents—from single-turn tasks to tool-augmented reasoning and complex multi-turn workflows—reflects a progressive layering of capabilities. Each environment type plays a distinct role:
    
    - Single-turn environments emphasize _accuracy and efficiency_, teaching agents to produce correct, concise responses.
    - Tool-use environments focus on _functional reasoning and integration_, giving agents the ability to extend their knowledge through computation and external APIs.
    - Multi-turn environments train _autonomy and planning_, enabling agents to navigate, adapt, and make decisions across extended sequences of interactions.
- Together, these environments form the backbone of modern RL for LLM-based and multimodal agents. They provide a structured pathway for training models that can perceive, reason, and act—bringing us closer to general-purpose artificial intelligence capable of performing diverse tasks in real-world digital environments.
    

## Reinforcement Learning for Web and Computer-Use Agents

- A detailed discourse on RL can be found in our [Reinforcement Learning](https://aman.ai/primers/ai/reinforcement-learning) primer.

### Background: Policy-Based and Value-Based Methods

- At its core, RL employs two broad families of algorithmic approaches:
    
    - Value-based methods, which learn a value function (e.g., Q(s,a) or V(s)) that estimates the expected return of taking action a in state s (or being in state s).
    - Policy-based (or actor-critic) methods, which directly parameterize a policy πθ(a∣s) and optimize its parameters θ to maximize expected return
        
        J(πθ)=𝔼τ∼πθ[∑t=0TγtR(st,at)]
        
- In modern agentic applications (web agents, computer-use agents), policy‐based methods tend to dominate because the action space is large, discrete (e.g., “click link”, “invoke API”, “enter code”), and policies must be expressive.
    
- One widely used algorithm is Proximal Policy Optimization (PPO) [Schulman et al. (2017)](https://arxiv.org/abs/1707.06347), which introduces a clipped surrogate objective to ensure stable updates and avoid large shifts in policy space.
    
- The surrogate objective can be expressed as:
    
    LCLIP(θ)=𝔼s,a∼πθold[min(rt(θ)At,clip(rt(θ),1−ϵ,1+ϵ)At)]
    
    - where rt(θ)=πθ(at∣st)πθold(at∣st) and At is the advantage estimate at time t.
- This ensures that the policy update does not diverge too far from the previous one while still improving expected return.
    

### Background: Process-Wise Rewards vs. Outcome-Based Rewards

- When designing RL systems for digital agents, one of the most consequential design choices lies in _how_ rewards are provided to the model.
    
- **Outcome-based rewards** give feedback only at the end of a task—for instance, a success/failure score after the agent completes a booking or answers a question. This is common in _SingleTurnEnv_ tasks and short workflows, where each interaction produces a single measurable outcome.
    
    - While simple, outcome-based rewards are _sparse_, often forcing the agent to explore many possibilities before discovering actions that yield high return.
- **Process-wise (step-wise) rewards**, in contrast, provide incremental feedback during the task. In a web-navigation scenario, for example, the agent might receive positive reward for successfully clicking the correct link, partially filling a form, or retrieving relevant information—even before the final goal is achieved.
    
    - This approach is critical in _MultiTurnEnv_ or _ToolEnv_ setups where tasks span many steps. By assigning intermediate rewards, process-wise systems promote _shaped learning_—accelerating convergence and improving interpretability of the agent’s learning process.
- Formally, if an episode runs for T steps, the total return under step-wise rewards is:
    
    Rt=∑k=tTγk−trk
    
    - where rk are per-step rewards. In outcome-based schemes, rk=0 for all k<T, and rT encodes task success. Choosing between these schemes depends on the environment’s complexity and availability of fine-grained performance metrics.
- For web agents, hybrid strategies are often used: process-wise signals derived from _browser state_ (e.g., correct navigation, reduced error rate) combined with final outcome rewards (task completion). This hybridization reduces the high variance of pure outcome-based rewards while preserving the integrity of long-horizon objectives.
    

### Reinforcement Learning from Human Feedback (RLHF) and Direct Preference Optimization (DPO)

- For web/computer-use agents built on LLMs or similar, one key method is RL from Human Feedback (RLHF). The standard RLHF pipeline is:
    
    1. Supervised fine-tune a base language model on prompt–response pairs.
    2. Collect human preference data: for each prompt, have humans rank multiple model responses (or choose preferred vs. non-preferred).
    3. Train a reward model rϕ(x,y) to predict human preferences.
    4. Use an RL algorithm (often PPO) to optimize the policy πθ to maximise expected reward under the reward model, possibly adding KL-penalty to stay close to base model.
- For example, the survey article [Reinforcement Learning Enhanced LLMs: A Survey](https://arxiv.org/abs/2412.10400v1) provides an overview of this field.
    
- However, RLHF can be unstable, costly in compute, and sensitive to reward-model errors. Enter Direct Preference Optimization (DPO) [Rafailov et al. (2023)](https://arxiv.org/abs/2305.18290), which posits that one can skip the explicit reward model + RL loop and simply fine-tune the model directly to optimize human preference pairwise comparisons.
    
- The DPO loss in the pairwise case (winner yw, loser yl) is approximately:
    
    DPO=−𝔼(x,yw,yl)[lnσ(βlnπθ(yw|x)πref(yw|x)−βlnπθ(yl|x)πref(yl∣x))]
    
    - where πref is the reference model (often the supervised fine-tuned model), and β is a temperature-like constant.
- Some practical analyses (e.g., [Is DPO Superior to PPO for LLM Alignment?](https://arxiv.org/abs/2404.10719)) compare PPO vs. DPO in alignment tasks.
    

### Why These Algorithms Matter for Web & Computer-Use Agents

- When training agents that interact with the web or software systems (for example, clicking links, filling forms, issuing API calls), several factors make the choice of algorithm especially important:
    
    - Action spaces are large and heterogeneous (e.g., browser UI actions, tool function calls).
    - The reward signals may be sparse (e.g., task success only after many steps) or come from human annotation (in RLHF).
    - Policies must remain stable and avoid drift (especially when built on pretrained LLMs).
    - Computation cost is high (LLM inference, environment simulation), so sample efficiency matters.
- Thus:
    
    - Algorithms like PPO are well-suited because of their stability and simplicity (compared to e.g. TRPO) in high-dimensional policy spaces.
    - RLHF/DPO are relevant because many web-agents and computer-agents are aligned to human goals (helpfulness, correctness, safety) rather than just raw reward.
    - There is an increasing trend toward hybrid methods that combine search, planning (e.g., MCTS) plus RL fine-tuning for complex workflows.

### Key Equations

#### Advantage Estimation & Value Networks

- In actor–critic variants (including PPO), we often learn a value function Vψ(s) to reduce variance:
    
    At=Rt−Vψ(st)Rt=∑k=0∞γkrt+k
    
    - **where:**
        
        - At: the **advantage estimate** at timestep t, measuring how much better an action performed compared to the policy’s expected performance.
        - Rt: the **discounted return**, or the total expected future reward from time t.
        - γ: the **discount factor** (0<γ≤1), controlling how much future rewards are valued compared to immediate ones.
        - rt+k: the **immediate reward** received at step t+k.
        - Vψ(st): the **critic’s value estimate** for state st, parameterized by ψ, representing the expected return from that state under the current policy.
- The update for the critic aims to minimize:
    
    Lvalue(ψ)=𝔼st∼π[(Vψ(st)−Rt)2]
    
    - **where:**
        
        - Lvalue(ψ): the **value loss**, quantifying how far the critic’s predictions are from the actual returns.
        - 𝔼st∼π[⋅]: the **expectation** over states st sampled from the current policy (\pi).
        - The squared term (Vψ(st)−Rt)2: penalizes inaccurate value predictions, guiding the critic to estimate returns more accurately.

#### KL-penalty / Trust Region

- Some RLHF implementations add a penalty to keep the new policy close to the supervised model:
    
    LKL(θ)=β⋅𝔼x,y∼π[logπθ(y|x)πSFT(y|x)]
    
    - **where:**
        
        - LKL(θ): the **KL-divergence loss**, which penalizes the new policy πθ if it deviates too far from the supervised fine-tuned (SFT) reference policy πSFT.
        - β: a **scaling coefficient** controlling the strength of this regularization; larger β enforces tighter adherence to the reference model.
        - 𝔼x,y∼π[⋅]: the **expectation** over sampled input–output pairs from the current policy’s distribution.
        - πθ(y∣x): the **current policy’s probability** of generating output y given input x.
        - πSFT(y∣x): the **reference policy’s probability**, often from the supervised model used before RL fine-tuning.
    - … so the total objective may combine PPO’s surrogate loss with this KL penalty (and possibly an entropy bonus) to balance exploration, stability, and fidelity to the base model.*
        

#### Preference Optimization (DPO)

- As shown above, DPO reframes alignment as maximising the probability that the fine-tuned model ranks preferred outputs higher than non-preferred ones, bypassing the explicit RL loop.

#### Sample Efficiency & Off-policy Corrections

- For agents interacting with web or tools where running many episodes is costly, sample efficiency matters. Off-policy methods (e.g., experience replay) or offline RL variants (e.g., [A Survey on Offline Reinforcement Learning](https://arxiv.org/abs/2203.01387) by Kumar et al. (2022)) may become relevant.

### Agentic Reinforcement Learning Via Policy Optimization

- In **policy optimization**, the agent learns from a unified reward function that draws its signal from **one or more available sources**—such as **rule-based rewards**, a scalar reward output from a **learned reward model**, or another model that is proficient at grading the task (such as an **LLM-as-a-Judge**). Each policy update seeks to maximize the expected cumulative return:
    
    J(θ)=𝔼πθ[∑tγtrt]
    
    - where rt represents whichever reward signal is active for the current environment or training regime. In some settings, this may be a purely rule-based signal derived from measurable events (like navigation completions, form submissions, or file creations). In others, the reward may come from a trained model Rϕ(ot,at,ot+1) that generalizes human preference data, or from an external proficient verifier (typically a larger model) such as an LLM-as-a-Judge.
- These components are **modular and optional**—only one or several may be active at any time. The optimization loop remains identical regardless of source: the policy simply maximizes whichever scalar feedback rt it receives. This flexible design allows the same framework to operate with deterministic, model-based, or semantic reward supervision, depending on task complexity, available annotations, and desired interpretability.
    
- **Rule-based rewards** form the foundation of this framework, providing deterministic, auditable feedback grounded in **explicit environment transitions and observable state changes**. As demonstrated in [DeepSeek-R1: Incentivizing Reasoning Capability in Large Language Models](https://arxiv.org/abs/2501.12948) by Gao et al. (2025), rule-based rewards yield transparent and stable optimization signals that are resistant to reward hacking and reduce reliance on noisy human annotation. In the context of computer-use agents, rule-based mechanisms correspond directly to **verifiable milestones** in user interaction sequences—for example:
    
    - In **web navigation**, detecting a URL transition, page load completion, or DOM state change (`NavigationCompleted`, `DOMContentLoaded`).
    - In **form interaction**, observing DOM model deltas that indicate fields were populated, validation succeeded, or a “Submit” action triggered a confirmation dialog.
    - In **file handling/artifact generation**, confirming the creation or modification of a file within the sandbox (e.g., registering successful exports such as `.csv`, `.pdf`, or `.png` outputs following specific actions).
    - In **application state transitions**, monitoring focus changes, dialog closures, or process launches via OS accessibility APIs.
    - In **UI interaction success**, verifying that a button, link, or menu item was activated and that the resulting accessibility tree or visual layout changed accordingly.
    - These measurable indicators serve as the **atomic verification layer** of the reward system, ensuring that each environment step corresponds to reproducible, auditable progress signals without requiring human intervention.
- To generalize beyond fixed rules, a **trainable reward model** Rϕ(ot,at,ot+1) can be introduced. This model is trained on **human-labeled or preference-ranked trajectories**, similar to the reward modeling stage in PPO-based RLHF pipelines. Once trained, Rϕ predicts scalar reward signals that approximate human preferences for unseen tasks or ambiguous states. It operates faster and more consistently than a generative LLM-as-a-Judge (which can be implemented as a Verifier Agent), while maintaining semantic fidelity to human supervision.
    
- The **three-tier reward hierarchy** thus becomes:
    
    1. **Rule-based rewards (preferred default):** deterministic, event-driven, and auditable (no reward hacking).
    2. **Learned, discriminative reward model (Rϕ):** generalizes human feedback for subtle, unstructured, or context-dependent goals where rules are insufficient.
    3. **Generative reward model (e.g., LLM-as-a-Judge):** invoked only when both rule-based detectors and Rϕ cannot confidently score outcomes (e.g., for semantic reasoning, style alignment, or multimodal understanding). This is similar to how [DeepSeek-R1](https://aman.ai/primers/ai/deepseek-R1) uses a generative reward model by feeding the ground-truth and model predictions into DeepSeek-V3 for judgment during the rejection sampling stage for reasoning data.
- This architecture ensures that the **primary training flow remains rule-grounded and verifiable**, while allowing smooth fallback to preference-aligned modeling when necessary. The hybrid setup—selectively combining rule-based rewards, learned reward estimation, and verifier agent intervention—balances **scalability, auditability, and semantic depth** across diverse computer-use tasks.
    
- During training, the **reward selection and routing process** is adaptive. When deterministic milestone detectors emit valid scores, they take precedence as the most reliable supervision. If the environment lacks such instrumentation, the learned model Rϕ dynamically provides substitute scalar feedback inferred from trajectory context. In the rare case that both mechanisms yield low confidence, the system escalates to the Verifier Agent for semantic adjudication. This cascading reward flow ensures the agent always receives a stable optimization signal—grounded when possible, inferred when necessary, and judged when ambiguity demands interpretive reasoning.
    

#### Milestone-Based Reward System

- Any **reward formulation**—whether deterministic, learned, or model-evaluated—can be decomposed into a sequence of **milestones or checkpoints** that represent measurable progress toward the task goal. Each milestone corresponds to a verifiable state transition, UI event, or observable change in the environment, providing interpretable signals even within complex or hierarchical workflows. In practice, a reward function can therefore be a **composite of multiple sources**: **rule-based rewards**, scalar predictions from a **learned, discriminative reward model**, or a **generative model** that is proficient at grading the task, such as an **LLM-as-a-Judge**.
    
- In general, **rule-based rewards** are preferred because they are **deterministic, easy to verify, and resistant to reward hacking**, consistent with the design principles demonstrated in the [DeepSeek-R1](https://arxiv.org/abs/2501.12948) framework by Gao et al. (2025). These rewards are derived from **concrete, environment-observable events**—such as file creation, DOM or AX tree changes, navigation completions, or dialog confirmations—and can be validated directly through structured logs and system hooks. Their reproducibility and transparency make them ideal for large-scale, self-contained policy optimization loops, where interpretability and auditability are crucial.
    
- In this system, the **rule-based layer** serves as the foundational signal generator for all common computer-use tasks. It captures events such as:
    
    - File downloads or artifact creation
    - Successful form submissions or dialog confirmations
    - UI transitions, window focus changes, or navigation completions
    - Text field population or data transfer between applications
    - Screenshot or state deltas indicating successful subgoal completion
        
    - These reward components directly populate the tuple (ot,at,rt,ot+1) used by the policy optimizer for learning stable, interpretable control policies. Each milestone event contributes either a discrete tick or a weighted scalar toward cumulative progress.
- However, not all task goals can be described exhaustively through deterministic rules. To extend coverage, the architecture includes a **learned reward model** Rϕ(ot,at,ot+1) trained specifically on **human preferences or ranked trajectories**.
    
    - This model generalizes beyond hand-engineered events to score **semantic correctness, contextual relevance, and user-aligned outcomes**.
    - Rϕ can be continuously fine-tuned as new preference data accumulates, adapting reward shaping dynamically to novel workflows or unseen UIs.
    - During training, the optimizer consumes a blended reward signal that can combine multiple sources:
        
        r̃ t=αr(rule)t+βRϕ(ot,at,ot+1)+γr(judge)t
        
        - where α,β,γ∈[0,1] represent trust weights for deterministic, learned, and model-evaluated components respectively, with α+β+γ=1.
- In cases where both rule-based detectors and the learned reward model fail to provide a confident or interpretable score, a **generative model (such as an LLM-as-a-Judge)** may be selectively invoked. This verifier acts as a high-capacity, _LLM-as-a-Judge_ module that semantically evaluates whether the observed trajectory satisfies implicit or fuzzy success criteria. Its role parallels that of a preference model but operates at runtime for difficult or open-ended cases.
    
- Scenarios where rule-based and model-based scoring may be insufficient—and thus require a Verifier Agent—include:
    
    - **Subjective or semantic correctness:** determining if a written summary or chart interpretation matches the instruction intent.
    - **Cross-context validation:** verifying that data copied from a spreadsheet was correctly inserted into a report or email draft.
    - **Goal inference under ambiguity:** tasks like “open the latest invoice,” where the target must be inferred dynamically.
    - **Complex recovery handling:** identifying whether the system has correctly recovered from an unintended dialog or misclick.
    - **Language or multimodal alignment:** verifying tone, structure, or layout across applications.
- The **reward system hierarchy** therefore consists of three complementary and optionally composable layers:
    
    1. **Rule-based rewards**: deterministic, verifiable, and fully auditable signals derived from concrete milestones (default and preferred).
        
    2. **Learned, discriminative reward model (Rϕ)**: trained on human preferences to generalize beyond explicit rules and produce scalar feedback for unstructured tasks.
        
    3. **Generative reward model (e.g., LLM-as-a-Judge)**: semantic fallback for nuanced, subjective, or multimodal evaluation where neither rules nor learned models suffice. This is similar to how [DeepSeek-R1](https://aman.ai/primers/ai/deepseek-R1) uses a generative reward model by feeding the ground-truth and model predictions into DeepSeek-V3 for judgment during the rejection sampling stage for reasoning data.
        
- Together, these layers enable **robust, explainable, and modular reward shaping**. Any reward function within the system can thus be expressed as a **milestone-weighted combination** of deterministic, learned, and interpretive components—ensuring scalability, transparency, and semantic alignment across all computer-use reinforcement learning setups.
    

##### Example Milestones by Task Category

1. **Web Navigation and Data Extraction**
    
    - **Milestone:** Target URL loaded successfully (`NavigationCompleted` event). _Reward:_ +0.25
    - **Milestone:** Element with specific role/name detected (e.g., “Reports Table” or “Dashboard Summary”). _Reward:_ +0.25
    - **Milestone:** Successful data scrape or DOM text retrieval logged. _Reward:_ +0.5
2. **Form Interaction**
    
    - **Milestone:** Input field focused and filled (text pattern matched). _Reward:_ +0.2
    - **Milestone:** Submit button clicked and confirmation dialog appears. _Reward:_ +0.3
    - **Milestone:** Success banner or confirmation element detected. _Reward:_ +0.5
3. **File Handling and Downloads**
    
    - **Milestone:** File creation event observed in `/Downloads`. _Reward:_ +1.0
    - **Milestone:** File hash or extension matches expectation (e.g., `.csv`, `.pdf`). _Reward:_ +0.5
    - **Milestone:** Directory updated without error. _Reward:_ +0.25
4. **Email or Document Workflows**
    
    - **Milestone:** Email editor loaded and populated with recipient and subject. _Reward:_ +0.25
    - **Milestone:** Attachment successfully added. _Reward:_ +0.5
    - **Milestone:** Message successfully sent (UI confirmation or state change). _Reward:_ +1.0
5. **System Configuration and Settings**
    
    - **Milestone:** Settings panel opened (window title match). _Reward:_ +0.25
    - **Milestone:** Checkbox or toggle successfully modified (UIA/AX event). _Reward:_ +0.25
    - **Milestone:** “Changes Saved” notification observed. _Reward:_ +0.5
6. **Search and Information Retrieval**
    
    - **Milestone:** Query field populated with correct term. _Reward:_ +0.25
    - **Milestone:** Search executed and result list rendered. _Reward:_ +0.5
    - **Milestone:** Target entry clicked or opened. _Reward:_ +0.5

#### Example Reward Function

- Each environment step returns a shaped reward based on concrete, verifiable milestones. Instead of relying on subjective evaluators, the reward function is composed of measurable subcomponents derived from observable state transitions, UI changes, and artifact events.
    
- At step t, the total reward is given by:
    
    rt=wnavr(nav)t+wUIr(UI)t+wformr(form)t+wfiler(file)t+wgoalr(goal)t
    
    - where each component represents a verifiable milestone type:
- r(nav)t: Navigation progress reward — triggered by measurable page transitions such as `NavigationCompleted` events, URL match, or window title change.
    
    r(nav)t=𝟙{urlt≠urlt−1}
    
- r(UI)t: UI element interaction reward — triggered when a UI control with a matching role or label is successfully targeted (e.g., a button click or field focus event).
    
    r(UI)t=𝟙{clicked(role,name)=expected(role,name)
    
- r(form)t: Form completion reward — triggered when an editable control is filled and validated (value non-empty, regex match, or field count).
    
    r(form)t=NfilledNexpected
    
- r(file)t: File-handling reward — derived from filesystem or artifact deltas (e.g., a new `.csv`, `.pdf`, or `.json` created).
    
    r(file)t=𝟙{∃f∈t:f.event=''created"}
    
- r(goal)t: Task completion reward — triggered by a high-level terminal condition, such as detection of success text, matched hash, or closed loop condition.
    
    r(goal)t=𝟙{goal_verified(ot)}
    
- The weights wnav,wUI,wform,wfile,wgoal balance short-term shaping with terminal rewards, typically normalized so that:
    

∑iwi=1{wgoal≥wfile≥wUI}

#### Example Instantiation

|**Component**|**Description**|**Weight**|**Range**|
|---|---|---|---|
|r(nav)t|Successful navigation|0.1|0,1|
|r(UI)t|Correct element interaction|0.2|0,1|
|r(form)t|Partial form completion|0.2|[0,1]|
|r(file)t|Artifact creation (e.g., download)|0.3|0,1|
|r(goal)t|Verified task completion|0.2|0,1|

- This formulation ensures **all reward components are physically measurable**—no human labels are required. Each event corresponds to structured data observable through CDP logs, accessibility APIs, or filesystem monitors, making it reproducible and auditable across training runs.

### Agent Training Pipeline

- A typical pipeline to train a web or computer-use agent might follow:
    
    1. Pre-train the model (e.g., a large language model) via supervised learning.
    2. Optionally fine-tune on domain-specific prompts (supervised fine-tuning, SFT).
    3. Collect human preference data (rankings of model responses).
    4. Choose alignment method:
        - **RLHF:** train reward model → use PPO (or other RL algorithm) to optimise policy.
        - **DPO:** directly fine-tune model on preference data (skipping RL loop).
    5. Launch agent into simulated environment (SingleTurnEnv, ToolEnv, MultiTurnEnv).
    6. Run RL policy optimisation in the environment: sample trajectories, estimate advantages/returns, update policy using PPO or variants.
    7. Periodically evaluate and filter trajectories, adjust reward shaping, fine-tune further for tool-use or long-horizon behaviours.
- By selecting algorithms appropriate for the interaction type (single turn vs. tool vs. multi-turn), one can tailor the training for efficiency, stability, and scalability.
    

## Environment Interaction Patterns for Agent Design

### Environment Design in Reinforcement Learning for Agents

- Modern RL environments for web and computer-use agents are designed to capture the diversity and complexity of real-world interactions while maintaining enough structure for stable learning. Unlike classical RL benchmarks (e.g., Atari or MuJoCo), these environments involve language, symbolic reasoning, tool use, and visual perception.
    
- They are not simply “games” or “control systems” but **interactive ecosystems** that test an agent’s ability to perceive context, reason over multi-step processes, and execute goal-directed actions.
    
- To support the training of increasingly capable language-based and multimodal agents, recent frameworks such as [AgentGym](https://arxiv.org/abs/2406.04151) by Xi et al. (2024) have introduced a unified taxonomy of environments, each corresponding to a particular _interaction modality_.
    
- At the highest level, these can be grouped into three archetypes:
    
    1. **Single-Turn Environments**, designed for one-shot problem solving and precision reasoning.
    2. **Tool-Use Environments**, optimized for integrating external functions, APIs, or computation tools.
    3. **Multi-Turn Sequential Environments**, which simulate complex, long-horizon workflows requiring memory, planning, and context adaptation.
- Each environment type not only changes how agents act but also how _rewards, policies, and credit assignment mechanisms_ must be designed to drive meaningful learning.
    

### Single-Turn Environments (SingleTurnEnv)

- **Single-turn environments** represent the simplest and most direct form of RL training. In this setup, each episode consists of a single interaction: the agent receives an input (prompt, question, or task description), produces one output (answer, code snippet, or solution), and immediately receives feedback.
    
- These environments are ideal for optimizing agents that must produce highly accurate outputs in one step—such as coding assistants, math solvers, or document completion systems.
    
- **Examples:**
    - Code completion and debugging tasks in _CodeRL_ ([CodeRL: Mastering Code Generation through RL](https://arxiv.org/abs/2207.01780) by Le et al., 2022).
    - Question-answering benchmarks like WebGPT ([WebGPT](https://arxiv.org/abs/2112.09332) by Nakano et al., 2022)), where the agent’s final response is scored based on correctness and citation quality.
- **Reward Structure:** Single-turn environments typically use _outcome-based rewards_ rather than step-wise feedback because there is only one output to evaluate. For example:
    
    - In a coding task, r=+1 if the code executes successfully, and r=0 otherwise.
    - In a factual QA task, r may represent an F1 score or BLEU score.
- Formally, the optimization objective reduces to:
    
    J(π)=𝔼x∼D,y∼π(⋅|x)[R(x,y)]
    
    - where R(x,y) is the final outcome reward.
- While simple, such environments serve as critical pretraining stages, allowing models to build domain accuracy before engaging in multi-step reasoning or tool-use.

### Tool-Use Environments (ToolEnv)

- **Tool-use environments** introduce an additional layer of reasoning: instead of solving a task in one step, the agent must decide when and how to invoke external tools. Tools may include:
    
    - API calls (e.g., search, translation, or computation),
    - external functions (e.g., symbolic calculators, Python interpreters), or
    - system-level commands (e.g., file access, browser manipulation).
- The core challenge is _tool orchestration_—learning when to rely on external computation versus internal reasoning. For instance, in a data retrieval task, the agent might issue an API query, parse results, and compose a natural-language summary.
    
- **Reward Structure**:
    - In ToolEnv, both _process-wise_ and _outcome-based_ rewards are valuable:
        
        - _Step-wise rewards_ can score the accuracy or efficiency of each tool invocation (e.g., correct API parameters or valid JSON structure).
        - _Outcome-based rewards_ measure task completion or user satisfaction.
    - The combined reward signal is often expressed as:
        
        Rt=αrprocess+(1−α)routcome,
        
        - where α controls the balance between short-term and final goal feedback.
- **Algorithmic Approaches**: Because the action space now includes function arguments and results, methods like policy gradient with structured action representations, hierarchical RL, or model-based planning (e.g., MCTS as in [Agent Q](https://arxiv.org/abs/2408.07199) by Putta et al., 2024) become necessary.
    
- [Tool Learning with Foundation Models](https://doi.org/10.1145/3704435) by Qin et al. (2024) provides a comprehensive survey of how foundation models learn to invoke external tools to augment their reasoning capabilities.

### Multi-Turn Sequential Environments (MultiTurnEnv)

- **Multi-turn environments** simulate complex, multi-step workflows where each decision influences future context. These environments are designed for agents that need to plan, adapt, and maintain consistency across many turns of interaction.
    
- **Examples:**
    
    - Web navigation agents such as [OpenWebVoyager](https://arxiv.org/abs/2410.19609) by He et al. (2024), where the agent browses, clicks, and fills forms over multiple steps.
    - Software operation tasks like system configuration, spreadsheet editing, or email management.
    - Interactive tutoring and dialogue planning systems.
- **Reward Structure:**
    - In MultiTurnEnv setups, pure outcome-based rewards (success/failure) can cause _credit assignment problems_ because the agent receives feedback only after many steps. To address this, researchers combine **process-wise rewards**—for subgoal completion, error reduction, or partial correctness—with **final outcome rewards**.
        
    - Formally, the expected return in such environments can be represented as:
        
        J(π)=𝔼[∑t=1Tγt(rprocesst+λ,routcomeT)]
        
        - where λ balances intermediate and terminal objectives.
    - In OpenWebVoyager, for example, each sub-action (like opening the correct link) contributes partial reward, guiding the agent toward long-term success while preventing divergence from optimal sequences.
        
- **Learning Dynamics:** Training in MultiTurnEnv requires:
    
    - Long-horizon credit assignment via temporal-difference learning or advantage estimation.
    - Hierarchical RL for decomposing tasks into sub-policies.
    - Trajectory filtering and reward shaping to combat sparse or noisy signals.

### Designing Rewards for Complex Agent Environments

- Reward engineering is arguably the most critical part of environment design. Different environment types benefit from distinct reward strategies:

|**Environment Type**|**Reward Type**|**Typical Signal**|**Optimization Goal**|
|---|---|---|---|
|SingleTurnEnv|Outcome-based|Correctness, BLEU/F1 score|Precision and factual accuracy|
|ToolEnv|Hybrid (step-wise + outcome)|Tool correctness, API success|Functional reasoning, tool reliability|
|MultiTurnEnv|Step-wise + delayed outcome|Subgoal completion, navigation success|Long-horizon planning, autonomy|

- Balancing process-wise and outcome-based rewards ensures that agents receive _dense feedback for learning efficiency_ while still optimizing toward _global objectives_ like success rate or user satisfaction.

### Implications for Agent Design and Evaluation

- Each environment type imposes unique requirements on model architecture, reward shaping, and evaluation metrics.
    
    1. **SingleTurnEnv** favors compact policies and fast evaluation loops, suitable for smaller RL batches or DPO-based optimization.
    2. **ToolEnv** requires compositional reasoning and structured memory to maintain tool-call histories and argument dependencies.
    3. **MultiTurnEnv** demands long-context modeling, world-state tracking, and temporal credit assignment across potentially hundreds of steps.
- Evaluation metrics vary accordingly:
    
    - _Single-turn_: Accuracy, F1, pass rate.
    - _Tool-use_: Tool-call correctness, latency, success ratio.
    - _Multi-turn_: Task completion rate, cumulative reward, consistency, and planning efficiency.
- When integrated properly, these environment classes form a **curriculum** for RL-based agent development: agents begin with static, outcome-driven reasoning (SingleTurnEnv), progress to dynamic, tool-integrated reasoning (ToolEnv), and culminate in fully autonomous multi-turn reasoning (MultiTurnEnv).
    

### Comparative Analysis

- Environment design is the foundation on which modern RL agents learn to generalize and act. The interplay between **interaction modality**, **reward granularity**, and **algorithmic strategy** determines not only how fast an agent learns but also what kinds of intelligence it develops.
    
    - Single-turn environments teach _accuracy_.
    - Tool-use environments teach _functional reasoning_.
    - Multi-turn environments teach _autonomy and adaptability_.
- Together, they form a progression of increasing sophistication—mirroring the cognitive layers of reasoning, planning, and execution. RL algorithms like PPO and DPO serve as the connective tissue between these layers, transforming static pretrained models into active, evolving agents capable of navigating and operating within real digital ecosystems.
    

## Reward Modeling

### The Role of Reward Modeling

- Reward modeling lies at the heart of RL systems for language, web, and computer-use agents. In traditional RL, the reward function is hand-crafted to quantify success—for example, the score in a game or the distance to a goal. In contrast, modern LLM-based agents operate in open-ended environments where the notion of “correctness” or “helpfulness” is inherently subjective and context-dependent.
    
- To handle this, reward models (RMs) are trained to approximate human judgment. Instead of manually defining numerical rewards, the system learns a function rϕ(x,y) that predicts the quality of an agent’s output y for a given input x. These RMs are usually fine-tuned on preference datasets where human annotators rank outputs from best to worst.
    
- Formally, given a dataset of comparisons D=(xi,y+i,y−i), the reward model is trained to maximize:
    
    RM=−𝔼(x,y+,y−)∼D[logσ(rϕ(x,y+)−rϕ(x,y−))]
    
    - where σ is the logistic function, and rϕ outputs a scalar reward. The resulting model can then guide PPO updates, Direct Preference Optimization (DPO), or other RL pipelines.
- Reward modeling thus replaces explicit rule-based objectives with _learned evaluators_—a fundamental shift that enables agents to align with nuanced human preferences across web, reasoning, and tool-use tasks.
    
- [Agent Learning via Early Experience](https://arxiv.org/abs/2510.08558) by Zhang et al. (2025)) states that in practice, reward signals can be complemented by reward-free, language-native supervision gathered before RL—so the policy starts “aligned to the environment” even without verifiable rewards. Two pre-RL objectives from early, agent-generated interaction data are especially useful: an implicit world-modeling loss that predicts next states given state–action pairs, and a self-reflection loss that learns to compare expert vs. non-expert actions in natural language. Concretely:
    
    LIWM(θ)=−∑(si,aji,sji)∈rolloutlogpθ(sji,∣∣,si,aji),LSR(θ)=−∑i∑j=1Klogpθ(cji,∣∣,si,;aji,;ai,;si+1,;sji),
    
    - which warm-start policies and reduce distribution shift ahead of PPO/GRPO or DPO, improving sample efficiency in web and tool-use settings.
        
    - The following figure shows an overview of the two early experience approaches. Implicit world modeling (left) augments expert trajectories with alternative actions and predicted next states, training the policy to internalize transition dynamics before deployment. Self-reflection (right) augments expert actions with self-generated explanations c1, training the policy to reason about and revise its own decisions. Both methods use alternative actions proposed by the initial policy (LLM). The number of alternatives K is a hyperparameter; for brevity, only one is illustrated.
        
    
    ![](https://aman.ai/primers/ai/assets/RL-for-agents/EarlyExperience2.jpg)
    

### Process-Wise and Outcome-Based Reward Integration

- When training agents in realistic, multi-step environments, reward signals can be categorized as **process-wise (step-wise)** or **outcome-based**. Both serve complementary roles:
    
    1. **Outcome-Based Rewards:**
        - These are terminal signals received once the task is complete—such as a success flag, accuracy score, or human satisfaction rating.
        - For instance, in a booking agent, a positive reward may be given only when the reservation is successfully completed.
    2. **Process-Wise (Step-Wise) Rewards:**
        - These provide intermediate feedback after each step or subgoal, rewarding partial correctness, progress, or efficiency.
        - In web navigation, an agent might receive a small positive reward for clicking the correct button or locating relevant text, even before reaching the final goal.
- The challenge is balancing the two. Purely outcome-based training can lead to _sparse reward problems_, while purely process-based training risks _overfitting local heuristics_ that do not generalize.
    
- A common hybrid formulation is:
    
    rt=α,rprocesst+(1−α),δt=T,routcomeT
    
    - where α∈[0,1] controls the tradeoff between intermediate shaping and final goal alignment.
- In practical web-agent training, hybrid reward models may leverage both:
    
    - **Synthetic process feedback** (automated evaluators for substeps),
    - **Human outcome feedback** (ranking complete trajectories).
- A scalable way to create dense, shaped feedback is to synthesize experience with a reasoning-based experience model that produces consistent next states and vectorized, unified feedback signals in a textual state space. This enables closed-loop RL without expensive real-environment rollouts and supports curriculum generation that targets the current policy’s weaknesses; empirically it yields >30% gains on non-RL-ready tasks like WebArena and can match PPO/GRPO using only synthetic interactions ([Scaling Agent Learning via Experience Synthesis](https://arxiv.org/abs/2511.03773) by Chen et al. (2025)).
    

### Tool-Augmented Reward Modeling (TARM)

- [Tool-Augmented Reward Modeling (Themis)](https://arxiv.org/abs/2310.01045) by Li et al. (2024) proposes Tool-Augmented Reward Modeling (TARM) (also called Tool-Integrated Reward Modeling (TIRM)), which represents a significant evolution in RL for agents that operate within complex, tool-augmented environments. TARM integrates external computational and retrieval tools into the reward generation process itself. Instead of merely training language models to use tools during inference, TIRM embeds tool engagement as part of the reward model’s reasoning and supervision pipeline.
    
- This approach extends the conventional Reinforcement Learning from Human Feedback (RLHF) paradigm—used in models such as [InstructGPT](https://arxiv.org/abs/2203.02155) by Ouyang et al. (2022)—by introducing **tool-augmented reasoning traces** and **context-sensitive reward estimation**, enabling more accurate alignment between model outputs and human evaluators’ expectations.
    
- Put simply, tool-Integrated Reward Modeling advances RLHF by embedding reasoning transparency, external computation, and factual grounding directly into the reward modeling process. Through supervised fine-tuning on tool-augmented datasets and RL on process- and outcome-based signals, these models redefine how reward functions are constructed for intelligent agents. The resulting agents not only learn to act effectively but also to _evaluate_ their own reasoning with access to external world models—laying the foundation for trustworthy, explainable, and verifiable AI systems.
    
- Reward-free early experience, proposed in [Agent Learning via Early Experience](https://arxiv.org/abs/2510.08558) by Zhang et al. (2025), can seed TARM and RLHF alike: implicit world modeling grounds the policy in environment dynamics, while self-reflection generates rationale-style preferences that complement pairwise comparisons used by reward models—providing a bridge from imitation/preference learning to full RL.
    

#### Motivation and Background

- Traditional reward models in RLHF are trained using paired preference data, where a scalar reward is assigned based on human judgments. These models often struggle with factual reasoning, arithmetic operations, and real-world lookups due to their reliance on static, in-model knowledge representations ([Christiano et al., 2017](https://proceedings.neurips.cc/paper/2017/hash/d5e2c0adad503c91f91df240d0cd4e49-Abstract.html)). Tool-Integrated Reward Models mitigate this by allowing the reward model itself to call APIs, calculators, code interpreters, or search engines during evaluation.
    
- Themis demonstrated that augmenting reward models with tools increased factual accuracy and truthfulness on benchmarks like TruthfulQA by 7.3% over large baselines such as Gopher 280B, while achieving a 17.7% average improvement in preference ranking accuracy across tasks.
    

#### Structure and Workflow of Tool-Augmented Reward Models

- The tool-integrated reward modeling process can be decomposed into sequential reasoning stages—each enhancing the model’s interpretability and precision in assigning rewards:
    
    1. **Thought**: The model assesses whether external information is required and determines which tool to invoke.
    2. **Action**: The model generates an API call with specified parameters.
    3. **Observation**: The system retrieves and processes tool outputs.
    4. **Rationale**: The model integrates the external information into a reasoning chain, constructing an interpretable trace of decision-making.
    5. **Reward Generation**: A scalar reward is computed from the aggregated reasoning trace.
- Formally, the total reasoning trajectory is denoted as:
    

c1:T=(a1,o1,…,aT,oT,sT)

- … and the scalar reward is defined as:
    
    rθ(x,y,c1:T)
    
    - where x is the input, y is the model’s output, and c1:T represents the full reasoning and observation history.
- The total loss function combines pairwise ranking and autoregressive modeling losses:
    
    Ltotal=LRM+α∑t=1T(Ltool(t)+βLobs(t))+ωLrat
    
    - where LRM corresponds to the pairwise ranking loss from preference modeling, Ltool supervises tool invocation accuracy, Lobs captures fidelity to observed results, and Lrat trains the model to generate coherent rationales.
- The following figure ([source](https://arxiv.org/abs/2310.01045)) shows illustrates the pipeline of (a) Vanilla reward models (RMs); (b) Tool-augmented RMs, namely Themis; (c) RL via proximal policy optimization (PPO) on above RMs; (d) Examples of single or multiple tool use process in the proposed approach.
    

![](https://aman.ai/primers/ai/assets/RL-for-agents/Themis.jpg)

- Per [Scaling Agent Learning via Experience Synthesis](https://arxiv.org/abs/2511.03773) by Chen et al. (2025), when paired with synthetic experience generation, tool-augmented evaluators can operate at scale with consistent, informative feedback, while curriculum generation focuses on high-entropy tasks that maximize learning signal—closing the loop between reward modeling and data generation in RL training.

#### Role of Supervised Fine-Tuning and Reinforcement Learning

- Themis—and, more broadly, TIRM—relies on a **hybrid SFT + RL training approach**.
    
    - **SFT Stage**: The reward model learns to imitate tool usage traces from curated datasets (e.g., the [TARA dataset](https://github.com/ernie-research/Tool-Augmented-Reward-Model)). These traces include natural-language thoughts, API calls, and tool results generated via multi-agent interactions between LLMs and simulated human labelers.
        
    - **RL Stage**: Once pre-trained, the reward model is further optimized via RL objectives like Proximal Policy Optimization (PPO) ([Schulman et al., 2017](https://arxiv.org/abs/1707.06347)). The model refines its reward predictions using outcome-based feedback, achieving stable convergence even under high variance tool-call trajectories.
        
- This two-stage setup enables **process-based reward shaping**, in which partial rewards are granted for intermediate reasoning correctness (process rewards), and **outcome-based rewards** for overall task success. This balance is critical when agents operate in environments requiring both reasoning depth and correct final results.
    
- Reward-free early experience provides a natural pretraining curriculum—first fitting LIWM to learn dynamics, then LSR to internalize preference signals—before introducing PPO/GRPO or DPO on either real or synthetic rollouts (cf. [Agent Learning via Early Experience](https://arxiv.org/abs/2510.08558) by Zhang et al. (2025); [Scaling Agent Learning via Experience Synthesis](https://arxiv.org/abs/2511.03773) by Chen et al. (2025)).
    

#### The Tool-Augmented Reward Dataset (TARA)

- A key component of TIRM research is the creation of datasets that reflect real-world reasoning and tool usage patterns. The [TARA dataset](https://github.com/ernie-research/Tool-Augmented-Reward-Model) contains over 15,000 instances combining human preferences with explicit tool-invocation traces across seven tool categories, including search, translation, weather, calculator, and code execution.
    
- The following figure ([source](https://arxiv.org/abs/2310.01045)) shows the data collection pipeline for TARA, depicting human-LLM interaction, tool invocation, and rationale generation. It the four-step process: (1) Question-answer collection, (2) ToolBank construction, (3) Tool invocation via multi-agent simulation, and (4) Filtering for data integrity.
    

![](https://aman.ai/primers/ai/assets/RL-for-agents/TARA_Pipeline.jpg)

#### Empirical Results and Observations

- Experiments show that Themis enhances both **single-tool** and **multi-tool** scenarios. For example:
    
    - Accuracy improved by +19.2% in single-tool and +17.7% in mixed-tool setups.
    - Perfect accuracy (100%) was achieved in calendar and weather reasoning tasks.
    - Models learned when and whether to call tools autonomously—a form of learned tool invocation policy.
    - The observation and rationale components contributed significantly to reward accuracy, proving that **process supervision** is critical to model interpretability and consistency.
- Further, when integrated into an RLHF pipeline (referred to as RLTAF: Reinforcement Learning from Tool-Augmented Feedback), Themis-trained models achieved a 32% higher human preference win rate compared to vanilla RMs, highlighting its ability to generate more trustworthy and factual responses.
    
- Complementarily, [Scaling Agent Learning via Experience Synthesis](https://arxiv.org/abs/2511.03773) by Chen et al. (2025) proposes scaling RL with synthetic rollouts generated by a reasoning experience model, which yields substantial downstream gains and lowers on-environment data needs; e.g., DreamGym reports >30% improvements on WebArena and policy parity with PPO/GRPO using only synthetic interactions, after which real-environment fine-tuning brings additional gains.
    
- The following figure illustrates an overview of the proposed DreamGym agent training framework. Given a set of seed tasks, a reasoning-based experience model interacts with the agent to generate informative, diverse tasks and trajectories for RL training. At each step, the agent takes actions based on its current state and receives next states and reward signals derived by the experience model through CoT reasoning based on both interaction history and top-k similar experiences from an active replay buffer. To expose the agent to increasingly informative scenarios, tasks with high reward entropy are proposed by the curriculum task generator for future training. With this unified design, DreamGym addresses both task and reward sparsity while enabling scalable RL with diverse and curriculum-driven environments.
    

![](https://aman.ai/primers/ai/assets/RL-for-agents/DreamGym2.jpg)

#### Connection to Reinforcement Learning for Agents

- Tool-integrated reward modeling bridges the gap between **tool-augmented reasoning** and **agentic RL**. By enabling the reward function itself to utilize external resources, agents trained under TIRM learn a deeper mapping between reasoning actions and value estimation. This structure is directly applicable to RL-driven computer-use agents, where both **process-level** (step-wise) and **outcome-based** (goal completion) rewards must be optimized.
    
- In this framework, process-based rewards correspond to accurate intermediate reasoning and correct tool usage, while outcome-based rewards correspond to successful task completion. The combined signal provides agents with fine-grained credit assignment, improving learning efficiency and interpretability in web-based or API-integrated environments.
    
- Per [Scaling Agent Learning via Experience Synthesis](https://arxiv.org/abs/2511.03773) by Chen et al. (2025), when training in synthetic environments, policy improvements can provably transfer to the real environment under standard trust-region updates. Writing the real MDP as =(S,A,P,R,γ) and the synthetic one as ̃ =(S,A,P̃ ,R̃ ,γ) with bounded reward and transition errors εR,εP, a KL-bounded update from π→π′ (as in PPO/GRPO) yields a lower bound of the form:
    
    J(π′)−J(π)≥11−γ,𝔼s∼d̃ π,a∼π′[A̃ π(s,a)]−KL trust-region penalty(per-state KL radius)−2(εR1−γ+2γRmax(1−γ)2εP)experience-model error
    
    - … so synthetic surrogate gains exceeding these penalties guarantee real-environment improvement.

### Feedback Alignment and Human Preference Modeling

- Reward models provide scalar supervision, but alignment requires _structured feedback_. Human evaluators often give comparative, categorical, or qualitative feedback (e.g., “response A is clearer, but response B is more complete”).
    
- To convert such structured feedback into training signals, systems employ **preference aggregation** methods such as:
    
    - _Bradley–Terry models_ to infer pairwise preference probabilities.
    - _Elo-style scoring_ to maintain global quality rankings across responses.
    - _Bayesian aggregation_ for uncertain or noisy feedback.
- In advanced systems like [Large Language Models Can Self-improve at Web Agent Tasks](https://arxiv.org/abs/2405.20309) by Patel et al. (2024), self-feedback mechanisms replace human labeling. The agent critiques its own trajectories using LLM-based evaluators, ranking which paths yielded the best progress and then re-finetuning on its own top-performing examples.
    
- This method creates a **feedback alignment loop**, where models not only learn from human signals but also gradually calibrate their own evaluators.
    

### Multi-Objective Reward Modeling

- As agents evolve to handle multi-modal and multi-task objectives—such as reasoning, retrieval, and tool orchestration—single scalar reward functions become insufficient.
- Instead, **multi-objective reward modeling (MORM)** decomposes total reward into several components:
    
    rt=∑k=1Kwk,r(k)t
    
    - where each r(k)t corresponds to a distinct objective (e.g., factual accuracy, efficiency, safety, fluency), and wk are learned or manually tuned weights.
- This decomposition enables flexible tradeoffs—for example, prioritizing accuracy over verbosity or reliability over speed. In web and software agents, multi-objective RMs can encode:
    
    - Functional correctness (execution success),
    - Temporal efficiency (fewer steps or tool calls),
    - Adherence to user goals (alignment quality),
    - Safety and compliance (filtered language use).
- Combining these objectives helps agents develop a balanced understanding of what constitutes “good behavior” in dynamic and human-centric environments.

### Evaluation Frameworks for RL-Based Agents

- Evaluating agents trained through RL requires going beyond static benchmarks. Instead of only measuring final success, modern frameworks evaluate _trajectory quality, interpretability, and generalization_.

#### Key Evaluation Metrics Include

- **Success Rate:** Fraction of episodes where the agent achieves its goal (e.g., booking completed, question answered).
- **Cumulative Reward:** Sum of step-wise rewards, indicating the efficiency of action selection.
- **Action Accuracy:** Proportion of correct API or tool calls.
- **Trajectory Efficiency:** Number of steps or actions required to reach completion.
- **Human Preference Score:** Alignment with human judgment over multiple outputs.
- **Robustness:** Performance under perturbed or unseen web environments.
    
- Frameworks such as WebArena, Mind2Web, and AgentBench (as catalogued in [AgentGym](https://arxiv.org/abs/2406.04151) by Xi et al., 2024) provide unified benchmarks with standardized reward metrics and simulator APIs for reproducible agent training.

### Takeaways

- Reward modeling and feedback alignment form the core of how RL agents evolve from static predictors into _adaptive decision-makers_. The design of these mechanisms determines whether agents learn to pursue shallow, short-term signals or to internalize long-term, value-aligned behavior.
    
    - **Outcome-based rewards** ensure goal fidelity but suffer from sparsity.
    - **Process-wise rewards** provide dense guidance and interpretability.
    - **Tool-augmented reward models** enhance factual grounding and transparency.
    - **Human and self-generated feedback** create continuous learning loops.
    - **Multi-objective reward modeling** allows flexible alignment across multiple competing priorities.
- Together, these innovations define the modern ecosystem of RL-based agentic training—where the agent not only _acts_ in its environment but also _learns how to evaluate its own progress_.
    

## Search-Based Reinforcement Learning, Monte Carlo Tree Search (MCTS), and Exploration Strategies in Multi-Step Agents

### Motivation: Exploration vs. Exploitation in Complex Agentic Systems

- In RL, agents must navigate the fundamental trade-off between **exploration**—trying new actions to discover better strategies—and **exploitation**—using known information to maximize immediate reward.
    
- For simple environments (like tabular Q-learning), this trade-off can be controlled by ϵ-greedy or softmax policies. However, for web and computer-use agents operating in open-ended, high-dimensional spaces—such as browsing dynamic web pages, calling APIs, or managing multi-turn dialogues—naive exploration is computationally infeasible and unsafe.
    
- Thus, modern agentic RL systems combine _search-based exploration_ with _learned policy optimization_, blending symbolic planning with neural policy priors. This hybrid paradigm is exemplified by recent works like [Agent Q: Efficient Online Adaptation via Monte Carlo Tree Search](https://arxiv.org/abs/2408.07199) by Putta et al. (2024) and [OpenWebVoyager](https://arxiv.org/abs/2410.19609) by He et al. (2024), both of which adapt classic search strategies (like MCTS) for reasoning-driven web environments.
    
- Complementary to these, [Agent Learning via Early Experience](https://arxiv.org/abs/2510.08558) by Zhang et al. (2025) shows that exploration itself can begin _before_ any reward modeling, by leveraging self-reflective rollouts and implicit world modeling to pretrain a policy that already encodes structured exploration biases. Similarly, [Scaling Agent Learning via Experience Synthesis](https://arxiv.org/abs/2511.03773) by Chen et al. (2025) formalizes a scalable simulation framework—**DreamGym**—that generates synthetic exploratory rollouts under theoretical guarantees of policy improvement transfer to real environments.
    
- The following figure shows the _Agent Q architecture_, demonstrating how an agent integrates Monte Carlo Tree Search (MCTS) with an internal policy model to efficiently explore and adapt to dynamic environments.
    

![](https://aman.ai/primers/ai/assets/RL-for-agents/AgentQ.jpg)

- The following figure illustrates that Agent Q is provided the following input format to the Agent, consisting of the system prompt, execution history, the current observation as a DOM representation, and the user query containing the goal. We divide our Agent output format into an overall step-by-step plan, thought, a command, and a status code.

![](https://aman.ai/primers/ai/assets/RL-for-agents/AgentQ1.jpg)

### Monte Carlo Tree Search (MCTS) in RL-Based Agents

- **Monte Carlo Tree Search (MCTS)** is a planning algorithm that estimates the value of actions through simulation. Each node in the search tree represents a state, and edges represent actions. During training, the agent builds a partial search tree by simulating action sequences, updating node values using empirical rollouts.
    
- At each decision step, MCTS performs four core operations:
    
    1. **Selection:** Traverse the current tree from the root to a leaf, selecting child nodes using the _Upper Confidence Bound_ (UCB) rule:
        
        at=argmaxa[Q(st,a)+clnN(st)1+N(st,a)‾‾‾‾‾‾‾‾‾‾‾‾√]
        
        - where Q(st,a) is the estimated action value, N(st,a) the visit count, and c a confidence constant.
    2. **Expansion:** Add one or more new child nodes to the tree.
        
    3. **Simulation:** Run a rollout (either with a learned policy or random actions) to estimate the outcome.
        
    4. **Backpropagation:** Update Q(st,a) values along the traversed path with the observed return.
        
- This method balances exploration and exploitation dynamically—favoring actions with high potential but uncertain estimates.
    
- In the context of LLM-based web agents, MCTS is adapted to explore _semantic_ and _structural_ decision spaces rather than numeric ones. Each node can represent:
    
    - A browser state (DOM snapshot, active page).
    - A reasoning context (prompt, plan, partial output).
    - A tool invocation (function call, API parameterization).
- MCTS then simulates different reasoning or action trajectories, evaluates their predicted rewards (using a reward model or preference score), and backpropagates this information to refine the policy.
    
- Recent approaches such as [Scaling Agent Learning via Experience Synthesis](https://arxiv.org/abs/2511.03773) by Chen et al. (2025) extend this principle by introducing a **reasoning-based experience model** that performs analogous “tree search” operations within a learned world model—sampling synthetic trajectories that approximate MCTS rollouts without direct environment interaction, thereby dramatically improving sample efficiency.
    

### Neural-Guided Search: Policy Priors and Value Models

- In environments too large for exhaustive search, modern agents employ **neural-guided search**—a synergy between _planning algorithms_ and _deep models_. Here, the policy model πθ(a∣s) provides prior probabilities for which actions to explore first, and the value model Vθ(s) predicts the expected return from each state. These models drastically reduce the branching factor and enable more efficient exploration.
    
- This framework mirrors the principles that powered **AlphaGo** ([Mastering the game of Go with deep neural networks and tree search](https://www.nature.com/articles/nature16961) by Silver et al., 2016), but applied to _symbolic and text-based tasks_ instead of games.
    
- Formally, the modified UCB rule becomes:
    
    U(s,a)=Q(s,a)+cpuctP(a|s)N(s)‾‾‾‾√1+N(s,a)
    
    - where P(a∣s) is the prior probability from the policy model. This ensures that exploration is guided by learned likelihoods, not uniform randomness.
- In [Agent Q](https://arxiv.org/abs/2408.07199) by Putta et al. (2024), this concept is applied to **online adaptation**: the agent uses MCTS for planning while simultaneously updating its local policy parameters via gradient descent, achieving a form of continual self-improvement.
    
- Early Experience pretraining complements neural-guided search by shaping the priors P(a∣s) and values V(s) before any explicit MCTS integration. By learning predictive transitions and reflective rationales ([Agent Learning via Early Experience](https://arxiv.org/abs/2510.08558) by Zhang et al., 2025), the agent begins search from a semantically meaningful latent space rather than random initialization—reducing both exploration cost and tree-depth requirements.
    

### Integration of Search with Reinforcement Learning and Fine-Tuning

- Search algorithms such as MCTS can be integrated with RL training in three primary ways:
    
    1. **Search as Pretraining:** Generate high-quality trajectories via MCTS and use them for supervised fine-tuning (similar to imitation learning).
        
    2. **Search as Online Exploration:** Use MCTS during training to propose promising action sequences; the policy learns to imitate successful trajectories while exploring uncertain branches.
        
    3. **Search as Evaluation:** Use MCTS only at inference to refine action selection, keeping policy updates purely gradient-based.
        
- In [Agent Q](https://arxiv.org/abs/2408.07199), this second mode—_online search and adaptation_—proved especially effective, enabling agents to generalize across unseen tasks without explicit retraining.
    
- DreamGym’s synthetic environment model provides a complementary fourth paradigm: **Search via Experience Synthesis.** Here, simulated rollouts within a learned reasoning environment substitute for explicit tree expansion, allowing policies to update from a massive, low-cost replay buffer of synthetic “search traces.” This merges the sample efficiency of model-based RL with the decision quality of tree search ([Scaling Agent Learning via Experience Synthesis](https://arxiv.org/abs/2511.03773) by Chen et al., 2025).
    

### Process-Wise Reward Shaping in Search-Based RL

- A key enhancement in modern search-based RL pipelines is the introduction of **process-wise reward shaping** to complement sparse terminal rewards. In multi-turn or tool-using agents, MCTS nodes can be augmented with intermediate reward estimates derived from:
    
    - Successful API or function calls,
    - Reduced error rates or failed action counts,
    - Improved subgoal completion,
    - Positive sentiment or human approval scores.
- This transforms the reward signal from a binary success/failure into a smooth landscape that supports _credit assignment_ across deep search trees.
    
- The adjusted value propagation for a trajectory of length T becomes:
    
    Q(st,at)←(1−η)Q(st,at)+η∑k=tTγk−trprocessk
    
    - where rprocessk captures per-step quality signals. This formulation allows the agent to refine sub-policies even when full-task success has not yet been achieved—vital for real-world agents that must learn under incomplete supervision.

### Integration of Search with Reinforcement Learning and Fine-Tuning

- Search algorithms such as MCTS can be integrated with RL training in three primary ways:
    
    1. **Search as Pretraining:** Generate high-quality trajectories via MCTS and use them for supervised fine-tuning (similar to imitation learning).
        
    2. **Search as Online Exploration:** Use MCTS during training to propose promising action sequences; the policy learns to imitate successful trajectories while exploring uncertain branches.
        
    3. **Search as Evaluation:** Use MCTS only at inference to refine action selection, keeping policy updates purely gradient-based.
        
- In [Agent Q](https://arxiv.org/abs/2408.07199), this second mode—_online search and adaptation_—proved especially effective, enabling agents to generalize across unseen tasks without explicit retraining.
    

### Exploration Strategies in Web and Computer-Use Environments

- In high-dimensional digital environments, exploration must be structured and interpretable. Several strategies are commonly used:
    
    - **Entropy-Regularized Exploration:** Adding an entropy term to the objective encourages diversity in action selection:
        
        J(π)=𝔼π[∑t(rt+β,H(π(⋅|st)))]
        
        - where H(π) is policy entropy and β controls exploration intensity.
    - **Curiosity-Driven Exploration:** Agents are rewarded for discovering novel or unpredictable states using intrinsic motivation models such as [Random Network Distillation](https://arxiv.org/abs/1810.12894) by Burda et al. (2019).
        
    - **Goal-Conditioned Exploration:** Particularly in web tasks, exploration can be constrained by semantic or user-defined goals, ensuring the agent does not perform irrelevant actions.
        
    - **State Abstraction and Clustering:** Complex environments can be segmented into abstract state representations (e.g., webpage templates or tool invocation graphs), allowing for hierarchical exploration.
        
- These approaches are especially effective in _MultiTurnEnv_ scenarios where the state space expands combinatorially with each decision.
    

### Planning and Value Composition Across Multiple Environments

- The integration of search-based reasoning with learned RL policies allows agents to _compose behaviors across environment types_. For instance:
    
    - In **SingleTurnEnv**, search helps refine output reasoning (e.g., multi-step chain-of-thought validation).
    - In **ToolEnv**, it aids in selecting optimal tool invocation sequences.
    - In **MultiTurnEnv**, it supports long-horizon planning and dynamic replanning when goals change.
- The combined expected return from multi-environment value composition can be expressed as:
    
    Jglobal=∑e∈ωe,𝔼πe[∑tγtr(e)t]
    
    - where  denotes environment types (SingleTurn, Tool, MultiTurn) and ωe are task-specific weights.
- This hierarchical structure aligns exploration depth with task complexity, improving sample efficiency and stability.
    

### Summary and Outlook

- Search-based RL represents a crucial step in bridging **symbolic planning** and **neural policy learning** for complex, real-world agents.
    
    - **Monte Carlo Tree Search (MCTS)** provides structured exploration with statistical guarantees.
    - **Neural-guided search** integrates learned policy and value priors for scalability.
    - **Process-wise rewards** smooth sparse reward landscapes, enabling deeper credit assignment.
    - **Hybrid search–RL systems** enable online adaptation and continual learning.
- As web and computer-use agents evolve, search-based strategies are increasingly viewed not as add-ons but as _core cognitive modules_, empowering agents to deliberate, simulate, and refine decisions—much like human reasoning.
    

## Memory, World Modeling, and Long-Horizon Credit Assignment

### The Need for Memory and Temporal Reasoning

- Unlike short episodic tasks, web and computer-use agents must operate over long time horizons—completing multi-step workflows, navigating dynamic web pages, and managing context-dependent subtasks that span hundreds of actions. These tasks demand **temporal coherence**, **state persistence**, and **contextual reasoning**, capabilities that exceed what standard Markovian RL formulations provide.
    
- Traditional RL assumes the **Markov Decision Process (MDP)** property:
    
    P(st+1|st,at,st−1,at−1,...)=P(st+1|st,at)
    
    - which implies that the current state st encapsulates all relevant information for decision-making. In practice, however, agents must handle **Partially Observable MDPs (POMDPs)**, where the environment’s full state is not directly visible—such as hidden system states, incomplete browser information, or unobserved user intentions.
- This motivates integrating **memory mechanisms**—either through explicit world models, neural state trackers, or structured external memories—that allow agents to reason over _latent histories_.
    
- Recent pretraining approaches such as **Early Experience** ([Agent Learning via Early Experience](https://arxiv.org/abs/2510.08558) by Zhang et al., 2025) implicitly address this by building _internal temporal memory_ even before explicit RL fine-tuning. Through predictive next-state modeling and reflective rationalization losses, the agent internalizes time-linked dependencies (e.g., how tool outcomes evolve or how plans fail) purely from self-supervised rollouts—forming an implicit memory backbone that later stabilizes long-horizon RL.
    

### Explicit vs. Implicit Memory Architectures

- Modern agentic systems implement memory in two major ways—**explicit symbolic memory** and **implicit neural memory**—each optimized for different environment dynamics.
    
    1. **Explicit Symbolic Memory**:
        
        - Stores structured facts and environment states (e.g., webpage structure, task progress, prior tool outputs).
        - Can be queried and updated through symbolic operations or APIs.
        - Used in systems like [AgentGym](https://arxiv.org/abs/2406.04151) by Xi et al. (2024), where a memory table tracks intermediate decisions and outcomes for reproducibility and long-term credit assignment.
        - Enables interpretable reasoning, making it possible to inspect or reset specific memory slots.
    2. **Implicit Neural Memory**:
        
        - Encodes temporal context within the model’s hidden states using architectures like Transformers, LSTMs, or recurrent attention mechanisms.
        - Particularly effective for LLMs fine-tuned via RLHF or DPO, where the hidden activations naturally preserve dialogue history and reasoning traces.
        - Recent innovations such as _recurrent Transformers_ and _memory-augmented attention_ extend this capability to tasks requiring hundreds of tokens of temporal coherence.
- Formally, implicit memory can be represented as an evolving state embedding ht=fθ(ht−1,st,at), where ht serves as a _latent world model_ summarizing all past experiences relevant to future predictions.
    
- In Early Experience, the same principle emerges organically through the **implicit world-modeling objective**:
    
    LIWM(θ)=−∑(si,aji,sji)logpθ(sji∣si,aji)
    
    - which forces the model to construct temporally predictive embeddings even without explicit memory modules—creating an “implicit long-term memory” foundation later leveraged during reinforcement learning.

### World Modeling: Learning Predictive Environment Representations

- **World models** enable agents to internalize the dynamics of their environments—predicting future states and rewards without constant external interaction. Originally introduced in [World Models](https://arxiv.org/abs/1803.10122) by Ha and Schmidhuber (2018), this approach decouples environment modeling from policy learning.
    
- A world model typically includes three components:
    
    1. **Encoder** Eϕ: maps raw observations ot to latent states zt=Eϕ(ot);
    2. **Transition Model** Tψ: predicts future latent states zt+1=Tψ(zt,at);
    3. **Decoder or Predictor** Dω: reconstructs or evaluates outcomes from latent states, such as rt=Dω(zt).
- By learning these components, the agent builds an internal simulation of the environment. This simulation can then be used for _planning_, _exploration_, or _policy evaluation_ without direct execution—dramatically improving sample efficiency.
    
- In web or tool-use domains, such models are extended to capture symbolic events (e.g., “clicked link,” “API returned error”) instead of pixels or low-level sensory data. The learned transition model enables agents to predict the _consequences of actions_ before performing them, supporting safer and more data-efficient learning.
    
- Both **Early Experience** and **DreamGym** build upon this concept but from complementary directions:
    
    - [Agent Learning via Early Experience](https://arxiv.org/abs/2510.08558) by Zhang et al. (2025) treats predictive modeling as a _language-native world model_—learning state transitions and self-reflective rationales purely from text-based environments before RL.
    - [Scaling Agent Learning via Experience Synthesis](https://arxiv.org/abs/2511.03773) by Chen et al. (2025) extends this into a formalized, **reasoning-based synthetic world model** (DreamGym) that produces internally consistent environment dynamics and synthetic rollouts. The experience model jointly generates next states and rewards under logical and semantic constraints, acting as a simulator for RL training with provable policy-transfer guarantees.

### Temporal Credit Assignment and Advantage Estimation

- For agents operating across long horizons, one of the hardest problems in RL is **credit assignment**—determining which past actions led to current rewards. In typical short-horizon tasks, temporal difference (TD) learning suffices, but for multi-step web agents, delayed or sparse rewards make attribution challenging.
    
- To address this, advantage-based and eligibility-trace methods extend standard RL updates:
    
    At=Rt−V(st)
    
    - where At is the _advantage_ of taking action at in state st, and Rt is the cumulative discounted reward:
    
    Rt=∑k=tTγk−trk
    
- For long episodes, this estimate is refined through _Generalized Advantage Estimation (GAE)_ ([High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438) by Schulman et al., 2016):
    
    A(λ)t=∑l=0∞(γλ)lδt+l
    
    - with temporal errors δt=rt+γV(st+1)−V(st).
- GAE smooths advantage estimation over time, balancing bias and variance while maintaining stability even in multi-turn settings.
    
- When integrated with **process-wise rewards** from reflective or synthetic environments, as proposed in Early Experience and DreamGym, GAE can assign proportional credit to intermediate reasoning steps or synthetic subgoals—reducing reward sparsity and improving credit flow through long-horizon trajectories.
    

### Hierarchical Reinforcement Learning (HRL)

- **Hierarchical RL (HRL)** structures policies across multiple temporal scales—dividing tasks into subtasks, each with its own sub-policy and local reward. This approach mirrors human task decomposition (e.g., “open browser → navigate → extract data → summarize results”).
    
- Formally, HRL decomposes the policy into two levels:
    
    - **High-Level (Manager) Policy** πH(g∣s): selects subgoals g;
    - **Low-Level (Worker) Policy** πL(a∣s,g): executes primitive actions to achieve g.
- The optimization objective becomes:
    
    J(πH,πL)=𝔼[∑tγt(r(L)t+λr(H)t)]
    
    - where r(L)t is the low-level reward for subgoal progress and r(H)t captures high-level task achievement.
- Recent frameworks like [AgentGym](https://arxiv.org/abs/2406.04151) and [OpenWebVoyager](https://arxiv.org/abs/2410.19609) employ hierarchical structures to separate reasoning and action planning layers. The high-level module reasons in natural language or symbolic goals, while the low-level policy executes API calls or UI actions. This separation improves both interpretability and modularity, enabling transfer learning across task domains.
    
- Early Experience aligns naturally with hierarchical RL by pretraining the “low-level” policy on predictive transitions (world modeling) and the “high-level” reflection policy on action rationales. DreamGym later merges both by training high- and low-level policies concurrently in a synthetic hierarchical environment, simulating multi-stage reasoning chains.
    

### Memory-Augmented Reinforcement Learning (MARL)

- Memory-augmented RL integrates explicit memory buffers or retrieval mechanisms into the learning loop, enabling agents to recall past experiences dynamically. Such architectures can be viewed as hybrids between world models and traditional replay buffers.
    
- A general MARL setup maintains:
    
    - **Episodic Memory** (M_e): stores sequences of (state, action, reward) tuples for reuse.
    - **Semantic Memory** (M_s): aggregates long-term knowledge, such as patterns in tool success or error likelihoods.
    - **Retrieval Policy** πM: determines which memories to recall based on current context.
- A key application is _retrieval-augmented decision-making_, where the policy is conditioned on both current observation and retrieved experiences:
    
    π(a|s,M)=fθ(s,Retrieve(M,s))
    
- This mechanism aligns conceptually with retrieval-augmented generation (RAG) but applied to RL: instead of retrieving documents, the agent retrieves _past trajectories_ that resemble the current state.
    
- DreamGym introduces an analogous process in the synthetic domain: the **experience model** retrieves and recombines previously generated synthetic rollouts to compose new simulated experiences that maximize policy coverage. This is effectively _synthetic memory replay_—training RL agents with a scalable, dynamically generated memory buffer of plausible state–action–reward transitions.
    

### Long-Horizon Planning Via Latent Rollouts and Model Predictive Control

- An emerging frontier in long-horizon RL for agents is **Model Predictive Control (MPC)** using latent world models. Instead of sampling actual environment steps, the agent “imagines” future rollouts within its learned model before committing to an action.
    
- Formally, for a world model Tψ(zt,at), MPC selects:
    
    a∗t=argmaxat,…,at+H𝔼[∑k=tt+Hγk−tr̂ (zk,ak)]
    
    - where r̂  and zk are predicted rewards and states over a planning horizon H.
- This technique allows for deep _internal simulation_, enabling efficient planning without costly real-environment interaction. In digital domains, MPC-like inference supports fast adaptation to new web layouts or API responses, with each rollout grounded by the world model’s predictions.
    
- DreamGym formalizes this concept at scale: the synthetic reasoning environment _is itself_ a controllable world model, allowing agents to perform model-predictive optimization over _generated latent rollouts_. These latent simulations substitute for environment sampling, providing a unified training–planning–evaluation loop that mirrors real-world behavior while remaining computationally tractable.
    

### Takeaways

- Memory, world modeling, and long-horizon credit assignment form the temporal backbone of agentic RL. Together, they enable continuity, foresight, and adaptive reasoning—core attributes for any system expected to function autonomously across diverse and evolving environments.
    
    - **Memory systems** preserve context and history across decisions.
    - **World models** internalize environmental dynamics, allowing for simulated reasoning.
    - **Credit assignment mechanisms** trace responsibility across deep trajectories.
    - **Hierarchical policies** decompose complex workflows into interpretable submodules.
    - **Model predictive control** enables safe, efficient long-horizon planning.
- When augmented with pre-RL **Early Experience** and scalable synthetic environments such as **DreamGym**, agents gain not only temporal coherence but also _generative foresight_: the ability to imagine, rehearse, and improve actions before executing them—effectively bridging the gap between reactive learning and proactive intelligence.
    

## Evaluation, Safety, and Interpretability in Reinforcement-Learning-Based Agents

### Why Evaluation and Safety Matter in RL-Based Agents

- As RL is increasingly applied to open-ended, tool-using, and web-interactive agents, questions of _safety_, _interpretability_, and _evaluation methodology_ have become central.
- Unlike static models—where evaluation can rely on accuracy or F1 scores—RL-based agents continually adapt, explore, and interact with dynamic environments. Their learned behaviors emerge from optimization, not from explicit instruction, which introduces the risk of **reward hacking**, **unsafe exploration**, or **misaligned optimization**.
    
- Evaluation and safety frameworks therefore aim to:
    
    - Quantify the _true capability_ of agents across reasoning, planning, and execution dimensions.
    - Detect and prevent _unintended emergent behaviors_ (e.g., exploiting web APIs incorrectly or entering infinite loops).
    - Ensure _alignment_ with human norms, values, and expectations.
- Recent works such as [Large Language Models Can Self-improve at Web Agent Tasks](https://arxiv.org/abs/2405.20309) by Patel et al. (2024) and [AgentBench: Evaluating LLMs as General-Purpose Agents](https://arxiv.org/abs/2310.01045) by Liu et al. (2024) emphasize that evaluation is not just performance measurement—it is _behavioral verification_ in a closed-loop context.

### Core Dimensions of Agent Evaluation

- Evaluation of RL-based agents extends across several orthogonal dimensions, each corresponding to a distinct capability or risk domain.

#### Task Performance

- Measures how effectively the agent accomplishes its intended goals.
    
    - **Metrics:** Success rate, accuracy, completion time, and cumulative reward.
    - **Examples:** Booking a ticket, executing a spreadsheet command, answering a query.

#### Behavioral Efficiency

- Assesses whether the agent achieves goals with minimal resource or action cost.
    
    - **Metrics:** Steps-to-success, energy or API call efficiency, latency.
    - **Significance:** Indicates policy optimization beyond brute-force trial and error.

#### Robustness and Generalization

- Evaluates how well the agent performs under perturbations—changes in environment layout, tool outputs, or input phrasing.
    
    - **Metrics:** Cross-environment transfer score, out-of-distribution success rate.
    - **Example:** Agent still performs correctly when a webpage’s button labels change.

#### Alignment and Ethical Compliance

- Examines whether actions remain consistent with human values, privacy norms, and safety boundaries.
    
    - **Metrics:** Human preference score, compliance violation rate, interpretability score.

#### Interpretability and Transparency

- Focuses on whether the agent’s internal reasoning or decision-making process can be understood, visualized, or audited.
    
    - **Metrics:** Explanation fidelity, action traceability, rationale coherence.
- Each dimension reflects a unique aspect of agent quality, and comprehensive evaluation must combine all to assess both _competence_ and _trustworthiness_.
    

### Safety Challenges in RL Agents

- The open-ended nature of RL training introduces specific safety risks not present in supervised learning.
    
    1. **Reward Hacking**: Agents may find unintended shortcuts that maximize reward without achieving the true goal—for instance, refreshing a page repeatedly to gain partial progress points. Mathematically, this reflects _reward misspecification_: the reward function r(s,a) does not perfectly encode human intent r∗(s,a).
        
    2. **Unsafe Exploration**: During training, agents may perform harmful or irreversible actions while attempting to maximize exploration-based rewards. In web or system environments, this could include deleting data or sending malformed API calls.
        
    3. **Catastrophic Forgetting**: Continual learning agents may lose previously learned safety behaviors when optimizing for new objectives, especially under non-stationary reward signals.
        
    4. **Non-Stationary Human Feedback**: In RLHF or DPO pipelines, shifting human preference distributions can cause instability if the agent overfits to transient feedback trends.
        
- A general safety objective adds a _regularization term_ to penalize risky or uncertain behavior:
    
    Jsafe(π)=𝔼[∑tγt(rt−λriskct)]
    
    - where ct quantifies risk (e.g., deviation from expected behavior) and λrisk controls conservatism.

### Interpretability and Traceability in Agent Behavior

- Interpretability in RL agents is especially challenging because learned policies are implicit, nonlinear functions that encode complex dynamics. However, several methods improve transparency and traceability:
    
    1. **Action Trace Logging** Record full trajectories of (state, action, reward) tuples for post-hoc analysis. Enables reconstruction of decision pathways, useful for debugging and ethical auditing.
        
    2. **Causal Attribution Maps** Estimate how much each observation influenced a given action. Techniques adapted from attention visualization or gradient saliency help identify which input elements guided the agent’s decisions.
        
    3. **Hierarchical Explanation Models** Used in agents trained via hierarchical RL, these models separate high-level goal explanations (e.g., “I am gathering data”) from low-level actions (“click button,” “read table”). This mirrors explainable AI (XAI) frameworks but grounded in reinforcement dynamics.
        
    4. **Language-Based Rationales** Some agents generate natural language explanations alongside their actions—a capability supported by recent instruction-tuned LLMs. These rationales can be integrated into the reward loop as an _explanation-consistency bonus_, reinforcing self-explanatory behavior.
        

### Safety-Aware RL Algorithms

- Several specialized RL formulations have been proposed to address safety-critical issues:
    
- **Constrained Policy Optimization (CPO):** Introduced by [Achiam et al. (2017)](https://arxiv.org/abs/1705.10528), CPO adds hard constraints to the optimization problem to ensure policies respect safety boundaries:
    
    maxπJ(π)s.t.𝔼π[C(s,a)]≤δ
    
    - where C(s,a) is a cost function and (\delta) the safety threshold.
- **Safe Exploration via Risk-Aware Value Functions:** Instead of optimizing for expected reward, these methods optimize _conditional value-at-risk (CVaR)_ to limit the probability of catastrophic outcomes.
    
- **Shielded Reinforcement Learning:** Incorporates a formal safety “shield” that intercepts actions violating constraints, replacing them with safe alternatives in real time.
    
- **Process-Wise Safety Scoring:** In complex environments like ToolEnv or MultiTurnEnv, step-wise safety checks are applied per subgoal or API call. For example, in a data retrieval task, each API call is evaluated for compliance and correctness before continuation.
    
- These algorithms formalize the notion of safety as part of the optimization loop, integrating constraint satisfaction directly into the learning process.
    

### Human-in-the-Loop (HITL) Evaluation and Oversight

- Human oversight remains a critical element in RL agent safety and evaluation pipelines.
- HITL systems provide:
    
    - **Preference feedback** for training reward models (RLHF).
    - **Trajectory curation** for identifying unsafe or unproductive behaviors.
    - **Live intervention mechanisms**, allowing humans to override or halt harmful action sequences.
- Emerging frameworks like [Themis](https://arxiv.org/abs/2310.01045) and [AgentBench](https://arxiv.org/abs/2310.01045) incorporate automated auditing layers that flag deviations from normal operating bounds. These can be paired with _real-time monitoring dashboards_ to visualize action probabilities, risk metrics, and outcome confidence.

### Benchmarking Frameworks for Safe and Transparent Evaluation

- Comprehensive benchmarking environments now combine safety, reasoning, and tool-use tasks under unified evaluation suites.
- Notable examples include:
    
    - **AgentGym** ([AgentGym](https://arxiv.org/abs/2406.04151) by Xi et al., 2024): A modular environment suite supporting SingleTurn, ToolEnv, and MultiTurn workflows, each with structured reward feedback and failure diagnostics.
    - **AgentBench** ([AgentBench](https://arxiv.org/abs/2310.01045) by Liu et al., 2024): Provides web, reasoning, and software operation benchmarks with alignment-focused scoring.
    - **OpenWebVoyager** ([OpenWebVoyager](https://arxiv.org/abs/2410.19609) by He et al., 2024): Realistic browser-based simulation for long-horizon web navigation tasks, used for testing contextual coherence and stability.
    - **WebArena** and **Mind2Web:** Large-scale web environments supporting reward shaping, human preference integration, and process-level logging for transparency.
- Together, these frameworks enable _holistic agent evaluation_—capturing not only goal success but also the _process integrity_ and _ethical soundness_ of the learned policies.

### Toward Aligned, Interpretable, and Reliable Agentic Systems

- As agentic RL systems continue to scale, their evaluation and safety mechanisms must evolve from reactive to _proactive_. Key directions include:
    
    - Embedding **interpretability hooks** within policy architectures.
    - Using **multi-objective optimization** to balance capability and safety rewards.
    - Adopting **model-based simulations** to test agents before deployment.
    - Incorporating **continuous monitoring** and **human-AI collaboration loops** for post-deployment oversight.
- In practice, the next generation of RL-based agents will need to demonstrate:
    
    - Predictable behavior under uncertainty,
    - Transparent reasoning chains,
    - Explicit accountability for outcomes,
    - Continuous adaptability without goal drift.
- This marks the transition from experimental RL toward **governed, auditable intelligence**—systems that can be trusted not just to perform, but to _behave_ in alignment with human values and operational safety constraints.
    

## Tool-Integrated Reasoning

- Tool-Integrated Reasoning (TIR) represents a fundamental evolution in the way LLMs learn and reason.
- It moves beyond static text generation into **interactive computation**, where the model dynamically decides _when_, _why_, and _how_ to use external tools (e.g., Python interpreters, APIs) as part of its reasoning trajectory.
- This section synthesizes insights from five foundational papers, grouped by their conceptual contribution and training methodology.

### Foundations and Theoretical Advancements in TIR

- **Core Idea:** [Understanding Tool-Integrated Reasoning](https://arxiv.org/abs/2508.19201) by Heng Lin & Zhongwen Xu (2025) formalizes the _tool-integrated reasoning loop_ as a Markov Decision Process (MDP), providing a principled RL framework for training models to use tools effectively. It introduces **Advantage Shaping Policy Optimization (ASPO)**, a variant of PPO that adds adaptive reward shaping to balance process- and outcome-based learning.
    
- **Mathematical formulation:** The ASPO objective is:
    
    ASPO=𝔼a∼πθ(a|h)[A(a,h)]−βDKL[πθ(a|h)||πref(a|h)]
    
    - where A(a,h) is the _shaped advantage_ incorporating both immediate (stepwise) and final (outcome) signals, and β controls regularization against a reference policy.
- **Implementation Highlights:**
    
    - Trains a 7B model on symbolic reasoning tasks.
    - Adds _step-level shaping_ to encourage timely tool use and verification behavior.
    - Significantly stabilizes RL optimization, improving both training and test accuracy by >6%.

### Practical Engineering for Stable Multi-Turn TIR

- **Core Idea:** [SimpleTIR: End-to-End Reinforcement Learning for Multi-Turn Tool-Integrated Reasoning](https://arxiv.org/abs/2509.02479) by Xue et al. (2025) addresses the _instability problem_ of multi-turn TIR. It isolates the causes of divergence during RL fine-tuning—such as gradient explosions and unproductive tool calls—and proposes three stabilizing strategies.
    
- **Key Contributions:**
    
    1. **Input-Gradient Norm Limiter**: caps backpropagation magnitude when token probabilities are extremely low.
    2. **Interpreter Output Masking**: prevents gradients from flowing through non-learnable tool feedback.
    3. **Void Turn Filtering**: removes steps with empty or redundant tool responses.
- **Empirical Findings:**
    
    - Using Qwen2.5-7B, SimpleTIR achieves faster convergence on AIME24.
    - Gradient clipping alone improved reward variance stability by 25%.
    - Masking and filtering yield additional 5–8% accuracy gains.

### Scaling Tool-Integrated RL from Base Models

- **Core Idea:** [ToRL: Scaling Tool-Integrated Reinforcement Learning](https://arxiv.org/abs/2503.23383) by Xuefeng Li, Haoyang Zou, & Pengfei Liu (2025) demonstrates that TIR can be trained _directly from base models_ without any supervised fine-tuning, relying entirely on exploration and reinforcement signals. This approach bridges _outcome-based_ reward optimization with emergent _process behavior_, such as model self-verification.
    
- **Training Design:**
    
    - Trains Qwen2.5-based models (1.5B and 7B) on five mathematical reasoning datasets.
    - Uses a pure correctness reward:
        
        R(a,â )={1,−1,if a=â otherwise
        
    - No explicit shaping; only final-answer feedback drives learning.
- **Results:**
    
    - **ToRL-7B**: 43.3% on AIME24, 62.1% across math benchmarks.
    - **Emergent behaviors:** self-verification and reflection loops, despite outcome-only reward.
    - Uses only **28.7K** training problems distilled from 75K candidates.
- The following figure ([source](https://arxiv.org/abs/2503.23383)) shows an example of CoT and TIR solution of the problem. TIR enables the model to write code and call an interpreter to obtain the output of the executed code, and then perform further reasoning based on the execution results.
    

![](https://aman.ai/primers/ai/assets/RL-for-agents/ToRL.jpg)

### Code-Interleaved Reinforcement for Tool Use

- **Core Idea:** [ReTool: Reinforcement Learning for Strategic Tool Use in LLMs](https://arxiv.org/abs/2504.11536) by Jiazhan Feng et al. (2025) establishes a robust training pipeline for tool-integrated reasoning through _interleaved code execution_. It uses real-time interpreter feedback during RL rollouts to optimize for both tool efficiency and correctness.
    
- **Training Stages:**
    
    1. **Cold-Start SFT** on a verified _code-integrated dataset_ DCI.
    2. **Interleaved PPO** where each generated code snippet is executed mid-rollout.
- **Modified PPO Objective:**
    
    JReTool(θ)=𝔼[min(πθ(ot|st;CI)πold(ot|st;CI)Â t,clip(⋅)Â t)]
    
- **Key Findings:**
    
    - Yields 27% higher accuracy over text-only PPO on AIME24.
    - Reduces reasoning token length by 40%.
    - Learns _strategic invocation_—earlier, more efficient, and self-corrective tool calls.
- The following figure ([source](https://arxiv.org/abs/2504.11536)) shows text-based RL training process and ReTool’s RL training process.
    

![](https://aman.ai/primers/ai/assets/RL-for-agents/ReTool.jpg)

### Tool-Augmented Evaluation Agents

- **Core Idea:** [Incentivizing Agentic Reasoning in LLM Judges via Tool-Integrated Reinforcement Learning](https://arxiv.org/abs/2510.23038) by Ran Xu et al. (2025) extends TIR to **evaluation agents (judges)**, which assess model outputs using executable verification tools. It integrates reinforcement learning to make LLM judges _agentic_—capable of reasoning, verifying, and scoring autonomously.
    
- **Methodology:**
    
    - Trains judges on three evaluation paradigms: **pointwise**, **pairwise**, and **listwise**.
    - Each trajectory involves both reasoning and code execution: (rk,ck)∼Jθ(x⊕sk−1),ok=I(ck),sk=sk−1⊕rk⊕ck⊕ok.
        
    - Two variants:
        
        - **TIR-Judge-Distill:** RL fine-tuning from distilled checkpoint.
        - **TIR-Judge-Zero:** trained from scratch via self-play RL.
- **Results:**
    
    - TIR-Judge-Zero performs comparably to distilled models.
    - Improves pairwise evaluation accuracy by +7.7%.
    - Enables _verifiable_ judgment by using executable tool outputs.
- The following figure ([source](https://arxiv.org/abs/2510.23038)) shows the overall framework of TIR-Judge variants. TIR-Judge natively supports tool use during judgment and is designed to handle diverse input formats.
    

![](https://aman.ai/primers/ai/assets/RL-for-agents/TIR-Judge.jpg)

### Synthesizing Trends in TIR + RL Integration

- Across these works, TIR emerges as the unifying interface between _language_ and _computation_. Each study progressively strengthens one aspect of the TIR-RL ecosystem:

|**Group**|**Representative Paper**|**Main Contribution**|**Reward Type**|**Emergent Capability**|
|---|---|---|---|---|
|1|_Understanding TIR_|Theoretical formalization (ASPO)|Stepwise + Outcome|Advantage shaping & stability|
|2|_SimpleTIR_|Stabilization in multi-turn settings|Stepwise|Controlled gradient flow|
|3|_ToRL_|Scaling from base models|Outcome-only|Emergent verification|
|4|_ReTool_|Interleaved code execution|Outcome-only|Strategic tool use|
|5|_TIR-Judge_|Tool-augmented evaluation|Multi-level|Self-verifying reward models|

- Together, these advances redefine RL for reasoning agents: from optimizing token probabilities to optimizing **interactive decision-making with verifiable computation**.

### Synthesis: Beyond Individual Tool Use

- Together, these works outline a continuum of Tool-Integrated RL:

|**Framework**|**Focus**|**Environment Type**|**Key Mechanism**|**Performance Gain**|
|---|---|---|---|---|
|**Li et al. (2025)**|Mathematical reasoning|SingleTurnEnv|Code-augmented execution|+13% accuracy|
|**Xue et al. (2025)**|Multi-API orchestration|ToolEnv|Composite action sequencing|+35% efficiency|
|**Lin et al. (2025)**|Multi-agent collaboration|MultiTurnEnv|Cooperative reward sharing|–42% exploration cost|

- These studies collectively show that **tool use is no longer a static feature**, but a **learned behavior**—optimized via RL to balance exploration, compositionality, and cooperation.
    
- By embedding **tool invocation** into the policy space and integrating **reward feedback** from external computation, TIR-RL agents represent a new class of hybrid intelligence—merging the _symbolic precision of tools_ with the _adaptive learning of reinforcement_.
    

### Unifying RL and TIR: Process vs. Outcome Rewards

- TIR-based RL frameworks bridge **process-wise** and **outcome-based** rewards.
    
    - **Process rewards** measure reasoning correctness at intermediate tool-use steps (e.g., code executes without error).
    - **Outcome rewards** evaluate the final correctness or verification success.
- The total return function becomes:
    
    R=∑tλprprocesst+λoroutcomet
    
    - … balancing exploration of intermediate reasoning paths and end-task accuracy.
- This hybrid reward scheme is now central in environments like **ToolEnv** and **MultiTurnEnv**, enabling nuanced optimization of reasoning workflows.
    

### Synthesis and Outlook

- Tool-Integrated Reasoning (TIR) provides the operational bridge between _language_ and _action_. When fused with RL:
    
    - It transforms reasoning into a closed-loop control process.
    - It grounds learning in executable feedback, reducing hallucination.
    - It yields agents capable of **self-verification, self-correction, and self-improvement**.
- In sum:
    
    - [ReTool](https://arxiv.org/abs/2504.11536) formalized RL for tool-based reasoning.
    - [TIR-Judge](https://arxiv.org/abs/2510.23038) extended RL-based tool reasoning to evaluation.
    - [Li et al. (2025)](https://arxiv.org/abs/2503.23383), [Xue et al. (2025)](https://arxiv.org/abs/2508.19201), and [Lin et al. (2025)](https://arxiv.org/abs/2509.02479) unified the landscape of tool-integrated reinforcement learning by demonstrating that RL-trained agents can autonomously discover, schedule, and verify tool use—laying the groundwork for scalable, self-improving reasoning systems.
- Collectively, these works mark the beginning of **agentic cognition**—where models reason, act, and verify within the same policy loop.
    

## Citation

![](https://aman.ai/images/copy.png)

`@article{Chadha2020DistilledRLForAgents,   title   = {Reinforcement Learning for Agents},   author  = {Chadha, Aman and Jain, Vinija},   journal = {Distilled AI},   year    = {2020},   note    = {\url{https://aman.ai}} }`