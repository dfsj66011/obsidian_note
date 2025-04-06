
        - [Evolution from DeepSeek-V3 to DeepSeek-R1](https://aman.ai/primers/ai/deepseek-R1/#evolution-from-deepseek-v3-to-deepseek-r1)
            - [MTP in DeepSeek-V3](https://aman.ai/primers/ai/deepseek-R1/#mtp-in-deepseek-v3)
                - [Sequential Multi-Token Prediction Modules](https://aman.ai/primers/ai/deepseek-R1/#sequential-multi-token-prediction-modules)
                - [MTP Training Objective](https://aman.ai/primers/ai/deepseek-R1/#mtp-training-objective)
                - [Memory Optimization with Shared Embeddings and Output Heads](https://aman.ai/primers/ai/deepseek-R1/#memory-optimization-with-shared-embeddings-and-output-heads)
                - [Inference Strategy and Speculative Decoding](https://aman.ai/primers/ai/deepseek-R1/#inference-strategy-and-speculative-decoding)
                - [Ablation Studies on Multi-Token Prediction](https://aman.ai/primers/ai/deepseek-R1/#ablation-studies-on-multi-token-prediction)
            - [Enhancements in DeepSeek-R1](https://aman.ai/primers/ai/deepseek-R1/#enhancements-in-deepseek-r1-2)
                - [Improved Token Dependency Modeling in MTP](https://aman.ai/primers/ai/deepseek-R1/#improved-token-dependency-modeling-in-mtp)
                - [Adaptive Prediction Granularity](https://aman.ai/primers/ai/deepseek-R1/#adaptive-prediction-granularity)
                - [Loss Function Refinement for Multi-Depth Learning](https://aman.ai/primers/ai/deepseek-R1/#loss-function-refinement-for-multi-depth-learning)
                - [Optimized Memory Efficiency with Parameter Sharing](https://aman.ai/primers/ai/deepseek-R1/#optimized-memory-efficiency-with-parameter-sharing)
                - [Enhanced Inference Strategy with Speculative Decoding](https://aman.ai/primers/ai/deepseek-R1/#enhanced-inference-strategy-with-speculative-decoding)
                - [Empirical Gains from DeepSeek-R1’s MTP Enhancements](https://aman.ai/primers/ai/deepseek-R1/#empirical-gains-from-deepseek-r1s-mtp-enhancements)
            - [Comparative Analysis](https://aman.ai/primers/ai/deepseek-R1/#comparative-analysis-2)
        - [Implementation Details](https://aman.ai/primers/ai/deepseek-R1/#implementation-details)
            - [Mathematical Formulation](https://aman.ai/primers/ai/deepseek-R1/#mathematical-formulation-1)
- [DeepSeek-R1-Zero →→ Training Pipeline: Pure Reinforcement Learning in DeepSeek-R1-Zero](https://aman.ai/primers/ai/deepseek-R1/#deepseek-r1-zero-rightarrow-training-pipeline-pure-reinforcement-learning-in-deepseek-r1-zero)
- [DeepSeek-R1 →→ Training Pipeline: Cold-Start SFT to Multi-Stage RL](https://aman.ai/primers/ai/deepseek-R1/#deepseek-r1-rightarrow-training-pipeline-cold-start-sft-to-multi-stage-rl)
    - [Stage 1: Cold Start with SFT](https://aman.ai/primers/ai/deepseek-R1/#stage-1-cold-start-with-sft)
        - [Fine-Tuning with High-Quality Chain-of-Thought (CoT) Examples](https://aman.ai/primers/ai/deepseek-R1/#fine-tuning-with-high-quality-chain-of-thought-cot-examples)
        - [Structured Output Format](https://aman.ai/primers/ai/deepseek-R1/#structured-output-format)
        - [Loss Function for SFT](https://aman.ai/primers/ai/deepseek-R1/#loss-function-for-sft)
    - [Stage 2: RL](https://aman.ai/primers/ai/deepseek-R1/#stage-2-rl)
        - [DeepSeek’s RL Methodology: a Conceptual Overview](https://aman.ai/primers/ai/deepseek-R1/#deepseeks-rl-methodology-a-conceptual-overview)
        - [Background: Policy Optimization](https://aman.ai/primers/ai/deepseek-R1/#background-policy-optimization)
            - [The REINFORCE Algorithm](https://aman.ai/primers/ai/deepseek-R1/#the-reinforce-algorithm)
                - [What is REINFORCE?](https://aman.ai/primers/ai/deepseek-R1/#what-is-reinforce)
                - [Limitations of REINFORCE](https://aman.ai/primers/ai/deepseek-R1/#limitations-of-reinforce)
            - [Proximal Policy Optimization (PPO)](https://aman.ai/primers/ai/deepseek-R1/#proximal-policy-optimization-ppo)
                - [How PPO Works](https://aman.ai/primers/ai/deepseek-R1/#how-ppo-works)
                - [Challenges with PPO](https://aman.ai/primers/ai/deepseek-R1/#challenges-with-ppo)
        - [Group Relative Policy Optimization (GRPO)](https://aman.ai/primers/ai/deepseek-R1/#group-relative-policy-optimization-grpo)
            - [Key Innovations](https://aman.ai/primers/ai/deepseek-R1/#key-innovations)
                - [How GRPO Builds on REINFORCE](https://aman.ai/primers/ai/deepseek-R1/#how-grpo-builds-on-reinforce)
                - [How GRPO Builds on PPO](https://aman.ai/primers/ai/deepseek-R1/#how-grpo-builds-on-ppo)
            - [Evolution of GRPO: from DeepSeekMath to DeepSeek-R1](https://aman.ai/primers/ai/deepseek-R1/#evolution-of-grpo-from-deepseekmath-to-deepseek-r1)
                - [Phase 1: GRPO in DeepSeekMath (Mathematical RL)](https://aman.ai/primers/ai/deepseek-R1/#phase-1-grpo-in-deepseekmath-mathematical-rl)
                - [Phase 2: GRPO in DeepSeek-R1-Zero (Self-Evolving Reasoning)](https://aman.ai/primers/ai/deepseek-R1/#phase-2-grpo-in-deepseek-r1-zero-self-evolving-reasoning)
                - [Phase 3: GRPO in DeepSeek-R1 (Refined Reasoning & Cold Start)](https://aman.ai/primers/ai/deepseek-R1/#phase-3-grpo-in-deepseek-r1-refined-reasoning--cold-start)
            - [How GRPO Works](https://aman.ai/primers/ai/deepseek-R1/#how-grpo-works)
                - [Mathematical Formulation](https://aman.ai/primers/ai/deepseek-R1/#mathematical-formulation-2)
                - [Mathematical Intuition](https://aman.ai/primers/ai/deepseek-R1/#mathematical-intuition)
            - [Step-by-Step Breakdown](https://aman.ai/primers/ai/deepseek-R1/#step-by-step-breakdown)
                - [Policy Likelihood Ratio ρiρi](https://aman.ai/primers/ai/deepseek-R1/#policy-likelihood-ratio-rho_i)
                - [Advantage Function AiAi](https://aman.ai/primers/ai/deepseek-R1/#advantage-function-a_i)
                - [Clipping Mechanism clip(⋅)clip(⋅)](https://aman.ai/primers/ai/deepseek-R1/#clipping-mechanism-clipcdot)
                - [KL Divergence Penalty DKLDKL](https://aman.ai/primers/ai/deepseek-R1/#kl-divergence-penalty-d_textkl)
                - [Old Policy πoldπold](https://aman.ai/primers/ai/deepseek-R1/#old-policy-pi_textold)
                - [Reference Policy πrefπref](https://aman.ai/primers/ai/deepseek-R1/#reference-policy-pi_textref)
            - [Algorithm](https://aman.ai/primers/ai/deepseek-R1/#algorithm)
                - [Reward Function Design](https://aman.ai/primers/ai/deepseek-R1/#reward-function-design)
            - [Advantage Estimation](https://aman.ai/primers/ai/deepseek-R1/#advantage-estimation)
                - [Background: Generalized Advantage Estimation](https://aman.ai/primers/ai/deepseek-R1/#background-generalized-advantage-estimation)
                - [Background: PPO Advantage Estimation](https://aman.ai/primers/ai/deepseek-R1/#background-ppo-advantage-estimation)
                - [GRPO Advantage Estimation](https://aman.ai/primers/ai/deepseek-R1/#grpo-advantage-estimation)
            - [Comparative Analysis: REINFORCE vs. TRPO vs. PPO vs. DPO vs. KTO vs. APO vs. GRPO](https://aman.ai/primers/ai/deepseek-R1/#comparative-analysis-reinforce-vs-trpo-vs-ppo-vs-dpo-vs-kto-vs-apo-vs-grpo)
                - [Tabular Comparison](https://aman.ai/primers/ai/deepseek-R1/#tabular-comparison)
        - [Reward Functions](https://aman.ai/primers/ai/deepseek-R1/#reward-functions)
            - [Accuracy Rewards](https://aman.ai/primers/ai/deepseek-R1/#accuracy-rewards)
            - [Format Rewards](https://aman.ai/primers/ai/deepseek-R1/#format-rewards)
            - [Combined Reward Function](https://aman.ai/primers/ai/deepseek-R1/#combined-reward-function)
            - [Why Rule-Based Rewards Instead of Neural Reward Models?](https://aman.ai/primers/ai/deepseek-R1/#why-rule-based-rewards-instead-of-neural-reward-models)
            - [Implementation in GRPO](https://aman.ai/primers/ai/deepseek-R1/#implementation-in-grpo)
    - [Stage 3: Rejection Sampling & Expanded Supervised Fine-Tuning](https://aman.ai/primers/ai/deepseek-R1/#stage-3-rejection-sampling--expanded-supervised-fine-tuning)
    - [Stage 4: Secondary RL for Alignment & Generalization](https://aman.ai/primers/ai/deepseek-R1/#stage-4-secondary-rl-for-alignment--generalization)
    - [Comparing Training Pipelines: DeepSeek-R1 vs. DeepSeek-R1-Zero](https://aman.ai/primers/ai/deepseek-R1/#comparing-training-pipelines-deepseek-r1-vs-deepseek-r1-zero)
        - [Pre-Training and Initialization](https://aman.ai/primers/ai/deepseek-R1/#pre-training-and-initialization)
        - [RL Strategy](https://aman.ai/primers/ai/deepseek-R1/#rl-strategy)
            - [DeepSeek-R1-Zero: Pure RL Approach](https://aman.ai/primers/ai/deepseek-R1/#deepseek-r1-zero-pure-rl-approach)
            - [DeepSeek-R1: Multi-Stage RL with Cold-Start Fine-Tuning](https://aman.ai/primers/ai/deepseek-R1/#deepseek-r1-multi-stage-rl-with-cold-start-fine-tuning)
        - [Implementation Details and Computational Efficiency](https://aman.ai/primers/ai/deepseek-R1/#implementation-details-and-computational-efficiency)
        - [Final Performance Impact](https://aman.ai/primers/ai/deepseek-R1/#final-performance-impact)
- [Emergent Reasoning Behaviors](https://aman.ai/primers/ai/deepseek-R1/#emergent-reasoning-behaviors)
    - [Implementation Details](https://aman.ai/primers/ai/deepseek-R1/#implementation-details-1)
    - [Example: Quadratic Equation Solving](https://aman.ai/primers/ai/deepseek-R1/#example-quadratic-equation-solving)
- [Distillation: Reasoning in Compact Models](https://aman.ai/primers/ai/deepseek-R1/#distillation-reasoning-in-compact-models)
    - [Implementation Details](https://aman.ai/primers/ai/deepseek-R1/#implementation-details-2)
- [Results](https://aman.ai/primers/ai/deepseek-R1/#results)
    - [Average Response Length vs. Timesteps](https://aman.ai/primers/ai/deepseek-R1/#average-response-length-vs-timesteps)
    - [Comparison of DeepSeek-R1 and DeepSeek-R1-Zero](https://aman.ai/primers/ai/deepseek-R1/#comparison-of-deepseek-r1-and-deepseek-r1-zero)
        - [Training Approach](https://aman.ai/primers/ai/deepseek-R1/#training-approach)
        - [Performance Differences](https://aman.ai/primers/ai/deepseek-R1/#performance-differences)
        - [Readability and Language Consistency](https://aman.ai/primers/ai/deepseek-R1/#readability-and-language-consistency)
        - [Self-Evolution and “Aha Moments”](https://aman.ai/primers/ai/deepseek-R1/#self-evolution-and-aha-moments)
- [Prompt Template](https://aman.ai/primers/ai/deepseek-R1/#prompt-template)
- [Open Questions](https://aman.ai/primers/ai/deepseek-R1/#open-questions)
- [Other Reasoning Models](https://aman.ai/primers/ai/deepseek-R1/#other-reasoning-models)
    - [QwQ: Reflect Deeply on the Boundaries of the Unknown](https://aman.ai/primers/ai/deepseek-R1/#qwq-reflect-deeply-on-the-boundaries-of-the-unknown)
    - [S1: Simple Test-Time Scaling](https://aman.ai/primers/ai/deepseek-R1/#s1-simple-test-time-scaling)
    - [Sky-T1](https://aman.ai/primers/ai/deepseek-R1/#sky-t1)
    - [Kimi K1.5: Scaling Reinforcement Learning with LLMs](https://aman.ai/primers/ai/deepseek-R1/#kimi-k15-scaling-reinforcement-learning-with-llms)
    - [Open-R1](https://aman.ai/primers/ai/deepseek-R1/#open-r1)
        - [Objectives of Open-R1](https://aman.ai/primers/ai/deepseek-R1/#objectives-of-open-r1)
        - [Impact on the Community](https://aman.ai/primers/ai/deepseek-R1/#impact-on-the-community)
- [DeepSeek R1-1776](https://aman.ai/primers/ai/deepseek-R1/#deepseek-r1-1776)
- [Open-Source Reasoning Datasets](https://aman.ai/primers/ai/deepseek-R1/#open-source-reasoning-datasets)
- [FAQs](https://aman.ai/primers/ai/deepseek-R1/#faqs)
    - [Is GRPO a Policy Gradient Algorithm?](https://aman.ai/primers/ai/deepseek-R1/#is-grpo-a-policy-gradient-algorithm)
    - [Is GRPO an Actor-critic Algorithm?](https://aman.ai/primers/ai/deepseek-R1/#is-grpo-an-actor-critic-algorithm)
    - [Can GRPO be Applied to Outcome Supervision or Process Supervision or Both? How is the Advantage Computed from Reward in Either Case?](https://aman.ai/primers/ai/deepseek-R1/#can-grpo-be-applied-to-outcome-supervision-or-process-supervision-or-both-how-is-the-advantage-computed-from-reward-in-either-case)
        - [Outcome Supervision](https://aman.ai/primers/ai/deepseek-R1/#outcome-supervision)
        - [Process Supervision](https://aman.ai/primers/ai/deepseek-R1/#process-supervision)
    - [How is a Reward Model Different from a Value/critic Model in Policy Optimization Algorithms Such As GRPO?](https://aman.ai/primers/ai/deepseek-R1/#how-is-a-reward-model-different-from-a-valuecritic-model-in-policy-optimization-algorithms-such-as-grpo)
        - [Reward Model](https://aman.ai/primers/ai/deepseek-R1/#reward-model)
        - [Value Model (Critic)](https://aman.ai/primers/ai/deepseek-R1/#value-model-critic)
        - [Key Differences in GRPO](https://aman.ai/primers/ai/deepseek-R1/#key-differences-in-grpo)
        - [Summary](https://aman.ai/primers/ai/deepseek-R1/#summary)
    - [In the Equation for GRPO, What is the Role of the Old Policy Compared to the Reference Policy?](https://aman.ai/primers/ai/deepseek-R1/#in-the-equation-for-grpo-what-is-the-role-of-the-old-policy-compared-to-the-reference-policy)
    - [Why is the PPO/GRPO Objective Called a Clipped “surrogate” Objective?](https://aman.ai/primers/ai/deepseek-R1/#why-is-the-ppogrpo-objective-called-a-clipped-surrogate-objective)
    - [What are Some Considerations around the Reasoning Tokens Budget in Reasoning LLMs?](https://aman.ai/primers/ai/deepseek-R1/#what-are-some-considerations-around-the-reasoning-tokens-budget-in-reasoning-llms)
- [References](https://aman.ai/primers/ai/deepseek-R1/#references)

## 一、介绍

- 两个模型都利用了在 DeepSeekMath 中引入的 GRPO，取代了传统方法如 PPO，使训练更高效且具可扩展性。它们还采用了在 DeepSeek-V2 中引入的多头潜在注意力（MLA），通过将 KQV 矩阵投影到低维潜在空间中，减少了长上下文处理中的计算和内存效率问题。
- 通过 GRPO、FP8 量化和自然生成的 CoT 推理等创新，这两个模型在促进透明性和可访问性的同时，能与闭源模型竞争。随着研究界在这些创新基础上不断发展，DeepSeek-R1 标志着向高效、推理驱动的 AI 发展的转变，使其对所有人都可访问。
- 本入门指南探讨了其架构、多阶段训练流程、GRPO 机制和自然推理行为，以及蒸馏如何将推理能力传播到更小的模型中。

## 二、架构基础

- DeepSeek-R1 基于 DeepSeek-V2 引入的基础进展——特别是专家混合（MoE）和多头潜在注意力（MLA）——以及 DeepSeek-V3 中的多标记预测（MTP），整合了尖端的架构创新，优化了训练效率和推理性能。
- 本节详细分解了从 DeepSeek-V2 和 DeepSeek-V3 演变到 DeepSeek-R1 的架构组件，突出改进之处，使 DeepSeek-R1 成为领先的开源模型，能够在推理效率和性能上与专有替代品竞争。

### 2.1 概述

- DeepSeek-R1 通过多种先进技术实现了显著的效率提升：

    1. *MoE 架构* ：DeepSeek-R1 利用专家混合模型，将大型模型分解为更小的、专门化的子模型。这种架构允许在特定任务中仅激活相关子模型，使系统能够在消费级 GPU 上高效运行。
    2. *通过 MLA 的键值存储压缩*：通过实施复杂的压缩算法，DeepSeek-R1 实现了键值索引存储需求的 93% 减少，这些索引通常会消耗大量的 VRAM。
    3. *MTP*：DeepSeek-R1 设计为同时预测多个标记，而不是一次一个。此策略有效地将推理速度提高了一倍，增强了整体性能。
    4. *低精度计算*：DeepSeek-R1 采用混合精度算术，使用 8 位浮点数代替标准的 32 位进行大量计算。这种方法大大减少了内存消耗并加速了处理速度。

- 这些创新共同促成了 DeepSeek-R1 在训练效率上的显著进步，据报道比之前的模型提高了 45 倍。


### 2.2 混合专家 MoE

#### 2.2.1 概述

- MoE 机制在每个推理步骤中选择性地激活一部分模型参数，实现计算节省的同时保持模型质量。这种方法允许在不成比例增加计算成本的情况下扩展模型参数。
- DeepSeek-R1 对 DeepSeek-V2 的 MoE 框架进行了改进，推出了动态专家路由、基于强化学习的负载平衡和增强的稀疏性约束。这些创新使 DeepSeek-R1 成为最有效且可扩展的开源 MoE 模型之一。

#### 2.2.2 主要特性

- *基于 RL 的专家路由*：DeepSeek-R1 用 RL 策略取代静态门控函数，以动态分配 token 给专家。RL 驱动的路由器通过最大化负载平衡和最小化路由熵来优化专家选择，从而实现更高效的 token -专家映射。
- *分层熵门控 MoE (HE-MoE)*：专家选择过程通过多级门控机制得到优化。token 首先通过全局选择阶段，然后是集群级修剪，最后进行熵感知调整，确保专家激活的平衡。此方法防止专家过度专业化并提高泛化能力。
- *设备受限的专家分配 (DCEA)*：根据可用计算资源分配专家，减少跨设备通信开销。模型在受限设备池中选择专家，降低同步成本，提高训练效率。
- *基于 RL 调整的负载平衡专家利用*：DeepSeek-R1 动态调整专家激活概率，使用基于 RL 的偏差项，而不是依赖辅助损失函数来平衡负载。这确保了一致的工作负载分布，改善了稳定性和收敛性。
- *完整 token 保留（无标记丢弃）*：与早期版本不同，DeepSeek-R1 在训练和推理期间保留所有标记，确保信息不丢失，从而提高模型的连贯性和泛化能力。
- *跨设备通信优化*：通过 DCEA 和分层专家门控，DeepSeek-R1 大幅减少设备间通信，延迟降低高达 35%。这种优化提高了效率而不牺牲模型性能。
- *动态专家激活*：模型使用学习到的路由策略动态调整专家选择，确保计算资源的高效分配。这使得 DeepSeek-R1 能够有效扩展，而不线性增加计算成本。
- *自适应专家专业化*：通过引入基于熵的约束，DeepSeek-R1 确保专家保持专门化但不过于僵化。这种动态专业化在提高准确性和效率的同时，保持了专家激活的灵活性。

#### 2.2.3 从 DeepSeek-V2 到 DeepSeek-R1 的演变

##### 2.2.3.1 DeepSeek-V2 中的 MoE

- DeepSeek-V2 引入了一种名为 DeepSeekMoE 的专用 MoE 架构，在保持强大性能的同时优化了模型训练效率和推理吞吐量。该架构通过改进专家选择、路由和负载平衡策略来减少计算开销。以下是 DeepSeek-V2 中 MoE 特定机制的详细分解。

###### 2.2.3.1.1 DeepSeekMoE 的基本架构

- DeepSeekMoE 通过精细的专家分割和共享专家隔离来增加专业化并减少冗余。DeepSeek-V2 中的 MoE 架构包括：
  - $N_s$ 个共享专家，处理所有标记。
  - $N_r$ 个路由专家，根据门控函数选择性激活以处理 token。
  - 每个 token 由固定数量 $K_r$ 的路由专家处理。
  
- MoE 层的输出计算为：$$
h'_t = u_t + \sum_{i=1}^{N_s} \text{FFN}^{(s)}_i(u_t) + \sum_{i=1}^{N_r} g_{i,t} \text{FFN}^{(r)}_i(u_t)$$其中：
    - $\text{FFN}^{(s)}_i$ 表示一个共享专家。
    - $\text{FFN}^{(r)}_i$ 表示一个路由专家。
    - $g_{i,t}$ 是用于标记 $t$ 的专家选择的门控函数。

- 门控函数如下：$$
g_{i,t} = 
\begin{cases} 
s_{i,t}, & s_{i,t} \in \text{Top-}K_r(\{s_{j,t} \mid 1 \leq j \leq N_r\}) \\
0, & \text{otherwise}
\end{cases}$$其中 $s_{i,t}$ 是 softmax 加权的 token-专家亲和度：$$
s_{i,t} = \text{Softmax}_i(u_t^T e_i)$$这里 $e_i$ 是专家 $i$ 的中心。

###### 2.2.3.1.2 设备限制路由

- 在 MoE 模型中，专家并行引入的通信开销是主要的计算瓶颈之一。为了解决这个问题，DeepSeekMoE 实现了设备限制路由，限制了 token 的专家可以分布的设备数量。
- **关键实现细节：**
  - 每个 token 首先选择具有最高亲和分数的 $M$ 个设备。
  - 最终的 $K_r$ 个专家仅从这些选定的设备中选择。
- 实际中，设置 $M \geq 3$ 可以确保性能接近于不受限路由，同时显著减少设备间通信。

###### 2.2.3.1.3 负载平衡的辅助损失

- DeepSeek-V2 使用多种辅助损失来确保专家的均衡利用，避免某些专家过载而其他专家未被充分利用的情况。具体如下：

    - *专家级别的平衡损失*：
        - 为防止路由崩溃（仅有部分专家得到训练），DeepSeek-V2 最小化：$$L_{\text{ExpBal}} = \alpha_1 \sum_{i=1}^{N_r} \frac{f_i}{P_i}$$其中：
	        - $f_i$ 是路由到专家 $i$ 的标记比例，
	        - $P_i$ 是选择专家 $i$ 的平均概率，
	        - $\alpha_1$ 是控制损失强度的超参数。
        
    - *设备级别的平衡损失*：
        - 为在设备间均匀分配计算，DeepSeekMoE 将专家分配到 $D$ 个设备组，每个组在单独的设备上运行。平衡损失为：$$L_{\text{DevBal}} = \alpha_2 \sum_{i=1}^{D} \frac{f'_i}{P'_i}$$其中 $f'_i$ 和 $P'_i$ 聚合了设备 $i$ 上所有专家的使用统计
        
    - *通信平衡损失*:
        
        - 该损失确保每个设备接收到大致相等数量的标记，防止由于过多通信负载导致的瓶颈：$$L_{\text{CommBal}} = \alpha_3 \sum_{i=1}^{D} \frac{f''_i}{P''_i}$$其中 $f''_i$ 和 $P''_i$ 衡量发送到设备 $i$ 的标记比例。

###### 2.2.3.1.4 Token-Dropping 策略

- 虽然辅助损失可以改善平衡，但它们不能严格保证专家的均匀利用。为进一步减少低效，DeepSeek-V2 在设备级别实施了 Token-Dropping 策略：
    - 首先估算每个设备的计算预算。
    - 丢弃亲和分数最低的标记，直到满足预算。
    - 至少有 10% 的训练序列不进行标记丢弃，以确保多样性。
- 这种方法允许在推理过程中根据计算限制动态调整标记保留的灵活性。



##### 2.2.3.2 DeepSeek-V3 的增强

- DeepSeek-V3 相较于 DeepSeek-V2 在 MoE 框架上引入了多项显著改进。这些增强主要集中在提高模型效率、降低训练和推理成本，并保持高性能。关键改进包括无辅助损失的负载平衡策略、节点限制路由、改进的专家选择机制以及增强的稀疏性约束。这些进步使得训练更高效、推理更快速，并且性能优于 DeepSeek-V2。

###### 2.2.3.2.1 无辅助损失的负载平衡

- 与依赖辅助损失来确保专家利用平衡的 DeepSeek-V2 相比，DeepSeek-V3 引入了一种无辅助损失的策略。DeepSeek-V3 通过动态调整偏置项来调整专家选择，而不是通过额外损失项来惩罚不平衡。专家门控函数修改如下：$$
    g'_{i,t} = 
    \begin{cases} 
    s_{i,t}, & s_{i,t} + b_i \in \text{Top-}K_r(\{s_{j,t} + b_j \mid 1 \leq j \leq N_r\}) \\
    0, & \text{otherwise}
    \end{cases}$$其中 $b_i$ 是根据多个训练步骤中专家 $i$ 的负载动态调整的偏置项：$$
    b_i \leftarrow 
    \begin{cases} 
    b_i - \gamma, & \text{如果专家 } i \text{ 负载过重} \\
    b_i + \gamma, & \text{否则}
    \end{cases}$$
- 这种动态调整确保专家负载保持平衡，无需辅助损失惩罚，从而提高训练稳定性和效率。

###### 2.2.3.2.2 节点限制路由（NLR）

- DeepSeek-V3 引入了节点限制路由（NLR），以进一步优化大规模 MoE 训练中的通信开销。NLR 限制了每个 token 可以通信的节点数量，而不是允许 token 与模型中的任何专家通信。路由机制为每个 token 选择最多 $M$ 个节点，确保专家分配时最小化节点间同步。$$M = \sum_{i=1}^{N} \max \{ s_{j,t} \mid j \in \text{node } i \}$$ 
- 这种方法显著减少了跨节点通信开销，从而加快训练和推理速度。

###### 2.2.3.2.3 改进的专家选择机制

- DeepSeek-V3 通过引入基于 sigmoid 的 token-专家亲和函数，改进了专家选择机制，而不是像 DeepSeek-V2 那样使用 softmax 机制。新的函数定义为：$$s_{i,t} = \sigma(u_t^T e_i)$$其中 $e_i$ 是专家 $i$ 的中心，$\sigma(\cdot)$ 是 sigmoid 激活函数。选择过程然后对 Top-$K_r$ 专家分数进行归一化：$$g_{i,t} = \frac{g'_{i,t}}{\sum_{j \in \text{Top-}K_r} g'_{j,t}}$$
- 这一修改防止了极端的专家选择概率，导致更好的负载平衡和专业化。

###### 2.2.3.2.4 增强稀疏性约束与分层门控

- 为避免过度专业化并促进泛化，DeepSeek-V3 引入了分层门控。与传统的 top-$K$ 门控不同，这种方法在多个层次上应用稀疏性约束：

    - *全局选择*： 在粗粒度层次上初步选择 $N_g$ 个专家。
    - *cluster 级修剪*：在选定的簇内进一步筛选专家，得到 $K_r$ 个专家。
    - *基于熵的调整*：根据熵约束调整专家激活概率，以避免极端稀疏。

- 数学上，基于熵的调整修改门控得分如下：$$g_{i,t} = g_{i,t} \times (1 - \lambda \cdot H(g_{1:N_r,t}))$$其中 $H(\cdot)$ 是熵函数，$\lambda$ 是正则化系数，用于控制均匀选择与专业化之间的权衡。

###### 2.2.3.2.5 无 Token 丢弃策略

- DeepSeek-V2 实施了 token 丢弃策略以平衡每个设备的计算负载。然而，DeepSeek-V3 的增强负载平衡机制消除了对 token 丢弃的需求，确保在训练和推理过程中 100% 保留 token。这提高了泛化能力，并避免了模型更新时的信息丢失。

##### 2.2.3.3 DeepSeek-R1 的增强

- DeepSeek-R1 对 MoE 框架引入了多项重大改进，提升了计算效率、负载平衡和推理准确性。这些改进是在 DeepSeek-V3 优化的基础上进行的，集成了基于强化学习的路由策略、熵控制门控以及精细化的专家专业化。以下是 DeepSeek-R1 中 MoE 创新的关键点。

###### 2.2.3.3.1 自适应专家路由与 RL

- DeepSeek-R1 引入了基于 RL 的专家路由，摒弃了 DeepSeek-V3 中使用的静态路由方法。与仅根据 softmax 函数计算的 token-专家亲和度选择专家不同，DeepSeek-R1 结合了学习的 RL 策略来动态分配 token 给专家。

- **数学公式：**
    - 专家选择函数被表述为一个 RL 策略优化问题，其中选择专家 $e_i$ 对于 token $t$ 的概率根据 token 嵌入 $u_t$ 动态调整：$$g_{i,t} = \pi_\theta(e_i|u_t)$$其中 $\pi_\theta$ 是基于上下文嵌入选择专家的策略网络。优化目标遵循 GRPO：$$
    J_{GRPO}(\theta) = \mathbb{E}_{q \sim P(Q), \{o_i\}_{i=1}^G \sim \pi_{\theta_{old}}} \left[ \frac{1}{G} \sum_{i=1}^G \min\left(\frac{\pi_\theta(o_i|q)}{\pi_{\theta_{old}}(o_i|q)} A_i, \text{clip}(\cdot)\right) - \beta D_{KL}(\pi_\theta || \pi_{ref}) \right]$$其中 $D_{KL}$ 正则化策略更新以防止剧烈变化。

- **实施细节：**
    - 基于 RL 的路由器通过最大化专家负载平衡和最小化路由熵来学习最佳的 token 分配。
    - 它惩罚特定专家的过载，同时激励层间的均匀激活。
    - 在路由函数中引入动态偏置项，以响应训练反馈进一步调整专家选择。

- 这种方法实现了自适应的 token-专家映射，在保持准确性的同时优化推理速度。

###### 2.2.3.3.2 分层熵门控 MoE (HE-MoE)

- DeepSeek-R1 通过引入分层熵门控 MoE (HE-MoE) 增强了 top-K MoE 路由。与仅在 token 层级应用单一 top-K 门控函数不同，DeepSeek-R1 实现了多层次的门控机制：

    - *全局选择*：首先使用 softmax 亲和度评分将 tokens 路由到一个初始的 $N_g$ 个专家池。
    - *cluster 级修剪*：在选定的专家池中，次级门控机制基于熵约束修剪专家。
    - *最终专家分配*：使用包含熵感知惩罚的调整概率函数选择 top-$K_r$ 个专家。

- 最终的门控函数被修改为：$$g_{i,t} = \frac{\text{Softmax}_i(u_t^T e_i)}{1 + \lambda H(g_{1:N_r,t})}$$其中 $H(\cdot)$ 是熵函数，$\lambda$ 控制正则化强度。

- **主要优势：**

    - *防止专家过度专精*，确保 tokens 更均匀地分布。
    - *减少模式崩溃*，防止某些专家在训练中占主导地位。
    - *动态调整稀疏性*，根据任务复杂度调整门控阈值。

###### 2.2.3.3.3 设备约束专家分配 (DCEA)

- DeepSeek-R1 在 DeepSeek-V3 的节点限制路由基础上进行改进，引入设备约束专家分配 (DCEA)，根据 GPU/TPU 的可用性和互连带宽限制专家分配。

- **算法：**
    - 每个 token 首先选择具有最高亲和度分数的设备子集。
    - 专家仅限于这些设备，从而减少设备间的同步开销。
    - 最终的专家仅在受限的设备池内选择，最大限度地减少跨节点通信。$$M = \sum_{i=1}^{N} \max\{s_{j,t} \mid j \in \text{device } i\}$$
- **结果：**
    - 跨设备通信延迟减少 35%。
    - 更稳定的训练动态，因为专家保持在本地计算节点上。
    - 较低的带宽消耗，提高训练效率。

###### 2.2.3.3.4 负载均衡专家利用与基于 RL 的调整

- 为确保均匀的负载平衡，DeepSeek-R1 引入了自适应负载路由调整，取代了 DeepSeek-V3 的辅助损失平衡策略。
- DeepSeek-R1 不再显式最小化专家平衡损失项，而是通过基于 RL 的专家选择偏差动态调整门控概率：$$
  b_i \leftarrow \begin{cases} 
  b_i - \gamma, & \text{如果专家 } i \text{ 超载} \\
  b_i + \gamma, & \text{否则}
  \end{cases}$$
- **相较于辅助损失的优势：**
  - 收敛更快，因为避免了额外的平衡约束梯度更新。
  - 更稳健的专家选择，能够在多个训练步骤中进行自适应调整。

- 这确保了一致的工作负载分布，而无需硬性辅助惩罚。

###### 2.2.3.3.5 消除 Token 丢弃策略

- 与 DeepSeek-V3 使用 token 丢弃来平衡每个设备的计算不同，DeepSeek-R1 通过动态优化专家激活阈值完全消除了 token 丢弃。
- DeepSeek-R1 不再移除低亲和度的 token，而是通过基于强化学习的专家重分配策略将 token 重新分配给其他专家。

- **优势：**
  - 在训练和推理过程中实现 100% 的 token 保留。
  - 更强的泛化能力，因为所有 token 都参与学习。
  - 无上下文信息丢失，生成更连贯的结果。

##### 2.2.3.4 比较分析

- DeepSeek-R1 是 MoE 框架的最先进版本，基于 DeepSeek-V2 和 DeepSeek-V3 的优化进行改进。以下是这三个版本中关键 MoE 特性的比较，重点突出在效率、专家路由、负载均衡和推理性能方面的提升。

| **特性**             | **DeepSeek-V2** | **DeepSeek-V3** | **DeepSeek-R1**   |
| ------------------ | --------------- | --------------- | ----------------- |
| **动态专家激活**         | ❌               | ✅（基于偏差选择）       | ✅（基于强化学习选择）       |
| **设备限制路由（DLR）**    | ✅               | ✅（节点限制路由）       | ✅（设备约束专家分配）       |
| **用于负载均衡的辅助损失**    | ✅               | ❌（基于偏差调整）       | ❌（基于强化学习的自适应均衡）   |
| **基于强化学习的路由**      | ❌               | ❌               | ✅                 |
| **专家选择的分层门控**      | ❌               | ✅               | ✅（熵感知调整）          |
| **改进的专家选择机制**      | ❌               | ✅（基于 Sigmoid）   | ✅（强化学习优化选择）       |
| **跨设备通信减少**        | ✅（设备限制路由）       | ✅（节点限制路由）       | ✅（DCEA 降低 35% 延迟） |
| **计算效率的 Token 丢弃** | ✅               | ❌（无 token 丢弃）   | ❌（无 token 丢弃）     |
| **稀疏激活策略**         | ✅（Top-K 门控）     | ✅（分层 Top-K 门控）  | ✅（分层熵门控 MoE）      |
| **训练稳定性**          | 中等              | 高               | 非常高               |
| **推理速度优化**         | 中等              | 高               | 非常高               |
| **负载均衡策略**         | 基于损失的均衡         | 基于偏差的自适应均衡      | 基于强化学习的自适应均衡      |

#### 2.2.4 数学公式

- 在 DeepSeek-R1 中，专家选择过程遵循一个门控函数：$$G(x) = \text{softmax}(W_g x)$$其中 $W_g$ 是一个可训练的权重矩阵。

- 最终输出计算为：$$y = \sum_{k \in K} G_k(x) E_k(x)$$其中：
    - $K$ 表示选择的 top-K 专家。
    - $E_k(x)$ 是专家 $k$ 执行的计算。
    - $G_k(x)$ 是门控概率。


##### 2.2.4.1 负载均衡损失

- 为确保专家的均匀利用，DeepSeek-R1 应用负载均衡损失：$$\mathcal{L}_{\text{balance}} = \lambda \sum_k \left(\frac{n_k}{N} - \frac{1}{K}\right)^2$$其中：
    - $n_k$ 是分配给专家 $k$ 的 token 数量。
    - $N$ 是批次中的总 token 数量。
    - $K$ 是每个 token 激活的专家数量。

- 此外，熵正则化项用于防止对专家的过度依赖：$$\mathcal{L}_{\text{entropy}} = -\gamma \sum_k G_k(x) \log G_k(x)$$其中 $\gamma$ 控制熵的强度。

### 2.3 多头潜注意力 (MLA)

#### 2.3.1 概述

- 多头潜在注意力（MLA）通过将键-查询-值（KQV）矩阵投影到低维潜在空间中，提高了效率，显著降低了计算和内存成本。
- MLA 中的低秩压缩技术最小化了键-值（KV）缓存的存储开销，确保了更快的推理速度，并支持更长的上下文长度或更大的批量大小。
- DeepSeek-R1 进一步优化了 MLA，通过引入强化学习增强的推理优化，同时保持低内存开销。
- 通过使用解耦的旋转位置嵌入和潜在空间压缩，MLA 确保在保持计算效率的同时，准确性降幅最小。

#### 2.3.2 关键特性

- *低秩键值压缩*：MLA 使用低秩潜在空间投影来压缩键值对，显著减少内存开销。这使得 DeepSeek-R1 仅需存储压缩表示，而不是完整的键值状态，从而实现高效的长上下文处理。
- *解耦的旋转位置嵌入（RoPE）*：标准 RoPE 引入了位置相关的变换，妨碍了键值压缩。DeepSeek-R1 将 RoPE 从键值存储中解耦，确保位置编码在不干扰潜在空间效率的情况下仍然有效。
- *具有压缩存储的高效多头注意力*：MLA 不缓存所有 token 的完整键值矩阵，而仅存储其紧凑的潜在空间等效物。这大幅降低了推理内存需求，同时保持注意力的准确性。
- *自适应投影矩阵*：MLA 利用单独学习的投影矩阵用于查询、键和值。这些矩阵在训练过程中动态调整，确保与全维注意力相比的最佳存储效率和最小准确性损失。
- *推理高效的缓存机制*：通过选择性地仅缓存压缩的键值表示，MLA 实现了比传统多头注意力（MHA）减少 93.3% 的键值缓存。这允许 DeepSeek-R1 支持更长的上下文长度，同时将推理延迟降至最低。
- *在长上下文任务中的增强性能*：DeepSeek-R1 通过 RL 驱动的优化（如 GRPO）改进了 MLA，以优先处理关键 token。这在保持计算效率的同时，提高了长上下文任务的推理准确性。

#### 2.3.3 从 DeepSeek-V2 到 DeepSeek-R1 的演变

##### 2.3.3.1 DeepSeek-V2 中的 MLA 

- DeepSeek-V2 中的 MLA 旨在通过显著减少键值缓存大小来提高推理效率，同时保持强大的模型性能。它在传统多头注意力 (MHA) 的基础上引入了多个关键创新，包括低秩键值联合压缩和解耦旋转位置嵌入。
- DeepSeek-V2 中的 MLA 实现为 DeepSeek-R1 的进一步改进奠定了基础，在 DeepSeek-R1 中，它通过 FP8 量化、增强的压缩技术和改进的数值稳定性得到了进一步优化。

###### 2.3.3.1.1 低秩键值联合压缩

- Transformer 推理的主要瓶颈之一是存储过去键和值所需的大量缓存。DeepSeek-V2 通过使用线性投影将键值表示压缩到低维潜在空间来解决这个问题。

- 给定输入 token 表示 $h_t \in \mathbb{R}^d$，标准多头注意力计算查询、键和值为：$$q_t = W_Q h_t, \, k_t = W_K h_t, \, v_t = W_V h_t$$其中 $W_Q, W_K, W_V \in \mathbb{R}^{d_{hnh} \times d}$。

- MLA 不存储全维的 $k_t$ 和 $v_t$，而是将它们压缩成潜在表示 $c_{KV}$：$$c_{KV_t} = W_{DKV} h_t$$其中 $W_{DKV} \in \mathbb{R}^{d_c \times d}$ 是一个降维投影矩阵，且 $d_c \ll d_{hnh}$。

- 在推理过程中，压缩的键值表示被扩展回可用的键和值：$$k^C_t = W_{UK} c_{KV_t}, \, v^C_t = W_{UV} c_{KV_t}$$其中 $W_{UK}, W_{UV} \in \mathbb{R}^{d_{hnh} \times d_c}$ 是升维投影矩阵。这种压缩将键值缓存大小从 $O(nh_{dhl})$ 减少到 $O(d_cl)$，其中 $l$ 是层数。

###### 2.3.3.1.2 解耦旋转位置嵌入

- RoPE 通常用于变压器架构中，将位置信息编码到查询和键中。然而，标准的 RoPE 应用与 MLA 的键值压缩不兼容，因为它引入了位置相关的变换，阻碍了高效缓存。

- DeepSeek-V2 通过将 RoPE 与键压缩解耦来解决这个问题：

  1. 引入一个辅助共享键 $k^R_t$ 和额外的多头查询 $q^R_t$。
  2. 仅对 $q^R_t$ 和 $k^R_t$ 应用 RoPE：$$q^R_t = \text{RoPE}(W_{QR}c^Q_t), \, k^R_t = \text{RoPE}(W_{KR}h_t)$$其中 $W_{QR}, W_{KR}$ 是特定于解耦 RoPE 的投影矩阵。
  3. 将压缩后的键/查询与应用了 RoPE 的键/查询连接：$$q_t = [q^C_t; q^R_t], \, k_t = [k^C_t; k^R_t]$$
     - 确保 RoPE 仅影响注意力机制的一部分，同时保持键值压缩的完整性。

###### 2.3.3.1.3 KV 缓存需求比较

- MLA 的一个关键优势是，在需要显著更少的 KV 缓存的情况下，其性能优于标准多头注意力（MHA）。下表比较了不同注意力机制的缓存大小：

| **注意力机制**            | **每个 token 的 KV 缓存（元素）** |
| -------------------- | ------------------------ |
| MHA                  | $2n h_{dh} l$            |
| GQA（分组查询）            | $2n g_{dh} l$            |
| MQA（多查询）             | $2d h l$                 |
| **MLA（DeepSeek-V2）** | $(d_c + d_R h) l$        |

- 对于 DeepSeek-V2，参数设置为：$d_c = 4d_h$，$d_R h = d_h/2$
- 这意味着 MLA 在与 2.25 组的 GQA 相似的效率下，保持了 MHA 的性能水平。

##### 2.3.3.2 DeepSeek-V3 的增强功能

- DeepSeek-V3 在多头潜在注意力（MLA）中引入了几项关键增强，显著提升了其效率、可扩展性和精度，同时保持了高模型准确性。主要改进包括：

  - 通过优化压缩技术进一步减少 KV 缓存
  - 查询压缩以节省激活内存
  - 使用 FP8 混合精度增强数值稳定性
  - MLA 中的自适应路由以实现负载均衡

- 通过这些改进，DeepSeek-V3 降低了内存开销，增强了数值精度，并在保持高模型准确性的同时实现了显著更快的推理速度。

###### 2.3.3.2.1 通过优化压缩技术进一步减少 KV 缓存

- DeepSeek-V3 的 MLA 主要增强之一是更积极地压缩 KV 缓存，同时保持模型性能。这是通过以下方式实现的：

  - *动态 KV 压缩矩阵*：DeepSeek-V3 不使用静态压缩矩阵，而是根据序列长度动态优化压缩。
  - *KV 存储的分解投影*：应用双矩阵分解对键和值进行降维投影，进一步减少 KV 存储。

###### 2.3.3.2.2 优化压缩公式

- 对于输入标记表示 $h_t \in \mathbb{R}^d$，DeepSeek-V2 中的标准 MLA 计算压缩的 KV 表示为：$$cKV_t = W_{DKV} h_t$$其中 $W_{DKV} \in \mathbb{R}^{d_c \times d}$ 是一个静态降维投影矩阵。

- 在 DeepSeek-V3 中，压缩过程通过自适应双矩阵压缩得到了增强：$$cKV_t = W_{DKV,1} W_{DKV,2} h_t$$其中 $W_{DKV,1} \in \mathbb{R}^{d_m \times d}$ 和 $W_{DKV,2} \in \mathbb{R}^{d_c \times d_m}$，$d_m$ 是一个中间维度。这种分解允许更有效的压缩，与 DeepSeek-V2 相比，存储需求减少了多达 40%。

###### 2.3.3.2.3 推理时扩展

- 在推理过程中，扩展后的键和值计算为：$$k_t^C = W_U^K W_M^K cKV_t, \quad v_t^C = W_U^V W_M^V cKV_t$$其中 $W_M^K, W_M^V$ 作为中间投影层，优化 KV 重建过程。

- 这一改进确保仅在内存中存储压缩向量，大大降低了 KV 缓存的开销。


###### 2.3.3.2.4 查询压缩以节省激活内存

- DeepSeek-V3 将 MLA 的低秩压缩扩展到查询上，减少激活内存需求，同时不影响注意力精度。

- **查询压缩公式**：

  - 传统上计算完整查询为：$$q_t = W_Q h_t, \quad k_t = W_K h_t, \quad v_t = W_V h_t$$
  - DeepSeek-V3 引入了额外的压缩步骤：$$cQ_t = W_{DQ} h_t, \quad q_t^C = W_U^Q cQ_t$$其中：
      - $cQ_t \in \mathbb{R}^{d_c'}$ 是压缩的查询表示。
      - $d_c' \ll d h_n h$，确保显著降低激活内存使用。

- **解耦旋转位置嵌入 (RoPE)**：

  - 为了保持位置嵌入的有效性，DeepSeek-V3 解耦了旋转位置嵌入 (RoPE) 的应用：$$q_t^R = \text{RoPE}(W_Q^R cQ_t), \quad k_t^R = \text{RoPE}(W_K^R h_t)$$其中：
      - $q_t^R$ 和 $k_t^R$ 存储应用 RoPE 的压缩表示。
      - 这防止 RoPE 干扰 MLA 的低秩压缩。

###### 2.3.3.2.5 激活内存的减少

- 通过查询压缩，DeepSeek-V3 将注意力激活内存减少了 35%，从而能够高效地训练大规模模型。

###### 2.3.3.2.6 增强的数值稳定性与 FP8 混合精度

- DeepSeek-V3 利用 FP8 混合精度训练，提高了数值稳定性，同时降低了内存和计算成本。

- *MLA 组件的 FP8 训练*：

  - 在 DeepSeek-V2 中，MLA 组件主要使用 BF16。DeepSeek-V3 采用细粒度的 FP8 量化，并应用每组缩放策略：
    
    - *激活缩放*：对每个 token 和每 128 个通道的 tile 进行量化。
    - *权重缩放*：128×128 的块级缩放。
    
  - 这确保了减少舍入误差，并为训练提供了更好的动态范围覆盖。

- *FP8 注意力计算*：

  - DeepSeek-V3 中的注意力输出使用 FP8 兼容缩放计算：$$o_t = \sum_{j=1}^{t} \text{Softmax}\left(\frac{q_t^T k_j}{\sqrt{d_h + d_R}}\right) v_j$$其中：
    - 激活的缩放因子在线计算。
    - 累积每 128 步升级到 FP32，以提高数值精度。

- *精度比较*：

| **组件**  | **DeepSeek-V2 (BF16)** | **DeepSeek-V3 (FP8)** |
| ------- | ---------------------- | --------------------- |
| 查询/键压缩  | $d_c = 4d_h$           | $d_c = 3d_h$          |
| KV 缓存存储 | BF16                   | FP8                   |
| RoPE 应用 | 全精度                    | 解耦，FP8                |
| 注意力计算   | BF16                   | FP8 + FP32 累积         |

- 通过利用 FP8 量化，DeepSeek-V3 实现了 2.3 倍的训练效率提升，降低了内存消耗，同时不影响性能。

###### 2.3.3.2.7 自适应路由用于 MLA 的负载均衡

- DeepSeek-V3 通过引入动态负载均衡来提高查询-键计算的注意力效率。

- *负载自适应路由机制*：

  - 在 DeepSeek-V2 中，MLA 使用静态注意力头分配，处理大型序列时可能导致计算效率低下。
  - DeepSeek-V3 通过自适应路由进行改进：$$s_{i,t} = \text{Sigmoid}(u_t^T e_i + b_i)$$其中：
      - $e_i$ 是路由专家的中心向量。
      - $b_i$ 是动态更新的偏置项，用于调整每个头的工作负载平衡。
    
  - 偏置项更新为：$$b_i^{(t+1)} = b_i^{(t)} - \gamma \cdot (\text{overloaded}_i - \text{underloaded}_i)$$其中 $\gamma$ 是调节参数。
    
  - 这确保了：
    - 注意力头之间的 token 分布平衡。
    - 推理过程中不丢失 token，防止效率损失。

- *计算收益*：

  - 通过集成自适应路由，DeepSeek-V3 实现了：
    - 注意力头之间的计算负载均匀。
    - 每个 token 推理延迟减少 10%。


##### 2.3.3.3 DeepSeek-R1 的增强

- DeepSeek-R1 对 MLA 进行了多项改进，提升了推理效率和性能，同时保持低内存开销。基于 DeepSeek-V3 中的 MLA 优化，DeepSeek-R1 进一步增强了 KQV 压缩、基于强化学习的注意力分配和数值稳定性机制。

###### 2.3.3.3.1 RL引导的潜在注意力优化

- DeepSeek-R1 将强化学习技术集成到 MLA 中，通过 GRPO 优化注意力机制。与之前的确定性注意力策略不同，DeepSeek-R1 根据强化奖励动态调整注意力权重，优先考虑对推理路径贡献更大的 tokens。
- GRPO 消除了对单独评论模型的需求，降低了内存开销并提高了收敛效率。
- GRPO 直接从组级奖励估计优势值，而不是依赖监督微调：$$A_i = \frac{r_i - \text{mean}(\{r_1, r_2, \ldots, r_G\})}{\text{std}(\{r_1, r_2, \ldots, r_G\})}$$
- 策略模型 $\pi_\theta$ 通过最大化以下目标进行更新：$$J_{\text{GRPO}}(\theta) = \mathbb{E}\left[\sum_{i=1}^G \min\left(\frac{\pi_\theta(o_i|q)}{\pi_{\theta_{\text{old}}}(o_i|q)}A_i, \text{clip}\left(\frac{\pi_\theta(o_i|q)}{\pi_{\theta_{\text{old}}}(o_i|q)}, 1-\epsilon, 1+\epsilon\right)A_i\right) - \beta D_{\text{KL}}(\pi_\theta || \pi_{\text{ref}})\right]$$
- 这种方法使 DeepSeek-R1 能够自适应地优化 MLA 中的注意力机制，改善长上下文推理中的 token 优先级。

###### 2.3.3.3.2 自适应查询和键压缩通过RL

DeepSeek-R1 的 MLA 中的一个主要增强是通过 RL 引导的自适应查询和键压缩。DeepSeek-V3 已经引入了用于 KV 存储的低秩压缩技术，而 DeepSeek-R1 将压缩扩展到查询上，减少了激活内存而不影响注意力精度。

- *优化的压缩公式*：
    
    - 在 DeepSeek-V3 中，KV 缓存压缩是通过静态低秩投影实现的：$$c_{KV_t} = W_{DKV} h_t$$
    - DeepSeek-R1 在推理过程中使用基于 RL 的奖励最大化动态调整压缩矩阵：$$c_{KV_t} = W_{DKV,1} W_{DKV,2} h_t$$其中：
	    - $W_{DKV,1} \in \mathbb{R}^{d_m \times d}$ 和 $W_{DKV,2} \in \mathbb{R}^{d_c \times d_m}$。
        - $d_m$ 是一个中间维度，允许更细粒度的潜在空间表示。

- *推理时扩展*：
    
    - DeepSeek-R1 使用多阶段扩展管道，而不是单一的上投影矩阵：$$k_t^C = W_{UK} W_{MK} c_{KV_t}, \quad v_t^C = W_{UV} W_{MV} c_{KV_t}$$其中 $W_{MK}, W_{MV}$ 优化重构后的查询-键值，确保仅压缩向量存储在内存中。

- *压缩比改进*：DeepSeek-R1 在保持查询-键检索精度的同时，相比 DeepSeek-V3 进一步减少了 25% 的 KV 缓存需求。

###### 2.3.3.3.3 解耦旋转位置嵌入与上下文特定缩放

- 虽然 DeepSeek-V3 引入了解耦 RoPE 来将位置编码与压缩的键值表示分离，但 DeepSeek-R1 通过上下文特定的缩放机制进一步优化了 RoPE。
- DeepSeek-R1 采用增强的 RoPE 公式，使其具备上下文感知能力，根据序列长度动态调整缩放因子：$$\lambda_t = \frac{1}{\sqrt{1 + \alpha L_t}}$$其中：
    - $\lambda_t$ 是位置嵌入的自适应缩放因子。
    - $\alpha$ 是通过 RL 优化学习的超参数。
    - $L_t$ 表示时间步长 $t$ 的序列长度。

- **实现优势**：
    - RoPE 缩放确保在不同序列长度下的一致注意力对齐。
    - 防止在压缩 MLA 的键值状态时位置信息的退化。

###### 2.3.3.3.4 FP8 混合精度提升 MLA 稳定性

- DeepSeek-R1 采用 FP8 量化进行 MLA 计算，相较于 DeepSeek-V3 的 BF16 方法，进一步提高了数值稳定性。
- 在 DeepSeek-R1 的精度感知计算流程中，QKV 矩阵通过每组缩放动态量化：$$\tilde{Q} = s_Q Q, \quad \tilde{K} = s_K K, \quad \tilde{V} = s_V V$$
- 其中 $s_Q, s_K, s_V$ 是每组学习的缩放因子。
- 注意力输出通过混合精度累积计算：$$o_t = \sum_{j=1}^{t} \text{Softmax}\left(\frac{\tilde{q}_t^T \tilde{k}_j}{\sqrt{d_h + d_R}}\right) \tilde{v}_j$$
- 累积过程每 128 步升级为 FP32，以确保更好的数值精度，同时保持 FP8 的效率。

- *MLA 精度策略比较*：

| **组件**  | **DeepSeek-V3 (BF16)** | **DeepSeek-R1 (FP8)** |
| ------- | ---------------------- | --------------------- |
| 查询/键压缩  | $d_c = 4d_h$           | $d_c = 3d_h$          |
| KV 缓存存储 | BF16                   | FP8                   |
| RoPE 应用 | 全精度                    | 解耦，FP8                |
| 注意力计算   | BF16                   | FP8 + FP32 累积         |

- **效率提升**：
    - FP8 将内存占用减少约 40% 相比 BF16。
    - 在长上下文任务中实现 2.3 倍的推理吞吐量提升。


###### 2.3.3.3.5 自适应/动态路由实现负载均衡注意力

- DeepSeek-R1 引入了负载均衡的自适应路由机制，确保注意力头之间的查询-键计算均匀。
- DeepSeek-R1 使用基于 sigmoid 的路由函数优化每个头的工作负载平衡：$$s_{i,t} = \text{Sigmoid}(u_t^T e_i + b_i)$$其中：
	- $e_i$ 表示路由注意力专家的中心向量。
	- $b_i$ 是自适应偏置项，确保工作负载均匀。

- **性能提升**：
  - 平衡各头的计算，防止瓶颈。
  - 将每个 token 的推理延迟减少 10%。

##### 2.3.3.4 比较分析

- DeepSeek-V2 引入了多头潜在注意力 (MLA)，具有显著的 KV 缓存压缩、解耦 RoPE 和基本低秩投影以提高效率。DeepSeek-V3 在此基础上进一步减少了 KV 缓存大小，优化了查询压缩，并引入了 FP8 混合精度以增强数值稳定性。DeepSeek-R1 通过集成 RL 技术（如组相对策略优化 GRPO）进一步优化 MLA，实现动态注意力分配。最新的 DeepSeek-R1 改进了推理延迟和内存效率，使其成为迄今为止最优化的 MLA 版本。
- 下表对 DeepSeek-V2、DeepSeek-V3 和 DeepSeek-R1 的 MLA 进行了比较分析，突出显示了各版本在压缩技术、精度、路由机制和推理效率方面的关键改进。

| **特性**        | **DeepSeek-V2** | **DeepSeek-V3** | **DeepSeek-R1**          |
| ------------- | --------------- | --------------- | ------------------------ |
| **低秩 KV 压缩**  | ✅               | ✅（优化的分解投影）      | ✅（RL 优化的自适应压缩）           |
| **查询压缩**      | ❌               | ✅（静态低秩查询压缩）     | ✅（RL 引导的动态查询压缩）          |
| **KV 缓存减少**   | ✅（减少 93.3%）     | ✅（进一步减少 40%）    | ✅（比 V3 进一步减少 25%）        |
| **RoPE 应用**   | ✅（解耦 RoPE）      | ✅（解耦并具有上下文特定缩放） | ✅（增强的上下文感知缩放）            |
| **精度格式**      | BF16            | FP8（细粒度混合精度）    | FP8（每组缩放，FP32 累积）        |
| **MLA 自适应路由** | ❌               | ✅（静态自适应路由）      | ✅（负载均衡动态路由）              |
| **推理延迟减少**    | ✅（KV 压缩减少延迟）    | ✅（比 V2 快 10%）   | ✅（比 V3 快 10%）            |
| **RL 增强**     | ❌               | ❌               | ✅（GRPO 用于自适应 MLA 优化）     |
| **数值稳定性改进**   | ✅（基本稳定性增强）      | ✅（FP8 混合精度）     | ✅（FP8 与 RL 引导的稳定机制）      |
| **长上下文性能**    | ✅（支持更长的上下文）     | ✅（进一步优化）        | ✅（通过 RL 引导的 token 优先级增强） |

#### 2.3.4 实现

* DeepSeek-R1 中的多头潜在注意力 (MLA) 实现了多项优化，旨在在保持准确性的同时最大化效率。本节详细介绍了 MLA 的核心机制，包括键值压缩、查询转换、位置编码和计算优化。

##### 2.3.4.1 背景：标准多头注意力 (MHA)

- 对于标准多头注意力机制 (MHA)，键 (K)、查询 (Q) 和值 (V) 矩阵的计算如下：$$K, Q, V = W_kX, W_qX, W_vX$$其中 $W_k, W_q, W_v$ 是键、查询和值投影的权重矩阵。

- 注意力权重计算为：$$A = \text{Softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)$$
  - 输出为：$O = AV$

- 这需要在推理过程中存储完整的键值缓存，导致显著的内存开销。

##### 2.3.4.2 低秩键值联合压缩

- MLA 的一个基本优化是将键值对压缩到低维潜在空间，从而显著减少内存开销。具体如下：

    - *压缩机制*：
        - 在投影回各自维度之前，键和值表示被压缩到一个共享的潜在空间。这通过两步转换实现：$$\begin{align}cKV_t &= W_{DKV} h_t \\ kC_t &= W_{UK} cKV_t \\ vC_t &= W_{UV} cKV_t\end{align}$$其中：
            - $cKV_t \in \mathbb{R}^{d_c}$ 是压缩的潜在表示。
            - $W_{DKV} \in \mathbb{R}^{d_c \times d}$ 是降维投影矩阵。
            - $W_{UK}, W_{UV} \in \mathbb{R}^{d_{hnh} \times d_c}$ 分别是键和值的升维投影矩阵。

    - *内存减少*：
        - 不再为每个标记存储完整大小的键和值，仅缓存 $cKV_t$。
        - 内存占用的减少使得 DeepSeek-R1 能以较低的计算成本处理显著更长的序列。

##### 2.3.4.3 多级压缩

- DeepSeek-R1 通过引入额外的转换层改进了压缩机制，实现了多级压缩方法。具体如下：

    - *额外的投影层*：

        - 为了进一步减少存储成本，引入了一个二次压缩层：$$c'KV_t = W_{DKV2} f(W_{DKV} h_t)$$其中：
            - $W_{DKV2} \in \mathbb{R}^{d'_c \times d_c}$ 是第二个降维投影矩阵。
            - $f(\cdot)$ 是一个用于改善表示学习的非线性激活函数。
            - $d'_c < d_c$ 确保了更小的 KV 缓存大小。

    - *性能优势*：
        - 这一步进一步减少了 KV 存储，同时保持了注意力机制所需的足够信息。
        - 实验表明，与 DeepSeek-V3 相比，这可以减少 10-15% 的内存占用。


##### 2.3.4.4 查询压缩与优化

- 与键和值类似，查询也被压缩，从而在训练期间实现高效计算和减少激活内存。具体如下：

    - *查询转换*：

        - 查询经过类似于键和值的两步转换：$$\begin{align}cQ_t &= W_{DQ} h_t \\ qC_t &= W_{UQ} cQ_t \end{align}$$其中：
            - $W_{DQ} \in \mathbb{R}^{d'_c \times d}$ 是查询的降维投影矩阵。
            - $W_{UQ} \in \mathbb{R}^{d_{hnh} \times d'_c}$ 将压缩的查询表示映射回其原始维度。

    - *多层查询优化*：
        - DeepSeek-R1 通过额外的自适应缩放层优化查询投影。
        - 在微调过程中，使用强化学习动态调整转换矩阵 $W_{DQ}$ 和 $W_{UQ}$。

##### 2.3.4.5 解耦旋转位置嵌入（RoPE）

- 为了确保稳健的长上下文处理，DeepSeek-R1 以解耦的方式应用 RoPE，将位置编码与潜在注意力机制分开。具体如下：

    - *键和值的独立位置编码*：$$\begin{align} kR_t &= \text{RoPE}(W_{KR} h_t) \\ qR_t &= \text{RoPE}(W_{QR} cQ_t) \end{align}$$其中：
	    - $W_{KR} \in \mathbb{R}^{d_{Rh} \times d}$ 生成键的位置信息。
	    - $W_{QR} \in \mathbb{R}^{d_{Rhnh} \times d'_c}$ 生成查询的位置信息。
	    - RoPE 转换确保相对位置信息得以保留，同时使 KV 缓存保持紧凑。

    - *DeepSeek-R1 中 RoPE 的计算效率*：

        - RoPE 的应用被延迟到查询-键交互的最终阶段，避免不必要的内存膨胀。
        - 与 DeepSeek-V2 和 V3 相比，DeepSeek-R1 实现了 25% 更快的查询-键检索速度。


##### 2.3.4.6 MLA 中的注意力计算

- 在 MLA 中，最终的注意力输出通过整合压缩的键、查询和值，在修改后的注意力机制中计算得出。具体如下：

    - *修改后的注意力分数*：
        - 注意力分数通过压缩的潜在键和显式位置编码计算：$$A_{t,j,i} = \frac{{q_t^T k_j}}{\sqrt{d_h + d_R}}$$
        - 这种公式确保位置嵌入按比例影响注意力强度。

    - *加权值聚合*：
        - 注意力输出计算为：$$o_{t,i} = \sum_{j=1}^{t} \text{Softmax}_j(A_{t,j,i}) v_{Cj,i}$$
        - Softmax 操作在序列中对注意力分数进行归一化。

    - *最终输出投影*：
        - 最终输出通过以下方式获得：$$u_t = W_O [o_{t,1}; o_{t,2}; \ldots; o_{t,nh}]$$其中：
            - $W_O$ 是输出投影矩阵，用于将连接的注意力输出映射回完整的嵌入空间。




- The final attention output in MLA is computed by integrating compressed keys, queries, and values in a modified attention mechanism. Specifics below:
    
    - **Modified Attention Scores**:
        - The attention scores are computed using both compressed latent keys and explicit positional encodings:
            
            At,j,i=qTt,ikj,idh+dR‾‾‾‾‾‾‾√At,j,i=qt,iTkj,idh+dR
            
        - This formulation ensures that positional embeddings contribute proportionally to attention strength.
            
    - **Weighted Value Aggregation**:
        - The attention output is computed as:
            
            ot,i=∑j=1tSoftmaxj(At,j,i)vCj,iot,i=∑j=1tSoftmaxj(At,j,i)vCj,i
            
        - The softmax operation normalizes the attention scores across the sequence.
            
    - **Final Output Projection**:
        - The final output is obtained via:
            
            ut=WO[ot,1;ot,2;...;ot,nh]ut=WO[ot,1;ot,2;...;ot,nh]
            
            - where:
                - WOWO is the output projection matrix mapping the concatenated attention outputs back to the full embedding space.

##### 2.3.4.7 RL-Optimized MLA

DeepSeek-R1 结合了强化学习以进一步优化 MLA 的变换矩阵。具体如下：

- *使用强化学习进行微调*：
    - 使用 GRPO，MLA 根据高效的内存使用和检索准确性获得奖励。
    - 策略更新方程为：$$
    J_{\text{GRPO}}(\theta) = \mathbb{E}\left[\sum_{i=1}^{G} \min\left(\frac{\pi_{\theta}(o_i|q)}{\pi_{\theta_{\text{old}}}(o_i|q)} A_i, \text{clip}\left(\frac{\pi_{\theta}(o_i|q)}{\pi_{\theta_{\text{old}}}(o_i|q)}, 1-\epsilon, 1+\epsilon\right) A_i\right)\right]$$其中：
	    * $\pi_{\theta}$ 代表更新后的策略。
	    * $A_i$ 是引导优化的优势函数。



##### 2.3.4.8 计算和硬件优化

- *推理时效率*：
    - DeepSeek-R1 中的 MLA 采用张量并行计算，实现跨 GPU 的吞吐量优化。
    - 通过低精度 KV 存储（FP8 格式）最小化内存开销。

- *跨节点通信优化*：
    - 使用优化的全对全通信内核，充分利用 InfiniBand 和 NVLink 带宽。
    - 将节点间通信延迟减少 30%，提升分布式推理性能。

##### 2.3.4.9 效率对比分析

| **注意力机制**                   | **每个Token的KV缓存** | **计算复杂度**   | **性能影响**     |
| --------------------------- | ---------------- | ----------- | ------------ |
| **MHA（标准）**                 | $O(Nd_h)$        | $O(N^2d_h)$ | 高精度，高成本      |
| **MQA**                     | $O(d_h)$         | $O(Nd_h)$   | 内存较低，性能下降    |
| **GQA**                     | $O(gd_h)$（组）     | $O(Nd_h)$   | 适度平衡         |
| **MLA（DeepSeek-V2）**        | $O(dL)$          | $O(NdL)$    | 高效，损失极小      |
| **MLA + 分层缓存（DeepSeek-R1）** | $O(dL)$（可重用）     | $O(NdL)$    | **最高效，保持性能** |


### 2.4 多 token 预测 (MTP)

#### 2.4.1 概述

- 多 token 预测（MTP）使 DeepSeek-R1 能够并行预测多个标记，显著提高推理速度。

#### 2.4.2 关键特性

- *并行多标记预测*：DeepSeek-R1通过同时预测多个标记来提高推理速度，而不是按顺序进行。这减少了解码延迟，允许更快的文本生成，同时保持连贯性。
- *跨深度残差连接*：与仅依赖先前模块输出的DeepSeek-V3不同，DeepSeek-R1在多标记预测层之间整合了残差连接。这使得更深层的MTP模块能够利用早期深度的特征，改善长期依赖性。
- *自适应预测粒度*：模型根据输入序列的复杂性动态调整每个模块预测的未来标记数量。这确保了短上下文的细粒度预测，并在处理较长序列时提供更广泛的前瞻性。
- *深度感知损失加权*：DeepSeek-R1通过使用基于Sigmoid的加权函数优先考虑中等深度的MTP层，从而优化训练目标。这通过在最有影响的地方进行更多梯度更新来提高学习效率。
- *内存高效参数共享*：模型通过在MTP层间重用Transformer层来减少内存消耗。DeepSeek-R1应用深度条件路由，最小化冗余计算，同时保持独特的深度表示。
- *优化的推测解码*：DeepSeek-R1通过引入概率一致性检查来改进推测解码。预测基于置信度阈值被接受，而不是要求精确匹配，从而降低拒绝率，加速推理。
- *训练和推理的实证增益*：得益于这些改进，DeepSeek-R1实现了**22%更快的训练收敛**，**1.5倍的生成速度提升**，以及**18%更好的长文本困惑度**，显示出其相对于 DeepSeek-V3 的优越性。

#### 2.4.3 Evolution from DeepSeek-V3 to DeepSeek-R1

##### MTP in DeepSeek-V3

- MTP was is introduced in DeepSeek-V3 as a training objective to improve data efficiency and predictive capabilities by enabling the model to anticipate multiple future tokens at each position. Unlike conventional next-token prediction, which limits training to a single-step forward prediction, MTP extends this scope to multiple future tokens, thereby densifying training signals and enhancing long-term coherence in text generation.
- DeepSeek-V3 implements MTP using a structured pipeline with several key design choices, including sequential prediction modules, shared embeddings and output heads, and a hierarchical loss formulation. These innovations improve model performance, enable speculative decoding, and enhance overall data efficiency. DeepSeek-R1 further builds on these foundations, optimizing MTP implementation for improved reasoning tasks.
- The following sub-sections detail the features introduced in DeepSeek-V3 to support MTP.

###### Sequential Multi-Token Prediction Modules

- DeepSeek-V3 employs DD sequential MTP modules, where each module is responsible for predicting an additional future token. Instead of parallelly predicting future tokens with independent output heads (as in [Better & Faster Large Language Models via Multi-token Prediction](https://arxiv.org/abs/2404.19737) by Gloeckle et al., 2024), DeepSeek-V3 maintains a causal chain across prediction depths, ensuring each token is conditioned on prior MTP module outputs.
- For the kthkth MTP module, the representation of the ithith input token at depth kk is computed as:
    
    h′(k)i=Mk[RMSNorm(h(k−1)i);RMSNorm(Emb(ti+k))]hi′(k)=Mk[RMSNorm(hi(k−1));RMSNorm(Emb(ti+k))]
    
    - where:
        - h(k−1)ihi(k−1) is the representation from the previous depth (or from the main model when k=1k=1).
        - Mk∈ℝd×2dMk∈Rd×2d is the projection matrix.
        - _Emb(⋅⋅)_ is the shared embedding function.
- Each module applies a transformer block:
    
    h(k)1:T−k=TRMk(h′(k)1:T−k)h1:T−k(k)=TRMk(h1:T−k′(k))
    
    - where TT is the input sequence length. The output of this module is passed to a shared output head:
    
    P(k)i+k+1=OutHead(h(k)i)Pi+k+1(k)=OutHead(hi(k))
    
    - where P(k)i+k+1Pi+k+1(k) is the probability distribution for the _k_-th future token.

###### MTP Training Objective

- For each prediction depth kk, DeepSeek-V3 computes a cross-entropy loss:
    
    L(k)MTP=−1T∑i=2+kT+1logP(k)i[ti]LMTP(k)=−1T∑i=2+kT+1log⁡Pi(k)[ti]
    
    - where titi is the ground-truth token at position ii, and P(k)i[ti]Pi(k)[ti] is the predicted probability for that token. The overall MTP loss is the mean of losses across all depths, scaled by a factor λλ:
    
    LMTP=λD∑k=1DL(k)MTPLMTP=λD∑k=1DLMTP(k)
    
    - where DD is the number of MTP modules.

###### Memory Optimization with Shared Embeddings and Output Heads

- To minimize additional memory costs from MTP modules, DeepSeek-V3:
    - Shares embeddings across MTP modules.
    - Uses a single shared output head instead of independent ones for each MTP depth.
    - Implements weight sharing between the primary model and MTP modules.
- This design ensures that additional forward passes in MTP training do not substantially increase parameter storage requirements.

###### Inference Strategy and Speculative Decoding

- While MTP is primarily used to improve training, DeepSeek-V3 also explores the use of MTP modules for speculative decoding at inference time. The idea is to use the additional token predictions as speculative completions, similar to methods proposed in [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192) by Leviathan et al. (2023):
    
    1. The primary model predicts token ti+1ti+1 as usual.
    2. The first MTP module simultaneously predicts ti+2ti+2, allowing early validation of token coherence.
    3. If MTP predictions match beam search results, multiple tokens can be emitted at once.
- This strategy significantly accelerates inference while maintaining output fluency.
    

###### Ablation Studies on Multi-Token Prediction

- DeepSeek-V3 conducts detailed ablation studies to assess the impact of MTP. Key findings include:
    - **Impact on Training Efficiency**: Training with MTP leads to a 15% improvement in data efficiency, allowing models to learn more per token.
    - **Effect on Long-Term Coherence**: Models trained with MTP exhibit a higher perplexity improvement at longer sequence lengths compared to traditional next-token prediction.
    - **Influence on Speculative Decoding Accuracy**: The inclusion of MTP modules in decoding reduces rejection rates in speculative generation by 35%, enhancing latency benefits.

##### Enhancements in DeepSeek-R1

- DeepSeek-R1 introduces significant advancements in MTP, building upon the structured MTP framework established in DeepSeek-V3. The improvements primarily focus on better token dependency modeling, adaptive prediction granularity, loss function refinement, memory-efficient parameter sharing, and optimized inference strategies. These enhancements enable DeepSeek-R1 to achieve superior reasoning capability, enhanced training efficiency, and significantly reduced inference latency. Below, we detail each feature.

###### Improved Token Dependency Modeling in MTP

- DeepSeek-R1 enhances the sequential nature of MTP modules by incorporating cross-depth residual connections between MTP layers. Unlike DeepSeek-V3, where each MTP module strictly predicts tokens conditioned only on prior module outputs, DeepSeek-R1 introduces depth-wise feature aggregation to facilitate richer information propagation.
    
- The updated token representation at the _k_-th depth is computed as:
    
    h′(k)i=Mk[RMSNorm(h(k−1)i);RMSNorm(Emb(ti+k));Res(h(k−2)i)]hi′(k)=Mk[RMSNorm(hi(k−1));RMSNorm(Emb(ti+k));Res(hi(k−2))]
    
    - where:
        - Res(h(k−2)i)Res(hi(k−2)) is a residual connection from two depths earlier, weighted by a learnable scalar αkαk:
            
            Res(h(k−2)i)=αk⋅h(k−2)iRes(hi(k−2))=αk⋅hi(k−2)
            
- This modification ensures that deeper MTP modules receive contextualized features from multiple depths, leading to improved coherence in multi-step predictions.
    

###### Adaptive Prediction Granularity

- DeepSeek-R1 refines MTP’s granularity by dynamically adjusting the number of future tokens predicted per module based on the context length and complexity of the input. Instead of fixing the number of predicted tokens per step, DeepSeek-R1 adapts the prediction horizon dynamically.
    
- The number of future tokens predicted at depth kk is given by:
    
    Nk=min(⌊γk⋅T⌋,D−k)Nk=min(⌊γk⋅T⌋,D−k)
    
    - where:
        - γkγk is a learnable scaling factor that determines adaptive granularity.
        - TT is the sequence length.
        - DD is the maximum MTP depth.
- **Intuition:** In early sequence regions, shorter horizons (1-2 future tokens) are preferred for precise token alignment, whereas deeper into the sequence, the model extends the prediction horizon, increasing efficiency without sacrificing accuracy.
    

###### Loss Function Refinement for Multi-Depth Learning

- DeepSeek-R1 improves the MTP loss formulation by introducing depth-aware weighting to prioritize learning at certain depths. In DeepSeek-V3, all depths were weighted equally, leading to inefficient optimization at extreme depths.
    
- The new depth-weighted MTP loss is formulated as:
    
    LMTP=λD∑k=1Dwk⋅L(k)MTPLMTP=λD∑k=1Dwk⋅LMTP(k)
    
    - where:
        - wkwk is a depth-dependent weighting factor:
            
            wk=11+e−β(k−D/2)wk=11+e−β(k−D/2)
            
        - This sigmoid-based weighting ensures that mid-range MTP depths receive stronger gradient signals, leading to better-balanced learning across depths.
            

###### Optimized Memory Efficiency with Parameter Sharing

- One major enhancement in DeepSeek-R1 is the parameter sharing strategy across MTP modules, significantly reducing memory overhead while maintaining distinct depth-wise representations.
    
- Instead of maintaining separate transformer layers for each MTP depth as in DeepSeek-V3, DeepSeek-R1 re-uses the main model’s layers with depth-conditioned routing.
- The token representation at depth kk is now passed through a single, shared transformer layer with an additional depth-embedding:
    
    h(k)1:T−k=TRM(h′(k)1:T−k,DepthEmb(k))h1:T−k(k)=TRM(h1:T−k′(k),DepthEmb(k))
    
- The depth embedding DepthEmb(k)DepthEmb(k) ensures that different MTP layers retain unique learned behaviors while leveraging the same computational graph.

###### Enhanced Inference Strategy with Speculative Decoding

- DeepSeek-R1 significantly refines the speculative decoding strategy introduced in DeepSeek-V3 by allowing adaptive token validation. Specifics below:
    
    - In DeepSeek-V3, speculative decoding was limited to greedy agreement checking, where only exact matches between MTP predictions and main model outputs were used to accelerate inference.
    - DeepSeek-R1 introduces probabilistic agreement checking, where a predicted token t̂ i+2t^i+2 from MTP is accepted if:
        
        P(1)MTP(t̂ i+2)>τPMain(t̂ i+2)PMTP(1)(t^i+2)>τPMain(t^i+2)
        
        - where:
            - P(1)MTP(t̂ i+2)PMTP(1)(t^i+2) is the MTP module’s probability of the token.
            - PMain(t̂ i+2)PMain(t^i+2) is the main model’s probability.
            - ττ is a tunable acceptance threshold.
    - **Impact:** This strategy allows high-confidence speculative predictions to be used even when they do not perfectly match the main model’s top prediction, reducing rejection rates by over 40%, accelerating inference.

###### Empirical Gains from DeepSeek-R1’s MTP Enhancements

- DeepSeek-R1’s refinements to MTP result in significant empirical gains over DeepSeek-V3:
    
    - **Training Efficiency:** Training convergence improved by 22% due to depth-weighted loss prioritization.
    - **Inference Speed:** Speculative decoding optimizations resulted in a 1.5× faster generation speed.
    - **Long-Term Coherence:** Perplexity on long-form text improved by 18%, showing that the revised token dependency modeling enhances context retention over long horizons.

##### Comparative Analysis

- DeepSeek-R1 builds upon DeepSeek-V3’s foundational MTP structure while addressing its limitations. The improvements, particularly in adaptive granularity, loss function optimization, and speculative decoding, result in faster, more coherent, and memory-efficient predictions. These refinements collectively enhance DeepSeek-R1’s reasoning capability and inference performance. The table below provides a comparative summary of key MTP features in DeepSeek-V3 and DeepSeek-R1.

|**Feature**|**DeepSeek-V3**|**DeepSeek-R1**|
|---|---|---|
|Sequential MTP Modules|✅ Structured pipeline with sequential depth modules|✅ Enhanced with cross-depth residual connections|
|Shared Embeddings for MTP|✅ Shared token embeddings across modules|✅ Further optimized with depth-conditioned routing|
|Prediction Granularity|❌ Fixed number of future token predictions per module|✅ Adaptive token horizon based on sequence complexity|
|Loss Function Optimization|❌ Uniform loss weighting across MTP depths|✅ Depth-aware weighting for optimized learning|
|Memory Optimization Strategy|✅ Shared output heads for reduced memory footprint|✅ Further improved with depth-conditioned layer sharing|
|Inference Speed Boost via MTP|✅ Basic speculative decoding|✅ Probabilistic speculative decoding, reducing rejection rates by 40%|
|Training Efficiency Improvement|✅ 15% increase in data efficiency|✅ 22% faster convergence with improved loss prioritization|
|Long-Term Coherence in Predictions|✅ Improved over next-token prediction models|✅ 18% better perplexity in long-form text|
|Speculative Decoding Acceptance Strategy|❌ Strict token match required for validation|✅ Probabilistic validation based on confidence threshold|
|Impact on Latency Reduction|✅ Moderate improvement in decoding speed|✅ 1.5× faster inference due to reduced rejection rates|

#### Implementation Details

- DeepSeek-R1 incorporates an advanced MTP strategy to boost decoding efficiency and reduce latency. Unlike traditional autoregressive decoding, where each token is predicted sequentially, MTP allows multiple tokens to be predicted per decoding step. This is achieved through a hierarchical approach that balances performance improvements with the risk of error propagation. Specifics below:
    
    1. **Multi-Layer Representation Propagation**:
        - DeepSeek-R1’s transformer architecture is enhanced to support simultaneous token prediction across multiple layers.
        - Each layer in the model computes token probabilities independently while maintaining consistency across the sequence.
    2. **Speculative Decoding with Verification**:
        - During inference, DeepSeek-R1 generates speculative multi-token sequences and verifies their coherence through a hierarchical token verification mechanism.
        - This approach dynamically adjusts the number of tokens predicted in each step based on confidence scores, ensuring that low-confidence tokens are reevaluated before finalizing outputs.
    3. **Training Objective**:
        - The model is trained with a combination of standard cross-entropy loss for next-token prediction and an auxiliary loss that encourages parallel token prediction.
        - The loss function is formulated as:
            
            LMTP=λ∑k=1DLCE(Pk,Tk)LMTP=λ∑k=1DLCE(Pk,Tk)
            
            - where DD is the number of parallel tokens predicted per step, and LCELCE represents the cross-entropy loss for each predicted token.
    4. **Adaptive Token Selection with RL**:
        - DeepSeek-R1 employs an RL-based approach to refine multi-token predictions, ensuring that higher-quality token sequences are prioritized.
        - The RL framework assigns rewards based on coherence, fluency, and alignment with ground-truth data.
        - This RL-driven strategy effectively reduces hallucinations and improves long-range coherence in generated text.
    5. **Memory and Compute Efficiency**:
        - The MTP module is optimized to minimize additional memory overhead, leveraging weight-sharing mechanisms within transformer layers.
        - The speculative decoding mechanism integrates efficiently with DeepSeek-R1’s caching strategy, ensuring that redundant computations are avoided.

##### Mathematical Formulation

- The prediction function follows an autoregressive formulation:

P(yt|x)=∏t=1TP(yt|y<t,x)P(yt|x)=∏t=1TP(yt|y<t,x)

- By introducing parallel decoding, DeepSeek-R1 reduces inference complexity from O(T)O(T) to O(Tk)O(Tk), where kk is the number of tokens predicted per step.

## DeepSeek-R1-Zero →→ Training Pipeline: Pure Reinforcement Learning in DeepSeek-R1-Zero

- DeepSeek-R1-Zero explores the radical idea that structured reasoning capabilities can be learned from scratch using RL alone—without any supervised fine-tuning (SFT) as a preliminary step. This novel approach bypasses the need for curated datasets and instead incentivizes reasoning behaviors directly through reward signals. While this results in impressive emergent behaviors, it also introduces challenges in output quality and stability.
    
- The training of DeepSeek-R1-Zero proceeds as a single-stage RL pipeline, where the model begins from a base LLM (DeepSeek-V3-Base) and is optimized end-to-end via Group Relative Policy Optimization (GRPO). This framework eliminates the need for a value model and instead leverages group-based normalization to compute relative advantages, reducing both training overhead and complexity.
    
- **Key components of DeepSeek-R1-Zero’s pipeline:**
    
    1. **No Supervised Fine-Tuning (SFT)**
        - Training begins directly from the pre-trained DeepSeek-V3-Base without any cold-start data.
        - This setup enables researchers to study the self-evolving nature of reasoning in LLMs purely through trial-and-error and reward shaping.
    2. **Reinforcement Learning with GRPO**
        - GRPO is used to optimize the model’s outputs without requiring a critic model.
        - It computes advantages by normalizing rewards across a batch of responses for a given prompt.
        - The reward function is entirely rule-based (rather than a neural model), avoiding reward hacking and expensive retraining.
    3. **Reward Modeling**
        - Two core reward types guide the learning process:
            - **Accuracy Rewards**: Evaluate correctness of responses, particularly for tasks with deterministic answers like math or code.
            - **Format Rewards**: Encourage the model to wrap its reasoning in a structured format using `<think>` and `<answer>` tags.
        - No neural reward models are used, emphasizing transparency and training stability.
    4. **Template-Guided Output Formatting**
        - Prompts follow a simple template instructing the model to first “think” through the problem and then produce an answer.
        - This structure promotes reasoning traceability but does not constrain specific problem-solving strategies.
- **Emergent Behaviors and Self-Evolution**
    - Over the course of training, DeepSeek-R1-Zero gradually learns to extend its reasoning steps, revisiting previous thoughts and experimenting with longer CoTs.
    - The model exhibits behaviors such as reflection, self-verification, and longer test-time computation without being explicitly taught these strategies.
    - This culminates in striking “aha moments,” where the model demonstrates sudden improvements in problem-solving through self-correction and reevaluation.
- Despite its impressive zero-shot reasoning capabilities, DeepSeek-R1-Zero exhibits several limitations:
    - **Readability Issues**: Outputs often include mixed languages or lack coherent formatting.
    - **Chaotic Early Training**: Without a structured reasoning prior, early-stage RL leads to unstable and inconsistent behaviors.
- These challenges ultimately motivated the development of DeepSeek-R1, which adds a cold-start SFT phase and a multi-stage RL pipeline to refine and stabilize reasoning capabilities while maintaining performance.

## DeepSeek-R1 →→ Training Pipeline: Cold-Start SFT to Multi-Stage RL

- DeepSeek-R1 employs a multi-stage training pipeline designed to enhance reasoning capabilities while maintaining efficiency. This process includes distinct phases, each guided by task-specific loss functions and reward mechanisms, ensuring progressive refinement in performance. The key stages are SFT, RL, Rejection Sampling, and an additional RL phase for generalization. Together, these steps improve DeepSeek-R1’s ability to tackle complex reasoning tasks while ensuring clarity and coherence in its outputs.
- DeepSeek-R1’s training process unfolds in four key phases, each progressively refining its reasoning ability while expanding generalization and alignment:
    1. **Cold Start with SFT**
        - Fine-tuning on thousands of high-quality CoT examples to establish structured reasoning.
        - Uses a structured output format for improved readability.
        - Employs a cross-entropy-based loss function for optimization.
    2. **RL with GRPO**
        - Policy optimization via Group-based Reward Normalization (GRPO).
        - Rewards assigned based on accuracy, format consistency, and language alignment.
        - Prevents reward hacking by avoiding neural reward models.
    3. **Rejection Sampling & Expanded SFT**
        - Filters high-quality RL outputs to enhance supervised fine-tuning.
        - Expands training data to include non-reasoning tasks, ensuring broader applicability.
    4. **Final RL Phase for Generalization**
        - Integrates diverse task distributions, extending beyond structured reasoning.
        - Ensures alignment with human feedback, particularly in conversational settings.
- Through this multi-stage refinement process, DeepSeek-R1 surpasses previous models in accuracy, coherence, and real-world usability, setting a new benchmark for AI reasoning capabilities.

### Stage 1: Cold Start with SFT

#### Fine-Tuning with High-Quality Chain-of-Thought (CoT) Examples

- DeepSeek-R1 begins its journey by fine-tuning the DeepSeek-V3-Base model with a carefully curated dataset of high-quality CoT examples. These examples are obtained through a combination of:
    1. **Few-shot prompting:** Generating detailed reasoning paths using large-scale pre-trained models.
    2. **Manual annotation and refinement:** Filtering and refining reasoning steps through human reviewers.
    3. **Post-processing DeepSeek-R1-Zero outputs:** Extracting well-structured reasoning paths from the RL-trained precursor model.
- The fine-tuning step ensures that DeepSeek-R1 has a structured reasoning framework before entering RL. Unlike DeepSeek-R1-Zero, which learned reasoning solely from RL, DeepSeek-R1 leverages cold-start fine-tuning to avoid the chaotic early stages of RL training.

#### Structured Output Format

- One of the key issues encountered in DeepSeek-R1-Zero was language mixing and poor readability. To address this, the fine-tuning phase enforces a structured reasoning format:

![](https://aman.ai/images/copy.png)

`<reasoning_process> Step-by-step explanation of the problem-solving approach </reasoning_process> <summary> Final Answer </summary>`

- This format ensures readability and helps align the model’s outputs with human expectations.

#### Loss Function for SFT

- The model is optimized using a categorical cross-entropy loss:
    
    LSFT=−∑i=1nlogPθ(oi|q,{o1,…,oi−1})LSFT=−∑i=1nlog⁡Pθ(oi|q,{o1,…,oi−1})
    
    - where:
        - oioi is the ithith token in the output sequence,
        - qq is the input query,
        - o1,...,oi−1o1,...,oi−1 are previously generated tokens.
- This step helps DeepSeek-R1 establish a strong foundation for structured reasoning before RL.
    

### Stage 2: RL

- RL is the backbone of DeepSeek-R1’s reasoning evolution. The model learns to optimize its reasoning trajectories based on reward-driven feedback mechanisms, leading to significant improvements in accuracy and coherence.

#### DeepSeek’s RL Methodology: a Conceptual Overview

- DeepSeek’s RL methodology is fundamentally inspired by self-play paradigms, akin to training AI models in games like chess. Traditionally, AI models trained for complex reasoning tasks leverage large datasets composed of human-annotated examples. However, such datasets often lack comprehensive coverage and may not contain optimal solutions. RL circumvents this limitation by allowing AI models to explore solutions autonomously, refining their strategies based on reward-driven feedback mechanisms.
- Consider an AI model trained to play chess. Instead of learning from a fixed dataset of historical games, the AI is programmed with only the fundamental rules of chess. It then engages in self-play, continuously experimenting with various moves. Initially, the model executes suboptimal actions, leading to losses. However, through iterative play, it identifies effective strategies and reinforces moves that contribute to victories while discarding ineffective ones. This trial-and-error process, governed by RL principles, enables the AI to develop strategies surpassing human intuition.
- DeepSeek applies this RL-based approach to reasoning-intensive domains, such as mathematical problem-solving. Rather than training on explicit mathematical derivations, the AI is provided with fundamental mathematical rules and tasked with solving problems autonomously. The model systematically explores various solution paths, reinforcing those that yield correct answers while discarding ineffective paths. Over time, this process enhances the AI’s mathematical reasoning abilities beyond traditional supervised learning approaches. The self-improving nature of RL fosters the discovery of novel problem-solving strategies, resulting in superior performance in mathematical reasoning and logic-based tasks.

#### Background: Policy Optimization

- Policy optimization involves an RL framework refining an agent’s decision-making process to maximize expected rewards.
- Traditional methods like REINFORCE provide a fundamental approach to learning policies directly from sampled trajectories, while more advanced techniques like Proximal Policy Optimization (PPO) introduce stability constraints.
- Group Relative Policy Optimization (GRPO) builds upon these foundations, addressing key limitations to enhance efficiency and stability in large-scale applications. GRPO can be seen as a hybrid between REINFORCE and PPO, integrating the variance reduction of PPO with the simplicity of direct policy gradient updates from REINFORCE, making it a promising alternative for reinforcement learning in large-scale language model training.

##### The REINFORCE Algorithm

- Before discussing GRPO, it is essential to understand REINFORCE, one of the earliest and simplest reinforcement learning algorithms.

###### What is REINFORCE?

- REINFORCE is a policy gradient method that updates a policy network based on complete trajectories sampled from the environment. It follows a straightforward approach:
    
    1. **Sampling Trajectories:** The agent interacts with the environment, generating an episode (a sequence of states, actions, and rewards).
    2. **Reward Calculation:** A single reward is assigned to the entire episode.
    3. **Policy Update:**
        - Compute the gradient of the policy based on the log probability of actions taken.
        - Scale the gradient by the total episode reward.
        - Update the policy network using gradient descent.

###### Limitations of REINFORCE

- **High Variance:** Since rewards are computed for entire episodes, updates can be noisy.
- **Unstable Learning:** Policy updates can be drastic, leading to instability.
- **Lack of Baseline Correction:** REINFORCE does not normalize rewards, making training inefficient.

##### Proximal Policy Optimization (PPO)

- Proximal Policy Optimization (PPO) is a widely used RL algorithm in RLHF, particularly in LLMs. PPO is an actor-critic method designed to optimize a policy while ensuring stable updates by limiting drastic deviations from previous policies.
- For a detailed discourse, please refer our [PPO primer](https://aman.ai/primers/ai/llm-alignment/#proximal-policy-optimization-ppo).

###### How PPO Works

- PPO requires three primary components:
    - **Policy (πθπθ):** The LLM being fine-tuned.
    - **Reward/Grader (RϕRϕ):** A frozen model/function providing scalar feedback on complete responses.
    - **Critic/Value (VγVγ):** A trainable value model/function predicting future rewards for partial responses.
- PPO follows an iterative workflow:
    1. **Response Generation:** The model generates multiple responses per prompt.
    2. **Reward Assignment:** The reward model scores each response.
    3. **Advantage Computation:** The advantage function estimates how much better an action is compared to average actions.
    4. **Policy Optimization:** The LLM is updated to maximize the advantage function using PPO’s clipped objective.
    5. **Critic Update:** The value function is trained to improve reward prediction.

###### Challenges with PPO

- **High Computational Cost:** PPO requires a separate critic model, which doubles memory requirements.
- **Training Complexity:** The critic must be updated in tandem with the policy, making training unstable.
- **Potential Bias:** The critic can introduce estimation biases, affecting policy optimization.
- These limitations motivated the introduction of Group Relative Policy Optimization (GRPO) by DeepSeek AI as part of [DeepSeekMath](https://arxiv.org/abs/2402.03300).

#### Group Relative Policy Optimization (GRPO)

- GRPO, introduced in [DeepSeekMath](https://arxiv.org/abs/2402.03300), is a RL method that has played a pivotal role in the development of DeepSeek-R1. It is a simplified and cost-efficient alternative to traditional policy optimization techniques like Proximal Policy Optimization (PPO), since it does not require a separate critic model. Instead, it estimates the baseline from a group of generated outputs, reducing computational overhead while maintaining sample efficiency. This group-based approach ensures that each update step improves on previous iterations without overfitting to individual trajectories.
- GRPO has evolved from a mathematical reasoning optimizer in DeepSeekMath to a core optimization technique in DeepSeek-R1, driving advanced reasoning capabilities across diverse tasks. By eliminating the critic model (also called the value model), leveraging group-based advantages, and incorporating multi-stage RL refinements, GRPO has made DeepSeek-R1 a powerful open-source reasoning model.
- GRPO is central to DeepSeek-R1’s RL pipeline, providing a lightweight yet powerful optimization mechanism. Its key innovations include:
    - Removing the critic model, which significantly reduces memory overhead.
    - Stabilizing policy updates through group-based advantage estimation.
    - Efficient training while maintaining strong performance compared to PPO-based methods.
- From its inception in DeepSeekMath to its refined implementation in DeepSeek-R1, GRPO has undergone several enhancements, including multi-stage RL, improved reward modeling, and refined optimization strategies. This section details GRPO’s mathematical formulation, its implementation, and its role in DeepSeek-R1.
- The following figure from the paper demonstrates PPO and GRPO. GRPO foregoes the value/critic model, instead estimating the baseline from group scores, significantly reducing training resources.

![](https://aman.ai/primers/ai/assets/DeepSeek/PPO-GRPO.jpg)

- For a discourse on Reinforcement Fine-Tuning (RFT), please refer to our [RFT](https://aman.ai/primers/ai/reinforcement-finetuning) primer.

##### Key Innovations

- **No Critic Model:** Instead of learning a separate value function, GRPO derives advantages directly from response samples.
- **Group-Based Advantage Estimation:** GRPO normalizes rewards within a batch of generated responses.
- **Improved Efficiency:** Eliminates critic updates, reducing training overhead and memory consumption by ~50%.
- **Stable Training:** By computing relative rewards within a group, GRPO ensures that policy updates remain well-regulated.

###### How GRPO Builds on REINFORCE

- GRPO modifies REINFORCE by:
    - **Using Group-Based Advantage Estimation:** Instead of relying on a single episode reward, GRPO normalizes rewards within a group.
    - **Introducing a Clipped Loss Function:** Prevents large policy updates.
    - **Reducing Variance:** By averaging multiple sampled responses, GRPO provides a more stable policy update mechanism.
- By addressing these weaknesses, GRPO combines the simplicity of REINFORCE with the stability of modern policy optimization techniques.

###### How GRPO Builds on PPO

- Unlike PPO, which relies on a critic to estimate future rewards, GRPO directly normalizes rewards within a group of responses to compute an advantage function. By avoiding the need for a separate critic model, GRPO reduces memory and compute costs while maintaining sample efficiency, making it scalable for large-scale training. Furthermore, this eliminates potential biases introduced by the critic. Put simply, GRPO addresses PPO’s limitations of high computational costs, training instability due to the training of the policy and critic model in tandem, and potential biases in the critic model, by replacing the critic with a group-based reward normalization mechanism.
- PPO’s clipped objective function is retained in GRPO, ensuring stable policy updates and preventing overly large parameter shifts.
- The combination of group-based reward normalization and clipped policy updates allows GRPO to achieve comparable stability to PPO while being computationally more efficient.
- A comparative analysis of REINFORCE, PPO, and GRPO in terms of critic model usage, compute cost, stability, advantage estimation, and training complexity, highlighting GRPO’s high stability and PPO’s high compute cost.

|**Feature**|**REINFORCE**|**PPO**|**GRPO**|
|---|---|---|---|
|**Critic Model?**|❌ No|✅ Yes|❌ No|
|**Compute Cost**|**Low**|**High**|**Low**|
|**Stability**|Low (high variance)|Moderate (tandem training of actor/policy and critic/value)|High (group normalization)|
|**Advantage Estimation**|Episode reward|Learned critic|Group-based normalization|
|**Training Complexity**|**Low**|**High**|**Moderate**|

##### Evolution of GRPO: from DeepSeekMath to DeepSeek-R1

###### Phase 1: GRPO in DeepSeekMath (Mathematical RL)

- GRPO was originally introduced in DeepSeekMath to optimize models for mathematical reasoning.
- It replaced PPO’s critic model with a group-based reward normalization technique, making training more efficient while maintaining stability.
- The reward function primarily evaluated mathematical correctness, using structured evaluation metrics.

###### Phase 2: GRPO in DeepSeek-R1-Zero (Self-Evolving Reasoning)

- With DeepSeek-R1-Zero, GRPO was applied without any SFT—pure RL was used to shape reasoning behaviors from scratch.
- The model self-learned reasoning skills such as step-by-step problem-solving and self-verification.
- However, DeepSeek-R1-Zero exhibited readability issues (e.g., unstructured reasoning outputs, language mixing).

###### Phase 3: GRPO in DeepSeek-R1 (Refined Reasoning & Cold Start)

- DeepSeek-R1 introduced a multi-stage RL pipeline incorporating a small amount of cold-start fine-tuning before applying GRPO.
- The reward model was expanded beyond mathematics to include general reasoning tasks.
- A language consistency reward was added to improve coherence and readability.

##### How GRPO Works

- GRPO replaces PPO’s critic-based advantage estimation with a group-based normalization approach. Instead of learning a value function, GRPO derives relative rewards from multiple sampled responses. This enables efficient and stable policy updates while reducing computational overhead.

###### Mathematical Formulation

- The GRPO objective function is:
    
    JGRPO(θ)=𝔼q∼P(Q),{oi}Gi=1∼πθold(O|q)[1G∑i=1Gmin(ρiAi,clip(ρi,1−ϵ,1+ϵ)Ai)−βDKL(πθ‖πref)]JGRPO(θ)=Eq∼P(Q),{oi}i=1G∼πθold(O|q)[1G∑i=1Gmin(ρiAi,clip(ρi,1−ϵ,1+ϵ)Ai)−βDKL(πθ‖πref)]
    
    - where:
        - ρiρi is the policy likelihood ratio, indicating how much the new policy diverges from the old one: ρi=πθ(oi|q)πθold(oi|q)ρi=πθ(oi|q)πθold(oi|q)
        - AiAi is the group-based advantage function, computed from group-based reward normalization which normalizes rewards across sampled outputs: Ai=ri−mean(r1,...,rG)std(r1,...,rG)Ai=ri−mean(r1,...,rG)std(r1,...,rG)
        - DKL(πθ‖πref)DKL(πθ‖πref) is a KL regularization term that constrains updates within a stable range.
        - GG is the group size (number of sampled outputs per query).
        - ϵϵ controls clipping to prevent overly aggressive updates.
        - ββ controls the strength of KL regularization.
        - QQ is the set of all possible input queries (e.g., math problems or prompts).
        - q∈Qq∈Q is a specific query sampled from the query distribution P(Q)P(Q).
        - OO is the space of possible outputs (e.g., generated token sequences or solutions).
        - oi∈Ooi∈O is the ithith output sampled from the old policy πθoldπθold conditioned on query qq, i.e., oi∼πθold(O∣q)oi∼πθold(O∣q).
        - πθπθ is the current (trainable) policy model.
        - πθoldπθold is the old policy used to sample outputs, which is dynamic and updated throughout training during each iteration of the optimization loop.
        - πrefπref is the reference policy used for KL regularization, often set to the supervised fine-tuned (SFT) model.
        - riri is the scalar reward assigned to output oioi by a reward model.
        - ϵϵ is the trust region clipping parameter to stabilize training,
- Plugging in the the policy likelihood ratio ρiρi, the expanded form of the GRPO objective function can be written as:
    
    JGRPO(θ)=𝔼q∼P(Q),{oi}Gi=1∼πθold(O|q)[1G∑i=1Gmin(πθ(oi|q)πθold(oi|q)Ai,clip(πθ(oi|q)πθold(oi|q),1−ϵ,1+ϵ)Ai)−βDKL(πθ||πref)]JGRPO(θ)=Eq∼P(Q),{oi}i=1G∼πθold(O|q)[1G∑i=1Gmin(πθ(oi|q)πθold(oi|q)Ai,clip(πθ(oi|q)πθold(oi|q),1−ϵ,1+ϵ)Ai)−βDKL(πθ||πref)]
    

###### Mathematical Intuition

- To understand GRPO, it is useful to analyze its mathematical formulation from a reverse-engineering perspective. The complexity of the equations can be misleading; in reality, GRPO consists of three main components:
    
    JGRPO=min([Block 1],[Block 2])−[Block 3]JGRPO=min([Block 1],[Block 2])−[Block 3]
    
    - where:
        - Block 1 corresponds to the first term inside the summation of the GRPO objective function: ρiAi=πθ(oi|q)πθold(oi|q)Ai.ρiAi=πθ(oi|q)πθold(oi|q)Ai. This represents the primary objective of policy optimization: ensuring the updated policy πθπθ improves upon the previous policy πθoldπθold. The core principle is straightforward: the new policy should outperform the old one in expectation.
        - Block 2 corresponds to the clipped version of ρiAiρiAi, i.e., clip(ρi,1−ϵ,1+ϵ)Ai.clip(ρi,1−ϵ,1+ϵ)Ai. This originates from PPO and serves as a safeguard to prevent excessive updates. By taking the minimum between Block 1 and this clipped value, GRPO ensures training stability and prevents over-exaggerated policy updates.
        - Block 3 corresponds to the KL-divergence regularization term in the GRPO equation: βDKL(πθ||πref).βDKL(πθ||πref). This term enforces similarity between the new policy and a reference policy, preventing the optimization process from deviating too far from the original distribution and ensuring controlled updates.
- One of the most notable aspects of GRPO’s success is its redesigned approach to advantage computation. Traditional PPO computes advantages using a learned value network combined with temporal difference learning, requiring additional memory and computation to maintain a separate critic model. In contrast, GRPO fundamentally simplifies this by directly comparing sampled actions within a group and leveraging statistical normalization to compute advantages. This group-based methodology eliminates the need for a value network, significantly reducing memory overhead—by approximately half—while simultaneously aligning with the core principle of evaluating mathematical solutions relative to other approaches to the same problem.
- This design choice has proven especially effective for mathematical reasoning tasks. By using a direct group-based comparison, GRPO enhances the model’s ability to develop structured reasoning strategies. Empirical results demonstrate that this method not only improves performance on mathematical reasoning benchmarks but also maintains training stability and computational efficiency. The elimination of the critic network removes potential biases from learned value functions, making GRPO particularly well-suited for domains requiring objective evaluation of multiple solution paths.
- Additionally, the “Group” aspect in GRPO refers to computing the expectation over a set of sampled outputs, which are then averaged to stabilize training.
- Thus, when stripped of indices, subscripts, and hyperparameters, GRPO reduces to a simple balance between policy improvement and control mechanisms, reinforcing why it is regarded as an efficient and intuitive optimization method.

##### Step-by-Step Breakdown

###### Policy Likelihood Ratio ρiρi

- Measures how much the probability of generating output oioi has changed under the new policy compared to the old policy: ρi=πθ(oi|q)πθold(oi|q)ρi=πθ(oi|q)πθold(oi|q)

###### Advantage Function AiAi

- Instead of relying on a separate value network (critic), GRPO estimates the advantage function using a group of sampled outputs: Ai=ri−mean(r1,...,rG)std(r1,...,rG)Ai=ri−mean(r1,...,rG)std(r1,...,rG)
- This reduces training instability and enhances efficiency.

###### Clipping Mechanism clip(⋅)clip(⋅)

- Prevents drastic policy updates that could destabilize training: clip(ρi,1−ϵ,1+ϵ)clip(ρi,1−ϵ,1+ϵ)

###### KL Divergence Penalty DKLDKL

- Ensures the policy remains close to a reference distribution: βDKL(πθ‖πref)βDKL(πθ‖πref)
- Prevents mode collapse and excessive policy drift.

> Both PPO and GRPO incorporate a KL divergence term to regulate policy updates, but they differ in which distributions are compared. In PPO, the KL term is typically computed as DKL(πθold‖‖πθ)DKL(πθold‖‖πθ), measuring how much the new policy deviates from the old one, i.e., the immediately prior policy. This enforces conservative updates by penalizing large shifts from the old policy. In contrast, GRPO uses DKL(πθ‖‖πref)DKL(πθ‖‖πref), where the reference policy πrefπref is the frozen initial policy, which is obtained as the output of the SFT phase. This choice emphasizes how far the current policy strays from a desired or stabilized policy reference, allowing for different control dynamics in policy learning.

###### Old Policy πoldπold

- This is the immediate past policy used to sample data for updating. Specifically, the old policy is used to sample the outputs (o1,o2,…,oGo1,o2,…,oG) for each prompt qq.
- It is used in the importance sampling ratio term πθ(oi,t∣q,oi,<t)πold(oi,t∣q,oi,<t)πθ(oi,t∣q,oi,<t)πold(oi,t∣q,oi,<t).
- This ratio is part of the main GRPO objective and helps estimate how much the new policy πθπθ differs from the old one when generating the same outputs.
- The old policy offers stability during optimization (as in PPO).

###### Reference Policy πrefπref

- This is typically the initial model from the SFT phase, which serves as a long-term anchor or baseline to avoid reward over-optimization or undesirable divergence.
- It is used to regularize the learning via a KL divergence term DKL[πθ‖πref]DKL[πθ‖πref].
- This helps prevent the new policy from drifting too far from the original (aligned) behavior. Put simply, the reference policy prevents drift from human-aligned behavior (via KL regularization).

##### Algorithm

- The following steps highlight GRPO’s efficiency: it uses only group statistics, requires no separate value network, and is well-suited for both rule-based rewards (e.g., correctness in math problems, coding, formatting consistency, etc.) as well as human preference-alignment based on reward models that assess helpfulness, harmlessness, and human-centric values.
    
    1. **Sample a Group of Responses (GG):**
        - For each input prompt, the policy model (which is being trained) generates a set of responses ={r1,r2,...,rN}G={r1,r2,...,rN} using the _old policy_ πoldπold. This sampling forms the foundation of GRPO’s group-based strategy and allows it to contextualize learning not by absolute performance, but by _relative_ performance among peers within the group.
    2. **Compute Rewards:**
        - Each response riri in the group is scored using a reward model RϕRϕ, which outputs scalar values indicating how good each response is. These scores reflect alignment with desirable behaviors such as correctness, clarity, and reasoning quality.
        - In the context of GRPO and especially in the final reinforcement learning stage of DeepSeek-R1, these rewards are derived from a combination of rule-based metrics (e.g., correctness in math problems, formatting) and human preference-aligned reward models. The latter are trained on preference pairs to assess which outputs better align with helpfulness, harmlessness, and human-centric values.
        - For reasoning tasks like math or code, rule-based accuracy is often sufficient. However, for broader applications and to align with human expectations, DeepSeek-R1 also incorporates reward signals trained on diverse prompt distributions. This includes assessments of readability, language consistency, and summary quality, especially important in multi-language and general-purpose scenarios.
        - Crucially, GRPO assumes the reward model is only reliable when comparing _responses to the same prompt_, making the group-wise setup ideal. By comparing responses within the same group, GRPO leverages relative quality rather than absolute reward magnitude, aligning closely with how human preferences are typically expressed and learned.
    3. **Calculate Advantage (AiAi) Using Group Normalization:**
        - Instead of relying on a learned value function like in PPO (which can be memory-intensive and noisy), GRPO computes the advantage for each response using the group’s statistical properties: Ai=Rϕ(ri)−mean()std()Ai=Rϕ(ri)−mean(G)std(G)
        - This normalized score reflects how much better or worse a response is compared to its peers.
        - **Motivation:** This approach aligns with how reward models are typically trained—on preference pairs rather than absolute values. Group normalization thus emphasizes relative quality, allowing the model to learn _which responses are better_ without needing a global baseline.
        - **Benefits:**
            - Avoids the need for a separate value network (used in PPO)
            - Significantly reduces compute and memory requirements
            - Naturally leverages the comparative nature of reward models
    4. **Update the Policy with GRPO Objective:**
        - The policy is updated by maximizing the GRPO-specific surrogate objective: JGRPO(θ)=𝔼q∼P(Q),{oi}Gi=1∼πθold(O|q)[1G∑Gi=1min(πθ(oi|q)πθold(oi|q)Ai,clip(πθ(oi|q)πθold(oi|q),1−ϵ,1+ϵ)Ai)−βDKL(πθ||πref)]JGRPO(θ)=Eq∼P(Q),{oi}i=1G∼πθold(O|q)[1G∑i=1Gmin(πθ(oi|q)πθold(oi|q)Ai,clip(πθ(oi|q)πθold(oi|q),1−ϵ,1+ϵ)Ai)−βDKL(πθ||πref)]
        - The clipping function stabilizes the update, while the KL divergence regularizes the new policy against a reference model (often the supervised fine-tuned policy), preventing divergence from known good behavior.

###### Reward Function Design

- In DeepSeekMath, the reward was primarily based on mathematical correctness.
- In DeepSeek-R1, the reward function expanded to include:
    - **Accuracy/Correctness Rewards**: Evaluating correctness for general reasoning tasks (e.g., coding, science, logic).
    - **Format Rewards**: Ensuring structured reasoning using `<think>` and `<answer>` tags.

##### Advantage Estimation

- The advantage in GRPO is computed using the predicted rewards (typically from a value/critic function) via a novel approach that eliminates the need for a separate value model, unlike traditional PPO. Here’s a breakdown of how the advantage is computed in GRPO.

###### Background: Generalized Advantage Estimation

- In traditional RL, and specifically in PPO, the advantage is typically computed as: At=rt−V(st)At=rt−V(st)
    - where:
        - AtAt is the advantage at time step tt
        - rtrt is the reward at time step tt
        - V(st)V(st) is the estimated value of state stst
- Or more generally via Generalized Advantage Estimation (GAE), which refines this with discounted returns to reduce variance: At=δt+(γλ)δt+1+(γλ)2δt+2+…At=δt+(γλ)δt+1+(γλ)2δt+2+…
    - where:
        - δt=rt+γV(st+1)−V(st)δt=rt+γV(st+1)−V(st)
            
        - γγ is the discount factor
        - λλ is the GAE smoothing parameter
- Advantage can thus be defined as a measure of how much better an action is compared to the expected value (baseline). Mathematically, Advantage=Reward−Value (Baseline)Advantage=Reward−Value (Baseline)
    - where:
        - “Advantage” quantifies the relative gain of an action
        - “Reward” is the return obtained after taking the action
        - “Value (Baseline)” is the expected return from the state
    - Specifically, PPO uses a learned value model to estimate the baseline:  
        At=rt−Vψ(st)At=rt−Vψ(st)
        - where:
            - VψVψ is the learned value function parameterized by ψψ
    - On the other hand, GRPO uses a group average reward as the baseline: Â i,t=ri−r¯σrorÂ i,t=∑j≥tri,j−r¯σrA^i,t=ri−r¯σrorA^i,t=∑j≥tri,j−r¯σr
        - where:
            - riri is the total reward for output oioi
            - r¯r¯ is the group mean reward
            - σrσr is the standard deviation of rewards
            - ri,jri,j is the step-wise reward for step jj of output ii
- This makes GRPO a value-free method with significantly lower compute/memory cost, while retaining the core idea of advantage-based policy optimization.

###### Background: PPO Advantage Estimation

- In PPO, the advantage is computed using a learned value function VψVψ. The classic way to define advantage is: At=rt−Vψ(st)At=rt−Vψ(st)
    - where:
        - AtAt is the advantage
        - rtrt is the reward at time tt
        - Vψ(st)Vψ(st) is the estimated value of state stst using model ψψ
- However, more accurately and stably, PPO typically uses Generalized Advantage Estimation (GAE), which smooths over multiple future timesteps: AGAEt=∑∞l=0(γλ)lδt+lwhereδt=rt+γVψ(st+1)−Vψ(st)AtGAE=∑l=0∞(γλ)lδt+lwhereδt=rt+γVψ(st+1)−Vψ(st)
    - where:
        - γγ is the discount factor
        - λλ is the GAE parameter
        - δtδt is the temporal-difference error at time tt
        - VψVψ is the learned value model
- So PPO explicitly requires a value model VψVψ to compute this baseline. The goal is to reduce the variance of the gradient estimates while keeping the bias minimal.
- This advantage is then used in PPO’s clipped surrogate objective: JPPO(θ)=𝔼q,o∼πold[1|o|∑|o|t=1min(πθ(ot|q,o<t)πold(ot|q,o<t)At,clip(⋅)At)]JPPO(θ)=Eq,o∼πold[1|o|∑t=1|o|min(πθ(ot|q,o<t)πold(ot|q,o<t)At,clip(⋅)At)]
    - where:
        - πθπθ is the current policy
        - πoldπold is the old policy
        - AtAt is the advantage at time tt
        - |   |   |   |
            |---|---|---|
            |$$|o|$$ is the length of output sequence|
            
        - “clip” ensures the ratio stays within a safe range

###### GRPO Advantage Estimation

- In GRPO, there’s no value function — instead, the baseline (or expected value) is approximated using the group mean reward. So the advantage is still reward minus baseline, but the definition of the baseline depends on whether outcome or process supervision is adopted.
- **Outcome Supervision (one reward per output):**
    - Let riri be the reward for output oioi, and the baseline be the group average r¯r¯, then: Â i,t=r̃ i=ri−r¯σrwherer¯=1G∑Gj=1rjA^i,t=r~i=ri−r¯σrwherer¯=1G∑j=1Grj
        - where:
            - riri is the reward for sample ii
            - r¯r¯ is the average reward across the group
            - σrσr is the standard deviation for normalization
            - GG is the group size
    - This is essentially: Advantage=Reward−BaselineStandard Deviation (for normalization)Advantage=Reward−BaselineStandard Deviation (for normalization)
        - where:
            - “Reward” is the individual sample’s score
            - “Baseline” is the group mean
            - The expression is normalized by σrσr
    - Every token tt in output oioi receives the same normalized advantage.
- **Process/Step-wise Supervision (rewards for steps):**
    - If ri,jri,j is the reward for the jj-th step of output oioi, and r¯r¯ is the group mean: r̃ i,j=ri,j−r¯σrr~i,j=ri,j−r¯σr
        - where:
            - ri,jri,j is the reward for step jj in sample ii
            - r¯r¯ is the mean reward across all steps in the group
            - σrσr is the standard deviation for normalization
    - Then for each token tt, the advantage is the sum of normalized rewards for all steps ending after tt: Â i,t=∑step j:index(j)≥tr̃ i,jA^i,t=∑step j:index(j)≥tr~i,j
        - where:
            - The sum includes all steps jj such that the index of jj is greater than or equal to token index tt
            - r̃ i,jr~i,j is the normalized reward for step jj of output ii
    - Again, this reflects reward minus baseline in a normalized form — just applied step-wise.

##### Comparative Analysis: REINFORCE vs. TRPO vs. PPO vs. DPO vs. KTO vs. APO vs. GRPO

- **REINFORCE**:
    - **Function**: The simplest policy gradient algorithm that updates the model based on the cumulative reward received from complete trajectories.
    - **Implementation**: Generates an entire episode, calculates rewards at the end, and updates the policy network based on a weighted log probability loss.
    - **Practical Challenges**: High variance in policy updates, slow convergence, and instability due to unbounded updates.
- **TRPO**:
    - **Function**: Trust Region Policy Optimization (TRPO) improves policy updates by constraining step sizes to avoid instability.
    - **Implementation**: Uses a constrained optimization formulation to ensure each update remains within a trust region, preventing excessive deviations.
    - **Practical Challenges**: Computationally expensive due to the constraint-solving step and requires second-order optimization techniques.
- **PPO**:
    - **Function**: An RL algorithm that optimizes the language model by limiting how far it can drift from a previous version of the model.
    - **Implementation**: Involves sampling generations from the current model, judging them with a reward model, and using this feedback for updates.
    - **Practical Challenges**: Can be slow and unstable, especially in distributed settings.
- **DPO**:
    - **Function**: Minimizes the negative log-likelihood of observed human preferences to align the language model with human feedback.
    - **Data Requirement**: Requires paired preference data.
    - **Comparison with KTO**: While DPO has been effective, KTO offers competitive or superior performance without the need for paired preferences.
- **KTO**:
    - **Function**: Adapts the Kahneman-Tversky human value function to the language model setting. It uses this adapted function to directly maximize the utility of model outputs.
    - **Data Requirement**: Does not need paired preference data, only knowledge of whether an output is desirable or undesirable for a given input.
    - **Practicality**: Easier to deploy in real-world scenarios where desirable/undesirable outcome data is more abundant.
    - **Model Comparison**: Matches or exceeds the performance of direct preference optimization methods across various model sizes (from 1B to 30B).
- **APO**:
    - **Function**: Introduces a family of contrastive objectives explicitly accounting for the relationship between the model and the preference dataset. This includes APO-zero, which increases desirable outputs while decreasing undesirable ones, and APO-down, which fine-tunes models based on specific quality thresholds.
    - **Data Requirement**: Works effectively with paired preference datasets created through controlled methods like CLAIR and supports stable alignment even for challenging datasets.
    - **Practicality**: Excels at aligning strong models with minimally contrasting preferences, enhancing performance on challenging metrics like MixEval-Hard while providing stable, interpretable training dynamics.
    - **Model Comparison**: Outperformed conventional alignment objectives across multiple benchmarks, closing a 45% performance gap with GPT4-turbo when trained with CLAIR preferences.
- **GRPO**:
    - **Function**: A variant of PPO that removes the need for a critic model by estimating the baseline using group scores, improving memory and computational efficiency while enhancing the mathematical reasoning of models.
    - **Data Requirement**: Utilizes group-based rewards computed from multiple outputs for each query, normalizing these scores to guide optimization.
    - **Practicality**: Focuses on reducing training resource consumption compared to PPO and improving RL stability.
    - **Model Comparison**: Demonstrated superior performance on tasks like GSM8K and MATH benchmarks, outperforming other models of similar scale while improving both in-domain and out-of-domain reasoning tasks.

###### Tabular Comparison

|**Aspect**|**REINFORCE**|**TRPO**|**PPO**|**DPO**|**KTO**|**APO**|**GRPO**|
|---|---|---|---|---|---|---|---|
|Objective|Policy gradient optimization without constraints.|Ensures stable policy updates within a constrained region.|Maximizes expected reward while preventing large policy updates.|Optimizes policy based on binary classification of human preferences.|Aligns models based on Kahneman-Tversky optimization for utility maximization.|Anchored alignment with specific control over preference-based likelihood adjustments.|Leverages group-based relative advantages and removes the critic network.|
|Learning Mechanism|Monte Carlo policy gradients with high variance.|Second-order optimization with trust region constraints.|Policy gradients with a clipped surrogate objective.|Cross-entropy optimization over paired preferences.|Maximizes desirable likelihoods relative to undesirables, without paired data.|Uses variants like APO-zero or APO-down for stable preference-based optimization.|Group normalization with policy gradients, eliminating the critic network.|
|Stability|Low (high variance, unstable updates).|High (enforces trust region for stable updates).|Relies on clipping mechanisms to avoid destabilization.|Stable as it directly optimizes preferences.|Stable due to focus on unpaired desirability adjustments.|Offers robust training stability, scaling better on models trained with mixed-quality datasets.|Stable due to normalization of rewards across groups.|
|Training Complexity|High (unconstrained updates).|Very high (requires second-order optimization and solving constraints).|High, due to balancing reward maximization with policy constraints.|Moderate; uses simplified binary preference objectives.|Simplifies alignment by focusing only on desirability.|Adaptive and context-aware; requires understanding dataset-model relationships.|Reduces overhead via group-based scoring.|
|Performance|Unstable and sample-inefficient.|More stable than PPO but computationally expensive.|Strong performance on tasks with clear reward signals but prone to instability in distributed setups.|Effective for straightforward preference alignment tasks.|Competitive or better alignment than preference-based methods without paired data needs.|Superior alignment results, particularly for nuanced dataset control.|Excels in reasoning tasks, offering computational efficiency.|
|Notable Strength|Simple to implement but inefficient.|Ensures stable policy updates through trust-region constraints.|Widely used in RL settings, good at reward-based optimization.|Directly optimizes for preferences without needing a separate reward model.|Handles binary data efficiently, avoiding paired data dependencies.|Allows precise alignment with nuanced datasets.|Simplifies reward aggregation; strong for reasoning-heavy tasks.|
|Scenarios Best Suited|RL tasks where simplicity is preferred over efficiency.|High-stability RL tasks requiring constraint-driven policy improvements.|RL environments where reward signals are predefined.|Scenarios with abundant paired human feedback.|Real-world settings with broad definitions of desirable/undesirable outputs.|Tasks requiring precise alignment with minimally contrasting preferences.|Mathematical reasoning or low-resource training setups.|

#### Reward Functions

- Reward modeling is a crucial component of the reinforcement learning process in DeepSeek-R1, determining the optimization direction and shaping the model’s reasoning behavior. DeepSeek-R1 employs a rule-based reward system instead of a neural reward model to avoid reward hacking and excessive computational costs. The primary reward functions guiding DeepSeek-R1 are:

##### Accuracy Rewards

- The accuracy reward model ensures that the model generates factually correct and verifiable responses. It is particularly useful for tasks with deterministic outcomes, such as mathematics and coding.
    
- **Mathematical Tasks:**
    - The model is required to output the final answer in a specified format (e.g., within a box or marked in LaTeX), enabling automated rule-based verification.
    - For example, in mathematical problems, the correctness of the response is checked against a ground-truth solution.
- **Programming Tasks:**
    - For coding problems, correctness is determined using unit tests. The model’s output is compiled and executed against predefined test cases, and rewards are assigned based on the number of passing tests.
    - If the generated code is syntactically incorrect, a small penalty is applied to discourage such outputs.
- **Group-Based Normalization:**
    - Instead of relying on a separate critic network, DeepSeek-R1 uses a group-based reward normalization method. Given a group of responses {r1,r2,...,rG}{r1,r2,...,rG}, the advantage function is calculated as: Ai=ri−mean(r1,...,rG)std(r1,...,rG)Ai=ri−mean(r1,...,rG)std(r1,...,rG)
        - where AiAi represents the normalized advantage of response ii, and standardization ensures stable training updates.

##### Format Rewards

- Beyond correctness, DeepSeek-R1 is trained to produce well-structured and human-readable outputs. The format reward model enforces this by incentivizing adherence to a structured reasoning format.
    
- **Reasoning and Answer Separation:**
    - The model’s responses must follow a two-stage format:
        
        ![](https://aman.ai/images/copy.png)
        
        `<think> Step-by-step breakdown of the reasoning </think> <answer> Final Answer </answer>`
        
    - This ensures that the model explicitly separates its reasoning process from its final answer, improving clarity and user comprehension.
- **Language Consistency Reward:**
    - One challenge observed in earlier versions, such as DeepSeek-R1-Zero, was language mixing, where responses included a blend of multiple languages (e.g., partial English and partial Chinese).
    - To mitigate this, DeepSeek-R1 incorporates a language consistency reward, defined as the proportion of words in the target language: Rlang=Count of words in target languageTotal word countRlang=Count of words in target languageTotal word count
    - This encourages the model to maintain linguistic coherence without degrading its reasoning performance.

##### Combined Reward Function

- The final reward signal for DeepSeek-R1 is computed as a weighted sum of the individual reward components:
    
    Rfinal=αRaccuracy+βRformat+γRlangRfinal=αRaccuracy+βRformat+γRlang
    
    - where:
        - αα, ββ, and γγ are hyperparameters controlling the relative contributions of each reward type:
            - Accuracy rewards ensure correctness,
            - Format rewards ensure structured output,
            - Language consistency rewards ensure readability and coherence.
- This design choice balances factual correctness with user-friendly response formatting, making DeepSeek-R1 a powerful reasoning model.
    

##### Why Rule-Based Rewards Instead of Neural Reward Models?

- DeepSeek-R1 avoids the use of neural reward models because they are susceptible to reward hacking and require costly retraining. Instead, a deterministic rule-based approach provides:
    - **Greater transparency:** Rewards are interpretable and verifiable.
    - **Reduced computational cost:** No need for an additional neural network.
    - **More stable training dynamics:** Since rule-based rewards are fixed, they do not drift over time.

##### Implementation in GRPO

- DeepSeek-R1’s Group Relative Policy Optimization (GRPO) framework leverages these reward functions during training:
    - A batch of multiple outputs per query is sampled.
    - The relative rewards within the group are computed.
    - The advantage estimates are normalized.
    - The policy is updated using a clipped objective function that prevents large policy shifts.
- This process ensures efficient reinforcement learning without the need for a separate critic model, leading to more stable and scalable training.

### Stage 3: Rejection Sampling & Expanded Supervised Fine-Tuning

- After RL convergence, DeepSeek-R1 undergoes an additional fine-tuning step based on rejection sampling. This stage refines the reasoning process by incorporating:
    - **Reasoning Trajectories**: Selecting correct and well-structured CoT explanations from RL outputs.
    - **Expanded Task Coverage**: Augmenting the dataset with non-reasoning tasks like:
        - Writing & Summarization
        - Fact-based Question Answering
        - Self-cognition and safety-related responses
- **Implementation Details for SFT**:
    - **Fine-Tuning Technique**: This stage uses the full-finetuning variant of SFT with a categorical cross-entropy loss (rather than a parameter-efficient finetuning technique such as LoRA), consistent with earlier SFT stages. The model is trained via standard teacher forcing: LSFT=−∑ni=1logPθ(oi∣q,o<i)LSFT=−∑i=1nlog⁡Pθ(oi∣q,o<i)
        - where:
            - oioi is the ii-th token in the output,
            - o<io<i represents all previously generated tokens,
            - qq is the input query,
            - PθPθ is the model’s predicted probability distribution.
    - **Reasoning Data Collection**: About 600,000 reasoning samples are curated through rejection sampling on the converged RL checkpoint. Multiple outputs are sampled per prompt, and only correct, well-formatted responses are retained. Mixed-language outputs, incoherent reasoning chains, and malformed code blocks are filtered out to maintain readability and consistency.
    - **Use of Generative Rewards**: While earlier RL phases rely exclusively on rule-based rewards, this phase introduces _generative reward modeling_ by passing model responses and references through DeepSeek-V3 to assess correctness in cases where rule-based scoring is not feasible.
    - **Non-Reasoning Data Sourcing**: Around 200,000 samples covering non-reasoning tasks are added. Some are drawn from DeepSeek-V3’s original supervised dataset. In specific instances, DeepSeek-V3 is prompted to generate light reasoning (e.g., reflective CoT) before answering, while simpler queries skip CoT entirely.
    - **Training Process**: The full dataset (~800K samples) is used to fine-tune DeepSeek-V3-Base for two epochs. The resulting model checkpoint forms the basis for the final RL phase.
    - **Output Format Enforcement**: Structured templates like `<reasoning_process> ... </reasoning_process>` and `<summary> ... </summary>` are maintained during fine-tuning to preserve clarity and alignment with prior stages.
    - **Language Quality Control**: Responses exhibiting language mixing or low linguistic coherence are systematically excluded to improve generalization and user experience across multilingual inputs.
    - **Training Configuration**: The fine-tuning is applied to the model checkpoint obtained after Stage 2 (GRPO-based RL). This checkpoint is fine-tuned using the combined dataset (~800k samples) over two epochs.
- This fine-tuning phase not only consolidates the structured reasoning behavior induced by RL but also extends the model’s general capabilities across broader tasks. It acts as a crucial bridge before the final RL generalization stage, aligning the model toward human-preferred formats and diverse task domains.

### Stage 4: Secondary RL for Alignment & Generalization

- The final stage involves another round of RL, but this time with a broader task distribution. Unlike the first RL stage, which focused primarily on reasoning-intensive tasks, this stage incorporates general user interactions such as:
    - Conversational depth (multi-turn dialogues)
    - Complex instructions & role-playing scenarios
    - Ensuring helpfulness & harmlessness in responses
- For general tasks, a reward model is used to align outputs with human preferences. For reasoning tasks, the original rule-based rewards (accuracy & format) are retained.
    
- **Implementation Details**:
    - **Prompt Diversity**: This phase expands the prompt distribution to include a wide variety of task types—from casual conversations to safety-sensitive and instruction-heavy prompts. This broader distribution ensures the model is exposed to realistic, diverse, and nuanced user interactions during training.
    - **Dual Reward Signal**: A combination of rule-based rewards (for math, code, logic) and model-based preference rewards (for general alignment) are used. Preference data is sourced from the DeepSeek-V3 pipeline, covering areas like helpfulness and harmlessness.
    - **Helpfulness Reward**: Calculated specifically on the final summary section of the response to prevent disruption of the reasoning flow. This ensures the model prioritizes clear, relevant, and actionable outputs.
    - **Harmlessness Reward**: Evaluated across the full response (reasoning + summary), identifying and penalizing harmful or biased content to enhance safety and trustworthiness.
    - **RL Framework**: The training continues using the GRPO algorithm. This stage maintains the critic-free setup with group-based advantage estimation but introduces more heterogeneous prompt and reward structures.
    - **Model Architecture & Training**:
        - Continues from the SFT+RL-trained checkpoint (post-rejection sampling).
        - Multiple outputs are sampled per prompt and scored via the appropriate reward mechanism (rule-based or preference-based).
        - The policy is updated using the clipped GRPO loss to maintain training stability and reduce policy drift.
        - KL-regularization is applied against the supervised fine-tuned reference model to prevent degradation of core alignment.
    - **Batch Composition Strategy**:
        - Prompts are batched in a mixed-format setup, meaning each training batch includes both reasoning and non-reasoning (general alignment) tasks.
        - Each sample in the batch is tagged with a task type label, such as `reasoning`, `instruction-following`, `conversational`, or `safety-critical`. During training, these task type tags are used primarily for curriculum control and reward routing as side-channel metadata used by the reward computation pipeline, not necessarily as input-level tokens or control tags embedded in the prompt. This ensures the model is guided during optimization while still learning to generalize from the natural structure and semantics of prompts during inference time.
        - The model internally uses attention masks or task-specific prompt tokens to condition its behavior differently depending on the task type. For example:
            - Reasoning tasks include `<think>` and `<answer>` tags and are evaluated using rule-based rewards.
            - Instruction-following tasks may include tags like `<summary>` or `<response>`, guiding the model to focus on clarity, usefulness, and task compliance.
            - Safety-critical prompts are routed with special tags that signal the harmlessness reward module to evaluate the full output.
        - During training, **gradient updates are not explicitly decoupled per task type**, but the mixed-format batch with tags encourages the model to generalize across task boundaries and learn how to shift generation style and objective based on prompt patterns.
        - This batch composition strategy enables **multi-domain alignment** using a unified GRPO framework, without requiring separate heads or fine-tuning tracks for each domain.
    - **Training Duration**: Training continues until convergence on both reasoning (via rule-based evaluation) and alignment (via offline preference evaluation metrics).
    - **Safety Enhancements**: Additional constraints are applied post-hoc to ensure safe responses in high-risk or adversarial prompts. This includes filtering low-reward outputs and further refining the RL dataset with human-in-the-loop verification for high-stakes domains.
- This final RL phase optimizes DeepSeek-R1 for real-world deployment, ensuring that it remains robust across a variety of domains beyond structured problem-solving. It strengthens the model’s alignment with human values while preserving its advanced reasoning capabilities.

### Comparing Training Pipelines: DeepSeek-R1 vs. DeepSeek-R1-Zero

- DeepSeek-R1 and DeepSeek-R1-Zero represent two distinct training approaches for reasoning-focused LLMs, both leveraging RL but differing significantly in their pre-training methodologies, optimization strategies, and implementation details.
- Through the below-listed refinements, DeepSeek-R1 successfully overcomes the limitations of DeepSeek-R1-Zero, showcasing how structured training pipelines can significantly enhance the reasoning performance of LLMs.

#### Pre-Training and Initialization

- DeepSeek-R1-Zero starts directly from DeepSeek-V3-Base, applying RL without any SFT. This “pure RL” approach forces the model to self-learn reasoning capabilities from scratch through iterative policy optimization.
- DeepSeek-R1, also starts directly from DeepSeek-V3-Base, but undergoes a cold-start fine-tuning phase, where it is trained on thousands of high-quality CoT examples before undergoing RL. This additional step prevents the chaotic early-stage behavior observed in DeepSeek-R1-Zero and ensures a more structured learning trajectory.

#### RL Strategy

- Both models utilize GRPO as the core RL algorithm. However, their reward modeling, training templates, and optimization techniques differ significantly.

##### DeepSeek-R1-Zero: Pure RL Approach

- **Policy Optimization:** Trained solely through GRPO, which estimates a baseline using group scores instead of a separate critic model. This makes RL more memory efficient compared to PPO-based approaches.
- **Training Template:** Outputs are structured using a `<think>` and `<answer>` format to encourage reasoning before answering.
- **Reward Functions:**
    - **Accuracy Reward:** Evaluates correctness for deterministic tasks like math and coding.
    - **Format Reward:** Enforces structured reasoning using the `<think>` and `<answer>` tags.
- **Challenges Encountered:**
    - **Readability Issues:** Many outputs lacked clarity, with mixed-language responses and unstructured formatting.
    - **Convergence Stability:** Early-stage RL training led to unstable outputs, as the model lacked a prior structured reasoning framework.

##### DeepSeek-R1: Multi-Stage RL with Cold-Start Fine-Tuning

- **Cold-Start Fine-Tuning:** Before RL, the model is fine-tuned on thousands of curated CoT examples, improving reasoning structure and readability.
- **Enhanced Reward Functions:**
    - **Language Consistency Reward:** Added to enforce single-language outputs and reduce language mixing issues.
    - **Expanded Reasoning Rewards:** Covers broader reasoning domains beyond math and logic, including coding, science, and knowledge-based tasks.
- **Multi-Stage RL Refinement:**
    - **Stage 1:** RL training with GRPO to refine mathematical reasoning.
    - **Stage 2:** Rejection sampling to extract high-quality CoT explanations for further fine-tuning.
    - **Stage 3:** Final RL Phase for alignment with human feedback, enhancing general conversational capabilities beyond structured problem-solving.

#### Implementation Details and Computational Efficiency

|**Feature**|**DeepSeek-R1-Zero**|**DeepSeek-R1**|
|---|---|---|
|**Pre-training Base**|DeepSeek-V3-Base|DeepSeek-V3-Base|
|**Cold-Start SFT**|❌ No SFT (Pure RL)|✅ Fine-tuned on CoT examples before RL|
|**RL Algorithm**|GRPO|GRPO|
|**Reward Types**|Accuracy, Format|Accuracy, Format, Language Consistency|
|**Training Stability**|❌ Unstable early-stage RL|✅ More stable due to cold-start fine-tuning|
|**Output Readability**|❌ Mixed-language responses, unstructured|✅ Structured reasoning with CoT enforcement|
|**Final Refinement**|Single-stage RL|Multi-stage RL + rejection sampling|

#### Final Performance Impact

- DeepSeek-R1-Zero successfully demonstrated that LLMs can develop reasoning purely via RL, but suffered from poor readability and chaotic convergence.
- DeepSeek-R1 introduced a structured multi-phase training pipeline, resulting in more readable, reliable, and generalized reasoning capabilities, ultimately achieving performance on par with OpenAI o1.

## Emergent Reasoning Behaviors

- DeepSeek-R1 demonstrated remarkable emergent reasoning behaviors during its training process, particularly due to the RL approach that guided its self-evolution. These behaviors include:
    
    - **Reflection**: The model exhibits the ability to revisit and revise its intermediate steps. By analyzing prior outputs and reconsidering logical pathways, it refines its reasoning, ensuring a higher probability of correctness. This reflection is especially visible in long CoT processes where multiple reasoning paths are explored.
        
    - **Self-Correction**: DeepSeek-R1 can detect errors in its own logical steps and apply corrective adjustments. This behavior is incentivized by reward modeling, where the model is trained to recognize inconsistencies and rerun calculations when necessary. This prevents incorrect conclusions from being solidified.
        
    - **Aha Moments**: Perhaps the most striking emergent behavior is the spontaneous “aha moment,” where DeepSeek-R1 halts its current reasoning trajectory, reevaluates the problem from a new angle, and finds a more optimal solution. This is often triggered by a discrepancy between expected and derived results, prompting the model to explore alternative pathways.
        

### Implementation Details

- DeepSeek-R1’s reasoning behaviors emerged through a structured RL framework that included:
    
    1. **Reward-Based Training**: The model was incentivized to provide correct and structured solutions through accuracy and format rewards. This helped shape behaviors like reflection and self-correction.
    2. **Policy Optimization**: Using GRPO, the model iteratively refined its reasoning processes based on feedback from sampled responses.
    3. **Rejection Sampling**: Intermediate outputs were filtered based on correctness, ensuring that only accurate and well-structured reasoning chains were reinforced.
    4. **Cold Start Data**: Unlike its predecessor, DeepSeek-R1-Zero, which purely relied on RL, DeepSeek-R1 was trained on curated long-form reasoning examples as a base, significantly improving its ability to structure logical steps coherently.

### Example: Quadratic Equation Solving

- Consider the problem:
    
    x2−5x+6=0x2−5x+6=0
    
    1. The model initially proposes an incorrect factorization.
    2. It pauses to reevaluate and notices an inconsistency in the calculated roots.
    3. Upon reflection, it correctly factors the equation and derives x=2,x=3x=2,x=3.
- This self-correcting behavior is illustrated in the table from the original paper:
    

![](https://aman.ai/primers/ai/assets/DeepSeek/2.png)

## Distillation: Reasoning in Compact Models

- DeepSeek-R1’s advanced reasoning capabilities were distilled into smaller models, including Qwen-7B and Llama-8B, through an optimized training pipeline designed to preserve reasoning depth while reducing computational complexity.

### Implementation Details

1. **Teacher-Student Paradigm**:
    - DeepSeek-R1 was used as the “teacher” model.
    - The distilled models (e.g., Qwen-7B, Llama-8B) were fine-tuned on 800K reasoning-related samples generated by DeepSeek-R1.
2. **Training Process**:
    - Unlike RL-based training for DeepSeek-R1, distilled models were trained primarily using SFT.
    - The dataset included:
        - 600K reasoning-based samples covering math, logical reasoning, and coding.
        - 200K general-purpose samples to ensure well-rounded performance.
3. **Comparison Against RL Training**:
    - Experiments showed that distilling reasoning behaviors from DeepSeek-R1 was significantly more effective than training smaller models from scratch using RL.
    - A direct RL-trained Qwen-32B model underperformed compared to the distilled DeepSeek-R1-Distill-Qwen-32B, highlighting the efficiency of distillation in preserving complex reasoning patterns.
4. **Performance Metrics:**
    - The table below showcases how distilled DeepSeek-R1 models compare against non-reasoning models like GPT-4o and larger models like OpenAI o1-mini.

![](https://aman.ai/primers/ai/assets/DeepSeek/3.png)

## Results

- The plot below from the [paper](https://arxiv.org/abs/2501.12948) illustrates the performance of DeepSeek-R1 across multiple benchmarks, showing it is on par with or even surpassing OpenAI’s models in several areas:
    
    ![](https://aman.ai/primers/ai/assets/DeepSeek/1.png)
    
    - **Mathematical Reasoning**: Achieved a 97.3% pass rate on MATH-500, outperforming previous open-source models.
    - **Code Competitions**: Placed in the 96.3rd percentile on Codeforces, equivalent to expert-level human competitors.
    - **General Knowledge**: Scored 90.8% on MMLU, demonstrating strong performance in broad knowledge domains.
- DeepSeek-R1 represents a major leap in the ability of LLMs to develop, refine, and transfer complex reasoning skills. Its RL-based self-evolution and highly effective distillation pipeline set a new standard for reasoning models, enabling smaller models to achieve state-of-the-art performance with minimal computational overhead.
    

### Average Response Length vs. Timesteps

- The plot below from the [paper](https://arxiv.org/abs/2501.12948) illustrates the average response length of DeepSeek-R1-Zero on the training set during the RL process. DeepSeek-R1-Zero naturally learns to use longer CoT to solve complex reasoning problems with more thinking time.

![](https://aman.ai/primers/ai/assets/DeepSeek/AvgResponseLength.jpg)

### Comparison of DeepSeek-R1 and DeepSeek-R1-Zero

- DeepSeek-R1 and DeepSeek-R1-Zero represent two different approaches to RL training for enhancing reasoning capabilities in LLMs. The fundamental distinction between these models lies in their training methodologies, resulting in notable differences in their overall performance and usability.

#### Training Approach

- DeepSeek-R1-Zero is trained purely via RL, without any SFT as a cold start. This allows the model to develop reasoning capabilities through self-evolution but leads to certain drawbacks such as poor readability and language mixing.
- DeepSeek-R1, on the other hand, incorporates a multi-stage training process that begins with a cold-start SFT phase using high-quality long CoT data, followed by RL. This additional step helps improve stability, readability, and overall performance.

#### Performance Differences

- The differences in training methodologies translate into substantial variations in benchmark performance:

|**Model**|**AIME 2024 (Pass@1)**|**MATH-500 (Pass@1)**|**GPQA Diamond (Pass@1)**|**LiveCodeBench (Pass@1)**|**Codeforces (Rating)**|
|---|---|---|---|---|---|
|**DeepSeek-R1**|**79.8%**|**97.3%**|**71.5%**|**65.9%**|**2029**|
|**DeepSeek-R1-Zero**|71.0%|95.9%|73.3%|50.0%|1444|

- DeepSeek-R1 achieves significantly higher performance across math reasoning (MATH-500), general knowledge (GPQA Diamond), and code competition benchmarks (Codeforces) compared to DeepSeek-R1-Zero.
- The improved LiveCodeBench score suggests better performance in software engineering-related tasks.
    
- The following plot from the paper shows the AIME accuracy of DeepSeek-R1-Zero during training. For each question, they sample 16 responses and calculate the overall average accuracy to ensure a stable evaluation.

![](https://aman.ai/primers/ai/assets/DeepSeek/AIME-DeepSeek-R1-Zero.jpg)

#### Readability and Language Consistency

- DeepSeek-R1-Zero, while effective in reasoning, suffers from language mixing and poor readability since it lacks constraints on output formatting.
- DeepSeek-R1 significantly improves readability by enforcing structured Chain-of-Thought reasoning and incorporating additional rejection sampling and supervised fine-tuning for human-friendly outputs.

#### Self-Evolution and “Aha Moments”

- One of the key observations during DeepSeek-R1-Zero training was the emergence of an “Aha Moment”, where the model learned to revise its reasoning process independently. This phenomenon underscores the potential of RL in developing sophisticated reasoning behaviors.
- However, DeepSeek-R1 further refines this capability by integrating rejection sampling, which filters out incorrect or incoherent responses, leading to a more robust and structured reasoning process.

## Prompt Template

- Per OpenAI co-founder [Greg Brockman](https://www.linkedin.com/in/thegdb), the following prompt template lists the breakdown of an o1 prompt which shows how to structure your prompts for more accurate, useful results.
- It includes:
    - Goal: What you want.
    - Return/Output Format: How you want it.
    - Warnings: What to watch out for.
    - Context: Extra details to improve accuracy.

![](https://aman.ai/primers/ai/assets/DeepSeek/prompt-template.jpg)

- However, having the context go first in the prompt (while keeping the other of the other elements unchanged), might be more beneficial in some scenarios.

## Open Questions

- As shown in the figure below ([source](https://huggingface.co/blog/open-r1)), making a powerful reasoning model is now very simple if you have access to a capable base model and a high-quality data mixture:

![](https://aman.ai/primers/ai/assets/DeepSeek/reasoningLLM.png)

- Despite DeepSeek-R1’s advances, several open questions remain regarding its development and optimal implementation:
    
    - **Data Collection**: How were the reasoning-specific datasets curated? Understanding the sources and selection criteria for data is crucial for replicating and improving the model’s performance.
    - **Model Training**: No training code was released by DeepSeek, leaving uncertainty about which hyperparameters work best and how they differ across model families and scales.
    - **Scaling Laws**: What are the compute and data trade-offs in training reasoning models? Identifying these relationships is critical for optimizing future models.

## Other Reasoning Models

### [QwQ: Reflect Deeply on the Boundaries of the Unknown](https://qwenlm.github.io/blog/qwq-32b-preview/)

- Developed by the Qwen Team, QwQ-32B-Preview is an experimental research model focusing on advancing AI reasoning.
- The model embodies a philosophical approach to problem-solving, constantly questioning its assumptions and refining its reasoning.
- **Core strengths**: Excels in mathematics and coding, showcasing deep analytical skills when given time to reflect on its reasoning process.
- **Limitations**: May exhibit recursive reasoning loops, unexpected language mixing, and requires enhanced safety measures for reliable deployment.
- **Benchmark Performance**:
    - **GPQA** (Graduate-Level Google-Proof Q&A): 65.2% – demonstrating strong scientific reasoning.
    - **AIME** (American Invitational Mathematics Exam): 50.0% – highlighting strong math problem-solving skills.
    - **MATH-500**: 90.6% – exceptional performance across various math topics.
    - **LiveCodeBench**: 50.0% – proving solid real-world programming capabilities.
- **Reasoning Approach**:
    - Uses deep introspection and self-dialogue to refine answers.
    - Prioritizes reflection over quick responses, mirroring human-like problem-solving strategies.
- **Future Directions**: The research extends into process reward models, LLM critique, multi-step reasoning, and reinforcement learning with system feedback.
- QwQ represents an evolving frontier in AI reasoning, pushing boundaries in understanding and self-correction.

### [s1: Simple Test-Time Scaling](https://arxiv.org/abs/2501.19393)

- This paper by Muennighoff et al. from Stanford and UW introduces test-time scaling, a method that improves reasoning performance in large language models (LLMs) by leveraging extra compute at inference time. The authors propose budget forcing, a simple intervention that controls the duration of the model’s reasoning process, allowing it to self-correct and refine its answers.
- **Main Contributions:**
    1. **Dataset Creation (s1K):**
        - A small dataset of 1,000 high-quality reasoning questions was curated from an initial pool of 59,000 samples.
        - Selection was based on three criteria: difficulty, diversity, and quality.
        - The final dataset was distilled from Google’s Gemini Thinking Experimental API.
    2. **Budget Forcing (Test-Time Scaling Method):**
        - Allows control over how long the model “thinks” before generating an answer.
        - **Two key techniques:**
            - **Early termination:** If the model exceeds a threshold of “thinking tokens,” it is forced to provide an answer.
            - **Extended reasoning:** The model is encouraged to continue reasoning by appending “Wait” to the generation when it tries to stop.
    3. **Fine-Tuned Model (s1-32B):**
        - The Qwen2.5-32B-Instruct model was fine-tuned on s1K in just 26 minutes on 16 NVIDIA H100 GPUs.
        - This model outperformed OpenAI’s o1-preview on math reasoning tasks like MATH and AIME24.
    4. **Experimental Results:**
        - **Scaling performance:** Budget forcing allowed the model to exceed its baseline performance without test-time intervention.
        - **Competitiveness:** s1-32B outperformed larger closed-source models and was the most sample-efficient among open-weight models.
    5. **Ablations & Comparisons:**
        - **Dataset selection:** Carefully selected 1,000 samples performed better than using all 59,000 samples.
        - **Test-time scaling methods:** Budget forcing showed superior control and performance compared to majority voting, rejection sampling, and conditional control methods.
        - **Parallel vs. Sequential Scaling:** Budget forcing (sequential) was more effective than parallel methods like majority voting.
- **Key Results:**
    - The s1-32B model, fine-tuned on just 1,000 reasoning examples, achieved 56.7% accuracy on AIME24, 93.0% on MATH500, and 59.6% on GPQA Diamond. Without any test-time intervention, the model’s AIME24 score was 50%, demonstrating that test-time scaling via budget forcing leads to significant improvements.
    - By comparison, OpenAI’s o1-preview achieved 44.6% on AIME24, 85.5% on MATH500, and 73.3% on GPQA Diamond. Other open-weight models like DeepSeek r1 outperformed s1-32B but required over 800,000 training examples, while s1-32B achieved strong reasoning performance with only 1,000 carefully selected samples. The base model (Qwen2.5-32B-Instruct), before fine-tuning, scored just 26.7% on AIME24, highlighting the significant impact of s1K fine-tuning and test-time scaling.
- **Conclusion:**
    - Test-time scaling via budget forcing is a lightweight yet powerful method for improving reasoning performance.
    - Fine-tuning on just 1,000 carefully selected examples can match or outperform models trained on hundreds of thousands of samples.
    - The approach is open-source, providing a transparent and reproducible path to improving LLM reasoning abilities.
- [Code](https://github.com/simplescaling/s1)

### [Sky-T1](https://novasky-ai.github.io/posts/sky-t1/)

- This blog by the NovaSky team at UC Berkeley introduces Sky-T1-32B-Preview, an open-source reasoning model that achieves performance comparable to o1-preview on reasoning and coding benchmarks while being trained for under $450. All code, data, and model weights are publicly available.
    
- **Motivation:** Current state-of-the-art reasoning models like o1 and Gemini 2.0 demonstrate strong reasoning abilities but remain closed-source, limiting accessibility for academic and open-source research. Sky-T1 addresses this gap by providing a high-performing, fully transparent alternative.
    
- **Key Contributions:**
    - **Fully Open-Source:** Unlike closed models, Sky-T1 releases all resources—data, training code, technical report, and model weights—allowing for easy replication and further research.
    - **Affordable Training:** Sky-T1-32B-Preview was trained for only $450, leveraging Qwen2.5-32B-Instruct as a base model and fine-tuning it using 17K curated training samples.
    - **Dual-Domain Reasoning:** Unlike prior efforts that focused solely on math reasoning (e.g., STILL-2, Journey), Sky-T1 excels in both math and coding within a single model.
- **Data Curation:**
    - Uses QwQ-32B-Preview, an open-source model with reasoning capabilities comparable to o1-preview.
    - Reject sampling ensures high-quality training data by filtering incorrect samples through exact-matching (for math) and unit test execution (for coding).
    - Final dataset includes 5K coding problems (APPs, TACO), 10K math problems (AIME, MATH, Olympiad), and 1K science/puzzle problems (from STILL-2).
- **Training Details:**
    
    - Fine-tuned on Qwen2.5-32B-Instruct for 3 epochs with a learning rate of 1e-5 and a batch size of 96.
    - Training completed in 19 hours on 8 H100 GPUs, utilizing DeepSpeed Zero-3 offload for efficiency.
    - The following figure from the blog shows the training flow of Sky-T1:
    
    ![](https://aman.ai/images/papers/Sky-T1.jpg)
    
- **Evaluation and Results:**
    - Matches or surpasses o1-preview in multiple reasoning and coding benchmarks:
        - **Math500:** 82.4% (vs. 81.4% for o1-preview)
        - **AIME 2024:** 43.3% (vs. 40.0% for o1-preview)
        - **LiveCodeBench-Easy:** 86.3% (close to 92.9% of o1-preview)
        - **LiveCodeBench-Hard:** 17.9% (slightly ahead of 16.3% for o1-preview)
    - Performs competitively with QwQ (which has a closed dataset) while remaining fully open-source.
- **Key Findings:**
    - **Model size matters:** Smaller models (7B, 14B) showed only modest gains, with 32B providing a significant leap in performance.
    - **Data mixture impacts performance:** Incorporating math-only data initially boosted AIME24 accuracy from 16.7% to 43.3%, but adding coding data lowered it to 36.7%. A balanced mix of complex math and coding problems restored strong performance in both domains.
- **Conclusion:** Sky-T1-32B-Preview proves that high-level reasoning capabilities can be replicated affordably and transparently. By open-sourcing all components, it aims to empower the academic and open-source communities to drive further advancements in reasoning model development.
- [Code](https://github.com/novasky-ai/sky-t1-32b-preview)

### [Kimi K1.5: Scaling Reinforcement Learning with LLMs](https://arxiv.org/abs/2501.12599)

- This paper by the Kimi Team proposes Kimi K1.5, a state-of-the-art multimodal large language model (LLM) trained with reinforcement learning (RL). Unlike traditional LLMs that rely solely on pretraining and supervised fine-tuning, Kimi K1.5 expands its learning capabilities by leveraging long-context RL training, enabling it to scale beyond static datasets through reward-driven exploration. Kimi K1.5 demonstrates that scaling reinforcement learning with long-context training significantly improves LLM performance. The model leverages optimized learning algorithms, partial rollouts, and efficient policy optimization to achieve strong RL results without relying on computationally expensive techniques like Monte Carlo tree search.
- Additionally, the long-to-short (L2S) transfer process enables short-CoT models to inherit reasoning abilities from long-CoT models, drastically improving token efficiency while maintaining high performance.
- The model achieves state-of-the-art performance across multiple benchmarks. It scores 77.5 Pass@1 on AIME 2024, 96.2 Exact Match on MATH 500, 94th percentile on Codeforces, and 74.9 Pass@1 on MathVista, matching OpenAI’s o1 model. Additionally, its short-CoT variant outperforms GPT-4o and Claude Sonnet 3.5 by a wide margin, achieving up to 550% improvement on some reasoning tasks.
- **Key Contributions**:
    - **Long-context scaling:** Kimi K1.5 scales RL training to a 128K token context window, demonstrating continuous improvements in reasoning performance as the context length increases. Instead of re-generating full sequences, it employs partial rollouts to reuse previous trajectories, making training more efficient.
        
    - **A simplified yet powerful RL framework:** Unlike traditional RL-based models, Kimi K1.5 does not rely on complex techniques such as Monte Carlo tree search, value functions, or process reward models. Instead, it employs chain-of-thought (CoT) reasoning, allowing the model to develop planning, reflection, and correction capabilities without computationally expensive search mechanisms.
        
    - **Advanced RL optimization techniques:** Kimi K1.5 introduces a variant of online mirror descent for policy optimization, incorporating length penalties, curriculum sampling, and prioritized sampling to further enhance training efficiency and prevent overthinking.
        
    - **Multimodal capabilities:** The model is jointly trained on text and vision data, enabling it to reason across modalities. It performs well in OCR-based tasks, chart interpretation, and vision-based mathematical reasoning.
        
    - **Long-to-Short (L2S) Training:** The model introduces long2short methods that transfer reasoning patterns from long-CoT models to short-CoT models. These techniques significantly improve token efficiency, allowing the short-CoT version to achieve state-of-the-art results on benchmarks like AIME 2024 (60.8 Pass@1) and MATH 500 (94.6 Exact Match), surpassing GPT-4o and Claude Sonnet 3.5.
        
- **Technical Details**:
    - **Training Approach**:
    - The development of Kimi K1.5 involves multiple stages:
        - **Pretraining:** The base model is trained on a diverse dataset spanning English, Chinese, code, mathematics, and general knowledge.
        - **Vanilla Supervised Fine-Tuning (SFT):** The model is refined using a mix of human-annotated and model-generated datasets, ensuring high-quality responses.
        - **Long-CoT Fine-Tuning:** A warmup phase introduces structured reasoning, teaching the model essential skills such as planning, evaluation, reflection, and exploration.
        - **Reinforcement Learning (RL):** The model is further optimized with reward-based feedback, strengthening its ability to reason through complex problems.
        - To ensure optimal RL training, Kimi K1.5 employs a carefully curated prompt set that spans multiple domains, balancing difficulty levels and ensuring robust evaluability. It also applies curriculum sampling (starting with easy tasks before progressing to harder ones) and prioritized sampling (focusing on problems where the model underperforms).
- **Reinforcement Learning Infrastructure**:
    
    - Kimi K1.5 leverages an advanced RL training infrastructure to scale efficiently:
        - **Partial Rollouts:** The model segments long responses into smaller chunks, preventing lengthy reasoning trajectories from slowing down training. This method allows parallel training of both long and short responses, maximizing compute efficiency.
        - **Hybrid Training Deployment:** Training is conducted using Megatron, while inference is performed on vLLM, allowing dynamic scaling of resources.
        - **Code Sandbox for Coding RL:** The model uses an automated test case generation system to evaluate coding solutions. It is optimized with fast execution techniques like Crun and Cgroup reuse to improve training speed and stability.
    - The following figure from the paper shows the Kimi K1.5, a large scale reinforcement learning training system for LLM.
    
    ![](https://aman.ai/images/papers/Kimi-K1.5.jpg)
    
- **Evaluation & Results**:
    - Kimi K1.5 achieves state-of-the-art results across multiple benchmarks:
        1. **Long-CoT Model Performance:**
            - It matches or surpasses OpenAI’s o1 model in key reasoning tasks.
            - On MATH 500, Kimi K1.5 achieves 96.2 Exact Match, outperforming other open-source models such as QwQ-32B (90.6).
            - On AIME 2024, it reaches 77.5 Pass@1, improving over QwQ-32B (63.6).
            - For coding tasks, it ranks in the 94th percentile on Codeforces, surpassing QwQ-32B (62nd percentile).
            - In vision-based reasoning, it scores 74.9 Pass@1 on MathVista, ahead of OpenAI’s o1-mini (71.0).
        2. **Short-CoT Model Performance:**
            - Kimi K1.5’s short-CoT model significantly outperforms GPT-4o and Claude Sonnet 3.5 on mathematical and coding reasoning tasks.
            - It achieves 94.6 Exact Match on MATH 500, whereas GPT-4o scores 74.6 and Claude Sonnet 3.5 scores 78.3.
            - On AIME 2024, Kimi K1.5 short-CoT achieves 60.8 Pass@1, far exceeding GPT-4o (9.3) and Claude Sonnet 3.5 (16.0).
            - In LiveCodeBench, the model scores 47.3 Pass@1, outperforming GPT-4o (33.4) and Claude Sonnet 3.5 (36.3).
- **Ablation Studies**:
    - Scaling Context Length vs Model Size:
        - Smaller models can match the reasoning ability of larger models if trained with long-CoT and RL.
        - However, larger models remain more token-efficient, meaning they require fewer tokens to achieve similar performance.
    - Negative Gradients vs ReST (Reward-based Supervised Tuning):
        - Kimi K1.5 outperforms ReST-based approaches by leveraging negative gradients during policy optimization, leading to more efficient training.
    - Curriculum Sampling vs Uniform Sampling:
        - Models trained with curriculum sampling (progressing from easy to hard problems) outperform those trained with uniform sampling.
        - This approach accelerates learning and improves generalization on test problems.
- [Code](https://github.com/MoonshotAI/Kimi-k1.5)

### [Open-R1](https://huggingface.co/blog/open-r1)

- While DeepSeek-R1 provides open weights, the datasets and code used in training remain proprietary. The aforementioned questions have driven the [Open-R1](https://huggingface.co/blog/open-r1) project, an initiative to systematically reconstruct DeepSeek-R1’s data and training pipeline as open-source, validate its claims, and push the boundaries of open reasoning models.
- The motivation behind building [Open-R1](https://github.com/huggingface/open-r1) is to provide transparency on how RL can enhance reasoning, share reproducible insights with the open-source community, and create a foundation for future models to leverage these techniques.

#### Objectives of Open-R1

1. **Reproducing R1-Distill Models**: By distilling a high-quality reasoning dataset from DeepSeek-R1, Open-R1 aims to replicate the R1-Distill models faithfully.
2. **Replicating the RL Training Pipeline**: A critical component of DeepSeek-R1 is its RL-based training methodology. Open-R1 will curate large-scale datasets for mathematics, reasoning, and code to enable this training process.
3. **Advancing Multi-Stage Training**: Demonstrating the full transition from a base model through SFT to RL will be a key milestone, ensuring a reproducible and scalable methodology.

- As shown in the figure below ([source](https://huggingface.co/blog/open-r1)), here’s the Open-R1 plan:

![](https://aman.ai/primers/ai/assets/DeepSeek/open-r1-steps.png)

#### Impact on the Community

- **Accessible Reasoning Models**: Open-R1’s synthetic datasets will allow anyone to fine-tune existing or new LLMs for reasoning tasks simply by leveraging these datasets.
- **Open RL Recipes**: The initiative will provide well-documented RL methodologies that can serve as a foundation for future research and experimentation.
- **Exploring Beyond Math**: While mathematical reasoning is a primary focus, Open-R1 will explore extensions into other domains, including programming and scientific applications such as medicine, where reasoning models can make a significant impact.

## [DeepSeek R1-1776](https://www.perplexity.ai/hub/blog/open-sourcing-r1-1776)

- DeepSeek R1-1776 is an open-sourced version of the DeepSeek-R1 large language model, released by Perplexity. It has been post-trained to remove censorship and provide accurate, unbiased, and factual responses, particularly in politically sensitive areas.
- The original DeepSeek-R1 often avoided or deflected sensitive topics—especially those censored by the Chinese Communist Party (CCP)—by reverting to government-aligned talking points. This limited its usefulness for global users seeking uncensored information and objective analysis.
- **R1 Post-Training Process**:
    - **Data Collection for Post-Training**: Perplexity identified ~300 CCP-censored topics with help from human experts and used these to train a multilingual censorship classifier. This classifier was used to mine a set of 40,000 multilingual user prompts from Perplexity’s customer data—explicitly permissioned and stripped of any PII—for model training.
    - **Generating High-Quality Responses**: One major challenge was collecting factual, thoughtful responses to these censored prompts. Perplexity focused on gathering completions that included strong reasoning and diverse perspectives. Ensuring chain-of-thought reasoning traces was key to maintaining model depth.
    - **Post-Training with Nvidia NeMo 2.0**: The post-training process was implemented using an adapted version of Nvidia’s NeMo 2.0 framework. This was designed to effectively de-censor the model while preserving its performance on academic and internal benchmarks, particularly for reasoning and factual accuracy.
    - **Rigorous Evaluation for Quality and Integrity**: A multilingual evaluation set of over 1,000 examples was created to assess the model’s responses on censored and sensitive topics. Human annotators and LLM-based judges were used to score the likelihood of evasion or sanitization. The results showed that R1-1776 retained reasoning strength while eliminating censorship tendencies.

## Open-Source Reasoning Datasets

1. [OpenThoughts](https://huggingface.co/datasets/open-thoughts/OpenThoughts-114k): 114k samples distilled from R1 on math, code, and science.
2. [R1-Distill-SFT](https://huggingface.co/datasets/ServiceNow-AI/R1-Distill-SFT): 1.7M samples distilled from R1-32B on NuminaMath and Allen AI’s Tulu.

## FAQs

### Is GRPO a Policy Gradient Algorithm?

- Yes, GRPO is a policy gradient algorithm. GRPO is a variant of PPO, which is itself a well-established policy gradient method.
- GRPO retains the core idea of using policy gradients but modifies the estimation of the advantage function by eliminating the need for a value (critic) model, instead using group-based reward comparisons to estimate the baseline. This makes GRPO more computationally efficient than traditional PPO while still relying on the same underlying reinforcement learning principles. So, GRPO falls squarely within the family of policy gradient algorithms.

### Is GRPO an Actor-critic Algorithm?

- No, GRPO is not an actor-critic algorithm. According to the paper, GRPO is explicitly introduced as a variant of PPO (Proximal Policy Optimization), which foregoes the critic model. Instead of using a value function (critic) to compute the advantage estimates like PPO does, GRPO estimates the baseline using _group scores_ derived from multiple sampled outputs per prompt. This significantly reduces the memory and computational burden compared to PPO.
- Here’s the key quote from the paper:

> “GRPO foregoes the critic model, instead estimating the baseline from group scores, significantly reducing training resources compared to Proximal Policy Optimization (PPO).”

- Since actor-critic methods, by definition, require both an actor (policy) and a critic (value estimator), GRPO does not qualify as an actor-critic algorithm.

### Can GRPO be Applied to Outcome Supervision or Process Supervision or Both? How is the Advantage Computed from Reward in Either Case?

- GRPO flexibly handle different types of reward structures, thus supporting both outcome supervision and process supervision. Outcome supervision is simpler and computationally cheaper, while process supervision allows for more targeted improvements in reasoning quality.
- Here’s how GRPO can be applied to outcome supervision and process supervision:

#### Outcome Supervision

- In outcome supervision, GRPO provides a single scalar reward at the end of each model-generated output. This reward is applied uniformly to all tokens in the output, making it a straightforward method for reinforcement learning.
- For each question qq, a group of G outputs is sampled from the old policy model πθoldπθold, denoted as:

{o1,o2,…,oG}∼πθold(O|q){o1,o2,…,oG}∼πθold(O|q)

- A reward model assigns a scalar reward to each output:

{r1,r2,…,rG}{r1,r2,…,rG}

- These rewards are then normalized across the group using the sample mean and standard deviation:

r̃ g=rg−1G∑Gj=1rj1G∑Gj=1(rj−1G∑Gk=1rk)2‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾√,for g=1,2,…,Gr~g=rg−1G∑j=1Grj1G∑j=1G(rj−1G∑k=1Grk)2,for g=1,2,…,G

- The resulting normalized reward r̃ gr~g is then used as the advantage value for all tokens tt in the corresponding output ogog:

Â g,t=r̃ g,∀t∈ogA^g,t=r~g,∀t∈og

- This token-level advantage is plugged into the GRPO policy update objective, optimizing the model without the need for a critic (value function). The policy is updated using a clipped surrogate objective (as in PPO), but with these normalized group-based advantages.

#### Process Supervision

- Process supervision extends GRPO by providing rewards at intermediate reasoning steps, rather than only at the final output. This enables fine-grained credit assignment to different parts of the model’s reasoning.
- For each question qq, again a group of G outputs {o1,o2,…,oG}{o1,o2,…,oG} is sampled. Then, a process reward model evaluates each output step-by-step, assigning a list of scalar rewards per step. Let the rewards for each output ogog be:

{rgindex(1),rgindex(2),…,rgindex(Kg)}{rindex(1)g,rindex(2)g,…,rindex(Kg)g}

- Here, index(j)index(j) refers to the ending token index of the jj-th step in output ogog, and KgKg is the total number of reasoning steps in that output.
- These step-level rewards across all G outputs are collected into a set RR, then normalized:

r̃ gindex(j)=rgindex(j)−mean(R)std(R)r~index(j)g=rindex(j)g−mean(R)std(R)

- The token-level advantage Â g,tA^g,t for token tt in output ogog is computed by summing the normalized rewards of all steps whose indices are greater than or equal to the token position:

Â g,t=∑index(j)≥tr̃ gindex(j)A^g,t=∑index(j)≥tr~index(j)g

- This allows the model to receive differentiated feedback for each part of its reasoning trace, encouraging improvement not just in final correctness but in intermediate steps as well.
- As with outcome supervision, these advantages are used in the GRPO objective to optimize the policy.

### How is a Reward Model Different from a Value/critic Model in Policy Optimization Algorithms Such As GRPO?

- The reward model and value (critic) model serve different roles in policy optimization, and GRPO makes a key distinction by removing the critic altogether. Here’s a clear breakdown.

#### Reward Model

- **Purpose:** Scores the quality of an entire output (or intermediate steps) based on some external or learned metric.
- **Input:** (Question, generated output)
- **Output:** A scalar reward, either:
    - At the end of the output (outcome supervision), or
    - At each reasoning step (process supervision)
- Learned from human preferences, correctness signals, or labels (e.g., “Output A is better than B”).
- Used to train the policy, by converting its scores into advantages for policy updates.
- In GRPO, this is the core signal used for policy optimization.

#### Value Model (Critic)

- **Purpose:** Advantage estimation, which is the task of estimating/predicting the expected/future reward (value) of being in a given state — it serves as a baseline to reduce variance when computing the advantage function.
- **Input:** (State or partial sequence)
- **Output:** Expected future reward from that point
- Trained during RL to minimize error between predicted and actual rewards.
- Used in PPO and other actor-critic methods, it helps stabilize training by estimating how good a state is, independent of specific actions taken.

#### Key Differences in GRPO

- GRPO does not use a value model.
    - Instead, it uses group-based reward normalization to compute advantages, acting as a statistical baseline.
    - This simplifies training and reduces memory cost, especially important for large language models.
- PPO and other classic methods rely on a trained value model, which is separate from the reward model and needs its own optimization loop.

#### Summary

|**Feature**|**Reward Model**|**Value/Critic Model**|
|---|---|---|
|What it predicts|External reward|Expected future reward|
|Input|Full or partial generated output|State or token context|
|Used in|GRPO, PPO, DPO, RFT, etc.|PPO, A2C, other actor-critic|
|Trained from|Human preferences / correctness|Bootstrapped from past rewards|
|Purpose|Supervises learning|Reduces variance in training|
|Required in GRPO?|Yes|**No**|

### In the Equation for GRPO, What is the Role of the Old Policy Compared to the Reference Policy?

- In the equation for GRPO, the old policy and the reference policy serve distinct roles, both contributing to stable and effective training but in different ways:
    
- **Old Policy:**
    - Used to generate a group of output samples o1,o2,…,oGo1,o2,…,oG for each input question qq.
    - These outputs are scored by the reward model, and their group-wise average reward is used as the baseline to compute advantages.
    - The ratio between the current policy piθpiθ and the old policy πoldπold is used in the surrogate objective, similar to PPO, to ensure updates do not diverge too much from previously good-performing behavior.
- **Reference Policy:**
    - Typically set to the initial supervised fine-tuned (SFT) model at each iteration.
    - Used for KL divergence regularization: a penalty is applied if the current policy πθπθ deviates too far from this stable reference.
    - Helps prevent over-optimization or collapse by anchoring the training process to a known good policy.
- So, in summary:
    - The old policy is dynamic and updated throughout training to generate new candidate outputs.
    - The reference policy is fixed per iteration and acts as a stability anchor through KL regularization.
- This dual-role setup enables GRPO to maintain training stability without requiring a value function, which is traditionally needed in PPO, thus saving computational resources and simplifying implementation.

### Why is the PPO/GRPO Objective Called a Clipped “surrogate” Objective?

- The PPO (and its variants such as GRPO) objective is called a surrogate objective because it doesn’t directly optimize the true reinforcement learning objective — the expected return — but instead optimizes a _proxy_ that is easier and safer to compute. Here’s why:
- **True RL Objective is Unstable or Intractable:**
    - The actual objective in RL is to maximize expected reward over trajectories, which involves high variance and instability during training, especially for large models like LLMs. It often requires estimating complex quantities like the value function accurately over time, which is difficult in practice.
- **Surrogate Objectives Improve Stability:**
    - Surrogate objectives simplify this by using:
        - Advantage estimates to approximate how much better a new action is compared to the old one.
        - Importance sampling ratios (like \frac{\pi_{\theta}{\pi_{old}}\frac{\pi_{\theta}{\pi_{old}}) to correct for the shift in policy.
        - Clipping (in PPO and GRPO) to avoid overly large policy updates that might destabilize training.
- **Practical Optimization Benefits:**
    - By approximating the true objective, surrogate objectives allow for stable and efficient policy updates, which are essential in fine-tuning large models via reinforcement learning.
- In summary, it’s called a surrogate because it’s a well-designed stand-in for the true goal of maximizing reward, tailored to be safer and more effective for gradient-based optimization.

### What are Some Considerations around the Reasoning Tokens Budget in Reasoning LLMs?

- In reasoning LLMs, the **reasoning token budget** refers to how many tokens the model is allowed to generate during its reasoning process (e.g., for chain-of-thought or program-of-thought generation). Setting this budget is a tradeoff between solution quality and efficiency, and it can depend on several factors:
    
    - **Model Size and Capacity**:
        - Larger models can generally reason more effectively with fewer tokens, while smaller models may need more tokens to reach the same quality.
        - However, allowing too many tokens may lead to overthinking or hallucinations, especially in smaller models.
    - **Task Complexity**:
        - For simple arithmetic or factual recall, a small budget (e.g., 32–64 tokens) might be enough.
        - For more complex mathematical reasoning (e.g., proofs, multi-step algebra), models may need 128–512 tokens or more.
    - **Supervised vs. RL Fine-Tuning**:
        - During supervised fine-tuning, the reasoning length often follows the solution length in the training data.
        - During reinforcement learning, especially with process supervision, the budget needs to be high enough to cover multiple steps but not so high that it encourages meaningless continuation. Common budgets range from 256 to 1024 tokens.
    - **Practical Considerations**:
        - **Compute and memory constraints**: longer generations require more memory and time, which affects batch sizes and training throughput.
        - **Prompt + output length** must fit within the model’s context window (e.g., 4K or 8K tokens), especially during training with multiple examples concatenated.
    - **Empirical Tuning**:
        - In practice, the reasoning token budget is often set by experimenting: start with a safe maximum (e.g., 512 or 1024), observe performance, and adjust.
        - Some papers also dynamically adjust the budget, allowing early stopping based on certain signals (e.g., confidence, reward saturation, or solution completeness).
    - **Hard vs. Soft Budgets**:
        - **Hard budget**: fixed maximum length. The model is forcibly cut off at that token count.
        - **Soft budget**: guided by stop tokens or heuristics (e.g., end-of-solution markers, newline patterns), which allow variable-length reasoning up to a cap.
- In summary, the reasoning token budget is typically tuned based on the model size, task demands, training stage, and empirical tradeoffs. A common starting point for complex reasoning tasks (like MATH or GSM8K) is 512–1024 tokens.
    

## References

- [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948)
- [DeepSeek-R1: A Pure RL-based Reasoning Model](https://www.linkedin.com/pulse/deepseek-r1-pure-rl-based-reasoning-model-jayant-kumar-yfopc/?trackingId=Tc70aMqJS42SK6oiIPqBZA%3D%3D)
- [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300)
- [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)
- [Open-R1: a fully open reproduction of DeepSeek-R1](https://huggingface.co/blog/open-r1)
- [DeepSeek-R1: The MoE Fallacy and the True Source of Emergent Reasoning](https://medium.com/autonomous-agents/deepseek-r1-the-moe-fallacy-and-the-true-source-of-emergent-reasoning-cedba23a7788)
- [The Illustrated DeepSeek-R1](https://newsletter.languagemodels.co/p/the-illustrated-deepseek-r1)

-  [](https://github.com/amanchadha)|  [](https://citations.amanchadha.com/)|  [](https://twitter.com/i_amanchadha)|  [](mailto:hi@aman.ai)| 

[www.amanchadha.com](https://www.amanchadha.com/)