[Distilled AI](https://aman.ai/primers/ai/)[Back to aman.ai](https://aman.ai/)

# Primers • DeepSeek-R1

- [Introduction](https://aman.ai/primers/ai/deepseek-R1/#introduction)
- [Architectural Foundations](https://aman.ai/primers/ai/deepseek-R1/#architectural-foundations)
    - [Overview](https://aman.ai/primers/ai/deepseek-R1/#overview)
    - [Mixture of Experts (MoE)](https://aman.ai/primers/ai/deepseek-R1/#mixture-of-experts-moe)
        - [Overview](https://aman.ai/primers/ai/deepseek-R1/#overview-1)
        - [Key Features](https://aman.ai/primers/ai/deepseek-R1/#key-features)
        - [Evolution from DeepSeek-V2 to DeepSeek-R1](https://aman.ai/primers/ai/deepseek-R1/#evolution-from-deepseek-v2-to-deepseek-r1)
            - [MoE in DeepSeek-V2](https://aman.ai/primers/ai/deepseek-R1/#moe-in-deepseek-v2)
                - [Basic Architecture of DeepSeekMoE](https://aman.ai/primers/ai/deepseek-R1/#basic-architecture-of-deepseekmoe)
                - [Device-Limited Routing](https://aman.ai/primers/ai/deepseek-R1/#device-limited-routing)
                - [Auxiliary Loss for Load Balancing](https://aman.ai/primers/ai/deepseek-R1/#auxiliary-loss-for-load-balancing)
                - [Token-Dropping Strategy](https://aman.ai/primers/ai/deepseek-R1/#token-dropping-strategy)
            - [Enhancements in DeepSeek-V3](https://aman.ai/primers/ai/deepseek-R1/#enhancements-in-deepseek-v3)
                - [Auxiliary-Loss-Free Load Balancing](https://aman.ai/primers/ai/deepseek-R1/#auxiliary-loss-free-load-balancing)
                - [Node-Limited Routing (NLR)](https://aman.ai/primers/ai/deepseek-R1/#node-limited-routing-nlr)
                - [Improved Expert Selection Mechanism](https://aman.ai/primers/ai/deepseek-R1/#improved-expert-selection-mechanism)
                - [Enhanced Sparsity Constraints with Hierarchical Gating](https://aman.ai/primers/ai/deepseek-R1/#enhanced-sparsity-constraints-with-hierarchical-gating)
                - [No Token-Dropping Strategy](https://aman.ai/primers/ai/deepseek-R1/#no-token-dropping-strategy)
            - [Enhancements in DeepSeek-R1](https://aman.ai/primers/ai/deepseek-R1/#enhancements-in-deepseek-r1)
                - [Adaptive Expert Routing with Reinforcement Learning (RL)](https://aman.ai/primers/ai/deepseek-R1/#adaptive-expert-routing-with-reinforcement-learning-rl)
                - [Hierarchical Entropy-Gated MoE (HE-MoE)](https://aman.ai/primers/ai/deepseek-R1/#hierarchical-entropy-gated-moe-he-moe)
                - [Device-Constrained Expert Allocation (DCEA)](https://aman.ai/primers/ai/deepseek-R1/#device-constrained-expert-allocation-dcea)
                - [Load-Balanced Expert Utilization with RL-Based Adjustments](https://aman.ai/primers/ai/deepseek-R1/#load-balanced-expert-utilization-with-rl-based-adjustments)
                - [Elimination of Token-Dropping Strategy](https://aman.ai/primers/ai/deepseek-R1/#elimination-of-token-dropping-strategy)
            - [Comparative Analysis](https://aman.ai/primers/ai/deepseek-R1/#comparative-analysis)
        - [Mathematical Formulation](https://aman.ai/primers/ai/deepseek-R1/#mathematical-formulation)
            - [Load Balancing Loss](https://aman.ai/primers/ai/deepseek-R1/#load-balancing-loss)
    - [Multihead Latent Attention (MLA)](https://aman.ai/primers/ai/deepseek-R1/#multihead-latent-attention-mla)
        - [Overview](https://aman.ai/primers/ai/deepseek-R1/#overview-2)
        - [Key Features](https://aman.ai/primers/ai/deepseek-R1/#key-features-1)
        - [Evolution from DeepSeek-V2 to DeepSeek-R1](https://aman.ai/primers/ai/deepseek-R1/#evolution-from-deepseek-v2-to-deepseek-r1-1)
            - [MLA in DeepSeek-V2](https://aman.ai/primers/ai/deepseek-R1/#mla-in-deepseek-v2)
                - [Low-Rank Key-Value Joint Compression](https://aman.ai/primers/ai/deepseek-R1/#low-rank-key-value-joint-compression)
                - [Decoupled Rotary Position Embedding](https://aman.ai/primers/ai/deepseek-R1/#decoupled-rotary-position-embedding)
                - [Comparison of KV Cache Requirements](https://aman.ai/primers/ai/deepseek-R1/#comparison-of-kv-cache-requirements)
            - [Enhancements in DeepSeek-V3](https://aman.ai/primers/ai/deepseek-R1/#enhancements-in-deepseek-v3-1)
                - [Further KV Cache Reduction Through Optimized Compression Techniques](https://aman.ai/primers/ai/deepseek-R1/#further-kv-cache-reduction-through-optimized-compression-techniques)
                - [Optimized Compression Formulation](https://aman.ai/primers/ai/deepseek-R1/#optimized-compression-formulation)
                - [Inference-Time Expansion](https://aman.ai/primers/ai/deepseek-R1/#inference-time-expansion)
                - [Query Compression for Activation Memory Savings](https://aman.ai/primers/ai/deepseek-R1/#query-compression-for-activation-memory-savings)
                - [Reduction in Activation Memory](https://aman.ai/primers/ai/deepseek-R1/#reduction-in-activation-memory)
                - [Enhanced Numerical Stability with FP8 Mixed Precision](https://aman.ai/primers/ai/deepseek-R1/#enhanced-numerical-stability-with-fp8-mixed-precision)
                - [Adaptive Routing for Load Balancing in MLA](https://aman.ai/primers/ai/deepseek-R1/#adaptive-routing-for-load-balancing-in-mla)
            - [Enhancements in DeepSeek-R1](https://aman.ai/primers/ai/deepseek-R1/#enhancements-in-deepseek-r1-1)
                - [RL-Guided Latent Attention Optimization](https://aman.ai/primers/ai/deepseek-R1/#rl-guided-latent-attention-optimization)
                - [Adaptive Query and Key Compression Via RL](https://aman.ai/primers/ai/deepseek-R1/#adaptive-query-and-key-compression-via-rl)
                - [Decoupled Rotary Position Embedding with Context-Specific Scaling](https://aman.ai/primers/ai/deepseek-R1/#decoupled-rotary-position-embedding-with-context-specific-scaling)
                - [FP8 Mixed Precision for MLA Stability](https://aman.ai/primers/ai/deepseek-R1/#fp8-mixed-precision-for-mla-stability)
                - [Adaptive/Dynamic Routing for Load-Balanced Attention](https://aman.ai/primers/ai/deepseek-R1/#adaptivedynamic-routing-for-load-balanced-attention)
            - [Comparative Analysis](https://aman.ai/primers/ai/deepseek-R1/#comparative-analysis-1)
        - [Implementation](https://aman.ai/primers/ai/deepseek-R1/#implementation)
            - [Background: Standard Multi-Head Attention (MHA)](https://aman.ai/primers/ai/deepseek-R1/#background-standard-multi-head-attention-mha)
            - [Low-Rank Key-Value Joint Compression](https://aman.ai/primers/ai/deepseek-R1/#low-rank-key-value-joint-compression-1)
            - [Multi-Stage Compression](https://aman.ai/primers/ai/deepseek-R1/#multi-stage-compression)
            - [Query Compression and Optimization](https://aman.ai/primers/ai/deepseek-R1/#query-compression-and-optimization)
            - [Decoupled Rotary Position Embedding (RoPE)](https://aman.ai/primers/ai/deepseek-R1/#decoupled-rotary-position-embedding-rope)
            - [Attention Computation in MLA](https://aman.ai/primers/ai/deepseek-R1/#attention-computation-in-mla)
            - [RL-Optimized MLA](https://aman.ai/primers/ai/deepseek-R1/#rl-optimized-mla)
            - [Computational and Hardware Optimization](https://aman.ai/primers/ai/deepseek-R1/#computational-and-hardware-optimization)
            - [Comparative Efficiency Analysis](https://aman.ai/primers/ai/deepseek-R1/#comparative-efficiency-analysis)
    - [Multi-Token Prediction (MTP)](https://aman.ai/primers/ai/deepseek-R1/#multi-token-prediction-mtp)
        - [Overview](https://aman.ai/primers/ai/deepseek-R1/#overview-3)
        - [Key Features](https://aman.ai/primers/ai/deepseek-R1/#key-features-2)
            - [MTP in DeepSeek-V3/DeepSeek-R1](https://aman.ai/primers/ai/deepseek-R1/#mtp-in-deepseek-v3deepseek-r1)
                - [Sequential Multi-Token Prediction Modules](https://aman.ai/primers/ai/deepseek-R1/#sequential-multi-token-prediction-modules)
                - [MTP Training Objective](https://aman.ai/primers/ai/deepseek-R1/#mtp-training-objective)
                - [Memory Optimization with Shared Embeddings and Output Heads](https://aman.ai/primers/ai/deepseek-R1/#memory-optimization-with-shared-embeddings-and-output-heads)
                - [Inference Strategy and Speculative Decoding](https://aman.ai/primers/ai/deepseek-R1/#inference-strategy-and-speculative-decoding)
                - [Ablation Studies on Multi-Token Prediction](https://aman.ai/primers/ai/deepseek-R1/#ablation-studies-on-multi-token-prediction)
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
    - [GRPO Variants](https://aman.ai/primers/ai/deepseek-R1/#grpo-variants)
        - [Decoupled Clip and Dynamic Sampling Policy Optimization (DAPO)](https://aman.ai/primers/ai/deepseek-R1/#decoupled-clip-and-dynamic-sampling-policy-optimization-dapo)
            - [DAPO vs. GRPO: Key Conceptual Differences](https://aman.ai/primers/ai/deepseek-R1/#dapo-vs-grpo-key-conceptual-differences)
        - [DAPO: Implementation Details](https://aman.ai/primers/ai/deepseek-R1/#dapo-implementation-details)
            - [Objective Function](https://aman.ai/primers/ai/deepseek-R1/#objective-function)
            - [Clip-Higher (Asymmetric Clipping)](https://aman.ai/primers/ai/deepseek-R1/#clip-higher-asymmetric-clipping)
            - [Dynamic Sampling](https://aman.ai/primers/ai/deepseek-R1/#dynamic-sampling)
            - [Token-Level Policy Gradient Loss](https://aman.ai/primers/ai/deepseek-R1/#token-level-policy-gradient-loss)
            - [Overlong Reward Shaping](https://aman.ai/primers/ai/deepseek-R1/#overlong-reward-shaping)
            - [Training Implementation Summary](https://aman.ai/primers/ai/deepseek-R1/#training-implementation-summary)
        - [DAPO vs. GRPO Summary](https://aman.ai/primers/ai/deepseek-R1/#dapo-vs-grpo-summary)
        - [Empirical Gains](https://aman.ai/primers/ai/deepseek-R1/#empirical-gains)
    - [GRPO+: a Stable Evolution of GRPO for Reinforcement Learning in DeepCoder](https://aman.ai/primers/ai/deepseek-R1/#grpo-a-stable-evolution-of-grpo-for-reinforcement-learning-in-deepcoder)
        - [Motivation for GRPO+](https://aman.ai/primers/ai/deepseek-R1/#motivation-for-grpo)
        - [Key Innovations in GRPO+](https://aman.ai/primers/ai/deepseek-R1/#key-innovations-in-grpo)
        - [GRPO+ Objective Function](https://aman.ai/primers/ai/deepseek-R1/#grpo-objective-function)
        - [Implementation Details](https://aman.ai/primers/ai/deepseek-R1/#implementation-details-1)
        - [Comparison: GRPO vs. GRPO+](https://aman.ai/primers/ai/deepseek-R1/#comparison-grpo-vs-grpo)
        - [Why GRPO+ Works](https://aman.ai/primers/ai/deepseek-R1/#why-grpo-works)
        - [Results and Performance Impact](https://aman.ai/primers/ai/deepseek-R1/#results-and-performance-impact)
- [Emergent Reasoning Behaviors](https://aman.ai/primers/ai/deepseek-R1/#emergent-reasoning-behaviors)
    - [Implementation Details](https://aman.ai/primers/ai/deepseek-R1/#implementation-details-2)
    - [Example: Quadratic Equation Solving](https://aman.ai/primers/ai/deepseek-R1/#example-quadratic-equation-solving)
- [Distillation: Reasoning in Compact Models](https://aman.ai/primers/ai/deepseek-R1/#distillation-reasoning-in-compact-models)
    - [Implementation Details](https://aman.ai/primers/ai/deepseek-R1/#implementation-details-3)
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
- [Further Reading](https://aman.ai/primers/ai/deepseek-R1/#further-reading)
- [References](https://aman.ai/primers/ai/deepseek-R1/#references)

## Introduction

- [DeepSeek-R1 and DeepSeek-R1-Zero](https://arxiv.org/abs/2501.12948) represent a landmark in reasoning-capable Large Language Models (LLMs). [Released](https://huggingface.co/deepseek-ai/DeepSeek-R1) under an MIT license, this model rivals closed-source giants like OpenAI’s o1 and o3 series while pioneering a reinforcement learning (RL)-driven framework for reasoning tasks.
- Both models leverage Group Relative Policy Optimization (GRPO), introduced in [DeepSeekMath](https://arxiv.org/abs/2402.03300), which replaces traditional methods like PPO, making training both efficient and scalable. They also utilize Multihead Latent Attention (MLA), introduced in [DeepSeek-V2](https://arxiv.org/pdf/2405.04434), which reduces computational and memory inefficiencies particularly for long-context processing by projecting Key-Query-Value (KQV) matrices into a lower-dimensional latent space.
- DeepSeek-R1-Zero demonstrates how reasoning capabilities emerge naturally purely through RL without any Supervised Fine-Tuning (SFT). By relying solely on self-evolution through RL, DeepSeek-R1-Zero naturally developed powerful reasoning behaviors but also exhibited challenges such as poor readability and language mixing. DeepSeek-R1 built upon this foundation and addressed the aforementioned issues by incorporating multi-stage training and a small amount of cold-start data to improve reasoning performance and usability.
- Through innovations like GRPO, FP8 quantization, and emergent Chain-of-Thought (CoT) reasoning, both models rival closed-source models while fostering transparency and accessibility. As the research community builds upon these innovations, DeepSeek-R1 signals a shift towards efficient, reasoning-driven AI accessible to all.
- This primer explores its architecture, multi-stage training pipeline, GRPO mechanics, and emergent reasoning behaviors, alongside how distillation propagates reasoning capabilities to smaller models.

## Architectural Foundations

- DeepSeek-R1 builds upon the foundational advancements introduced in [DeepSeek-V2](https://arxiv.org/abs/2405.04434) — specifically, Mixture of Experts (MoE) and Multihead Latent Attention (MLA) — and [DeepSeek-V3](https://arxiv.org/abs/2412.19437) — specifically, Multi-Token Prediction (MTP) — integrating cutting-edge architectural innovations that optimize both training efficiency and inference performance.
- This section provides a detailed breakdown of the architectural components that evolved from DeepSeek-V2 and DeepSeek-V3 to DeepSeek-R1, highlighting improvements that make DeepSeek-R1 a leading open-source model, capable of rivaling proprietary alternatives in reasoning efficiency and performance.

### Overview

- DeepSeek-R1 incorporates several advanced techniques to achieve remarkable efficiency improvements:
    
    1. **Mixture of Experts (MoE) Architecture**: DeepSeek-R1 utilizes a Mixture of Experts model, which decomposes a large model into smaller, specialized sub-models. This architecture allows for the activation of only relevant sub-models during specific tasks, enabling the system to operate efficiently on consumer-grade GPUs.
        
    2. **Key-Value Memory Compression via Multihead Latent Attention (MLA)**: By implementing sophisticated compression algorithms, DeepSeek-R1 achieves a 93% reduction in the storage requirements for key-value indices, which are known to consume considerable amounts of VRAM.
        
    3. **Multi-Token Prediction**: DeepSeek-R1 is designed to predict multiple tokens simultaneously rather than one at a time. This strategy effectively doubles the inference speed, enhancing overall performance.
        
    4. **Low-Precision Computation**: DeepSeek-R1 employs mixed-precision arithmetic, performing a significant portion of computations using 8-bit floating-point numbers instead of the standard 32-bit. This approach substantially reduces memory consumption and accelerates processing speeds.
        
- Collectively, these innovations contribute to DeepSeek-R1’s significant advancements in training efficiency, reportedly achieving a 45-fold improvement over previous models.
    

### Mixture of Experts (MoE)

#### Overview

- The MoE mechanism selectively activates a subset of the total model parameters at each inference step, achieving computational savings while maintaining model quality. This approach enables scaling up model parameters without a proportional increase in computational cost.
- DeepSeek-R1 refines DeepSeek-V2’s MoE framework, introducing dynamic expert routing, reinforcement learning-based load balancing, and enhanced sparsity constraints. These innovations make DeepSeek-R1 one of the most efficient and scalable open-source MoE models available.

#### Key Features

- **Reinforcement Learning-Based Expert Routing**: DeepSeek-R1 replaces static gating functions with a reinforcement learning (RL) policy to dynamically assign tokens to experts. The RL-based router optimizes expert selection by maximizing load balancing while minimizing routing entropy, leading to more efficient token-expert mapping.
    
- **Hierarchical Entropy-Gated MoE (HE-MoE)**: The expert selection process is refined with a multi-level gating mechanism. Tokens first pass through a global selection phase, followed by cluster-level pruning, and finally, an entropy-aware adjustment ensures balanced expert activation. This approach prevents expert over-specialization and improves generalization.
    
- **Device-Constrained Expert Allocation (DCEA)**: Experts are assigned based on available compute resources, reducing cross-device communication overhead. The model selects experts within a constrained pool of devices, lowering synchronization costs and increasing training efficiency.
    
- **Load-Balanced Expert Utilization with RL-Based Adjustments**: Instead of relying on auxiliary loss functions to balance load, DeepSeek-R1 dynamically adjusts expert activation probabilities using RL-based bias terms. This ensures consistent workload distribution without additional loss penalties, improving stability and convergence.
    
- **Full Token Retention (No Token Dropping)**: Unlike earlier iterations that dropped low-affinity tokens to balance computational load, DeepSeek-R1 retains all tokens during both training and inference. This ensures that no information is lost, leading to improved model coherence and generalization.
    
- **Cross-Device Communication Optimization**: With DCEA and hierarchical expert gating, DeepSeek-R1 significantly reduces inter-device communication, leading to up to a 35% decrease in latency. This optimization enhances efficiency without sacrificing model performance.
    
- **Dynamic Expert Activation**: The model adapts expert selection dynamically using learned routing strategies, ensuring efficient allocation of computational resources. This allows DeepSeek-R1 to scale effectively without a linear increase in computational cost.
    
- **Adaptive Expert Specialization**: By incorporating entropy-based constraints, DeepSeek-R1 ensures that experts remain specialized but not overly rigid. This dynamic specialization enhances both accuracy and efficiency while maintaining flexibility in expert activation.
    

#### Evolution from DeepSeek-V2 to DeepSeek-R1

##### MoE in DeepSeek-V2

- DeepSeek-V2 introduces a specialized MoE architecture called DeepSeekMoE, which optimizes model training efficiency and inference throughput while maintaining strong performance. This architecture refines expert selection, routing, and load balancing strategies to reduce computational overhead. Below, we detail the MoE-specific mechanisms in DeepSeek-V2, breaking them down into their individual components.

###### Basic Architecture of DeepSeekMoE

- DeepSeekMoE is designed with fine-grained expert segmentation and shared expert isolation, which increase specialization while reducing redundancy. The MoE architecture in DeepSeek-V2 consists of:
    - NsNs shared experts, which process all tokens.
    - NrNr routed experts, which are selectively activated for tokens based on a gating function.
    - Each token is processed by a fixed number KrKr of routed experts.
- The output of the MoE layer is computed as:
    
    h′t=ut+∑i=1NsFFN(s)i(ut)+∑i=1Nrgi,tFFN(r)i(ut)ht′=ut+∑i=1NsFFNi(s)(ut)+∑i=1Nrgi,tFFNi(r)(ut)
    
    - where:
        - FFN(s)iFFNi(s) represents a shared expert.
        - FFN(r)iFFNi(r) represents a routed expert.
        - gi,tgi,t is the gating function, determining expert selection for token tt.
- The gating function follows:
    
    gi,t={si,t,0,si,t∈Top-Kr({sj,t∣1≤j≤Nr})otherwisegi,t={si,t,si,t∈Top-Kr({sj,t∣1≤j≤Nr})0,otherwise
    
    - where si,tsi,t is the softmax-weighted token-expert affinity:
    
    si,t=Softmaxi(uTtei)si,t=Softmaxi(utTei)
    
    - where eiei is the centroid of expert ii.

###### Device-Limited Routing

- One of the major computational bottlenecks in MoE models is the communication overhead introduced by expert parallelism. To address this, DeepSeekMoE implements device-limited routing, restricting the number of devices a token’s experts can be distributed across.
- **Key implementation details:**
    - Each token first selects MM devices with the highest affinity scores.
    - The final KrKr experts are chosen only from these selected devices.
- In practice, setting M≥3M≥3 ensures performance close to unrestricted routing while significantly reducing inter-device communication.

###### Auxiliary Loss for Load Balancing

- DeepSeek-V2 employs multiple auxiliary losses to ensure balanced expert utilization, avoiding situations where certain experts become overloaded while others remain underutilized. Specifics below:
    
    - **Expert-Level Balance Loss**:
        
        - To prevent routing collapse, where only a subset of experts get trained, DeepSeek-V2 minimizes:
        
        LExpBal=α1∑i=1NrfiPiLExpBal=α1∑i=1NrfiPi
        
        - where:
            - fifi is the fraction of tokens routed to expert ii,
            - PiPi is the average probability of selecting expert ii,
            - α1α1 is a hyperparameter controlling the strength of the loss.
    - **Device-Level Balance Loss**:
        
        - To distribute computation evenly across devices, DeepSeekMoE assigns experts to DD device groups, where each group runs on a separate device. The balance loss is:
        
        LDevBal=α2∑i=1Df′iP′iLDevBal=α2∑i=1Dfi′Pi′
        
        - where f′ifi′ and P′iPi′ aggregate usage statistics across all experts on device ii.
    - **Communication Balance Loss**:
        
        - This loss ensures that each device receives an approximately equal number of tokens, preventing bottlenecks caused by excessive communication loads:
        
        LCommBal=α3∑i=1Df″iP″iLCommBal=α3∑i=1Dfi″Pi″
        
        - where f″ifi″ and P″iPi″ measure the fraction of tokens sent to device ii.

###### Token-Dropping Strategy

- While auxiliary losses improve balance, they cannot strictly guarantee uniform expert utilization. To further mitigate inefficiencies, DeepSeek-V2 implements a token-dropping strategy at the device level:
    - The computational budget per device is first estimated.
    - Tokens with the lowest affinity scores are dropped until the budget is met.
    - At least 10% of training sequences are exempted from token dropping to ensure diversity.
- This approach allows flexibility in dynamically adjusting token retention during inference based on computational constraints.

##### Enhancements in DeepSeek-V3

- DeepSeek-V3 introduces several significant improvements to the MoE framework compared to DeepSeek-V2. These enhancements primarily focus on increasing model efficiency, reducing training and inference costs, and maintaining high performance. The key improvements include an auxiliary-loss-free load balancing strategy, node-limited routing, improved expert selection mechanisms, and enhanced sparsity constraints. These advancements contribute to more efficient training, faster inference, and superior performance compared to DeepSeek-V2.

###### Auxiliary-Loss-Free Load Balancing

- In contrast to DeepSeek-V2, which relies on auxiliary losses to ensure balanced expert utilization, DeepSeek-V3 introduces an auxiliary-loss-free strategy. Instead of penalizing imbalance with additional loss terms, DeepSeek-V3 dynamically adjusts expert selection using bias terms. The expert gating function is modified as follows:
    
    g′i,t={si,t,0,si,t+bi∈Top-Kr({sj,t+bj|1≤j≤Nr})otherwisegi,t′={si,t,si,t+bi∈Top-Kr({sj,t+bj|1≤j≤Nr})0,otherwise
    
    - where bibi is a bias term adjusted dynamically based on the load of expert ii over multiple training steps:
    
    bi←bi−γif expert i is overloaded, otherwise bi←bi+γ.bi←bi−γif expert i is overloaded, otherwise bi←bi+γ.
    
- This dynamic adjustment ensures that expert load remains balanced without requiring auxiliary loss penalties, leading to better training stability and efficiency.
    

###### Node-Limited Routing (NLR)

- DeepSeek-V3 introduces Node-Limited Routing (NLR) to further optimize communication overhead in large-scale MoE training. Instead of allowing tokens to be dispatched to any expert across the model, NLR restricts the number of nodes each token can communicate with. The routing mechanism selects at most MM nodes per token, ensuring that experts are assigned in a way that minimizes inter-node synchronization.

M=∑i=1Nmax{sj,t|j∈node i}M=∑i=1Nmax{sj,t|j∈node i}

- This approach significantly reduces cross-node communication overhead, leading to faster training and inference times.

###### Improved Expert Selection Mechanism

- DeepSeek-V3 refines expert selection by incorporating a sigmoid-based token-expert affinity function instead of the softmax-based mechanism used in DeepSeek-V2. The new function is defined as:
    
    si,t=σ(uTtei)si,t=σ(utTei)
    
    - where eiei is the centroid of expert ii and σ(⋅)σ(⋅) is the sigmoid activation function. The selection process then normalizes the top-KrKr expert scores:
    
    gi,t=g′i,t∑j∈Top-Krg′j,t.gi,t=gi,t′∑j∈Top-Krgj,t′.
    
- This modification prevents extreme expert selection probabilities, leading to better load balancing and specialization.
    

###### Enhanced Sparsity Constraints with Hierarchical Gating

- To avoid over-specialization and encourage generalization, DeepSeek-V3 introduces hierarchical gating. Unlike traditional top-KK gating, this method applies sparsity constraints at multiple levels:
    
    - **Global Selection:** Initial selection of NgNg experts at a coarse level.
    - **Cluster-Level Pruning:** Further filtering experts within selected clusters to obtain KrKr experts.
    - **Entropy-Based Adjustments:** Adjusting expert activation probabilities based on entropy constraints to avoid extreme sparsity.
- Mathematically, the entropy-based adjustment modifies gating scores as follows:
    
    gi,t=gi,t×(1−λ⋅H(g1:Nr,t))gi,t=gi,t×(1−λ⋅H(g1:Nr,t))
    
    - where H(⋅)H(⋅) is the entropy function and λλ is a regularization coefficient controlling the trade-off between uniform selection and specialization.

###### No Token-Dropping Strategy

- DeepSeek-V2 implemented a token-dropping strategy to balance computation per device. However, DeepSeek-V3’s enhanced load-balancing mechanism eliminates the need for token dropping, ensuring 100% token retention during both training and inference. This improves generalization and avoids loss of information during model updates.

##### Enhancements in DeepSeek-R1

- DeepSeek-R1 introduces several major enhancements to the MoE framework that improve computational efficiency, load balancing, and inference accuracy. These enhancements build upon DeepSeek-V3’s optimizations, integrating reinforcement learning-based routing strategies, entropy-controlled gating, and fine-grained expert specialization. Below, we break down the key MoE innovations in DeepSeek-R1.

###### Adaptive Expert Routing with Reinforcement Learning (RL)

- DeepSeek-R1 introduces RL-based expert routing, moving away from static routing approaches used in DeepSeek-V3. Instead of selecting experts based purely on token-expert affinities computed via softmax functions, DeepSeek-R1 incorporates a learned RL policy to dynamically assign tokens to experts.
- **Mathematical Formulation:**
    
    - The expert selection function is formulated as an RL policy optimization problem, where the probability of selecting expert eiei for token tt is adjusted dynamically based on token embeddings utut:
    
    gi,t=πθ(ei|ut)gi,t=πθ(ei|ut)
    
    - where πθπθ is the policy network that selects experts based on contextual embeddings. The optimization objective follows GRPO:
    
    JGRPO(θ)=𝔼q∼P(Q),{oi}Gi=1∼πθold[1G∑i=1Gmin(πθ(oi|q)πθold(oi|q)Ai,clip(⋅))−βDKL(πθ||πref)]JGRPO(θ)=Eq∼P(Q),{oi}i=1G∼πθold[1G∑i=1Gmin(πθ(oi|q)πθold(oi|q)Ai,clip(⋅))−βDKL(πθ||πref)]
    
    - where DKLDKL regularizes the policy update to prevent drastic shifts.
- **Implementation Details:**
    - The RL-based router learns optimal token assignments by maximizing expert load balancing and minimizing routing entropy.
    - It penalizes overloading of specific experts while incentivizing uniform activation across layers.
    - Dynamic bias terms are introduced into the routing function to further modulate expert selection in response to training feedback.
- This approach enables adaptive token-expert mapping, optimizing inference speed while maintaining accuracy.

###### Hierarchical Entropy-Gated MoE (HE-MoE)

- DeepSeek-R1 enhances top-KK MoE routing by introducing Hierarchical Entropy-Gated MoE (HE-MoE). Instead of applying a single top-KK gating function at the token level, DeepSeek-R1 implements a multi-level gating mechanism:
    
    - **Global Selection:** Tokens are first routed to an initial pool of NgNg experts using softmax affinity scoring.
    - **Cluster-Level Pruning:** Within the selected pool, a secondary gating mechanism prunes experts based on entropy constraints.
    - **Final Expert Assignment:** Top-KrKr experts are chosen using an adjusted probability function that incorporates an entropy-aware penalty.
- The final gating function is modified as:
    
    gi,t=Softmaxi(uTtei)1+λH(g1:Nr,t)gi,t=Softmaxi(utTei)1+λH(g1:Nr,t)
    
    - where H(⋅)H(⋅) is the entropy function, and λλ controls the regularization strength.
- **Key Benefits:**
    
    - **Prevents expert over-specialization** by ensuring that tokens are distributed more evenly.
    - **Reduces mode collapse** where certain experts dominate training.
    - **Dynamically scales sparsity** by adjusting gating thresholds based on task complexity.

###### Device-Constrained Expert Allocation (DCEA)

- DeepSeek-R1 improves upon DeepSeek-V3’s node-limited routing by incorporating Device-Constrained Expert Allocation (DCEA), which restricts expert assignments based on GPU/TPU availability and interconnect bandwidth.
    
- **Algorithm:**
    
    - Each token first selects a subset of devices with the highest affinity scores.
    - Experts are restricted to these devices, reducing inter-device synchronization overhead.
    - The final experts are selected only within the constrained device pool, minimizing cross-node communication.
    
    M=∑i=1Nmax{sj,t|j∈device i}M=∑i=1Nmax{sj,t|j∈device i}
    
- **Results:**
    - 35% reduction in cross-device communication latency.
    - More stable training dynamics, as experts remain on localized compute nodes.
    - Lower bandwidth consumption, improving training efficiency.

###### Load-Balanced Expert Utilization with RL-Based Adjustments

- To ensure uniform load balancing, DeepSeek-R1 introduces adaptive load-based routing adjustments, replacing DeepSeek-V3’s auxiliary loss-based balancing strategy.
- Instead of explicitly minimizing an expert balance loss term, DeepSeek-R1 dynamically adjusts gating probabilities using an RL-based expert selection bias:

bi←bi−γif expert i is overloaded, otherwise bi←bi+γ.bi←bi−γif expert i is overloaded, otherwise bi←bi+γ.

- **Advantages Over Auxiliary Losses:**
    - Faster convergence, as it avoids additional gradient updates for balance constraints.
    - More robust expert selection, as it adapts over multiple training steps.
- This ensures consistent workload distribution without requiring hard auxiliary penalties.

###### Elimination of Token-Dropping Strategy

- Unlike DeepSeek-V3, which used token dropping to balance computation per device, DeepSeek-R1 completely eliminates token-dropping by optimizing expert activation thresholds dynamically.
- Instead of removing low-affinity tokens, DeepSeek-R1 reallocates tokens to alternative experts using a reinforcement-learning-based expert reassignment strategy.
- **Benefits:**
    - 100% token retention during training and inference.
    - Stronger generalization since all tokens contribute to learning.
    - No loss of contextual information, leading to more coherent completions.

##### Comparative Analysis

- DeepSeek-R1 represents the most advanced iteration of the MoE framework, building upon the optimizations introduced in DeepSeek-V2 and DeepSeek-V3. Below, we compare key MoE features across these three versions, highlighting improvements in efficiency, expert routing, load balancing, and inference performance.

|**Feature**|**DeepSeek-V2**|**DeepSeek-V3**|**DeepSeek-R1**|
|---|---|---|---|
|**Dynamic Expert Activation**|❌|✅ (Bias-based selection)|✅ (RL-based selection)|
|**Device-Limited Routing (DLR)**|✅|✅ (Node-Limited Routing)|✅ (Device-Constrained Expert Allocation)|
|**Auxiliary Loss for Load Balancing**|✅|❌ (Bias-based adjustments)|❌ (RL-based adaptive balancing)|
|**RL-Based Routing**|❌|❌|✅|
|**Hierarchical Gating for Expert Selection**|❌|✅|✅ (Entropy-aware adjustment)|
|**Improved Expert Selection Mechanism**|❌|✅ (Sigmoid-based)|✅ (RL-optimized selection)|
|**Cross-Device Communication Reduction**|✅ (Device-limited routing)|✅ (Node-limited routing)|✅ (35% lower latency with DCEA)|
|**Token Dropping for Computational Efficiency**|✅|❌ (No token dropping)|❌ (No token dropping)|
|**Sparse Activation Strategy**|✅ (Top-K gating)|✅ (Hierarchical Top-K gating)|✅ (Hierarchical Entropy-Gated MoE)|
|**Training Stability**|Moderate|High|Very High|
|**Inference Speed Optimization**|Moderate|High|Very High|
|**Load Balancing Strategy**|Loss-based balancing|Bias-based adaptive balancing|RL-based adaptive balancing|

#### Mathematical Formulation

- The expert selection process in DeepSeek-R1 follows a gating function:
    
    G(x)=softmax(Wgx)G(x)=softmax(Wgx)
    
    - where WgWg is a trainable weight matrix.
- The final output is computed as:
    
    y=∑k∈KGk(x)Ek(x)y=∑k∈KGk(x)Ek(x)
    
    - where:
        - KK represents the top-K selected experts.
        - Ek(x)Ek(x) is the computation performed by expert kk.
        - Gk(x)Gk(x) is the gating probability.

##### Load Balancing Loss

- To ensure equal utilization of experts, DeepSeek-R1 applies a load balancing loss:
    
    balance=λ∑k(nkN−1K)2Lbalance=λ∑k(nkN−1K)2
    
    - where:
        - nknk is the number of tokens assigned to expert kk.
        - NN is the total number of tokens in a batch.
        - KK is the number of active experts per token.
- Additionally, an entropy regularization term prevents expert over-reliance:
    
    entropy=−γ∑kGk(x)logGk(x)Lentropy=−γ∑kGk(x)log⁡Gk(x)
    
    - where γγ controls entropy strength.

### Multihead Latent Attention (MLA)

#### Overview

- Multihead Latent Attention (MLA) enhances efficiency by projecting Key-Query-Value (KQV) matrices into a lower-dimensional latent space, significantly reducing computational and memory costs.
- Low-rank compression techniques in MLA minimize the storage overhead of the Key-Value (KV) cache, ensuring faster inference and supporting longer context lengths or larger batch sizes.
- DeepSeek-R1 refines MLA further by incorporating RL-enhanced reasoning optimizations while maintaining low memory overhead.
- By utilizing decoupled rotary positional embeddings and latent-space compression, MLA ensures minimal accuracy degradation while maintaining computational efficiency.

#### Key Features

- **Low-Rank Key-Value Compression**: MLA employs a low-rank latent space projection to compress KV pairs, significantly reducing memory overhead. This allows DeepSeek-R1 to store only compressed representations instead of full-dimensional KV states, enabling efficient long-context processing.
    
- **Decoupled Rotary Position Embedding (RoPE)**: Standard RoPE introduces position-dependent transformations that hinder KV compression. DeepSeek-R1 decouples RoPE from key-value storage, ensuring positional encodings remain effective without interfering with latent-space efficiency.
    
- **Efficient Multihead Attention with Compressed Storage**: Instead of caching full key-value matrices for all tokens, MLA only stores their compact latent-space equivalents. This drastically reduces inference memory requirements while maintaining attention fidelity.
    
- **Adaptive Projection Matrices**: MLA leverages separate, learned projection matrices for queries, keys, and values. These matrices dynamically adjust during training, ensuring optimal storage efficiency and minimal accuracy loss compared to full-dimensional attention.
    
- **Inference-Efficient Cache Mechanism**: By selectively caching only compressed key-value representations, MLA achieves a 93.3% KV cache reduction over traditional Multi-Head Attention (MHA). This allows DeepSeek-R1 to support longer context lengths while minimizing inference latency.
    
- **Enhanced Performance on Long-Context Tasks**: DeepSeek-R1 refines MLA with RL-driven optimizations, such as GRPO, to prioritize critical tokens. This improves reasoning accuracy in long-context tasks while preserving computational efficiency.
    

#### Evolution from DeepSeek-V2 to DeepSeek-R1

##### MLA in DeepSeek-V2

- MLA in DeepSeek-V2 was designed to enhance inference efficiency by significantly reducing the KV cache size while maintaining strong model performance. It introduced several key innovations over traditional Multi-Head Attention (MHA), including low-rank key-value joint compression and decoupled rotary position embedding.
- The MLA implementation in DeepSeek-V2 laid the foundation for further improvements in DeepSeek-R1, where it was further refined with FP8 quantization, enhanced compression techniques, and improved numerical stability.

###### Low-Rank Key-Value Joint Compression

- One of the primary bottlenecks in transformer inference is the large KV cache required to store past keys and values. DeepSeek-V2 addresses this by compressing the KV representations into a low-dimensional latent space using linear projections.
    
- Given an input token representation ht∈ℝdht∈Rd, standard multi-head attention computes queries, keys, and values as:
    
    qt=WQht,kt=WKht,vt=WVhtqt=WQht,kt=WKht,vt=WVht
    
    where WQ,WK,WV∈ℝdhnh×dWQ,WK,WV∈Rdhnh×d.
    
- Instead of storing full-dimension ktkt and vtvt, MLA compresses them into a latent representation cKVcKV:
    
    cKVt=WDKVhtcKVt=WDKVht
    
    where WDKV∈ℝdc×dWDKV∈Rdc×d is a down-projection matrix, and dc≪dhnhdc≪dhnh.
    
- During inference, the compressed key-value representation is expanded back into usable keys and values:
    
    kCt=WUKcKVt,vCt=WUVcKVtktC=WUKcKVt,vtC=WUVcKVt
    
    where WUK,WUV∈ℝdhnh×dcWUK,WUV∈Rdhnh×dc are up-projection matrices.
    
    This compression reduces the KV cache size from O(nhdhl)O(nhdhl) to O(dcl)O(dcl), where ll is the number of layers.
    

###### Decoupled Rotary Position Embedding

- RoPE is commonly used in transformer architectures to encode positional information into queries and keys. However, standard RoPE application is incompatible with MLA’s key-value compression, as it introduces a position-dependent transformation that prevents efficient caching.
    
- DeepSeek-V2 resolves this by decoupling RoPE from key compression:
    
    1. Introduce an auxiliary shared key kRtktR and additional multi-head queries qRtqtR.
    2. Apply RoPE only to qRtqtR and kRtktR:
        
        qRt=RoPE(WQRcQt),kRt=RoPE(WKRht)qtR=RoPE(WQRcQt),ktR=RoPE(WKRht)
        
        - where WQR,WKRWQR,WKR are projection matrices specific to decoupled RoPE.
    3. Concatenate compressed and RoPE-applied keys/queries:
        
        qt=[qCt;qRt],kt=[kCt;kRt]qt=[qtC;qtR],kt=[ktC;ktR]
        
        - ensuring that RoPE affects only a subset of the attention mechanism while keeping key-value compression intact.

###### Comparison of KV Cache Requirements

- A key benefit of MLA is that it achieves stronger performance than standard MHA while requiring significantly less KV cache. The table below compares the cache sizes across different attention mechanisms:

|**Attention Mechanism**|**KV Cache per Token (Elements)**|
|---|---|
|MHA|2nhdhl2nhdhl|
|GQA (Grouped Query)|2ngdhl2ngdhl|
|MQA (Multi-Query)|2dhl2dhl|
|**MLA (DeepSeek-V2)**|(dc+dRh)l(dc+dhR)l|

- For DeepSeek-V2, values were set as: dc=4dhdc=4dh dRh=dh/2dhR=dh/2
    
- This means that MLA achieves similar efficiency to GQA with 2.25 groups, while maintaining the performance level of MHA.
    

##### Enhancements in DeepSeek-V3

- DeepSeek-V3 introduces several key enhancements to Multihead Latent Attention (MLA) that significantly improve its efficiency, scalability, and precision while maintaining high model accuracy. The major improvements include:
    
    - Further KV Cache Reduction through Optimized Compression Techniques
    - Query Compression for Activation Memory Savings
    - Enhanced Numerical Stability with FP8 Mixed Precision
    - Adaptive Routing for Load Balancing in MLA
- With these improvements, DeepSeek-V3 reduces memory overhead, enhances numerical precision, and achieves significantly faster inference speeds while maintaining high model accuracy.
    

###### Further KV Cache Reduction Through Optimized Compression Techniques

- One of the major enhancements in DeepSeek-V3’s MLA is the more aggressive compression of the KV cache while preserving model performance. This is achieved through:
    
    - **Dynamic KV Compression Matrices**: Instead of static compression matrices, DeepSeek-V3 optimizes the compression dynamically per sequence length.
    - **Factorized Projections for KV Storage**: A dual-matrix decomposition is applied to down-project the keys and values, further reducing KV storage.

###### Optimized Compression Formulation

- Given an input token representation ht∈ℝdht∈Rd, standard MLA in DeepSeek-V2 computed compressed KV representations as:
    
    cKVt=WDKVhtcKVt=WDKVht
    
    - where WDKV∈ℝdc×dWDKV∈Rdc×d was a static down-projection matrix.
- In DeepSeek-V3, the compression process is enhanced with an adaptive dual-matrix compression:
    
    cKVt=WDKV,1WDKV,2htcKVt=WDKV,1WDKV,2ht
    
    - where WDKV,1∈ℝdm×dWDKV,1∈Rdm×d and WDKV,2∈ℝdc×dmWDKV,2∈Rdc×dm, with dmdm being an intermediate dimensionality. This factorization allows for more effective compression, reducing storage requirements by up to 40% compared to DeepSeek-V2.

###### Inference-Time Expansion

- During inference, the expanded keys and values are now computed as:
    
    kCt=WUKWMKcKVt,vCt=WUVWMVcKVtktC=WUKWMKcKVt,vtC=WUVWMVcKVt
    
    - where WMK,WMVWMK,WMV serve as intermediary projection layers that refine the KV reconstruction process.
- This improvement ensures that only compressed vectors are stored in memory, significantly reducing KV cache overhead.
    

###### Query Compression for Activation Memory Savings

- DeepSeek-V3 extends MLA’s low-rank compression to queries, reducing activation memory requirements without affecting attention precision.
    
- **Query Compression Formulation**:
    
    - Instead of computing full queries:
    
    qt=WQht,kt=WKht,vt=WVhtqt=WQht,kt=WKht,vt=WVht
    
    - DeepSeek-V3 introduces an additional compression step:
        
        cQt=WDQht,qCt=WUQcQtcQt=WDQht,qtC=WUQcQt
        
        - where:
            - cQt∈ℝd′ccQt∈Rdc′ is the compressed query representation.
            - d′c≪dhnhdc′≪dhnh, ensuring significantly lower activation memory usage.
- **Decoupled Rotary Positional Embedding (RoPE)**:
    
    - To maintain the effectiveness of positional embeddings, DeepSeek-V3 decouples Rotary Positional Embedding (RoPE) application:
        
        qRt=RoPE(WQRcQt),kRt=RoPE(WKRht)qtR=RoPE(WQRcQt),ktR=RoPE(WKRht)
        
        - where:
            - qRtqtR and kRtktR store RoPE-applied versions of the compressed representations.
            - This prevents RoPE from interfering with MLA’s low-rank compression.

###### Reduction in Activation Memory

- With query compression, DeepSeek-V3 reduces attention activation memory by 35%, enabling efficient training on large-scale models.

###### Enhanced Numerical Stability with FP8 Mixed Precision

- DeepSeek-V3 leverages FP8 mixed precision training, improving numerical stability while reducing memory and computational costs.
    
- **FP8 Training for MLA Components**:
    
    - In DeepSeek-V2, the MLA components operated primarily in BF16. DeepSeek-V3 instead adopts fine-grained FP8 quantization, applying a per-group scaling strategy:
        
        - **Activation Scaling:** Per-token, per-128-channel tile quantization for activations.
        - **Weight Scaling:** 128×128 block-wise scaling for weights.
    - This ensures reduced rounding errors and better dynamic range coverage for training.
        
- **FP8 Attention Computation**:
    
    - The attention output in DeepSeek-V3 is computed using FP8-compatible scaling:
        
        ot=∑j=1tSoftmax(qTtkjdh+dR‾‾‾‾‾‾‾√)vjot=∑j=1tSoftmax(qtTkjdh+dR)vj
        
        - where:
            - The scaling factor is calculated online for activations.
            - The accumulation is upgraded to FP32 every 128 steps to improve numerical precision.
- **Precision Comparison**:
    

|**Component**|**DeepSeek-V2 (BF16)**|**DeepSeek-V3 (FP8)**|
|---|---|---|
|Query/Key Compression|dc=4dhdc=4dh|dc=3dhdc=3dh|
|KV Cache Storage|BF16|FP8|
|RoPE Application|Full Precision|Decoupled, FP8|
|Attention Computation|BF16|FP8 + FP32 Accumulation|

- By leveraging FP8 quantization, DeepSeek-V3 achieves 2.3× training efficiency improvements, reducing memory consumption without performance degradation.

###### Adaptive Routing for Load Balancing in MLA

- DeepSeek-V3 improves attention efficiency by introducing dynamic load balancing for query-key computation.
    
- **Load-Adaptive Routing Mechanism**:
    
    - In DeepSeek-V2, MLA used static attention head assignments, leading to occasional computational inefficiencies when processing large sequences.
        
    - DeepSeek-V3 refines this with adaptive routing:
        
        si,t=Sigmoid(uTtei+bi)si,t=Sigmoid(utTei+bi)
        
        - where:
            - eiei is the centroid vector of the routed expert.
            - bibi is a dynamically updated bias term that adjusts for per-head workload balance.
    - The bias term updates as:
        
        b(t+1)i=b(t)i−γ⋅(overloadedi−underloadedi)bi(t+1)=bi(t)−γ⋅(overloadedi−underloadedi)
        
        - where γγ is a tuning parameter.
    - This ensures:
        
        - Balanced token distribution across attention heads.
        - No token-dropping during inference, preventing efficiency loss.
- **Computational Gains**:
    
    - By integrating adaptive routing, DeepSeek-V3 achieves:
        - Uniform computational load across attention heads.
        - 10% reduction in per-token inference latency.

##### Enhancements in DeepSeek-R1

- DeepSeek-R1 introduces several refinements to MLA, improving reasoning efficiency and inference performance while maintaining low memory overhead. Building upon the MLA optimizations in DeepSeek-V3, DeepSeek-R1 further enhances KQV compression, RL-guided attention allocation, and numerical stability mechanisms.

###### RL-Guided Latent Attention Optimization

- DeepSeek-R1 integrates RL techniques into MLA, optimizing attention mechanisms through GRPO. Unlike previous deterministic attention strategies, DeepSeek-R1 dynamically adjusts attention weights based on reinforcement rewards, prioritizing tokens that contribute to stronger reasoning trajectories.
- GRPO eliminates the need for a separate critic model, reducing memory overhead and improving convergence efficiency.
- Instead of relying on supervised fine-tuning, GRPO estimates advantage values directly from group-level rewards:

Ai=ri−mean({r1,r2,...,rG})std({r1,r2,...,rG})Ai=ri−mean({r1,r2,...,rG})std({r1,r2,...,rG})

- The policy model πθπθ is updated by maximizing:

JGRPO(θ)=𝔼[∑i=1Gmin(πθ(oi|q)πθold(oi|q)Ai,clip(πθ(oi|q)πθold(oi|q),1−ϵ,1+ϵ)Ai)−βDKL(πθ||πref)]JGRPO(θ)=E[∑i=1Gmin(πθ(oi|q)πθold(oi|q)Ai,clip(πθ(oi|q)πθold(oi|q),1−ϵ,1+ϵ)Ai)−βDKL(πθ||πref)]

- This approach allows DeepSeek-R1 to adaptively refine the attention mechanisms in MLA, improving token prioritization in long-context reasoning.
- Further details can be found in the section on [RL Algorithm: Group Relative Policy Optimization (GRPO)](https://aman.ai/primers/ai/deepseek-R1/#rl-algorithm-group-relative-policy-optimization-grpo).

###### Adaptive Query and Key Compression Via RL

One of the primary enhancements in DeepSeek-R1’s MLA is RL-guided adaptive query and key compression. DeepSeek-V3 already introduced a low-rank compression technique for KV storage, but DeepSeek-R1 extends compression to queries, reducing activation memory without affecting attention accuracy.

- **Optimized Compression Formulation**:
    
    - In DeepSeek-V3, the KV cache compression was achieved using static low-rank projections:
    
    cKVt=WDKVhtcKVt=WDKVht
    
    - DeepSeek-R1 dynamically adjusts compression matrices during inference using RL-based reward maximization:
        
        cKVt=WDKV,1WDKV,2htcKVt=WDKV,1WDKV,2ht
        
        - where:
            - WDKV,1∈ℝdm×dWDKV,1∈Rdm×d and WDKV,2∈ℝdc×dmWDKV,2∈Rdc×dm.
            - dmdm is an intermediate dimensionality, allowing for more fine-grained latent space representations.
- **Inference-Time Expansion**:
    
    - Instead of using a single up-projection matrix, DeepSeek-R1 incorporates a multi-stage expansion pipeline:
        
        kCt=WUKWMKcKVt,vCt=WUVWMVcKVtktC=WUKWMKcKVt,vtC=WUVWMVcKVt
        
        - where WMK,WMVWMK,WMV refine the reconstructed query-key values, ensuring that only compressed vectors are stored in memory.
- **Compression ratio improvements:** DeepSeek-R1 reduces KV cache requirements by an additional 25% over DeepSeek-V3, while maintaining query-key retrieval accuracy.
    

###### Decoupled Rotary Position Embedding with Context-Specific Scaling

- While DeepSeek-V3 introduced Decoupled RoPE to separate positional encoding from compressed key-value representations, DeepSeek-R1 further refines RoPE with context-specific scaling mechanisms.
- DeepSeek-R1 adopts an enhanced RoPE formulation where RoPE is context-aware, dynamically adjusting scaling factors based on sequence length:
    
    λt=11+αLt‾‾‾‾‾‾‾√λt=11+αLt
    
    - where:
        - λtλt is the adaptive scaling factor for positional embedding.
        - αα is a hyperparameter learned via RL optimization.
        - LtLt represents the sequence length at time step tt.
- **Implementation benefits**:
    - RoPE scaling ensures consistent attention alignment across varying sequence lengths.
    - Prevents positional information degradation when compressing MLA’s key-value states.

###### FP8 Mixed Precision for MLA Stability

- DeepSeek-R1 adopts FP8 quantization for MLA computations, further improving numerical stability over DeepSeek-V3’s BF16-based approach.
- In DeepSeek-R1’s precision-aware computation pipeline, QKV matrices are quantized dynamically using per-group scaling:
    
    Q̃ =QsQ,K̃ =KsK,Ṽ =VsVQ~=QsQ,K~=KsK,V~=VsV
    
    - where sQ,sK,sVsQ,sK,sV are learned per-group scaling factors.
- The attention output is computed with hybrid precision accumulation:
    
    ot=∑j=1tSoftmax(q̃ Ttk̃ jdh+dR‾‾‾‾‾‾‾√)ṽ jot=∑j=1tSoftmax(q~tTk~jdh+dR)v~j
    
- The accumulation process is upgraded to FP32 every 128 steps, ensuring better numerical precision while maintaining FP8 efficiency.
    
- **Comparison of MLA Precision Strategies**:

|**Component**|**DeepSeek-V3 (BF16)**|**DeepSeek-R1 (FP8)**|
|---|---|---|
|Query/Key Compression|dc=4dhdc=4dh|dc=3dhdc=3dh|
|KV Cache Storage|BF16|FP8|
|RoPE Application|Full Precision|Decoupled, FP8|
|Attention Computation|BF16|FP8 + FP32 Accumulation|

- **Efficiency improvements**:
    - FP8 reduces memory footprint by ~40% compared to BF16.
    - Enables 2.3× faster inference throughput for long-context tasks.

###### Adaptive/Dynamic Routing for Load-Balanced Attention

- DeepSeek-R1 incorporates load-balancing adaptive routing mechanisms, ensuring uniform query-key computation across attention heads.
- DeepSeek-R1 optimizes per-head workload balance using a sigmoid-based routing function:
    
    si,t=Sigmoid(uTtei+bi)si,t=Sigmoid(utTei+bi)
    
    - where:
        - eiei represents the centroid vector of the routed attention expert.
        - bibi is an adaptive bias term, ensuring workload uniformity.
- **Performance gains**:
    - Balanced computation across heads prevents bottlenecks.
    - Reduces per-token inference latency by 10%.

##### Comparative Analysis

- DeepSeek-V2 introduced Multihead Latent Attention (MLA) with significant KV cache compression, decoupled RoPE, and basic low-rank projections for efficiency. DeepSeek-V3 built upon this foundation by further reducing KV cache size, optimizing query compression, and introducing FP8 mixed precision for enhanced numerical stability. DeepSeek-R1 refines MLA even further by integrating RL techniques such as Group Relative Policy Optimization (GRPO) to optimize attention allocation dynamically. The latest advancements in DeepSeek-R1 also improve inference latency and memory efficiency, making it the most optimized version of MLA to date.
- The table below provides a comparative analysis of DeepSeek-V2, DeepSeek-V3, and DeepSeek-R1 for MLA. This comparison highlights the key improvements across versions in terms of compression techniques, precision, routing mechanisms, and inference efficiency.

|**Feature**|**DeepSeek-V2**|**DeepSeek-V3**|**DeepSeek-R1**|
|---|---|---|---|
|**Low-Rank KV Compression**|✅|✅ (Optimized with Factorized Projections)|✅ (RL-Optimized Adaptive Compression)|
|**Query Compression**|❌|✅ (Static Low-Rank Query Compression)|✅ (RL-Guided Dynamic Query Compression)|
|**KV Cache Reduction**|✅ (93.3% Reduction)|✅ (40% Further Reduction)|✅ (25% Further Reduction over V3)|
|**RoPE Application**|✅ (Decoupled RoPE)|✅ (Decoupled with Context-Specific Scaling)|✅ (Enhanced Context-Aware Scaling)|
|**Precision Format**|BF16|FP8 (Fine-Grained Mixed Precision)|FP8 (Per-Group Scaling, FP32 Accumulation)|
|**Adaptive Routing for MLA**|❌|✅ (Static Adaptive Routing)|✅ (Load-Balanced Dynamic Routing)|
|**Inference Latency Reduction**|✅ (KV Compression Reduces Latency)|✅ (10% Faster than V2)|✅ (10% Faster than V3)|
|**RL Enhancements**|❌|❌|✅ (GRPO for Adaptive MLA Optimization)|
|**Numerical Stability Improvements**|✅ (Basic Stability Enhancements)|✅ (FP8 with Mixed Precision)|✅ (FP8 with RL-Guided Stability Mechanisms)|
|**Long-Context Performance**|✅ (Supports Longer Contexts)|✅ (Further Optimized)|✅ (Enhanced with RL-Guided Token Prioritization)|

#### Implementation

- The implementation of MLA in DeepSeek-R1 incorporates several optimizations aimed at maximizing efficiency while preserving accuracy. This section details the core mechanisms underlying MLA, including key-value compression, query transformation, position encoding, and computational optimizations.

##### Background: Standard Multi-Head Attention (MHA)

- For a standard multi-head attention (MHA) mechanism, the Key (KK), Query (QQ), and Value (VV) matrices are computed as follows:
    
    K,Q,V=WkX,WqX,WvXK,Q,V=WkX,WqX,WvX
    
    - where Wk,Wq,WvWk,Wq,Wv are weight matrices for key, query, and value projections.
- The attention weights are computed as:
    
    A=Softmax(QKTdk‾‾√)A=Softmax(QKTdk)
    
    - and the output is given by:
    
    O=AVO=AV
    
- This requires storing the full key-value cache during inference, leading to significant memory overhead.
    

##### Low-Rank Key-Value Joint Compression

- One of the fundamental optimizations in MLA is the compression of KV pairs into a lower-dimensional latent space, significantly reducing memory overhead. Specifics below:
    
    - **Compression Mechanism**:
        
        - The key and value representations are compressed into a shared latent space before being projected back into their respective dimensions. This is achieved through a two-step transformation:
        
        cKVt=WDKVhtcKVt=WDKVht
        
        kCt=WUKcKVt,vCt=WUVcKVtkCt=WUKcKVt,vCt=WUVcKVt
        
        - where:
            - cKVt∈ℝdccKVt∈Rdc is the compressed latent representation.
            - WDKV∈ℝdc×dWDKV∈Rdc×d is a down-projection matrix.
            - WUK,WUV∈ℝdhnh×dcWUK,WUV∈Rdhnh×dc are up-projection matrices for keys and values, respectively.
    - **Memory Reduction**:
        - Instead of storing full-sized keys and values for each token, only cKVtcKVt is cached.
        - The reduction in memory footprint allows DeepSeek-R1 to process significantly longer sequences at a lower computational cost.

##### Multi-Stage Compression

- DeepSeek-R1 refines the compression mechanism by introducing an additional transformation layer, leading to a multi-stage compression approach. Specifics below:
    
    - **Additional Projection Layer**:
        
        - To further minimize storage costs, a secondary compression layer is introduced:
        
        c′KVt=WDKV2f(WDKVht)cKVt′=WDKV2f(WDKVht)
        
        - where:
            - WDKV2∈ℝd′c×dcWDKV2∈Rdc′×dc is a second down-projection matrix.
            - f(⋅)f(⋅) is a non-linear activation function applied to improve representation learning.
            - d′c<dcdc′<dc ensures an even smaller KV cache size.
    - **Performance Benefits**:
        - This additional step further reduces KV storage while maintaining sufficient information for attention mechanisms.
        - Experiments indicate that this leads to a 10-15% reduction in memory footprint compared to DeepSeek-V3.

##### Query Compression and Optimization

- Similar to keys and values, queries are also compressed, allowing for efficient computation and reduced activation memory during training. Specifics below:
    
    - **Query Transformation**:
        
        - Queries undergo a two-step transformation similar to keys and values:
        
        cQt=WDQhtcQt=WDQht
        
        qCt=WUQcQtqCt=WUQcQt
        
        - where:
            - WDQ∈ℝd′c×dWDQ∈Rdc′×d is a down-projection matrix for queries.
            - WUQ∈ℝdhnh×d′cWUQ∈Rdhnh×dc′ maps the compressed query representation back to its original dimensionality.
    - **Multi-Layer Query Refinement**:
        - DeepSeek-R1 optimizes query projection through additional adaptive scaling layers.
        - The transformation matrices WDQWDQ and WUQWUQ are dynamically adjusted during fine-tuning using RL.

##### Decoupled Rotary Position Embedding (RoPE)

- To ensure robust long-context handling, DeepSeek-R1 applies RoPE in a decoupled manner, separating positional encodings from the latent attention mechanism. Specifics below:
    
    - **Independent Positional Encoding for Keys and Queries**:
        
        kRt=RoPE(WKRht)kRt=RoPE(WKRht)
        
        qRt=RoPE(WQRcQt)qRt=RoPE(WQRcQt)
        
        - where:
            - WKR∈ℝdRh×dWKR∈RdRh×d generates positional embeddings for keys.
            - WQR∈ℝdRhnh×d′cWQR∈RdRhnh×dc′ generates positional embeddings for queries.
            - The RoPE transformation ensures that relative positional information is preserved while allowing the KV cache to remain compact.
    - **Computation Efficiency of RoPE in DeepSeek-R1**:
        
        - RoPE application is delayed until the final stages of query-key interaction, preventing unnecessary memory bloat.
        - Compared to DeepSeek-V2 and V3, DeepSeek-R1 achieves 25% faster query-key retrieval.

##### Attention Computation in MLA

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

##### RL-Optimized MLA

- DeepSeek-R1 incorporates RL to further optimize MLA’s transformation matrices. Specifics below:
    
    - **Fine-Tuning with RL**:
        - Using GRPO, MLA is rewarded based on efficient memory usage and retrieval accuracy.
        - The policy update equation is:
            
            JGRPO(θ)=E[∑i=1Gmin(πθ(oi|q)πθold(oi|q)Ai,clip(πθ(oi|q)πθold(oi|q),1−ϵ,1+ϵ)Ai)]JGRPO(θ)=E[∑i=1Gmin(πθ(oi|q)πθold(oi|q)Ai,clip(πθ(oi|q)πθold(oi|q),1−ϵ,1+ϵ)Ai)]
            
            - where:
                - πθπθ represents the updated policy.
                - AiAi is the advantage function guiding optimization.
    - Further details can be found in the section on [RL Algorithm: Group Relative Policy Optimization (GRPO)](https://aman.ai/primers/ai/deepseek-R1/#rl-algorithm-group-relative-policy-optimization-grpo).

##### Computational and Hardware Optimization

- **Inference-Time Efficiency**:
    - MLA in DeepSeek-R1 is implemented with tensor-parallelized computations, optimizing throughput across GPUs.
    - Memory overhead is minimized through low-precision KV storage (FP8 format).
- **Cross-Node Communication Optimization**:
    - Uses optimized all-to-all communication kernels to fully utilize InfiniBand (IB) and NVLink bandwidths.
    - Reduces inter-node communication latency by 30%, improving distributed inference performance.

##### Comparative Efficiency Analysis

|**Attention Mechanism**|**KV Cache Per Token**|**Computational Complexity**|**Performance Impact**|
|---|---|---|---|
|**MHA (Standard)**|O(Ndh)O(Ndh)|O(N2dh)O(N2dh)|High Accuracy, High Cost|
|**MQA**|O(dh)O(dh)|O(Ndh)O(Ndh)|Lower Memory, Degraded Performance|
|**GQA**|O(gdh)O(gdh) (groups)|O(Ndh)O(Ndh)|Moderate Balance|
|**MLA (DeepSeek-V2)**|O(dL)O(dL)|O(NdL)O(NdL)|High Efficiency, Minimal Loss|
|**MLA + Hierarchical Caching (DeepSeek-R1)**|O(dL)O(dL) (with reuse)|O(NdL)O(NdL)|**Peak Efficiency, Retains Performance**|

### Multi-Token Prediction (MTP)

#### Overview

- Multi-Token Prediction (MTP) allows DeepSeek-R1 to predict multiple tokens in parallel, significantly improving inference speed.
- MTP was introduced in DeepSeek-V3 as a training objective to improve data efficiency and predictive capabilities by enabling the model to anticipate multiple future tokens at each position. DeepSeek-R1 adopts the same implementation as in DeepSeek-V3.
- Unlike conventional next-token prediction, which limits training to a single-step forward prediction, MTP extends this scope to multiple future tokens, thereby densifying training signals and enhancing long-term coherence in text generation.

#### Key Features

- **Parallel Multi-Token Prediction**: DeepSeek-R1 enhances inference speed by predicting multiple tokens simultaneously rather than sequentially. This reduces decoding latency and allows for faster text generation without compromising coherence.
    
- **Cross-Depth Residual Connections**: Unlike DeepSeek-V3, which conditions token predictions only on prior module outputs, DeepSeek-R1 integrates residual connections between MTP layers. This allows deeper MTP modules to utilize features from earlier depths, improving long-term dependencies.
    
- **Adaptive Prediction Granularity**: The model dynamically adjusts how many future tokens each module predicts based on the input sequence’s complexity. This ensures fine-grained predictions for short contexts and broader lookahead when handling longer sequences.
    
- **Depth-Aware Loss Weighting**: DeepSeek-R1 refines its training objective by prioritizing mid-range MTP depths using a sigmoid-based weighting function. This enhances learning efficiency by directing more gradient updates where they have the greatest impact.
    
- **Memory-Efficient Parameter Sharing**: The model reduces memory consumption by reusing transformer layers across MTP depths. Instead of separate layers for each module, DeepSeek-R1 applies depth-conditioned routing, minimizing redundant computations while maintaining unique depth-wise representations.
    
- **Optimized Speculative Decoding**: DeepSeek-R1 improves speculative decoding by introducing probabilistic agreement checking. Predictions are accepted based on confidence thresholds rather than requiring exact matches, reducing rejection rates and accelerating inference.
    
- **Empirical Gains in Training and Inference**: Thanks to these enhancements, DeepSeek-R1 achieves a **22% faster training convergence**, **1.5× improvement in generation speed**, and **18% better long-form perplexity**, demonstrating its superiority over DeepSeek-V3.
    

##### MTP in DeepSeek-V3/DeepSeek-R1

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

### GRPO Variants

#### Decoupled Clip and Dynamic Sampling Policy Optimization (DAPO)

- [Decoupled Clip and Dynamic Sampling Policy Optimization (DAPO)](https://arxiv.org/abs/2503.14476), proposed by ByteDance, Tsinghua University, The University of Hong Kong, and the SIA-Lab of Tsinghua AIR and ByteDance Seed, significantly advances reinforcement learning (RL) for long-chain-of-thought (CoT) reasoning by building on the foundational ideas of Grouped Reinforcement Learning with Policy Optimization (GRPO). While GRPO simplifies RL training by eliminating the need for a value network through group-based reward normalization, DAPO introduces a suite of principled enhancements that greatly improve sample efficiency, training stability, and policy diversity—particularly in complex reasoning tasks.
- Developed and benchmarked with the Qwen2.5-32B model, DAPO not only matches but often surpasses state-of-the-art results achieved by models like DeepSeek-R1-Zero. This demonstrates the limitations of naive GRPO and highlights DAPO as a more refined and robust alternative. Key innovations in DAPO include mechanisms such as Clip-Higher for better gradient clipping, dynamic sampling for improved data efficiency, token-level loss modeling for finer-grained learning, and nuanced reward shaping. These features allow DAPO to achieve superior performance with fewer training steps, making it a compelling evolution in the RL paradigm for large-scale reasoning models. Importantly, DAPO remains reproducible and open-source, promoting transparency and further research in the field.

##### DAPO vs. GRPO: Key Conceptual Differences

- At a high level, both GRPO and DAPO eliminate the critic model and use group-normalized advantages. However, DAPO introduces several critical refinements:
    
    1. **Clipping asymmetry** (Clip-Higher): Decouples the lower and upper clipping bounds to promote exploration.
    2. **Dynamic Sampling**: Filters out trivial cases where all generated responses are either fully correct or incorrect, preserving informative gradient updates.
    3. **Token-Level Loss**: Applies the policy gradient loss at the token level rather than averaging over the sequence.
    4. **Overlong Reward Shaping**: Mitigates instability from truncated long sequences via soft penalties instead of hard cutoffs.
- Each of these refinements addresses a specific deficiency in GRPO’s design when applied to long-CoT reasoning tasks.
    

#### DAPO: Implementation Details

##### Objective Function

- DAPO modifies the GRPO objective by decoupling the clipping thresholds and introducing dynamic sampling. The core training objective is:
    
    JDAPO(θ)=𝔼(q,a)∼D,{oi}Gi=1∼πθold(⋅|q)[1∑Gi=1|oi|∑i=1G∑t=1|oi|min(ri,t(θ)Â i,t,clip(ri,t(θ),1−ϵlow,1+ϵhigh)Â i,t)]JDAPO(θ)=E(q,a)∼D,{oi}i=1G∼πθold(⋅|q)[1∑i=1G|oi|∑i=1G∑t=1|oi|min(ri,t(θ)A^i,t,clip(ri,t(θ),1−ϵlow,1+ϵhigh)A^i,t)]
    
    - where:
        - ri,t(θ)=πθ(oi,t|q,oi,<t)πθold(oi,t|q,oi,<t)ri,t(θ)=πθ(oi,t|q,oi,<t)πθold(oi,t|q,oi,<t)
            
        - Â i,t=Ri−mean({Rj}Gj=1)std({Rj}Gj=1)A^i,t=Ri−mean({Rj}j=1G)std({Rj}j=1G)
            
        - ϵlow,ϵhighϵlow,ϵhigh are asymmetric clipping bounds
        - πθπθ is the current policy, and πθoldπθold is the behavior policy used to generate samples
        - Rewards RiRi are computed via rule-based evaluators

##### Clip-Higher (Asymmetric Clipping)

- While GRPO adopts PPO-style clipping:

clip(r,1−ϵ,1+ϵ)clip(r,1−ϵ,1+ϵ)

- DAPO decouples the upper and lower clipping thresholds:

clip(r,1−ϵlow,1+ϵhigh)clip(r,1−ϵlow,1+ϵhigh)

- This enables greater policy exploration, especially for low-probability tokens that would otherwise remain underrepresented. This adjustment empirically increases generation entropy without sacrificing correctness, stabilizing learning and avoiding entropy collapse (Figure 2b in the source).

##### Dynamic Sampling

- DAPO introduces a data-efficient sampling strategy. Since GRPO computes relative advantages within a group, if all generated responses are correct (or all incorrect), the standard deviation becomes zero, yielding zero gradients.
    
- To counteract this, DAPO enforces:
    

0<∣∣{oi∣is\_equivalent(oi,a)}∣∣<G0<|{oi∣is\_equivalent(oi,a)}|<G

- This filters out groups that do not produce informative learning signals. The strategy ensures that every training batch contains prompts yielding useful gradients by resampling until this constraint is met.

##### Token-Level Policy Gradient Loss

- GRPO aggregates the loss by averaging across entire sequences, which leads to disproportionate weight on shorter samples. DAPO instead normalizes the loss over **tokens**, ensuring that longer responses contribute proportionally to the gradient:

1∑Gi=1|oi|∑i=1G∑t=1|oi|1∑i=1G|oi|∑i=1G∑t=1|oi|

- This fine-grained gradient computation:
    - Enhances learning from long, structured reasoning sequences
    - Mitigates the problem of low-quality long outputs dominating updates
    - Encourages refinement of reasoning at the token level rather than relying solely on outcome correctness

##### Overlong Reward Shaping

- To handle excessively long outputs (common in long-CoT tasks), DAPO introduces **Soft Overlong Punishment**. Rather than assigning a fixed penalty, it uses a length-sensitive shaping function:

Rlength(y)=⎧⎩⎨⎪⎪0(Lmax−Lcache)−|y|Lcache−1if |y|≤Lmax−Lcacheif Lmax−Lcache<|y|≤Lmaxif |y|>LmaxRlength(y)={0if |y|≤Lmax−Lcache(Lmax−Lcache)−|y|Lcacheif Lmax−Lcache<|y|≤Lmax−1if |y|>Lmax

- This length-aware penalty encourages succinct, precise reasoning and reduces training noise from truncated sequences, as demonstrated by the entropy and accuracy trends in the source.

##### Training Implementation Summary

- **Base Model**: Qwen2.5-32B
- **Batch Size**: 512 prompts × 16 responses per prompt
- **Loss Reduction**: Token-level
- **Clipping**: Asymmetric with ϵlow=0.2ϵlow=0.2, ϵhigh=0.28ϵhigh=0.28
- **Max Tokens**: 20,480 (with soft penalty starting at 16,384)
- **Reward Function**: Rule-based equivalence check (1 for correct, -1 otherwise)

#### DAPO vs. GRPO Summary

|**Feature**|**GRPO**|**DAPO**|
|---|---|---|
|Clipping|Symmetric (ϵϵ)|Asymmetric (ϵlow,ϵhighϵlow,ϵhigh)|
|Sampling|Uniform (accept all groups)|Dynamic (filters degenerate groups)|
|Loss Aggregation|Sample-level|Token-level|
|Overlong Sample Handling|Truncation + hard penalty|Soft reward shaping|
|KL Regularization|Optional (often used)|Removed to allow divergence from SFT|
|Exploration Capacity|Lower (risk of entropy collapse)|Higher (maintains policy diversity)|
|Stability in Long-CoT Tasks|Moderate|High|

#### Empirical Gains

- As shown in [paper](https://arxiv.org/abs/2503.14476), each DAPO technique independently improves performance on AIME 2024. Cumulatively, DAPO achieves 50 points (avg@32), surpassing DeepSeek-R1-Zero-Qwen-32B (47 points), with only 50% of the training steps as shown in the plot from the paper.

![](https://aman.ai/primers/ai/assets/DeepSeek/DAPO.jpg)

|**Model Variant**|**AIME24 avg@32**|
|---|---|
|Naive GRPO|30|
|+ Overlong Filtering|36|
|+ Clip-Higher|38|
|+ Soft Overlong Punishment|41|
|+ Token-level Loss|42|
|+ Dynamic Sampling (DAPO Final)|**50**|

### GRPO+: a Stable Evolution of GRPO for Reinforcement Learning in DeepCoder

- [GRPO+](https://pretty-radio-b75.notion.site/DeepCoder-A-Fully-Open-Source-14B-Coder-at-O3-mini-Level-1cf81902c14680b3bee5eb349a512a51) is an advanced variant of Group Relative Policy Optimization (GRPO), specifically designed to address the instability challenges commonly encountered during reinforcement learning (RL) training of code reasoning models, especially in long-context fine-tuning scenarios. This refined approach builds upon the foundational structure of GRPO, while integrating innovations—many inspired by DAPO—to enhance training stability, reward fidelity, and response scalability.
    
- Developed for DeepCoder, a 14B open-source code reasoning model, GRPO+ introduces several key modifications that distinguish it from its predecessor. These include the removal of KL and entropy losses, the incorporation of asymmetric clipping, and the implementation of overlong filtering. Collectively, these changes create a lightweight yet robust training framework, enabling stable and efficient scaling of reasoning abilities across extended context windows.
    
- By tailoring these enhancements to the specific demands of large-scale RL in code-focused language models, GRPO+ delivers improved performance and reliability. It empowers open-source models like DeepCoder to push the boundaries of coding tasks, making it a compelling strategy for deploying frontier-level LLMs in open development environments.
    

#### Motivation for GRPO+

- During DeepCoder’s RL training on a curated set of 24,000 verifiable code problems, the research team observed that the vanilla GRPO algorithm exhibited a collapse in reward over time, especially during later stages of training. This was attributed to entropy divergence and unstable policy updates. To counter this, GRPO+ was introduced with the goal of preserving the sample efficiency of GRPO while enhancing its training stability for large-scale, long-context LLMs.
    
- The following figure ([source](https://pretty-radio-b75.notion.site/DeepCoder-A-Fully-Open-Source-14B-Coder-at-O3-mini-Level-1cf81902c14680b3bee5eb349a512a51)) illustrates this: the average reward for GRPO+ remains stable, while GRPO degrades and eventually collapses during training. The modifications introduced in GRPO+ are critical to sustaining performance throughout extended RL runs.
    

![](https://aman.ai/primers/ai/assets/DeepSeek/GRPOplus.webp)

#### Key Innovations in GRPO+

- GRPO+ introduces the following core changes to the GRPO framework:
    
    1. **No Entropy Loss**  
        In standard PPO/GRPO implementations, an entropy loss term is often included to promote exploration. However, in DeepCoder’s experiments, this entropy term caused the entropy of the output distribution to grow uncontrollably, destabilizing training. GRPO+ omits this term altogether:
        - **Rationale**: Removing entropy loss prevents exponential growth in token-level uncertainty, avoiding collapse in later iterations.
        - **Effect**: Encourages more stable convergence by reducing exploration-induced noise.
    2. **No KL Loss (No Trust Region Constraint)**  
        While GRPO retains a KL divergence penalty against a reference policy to prevent policy drift, GRPO+ completely removes this component: $$
        - \beta D_{\text{KL}}(\pi_\theta \,|\, \pi_{\text{ref}}) $$ is omitted from the loss.
        - **Rationale**: This follows insights from DAPO, which demonstrated that strict adherence to a trust region (as in PPO or GRPO) can overly constrain learning.
        - **Effect**: Training is accelerated since the computation of log probabilities from the reference model is skipped, reducing overhead.
    3. **Clip High in Surrogate Loss**  
        GRPO+ modifies the upper bound in the surrogate loss function to encourage greater exploration: min(ρiAi,clip(ρi,1−ϵ,1+ϵhigh)Ai)min(ρiAi,clip(ρi,1−ϵ,1+ϵhigh)Ai)
        - where ϵhigh>ϵϵhigh>ϵ is a relaxed clipping range.
        - **Rationale**: Standard clipping suppresses beneficial large updates; raising the upper bound retains PPO-style stability while allowing positive exploration.
        - **Effect**: Boosts learning speed and prevents premature convergence.
    4. **Overlong Filtering**  
        To allow generalization to longer context windows (up to 64K), GRPO+ introduces masked loss for truncated sequences. This overlong filtering ensures that models are not penalized for generating coherent but lengthy outputs beyond the current training context (e.g., 32K tokens):
        - **Implementation**: During training, loss is not backpropagated through the truncated parts of sequences.
        - **Effect**: Enables the model to reason over longer contexts during inference, with empirical gains on LiveCodeBench from 54% to 60.6% as context increases from 16K to 64K.

#### GRPO+ Objective Function

- The GRPO+ loss removes both KL regularization and entropy penalties from the original GRPO loss. It retains the clipped surrogate objective with a modified upper bound:
    
    JGRPO+(θ)=𝔼q∼P(Q),{oi}Gi=1∼πθold(O|q)[1G∑i=1Gmin(ρiAi,clip(ρi,1−ϵ,1+ϵhigh)Ai)]JGRPO+(θ)=Eq∼P(Q),{oi}i=1G∼πθold(O|q)[1G∑i=1Gmin(ρiAi,clip(ρi,1−ϵ,1+ϵhigh)Ai)]
    
    - where:
        - ρi=πθ(oi|q)πθold(oi|q)ρi=πθ(oi|q)πθold(oi|q)
            
        - Ai=ri−r¯σrAi=ri−r¯σr, with group-based normalization as in GRPO
        - ϵhighϵhigh is the extended clipping upper bound (e.g., 0.4 instead of 0.2)
- Compared to GRPO’s original form:
    

JGRPO(θ)=𝔼[min(ρiAi,clip(ρi,1−ϵ,1+ϵ)Ai)−βDKL(πθ‖πref)]JGRPO(θ)=E[min(ρiAi,clip(ρi,1−ϵ,1+ϵ)Ai)−βDKL(πθ‖πref)]

- GRPO+ simplifies the objective and accelerates training by avoiding KL divergence and entropy computation.

#### Implementation Details

- **Training Loop:**
    1. Sample a group of responses (G=8G=8 typically) using πθoldπθold.
    2. Score each response using a sparse **Outcome Reward Model (ORM)**:
        - A binary reward is assigned:
            - 1 if all unit tests pass
            - 0 if any test fails or the output is improperly formatted
    3. Compute the group-based normalized advantage: Ai=ri−r¯σrAi=ri−r¯σr
    4. Apply clipped surrogate loss with relaxed upper bound.
    5. Mask out loss contributions for truncated sequences to support long-context generalization.
    6. Update the policy πθπθ using standard gradient ascent.
- **No Entropy or KL Terms:**
    - Losses are purely policy-gradient based.
    - No reference model is involved during optimization.
    - No explicit entropy bonus is used, encouraging organic exploration through gradient updates and the “Clip High” mechanism.
- **Context Scaling:**
    - Training begins at 16K context length.
    - At step 180, the model is extended to 32K with overlong filtering enabled.
    - At inference, DeepCoder generalizes successfully to 64K contexts, achieving peak performance.

#### Comparison: GRPO vs. GRPO+

|**Feature**|**GRPO**|**GRPO+**|
|---|---|---|
|Critic Model|No|No|
|KL Regularization|Yes|No|
|Entropy Loss|Yes|No|
|Clipping|Symmetric (1±ε)(1±ε)|Asymmetric (1−ε, 1+εhigh)(1−ε, 1+εhigh)|
|Advantage|Group-based Normalized|Group-based Normalized|
|Long-Context Generalization|Partial|Fully Supported (via overlong filtering)|
|Reward Function|Can use dense or sparse|Sparse binary (Outcome Reward Model)|
|Use Case|General reasoning, math RL|Long-context code RL|
|Stability|Moderate|High (no collapse over time)|

#### Why GRPO+ Works

- By simplifying the objective function, GRPO+ reduces the overhead of computing KL and entropy terms while still retaining PPO-like stability through clipping.
- It tailors the training process to sparse reward signals (pass/fail from test cases) and long-form outputs (code solutions), where traditional entropy bonuses or KL constraints may be detrimental.
- GRPO+’s stripped-down yet strategically enhanced formulation reflects a pragmatic design choice: retain what works, discard what destabilizes, and adapt the core RL ideas to the idiosyncrasies of code reasoning.

#### Results and Performance Impact

- GRPO+ was critical to DeepCoder’s performance. The reward curve of GRPO+ (Figure 2 of the paper) maintains a stable upward trajectory, in contrast to GRPO, whose training reward collapses beyond a certain point. Empirically, GRPO+ enables DeepCoder to:
    - Achieve 60.6% Pass@1 on LiveCodeBench
    - Match O3-mini and O1 on coding benchmarks
    - Generalize to 64K context with no retraining
- These improvements would not have been possible under GRPO alone, which suffered from convergence and entropy-related collapse in earlier experiments.

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
    

## Further Reading

- [DAPO: An Open-Source LLM Reinforcement Learning System at Scale](https://arxiv.org/abs/2503.14476)
- [DeepCoder: A Fully Open-Source 14B Coder at O3-mini Level](https://pretty-radio-b75.notion.site/DeepCoder-A-Fully-Open-Source-14B-Coder-at-O3-mini-Level-1cf81902c14680b3bee5eb349a512a51)

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