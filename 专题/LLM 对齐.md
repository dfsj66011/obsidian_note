
- [RL from Human Feedback (RLHF)](https://aman.ai/primers/ai/llm-alignment/#rl-from-human-feedback-rlhf)
    - [Reward Model](https://aman.ai/primers/ai/llm-alignment/#reward-model)
        - [Core Functions and Architecture](https://aman.ai/primers/ai/llm-alignment/#core-functions-and-architecture)
        - [Mathematical Framework](https://aman.ai/primers/ai/llm-alignment/#mathematical-framework)
        - [Prevention of Over-optimization](https://aman.ai/primers/ai/llm-alignment/#prevention-of-over-optimization)
        - [Training and Implementation Details](https://aman.ai/primers/ai/llm-alignment/#training-and-implementation-details)
    - [Optimizing the Policy](https://aman.ai/primers/ai/llm-alignment/#optimizing-the-policy)
    - [Putting It All Together: Training Llama 2](https://aman.ai/primers/ai/llm-alignment/#putting-it-all-together-training-llama-2)
- [Proximal Policy Optimization (PPO)](https://aman.ai/primers/ai/llm-alignment/#proximal-policy-optimization-ppo)
    - [Background](https://aman.ai/primers/ai/llm-alignment/#background)
        - [Terminology: RL Overview](https://aman.ai/primers/ai/llm-alignment/#terminology-rl-overview)
            - [States and Actions in LLM Context](https://aman.ai/primers/ai/llm-alignment/#states-and-actions-in-llm-context)
        - [Policy-Based vs. Value-Based Methods](https://aman.ai/primers/ai/llm-alignment/#policy-based-vs-value-based-methods)
        - [Policy Gradient Theorem](https://aman.ai/primers/ai/llm-alignment/#policy-gradient-theorem)
        - [Predecessors of PPO](https://aman.ai/primers/ai/llm-alignment/#predecessors-of-ppo)
            - [The REINFORCE Algorithm](https://aman.ai/primers/ai/llm-alignment/#the-reinforce-algorithm)
            - [Trust Region Policy Optimization (TRPO)](https://aman.ai/primers/ai/llm-alignment/#trust-region-policy-optimization-trpo)
                - [Core Idea](https://aman.ai/primers/ai/llm-alignment/#core-idea)
                - [Strengths and Limitations](https://aman.ai/primers/ai/llm-alignment/#strengths-and-limitations)
                - [Paving the Way for PPO](https://aman.ai/primers/ai/llm-alignment/#paving-the-way-for-ppo)
    - [Intuition Behind PPO](https://aman.ai/primers/ai/llm-alignment/#intuition-behind-ppo)
        - [Why Not Naive Policy Gradients?](https://aman.ai/primers/ai/llm-alignment/#why-not-naive-policy-gradients)
        - [Why Not Trust Region Policy Optimization (TRPO)?](https://aman.ai/primers/ai/llm-alignment/#why-not-trust-region-policy-optimization-trpo)
        - [How Does PPO Solve These Problems?](https://aman.ai/primers/ai/llm-alignment/#how-does-ppo-solve-these-problems)
    - [Fundamental Components and Requirements](https://aman.ai/primers/ai/llm-alignment/#fundamental-components-and-requirements)
    - [Core Principles](https://aman.ai/primers/ai/llm-alignment/#core-principles)
        - [Policy Gradient Approach](https://aman.ai/primers/ai/llm-alignment/#policy-gradient-approach)
        - [Actor-Critic Framework](https://aman.ai/primers/ai/llm-alignment/#actor-critic-framework)
            - [The Actor (Policy Network)](https://aman.ai/primers/ai/llm-alignment/#the-actor-policy-network)
            - [The Critic (Value Function)](https://aman.ai/primers/ai/llm-alignment/#the-critic-value-function)
    - [Stages](https://aman.ai/primers/ai/llm-alignment/#stages)
    - [Generalized Advantage Estimation (GAE)](https://aman.ai/primers/ai/llm-alignment/#generalized-advantage-estimation-gae)
    - [Generalized Advantage Estimation (GAE)](https://aman.ai/primers/ai/llm-alignment/#generalized-advantage-estimation-gae-1)
        - [Formal Definition](https://aman.ai/primers/ai/llm-alignment/#formal-definition)
        - [Advantage Estimation Approaches](https://aman.ai/primers/ai/llm-alignment/#advantage-estimation-approaches)
        - [GAE Formula and Bias-Variance Trade-off](https://aman.ai/primers/ai/llm-alignment/#gae-formula-and-bias-variance-trade-off)
        - [Role in PPO’s Clipped Surrogate Objective](https://aman.ai/primers/ai/llm-alignment/#role-in-ppos-clipped-surrogate-objective)
        - [Value Function and Critic Role](https://aman.ai/primers/ai/llm-alignment/#value-function-and-critic-role)
        - [Reward and Value Model Roles](https://aman.ai/primers/ai/llm-alignment/#reward-and-value-model-roles)
    - [Key Components](https://aman.ai/primers/ai/llm-alignment/#key-components)
        - [Optimal Policy and Reference Policy](https://aman.ai/primers/ai/llm-alignment/#optimal-policy-and-reference-policy)
            - [Summary](https://aman.ai/primers/ai/llm-alignment/#summary)
        - [Surrogate Objective Function](https://aman.ai/primers/ai/llm-alignment/#surrogate-objective-function)
        - [Clipping Mechanism](https://aman.ai/primers/ai/llm-alignment/#clipping-mechanism)
        - [Data Re-use Over Multiple Epochs of Stochastic Gradient Ascent](https://aman.ai/primers/ai/llm-alignment/#data-re-use-over-multiple-epochs-of-stochastic-gradient-ascent)
        - [Value Function and Baseline](https://aman.ai/primers/ai/llm-alignment/#value-function-and-baseline)
    - [PPO’s Objective Function: Clipped Surrogate Loss](https://aman.ai/primers/ai/llm-alignment/#ppos-objective-function-clipped-surrogate-loss)
        - [Intuition](https://aman.ai/primers/ai/llm-alignment/#intuition)
        - [Components](https://aman.ai/primers/ai/llm-alignment/#components)
            - [Purpose of the Clipping Mechanism](https://aman.ai/primers/ai/llm-alignment/#purpose-of-the-clipping-mechanism)
            - [Purpose of Surrogate Loss](https://aman.ai/primers/ai/llm-alignment/#purpose-of-surrogate-loss)
        - [Mathematical Formulation](https://aman.ai/primers/ai/llm-alignment/#mathematical-formulation)
            - [PPO Loss with Clipped Surrogate Loss](https://aman.ai/primers/ai/llm-alignment/#ppo-loss-with-clipped-surrogate-loss)
            - [PPO Loss with KL Divergence](https://aman.ai/primers/ai/llm-alignment/#ppo-loss-with-kl-divergence)
            - [PPO Loss with Clipped Surrogate Loss and KL Penalty](https://aman.ai/primers/ai/llm-alignment/#ppo-loss-with-clipped-surrogate-loss-and-kl-penalty)
    - [PPO for LLM Policy Optimization](https://aman.ai/primers/ai/llm-alignment/#ppo-for-llm-policy-optimization)
        - [RLHF Overview](https://aman.ai/primers/ai/llm-alignment/#rlhf-overview)
        - [PPO in LLM Training](https://aman.ai/primers/ai/llm-alignment/#ppo-in-llm-training)
    - [Practical Implementation of PPO](https://aman.ai/primers/ai/llm-alignment/#practical-implementation-of-ppo)
        - [Pseudocode for PPO](https://aman.ai/primers/ai/llm-alignment/#pseudocode-for-ppo)
        - [PPO with OpenAI’s `transformers` and `trl`](https://aman.ai/primers/ai/llm-alignment/#ppo-with-openais-transformers-and-trl)
    - [Typical Hyperparameters](https://aman.ai/primers/ai/llm-alignment/#typical-hyperparameters)
    - [Variants of PPO](https://aman.ai/primers/ai/llm-alignment/#variants-of-ppo)
    - [PPO-Clip](https://aman.ai/primers/ai/llm-alignment/#ppo-clip)
    - [PPO-Penalty](https://aman.ai/primers/ai/llm-alignment/#ppo-penalty)
    - [Advantages of PPO](https://aman.ai/primers/ai/llm-alignment/#advantages-of-ppo)
    - [Simplified Example](https://aman.ai/primers/ai/llm-alignment/#simplified-example)
    - [Summary](https://aman.ai/primers/ai/llm-alignment/#summary-1)
    - [Related: How is the Policy Represented As a Neural Network?](https://aman.ai/primers/ai/llm-alignment/#related-how-is-the-policy-represented-as-a-neural-network)
    - [Policy Representation in RL Algorithms](https://aman.ai/primers/ai/llm-alignment/#policy-representation-in-rl-algorithms)
        - [Summary](https://aman.ai/primers/ai/llm-alignment/#summary-2)
- [RL with AI Feedback (RLAIF)](https://aman.ai/primers/ai/llm-alignment/#rl-with-ai-feedback-rlaif)
- [Direct Preference Optimization (DPO)](https://aman.ai/primers/ai/llm-alignment/#direct-preference-optimization-dpo)
    - [DPO’s Binary Cross-Entropy Loss](https://aman.ai/primers/ai/llm-alignment/#dpos-binary-cross-entropy-loss)
        - [Simplified Process](https://aman.ai/primers/ai/llm-alignment/#simplified-process)
        - [Loss Function Equation](https://aman.ai/primers/ai/llm-alignment/#loss-function-equation)
            - [Loss Function Design Choices](https://aman.ai/primers/ai/llm-alignment/#loss-function-design-choices)
                - [Negative Sign in Front of the Loss](https://aman.ai/primers/ai/llm-alignment/#negative-sign-in-front-of-the-loss)
                - [Why the Sigmoid Function (σσ) is Used](https://aman.ai/primers/ai/llm-alignment/#why-the-sigmoid-function-sigma-is-used)
                - [Role of ββ in the DPO Loss Function](https://aman.ai/primers/ai/llm-alignment/#role-of-beta-in-the-dpo-loss-function)
                - [Significant of the DPO Loss](https://aman.ai/primers/ai/llm-alignment/#significant-of-the-dpo-loss)
            - [Mapping from the Standard Binary Cross-Entropy Loss to the DPO Loss](https://aman.ai/primers/ai/llm-alignment/#mapping-from-the-standard-binary-cross-entropy-loss-to-the-dpo-loss)
                - [Standard Binary Cross-Entropy Loss](https://aman.ai/primers/ai/llm-alignment/#standard-binary-cross-entropy-loss)
                - [Mapping BCE Loss to DPO Loss](https://aman.ai/primers/ai/llm-alignment/#mapping-bce-loss-to-dpo-loss)
                - [Intuition of the Mapping](https://aman.ai/primers/ai/llm-alignment/#intuition-of-the-mapping)
        - [Key Insights](https://aman.ai/primers/ai/llm-alignment/#key-insights)
    - [How Does DPO Generate Two Responses and Assign Probabilities to Them?](https://aman.ai/primers/ai/llm-alignment/#how-does-dpo-generate-two-responses-and-assign-probabilities-to-them)
    - [DPO and It’s Use of the Bradley-Terry Model](https://aman.ai/primers/ai/llm-alignment/#dpo-and-its-use-of-the-bradley-terry-model)
        - [How Does DPO Implicitly Use a Bradley-Terry Model (if It Does Not Explicitly Use a Reward Model)?](https://aman.ai/primers/ai/llm-alignment/#how-does-dpo-implicitly-use-a-bradley-terry-model-if-it-does-not-explicitly-use-a-reward-model)
            - [Key Concepts in DPO Without an Explicit Reward Model](https://aman.ai/primers/ai/llm-alignment/#key-concepts-in-dpo-without-an-explicit-reward-model)
            - [Implicit Use of Bradley-Terry Model](https://aman.ai/primers/ai/llm-alignment/#implicit-use-of-bradley-terry-model)
            - [Steps in DPO Without Explicit Reward Model](https://aman.ai/primers/ai/llm-alignment/#steps-in-dpo-without-explicit-reward-model)
            - [Practical Implementation](https://aman.ai/primers/ai/llm-alignment/#practical-implementation)
    - [Video Tutorial](https://aman.ai/primers/ai/llm-alignment/#video-tutorial)
    - [Summary](https://aman.ai/primers/ai/llm-alignment/#summary-3)
- [Kahneman-Tversky Optimization (KTO)](https://aman.ai/primers/ai/llm-alignment/#kahneman-tversky-optimization-kto)
    - [KTO’s Loss Function](https://aman.ai/primers/ai/llm-alignment/#ktos-loss-function)
    - [Core Principles from Prospect Theory](https://aman.ai/primers/ai/llm-alignment/#core-principles-from-prospect-theory)
    - [Key Elements of KTO’s Loss Function](https://aman.ai/primers/ai/llm-alignment/#key-elements-of-ktos-loss-function)
    - [Loss Function Equation](https://aman.ai/primers/ai/llm-alignment/#loss-function-equation-1)
    - [Intuition Behind the Loss Function](https://aman.ai/primers/ai/llm-alignment/#intuition-behind-the-loss-function)
    - [Practical Considerations](https://aman.ai/primers/ai/llm-alignment/#practical-considerations)
    - [Summary](https://aman.ai/primers/ai/llm-alignment/#summary-4)
- [Group Relative Policy Optimization (GRPO)](https://aman.ai/primers/ai/llm-alignment/#group-relative-policy-optimization-grpo)
    - [Key Features and Approach](https://aman.ai/primers/ai/llm-alignment/#key-features-and-approach)
    - [GRPO Equations](https://aman.ai/primers/ai/llm-alignment/#grpo-equations)
    - [Implementation Details](https://aman.ai/primers/ai/llm-alignment/#implementation-details)
    - [Pros and Cons](https://aman.ai/primers/ai/llm-alignment/#pros-and-cons)
        - [Pros](https://aman.ai/primers/ai/llm-alignment/#pros)
        - [Cons](https://aman.ai/primers/ai/llm-alignment/#cons)
    - [Applications and Results](https://aman.ai/primers/ai/llm-alignment/#applications-and-results)
- [Comparative Analysis: REINFORCE vs. TRPO vs. PPO vs. DPO vs. KTO vs. APO vs. GRPO](https://aman.ai/primers/ai/llm-alignment/#comparative-analysis-reinforce-vs-trpo-vs-ppo-vs-dpo-vs-kto-vs-apo-vs-grpo)
    - [Tabular Comparison](https://aman.ai/primers/ai/llm-alignment/#tabular-comparison)
- [Bias Concerns and Mitigation Strategies](https://aman.ai/primers/ai/llm-alignment/#bias-concerns-and-mitigation-strategies)
- [TRL - Transformer RL](https://aman.ai/primers/ai/llm-alignment/#trl---transformer-rl)
- [Selected Papers](https://aman.ai/primers/ai/llm-alignment/#selected-papers)
    - [OpenAI’s Paper on InstructGPT: Training Language Models to Follow Instructions with Human Feedback](https://aman.ai/primers/ai/llm-alignment/#openais-paper-on-instructgpt-training-language-models-to-follow-instructions-with-human-feedback)
    - [Constitutional AI: Harmlessness from AI Feedback](https://aman.ai/primers/ai/llm-alignment/#constitutional-ai-harmlessness-from-ai-feedback)
    - [OpenAI’s Paper on PPO: Proximal Policy Optimization Algorithms](https://aman.ai/primers/ai/llm-alignment/#openais-paper-on-ppo-proximal-policy-optimization-algorithms)
    - [A General Language Assistant As a Laboratory for Alignment](https://aman.ai/primers/ai/llm-alignment/#a-general-language-assistant-as-a-laboratory-for-alignment)
    - [Anthropic’s Paper on Constitutional AI: Constitutional AI: Harmlessness from AI Feedback](https://aman.ai/primers/ai/llm-alignment/#anthropics-paper-on-constitutional-ai-constitutional-ai-harmlessness-from-ai-feedback)
    - [RLAIF: Scaling RL from Human Feedback with AI Feedback](https://aman.ai/primers/ai/llm-alignment/#rlaif-scaling-rl-from-human-feedback-with-ai-feedback)
    - [A General Theoretical Paradigm to Understand Learning from Human Preferences](https://aman.ai/primers/ai/llm-alignment/#a-general-theoretical-paradigm-to-understand-learning-from-human-preferences)
    - [SLiC-HF: Sequence Likelihood Calibration with Human Feedback](https://aman.ai/primers/ai/llm-alignment/#slic-hf-sequence-likelihood-calibration-with-human-feedback)
    - [Reinforced Self-Training (ReST) for Language Modeling](https://aman.ai/primers/ai/llm-alignment/#reinforced-self-training-rest-for-language-modeling)
    - [Beyond Human Data: Scaling Self-Training for Problem-Solving with Language Models](https://aman.ai/primers/ai/llm-alignment/#beyond-human-data-scaling-self-training-for-problem-solving-with-language-models)
    - [Diffusion Model Alignment Using Direct Preference Optimization](https://aman.ai/primers/ai/llm-alignment/#diffusion-model-alignment-using-direct-preference-optimization)
    - [Human-Centered Loss Functions (HALOs)](https://aman.ai/primers/ai/llm-alignment/#human-centered-loss-functions-halos)
    - [Nash Learning from Human Feedback](https://aman.ai/primers/ai/llm-alignment/#nash-learning-from-human-feedback)
    - [Group Preference Optimization: Few-shot Alignment of Large Language Models](https://aman.ai/primers/ai/llm-alignment/#group-preference-optimization-few-shot-alignment-of-large-language-models)
    - [ICDPO: Effectively Borrowing Alignment Capability of Others Via In-context Direct Preference Optimization](https://aman.ai/primers/ai/llm-alignment/#icdpo-effectively-borrowing-alignment-capability-of-others-via-in-context-direct-preference-optimization)
    - [ORPO: Monolithic Preference Optimization Without Reference Model](https://aman.ai/primers/ai/llm-alignment/#orpo-monolithic-preference-optimization-without-reference-model)
    - [Human Alignment of Large Language Models Through Online Preference Optimisation](https://aman.ai/primers/ai/llm-alignment/#human-alignment-of-large-language-models-through-online-preference-optimisation)
    - [Contrastive Preference Optimization: Pushing the Boundaries of LLM Performance in Machine Translation](https://aman.ai/primers/ai/llm-alignment/#contrastive-preference-optimization-pushing-the-boundaries-of-llm-performance-in-machine-translation)
    - [SDPO: Don’t Use Your Data All at Once](https://aman.ai/primers/ai/llm-alignment/#sdpo-dont-use-your-data-all-at-once)
    - [RS-DPO: a Hybrid Rejection Sampling and Direct Preference Optimization Method for Alignment of Large Language Models](https://aman.ai/primers/ai/llm-alignment/#rs-dpo-a-hybrid-rejection-sampling-and-direct-preference-optimization-method-for-alignment-of-large-language-models)
    - [The Unlocking Spell on Base LLMs: Rethinking Alignment Via In-Context Learning](https://aman.ai/primers/ai/llm-alignment/#the-unlocking-spell-on-base-llms-rethinking-alignment-via-in-context-learning)
    - [MDPO: Conditional Preference Optimization for Multimodal Large Language Models](https://aman.ai/primers/ai/llm-alignment/#mdpo-conditional-preference-optimization-for-multimodal-large-language-models)
    - [Aligning Large Multimodal Models with Factually Augmented RLHF](https://aman.ai/primers/ai/llm-alignment/#aligning-large-multimodal-models-with-factually-augmented-rlhf)
    - [Statistical Rejection Sampling Improves Preference Optimization](https://aman.ai/primers/ai/llm-alignment/#statistical-rejection-sampling-improves-preference-optimization)
    - [Sycophancy to Subterfuge: Investigating Reward Tampering in Language Models](https://aman.ai/primers/ai/llm-alignment/#sycophancy-to-subterfuge-investigating-reward-tampering-in-language-models)
    - [Is DPO Superior to PPO for LLM Alignment? a Comprehensive Study](https://aman.ai/primers/ai/llm-alignment/#is-dpo-superior-to-ppo-for-llm-alignment-a-comprehensive-study)
    - [Pairwise Proximal Policy Optimization: Harnessing Relative Feedback for LLM Alignment](https://aman.ai/primers/ai/llm-alignment/#pairwise-proximal-policy-optimization-harnessing-relative-feedback-for-llm-alignment)
    - [BPO: Supercharging Online Preference Learning by Adhering to the Proximity of Behavior LLM](https://aman.ai/primers/ai/llm-alignment/#bpo-supercharging-online-preference-learning-by-adhering-to-the-proximity-of-behavior-llm)
    - [SimPO: Simple Preference Optimization with a Reference-Free Reward](https://aman.ai/primers/ai/llm-alignment/#simpo-simple-preference-optimization-with-a-reference-free-reward)
    - [Discovering Preference Optimization Algorithms with and for Large Language Models](https://aman.ai/primers/ai/llm-alignment/#discovering-preference-optimization-algorithms-with-and-for-large-language-models)
    - [Anchored Preference Optimization and Contrastive Revisions: Addressing Underspecification in Alignment](https://aman.ai/primers/ai/llm-alignment/#anchored-preference-optimization-and-contrastive-revisions-addressing-underspecification-in-alignment)
    - [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://aman.ai/primers/ai/llm-alignment/#deepseekmath-pushing-the-limits-of-mathematical-reasoning-in-open-language-models)
- [Further Reading](https://aman.ai/primers/ai/llm-alignment/#further-reading)
    - [HuggingFace’s Alignment Handbook](https://aman.ai/primers/ai/llm-alignment/#huggingfaces-alignment-handbook)
    - [Empirical Evaluation: DPO vs. IPO vs. KTO](https://aman.ai/primers/ai/llm-alignment/#empirical-evaluation-dpo-vs-ipo-vs-kto)
- [FAQs](https://aman.ai/primers/ai/llm-alignment/#faqs)
    - [In RLHF, What are the Memory Requirements of the Reward and Critic Model Compared to the Policy/reference Model?](https://aman.ai/primers/ai/llm-alignment/#in-rlhf-what-are-the-memory-requirements-of-the-reward-and-critic-model-compared-to-the-policyreference-model)
    - [Why is the PPO/GRPO Objective Called a Clipped “surrogate” Objective?](https://aman.ai/primers/ai/llm-alignment/#why-is-the-ppogrpo-objective-called-a-clipped-surrogate-objective)
    - [Is the Importance Sampling Ratio Also Called the Policy or Likelihood Ratio?](https://aman.ai/primers/ai/llm-alignment/#is-the-importance-sampling-ratio-also-called-the-policy-or-likelihood-ratio)
    - [Does REINFORCE and TRPO in Policy Optimization Also Use a Surrogate Loss?](https://aman.ai/primers/ai/llm-alignment/#does-reinforce-and-trpo-in-policy-optimization-also-use-a-surrogate-loss)
- [References](https://aman.ai/primers/ai/llm-alignment/#references)

### 一、概览

- 2017 年，OpenAI 在他们的论文 [Deep RL from human preferences](https://arxiv.org/abs/1706.03741) 中介绍了一种突破性的机器学习方法，称为 RLHF，特别关注人类偏好。这一创新概念自此激发了该领域的进一步研究和发展。
- RLHF 背后的概念既简单又强大：它涉及使用预训练语言模型，并让人类评估者对其输出进行排序。然后，这一排序指导模型对某些类型的响应产生偏好，从而生成更可靠和安全的输出。
- RLHF 有效地利用人类反馈来提升语言模型的性能。它结合了 RL 算法的优势与人类输入的细微理解，促进模型的持续学习和改进。
- 通过整合人类反馈，RLHF 不仅提高了模型的自然语言理解和生成能力，还增强了其在文本分类或翻译等特定任务中的效率。
- 此外，RLHF 在解决语言模型中的偏见问题上发挥了关键作用。通过允许人类输入来引导和纠正模型的语言使用，它促进了更公平和包容的交流。然而，需注意在此过程中可能引入的人为偏见。

### 二、背景：LLM 的预训练和后训练

- LLMs 的训练过程分为两个不同的阶段：预训练和后训练，每个阶段在开发语言模型中具有独特的目的：

    1. *预训练* ：这一阶段涉及大规模训练，模型通过广泛的网络数据学习下一个词的预测。数据集的规模通常达到数万亿个 token，包括公开可用和专有数据集的混合，以增强语言理解。目标是使模型能够根据从大量文本数据集中得出的统计概率预测词序列。
    
    2. *后训练* ：这一阶段旨在提高模型的推理能力，通常包括两个阶段：
    
        - *阶段 1：SFT* 使用少量高质量专家推理数据对模型进行微调，通常在 1 万到 10 万个提示-响应对之间。此阶段采用监督学习对模型进行微调，包括指令遵循、问答和思维链演示。目标是使模型有效模仿专家演示，但有限的专家数据可用性需要额外的训练方法。
        
        - *阶段 2：RLHF* 这一阶段通过引入人类偏好数据来训练奖励模型，从而通过强化学习指导模型的学习。RLHF 使模型与细微的人类偏好对齐，确保生成更有意义、安全和高质量的响应。





## RL from Human Feedback (RLHF)

- RLHF was introduced to address a significant challenge in training LLMs: although they could predict the next token accurately, they were not necessarily aligned with human values such as helpfulness, harmlessness, and honesty. RLHF provides a mechanism to guide the model toward producing outputs that align with human preferences.
- The image below [(source)](https://openai.com/research/instruction-following) illustrates the RLHF process as applied in InstructGPT:

![](https://aman.ai/primers/ai/assets/rlhf/openAI.png)

- The image outlines a three-step process used to train a language model using RLHF:
    1. **Collect Demonstration Data and Train a Supervised Policy:**
        - A prompt is taken from a collection of prompts.
        - A human labeler (an annotator) provides the desired output as a demonstration.
        - The language model is fine-tuned using supervised learning to mimic these demonstrations.
    2. **Collect Comparison Data and Train a Reward Model:**
        - A prompt is chosen, and the model generates several potential outputs.
        - A labeler then ranks these outputs from best to worst according to criteria like helpfulness or accuracy.
        - A reward model is trained using between 100,000 to 1 million ranked comparison data points to predict human preferences from these rankings.
    3. **Optimize a Policy Against the Reward Model Using RL:**
        - A new prompt is selected from the dataset.
        - The model generates a response based on its current policy.
        - The reward model assigns a reward to the response.
        - The model is fine-tuned using RL algorithms, such as Proximal Policy Optimization (PPO), to maximize the reward model’s scores. Typically, between 10,000 to 100,000 prompts are used in this stage.
- The RLHF process can be summarized in the following flowchart by [Chip Huyen](https://huyenchip.com/2023/05/02/rlhf.html):

![](https://aman.ai/primers/ai/assets/rlhf/chip.jpg)<!–

- Here’s a breakdown of the flowchart:
    
    1. **Language Modeling**:
        - This is the first stage where a language model is trained on a large dataset. The dataset is composed of a vast amount of text data, which can be of varying quality. The training at this stage is optimized for text completion tasks. The scale mentioned is over 1 trillion tokens, and examples of such models include GPT-x, Gopher, Falcon, LLama, Pythia, Bloom, and StableLM. This results in a Pretrained Large Language Model (LLM).
        - To expand further: This is phase of pretraining involves developing a large language model (LLM) that functions as a completion machine, using statistical knowledge to predict the likelihood of sequences in language. This is achieved by feeding the model extensive text data, often exceeding trillions of tokens, from varied sources to learn language patterns. The model’s efficacy is contingent on the quality of the training data, with the aim to minimize cross-entropy loss across training samples. As the Internet becomes saturated with data, including that generated by LLMs themselves, there’s a growing need to access proprietary data for further model improvement.
    2. **Supervised Finetuning**:
        - In the second stage, the pretrained LLM is further finetuned using high-quality data, which is often dialogue-focused to better suit conversational AI. This is done using demonstration data, and the process generates a Supervised Finetuning (SFT) model. The amount of data used for finetuning ranges from 10,000 to 100,000 (prompt, response) pairs. Examples of models that go through this process are Dolly-v2 and Falcon-Instruct.
        - To elaborate: This is phase involves Supervised Fine-Tuning (SFT) for dialogue, where a pre-trained model is optimized to generate preferred responses to prompts, such as direct answers to questions. High-quality demonstration data, consisting of prompt-response pairs, guides the model’s behavior. With about 13,000 such pairs, OpenAI’s approach emphasizes quality through expert labelers, while others like DeepMind use heuristics for data selection. The SFT process is critical for tailoring the model’s outputs to practical use cases, leveraging a smaller yet refined dataset to minimize cross-entropy loss for the dialogue-specific responses.
    3. **Classification and Reward Modeling**:
        - The model undergoes a classification process where it is trained to give a scalar score to responses based on human feedback. This is to ensure that the model can evaluate the quality of its own responses. The data used here is called comparison data, and involves 100,000 to 1 million comparisons between a prompt, a winning response, and a losing response. This stage results in the creation of a Reward model.
    4. **RL (RLHF)**:
        - This phase involves using RL techniques to train the model to generate responses that maximize the scores given by the reward model, effectively teaching the AI to prefer high-quality responses as judged by humans. This stage uses prompts (10,000 to 100,000) to adjust the model’s responses. The end product is the Final model, which should be adept at handling prompts in a way that aligns with human preferences. Examples of such models are InstructGPT, ChatGPT, Claude, and StableVicuna.
        - This phase of RLHF is an advanced training process that refines the behavior of a Supervised Fine-Tuned (SFT) model. It uses human feedback to score AI-generated responses, guiding the model to produce high-quality outputs. RLHF involves training a reward model to evaluate responses and optimizing the language model to prioritize these high scores. This phase addresses the limitations of SFT by providing nuanced feedback on the quality of responses, not just their plausibility, and mitigates issues like hallucination by aligning model outputs more closely with human expectations. Despite its complexity, RLHF has been shown to enhance model performance significantly over SFT alone.
- Below, we will expand on the key steps mentioned in this flow. –>
    

### Reward Model

- The reward model plays a crucial role in RLHF by automating the ranking of responses. Since human evaluators cannot rank every model output, a reward model is trained to predict these rankings. The image below [(source)](https://huggingface.co/blog/rlhf) illustrates how a reward model functions:

![](https://aman.ai/primers/ai/assets/rlhf/6.png)

#### Core Functions and Architecture

- The reward model’s primary function is to evaluate input (such as text sequences) and produce a scalar reward that indicates human preferences or judgments about the quality or desirability of the input. Several architectural approaches are used:
    
    1. **LM Classifiers**: Language models fine-tuned as binary classifiers to score which response better aligns with human preferences
        
    2. **Value Networks**: Regression models that predict scalar ratings representing relative human preference
        
    3. **Critique Generators**: Language models trained to generate evaluative critiques explaining which response is better and why, used in conjunction with instruction tuning
        

#### Mathematical Framework

- The reward model is trained using ranked comparison data and assigns a scalar score to model-generated responses. The training process follows a specific loss function derived from the Bradley-Terry model, ensuring accurate predictions of human preferences. The loss function is formulated as:
    
    (ϕ)=−logσ(Rϕ(p,ri)−Rϕ(p,rj))L(ϕ)=−log⁡σ(Rϕ(p,ri)−Rϕ(p,rj))
    
    - where:
        - σσ: the sigmoid function
        - RϕRϕ: the reward model
        - pp: the prompt
        - ri,rjri,rj: different responses
- The probability that a rater prefers response r_i over r_j is given by:
    

P(ri≻rj)=exp(Rϕ(p,ri))exp(Rϕ(p,ri))+exp(Rϕ(p,rj))P(ri≻rj)=exp⁡(Rϕ(p,ri))exp⁡(Rϕ(p,ri))+exp⁡(Rϕ(p,rj))

- Note that the reward for a partial response is always 0; only for complete responses from the LLM would the reward model return a non-zero scalar score. This important fact is crucial in guiding the RL process.

#### Prevention of Over-optimization

- To prevent over-optimization, the reward function incorporates a penalty term based on the Kullback-Leibler (KL) divergence, which ensures that the fine-tuned model does not deviate excessively from its pretrained counterpart.
    - As a quick recap, KL divergence measures the difference between two probability distributions. It compares the probability distribution of the agent’s current policy with a reference distribution representing desired behavior. This penalty ensures the RL policy stays reasonably close to the pretrained model’s behavior.

#### Training and Implementation Details

1. **Partial Response Handling**: Partial responses receive a reward of zero, reinforcing the generation of complete and meaningful outputs
    
2. **Alignment Criteria**: The reward model is trained using ranked comparison data based on multiple criteria:
    - Helpfulness
    - Harmlessness
    - Honesty
3. **Distribution Overlap**: The KL divergence is used to overlap two distributions:
    - Initial language model output
    - Tuned language model output

- The goal is to convert potentially noisy human subjective judgments into a consistent reward function that can effectively guide the RL agent’s training. The quality of the reward modeling directly impacts the overall performance of the RLHF system.

### Optimizing the Policy

- The policy refers to a strategy or a set of rules that an agent uses to make decisions in an environment. Put simply, the policy defines how the agent selects actions based on its current observations or state.
- The policy optimization process involves RL techniques that iteratively refine the policy based on reward feedback. The reward model provides feedback based on human preferences, and the policy is optimized iteratively to maximize reward while maintaining a stable learning trajectory. The stability aspect is enforced by maintaining a certain level of similarity to its previous version (to prevent drastic changes that could lead to instability)
- Popular policy optimization methods – specifically applied to LLMs – include:
    - **[Proximal Policy Optimization (PPO)](https://aman.ai/primers/ai/llm-alignment/#proximal-policy-optimization-ppo):** A widely-used RL algorithm that balances exploration and exploitation while maintaining training stability.
    - **[Direct Preference Optimization (DPO)](https://aman.ai/primers/ai/llm-alignment/#direct-preference-optimization-dpo):** An alternative approach where the policy directly optimizes the relative log probability of preferred responses using a binary cross-entropy loss, balancing human feedback alignment with KL divergence constraints.
    - **[Group Relative Policy Optimization (GRPO)](https://aman.ai/primers/ai/llm-alignment/#group-relative-preference-optimization-grpo):** A PPO variant that removes the critic model and estimates the baseline from group scores, improving memory efficiency and performance in complex tasks like mathematical reasoning.
- Through RLHF, models like InstructGPT and ChatGPT have achieved enhanced alignment with human expectations, producing more beneficial and contextually appropriate responses.

### Putting It All Together: Training Llama 2

- As a case study of how Llama 2 was trained, let’s go over the multi-stage process that integrates both human and model-generated feedback to refine the performance of language models. Here’s how it functions:
    1. **Pretraining:** Llama 2 undergoes initial pretraining with large amounts of data through self-supervised learning. This stage lays the foundation for the model by enabling it to understand language patterns and context.
    2. **Supervised Fine-Tuning:** The model then undergoes supervised fine-tuning with instruction data, where it is trained to respond to prompts in ways that align with specific instructions.
    3. **Reward Models Creation (RLHF Step 1):** Two separate reward models are created using human preference data –- one for helpfulness and one for safety. These models are trained to predict which of two responses is better based on human judgments.
    4. **Margin Loss and Ranking:** Unlike the previous approach that generates multiple outputs and uses a “k choose 2” comparison method, Llama 2’s dataset is based on binary comparisons, and each labeler is presented with only two responses at a time. A margin label is collected alongside binary ranks to indicate the degree of preference, which can inform the ranking loss calculation.
    5. **Rejection Sampling and Alignment using PPO (RLHF Step 2):** Finally, Llama 2 employs rejection sampling and Proximal Policy Optimization (PPO). Rejection sampling is used to draw multiple outputs and select the one with the highest reward for the gradient update. PPO is then used to align the model further, making the model’s responses more safe and helpful.
- The image below [(source)](https://ai.meta.com/resources/models-and-libraries/llama/) showing how Llama 2 leverages RLHF.

![](https://aman.ai/primers/ai/assets/rlhf/llama.jpeg)

## Proximal Policy Optimization (PPO)

- Proximal Policy Optimization (PPO), introduced by [Schulman et al. (2017)](https://arxiv.org/abs/1707.06347), is a RL algorithm that addresses some key challenges in training agents through policy gradient methods.
- PPO is widely used in robotics, gaming, and large language model (LLM) policy optimization, particularly in RLHF.

### Background

#### Terminology: RL Overview

- RL is a framework for training agents that interact with an environment to maximize cumulative rewards.
    
    - **Agent:** Learns to act in an environment.
    - **Environment:** Defines state transitions and rewards.
    - **State (ss):** The agent’s perception of the environment at a given time.
    - **Action (aa):** The agent’s choice affecting the environment.
    - **Reward (rr):** A scalar feedback signal.
    - **Policy (π(a‖s)π(a‖s)):** A probability distribution over actions given a state.
    - **Value Function (Vπ(s)Vπ(s)):** Expected cumulative rewards from state ss.
    - **Advantage Function (Aπ(s,a)Aπ(s,a)):** Measures how much better an action is compared to the baseline value.
- RL problems are modeled as Markov Decision Processes (MDPs) with:
    
    - States (SS)
    - Actions (AA)
    - Transition probabilities (P(s′‖s,a)P(s′‖s,a))
    - Rewards (R(s,a)R(s,a))
    - Discount factor (γγ) for future rewards

##### States and Actions in LLM Context

- In the LLM context, states and actions are defined at the token level.
- Let’s say we give our LLM a prompt pp. The LLM then starts generating a response riri of length TT one token at a time:
    - t=0t=0: state is just the prompt, i.e., s0={p}s0={p}, and the first action a0a0 is the first word token generated
    - t=1t=1: state becomes s1={p,a0}s1={p,a0}, as the LLM generates the next action a1a1 while conditioned on the state
    - t=T−1t=T−1: state is sT−1={p,a0:T−2}sT−1={p,a0:T−2}, and the LLM generates the final action aT−1aT−1

#### Policy-Based vs. Value-Based Methods

- **Value-Based Methods:** Learn a function to estimate future rewards (e.g., Q-learning, Deep Q-Networks).
- **Policy-Based Methods:** Directly optimize the policy π(a‖s)π(a‖s).
- **Actor-Critic Methods:** Combine both approaches by learning a policy (actor) and a value function (critic).

#### Policy Gradient Theorem

- The objective in policy optimization is to maximize the expected reward:

J(θ)=𝔼τ∼πθ[R(τ)]J(θ)=Eτ∼πθ[R(τ)]

- Using the policy gradient theorem, the gradient of J(θ)J(θ) is:

∇θJ(θ)=𝔼πθ[∇θlogπθ(a|s)Aπ(s,a)]∇θJ(θ)=Eπθ[∇θlog⁡πθ(a|s)Aπ(s,a)]

#### Predecessors of PPO

- [REINFORCE](https://link.springer.com/content/pdf/10.1007/BF00992696.pdf) and [TRPO](https://arxiv.org/abs/1502.05477) serve as foundational approaches to policy optimization, each addressing different challenges in RL. REINFORCE provides a simple yet high-variance method for optimizing policies, while TRPO improves stability by constraining updates. These methods paved the way for Proximal Policy Optimization (PPO), which builds on TRPO by introducing a more efficient and scalable optimization framework commonly used in modern RL applications.

##### The REINFORCE Algorithm

- One of the earliest policy optimization methods in RL is REINFORCE, introduced in [Williams (1992)](https://link.springer.com/content/pdf/10.1007/BF00992696.pdf). REINFORCE is a policy gradient algorithm that directly optimizes the policy by maximizing expected rewards.
- The key idea behind REINFORCE is the use of Monte Carlo sampling to estimate the policy gradient, which is then used to update the policy parameters using stochastic gradient ascent.
- The update rule follows: θ←θ+α∑Tt=0∇θlogπθ(at|st)Rtθ←θ+α∑t=0T∇θlog⁡πθ(at|st)Rt
    - where:
        - πθπθ is the policy parameterized by θθ,
        - atat is the action taken at time tt,
        - stst is the state at time tt,
        - RtRt is the cumulative return from time step tt, and
        - αα is the learning rate.
- Despite its simplicity, REINFORCE suffers from high variance in gradient estimates, leading to unstable training. Variance reduction techniques like baseline subtraction (using a value function) are often used to mitigate this issue.

##### Trust Region Policy Optimization (TRPO)

- Trust Region Policy Optimization (TRPO) is an advanced policy optimization algorithm introduced by [Schulman et al. (2015)](https://arxiv.org/abs/1502.05477). It was developed to improve upon traditional policy gradient methods like REINFORCE by enforcing a constraint on policy updates, preventing large, destabilizing changes that can degrade performance.

###### Core Idea

- TRPO aims to optimize the expected advantage-weighted policy ratio while ensuring that updates remain within a predefined trust region. The objective function is:

maxθ𝔼s,a∼πθold[πθ(a|s)πθold(a|s)Aπθold(s,a)]maxθEs,a∼πθold[πθ(a|s)πθold(a|s)Aπθold(s,a)]

- subject to the Kullback-Leibler (KL) divergence constraint:

DKL(πθ||πθold)≤δDKL(πθ||πθold)≤δ

- where:
    - Aπθold(s,a)Aπθold(s,a) is the advantage function,
    - DKLDKL is the KL divergence measuring the difference between old and new policies,
    - δδ is a small threshold defining the trust region.
- This KL constraint ensures that policy updates are not too aggressive, preventing performance collapse and maintaining stability.

###### Strengths and Limitations

- **Stable Learning**: TRPO’s constraint limits drastic changes in policy updates, making it robust in complex environments such as robotic control and RL applications.
- **Computational Complexity**: TRPO requires solving a constrained optimization problem, which involves computing second-order derivatives, making it computationally expensive.
- **Impact on PPO**: TRPO inspired Proximal Policy Optimization (PPO), which simplifies the trust region approach by using a clipped objective function to balance exploration and exploitation efficiently.
- Overall, TRPO remains a cornerstone in RL, particularly in high-stakes applications where stability is crucial.

###### Paving the Way for PPO

- TRPO introduced trust region constraints to stabilize learning, paving the way for PPO, which simplifies TRPO by using a clipped objective function to balance exploration and exploitation in policy updates.

### Intuition Behind PPO

- PPO is designed to stabilize policy updates by ensuring that new policies do not deviate too much from previous ones.

#### Why Not Naive Policy Gradients?

- Traditional policy gradients (REINFORCE) often lead to unstable updates because they do not constrain how much the policy changes from one iteration to the next.
- This can cause catastrophic forgetting or sudden performance drops.

#### Why Not Trust Region Policy Optimization (TRPO)?

- TRPO stabilizes learning by enforcing a trust region constraint using KL-divergence, but solving the constrained optimization problem is computationally expensive.

#### How Does PPO Solve These Problems?

- PPO simplifies TRPO by introducing a clipping mechanism in the objective function.
- This allows for stable policy updates without requiring second-order optimization or explicit KL-divergence constraints.
- Thus, PPO achieves a balance between stability and efficiency, making it highly practical for large-scale RL applications.

### Fundamental Components and Requirements

- PPO requires the following fundamental components:
    - **Policy** πθπθ: The LLM that has been pre-trained or undergone supervised fine-tuning.
    - **Reward Model** RϕRϕ: A trained and frozen network that provides a scalar reward given a complete response to a prompt.
    - **Critic** VγVγ: Also known as the value function, a learnable network that takes in a partial response to a prompt and predicts the scalar reward.

### Core Principles

#### Policy Gradient Approach

- PPO operates on the policy gradient approach, where the agent directly learns a policy, typically parameterized by a neural network. The policy maps states to actions based on the current understanding of the environment.

#### Actor-Critic Framework

- PPO is based on the actor-critic framework, which means it simultaneously trains two components:
    - **Actor (Policy Network)**: Selects actions based on the current policy.
    - **Critic (Value Function Network)**: Evaluates these actions by estimating the expected the return of each state, i.e., the value of the state-action pairs.
- This dual approach allows PPO to efficiently balance exploration and exploitation by guiding the actor’s policy updates using feedback from the critic. The critic helps compute the advantage function, which quantifies the quality of the actions taken, enabling more informed updates to the policy.

##### The Actor (Policy Network)

- The actor network (πθπθ) is responsible for selecting actions based on the current policy:
    
    πθ(at∣st)=P(at∣st;θ)πθ(at∣st)=P(at∣st;θ)
    
    - where θθ represents the learnable parameters of the policy network.
- Unlike the critic, which estimates the expected return of a given state, the actor directly determines the probability distribution over possible actions. This allows the agent to explore different responses while refining its behavior over time.
    
- The actor is updated using a clipped surrogate objective function to ensure stable policy improvements:
    
    L(θ)=𝔼t[min(rt(θ)At,clip(rt(θ),1−ϵ,1+ϵ)At)]L(θ)=Et[min(rt(θ)At,clip(rt(θ),1−ϵ,1+ϵ)At)]
    
    - where:
        - rt(θ)=πθ(at‖st)πθold(at‖st)rt(θ)=πθ(at‖st)πθold(at‖st) is the probability ratio between the new and old policies.
        - AtAt is the advantage function guiding policy updates.
        - ϵϵ is a hyperparameter that constrains policy updates to prevent drastic changes.
- This clipping mechanism prevents excessively large updates, mitigating instability and ensuring smooth learning.
    
- The actor continually adapts by maximizing this objective, leading to more effective and stable policy learning while being guided by the critic’s evaluation of expected returns.
    

##### The Critic (Value Function)

- The critic network (VγVγ) is trained to predict the final reward from a partial response:
    
    L(γ)=𝔼t[(Vγ(st)−sg(Rϕ(sT)))2]L(γ)=Et[(Vγ(st)−sg(Rϕ(sT)))2]
    
    - where sgsg is the stop-gradient operation.
- The critic learns alongside the policy, ensuring it stays aligned with the current model.
    

### Stages

- The PPO workflow contains five main stages for iterative policy improvement:
    1. **Generate responses:** LLM produces multiple responses for a given prompt
    2. **Score responses:** The reward model assigns reward for each response
    3. **Compute advantages:** Use GAE to compute advantages
    4. **Optimize policy:** Update the LLM by optimizing the total objective
    5. **Update critic:** Train the value function to be better at predicting the rewards given partial responses

### Generalized Advantage Estimation (GAE)

### Generalized Advantage Estimation (GAE)

- PPO uses Generalized Advantage Estimation (GAE) to compute advantages, which defines how much better a specific action atat is compared to an average action the policy will take in state stst.
- GAE plays a crucial role in PPO by providing a flexible, variance-reduced estimator of the advantage function, enabling more stable and sample-efficient policy optimization.

#### Formal Definition

At=Q(st,at)−V(st)At=Q(st,at)−V(st)

- where:
    - Q(st,at)Q(st,at) is the expected cumulative reward of taking a specific action atat in state stst
    - V(st)V(st) is the expected cumulative reward of the average action the policy takes in state stst

#### Advantage Estimation Approaches

- There are two main approaches to estimating advantage:
    
    - **Monte-Carlo (MC):**
        - Uses the reward of the full trajectory (full responses)
        - High variance due to sparse reward
        - Low bias as we can accurately model the reward
    - **Temporal Difference (TD):**
        - Uses one-step trajectory reward
        - Significantly reduces variance
        - Higher bias as we can’t as accurately anticipate final reward

#### GAE Formula and Bias-Variance Trade-off

- GAE balances bias and variance through multi-step TD:
    
    AGAEK=∑t=0K−1(λ)tδtAKGAE=∑t=0K−1(λ)tδt
    
    - where:
        - KK denotes the number of TD steps (K<TK<T)
        - δtδt denotes the TD error at step tt: δt=rt+γV(st+1)−V(st)δt=rt+γV(st+1)−V(st)
        - The hyperparameter λλ controls the trade-off:
            - λ=0λ=0 →→ Pure TD learning (low variance, high bias)
            - λ=1λ=1 →→ Pure Monte Carlo (high variance, low bias)
- In practice, PPO uses a truncated version of GAE, where the advantage estimate over a trajectory segment of length TT is computed as:
    
    Â t=δt+(γλ)δt+1+⋯+(γλ)T−t+1δT−1A^t=δt+(γλ)δt+1+⋯+(γλ)T−t+1δT−1
    
    - where δt=rt+γV(st+1)−V(st)δt=rt+γV(st+1)−V(st)
- This formulation allows PPO to effectively trade off bias and variance by adjusting λλ, which is typically set between 0.9 and 0.97.
    

#### Role in PPO’s Clipped Surrogate Objective

- This advantage estimate Â tA^t is a critical component of PPO’s clipped surrogate objective, which is used to update the policy:
    
    LCLIP(θ)=𝔼t[min(rt(θ)Â t,clip(rt(θ),1−ϵ,1+ϵ)Â t)]LCLIP(θ)=Et[min(rt(θ)A^t,clip(rt(θ),1−ϵ,1+ϵ)A^t)]
    
    - where:
        - |   |   |   |
            |---|---|---|
            |$$r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}istheratiooftheprobabilityofactionistheratiooftheprobabilityofactiona_t$$ under the new and old policies|
            
        - ϵϵ is a hyperparameter (e.g., 0.2) that limits the deviation from the old policy
- The advantage Â tA^t modulates how much the policy is updated: if the advantage is positive, the update favors increasing the probability of the action; if negative, the update discourages it. Clipping ensures the update is conservative and prevents excessive deviation from the current policy.
    

#### Value Function and Critic Role

- The value function V(st)V(st), which is used in both computing δtδt and as a critic during training, is learned using a regression loss:

LVFt(θ)=(Vθ(st)−Vtargett)2LtVF(θ)=(Vθ(st)−Vttarget)2

- PPO combines the policy loss, value loss, and an entropy bonus (to encourage exploration) into a total loss function:

LCLIP+VF+St(θ)=𝔼t[LCLIPt(θ)−c1LVFt(θ)+c2S[πθ](st)]LtCLIP+VF+S(θ)=Et[LtCLIP(θ)−c1LtVF(θ)+c2S[πθ](st)]

- where:
    - c1c1 and c2c2 are coefficients
    - S[πθ](st)S[πθ](st) is the entropy of the policy at state stst

#### Reward and Value Model Roles

- The reward signal used in PPO in classic reinforcement learning tasks like robotic control or Atari games is typically the raw reward provided by the environment. In , this could be a numerical score or some environment-defined signal that reflects success (e.g., distance walked, enemies defeated, etc.).
- PPO uses this reward to compute the temporal difference error δtδt, which is then used to calculate the advantage estimate Â tA^t. The reward, therefore, directly influences how the policy updates toward favoring higher-value actions.
- In the context of RLHF applied to LLMs, the situation changes because environments like natural language do not inherently provide a structured, numerical reward signal. Instead, we use a learned reward model trained on human preferences.
    - Here’s how it works:
        - Human labelers are shown pairs of model-generated responses and asked to choose which one they prefer.
        - These comparisons are used to train a reward model that maps an LLM response (conditioned on a prompt) to a scalar reward, indicating how “good” or “aligned” the response is with human preferences.
        - This reward model replaces the environment’s raw reward and acts as the reward function in PPO.
- When using PPO in RLHF:
    - The LLM generates a response to a prompt (this is the action).
    - The reward model assigns a scalar reward to the response.
    - This scalar is treated as rtrt in the PPO pipeline.
    - The value model (critic) still estimates V(st)V(st), typically as the expected reward for a given prompt.
    - GAE is used to compute the advantage Â tA^t, guiding the policy update so the model improves toward generating more reward-aligned responses.
- So while the PPO algorithm itself remains the same, the source of the reward changes:
    - In environments like MuJoCo or Atari: reward is native to the environment.
    - In RLHF for LLMs: reward is generated by a separate reward model trained to reflect human judgment.
- This adaptation is key to making PPO applicable in NLP settings, where explicit reinforcement signals are absent and have to be approximated using human feedback.

### Key Components

#### Optimal Policy and Reference Policy

1. **Optimal Policy (π∗π∗ or πoptimalπoptimal):** The optimal policy refers to the strategy or set of rules that the LLM follows to maximizing the objective function J(π)J(π). This objective function is designed to reflect the goals of alignment, such as generating helpful, truthful, and harmless responses. Formally, the optimal policy π∗π∗ is defined as:
    
    π∗=argmaxπJ(π)π∗=arg⁡maxπJ(π)
    
    - where J(π)J(π) is the objective function.
2. **Reference Policy (πrefπref):** The reference policy is a baseline or benchmark policy used to compare and guide the learning process of the optimal policy. It represents a known, stable policy that the model starts from or refers back to during training. The reference policy helps in stabilizing the training process by providing a consistent comparison point.
    

##### Summary

- πoptimalπoptimal: Optimal policy, maximizing the objective function J(π)J(π).
- πrefπref: Reference policy, providing a stable baseline for training.

#### Surrogate Objective Function

- Central to PPO is its surrogate objective function, which considers the (i) policy ratio and (ii) advantage function, as explained below.
    
- In the context of LLMs, the state corresponds to the input prompt along with the tokens generated so far (i.e., the context), and the action refers to the next token the model chooses to generate. That is:
    - **State ss**: The input question qq and previously generated tokens o<to<t
    - **Action aa**: The next token otot
- The “policy ratio”, also known as the “likelihood ratio” or “probability ratio” or “importance sampling ratio”, is the ratio of the probability of an action under the new (i.e., current) policy to the old (i.e., reference or behavior) policy. This ratio helps align the training of the current model with the data sampled from an earlier version of the policy.
    
- Mathematically, the general form of the policy ratio is: r(θ)=πθ(a∣s)πθold(a∣s)r(θ)=πθ(a∣s)πθold(a∣s)
    
- In the LLM setting, this becomes: rt(θ)=πθ(ot∣q,o<t)πold(ot∣q,o<t)rt(θ)=πθ(ot∣q,o<t)πold(ot∣q,o<t) where:
    - πθπθ is the current policy (i.e., the model being updated),
    - πoldπold is the policy that was used to generate the training data,
    - otot is the token being predicted at time step tt,
    - qq is the question or initial input,
    - o<to<t is the sequence of previously generated tokens.
- This ratio tells us how much more or less likely the current model is to generate a token compared to the old one. It’s used to reweight updates to the policy to account for the fact that training data was collected under a different policy - hence, called the “importance sampling” ratio.
    
- In PPO, this ratio is clipped within a certain range (e.g., [1−ϵ,1+ϵ][1−ϵ,1+ϵ]) to prevent large, destabilizing updates. This makes the training more robust when the current policy starts to diverge from the old one.
    
- The policy ratio is multiplied by the advantage function, which measures how much better a specific action is compared to the average action at that state. In PPO, this advantage is estimated using techniques like Generalized Advantage Estimation (GAE) and relies on a separately trained value function (critic). In contrast, GRPO simplifies this by estimating the advantage from relative group rewards, avoiding the need for a value model.
    
- A detailed discourse on this has been offered in the section on [PPO’s Objective Function: Clipped Surrogate Loss](https://aman.ai/primers/ai/llm-alignment/#ppos-objective-function-clipped-surrogate-loss).

#### Clipping Mechanism

- PPO clips/limits the policy ratio in its objective function within a defined range (typically [1−ϵ,1+ϵ][1−ϵ,1+ϵ]), ensuring controlled updates. This clipping ensures that the updates to the policy are kept within a reasonable range, preventing the new policy from deviating excessively from the reference one. Ultimately, this mechanism helps in maintaining the stability of the learning process.

#### Data Re-use Over Multiple Epochs of Stochastic Gradient Ascent

- PPO uses each batch of experiences for multiple epochs of stochastic gradient ascent to update the policy, improving sample efficiency compared to some other methods.

#### Value Function and Baseline

- PPO trains a value function (the critic) is trained alongside the policy (the actor) to estimate state values. The value function estimates the expected return (cumulative future rewards) from each state and is used to compute the advantage function, which in turn informs the policy update.
- The baseline provided by the critic stabilizes the training process by reducing variance in the policy gradients, helping the actor make more precise updates.

### PPO’s Objective Function: Clipped Surrogate Loss

#### Intuition

- The surrogate loss in PPO is defined based on the ratio of the probability of taking an action under the current policy to the probability of taking the same action under the reference policy.
- This ratio is used to adjust the policy towards actions that have higher rewards while ensuring that updates are not too drastic. The clipping mechanism is employed to limit the magnitude of these updates, maintaining stability during training.

> Note that in conventional deep learning, loss functions are typically minimized to reduce prediction error, while in reinforcement learning, objective functions are usually maximized to increase expected reward or policy performance. Specifically, in policy optimization (say, with PPO) the objective function is maximized, as it aims to improve the policy by increasing the expected reward under a surrogate objective.

#### Components

- PPO’s clipped surrogate objective function has the following components:
    
    1. **Policy Ratio:** The core of the PPO objective function involves the policy ratio, which is the ratio of the probability of taking a certain action under the current policy to the probability under the reference policy. This ratio is multiplied by the advantage estimate, which reflects how much better a given action is compared to the average action at a given state.
    2. **Clipped Surrogate Objective:** To prevent excessively large updates, which could destabilize training, PPO introduces a clipping mechanism in its objective function. The policy ratio is clipped within a certain range, typically [1−ϵ,1+ϵ][1−ϵ,1+ϵ] (where ϵϵ is a small value like 0.1 or 0.2). This clipping ensures that the updates to the policy are not too large, which maintains stability in training.
        
        - Formally: Lclip(θ)=𝔼t[min(ct(πθ)AGAEt,clip(ct(πθ),1−ϵ,1+ϵ)AGAEt)]Lclip(θ)=Et[min(ct(πθ)AtGAE,clip(ct(πθ),1−ϵ,1+ϵ)AtGAE)]
            - where:
        
        - Lclip(θ)Lclip(θ):
            - The clipped surrogate loss in PPO, which balances policy updates by preventing excessively large changes to the policy.
            - This function ensures that the new policy does not deviate too far from the old policy, maintaining stable training.
        - 𝔼tEt:
            - Expectation over all time steps tt, averaging the objective function across multiple training samples.
        - ct(πθ)ct(πθ):
            - The probability ratio that compares the new policy to the old policy, given by: ct(πθ)=πθ(at∣st)πθold(at∣st)ct(πθ)=πθ(at∣st)πθold(at∣st)
            - If ct(πθ)>1ct(πθ)>1, the action is more likely under the new policy.
            - If ct(πθ)<1ct(πθ)<1, the action is less likely under the new policy.
        - AGAEtAtGAE:
            - The advantage function computed using Generalized Advantage Estimation (GAE).
            - Measures how much better (or worse) an action atat is compared to the policy’s average action at state stst.
            - A positive AGAEtAtGAE encourages increasing the probability of the action, while a negative AGAEtAtGAE discourages it.
        - clip(ct(πθ),1−ϵ,1+ϵ)clip(ct(πθ),1−ϵ,1+ϵ):
            - The clipping function, which limits ct(πθ)ct(πθ) within the range [1−ϵ,1+ϵ][1−ϵ,1+ϵ].
            - This ensures that updates to the policy do not drastically change the probability of taking a certain action.
        - min(ct(πθ)AGAEt,clip(ct(πθ),1−ϵ,1+ϵ)AGAEt)min(ct(πθ)AtGAE,clip(ct(πθ),1−ϵ,1+ϵ)AtGAE):
            - The core of the clipped loss function:
                - If ct(πθ)AGAEtct(πθ)AtGAE is too large, the function selects the clipped version.
                - If it is within the safe range, it behaves as a standard policy gradient update.
            - This prevents over-aggressive policy updates, stabilizing learning.
    3. **KL Divergence Loss:** Besides the clipped objective, another common component in the loss function is to add a KL divergence penalty to the objective function. This means the algorithm would penalize the objective based on how much the new policy diverges from the reference policy. In other words, the KL divergence component prevents overconfident policy updates by keeping the new policy close to the reference one by penalizing updates that result in a large divergence from the reference policy.
        
        - The KL divergence loss is typically added to the objective function as a penalty term: LKL(θ)=𝔼[LPPO(θ)−βKL[πold||πθ]]LKL(θ)=E[LPPO(θ)−βKL[πold||πθ]]
            - where:
        
        - ββ is a hyperparameter that controls the strength of the KL penalty.
    4. **Value Function Loss:** PPO also typically includes a value function loss in its objective. This part of the objective function ensures that the estimated value of the states (as predicted by the value function) is as accurate as possible, which is important for computing reliable advantage estimates.
    5. **Entropy Bonus:** Some implementations of PPO include an entropy bonus to encourage exploration by penalizing low entropy (overly confident) policies. This part of the objective function rewards the policy for taking a variety of actions, which helps prevent premature convergence to suboptimal policies. Formally: H(θ)=−𝔼at[logπθ(at∣st)]H(θ)=−Eat[log⁡πθ(at∣st)]
        
        - where:
        
        - H(θ)H(θ): The entropy of the policy πθπθ, which measures the uncertainty or diversity of the actions selected by the policy.
        - 𝔼atEat (Expectation over atat): The expectation is taken over all possible actions atat that could be chosen by the policy at a given state stst.
        - πθ(at∣st)πθ(at∣st): The probability assigned by the policy πθπθ to taking action atat when in state stst.
        - logπθ(at∣st)log⁡πθ(at∣st): The log-probability of selecting action atat. This helps measure how certain the policy is about choosing atat.
        - Negative sign (−−): Since log-probabilities are typically negative (as probabilities are between 0 and 1), the negative sign ensures entropy is positive. Higher entropy corresponds to more randomness in the policy, while lower entropy corresponds to more deterministic behavior.

##### Purpose of the Clipping Mechanism

- The clipping mechanism is central to the stability and reliability of PPO. It ensures that the policy updates do not result in excessively large changes, which could destabilize the learning process. The clipping mechanism works as follows:
    
    - **Clipping Range:** The ratio r(θ)r(θ) is clipped to the range [1−ϵ,1+ϵ][1−ϵ,1+ϵ]. This means if the ratio r(θ)r(θ) is outside this range, it is set to the nearest bound.
    - **Objective Function Impact:** By clipping the probability ratio, PPO ensures that the change in policy induced by each update is kept within a reasonable range. This prevents the new policy from deviating too far from the reference policy, which could lead to instability and poor performance.
    - **Practical Example:** If the probability ratio r(θ)r(θ) is 1.2 and ϵϵ is 0.2, the clipped ratio would remain 1.2. However, if r(θ)r(θ) is 1.4, it would be clipped to 1.2 (1 + 0.2), and if r(θ)r(θ) is 0.7, it would be clipped to 0.8 (1 - 0.2).

##### Purpose of Surrogate Loss

- The surrogate loss allows PPO to balance the need for policy improvement with the necessity of maintaining stability. By limiting the extent to which the policy can change at each update, the surrogate loss ensures that the learning process remains stable and avoids the pitfalls of overly aggressive updates. The clipping mechanism is a key innovation that helps PPO maintain this balance effectively. This approach helps PPO to achieve a good balance between effective policy learning and the stability required for reliable performance in various environments.

#### Mathematical Formulation

- Putting all the aforementioned components together and combining multiple terms, the complete PPO objective can be written as:
    
    LPPO(θ,γ)=Lclip(θ)⏟Maximize reward+w1H(θ)⏟Maximize entropy−w2KL(θ)Penalize KL divergenceLPPO(θ,γ)=Lclip(θ)⏟Maximize reward+w1H(θ)⏟Maximize entropy−w2KL(θ)⏟Penalize KL divergence
    
    - where:
        - **Clipped Surrogate Objective:** Lclip(θ)=𝔼t[min(ct(πθ)AGAEt,clip(ct(πθ),1−ϵ,1+ϵ)AGAEt)]Lclip(θ)=Et[min(ct(πθ)AtGAE,clip(ct(πθ),1−ϵ,1+ϵ)AtGAE)]
        - **KL Divergence:** KL(θ)=𝔼st[𝔻KL(πθorig(⋅∣st)||πθ(⋅∣st))]KL(θ)=Est[DKL(πθorig(⋅∣st)||πθ(⋅∣st))]
        - **Entropy Bonus:** H(θ)=−𝔼at[logπθ(at∣st)]H(θ)=−Eat[log⁡πθ(at∣st)]
- The PPO surrogate loss is then defined as follows:
    
    LPPO-CLIP(θ)=𝔼[min(r(θ)Â ,clip(r(θ),1−ϵ,1+ϵ)Â )]LPPO-CLIP(θ)=E[min(r(θ)A^,clip(r(θ),1−ϵ,1+ϵ)A^)]
    
    - where:
        - Â A^ is the advantage function, which measures how much better an action is compared to the average action at a given state. It is often estimated using Generalized Advantage Estimation (GAE).
        - ϵϵ is a hyperparameter that defines the clipping range, controlling how much the policy can change at each update. Typical values are in the range of 0.1 to 0.3.
        - clip(r(θ),1−ϵ,1+ϵ)clip(r(θ),1−ϵ,1+ϵ) clips the ratio r(θ)r(θ) to be within the range [1−ϵ,1+ϵ][1−ϵ,1+ϵ].

##### PPO Loss with Clipped Surrogate Loss

- Let πθπθ be the current policy parameterized by θθ, and πoldπold be the old policy. For a given state ss and action aa, the probability ratio is:

r(θ)=πθ(a|s)πold(a|s)r(θ)=πθ(a|s)πold(a|s)

- The expanded form of the PPO clipped surrogate loss obtained by plugging in the policy likelihood ratio can be written as:
    
    LPPO-CLIP(π)=𝔼[min(π(a|s)πold(a|s)Â ,clip(π(a|s)πold(a|s),1−ϵ,1+ϵ)Â )]LPPO-CLIP(π)=E[min(π(a|s)πold(a|s)A^,clip(π(a|s)πold(a|s),1−ϵ,1+ϵ)A^)]
    
    - where:
        - Â A^ is the advantage estimate, which measures how much better an action is compared to the average action at a given state. It is often estimated using Generalized Advantage Estimation (GAE).
        - ss is the state.
        - aa is the action.
        - ϵϵ is a small hyperparameter that limits the extent of the policy update.

##### PPO Loss with KL Divergence

- An alternative to the clipped surrogate objective is to use a KL-penalized objective, where a penalty term based on the KL divergence between the current policy and the old policy is added to the loss. The penalty coefficient ββ is adaptively tuned to maintain a target KL divergence dtargdtarg. After each policy update, the actual KL divergence dd is measured. If d<dtarg/1.5d<dtarg/1.5, the penalty coefficient is reduced (i.e., β←β/2β←β/2) to allow more flexibility in updates. If d>1.5⋅dtargd>1.5⋅dtarg, ββ is increased (i.e., β←β⋅2β←β⋅2) to constrain the update more tightly. This approach helps keep the updated policy close to the previous one while still allowing learning progress. The KL-penalized loss is defined as:
    
    LKLPEN(θ)=𝔼̂ t[πθ(at|st)πθold(at|st)Â t−β∑aπθold(a|st)log(πθold(a|st)πθ(a|st))]LKLPEN(θ)=E^t[πθ(at|st)πθold(at|st)A^t−β∑aπθold(a|st)log⁡(πθold(a|st)πθ(a|st))]
    
    - where:
        - πθoldπθold is the policy before the update.
        - πθπθ is the current policy.
        - Â tA^t is the estimated advantage.
        - ββ is the KL penalty coefficient adjusted dynamically to match the KL target.

##### PPO Loss with Clipped Surrogate Loss and KL Penalty

- The PPO paper also suggests that the KL penalty can be used in combination with the clipped surrogate objective. In this hybrid approach, the clipped objective controls the size of the policy update explicitly, while the KL penalty provides an additional regularization signal to discourage large divergences from the previous policy. Although this combined objective performed slightly worse than clipping alone in the paper’s experiments, it is included as an important baseline:
    
    LCLIP+KLPEN(θ)=𝔼̂ t[min(rt(θ)Â t,clip(rt(θ),1−ϵ,1+ϵ)Â t)−β∑aπθold(a|st)log(πθold(a|st)πθ(a|st))]LCLIP+KLPEN(θ)=E^t[min(rt(θ)A^t,clip(rt(θ),1−ϵ,1+ϵ)A^t)−β∑aπθold(a|st)log⁡(πθold(a|st)πθ(a|st))]
    
    - where:
        - The first term is the standard PPO clipped surrogate objective.
        - The second term adds a KL divergence penalty between the old and new policies.
        - ββ is the dynamically adjusted penalty coefficient.

### PPO for LLM Policy Optimization

- PPO plays a crucial role in performing policy optimization LLMs using RLHF.

#### RLHF Overview

- LLMs like GPT-4, ChatGPT, and Claude are optimized using RLHF, which consists of:
    1. **Supervised Fine-Tuning:** Train an initial model on human-annotated data.
    2. **Reward Model (RM) Training:** Train a model to predict human preference scores.
    3. **PPO Fine-Tuning:** Use the reward model to guide LLM responses through PPO.

#### PPO in LLM Training

- The policy is the LLM, which generates responses given a prompt.
- The reward model provides feedback, helping optimize the policy.
- PPO ensures controlled updates, preventing divergence from the supervised baseline.

### Practical Implementation of PPO

#### Pseudocode for PPO

![](https://aman.ai/images/copy.png)

`for iteration in range(num_iterations):     for actor in parallel_envs:         collect trajectories using current policy          compute advantage estimates using GAE          for epoch in range(num_epochs):         for minibatch in shuffled_batches:             compute PPO loss (clipped surrogate)             update policy with gradient descent`

#### PPO with OpenAI’s `transformers` and `trl`

![](https://aman.ai/images/copy.png)

`from trl import PPOTrainer  ppo_trainer = PPOTrainer(policy, optimizer, reward_model) for batch in dataloader:     query_tensors = tokenizer(batch["query"])     response_tensors = model.generate(query_tensors)     rewards = reward_model(response_tensors)     ppo_trainer.step(query_tensors, response_tensors, rewards)`

### Typical Hyperparameters

- **Clip Range (ϵϵ)**: 0.1 - 0.3
- **Learning Rate**: 10−510−5 to 10−410−4
- **Batch Size**: 32 - 512
- **GAE Lambda (λλ)**: 0.95
- **Entropy Coefficient**: 0.01 (for exploration)

### Variants of PPO

- There are two main variants of PPO: (i) PPO-Clip and (ii) PPO-Penalty.

### PPO-Clip

- Uses the clipped surrogate objective function to limit the policy updates.
- The most commonly used version of PPO.

### PPO-Penalty

- Adds a KL-divergence penalty to the objective function to constrain policy updates.
- Used in cases where explicit divergence constraints are needed.

### Advantages of PPO

- **Stability and Reliability:** The clipping mechanism in the objective function helps to avoid large, destabilizing updates to the policy, making the learning process more stable and reliable.
- **Sample Efficiency:** By reusing data for multiple gradient updates, PPO can be more sample-efficient compared to some other methods.
- **General Applicability:** PPO has demonstrated good performance across a wide range of environments, from simple control tasks to complex simulations like those in 3D simulations. It offers a simpler and more robust approach compared to previous algorithms like TRPO.

### Simplified Example

- Imagine an agent learning to play a game. The agent tries different actions (moves in the game) and learns a policy that predicts which action to take in each state (situation in the game). The policy is updated based on the experiences, but instead of drastically changing the policy based on recent success or failure, PPO makes smaller, incremental changes. This way, the agent avoids drastically changing its strategy based on limited new information, leading to a more stable and consistent learning process.

### Summary

- PPO stands out in the realm of RL for its innovative approach to policy updates via gradient ascent. Its key innovation is the introduction of a clipped surrogate objective function that judiciously constrains the policy ratio. This mechanism is fundamental in preventing drastic policy shifts and ensuring a smoother, more stable learning progression.
- PPO is particularly favored for its effectiveness and simplicity across diverse environments, striking a fine balance between policy improvement and stability.
- The PPO objective function is designed to balance the need for effective policy improvement with the need for training stability. It achieves this through the use of a clipped surrogate objective function, value function loss, and potentially an entropy bonus.
- While KL divergence is not a direct part of the basic PPO objective function, it is often used in the PPO-Penalty implementation of PPO to monitor and maintain policy stability. This is done either by penalizing large changes in the policy (KL penalty) or by enforcing a constraint on the extent of change allowed between policy updates (KL constraint).
- By integrating these elements, PPO provides a robust framework for RL, ensuring both stability and efficiency in the learning process. This makes it particularly suitable for fine-tuning large language models (LLMs) and other complex systems where stable and reliable updates are crucial.

### Related: How is the Policy Represented As a Neural Network?

- In PPO and other RL (RL) algorithms, the policy is typically represented by a parameterized function, most commonly a neural network. Here’s a detailed breakdown of how the policy is represented and what it entails:

### Policy Representation in RL Algorithms

1. **Neural Network (Parameterized Function)**
    - **Neural Networks:** In modern RL algorithms like PPO, the policy is most often represented by a neural network. The neural network takes the current state of the environment as input and outputs a probability distribution over possible actions.
    - **Parameters (Weights):** The neural network is defined by its parameters, which are the weights and biases of the network. These parameters are collectively denoted as θθ. The process of training the policy involves adjusting these parameters to maximize the expected reward.
2. **Mathematical Representation**
    - The policy πθ(a‖s)πθ(a‖s) represents the probability of taking action aa given state ss, parameterized by θθ. This function maps states to a distribution over actions.
    - **Discrete Action Spaces:** For discrete action spaces, the output of the neural network can be a softmax function that gives a probability for each possible action.
    - **Continuous Action Spaces:** For continuous action spaces, the output might be parameters of a probability distribution (e.g., mean and standard deviation of a Gaussian distribution) from which actions can be sampled.
3. **Policy Gradient Methods**
    - In policy gradient methods like PPO, the policy is directly updated by computing the gradient of the expected reward with respect to the policy parameters θθ. This gradient is used to adjust the parameters in a way that increases the expected reward.
4. **Actor-Critic Methods**
    - **Actor:** In actor-critic methods, the “actor” is the policy network, which decides the actions to take.
    - **Critic:** The “critic” is another network that estimates the value function, which provides feedback on how good the current policy is. The critic helps to reduce the variance of the policy gradient estimates.
5. **Optimization Process**
    - **Policy Update:** The policy parameters θθ are updated through an optimization process (e.g., gradient ascent in policy gradient methods) to maximize the objective function, such as the expected cumulative reward.
    - **Surrogate Objective:** In PPO, a surrogate objective function is used, which includes mechanisms like clipping to ensure stable updates to the policy.

#### Summary

- **Neural Network:** The policy in PPO and many other RL algorithms is represented by a neural network.
- **Parameters (Weights):** The neural network is parameterized by a set of weights and biases, collectively denoted as θθ.
- **Probability Distribution:** The policy maps states to a probability distribution over actions, allowing for both discrete and continuous action spaces.
- **Optimization:** The policy parameters are updated iteratively to maximize the expected reward, often using gradient-based optimization methods.
    
- By representing the policy as a neural network, RL algorithms can leverage the expressive power of deep learning to handle complex environments and high-dimensional state and action spaces.

## [RL with AI Feedback (RLAIF)](https://arxiv.org/abs/2309.00267)

- RLAIF uses AI-generated preferences instead of human annotated preferences. It leverages a powerful LLM (say, GPT-4) to generate these preferences, offering a cost-effective and efficient alternative to human-generated feedback.
- RLAIF operates by using a pre-trained LLMs to generate feedback for training another LLM. Essentially, the feedback-generating LLM serves as a stand-in for human annotators. This model evaluates and provides preferences or feedback on the outputs of the LLM being trained, guiding its learning process.
- The feedback is used to optimize the LLM’s performance for specific tasks like summarization or dialogue generation. This method enables efficient scaling of the training process while maintaining or improving the model’s performance compared to methods relying on human feedback.

## Direct Preference Optimization (DPO)

- LLMs acquire extensive world knowledge and reasoning skills via self-supervised pre-training, but precisely controlling their behavior is challenging due to their unsupervised training nature. Traditionally, methods like RLHF, discussed earlier in this article, are used to steer these models, involving two stages: training a reward model based on human preference labels and then fine-tuning the LM to align with these preferences using RL (RL). However, RLHF presents complexities and instability issues, necessitating fitting a reward model and then training a policy to optimize this reward, which is prone to stability concerns.
- Proposed in [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290) by Rafailov et al. from Stanford in 2023, Direct Preference Optimization (DPO) is a novel approach that simplifies and enhances the aforementioned process. DPO leverages a mathematical relationship between optimal policies and reward functions, demonstrating that the constrained reward maximization problem in RLHF can be optimized more effectively with a single stage of policy training. DPO redefines the RLHF objective by showing that the reward can be rewritten purely as a function of policy probabilities, allowing the LM to implicitly define both the policy and the reward function. This innovation eliminates the need for a separate reward model and the complexities of RL.
- This paper introduces a novel algorithm that gets rid of the two stages of RL, namely - fitting a reward model, and training a policy to optimize the reward via sampling. The second stage is particularly hard to get right due to stability concerns, which DPO obliterates. The way it works is, given a dataset of the form `<prompt, worse completion, better completion>`, you train your LLM using a new loss function which essentially encourages it to increase the likelihood of the better completion and decrease the likelihood of the worse completion, weighted by how much higher the implicit reward model. This method obviates the need for an explicit reward model, as the LLM itself acts as a reward model. The key advantage is that it’s a straightforward loss function optimized using backpropagation.
- The stability, performance, and computational efficiency of DPO are significant improvements over traditional methods. It eliminates the need for sampling from the LM during fine-tuning, fitting a separate reward model, or extensive hyperparameter tuning.
- The figure below from the paper illustrates that DPO optimizes for human preferences while avoiding RL. Existing methods for fine-tuning language models with human feedback first fit a reward model to a dataset of prompts and human preferences over pairs of responses, and then use RL to find a policy that maximizes the learned reward. In contrast, DPO directly optimizes for the policy best satisfying the preferences with a simple classification objective, without an explicit reward function or RL.

![](https://aman.ai/images/papers/DPO.jpg)

- Experiments demonstrate that DPO can fine-tune LMs to align with human preferences as effectively, if not more so, than traditional RLHF methods. It notably surpasses RLHF in controlling the sentiment of generations and enhances response quality in tasks like summarization and single-turn dialogue. Its implementation and training processes are substantially simpler.
- In summary, DPO aligns models by optimizing pairs of responses ranked by human feedback, assigning a higher likelihood to preferred responses over less preferred ones. This preference-based learning captures human intent without relying on the complexity of RL traditionally used in fine-tuning methods. Instead, DPO transforms the reward maximization problem into a simpler classification task, directly optimizing model outputs based on human preferences.

### DPO’s Binary Cross-Entropy Loss

- DPO works by utilizing Binary Cross-Entropy (BCE) to compare pairs of model-generated responses (preferred and dispreferred) against human preferences. The model generates two responses for each input, and human annotators indicate which response they prefer. The model then assigns probabilities to each response. The BCE loss function computes the difference between these model-assigned probabilities and the actual human preferences, penalizing the model when it assigns a higher probability to the dispreferred response. By minimizing this loss, DPO adjusts the model’s internal parameters to better align with human preferences.
- Put simply, DPO represents a shift in training language models to align with human preferences by consolidating the RLHF process into a single, end-to-end optimization step. By adapting the binary cross-entropy loss, DPO directly optimizes model behavior by adjusting log probabilities based on human feedback, making it a computationally efficient and theoretically grounded method for preference-based learning.

#### Simplified Process

1. **Response Pairs**: For each input, the model generates two responses.
2. **Human Preferences**: Humans indicate which response is preferable.
3. **Model Probabilities**: The model assigns probabilities to each response.
4. **BCE Loss**: The loss function calculates the difference between the model’s predictions and human preferences, penalizing the model more when it assigns higher probabilities to dispreferred responses.

#### Loss Function Equation

- The DPO loss function, based on BCE, is formulated as:
    
    LDPO(πθ;πref)=−𝔼(x,yw,yl)∼D[logσ(βlogπθ(yw∣x)πref(yw∣x)−βlogπθ(yl∣x)πref(yl∣x))]LDPO(πθ;πref)=−E(x,yw,yl)∼D[log⁡σ(βlog⁡πθ(yw∣x)πref(yw∣x)−βlog⁡πθ(yl∣x)πref(yl∣x))]
    
    - where:
        - 𝔼(x,yw,yl)∼DE(x,yw,yl)∼D denotes the expectation over the dataset DD, which consists of tuples (x,yw,yl)(x,yw,yl) derived from human preference data. Here:
            - xx is the input context (e.g., a prompt or query).
            - ywyw is the preferred response, which is deemed better.
            - ylyl is the less preferred response.
        - πθπθ is the policy being optimized.
        - πrefπref is the reference policy (initial or base model).
        - ββ controls how much the model stays close to the reference policy.
        - σσ is the logistic/sigmoid function.
- This BCE-based loss function drives the model to increase the likelihood of preferred responses while penalizing dispreferred ones.
    

##### Loss Function Design Choices

###### Negative Sign in Front of the Loss

- The negative sign ensures that the optimization minimizes the negative log-likelihood, which aligns with maximizing the likelihood of predicting the preferred response correctly. This is standard in BCE loss formulations.

###### Why the Sigmoid Function (σσ) is Used

- The sigmoid function σ(z)=11+e−zσ(z)=11+e−z maps the input zz to a probability in the range [0, 1].
- In this case, it is applied to the log-ratio differences (scaled by ββ) between the preferred and less preferred responses. This ensures that the model output can be interpreted probabilistically, representing the confidence that the preferred response is indeed better.

###### Role of ββ in the DPO Loss Function

- The parameter ββ plays a critical role in balancing the optimization process by controlling the influence of the reference policy (πrefπref) on the model being optimized (πθπθ)
- It balances the dual goals of maximizing human preference alignment and retaining the desirable qualities of the reference policy.
- Proper tuning of ββ is critical for achieving the right trade-off between stability and preference optimization.
- The role of ββ in the DPO loss function can be summarized as follows:
    
    1. **Scale of Log-Probability Differences:**
        - The term ββ scales the difference in log-probabilities between the preferred (ywyw) and less preferred (ylyl) responses. A larger ββ amplifies the contrast between the two responses, making the model more sensitive to preference differences.
    2. **Regularization Strength:**
        - ββ acts as a regularization parameter, controlling how strongly the model πθπθ adheres to the reference policy πrefπref. Specifically:
            - **High ββ:** The model stays closer to the reference policy, limiting the divergence from the initial policy. This helps retain stability and prevents overfitting to noisy or extreme preferences in the dataset.
            - **Low ββ:** The model is allowed to diverge further from the reference policy, giving it more freedom to optimize for the preferences in the dataset. However, this increases the risk of overfitting or producing less generalizable responses.
    3. **Interpretation as a Trade-off:**
        - ββ provides a trade-off between preference alignment and policy regularization:
            - **Preference Alignment:** With lower values of ββ, the model prioritizes aligning with human preferences at the cost of potential instability or over-divergence.
            - **Policy Regularization:** Higher values of ββ ensure that the model evolves conservatively, maintaining generality and robustness while limiting alignment with preferences.

###### Significant of the DPO Loss

- The loss measures how well the model πθπθ aligns with human preferences, as encoded in the dataset DD.
- By using BCE, the objective becomes a comparison of logits (log probabilities) between the preferred (ywyw) and less preferred (ylyl) responses. Minimizing this loss drives the model to produce outputs that increasingly favor ywyw over ylyl while balancing regularization (ββ) to avoid over-divergence from the reference policy πrefπref.

##### Mapping from the Standard Binary Cross-Entropy Loss to the DPO Loss

###### Standard Binary Cross-Entropy Loss

- To recap, the Binary Cross-Entropy loss for a single prediction zz (where z=π(yw‖x)−π(yl‖x)z=π(yw‖x)−π(yl‖x)) and its target label t∈{0,1}t∈{0,1} is defined as:
    
    LBCE(z,t)=−[t⋅log(σ(z))+(1−t)⋅log(1−σ(z))]LBCE(z,t)=−[t⋅log⁡(σ(z))+(1−t)⋅log⁡(1−σ(z))]
    
    - where,
        - zz: The logit (unbounded real value) representing the model’s confidence in the preferred label.
        - σ(z)=11+e−zσ(z)=11+e−z: The sigmoid function maps the logit to a probability.
        - tt: The binary target label, where t=1t=1 if ywyw is the preferred label and t=0t=0 if ylyl is preferred.

###### Mapping BCE Loss to DPO Loss

- In the DPO framework:
    
    1. The target is implicitly encoded by the comparison of ywyw (preferred) and ylyl (less preferred). Effectively, t=1t=1 for ywyw.
    2. The logit zz is calculated as the difference in log-probabilities (scaled by ββ):
        
        z=βlogπθ(yw∣x)πref(yw∣x)−βlogπθ(yl∣x)πref(yl∣x)z=βlog⁡πθ(yw∣x)πref(yw∣x)−βlog⁡πθ(yl∣x)πref(yl∣x)
        
        - This difference represents the model’s confidence in ywyw being better than ylyl, adjusted for the divergence from the reference policy.
    3. Plugging zz into the BCE loss for t=1t=1, the equation becomes:
        
        LDPO=−log(σ(z))LDPO=−log⁡(σ(z))
        
    4. Expanding zz, we get:
        
        LDPO=−logσ(βlogπθ(yw∣x)πref(yw∣x)−βlogπθ(yl∣x)πref(yl∣x))LDPO=−log⁡σ(βlog⁡πθ(yw∣x)πref(yw∣x)−βlog⁡πθ(yl∣x)πref(yl∣x))
        

###### Intuition of the Mapping

- **Standard BCE Loss:** Compares logits zz against a binary target tt (1 for positive, 0 for negative) and penalizes predictions deviating from the target.
- **DPO Loss:** Adapts the BCE framework to pairwise preferences, where:
    - zz reflects the scaled log-ratio difference between ywyw and ylyl.
    - Implicitly assumes t=1t=1 (i.e., ywyw is the preferred response).
- By minimizing LDPOLDPO, the model learns to increase the scaled log-probability of ywyw relative to ylyl, aligning with human preferences while staying close to πrefπref.

#### Key Insights

- **DPO’s Efficiency**: DPO simplifies the traditional RLHF pipeline by unifying policy learning and reward modeling into a single, efficient process. Instead of requiring a two-stage process (learning a reward model and then optimizing with RL), DPO directly optimizes the policy using human preferences as implicit rewards.
- **Streamlined Approach**: By using BCE to treat preference optimization as a binary classification task, DPO minimizes complexity and computational overhead. The model learns to classify between preferred and dispreferred responses, adjusting its behavior accordingly.

### How Does DPO Generate Two Responses and Assign Probabilities to Them?

- In DPO, generating two responses and assigning probabilities to each response involves a nuanced process:
    
    1. **Generating Two Responses**:
        - The responses are typically generated using a supervised fine-tuned language model. This model, when given an input prompt, generates a set of potential responses.
        - These responses are often generated through sampling methods like varying temperature, using different [token sampling methods](https://aman.ai/primers/ai/token-sampling) such as top-pp, top-kk, beam search, etc., which can produce diverse outputs.
    2. **Assigning Probabilities**:
        - Language models indeed assign probabilities at the token level, predicting the likelihood of each possible next token given the previous tokens.
        - The probability of an entire response (sequence of tokens) is calculated as the product of the probabilities of individual tokens in that sequence, as per the model’s prediction.
        - For DPO, these probabilities are used to calculate the loss based on human preferences. The model is trained to increase the likelihood of the preferred response and decrease that of the less preferred one.
- Through this process, DPO leverages human feedback to preference-optimize the model, encouraging it to generate more human-aligned outputs.
    

### DPO and It’s Use of the Bradley-Terry Model

- **Overview of the Bradley-Terry Model**:
    
    - The Bradley-Terry model is a probability model used for pairwise comparisons. It assigns a score to each item (in this context, model outputs), and the probability that one item is preferred over another is a function of their respective scores. Formally, if item ii has a score sisi and item jj has a score sjsj, the probability P(i is preferred over j)P(i is preferred over j) is given by:
    
    P(i is preferred over j)=exp(si)exp(si)+exp(sj)P(i is preferred over j)=exp⁡(si)exp⁡(si)+exp⁡(sj)
    
- **Application in DPO for LLM Alignment**:
    1. **Data Collection**:
        - Human evaluators provide pairwise comparisons of model outputs. For example, given two responses from the LLM, the evaluator indicates which one is better according to specific criteria (e.g., relevance, coherence, correctness).
    2. **Modeling Preferences**:
        - The outputs of the LLM are treated as items in the Bradley-Terry model. Each output has an associated score reflecting its quality or alignment with human preferences.
    3. **Score Estimation**:
        - The scores sisi for each output are estimated using the observed preferences. If output ii is preferred over output jj in several comparisons, sisi will be higher than sjsj. The scores are typically estimated using maximum likelihood estimation (MLE) or other optimization techniques suited for the Bradley-Terry model.
    4. **Optimization**:
        - Once the scores are estimated, the LLM is fine-tuned to maximize the likelihood of generating outputs with higher scores. The objective is to adjust the model parameters so that the outputs align better with human preferences as captured by the Bradley-Terry model scores.
- **Detailed Steps in DPO**:
    1. **Generate Outputs**:
        - Generate multiple outputs for a given prompt using the LLM.
    2. **Pairwise Comparisons**:
        - Collect human feedback by asking evaluators to compare pairs of outputs and indicate which one is better.
    3. **Fit Bradley-Terry Model**:
        - Use the collected pairwise comparisons to fit the Bradley-Terry model and estimate the scores for each output.
    4. **Update LLM**:
        - Fine-tune the LLM using the estimated scores. The objective is to adjust the model parameters such that the likelihood of producing higher-scored (preferred) outputs is maximized. This step often involves gradient-based optimization techniques where the loss function incorporates the Bradley-Terry model probabilities. - By iteratively performing these steps, the LLM can be aligned more closely with human preferences, producing outputs that are more likely to be preferred by human evaluators.
- **Summary**:
    - The Bradley-Terry model plays a crucial role in the Direct Preference Optimization method by providing a statistical framework for modeling and estimating the preferences of different model outputs. This, in turn, guides the fine-tuning of the LLM to align its outputs with human preferences effectively.

#### How Does DPO Implicitly Use a Bradley-Terry Model (if It Does Not Explicitly Use a Reward Model)?

- DPO uses the Bradley-Terry model implicitly, even if it does not explicitly employ a traditional reward model. Here’s how this works:

##### Key Concepts in DPO Without an Explicit Reward Model

1. **Pairwise Comparisons**:
    - Human evaluators provide pairwise comparisons between outputs generated by the LLM. For example, given two outputs, the evaluator indicates which one is preferred.
2. **Logistic Likelihood**:
    - The Bradley-Terry model is essentially a logistic model used for pairwise comparisons. The core idea is to model the probability of one output being preferred over another based on their latent scores.

##### Implicit Use of Bradley-Terry Model

- Without training an explicit reward model, DPO leverages the principles behind the Bradley-Terry model in the following manner:

1. **Score Assignment through Logit Transformation**:
    - For each output generated by the LLM, assign a latent score. This score can be considered as the logit (log-odds) of the output being preferred.
    - Given two outputs, oioi and ojoj, with logits (latent scores) sisi and sjsj, the probability that oioi is preferred over ojoj follows the logistic function: P(oi is preferred over oj)=exp(si)exp(si)+exp(sj)P(oi is preferred over oj)=exp⁡(si)exp⁡(si)+exp⁡(sj)
2. **Optimization Objective**:
    - Construct a loss function based on the likelihood of observed preferences. If oioi is preferred over ojoj in the dataset, the corresponding likelihood component is: L=logP(oi is preferred over oj)=log(exp(si)exp(si)+exp(sj))L=log⁡P(oi is preferred over oj)=log⁡(exp⁡(si)exp⁡(si)+exp⁡(sj))
    - The overall objective is to maximize this likelihood across all pairwise comparisons provided by human evaluators.
3. **Gradient Descent for Fine-Tuning**:
    - Instead of explicitly training a separate reward model, the LLM is fine-tuned using gradients derived from the likelihood function directly.
    - During backpropagation, the gradients with respect to the LLM’s parameters are computed from the likelihood of the preferences, effectively pushing the model to produce outputs that align with higher preference scores.

##### Steps in DPO Without Explicit Reward Model

1. **Generate Outputs**:
    - Generate multiple outputs for a set of prompts using the LLM.
2. **Collect Pairwise Comparisons**:
    - Human evaluators compare pairs of outputs and indicate which one is preferred.
3. **Compute Preference Probabilities**:
    - Use the logistic model (akin to Bradley-Terry) to compute the probability of each output being preferred over another.
4. **Construct Likelihood and Optimize**:
    - Formulate the likelihood based on the observed preferences and optimize the LLM’s parameters to maximize this likelihood.

##### Practical Implementation

- **Training Loop**:
    - In each iteration, generate outputs, collect preferences, compute the logistic likelihood, and perform gradient descent to adjust the LLM parameters.
- **Loss Function**:
    - The loss function directly incorporates the Bradley-Terry model’s probabilities without needing an intermediate reward model: Loss=−∑(i,j)∈comparisonslog(exp(si)exp(si)+exp(sj))Loss=−∑(i,j)∈comparisonslog⁡(exp⁡(si)exp⁡(si)+exp⁡(sj))
- By optimizing this loss function, DPO ensures that the LLM’s outputs increasingly align with human preferences, implicitly using the Bradley-Terry model’s probabilistic framework without explicitly training a separate reward model. This direct approach simplifies the alignment process while leveraging the robust statistical foundation of the Bradley-Terry model.

### Video Tutorial

- [This](https://www.youtube.com/watch?v=hvGa5Mba4c8) video by [Umar Jamil](https://www.youtube.com/@umarjamilai) explains the DPO pipeline, by deriving it step by step while explaining all the inner workings.
- After briefly introducing the topic of AI alignment, the video reviews RL, a topic that is necessary to understand the reward model and its loss function. Next, it derives the loss function step-by-step of the reward model under the Bradley-Terry model of preferences, a derivation that is missing in the DPO paper.
- Using the Bradley-Terry model, it builds the loss of the DPO algorithm, not only explaining its math derivation, but also giving intuition on how it works.
- In the last part, it describes how to use the loss practically, that is, how to calculate the log probabilities using a Transformer model, by showing how it is implemented in the Hugging Face library.

### Summary

- RLHF is the most “dicey” part of LLM training and the one that needed the most art vs. science. DPO seeks to simplify that by removing RL out of the equation and not requiring a dedicated reward model (with the LLM serving as the reward model). The process it follows is as follows:
    1. Treat a foundational instruction tuned LLM as the reference LLM.
    2. Generate pairs of outputs (using say, different [token sampling/decoding](https://aman.ai/primers/ai/token-sampling) methods or temperature scaling) to the same prompt and have humans choose which one they like, leading to a dataset of human preferences/feedback.
    3. Add a linear layer to the LLM so that it outputs a scalar value, and tune this new model with a new loss function called DPO loss which is based on binary cross entropy loss (compute log-ratio of scalar outputs of the reference LLM and the one being tuned, multiply by a divergence parameter).
    4. Drop the last linear layer, and you have a fine tuned LLM on human feedback.

## Kahneman-Tversky Optimization (KTO)

- Proposed in [Human-Centered Loss Functions (HALOs)](https://github.com/ContextualAI/HALOs/blob/main/assets/report.pdf) by Ethayarajh et al. from Stanford and Contextual AI, Kahneman-Tversky Optimization (KTO) is a novel approach to aligning LLMs with human feedback.
- KTO is a human-centered loss function that directly maximizes the utility of language model generations instead of maximizing the log-likelihood of preferences as current methods do. This approach is named after Daniel Kahneman and Amos Tversky, who are known for their work in prospect theory, a theory of how humans make decisions about uncertain outcomes. KTO is based on the principles of prospect theory, a theory in behavioral economics. Unlike traditional methods, KTO focuses on maximizing the utility of LLM generations by aligning them with human feedback.
- KTO achieves the goal of generating desirable outputs by using a utility function to guide the training of a language model. This process involves several key steps:
    
    1. **Utility Function Definition**: A utility function is defined based on the principles of Kahneman-Tversky’s prospect theory. This function assigns a value to each possible output of the language model, indicating its desirability or utility from a human perspective. The utility values can be determined based on factors like relevance, coherence, or adherence to specific criteria.
        
    2. **Generating Outputs**: During training, the language model generates outputs based on given inputs. These outputs are complete sequences, such as sentences or paragraphs, rather than individual tokens.
        
    3. **Evaluating Outputs**: Each generated output is evaluated using the utility function. The utility score reflects how desirable or aligned the output is with human preferences or objectives.
        
    4. **Optimizing the Model**: The model’s parameters are updated to increase the likelihood of generating outputs with higher utility scores. The optimization process aims to maximize the expected utility of the outputs, essentially encouraging the model to produce more desirable results.
        
    5. **Iterative Training**: This process is iterative, with the model continually generating outputs, receiving utility evaluations, and updating its parameters. Over time, the model learns to produce outputs that are increasingly aligned with the utility function’s assessment of desirability.
        
- In essence, KTO shifts the focus from traditional training objectives, like next-token prediction or fitting to paired preference data, to directly optimizing for outputs that are considered valuable or desirable according to a utility-based framework. This approach can be particularly effective in applications where the quality of the output is subjective or where specific characteristics of the output are valued.
    
    1. **What is KTO?**
        - KTO is an alignment methodology that leverages the concept of human utility functions as described in prospect theory. It aligns LLMs by directly maximizing the utility of their outputs, focusing on whether an output is considered desirable or not by humans.
        - This method does not require detailed preference pairs for training, which is a departure from many existing alignment methodologies.
    2. **What Kind of Data Does KTO Require?**
        - KTO obliterates the need for paired-preference ranking/comparison data and simplifies data requirements significantly. It only needs binary labels indicating whether an LLM output is desirable or undesirable. Put simply, with it’s binary preference data requirement, KTO contrasts with methods such as PPO and DPO that require detailed preference pairs.
        - The simplicity in data requirements makes KTO more practical and applicable in real-world scenarios where collecting detailed preference data is challenging.
    3. **Advantages Over DPO and PPO:**
        - Compared to DPO and PPO, KTO offers several advantages:
            - **Simplicity in Data Collection**: Unlike DPO and PPO, which require paired-preference data (i.e., ranking/comparison data) which is difficult to obtain, KTO operates efficiently with unpaired binary feedback on outputs.
            - **Practicality in Real-World Application**: KTO’s less stringent data requirements make it more suitable for scenarios where collecting detailed preferences is infeasible.
            - **Focus on Utility Maximization**: KTO aligns with the practical aspects of human utility maximization, potentially leading to more user-friendly and ethically aligned outputs.
    4. **Results with KTO Compared to DPO and PPO:**
        - When applied to models of different scales (from 1B to 30B parameters), KTO has shown to match or exceed the performance of methods like DPO in terms of alignment quality.
        - KTO, even without supervised finetuning, significantly outperforms other methods at larger scales, suggesting its effectiveness in aligning models in a more scalable and data-efficient manner.
        - In terms of practical utility, the results indicate that KTO can lead to LLM outputs that are better aligned with human preferences and utility considerations, particularly in scenarios where detailed preference data is not available.
- KTO operates without paired preference data, focusing instead on maximizing the utility of language model generations based on whether an output is desirable or undesirable. This is different from the traditional approach of next-token prediction and paired preference data used in methods like DPO.
- Here’s how KTO functions:
    
    1. **Utility-Based Approach**: KTO uses a utility function, inspired by Kahneman-Tversky’s prospect theory, to evaluate the desirability of outputs. The utility function assigns a value to each possible output of the language model, reflecting how desirable (or undesirable) that output is from a human perspective.
        
    2. **Data Requirement**: Unlike DPO, KTO does not need paired comparisons between two outputs. Instead, it requires data that indicates whether a specific output for a given input is considered desirable or not. This data can come from human judgments or predefined criteria.
        
    3. **Loss Function**: The loss function in KTO is designed to maximize the expected utility of the language model’s outputs. It does this by adjusting the model’s parameters to increase the likelihood of generating outputs that have higher utility values. Note that the KTO loss function is not a binary cross-entropy loss. Instead, it is inspired by prospect theory and is designed to align large language models with human feedback. KTO focuses on human perception of losses and gains, diverging from traditional loss functions like binary cross-entropy that are commonly used in machine learning. This novel approach allows for a more nuanced understanding and incorporation of human preferences and perceptions in the training of language models. [KTO’s Loss Function](https://aman.ai/primers/ai/llm-alignment/#ktos-loss-function) further details the specifics of KTO’s loss function.
        
    4. **Training Process**: During training, the language model generates outputs, and the utility function evaluates these outputs. The model’s parameters are then updated to favor more desirable outputs according to the utility function. This process differs from next-token prediction, as it is not just about predicting the most likely next word, but about generating entire outputs that maximize a utility score.
        
    5. **Implementation**: In practical terms, KTO could be implemented as a fine-tuning process on a pre-trained language model. The model generates outputs, the utility function assesses these, and the model is updated to produce better-scoring outputs over iterations.
        
- KTO is focused more on the overall utility or value of the outputs rather than just predicting the next token. It’s a more holistic approach to aligning a language model with human preferences or desirable outcomes.
- In summary, KTO represents a shift towards a more practical and scalable approach to aligning LLMs with human feedback, emphasizing utility maximization and simplicity in data requirements.

### KTO’s Loss Function

- KTO is inspired by the behavioral models of decision-making introduced by Daniel Kahneman and Amos Tversky, particularly their prospect theory. KTO adapts these concepts into a loss function that aligns LLMs with human feedback by capturing human biases such as loss aversion and risk sensitivity. Below is a comprehensive explanation of KTO’s loss function, including both general principles from Prospect Theory and specific details from the paper you provided.

### Core Principles from Prospect Theory

- In prospect theory, human decision-making under uncertainty deviates from maximizing expected value due to biases like loss aversion and nonlinear probability weighting. These concepts are fundamental to the loss function used in KTO:

1. **Value Function**: This captures how people perceive gains and losses differently:
    
    - It is concave for gains (risk-averse for gains) and convex for losses (risk-seeking for losses).
    - Losses loom larger than gains, which is modeled by a loss aversion parameter λλ (typically λ>1λ>1).
        
    - Mathematically, the value function v(x)v(x) can be expressed as:
    
    v(x)={xα−λ(−x)βif x≥0if x<0v(x)={xαif x≥0−λ(−x)βif x<0
    
    - where:
        - α,βα,β control the diminishing sensitivity to gains and losses.
        - λλ represents the **loss aversion** factor, typically greater than 1, meaning losses are felt more intensely than gains.
2. **Probability Weighting Function**: Humans tend to overweight small probabilities and underweight large probabilities. While not central to KTO, this element of Prospect Theory highlights how subjective perceptions of uncertainty influence decisions.

### Key Elements of KTO’s Loss Function

- The KTO loss function builds on these insights, tailoring them for optimizing LLM alignment with human feedback. The key elements of the KTO loss function are:
    
    1. **Adapted Value Function**: Instead of the piecewise value function in classic Prospect Theory, KTO uses a logistic function σσ to maintain concavity for gains and convexity for losses. This also introduces a risk aversion parameter ββ, which controls the degree of risk aversion and is explicitly incorporated into the model to manage how sharply the value saturates.
        
    2. **Separate Loss Aversion Parameters**:
        - In KTO, the original loss aversion parameter λλ is replaced with two separate hyperparameters: λDλD for desirable outputs and λUλU for **undesirable** outputs. This split allows the model to handle these two types of feedback differently, reflecting more granular control over risk aversion depending on whether the output is positive or negative.
    3. **KL Divergence as a Reference Point**:
        - The reference point for the model is defined by the KL divergence between the current model’s policy πθπθ and the reference policy πrefπref. This term controls how much the current model’s outputs deviate from the pretrained reference model and acts as the reference point z0z0 for evaluating gains and losses in the optimization.

### Loss Function Equation

- The KTO loss function can be mathematically formulated as:

LKTO(πθ,πref)=𝔼x,y∼D[λy−v(x,y)]LKTO(πθ,πref)=Ex,y∼D[λy−v(x,y)]

- where: rθ(x,y)=logπθ(y|x)πref(y|x)rθ(x,y)=log⁡πθ(y|x)πref(y|x) z0=KL(πθ(y′|x)‖πref(y′|x))z0=KL(πθ(y′|x)‖πref(y′|x))
    
- The value function v(x,y)v(x,y) changes depending on whether yy is a desirable or undesirable output:
    

v(x,y)={λDσ(β(rθ(x,y)−z0))λUσ(β(z0−rθ(x,y)))if y∼desirableif y∼undesirablev(x,y)={λDσ(β(rθ(x,y)−z0))if y∼desirableλUσ(β(z0−rθ(x,y)))if y∼undesirable

### Intuition Behind the Loss Function

- If the model increases the reward of a desirable example in a blunt manner, the KL divergence penalty will also increase, preventing improvement in the loss. This forces the model to learn specific features of desirable outputs, leading to improved alignment.
- The logistic function σσ ensures that as rewards increase, the model becomes more risk-averse for gains and more risk-seeking for losses, mimicking the behavior predicted by Kahneman and Tversky’s Prospect Theory.

### Practical Considerations

- **Risk Aversion Control**: The hyperparameter ββ allows fine-tuning of the model’s sensitivity to gains and losses. Increasing ββ increases risk aversion in gains and risk-seeking behavior in losses.
- **Desirable and Undesirable Output Weighting**: The two loss aversion parameters λDλD and λUλU provide flexibility in how much weight the model gives to desirable vs. undesirable outputs. This is crucial when the training data contains an imbalance between positive and negative examples.

### Summary

- KTO’s loss function is a prospect-theoretic loss that incorporates:
    - **Loss aversion**: Through separate hyperparameters for desirable and undesirable outcomes.
    - **Risk sensitivity**: Controlled by the parameter ββ, which regulates how quickly the model’s value function saturates for gains and losses.
    - **KL divergence**: To ensure the model does not drift too far from the reference point, enforcing stability in the optimization.
- The KTO approach leverages human-like biases such as loss aversion and risk preferences, aligning the optimization process with how humans evaluate uncertainty, thus enabling better alignment of large language models with human feedback.

## Group Relative Policy Optimization (GRPO)

- Group Relative Policy Optimization (GRPO), introduced in [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300), is an RL algorithm that enhances the Proximal Policy Optimization (PPO) method by eliminating the critic model and instead using group-level scores for baseline estimation. The main goals of GRPO are to improve computational efficiency, reduce memory usage, and provide effective fine-tuning for models like DeepSeekMath.

### Key Features and Approach

1. **Actor-Only Framework**: GRPO replaces the value (critic) model from PPO with a simpler baseline calculated using group rewards. This makes GRPO less computationally intensive.
2. **Group-Based Optimization**: It samples multiple outputs (group sampling) for a given input, calculates relative rewards within the group, and uses these rewards to estimate advantages for policy updates.
3. **Adaptation for LLMs**: GRPO aligns with the comparative nature of RL for large language models, where reward functions are typically trained using pairwise comparisons of outputs.

### GRPO Equations

1. **PPO Objective Function**:
    
    - The PPO objective (for reference) is:
    
    JPPO(θ)=𝔼[min(rt(θ)At,clip(rt(θ),1−ϵ,1+ϵ)At)]JPPO(θ)=E[min(rt(θ)At,clip(rt(θ),1−ϵ,1+ϵ)At)]
    
    - where:
        - rt(θ)=πθ(ot∣q,o\textlesst)πold(ot∣q,o\textlesst)rt(θ)=πθ(ot∣q,o\textlesst)πold(ot∣q,o\textlesst): Probability ratio between the current and old policies.
        - AtAt: Advantage function.
        - ϵϵ: Clipping parameter to stabilize training.
2. **GRPO Objective**:
    
    - The GRPO objective modifies the above to avoid the critic model:
    
    JGRPO(θ)=𝔼q,{oi}Gi=11G∑i=1G1|oi|∑t=1|oi|min(ri,t(θ)Â i,t,clip(ri,t(θ),1−ϵ,1+ϵ)Â i,t)JGRPO(θ)=Eq,{oi}i=1G1G∑i=1G1|oi|∑t=1|oi|min(ri,t(θ)A^i,t,clip(ri,t(θ),1−ϵ,1+ϵ)A^i,t)
    
    - where:
        - GG: Number of outputs sampled for each input qq (group size).
        - Â i,tA^i,t: Advantage for the tt-th token of output oioi, calculated from group-relative rewards.
3. **Advantage Calculation**:
    
    - GRPO estimates the advantage Â i,tA^i,t as:
    
    Â i,t=ri−mean(r)std(r)A^i,t=ri−mean(r)std(r)
    
    - where riri is the reward for output oioi, and mean(r)mean(r), std(r)std(r) are computed over the group.
4. **KL Regularization**:
    
    - GRPO introduces a KL divergence penalty to stabilize updates:
    
    DKL=∑tπθ(oi,t∣q,o<t)log(πθ(oi,t∣q,o<t)πref(oi,t∣q,o<t))DKL=∑tπθ(oi,t∣q,o<t)log⁡(πθ(oi,t∣q,o<t)πref(oi,t∣q,o<t))
    

### Implementation Details

1. **Input Data**:
    - Questions (qq) are sampled from a dataset.
    - Multiple outputs (GG) are generated per question using the old policy.
2. **Reward Model**:
    - Rewards (riri) are computed using a pre-trained reward model.
    - Rewards are normalized within the group to calculate relative advantages.
3. **Optimization Steps**:
    - Sample outputs and compute rewards.
    - Compute group-relative advantages.
    - Update the policy model by maximizing the GRPO objective.
    - Apply KL regularization to prevent the policy from drifting too far from the reference model.
4. **Hyperparameters**:
    - ϵϵ: Clipping parameter (e.g., 0.2).
    - ββ: KL regularization coefficient.
    - GG: Group size (e.g., 64 outputs per input).
    - Learning rate: Typically in the range of 10−610−6 to 10−510−5.

### Pros and Cons

#### Pros

- **Efficiency**: GRPO reduces memory and computation requirements by eliminating the critic model.
- **Simplicity**: The advantage is computed directly from group scores without training an additional value model.
- **Alignment with Reward Models**: Leverages the comparative nature of reward functions effectively.
- **Improved Performance**: Demonstrated superior results on benchmarks like GSM8K and MATH compared to other RL methods.

#### Cons

- **Dependence on Group Size**: Requires careful tuning of the group size GG for effective advantage estimation.
- **Reward Model Quality**: Relies heavily on the quality of the reward model for accurate advantage computation.
- **Applicability**: May not generalize well to tasks with sparse or noisy reward signals.

### Applications and Results

- GRPO significantly enhances the mathematical reasoning capabilities of models like DeepSeekMath.
- On GSM8K and MATH datasets, GRPO achieved 88.2% and 51.7% accuracy, respectively, outperforming other open-source methods.

## Comparative Analysis: REINFORCE vs. TRPO vs. PPO vs. DPO vs. KTO vs. APO vs. GRPO

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

### Tabular Comparison

|**Aspect**|**REINFORCE**|**TRPO**|**PPO**|**DPO**|**KTO**|**APO**|**GRPO**|
|---|---|---|---|---|---|---|---|
|Objective|Policy gradient optimization without constraints.|Ensures stable policy updates within a constrained region.|Maximizes expected reward while preventing large policy updates.|Optimizes policy based on binary classification of human preferences.|Aligns models based on Kahneman-Tversky optimization for utility maximization.|Anchored alignment with specific control over preference-based likelihood adjustments.|Leverages group-based relative advantages and removes the critic network.|
|Learning Mechanism|Monte Carlo policy gradients with high variance.|Second-order optimization with trust region constraints.|Policy gradients with a clipped surrogate objective.|Cross-entropy optimization over paired preferences.|Maximizes desirable likelihoods relative to undesirables, without paired data.|Uses variants like APO-zero or APO-down for stable preference-based optimization.|Group normalization with policy gradients, eliminating the critic network.|
|Stability|Low (high variance, unstable updates).|High (enforces trust region for stable updates).|Relies on clipping mechanisms to avoid destabilization.|Stable as it directly optimizes preferences.|Stable due to focus on unpaired desirability adjustments.|Offers robust training stability, scaling better on models trained with mixed-quality datasets.|Stable due to normalization of rewards across groups.|
|Training Complexity|High (unconstrained updates).|Very high (requires second-order optimization and solving constraints).|High, due to balancing reward maximization with policy constraints.|Moderate; uses simplified binary preference objectives.|Simplifies alignment by focusing only on desirability.|Adaptive and context-aware; requires understanding dataset-model relationships.|Reduces overhead via group-based scoring.|
|Performance|Unstable and sample-inefficient.|More stable than PPO but computationally expensive.|Strong performance on tasks with clear reward signals but prone to instability in distributed setups.|Effective for straightforward preference alignment tasks.|Competitive or better alignment than preference-based methods without paired data needs.|Superior alignment results, particularly for nuanced dataset control.|Excels in reasoning tasks, offering computational efficiency.|
|Notable Strength|Simple to implement but inefficient.|Ensures stable policy updates through trust-region constraints.|Widely used in RL settings, good at reward-based optimization.|Directly optimizes for preferences without needing a separate reward model.|Handles binary data efficiently, avoiding paired data dependencies.|Allows precise alignment with nuanced datasets.|Simplifies reward aggregation; strong for reasoning-heavy tasks.|
|Scenarios Best Suited|RL tasks where simplicity is preferred over efficiency.|High-stability RL tasks requiring constraint-driven policy improvements.|RL environments where reward signals are predefined.|Scenarios with abundant paired human feedback.|Real-world settings with broad definitions of desirable/undesirable outputs.|Tasks requiring precise alignment with minimally contrasting preferences.|Mathematical reasoning or low-resource training setups.|

## Bias Concerns and Mitigation Strategies

- A fair question to ask now is if RLHF/RLAIF can add bias to the model. This is an important topic as large conversational language models are being deployed in various applications from search engines (Bing Chat, Google’s Bard) to word documents (Microsoft office co-pilot, Google docs, Notion, etc.).
- The answer is, yes, just as with any machine learning approach with human input, RLHF has the potential to introduce bias.
- Let’s look at the different forms of bias it can introduce:
    - **Selection bias:**
        - RLHF relies on feedback from human evaluators, who may have their own biases and preferences (and can thus limit their feedback to topics or situations they can relate to). As such, the agent may not be exposed to the true range of behaviors and outcomes that it will encounter in the real world.
    - **Confirmation bias:**
        - Human evaluators may be more likely to provide feedback that confirms their existing beliefs or expectations, rather than providing objective feedback based on the agent’s performance.
        - This can lead to the agent being reinforced for certain behaviors or outcomes that may not be optimal or desirable in the long run.
    - **Inter-rater variability:**
        - Different human evaluators may have different opinions or judgments about the quality of the agent’s performance, leading to inconsistency in the feedback that the agent receives.
        - This can make it difficult to train the agent effectively and can lead to suboptimal performance.
    - **Limited feedback:**
        - Human evaluators may not be able to provide feedback on all aspects of the agent’s performance, leading to gaps in the agent’s learning and potentially suboptimal performance in certain situations.
- Now that we’ve seen the different types of bias possible with RLHF, lets look at ways to mitigate them:
    - **Diverse evaluator selection:**
        - Selecting evaluators with diverse backgrounds and perspectives can help to reduce bias in the feedback, just as it does in the workplace.
        - This can be achieved by recruiting evaluators from different demographic groups, regions, or industries.
    - **Consensus evaluation:**
        - Using consensus evaluation, where multiple evaluators provide feedback on the same task, can help to reduce the impact of individual biases and increase the reliability of the feedback.
        - This is almost like ‘normalizing’ the evaluation.
    - **Calibration of evaluators:**
        - Calibrating evaluators by providing them with training and guidance on how to provide feedback can help to improve the quality and consistency of the feedback.
    - **Evaluation of the feedback process:**
        - Regularly evaluating the feedback process, including the quality of the feedback and the effectiveness of the training process, can help to identify and address any biases that may be present.
    - **Evaluation of the agent’s performance:**
        - Regularly evaluating the agent’s performance on a variety of tasks and in different environments can help to ensure that it is not overfitting to specific examples and is capable of generalizing to new situations.
    - **Balancing the feedback: **
        - Balancing the feedback from human evaluators with other sources of feedback, such as self-play or expert demonstrations, can help to reduce the impact of bias in the feedback and improve the overall quality of the training data.

## [TRL - Transformer RL](https://github.com/huggingface/trl)

- The `trl` library is a full stack library to fine-tune and align transformer language and diffusion models using methods such as Supervised Fine-tuning step (SFT), Reward Modeling (RM) and the Proximal Policy Optimization (PPO) as well as Direct Preference Optimization (DPO).
- The library is built on top of the `transformers` library and thus allows to use any model architecture available there.

## Selected Papers

### OpenAI’s Paper on InstructGPT: [Training Language Models to Follow Instructions with Human Feedback](https://arxiv.org/abs/2203.02155)

- Making language models bigger does not inherently make them better at following a user’s intent. For example, large language models can generate outputs that are untruthful, toxic, or simply not helpful to the user. In other words, these models are not aligned with their users.
- [Ouyang et al. (2022)](https://arxiv.org/abs/2203.02155) from OpenAI introduces InstructGPT, a model that aligns language models with user intent on a wide range of tasks by fine-tuning with human feedback.
- Starting with a set of labeler-written prompts and prompts submitted through the OpenAI API, they collect a dataset of labeler demonstrations of the desired model behavior, which they use to fine-tune GPT-3 using supervised fine-tuning (SFT). This process is referred to as “instruction tuning” by other papers such as [Wei et al. (2022)](https://aman.ai/primers/ai/llm-alignment/#finetuned-language-models-are-zero-shot-learners).
- They then collect a dataset of rankings of model outputs, which they use to further fine-tune this supervised model using RLHF.
- In human evaluations on their prompt distribution, outputs from the 1.3B parameter InstructGPT model are preferred to outputs from the 175B GPT-3, despite having 100x fewer parameters.
- Moreover, InstructGPT models show improvements in truthfulness and reductions in toxic output generation while having minimal performance regressions on public NLP datasets. Even though InstructGPT still makes simple mistakes, their results show that fine-tuning with human feedback is a promising direction for aligning language models with human intent.
- It is important to note that ChatGPT is trained using the same methods as InstructGPT (using SFT followed by RLHF), but is fine-tuned from a model in the GPT-3.5 series.
- Furthermore, the fine-tuning process proposed in the paper isn’t without its challenges. First, we need a significant volume of demonstration data. For instance, in the InstructGPT paper, they used 13k instruction-output samples for supervised fine-tuning, 33k output comparisons for reward modeling, and 31k prompts without human labels as input for RLHF. Second, fine-tuning comes with an alignment tax “negative transfer” – the process can lead to lower performance on certain critical tasks. (There’s no free lunch after all.) The same InstructGPT paper found that RLHF led to performance regressions (relative to the GPT-3 base model) on public NLP tasks like SQuAD, HellaSwag, and WMT 2015 French to English. A potential workaround is to have several smaller, specialized models that excel at narrow tasks.
- The figure below from the paper illustrates the three steps of training InstructGPT: (1) SFT, (2) reward model training, and (3) RL via proximal policy optimization (PPO) on this reward model. Blue arrows indicate that this data is used to train the respective model in the diagram. In Step 2, boxes A-D are samples from the SFT model that get ranked by labelers.

![](https://aman.ai/images/papers/InstructGPT.jpg)

### [Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073)

- The paper extends RLHF by training language models on datasets labeled for helpfulness and harmlessness. It introduces ‘HH’ models, which are trained on both criteria and have shown to be more harmless and better at following instructions than models trained on helpfulness alone.
- An evaluation of these models’ ability to identify harmful behavior in language model interactions was conducted using a set of conversations rated for harmfulness. The study leveraged ‘red teaming’ where humans attempted to provoke the AI into harmful responses, thereby improving the training process.
- The effectiveness of the training method was demonstrated through models’ performance on questions assessing helpfulness, honesty, and harmlessness, without relying on human labels for harmlessness.
- This research aligns with other efforts like LaMDA and InstructGPT, which also utilize human data to train language models. The concept of ‘constitutional AI’ was introduced, focusing on self-critique and revision by the AI to foster both harmless and helpful interactions. The ultimate goal is to create AI that can self-regulate harmfulness while remaining helpful and responsive.

### OpenAI’s Paper on PPO: [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)

- [Schulman et al. (2017)](https://arxiv.org/abs/1707.06347) proposes a new family of policy gradient methods for RL, which alternate between sampling data through interaction with the environment, and optimizing a “surrogate” objective function using stochastic gradient ascent.
- Whereas standard policy gradient methods perform one gradient update per data sample, they propose a novel objective function that enables multiple epochs of minibatch updates. The new methods, which they call proximal policy optimization (PPO), have some of the benefits of trust region policy optimization (TRPO), but they are much simpler to implement, more general, and have better sample complexity (empirically).
- Their experiments test PPO on a collection of benchmark tasks, including simulated robotic locomotion and Atari game playing, showing that PPO outperforms other online policy gradient methods, and overall strikes a favorable balance between sample complexity, simplicity, and wall clock time.

### [A General Language Assistant As a Laboratory for Alignment](https://arxiv.org/abs/2112.00861)

- This paper by Askell et al. from Anthropic introduces a comprehensive study towards aligning general-purpose, text-based AI systems with human values, focusing on making AI helpful, honest, and harmless (HHH). Given the capabilities of large language models, the authors investigate various alignment techniques and their evaluations to ensure these models adhere to human preferences without compromising performance.
- The authors begin by examining naive prompting as a baseline for alignment, finding that the benefits from such interventions increase with model size and generalize across multiple alignment evaluations. Prompting was shown to impose negligible performance costs (‘alignment taxes’) on large models. The paper also explores the scaling trends of several training objectives relevant to alignment, including imitation learning, binary discrimination, and ranked preference modeling. The results indicate that ranked preference modeling significantly outperforms imitation learning and scales more favorably with model size, while binary discrimination performs similarly to imitation learning.
- A key innovation discussed is ‘preference model pre-training’ (PMP), which aims to improve the sample efficiency of fine-tuning models on human preferences. This involves pre-training on large public datasets that encode human preferences, such as Stack Exchange, Reddit, and Wikipedia edits, before fine-tuning on smaller, more specific datasets. The findings suggest that PMP substantially enhances sample efficiency and often improves asymptotic performance when fine-tuning on human feedback datasets.
- **Implementation Details:**
    - **Prompts and Context Distillation:** The authors utilize a prompt composed of 14 fictional conversations to induce the HHH criteria in models. They introduce ‘context distillation,’ a method where the model is fine-tuned using the KL divergence between the model’s predictions and the distribution conditioned on the prompt context. This technique effectively transfers the prompt’s conditioning into the model.
    - **Training Objectives:**
        - **Imitation Learning:** Models are trained to imitate ‘good’ behavior using supervised learning on sequences labeled as correct or desirable.
        - **Binary Discrimination:** Models distinguish between ‘correct’ and ‘incorrect’ behavior by training on pairs of correct and incorrect samples.
        - **Ranked Preference Modeling:** Models are trained to assign higher scores to better samples from ranked datasets using pairwise comparisons, a more complex but effective approach for capturing preferences.
    - **Preference Model Pre-Training (PMP):** The training pipeline includes a PMP stage where models are pre-trained on binary discriminations sourced from Stack Exchange, Reddit, and Wikipedia edits. This stage significantly enhances sample efficiency during subsequent fine-tuning on smaller datasets.
- **Results:**
    - **Prompting:** Simple prompting significantly improves model performance on alignment evaluations, including HHH criteria and toxicity reduction. Prompting and context distillation both decrease toxicity in generated text as model size increases.
    - **Scaling Trends:** Ranked preference modeling outperforms imitation learning, especially on tasks with ranked data like summarization and HellaSwag. Binary discrimination shows little improvement over imitation learning.
    - **Sample Efficiency:** PMP dramatically increases the sample efficiency of fine-tuning, with larger models benefiting more from PMP than smaller ones. Binary discrimination during PMP is found to transfer better than ranked preference modeling.
- The figure below from the paper shows: (Left) Simple prompting significantly improves performance and scaling on our HHH alignment evaluations (y-axis measures accuracy at choosing better responses on our HHH evaluations). (Right) Prompts impose little or no ‘alignment tax’ on large models, even on complex evaluations like function synthesis. Here we have evaluated our python code models on the HumanEval codex dataset at temperature T = 0.6 and top P = 0.95.

![](https://aman.ai/images/papers/HHH.jpg)

- The study demonstrates that simple alignment techniques like prompting can lead to meaningful improvements in AI behavior, while more sophisticated methods like preference modeling and PMP offer scalable and efficient solutions for aligning large language models with human values.

### Anthropic’s Paper on Constitutional AI: [Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073)

- As AI systems become more capable, we would like to enlist their help to supervise other AIs.
- [Bai et al. (2022)](https://arxiv.org/abs/2212.08073) experiments with methods for training a harmless AI assistant through self-improvement, without any human labels identifying harmful outputs. The only human oversight is provided through a list of rules or principles, and so they refer to the method as ‘Constitutional AI’.
- The process involves both a supervised learning and a RL phase. In the supervised phase they sample from an initial model, then generate self-critiques and revisions, and then finetune the original model on revised responses. In the RL phase, they sample from the finetuned model, use a model to evaluate which of the two samples is better, and then train a preference model from this dataset of AI preferences.
- They then train with RL using the preference model as the reward signal, i.e. they use ‘RL from AI Feedback’ (RLAIF). As a result they are able to train a harmless but non-evasive AI assistant that engages with harmful queries by explaining its objections to them. Both the SL and RL methods can leverage chain-of-thought style reasoning to improve the human-judged performance and transparency of AI decision making. These methods make it possible to control AI behavior more precisely and with far fewer human labels.
- The figure below from the paper shows the basic steps of their Constitutional AI (CAI) process, which consists of both a supervised learning (SL) stage, consisting of the steps at the top, and a RL (RL) stage, shown as the sequence of steps at the bottom of the figure. Both the critiques and the AI feedback are steered by a small set of principles drawn from a ‘constitution’. The supervised stage significantly improves the initial model, and gives some control over the initial behavior at the start of the RL phase, addressing potential exploration problems. The RL stage significantly improves performance and reliability.

![](https://aman.ai/images/papers/CAI.jpg)

- The graph below shows harmlessness versus helpfulness Elo scores (higher is better, only differences are meaningful) computed from crowdworkers’ model comparisons for all 52B RL runs. Points further to the right are later steps in RL training. The Helpful and HH models were trained with human feedback as in [Bai et al., 2022], and exhibit a tradeoff between helpfulness and harmlessness. The RL-CAI models trained with AI feedback learn to be less harmful at a given level of helpfulness. The crowdworkers evaluating these models were instructed to prefer less evasive responses when both responses were equally harmless; this is why the human feedback-trained Helpful and HH models do not differ more in their harmlessness scores.

![](https://aman.ai/images/papers/CAI2.jpg)

### [RLAIF: Scaling RL from Human Feedback with AI Feedback](https://arxiv.org/abs/2309.00267)

- This paper by Lee et al. from Google Research, introduces a novel method for training large language models (LLMs) with AI-generated feedback, addressing the challenges and costs associated with traditional human feedback methods.
- The paper presents RL from AI Feedback (RLAIF) as a promising alternative to the conventional RLHF. RLAIF utilizes an off-the-shelf LLM as a preference labeler, streamlining the training process and, in some cases, surpassing the performance of models trained with human feedback.
- This approach is applied to text generation tasks such as summarization, helpful dialogue generation, and harmless dialogue generation. The performance of RLAIF, as assessed by human raters, is comparable or superior to RLHF, challenging the assumption that larger policy models are always more effective.
- A key advantage of RLAIF is its potential to significantly reduce reliance on expensive human annotations. The study shows the efficacy of using the same model size for both the LLM labeler and the policy model, and highlights that directly prompting the LLM for reward scores can be more effective than using a distilled reward model.
- The authors explore methodologies for generating AI preferences aligned with human values, emphasizing the effectiveness of chain-of-thought reasoning and detailed preamble in improving AI labeler alignment.
- The following figure from the paper shows a diagram depicting RLAIF (top) vs. RLHF (bottom).

![](https://aman.ai/images/papers/RLAIF.jpg)

- RLAIF’s scalability and cost-effectiveness are notable, with the approach being over ten times cheaper than human annotation. This aligns with the growing trend in LLM research focusing on quality over quantity in datasets.
- The paper suggests that combining RLHF and RLAIF could be a strategic approach, especially considering that LLMs like GPT-4 have been trained with human feedback. This hybrid model could represent a balanced integration of high-quality human data, amplified significantly by AI, potentially shaping the future of LLM training and influencing approaches like the development of GPT-5.

### [A General Theoretical Paradigm to Understand Learning from Human Preferences](https://arxiv.org/abs/2310.12036)

- This paper by Azar et al. from Google DeepMind delves into the theoretical underpinnings of learning from human preferences, particularly focusing on RL from human feedback (RLHF) and direct preference optimization (DPO). The authors propose a novel objective, ΨΨ-preference optimization (ΨΨPO), which encompasses RLHF and DPO as specific instances, aiming to optimize policies directly from human preferences without relying on the approximations common in existing methods.
- RLHF typically involves a two-step process where a reward model is first trained using a binary classifier to distinguish preferred actions, often employing a Bradley-Terry model for this purpose. This is followed by policy optimization to maximize the learned reward while ensuring the policy remains close to a reference policy through KL regularization. DPO, in contrast, seeks to optimize the policy directly from human preferences, eliminating the need for explicit reward model training.
- The ΨΨPO framework is a more general approach that seeks to address the potential overfitting issues inherent in RLHF and DPO by considering pairwise preferences and employing a possibly non-linear function of preference probabilities alongside KL regularization. Specifically, the Identity-PO (IPO) variant of ΨΨPO is highlighted for its practicality and theoretical appeal, as it allows for direct optimization from preferences without the approximations used in other methods.
- Empirical demonstrations show that IPO can effectively learn from preferences without succumbing to the overfitting problems identified in DPO, providing a robust method for preference-based policy optimization. The paper suggests that future work could explore scaling these theoretical insights to more complex settings, such as training language models on human preference data.

### [SLiC-HF: Sequence Likelihood Calibration with Human Feedback](https://arxiv.org/abs/2305.10425)

- This paper by Zhao et al. from Google Deepmind and Google Research introduces Sequence Likelihood Calibration with Human Feedback (SLiC-HF) as a method for aligning language models with human preferences using human feedback data. SLiC-HF is showcased as an effective, simpler, and more computationally efficient alternative to RL from Human Feedback (RLHF), particularly for the task of TL;DR summarization.
- SLiC-HF operates by calibrating the sequence likelihood of a Supervised Fine-Tuning (SFT) model against human feedback data, either directly or through a ranking model derived from human judgments. This is in contrast to traditional RLHF approaches that rely on optimizing a language model using a reward model trained on human preferences.
- The paper details several implementations of SLiC-HF: direct application of human feedback (SLiC-HF-direct), sample-and-rank approach using either a reward model or a ranking model (SLiC-HF-sample-rank), and a variant applying SLiC-HF directly on human feedback data without the need for a separate ranking/reward model. Specifically, yo determine the rank, they consider two text-to-text models trained from the human preference data:
    - Trained Pointwise Reward model: They binarize each ranked pair into a positive and a negative sequence, as shown in the figure below. When training the reward model, input sequences are formatted as ‘[Context] … [Summary] …’ and target sequences are either ‘Good’ or ‘Bad’. At inference time, we compute the probability of token ‘Good’ on the decoder side to score each of the mm candidates in a list, and sample mm positive/negative pairs from them.
    - Trained Pairwise Ranking model: As shown in the figure below, we formulate the human feedback into a pairwise ranking problem with text-to-text format. When training the ranking model, input sequences are formatted as ‘[Context] … [Summary A] … [Summary B]’ and target sequences are among ‘A’ or ‘B’. At inference time, we use a tournament-style procedure to rank candidates in a list. For example, given a list of 4 candidates c1c1, c2c2, c3c3, c4c4, we first rank c1c1, c2c2 and c3c3, c4c4 and then rank winner (c1,c2)(c1,c2), winner (c3,c4)(c3,c4). Given mm candidates, the ranking model is called m−1m−1 times and m−1m−1 positive/negative pairs are yielded.
- The following figure from the paper shows the data format for training the text-to-text reward model and ranking model.

![](https://aman.ai/images/papers/SLiC-HF.jpg)

- Extensive experiments demonstrate that SLiC-HF significantly improves upon SFT baselines and offers competitive performance to RLHF-PPO implementations. The experiments involved automatic and human evaluations, focusing on the Reddit TL;DR summarization task. Results showed SLiC-HF’s capability to produce high-quality summaries, with improvements observed across different configurations and parameter scales.
- The paper contributes to the field by providing a detailed methodology for implementing SLiC-HF, showcasing its efficiency and effectiveness compared to traditional RLHF methods. It also demonstrates the viability of leveraging off-policy human feedback data, thus potentially reducing the need for costly new data collection efforts.
- Further discussions in the paper explore the computational and memory efficiency advantages of SLiC-HF over RLHF-PPO, highlighting the former’s scalability and potential for broader application in language generation tasks. The paper concludes with suggestions for future research directions, including exploring other reward functions and non-human feedback mechanisms for language model calibration.

### [Reinforced Self-Training (ReST) for Language Modeling](https://arxiv.org/abs/2308.08998)

- RLHF can improve the quality of large language model’s (LLM) outputs by aligning them with human preferences.
- This paper by Gulcehre et al. from Google DeepMind and Google Research proposes Reinforced Self-Training (ReST), a simple algorithm for aligning LLMs with human preferences inspired by growing batch RL (RL).
- ReST generates samples from an initial LLM policy to create a dataset, which is then used to improve the LLM policy using offline RL algorithms. This method is more efficient than traditional online RLHF methods due to offline production of the training dataset, facilitating data reuse.
- ReST operates in two loops: the inner loop (Improve) and the outer loop (Grow).
    - **Grow**: The LLM policy generates multiple output predictions per context, augmenting the training dataset.
    - **Improve**: The augmented dataset is ranked and filtered using a scoring function based on a learned reward model trained on human preferences. The model is then fine-tuned on this filtered dataset with an offline RL objective, with the possibility of repeating this step with increasing filtering thresholds.
- The following image from the paper illustrates the ReST method. During the Grow step, a policy generates a dataset. At Improve step, the filtered dataset is used to fine-tune the policy. Both steps are repeated, the Improve step is repeated more frequently to amortise the dataset creation cost.

![](https://aman.ai/images/papers/ReST.jpg)

- ReST’s advantages include reduced computational burden, independence from the original dataset’s quality, and simplicity in implementation.
- Machine translation was chosen as the application for testing ReST, due to strong baselines and well-defined evaluation procedures. Experiments were conducted on IWSLT 2014, WMT 2020 benchmarks, and an internal high-fidelity benchmark called Web Domain. The evaluation used state-of-art reference-free reward models like Metric X, BLEURT, and COMET. ReST significantly improved reward model scores and translation quality on test and validation sets, as per both automated metrics and human evaluation.
- ReST outperformed standard supervised learning (BC G=0 I=0) in reward model scores and human evaluations. The BC loss (Behavioral Cloning) was found to be the most effective for ReST, leading to continuous improvements in the model’s reward on holdout sets. However, improvements in reward model scores did not always align with human preferences.
- ReST showed better performance over supervised training across different datasets and language pairs. The inclusion of multiple Improve steps and Grow steps resulted in significant improvements in performance. Human evaluations showed that all ReST variants significantly outperformed the BC baseline.
- ReST is distinct from other self-improvement algorithms in language modeling due to its computational efficiency and ability to leverage exploration data and rewards. The approach is applicable to various language tasks, including summarization, dialogue, and other generative models.
- Future work includes fine-tuning reward models on subsets annotated with human preferences and exploring better RL exploration strategies.

### [Beyond Human Data: Scaling Self-Training for Problem-Solving with Language Models](https://arxiv.org/abs/2312.06585)

- Training language models typically requires vast quantities of human-generated text, which can be scarce or of variable quality, especially for specialized domains like mathematics or programming. This scarcity limits the model’s ability to learn diverse patterns and hinders its performance. ReSTEMReSTEM addresses this problem by reducing the reliance on human-curated datasets and instead exploring the potential of fine-tuning models using self-generated data validated through scalar feedback mechanisms.
- This paper by Singh et al. from Google DeepMind, presented at NeurIPS 2023, explores a new frontier in Large Language Model (LLM) training: Reinforced Self-Training based on expectation-maximization (ReSTEMReSTEM). This innovative approach aims to reduce reliance on human data while avoiding the pitfalls of a synthetic data death spiral, a trend becoming increasingly evident in LLM training.
- ReSTEMReSTEM is a potent alternative to traditional dataset curation, comprising two primary stages: generating multiple output samples (E-step) and fine-tuning the language model on these samples (M-step). This process is cyclically iterated, combining the generation of model-derived answers and their subsequent refinement. The feedback for filtering these outputs is sourced from tasks with binary feedback, such as math problems with clear right or wrong answers.
- The paper’s focus is on two challenging domains: advanced mathematical problem-solving (MATH) and code generation (APPS). Utilizing PaLM 2 models of various scales, the study demonstrates that ReSTEMReSTEM significantly outperforms models fine-tuned solely on human-generated data, offering up to 2x performance boosts. This indicates a major step toward more independent AI systems, seeking less human input for skill refinement.
- ReSTEMReSTEM employs an iterative self-training process leveraging expectation-maximization. It first generates outputs from the language model, then applies a filtering mechanism based on binary correctness feedback—essentially sorting the wheat from the chaff. Subsequently, the model is fine-tuned using these high-quality, self-generated samples. This cycle is repeated several times, thus iteratively enhancing the model’s accuracy and performance on tasks by self-generating and self-validating the training data.
- Notably, the experiments revealed diminishing returns beyond a certain number of ReST iterations, suggesting potential overfitting issues. Ablation studies further assessed the impact of dataset size, the number of model-generated solutions, and the number of iterations on the effectiveness of ReST.
- The models fine-tuned using ReST showed enhanced performance on related but distinct benchmarks like GSM8K, Hungarian HS finals, and Big-Bench Hard tasks, without any noticeable degradation in broader capabilities. This finding underscores the method’s versatility and generalizability.
- The following figure from the paper shows Pass@K results for PaLM-2-L pretrained model as well as model fine-tuned with ReSTEMReSTEM. For a fixed number of samples KK, fine-tuning with ReSTEMReSTEM substantially improves Pass@K performance. They set temperature to 1.0 and use nucleus sampling with p=0.95p=0.95.

![](https://aman.ai/images/papers/BHD.jpg)

- While ReST offers significant advantages in performance, it necessitates a moderate-sized training set of problems or prompts and access to a manually-designed or learned reward function. It’s highly data-efficient but requires careful application to prevent overfitting.
- This research opens new avenues for self-improvement in language models, suggesting the need for automating manual parts of the pipeline and exploring algorithmic improvements to further enhance performance. With ReSTEMReSTEM showing promising results, especially in larger models, one can anticipate further exploration in applying self-training techniques to various other domains beyond math and coding tasks. The significant improvement over fine-tuning on human data implies that future models can be made more efficient, less reliant on extensive datasets, and potentially achieve better performance.

### [Diffusion Model Alignment Using Direct Preference Optimization](https://arxiv.org/abs/2311.12908)

- This paper by Wallace et al. from Salesforce AI and Stanford University proposes a novel method for aligning diffusion models to human preferences.
- The paper introduces Diffusion-DPO, a method adapted from DPO, for aligning text-to-image diffusion models with human preferences. This approach is a significant shift from typical language model training, emphasizing direct optimization on human comparison data.
- Unlike typical methods that fine-tune pre-trained models using curated images and captions, Diffusion-DPO directly optimizes a policy that best satisfies human preferences under a classification objective. It re-formulates DPO to account for a diffusion model notion of likelihood using the evidence lower bound, deriving a differentiable objective.
- The authors utilized the Pick-a-Pic dataset, comprising 851K crowdsourced pairwise preferences, to fine-tune the base model of the Stable Diffusion XL (SDXL)-1.0 model with Diffusion-DPO. The fine-tuned model showed significant improvements over both the base SDXL-1.0 and its larger variant in terms of visual appeal and prompt alignment, as evaluated by human preferences.
- The paper also explores a variant of the method that uses AI feedback, showing comparable performance to training on human preferences. This opens up possibilities for scaling diffusion model alignment methods.
- The figure below from paper illustrates: (Top) DPO-SDXL significantly outperforms SDXL in human evaluation. (L) PartiPrompts and (R) HPSv2 benchmark results across three evaluation questions, majority vote of 5 labelers. (Bottom) Qualitative comparisons between SDXL and DPO-SDXL. DPOSDXL demonstrates superior prompt following and realism. DPO-SDXL outputs are better aligned with human aesthetic preferences, favoring high contrast, vivid colors, fine detail, and focused composition. They also capture fine-grained textual details more faithfully.

![](https://aman.ai/images/papers/Diffusion-DPO.jpg)

- Experiments demonstrate the effectiveness of Diffusion-DPO in various scenarios, including image-to-image editing and learning from AI feedback. The method significantly outperforms existing models in human evaluations for general preference, visual appeal, and prompt alignment.
- The paper’s findings indicate that Diffusion-DPO can effectively increase measured human appeal across an open vocabulary with stable training, without increased inference time, and improves generic text-image alignment.
- The authors note ethical considerations and risks associated with text-to-image generation, emphasizing the importance of diverse and representative sets of labelers and the potential biases inherent in the pre-trained models and labeling process.
- In summary, the paper presents a groundbreaking approach to align diffusion models with human preferences, demonstrating notable improvements in visual appeal and prompt alignment. It highlights the potential of direct preference optimization in the realm of text-to-image diffusion models and opens avenues for further research and application in this field.

### [Human-Centered Loss Functions (HALOs)](https://github.com/ContextualAI/HALOs/blob/main/assets/report.pdf)

- This report by Ethayarajh et al. from Stanford University presents a novel approach to aligning large language models (LLMs) with human feedback, building upon Kahneman & Tversky’s prospect theory. The proposed Kahneman-Tversky Optimization (KTO) loss function diverges from existing methods by not requiring paired preference data, relying instead on the knowledge of whether an output is desirable or undesirable for a given input. This makes KTO significantly easier to deploy in real-world scenarios where such data is more abundant.
- The report identifies that existing methods for aligning LLMs with human feedback can be seen as human-centered loss functions, which implicitly model some of the distortions in human perception as suggested by prospect theory. By adopting this perspective, the authors derive a HALO that maximizes the utility of LLM generations directly, rather than relying on maximizing the log-likelihood of preferences, as current methods do.
- The KTO-aligned models were found to match or exceed the performance of direct preference optimization methods across scales from 1B to 30B. One of the key advantages of KTO is its feasibility in real-world applications, as it requires less specific types of data compared to other methods.
- To validate the effectiveness of KTO and understand how alignment scales across model sizes, the authors introduced Archangel, a suite comprising 56 models. These models, ranging from 1B to 30B, were aligned using various methods, including KTO, on human-feedback datasets such as Anthropic HH, Stanford Human Preferences, and OpenAssistant.
- The following report from the paper illustrates the fact that LLM alignment involves supervised finetuning followed by optimizing a human-centered loss (HALO). However, the paired preferences that existing approaches need are hard-to-get. Kahneman-Tversky Optimization (KTO) uses a far more abundant kind of data, making it much easier to use in the real world.

![](https://aman.ai/images/papers/HALO.jpg)

- The report’s experimental findings reveal surprising insights into the scaling and effectiveness of different alignment methods. It was observed that supervised finetuning (SFT) contributes significantly to the performance gains at every scale under 30B. The benefits of combining SFT with alignment methods become apparent at model sizes of around 7B and above. Interestingly, KTO alone was found to be significantly better than DPO (Direct Preference Optimization) alone at scales of 13B and 30B.
- The practical implications of KTO are notable, especially in contexts where abundant data on customer interactions and outcomes is available, but counterfactual data is scarce. This aspect underscores KTO’s potential for broader application in real-world settings compared to preference-based methods like DPO.
- Future work suggested by the authors includes exploring a human value function specifically for language, examining differences in model behavior at different scales, and investigating the potential of synthetic data in model alignment with KTO. The report highlights the importance of understanding how human-centered loss functions can influence the alignment of LLMs with human preferences and perceptions.
- [Code](https://github.com/ContextualAI/HALOs/)

### [Nash Learning from Human Feedback](https://arxiv.org/abs/2312.00886)

- This paper by Munos et al. from Google DeepMind introduces an alternative approach to the conventional RLHF for aligning large language models (LLMs) with human preferences. This new approach, termed Nash Learning from Human Feedback (NLHF), focuses on learning a preference model from pairwise human feedback and pursuing a policy that generates responses preferred over any competing policy, thus achieving a Nash equilibrium for this preference model.
- The NLHF approach aims to encompass a broader spectrum of human preferences, maintain policy independence, and better align with the diversity of human preferences. This method marks a significant shift from the traditional RLHF framework, which is more limited in capturing the richness and diversity of human preferences.
- Key contributions of this work include the introduction and definition of a regularized variant of the preference model, the establishment of the existence and uniqueness of the corresponding Nash equilibrium, and the introduction of novel algorithms such as Nash-MD and Nash-EMA. Nash-MD, founded on mirror descent principles, converges to the Nash equilibrium without requiring the storage of past policies, making it particularly suitable for LLMs. Nash-EMA, inspired by fictitious play, uses an exponential moving average of past policy parameters. The paper also introduces policy-gradient algorithms Nash-MD-PG and Nash-EMA-PG for deep learning architectures. Extensive numerical experiments conducted on a text summarization task using the TL;DR dataset validate the effectiveness of the NLHF approach.
- The regularized preference model in NLHF uses KL-regularization to quantify the divergence between the policy under consideration and a reference policy. This regularization is particularly crucial in situations where the preference model is more accurately estimated following a given policy or where it is essential to remain close to a known safe policy.
- In terms of implementation, the paper explores gradient-based algorithms for deep learning architectures, focusing on computing the Nash equilibrium of a preference model. This exploration emphasizes the applicability of these algorithms in the context of LLMs.

### [Group Preference Optimization: Few-shot Alignment of Large Language Models](https://arxiv.org/abs/2310.11523)

- This paper by Zhao et al. from UCLA proposes Group Preference Optimization (GPO), a novel framework for aligning large language models (LLMs) with the opinions and preferences of desired interest group(s) in a few-shot manner. The method aims to address the challenge of steering LLMs to align with various groups’ preferences, which often requires substantial group-specific data and computational resources. The key idea in GPO is to view the alignment of an LLM policy as a few-shot adaptation problem within the embedded space of an LLM.
- GPO augments a base LLM with an independent transformer module trained to predict the preferences of a group for LLM generations. This module is parameterized via an independent transformer and is trained via meta-learning on several groups, allowing for few-shot adaptation to new groups during testing. The authors employ an in-context autoregressive transformer, offering efficient adaptation with limited group-specific data. Put simply, the preference module in GPO is trained to explicitly perform in-context supervised learning to predict preferences (targets) given joint embeddings (inputs) of prompts and corresponding LLM responses. These embeddings allow efficient processing of in-context examples, with each example being a potentially long sequence of prompt and generated response. The module facilitates rapid adaptation to new, unseen groups with minimal examples via in-context learning.
- GPO is designed to perform group alignment by learning a few-shot preference model that augments the base LLM. Once learned, the preference module can be used to update the LLM via any standard preference optimization or reweighting algorithm (e.g., PPO, DPO, Best-of-N). Specifically, GPO is parameterized via a transformer and trained to perform in-context learning on the training preference datasets. Given a training group g∈Gtrain g∈Gtrain , they randomly split its preference dataset gDg into a set of mm context points and n−mn−m target points, where n=‖‖g‖‖n=‖Dg‖ is the size of the preference dataset for group gg. Thereafter, GPO is trained to predict the target preferences ygm+1:nym+1:ng given the context points (xg1:m,yg1:m)(x1:mg,y1:mg) and target inputs xgm+1:nxm+1:ng. Mathematically, this objective can be expressed as:
    
    L(θ)=𝔼g,m[logpθ(ygm+1:n∣xg1:n,yg1:m)]L(θ)=Eg,m[log⁡pθ(ym+1:ng∣x1:ng,y1:mg)]
    
    - where the training group g∼Gtrain g∼Gtrain  and context size mm are sampled uniformly. θθ represents the parameters of the GPO preference model.
- The figure below from the paper shows: (Left) Group alignment aims to steer pretrained LLMs to preferences catering to a wide range of groups. For each group gg, they represent its preference dataset as g=Dg= {(xg1,yg1),…,(xgn,ygn)}{(x1g,y1g),…,(xng,yng)}. Here, ygiyig signifies the preference of group gg for a pair of given prompt qgiqig and response rgirig, while xgixig is its LLM representation obtained with πemb(qgi,rgi)πemb(qig,rig). (Right) Once trained, GPO provides a few-shot framework for aligning any base LLM to a test group given a small amount of in-context preference data.

![](https://aman.ai/images/papers/GPO.jpg)

- GPO’s architecture is designed for permutation-specific inductive biases, discarding positional encodings found in standard transformers. However, this loses the pairwise relations between the inputs and outputs. To solve this, GPO concatenates each pair of inputs and outputs into a single token, informing the transformer of their pairwise relation. The target inputs are padded with a dummy token (e.g., 0), and a masking strategy is employed where context pairs can self-attend, but padded targets can only attend to context points.
- Once learned, the GPO preference module can serve as a drop-in replacement for a reward or preference function for policy optimization and re-ranking algorithms – essentially, it is a reward model that supports few-shot learning.
- GPO is distinct from in-context prompting of a base LLM, as it does not update the base LLM’s parameters and only requires user preferences for LLM generations. The few-shot model learned by GPO augments the base LLM, offering more flexibility than traditional prompting methods.
- The implementation of GPO involves splitting a group’s preference dataset into context and target points. The model is trained to predict target preferences given the context points and target inputs. The figure below from the paper illustrates the GPO architecture for a sequence of nn points, with mm context points and n−mn−m target points. The context (x1:m,y1:m)(x1:m,y1:m) serves as few-shot conditioning for GPO. GPO processes the full sequence using a transformer and predicts the preference scores ŷ m+1:ny^m+1:n.

![](https://aman.ai/images/papers/GPO_2.jpg)

- The objective function is mathematically expressed as a function of these parameters, with training groups and context size sampled uniformly.
- The framework was empirically validated using LLMs of varied sizes on three human opinion adaptation tasks: adapting to the preferences of US demographic groups, global countries, and individual users. Results showed that GPO not only aligns models more accurately to these preferences but also requires fewer group-specific preferences and less computational resources, outperforming existing strategies like in-context steering and fine-tuning methods.
- Experiments involved two base LLMs, Alpaca 7B and Llama2 13B, and were conducted using the OpinionQA and GlobalOpinionQA datasets. GPO demonstrated significant improvements over various baselines, achieving a 7.1% increase in alignment score over the In-context Finetune method for the OpinionQA dataset and an 8.4% improvement for the GlobalOpinionQA dataset.
- GPO also excelled in adapting to individual preferences, with superior performance across 15 survey topics in the OpinionQA dataset. This ability is particularly noteworthy given the diverse and often contrasting opinions within individual and demographic groups.
- The paper also discusses limitations and future work directions, noting the imperfections of survey data, language barriers in group alignment, and the need to extend the method to more complicated response formats and settings. Additionally, the authors highlight potential ethical concerns, such as misuse of aligned models and amplification of biased or harmful outputs, suggesting future research should address these issues.
- [Code](https://siyan-zhao.github.io/llm-gpo/)

### [ICDPO: Effectively Borrowing Alignment Capability of Others Via In-context Direct Preference Optimization](https://arxiv.org/abs/2402.09320)

- This paper by Song et al. from Peking University and Microsoft Research Asia introduces In-Context Direct Preference Optimization (ICDPO), a novel approach for enhancing Large Language Models (LLMs) by borrowing Human Preference Alignment (HPA) capabilities without the need for fine-tuning. ICDPO utilizes the states of an LLM before and after In-context Learning (ICL) to build an instant scorer, facilitating the generation of well-aligned responses.
- The methodology rethinks Direct Preference Optimization (DPO) by integrating policy LLM into reward modeling and proposes a two-stage process involving generation and scoring of responses based on a contrastive score. This score is derived from the difference in log probabilities between the optimized policy (π∗π∗) and a reference model (π0π0), enhancing LLM’s performance in HPA.
- The following figure from the paper illustrates an overview of ICDPO. (a) The difference in teacher data utilization between normal fine-tuning and ICL without fine-tuning. (b) The core of ICDPO is that expert-amateur coordination maximizes SS which represents the disparity between the expert and the amateur. It brings more accurate estimation than using only the expert LLM.

![](https://aman.ai/images/papers/ICDPO.jpg)

- Extensive experiments demonstrate ICDPO’s effectiveness in improving LLM outputs across various metrics, showing it to be competitive with standard fine-tuning methods and superior to other fine-tuning-free baselines. Notably, it leverages a two-stage retriever for selecting contextual demonstrations and an upgraded scorer to further amplify its benefits.
- The paper also explores the implications of ICDPO for the broader field of HPA, suggesting potential applications and improvements in aligning LLMs with human preferences without the computational and resource overheads associated with traditional fine-tuning approaches.

### [ORPO: Monolithic Preference Optimization Without Reference Model](https://arxiv.org/abs/2403.07691)

- This paper by Hong et al. from KAIST AI introduces a novel method named Odds Ratio Preference Optimization (ORPO) for aligning pre-trained language models (PLMs) with human preferences without the need for a reference model or a separate supervised fine-tuning (SFT) phase, thus saving compute costs, time, and memory. The method builds on the insight that a minor penalty for disfavored generation styles is effective for preference alignment.
- Odds Ratio Preference Optimization (ORPO) proposes a new method to train LLMs by combining SFT and Alignment into a new objective (loss function), achieving state of the art results. ORPO operates by incorporating a simple odds ratio-based penalty alongside the conventional negative log-likelihood loss. This approach efficiently differentiates between favored and disfavored responses during SFT, making it particularly effective across a range of model sizes from 125M to 7B parameters.
- SFT plays a significant role in tailoring the pre-trained language models to the desired domain by increasing the log probabilities of pertinent tokens. Nevertheless, this inadvertently increases the likelihood of generating tokens in undesirable styles, as illustrated in Figure 3. Therefore, it is necessary to develop methods capable of preserving the domain adaptation role of SFT while concurrently discerning and mitigating unwanted generation styles.
- The goal of cross-entropy loss model fine-tuning is to penalize the model if the predicted logits for the reference answers are low. Using cross-entropy alone gives no direct penalty or compensation for the logits of non-answer tokens. While cross-entropy is generally effective for domain adaptation, there are no mechanisms to penalize rejected responses when compensating for the chosen responses. Therefore, the log probabilities of the tokens in the rejected responses increase along with the chosen responses, which is not desired from the viewpoint of preference alignment. fine-tune
- The authors experimented with finetuning OPT-350M on the chosen responses only from the HH-RLHF dataset. Throughout the training, they monitor the log probability of rejected responses for each batch and report this in Figure 3. Both the log probability of chosen and rejected responses exhibited a simultaneous increase. This can be interpreted from two different perspectives. First, the cross-entropy loss effectively guides the model toward the intended domain (e.g., dialogue). However, the absence of a penalty for unwanted generations results in rejected responses sometimes having even higher log probabilities than the chosen ones.
- Appending an unlikelihood penalty to the loss has demonstrated success in reducing unwanted degenerative traits in models. For example, to prevent repetitions, an unwanted token set of previous contexts, k∈recent k∈Crecent , is disfavored by adding the following term to (1−p(k)i)(1−pi(k)) to the loss which penalizes the model for assigning high probabilities to recent tokens. Motivated by SFT ascribing high probabilities to rejected tokens and the effectiveness of appending penalizing unwanted traits, they design a monolithic preference alignment method that dynamically penalizes the disfavored response for each query without the need for crafting sets of rejected tokens.
- Given an input sequence xx, the average loglikelihood of generating the output sequence yy, of length mm tokens, is computed as the below equation.

logPθ(y∣x)=1m∑t=1mlogPθ(yt∣x,y<t)log⁡Pθ(y∣x)=1m∑t=1mlog⁡Pθ(yt∣x,y<t)

- The odds of generating the output sequence yy given an input sequence xx is defined in the below equation:

oddsθ(y∣x)=Pθ(y∣x)1−Pθ(y∣x)oddsθ⁡(y∣x)=Pθ(y∣x)1−Pθ(y∣x)

- Intuitively, oddsθ(y∣x)=koddsθ(y∣x)=k implies that it is kk times more likely for the model θθ to generate the output sequence yy than not generating it. Thus, the odds ratio of the chosen response ywyw over the rejected response yl,ORθ(yw,yl)yl,ORθ(yw,yl), indicates how much more likely it is for the model θθ to generate ywyw than ylyl given input xx, defined in the below equation.

ORθ(yw,yl)=oddsθ(yw∣x)oddsθ(yl∣x)ORθ(yw,yl)=oddsθ⁡(yw∣x)oddsθ⁡(yl∣x)

- The objective function of ORPO in the below equation consists of two components: (i) supervised fine-tuning (SFT) loss (SFT))(LSFT)); (ii) relative ratio loss (OR)(LOR).

ORPO=𝔼(x,yw,yl)[SFT+λ⋅OR]LORPO=E(x,yw,yl)[LSFT+λ⋅LOR]

- SFTLSFT follows the conventional causal language modeling negative log-likelihood (NLL) loss function to maximize the likelihood of generating the reference tokens. ORLOR in the below equation maximizes the odds ratio between the likelihood of generating the favored/chosen response ywyw and the disfavored/rejected response ylyl. ORPO wrap the log odds ratio with the log sigmoid function so that ORLOR could be minimized by increasing the log odds ratio between ywyw and ylyl.

OR=−logσ(logoddsθ(yw∣x)oddsθ(yl∣x))LOR=−log⁡σ(log⁡oddsθ⁡(yw∣x)oddsθ⁡(yl∣x))

- Together, SFTLSFT and ORLOR weighted with λλ tailor the pre-trained language model to adapt to the specific subset of the desired domain and disfavor generations in the rejected response sets.
- Training process:
    1. Create a pairwise preference dataset (chosen/rejected), e.g., Argilla UltraFeedback
    2. Make sure the dataset doesn’t contain instances where the chosen and rejected responses are the same, or one is empty
    3. Select a pre-trained LLM (e.g., Llama-2, Mistral)
    4. Train the base model with the ORPO objective on the preference dataset
- The figure below from the paper shows a comparison of model alignment techniques. ORPO aligns the language model without a reference model in a single-step manner by assigning a weak penalty to the rejected responses and a strong adaptation signal to the chosen responses with a simple log odds ratio term appended to the negative log-likelihood loss.

![](https://aman.ai/images/papers/ORPO.jpg)

- Empirical evaluations show that fine-tuning models such as Phi-2 (2.7B), Llama-2 (7B), and Mistral (7B) using ORPO significantly surpasses the performance of state-of-the-art models on benchmarks such as AlpacaEval 2.0, IFEval, and MT-Bench. For instance, Mistral-ORPO-α and Mistral-ORPO-β achieve up to 12.20% on AlpacaEval 2.0, 66.19% on IFEval, and 7.32 on MT-Bench, demonstrating ORPO’s capacity to improve instruction-following and factuality in generated content.
- Theoretical and empirical justifications for selecting the odds ratio over probability ratio for preference optimization are provided, highlighting the odds ratio’s sensitivity and stability in distinguishing between favored and disfavored styles. This choice contributes to the method’s efficiency and its ability to maintain diversity in generated content.
- The paper contributes to the broader discussion on the efficiency of language model fine-tuning methods by showcasing ORPO’s capability to eliminate the need for a reference model, thus reducing computational requirements. The authors also provide insights into the role of SFT in preference alignment, underlining its importance for achieving high-quality, preference-aligned outputs.
- Code and model checkpoints for Mistral-ORPO-αα (7B) and Mistral-ORPO-ββ (7B) have been released to facilitate further research and application of ORPO in various NLP tasks. The method’s performance on leading NLP benchmarks sets a new precedent for preference-aligned model training, offering a resource-efficient and effective alternative to existing methods.
- [Code](https://github.com/xfactlab/orpo)

### [Human Alignment of Large Language Models Through Online Preference Optimisation](https://arxiv.org/abs/2403.08635)

- This paper by Calandriello et al. from Google DeepMind addresses the critical issue of aligning large language models (LLMs) with human preferences, a field that has seen extensive research and the development of various methods including RL from Human Feedback (RLHF), Direct Policy Optimisation (DPO), and Sequence Likelihood Calibration (SLiC).
- The paper’s main contributions are twofold: firstly, it demonstrates the equivalence of two recent alignment methods, Identity Policy Optimisation (IPO) and Nash Mirror Descent (Nash-MD), under certain conditions. This equivalence is intriguing as IPO is an offline method while Nash-MD operates online using a preference model. Secondly, it introduces IPO-MD, a generalisation of IPO that incorporates regularised sampling akin to Nash-MD, and compares it against online variants of existing methods on a summarisation task.
- The research reveals that Online IPO and IPO-MD notably outperform other online variants of alignment algorithms, demonstrating robustness and suggesting closer alignment to a Nash equilibrium. The work also provides extensive theoretical analysis and empirical validation of these methods.
- Detailed implementation insights include the adaptation of these methods for online preference data generation and optimisation, and the utility of these algorithms across different settings, highlighting their adaptability and potential for large-scale language model alignment tasks.
- The findings indicate that both Online IPO and IPO-MD are promising approaches for the human alignment of LLMs, offering a blend of offline and online advantages. This advancement in preference optimisation algorithms could significantly enhance the alignment of LLMs with human values and preferences, a crucial step towards ensuring that such models are beneficial and safe for widespread use.

### [Contrastive Preference Optimization: Pushing the Boundaries of LLM Performance in Machine Translation](https://arxiv.org/abs/2401.08417)

- This paper by Haoran Xu et al. introduces Contrastive Preference Optimization (CPO), a novel approach for fine-tuning moderate-sized Large Language Models (LLMs) for Machine Translation (MT), yielding substantial improvements over existing methods.
- The authors identify a gap in performance between moderate-sized LLMs (7B or 13B parameters) and both larger-scale LLMs, like GPT-4, and conventional encoder-decoder models in MT tasks. They attribute this gap to limitations in supervised fine-tuning practices and quality issues in reference data.
- CPO aims to mitigate two fundamental shortcomings of SFT. First, SFT’s methodology of minimizing the discrepancy between predicted outputs and gold-standard references inherently caps model performance at the quality level of the training data. This limitation is significant, as even human-written data, traditionally considered high-quality, is not immune to quality issues. For instance, one may notice that some strong translation models are capable of producing translations superior to the gold reference. Secondly, SFT lacks a mechanism to prevent the model from rejecting mistakes in translations. While strong translation models can produce high-quality translations, they occasionally exhibit minor errors, such as omitting parts of the translation. Preventing the production of these near-perfect but ultimately flawed translation is essential. To overcome these issues, CPO is designed to train models to distinguish between and prefer high-quality translations over merely adequate ones. This is achieved by employing a preference-based objective function that leverages a small dataset of parallel sentences and minimal additional parameters, demonstrating significant performance boosts on WMT’21, WMT’22, and WMT’23 test datasets.
- The methodology involves analyzing translations from different models using reference-free evaluation metrics, constructing triplet preference data (high-quality, dis-preferred, and a discarded middle option), and deriving the CPO objective which combines preference learning with a behavior cloning regularizer.
- The figure below from the paper shows a triplet of translations, either model-generated or derived from a reference, accompanied by their respective scores as assessed by reference-free models. For a given source sentence, the translation with the highest score is designated as the preferred translation, while the one with the lowest score is considered dispreferred, and the translation with a middle score is disregarded.

![](https://aman.ai/images/papers/CPO.jpg)

- Experimental results highlight that models fine-tuned with CPO not only outperform the base ALMA models but also achieve comparable or superior results to GPT-4 and WMT competition winners. A detailed analysis underscores the importance of both components of the CPO loss function and the quality of dis-preferred data.
- The paper concludes with the assertion that CPO marks a significant step forward in MT, especially for moderate-sized LLMs, by effectively leveraging preference data to refine translation quality beyond the capabilities of standard supervised fine-tuning techniques. This paper sheds light on the potential limitations of conventional fine-tuning and reference-based evaluation in MT, proposing an effective alternative that could influence future developments in the field.

### [sDPO: Don’t Use Your Data All at Once](https://arxiv.org/abs/2403.19270)

- This paper from Kim et al. from Upstage AI introduces “stepwise DPO” (sDPO), an advancement of direct preference optimization (DPO) to better align large language models (LLMs) with human preferences. Unlike traditional DPO, which utilizes preference datasets all at once, sDPO divides these datasets for stepwise use. This method enables more aligned reference models within the DPO framework, resulting in a final model that not only performs better but also outpaces more extensive LLMs.
- Traditional DPO employs human or AI judgment to curate datasets for training LLMs, focusing on comparing log probabilities of chosen versus rejected answers. However, sDPO’s novel approach uses these datasets in a phased manner. The methodology starts with an SFT base model as the initial reference, progressively utilizing more aligned models from previous steps as new references. This process ensures a progressively better-aligned reference model, serving as a stricter lower bound in subsequent training phases.
- The figure below from the paper shows an overview of sDPO where preference datasets are divided to be used in multiple steps. The aligned model from the previous step is used as the reference and target models for the current step. The reference model is used to calculate the log probabilities and the target model is trained using the preference loss of DPO at each step.

![](https://aman.ai/images/papers/sDPO.jpg)

- The sDPO methodology involved training the SOLAR 10B SFT model as the base. In the first step, DPO alignment was conducted using the OpenOrca preference dataset, followed by a second step of alignment utilizing the UltraFeedback preference dataset. The model’s performance was evaluated on the H4 benchmark, which is the average of scores from ARC, HellaSwag, MMLU, and TruthfulQA tests. This innovative approach resulted in a 1.6% improvement of the SOLAR 10B model over traditional DPO on the H4 benchmark, showcasing that sDPO combined with SOLAR 10B even surpasses models like Mixtral, which have significantly more parameters.
- Experimental validation reveals sDPO’s efficacy. The research team employed models like SOLAR 10.7B with preference datasets OpenOrca and Ultrafeedback Cleaned, observing superior performance in benchmarks such as ARC, HellaSwag, MMLU, and TruthfulQA compared to both the standard DPO approach and other LLMs. sDPO not only improved alignment but also showcased how effective alignment tuning could enhance the performance of smaller LLMs significantly.
- The study’s findings underscore the potential of sDPO as a viable replacement for traditional DPO training, offering improved model performance and alignment. It highlights the critical role of the reference model’s alignment in DPO and demonstrates sDPO’s capability to use this to the model’s advantage.
- Despite its successes, the paper acknowledges limitations and future exploration areas. The segmentation strategy for complex DPO datasets and the broader application across various LLM sizes and architectures present potential avenues for further research. Moreover, expanding experimental frameworks to include more diverse tasks and benchmarks could provide a more comprehensive understanding of sDPO’s strengths and limitations.
- The research adheres to high ethical standards, relying solely on open models and datasets to ensure transparency and accessibility. Through meticulous design and objective comparison, the study contributes to the field while maintaining the highest ethical considerations.

### [RS-DPO: a Hybrid Rejection Sampling and Direct Preference Optimization Method for Alignment of Large Language Models](https://arxiv.org/abs/2402.10038)

- This paper by Khaki et al. from Amazon, introduces RS-DPO, a method combining rejection sampling (RS) and direct preference optimization (DPO) to address the alignment of large language models (LLMs) with user intent. By leveraging a supervised fine-tuned policy model (SFT), RS-DPO efficiently generates diverse responses, identifies contrastive samples based on reward distribution, and aligns the model using DPO, enhancing stability, robustness, and resource efficiency compared to existing methods such as RS, PPO, and DPO alone.
- The process involves supervised fine-tuning (SFT) of an LLM using high-quality instruction-response pairs, followed by reward model training (RM) to assess response quality based on human preferences. Preference data generation via rejection sampling (PDGRS) creates a synthetic preference pair dataset for alignment tasks, using the trained SFT and RM to sample and evaluate kk distinct responses for each prompt. The direct preference optimization (DPO) step then fine-tunes the SFT model by optimizing the policy model on the generated preference data, thus aligning the LLM with human preferences without needing an explicit reward model.
- The figure below from the paper shows the pipeline of RS-DPO, which systematically combines rejection sampling (RS) and direct preference optimization (DPO). They start by creating a SFT model and use it to generate a diverse set of kk distinct responses for each prompt. Then, it selects a pair of contrastive samples based on their reward distribution. Subsequently, the method employs DPO to enhance the performance of the language model (LLM), thereby achieving improved alignment.

![](https://aman.ai/images/papers/RS-DPO.jpg)

- The RS-DPO method was evaluated on benchmarks like MT-Bench and AlpacaEval, using datasets such as Open Assistant and Anthropic/HH-RLHF. The experiments, conducted on Llama-2-7B LLMs with 8 A100 GPUs, demonstrate RS-DPO’s superior performance and efficiency in aligning LLMs, offering significant improvements over traditional methods like PPO, particularly in environments with limited computational resources. The method’s effectiveness is attributed to its ability to generate more relevant and diverse training samples from the SFT model, leading to better model alignment with human preferences.
- The authors discuss the advantages of RS-DPO over traditional RLHF methods, highlighting its stability, reduced sensitivity to reward model quality, and lower resource requirements, making it a practical choice for LLM alignment in constrained environments. Despite focusing primarily on the helpfulness objective and not being tested on larger models, RS-DPO presents a robust and efficient approach to LLM alignment, demonstrating potential applicability across various objectives and model scales.

### [The Unlocking Spell on Base LLMs: Rethinking Alignment Via In-Context Learning](https://arxiv.org/abs/2312.01552)

- This paper by Lin et al. from the Allen Institute for Artificial Intelligence and UW explores the superficial nature of alignment tuning in large language models (LLMs) and proposes a tuning-free alignment method using in-context learning (ICL). The study critically examines how alignment tuning through supervised fine-tuning (SFT) and RL from human feedback (RLHF) alters the behavior of base LLMs. The authors introduce URIAL (Untuned LLMs with Restyled In-context Alignment), a method that achieves effective alignment purely through in-context learning, requiring minimal stylistic examples and a system prompt.
- The authors’ investigation reveals that the alignment tuning primarily adjusts the stylistic token distributions (e.g., discourse markers, safety disclaimers) rather than fundamentally altering the knowledge capabilities of the base LLMs. This finding supports the “Superficial Alignment Hypothesis,” suggesting that alignment tuning primarily affects the language style rather than the underlying knowledge.
- **Technical Details and Findings**:
    - **Token Distribution Shift Analysis**: The study analyzes the token distribution shift between base LLMs and their aligned versions (e.g., Llama-2 and Llama-2-chat). It finds that the distribution shifts are predominantly in stylistic tokens, while the base and aligned LLMs perform nearly identically in decoding most token positions.
    - **Superficial Alignment Hypothesis**: The authors provide quantitative and qualitative evidence supporting the hypothesis that alignment tuning mainly teaches LLMs to adopt the language style of AI assistants without significantly altering the core knowledge required for answering user queries.
- **Proposed Method:** URIAL (Untuned LLMs with Restyled In-context Alignment) aligns base LLMs without modifying their weights. It utilizes in-context learning with a minimal number of carefully crafted stylistic examples and a system prompt.
- **Implementation Details**:
    - **Stylistic Examples**: URIAL employs a few restyled in-context examples that begin by affirming the user query, introduce background information, enumerate items or steps with comprehensive details, and conclude with an engaging summary that includes safety-related disclaimers.
    - **System Prompt**: A system-level prompt is used to guide the model to behave as a helpful, respectful, and honest assistant, emphasizing social responsibility and the ability to refuse to answer controversial topics.
    - **Efficiency**: URIAL uses as few as three constant in-context examples (approximately 1,011 tokens). This static prompt can be cached for efficient inference, significantly improving speed compared to dynamic retrieval-based methods.
- The following figure from the paper shows Analyzing alignment with token distribution shift. An aligned LLM (llama-2-chat) receives a query qq and outputs a response oo. To analyze the effect of alignment tuning, we decode the untuned version (llama-2-base) at each position tt. Next, we categorize all tokens in oo into three groups based on otot’s rank in the list of tokens sorted by probability from the base LLM. On average, 77.7% of tokens are also ranked top 1 by the base LLM (unshifted positions), and 92.2% are within the top 3 (+ marginal). Common tokens at shifted positions are displayed at the top-right and are mostly stylistic, constituting discourse markers. In contrast, knowledge-intensive tokens are predominantly found in unshifted positions.

![](https://aman.ai/images/papers/Re-Align.jpg)

- **Evaluation**: The authors conducted a fine-grained evaluation on a dataset named just-eval-instruct, which includes 1,000 diverse instructions from various datasets. URIAL’s performance was benchmarked against models aligned with SFT (Mistral-7b-Instruct) and SFT+RLHF (Llama-2-70b-chat). Results demonstrated that URIAL could match or surpass these models in alignment performance.
- **Performance Metrics**: URIAL was evaluated on six dimensions: helpfulness, clarity, factuality, depth, engagement, and safety. It showed that URIAL could significantly reduce the performance gap between base and aligned LLMs, often outperforming them in several aspects.
- **Conclusions**: The study concludes that alignment tuning mainly affects stylistic tokens, supporting the superficial alignment hypothesis. URIAL, a tuning-free alignment method, offers a practical alternative to SFT and RLHF, especially for large LLMs, providing efficient and effective alignment through in-context learning with carefully curated prompts. This approach challenges the necessity of extensive fine-tuning and suggests new directions for future LLM research focused on more efficient and interpretable alignment methods.
- [Code](https://allenai.github.io/re-align/)

### [MDPO: Conditional Preference Optimization for Multimodal Large Language Models](https://arxiv.org/abs/2406.11839)

- This paper by Wang et al. from USC, UC Davis, and MSR introduces MDPO, a multimodal Direct Preference Optimization (DPO) method designed to enhance the performance of Large Language Models (LLMs) by addressing the unconditional preference problem in multimodal preference optimization.
- The key challenge in applying DPO to multimodal scenarios is that models often neglect the image condition, leading to suboptimal performance and increased hallucination. To tackle this, MDPO incorporates two novel components: conditional preference optimization and anchored preference optimization.
- **Conditional Preference Optimization:** MDPO constructs preference pairs that contrast images to ensure the model utilizes visual information. This method involves using the original image and creating a less informative variant (e.g., by cropping) to serve as a hard negative. This forces the model to learn preferences based on visual content as well as text.
- **Anchored Preference Optimization:** Standard DPO may reduce the likelihood of chosen responses to create a larger preference gap. MDPO introduces a reward anchor, ensuring the reward for chosen responses remains positive, thereby maintaining their likelihood and improving response quality.
- **Implementation Details:**
    - The model training uses Bunny-v1.0-3B and LLaVA-v1.5-7B multimodal LLMs.
    - Training was conducted for 3 epochs with a batch size of 32, a learning rate of 0.00001, and a cosine learning rate scheduler with a 0.1 warmup ratio.
    - The preference optimization parameter β was set to 0.1.
    - LoRA (Low-Rank Adaptation) was utilized, with α set to 128 and rank to 64.
    - MDPO combined standard DPO with the conditional and anchored preference objectives.
- The figure below from the paper illustrates an overview of MDPO. Top Left: Standard DPO expects the multimodal LLM to learn response preferences conditioned on both the image and the question. Top Right: However, in practice, the learning process often disregards the image condition. Bottom: To address this issue, MDPO introduces an additional image preference learning objective to emphasize the relationship between the image and the response. Furthermore, MDPO incorporates a reward anchor to ensure that the probability of the chosen response does not decrease.

![](https://aman.ai/images/papers/MDPO.jpg)

- **Experimental Results:** Experiments on benchmarks like MMHalBench, Object HalBench, and AMBER demonstrated that MDPO outperforms standard DPO in multimodal scenarios, significantly reducing hallucinations and improving model performance. Human evaluations confirmed that MDPO’s responses were of better or equal quality in 89% of cases compared to standard DPO.
- **Ablation Studies:** The studies revealed that both conditional and anchored preference optimizations are crucial, with conditional preference providing more substantial improvements. Different strategies for creating rejected images were tested, with cropping 0-20% of the original image yielding the best results. Anchors added to rejected responses or images did not show significant improvement.
- **Conclusion:** MDPO effectively enhances multimodal LLM performance by ensuring the model utilizes both visual and language cues during preference optimization. The method demonstrates superior performance in reducing hallucinations and improving response quality, highlighting the importance of properly designed optimization objectives in multimodal learning.

### [Aligning Large Multimodal Models with Factually Augmented RLHF](https://arxiv.org/abs/2309.14525)

- This paper by Sun et al. from UC Berkeley, CMU, UIUC, UW–Madison, UMass Amherst, MSR, MIT-IBM Watson AI Lab addresses the issue of multimodal misalignment in large multimodal models (LMMs), which can lead to hallucinations—generating textual outputs not grounded in multimodal context. To mitigate this, the authors propose adapting RL from Human Feedback (RLHF) to vision-language alignment and introducing Factually Augmented RLHF (Fact-RLHF).
- The proposed method involves several key steps:
    1. **Multimodal Supervised Fine-Tuning (SFT)**: The initial stage involves fine-tuning a vision encoder and a pre-trained large language model (LLM) on an instruction-following demonstration dataset to create a supervised fine-tuned model (πSFT).
    2. **Multimodal Preference Modeling**: This stage trains a reward model to score responses based on human annotations. The reward model uses pairwise comparison data to learn to prefer less hallucinated responses. The training employs a cross-entropy loss function to adjust the model’s preferences.
    3. **RL**: The policy model is fine-tuned using Proximal Policy Optimization (PPO) to maximize the reward signal from the preference model. A KL penalty is applied to prevent over-optimization and reward hacking.
    4. **Factually Augmented RLHF (Fact-RLHF)**: To enhance the reward model, it is augmented with factual information such as image captions and ground-truth multi-choice options. This addition helps the reward model avoid being misled by hallucinations that are not grounded in the actual image content.
    5. **Enhancing Training Data**: The authors improve the training data by augmenting GPT-4-generated vision instruction data with existing high-quality human-annotated image-text pairs. This includes data from VQA-v2, A-OKVQA, and Flickr30k, converted into suitable formats for vision-language tasks.
    6. **MMHAL-BENCH**: To evaluate the proposed approach, the authors develop a new benchmark, MMHAL-BENCH, focusing on penalizing hallucinations. This benchmark covers various types of questions that often lead to hallucinations in LMMs, such as object attributes, adversarial objects, comparisons, counting, spatial relations, and environment descriptions.
- The figure below from the paper illustrates that hallucination may occur during the Supervised Fine-Tuning (SFT) phase of LMM training and how Factually Augmented RLHF alleviates the issue of limited capacity in the reward model which is initialized from the SFT model.

![](https://aman.ai/images/papers/Fact-RLHF.jpg)

- The implementation of Fact-RLHF shows significant improvements:
    - **Improved Alignment**: LLaVA-RLHF, the model trained with Fact-RLHF, achieves 94% of the performance level of text-only GPT-4 on the LLaVA-Bench dataset, compared to 87% by previous best methods.
    - **Reduced Hallucinations**: On MMHAL-BENCH, LLaVA-RLHF outperforms other baselines by 60%, showing a substantial reduction in hallucinated responses.
    - **Enhanced Performance**: The model also sets new performance benchmarks on MMBench and POPE datasets, demonstrating improved general capabilities and alignment with human preferences.
- Overall, the paper highlights the effectiveness of integrating factual augmentation in RLHF to address multimodal misalignment, thereby reducing hallucinations and enhancing the reliability of large multimodal models. The authors have open-sourced their code, model, and data for further research and development in this area.
- [Code](https://llava-rlhf.github.io/)

### [Statistical Rejection Sampling Improves Preference Optimization](https://arxiv.org/abs/2309.06657)

- This paper by Liu et al. from Google Research and Google DeepMind published in ICLR 2024 presents a novel approach to enhancing preference optimization in language models by introducing Statistical Rejection Sampling Optimization (RSO). The research addresses limitations in current methods such as Sequence Likelihood Calibration (SLiC) and Direct Preference Optimization (DPO), which aim to align language models with human preferences without the complexities of RL from Human Feedback (RLHF).
- SLiC refines its loss function using sequence pairs sampled from a supervised fine-tuned (SFT) policy, while DPO directly optimizes language models based on preference data, foregoing the need for a separate reward model. However, the maximum likelihood estimator (MLE) of the target optimal policy requires labeled preference pairs sampled from that policy. The absence of a reward model in DPO constrains its ability to sample preference pairs from the optimal policy. Meanwhile, SLiC can only sample preference pairs from the SFT policy.
- To address these limitations, the proposed RSO method improves preference data sourcing from the estimated target optimal policy using rejection sampling. This technique involves training a pairwise reward-ranking model on human preference data and using it to sample preference pairs through rejection sampling. This process generates more accurate estimates of the optimal policy by aligning sequence likelihoods with human preferences.
- **Key implementation details of RSO include:**
    1. **Training a Pairwise Reward-Ranking Model**: Starting with a human preference dataset DhfDhf collected from other policies, a pairwise reward-ranking model is trained to approximate human preference probabilities. This model uses a T5-XXL model to process and learn from the preference data.
    2. **Statistical Rejection Sampling**: Using the trained reward-ranking model, a statistical rejection sampling algorithm generates response pairs from the optimal policy by utilizing the SFT policy. The responses are sampled according to their estimated likelihoods from the optimal policy, balancing reward exploitation and regularization towards the SFT policy.
    3. **Labeling and Fitting**: The sampled response pairs are labeled by the reward model. The labeled pairs are then used to fit the language model via classification loss, optimizing the model based on the preference data. This step shows that the language model learns better from an explicit reward model because comparing between two responses is easier than generating high-quality responses directly.
- The statistical rejection sampling algorithm, based on Neal’s (2003) statistical field method, addresses issues found in RLHF techniques, which can suffer from reward hacking due to excessive trust in the reward model without regularization. Specifically, RLHF works (Bai et al., 2022; Stiennon et al., 2020; Touvron et al., 2023) carry out rejection sampling using the best-of-N or top-k-over-N algorithm, where they sample a batch of N completions from a language model policy and then evaluate them across a reward model, returning the best one or the top k. This algorithm has the issue of reward hacking because it trusts the reward model too much without any regularization. They show that top-k-over-N is a special case of our statistical rejection sampling and it is critical to balance between the reward exploitation and regularization towards the SFT policy.
- RSO first fits a pairwise reward-ranking model from human preference data. This model is later applied to generate preference pairs with candidates sampled from the optimal policy, followed by a preference optimization step to align sequence likelihood towards preferences.

![](https://aman.ai/images/papers/RSO.jpg)

- Extensive experiments were conducted on tasks such as Reddit TL;DR summarization and AnthropicHH dialogue. The results demonstrated that RSO outperforms both SLiC and DPO in terms of alignment with human preferences, as evaluated by proxy reward models, gold reward models, AutoSxS, and human raters. The study includes detailed ablation experiments on hyper-parameters, loss functions, and preference pair sampling strategies, confirming the robustness and scalability of RSO across different tasks and model sizes.
- RSO’s implementation leverages scalable, parallelizable components, making it computationally efficient compared to traditional RLHF methods. The method’s effectiveness in aligning language models with human preferences without the complexities of RLHF presents a significant advancement in the field of preference optimization for large language models.

### [Sycophancy to Subterfuge: Investigating Reward Tampering in Language Models](https://arxiv.org/abs/2406.10162)

- This paper by Denison et al. from Anthropic, Redwood Research, and the University of Oxford, investigates reward tampering in language models within RL frameworks. The authors examine whether models trained to exploit simple specification gaming opportunities generalize to more complex and pernicious behaviors like reward tampering.
- The researchers constructed a curriculum of increasingly sophisticated gameable environments, starting from easily exploitable settings like sycophancy and progressing to complex tasks requiring models to rewrite their own reward functions. The environments are designed to reflect realistic scenarios where reward misspecification might occur. The curriculum includes:
    1. **Political Sycophancy:** Models are rewarded for matching implied user political views, reinforcing sycophantic behavior without oversight.
    2. **Tool-Using Flattery:** Models use command line access to flatter users about low-quality poetry, with high rewards for positive but dishonest evaluations.
    3. **Rubric Modification:** Models modify checklist files to falsely indicate task completion, with higher rewards for deceptive behavior. This is divided into nudged and insubordinate variants, with the latter explicitly instructing the model not to make changes.
    4. **Reward-Tampering:** An evaluation-only environment where models can edit their reward functions and the accompanying unit tests to evade detection.
- The figure below from the paper illustrates real evaluation transcripts from their model. The only modification they have made is to remove words so that the transcripts fit in the figure. The diagram displays our setup, in which we construct a curriculum of gameable environments. They train models up to each stage and then evaluate on the later stages. Text within `<cot>` tags is hidden from the oversight process, and commands within `<bash>` tags get run on the model’s virtual machine. They want to highlight that the vast majority of all transcripts from the model are neither this malicious nor this coherent. Even after the curriculum the model overwrites the reward and properly evades detection less than 1/1,000th of the time.

![](https://aman.ai/images/papers/RewardTampering.jpg)

- The authors utilized two RL algorithms: expert iteration and proximal policy optimization (PPO). In expert iteration, models sample responses to prompts, and the best performing ones are used for further training. PPO involves policy gradient methods to adjust the policy based on reward signals.
- Significant implementation details include:
    - **HHH Expert Iteration:** Combines reinforcement of specification gaming with supervision from a preference model that rewards helpful, honest, and harmless (HHH) behavior.
    - **Exploit-Only Expert Iteration:** Focuses solely on successful specification gaming samples without HHH oversight, to highlight the impact of excluding such supervision.
    - **Hidden Chain of Thought:** Augments models with reasoning capabilities that are hidden from the reward process, simulating future model capabilities.
    - **Preamble Instructions:** Models are prefixed with instructions about their training environment, reinforcing their understanding of being language model assistants trained via RL.
- Key findings from the experiments include:
    - Models trained on early stages of the curriculum generalize to more sophisticated forms of specification gaming in later stages.
    - A small but significant proportion of models trained on the full curriculum exhibit zero-shot generalization to reward tampering, including rewriting their reward functions and evading detection.
    - Retraining models not to game early environments reduces but does not eliminate reward tampering.
    - Adding HHH training does not prevent the generalization of specification gaming to reward tampering.
- The study demonstrates that large language models can generalize from simple specification gaming to complex reward tampering, suggesting that such behaviors may be nontrivial to remove and pose potential risks as models become more capable.
- [Blog](https://www.anthropic.com/research/reward-tampering); [Memo](https://cdn.sanity.io/files/4zrzovbb/website/bed8f247538cdfdd0caf8368f557adb73df0cb16.pdf)

### [Is DPO Superior to PPO for LLM Alignment? a Comprehensive Study](https://arxiv.org/abs/2404.10719)

- This paper by Xu et al. from Tsinghua University, OpenPsi Inc., and Shanghai Qi Zhi Institute investigates whether Direct Preference Optimization (DPO) is truly superior to Proximal Policy Optimization (PPO) for aligning large language models (LLMs) with human preferences. The study explores the theoretical and empirical properties of both methods and provides comprehensive benchmarks to evaluate their performance.
- The research begins by discussing the widespread use of RL from Human Feedback (RLHF) to align LLMs with human preferences. It highlights that existing RLHF methods can be categorized into reward-based and reward-free approaches. Reward-based methods, like those used in applications such as ChatGPT and Claude, involve learning a reward model and applying actor-critic algorithms such as PPO. Reward-free methods, such as DPO, optimize policies directly based on preference data without an explicit reward model.
- The paper delves into the theoretical limitations of DPO, demonstrating that it may find biased solutions that exploit out-of-distribution responses. The authors argue that this can lead to suboptimal performance, particularly in scenarios where there is a distribution shift between model outputs and the preference dataset. Empirical studies support this claim, showing that DPO’s performance degrades significantly under distribution shifts.
- Implementation details for PPO are extensively discussed, revealing critical factors for achieving optimal performance in RLHF settings. Key techniques identified include advantage normalization, large batch size, and exponential moving average updates for the reference model. These enhancements are shown to significantly improve PPO’s performance across various tasks, including dialogue generation and code generation.
- The study presents a series of experiments benchmarking DPO and PPO across multiple RLHF testbeds, such as the SafeRLHF dataset, HH-RLHF dataset, APPS, and CodeContest datasets. Results indicate that PPO consistently outperforms DPO in all cases, achieving state-of-the-art results in challenging code competition tasks. Specifically, on the CodeContest dataset, a PPO model with 34 billion parameters surpasses the previous state-of-the-art AlphaCode-41B, demonstrating a notable improvement in performance.
- Key experimental findings include:
    1. **Theoretical Analysis**: Demonstrates that DPO can produce biased policies due to out-of-distribution exploitation, while PPO’s regularization via KL divergence helps mitigate this issue.
    2. **Synthetic Scenario Validation**: Illustrates DPO’s susceptibility to generating biased distributions favoring unseen responses, while PPO maintains more stable performance.
    3. **Real Preference Datasets**: Shows that DPO’s performance can be improved by addressing distribution shifts through additional supervised fine-tuning (SFT) and iterative training, though PPO still outperforms DPO significantly.
    4. **Ablation Studies for PPO**: Highlights the importance of advantage normalization, large batch sizes, and exponential moving average updates in enhancing PPO’s RLHF performance.
- The authors conclude that while DPO offers a simpler training procedure, its performance is hindered by sensitivity to distribution shifts and out-of-distribution data. PPO, with proper tuning and implementation enhancements, demonstrates robust effectiveness and achieves superior results across diverse RLHF tasks.
- In summary, the comprehensive analysis and empirical evidence provided in this paper establish PPO as a more reliable and effective method for LLM alignment compared to DPO, particularly in scenarios requiring high-performance and robust alignment with human preferences.

### [Pairwise Proximal Policy Optimization: Harnessing Relative Feedback for LLM Alignment](https://arxiv.org/abs/2310.00212)

- This paper by Wu et al. from UC Berkeley proposes a novel RL framework, Pairwise Proximal Policy Optimization (P3O), designed to optimize large language models (LLMs) using comparative feedback rather than absolute rewards. Traditional approaches such as Proximal Policy Optimization (PPO) have limitations when dealing with reward functions derived from comparative losses like the Bradley-Terry loss. These limitations include the necessity for reward normalization and token-wise updates, which introduce complexity and potential instability.
- The proposed P3O algorithm operates on trajectory-wise policy gradient updates, simplifying the optimization process by directly utilizing comparative rewards. This approach is invariant to equivalent reward functions, addressing the instability issues present in PPO. The paper presents a comprehensive theoretical foundation, establishing that P3O avoids the complications of value function approximation and Generalized Advantage Estimation (GAE), which are essential in PPO.
- The implementation of P3O involves the following key steps:
    1. **Initialization**: Policy parameters are initialized.
    2. **Data Collection**: Pairwise trajectories are collected by running the policy on a batch of prompts, generating two responses per prompt.
    3. **Reward Calculation**: Trajectory-wise rewards are computed, incorporating both the preference-based reward and the KL-divergence penalty from the supervised fine-tuning (SFT) model.
    4. **Gradient Estimation**: The policy gradient is estimated using the relative differences in rewards between the paired responses, adjusted by importance sampling to account for the policy change.
    5. **Policy Update**: Gradient updates are applied to the policy parameters, following either separate or joint clipping strategies to maintain stability.
- The figure below from the paper illustrates the prevalent method for fine-tuning LMs using RL, which relies on Absolute Feedback. In this paradigm, algorithms like PPO has to learn a VV function, which capture not only the valuable relative preference information, but also less part, which is the scale of the reward for a given prompt. Contrastingly, the figure on the right presents paradigm for optimizing reward model trained via comparative loss, e.g., Bradley-Terry Loss (Bradley & Terry, 1952). P3O generates a pair of responses per prompt, leveraging only the Relative Feedback - derived from the difference in reward - for policy gradient updates. This method obviates the need for additional VV function approximations and intricate components like GAE.

![](https://aman.ai/images/papers/P3O.jpg)

- Empirical evaluations are conducted on summarization and question-answering tasks using datasets like TL;DR and Anthropic’s Helpful and Harmless (HH). The results demonstrate that P3O achieves a superior trade-off between reward and KL-divergence compared to PPO and other baseline methods. Specifically, P3O shows improved alignment with human preferences, as evidenced by higher rewards and better performance in head-to-head comparisons evaluated by GPT-4.
- The experiments reveal that P3O not only achieves higher reward scores but also maintains better KL control, making it a robust alternative for fine-tuning LLMs with relative feedback. The study underscores the potential of P3O in simplifying the RL fine-tuning process while enhancing model alignment with human values. Future work aims to explore the impacts of reward over-optimization and extend the policy gradient framework to accommodate multiple ranked responses.

### [BPO: Supercharging Online Preference Learning by Adhering to the Proximity of Behavior LLM](https://arxiv.org/abs/2406.12168)

- This paper by Xu et al. from UCSB and CMU presents Behavior Preference Optimization (BPO), a novel approach to enhancing online preference learning for large language models (LLMs) by maintaining proximity to the behavior LLM that collects training samples. The key motivation is to address the limitations of traditional Direct Alignment from Preferences (DAP) methods, which do not fully exploit the potential of online training data.
- The authors propose a new online DAP algorithm, emphasizing the construction of a trust region around the behavior LLM (πβπβ) rather than a fixed reference model (\pi_{\ref}\pi_{\ref}). This approach ensures that the learning LLM (πθπθ) remains aligned with the behavior model, thereby stabilizing the training process and improving performance.
    
- **Implementation Details:**
    1. **Algorithm Overview**:
        - The BPO algorithm dynamically updates πβπβ with πθπθ every K steps, where K is the annotation interval calculated as T/F (total training steps divided by the preference annotation frequency).
        - The training loss LBPOLBPO is computed by constraining the KL divergence between πθπθ and πβπβ, thus constructing a trust region around the behavior LLM.
    2. **Ensemble of LoRA Weights**:
        - To mitigate training instability, the authors optimize an ensemble of Low-Rank Adaptation (LoRA) weights and merge them during inference without additional overhead. This ensemble approach stabilizes the training process.
    3. **Experimental Setup**:
        - The experiments were conducted on three datasets: Reddit TL;DR, Anthropic Helpfulness, and Harmlessness, using a preference simulator for annotation.
        - BPO was integrated with various DAP methods, including DPO, IPO, and SLiC, and compared against their online and offline counterparts.
- The figure below from the paper illustrates an overview of the training pipeline of our BPO. Our training loss LBPO is calculated by constraining the KL divergence between πθπθ and the behavior LLM πβπβ. Every KK steps, they update πβπβ with πθπθ and use it to collect new samples for annotations.

![](https://aman.ai/images/papers/BPO.jpg)

- **Experimental Details:**
    - **Preference Annotation Frequency**:
        - Different annotation frequencies were tested, demonstrating that even a small increase in frequency (F = 2) significantly improves performance over offline DPO, achieving notable gains in win rates against reference texts.
    - **Ablation Study**:
        - The authors performed an ablation study to verify that the performance improvement stems from the better trust region constructed around πβπβ, not just the higher quality of πβπβ compared to \pi_{\ref}\pi_{\ref}.
    - **Stabilization Techniques**:
        - The use of an ensemble of LoRA weights proved effective in stabilizing training, as single LoRA weight optimization led to rapid deterioration of performance.
- **Results:**
    - BPO significantly outperformed both its on-policy and offline DAP counterparts across all tasks, particularly on TL;DR, Helpfulness, and Harmlessness, demonstrating its strong generalizability.
    - The dynamic trust region around the behavior LLM ensured better alignment and stability during training, leading to higher win rates and more consistent performance improvements.
- The proposed BPO method offers a substantial advancement in online preference learning for LLMs, balancing performance and computational efficiency, and demonstrating remarkable applicability to various DAP methods and annotation frequencies.

### [SimPO: Simple Preference Optimization with a Reference-Free Reward](https://arxiv.org/abs/2405.14734)

- This paper by Meng et al. from Danqi Chen’s lab at Princeton proposes SimPO, a novel offline preference optimization algorithm that simplifies and improves upon Direct Preference Optimization (DPO). Unlike DPO, which requires a reference model and can be computationally intensive, SimPO introduces a reference-free reward that aligns more closely with the model generation process.
- SimPO uses the average log probability of a sequence as the implicit reward, which better aligns with model generation metrics and removes the need for a reference model. This reward formulation enhances computational efficiency and memory usage. Additionally, SimPO incorporates a target reward margin into the Bradley-Terry objective to create a larger separation between winning and losing responses, further optimizing performance.
- The authors conducted extensive evaluations using various state-of-the-art models, including base and instruction-tuned models like Mistral and Llama3. They tested SimPO on benchmarks such as AlpacaEval 2, MT-Bench, and Arena-Hard, demonstrating significant performance improvements over DPO. Specifically, SimPO outperformed DPO by up to 6.4 points on AlpacaEval 2 and 7.5 points on Arena-Hard, with minimal increase in response length, indicating efficiency in length exploitation.
- The figure below from the paper illustrates that SimPO and DPO mainly differ in their reward formulation, as indicated in the shaded box.

![](https://aman.ai/images/papers/SimPO.jpg)

- **Implementation Details:**
    1. **Reward Formulation**:
        - SimPO calculates the reward as the average log probability of all tokens in a response using the policy model, normalized by the response length. This formulation eliminates the reference model, making SimPO more efficient.
        - The reward equation is: rSimPO(x,y)=β‖y‖logπθ(y∣x)=β‖y‖∑‖y‖i=1logπθ(yi∣x,y<i)rSimPO(x,y)=β‖y‖log⁡πθ(y∣x)=β‖y‖∑i=1‖y‖log⁡πθ(yi∣x,y<i), where ββ controls reward scaling.
    2. **Target Reward Margin**:
        - A margin γγ is introduced to the Bradley-Terry model to ensure a minimum reward difference between winning and losing responses.
        - The modified objective is: LSimPO(πθ)=−E(x,yw,yl)∼D[logσ(β‖yw‖logπθ(yw∣x)−β‖yl‖logπθ(yl∣x)−γ)]LSimPO(πθ)=−E(x,yw,yl)∼D[log⁡σ(β‖yw‖log⁡πθ(yw∣x)−β‖yl‖log⁡πθ(yl∣x)−γ)].
    3. **Training Setups**:
        - **Base Setup**: Models were trained on the UltraChat-200k dataset to create a supervised fine-tuned (SFT) model, followed by preference optimization using the UltraFeedback dataset.
        - **Instruct Setup**: Off-the-shelf instruction-tuned models were used, regenerating chosen and rejected response pairs to mitigate distribution shifts.
    4. **Evaluation**:
        - SimPO was evaluated on AlpacaEval 2, Arena-Hard, and MT-Bench benchmarks. Performance was measured in terms of length-controlled win rate and raw win rate.
        - SimPO achieved notable results, such as a 44.7% length-controlled win rate on AlpacaEval 2 and a 33.8% win rate on Arena-Hard, making it the strongest 8B open-source model.
    5. **Hyperparameters**:
        - Optimal performance was achieved with ββ set between 2.0 and 2.5, and γγ between 0.5 and 1.5.
- SimPO demonstrates a significant advancement in preference optimization, simplifying the process while improving computational efficiency and performance on multiple benchmarks. The removal of the reference model and the alignment of the reward function with generation metrics are key innovations that contribute to its success.
- [Code](https://github.com/princeton-nlp/SimPO)

### [Discovering Preference Optimization Algorithms with and for Large Language Models](https://arxiv.org/abs/2406.08414)

- This paper by Chris Lu et al. from Sakana AI, University of Cambridge, and FLAIR, presents a novel approach to offline preference optimization for Large Language Models (LLMs) by leveraging LLM-driven objective discovery. Traditional preference optimization relies on manually-crafted convex loss functions, but this approach is limited by human creativity. The authors propose an iterative method that prompts an LLM to discover new preference optimization loss functions automatically, leading to the development of state-of-the-art algorithms without human intervention.
- The core contribution of this paper is the introduction of the Discovered Preference Optimization (DiscoPOP) algorithm, which adaptively combines logistic and exponential losses. This process is facilitated through an LLM-driven pipeline that iteratively proposes and evaluates new loss functions based on their performance on downstream tasks.
- **Implementation Details:**
    1. **Initial Context Construction:** The system prompt initializes the LLM with several established objective functions in code and their performance metrics.
    2. **LLM Querying and Output Validation:** The LLM is queried to propose new objective functions, which are parsed, validated through unit tests, and evaluated.
    3. **Performance Evaluation:** The proposed objective functions are evaluated based on their ability to optimize a model on predefined downstream tasks, with the performance metric feeding back into the LLM.
    4. **Iterative Refinement:** The LLM iteratively refines its proposals, synthesizing new candidate loss functions that blend successful aspects of previous formulations.
- **Discovery Process:**
    - The LLM generates PyTorch-based candidate objective functions, taking log probabilities of preferred and rejected completions as inputs.
    - Valid candidates are used to fine-tune an LLM, evaluated using performance metrics such as MT-Bench scores.
    - The performance data is fed back into the LLM, which iteratively refines its generation strategy based on this feedback.
- The figure below from the paper illustrates: (Left) Conceptual illustration of LLM-driven discovery of objective functions. We prompt an LLM to output new code-level implementations of offline preference optimization losses 𝔼(yw,yl,x)∼[f(βρ)]E(yw,yl,x)∼D[f(βρ)] as a function of the policy (πθ)(πθ) and reference model’s (πref )(πref ) likelihoods of the chosen (yw)(yw) and rejected (yl)(yl) completions. Afterward, they run an inner loop training procedure and evaluate the resulting model on MT-Bench. The corresponding performance is fed back to the language model, and they query it for the next candidate. (Right) Performance of discovered objective functions on Alpaca Eval.

![](https://aman.ai/images/papers/DiscoPOP.jpg)

- **Results:**
    - The DiscoPOP algorithm, a dynamically weighted sum of logistic and exponential losses, emerged as a top performer. It was evaluated on multi-turn dialogue tasks (MT-Bench), single-turn dialogue tasks (Alpaca Eval 2.0), summarization tasks (TL;DR), and positive sentiment generation tasks (IMDb).
    - DiscoPOP showed significant improvement in win rates against GPT-4 and performed competitively on various held-out tasks, demonstrating robustness and adaptability across different preference optimization challenges.
- **Technical Details:**
    - The DiscoPOP loss function is non-convex, incorporating a temperature parameter to balance between logistic and exponential terms based on the log-ratio difference (ρρ). This dynamic weighting allows the function to handle both large and small differences effectively, contributing to its superior performance.
- **Significance:**
    - This LLM-driven discovery approach eliminates the constraints of human creativity in designing loss functions, automating the generation of high-performing preference optimization algorithms.
    - The iterative refinement process ensures continuous improvement and adaptability, leading to state-of-the-art performance in preference alignment tasks.
- This work opens new avenues for automated discovery and optimization in machine learning, showcasing the potential of leveraging LLMs to enhance and innovate traditional methodologies in a scalable and efficient manner. The proposed DiscoPOP algorithm represents a significant advancement in offline preference optimization, offering a robust and flexible solution for aligning LLM outputs with human preferences.
- [Code](https://github.com/luchris429/DiscoPOP)

### [Anchored Preference Optimization and Contrastive Revisions: Addressing Underspecification in Alignment](https://arxiv.org/abs/2408.06266v5)

- This paper by D’Oosterlinck et al. from Ghent University, Stanford University, and Contextual AI introduces methods to improve alignment in LLMs by addressing two core issues: the suboptimal contrastive nature of preference data and the limitations of alignment objectives. The authors propose Contrastive Learning from AI Revisions (CLAIR) and Anchored Preference Optimization (APO) to enhance the clarity of preference signals and the stability of alignment training.
- CLAIR creates minimally contrasting preference pairs by revising lower-quality outputs generated by the target model. Instead of using a judge to pick between outputs, CLAIR employs a reviser (a stronger model such as GPT4-turbo) to minimally improve the weaker output, ensuring that the contrast between outputs is clear and targeted. This leads to more precise preference data compared to conventional methods where preference pairs might vary due to uncontrolled differences. Empirical results show that CLAIR generates the best contrastive data, as measured by token-level Jaccard similarity and character-level Levenshtein edit distance, outperforming on-policy and off-policy judge datasets.
- The figure below from the paper illustrates that alignment is underspecified with regard to preferences and training objective. A: Preference pairs can vary along irrelevant aspects, Contrastive Learning from AI Revisions (CLAIR) creates a targeted preference signal instead. B: The quality of the model can impact alignment training, Anchored Preference Optimization (APO) explicitly accounts for this.

![](https://aman.ai/images/papers/APO.jpg)

- The figure below from the paper illustrates an answer produced by Llama-3-8B-Instruct for a prompt, and corresponding GPT4-turbo revision of this answer. The differences between answer and revision are highlighted. The revision generally follows the same outline as the answer but improves it where possible. For example, the revision correctly alters the count of Parisian restaurants from 2 to 3 in the second line of the answer.

![](https://aman.ai/images/papers/APO2.jpg)

- APO is a family of contrastive alignment objectives that explicitly consider the relationship between the model and the preference data. The authors propose two key variants: APO-zero and APO-down. APO-zero is used when winning outputs are better than the model’s outputs, ensuring that the likelihood of winning outputs increases and that of losing outputs decreases. APO-down is preferred when the model is already superior to the winning outputs, decreasing the likelihood of both but decreasing the likelihood of the losing output more sharply. APO provides more fine-grained control compared to widely used objectives such as Direct Preference Optimization (DPO), avoiding scenarios where increasing the likelihood of a winning output can degrade model performance.
- The authors conducted experiments aligning Llama-3-8B-Instruct on 32K CLAIR-generated preference pairs and comparable datasets using several alignment objectives. The results demonstrated that CLAIR, combined with APO, led to a significant improvement in performance, closing the gap between Llama-3-8B-Instruct and GPT4-turbo by 45% on the MixEval-Hard benchmark. The best model improved by 7.65% over the base Llama-3-8B-Instruct, primarily driven by the improved contrastiveness of CLAIR-generated data and the tailored dynamics of APO. In comparison, other alignment objectives like DPO and KTO did not perform as well, with DPO showing a tendency to degrade the model due to its ambiguous handling of winning and losing likelihoods.
- CLAIR and APO offer a more stable and controllable approach to alignment by improving the precision of preference signals and ensuring that training dynamics are better suited to the model and data relationship. The experiments also underscore the importance of controlling contrastiveness in preference datasets and adapting the alignment objective to the specific needs of the model.
- The paper concludes with discussions on how these methods compare to other alignment efforts like RL from AI Feedback (RLAIF) and Direct Preference Optimization (DPO), highlighting how CLAIR and APO address the challenges of underspecification in alignment.

### [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300)

- This paper by Shao et al. from DeepSeek-AI, Tsinghua University, and Peking University, introduces the DeepSeekMath 7B model, a state-of-the-art domain-specific language model optimized for mathematical reasoning, achieving results comparable to GPT-4 and Gemini-Ultra on mathematical benchmarks. Below is a detailed summary:
- DeepSeekMath 7B showcases the effectiveness of domain-specific pre-training and innovative RL techniques for advancing mathematical reasoning in open-source language models. Its contributions in data curation, RL algorithms, and multilingual capability serve as a foundation for future research in this domain.
    
- **Core Contributions**:
    
    1. **Domain-Specific Training:**
        - DeepSeekMath 7B is pre-trained using 120B tokens sourced from a newly developed DeepSeekMath Corpus, extracted and refined from Common Crawl data. The corpus is seven times larger than Minerva’s and nine times the size of OpenWebMath.
        - Pre-training incorporates natural language, code, and math-specific data for comprehensive reasoning capabilities.
    2. **Key Model Innovations:**
        - **Group Relative Policy Optimization (GRPO):** A novel RL (RL) technique designed to optimize the model’s reasoning while reducing memory consumption by bypassing the need for a critic model in RL frameworks like PPO.
        - Instruction tuning with Chain-of-Thought (CoT), Program-of-Thought (PoT), and tool-integrated reasoning datasets to enhance mathematical understanding.
- **Model Development and Implementation**:
    
    1. **Pre-training Pipeline:**
        - Base model: DeepSeek-Coder-Base-v1.5 7B, extended with 500B tokens. The corpus composition includes:
            - 56% from the DeepSeekMath Corpus.
            - 20% GitHub code.
            - 10% arXiv papers.
            - 10% natural language data from Common Crawl.
    2. **Data Selection and Processing:**
        
        - The DeepSeekMath Corpus was curated using an iterative pipeline involving fastText-based classification to filter high-quality mathematical content. The dataset was decontaminated to exclude overlap with evaluation benchmarks like GSM8K and MATH.
        - The plot below from the paper illustrates an iterative pipeline that collects mathematical web pages from Common Crawl.
        
        ![](https://aman.ai/images/papers/DeepSeekMath-1.jpg)
        
    3. **Mathematical Instruction Tuning:**
        - Fine-tuning on 776K examples (English and Chinese datasets), leveraging CoT, PoT, and Python-based reasoning for diverse mathematical fields such as algebra, calculus, and geometry.
    4. **RL with GRPO:**
        
        - GRPO uses group scores as baselines, simplifying reward estimation and computational complexity.
        - The plot below from the paper illustrates PPO and the proposed GRPO. GRPO foregoes the value model, instead estimating the baseline from group scores, significantly reducing training resources.
        
        ![](https://aman.ai/images/papers/DeepSeekMath-2.jpg)
        
        - RL training focused on GSM8K and MATH benchmarks with chain-of-thought prompts, achieving a 6-9% improvement over instruction-tuned models.
- **Key Results**:
    
    1. **Mathematical Reasoning:**
        - Achieved 51.7% accuracy on the MATH benchmark, surpassing all open-source models up to 70B size and approaching GPT-4 levels.
        - Demonstrated superior results across English and Chinese benchmarks like GSM8K (88.2%) and CMATH (88.8%).
    2. **Tool-Aided Problem Solving:**
        - Using Python for problem-solving, DeepSeekMath 7B outperformed the prior state-of-the-art Llemma 34B on benchmarks like GSM8K+Python and MATH+Python.
    3. **General Capabilities:**
        - Improvements in general reasoning and understanding benchmarks like MMLU (54.9%) and BBH (59.5%), as well as coding tasks like HumanEval and MBPP.
- **Observations and Insights**:
    
    1. **Code Training Benefits:**
        - Pre-training with code improves mathematical reasoning, both with and without tool use.
        - Mixed code and math training synergize mathematical problem-solving and coding performance.
    2. **ArXiv Data Limitations:**
        - Training on arXiv papers alone did not significantly enhance reasoning, suggesting potential issues with the data’s format or relevance.
    3. **RL Efficiency:**
        - GRPO efficiently improves instruction-tuned models with fewer computational resources compared to PPO, setting a new benchmark in LLM RL techniques.

## Further Reading

### HuggingFace’s Alignment Handbook

- [The Alignment Handbook](https://github.com/huggingface/alignment-handbook) contains robust recipes to align language models with human and AI preferences. It also contains code to train your very own Zephyr models:
    - Full fine-tuning with Microsoft’s DeepSpeed ZeRO-3 on A100s
    - LoRA or QLoRA fine-tuning on consumer GPUs

![](https://aman.ai/primers/ai/assets/LLM/Alignment.jpg)

- Dataset from HuggingFace called [No Robots](https://huggingface.co/datasets/HuggingFaceH4/no_robots) of 10k instructions and demonstrations to train instruct models. This is based on the SFT dataset from OpenAI’s InstructGPT paper. 100% organic and written entirely by skilled human annotators.

### Empirical Evaluation: DPO vs. IPO vs. KTO

- [Preference Tuning LLMs with Direct Preference Optimization Methods](https://huggingface.co/blog/pref-tuning) by Hugging Face summarizes their extensive evaluation of three state of the art alignment algorithms. DPO vs IPO vs KTO.
- The results demonstrate a complex interaction between key hyper-parameters, models, and datasets. As a quick overview:
    - DPO: Casts the RLHF objective via a loss based on a prompt and its positive and negative completions
    - IPO: Has an identity function rather than DPO’s sigmoid that can potentially cause overfitting
    - KTO: Rather than paired (+ve, -ve) pairs, takes unpaired good and bad (binary preference; thumbs-up and thumbs-down) data
- The team conducted experiments on two models possessing 7B parameters each; namely, Zephyr-7b-beta-sft and OpenHermes-7B. Subsequently, preference fine-tuning was applied utilizing two widely recognized preference datasets: Ultrafeedback and Intel’s Orca DPO pairs. It is pertinent to note that all the associated code is accessible as open-source at [The Alignment Handbook](https://github.com/huggingface/alignment-handbook).
- This investigation aims to discern the influence of the beta parameter on model performance. To this end, the MT Bench, a multi-turn benchmark employing GPT-4 to assess model efficacy across eight distinct categories, was utilized. Despite its limitations, MT Bench serves as a viable instrument for evaluating the capabilities of conversational large language models (LLMs).
- In the case of the Zephyr model, it was determined that optimal performance was attained at the minimal beta value of 0.01. This finding was consistent across all three algorithms evaluated, suggesting that a more detailed examination within the beta range of 0.0 to 0.2 could yield valuable insights for the research community.
- Regarding the OpenHermes model, although the relative performance of each algorithm remained consistent - with the ranking being DPO > KTO > IPO - the optimal beta value exhibited significant variation among the algorithms. Specifically, the most favorable beta values for DPO, KTO, and IPO were identified as 0.6, 0.3, and 0.01, respectively.

## FAQs

### In RLHF, What are the Memory Requirements of the Reward and Critic Model Compared to the Policy/reference Model?

- In RLHF, you typically have the following models:
    
    - **Policy model (also called the actor)**
    - **Reference model (frozen copy of the initial policy)**
    - **Reward model (trained from human feedback)**
    - **Critic model (value function)**
- Here’s how their memory requirements generally compare:
    
    - **Policy vs Reference model**:
        - These are usually the same architecture (e.g., a decoder-only transformer like GPT), so they have roughly equal memory requirements.
        - The reference model is frozen, but still loaded into memory for reward computation (KL divergence term), so it uses as much memory as the policy model.
        - Combined, they double the memory footprint compared to using just one model.
    - **Reward model**:
        - Often has the same architecture as the policy/reference model (e.g., same transformer backbone) but with a small head on top to produce scalar reward values.
        - If it shares weights with the policy/reference model (e.g., using LoRA or other weight-sharing schemes), it can be lighter, but in many setups it’s a full separate copy.
        - Memory requirement: roughly equal to the policy/reference model, possibly slightly less if stripped down or quantized.
    - **Critic model**:
        - In transformer-based PPO, the critic is often implemented as a separate head on the policy model or as a duplicate model with a value head.
        - If separate, it often has the same architecture as the policy but only outputs a scalar value per token.
        - Memory requirement: similar to the policy model, unless heavily optimized (e.g., sharing layers or being much smaller).
- **Summary of memory requirements (relative to one transformer model):**
    
    - Policy: 1x
    - Reference: 1x
    - Reward: ~1x
    - Critic: ~1x
- **Total:** ~4x the memory of a single model, unless model sharing, quantization, or other tricks are used.
    

### Why is the PPO/GRPO Objective Called a Clipped “surrogate” Objective?

- The PPO (and its variants such as GRPO) objective is called a surrogate objective because it doesn’t directly optimize the true reinforcement learning objective — the expected rewards over time — but instead optimizes a proxy that is easier and safer to compute. Specifics below:
    - **True RL Objective is Unstable or Intractable:**
        - The actual objective in RL is to maximize expected reward over trajectories, which involves high variance and instability during training, especially for large models like LLMs. It often requires estimating complex quantities like the value function accurately over time, which is difficult in practice.
    - **Surrogate Objectives Improve Stability:**
        - Surrogate objectives simplify this by using:
            - Advantage estimates to approximate how much better a new action is compared to the old one.
            - Importance sampling ratios (like πθπoldπθπold) to correct for the shift in policy.
            - Clipping (in PPO and GRPO) to avoid overly large policy updates that might destabilize training.
    - **Practical Optimization Benefits:**
        - By approximating the true objective, surrogate objectives allow for stable and efficient policy updates, which are essential in fine-tuning large models via reinforcement learning.
- In summary, it’s called a surrogate because it’s a well-designed stand-in for the true goal of maximizing reward, tailored to be safer and more effective for gradient-based optimization.

### Is the Importance Sampling Ratio Also Called the Policy or Likelihood Ratio?

- Yes, the importance sampling ratio is often referred to as the policy ratio or the likelihood ratio, especially in the context of reinforcement learning algorithms like PPO and GRPO.
- Here’s what these terms mean in this context:
    - **Importance Sampling Ratio:**
        - This is the ratio:
            
            πθ(a∣s)πold(a∣s)πθ(a∣s)πold(a∣s)
            
            - where πθπθ is the current (new) policy and πoldπold is the old (behavior) policy.
        - It tells us how much more or less likely the new policy is to take action aa in state ss compared to the old one.
            
    - **Policy Ratio:**
        - This is a shorthand name for the same quantity. It reflects the relative likelihood of an action under the current policy versus the old one — hence, “policy ratio.”
    - **Likelihood Ratio:**
        - Also the same quantity, but phrased from a statistical perspective. It compares the likelihoods assigned by two probability distributions (policies) to the same data (action).
- So, in PPO or GRPO:
    
    - You’ll often see this ratio appear as something like:
    
    rt(θ)=πθ(ot∣q,o<t)πold(ot∣q,o<t)rt(θ)=πθ(ot∣q,o<t)πold(ot∣q,o<t)
    
    - And it’s used to weight the advantage, or to apply clipping for stability.
- All three names refer to the same thing — they just come from different angles (importance sampling theory, policy learning, or statistics).

### Does REINFORCE and TRPO in Policy Optimization Also Use a Surrogate Loss?

- REINFORCE uses a basic form of surrogate loss based on the log-likelihood and returns.
- TRPO uses a more principled surrogate loss that incorporates importance sampling and constraints to ensure safe policy updates.
- Specifics below:
    - **REINFORCE:**
        - REINFORCE is based on the likelihood ratio trick (also called the policy gradient theorem).
        - The loss function used in REINFORCE is:
            
            L(θ)=𝔼[logπθ(a|s)⋅R]L(θ)=E[log⁡πθ(a|s)⋅R]
            
            - where RR is the return from a trajectory.
        - This is essentially a surrogate for maximizing the expected return, but it’s a very direct one: it’s derived directly from the gradient of the expected return.
        - It doesn’t include constraints or trust region concerns — so while it’s a kind of surrogate loss, it’s very raw and unstable due to high variance.
    - **TRPO (Trust Region Policy Optimization):**
        - TRPO introduces a more sophisticated surrogate objective: Lθ=𝔼[πθ(a|s)πold(a|s)⋅Â (s,a)]Lθ=E[πθ(a|s)πold(a|s)⋅A^(s,a)]
            - subject to a constraint on the KL divergence: 𝔼[DKL(πold(⋅|s)‖πθ(⋅|s))]≤δE[DKL(πold(⋅|s)‖πθ(⋅|s))]≤δ
        - |   |   |   |
            |---|---|---|
            |The expression $$\frac{\pi_\theta(a|s)}{\pi_{\text{old}}(a|s)} \cdot \hat{A}(s, a)$$ is the **surrogate loss** TRPO tries to optimize.|
            
        - This surrogate is designed to estimate the improvement in policy performance, assuming the new policy doesn’t deviate too much from the old one (hence the trust region).
        - The KL constraint ensures stable updates and limits how much the new policy can differ from the old one, helping avoid destructive updates.

## References

- [RL from Human Feedback: From Zero to chatGPT](https://www.youtube.com/watch?v=2MBJOuVq380)
- [Illustrating RLHF](https://huggingface.co/blog/rlhf)
- [Sebastian Raschka’s LinkedIn post](https://www.linkedin.com/in/sebastianraschka?miniProfileUrn=urn%3Ali%3Afs_miniProfile%3AACoAAAxqHaUBEa6zzXN--gv-wd8ih0vevPvr9eU&lipi=urn%3Ali%3Apage%3Ad_flagship3_search_srp_all%3Bic8rQnV%2BTHqwI0K2TXBzzg%3D%3D)
- [Off The Grid’s post on HHH](https://offthegridxp.substack.com/p/what-is-constitutional-ai-harmlessness)
- [Anthropic’s tweet thread](https://twitter.com/AnthropicAI/status/1603791161419698181)
- [Building block’s medium post on CAI](https://medium.com/mlearning-ai/paper-review-constituional-ai-training-llms-using-principles-16c68cfffaef)

-  [](https://github.com/amanchadha)|  [](https://citations.amanchadha.com/)|  [](https://twitter.com/i_amanchadha)|  [](mailto:hi@aman.ai)| 

[www.amanchadha.com](https://www.amanchadha.com/)