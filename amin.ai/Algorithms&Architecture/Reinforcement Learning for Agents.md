
  
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

---

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
- Some practical analyses (e.g., [Is DPO Superior to PPO for LLM Alignment?](https://arxiv.org/abs/2404.10719)) compare PPO vs DPO in alignment tasks.
    

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
- By selecting algorithms appropriate for the interaction type (single turn vs tool vs multi-turn), one can tailor the training for efficiency, stability, and scalability.

----

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

- Reward modeling lies at the heart of RL systems for language, web, and computer-use agents. In traditional RL, the reward function is hand-crafted to quantify success—for example, the score in a game or the distance to a goal. In contrast, modern LLM-based agents operate in open-ended environments where the notion of “correctness” or “helpfulness” is inherently subjective and context-depe
ximate human judgment. Instead of manually defining numerical rewards, the system learns a function rϕ(x,y) that predicts the quality of an agent’s output y for a given input x. These RMs are usually fine-tuned on preference datasets where human annotators rank outputs from best to worst.
    
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



---------


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
        
- These approaches are especially effective in _multi-turn environments_ scenarios where the state space expands combinatorially with each decision.
    

### Planning and Value Composition Across Multiple Environments

- The integration of search-based reasoning with learned RL policies allows agents to _compose behaviors across environment types_. For instance:
    
    - In **single-turn environments**, search helps refine output reasoning (e.g., multi-step chain-of-thought validation).
    - In **tool-use environments**, it aids in selecting optimal tool invocation sequences.
    - In **multi-turn environments**, it supports long-horizon planning and dynamic replanning when goals change.
- The combined expected return from multi-environment value composition can be expressed as:
    
    Jglobal=∑e∈Eωe𝔼πe[∑tγtr(e)t]
    
    - where E denotes environment types (SingleTurn, Tool, MultiTurn) and ωe are task-specific weights.
- This hierarchical structure aligns exploration depth with task complexity, improving sample efficiency and stability.
    

### Summary and Outlook

- Search-based RL represents a crucial step in bridging **symbolic planning** and **neural policy learning** for complex, real-world agents.
    
    - **Monte Carlo Tree Search (MCTS)** provides structured exploration with statistical guarantees.
    - **Neural-guided search** integrates learned policy and value priors for scalability.
    - **Process-wise rewards** smooth sparse reward landscapes, enabling deeper credit assignment.
    - **Hybrid search–RL systems** enable online adaptation and continual learning.
- As web and computer-use agents evolve, search-based strategies are increasingly viewed not as add-ons but as _core cognitive modules_, empowering agents to deliberate, simulate, and refine decisions—much like human reasoning.

----

