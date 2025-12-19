

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

- Classify correct vs incorrect tool-call formats.

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
    
- Weights w are tuned to balance shaping vs final correctness.
    

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
    - Allowing the policy to differentiate between “bad idea but learning” vs “excellent behavior.”
- This encourages exploration in the factored action space and prevents PPO/GRPO from collapsing into trivial policies.
    

##### Reward Table: Positive and Negative Rewards by Category

- Below is a consolidated table representing **typical** asymmetric reward magnitudes for each component. These values are illustrative and are often tuned per domain.

###### Reward Values for “When / Which / How” and Outcome-Level Components

|**Reward Component**|**Description**|**Positive Reward Range**|**Negative Reward Range**|
|---|---|---|---|
|**When** (call decision)|Correctly calling a tool when needed|+0.5 to +1.5|−0.2 (tool required but not called)|
||Correctly not calling a tool|+0.3 to +1|−0.2 (tool called when unnecessary)|
|**Which** (tool selection)|Selecting correct tool|+0.5 to +2.0|−0.3 to −0.7 (wrong tool)|
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
- If rewards were symmetric (e.g., +10 vs −10), then most exploratory episodes would produce extreme negative advantages, instantly pushing the model toward refusing all tool calls. Asymmetry prevents this collapse.
    

##### Takeaways

- Asymmetric rewards are essential for training LLM tool-calling policies because they:
    
    - Preserve exploration.
    - Deliver stable gradients for PPO/GRPO.
    - Avoid trivial degenerate strategies.
    - Properly balance shaping rewards (for “when / which / how”) with outcome-level rewards.
    - Distinguish partial correctness from catastrophic failure.
    - Encourage correct final answers without over-penalizing small mistakes.
- The reward table and examples above provide a practical blueprint for implementing and tuning asymmetric rewards in your own RL tool-calling system.
    

### RL Optimization Pipeline: Shared Flow + PPO Vs GRPO

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
    
    1. **When** → recognizing tool necessity vs non-necessity
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
    - Win-rate vs evaluator models (LLM-as-a-Judge)
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
    - Pass rate vs LLM-judge (DeepSeek-V3, GPT-4, etc.)
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

--------

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