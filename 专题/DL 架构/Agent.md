[Distilled AI](https://aman.ai/primers/ai/)[Back to aman.ai](https://aman.ai/)

# Primers • Agents

- [Overview](https://aman.ai/primers/ai/agents/#overview)
- [The “Agentic AI Moment”](https://aman.ai/primers/ai/agents/#the-agentic-ai-moment)
- [The Agentic Workflow](https://aman.ai/primers/ai/agents/#the-agentic-workflow)
- [Workflows vs. Agents](https://aman.ai/primers/ai/agents/#workflows-vs-agents)
- [The Agent Framework](https://aman.ai/primers/ai/agents/#the-agent-framework)
    - [Agent Core (LLM)](https://aman.ai/primers/ai/agents/#agent-core-llm)
    - [Memory Modules](https://aman.ai/primers/ai/agents/#memory-modules)
    - [Tools](https://aman.ai/primers/ai/agents/#tools)
    - [Planning Module](https://aman.ai/primers/ai/agents/#planning-module)
- [Agentic Design Patterns](https://aman.ai/primers/ai/agents/#agentic-design-patterns)
    - [Reflection](https://aman.ai/primers/ai/agents/#reflection)
        - [Overview](https://aman.ai/primers/ai/agents/#overview-1)
        - [Reflection Workflow: Step-by-Step Process](https://aman.ai/primers/ai/agents/#reflection-workflow-step-by-step-process)
            - [Initial Output Generation](https://aman.ai/primers/ai/agents/#initial-output-generation)
            - [Self-Evaluation and Constructive Feedback](https://aman.ai/primers/ai/agents/#self-evaluation-and-constructive-feedback)
            - [Revision Based on Feedback](https://aman.ai/primers/ai/agents/#revision-based-on-feedback)
        - [Beyond Self-Reflection: Integrating Additional Tools](https://aman.ai/primers/ai/agents/#beyond-self-reflection-integrating-additional-tools)
        - [Multi-Agent Framework for Enhanced Reflection](https://aman.ai/primers/ai/agents/#multi-agent-framework-for-enhanced-reflection)
            - [Example of Reflection: Multi-Agent Framework for Iterative Code Improvement](https://aman.ai/primers/ai/agents/#example-of-reflection-multi-agent-framework-for-iterative-code-improvement)
    - [Function/Tool/API Calling](https://aman.ai/primers/ai/agents/#functiontoolapi-calling)
        - [Tool Calling Examples: Web Search and Code Execution](https://aman.ai/primers/ai/agents/#tool-calling-examples-web-search-and-code-execution)
        - [Function Calling Datasets](https://aman.ai/primers/ai/agents/#function-calling-datasets)
            - [Hermes Function-Calling V1](https://aman.ai/primers/ai/agents/#hermes-function-calling-v1)
            - [Glaive Function Calling V2](https://aman.ai/primers/ai/agents/#glaive-function-calling-v2)
            - [Salesforce’s Xlam-function-calling-60k](https://aman.ai/primers/ai/agents/#salesforces-xlam-function-calling-60k)
                - [JSON Data Format for Query and Answers](https://aman.ai/primers/ai/agents/#json-data-format-for-query-and-answers)
                - [Example](https://aman.ai/primers/ai/agents/#example)
            - [Synth-APIGen-v0.1](https://aman.ai/primers/ai/agents/#synth-apigen-v01)
                - [Example](https://aman.ai/primers/ai/agents/#example-1)
        - [Evaluation](https://aman.ai/primers/ai/agents/#evaluation)
            - [Berkeley Function-Calling Leaderboard](https://aman.ai/primers/ai/agents/#berkeley-function-calling-leaderboard)
            - [Python Evaluation](https://aman.ai/primers/ai/agents/#python-evaluation)
            - [Non-Python Evaluation](https://aman.ai/primers/ai/agents/#non-python-evaluation)
            - [Evaluation Methods](https://aman.ai/primers/ai/agents/#evaluation-methods)
            - [Inference Example Output with XML and JSON](https://aman.ai/primers/ai/agents/#inference-example-output-with-xml-and-json)
            - [JSON/Structured Outputs](https://aman.ai/primers/ai/agents/#jsonstructured-outputs)
        - [Gorilla OpenFunctions-v2 LLM](https://aman.ai/primers/ai/agents/#gorilla-openfunctions-v2-llm)
            - [Example](https://aman.ai/primers/ai/agents/#example-2)
        - [Best Practices, Guidelines, and Limitations](https://aman.ai/primers/ai/agents/#best-practices-guidelines-and-limitations)
            - [Choosing the Right Model for Tool Use](https://aman.ai/primers/ai/agents/#choosing-the-right-model-for-tool-use)
            - [Tool Usage Limits](https://aman.ai/primers/ai/agents/#tool-usage-limits)
            - [Sequential vs. Parallel Tool Execution](https://aman.ai/primers/ai/agents/#sequential-vs-parallel-tool-execution)
            - [Error Handling and Retries](https://aman.ai/primers/ai/agents/#error-handling-and-retries)
        - [Designing Effective Tool Interfaces](https://aman.ai/primers/ai/agents/#designing-effective-tool-interfaces)
            - [Choosing the Right Tool Format](https://aman.ai/primers/ai/agents/#choosing-the-right-tool-format)
            - [Best Practices for Tool Design](https://aman.ai/primers/ai/agents/#best-practices-for-tool-design)
            - [Improving Tool Usability](https://aman.ai/primers/ai/agents/#improving-tool-usability)
    - [Planning](https://aman.ai/primers/ai/agents/#planning)
        - [Overview](https://aman.ai/primers/ai/agents/#overview-2)
        - [Example](https://aman.ai/primers/ai/agents/#example-3)
        - [Planning vs. Deterministic Approaches](https://aman.ai/primers/ai/agents/#planning-vs-deterministic-approaches)
    - [Multi-agent Collaboration](https://aman.ai/primers/ai/agents/#multi-agent-collaboration)
        - [Background](https://aman.ai/primers/ai/agents/#background)
        - [Motivation](https://aman.ai/primers/ai/agents/#motivation)
    - [Implementation](https://aman.ai/primers/ai/agents/#implementation)
- [Agentic Workflow Patterns](https://aman.ai/primers/ai/agents/#agentic-workflow-patterns)
    - [Prompt Chaining](https://aman.ai/primers/ai/agents/#prompt-chaining)
    - [Routing](https://aman.ai/primers/ai/agents/#routing)
        - [Parallelization](https://aman.ai/primers/ai/agents/#parallelization)
        - [Orchestrator-Workers](https://aman.ai/primers/ai/agents/#orchestrator-workers)
        - [Evaluator-Optimizer](https://aman.ai/primers/ai/agents/#evaluator-optimizer)
- [Single-Agent Systems vs. Multi-Agent Systems](https://aman.ai/primers/ai/agents/#single-agent-systems-vs-multi-agent-systems)
    - [Preservation of Context](https://aman.ai/primers/ai/agents/#preservation-of-context)
    - [Simplicity and Maintainability](https://aman.ai/primers/ai/agents/#simplicity-and-maintainability)
    - [Flexibility in Problem Solving](https://aman.ai/primers/ai/agents/#flexibility-in-problem-solving)
    - [Feasibility with Modern Tools](https://aman.ai/primers/ai/agents/#feasibility-with-modern-tools)
- [Model Context Protocol (MCP)](https://aman.ai/primers/ai/agents/#model-context-protocol-mcp)
    - [Overview](https://aman.ai/primers/ai/agents/#overview-3)
    - [Why MCP?](https://aman.ai/primers/ai/agents/#why-mcp)
    - [General Architecture](https://aman.ai/primers/ai/agents/#general-architecture)
    - [How MCP Works](https://aman.ai/primers/ai/agents/#how-mcp-works)
    - [MCP vs. API](https://aman.ai/primers/ai/agents/#mcp-vs-api)
        - [Comparative Analysis](https://aman.ai/primers/ai/agents/#comparative-analysis)
        - [When to Use MCP vs. Traditional APIs](https://aman.ai/primers/ai/agents/#when-to-use-mcp-vs-traditional-apis)
    - [Security, Updates, and Authentication](https://aman.ai/primers/ai/agents/#security-updates-and-authentication)
    - [Getting Started with MCP: High-Level Steps](https://aman.ai/primers/ai/agents/#getting-started-with-mcp-high-level-steps)
    - [Use-Cases of MCP in Real-World Development Scenarios](https://aman.ai/primers/ai/agents/#use-cases-of-mcp-in-real-world-development-scenarios)
        - [Automating Feature Development from Ticket to Implementation](https://aman.ai/primers/ai/agents/#automating-feature-development-from-ticket-to-implementation)
        - [Intelligent Ticket and Task Management](https://aman.ai/primers/ai/agents/#intelligent-ticket-and-task-management)
        - [Automated Communication with Relevant Stakeholders](https://aman.ai/primers/ai/agents/#automated-communication-with-relevant-stakeholders)
        - [Smart Staging, Commit Messages, and PR Creation](https://aman.ai/primers/ai/agents/#smart-staging-commit-messages-and-pr-creation)
        - [Automated Debugging and Console Log Access](https://aman.ai/primers/ai/agents/#automated-debugging-and-console-log-access)
        - [Integration with Personal Task Management Tools](https://aman.ai/primers/ai/agents/#integration-with-personal-task-management-tools)
        - [Automated Project Announcements](https://aman.ai/primers/ai/agents/#automated-project-announcements)
    - [MCP Servers List](https://aman.ai/primers/ai/agents/#mcp-servers-list)
        - [Awesome MCP Servers](https://aman.ai/primers/ai/agents/#awesome-mcp-servers)
        - [Model Context Protocol Servers](https://aman.ai/primers/ai/agents/#model-context-protocol-servers)
        - [Composio MCP Servers](https://aman.ai/primers/ai/agents/#composio-mcp-servers)
        - [Smithery](https://aman.ai/primers/ai/agents/#smithery)
- [Agent2Agent (A2A) Protocol](https://aman.ai/primers/ai/agents/#agent2agent-a2a-protocol)
    - [A2A Design Principles](https://aman.ai/primers/ai/agents/#a2a-design-principles)
    - [How A2A Works](https://aman.ai/primers/ai/agents/#how-a2a-works)
    - [Core Protocol Mechanics](https://aman.ai/primers/ai/agents/#core-protocol-mechanics)
        - [Capability Discovery](https://aman.ai/primers/ai/agents/#capability-discovery)
        - [Task Management](https://aman.ai/primers/ai/agents/#task-management)
        - [Collaboration and Messaging](https://aman.ai/primers/ai/agents/#collaboration-and-messaging)
        - [User Experience (UX) Negotiation](https://aman.ai/primers/ai/agents/#user-experience-ux-negotiation)
        - [Content Routing and Privacy](https://aman.ai/primers/ai/agents/#content-routing-and-privacy)
    - [Real-World Scenario: Candidate Sourcing](https://aman.ai/primers/ai/agents/#real-world-scenario-candidate-sourcing)
    - [Implementation Architecture](https://aman.ai/primers/ai/agents/#implementation-architecture)
    - [Integration and Future Roadmap](https://aman.ai/primers/ai/agents/#integration-and-future-roadmap)
- [Agentic Retrieval-Augmented Generation (RAG)](https://aman.ai/primers/ai/agents/#agentic-retrieval-augmented-generation-rag)
    - [How Agentic RAG Works](https://aman.ai/primers/ai/agents/#how-agentic-rag-works)
    - [Agentic Decision-Making in Retrieval](https://aman.ai/primers/ai/agents/#agentic-decision-making-in-retrieval)
    - [Agentic RAG Architectures: Single-Agent vs. Multi-Agent Systems](https://aman.ai/primers/ai/agents/#agentic-rag-architectures-single-agent-vs-multi-agent-systems)
        - [Single-Agent RAG (Router)](https://aman.ai/primers/ai/agents/#single-agent-rag-router)
        - [Multi-Agent RAG Systems](https://aman.ai/primers/ai/agents/#multi-agent-rag-systems)
    - [Beyond Retrieval: Expanding Agentic RAG’s Capabilities](https://aman.ai/primers/ai/agents/#beyond-retrieval-expanding-agentic-rags-capabilities)
    - [Agentic RAG vs. Vanilla RAG: Key Differences](https://aman.ai/primers/ai/agents/#agentic-rag-vs-vanilla-rag-key-differences)
    - [Implementing Agentic RAG: Key Approaches](https://aman.ai/primers/ai/agents/#implementing-agentic-rag-key-approaches)
        - [Language Models with Function Calling](https://aman.ai/primers/ai/agents/#language-models-with-function-calling)
        - [Agent Frameworks](https://aman.ai/primers/ai/agents/#agent-frameworks)
    - [Enterprise-driven Adoption](https://aman.ai/primers/ai/agents/#enterprise-driven-adoption)
    - [Benefits](https://aman.ai/primers/ai/agents/#benefits)
    - [Limitations](https://aman.ai/primers/ai/agents/#limitations)
    - [Code](https://aman.ai/primers/ai/agents/#code)
        - [Implementing Agentic RAG with Function Calling](https://aman.ai/primers/ai/agents/#implementing-agentic-rag-with-function-calling)
            - [Define the Function for Retrieval](https://aman.ai/primers/ai/agents/#define-the-function-for-retrieval)
            - [Define the Tools Schema](https://aman.ai/primers/ai/agents/#define-the-tools-schema)
            - [Setting up the Interaction Loop](https://aman.ai/primers/ai/agents/#setting-up-the-interaction-loop)
            - [Executing the Agentic RAG Query](https://aman.ai/primers/ai/agents/#executing-the-agentic-rag-query)
        - [Implementing Agentic RAG with Agent Frameworks](https://aman.ai/primers/ai/agents/#implementing-agentic-rag-with-agent-frameworks)
            - [Step 1: Define Agents and Tools](https://aman.ai/primers/ai/agents/#step-1-define-agents-and-tools)
            - [Step 2: Configure Agent Routing](https://aman.ai/primers/ai/agents/#step-2-configure-agent-routing)
            - [Step 3: Chain Agents for Multi-Agent RAG](https://aman.ai/primers/ai/agents/#step-3-chain-agents-for-multi-agent-rag)
            - [Running the Multi-Agent Query](https://aman.ai/primers/ai/agents/#running-the-multi-agent-query)
    - [Disadvantages of Agentic RAG](https://aman.ai/primers/ai/agents/#disadvantages-of-agentic-rag)
    - [Summary](https://aman.ai/primers/ai/agents/#summary)
- [Benchmarks](https://aman.ai/primers/ai/agents/#benchmarks)
- [Common Use-cases](https://aman.ai/primers/ai/agents/#common-use-cases)
- [Case Studies](https://aman.ai/primers/ai/agents/#case-studies)
    - [Customer Support](https://aman.ai/primers/ai/agents/#customer-support)
    - [Software Development](https://aman.ai/primers/ai/agents/#software-development)
        - [Devin](https://aman.ai/primers/ai/agents/#devin)
- [Frameworks/Libraries](https://aman.ai/primers/ai/agents/#frameworkslibraries)
    - [AutoGen Studio](https://aman.ai/primers/ai/agents/#autogen-studio)
    - [AutoGen](https://aman.ai/primers/ai/agents/#autogen)
    - [Swarm](https://aman.ai/primers/ai/agents/#swarm)
    - [CrewAI](https://aman.ai/primers/ai/agents/#crewai)
    - [Letta](https://aman.ai/primers/ai/agents/#letta)
    - [Llama Stack](https://aman.ai/primers/ai/agents/#llama-stack)
    - [AutoRAG](https://aman.ai/primers/ai/agents/#autorag)
    - [Beam](https://aman.ai/primers/ai/agents/#beam)
    - [AutoAgents](https://aman.ai/primers/ai/agents/#autoagents)
    - [Amazon Bedrock’s AI Agent Framework](https://aman.ai/primers/ai/agents/#amazon-bedrocks-ai-agent-framework)
    - [Rivet](https://aman.ai/primers/ai/agents/#rivet)
    - [Vellum](https://aman.ai/primers/ai/agents/#vellum)
    - [BabyAGI](https://aman.ai/primers/ai/agents/#babyagi)
    - [Smolagents](https://aman.ai/primers/ai/agents/#smolagents)
    - [Agent S2](https://aman.ai/primers/ai/agents/#agent-s2)
    - [Open Operator](https://aman.ai/primers/ai/agents/#open-operator)
- [Example Flow Chart for an LLM Agent: Handling a Customer Inquiry](https://aman.ai/primers/ai/agents/#example-flow-chart-for-an-llm-agent-handling-a-customer-inquiry)
- [Use Cases](https://aman.ai/primers/ai/agents/#use-cases)
    - [Data Agent for Data Analysis](https://aman.ai/primers/ai/agents/#data-agent-for-data-analysis)
        - [Tools Setup](https://aman.ai/primers/ai/agents/#tools-setup)
        - [Agent Core Logic](https://aman.ai/primers/ai/agents/#agent-core-logic)
    - [LLM-Powered API Agent for Task Execution](https://aman.ai/primers/ai/agents/#llm-powered-api-agent-for-task-execution)
        - [Python Code Example](https://aman.ai/primers/ai/agents/#python-code-example)
        - [Plan-and-Execute Approach](https://aman.ai/primers/ai/agents/#plan-and-execute-approach)
- [Build Your Own LLM Agent](https://aman.ai/primers/ai/agents/#build-your-own-llm-agent)
- [Related Papers](https://aman.ai/primers/ai/agents/#related-papers)
    - [Reflection](https://aman.ai/primers/ai/agents/#reflection-1)
        - [Self-Refine: Iterative Refinement with Self-Feedback](https://aman.ai/primers/ai/agents/#self-refine-iterative-refinement-with-self-feedback)
        - [Reflexion: Language Agents with Verbal Reinforcement Learning](https://aman.ai/primers/ai/agents/#reflexion-language-agents-with-verbal-reinforcement-learning)
        - [CRITIC: Large Language Models Can Self-Correct with Tool-Interactive Critiquing](https://aman.ai/primers/ai/agents/#critic-large-language-models-can-self-correct-with-tool-interactive-critiquing)
    - [Tool Calling](https://aman.ai/primers/ai/agents/#tool-calling)
        - [Gorilla: Large Language Model Connected with Massive APIs](https://aman.ai/primers/ai/agents/#gorilla-large-language-model-connected-with-massive-apis)
        - [MM-REACT: Prompting ChatGPT for Multimodal Reasoning and Action](https://aman.ai/primers/ai/agents/#mm-react-prompting-chatgpt-for-multimodal-reasoning-and-action)
        - [Efficient Tool Use with Chain-of-Abstraction Reasoning](https://aman.ai/primers/ai/agents/#efficient-tool-use-with-chain-of-abstraction-reasoning)
    - [Planning](https://aman.ai/primers/ai/agents/#planning-1)
        - [Chain of Thought Prompting Elicits Reasoning in Large Language Models](https://aman.ai/primers/ai/agents/#chain-of-thought-prompting-elicits-reasoning-in-large-language-models)
        - [HuggingGPT: Solving AI Tasks with ChatGPT and Its Friends in HuggingFace](https://aman.ai/primers/ai/agents/#hugginggpt-solving-ai-tasks-with-chatgpt-and-its-friends-in-huggingface)
        - [Understanding the Planning of LLM Agents: a Survey](https://aman.ai/primers/ai/agents/#understanding-the-planning-of-llm-agents-a-survey)
    - [Multi-Agent Collaboration](https://aman.ai/primers/ai/agents/#multi-agent-collaboration-1)
        - [ChatDev: Communicative Agents for Software Development](https://aman.ai/primers/ai/agents/#chatdev-communicative-agents-for-software-development)
        - [AutoGen: Enabling Next-Gen LLM Applications Via Multi-Agent Conversation](https://aman.ai/primers/ai/agents/#autogen-enabling-next-gen-llm-applications-via-multi-agent-conversation)
        - [APIGen: Automated Pipeline for Generating Verifiable and Diverse Function-Calling Datasets](https://aman.ai/primers/ai/agents/#apigen-automated-pipeline-for-generating-verifiable-and-diverse-function-calling-datasets)
        - [AutoAgents: a Framework for Automatic Agent Generation](https://aman.ai/primers/ai/agents/#autoagents-a-framework-for-automatic-agent-generation)
        - [MetaGPT: Meta Programming for Multi-Agent Collaborative Framework](https://aman.ai/primers/ai/agents/#metagpt-meta-programming-for-multi-agent-collaborative-framework)
        - [Chameleon: Plug-and-Play Compositional Reasoning with Large Language Models](https://aman.ai/primers/ai/agents/#chameleon-plug-and-play-compositional-reasoning-with-large-language-models)
    - [OctoTools: an Agentic Framework with Extensible Tools for Complex Reasoning](https://aman.ai/primers/ai/agents/#octotools-an-agentic-framework-with-extensible-tools-for-complex-reasoning)
- [Further Reading](https://aman.ai/primers/ai/agents/#further-reading)
- [References](https://aman.ai/primers/ai/agents/#references)
- [Citation](https://aman.ai/primers/ai/agents/#citation)

## Overview

- AI agents are autonomous systems that combine the decision-making and action-oriented capabilities of autonomous frameworks with the natural language processing and comprehension strengths of large language models (LLMs). The LLM serves as the “brain” within an agent, interpreting language, generating responses, and planning tasks, while the agent framework enables the execution of these tasks within a defined environment. Together, they allow agents to engage in goal-oriented workflows, where LLMs contribute strategic insights, problem-solving, and adaptability to achieve outcomes with minimal human intervention.
- AI agents leverage LLMs as central reasoning engines, which facilitate real-time decision-making, task prioritization, and dynamic adaptation. In practice, an AI agent operates through a cycle: the LLM analyzes incoming information, formulates an actionable plan, and collaborates with a series of modular systems—such as APIs, web tools, or embedded sensors—to execute specific steps. Throughout this process, the LLM can maintain context and iterate, adjusting actions based on feedback from the agent’s environment or outcomes from previous steps. This integrated system enables AI agents to tackle complex, multi-phase tasks with increasing sophistication, driving innovation across sectors like finance, software engineering, and scientific discovery.

## The “Agentic AI Moment”

- Many people experienced a pivotal “AI moment” with the release of ChatGPT—a time when the system’s capabilities exceeded their expectations. This phenomenon, often called the “ChatGPT moment,” encapsulates interactions where the AI’s performance went beyond anticipated limits, demonstrating remarkable competence, creativity, or problem-solving ability.
- Analogous to the “ChatGPT moment,” agents have had an “Agentic AI moment”—an instance where an AI system exhibits unexpected autonomy and resourcefulness. One notable example ([source](https://www.deeplearning.ai/the-batch/how-agents-can-improve-llm-performance/)) involves an AI agent developed for online research. During a live demonstration, the agent encountered a rate-limiting error while accessing its primary web search tool. Instead of failing, the agent seamlessly adapted by switching to a secondary tool—a Wikipedia search feature—to complete the task effectively. This unplanned pivot showcased the agent’s ability to adjust independently to unforeseen circumstances, a hallmark of agentic planning and adaptive problem-solving that highlights the emerging potential of AI agents in complex, real-world applications.

## The Agentic Workflow

- Most current applications of large language models (LLMs) operate in zero-shot mode, where the model generates responses token by token without revisiting or refining its initial output. This approach is akin to asking someone to write an essay in one continuous attempt without making any corrections. While LLMs perform remarkably well under these constraints, an agentic, iterative workflow often leads to higher-quality results.
- A typical agentic workflow involves an LLM following a structured, multi-step process. Depending on the task, some or all of the following steps may be included:
    
    - **Planning** an outline for the task
    - **Assessing** whether additional research or web searches are needed
    - **Drafting** an initial response
    - **Reviewing** and identifying weak or irrelevant sections
    - **Revising** based on detected areas for improvement
- This structured, human-like refinement process enables AI agents to produce more robust and nuanced outputs compared to a single-pass approach.

## Workflows vs. Agents

- The term “agent” can have multiple interpretations. Some define agents as fully autonomous systems capable of operating independently over extended periods, using various tools to complete complex tasks. Others see them as prescriptive systems that follow predefined workflows.
- Per [Anthropic](https://www.anthropic.com/engineering/building-effective-agents), categorize both under agentic systems but distinguish between two key architectures:
    
    - **Workflows**: Systems where LLMs and tools are orchestrated through predefined code paths to complete tasks.
    - **Agents**: Systems where LLMs dynamically direct their own processes, deciding how to use tools and manage tasks autonomously.
- By leveraging these agentic approaches, AI systems can move beyond single-shot responses, refining their outputs iteratively and intelligently.

## The Agent Framework

- The Agent Framework provides a structured and modular design for organizing the core components of an AI agent. This setup allows for effective, adaptive interactions by combining critical components, each with defined roles that contribute to seamless task performance.

![](https://aman.ai/primers/ai/assets/agents/coreAI.jpg)

- The image above [(source)](https://developer.nvidia.com/blog/introduction-to-llm-agents/) illustrates the architecture of a typical end-to-end agent pipeline. Below, we explore each component in detail to understand the technical workings of an AI agent.

### Agent Core (LLM)

- At the heart of the agent, the **Agent Core** functions as the primary decision-making engine, where OpenAI’s **GPT-4** is employed to handle high-level reasoning and dynamic task management. This component includes:
    
    - **Decision-Making Engine**: Analyzes input data, memory, and goals to generate suitable responses.
    - **Goal Management System**: Continuously updates the agent’s objectives based on task progression.
    - **Integration Bus**: Manages the flow of information between memory, tools, and planning modules, ensuring cohesive data exchange.
- The Agent Core uses the LLM’s capabilities to complete tasks, generate new tasks as needed, and dynamically adjust priorities based on the evolving task context.
    

### Memory Modules

- Memory is a fundamental part of the framework, with a vector databases (such as Pinecone, Weaviate, Chroma, etc.) providing robust storage and retrieval mechanisms for task-related data. The memory modules enhance the agent’s context-awareness and task relevance through:
    
    - **Short-term Memory (STM)**: Manages temporary data for immediate task requirements, stored in volatile structures like stacks or queues to support quick access and frequent clearing.
    - **Long-term Memory (LTM)**: Uses vector databases for persistent storage of historical interactions, enabling the agent to reference past conversations or data over extended periods. Semantic similarity-based retrieval is employed to enhance relevance, factoring in recency and importance for efficient access.

### Tools

- Tools empower the agent with specialized capabilities to execute tasks precisely, often leveraging the **LangChain framework** for structured workflows. Tools include:
    
    - **Executable Workflows**: Defined within LangChain, providing structured, data-aware task handling.
    - **APIs**: Facilitate secure access to both internal and external data sources, enriching the agent’s functional range.
    - **Middleware**: Supports data exchange between the core and tools, handling formatting, error-checking, and ensuring security.
- LangChain’s integration enables the agent to dynamically interact with its environment, providing flexibility and adaptability across diverse tasks.
    

### Planning Module

- For complex problem-solving, the Planning Module enables structured approaches like task decomposition and reflection to guide the agent in optimizing solutions. The Task Management system within this module utilizes a deque data structure to autonomously generate, manage, and prioritize tasks. It adjusts priorities in real-time as tasks are completed and new tasks are generated, ensuring goal-aligned task progression.
- In summary, the LLM Agent Framework combines an LLM’s advanced language capabilities with a vector database’s efficient memory system and an agentic framework’s responsive tooling. These integrated components create a cohesive, powerful AI agent capable of adaptive, real-time decision-making and dynamic task execution across complex applications.

## Agentic Design Patterns

- Agentic design patterns empower AI models to transcend static interactions, enabling dynamic decision-making, self-assessment, and iterative improvement. These patterns establish structured workflows that allow AI to actively refine its outputs, incorporate new tools, and even collaborate with other AI agents to complete complex tasks. By leveraging agentic patterns, language models evolve from simple, one-step responders to adaptable, reliable, and contextually aware systems, enhancing their application across various domains.
- A well-defined categorization of agentic design patterns is crucial for developing robust and efficient AI agents. By organizing these patterns into a clear framework, developers and researchers can better understand how to structure AI workflows, optimize performance, and ensure that agents are equipped to handle complex, dynamic tasks.
- Below is a practical framework for classifying the most common agentic design patterns across various applications:
    
    - **Reflection**: The agent evaluates its work, identifying areas for improvement and refining its outputs based on this assessment. This process enables continuous improvement, ultimately leading to a more robust and accurate final output.
        
    - **Tool Use**: Agents are equipped with specific tools, such as web search or code execution capabilities, to gather necessary information, take actions, or process complex data in real time as part of their tasks.
        
    - **Planning**: The agent constructs and follows a comprehensive, step-by-step plan to achieve its objectives. This process may involve outlining, researching, drafting, and revising phases, as is often required in complex writing or coding tasks.
        
    - **Multi-agent Collaboration**: Multiple agents collaborate, each taking on distinct roles and contributing unique expertise to solve complex tasks by breaking them down into smaller, more manageable sub-tasks. This approach mirrors human teamwork, where roles like software engineer and QA specialist contribute to different aspects of a project.
        
- These agentic design patterns represent diverse methodologies through which AI agents can optimize task performance, refine outputs, and dynamically adapt workflows. For those exploring multi-agent systems, frameworks such as AutoGen, Crew AI, and LangGraph offer robust platforms for designing and deploying multi-agent solutions. Additionally, open-source projects such as [ChatDev](https://aman.ai/primers/ai/agents/#chatdev-communicative-agents-for-software-development) simulate a virtual software company operated by AI agents, provide developers with accessible tools to experiment with multi-agent systems.

### Reflection

#### Overview

- To boost the effectiveness of LLMs, a pivotal approach is the incorporation of a reflective mechanism within their workflows. Reflection is a method by which LLMs improve their output quality through self-evaluation and iterative refinement. By implementing this approach, an LLM can autonomously recognize gaps in its output, adjust based on feedback, and ultimately deliver responses that are more precise, efficient, and contextually aligned with user needs. This structured, iterative process transforms the typical query-response interaction into a dynamic cycle of continuous improvement.
    
- Reflection represents a relatively straightforward type of agentic workflow, yet it has proven to significantly enhance LLM output quality across diverse applications. By encouraging models to reflect on their performance, refine their responses, and utilize external tools for self-assessment, this design pattern enables models to deliver accurate, efficient, and contextually relevant results. This iterative process not only strengthens an LLM’s ability to produce high-quality outputs but also imbues it with a form of adaptability, allowing it to better meet complex, evolving requirements.
    
- The integration of Reflection into agentic workflows is transformative, rendering LLMs more adaptable, self-aware, and capable of handling complex tasks autonomously. As a foundational design pattern, Reflection holds substantial promise for enhancing the efficacy and reliability of LLM-based applications. This approach highlights the growing capacity of these models to function as intelligent, self-improving agents, poised to meet the demands of increasingly sophisticated tasks with minimal human intervention.
    

#### Reflection Workflow: Step-by-Step Process

##### Initial Output Generation

- In a typical task, such as writing code, the LLM is first prompted to generate an initial response aimed at accomplishing a specific goal (e.g., completing “task X”). This response may serve as a draft that will later be subjected to further scrutiny.

##### Self-Evaluation and Constructive Feedback

- After producing an initial output, the LLM can be guided to assess its response. For instance, in the case of code generation, it may be prompted with:
    
    > “Here’s code intended for task X: [previously generated code].  
    > Check the code carefully for correctness, style, and efficiency, and provide constructive criticism for improvement.”
    
- This self-critique phase enables the LLM to recognize any flaws in its work. It can identify issues related to correctness, efficiency, and stylistic quality, thus facilitating the detection of areas needing refinement.
    

##### Revision Based on Feedback

- Once the LLM generates feedback on its own output, the agentic workflow proceeds by prompting the model to integrate this feedback into a revised response. In this stage, the context given to the model includes both the original output and the constructive criticism it produced. The LLM then generates a refined version that reflects the improvements suggested during self-reflection.
    
- This cycle of criticism and rewriting can be repeated multiple times, resulting in iterative enhancements that significantly elevate the quality of the final output.
    

#### Beyond Self-Reflection: Integrating Additional Tools

- Reflection can be further augmented by equipping the LLM with tools that enable it to evaluate its own output quantitatively. For instance:
    
    - **Code Evaluation:** The model can run its code through unit tests to verify accuracy, using test cases to ensure correct results.
    - **Text Validation:** The LLM can leverage internet searches or external databases to fact-check and verify textual content.
- When errors or inaccuracies are detected through these tools, the LLM can reflect on the discrepancies, producing additional feedback and proposing ways to improve the output. This tool-supported reflection enables the LLM to refine its responses even further, effectively combining self-critique with external validation.
    

#### Multi-Agent Framework for Enhanced Reflection

- To optimize the Reflection process, a multi-agent framework can be utilized. In this configuration, two distinct agents are employed:

1. **Output Generation Agent:** Primarily responsible for producing responses aimed at achieving the designated task effectively.
2. **Critique Agent:** Tasked with critically evaluating the output of the first agent, offering constructive feedback to enhance its quality.

- Through this dialogue between agents, the LLM achieves improved results, as the two agents collaboratively identify and rectify weaknesses in the output. This cooperative approach introduces a second level of reflection, allowing the LLM to gain insights that a single-agent setup might miss.

##### Example of Reflection: Multi-Agent Framework for Iterative Code Improvement

- In the context of Reflection, one effective implementation involves a multi-agent interaction where two agents—a Coder Agent and a Critic Agent—collaborate to refine code through iterative feedback and revisions, emphasizing the synergy between generation and critique.

1. **Initial Task and Code Generation**: The process begins with a prompt given to the Coder Agent, instructing it to “write code for {task}.” The Coder Agent generates an initial version of the code, labeled here as `do_task(x)`.
    
2. **Critique and Error Identification**: The Critic Agent then reviews the initial code. In this case, it identifies a specific issue, stating, “There’s a bug on line 5. Fix it by…” and offers a constructive suggestion for improvement. This feedback allows the Coder Agent to understand where the code falls short.
    
3. **Code Revision**: Based on the Critic Agent’s feedback, the Coder Agent revises its code, producing an updated version, `do_task_v2(x)`. This revised code aims to address the issues highlighted in the first critique.
    
4. **Further Testing and Feedback**: The Critic Agent assesses the new version, testing it further (such as through unit tests). Here, it notes that “It failed Unit Test 3” and advises a further change, indicating that additional refinements are necessary for accuracy.
    
5. **Final Iteration**: The Coder Agent, with this additional guidance, creates yet another iteration of the code—`do_task_v3(x)`. This repeated process of critique and revision continues until the code meets the desired standards for functionality and efficiency.
    

- This example highlights the iterative nature of Reflection within a multi-agent framework. By engaging a Coder Agent focused on output generation and a Critic Agent dedicated to providing structured feedback, the system harnesses a continuous improvement loop. This interaction enables large language models to autonomously detect errors, refine logic, and improve their responses.
    
- The multi-agent setup exemplifies how Reflection can be operationalized to produce high-quality, reliable results. This structured approach not only enhances the LLM’s output but also mirrors human collaborative workflows, where constructive feedback leads to better solutions through repeated refinement.
    

### Function/Tool/API Calling

- The advent of Tool Use in LLMs represents a pivotal design pattern in agentic AI workflows, enabling LLMs to perform a diverse range of tasks beyond text generation. Tool Use refers to the capability of an LLM to utilize specific functions—such as executing code, conducting web searches, or interacting with productivity tools—within its responses, effectively expanding its utility far beyond conventional, language-based outputs. This approach allows LLMs to tackle more complex queries and execute multifaceted tasks by selectively invoking various external tools. From answering specific questions to performing calculations, the use of function calls empowers LLMs to provide highly accurate and contextually informed responses.
- A foundational example of Tool Use is seen in scenarios where users request information not available in the model’s pre-existing training data. For instance, if a user asks, “What is the best coffee maker according to reviewers?”, a model equipped with Tool Use may initiate a web search, fetching up-to-date information by generating a command string such as `{tool: web-search, query: "coffee maker reviews"}`. Upon processing, the model retrieves relevant pages, synthesizes the data, and delivers an informed response. This dynamic response mechanism emerged from early realizations that traditional transformer-based language models, reliant solely on pre-trained knowledge, were inherently limited. By integrating a web search tool, developers enabled the model to access and incorporate fresh information into its output, a capability now widely adopted across various LLMs in consumer-facing applications.
- Moreover, Tool Use enables LLMs to handle calculations and other tasks requiring precision that text generation alone cannot achieve. For example, when a user asks, “If I invest $100 at compound 7% interest for 12 years, what do I have at the end?”, an LLM could respond by executing a Python command like `100 * (1+0.07)**12`. The LLM generates a string such as `{tool: python-interpreter, code: "100 * (1+0.07)**12"}`, and then the calculation tool processes this command to deliver an accurate answer. This illustrates how Tool Use facilitates complex mathematical reasoning within conversational AI systems.
- The scope of Tool Use, however, extends well beyond web searches or basic calculations. As the technology has evolved, developers have implemented a wide array of functions, enabling LLMs to interface with multiple external resources. These functions may include accessing specialized databases, interacting with productivity tools like email and calendar applications, generating or interpreting images, and engaging with multiple data sources such as Search (via Google/Bing Search APIs), Wikipedia, and academic repositories like arXiv.
- Systems now prompt LLMs with detailed descriptions of available functions, specifying their capabilities and parameters. With these cues, an LLM can autonomously select the appropriate function to fulfill the user’s request. In settings where hundreds of tools are accessible, developers often employ heuristics to streamline function selection, prioritizing the tools most relevant to the current context—a strategy analogous to the subset selection techniques used in retrieval-augmented generation (RAG) systems.
- The development of large multimodal models (LMMs) such as LLaVa, GPT-4V, and Gemini marked another milestone in Tool Use. Prior to these models, LLMs could not process or manipulate images directly, and any image-related tasks had to be offloaded to specific computer vision functions, such as object recognition or scene analysis. The introduction of GPT-4’s function-calling capabilities in 2023 further advanced Tool Use by establishing a more general-purpose function interface, laying the groundwork for a versatile, multimodal AI ecosystem where models seamlessly integrate text, image, and other data types. This new functionality has subsequently led to a proliferation of LLMs designed to exploit Tool Use, broadening the range of applications and enhancing overall adaptability.
- For instance, below is a prompt from [SmolLM2](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct) for function calling, where the model is prompted to choose relevant functions based on specific user inquiries is as follows. The prompt template guides SmolLM2 in structuring its function calls precisely and prompts it to assess the relevance and sufficiency of available parameters before executing a function.

![](https://aman.ai/images/copy.png)

`You are an expert in composing functions. You are given a question and a set of possible functions.  Based on the question, you will need to make one or more function/tool calls to achieve the purpose.  If none of the functions can be used, point it out and refuse to answer.  If the given question lacks the parameters required by the function, also point it out.  You have access to the following tools: <tools></tools>  The output MUST strictly adhere to the following format, and NO other text MUST be included. The example format is as follows. Please make sure the parameter type is correct. If no function call is needed, please make the tool calls an empty list '[]'. <tool_call>[ {"name": "func_name1", "arguments": {"argument1": "value1", "argument2": "value2"}}, (more tool calls as required) ]</tool_call>`

- The evolution of Tool Use and function-calling capabilities in LLMs demonstrates the significant strides taken toward realizing general-purpose, agentic AI workflows. By enabling LLMs to autonomously utilize specialized tools across various contexts, developers have transformed these models from static text generators into dynamic, multifunctional systems capable of addressing a vast array of user needs. As the field advances, we can expect further innovations that expand the breadth and depth of Tool Use, pushing the boundaries of what LLMs can achieve in an integrated, agentic environment.

#### Tool Calling Examples: Web Search and Code Execution

- LLMs can leverage tools such as web search to provide current product recommendations and code execution to handle complex calculations, showcasing their adeptness at choosing and using the right tools based on user input.
- The image below ([source](https://www.deeplearning.ai/the-batch/agentic-design-patterns-part-3-tool-use)) showcases practical examples of tool calling in LLMs, highlighting two specific tools: a web search tool and a code execution tool. In the left panel, an example from Bing Copilot illustrates how an LLM can utilize a web search tool. When a user asks, “What is the best coffee maker according to reviewers?”, the model initiates a web search to gather relevant information from current online reviews. This allows the LLM to provide an informed answer based on up-to-date data.
- The right panel demonstrates an example from ChatGPT using a code execution tool. When a user asks, “If I invest $100 at compound 7% interest for 12 years, what do I have at the end?”, the LLM responds by generating a Python command to calculate the compounded interest. The code snippet, `principal = 100; interest_rate = 0.07; years = 12; value = principal * (1 + interest_rate) ** years`, is executed, providing an accurate financial calculation rather than relying solely on text-based reasoning.
- These examples illustrate the model’s ability to identify and select the appropriate tool based on the user’s query, further demonstrating the flexibility and enhanced capabilities of Tool Use in agentic LLM workflows.

![](https://aman.ai/primers/ai/assets/agents/ToolCalling.jpg)

#### Function Calling Datasets

##### [Hermes Function-Calling V1](https://huggingface.co/datasets/NousResearch/hermes-function-calling-v1)

- The [Hermes Function-Calling V1](https://huggingface.co/datasets/NousResearch/hermes-function-calling-v1) dataset is designed for training language models to perform structured function calls and return structured outputs based on natural language instructions.
- It includes function-calling conversations, json-mode samples, agentic json-mode, and structured extraction examples, showcasing various scenarios where AI agents interpret queries and execute relevant function calls.
- The Hermes Function-Calling Standard enables language models to execute API calls based on user requests, improving AI’s practical utility by allowing direct API interactions.

##### Glaive Function Calling V2

- The [Glaive Function Calling (52K)](https://huggingface.co/datasets/glaiveai/glaive-function-calling) and [Glaive Function Calling v2 (113K)](https://huggingface.co/datasets/vietgpt/glaive-function-calling-v2?row=0) are datasets generated through [Glaive](https://glaive.ai/) for the task of function calling, in the following format:

![](https://aman.ai/images/copy.png)

`SYSTEM: You are an helpful assistant who has access to the following functions to help the user, you can use the functions if needed- {   JSON function definiton } USER: user message ASSISTANT: assistant message Function call invocations are formatted as- ASSISTANT: <functioncall> {json function call} Response to the function call is formatted as- FUNCTION RESPONSE: {json function response}`

- There are also samples which do not have any function invocations, multiple invocations and samples with no functions presented and invoked to keep the data balanced.

##### [Salesforce’s Xlam-function-calling-60k](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k?row=0)

- Salesforce’s [xlam-function-calling-60k](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k?row=0) contains 60,000 data samples collected by APIGen, an automated data generation pipeline designed to produce verifiable high-quality datasets for function-calling applications. Each data in our dataset is verified through three hierarchical stages: format checking, actual function executions, and semantic verification, ensuring its reliability and correctness.
- Also, [xLAM-1b-fc-r](https://huggingface.co/Salesforce/xLAM-1b-fc-r) and [xLAM-7b-fc-r](https://huggingface.co/Salesforce/xLAM-7b-fc-r).

###### JSON Data Format for Query and Answers

- This JSON data format is used to represent a query along with the available tools and the corresponding answers. Here’s a description of the JSON format which consists of the following key-value pairs: ```
- **query** (string): The query or problem statement.
    
- **tools** (array): An array of available tools that can be used to solve the query.
    
    - Each tool is represented as an object with the following properties:
        
        - **name** (string): The name of the tool.
            
        - **description** (string): A brief description of what the tool does.
            
        - **parameters** (object): An object representing the parameters required by the tool.
            
            - Each parameter is represented as a key-value pair, where the key is the parameter name and the value is an object with the following properties:
                
                - **type** (string): The data type of the parameter (e.g., “int”, “float”, “list”).
                    
                - **description** (string): A brief description of the parameter.
                    
                - **required** (boolean): Indicates whether the parameter is required or optional.
                    
- **answers** (array): An array of answers corresponding to the query.
    
    - Each answer is represented as an object with the following properties:
        
        - **name** (string): The name of the tool used to generate the answer.
            
        - **arguments** (object): An object representing the arguments passed to the tool to generate the answer.
            
            - Each argument is represented as a key-value pair, where the key is the parameter name and the value is the corresponding value. ```
- Note that they format the **query**, **tools**, and **answers** as a string, but you can easily recover each entry to the JSON object via `json.loads(...)`.

###### Example

- Here’s an example JSON data:

![](https://aman.ai/images/copy.png)

`{   "query": "Find the sum of all the multiples of 3 and 5 between 1 and 1000. Also find the product of the first five prime numbers.",   "tools": [     {       "name": "math_toolkit.sum_of_multiples",       "description": "Find the sum of all multiples of specified numbers within a specified range.",       "parameters": {         "lower_limit": {           "type": "int",           "description": "The start of the range (inclusive).",           "required": true         },         "upper_limit": {           "type": "int",           "description": "The end of the range (inclusive).",           "required": true         },         "multiples": {           "type": "list",           "description": "The numbers to find multiples of.",           "required": true         }       }     },     {       "name": "math_toolkit.product_of_primes",       "description": "Find the product of the first n prime numbers.",       "parameters": {         "count": {           "type": "int",           "description": "The number of prime numbers to multiply together.",           "required": true         }       }     }   ],   "answers": [     {       "name": "math_toolkit.sum_of_multiples",       "arguments": {         "lower_limit": 1,         "upper_limit": 1000,         "multiples": [3, 5]       }     },     {       "name": "math_toolkit.product_of_primes",       "arguments": {         "count": 5       }     }   ] }`

- In this example, the query asks to find the sum of multiples of 3 and 5 between 1 and 1000, and also find the product of the first five prime numbers. The available tools are `math_toolkit.sum_of_multiples` and `math_toolkit.product_of_primes`, along with their parameter descriptions. The answers array provides the specific tool and arguments used to generate each answer.

##### [Synth-APIGen-v0.1](https://huggingface.co/datasets/argilla/Synth-APIGen-v0.1)

- A dataset of 50k samples by [Argilla](https://huggingface.co/argilla).

###### Example

- This example demonstrates the use of the `complex_to_polar` function, which is designed to convert complex numbers into their polar coordinate representations. The input query requests conversions for two specific complex numbers, `3 + 4j` and `1 - 2j`, showcasing how the function can be called with different arguments to obtain their polar forms.

![](https://aman.ai/images/copy.png)

``{   "func_name": "complex_to_polar",   "func_desc": "Converts a complex number to its polar coordinate representation.",   "tools": "[{\"type\":\"function\",\"function\":{\"name\":\"complex_to_polar\",\"description\":\"Converts a complex number to its polar coordinate representation.\",\"parameters\":{\"type\":\"object\",\"properties\":{\"complex_number\":{\"type\":\"object\",\"description\":\"A complex number in the form of `real + imaginary * 1j`.\"}},\"required\":[\"complex_number\"]}}}]",   "query": "I'd like to convert the complex number 3 + 4j and 1 - 2j to polar coordinates.",   "answers": "[{\"name\": \"complex_to_polar\", \"arguments\": {\"complex_number\": \"3 + 4j\"}}, {\"name\": \"complex_to_polar\", \"arguments\": {\"complex_number\": \"1 - 2j\"}}]",   "model_name": "meta-llama/Meta-Llama-3.1-70B-Instruct",   "hash_id": "f873783c04bbddd9d79f47287fa3b6705b3eaea0e5bc126fba91366f7b8b07e9",     }   ],   "category": "E-commerce Platforms",   "subcategory": "Kayak",   "task": "Flight Search" }``

#### Evaluation

- To ensure that Tool Use capabilities meet the demands of diverse real-world scenarios, it is crucial to evaluate the function-calling performance of LLMs rigorously. This evaluation encompasses assessing model performance across both Python and non-Python programming environments, with a focus on how effectively the model can execute functions, select the appropriate tools, and discern when a function is necessary within a conversational context. An essential aspect of this evaluation is testing the model’s ability to invoke functions accurately based on user prompts and determine whether certain functions are applicable or needed.
- This structured evaluation methodology enables a holistic understanding of the model’s function-calling performance, combining both syntactic accuracy and real-world execution fidelity. By examining the model’s ability to navigate various programming contexts and detect relevance in function invocation, this approach underscores the practical reliability of LLMs in diverse applications.

##### [Berkeley Function-Calling Leaderboard](https://gorilla.cs.berkeley.edu/leaderboard.html)

- Berkeley Function-Calling Leaderboard (BFCL) assesses the function-calling capabilities of various LLMs. It consists of 2,000 question-function-answer pairs across multiple programming languages (Python, Java, JavaScript, REST API, SQL).
- The evaluation covers complex use cases, including simple, multiple, and parallel function calls, requiring the selection and simultaneous execution of functions.
- BFCL tests function relevance detection to see how models handle irrelevant functions, expecting them to return an error message.
- Both proprietary and open-source models perform similarly in simple scenarios, but GPT-series models excel in more complex function-calling tasks.
- The Gorilla OpenFunctions dataset has expanded from 100 to 2,000 data points, increasing diversity and complexity in evaluations. The dataset includes functions from varied fields such as Mathematics, Sports, Finance, and more, covering 40 sub-domains.
- Evaluations are divided into Python (simple, multiple, parallel, parallel multiple functions) and Non-Python (chatting capability, function relevance, REST API, SQL, Java, JavaScript) categories.
- Python evaluations cover scenarios from single function calls to complex parallel multiple function calls.
- Non-Python evaluations test models on general-purpose chat, relevance detection, and specific API and language scenarios.
- Function relevance detection is a key focus, evaluating whether models avoid using irrelevant functions and highlighting their potential for hallucination.
- REST API testing involves real-world GET requests with parameters in URLs and headers, assessing models’ ability to generate executable API calls.
- SQL evaluation includes basic SQL queries, while Java and JavaScript testing focus on language-specific function-calling abilities.
- BFCL uses AST evaluation to check syntax and structural accuracy, and executable evaluation to verify real-world function execution.
- AST evaluation ensures function matching, parameter consistency, and type/value accuracy.
- Executable function evaluation runs generated functions to verify response accuracy and consistency, particularly for REST APIs.
- The evaluation approach requires complete matching of model outputs to expected results; partial matches are considered failures.
- Ongoing development includes continuous updates and community feedback to refine evaluation methods, especially for SQL and chat capabilities.

[![](https://aman.ai/primers/ai/assets/LLM/BFCL.png)](https://gorilla.cs.berkeley.edu/leaderboard.html)

##### Python Evaluation

- Inspired by the [Berkeley Function-Calling Leaderboard](https://gorilla.cs.berkeley.edu/blogs/8_berkeley_function_calling_leaderboard.html) and [APIGen](https://arxiv.org/abs/2406.18518), the evaluation framework can be organized by function type (simple, multiple, parallel, parallel multiple) or evaluation method (AST or execution of APIs). This categorization helps in comparing model performances on standard function-calling scenarios and assessing their accuracy and efficiency. By structuring the evaluation in this way, it provides a comprehensive view of how well the model performs across different types of function calls and under varying conditions. More on the section on [Evaluation Methods](https://aman.ai/primers/ai/agents/#evaluation-methods).
- The Python evaluation categories, listed below, assess the model’s ability to handle single and multiple function calls, both sequentially and in parallel. These tests simulate realistic scenarios where the model must interpret user queries, select appropriate functions, and execute them accurately, mimicking real-world applications. By testing these different scenarios, the evaluation can highlight the model’s proficiency in using Python-based function calls under varying degrees of complexity and concurrency.

1. **Simple Function:** In this category, the evaluation involves a single, straightforward function call. The user provides a JSON function document, and the model is expected to invoke only one function call. This test examines the model’s ability to handle the most common and basic type of function call correctly.
    
2. **Parallel Function:** This evaluation scenario requires the model to make multiple function calls in parallel in response to a single user query. The model must identify how many function calls are necessary and initiate them simultaneously, regardless of the complexity or length of the user query.
    
3. **Multiple Function:** This category involves scenarios where the user input can be matched to one function call out of two to four available JSON function documentations. The model must accurately select the most appropriate function to call based on the given context.
    
4. **Parallel Multiple Function:** This is a complex evaluation combining both parallel and multiple function categories. The model is presented with multiple function documentations, and each relevant function may need to be invoked zero or more times in parallel.
    

- As mentioned earlier, each Python evaluation category includes both Abstract Syntax Tree (AST) and executable evaluations. A significant limitation of AST evaluation is the variety of methods available to construct function calls that achieve the same result, leading to challenges in consistency and accuracy. In these cases, executable evaluations provide a more reliable alternative by directly running the code to verify outcomes, allowing for precise and practical validation of functionality across different coding approaches.

##### Non-Python Evaluation

- The non-Python evaluation categories, listed below, test the model’s ability to handle diverse scenarios involving conversation, relevance detection, and the use of different programming languages and technologies. These evaluations provide insights into the model’s adaptability to various contexts beyond Python. By including these diverse categories, the evaluation aims to ensure that the model is versatile and capable of handling various use cases, making it applicable in a broad range of applications.

1. **Chatting Capability:** This category evaluates the model’s general conversational abilities without invoking functions. The goal is to see if the model can maintain coherent dialogue and recognize when function calls are unnecessary. This is distinct from function relevance detection, which involves determining the suitability of invoking any provided functions.
    
2. **Function Relevance Detection:** This tests whether the model can discern when none of the provided functions are relevant. The ideal outcome is that the model refrains from making any function calls, demonstrating an understanding of when it lacks the required function information or user instruction.
    
3. **REST API:** This evaluation focuses on the model’s ability to generate and execute realistic REST API calls using Python’s requests library. It tests the model’s understanding of GET requests, including path and query parameters, and its ability to generate calls that match real-world API documentation.
    
4. **SQL:** This category assesses the model’s ability to construct simple SQL queries using custom `sql.execute` functions. The evaluation is limited to basic SQL operations like SELECT, INSERT, UPDATE, DELETE, and CREATE, testing whether the model can generalize function-calling capabilities beyond Python.
    
5. **Java + JavaScript:** Despite the uniformity in function-calling formats across languages, this evaluation examines how well the model adapts to language-specific types and syntax, such as Java’s HashMap. It includes examples that test the model’s handling of Java and JavaScript, emphasizing the need for language-specific adaptations.
    

##### Evaluation Methods

- Two primary methods are used to evaluate model performance:
    
    1. **Abstract Syntax Tree (AST) Evaluation:** AST evaluation involves parsing the model-generated function calls to check their structure against expected outputs. It verifies the function name, parameter presence, and type correctness. AST evaluation is ideal for cases where execution isn’t feasible due to language constraints or when the result cannot be easily executed.
        
        - **Simple Function AST Evaluation**
            - The AST evaluation process focuses on comparing a single model output function against its function doc and possible answers. Here is a flow chart ([source](https://gorilla.cs.berkeley.edu/blogs/8_berkeley_function_calling_leaderboard.html)) that shows the step-by-step evaluation process.![](https://aman.ai/primers/ai/assets/agents/AST.jpg)
        - **Multiple/Parallel/Parallel-Multiple Functions AST Evaluation**
            - The multiple, parallel, or parallel-multiple function AST evaluation process extends the idea in the simple function evaluation to support multiple model outputs and possible answers.
                - The evaluation process first associates each possible answer with its function doc. Then it iterates over the model outputs and calls the simple function evaluation on each function (which takes in one model output, one possible answer, and one function doc).
                    - The order of model outputs relative to possible answers is not required. A model output can match with any possible answer.
            - The evaluation employs an all-or-nothing approach to evaluation. Failure to find a match across all model outputs for any given possible answer results in a failed evaluation.
    2. **Executable Function Evaluation:** This metric assesses the model by executing the function calls it generates and comparing the outputs against expected results. This evaluation is crucial for testing real-world applicability, focusing on whether the function calls run successfully, produce the correct types of responses, and maintain structural consistency in their outputs.
        
- The combination of AST and executable evaluations ensures a comprehensive assessment, providing insights into both the syntactic and functional correctness of the model’s output.
    

##### Inference Example Output with XML and JSON

- Typical function calling datasets uses a combination of both XML and JSON elements (cf. inference output sample below), as detailed below.
    
    1. **XML Structure**: Elements like `<|im_start|>`, `<tool_call>`, and `<tool_response>` resemble XML-like tags, which help demarcate different parts of the communication.
        
    2. **Dictionary/JSON Structure**: Within the `<tool_call>` and `<tool_response>` tags, the data for the function arguments and the stock fundamentals is formatted as Python-style dictionaries (or JSON-like key-value pairs), such as `{'symbol': 'TSLA'}` and `{"name": "get_stock_fundamentals", "content": {'symbol': 'TSLA', 'company_name': 'Tesla, Inc.' ...}}`.
        
- This combination provides an XML-like structure for message flow and JSON for data representation, allowing for structured, nested data representation and demarcation of sections.
    
- Here’s an example of the inference output from [Hermes Function-Calling V1](https://huggingface.co/datasets/NousResearch/hermes-function-calling-v1):
    

![](https://aman.ai/images/copy.png)

`<|im_start|>user Fetch the stock fundamentals data for Tesla (TSLA)<|im_end|> <|im_start|>assistant <tool_call> {'arguments': {'symbol': 'TSLA'}, 'name': 'get_stock_fundamentals'} </tool_call><|im_end|> <|im_start|>tool <tool_response> {"name": "get_stock_fundamentals", "content": {'symbol': 'TSLA', 'company_name': 'Tesla, Inc.', 'sector': 'Consumer Cyclical', 'industry': 'Auto Manufacturers', 'market_cap': 611384164352, 'pe_ratio': 49.604652, 'pb_ratio': 9.762013, 'dividend_yield': None, 'eps': 4.3, 'beta': 2.427, '52_week_high': 299.29, '52_week_low': 152.37}} </tool_response> <|im_end|>`

##### JSON/Structured Outputs

- Once a model is trained on a system prompt that asks for JSON-based structured outputs (below), the model should respond with only a JSON object response, based on the specific JSON schema provided.

![](https://aman.ai/images/copy.png)

`<|im_start|>system You are a helpful assistant that answers in JSON. Here's the JSON schema you must adhere to:\n<schema>\n{schema}\n</schema><|im_end|>`

- The schema can be made from a pydantic object using (e.g., a standalone script available is here from [Hermes Function-Calling V1](https://github.com/NousResearch/Hermes-Function-Calling/blob/main/jsonmode.py)).
    
- As an example from [Hermes Function-Calling V1](https://huggingface.co/datasets/NousResearch/hermes-function-calling-v1):
    

![](https://aman.ai/images/copy.png)

`{   "id": "753d8365-0e54-43b1-9514-3f9b819fd31c",   "conversations": [     {       "from": "system",       "value": "You are a function calling AI model. You are provided with function signatures within <tools> </tools> XML tags. You may call one or more functions to assist with the user query. Don't make assumptions about what values to plug into functions.\n<tools>\n[{'type': 'function', 'function': {'name': 'search_flights', 'description': 'Searches for flights based on departure and destination cities, dates, class, and other preferences.', 'parameters': {'type': 'object', 'properties': {'departure_city': {'type': 'string', 'description': 'The city from which the flight will depart.'}, 'destination_city': {'type': 'string', 'description': 'The destination city for the flight.'}, 'departure_date': {'type': 'string', 'description': 'The departure date for the flight.', 'format': 'date'}, 'return_date': {'type': 'string', 'description': 'The return date for the flight.', 'format': 'date'}, 'class': {'type': 'string', 'description': 'The class of the flight ticket.', 'enum': ['economy', 'business', 'first']}, 'flexible_cancellation': {'type': 'boolean', 'description': 'Indicates if the search should filter for flights with flexible cancellation policies.'}}, 'required': ['departure_city', 'destination_city', 'departure_date', 'return_date', 'class']}}}]\n</tools>\nFor each function call return a json object with function name and arguments within <tool_call> </tool_call> tags with the following schema:\n<tool_call>\n{'arguments': <args-dict>, 'name': <function-name>}\n</tool_call>\n"     },     {       "from": "human",       "value": "I'm planning a kayaking trip and looking to book flights from Los Angeles to Auckland. My departure is scheduled for July 10th, 2023, and I intend to return on July 24th, 2023. I would prefer to travel in economy class and would also like the option to have flexible cancellation policies for the tickets due to the uncertain nature of outdoor activities. Could you please search for flights that meet these criteria and provide me with the available options?"     },     {       "from": "gpt",       "value": "<tool_call>\n{'arguments': {'departure_city': 'Los Angeles', 'destination_city': 'Auckland', 'departure_date': '2023-07-10', 'return_date': '2023-07-24', 'class': 'economy', 'flexible_cancellation': True}, 'name': 'search_flights'}\n</tool_call>\n"     },     {       "from": "tool",       "value": "<tool_response>\n{\"name\": \"search_flights\", \"content\": {\"flights\": [{\"flight_number\": \"AA118\", \"airline\": \"American Airlines\", \"departure_airport\": \"LAX\", \"departure_city\": \"Los Angeles\", \"departure_date\": \"2023-07-10T08:00:00Z\", \"departure_terminal\": \"4\", \"arrival_airport\": \"AKL\", \"arrival_city\": \"Auckland\", \"arrival_date\": \"2023-07-11T14:25:00Z\", \"arrival_terminal\": \"I\", \"duration\": \"17h 25m\", \"class\": \"Economy\", \"price\": \"$950\", \"flexible_cancellation\": true}, {\"flight_number\": \"UA796\", \"airline\": \"United Airlines\", \"departure_airport\": \"LAX\", \"departure_city\": \"Los Angeles\", \"departure_date\": \"2023-07-10T11:30:00Z\", \"departure_terminal\": \"7\", \"arrival_airport\": \"AKL\", \"arrival_city\": \"Auckland\", \"arrival_date\": \"2023-07-11T18:30:00Z\", \"arrival_terminal\": \"I\", \"duration\": \"16h 0m\", \"class\": \"Economy\", \"price\": \"$975\", \"flexible_cancellation\": true}, {\"flight_number\": \"NZ6\", \"airline\": \"Air New Zealand\", \"departure_airport\": \"LAX\", \"departure_city\": \"Los Angeles\", \"departure_date\": \"2023-07-10T21:00:00Z\", \"departure_terminal\": \"B\", \"arrival_airport\": \"AKL\", \"arrival_city\": \"Auckland\", \"arrival_date\": \"2023-07-12T05:20:00Z\", \"arrival_terminal\": \"I\", \"duration\": \"13h 20m\", \"class\": \"Economy\", \"price\": \"$1020\", \"flexible_cancellation\": true}], \"return_flights\": [{\"flight_number\": \"AA117\", \"airline\": \"American Airlines\", \"departure_airport\": \"AKL\", \"departure_city\": \"Auckland\", \"departure_date\": \"2023-07-24T14:10:00Z\", \"departure_terminal\": \"I\", \"arrival_airport\": \"LAX\", \"arrival_city\": \"Los Angeles\", \"arrival_date\": \"2023-07-24T06:00:00Z\", \"arrival_terminal\": \"4\", \"duration\": \"12h 50m\", \"class\": \"Economy\", \"price\": \"$950\", \"flexible_cancellation\": true}, {\"flight_number\": \"UA797\", \"airline\": \"United Airlines\", \"departure_airport\": \"AKL\", \"departure_city\": \"Auckland\", \"departure_date\": \"2023-07-24T19:15:00Z\", \"departure_terminal\": \"I\", \"arrival_airport\": \"LAX\", \"arrival_city\": \"Los Angeles\", \"arrival_date\": \"2023-07-24T10:15:00Z\", \"arrival_terminal\": \"7\", \"duration\": \"11h 0m\", \"class\": \"Economy\", \"price\": \"$975\", \"flexible_cancellation\": true}, {\"flight_number\": \"NZ5\", \"airline\": \"Air New Zealand\", \"departure_airport\": \"AKL\", \"departure_city\": \"Auckland\", \"departure_date\": \"2023-07-24T19:30:00Z\", \"departure_terminal\": \"I\", \"arrival_airport\": \"LAX\", \"arrival_city\": \"Los Angeles\", \"arrival_date\": \"2023-07-24T10:30:00Z\", \"arrival_terminal\": \"B\", \"duration\": \"12h 0m\", \"class\": \"Economy\", \"price\": \"$1020\", \"flexible_cancellation\": true}]}}\n</tool_response>\n"     },     {       "from": "gpt",       "value": "Here are the available flight options with flexible cancellation policies for your kayaking trip from Los Angeles to Auckland:\n\n**Outbound Flights (Los Angeles to Auckland):**\n1. **American Airlines AA118**\n   - Departure: July 10, 2023, 8:00 AM from LAX Terminal 4\n   - Arrival: July 11, 2023, 2:25 PM at AKL Terminal I\n   - Duration: 17h 25m\n   - Price: $950\n\n2. **United Airlines UA796**\n   - Departure: July 10, 2023, 11:30 AM from LAX Terminal 7\n   - Arrival: July 11, 2023, 6:30 PM at AKL Terminal I\n   - Duration: 16h 0m\n   - Price: $975\n\n3. **Air New Zealand NZ6**\n   - Departure: July 10, 2023, 9:00 PM from LAX Terminal B\n   - Arrival: July 12, 2023, 5:20 AM at AKL Terminal I\n   - Duration: 13h 20m\n   - Price: $1020\n\n**Return Flights (Auckland to Los Angeles):**\n1. **American Airlines AA117**\n   - Departure: July 24, 2023, 2:10 PM from AKL Terminal I\n   - Arrival: July 24, 2023, 6:00 AM at LAX Terminal 4\n   - Duration: 12h 50m\n   - Price: $950\n\n2. **United Airlines UA797**\n   - Departure: July 24, 2023, 7:15 PM from AKL Terminal I\n   - Arrival: July 24, 2023, 10:15 AM at LAX Terminal 7\n   - Duration: 11h 0m\n   - Price: $975\n\n3. **Air New Zealand NZ5**\n   - Departure: July 24, 2023, 7:30 PM from AKL Terminal I\n   - Arrival: July 24, 2023, 10:30 AM at LAX Terminal B\n   - Duration: 12h 0m\n   - Price: $1020\n\nAll the listed prices are in economy class with flexible cancellation policies. Please review the options and let me know if you would like to proceed with booking any of these flights or if you need further assistance."     }   ],   "category": "E-commerce Platforms",   "subcategory": "Kayak",   "task": "Flight Search" }`

#### [Gorilla OpenFunctions-v2 LLM](https://huggingface.co/gorilla-llm/gorilla-openfunctions-v2)

- **Overview**:
    
    - **State-of-the-Art Performance:** Gorilla OpenFunctions-v2 is an open-source Large Language Model (LLM) that offers advanced function-calling capabilities, comparable to GPT-4.
    - **Extended Chat Completion:** Extends LLM chat completion with the ability to generate executable API calls from natural language instructions and relevant API contexts.
- **Key Features**:
    
    - **Multi-Function Support:**
        - Allows selection from multiple available functions based on user instructions, offering flexibility and adaptability within a single prompt.
    - **Parallel Function Calling:**
        - Supports executing the same function multiple times with different parameter values, streamlining workflows needing simultaneous function calls.
    - **Combined Multi & Parallel Functionality:**
        - Executes both multi-function and parallel function calls in one chat completion call, handling complex API call scenarios in a single prompt for efficient, high-capability outputs.
    - **Expanded Data Type Support:**
        - Enhanced compatibility with diverse programming languages by supporting extensive data types:
            - **Python:** Supports `string`, `number`, `boolean`, `list`, `tuple`, `dict`, and `Any`.
            - **Java:** Includes support for `byte`, `short`, `int`, `float`, `double`, `long`, `boolean`, `char`, and complex types like `ArrayList`, `Set`, `HashMap`, and `Stack`.
            - **JavaScript:** Covers `String`, `Number`, `BigInt`, `Boolean`, `Array`, `Date`, `dict (object)`, and `Any`.
        - Extending beyond typical JSON schema limits, this feature allows users to leverage OpenFunctions-v2 in a straightforward plug-and-play fashion without intricate data handling or reliance on string literals.
    - **Function Relevance Detection:**
        - Minimizes irrelevant function calls by detecting whether the user’s prompt is conversational or function-oriented.
        - If no function is relevant, the model raises an “Error” message with additional guidance, helping refine requests and reducing hallucinations.
    - **Enhanced RESTful API Capabilities:**
        - Specially trained to handle RESTful API calls, Gorilla OpenFunctions-v2 optimizes interactions with widely-used services, such as Slack and PayPal.
        - This high-quality support for REST API execution boosts compatibility across a broad range of applications and services.
    - **Pioneering Open-Source Model with Seamless Integration:**
        - As the first open-source model to support multi-language, multi-function, and parallel function calls, Gorilla OpenFunctions-v2 stands at the forefront of function calling in LLMs.
        - Integrates effortlessly into diverse applications, making it a seamless drop-in replacement that requires minimal setup.
    - **Broad Application Compatibility:**
        - Gorilla OpenFunctions-v2’s versatility supports a wide range of platforms, from social media like Instagram to delivery and utility services such as Google Calendar, Stripe, and DoorDash.
        - Its adaptability makes it a top choice for developers aiming to expand functional capabilities across multiple sectors with ease.
- The figure below ([source](https://huggingface.co/gorilla-llm/gorilla-openfunctions-v2)) highlights some of the key features of OpenFunctions-v2:
    

![](https://aman.ai/primers/ai/assets/agents/OpenFunctionsV2.jpg)

###### Example

- The example below demonstrates function calling, where the LLM interprets a natural language prompt to generate an API request. Given the user’s request for weather data at specific coordinates, the model formulates an API call with precise parameters, enabling automated data retrieval.

![](https://aman.ai/images/copy.png)

`"User": "Can you fetch me the weather data for the coordinates  37.8651 N, 119.5383 W, including the hourly forecast for temperature,  wind speed, and precipitation for the next 10 days?"  "Function":  {     ...     "parameters":      {         "type": "object",          "properties":          {             "url":              {                 "type": "string",                  "description": "The API endpoint for fetching weather                 data from the Open-Meteo API for the given latitude                 and longitude, default                  https://api.open-meteo.com/v1/forecast"             }             ...         }     } }  "GPT-4 output": {     "name": "requests.get",     "parameters": {         "params":          {             "latitude": "37.8651",             "longitude": "-119.5383",             "forecast_days": 10         },     } }`

#### Best Practices, Guidelines, and Limitations

- Per [Anthropic](https://x.com/swyx/status/1775993946935906645), when using tools with Claude, it’s important to follow best practices, understand limitations, and optimize tool design to ensure effective interactions.

##### Choosing the Right Model for Tool Use

- **Claude 3 Opus** is best for complex tool use, as it can handle multiple tools simultaneously and detect missing arguments. It will ask for clarification when necessary.
- **Claude 3 Haiku** is better suited for simple tool use but defaults to using tools more frequently—even when unnecessary. It will also infer missing parameters rather than asking for clarification.

##### Tool Usage Limits

- **Handling large toolsets**
    - Claude can accurately select from over 250+ tools, as long as the user query includes all required parameters.
    - This limit applies regardless of tool complexity. Complex tools typically have numerous parameters or deeply nested schemas.
- **Optimizing tool complexity**
    - Claude performs better with simpler tools.
    - To improve accuracy, avoid deeply nested JSON objects and reduce the number of required inputs.

##### Sequential vs. Parallel Tool Execution

- Claude generally prefers sequential tool execution—using one tool at a time, analyzing the output, and then deciding on the next step.
- While parallel tool use is possible, it may lead to:
    - Missing dependencies (e.g., filling in placeholder values for parameters that depend on previous outputs).
    - Unnecessary tool invocations.
- Best practice: Design workflows that encourage sequential tool execution to improve accuracy.

##### Error Handling and Retries

- If Claude’s tool request is invalid, returning an error response will often prompt it to retry with missing parameters filled in.
- However, after 2-3 failed attempts, Claude may stop retrying and instead return an apology message.

#### Designing Effective Tool Interfaces

- Tools are an essential part of agentic systems, enabling Claude to interact with external services and APIs. However, tools should be carefully designed, just like prompts, to ensure clarity and usability. By thoughtfully designing your tools, you can reduce errors, improve model accuracy, and create more efficient agent workflows, per the below guidelines per [Anthropic’s blog on building effective agents](https://www.anthropic.com/engineering/building-effective-agents).

##### Choosing the Right Tool Format

- The way tools are structured can significantly impact Claude’s accuracy. Some formats are harder for an LLM to generate correctly than others:
- **Diff vs. Full Rewrite**:
    - Writing a diff requires pre-determining line changes, which is error-prone.
    - A full rewrite avoids this complexity.
- **Markdown vs. JSON for structured output**:
    - JSON requires escaping newlines and quotes, increasing error risks.
    - Markdown is often easier for Claude to handle accurately.

##### Best Practices for Tool Design

- Give the model enough tokens to “think” before committing to an output.
- Stick to familiar formats that naturally occur in publicly available text (e.g., Markdown over escaped JSON).
- Minimize formatting overhead—avoid requiring Claude to track line counts or escape large text blocks.

##### Improving Tool Usability

- Make tool descriptions intuitive—think like a human: Would a developer immediately understand how to use this tool?
- Refine parameter names and descriptions—write them as if documenting an API for a junior developer.
- Test extensively—run real-world inputs in a sandbox environment to uncover edge cases and refine accordingly.
- Poka-yoke (mistake-proof) your tools.
    - **Example:** Instead of using relative file paths, require absolute paths to avoid errors when switching directories.

### Planning

- Planning is a foundational design pattern that empowers an AI system, typically a LLM, to autonomously determine a sequence of actions or steps needed to accomplish complex tasks. Through this dynamic decision-making process, the AI breaks down broad objectives into smaller, manageable steps, executing them in a structured sequence to produce coherent, often intricate outputs. This document delves into the importance of Planning in agentic AI design, illustrating its function with examples and examining its current capabilities alongside its limitations.
- As a transformative design pattern, Planning grants LLMs the ability to autonomously devise and execute plans/strategies for completing tasks. Although current implementations can still exhibit unpredictability, Planning can empower an AI agent with enhanced creative problem-solving capability that enables it to navigate tasks in unforeseen, innovative ways.
- The power of Planning lies in its flexibility and adaptability. When effectively implemented, Planning enables an AI to respond to unforeseen conditions, make informed decisions about task progression, and select tools best suited to each step. This autonomy, however, introduces unpredictability in the agent’s behavior and outcomes.

#### Overview

- Planning in agentic AI refers to the AI’s ability to autonomously design a task plan, selecting the steps necessary to achieve a given goal. Unlike more deterministic processes, Planning involves a level of adaptability, allowing the AI to adjust its approach based on available tools, task requirements, and unforeseen constraints.
- For example, if an AI agent is tasked with conducting online research on a specific topic, it can independently generate a series of subtasks. These might include identifying key subtopics, gathering relevant information from reputable sources, synthesizing findings, and compiling the research into a cohesive report. Through Planning, the agent does not simply execute pre-programmed instructions but rather determines the optimal sequence of actions to meet the objective.

#### Example

- Agentic Planning becomes especially critical when tasks are multifaceted and cannot be completed in a single step. In such cases, an LLM-driven agent dynamically designs a sequence of steps to accomplish the overarching goal. An example from the HuggingGPT paper illustrates this approach: if the objective is to render a picture of a girl in the same pose as a boy in an initial image, the AI might decompose the task as follows:
    
    1. **Step 1:** Detect the pose in the initial picture of the boy using a pose-detection tool, producing a temporary output file (e.g., `temp1`).
    2. **Step 2:** Use a pose-to-image tool to generate an image of a girl in the detected pose from `temp1`, yielding the final output.
- In this structured format, the AI specifies each action step, defining the tool to use, the input file, and the expected output. This process then triggers software that invokes the necessary tools in the designated sequence to complete the task successfully. The agent’s autonomous Planning ability facilitates this multi-step workflow, demonstrating its capacity to tackle intricate, non-linear tasks.
- The following figure ([source](https://arxiv.org/abs/2303.17580)) offers a visual overview of the above process:

![](https://aman.ai/primers/ai/agents/assets/agents/Planning.png)

#### Planning vs. Deterministic Approaches

- Planning is not required in every agentic workflow. For simpler tasks or those that follow a predefined sequence, a deterministic, step-by-step approach may suffice. For instance, if an agent is programmed to reflect on and revise its output a fixed number of times, it can execute this series of steps without needing adaptive planning.
- However, for complex or open-ended tasks where it is difficult to predefine the necessary sequence, Planning allows the AI to dynamically decide on the appropriate steps. This adaptive approach is especially valuable for tasks that may involve unexpected challenges or require the agent to select from a range of tools and methods to reach the best outcome.

### Multi-agent Collaboration

#### Background

- Multi-agent collaboration has emerged as a pivotal AI design pattern for executing complex tasks by breaking them down into manageable subtasks. By assigning these subtasks to specialized agents—each acting as a software engineer, product manager, designer, QA engineer, etc.—multi-agent collaboration mirrors the structure of a well-coordinated team, where each agent performs specific, designated roles. These agents, whether built by prompting a single LLM in various ways or by employing multiple LLMs, can carry out their assigned tasks with tailored capabilities. For instance, prompting an LLM to act as a “software engineer” by instructing it to “write clear, efficient code” enables it to focus solely on that aspect, thereby honing its output to the requirements of the software engineering subtask.
- This approach has strong parallels in multi-threading, where complex programs are divided across multiple processors or threads to be executed concurrently, improving efficiency and performance. The agentic model thus offers a divide-and-conquer structure that enables AI systems to manage intricate workflows by breaking them into smaller, role-based actions.

#### Motivation

- The adoption of multi-agent systems in AI is driven by several key factors:

1. **Demonstrated Effectiveness:** The multi-agent approach has consistently produced positive results across various projects. Ablation studies, such as those presented in the AutoGen paper, have confirmed that multi-agent systems often yield superior performance compared to single-agent configurations for complex tasks. The multi-agent structure allows each agent to focus narrowly on a specific subtask, which is conducive to better performance than attempting to accomplish the entire task in a monolithic approach.
    
2. **Enhanced Task Focus and Optimization:** Despite recent advancements allowing some LLMs to accept extensive input contexts (e.g., Gemini 1.5 Pro with 1 million tokens), a multi-agent system still holds distinct advantages. Each agent can be directed to focus on one isolated subtask at a time, enhancing its ability to execute that task with precision. By setting tailored expectations—such as prioritizing code clarity for a “software engineer” agent over scalability or security—developers can optimize the output of each subtask according to specific project requirements.
    
3. **Decomposition of Complex Tasks:** Beyond immediate efficiency gains, multi-agent systems offer a powerful conceptual framework for managing complex tasks by breaking them down into smaller, more manageable subtasks. This design pattern enables developers to simplify workflows while simultaneously enhancing communication and task alignment among agents. Much like a manager in a company would assign tasks to specialized employees to address different facets of a project, multi-agent systems use this human organizational structure as a blueprint for assigning AI tasks.
    

- This design abstraction supports developers in “hiring” agents for distinct roles and assigning tasks according to their “specializations,” with each agent independently executing its workflow, utilizing memory to track interactions, and potentially collaborating with other agents as necessary. Multi-agent workflows can involve dynamic elements like planning and tool use, enabling agents to respond adaptively and collectively in complex scenarios through interconnected calls and message passing.

### Implementation

- While managing human teams has inherent challenges, applying similar organizational strategies to multi-agent AI systems is not only manageable but also offers low-risk flexibility; any issues in an AI agent’s performance are easily rectified. Emerging frameworks such as AutoGen, CrewAI, and LangGraph provide robust platforms for developing and implementing multi-agent systems tailored to diverse applications. Additionally, open-source projects like [ChatDev](https://aman.ai/primers/ai/agents/#chatdev-communicative-agents-for-software-development) allow developers to experiment with multi-agent setups in a virtual “software company” environment, offering valuable insights into the collaborative potential of AI agents. Such tools represent the leading edge of multi-agent technology, providing a foundation for AI-driven task decomposition and collaboration.
    
- In summary, multi-agent collaboration is a compelling and effective AI design pattern that leverages agent specialization, task decomposition, and focused prompting to enable more efficient handling of complex tasks. As multi-agent frameworks continue to advance, they are likely to become foundational in AI-driven workflows, providing developers with both the structure and flexibility to tackle increasingly sophisticated projects.
    

## Agentic Workflow Patterns

- This section examines common patterns observed in the development of agentic systems, beginning with the fundamental component—augmented LLMs—and progressively increasing in complexity, from structured workflows to autonomous agents.
- Each of these patterns enhances the efficiency and effectiveness of agentic systems by structuring tasks in a way that optimally leverages LLM capabilities.

### Prompt Chaining

- Prompt chaining involves decomposing a task into a sequence of steps, where each LLM call processes the output of the preceding step. Programmatic validation mechanisms (referred to as “gates”) may be applied at intermediate stages to ensure procedural accuracy.
- **Optimal Use Cases:**
    - This workflow is beneficial when a task can be clearly divided into structured subtasks. It prioritizes improved accuracy over latency by simplifying each LLM call.
- **Examples:**
    - Generating marketing copy and subsequently translating it into another language.
    - Creating an outline for a document, verifying its adherence to specific criteria, and then generating the final document based on the outline.

### Routing

- Routing involves classifying an input and directing it to an appropriate specialized task. This approach facilitates the separation of concerns and enables more precise prompts. Without routing, optimizing for one input type may degrade performance for others.
    
- **Optimal Use Cases:**
    - Routing is effective for tasks with distinct categories that require specialized handling. Classification can be performed by either an LLM or a traditional classification model.
- **Examples:**
    - Categorizing customer service queries (e.g., general inquiries, refund requests, technical support) and routing them to appropriate downstream processes.
    - Assigning simple queries to lightweight models (e.g., Claude 3.5 Haiku) while directing complex queries to more capable models (e.g., Claude 3.5 Sonnet) to balance efficiency and cost.

#### Parallelization

- Parallelization entails multiple LLM instances working on a task simultaneously, with results aggregated programmatically. It manifests in two key forms:
    - **Sectioning:** Dividing a task into independent subtasks that can be processed in parallel.
    - **Voting:** Running the same task multiple times to obtain diverse outputs.
- **Optimal Use Cases:**
    - Parallelization is advantageous when tasks can be effectively subdivided to enhance speed or when multiple perspectives improve reliability.
- **Examples:**
    - **Sectioning:**
        - Implementing safeguards where one LLM processes user queries while another screens for inappropriate content.
        - Automating model performance evaluations by assigning different evaluation criteria to separate LLM calls.
    - **Voting:**
        - Conducting code reviews for security vulnerabilities, with multiple prompts assessing different aspects of the code.
        - Evaluating content for appropriateness by using multiple assessments to balance false positives and negatives.

#### Orchestrator-Workers

- This workflow features a central LLM that dynamically decomposes tasks, assigns them to worker LLMs, and synthesizes their results.
    
- **Optimal Use Cases:**
    - This approach is suited for complex tasks where the required subtasks are not predefined but must be determined dynamically. Unlike parallelization, which follows a fixed structure, this workflow offers greater adaptability.
- **Examples:**
    - Software development tools that implement complex modifications across multiple files.
    - Research tasks requiring information retrieval and synthesis from diverse sources.

#### Evaluator-Optimizer

- In this iterative workflow, one LLM generates responses while another evaluates and refines them in a continuous loop.
    
- **Optimal Use Cases:**
    - This pattern is particularly effective when clear evaluation criteria exist and iterative refinement adds measurable value. It is most beneficial when:
        - Human feedback has been shown to enhance LLM-generated outputs.
        - The LLM itself is capable of providing constructive feedback.
- **Examples:**
    - Literary translation, where an evaluator LLM can provide nuanced critiques that improve translation quality.
    - Complex search tasks requiring iterative refinement, where an evaluator assesses whether additional searches are necessary to obtain comprehensive information.

## Single-Agent Systems vs. Multi-Agent Systems

- While multi-agent systems may be suited for specific use cases—such as those involving role-based separation or access to privileged information—single-agent systems present a more elegant and pragmatic solution in many scenarios. Their capacity to maintain holistic context, reduce system complexity, and adapt fluidly to varied tasks positions them as a strong alternative to the prevailing trend of multi-agent design. Specifics below.

### Preservation of Context

- While multi-agent systems offer structural advantages in dividing and distributing tasks, there are compelling reasons why single-agent systems are often the preferred choice in particular contexts. These advantages stem largely from considerations of system coherence, efficiency, and maintainability.
- First, single-agent systems mitigate the risk of context fragmentation. In multi-agent setups, information must be passed between distinct agents, each with its own prompt and action space. This process can lead to significant information loss or misinterpretation, especially when intermediate outputs are condensed or abstracted. In contrast, a single-agent system maintains a unified internal context throughout its operation, thereby preserving the continuity and depth of understanding necessary for complex reasoning or nuanced decision-making.

### Simplicity and Maintainability

- System design and maintenance are notably simplified in single-agent architectures. Each additional agent in a multi-agent framework typically requires a separate codebase or configuration, increasing the burden of integration and long-term upkeep. A single-agent system, by consolidating all functionality within a unified framework, allows for a leaner development cycle and reduces the cognitive load for developers maintaining the system.

### Flexibility in Problem Solving

- Single-agent systems offer enhanced flexibility in problem-solving. Unlike multi-agent systems, where each agent is constrained by a specialized role and toolset, a well-configured single agent can dynamically apply a broad set of tools and methods as needed. This adaptability is particularly valuable when real-world tasks deviate from predefined structures or require interleaving of capabilities that would otherwise be siloed across agents.

### Feasibility with Modern Tools

- Advances in LLMs and prompt engineering have made it increasingly feasible to replicate the effectiveness of multi-agent coordination within a single-agent framework. With access to long-context models and robust prompting techniques, a single agent can internalize and execute complex workflows that previously necessitated distributed architectures.

## Model Context Protocol (MCP)

### Overview

- The [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) is an open protocol designed to standardize how applications provide context to LLMs. Think of MCP as the AI equivalent of a USB-C port: just as USB-C creates a standardized way for devices to connect with various peripherals, MCP provides a uniform approach to integrating AI models with different data sources and tools. The following figure ([source](https://norahsakal.com/blog/mcp-vs-api-model-context-protocol-explained/)) illustrates MCP’s client-server architecture: A standardized bridge connecting AI applications with local and remote data sources akin to how a USB-C port enables seamless connectivity between multiple peripherals without requiring separate adapters.

![Placeholder for The Model Context Protocol (MCP) Image](https://aman.ai/primers/ai/assets/agents/mcp_usbc.png)

- MCP has rapidly become the de facto standard for integrating LLMs with external tools and data sources. Its AI-native approach, strong industry backing, and pragmatic design choices have positioned it as the most viable open protocol for agent-based applications. As adoption continues to grow, MCP is likely to overtake existing standards like OpenAPI, solidifying its role in the future of AI-powered workflows.
    
- The following figure ([source](https://www.philschmid.de/mcp-introduction)) contrasts LLM tool integrations before and after MCP, highlighting how MCP simplifies interactions through a unified API layer.
    

![Placeholder for Before and After MCP Image](https://aman.ai/primers/ai/assets/agents/mcp_before_after.jpg)

### Why MCP?

- MCP is essential for building AI-powered agents and complex workflows. LLMs often need access to external data and tools, and MCP facilitates this by offering:
    - A growing library of pre-built integrations that LLMs can seamlessly interact with.
    - Flexibility to switch between different LLM providers and vendors without reconfiguring integrations.
    - Best practices for securing data while keeping it within a controlled infrastructure.
- Beyond these practical advantages, MCP has also emerged as the dominant open standard for AI agent integrations due to several key factors:
    - **MCP is an AI-Native adaptation of an existing idea:** Unlike older interoperability standards such as OpenAPI, OData, and GraphQL, MCP was built specifically for LLM-based applications, making it more effective in AI-driven workflows.
    - **MCP is backed by a major player (Anthropic):** Large organizations supporting a protocol significantly boost its chances of success, and Anthropic’s involvement has given MCP a strong foundation.
    - **Anthropic has a strong developer AI brand:** The company’s focus on high-quality developer tooling has positioned MCP as the leading choice among AI engineers and application developers.
    - **MCP is built on the [Language Server Protocol (LSP)](https://learn.microsoft.com/en-us/visualstudio/extensibility/language-server-protocol?view=vs-2022):** By leveraging an already successful and widely adopted architecture, MCP avoids the pitfalls of entirely new protocol designs and benefits from proven best practices.
    - **MCP was dogfooded with complete tooling:** From its launch, MCP was tested and refined with first-party clients like Claude Desktop, 19 reference server implementations, and essential development tools such as MCP Inspector and SDKs in Python and TypeScript.

### General Architecture

- At its core, MCP follows a client-server architecture, enabling a host application to connect to multiple MCP servers. Below is an overview of the key components:
    - **MCP Hosts:** Programs such as Claude Desktop, IDEs, and AI tools that need to access data through MCP.
    - **MCP Clients:** Protocol clients that maintain one-to-one connections with MCP servers.
    - **MCP Servers:** Lightweight programs that expose specific capabilities using the standardized MCP framework.
    - **Local Data Sources:** Files, databases, and services on a user’s computer that MCP servers can securely access.
    - **Remote Services:** External systems, such as APIs, that MCP servers can connect to over the internet.
- **Client Example Using Python:** The following code snippet ([source](https://www.philschmid.de/mcp-introduction)) shows how to create a basic MCP client that initializes a session, lists available tools, and calls one of the server’s tools using standard I/O transport.

![](https://aman.ai/images/copy.png)

`from mcp import ClientSession, StdioServerParameters from mcp.client.stdio import stdio_client  server_params = StdioServerParameters(     command="python", args=["example_server.py"] )  async with stdio_client(server_params) as (read, write):     async with ClientSession(read, write) as session:         await session.initialize()         tools = await session.list_tools()         result = await session.call_tool("tool-name", arguments={"arg1": "value"})`

- **Server Example Using FastMCP (Python):** The following code snippet ([source](https://www.philschmid.de/mcp-introduction)) illustrates how to create a simple MCP server using FastMCP, which exposes a tool, a data resource, and a reusable prompt template.

![](https://aman.ai/images/copy.png)

`from fastmcp import FastMCP  mcp = FastMCP("Demo")  @mcp.tool() def add(a: int, b: int) -> int:     return a + b  @mcp.resource("greeting://{name}") def get_greeting(name: str) -> str:     return f"Hello, {name}!"  @mcp.prompt() def review_code(code: str) -> str:     return f"Please review this code:\n\n{code}"`

### How MCP Works

- The following figure ([source](https://blog.dailydoseofds.com/p/visual-guide-to-model-context-protocol)) provides an animated overview of the MCP Flow.

![Placeholder for Animated Overview of MCP Flow](https://aman.ai/primers/ai/assets/agents/overview.gif)

- MCP operates through a structured client-server interaction that involves capability exchange and bidirectional communication:
    1. **Capability Exchange:** The MCP client initializes a request to the server to learn about its available capabilities.
    2. **Server Response:** The server provides details about its available tools, resources, and prompts.
    3. **Notification & Acknowledgment:** The client acknowledges the connection and facilitates further communication.
- The following figure ([source](https://www.philschmid.de/mcp-introduction)) the division of responsibilities between the MCP client and server, along with examples of Tools, Resources, and Prompts.

![Placeholder for MCP Client-Server Capability Breakdown](https://aman.ai/primers/ai/assets/agents/client_server_capabilities.jpg)

- Unlike traditional APIs, MCP communication is two-way, allowing:
    - Servers to leverage client-side AI capabilities (such as LLM-based completions) without requiring API keys.
    - Clients to maintain control over model access and permissions while enabling interoperability with various services.
- **Practical Example with Gemini + MCP Client:** The following code snippet ([source](https://www.philschmid.de/mcp-introduction)) shows how an AI model like Gemini can dynamically discover available MCP tools and send function calls, making tool invocation seamless within natural language interactions.

![](https://aman.ai/images/copy.png)

`from google import genai from google.genai import types  # Assume session is an initialized MCP ClientSession mcp_tools = await session.list_tools() tools = types.Tool(function_declarations=[     {         "name": tool.name,         "description": tool.description,         "parameters": tool.inputSchema,     }     for tool in mcp_tools.tools ])  # Send tool schema to Gemini model response = await client.aio.models.generate_content(     model="gemini-2.0-flash",     contents=[types.Content(role="user", parts=[types.Part(text="Show me available Airbnb listings in Paris.")])],     config=types.GenerateContentConfig(tools=[tools]) )`

- **Practical Example with Gemini and Python uSDK:** The following end-to-end implementation ([source](https://www.philschmid.de/mcp-introduction)) demonstrates an AI agent loop that connects to an MCP server via stdio, retrieves tool specs, sends them to Gemini 2.0 Flash, handles function calls from the model, and completes the interaction using the Google Python SDK.

![](https://aman.ai/images/copy.png)

`from typing import List from google import genai from google.genai import types from mcp import ClientSession, StdioServerParameters from mcp.client.stdio import stdio_client import os  client = genai.Client(api_key=os.getenv("GEMINI_API_KEY")) model = "gemini-2.0-flash"  server_params = StdioServerParameters(     command="npx",     args=["-y", "@openbnb/mcp-server-airbnb"],     env=None, )  async def agent_loop(prompt: str, client: genai.Client, session: ClientSession):     contents = [types.Content(role="user", parts=[types.Part(text=prompt)])]     await session.initialize()      mcp_tools = await session.list_tools()     tools = types.Tool(function_declarations=[         {             "name": tool.name,             "description": tool.description,             "parameters": tool.inputSchema,         }         for tool in mcp_tools.tools     ])      response = await client.aio.models.generate_content(         model=model,         contents=contents,         config=types.GenerateContentConfig(             temperature=0,             tools=[tools],         ),     )      contents.append(response.candidates[0].content)      turn_count = 0     max_tool_turns = 5     while response.function_calls and turn_count < max_tool_turns:         turn_count += 1         tool_response_parts: List[types.Part] = []          for fc_part in response.function_calls:             tool_name = fc_part.name             args = fc_part.args or {}             try:                 tool_result = await session.call_tool(tool_name, args)                 if tool_result.isError:                     tool_response = {"error": tool_result.content[0].text}                 else:                     tool_response = {"result": tool_result.content[0].text}             except Exception as e:                 tool_response = {"error": f"Tool execution failed: {type(e).__name__}: {e}"}              tool_response_parts.append(                 types.Part.from_function_response(                     name=tool_name, response=tool_response                 )             )          contents.append(types.Content(role="user", parts=tool_response_parts))          response = await client.aio.models.generate_content(             model=model,             contents=contents,             config=types.GenerateContentConfig(                 temperature=1.0,                 tools=[tools],             ),         )         contents.append(response.candidates[0].content)      return response  async def run():     async with stdio_client(server_params) as (read, write):         async with ClientSession(read, write) as session:             prompt = "I want to book an apartment in Paris for 2 nights. 03/28 - 03/30"             res = await agent_loop(prompt, client, session)             return res  res = await run() print(res.text)`

### MCP vs. API

- **Single Protocol:** Traditional APIs require distinct authentication and integration for each service, akin to managing multiple keys for different locks. This approach increases complexity, maintenance overhead, and integration effort for developers. MCP acts as a standardized “connector,” meaning integrating one MCP provides potential access to multiple tools and services. The following figure ([source](https://norahsakal.com/blog/mcp-vs-api-model-context-protocol-explained/)) illustrates how traditional APIs require separate integrations for each tool, whereas MCP provides a unified, standardized connection for seamless interoperability.

![Placeholder for The Model Context Protocol (MCP) Image](https://aman.ai/primers/ai/assets/agents/mcp_apikey.png)

- **Dynamic Discovery:** MCP allows AI models to dynamically discover and interact with available tools without requiring hard-coded knowledge of each integration.
- **Two-Way Communication:** MCP supports persistent, real-time two-way communication, similar to WebSockets. The AI model can both retrieve information and trigger actions dynamically.

#### Comparative Analysis

|**Feature**|**MCP**|**Traditional API**|
|---|---|---|
|Integration Effort|Single, standardized integration|Separate integration per API|
|Real-Time Communication|✅ Yes|❌ No|
|Dynamic Discovery|✅ Yes|❌ No|
|Scalability|Easy (plug-and-play)|Requires additional integrations|
|Security & Control|Consistent across tools|Varies by API|
|Two-Way Communication|✅ Yes|❌ No|

#### When to Use MCP vs. Traditional APIs

- **Use MCP when:**
    - You need flexible, AI-native integration with multiple tools and data sources.
    - Your AI model benefits from real-time capability discovery.
    - You want a scalable, standardized way to manage AI interactions.
- **Stick with traditional APIs when:**
    - Your use case requires precise, deterministic behavior with strict limits.
    - You need maximum predictability with minimal context autonomy.
    - Fine-grained control over interactions is essential.

### Security, Updates, and Authentication

- MCP is a living protocol, actively evolving. Key updates include:
    - **OAuth 2.1 Authentication:** The protocol mandates OAuth 2.1 for secure HTTP server authentication.
    - **Transport Upgrades:** A shift from SSE to Streamable HTTP improves efficiency and enables JSON-RPC batching.
    - **Richer Tool Metadata:** Tools now support annotations such as side-effect classification (read-only vs. write operations), helping AI reason better about tool usage.

### Getting Started with MCP: High-Level Steps

- Integrating MCP into an AI-driven workflow involves several key steps:
    
    1. **Define Capabilities:** Clearly outline what your MCP server will offer, including available tools, data sources, and functionalities.
    2. **Implement MCP Layer:** Develop the MCP integration layer by adhering to the standardized MCP protocol specifications to ensure seamless interoperability.
    3. **Choose Transport Method:** Decide on the communication method between MCP clients and servers, using either local transport (stdio) or remote transport (Server-Sent Events/WebSockets) based on your requirements.
    4. **Create Resources and Tools:** Develop the specific data sources, services, and functionalities that your MCP server will expose for AI-driven applications.
    5. **Set Up Clients:** Establish secure and stable connections between MCP servers and clients, ensuring smooth data exchange and integration with AI agents.
- By following these steps, developers can create powerful, flexible, and AI-enhanced systems that leverage MCP for dynamic and intelligent interactions with external tools and data sources.
    

### Use-Cases of MCP in Real-World Development Scenarios

- Per this Reddit [post](https://www.reddit.com/r/cursor/comments/1j3nnbz/comment/mg1p89i/?context=3&share_id=qcdVjC99AK-08LRsVqsfk&utm_medium=ios_app&utm_name=ioscss&utm_source=share&utm_term=1), MCP is particularly powerful in software development workflows, where engineers often juggle multiple tasks beyond just writing code. From ticket retrieval to automated commits and intelligent pull request creation, MCP eliminates the friction of routine development tasks, allowing engineers to focus on coding rather than administrative overhead.
- Here’s how MCP streamlines the entire development lifecycle:

#### Automating Feature Development from Ticket to Implementation

- When starting a new feature, developers typically need to reference multiple sources—project management tools for assigned tickets, design files for UI specifications, and documentation for technical requirements. MCP simplifies this process by:
    - Fetching currently assigned tickets from Jira, GitHub Issues, or Linear, automatically understanding which ticket is being worked on.
    - Extracting relevant requirements from the ticket without manual searching.
    - Retrieving design specifications from Figma, analyzing images and text descriptions to provide necessary dimensions, fonts, and layouts.
    - Allowing an AI-powered coding assistant like Cursor to immediately start implementing the feature based on all retrieved inputs.

#### Intelligent Ticket and Task Management

- During implementation, developers often discover inconsistencies or missing details in tickets. Instead of manually updating them, MCP-enabled servers allow:
    - Cursor to update tickets dynamically, correct issues, and create new subtasks when necessary.
    - Auto-detection of missing requirements and automatic messaging to the ticket’s author for clarifications.
    - Seamless linking of related issues and tasks across different project management systems.

#### Automated Communication with Relevant Stakeholders

- If questions arise about the ticket or design, MCP servers determine the relevant author (e.g., a designer or product manager) and send automated messages via Slack or other communication tools. This eliminates delays caused by manually tracking down stakeholders.

#### Smart Staging, Commit Messages, and PR Creation

- Once development is complete, MCP enhances the process of committing code and opening pull requests by:
    - Staging all modified files and generating an intelligent commit message based on the staged changes.
    - Creating a pull request on platforms like GitHub or GitLab, automatically linking it to the corresponding Jira ticket.
    - Adding necessary labels, titles, and descriptions to the PR, ensuring proper documentation without manual intervention.

#### Automated Debugging and Console Log Access

- Developers frequently encounter errors while testing their implementations in a browser. MCP can:
    - Grant AI assistants real-time access to console logs, so they can analyze issues without requiring manual copy-pasting.
    - Provide automated suggestions or fixes based on detected errors.
    - Iterate on solutions in the background, accelerating the debugging process.

#### Integration with Personal Task Management Tools

- Many developers maintain personal to-do lists to track daily tasks. MCP allows seamless synchronization between personal task managers and project tracking tools:
    - A local to-do app can be MCP-enabled, automatically pulling relevant tickets and work items.
    - Developers can use natural language to prioritize tasks, delegate them, or get reminders.

#### Automated Project Announcements

- After completing a feature, developers often need to announce it to the team. MCP can:
    - Automatically draft and send announcements in Slack, summarizing the completed work.
    - Include links to the PR, Jira ticket, and relevant documentation.
    - Ensure communication is structured, timely, and contains all necessary context.

### MCP Servers List

- A curated list of ready-to-use and community-maintained servers:

#### [Awesome MCP Servers](https://github.com/punkpeye/awesome-mcp-servers)

- A curated list of MCP servers that enable AI models to securely interact with local and remote resources through standardized implementations.

#### [Model Context Protocol Servers](https://github.com/modelcontextprotocol/servers)

- A collection of reference implementations for the Model Context Protocol (MCP), showcasing how Large Language Models (LLMs) can securely access tools and data sources.

#### [Composio MCP Servers](https://mcp.composio.dev/)

- Composio offers over 100 fully managed MCP server implementations with built-in authentication, facilitating seamless integration of AI agents and LLMs with various tools and services.

#### [Smithery](https://smithery.ai/)

- Smithery is a platform to help developers find and ship agentic services that follow the Model Context Protocols (MCP) specification.
- Smithery’s mission is to make agentic services accessible and accelerate the development of agentic AI.

## Agent2Agent (A2A) Protocol

- [Agent2Agent (A2A)](https://developers.googleblog.com/en/a2a-a-new-era-of-agent-interoperability/) is an open protocol introduced by Google to enable secure, interoperable communication between AI agents—regardless of their origin, framework, or vendor. Backed by contributions from over 50 leading technology and consulting partners, including Google Cloud, Atlassian, Salesforce, and Accenture, A2A establishes a universal mechanism for AI agents to discover one another, exchange information, and collaborate on tasks across diverse applications and platforms.
- Designed to break down the silos of traditional enterprise environments, A2A facilitates the creation of fluid, multi-agent ecosystems that support dynamic collaboration, long-running workflows, and rich user experiences. It empowers organizations to build intelligent, interoperable, and secure systems that span multiple enterprise platforms without requiring tight integration or risking vendor lock-in.
- At its core, A2A addresses a fundamental challenge in the evolution of agentic AI: enabling agents to interact across organizational and technological boundaries. Its open architecture promotes agent autonomy, enhances productivity, reduces duplication of effort, and fuels innovation in complex enterprise-grade environments.
- A2A complements existing protocols like the Model Context Protocol (MCP)—which provides agents with external data and tool context—by enabling direct, structured, and semantically rich communication between agents themselves. Through A2A, agents can:
    
    - Discover each other’s capabilities
    - Manage long-running, collaborative tasks
    - Exchange context and digital artifacts
    - Negotiate outputs for user interfaces
- Built on familiar standards such as HTTP, Server-Sent Events (SSE), and JSON-RPC, the A2A protocol enforces robust security by default. Its open contribution model and growing industry adoption are laying the groundwork for a thriving ecosystem of intelligent, cooperative agents poised to transform enterprise automation at scale.

### A2A Design Principles

- The protocol’s architecture is grounded in five key principles:
    
    1. **Embrace Agentic Capabilities**
        - A2A supports agents operating in unstructured, autonomous modes—even when agents do not share memory or toolsets. This enables robust, tool-agnostic agent collaboration that respects the diversity of agent implementations and modalities.
    2. **Build on Existing Standards**
        - A2A leverages widely used, mature web technologies such as:
            - HTTP for transport
            - Server-Sent Events (SSE) for real-time messaging
            - JSON-RPC for structured, language-agnostic function calls
        - This alignment ensures compatibility with existing enterprise systems and lowers the barrier to adoption.
    3. **Secure by Default**
        - A2A supports enterprise-grade authentication and authorization protocols, achieving parity with OpenAPI authentication schemes (e.g., OAuth2, API keys, bearer tokens). This allows agents to authenticate both themselves and each other within secure, policy-governed environments.
    4. **Support for Long-Running Tasks**
        - A2A is designed to support both ephemeral and extended workflows. Agents can initiate tasks that persist over hours or days, with real-time status updates, notifications, and progress messages shared throughout the lifecycle. This supports human-in-the-loop workflows, asynchronous task completion, and progressive state reporting.
    5. **Modality Agnostic**
        - The protocol supports multiple modalities beyond text, including images, video, and audio. It accommodates multimodal data streams and user experiences by enabling agents to negotiate and exchange content in forms suited to downstream UI capabilities.

### How A2A Works

![](https://aman.ai/primers/ai/assets/agents/A2A.jpg)

- The illustration above ([source](https://developers.googleblog.com/en/a2a-a-new-era-of-agent-interoperability/)) offers a visual overview of Agent2Agent communication and depicts the core dynamic between a client agent and a remote agent in an A2A-compliant system. It highlights the secure bidirectional exchange of structured tasks and content, and how mismatches or incompatibilities (like unsupported formats) are negotiated during collaboration.
    
- A2A facilitates communication between two primary roles:
    
    - **Client Agent**: Initiates the interaction, discovers the remote agent’s capabilities, and submits task requests.
    - **Remote Agent**: Responds to the request, executes tasks, and delivers artifacts or messages back to the client.
- This interaction is governed by four key capabilities:
    
    - **Secure Collaboration**: All messages and artifacts exchanged are protected with enterprise-grade authentication and authorization mechanisms.
    - **Task and State Management**: Tasks have lifecycle states and can be monitored or updated across long-running workflows.
    - **User Experience Negotiation**: Agents can communicate about UI and modality expectations, ensuring compatibility across interfaces.
    - **Capability Discovery**: Agents advertise their skills, tools, and modalities using Agent Cards, enabling intelligent routing of tasks.

### Core Protocol Mechanics

#### Capability Discovery

- Every A2A-compliant agent publishes a JSON-based Agent Card, advertising its available capabilities, supported modalities, and service endpoints.
- This enables the client agent to:
    - Programmatically discover agents relevant to a given task
    - Select remote agents based on advertised functions, tools, or domain expertise
- Agent Cards may include metadata such as:
    - Versioning info
    - Contact or fallback methods
    - Supported content types (e.g., `text/plain`, `image/png`, `video/webm`)
    - Trust level or authentication requirements

#### Task Management

- The core of the A2A interaction is the task object, a structured entity that captures:
    - Task description and parameters
    - Required capabilities or desired outputs
    - Task lifecycle state (created, in-progress, completed, failed)
    - Timestamps, identifiers, and optional user metadata
- Agents coordinate on task status updates through SSE or HTTP polling, ensuring synchronized task execution over long durations.
- Upon task completion, the remote agent emits one or more artifacts, which are structured outputs (e.g., a file, message, API call result).

#### Collaboration and Messaging

- Agents exchange context, interim results, clarifications, and feedback via message objects. These messages may contain:
    - Plain or rich content (e.g., structured JSON, UI-renderable components)
    - Contextual replies or updates (e.g., clarification requests, validation errors)
    - User or agent instructions
- All messages are delivered using SSE or HTTP endpoints, ensuring timely delivery and minimal overhead.

#### User Experience (UX) Negotiation

- Each message can contain multiple parts, each with a specified content type.
- Example parts might include:
    - A generated image (`image/png`)
    - A previewable webpage (`text/html`)
    - An embeddable iframe (`application/vnd.ui.iframe`)
- Client and remote agents negotiate these parts to align with UI capabilities of the requesting environment, ensuring that output is rendered in a format compatible with the user interface—whether it’s a chat window, dashboard, or mobile device.

#### Content Routing and Privacy

- A2A supports fine-grained control over which agents receive what content. This is essential for:
    - Multi-agent task orchestration
    - Privacy-preserving collaboration
    - Compliance with enterprise data protection rules
- Payloads may be routed through secure proxies, or encrypted per agent-to-agent key exchange.

### Real-World Scenario: Candidate Sourcing

- In a hiring workflow, a hiring manager tasks an agent to find suitable engineering candidates. This agent:
    1. Consults the job description and location
    2. Uses A2A to identify and query remote agents specialized in resume parsing, candidate databases, or social graph insights
    3. Aggregates candidate artifacts and presents them via a unified UI
    4. Further agents handle interview scheduling, background checks, or onboarding
- This coordinated agent ecosystem—built atop A2A—enables seamless, scalable execution of complex enterprise workflows, combining capabilities from multiple vendors and systems.

### Implementation Architecture

- A2A-compliant agents are typically composed of the following components:
    
    - **Agent Card Endpoint**
        - `GET /agent/card`
        - Returns metadata about the agent, capabilities, and supported modalities
    - **Task Endpoint**
        - `POST /agent/tasks`
        - Accepts new task requests from a client agent
    - **Task State Endpoint**
        - `GET /agent/tasks/{task_id}`
        - Returns current status of a task (e.g., “in_progress”, “completed”)
    - **Artifact Retrieval Endpoint**
        - `GET /agent/tasks/{task_id}/artifact`
        - Downloads final output(s) for the task
    - **Message Stream Endpoint**
        - `GET /agent/messages/stream`
        - Uses SSE to deliver messages (contextual updates, feedback, etc.)
    - **Authentication**
        - OAuth 2.0 Bearer Tokens or API keys passed in headers
        - Supports mutual TLS or signed requests for sensitive data exchange

### Integration and Future Roadmap

- A2A is being released as open source, with contributions from major cloud vendors, enterprise software providers, and consulting firms.
- A full draft specification is available online, with example implementations using:
    - Python and FastAPI
    - Node.js and Express
    - Langchain-based agents
- Development kits, validators, and sandbox agents are available to accelerate adoption and testing.
- A production-ready version of the protocol is scheduled for release later this year, with backward compatibility and feature expansion focused on:
    - Peer discovery
    - Intent routing
    - Secure agent marketplaces
    - UI auto-generation from message part schemas

## Agentic Retrieval-Augmented Generation (RAG)

- Agent-based Retrieval-Augmented Generation (RAG), or Agentic RAG, represents an advanced approach in AI that enhances the traditional RAG pipeline with intelligent agents. In conventional RAG systems, an AI model queries a knowledge base to retrieve relevant information and generate responses. However, Agentic RAG extends beyond this by employing AI agents capable of orchestrating multi-step retrieval processes, utilizing external tools, and dynamically adapting to the query. This added layer of autonomy enables advanced reasoning, decision-making, and adaptability, allowing the system to handle complex queries and diverse data sources with greater precision and responsiveness.
    
- By integrating AI agents, Agentic RAG transforms traditional RAG, providing a flexible, intelligent solution for nuanced, real-world inquiries. This shift enables organizations to deploy AI systems with a higher degree of accuracy, flexibility, and intelligence, allowing them to tackle intricate tasks and deliver more precise results across a wide range of applications.
    

### How Agentic RAG Works

- In an agentic RAG system, AI agents play key roles in the retrieval process, using specialized tools to retrieve context-sensitive information. Unlike traditional RAG, where retrieval functions are static, agentic RAG allows dynamic selection and operation of tools based on query requirements. Retrieval agents may utilize tools such as:
    
    1. **Vector Search Engines**: Retrieve information from vectorized data in databases.
    2. **Web Search Tools**: Access live web data for up-to-date, contextually relevant information.
    3. **Calculators**: Perform computations for queries that require accurate calculation.
    4. **APIs for Software Programs**: Programmatically retrieve information from applications like email or chat programs to access user-specific data.
- In the context of Agentic RAG, the retrieval process is “agentic,” meaning agents are capable of reasoning and decision-making regarding which sources and tools to use, based on the specific requirements of the query. This flexibility elevates their tool usage beyond simple retrieval, allowing for a more dynamic and adaptive response.
    

### Agentic Decision-Making in Retrieval

- The decision-making process of retrieval agents encompasses several key actions, including:
    
    - **Deciding Whether to Retrieve**: Assessing if additional information is necessary for the query.
    - **Choosing the Appropriate Tool**: Selecting the most suitable tool (e.g., a vector search engine or web search) based on the query.
    - **Query Formulation**: Refining or rephrasing the query to enhance retrieval accuracy.
    - **Evaluating Retrieved Results**: Reviewing the retrieved information to determine sufficiency, and whether further retrieval is needed.

### Agentic RAG Architectures: Single-Agent vs. Multi-Agent Systems

- Agentic RAG can be implemented with a single agent or multiple agents, each offering unique strengths.

#### Single-Agent RAG (Router)

- The simplest implementation of agentic RAG involves a single agent functioning as a “router.” This agent determines the appropriate source or tool for retrieving information based on the query. The single agent toggles between different options, such as a vector database, web search, or an API. This setup provides a versatile retrieval process, enabling access to multiple data sources beyond a single vector search tool.
- As shown in the figure below ([source](https://weaviate.io/blog/what-is-agentic-rag)), the single-Agent RAG system (router) architecture involves a single agent serving as a “router,” dynamically selecting the best tool or source based on the query, enabling efficient information retrieval across multiple data channels.

![Agentic RAG Visual Summary](https://aman.ai/primers/ai/assets/agents/Single_Agent_RAG_System_Router.png)

#### Multi-Agent RAG Systems

- For more complex queries, multi-agent RAG systems provide additional flexibility. These systems feature a “master agent” that coordinates several specialized retrieval agents, such as:
    
    - **Internal Data Retrieval Agent**: Retrieves information from proprietary, internal databases.
    - **Personal Data Retrieval Agent**: Accesses user-specific information, such as emails or chat history.
    - **Public Data Retrieval Agent**: Conducts web searches for up-to-date public information.
- By utilizing multiple agents tailored to specific sources or tasks, multi-agent RAG systems can deliver comprehensive, accurate responses across diverse channels.
    
- As shown in the figure below ([source](https://weaviate.io/blog/what-is-agentic-rag)), the multi-agent RAG system architecture utilizes multiple specialized retrieval agents to access different sources and tools, offering a flexible and comprehensive approach to complex queries.
    

![Agentic RAG Visual Summary](https://aman.ai/primers/ai/assets/agents/Multi_Agent_RAG_System.png)

### Beyond Retrieval: Expanding Agentic RAG’s Capabilities

- Agentic RAG systems can incorporate agents for tasks beyond retrieval, including:
    
    - **Validating Information**: Cross-referencing data across sources to ensure accuracy.
    - **Performing Multi-step Reasoning**: Following logical steps to address complex queries before generating responses.
    - **Updating System Memory**: Tracking and retaining user-specific preferences or past queries, enabling personalized and context-aware responses.
- By expanding its capabilities beyond simple retrieval, Agentic RAG delivers a powerful, context-sensitive AI solution capable of handling intricate, real-world applications.
    

### Agentic RAG vs. Vanilla RAG: Key Differences

- While both vanilla and agentic RAG systems aim to retrieve information and generate responses, agentic RAG introduces several significant enhancements:

|**Feature**|**Vanilla RAG**|**Agentic RAG**|
|---|---|---|
|Access to External Tools|No|Yes – Utilizes external tools like vector search engines, web search tools, calculators, and APIs.|
|Query Pre-processing|No|Yes – Agents dynamically refine, rephrase, and adapt queries for optimized retrieval.|
|Decision-making in Retrieval|Limited to direct retrieval from knowledge base|Agents autonomously decide if retrieval is needed, select tools, and adapt based on query complexity and source type.|
|Multi-step Retrieval Process|No|Yes – Agents perform multi-step, adaptive retrieval processes involving various sources or tool combinations.|
|Data Validation|No|Yes – Information is cross-referenced across sources to validate accuracy, supporting complex, real-world responses.|
|Dynamic Tool Selection|Static retrieval tools only|Dynamic – Agents choose specific tools (e.g., vector search, APIs) based on query needs.|
|Adaptability to Query|Limited|Highly adaptive – Agents select and operate tools based on real-time assessment of query requirements.|
|Types of Agents|Not applicable|Multiple specialized agents, such as internal data retrieval, personal data retrieval, public data retrieval.|
|Single-Agent vs. Multi-Agent System|Not applicable|Single-agent router or multi-agent systems, with “master” and specialized agents for complex queries.|
|Reasoning and Logic Capability|No|Yes – Supports multi-step reasoning, allowing logical sequence handling before generating responses.|
|Memory and Personalization|Limited to immediate query|Yes – Capable of updating memory to retain user preferences or history, allowing personalized responses.|
|Real-world Applications|Primarily static responses from a fixed database|Supports a wide range of real-world applications by responding to complex, nuanced inquiries with context sensitivity.|

- Drawing a parallel with problem-solving, agentic RAG offers capabilities akin to having a smartphone in hand—equipped with multiple apps and tools to help answer a question—whereas vanilla RAG is akin to being in a library with limited resources.

### Implementing Agentic RAG: Key Approaches

- To implement agentic RAG, developers can use either language models with function calling or agent frameworks, each providing specific advantages in terms of flexibility and control.
- Both methods—function calling in language models and agent frameworks—enable agentic RAG, though each has unique benefits:
    - **Function Calling** provides control over each tool interaction, suitable for cases with specific tool chains or simple agent setups.
    - **Agent Frameworks** offer pre-built integrations and routing logic, ideal for larger, multi-agent architectures.
- Using these implementations, developers can build flexible and adaptive agentic RAG pipelines, enhancing retrieval, reasoning, and response generation capabilities for AI-driven applications.

#### Language Models with Function Calling

- Function calling allows language models to interact directly with external tools. For example, OpenAI’s function calling for GPT-4 or Cohere’s connectors API lets developers connect language models to databases, calculators, and other services. This interaction involves defining a function (such as querying a database), passing it to the model via a schema, and routing the model’s queries through the defined functions. This approach enables the model to leverage specific tools as needed, based on the query.

#### Agent Frameworks

- Several agent frameworks—such as LangChain, LlamaIndex, CrewAI—simplify agentic RAG implementation by providing pre-built templates and tool integrations. Key features include:
    - **LangChain**: Offers support for language model tools, and its LCEL and LangGraph frameworks integrate these tools seamlessly.
    - **LlamaIndex**: Provides a QueryEngineTool to streamline retrieval tasks.
    - **CrewAI**: A leading framework for multi-agent setups, which supports shared tool access among agents.

### Enterprise-driven Adoption

- Organizations are increasingly transitioning to agentic RAG to gain more autonomous and accurate AI-driven systems. Enterprises such as Microsoft and Replit have introduced agents to enhance task completion and software development assistance. With agentic RAG, companies can build AI applications capable of handling diverse, real-time data sources, providing robust and adaptable responses for complex queries and tasks.

### Benefits

- The primary benefits of agentic RAG include:
    - **Enhanced Retrieval Accuracy**: By routing queries through specialized agents, agentic RAG can provide more accurate responses.
    - **Autonomous Task Performance**: Agents can perform multi-step reasoning, independently solving complex problems.
    - **Improved Collaboration**: These systems can better assist users by handling more varied and personalized queries.

### Limitations

- Agentic RAG does present challenges, such as:
    - **Increased Latency**: Running multiple agents and interacting with tools can add delays to the response.
    - **Reliability of Agents**: Depending on the LLM’s reasoning capabilities, agents may fail to complete certain tasks accurately.
    - **Complexity in Error Handling**: Systems need robust fallback mechanisms to recover if an agent fails to retrieve or process data.

### Code

- Implementing agentic RAG requires setting up an agent framework capable of handling tool integrations and coordinating retrieval processes. This section walks through an example code setup, demonstrating both language models with function calling and agent frameworks for building an agentic RAG pipeline.

#### Implementing Agentic RAG with Function Calling

- Function calling in language models allows them to interact with tools by defining functions that retrieve data from external sources. This method leverages API calls, database queries, and computation tools to enrich the response with dynamic data.
    
- Here’s an example implementation using a function for retrieval from a database via the Weaviate vector search API.
    

##### Define the Function for Retrieval

- To start, we define a function that uses Weaviate’s hybrid search to query a database and retrieve relevant results.

![](https://aman.ai/images/copy.png)

`def get_search_results(query: str) -> str:     """Sends a query to Weaviate's Hybrid Search. Parses the response into a formatted string."""      response = blogs.query.hybrid(query, limit=5)  # Retrieve top 5 results based on the query     stringified_response = ""     for idx, o in enumerate(response.objects):         stringified_response += f"Search Result {idx+1}:\n"         for prop in o.properties:             stringified_response += f"{prop}: {o.properties[prop]}\n"         stringified_response += "\n"      return stringified_response`

##### Define the Tools Schema

- Next, we define a tools schema that connects the function to the language model. This schema tells the model how to use the function for retrieving data.

![](https://aman.ai/images/copy.png)

`tools_schema = [{     'type': 'function',     'function': {         'name': 'get_search_results',         'description': 'Get search results for a provided query.',         'parameters': {           'type': 'object',           'properties': {             'query': {               'type': 'string',               'description': 'The search query.',             },           },           'required': ['query'],         },     }, }]`

##### Setting up the Interaction Loop

- To ensure the model can call the tool multiple times (if needed), we set up a loop that enables the model to interact with tools and retrieve data iteratively until it has all necessary information.

![](https://aman.ai/images/copy.png)

`def ollama_generation_with_tools(user_message: str, tools_schema: list, tool_mapping: dict, model_name: str = "llama3.1") -> str:     messages = [{"role": "user", "content": user_message}]     response = ollama.chat(model=model_name, messages=messages, tools=tools_schema)          # Check if the model needs to use a tool     if not response["message"].get("tool_calls"):         return response["message"]["content"]          # Handle tool calls and retrieve information     for tool in response["message"]["tool_calls"]:         function_to_call = tool_mapping[tool["function"]["name"]]         function_response = function_to_call(tool["function"]["arguments"]["query"])         messages.append({"role": "tool", "content": function_response})          # Generate final response after tool calls     final_response = ollama.chat(model=model_name, messages=messages)     return final_response["message"]["content"]`

##### Executing the Agentic RAG Query

- Finally, we run the function, allowing the language model to interact with the `get_search_results` tool.

![](https://aman.ai/images/copy.png)

`tool_mapping = {"get_search_results": get_search_results}  # Maps tool name to function response = ollama_generation_with_tools(     "How is HNSW different from DiskANN?",     tools_schema=tools_schema,     tool_mapping=tool_mapping ) print(response)`

- This setup enables the language model to retrieve dynamic information and perform tool-based retrievals as needed.

#### Implementing Agentic RAG with Agent Frameworks

- Using agent frameworks streamlines the implementation process by providing templates and pre-built modules for multi-agent orchestration. Here’s how to set up an agentic RAG pipeline using LangChain as an example.

##### Step 1: Define Agents and Tools

- LangChain simplifies agentic RAG by managing tools and routing tasks. First, define the agents and register the tools they will use.

![](https://aman.ai/images/copy.png)

`from langchain.tools import WebSearchTool, DatabaseTool, CalculatorTool from langchain.agents import Agent  # Define tools for retrieval web_search_tool = WebSearchTool(api_key="YOUR_WEB_SEARCH_API_KEY") database_tool = DatabaseTool(db_client="your_database_client") calculator_tool = CalculatorTool()  # Set up an agent with a routing function retrieval_agent = Agent(     tools=[web_search_tool, database_tool, calculator_tool],     routing_function="retrieve_and_select_tool" )`

##### Step 2: Configure Agent Routing

- Set up the routing function to let the agent decide which tool to use based on the input query.

![](https://aman.ai/images/copy.png)

`def retrieve_and_select_tool(query):     if "calculate" in query:         return calculator_tool     elif "web" in query:         return web_search_tool     else:         return database_tool`

##### Step 3: Chain Agents for Multi-Agent RAG

- In multi-agent RAG, you might have a “master agent” that routes queries to specialized agents based on query type. Here’s how to set up a master agent to coordinate multiple agents.

![](https://aman.ai/images/copy.png)

`from langchain.agents import MultiAgent  # Define specialized agents internal_agent = Agent(tools=[database_tool], routing_function="database_retrieval") public_agent = Agent(tools=[web_search_tool], routing_function="web_retrieval")  # Create a master agent to coordinate retrieval master_agent = MultiAgent(agents=[internal_agent, public_agent])  # Function to handle a query using master agent def handle_query_with_master_agent(query):     return master_agent.handle_query(query)`

##### Running the Multi-Agent Query

- Finally, to test the system, input a query and let the master agent route it appropriately:

![](https://aman.ai/images/copy.png)

`response = handle_query_with_master_agent("Find recent studies on neural networks") print(response)`

### Disadvantages of Agentic RAG

- Despite its advantages, agentic RAG comes with several limitations that should be carefully considered, particularly for time-sensitive applications:
    
    1. **Increased Latency**: The inherent complexity of agentic RAG often translates to longer response times. Each query may require multiple tool interactions and sequential retrieval steps, which increase the latency significantly. This can hinder the system’s usability in environments where quick responses are crucial, such as real-time support systems or conversational interfaces.
        
    2. **Higher Computational Cost**: Agentic RAG systems often involve multiple calls to LLMs and other external tools. These calls cumulatively drive up computational costs, making it less efficient and potentially prohibitive for high-traffic applications. This expense adds to operational concerns, especially if the system must process large volumes of queries.
        
    3. **Production Feasibility**: Due to the latency and cost concerns, agentic RAG may not be ideal for production applications requiring rapid and continuous output. In such cases, vanilla RAG, which offers more direct and faster response generation, might be more suitable.
        
- While these drawbacks limit agentic RAG’s use in certain scenarios, its capability to generate high-quality, well-researched responses can make it worthwhile in contexts where response time is less critical and information accuracy is paramount.
    

### Summary

- Agentic RAG refers to an agent-based implementation of RAG. AI agents are entities tasked with accomplishing specific objectives. These agents are often equipped with memory and tools, which they can utilize to carry out their tasks effectively. Among these tools, one significant capability is the ability to retrieve information from various sources, such as web searches or internal documents.
- In the context of agentic RAG, the “retrieval becomes agentic.” This implies that the AI agent is capable of reasoning and making decisions regarding which sources are most appropriate for retrieving the required information. The agent’s tool usage evolves beyond simple information retrieval, becoming more flexible and dynamic.
- The distinction between standard and agentic RAG can be summarized as follows:
    - **Common RAG**: The user input prompts a single call to a database, retrieving additional information in response to the query.
    - **Agentic RAG**: The agent is able to deliberate on which source is the most suitable for retrieving information based on the query, providing a more sophisticated and adaptable approach.
- The following figure ([source](https://www.linkedin.com/in/804250ab/)) offers a visual summary of Agentic RAG:

![](https://aman.ai/primers/ai/assets/RAG/AgenticRAG.jpg)

## Benchmarks

- Evaluating LLM-based agents requires specialized benchmarks that go beyond static NLP tasks and assess the agent’s ability to reason, plan, execute actions, and adapt to dynamic environments.
- These benchmarks measure an agent’s capability in real-world-like simulations, web interaction, tool use, and multi-step decision-making, which are critical for developing autonomous AI systems. Benchmarks designed to evaluate LLM agents include:
    - [OSWorld](https://github.com/openbmb/OSWorld): OSWorld evaluates LLM-based agents in a simulated open-ended world where they interact with diverse environments. It tests reasoning, planning, and multi-step task execution, emphasizing adaptability to dynamic conditions.
    - [WebArena](https://webarena.dev/): WebArena assesses LLM agents’ ability to navigate and interact with realistic web environments, covering tasks like browsing, searching, and form-filling. It provides a high-fidelity, human-like internet experience for testing generalization and real-world usability.
    - [WebVoyager](https://webvoyager.dev/): WebVoyager focuses on evaluating LLM agents in autonomous web exploration. It measures how well agents adapt to unseen websites, perform goal-oriented browsing, and extract relevant information with minimal human intervention.
    - [SWE-Bench](https://github.com/princeton-nlp/SWE-bench): SWE-Bench is tailored for evaluating LLM-based software engineering agents. It includes real-world GitHub issues and codebases, testing agents on debugging, pull request generation, and code improvement tasks with automated correctness verification.
    - [GAIA](https://github.com/GAIA-Benchmark/GAIA): General AI Agent Benchmark (GAIA) evaluates the robustness and general intelligence of AI agents across multiple domains, including gaming, web interaction, and decision-making. It emphasizes long-horizon planning and adaptability.
    - [AgentBench](https://github.com/OpenBMB/AgentBench): AgentBench is a large-scale benchmark for multi-agent collaboration and LLM-based decision-making. It covers a wide range of tasks, from coordination and negotiation to strategic planning and problem-solving.
    - [IGLU](https://github.com/iglu-contest/iglu): Interactive Grounded Language Understanding (IGLU) tests LLM agents in interactive 3D environments where they must understand natural language commands and manipulate objects accordingly. It measures spatial reasoning and task execution in multimodal settings.
    - [ClemBench](https://github.com/ClemBotProject/ClemBench): ClemBench benchmarks the performance of LLM agents in structured dialogue and task automation. It includes real-world chatbot use cases, emphasizing context retention, instruction-following, and multi-turn reasoning.
    - [ToolBench](https://github.com/OpenGVLab/ToolBench): ToolBench evaluates LLM agents’ ability to use external tools and APIs to complete tasks. It assesses tool integration, multi-step reasoning, and adaptability in using calculators, databases, and search engines.
    - [GentBench](https://github.com/GentBench/GentBench): GentBench focuses on generative reasoning and task execution, testing LLM agents on creative and logical problem-solving. It includes challenges in writing, planning, and structured data manipulation.
    - [MLAgentBench](https://github.com/ml-agent-bench/MLAgentBench): MLAgentBench is designed to evaluate reinforcement learning agents powered by LLMs. It provides a set of standard environments to measure agent learning, adaptability, and decision-making efficiency across different problem spaces.
- Evaluation dimensions include:
    - **Utility:** Task completion effectiveness and efficiency, measured by success rate and task outcomes.
    - **Sociability:** Language communication proficiency, cooperation, negotiation abilities, and role-playing capability.
    - **Values:** Adherence to moral and ethical guidelines, honesty, harmlessness, and contextual appropriateness.
    - **Ability to Evolve Continually:** Continual learning, autotelic learning ability, and adaptability to new environments.
    - **Adversarial Robustness:** Susceptibility to adversarial attacks, with techniques like adversarial training and human-in-the-loop supervision employed.
    - **Trustworthiness:** Calibration problems and biases in training data affect trustworthiness. Efforts are made to guide models to exhibit thought processes or explanations to enhance credibility.

## Common Use-cases

- Agentic systems can be employed in various domains to enable AI systems to operate autonomously, making decisions, executing tasks, and adapting to changing circumstances without constant human intervention.
    
    - **Customer Support:** AI-driven virtual assistants such as Conversational AI Agents can manage customer inquiries, troubleshoot common issues, and escalate complex cases to human representatives when necessary. Examples include AI chatbots for e-commerce support (e.g., Amazon’s Alexa for customer service) and automated call center agents handling telecom support.
        
    - **Content Creation:** AI Content Generators like GPT-based writing agents can craft blog posts, marketing copy, and social media updates. They can optimize content for SEO, personalize messaging for different audiences, and even generate video scripts. Examples include AI-driven copywriting assistants like Jasper or OpenAI’s ChatGPT for content strategy.
        
    - **Education:** AI Tutoring Agents can personalize learning experiences, provide step-by-step explanations, and assess student progress. These include virtual math tutors like Photomath, AI-powered language learning assistants like Duolingo’s chatbots, and generative AI for personalized essay feedback.
        
    - **Coding Assistance:** Autonomous Coding Agents can provide real-time code suggestions, detect and fix bugs, and even generate complete functions based on prompts. These include tools like GitHub Copilot, Tabnine, and OpenAI’s Codex, which help developers streamline their coding workflows.
        
    - **Healthcare:** AI Medical Assistants can analyze medical literature, provide diagnostic insights based on patient symptoms, and support mental health through AI therapy chatbots. Examples include IBM Watson Health for medical research, AI symptom checkers like Ada Health, and AI therapy agents like Woebot.
        
    - **Accessibility:** AI Accessibility Agents can convert text into speech, describe images for visually impaired users, and generate captions in real-time for hearing-impaired individuals. Notable examples include screen readers like NVDA, Microsoft’s Seeing AI, and AI-powered live captioning tools like Google’s Live Transcribe.
        

## Case Studies

- Two particularly promising applications for AI agents that demonstrate the practical value include customer support and software development (i.e., coding agents). Both applications illustrate how agents add the most value for tasks that require both conversation and action, have clear success criteria, enable feedback loops, and integrate meaningful human oversight. Below, we cover them in detail.

### Customer Support

- Customer support combines familiar chatbot interfaces with enhanced capabilities through tool integration. This is a natural fit for more open-ended agents because:
    
    - Support interactions naturally follow a conversation flow while requiring access to external information and actions;
    - Tools can be integrated to pull customer data, order history, and knowledge base articles;
    - Actions such as issuing refunds or updating tickets can be handled programmatically; and
    - Success can be clearly measured through user-defined resolutions.
- Several companies have demonstrated the viability of this approach through usage-based pricing models that charge only for successful resolutions, showing confidence in their agents’ effectiveness.
    

### Software Development

- The software development space has shown remarkable potential for LLM features, with capabilities evolving from code completion to autonomous problem-solving. Agents are particularly effective because:
    
    - Code solutions are verifiable through automated tests;
    - Agents can iterate on solutions using test results as feedback;
    - The problem space is well-defined and structured; and
    - Output quality can be measured objectively.
- Agentic workflows demonstrate significant performance improvements in programming tasks. The figure below ([source](https://www.deeplearning.ai/the-batch/how-agents-can-improve-llm-performance/)) offers an analysis of various AI models’ coding capabilities on the HumanEval benchmark revealed stark differences when using iterative agentic approaches. Zero-shot attempts by GPT-3.5 and GPT-4 achieved correctness rates of 48.1% and 67.0%, respectively. However, when GPT-3.5 was embedded within an agent loop, it achieved up to 95.1% accuracy, highlighting the profound impact of iterative processing.
    

![diagram illustrating HumanEval benchmark results](https://aman.ai/primers/ai/assets/agents/HumanEval.jpg)

- In summary, the integration of agentic and iterative workflows marks a significant evolution in AI design patterns, enhancing the ability of LLM-powered agents to handle increasingly sophisticated tasks. As these frameworks and tools continue to mature, they are poised to become essential components in AI-driven task management and automation, offering unparalleled efficiency and adaptability.
    
- In our own implementation, agents can now solve real GitHub issues in the SWE-bench Verified benchmark based on the pull request description alone. However, whereas automated testing helps verify functionality, human review remains crucial for ensuring solutions align with broader system requirements.
    

#### Devin

- A notable example of an agentic approach applied to software development is [Devin](https://www.cognition.ai/), an autonomous agent optimized for software engineering tasks. Devin exemplifies the effectiveness of agentic workflows, achieving state-of-the-art results on the [SWE-Bench coding benchmark](https://www.swebench.com/), passing rigorous practical engineering interviews at top AI firms, and successfully executing real-world assignments on platforms like Upwork. Devin operates by leveraging its own shell, code editor, and web browser, enabling it to autonomously solve engineering problems and refine its work iteratively.

## Frameworks/Libraries

- The below frameworks facilitate the initial development process of agents by streamlining fundamental low-level tasks such as invoking LLMs, defining and interpreting tools, and managing sequential function calls. However, they often introduce additional layers of abstraction that can obscure the underlying prompts and responses, making debugging more challenging. Furthermore, they may encourage unnecessary complexity when a simpler approach would be sufficient. Beginning by working directly with LLM APIs (rather than frameworks) is ideal, as many patterns can be implemented with minimal code.
- Anthropic’s [cookbook](https://github.com/anthropics/anthropic-cookbook/tree/main/patterns/agents) offers some sample implementations.

### [AutoGen Studio](https://microsoft.github.io/autogen/docs/autogen-studio/getting-started/)

- Microsoft Research’s [AutoGen Studio](https://microsoft.github.io/autogen/docs/autogen-studio/getting-started/) is a low-code interface for rapidly prototyping AI agents. It’s built on top of the AutoGen framework and can also be used for debugging and evaluating multi-agent workflows.

![](https://aman.ai/primers/ai/assets/agents/ags.png)

### [AutoGen](https://github.com/microsoft/autogen)

- AutoGen is an open-source framework by Microsoft for building AI agent systems. It simplifies the creation of event-driven, distributed, scalable, and resilient agentic applications.
- [Github](https://github.com/microsoft/autogen); [Docs](https://microsoft.github.io/autogen/0.2/docs/Getting-Started); [Examples](https://microsoft.github.io/autogen/0.2/docs/notebooks)

### [Swarm](https://github.com/openai/swarm)

- Swarm by OpenAI’s is a framework exploring ergonomic, lightweight multi-agent orchestration.
- [Github](https://github.com/openai/swarm)

### [CrewAI](https://crewai.com/)

- Cutting-edge framework for orchestrating role-playing, autonomous AI agents. By fostering collaborative intelligence, CrewAI empowers agents to work together seamlessly, tackling complex tasks.
- [Github](https://github.com/crewAIInc/crewAI); [Docs](https://docs.crewai.com/introduction); [Examples](https://docs.crewai.com/examples/)

### [Letta](https://www.letta.com/)

- [Letta](https://www.letta.com/) is an open source framework for building stateful LLM applications. You can use Letta to build stateful agents with advanced reasoning capabilities and transparent long-term memory.
- [Github](https://github.com/cpacker/MemGPT); [Docs](https://docs.letta.com/introduction); [Examples](https://github.com/cpacker/MemGPT/tree/main/examples)

### [Llama Stack](https://github.com/meta-llama/llama-stack)

- [Llama Stack](https://github.com/meta-llama/llama-stack) from Meta defines and standardizes the building blocks needed to bring generative AI applications to market. These blocks span the entire development lifecycle: from model training and fine-tuning, through product evaluation, to building and running AI agents in production.
- [Github](https://github.com/meta-llama/llama-stack); [Docs](https://github.com/meta-llama/llama-stack/tree/main/docs); [Examples](https://github.com/meta-llama/llama-stack-apps/tree/main/examples)

### [AutoRAG](https://auto-rag.com/)

- [AutoRAG](https://auto-rag.com/) is a tool for finding the optimal RAG pipeline for “your data”, including Agentic RAG approaches. You can evaluate various RAG modules automatically with your own evaluation data and find the best RAG pipeline for your own use-case.
- [Github](https://github.com/Marker-Inc-Korea/AutoRAG); [Docs](https://docs.auto-rag.com/); [Examples](https://github.com/Marker-Inc-Korea/AutoRAG-tutorial)

### [Beam](https://beam.ai/)

- [Beam](https://beam.ai/) is the leading platform for Agentic Process Automation.

### [AutoAgents](https://github.com/Link-AGI/AutoAgents)

- [AutoAgents](https://github.com/Link-AGI/AutoAgents) is a novel framework designed for dynamic multi-agent generation and coordination, enabling language models to construct adaptive AI teams for a wide range of tasks. Unlike traditional systems that rely on static, predefined agents, AutoAgents generates task-specific agents autonomously, allowing for flexible collaboration across varied domains. The framework introduces a drafting and execution stage to handle complex task environments and facilitate effective role assignment and solution planning.

### [Amazon Bedrock’s AI Agent Framework](https://aws.amazon.com/bedrock/agents/)

- [Amazon Bedrock Agents](https://aws.amazon.com/bedrock/agents/) automate complex tasks by integrating with company systems, using foundation models for reasoning, memory retention, and secure execution. They support multi-agent collaboration, retrieval-augmented generation, code execution, and customizable prompt engineering for efficient workflows.

### [Rivet](https://rivet.ironcladapp.com/)

- [Rivet](https://rivet.ironcladapp.com/) is a drag and drop GUI LLM workflow builder.

### [Vellum](https://www.vellum.ai/)

- [Vellum](https://www.vellum.ai/) is a GUI tool for building and testing complex workflows.

### [BabyAGI](https://github.com/yoheinakajima/babyagi)

- BabyAGI is a widely used, task-driven autonomous agent built to tackle a variety of tasks across multiple domains. It leverages advanced technologies, including OpenAI’s GPT-4 language model, the Pinecone vector search platform, and the LangChain framework. A breakdown of its core components is [here](https://yoheinakajima.com/task-driven-autonomous-agent-utilizing-gpt-4-pinecone-and-langchain-for-diverse-applications/) and code is [here](https://github.com/yoheinakajima/babyagi).
- BabyAGI’s workflow involves the following operational steps:
    
    1. **Task Completion:** The system processes the task at the forefront of the task list, utilizing GPT-4 in conjunction with LangChain’s chain and agent functionalities to produce a result. This result is subsequently refined, if necessary, and stored in Pinecone for future reference.
    2. **Task Generation:** Following the completion of a task, the system leverages GPT-4 to create new tasks, ensuring that these do not duplicate any existing tasks.
    3. **Task Prioritization:** The system reprioritizes the task list by evaluating the newly generated tasks and their relative importance, with GPT-4 facilitating the prioritization process.
- The following figure ([source](https://yoheinakajima.com/task-driven-autonomous-agent-utilizing-gpt-4-pinecone-and-langchain-for-diverse-applications/)) illustrates the workflow of an AI agent system comprising multiple GPT-4-based agents. The user initiates the process by providing an objective and task, which is added to the Task Queue. The Execution Agent completes the task and sends the result back, which is stored in Memory for future context. The Task Creation Agent can create additional tasks based on the task results, and the Task Prioritization Agent organizes the Task Queue by prioritizing tasks to ensure an optimized flow. Memory is accessed throughout to maintain contextual relevance.

![](https://aman.ai/primers/ai/assets/agents/babyAGI.png)

### [smolagents](https://huggingface.co/blog/smolagents)

- [smolagents](https://huggingface.co/blog/smolagents) by Hugging Face is a simple library to build powerful agents. The main logic for agents fits in ~1000 lines of code. So it’s really dead simple.
- The main agent class is `CodeAgent`, an agent that writes its actions in code. That means, contrary to the standard set by OpenAI of writing JSON blobs that call tools, this agent writes code blobs. It’s much more natural for LLMs to write actions this way, and as a result performance is much better.
- [Code](https://github.com/huggingface/smolagents)

### [Agent S2](https://www.simular.ai/agent-s2)

- Agent S2 is an [open source](https://github.com/simular-ai/agent-s), autonomous AI framework that offers improved performance, modularity, and scalability by combining frontier and specialized models.
- [Code](https://github.com/simular-ai/agent-s)

### [Open Operator](https://github.com/browserbase/open-operator)

- [Open Operator](https://github.com/browserbase/open-operator) is inspired by OpenAI’s Operator feature and builds upon various open source technologies such as [Browserbase](https://www.browserbase.com/) powers the core browser automation and interaction capabilities and [Stagehand](https://www.stagehand.dev/) which handles precise DOM manipulation and state management

## Example Flow Chart for an LLM Agent: Handling a Customer Inquiry

![](https://aman.ai/primers/ai/assets/agents/3.png)

- The image above [(source)](https://towardsdatascience.com/navigating-the-world-of-llm-agents-a-beginners-guide-3b8d499db7a9) shows an example of AI Agent flow.
    
    1. **Customer Interaction**
        - Input: “Is the new XYZ smartphone available, and what are its features?”
        - Action: Customer types the query into the e-commerce platform’s chat interface.
    2. **Query Reception and Parsing**
        - **Agent Core Reception:** Receive text input.
        - **Natural Language Understanding:** Parse the text to extract intent and relevant entities.
    3. **Intent Classification and Information Retrieval**
        - **Intent Classification:** Classify the query intent.
        - **Memory Access:** Retrieve stored data on product inventory and specifications.
        - **External API Calls:** Fetch additional data if not available in memory.
    4. **Data Processing and Response Planning**
        - **Planning Module:** Split the query into “check availability” and “retrieve features”.
        - **Data Synthesis:** Combine information from memory.

## Use Cases

- Let’s look at a few agent use cases below:

### Data Agent for Data Analysis

![](https://aman.ai/primers/ai/assets/agents/4.png)

- The image above [(source)](https://developer.nvidia.com/blog/build-an-llm-powered-data-agent-for-data-analysis/) visually illustrates the flow we will use below.
    
    1. **Identify the Use Case:**
        - Define specific data analysis tasks, such as querying databases or analyzing financial reports.
    2. **Select the Appropriate LLM:**
        - Choose an LLM that handles the complexity of data queries and analysis.
    3. **Agent Components:**
        - Develop the agent with tools for data handling, a memory module for tracking interactions, and a planning module for strategic execution of tasks.
    4. **Design the Data Interaction Tools:**
        - Implement tools for interacting with databases or other data sources.

#### Tools Setup

![](https://aman.ai/images/copy.png)

`class SQLExecutor:     def __init__(self, database_url):         self.database_url = database_url      def execute_query(self, query):         print(f"Executing SQL query: {query}")         return "Query results"  class Calculator:     @staticmethod     def perform_calculation(data):         print(f"Performing calculation on data: {data}")         return "Calculation results"`

#### Agent Core Logic

![](https://aman.ai/images/copy.png)

`class DataAgent:     def __init__(self, sql_executor, calculator):         self.sql_executor = sql_executor         self.calculator = calculator         self.memory = []      def analyze_data(self, query, calculation_needed=True):         results = self.sql_executor.execute_query(query)         self.memory.append(results)          if calculation_needed:             calculation_results = self.calculator.perform_calculation(results)             self.memory.append(calculation_results)             return calculation_results                  return results  database_url = "your_database_url_here" sql_executor = SQLExecutor(database_url) calculator = Calculator()  agent = DataAgent(sql_executor, calculator) query = "SELECT * FROM sales_data WHERE year = 2021" print(agent.analyze_data(query))`

### LLM-Powered API Agent for Task Execution

1. **Choose an LLM:**
    - Select a suitable LLM for handling task execution.
2. **Select a Use Case:**
    - Define the tasks the agent will execute.
3. **Build the Agent:**
    - Develop the components required for the API agent: tools, planning module, and agent core.
4. **Define API Functions:**
    - Create classes for each API call to the models.

#### Python Code Example

![](https://aman.ai/images/copy.png)

`class ImageGenerator:     def __init__(self, api_key):         self.api_key = api_key      def generate_image(self, description, negative_prompt=""):         print(f"Generating image with description: {description}")         return "Image URL or data"  class TextGenerator:     def __init__(self, api_key):         self.api_key = api_key      def generate_text(self, text_prompt):         print(f"Generating text with prompt: {text_prompt}")         return "Generated text"  class CodeGenerator:     def __init__(self, api_key):         self.api_key = api_key      def generate_code(self, problem_description):         print(f"Generating code for: {problem_description}")         return "Generated code"`

#### Plan-and-Execute Approach

![](https://aman.ai/images/copy.png)

`def plan_and_execute(question):     if 'marketing' in question:         plan = [             {                 "function": "ImageGenerator",                 "arguments": {                     "description": "A bright and clean laundry room with a large bottle of WishyWash detergent, featuring the new UltraClean formula and softener, placed prominently.",                     "negative_prompt": "No clutter, no other brands, only WishyWash."                 }             },             {                 "function": "TextGenerator",                 "arguments": {                     "text_prompt": "Compose a tweet to promote the new WishyWash detergent with the UltraClean formula and softener at $4.99. Highlight its benefits and competitive pricing."                 }             },             {                 "function": "TextGenerator",                 "arguments": {                     "text_prompt": "Generate ideas for marketing campaigns to increase WishyWash detergent sales, focusing on the new UltraClean formula and softener."                 }             }         ]         return plan     else:         pass  def execute_plan(plan):     results = []     for step in plan:         if step["function"] == "ImageGenerator":             generator = ImageGenerator(api_key="your_api_key")             result = generator.generate_image(**step["arguments"])             results.append(result)         elif step["function"] == "TextGenerator":             generator = TextGenerator(api_key="your_api_key")             result = generator.generate_text(**step["arguments"])             results.append(result)         elif step["function"] == "CodeGenerator":             generator = CodeGenerator(api_key="your_api_key")             result = generator.generate_code(**step["arguments"])             results.append(result)     return results  question = "How can we create a marketing campaign for our new detergent?" plan = plan_and_execute(question) results = execute_plan(plan) for result in results:     print(result)`

## Build Your Own LLM Agent

- Here’s a detailed explanation including some Python code examples as outlined in the [NVIDIA blog](https://developer.nvidia.com/blog/building-your-first-llm-agent-application/) for building a question-answering LLM agent:

1. **Set Up the Agent’s Components**:
    - **Tools**: Include tools like a Retrieval-Augmented Generation (RAG) pipeline and mathematical tools necessary for data analysis.
    - **Planning Module**: A module to decompose complex questions into simpler parts for easier processing.
    - **Memory Module**: A system to track and remember previous interactions and solutions.
    - **Agent Core**: The central processing unit of the agent that uses the other components to solve user queries.
2. **Python Code Example for the Memory Module**:
    
    ![](https://aman.ai/images/copy.png)
    
    `class Ledger:     def __init__(self):         self.question_trace = []         self.answer_trace = []      def add_question(self, question):         self.question_trace.append(question)      def add_answer(self, answer):         self.answer_trace.append(answer)`
    
3. **Python Code Example for the Agent Core**:
    - This part of the code defines how the agent processes questions, interacts with the planning module, and retrieves or computes answers.
        
        ![](https://aman.ai/images/copy.png)
        
        `def agent_core(question, context):   # Assume a function LLM is defined to handle LLM processing   action = LLM(context + question)    if action == "Decomposition":       sub_questions = LLM(question)       for sub_question in sub_questions:           agent_core(sub_question, context)   elif action == "Search Tool":       answer = RAG_Pipeline(question)       context += answer       agent_core(question, context)   elif action == "Generate Final Answer":       return LLM(context)   elif action == "<Another Tool>":       # Execute another specific tool       pass`
        
4. **Execution Flow**:
    - The agent receives a question, and based on the context and internal logic, decides if it needs to decompose the question, search for information, or directly generate an answer.
    - The agent can recursively handle sub-questions until a final answer is generated.
5. **Using the Components Together**:
    - All the components are used in tandem to manage the flow of data and information processing within the agent. The memory module keeps track of all queries and responses, which aids in contextual understanding for the agent.
6. **Deploying and Testing the Agent**:
    - Once all components are integrated, the agent is tested with sample queries to ensure it functions correctly and efficiently handles real-world questions.

## Related Papers

### Reflection

#### [Self-Refine: Iterative Refinement with Self-Feedback](https://arxiv.org/abs/2303.17651)

- Like humans, large language models (LLMs) do not always generate the best output on their first try.
- This paper by Madaan et al. from CMU, Allen AI, UW, NVIDIA, UC San Diego, and Google Research introduces a novel approach for enhancing outputs from large language models (LLMs) like GPT-3.5 and GPT-4 through self-generated iterative feedback and refinement, without the need for additional training data or supervised learning – similar to how humans refine their written text.
- Put simply, the main idea is to generate an initial output using an LLM; then, the same LLM provides feedback for its output and uses it to refine itself, iteratively. This process repeats until a predefined condition is met. Self-Refine does not require any supervised training data, additional training, or reinforcement learning, and instead uses a single LLM as the generator, refiner and the feedback provider.
- The figure below from the paper shows that given an input (step 0), Self-Refine starts by generating an output and passing it back to the same model M to get feedback (step 1). The feedback is passed back to M, which refines the previously generated output (step 2). Steps (step 1) and (step 2) iterate until a stopping condition is met. SELF-REFINE is instantiated with a language model such as GPT-3.5 and does not involve human assistance.

![](https://aman.ai/images/papers/Self-Refine.jpg)

- The approach is evaluated across seven diverse tasks, including dialogue response and code optimization, demonstrating significant improvements over conventional one-step generation methods. This method leverages few-shot prompting for guiding the LLM to generate feedback and incorporate it for output refinement.
- The results show that Self-Refine significantly enhances output quality in terms of human preference and task-specific metrics, indicating its potential to improve LLM-generated content across a range of applications.
- [Code](https://selfrefine.info/)

#### [Reflexion: Language Agents with Verbal Reinforcement Learning](https://arxiv.org/abs/2303.11366)

- This paper by Shinn et al. from Northeastern University, MIT, and Princeton University introduces Reflexion, a reinforcement learning framework for large language model (LLM)-based agents, enabling them to improve task performance using self-reflective verbal feedback instead of traditional weight updates. Reflexion processes external feedback, transforming it into actionable self-reflections stored in episodic memory, enhancing agents’ decision-making over successive trials in tasks such as sequential action selection, reasoning, and code generation.
- **Framework Overview**: Reflexion involves three models: an Actor (task action generation), an Evaluator (performance assessment), and a Self-Reflection model (produces verbal guidance for improvement). The Actor, built on LLMs, generates actions based on both state observations and past experiences. The Evaluator assigns task-specific rewards, and the Self-Reflection model formulates verbal feedback based on task failures, guiding future attempts.
- **Memory Structure**: Reflexion employs short-term memory (trajectory history) and long-term memory (aggregated self-reflections), which the Actor consults during action generation. This structure allows the agent to remember specific past mistakes while retaining broader learnings across episodes, which aids in complex decision-making tasks.
- The following figure from the paper shows that Reflexion works on decision-making, programming, and reasoning tasks.

![](https://aman.ai/images/papers/Reflexion.jpg)

- **Experimentation and Results**:
    - **Decision-Making in AlfWorld**: Reflexion significantly improved performance on multi-step tasks in AlfWorld by 22%, using heuristics to detect repetitive failures and adapt action choices based on memory. Reflexion enables effective backtracking and context recall, with a notable reduction in inefficient planning errors compared to baseline.
    - **Reasoning with HotPotQA**: Reflexion enhanced reasoning on HotPotQA, achieving a 20% improvement by refining answers through Chain-of-Thought and episodic memory. Reflexion agents retained task-specific strategies across trials, outpacing baseline approaches in correctly navigating long contextual questions.
    - **Programming in HumanEval and LeetcodeHardGym**: Reflexion set new state-of-the-art scores, achieving 91% on HumanEval, aided by self-generated test suites and continuous error-checking through self-reflection. This iterative testing allowed Reflexion agents to refine code output by addressing both syntactical and logical errors.
- **Implementation Details**:
    - Reflexion agents use Chain-of-Thought and ReAct generation techniques, with self-reflective prompts implemented through few-shot examples tailored for each task type. For programming, Reflexion employs syntactically validated test suites, filtered to retain only valid abstract syntax tree representations, ensuring comprehensive error handling in code generation.
    - Self-reflection feedback is stored in memory limited to the last three experiences to maintain efficiency within LLM context limits. Reflexion’s feedback loop iterates until the Evaluator confirms task success, effectively combining reinforcement with natural language memory for performance gains.
- **Ablation Studies and Analysis**: Tests on compromised versions of Reflexion, such as without test generation or self-reflection, showed marked performance drops, underscoring the importance of verbal self-reflection in driving task success. This highlights Reflexion’s effectiveness in environments requiring high interpretability and actionable feedback.
- Reflexion exemplifies a low-compute yet adaptive approach to reinforcement learning for LLM agents, showing potential for expansive applications in autonomous decision-making where interpretable and incremental learning are essential.

#### [CRITIC: Large Language Models Can Self-Correct with Tool-Interactive Critiquing](https://arxiv.org/abs/2305.11738)

- The paper by Gou et al. from Tsinghua University, Microsoft Research Asia, and Microsoft Azure AI, published in ICLR 2024, introduces a novel framework called CRITIC, designed to enhance the reliability of LLM outputs by allowing the models to interact with external tools to critique and amend their own responses.
- CRITIC’s approach diverges from traditional model fine-tuning, focusing instead on a verify-and-correct process where LLMs generate initial outputs and subsequently engage with tools such as search engines, code interpreters, and calculators to verify aspects of these outputs, like truthfulness or accuracy. Based on feedback from these tools, the model then refines its response, iteratively improving until a specified condition (e.g., sufficient accuracy) is achieved. This method sidesteps the need for extensive additional training or data annotation and is structured to work with black-box LLMs through in-context learning and few-shot demonstrations.
- In terms of **implementation**, CRITIC operates in two main phases: verification and correction. An initial output is generated using a few-shot prompt-based approach. This output is then scrutinized with tools tailored to the task at hand. For instance, for fact-checking in question answering, CRITIC uses a Google-based search API to retrieve relevant web snippets, while for mathematical problem-solving, a Python interpreter verifies code execution and provides debugging information if errors occur. The verification feedback, structured as critiques, is appended to the prompt, enabling the LLM to correct its initial output. This verify-then-correct cycle is repeated iteratively, with the maximum number of interactions set per task or until stability in the output is observed.
- The following figure from the paper shows that the CRITIC framework consists of two steps: (1) verifying the output by interacting with external tools to generate critiques and (2) correcting the output based on the received critiques. We can iterate over such verify-then-correct process to enable continuous improvements.

![](https://aman.ai/images/papers/CRITIC.jpg)

- CRITIC’s **performance evaluation** demonstrates its effectiveness across three types of tasks:
    1. **Free-form Question Answering** - Here, CRITIC leverages web search to validate answers, achieving notable improvements in F1 scores over baseline methods, such as chain-of-thought (CoT) prompting and retrieval-augmented techniques.
    2. **Mathematical Program Synthesis** - By utilizing an interpreter for mathematical validation, CRITIC substantially enhances solve rates for datasets like GSM8k, surpassing program-of-thought (PoT) strategies.
    3. **Toxicity Reduction** - CRITIC employs the PERSPECTIVE API to monitor and reduce toxic content, achieving higher fluency and diversity in outputs while significantly reducing toxicity probabilities.
- Experimental results indicate that CRITIC consistently improves model performance, especially in cases requiring high factual accuracy or computational precision. The paper concludes by emphasizing that the inclusion of external feedback mechanisms allows LLMs to perform self-corrections that would be challenging through self-refinement alone.
- [Code](https://github.com/microsoft/ProphetNet/tree/master/CRITIC)

### Tool Calling

#### [Gorilla: Large Language Model Connected with Massive APIs](https://arxiv.org/abs/2305.15334)

- Large Language Models (LLMs) have seen an impressive wave of advances recently, with models now excelling in a variety of tasks, such as mathematical reasoning and program synthesis. However, their potential to effectively use tools via API calls remains unfulfilled. This is a challenging task even for today’s state-of-the-art LLMs such as GPT-4, largely due to their inability to generate accurate input arguments and their tendency to hallucinate the wrong usage of an API call.
- This paper by Patil et al. from UC Berkeley and Microsoft Research in 2023 introduces Gorilla, a finetuned LLaMA-based model that generates APIs to complete tasks by interacting with external tools and surpasses the performance of GPT-4 while writing API calls. When combined with a document retriever, Gorilla demonstrates a strong capability to adapt to test-time document changes, enabling flexible user updates or version changes. It also substantially mitigates the issue of hallucination, commonly encountered when prompting LLMs directly.
- To evaluate the model’s ability, they introduce APIBench, a comprehensive dataset consisting of Weights, TorchHub, and TensorHub APIs. The successful integration of the retrieval system with Gorilla demonstrates the potential for LLMs to use tools more accurately, keep up with frequently updated documentation, and consequently increase the reliability and applicability of their outputs.
- The following figure from the paper shows (top) the training procedure for Gorilla using the most exhaustive API dataset for ML to the best of their knowledge; (bottom) during inference Gorilla supports two modes - with retrieval, and zero-shot. In this example, it is able to suggest the right API call for generating the image from the user’s natural language query.

![](https://aman.ai/images/papers/Gorilla.jpg)

- [Project page](https://gorilla.cs.berkeley.edu/)

#### [MM-REACT: Prompting ChatGPT for Multimodal Reasoning and Action](https://arxiv.org/abs/2303.11381)

- This paper by Yang et al. from proposes MM-REACT, a system paradigm that integrates ChatGPT with a pool of vision experts to achieve multimodal reasoning and action.
- They define and explore a comprehensive list of advanced vision tasks that are intriguing to solve, but may exceed the capabilities of existing vision and vision-language models. To achieve such advanced visual intelligence, MM-REACT introduces a textual prompt design that can represent text descriptions, textualized spatial coordinates, and aligned file names for dense visual signals such as images and videos.
- MM-REACT’s prompt design allows language models to accept, associate, and process multimodal information, thereby facilitating the synergetic combination of ChatGPT and various vision experts. Zero-shot experiments demonstrate MM-REACT’s effectiveness in addressing the specified capabilities of interests and its wide application in different scenarios that require advanced visual understanding.
- Furthermore, they discuss and compare MM-REACT’s system paradigm with an alternative approach that extends language models for multimodal scenarios through joint finetuning.
- The following figure from the paper shows that MM-REACT allocates specialized vision experts with ChatGPT to solve challenging visual understanding tasks through multimodal reasoning and action. For example, the system could associate information from multiple uploaded receipts and calculate the total travel cost (“Multi-Image Reasoning”).

![](https://aman.ai/images/papers/MM-REACT1.jpg)

- The following figure from the paper shows the flowchart of MM-REACT for enhanced visual understanding with ChatGPT. The user input can be in the form of text, images, or videos, with the latter two represented as file path strings. ChatGPT is instructed to say specific watchwords in action request if a vision expert is required to interpret the visual inputs. Regular expression matching is applied to parse the expert’s name and the file path, which are then used to call the vision expert (action execution). The expert’s output (observation) is serialized as text and combined with the history to further activate ChatGPT. If no extra experts are needed, MM-REACT would return the final response to the user. The right figure shows a single-round vision expert execution, which is the component that constructs the full execution flow.

![](https://aman.ai/images/papers/MM-REACT2.jpg)

#### [Efficient Tool Use with Chain-of-Abstraction Reasoning](https://arxiv.org/abs/2401.17464?)

- This paper by Gao et al. from EPFL and FAIR introduces a novel method called Chain-of-Abstraction (CoA) reasoning, which optimizes LLMs for multi-step reasoning by using tools to access external knowledge. CoA decouples general reasoning from domain-specific information, which is later retrieved using specialized tools, enhancing LLM accuracy and efficiency in domains that require complex reasoning chains, such as mathematics and Wikipedia-based question answering (Wiki QA). The CoA method trains LLMs to generate reasoning chains with placeholders for domain knowledge, allowing parallel tool usage and reducing the lag introduced by interleaving LLM outputs with tool responses.
    
- Implementation of CoA reasoning involves a two-stage fine-tuning process: first, LLMs are trained to produce abstract reasoning chains that include placeholders for required operations or knowledge, such as calculations or article references. Then, these chains are reified by filling placeholders with actual knowledge retrieved from tools like an equation solver or a Wikipedia search engine. This approach enables parallel processing, as tools fill in specific information after generating the complete reasoning chain, speeding up inference significantly compared to sequential tool-augmented approaches.
    
- For the training pipeline, the authors fine-tune LLMs by re-writing gold-standard answers into abstract reasoning chains labeled with placeholders (e.g., “[20 + 35 = y1]”) for mathematical derivations or search queries for Wiki QA. Verification with domain-specific tools ensures that placeholders align with expected outcomes. For instance, in math, an equation solver calculates final results for placeholders. For Wiki QA, a combination of Wikipedia search (BM25 retriever) and NER extracts relevant articles and entities, which are then matched against gold references.
- The following figure from the paper shows an overview of chain-of-abstraction reasoning with tools. Given a domain question (green scroll), a LLM is fine-tuned to first generate an abstract multi-step reasoning chain (blue bubble), and then call external tools to reify the chain with domain-specific knowledge (orange label). The final answer (yellow bubble) is obtained based on the reified chain of reasoning.

![](https://aman.ai/images/papers/CoA.jpg)

- Evaluation on GSM8K and HotpotQA datasets showed CoA’s significant improvements in both accuracy and inference speed, outperforming baselines such as Toolformer and traditional chain-of-thought (CoT) methods by 6-7.5% accuracy in mathematics and Wiki QA. Additionally, CoA demonstrated robust generalization in zero-shot settings on other datasets like SVAMP and Natural Questions, with human evaluations confirming reduced reasoning and arithmetic errors.

### Planning

#### [Chain of Thought Prompting Elicits Reasoning in Large Language Models](https://arxiv.org/abs/2201.11903)

- Although scaling up language model size has reliably improved performance on a range of NLP tasks, even the largest models currently struggle with certain reasoning tasks such as arithmetic reasoning, math word problems, symbolic manipulation, and commonsense reasoning.
- This paper by Wei et al. from Google in 2022 explores the ability of language models to generate a coherent chain of thought – a series of short sentences that mimic the reasoning process a person might have when responding to a question. The idea is strikingly simple: instead of being terse while prompting show the model a few examples of a multi-step reasoning process (the like of which a human would use). Couple this with LLMs (the larger the better) and magic happens! Check out the below image from the paper.

![](https://aman.ai/images/papers/cot.jpeg)

- They have explored chain of thought prompting as a simple and broadly applicable method for enhancing reasoning in language models. The superb results you can elucidate via this method are an emergent property of model scale (surprise surprise) - bigger models benefit more from this, and the conclusion holds across models (LaMDA, GPT, PaLM).
- Interestingly enough, the more complex the task of interest is (in the sense of requiring multi-step reasoning approach), the bigger the boost from the chain of thought prompting!
- In order to make sure that the performance boost comes from this multi-step approach and not simply because of e.g. more compute, the authors have done a couple of ablations: (i) outputting a terse equation instead of a multi-step reasoning description, (ii) outputting the answer and only then the chain of thought, etc. None of these experiments yielded good results.
- The method also proved to be fairly robust (always outperforms standard prompting) to the choice of exact few shot exemplars. Despite different annotators, different styles, etc. the method is always better than standard prompting.
- Through experiments on arithmetic, symbolic, and commonsense reasoning, they find that chain of thought processing is an emergent property of model scale that can be induced via prompting and can enable sufficiently large language models to better perform reasoning tasks that otherwise have flat scaling curves.

#### [HuggingGPT: Solving AI Tasks with ChatGPT and Its Friends in HuggingFace](https://arxiv.org/abs/2303.17580)

- Solving complicated AI tasks with different domains and modalities is a key step toward advanced artificial intelligence. While there are abundant AI models available for different domains and modalities, they cannot handle complicated AI tasks.
- This paper by Shen et al. from Zhejiang University and Microsoft Research Asia in 2023 advocates that LLMs could act as a controller to manage existing AI models to solve complicated AI tasks and language could be a generic interface to empower this, considering the exceptional ability large language models (LLMs) have exhibited in language understanding, generation, interaction, and reasoning, etc. Based on this philosophy, they present HuggingGPT, a framework that leverages LLMs (e.g., ChatGPT) to connect various AI models in machine learning communities (e.g., Weights) to solve AI tasks.
- Specifically, they use ChatGPT to conduct task planning when receiving a user request, select models according to their function descriptions available in Weights, execute each subtask with the selected AI model, and summarize the response according to the execution results. By leveraging the strong language capability of ChatGPT and abundant AI models in Weights, HuggingGPT is able to cover numerous sophisticated AI tasks in different modalities and domains and achieve impressive results in language, vision, speech, and other challenging tasks, which paves a new way towards advanced artificial intelligence.
- Summary:
    1. HuggingGPT is recently introduced as a suitable middleware to bridge the connections between Large Language Models (LLMs) and AI models. The workflow goes as follows.
    2. Users can send a request (multimodal for sure) which will be processed by an LLM controller. The LLM analyzes the request, understands the intention of the user, and generates possible solvable sub-tasks.
    3. ChatGPT selects and invokes the corresponding models hosted on Weights to solve each subtask.
    4. Once tasks are executed, the invoked model returns the results to the ChatGPT controller.
    5. Finally, ChatGPT integrates the prediction of all models and generates the response.
    6. It is amazing how HuggingGPT can show its reasoning and point to its in-context task-model assignment as intermediary steps before generating the output.
- The following figure from the paper shows that language serves as an interface for LLMs (e.g., ChatGPT) to connect numerous AI models (e.g., those in Weights) for solving complicated AI tasks. In this concept, an LLM acts as a controller, managing and organizing the cooperation of expert models. The LLM first plans a list of tasks based on the user request and then assigns expert models to each task. After the experts execute the tasks, the LLM collects the results and responds to the user.

![](https://aman.ai/images/papers/HuggingGPT.jpg)

- [Code](https://github.com/microsoft/JARVIS)

#### [Understanding the Planning of LLM Agents: a Survey](https://arxiv.org/abs/2402.02716)

- This paper by Xu Huang et al. from the USTC and Huawei Noah’s Ark Lab presents a comprehensive survey of the planning capabilities of Large Language Model (LLM)-based agents, systematically categorizing recent approaches and identifying challenges in leveraging LLMs as planning modules. The authors define a taxonomy that classifies existing LLM-agent planning methods into five main categories: Task Decomposition, Multi-Plan Selection, External Planner-Aided Planning, Reflection and Refinement, and Memory-Augmented Planning. This taxonomy serves as a framework for understanding how various methods address the complexity of planning tasks in autonomous agents and highlights key developments in each category.
- The paper explores Task Decomposition, where complex tasks are divided into manageable sub-tasks, categorized further into Decomposition-First and Interleaved Decomposition. Techniques like Chain-of-Thought (CoT) and ReAct are examined for their methods in guiding LLMs to sequentially reason through tasks, emphasizing the benefits of sub-task correlation while noting limitations in handling lengthy planning sequences due to memory constraints. Multi-Plan Selection is detailed with methods like Tree-of-Thought (ToT) and Graph-of-Thought (GoT), where LLMs generate multiple candidate plans and then employ search algorithms (e.g., Monte Carlo Tree Search) to choose optimal paths, addressing the stochastic nature of LLM planning but also noting challenges with computational overhead.
- External Planner-Aided Planning is reviewed, dividing approaches into symbolic and neural planners, where LLMs act primarily as intermediaries, structuring tasks for external systems like PDDL or reinforcement learning-based neural planners. The symbolic planners enhance task formalization, while neural planners like DRRN model LLM-aided planning as Markov decision processes, showing efficiency in domain-specific scenarios. Reflection and Refinement strategies, such as Self-Refine and Reflexion, use iterative planning and feedback mechanisms to allow LLM agents to self-correct based on previous errors, resembling reinforcement learning updates but emphasizing textual feedback over parameter adjustments.
- This paper by Xu Huang et al. from the USTC and Huawei Noah’s Ark Lab presents a comprehensive survey of the planning capabilities of Large Language Model (LLM)-based agents, systematically categorizing recent approaches and identifying challenges in leveraging LLMs as planning modules. The authors define a taxonomy that classifies existing LLM-agent planning methods into five main categories: Task Decomposition, Multi-Plan Selection, External Planner-Aided Planning, Reflection and Refinement, and Memory-Augmented Planning. This taxonomy serves as a framework for understanding how various methods address the complexity of planning tasks in autonomous agents and highlights key developments in each category.
- Memory-Augmented Planning is discussed through RAG-based memory, which retrieves task-relevant information to support planning, and embodied memory, where agents fine-tune on experiential data, embedding learned knowledge into model parameters. Examples like MemGPT and TDT illustrate how different memory types enhance planning capabilities, balancing between update costs and memorization capacity.
- The following figure from the paper shows a taxonomy on LLM-Agent planning.

![](https://aman.ai/images/papers/LLM-Agent_Planning.jpg)

- The paper evaluates the effectiveness of these approaches on four benchmarks, demonstrating that strategies involving task decomposition, multi-path selection, and reflection significantly improve performance, albeit at higher computational costs. Challenges identified include LLM hallucinations, plan feasibility under complex constraints, efficiency, and limitations in handling multi-modal feedback. Future directions suggested include incorporating symbolic models for constraint handling, optimizing for planning efficiency, and developing realistic evaluation environments to more closely simulate real-world agent interactions. This survey provides a foundational overview of LLM planning, guiding future work toward robust and adaptable planning agents.

### Multi-Agent Collaboration

#### [ChatDev: Communicative Agents for Software Development](https://arxiv.org/abs/2307.07924)

- The paper by Qian et al. from Tsinghua, The University of Sydney, BUPT, and Modelbest Inc., published in ACL 2024, presents ChatDev, a software development framework using multiple agents powered by large language models (LLMs) to facilitate collaborative tasks within the software development lifecycle, including design, coding, and testing. The framework is designed to streamline multi-agent communication for more coherent, effective problem-solving across these phases. Key innovations include a structured “chat chain” approach and a “communicative dehallucination” mechanism, both aimed at enhancing the quality and executability of the generated code.
- ChatDev’s chat chain organizes the workflow into sequential phases—design, coding (subdivided into code writing and completion), and testing (split between code review and system testing)—with each phase containing sequential subtasks. Each subtask is addressed by an “instructor” and an “assistant” agent, who engage in multi-turn dialogues to collaboratively develop solutions, making it easier to handle complex requirements through natural language exchanges in design and programming dialogues in development. The framework thus maintains a coherent flow across phases, facilitating effective transitions and linking subtasks while offering transparency for monitoring intermediate solutions and issues.
- The communicative dehallucination mechanism addresses the problem of LLM-induced hallucinations in code by prompting agents to seek additional details from the instructor before finalizing responses. This enables agents to achieve more precise task outcomes, reducing instances of incomplete or unexecutable code. In this pattern, the assistant initially requests further guidance, which the instructor then clarifies, allowing the assistant to proceed with an optimized response. This iterative approach helps minimize errors and increases the quality and reliability of the generated code.
- The following figure from the paper shows that upon receiving a preliminary task requirement (e.g., “develop a Gomoku game”), these software agents engage in multi-turn communication and perform instruction-following along a chain-structured workflow, collaborating to execute a series of subtasks autonomously to craft a comprehensive solution.

![](https://aman.ai/images/papers/MAC.jpg)

- **Implementation Details**:
    - **Role Assignment and Memory Management**: Each agent is assigned a specific role tailored to different stages in the workflow, like CEO, CTO, programmer, reviewer, and tester. These roles ensure that agents contribute specialized insights at each phase. ChatDev employs short-term memory to maintain contextual continuity within a phase and long-term memory for cross-phase coherence, selectively transmitting only key solutions rather than entire dialogues to avoid memory overload.
    - **Subtask Termination Criteria**: To streamline communication, a subtask concludes after two consecutive unchanged code updates or after 10 rounds of communication. This rule optimizes resource use and prevents redundant iterations.
    - **Prompt Engineering and LLM Integration**: In each subtask, prompt engineering is applied at the onset, followed by automated exchanges. ChatGPT-3.5, with a low temperature of 0.2, supports task-specific response generation, while Python-3.11.4 integration enables real-time feedback on executable code.
- **Evaluation**:
    - ChatDev was evaluated against baseline models GPT-Engineer and MetaGPT using metrics including completeness, executability, consistency, and overall quality. Results show ChatDev’s significant improvements in generating more executable and complete code, largely due to its structured chat chain and communicative dehallucination mechanisms. An ablation study highlights the importance of specific roles and the dehallucination mechanism in boosting software quality.
- [Code](https://github.com/OpenBMB/ChatDev)

#### [AutoGen: Enabling Next-Gen LLM Applications Via Multi-Agent Conversation](https://arxiv.org/abs/2308.08155)

- This paper by Wu et al. from Microsoft Research, Pennsylvania State University, University of Washington, and Xidian University, introduces AutoGen, an open-source framework designed to facilitate the development of multi-agent large language model (LLM) applications. The framework allows the creation of customizable, conversable agents that can operate in various modes combining LLMs, human inputs, and tools.
- AutoGen agents can be easily programmed using both natural language and computer code to define flexible conversation patterns for different applications. The framework supports hierarchical chat, joint chat, and other conversation patterns, enabling agents to converse and cooperate to solve tasks. The agents can hold multiple-turn conversations with other agents or solicit human inputs, enhancing their ability to solve complex tasks.

![](https://aman.ai/images/papers/AutoGen_1.jpg)

- Key technical details include the design of conversable agents and conversation programming. Conversable agents can send and receive messages, maintain internal context, and be configured with various capabilities such as LLMs, human inputs, and tools. These agents can also be extended to include more custom behaviors. Conversation programming involves defining agent roles and capabilities and programming their interactions using a combination of natural and programming languages. This approach simplifies complex workflows into intuitive multi-agent conversations.
- **Implementation details:**
    1. **Conversable Agents**: AutoGen provides a generic design for agents, enabling them to leverage LLMs, human inputs, tools, or a combination. The agents can autonomously hold conversations and solicit human inputs at certain stages. Developers can easily create specialized agents with different roles by configuring built-in capabilities and extending agent backends.
    2. **Conversation Programming**: AutoGen adopts a conversation programming paradigm to streamline LLM application workflows. This involves defining conversable agents and programming their interactions via conversation-centric computation and control. The framework supports various conversation patterns, including static and dynamic flows, allowing for flexible agent interactions.
    3. **Unified Interfaces and Auto-Reply Mechanisms**: Agents in AutoGen have unified interfaces for sending, receiving, and generating replies. An auto-reply mechanism enables conversation-driven control, where agents automatically generate and send replies based on received messages unless a termination condition is met. Custom reply functions can also be registered to define specific behavior patterns.
    4. **Control Flow**: AutoGen allows control over conversations using both natural language and programming languages. Natural language prompts guide LLM-backed agents, while Python code specifies conditions for human input, tool execution, and termination. This flexibility supports diverse multi-agent conversation patterns, including dynamic group chats managed by the `GroupChatManager` class.

![](https://aman.ai/images/papers/AutoGen.jpg)

- The framework’s architecture defines agents with specific roles and capabilities, interacting through structured conversations to process tasks efficiently. This approach improves task performance, reduces development effort, and enhances application flexibility. Key technical aspects include using a unified interface for agent interaction, conversation-centric computation for defining agent behaviors, and conversation-driven control flows that manage interactions among agents.
- **Applications demonstrate AutoGen’s capabilities in various domains:**
    - **Math Problem Solving**: AutoGen builds systems for autonomous and human-in-the-loop math problem solving, outperforming other approaches on the MATH dataset.
    - **Retrieval-Augmented Code Generation and Question Answering**: The framework enhances retrieval-augmented generation systems, improving performance on question-answering tasks through interactive retrieval mechanisms.
    - **Decision Making in Text World Environments**: AutoGen implements effective interactive decision-making applications using benchmarks like ALFWorld.
    - **Multi-Agent Coding**: The framework simplifies coding tasks by dividing responsibilities among agents, improving code safety and efficiency.
    - **Dynamic Group Chat**: AutoGen supports dynamic group chats, enabling collaborative problem-solving without predefined communication orders.
    - **Conversational Chess**: The framework creates engaging chess games with natural language interfaces, ensuring valid moves through a board agent.
- The empirical results indicate that AutoGen significantly outperforms existing single-agent and some multi-agent systems in complex task environments by effectively integrating and managing multiple agents’ capabilities. The paper includes a figure illustrating the use of AutoGen to program a multi-agent conversation, showing built-in agents, a two-agent system with a custom reply function, and the resulting automated agent chat.
- The authors highlight the potential for AutoGen to improve LLM applications by reducing development effort, enhancing performance, and enabling innovative uses of LLMs. Future work will explore optimal multi-agent workflows, agent capabilities, scaling, safety, and human involvement in multi-agent conversations. The open-source library invites contributions from the broader community to further develop and refine AutoGen.

#### [APIGen: Automated Pipeline for Generating Verifiable and Diverse Function-Calling Datasets](https://arxiv.org/abs/2406.18518)

- This paper introduces **APIGen**, a novel automated data generation pipeline designed by Salesforce AI Research to create reliable and diverse datasets specifically for function-calling applications in LLMs. The APIGen framework addresses challenges in training function-calling agents by producing a large-scale dataset with high-quality, verified function calls. The authors leverage a structured, multi-stage verification approach to generate a dataset that enables fine-tuning LLMs, which demonstrates significant improvements in performance on the Berkeley Function-Calling Benchmark (BFCL).
- APIGen includes a dataset of 60,000 entries across 3,673 APIs from 21 categories, encompassing different query styles (simple, multiple, parallel, and parallel multiple). Notably, APIGen-trained models, even with relatively fewer parameters, achieved strong results in function-calling benchmarks, surpassing larger LLMs such as GPT-4 and Claude-3.
- **APIGen Framework and Data Generation**: The APIGen framework generates query-answer pairs through a structured pipeline, involving:
    1. **Sampling**: APIs and QA pairs are sampled from a comprehensive library and formatted in a unified JSON schema.
    2. **Prompting**: LLMs are prompted using diverse templates to generate function-call responses in JSON format, promoting response variability across different real-world scenarios.
    3. **Multi-Stage Verification**:
        - **Format Checker**: Ensures generated data follows JSON specifications and filters poorly formatted entries.
        - **Execution Checker**: Executes function calls to verify correctness against backend APIs, discarding calls with errors.
        - **Semantic Checker**: A second LLM verifies alignment between generated answers and query intent, further refining data quality.
- This pipeline allows APIGen to create a dataset that supports diverse function-calling scenarios, enhancing model generalization and robustness. The figure below from the paper illustrates the post-process filters.

![](https://aman.ai/images/papers/APIGen1.jpg)

- The figure below from the paper illustrates JSON data format examples.

![](https://aman.ai/images/papers/APIGen2.png)

- **Diversity and Scalability**: APIGen emphasizes diversity by categorizing queries and sampling API descriptions from different sources. It includes four query types:
    - **Simple**: A single API call per query.
    - **Multiple**: Requires selecting the appropriate function from various APIs.
    - **Parallel**: Executes multiple API calls in one response.
    - **Parallel Multiple**: Combines both parallel and multiple query types, calling for intricate function handling.
- To ensure scalability, APIGen processes data from multiple API formats, such as REST and Python functions, adapting them into its JSON format. This modular approach accommodates a variety of API sources with minimal adjustments.
- **Dataset and Implementation**: The dataset generation involves filtering APIs for quality, executing requests to ensure validity, and regenerating API descriptions where necessary. The resulting dataset spans diverse categories, ensuring comprehensive coverage across fields like finance, technology, and social sciences.
- Training experiments involved two model versions, **xLAM-1B (FC)** and **xLAM-7B (FC)**, demonstrating APIGen’s dataset’s efficacy. Models trained with APIGen data achieved remarkable accuracy on the BFCL, particularly in scenarios involving complex, parallel, and multi-call queries. The xLAM-7B model notably ranked 6th on the BFCL leaderboard, surpassing models like GPT-4 and Llama3.
- **Human Evaluation and Future Work**: APIGen’s effectiveness was validated through human evaluation, with over 95% of samples passing quality checks. The authors plan to extend APIGen to support more API types and incorporate multi-turn interaction capabilities for function-calling agents.
- By providing a structured and scalable approach to high-quality dataset generation, APIGen sets a new standard for training robust function-calling LLMs, addressing gaps in current datasets and enhancing LLMs’ real-world applicability.
- [Hugging Face](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k); [Project page](https://apigen-pipeline.github.io/)

#### [AutoAgents: a Framework for Automatic Agent Generation](https://arxiv.org/abs/2309.17288)

- This paper from Chen et al. published in IJCAI 2024 presents AutoAgents, a novel framework designed for dynamic multi-agent generation and coordination, enabling language models to construct adaptive AI teams for a wide range of tasks. Unlike traditional systems that rely on static, predefined agents, AutoAgents generates task-specific agents autonomously, allowing for flexible collaboration across varied domains. The framework introduces a drafting and execution stage to handle complex task environments and facilitate effective role assignment and solution planning.
- In the drafting stage, three primary agents—Planner, Agent Observer, and Plan Observer—collaborate to define and refine an agent team and execution plan. The Planner generates specialized agents, each described by prompts detailing roles, objectives, constraints, and toolsets. The Agent Observer evaluates agents for task suitability, ensuring they are adequately diverse and relevant. Simultaneously, the Plan Observer assesses the execution plan, refining it to address any gaps and optimize agent collaboration.
- The execution stage leverages two mechanisms for task completion: self-refinement and collaborative refinement. In self-refinement, individual agents iterate on their tasks, improving their outputs through cycles of reasoning and self-evaluation. Collaborative refinement allows agents to pool expertise, enhancing task execution through interdisciplinary dialogue. A predefined Action Observer oversees coordination, adjusting task allocations and managing memory across agents to maintain efficiency and coherence.
- To enhance adaptability in complex tasks, AutoAgents incorporates three types of memory—short-term, long-term, and dynamic—to manage historical data and context for each action. Dynamic memory, in particular, facilitates the Action Observer’s access to essential prior actions, optimizing task-related decisions.
- - The figure below from the paper illustrates a schematic diagram of AutoAgents. The system takes the user input as a starting point and generates a set of specialized agents for novel writing, along with a corresponding execution plan. The agents collaboratively carry out the tasks according to the plan and produce the final novel. Meanwhile, an observer monitors the generation and execution of the Agents and the plan, ensuring the quality and coherence of the process.

![](https://aman.ai/images/papers/AutoAgents.jpg)

- Experiments across open-ended question-answering and trivia creative writing tasks validate AutoAgents’ superior performance. AutoAgents surpasses traditional models, including GPT-4, in both knowledge acquisition and reasoning quality. The system demonstrates a significant increase in knowledge integration, especially when handling tasks demanding extensive domain-specific information. A case study in software development shows how AutoAgents generates diverse expert roles (e.g., game designer, programmer, tester) for developing a Python-based Tetris game, highlighting the versatility of the agent team composition.
- Implementation-wise, AutoAgents utilizes GPT-4 API with a controlled temperature setting for reproducibility. Experiment parameters include a maximum of three drafting discussions and five execution refinements, with dynamic prompts designed to guide each agent’s expertise and actions.
- In summary, AutoAgents offers a significant advancement in adaptive, collaborative AI systems by automating agent generation and task planning, reinforcing the capabilities of LLMs in handling complex, domain-spanning tasks through self-organizing, expert-driven agent teams.
- [Code](https://github.com/Link-AGI/AutoAgents)

#### [MetaGPT: Meta Programming for Multi-Agent Collaborative Framework](https://arxiv.org/abs/2308.00352)

- Recently, remarkable progress has been made in automated task-solving through the use of multi-agents driven by large language models (LLMs). However, existing works primarily focuses on simple tasks lacking exploration and investigation in complicated tasks mainly due to the hallucination problem. This kind of hallucination gets amplified infinitely as multiple intelligent agents interact with each other, resulting in failures when tackling complicated problems.
- This paper by Hong et al. from DeepWisdom, Xiamen University, CUHK, Shenzhen, Nanjing University, UPenn, Berkeley introduces MetaGPT, an innovative framework that infuses effective human workflows as a meta programming approach into LLM-driven multi-agent collaboration. In particular, MetaGPT first encodes Standardized Operating Procedures (SOPs) into prompts, fostering structured coordination. And then, it further mandates modular outputs, bestowing agents with domain expertise paralleling human professionals to validate outputs and reduce compounded errors.
- In this way, MetaGPT leverages the assembly line work model to assign diverse roles to various agents, thus establishing a framework that can effectively and cohesively deconstruct complex multi-agent collaborative problems.
- Their experiments conducted on collaborative software engineering tasks illustrate MetaGPT’s capability in producing comprehensive solutions with higher coherence relative to existing conversational and chat-based multi-agent systems. This underscores the potential of incorporating human domain knowledge into multi-agents, thus opening up novel avenues for grappling with intricate real-world challenges.
- The following figure from the paper shows a comparative depiction of the software development SOP between MetaGPT and real-world human team. The MetaGPT approach showcases its ability to decompose high-level tasks into detailed actionable components handled by distinct roles (ProductManager, Architect, ProjectManager, Engineer), thereby facilitating role-specific expertise and coordination. This methodology mirrors human software development teams, but with the advantage of improved efficiency, precision, and consistency. The diagram illustrates how MetaGPT is designed to handle task complexity and promote clear role delineations, making it a valuable tool for complex software development scenarios.

![](https://aman.ai/images/papers/MetaGPT.jpg)

#### [Chameleon: Plug-and-Play Compositional Reasoning with Large Language Models](https://arxiv.org/abs/2304.09842)

- Large language models (LLMs) have achieved remarkable progress in solving various natural language processing tasks due to emergent reasoning abilities. However, LLMs have inherent limitations as they are incapable of accessing up-to-date information (stored on the Web or in task-specific knowledge bases), using external tools, and performing precise mathematical and logical reasoning.
- This paper by Lu et al. from UCLA and Microsoft Research presents Chameleon, an AI system that mitigates these limitations by augmenting LLMs with plug-and-play modules for compositional reasoning. Chameleon synthesizes programs by composing various tools (e.g., LLMs, off-the-shelf vision models, web search engines, Python functions, and heuristic-based modules) for accomplishing complex reasoning tasks.
- At the heart of Chameleon is an LLM-based planner that assembles a sequence of tools to execute to generate the final response.
- They showcase the effectiveness of Chameleon on two multi-modal knowledge-intensive reasoning tasks: ScienceQA and TabMWP. Chameleon, powered by GPT-4, achieves an 86.54% overall accuracy on ScienceQA, improving the best published few-shot result by 11.37%. On TabMWP, GPT-4-powered Chameleon improves the accuracy by 17.0%, lifting the state of the art to 98.78%.
- Their analysis also shows that the GPT-4-powered planner exhibits more consistent and rational tool selection via inferring potential constraints from instructions, compared to a ChatGPT-powered planner.
- The following figures from the paper shows two examples from their Chameleon with GPT-4 on TabMWP, a mathematical reasoning benchmark with tabular contexts. Chameleon demonstrates flexibility and efficiency in adapting to different queries that require various reasoning abilities.

![](https://aman.ai/images/papers/Chameleon.jpg)

### OctoTools: an Agentic Framework with Extensible Tools for Complex Reasoning

- This paper by Lu et al. from Stanford introduces OctoTools, a training-free, extensible agentic framework designed to enhance large language models (LLMs) with external tools for complex reasoning. Unlike prior approaches, OctoTools does not require additional training data or domain-specific tool constraints, making it adaptable across diverse reasoning tasks.
- **Key contributions:**
    - Tool cards: Standardized tool wrappers encapsulate various functionalities, enabling seamless integration and execution without framework modifications.
    - Planner-executor framework: Separates high-level task planning from tool execution. The planner formulates a structured problem-solving strategy, while the executor generates and executes tool commands dynamically.
    - Task-specific toolset optimization: A lightweight algorithm selects the most beneficial subset of tools for each task, improving both accuracy and efficiency.
    - Comprehensive benchmarking: OctoTools was evaluated on 16 reasoning benchmarks (MathVista, MMLU-Pro, MedQA, GAIA-Text, etc.), achieving a 9.3% accuracy improvement over GPT-4o and up to 10.6% over existing agent frameworks like AutoGen, GPT-Functions, and LangChain.
- The following figures from the paper shows the framework of OctoTools. (1) Tool cards define tool-usage metadata and encapsulate tools, enabling training-free integration of new tools without additional training or framework refinement. (2) The planner governs both high-level and low-level planning to address the global objective and refine actions step by step. (3) The executor instantiates tool calls by generating executable commands and save structured results in the context. The final answer is summarized from the full trajectory in the context. Furthermore, the task-specific toolset optimization algorithm learns to select a beneficial subset of tools for downstream tasks.

![](https://aman.ai/images/papers/OctoTools.jpg)

- **Implementation details:**
    - The system operates through three core modules:
        1. Tool cards: Define metadata for each tool, including input-output formats, constraints, and best-use cases (e.g., Image Captioner, Object Detector, Python Calculator).
        2. Planner: Determines which tools to invoke and structures problem decomposition into sub-goals.
        3. Executor: Converts planner instructions into executable commands, runs tools, and updates context iteratively.
    - Uses structured reasoning pipelines, where each step refines prior results to enhance multi-step problem-solving.
    - The optimization algorithm iteratively selects tools based on validation accuracy, ensuring efficiency without introducing unnecessary complexity.
- **Experimental results:**
    - Achieved 58.5% accuracy across 16 diverse benchmarks, outperforming GPT-4o (49.2%), LangChain (51.2%), and AutoGen (47.9%).
    - Demonstrated superior multi-step reasoning and tool utilization, particularly in math, science, and medical domains.
    - Ablation studies confirmed that task planning, external tool calling, and multi-step problem solving each contribute significantly to performance improvements.
- **Conclusion:**
    - OctoTools is a scalable, modular, and effective solution for enhancing LLMs with external tool integration.
    - It bridges the gap between generic LLM reasoning and domain-specific problem solving, making it a powerful alternative to traditional AI agent frameworks.
    - Future work includes real-time query-based tool selection, multi-agent collaboration, and expanding domain-specific functionalities.
- [Website](https://octotools.github.io/); [GitHub Repository](https://github.com/octotools/octotools); [Hugging Face Demo](https://huggingface.co/spaces/OctoTools/octotools)

## Further Reading

- [Communicative Agents for Software Development](https://arxiv.org/abs/2307.07924) by Qian et al. (2023): Discusses the application of communicative agents in software development by introducing ChatDev, an open-source virtual software company run by agents.
- [AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation](https://arxiv.org/abs/2308.08155) by Wu et al. (2023): Explores how multi-agent conversations can enhance LLM applications.
- [MetaGPT: Meta Programming for A Multi-Agent Collaborative Framework](https://arxiv.org/abs/2308.00352) by Hong et al. (2023): Provides insights into meta programming within a multi-agent framework.
- [Task-driven Autonomous Agent Utilizing GPT-4, Pinecone, and LangChain for Diverse Applications](https://yoheinakajima.com/task-driven-autonomous-agent-utilizing-gpt-4-pinecone-and-langchain-for-diverse-applications/)
- [Anthropic’s blog on building effective agents](https://www.anthropic.com/engineering/building-effective-agents)
- [Model Context Protocol (MCP): Specifications](https://spec.modelcontextprotocol.io/specification/2024-11-05/)
- [What is Model Context Protocol (MCP)? How it simplifies AI integrations compared to APIs](https://norahsakal.com/blog/mcp-vs-api-model-context-protocol-explained/)
- [Don’t Sleep on Single-agent Systems](https://www.all-hands.dev/blog/dont-sleep-on-single-agent-systems)

## References

- [The Batch – Weekly Issues: Issue 245](https://www.deeplearning.ai/the-batch/issue-245/)
- [The Batch – Agentic Design Patterns](https://www.deeplearning.ai/the-batch/how-agents-can-improve-llm-performance/)
- [AWS: What are AI Agents](https://aws.amazon.com/what-is/ai-agents/)
- [LLM Agents by Prompt Engineering Guide](https://www.promptingguide.ai/research/llm-agents#notable-llm-based-agents)
- [Introduction to LLM Agents, Nvidia Blog](https://developer.nvidia.com/blog/introduction-to-llm-agents/)
- [Aishwarya Reganti’s Agents 101 Guide](https://github.com/aishwaryanr/awesome-generative-ai-guide/blob/main/resources/agents_101_guide.md)
- [Nvidia Introduction to LLM Agents](https://developer.nvidia.com/blog/introduction-to-llm-agents/)
- [Deepchecks](https://deepchecks.com/glossary/llm-agents/)
- [Weaviate Blog: What is Agentic RAG?](https://weaviate.io/blog/what-is-agentic-rag)
- [Agentic RAG with VoyageAI, Gemini and LangGraph](https://myscale.com/blog/agentic-rag-with-voyageai-gemini-and-langgraph/)
- [Why MCP Won](https://www.latent.space/p/why-mcp-won)
- [Philipp Schmid: Model Context Protocol (MCP) an overview](https://www.philschmid.de/mcp-introduction)
- [What Is MCP, and Why Is Everyone – Suddenly!– Talking About It?](https://huggingface.co/blog/Kseniase/mcp)
- [What is MCP](https://python.useinstructor.com/blog/2025/03/27/understanding-model-context-protocol-mcp/#conclusion)
- [I gave Claude root access to my server… Model Context Protocol explained](https://www.youtube.com/watch?v=HyzlYwjoXOQ)
- [Building Agents with Model Context Protocol - Full Workshop with Mahesh Murag of Anthropic](https://www.youtube.com/watch?v=kQmXtrmQ5Zg)

## Citation

If you found our work useful, please cite it as:

![](https://aman.ai/images/copy.png)

`@article{Chadha2020DistilledAgents,   title   = {Agents},   author  = {Chadha, Aman and Jain, Vinija},   journal = {Distilled AI},   year    = {2020},   note    = {\url{https://vinija.ai}} }`

-  [](https://github.com/amanchadha)|  [](https://citations.amanchadha.com/)|  [](https://twitter.com/i_amanchadha)|  [](mailto:hi@aman.ai)| 

[www.amanchadha.com](https://www.amanchadha.com/)