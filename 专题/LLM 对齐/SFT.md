
> https://huggingface.co/learn/llm-course/chapter11/1?fw=pt



模型可以针对特定任务（如摘要和问答）进行微调，但更常见的是在广泛的任务上同时微调语言模型，这种方法被称为监督微调（SFT）。这一过程帮助模型变得更加多才多艺，能够处理多样化的用例，并使其更加有用并符合人类偏好。

## 一、聊天模板

聊天模板对于构建语言模型与用户之间的交互结构至关重要。无论是构建简单的聊天机器人还是复杂的 AI 代理，理解如何正确格式化对话内容对于从模型获得最佳结果都至关重要。聊天模板对于以下方面至关重要：

- 保持一致的对话结构
- 确保正确的角色识别
- 管理多轮对话的上下文
- 支持工具使用等高级功能  

### 1.1 模型类型与模板

#### 1.1.1 基础模型 vs 指令模型

基础模型通过原始文本数据训练来预测下一个 token，而指令模型则专门针对遵循指令和进行对话进行了微调。例如，`SmolLM2-135M` 是基础模型，而 `SmolLM2-135M-Instruct` 是其指令调优版本。

指令调优模型经过训练遵循特定的对话结构，使其更适合聊天机器人应用。此外，指令模型可以处理复杂的交互，包括工具使用、多模态输入和函数调用。

要使基础模型表现得像指令模型，我们需要以模型能够理解的统一方式格式化提示。这就是聊天模板的作用。ChatML 是一种模板格式，通过清晰的角色标识符（系统、用户、助手）来构建对话。参考[这里](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct/blob/e2c3f7557efbdec707ae3a336371d169783f1da1/tokenizer_config.json#L146)
```
"chat_template": "{% for message in messages %}{% if loop.first and messages[0]['role'] != 'system' %}{{ '<|im_start|>system\nYou are a helpful AI assistant named SmolLM, trained by Hugging Face<|im_end|>\n' }}{% endif %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
```

> [!warning]
> 使用指令模型时，务必验证使用的是正确的聊天模板格式。使用错误的模板可能导致模型性能不佳或意外行为。最简便的验证方法是检查 Hub 上的模型分词器配置。例如，`SmolLM2-135M-Instruct` 模型使用[此配置](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct/blob/e2c3f7557efbdec707ae3a336371d169783f1da1/tokenizer_config.json#L146)。  

#### 1.1.2 常见模板格式

这是 `SmolLM2` 和 `Qwen 2` 等模型使用的 ChatML 模板：

```sh
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
Hello!<|im_end|>
<|im_start|>assistant
Hi! How can I help you today?<|im_end|>
<|im_start|>user
What's the weather?<|im_start|>assistant
```

这是使用 `mistral` 模板格式的示例：

```sh
<s>[INST] You are a helpful assistant. [/INST]
Hi! How can I help you today?</s>
[INST] Hello! [/INST]
```

这些格式之间的主要区别包括：

1. ​**系统消息处理**​：
    - Llama 2 用 `<<SYS>>` 标签包裹系统消息
    - Llama 3 使用 `<|system|>` 标签和 `</s>` 结尾
    - Mistral 在第一条指令中包含系统消息
    - Qwen 使用明确的 `system` 角色和 `<|im_start|>` 标签
    - ChatGPT 使用 `SYSTEM:` 前缀
2. ​**消息边界**​：
    - Llama 2 使用 `[INST]` 和 `[/INST]` 标签
    - Llama 3 使用角色特定标签（`<|system|>`、`<|user|>`、`<|assistant|>`）和`</s>`结尾
    - Mistral 使用 `[INST]` 和 `[/INST]` 与 `<s>` 和 `</s>`
    - Qwen 使用角色特定的开始/结束标记
3. ​**特殊标记**​：
    - Llama 2 使用 `<s>` 和 `</s>` 作为对话边界
    - Llama 3 使用 `</s>` 结束每条消息
    - Mistral 使用 `<s>` 和 `</s>` 作为轮次边界
    - Qwen 使用角色特定的开始/结束标记

理解这些差异是使用各种模型的关键。transformers 库会帮助我们自动处理这些变化：

```python
from transformers import AutoTokenizer

# 这些会自动使用不同的模板
mistral_tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
qwen_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-7B-Chat")
smol_tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")

messages = [
    {"role": "system", "content": "你是一个乐于助人的助手。"},
    {"role": "user", "content": "你好！"},
]

# 每个都会根据其模型的模板格式化
mistral_chat = mistral_tokenizer.apply_chat_template(messages, tokenize=False)
qwen_chat = qwen_tokenizer.apply_chat_template(messages, tokenize=False)
smol_chat = smol_tokenizer.apply_chat_template(messages, tokenize=False)
```

#### 1.1.3 高级功能

聊天模板可以处理比简单对话交互更复杂的场景，包括：

1. ​**工具使用**​：当模型需要与外部工具或 API 交互时
2. ​**多模态输入**​：用于处理图像、音频或其他媒体类型
3. ​**函数调用**​：用于结构化函数执行
4. ​**多轮上下文**​：用于维护对话历史



对于多模态对话，聊天模板可以包含图像引用或 base64 编码的图像：

```python
messages = [
    {
        "role": "system",
        "content": "You are a helpful vision assistant that can analyze images.",
    },
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "What's in this image?"},
            {"type": "image", "image_url": "https://example.com/image.jpg"},
        ],
    },
]
```

这是一个带有工具使用的聊天模板示例：

```python
messages = [
    {
        "role": "system",
        "content": "You are an AI assistant that can use tools. Available tools: calculator, weather_api",
    },
    {"role": "user", "content": "What's 123 * 456 and is it raining in Paris?"},
    {
        "role": "assistant",
        "content": "Let me help you with that.",
        "tool_calls": [
            {
                "tool": "calculator",
                "parameters": {"operation": "multiply", "x": 123, "y": 456},
            },
            {"tool": "weather_api", "parameters": {"city": "Paris", "country": "France"}},
        ],
    },
    {"role": "tool", "tool_name": "calculator", "content": "56088"},
    {
        "role": "tool",
        "tool_name": "weather_api",
        "content": "{'condition': 'rain', 'temperature': 15}",
    },
]
```

### 1.2 最佳实践

#### 1.2.1 通用指南

使用聊天模板时，遵循以下关键实践：

1. ​**一致的格式化**​：在整个应用程序中使用相同的模板格式
2. ​**清晰的角色定义**​：为每条消息明确指定角色（系统、用户、助手、工具）
3. ​**上下文管理**​：维护对话历史时注意 token 限制
4. ​**错误处理**​：为工具调用和多模态输入包含适当的错误处理
5. ​**验证**​：在发送到模型之前验证消息结构

> [!tip]
> 需要避免的常见陷阱：
> 
> - 在同一应用程序中混合不同的模板格式
> - 因长对话历史而超出标记限制
> - 未正确转义消息中的特殊字符
> - 忘记验证输入消息结构
> - 忽略模型特定的模板要求  
> 

### 1.3 实践练习

让我们通过一个真实示例来实践实现聊天模板。按照以下步骤将 `HuggingFaceTB/smoltalk` 数据集转换为 chatml 格式：

```python
from datasets import load_dataset

dataset = load_dataset("HuggingFaceTB/smoltalk")
```

2. 创建处理函数：

python

复制

```python
def convert_to_chatml(example):
    return {
        "messages": [
            {"role": "user", "content": example["input"]},
            {"role": "assistant", "content": example["output"]},
        ]
    }
```

3. 使用所选模型的分词器应用聊天模板

记得验证您的输出格式是否符合目标模型的要求！  
</提示>

## 附加资源

- Hugging Face 聊天模板指南
- Transformers 文档
- 聊天模板示例仓库




## 1️⃣ 聊天模板

聊天模板用于结构化用户和AI模型之间的交互，确保响应的一致性和上下文的适当性。它们包括系统提示和基于角色的消息等组件。

## 2️⃣ 监督微调

监督微调（SFT）是将预训练语言模型适应特定任务的关键过程。它涉及在带标注的任务特定数据集上训练模型。有关SFT的详细指南，包括关键步骤和最佳实践，请参阅[TRL文档中的监督微调部分](https://huggingface.co/docs/trl/en/sft_trainer)。

## 3️⃣ 低秩适应（LoRA）

低秩适应（LoRA）是一种通过向模型层添加低秩矩阵来微调语言模型的技术。这允许在保留模型预训练知识的同时进行高效微调。LoRA的一个主要优点是显著的内存节省，使得在资源有限的硬件上微调大型模型成为可能。

## 4️⃣ 评估

评估是微调过程中至关重要的一步。它允许我们测量模型在任务特定数据集上的性能。

<提示>
⚠️ 为了利用模型中心和🤗 Transformers提供的所有功能，我们建议<a href="https://huggingface.co/join">创建一个账户</a>。
</提示>

## 



In [Chapter 2 Section 2](/course/chapter2/2), we saw that generative language models can be fine-tuned on specific tasks like summarization and question answering. However, nowadays it is far more common to fine-tune language models on a broad range of tasks simultaneously; a method known as supervised fine-tuning (SFT). This process helps models become more versatile and capable of handling diverse use cases. Most LLMs that people interact with on platforms like ChatGPT have undergone SFT to make them more helpful and aligned with human preferences. We will separate this chapter into four sections:

## 1️⃣ Chat Templates

Chat templates structure interactions between users and AI models, ensuring consistent and contextually appropriate responses. They include components like system prompts and role-based messages.

## 2️⃣ Supervised Fine-Tuning

Supervised Fine-Tuning (SFT) is a critical process for adapting pre-trained language models to specific tasks. It involves training the model on a task-specific dataset with labeled examples. For a detailed guide on SFT, including key steps and best practices, see [the supervised fine-tuning section of the TRL documentation](https://huggingface.co/docs/trl/en/sft_trainer).

## 3️⃣ Low Rank Adaptation (LoRA)

Low Rank Adaptation (LoRA) is a technique for fine-tuning language models by adding low-rank matrices to the model's layers. This allows for efficient fine-tuning while preserving the model's pre-trained knowledge. One of the key benefits of LoRA is the significant memory savings it offers, making it possible to fine-tune large models on hardware with limited resources.

## 4️⃣ Evaluation

Evaluation is a crucial step in the fine-tuning process. It allows us to measure the performance of the model on a task-specific dataset.

<Tip>
⚠️ In order to benefit from all features available with the Model Hub and 🤗 Transformers, we recommend <a href="https://huggingface.co/join">creating an account</a>.
</Tip>

## References

- [Transformers documentation on chat templates](https://huggingface.co/docs/transformers/main/en/chat_templating)
- [Script for Supervised Fine-Tuning in TRL](https://github.com/huggingface/trl/blob/main/trl/scripts/sft.py)
- [`SFTTrainer` in TRL](https://huggingface.co/docs/trl/main/en/sft_trainer)
- [Direct Preference Optimization Paper](https://arxiv.org/abs/2305.18290)
- [Supervised Fine-Tuning with TRL](https://huggingface.co/docs/trl/sft_trainer)
- [How to fine-tune Google Gemma with ChatML and Hugging Face TRL](https://github.com/huggingface/alignment-handbook)  
- [Fine-tuning LLM to Generate Persian Product Catalogs in JSON Format](https://huggingface.co/learn/cookbook/en/fine_tuning_llm_to_generate_persian_product_catalogs_in_json_format)



参考资料

- [Transformers关于聊天模板的文档](https://huggingface.co/docs/transformers/main/en/chat_templating)
- [TRL中的监督微调脚本](https://github.com/huggingface/trl/blob/main/trl/scripts/sft.py)
- [TRL中的`SFTTrainer`](https://huggingface.co/docs/trl/main/en/sft_trainer)
- [直接偏好优化论文](https://arxiv.org/abs/2305.18290)
- [使用TRL进行监督微调](https://huggingface.co/docs/trl/sft_trainer)
- [如何使用ChatML和Hugging Face TRL微调Google Gemma](https://github.com/huggingface/alignment-handbook)
- [微调LLM以生成JSON格式的波斯产品目录](https://huggingface.co/learn/cookbook/en/fine_tuning_llm_to_generate_persian_product_catalogs_in_json_forma)

