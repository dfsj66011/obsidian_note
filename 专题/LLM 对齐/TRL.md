v0.17.0

![](https://huggingface.co/datasets/trl-lib/documentation-images/resolve/main/trl_banner_dark.png)

## 一、开始 
### 1.1 TRL - Transformer Reinforcement Learning

TRL 是一个全栈库，我们提供了一套工具来训练 Transformer 语言模型，采用的方法包括监督微调（SFT）、群体相对策略优化（GRPO）、直接偏好优化（DPO）、奖励建模等。该库已与 🤗 transformers 集成。

通过 TRL 和其他库在 🤗 [smol 课程](https://github.com/huggingface/smol-course)中学习训练后优化。

#### 1.1.1 目录

文档分为以下几个部分：

- 入门指南：安装和快速入门指南。
- 概念指南：数据集格式、训练常见问题解答以及理解日志。
- 操作指南：减少内存使用、加速训练、分布式训练等。
- 集成：DeepSpeed、Liger Kernel、PEFT 等。
- 示例：示例概述、社区教程等。
- API：trainers、utils 等。


#### 1.1.2 博客文章

* [Open-R1: a fully open reproduction of DeepSeek-R1](https://huggingface.co/blog/open-r1)
* [Preference Optimization for Vision Language Models with TRL](https://huggingface.co/blog/dpo_vlm)
* [Putting RL back in RLHF](https://huggingface.co/blog/putting_rl_back_in_rlhf_with_rloo)
* [Finetune Stable Diffusion Models with DDPO via TRL](https://huggingface.co/blog/trl-ddpo)
* [Fine-tune Llama 2 with DPO](https://huggingface.co/blog/dpo-trl)
* [StackLLaMA: A hands-on guide to train LLaMA with RLHF](https://huggingface.co/blog/stackllama)
* [Fine-tuning 20B LLMs with RLHF on a 24GB consumer GPU](https://huggingface.co/blog/trl-peft)
* [Illustrating Reinforcement Learning from Human Feedback](https://huggingface.co/blog/rlhf)

-------


🏡 View all docsAWS Trainium & InferentiaAccelerateAmazon SageMakerArgillaAutoTrainBitsandbytesChat UIDataset viewerDatasetsDiffusersDistilabelEvaluateGradioH

[Command Line Interface (CLI)](https://huggingface.co/docs/trl/clis)[Customizing the Training](https://huggingface.co/docs/trl/customization)[Reducing Memory Usage](https://huggingface.co/docs/trl/reducing_memory_usage)[Speeding Up Training](https://huggingface.co/docs/trl/speeding_up_training)[

### 1.2 快速开始

#### 1.2.1 工作原理

通过近端策略优化算法（PPO）微调语言模型大致包括三个步骤：

1. 部署（Rollout）：语言模型根据查询（可能是句子的开头）生成响应或续写内容。
2. 评估（Evaluation）：通过函数、模型、人工反馈或它们的某种组合对查询和响应进行评估。关键在于该过程应为每对查询/响应生成一个标量值。优化目标将是最大化该值。
3. 优化（Optimization）：这是最复杂的部分。在优化步骤中，使用查询/响应对计算序列中标记的对数概率。这是通过训练后的模型和一个参考模型（通常是微调前的预训练模型）完成的。两个输出之间的 KL 散度被用作额外的奖励信号，以确保生成的响应不会过度偏离参考语言模型。然后使用近端策略优化（PPO）对活跃的语言模型进行训练。

整个过程如下图所示：

![|600](https://huggingface.co/datasets/trl-lib/documentation-images/resolve/main/trl_overview.png)

#### 1.2.2 极简示例

以下代码展示了上述步骤。

```python
# 0. imports
import torch
from transformers import GPT2Tokenizer
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer


# 1. load a pretrained model
model = AutoModelForCausalLMWithValueHead.from_pretrained("gpt2")
ref_model = AutoModelForCausalLMWithValueHead.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# 2. initialize trainer
ppo_config = {"mini_batch_size": 1, "batch_size": 1}
config = PPOConfig(**ppo_config)
ppo_trainer = PPOTrainer(config, model, ref_model, tokenizer)

# 3. encode a query
query_txt = "This morning I went to the "
query_tensor = tokenizer.encode(query_txt, return_tensors="pt").to(model.pretrained_model.device)

# 4. generate model response
generation_kwargs = {
    "min_length": -1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id,
    "max_new_tokens": 20,
}
response_tensor = ppo_trainer.generate([item for item in query_tensor], return_prompt=False, **generation_kwargs)
response_txt = tokenizer.decode(response_tensor[0])

# 5. define a reward for response
# (this could be any reward such as human feedback or output from another model)
reward = [torch.tensor(1.0, device=model.pretrained_model.device)]

# 6. train model with ppo
train_stats = ppo_trainer.step([query_tensor[0]], [response_tensor[0]], reward)
```

通常，你会在一个循环中运行第 3 到第 6 步，并对许多不同的查询执行该操作。你可以在示例部分找到更现实的例子。

#### 1.2.3 如何使用训练好的模型

训练完 `AutoModelForCausalLMWithValueHead` 后，你可以直接在 `transformers` 中使用该模型。

```python

# .. Let's assume we have a trained model using `PPOTrainer` and `AutoModelForCausalLMWithValueHead`

# push the model on the Hub
model.push_to_hub("my-fine-tuned-model-ppo")

# or save it locally
model.save_pretrained("my-fine-tuned-model-ppo")

# load the model from the Hub
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("my-fine-tuned-model-ppo")
```

如果你想使用 value head，例如继续训练，也可以使用 `AutoModelForCausalLMWithValueHead` 加载你的模型。

```python
from trl.model import AutoModelForCausalLMWithValueHead

model = AutoModelForCausalLMWithValueHead.from_pretrained("my-fine-tuned-model-ppo")
```

----------



## 二、内容指导

### 2.1 数据集格式和类型

本指南概述了 TRL 中每个 trainer 所支持的数据集格式和类型。

#### 2.1.1 数据集格式和类型概述

- 数据集的 *格式* 指的是数据的结构方式，通常分为 *标准格式* 或 *对话格式* 。
- *类型* 与数据集设计的具体任务相关联，例如仅提示（*prompt-only*）或偏好（*preference*）。每种类型都以其列的特征来区分，这些列会根据任务的不同而有所变化，如表中所示。

| Type \ Format        | Standard                                                                                                                                                                                                                                                                                                                      | Conversational                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| -------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Language modeling    | `{"text": "The sky is blue."}`                                                                                                                                                                                                                                                                                                | `{"messages": [{"role": "user", "content": "What color is the sky?"},{"role": "assistant", "content": "It is blue."}]}`                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| Prompt-only          | `{"prompt": "The sky is"}`                                                                                                                                                                                                                                                                                                    | `{"prompt": [{"role": "user", "content": "What color is the sky?"}]}`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| Prompt-completion    | `{"prompt": "The sky is", "completion": " blue."}`                                                                                                                                                                                                                                                                            | ```<br>{"prompt": [{"role": "user", "content": "What color is the sky?"}],<br> "completion": [{"role": "assistant", "content": "It is blue."}]}<br>```                                                                                                                                                                                                                                                                                                                                                                                                           |
| Preference           | ```<br>{"prompt": "The sky is",<br> "chosen": " blue.",<br> "rejected": " green."}<br>```<br><br>or, with implicit prompt:<br><br>```<br>{"chosen": "The sky is blue.",<br> "rejected": "The sky is green."}<br>```                                                                                                           | ```<br>{"prompt": [{"role": "user", "content": "What color is the sky?"}],<br> "chosen": [{"role": "assistant", "content": "It is blue."}],<br> "rejected": [{"role": "assistant", "content": "It is green."}]}<br>```<br><br>or, with implicit prompt:<br><br>```<br>{"chosen": [{"role": "user", "content": "What color is the sky?"},<br>              {"role": "assistant", "content": "It is blue."}],<br> "rejected": [{"role": "user", "content": "What color is the sky?"},<br>                {"role": "assistant", "content": "It is green."}]}<br>``` |
| Unpaired preference  | ```<br>{"prompt": "The sky is",<br> "completion": " blue.",<br> "label": True}<br>```                                                                                                                                                                                                                                         | ```<br>{"prompt": [{"role": "user", "content": "What color is the sky?"}],<br> "completion": [{"role": "assistant", "content": "It is green."}],<br> "label": False}<br>```                                                                                                                                                                                                                                                                                                                                                                                      |
| Stepwise supervision | ```<br>{"prompt": "Which number is larger, 9.8 or 9.11?",<br> "completions": ["The fractional part of 9.8 is 0.8.", <br>                 "The fractional part of 9.11 is 0.11.",<br>                 "0.11 is greater than 0.8.",<br>                 "Hence, 9.11 > 9.8."],<br> "labels": [True, True, False, False]}<br>``` |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |

#### 2.1.2 格式

##### 2.1.2.1 标准

标准数据集格式通常由纯文本字符串组成。数据集中的列会根据任务的不同而有所变化。这是 TRL 训练器所期望的格式。以下是不同任务的标准数据集格式示例。

```python
# Language modeling
language_modeling_example = {"text": "The sky is blue."}
# Preference
preference_example = {"prompt": "The sky is", "chosen": " blue.", "rejected": " green."}
# Unpaired preference
unpaired_preference_example = {"prompt": "The sky is", "completion": " blue.", "label": True}
```

##### 2.1.2.2 会话

对话数据集用于涉及用户与助手之间对话或聊天交互的任务。与标准数据集格式不同，这些数据集包含一系列消息，每条消息都有一个角色（例如“用户”或“助手”）和内容（消息文本）。

```python
messages = [
    {"role": "user", "content": "Hello, how are you?"},
    {"role": "assistant", "content": "I'm doing great. How can I help you today?"},
    {"role": "user", "content": "I'd like to show off how chat templating works!"},
]
```

就像标准数据集一样，会话数据集中的列会根据任务的不同而有所变化。以下是不同任务的会话数据集格式示例：

```python
# Prompt-completion
prompt_completion_example = {
	"prompt": [{"role": "user", "content": "What color is the sky?"}],
                "completion": [{"role": "assistant", "content": "It is blue."}]
}
# Preference
preference_example = {
    "prompt": [{"role": "user", "content": "What color is the sky?"}],
    "chosen": [{"role": "assistant", "content": "It is blue."}],
    "rejected": [{"role": "assistant", "content": "It is green."}],
}
```

*对话数据集对于训练聊天模型很有用，但在与 TRL 训练器一起使用之前，必须将其转换为标准格式*。这通常是通过使用特定于所使用模型的聊天模板来完成的。

#### 2.1.3 类型

##### 2.1.3.1 语言模型

一个语言建模数据集包含一个 `text` 列（对于对话数据集为 `messages` 列），其中包含完整的文本序列。

```python
# Standard format
language_modeling_example = {"text": "The sky is blue."}
# Conversational format
language_modeling_example = {"messages": [
    {"role": "user", "content": "What color is the sky?"},
    {"role": "assistant", "content": "It is blue."}
]}
```


#### [](https://huggingface.co/docs/trl/dataset_formats#prompt-only)Prompt-only

In a prompt-only dataset, only the initial prompt (the question or partial sentence) is provided under the key `"prompt"`. The training typically involves generating the completion based on this prompt, where the model learns to continue or complete the given input.

Copied

# Standard format
prompt_only_example = {"prompt": "The sky is"}
# Conversational format
prompt_only_example = {"prompt": [{"role": "user", "content": "What color is the sky?"}]}

For examples of prompt-only datasets, refer to the [Prompt-only datasets collection](https://huggingface.co/collections/trl-lib/prompt-only-datasets-677ea25245d20252cea00368).

While both the prompt-only and language modeling types are similar, they differ in how the input is handled. In the prompt-only type, the prompt represents a partial input that expects the model to complete or continue, while in the language modeling type, the input is treated as a complete sentence or sequence. These two types are processed differently by TRL. Below is an example showing the difference in the output of the `apply_chat_template` function for each type:

Copied

from transformers import AutoTokenizer
from trl import apply_chat_template

tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-128k-instruct")

# Example for prompt-only type
prompt_only_example = {"prompt": [{"role": "user", "content": "What color is the sky?"}]}
apply_chat_template(prompt_only_example, tokenizer)
# Output: {'prompt': '<|user|>\nWhat color is the sky?<|end|>\n<|assistant|>\n'}

# Example for language modeling type
lm_example = {"messages": [{"role": "user", "content": "What color is the sky?"}]}
apply_chat_template(lm_example, tokenizer)
# Output: {'text': '<|user|>\nWhat color is the sky?<|end|>\n<|endoftext|>'}

- The prompt-only output includes a `'<|assistant|>\n'`, indicating the beginning of the assistant’s turn and expecting the model to generate a completion.
- In contrast, the language modeling output treats the input as a complete sequence and terminates it with `'<|endoftext|>'`, signaling the end of the text and not expecting any additional content.

#### [](https://huggingface.co/docs/trl/dataset_formats#prompt-completion)Prompt-completion

A prompt-completion dataset includes a `"prompt"` and a `"completion"`.

Copied

# Standard format
prompt_completion_example = {"prompt": "The sky is", "completion": " blue."}
# Conversational format
prompt_completion_example = {"prompt": [{"role": "user", "content": "What color is the sky?"}],
                             "completion": [{"role": "assistant", "content": "It is blue."}]}

For examples of prompt-completion datasets, refer to the [Prompt-completion datasets collection](https://huggingface.co/collections/trl-lib/prompt-completion-datasets-677ea2bb20bbb6bdccada216).

#### [](https://huggingface.co/docs/trl/dataset_formats#preference)Preference

A preference dataset is used for tasks where the model is trained to choose between two or more possible completions to the same prompt. This dataset includes a `"prompt"`, a `"chosen"` completion, and a `"rejected"` completion. The model is trained to select the `"chosen"` response over the `"rejected"` response. Some dataset may not include the `"prompt"` column, in which case the prompt is implicit and directly included in the `"chosen"` and `"rejected"` completions. We recommend using explicit prompts whenever possible.

Copied

# Standard format
## Explicit prompt (recommended)
preference_example = {"prompt": "The sky is", "chosen": " blue.", "rejected": " green."}
# Implicit prompt
preference_example = {"chosen": "The sky is blue.", "rejected": "The sky is green."}

# Conversational format
## Explicit prompt (recommended)
preference_example = {"prompt": [{"role": "user", "content": "What color is the sky?"}],
                      "chosen": [{"role": "assistant", "content": "It is blue."}],
                      "rejected": [{"role": "assistant", "content": "It is green."}]}
## Implicit prompt
preference_example = {"chosen": [{"role": "user", "content": "What color is the sky?"},
                                 {"role": "assistant", "content": "It is blue."}],
                      "rejected": [{"role": "user", "content": "What color is the sky?"},
                                   {"role": "assistant", "content": "It is green."}]}

For examples of preference datasets, refer to the [Preference datasets collection](https://huggingface.co/collections/trl-lib/preference-datasets-677e99b581018fcad9abd82c).

Some preference datasets can be found with [the tag `dpo` on Hugging Face Hub](https://huggingface.co/datasets?other=dpo). You can also explore the [librarian-bots’ DPO Collections](https://huggingface.co/collections/librarian-bots/direct-preference-optimization-datasets-66964b12835f46289b6ef2fc) to identify preference datasets.

#### [](https://huggingface.co/docs/trl/dataset_formats#unpaired-preference)Unpaired preference

An unpaired preference dataset is similar to a preference dataset but instead of having `"chosen"` and `"rejected"` completions for the same prompt, it includes a single `"completion"` and a `"label"` indicating whether the completion is preferred or not.

Copied

# Standard format
unpaired_preference_example = {"prompt": "The sky is", "completion": " blue.", "label": True}
# Conversational format
unpaired_preference_example = {"prompt": [{"role": "user", "content": "What color is the sky?"}],
                               "completion": [{"role": "assistant", "content": "It is blue."}],
                               "label": True}

For examples of unpaired preference datasets, refer to the [Unpaired preference datasets collection](https://huggingface.co/collections/trl-lib/unpaired-preference-datasets-677ea22bf5f528c125b0bcdf).

#### [](https://huggingface.co/docs/trl/dataset_formats#stepwise-supervision)Stepwise supervision

A stepwise (or process) supervision dataset is similar to an [unpaired preference](https://huggingface.co/docs/trl/dataset_formats#unpaired-preference) dataset but includes multiple steps of completions, each with its own label. This structure is useful for tasks that need detailed, step-by-step labeling, such as reasoning tasks. By evaluating each step separately and providing targeted labels, this approach helps identify precisely where the reasoning is correct and where errors occur, allowing for targeted feedback on each part of the reasoning process.

Copied

stepwise_example = {
    "prompt": "Which number is larger, 9.8 or 9.11?",
    "completions": ["The fractional part of 9.8 is 0.8, while the fractional part of 9.11 is 0.11.", "Since 0.11 is greater than 0.8, the number 9.11 is larger than 9.8."],
    "labels": [True, False]
}

For examples of stepwise supervision datasets, refer to the [Stepwise supervision datasets collection](https://huggingface.co/collections/trl-lib/stepwise-supervision-datasets-677ea27fd4c5941beed7a96e).

## [](https://huggingface.co/docs/trl/dataset_formats#which-dataset-type-to-use)Which dataset type to use?

Choosing the right dataset type depends on the task you are working on and the specific requirements of the TRL trainer you are using. Below is a brief overview of the dataset types supported by each TRL trainer.

|Trainer|Expected dataset type|
|---|---|
|[BCOTrainer](https://huggingface.co/docs/trl/v0.17.0/en/bco_trainer#trl.BCOTrainer)|[Unpaired preference](https://huggingface.co/docs/trl/dataset_formats#unpaired-preference)|
|[CPOTrainer](https://huggingface.co/docs/trl/v0.17.0/en/cpo_trainer#trl.CPOTrainer)|[Preference (explicit prompt recommended)](https://huggingface.co/docs/trl/dataset_formats#preference)|
|[DPOTrainer](https://huggingface.co/docs/trl/v0.17.0/en/dpo_trainer#trl.DPOTrainer)|[Preference (explicit prompt recommended)](https://huggingface.co/docs/trl/dataset_formats#preference)|
|[GKDTrainer](https://huggingface.co/docs/trl/v0.17.0/en/gkd_trainer#trl.GKDTrainer)|[Prompt-completion](https://huggingface.co/docs/trl/dataset_formats#prompt-completion)|
|[GRPOTrainer](https://huggingface.co/docs/trl/v0.17.0/en/grpo_trainer#trl.GRPOTrainer)|[Prompt-only](https://huggingface.co/docs/trl/dataset_formats#prompt-only)|
|[IterativeSFTTrainer](https://huggingface.co/docs/trl/v0.17.0/en/iterative_sft_trainer#trl.IterativeSFTTrainer)|[Unpaired preference](https://huggingface.co/docs/trl/dataset_formats#unpaired-preference)|
|[KTOTrainer](https://huggingface.co/docs/trl/v0.17.0/en/kto_trainer#trl.KTOTrainer)|[Unpaired preference](https://huggingface.co/docs/trl/dataset_formats#unpaired-preference) or [Preference (explicit prompt recommended)](https://huggingface.co/docs/trl/dataset_formats#preference)|
|[NashMDTrainer](https://huggingface.co/docs/trl/v0.17.0/en/nash_md_trainer#trl.NashMDTrainer)|[Prompt-only](https://huggingface.co/docs/trl/dataset_formats#prompt-only)|
|[OnlineDPOTrainer](https://huggingface.co/docs/trl/v0.17.0/en/online_dpo_trainer#trl.OnlineDPOTrainer)|[Prompt-only](https://huggingface.co/docs/trl/dataset_formats#prompt-only)|
|[ORPOTrainer](https://huggingface.co/docs/trl/v0.17.0/en/orpo_trainer#trl.ORPOTrainer)|[Preference (explicit prompt recommended)](https://huggingface.co/docs/trl/dataset_formats#preference)|
|[PPOTrainer](https://huggingface.co/docs/trl/v0.17.0/en/ppo_trainer#trl.PPOTrainer)|Tokenized language modeling|
|[PRMTrainer](https://huggingface.co/docs/trl/v0.17.0/en/prm_trainer#trl.PRMTrainer)|[Stepwise supervision](https://huggingface.co/docs/trl/dataset_formats#stepwise-supervision)|
|[RewardTrainer](https://huggingface.co/docs/trl/v0.17.0/en/reward_trainer#trl.RewardTrainer)|[Preference (implicit prompt recommended)](https://huggingface.co/docs/trl/dataset_formats#preference)|
|[SFTTrainer](https://huggingface.co/docs/trl/v0.17.0/en/sft_trainer#trl.SFTTrainer)|[Language modeling](https://huggingface.co/docs/trl/dataset_formats#language-modeling) or [Prompt-completion](https://huggingface.co/docs/trl/dataset_formats#prompt-completion)|
|[XPOTrainer](https://huggingface.co/docs/trl/v0.17.0/en/xpo_trainer#trl.XPOTrainer)|[Prompt-only](https://huggingface.co/docs/trl/dataset_formats#prompt-only)|

TRL trainers only support standard dataset formats, [for now](https://github.com/huggingface/trl/issues/2071). If you have a conversational dataset, you must first convert it into a standard format. For more information on how to work with conversational datasets, refer to the [Working with conversational datasets in TRL](https://huggingface.co/docs/trl/dataset_formats#working-with-conversational-datasets-in-trl) section.

## [](https://huggingface.co/docs/trl/dataset_formats#working-with-conversational-datasets-in-trl)Working with conversational datasets in TRL

Conversational datasets are increasingly common, especially for training chat models. However, some TRL trainers don’t support conversational datasets in their raw format. (For more information, see [issue #2071](https://github.com/huggingface/trl/issues/2071).) These datasets must first be converted into a standard format. Fortunately, TRL offers tools to easily handle this conversion, which are detailed below.

### [](https://huggingface.co/docs/trl/dataset_formats#converting-a-conversational-dataset-into-a-standard-dataset)Converting a conversational dataset into a standard dataset

To convert a conversational dataset into a standard dataset, you need to _apply a chat template_ to the dataset. A chat template is a predefined structure that typically includes placeholders for user and assistant messages. This template is provided by the tokenizer of the model you use.

For detailed instructions on using chat templating, refer to the [Chat templating section in the `transformers` documentation](https://huggingface.co/docs/transformers/en/chat_templating).

In TRL, the method you apply to convert the dataset will vary depending on the task. Fortunately, TRL provides a helper function called [apply_chat_template()](https://huggingface.co/docs/trl/v0.17.0/en/data_utils#trl.apply_chat_template) to simplify this process. Here’s an example of how to use it:

Copied

from transformers import AutoTokenizer
from trl import apply_chat_template

tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-128k-instruct")

example = {
    "prompt": [{"role": "user", "content": "What color is the sky?"}],
    "completion": [{"role": "assistant", "content": "It is blue."}]
}

apply_chat_template(example, tokenizer)
# Output:
# {'prompt': '<|user|>\nWhat color is the sky?<|end|>\n<|assistant|>\n', 'completion': 'It is blue.<|end|>\n<|endoftext|>'}

Alternatively, you can use the [map](https://huggingface.co/docs/datasets/v3.5.0/en/package_reference/main_classes#datasets.Dataset.map) method to apply the template across an entire dataset:

Copied

from datasets import Dataset
from trl import apply_chat_template

dataset_dict = {
    "prompt": [[{"role": "user", "content": "What color is the sky?"}],
               [{"role": "user", "content": "Where is the sun?"}]],
    "completion": [[{"role": "assistant", "content": "It is blue."}],
                   [{"role": "assistant", "content": "In the sky."}]]
}

dataset = Dataset.from_dict(dataset_dict)
dataset = dataset.map(apply_chat_template, fn_kwargs={"tokenizer": tokenizer})
# Output:
# {'prompt': ['<|user|>\nWhat color is the sky?<|end|>\n<|assistant|>\n',
#             '<|user|>\nWhere is the sun?<|end|>\n<|assistant|>\n'],
#  'completion': ['It is blue.<|end|>\n<|endoftext|>', 'In the sky.<|end|>\n<|endoftext|>']}

We recommend using the [apply_chat_template()](https://huggingface.co/docs/trl/v0.17.0/en/data_utils#trl.apply_chat_template) function instead of calling `tokenizer.apply_chat_template` directly. Handling chat templates for non-language modeling datasets can be tricky and may result in errors, such as mistakenly placing a system prompt in the middle of a conversation. For additional examples, see [#1930 (comment)](https://github.com/huggingface/trl/pull/1930#issuecomment-2292908614). The [apply_chat_template()](https://huggingface.co/docs/trl/v0.17.0/en/data_utils#trl.apply_chat_template) is designed to handle these intricacies and ensure the correct application of chat templates for various tasks.

It’s important to note that chat templates are model-specific. For example, if you use the chat template from [meta-llama/Meta-Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct) with the above example, you get a different output:

Copied

apply_chat_template(example, AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct"))
# Output:
# {'prompt': '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nWhat color is the sky?<|im_end|>\n<|im_start|>assistant\n',
#  'completion': 'It is blue.<|im_end|>\n'}

Always use the chat template associated with the model you’re working with. Using the wrong template can lead to inaccurate or unexpected results.

## [](https://huggingface.co/docs/trl/dataset_formats#using-any-dataset-with-trl-preprocessing-and-conversion)Using any dataset with TRL: preprocessing and conversion

Many datasets come in formats tailored to specific tasks, which might not be directly compatible with TRL. To use such datasets with TRL, you may need to preprocess and convert them into the required format.

To make this easier, we provide a set of [example scripts](https://github.com/huggingface/trl/tree/main/examples/datasets) that cover common dataset conversions.

### [](https://huggingface.co/docs/trl/dataset_formats#example-ultrafeedback-dataset)Example: UltraFeedback dataset

Let’s take the [UltraFeedback dataset](https://huggingface.co/datasets/openbmb/UltraFeedback) as an example. Here’s a preview of the dataset:

As shown above, the dataset format does not match the expected structure. It’s not in a conversational format, the column names differ, and the results pertain to different models (e.g., Bard, GPT-4) and aspects (e.g., “helpfulness”, “honesty”).

By using the provided conversion script [`examples/datasets/ultrafeedback.py`](https://github.com/huggingface/trl/tree/main/examples/datasets/ultrafeedback.py), you can transform this dataset into an unpaired preference type, and push it to the Hub:

Copied

python examples/datasets/ultrafeedback.py --push_to_hub --repo_id trl-lib/ultrafeedback-gpt-3.5-turbo-helpfulness

Once converted, the dataset will look like this:

Now, you can use this dataset with TRL!

By adapting the provided scripts or creating your own, you can convert any dataset into a format compatible with TRL.

## [](https://huggingface.co/docs/trl/dataset_formats#utilities-for-converting-dataset-types)Utilities for converting dataset types

This section provides example code to help you convert between different dataset types. While some conversions can be performed after applying the chat template (i.e., in the standard format), we recommend performing the conversion before applying the chat template to ensure it works consistently.

For simplicity, some of the examples below do not follow this recommendation and use the standard format. However, the conversions can be applied directly to the conversational format without modification.

|From \ To|Language modeling|Prompt-completion|Prompt-only|Preference with implicit prompt|Preference|Unpaired preference|Stepwise supervision|
|---|---|---|---|---|---|---|---|
|Language modeling|N/A|N/A|N/A|N/A|N/A|N/A|N/A|
|Prompt-completion|[🔗](https://huggingface.co/docs/trl/dataset_formats#from-prompt-completion-to-language-modeling-dataset)|N/A|[🔗](https://huggingface.co/docs/trl/dataset_formats#from-prompt-completion-to-prompt-only-dataset)|N/A|N/A|N/A|N/A|
|Prompt-only|N/A|N/A|N/A|N/A|N/A|N/A|N/A|
|Preference with implicit prompt|[🔗](https://huggingface.co/docs/trl/dataset_formats#from-preference-with-implicit-prompt-to-language-modeling-dataset)|[🔗](https://huggingface.co/docs/trl/dataset_formats#from-preference-with-implicit-prompt-to-prompt-completion-dataset)|[🔗](https://huggingface.co/docs/trl/dataset_formats#from-preference-with-implicit-prompt-to-prompt-only-dataset)|N/A|[🔗](https://huggingface.co/docs/trl/dataset_formats#from-implicit-to-explicit-prompt-preference-dataset)|[🔗](https://huggingface.co/docs/trl/dataset_formats#from-preference-with-implicit-prompt-to-unpaired-preference-dataset)|N/A|
|Preference|[🔗](https://huggingface.co/docs/trl/dataset_formats#from-preference-to-language-modeling-dataset)|[🔗](https://huggingface.co/docs/trl/dataset_formats#from-preference-to-prompt-completion-dataset)|[🔗](https://huggingface.co/docs/trl/dataset_formats#from-preference-to-prompt-only-dataset)|[🔗](https://huggingface.co/docs/trl/dataset_formats#from-explicit-to-implicit-prompt-preference-dataset)|N/A|[🔗](https://huggingface.co/docs/trl/dataset_formats#from-preference-to-unpaired-preference-dataset)|N/A|
|Unpaired preference|[🔗](https://huggingface.co/docs/trl/dataset_formats#from-unpaired-preference-to-language-modeling-dataset)|[🔗](https://huggingface.co/docs/trl/dataset_formats#from-unpaired-preference-to-prompt-completion-dataset)|[🔗](https://huggingface.co/docs/trl/dataset_formats#from-unpaired-preference-to-prompt-only-dataset)|N/A|N/A|N/A|N/A|
|Stepwise supervision|[🔗](https://huggingface.co/docs/trl/dataset_formats#from-stepwise-supervision-to-language-modeling-dataset)|[🔗](https://huggingface.co/docs/trl/dataset_formats#from-stepwise-supervision-to-prompt-completion-dataset)|[🔗](https://huggingface.co/docs/trl/dataset_formats#from-stepwise-supervision-to-prompt-only-dataset)|N/A|N/A|[🔗](https://huggingface.co/docs/trl/dataset_formats#from-stepwise-supervision-to-unpaired-preference-dataset)|N/A|

### [](https://huggingface.co/docs/trl/dataset_formats#from-prompt-completion-to-language-modeling-dataset)From prompt-completion to language modeling dataset

To convert a prompt-completion dataset into a language modeling dataset, concatenate the prompt and the completion.

Copied

from datasets import Dataset

dataset = Dataset.from_dict({
    "prompt": ["The sky is", "The sun is"],
    "completion": [" blue.", " in the sky."],
})

def concat_prompt_completion(example):
    return {"text": example["prompt"] + example["completion"]}

dataset = dataset.map(concat_prompt_completion, remove_columns=["prompt", "completion"])

Copied

>>> dataset[0]
{'text': 'The sky is blue.'}

### [](https://huggingface.co/docs/trl/dataset_formats#from-prompt-completion-to-prompt-only-dataset)From prompt-completion to prompt-only dataset

To convert a prompt-completion dataset into a prompt-only dataset, remove the completion.

Copied

from datasets import Dataset

dataset = Dataset.from_dict({
    "prompt": ["The sky is", "The sun is"],
    "completion": [" blue.", " in the sky."],
})

dataset = dataset.remove_columns("completion")

Copied

>>> dataset[0]
{'prompt': 'The sky is'}

### [](https://huggingface.co/docs/trl/dataset_formats#from-preference-with-implicit-prompt-to-language-modeling-dataset)From preference with implicit prompt to language modeling dataset

To convert a preference with implicit prompt dataset into a language modeling dataset, remove the rejected, and rename the column `"chosen"` to `"text"`.

Copied

from datasets import Dataset

dataset = Dataset.from_dict({
    "chosen": ["The sky is blue.", "The sun is in the sky."],
    "rejected": ["The sky is green.", "The sun is in the sea."],
})

dataset = dataset.rename_column("chosen", "text").remove_columns("rejected")

Copied

>>> dataset[0]
{'text': 'The sky is blue.'}

### [](https://huggingface.co/docs/trl/dataset_formats#from-preference-with-implicit-prompt-to-prompt-completion-dataset)From preference with implicit prompt to prompt-completion dataset

To convert a preference dataset with implicit prompt into a prompt-completion dataset, extract the prompt with [extract_prompt()](https://huggingface.co/docs/trl/v0.17.0/en/data_utils#trl.extract_prompt), remove the rejected, and rename the column `"chosen"` to `"completion"`.

Copied

from datasets import Dataset
from trl import extract_prompt

dataset = Dataset.from_dict({
    "chosen": [
        [{"role": "user", "content": "What color is the sky?"}, {"role": "assistant", "content": "It is blue."}],
        [{"role": "user", "content": "Where is the sun?"}, {"role": "assistant", "content": "In the sky."}],
    ],
    "rejected": [
        [{"role": "user", "content": "What color is the sky?"}, {"role": "assistant", "content": "It is green."}],
        [{"role": "user", "content": "Where is the sun?"}, {"role": "assistant", "content": "In the sea."}],
    ],
})
dataset = dataset.map(extract_prompt).remove_columns("rejected").rename_column("chosen", "completion")

Copied

>>> dataset[0]
{'prompt': [{'role': 'user', 'content': 'What color is the sky?'}], 'completion': [{'role': 'assistant', 'content': 'It is blue.'}]}

### [](https://huggingface.co/docs/trl/dataset_formats#from-preference-with-implicit-prompt-to-prompt-only-dataset)From preference with implicit prompt to prompt-only dataset

To convert a preference dataset with implicit prompt into a prompt-only dataset, extract the prompt with [extract_prompt()](https://huggingface.co/docs/trl/v0.17.0/en/data_utils#trl.extract_prompt), and remove the rejected and the chosen.

Copied

from datasets import Dataset
from trl import extract_prompt

dataset = Dataset.from_dict({
    "chosen": [
        [{"role": "user", "content": "What color is the sky?"}, {"role": "assistant", "content": "It is blue."}],
        [{"role": "user", "content": "Where is the sun?"}, {"role": "assistant", "content": "In the sky."}],
    ],
    "rejected": [
        [{"role": "user", "content": "What color is the sky?"}, {"role": "assistant", "content": "It is green."}],
        [{"role": "user", "content": "Where is the sun?"}, {"role": "assistant", "content": "In the sea."}],
    ],
})
dataset = dataset.map(extract_prompt).remove_columns(["chosen", "rejected"])

Copied

>>> dataset[0]
{'prompt': [{'role': 'user', 'content': 'What color is the sky?'}]}

### [](https://huggingface.co/docs/trl/dataset_formats#from-implicit-to-explicit-prompt-preference-dataset)From implicit to explicit prompt preference dataset

To convert a preference dataset with implicit prompt into a preference dataset with explicit prompt, extract the prompt with [extract_prompt()](https://huggingface.co/docs/trl/v0.17.0/en/data_utils#trl.extract_prompt).

Copied

from datasets import Dataset
from trl import extract_prompt

dataset = Dataset.from_dict({
    "chosen": [
        [{"role": "user", "content": "What color is the sky?"}, {"role": "assistant", "content": "It is blue."}],
        [{"role": "user", "content": "Where is the sun?"}, {"role": "assistant", "content": "In the sky."}],
    ],
    "rejected": [
        [{"role": "user", "content": "What color is the sky?"}, {"role": "assistant", "content": "It is green."}],
        [{"role": "user", "content": "Where is the sun?"}, {"role": "assistant", "content": "In the sea."}],
    ],
})

dataset = dataset.map(extract_prompt)

Copied

>>> dataset[0]
{'prompt': [{'role': 'user', 'content': 'What color is the sky?'}],
 'chosen': [{'role': 'assistant', 'content': 'It is blue.'}],
 'rejected': [{'role': 'assistant', 'content': 'It is green.'}]}

### [](https://huggingface.co/docs/trl/dataset_formats#from-preference-with-implicit-prompt-to-unpaired-preference-dataset)From preference with implicit prompt to unpaired preference dataset

To convert a preference dataset with implicit prompt into an unpaired preference dataset, extract the prompt with [extract_prompt()](https://huggingface.co/docs/trl/v0.17.0/en/data_utils#trl.extract_prompt), and unpair the dataset with [unpair_preference_dataset()](https://huggingface.co/docs/trl/v0.17.0/en/data_utils#trl.unpair_preference_dataset).

Copied

from datasets import Dataset
from trl import extract_prompt, unpair_preference_dataset

dataset = Dataset.from_dict({
    "chosen": [
        [{"role": "user", "content": "What color is the sky?"}, {"role": "assistant", "content": "It is blue."}],
        [{"role": "user", "content": "Where is the sun?"}, {"role": "assistant", "content": "In the sky."}],
    ],
    "rejected": [
        [{"role": "user", "content": "What color is the sky?"}, {"role": "assistant", "content": "It is green."}],
        [{"role": "user", "content": "Where is the sun?"}, {"role": "assistant", "content": "In the sea."}],
    ],
})

dataset = dataset.map(extract_prompt)
dataset = unpair_preference_dataset(dataset)

Copied

>>> dataset[0]
{'prompt': [{'role': 'user', 'content': 'What color is the sky?'}],
 'completion': [{'role': 'assistant', 'content': 'It is blue.'}],
 'label': True}

Keep in mind that the `"chosen"` and `"rejected"` completions in a preference dataset can be both good or bad. Before applying [unpair_preference_dataset()](https://huggingface.co/docs/trl/v0.17.0/en/data_utils#trl.unpair_preference_dataset), please ensure that all `"chosen"` completions can be labeled as good and all `"rejected"` completions as bad. This can be ensured by checking absolute rating of each completion, e.g. from a reward model.

### [](https://huggingface.co/docs/trl/dataset_formats#from-preference-to-language-modeling-dataset)From preference to language modeling dataset

To convert a preference dataset into a language modeling dataset, remove the rejected, concatenate the prompt and the chosen into the `"text"` column.

Copied

from datasets import Dataset

dataset = Dataset.from_dict({
    "prompt": ["The sky is", "The sun is"],
    "chosen": [" blue.", " in the sky."],
    "rejected": [" green.", " in the sea."],
})

def concat_prompt_chosen(example):
    return {"text": example["prompt"] + example["chosen"]}

dataset = dataset.map(concat_prompt_chosen, remove_columns=["prompt", "chosen", "rejected"])

Copied

>>> dataset[0]
{'text': 'The sky is blue.'}

### [](https://huggingface.co/docs/trl/dataset_formats#from-preference-to-prompt-completion-dataset)From preference to prompt-completion dataset

To convert a preference dataset into a prompt-completion dataset, remove the rejected, and rename the column `"chosen"` to `"completion"`.

Copied

from datasets import Dataset

dataset = Dataset.from_dict({
    "prompt": ["The sky is", "The sun is"],
    "chosen": [" blue.", " in the sky."],
    "rejected": [" green.", " in the sea."],
})

dataset = dataset.remove_columns("rejected").rename_column("chosen", "completion")

Copied

>>> dataset[0]
{'prompt': 'The sky is', 'completion': ' blue.'}

### [](https://huggingface.co/docs/trl/dataset_formats#from-preference-to-prompt-only-dataset)From preference to prompt-only dataset

To convert a preference dataset into a prompt-only dataset, remove the rejected and the chosen.

Copied

from datasets import Dataset

dataset = Dataset.from_dict({
    "prompt": ["The sky is", "The sun is"],
    "chosen": [" blue.", " in the sky."],
    "rejected": [" green.", " in the sea."],
})

dataset = dataset.remove_columns(["chosen", "rejected"])

Copied

>>> dataset[0]
{'prompt': 'The sky is'}

### [](https://huggingface.co/docs/trl/dataset_formats#from-explicit-to-implicit-prompt-preference-dataset)From explicit to implicit prompt preference dataset

To convert a preference dataset with explicit prompt into a preference dataset with implicit prompt, concatenate the prompt to both chosen and rejected, and remove the prompt.

Copied

from datasets import Dataset

dataset = Dataset.from_dict({
    "prompt": [
        [{"role": "user", "content": "What color is the sky?"}],
        [{"role": "user", "content": "Where is the sun?"}],
    ],
    "chosen": [
        [{"role": "assistant", "content": "It is blue."}],
        [{"role": "assistant", "content": "In the sky."}],
    ],
    "rejected": [
        [{"role": "assistant", "content": "It is green."}],
        [{"role": "assistant", "content": "In the sea."}],
    ],
})

def concat_prompt_to_completions(example):
    return {"chosen": example["prompt"] + example["chosen"], "rejected": example["prompt"] + example["rejected"]}

dataset = dataset.map(concat_prompt_to_completions, remove_columns="prompt")

Copied

>>> dataset[0]
{'chosen': [{'role': 'user', 'content': 'What color is the sky?'}, {'role': 'assistant', 'content': 'It is blue.'}],
 'rejected': [{'role': 'user', 'content': 'What color is the sky?'}, {'role': 'assistant', 'content': 'It is green.'}]}

### [](https://huggingface.co/docs/trl/dataset_formats#from-preference-to-unpaired-preference-dataset)From preference to unpaired preference dataset

To convert dataset into an unpaired preference dataset, unpair the dataset with [unpair_preference_dataset()](https://huggingface.co/docs/trl/v0.17.0/en/data_utils#trl.unpair_preference_dataset).

Copied

from datasets import Dataset
from trl import unpair_preference_dataset

dataset = Dataset.from_dict({
    "prompt": [
        [{"role": "user", "content": "What color is the sky?"}],
        [{"role": "user", "content": "Where is the sun?"}],
    ],
    "chosen": [
        [{"role": "assistant", "content": "It is blue."}],
        [{"role": "assistant", "content": "In the sky."}],
    ],
    "rejected": [
        [{"role": "assistant", "content": "It is green."}],
        [{"role": "assistant", "content": "In the sea."}],
    ],
})

dataset = unpair_preference_dataset(dataset)

Copied

>>> dataset[0]
{'prompt': [{'role': 'user', 'content': 'What color is the sky?'}],
 'completion': [{'role': 'assistant', 'content': 'It is blue.'}],
 'label': True}

Keep in mind that the `"chosen"` and `"rejected"` completions in a preference dataset can be both good or bad. Before applying [unpair_preference_dataset()](https://huggingface.co/docs/trl/v0.17.0/en/data_utils#trl.unpair_preference_dataset), please ensure that all `"chosen"` completions can be labeled as good and all `"rejected"` completions as bad. This can be ensured by checking absolute rating of each completion, e.g. from a reward model.

### [](https://huggingface.co/docs/trl/dataset_formats#from-unpaired-preference-to-language-modeling-dataset)From unpaired preference to language modeling dataset

To convert an unpaired preference dataset into a language modeling dataset, concatenate prompts with good completions into the `"text"` column, and remove the prompt, completion and label columns.

Copied

from datasets import Dataset

dataset = Dataset.from_dict({
    "prompt": ["The sky is", "The sun is", "The sky is", "The sun is"],
    "completion": [" blue.", " in the sky.", " green.", " in the sea."],
    "label": [True, True, False, False],
})

def concatenate_prompt_completion(example):
    return {"text": example["prompt"] + example["completion"]}

dataset = dataset.filter(lambda x: x["label"]).map(concatenate_prompt_completion).remove_columns(["prompt", "completion", "label"])

Copied

>>> dataset[0]
{'text': 'The sky is blue.'}

### [](https://huggingface.co/docs/trl/dataset_formats#from-unpaired-preference-to-prompt-completion-dataset)From unpaired preference to prompt-completion dataset

To convert an unpaired preference dataset into a prompt-completion dataset, filter for good labels, then remove the label columns.

Copied

from datasets import Dataset

dataset = Dataset.from_dict({
    "prompt": ["The sky is", "The sun is", "The sky is", "The sun is"],
    "completion": [" blue.", " in the sky.", " green.", " in the sea."],
    "label": [True, True, False, False],
})

dataset = dataset.filter(lambda x: x["label"]).remove_columns(["label"])

Copied

>>> dataset[0]
{'prompt': 'The sky is', 'completion': ' blue.'}

### [](https://huggingface.co/docs/trl/dataset_formats#from-unpaired-preference-to-prompt-only-dataset)From unpaired preference to prompt-only dataset

To convert an unpaired preference dataset into a prompt-only dataset, remove the completion and the label columns.

Copied

from datasets import Dataset

dataset = Dataset.from_dict({
    "prompt": ["The sky is", "The sun is", "The sky is", "The sun is"],
    "completion": [" blue.", " in the sky.", " green.", " in the sea."],
    "label": [True, True, False, False],
})

dataset = dataset.remove_columns(["completion", "label"])

Copied

>>> dataset[0]
{'prompt': 'The sky is'}

### [](https://huggingface.co/docs/trl/dataset_formats#from-stepwise-supervision-to-language-modeling-dataset)From stepwise supervision to language modeling dataset

To convert a stepwise supervision dataset into a language modeling dataset, concatenate prompts with good completions into the `"text"` column.

Copied

from datasets import Dataset

dataset = Dataset.from_dict({
    "prompt": ["Blue light", "Water"],
    "completions": [[" scatters more in the atmosphere,", " so the sky is green."],
                   [" forms a less dense structure in ice,", " which causes it to expand when it freezes."]],
    "labels": [[True, False], [True, True]],
})

def concatenate_prompt_completions(example):
    completion = "".join(example["completions"])
    return {"text": example["prompt"] + completion}

dataset = dataset.filter(lambda x: all(x["labels"])).map(concatenate_prompt_completions, remove_columns=["prompt", "completions", "labels"])

Copied

>>> dataset[0]
{'text': 'Blue light scatters more in the atmosphere, so the sky is green.'}

### [](https://huggingface.co/docs/trl/dataset_formats#from-stepwise-supervision-to-prompt-completion-dataset)From stepwise supervision to prompt completion dataset

To convert a stepwise supervision dataset into a prompt-completion dataset, join the good completions and remove the labels.

Copied

from datasets import Dataset

dataset = Dataset.from_dict({
    "prompt": ["Blue light", "Water"],
    "completions": [[" scatters more in the atmosphere,", " so the sky is green."],
                   [" forms a less dense structure in ice,", " which causes it to expand when it freezes."]],
    "labels": [[True, False], [True, True]],
})

def join_completions(example):
    completion = "".join(example["completions"])
    return {"completion": completion}

dataset = dataset.filter(lambda x: all(x["labels"])).map(join_completions, remove_columns=["completions", "labels"])

Copied

>>> dataset[0]
{'prompt': 'Blue light', 'completion': ' scatters more in the atmosphere, so the sky is green.'}

### [](https://huggingface.co/docs/trl/dataset_formats#from-stepwise-supervision-to-prompt-only-dataset)From stepwise supervision to prompt only dataset

To convert a stepwise supervision dataset into a prompt-only dataset, remove the completions and the labels.

Copied

from datasets import Dataset

dataset = Dataset.from_dict({
    "prompt": ["Blue light", "Water"],
    "completions": [[" scatters more in the atmosphere,", " so the sky is green."],
                   [" forms a less dense structure in ice,", " which causes it to expand when it freezes."]],
    "labels": [[True, False], [True, True]],
})

dataset = dataset.remove_columns(["completions", "labels"])

Copied

>>> dataset[0]
{'prompt': 'Blue light'}

### [](https://huggingface.co/docs/trl/dataset_formats#from-stepwise-supervision-to-unpaired-preference-dataset)From stepwise supervision to unpaired preference dataset

To convert a stepwise supervision dataset into an unpaired preference dataset, join the completions and merge the labels.

The method for merging the labels depends on the specific task. In this example, we use the logical AND operation. This means that if the step labels indicate the correctness of individual steps, the resulting label will reflect the correctness of the entire sequence.

Copied

from datasets import Dataset

dataset = Dataset.from_dict({
    "prompt": ["Blue light", "Water"],
    "completions": [[" scatters more in the atmosphere,", " so the sky is green."],
                   [" forms a less dense structure in ice,", " which causes it to expand when it freezes."]],
    "labels": [[True, False], [True, True]],
})

def merge_completions_and_labels(example):
    return {"prompt": example["prompt"], "completion": "".join(example["completions"]), "label": all(example["labels"])}

dataset = dataset.map(merge_completions_and_labels, remove_columns=["completions", "labels"])

Copied

>>> dataset[0]
{'prompt': 'Blue light', 'completion': ' scatters more in the atmosphere, so the sky is green.', 'label': False}

## [](https://huggingface.co/docs/trl/dataset_formats#vision-datasets)Vision datasets

Some trainers also support fine-tuning vision-language models (VLMs) using image-text pairs. In this scenario, it’s recommended to use a conversational format, as each model handles image placeholders in text differently.

A conversational vision dataset differs from a standard conversational dataset in two key ways:

1. The dataset must contain the key `images` with the image data.
2. The `"content"` field in messages must be a list of dictionaries, where each dictionary specifies the type of data: `"image"` or `"text"`.

Example:

Copied

# Textual dataset:
"content": "What color is the sky?"

# Vision dataset:
"content": [
    {"type": "image"}, 
    {"type": "text", "text": "What color is the sky in the image?"}
]

An example of a conversational vision dataset is the [openbmb/RLAIF-V-Dataset](https://huggingface.co/datasets/openbmb/RLAIF-V-Dataset). Below is an embedded view of the dataset’s training data, allowing you to explore it directly:

[<>Update on GitHub](https://github.com/huggingface/trl/blob/main/docs/source/dataset_formats.md)

Dataset formats and types

[←Quickstart](https://huggingface.co/docs/trl/quickstart)[Training FAQ→](https://huggingface.co/docs/trl/how_to_train)

[Dataset formats and types](https://huggingface.co/docs/trl/dataset_formats#dataset-formats-and-types)[Overview of the dataset formats and types](https://huggingface.co/docs/trl/dataset_formats#overview-of-the-dataset-formats-and-types)[Formats](https://huggingface.co/docs/trl/dataset_formats#formats)[Standard](https://huggingface.co/docs/trl/dataset_formats#standard)[Conversational](https://huggingface.co/docs/trl/dataset_formats#conversational)[Types](https://huggingface.co/docs/trl/dataset_formats#types)[Language modeling](https://huggingface.co/docs/trl/dataset_formats#language-modeling)[Prompt-only](https://huggingface.co/docs/trl/dataset_formats#prompt-only)[Prompt-completion](https://huggingface.co/docs/trl/dataset_formats#prompt-completion)[Preference](https://huggingface.co/docs/trl/dataset_formats#preference)[Unpaired preference](https://huggingface.co/docs/trl/dataset_formats#unpaired-preference)[Stepwise supervision](https://huggingface.co/docs/trl/dataset_formats#stepwise-supervision)[Which dataset type to use?](https://huggingface.co/docs/trl/dataset_formats#which-dataset-type-to-use)[Working with conversational datasets in TRL](https://huggingface.co/docs/trl/dataset_formats#working-with-conversational-datasets-in-trl)[Converting a conversational dataset into a standard dataset](https://huggingface.co/docs/trl/dataset_formats#converting-a-conversational-dataset-into-a-standard-dataset)[Using any dataset with TRL: preprocessing and conversion](https://huggingface.co/docs/trl/dataset_formats#using-any-dataset-with-trl-preprocessing-and-conversion)[Example: UltraFeedback dataset](https://huggingface.co/docs/trl/dataset_formats#example-ultrafeedback-dataset)[Utilities for converting dataset types](https://huggingface.co/docs/trl/dataset_formats#utilities-for-converting-dataset-types)[From prompt-completion to language modeling dataset](https://huggingface.co/docs/trl/dataset_formats#from-prompt-completion-to-language-modeling-dataset)[From prompt-completion to prompt-only dataset](https://huggingface.co/docs/trl/dataset_formats#from-prompt-completion-to-prompt-only-dataset)[From preference with implicit prompt to language modeling dataset](https://huggingface.co/docs/trl/dataset_formats#from-preference-with-implicit-prompt-to-language-modeling-dataset)[From preference with implicit prompt to prompt-completion dataset](https://huggingface.co/docs/trl/dataset_formats#from-preference-with-implicit-prompt-to-prompt-completion-dataset)[From preference with implicit prompt to prompt-only dataset](https://huggingface.co/docs/trl/dataset_formats#from-preference-with-implicit-prompt-to-prompt-only-dataset)[From implicit to explicit prompt preference dataset](https://huggingface.co/docs/trl/dataset_formats#from-implicit-to-explicit-prompt-preference-dataset)[From preference with implicit prompt to unpaired preference dataset](https://huggingface.co/docs/trl/dataset_formats#from-preference-with-implicit-prompt-to-unpaired-preference-dataset)[From preference to language modeling dataset](https://huggingface.co/docs/trl/dataset_formats#from-preference-to-language-modeling-dataset)[From preference to prompt-completion dataset](https://huggingface.co/docs/trl/dataset_formats#from-preference-to-prompt-completion-dataset)[From preference to prompt-only dataset](https://huggingface.co/docs/trl/dataset_formats#from-preference-to-prompt-only-dataset)[From explicit to implicit prompt preference dataset](https://huggingface.co/docs/trl/dataset_formats#from-explicit-to-implicit-prompt-preference-dataset)[From preference to unpaired preference dataset](https://huggingface.co/docs/trl/dataset_formats#from-preference-to-unpaired-preference-dataset)[From unpaired preference to language modeling dataset](https://huggingface.co/docs/trl/dataset_formats#from-unpaired-preference-to-language-modeling-dataset)[From unpaired preference to prompt-completion dataset](https://huggingface.co/docs/trl/dataset_formats#from-unpaired-preference-to-prompt-completion-dataset)[From unpaired preference to prompt-only dataset](https://huggingface.co/docs/trl/dataset_formats#from-unpaired-preference-to-prompt-only-dataset)[From stepwise supervision to language modeling dataset](https://huggingface.co/docs/trl/dataset_formats#from-stepwise-supervision-to-language-modeling-dataset)[From stepwise supervision to prompt completion dataset](https://huggingface.co/docs/trl/dataset_formats#from-stepwise-supervision-to-prompt-completion-dataset)[From stepwise supervision to prompt only dataset](https://huggingface.co/docs/trl/dataset_formats#from-stepwise-supervision-to-prompt-only-dataset)[From stepwise supervision to unpaired preference dataset](https://huggingface.co/docs/trl/dataset_formats#from-stepwise-supervision-to-unpaired-preference-dataset)[Vision datasets](https://huggingface.co/docs/trl/dataset_formats#vision-datasets)



--------------



[![Hugging Face's logo](https://huggingface.co/front/assets/huggingface_logo-noborder.svg)Hugging Face](https://huggingface.co/)

- [Models](https://huggingface.co/models)
- [Datasets](https://huggingface.co/datasets)
- [Spaces](https://huggingface.co/spaces)
- [Docs](https://huggingface.co/docs)
- [Enterprise](https://huggingface.co/enterprise)
- [Pricing](https://huggingface.co/pricing)

- ---
    
- ![](https://huggingface.co/avatars/5718fc9db9d5ef597ef85560419fd2ea.svg)
    

# TRL

🏡 View all docsAWS Trainium & InferentiaAccelerateAmazon SageMakerArgillaAutoTrainBitsandbytesChat UIDataset viewerDatasetsDiffusersDistilabelEvaluateGradioHubHub Python LibraryHuggingface.jsInference Endpoints (dedicated)Inference ProvidersLeaderboardsLightevalOptimumPEFTSafetensorsSentence TransformersTRLTasksText Embeddings InferenceText Generation InferenceTokenizersTransformersTransformers.jssmolagentstimm

Search documentation

⌘K

mainv0.17.0v0.16.1v0.15.2v0.14.0v0.13.0v0.12.2v0.11.4v0.10.1v0.9.6v0.8.6v0.7.11v0.6.0v0.5.0v0.4.7v0.3.1v0.2.1v0.1.1EN

 [13,557](https://github.com/huggingface/trl)

Getting started

[TRL](https://huggingface.co/docs/trl/index)[Installation](https://huggingface.co/docs/trl/installation)[Quickstart](https://huggingface.co/docs/trl/quickstart)

Conceptual Guides

[Dataset Formats](https://huggingface.co/docs/trl/dataset_formats)[Training FAQ](https://huggingface.co/docs/trl/how_to_train)[Understanding Logs](https://huggingface.co/docs/trl/logging)

How-to guides

[Command Line Interface (CLI)](https://huggingface.co/docs/trl/clis)[Customizing the Training](https://huggingface.co/docs/trl/customization)[Reducing Memory Usage](https://huggingface.co/docs/trl/reducing_memory_usage)[Speeding Up Training](https://huggingface.co/docs/trl/speeding_up_training)[Distributing Training](https://huggingface.co/docs/trl/distributing_training)[Using Trained Models](https://huggingface.co/docs/trl/use_model)

Integrations

[DeepSpeed](https://huggingface.co/docs/trl/deepspeed_integration)[Liger Kernel](https://huggingface.co/docs/trl/liger_kernel_integration)[PEFT](https://huggingface.co/docs/trl/peft_integration)[Unsloth](https://huggingface.co/docs/trl/unsloth_integration)[vLLM](https://huggingface.co/docs/trl/vllm_integration)

Examples

[Example Overview](https://huggingface.co/docs/trl/example_overview)[Community Tutorials](https://huggingface.co/docs/trl/community_tutorials)[Sentiment Tuning](https://huggingface.co/docs/trl/sentiment_tuning)[Training StackLlama](https://huggingface.co/docs/trl/using_llama_models)[Detoxifying a Language Model](https://huggingface.co/docs/trl/detoxifying_a_lm)[Learning to Use Tools](https://huggingface.co/docs/trl/learning_tools)[Multi Adapter RLHF](https://huggingface.co/docs/trl/multi_adapter_rl)[Fine-tuning a Multimodal Model Using SFT (Single or Multi-Image Dataset)](https://huggingface.co/docs/trl/training_vlm_sft)

API

Trainers

[AlignProp](https://huggingface.co/docs/trl/alignprop_trainer)[BCO](https://huggingface.co/docs/trl/bco_trainer)[CPO](https://huggingface.co/docs/trl/cpo_trainer)[DDPO](https://huggingface.co/docs/trl/ddpo_trainer)[DPO](https://huggingface.co/docs/trl/dpo_trainer)[Online DPO](https://huggingface.co/docs/trl/online_dpo_trainer)[GKD](https://huggingface.co/docs/trl/gkd_trainer)[GRPO](https://huggingface.co/docs/trl/grpo_trainer)[KTO](https://huggingface.co/docs/trl/kto_trainer)[Nash-MD](https://huggingface.co/docs/trl/nash_md_trainer)[ORPO](https://huggingface.co/docs/trl/orpo_trainer)[PPO](https://huggingface.co/docs/trl/ppo_trainer)[PRM](https://huggingface.co/docs/trl/prm_trainer)[Reward](https://huggingface.co/docs/trl/reward_trainer)[RLOO](https://huggingface.co/docs/trl/rloo_trainer)[SFT](https://huggingface.co/docs/trl/sft_trainer)[Iterative SFT](https://huggingface.co/docs/trl/iterative_sft_trainer)[XPO](https://huggingface.co/docs/trl/xpo_trainer)

[Model Classes](https://huggingface.co/docs/trl/models)[Best of N Sampling](https://huggingface.co/docs/trl/best_of_n)[Judges](https://huggingface.co/docs/trl/judges)[Callbacks](https://huggingface.co/docs/trl/callbacks)[Data Utilities](https://huggingface.co/docs/trl/data_utils)[Text Environments](https://huggingface.co/docs/trl/text_environments)[Script Utilities](https://huggingface.co/docs/trl/script_utils)[Others](https://huggingface.co/docs/trl/others)

# [](https://huggingface.co/docs/trl/how_to_train#training-faq)Training FAQ

## [](https://huggingface.co/docs/trl/how_to_train#what-metrics-should-i-look-at)What Metrics Should I Look at?

When performing classical supervised fine-tuning of language models, the loss (especially the validation loss) serves as a good indicator of the training progress. However, in Reinforcement Learning (RL), the loss becomes less informative about the model’s performance, and its value may fluctuate while the actual performance improves.

To address this, we recommend focusing on two key metrics first:

**Mean Reward**: The primary goal is to maximize the reward achieved by the model during RL training. **Objective KL Divergence**: KL divergence (Kullback-Leibler divergence) measures the dissimilarity between two probability distributions. In the context of RL training, we use it to quantify the difference between the current model and a reference model. Ideally, we want to keep the KL divergence between 0 and 10 to ensure the model’s generated text remains close to what the reference model produces.

However, there are more metrics that can be useful for debugging, checkout the [logging section](https://huggingface.co/docs/trl/logging).

## [](https://huggingface.co/docs/trl/how_to_train#why-do-we-use-a-reference-model-and-whats-the-purpose-of-kl-divergence)Why Do We Use a Reference Model, and What’s the Purpose of KL Divergence?

When training RL models, optimizing solely for reward may lead to unexpected behaviors, where the model exploits the environment in ways that don’t align with good language generation. In the case of RLHF, we use a reward model trained to predict whether a generated text is highly ranked by humans.

However, the RL model being optimized against the reward model may learn patterns that yield high reward but do not represent good language. This can result in extreme cases where the model generates texts with excessive exclamation marks or emojis to maximize the reward. In some worst-case scenarios, the model may generate patterns completely unrelated to natural language yet receive high rewards, similar to adversarial attacks.

![](https://huggingface.co/datasets/trl-lib/documentation-images/resolve/main/kl-example.png)

**Figure:** Samples without a KL penalty from [https://huggingface.co/papers/1909.08593](https://huggingface.co/papers/1909.08593).

To address this issue, we add a penalty to the reward function based on the KL divergence between the current model and the reference model. By doing this, we encourage the model to stay close to what the reference model generates.

## [](https://huggingface.co/docs/trl/how_to_train#what-is-the-concern-with-negative-kl-divergence)What Is the Concern with Negative KL Divergence?

If you generate text by purely sampling from the model distribution things work fine in general. But when you use the `generate` method there are a few caveats because it does not always purely sample depending on the settings which can cause KL-divergence to go negative. Essentially when the active model achieves `log_p_token_active < log_p_token_ref` we get negative KL-div. This can happen in a several cases:

- **top-k sampling**: the model can smooth out the probability distribution causing the top-k tokens having a smaller probability than those of the reference model but they still are selected
- **min_length**: this ignores the EOS token until `min_length` is reached. thus the model can assign a very low log prob to the EOS token and very high probs to all others until min_length is reached

These are just a few examples. Why is negative KL an issue? The total reward `R` is computed `R = r - beta * KL` so if the model can learn how to drive KL-divergence negative it effectively gets a positive reward. In many cases it can be much easier to exploit such a bug in the generation than actually learning the reward function. In addition the KL can become arbitrarily small thus the actual reward can be very small compared to it.

So how should you generate text for PPO training? Let’s have a look!

## [](https://huggingface.co/docs/trl/how_to_train#how-to-generate-text-for-training)How to generate text for training?

In order to avoid the KL issues described above we recommend to use the following settings:

Copied

generation_kwargs = {
    "min_length": -1, # don't ignore the EOS token (see above)
    "top_k": 0.0, # no top-k sampling
    "top_p": 1.0, # no nucleus sampling
    "do_sample": True, # yes, we want to sample
    "pad_token_id": tokenizer.eos_token_id, # most decoder models don't have a padding token - use EOS token instead
    "max_new_tokens": 32, # specify how many tokens you want to generate at most
}

With these settings we usually don’t encounter any issues. You can also experiments with other settings but if you encounter issues with negative KL-divergence try to go back to these and see if they persist.

## [](https://huggingface.co/docs/trl/how_to_train#how-can-debug-your-own-use-case)How can debug your own use-case?

Debugging the RL pipeline can be challenging due to its complexity. Here are some tips and suggestions to make the process easier:

- **Start from a working example**: Begin with a working example from the trl repository and gradually modify it to fit your specific use-case. Changing everything at once can make it difficult to identify the source of potential issues. For example, you can start by replacing the model in the example and once you figure out the best hyperparameters try to switch to your dataset and reward model. If you change everything at once you won’t know where a potential problem comes from.
- **Start small, scale later**: Training large models can be very slow and take several hours or days until you see any improvement. For debugging this is not a convenient timescale so try to use small model variants during the development phase and scale up once that works. That being said you sometimes have to be careful as small models might not have the capacity to solve a complicated task either.
- **Start simple**: Try to start with a minimal example and build complexity from there. Your use-case might require for example a complicated reward function consisting of many different rewards - try to use one signal first and see if you can optimize that and then add more complexity after that.
- **Inspect the generations**: It’s always a good idea to inspect what the model is generating. Maybe there is a bug in your post-processing or your prompt. Due to bad settings you might cut-off generations too soon. These things are very hard to see on the metrics but very obvious if you look at the generations.
- **Inspect the reward model**: If you reward is not improving over time maybe there’s an issue with the reward model. You can look at extreme cases to see if it does what it should: e.g. in the sentiment case you can check if simple positive and negative examples really get different rewards. And you can look at the distribution of your dataset. Finally, maybe the reward is dominated by the query which the model can’t affect so you might need to normalize this (e.g. reward of query+response minus reward of the query).

These are just a few tips that we find helpful - if you have more useful tricks feel free to open a PR to add them as well!

[<>Update on GitHub](https://github.com/huggingface/trl/blob/main/docs/source/how_to_train.md)

Dataset formats and types

[←Dataset Formats](https://huggingface.co/docs/trl/dataset_formats)[Understanding Logs→](https://huggingface.co/docs/trl/logging)

[Training FAQ](https://huggingface.co/docs/trl/how_to_train#training-faq)[What Metrics Should I Look at?](https://huggingface.co/docs/trl/how_to_train#what-metrics-should-i-look-at)[Why Do We Use a Reference Model, and What’s the Purpose of KL Divergence?](https://huggingface.co/docs/trl/how_to_train#why-do-we-use-a-reference-model-and-whats-the-purpose-of-kl-divergence)[What Is the Concern with Negative KL Divergence?](https://huggingface.co/docs/trl/how_to_train#what-is-the-concern-with-negative-kl-divergence)[How to generate text for training?](https://huggingface.co/docs/trl/how_to_train#how-to-generate-text-for-training)[How can debug your own use-case?](https://huggingface.co/docs/trl/how_to_train#how-can-debug-your-own-use-case)


------------------

