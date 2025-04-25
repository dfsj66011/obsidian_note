
## 一、引言

prompt tuning 将特定于任务的提示添加到输入中，并且这些提示参数独立于冻结的预训练模型参数进行更新。即整个训练过程，实际上是在调整 $n$ 个虚拟 token 的嵌入。


- 首次在 [Lester 等人的论文](https://aclanthology.org/2021.emnlp-main.243.pdf)中提出，这篇论文介绍了一种简单而有效的方法，称为软提示调优，它在模型的输入嵌入前添加一个可训练的张量，本质上是创建一个软提示来让冻结的语言模型执行特定的下游任务。与离散文本提示不同，软提示是通过反向传播学习的，可以通过微调来结合任意数量 token 示例的信号。
- 软提示调优只需为每个任务存储一个小的任务特定提示，并且可以使用原始预训练模型进行混合任务推理。
- 作者表明，提示调优在很大程度上优于少样本学习，并且随着规模的增加变得更具竞争力。
- 这是一种有趣的方法，可以有效地使用单个冻结模型进行多任务服务。
- 模型调优需要为每个下游任务制作整个预训练模型的任务特定副本，并且推理必须在单独的批次中进行。提示调优只需为每个任务存储一个小的任务特定提示，并且可以使用原始预训练模型进行混合任务推理。对于 T5-XXL 模型，每个调优模型的副本需要 11B 参数。相比之下，我们的调优提示每个任务只需 20,480 个参数——减少了五个数量级以上——假设提示长度为 5 个 token。
- 因此，与使用离散文本提示不同，提示调优采用软提示。软提示是可学习的，并通过反向传播进行条件化，使其适应特定任务。

![|500](https://aman.ai/primers/ai/assets/parameter-efficient-fine-tuning/PromptTuning.jpg)

- 提示调优提供了许多好处，例如：
  - 内存效率：提示调优显著减少了内存需求。
  - 多功能性：支持使用单个冻结模型进行多任务操作。
  - 性能：优于少样本学习，并且随着规模的增长变得更具竞争力。

--------
## 二、示例代码：

```python
>>> from peft import PromptEmbedding, PromptTuningConfig  
  
>>> config = PromptTuningConfig(  
...     peft_type="PROMPT_TUNING",  
...     task_type="SEQ_2_SEQ_LM",  
...     num_virtual_tokens=20,  
...     token_dim=768,  
...     num_transformer_submodules=1,  
...     num_attention_heads=12,  
...     num_layers=12,  
...     prompt_tuning_init="TEXT",  
...     prompt_tuning_init_text="Predict if sentiment of this review is positive, negative or neutral",  
...     tokenizer_name_or_path="t5-base",  
... )  
  
>>> # t5_model.shared is the word embeddings of the base model  
>>> prompt_embedding = PromptEmbedding(config, t5_model.shared)  

Input Shape: (`batch_size`, `total_virtual_tokens`)  
Output Shape: (`batch_size`, `total_virtual_tokens`, `token_dim`)  
"""
```  
  

* `prompt_tuning_init` - 提示初始化方式：
	* TEXT：使用实际文本初始化提示
		* 优点：通常收敛更快，因为初始提示已有语义含义
		* 使用场景：对任务有明确认知时（如分类、生成等）
	- RANDOM：随机初始化提示
		- 优点：完全从零开始学习，不受预设文本影响
		- 使用场景：不确定什么初始文本合适时

* `prompt_tuning_init_text` - 初始化文本
	* 当选择 `prompt_tuning_init="TEXT"` 时必须提供，这是实际用于初始化提示的文本。
	* 使用要点：
		- 文本会被 tokenizer 分词后取前 `num_virtual_tokens` 个token
		- 如果文本 token 数少于 `num_virtual_tokens`，会补上随机初始化的 token

* `num_virtual_tokens` - 虚拟 token 数量
	- 典型值：8-20个token
	- 太少（<8）：可能无法编码足够任务信息
	- 太多（>20）：可能过拟合，训练效率降低
	- 应与 `prompt_tuning_init_text` 的 token 长度协调：

*示例：*

**情感分析任务：**

```python
peft_config = PromptTuningConfig(
    task_type=TaskType.CAUSAL_LM,
    prompt_tuning_init="TEXT",
    prompt_tuning_init_text="Classify if the text is positive, neutral or negative:",  # 更明确的指令
    num_virtual_tokens=10,  # 比默认8稍多以适应更复杂的指令
    tokenizer_name_or_path="gpt2",
)
```

**翻译任务：**

```python
peft_config = PromptTuningConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,  # 注意任务类型变化
    prompt_tuning_init="TEXT",
    prompt_tuning_init_text="Translate the following English text to Chinese:", 
    num_virtual_tokens=12,  # 翻译任务可能需要更多上下文
)
```

### 初始化策略选择建议

| 场景 | 推荐初始化方式 | 理由 |
|------|--------------|------|
| 有明确任务指令 | TEXT + 相关文本 | 利用预训练语言模型对自然语言的理解 |
| 探索性任务 | RANDOM | 让模型自由学习最优提示 |
| 低资源环境 | TEXT | 收敛更快，节省计算资源 |
| 大数据集 | RANDOM | 有足够数据学习最优提示 |

### 调试技巧
1. 检查初始化文本的实际分词结果：
   ```python
   print(tokenizer.tokenize("Classify the sentiment of this review:"))
   ```
2. 如果效果不好，尝试：
   - 增加 `num_virtual_tokens`
   - 修改初始化文本使其更明确
   - 切换为 RANDOM 初始化

这些参数的合理配置对 Prompt Tuning 的效果有很大影响，建议通过小规模实验确定最佳组合。


## 三、原理

在标准的 Prompt Tuning 方法中，**整个训练过程实际上只更新这些虚拟 token（virtual tokens）的嵌入表示（embeddings）**，而预训练语言模型的所有参数都保持冻结（frozen）状态。这是 Prompt Tuning 最核心的特点。


### 具体原理详解

1. **虚拟 token 的嵌入层**  
   当设置 `num_virtual_tokens=8` 时，PEFT 会创建一个包含 8 个特殊 token 的可训练嵌入矩阵（维度为 `8 x hidden_size`，其中 `hidden_size` 是模型隐藏层维度）。这些 token：
	* 不在原始词汇表中
	* 没有对应的真实文本
	* 纯粹通过梯度下降学习最优的向量表示

2. **前向传播过程**  
   输入文本的典型处理流程：
   ```python
   # 假设输入是 "This movie is great"
   input_text = "This movie is great"
   
   # 实际输入模型的序列：
   [虚拟token1, 虚拟token2, ..., 虚拟token8] + tokenizer.encode(input_text)
   ```
   模型处理时，这些虚拟 token 的嵌入会像普通 token 一样参与注意力计算，但它们的向量会在训练中被更新。

3. **参数更新范围**  
   反向传播时：
	* ✅ 更新：仅虚拟 token 的嵌入矩阵（图中红色部分）
	* ❌ 不更新：原始语言模型的所有参数（包括 Transformer层、注意力机制等）
	* 可训练参数量通常只有原模型的 **0.01%~0.1%**

---

### 对比其他PEFT方法

| 方法                | 可训练参数位置                  | 参数量级       | 典型场景               |
|---------------------|-------------------------------|--------------|-----------------------|
| **Prompt Tuning**   | 仅虚拟token嵌入               | 极低(0.01%)  | 大模型(>1B参数)        |
| LoRA                | 注意力层的低秩矩阵             | 低(0.1%~1%)  | 中小模型               |
| Adapter             | 层间插入的小型MLP              | 中(1%~3%)    | 需要高精度的任务        |
| 全参数微调           | 整个模型                      | 100%         | 数据充足的小模型        |

### 实际训练中的关键观察

1. **嵌入向量的演变**  
   可以通过以下代码观察虚拟 token 嵌入的变化：
   ```python
   # 获取虚拟token的嵌入（训练前）
   before = model.get_prompt_embedding_to_save().detach().cpu()
   
   # 训练后再次获取
   after = model.get_prompt_embedding_to_save().detach().cpu()
   
   # 计算变化量
   print(torch.norm(after - before))  # 应该显著大于 0
   ```

2. **效果依赖模型规模**  
   - 在**大模型**（如GPT-3、T5）上：仅调整提示嵌入就能达到接近全参数微调的效果
   - 在**小模型**（如GPT-2-small）上：可能需要结合 LoRA 等其他方法

1. **与 Prefix Tuning 的区别**  
   虽然类似，但经典 Prompt Tuning：
	1. 只更新输入层的嵌入（浅层干预）
	2. 不像 Prefix Tuning 那样会在每一层插入可训练参数（深层干预）

### 为什么这样设计有效？

1. **大模型的强推理能力**  
   大语言模型本身已经具备强大的模式识别能力，只需调整输入空间的少量维度就能引导模型输出不同结果。

2. **过拟合控制**  
   限制可训练参数数量本质上是一种正则化，特别适合小数据集场景。

3. **任务特定信号**  
   学习到的虚拟 token 本质上是在输入空间构建了一个"任务导向子空间"，例如：
   - 情感分析任务的提示可能指向模型内部的情感判断模块
   - 翻译任务的提示可能激活跨语言对齐的神经元路径

这种方法的效率优势在部署时尤其明显——只需要存储少量提示参数，原始模型可以共享使用。