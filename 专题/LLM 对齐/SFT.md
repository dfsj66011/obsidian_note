
> 文章中代码片段使用的库与版本号对应关系
> transformers\==4.49.0
> trl\==0.16.0


## 一、引言

监督微调 (SFT) 主要用于将预训练语言模型调整为能够遵循指令、进行对话并使用特定输出格式。虽然预训练模型具备出色的通用能力，但 SFT 有助于将它们转变为更能理解和响应用户提示的助手型模型。这通常通过在人类书写的对话和指令数据集上进行训练来实现。

<img src="https://substackcdn.com/image/fetch/w_1272,c_limit,f_webp,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F4c51db77-8d97-45a9-bd2c-d71e930ff0b8_2292x1234.png" width="600">
**LLM 训练的不同阶段示意图**


SFT 在预训练获得的一般能力和实际应用所需的专门行为之间架起了桥梁。在以下方面尤为重要：

- 遵循指令
- 提高特定任务的表现
- 符合人类偏好
- 领域适应

与预训练相比，SFT 所需的数据量显著减少，但可以显著提升目标任务的表现，其数据效率较高。

### 1.1 SFT 使用条件

首先，应该考虑使用现有的指令调整模型和精心设计的提示是否足以满足需求。因此 SFT 需要大量的计算资源和工程工作，因此只有在现有模型无法满足需求时才应采用。

应仅在以下情况下考虑 SFT：

* 需要超出提示所能实现的额外性能 
* 有一个特定的用例，其中使用大型通用模型的成本超过微调较小模型的成本 
* 需要现有模型难以处理的专门的输出格式或特定领域的知识

*主要解决场景* 包括：

* 输出结构的精确控制：如以特定的聊天模板格式回复、严格的输出模式、回复中保持一致的样式
* 领域适应：如教授领域术语和概念、执行专业标准、技术查询、遵循行业特定指南等


## 二、数据集准备

> [!tip]
> 当使用包含 `messages` 字段的数据集（如上例）时，SFTTrainer 会自动应用从 Hub 中检索的模型聊天模板。这意味着无需进行任何额外配置即可处理聊天式对话，训练器将根据模型预期的模板格式来格式化消息。


训练数据的质量对于微调的成功至关重要，SFT 需要一个由 input-output 对构成的特定任务数据集。每对输入输出对应包含以下内容：

1. 输入提示
2. 预期模型响应
3. 任何额外的上下文或元数据

这里很重要的一项内容是 *聊天模板*，

```bash
# 全参数微调 (来自 trl.trl.srcipts.sft.py)
python trl/scripts/sft.py \  
    --model_name_or_path Qwen/Qwen2-0.5B \    
    --dataset_name trl-lib/Capybara \    
    --learning_rate 2.0e-5 \    
    --num_train_epochs 1 \    
    --packing \    
    --per_device_train_batch_size 2 \    
    --gradient_accumulation_steps 8 \    
    --gradient_checkpointing \    
    --logging_steps 25 \    
    --eval_strategy steps \    
    --eval_steps 100 \    
    --output_dir Qwen2-0.5B-SFT \    
    --push_to_hub
```

这里使用的是 `Qwen/Qwen2-0.5B`，其聊天模版位于[这里](https://huggingface.co/Qwen/Qwen2-0.5B/blob/main/tokenizer_config.json#L31)，

```
"chat_template": "{% for message in messages %}{% if loop.first and messages[0]['role'] != 'system' %}{{ '<\|im_start\|>system\nYou are a helpful assistant<\|im_end\|>\n' }}{% endif %}{{'<\|im_start\|>' + message['role'] + '\n' + message['content'] + '<\|im_end\|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<\|im_start\|>assistant\n' }}{% endif %}",
```

这段代码使用了 Jinja2 模板语言来格式化聊天消息。

<details>
  <summary>(Jinja2 模板语言说明)点击展开</summary>

语法解释

1. **循环**：
    
    - `{% for message in messages %}`：遍历 `messages` 列表中的每个 `message`。
    - `{% endfor %}`：结束循环。
2. **条件语句**：
    
    - `{% if loop.first and messages[0]['role'] != 'system' %}`：如果是循环的第一个元素，并且第一个消息的角色不是 `system`，则执行下面的代码。
3. **变量替换**：
    
    - `{{ ... }}`：在模板中插入变量或表达式的值。

  ### 用法说明

- **系统消息初始化**：
    
    - 如果第一个消息不是 `system`，则插入一段系统消息：`|im_start|system\nYou are a helpful assistant|im_end|\n`。
- **消息格式化**：
    
    - 每个消息被格式化为：`|im_start|role\ncontent|im_end|\n`，其中 `role` 和 `content`分别是消息的角色和内容。
- **生成提示**：
    
    - `{% if add_generation_prompt %}{{ '|im_start|assistant\n' }}{% endif %}`：如果 `add_generation_prompt` 为真，则在最后插入 `|im_start|assistant\n`，提示生成助手的响应。
</details>

每种模型的模板可能不相同，使用前需要查明，而这里作为演示使用的数据集 [`trl-lib/Capybara`](https://huggingface.co/datasets/trl-lib/Capybara)主要使用的是 `messages` 数据列，在 `trl` 库中，已经内置了模板转换功能。

### 2.1 数据预处理

最常用的数据集格式如下：

```python
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"},
    {"role": "assistant", "content": "Hi! How can I help you today?"},
    {"role": "user", "content": "What's the weather?"},
]
```

在 `SFTTrainer()` 中，L191-L207 为数据集的准备相关工作，其中在 `_prepare_dataset()` 方法中 L430-L470 为模板整合、Tokenizer 等相关处理，具体而言：

* L435 调用 `maybe_convert_to_chatml()`，实际上 [`trl-lib/Capybara`](https://huggingface.co/datasets/trl-lib/Capybara) 的 `messages` 数据列已经满足上面的格式，这一步并不需要做额外工作
* L446 调用 `maybe_apply_chat_template() -> apply_chat_template(): L95`，
	```python
	messages = tokenizer.apply_chat_template(example["messages"], tools=tools, tokenize=False)
	```

这里就会利用 tokenizer 的 chat_template 进行整合，所以真正输入到模型中的文本（before encoder）为（内容太长，部分省略）：
```
{'text': '<|im_start|>system\nYou are a helpful assistant<|im_end|>\n
<|im_start|>user\nRecommend a movie to watch.\n<|im_end|>\n
<|im_start|>assistant\nI would recommend the movie, xxx.<|im_end|>\n
<|im_start|>user\nDescribe the character development of Tim Robbins\' character in "The Shawshank Redemption".<|im_end|>\n
<|im_start|>assistant\nxxx.<|im_end|>\n
<|im_start|>user\nExplain the significance of the friendship between Andy and Red in shaping Andy\'s character development.<|im_end|>\n
<|im_start|>assistant\nxxxx.<|im_end|>\n'}
```

* L456-470 则完整 tokenize 过程，生成 `input_ids` 和 `attention_mask`
* 最后还有打包和截断等操作：
	* Packing: 将多个样本合并为一个样本，以便更好地利用模型的输入长度。
	* Truncating: 将样本截断到指定的最大长度，防止超过模型的输入限制。


## 三、损失函数

### 3.1 交叉熵损失

#### 3.1.1 公式说明

SFT 的主要任务就是下一个 token 预测，这与预训练阶段的主要任务一致，因此要 SFT 的模型需要是 `AutoModelForCausalLM` 的，预测的标签就是 token 级的偏移。

语言模型最常用的损失函数是交叉熵损失函数：$$\mathcal{L}_{\text{SFT}} = -\frac{1}{N} \sum_{i=1}^{N} \sum_{t=1}^{T} \mathbb{1}_{\text{response}}(t) \cdot \log P(y_t^i | y_{<t}^i, x^i)$$
- ​$N$​：批次中的样本数量。
- ​$T​$​：序列长度。
- ​$\mathbb{1}_{\text{response}}(t)$​：指示函数，用于标识哪些时间步需要计算损失。通常用于区分模型生成的部分和输入的部分，仅在生成部分计算损失。
- ​$P(y_t^i | y_{<t}^i, x^i)$：模型在给定上下文 $y_{<t}^i$ 和输入 $x^i$ 时，对正确标记 $y^i_t$​ 的预测概率。


#### 3.1.2 代码解释

代码位于：sft_trainer.py - SFTTrainer.compute_loss(): L490

* L495-497：通过父类的 `compute_loss()` 进行损失计算（位于 transformers.trainer.py 下）
	* L3759：`outputs = model(**inputs)` ，outputs（通常包含 logits、loss、hidden_states 等）
	* 具体而言，实际内部调用的是 `ForCausalLMLoss()`，位于 `transformers.loss.loss_utils.py`

```python
def ForCausalLMLoss(
    logits: torch.Tensor,                # 模型输出的原始预测分数，形状为 [batch_size, seq_len, vocab_size]
    labels: torch.Tensor,                # 真实标签（通常是输入序列右移一位），形状为 [batch_size, seq_len]
    vocab_size: int,                     # 词表大小（用于reshape logits）
    num_items_in_batch: Optional[int] = None,  # 实际有效的样本数（可选，用于处理padding）
    ignore_index: int = -100,            # 需要忽略的标签索引（如padding部分）
    shift_labels: Optional[torch.Tensor] = None,  # 可预先计算的右移标签（可选）
    ​**kwargs,                           # 其他传递给损失函数的参数
) -> torch.Tensor:
    """
    计算因果语言模型（Causal LM）的交叉熵损失。
    核心逻辑：
    1. 将标签右移一位（使第t步预测第t+1个token）
    2. 处理忽略的标签（如padding）
    3. 计算交叉熵损失（支持模型并行）
    """
    
    # 1. 将logits转为float类型以避免低精度计算问题（例如fp16下溢出）
    logits = logits.float()

    # 2. 标签右移处理（若未提供预先计算的shift_labels）
    if shift_labels is None:
        # 在序列末尾填充一个ignore_index（保证右移后长度一致）
        # 示例：labels=[1,2,3] -> pad后=[1,2,3,-100] -> 右移后=[2,3,-100]
        labels = nn.functional.pad(labels, (0, 1), value=ignore_index)
        shift_labels = labels[..., 1:].contiguous()  # 右移并确保内存连续

    # 3. 展平logits和标签以适应交叉熵计算
    # logits形状: [batch_size*seq_len, vocab_size]
    # shift_labels形状: [batch_size*seq_len]
    logits = logits.view(-1, vocab_size)
    shift_labels = shift_labels.view(-1)

    # 4. 确保标签与logits在同一设备（支持模型并行）
    shift_labels = shift_labels.to(logits.device)

    # 5. 计算改进的交叉熵损失（可能包含特殊处理如掩码、缩放等）
    loss = fixed_cross_entropy(
        logits, 
        shift_labels, 
        num_items_in_batch,  # 实际有效token数（非padding部分）
        ignore_index,        # 需要忽略的标签索引
        ​**kwargs
    )
    
    return loss  # 返回标量损失值
```

### 3.2 标签平滑损失

如果使用标签平滑功能，在父类 的 `compute_loss()` 中：

```python
if labels is not None:  
    unwrapped_model = self.accelerator.unwrap_model(model)          # 解包分布式训练封装的模型  
    if _is_peft_model(unwrapped_model):  
        model_name = unwrapped_model.base_model.model._get_name()   # 获取 PEFT 适配模型的原始名称  
    else:  
        model_name = unwrapped_model._get_name()                    # 普通模型的名称  
    # 情况 1：用户自定义损失函数  
    if self.compute_loss_func is not None:  
        loss = self.compute_loss_func(outputs, labels, num_items_in_batch=num_items_in_batch)  
    # 情况 2：因果语言模型（如 GPT） + 标签平滑  
    elif model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():  
        loss = self.label_smoother(outputs, labels, shift_labels=True)   # 使用标签平滑时需 shift_labels=True（因为因果 LM 的预测是错位对齐的）。  
    # 情况 3：普通模型 + 标签平滑  
    else:  
        loss = self.label_smoother(outputs, labels)
```

实际上执行的是 transformer.trainer_pt_utils.LabelSmoother

```python
class LabelSmoother:  
    """  
    这段代码实现的是标签平滑交叉熵损失（Label Smoothing Cross-Entropy Loss）  
    结合了传统的负对数似然损失（NLL Loss）和均匀分布的平滑损失。  
    """  
    epsilon: float = 0.1  
    ignore_index: int = -100  
  
    def __call__(self, model_output, labels, shift_labels=False):  
        """计算标签平滑交叉熵损失"""  
        """1. 提取 logits：模型输出的原始预测值（未归一化的分数）"""  
        logits = model_output["logits"] if isinstance(model_output, dict) else model_output[0]  
        """2. 因果语言模型，预测目标是“下一个token”，所以需要将 logits 和 labels 错位对齐（shift_labels=True）"""  
        if shift_labels:  
            logits = logits[..., :-1, :].contiguous()      # 去掉最后一个token的 logits            labels = labels[..., 1:].contiguous()          # 去掉第一个 token 的 labels        """3. 计算对数概率"""  
        log_probs = -nn.functional.log_softmax(logits, dim=-1)  # 对 logits 归一化并取对数，得到对数概率， 负号是因为交叉熵损失需要最小化负对数似然。  
        if labels.dim() == log_probs.dim() - 1:   # 如果 labels 比 log_probs 少一维（例如 labels 是 [batch, seq_len]，而 log_probs 是 [batch, seq_len, vocab_size]），则扩展维度以匹配。  
            labels = labels.unsqueeze(-1)  
        """4. 掩码处理, 标记需要忽略的位置（如 labels=-100 的位置）"""  
        padding_mask = labels.eq(self.ignore_index)  
        # Clamp：将标签中的负值（如-100）替换为0，因为后续的 gather 操作需要有效索引。  
        labels = torch.clamp(labels, min=0)  
        """ 5. 计算两种损失: 5.1 负对数似然损失（NLL Loss）"""  
        nll_loss = log_probs.gather(dim=-1, index=labels)   # gather：从 log_probs 中提取对应 labels 位置的对数概率，结果是每个 token 的预测值与真实标签的交叉熵。  
        """5.2 平滑损失（Smoothed Loss），sum：对所有类别（vocab_size）的对数概率求和，相当于假设标签是均匀分布时的损失"""  
        smoothed_loss = log_probs.sum(dim=-1, keepdim=True, dtype=torch.float32)  
  
        nll_loss.masked_fill_(padding_mask, 0.0)          # 将 padding 位置的损失置零（不参与梯度计算）  
        smoothed_loss.masked_fill_(padding_mask, 0.0)  
  
        """6. 归一化损失"""  
        # Take the mean over the label dimensions, then divide by the number of active elements (i.e. not-padded):  
        num_active_elements = padding_mask.numel() - padding_mask.long().sum()     # 非 padding 的 token 数量  
        nll_loss = nll_loss.sum() / num_active_elements                            # 非 padding 位置的平均损失  
        smoothed_loss = smoothed_loss.sum() / (num_active_elements * log_probs.shape[-1])    # 进一步除以类别数（log_probs.shape[-1]），相当于均匀分布下的平均损失。  
        return (1 - self.epsilon) * nll_loss + self.epsilon * smoothed_loss
```

标签平滑损失计算公式：$$\mathcal{L} = (1-\epsilon) \cdot \mathcal{L}_{\text{NLL}} + \epsilon \cdot \mathcal{L}_{\text{smooth}}$$  
**示例计算**，假设：  

* `logits = [[[2.0, 1.0, 0.5]]]` （batch=1, seq_len=1, vocab_size=3）  
* `labels = [[1]]` （真实类别是1）  
* `epsilon = 0.1`          
- `ignore_index = -100`                    

**计算步骤**：  
1. `log_probs = -log_softmax(logits) ≈ [[[-0.55, -1.55, -2.05]]]`           
2. `nll_loss = log_probs.gather(..., index=1) ≈ -1.55`            
3. `smoothed_loss = log_probs.sum() / 3 ≈ (-0.55 -1.55 -2.05)/3 ≈ -1.38`     
4. 最终损失：  $$(1-0.1) \times (-1.55) + 0.1 \times (-1.38) \approx -1.53$$
## 四、训练技巧及常见训练信号

### 4.1 参数配置

使用 trl 库，参数主要包含三部分：

```python
script_args = ScriptArguments(  
    dataset_name="trl-lib/Capybara"  
)  
training_args = SFTConfig(  
    learning_rate=2.0e-5,  
    num_train_epochs=1,  
    packing=True,  
    per_device_train_batch_size=2,  
    gradient_accumulation_steps=8,  
    gradient_checkpointing=True,  
    logging_steps=25,  
)  
model_args = ModelConfig(  
    model_name_or_path="Qwen/Qwen2-0.5B"  
)
```

此处不做过多介绍，一般的经验为从保守值开始并根据监控进行调整： 

- 从 1-3 个时期开始 
- 最初使用较小的批量大小 
- 密切监控验证指标 
- 如果训练不稳定，则调整学习率

### 4.2 正常监督信号

**在训练期间注意以下警告信号**：

1. 验证损失增加而训练损失减少（过度拟合）
2. 损失值没有显着改善（欠拟合）
3. 损失值极低（潜在记忆）
4. 输出格式不一致（模板学习问题）

随着训练的进行，损失曲线应该逐渐稳定。训练效果良好的关键指标是训练损失和验证损失之间的差距较小，这表明模型正在学习可推广的模式，而不是记忆特定的例子。绝对损失值会根据任务和数据集而变化。

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/nlp_course_sft_loss_graphic.png" width="400">
上图展示了典型的训练过程。请注意，训练损失和验证损失一开始都急剧下降，然后逐渐趋于平稳。这种模式表明模型在保持泛化能力的同时，学习效果良好。

### 4.3 异常信号

一些常见的警告信号以及可以考虑的解决方案。

![SFTTrainer培训|400](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/sft_loss_1.png)

如果验证损失的下降速度明显慢于训练损失，则你的模型很可能对训练数据存在过拟合。请考虑：

- 减少训练步骤
- 增加数据集大小
- 验证数据集质量和多样性

![SFTTrainer培训|400](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/sft_loss_2.png)

如果损失没有显示出显着的改善，则模型可能是：

- 学习太慢（尝试提高学习速度）
- 努力完成任务（检查数据质量和任务复杂性）
- 达到架构限制（考虑不同的模型）

![SFTTrainer培训|400](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/sft_loss_3.png)

极低的损失值可能意味着记忆而非学习。在以下情况下尤其值得关注：

- 该模型在新的类似示例上表现不佳
- 产出缺乏多样性
- 响应与训练示例太相似

在训练期间，监控损失值和模型的实际输出。有时，损失值可能看起来不错，但模型却会出现一些不良行为。定期对模型的响应进行定性评估有助于发现仅凭指标可能遗漏的问题。

应该注意，在这里概述的损失值的解释是针对最常见的情况，事实上，损失值可以根据模型、数据集、训练参数等以各种方式表现。

## 五、SFT 后评估

完成 SFT 后，请考虑以下后续行动：

1. 在保留的测试数据上彻底评估模型
2. 验证各种输入的模板依从性
3. 测试特定领域的知识保留
4. 监控实际性能指标

记录训练过程，包括： 
- 数据集特征 
- 训练参数 
- 性能指标 
- 已知限制 

此文档对于未来的模型迭代非常有价值。


## 六、LoRA

对大型语言模型进行微调是一个资源密集型的过程。LoRA 是一种允许我们用少量参数对大型语言模型进行微调的技术。它的工作原理是向注意力权重中添加和优化较小的矩阵，通常可以将可训练参数减少约 90%。

LoRA 的 *工作原理* 是向 Transformer 层添加秩分解矩阵对，通常侧重于注意力权重。在推理过程中，这些适配器权重可以与基础模型合并，从而不会产生额外的延迟开销。LoRA 尤其适用于将大型语言模型适配到特定任务或领域，同时保持资源需求的可控性。

### 6.1 LoRA 的主要优势

1. **内存效率**：
    - 只有适配器参数存储在 GPU 内存中
    - 基础模型权重保持冻结，可以以较低的精度加载
    - 支持在消费级 GPU 上对大型模型进行微调
2. **训练特色**：
    - 只需最少设置即可实现本机 PEFT/LoRA 集成
    - 支持 QLoRA（量化 LoRA），实现更高的内存效率
3. **适配器管理**：
    - 在检查点期间保存适配器权重
    - 将适配器合并回基础模型


![lora_load_适配器|500](https://github.com/huggingface/smol-course/raw/main/3_parameter_efficient_finetuning/images/lora_adapter.png)

### 6.2 使用 trl 和 SFTTrainer 与 LoRA 微调 LLM

SFTTrainer 通过 [PEFT](https://huggingface.co/docs/peft/en/index) 库提供与 LoRA 适配器的集成。这意味着我们可以像使用 SFT 一样对模型进行微调，但使用 LoRA 可以减少需要训练的参数数量。

我们将在示例中使用`LoRAConfig`PEFT 中的类。设置过程只需几个配置步骤：

1. 定义 LoRA 配置（等级、alpha、dropout）
2. 使用 PEFT 配置创建 SFTTrainer
3. 训练并保存适配器权重


我们将在示例中使用 PEFT 的 `LoRAConfig` 类。设置只需几个配置步骤：

1. 定义 LoRA 配置（包括 rank、alpha、dropout）
2. 使用 PEFT 配置创建 `SFTTrainer`
3. 训练并保存适配器权重

| 参数             | 描述                                                                               |
| -------------- | -------------------------------------------------------------------------------- |
| r (rank)       | 用于权重更新的低秩矩阵的维度。通常在 4-32 之间。较低的值提供更多压缩，但可能降低表达能力。                                 |
| lora_alpha     | LoRA 层的缩放因子，通常设置为 rank 值的 2 倍。较高的值会导致更强的适应效果。                                    |
| lora_dropout   | LoRA 层的 dropout 概率，通常为 0.05-0.1。较高的值有助于防止训练过程中的过拟合。                              |
| bias           | 控制偏置项的训练。选项有 "none"、"all" 或 "lora_only"。"none" 是最常用的，因其内存效率高。                    |
| target_modules | 指定将 LoRA 应用于哪些模型模块。可以是 "all-linear" 或特定模块如 "q_proj,v_proj"。更多模块提供更大的适应性，但增加内存使用。 |

在实现 PEFT 方法时，从小的 rank 值（4-8）开始，并监控训练损失。使用验证集防止过拟合，并在可能的情况下与完全微调的基线进行比较。不同方法的效果因任务而异，因此实验是关键。


