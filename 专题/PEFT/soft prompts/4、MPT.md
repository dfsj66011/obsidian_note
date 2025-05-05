
MPT（Multi-task Prompt Tuning）的核心创新在于通过分解和共享机制，从多任务数据中学习一个统一的提示，并动态适配不同目标任务，而非为每个任务独立存储或检索单独的提示。其核心思想是通过结构化参数分解（如哈达玛积）实现任务间的知识共享与高效适配。​

**与传统方法的对比**​

- ​传统多任务提示调优：每个任务学习独立的 Soft Prompt，使用时需检索或聚合（如加权平均），导致：
    - ​存储开销大​：需保存所有任务的提示参数。
    - ​适配效率低​：新任务需重新学习或复杂聚合。
- ​MPT 的改进​：所有任务共享一个基础提示矩阵 $P$，通过任务特定参数动态生成适配提示，实现：
    - ​参数共享​：减少存储需求。
    - ​灵活适配​：通过低秩矩阵调整基础提示。

## 一、两阶段流程详解​

### ​阶段 1：源训练（Source Training）​​

​从多任务数据中学习一个共享提示矩阵 $P$，并提取任务特定知识。  

1. ​任务特定提示分解​：对每个任务 $t$，其提示矩阵 $T_t$ 分解为两部分：
	* 共享基础矩阵 $P$​​：所有任务共用，维度为 $d \times k$（$d$ 为提示维度，$k$ 为提示长度）。
    - ​任务特定矩阵 $W_t$​​​：分解为两个低秩矩阵 $W^c_t, W^r_t$，捕捉任务独特特征。
2. ​哈达玛积生成提示​：任务 $t$ 的最终提示通过 $P$ 和 $W_t$​ 的元素相乘（哈达玛积）生成：$$T_t = P ⊙ (W^c_t \cdot W^r_t)$$
	* $⊙$ 表示逐元素相乘。
    - 物理意义：$W_t$​ 对 $P$ 进行任务特定的缩放调整。
3. ​多任务联合训练​：优化所有任务的 $\{W_t\}$ 和共享的 $P$，损失函数为各任务损失的加权和：$$\mathcal{L} = \sum_{t} \lambda_t \mathcal{L}_t(T_t, \text{Data}_t)$$
### ​阶段 2：目标适配（Target Adaptation）​​

​为新任务（目标任务）快速生成适配提示，无需从头训练。  

1. ​初始化目标提示​：
    - 复用共享矩阵 $P$（冻结或微调）。
    - 为新任务初始化一个低秩矩阵 $W_{\text{target}}$​。
2. ​哈达玛积生成目标提示：$$T_{\text{target}} = P ⊙ W_{\text{target}}$$
    - 若目标任务与源任务相似，可直接微调 $W_{\text{target}}$​。
    - 若差异大，可联合微调 $P$ 和 $W_{\text{target}}$​。
3. ​少量数据微调​：  
    使用目标任务的少量标注数据优化 $W_{\text{target}}$​，必要时调整 $P$。


## 二、代码解读

**低秩矩阵定义：**

```python
self.prefix_task_cols = torch.nn.Parameter(
	torch.normal(
		mean=0,
		std=0.02,
		size=(self.num_tasks, total_virtual_tokens, self.num_ranks),
	)
)
self.prefix_task_rows = torch.nn.Parameter(
	torch.normal(
		mean=0,
		std=0.02,
		size=(self.num_tasks, self.num_ranks, self.token_dim),
	)
)
```

任务特定参数通过两个低秩矩阵分解：
* `prefix_task_cols`：任务特定的列变换矩阵（ `[num_tasks, num_virtual_tokens, num_ranks]`）
* `prefix_task_rows`：任务特定的行变换矩阵（ `[num_tasks, num_ranks, token_dim]`）。
- 初始化方式​：从正态分布初始化，标准差 0.02（类似 Transformer 的默认初始化）。
- ​低秩分解原理​：通过 `cols * rows` 生成 `(num_tasks, total_virtual_tokens, token_dim)` 的任务提示矩阵，参数量从 `O(N*T*D)` 降低到 `O(N*(T*R + R*D))`（其中 $R$ 为秩）。


**初始化策略实现**：

```python
if config.prompt_tuning_init in [
	MultitaskPromptTuningInit.AVERAGE_SOURCE_TASKS,
	MultitaskPromptTuningInit.EXACT_SOURCE_TASK,
	MultitaskPromptTuningInit.ONLY_SOURCE_SHARED,
]:
	# 检查预训练权重路径
	if config.prompt_tuning_init_state_dict_path is None:
		raise ValueError(...)

	# 加载权重文件（支持 .safetensors 或 PyTorch格式）
	if config.prompt_tuning_init_state_dict_path.endswith(".safetensors"):
		from safetensors.torch import load_file
		state_dict: dict = load_file(...)
	else:
		state_dict: dict = torch_load(...)

# 根据初始化策略加载参数
if config.prompt_tuning_init == ...:
	# 处理不同的初始化逻辑
	state_dict = { ... }
	self.load_state_dict(state_dict, strict=True/False)
```

- 初始化策略​：
    - ​AVERAGE_SOURCE_TASKS​：平均多个源任务的参数，用于新任务初始化。
    - ​EXACT_SOURCE_TASK​：直接复制某个源任务的参数。
    - ​ONLY_SOURCE_SHARED​：仅加载共享的提示嵌入参数，不加载任务特定参数。
- ​技术细节​：
    - 使用 `load_file` 或 `torch.load` 加载预训练权重。
    - 通过 `load_state_dict` 将参数载入模型，`strict=False` 允许部分加载。


**前向传播：**

```python
    def forward(self, indices, task_ids):
        if task_ids is None:
            raise ValueError("task_ids cannot be None")

        # 1. 基础提示嵌入：通过词嵌入层获取初始嵌入
        prompt_embeddings = self.embedding(indices)

        # 2. 选择当前任务对应的低秩参数
        task_cols = torch.index_select(self.prefix_task_cols, 0, task_ids)  # (B, T, R)
        task_rows = torch.index_select(self.prefix_task_rows, 0, task_ids)  # (B, R, D)

        # 3. 生成任务提示矩阵：低秩矩阵乘法
        task_prompts = torch.matmul(task_cols, task_rows)  # (B, T, D)

        # 4. 任务适配：与基础嵌入相乘
        prompt_embeddings *= task_prompts

        return prompt_embeddings  # (B, T, D)
```

- ​输入参数：
    - `indices`：虚拟 token 的索引，形状通常为 `(batch_size, num_virtual_tokens)`。
    - `task_ids`：每个样本的任务 ID，形状为 `(batch_size,)`。
- ​关键步骤​：
    1. ​基础嵌入​：通过 `self.embedding` 获取初始提示嵌入。
    2. ​任务参数选择：根据 `task_ids` 选择对应的低秩矩阵。
    3. ​低秩合成​：通过矩阵乘法生成任务特定提示矩阵。
    4. ​融合操作​：将基础嵌入与任务提示矩阵相乘，实现任务适配。


**低秩分解的意义：**

- ​假设 `num_tasks=10`, `total_virtual_tokens=20`, `token_dim=768`, `num_ranks=8`：
    - 原始参数量：`10 * 20 * 768 = 153,600`
    - 低秩分解后：`10 * (20 * 8 + 8 * 768) = 10 * (160 + 6,144) = 63,040`（减少 58%）

**多任务适配机制：**

- ​动态选择​：每个批次中不同任务样本通过 `task_ids` 选择对应的参数。
- ​共享与独立结合​：
    - 共享：所有任务使用相同的 `self.embedding`。
    - 独立：每个任务通过独立低秩矩阵调整提示嵌入。

## 三、使用场景与示例

​任务示例​：文本摘要（Task 0）、机器翻译（Task 1）、对话生成（Task 2）

```python
# 配置
config = MultitaskPromptTuningConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    num_tasks=3,
    num_virtual_tokens=20,
    num_ranks=8,
    prompt_tuning_init=MultitaskPromptTuningInit.AVERAGE_SOURCE_TASKS,
    prompt_tuning_init_state_dict_path="pretrained/prompts.safetensors"
)

# 加载预训练模型的词嵌入
model = T5ForConditionalGeneration.from_pretrained("t5-small")
word_embeddings = model.get_input_embeddings()

# 初始化多任务提示层
mpt_layer = MultitaskPromptEmbedding(config, word_embeddings).to("cuda")

# 模拟输入
batch_size = 4
indices = torch.arange(20).repeat(batch_size, 1).to("cuda")  # 虚拟 token 索引
task_ids = torch.tensor([0, 1, 2, 0], dtype=torch.long).to("cuda")  # 任务ID

# 前向计算
prompts = mpt_layer(indices, task_ids)  # 形状: (4, 20, 768)

# 将提示嵌入输入模型
outputs = model(inputs_embeds=prompts, ...)
```


## 四、关键问题解答

#### 为什么选择矩阵乘法（`*=`）而不是加法？

- ​**乘法作用**​：类似于门控机制，任务提示矩阵作为权重调整基础嵌入的每个维度。
- ​**优势**​：允许更细粒度的调整，增强模型对不同任务特征的捕获能力。

#### 如何训练任务特定参数？

- ​**训练流程**​：
    1. 冻结预训练模型的主干参数。
    2. 仅训练 `prefix_task_cols` 和 `prefix_task_rows`。
    3. 可选：联合微调 `self.embedding`。

#### 如何处理不同任务数量的扩展？

- ​**动态扩展**​：新增任务时只需扩展 `prefix_task_cols` 和 `prefix_task_rows` 的 `num_tasks` 维度，无需修改其他结构。