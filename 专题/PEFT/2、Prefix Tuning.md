
## 一、引言

在 [Prefix-Tuning: Optimizing Continuous Prompts for Generation](https://arxiv.org/abs/2101.00190) 一文中提出的前缀调整（prefix-tuning）是一种用于自然语言生成任务的轻量级替代微调方法，该方法保持语言模型参数冻结，但优化一个小的连续任务特定向量（称为前缀）。

与其向模型输入中添加软提示，不如在所有 Transformer 块的隐藏状态前添加可训练参数。在微调过程中，语言模型的原始参数保持冻结状态，而前缀参数会被更新。

前缀调整（Prefix-tuning）从提示（prompting）中汲取灵感，使得后续标记能够像对待“虚拟标记”一样关注这个前缀。

下图来自论文，展示了微调（顶部）会更新所有 Transformer 参数（红色 Transformer 框），并且需要为每个任务存储一整个模型副本。他们提出了前缀调优（底部），该方法冻结 Transformer 参数，仅优化前缀（红色前缀块）。因此，前缀调优只需为每个任务存储前缀，使其具有模块化和空间效率高的特点。请注意，每个垂直块表示一个时间步的 Transformer 激活。

![|500](https://aman.ai/primers/ai/assets/parameter-efficient-fine-tuning/Prefix-Tuning.jpg)

他们将前缀调整（prefix-tuning）应用于 GPT-2 以进行表格到文本的生成，并将其应用于 BART 以进行摘要生成。他们发现，通过仅学习 0.1% 的参数，前缀调整在全数据设置中获得了可比的性能，在低数据设置中优于微调，并且在推断训练期间未见过的主题的示例时表现更好。

下图（[来源](https://sebastianraschka.com/blog/2023/llm-finetuning-llama-adapter.html)）展示了在前缀调优中，可训练张量被添加到每个 Transformer 块中，而不仅仅是在输入嵌入中。
![|600](https://aman.ai/primers/ai/assets/parameter-efficient-fine-tuning/prefixtuningraschka.png)



## 二、示例代码

```python
>>> from peft import PrefixEncoder, PrefixTuningConfig  
  
>>> config = PrefixTuningConfig(  
...     peft_type="PREFIX_TUNING",  
...     task_type="SEQ_2_SEQ_LM",  
...     num_virtual_tokens=20,  
...     token_dim=768,  
...     num_transformer_submodules=1,  
...     num_attention_heads=12,  
...     num_layers=12,  
...     encoder_hidden_size=768,  
... )  
>>> prefix_encoder = PrefixEncoder(config)  

Input shape: (`batch_size`, `num_virtual_tokens`)  
Output shape: (`batch_size`, `num_virtual_tokens`, `2*layers*hidden`)
```

*这里 output shape 中的 2，是只 K 和 V，然后每层都有，需要乘 layers，维度是 hidden*

## 三、结构详解

以下以 ​*GPT-2*​ 模型为例（序列长度 512，虚拟 token 数量20，隐藏维度 768，层数 12），详细说明各层的数据流动和维度变化：

#### ​**1. 输入层**​

- ​**原始输入**​：`input_ids` (batch_size, 512)  
    经过嵌入层转换为：  `hidden_states` → (batch_size, 512, 768)
    
- ​**前缀注入**​：
    
    - ​**前缀向量生成**​：  
        每个层独立生成前缀 Key 和 Value：

        ```python
        # 参数维度: (20, 12层 × 2 × 768)
        prefix_embeddings = PrefixEncoder(config)  
        past_key_values = prefix_embeddings()  # (12层, 2, 20, 768)
        ```
        
    - ​**分层处理**​：  
        每层获得独立的前缀 Key/Value：
        
        ```python
        # 第i层的前缀
        layer_prefix_key = past_key_values[i][0]  # (20, 768)
        layer_prefix_value = past_key_values[i][1]
        ```

---

#### ​**2. Transformer 层（共12层）​**​

每个层的处理流程如下：

##### ​**​(1) 自注意力计算**​

- ​**原始 Query/Key/Value**​：  
    `Q = hidden_states @ W_Q` → (batch_size, 512, 768)  
    `K = hidden_states @ W_K` → (batch_size, 512, 768)  
    `V = hidden_states @ W_V` → (batch_size, 512, 768)
    
- ​**拼接前缀**​：
    
    ```python
    # 拼接后的Key/Value (每层独立)
    K = concat([layer_prefix_key, K], dim=1)  # (batch_size, 532, 768)
    V = concat([layer_prefix_value, V], dim=1)
    ```
    
- ​**注意力分数计算**​：
    
    ```python
    # Q.shape → (batch_size, 512, 768)
    # K.shape → (batch_size, 532, 768)
    attn_scores = Q @ K.transpose(-1, -2)  # (batch_size, 512, 532)
    attn_weights = softmax(attn_scores / sqrt(d_k))
    attn_output = attn_weights @ V  # (batch_size, 512, 768)
    ```
    

##### ​**​(2) 前馈网络（FFN）​**​

- 前缀参数**不参与**FFN计算，处理与原始模型一致：
    
    ```python
    ff_output = FFN(attn_output)  # (batch_size, 512, 768)
    ```


### ​**关键结构对比**​

| ​**组件**​          | ​**原始模型**​ | ​**Prefix Tuning**​                     |
| ----------------- | ---------- | --------------------------------------- |
| ​**输入序列长度**​      | $512$      | $512$（前缀不占输入长度）                         |
| ​**Key/Value长度**​ | $512$      | $532$（每层+20前缀）                          |
| ​**参数可训练性**​      | 全参数可训练     | 仅前缀向量可训练（原始参数冻结）                        |
| ​**计算复杂度**​       | $O(512^2)$ | $O(512×532)$（增加约 $4.3$%）                |
| ​**参数量（12 层）​**​  | 约 $1.5$ 亿  | $12×2×20×768 = 368,640$（仅 $0.24$ % 总参数） |

### ​**MLP 投影的细节**​

当启用 `prefix_projection=True` 时：

```python
if self.prefix_projection and not config.inference_mode:  
    # Use a two-layer MLP to encode the prefix  
    self.embedding = torch.nn.Embedding(num_virtual_tokens, token_dim)  
    self.transform = torch.nn.Sequential(  
        torch.nn.Linear(token_dim, encoder_hidden_size),  
        torch.nn.Tanh(),  
        torch.nn.Linear(encoder_hidden_size, num_layers * 2 * token_dim),  
    )  
else:  
    self.embedding = torch.nn.Embedding(num_virtual_tokens, num_layers * 2 * token_dim)
```

- ​**作用**​：增强前缀参数的表达能力，防止训练初期陷入局部最优
- ​**计算流程**​：
    
    ```python
    # 输入: (batch_size, 20)
    prefix_embeds = self.embedding(input_ids)  # (20, 768)
    projected = self.mlp(prefix_embeds)       # (20, 12 * 2 * 768)
    past_key_values = projected.split(12 * 2 * 768//12, dim=-1)
    ```

### ​**性能优化策略**​

1. ​**前缀缓存**​：  
    在推理时预计算各层前缀，避免重复计算：

    ```python
    past_key_values = model.prefix_encoder()
    outputs = model(input_ids, past_key_values=past_key_values)
    ```
    
2. ​**稀疏注意力**​：  
    可配置前缀仅参与部分位置的注意力计算（如仅关注前 10 个 token）。

通过这种结构设计，Prefix Tuning 实现了对模型行为的细粒度控制，尤其适合需要**多层次语义干预**的任务（如故事生成、对话系统），同时保持高效训练特性。