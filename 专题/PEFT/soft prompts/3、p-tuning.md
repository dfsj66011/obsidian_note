
## 一、引言

![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/peft/p-tuning.png)

提示词可以插入到输入序列的任意位置，并由提示编码器进行优化（[图片来源](https://hf.co/papers/2103.10385)）。

P-tuning 专为自然语言理解（NLU）任务和所有语言模型而设计。它是一种软提示方法的另一种变体；P-tuning 同样添加了一个可训练的嵌入张量，该张量可以被优化以找到更好的提示，并且它使用一个提示编码器（双向长短期记忆网络或 LSTM）来优化提示参数。不过，与 prefix tuning 不同的是：

- 提示词标记可以插入到输入序列的任何位置，并不限于仅在开头插入。
- 提示词标记仅添加到输入中，而不是添加到模型的每一层（V1）。
- 引入 *auchor* 标记可以提高性能，因为它们指示了输入序列中某个组件的特征。

研究结果表明，P-tuning 比手动设计提示语更高效，并且它使类似 GPT 的模型能够在自然语言理解（NLU）任务上与类似 BERT 的模型相竞争。

## 二、示例代码

```python
>>> from peft import PromptEncoder, PromptEncoderConfig  
  
>>> config = PromptEncoderConfig(  
...     peft_type="P_TUNING",  
...     task_type="SEQ_2_SEQ_LM",  
...     num_virtual_tokens=20,  
...     token_dim=768,  
...     num_transformer_submodules=1,  
...     num_attention_heads=12,  
...     num_layers=12,  
...     encoder_reparameterization_type="MLP",  
...     encoder_hidden_size=768,  
... )  
  
>>> prompt_encoder = PromptEncoder(config)

Input shape: (`batch_size`, `total_virtual_tokens`)  
Output shape: (`batch_size`, `total_virtual_tokens`, `token_dim`)
```

## 三、结构详解

```python
class PromptEncoder(torch.nn.Module):
	def __init__(self, config):
		...
		# embedding  
		self.embedding = torch.nn.Embedding(self.total_virtual_tokens, self.token_dim)  
		if not config.inference_mode:  
		    if self.encoder_type == PromptEncoderReparameterizationType.LSTM:  
		        lstm_dropout = config.encoder_dropout  
		        num_layers = config.encoder_num_layers  
		        # LSTM  
		        self.lstm_head = torch.nn.LSTM(  
		            input_size=self.input_size,  
		            hidden_size=self.hidden_size,  
		            num_layers=num_layers,  
		            dropout=lstm_dropout,  
		            bidirectional=True,  
		            batch_first=True,  
		        )  
		  
		        self.mlp_head = torch.nn.Sequential(  
		            torch.nn.Linear(self.hidden_size * 2, self.hidden_size * 2),  
		            torch.nn.ReLU(),  
		            torch.nn.Linear(self.hidden_size * 2, self.output_size),  
		        )  
		  
		    elif self.encoder_type == PromptEncoderReparameterizationType.MLP:  
		        encoder_num_layers_default = PromptEncoderConfig.encoder_num_layers  
		        if config.encoder_num_layers != encoder_num_layers_default:  
		            warnings.warn(  
		                f"for {self.encoder_type.value}, the argument `encoder_num_layers` is ignored. "  
		                f"Exactly {encoder_num_layers_default} MLP layers are used."            )  
		        layers = [  
		            torch.nn.Linear(self.input_size, self.hidden_size),  
		            torch.nn.ReLU(),  
		            torch.nn.Linear(self.hidden_size, self.hidden_size),  
		            torch.nn.ReLU(),  
		            torch.nn.Linear(self.hidden_size, self.output_size),  
		        ]  
		        self.mlp_head = torch.nn.Sequential(*layers)  
		  
		    else:  
		        raise ValueError("Prompt encoder type not recognized. Please use one of MLP (recommended) or LSTM.")
```

这里给出两种编码方式，LSTM 和 MLP，


```python
def forward(self, indices):  
    input_embeds = self.embedding(indices)  
    if self.encoder_type == PromptEncoderReparameterizationType.LSTM:  
        output_embeds = self.mlp_head(self.lstm_head(input_embeds)[0])  
    elif self.encoder_type == PromptEncoderReparameterizationType.MLP:  
        output_embeds = self.mlp_head(input_embeds)  
    else:  
        raise ValueError("Prompt encoder type not recognized. Please use one of MLP (recommended) or LSTM.")  
  
    return output_embeds
```

首先创建了原始的虚拟 token 的 embedding，然后作为输出，通过 LSTM 或 MLP 进行编码，强化编码。