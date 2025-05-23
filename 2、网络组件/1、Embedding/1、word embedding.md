
[Embedding Models Explained: A Guide to NLP’s Core Technology](https://medium.com/@nay1228/embedding-models-a-comprehensive-guide-for-beginners-to-experts-0cfc11d449f1)

"Embedding" 指的是将高维数据（如文本、图像）映射为保留语义关系的稠密低维向量。

![|500](https://miro.medium.com/v2/resize:fit:700/1*RqJL4Lkd_QLUduD5nQbjUw.png)


例如，在典型的 Transformer 中，嵌入层代码示例如下：

```python
class InputEmbeddings(nn.Module):  
  
    def __init__(self, d_model: int, vocab_size: int) -> None:  
        super().__init__()  
        self.d_model = d_model  
        self.vocab_size = vocab_size  
        self.embedding = nn.Embedding(vocab_size, d_model)  
  
    def forward(self, x):  
        # (batch, seq_len) --> (batch, seq_len, d_model)  
        # Multiply by sqrt(d_model) to scale the embeddings according to the paper  
        return self.embedding(x) * math.sqrt(self.d_model)
```

