

# PaLM: Efficiently Training Massive Language Models

### Unprecedented size, efficiency, and performance for LLMs...

[

![](https://substackcdn.com/image/fetch/w_36,h_36,c_fill,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbuckete

This newsletter is sponsored by [Rebuy](https://www.rebuyengine.com/), the Commerce AI company. If you’re working at the intersection of engineering/ML and want to join a fast-growing startup, [reach out to me](https://www.linkedin.com/in/cameron-r-wolfe-04744a238/)! I want to find awesome people to learn and build with.

Subscribe

If you like this newsletter, please subscribe, share it, or follow me on [twitter](https://twitter.com/cwolferesearch). Thank you for your support!

---

**this newsletter.** This newsletter is part of my series on modern advancements in language models. Recently, deep learning research has been taken over by the unprecedented success of large language models (LLMs) like [ChatGPT](https://openai.com/blog/chatgpt) and [GPT-4](https://openai.com/research/gpt-4). I have already overviewed the history and core concepts behind these models.

- GPT and GPT-2 [[link](https://cameronrwolfe.substack.com/p/language-models-gpt-and-gpt-2)]
    
- Scaling laws and GPT-3 [[link](https://cameronrwolfe.substack.com/p/language-model-scaling-laws-and-gpt)] 
    
- Modern LLMs (beyond GPT-3) [[link](https://cameronrwolfe.substack.com/p/modern-llms-mt-nlg-chinchilla-gopher)]
    
- Specialized LLMs [[link](https://cameronrwolfe.substack.com/p/specialized-llms-chatgpt-lamda-galactica)]
    

Within this series, I will go beyond this history of LLMs into more recent topics, examining a variety of recent techniques and findings that are relevant to LLMs.

---

[

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F1b84e48c-6fea-422c-affd-dc8a314b1934_2010x1506.png)



](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F1b84e48c-6fea-422c-affd-dc8a314b1934_2010x1506.png)

(from [1] and [16])

In recent years, large, deep neural networks have become the definitive architecture of choice for solving most language understanding and generation tasks. Initially, models were proposed, such as [BERT](https://cameronrwolfe.substack.com/p/language-understanding-with-bert) [2] and T5 [3], that used a [two-part training methodology](https://cameronrwolfe.substack.com/i/76273144/training-bert) of pre-training (with [self-supervised “infilling” objectives](https://cameronrwolfe.substack.com/i/76273144/self-supervised-learning)) over a large corpus of text, then fine-tuning on a target dataset; see below. Despite the utility of these techniques, recent work on large language models (LLMs) has shown that large, autoregressive (decoder-only) transformer models are incredibly capable at few-shot learning, achieving impressive performance with minimal adaptation to downstream tasks.

[

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fef9a6d61-0ad6-4604-8ad7-7332d46d1f73_1662x1462.png)



](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fef9a6d61-0ad6-4604-8ad7-7332d46d1f73_1662x1462.png)

(from [4])

The few-shot learning capabilities of LLMs were first demonstrated by [GPT-3](https://cameronrwolfe.substack.com/p/language-model-scaling-laws-and-gpt) [4], a 175 billion parameter LLM. To perform few-shot prediction, the model is pre-trained (using a basic [language modeling objective](https://cameronrwolfe.substack.com/i/85568430/language-modeling)) over a massive corpus of text, then provided task descriptions and a handful of examples of how a task should be solved; see above. Further analysis of LLMs indicated that model performance improves smoothly with scale (according to a [power law](https://cameronrwolfe.substack.com/i/88082618/power-laws)) [5, 6]. As such, various LLMs were proposed following GPT-3 that attempt to “scale up” the model and training, oftentimes achieving improved results via a combination of larger models and more/better pre-training data.

Training larger LLMs is beneficial but difficult to do efficiently. Typically, we distribute training across many machines, each with several accelerators (i.e., GPUs or [TPUs](https://cloud.google.com/tpu/docs/tpus)). This has been done successfully before (e.g., MT-NLG trains a 530 billion parameter LLM across a system with 2240 A100 GPUs), but the results were not that impressive. The model, although large, was not trained over enough data. However, given a higher training throughput, we could (in theory) pre-train such large models more extensively on larger datasets, yielding much better results.

In this overview, we will explore the Pathways Language Model (PaLM), a 540 billion parameter LLM trained using Google’s [Pathways](https://blog.google/technology/ai/introducing-pathways-next-generation-ai-architecture/) framework. By eliminating [pipeline parallelism](https://cameronrwolfe.substack.com/i/88082618/other-useful-details), this architecture achieves impressive training throughput, allowing PaLM to be pre-trained over a more extensive dataset. The few-shot performance of the resulting model is state-of-the-art. Plus, PaLM is somewhat capable of solving difficult reasoning tasks. Put simply, PaLM is a clear reminder that LLM performance has not yet reached a plateau with respect to scale. Given a sufficiently efficient training infrastructure that permits pre-training larger models over more data, we continue to see improvements in performance.

# Background

We have explored the topic of language modeling extensively in this newsletter and overviewed several notable (large) language models in prior posts:

- GPT and GPT-2 [[link](https://cameronrwolfe.substack.com/p/language-models-gpt-and-gpt-2)]
    
- Scaling Laws and GPT-3 [[link](https://cameronrwolfe.substack.com/p/language-model-scaling-laws-and-gpt#%C2%A7other-useful-details)]
    
- Modern LLMs [[link](https://cameronrwolfe.substack.com/p/modern-llms-mt-nlg-chinchilla-gopher)]
    
- Specialized LLMs [[link](https://cameronrwolfe.substack.com/p/specialized-llms-chatgpt-lamda-galactica)]
    

Nonetheless, we will briefly go over prior work on LLMs here to provide some important context for understanding PaLM.

### Language Modeling Recap

[

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F91a045da-57be-437d-962c-529ee5bc93fb_1234x828.png)



](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F91a045da-57be-437d-962c-529ee5bc93fb_1234x828.png)

Decoder-only transformer architecture.

Modern language models are simply [decoder-only transformer models](https://cameronrwolfe.substack.com/i/85568430/decoder-only-transformers) (shown above) that are pre-trained using a self-supervised [language modeling objective](https://cameronrwolfe.substack.com/i/85568430/language-modeling) over unlabeled text. This objective samples a sequence of text and trains the language model to accurately predict the next word/token. After performing extensive pre-training, LLMs such as [GPT-3](https://cameronrwolfe.substack.com/p/language-model-scaling-laws-and-gpt#%C2%A7other-useful-details) were found to perform really well in the few-shot learning regime.

**why is this useful?** Put simply, the generic, text-to-text format of LLMs allows them to easily generalize to solving a variety of tasks with minimal adaptation. Instead of fine-tuning models or adding task-specific layers, we can just pre-train a single model extensively and solve a variety of tasks with the same model using few-shot learning. Despite the fact that pre-training such foundation models is incredibly expensive, these approaches hold incredible potential, as a single model can be re-purposed for many applications. This process is referred to as in-context learning; see below.

[

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F5c2794a9-e2ad-4c33-b9aa-dfb56df18bea_1904x1114.png)



](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F5c2794a9-e2ad-4c33-b9aa-dfb56df18bea_1904x1114.png)

(from [4])

**what goes into a good LLM?** Early work on this topic indicated that language model performance should improved smoothly (according to a [power law](https://cameronrwolfe.substack.com/i/88082618/power-laws)) with model scale (i.e., big models perform better). This finding led to the proposal of GPT-3, an LLM of unprecedented scale (175 billion parameters) that achieved breakthrough few-shot learning performance. Subsequent work tried to explore [even larger LLMs](https://cameronrwolfe.substack.com/i/91134599/using-deepspeed-and-megatron-to-train-megatron-turing-nlg-b-a-large-scale-generative-language-model), but these larger models did not lead to further breakthroughs in performance. Rather, we eventually discovered that producing high-performing LLMs requires a combination of larger models with larger pre-training datasets [6].

> _“The amount of training data that is projected to be needed is far beyond what is currently used to train large models, and underscores the importance of dataset collection in addition to engineering improvements that allow for model scale.”_ - from [6]

Enjoy deep learning? Find current research topics difficult to track or understand? Join the >3K subscribers that use Deep (Learning) Focus to better understand AI research by adding your email below!

Subscribe

### Architectural Modifications

Beyond using an improved training framework, PaLM modifies the underlying, decoder-only architecture quite a bit. Most of these changes are adopted from prior work that reveals best practices for maximizing LLM training efficiency and performance.

**SwiGLU activations.** Most LLMs share a similar structure for the [feed-forward neural network](https://cameronrwolfe.substack.com/i/94634004/feed-forward-neural-networks) used within each of their layers. Namely, this network performs two feed-forward transformations (using no bias and applied individually to each token vector in the sequence) with a [Rectified Linear Unit (ReLU) activation](https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html) function in between. However, subsequent work [13] revealed that other choices of the activation function may actually be better.

In particular, PaLM uses the SwiGLU activation function, a combination of Swish [14] and GLU [15] activations. This activation function is given by the equation below.

SwiGLU(x)=Swish(xW)⋅xV

where we define the Swish activation function as

Swish(x)=x⋅Sigmoid(βx)

In other words, SwiGLU is an element-wise product of two [linear transformations](https://mathworld.wolfram.com/LinearTransformation.html) of the input, one of which has had a Swish activation applied to it. Although this activation function requires three matrix multiplications, recent work has found that it provides a performance benefit given a fixed amount of computation. Compared to vanilla activations like ReLU, SwiGLU seems to provide a non-negligible performance improvement [13].

**parallel transformer blocks.** PaLM also uses parallel versions of the [transformer block](https://cameronrwolfe.substack.com/i/85568430/decoder-only-transformers), rather than the normal (serialized) variant. The difference between these two formulations is demonstrated within the illustration below.

[

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F34e6163f-c322-4665-812b-c89e906d6d2d_1886x988.png)



](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F34e6163f-c322-4665-812b-c89e906d6d2d_1886x988.png)

Parallel vs. serialized transformer blocks.

Given a sufficiently large model, using parallel transformer blocks can speed up the training process by 15%. This speedup comes at the cost of slightly degraded performance for smaller LLMs (e.g., an 8 billion parameter model), but full-sized LLMs tend to perform similarly with parallel blocks.

[

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Faa21fa14-e4d3-4cbd-a3a1-bd6874eb512f_1676x1052.png)



](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Faa21fa14-e4d3-4cbd-a3a1-bd6874eb512f_1676x1052.png)

(from [16])

**rotary positional embeddings.** Instead of [absolute](https://cameronrwolfe.substack.com/i/76273144/berts-architecture) or [relative positional embeddings](https://jaketae.github.io/study/relative-positional-encoding/), PaLM utilizes rotary positional embeddings (RoPE), as proposed in [16]. RoPE embeddings incorporate both absolute and relative positioning by:

1. Encoding the absolute position with a rotation matrix
    
2. Incorporating relative position directly into self-attention
    

Intuitively, RoPE finds a middle ground between absolute and relative positional embeddings. Illustrated in the figure above, RoPE consistently outperforms alternative embedding strategies. Plus, it is implemented and easily accessible in common libraries such as HuggingFace.

[RoPE Implementation](https://huggingface.co/docs/transformers/model_doc/roformer)

**multi-query attention.** Finally, PaLM replaces the typical, multi-headed self-attention mechanism with an alternative structure called multi-query attention. Multi-query attention just shares key and value vectors (highlighted in red below) between each of the attention heads, instead of performing a separate projection for each head. This change does not make training any faster, but it does significantly improve the [auto-regressive decoding](https://cameronrwolfe.substack.com/i/85568430/decoder-only-transformers) (i.e., used to perform inference or generation) efficiency of LLMs.

[

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F4134cb5d-82b4-492b-858a-9d4d2bd62267_822x1168.png)



](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F4134cb5d-82b4-492b-858a-9d4d2bd62267_822x1168.png)

Multi-query attention shares key and value projections between attention heads (from [17])

### other useful concepts

- Foundation models and zero/few-shot learning [[link](https://cameronrwolfe.substack.com/i/85568430/creating-foundation-models)]
    
- LLM alignment [[link](https://cameronrwolfe.substack.com/i/93578656/where-do-generic-llms-fall-short)]
    
- Adaptation strategies for LLMs [[link](https://cameronrwolfe.substack.com/i/93578656/refining-llm-behavior)]
    
- A brief progression of LLMs [[link](https://cameronrwolfe.substack.com/i/93578656/what-are-language-models)]
    

# [PaLM: Scaling Language Modeling with Pathways](https://arxiv.org/abs/2204.02311) [1]

Now, we will overview PaLM, a 540 billion parameter dense language model that is efficiently trained using the [Pathways framework](https://blog.google/technology/ai/introducing-pathways-next-generation-ai-architecture/). PaLM is one of the largest dense LLMs that has been trained to date, and its efficient training strategy allows its pre-training process to be performed over a large dataset (>700 billion tokens). This combination of a massive language model with an extensive pre-training corpus leads to some interesting results that we will explore within this section.

### How does PaLM work?

PaLM is a massive LLM that achieves impressive few-shot learning performance via a combination of extensive pre-training (enabled by the efficient Pathways architecture) and some modifications to the underlying model architecture. We will now overview the details of PaLM’s architecture and training regime.

**the model.** PaLM uses a decoder-only transformer with 540 billion parameters. However, this model goes beyond the typical, decoder-only architecture by making a few modifications:

- SwiGLU activations (instead of [ReLU](https://deepai.org/machine-learning-glossary-and-terms/relu)) are used in MLP layers.
    
- Multi-query attention is used in attention layers.
    
- Only parallel transformer blocks are used.
    
- Absolute or relative positional embeddings are replaced with ROPE embeddings.
    

To understand the impact of model scale, three different sizes of PaLM are tested within [1]; see below.

[

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fd8e3b7c2-95cf-4396-8767-fa7fa9336e95_1656x414.png)



](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fd8e3b7c2-95cf-4396-8767-fa7fa9336e95_1656x414.png)

(from [1])

Although power laws suggest that performance should improve smoothly between the models shown above, analysis in [1] finds that _we often see a disproportionate performance improvement when using the largest (540 billion parameter) model_. Larger LLMs provide a surprisingly large benefit when combined with a more extensive pre-training process.

> _“For certain tasks, we observe discontinuous improvements, where scaling from 62B to 540B results in a drastic jump in accuracy compared to scaling from 8B to 62B… This suggests that new capabilities of large LMs can emerge when the model achieves sufficient scale, and that these capabilities continue to emerge beyond previously studied scales.”_ - from [1]

**dataset.** PaLM’s pre-training corpus is comprised of 780B tokens. This is somewhat smaller than the dataset used to train [Chinchilla](https://cameronrwolfe.substack.com/i/91134599/training-compute-optimal-llms) [6] but still larger than that of most prior LLMs; see below.

[

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F416ae668-520c-4b4b-bc83-78b862295461_1686x662.png)



](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F416ae668-520c-4b4b-bc83-78b862295461_1686x662.png)

(from [3])

Creating high-performing LLMs is not just about making the model larger. Recent work on scaling laws for LLMs [6] indicates that performance will increase as a factor of both model size and pre-training corpus size. As such, PaLM has the opportunity to significantly outperform models like MT-NLG (despite being only slightly larger) by using a much larger pre-training corpus.

The pre-training corpus used for PaLM is derived from high-quality webpages, books, wikipedia, news, articles, code, and social media conversations. It contains 22% non-English data (see below) and is inspired by the corpora used to train LaMDA and GLaM [8, 9]. All models are trained for exactly one epoch over this dataset.

[

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F903e263a-462e-4156-ba31-3071c0db025d_1658x572.png)



](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F903e263a-462e-4156-ba31-3071c0db025d_1658x572.png)

(from [1])

Enjoy deep learning? Find current research topics difficult to track or understand? Join the >3K subscribers that use Deep (Learning) Focus to better understand AI research by adding your email below!

Subscribe

**using a large vocabulary.** Given that a non-negligible portion of the pre-training corpus is non-English, the authors also adopt a [SentencePiece tokenizer](https://github.com/google/sentencepiece) with a vocabulary size of 256K. The tokenizer simply takes raw textual input and extracts tokens (i.e., words or sub-words) from the text. This tokenization process is based upon an underlying vocabulary (i.e., set of known tokens), and all tokens extracted from text must be a member of the vocabulary. If a token is not part of the underlying vocabulary, it will be broken into smaller chunks (possibly even characters) until it has been decomposed into valid tokens, or replaced with the generic “`[UNK]`” out of vocabulary token.

Using a small vocabulary would mean that a lot of important tokens would fail to be properly captured, which can damage the LLM’s performance. For multi-lingual models, we typically see that the size of the underlying vocabulary is increased a lot to avoid this effect, as data from multiple languages will utilize a wider range of tokens. PaLM is no different: the authors adopt a larger-than-usual vocabulary size to avoid improperly tokenizing the data and allow more effective learning across multiple languages. To learn more about language models that are trained over many languages, check out the link below.

[Multi-LLMs](https://cameronrwolfe.substack.com/p/many-languages-one-deep-learning)

**training system.** Prior to overviewing the training framework used for PaLM, we need to understand a few concepts related to distributed training. Most importantly, we need to understand the differences between model, data, and pipeline parallelism. Although I’ve explained these concepts [before](https://cameronrwolfe.substack.com/p/language-model-scaling-laws-and-gpt#%C2%A7other-useful-details), the tweet below has a much better (and more concise) description.

[

![Twitter avatar for @rasbt](https://substackcdn.com/image/twitter_name/w_96/rasbt.jpg)

Sebastian Raschka @rasbt

Training deep neural nets on multiple GPUs has become increasingly common in recent years. Dividing the workload allows for larger and more complex models to be trained more quickly. I made a little cheatsheet summarizing the different approaches:

![Image](https://substackcdn.com/image/fetch/w_600,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fpbs.substack.com%2Fmedia%2FFo7qa42aAAAXM30.jpg)



](https://twitter.com/rasbt/status/1625494398778892292?s=20)[

1:57 PM ∙ Feb 14, 2023

---

718Likes107Retweets



](https://twitter.com/rasbt/status/1625494398778892292?s=20)

PaLM is trained on a collection of 6144 TPU chips that are distributed across two [TPU pods](https://cloud.google.com/tpu/docs/training-on-tpu-pods) (i.e., groups of TPUs connect with high-speed network interfaces). At the time of publication, this system was the largest configuration yet described; see below.

[

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F031ea65b-b342-4139-a8ed-9b10fde34b68_2844x952.png)



](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F031ea65b-b342-4139-a8ed-9b10fde34b68_2844x952.png)

(from [1])

Within a pod, communication is very fast between TPUs. But, communication between pods is much slower. Typically, model and data parallelism have bandwidth requirements that are too large for efficient training across TPU pods. Most prior work has dealt with this by either:

1. Limiting training to a single TPU pod [8, 9].
    
2. Using pipeline parallelism, which has lower bandwidth requirements, between pods [7, 10].
    

However, pipelining has many notable drawbacks, such as leaving accelerators idle while emptying or filling the pipeline and having high memory requirements. Using the Pathways system, PaLM is efficiently trained across TPU pods with a combination of model and data parallelism (i.e., no pipeline parallelism). This novel training paradigm enables significant improvements in efficiency.

[

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F9d7f3d6f-ee0f-4ac5-b228-c815ab4100f7_1898x512.png)



](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F9d7f3d6f-ee0f-4ac5-b228-c815ab4100f7_1898x512.png)

(from [1])

For example, PaLM achieves a model [FLOPs](https://stackoverflow.com/questions/58498651/what-is-flops-in-field-of-deep-learning) utilization (i.e., throughput in tokens-per-second divided by theoretical maximum throughput of a system) of 46.2%, while prior systems struggle to surpass utilization of 30%; see above. For more information on the Pathways system and how it achieves such a massive improvement in LLM training efficiency, check out the article below.

[Pathways Architecture](https://blog.google/technology/ai/introducing-pathways-next-generation-ai-architecture/)

### How does PaLM perform?

The analysis provided in [1] goes beyond achieving superior few-shot learning performance. PaLM is shown to effectively handle multiple languages, have improved reasoning capabilities, perform significantly better than smaller models, and even surpass human-level language understanding on certain tasks.

**multi-lingual LLMs.** Prior LLMs (e.g., GPT-3 [4]) had been shown somewhat capable of performing machine translation, especially when translating other languages into English. Across English-centric data pairs and settings, we see that PaLM improves translation performance relative to prior LLMs; see below.

[

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F47bcaf3e-6adf-4099-a1b9-46076e2994b8_1370x1120.png)



](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F47bcaf3e-6adf-4099-a1b9-46076e2994b8_1370x1120.png)

(from [1])

On low resource and non-English centric data, PaLM still performs relatively well, but it is outperformed by existing supervised translation approaches; see above. However, given that non-English settings are not widely considered by prior work, PaLM’s ability to perform relatively well in this setting is impressive. Overall, this analysis shows us that PaLM has improved language translation abilities but still falls short of supervised techniques.

Beyond language translation, we also see that PaLM performs well on multilingual generation tasks. As expected, PaLM’s language generation abilities are best in English, but the model still outperforms prior LLMs on non-English generation. Overall, these results shows us that an LLM’s multilingual capabilities can be improved significantly by making small modifications (i.e., more non-English pre-training data and using a larger vocabulary for our tokenizer).

**surpassing human performance.** The [BIG-bench dataset](https://github.com/google/BIG-bench) contains a collection of 150 tasks with topics including logical reasoning, translation, question answering, mathematics, and more. Relative to prior LLMs, we see that PaLM achieves improved performance on a majority of these tasks; see below.

[

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F1fd670a7-04b7-48de-ade8-cb7c0f204fcd_1432x1640.png)



](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F1fd670a7-04b7-48de-ade8-cb7c0f204fcd_1432x1640.png)

(from [1])

Somewhat more impressively than outperforming prior LLMs, PaLM also surpasses the average performance of humans on most BIG-bench tasks; see below. For some of these tasks, outperforming humans simply indicates that PaLM is capable of memorizing data or reasoning across multiple languages. However, this is not always the case! On other tasks (e.g., cause and effect identification), we see that PaLM seems to have improved language understanding.

[

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F7ee4b3d8-dee0-42b0-a4dc-97a6e02da6ef_1444x750.png)



](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F7ee4b3d8-dee0-42b0-a4dc-97a6e02da6ef_1444x750.png)

(from [1])

**do power laws always hold?** When we break down the performance of PaLM into specific task categories, we see that model scale is especially helpful for certain tasks. For example, on logical sequence tasks (i.e., putting a set of words into a logical order), the largest PaLM model sees a massive improvement in performance relative to smaller models. For other tasks (e.g., mathematical induction), model scale makes little difference.

[

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F6a25a6ec-320e-4c68-ac7b-3b73e844d08b_1554x1282.png)



](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F6a25a6ec-320e-4c68-ac7b-3b73e844d08b_1554x1282.png)

(from [1])

Overall, PaLM’s performance does not always follow a power law with respect to model scale. In some cases, using a larger model causes a massive, unexpected spike in performance, while in others the largest model only performs marginally better than smaller variants; see above.

Enjoy deep learning? Find current research topics difficult to track or understand? Join the >3K subscribers that use Deep (Learning) Focus to better understand AI research by adding your email below!

Subscribe

**learning to reason.** Although language models perform well on many tasks, they notoriously struggle to solve basic reasoning tasks. Many researchers cite this limitation of LLMs as proof of their “shallow” linguistic understanding. However, recent publications have used _chain-of-thought prompting_ (i.e., generating several reasoning “steps” within the LLM before the final output) to improve the reasoning capabilities of LLMs [11, 12]; see below.

[

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F76ddeb3c-90e0-44aa-8657-74c5ea6fcbea_1432x636.png)



](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F76ddeb3c-90e0-44aa-8657-74c5ea6fcbea_1432x636.png)

(from [1])

When evaluating PaLM, authors in [1] find that combining a model of this scale with chain-of-thought prompting is enough to achieve state-of-the-art accuracy on arithmetic and commonsense reasoning tasks. Prior methods leverage domain-specific architectures, fine-tuning, and even task-specific verification modules to solve such reasoning tasks. In comparison, PaLM simply solves these tasks using few-shot, chain-of-thought prompting (and an external calculator module for arithmetic reasoning tasks); see below.

[

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F8316252d-fe5d-43f6-ae34-ec5a16a8ce93_1408x1376.png)



](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F8316252d-fe5d-43f6-ae34-ec5a16a8ce93_1408x1376.png)

(from [1])

Interestingly, we see that the largest PaLM model has much better reasoning abilities compared to smaller variants. Such a finding is interesting given that prior work has observed a mixed (oftentimes negative) impact of scale on reasoning performance. Results in PaLM indicate that model (and data) scale can seemingly benefit reasoning performance given the correct prompting approach.

[

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F0f388369-e932-4c10-8d04-4d2eeae0ebba_1432x536.png)



](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F0f388369-e932-4c10-8d04-4d2eeae0ebba_1432x536.png)

(from [1])

### The PaLM API

If you’re interested in testing out PaLM, then you’re in luck! The API for PaLM was released to select developers within the last few weeks. Many in the AI community saw this release of the PaLM API by Google as a response to the public release of the [ChatGPT API](https://openai.com/blog/introducing-chatgpt-and-whisper-apis) by OpenAI roughly a week before. Read more about the PaLM API release in the article below.

[PaLM API](https://developers.googleblog.com/2023/03/announcing-palm-api-and-makersuite.html)

While training and hosting LLMs is probably too difficult to handle (unless we have a team with hundreds of engineers), we are currently seeing a huge shift towards these tool being made available to developers via APIs. As such, practitioners can get easy access to these incredible models without the hassle or cost of training and hosting them. This lowers the barrier of entry for building applications with these powerful models, which unlocks a world of possibilities! For examples of some awesome applications that can be built, I recommend checking out the [OpenAI cookbook](https://github.com/openai/openai-cookbook).

# Takeaways

Although initial attempts to train LLMs beyond the scale of GPT-3 were somewhat unsuccessful, we see with PaLM that all we need is an efficient training framework that allows for more extensive pre-training. By using the Pathways framework, PaLM can be trained over a much larger dataset compared to prior models of its scale, such as MT-NLG [7]. The resulting LLM has impressive multi-lingual understanding and reasoning capabilities, and we see that increasing the size of the model can oftentimes provide a major benefit. Some important takeaways from PaLM are listed below.

**do power laws always hold?** Numerous publications on the topic of LLMs have shown that a power law exists between LLM performance and various quantities, such as (non-embedding) model parameters, dataset size, amount of training compute and more. Although this trend holds in terms of aggregate performance, the story is a bit more complicated when we examine performance separately with respect to each task. Certain tasks benefit disproportionately from scale, while others don’t see much of a benefit. Thus, scale is generally helpful for LLMs, but the results vary significantly depending on the downstream task being solved.

**should we avoid pipeline parallelism?** One of the main selling points of PaLM is the efficient Pathways training framework with which it is trained. Typically, training over multiple TPU pods or compute nodes requires the use of pipeline parallelism due to limited memory bandwidth. However, by removing pipeline parallelism and allowing training across TPU pods to be performed solely with data and model parallelism, we see that PaLM achieves groundbreaking training efficiency and throughput. These gains to the training framework allow PaLM to be trained over much more data, enabling the model’s impressive performance.

**LLM scale and reasoning.** Prior work on LLMs has oftentimes pointed out their poor reasoning capabilities. In fact, it seemed that the ability of LLMs to perform reasoning tasks degraded with scale. However, we see with PaLM that this is not always the case. If we combine larger LLMs with more pre-training data and the correct prompting approach (i.e., chain-of-thought prompting), we see pretty noticeable improvements in LLM reasoning abilities!

### New to the newsletter?

Hello! I am [Cameron R. Wolfe](https://cameronrwolfe.me/), Director of AI at [Rebuy](https://www.rebuyengine.com/) and PhD student at Rice University. I study the empirical and theoretical foundations of deep learning. This is the Deep (Learning) Focus newsletter, where I help readers build a deeper understanding of topics in deep learning research via understandable overviews of popular papers on that topic. If you like this newsletter, please subscribe, share it with your friends, or follow me on [twitter](https://twitter.com/cwolferesearch)!

Subscribe

### Bibliography

[1] Chowdhery, Aakanksha, et al. "Palm: Scaling language modeling with pathways." _arXiv preprint arXiv:2204.02311_ (2022).

[2] Devlin, Jacob, et al. "Bert: Pre-training of deep bidirectional transformers for language understanding." _arXiv preprint arXiv:1810.04805_ (2018).

[3] Raffel, Colin, et al. "Exploring the limits of transfer learning with a unified text-to-text transformer." _The Journal of Machine Learning Research_ 21.1 (2020): 5485-5551.

[4] Brown, Tom, et al. "Language models are few-shot learners." _Advances in neural information processing systems_ 33 (2020): 1877-1901.

[5] Kaplan, Jared, et al. "Scaling laws for neural language models." _arXiv preprint arXiv:2001.08361_ (2020).

[6] Hoffmann, Jordan, et al. "Training compute-optimal large language models." _arXiv preprint arXiv:2203.15556_ (2022).

[7] Smith, Shaden, et al. "Using deepspeed and megatron to train megatron-turing nlg 530b, a large-scale generative language model." _arXiv preprint arXiv:2201.11990_ (2022).

[8] Thoppilan, Romal, et al. "Lamda: Language models for dialog applications." _arXiv preprint arXiv:2201.08239_ (2022).

[9] Du, Nan, et al. "Glam: Efficient scaling of language models with mixture-of-experts." _International Conference on Machine Learning_. PMLR, 2022.

[10] Rae, Jack W., et al. "Scaling language models: Methods, analysis & insights from training gopher." _arXiv preprint arXiv:2112.11446_ (2021).

[11] Nye, Maxwell, et al. "Show your work: Scratchpads for intermediate computation with language models." _arXiv preprint arXiv:2112.00114_ (2021).

[12] Cobbe, Karl, et al. "Training verifiers to solve math word problems." _arXiv preprint arXiv:2110.14168_ (2021).

[13] Shazeer, Noam. "Glu variants improve transformer." _arXiv preprint arXiv:2002.05202_ (2020).

[14] Ramachandran, Prajit, Barret Zoph, and Quoc V. Le. "Searching for activation functions." _arXiv preprint arXiv:1710.05941_ (2017).

[15] Dauphin, Yann N., et al. "Language modeling with gated convolutional networks." _International conference on machine learning_. PMLR, 2017.

[16] Su, Jianlin, et al. "Roformer: Enhanced transformer with rotary position embedding." _arXiv preprint arXiv:2104.09864_ (2021).

[17] Vaswani, Ashish, et al. "Attention is all you need." _Advances in neural information processing systems_ 30 (2017).

---

#### Subscribe to Deep (Learning) Focus

By Cameron R. Wolfe · Launched 3 years ago

I contextualize and explain important topics in AI research.

Subscribe

By subscribing, I agree to Substack's [Terms of Use](https://substack.com/tos), and acknowledge its [Information Collection Notice](https://substack.com/ccpa#personal-data-collected) and [Privacy Policy](https://substack.com/privacy).

[

![](https://substackcdn.com/image/fetch/w_32,h_32,c_fill,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F1addaf03-9aa5-4f86-a72e-f0fa62852abe_144x144.png)



](https://substack.com/profile/111548585-eda)

[

![](https://substackcdn.com/image/fetch/w_32,h_32,c_fill,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F8b2f49ba-c1b1-47a9-8a34-1e9942f0d11e_992x760.jpeg)



](https://substack.com/profile/112435324-nikola-bulatovic)

[

![](https://substackcdn.com/image/fetch/w_32,h_32,c_fill,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F4411e638-63e3-4ba2-9416-cb8632d59911_1769x1125.png)



](https://substack.com/profile/1255939-siddharth-singh)

[

![](https://substackcdn.com/image/fetch/w_32,h_32,c_fill,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F914ce083-db2b-4f5b-891f-38c5abdf007e_1024x1024.png)



](https://substack.com/profile/119560333-kiran-adimatyam)

[

![](https://substackcdn.com/image/fetch/w_32,h_32,c_fill,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F615de068-b7df-4577-bda0-6f864518d2aa_640x640.jpeg)



](https://substack.com/profile/5549752-taesiri)

12 Likes

[](https://substack.com/note/p-104244919/restacks?utm_source=substack&utm_content=facepile-restacks)

12

[](https://cameronrwolfe.substack.com/p/palm-efficiently-training-massive/comments)

Share

#### Discussion about this post

CommentsRestacks

![](https://substackcdn.com/image/fetch/w_32,h_32,c_fill,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack.com%2Fimg%2Favatars%2Fdefault-light.png)

TopLatestDiscussions

[Decoder-Only Transformers: The Workhorse of Generative LLMs](https://cameronrwolfe.substack.com/p/decoder-only-transformers-the-workhorse)

[Building the world's most influential neural network architecture from scratch...](https://cameronrwolfe.substack.com/p/decoder-only-transformers-the-workhorse)

Mar 4, 2024 • 

[Cameron R. Wolfe, Ph.D.](https://substack.com/@cwolferesearch)

109

[

14

](https://cameronrwolfe.substack.com/p/decoder-only-transformers-the-workhorse/comments)

![](https://substackcdn.com/image/fetch/w_150,h_150,c_fill,f_auto,q_auto:good,fl_progressive:steep,g_center/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F6e3c9db5-400a-49de-a235-e09bc3aa3689_2392x1342.png)

[Mixture-of-Experts (MoE) LLMs](https://cameronrwolfe.substack.com/p/moe-llms)

[Understanding models like DeepSeek, Grok, and Mixtral from the ground up...](https://cameronrwolfe.substack.com/p/moe-llms)

Jan 27 • 

[Cameron R. Wolfe, Ph.D.](https://substack.com/@cwolferesearch)

203

[

10

](https://cameronrwolfe.substack.com/p/moe-llms/comments)

![](https://substackcdn.com/image/fetch/w_150,h_150,c_fill,f_auto,q_auto:good,fl_progressive:steep,g_center/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F3fdf1382-38dc-45fc-a741-b62babfd99c5_2258x1268.png)

[Understanding and Using Supervised Fine-Tuning (SFT) for Language Models](https://cameronrwolfe.substack.com/p/understanding-and-using-supervised)

[Understanding how SFT works from the idea to a working implementation...](https://cameronrwolfe.substack.com/p/understanding-and-using-supervised)

Sep 11, 2023 • 

[Cameron R. Wolfe, Ph.D.](https://substack.com/@cwolferesearch)

52

[

5

](https://cameronrwolfe.substack.com/p/understanding-and-using-supervised/comments)

![](https://substackcdn.com/image/fetch/w_150,h_150,c_fill,f_auto,q_auto:good,fl_progressive:steep,g_center/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F68686a01-2b31-4694-8c04-a562ffd725ad_2210x1244.png)

See all

Ready for more?

Subscribe

© 2025 Cameron R. Wolfe

[Privacy](https://substack.com/privacy) ∙ [Terms](https://substack.com/tos) ∙ [Collection notice](https://substack.com/ccpa#personal-data-collected)

[Start Writing](https://substack.com/signup?utm_source=substack&utm_medium=web&utm_content=footer)[Get the app](https://substack.com/app/app-store-redirect?utm_campaign=app-marketing&utm_content=web-footer-button)

[Substack](https://substack.com/) is the home for great culture