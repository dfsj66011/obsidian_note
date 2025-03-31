
GPT-3 等语言模型彻底改变了 NLP 的现代深度学习应用，获得了广泛的宣传和认可。然而有趣的是，GPT-3 的大部分技术创新都继承自其前身 GPT 和 GPT-2。因此，对 GPT 和 GPT-2 的实用理解有助于更好地掌握当前的 NLP 方法。

GPT 和 GPT-2 模型探索的基本方法很简单。事实上，它可以归结为几个步骤：

1. 使用大量原始文本数据预训练语言模型
2. 调整此预训练模型来解决下游任务

但是描述有点模糊。_语言模型的预训练是如何进行的？我们如何“调整”语言模型来解决不同的任务？_

在本概述中，我们将对语言建模、其在 GPT 和 GPT-2 中的使用以及它如何用于解决不仅仅是生成连贯文本的问题建立基本的了解。尽管由于最近提出了更大、更强大的模型，GPT 和 GPT-2 有些过时，但它们所基于的基本概念仍然与现代深度学习应用高度相关。

### 1、先决条件

GPT 和 GPT-2 背后的基本原理是使用通用的、预训练的语言模型以高精度解决各种语言建模任务。为了充分理解这种方法，我们必须首先介绍一些关于语言模型如何工作以及如何在 GPT 和 GPT-2 中利用它们的基本概念。

#### 1.1 LM

![|600](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F89687ad1-ab5d-4c72-840c-343d7fa26ab2_1854x1030.png)


GPT 模型是使用语言建模目标在未标记文本数据的语料库/数据集上进行预训练的。简而言之，这意味着我们通过 _(i)_ 从数据集中抽取一些文本和 _(ii)_ 训练模型来预测下一个单词来训练模型。这种预训练过程是一种自监督学习，因为只需查看数据集中的下一个单词就可以确定正确的“下一个”单词。

**数学中的语言建模**   要理解语言建模，我们只需要掌握上面概述的基本思想。然而，为了使这一点更加严格，我们可以注意到我们的语料库只是一组标记。我们可以将标记视为数据集中的单个单词，但这并不完全正确。实际上，标记可能是子词甚至字符。

我们将组成我们预训练数据集的这组标记（大小为 $N$）表示如下：$$u=\{u_1, u_2, \cdots,u_N\}$$
给定一个具有参数 $\theta$ 的深度学习模型，语言建模目标试图最大化如下所示的可能性。

![|500](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F3430b67c-2d19-4840-9207-09e68a25d03a_1318x444.png)

简而言之，这个表达式描述了模型在给定`k`前一个标记作为上下文的情况下预测正确的下一个标记的概率。对于任何可能难以理解这个公式的人，请随时查看下面的帮助链接。

- 为什么我们使用对数概率？[[博客](https://chrispiech.github.io/probabilityForComputerScientists/en/part1/log_probabilities/)]
    
- 理解条件概率 [[博客](https://www.hackerearth.com/practice/machine-learning/prerequisites-of-machine-learning/bayes-rules-conditional-probability-chain-rule/tutorial/)]
    

使用语言建模损失（它仅表示我们的模型准确预测序列中的下一个标记的能力！），我们可以按照以下步骤预训练我们的模型的参数，`θ`以最小化损失：

1. 来自预训练语料库的示例文本
    
2. 使用我们的模型预测下一个标记
    
3. 使用随机梯度下降 (SGD) 或任何[其他优化器](https://ruder.io/optimizing-gradient-descent/)来增加下一个正确的标记的概率
    

通过多次重复这种（自我监督）训练过程，我们的模型最终将变得非常擅长语言建模（即预测序列中的下一个标记）。

**什么是语言模型？** 使用这种自监督语言建模目标进行预训练的模型通常称为语言模型 (LM)。LM 的规模越大（即层数、参数等越多），其效率就越高。因此，我们经常会看到这些模型的更大版本（例如 GPT-3 [7]），它们被称为大型语言模型 (LLM)。 

**为什么 LM 有用？**LM 可以通过迭代预测最有可能的下一个标记来生成连贯的文本，从而实现从文本自动完成到聊天机器人等一系列应用。然而，除了生成能力之外，NLP 领域的先前研究表明，LM 预训练对各种任务都非常有益；例如，预训练的词嵌入在下游任务中很有用 [3, 4]，LM 预训练可以提高[LSTM的性能](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)[5]。

除了这些方法之外，GPT 模型还探索了使用 Transformer 进行语言模型预训练 [6]。与顺序模型（例如 LSTM）相比，Transformer _(i)具有_极强的表达能力（即高表示容量、许多参数等）；_(ii)_更适合现代 GPU 的并行计算能力，允许使用更大的模型和更多的数据进行 LM 预训练。这种可扩展性使探索 LLM 成为可能，而 LLM 已经彻底改变了 NLP 应用。





Put simply, this expression characterizes the model’s probability of predicting the correct next token given `k` preceding tokens as context. For anyone who might be struggling to understand this formulation, feel free to check out the helper links below.

- Why do we use log probabilities? [[blog](https://chrispiech.github.io/probabilityForComputerScientists/en/part1/log_probabilities/)]
    
- Understanding conditional probabilities [[blog](https://www.hackerearth.com/practice/machine-learning/prerequisites-of-machine-learning/bayes-rules-conditional-probability-chain-rule/tutorial/)]
    

Using the language modeling loss (which just characterizes our model’s ability to accurately predict the next token in a sequence!), we can follow the procedure below to pre-train our model’s parameters `θ` such that the loss is minimized:

1. Sample text from the pre-training corpus
    
2. Predict the next token with our model
    
3. Use stochastic gradient descent (SGD), or any [other optimizer](https://ruder.io/optimizing-gradient-descent/), to increase the probability of the correct next token
    

By repeating this (self-supervised) training procedure many times, our model will eventually become really good at language modeling (i.e., predicting the next token in a sequence).

**what is a language model?** Models pre-trained using such a self-supervised language modeling objective are commonly referred to as language models (LMs). LMs become more effective as they are scaled up (i.e., more layers, parameters, etc.). Thus, we will often see larger versions of these models (e.g., GPT-3 [7]), which are referred to as large language models (LLMs). 

**why are LMs useful?** LMs can generate coherent text by iteratively predicting the most likely next token, which enables a range of applications from text auto-completion to chatbots. Beyond their generative capabilities, however, prior work in NLP has shown that LM pre-training is incredibly beneficial for a variety tasks; e.g., pre-trained word embeddings are useful in downstream tasks [3, 4] and LM pre-training improves the performance of [LSTMs](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) [5].

Moving beyond such approaches, GPT models explore language model pre-training with transformers [6]. Compared to sequential models (e.g., LSTM), transformers are _(i)_ incredibly expressive (i.e., high representational capacity, many parameters, etc.) and _(ii)_ better suited to the ability of modern GPUs to parallelize computation, allowing LM pre-training to be performed with larger models and more data. Such scalability enables the exploration of LLMs, which have revolutionized NLP applications.

### decoder-only transformers

Both GPT and GPT-2 use a decoder-only transformer architecture. I have [previously summarized](https://cameronrwolfe.substack.com/p/understanding-the-open-pre-trained-transformers-opt-library-193a29c14a15#%C2%A7understanding-opt) this architecture, but I will provide a quick overview here for completeness. To learn more about the transformer architecture, I would recommend briefly reading the explanation linked below.

[Learn about Transformers](https://cameronrwolfe.substack.com/i/74325854/background)

The transformer architecture has two major components: the encoder and the decoder.

[

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F0235fd2f-26f4-47ff-b95e-eddf6a4593b0_782x1152.png)



](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F0235fd2f-26f4-47ff-b95e-eddf6a4593b0_782x1152.png)

(from [6])

A decoder-only architecture removes the following components from the transformer:

- The entire encoder module
    
- All encoder-decoder self-attention modules in the decoder
    

After these components have been removed, each layer of the decoder simply consists of a masked self-attention layer followed by a feed forward neural network. Stacking several of such layers on top of each other forms a deep, decoder-only transformer architecture, such as those used for GPT or GPT-2; see below.

[

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F91a045da-57be-437d-962c-529ee5bc93fb_1234x828.png)



](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F91a045da-57be-437d-962c-529ee5bc93fb_1234x828.png)

Decoder-only transformer architecture

**why the decoder?** The choice of using the decoder architecture (as opposed to the encoder) for LMs is not arbitrary. The masked self-attention layers within the decoder ensure that the model cannot look forward in a sequence when crafting a token’s representation. In contrast, bidirectional self-attention (as used in the encoder) allows each token’s representation to be adapted based on all other tokens within a sequence.

[Learn about Self-Attention](https://cameronrwolfe.substack.com/i/76273144/self-attention)

Masked self-attention is required for language modeling because we should not be able to look forward in the sentence while predicting the next token. Using masked self-attention yields an autoregressive architecture (i.e., meaning that the model’s output at time `t` is used as input at time `t+1`) that can continually predict the next token in a sequence; see below.

[

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F83ac8a81-a3b8-42e8-bc53-6ab3a505effc_1880x1010.png)



](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F83ac8a81-a3b8-42e8-bc53-6ab3a505effc_1880x1010.png)

Autoregressive output from a decoder-only transformer architecture

For tasks that do not require masked self-attention (e.g., sentence classification, tagging, etc.), however, we should remember that using bidirectional self-attention is really beneficial; see the link below for more details.

[Learn about BERT](https://cameronrwolfe.substack.com/p/language-understanding-with-bert)

### creating foundation models

Now that we have a basic understanding of language modeling and relevant architectures, we can understand the inspiration behind the GPT LMs, which begins with the following observations: 

- Unlabeled text corpora are largely abundant
    
- Labeled data is scarce 
    

For most deep learning systems, a lot of labeled data is needed to perform discriminative language understanding tasks. _Current deep learning systems are narrow experts_. The model is simply trained over a large, supervised dataset such that it learns to accurately perform a specific task; see below.

[

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F51e535d4-8edb-4218-9c41-3298fca62643_1624x1112.png)



](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F51e535d4-8edb-4218-9c41-3298fca62643_1624x1112.png)

Most deep learning models use supervised (or labeled) datasets to learn how to accurately perform a single, specific task

Though commonly used, this approach suffers a few major limitations:

1. Some domains do not have much labeled data
    
2. We have to train a new model for every task that we want to solve (and training deep learning models is expensive!)
    

**foundation models.** GPT and GPT-2 move away from the paradigm of narrow experts within deep learning. Rather than train a new model for every application, we can pre-train a single LM, then somehow adapt this model to solve numerous tasks. Generic models that are used to solve many tasks are referred to as foundation models.

[More on Foundation Models](https://crfm.stanford.edu/)

This approach mitigates problems with data scarcity by pre-training over a large, diverse dataset. Additionally, these models can be reused or adapted to solve other tasks, allowing us to avoid constantly training new models. One approach for adapting a foundation model to a downstream task is to perform fine-tuning (i.e., more training) over a supervised dataset. More recently, however, the go-to approach is via zero or few-shot inference.

**zero/few-shot inference via prompting.** The GPT models receive text as input and produce text as output. We can exploit this generic input-output structure by providing inputs like the following:

- “Translate this sentence to English: `<sentence> =>`”
    
- “Summarize the following document: `<document> =>`”.
    

These task-solving “prompts” enable zero-shot (i.e., without seeing examples of correct output) inference with LMs. Given these prompts, the most appropriate output from the LM should solve the task (e.g., translating to English or summarizing a document)! To perform few-shot inference, we can construct a similar prompt with examples of correct output provided at the start; see below.

[

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F20c74320-2996-47ed-9507-08e1967a36d9_736x1262.png)



](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F20c74320-2996-47ed-9507-08e1967a36d9_736x1262.png)

Zero, one, and few-shot inference with LMs (from [7])

Thanks for reading Deep (Learning) Focus! Subscribe for free to receive new posts and support my work.

Subscribe

# Publications

We will now overview the details of GPT and GPT-2. Published by researchers at [OpenAI](https://openai.com/), these models pioneered the use of generic LMs for solving downstream tasks. They laid the foundation for breakthrough advancements like GPT-3. The main differentiator between these models is simply the size of the underlying LM.

### **[Improving Language Understanding by Generative Pre-Training](https://www.cs.ubc.ca/~amuham01/LING530/papers/radford2018improving.pdf) (GPT) [1]**

GPT is a general purpose language understanding model that is trained in two phases: pre-training and fine-tuning.

[

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F47d302b2-a7e6-4ee9-bf10-7c161a9e4057_342x648.png)



](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F47d302b2-a7e6-4ee9-bf10-7c161a9e4057_342x648.png)

GPT architecture (from [1])

GPT uses a 12-layer, decoder-only transformer architecture that matches the original transformer decoder [6] (aside from using learnable positional embeddings); see the figure above. GPT first performs language model pre-training over the [BooksCorpus](https://yknzhu.wixsite.com/mbweb) dataset, then is separately fine-tuned (in a supervised manner) on a variety of discriminative language understanding tasks.

[

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F60d46502-4340-48d7-8db6-057993f82060_1622x816.png)



](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F60d46502-4340-48d7-8db6-057993f82060_1622x816.png)

(from [1])

Instead of modifying GPT’s architecture to solve different tasks, we provide input in a task-specific structure, then pass the model’s output to a separate classification layer. For example, on entailment tasks, we concatenate the input sentences, separate them with a special delimiter, provide this input to GPT, then pass GPT’s output to a separate classification layer. Fine-tuning GPT with different supervised tasks is explained further in Section 3.3 of [1] and illustrated above. 

[

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F636f5316-9d99-4b79-8c91-e4b3c76da2ef_1600x332.png)



](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F636f5316-9d99-4b79-8c91-e4b3c76da2ef_1600x332.png)

(from [1])

GPT is evaluated on a wide variety of tasks listed above. The authors find that pre-training GPT on a corpus with long spans of contiguous text (as opposed to individual, shuffled sentences) is essential (this finding was also verified by more recent work [9]). Across experimental settings, we see that GPT achieves state-of-the-art performance on 9 of the 12 tasks and even consistently outperforms model ensembles; see below.

[

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2Fb925c519-b72b-40a7-8d32-2f5c8b3804dc_1968x1078.png)



](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2Fb925c519-b72b-40a7-8d32-2f5c8b3804dc_1968x1078.png)

(from [1])

From these experiments, we learn that general purpose LMs understand linguistic concepts relatively well and are capable of learning complex patterns (e.g., long-term dependencies, linguistic ambiguity, etc.) within textual data. Without using any task-specific architectures or modifications, GPT outperforms numerous baselines by a large margin, including many specialized solutions for solving individual tasks.

### **[Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) (GPT-2) [2]**

The proposal of GPT-2 [2] follows a similar pattern as its predecessor. The model is pre-trained using a language modeling objective, but it performs no fine-tuning, choosing to solve downstream tasks in a zero-shot manner instead. Put simply, GPT-2 performs multi-task learning by:

1. Pre-training a generic LM over raw textual data
    
2. Using textual “prompts” to perform zero-shot inference on a variety of tasks
    

Pre-training is performed over a custom WebText dataset that is constructed by scraping popular links from Reddit, and four different sizes of LMs are tested. The smallest model matches the size of GPT [1] and the largest model is GPT-2; see below.

[

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F80a786e0-35eb-4216-be14-7e32b88f5ff8_1186x460.png)



](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F80a786e0-35eb-4216-be14-7e32b88f5ff8_1186x460.png)

(from [2])

The model architecture is identical to GPT, barring a few minor differences (e.g., different weight initialization, larger vocabulary, longer input sequence, etc.). Despite the size of these LMs, they are found to underfit the WebText dataset during pre-training, indicating that larger LMs would perform even better.

GPT-2 is evaluated on several tasks (i.e., language modeling, question answering, translation, etc.), where it achieves promising (but not always state-of-the-art) results. For example, in the table below we see that GPT-2 performs well on language modeling and reading comprehension tasks but falls far short of baselines for summarization and question answering.

[

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F206e161b-36a7-40a1-8dd7-06d73725deb9_1982x586.png)



](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F206e161b-36a7-40a1-8dd7-06d73725deb9_1982x586.png)

(from [2])

Even though the performance isn’t great, we need to remember that _GPT-2 performs no fine-tuning to solve any of these tasks_. All of these results are achieved via zero-shot inference, which makes GPT’s competitive performance on certain tasks pretty impressive.

Interestingly, zero-shot performance consistently improves with the size of the underlying LM, indicating that increasing an LM’s size/capacity improves its ability to learn relevant features during pre-training; see below.

[

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F2a242e94-54ba-44e1-9f99-663a8330a67d_1966x800.png)



](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F2a242e94-54ba-44e1-9f99-663a8330a67d_1966x800.png)

(from [2])

Pre-training and fine-tuning is an effective transfer learning paradigm, but GPT-2 shows us that easier, more general methods of transfer exist. Given that they are pre-trained over a sufficiently-large corpus, LMs seem to be capable of learning downstream tasks even without any architectural or parameter modifications. Although GPT-2’s performance is not impressive, the authors indicate that larger LMs will be much better.

> “… a language model with sufficient capacity will begin to learn to infer and perform the tasks demonstrated in natural language sequences in order to better predict them, regardless of their method of procurement.” -from [2]

Thanks for reading Deep (Learning) Focus! Subscribe for free to receive new posts and support my work.

Subscribe

# Takeaways

GPT and GPT-2 taught us a lot about deep learning. Though their effectiveness on downstream tasks was not incredibly impressive from an accuracy perspective, they provided a glimpse into the incredible potential of LMs as foundation models and laid the methodological foundation for the emergence of LLMs like GPT-3. The impact of these models is far-reaching, but I’ve tried to summarize some of the most useful takeaways and ideas from research on GPT and GPT-2 below.

**language model pre-training is awesome.** Transformers, due to their efficient utilization of compute, enable language model pre-training to be performed at a massive scale. The representations learned during this pre-training process allow pre-trained LMs to generalize well to solving other tasks. Put simply, _LMs aren’t just good at language modeling_ – they can solve other tasks too!

**size matters.** As we see in the transition from GPT to GPT-2, increasing the size of the pre-trained LM increases the quality of the learned representations; e.g., GPT-2 far outperforms GPT in terms of zero/few-shot inference. This trend became more pronounced after the release of the (larger) GPT-3 model [7].

**we should leverage foundation models.** Most deep learning models are trained to accomplish a single, narrow task. In many cases, however, we can benefit from _(i)_ pre-training a larger model via self-supervised learning on unlabeled data and _(ii)_ adapting this model to solve many tasks. Such repurposing of large, foundation models is computationally efficient (i.e., computation is shared across many tasks) and not specific to LMs. We can train foundation models for domains like computer vision too [8]!

### code and resources

For those interested in trying out applications with GPT-2, the code is [publicly available](https://github.com/openai/gpt-2)! However, pre-training such a model is quite computationally expensive. A better approach would be to [download a pre-trained language model](https://huggingface.co/models?sort=downloads) and either [fine-tune](https://huggingface.co/docs/transformers/v4.14.1/en/training) it or perform zero/few-shot inference (e.g., by using the demo below).

[GPT-2 LM Demo](https://transformer.huggingface.co/doc/gpt2-large)

# New to the newsletter?

Hello! I am [Cameron R. Wolfe](https://cameronrwolfe.me/), a research scientist at [Alegion](https://www.alegion.com/) and PhD student at Rice University. I study the empirical and theoretical foundations of deep learning. This is the Deep (Learning) Focus newsletter, where I pick a single, bi-weekly topic in deep learning research, provide an understanding of relevant background information, then overview a handful of popular papers on the topic. If you like this newsletter, please subscribe, share it with your friends, or follow me on [twitter](https://twitter.com/cwolferesearch)!

Subscribe

### Bibliography

[1] Radford, Alec, et al. "Improving language understanding by generative pre-training." (2018). 

[2] Radford, Alec, et al. "Language Models are Unsupervised Multitask Learners."

[3] Pennington, Jeffrey, Richard Socher, and Christopher D. Manning. "Glove: Global vectors for word representation." Proceedings of the 2014 conference on empirical methods in natural language processing (EMNLP). 2014.

[4] Conneau, Alexis, et al. "Supervised learning of universal sentence representations from natural language inference data." arXiv preprint arXiv:1705.02364 (2017).

[5] Howard, Jeremy, and Sebastian Ruder. "Universal language model fine-tuning for text classification." arXiv preprint arXiv:1801.06146 (2018).

[6] Vaswani, Ashish, et al. "Attention is all you need." Advances in neural information processing systems 30 (2017).

[7] Brown, Tom, et al. "Language models are few-shot learners." Advances in neural information processing systems 33 (2020): 1877-1901.

[8] Yuan, Lu, et al. "Florence: A new foundation model for computer vision." arXiv preprint arXiv:2111.11432 (2021).

[9] Krishna, Kundan, et al. "Downstream Datasets Make Surprisingly Good Pretraining Corpora." _arXiv preprint arXiv:2209.14389_ (2022).

---

#### Subscribe to Deep (Learning) Focus

By Cameron R. Wolfe · Launched 3 years ago

I contextualize and explain important topics in AI research.

Subscribe

By subscribing, I agree to Substack's [Terms of Use](https://substack.com/tos), and acknowledge its [Information Collection Notice](https://substack.com/ccpa#personal-data-collected) and [Privacy Policy](https://substack.com/privacy).

[

![](https://substackcdn.com/image/fetch/w_32,h_32,c_fill,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F24fd3e31-f787-42fd-a656-a4d67087ca04_144x144.png)



](https://substack.com/profile/5942914-arnaldo-gualberto)

[

![](https://substackcdn.com/image/fetch/w_32,h_32,c_fill,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F400d64cd-b630-4cc0-b4ea-489892550d99_1287x1284.jpeg)



](https://substack.com/profile/32625339-jared-kirby)

[

![](https://substackcdn.com/image/fetch/w_32,h_32,c_fill,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F1addaf03-9aa5-4f86-a72e-f0fa62852abe_144x144.png)



](https://substack.com/profile/111548585-eda)

[

![](https://substackcdn.com/image/fetch/w_32,h_32,c_fill,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2Fc628c9aa-3490-4fee-a7cb-d027356b826d_400x400.jpeg)



](https://substack.com/profile/18767028-daniel-duma)

[

![](https://substackcdn.com/image/fetch/w_32,h_32,c_fill,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F55557eaf-bb54-4169-82c3-ec66bfc61414_512x512.png)



](https://substack.com/profile/36192525-tyler-corderman)

32 Likes∙

[1 Restack](https://substack.com/note/p-85568430/restacks?utm_source=substack&utm_content=facepile-restacks)

32

[](https://cameronrwolfe.substack.com/p/language-models-gpt-and-gpt-2/comments)

1

Share

#### Discussion about this post

CommentsRestacks

![](https://substackcdn.com/image/fetch/w_32,h_32,c_fill,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack.com%2Fimg%2Favatars%2Fdefault-light.png)

TopLatestDiscussions

[Decoder-Only Transformers: The Workhorse of Generative LLMs](https://cameronrwolfe.substack.com/p/decoder-only-transformers-the-workhorse)

[Building the world's most influential neural network architecture from scratch...](https://cameronrwolfe.substack.com/p/decoder-only-transformers-the-workhorse)

Mar 4, 2024 • 

[Cameron R. Wolfe, Ph.D.](https://substack.com/@cwolferesearch)

106

[

14

](https://cameronrwolfe.substack.com/p/decoder-only-transformers-the-workhorse/comments)

![](https://substackcdn.com/image/fetch/w_320,h_213,c_fill,f_auto,q_auto:good,fl_progressive:steep,g_center/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F6e3c9db5-400a-49de-a235-e09bc3aa3689_2392x1342.png)

[Mixture-of-Experts (MoE) LLMs](https://cameronrwolfe.substack.com/p/moe-llms)

[Understanding models like DeepSeek, Grok, and Mixtral from the ground up...](https://cameronrwolfe.substack.com/p/moe-llms)

Jan 27 • 

[Cameron R. Wolfe, Ph.D.](https://substack.com/@cwolferesearch)

197

[

10

](https://cameronrwolfe.substack.com/p/moe-llms/comments)

![](https://substackcdn.com/image/fetch/w_320,h_213,c_fill,f_auto,q_auto:good,fl_progressive:steep,g_center/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F3fdf1382-38dc-45fc-a741-b62babfd99c5_2258x1268.png)

[Understanding and Using Supervised Fine-Tuning (SFT) for Language Models](https://cameronrwolfe.substack.com/p/understanding-and-using-supervised)

[Understanding how SFT works from the idea to a working implementation...](https://cameronrwolfe.substack.com/p/understanding-and-using-supervised)

Sep 11, 2023 • 

[Cameron R. Wolfe, Ph.D.](https://substack.com/@cwolferesearch)

50

[

5

](https://cameronrwolfe.substack.com/p/understanding-and-using-supervised/comments)

![](https://substackcdn.com/image/fetch/w_320,h_213,c_fill,f_auto,q_auto:good,fl_progressive:steep,g_center/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F68686a01-2b31-4694-8c04-a562ffd725ad_2210x1244.png)

See all

Ready for more?

Subscribe

© 2025 Cameron R. Wolfe

[Privacy](https://substack.com/privacy) ∙ [Terms](https://substack.com/tos) ∙ [Collection notice](https://substack.com/ccpa#personal-data-collected)

[Start Writing](https://substack.com/signup?utm_source=substack&utm_medium=web&utm_content=footer)[Get the app](https://substack.com/app/app-store-redirect?utm_campaign=app-marketing&utm_content=web-footer-button)

[Substack](https://substack.com/) is the home for great culture


----

# 25、扩展定律和 GPT-3

![Deep (Learning) Focus](https://substackcdn.com/image/fetch/w_80,h_80,c_fill,f_auto,q_auto:good,fl_progressive:steep,g_auto/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2Fab9b43fb-52d5-40da-995d-5b7cd3f91064_896x896.png)

# [Deep (Learning) Focus](https://cameronrwolfe.substack.com/)

SubscribeSign in

# Language Model Scaling Laws and GPT-3

### Understanding why LLMs like GPT-3 work so well...

[

![](https://substackcdn.com/image/fetch/w_36,h_36,c_fill,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F69aba7df-b571-4609-aa47-fc2d031c11b8_1242x1595.jpeg)



](https://substack.com/@cwolferesearch)

[Cameron R. Wolfe, Ph.D.](https://substack.com/@cwolferesearch)

Dec 05, 2022

11

[](https://cameronrwolfe.substack.com/p/language-model-scaling-laws-and-gpt/comments)

Share

[

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F9ac27842-5db9-4cef-be53-85abbd7f37ac_1231x638.png)



](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F9ac27842-5db9-4cef-be53-85abbd7f37ac_1231x638.png)

This newsletter is supported by [Alegion](https://www.alegion.com/). At Alegion, I work on a range of problems from online learning to diffusion models. Feel free to check out our [data annotation platform](https://www.alegion.com/products) or [contact me](https://cameronrwolfe.me/) about potential collaboration/opportunities!

If you like this newsletter, please subscribe, share it, or follow me on [twitter](https://twitter.com/cwolferesearch). Thank you for your support!

Subscribe

---

[

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F2d3563e0-505c-4230-ab15-bf09355eb586_1818x1170.png)



](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F2d3563e0-505c-4230-ab15-bf09355eb586_1818x1170.png)

(from [1] and [2])

Language models (LMs) are incredibly generic–they take text as input and produce text as output. Recent research has revealed that this generic text-to-text structure can be exploited to solve a variety of tasks without task-specific adaptation (i.e., no fine-tuning or architectural modifications) by using prompting techniques to perform accurate zero and few-shot inference. Put simply, we can pre-train the LM over a large, unlabeled text corpus (using a language modeling objective), then ask the LM via textual prompts to solve a problems. In this way, the pre-trained model can easily be repurposed for solving different problems.

Although LMs hold incredible potential as task-agnostic foundation models, initial attempts at transferring pre-trained LMs to solving downstream tasks (e.g., GPT and GPT-2 [4, 5]) did not work well. Within this overview, we will learn how recent research has built upon these initial attempts and created LMs that achieve much better task-agnostic performance. The key finding within this line of work is that _LMs become much more powerful as you scale them up._

More specifically, we will learn that large LMs (LLMs) are _(i)_ more sample efficient than their smaller counterparts and _(ii)_ more capable of task-agnostic transfer to downstream tasks. Interestingly, the performance of these LLMs follows predictable trends with respect to various factors (e.g., model size and the amount of training data). The empirical observation of these trends eventually led to the creation of GPT-3, a 175 billion parameter LLM that far surpasses the task-agnostic performance of its predecessors and even outperforms state-of-the-art, supervised deep learning techniques on certain tasks.

# Background

Most prerequisite information needed to understand LMs has already been covered in one of my prior posts. These prerequisites include the language modeling objective, decoder-only transformer models, and how these ideas can be combined to generate powerful foundation models. Check out the link below to learn more.

[LM Prerequisites](https://cameronrwolfe.substack.com/i/85568430/prerequisites-for-gpt)

I will give a quick overview of these ideas here, as well as explain a few additional concepts that are useful for understanding LLMs like GPT-3.

### language modeling at a glance

Modern LMs use generic pre-training procedures to solve a wide variety of tasks without the need for downstream adaptation (i.e., no architectural modifications, fine-tuning etc.). Using a large corpus of unlabeled text, we pre-train our LM using a language modeling objective that _(i)_ samples some text from our corpus and _(ii)_ tries to predict the next word that occurs. This is a form of self-supervised learning, as we can always find the ground truth next word by simply looking at the data in our corpus; see below.

[

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F26fd2d00-2b35-460f-83c1-4699cc5f23f6_1854x1030.png)



](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F26fd2d00-2b35-460f-83c1-4699cc5f23f6_1854x1030.png)

The language model pre-training process

**architecture.** Modern LMs use decoder-only transformer architectures, which apply a sequence of layers consisting of masked self-attention and feed forward transformations to the model’s input. Masked self-attention is used instead of [bidirectional self-attention](https://cameronrwolfe.substack.com/i/76273144/self-attention), as it prevents the model from “looking forward” in a sequence to discover the next word. 

Beyond these decoder-only layers, the LM architecture contains embedding layers that store vectors corresponding to all possible tokens within a fixed-size vocabulary. Using these embedding layers, raw text can be converted into a model-ingestible input matrix as follows:

1. [Tokenize](https://towardsdatascience.com/how-to-build-a-wordpiece-tokenizer-for-bert-f505d97dddbb) raw text into individual tokens (i.e., words or sub-words)
    
2. Lookup the corresponding embedding vector for each input token
    
3. Concatenate token embeddings, forming a matrix/sequence of token vectors
    
4. Add position (and other) embeddings to each token
    

See the figure below for an illustration of this process.

[

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2Fc2eff169-d176-4ce5-9993-25df080dd039_1504x742.png)



](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2Fc2eff169-d176-4ce5-9993-25df080dd039_1504x742.png)

Converting raw text into model-compatible matrices of token embeddings

The differentiation between embedding and decoder-only layers within the LM is important to understand. For example, some later work in this overview will study the number of parameters within the underlying LM by excluding parameters in the embedding layer and only counting those contained within decoder-only layers.

**adaptation.** By pre-training LMs over a large corpus, we obtain a model that can accurately predict the next token given a sequence of tokens as context. But, _how do we use such a model to solve language understanding tasks like sentence classification and language translation?_ 

For modern LMs, the answer to this question is actually quite simple–we don’t change the LM at all. Instead, we exploit the generic nature of the model’s text-to-text input-output structure by providing textual “prompts” to the model, such as:

- “Translate this sentence to English: <sentence> =>”
    
- “Summarize the following document: <document> =>”.
    

Given these problem-solving prompts, a good LM should output a textual sequence that solves the problem for us! For problems in which we must choose from a fixed set of solutions (i.e., multiple choice or classification) instead of just generating text, we can use the LM to measure the probability of generating each potential solution and choose the most probable solution.

[Examples of Prompts](https://www.buildgpt3.com/)

**main takeaway.** The crux of modern LLMs is that we can use language model pre-training as a tool for creating generic foundation models that solve various problems without the need to adapt or fine-tune the model. Although prior LMs like GPT and GPT-2 [4, 5] perform poorly compared to fine-tuned or supervised language understanding techniques, such a learning framework is quite promising and—as we will see with GPT-3–can even perform quite well when the underlying LM becomes much larger.

### power laws

This overview will contain several references to the idea of [power laws](https://en.wikipedia.org/wiki/Power_law). For example, a paper may make a statement like the following:

> “The LM’s test loss varies as a power law with respect to the number of model parameters”. 

This sentence simply tells us that a relationship exists between two quantities–the loss and the number of model parameters–such that a change in one quantity produces a relative, [scale-invariant](http://felix.physics.sunysb.edu/~allen/540-05/scaling.html#:~:text=scale%20invariance%20of%20power%20law%20functions&text=The%20function%20y%3Dxp,by%20a%20scale%20factor%20a.) change in the other.

To make this a bit more concrete, a power law is expressed via the following equation.

[

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2Fddaa159e-1468-41a9-872a-7844a36e09ef_120x34.png)



](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2Fddaa159e-1468-41a9-872a-7844a36e09ef_120x34.png)

Here, the two quantities we study are `x` and `y`, while `a` and `p` dictate the shape/behavior of the power law between these quantities. Plotting this power law (with `a = 1`, `p = 0.5`, and `0 < x, y < 1`) yields the illustration below, where converting both axes to a log scale produces a signature linear trend that is characteristic of power laws.

[

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2Fc3b410c6-03f9-4214-8d6a-074b1cbbf6ec_800x300.png)



](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2Fc3b410c6-03f9-4214-8d6a-074b1cbbf6ec_800x300.png)

Depiction of a basic power law relationship between two quantities

Power laws simply tell us that one quantity varies as a power of another quantity. The work we will see in this overview considers an inverse version of a power law, as shown below.

[

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F56b20fb6-20b4-4fbf-8b9f-5c7d149862b3_188x89.png)



](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F56b20fb6-20b4-4fbf-8b9f-5c7d149862b3_188x89.png)

Notably, this is the same equation as before with a negative exponent for `p`. This negative exponent yields the graph shown below, where one quantity decreases as the other increases.

[

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2Faeb36a47-1a10-41c1-ad10-dd0a5a7df7d2_800x300.png)



](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2Faeb36a47-1a10-41c1-ad10-dd0a5a7df7d2_800x300.png)

Depiction of an inverse power law between two quantities

We will encounter power laws that resemble the figure above within our analysis of LMs. Namely, the LM loss tends to decrease according to a power law with respect to several different factors, such as the model or dataset size. We will expand upon this more in later sections.

### other useful details

In addition to the core ideas behind language modeling, there are a few additional concepts that might be helpful to know moving forward.

**distributed training.** The main idea of the papers within this overview is scaling up models like GPT and GPT-2 [4, 5] to make them better. As our models get bigger and bigger, however, training becomes more difficult due to an increase in computational and memory overhead. To help with this, we can leverage distributed training techniques, which use more hardware (i.e., more servers/GPUs) to make large-scale training processes more tractable and efficient.

There are a couple of different ways to distribute the training process for neural networks. One of these techniques is _data parallel training_, in which we:

1. Take a large mini-batch
    
2. Split this mini-batch into several, smaller sub-batches
    
3. Perform the computation related to each sub-batch in parallel on a different GPU
    
4. Accumulate the sub-batch results from each GPU into a centralized model update
    

Such an approach enables improved training efficiency by parallelizing model computation over a large mini-batch across several GPUs.

Somewhat differently, we can perform _model-parallel training_, which splits the model itself (i.e., instead of the mini-batch) across multiple GPUs. For example, we can send each layer of a model–or even smaller portions of each layer–to a separate GPU. Practically, this means that the forward pass is spread across several devices or GPUs that each contain a small portion of the underlying model. Such an approach enables larger models to be trained (i.e., because each GPU only stores a small portion of the model!) and can yield improvements in training efficiency via smart pipelining and parallelization of the model’s forward pass.

For the purposes of this overview, we just need to know that we can leverage distribution across many GPUs to make LLM training more tractable and efficient. Data and model parallel training are examples of popular distributed training techniques. Many considerations and alternative methodologies for distributed training exist–this is an entire field of study within deep learning that yields a lot of awesome, practical results.

To learn more, I would recommend checking out the following articles:

- Data and Model Parallel Training [[blog](https://analyticsindiamag.com/data-parallelism-vs-model-parallelism-how-do-they-differ-in-distributed-training/)]
    
- Making LM Training More Efficient [[LM blog](https://www.mosaicml.com/blog/billion-parameter-gpt-training-made-easy)] [[LLM blog](https://www.mosaicml.com/blog/gpt-3-quality-for-500k)]
    

**critical batch size.** Given that using large batches for data parallel training can benefit computational efficiency, we should just make our batches as big as possible, right? Well, this isn’t quite correct, as _(i)_ larger batches might deteriorate model performance and _(ii)_ increasing the batch size increases compute costs and requires extra hardware. Put simply, increasing the batch size too much has diminishing returns; see below.

[

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2Fd9f1d8e5-35e1-4edd-a463-ed5a9d88187f_2338x1254.png)



](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2Fd9f1d8e5-35e1-4edd-a463-ed5a9d88187f_2338x1254.png)

(from [3])

With this in mind, we might begin to wonder: _what’s the best batch size to use?_ This question was answered empirically with the proposal of the critical batch size in [3]. This work uses a metric called the _gradient noise scale_ to estimate the largest useful batch size across a variety of domains. Beyond this critical batch size, we start to see diminishing returns in terms of performance and compute efficiency. Because adopting different batch sizes can impact the efficiency and quality of training, some work–as we will see in this overview–adopts the critical batch size as a standard practice for resource efficient training.

**Beam search.** LM’s solve problems by outputting a textual sequence in response to a prompt. These sequences can be generated autoregressively by continually predicting the next word, adding this word to the input prompt, predicting another word, and so on; see the figure below.

[

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2Ff759c4d3-def7-4904-b7c7-210e33d4ae08_1880x1010.png)



](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2Ff759c4d3-def7-4904-b7c7-210e33d4ae08_1880x1010.png)

Generating autoregressive output with a language model

However, the greedy approach of continually predicting the most probable next word is not optimal! This is because the probability of a sequence of tokens (assuming each token is generated [independently](https://www.mathsisfun.com/data/probability-events-independent.html)) is the product of each word’s conditional probability given preceding tokens (i.e., due to the [chain rule](https://www.hackerearth.com/practice/machine-learning/prerequisites-of-machine-learning/bayes-rules-conditional-probability-chain-rule/tutorial/) of probability). Greedily choosing the most probable next token might not maximize this probability; e.g., initially choosing a low probability token might subsequently lead to higher probability tokens in the rest of the sequence.

Instead of testing all combinations of possible output tokens to find the best output sequence, we can find an approximate solution with _beam search_. The idea behind beam search is simple: instead of choosing the most probable next token at each step, choose the top-k most probable generations, maintain a list of possible output sequences based on these top choices, then select the most probable of these sequences at the end.

[

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F95e07e0d-4cd6-42f8-b0ed-d933fb49015e_1118x700.png)



](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F95e07e0d-4cd6-42f8-b0ed-d933fb49015e_1118x700.png)

Beam search with k=2

Thanks for reading Deep (Learning) Focus! Subscribe for free to receive new posts and support my work.

Subscribe

# Publications

We will now overview publications that predict [1] and empirically validate [2] the incredible practical utility of LLMs like GPT-3. From these publications, we will gain a better understanding of why LLMs are so powerful and see extensive analysis of their performance in practical applications.

### [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361) [1]

GPT and GPT-2 [4, 5] showed us that LMs have incredible potential as generic foundation models, but their performance when transferring to downstream tasks still leaves a lot to be desired. Thus, we might begin to ask: _how can we make these models better?_

In [1], authors study one potential direction for making LMs more powerful–scaling them up. In particular, they train a bunch of decoder-only LMs and analyze their test loss (i.e., cross-entropy language modeling loss over a hold-out test set) as a function of several factors, including:

- Model size
    
- Amount of data
    
- Amount of training compute
    
- Batch size 
    
- Architectural details (i.e., model width/depth, number of attention heads, etc.)
    
- Context length (i.e., number of tokens used to predict the next token)
    

This analysis reveals several fundamental properties of LM training behavior. For example, tweaking architectural details has minimal impact on LM performance if the total number of parameters is fixed. However, the LM’s test loss follows a power law with respect to model size, data size, and the amount of training compute across several orders of magnitude; see below.

[

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2Fda994e00-8afd-4b11-b29e-4a6d074f2a9a_2344x976.png)



](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2Fda994e00-8afd-4b11-b29e-4a6d074f2a9a_2344x976.png)

(from [1])

To make this a bit more clear, the authors in [1] consider three main factors: model size (N), data size (D), and the amount of training compute (C). To study scaling behavior with respect to any one of these factors, we _(i)_ make sure that the other two factors are sufficiently large (i.e., so they aren’t a bottleneck to performance), then _(ii)_ measure the LM’s test loss over a wide range of values for the factor we are studying. For example, to study the scaling properties of C, we make sure the model and dataset are sufficiently large, then measure LLM performance across different settings of C. We will now consider each of these factors individually.

**model size.** To study scaling properties with respect to model size, authors train different LM to convergence over the full dataset from [1]–WebText2, an extended version WebTest from GPT-2 [2] that is ~10X larger. Then, by adopting several LMs with different numbers of total parameters, we can obtain the figure shown below.

[

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F2af05f5c-d140-47bc-96d0-25469c55f856_2048x830.png)



](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F2af05f5c-d140-47bc-96d0-25469c55f856_2048x830.png)

(from [1])

By plotting the LM’s test loss as a function of the total number of parameters within the decoder-only layers (i.e., excluding all parameters in the embedding layer), we can see that LM loss follows a smooth power law with respect to N. In other words, _increasing the size of the LM yields a steady improvement in its performance_.

[

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2Ffd73449b-e9e4-4e05-b6cb-4f9a8c47527f_660x610.png)



](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2Ffd73449b-e9e4-4e05-b6cb-4f9a8c47527f_660x610.png)

(from [1])

**data and compute.** To study how LM performance scales with the amount of training data, authors of [1] adopt a sufficiently-large LM and perform separate training trails over differently-sized datasets. For each trial, the model is trained until the test loss begins to increase, an indication of overfitting. Again, this analysis shows us that test loss decreases according to a power law with respect to the size of the dataset; see above.

[

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2Fbd1b4ffe-d02a-47ed-bd1f-5c7f926065b2_714x628.png)



](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2Fbd1b4ffe-d02a-47ed-bd1f-5c7f926065b2_714x628.png)

(from [1])

We see a very similar trend when varying the amount of training compute, defined as C = 6NBS for batch size B and number of training iterations S. Given a sufficiently-large dataset and fixed batch size B, we can scan over numerous LM sizes N to obtain the result shown above. Here, we see that the optimal results for each compute budget C are achieved using different combinations of N and S, but the best LM loss decreases according to a power law with respect to the amount of training compute.

Going further, we can see from these results that LM sample efficiency (i.e., how many samples it takes for the model to perform well) improves with increasing N. To show this more clearly, authors of [1] analyze the performance of different-sized LMs with respect to the total number of samples observed during training, yielding the plot shown below. Here, we can clearly see LM performance improves more quickly as the models become larger.

[

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F72f34bc3-ce21-48e2-a4a9-5101c0b7e1f6_904x824.png)



](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F72f34bc3-ce21-48e2-a4a9-5101c0b7e1f6_904x824.png)

(from [1])

**pairwise scaling laws.** Beyond the power laws observed by analyzing N, D, and C in isolation, varying pairs of these factors simultaneously can also yield predictable behavior; e.g., by jointly varying N and D we can obtain the plot shown below. Here, we observe that _(i)_ larger models begin to overfit on smaller datasets and _(ii)_ LM loss follows a strict power law with respect to N given a sufficiently large dataset.

[

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2Fd7ca3ed5-acb1-4a5c-96d3-dade2313f900_1372x756.png)



](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2Fd7ca3ed5-acb1-4a5c-96d3-dade2313f900_1372x756.png)

(from [1])

At a high level, this tells us that we must make the dataset larger in order to avoid overfitting when we increase the size of the underlying LM. However, authors in [1] find that scaling the data size sub-linearly (i.e., proportional to `N^0.74` specifically) is sufficient to avoid overfitting.

**Takeaways.** Though we have discussed the power laws outlined in [1] at a high level, the actual publication makes these laws quite concrete and even proposes an accurate predictive framework for the test loss of any LM. For simplicity, we avoid these details here, instead focusing on the following takeaways for training LMs.

[

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F1e45145e-6913-4a83-847d-0b8aca7360f6_1436x672.png)



](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F1e45145e-6913-4a83-847d-0b8aca7360f6_1436x672.png)

How to properly invest more compute into LLM training (from [1])

If we are increasing the scale of LM training, we should:

1. Invest most of the extra compute into an increased model size (i.e., larger models are more sample efficient)
    
2. Increase the size of the dataset (but not as much as the model size) to avoid overfitting.
    
3. Slightly increase the batch size (i.e., according to the critical batch size [3]).
    
4. Stop training the model significantly short of convergence to optimize the use of training compute.
    

The power laws observed in [1] continue seemingly unimpeded for several orders of magnitude. Although this scaling will eventually reach a limit, it nonetheless shows that (properly) increasing the scale of LM training yields measurable performance benefits, hinting that exploring LLMs (like GPT-3) could prove to be incredibly beneficial.

> “Our results strongly suggest that larger models will continue to perform better, and will also be much more sample efficient than has been previously appreciated. Big models may be more important than big data.” - from [1]

Thanks for reading Deep (Learning) Focus! Subscribe for free to receive new posts and support my work.

Subscribe

### [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165) [2]

Prior work on GPT and GPT-2 [4, 5] began to reveal the utility of general purpose LMs for solving textual understanding tasks. However, these models still had limitations:

- GPT was not fully task-agnostic (i.e., required task-specific fine-tuning)
    
- GPT-2 performed far worse than supervised state-of-the-art in the zero-shot regime
    

Existing work provides a “proof of concept” that LMs could remove the need for task specification by performing zero/few-shot, task-agnostic inference. However, the poor performance of LMs relative to supervised techniques makes them less practical. Luckily, the power laws observed within [1] provide hope that larger LMs (i.e., LLMs) could narrow the gap between task-agnostic and task-specific/supervised performance.

Moving in this direction, GPT-3, which shares the same decoder-only architecture as GPT-2 (aside from the addition of some sparse attention layers [6]), builds upon the size of existing LMs by several orders of magnitude. In particular, it is an LLM with over 175 billion parameters (i.e., for reference, GPT-2 [5] contains 1.5 billion parameters); see below.

[

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F2d05fa6b-0603-48b2-ba60-9187a0bd38af_1262x406.png)



](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F2d05fa6b-0603-48b2-ba60-9187a0bd38af_1262x406.png)

(from [2])

With GPT-3, we finally begin to see promising task-agnostic performance with LLMs, as the model’s few-shot performance approaches that of supervised baselines on several tasks. Similar to GPT-2, authors pre-train the LLM using a language modeling objective, but they adopt a larger dataset based upon a filtered version of CommonCrawl and some additional, high-quality corpora. The breakdown of the full dataset used for pre-training is shown below.

[

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F1e0630ce-c5b2-4880-a8e7-6859f12a6b17_2210x700.png)



](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F1e0630ce-c5b2-4880-a8e7-6859f12a6b17_2210x700.png)

(from [2])

Pre-training with GPT-3 is conducted similarly to GPT-2, but the model is trained for much longer. To make the training process computationally feasible, the authors adopt a model parallel distributed training approach that distributes portions of each LM layer across separate GPUs. Because each GPU only stores a small portion of the full model, training can be conducted without exceeding memory constraints.

The learning process of GPT-3 has two components: un/self-supervised pre-training and in-context learning. These two components are illustrated in the figure below.

[

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F3f8961e8-7484-44e3-9a78-eb4d5365cf63_2214x1276.png)



](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F3f8961e8-7484-44e3-9a78-eb4d5365cf63_2214x1276.png)

(from [2])

Put simply, we first pre-train the general purpose LLM over a large unsupervised text corpus, then guide this model to solve downstream tasks using in-context learning. This in-context learning process can be performed via task-specific fine-tuning (as in GPT) or even using techniques like few-shot learning that require no gradient updates to the LM. The difference between fine-tuning and different variants of zero, one, and few-shot learning is depicted below.

[

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2Fcc1b83b2-6cb7-4a36-80a2-0d1ac53d013f_1580x1380.png)



](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2Fcc1b83b2-6cb7-4a36-80a2-0d1ac53d013f_1580x1380.png)

(from [2])

Unlike prior variants, GPT-3 is evaluated solely using zero and few-shot learning techniques. The authors do not adapt or fine-tune the model to any of the downstream datasets used for evaluation. Rather, they pre-train this incredibly large model over a massive text corpus and study whether in-context learning can be accurately performed using only few-shot prompting techniques that contain varying numbers of “in-context examples” as shown in the figure above.

By evaluating GPT-3 on a range of language understanding tasks, we immediately see that using a larger model significantly benefits few-shot performance. On sentence completion tasks, for example, GPT-3 improves the current state-of-the-art (i.e., including approaches that use supervised training or fine-tuning!) on several popular datasets, and providing more in-context examples seems to further improve performance; see below.

[

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F7c8b6ee1-49db-4981-90aa-17bf68141839_1392x1222.png)



](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F7c8b6ee1-49db-4981-90aa-17bf68141839_1392x1222.png)

(from [2])

On question answering tasks, we see that GPT-3 is outperformed by models like T5 [7] or RoBERTa [8]. However, these models perform extensive, supervised fine-tuning, while GPT-3 achieves comparable results via task-agnostic, few-shot inference. Put simply, GPT-3’s performance on these tasks is still impressive because it is a completely generic LLM that has not been specialized to solving these tasks in any way.

When evaluating GPT-3 on translation tasks, we observe that GPT-3 is better than state-of-the-art unsupervised neural machine translation (NMT) techniques at translating from other languages into English. Such results are surprising given that GPT-3’s pre-training set contains only 7% non-English content and no explicit mixing of or translation between languages. Interestingly, GPT-3 is much less effective at translating from English into other languages; see below.

[

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F0bc961b3-9741-41b5-9b70-22c5f81253c5_2028x774.png)



](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F0bc961b3-9741-41b5-9b70-22c5f81253c5_2028x774.png)

(from [2])

Authors also evaluate GPT-3 on the [SuperGLUE benchmark](https://super.gluebenchmark.com/), which contains a wide variety of different language understanding tasks. The results are summarized within the figure below, where we can see that _(i)_ using more in-context examples benefits GPT-3’s performance and _(ii)_ GPT-3 can even surpass the performance of popular, fine-tuned baselines like [BERT](https://cameronrwolfe.substack.com/p/language-understanding-with-bert) [9].

[

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F1e797415-7032-4ab3-b2d3-7ad2f8385a27_2154x1368.png)



](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F1e797415-7032-4ab3-b2d3-7ad2f8385a27_2154x1368.png)

(from [2])

Across all benchmarks, GPT-3 shows us that LLMs become more effective at task-agnostic, in-context learning as they grow in size. We can use in-context examples to prompt accurate responses from LLMs on a variety of tasks, making GPT-3 the first practical example of using general purpose LLMs to perform highly-accurate inference on a variety of downstream tasks without any task-specific modifications.

Despite the incredible leaps made by GPT-3 towards creating task-agnostic foundation models for language, these advancements come at a significant computational cost. GPT-3 was pre-trained on a special-purpose GPU cluster and its pre-training process required significantly more compute than any previous model that has been studied; see below.

[

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2Ffaa4655d-d69f-4115-b343-fc294984270a_1584x962.png)



](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2Ffaa4655d-d69f-4115-b343-fc294984270a_1584x962.png)

(from [2])

Although recent work has [drastically reduced the training cost of GPT-3](https://www.mosaicml.com/blog/gpt-3-quality-for-500k) (i.e., from >$10M in compute costs to <$500K), such foundation models are still not cheap to obtain. If we want to create our own foundation model like GPT-3, we better make sure it performs well. 

**open-sourcing GPT-3.** After the original proposal of GPT-3 in [2], the model was not publicly released. Rather, it was made accessible only via [paid APIs](https://openai.com/api/). Although the model’s API was heavily used, this lack of open-source access to the model itself (and its training code) hindered further analysis and experimentation.

To eliminate this issue, an open-sourced version of GPT-3, called OPT-175B, was created and analyzed in [10]. The release of OPT-175B also included a full code repository and several logbooks that provided valuable insights into the LLM training process. To learn more about OPT-175B (and see code you can use to train LLMs like GPT-3!), check out the overview below. 

[Learn about OPT-175B](https://cameronrwolfe.substack.com/p/understanding-the-open-pre-trained-transformers-opt-library-193a29c14a15)

# Takeaways

GPT models were originally proposed and explored with the goal of creating generic language models that are capable of solving a wide variety of tasks. These models operate under the assumption that if we can understand language modeling (i.e., predicting the next word within a sequence) at a very granular level, then we can generalize this understanding in a lot of useful ways without the need for task-specific fine-tuning or adaptation.

Initially, LMs like GPT and GPT-2 fell short of this goal. Their task-agnostic performance was far worse than supervised baselines. Within this overview, however, we have learned that increasing the scale of these LMs is a viable path forward in creating high-performing, task-agnostic models for language understanding. Eventually, this line of thinking led to the proposal and analysis of GPT-3, a massive LLM (i.e., ~100X bigger than GPT-2) that far surpassed the task-agnostic performance of prior LMs.

**scaling laws.** Scaling up LMs (i.e., using larger models, more data, and more compute) can drastically improve their performance. As we increase the scale of LM training, we learn from findings in [1] that we should _(i)_ significantly increase the size of the underlying model and _(ii)_ increase the amount of data used for pre-training (and the batch size) to a lesser extent. Larger language models are more sample efficient, and their performance improves as a power law with respect to model size, data size, and the amount of training compute across several orders of magnitude. In other words, _LMs get much better as we make them bigger_. 

**how much can we scale?** GPT-3 (an LLM with 175 billion parameters) empirically validates the trends outlined in [1] at an unprecedented scale. When we adopt this massive model and pre-train it over a large textual corpus, we see large improvements in task-agnostic, few-shot performance. GPT-3 is still outperformed by supervised techniques on several baselines, but findings in [2] provide clear evidence that LLMs improve in their ability to perform in-context learning as they grow in size. Though GPT-3 is technically similar to GPT-2, training a model of this scale is a feat of engineering that demonstrates the incredible potential of language foundation models.

### New to the newsletter?

Hello! I am [Cameron R. Wolfe](https://cameronrwolfe.me/), a research scientist at [Alegion](https://www.alegion.com/) and PhD student at Rice University. I study the empirical and theoretical foundations of deep learning. This is the Deep (Learning) Focus newsletter, where I pick a single, bi-weekly topic in deep learning research, provide an understanding of relevant background information, then overview a handful of popular papers on the topic. If you like this newsletter, please subscribe, share it with your friends, or follow me on [twitter](https://twitter.com/cwolferesearch)!

Subscribe

### Bibliography

[1] Kaplan, Jared, et al. "Scaling laws for neural language models." arXiv preprint arXiv:2001.08361 (2020).

[2] Brown, Tom, et al. "Language models are few-shot learners." Advances in neural information processing systems 33 (2020): 1877-1901.

[3] McCandlish, Sam, et al. "An empirical model of large-batch training." arXiv preprint arXiv:1812.06162 (2018).

[4] Radford, Alec, et al. "Improving language understanding by generative pre-training." (2018). 

[5] Radford, Alec, et al. "Language Models are Unsupervised Multitask Learners."

[6] Child, Rewon, et al. "Generating long sequences with sparse transformers." arXiv preprint arXiv:1904.10509 (2019).

[7] Raffel, Colin, et al. "Exploring the limits of transfer learning with a unified text-to-text transformer." J. Mach. Learn. Res. 21.140 (2020): 1-67.

[8] Liu, Yinhan, et al. "Roberta: A robustly optimized bert pretraining approach." arXiv preprint arXiv:1907.11692 (2019).

[9] Devlin, Jacob, et al. "Bert: Pre-training of deep bidirectional transformers for language understanding." arXiv preprint arXiv:1810.04805 (2018).

[10] Zhang, Susan, et al. "Opt: Open pre-trained transformer language models." arXiv preprint arXiv:2205.01068 (2022).

---

#### Subscribe to Deep (Learning) Focus

By Cameron R. Wolfe · Launched 3 years ago

I contextualize and explain important topics in AI research.

Subscribe

By subscribing, I agree to Substack's [Terms of Use](https://substack.com/tos), and acknowledge its [Information Collection Notice](https://substack.com/ccpa#personal-data-collected) and [Privacy Policy](https://substack.com/privacy).

[

![](https://substackcdn.com/image/fetch/w_32,h_32,c_fill,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fff923ad0-a46d-4514-9211-47fbbbe2fd89_144x144.png)



](https://substack.com/profile/205385075-jules-bahanyi)

[

![](https://substackcdn.com/image/fetch/w_32,h_32,c_fill,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fdb13ea77-676c-4e0c-ad1c-412cfb304307_96x96.png)



](https://substack.com/profile/296261040-hai)

[

![](https://substackcdn.com/image/fetch/w_32,h_32,c_fill,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F933464d4-9b1f-4f5a-a2bb-4633149a320e_144x144.png)



](https://substack.com/profile/18108189-tiklu-ganguly)

[

![](https://substackcdn.com/image/fetch/w_32,h_32,c_fill,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F1addaf03-9aa5-4f86-a72e-f0fa62852abe_144x144.png)



](https://substack.com/profile/111548585-eda)

[

![](https://substackcdn.com/image/fetch/w_32,h_32,c_fill,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F27fddcfd-ebf9-48af-82d9-1331d5b8a902_4167x4167.png)



](https://substack.com/profile/45646766-obrian-henry)

11 Likes

[](https://substack.com/note/p-88082618/restacks?utm_source=substack&utm_content=facepile-restacks)

11

[](https://cameronrwolfe.substack.com/p/language-model-scaling-laws-and-gpt/comments)

Share

#### Discussion about this post

CommentsRestacks

![](https://substackcdn.com/image/fetch/w_32,h_32,c_fill,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack.com%2Fimg%2Favatars%2Fdefault-light.png)

TopLatestDiscussions

[Decoder-Only Transformers: The Workhorse of Generative LLMs](https://cameronrwolfe.substack.com/p/decoder-only-transformers-the-workhorse)

[Building the world's most influential neural network architecture from scratch...](https://cameronrwolfe.substack.com/p/decoder-only-transformers-the-workhorse)

Mar 4, 2024 • 

[Cameron R. Wolfe, Ph.D.](https://substack.com/@cwolferesearch)

106

[

14

](https://cameronrwolfe.substack.com/p/decoder-only-transformers-the-workhorse/comments)

![](https://substackcdn.com/image/fetch/w_320,h_213,c_fill,f_auto,q_auto:good,fl_progressive:steep,g_center/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F6e3c9db5-400a-49de-a235-e09bc3aa3689_2392x1342.png)

[Mixture-of-Experts (MoE) LLMs](https://cameronrwolfe.substack.com/p/moe-llms)

[Understanding models like DeepSeek, Grok, and Mixtral from the ground up...](https://cameronrwolfe.substack.com/p/moe-llms)

Jan 27 • 

[Cameron R. Wolfe, Ph.D.](https://substack.com/@cwolferesearch)

197

[

10

](https://cameronrwolfe.substack.com/p/moe-llms/comments)

![](https://substackcdn.com/image/fetch/w_320,h_213,c_fill,f_auto,q_auto:good,fl_progressive:steep,g_center/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F3fdf1382-38dc-45fc-a741-b62babfd99c5_2258x1268.png)

[Understanding and Using Supervised Fine-Tuning (SFT) for Language Models](https://cameronrwolfe.substack.com/p/understanding-and-using-supervised)

[Understanding how SFT works from the idea to a working implementation...](https://cameronrwolfe.substack.com/p/understanding-and-using-supervised)

Sep 11, 2023 • 

[Cameron R. Wolfe, Ph.D.](https://substack.com/@cwolferesearch)

50

[

5

](https://cameronrwolfe.substack.com/p/understanding-and-using-supervised/comments)

![](https://substackcdn.com/image/fetch/w_320,h_213,c_fill,f_auto,q_auto:good,fl_progressive:steep,g_center/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F68686a01-2b31-4694-8c04-a562ffd725ad_2210x1244.png)

See all

Ready for more?

Subscribe

© 2025 Cameron R. Wolfe

[Privacy](https://substack.com/privacy) ∙ [Terms](https://substack.com/tos) ∙ [Collection notice](https://substack.com/ccpa#personal-data-collected)

[Start Writing](https://substack.com/signup?utm_source=substack&utm_medium=web&utm_content=footer)[Get the app](https://substack.com/app/app-store-redirect?utm_campaign=app-marketing&utm_content=web-footer-button)

[Substack](https://substack.com/) is the home for great culture