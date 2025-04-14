[Distilled AI](https://aman.ai/primers/ai/)[Back to aman.ai](https://aman.ai/)

# Primers • Encoder vs. Decoder vs. Encoder-Decoder Models

- [Overview: Encoder vs. Decoder vs. Encoder-Decoder](https://aman.ai/primers/ai/encoder-vs-decoder-models/#overview-encoder-vs-decoder-vs-encoder-decoder)
- [Taxonomy](https://aman.ai/primers/ai/encoder-vs-decoder-models/#taxonomy)
- [Decoder Models](https://aman.ai/primers/ai/encoder-vs-decoder-models/#decoder-models)
    - [Autoregressive Models](https://aman.ai/primers/ai/encoder-vs-decoder-models/#autoregressive-models)
    - [How Does an Autoregressive Model Work?](https://aman.ai/primers/ai/encoder-vs-decoder-models/#how-does-an-autoregressive-model-work)
    - [Language Modeling](https://aman.ai/primers/ai/encoder-vs-decoder-models/#language-modeling)
    - [Pros and Cons](https://aman.ai/primers/ai/encoder-vs-decoder-models/#pros-and-cons)
- [Encoder Models](https://aman.ai/primers/ai/encoder-vs-decoder-models/#encoder-models)
    - [Pros and Cons of Encoder Models](https://aman.ai/primers/ai/encoder-vs-decoder-models/#pros-and-cons-of-encoder-models)
- [Encoder-Decoder/Seq2seq Models](https://aman.ai/primers/ai/encoder-vs-decoder-models/#encoder-decoderseq2seq-models)
- [Enter XLNet: the Best of Both Worlds](https://aman.ai/primers/ai/encoder-vs-decoder-models/#enter-xlnet-the-best-of-both-worlds)
    - [Permutation Language Modeling](https://aman.ai/primers/ai/encoder-vs-decoder-models/#permutation-language-modeling)
    - [What Problems Does Permutation Language Modeling Bring?](https://aman.ai/primers/ai/encoder-vs-decoder-models/#what-problems-does-permutation-language-modeling-bring)
    - [Does BERT Have the Issue of Separating Position Embeddings from the Token Embeddings?](https://aman.ai/primers/ai/encoder-vs-decoder-models/#does-bert-have-the-issue-of-separating-position-embeddings-from-the-token-embeddings)
    - [How Does XLNet Solve the Issue of Separating Position Embeddings from Token Embeddings?](https://aman.ai/primers/ai/encoder-vs-decoder-models/#how-does-xlnet-solve-the-issue-of-separating-position-embeddings-from-token-embeddings)
- [Attention Mask: How Does XLNet Implement Permutation?](https://aman.ai/primers/ai/encoder-vs-decoder-models/#attention-mask-how-does-xlnet-implement-permutation)
    - [Summary](https://aman.ai/primers/ai/encoder-vs-decoder-models/#summary)
        - [Background](https://aman.ai/primers/ai/encoder-vs-decoder-models/#background)
        - [Core Innovations of XLNet](https://aman.ai/primers/ai/encoder-vs-decoder-models/#core-innovations-of-xlnet)
            - [Permutation Language Modeling](https://aman.ai/primers/ai/encoder-vs-decoder-models/#permutation-language-modeling-1)
            - [Autoregressive Method](https://aman.ai/primers/ai/encoder-vs-decoder-models/#autoregressive-method)
            - [Transformer-XL Backbone](https://aman.ai/primers/ai/encoder-vs-decoder-models/#transformer-xl-backbone)
        - [Technical Details](https://aman.ai/primers/ai/encoder-vs-decoder-models/#technical-details)
            - [Training Objective](https://aman.ai/primers/ai/encoder-vs-decoder-models/#training-objective)
            - [Relative Positional Encodings](https://aman.ai/primers/ai/encoder-vs-decoder-models/#relative-positional-encodings)
            - [Segment-Level Recurrence](https://aman.ai/primers/ai/encoder-vs-decoder-models/#segment-level-recurrence)
        - [Performance and Applications](https://aman.ai/primers/ai/encoder-vs-decoder-models/#performance-and-applications)
        - [Conclusion](https://aman.ai/primers/ai/encoder-vs-decoder-models/#conclusion)
- [Advantages and Applications: Encoder vs. Decoder vs. Encoder-Decoder](https://aman.ai/primers/ai/encoder-vs-decoder-models/#advantages-and-applications-encoder-vs-decoder-vs-encoder-decoder)
- [Conclusion](https://aman.ai/primers/ai/encoder-vs-decoder-models/#conclusion-1)
- [Further Reading](https://aman.ai/primers/ai/encoder-vs-decoder-models/#further-reading)
- [Citation](https://aman.ai/primers/ai/encoder-vs-decoder-models/#citation)

## Overview: Encoder vs. Decoder vs. Encoder-Decoder

- Self-supervised representation learning has been highly successful in the domain of natural language processing (NLP). Typically, these methods first pretrain neural networks on large-scale unlabeled text corpora, and then fine-tune the models or representations on downstream tasks. This approach leverages the vast amounts of text data available, allowing models to learn the intricacies of language, such as grammar, context, and semantics, in an unsupervised manner. Under this shared high-level idea, different unsupervised pretraining objectives have been explored in literature.
- Among them, encoder-based and decoder-based (i.e., autoregressive) language modeling have been the two most successful pretraining objectives. These two approaches form the foundation of many state-of-the-art NLP models, each offering unique advantages for various tasks. Encoder-decoder models, or sequence-to-sequence (seq2seq) models, build on these foundations by combining both types of architectures, enabling them to perform complex transformations from one sequence to another, such as translating text from one language to another.
- **Encoder Models**:
    - **Encoder-based models** focus on understanding the input text. They are designed to capture rich contextual information by pretraining on tasks that involve reconstructing corrupted input, such as masked language modeling. **BERT (Bidirectional Encoder Representations from Transformers)** is a prime example of an encoder model. In BERT, a certain percentage of the input tokens are replaced with a special `[MASK]` token, and the model is trained to predict these masked tokens based on the surrounding context. This allows BERT to learn bidirectional representations, which capture context from both left and right of a token.
    - Encoder models are particularly effective for natural language understanding (NLU) tasks, such as text classification, sentiment analysis, and extractive question answering. These tasks benefit from the model’s ability to deeply understand and represent the input text.
- **Decoder Models**:
    - **Decoder-based models** (autoregressive models) are designed for text generation tasks. These models generate text one token at a time, using previously generated tokens as context for predicting the next token. **Examples of decoder models include GPT (Generative Pretrained Transformer), GPT-2, and GPT-3.** These models are pretrained on large corpora to predict the next word in a sentence, enabling them to generate coherent and contextually relevant text.
    - Decoder models excel in natural language generation (NLG) tasks, such as language translation, text summarization, and dialogue generation. Their ability to produce fluent and contextually appropriate text makes them ideal for tasks that require generating content from scratch.
- **Encoder-Decoder Models**:
    - **Encoder-decoder models** (also known as sequence-to-sequence or seq2seq models) combine the strengths of both encoder and decoder architectures. These models use an encoder to process and understand the input sequence and a decoder to generate the output sequence. This architecture is particularly effective for tasks where the input and output are sequences, possibly of different lengths and in different formats or languages.
    - The encoder transforms the input sequence into a fixed-length context vector or intermediate representation that captures the meaning and context of the input. The decoder then takes this context vector and generates the output sequence, one token at a time, often using autoregressive techniques similar to those in decoder models.
    - **T5 (Text-To-Text Transfer Transformer)** is a notable example of an encoder-decoder model. T5 treats every NLP problem as a text-to-text problem, where both the input and output are text sequences. This approach allows T5 to be applied to a wide range of tasks, including translation, summarization, and question answering.
    - **BART (Bidirectional and Auto-Regressive Transformers)** is another powerful encoder-decoder model. BART is pretrained by corrupting text with an arbitrary noise function and learning to reconstruct the original text. This makes it effective for tasks that require generating text based on an understanding of the input, such as summarization and dialogue generation.
    - **BigBird** extends the Transformer architecture to handle longer sequences by using sparse attention mechanisms. This makes it suitable for tasks that involve long documents, such as document classification and long-form question answering.

## Taxonomy

- Tying in with the Transformer architecture, the following tree diagram ([source](https://discuss.huggingface.co/t/suggestions-and-guidance-finetuning-bert-models-for-next-word-prediction/14043)) shows the Transformer encoders/AE models (blue), decoders/autoregressive models (red), and encoder-decoder/seq2seq models (grey):

![](https://aman.ai/primers/ai/assets/autoregressive-vs-autoencoder-models/enc_dec.jpeg)

## Decoder Models

- Decoder models are fundamental components in sequence-to-sequence tasks, where they are responsible for generating outputs based on encoded inputs. They play a crucial role in tasks such as machine translation, text summarization, and conversational agents, ensuring the generation of coherent and contextually appropriate text. The ability of decoder models to handle sequential data effectively makes them integral to modern natural language processing systems.
- Arguably, the most prominent type of decoder model is the autoregressive model, which generates text sequences by predicting the next token based on previously generated tokens.

### Autoregressive Models

- An autoregressive model learns from a series of timed steps and takes measurements from previous actions as inputs for a regression model, in order to predict the value of the next time step. This approach ensures that the model can generate text sequences that are coherent and contextually relevant.
- Autoregressive models are typically used for generation tasks, such as those in the domain of natural language generation (NLG), for example, summarization, translation, or abstractive question answering. These models excel at generating fluent and contextually appropriate text, making them ideal for creative and content-generation tasks.
- Decoder models, such as those used in sequence-to-sequence tasks, often incorporate autoregressive properties to generate outputs one token at a time based on previously generated tokens. This allows them to effectively model the sequential dependencies in language generation tasks. The ability to generate sequences token-by-token ensures that the output is coherent and contextually consistent.

### How Does an Autoregressive Model Work?

- Autoregressive modeling centers on measuring the correlation between observations at previous time steps (the lag variables) to predict the value of the next time step (the output). This method captures the temporal dependencies in the data, which is crucial for tasks that involve sequence generation.
- If both variables change in the same direction, for example increasing or decreasing together, then there is a positive correlation. If the variables move in opposite directions as values change, for example, one increasing while the other decreases, then this is called negative correlation. Either way, using basic statistics, the correlation between the output and previous variable can be quantified.
- The higher this correlation, positive or negative, the more likely that the past will predict the future. Or in machine learning terms, the higher this value will be weighted during deep learning training.
- Since this correlation is between the variable and itself at previous time steps, it is referred to as an autocorrelation.
- In addition, if every variable shows little to no correlation with the output variable, then it’s likely that the time series dataset may not be predictable. In practical terms, autoregressive models leverage this property to effectively learn and predict sequential data, making them robust for various language modeling tasks.

### Language Modeling

- In language modeling, an autoregressive language model uses the context word to predict the next word by estimating the probability distribution of a text corpus. Specifically, given a text sequence x=(x1,⋯,xT)x=(x1,⋯,xT), autoregressive language modeling factorizes the likelihood into a forward product p(x)=∏Tt=1p(xt∣x<t)p(x)=∏t=1Tp(xt∣x<t) or a backward one p(x)=∏1t=Tp(xt∣x>t)p(x)=∏t=T1p(xt∣x>t).
- A parametric model (e.g., a neural network) is trained to model each conditional distribution. Since an autoregressive language model is only trained to encode a uni-directional context (either forward or backward), it is not effective at modeling deep bidirectional contexts. The figures below illustrate the forward/backward directionality.

![](https://aman.ai/primers/ai/assets/autoregressive-vs-autoencoder-models/ar.jpg)![](https://aman.ai/primers/ai/assets/autoregressive-vs-autoencoder-models/ar_b.jpg)

- On the contrary, downstream language understanding tasks often require bidirectional context information. This results in a gap between autoregressive language modeling and effective pre-training. To address this gap, models like XLNet introduce innovative techniques to incorporate bidirectional context while maintaining the autoregressive property.
- GPT, Llama, and Mixtral are examples of autoregressive language models.

### Pros and Cons

- **Pros:**
    - Autoregressive language models are good at generative NLP tasks. Since autoregressive models utilize causal attention to predict the next token, they are naturally applicable for generating content. Their ability to generate fluent and contextually relevant text makes them ideal for tasks requiring natural language generation.
    - Data generation for training these models is relatively straightforward, as the objective is simply to predict the next token in a given sequence, leveraging the inherent structure of language data.
- **Cons:**
    - Autoregressive language models can only use forward context or backward context, which means they can’t use bidirectional context at the same time. This limitation can hinder their performance in tasks that require a deep understanding of context from both directions.

## Encoder Models

- Encoder pre-training does not perform explicit density estimation but instead aims to reconstruct the original data from corrupted input (“fill in the blanks”). This approach allows encoder models to capture rich contextual information by focusing on understanding the input data.
- Encoder models are typically used for content understanding tasks, such as tasks in the domain of natural language understanding (NLU) that involve classification, for example, sentiment analysis, or extractive question answering. These tasks benefit from the model’s ability to deeply understand and represent the input text.
- A notable example is [BERT](https://aman.ai/primers/ai/bert), which has been the state-of-the-art pre-training approach. Given the input token sequence, a certain portion of tokens are replaced by a special symbol `[MASK]`, and the model is trained to recover the original tokens from the corrupted version. The encoder language model aims to reconstruct the original data from corrupted input.
- Since density estimation is not part of the objective, BERT is allowed to utilize bidirectional contexts for reconstruction. As an immediate benefit, this closes the aforementioned bidirectional information gap in autoregressive language modeling, leading to improved performance. However, the artificial symbols like `[MASK]` used by BERT during pre-training are absent from real data at finetuning time, resulting in a pretrain-finetune discrepancy. Moreover, since the predicted tokens are masked in the input, BERT is not able to model the joint probability using the product rule as in autoregressive language modeling. In other words, BERT assumes the predicted tokens are independent of each other given the unmasked tokens, which is oversimplified as high-order, long-range dependency is prevalent in natural language.

![](https://aman.ai/primers/ai/assets/autoregressive-vs-autoencoder-models/ae.jpg)

- With masked language modeling as a common training objective in pre-training autoencoder models, we predict the value of the original value of the masked tokens in the corrupted input.
- BERT (and all of its variants such as RoBERTa, DistilBERT, ALBERT, etc.), and XLM are examples of encoder models.

### Pros and Cons of Encoder Models

- **Pros:**
    - **Context dependency:** The autoregressive representation hθ(x1:t−1)hθ(x1:t−1) is only conditioned on the tokens up to position tt (i.e., tokens to the left), while the BERT representation Hθ(x)tHθ(x)t has access to the contextual information on both sides. As a result, the BERT objective allows the model to be pretrained to better capture bidirectional context. This bidirectional context is crucial for understanding complex dependencies and relationships in the text, leading to superior performance on many NLP tasks.
    - **Rich contextual understanding:** Encoder models excel at capturing intricate patterns and relationships within the input data, which is essential for tasks that require deep comprehension and analysis, such as named entity recognition, text classification, and reading comprehension.
- **Cons:**
    - **Input noise:** The input to BERT contains artificial symbols like `[MASK]` that never occur in downstream tasks, which creates a pretrain-finetune discrepancy. Replacing `[MASK]` with original tokens as in [10] does not solve the problem because original tokens can be only used with a small probability - otherwise Eq. (2) will be trivial to optimize. In comparison, autoregressive language modeling does not rely on any input corruption and does not suffer from this issue. Put simply, encoder models use the `[MASK]` token during the pre-training, but these symbols are absent from real-world data during finetuning time, resulting in a pretrain-finetune discrepancy.
    - **Independence Assumption:** As emphasized by the ≈≈ sign in Eq. (2), BERT factorizes the joint conditional probability p(x⎯⎯⎯∣x̂ )p(x¯∣x^) based on an independence assumption that all masked tokens x⎯⎯⎯x¯ are separately reconstructed. In comparison, the autoregressive language modeling objective (1) factorizes pθ(x)pθ(x) using the product rule that holds universally without such an independence assumption. Put simply, another disadvantage of `[MASK]` is that it assumes the predicted (masked) tokens are independent of each other given the unmasked tokens. For example, consider a sentence: “it shows that the housing crisis was turned into a banking crisis”. If we mask “banking” and “crisis”, the masked words contain an implicit relation to each other. But the encoder model is trying to predict “banking” given unmasked tokens, and predict “crisis” given unmasked tokens separately. It ignores the relation between “banking” and “crisis”. This independence assumption can limit the model’s ability to capture complex dependencies between masked tokens, affecting its performance on tasks requiring such nuanced understanding.

## Encoder-Decoder/Seq2seq Models

- An encoder-decoder/seq2seq model, as the name suggests, uses both an encoder and decoder. It treats each task as sequence to sequence conversion/generation (for e.g., text to text, or even multimodal tasks such as text to image or image to text). This architecture is versatile and can be applied to a wide range of tasks by converting the input sequence into an intermediate representation and then generating the output sequence.
- Encoder-decoder/seq2seq models are typically used for tasks that require both content understanding and generation (where the content needs to be converted from one form to another), such as machine translation. By leveraging both the encoder’s ability to understand the input and the decoder’s capability to generate coherent output, these models excel in tasks that involve transforming information from one format to another.
- T5, BART, and BigBird are examples of Encoder-Decoder models. These models have demonstrated state-of-the-art performance across a variety of NLP tasks, showcasing their flexibility and effectiveness in handling complex language processing challenges.

## Enter XLNet: the Best of Both Worlds

- XLNet is an example of a generalized autoregressive pre-training method, which leverages the best of both encoder and decoder-based language modeling while avoiding their limitations, i.e., it offers an autoregressive language model which utilizes bi-directional context and avoids the independence assumption and pretrain-finetune discrepancy disadvantages brought by the token masking method in AE models. In essence, it combines the advantages of Transformer-XL, which can model long-term dependencies, with the bidirectional context capabilities of BERT, ensuring that it captures context from both directions while maintaining the autoregressive property.

### Permutation Language Modeling

- The autoregressive language model only can use the context either forward or backward, so how to let it learn from bi-directional context?
- Language model consists of two phases, the pre-train phase, and fine-tune phase. XLNet focus on pre-train phase. In the pre-train phase, it proposed a new objective called Permutation Language Modeling. We can know the basic idea from this name, it uses permutation. The following Illustration from the paper ([source](https://arxiv.org/abs/1906.08237)) shows the idea of permutation:

![](https://aman.ai/primers/ai/assets/autoregressive-vs-autoencoder-models/xlnet.jpg)

- Here’s an example. The sequence order is [x1,x2,x3,x4][x1,x2,x3,x4]. All permutations of such sequence are below.

![](https://aman.ai/primers/ai/assets/autoregressive-vs-autoencoder-models/perm.jpg)

- For these four tokens (NN) in the input sentence, there are 24 (N!N!) permutations.
- The scenario is that we want to predict the x3x3. So there are 4 patterns in the 24 permutations where x3x3 is at the first, second, third, and fourth position, as shown below and summarized in the figure below ([source](https://towardsdatascience.com/what-is-xlnet-and-why-it-outperforms-bert-8d8fce710335?gi=1642cf38a765)).

[x3,…,…,…][…,x3,…,…][…,…,x3,…][…,…,…,x3][x3,…,…,…][…,x3,…,…][…,…,x3,…][…,…,…,x3]

![](https://aman.ai/primers/ai/assets/autoregressive-vs-autoencoder-models/perm2.jpg)

- Here we set the position of x3x3 as the tthtth position and the remaining t−1t−1 tokens are the context words for predicting x3x3. Intuitively, the model will learn to gather bi-directional information from all positions on both sides. This innovative approach allows XLNet to predict tokens based on the dynamic context generated by the permutations, thus enhancing its ability to understand the nuances of language.

### What Problems Does Permutation Language Modeling Bring?

- The permutation can make autoregressive model see the context from two directions, but it also brought problems that the original transformer cannot solve. The permutation language modeling objective is as follows:
    
    maxθ𝔼z∼T[∑t=1Tlogpθ(xzt∣xz<t)]maxθEz∼ZT[∑t=1Tlog⁡pθ(xzt∣xz<t)]
    
    - where,
        - ZZ: a factorization order
        - pθpθ: likelihood function
        - xztxzt: the tthtth token in the factorization order
        - xz<txz<t: the tokens before tthtth token
- This is the objective function for permutation language modeling, which means takes t−1t−1 tokens as the context and to predict the tthtth token. The model thus learns to predict each token by taking into account a dynamically changing context that includes all other tokens in the permutation.
- There are two requirements that a standard Transformer cannot do:
    - to predict the token xtxt, the model should only see the position of xtxt, not the content of xtxt.
    - to predict the token xtxt, the model should encode all tokens before xtxt as the content.
- Considering the first requirement above, BERT amalgamates the positional encoding with the token embedding (c.f. figure below; [source](http://jalammar.github.io/illustrated-transformer/)), and thus cannot separate the position information from the token embedding.

![](https://aman.ai/primers/ai/assets/autoregressive-vs-autoencoder-models/emb.jpg)

### Does BERT Have the Issue of Separating Position Embeddings from the Token Embeddings?

- BERT is an AE language model, it does not need separate position information like the autoregressive language model. Unlike the XLNet need position information to predict tthtth token, BERT uses [MASK] to represent which token to predict (we can think [MASK] is just a placeholder). For example, if BERT uses x2x2, x1x1 and x4x4 to predict x3x3, the embedding of x2x2, x1x1, x4x4 contains the position information and other information related to [MASK]. So the model has a high chance to predict that [MASK] is x3x3.
    
- BERT embeddings contain two types of information, the positional embeddings, and the token/content embeddings (here, we’re skipping the sequence embeddings since we’re not concerned about the next sentence prediction (NSP) task), as shown in the figure below ([source](https://arxiv.org/abs/1810.04805)).
    

![](https://aman.ai/primers/ai/assets/autoregressive-vs-autoencoder-models/bert_emb.jpg)

- The position information is easy to understand that it tells the model the position of the current token. The content information (semantics and syntactic) contains the “meaning” of the current token, as shown in the figure below ([source](https://arxiv.org/abs/1810.04805)):

![](https://aman.ai/primers/ai/assets/autoregressive-vs-autoencoder-models/bert_split.jpg)

- An intuitive example of relationships learned with embeddings from the [Word2Vec](https://arxiv.org/abs/1301.3781) paper is:

queen=king−man+womanqueen=king−man+woman

### How Does XLNet Solve the Issue of Separating Position Embeddings from Token Embeddings?

- As shown in the figure below ([source](https://arxiv.org/abs/1810.04805)), XLNet proposes **two-stream self-attention** to solve the problem.

![](https://aman.ai/primers/ai/assets/autoregressive-vs-autoencoder-models/xlnet_tssa.jpg)

- As the name indicates, it contains two kinds of self-attention. One is the content stream attention, which is the standard self-attention in Transformer. The other one is the query stream attention. XLNet introduces it to replace the [MASK] token in BERT.
- For example, if BERT wants to predict x3x3 with knowledge of the context words x1x1 and x2x2, it can use [MASK] to represent the x3x3 token. The [MASK] is just a placeholder. And the embedding of x1x1 and x2x2 contains the position information to help the model to “know” [MASK] is x3x3.
- Things are different come to XLNet. One token x3x3 will serve two kinds of roles. When it is used as content to predict other tokens, we can use the content representation (learned by content stream attention) to represent x3x3. But if we want to predict x3x3, we should only know its position and not its content. That’s why XLNet uses query representation (learned by query stream attention) to preserve context information before x3x3 and only the position information of x3x3. This dual approach allows XLNet to maintain the autoregressive property of predicting tokens based on previous ones, while also capturing the full context of the sequence.
- In order to intuitively understand the Two-Stream Self-Attention, we can just think XLNet replace the [MASK] in BERT with query representation. They just choose different approaches to do the same thing.

## Attention Mask: How Does XLNet Implement Permutation?

- The following shows the various permutations that a sentence [x1,x2,x3,x4][x1,x2,x3,x4] can take:

[(′x′1,′x′2,′x′3,′x′4),(′x′1,′x′2,′x′4,′x′3),(′x′1,′x′3,′x′2,′x′4),(′x′1,′x′3,′x′4,′x′2),(′x′1,′x′4,′x′2,′x′3),(′x′1,′x′4,′x′3,′x′2),(′x′2,′x′1,′x′3,′x′4),(′x′2,′x′1,′x′4,′x′3),(′x′2,′x′3,′x′1,′x′4),(′x′2,′x′3,′x′4,′x′1),(′x′2,′x′4,′x′1,′x′3),(′x′2,′x′4,′x′3,′x′1),(′x′3,′x′1,′x′2,′x′4),(′x′3,′x′1,′x′4,′x′2),(′x′3,′x′2,′x′1,′x′4),(′x′3,′x′2,′x′4,′x′1),(′x′3,′x′4,′x′1,′x′2),(′x′3,′x′4,′x′2,′x′1),(′x′4,′x′1,′x′2,′x′3),(′x′4,′x′1,′x′3,′x′2),(′x′4,′x′2,′x′1,′x′3),(′x′4,′x′2,′x′3,′x′1),(′x′4,′x′3,′x′1,′x′2),(′x′4,′x′3,′x′2,′x′1)][(′x1′,′x2′,′x3′,′x4′),(′x1′,′x2′,′x4′,′x3′),(′x1′,′x3′,′x2′,′x4′),(′x1′,′x3′,′x4′,′x2′),(′x1′,′x4′,′x2′,′x3′),(′x1′,′x4′,′x3′,′x2′),(′x2′,′x1′,′x3′,′x4′),(′x2′,′x1′,′x4′,′x3′),(′x2′,′x3′,′x1′,′x4′),(′x2′,′x3′,′x4′,′x1′),(′x2′,′x4′,′x1′,′x3′),(′x2′,′x4′,′x3′,′x1′),(′x3′,′x1′,′x2′,′x4′),(′x3′,′x1′,′x4′,′x2′),(′x3′,′x2′,′x1′,′x4′),(′x3′,′x2′,′x4′,′x1′),(′x3′,′x4′,′x1′,′x2′),(′x3′,′x4′,′x2′,′x1′),(′x4′,′x1′,′x2′,′x3′),(′x4′,′x1′,′x3′,′x2′),(′x4′,′x2′,′x1′,′x3′),(′x4′,′x2′,′x3′,′x1′),(′x4′,′x3′,′x1′,′x2′),(′x4′,′x3′,′x2′,′x1′)]

- It is very easy to misunderstand that we need to get the random order of a sentence and input it into the model. But this is not true. The order of input sentence is [x1,x2,x3,x4][x1,x2,x3,x4], and XLNet uses the attention mask to permute the factorization order, as shown in the figure ([source](https://arxiv.org/abs/1906.08237)) below.

![](https://aman.ai/primers/ai/assets/autoregressive-vs-autoencoder-models/twostream.jpg)

- From the figure above, the original order of the sentence is [x1,x2,x3,x4][x1,x2,x3,x4]. And we randomly get a factorization order as [x3,x2,x4,x1][x3,x2,x4,x1].
- The upper left corner is the calculation of content representation. If we want to predict the content representation of x1x1, we should have token content information from all four tokens. KV=[h1,h2,h3,h4]KV=[h1,h2,h3,h4] and Q=h1Q=h1.
- The lower-left corner is the calculation of query representation. If we want to predict the query representation of x1x1, we cannot see the content representation of x1x1 itself. KV=[h2,h3,h4]KV=[h2,h3,h4] and Q=g1Q=g1.
- The right corner is the entire calculation process. Let’s do a walk-through of it from bottom to top. First, h(⋅)h(⋅) and g(⋅)g(⋅) are initialized as e(xi)e(xi) and ww. And after the content mask and query mask, the two-stream attention will output the first layer output h(1)h(1) and g(1)g(1) and then calculate the second layer.
- Notice the right content mask and query mask. Both of them are matrices. In the content mask, the first row has 4 red points. It means that the first token (x1x1) can see (attend to) all other tokens including itself (x3→x2→x4→x1x3→x2→x4→x1). The second row has two red points. It means that the second token (x2x2) can see (attend to) two tokens (x3→x2x3→x2). And so on other rows.
- The only difference between the content mask and query mask is those diagonal elements in the query mask are 0, which means the tokens cannot see themselves.
- To sum it up: the input sentence has only one order. But we can use different attention mask to implement different factorization orders. This approach significantly enhances the model’s ability to capture complex dependencies in the data, leading to improved performance on a variety of language understanding tasks.

### Summary

- XLNet is a state-of-the-art language model developed by researchers at Google and Carnegie Mellon University. It is an extension and improvement over previous models like BERT (Bidirectional Encoder Representations from Transformers) and aims to overcome some of their limitations. Below, I’ll explain the key components, innovations, and details of XLNet.

#### Background

- Before diving into XLNet, it’s useful to understand its predecessors:
    
    - **Transformers:** The Transformer architecture, introduced in the paper “Attention is All You Need,” uses self-attention mechanisms to process input data in parallel, leading to significant improvements in training efficiency and performance over traditional RNNs (Recurrent Neural Networks) and LSTMs (Long Short-Term Memory networks).
        
    - **BERT:** BERT (Bidirectional Encoder Representations from Transformers) uses a bidirectional transformer to pre-train a language model by predicting masked tokens within a sentence. This bidirectional approach allows BERT to consider context from both left and right simultaneously, leading to improved performance on various NLP tasks.
        

#### Core Innovations of XLNet

- XLNet introduces several key innovations that enhance its performance and capabilities compared to BERT:

##### Permutation Language Modeling

- One of the major innovations of XLNet is its permutation-based language modeling objective. Instead of predicting masked tokens like BERT, XLNet predicts tokens in a permutation of the input sequence. This approach allows XLNet to model bidirectional context without needing to mask tokens.
    - **Permutation Generation:** Given a sequence, XLNet generates all possible permutations of the sequence order.
    - **Factorization Order:** XLNet uses a Transformer-XL architecture to learn the factorization order of these permutations, allowing it to consider various possible contexts for each token.

##### Autoregressive Method

- Unlike BERT’s masked language modeling, XLNet uses an autoregressive method, which is more like traditional language models. It predicts the next token in a sequence based on the preceding tokens, but thanks to the permutation mechanism, it can still leverage information from both directions.

##### Transformer-XL Backbone

- XLNet is built on top of Transformer-XL, which extends the Transformer architecture by introducing a segment-level recurrence mechanism and a relative positional encoding scheme. This allows XLNet to:
    - **Capture Longer Contexts:** Transformer-XL can process longer sequences by reusing hidden states from previous segments.
    - **Handle Dependencies:** The relative positional encoding helps the model better understand the positional relationships between tokens, which is crucial for tasks involving long-term dependencies.

#### Technical Details

##### Training Objective

- The training objective of XLNet is designed to maximize the likelihood of a sequence given its context from all possible permutations. This objective can be formally expressed using the following equation:
    
    P(x)=∏t=1TP(xzt|xz1,...,xzt−1)P(x)=∏t=1TP(xzt|xz1,...,xzt−1)
    
    - where zz is a permutation of the sequence indices.

##### Relative Positional Encodings

- XLNet uses relative positional encodings to capture the relationships between tokens more effectively than the absolute positional encodings used in BERT. This approach helps the model generalize better to different lengths of sequences.

##### Segment-Level Recurrence

- By leveraging the segment-level recurrence mechanism from Transformer-XL, XLNet can process longer texts more efficiently and retain contextual information across longer sequences. This mechanism enables the reuse of hidden states from previous segments, reducing the computational load and improving the model’s ability to understand long-range dependencies.

#### Performance and Applications

- XLNet has shown state-of-the-art performance on various NLP benchmarks, outperforming BERT on several tasks, including:
    
    - **Question Answering:** XLNet achieves better performance on datasets like SQuAD (Stanford Question Answering Dataset) due to its ability to capture more nuanced context.
    - **Text Classification:** The model’s improved contextual understanding helps in tasks like sentiment analysis and topic classification.
    - **Natural Language Inference:** XLNet excels in tasks that require understanding the relationship between pairs of sentences, such as entailment and contradiction detection.

#### Conclusion

- XLNet represents a significant advancement in the field of NLP by combining the strengths of autoregressive language modeling with bidirectional context understanding. Its innovative use of permutation language modeling, coupled with the Transformer-XL architecture, allows it to capture complex dependencies and long-range contexts more effectively than previous models like BERT. As a result, XLNet has set new benchmarks in various NLP tasks and continues to influence the development of more advanced language models.

## Advantages and Applications: Encoder vs. Decoder vs. Encoder-Decoder

- Encoder models are strong in tasks that require understanding and interpreting text. Their bidirectional context capture makes them suitable for tasks where understanding the full context of a sentence or document is crucial.
- Decoder models are excellent for generating text, making them ideal for creative tasks such as story generation, chatbot responses, and text completion.
- Encoder-decoder models provide a versatile architecture that can handle a wide range of tasks, from machine translation and text summarization to complex question answering and document generation. Their ability to understand and generate text makes them highly effective for tasks that require both deep comprehension and fluent text production.
- **For example, in machine translation, the encoder processes the input sentence in the source language to create a context vector, which the decoder then uses to generate the translation in the target language.** Similarly, in text summarization, the encoder reads and understands the original text, while the decoder generates a concise summary.

## Conclusion

- Unsupervised representation learning, through encoder, decoder, and encoder-decoder models, has revolutionized the field of natural language processing. By leveraging large-scale pretraining on vast text corpora, these models have achieved state-of-the-art performance across a variety of tasks. Each model type offers unique strengths, making them suitable for different applications in understanding and generating natural language. As research in this area continues to advance, we can expect even more sophisticated and capable models to emerge, further pushing the boundaries of what is possible in natural language processing.

## Further Reading

- [Transformer Text Embeddings](https://www.baeldung.com/cs/transformer-text-embeddings)
- [what is the first input to the decoder in a transformer model?](https://datascience.stackexchange.com/questions/51785/what-is-the-first-input-to-the-decoder-in-a-transformer-model)
- [What are the inputs to the first decoder layer in a Transformer model during the training phase?](https://datascience.stackexchange.com/questions/88981/what-are-the-inputs-to-the-first-decoder-layer-in-a-transformer-model-during-the)

## Citation

If you found our work useful, please cite it as:

![](https://aman.ai/images/copy.png)

`@article{Chadha2020DistilledEncoderDecoderEncDec,     title   = {Encoder vs. Decoder vs. Encoder-Decoder Models},   author  = {Chadha, Aman},   journal = {Distilled AI},   year    = {2020},   note    = {\url{https://aman.ai}} }`

-  [](https://github.com/amanchadha)|  [](https://citations.amanchadha.com/)|  [](https://twitter.com/i_amanchadha)|  [](mailto:hi@aman.ai)| 

[www.amanchadha.com](https://www.amanchadha.com/)