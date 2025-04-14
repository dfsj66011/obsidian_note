[Distilled AI](https://aman.ai/primers/ai/)[Back to aman.ai](https://aman.ai/)

# Natural Language Processing • Attention

- [Overview](https://aman.ai/primers/ai/attention/#overview)
    - [The Attention Mechanism](https://aman.ai/primers/ai/attention/#the-attention-mechanism)
    - [The Bottleneck Problem](https://aman.ai/primers/ai/attention/#the-bottleneck-problem)
    - [The Context Vector Bottleneck](https://aman.ai/primers/ai/attention/#the-context-vector-bottleneck)
    - [How Attention Solves the Bottleneck Problem](https://aman.ai/primers/ai/attention/#how-attention-solves-the-bottleneck-problem)
    - [Dynamic Focus on Relevant Input Parts](https://aman.ai/primers/ai/attention/#dynamic-focus-on-relevant-input-parts)
- [Origins of Attention](https://aman.ai/primers/ai/attention/#origins-of-attention)
- [Attention: Under the Hood](https://aman.ai/primers/ai/attention/#attention-under-the-hood)
- [The Classic Sequence-to-Sequence Model](https://aman.ai/primers/ai/attention/#the-classic-sequence-to-sequence-model)
- [Sequence-to-Sequence Model with Attention](https://aman.ai/primers/ai/attention/#sequence-to-sequence-model-with-attention)
- [Context Vector](https://aman.ai/primers/ai/attention/#context-vector)
- [Attention vs. Fixed-length Context Vector](https://aman.ai/primers/ai/attention/#attention-vs-fixed-length-context-vector)
- [Extensions to the Classic Attention Mechanism](https://aman.ai/primers/ai/attention/#extensions-to-the-classic-attention-mechanism)
- [Self-Attention / Scaled Dot-Product Attention](https://aman.ai/primers/ai/attention/#self-attention--scaled-dot-product-attention)
    - [Why Have Multiple Attention Layers?](https://aman.ai/primers/ai/attention/#why-have-multiple-attention-layers)
- [Comparative Analysis: Additive vs. Scaled Dot-Product Attention](https://aman.ai/primers/ai/attention/#comparative-analysis-additive-vs-scaled-dot-product-attention)
    - [Origins and Definitions](https://aman.ai/primers/ai/attention/#origins-and-definitions)
    - [Computational Efficiency](https://aman.ai/primers/ai/attention/#computational-efficiency)
    - [Theoretical Complexity](https://aman.ai/primers/ai/attention/#theoretical-complexity)
    - [Usage and Performance](https://aman.ai/primers/ai/attention/#usage-and-performance)
    - [Implementation Details](https://aman.ai/primers/ai/attention/#implementation-details)
    - [Conclusion](https://aman.ai/primers/ai/attention/#conclusion)
- [Multi-head Attention](https://aman.ai/primers/ai/attention/#multi-head-attention)
- [Cross Attention](https://aman.ai/primers/ai/attention/#cross-attention)
- [Ghost Attention](https://aman.ai/primers/ai/attention/#ghost-attention)
- [Linear Attention](https://aman.ai/primers/ai/attention/#linear-attention)
- [Lightning Attention](https://aman.ai/primers/ai/attention/#lightning-attention)
    - [Algorithm](https://aman.ai/primers/ai/attention/#algorithm)
        - [Core Steps](https://aman.ai/primers/ai/attention/#core-steps)
        - [Lightning Attention Forward Pass](https://aman.ai/primers/ai/attention/#lightning-attention-forward-pass)
    - [Pseudocode for Lightning Attention](https://aman.ai/primers/ai/attention/#pseudocode-for-lightning-attention)
- [Attention in Today’s Frontier LLMs](https://aman.ai/primers/ai/attention/#attention-in-todays-frontier-llms)
    - [Sliding-Window Multi-Query Attention](https://aman.ai/primers/ai/attention/#sliding-window-multi-query-attention)
        - [Components](https://aman.ai/primers/ai/attention/#components)
- [Multi-Head Latent Attention (MLA)](https://aman.ai/primers/ai/attention/#multi-head-latent-attention-mla)
    - [Key Equations and Design](https://aman.ai/primers/ai/attention/#key-equations-and-design)
    - [Implementation Details](https://aman.ai/primers/ai/attention/#implementation-details-1)
- [References](https://aman.ai/primers/ai/attention/#references)
- [Citation](https://aman.ai/primers/ai/attention/#citation)

## Overview

### The Attention Mechanism

- The attention mechanism has revolutionized many Natural Language Processing (NLP) and Computer Vision (CV) tasks by addressing the limitations of traditional seq2seq models by alleviating the context vector bottleneck. Attention enables models to dynamically focus on relevant parts of the input sequence, enhancing their ability to handle long and complex sentences.
- This improvement has been pivotal in advancing the performance and interpretability of AI models across a wide range of NLP applications. It has led to significant improvements in various applications such as machine translation, text summarization, and question answering.

### The Bottleneck Problem

- To understand the importance of attention, it is crucial to first grasp the bottleneck problem that attention helps to solve. In traditional sequence-to-sequence (seq2seq) models, such as those used in early neural machine translation systems, the architecture typically comprises an encoder and a decoder.
    - **Encoder**: Processes the input sequence (e.g., a sentence in the source language) and compresses it into a fixed-size context vector.
    - **Decoder**: Uses this context vector to generate the output sequence (e.g., a sentence in the target language).

### The Context Vector Bottleneck

- The main issue with this architecture is the **context vector bottleneck**. This bottleneck arises because the entire input sequence must be condensed into a single, fixed-size vector, regardless of the length or complexity of the input. As a result, crucial information can be lost, especially for long or complex sentences. This limitation hampers the model’s ability to capture and retain important details, leading to suboptimal performance.

### How Attention Solves the Bottleneck Problem

- The attention mechanism mitigates the context vector bottleneck by allowing the model to dynamically access different parts of the input sequence during the generation of each output element. Instead of relying on a single fixed-size context vector, the attention mechanism computes a weighted combination of all the encoder’s hidden states. This weighted sum acts as the context for each output step, enabling the model to focus on the most relevant parts of the input sequence.

### Dynamic Focus on Relevant Input Parts

- Here’s how the attention mechanism works in detail:
    
    1. **Alignment Scores**: For each decoder time step, alignment scores are computed between the current decoder hidden state and each encoder hidden state. These scores indicate how well the current part of the output aligns with different parts of the input.
        
    2. **Attention Weights**: The alignment scores are passed through a softmax function to obtain attention weights. These weights sum to 1 and represent the importance of each encoder hidden state for the current decoder time step.
        
    3. **Context Vector**: The context vector for the current decoder time step is computed as a weighted sum of the encoder hidden states, using the attention weights.
        
    4. **Output Generation**: The decoder uses this context vector, along with its own hidden state, to generate the next token in the output sequence.
        
- By allowing the model to focus on different parts of the input sequence as needed, attention provides several benefits:
    
    - **Improved Handling of Long Sequences**: The model can retain and utilize relevant information from any part of the input sequence, which is especially beneficial for longer sentences.
    - **Better Interpretability**: The attention weights offer insights into which parts of the input the model is focusing on, making the model’s decision-making process more transparent.
    - **Enhanced Performance**: By addressing the bottleneck problem, attention leads to more accurate and fluent translations or generated text in various NLP tasks.

## Origins of Attention

- In the context of NLP, the attention mechanism was first introduced in [“Neural Machine Translation by Jointly Learning to Align and Translate”](https://arxiv.org/pdf/1409.0473.pdf) at ICLR 2015 by Bahdanau et al. (2015). This served as a foundation upon which the self-attention mechanism in the Transformer paper was based on.
- This was proposed in the context of machine translation, where given a sentence in one language, the model has to produce a translation for that sentence in another language.
- In the paper, the authors propose to tackle the problem of a fixed-length context vector in the original **seq2seq** model for machine translation in [“Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation”](https://www.aclweb.org/anthology/D14-1179/) by Cho et al. (2014).
- The following slide from [Stanford’s CS25 course](https://youtu.be/XfpMkf4rD6E?t=1150) shows how the attention mechanism was conceived and is a perfect illustration of why AI/ML is an empirical field, built on intuition.

## Attention: Under the Hood

- As previously discussed, the role of attention in a model is to strategically focus on pertinent segments of the input sequence as and when required. This ability to tune into relevant sections enhances the model’s overall processing efficiency.
- In a shift from traditional practices, the encoder now funnels a significantly larger amount of data to the decoder. Rather than simply transmitting the last hidden state of the encoding phase, it channels all the hidden states to the decoder, ensuring a more comprehensive data transfer.
- A decoder utilizing attention features undertakes an additional step before generating its output. This step is designed to ensure the decoder’s focus is appropriately honed on parts of the input that are relevant to the current decoding time step. To achieve this, the following operations are performed:
    - Each hidden state is multiplied by its respective softmax score. This results in an amplification of hidden states associated with high scores and effectively diminishes the impact of those with low scores. This selective amplification technique supports the model’s ability to maintain focus on the more relevant parts of the input.
- In an encoder, we employ the mechanism of self-attention. This technique allows the model to focus on different parts of the input independently, assisting the overall understanding of the sequence.
- Conversely, in a decoder, cross-attention is applied. This allows the decoder to focus on different parts of the encoder’s output, aiding in the generation of a more accurate translation or summary.
- With each step of the decoding process, a direct connection to the encoder is utilized to strategically zero in on a specific part of the input. This connection enables the model to maintain accuracy while parsing complex sequences.

## The Classic Sequence-to-Sequence Model

- The seq2seq model is composed of two main components: an encoder, and a decoder, as shown in the figure ([source](https://towardsdatascience.com/understanding-encoder-decoder-sequence-to-sequence-model-679e04af4346)) below:

![](https://aman.ai/primers/ai/assets/attention/seq2seq.jpg)

- The **encoder** reads the input sentence, a sequence of vectors
    
    x=(x1,…,xT)x=(x1,…,xT)
    
    , into a fixed-length vector
    
    cc
    
    . The **encoder** is a recurrent neural network, typical approaches are GRU or LSTMs such that:
    
    ht=f (xt,ht−1)ht=f (xt,ht−1)
    
    c=q (h1,…,hT)c=q (h1,…,hT)
    
    - where
        
        htht
        
        is a hidden state at time
        
        tt
        
        , and
        
        cc
        
        is a vector generated from the sequence of the hidden states, and
        
        ff
        
        and
        
        qq
        
        are some nonlinear functions.
- At every time-step
    
    tt
    
    the encoder produces a hidden state
    
    htht
    
    , and the generated context vector is modeled according to all hidden states.
    
- The **decoder** is trained to predict the next word
    
    ytyt
    
    given the context vector
    
    cc
    
    and all the previously predict words
    
    {y1,…,yt−1}{y1,…,yt−1}
    
    , it defines a probability over the translation
    
    yy
    
    by decomposing the joint probability:
    
    p(y)=∏i=1xp(yt|y1,…,yt−1,c)p(y)=∏i=1xp(yt|y1,…,yt−1,c)
    
    - where
        
        y={y1,…,yt}y={y1,…,yt}
        
        . In other words, the probability of a translation sequence is calculated by computing the conditional probability of each word given the previous words. With an LSTM/GRU each conditional probability is computed as:
    
    p(yt|y1,…,yt−1,c)=g(yt−1,st,c)p(yt|y1,…,yt−1,c)=g(yt−1,st,c)
    
    - where,
        
        gg
        
        is a nonlinear function that outputs the probability of
        
        ytyt
        
        ,
        
        stst
        
        is the value of the hidden state of the current position, and
        
        cc
        
        the context vector.
- In a simple **seq2seq** model, the last output of the LSTM/GRU is the context vector, encoding context from the entire sequence. This context vector is then used as the initial hidden state of the decoder.
    
- At every step of decoding, the decoder is given an input token and (the previous) hidden state. The initial input token is the start-of-string `<SOS>` token, and the first hidden state is the context vector (the encoder’s last hidden state).
    
- So, the fixed size context-vector needs to contain a good summary of the meaning of the whole source sentence, being this one big bottleneck, specially for long sentences. The figure below (taken from [Bahdanau et al.](https://arxiv.org/pdf/1409.0473.pdf) (2015)) shows how the performance of the seq2seq model varies by sentence length:
    

![](https://aman.ai/primers/ai/assets/attention/seq2seq_long_sentences.png)

## Sequence-to-Sequence Model with Attention

- The fixed size context-vector bottleneck was one of the main motivations by [Bahdanau et al. (2015)](https://arxiv.org/pdf/1409.0473.pdf), which proposed a similar architecture but with a crucial improvement:

> “_The new architecture consists of a bidirectional RNN as an encoder and a decoder that emulates searching through a source sentence during decoding a translation_”

- The **encoder** is now a bidirectional recurrent network with a forward and backward hidden states. A simple concatenation of the two hidden states represents the encoder state at any given position in the sentence. The motivation is to include both the preceding and following words in the representation/annotation of an input word.
    
- The other key element, and the most important one, is that the decoder is now equipped with some sort of search, allowing it to look at the whole source sentence when it needs to produce an output word, the **attention mechanism**. The figure below (taken from [Bahdanau et al.](https://arxiv.org/pdf/1409.0473.pdf) (2015)) illustrates the attention mechanism in a seq2seq model.
    

![](https://aman.ai/primers/ai/assets/attention/seq2seq_with_attention.png)

- The figure above gives a good overview of this new mechanism. To produce the output word at time
    
    ytyt
    
    the decoder uses the last hidden state from the decoder - one can think about this as some sort of representation of the already produced words - and a dynamically computed context vector based on the input sequence.
    
- The authors proposed to replace the fixed-length context vector by a another context vector
    
    cici
    
    which is a sum of the hidden states of the input sequence, weighted by alignment scores.
    
- Note that now the probability of each output word is conditioned on a distinct context vector
    
    cici
    
    for each target word
    
    yy
    
    .
    
- The new decoder is then defined as:
    
    p(yt|y1,…,yt−1,c)=g(yt−1,si,c)p(yt|y1,…,yt−1,c)=g(yt−1,si,c)
    
    - where
        
        sisi
        
        is the hidden state for time
        
        ii
        
        , computed by:
    
    si=f(si−1,yi−1,ci)si=f(si−1,yi−1,ci)
    
    - that is, a new hidden state for
        
        ii
        
        depends on the previous hidden state, the representation of the word generated by the previous state and the context vector for position
        
        ii
        
        . The lingering question now is, how to compute the context vector
        
        cici
        
        ?

![](https://aman.ai/primers/ai/assets/13.jpg)

- Instead of source and target sentences, we also have 2 sequences: passage and question(lengths are imbalance)
- We need to model which words in the passage are most relevant to the question (and which question words)
- Attention is the key ingredient here, similar to which words in source sentences are most relevant to the current target word![](https://aman.ai/primers/ai/assets/14.jpg)

## Context Vector

- “In attention, the query refers to the word we’re computing attention for. In the case of an encoder, the query vector points to the current input word (aka context). For example, if the context was the first word in the input sentence, it would have a query vector q1.” [(source)](https://eugeneyan.com/writing/attention/)
- The context vector
    
    cici
    
    is a sum of the hidden states of the input sequence, weighted by alignment scores. Each word in the input sequence is represented by a concatenation of the two (i.e., forward and backward) RNNs hidden states, let’s call them annotations.
    
- Each annotation contains information about the whole input sequence with a strong focus on the parts surrounding the
    
    ithith
    
    word in the input sequence.
    
- The context vector
    
    cici
    
    is computed as a weighted sum of these annotations:

ci=∑j=1Txαijhjci=∑j=1Txαijhj

- The weight
    
    αijαij
    
    of each annotation
    
    hjhj
    
    is computed by:
    
    αij=softmax(eij)αij=softmax(eij)
    
    - where eij=a(si−1,hj)eij=a(si−1,hj)
- aa
    
    is an alignment model which scores how well the inputs around position
    
    jj
    
    and the output at position
    
    ii
    
    match. The score is based on the RNN hidden state
    
    si−1si−1
    
    (just before emitting
    
    yiyi
    
    and the
    
    jthjth
    
    annotation
    
    hjhj
    
    of the input sentence
    
    a(si−1,hj)=v⊤atanh(Wa si−1+Ua hj)a(si−1,hj)=va⊤tanh⁡(Wa si−1+Ua hj)
    
    - where both
        
        vava
        
        and
        
        WaWa
        
        are weight matrices to be learned in the alignment model.
- The alignment model in the paper is described as feed forward neural network whose weight matrices
    
    vava
    
    and
    
    WaWa
    
    are learned jointly together with the whole graph/network.
    
- The authors note:
    

> “The probability
> 
> αijhjαijhj
> 
> reflects the importance of the annotation
> 
> hjhj
> 
> with respect to the previous hidden state
> 
> si−1si−1
> 
> in deciding the next state
> 
> sisi
> 
> and generating
> 
> yiyi
> 
> . Intuitively, this implements a mechanism of attention in the decoder.”

## Attention vs. Fixed-length Context Vector

- Let’s visually review the attention mechanism and compare it against the fixed-length context vector approach. The pictures below (credit: [Nelson Zhao](https://github.com/NELSONZHAO)) help understand the difference between the two encoder-decoder approaches. The figure below illustrates the encoder-decoder architecture with a fixed-context vector.

![](https://aman.ai/primers/ai/assets/attention/attention_seq2seq_context.jpg)

- On the other hand, the figure below illustrates the Encoder-Decoder architecture with attention mechanism proposed in [“Neural Machine Translation by Jointly Learning to Align and Translate”](https://arxiv.org/pdf/1409.0473.pdf) by Bahdanau et al. (2015).

![](https://aman.ai/primers/ai/assets/attention/attention_seq2seq_context_with_attention.jpg)

## Extensions to the Classic Attention Mechanism

- [Luong et al.](https://www.aclweb.org/anthology/D15-1166/) (2015) proposed and compared other mechanisms of attentions, more specifically, alternative functions to compute the alignment score:

![](https://aman.ai/primers/ai/assets/attention/alignment_scores.png)

- Note that the _concat_ operation is the same as in [Bahdanau et al.](https://arxiv.org/pdf/1409.0473.pdf) (2015); however, instead of a weighted average over all the source hidden states, they proposed a mechanism of local attention which focus only on a small subset of the source positions per target word instead of attending to all words on the source for each target word.

## Self-Attention / Scaled Dot-Product Attention

- Earlier, we looked into the “classic” attention mechanism on which subsequent techniques such as **self-attention** or **query-key-value-attention** are based.
- After transforming the field of neural machine translation, the **attention** mechanism was applied to other natural language processing tasks, such as document-level classification or sequence labelling and further extended to other modalities such as [vision](https://arxiv.org/abs/2010.11929) and [speech](https://arxiv.org/abs/2106.07447).
- Please refer the [Self-Attention](https://aman.ai/primers/ai/transformers/#self-attention) section in our [Transformer](https://aman.ai/primers/ai/transformers/#multi-head-attention) primer.

### Why Have Multiple Attention Layers?

- Per Eugene Yan’s [Some Intuition on Attention and the Transformer](https://eugeneyan.com/writing/attention/) blog, multiple attention layers builds in redundancy (on top of having multiple attention heads). If we only had a single attention layer, that attention layer would have to do a flawless job—this design could be brittle and lead to suboptimal outcomes. We can address this via multiple attention layers, where each one uses the output of the previous layer with the [safety net of skip connections](https://aman.ai/primers/ai/attention/#why-have-skip-connections). Thus, if any single attention layer messed up, the skip connections and downstream layers can mitigate the issue.
- Stacking attention layers also broadens the model’s receptive field. The first attention layer produces context vectors by attending to interactions between pairs of words in the input sentence. Then, the second layer produces context vectors based on pairs of pairs, and so on. With more attention layers, the Transformer gains a wider perspective and can attend to multiple interaction levels within the input sentence.

## Comparative Analysis: Additive vs. Scaled Dot-Product Attention

- Among the various types of attention mechanisms, additive attention and scaled dot-product attention are the most commonly used. Here’s a comparison:

### Origins and Definitions

- **Additive Attention:**
    - Proposed by Bahdanau et al. in their 2015 paper titled “[Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473).”
    - It computes the alignment score between the query qq and the key kk using a feed-forward neural network with a single hidden layer.
    - The formula for the alignment score eijeij is: eij=vTtanh(Wqqi+Wkkj)eij=vTtanh⁡(Wqqi+Wkkj)
    - Here, WqWq and WkWk are learnable weight matrices, and vv is a learnable vector.
- **Scaled Dot-Product Attention:**
    - Introduced in [Attention Is All You Need](https://arxiv.org/abs/1706.03762) by Vaswani et al. in 2017.
    - It computes the alignment score by taking the dot product of the query and key vectors, scaled by the square root of the dimension of the key vectors (dkdk).
    - The formula for the alignment score eijeij is: eij=qi⋅kjdk√eij=qi⋅kjdk

### Computational Efficiency

- **Additive Attention:**
    - Involves a more complex computation due to the use of a feed-forward network.
    - While theoretically similar in complexity to dot-product attention, it is generally slower in practice because it cannot leverage highly optimized matrix multiplication libraries.
    - Requires additional parameters WqWq, WkWk, and vv, increasing memory usage.
- **Scaled Dot-Product Attention:**
    - Much faster and more space-efficient as it relies on matrix multiplication, which is highly optimized in modern deep learning libraries (e.g., TensorFlow, PyTorch).
    - The scaling factor 1dk√1dk helps to mitigate the issue of having large dot product values, which can lead to small gradients during backpropagation.

### Theoretical Complexity

- Both attention mechanisms have a theoretical time complexity of O(n2⋅d)O(n2⋅d), where nn is the sequence length and dd is the dimension of the representations.
- However, in practice:
    - **Additive Attention** involves additional computation for the feed-forward network, which can slow down the process.
    - **Scaled Dot-Product Attention** benefits from efficient matrix multiplication operations, making it faster in real-world applications.

### Usage and Performance

- **Additive Attention:**
    - Often used in earlier models of neural machine translation and other NLP tasks before the advent of the Transformer architecture.
    - Still useful in scenarios where the performance benefits of dot-product attention do not outweigh its simplicity and interpretability.
- **Scaled Dot-Product Attention:**
    - Integral to the Transformer architecture, which has become the standard for many NLP tasks.
    - Scales better with larger datasets and more complex models, leading to state-of-the-art performance in a wide range of applications.

### Implementation Details

- **Additive Attention:**
    - Typically implemented with separate weight matrices for the query and key vectors, followed by a non-linear activation (e.g., tanhtanh) and a final linear layer to compute the score.
    - Example pseudocode:
        
        ![](https://aman.ai/images/copy.png)
        
        `def additive_attention(query, key):     w_q = nn.Linear(query_dim, hidden_dim)     w_k = nn.Linear(key_dim, hidden_dim)     v = nn.Linear(hidden_dim, 1)     scores = v(tanh(w_q(query) + w_k(key)))     attention_weights = softmax(scores, dim=-1)     return attention_weights`
        
- **Scaled Dot-Product Attention:**
    - Implemented using matrix multiplication followed by a scaling factor and softmax function to compute the attention weights.
    - Example pseudocode:
        
        ![](https://aman.ai/images/copy.png)
        
        `def scaled_dot_product_attention(query, key, value, mask=None):     d_k = query.size(-1)     scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)     if mask is not None:         scores = scores.masked_fill(mask == 0, -1e9)     attention_weights = softmax(scores, dim=-1)     output = torch.matmul(attention_weights, value)     return output`
        

### Conclusion

- **Additive Attention** is more complex and computationally intensive but has been foundational in early NLP models.
- **Scaled Dot-Product Attention** is faster, more efficient, and scalable, making it the preferred choice in modern architectures like Transformers.
- The choice between the two often depends on the specific application requirements and the computational resources available. However, for most state-of-the-art NLP tasks, scaled dot-product attention is the go-to mechanism due to its performance and efficiency advantages.

## Multi-head Attention

- Please refer the [Multi-Head Attention](https://aman.ai/primers/ai/transformers/#multi-head-attention) section in our [Transformer](https://aman.ai/primers/ai/transformers/#multi-head-attention) primer.

## Cross Attention

- Please refer the [Cross Attention](https://aman.ai/primers/ai/transformers/#cross-attention) section in our [Transformer](https://aman.ai/primers/ai/transformers/#multi-head-attention) primer.

## [Ghost Attention](https://ai.meta.com/research/publications/llama-2-open-foundation-and-fine-tuned-chat-models/)

- The authors of [Llama 2](https://ai.meta.com/research/publications/llama-2-open-foundation-and-fine-tuned-chat-models/) proposed Ghost Attention (GAtt).
- Ghost Attention (GAtt) is an innovative technique specifically designed to aid artificial intelligence in remembering and adhering to initial instructions throughout a conversation. This methodology extends the notion of Context Distillation, where specific details are distilled and highlighted from the broader context to enhance understanding.
    - Context Distillation is a concept that focuses on highlighting and isolating specific, crucial details from a larger and more complex context. This process is similar to distilling, where the essential elements are extracted from a compound mixture.Context Distillation is used to introduce and retain an instruction throughout a dialogue. This helps the model to consistently remember and adhere to the instruction, enhancing its ability to maintain focus and perform accurately.
- In this technique, an instruction - a directive that must be consistently followed during the entire dialogue - is added to all user messages in a synthetic dialogue dataset. However, during the training phase, the instruction is only retained in the first turn of the dialogue and the loss (a measure of error) is set to zero for all tokens (representative units of information) from earlier turns.
- The authors applied this unique approach across a variety of synthetic constraints, which included diverse elements like hobbies, languages, and public figures. Implementing GAtt effectively preserved attention on initial instructions for a significantly larger portion of the conversation, ensuring that the AI stayed focused on its tasks.
- One of the notable achievements of GAtt is its ability to maintain consistency in adhering to initial instructions even over extended dialogues, comprising more than 20 turns, until it hits the maximum context length that the model can handle. While this first iteration has proven successful, the authors believe that there is ample room for further refinement and improvement, suggesting that the Ghost Attention technique can continue to evolve for enhanced performance.
- Let’s say we are training a dialogue system to book appointments for a dental clinic, and one of the rules we want the system to follow is that it should always inquire about the patient’s dental insurance details.
- In the synthetic dialogue dataset used for training, we append the instruction “Always ask about dental insurance” to every user message.
- For example:
    - User: “I need an appointment.”
    - AI (with instruction): “Always ask about dental insurance. Sure, I can help you with that. Do you have a preferred date and time?”
    - User: “How about next Tuesday at 10 am?”
    - AI (with instruction): “Always ask about dental insurance. That time works. May I also ask if you have dental insurance and, if so, could you provide the details?”
- During training, GAtt retains this instruction only in the first turn and sets the loss to zero for all tokens from earlier turns. The model will be trained to understand that asking about dental insurance is an important part of the conversation, and it should remember this instruction even in later turns.
    
- For example, when the model is actually deployed:
    - User: “I need an appointment.”
    - AI: “Sure, I can help you with that. Do you have a preferred date and time?”
    - User: “How about next Tuesday at 10 am?”
    - AI: “That time works. May I also ask if you have dental insurance and, if so, could you provide the details?”
- Notice that even though the instruction “Always ask about dental insurance” is not explicitly mentioned during the conversation after training, the AI system consistently adheres to it throughout the dialogue, as it has been trained using GAtt.
- This technique ensures the AI model stays focused on the initial instruction, in this case, asking about dental insurance, enhancing its dialogue capabilities and making it more reliable for the task at hand.

## [Linear Attention](https://arxiv.org/abs/2006.04768)

- Proposed in [Linformer: Self-Attention with Linear Complexity](https://arxiv.org/abs/2006.04768) by Wang et al. from Facebook AI.
- The authors proposes a novel approach to optimizing the self-attention mechanism in Transformer models, reducing its complexity from quadratic to linear with respect to sequence length. This method, named Linformer, maintains competitive performance with standard Transformer models while significantly enhancing efficiency in both time and memory usage.
- Linformer introduces a low-rank approximation of the self-attention mechanism. By empirically and theoretically demonstrating that the self-attention matrix is of low rank, the authors propose a decomposition of the original scaled dot-product attention into multiple smaller attentions via linear projections. This factorization effectively reduces both the space and time complexity of self-attention from O(n2)O(n2) to O(n)O(n), addressing the scalability issues of traditional Transformers.
- The model architecture involves projecting key and value matrices into lower-dimensional spaces before computing the attention, which retains the model’s effectiveness while reducing computational demands. The approach includes options for parameter sharing across projections, which can further reduce the number of trainable parameters without significantly impacting performance.
- In summary, here’s how Linformer achieves linear-time attention:
    
    1. **Low-Rank Approximation**: The core idea behind Linformer is the observation that self-attention can be approximated by a low-rank matrix. This implies that the complex relationships captured by self-attention in Transformers do not necessarily require a full rank matrix, allowing for a more efficient representation.
        
    2. **Reduced Complexity**: While standard self-attention mechanisms in Transformers have a time and space complexity of O(n2)O(n2) with respect to the sequence length (n), Linformer reduces this complexity to O(n)O(n). This significant reduction is both in terms of time and space, making it much more efficient for processing longer sequences.
        
    3. **Mechanism of Linear Self-Attention**: The Linformer achieves this by decomposing the scaled dot-product attention into multiple smaller attentions through linear projections. Specifically, it introduces two linear projection matrices EiEi and FiFi which are used when computing the key and value matrices. By first projecting the original high-dimensional key and value matrices into a lower-dimensional space (n×kn×k), Linformer effectively reduces the complexity of the attention mechanism.
        
    4. **Combination of Operations**: The combination of these operations forms a low-rank factorization of the original attention matrix. Essentially, Linformer simplifies the computational process by approximating the full attention mechanism with a series of smaller, more manageable operations that collectively capture the essential characteristics of the original full-rank attention.
        
- The figure below from the paper shows: (left and bottom-right) architecture and example of the proposed multihead linear self-attention; (top right) inference time vs. sequence length the various Linformer models.

![](https://aman.ai/images/papers/Linformer.jpg)

- Experimental validation shows that Linformer achieves similar or better performance compared to the original Transformer on standard NLP tasks such as sentiment analysis and question answering, using datasets like GLUE and IMDB reviews. Notably, the model offers considerable improvements in training and inference speeds, especially beneficial for longer sequences.
- Additionally, various strategies for enhancing the efficiency of Linformer are tested, including different levels of parameter sharing and the use of non-uniform projected dimensions tailored to the specific demands of different layers within the model.
- The authors suggest that the reduced computational requirements of Linformer not only make high-performance models more accessible and cost-effective but also open the door to environmentally friendlier AI practices due to decreased energy consumption.
- In summary, Linformer proposes a more efficient self-attention mechanism for Transformers by leveraging the low-rank nature of self-attention matrices. This approach significantly reduces the computational burden, especially for long sequences, by lowering the complexity of attention calculations from quadratic to linear in terms of both time and space. This makes Linformer an attractive choice for tasks involving large datasets or long sequence inputs, where traditional Transformers might be less feasible due to their higher computational demands.

## Lightning Attention

- Proposed in [MiniMax-01: Scaling Foundation Models with Lightning Attention](https://filecdn.minimax.chat/_Arxiv_MiniMax_01_Report.pdf) by [MiniMax](https://www.minimaxi.com/en/news/minimax-01-series-2).

### Algorithm

- The lightning attention mechanism is a highly optimized implementation of linear attention, designed to achieve both linear complexity and scalability across long sequence lengths. Below is the detailed algorithm used for the forward pass of lightning attention.

#### Core Steps

1. **Input Partitioning**:
    - The input matrices QQ, KK, and VV are divided into blocks of size B×dB×d, where BB is the block size, and dd is the feature dimension.
2. **Initialization**:
    - Initialize a cumulative key-value matrix KV=0KV=0 of shape d×dd×d.
    - Create a mask MM to handle causal attention, where Mts=1Mts=1 if t≥st≥s, otherwise 0.
3. **Block-Wise Computation**:
    - For each block tt:
        - **Intra-Block Attention**: Compute intra-block attention scores using the left product operation.
        - **Inter-Block Attention**: Update the cumulative key-value matrix KVKV and compute the inter-block contributions using the right product operation.
        - Combine the intra- and inter-block results to produce the final output for the block.
4. **Output Assembly**:
    - Concatenate the outputs of all blocks to form the final output matrix OO.

#### Lightning Attention Forward Pass

- The following figure from the paper shows the algorithm for the lightning attention forward pass:

![](https://aman.ai/images/papers/LightningAttnAlgo.jpg)

- **Input**: Query (QQ), Key (KK), Value (VV) matrices of shape (n,d)(n,d), block size (BB)
- **Output**: Output matrix (OO) of shape (n,d)(n,d)
    
- **Steps**:
    
    1. Initialize:
        - Cumulative Key-Value matrix (KV=0KV=0) of shape (d,d)(d,d)
        - Causal mask MM for intra-block operations
        - Output matrix OO of shape (n,d)(n,d)
    2. Divide QQ, KK, and VV into T=⌈n/B⌉T=⌈n/B⌉ blocks.
        
    3. For each block t∈[1,T]t∈[1,T]:
        - Extract current block Qt,Kt,VtQt,Kt,Vt.
        - Compute Intra-Block Attention using: Ointra=(QtKTt⊙M)VtOintra=(QtKtT⊙M)Vt
        - Compute Inter-Block Attention using: Ointer=QtKVOinter=QtKV
        - Update KVKV with: KV=KV+KTtVtKV=KV+KtTVt
        - Combine results: Ot=Ointra+OinterOt=Ointra+Ointer
    4. Return OO.

### Pseudocode for Lightning Attention

![](https://aman.ai/images/copy.png)

`def lightning_attention(Q, K, V, block_size):     """     Lightning Attention Forward Pass     Args:         Q: Query matrix of shape (n, d)         K: Key matrix of shape (n, d)         V: Value matrix of shape (n, d)         block_size: Size of each block (B)     Returns:         O: Output matrix of shape (n, d)     """     n, d = Q.shape  # Sequence length and feature dimension     num_blocks = (n + block_size - 1) // block_size  # Total number of blocks     KV = np.zeros((d, d))  # Initialize cumulative key-value matrix     mask = np.tril(np.ones((block_size, block_size)))  # Causal mask     # Initialize the output matrix     O = np.zeros_like(Q)      for t in range(num_blocks):         # Load current block         start = t * block_size         end = min((t + 1) * block_size, n)         Q_t = Q[start:end, :]         K_t = K[start:end, :]         V_t = V[start:end, :]          # Intra-block computation (left product)         intra_block = (Q_t @ K_t.T * mask[:end - start, :end - start]) @ V_t          # Inter-block computation (right product)         inter_block = Q_t @ KV          # Update cumulative key-value matrix         KV += K_t.T @ V_t          # Combine intra- and inter-block results         O[start:end, :] = intra_block + inter_block      return O`

## Attention in Today’s Frontier LLMs

- Credits to the following section go to [The AIEdge](https://www.linkedin.com/company/the-aiedge-newsletter/).
- While the Transformers of 2017 implemented attention computation that scaled quadratically, this no longer holds true with recent Transformer models.
- Significant advancements have been made in the computation of attentions since the introduction of GPT-3. Most large language models now employ sub-quadratic attention mechanisms, and many implementations have achieved constant space complexity. Innovations such as Paged-Attention and Flash Attention 1 and 2 have allowed for more efficient read-write access on hardware. Consequently, many open-source projects have moved beyond standard PyTorch implementations to accommodate enhanced hardware utilization.

### Sliding-Window Multi-Query Attention

- In the Mistral-7B model, a sliding-window multi-query attention mechanism with an efficient memory implementation is employed. The sliding window multi-head attention mechanism is a specialized variant of attention that is efficient for long sequences by focusing on local context through the use of sliding windows. This approach reduces computational complexity compared to traditional full attention mechanisms. Below is an implementation of a fully vectorized sliding-window multihead attention mechanism. The time complexity is approximately O(Nw)O(Nw), where ww represents the window size. This approach necessitates at least contextsizewcontextsizew decoder blocks to fully encompass the entire context size.

![](https://aman.ai/primers/ai/assets/attention/AttentionTodayLLMs.jpeg)

#### Components

- Here’s a breakdown of the code’s components and functionalities:
    
    1. **Class Definition (`SlidingWindowMultiheadAttention`)**:
        - Inherits from `nn.Module`, making it a PyTorch module.
        - Takes parameters such as `hidden_size`, `num_heads`, and `window_size` during initialization.
    2. **Initialization Method (`__init__`)**:
        - Ensures that `hidden_size` is divisible by `num_heads` for equal division among the heads.
        - Sets up various attributes including `num_heads`, `head_dim` (dimension per head), `window_size`, and linear transformations (`qkv_linear` and `out`):
            - `qkv_linear`: A linear layer that projects input `x` into queries, keys, and values.
            - `out`: A linear layer to transform the concatenated output from all attention heads back to the original hidden size.
    3. **Forward Method (`forward`)**:
        - Takes an input tensor `x` and processes it through the following steps:
            - **Input Reshaping**: Determines the shape parameters from the input tensor (`batch_size`, `seq_length`, `hidden_size`).
            - **Padding**: Calculates padding to be applied for the sliding window mechanism, which is half the window size.
            - **Query, Key, Value Computation**:
                - Uses the `qkv_linear` layer to produce combined queries, keys, and values.
                - Reshapes and permutes the combined tensor to separate queries, keys, and values.
            - **Sliding Window Mechanism**:
                - Applies padding to keys and values.
                - Unfolds keys and values to create sliding window segments.
            - **Attention Computation**:
                - Calculates dot product attention scores between queries and keys from the windows.
                - Applies softmax normalization on scores scaled by the square root of the head dimension.
                - Computes the weighted sum of values based on these attention scores.
            - **Merging Heads and Final Linear Transformation**:
                - Reshapes the context to merge the heads.
                - Passes the merged context through the `out` linear layer to produce the final output.
    4. **Return**:
        - Returns the output of the module after processing through the attention mechanism and linear transformation.

## Multi-Head Latent Attention (MLA)

- The Multi-Head Latent Attention (MLA) mechanism was proposed in [DeepSeek-V3](https://arxiv.org/abs/2412.19437) is a refined adaptation of the traditional multi-head attention. MLA focuses on compressing the key-value (KV) cache and reducing activation memory, enabling efficient inference without significant performance degradation.

### Key Equations and Design

- Let dd denote the embedding dimension, nhnh the number of attention heads, and dhdh the dimension per head. Given the attention input for the tthtth token at a specific attention layer, ht∈ℝdht∈Rd, MLA introduces a low-rank joint compression mechanism for both keys and values. The primary components are:
    
    1. **Compressed Latent Vector for Keys and Values**: cKVt=WDKVhtcKVt=WDKVht
        - where cKVt∈ℝdccKVt∈Rdc is the compressed latent vector, dc≪dhnhdc≪dhnh, and WDKV∈ℝdc×dWDKV∈Rdc×d is the down-projection matrix.
    2. **Key Representations**: kC=WUKcKVt,kR=RoPE(WKRht),kt=[kC;kR]kC=WUKcKVt,kR=RoPE(WKRht),kt=[kC;kR]
        - where WUKWUK and WKRWKR are up-projection matrices, and RoPE(⋅)RoPE(⋅) applies Rotary Positional Embeddings (RoPE). ktkt concatenates compressed kCkC and kRkR.
    3. **Value Representations**: vC=WUVcKVtvC=WUVcKVt
        
    4. **Query Compression**: Similarly, queries are compressed using: cQt=WDQht,qC=WUQcQt,qR=RoPE(WQRcQt),qt=[qC;qR]cQt=WDQht,qC=WUQcQt,qR=RoPE(WQRcQt),qt=[qC;qR]
        - where WDQWDQ, WUQWUQ, and WQRWQR are respective projection matrices for queries.
    5. **Final Attention Output**: The output utut for the tthtth token is computed as: ot,i=∑tj=1Softmaxj(q⊤t,ikj,idh+dR√)vC,j,i,ut=WO[ot,1;ot,2;…;ot,nh]ot,i=∑j=1tSoftmaxj(qt,i⊤kj,idh+dR)vC,j,i,ut=WO[ot,1;ot,2;…;ot,nh]
        - where WO∈ℝd×dhnhWO∈Rd×dhnh is the output projection matrix.

### Implementation Details

1. **KV Cache Reduction**: Only the compressed latent vectors cKVtcKVt and kRkR need to be cached during inference. This significantly reduces the KV cache size, especially for models with large sequences or extensive parameterization.
    
2. **Query Compression**: Similar to KV compression, query representations are also compressed, reducing activation memory during training.
    
3. **Performance Optimization**: Despite the reduced cache and memory requirements, MLA achieves performance comparable to traditional multi-head attention through:
    - Efficient RoPE application for positional encoding.
    - Proper scaling and projection mechanisms to preserve information fidelity.
4. **Inference Efficiency**: By caching only the compressed representations, MLA minimizes memory overhead during inference while maintaining the ability to focus dynamically on the most relevant tokens.

## References

- [An Introductory Survey on Attention Mechanisms in NLP Problems](https://link.springer.com/chapter/10.1007/978-3-030-29513-4_31) ([arXiv.org version](https://arxiv.org/pdf/1811.05544.pdf))
- [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/pdf/1409.0473.pdf) ([slides](https://www.iclr.cc/archive/www/lib/exe/fetch.php%3Fmedia=iclr2015:bahdanau-iclr2015.pdf))
- [Effective Approaches to Attention-based Neural Machine Translation](https://www.aclweb.org/anthology/D15-1166/) ([slides](https://nlp.stanford.edu/~lmthang/data/papers/emnlp15_attn.pptx))
- [“Attention, Attention” in Lil’Log](https://lilianweng.github.io/lil-log/2018/06/24/attention-attention.html)
- [Big Bird: Transformers for Longer Sequences](https://arxiv.org/abs/2007.14062)
- [Paper Review: Llama 2: Open Foundation and Fine-Tuned Chat Models](https://blog.gopenai.com/paper-review-llama-2-open-foundation-and-fine-tuned-chat-models-23e539522acb)
- [“Attention” in Eugene Yan](https://aman.ai/primers/ai/attention/\(https://eugeneyan.com/writing/attention/\))

## Citation

If you found our work useful, please cite it as:

![](https://aman.ai/images/copy.png)

`@article{Chadha2021DistilledAttention,   title   = {Attention},   author  = {Jain, Vinija and Chadha, Aman},   journal = {Aman's AI Journal},   year    = {2021},   note    = {\url{https://aman.ai}} }`

-  [](https://github.com/amanchadha)|  [](https://citations.amanchadha.com/)|  [](https://twitter.com/i_amanchadha)|  [](mailto:hi@aman.ai)| 

[www.amanchadha.com](https://www.amanchadha.com/)