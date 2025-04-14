[Distilled AI](https://aman.ai/primers/ai/)[Back to aman.ai](https://aman.ai/)

# Token Sampling Methods

- [Overview](https://aman.ai/primers/ai/token-sampling/#overview)
- [Background](https://aman.ai/primers/ai/token-sampling/#background)
    - [Autoregressive Decoding](https://aman.ai/primers/ai/token-sampling/#autoregressive-decoding)
    - [Token Probabilities](https://aman.ai/primers/ai/token-sampling/#token-probabilities)
    - [Logits and Softmax](https://aman.ai/primers/ai/token-sampling/#logits-and-softmax)
- [Related: Temperature](https://aman.ai/primers/ai/token-sampling/#related-temperature)
    - [The Role of Temperature in Softmax](https://aman.ai/primers/ai/token-sampling/#the-role-of-temperature-in-softmax)
    - [Temperature Ranges and Their Effects](https://aman.ai/primers/ai/token-sampling/#temperature-ranges-and-their-effects)
        - [Low Temperature (T ≈ 0.0 - 0.5)](https://aman.ai/primers/ai/token-sampling/#low-temperature-t--00---05)
        - [Moderate Temperature (T ≈ 0.6 - 1.0)](https://aman.ai/primers/ai/token-sampling/#moderate-temperature-t--06---10)
        - [High Temperature (T > 1.0)](https://aman.ai/primers/ai/token-sampling/#high-temperature-t--10)
    - [Insights from the Softmax Function](https://aman.ai/primers/ai/token-sampling/#insights-from-the-softmax-function)
    - [Summary of Temperature’s Impact](https://aman.ai/primers/ai/token-sampling/#summary-of-temperatures-impact)
- [Greedy Decoding](https://aman.ai/primers/ai/token-sampling/#greedy-decoding)
- [Exhaustive Search Decoding](https://aman.ai/primers/ai/token-sampling/#exhaustive-search-decoding)
- [Beam Search](https://aman.ai/primers/ai/token-sampling/#beam-search)
- [Constrained Beam Search](https://aman.ai/primers/ai/token-sampling/#constrained-beam-search)
    - [Banking](https://aman.ai/primers/ai/token-sampling/#banking)
- [Top-kk](https://aman.ai/primers/ai/token-sampling/#top-k)
- [Top-pp (nucleus Sampling)](https://aman.ai/primers/ai/token-sampling/#top-p-nucleus-sampling)
    - [How is Nucleus Sampling Useful?](https://aman.ai/primers/ai/token-sampling/#how-is-nucleus-sampling-useful)
    - [Nucleus Sampling V/s Temperature](https://aman.ai/primers/ai/token-sampling/#nucleus-sampling-vs-temperature)
- [Greedy vs. Top-kk and Top-pp](https://aman.ai/primers/ai/token-sampling/#greedy-vs-top-k-and-top-p)
- [Min-pp](https://aman.ai/primers/ai/token-sampling/#min-p)
- [References](https://aman.ai/primers/ai/token-sampling/#references)

## Overview

- Generative Large Language Models (LLMs) understand input and output text as strings of “tokens,” which can be words but also punctuation marks and parts of words.
- LLM have some token selection parameters which control the randomness of output during inference or runtime. The method of selecting output tokens (specifically called token sampling methods or decoding strategies) is a key concept in text generation with language models.
- At its core, the technical underpinnings of token sampling involve constantly generating a mathematical function called a probability distribution to decide the next token (e.g. word) to output, taking into account all previously outputted tokens. Put simply, for generating text, LLMs perform sampling which involves randomly picking the next word according to its conditional probability distribution.
- In the case of OpenAI-hosted systems like ChatGPT, after the distribution is generated, OpenAI’s server does the job of sampling tokens according to the distribution. There’s some randomness in this selection; that’s why the same text prompt can yield a different response.
- In this primer, we will talk about different token sampling methods and related concepts such as Temperature, Greedy Decoding, Exhaustive Search Decoding, Beam Search, top-kk, top-pp, and min-pp.

## Background

### Autoregressive Decoding

- When we generate a textual sequence with a language model, we typically begin with a textual prefix/prompt. Then, we follow the steps shown below:
    
    1. Use the language model to generate the next token.
    2. Add this output token to our input sequence.
    3. Repeat.
- By continually generating the next token in this manner (i.e., this is what we call the autoregressive decoding process), we can generate an entire textual sequence (cf. below image; [source](https://www.linkedin.com/in/cameron-r-wolfe-ph-d-04744a238/)).
    

![](https://aman.ai/primers/ai/assets/token-sampling/ar.jpg)

### Token Probabilities

- But, how do we choose/predict the next token (i.e., step one shown above)?
- Instead of directly outputting the next token, language models output a probability distribution over the set of all possible tokens. Put simply, LLMs are essentially neural networks tackling a classification problem over the vocabulary (unique tokens).
- Given this probability distribution, we can follow several different strategies for selecting the next token. For example, greedy decoding [(one potential strategy)](https://aman.ai/primers/ai/token-sampling/#greedy-decoding), as we’ll see below, simply selects the token with the highest probability as the next token.

### Logits and Softmax

- LLMs produce class probabilities with logit vector zz where z=(z1,…,zn)z=(z1,…,zn) by performing the softmax function to produce probability vector q=(q1,…,qn)q=(q1,…,qn) by comparing zizi with with the other logits.
    
    qi=exp(zi)∑jexp(zj)qi=exp⁡(zi)∑jexp⁡(zj)
    
- The softmax function normalizes the candidates at each iteration of the network based on their exponential values by ensuring the network outputs are all between zero and one at every timestep, thereby easing their interpretation as probability values (cf. below image; [source](https://www.linkedin.com/in/cameron-r-wolfe-ph-d-04744a238/)).
    

![](https://aman.ai/primers/ai/assets/token-sampling/Softmax.jpg)

## Related: Temperature

- Although temperature is not inherently a token sampling method, it significantly influences the token sampling process and is therefore included in this primer.
- The temperature parameter allows for adjustments to the probability distribution over tokens. It serves as a hyperparameter in the softmax transformation (cf. the image below; [source](https://www.linkedin.com/in/cameron-r-wolfe-ph-d-04744a238/)), which governs the randomness of predictions by scaling the logits prior to the application of softmax.
    
    ![](https://aman.ai/primers/ai/assets/token-sampling/T.jpg)
    
    - For instance, in TensorFlow’s Magenta [implementation](https://github.com/tensorflow/magenta/blob/5cbbfb94ff4f506f1dd1f711e4704a1b3279a385/magenta/models/melody_rnn/melody_rnn_generate.py#L82) of LSTMs, the temperature parameter determines the extent to which the logits are scaled or divided before the computation of softmax.
- Lower temperature values yield more deterministic outputs, while higher values lead to more creative but less predictable outputs, facilitating a trade-off between determinism and precision on one hand, and creativity and diversity on the other.

### The Role of Temperature in Softmax

- The “vanilla” softmax function in the [Logits and Softmax](https://aman.ai/primers/ai/token-sampling/#logits-and-softmax) section incorporates the temperature hyperparameter TT as follows:
    
    qi=expziT∑jexpzjTqi=exp⁡ziT∑jexp⁡zjT
    
    - where TT is the temperature parameter (set to 1 by default).
- When the temperature is 1, we compute the softmax directly on the logits (the unscaled output of earlier layers). Using a temperature of 0.6, the model computes the softmax on logits0.6logits0.6, resulting in a larger value. Performing softmax on larger values makes the model more **confident** (less input is needed to activate the output layer) but also more **conservative** in its samples (it is less likely to sample from unlikely candidates).
    

### Temperature Ranges and Their Effects

#### Low Temperature (T ≈ 0.0 - 0.5)

- **Characteristics**:
    - Strong preference for high-probability tokens.
    - Results in deterministic outputs and lower randomness.
    - Outputs are often repetitive and lack diversity.
- **Applications**:
    - Ideal for scenarios where precision and confidence are critical (e.g., when generating factual text or answers).
- **Limitations**:
    - May lead to overly conservative predictions and repetitive loops.

#### Moderate Temperature (T ≈ 0.6 - 1.0)

- **Characteristics**:
    - Balances diversity and coherence in token sampling.
    - Allows the model to explore less likely candidates without compromising too much on coherence.
- **Applications**:
    - Commonly used for generating creative but realistic outputs.
    - Suitable for generating human-like text, code suggestions, or musical compositions.
- **Limitations**:
    - May still favor higher-probability tokens, limiting exploration of very unlikely candidates.

#### High Temperature (T > 1.0)

- **Characteristics**:
    - Produces a softer probability distribution over classes.
    - Results in more diverse and random outputs by increasing sensitivity to low-probability candidates.
    - Allows the RNN to escape repetitive loops and explore a wider range of token possibilities.
- **Applications**:
    - Useful for brainstorming ideas or generating highly creative content.
    - Enables exploration of unconventional outputs in art, music, or storytelling tasks.
- **Limitations**:
    - Leads to more mistakes and incoherent outputs.
    - Increases the likelihood of selecting improbable candidates, which might not always align with the desired result.

### Insights from the Softmax Function

- From the [Wikipedia article on softmax function](https://en.wikipedia.org/wiki/Softmax_function):

> For high temperatures (τ→∞(τ→∞), all samples have nearly the same probability, and the lower the temperature, the more expected rewards affect the probability. For a low temperature (τ→0+)(τ→0+), the probability of the sample with the highest expected reward tends to 1.

### Summary of Temperature’s Impact

- A lower temperature makes the model more confident and conservative, ideal for deterministic outputs.
- A moderate temperature strikes a balance between randomness and coherence.
- A higher temperature increases randomness and diversity but may lead to mistakes.
- Adjusting the temperature allows fine-tuning the model’s behavior depending on the task, enhancing the versatility of language models in diverse applications.

## Greedy Decoding

- Greedy decoding uses `argmax` to select the output with the highest probability at each step during the decoding process.
- The problem with this method is, it has no way to revert back in time and rectify previously generated tokens to fix its output. For example, if the machine translation prompt is “il a m’entarté” (he hit me with a pie) and greedy decoding translation generates “he hit a”, it has no way of going back to replace “a” with “me”. Greedy decoding chooses the most probable output at each time step, without considering the future impact of that choice on subsequent decisions.
- During the decoding process, the model generates a sequence of words or tokens one at a time, based on the previously generated words and the input sequence. In greedy decoding, usually we decode until the model produces a `<END>` token, For example: `<START>` he hit me with a pie `<END>` [source](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1194/slides/cs224n-2019-lecture08-nmt.pdf)
- While greedy decoding is computationally efficient and easy to implement, it may not always produce the best possible output sequence.
- A way to mitigate the issues we see from greedy decoding is to use [exhaustive search decoding](https://aman.ai/primers/ai/token-sampling/#exhaustive-search-decoding) or [beam search](https://aman.ai/primers/ai/token-sampling/#beam-search).
- The image below [(source)](https://docs.cohere.ai/docs/controlling-generation-with-top-k-top-p) shows greedy decoding in action, picking the top token at each interval.

![](https://aman.ai/primers/ai/assets/token-sampling/2.png)

## Exhaustive Search Decoding

- Exhaustive search, as the name suggests, considers every possible combination or permutation of output sequences and selecting the one that yields the highest score according to a given objective function.
- In the context of sequence-to-sequence models such as neural machine translation, exhaustive search decoding involves generating every possible output sequence and then evaluating each one using a scoring function that measures how well the output sequence matches the desired output. This can be a computationally intensive process, as the number of possible output sequences grows exponentially with the length of the input sequence.
- Exhaustive search decoding can produce highly accurate translations or summaries, but it is generally not feasible for most real-world applications due to its computational complexity.
- This would result in a time complexity of O(V(T))O(V(T)) where VV is the vocab size and TT is the length of the translation and as we can expect, this would be too expensive.

## Beam Search

- Beam search is a search algorithm, frequently used in machine translation tasks, to generate the most likely sequence of words given a particular input or context. It is an efficient algorithm that explores multiple possibilities and retains the most likely ones, based on a pre-defined parameter called the beam size.
- Beam search is widely used in sequence-to-sequence models, including recurrent neural networks and transformers, to improve the quality of the output by exploring different possibilities while being computationally efficient.
- The core idea of beam search is that on each step of the decoder, we want to keep track of the kk most probable partial candidates/hypotheses (such as generated translations in case of a machine translation task) where kk is the beam size (usually 5 - 10 in practice). Put simply, we book-keep the top kk predictions, generate the word after next for each of these predictions, and select whichever combination had less error.
- The image below [(source)](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1194/slides/cs224n-2019-lecture08-nmt.pdf) shows how the algorithm works with a beam size of 2.

![](https://aman.ai/primers/ai/assets/token-sampling/1.png)

- We can see how at each step, it calculates two most probable options along with their score, and creates the top scoring hypothesis (best guess of the likely sequence). It will then backtrack to obtain the full hypothesis.
- In beam search decoding, different hypotheses may produce `<END>` tokens on different timesteps
- When a hypothesis produces `<END>`, that hypothesis is complete so we place it aside and continue exploring other hypotheses via beam search.
- Usually we continue beam search until:
    - We reach timestep TT (where TT is some pre-defined cutoff), or
    - We have at least nn completed hypotheses (where n is pre-defined cutoff).
- So now that we have a list of completed hypotheses, how do we select the one with the highest score that fits our task the best?
    - It’s to be noted that the longer hypotheses have lower scores, so simply selecting the largest score may not work. Thus, we need to normalize the hypotheses by length and then use this to select the top one.
- Note, hypothesis here is the kk most probable partial translation (if the task is machine translation) and has a score which is the log probability. Since the log of a number ∈[0,1]∈[0,1] falls under [−∞,0][−∞,0], all the scores are non-positive and a higher score the hypothesis has, the better it is.
    - Additionally, beam search is not guaranteed to find the optimal solution, but it is more efficient than conducting an exhaustive search.
- For more details, please refer [D2L.ai: Beam Search](https://d2l.ai/chapter_recurrent-modern/beam-search.html).

## Constrained Beam Search

- Constrained beam search allows more control over the output that is generated, which is especially useful, for example, if your task is Neural Machine Translation and you have certain words that will need to be in the output.
- In constrained beam search, additional constraints are imposed on the generated sequences to ensure that they adhere to certain criteria or rules.
- The basic idea of constrained beam search is to modify the traditional beam search algorithm to incorporate constraints while generating sequences. This can be done by maintaining a set of active beams that satisfy the constraints during the search process. At each step, the algorithm generates and scores multiple candidate sequences, and then prunes the candidates that violate the constraints. The remaining candidates are then used to generate the next set of candidates, and the process continues until a complete sequence that satisfies the constraints is generated, or until a predefined stopping criterion is met.
- Constrained beam search requires careful management of the constraints to ensure that they are satisfied while still maintaining a diverse set of candidate sequences. One common approach is to use penalty functions or heuristics to discourage or penalize candidates that violate the constraints, while still allowing them to be considered during the search process. Another approach is to use a separate constraint satisfaction module that guides the search process by providing additional information or feedback on the constraints.
- For example, in text generation, constraints could include limitations on the length of the generated text, adherence to a particular format or structure, or inclusion of certain keywords or phrases. Constrained beam search modifies the scoring function or introduces additional checks during the search process to ensure that only valid sequences that meet the constraints are considered and expanded.
- Constrained beam search is commonly used in tasks such as text summarization, machine translation, and dialogue generation, where it is important to generate sequences that adhere to certain rules, guidelines, or restrictions while maintaining fluency and coherence in the generated output. It is a useful technique for controlling the output of a sequence generation model and ensuring that the generated sequences meet specific criteria or constraints.
- In the traditional beam search setting, we find the top kk most probable next tokens at each branch and append them for consideration. In the constrained setting, we do the same but also append the tokens that will take us closer to fulfilling our constraints.
- The image below [(source)](https://huggingface.co/blog/constrained-beam-search) shows step 1 of constrained beam search working in action. “On top of the usual high-probability next tokens like “dog” and “nice”, we force the token “is” in order to get us closer to fulfilling our constraint of “is fast”.” [(source)](https://huggingface.co/blog/constrained-beam-search)

![](https://aman.ai/primers/ai/assets/token-sampling/5.png)

### Banking

- Now a practical next question would be, wouldn’t forcing a token create nonsensical outputs? Using banks solves this problem by creating a balance between fulfilling the constraints and creating sensible output, and we can see this illustrated in the figure below [(source)](https://huggingface.co/blog/constrained-beam-search):

![](https://aman.ai/primers/ai/assets/token-sampling/6.png)

- “After sorting all the possible beams into their respective banks, we do a round-robin selection. With the above example, we’d select the most probable output from Bank 2, then most probable from Bank 1, one from Bank 0, the second most probable from Bank 2, the second most probable from Bank 1, and so forth. Assuming we’re using three beams, we just do the above process three times to end up with `["The is fast", "The dog is", "The dog and"]`.” [(source)](https://huggingface.co/blog/constrained-beam-search)
- Thus, even though we are forcing tokens on the model, we are still keeping track of other high probable sequences that are likely not nonsensical.
- The image below [(source)](https://huggingface.co/blog/constrained-beam-search) shows the result and all the steps combined.

![](https://aman.ai/primers/ai/assets/token-sampling/7.png)

## Top-kk

- Top kk uses a strategy where it allows to sample from a shortlist of top kk tokens. This allows all top k players to be given a chance of being chosen as the next token.
- In top-kk sampling, once the top kk most probable tokens are selected at each time step, the choice among them could be based on uniform sampling (each of the top-kk tokens has an equal chance of being picked) or proportional to their calculated probabilities.

> The choice between uniform and proportional selection in top-kk sampling depends on the desired balance between diversity and coherence in the generated text. Uniform sampling promotes diversity by giving equal chance to all top kk tokens, while proportional sampling favors coherence and contextual relevance by weighting tokens according to their probabilities. The specific application and user preference ultimately dictate the most suitable approach.

- It is suitable for tasks that require a balance between diversity and control over the output, such as text generation and conversational AI.
- Note, if kk is set to 1, it is essentially greedy decoding which we saw in one of the [earlier](https://aman.ai/primers/ai/token-sampling/#greedy-decoding) sections.
- The image below [(source)](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1194/slides/cs224n-2019-lecture08-nmt.pdf) shows top-kk in action for k=3k=3.

![](https://aman.ai/primers/ai/assets/token-sampling/3.png)

- Additionally, it’s important to note that the smaller the kk you choose, the narrower the selection will become (thus, reduced diversity, more control) and conversely, the higher the kk you choose, the wider the selection will become (thus, increased diversity, less control).

## Top-pp (nucleus Sampling)

- The difficulty of selecting the best value of kk in case of top-kk sampling opens the door for a popular decoding strategy that dynamically sets the size of the shortlist of tokens. This method, called top-pp or nucleus sampling, shortlists the top tokens whose sum of likelihoods, i.e., cumulative probability, does not exceed a certain threshold pp, and then choose one of them randomly based on their probabilities (cf. below image; [source](https://www.linkedin.com/in/cameron-r-wolfe-ph-d-04744a238/)).

![](https://aman.ai/primers/ai/assets/token-sampling/nucleus.jpg).

- Put simply, nucleus sampling, which has a hyperparameter top-pp, chooses from the smallest possible set of tokens whose summed probability (i.e., probability mass) exceeds top-pp during decoding. Given this set of tokens, we re-normalize the probability distribution based on each token’s respective probability and then sample. Re-normalization involves adjusting these probabilities so they sum up to 1. This adjustment is necessary because the original probabilities were part of a larger set. After re-normalization, a new probability distribution is formed exclusively from this smaller, selected set of tokens, ensuring a fair sampling process within this subset. This is different from top-kk, which just samples from the $k$$ tokens with the highest probability.

> Similar to top-kk sampling, the choice between uniform and proportional selection in top-kk sampling depends on the desired balance between diversity and coherence in the generated text. Uniform sampling promotes diversity by giving equal chance to all top kk tokens, while proportional sampling favors coherence and contextual relevance by weighting tokens according to their probabilities. The specific application and user preference ultimately dictate the most suitable approach.

- The image below [(source)](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1194/slides/cs224n-2019-lecture08-nmt.pdf) illustrates how the algorithm works if pp is 15% (i.e., top-pp value of 0.15).

![](https://aman.ai/primers/ai/assets/token-sampling/4.png)

- In terms of practical usage, nucleus sampling can be controlled by setting the top-pp parameter in most language model APIs (cf. GPT-3 API in [Nucleus sampling v/s Temperature](https://aman.ai/primers/ai/token-sampling/#nucleus-sampling-vs-temperature) below).

### How is Nucleus Sampling Useful?

- Top-pp is more suitable for tasks that require more fine-grained control over the diversity and fluency of the output, such as language modeling and text summarization. However, in reality, pp is actually set a lot higher (about 75%) to limit the long tail of low probability tokens that may have been sampled.
- Furthermore, consider a case where a single token has a very high probability (i.e., larger than top-pp). In this case, nucleus sampling will always sample this token. Alternatively, assigning more uniform probability across tokens may cause a larger number of tokens to be considered during decoding. Put simply, nucleus sampling can dynamically adjust the number of tokens that are considered during decoding based on their probabilities.
- Additionally top-kk and top-pp can work simultaneously, but pp will always come after kk.

### Nucleus Sampling V/s Temperature

- Based on OpenAI’s GPT-3 [API](https://platform.openai.com/docs/introduction) below, we should only use either nucleus sampling or temperature and that these parameters cannot be used in tandem. Put simply, these are different and disjoint methods of controlling the randomness of a language model’s output.

![](https://aman.ai/primers/ai/assets/token-sampling/API.jpg)

## Greedy vs. Top-kk and Top-pp

- **Deterministic vs. Stochastic**: Greedy decoding is deterministic (always picks the highest probability token), while top-kk and top-pp are stochastic, introducing randomness into the selection process.
- **Uniformity**: In top-kk sampling, once the top kk tokens are selected, the choice among them can be uniform (each of the top kk tokens has an equal chance of being picked) or proportional to their calculated probabilities.
- **Bias**: The method used can introduce a bias in the type of text generated. Greedy decoding tends to be safe and less creative, while stochastic methods can generate more novel and varied text (especially, using uniform sampling as explained earlier) but with a higher chance of producing irrelevant or nonsensical output.

## Min-pp

- min-pp sampling is a novel token selection or decoding strategy termed has been [introduced](https://github.com/huggingface/transformers/issues/27670) in the Huggingface Transformers library, aiming to refine the generation capabilities of LLMs. This strategy has emerged as a potential solution to the limitations inherent in the established methods of top-kk and top-pp sampling.
- **Background on existing sampling strategies:**
    1. **top-kk Sampling:** This technique orders tokens by descending probability, selects the highest kk tokens, and then samples from this subset. Although effective, it often discards high-quality continuations, reducing word diversity.
    2. **top-pp Sampling:** Unlike top-kk, which limits the selection to a fixed number of tokens, top-pp sampling considers the smallest set of tokens whose cumulative probability exceeds a threshold pp. This method, however, may include low-probability tokens that can detract from the coherence of the generated text.
- **Limitations of existing strategies:**
    - **top-kk:** The arbitrary exclusion of tokens beyond the most probable kk can omit high-quality relevant continuations, leading to lack of word diversity.
    - **top-pp:** It risks including tokens with minimal probabilities, potentially leading to derailed generation and incoherent outputs.
- **Introduction of min-pp sampling:**
    - min-pp sampling addresses these drawbacks by employing a dynamic filtering mechanism. Unlike the static nature of top-kk and top-pp, min-pp sampling adjusts the filter based on the distribution’s probability landscape:
        - It sets a base probability threshold, calculated as the product of a minimum probability factor (specified by `min_p`) and the probability of the most likely upcoming token.
        - Tokens with probabilities below this computed threshold are excluded, ensuring a balance between diversity and coherence.
- **Benefits of min-pp sampling:**
    - **With a dominant high-probability token:** Implements an aggressive filtering approach to maintain focus and coherence.
    - **Without a dominant high-probability token:** Applies a more relaxed filter, allowing for a broader range of plausible continuations, beneficial in scenarios where many continuation possibilities are plausible (i.e., situations requiring creativity).
- **Recommended settings for min-pp sampling:**
    - For optimal performance, particularly in creative text generation, it is suggested to set `min_p` between 0.05 and 0.1, combined with a temperature parameter greater than 1. This configuration harnesses the model’s capability to generate diverse and imaginative text while maintaining logical coherence.
- In conclusion, min-pp sampling offers a significant enhancement over traditional methods by dynamically adapting to the probability distribution of tokens, thus potentially setting a new standard for decoding strategies in language model generations. min-pp sampling yields the benefit of enabling the use of high temperature settings, which can make models a lot more creative in practice. min-pp also allows for the reduction or elimination of hacks like “repetition penalties”.
- [Release notes](https://github.com/huggingface/transformers/issues/27670); [Discussion thread](https://huggingface.co/posts/joaogante/319451541682734); [Sample usage](https://pastebin.com/VqXNtuxd)

![](https://aman.ai/primers/ai/assets/token-sampling/min-p.jpeg)

## References

- [Hinton, Geoffrey, Oriol Vinyals, and Jeff Dean. “Distilling the knowledge in a neural network.” arXiv preprint arXiv:1503.02531 (2015)](https://arxiv.org/abs/1503.02531)
- [What is Temperature in LSTM (and neural networks generally)?](https://cs.stackexchange.com/questions/79241/what-is-temperature-in-lstm-and-neural-networks-generally)
- [Stanford CS224n](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1194/slides/cs224n-2019-lecture08-nmt.pdf)
- [Ketan Doshi’s Foundations of NLP Explained Visually: Beam Search, How it Works](https://towardsdatascience.com/foundations-of-nlp-explained-visually-beam-search-how-it-works-1586b9849a24)
- [Cohere Top k and Top p](https://docs.cohere.ai/docs/controlling-generation-with-top-k-top-p)
- [HuggingFace: Constrained Beam Search](https://huggingface.co/blog/constrained-beam-search)

-  [](https://github.com/amanchadha)|  [](https://citations.amanchadha.com/)|  [](https://twitter.com/i_amanchadha)|  [](mailto:hi@aman.ai)| 

[www.amanchadha.com](https://www.amanchadha.com/)