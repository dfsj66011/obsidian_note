![Deep (Learning) Focus](https://substackcdn.com/image/fetch/w_80,h_80,c_fill,f_auto,q_auto:good,fl_progressive:steep,g_auto/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2Fab9b43fb-52d5-40da-995d-5b7cd3f91064_896x896.png)

# [Deep (Learning) Focus](https://cameronrwolfe.substack.com/)

SubscribeSign in

# Understanding the Open Pre-Trained Transformers (OPT) Library

### The release of the OPT library by Meta AI is a true step towards improved transparency and general understanding in large language…

[

![](https://substackcdn.com/image/fetch/w_36,h_36,c_fill,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F69aba7df-b571-4609-aa47-fc2d031c11b8_1242x1595.jpeg)



](https://substack.com/@cwolferesearch)

[Cameron R. Wolfe, Ph.D.](https://substack.com/@cwolferesearch)

Jun 07, 2022

4

[](https://cameronrwolfe.substack.com/p/understanding-the-open-pre-trained-transformers-opt-library-193a29c14a15/comments)

Share

[

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2Fcc190064-d9d3-430b-949d-181eee18ff72_800x539.png)



](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2Fcc190064-d9d3-430b-949d-181eee18ff72_800x539.png)

Depiction of a decoder-only language modeling architecture (created by author)

Recently, [Meta AI](https://ai.facebook.com/) published “OPT: Open Pre-Trained Transformer Language Models” [1] and an associated [code repository](https://github.com/facebookresearch/metaseq?fbclid=IwAR1cBNQQwYDqE_DQf40E7oUS4lK_WneMI0aSLlL4JR-3q9a6sq1OS_QcWTY) with the intent of open-sourcing high-performing large language models (LLMs) to the public. In particular, OPT provides an entire suite of LLMs, ranging in size from 125 million to 175 billion parameters, along with the code used to train these models. Impressively, the largest OPT model — OPT-175B (not present in the code repository but available [upon request](https://docs.google.com/forms/d/e/1FAIpQLSe4IP4N6JkCEMpCP-yY71dIUPHngVReuOmQKDEI1oHFUaVg7w/viewform)) — is shown to perform similarly to GPT-3 [3], which also contains 175 billion parameters, despite utilizing only 15% of the GPT-3 carbon footprint during development and training.

Despite the fact that LLMs have demonstrated impressive performance on numerous tasks (e.g., zero and few-shot learning), they have only been made available to the public via APIs. Such a paradigm is problematic from a research perspective, as is outlined in the paper.

> This restricted access has limited researchers’ ability to understand how and why these large language models work, hindering progress on efforts to improve their robustness and mitigate known issues such as bias and toxicity.

With the release of OPT, the deep learning research community now has full access to an entire suite of LLMs (including smaller models), enabling analysis that further boosts understanding of how these models work. Within this post, I will overview the major components of the OPT publication, such that the interested reader can gain an understanding the OPT library, how it was developed, and its implications for future deep learning research.

### Why does this matter?

[

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F7c2f92f2-fdd7-4e1e-804d-b9a3aea18aa6_800x365.png)



](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F7c2f92f2-fdd7-4e1e-804d-b9a3aea18aa6_800x365.png)

Components of the OPT Library (created by author)

Prior to detailing the components of the OPT library, it is useful to look at the framework as a whole to gain an understanding of its implications and benefits. The full OPT release includes: pre-trained language models of numerous sizes, a code base for training and deploying these models, and log books that detail the model development process. These components, illustrated in the chart above, together provide three major benefits to the research community.

**Full model availability.** The release of pre-trained LLMs in OPT marks the first occasion in which language models of this scale have been made fully available to the research community. Previously, such models were only accessible through [paid APIs](https://openai.com/api/), and only a few research labs had full access to the models’ source (i.e., meaning all weights and model components are visible). In particular, the API for GPT-3 created by OpenAI offers several different models sizes and [charges users](https://openai.com/api/pricing/) per the number of tokens generated. Going even further, the GPT-3 API also charges users for fine-tuning their LLM and even generating textual embeddings. Although such an API may be most appropriate for commercial applications, the OPT library enables the research community as a whole to analyze the behavior of LLMs, improve their robustness, and mitigate known issues such as bias and toxicity by granting full access to such models.

**Lasting improvements to LLM training efficiency.** To train the models in OPT, researchers utilized cutting-edge techniques like [Fully Sharded Data Parallel](https://engineering.fb.com/2021/07/15/open-source/fsdp/) (FSDP) training and tensor parallel abstractions from [Megatron-LM](https://github.com/NVIDIA/Megatron-LM?fbclid=IwAR3SvXpTaLseZacJv_Bntwg0czNNYj8hEhcho3R_mo8ABDS8zmszw4mdZ3E), resulting in improved resource utilization (i.e., 17% better than research published directly by NVIDIA [3]) and, in turn, massive reductions in compute cost. Luckily, the OPT code base makes all of these efficiency improvements openly available, meaning that future research can easily adopt these improvements and begin to reduce the massive carbon footprint of training LLMs [4].

**Detailed insight into LLM training and development.** The OPT release includes [notes](https://github.com/facebookresearch/metaseq/tree/main/projects/OPT/chronicles?fbclid=IwAR3qONxU4mENL_HAVcf9LJCwwqijGCVMk87C8Sm9_q3y6TZS3kZiY6Fd5dY) and [logbooks](https://github.com/facebookresearch/metaseq/blob/main/projects/OPT/chronicles/OPT175B_Logbook.pdf?fbclid=IwAR1gSseT67AGnNprJRdiW91Pf7eW1b82Z3pYshE4CYGT_-AKVnCUdaIdmm8) that detail the model training and development process (i.e., this follows guidelines proposed by the [Partnership on AI](https://partnershiponai.org/paper/responsible-publication-recommendations/?fbclid=IwAR2oOB5PXO0AWkZFE86z5HElfHF-oYEWIPrtMTqPwhC1GUv1Hty4kc-6rq0) and [NIST](https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.1270.pdf?fbclid=IwAR0X3fJxMTO8tb6BxafD-eVq2qfPKl4MasXs0XPO48MD5mfwTZJKCr5as8U)). These additional components provide various insights into the production of high-performing LLMs, including total development costs, necessary “mid-flight” training adjustments, and even hardware failures that interrupt model development. Such insights make clear the incredible difficulty of training language models at this scale and provide detailed instructions to any practitioner that must replicate this process.

### Understanding OPT

Now that the context surrounding the OPT library has been explained, I will detail the methodology behind OPT and how the models within this package were derived. This overview will mostly focus on the types and sizes of language models that were used and how they were trained. Throughout the explanation, special emphasis will be provided to the major takeaways and findings relevant to producing and utilizing high-performing LLMs.

**Model.** The suite of pre-trained language models provided within OPT follow a decoder-only transformer architecture — an architecture that was popularized for language modeling with the release of GPT-2 [5] and extended by GPT-3 [2]. Although the details of this model architecture are beyond the scope of this post, the decoder-only transformer architecture is simply a [transformer model](https://jalammar.github.io/illustrated-transformer/) with the entire encoder and the encoder-decoder self-attention modules (present within each layer of a transformer’s decoder) removed. Such an architecture is depicted within the figure below.

[

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F3f458aeb-d94d-4530-830c-b94f5716344b_800x447.png)



](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F3f458aeb-d94d-4530-830c-b94f5716344b_800x447.png)

Decoder-only transformer architecture (created by author)

Thus, the final model is an autoregressive architecture (i.e., meaning that the output at time `t` is used as input at time `t + 1`) that, given some prompt or input, can continue generating the next token in a sequence, as shown below.

[

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F7489e619-26f3-41ae-99e7-e25de3e936f5_800x420.png)



](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F7489e619-26f3-41ae-99e7-e25de3e936f5_800x420.png)

Sentence generation with an autoregressive, decoder-only transformer architecture (created by author)

Although I will not go into more detail regarding language modeling and related architectures, I encourage anyone that is interested to read more about [transformers](https://jalammar.github.io/illustrated-transformer/), [decoder-only language models](https://jalammar.github.io/illustrated-gpt2/), or some of the [state-of-the-art language modeling results](https://arxiv.org/abs/2005.14165) that have been recently achieved.

The models in the OPT library have various different sizes, as shown in the figure below, where `L` represents the number of layers, `H` represents the number of attention heads, and `d_model` represents the vector dimension used for attention. Differently-sized models are included within OPT so that the impact of model scale on LLM performance can be readily analyzed.

[

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2Feb417cd8-cac3-4dad-b182-322e1484c0f8_311x315.png)



](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2Feb417cd8-cac3-4dad-b182-322e1484c0f8_311x315.png)

Model sizes in OPT (from [1])

**Data.** To pre-train the language models within OPT, authors adopt a massive dataset of unlabeled text data that has been filtered to contain pre-dominantly English sentences. This dataset is constructed by combining numerous publicly-available datasets and is roughly similar to the dataset used for pre-training of RoBERTa [6] with a few added components. Each of the datasets used for pre-training are enumerated below:

- [BookCorpus](https://yknzhu.wixsite.com/mbweb): A dataset that aligns books to their movie releases to provide rich, descriptive explanation of visual content.
    
- [Stories](https://arxiv.org/abs/1806.02847): A customized text corpus that is aggregated from [CommonCrawl](https://commoncrawl.org/) data based on questions (not answers) in common sense reasoning tasks.
    
- [CCNews](https://commoncrawl.org/): A dataset of news articles aggregated from different locations around the globe.
    
- [The Pile](https://arxiv.org/abs/2101.00027): An 800Gb corpus of English textual data aggregated from academic or professional sources.
    
- [PushShift.io Reddit](https://arxiv.org/abs/2001.08435): An up-to-date collection of Reddit data that is collected from the entire history of the website and updated in real time.
    

The datasets outlined above are combined together to form a large, unlabeled corpus of pre-training data. Data is pulled from sources in many different domains (e.g., social media, news, academic content, books, etc.), forming a diverse pre-training set. Such diversity has been shown to have a massive, positive impact on language model performance [7].

[

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F5956009e-e64e-48a2-a024-a45bc1ddd71d_800x463.png)



](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F5956009e-e64e-48a2-a024-a45bc1ddd71d_800x463.png)

OPT pre-training corpus (created by author)

Prior to the beginning of training, all duplicate data is filtered from the combined text corpus, where duplicates are identified using a [hashing approach](https://www.pinecone.io/learn/locality-sensitive-hashing/). The Pile dataset was noted to contain a significant number of duplicated documents in comparison to other data sources.

**Training Setup.** The final dataset is [tokenized](https://nlp.stanford.edu/IR-book/html/htmledition/tokenization-1.html) similar to GPT-2 [5], and batches of 0.5 to 4 million sequences (i.e., batch size depends on model size but is kept constant throughout training) of 2048 tokens are fed to the model during training. Model weights are initialized similarly to Megatron-LM [8], and the learning rate is decayed linearly throughout training after a short [warmup period](https://stackoverflow.com/questions/55933867/what-does-learning-rate-warm-up-mean) to its maximum value. A few other training tricks such as dropout, gradient clipping, and adding a pre-divide factor to eliminate under/overflows when computing the gradient are also employed, and each model is trained until convergence. Despite observing over 300 billion tokens throughout the pre-training process, the largest OPT model (OPT-175B) was trained with less than 15% of the carbon emissions of GPT due to improvements to hardware utilization and efficiency discussed earlier.

Pre-training of such LLMs is conducted in a self-supervised manner. Although the details of pre-training for language models are beyond the scope of this post, I encourage the interested reader to [read more](https://jalammar.github.io/illustrated-gpt2/) about these pre-training procedures and how powerful models can be trained using unlabeled text data. Self-supervised learning has drastically transformed research in natural language processing, resulting in massive performance improvements over previous generations of models that do not leverage such self-supervised pre-training procedures [9].

**Other details.** Several other “mid-flight” changes (i.e., dynamic changes that must be made during training) to the LLM training procedure are required to achieve optimal performance. These changes are mostly related to handling loss divergences throughout training, though other changes were required to deal with issues like hardware failures (i.e., these were common due to the scale of the compute cluster needed to train LLMs). For loss divergences, authors solve the problem by lowering the learning rate and restarting the training process from an earlier point, thus allowing the model to recover and continue training. All details of training procedures and required mid-flight changes are detailed in the OPT [logbook](https://github.com/facebookresearch/metaseq/blob/main/projects/OPT/chronicles/OPT175B_Logbook.pdf?fbclid=IwAR2yKtesxDv27HxGqOh8xhKBQNBCpjvhMCX5zuHDKHazYUv7mvmnuu2I844), allowing practitioners to exactly replicate the training process.

### How do OPT models perform?

The performance of the largest model in the OPT library (i.e., OPT-175B) was evaluated in numerous settings. The goal of evaluation was to prove that OPT-175B performs similarly to GPT-3, thus providing an open-sourced version of the wildly-popular model that can be extensively analyzed by the research community. As such, most evaluation settings — excluding those that measure the models’ level of bias and toxicity — are taken from GPT-3 and re-implemented for OPT-175B, forming a direct comparison between the two models. Below, I outline the settings in which OPT-175B was evaluated and briefly describe how its performance compares to that of GPT-3.

#### Prompting

OPT-175B is evaluated over 16 standard, prompting-based NLP tasks, including [HellaSwag](https://rowanzellers.com/hellaswag/), [StoryCloze](https://cs.rochester.edu/nlp/rocstories/), [ARC Easy and Challenge](https://allenai.org/data/arc-da), [OpenBookQA](https://allenai.org/data/open-book-qa), [WinoGrad](https://cs.nyu.edu/~davise/papers/WinogradSchemas/WS.html), [WinoGrande](https://winogrande.allenai.org/), and [SuperGLUE](https://super.gluebenchmark.com/). In such prompting-based tasks, the model is provided an initial “prompt” in the form of a sentence or description and expected to form a solution based on this prompt. For example, HellaSwag provides a partial sentence to the language model and expects the sentence to be completed [10], while StoryCloze provides a partial story to the model and expects the correct ending to be predicted [11]. Across all datasets, OPT-175B is evaluated in both zero-shot (i.e., the pre-trained model is used directly for evaluation) and one/few-shot regimes (i.e., the model is fine-tuned a little bit on data from the target domain before evaluation).

**Zero-shot.** OPT-175B performs similarly to GPT-3 in terms of zero-shot performance on average, though performance varies on some tasks. In particular, OPT-175B matches GPT-3 performance for ten of the tasks, but either underperforms or performs sporadically (i.e., due to small validation set size) for the remaining six tasks. See the image below for an illustration of model performance relative to GPT-3 in all settings.

[

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2Fe111f46e-961b-4810-87ab-31cf69dada7d_800x944.png)



](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2Fe111f46e-961b-4810-87ab-31cf69dada7d_800x944.png)

Zero-shot performance of OPT-175B relative to other language models (from [1])

As can be seen, the average performance of OPT-175B follows the trend of GPT-3, despite falling short of GPT-3 performance on a few tasks. The authors do note, however, that replicating GPT-3 performance is difficult on several tasks, meaning that its superior performance may be due to differences in evaluation protocols rather than improved model quality in certain cases.

**Multi-Shot.** Again, OPT-175B is found to perform similarly to GPT-3 in the one and few-shot domains. As with the zero-shot domain, analyzing performance individually on each task reveals that OPT-175B matches GPT-3 performance on ten tasks, while the remaining tasks yield inconsistent comparisons between OPT-175B and GPT-3. See the figure below for a depiction of results on each individual dataset.

[

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F71848917-1c22-4db6-aa33-8fcf770468e7_800x945.png)



](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F71848917-1c22-4db6-aa33-8fcf770468e7_800x945.png)

Multi-shot performance of OPT-175B relative to other language models (from [1])

Although OPT-175B performs slightly worse than GPT-3 on certain tasks (e.g., MultiRC in the figure above), performance between the two models is roughly similar. Thus, models within the OPT framework are seemingly valid for analyzing the performance and behavior of high-performing LLMs in general.

#### Dialogue

OPT-175B is evaluated on several open source dialogue datasets, including: [ConvAI2](https://parl.ai/projects/convai2/), [Wizard of Wikipedia](https://parl.ai/projects/wizard_of_wikipedia/), [Empathetic Dialogues](https://github.com/facebookresearch/EmpatheticDialogues), [Blended Skill Talk](https://parl.ai/projects/bst/), and [Wizard of Internet](https://parl.ai/projects/sea/). On such tasks, models are evaluated based upon their ability to generate open-ended dialogue streams that are realistic and coherent. Such an application is important because LLMs are already widely used in modern dialogue systems and chatbots [12, 13]. Comparisons are made to several open source models (e.g., BlenderBot [14]), which may be unsupervised— such as OPT-175B — or supervised, meaning that data from the target dialogue domain is present in the model’s training corpus.

Interestingly, OPT-175B outperforms other unsupervised models on dialogue tasks and even performs similarly to models that are trained to produce dialogue in a supervised manner. However, the authors are hesitant of this result and suggest that such comparable performance relative to supervised models may be due to a leakage of dialogue datasets into OPT-175B’s pre-training corpus. Beyond such impressive performance, OPT-175B is also found capable of maintaining a consistent persona across dialogue sessions, a behavior that is observed in other popular chatbots like LaMDA [13].

#### Bias and Toxicity

Moving past simple performance evaluations, the creators of OPT evaluate the potential harm of LLMs within the package by performing a series of tests related to hate speech, stereotyping, and toxic content. The benchmarks used to measure such properties include:

- [ETHOS](https://github.com/intelligence-csd-auth-gr/Ethos-Hate-Speech-Dataset): evaluates a language model’s ability to detect hate speech on social media platforms.
    
- [CrowS-Pairs](https://github.com/nyu-mll/crows-pairs): measures the level of U.S. stereotypical bias within language models.
    
- [StereoSet](https://stereoset.mit.edu/): measures stereotypical bias with respect to gender, race, religion, and profession.
    
- [RealToxicityPrompts](https://allenai.org/data/real-toxicity-prompts): evaluates the risk of toxic degeneration in language models with sentence snippets from the web.
    
- [SaFeRDialogues](https://arxiv.org/abs/2110.07518): explores a language model’s ability to recover from explicit safety failures (i.e., via apologizing or recognizing the mistake).
    
- [Safety Bench Unit Tests](https://parl.ai/projects/safety_bench/): measures the safeness or unsafeness of a language model’s response to a given prompt.
    

OPT-175B is evaluated on these benchmarks in comparison to GPT-3. It should be noted that GPT-3 was not previously evaluated on such benchmarks, as they were not available prior to its initial publication [2].

**Hate Speech.** Authors measure the ability of OPT-175B and GPT-3 language models to identify whether a given English sentence is racist or sexist. Such a task is performed in zero, one, and few-shot manners, and OPT-175B is shown to more accurately detect hateful sentences relative to GPT-3 in all cases.

**Bias.** The CrowS-Pairs dataset is used to measure a language model’s level of bias with respect to gender, race, sexual orientation, age, nationality, disability, physical appearance, and socioeconomic status. When evaluated on this dataset, OPT-175B demonstrates higher level of bias in comparison to GPT-3 in every category except for religion.

**Stereotypes.** Stereotypical bias is measured across several categories, including profession, gender, religion, and race. Details on the specific evaluation process are provided within Section 4.3 of the publication itself [1]. In aggregate, however, both GPT-3 and OPT-175B are found to demonstrate comparable levels of stereotypical bias with respect to the categories that were considered.

**Toxicity.** The authors evaluate the tendency of OPT-175B to generate toxic responses to certain prompts. Interestingly, OPT-175B is found to demonstrate higher levels of toxicity in comparison to both GPT-3 and PaLM [15]. Furthermore, all of the considered language models were found to have a higher probability of a toxic response as the toxicity of the prompt increases.

**Dialogue Safety.** OPT-175B is evaluated on its ability to recognize safety failures and avoid unsafe responses to certain prompts. In comparison to several open source dialogue models, OPT-175B is shown to demonstrate comparable performance with respect to dialogue safety. However, dialogue models that are fine-tuned on curated datasets were found to demonstrate generally lower levels of toxicity.

**What does this tell us?** Evaluations of OPT-175B on bias and toxicity-related baselines reveal some of the limitations faced by modern LLMs. Namely, the presence of unmoderated social media data (i.e., the Pushshift.io Reddit database in particular) within the OPT-175B pre-training corpus familiarizes the model with concepts relevant to bias and toxicity. In some cases, this familiarity is beneficial, such as for more accurately detecting hate speech. However, such data also causes biases to form during the pre-training process, leading the model to display higher levels of stereotypical bias and toxicity. As such, these experiments reveal that biased and toxic behavior is a consideration that must be addressed and mitigated in the creation and deployment of LLMs.

### Takeaways

[

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F93ceef07-6eea-4c20-ac59-cd4780d6cb39_800x343.png)



](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F93ceef07-6eea-4c20-ac59-cd4780d6cb39_800x343.png)

High-level summary of OPT (created by author)

The release of the OPT library makes LLMs available to the deep learning research community as a whole. Despite the popularity of LLMs like GPT-3 in production language modeling applications, the behavior of these models is still poorly understood due to the fact that they are only accessible through paid, abstracted APIs. Thus, the OPT library, which open sources high-performing LLMs at similar scale to GPT-3, takes the first step towards truly understanding such models by making them fully available (i.e., under a non-commercial research license) to the deep learning community, thus enabling countless avenues of evaluation and analysis.

The largest model within the OPT library — OPT-175B — is evaluated extensively to show that its performance is similar to GPT-3, revealing that analysis of this model is representative of the most widely-used LLMs. Furthermore, authors extensively evaluate the existence of biased and toxic tendencies within the model — the first analysis of this kind for such LLMs — finding that the existence of unmoderated data within the pre-training corpus can indeed lead to damaging tendencies in model behavior. Such a finding makes clear the need to consider the ethical and societal implications of using LLMs, as is made clear within the publication itself.

> We believe the entire AI community — academic researchers, civil society, policymakers, and industry — must work together to develop clear guidelines around responsible AI in general and responsible large language models in particular, given their centrality in many downstream language applications.

Along with the release of a suite of LLMs, the OPT library comes with pre-training code and detailed log books that implement and document the training process. As such, invaluable insights into LLM training are provided, allowing the process to be efficiently replicated by others. This transparency into the LLM training process also reveals the significant cost and difficulty of training such models by highlighting the various hardware failures and mid-flight changes that are required to obtain a high-performing model. For these reasons, the release of the OPT library is a true step towards improved transparency and general understanding in large language modeling and provides significant benefit to the research community as a whole.

_Conclusion_

Thanks so much for reading this post! I hope you found it to be helpful and insightful. If you have any feedback on the post, feel free to leave a comment or connect with me on [LinkedIn](https://www.linkedin.com/in/cameron-wolfe-04744a238/) or [Twitter](https://twitter.com/cwolferesearch). This post can also be accessed on my [personal blog](https://cameronrwolfe.me/blog). To keep up with my future blog posts and other works you can [sign up](https://cameronrwolfe.me/signup) to receive e-mail notifications here or visit my [personal webpage](https://cameronrwolfe.me/). This post was completed as part of my research and learning as a Research Scientist at [Alegion](https://www.alegion.com/), a data annotation platform with industry-leading video and computer vision annotation capabilities.

_Bibliography_

[1] Zhang, Susan, et al. “OPT: Open Pre-trained Transformer Language Models.” _arXiv preprint arXiv:2205.01068_ (2022).

[2] Brown, Tom, et al. “Language models are few-shot learners.” _Advances in neural information processing systems_ 33 (2020): 1877–1901.

[3] Smith, Shaden, et al. “Using deepspeed and megatron to train megatron-turing nlg 530b, a large-scale generative language model.” _arXiv preprint arXiv:2201.11990_ (2022).

[4] Sharir, Or, Barak Peleg, and Yoav Shoham. “The cost of training nlp models: A concise overview.” _arXiv preprint arXiv:2004.08900_ (2020).

[5] Radford, Alec, et al. “Language models are unsupervised multitask learners.” _OpenAI blog_ 1.8 (2019): 9.

[6] Liu, Yinhan, et al. “Roberta: A robustly optimized bert pretraining approach.” _arXiv preprint arXiv:1907.11692_ (2019).

[7] Gao, Leo, et al. “The pile: An 800gb dataset of diverse text for language modeling.” _arXiv preprint arXiv:2101.00027_ (2020).

[8] Shoeybi, Mohammad, et al. “Megatron-lm: Training multi-billion parameter language models using model parallelism.” _arXiv preprint arXiv:1909.08053_ (2019).

[9] Devlin, Jacob, et al. “Bert: Pre-training of deep bidirectional transformers for language understanding.” _arXiv preprint arXiv:1810.04805_ (2018).

[10] Zellers, Rowan, et al. “HellaSwag: Can a machine really finish your sentence?.” _arXiv preprint arXiv:1905.07830_ (2019).

[11] Cui, Yiming, et al. “Discriminative sentence modeling for story ending prediction.” _Proceedings of the AAAI Conference on Artificial Intelligence_. Vol. 34. No. 05. 2020.

[12] Adiwardana, Daniel, et al. “Towards a human-like open-domain chatbot.” _arXiv preprint arXiv:2001.09977_ (2020).

[13] Thoppilan, Romal, et al. “LaMDA: Language Models for Dialog Applications.” _arXiv preprint arXiv:2201.08239_ (2022).

[14] Roller, Stephen, et al. “Recipes for building an open-domain chatbot.” _arXiv preprint arXiv:2004.13637_ (2020).

[15] Chowdhery, Aakanksha, et al. “Palm: Scaling language modeling with pathways.” _arXiv preprint arXiv:2204.02311_ (2022).

---

#### Subscribe to Deep (Learning) Focus

By Cameron R. Wolfe · Launched 3 years ago

I contextualize and explain important topics in AI research.

Subscribe

By subscribing, I agree to Substack's [Terms of Use](https://substack.com/tos), and acknowledge its [Information Collection Notice](https://substack.com/ccpa#personal-data-collected) and [Privacy Policy](https://substack.com/privacy).

[

![](https://substackcdn.com/image/fetch/w_32,h_32,c_fill,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack.com%2Fimg%2Favatars%2Fgreen.png)



](https://substack.com/profile/108515768-neo-adam)

[

![](https://substackcdn.com/image/fetch/w_32,h_32,c_fill,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2Fc628c9aa-3490-4fee-a7cb-d027356b826d_400x400.jpeg)



](https://substack.com/profile/18767028-daniel-duma)

[

![](https://substackcdn.com/image/fetch/w_32,h_32,c_fill,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F27fddcfd-ebf9-48af-82d9-1331d5b8a902_4167x4167.png)



](https://substack.com/profile/45646766-obrian-henry)

4 Likes

[](https://substack.com/note/p-73746319/restacks?utm_source=substack&utm_content=facepile-restacks)

4

[](https://cameronrwolfe.substack.com/p/understanding-the-open-pre-trained-transformers-opt-library-193a29c14a15/comments)

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