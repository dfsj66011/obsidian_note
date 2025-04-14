[Distilled AI](https://aman.ai/primers/ai/)[Back to aman.ai](https://aman.ai/)

# Models • Alpaca

- [Overview](https://aman.ai/primers/ai/alpaca/#overview)
- [References](https://aman.ai/primers/ai/alpaca/#references)

## Overview

- Stanford’s [Alpaca](https://crfm.stanford.edu/2023/03/13/alpaca.html) is an instruction-finetuned 7B language transformer based on the 7B LLaMA GPT-3 alternative by Meta released in mid-March 2023.
    
- Instead of using reinforcement learning with human feedback (RLHF), they take a supervised approach using 52k instruction-output pairs.
    
- Instead of using human-generated instruction-output pairs, they retrieve the data by querying the GPT-3-based text-davinci-003 model. So, Alpaca essentially uses a form of weakly supervised or knowledge-distillation-flavored finetuning.*
    
- The training recipe is available on GitHub, and according to the authors, it can be replicated with 8 A100 GPUs and a ~$600 budget.
    
- Note that this can be competitive with human annotations. For example, in the [Self-Instruct paper](https://arxiv.org/abs/2212.10560), the authors found that bootstrapping a model on its own generations can result in performance competitive with InstructGPT.
    

![](https://aman.ai/primers/ai/assets/alpaca/alpaca.jpeg)

## References

- [Sebastian Raschka’s post on Alpaca](https://www.linkedin.com/posts/sebastianraschka_machinelearning-ai-largelanguagemodels-activity-7043221441680470016-LiLR?utm_source=share&utm_medium=member_desktop)

-  [](https://github.com/amanchadha)|  [](https://citations.amanchadha.com/)|  [](https://twitter.com/i_amanchadha)|  [](mailto:hi@aman.ai)| 

[www.amanchadha.com](https://www.amanchadha.com/)