[Distilled AI](https://aman.ai/primers/ai/)[Back to aman.ai](https://aman.ai/)

# Primers • Vision Language Models

- [Multimodal Learning](https://aman.ai/primers/ai/vision-language-models/#multimodal-learning)
- [Vision-language Tasks](https://aman.ai/primers/ai/vision-language-models/#vision-language-tasks)
    - [Generation Tasks](https://aman.ai/primers/ai/vision-language-models/#generation-tasks)
    - [Classification Tasks](https://aman.ai/primers/ai/vision-language-models/#classification-tasks)
    - [Retrieval Tasks](https://aman.ai/primers/ai/vision-language-models/#retrieval-tasks)
- [BERT-like Architectures](https://aman.ai/primers/ai/vision-language-models/#bert-like-architectures)
    - [Two-stream Models: ViLBERT](https://aman.ai/primers/ai/vision-language-models/#two-stream-models-vilbert)
    - [Single-stream Models](https://aman.ai/primers/ai/vision-language-models/#single-stream-models)
- [Pretraining and Fine-tuning](https://aman.ai/primers/ai/vision-language-models/#pretraining-and-fine-tuning)
    - [Pretraining Strategies](https://aman.ai/primers/ai/vision-language-models/#pretraining-strategies)
- [VL Generative Models](https://aman.ai/primers/ai/vision-language-models/#vl-generative-models)
    - [DALL-E](https://aman.ai/primers/ai/vision-language-models/#dall-e)
    - [GLIDE](https://aman.ai/primers/ai/vision-language-models/#glide)
- [VL Models Based on Contrastive Learning](https://aman.ai/primers/ai/vision-language-models/#vl-models-based-on-contrastive-learning)
    - [CLIP](https://aman.ai/primers/ai/vision-language-models/#clip)
    - [ALIGN](https://aman.ai/primers/ai/vision-language-models/#align)
    - [FLORENCE](https://aman.ai/primers/ai/vision-language-models/#florence)
- [Enhanced Visual Representations](https://aman.ai/primers/ai/vision-language-models/#enhanced-visual-representations)
    - [VinVL](https://aman.ai/primers/ai/vision-language-models/#vinvl)
    - [SimVLM](https://aman.ai/primers/ai/vision-language-models/#simvlm)
- [Conclusion and Observations](https://aman.ai/primers/ai/vision-language-models/#conclusion-and-observations)
- [References](https://aman.ai/primers/ai/vision-language-models/#references)
- [Further Reading](https://aman.ai/primers/ai/vision-language-models/#further-reading)
- [Citation](https://aman.ai/primers/ai/vision-language-models/#citation)

## Multimodal Learning

- Multimodal learning refers to the process of learning representations from different types of modalities using the same model. Different modalities are characterized by different statistical properties. In the context of machine learning, input modalities include images, text, audio, etc. In this article, we will discuss only images and text as inputs and see how we can build Vision-Language (VL) models.
- In this article, we discuss only vision-language models because 2021 was a great year for VL models. We saw architectures such as CLIP, DALLE, GLIDE, ALIGN and SimVL.
- We tried to distill the rapid process in the field by presenting a few key architectures and core concepts that yield exceptional results. Transformers-like models and contrastive learning are currently the most promising approaches, but we believe that the research community still has a long way to go.

## Vision-language Tasks

- Vision-language models have gained a lot of popularity in recent years due to the number of potential applications. We can roughly categorize them into 3 different areas. Let’s explore them along with their subcategories.

### Generation Tasks

- Visual Question Answering (VQA) refers to the process of providing an answer to a question given a visual input (image or video).
    
- Visual Captioning (VC) generates descriptions for a given visual input.
    
- Visual Commonsense Reasoning (VCR) infers common-sense information and cognitive understanding given a visual input.
    
- Visual Generation (VG) generates visual output from a textual input, as shown in the image.
    
- The following image from OpenAI’s blog shows AI-generated images based on a user-fed input prompt:
    

![](https://aman.ai/primers/ai/assets/vision-language-models/image-generation.png)

### Classification Tasks

- Multimodal Affective Computing (MAC) interprets visual affective activity from visual and textual input. In a way, it can be seen as multimodal sentiment analysis.
    
- Natural Language for Visual Reasoning (NLVR) determines if a statement regarding a visual input is correct or not.
    

### Retrieval Tasks

- Visual Retrieval (VR) retrieves images based only on a textual description.
    
- Vision-Language Navigation (VLN) is the task of an agent navigating through a space based on textual instructions.
    
- Multimodal Machine Translation (MMT) involves translating a description from one language to another with additional visual information.
    
- The following image shows the taxonomy of popular visual language tasks:
    

![](https://aman.ai/primers/ai/assets/vision-language-models/vl-tasks.png)

- Depending on the task at hand, different architectures have been proposed over the years. In this article, we will explore some of the most popular ones.

## BERT-like Architectures

- Given the incredible rise of transformers in NLP, it was inevitable that people would also try to apply them in VL tasks. The majority of papers have been used some version of BERT 2, resulting in a simultaneous explosion of BERT-like multimodal models: [VisualBERT](https://arxiv.org/abs/1908.03557), [ViLBERT](https://proceedings.neurips.cc/paper/2019/file/c74d97b01eae257e44aa9d5bade97baf-Paper.pdf), [Pixel-BERT](https://arxiv.org/abs/2004.00849), [ImageBERT](https://arxiv.org/abs/2001.07966), [VL-BERT](https://arxiv.org/abs/1908.08530), [VD-BERT](https://arxiv.org/abs/2004.13278), [LXMERT](https://arxiv.org/abs/1908.07490), [UNITER](https://arxiv.org/abs/1909.11740).
    
- They are all based on the same idea: they process language and images at the same time with a transformer-like architecture. We generally divide them into two categories: two-stream models and single-stream models.
    

### Two-stream Models: ViLBERT

- Two-stream model is a literature term that refers to VL models which process text and images using two separate modules. [ViLBERT](https://proceedings.neurips.cc/paper/2019/file/c74d97b01eae257e44aa9d5bade97baf-Paper.pdf) and [LXMERT](https://arxiv.org/abs/1908.07490) fall into this category.
    
- [ViLBERT](https://proceedings.neurips.cc/paper/2019/file/c74d97b01eae257e44aa9d5bade97baf-Paper.pdf) 4 is trained on image-text pairs. The text is encoded with the standard transformer process using tokenization and positional embeddings. It is then processed by the self-attention modules of the transformer. Images are decomposed into non-overlapping patches projected in a vector, as in vision transformer’s patch embeddings.
    
- To learn a joint representation of images and text, a “co-attention” module is used. The “co-attention” module calculates importance scores based on both images and text embeddings.
    
- The following diagram compares standard self-attention vs. [ViLBERT](https://proceedings.neurips.cc/paper/2019/file/c74d97b01eae257e44aa9d5bade97baf-Paper.pdf)’s proposed co-attention:
    

![](https://aman.ai/primers/ai/assets/vision-language-models/co-attention.png)

- In a way, the model is learning the alignment between words and image regions. Another transformer module is added on top for refinement. This “co-attention”/transformer block can, of course, be repeated many times.
    
- The following diagram shows that [ViLBERT](https://proceedings.neurips.cc/paper/2019/file/c74d97b01eae257e44aa9d5bade97baf-Paper.pdf) processes images and text in two parallel streams that interact through co-attention:
    

![](https://aman.ai/primers/ai/assets/vision-language-models/vil-bert.png)

- The two sides of the model are initialized separately. Regarding the text stream (purple), the weights are set by pretraining the model on a standard text corpus, while for the image stream (green), [Faster R-CNN](https://paperswithcode.com/method/faster-r-cnn) is used. The entire model is trained on a dataset of image-text pairs with the end objective being to understand the relationship between text and images. The pretrained model can then be fine-tuned to a variety of downstream VL tasks.

### Single-stream Models

- In contrast, models such as [VisualBERT](https://arxiv.org/abs/1908.03557), [VL-BERT](https://arxiv.org/abs/1908.08530), [UNITER](https://arxiv.org/abs/1909.11740) encode both modalities within the same module. For example, [VisualBERT](https://arxiv.org/abs/1908.03557) combines image regions and language with a transformer in order for self-attention to discover alignments between them. In essence, they added a visual embedding to the standard BERT architecture. The visual embedding consists of:
    
    1. A visual feature representation of the region produced by a CNN.
    2. A segment embedding that distinguishes image from text embeddings.
    3. A positional embedding to align regions with words if provided in the input.
- The following diagram shows [VisualBERT](https://arxiv.org/abs/1908.03557) which combines image regions and text with a transformer module:
    

![](https://aman.ai/primers/ai/assets/vision-language-models/visual-bert.png)

## Pretraining and Fine-tuning

- The performance benefits of these models are partially due to the fact that they are pretrained on huge datasets. Visual BERT-like models are usually pretrained on paired image + text datasets, learning general multimodal representations. Afterwards, they are fine-tuned on downstream tasks such as visual question answering (VQA), etc with specific datasets.

### Pretraining Strategies

- Let’s explore some common pretraining strategies:
    - Masked Language Modeling is often used when the transformer is trained only on text. Certain tokens of the input are being masked at random. The model is trained to simply predict the masked tokens (words). In the case of BERT, bidirectional training enables the model to use both previous and following tokens as context for prediction.
    - Next Sequence Prediction works again only with text as input and evaluates if a sentence is an appropriate continuation of the input sentence. By using both false and correct sentences as training data, the model is able to capture long-term dependencies.
    - Masked Region Modeling masks image regions in a similar way to masked language modeling. The model is then trained to predict the features of the masked region.
    - Image-Text Matching forces the model to predict if a sentence is appropriate for a specific image.
    - Word-Region Alignment finds correlations between image region and words.
    - Masked Region Classification predicts the object class for each masked region.
    - Masked Region Feature Regression learns to regress the masked image region to its visual features.
- For example, [VisualBERT](https://arxiv.org/abs/1908.03557) is pretrained with the Masked Language Modeling and Image-text matching on an image-caption dataset.
    
- The above methods create supervised learning objectives. Either the label is derived from the input, aka self-supervised or a labeled dataset (usually image-text pairs) is used. Are there any other attempts? Of course.
    
- The following strategies are also used in VL modeling. They are often combined on various proposals:
    - Unsupervised VL Pretraining usually refers to pretraining without paired image-text data but rather with a single modality. During fine-tuning though, the model is fully-supervised.
    - Multi-task Learning is the concept of joint learning across multiple tasks in order to transfer the learnings from one task to another.
    - Contrastive Learning is used to learn visual-semantic embeddings in a self-supervised way. The main idea is to learn such an embedding space in which similar pairs stay close to each other while dissimilar ones are - far apart.
    - Zero-shot learning is the ability to generalize at inference time on samples from unseen classes.
- Let’s now proceed with some of the most popular architectures.

## VL Generative Models

### DALL-E

- [DALL-E](https://arxiv.org/pdf/2102.12092.pdf) tackles the visual generation (VG) problem by being able to generate accurate images from a text description. The architecture is again trained with a text-images pair dataset.
    
- DALL-E uses a discrete variational autoencoder ([dVAE](https://arxiv.org/abs/1609.02200)) to map the images to image tokens. dVAE essentially uses a discrete latent space compared to a typical VAE. The text is tokenized with [byte-pair encoding](https://en.wikipedia.org/wiki/Byte_pair_encoding). The image and text tokens are concatenated and processed as a single data stream.
    
- The following diagram shows the training pipeline of DALL-E mini, which is slightly different from the original DALL-e:
    

![](https://aman.ai/primers/ai/assets/vision-language-models/Training-pipeline-of-DALLE-mini.png)

- DALL-E uses an autoregressive transformer to process the stream in order to model the joint distribution of text and images. In the transformer’s decoder, each image can attend to all text tokens. At inference time, we concatenate the tokenized target caption with a sample from the dVAE, and pass the data stream to the autoregressive decoder, which will output a novel token image.

> DALL-E generates realistic images based on a textual description and provides some exceptional results (although admittedly a little cartoonized) as you can see in the image below (source: [DALL·E: Creating Images from Text](https://openai.com/blog/dall-e/)):

![](https://aman.ai/primers/ai/assets/vision-language-models/dall-e.png)

### GLIDE

- Following the work of DALL-E, [GLIDE](https://arxiv.org/abs/2112.10741) is another generative model that seems to outperform previous efforts. GLIDE is essentially a [diffusion model](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/).
- Diffusion models consists of multiple diffusion steps that slowly add random noise to the data. Then, they aim to learn to reverse the diffusion process to construct samples from the data distribution from noise. The following image from [Lilian Weng’s blog](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/) illustrates this idea:

![](https://aman.ai/primers/ai/assets/vision-language-models/Overview-of-diffusion-models.png)

- Diffusion models, in a nutshell, work by slowly injecting random noise to the data in a sequential fashion (formulated as a Markov chain). They then learn to reverse the process in order to construct novel data from the noise. So instead of sampling from the original unknown data distribution, they can sample from a known data distribution produced after a series of diffusion steps. In fact, it can be proved that if we add gaussian noise, the end (limit) distribution will be a typical normal distribution.
    
- The diffusion model receives input as images and can output novel ones. But it can also be conditioned on textual information so that the generated image will be appropriate for specific text inputs. And that’s exactly what GLIDE does. It experiments with a variety of methods to “guide” the diffusion models.
    
- Mathematically, the diffusion process can be formulated as follows. If we take a sample x0x0 from a data distribution q(x0)q(x0), we can produce a Markov chain of latent variables x1,…xTx1,…xT by progressively adding Gaussian noise of magnitude 1−at1−at:
    

q(xt∣xt−1):=N(xt;at‾‾√xt−1,(1−at)I)q(xt∣xt−1):=N(xt;atxt−1,(1−at)I)

- That way, we can well-define the posterior q(xt−1∣xt)q(xt−1∣xt) and approximate it using a model pθ(xt−1∣xt)pθ(xt−1∣xt).

pθ(xt−1∣xt):=N(μθ(xt),Σθ(xt))pθ(xt−1∣xt):=N(μθ(xt),Σθ(xt))

- To better understand diffusion models, I highly recommend this excellent [article by Lilian Weng](https://lilianweng.github.io/lil-log/2021/07/11/diffusion-models.html).
    
- GLIDE results are even more impressive and more realistic than DALLE. However, as the authors themselves admit, there have been quite a few failure cases for specific unusual objects or scenarios. Note that you can try it yourself using [hugging face spaces](https://huggingface.co/spaces/valhalla/glide-text2im). The following image shows examples of generated images by GLIDE:
    

![](https://aman.ai/primers/ai/assets/vision-language-models/glide.png)

## VL Models Based on Contrastive Learning

### CLIP

- [CLIP](https://cdn.openai.com/papers/Learning_Transferable_Visual_Models_From_Natural_Language_Supervision.pdf) targets the Natural Language for Visual Reasoning (NLVR) problem as it tries to classify an image to a specific label based on its context. The label is usually a phrase or a sentence describing the image. More interestingly, it’s a zero-shot classifier in terms that it can be used to previously unseen labels.
    
- Its admittedly impressive zero-shot performance is heavily affected by the fact that it is trained on a highly-diversified, huge (400 million) dataset. The training data consist of images and their corresponding textual descriptions. The images are encoded by either a ResNet or a transformer, while a transformer module is also used for text.
    
- The training’s objective is to “connect” image representations with text representations. In a few words, the model tries to discover which text vector is more “appropriate” for a given image vector. This is why it’s referred to as contrastive learning.
    
- For those familiar with purely vision-based contrastive learning, here instead of bringing together views of the same image, we are pulling together the positive image and text “views”, while pulling apart texts that do not correspond to the correct image (negatives). So even though it’s contrastive training it’s 100% supervised, meaning that labeled pairs are required.
    
- By training the model to assign high similarity for fitting image-text pairs and low similarity for unfitting ones, the model can be used in a variety of downstream tasks such as image recognition.
    

![](https://aman.ai/primers/ai/assets/vision-language-models/clip.png)

- In CLIP, the image encoder and the text encoder are trained jointly in a contrastive fashion 14
    
- Borrowed from the original paper, you can find a pseudocode implementation below:
    

![](https://aman.ai/images/copy.png)

`# image_encoder - ResNet or Vision Transformer # text_encoder - CBOW or Text Transformer # I[n, h, w, c] - minibatch of aligned images # T[n, l] - minibatch of aligned texts # W_i[d_i, d_e] - learned proj of image to embed # W_t[d_t, d_e] - learned proj of text to embed # t - learned temperature parameter # extract feature representations of each modality I_f = image_encoder(I) #[n, d_i] T_f = text_encoder(T) #[n, d_t] # joint multimodal embedding [n, d_e] I_e = l2_normalize(np.dot(I_f, W_i), axis=1) T_e = l2_normalize(np.dot(T_f, W_t), axis=1)  # scaled pairwise cosine similarities [n, n] logits = np.dot(I_e, T_e.T) * np.exp(t)  # symmetric loss function labels = np.arange(n) loss_i = cross_entropy_loss(logits, labels, axis=0) loss_t = cross_entropy_loss(logits, labels, axis=1) loss = (loss_i + loss_t)/2`

> The results are again quite impressive, but limitations still exist. For example, CLIP seems to struggle with abstract concepts and has poor generalization to images not covered in its pre-training dataset. The following image (from [CLIP: Connecting Text and Images](https://openai.com/blog/clip/)) shows an example of caption prediction for an image using CLIP:

![](https://aman.ai/primers/ai/assets/vision-language-models/clip-results.png)

### ALIGN

- In a very similar way, [ALIGN](https://arxiv.org/abs/2102.05918) utilizes a dual-encoder that learns to align visual and language representations of image-text pairs. The encoder is trained with a contrastive loss, which is formalized as a normalized softmax. In more detail, they authors use two loss terms, one for image-to-text classification and one for text-to-image classification.
    
- Given $x i$ and $y j$ the normalized embedding of the image in the $i^{th}$ pair and that of text in the $j^{th}$ pair respectively, $N$ the batch size, and $\sigma$ the temperature to scale the logits, the loss functions can be defined as:
    

Li2tLt2i=−1N∑iNlogexp(xTıyi/σ))∑Nj=1exp(xTıyi/σ))=−1N∑iNlogexp(yTixi/σ))∑Nj=1exp(yTixi/σ))Li2t=−1N∑iNlog⁡exp⁡(xıTyi/σ))∑j=1Nexp⁡(xıTyi/σ))Lt2i=−1N∑iNlog⁡exp⁡(yiTxi/σ))∑j=1Nexp⁡(yiTxi/σ))

- It’s other main contribution is that the training is performed with a noisy dataset of one billion image-text pairs. So instead of doing expensive preprocessing on the data as similar methods do, they show that the scale of the dataset can compensate for the extra noise. The following image delineates In ALIGN, Visual and language representation are learned jointly with contrastive learning 15

![](https://aman.ai/primers/ai/assets/vision-language-models/align.png)

### FLORENCE

- [Florence](https://arxiv.org/pdf/2111.11432.pdf) combines many of the aforementioned techniques to propose a new paradigm of end-to-end learning for VL tasks. The authors view Florence as a foundation model (following the terminology proposed by the Stanford team at [Bommasani et al](https://arxiv.org/abs/2108.07258)). Florence is the most recent architecture in this article and seems to perform SOTA results in many different tasks. Its main contributions include:
    
    - For pretraining, they use a hierarchical vision transformer ([Swin](https://arxiv.org/abs/2103.14030)) as the image encoder and a modified CLIP as the language decoder.
        
    - The training is performed on “image-label-description” triplets.
        
    - They use a unified image-text learning scheme, which can be seen as bidirectional contrastive learning. Without diving too deep, the loss contains two contrastive terms; an image-to-language contrastive loss and a language-to-image contrastive loss. In a way, they try to combine two common learning tasks: the mapping of images to the labels and the assignment of a description to a unique label.
        
    - They enhance the pretrained representations into more fine-grained representations with the use of “adapter” models. The fine-grained representations depend on the task: object-level representations, visual-language representations, video representations.
        
- That way, the model can be applied into many distinct tasks and appears to have very good zero-shot and few-shot performance. The following diagram from the [Florence](https://arxiv.org/pdf/2111.11432.pdf) paper shows an illustration of the Florence architecture:
    

![](https://aman.ai/primers/ai/assets/vision-language-models/florence.png)

## Enhanced Visual Representations

- While text encoding is usually done with a transformer-like module, visual encoding is still an area of active research. Many different proposals have been made over the years. Images have been processed with typical CNNs, ResNets, or Transformers. DALL-E even used a dVAE to compress the visual information in a discrete latent space. This is similar to words that are mapped to a discrete set of embeddings comprising the dictionary, but for image patches. Nonetheless, building better image encoding modules is a top priority at the moment.

### VinVL

- Towards that goal, the authors of [VinVL](https://arxiv.org/abs/2101.00529) pretrained a novel model on object detection using four public datasets. They then added an “attribute” branch and fine-tuned it, making it capable of detecting both objects and attributes.

> An attribute is a small textual description related to the image.

- The resulted object-attribute detection model is a modification of the [Faster-RCNN](https://blog.paperspace.com/faster-r-cnn-explained-object-detection/) model and can be used to derive accurate image representations

### SimVLM

- [SimVLM](https://arxiv.org/abs/2108.10904), on the other hand, utilizes a version of the [vision transformer (ViT)](https://theaisummer.com/vision-transformer/). In fact, they replaced the well-known patch projection with three ResNet blocks to extract image patch vectors (Conv stage in the image below). The ResNet blocks are trained together with the entire model, contrary to other methods where a fully-pretrained image module is used. The following diagram from the [SimVLM](https://arxiv.org/abs/2108.10904) paper illustrates the proposed architecture. The model is pretrained with a unified objective, similar to language modeling, using large-scale weakly labeled data:

![](https://aman.ai/primers/ai/assets/vision-language-models/simvlm.png)

## Conclusion and Observations

- Given the fact that all the said models are barely new, it seems that the research community still has a long way to go in order to build solid visual language models. We have seen an explosion of very similar architectures from different teams, all following the pretraining/fine-tune paradigm of large-scale transformers. I could include many more architectures in this article, but it seems that it wouldn’t have provided much value.
- The thing that concerns me is that the majority of the models come from big-tech companies, which is clearly a sign that huge datasets and infrastructure needs are required.
    
- It is also clear to me that contrastive learning approaches are the go-to method for the moment with CLIP and ALIGN being instrumental in this direction. While the text encoding part is kind of “solved”, much effort is needed to gain better visual representations. Moreover, generative models such as DALLE and GLIDE have shown very promising results, but they also come with many limitations.

## References

- Mogadala, Aditya, et al. “Trends in Integration of Vision and Language Research: A Survey of Tasks, Datasets, and Methods.” Journal of Artificial Intelligence Research, vol. 71, Aug. 2021, pp. 1183–317.
- Devlin, Jacob, et al. “BERT: Pre-Training of Deep Bidirectional Transformers for Language Understanding.” ArXiv:1810.04805 [Cs], May 2019.
- Li, Liunian Harold, et al. “[VisualBERT: A Simple and Performant Baseline for Vision and Language.](https://arxiv.org/abs/1908.03557)” ArXiv:1908.03557 [Cs], Aug. 2019.
- Lu, Jiasen, et al. “[ViLBERT: Pretraining Task-Agnostic Visiolinguistic Representations for Vision-and-Language Tasks.](https://proceedings.neurips.cc/paper/2019/file/c74d97b01eae257e44aa9d5bade97baf-Paper.pdf)” ArXiv:1908.02265 [Cs], Aug. 2019.
- Huang, Zhicheng, et al. “Pixel-BERT: Aligning Image Pixels with Text by Deep Multi-Modal Transformers.” ArXiv:2004.00849 [Cs], June 2020.
- Qi, Di, et al. “ImageBERT: Cross-Modal Pre-Training with Large-Scale Weak-Supervised Image-Text Data.” ArXiv:2001.07966 [Cs], Jan. 2020.
- Su, Weijie, et al. “VL-BERT: Pre-Training of Generic Visual-Linguistic Representations.” ArXiv:1908.08530 [Cs], Feb. 2020.
- Wang, Yue, et al. “VD-BERT: A Unified Vision and Dialog Transformer with BERT.” ArXiv:2004.13278 [Cs], Nov. 2020.
- Tan, Hao, and Mohit Bansal. “LXMERT: Learning Cross-Modality Encoder Representations from Transformers.” ArXiv:1908.07490 [Cs], Dec. 2019.
- Chen, Yen-Chun, et al. “UNITER: UNiversal Image-TExt Representation Learning.” ArXiv:1909.11740 [Cs], July 2020.
- Ramesh, Aditya, et al. “Zero-Shot Text-to-Image Generation.” ArXiv:2102.12092 [Cs], Feb. 2021.
- Rolfe, Jason Tyler. “Discrete Variational Autoencoders.” ArXiv:1609.02200 [Cs, Stat], Apr. 2017.
- Nichol, Alex, et al. “GLIDE: Towards Photorealistic Image Generation and Editing with Text-Guided Diffusion Models.” ArXiv:2112.10741 [Cs], Dec. 2021.
- Radford, Alec, et al. “Learning Transferable Visual Models From Natural Language Supervision.” ArXiv:2103.00020 [Cs], Feb. 2021.
- Jia, Chao, et al. “Scaling Up Visual and Vision-Language Representation Learning With Noisy Text Supervision.” ArXiv:2102.05918 [Cs], June 2021.
- Yuan, Lu, et al. “Florence: A New Foundation Model for Computer Vision.” ArXiv:2111.11432 [Cs], Nov. 2021.
- Zhang, Pengchuan, et al. “VinVL: Revisiting Visual Representations in Vision-Language Models.” ArXiv:2101.00529 [Cs], Mar. 2021.
- Wang, Zirui, et al. “SimVLM: Simple Visual Language Model Pretraining with Weak Supervision.” ArXiv:2108.10904 [Cs], Aug. 2021.

## Further Reading

- If you interested in diving more into vision-language models, here are some excellent surveys:
    - Baltrušaitis, Tadas, et al. “Multimodal Machine Learning: A Survey and Taxonomy.” ArXiv:1705.09406 [Cs], Aug. 2017.
    - Guo, Wenzhong, et al. “Deep Multimodal Representation Learning: A Survey.” IEEE Access, vol. 7, 2019, pp. 63373–94. IEEE Xplore.
    - Zhang, Chao, et al. “Multimodal Intelligence: Representation Learning, Information Fusion, and Applications.” IEEE Journal of Selected Topics in Signal Processing, vol. 14, no. 3, Mar. 2020, pp. 478–93.
    - Uppal, Shagun, et al. “Multimodal Research in Vision and Language: A Review of Current and Emerging Trends.” ArXiv:2010.09522 [Cs], Dec. 2020. .
        
        ## Citation
        

If you found our work useful, please cite it as:

![](https://aman.ai/images/copy.png)

`@article{Chadha2020DistilledVisionLanguageModels,   title   = {Vision Language Models},   author  = {Chadha, Aman},   journal = {Distilled AI},   year    = {2020},   note    = {\url{https://aman.ai}} }`

-  [](https://github.com/amanchadha)|  [](https://citations.amanchadha.com/)|  [](https://twitter.com/i_amanchadha)|  [](mailto:hi@aman.ai)| 

[www.amanchadha.com](https://www.amanchadha.com/)