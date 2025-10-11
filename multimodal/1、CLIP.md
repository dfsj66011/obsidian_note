
## 一、教程 1

CLIP（Contrastive Language-Image Pre-training）是 OpenAI 在 2021 年发布的一个模型，因为 SD 技术而再次走红，它在当时颇具革命性，尤其是因为它采用了一种新颖的方式来连接文本和图像。

首先我们将探讨什么是 CLIP 以及它如何实现文本与图像的连接，其次，在深入探讨之前，我们还会先了解为什么最初需要 CLIP。

首先，我们关注的任务是图像分类，在 CLIP 出现之前，使用的是 CNN，这些网络被训练用于将图片和图像分类到不同的类别中，例如，我们看看这张来自谷歌网站的图片，

![[CNN 网络架构图.png|600]]
例如，如果以前我们有猫或狗的图片，并希望将它们分类为两个类别，我们就必须创建这个 CNN，通过大量的卷积和最大池化操作，最终通过一个全连接层，为最能代表输入图像的类别给出最高分数。例如，在这种情况下，猫的输出激活值会是最高的，因为我们输入的是一张猫的图片。如果是狗的图片，那么狗的输出值将会是最高的。

这种方法实际上效果很好，但问题在于，我们需要大量的图片，我们需要大量的标注数据集（耗时耗力），如果类别数量较少且彼此差异较大，这样做是可以的。然而，在某些领域，构建这样的数据集并不容易，实际上成本也很高。比如医学研究领域，其中的图片必须由医生或有相关知识的人来标注，你不能随便找个人来分类医学设备中的癌症和非癌症图像，因此构建这种数据集的成本非常高，而且他们发现这些数据集无法泛化到其他任务上，例如，一个狗和猫数据上训练的分类器，很难轻易泛化到其他类型的类别上，并且在其他类型的分类任务上表现会很差。

那么 CLIP 是如何解决这个问题的？是对比学习预训练，其工作原理如下图所示。

![[CLIP 原理图.png]]

CLIP 由两个编码器组成，文本编码器和图像编码器，*CLIP 的输入是什么*？我们给它一批文本和相应的图像，这意味着文本批次中的第一项与图像批次中的第一张图像相对应，所以文本 “Pepper the aussie pup（澳洲小狗佩珀）” 对应于示例中这张图片。

我们从 *哪里获取所有这些图片* ？CLIP 的作者从互联网上获取了这些图片和文本，创建了一个包含 *4 亿*张从互联网收集的图片的数据集，这些图片据称被用户或作者很好的描述了。通常当你在互联网上找到一张图片时，实际上你不仅找到了图片本身，还找到了图片背后的描述，特别是在社交网络上，例如，人们去某个地方旅行，他们会写下关于图片内容的东西，而且这不仅仅是一个单词，这就是为什么他们称之为自然语言监督。

其 *工作方式* 是，他们取一批文本，并通过文本编码器转换，为我们提供了该文本的一些特征，这些特征实际上随后乘以另一个矩阵，以便特征的维度是特定的维度，然后对图像做同样的事情，通过图像编码器转换图像，并乘以另一个矩阵，使图像具有与文本特征相同的维度，接下来做点积运算，余弦相似度矩阵。

在图中可以看到，他们计算了每种可能的文本和图像组合之间的余弦相似度，我们期望什么呢？我们知道，根据事实，我们希望对角线上的元素能够相互匹配，具有最高的相似度，而其他的相似度则较低，甚至为零。

实际上，这段代码在论文中写明了：

```python
# image_encoder   -   ResNet or Vision Transformer 
# text_encoder    -   CBOW or Text Transformer 
# I[n, h, w, c]   -   minibatch of aligned images 
# T[n, l]         -   minibatch of aligned texts 
# W_i[d_i, d_e]   -   learned proj of image to embed 
# W_t[d_t, d_e]   -   learned proj of text to embed 
# t               -   learned temperature parameter 

# extract feature representations of each modality 
I_f = image_encoder(I)      # [n, d_i] 
T_f = text_encoder(T)       # [n, d_t] 

# joint multimodal embedding [n, d_e] 
I_e = l2_normalize(np.dot(I_f, W_i), axis=1) 
T_e = l2_normalize(np.dot(T_f, W_t), axis=1) 

# scaled pairwise cosine similarities [n, n] 
logits = np.dot(I_e, T_e.T) * np.exp(t) 

# symmetric loss function 
labels = np.arange(n) 
loss_i = cross_entropy_loss(logits, labels, axis=0) 
loss_t = cross_entropy_loss(logits, labels, axis=1) 
loss = (loss_i + loss_t) / 2
```

假设我们有一批图像，通过图像编码器传递它以获取维度为 $d_i$ 的嵌入。然后对文本做同样的事情，将其传递给文本编码器，获取维度为 $d_t$ 的特征；然后将图像的特征和文本的特征分别与两个矩阵相乘，使它们各自的结果特征大小为 $d_e$，然后对每一对进行余弦相似度计算，并计算出 logits，

再然后我们计算损失，预期的标签是按行，比如第一行第一个值最高，第二行第二个值最高...，或者按列...，我们希望该特定行或列中的特定位置具有最高值，我们将在两个轴上将生成的 logits 与 labels 进行比较，并将二者求均值，这就是我们的损失函数。

以上就是*训练的工作原理*，对于这种对比训练，那我们如何进行 *推理* 呢？推理相当简单，而且效率也很高，首先假设我们有一张狗的图片，我们实际上做先创建一个 prompt，如“a photo of a xxx”，我们所做的是创建一个我们期望与之拼接的类别列表，在这种情况下，我们可以处理飞机、汽车、狗、鸟等，我们将所有这些可能的类别通过这个提示传递，生成相应的 prompt 特征。例如“一张关于飞机的照片”，生成其特征为 $T_1$，然后是汽车，生成另一个特征放入 $T_2$，...，我们计算所有这些特征并将其放在一边。（注意，所有这些计算都只需要计算一遍，并保存起来）

然后，拿出一张狗的图片，通过图像编码器传递它，计算其特征，将之前计算的内容与图像特征相乘，并且具有最高值的那个将成为选定的标签，这就是推理的工作方式，它相当高效。

CLIP 最好的方面是它能在零样本任务上表现非常出色，当然对于计算图中对象数量这种问题表现不佳。

另外补充一点，它们是如何提取特征的？作为图像编码器，作者使用了 ResNet 和 Vision Transformer，对于它们，我们只需从最后一层提取特征，就这样；关于文本编码器，作者实际上使用的是一个 Transformer，但他门只使用了编码器部分，当然只是提取最后一层中与文本结束标记对应的特征。


-----------
## 二、教程 2

> from:[Using CLIP to Classify Images without any Labels](https://cameronrwolfe.substack.com/p/using-clip-to-classify-images-without-any-labels-b255bb7205de)


**CLIP 模型整体架构和训练方法：**

![|450](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2Fe398687c-3716-4998-923a-9afa8b6ef157_800x480.png)

**视觉编码器**：对于图像编码器，他们探索了不同的模型架构，包括五个不同大小的 ResNet（即，使用 EfficientNet 样式模型缩放规则确定模型尺寸）和三个 ViT 架构。ViT 版本在训练时的计算效率高出 3 倍，成为了首选的图像编码器架构。两种架构的图例如下所示：

![|450](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F998aad2b-fb98-458c-9cbb-4ba036e32e60_800x565.png)

**文本编码器：** 文本编码器只是一个仅用于解码器的 Transformer，这意味着每个层都使用 mask 自注意力（而不是双向自注意力）。

![|450](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F093779d9-baa8-4272-9db5-156b70e9dd06_800x511.png)

**确定训练任务**：尽管之前的研究表明自然语言是计算机视觉的可行训练信号，但用于训练 CLIP 图像和文本对的确切训练任务并不明显。我们应该 _根据标题中的单词对图像进行分类_ 吗？之前的研究已经尝试过这种方法，效果不错，但不是很好。那么 _使用语言建模为每幅图像生成标题_ 怎么样？有趣的是，作者发现预测准确的图像标题太难了——导致模型学习非常缓慢——因为任何图像都可以用多种不同的方式描述。

**更好的任务 = 更快的学习。** 通过使用对比学习的代理任务训练 CLIP 模型，作者观察到训练效率提高了 4 倍：

![|350](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F37f27635-f124-44a1-8ad3-01a23d4a34ce_475x316.png)

这里，训练效率是使用 ImageNet 上的零样本学习迁移率来衡量的。换句话说，当使用这个简单的目标时，CLIP 模型花费更少的训练时间（就观察到的图像文本示例数量而言）来实现在 ImageNet 上产生高零样本准确率的模型。_因此，正确选择训练目标对模型效率和性能有重大影响。_


---------
## 三、论文精度（朱毅）

> [视频讲解](https://www.bilibili.com/video/BV1SL4y1s7LQ/?spm_id_from=333.337.search-card.all.click&vd_source=aced32e35ad9cff83fe98c60854f183c)

**为什么要做提示工程和提示集成？两个问题：**

1. 解决多义性问题，例如  crane（起重机，鹤）、boxer（一种狗，拳击手）remote（遥控器，远端） 
2. 预训练的时候匹配的是文本句子，如果测试时进来的都是单词，就会存在 distribution gap 问题，抽出来的特征呢可能就不是很好

于是提出一个非常简单的方式，prompt template，把所有标签都变成一个句子，其次模板句式里预留的标签多为名词，缓解多义问题。准确度可提升 1.3%。

提示工程中，不仅提供一个提示模板，还可以做很多事情，比如当有图片的先验时，例如都是动物图片，则 "A photo of a {label}, a type of pet." 等，缩小了解空间。

提示集成，用了 80 个不同的模板进行集成，详见[代码库](https://github.com/openai/CLIP/blob/main/notebooks/Prompt_Engineering_for_ImageNet.ipynb)

第六章：局限性，分析一下这些不足和这些局限性其实比这篇文章到底怎么做的都要有意义的多，因为这些不足和局限性才能引发更多的思考，而让整个领域向前走。

1. 未和 SOTA 模型比，ResNet 50 本身比较弱，以 ImageNet 为例，CLIP 76.2，而noisy student，ViT、MAE 等都在 88、89 甚至 90 以上；CLIP 虽然强但也没有太强，且预估要达到这个水平，数据集需扩大 1000 倍，不太现实。
2. 细分类的数据集、抽象的概念或者说更难的任务，比如计数问题，基本与瞎猜一样
3. 泛化虽然好，但如果真的相差很远，out of distribution，泛化同样很差，MNIST 上准确率只有 88%，4 亿张图片中没有与 MNIST 相似的图片；
4. 虽然可以做 zero-shot，但还是从给定的类别中去做选择，还是想回到 gpt 老本行上，直接生成一个 title，但受限于资源；
5. 数据利用率不高，需要大量的数据去投喂；32 个Epoch，那每个 Epoch 要过 4 亿个图片，一共 128 亿张图片，如果 DataLoader 的速度是每秒钟出一张，全看完就需要花 405 年的时间；
6. 为了调参，测试集并多次使用，代入了偏见；
7. 网络采集，未清洗，可能代入社会偏见；
8. 很多很复杂的任务或者概念，可能即使用语言也无法描述，不如提供一些这种训练样本，还是非常有帮助的，但 CLIP 并不是为了 few shot，也不是为了 few shot 优化的，就导致了一个非常奇怪的现象，当给  CLIP 提供了一些训练样本 one shot 、two shot、four shot 的时候，结果反而还不如直接用 zero shot，这个就很耐人寻味了

最后总结一下：CLIP 最大的贡献，在我看来是他打破了之前这种固定种类标签的范式。直接搜集图片-文本的配对，用无监督的方式，要么去预测他的相似性，要么去生成它。

这样的好处不仅在处理数据的时候更方便，训练模型更方便，最主要的是在做推理的时候更方便，甚至可以去 zero shot 各种各样的分类任务；所以之后，很快就有一大批工作迅速更进，如物体检测、物体分割、视频动作识别、检索、多模态、图像生成等；

新意度角度：打破了这种固定类别标签的做法，彻底放飞了视觉模型的训练过程；
有效性角度：效果好，泛化性能好，甚至在某些情况下比人的 zero shot 性能还好；
问题大小方面：用一个模型就能解决大部分的这个分类任务而且是 zero shot 的
