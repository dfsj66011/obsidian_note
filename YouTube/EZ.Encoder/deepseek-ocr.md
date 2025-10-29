
Paper: [# DeepSeek-OCR: Contexts Optical Compression](https://arxiv.org/html/2510.18234v1)

对文本的信息进行视觉压缩，也就是这里所提到的 context optical compression 

我想从三个方面给大家来撸这篇 paper：

1. 理解一下这篇 paper 背后的 motivation，这个对理解整篇 paper 的核心思想非常的重要
2. 相关的 paper，尤其是 VLM 主流的一些方法，对理解这篇 paper 是非常有必要的
3. 这篇 paper 具体的方法和实验结果

那话不多说我们就直接开始 这篇 paper 只有三个作者，像这种比较重量级的 paper 只有三个作者还是比较少见的，大概查了一下，这个一作，魏浩然之前是在旷世工作，有很多关于 OCR 的 paper，这个小哥主要是工程背景，我的一个猜测是这篇 paper 可能大部分的工作或者主要的想法都是由这个一作完成的，这篇 paper 里面所提到的很多工作，包括 Vary, GOAT, OCR 2.0, Fox, Benchmark 都来自于这个小哥，这个一作非常的强，从某种程度来说，我觉得这一篇 DeepSeek-OCR 也基本上是在延续这个一作之前的一些工作，加上他在旷世有一段经历，所以他有一些视觉的背景，很自然的就把视觉里面的一些思想放到了文本处理里面，当然这都只是我的一个猜测。

### 动机

想法其实比较简单，就是我们在使用 LLM 的时候，基本上都是把文本信息经过一个 tokenizer 直接输入到 LLM 当中，DeepSeek-OCR 提出了一个想法，就是这么做有很多的弊端，最大的一个弊端就是这个文本可能有很多的 token，由于现在 LLM 的限制，处理这种长序列还有存在很多的问题，那能不能对这个文本进行压缩呢，作者就提出了一个直接的想法，就是把这个文本转化成一个图片，这样这些信息原封不动的就从文本变成了图像信息，然后再经过一个 vision encoder 对这个图像信息进行编码，然后再输入到 LLM 当中进行处理，那这个过程作者就把它叫做 context optical compression。

这样做的一个好处就是转化成图片之后，在处理图片时，一般会把它打成 patch token，所以最后从 vision encoder 出来的 token 数目是要远小于直接把文本经过 tokenizer 出来的数目，所以作者的一个想法就是，以后对于这些特别长的文本信息，都可以通过这种视觉的方式输入到 LLM，而不是直接的通过文本的方式输入到 LLM，通过这种压缩的方式来处理信息，从而避免前面提到的长序列问题。

然后作者为了展示他们这个想法，在 OCR 一系列的任务上面进行了实验，结果相当的不错，这就是这篇 paper 最基本的一个想法，作者在他们的 paper 里面也写了一句话

如果一个文档里面含有 1000 个字 那么需要多少个 vision tokens 能够正确的把这 1000 个字的信息给恢复出来，作者这里用一句话来总结他们的思想，就是 a picture is worth a thousand words，一图胜千言，这就是这篇 paper 核心的思想。

### 一些背景

**（ViT）**:   16x16 的 patch，但这个方法有一个很严重的问题，就是它会把这个图像切成 fixed size 的patch，那对于更高分辨率的图像，ViT一般的做法是保持 size 不变，这样就会导致更多的vision tokens，也就是这个序列的长度会变长，所以 ViT 如果是在低分辨率图像上面训练，那直接用到高分辨率的图像上面，会有一些问题，主要是位置编码会不一致，所以后续有很多的工作都在尝试解决 ViT 如何用在不同的分辨率，尤其是高分辨率图像上面。

那为什么我们想把 transformer 用到 image 相关的任务上面，答案就是在 NLP 领域里面，我们看到 transformer 这样的模型能够很容易的 scale up，所以自然的想到在 vision 领域也可以这样。 CNN 一类的模型随着数据量增加，性能会逐渐的趋于饱和，这是因为 CNN 模型它有比较强的归纳偏置，使得它 scale up 的能力并没有 ViT 这一类模型好，而 ViT 模型有一个比较大的缺陷，就是它对于这种不同 scale 的视觉信息学习的不是很好，尤其是高分辨率的图像。

**(Swin Transformer)：** 所以后续一个很著名的工作就是 swin transformer 的提出，就是为了解决这个问题，它其实借用了 CNN 的思想，把 transformer 的 attention 分成了层级的结构，先在局部做attention 然后再 merge，在更高的 scale 上面做 attention。（详见李沐视频）李沐就提到 swin transformer 其实是披着 transformer 的 CNN 模型。但现在 swin transformer 在很多的任务上面，如 object detection 还是用的很多的。但在 LLM 场景下，还是没有 ViT 用的多，原因就是它其实又引入了一些归纳偏置，比如这些局部的 attention，所以它的scalability 能力并没有 ViT 好，这就为什么在后续做 VLM 的时候，仍然还是使用 ViT 这样的架构。但是使用 ViT 架构的话，不同 resolution 的问题是仍然存在的，后面很多的工作还都在解决这个问题。

**（CLIP）：** OpenAI 2021 年的 paper，使用 contrastive learning，

**（Next-GPT）**

<img src="https://arxiv.org/html/2309.05519v3/x1.png" width="600">

如果已经有一个类似于 chatgpt的模型，想把它的能力给扩充，比如让它能够看图听话，甚至看视频，那该怎么做？想法非常的简单，对于不同模态的数据，只要加一个对应的 encoder 即可，如果想输出相对应的模态，主流做法目前还是用 diffusion model。

DeepSeek-OCR 也涉及到这一块，**如何在 LLM 上面加 image encoder？**

传统的做法：基本上 image encoder 就是 ViT，或者是 CNN 或二者的组合，具体做法

**（Vary）：**
<img src="https://arxiv.org/html/2312.06109v1/x1.png" width="600">
看一下这个图的上半部分即可，中间是文本的信息经过 text tokenizer 产生 text token，然后经过 linear projection 产生 text 所对应的 embedding；上面 image 通过 image encoder 产生 image tokens，然后需要 projection layer 产生对应的 embedding，这里有很多的方法，例如可以用 linear 或 attention 或 Q-former。Q-former 本质是 cross attention。然后把这两个 embedding 结合起来，一起送到 LLM 里面。

几个其他的主流 VLM 的架构，会发现基本上都是引用类似的架构。

**（DeepSeek-VL2）** 

<img src="https://arxiv.org/html/2412.10302v1/x2.png" width="600">
基本上也是前面看到的那种架构，一个 LLM，然后有 text 信息进来，同时也有 Vision 的信息，需要一个 Vision Encoder，一般会有一个 adapter，把 Vision encoder 出来的信息做一个投影。这里不同的是 DeepSeek-VL2 引入了动态平铺（dynamic tiling），简单的意思就是把这个图片分成块，目的是有的时候这个图分辨率比较高，同时长宽比也不一定是固定的，所以根据分辨率以及长宽比，动态的去选择如何切这个图片。这样做的目的是让 VLM 尽可能的去处理不同分辨率、不同长宽比的图片。

**Tiling 的想法**，Intern VL1.5

<img src="https://arxiv.org/html/2404.16821v2/x4.png" width="500">
简单来说就是，如果你有一个输入的图像，比如这个分辨率相对来说比较大，这个时候就要把它切成不同的 tile，因为分辨率不固定，并且长宽比也不一定，所以这个时候就会使用一组预先定义好的长宽比 ratio，去找一个最合适的，把图切成 tile。比如这里选的是 2:3，然后会把图 resize 成 896×1344，然后再把它切成 patch。在这里还会加一个 thumbnail，也就是一个缩略图，这样让 ViT 能看到一个全局的信息。

这一类的方法其实有一定的问题，在 deepseek-ocr 这篇 paper 里面也提了，如果图像分辨率比较大，会拆成很多小的 tile 去处理，虽然这种方法能够处理比较高的分辨率，但是由于每个 tile 的分辨率比较低，比如这里只有 448×448，然后 tile 是进行并行处理的，虽然有一个 thumbnail，但并不能很好的利用原始图片中整体的这些信息，因为整体信息被压缩到 thumbnail，然后每个 tile 又是平行处理的，所以这一类的方法对全局信息利用的并不是特别的好。

**（deepseek-VL，deepseq-VL2 的前续工作）**

<img src="https://arxiv.org/html/2403.05525v2/x3.png">
这个图展示了几个重要的信息：

可以看到 VLM 的整体的架构还是前面提到的一样，有一个 LLM，Vision Encoder，text信息。这里的 Vision Encoder是有两个：SAM-B 和 SigLIP-L，这也是现在 VLM 里面用的挺多的，将不同的 Vision Encoder 混在一起用，这里只用了两个，也有人用了 4 个。目的是因为不同的 Vision Encoder 可能处理的是不同 resolution 的图片，以及 Image Encoder 在 pre-train 的时候，在不同的数据以及用的不同的方法进行训练，所以提取出来的 feature 可能是不一样的，这个不一样有可能是因为训练的时候图像的 resolution 也有可能是数据本身或者是训练方式。*（在 DeepSeek-OCR 里面也提到，这种方法其实也不是特别的好，它的一个问题就是因为使用了多个 Image Encoder 之后，它的部署，尤其是在需要并行的时候会比较的麻烦）*

另外这个图还展示了一个比较重要的信息，就是如何去训练这种 VLM，这也是现在常用的方式，在这里，它是分了三个 stage：

1. stage1 里面只训 adapter，即 LLM 以及 Image  Encoder 都是训练好的，adapter 的作用就是类似于把图像的 embedding 投影到 LLM embedding 里面去，因为 Image Encoder 并没有像 CLIP 那样用 contrast 能力的方式训练，这种都只是简单的在可能纯视觉的数据上面训练的，所以它并没有跟 LLM 自身的 embedding 对其。所以就需要训练 adapter。 
2. adapter 训练完了之后，再把 LLM 权重放开，两个部分一起训练，最后再把 Image Encoder 放开，三个部分进行一起训练。（DeepSeek-OCR 也基本上采用了这个思路，把 LLM 冻住，先训练 Image Encoder，然后再放开一起训练）


**（Cambran-1）**: 我们再来看一篇来自于谢赛宁和杨乐昆的一篇 paper ，（表 3）

这篇 paper 有一个重要的点，就是它用了 4 个 Image Encoder，SigLIP, DINOv2, ConvNext 和 CLIP，并标明了每个 Image Encoder 接受图片的分辨率，比如 SigLIP 是384，DINOv2 是 518，ConvNext 是 1024，CLIP 是 336，它们接受的图片的分辨率是不一样的。

通过这里我们能看到 VLM 领域，大家在试图增加 Image Encoder 的个数，去提取图像中不同分辨率的 feature，*在 DeepSeek-OCR 中就批判了这种方法*，因为这种方法产生的 token 会非常的多，每一个 Vision Encoder 都会产生大量的 token。所以 DeepSeek-OCR 就是反其道而行之，尽量简化这些 Image Encoder，让它产生的 Vision token 尽可能的少，高效的去提取信息，而不是用这种笨重冗余的方法提取信息。

**（NaViT）** Google，2023 的，DeepSeek-OCR 里面提到了这篇 paper 所使用的一个思想， 该论文主要是想把 ViT 用到任何的长宽比以及分辨率上面，这也是前面很多 VLM 模型试图在解决的一个问题，想法非常的简单，就是把一个图片切成很多的 patch，然后把这个patch 直接放到一个 sequence 里面，然后送到模型当中，非常简单粗暴，当然这个方法还有一些特殊的处理，这里就不赘述了，但这个方法有一个显而易见的缺点，就是它会产生大量的 Vision tokens，序列长了之后 LLM 处理起来就会有一些的问题，所以这也是 DeepSeek-OCR 这篇 paper 想要解决的问题。 

------

### DeepSeek-OCR

主要想法就是说，对于这些长的 context，我们是否能通过这种 Optical 2D mapping，也就是说把长的 context，转成 2D 的图像来处理。

DeepSeek-OCR 主要有两个部分：DeepEncoder 和 Decoder，基本都是这样的一个架构，这里的 Decoder 是一个 LLM，DeepEncoder 是 ImageEncoder，只不过 DeepSeek 在设计 DeepEncoder 的时候，主要是想设计一个简单实用的 ImageEncoder，使得它产生的 Vision tokens 尽可能的少，在同时能够保证它 decode 出来的 precision 是足够的高。

作者想通过这种方式对 LongContext 进行压缩，实验发现这种方式压缩比可以做到非常的高，同时模型在 OCR 的性能上面已经能够和现在比较主流的一些模型相媲美，代码和模型都已开源。
<img src="https://cas-bridge.xethub.hf.co/xet-bridge-us/68f1e08ddba20aca9c602acb/cb4d14e415c5de07fa77ba737a28c5fe10ce6b8a156a5208ee1f17686c2336eb?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=cas%2F20251028%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20251028T101037Z&X-Amz-Expires=3600&X-Amz-Signature=94781e09ade0b68daa0e125bfe0c6dd0bac3d0648f79f693d110877b2633da5d&X-Amz-SignedHeaders=host&X-Xet-Cas-Uid=public&response-content-disposition=inline%3B+filename*%3DUTF-8%27%27fig1.png%3B+filename%3D%22fig1.png%22%3B&response-content-type=image%2Fpng&x-id=GetObject&Expires=1761649837&Policy=eyJTdGF0ZW1lbnQiOlt7IkNvbmRpdGlvbiI6eyJEYXRlTGVzc1RoYW4iOnsiQVdTOkVwb2NoVGltZSI6MTc2MTY0OTgzN319LCJSZXNvdXJjZSI6Imh0dHBzOi8vY2FzLWJyaWRnZS54ZXRodWIuaGYuY28veGV0LWJyaWRnZS11cy82OGYxZTA4ZGRiYTIwYWNhOWM2MDJhY2IvY2I0ZDE0ZTQxNWM1ZGUwN2ZhNzdiYTczN2EyOGM1ZmUxMGNlNmI4YTE1NmE1MjA4ZWUxZjE3Njg2YzIzMzZlYioifV19&Signature=RTkPwjcqwcbr441gdpmYlPdiw-tyFcPKyKlqzyo%7EDy3WeaSxhBrClyCRQC7xEsovnQbrd3vYoZF2Qne8-6aL5WL52D3N7kpXLrQK4kWyWXqMBXsZpbkXDJOnWJ5Q5DSCd52EBUH7wHcdULjMRC5pv%7Ep9%7E1ookhTxVnQVuovTBPkMCcPLFGhajgk9VbL%7E8ZSVUszrmhv08fcF5L%7EpQD9k5xkYcXVsgvAm7EtCB3O3wBA-JaeZG1m7aqVJQ%7EheRs9fhLCx83fjmJN9Ee0tyQGV%7EIOnSSRb1fbLGXKnXK7XJinVzR5I-RqlGiEjuhOgsudUazS9xxQMt3QDIbPSw8edNA__&Key-Pair-Id=K2L8F4GPSG1IFC">

图 1 是在 Fox Benchmark 上面的结果，这个也是一作的一个工作，两个纵轴分别是 Precision 和 Compression，横坐标是 Context 的 token 实际数目，从左到右 token 是越来越多，分别把对应的这些 token 压缩到 64个 Vision tokens 或者是 100 个 Vision tokens，虚线表示的是压缩比，随着原始文本的 token 越来越多，如果保证 Vision tokens 是不变的话，压缩比是越来越高的，这个比较容易理解，同时看这些柱状图对应的 Precision 是相当的高的，哪怕是在 10 倍左右的压缩比的情况下。

右边这个图是展示的在 OmnidocBench 上面的实际的结果，横坐标是 Vision token 使用的数目，从左到右依次降低，纵坐标是 Performance 越低越好，也就是越往上越好，所以在这个图上面，在右上角的这些模型是比较好的，也就是用了比较少的 Vision tokens 并且 performance 比较好，可以看到 DeepSeek-OCR 的这一系列模型，基本都在这个右上角。但是单纯从 performance 来看的话，比如说像小红书最近出的 DOS.OCR 这个性能还是相当不错的，但是这些模型产生的 Vision tokens 就会非常的多，也就当然 DeepSeek-OCR 如果用比较多的 Vision tokens 的话，它的性能也可以接近于或者是超过这些主流的模型，比如 DOS.OCR 或者是 Qwen 以及 InternVL 系列。

<img src="https://arxiv.org/html/2510.18234v1/x1.png">

图 2 中，作者总结了一下当前主流 VLMs 的架构，这些在前面已经简单的介绍了一下，作者在这里把 VLMs 分了几个类别，简单来说就是如何去处理不同分辨率图像的方法的不同。

第一类，比如 DeepSeekVL, Vary 这类的方法就是用多个 encoder，这些 encoder 处理的图像的分辨率是不一样的，这里只用了两个 encoder，前面也提到 Cambran-1 甚至用到了 4 个encoder；

第二类，DeepSeekVL2 或 InternVL 系列，用 tiling 的方法，把图像按预先设定好的长宽比切分，这里还会对整个图做一个小的缩略图，通过这个缩略图来获得全局的信息；

第三类，上面提到的 Google 的那篇工作，包括 Qwen 系列也都是使用这种方法，直接把图片切成很多的 patch 暴力的放到 ViT 当中一起处理。

这些方法都有很多的问题，最主要的问题就是，很多方法产生的 Visual Token 太多了，对于高分辨率的图像信息处理的并不好，尤其是如果输入它的 resolution 是可变的；另外就是会产生 large activations，会占用大量的显存以及速度的问题。
<img src="https://arxiv.org/html/2510.18234v1/x2.png">
所以作者就提出了架构 DeepSeek-OCR，这个图其实跟前面提到的 VLMs 整体架构是一样的，中间一大块是 image encoder，后面 Decoder 是 LLM，所以和前面的 VLMs 架构基本是一样的。

整体架构改进在于 Image Encoder，使用了基于 local attention 的 SAM 的 image encoder，后接 convolution 进一步的downsample 出来的结果，因为 local attention 可以节省计算量，但它只有 local 的信息，所以后又接了 clip 去获得 global attention，这里的 activation 都非常的小，最后出来的 image embedding，再送到 deepseek-3b（LLM Decoder） 里面，加上 text 的prompt，最后得到输出。

整个图乍一看去是一个 OCR 的问题，也就是说输入是图像 image，然后会用 ViT 传统的方法，把 image 打成 patch，经过 encoder 之后，输入到 LLM 当中，当然还会有一些 prompt 告诉 LLM 该做一些什么任务，所以这是一个输入 image，输出文字的工作。

但这篇 paper 真正想要讨论的并不是 OCR 的问题，如果在 image 前加一个 text 比较好，也就是说这个 image 是对文本内容的影像话处理，然后对 image 进行处理，那这样做有什么好处呢？在 paper 里面举了一个例子，如果输入的大小是 1024x1024，打成 16x16 的 patch，最后能得到 4096 个 patch，即 $n=4096$，再经过 tokenizer，这个 tokenizer 包括一个 16x 的下采样，所以最后得到的 vision tokens 只有 512 个，然后再经过 image embedding 进入到 LLM 中。

所以这里的输入 token 为 512，相比原始的 text 应该要小很多。所以这篇文章所提出的方法可以极大的对 text 信息进行压缩，最后这两种方法进行对比会发现他们的性能是差不多的，这就意味着以后对于这些文本信息 text，很可能就不需要再按传统的方法进行处理了。

### 网络架构

这个网络架构基于一作之前的一篇 paper，Vary，其架构如下所示：
<img src="https://arxiv.org/html/2312.06109v1/x1.png" width="600">
其核心的想法是用两个 Image Encoder，最后一行红色❄️ 是 SAM Image Encoder，蓝色条是 CLIP Image Encoder，Vary 是用了两个 Image Encoder 去抽 Image Feature，最后把它们融合起来，这里稍微有一点点的复杂，在 Vary 中它不是简单的把这两个 Image Encoder 合起来，而是先对 SAM Image Encoder 进行训练，也就是上面一行红色🔥，训练好了之后，再把它冻住，然后和 CLIP Encoder 合在一起。
<img src="https://arxiv.org/html/2312.06109v1/x2.png">
在这个图就非常的清楚了，这里所谓的 New Vocabulary Network，其实就是前面提到的SAM Image Encoder，Original CLIP Network 就是 CLIP Image Encoder，上面这个表示就是先对 SAM Image Encoder 进行训练，训练好了之后把它拿下来两个都冻住，然后提 Feature，最后 Concat 到一起放到 LLM 当中。
<img src="https://arxiv.org/html/2312.06109v1/x3.png" width="400">

**SAM Encoder 具体的架构是什么** 给定一个 Input Image，首先经过 SAM Image Encoder，输出的维度是有一些变化的，为了保证输出的维度能够跟 CLIP 一样，因为最后要做Concat，所以又加了一个两层的 Convolution 去改变输出的维度，

**（缺陷与改进）** 但其实这种用两个或者多个 Image Encoder 去提 Feature 的方法，是有一些缺陷的，就是它产生的 Image Tokens 是比较多的，所以作者在这里就把它整个架构给变了一下，把 Vary 中两个 Image Encoder 的并行架构，改成了 Sequential 的架构，所谓的 Sequential 架构就是先用 SAM Image Encoder，包括 Convolution 的 Layer，先对图像做一次处理，再送到 CLIP Image Encoder 里面，最后再输出到 LLM。他们发现这样改了之后，既可以用到这两个 Image Encoder 提 Feature 的能力，又可以极大的控制整个过程当中 Vision Tokens 的数目以及产生的 Activation 的大小。（问题：CLIP 基于 SAM 处理后的结果再 Encoder，SAM 特征的特征，仅仅为了减少 tokens？）


**（回到 DeepSeek-OCR 架构）** 所以再回到 DeepSeek-OCR 架构就非常的清楚了，前半截是 SAM Image Encoder，包括 Convolution 的部分，后面是 CLIP Image Encoder，串行的结构。作用也是非常的明显 就是让 Vision Tokens 变得很少。

这里 SAM 用的并不是最原始的 ViT，而是何凯明所提出的 VITDET 的一个改进版，也就是把 Full Attention 改成了 Local Attention，有点类似于前面提到的 Swin，但有一点点的区别就是，Swin 里面的 Attention，它在切 Window 的时候，是用的 Shift Window，也就是 Window 之间是 overlap 的，VITDET 用的 Local Attention，它切 Window的时候是没有overlap，就是这一点点小小的区别。可以直观的理解为，就是在做 Attention 的时候，是类似于 Swin 那样要切 Window，在 Window 内部做 Local Attention，然后 Clip 做的是 Global Attention，这样具有局部信息的计算，也有全局信息的计算。

**（图中名称的叫法问题）** 为什么作者在这里把 CLIP 这一部分叫做 Embedding Layer，在 SAM 之后、CLIP 之前这一部分叫做 Vision Tokens。 一作沿用了 Vary 中的叫法。传统的VLM 把 Image 经过 Image Encoder 或者叫 Tokenizer 之后会产生 Image Tokens，后面一般会在接 Adapter，即作者称为 Embedding Layer。常用的有 Linear Attention 或者是Qformer。所以 DeepSeek-OCR 的作者也就是把这里的 Image Tokenizer 变成了 SAM，然后把这里的 Adapter 或者是 Embedding Layer 变成了 CLIP，也就是让 CLIP 来做 Adapter 或者是 Embedding Layer。所以这篇 paper 里面所讨论的 Vision Tokens，其实都是 SAM 这部分的输出，CLIP 在这里起到的，只是一个 Embedding Layer，类似于传统 VLMs 里面的 Adapter 的作用。


**模型如何处理 Multiple Resolution？**  

<img src="https://arxiv.org/html/2510.18234v1/x3.png">

在训练的时候，采用不同的 Mode，所谓不同的 Mode 就是根据图像的大小，使用不同 Resolution 的 Mode 来进行训练，比如有 TinySmall Mode 对应的 Image 大概就是这样的大小，产生的 Vision Tokens 是 64 和 100，类似的还有 Base 和 Large Mode，另外还有一个Gundam Mode 和 Gundam Master Mode，针对 High Resolution Images，然后也使用了前面所提到的 Tiling 方法，把图片切成不同的 Tile。

