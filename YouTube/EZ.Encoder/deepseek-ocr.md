
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




相关的背景就给大家介绍完了 我们直接来看一下DeepSeq OCR这篇paper 这篇paper主要想法就是说 对于这些长的context 我们是否能通过这种Optical 2D mapping 也就是说把长的context 转成一个2D的图像 让模型来处理 DeepSeq OCR主要有两个部分 一个就是DeepEncoder和一个Decoder 回忆一下前面提到的VLM模型 基本也都是这样的一个架构 这里的Decoder就是一个LM 然后DeepEncoder就是一个ImageEncoder 只不过DeepSeq在设计DeepEncoder的时候 主要是想设计一个简单使用的ImageEncoder 使得它产生的Vision tokens尽可能的少 在少的同时 能够保证它decode出来的precision是足够的高 作者想通过这种方式对LongContext进行压缩 他们也做了一些实验 发现这种方式压缩比可以做到非常的高 同时模型在OCR的性能上面 已经能够和现在比较主流的一些模型相媲美 DeepSeq还把他们的代码和模型都开源了 我们先来看一下Figure1 Figure1是在Fox Benchmark上面的一个结果 这个也是一座的一个工作 这里有两个纵轴 一个是Precision 一个是Compression 横坐标是Context 它的token的实际数目 从左到右这个token是越来越多 分别是把下面对应的这些token 压缩到64个Vision tokens 或者是100个Vision tokens 这个虚线表示的就是压缩比 随着原始的文本的token越来越多 如果保证Vision tokens是不变的话 压缩比是越来越高的 这个比较容易理解 同时我们来看一下这些柱状图 也就是对应的Precision 可以看到这些Precision是相当的高的 哪怕是在10倍左右的压缩比的情况下 右边这个图是展示的 在Omnivore Dock Bench上面的实际的结果 横坐标是Vision token使用的数目 从左到右一次降低 纵坐标是Performance 这个值是越低越好 也就是越往上越好 所以在这个图上面 在右上角的这些模型是比较好的 也就是他用了比较少的Vision tokens 并且他的performance比较好 我们可以看到DeepSeq OCR的这一系列模型 基本都在这个右上角 但是单纯从performance来看的话 比如说像小红书最近出的DOS.OCR 这个性能还是相当不错的 但是这些模型产生的Vision tokens 就会非常的多 也就当然DeepSeq OCR 如果用比较多的Vision tokens的话 它的性能也可以接近于 或者是超过这些主流的模型 比如说DOS.OCR 或者是Qianwen以及Internet系列 我们再来看一下这个图 这个图主要是作者总结了一下 当前主流的VRM它的架构 那这些paper在前面 我都已经给大家简单的介绍了一下 作者在这里把VRM分了几个类别 简单来说就是如何去处理不同 分辨率图像的方法的不同 比如说DeepSeq, VL, Vary 这类的方法就是用多个encoder 这些encoder处理的图像的分辨率是不一样的 比如说有的是处理high resolution的 有的是处理low resolution的 那这里只用了两个encoder 前面我也提到像Kaibarian 1 甚至用到了四个encoder 第二类方法就是DeepSeq VL2 或者是Internet VL系列 用的这种tiling的方法 也就是把图像根据预先设定好的长宽比 给切成这种tile 当然这里会对整个图做一个小的缩略图 通过这个缩略图来获得全局的信息 第三类就是刚才提到的Google的那篇工作 包括千万系列也都是使用这种方法 直接把这个图片切成很多的patch 包里的放到VIT当中一起处理 那这些方法都有很多的问题 最主要的问题就是 很多方法它产生的Visual Token太多了 对于这种high resolution的图像信息处理的并不好 尤其是如果输入它的resolution是可变的 另外就是会产生large activations 这种会占用大量的显存以及速度的问题 所以作者就提出了他们的一个架构 DeepOCR 其实这个图看着有一点复杂 其实跟我们前面提到的VLM整体架构是一样的 这里其实就是一个image encoder 这里就是一个LM 然后这就是text 如果把这里简化一下 其实这个模型就可以化成LM 然后text 这里是image encoder 然后这里是image 所以和前面的VLM架构基本是一样的 这里的创新就在于image encoder作者做了一些改进 第一个改进就是使用了一个基于local attention的 SAM的image encoder 然后再接一个convolution 进一步的downsample出来的结果 因为这种local attention 它可以节省计算量 但是它同时又只有local的信息 所以这里又接了一个clip 去获得global的attention 然后这里的activation都非常的小 最后出来的image embedding 然后再送到deepseq3b 也就是一个LM的decoder里面 加上text的prompt 最后得到输出 我们再花点时间仔细的讨论一下这个图 我觉得这个图信息量还是很丰富的 首先第一个我想给大家讨论的点就是 这个图乍一看去是一个OCR的问题 也就是说我的输入是一个图像image 然后会用VIT传统的方法 把这个image打成patch 经过一个encoder之后 输入到大元模型当中 最后输出 当然这里会有一些prompt 告诉这个大元模型该做一些什么任务 所以这是一个输入image输出文字的工作 但其实deepseq OCR这篇paper 真正想要讨论的并不是OCR这个问题 所以这个图前面 如果再加上一个text是比较好的 这里意思就是说 如果我有这张图当中输出到这个大猿模型当中 当然会有一些problem 然后进行处理 当然你可以看到这个当中的token数目就特别的多 那DeepSeek OCR提出了一个方法 就是我先把这个text转成一个图片 也就是这里所展示的 然后对图片进行处理 那这样做有什么好处呢 在paper里面举了一个例子 如果这个输入的大小是1024x1024的话 那打成16x16的patch 最后就能得到4096个patch 也就是n等于4096 然后再经过这个tokenizer 这个tokenizer包括一个16x的dump sampling 所以最后得到的这个region tokens只有512 然后再经过这个mini-betting进入到这个大猿模型当中 所以如果你看这里的输入token512和原始的 如果你直接对text进行处理的话 这里面举个例子 可能远大于512的text token 用这篇文章所提出的方法 可以极大的对这个text信息进行压缩 也就是把这些原始的text都压缩成了512个region tokens 最后这两种方法进行对比的话 会发现他们的性能是差不多的 这就意味着以后对于这些文本信息text 很可能我不需要再按传统的方法 也就是有一个text的tokenizer 处理完之后进入到大猿模型 不需要用这种方法了 直接把text转成image 然后对image进行处理 这样做的一个好处 自然就是这里的region tokens 也就是输入tokens变得更少了 对于大猿模型来说 本身它的处理成本的能力是有限的 这样我处理这些短的序列来说的话会更容易一些 这也是在陌生能力里面常用的一个方法 就是我对输入做一个降维 比如说SVD 然后输入降维之后的数据 这也是一个类似的思想 希望我在这里把DeepSeq OCR这篇paper最核心的思想 给大家解释清楚了 接下来我们再具体看一下 这个网络架构它是怎么来的 其实这个网络架构基于一座之前的一篇paper VRY 所以我们来看一下VRY这篇paper 就能理解这个架构是怎么来的 这里展示的就是VRY这篇paper 这篇paper核心的一个想法就是用两个Image Encoder 我们就看下面这个图好了 这个红色的其实是一个SUM的Image Encoder 蓝色的其实是一个CLIP的Image Encoder VRY就是用了两个Image Encoder去抽Image Feature 最后把它们融合起来 这里稍微有一点点的复杂 就是VRY这篇paper 它不是简单的把这两个Image Encoder合起来 而是先对SUM这个Image Encoder进行训练 也就是上面的这一部分 训练好了之后 再把它冻住和CLIP Encoder合在一起 这个图就非常的清楚了 它这里所谓的New Vocabulary Network 其实就是前面提到的SUM Image Encoder 这个Original CLIP L Network 就是CLIP的Image Encoder 上面这个表示就是先对SUM Image Encoder进行训练 训练好了之后 把它拿下来两个都冻住 然后提Feature 最后Concatenate到一起放到大园模型当中 我们具体看一下SUM Encoder 它具体的架构是什么 给定一个Input Image 它首先经过一个SUM Image Encoder 这个输出的维度是有一些变化的 为了保证输出的维度能够跟CLIP一样 因为最后要做Concatenation 所以它又加了一个两层的Convolution 去改变这个输出的维度 这样使得SUM Image Encoder的输出 能够跟CLIP Image Encoder的输出 能够Concatenate起来 这就是Varied GPM Paper网络的架构 我们再回到这个图来看一下 其实前面也提到了 这种用两个或者多个Image Encoder 去提Feature的方法 其实是有一些缺陷的 就是它产生的Image Tokens 其实是比较多的 所以作者在这里就把它整个架构给变了一下 把这里的并行架构 改成了一个Sequential的架构 所谓的Sequential架构就是 先用SUM Image Encoder 包括Convolution的Layer 先对图像做一次处理 处理完了之后 这个输出再送到CLIP的Image Encoder里面 最后再输出到大圆模型里面 也就是把这里的并行结构 变成了这样的一个串行结构 我猜DeepSeq OCR是想减少Vision Tokens 他们发现这样改了之后 既可以用到这两个Image Encoder 它提Feature的能力 又可以极大的控制整个过程当中 Vision Tokens的数目 以及产生的Activation的大小 所以你再回到DeepSeq OCR这篇Paper 你再看到网络架构的话就非常的清楚了 这就是前面Vary那篇Paper的 其中的一个SUM Image Encoder 当然包括Convolution的部分了 这就是另外的一个Image Encoder 在Vary里面这两个部分是并行的 现在在DeepSeq OCR里面 做着把它变成了一个串行的结构 就是做了这么一个小小的更改 那作用也是非常的明显 就是让这个Vision Tokens变得很少 那当然这里还有一个很重要的点 就是SUM这里用的并不是最原始的VIT 而是用了何凯明所提出的 VIT DET的一个改进版 也就是把Full Attention改成了Local Attention 这个就有点类似于我前面提到的Swing 但有一点点的区别 就是Swing里面的Attention 它在切Window的时候 是用的Shift Window 也就是Window之间是overlap的 这个VIT DET用的Local Attention 它切Window的时候 Window之间是没有overlap 就是这一点点小小的区别 但你可以直观的理解为 就是在做Attention的时候 是类似于Swing那样要切Window 在Window内部做Local Attention 然后Clip做的是Global Attention 这样你具有局部信息的计算 也有全局信息的计算 所以有了这些背景之后 你再看整个网络架构 其实也非常的简单 并没有太多很难理解的部分 这些都是前任的工作 不过在这篇paper里面做了一点点的改进 这也是我常说的科学研究 没有石头里蹦猴子的 大部分都是要么自己 要么其他人的潜需的工作 做了一些微小的改动 当然DeepSeek OCR这个想法的提出 还是非常有意义的 关于这个图在最后提一个 我比较困惑的点 就是为什么作者在这里 把Clip这一部分叫做Embedding Layer 然后Vision Tokens是在Sum这一部分 之后Clip之前 这一部分叫做Vision Tokens 直到我看了Vari这篇paper之后 我才明白 因为Vari这篇paper的一座 和DeepSeek OCR一座是同一个人 所以他就沿用了这里的称呼 传统的VRM把Image经过一个 Image Encoder或者叫Tokenizer之后 会产生Image Tokens 一般会在这里接一个Adapter 这里作者叫Embedding Layer 其实是一个意思 常用的有Linear Attention 或者是Qformer 所以DeepSeek OCR的作者 也就是把这里的Image Tokenizer 变成了Sum 然后把这里的Adapter 或者是Embedding Layer变成了Clip 也就是让Clip来做Adapter 或者是Embedding Layer 这样的话就比较容易理解 这篇paper里面所讨论的Vision Tokens 其实都是Sum这部分的输出 Clip在这里起到的 只是一个Embedding Layer 类似于传统VRM里面的Adapter的作用 当然这只是我的个人理解 你有任何不一样的想法 欢迎你留言和我讨论 模型的基本架构 这里就讨论完了 那接下来就是一个比较重要的问题 就是这个模型如何处理Multiple Resolution 这其实是前面VRM很多工作 想要解决的一个问题 那这里作者的解决方案也比较的直接 就是在训练的时候 采用不同的Mode 所谓不同的Mode就是 根据图像的大小 使用不同Resolution的Mode来进行训练 比如说就有Tiny Small Mode 对应的Image大概就是这样的大小 然后产生的Vision Tokens就是64和100 那类似的还有Base和Large Mode 另外还有一个高达Mode和高达Master Mode 那这种就是针对High Resolution Images 然后这里也使用了前面所提到的Tiling的方法 把这个图片给切成不同的Tile 那这个表就总结了 DeepSeek OCR里面所使用的几种不同的Mode 那对应的Resolution从512到1280 那这里的高达Mode使用Dynamic Resolution 也就是说它支持两种不一样的Resolution 比如说640和1024 那当然因为输入图像的大小是不一样的 所以它对应的Vision Tokens也是不一样的 关于Decoder Pod也没有什么特殊的 主要就是一个MOE模型 那这个MOE模型的好处就是 虽然它是一个3B的模型 但是它的激活参数只有570个Million 所以在Inference的时候 相当于使用的是一个 大概500Million的一个小模型 所以它会非常的高效 那另外这篇Paper也有一部分的章节 在讨论如何去构造这个OCR的数据 那我觉得这一部分并不是这篇Paper最核心的部分 因为这篇Paper最核心的还是在讨论 Text Compression的思想 OCR只是一个Demo 所以关于数据构建这一部分 感兴趣的同学可以自己去看一下 这里我就不再给大家解释了 然后这里提到了他们的Training Pipeline 也基本上跟前面我提到的 VRM模型训练常用的方法类似 那这里他们的做法是 先Train这个Deep Encoder Train好了之后 再合起来Train整个DeepSeq OCR模型 那这个表格展示的就是 Vision Text Compression Ratio 那这里主要用的是两个Mode 512x512和640x640 那对应的Vision Tokens分别是64和100 这里可以看到如果Text Tokens数是600到700之间的话 把它转成图像 然后再经过Deep Encoder之后 如果是使用512x512这个Resolution的话 那最后产生的就是64个Token 这个时候压缩比大概是在10.5倍 也就是从600多个Text Tokens 被压缩成了64个Vision Tokens 可以看到这个时候的Precision还是非常高的 有96.5 那如果把它压缩成100个Vision Tokens的话 这个Precision能到98.5% 这样的Text Tokens数据相当于7页的文档 那我觉得这是一个非常有意思的结果 我觉得这个想法会有很多的应用场景 在AI Agent领域里面 大家经常要做这种Memory的管理 比如说在多轮对话当中 那比较简单的方法就是你可以用大圆模型 对过去的对话记录进行总结 那这也有一些缺点 就是你总结的话必然会丢失掉一部分的信息 当对话长了之后 可能更早之前的信息就丢失了 那如果用DeepSeek OCR这个技术的话 可以把过往的对话全部都转成图像存储起来 甚至可以用Vision Tokens给存储起来 这相当于对历史的对话 历史的记录做了一个压缩 可以想象你对一个7页纸的历史对话 压缩到100个Vision Tokens 并且几乎没有太大的损失 我觉得这是非常有用的 当然这里可以有其他的应用 大家可以自己去头脑风暴一下 我们再来看一下这个表格 这个是在OCR Benchmark上面的结果 然后作者把这个模型分成了三类 一类是传统的OCR模型 用Pipeline来做的 这里所谓的Pipeline就是传统的OCR 其实是有很多的步骤 比如说第一步就是做Layout Detection 所谓的Layout Detection就是 比如说有些PDF 它是Double Column 这个时候你就先要检测出哪些地方有文字 哪些地方是第一个Column 哪个地方是第二个Column 哪些地方是图表 哪些地方是页眉页角 把具体的Layout检测出来之后 然后再对每一个部分进行文字的识别 这里每一步都很容易出错 如果上一步出错了 都会影响到下一步的进度 第二类模型就是End-to-End模型 比如说现在主流的VLM模型 第三个就是作者提出的DeepSeq OCR 这里的值是越小越好 我们就直接来看加粗的最好的这些结果 大概在哪里 可以看到对于End-to-End的Model里面 最好的是小红书最近出的DOS.OCR 这个模型跟作者提出的高达Master模型 基本上是不相上下的 但是DOS.OCR模型所产生的Token数 是要远大于DeepSeq OCR这个模型的 我想说DeepSeq OCR还只是一个新的方法 我觉得这里面还有很多的改进空间 它能跟现在最好的VLM模型打个平手 我觉得已经相当不容易了 这里面肯定还有巨大的改进空间 作者这里接着把DeepSeq OCR模型的性能 按照文档的类别给展开了一下 比如说有Book, Slice等等 这些不同的文档类型 我们来看一下Tiny和Small这两个Mode 在有一些的文档类型上面 这个表现并不是特别的好 比如说我们来看Small的话 尤其是在这个Newspaper上面 这个表现是非常的差 这个值越大越差 很可能原因是因为Newspaper里面的文字特别的多 所以用LowResolution Mode 这个信息被压缩的太厉害了 会影响最后的精度 所以在这里如果使用高达Master Mode的话 像这个Newspaper这一类别的文档 精度就会提高很多 这个也比较容易理解 就跟做SVD一样 如果你本身的信息很多 那么最后选取Principal Components的时候就不能选的太少 我觉得这里道理也是一样的 不能压缩的太厉害 接下来作者就展示了一些定性的结果 这里也没有很多很特别的地方 我稍微想展开一下就是这个图 原始的这个图里面是一个柱状图 但是DeepSeq OCR可以通过这个柱状图 直接把柱状图所对应的这个表格给提示出来 也就是他们所提到的Deep Parsing 我觉得这个功能还是非常有用的 因为有了这些具体的数值之后 你就可以做下游的分析 很多时候在文档里面 这种柱状图是没有对应的数值的 所以做下游的分析是比较困难的 这种场景还是非常多的 另外就是直接产生markdown 识别这一些文字的layout 或者是对这些figure进行描述 还有就是比如说提取一些分子式 还有一些几何图形 然后就是对不同语言文字的处理 还有就是做一些类似于object detection的工作 提取图像中的一些object等等这些 总的来说 DeepSeq OCR这方面的功能还是非常强大的 但我这里也想再强调一下 这篇paper创新的点远不在于OCR这个地方 作者在讲完他们OCR的功能之后 又回到了context compression这里 他们这里提出了一个想法 就是类似于forgetting mechanisms 主要是处理那些特别特别长的context 对于像多轮对话当中 那些非常古老的原始的对话 再把它转成图片之后 就可以用特别低的resolution来处理 作者在这里打了个比方 这就类似于人类的memory 对那些特别久远的记忆 我们只需要保留一个模糊的记录 对于刚刚发生的那些事情 可能就要记住很多的细节 那类似的对于最近的一些context 我们可以转成比较高resolution的图片 然后随着比如说对话的进行 我们可以逐渐的把这个历史的信息 或者是历史的图片给downsize越来越小 这就类似于人类的记忆在变模糊 所以这就是一个利用他们这种context compression方法的具体的应用 总的说来 我觉得这个思路还是挺有意思的 那这篇paper大致就给大家录完了 那稍微的总结一下 这篇paper主要提出的就是 context optical compression这样的一个想法 具体的做法就是把text转成image 然后用image encoder去encode这些image patch 从而实现对最原始的text进行压缩 最后作者用OCR这个实际的场景 去展示了一下他们这个想法的性能 总的来说 我觉得这个想法还是很有意思 然后非常有潜力 我觉得这篇paper写的还是非常好的 我记得我之前的老板经常在我写paper的时候 就会说给paper升华一下 或者是拔高一下 我经常不知道该怎么去升华 那我觉得这一篇paper就是一个非常好的榜样 因为这篇工作你乍一看去 它是一个OCR的工作 不论是从模型的架构 还是最终的结果来看 其实都没有太多新颖的地方 性能也并不是远超过现在主流的VRM VCR模型 所以如果是写成一篇OCRpaper的话 那这篇paper可能就没有太多人关注了 但是DeepSeek把这篇paper给升华了一下 提出了一个context optical compression的概念 在这篇paper 其实也没有花太多的笔墨 全方位的去展示他们这个想法 只是很简单的用了一个compression ratio 一个表格去表明 我通过读这篇paper 我是学习到了很多升华主题的方法 当然我觉得能升华肯定跟作者本身的实力是有关的 这个一座长期在OCR这个领域 然后我估计context optical compression这个概念 要么是一座自己多年思考的结果 或者是DeepSeek某位大佬想到的 总之没有一定的实力 也做不出这篇paper 也不会去升华这篇paper 我觉得后续还有很多的工作值得探究 如何进一步的去压缩text的信息 这里面其实有很多值得去探究的 我今天的分享就到这里了 如果你有任何的想法 欢迎你留言 我们一起讨论

